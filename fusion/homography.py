"""
Homography Estimation
=====================
Fits a homography matrix that maps world coordinates (AIS lat/long,
converted to local metres) to image pixel coordinates, using
correspondence pairs from BONK-pose.

Paper context:
    Paper 1 (WACV 2024, Gulsoylu et al.):
        - Used homography as the core world-to-image transform.
        - Hand-picked 10+ static landmarks (piers, buildings) visible
          in both the camera image and Google Maps.
        - Deliberately avoided using vessel positions as keypoints,
          because AIS positions are noisy/laggy - used fixed structures
          spread across the whole image plane instead.
        - Applied RANSAC for outlier rejection during fitting.

    Paper 2 (BONK-pose 2025, Holst et al.):
        - Replaced homography with PnP (Perspective-n-Point).
        - PnP builds a full 3D camera model (intrinsic + extrinsic).
        - Reduced mean projection error from ~60px to ~12px.
        - Reports ~70.56% image-AIS association accuracy in their
          own matching pipeline - meaning a meaningful fraction of
          identifier links in the released data can be mismatches.

    Our approach and findings:
        - Same homography math as Paper 1.
        - Correspondence points sourced from BONK-pose labels:
          each vessel's AIS lat/long paired with its pixel bounding-box
          centre, linked by the shared 'identifier' field.
        - World points converted from degrees to local metres before
          fitting, to avoid numerical conditioning problems.
        - Diagnosed three compounding issues that make vessel-derived
          correspondences harder to fit than Paper 1's static landmarks:
            1. Coordinate scale mismatch (fixed by metre conversion).
            2. Moderate collinearity - vessels are confined to a narrow
               river channel, unlike spatially spread static landmarks.
            3. AIS/detection mismatches consistent with Paper 2's own
               ~70% association accuracy.
        - Uses RANSAC (via cv2.findHomography) for robustness.
"""

import logging
from pathlib import Path

import cv2
import numpy as np

from data.bonk_loader import load_sixd_bundle, list_sixd_names
from fusion.coord_utils import lonlat_to_local_meters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def extract_correspondences(names: list[str] | None = None):
    """
    Extract (world_meters, pixel) correspondence pairs from the 6D dataset.

    World points are converted from (lon, lat) degrees to local flat
    (x, y) metres, centred on the mean position of all vessels seen.
    This avoids numerical conditioning problems from mixing tiny
    degree values with large pixel values in the same fit.

    Returns:
        world_points: np.ndarray of shape (N, 2) - (x_m, y_m) pairs,
                       local metres relative to the mean AIS position
        pixel_points: np.ndarray of shape (N, 2) - (px, py) pairs
        count_by_image: dict mapping image name to number of pairs
        ref_point: (ref_lon, ref_lat) - the reference point used for
                   the metre conversion, needed to project new points
                   consistently later (Stage 3)
    """
    if names is None:
        names = list_sixd_names()

    raw_lonlat = []
    pixel_points = []
    count_by_image = {}

    for name in names:
        bundle = load_sixd_bundle(name)

        # Build a lookup from identifier -> AIS vessel data.
        # Both 'objects' and 'vessels' share the same 'identifier'
        # string for the same physical ship.
        vessel_by_id = {}
        for v in bundle.vessels:
            vessel_by_id[v["identifier"]] = v

        pairs_this_image = 0

        for obj in bundle.objects:
            obj_id = obj["identifier"]

            # Skip objects with no matching AIS vessel - this is the
            # "dark vessel" case (detected by camera but no AIS),
            # which becomes our hazard flag in Stage 4.
            if obj_id not in vessel_by_id:
                continue

            vessel = vessel_by_id[obj_id]
            lon = vessel["long"]
            lat = vessel["lat"]

            # bbImage2d is [xmin, ymin, xmax, ymax] (corner format,
            # confirmed visually in our bbox format check).
            x1, y1, x2, y2 = obj["bbImage2d"]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            raw_lonlat.append([lon, lat])
            pixel_points.append([cx, cy])
            pairs_this_image += 1

        count_by_image[name] = pairs_this_image
        logger.info("Image %s: %d correspondence pair(s)", name,
                    pairs_this_image)

    raw_lonlat = np.array(raw_lonlat, dtype=np.float64)
    pixel_pts = np.array(pixel_points, dtype=np.float64)

    # Convert to local metres, centred on the mean position.
    # This fixes the scale mismatch between tiny degree values and
    # large pixel values, which otherwise hurts RANSAC's numerical
    # conditioning during the fit.
    ref_lon, ref_lat = raw_lonlat.mean(axis=0)
    x_m, y_m = lonlat_to_local_meters(
        raw_lonlat[:, 0], raw_lonlat[:, 1], ref_lon, ref_lat
    )
    world_pts = np.column_stack([x_m, y_m])

    logger.info("Total correspondence pairs: %d from %d image(s)",
                len(world_pts), len(names))
    logger.info("World points converted to local metres, ref=(%.6f, %.6f)",
                ref_lon, ref_lat)

    return world_pts, pixel_pts, count_by_image, (ref_lon, ref_lat)


def fit_homography(world_pts: np.ndarray, pixel_pts: np.ndarray,
                   ransac_threshold: float = 5.0):
    """
    Fit a homography matrix H that maps world (x_m, y_m) -> pixel (px, py).

    Uses RANSAC to reject outlier correspondences (noisy AIS, bad boxes).

    Parameters:
        world_pts:        (N, 2) array of local metre coordinates
        pixel_pts:        (N, 2) array of (px, py) pairs
        ransac_threshold: maximum reprojection error (pixels) for a point
                          to be considered an inlier.

    Returns:
        H:       3x3 homography matrix (np.ndarray)
        mask:    (N,) boolean array - True for inliers, False for outliers
        error:   mean reprojection error across inliers (pixels)
    """
    if len(world_pts) < 4:
        raise ValueError(
            f"Need at least 4 correspondence pairs to fit a homography, "
            f"got {len(world_pts)}. Download more 6D images from BONK-pose."
        )

    H, mask = cv2.findHomography(
        srcPoints=world_pts,
        dstPoints=pixel_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold
    )

    mask = mask.ravel().astype(bool)

    inlier_world = world_pts[mask]
    inlier_pixel = pixel_pts[mask]

    ones = np.ones((len(inlier_world), 1))
    world_h = np.hstack([inlier_world, ones])

    projected_h = (H @ world_h.T).T
    projected = projected_h[:, :2] / projected_h[:, 2:3]

    errors = np.linalg.norm(projected - inlier_pixel, axis=1)
    mean_error = float(np.mean(errors))

    n_inliers = int(mask.sum())
    n_outliers = int((~mask).sum())

    logger.info("Homography fit: %d inliers, %d outliers, "
                "mean reprojection error: %.2f px",
                n_inliers, n_outliers, mean_error)

    return H, mask, mean_error


def project_world_to_pixel(H: np.ndarray, lon: float, lat: float,
                           ref_point: tuple):
    """
    Project a single world point (lon, lat) to pixel (px, py) using H.

    Parameters:
        H:         3x3 homography matrix
        lon:       longitude (degrees)
        lat:       latitude (degrees)
        ref_point: (ref_lon, ref_lat) - the same reference point used
                   when extract_correspondences() built the metre
                   coordinates this H was fitted on

    Returns:
        (px, py): predicted pixel coordinates
    """
    ref_lon, ref_lat = ref_point
    x_m, y_m = lonlat_to_local_meters(lon, lat, ref_lon, ref_lat)

    pt = np.array([x_m, y_m, 1.0])
    result = H @ pt

    px = result[0] / result[2]
    py = result[1] / result[2]

    return float(px), float(py)
