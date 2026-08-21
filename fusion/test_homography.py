"""
Test homography fitting on available 6D samples.
Extracts correspondences (in local metres), fits H, reports error,
and visualises projected AIS positions on one image.
"""
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path

import numpy as np

from data.bonk_loader import load_sixd_bundle, list_sixd_names
from fusion.homography import (
    extract_correspondences,
    fit_homography,
    project_world_to_pixel,
)

OUTPUT = Path("output")
OUTPUT.mkdir(exist_ok=True)

print("=" * 50)
print("Extracting correspondences from 6D dataset...")
print("=" * 50)
world_pts, pixel_pts, counts, ref_point = extract_correspondences()

print(f"\nTotal pairs: {len(world_pts)}")
print(f"Reference point (lon, lat): {ref_point}")

if len(world_pts) < 4:
    print(f"\nERROR: Only {len(world_pts)} pairs found.")
    exit(1)

print("\n" + "=" * 50)
print("Fitting homography with RANSAC (5px threshold)...")
print("=" * 50)
H, mask, mean_error = fit_homography(world_pts, pixel_pts)

print(f"\nHomography matrix H:")
print(H)
print(f"\nInliers: {mask.sum()} / {len(mask)}")
print(f"Mean reprojection error (inliers only): {mean_error:.2f} pixels")

print("\n" + "=" * 50)
print("Visualising projected AIS positions...")
print("=" * 50)
names = list_sixd_names()
bundle = load_sixd_bundle(names[0])

image = Image.open(bundle.image_path)
fig, ax = plt.subplots(figsize=(14, 10))
ax.imshow(image)

for obj in bundle.objects:
    x1, y1, x2, y2 = obj["bbImage2d"]
    rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=2, edgecolor="cyan", facecolor="none"
    )
    ax.add_patch(rect)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    ax.plot(cx, cy, "o", color="cyan", markersize=8)
    ax.text(x1, y1 - 10, "box centre",
            color="cyan", fontsize=7, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="black",
                      ec="cyan", alpha=0.7))

for vessel in bundle.vessels:
    px, py = project_world_to_pixel(H, vessel["long"], vessel["lat"], ref_point)
    ax.plot(px, py, "x", color="red", markersize=12, markeredgewidth=3)
    ax.text(px + 10, py, "AIS projected",
            color="red", fontsize=7, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="black",
                      ec="red", alpha=0.7))

ax.set_title(
    f"{bundle.name}  |  Mean reproj error: {mean_error:.1f}px  |  "
    f"Cyan=box centre, Red=AIS projected",
    fontsize=10
)
ax.axis("off")
out_path = OUTPUT / "stage2_homography_check.png"
plt.savefig(out_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nSaved -> {out_path}")

# ============================================================
# DIAGNOSTIC: per-point errors + comparison across strategies
# ============================================================
print("\n" + "=" * 50)
print("DIAGNOSTIC: per-point errors under the RANSAC(5px) H")
print("=" * 50)

ones_all = np.ones((len(world_pts), 1))
world_h_all = np.hstack([world_pts, ones_all])
projected_h_all = (H @ world_h_all.T).T
projected_all = projected_h_all[:, :2] / projected_h_all[:, 2:3]
errors_all = np.linalg.norm(projected_all - pixel_pts, axis=1)

n_show = min(len(errors_all), 50)
for i in range(n_show):
    tag = "INLIER" if mask[i] else "outlier"
    print(f"  Point {i:3d} [{tag:7s}]  error = {errors_all[i]:10.1f} px")
if len(errors_all) > n_show:
    print(f"  ... ({len(errors_all) - n_show} more points not shown)")

print(f"\n  Median error (all {len(errors_all)} points): {np.median(errors_all):.1f} px")
print(f"  Mean error (all {len(errors_all)} points):   {np.mean(errors_all):.1f} px")
print(f"  Max error (all {len(errors_all)} points):    {np.max(errors_all):.1f} px")

print("\n" + "=" * 50)
print("Comparison: RANSAC with 50px threshold")
print("=" * 50)
H_loose, mask_loose, err_loose = fit_homography(
    world_pts, pixel_pts, ransac_threshold=50.0
)
print(f"  Inliers: {mask_loose.sum()} / {len(mask_loose)}")
print(f"  Mean reprojection error (inliers only): {err_loose:.2f} px")

print("\n" + "=" * 50)
print("Comparison: plain least-squares (no RANSAC, all points used)")
print("=" * 50)
H_lstsq, mask_lstsq = cv2.findHomography(world_pts, pixel_pts, method=0)
ones2 = np.ones((len(world_pts), 1))
world_h2 = np.hstack([world_pts, ones2])
proj_h2 = (H_lstsq @ world_h2.T).T
proj2 = proj_h2[:, :2] / proj_h2[:, 2:3]
errors_lstsq = np.linalg.norm(proj2 - pixel_pts, axis=1)
print(f"  Mean error (all {len(errors_lstsq)} points, no rejection): {np.mean(errors_lstsq):.1f} px")
print(f"  Median error: {np.median(errors_lstsq):.1f} px")

print("\n" + "=" * 50)
print("Summary")
print("=" * 50)
print(f"  RANSAC 5px  : {mask.sum():3d}/{len(mask)} inliers, "
      f"mean error (inliers) = {mean_error:8.2f} px")
print(f"  RANSAC 50px : {mask_loose.sum():3d}/{len(mask_loose)} inliers, "
      f"mean error (inliers) = {err_loose:8.2f} px")
print(f"  Least-sq.   : {len(world_pts):3d}/{len(world_pts)} used,    "
      f"mean error (all)     = {np.mean(errors_lstsq):8.2f} px")
