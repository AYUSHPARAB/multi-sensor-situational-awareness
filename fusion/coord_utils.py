"""
Coordinate conversion utilities.

Converts (lon, lat) in degrees to a local flat ENU (East-North-Up)
coordinate system in metres, centred on a reference point.

This fixes two real problems with using raw degrees for homography
fitting: (1) a large scale mismatch against pixel coordinates that
hurts numerical conditioning, and (2) it puts world points in units
where distances are physically meaningful (metres).
"""
import numpy as np

EARTH_RADIUS_M = 6371000.0


def lonlat_to_local_meters(lon, lat, ref_lon, ref_lat):
    """
    Convert (lon, lat) in degrees to local (x, y) in metres,
    relative to a reference point (ref_lon, ref_lat).

    Uses a simple equirectangular approximation - accurate enough
    for a small area like a single river harbour.
    """
    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)
    ref_lon_rad = np.radians(ref_lon)
    ref_lat_rad = np.radians(ref_lat)

    x = (lon_rad - ref_lon_rad) * np.cos(ref_lat_rad) * EARTH_RADIUS_M
    y = (lat_rad - ref_lat_rad) * EARTH_RADIUS_M

    return x, y
