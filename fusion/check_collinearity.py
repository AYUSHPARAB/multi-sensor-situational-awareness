"""
Check whether our AIS world points (lon, lat) are close to collinear.
If so, that explains the ill-conditioned / degenerate homography fits
we're seeing - a geometric property of the scene (vessels confined to
a narrow river channel), not a data-quality bug.
"""
import numpy as np
from fusion.homography import extract_correspondences

world_pts, pixel_pts, _ = extract_correspondences()

# Center the points, then look at the spread (variance) along the
# two principal directions. If nearly all the spread is along ONE
# direction, the points are close to collinear.
centered = world_pts - world_pts.mean(axis=0)
cov = np.cov(centered.T)
eigenvalues = np.linalg.eigvalsh(cov)
eigenvalues = np.sort(eigenvalues)[::-1]  # largest first

print(f"World point spread (eigenvalues of covariance): {eigenvalues}")
print(f"Ratio of spread (major/minor axis): {eigenvalues[0] / eigenvalues[1]:.1f}")
print()
if eigenvalues[0] / eigenvalues[1] > 20:
    print("STRONGLY collinear - points lie close to a single line.")
    print("This explains the ill-conditioned homography fits.")
elif eigenvalues[0] / eigenvalues[1] > 5:
    print("MODERATELY collinear - limited 2D spread.")
else:
    print("Points have reasonable 2D spread.")
