"""
Check whether the camera intrinsic matrix is truly constant across
our sampled images. If it varies, a single global homography fit
across all images is invalid - each distinct camera geometry needs
its own homography (same reason Paper 1's panning cameras score
far worse than fixed cameras).
"""
import numpy as np
from data.bonk_loader import load_sixd_bundle, list_sixd_names

names = list_sixd_names()
matrices = []

for name in names:
    bundle = load_sixd_bundle(name)
    matrices.append(bundle.camera_matrix)

matrices = np.array(matrices)  # shape (N, 3, 3)

print(f"Checked {len(names)} calibration files.\n")

mean_matrix = matrices.mean(axis=0)
std_matrix = matrices.std(axis=0)

print("Mean camera matrix:")
print(mean_matrix)
print("\nStd deviation across all samples:")
print(std_matrix)

# Count how many distinct matrices exist (rounded to nearest int)
rounded = [tuple(np.round(m.flatten(), 0)) for m in matrices]
unique = set(rounded)
print(f"\nNumber of DISTINCT calibration matrices (rounded): {len(unique)}")

if len(unique) > 1:
    print("\nCAMERA GEOMETRY VARIES ACROSS IMAGES.")
    print("A single global homography is not valid - points must be")
    print("grouped by matching calibration before fitting.")
else:
    print("\nCamera geometry is constant. The homography error is")
    print("likely due to AIS/detection mismatches, not camera variation.")

# Show the distribution of fx (focal length) as a quick sanity check
fx_values = matrices[:, 0, 0]
print(f"\nfx range: {fx_values.min():.1f} to {fx_values.max():.1f}")
print(f"fx unique count: {len(set(np.round(fx_values, 0)))}")
