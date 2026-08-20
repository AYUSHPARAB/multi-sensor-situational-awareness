"""
Stage 1 visual check — multi-image version.
Runs the detector on several BONK ship_detection images and saves
annotated PNGs showing YOLO detections (lime) vs ground truth (red).

Paper context:
    Both papers evaluate their detector across the full dataset, not
    single images. Paper 2 (BONK-pose 2025) benchmarks on 1000 images.
    We run on a sample here for visual inspection; full mAP evaluation
    across all 1000 images comes in vision/evaluate.py.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path

from data.bonk_loader import load_ship_detection_coco
from vision.detector import VesselDetector

OUTPUT = Path("output")
OUTPUT.mkdir(exist_ok=True)

# Number of images to check visually
NUM_IMAGES = 5

CATEGORY_NAMES = {
    0: "ship",
    1: "leaving_frame",
    2: "moored",
    3: "partial",
    4: "subvessel",
    5: "boat",
}

coco = load_ship_detection_coco()
detector = VesselDetector()

image_ids = list(coco.keys())[:NUM_IMAGES]

for img_id in image_ids:
    entry = coco[img_id]
    image_path = entry["image_path"]
    gt_boxes = entry["boxes"]
    cat_ids = entry["category_ids"]

    detections = detector.detect(image_path)

    print(f"\n{'='*50}")
    print(f"Image    : {entry['file_name']}")
    print(f"GT boxes : {len(gt_boxes)}")
    print(f"Detected : {len(detections)}")

    # Print each ground-truth box with its category name
    for i, ((x, y, w, h), cat) in enumerate(zip(gt_boxes, cat_ids)):
        cat_name = CATEGORY_NAMES.get(cat, f"unknown_{cat}")
        print(f"  GT  #{i}: {cat_name:15s}  box=({x:.0f}, {y:.0f}, {w:.0f}, {h:.0f})")

    # Print each YOLO detection with its confidence
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det.xyxy
        print(f"  DET #{i}: conf={det.conf:.3f}  box=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")

    # Draw the annotated image
    image = Image.open(image_path)
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(image)

    # Ground truth in red (COCO format: x, y, w, h)
    for (x, y, w, h), cat in zip(gt_boxes, cat_ids):
        cat_name = CATEGORY_NAMES.get(cat, "?")
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2, edgecolor="red",
            facecolor="none", linestyle="--"
        )
        ax.add_patch(rect)
        ax.text(x, y - 8, cat_name, color="red", fontsize=7,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="red", alpha=0.7))

    # YOLO detections in lime (xyxy format: xmin, ymin, xmax, ymax)
    for det in detections:
        x1, y1, x2, y2 = det.xyxy
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="lime",
            facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(x1, y1 - 8, f"YOLO {det.conf:.2f}",
                color="lime", fontsize=7, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="black",
                          ec="lime", alpha=0.7))

    ax.set_title(
        f"{entry['file_name']}  |  GT: {len(gt_boxes)}  |  "
        f"Detected: {len(detections)}",
        fontsize=10
    )
    ax.axis("off")

    # Save each image with a unique filename
    safe_name = Path(entry["file_name"]).stem
    out_path = OUTPUT / f"stage1_{safe_name}.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved  : {out_path}")

print(f"\n{'='*50}")
print(f"Done. {len(image_ids)} images processed.")

