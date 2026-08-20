"""Quick visual check: run YOLO on one BONK image and draw the detections."""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from data.bonk_loader import load_ship_detection_coco
from vision.detector import VesselDetector

coco = load_ship_detection_coco()
first = coco[next(iter(coco))]
image_path = first["image_path"]

detector = VesselDetector()
detections = detector.detect(image_path)
print(f"{len(detections)} vessels detected in {image_path.name}")

image = Image.open(image_path)
fig, ax = plt.subplots(figsize=(12, 9))
ax.imshow(image)
for det in detections:
    x1, y1, x2, y2 = det.xyxy
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                             linewidth=2, edgecolor="lime", facecolor="none")
    ax.add_patch(rect)
    ax.text(x1, y1 - 5, f"{det.conf:.2f}", color="lime", fontsize=9)
ax.axis("off")
plt.savefig("output/stage1_detections.png", dpi=120, bbox_inches="tight")
print(f"Saved to output/stage1_detections.png")
