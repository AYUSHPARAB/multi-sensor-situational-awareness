"""
Vessel Detector
===============
Runs YOLOv8 on harbour images and returns vessel detections.

Both BONK-pose papers use a YOLO-family detector as the pipeline entry point:
    - WACV 2024 (Gulsoylu et al.): YOLOv5-XL, fine-tuned to a single
      "vessel" class (mAP rose from 0.332 to 0.951 at IoU 0.5).
    - BONK-pose 2025 (Holst et al.): selected YOLOX-X after evaluating
      several detectors (0.80 mAP at IoU 0.5).
This module uses YOLOv8 off-the-shelf (COCO 'boat' class), no fine-tuning,
as a modern baseline. Fine-tuning on BONK labels is the known next step.
"""


import logging
from dataclasses import dataclass

from ultralytics import YOLO


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_NAME = "yolov8x.pt"           
COCO_BOAT_CLASS = 8                 #class 8 for boat
CONF_THRESHOLD = 0.25               #drop detetcions less confident than 0.25


@dataclass
class Detection:
    xyxy: tuple      # (xmin, ymin, xmax, ymax) in pixels
    conf: float      # confidence score
    cls: int


class VesselDetector:
    def __init__(self, model_name: str = MODEL_NAME, conf: float = CONF_THRESHOLD):
        self.conf = conf
        logger.info("Loading YOLO model: %s", model_name)
        self.model = YOLO(model_name)

    def detect(self, image_path) -> list[Detection]:
        results = self.model(image_path, conf=self.conf, verbose = False)[0]


        detections = []
        for box in results.boxes:
            cls = int(box.cls[0])
            if cls != COCO_BOAT_CLASS:
                continue
            xyxy = tuple(float(v)for v in box.xyxy[0])
            conf = float(box.conf[0])
            detections.append(Detection(xyxy=xyxy, conf=conf, cls=cls))


        return detections
