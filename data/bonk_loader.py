"""
BONK-pose Dataset Loader
=========================
Loads and pairs the BONK-pose dataset files:
    - 6D pose set: calib/ (camera intrinsics) + image/ + label/ (AIS + fused pose)
    - ship_detection set: images/ + result.json (COCO format)

Files in the 6D set are matched across folders by a shared basename,
e.g. calib/ffc4f60725d0d92d.txt <-> image/....jpg <-> label/....json
"""

import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np


BONK_ROOT = Path("data/raw/bonk_pose")
SIXD_DIR = BONK_ROOT / "6d_pose_estimation"
SHIP_DET_DIR = BONK_ROOT / "ship_detection"

# Using the GitHub (compressed) copy since it's fully downloaded (1000/1000 images).
SHIP_DET_COCO_PATH = BONK_ROOT / "bonk_pose_coco_github" / "result.json"


@dataclass
class SixDBundle:
    """One matched (calib, image, label) triple from the 6D dataset."""
    name: str
    calib_path: Path
    image_path: Path
    camera_matrix: np.ndarray   # 3x3 intrinsic matrix K
    objects: list                # fused output: bbImage2d, position, size
    vessels: list                # raw-ish AIS: lat, long, heading, speed, size


def load_sixd_bundle(name: str) -> SixDBundle:
    """Load one matched (calib, image, label) triple by its shared basename."""
    calib_path = SIXD_DIR / "calib" / f"{name}.txt"
    image_path = SIXD_DIR / "image" / f"{name}.jpg"
    label_path = SIXD_DIR / "label" / f"{name}.json"

    camera_matrix = np.loadtxt(calib_path)

    with open(label_path) as f:
        label = json.load(f)

    return SixDBundle(
        name=name,
        calib_path=calib_path,
        image_path=image_path,
        camera_matrix=camera_matrix,
        objects=label["objects"],
        vessels=label["vessels"],
    )


def list_sixd_names() -> list[str]:
    """Return every basename currently present in the 6D image/ folder."""
    return sorted(p.stem for p in (SIXD_DIR / "image").glob("*.jpg"))


def load_ship_detection_coco(json_path: Path = SHIP_DET_COCO_PATH) -> dict:
    """
    Load ship_detection-style COCO annotations (works for either the
    GitHub compressed copy or the native UHH copy, same format).
    Returns a dict mapping image_id -> {file_name, image_path, boxes, category_ids}
    """
    with open(json_path) as f:
        coco = json.load(f)

    base_dir = json_path.parent

    result = {}
    for img in coco["images"]:
        result[img["id"]] = {
            "file_name": img["file_name"],
            "image_path": base_dir / img["file_name"],
            "boxes": [],
            "category_ids": [],
        }

    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        result[img_id]["boxes"].append(tuple(ann["bbox"]))
        result[img_id]["category_ids"].append(ann["category_id"])

    return result
