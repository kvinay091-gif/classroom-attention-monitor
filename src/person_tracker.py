"""
Wraps Ultralytics YOLOv8 to detect + track students (person class) and
mobile phones (cell phone class) frame by frame.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from ultralytics import YOLO

from . import config


@dataclass
class Detection:
    track_id: Optional[int]
    bbox: np.ndarray   # [x1, y1, x2, y2]
    conf: float
    cls_id: int


class PersonPhoneTracker:
    """Runs one YOLOv8 model, splits results into tracked persons + phones."""

    def __init__(self, model_path: str = config.YOLO_MODEL_PATH):
        self.model = YOLO(model_path)

    def infer(self, frame: np.ndarray) -> "tuple[List[Detection], List[Detection]]":
        """
        Returns (persons, phones) for a single BGR frame.
        Persons carry a stable track_id (via ByteTrack) so we can maintain
        per-student rolling state across frames. Phones don't need tracking.
        """
        results = self.model.track(
            frame,
            persist=True,
            tracker=config.TRACKER_CFG,
            conf=config.YOLO_CONF_THRESHOLD,
            classes=[config.YOLO_PERSON_CLASS_ID, config.YOLO_PHONE_CLASS_ID],
            verbose=False,
        )

        persons: List[Detection] = []
        phones: List[Detection] = []

        if not results or results[0].boxes is None:
            return persons, phones

        boxes = results[0].boxes
        ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)

        for box, tid in zip(boxes, ids):
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            xyxy = box.xyxy[0].cpu().numpy()

            det = Detection(track_id=tid, bbox=xyxy, conf=conf, cls_id=cls_id)
            if cls_id == config.YOLO_PERSON_CLASS_ID:
                persons.append(det)
            elif cls_id == config.YOLO_PHONE_CLASS_ID:
                phones.append(det)

        return persons, phones


def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Standard IoU between two [x1,y1,x2,y2] boxes."""
    xa1, ya1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    xa2, ya2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])

    inter_w = max(0.0, xa2 - xa1)
    inter_h = max(0.0, ya2 - ya1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0.0


def expand_box(box: np.ndarray, frame_w: int, frame_h: int, margin_ratio: float = 0.25) -> np.ndarray:
    """Expand a person box outward so a phone held near the body still overlaps it."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    mx, my = w * margin_ratio, h * margin_ratio
    return np.array([
        max(0, x1 - mx), max(0, y1 - my),
        min(frame_w, x2 + mx), min(frame_h, y2 + my),
    ])
