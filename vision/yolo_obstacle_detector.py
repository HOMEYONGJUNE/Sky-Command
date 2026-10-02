"""Ultralytics YOLO(.pt) 기반 장애물 검출기."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    box: Tuple[int, int, int, int]
    confidence: float
    class_id: int
    label: str


class YOLOObstacleDetector:
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        image_size: int = 320,
        box_scale: float = 1.5,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics가 필요합니다. 관제 PC에서 'pip install ultralytics'를 실행하세요."
            ) from exc

        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.device = self._select_device()
        self.box_scale = max(1.0, float(box_scale))

    @staticmethod
    def _select_device() -> str:
        try:
            import torch
            if torch.backends.mps.is_available():
                print("[YOLO] Using Apple Silicon MPS acceleration")
                return "mps"
        except (ImportError, AttributeError):
            pass
        print("[YOLO] MPS unavailable; using CPU inference")
        return "cpu"

    def detect(self, frame: np.ndarray) -> Tuple[List[Detection], np.ndarray]:
        if frame is None or frame.size == 0:
            raise ValueError("장애물 검출 입력 영상이 비어 있습니다.")

        result = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device=self.device,
            max_det=20,
            verbose=False,
        )[0]
        height, width = frame.shape[:2]
        detections: List[Detection] = []
        mask = np.zeros((height, width), dtype=np.uint8)
        names = result.names

        if result.boxes is None:
            return detections, mask

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for coordinates, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = coordinates.astype(int)
            x1 = int(np.clip(x1, 0, width - 1))
            y1 = int(np.clip(y1, 0, height - 1))
            x2 = int(np.clip(x2, 0, width - 1))
            y2 = int(np.clip(y2, 0, height - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            half_width = (x2 - x1) * self.box_scale / 2.0
            half_height = (y2 - y1) * self.box_scale / 2.0
            x1 = int(np.clip(center_x - half_width, 0, width - 1))
            y1 = int(np.clip(center_y - half_height, 0, height - 1))
            x2 = int(np.clip(center_x + half_width, 0, width - 1))
            y2 = int(np.clip(center_y + half_height, 0, height - 1))
            if isinstance(names, dict):
                label = str(names.get(int(class_id), class_id))
            else:
                label = str(names[int(class_id)]) if int(class_id) < len(names) else str(class_id)
            detection = Detection(
                box=(x1, y1, x2 - x1, y2 - y1),
                confidence=float(score),
                class_id=int(class_id),
                label=label,
            )
            detections.append(detection)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        return detections, mask
