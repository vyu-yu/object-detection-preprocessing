from typing import List, Tuple

from .iou_calculator import IoUCalculator
from .obb import OrientedBoundingBox


class OverlapDetector:
    """Detect and optionally filter heavily overlapping bounding boxes."""

    def __init__(self, iou_threshold: float = 0.5) -> None:
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("IoU threshold must be between 0 and 1")
        self.iou_threshold = iou_threshold

    def detect_overlaps(
        self, boxes: List[OrientedBoundingBox]
    ) -> List[Tuple[int, int, float]]:
        overlaps = []
        for index1 in range(len(boxes)):
            for index2 in range(index1 + 1, len(boxes)):
                iou = IoUCalculator.calculate_iou(boxes[index1], boxes[index2])
                if iou > self.iou_threshold:
                    overlaps.append((index1, index2, iou))
        return overlaps

    def filter_overlaps(
        self,
        boxes: List[OrientedBoundingBox],
        strategy: str = "keep_largest",
    ) -> List[OrientedBoundingBox]:
        if strategy not in {"keep_largest", "keep_smallest"}:
            raise ValueError("Strategy must be 'keep_largest' or 'keep_smallest'")

        to_remove = set()
        for index1, index2, _ in self.detect_overlaps(boxes):
            area1 = boxes[index1].area()
            area2 = boxes[index2].area()
            if strategy == "keep_largest":
                to_remove.add(index2 if area1 >= area2 else index1)
            else:
                to_remove.add(index1 if area1 >= area2 else index2)

        return [box for index, box in enumerate(boxes) if index not in to_remove]
