from typing import List, Tuple

import numpy as np

from .obb import OrientedBoundingBox


class IoUCalculator:
    """Geometry helpers for oriented bounding boxes."""

    @staticmethod
    def calculate_iou(box1: OrientedBoundingBox, box2: OrientedBoundingBox) -> float:
        if box1.is_normalized != box2.is_normalized:
            raise ValueError("Both bounding boxes must use the same coordinate system")

        intersection = box1.polygon.intersection(box2.polygon).area
        union = box1.area() + box2.area() - intersection
        if union <= 0:
            return 0.0
        return float(intersection / union)

    @staticmethod
    def calculate_iou_matrix(
        boxes1: List[OrientedBoundingBox],
        boxes2: List[OrientedBoundingBox],
    ) -> np.ndarray:
        matrix = np.zeros((len(boxes1), len(boxes2)), dtype=float)
        for index1, box1 in enumerate(boxes1):
            for index2, box2 in enumerate(boxes2):
                matrix[index1, index2] = IoUCalculator.calculate_iou(box1, box2)
        return matrix

    @staticmethod
    def find_max_iou_pair(
        boxes1: List[OrientedBoundingBox],
        boxes2: List[OrientedBoundingBox],
    ) -> Tuple[int, int, float]:
        if not boxes1 or not boxes2:
            raise ValueError("Both collections must contain at least one bounding box")

        matrix = IoUCalculator.calculate_iou_matrix(boxes1, boxes2)
        flat_index = int(np.argmax(matrix))
        index1, index2 = np.unravel_index(flat_index, matrix.shape)
        return int(index1), int(index2), float(matrix[index1, index2])
