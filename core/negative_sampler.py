import random
from typing import List, Optional

from .iou_calculator import IoUCalculator
from .obb import OrientedBoundingBox


class NegativeSampler:
    """Generate in-image boxes whose maximum IoU with real boxes is below a limit."""

    def __init__(
        self,
        min_iou: float = 0.0,
        max_iou: float = 0.3,
        seed: Optional[int] = None,
    ) -> None:
        if not 0.0 <= min_iou <= max_iou <= 1.0:
            raise ValueError("IoU bounds must satisfy 0 <= min_iou <= max_iou <= 1")
        self.min_iou = min_iou
        self.max_iou = max_iou
        self.random = random.Random(seed)

    def generate_negative_box(
        self,
        img_width: int,
        img_height: int,
        existing_boxes: List[OrientedBoundingBox],
        min_size: float = 0.1,
        max_size: float = 0.5,
        max_attempts: int = 100,
    ) -> Optional[OrientedBoundingBox]:
        if img_width <= 0 or img_height <= 0:
            raise ValueError("Image dimensions must be positive")
        if not 0 < min_size <= max_size <= 1:
            raise ValueError("Size ratios must satisfy 0 < min_size <= max_size <= 1")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")

        for _ in range(max_attempts):
            width = self.random.uniform(min_size, max_size) * img_width
            height = self.random.uniform(min_size, max_size) * img_height
            x_center = self.random.uniform(width / 2, img_width - width / 2)
            y_center = self.random.uniform(height / 2, img_height - height / 2)

            x_min = x_center - width / 2
            y_min = y_center - height / 2
            x_max = x_center + width / 2
            y_max = y_center + height / 2
            candidate = OrientedBoundingBox(
                points=[x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max],
                class_id=-1,
                difficult=0,
                is_normalized=False,
            )

            max_observed_iou = max(
                (IoUCalculator.calculate_iou(candidate, box) for box in existing_boxes),
                default=0.0,
            )
            if self.min_iou <= max_observed_iou <= self.max_iou:
                return candidate

        return None

    def generate_negative_boxes(
        self,
        img_width: int,
        img_height: int,
        existing_boxes: List[OrientedBoundingBox],
        num_boxes: int = 5,
        **kwargs,
    ) -> List[OrientedBoundingBox]:
        if num_boxes < 0:
            raise ValueError("num_boxes cannot be negative")

        occupied = list(existing_boxes)
        generated = []
        for _ in range(num_boxes):
            box = self.generate_negative_box(img_width, img_height, occupied, **kwargs)
            if box is not None:
                generated.append(box)
                occupied.append(box)
        return generated
