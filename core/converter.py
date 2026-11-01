from typing import List, Tuple

from .obb import OrientedBoundingBox


class CoordinateConverter:
    """Convert oriented boxes between normalized and absolute coordinates."""

    @staticmethod
    def convert_boxes_to_absolute(
        boxes: List[OrientedBoundingBox],
        img_width: int,
        img_height: int,
    ) -> List[OrientedBoundingBox]:
        return [box.to_absolute(img_width, img_height) for box in boxes]

    @staticmethod
    def convert_boxes_to_normalized(
        boxes: List[OrientedBoundingBox],
        img_width: int,
        img_height: int,
    ) -> List[OrientedBoundingBox]:
        return [box.to_normalized(img_width, img_height) for box in boxes]

    @staticmethod
    def convert_to_axis_aligned(
        boxes: List[OrientedBoundingBox],
    ) -> List[Tuple[float, float, float, float]]:
        return [box.get_axis_aligned_bbox() for box in boxes]
