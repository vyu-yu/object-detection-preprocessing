import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import Polygon


class OrientedBoundingBox:
    """A four-point oriented bounding box represented by eight coordinates."""

    def __init__(
        self,
        points: List[float],
        class_id: int,
        difficult: int = 0,
        image_id: Optional[str] = None,
        is_normalized: bool = False,
    ) -> None:
        if len(points) != 8:
            raise ValueError("An oriented bounding box requires exactly 8 coordinates")
        if not all(math.isfinite(float(value)) for value in points):
            raise ValueError("Bounding box coordinates must be finite numbers")

        self.points = [float(value) for value in points]
        self.class_id = int(class_id)
        self.difficult = int(difficult)
        self.image_id = image_id
        self.is_normalized = bool(is_normalized)
        self.polygon = self._create_polygon()

    def _create_polygon(self) -> Polygon:
        points = [(self.points[i], self.points[i + 1]) for i in range(0, 8, 2)]
        polygon = Polygon(points)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        return polygon

    def area(self) -> float:
        return float(self.polygon.area)

    def to_normalized(self, img_width: int, img_height: int) -> "OrientedBoundingBox":
        if self.is_normalized:
            return self
        if img_width <= 0 or img_height <= 0:
            raise ValueError("Image dimensions must be positive")

        points = []
        for index in range(0, 8, 2):
            points.extend(
                [
                    self.points[index] / img_width,
                    self.points[index + 1] / img_height,
                ]
            )
        return OrientedBoundingBox(
            points=points,
            class_id=self.class_id,
            difficult=self.difficult,
            image_id=self.image_id,
            is_normalized=True,
        )

    def to_absolute(self, img_width: int, img_height: int) -> "OrientedBoundingBox":
        if not self.is_normalized:
            return self
        if img_width <= 0 or img_height <= 0:
            raise ValueError("Image dimensions must be positive")

        points = []
        for index in range(0, 8, 2):
            points.extend(
                [
                    self.points[index] * img_width,
                    self.points[index + 1] * img_height,
                ]
            )
        return OrientedBoundingBox(
            points=points,
            class_id=self.class_id,
            difficult=self.difficult,
            image_id=self.image_id,
            is_normalized=False,
        )

    def get_axis_aligned_bbox(self) -> Tuple[float, float, float, float]:
        x_coordinates = self.points[0::2]
        y_coordinates = self.points[1::2]
        return (
            min(x_coordinates),
            min(y_coordinates),
            max(x_coordinates),
            max(y_coordinates),
        )

    def get_center_point(self) -> Tuple[float, float]:
        x_coordinates = self.points[0::2]
        y_coordinates = self.points[1::2]
        return sum(x_coordinates) / 4.0, sum(y_coordinates) / 4.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "points": self.points,
            "class_id": self.class_id,
            "difficult": self.difficult,
            "image_id": self.image_id,
            "is_normalized": self.is_normalized,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrientedBoundingBox":
        return cls(
            points=data["points"],
            class_id=data["class_id"],
            difficult=data.get("difficult", 0),
            image_id=data.get("image_id"),
            is_normalized=data.get("is_normalized", False),
        )

    def __repr__(self) -> str:
        return (
            f"OrientedBoundingBox(class_id={self.class_id}, "
            f"difficult={self.difficult}, points={self.points})"
        )
