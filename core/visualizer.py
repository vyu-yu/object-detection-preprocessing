from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import CLASS_MAP, VISUALIZATION_DIR
from .obb import OrientedBoundingBox
from .overlap_detector import OverlapDetector


class Visualizer:
    """Draw oriented boxes and save visualization results."""

    def __init__(
        self,
        class_map: Optional[Dict[int, str]] = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.class_map = class_map or CLASS_MAP
        self.output_dir = Path(output_dir) if output_dir else Path(VISUALIZATION_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.colors = self._generate_colors(len(self.class_map))
        self.negative_color = (0, 128, 255)

    @staticmethod
    def _generate_colors(num_colors: int) -> List[Tuple[int, int, int]]:
        rng = np.random.default_rng(42)
        return [
            tuple(int(value) for value in rng.integers(40, 240, size=3))
            for _ in range(max(num_colors, 1))
        ]

    def _color_for(self, class_id: int) -> Tuple[int, int, int]:
        if class_id < 0:
            return self.negative_color
        return self.colors[class_id % len(self.colors)]

    @staticmethod
    def _read_image(image_path: str | Path) -> np.ndarray:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"OpenCV could not read image: {image_path}")
        return image

    @staticmethod
    def _as_absolute(
        boxes: List[OrientedBoundingBox],
        img_width: int,
        img_height: int,
    ) -> List[OrientedBoundingBox]:
        return [
            box.to_absolute(img_width, img_height) if box.is_normalized else box
            for box in boxes
        ]

    @staticmethod
    def _box_points(box: OrientedBoundingBox) -> List[Tuple[int, int]]:
        return [
            (int(round(box.points[index])), int(round(box.points[index + 1])))
            for index in range(0, 8, 2)
        ]

    def draw_boxes(
        self,
        image: np.ndarray,
        boxes: List[OrientedBoundingBox],
        img_width: Optional[int] = None,
        img_height: Optional[int] = None,
        show_labels: bool = True,
        thickness: int = 2,
    ) -> np.ndarray:
        output = image.copy()
        height, width = output.shape[:2]
        img_width = img_width or width
        img_height = img_height or height

        for box in boxes:
            absolute_box = (
                box.to_absolute(img_width, img_height) if box.is_normalized else box
            )
            points = self._box_points(absolute_box)
            color = self._color_for(box.class_id)

            for index in range(4):
                cv2.line(
                    output,
                    points[index],
                    points[(index + 1) % 4],
                    color,
                    thickness,
                    cv2.LINE_AA,
                )

            if show_labels:
                label = (
                    "negative"
                    if box.class_id < 0
                    else self.class_map.get(box.class_id, f"class-{box.class_id}")
                )
                (text_width, text_height), baseline = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    1,
                )
                text_x = max(0, min(points[0][0], width - text_width - 1))
                text_y = points[0][1] - 5
                if text_y < text_height + baseline:
                    text_y = min(
                        height - 1,
                        points[0][1] + text_height + baseline + 2,
                    )

                cv2.rectangle(
                    output,
                    (text_x, text_y - text_height - baseline),
                    (text_x + text_width, text_y),
                    color,
                    -1,
                )
                cv2.putText(
                    output,
                    label,
                    (text_x, text_y - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            cv2.circle(output, points[0], 4, (0, 0, 255), -1, cv2.LINE_AA)

        return output

    def _write(self, path: Path, image: np.ndarray) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), image):
            raise OSError(f"Could not write visualization: {path}")
        return path

    def visualize_and_save(
        self,
        image_path: str | Path,
        boxes: List[OrientedBoundingBox],
        output_name: Optional[str] = None,
        output_dir: str | Path | None = None,
        **draw_options,
    ) -> Path:
        image = self._read_image(image_path)
        height, width = image.shape[:2]
        visualized = self.draw_boxes(image, boxes, width, height, **draw_options)
        base_name = Path(image_path).stem
        output_name = output_name or f"{base_name}_visualized"
        directory = Path(output_dir) if output_dir else self.output_dir
        return self._write(directory / f"{output_name}.jpg", visualized)

    def visualize_iou(
        self,
        image_path: str | Path,
        box1: OrientedBoundingBox,
        box2: OrientedBoundingBox,
        iou_value: float,
        output_name: Optional[str] = None,
        output_dir: str | Path | None = None,
    ) -> Path:
        image = self._read_image(image_path)
        height, width = image.shape[:2]
        box1, box2 = self._as_absolute([box1, box2], width, height)
        image = self.draw_boxes(image, [box1], width, height, show_labels=False)
        image = self.draw_boxes(image, [box2], width, height, show_labels=False)
        self._fill_intersection(image, box1, box2, (0, 255, 255))

        cv2.putText(
            image,
            f"IoU: {iou_value:.3f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        base_name = Path(image_path).stem
        output_name = output_name or f"{base_name}_iou_{iou_value:.3f}"
        directory = Path(output_dir) if output_dir else self.output_dir
        return self._write(directory / f"{output_name}.jpg", image)

    def visualize_overlaps(
        self,
        image_path: str | Path,
        boxes: List[OrientedBoundingBox],
        output_name: Optional[str] = None,
        output_dir: str | Path | None = None,
        iou_threshold: float = 0.1,
    ) -> Path:
        image = self._read_image(image_path)
        height, width = image.shape[:2]
        boxes = self._as_absolute(boxes, width, height)
        output = self.draw_boxes(image, boxes, width, height, show_labels=False)
        overlaps = OverlapDetector(iou_threshold).detect_overlaps(boxes)

        for index1, index2, iou in overlaps:
            self._fill_intersection(
                output,
                boxes[index1],
                boxes[index2],
                (0, 255, 255),
                label=f"{iou:.2f}",
            )

        base_name = Path(image_path).stem
        output_name = output_name or f"{base_name}_overlaps"
        directory = Path(output_dir) if output_dir else self.output_dir
        return self._write(directory / f"{output_name}.jpg", output)

    @staticmethod
    def _fill_intersection(
        image: np.ndarray,
        box1: OrientedBoundingBox,
        box2: OrientedBoundingBox,
        color: Tuple[int, int, int],
        label: Optional[str] = None,
    ) -> None:
        intersection = box1.polygon.intersection(box2.polygon)
        if intersection.is_empty:
            return

        polygons = (
            [intersection]
            if intersection.geom_type == "Polygon"
            else list(intersection.geoms)
            if intersection.geom_type == "MultiPolygon"
            else []
        )
        for polygon in polygons:
            points = np.array(
                [(int(x), int(y)) for x, y in polygon.exterior.coords],
                dtype=np.int32,
            )
            cv2.fillPoly(image, [points], color, cv2.LINE_AA)
            if label:
                center = polygon.representative_point()
                cv2.putText(
                    image,
                    label,
                    (int(center.x), int(center.y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
