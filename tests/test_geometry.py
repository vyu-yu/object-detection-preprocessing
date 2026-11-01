import pytest

from config import CLASS_NAME_TO_ID
from core.converter import CoordinateConverter
from core.iou_calculator import IoUCalculator
from core.obb import OrientedBoundingBox
from core.overlap_detector import OverlapDetector


def box(points):
    return OrientedBoundingBox(
        points=points,
        class_id=CLASS_NAME_TO_ID["small-vehicle"],
    )


def test_normalized_and_absolute_coordinate_conversion() -> None:
    absolute = box([10, 20, 40, 20, 40, 50, 10, 50])

    normalized = absolute.to_normalized(100, 200)
    restored = normalized.to_absolute(100, 200)

    assert normalized.is_normalized is True
    assert normalized.points == pytest.approx([0.1, 0.1, 0.4, 0.1, 0.4, 0.25, 0.1, 0.25])
    assert restored.points == pytest.approx(absolute.points)
    assert CoordinateConverter.convert_to_axis_aligned([absolute]) == [
        (10.0, 20.0, 40.0, 50.0)
    ]


def test_iou_for_identical_and_disjoint_boxes() -> None:
    first = box([0, 0, 10, 0, 10, 10, 0, 10])
    identical = box([0, 0, 10, 0, 10, 10, 0, 10])
    disjoint = box([20, 20, 30, 20, 30, 30, 20, 30])

    assert IoUCalculator.calculate_iou(first, identical) == pytest.approx(1.0)
    assert IoUCalculator.calculate_iou(first, disjoint) == pytest.approx(0.0)


def test_iou_rejects_mixed_coordinate_systems() -> None:
    absolute = box([0, 0, 10, 0, 10, 10, 0, 10])
    normalized = absolute.to_normalized(100, 100)

    with pytest.raises(ValueError, match="same coordinate system"):
        IoUCalculator.calculate_iou(absolute, normalized)


def test_overlap_detection_and_filtering() -> None:
    large = box([0, 0, 10, 0, 10, 10, 0, 10])
    duplicate = box([0, 0, 10, 0, 10, 10, 0, 10])
    small = box([1, 1, 2, 1, 2, 2, 1, 2])
    detector = OverlapDetector(iou_threshold=0.0)

    overlaps = detector.detect_overlaps([large, duplicate, small])

    assert len(overlaps) == 3
    assert detector.filter_overlaps([large, duplicate, small], "keep_smallest") == [small]
