import pytest

from config import CLASS_NAME_TO_ID
from core.negative_sampler import NegativeSampler
from core.obb import OrientedBoundingBox


def existing_box() -> OrientedBoundingBox:
    return OrientedBoundingBox(
        points=[0, 0, 20, 0, 20, 20, 0, 20],
        class_id=CLASS_NAME_TO_ID["small-vehicle"],
    )


def test_same_seed_reproduces_negative_boxes() -> None:
    first = NegativeSampler(max_iou=1.0, seed=123).generate_negative_boxes(
        320,
        240,
        [existing_box()],
        num_boxes=3,
    )
    second = NegativeSampler(max_iou=1.0, seed=123).generate_negative_boxes(
        320,
        240,
        [existing_box()],
        num_boxes=3,
    )

    assert [box.points for box in first] == [box.points for box in second]


def test_negative_boxes_stay_inside_image() -> None:
    boxes = NegativeSampler(max_iou=1.0, seed=7).generate_negative_boxes(
        320,
        240,
        [],
        num_boxes=10,
    )

    assert len(boxes) == 10
    for box in boxes:
        x_values = box.points[0::2]
        y_values = box.points[1::2]
        assert min(x_values) >= 0
        assert max(x_values) <= 320
        assert min(y_values) >= 0
        assert max(y_values) <= 240


def test_invalid_sampler_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="IoU bounds"):
        NegativeSampler(min_iou=0.8, max_iou=0.3)
