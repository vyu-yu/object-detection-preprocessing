from pathlib import Path

import pytest

from config import CLASS_NAME_TO_ID
from core.obb import OrientedBoundingBox
from core.parser import LabelParser


def make_box(points=None) -> OrientedBoundingBox:
    return OrientedBoundingBox(
        points=points or [10, 20, 40, 20, 40, 50, 10, 50],
        class_id=CLASS_NAME_TO_ID["small-vehicle"],
        difficult=1,
        image_id="sample",
    )


def test_label_round_trip_uses_whitespace_not_commas(tmp_path: Path) -> None:
    metadata = {
        "acquisition_dates": "2025-01-01",
        "imagesource": "GoogleEarth",
        "gsd": "0.25",
    }
    output = tmp_path / "round_trip.txt"

    LabelParser.save_obb_txt(metadata, [make_box()], output)
    parsed_metadata, parsed_boxes = LabelParser.parse_obb_txt(output)

    assert parsed_metadata == metadata
    assert len(parsed_boxes) == 1
    assert parsed_boxes[0].points == pytest.approx(make_box().points)
    assert "," not in output.read_text(encoding="utf-8")


def test_parser_accepts_commas_from_legacy_files(tmp_path: Path) -> None:
    label = tmp_path / "legacy.txt"
    label.write_text(
        "10,20,40,20,40,50,10,50,small-vehicle,0\n",
        encoding="utf-8",
    )

    _, boxes = LabelParser.parse_obb_txt(label)

    assert len(boxes) == 1
    assert boxes[0].points == [10.0, 20.0, 40.0, 20.0, 40.0, 50.0, 10.0, 50.0]


def test_parser_reports_unknown_category(tmp_path: Path) -> None:
    label = tmp_path / "unknown.txt"
    label.write_text("10 20 40 20 40 50 10 50 unknown-class 0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown category 'unknown-class'"):
        LabelParser.parse_obb_txt(label)
