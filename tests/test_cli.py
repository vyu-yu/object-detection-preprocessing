import json
from pathlib import Path

import cv2
import numpy as np

import main
from core.parser import LabelParser


def test_cli_end_to_end_uses_requested_output_directory(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpg"
    label_path = tmp_path / "sample.txt"
    output_dir = tmp_path / "custom-output"
    assert cv2.imwrite(str(image_path), np.zeros((128, 128, 3), dtype=np.uint8))

    label_path.write_text(
        "imagesource:GoogleEarth\n"
        "gsd:0.2\n"
        "10 10 40 10 40 40 10 40 small-vehicle 0\n"
        "50 50 80 50 80 80 50 80 large-vehicle 1\n",
        encoding="utf-8",
    )

    result = main.main(
        [
            "--image",
            str(image_path),
            "--label",
            str(label_path),
            "--output",
            str(output_dir),
            "--visualize",
            "--visualize-overlaps",
            "--generate-negatives",
            "--num-negatives",
            "2",
            "--seed",
            "99",
        ]
    )

    assert result == 0
    assert (output_dir / "sample.txt").is_file()
    assert (output_dir / "visualizations" / "original_boxes.jpg").is_file()
    assert (output_dir / "visualizations" / "overlaps.jpg").is_file()
    assert (output_dir / "visualizations" / "with_negatives.jpg").is_file()

    negative_output = output_dir / "negative_boxes" / "sample.json"
    payload = json.loads(negative_output.read_text(encoding="utf-8"))
    assert payload["seed"] == 99
    assert isinstance(payload["boxes"], list)

    metadata, boxes = LabelParser.parse_obb_txt(output_dir / "sample.txt")
    assert metadata["imagesource"] == "GoogleEarth"
    assert len(boxes) == 2
