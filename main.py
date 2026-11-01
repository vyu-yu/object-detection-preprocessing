import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import cv2

from config import OUTPUT_DIR
from core.iou_calculator import IoUCalculator
from core.negative_sampler import NegativeSampler
from core.overlap_detector import OverlapDetector
from core.parser import LabelParser
from core.visualizer import Visualizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess DOTA-style oriented bounding box labels."
    )
    parser.add_argument("--image", type=Path, required=True, help="input image path")
    parser.add_argument("--label", type=Path, required=True, help="input label path")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR,
        help="output directory (default: ./outputs)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="save annotated boxes and the first IoU pair",
    )
    parser.add_argument(
        "--visualize-overlaps",
        "--visualize_overlaps",
        dest="visualize_overlaps",
        action="store_true",
        help="save a visualization of overlapping boxes",
    )
    parser.add_argument(
        "--iou-threshold",
        "--iou_threshold",
        dest="iou_threshold",
        type=float,
        default=0.5,
        help="overlap threshold in [0, 1] (default: 0.5)",
    )
    parser.add_argument(
        "--generate-negatives",
        "--generate_negatives",
        dest="generate_negatives",
        action="store_true",
        help="generate negative boxes with low IoU against real boxes",
    )
    parser.add_argument(
        "--num-negatives",
        "--num_negatives",
        dest="num_negatives",
        type=int,
        default=3,
        help="maximum number of negative boxes to generate (default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for reproducible negative sampling (default: 42)",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    image_path = args.image.expanduser().resolve()
    label_path = args.label.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()

    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not label_path.is_file():
        raise FileNotFoundError(f"Label file not found: {label_path}")
    if args.num_negatives < 0:
        raise ValueError("--num-negatives cannot be negative")
    if not 0.0 <= args.iou_threshold <= 1.0:
        raise ValueError("--iou-threshold must be between 0 and 1")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")
    image_height, image_width = image.shape[:2]

    metadata, boxes = LabelParser.parse_obb_txt(label_path, image_id=image_path.stem)
    print(f"Parsed {len(boxes)} oriented bounding boxes from {label_path.name}")
    print(f"Metadata: {metadata}")

    visualizer = Visualizer(output_dir=output_dir / "visualizations")
    detector = OverlapDetector(iou_threshold=args.iou_threshold)
    overlaps = detector.detect_overlaps(boxes)
    print(f"Detected {len(overlaps)} overlapping box pairs")

    if len(boxes) >= 2:
        iou = IoUCalculator.calculate_iou(boxes[0], boxes[1])
        print(f"IoU of the first two boxes: {iou:.4f}")
        if args.visualize:
            output_path = visualizer.visualize_iou(
                image_path,
                boxes[0],
                boxes[1],
                iou,
            )
            print(f"Saved IoU visualization: {output_path}")

    if args.visualize:
        output_path = visualizer.visualize_and_save(
            image_path,
            boxes,
            output_name="original_boxes",
        )
        print(f"Saved box visualization: {output_path}")

    if args.visualize_overlaps:
        output_path = visualizer.visualize_overlaps(
            image_path,
            boxes,
            output_name="overlaps",
        )
        print(f"Saved overlap visualization: {output_path}")

    if args.generate_negatives:
        sampler = NegativeSampler(seed=args.seed)
        negative_boxes = sampler.generate_negative_boxes(
            image_width,
            image_height,
            boxes,
            num_boxes=args.num_negatives,
        )
        print(f"Generated {len(negative_boxes)} negative boxes")
        negative_output = output_dir / "negative_boxes" / f"{label_path.stem}.json"
        negative_output.parent.mkdir(parents=True, exist_ok=True)
        negative_output.write_text(
            json.dumps(
                {
                    "seed": args.seed,
                    "max_iou": 0.3,
                    "boxes": [box.points for box in negative_boxes],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Saved negative boxes: {negative_output}")
        if args.visualize:
            output_path = visualizer.visualize_and_save(
                image_path,
                boxes + negative_boxes,
                output_name="with_negatives",
            )
            print(f"Saved negative-box visualization: {output_path}")

    label_output = output_dir / label_path.name
    LabelParser.save_obb_txt(metadata, boxes, label_output)
    print(f"Saved processed label: {label_output}")
    print(f"Output root: {output_dir}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
