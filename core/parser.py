import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import CLASS_MAP, CLASS_NAME_TO_ID
from .obb import OrientedBoundingBox


_META_PATTERN = re.compile(r"^['\"]?([A-Za-z][A-Za-z _-]*)['\"]?\s*:\s*(.*)$")


def _normalise_metadata_key(key: str) -> str:
    return key.strip().strip("'\"").lower().replace(" ", "_")


class LabelParser:
    """Parse and write DOTA-style oriented bounding box labels."""

    @staticmethod
    def parse_obb_txt(
        file_path: str | Path,
        image_id: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], List[OrientedBoundingBox]]:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Label file not found: {path}")

        metadata: Dict[str, Any] = {}
        boxes: List[OrientedBoundingBox] = []

        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue

                metadata_match = _META_PATTERN.match(line)
                if metadata_match:
                    key = _normalise_metadata_key(metadata_match.group(1))
                    if key:
                        metadata[key] = metadata_match.group(2).strip()
                    continue

                # Commas are accepted so files produced by the old writer remain readable.
                parts = line.replace(",", " ").split()
                if len(parts) < 10:
                    raise ValueError(
                        f"Invalid label at {path}:{line_number}; expected 8 coordinates, "
                        "category and difficult flag"
                    )

                try:
                    points = [float(value) for value in parts[:8]]
                    category = parts[8]
                    difficult = int(float(parts[9]))
                except ValueError as exc:
                    raise ValueError(f"Invalid numeric value at {path}:{line_number}") from exc

                class_id = CLASS_NAME_TO_ID.get(category)
                if class_id is None:
                    known = ", ".join(sorted(CLASS_NAME_TO_ID))
                    raise ValueError(
                        f"Unknown category '{category}' at {path}:{line_number}. "
                        f"Supported categories: {known}"
                    )

                boxes.append(
                    OrientedBoundingBox(
                        points=points,
                        class_id=class_id,
                        difficult=difficult,
                        image_id=image_id,
                        is_normalized=False,
                    )
                )

        return metadata, boxes

    @staticmethod
    def save_obb_txt(
        metadata: Dict[str, Any],
        boxes: List[OrientedBoundingBox],
        output_path: str | Path,
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        metadata_lines = []
        for key in ("acquisition_dates", "imagesource", "gsd"):
            if key in metadata and metadata[key] not in (None, ""):
                display_key = "acquisition dates" if key == "acquisition_dates" else key
                metadata_lines.append(f"{display_key}:{metadata[key]}")

        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for line in metadata_lines:
                handle.write(f"{line}\n")

            for box in boxes:
                points = " ".join(f"{value:.6f}" for value in box.points)
                class_name = CLASS_MAP.get(box.class_id, f"class_{box.class_id}")
                handle.write(f"{points} {class_name} {box.difficult}\n")

        return path
