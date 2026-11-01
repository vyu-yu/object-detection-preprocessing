import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("ODP_DATA_DIR", BASE_DIR / "data"))
IMAGE_DIR = DATA_DIR / "images"
LABEL_DIR = DATA_DIR / "labels"
OUTPUT_DIR = Path(os.environ.get("ODP_OUTPUT_DIR", BASE_DIR / "outputs"))
VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"
PROCESSED_LABELS_DIR = OUTPUT_DIR / "processed_labels"


CLASS_MAP = {
    0: "plane",
    1: "ship",
    2: "storage-tank",
    3: "baseball-diamond",
    4: "tennis-court",
    5: "basketball-court",
    6: "ground-track-field",
    7: "harbor",
    8: "bridge",
    9: "large-vehicle",
    10: "small-vehicle",
    11: "helicopter",
    12: "roundabout",
    13: "soccer-ball-field",
    14: "swimming-pool",
    15: "container-crane",
}

CLASS_NAME_TO_ID = {name: class_id for class_id, name in CLASS_MAP.items()}
