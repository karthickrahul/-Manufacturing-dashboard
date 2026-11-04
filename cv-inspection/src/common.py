"""Shared constants and helpers."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DS = ROOT / "datasets" / "neu"
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
SEED = 42
IMGSZ = 320          # source images are 200x200; 320 keeps small defects visible and is fast on CPU


def cls_of(stem: str) -> str:
    """File names look like '<class>_<number>' (class may contain '-' or '_')."""
    return stem.rsplit("_", 1)[0]


def iou(a, b):
    """IoU of two boxes in (x1, y1, x2, y2) format."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0
