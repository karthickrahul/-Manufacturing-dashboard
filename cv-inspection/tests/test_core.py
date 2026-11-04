"""Fast unit tests for the pieces that are easy to get silently wrong."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common import cls_of, iou  # noqa: E402
from evaluate import choose_threshold, match_image  # noqa: E402
from spc import p_chart  # noqa: E402
import pandas as pd  # noqa: E402


def test_cls_of():
    assert cls_of("rolled-in_scale_176") == "rolled-in_scale"
    assert cls_of("pitted_surface_3") == "pitted_surface"
    assert cls_of("crazing_63") == "crazing"


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9


def test_match_image_counts():
    gt = [(0, (0, 0, 10, 10)), (1, (50, 50, 80, 80))]
    preds = [(0, 0.9, (0, 0, 10, 10)),        # TP
             (1, 0.8, (100, 100, 120, 120)),   # FP (no overlap)
             (1, 0.2, (50, 50, 80, 80))]       # TP only if below-threshold boxes are kept
    tp, fp, fn, _ = match_image(gt, preds, thr=0.5)
    assert (tp, fp, fn) == (1, 1, 1)
    tp, fp, fn, _ = match_image(gt, preds, thr=0.1)
    assert (tp, fp, fn) == (2, 1, 0)


def test_wrong_class_is_not_a_match():
    tp, fp, fn, _ = match_image([(0, (0, 0, 10, 10))], [(1, 0.9, (0, 0, 10, 10))], thr=0.5)
    assert (tp, fp, fn) == (0, 1, 1)


def test_choose_threshold_prefers_highest_that_meets_recall():
    sw = pd.DataFrame({"thr": [0.1, 0.2, 0.3], "recall": [0.95, 0.88, 0.70], "precision": [0.6, 0.8, 0.9]})
    thr, rule = choose_threshold(sw, 0.85)
    assert thr == 0.2 and "highest" in rule
    thr, rule = choose_threshold(sw, 0.99)
    assert "fell back" in rule


def test_p_chart_detects_shift_and_ignores_baseline():
    rng = np.random.default_rng(0)
    flags = np.concatenate([rng.binomial(200, 0.03, 20), rng.binomial(200, 0.12, 20)])
    out = p_chart(flags, np.full(40, 200), k_baseline=20)
    assert not out["alarm"][:20].any()
    assert out["alarm"][20:].any()
    assert (out["lcl"] >= 0).all()


def test_choose_threshold_part_level_metric():
    sw = pd.DataFrame({"thr": [0.1, 0.2, 0.3], "recall": [0.8, 0.7, 0.6], "flag_recall": [0.99, 0.97, 0.90],
                       "precision": [0.5, 0.6, 0.7]})
    thr, rule = choose_threshold(sw, 0.98, "flag_recall")
    assert thr == 0.1 and "flag_recall" in rule
