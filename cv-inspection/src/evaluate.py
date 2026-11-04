"""Evaluate a trained YOLO model, recall-first.

Two-step protocol (prevents tuning on the test set):
  1) python src/evaluate.py --weights W --split val  --recall-target 0.85
       -> sweeps the confidence threshold on val, writes results/threshold.json
  2) python src/evaluate.py --weights W --split test
       -> refuses to run without threshold.json; reports final numbers at the frozen threshold

Box-level matching: a detection is a true positive if it has the same class as an unmatched
ground-truth box and IoU >= 0.5. Part-level ("flag") metrics treat an image as flagged if the model
outputs at least one box above the threshold - this is the decision an inspection station makes.
Every NEU-DET image contains a defect, so false alarms on good parts CANNOT be measured here.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from ultralytics import YOLO

from common import CLASSES, DS, FIGS, IMGSZ, RESULTS, SEED, cls_of, iou

THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)


def load_gt(label_path, w, h):
    out = []
    txt = label_path.read_text().strip()
    for line in txt.splitlines() if txt else []:
        c, x, y, bw, bh = line.split()
        x, y, bw, bh = float(x) * w, float(y) * h, float(bw) * w, float(bh) * h
        out.append((int(c), (x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2)))
    return out


def run_predictions(model, split, imgsz):
    paths = sorted((DS / "images" / split).glob("*.jpg"))
    recs = []
    for r in model.predict(source=[str(p) for p in paths], conf=0.001, imgsz=imgsz,
                           stream=True, verbose=False, device="cpu"):
        h, w = r.orig_shape
        gt = load_gt(DS / "labels" / split / f"{Path(r.path).stem}.txt", w, h)
        preds = [(int(c), float(s), tuple(b)) for c, s, b in
                 zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())]
        recs.append({"path": r.path, "gt": gt, "preds": preds})
    return recs


def match_image(gt, preds, thr):
    """Greedy match by confidence. Returns (tp, fp, fn, per-class tp, per-class gt, fp/fn flags)."""
    used, tp, fp = set(), 0, 0
    tp_cls = {c: 0 for c in range(len(CLASSES))}
    for c, s, b in sorted([p for p in preds if p[1] >= thr], key=lambda p: -p[1]):
        best, best_j = 0.0, None
        for j, (gc, gb) in enumerate(gt):
            if j in used or gc != c:
                continue
            v = iou(b, gb)
            if v > best:
                best, best_j = v, j
        if best >= 0.5:
            tp += 1
            used.add(best_j)
            tp_cls[c] += 1
        else:
            fp += 1
    return tp, fp, len(gt) - len(used), tp_cls


def stats_at(recs, thr):
    tp = fp = fn = flagged = top1_ok = 0
    tp_cls = {c: 0 for c in range(len(CLASSES))}
    gt_cls = {c: 0 for c in range(len(CLASSES))}
    for r in recs:
        a, b, c_, tpc = match_image(r["gt"], r["preds"], thr)
        tp, fp, fn = tp + a, fp + b, fn + c_
        for k, v in tpc.items():
            tp_cls[k] += v
        for gc, _ in r["gt"]:
            gt_cls[gc] += 1
        keep = [p for p in r["preds"] if p[1] >= thr]
        if keep:
            flagged += 1
            top = max(keep, key=lambda p: p[1])
            top1_ok += int(CLASSES[top[0]] == cls_of(Path(r["path"]).stem))
    n = len(recs)
    return {"thr": float(thr), "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "false_detections_per_image": fp / n,
            "flag_recall": flagged / n,          # share of defective parts with >=1 detection
            "top1_class_accuracy": top1_ok / n,
            "per_class_recall": {CLASSES[c]: (tp_cls[c] / gt_cls[c] if gt_cls[c] else 0.0) for c in gt_cls}}


def choose_threshold(sweep, target, metric="recall"):
    """Highest threshold whose `metric` (box 'recall' or part-level 'flag_recall') meets the target."""
    ok = sweep[sweep[metric] >= target]
    if len(ok):
        row = ok.sort_values("thr").iloc[-1]
        return float(row.thr), f"highest threshold with {metric} >= {target}"
    f1 = 2 * sweep.precision * sweep.recall / (sweep.precision + sweep.recall).replace(0, np.nan)
    row = sweep.loc[f1.idxmax()]
    return float(row.thr), f"{metric} target {target} not reachable on val; fell back to max-F1 threshold"


def image_level(recs, thr):
    rows = []
    for r in recs:
        keep = [p for p in r["preds"] if p[1] >= thr]
        true_cls = cls_of(Path(r["path"]).stem)
        if keep:
            top = max(keep, key=lambda p: p[1])
            rows.append({"image": Path(r["path"]).name, "true_class": true_cls,
                         "pred_class": CLASSES[top[0]], "conf": top[1], "n_boxes": len(keep)})
        else:
            rows.append({"image": Path(r["path"]).name, "true_class": true_cls,
                         "pred_class": "none", "conf": 0.0, "n_boxes": 0})
    return pd.DataFrame(rows)


def draw(recs_sel, thr, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    for tag, r in recs_sel:
        im = cv2.imread(r["path"])
        for c, b in r["gt"]:
            cv2.rectangle(im, tuple(map(int, b[:2])), tuple(map(int, b[2:])), (0, 200, 0), 1)
        for c, s, b in r["preds"]:
            if s >= thr:
                cv2.rectangle(im, tuple(map(int, b[:2])), tuple(map(int, b[2:])), (0, 0, 255), 1)
                cv2.putText(im, f"{CLASSES[c][:6]} {s:.2f}", (int(b[0]), max(10, int(b[1]) - 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        cv2.imwrite(str(outdir / f"{tag}_{Path(r['path']).name}"), im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument("--recall-target", type=float, default=0.98)
    ap.add_argument("--metric", choices=["recall", "flag_recall"], default="flag_recall",
                    help="recall = box-level (IoU 0.5); flag_recall = share of defective parts with >=1 detection")
    ap.add_argument("--tag", default="yolo")
    a = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    model = YOLO(a.weights)
    recs = run_predictions(model, a.split, a.imgsz)
    sweep = pd.DataFrame([{k: v for k, v in stats_at(recs, t).items() if k != "per_class_recall"}
                          for t in THRESHOLDS])
    sweep.to_csv(RESULTS / f"{a.tag}_{a.split}_sweep.csv", index=False)
    print(sweep.round(3).to_string(index=False))

    if a.split == "val":
        thr, rule = choose_threshold(sweep, a.recall_target, a.metric)
        alt_thr, alt_rule = choose_threshold(sweep, 0.85, "recall")     # reference operating point
        (RESULTS / "threshold.json").write_text(json.dumps(
            {"threshold": thr, "rule": rule, "metric": a.metric, "target": a.recall_target,
             "alt_threshold_box_recall_0.85": alt_thr, "alt_rule": alt_rule,
             "chosen_on": "val", "weights": a.weights, "imgsz": a.imgsz}, indent=2))
        print(f"\nChosen threshold {thr} ({rule}). Reference box-recall operating point: {alt_thr}. "
              "Frozen in results/threshold.json")
        return

    # ---- test: frozen threshold only ----
    tj = RESULTS / "threshold.json"
    if not tj.exists():
        raise SystemExit("Run --split val first: the threshold must be chosen on val, not test.")
    cfg = json.loads(tj.read_text())
    thr = cfg["threshold"]
    s = stats_at(recs, thr)
    m = model.val(data=str(DS / "data.yaml"), split="test", imgsz=a.imgsz, device="cpu",
                  plots=True, name=f"{a.tag}_eval_test", exist_ok=True, verbose=False)
    s["mAP50"], s["mAP50_95"] = float(m.box.map50), float(m.box.map)
    s["ap50_per_class"] = {m.names[ci]: float(m.box.ap50[j]) for j, ci in enumerate(m.box.ap_class_index)}

    df = image_level(recs, thr)
    df.to_csv(RESULTS / "test_predictions.csv", index=False)
    labels = CLASSES + ["none"]
    pd.DataFrame(confusion_matrix(df.true_class, df.pred_class, labels=labels),
                 index=[f"true_{c}" for c in labels], columns=[f"pred_{c}" for c in labels]
                 ).to_csv(RESULTS / "yolo_test_confusion.csv")

    # Bootstrap over test images: uncertainty on the part-level flag recall (feeds the impact model)
    rng = np.random.default_rng(SEED)
    flagged = (df.n_boxes > 0).to_numpy().astype(float)
    boot = np.array([rng.choice(flagged, size=len(flagged)).mean() for _ in range(2000)])
    pd.DataFrame({"flag_recall": boot}).to_csv(RESULTS / "bootstrap_flag_recall.csv", index=False)
    s["flag_recall_ci95"] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    alt = stats_at(recs, cfg["alt_threshold_box_recall_0.85"])
    s["alt_operating_point_box_recall_0.85"] = {k: v for k, v in alt.items() if k != "per_class_recall"}
    s["n_test_images"] = len(recs)
    s["threshold_rule"] = cfg["rule"]
    s["weights"] = cfg["weights"]
    (RESULTS / "test_metrics.json").write_text(json.dumps(s, indent=2))
    print(json.dumps({k: v for k, v in s.items() if k != "per_class_recall"}, indent=2))
    print("per-class box recall:", {k: round(v, 3) for k, v in s["per_class_recall"].items()})
    print(classification_report(df.true_class, df.pred_class, labels=CLASSES + ["none"], zero_division=0))

    # error analysis images: misses and false alarms
    miss, fa = [], []
    for r in recs:
        _, fpi, fni, _ = match_image(r["gt"], r["preds"], thr)
        if fni:
            miss.append(("miss", r))
        if fpi:
            fa.append(("falsealarm", r))
    draw(miss[:10] + fa[:10], thr, RESULTS / "error_analysis")
    print(f"error analysis: {len(miss)} images with a missed box, {len(fa)} with a false detection")


if __name__ == "__main__":
    main()
