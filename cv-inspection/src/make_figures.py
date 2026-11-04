"""Figures for the README / report: threshold trade-off, confusion matrices, example detections, SPC demo.

Output: results/figures/*.png (all derived from files in results/, plus one model pass for the montage)
"""
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from ultralytics import YOLO  # noqa: E402

import config as C  # noqa: E402
from common import CLASSES, DS, FIGS, RESULTS  # noqa: E402
from spc import p_chart  # noqa: E402

BLUE, ORANGE, GREY, GREEN = "#2a6fdb", "#d9822b", "#6b7280", "#2f9e5b"


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def threshold_tradeoff(thr):
    sw = pd.read_csv(RESULTS / "yolo_test_sweep.csv")
    sw.loc[sw.tp + sw.fp == 0, "precision"] = np.nan          # undefined with no detections
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(sw.thr, sw.flag_recall, color=GREEN, marker="o", ms=3, label="part-level flag recall")
    ax.plot(sw.thr, sw.recall, color=BLUE, marker="o", ms=3, label="box recall (IoU 0.5)")
    ax.plot(sw.thr, sw.precision, color=ORANGE, marker="o", ms=3, label="box precision")
    ax.axvline(thr, color=GREY, ls="--", lw=1)
    ax.text(thr + 0.01, 0.03, f"threshold {thr:.2f}\n(frozen from val)", fontsize=8, color=GREY)
    ax.set_xlabel("Confidence threshold"); ax.set_ylabel("Share"); ax.set_ylim(0, 1.02)
    ax.set_title("Test split: what each threshold buys", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3); clean(ax)
    fig.tight_layout(); fig.savefig(FIGS / "threshold_tradeoff.png", dpi=160); plt.close(fig)


def confusions():
    svm = pd.read_csv(RESULTS / "baseline_test_confusion.csv", index_col=0)
    yolo = pd.read_csv(RESULTS / "yolo_test_confusion.csv", index_col=0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, cm, title in ((axes[0], svm.to_numpy(), "SVM (LBP+HOG), image level"),
                          (axes[1], yolo.to_numpy()[:6], "YOLO11n top-1 box per image")):
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=45)
        cols = CLASSES if cm.shape[1] == 6 else CLASSES + ["none"]
        ax.set_xticks(range(len(cols)), cols, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(6), CLASSES, fontsize=8)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if cm[i, j]:
                    ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                            color="white" if cm[i, j] > 25 else "black")
        ax.set_title(title, fontsize=10); ax.set_xlabel("predicted"); ax.set_ylabel("true")
    fig.tight_layout(); fig.savefig(FIGS / "confusion_matrices.png", dpi=160); plt.close(fig)


DISPLAY_THR = 0.40      # boxes below this confidence are hidden in the montages for readability only


def tile(model, path, imgsz, label):
    im = cv2.imread(str(path))
    h, w = im.shape[:2]
    for line in (DS / "labels" / "test" / f"{path.stem}.txt").read_text().splitlines():
        _, x, y, bw, bh = (float(v) for v in line.split())
        cv2.rectangle(im, (int((x - bw / 2) * w), int((y - bh / 2) * h)),
                      (int((x + bw / 2) * w), int((y + bh / 2) * h)), (0, 200, 0), 1)
    r = model.predict(str(path), conf=DISPLAY_THR, imgsz=imgsz, verbose=False, device="cpu")[0]
    for b, sc in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
        cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 1)
        cv2.putText(im, f"{sc:.2f}", (int(b[0]) + 2, int(b[1]) + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    im = cv2.resize(im, (300, 300), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(im, (0, 0), (300, 20), (30, 30, 30), -1)
    cv2.putText(im, label, (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return im


def montage(tiles, name, cols=3):
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(FIGS / name), np.vstack(rows))


def examples(weights, imgsz):
    model = YOLO(weights)
    pred = pd.read_csv(RESULTS / "test_predictions.csv")
    rng = np.random.default_rng(0)
    ok, bad = [], []
    for c in CLASSES:      # one RANDOM correctly classified test image per class (not the best-looking ones)
        row = pred[(pred.true_class == c) & (pred.pred_class == c)].sample(1, random_state=int(rng.integers(1e6))).iloc[0]
        ok.append(tile(model, DS / "images" / "test" / row.image, imgsz, c))
    montage(ok, "examples.png")
    wrong = pred[pred.true_class != pred.pred_class]      # every part-level failure on the test split
    for _, row in wrong.iterrows():
        bad.append(tile(model, DS / "images" / "test" / row.image, imgsz, f"{row.true_class} -> {row.pred_class}"))
    if bad:
        montage(bad, "failures.png")


def spc_demo():
    df = pd.read_csv(RESULTS / "stream.csv")
    df["flagged"] = df.detected_class != "none"
    g = df.groupby("batch")
    ch = p_chart(g.flagged.sum().to_numpy(), g.size().to_numpy(), C.BASELINE_BATCHES)
    b = g.size().index.to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(b, ch["p"], marker="o", ms=3, color=BLUE, label="fraction flagged")
    ax.axhline(ch["pbar"], color=GREY, lw=1, label="centre line")
    ax.plot(b, ch["ucl"], "--", color=ORANGE, lw=1, label="UCL"); ax.plot(b, ch["lcl"], "--", color=ORANGE, lw=1)
    ax.scatter(b[ch["alarm"]], ch["p"][ch["alarm"]], color="red", s=20, zorder=3, label="alarm")
    ax.axvline(C.SHIFT_AT - 0.5, color="black", ls=":", lw=1); ax.text(C.SHIFT_AT, ax.get_ylim()[1] * 0.97, "true shift", fontsize=8, va="top")
    ax.axvline(C.BASELINE_BATCHES + 0.5, color=GREY, ls=":", lw=1)
    ax.set_xlabel("Batch"); ax.set_ylabel("Fraction flagged"); ax.set_title("p-chart, SIMULATED stream", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=4); clean(ax)
    cnt = df[df.flagged].detected_class.value_counts(); cum = cnt.cumsum() / cnt.sum() * 100
    ax = axes[1]; ax.bar(cnt.index, cnt.values, color=BLUE)
    ax2 = ax.twinx(); ax2.plot(cnt.index, cum.values, color=ORANGE, marker="o", ms=4); ax2.set_ylim(0, 105)
    ax2.set_ylabel("Cumulative %"); ax.set_ylabel("Flagged parts"); plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_title("Pareto of detected defect types", fontsize=10); clean(ax)
    fig.tight_layout(); fig.savefig(FIGS / "spc_demo.png", dpi=160); plt.close(fig)


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((RESULTS / "threshold.json").read_text())
    threshold_tradeoff(cfg["threshold"])
    confusions()
    examples(cfg["weights"], cfg["imgsz"])
    spc_demo()
    print("figures written to", FIGS)


if __name__ == "__main__":
    main()
