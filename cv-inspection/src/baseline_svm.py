"""Classical baseline: LBP + HOG + intensity statistics -> RBF SVM (image-level defect type).

Tuned on val only; the test split is scored once with the frozen best setting.
Outputs: results/baseline_val_grid.csv, results/baseline_test.json, results/baseline_test_confusion.csv
"""
import itertools
import json

import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog, local_binary_pattern
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import CLASSES, DS, RESULTS, SEED, cls_of

P, R = 8, 1
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def features(path, use_clahe):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    if use_clahe:
        img = _clahe.apply(img)
    lbp = local_binary_pattern(img, P, R, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, P + 3), density=True)
    hog_vec = hog(img, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2))
    return np.concatenate([lbp_hist, hog_vec, [img.mean(), img.std()]])


def load(split, use_clahe):
    paths = sorted((DS / "images" / split).glob("*.jpg"))
    X = np.array([features(p, use_clahe) for p in paths])
    y = np.array([cls_of(p.stem) for p in paths])
    return X, y


def main():
    RESULTS.mkdir(exist_ok=True)
    data = {c: {s: load(s, c) for s in ("train", "val")} for c in (False, True)}

    rows = []
    for clahe, C, gamma in itertools.product([False, True], [1, 10, 100], ["scale", 0.01]):
        Xtr, ytr = data[clahe]["train"]
        Xva, yva = data[clahe]["val"]
        clf = make_pipeline(StandardScaler(), SVC(C=C, kernel="rbf", gamma=gamma, random_state=SEED))
        clf.fit(Xtr, ytr)
        rows.append({"clahe": clahe, "C": C, "gamma": gamma,
                     "val_accuracy": accuracy_score(yva, clf.predict(Xva))})
    grid = pd.DataFrame(rows).sort_values("val_accuracy", ascending=False)
    grid.to_csv(RESULTS / "baseline_val_grid.csv", index=False)
    print(grid.to_string(index=False))

    best = grid.iloc[0]
    clahe, C = bool(best.clahe), best.C
    gamma = best.gamma if best.gamma == "scale" else float(best.gamma)
    Xtr, ytr = data[clahe]["train"]
    Xva, yva = data[clahe]["val"]
    # final model: train on train only (same data as YOLO), score test once
    clf = make_pipeline(StandardScaler(), SVC(C=C, kernel="rbf", gamma=gamma, random_state=SEED))
    clf.fit(Xtr, ytr)
    Xte, yte = load("test", clahe)
    pred = clf.predict(Xte)
    rep = classification_report(yte, pred, labels=CLASSES, output_dict=True, zero_division=0)
    out = {"best_setting": {"clahe": clahe, "C": float(C), "gamma": str(gamma)},
           "test_accuracy": accuracy_score(yte, pred),
           "per_class_recall": {c: rep[c]["recall"] for c in CLASSES}}
    (RESULTS / "baseline_test.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame(confusion_matrix(yte, pred, labels=CLASSES), index=CLASSES, columns=CLASSES).to_csv(
        RESULTS / "baseline_test_confusion.csv")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
