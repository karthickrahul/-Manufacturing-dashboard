"""Build the SIMULATED production stream shown in the dashboard.

Defective parts reuse REAL model outputs from the NEU-DET test split (results/test_predictions.csv),
so missed and mislabelled defects appear exactly as often as the model produces them. Good parts, the
defect rate, the process shift and the false-alarm rate are ASSUMPTIONS (src/config.py).
Output: results/stream.csv
"""
import numpy as np
import pandas as pd

import config as C
from common import CLASSES, RESULTS

SEED_STREAM = 7


def main():
    rng = np.random.default_rng(SEED_STREAM)
    pred = pd.read_csv(RESULTS / "test_predictions.csv")
    assert set(C.MIX) == set(CLASSES) == set(pred.true_class)
    by_class = {c: pred[pred.true_class == c].reset_index(drop=True) for c in CLASSES}
    classes, weights = list(C.MIX), list(C.MIX.values())

    rows = []
    for b in range(1, C.N_BATCHES + 1):
        p = C.P0 if b < C.SHIFT_AT else C.P1
        for i in range(C.BATCH_SIZE):
            if rng.random() < p:
                cls = rng.choice(classes, p=weights)
                r = by_class[cls].iloc[rng.integers(len(by_class[cls]))]
                rows.append((b, i, 1, cls, r.pred_class))
            else:
                det = rng.choice(classes) if rng.random() < C.FALSE_ALARM_AI else "none"
                rows.append((b, i, 0, "good", det))
    df = pd.DataFrame(rows, columns=["batch", "part", "truth_defective", "true_class", "detected_class"])
    df.to_csv(RESULTS / "stream.csv", index=False)
    print(f"{len(df)} parts, {int(df.truth_defective.sum())} defective, shift at batch {C.SHIFT_AT}")


if __name__ == "__main__":
    main()
