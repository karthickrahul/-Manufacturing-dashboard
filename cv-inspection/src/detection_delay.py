"""Monte Carlo: how fast does a p-chart notice a process shift under two inspection strategies?

  ai      - every part is screened; part flagged with probability recall_ai (MEASURED on the NEU-DET
            test split, resampled from the bootstrap) or FALSE_ALARM_AI for good parts (ASSUMED)
  manual_full   - a person inspects EVERY part and misses MISS_MANUAL of defects (ASSUMED)
  manual_sample - a person inspects MANUAL_SAMPLE parts per batch, same miss rate (ASSUMED)

The process shifts from defect rate P0 to P1 at batch SHIFT_AT. Control limits come from the first
BASELINE_BATCHES batches. Delay = batches from the shift until the first alarm (0 = same batch).
Outputs: results/detection_delay.csv, results/detection_delay_summary.json
"""
import argparse
import json

import numpy as np
import pandas as pd

import config as C
from common import RESULTS, SEED
from spc import p_chart


def simulate_run(rng, strategy, recall_ai):
    flags = np.zeros(C.N_BATCHES)
    sizes = np.zeros(C.N_BATCHES)
    for b in range(C.N_BATCHES):
        p = C.P0 if b + 1 < C.SHIFT_AT else C.P1
        n_def = rng.binomial(C.BATCH_SIZE, p)
        n_good = C.BATCH_SIZE - n_def
        if strategy == "ai":
            sizes[b] = C.BATCH_SIZE
            flags[b] = rng.binomial(n_def, recall_ai) + rng.binomial(n_good, C.FALSE_ALARM_AI)
        else:
            n = C.BATCH_SIZE if strategy == "manual_full" else C.MANUAL_SAMPLE
            sizes[b] = n
            d = rng.hypergeometric(n_def, n_good, n)
            flags[b] = rng.binomial(d, 1 - C.MISS_MANUAL) + rng.binomial(n - d, C.FALSE_ALARM_MANUAL)
    alarm = p_chart(flags, sizes, C.BASELINE_BATCHES)["alarm"]
    shift_idx = C.SHIFT_AT - 1
    false_alarm = bool(alarm[:shift_idx].any())
    after = np.flatnonzero(alarm[shift_idx:])
    if len(after):
        return int(after[0]), False, false_alarm
    return C.N_BATCHES - shift_idx, True, false_alarm      # censored: never alarmed


def run(recall_samples, runs, seed=SEED):
    rng = np.random.default_rng(seed)
    rows = []
    for strategy in ("ai", "manual_full", "manual_sample"):
        for i in range(runs):
            r = float(rng.choice(recall_samples))
            d, cens, fa = simulate_run(rng, strategy, r)
            rows.append({"strategy": strategy, "run": i, "delay_batches": d,
                         "delay_hours": d * C.BATCH_HOURS, "censored": cens, "false_alarm_pre_shift": fa})
    return pd.DataFrame(rows)


def summarize(df):
    out = {}
    for s, g in df.groupby("strategy"):
        out[s] = {"mean_delay_batches": float(g.delay_batches.mean()),
                  "median_delay_batches": float(g.delay_batches.median()),
                  "p90_delay_batches": float(g.delay_batches.quantile(0.9)),
                  "mean_delay_hours": float(g.delay_hours.mean()),
                  "censored_share": float(g.censored.mean()),
                  "pre_shift_false_alarm_share": float(g.false_alarm_pre_shift.mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3000)
    a = ap.parse_args()
    recall = pd.read_csv(RESULTS / "bootstrap_flag_recall.csv").flag_recall.to_numpy()
    df = run(recall, a.runs)
    df.to_csv(RESULTS / "detection_delay.csv", index=False)
    summ = summarize(df)
    summ["assumptions"] = {"P0": C.P0, "P1": C.P1, "batch_size": C.BATCH_SIZE, "manual_sample": C.MANUAL_SAMPLE,
                           "miss_manual": C.MISS_MANUAL, "false_alarm_ai": C.FALSE_ALARM_AI,
                           "false_alarm_manual": C.FALSE_ALARM_MANUAL, "runs_per_strategy": a.runs}
    (RESULTS / "detection_delay_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
