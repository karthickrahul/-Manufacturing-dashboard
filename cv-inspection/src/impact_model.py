"""Monte Carlo impact model: AI-assisted vs 100% manual visual inspection. ALL savings are MODELLED.

What is measured vs assumed
  measured : AI part-level recall (bootstrap over the NEU-DET test split) - results/bootstrap_flag_recall.csv
  simulated: shift-detection delay of AI vs manual under the stated process assumptions - detection_delay.csv
  assumed  : everything in src/config.py (volumes, costs, human miss rate, AI false-alarm rate, ...)

Baseline: people visually inspect 100% of parts today.
AI-assisted: the model screens 100% of parts; people re-check only flagged parts and are assumed to resolve
each flagged part correctly. A part the model does not flag ships.

Two scenarios
  expected    - measured recall as is
  pessimistic - recall reduced by HAIRCUT (public clean images are easier than a real line) and a higher
                false-alarm rate

Outputs: results/impact_summary.csv, results/impact_breakeven.csv, results/figures/impact_*.png
"""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config as C  # noqa: E402
from common import FIGS, RESULTS, SEED  # noqa: E402

RUNS = 20_000
HAIRCUT = 0.15
PESSIMISTIC_FALSE_ALARM = (0.02, 0.05, 0.10)


def per_day(defect_rate, t_manual, miss, recall, fa, t_review, cost_escape):
    defects = C.PARTS_PER_DAY * defect_rate
    good = C.PARTS_PER_DAY - defects
    hours_m = C.PARTS_PER_DAY * t_manual / 3600
    esc_m = defects * miss
    flagged = defects * recall + good * fa
    hours_a = flagged * t_review / 3600
    esc_a = defects * (1 - recall)
    cost_m = hours_m * C.LABOUR_EUR_H + esc_m * cost_escape
    cost_a = hours_a * C.LABOUR_EUR_H + esc_a * cost_escape + C.SYSTEM_EUR_DAY
    return hours_m - hours_a, esc_m - esc_a, cost_m - cost_a


def simulate(scenario, recall_boot, delays, rng):
    tri = lambda t: rng.triangular(*t, RUNS)  # noqa: E731
    defect_rate, t_manual, miss = tri(C.DEFECT_RATE), tri(C.T_MANUAL_S), tri(C.MISS_MANUAL_RANGE)
    t_review, cost_escape = tri(C.T_REVIEW_S), tri(C.COST_ESCAPE)
    shifts, scrap_part = tri(C.SHIFTS_PER_YEAR), tri(C.SCRAP_PART)
    recall = rng.choice(recall_boot, RUNS)
    fa = rng.triangular(0.005, C.FALSE_ALARM_AI, 0.05, RUNS)
    if scenario == "pessimistic":
        recall = np.clip(recall - HAIRCUT, 0.05, 1.0)
        fa = rng.triangular(*PESSIMISTIC_FALSE_ALARM, RUNS)
    d_ai = rng.choice(delays["ai"], RUNS)
    d_man = rng.choice(delays["manual_full"], RUNS)

    hours_saved, esc_avoided, saving_day = per_day(defect_rate, t_manual, miss, recall, fa, t_review, cost_escape)
    scrap_avoided = shifts * np.clip(d_man - d_ai, 0, None) * C.PARTS_PER_HOUR * (C.P1 - C.P0) * scrap_part
    out = pd.DataFrame({"inspection_hours_saved_per_day": hours_saved,
                        "escaped_defects_avoided_per_day": esc_avoided,
                        "scrap_avoided_per_year_eur": scrap_avoided,
                        "total_saving_per_year_eur": saving_day * C.WORKING_DAYS + scrap_avoided})
    inputs = pd.DataFrame({"defect_rate": defect_rate, "t_manual_s": t_manual, "miss_manual": miss,
                           "recall_ai": recall, "false_alarm_ai": fa, "t_review_s": t_review,
                           "cost_escape_eur": cost_escape, "shifts_per_year": shifts,
                           "scrap_part_eur": scrap_part, "delay_ai_h": d_ai, "delay_manual_h": d_man})
    return out, inputs


def breakeven(delays):
    """Lowest AI recall at which the central-case yearly saving is positive (all other inputs at their mode)."""
    mode = lambda t: t[1]  # noqa: E731
    d_gain = max(np.mean(delays["manual_full"]) - np.mean(delays["ai"]), 0)
    scrap = mode(C.SHIFTS_PER_YEAR) * d_gain * C.PARTS_PER_HOUR * (C.P1 - C.P0) * mode(C.SCRAP_PART)
    for r in np.arange(0.0, 1.0001, 0.005):
        _, _, day = per_day(mode(C.DEFECT_RATE), mode(C.T_MANUAL_S), mode(C.MISS_MANUAL_RANGE), r,
                            C.FALSE_ALARM_AI, mode(C.T_REVIEW_S), mode(C.COST_ESCAPE))
        if day * C.WORKING_DAYS + scrap > 0:
            return float(round(r, 3))
    return None


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    recall_boot = pd.read_csv(RESULTS / "bootstrap_flag_recall.csv").flag_recall.to_numpy()
    dd = pd.read_csv(RESULTS / "detection_delay.csv")
    delays = {s: g.delay_hours.to_numpy() for s, g in dd.groupby("strategy")}

    rows, results, inputs_by = [], {}, {}
    for scenario in ("expected", "pessimistic"):
        out, inp = simulate(scenario, recall_boot, delays, rng)
        results[scenario], inputs_by[scenario] = out, inp
        for m in out.columns:
            rows.append({"scenario": scenario, "metric": m, "p5": np.percentile(out[m], 5),
                         "median": np.median(out[m]), "p95": np.percentile(out[m], 95),
                         "share_positive": float((out[m] > 0).mean())})
    summ = pd.DataFrame(rows)
    summ.to_csv(RESULTS / "impact_summary.csv", index=False)
    be = pd.DataFrame([{"breakeven_ai_recall": breakeven(delays),
                        "human_recall_assumed": 1 - C.MISS_MANUAL,
                        "measured_recall_median": float(np.median(recall_boot))}])
    be.to_csv(RESULTS / "impact_breakeven.csv", index=False)
    print(summ.round(2).to_string(index=False))
    print(be.round(3).to_string(index=False))

    col = {"expected": "#2a6fdb", "pessimistic": "#d9822b"}
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for sc in ("expected", "pessimistic"):
        v = results[sc].total_saving_per_year_eur / 1000
        ax.hist(v, bins=60, alpha=0.6, color=col[sc], label=f"{sc} (median {np.median(v):.0f} k EUR)")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Modelled total saving per year (k EUR)")
    ax.set_ylabel("Monte Carlo runs")
    ax.set_title("Modelled impact vs 100% manual inspection (assumption-based, NOT measured)", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "impact_saving_hist.png", dpi=160)
    plt.close(fig)

    inp, tot = inputs_by["expected"], results["expected"].total_saving_per_year_eur
    sens = inp.corrwith(tot, method="spearman").sort_values(key=abs)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.barh(sens.index, sens.values, color=["#2a6fdb" if v >= 0 else "#d9822b" for v in sens.values])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title("Which assumptions drive the yearly saving (expected scenario)", fontsize=9)
    ax.set_xlabel("Spearman rank correlation with yearly saving")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "impact_sensitivity.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
