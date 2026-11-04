"""Streamlit dashboard: AI-assisted visual inspection with SPC monitoring.

Run:  streamlit run app/dashboard.py
Reads only files in results/ (produced by `make all`). Detections are REAL model outputs on NEU-DET
test images; the production stream, defect rates and all money figures are SIMULATED / MODELLED.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spc import p_chart  # noqa: E402
import config as C  # noqa: E402

RES = ROOT / "results"
BLUE, ORANGE, GREY = "#2a6fdb", "#d9822b", "#6b7280"

st.set_page_config(page_title="Visual inspection monitor", layout="wide")
st.title("AI-assisted visual inspection: SPC monitor")
st.caption("Detections come from a YOLO model run on real NEU-DET **test** images. "
           "The production stream, defect rates, false-alarm rate and all EUR figures are **simulated / modelled**.")


def need(*names):
    missing = [n for n in names if not (RES / n).exists()]
    if missing:
        st.warning(f"Missing results: {', '.join(missing)}. Run `make all` first.")
        return False
    return True


tab1, tab2, tab3 = st.tabs(["Process monitor (simulated stream)", "Model performance (measured)", "Impact model (modelled)"])

with tab1:
    if need("stream.csv"):
        df = pd.read_csv(RES / "stream.csv")
        df["flagged"] = df.detected_class != "none"
        k = st.sidebar.slider("Baseline batches (set the control limits)", 10, 30, C.BASELINE_BATCHES)
        g = df.groupby("batch")
        chart = p_chart(g.flagged.sum().to_numpy(), g.size().to_numpy(), k)
        batches = g.size().index.to_numpy()
        alarm_batches = batches[chart["alarm"]]
        first = int(alarm_batches.min()) if len(alarm_batches) else None

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Parts inspected", f"{len(df):,}")
        c2.metric("Parts flagged", f"{int(df.flagged.sum()):,}")
        escaped = int(((df.truth_defective == 1) & (~df.flagged)).sum())
        c3.metric("Escaped defects (simulation truth)", escaped)
        c4.metric("First alarm at batch", first if first else "none",
                  delta=(f"{first - C.SHIFT_AT:+d} vs true shift at {C.SHIFT_AT}" if first else None), delta_color="off")

        left, right = st.columns(2)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(batches, chart["p"], marker="o", ms=3, color=BLUE, label="fraction flagged")
        ax.axhline(chart["pbar"], color=GREY, lw=1, label="centre line")
        ax.plot(batches, chart["ucl"], "--", color=ORANGE, lw=1, label="UCL")
        ax.plot(batches, chart["lcl"], "--", color=ORANGE, lw=1)
        ax.scatter(batches[chart["alarm"]], chart["p"][chart["alarm"]], color="red", zorder=3, s=22, label="alarm")
        ax.axvline(k + 0.5, color=GREY, ls=":", lw=1)
        ax.axvline(C.SHIFT_AT - 0.5, color="black", ls=":", lw=1)
        ax.text(C.SHIFT_AT - 0.3, ax.get_ylim()[1] * 0.95, "true shift", fontsize=8, va="top")
        ax.set_xlabel("Batch"); ax.set_ylabel("Fraction flagged"); ax.legend(fontsize=8, frameon=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        left.subheader("p-chart"); left.pyplot(fig); plt.close(fig)

        cnt = df[df.flagged].detected_class.value_counts()
        cum = cnt.cumsum() / cnt.sum() * 100
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.bar(cnt.index, cnt.values, color=BLUE)
        ax3 = ax2.twinx()
        ax3.plot(cnt.index, cum.values, color=ORANGE, marker="o", ms=4)
        ax3.set_ylim(0, 105); ax3.set_ylabel("Cumulative %")
        ax2.set_ylabel("Flagged parts"); plt.setp(ax2.get_xticklabels(), rotation=30, ha="right")
        for a_ in (ax2, ax3):
            a_.spines["top"].set_visible(False)
        right.subheader("Pareto of detected defect types"); right.pyplot(fig2); plt.close(fig2)
        st.caption("Rules: one point above UCL, or eight in a row above the centre line. "
                   "Limits come from the baseline batches (Phase I).")

with tab2:
    if need("test_metrics.json"):
        m = json.loads((RES / "test_metrics.json").read_text())
        base = json.loads((RES / "baseline_test.json").read_text()) if (RES / "baseline_test.json").exists() else None
        c = st.columns(5)
        c[0].metric("mAP50", f"{m['mAP50']:.3f}")
        c[1].metric("Box recall", f"{m['recall']:.3f}")
        c[2].metric("Box precision", f"{m['precision']:.3f}")
        lo, hi = m["flag_recall_ci95"]
        c[3].metric("Part-level flag recall", f"{m['flag_recall']:.3f}", help=f"95% bootstrap CI {lo:.3f} to {hi:.3f}")
        c[4].metric("Threshold (chosen on val)", f"{m['thr']:.2f}")
        st.caption(f"{m['n_test_images']} held-out test images. Threshold rule: {m['threshold_rule']}. "
                   "NEU-DET has no defect-free images, so false alarms on good parts cannot be measured.")
        l, r = st.columns(2)
        l.subheader("Per-class box recall (YOLO)")
        l.bar_chart(pd.Series(m["per_class_recall"], name="recall"))
        if base:
            r.subheader("Image-level class accuracy")
            r.dataframe(pd.DataFrame({"model": ["SVM (LBP+HOG) baseline", "YOLO top-1 class at threshold"],
                                      "accuracy": [base["test_accuracy"], m["top1_class_accuracy"]]}), hide_index=True)
        if (RES / "yolo_test_sweep.csv").exists():
            sw = pd.read_csv(RES / "yolo_test_sweep.csv")
            st.subheader("Precision / recall vs confidence threshold (test)")
            st.line_chart(sw.set_index("thr")[["precision", "recall", "flag_recall"]])
        st.caption("The threshold is frozen from val; the test sweep is shown for transparency, not for tuning.")

with tab3:
    if need("impact_summary.csv"):
        st.warning("Every number here is MODELLED from the assumptions in src/config.py. "
                   "Only AI part-level recall is measured (on clean public images).")
        s = pd.read_csv(RES / "impact_summary.csv")
        t = s[s.metric == "total_saving_per_year_eur"][["scenario", "p5", "median", "p95", "share_positive"]]
        st.dataframe(t.round(2), hide_index=True)
        if (RES / "impact_breakeven.csv").exists():
            st.subheader("Break-even AI recall (central case)")
            st.dataframe(pd.read_csv(RES / "impact_breakeven.csv"), hide_index=True)
        for img in ("impact_saving_hist.png", "impact_sensitivity.png"):
            if (RES / "figures" / img).exists():
                st.image(str(RES / "figures" / img))
