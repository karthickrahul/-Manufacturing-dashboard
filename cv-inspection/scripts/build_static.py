"""Build a static, self-contained version of the dashboard (web/index.html) from results/.

Why: Netlify and similar hosts serve static files only, so they cannot run the Streamlit app.
The page is generated from the same saved results the Streamlit app reads, so the numbers match.
No server, no external scripts: charts are inline SVG drawn by a few lines of JavaScript.

Usage: python scripts/build_static.py      (writes web/index.html and web/figures/*.png)
"""
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import config as C  # noqa: E402

RES, WEB = ROOT / "results", ROOT / "web"
FIGURES = ["confusion_matrices.png", "examples.png", "impact_saving_hist.png", "impact_sensitivity.png"]


def nn(v):
    """JSON-safe number: NaN -> None."""
    return None if pd.isna(v) else float(v)


def build_data():
    stream = pd.read_csv(RES / "stream.csv")
    stream["flagged"] = stream.detected_class != "none"
    g = stream.groupby("batch")
    pareto = stream[stream.flagged].detected_class.value_counts()
    m = json.loads((RES / "test_metrics.json").read_text())
    base = json.loads((RES / "baseline_test.json").read_text())
    sweep = pd.read_csv(RES / "yolo_test_sweep.csv")
    sweep.loc[sweep.tp + sweep.fp == 0, "precision"] = float("nan")
    impact = pd.read_csv(RES / "impact_summary.csv")
    total = impact[impact.metric == "total_saving_per_year_eur"]
    breakeven = pd.read_csv(RES / "impact_breakeven.csv").iloc[0]
    return {
        "cfg": {"shiftAt": C.SHIFT_AT, "baseline": C.BASELINE_BATCHES},
        "batches": [{"b": int(b), "n": int(n), "f": int(f)}
                    for b, n, f in zip(g.size().index, g.size().to_numpy(), g.flagged.sum().to_numpy())],
        "totals": {"inspected": int(len(stream)), "flagged": int(stream.flagged.sum()),
                   "escaped": int(((stream.truth_defective == 1) & (~stream.flagged)).sum())},
        "pareto": {k: int(v) for k, v in pareto.items()},
        "model": {"map50": m["mAP50"], "recall": m["recall"], "precision": m["precision"],
                  "flagRecall": m["flag_recall"], "flagCI": m["flag_recall_ci95"], "thr": m["thr"],
                  "n": m["n_test_images"], "rule": m["threshold_rule"],
                  "perClass": m["per_class_recall"], "top1": m["top1_class_accuracy"],
                  "svm": base["test_accuracy"]},
        "sweep": [{"thr": float(r.thr), "precision": nn(r.precision), "recall": float(r.recall),
                   "flag": float(r.flag_recall)} for r in sweep.itertuples()],
        "impact": [{"scenario": r.scenario, "p5": float(r.p5), "median": float(r.median),
                    "p95": float(r.p95), "pos": float(r.share_positive)} for r in total.itertuples()],
        "breakeven": {"ai": float(breakeven.breakeven_ai_recall), "human": float(breakeven.human_recall_assumed),
                      "measured": float(breakeven.measured_recall_median)},
    }


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Visual Inspection Monitor</title>
<meta name="description" content="YOLO steel-defect detection with an SPC monitor and a modelled impact analysis. Detector results are measured on NEU-DET; the production stream and all EUR figures are simulated or assumed.">
<style>
:root{--bg:#f7f8fa;--card:#fff;--ink:#14181f;--mute:#5b6472;--line:#e3e6eb;--blue:#2a6fdb;--orange:#c9731c;--red:#d92d20;--green:#2f9e5b;--warn-bg:#fff4e5;--warn-ink:#7a4a00;--warn-line:#f1c27d}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0f1318;--card:#171c23;--ink:#e8ebf0;--mute:#9aa4b2;--line:#2a313b;--blue:#6ea2ff;--orange:#f0a04b;--red:#ff6b5e;--green:#5fd08c;--warn-bg:#2b2113;--warn-ink:#f3d29b;--warn-line:#5a4521}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:28px 16px 48px}
h1{font-size:clamp(1.5rem,3.2vw,2rem);margin:0 0 6px;letter-spacing:-.01em}
h2{font-size:1.1rem;margin:0 0 10px}
.cap{color:var(--mute);margin:0 0 18px;max-width:80ch}
.cap strong{color:var(--ink)}
.tabs{display:flex;gap:4px;border-bottom:1px solid var(--line);margin-bottom:20px;overflow-x:auto}
.tabs button{background:none;border:0;border-bottom:3px solid transparent;color:var(--mute);font:inherit;padding:10px 14px;cursor:pointer;white-space:nowrap}
.tabs button[aria-selected="true"]{color:var(--ink);border-color:var(--blue);font-weight:600}
.tabs button:focus-visible,input:focus-visible{outline:2px solid var(--blue);outline-offset:2px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-bottom:20px}
.kpi,.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;min-width:0}
.card{overflow-x:auto}
.kpi .l{color:var(--mute);font-size:.85rem}
.kpi .v{font-size:1.7rem;font-weight:650;line-height:1.2;font-variant-numeric:tabular-nums}
.kpi .s{color:var(--mute);font-size:.8rem}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,440px),1fr));gap:16px;margin-bottom:16px}
.ctrl{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:16px}
.ctrl input{width:min(260px,100%)}
svg{width:100%;height:auto;display:block}
svg text{fill:var(--mute);font-size:11px;font-family:inherit}
.ax{stroke:var(--line)}.grd{stroke:var(--line);stroke-dasharray:2 3}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:.82rem;color:var(--mute);margin-top:6px}
.legend i{display:inline-block;width:14px;height:0;border-top:2px solid;margin-right:6px;vertical-align:middle}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:8px 10px;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
th{color:var(--mute);font-weight:600;font-size:.85rem}
.warn{background:var(--warn-bg);color:var(--warn-ink);border:1px solid var(--warn-line);border-radius:10px;padding:12px 14px;margin-bottom:16px}
img{max-width:100%;height:auto;border-radius:8px;border:1px solid var(--line);background:#fff}
.note{color:var(--mute);font-size:.85rem;margin:8px 0 0}
footer{color:var(--mute);font-size:.85rem;margin-top:28px}
footer a{color:var(--blue)}
[hidden]{display:none!important}
</style>
</head>
<body>
<div class="wrap">
<h1>AI-assisted visual inspection: SPC monitor</h1>
<p class="cap">Detections come from a YOLO11n model run on real NEU-DET <strong>test</strong> images. The production stream, defect rates, false-alarm rate and all EUR figures are <strong>simulated / modelled</strong>.</p>

<div class="tabs" role="tablist">
  <button role="tab" id="t1" aria-selected="true" aria-controls="p1">Process monitor (simulated stream)</button>
  <button role="tab" id="t2" aria-selected="false" aria-controls="p2">Model performance (measured)</button>
  <button role="tab" id="t3" aria-selected="false" aria-controls="p3">Impact model (modelled)</button>
</div>

<section id="p1" role="tabpanel" aria-labelledby="t1">
  <div class="ctrl"><label for="k">Baseline batches (set the control limits): <strong id="kv"></strong></label><input id="k" type="range" min="10" max="30" step="1"></div>
  <div class="kpis" id="k1"></div>
  <div class="grid">
    <div class="card"><h2>p-chart</h2><div id="pc"></div>
      <div class="legend"><span><i style="border-color:var(--blue)"></i>fraction flagged</span><span><i style="border-color:var(--mute)"></i>centre line</span><span><i style="border-color:var(--orange);border-top-style:dashed"></i>control limits</span><span><i style="border-color:var(--red)"></i>alarm</span></div></div>
    <div class="card"><h2>Pareto of detected defect types</h2><div id="pa"></div>
      <div class="legend"><span><i style="border-color:var(--blue);border-top-width:6px"></i>flagged parts</span><span><i style="border-color:var(--orange)"></i>cumulative %</span></div></div>
  </div>
  <p class="note">Rules: one point above the upper control limit (UCL), or eight in a row above the centre line. Limits come from the baseline batches (Phase I); only later batches can alarm.</p>
</section>

<section id="p2" role="tabpanel" aria-labelledby="t2" hidden>
  <div class="kpis" id="k2"></div>
  <p class="note" id="m-note" style="margin:0 0 16px"></p>
  <div class="grid">
    <div class="card"><h2>Per-class box recall (YOLO)</h2><div id="pcr"></div></div>
    <div class="card"><h2>Image-level class accuracy</h2><table id="acc"></table>
      <p class="note">YOLO and the SVM are about equal at naming the defect (a 3-image difference on 270 images). YOLO adds localisation.</p></div>
  </div>
  <div class="card" style="margin-bottom:16px"><h2>Precision / recall vs confidence threshold (test)</h2><div id="sw"></div>
    <div class="legend"><span><i style="border-color:var(--green)"></i>part-level flag recall</span><span><i style="border-color:var(--blue)"></i>box recall (IoU 0.5)</span><span><i style="border-color:var(--orange);border-top-style:dashed"></i>box precision</span></div>
    <p class="note">The threshold is frozen from the validation split; this sweep is shown for transparency, not for tuning.</p></div>
  <div class="grid">
    <div class="card"><h2>Confusion matrices (test)</h2><img src="figures/confusion_matrices.png" alt="Confusion matrices for the SVM baseline and the YOLO top-1 box, test split" loading="lazy"></div>
    <div class="card"><h2>Example detections</h2><img src="figures/examples.png" alt="Example detections: green boxes are ground truth, red boxes are model predictions" loading="lazy"><p class="note">Green: ground truth. Red: model detections.</p></div>
  </div>
</section>

<section id="p3" role="tabpanel" aria-labelledby="t3" hidden>
  <div class="warn"><strong>Every number here is modelled</strong> from assumptions (volumes, costs, human miss rate, AI false-alarm rate). Only the AI part-level recall is measured, on clean public images. The modelled saving is driven mostly by the assumed cost of an escaped defect, the assumed defect rate and the assumed human miss rate. It is not a business case.</div>
  <div class="card" style="margin-bottom:16px"><h2>Modelled total saving per year vs 100% manual inspection</h2><table id="imp"></table>
    <p class="note">Expected uses the measured recall. Pessimistic cuts recall by 15 points and raises the assumed false-alarm rate. 20,000 Monte Carlo runs each.</p></div>
  <div class="card" style="margin-bottom:16px"><h2>Break-even AI recall (central case)</h2><table id="be"></table></div>
  <div class="grid">
    <div class="card"><h2>Distribution of the modelled saving</h2><img src="figures/impact_saving_hist.png" alt="Histogram of modelled yearly saving for the expected and pessimistic scenarios" loading="lazy"></div>
    <div class="card"><h2>Which assumptions drive it</h2><img src="figures/impact_sensitivity.png" alt="Rank correlation of each input with the modelled yearly saving" loading="lazy"></div>
  </div>
</section>

<footer>Static build of the Streamlit dashboard in this project. Dataset: NEU surface defect database (Song &amp; Yan, Northeastern University). Source code and full limitations are in the repository README.</footer>
</div>

<script>
const D = __DATA__;
const $ = s => document.querySelector(s);
const NS = "http://www.w3.org/2000/svg";
const f0 = n => n.toLocaleString("en-US");
const pct = x => (x * 100).toFixed(1) + "%";

/*PCHART_START*/
function pchart(flags, sizes, k) {
  const L = flags.length;
  const p = flags.map((f, i) => f / sizes[i]);
  const sumF = flags.slice(0, k).reduce((a, b) => a + b, 0);
  const sumN = sizes.slice(0, k).reduce((a, b) => a + b, 0);
  const pbar = sumF / sumN;
  const sigma = sizes.map(n => Math.sqrt(pbar * (1 - pbar) / n));
  const ucl = sigma.map(s => pbar + 3 * s);
  const lcl = sigma.map(s => Math.max(pbar - 3 * s, 0));
  const above = p.map(v => (v > pbar ? 1 : 0));
  const alarm = p.map((v, i) => {
    let run = 0;
    for (let j = Math.max(0, i - 7); j <= i; j++) run += above[j];
    const run8 = (i >= 7) ? run === 8 : false;     // partial windows (i < 7) never sum to 8
    return (v > ucl[i] || run8) && i >= k;
  });
  return { p, pbar, ucl, lcl, alarm };
}
/*PCHART_END*/

// chart width = its container's width, so text stays at a readable size on phones
function cw(sel) { return Math.max(300, Math.floor($(sel).clientWidth || 640)); }

function svgEl(w, h, label) {
  const s = document.createElementNS(NS, "svg");
  s.setAttribute("viewBox", `0 0 ${w} ${h}`);
  s.setAttribute("role", "img");
  s.setAttribute("aria-label", label);
  return s;
}
function add(parent, tag, attrs, text) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (text !== undefined) e.textContent = text;
  parent.appendChild(e);
  return e;
}
function line(svg, x1, y1, x2, y2, cls, extra) { return add(svg, "line", Object.assign({ x1, y1, x2, y2, class: cls }, extra || {})); }
function path(svg, pts, stroke, extra) {
  const d = pts.map((q, i) => (i ? "L" : "M") + q[0].toFixed(1) + " " + q[1].toFixed(1)).join(" ");
  return add(svg, "path", Object.assign({ d, fill: "none", stroke, "stroke-width": 2, "stroke-linejoin": "round" }, extra || {}));
}
function frame(svg, W, H, m, xs, ys, xticks, yticks, xlab, ylab, yfmt) {
  yticks.forEach(t => {
    line(svg, m.l, ys(t), W - m.r, ys(t), "grd");
    add(svg, "text", { x: m.l - 6, y: ys(t) + 4, "text-anchor": "end" }, yfmt(t));
  });
  line(svg, m.l, H - m.b, W - m.r, H - m.b, "ax");
  xticks.forEach(t => add(svg, "text", { x: xs(t), y: H - m.b + 16, "text-anchor": "middle" }, String(t)));
  if (xlab) add(svg, "text", { x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle" }, xlab);
  if (ylab) add(svg, "text", { x: 12, y: (m.t + H - m.b) / 2, "text-anchor": "middle", transform: `rotate(-90 12 ${(m.t + H - m.b) / 2})` }, ylab);
}

function drawP() {
  const k = +$("#k").value;
  $("#kv").textContent = k;
  const flags = D.batches.map(x => x.f), sizes = D.batches.map(x => x.n), bs = D.batches.map(x => x.b);
  const c = pchart(flags, sizes, k);
  const firstIdx = c.alarm.indexOf(true);
  const first = firstIdx >= 0 ? bs[firstIdx] : null;
  const shift = D.cfg.shiftAt;
  const kpi = (l, v, s) => `<div class="kpi"><div class="l">${l}</div><div class="v">${v}</div>${s ? `<div class="s">${s}</div>` : ""}</div>`;
  $("#k1").innerHTML = kpi("Parts inspected", f0(D.totals.inspected)) + kpi("Parts flagged", f0(D.totals.flagged))
    + kpi("Escaped defects (simulation truth)", D.totals.escaped)
    + kpi("First alarm at batch", first === null ? "none" : first, first === null ? "" : `${first - shift >= 0 ? "+" : ""}${first - shift} vs true shift at ${shift}`);

  const W = cw("#pc"), H = 340, m = { l: 52, r: 16, t: 14, b: 42 };
  const svg = svgEl(W, H, `p-chart: fraction of parts flagged per batch with control limits. First alarm at batch ${first === null ? "none" : first}.`);
  const nb = bs.length, ymax = Math.ceil(Math.max(...c.p, ...c.ucl) * 100 * 1.08 / 2) * 2 / 100;
  const xs = b => m.l + (b - 1) / (nb - 1) * (W - m.l - m.r), ys = v => H - m.b - v / ymax * (H - m.t - m.b);
  const yt = []; for (let v = 0; v <= ymax + 1e-9; v += 0.02) yt.push(+v.toFixed(2));
  frame(svg, W, H, m, xs, ys, [1, 10, 20, 30, 40, 50, 60].filter(t => t <= nb), yt, "Batch", "Fraction flagged", v => v.toFixed(2));
  line(svg, xs(k + 0.5), m.t, xs(k + 0.5), H - m.b, "", { stroke: "var(--mute)", "stroke-dasharray": "2 3" });
  add(svg, "text", { x: xs(k + 0.5) - 4, y: m.t + 10, "text-anchor": "end" }, "baseline");
  line(svg, xs(shift - 0.5), m.t, xs(shift - 0.5), H - m.b, "", { stroke: "var(--ink)", "stroke-dasharray": "2 3" });
  add(svg, "text", { x: xs(shift - 0.5) + 4, y: m.t + 10 }, "true shift");
  path(svg, bs.map((b, i) => [xs(b), ys(c.ucl[i])]), "var(--orange)", { "stroke-width": 1.5, "stroke-dasharray": "5 4" });
  path(svg, bs.map((b, i) => [xs(b), ys(c.lcl[i])]), "var(--orange)", { "stroke-width": 1.5, "stroke-dasharray": "5 4" });
  line(svg, m.l, ys(c.pbar), W - m.r, ys(c.pbar), "", { stroke: "var(--mute)", "stroke-width": 1.5 });
  path(svg, bs.map((b, i) => [xs(b), ys(c.p[i])]), "var(--blue)");
  bs.forEach((b, i) => {
    const dot = add(svg, "circle", { cx: xs(b), cy: ys(c.p[i]), r: c.alarm[i] ? 4.5 : 2.5, fill: c.alarm[i] ? "var(--red)" : "var(--blue)" });
    add(dot, "title", {}, `Batch ${b}: ${pct(c.p[i])} flagged${c.alarm[i] ? " (alarm)" : ""}`);
  });
  $("#pc").replaceChildren(svg);
}

function drawPareto() {
  const rows = Object.entries(D.pareto).sort((a, b) => b[1] - a[1]);
  const tot = rows.reduce((a, r) => a + r[1], 0);
  const W = cw("#pa"), H = 340, m = { l: 52, r: 46, t: 14, b: 70 };
  const svg = svgEl(W, H, "Pareto chart of detected defect types with cumulative percentage");
  const ymax = Math.ceil(rows[0][1] / 50) * 50;
  const bw = (W - m.l - m.r) / rows.length;
  const ys = v => H - m.b - v / ymax * (H - m.t - m.b), y2 = v => H - m.b - v / 100 * (H - m.t - m.b);
  const xs = i => m.l + bw * (i + 0.5);
  const yt = []; for (let v = 0; v <= ymax; v += 50) yt.push(v);
  frame(svg, W, H, m, xs, ys, [], yt, null, "Flagged parts", v => String(v));
  [0, 20, 40, 60, 80, 100].forEach(t => add(svg, "text", { x: W - m.r + 6, y: y2(t) + 4 }, String(t)));
  add(svg, "text", { x: W - 8, y: (m.t + H - m.b) / 2, "text-anchor": "middle", transform: `rotate(90 ${W - 8} ${(m.t + H - m.b) / 2})` }, "Cumulative %");
  let cum = 0; const pts = [];
  rows.forEach((r, i) => {
    cum += r[1];
    const bar = add(svg, "rect", { x: xs(i) - bw * 0.36, y: ys(r[1]), width: bw * 0.72, height: H - m.b - ys(r[1]), fill: "var(--blue)", rx: 2 });
    add(bar, "title", {}, `${r[0]}: ${r[1]} flagged parts (${(cum / tot * 100).toFixed(0)}% cumulative)`);
    add(svg, "text", { x: xs(i), y: H - m.b + 14, "text-anchor": "end", transform: `rotate(-30 ${xs(i)} ${H - m.b + 14})` }, r[0]);
    pts.push([xs(i), y2(cum / tot * 100)]);
  });
  path(svg, pts, "var(--orange)");
  pts.forEach(q => add(svg, "circle", { cx: q[0], cy: q[1], r: 3, fill: "var(--orange)" }));
  $("#pa").replaceChildren(svg);
}

function drawModel() {
  const M = D.model;
  const kpi = (l, v, s) => `<div class="kpi"><div class="l">${l}</div><div class="v">${v}</div>${s ? `<div class="s">${s}</div>` : ""}</div>`;
  $("#k2").innerHTML = kpi("mAP50", M.map50.toFixed(3)) + kpi("Box recall", M.recall.toFixed(3)) + kpi("Box precision", M.precision.toFixed(3))
    + kpi("Part-level flag recall", M.flagRecall.toFixed(3), `95% CI ${M.flagCI[0].toFixed(3)} to ${M.flagCI[1].toFixed(3)}`)
    + kpi("Threshold (chosen on val)", M.thr.toFixed(2));
  $("#m-note").textContent = `${M.n} held-out test images. Threshold rule: ${M.rule}. NEU-DET has no defect-free images, so false alarms on good parts cannot be measured.`;

  const cls = Object.entries(M.perClass);
  const W = cw("#pcr"), H = 250, m = { l: 118, r: 40, t: 8, b: 8 }, rh = (H - m.t - m.b) / cls.length;
  const bar = svgEl(W, H, "Per-class box recall");
  cls.forEach(([n, v], i) => {
    const y = m.t + i * rh;
    add(bar, "text", { x: m.l - 8, y: y + rh / 2 + 4, "text-anchor": "end" }, n);
    add(bar, "rect", { x: m.l, y: y + 5, width: (W - m.l - m.r) * v, height: rh - 10, fill: "var(--blue)", rx: 2 });
    add(bar, "text", { x: m.l + (W - m.l - m.r) * v + 6, y: y + rh / 2 + 4 }, v.toFixed(2));
  });
  $("#pcr").replaceChildren(bar);

  $("#acc").innerHTML = `<tr><th>Model</th><th>Accuracy</th></tr><tr><td>SVM (LBP + HOG) baseline</td><td>${M.svm.toFixed(3)}</td></tr><tr><td>YOLO top-1 class at threshold</td><td>${M.top1.toFixed(3)}</td></tr>`;

  const sw = D.sweep, W2 = cw("#sw"), H2 = 300, m2 = { l: 48, r: 16, t: 12, b: 40 };
  const s2 = svgEl(W2, H2, "Precision and recall versus confidence threshold on the test split");
  const xs = t => m2.l + (t - 0.05) / 0.9 * (W2 - m2.l - m2.r), ys = v => H2 - m2.b - v * (H2 - m2.t - m2.b);
  frame(s2, W2, H2, m2, xs, ys, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [0, 0.2, 0.4, 0.6, 0.8, 1], "Confidence threshold", null, v => v.toFixed(1));
  line(s2, xs(M.thr), m2.t, xs(M.thr), H2 - m2.b, "", { stroke: "var(--mute)", "stroke-dasharray": "3 3" });
  add(s2, "text", { x: xs(M.thr) + 4, y: ys(0.04) }, `threshold ${M.thr.toFixed(2)} (frozen from val)`);
  path(s2, sw.map(r => [xs(r.thr), ys(r.flag)]), "var(--green)");
  path(s2, sw.map(r => [xs(r.thr), ys(r.recall)]), "var(--blue)");
  path(s2, sw.filter(r => r.precision !== null).map(r => [xs(r.thr), ys(r.precision)]), "var(--orange)", { "stroke-dasharray": "6 4" });
  $("#sw").replaceChildren(s2);
}

function drawImpact() {
  const eur = v => (v < 0 ? "-" : "") + "€" + Math.round(Math.abs(v)).toLocaleString("en-US");
  $("#imp").innerHTML = "<tr><th>Scenario</th><th>5th percentile</th><th>Median</th><th>95th percentile</th><th>Runs with a positive saving</th></tr>"
    + D.impact.map(r => `<tr><td>${r.scenario}</td><td>${eur(r.p5)}</td><td>${eur(r.median)}</td><td>${eur(r.p95)}</td><td>${(r.pos * 100).toFixed(1)}%</td></tr>`).join("");
  const B = D.breakeven;
  $("#be").innerHTML = `<tr><th>Break-even AI recall</th><th>Human recall (assumed)</th><th>AI part-level recall (measured, median)</th></tr><tr><td>${B.ai.toFixed(2)}</td><td>${B.human.toFixed(2)}</td><td>${B.measured.toFixed(3)}</td></tr>`;
}

// tabs: draw only the visible panel (hidden panels have no width), and redraw on resize
const tabs = ["t1", "t2", "t3"];
const draws = { t1: [drawP, drawPareto], t2: [drawModel], t3: [drawImpact] };
let current = "t1";
const render = () => draws[current].forEach(fn => fn());
tabs.forEach((id, i) => $("#" + id).addEventListener("click", () => {
  current = id;
  tabs.forEach((t, j) => { $("#" + t).setAttribute("aria-selected", i === j); $("#p" + (j + 1)).hidden = i !== j; });
  render();
}));
const k = $("#k"); k.value = D.cfg.baseline; k.addEventListener("input", drawP);
let raf = 0;
addEventListener("resize", () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(render); });
render();
</script>
</body>
</html>
"""


def main():
    WEB.mkdir(exist_ok=True)
    (WEB / "figures").mkdir(exist_ok=True)
    for f in FIGURES:
        shutil.copy(RES / "figures" / f, WEB / "figures" / f)
    html = TEMPLATE.replace("__DATA__", json.dumps(build_data(), separators=(",", ":")))
    (WEB / "index.html").write_text(html, encoding="utf-8")
    print(f"Wrote {WEB / 'index.html'} ({len(html) / 1024:.0f} KB) and {len(FIGURES)} figures")


if __name__ == "__main__":
    main()
