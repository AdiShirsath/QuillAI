"""
Run this after test_with_datasets.py to generate a visual HTML report.
Usage: python generate_report.py
Output: data/outputs/eval_report.html
"""

import base64
import glob
import os
from datetime import datetime

# ── Paste your results here (or load from a JSON file if you save them) ────────
RESULTS = [
    {
        "name": "breast_cancer — exploratory — medium",
        "findings": 10,
        "figures": 1,
        "corrections": 0,
        "latency": 67.6,
        "confidence": 0.87,
        "status": "success",
    },
    {
        "name": "breast_cancer — predictive — medium",
        "findings": 6,
        "figures": 0,
        "corrections": 1,
        "latency": 122.7,
        "confidence": 0.82,
        "status": "partial",
    },
    {
        "name": "breast_cancer — diagnostic — simple",
        "findings": 10,
        "figures": 0,
        "corrections": 1,
        "latency": 134.0,
        "confidence": 0.85,
        "status": "partial",
    },
    {
        "name": "breast_cancer — diagnostic — medium",
        "findings": 10,
        "figures": 0,
        "corrections": 0,
        "latency": 102.1,
        "confidence": 0.88,
        "status": "success",
    },
    {
        "name": "breast_cancer — predictive — medium (DT)",
        "findings": 5,
        "figures": 0,
        "corrections": 1,
        "latency": 116.5,
        "confidence": 0.88,
        "status": "partial",
    },
    {
        "name": "breast_cancer — exploratory — simple",
        "findings": 10,
        "figures": 1,
        "corrections": 0,
        "latency": 101.3,
        "confidence": 0.81,
        "status": "success",
    },
    {
        "name": "breast_cancer — comparative — medium",
        "findings": 10,
        "figures": 0,
        "corrections": 0,
        "latency": 95.8,
        "confidence": 0.75,
        "status": "success",
    },
    {
        "name": "breast_cancer — prescriptive — advanced",
        "findings": 2,
        "figures": 0,
        "corrections": 1,
        "latency": 87.1,
        "confidence": 0.90,
        "status": "partial",
    },
    {
        "name": "telco_churn — exploratory — medium",
        "findings": 10,
        "figures": 0,
        "corrections": 0,
        "latency": 79.3,
        "confidence": 0.74,
        "status": "success",
    },
    {
        "name": "telco_churn — exploratory — medium (sr)",
        "findings": 10,
        "figures": 1,
        "corrections": 0,
        "latency": 80.2,
        "confidence": 0.77,
        "status": "success",
    },
    {
        "name": "telco_churn — predictive — advanced",
        "findings": 2,
        "figures": 0,
        "corrections": 1,
        "latency": 62.2,
        "confidence": 0.80,
        "status": "partial",
    },
    {
        "name": "telco_churn — diagnostic — medium (DT)",
        "findings": 2,
        "figures": 0,
        "corrections": 1,
        "latency": 54.0,
        "confidence": 0.90,
        "status": "partial",
    },
    {
        "name": "telco_churn — diagnostic — medium (TC)",
        "findings": 10,
        "figures": 0,
        "corrections": 1,
        "latency": 73.5,
        "confidence": 0.78,
        "status": "partial",
    },
    {
        "name": "telco_churn — prescriptive — advanced",
        "findings": 10,
        "figures": 0,
        "corrections": 1,
        "latency": 126.9,
        "confidence": 0.82,
        "status": "partial",
    },
    {
        "name": "telco_churn — comparative — medium",
        "findings": 5,
        "figures": 0,
        "corrections": 0,
        "latency": 47.5,
        "confidence": 0.69,
        "status": "success",
    },
    {
        "name": "telco_churn — predictive — advanced (SA)",
        "findings": 5,
        "figures": 0,
        "corrections": 2,
        "latency": 121.8,
        "confidence": 0.93,
        "status": "partial",
    },
    {
        "name": "tips — exploratory — simple",
        "findings": 10,
        "figures": 0,
        "corrections": 1,
        "latency": 61.6,
        "confidence": 0.85,
        "status": "partial",
    },
    {
        "name": "tips — exploratory — medium",
        "findings": 10,
        "figures": 1,
        "corrections": 0,
        "latency": 43.0,
        "confidence": 0.83,
        "status": "success",
    },
    {
        "name": "tips — predictive — medium (clf)",
        "findings": 2,
        "figures": 0,
        "corrections": 1,
        "latency": 40.5,
        "confidence": 0.90,
        "status": "partial",
    },
    {
        "name": "tips — predictive — medium (reg)",
        "findings": 0,
        "figures": 0,
        "corrections": 1,
        "latency": 37.6,
        "confidence": 0.70,
        "status": "failed",
    },
    {
        "name": "tips — diagnostic — medium",
        "findings": 10,
        "figures": 0,
        "corrections": 0,
        "latency": 41.8,
        "confidence": 0.77,
        "status": "success",
    },
    {
        "name": "tips — diagnostic — simple",
        "findings": 10,
        "figures": 1,
        "corrections": 0,
        "latency": 40.8,
        "confidence": 0.78,
        "status": "success",
    },
    {
        "name": "tips — prescriptive — advanced",
        "findings": 2,
        "figures": 0,
        "corrections": 1,
        "latency": 41.2,
        "confidence": 0.70,
        "status": "partial",
    },
    {
        "name": "tips — comparative — medium",
        "findings": 0,
        "figures": 0,
        "corrections": 1,
        "latency": 47.3,
        "confidence": 0.65,
        "status": "failed",
    },
    {
        "name": "titanic — exploratory — medium",
        "findings": 10,
        "figures": 1,
        "corrections": 1,
        "latency": 84.3,
        "confidence": 0.77,
        "status": "partial",
    },
    {
        "name": "titanic — exploratory — medium (corr)",
        "findings": 1,
        "figures": 0,
        "corrections": 1,
        "latency": 45.4,
        "confidence": 0.90,
        "status": "partial",
    },
    {
        "name": "titanic — predictive — medium (lr)",
        "findings": 8,
        "figures": 0,
        "corrections": 3,
        "latency": 113.9,
        "confidence": 0.90,
        "status": "partial",
    },
    {
        "name": "titanic — predictive — medium (DT)",
        "findings": 0,
        "figures": 0,
        "corrections": 1,
        "latency": 44.6,
        "confidence": 0.60,
        "status": "failed",
    },
    {
        "name": "titanic — diagnostic — medium",
        "findings": 9,
        "figures": 1,
        "corrections": 1,
        "latency": 89.4,
        "confidence": 0.81,
        "status": "partial",
    },
    {
        "name": "titanic — diagnostic — advanced",
        "findings": 3,
        "figures": 0,
        "corrections": 1,
        "latency": 63.7,
        "confidence": 0.82,
        "status": "partial",
    },
    {
        "name": "titanic — prescriptive — advanced",
        "findings": 10,
        "figures": 0,
        "corrections": 1,
        "latency": 92.2,
        "confidence": 0.86,
        "status": "partial",
    },
    {
        "name": "titanic — comparative — medium",
        "findings": 10,
        "figures": 1,
        "corrections": 0,
        "latency": 77.3,
        "confidence": 0.83,
        "status": "success",
    },
]

# ── Aggregate stats ─────────────────────────────────────────────────────────────
total = len(RESULTS)
success = sum(1 for r in RESULTS if r["status"] == "success")
partial = sum(1 for r in RESULTS if r["status"] == "partial")
failed = sum(1 for r in RESULTS if r["status"] == "failed")
avg_conf = sum(r["confidence"] for r in RESULTS) / total
avg_latency = sum(r["latency"] for r in RESULTS) / total
total_figs = sum(r["figures"] for r in RESULTS)
total_finds = sum(r["findings"] for r in RESULTS)
self_correct = sum(r["corrections"] for r in RESULTS)


# ── Load saved figures ──────────────────────────────────────────────────────────
def load_figures():
    figures = {}
    output_dir = "data/outputs"
    if not os.path.exists(output_dir):
        return figures
    for folder in os.listdir(output_dir):
        folder_path = os.path.join(output_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        imgs = glob.glob(os.path.join(folder_path, "*.png"))
        if imgs:
            figures[folder] = []
            for img_path in sorted(imgs):
                with open(img_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                figures[folder].append(b64)
    return figures


figures = load_figures()

# ── Status styling ──────────────────────────────────────────────────────────────
STATUS_STYLE = {
    "success": ("✅", "#22c55e", "#dcfce7"),
    "partial": ("⚠️", "#f59e0b", "#fef3c7"),
    "failed": ("❌", "#ef4444", "#fee2e2"),
}


def confidence_bar(conf):
    pct = int(conf * 100)
    color = "#22c55e" if pct >= 80 else "#f59e0b" if pct >= 70 else "#ef4444"
    return f"""
    <div style="display:flex;align-items:center;gap:8px;">
      <div style="flex:1;background:#e5e7eb;border-radius:4px;height:8px;">
        <div style="width:{pct}%;background:{color};height:8px;border-radius:4px;"></div>
      </div>
      <span style="font-size:12px;font-weight:600;color:{color};min-width:36px;">{pct}%</span>
    </div>"""


def result_rows():
    rows = []
    for r in RESULTS:
        icon, color, bg = STATUS_STYLE[r["status"]]
        dataset = r["name"].split(" — ")[0]
        label = " — ".join(r["name"].split(" — ")[1:])
        rows.append(
            f"""
        <tr style="border-bottom:1px solid #f3f4f6;">
          <td style="padding:10px 12px;font-weight:600;color:#1e293b;font-size:13px;">{dataset}</td>
          <td style="padding:10px 12px;color:#64748b;font-size:13px;">{label}</td>
          <td style="padding:10px 12px;text-align:center;">
            <span style="background:{bg};color:{color};padding:2px 10px;border-radius:12px;font-size:12px;font-weight:600;">{icon} {r['status']}</span>
          </td>
          <td style="padding:10px 12px;">{confidence_bar(r['confidence'])}</td>
          <td style="padding:10px 12px;text-align:center;color:#1e293b;font-size:13px;">{r['findings']}</td>
          <td style="padding:10px 12px;text-align:center;color:#1e293b;font-size:13px;">{r['figures']}</td>
          <td style="padding:10px 12px;text-align:center;color:#64748b;font-size:13px;">{r['corrections']}</td>
          <td style="padding:10px 12px;text-align:center;color:#64748b;font-size:13px;">{r['latency']:.1f}s</td>
        </tr>"""
        )
    return "\n".join(rows)


def figure_gallery():
    if not figures:
        return "<p style='color:#94a3b8;'>No figures found in data/outputs/</p>"
    items = []
    for folder, imgs in sorted(figures.items()):
        for i, b64 in enumerate(imgs):
            label = folder.replace("_", " ")
            items.append(
                f"""
            <div style="background:#fff;border-radius:12px;box-shadow:0 1px 4px rgba(0,0,0,0.08);overflow:hidden;">
              <img src="data:image/png;base64,{b64}" style="width:100%;display:block;" />
              <div style="padding:8px 12px;font-size:11px;color:#64748b;border-top:1px solid #f1f5f9;">
                {label} · figure {i+1}
              </div>
            </div>"""
            )
    return "\n".join(items)


# ── Build HTML ──────────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>QuillAI — Evaluation Report</title>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; background:#f8fafc; color:#1e293b; }}
  .hero {{ background:linear-gradient(135deg,#1e293b 0%,#0f172a 100%); color:#fff; padding:48px 40px 40px; }}
  .hero h1 {{ font-size:28px; font-weight:700; letter-spacing:-0.5px; }}
  .hero p  {{ color:#94a3b8; margin-top:6px; font-size:14px; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:16px; padding:32px 40px 0; }}
  .card {{ background:#fff; border-radius:12px; padding:20px; box-shadow:0 1px 3px rgba(0,0,0,0.07); }}
  .card .val {{ font-size:32px; font-weight:700; color:#1e293b; line-height:1; }}
  .card .lbl {{ font-size:12px; color:#94a3b8; margin-top:6px; text-transform:uppercase; letter-spacing:.5px; }}
  .section {{ padding:32px 40px; }}
  .section h2 {{ font-size:16px; font-weight:700; color:#1e293b; margin-bottom:16px; padding-bottom:10px; border-bottom:2px solid #e2e8f0; }}
  table {{ width:100%; border-collapse:collapse; background:#fff; border-radius:12px; overflow:hidden; box-shadow:0 1px 3px rgba(0,0,0,0.07); }}
  thead tr {{ background:#f8fafc; }}
  th {{ padding:10px 12px; text-align:left; font-size:11px; color:#94a3b8; text-transform:uppercase; letter-spacing:.5px; font-weight:600; }}
  tr:hover {{ background:#fafafa; }}
  .gallery {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:20px; }}
  .badge {{ display:inline-flex; gap:8px; }}
  .pill {{ background:#f1f5f9; color:#475569; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:600; }}
  .pill.green {{ background:#dcfce7; color:#16a34a; }}
  .pill.yellow {{ background:#fef9c3; color:#ca8a04; }}
  .pill.red {{ background:#fee2e2; color:#dc2626; }}
</style>
</head>
<body>

<div class="hero">
  <h1>🤖 QuillAI — Autonomous Agent Evaluation</h1>
  <p>Generated {datetime.now().strftime("%B %d, %Y at %H:%M")} · 4 datasets · {total} tasks · Plan-Execute-Observe-Adapt loop</p>
</div>

<div class="cards">
  <div class="card"><div class="val">{total}</div><div class="lbl">Total Tasks</div></div>
  <div class="card"><div class="val" style="color:#22c55e">{success}</div><div class="lbl">Fully Complete</div></div>
  <div class="card"><div class="val" style="color:#f59e0b">{partial}</div><div class="lbl">Partial (replanned)</div></div>
  <div class="card"><div class="val" style="color:#ef4444">{failed}</div><div class="lbl">Failed</div></div>
  <div class="card"><div class="val">{avg_conf:.0%}</div><div class="lbl">Avg Confidence</div></div>
  <div class="card"><div class="val">{avg_latency:.0f}s</div><div class="lbl">Avg Latency</div></div>
  <div class="card"><div class="val">{self_correct}</div><div class="lbl">Self-Corrections</div></div>
  <div class="card"><div class="val">{total_figs}</div><div class="lbl">Figures Generated</div></div>
  <div class="card"><div class="val">{total_finds}</div><div class="lbl">Total Findings</div></div>
</div>

<div class="section">
  <h2>📋 Task Results</h2>
  <table>
    <thead>
      <tr>
        <th>Dataset</th><th>Task</th><th>Status</th><th>Confidence</th>
        <th>Findings</th><th>Figures</th><th>Corrections</th><th>Latency</th>
      </tr>
    </thead>
    <tbody>
      {result_rows()}
    </tbody>
  </table>
</div>

<div class="section">
  <h2>📊 Generated Figures</h2>
  <div class="gallery">
    {figure_gallery()}
  </div>
</div>

</body>
</html>"""

os.makedirs("data/outputs", exist_ok=True)
out_path = "data/outputs/eval_report.html"
with open(out_path, "w") as f:
    f.write(html)

print(f"✅ Report saved to {out_path}")
print(f"   Open in browser: open {out_path}")
