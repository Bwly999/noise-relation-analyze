from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from noise_relation_analyze.scoring import load_model
from noise_relation_analyze.severity import load_single_type_model


def render_noise_report_html(
    report_json: Path,
    scores_csv: Path,
    phone_features_csv: Path,
    model_path: Path,
    output_dir: Path,
    highlight_feature: str,
    highlight_factor: str,
) -> Path:
    import shap

    report = json.loads(report_json.read_text(encoding="utf-8"))
    scores = _read_csv_rows(scores_csv)
    feature_rows = _read_csv_rows(phone_features_csv)
    model = load_model(model_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    x = np.asarray(
        [[float(row[key]) for key in model.feature_keys] for row in feature_rows],
        dtype=float,
    )
    explainer = shap.TreeExplainer(model.estimator)
    explanation = explainer(x)
    target_label = report["noise_type"]
    target_index = list(model.estimator.classes_).index(target_label)
    target_explanation = _select_target_explanation(explanation, x, model.feature_keys, target_index)

    _save_shap_bar(target_explanation, assets_dir / "shap_bar.png")
    _save_shap_beeswarm(target_explanation, assets_dir / "shap_beeswarm.png")
    _save_shap_dependence(
        target_explanation,
        highlight_feature,
        assets_dir / f"shap_dependence_{highlight_feature}.png",
    )
    _save_factor_ranking(report, assets_dir / "factor_ranking.png")
    _save_factor_trend(report, highlight_factor, assets_dir / f"factor_trend_{highlight_factor}.png")
    _save_risk_distribution(scores, target_label, assets_dir / "risk_distribution.png")

    html_path = output_dir / "report.html"
    html_path.write_text(
        _build_html(
            report=report,
            target_label=target_label,
            highlight_feature=highlight_feature,
            highlight_factor=highlight_factor,
        ),
        encoding="utf-8",
    )
    return html_path


def render_single_type_report_html(
    report_json: Path,
    input_csv: Path,
    model_path: Path,
    output_dir: Path,
    highlight_factor: str,
) -> Path:
    import shap

    report = json.loads(report_json.read_text(encoding="utf-8"))
    rows = _read_csv_rows(input_csv)
    model = load_single_type_model(model_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    x = np.asarray(
        [[float(row[key]) for key in model.factor_keys] for row in rows],
        dtype=float,
    )
    severity_explainer = shap.TreeExplainer(model.severity_estimator)
    severity_explanation = _normalize_explanation(severity_explainer(x), x, model.factor_keys)
    ng_explainer = shap.TreeExplainer(model.ng_estimator)
    ng_explanation = _normalize_explanation(ng_explainer(x), x, model.factor_keys)

    _save_shap_bar(severity_explanation, assets_dir / "severity_shap_bar.png")
    _save_shap_beeswarm(severity_explanation, assets_dir / "severity_shap_beeswarm.png")
    _save_shap_dependence(
        severity_explanation,
        highlight_factor,
        assets_dir / f"severity_shap_dependence_{highlight_factor}.png",
    )
    _save_shap_bar(ng_explanation, assets_dir / "ng_shap_bar.png")
    _save_shap_beeswarm(ng_explanation, assets_dir / "ng_shap_beeswarm.png")
    _save_shap_dependence(
        ng_explanation,
        highlight_factor,
        assets_dir / f"ng_shap_dependence_{highlight_factor}.png",
    )
    _save_histogram(
        values=[float(row["severity_score"]) for row in rows],
        title="Acoustic Severity Distribution",
        xlabel="Severity Score",
        output_path=assets_dir / "severity_distribution.png",
        color="#ca3c25",
    )
    _save_histogram(
        values=[float(row["is_ng"]) for row in rows],
        title="NG Label Distribution",
        xlabel="NG Label",
        output_path=assets_dir / "ng_distribution.png",
        color="#1d4c7d",
    )

    html_path = output_dir / "report.html"
    html_path.write_text(
        _build_single_type_html(
            report=report,
            highlight_factor=highlight_factor,
        ),
        encoding="utf-8",
    )
    return html_path


def _save_shap_bar(explanation, output_path: Path) -> None:
    import shap

    plt.figure(figsize=(9, 5.5))
    shap.plots.bar(explanation, max_display=10, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close("all")


def _save_shap_beeswarm(explanation, output_path: Path) -> None:
    import shap

    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(explanation, max_display=10, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close("all")


def _save_shap_dependence(explanation, feature_key: str, output_path: Path) -> None:
    import shap

    plt.figure(figsize=(8, 5.5))
    shap.plots.scatter(explanation[:, feature_key], show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close("all")


def _save_factor_ranking(report: dict, output_path: Path) -> None:
    top_factors = report["top_factors"]
    factor_keys = [item["factor_key"] for item in top_factors]
    values = [float(item["mean_abs_shap"]) for item in top_factors]
    colors = ["#ca3c25" if item["direction"] == "positive" else "#315c8f" for item in top_factors]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.barh(factor_keys[::-1], values[::-1], color=colors[::-1], edgecolor="#111111", linewidth=1.2)
    ax.set_title("Dimension Factor Impact Ranking", fontsize=14, pad=14)
    ax.set_xlabel("Mean Absolute Contribution")
    ax.grid(axis="x", alpha=0.2, linestyle="--")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_factor_trend(report: dict, factor_key: str, output_path: Path) -> None:
    factor = next(item for item in report["top_factors"] if item["factor_key"] == factor_key)
    bin_means = [float(value) for value in factor["bin_means"]]
    x_values = list(range(1, len(bin_means) + 1))

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(x_values, bin_means, marker="o", color="#ca3c25", linewidth=2.2)
    ax.fill_between(x_values, bin_means, color="#f3b2a8", alpha=0.35)
    ax.set_title(f"Factor Impact Curve: {factor_key}", fontsize=14, pad=14)
    ax.set_xlabel("Factor Bin")
    ax.set_ylabel("Mean Risk Score")
    ax.set_ylim(0, max(1.0, max(bin_means) * 1.08))
    ax.grid(alpha=0.18, linestyle="--")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_risk_distribution(scores: list[dict[str, str]], target_label: str, output_path: Path) -> None:
    values = [float(row[f"score_{target_label}"]) for row in scores if f"score_{target_label}" in row]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.hist(values, bins=12, color="#1d4c7d", alpha=0.88, edgecolor="#f5f0e8", linewidth=1.0)
    ax.set_title(f"Risk Distribution: {target_label}", fontsize=14, pad=14)
    ax.set_xlabel("Predicted Risk Score")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.18, linestyle="--")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_histogram(
    values: list[float],
    title: str,
    xlabel: str,
    output_path: Path,
    color: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.hist(values, bins=12, color=color, alpha=0.88, edgecolor="#f5f0e8", linewidth=1.0)
    ax.set_title(title, fontsize=14, pad=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.18, linestyle="--")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _select_target_explanation(explanation, x: np.ndarray, feature_keys: list[str], target_index: int):
    import shap

    values = np.asarray(explanation.values)
    if values.ndim == 3:
        target_values = values[:, :, target_index]
    else:
        target_values = values

    base_values = np.asarray(explanation.base_values)
    if base_values.ndim == 2:
        target_base_values = base_values[:, target_index]
    else:
        target_base_values = base_values

    return shap.Explanation(
        values=target_values,
        base_values=target_base_values,
        data=x,
        feature_names=feature_keys,
    )


def _normalize_explanation(explanation, x: np.ndarray, feature_keys: list[str]):
    import shap

    values = np.asarray(explanation.values, dtype=float)
    if values.ndim == 3:
        values = values[:, :, 0]

    base_values = np.asarray(explanation.base_values, dtype=float)
    if base_values.ndim == 2:
        base_values = base_values[:, 0]

    return shap.Explanation(
        values=values,
        base_values=base_values,
        data=x,
        feature_names=feature_keys,
    )


def _build_html(report: dict, target_label: str, highlight_feature: str, highlight_factor: str) -> str:
    model_summary = report.get("model_summary", {})
    top_factors_rows = "\n".join(
        f"""
        <tr>
          <td>{item['factor_key']}</td>
          <td>{item['direction']}</td>
          <td>{item['spearman_strength']}</td>
          <td>{item['bootstrap_stability']}</td>
          <td>{item['mean_abs_shap']}</td>
        </tr>
        """
        for item in report["top_factors"]
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Noise Analysis Report</title>
  <style>
    :root {{
      --bg: #f3efe6;
      --panel: #fffaf2;
      --ink: #151515;
      --muted: #6f6b66;
      --accent: #ca3c25;
      --accent-2: #1d4c7d;
      --line: #d9cfbf;
      --shadow: 0 14px 40px rgba(38, 26, 10, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(202,60,37,0.10), transparent 28%),
        radial-gradient(circle at top right, rgba(29,76,125,0.12), transparent 26%),
        linear-gradient(180deg, #f6f2ea 0%, var(--bg) 100%);
    }}
    .shell {{
      width: min(1180px, calc(100vw - 48px));
      margin: 0 auto;
      padding: 40px 0 72px;
    }}
    .hero {{
      display: grid;
      grid-template-columns: 1.3fr 1fr;
      gap: 22px;
      align-items: stretch;
      margin-bottom: 24px;
    }}
    .hero-card, .panel {{
      background: rgba(255,250,242,0.92);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
      border-radius: 22px;
      overflow: hidden;
    }}
    .hero-card {{
      padding: 28px;
      min-height: 220px;
      position: relative;
    }}
    .eyebrow {{
      font-size: 12px;
      letter-spacing: 0.24em;
      text-transform: uppercase;
      color: var(--accent);
      margin-bottom: 14px;
    }}
    h1 {{
      font-size: clamp(34px, 5vw, 60px);
      line-height: 0.95;
      margin: 0 0 18px;
      max-width: 10ch;
    }}
    .hero-copy {{
      max-width: 56ch;
      color: var(--muted);
      font-size: 16px;
      line-height: 1.6;
    }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 14px;
      padding: 18px;
    }}
    .kpi {{
      background: linear-gradient(180deg, rgba(255,255,255,0.65), rgba(250,242,229,0.96));
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
    }}
    .kpi-label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      margin-bottom: 10px;
    }}
    .kpi-value {{
      font-size: 30px;
      font-weight: bold;
    }}
    .section {{
      margin-top: 26px;
      display: grid;
      gap: 18px;
    }}
    .section-title {{
      font-size: 20px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--accent-2);
      margin: 0 0 6px;
    }}
    .chart-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }}
    .panel {{
      padding: 18px;
    }}
    .panel h2 {{
      margin: 0 0 12px;
      font-size: 18px;
    }}
    .panel p {{
      margin: 0 0 12px;
      color: var(--muted);
      line-height: 1.5;
    }}
    img {{
      width: 100%;
      display: block;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: #fff;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      text-align: left;
      padding: 12px 10px;
      border-bottom: 1px solid var(--line);
    }}
    th {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }}
    .footnote {{
      margin-top: 24px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
    }}
    @media (max-width: 900px) {{
      .hero, .chart-grid {{
        grid-template-columns: 1fr;
      }}
      .shell {{
        width: min(100vw - 24px, 1180px);
        padding-top: 24px;
      }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <article class="hero-card">
        <div class="eyebrow">Noise Analysis Report</div>
        <h1>{target_label} Risk</h1>
        <p class="hero-copy">
          This report combines model reliability, factor ranking, SHAP contribution ranking, and factor impact curves.
          The current highlight feature is <strong>{highlight_feature}</strong>, and the highlighted structural factor is
          <strong>{highlight_factor}</strong>.
        </p>
      </article>
      <aside class="hero-card">
        <div class="kpi-grid">
          <div class="kpi">
            <div class="kpi-label">Sample Count</div>
            <div class="kpi-value">{report.get('sample_count', '-')}</div>
          </div>
          <div class="kpi">
            <div class="kpi-label">Positive Labels</div>
            <div class="kpi-value">{report.get('positive_label_count', '-')}</div>
          </div>
          <div class="kpi">
            <div class="kpi-label">CV Accuracy</div>
            <div class="kpi-value">{model_summary.get('cv_accuracy', '-')}</div>
          </div>
          <div class="kpi">
            <div class="kpi-label">CV Macro F1</div>
            <div class="kpi-value">{model_summary.get('cv_macro_f1', '-')}</div>
          </div>
        </div>
      </aside>
    </section>

    <section class="section">
      <h2 class="section-title">Model View</h2>
      <div class="chart-grid">
        <article class="panel">
          <h2>SHAP Contribution Ranking</h2>
          <p>Global acoustic-feature impact from the trained tree model.</p>
          <img src="assets/shap_bar.png" alt="SHAP bar ranking" />
        </article>
        <article class="panel">
          <h2>SHAP Beeswarm</h2>
          <p>Feature contribution spread across the full sample set.</p>
          <img src="assets/shap_beeswarm.png" alt="SHAP beeswarm" />
        </article>
        <article class="panel">
          <h2>Feature Impact Curve</h2>
          <p>SHAP dependence for the highlighted feature <strong>{highlight_feature}</strong>.</p>
          <img src="assets/shap_dependence_{highlight_feature}.png" alt="SHAP dependence plot" />
        </article>
        <article class="panel">
          <h2>Risk Distribution</h2>
          <p>Predicted risk-score distribution for the current noise target.</p>
          <img src="assets/risk_distribution.png" alt="Risk distribution" />
        </article>
      </div>
    </section>

    <section class="section">
      <h2 class="section-title">Factor View</h2>
      <div class="chart-grid">
        <article class="panel">
          <h2>Factor Impact Ranking</h2>
          <p>Ranking of structural dimensions by mean absolute contribution.</p>
          <img src="assets/factor_ranking.png" alt="Factor ranking" />
        </article>
        <article class="panel">
          <h2>Factor Impact Curve</h2>
          <p>Binned mean risk trend for <strong>{highlight_factor}</strong>.</p>
          <img src="assets/factor_trend_{highlight_factor}.png" alt="Factor trend" />
        </article>
      </div>
    </section>

    <section class="section">
      <article class="panel">
        <h2>Top Factors</h2>
        <table>
          <thead>
            <tr>
              <th>Factor</th>
              <th>Direction</th>
              <th>Spearman</th>
              <th>Bootstrap</th>
              <th>Mean Abs SHAP</th>
            </tr>
          </thead>
          <tbody>
            {top_factors_rows}
          </tbody>
        </table>
      </article>
    </section>

    <p class="footnote">
      Generated from the local analysis pipeline. SHAP figures use SHAP's built-in plotting functions; factor ranking and trend charts are exported as static PNG assets for offline review.
    </p>
  </main>
</body>
</html>
"""


def _build_single_type_html(report: dict, highlight_factor: str) -> str:
    severity_factors_rows = "\n".join(
        f"""
        <tr>
          <td>{item['factor_key']}</td>
          <td>{item['direction']}</td>
          <td>{item['spearman_strength']}</td>
          <td>{item['bootstrap_stability']}</td>
          <td>{item['mean_abs_shap']}</td>
        </tr>
        """
        for item in report["severity_top_factors"]
    )
    ng_factors_rows = "\n".join(
        f"""
        <tr>
          <td>{item['factor_key']}</td>
          <td>{item['direction']}</td>
          <td>{item['spearman_strength']}</td>
          <td>{item['bootstrap_stability']}</td>
          <td>{item['mean_abs_shap']}</td>
        </tr>
        """
        for item in report["ng_top_factors"]
    )

    severity_model = report["severity_model"]
    ng_model = report["ng_model"]
    acoustic = report["acoustic_quantification"]
    truth_alignment = ""
    if "severity_truth_spearman" in acoustic:
        truth_alignment = f"""
          <div class="kpi">
            <div class="kpi-label">Truth Spearman</div>
            <div class="kpi-value">{acoustic['severity_truth_spearman']}</div>
          </div>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Single-Type Severity Report</title>
  <style>
    :root {{
      --bg: #f3efe6;
      --panel: #fffaf2;
      --ink: #151515;
      --muted: #6f6b66;
      --accent: #ca3c25;
      --accent-2: #1d4c7d;
      --line: #d9cfbf;
      --shadow: 0 14px 40px rgba(38, 26, 10, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(202,60,37,0.10), transparent 28%),
        radial-gradient(circle at top right, rgba(29,76,125,0.12), transparent 26%),
        linear-gradient(180deg, #f6f2ea 0%, var(--bg) 100%);
    }}
    .shell {{
      width: min(1220px, calc(100vw - 40px));
      margin: 0 auto;
      padding: 36px 0 72px;
    }}
    .hero {{
      display: grid;
      grid-template-columns: 1.4fr 1fr;
      gap: 22px;
      margin-bottom: 24px;
    }}
    .hero-card, .panel {{
      background: rgba(255,250,242,0.92);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
      border-radius: 22px;
      overflow: hidden;
    }}
    .hero-card {{
      padding: 28px;
    }}
    .eyebrow {{
      font-size: 12px;
      letter-spacing: 0.24em;
      text-transform: uppercase;
      color: var(--accent);
      margin-bottom: 14px;
    }}
    h1 {{
      font-size: clamp(32px, 4.8vw, 56px);
      line-height: 0.96;
      margin: 0 0 18px;
      max-width: 11ch;
    }}
    .hero-copy {{
      max-width: 62ch;
      color: var(--muted);
      font-size: 16px;
      line-height: 1.6;
    }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 14px;
      padding: 18px;
    }}
    .kpi {{
      background: linear-gradient(180deg, rgba(255,255,255,0.65), rgba(250,242,229,0.96));
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
    }}
    .kpi-label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      margin-bottom: 10px;
    }}
    .kpi-value {{
      font-size: 28px;
      font-weight: bold;
    }}
    .section {{
      margin-top: 26px;
      display: grid;
      gap: 18px;
    }}
    .section-title {{
      font-size: 20px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--accent-2);
      margin: 0 0 6px;
    }}
    .chart-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }}
    .panel {{
      padding: 18px;
    }}
    .panel h2 {{
      margin: 0 0 12px;
      font-size: 18px;
    }}
    .panel p {{
      margin: 0 0 12px;
      color: var(--muted);
      line-height: 1.5;
    }}
    img {{
      width: 100%;
      display: block;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: #fff;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      text-align: left;
      padding: 12px 10px;
      border-bottom: 1px solid var(--line);
    }}
    th {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }}
    @media (max-width: 900px) {{
      .hero, .chart-grid {{
        grid-template-columns: 1fr;
      }}
      .shell {{
        width: min(100vw - 24px, 1220px);
        padding-top: 24px;
      }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <article class="hero-card">
        <div class="eyebrow">Single Noise Type Severity Analysis</div>
        <h1>{report['noise_type']} Severity</h1>
        <p class="hero-copy">
          This report quantifies severity from same-type audio, then explains how dimensional factors influence severity and NG probability.
          The highlighted structural factor is <strong>{highlight_factor}</strong>.
        </p>
      </article>
      <aside class="hero-card">
        <div class="kpi-grid">
          <div class="kpi">
            <div class="kpi-label">Sample Count</div>
            <div class="kpi-value">{report['sample_count']}</div>
          </div>
          <div class="kpi">
            <div class="kpi-label">NG Count</div>
            <div class="kpi-value">{report['ng_count']}</div>
          </div>
          <div class="kpi">
            <div class="kpi-label">Severity NG AUC</div>
            <div class="kpi-value">{acoustic['severity_ng_auc']}</div>
          </div>
          <div class="kpi">
            <div class="kpi-label">NG Gap</div>
            <div class="kpi-value">{acoustic['severity_ng_gap']}</div>
          </div>
          {truth_alignment}
        </div>
      </aside>
    </section>

    <section class="section">
      <h2 class="section-title">Acoustic Severity</h2>
      <div class="chart-grid">
        <article class="panel">
          <h2>Severity Distribution</h2>
          <p>Audio-derived severity score distribution for this same noise type.</p>
          <img src="assets/severity_distribution.png" alt="Severity distribution" />
        </article>
        <article class="panel">
          <h2>NG Distribution</h2>
          <p>Observed binary NG labels in the current dataset.</p>
          <img src="assets/ng_distribution.png" alt="NG distribution" />
        </article>
      </div>
    </section>

    <section class="section">
      <h2 class="section-title">Severity Model</h2>
      <div class="chart-grid">
        <article class="panel">
          <h2>SHAP Ranking</h2>
          <p>Global dimension contribution ranking for severity prediction.</p>
          <img src="assets/severity_shap_bar.png" alt="Severity SHAP ranking" />
        </article>
        <article class="panel">
          <h2>SHAP Beeswarm</h2>
          <p>Sample-level contribution spread for severity prediction.</p>
          <img src="assets/severity_shap_beeswarm.png" alt="Severity SHAP beeswarm" />
        </article>
        <article class="panel">
          <h2>Factor Impact Curve</h2>
          <p>SHAP dependence curve for <strong>{highlight_factor}</strong> against severity.</p>
          <img src="assets/severity_shap_dependence_{highlight_factor}.png" alt="Severity SHAP dependence" />
        </article>
        <article class="panel">
          <h2>Model Metrics</h2>
          <p>`XGBoost` regression performance on cross-validation.</p>
          <table>
            <tbody>
              <tr><th>CV R2</th><td>{severity_model['cv_r2']}</td></tr>
              <tr><th>CV MAE</th><td>{severity_model['cv_mae']}</td></tr>
              <tr><th>CV Spearman</th><td>{severity_model['cv_spearman']}</td></tr>
            </tbody>
          </table>
        </article>
      </div>
    </section>

    <section class="section">
      <h2 class="section-title">NG Model</h2>
      <div class="chart-grid">
        <article class="panel">
          <h2>SHAP Ranking</h2>
          <p>Global dimension contribution ranking for NG probability.</p>
          <img src="assets/ng_shap_bar.png" alt="NG SHAP ranking" />
        </article>
        <article class="panel">
          <h2>SHAP Beeswarm</h2>
          <p>Sample-level contribution spread for NG probability.</p>
          <img src="assets/ng_shap_beeswarm.png" alt="NG SHAP beeswarm" />
        </article>
        <article class="panel">
          <h2>Factor Impact Curve</h2>
          <p>SHAP dependence curve for <strong>{highlight_factor}</strong> against NG probability.</p>
          <img src="assets/ng_shap_dependence_{highlight_factor}.png" alt="NG SHAP dependence" />
        </article>
        <article class="panel">
          <h2>Model Metrics</h2>
          <p>`XGBoost` binary classification performance on cross-validation.</p>
          <table>
            <tbody>
              <tr><th>CV AUC</th><td>{ng_model['cv_auc']}</td></tr>
              <tr><th>CV F1</th><td>{ng_model['cv_f1']}</td></tr>
              <tr><th>Positive Rate</th><td>{ng_model['positive_rate']}</td></tr>
            </tbody>
          </table>
        </article>
      </div>
    </section>

    <section class="section">
      <div class="chart-grid">
        <article class="panel">
          <h2>Severity Top Factors</h2>
          <table>
            <thead>
              <tr>
                <th>Factor</th>
                <th>Direction</th>
                <th>Spearman</th>
                <th>Bootstrap</th>
                <th>Mean Abs SHAP</th>
              </tr>
            </thead>
            <tbody>
              {severity_factors_rows}
            </tbody>
          </table>
        </article>
        <article class="panel">
          <h2>NG Top Factors</h2>
          <table>
            <thead>
              <tr>
                <th>Factor</th>
                <th>Direction</th>
                <th>Spearman</th>
                <th>Bootstrap</th>
                <th>Mean Abs SHAP</th>
              </tr>
            </thead>
            <tbody>
              {ng_factors_rows}
            </tbody>
          </table>
        </article>
      </div>
    </section>
  </main>
</body>
</html>
"""


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
