"""Local Gradio demo of the calibration layer. Launch: python app.py

Two modes, both reusing the existing src/ functions unchanged:
  A (default, fast) — upload a predictions CSV (K probability columns +
    0-based `label` column); the calibration -> conformal -> selective
    stages are reassembled directly on the uploaded matrix.
  B (slow) — upload an MCQ CSV in the src/custom_csv.py contract; runs the
    full pipeline (scores every question with the configured model) and
    displays the artefacts it writes to outputs/.

No new ML lives here: every number shown comes from src/.
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import yaml

from src import calibration as cal
from src import conformal, selective
from src import data as data_mod
from src.custom_csv import CSVValidationError
from src.pipeline import OUTPUTS, load_config, run as run_pipeline
from src.plots import plot_reliability, plot_risk_coverage
from src.scoring import cache_path, option_letters

MAX_TABLE_ROWS = 1000
DEFER_COLUMNS = ["row", "decision", "prediction", "calibrated confidence",
                 "conformal set", "true label", "correct"]


def _pipeline_alpha(alphas: list[float]) -> float:
    """The alpha the pipeline uses for its ANSWER/DEFER artefacts."""
    return alphas[1] if len(alphas) > 1 else alphas[0]


def _coverage_table(report: dict) -> pd.DataFrame:
    """Flatten conformal.coverage_check output (alpha keys are strings)."""
    rows = [
        {
            "method": method.upper(),
            "alpha": a,
            "target coverage": r["target_coverage"],
            "empirical coverage": r["empirical_coverage"],
            "mean set size": r["mean_set_size"],
        }
        for method, per_alpha in report.items()
        for a, r in per_alpha.items()
    ]
    return pd.DataFrame(rows)


def _defer_table(
    P_raw: np.ndarray,
    P_cal_scaled: np.ndarray,
    sets: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    questions: list[str] | None = None,
) -> pd.DataFrame:
    """The 'knows when to shut up' table: ANSWER on singleton sets, else DEFER."""
    pred = P_raw.argmax(axis=1)
    conf = P_cal_scaled.max(axis=1)
    abstain = conformal.abstain_mask_from_sets(sets)
    rows = []
    for i in range(min(len(y), MAX_TABLE_ROWS)):
        members = ", ".join(class_names[k] for k in np.where(sets[i])[0])
        rows.append({
            "row": questions[i] if questions is not None else i,
            "decision": "DEFER" if abstain[i] else "ANSWER",
            "prediction": class_names[pred[i]],
            "calibrated confidence": round(float(conf[i]), 3),
            "conformal set": "{" + members + "}",
            "true label": class_names[int(y[i])],
            "correct": "✓" if pred[i] == y[i] else "✗",
        })
    return pd.DataFrame(rows, columns=DEFER_COLUMNS if questions is None else
                        ["row", *DEFER_COLUMNS[1:]])


def _summary_md(n_cal: int, n_test: int, T: float, ece_raw: float, ece_cal: float,
                aurc_raw: float, aurc_cal: float, head: dict, alpha: float,
                n_defer: int, n_total: int, warnings: list[str]) -> str:
    md = [
        f"### {head['sentence']}",
        f"- calibration/evaluation split: **{n_cal} / {n_test}** rows",
        f"- fitted temperature **T = {T:.3f}** — ECE **{ece_raw:.4f} → {ece_cal:.4f}**",
        f"- AURC raw **{aurc_raw:.4f}** → calibrated **{aurc_cal:.4f}**",
        f"- at alpha = {alpha}: defers on **{n_defer} / {n_total}** "
        f"({n_defer / max(n_total, 1):.0%}) of evaluation rows",
    ]
    if n_total > MAX_TABLE_ROWS:
        md.append(f"- table truncated to the first {MAX_TABLE_ROWS} rows")
    for w in warnings:
        md.append(f"- ⚠️ {w}")
    return "\n".join(md)


# --------------------------------------------------------------- Mode A ----

def _validate_predictions_csv(
    df: pd.DataFrame, cal_frac: float, alphas: list[float]
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    """Return (P, y, class_names, errors, warnings) for a predictions CSV."""
    errors: list[str] = []
    warnings: list[str] = []

    cols = {c.strip().lower(): c for c in df.columns}
    if "label" not in cols:
        return None, None, None, ["missing required column: label (0-based true class index)"], []
    label_col = cols["label"]
    prob_cols = [c for c in df.columns if c != label_col]
    if len(prob_cols) < 2:
        errors.append("need >= 2 probability columns (one per class) besides `label`")
    if len(df) == 0:
        errors.append("file has a header but no data rows")
    if errors:
        return None, None, None, errors, []

    try:
        P = df[prob_cols].to_numpy(dtype=float)
    except (ValueError, TypeError) as e:
        return None, None, None, [f"probability columns must be numeric: {e}"], []
    if not np.isfinite(P).all() or (P < 0).any():
        errors.append("probabilities must be finite and non-negative")

    try:
        y_float = df[label_col].to_numpy(dtype=float)
    except (ValueError, TypeError) as e:
        return None, None, None, [f"label column must be numeric: {e}"], []
    if not np.all(y_float == np.round(y_float)):
        errors.append("labels must be integers (0-based class indices)")
    y = y_float.astype(int)
    k = len(prob_cols)
    bad = (y < 0) | (y >= k)
    if bad.any():
        errors.append(
            f"{int(bad.sum())} label(s) outside 0..{k - 1} "
            f"(e.g. row {int(np.where(bad)[0][0]) + 1})"
        )

    # same calibration-size floor custom_csv enforces: below ceil(1/alpha_min)
    # the conformal quantile is undefined and sets are always full
    n_cal_est = int(round(len(df) * cal_frac))
    hard_floor = math.ceil(1.0 / min(alphas))
    if n_cal_est < hard_floor:
        errors.append(
            f"estimated calibration size {n_cal_est} < {hard_floor}: at alpha="
            f"{min(alphas)} the conformal quantile is undefined. Add rows or raise cal_frac."
        )
    if errors:
        return None, None, None, errors, []

    sums = P.sum(axis=1)
    if (sums <= 0).any():
        return None, None, None, ["some rows have zero/negative probability mass"], []
    off = np.abs(sums - 1.0) > 5e-3
    if off.any():
        warnings.append(
            f"{int(off.sum())} row(s) did not sum to 1 (max deviation "
            f"{float(np.abs(sums - 1).max()):.3f}); renormalised."
        )
    P = P / sums[:, None]
    return P, y, [str(c) for c in prob_cols], [], warnings


def run_mode_a(file, alpha_choice: str):
    """Reassemble calibrate -> conformal -> gate on an uploaded (P, label) CSV."""
    empty = (None,) * 5
    if file is None:
        return "**Upload a predictions CSV first.**", *empty
    cfg = load_config()
    alphas = [float(a) for a in cfg["alpha"]]
    try:
        df = pd.read_csv(file if isinstance(file, str) else file.name)
    except Exception as e:
        return f"**Could not parse CSV:** {e}", *empty

    P, y, names, errors, warnings = _validate_predictions_csv(df, cfg["cal_frac"], alphas)
    if errors:
        return "**Validation failed:**\n" + "\n".join(f"- ERROR: {e}" for e in errors), *empty

    # seeded calibration/evaluation split honouring cal_frac
    rng = np.random.default_rng(cfg["seed"])
    perm = rng.permutation(len(y))
    n_cal = int(round(len(y) * cfg["cal_frac"]))
    cal_idx, test_idx = perm[:n_cal], perm[n_cal:]

    # temperature scaling is defined on LOGITS -> convert probabilities first
    logits = np.log(np.clip(P, 1e-12, 1.0))
    T = cal.fit_temperature(logits[cal_idx], y[cal_idx])
    P_cal_s = cal.apply_temperature(logits[cal_idx], T)
    P_test_s = cal.apply_temperature(logits[test_idx], T)

    P_test_raw, y_cal, y_test = P[test_idx], y[cal_idx], y[test_idx]
    correct = (P_test_raw.argmax(axis=1) == y_test).astype(float)
    conf_raw, conf_cal = P_test_raw.max(axis=1), P_test_s.max(axis=1)
    ece_raw, ece_cal = cal.ece(conf_raw, correct), cal.ece(conf_cal, correct)

    tmp = Path(tempfile.mkdtemp(prefix="calibration-demo-"))
    plot_reliability(conf_raw, conf_cal, correct, ece_raw, ece_cal, T,
                     tmp / "reliability.png")

    report = _coverage_table(conformal.coverage_check(
        P_cal_s, y_cal, P_test_s, y_test, alphas, rng=np.random.default_rng(cfg["seed"])
    ))

    cov_r, risk_r = selective.risk_coverage(conf_raw, correct)
    cov_c, risk_c = selective.risk_coverage(conf_cal, correct)
    aurc_r, aurc_c = selective.aurc(cov_r, risk_r), selective.aurc(cov_c, risk_c)
    head = selective.headline(conf_raw, conf_cal, correct)
    plot_risk_coverage({"raw": (cov_r, risk_r, aurc_r),
                        "calibrated": (cov_c, risk_c, aurc_c)},
                       head, tmp / "risk_coverage.png")

    alpha = float(alpha_choice)
    sets, _ = conformal.predict_sets(P_cal_s, y_cal, P_test_s, alpha, method="lac")
    defer_df = _defer_table(P_test_raw, P_test_s, sets, y_test, names)
    n_defer = int(conformal.abstain_mask_from_sets(sets).sum())

    summary = _summary_md(n_cal, len(y_test), T, ece_raw, ece_cal, aurc_r, aurc_c,
                          head, alpha, n_defer, len(y_test), warnings)
    return (summary, defer_df, str(tmp / "reliability.png"), report,
            str(tmp / "risk_coverage.png"))


# --------------------------------------------------------------- Mode B ----

def _mode_b_defer_table(cfg: dict, csv_path: str, answer_format: str,
                        T: float, alpha: float) -> pd.DataFrame | None:
    """Per-item table from the score cache pipeline.run just wrote (no
    re-scoring, no refitting: T comes from metrics.json)."""
    ds = data_mod.load_csv(csv_path, cfg["alpha"], cal_frac=cfg["cal_frac"],
                           seed=cfg["seed"], answer_format=answer_format)
    model = cfg["hf_model"] if cfg["model_route"] == "hf" else cfg["api_model"]
    cache = {}
    for split in ("cal", "test"):
        for mode in ("letter", "cloze", "api"):
            p = cache_path(OUTPUTS / "cache", model, ds["name"], split, mode)
            if p.exists():
                cache[split] = np.load(p, allow_pickle=False)
                break
        if split not in cache:
            return None
    P_cal_s = cal.apply_temperature(cache["cal"]["logits"], T)
    P_test_s = cal.apply_temperature(cache["test"]["logits"], T)
    P_test_raw = cache["test"]["P"]
    y_cal, y_test = cache["cal"]["labels"], cache["test"]["labels"]
    sets, _ = conformal.predict_sets(P_cal_s, y_cal, P_test_s, alpha, method="lac")
    questions = [it.question[:90] for it in ds["test"]]
    letters = option_letters(ds["n_options"])
    return _defer_table(P_test_raw, P_test_s, sets, y_test, letters, questions)


def run_mode_b(file, answer_format: str):
    empty = (None,) * 6
    if file is None:
        return "**Upload an MCQ CSV first.**", *empty
    csv_path = file if isinstance(file, str) else file.name
    cfg = load_config()
    cfg_b = dict(cfg)
    cfg_b.update(dataset="csv", csv_path=csv_path, answer_format=answer_format)
    fd, tmp_cfg = tempfile.mkstemp(suffix=".yaml", prefix="calibration-demo-")
    Path(tmp_cfg).write_text(yaml.safe_dump(cfg_b))

    try:
        metrics = run_pipeline(tmp_cfg)
    except CSVValidationError as e:
        issues = "\n".join(f"- {i}" for i in e.issues)
        return f"**CSV failed validation:**\n{issues}", *empty
    except Exception as e:
        return f"**Pipeline failed:** {e}", *empty

    head = metrics["headline"]
    alpha = _pipeline_alpha([float(a) for a in cfg["alpha"]])
    defer_df = _mode_b_defer_table(cfg, csv_path, answer_format,
                                   metrics["temperature"], alpha)
    n_defer = int((defer_df["decision"] == "DEFER").sum()) if defer_df is not None else 0
    n_total = metrics["n_test"]

    summary = _summary_md(metrics["n_cal"], n_total, metrics["temperature"],
                          metrics["ece_raw"], metrics["ece_calibrated"],
                          metrics["aurc_raw"], metrics["aurc_calibrated"],
                          head, alpha, n_defer, n_total, [])
    coverage = _coverage_table(json.loads((OUTPUTS / "coverage_check.json").read_text()))
    caught = (OUTPUTS / "caught_examples.md").read_text()
    return (summary, defer_df, str(OUTPUTS / "reliability.png"), coverage,
            str(OUTPUTS / "risk_coverage.png"), caught)


# ------------------------------------------------------------------ UI -----

def build_demo() -> gr.Blocks:
    cfg = load_config()
    alpha_choices = [str(a) for a in cfg["alpha"]]
    default_alpha = str(_pipeline_alpha(cfg["alpha"]))

    with gr.Blocks(title="calibration layer demo") as demo:
        gr.Markdown(
            "# Calibration layer — answer only when sure\n"
            "Calibrated confidence + conformal prediction sets on top of any "
            "classifier. **ANSWER** when the conformal set is a single option, "
            "**DEFER** otherwise."
        )
        with gr.Tab("Mode A — bring your own predictions (fast, no model)"):
            gr.Markdown(
                "Upload a CSV with K numeric probability columns (one per class) "
                "and a `label` column with the 0-based true class index."
            )
            with gr.Row():
                file_a = gr.File(label="predictions CSV", file_types=[".csv"])
                alpha_a = gr.Dropdown(alpha_choices, value=default_alpha,
                                      label="alpha (miscoverage) for the gate")
            btn_a = gr.Button("Calibrate + gate", variant="primary")
            summary_a = gr.Markdown()
            defer_a = gr.Dataframe(label="ANSWER / DEFER — per item", wrap=True)
            with gr.Row():
                rel_a = gr.Image(label="reliability (before vs after)", type="filepath")
                rc_a = gr.Image(label="risk vs coverage", type="filepath")
            cov_a = gr.Dataframe(label="empirical conformal coverage vs target")
            btn_a.click(run_mode_a, [file_a, alpha_a],
                        [summary_a, defer_a, rel_a, cov_a, rc_a])

        with gr.Tab("Mode B — bring your own questions (scores with the model — slow)"):
            gr.Markdown(
                "Upload an MCQ CSV in the `src/custom_csv.py` contract "
                "(`question`, `option_a`..., `answer`). ⚠️ **Every question is "
                f"scored with `{cfg['hf_model']}` on first run — this can take "
                "minutes and downloads the model if absent.** Re-runs of the "
                "same file are served from the score cache."
            )
            with gr.Row():
                file_b = gr.File(label="MCQ CSV", file_types=[".csv"])
                fmt_b = gr.Dropdown(["auto", "letter", "index0", "index1"],
                                    value="auto", label="answer format")
            btn_b = gr.Button("Run full pipeline", variant="primary")
            summary_b = gr.Markdown()
            defer_b = gr.Dataframe(label="ANSWER / DEFER — per question", wrap=True)
            with gr.Row():
                rel_b = gr.Image(label="reliability (before vs after)", type="filepath")
                rc_b = gr.Image(label="risk vs coverage", type="filepath")
            cov_b = gr.Dataframe(label="empirical conformal coverage vs target")
            caught_b = gr.Markdown(label="caught examples")
            btn_b.click(run_mode_b, [file_b, fmt_b],
                        [summary_b, defer_b, rel_b, cov_b, rc_b, caught_b])
    return demo


if __name__ == "__main__":
    build_demo().launch()
