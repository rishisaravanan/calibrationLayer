"""End-to-end pipeline: data -> (scoring) -> calibration -> conformal ->
selective analysis -> the four artefacts + metrics.json.

Run with:  python -m src.pipeline  [path/to/config.yaml]
Idempotent and cache-aware: LLM inference is cached after the first run.
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import yaml

from . import calibration as cal
from . import conformal, selective
from .plots import plot_reliability, plot_risk_coverage

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
HEADLINE_COVERAGE = 0.70
COVERAGE_POINTS = [1.00, 0.90, 0.80, 0.70, 0.50]


def load_config(path: str | Path | None = None) -> dict:
    """Read config.yaml (defaults to the repo root copy)."""
    with open(path or ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> np.random.Generator:
    """Seed python, numpy (and torch if present); return a Generator."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
    return np.random.default_rng(seed)


def _load_llm_matrices(cfg: dict) -> dict:
    """Score (or load cached) probability matrices for arc/mmlu."""
    from . import data as data_mod
    from .scoring import score_split

    if cfg["dataset"] == "arc":
        ds = data_mod.load_arc()
    elif cfg["dataset"] == "csv":
        if not cfg.get("csv_path"):
            raise ValueError("dataset: csv requires csv_path in config.yaml")
        ds = data_mod.load_csv(
            cfg["csv_path"],
            cfg["alpha"],
            cal_frac=cfg["cal_frac"],
            seed=cfg["seed"],
            answer_format=cfg.get("answer_format", "auto"),
        )
    else:
        ds = data_mod.load_mmlu(cfg.get("mmlu_subjects"))
    if any(ds["n_dropped"].values()):
        print(f"[data] dropped non-4-option questions: {ds['n_dropped']}")

    kwargs = dict(
        dataset=ds["name"],
        model_route=cfg["model_route"],
        hf_model=cfg["hf_model"],
        api_model=cfg.get("api_model"),
        score_mode=cfg.get("score_mode", "letter"),
        n_options=ds.get("n_options", 4),
        cache_dir=OUTPUTS / "cache",
    )
    cal_scored = score_split(ds["cal"], split="cal", **kwargs)
    test_scored = score_split(ds["test"], split="test", **kwargs)
    test_items = {it.qid: it for it in ds["test"]}
    return {
        "P_cal": cal_scored["P"],
        "y_cal": cal_scored["labels"],
        "logits_cal": cal_scored["logits"],
        "P_test": test_scored["P"],
        "y_test": test_scored["labels"],
        "logits_test": test_scored["logits"],
        "qids_test": list(test_scored["qids"]),
        "items_test": test_items,
    }


def _load_matrices(cfg: dict) -> dict:
    if cfg["dataset"] == "tabular":
        from .data import load_tabular

        td = load_tabular(seed=cfg["seed"], cal_frac=cfg["cal_frac"])
        return {
            "P_cal": td.P_cal,
            "y_cal": td.y_cal,
            "logits_cal": td.logits_cal,
            "P_test": td.P_test,
            "y_test": td.y_test,
            "logits_test": td.logits_test,
            "qids_test": None,
            "items_test": None,
        }
    if cfg["dataset"] in ("arc", "mmlu", "csv"):
        return _load_llm_matrices(cfg)
    raise ValueError(f"unknown dataset: {cfg['dataset']!r}")


def _caught_examples_md(
    d: dict,
    P_test_cal: np.ndarray,
    sets: np.ndarray,
    conf_threshold: float,
    alpha: float,
    n_examples: int = 3,
) -> str:
    """2-3 cases the model got confidently wrong but the layer flagged."""
    P_raw, y = d["P_test"], d["y_test"]
    pred = P_raw.argmax(axis=1)
    conf_raw = P_raw.max(axis=1)
    conf_cal = P_test_cal.max(axis=1)
    set_sizes = sets.sum(axis=1)

    wrong = pred != y
    flagged = (set_sizes >= 2) | (conf_cal < conf_threshold)
    candidates = np.where(wrong & flagged)[0]
    # prefer cases the conformal set itself flagged, then by raw confidence
    rank = np.lexsort((-conf_raw[candidates], set_sizes[candidates] < 2))
    candidates = candidates[rank][:n_examples]

    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lines = [
        "# Confidently wrong — and caught\n",
        f"Cases where the base model was wrong with high raw confidence, but the "
        f"uncertainty layer flagged them: conformal set size >= 2 (alpha = {alpha}) "
        f"or calibrated confidence below the abstention threshold "
        f"({conf_threshold:.3f}, the cutoff for {HEADLINE_COVERAGE:.0%} coverage).\n",
    ]
    for rank, i in enumerate(candidates, 1):
        lines.append(f"## Example {rank}\n")
        if d["items_test"] is not None:
            item = d["items_test"][d["qids_test"][i]]
            lines.append(f"**Question:** {item.question}\n")
            for k, opt in enumerate(item.options):
                marks = []
                if k == pred[i]:
                    marks.append("model's answer")
                if k == y[i]:
                    marks.append("correct")
                suffix = f"  <- {', '.join(marks)}" if marks else ""
                lines.append(f"- {letters[k]}. {opt}{suffix}")
            lines.append("")
        else:
            lines.append(
                f"Test row {i}: model predicted class {pred[i]}, true class {y[i]}.\n"
            )
        in_set = [letters[k] if d["items_test"] is not None else str(k)
                  for k in np.where(sets[i])[0]]
        reasons = []
        if set_sizes[i] != 1:
            reasons.append(
                f"conformal set {{{', '.join(in_set)}}} (size {int(set_sizes[i])}) is not a singleton"
            )
        if conf_cal[i] < conf_threshold:
            reasons.append(
                f"calibrated confidence {conf_cal[i]:.2f} < abstention threshold {conf_threshold:.2f}"
            )
        lines.append(
            f"Raw confidence **{conf_raw[i]:.2f}** — and wrong. "
            f"Flagged because {'; and '.join(reasons)} -> **abstain / escalate**.\n"
        )
    return "\n".join(lines)


def run(config_path: str | Path | None = None) -> dict:
    """Execute the full pipeline; returns the metrics dict."""
    cfg = load_config(config_path)
    rng = set_seed(cfg["seed"])
    OUTPUTS.mkdir(exist_ok=True)

    print(f"[pipeline] dataset={cfg['dataset']}  seed={cfg['seed']}")
    d = _load_matrices(cfg)
    y_test = d["y_test"]
    n_classes = d["P_test"].shape[1]
    print(f"[pipeline] cal={len(d['y_cal'])}  test={len(y_test)}  classes={n_classes}")

    # ---- calibration: fit T on cal, evaluate ECE on test ------------------
    T = cal.fit_temperature(d["logits_cal"], d["y_cal"])
    P_test_cal = cal.apply_temperature(d["logits_test"], T)
    P_cal_cal = cal.apply_temperature(d["logits_cal"], T)

    pred = d["P_test"].argmax(axis=1)
    correct = (pred == y_test).astype(float)
    conf_raw = d["P_test"].max(axis=1)
    conf_cal = P_test_cal.max(axis=1)

    ece_raw = cal.ece(conf_raw, correct)
    ece_cal = cal.ece(conf_cal, correct)
    print(f"[calibration] T={T:.3f}  ECE raw={ece_raw:.4f} -> calibrated={ece_cal:.4f}")

    # ---- conformal coverage sweep (on calibrated probabilities) -----------
    coverage_report = conformal.coverage_check(
        P_cal_cal, d["y_cal"], P_test_cal, y_test, cfg["alpha"], rng=rng
    )
    with open(OUTPUTS / "coverage_check.json", "w") as f:
        json.dump(coverage_report, f, indent=2)

    # ---- selective prediction ---------------------------------------------
    cov_raw, risk_raw = selective.risk_coverage(conf_raw, correct)
    cov_cal, risk_cal = selective.risk_coverage(conf_cal, correct)
    aurc_raw = selective.aurc(cov_raw, risk_raw)
    aurc_cal = selective.aurc(cov_cal, risk_cal)
    acc_at_cov = selective.accuracy_at_coverage(conf_cal, correct, COVERAGE_POINTS)
    head = selective.headline(conf_raw, conf_cal, correct, HEADLINE_COVERAGE)

    # ---- figures ------------------------------------------------------------
    plot_reliability(
        conf_raw, conf_cal, correct, ece_raw, ece_cal, T, OUTPUTS / "reliability.png"
    )
    plot_risk_coverage(
        {
            "raw": (cov_raw, risk_raw, aurc_raw),
            "calibrated": (cov_cal, risk_cal, aurc_cal),
        },
        head,
        OUTPUTS / "risk_coverage.png",
    )

    # ---- caught examples ----------------------------------------------------
    alpha0 = cfg["alpha"][1] if len(cfg["alpha"]) > 1 else cfg["alpha"][0]
    sets, _ = conformal.predict_sets(
        P_cal_cal, d["y_cal"], P_test_cal, alpha0, method="lac"
    )
    k70 = max(1, math.ceil(HEADLINE_COVERAGE * len(conf_cal)))
    conf_threshold = float(np.sort(conf_cal)[::-1][k70 - 1])
    (OUTPUTS / "caught_examples.md").write_text(
        _caught_examples_md(d, P_test_cal, sets, conf_threshold, alpha0)
    )

    # ---- metrics.json ---------------------------------------------------------
    metrics = {
        "dataset": cfg["dataset"],
        "model": cfg["hf_model"] if cfg["model_route"] == "hf" else cfg["api_model"],
        "n_cal": int(len(d["y_cal"])),
        "n_test": int(len(y_test)),
        "accuracy": float(correct.mean()),
        "temperature": T,
        "ece_raw": ece_raw,
        "ece_calibrated": ece_cal,
        "aurc_raw": aurc_raw,
        "aurc_calibrated": aurc_cal,
        "accuracy_at_coverage": {f"{c:.0%}": v for c, v in acc_at_cov.items()},
        "empirical_coverage": coverage_report,
        "headline": head,
    }
    with open(OUTPUTS / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    if cfg.get("crosscheck"):
        from .crosscheck import run_crosscheck

        run_crosscheck(P_cal_cal, d["y_cal"], P_test_cal, y_test, cfg["alpha"])

    # ---- summary --------------------------------------------------------------
    print("\n=== uncertainty-gate summary ===")
    print(f"accuracy (full coverage): {correct.mean():.3f}")
    print(f"ECE: {ece_raw:.4f} -> {ece_cal:.4f} (T = {T:.3f})")
    print(f"AURC: raw {aurc_raw:.4f} -> calibrated {aurc_cal:.4f}")
    for method, rows in coverage_report.items():
        for a, row in rows.items():
            print(
                f"coverage[{method}, alpha={a}]: target {row['target_coverage']:.2f}, "
                f"empirical {row['empirical_coverage']:.3f}, "
                f"mean set size {row['mean_set_size']:.2f}"
            )
    print(head["sentence"])
    return metrics


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
