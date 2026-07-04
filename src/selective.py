"""Selective prediction: risk-coverage curves, AURC, accuracy at coverage.

Coverage = fraction of questions the model answers (highest-confidence
first); risk = error rate on the answered subset. A useful confidence signal
makes risk collapse as coverage drops — that is the money chart.
"""
from __future__ import annotations

import math

import numpy as np


def risk_coverage(conf: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Risk-coverage curve: answer in descending-confidence order.

    Returns:
        coverage: (N,) values 1/N, 2/N, ..., 1.
        risk: (N,) cumulative error rate on the answered prefix.
    """
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    assert conf.shape == correct.shape and conf.ndim == 1
    order = np.argsort(-conf, kind="stable")
    n = len(conf)
    errors = 1.0 - correct[order]
    coverage = np.arange(1, n + 1) / n
    risk = np.cumsum(errors) / np.arange(1, n + 1)
    return coverage, risk


def aurc(coverage: np.ndarray, risk: np.ndarray) -> float:
    """Area under the risk-coverage curve (trapezoid rule; lower is better)."""
    return float(np.trapezoid(risk, coverage))


def accuracy_at_coverage(
    conf: np.ndarray, correct: np.ndarray, points: list[float]
) -> dict[float, float]:
    """Accuracy on the top ceil(c * N) most confident answers for each c."""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    order = np.argsort(-conf, kind="stable")
    n = len(conf)
    out: dict[float, float] = {}
    for c in points:
        k = max(1, math.ceil(c * n))
        out[c] = float(correct[order[:k]].mean())
    return out


def headline(
    conf_raw: np.ndarray,
    conf_cal: np.ndarray,
    correct: np.ndarray,
    coverage: float = 0.70,
) -> dict:
    """Raw vs calibrated error at the headline coverage level, plus a
    ready-to-print sentence for the chart annotation and README."""
    err_full = 1.0 - float(np.asarray(correct, dtype=float).mean())
    err_raw = 1.0 - accuracy_at_coverage(conf_raw, correct, [coverage])[coverage]
    err_cal = 1.0 - accuracy_at_coverage(conf_cal, correct, [coverage])[coverage]
    return {
        "coverage": coverage,
        "error_full_coverage": err_full,
        "error_raw": err_raw,
        "error_calibrated": err_cal,
        "sentence": (
            f"At {coverage:.0%} coverage, error drops from {err_full:.1%} "
            f"(answer everything) to {err_cal:.1%} (calibrated confidence gate); "
            f"raw-confidence gating gives {err_raw:.1%}."
        ),
    }
