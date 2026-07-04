"""Temperature scaling and calibration diagnostics, implemented from scratch.

Calibration claim: after temperature scaling, when the model says 0.9 it is
right ~90% of the time. `fit_temperature` learns a single scalar T on the
calibration split; ECE is always reported on the held-out test split.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    z = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def _bin_indices(conf: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-width bin index in [0, n_bins) for confidences in [0, 1]."""
    return np.clip((conf * n_bins).astype(int), 0, n_bins - 1)


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error with equal-width bins.

    Each bin contributes (bin_count / N) * |bin_accuracy - bin_mean_conf|.

    Args:
        conf: (N,) predicted confidence (p_max) in [0, 1].
        correct: (N,) 0/1 whether the prediction was right.
        n_bins: number of equal-width bins over [0, 1].
    """
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    assert conf.shape == correct.shape, "conf and correct must align"
    idx = _bin_indices(conf, n_bins)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        gap = abs(correct[mask].mean() - conf[mask].mean())
        total += mask.mean() * gap
    return float(total)


def reliability_data(
    conf: np.ndarray, correct: np.ndarray, n_bins: int = 15
) -> list[tuple[float, float, int]]:
    """Per-bin (mean_conf, accuracy, count) for reliability diagrams.

    Empty bins are returned with count 0 and NaN mean/accuracy so the
    plotting code can decide how to render them.
    """
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    idx = _bin_indices(conf, n_bins)
    rows: list[tuple[float, float, int]] = []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            rows.append((float("nan"), float("nan"), 0))
        else:
            rows.append((float(conf[mask].mean()), float(correct[mask].mean()), n))
    return rows


def nll(logits: np.ndarray, y: np.ndarray, T: float = 1.0) -> float:
    """Mean negative log-likelihood of labels under softmax(logits / T)."""
    logits = np.asarray(logits, dtype=float)
    y = np.asarray(y, dtype=int)
    logp = np.log(softmax(logits / T) + 1e-12)
    return float(-logp[np.arange(len(y)), y].mean())


def fit_temperature(logits_cal: np.ndarray, y_cal: np.ndarray) -> float:
    """Fit scalar temperature T > 0 by minimising NLL on the calibration set.

    T > 1 softens an overconfident model; T < 1 sharpens an underconfident one.
    """
    logits_cal = np.asarray(logits_cal, dtype=float)
    y_cal = np.asarray(y_cal, dtype=int)
    assert logits_cal.ndim == 2 and len(logits_cal) == len(y_cal)
    res = minimize_scalar(
        lambda T: nll(logits_cal, y_cal, T), bounds=(0.05, 10.0), method="bounded"
    )
    return float(res.x)


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Calibrated probability matrix softmax(logits / T)."""
    assert T > 0, "temperature must be positive"
    return softmax(np.asarray(logits, dtype=float) / T)
