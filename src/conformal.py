"""Split conformal prediction, implemented from scratch.

The entire guarantee lives here. Split conformal: compute a nonconformity
score for every calibration point, take the finite-sample-corrected
(1 - alpha) quantile `qhat`, and build test prediction sets from `qhat`.
If calibration and test points are exchangeable, the set contains the true
label with probability >= 1 - alpha (marginally). `tests/test_conformal.py`
verifies this empirically over many random splits.

Two score functions:
  - LAC  (least ambiguous set-valued classifier): s = 1 - p(true class).
  - APS  (adaptive prediction sets): s = cumulative probability mass of all
    classes ranked at or above the true class.
"""
from __future__ import annotations

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample-corrected (1 - alpha) empirical quantile of the scores.

    With n calibration scores, returns the ceil((n+1)(1-alpha))/n quantile
    (method="higher"). If that level exceeds 1 (n too small for this alpha),
    returns +inf, which yields the full label set.
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    level = np.ceil((n + 1) * (1 - alpha)) / n
    if level > 1:
        return float("inf")
    return float(np.quantile(scores, level, method="higher"))


# ---------------------------------------------------------------- LAC ------

def lac_scores(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """LAC nonconformity: s_i = 1 - P[i, y_i]."""
    return 1.0 - P[np.arange(len(y)), np.asarray(y, dtype=int)]


def lac_sets(P: np.ndarray, qhat: float) -> np.ndarray:
    """Boolean (N, K) prediction sets: include k where P[i, k] >= 1 - qhat."""
    return P >= 1.0 - qhat


# ---------------------------------------------------------------- APS ------

def aps_scores(
    P: np.ndarray, y: np.ndarray, rng: np.random.Generator | None = None
) -> np.ndarray:
    """APS nonconformity: cumulative prob of classes ranked >= the true class.

    If `rng` is given, the true class's own mass is randomised (subtract
    U * p_true), which makes coverage exact rather than conservative.
    """
    y = np.asarray(y, dtype=int)
    n = len(y)
    order = np.argsort(-P, axis=1)
    P_sorted = np.take_along_axis(P, order, axis=1)
    cum = np.cumsum(P_sorted, axis=1)
    # rank position of the true class in the descending sort
    rank = np.argmax(order == y[:, None], axis=1)
    scores = cum[np.arange(n), rank]
    if rng is not None:
        scores = scores - rng.uniform(size=n) * P[np.arange(n), y]
    return scores


def aps_sets(P: np.ndarray, qhat: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Boolean (N, K) APS prediction sets.

    Deterministic (rng=None): add classes in descending prob order until the
    cumulative mass reaches qhat (the crossing class is included). Never
    empty; valid but conservative when scores are granular.

    Randomised (rng given): include sorted class j iff cum_j - u * p_j <= qhat
    with one u per row — the same randomised score used in `aps_scores`, so
    coverage is exact. May produce empty sets (treated as abstentions).
    """
    n, k = P.shape
    order = np.argsort(-P, axis=1)
    P_sorted = np.take_along_axis(P, order, axis=1)
    cum = np.cumsum(P_sorted, axis=1)
    if rng is None:
        # class at sorted position j is included iff the cumulative mass
        # strictly before it is still below qhat (top class always included)
        include_sorted = np.concatenate(
            [np.ones((n, 1), dtype=bool), cum[:, :-1] < qhat], axis=1
        )
    else:
        u = rng.uniform(size=(n, 1))
        include_sorted = (cum - u * P_sorted) <= qhat  # increasing in j -> prefix
    sets = np.zeros_like(include_sorted)
    np.put_along_axis(sets, order, include_sorted, axis=1)
    return sets


# ------------------------------------------------------------ evaluation ---

def empirical_coverage(sets: np.ndarray, y_test: np.ndarray) -> float:
    """Fraction of test points whose prediction set contains the true label."""
    y_test = np.asarray(y_test, dtype=int)
    return float(sets[np.arange(len(y_test)), y_test].mean())


def mean_set_size(sets: np.ndarray) -> float:
    """Average number of labels per prediction set."""
    return float(sets.sum(axis=1).mean())


def predict_sets(
    P_cal: np.ndarray,
    y_cal: np.ndarray,
    P_test: np.ndarray,
    alpha: float,
    method: str = "lac",
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float]:
    """Calibrate qhat on (P_cal, y_cal) and return (test sets, qhat)."""
    if method == "lac":
        qhat = conformal_quantile(lac_scores(P_cal, y_cal), alpha)
        return lac_sets(P_test, qhat), qhat
    if method == "aps":
        qhat = conformal_quantile(aps_scores(P_cal, y_cal, rng=rng), alpha)
        return aps_sets(P_test, qhat, rng=rng), qhat
    raise ValueError(f"unknown conformal method: {method!r} (use 'lac' or 'aps')")


def coverage_check(
    P_cal: np.ndarray,
    y_cal: np.ndarray,
    P_test: np.ndarray,
    y_test: np.ndarray,
    alphas: list[float],
    rng: np.random.Generator | None = None,
) -> dict:
    """Sweep alpha for both methods; report empirical coverage vs target.

    The result (written to coverage_check.json by the pipeline) is the
    in-repo evidence that the guarantee is real: empirical coverage should
    sit at or just above 1 - alpha for every row.
    """
    report: dict = {}
    for method in ("lac", "aps"):
        report[method] = {}
        for alpha in alphas:
            sets, qhat = predict_sets(P_cal, y_cal, P_test, alpha, method, rng=rng)
            report[method][str(alpha)] = {
                "target_coverage": round(1 - alpha, 4),
                "empirical_coverage": round(empirical_coverage(sets, y_test), 4),
                "mean_set_size": round(mean_set_size(sets), 4),
                "qhat": round(qhat, 6) if np.isfinite(qhat) else "inf",
            }
    return report


def abstain_mask_from_sets(sets: np.ndarray) -> np.ndarray:
    """Abstain wherever the prediction set is not a singleton."""
    return sets.sum(axis=1) != 1
