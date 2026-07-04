"""Optional agreement check of the from-scratch conformal code against MAPIE.

This exists to show the hand-rolled maths in `conformal.py` is correct — it
never replaces it. Enabled with `crosscheck: true` in config.yaml (requires
`pip install mapie`).
"""
from __future__ import annotations

import numpy as np

from . import conformal


class _PrefitProbs:
    """sklearn-style classifier that replays precomputed probabilities.

    X is an (N, 1) array of row indices into the stored probability matrix,
    which lets MAPIE consume exactly the same P as the from-scratch code.
    """

    def __init__(self, P: np.ndarray, n_classes: int):
        self.P = P
        self.classes_ = np.arange(n_classes)
        self.__sklearn_is_fitted__ = lambda: True

    def fit(self, X, y):  # pragma: no cover - never called with cv="prefit"
        return self

    def predict_proba(self, X):
        return self.P[np.asarray(X, dtype=int).ravel()]

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


def run_crosscheck(
    P_cal: np.ndarray,
    y_cal: np.ndarray,
    P_test: np.ndarray,
    y_test: np.ndarray,
    alphas: list[float],
    tol_coverage: float = 0.02,
    tol_set_size: float = 0.10,
) -> bool:
    """Compare LAC/APS coverage and set size against MAPIE (>= 1.0). Returns
    True on agreement; raises AssertionError with a clear report otherwise."""
    try:
        from mapie.classification import SplitConformalClassifier
    except ImportError as e:
        print(f"[crosscheck] skipped: {e} (pip install mapie)")
        return False

    P_all = np.vstack([P_cal, P_test])
    est = _PrefitProbs(P_all, P_all.shape[1])
    X_cal = np.arange(len(P_cal)).reshape(-1, 1)
    X_test = (len(P_cal) + np.arange(len(P_test))).reshape(-1, 1)

    ok = True
    for method in ("lac", "aps"):
        mapie = SplitConformalClassifier(
            estimator=est,
            confidence_level=[1 - a for a in alphas],
            conformity_score=method,
            prefit=True,
        )
        mapie.conformalize(X_cal, y_cal)
        _, mapie_sets_all = mapie.predict_set(X_test)
        for j, alpha in enumerate(alphas):
            mapie_sets = mapie_sets_all[:, :, j]
            if method == "aps":
                # mirror MAPIE's exact recipe: randomised calibration scores
                # (Romano et al.) + deterministic include-last-label test sets
                rng = np.random.default_rng(0)
                scores = conformal.aps_scores(P_cal, y_cal, rng=rng)
                qhat = conformal.conformal_quantile(scores, alpha)
                ours = conformal.aps_sets(P_test, qhat)
            else:
                ours, _ = conformal.predict_sets(P_cal, y_cal, P_test, alpha, method)
            cov_m = conformal.empirical_coverage(mapie_sets, y_test)
            cov_o = conformal.empirical_coverage(ours, y_test)
            size_m = conformal.mean_set_size(mapie_sets)
            size_o = conformal.mean_set_size(ours)
            agree = abs(cov_m - cov_o) <= tol_coverage and abs(size_m - size_o) <= tol_set_size
            ok &= agree
            print(
                f"[crosscheck] {method} alpha={alpha}: coverage ours={cov_o:.3f} "
                f"mapie={cov_m:.3f} | set size ours={size_o:.2f} mapie={size_m:.2f} "
                f"-> {'AGREE' if agree else 'DISAGREE'}"
            )
    assert ok, "from-scratch conformal disagrees with MAPIE beyond tolerance"
    print("[crosscheck] from-scratch implementation agrees with MAPIE.")
    return True
