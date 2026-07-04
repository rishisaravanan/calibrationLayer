"""The centrepiece test: the split conformal coverage guarantee holds.

On synthetic data with *known miscalibrated* probabilities, mean empirical
coverage over many random calibration/test splits must be ~= 1 - alpha for
both LAC and APS. No model, no network — pure maths.
"""
import numpy as np
import pytest

from src.conformal import (
    abstain_mask_from_sets,
    conformal_quantile,
    empirical_coverage,
    predict_sets,
)

N_TRIALS = 200
N_CAL = 800
N_TEST = 800
K = 4


def _synthetic_miscalibrated(rng: np.random.Generator, n: int):
    """Labels drawn from true probs; model reports sharpened (overconfident)
    versions of those probs, so it is informative but miscalibrated."""
    true_p = rng.dirichlet(np.ones(K) * 1.2, size=n)
    y = np.array([rng.choice(K, p=p) for p in true_p])
    sharp = true_p ** 3.0
    P = sharp / sharp.sum(axis=1, keepdims=True)
    return P, y


@pytest.mark.parametrize("method", ["lac", "aps"])
@pytest.mark.parametrize("alpha", [0.10, 0.20])
def test_mean_coverage_matches_guarantee(method, alpha):
    rng = np.random.default_rng(0)
    coverages = []
    for _ in range(N_TRIALS):
        P, y = _synthetic_miscalibrated(rng, N_CAL + N_TEST)
        perm = rng.permutation(len(y))
        cal_idx, test_idx = perm[:N_CAL], perm[N_CAL:]
        sets, _ = predict_sets(
            P[cal_idx], y[cal_idx], P[test_idx], alpha, method,
            rng=rng if method == "aps" else None,  # randomised APS -> exact
        )
        coverages.append(empirical_coverage(sets, y[test_idx]))
    mean_cov = np.mean(coverages)
    # guarantee: 1 - alpha <= coverage <= 1 - alpha + 1/(n_cal + 1), plus MC noise
    assert mean_cov >= 1 - alpha - 0.01, f"coverage {mean_cov:.3f} below target"
    assert mean_cov <= 1 - alpha + 0.015, f"coverage {mean_cov:.3f} too conservative"


def test_nonrandomised_aps_is_valid():
    """Deterministic APS (include the crossing class) must never under-cover.

    It can be conservative when the model's probabilities are sharp — the
    exactness claim is tested above with the randomised variant.
    """
    rng = np.random.default_rng(1)
    alpha = 0.10
    coverages = []
    for _ in range(100):
        P, y = _synthetic_miscalibrated(rng, N_CAL + N_TEST)
        perm = rng.permutation(len(y))
        sets, _ = predict_sets(P[perm[:N_CAL]], y[perm[:N_CAL]], P[perm[N_CAL:]], alpha, "aps")
        coverages.append(empirical_coverage(sets, y[perm[N_CAL:]]))
    assert np.mean(coverages) >= 1 - alpha - 0.01


def test_quantile_small_n_gives_full_sets():
    # with n too small for alpha, qhat = inf -> every set is the full label set
    assert conformal_quantile(np.array([0.1, 0.2]), alpha=0.05) == float("inf")


def test_abstain_mask():
    sets = np.array([[True, False], [True, True], [False, False]])
    assert list(abstain_mask_from_sets(sets)) == [False, True, True]
