"""Risk-coverage behaviour for informative vs uninformative confidence."""
import numpy as np

from src.selective import accuracy_at_coverage, aurc, headline, risk_coverage


def test_risk_monotone_for_informative_confidence():
    # errors concentrated at the low-confidence end -> cumulative risk is
    # monotone non-decreasing in coverage (non-increasing as coverage drops)
    n = 1000
    conf = np.linspace(0.01, 0.99, n)
    correct = (conf >= np.quantile(conf, 0.25)).astype(float)
    coverage, risk = risk_coverage(conf, correct)
    assert np.all(np.diff(risk) >= -1e-12)
    assert risk[-1] > risk[n // 2]  # answering everything is worse than half


def test_aurc_informative_beats_random():
    rng = np.random.default_rng(0)
    n = 20_000
    conf = rng.uniform(size=n)
    correct = (rng.random(n) < conf).astype(float)  # perfectly calibrated ranker
    informative = aurc(*risk_coverage(conf, correct))
    random_rank = aurc(*risk_coverage(rng.permutation(conf), correct))
    assert informative < random_rank


def test_accuracy_at_full_coverage_is_overall_accuracy():
    rng = np.random.default_rng(1)
    conf = rng.uniform(size=500)
    correct = rng.integers(0, 2, size=500).astype(float)
    acc = accuracy_at_coverage(conf, correct, [1.0])[1.0]
    assert np.isclose(acc, correct.mean())


def test_headline_shape():
    rng = np.random.default_rng(2)
    n = 2000
    conf = rng.uniform(size=n)
    correct = (rng.random(n) < conf).astype(float)
    h = headline(conf, conf, correct, coverage=0.7)
    assert h["error_calibrated"] <= h["error_full_coverage"]
    assert "70%" in h["sentence"]
