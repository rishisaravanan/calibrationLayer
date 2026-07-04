"""Temperature scaling must fix a deliberately overconfident model."""
import numpy as np

from src.calibration import apply_temperature, ece, fit_temperature, nll, softmax

K = 4


def _overconfident_model(rng: np.random.Generator, n: int, scale: float = 3.0):
    """True logits z generate labels; the model reports scale * z, so it is
    overconfident by construction and the ideal temperature is `scale`."""
    z = rng.normal(size=(n, K))
    probs = softmax(z)
    y = np.array([rng.choice(K, p=p) for p in probs])
    return scale * z, y


def test_temperature_scaling_fixes_overconfidence():
    rng = np.random.default_rng(0)
    logits_cal, y_cal = _overconfident_model(rng, 4000)
    logits_test, y_test = _overconfident_model(rng, 4000)

    T = fit_temperature(logits_cal, y_cal)
    assert T > 1, f"overconfident model should need T > 1, got {T:.3f}"
    assert 2.0 < T < 4.5, f"T should be near the true scale 3, got {T:.3f}"

    # held-out NLL improves
    assert nll(logits_test, y_test, T) < nll(logits_test, y_test, 1.0)

    # held-out ECE improves
    P_raw = softmax(logits_test)
    P_cal = apply_temperature(logits_test, T)
    correct = (P_raw.argmax(1) == y_test).astype(float)  # argmax unchanged by T
    ece_raw = ece(P_raw.max(1), correct)
    ece_after = ece(P_cal.max(1), correct)
    assert ece_after < ece_raw
    assert ece_after < 0.03, f"post-calibration ECE should be small, got {ece_after:.3f}"


def test_ece_zero_for_perfect_calibration():
    rng = np.random.default_rng(1)
    n = 200_000
    conf = rng.uniform(0.25, 1.0, size=n)
    correct = (rng.random(n) < conf).astype(float)
    assert ece(conf, correct) < 0.01


def test_apply_temperature_rows_sum_to_one():
    rng = np.random.default_rng(2)
    P = apply_temperature(rng.normal(size=(100, K)), 2.5)
    assert np.allclose(P.sum(axis=1), 1.0)
    assert (P > 0).all()
