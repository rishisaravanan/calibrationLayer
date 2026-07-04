"""Bring-your-own-CSV path: contract validation, K > 4 end-to-end, and the
letter-vs-cloze tokeniser guard. No model, no network — synthetic data only,
matching the style of test_conformal.py."""
import csv

import numpy as np
import pytest

from src.calibration import apply_temperature, fit_temperature, softmax
from src.conformal import empirical_coverage, predict_sets
from src.custom_csv import CSVValidationError, validate_and_load
from src.data import MCQItem, load_csv
from src.scoring import build_prompt, choose_score_mode, option_letters

K = 6
ALPHAS = [0.10]


def _write_csv(path, n=600, k=K):
    """Synthetic contract-conforming CSV with K letter-suffixed options."""
    cols = ["question"] + [f"option_{chr(ord('a') + i)}" for i in range(k)] + ["answer"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(n):
            options = [f"choice {j} for q{i}" for j in range(k)]
            w.writerow([f"synthetic question {i}?"] + options + [chr(ord("a") + i % k)])


# ------------------------------------------------------- loader mapping ----

def test_load_csv_maps_onto_arc_record_shape(tmp_path):
    p = tmp_path / "synth.csv"
    _write_csv(p)
    ds = load_csv(str(p), ALPHAS, cal_frac=0.30, seed=42)

    # same dict contract as load_arc, plus n_options from the file
    assert set(ds) >= {"name", "cal", "test", "n_dropped", "n_options"}
    assert ds["n_options"] == K
    assert len(ds["cal"]) + len(ds["test"]) == 600
    assert len(ds["cal"]) == 180  # cal_frac of 600
    for it in ds["cal"][:5] + ds["test"][:5]:
        assert isinstance(it, MCQItem)
        assert len(it.options) == K
        assert 0 <= it.label_idx < K
        assert it.qid and it.question

    # content-hashed name -> distinct CSVs cannot share a score cache
    p2 = tmp_path / "synth2.csv"
    _write_csv(p2, n=598)
    assert load_csv(str(p2), ALPHAS)["name"] != ds["name"]


def test_load_csv_is_deterministic_given_seed(tmp_path):
    p = tmp_path / "synth.csv"
    _write_csv(p)
    a = load_csv(str(p), ALPHAS, seed=7)
    b = load_csv(str(p), ALPHAS, seed=7)
    assert [i.question for i in a["cal"]] == [i.question for i in b["cal"]]


# ------------------------------------------------- K > 4 end-to-end --------

def test_k6_csv_end_to_end_calibration_and_coverage(tmp_path):
    """CSV -> records -> synthetic (N, 6) logits -> temperature scaling ->
    conformal sets over 6 classes. Everything downstream is K-agnostic."""
    p = tmp_path / "synth.csv"
    _write_csv(p, n=1200)
    ds = load_csv(str(p), ALPHAS, cal_frac=0.30, seed=42)
    rng = np.random.default_rng(0)

    def fake_logits(items):
        y = np.array([it.label_idx for it in items])
        z = rng.normal(size=(len(items), K))
        z[np.arange(len(items)), y] += 2.0  # informative
        return 3.0 * z, y  # overconfident by construction

    logits_cal, y_cal = fake_logits(ds["cal"])
    logits_test, y_test = fake_logits(ds["test"])

    T = fit_temperature(logits_cal, y_cal)
    assert T > 1  # detects the overconfidence
    P_cal = apply_temperature(logits_cal, T)
    P_test = apply_temperature(logits_test, T)
    assert P_test.shape == (len(ds["test"]), K)

    for method in ("lac", "aps"):
        sets, _ = predict_sets(P_cal, y_cal, P_test, alpha=0.10, method=method)
        assert sets.shape == (len(ds["test"]), K)
        cov = empirical_coverage(sets, y_test)
        assert cov >= 0.90 - 0.04, f"{method} coverage {cov:.3f} far below target"


# -------------------------------------------- letter/cloze tokeniser guard -

class FakeTokenizer:
    """Encodes a configurable set of strings as single tokens, rest as two."""

    def __init__(self, single: set[str]):
        self.single = single

    def encode(self, text, add_special_tokens=False):
        return [hash(text) % 10_000] if text in self.single else [1, 2]


def _tok(letters: list[str]) -> FakeTokenizer:
    return FakeTokenizer({f" {l}" for l in letters} | set(letters))


def test_letter_mode_falls_back_to_cloze_when_letters_multi_token():
    only_ad = _tok(["A", "B", "C", "D"])  # E, F are multi-token
    assert choose_score_mode(only_ad, 4, "letter") == "letter"
    assert choose_score_mode(only_ad, K, "letter") == "cloze"  # K > 4 guard fires


def test_letter_mode_kept_when_all_k_letters_verified():
    all_six = _tok(option_letters(K))
    assert choose_score_mode(all_six, K, "letter") == "letter"


def test_explicit_cloze_request_never_touches_letters():
    broken = _tok([])
    assert choose_score_mode(broken, K, "cloze") == "cloze"


def test_prompt_renders_all_k_letters():
    item = MCQItem("q0", "pick one", [f"opt{j}" for j in range(K)], 0)
    prompt = build_prompt(item)
    assert "F. opt5" in prompt and prompt.endswith("Answer:")


# ----------------------------------------------------- contract rejects ----

def test_ambiguous_one_based_integers_rejected_unless_forced(tmp_path):
    p = tmp_path / "ambig.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question", "option_a", "option_b", "option_c", "answer"])
        for i in range(60):
            w.writerow([f"q{i}", "x", "y", "z", 1 + i % 3])  # 1..3, never 0
    with pytest.raises(CSVValidationError, match="ambiguous"):
        validate_and_load(str(p), ALPHAS)
    records, k, _ = validate_and_load(str(p), ALPHAS, answer_format="index1")
    assert k == 3 and records[0].answer_idx == 0


# ------------------------------------------------------- ARC regression ----

def test_arc_loader_unchanged():
    """The existing ARC path must load exactly as before the csv branch."""
    try:
        from src.data import load_arc

        ds = load_arc()
    except Exception as e:  # dataset cache missing on a fresh machine
        pytest.skip(f"ARC unavailable offline: {e}")
    assert len(ds["cal"]) == 295 and len(ds["test"]) == 1165
    assert ds["n_dropped"] == {"cal": 4, "test": 7}
    assert "n_options" not in ds  # ARC branch untouched; pipeline defaults to 4
    for it in (ds["cal"][0], ds["test"][0]):
        assert isinstance(it, MCQItem) and len(it.options) == 4
        assert 0 <= it.label_idx < 4
