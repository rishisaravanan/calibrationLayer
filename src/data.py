"""Dataset loaders.

Each loader returns everything the pipeline needs downstream:
  - tabular: precomputed probability matrices (no LLM required) — the spine.
  - arc / mmlu: lists of MCQ items to be scored by `scoring.py`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_LETTERS = ["A", "B", "C", "D"]
_NUM_KEYS = {"1": 0, "2": 1, "3": 2, "4": 3}


@dataclass
class MCQItem:
    """One multiple-choice question with exactly four options."""

    qid: str
    question: str
    options: list[str]
    label_idx: int


@dataclass
class TabularData:
    """Probability matrices and labels for the LLM-free spine."""

    P_cal: np.ndarray
    y_cal: np.ndarray
    P_test: np.ndarray
    y_test: np.ndarray
    logits_cal: np.ndarray
    logits_test: np.ndarray


def _normalise_items(rows, qid_key, q_key, choices_fn, label_fn) -> tuple[list[MCQItem], int]:
    """Keep only 4-option questions; return (items, n_dropped)."""
    items: list[MCQItem] = []
    dropped = 0
    for row in rows:
        options, label = choices_fn(row), label_fn(row)
        if options is None or label is None or len(options) != 4:
            dropped += 1
            continue
        items.append(
            MCQItem(
                qid=str(row[qid_key]),
                question=str(row[q_key]),
                options=[str(o) for o in options],
                label_idx=int(label),
            )
        )
    return items, dropped


def load_arc() -> dict:
    """ARC-Challenge: `validation` split for calibration, `test` for eval.

    ARC contains a few 3- and 5-option questions; those are dropped and the
    counts recorded in the returned dict.
    """
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")

    def choices(row):
        return row["choices"]["text"]

    def label(row):
        key = str(row["answerKey"]).strip()
        labels = [str(l).strip() for l in row["choices"]["label"]]
        if key not in labels:
            return None
        return labels.index(key)

    cal, dropped_cal = _normalise_items(ds["validation"], "id", "question", choices, label)
    test, dropped_test = _normalise_items(ds["test"], "id", "question", choices, label)
    return {
        "name": "arc",
        "cal": cal,
        "test": test,
        "n_dropped": {"cal": dropped_cal, "test": dropped_test},
    }


def load_mmlu(subjects: list[str] | None = None) -> dict:
    """MMLU: `validation` split for calibration, `test` for eval (always 4 options)."""
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all")

    def keep(row):
        return subjects is None or row["subject"] in subjects

    def choices(row):
        return row["choices"]

    def label(row):
        return int(row["answer"])

    def rows_of(split):
        return [
            {**row, "id": f"{row['subject']}/{i}"}
            for i, row in enumerate(ds[split])
            if keep(row)
        ]

    cal, dropped_cal = _normalise_items(rows_of("validation"), "id", "question", choices, label)
    test, dropped_test = _normalise_items(rows_of("test"), "id", "question", choices, label)
    return {
        "name": "mmlu",
        "cal": cal,
        "test": test,
        "n_dropped": {"cal": dropped_cal, "test": dropped_test},
    }


def load_tabular(
    seed: int = 42,
    n_samples: int = 20_000,
    cal_frac: float = 0.30,
    label_noise: float = 0.10,
) -> TabularData:
    """LLM-free spine: Covertype + gradient boosting.

    Subsamples Covertype, trains a HistGradientBoostingClassifier on half,
    and returns predict_proba matrices on a calibration/test split of the
    other half. Label noise is injected into the *training* labels so test
    accuracy lands in the 75-85% band where the selective story is vivid.
    log(P) stands in for logits when temperature-scaling this path.
    """
    from sklearn.datasets import fetch_covtype
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(seed)
    X, y = fetch_covtype(return_X_y=True)
    y = y.astype(int) - 1  # 1..7 -> 0..6

    idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
    X, y = X[idx], y[idx]

    X_train, X_pool, y_train, y_pool = train_test_split(
        X, y, test_size=0.5, random_state=seed, stratify=y
    )

    if label_noise > 0:
        flip = rng.random(len(y_train)) < label_noise
        y_train = y_train.copy()
        y_train[flip] = rng.integers(0, y.max() + 1, size=flip.sum())

    clf = HistGradientBoostingClassifier(random_state=seed)
    clf.fit(X_train, y_train)
    P_pool = clf.predict_proba(X_pool)

    P_cal, P_test, y_cal, y_test = train_test_split(
        P_pool, y_pool, test_size=1 - cal_frac, random_state=seed, stratify=y_pool
    )
    logits_cal = np.log(np.clip(P_cal, 1e-12, None))
    logits_test = np.log(np.clip(P_test, 1e-12, None))
    return TabularData(P_cal, y_cal, P_test, y_test, logits_cal, logits_test)
