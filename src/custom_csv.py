"""
custom_csv.py -- bring-your-own labelled dataset for the calibration layer.

WHY THIS FILE EXISTS
--------------------
The calibration + conformal core is model- and task-agnostic: it consumes a
probability matrix P (N x K) and the true labels, nothing else. The only thing
tying the repo to ARC is the *loader* -- the step that turns raw data into
(question, options, answer) records. This module is a drop-in replacement for
that step: point it at any conforming CSV and it produces the same normalised
records the rest of the pipeline already expects.

Labels never pass through the model. The questions/options go to scoring.py to
produce P; the answers are held aside and only used at the calibration and
evaluation step to check P against reality. Keeping labels out of scoring is
what preserves the coverage guarantee -- so this module returns them
separately and never hands them downstream of scoring.

CSV CONTRACT (multiple-choice, v1)
----------------------------------
Columns (header row required, names are case-insensitive, whitespace-trimmed):
  question    : non-empty text, one per row.
  option_a,   : K >= 2 option columns, named contiguously either with letter
  option_b,     suffixes (option_a, option_b, ...) OR numeric suffixes
  ...           (option_1, option_2, ...). Every row must fill all K cells;
                K is fixed per file.
  answer      : the correct option, given as ONE of:
                  - a letter (a/b/c/... matching the option columns), or
                  - a 0-based index, or
                  - a 1-based index.
                Format is auto-detected but must be UNAMBIGUOUS across the
                file. Integer answers whose minimum is 1 (no 0 ever appears)
                cannot be told apart from 0-based data that merely never
                selects the first option -- so they are REJECTED, not guessed.
                Pass answer_format="index0"/"index1"/"letter" to force it.

Example:
  question,option_a,option_b,option_c,option_d,answer
  "Which gas do plants absorb?","oxygen","carbon dioxide","nitrogen","hydrogen",b

VALIDATION PHILOSOPHY
---------------------
A malformed CSV that slips through does not crash -- it silently yields a
confident-looking coverage number that is wrong. So validation is part of the
correctness guarantee, not defensive boilerplate. Two design choices follow:
  * ERRORS are aggregated -- every blocking problem is collected and reported
    at once, so a user fixes them in one pass instead of one-per-run.
  * The calibration-size floor is tied to the smallest alpha: below
    ceil(1/alpha) calibration points the conformal quantile is undefined and
    prediction sets are always full, so that is a hard ERROR, not a warning.

CLI: python -m src.custom_csv path/to/data.csv  [--answer-format letter]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import ceil
import re

import pandas as pd


# --------------------------------------------------------------------------- #
# Reporting primitives
# --------------------------------------------------------------------------- #
class Level(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class Issue:
    level: Level
    message: str
    row: int | None = None  # 1-based data row (header excluded); None = file-level

    def __str__(self) -> str:
        where = f" [row {self.row}]" if self.row is not None else ""
        return f"{self.level.value}{where}: {self.message}"


class CSVValidationError(Exception):
    """Raised when a CSV fails validation. Carries ALL blocking errors."""

    def __init__(self, issues: list[Issue]):
        self.issues = issues
        body = "\n".join("  " + str(i) for i in issues)
        super().__init__(f"{len(issues)} validation error(s):\n{body}")


@dataclass
class MCQRecord:
    """Normalised internal record. answer_idx is ALWAYS 0-based."""

    question: str
    options: list[str]  # length K
    answer_idx: int     # 0 <= answer_idx < K


@dataclass
class ValidationReport:
    issues: list[Issue] = field(default_factory=list)
    n_rows: int = 0
    k_options: int = 0
    answer_format: str = ""
    class_counts: dict[int, int] = field(default_factory=dict)

    def add(self, level: Level, message: str, row: int | None = None) -> None:
        self.issues.append(Issue(level, message, row))

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level is Level.ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level is Level.WARNING]

    def raise_if_errors(self) -> None:
        if self.errors:
            raise CSVValidationError(self.errors)

    def summary(self) -> str:
        head = (
            f"rows: {self.n_rows}  |  options K: {self.k_options}  |  "
            f"answer format: {self.answer_format or '-'}  |  "
            f"errors: {len(self.errors)}  warnings: {len(self.warnings)}"
        )
        lines = [head]
        if self.class_counts:
            dist = ", ".join(
                f"{chr(ord('a') + k)}={v}" for k, v in sorted(self.class_counts.items())
            )
            lines.append(f"answer distribution: {dist}")
        for i in self.issues:
            lines.append("  " + str(i))
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Structural detection
# --------------------------------------------------------------------------- #
_OPTION_RE = re.compile(r"^option[_]?([a-z]|\d+)$")


def _detect_option_columns(columns: list[str]) -> tuple[list[str], list[Issue]]:
    """Find option columns, verify they are contiguous, return them in order."""
    issues: list[Issue] = []
    found: dict[str, str] = {}
    for col in columns:
        m = _OPTION_RE.match(col.strip().lower())
        if m:
            found[col] = m.group(1)

    if len(found) < 2:
        issues.append(
            Issue(
                Level.ERROR,
                "need >= 2 option columns named option_a, option_b, ... "
                "(or option_1, option_2, ...)",
            )
        )
        return [], issues

    letters = [k for k in found.values() if k.isalpha()]
    numbers = [k for k in found.values() if k.isdigit()]
    if letters and numbers:
        issues.append(
            Issue(Level.ERROR, "option columns mix letter and numeric suffixes; pick one scheme")
        )
        return [], issues

    if letters:
        order = sorted(found, key=lambda c: found[c])
        got: list = [found[c] for c in order]
        expected: list = [chr(ord("a") + i) for i in range(len(order))]
    else:
        order = sorted(found, key=lambda c: int(found[c]))
        got = [int(found[c]) for c in order]
        expected = list(range(1, len(order) + 1))

    if got != expected:
        issues.append(
            Issue(Level.ERROR, f"option columns not contiguous; expected {expected}, found {got}")
        )
        return [], issues

    return order, issues


def _detect_answer_format(values: list[str], k: int, report: ValidationReport) -> str | None:
    """Infer 'letter' | 'index0' | 'index1', refusing ambiguous integer cases."""
    raw = [str(v).strip() for v in values]

    if all(re.fullmatch(r"[A-Za-z]", v) for v in raw):
        return "letter"

    try:
        ints = [int(v) for v in raw]
    except ValueError:
        report.add(
            Level.ERROR,
            "answer column must be single letters (a, b, ...) or integer indices",
        )
        return None

    lo, hi = min(ints), max(ints)
    if lo == 0:
        if hi <= k - 1:
            return "index0"
        report.add(Level.ERROR, f"0-based answer index out of range 0..{k - 1} (found max {hi})")
        return None
    if lo >= 1 and hi <= k:
        # No zero ever appears -> cannot distinguish 1-based from sparse 0-based.
        report.add(
            Level.ERROR,
            "integer answers are ambiguous (no 0 present): cannot tell 0-based "
            "from 1-based. Re-encode as letters, or pass answer_format='index1' "
            "(or 'index0') explicitly.",
        )
        return None
    report.add(Level.ERROR, f"answer indices out of range for K={k} (found {lo}..{hi})")
    return None


def _answer_to_index(v: str, fmt: str, k: int) -> int | None:
    v = str(v).strip()
    if fmt == "letter":
        v = v.lower()
        if len(v) == 1 and "a" <= v <= chr(ord("a") + k - 1):
            return ord(v) - ord("a")
        return None
    try:
        n = int(v)
    except ValueError:
        return None
    if fmt == "index0":
        return n
    if fmt == "index1":
        return n - 1
    return None


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def validate_and_load(
    path: str,
    alphas,
    answer_format: str = "auto",
    cal_frac: float = 0.30,
    min_cal_recommended: int = 200,
) -> tuple[list[MCQRecord], int, ValidationReport]:
    """
    Validate a CSV against the contract and return (records, K, report).

    Raises CSVValidationError (aggregating all blocking errors) if the file is
    unusable. Warnings and info are attached to the returned report.

    Parameters
    ----------
    alphas : iterable of miscoverage levels the pipeline will sweep. The
             smallest one sets the hard calibration-size floor.
    """
    report = ValidationReport()

    # 1. read -----------------------------------------------------------------
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except FileNotFoundError:
        report.add(Level.ERROR, f"file not found: {path}")
        report.raise_if_errors()
    except Exception as e:  # pandas parser errors, encoding issues, etc.
        report.add(Level.ERROR, f"could not parse CSV: {e}")
        report.raise_if_errors()

    df.columns = [c.strip().lower() for c in df.columns]

    # 2. required columns + structure ----------------------------------------
    if "question" not in df.columns:
        report.add(Level.ERROR, "missing required column: question")
    if "answer" not in df.columns:
        report.add(Level.ERROR, "missing required column: answer")

    option_cols, opt_issues = _detect_option_columns(list(df.columns))
    report.issues.extend(opt_issues)
    report.raise_if_errors()  # cannot proceed without a valid column layout

    k = len(option_cols)
    report.k_options = k
    report.n_rows = len(df)

    if len(df) == 0:
        report.add(Level.ERROR, "file has a header but no data rows")
        report.raise_if_errors()

    # 3. per-row content ------------------------------------------------------
    for pos, (_, row) in enumerate(df.iterrows(), start=1):
        if not str(row["question"]).strip():
            report.add(Level.ERROR, "empty question", row=pos)
        for oc in option_cols:
            if not str(row[oc]).strip():
                report.add(Level.ERROR, f"empty option cell '{oc}'", row=pos)
        opts = [str(row[oc]).strip() for oc in option_cols]
        non_empty = [o for o in opts if o]
        if len(set(non_empty)) < len(non_empty):
            report.add(Level.WARNING, "duplicate option text in row", row=pos)

    # 4. answer format + validity --------------------------------------------
    fmt = answer_format
    if fmt == "auto":
        fmt = _detect_answer_format(df["answer"].tolist(), k, report)
    elif fmt not in ("letter", "index0", "index1"):
        report.add(Level.ERROR, f"unknown answer_format '{fmt}'")
        fmt = None
    report.answer_format = fmt or "invalid"
    report.raise_if_errors()

    answers_idx: list[int] = []
    for pos, v in enumerate(df["answer"].tolist(), start=1):
        idx = _answer_to_index(v, fmt, k)
        if idx is None or not (0 <= idx < k):
            report.add(Level.ERROR, f"answer '{str(v).strip()}' does not map to an option 0..{k - 1}", row=pos)
            answers_idx.append(-1)
        else:
            answers_idx.append(idx)
    report.raise_if_errors()

    # 5. dataset-level checks -------------------------------------------------
    dupes = int(df["question"].str.strip().duplicated().sum())
    if dupes:
        report.add(Level.WARNING, f"{dupes} duplicate question(s); may weaken the exchangeability assumption")

    a_min = min(alphas)
    n_cal_est = int(round(len(df) * cal_frac))
    hard_floor = ceil(1.0 / a_min)
    if n_cal_est < hard_floor:
        report.add(
            Level.ERROR,
            f"estimated calibration size {n_cal_est} < {hard_floor}: at alpha="
            f"{a_min} the conformal quantile is undefined and prediction sets "
            f"would always be full. Add rows or raise cal_frac.",
        )
    elif n_cal_est < min_cal_recommended:
        report.add(
            Level.WARNING,
            f"calibration size ~{n_cal_est} is small; empirical coverage can "
            f"fluctuate several points around 1-alpha. >= {min_cal_recommended} recommended.",
        )

    counts: dict[int, int] = {}
    for a in answers_idx:
        counts[a] = counts.get(a, 0) + 1
    report.class_counts = counts
    missing = [c for c in range(k) if counts.get(c, 0) == 0]
    if missing:
        report.add(
            Level.WARNING,
            f"option(s) {[chr(ord('a') + c) for c in missing]} are never the correct answer",
        )

    report.raise_if_errors()

    # 6. build normalised records --------------------------------------------
    records = [
        MCQRecord(
            question=str(df.iloc[i]["question"]).strip(),
            options=[str(df.iloc[i][oc]).strip() for oc in option_cols],
            answer_idx=answers_idx[i],
        )
        for i in range(len(df))
    ]
    return records, k, report


def split_calibration_eval(
    records: list[MCQRecord],
    cal_frac: float = 0.30,
    seed: int = 0,
    stratify: bool = True,
):
    """Reproducible (calibration, evaluation) split, stratified by answer."""
    from sklearn.model_selection import train_test_split

    labels = [r.answer_idx for r in records]
    strat = labels if (stratify and len(set(labels)) > 1) else None
    try:
        cal, ev = train_test_split(records, train_size=cal_frac, random_state=seed, stratify=strat)
    except ValueError:
        # too few per class to stratify -> fall back to a plain split
        cal, ev = train_test_split(records, train_size=cal_frac, random_state=seed, stratify=None)
    return cal, ev


def load_custom_dataset(
    csv_path: str,
    alphas,
    cal_frac: float = 0.30,
    seed: int = 0,
    answer_format: str = "auto",
    verbose: bool = True,
):
    """
    End-to-end entry point mirroring the existing dataset loaders.

    Returns (cal_records, eval_records, k, report). Wire this into data.py under
    a new `dataset: csv` branch, mapping MCQRecord onto whatever shape your ARC
    loader returns (fields: question, options, answer_idx).
    """
    records, k, report = validate_and_load(csv_path, alphas, answer_format, cal_frac)
    if verbose:
        print(report.summary())
    cal, ev = split_calibration_eval(records, cal_frac, seed)
    return cal, ev, k, report


# --------------------------------------------------------------------------- #
# CLI: validate a file and print every problem at once
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse
    import sys

    p = argparse.ArgumentParser(description="Validate a calibration-layer CSV.")
    p.add_argument("path")
    p.add_argument("--answer-format", default="auto", choices=["auto", "letter", "index0", "index1"])
    p.add_argument("--alpha", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    p.add_argument("--cal-frac", type=float, default=0.30)
    args = p.parse_args()

    try:
        records, k, report = validate_and_load(
            args.path, args.alpha, args.answer_format, args.cal_frac
        )
        print(report.summary())
        print(f"\nOK: {len(records)} valid records, K={k}.")
    except CSVValidationError as e:
        print(e)
        sys.exit(1)
