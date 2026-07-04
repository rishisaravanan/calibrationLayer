"""Publication-legible figures: reliability diagram and risk-coverage curve."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .calibration import reliability_data

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    }
)


def plot_reliability(
    conf_raw: np.ndarray,
    conf_cal: np.ndarray,
    correct: np.ndarray,
    ece_raw: float,
    ece_cal: float,
    temperature: float,
    path: str | Path,
    n_bins: int = 15,
) -> None:
    """Two-panel reliability diagram: before vs after temperature scaling."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    panels = [
        (axes[0], conf_raw, f"Before calibration\nECE = {ece_raw:.3f}"),
        (axes[1], conf_cal, f"After temperature scaling (T = {temperature:.2f})\nECE = {ece_cal:.3f}"),
    ]
    width = 1.0 / n_bins
    for ax, conf, title in panels:
        rows = reliability_data(conf, correct, n_bins)
        centers = (np.arange(n_bins) + 0.5) * width
        acc = np.array([r[1] for r in rows])
        counts = np.array([r[2] for r in rows], dtype=float)
        filled = counts > 0
        ax.bar(
            centers[filled],
            acc[filled],
            width=width * 0.92,
            color="#4878cf",
            edgecolor="white",
            label="accuracy in bin",
        )
        # sample-share overlay so sparse bins are visibly sparse
        ax.bar(
            centers[filled],
            (counts[filled] / counts.sum()),
            width=width * 0.92,
            color="#c44e52",
            alpha=0.35,
            label="share of samples",
        )
        ax.plot([0, 1], [0, 1], ls="--", c="black", lw=1, label="perfect calibration")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted confidence")
        ax.set_title(title)
    axes[0].set_ylabel("Observed accuracy")
    axes[0].legend(loc="upper left", frameon=False, fontsize=10)
    fig.suptitle("Reliability: does 0.9 confidence mean right 90% of the time?", y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_risk_coverage(
    curves: dict[str, tuple[np.ndarray, np.ndarray, float]],
    headline: dict,
    path: str | Path,
) -> None:
    """Risk-coverage curves (label -> (coverage, risk, aurc)) with the
    headline point annotated on the calibrated curve."""
    fig, ax = plt.subplots(figsize=(8, 5.2))
    colors = {"raw": "#c44e52", "calibrated": "#4878cf"}
    for label, (cov, risk, area) in curves.items():
        ax.plot(
            cov,
            risk * 100,
            lw=2.2,
            color=colors.get(label, None),
            label=f"{label} confidence (AURC = {area:.4f})",
        )
    hc = headline["coverage"]
    hr = headline["error_calibrated"] * 100
    ymax = max(float(np.max(r)) for _, r, _ in curves.values()) * 100
    ax.plot([hc], [hr], "o", ms=9, color="#4878cf", zorder=5)
    ax.annotate(
        f"at {hc:.0%} coverage:\n"
        f"{headline['error_full_coverage']:.1%} → {headline['error_calibrated']:.1%} error",
        xy=(hc, hr),
        xytext=(hc - 0.28, min(hr + 0.35 * ymax, 0.82 * ymax)),
        ha="center",
        fontsize=11,
        arrowprops=dict(arrowstyle="->", lw=1.2),
    )
    ax.set_xlabel("Coverage (fraction of questions answered)")
    ax.set_ylabel("Risk (% wrong among answered)")
    ax.set_title("Answer only when confident: risk vs coverage")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
