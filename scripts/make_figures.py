"""Figures for the README from the outputs of `pfltc experiments`.

Usage:
    python scripts/make_figures.py --results results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Validated categorical palette (light surface), fixed slot order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

REGIME_LABEL = {"centralized": "Centralised", "local": "Local only", "fedavg": "FedAvg", "fedper": "FedPer"}
VARIANT_LABEL = {
    "basic": "Basic baseline",
    "personalized_baseline": "Personalised baseline",
    "personalized_model": "Personalised model",
}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK_2, length=0, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def mae_by_variant(results: Path, out: Path) -> None:
    runs = pd.read_csv(results / "runs.csv")
    naive = pd.read_csv(results / "naive_baselines.csv")
    variants = [v for v in VARIANT_LABEL if v in runs["variant"].unique()]
    regimes = [r for r in REGIME_LABEL if r in runs["regime"].unique()]
    stats = runs.groupby(["variant", "regime"])["mae"].agg(["mean", "std"])

    fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=150, facecolor=SURFACE)
    _style(ax)
    width = 0.19
    gap = 0.012  # surface gap between touching bars
    x = np.arange(len(variants))
    for i, regime in enumerate(regimes):
        offset = (i - (len(regimes) - 1) / 2) * width
        means = [stats.loc[(v, regime), "mean"] for v in variants]
        stds = [stats.loc[(v, regime), "std"] for v in variants]
        ax.bar(x + offset, means, width - gap, color=SERIES[i], label=REGIME_LABEL[regime], zorder=2)
        ax.errorbar(x + offset, means, yerr=stds, fmt="none", ecolor=INK_2, elinewidth=1, capsize=0, zorder=3)

    for name, label in (("train_mean", "predict training mean"), ("cabin_temperature", "predict cabin temperature")):
        if name in naive["regime"].unique():
            level = naive.loc[naive["regime"] == name, "mae"].mean()
            ax.axhline(level, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
            ax.annotate(
                f"{label}: {level:.2f} K",
                xy=(x[-1] + 0.5, level),
                xytext=(6, 0),
                textcoords="offset points",
                color=INK_2,
                fontsize=8,
                va="center",
                ha="left",
                annotation_clip=False,
            )

    ax.set_xticks(x, [VARIANT_LABEL[v] for v in variants], color=INK)
    ax.set_ylabel("Test MAE (K)", color=INK_2, fontsize=9)
    ax.set_xlim(-0.5, len(variants) - 0.5)
    ax.set_title(
        "Test MAE per model configuration and training regime (mean ± s.d. over seeds)",
        color=INK,
        fontsize=10,
        loc="left",
    )
    ax.legend(frameon=False, ncol=len(regimes), fontsize=9, loc="upper left", bbox_to_anchor=(0, -0.08), labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out / "mae_by_variant.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def example_predictions(results: Path, out: Path, variant: str, regime: str, seed: int, zone_col: str) -> None:
    path = results / "predictions" / f"{variant}__{regime}__seed{seed}.csv"
    if not path.exists():
        print(f"skip example plot: {path} not found")
        return
    pred = pd.read_csv(path, parse_dates=["timestamp"])
    drives = sorted(pred["drive_id"].unique())
    fig, axes = plt.subplots(1, len(drives), figsize=(10, 2.8), dpi=150, sharey=True, facecolor=SURFACE)
    for ax, drive in zip(np.atleast_1d(axes), drives, strict=False):
        _style(ax)
        d = pred[pred["drive_id"] == drive].sort_values("elapsed_s")
        # Break the lines where the logger paused, instead of bridging the gap.
        gap = d["elapsed_s"].diff() > 60.0
        d = d.astype({f"{zone_col}_true": float, f"{zone_col}_pred": float})
        d.loc[gap, [f"{zone_col}_true", f"{zone_col}_pred"]] = np.nan
        minutes = d["elapsed_s"] / 60.0
        ax.plot(minutes, d[f"{zone_col}_true"], color=SERIES[0], linewidth=2, label="Target (from vote)")
        ax.plot(minutes, d[f"{zone_col}_pred"], color=SERIES[1], linewidth=2, label="Prediction")
        ax.set_title(f"Drive {drive}", color=INK, fontsize=9, loc="left")
        ax.set_xlabel("Minutes since start of drive", color=INK_2, fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel("Neutral temperature (°C)", color=INK_2, fontsize=8)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, fontsize=9, loc="upper right", labelcolor=INK)
    fig.suptitle(
        f"Driver seat, test part of every drive: {VARIANT_LABEL[variant]}, {REGIME_LABEL[regime]}, seed {seed}",
        color=INK,
        fontsize=10,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(out / "example_predictions.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--variant", default="basic")
    parser.add_argument("--regime", default="fedavg")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--zone-col", default="target_z1_c")
    args = parser.parse_args()
    out = args.results / "figures"
    out.mkdir(parents=True, exist_ok=True)
    mae_by_variant(args.results, out)
    example_predictions(args.results, out, args.variant, args.regime, args.seed, args.zone_col)
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
