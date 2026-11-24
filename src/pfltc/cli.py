"""Command-line interface.

    pfltc train        one model (variant x regime), prints test metrics
    pfltc experiments  all variants x regimes x seeds, writes results/
    pfltc tune         grid search on the validation split (centralised)

Common options: ``--config file.yaml`` and ``--set key.subkey=value``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import __version__
from .config import apply_override, load_config
from .data import VARIANTS, prepare
from .train import REGIMES, naive_baselines, run_regime

SUMMARY_METRICS = ["mse", "rmse", "mae", "mape", "r2"]


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="YAML file merged onto the packaged defaults")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override one config value, e.g. --set windows.seq_len=20 (repeatable)",
    )


def _environment() -> dict:
    import flwr
    import tensorflow as tf

    return {
        "pfltc": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "flwr": flwr.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation over seeds for every (variant, regime)."""
    cols = [c for c in SUMMARY_METRICS + ["train_seconds", "best_step"] if c in runs.columns]
    grouped = runs.groupby(["variant", "regime"], sort=False)[cols]
    summary = grouped.mean().add_suffix("_mean").join(grouped.std(ddof=1).add_suffix("_std"))
    summary["n_seeds"] = grouped.size()
    return summary.reset_index()


def _fmt(row: pd.Series, metric: str, digits: int = 3) -> str:
    mean, std = row.get(f"{metric}_mean", np.nan), row.get(f"{metric}_std", np.nan)
    if pd.isna(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def summary_markdown(summary: pd.DataFrame) -> str:
    lines = [
        "| Variant | Regime | MSE (K²) | RMSE (K) | MAE (K) | MAPE (%) | R² | Train time (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['variant']} | {r['regime']} | {_fmt(r, 'mse')} | {_fmt(r, 'rmse')} | {_fmt(r, 'mae')} | "
            f"{_fmt(r, 'mape', 2)} | {_fmt(r, 'r2')} | {_fmt(r, 'train_seconds', 1)} |"
        )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
def cmd_train(args: argparse.Namespace) -> None:
    cfg = load_config(args.config, args.overrides)
    seed = cfg["seed"] if args.seed is None else args.seed
    ws = prepare(cfg, seed=seed)
    result = run_regime(ws, args.variant, args.regime, cfg, seed, verbose=args.verbose)
    print(json.dumps(result.row(), indent=2))
    if args.predictions:
        Path(args.predictions).parent.mkdir(parents=True, exist_ok=True)
        result.predictions.to_csv(args.predictions, index=False)
        print(f"predictions written to {args.predictions}")


def cmd_experiments(args: argparse.Namespace) -> None:
    cfg = load_config(args.config, args.overrides)
    ecfg = cfg["experiment"]
    out = Path(args.output or ecfg["output_dir"])
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    (out / "histories").mkdir(parents=True, exist_ok=True)
    with open(out / "config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    with open(out / "environment.json", "w", encoding="utf-8") as fh:
        json.dump(_environment(), fh, indent=2)

    rows, naive_rows = [], []
    combos = list(itertools.product(ecfg["seeds"], ecfg["variants"], ecfg["regimes"]))
    start = time.perf_counter()
    prepared: dict[int, object] = {}
    for i, (seed, variant, regime) in enumerate(combos, 1):
        if seed not in prepared:
            prepared[seed] = prepare(cfg, seed=seed)
            for name, metrics in naive_baselines(prepared[seed], cfg).items():
                naive_rows.append({"regime": name, "variant": "-", "seed": seed, **metrics})
        result = run_regime(prepared[seed], variant, regime, cfg, seed)
        rows.append(result.row())
        tag = f"{variant}__{regime}__seed{seed}"
        result.predictions.to_csv(out / "predictions" / f"{tag}.csv", index=False)
        pd.DataFrame(result.history).to_csv(out / "histories" / f"{tag}.csv", index=False)
        print(
            f"[{i}/{len(combos)}] seed={seed} {variant:22s} {regime:11s} "
            f"MAE={result.metrics['mae']:.3f} RMSE={result.metrics['rmse']:.3f} "
            f"R2={result.metrics['r2']:+.3f} ({result.train_seconds:.1f}s)",
            flush=True,
        )

    runs = pd.DataFrame(rows)
    runs.to_csv(out / "runs.csv", index=False)
    naive = pd.DataFrame(naive_rows)
    naive.to_csv(out / "naive_baselines.csv", index=False)
    summary = pd.concat([summarise(runs), summarise(naive.assign(train_seconds=0.0))], ignore_index=True)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "summary.md").write_text(summary_markdown(summary), encoding="utf-8")
    print(f"\nfinished {len(combos)} runs in {time.perf_counter() - start:.0f}s\n")
    print(summary_markdown(summary))


def cmd_tune(args: argparse.Namespace) -> None:
    """Grid search scored on the validation split only (the test split is not used)."""
    base = load_config(args.config, args.overrides)
    grid = {}
    for item in args.grid:
        key, values = item.split("=", 1)
        grid[key] = [yaml.safe_load(v) for v in values.split(",")]
    keys = list(grid)
    seeds = args.seeds or base["experiment"]["seeds"][:3]
    rows = []
    for values in itertools.product(*(grid[k] for k in keys)):
        cfg = load_config(args.config, args.overrides)
        for k, v in zip(keys, values, strict=True):
            apply_override(cfg, f"{k}={json.dumps(v)}")
        for seed in seeds:
            ws = prepare(cfg, seed=seed)
            result = run_regime(ws, args.variant, args.regime, cfg, seed)
            rows.append(
                {
                    **dict(zip(keys, values, strict=True)),
                    "seed": seed,
                    "val_loss": result.best_val_loss,
                    "best_step": result.best_step,
                }
            )
            print(rows[-1], flush=True)
    table = (
        pd.DataFrame(rows)
        .groupby(keys, as_index=False)
        .agg(val_loss=("val_loss", "mean"), val_loss_std=("val_loss", "std"), best_step=("best_step", "mean"))
        .sort_values("val_loss")
    )
    print(table.to_string(index=False))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="pfltc", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--version", action="version", version=f"pfltc {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="train and evaluate one model")
    _add_common(p)
    p.add_argument("--variant", choices=VARIANTS, default="personalized_model")
    p.add_argument("--regime", choices=REGIMES, default="fedper")
    p.add_argument("--seed", type=int)
    p.add_argument("--predictions", help="CSV file for the test predictions")
    p.add_argument("--verbose", type=int, default=0)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("experiments", help="run the full comparison")
    _add_common(p)
    p.add_argument("--output", help="output directory (default: experiment.output_dir)")
    p.set_defaults(func=cmd_experiments)

    p = sub.add_parser("tune", help="grid search on the validation split")
    _add_common(p)
    p.add_argument("--variant", choices=VARIANTS, default="personalized_baseline")
    p.add_argument("--regime", choices=REGIMES, default="centralized")
    p.add_argument(
        "--grid",
        action="append",
        default=[],
        metavar="KEY=V1,V2",
        help="e.g. --grid windows.seq_len=1,10,30 (repeatable)",
    )
    p.add_argument("--seeds", type=int, nargs="*")
    p.add_argument("--output", help="CSV file for the tuning table")
    p.set_defaults(func=cmd_tune)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
