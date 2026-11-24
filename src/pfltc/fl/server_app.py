"""Flower ServerApp: FedAvg over the shared layers.

With ``strategy = "fedavg"`` every layer is shared; with ``"fedper"`` only the
environmental part is, while zone branches and the output head stay on the
vehicles. Aggregation is weighted by the number of training windows.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from flwr.app import ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from ..models import get_named_weights, split_weights
from .task import config_from_run, from_record, make_model, to_record

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    run_config = context.run_config
    cfg = config_from_run(run_config)
    model, _, personal = make_model(cfg)
    shared, _ = split_weights(get_named_weights(model), personal)
    if not shared:
        raise ValueError("strategy 'local' has no shared layers; run it with `pfltc train --regime local`")

    n_clients = int(run_config["num-clients"])
    strategy = FedAvg(
        fraction_train=1.0,
        fraction_evaluate=1.0,
        min_train_nodes=n_clients,
        min_evaluate_nodes=n_clients,
        min_available_nodes=n_clients,
    )
    start = time.perf_counter()
    result = strategy.start(
        grid=grid,
        initial_arrays=to_record(shared),
        num_rounds=int(cfg["federated"]["rounds"]),
        train_config=ConfigRecord({"local-epochs": int(cfg["federated"]["local_epochs"])}),
    )
    seconds = time.perf_counter() - start
    print(f"Federated training finished in {seconds:.1f}s")

    out_dir = run_config.get("output-dir")
    if out_dir:
        out = Path(str(out_dir)).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        np.savez(
            out / "global_shared_weights.npz",
            **{k.replace("/", "__"): v for k, v in from_record(result.arrays).items()},
        )
        history = {
            "seconds": seconds,
            "train": {int(r): dict(m) for r, m in result.train_metrics_clientapp.items()},
            "evaluate": {int(r): dict(m) for r, m in result.evaluate_metrics_clientapp.items()},
        }
        (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"Saved global weights and metrics to {out}")
