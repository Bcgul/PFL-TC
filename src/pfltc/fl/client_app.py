"""Flower ClientApp: one vehicle of the federation.

Each client keeps its data and, with ``strategy = "fedper"``, its personal
layers (zone branches and output head) in ``context.state``; only the shared
layers are sent to the server.
"""

from __future__ import annotations

import numpy as np
from flwr.app import Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from ..data import denormalize_target
from ..models import forward, get_named_weights, set_named_weights, split_weights
from ..train import round_seed, set_seed
from .task import PERSONAL_STATE_KEY, client_partition, config_from_run, from_record, make_model, to_record

app = ClientApp()


def _setup(msg: Message, context: Context):
    cfg = config_from_run(context.run_config)
    parts = client_partition(cfg, int(context.node_config["partition-id"]), int(context.node_config["num-partitions"]))
    model, groups, personal = make_model(cfg)
    weights = get_named_weights(model)  # seeded initialisation (first round)
    weights.update(from_record(msg.content["arrays"]))  # global shared layers
    if PERSONAL_STATE_KEY in context.state:
        weights.update(from_record(context.state[PERSONAL_STATE_KEY]))  # this client's personal layers
    set_named_weights(model, weights)
    return cfg, parts, model, groups, personal


@app.train()
def train(msg: Message, context: Context) -> Message:
    cfg, parts, model, groups, personal = _setup(msg, context)
    # Different (but reproducible) batch shuffling per round and client.
    server_round = int(msg.content["config"]["server-round"])
    set_seed(round_seed(cfg["seed"], server_round, int(context.node_config["partition-id"])))
    data = parts["train"]
    history = model.fit(
        data.inputs(groups),
        data.y,
        epochs=int(cfg["federated"]["local_epochs"]),
        batch_size=int(cfg["model"]["batch_size"]),
        shuffle=True,
        verbose=0,
    )
    shared, local = split_weights(get_named_weights(model), personal)
    if local:
        context.state[PERSONAL_STATE_KEY] = to_record(local)
    metrics = MetricRecord({"train_loss": float(history.history["loss"][-1]), "num-examples": len(data)})
    return Message(content=RecordDict({"arrays": to_record(shared), "metrics": metrics}), reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    cfg, parts, model, groups, _ = _setup(msg, context)
    out: dict[str, float] = {}
    for split in ("val", "test"):
        data = parts[split]
        pred = forward(model, data.inputs(groups))
        err_c = denormalize_target(pred, cfg["normalization"]) - data.y_c
        out[f"{split}_loss"] = float(np.mean((pred - data.y) ** 2))
        out[f"{split}_mae_c"] = float(np.mean(np.abs(err_c)))
        out[f"{split}_rmse_c"] = float(np.sqrt(np.mean(err_c**2)))
    out["num-examples"] = len(parts["val"])
    return Message(content=RecordDict({"metrics": MetricRecord(out)}), reply_to=msg)
