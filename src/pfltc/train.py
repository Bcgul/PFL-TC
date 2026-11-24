"""Centralised, local-only and federated (FedAvg / FedPer) training.

The federated loop below is an in-process simulation of Flower's FedAvg: every
round, each client starts from the current global (shared) weights plus its own
personal layers, trains ``local_epochs`` epochs with a fresh optimiser, and the
server averages the shared weights weighted by the number of training windows
(McMahan et al., 2017). The Flower apps in ``pfltc.fl`` run the same client
code over the network and reproduce the losses of this simulation for the same
seed; the simulation is used for the experiments because it is faster and needs
no Flower processes.
"""

from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras  # noqa: E402

from .data import WindowSet, denormalize_target, groups_for_variant  # noqa: E402
from .metrics import regression_metrics  # noqa: E402
from .models import (  # noqa: E402
    build_model,
    forward,
    get_named_weights,
    personal_layers,
    reset_optimizer,
    set_named_weights,
    split_weights,
)

REGIMES = ("centralized", "local", "fedavg", "fedper")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and TensorFlow and make TF ops deterministic."""
    keras.utils.set_random_seed(int(seed))
    try:
        import tensorflow as tf

        tf.config.experimental.enable_op_determinism()
    except (ImportError, AttributeError):  # pragma: no cover - other backends
        pass


@dataclass
class RunResult:
    regime: str
    variant: str
    seed: int
    metrics: dict[str, float]  # test metrics, degC
    best_val_loss: float  # normalised MSE on the validation windows
    best_step: int  # best epoch (centralised) or round (federated)
    train_seconds: float
    history: list[dict] = field(default_factory=list)
    predictions: pd.DataFrame | None = None

    def row(self) -> dict:
        return {
            "regime": self.regime,
            "variant": self.variant,
            "seed": self.seed,
            "best_val_loss": self.best_val_loss,
            "best_step": self.best_step,
            "train_seconds": self.train_seconds,
            **self.metrics,
        }


def _predict(model, ws: WindowSet, groups: list[str], cfg: dict) -> np.ndarray:
    return denormalize_target(forward(model, ws.inputs(groups)), cfg["normalization"])


def _mse(model, ws: WindowSet, groups: list[str]) -> float:
    """Validation loss on the normalised scale (same as the training loss)."""
    return float(np.mean((forward(model, ws.inputs(groups)) - ws.y) ** 2))


def _prediction_frame(ws: WindowSet, y_pred_c: np.ndarray) -> pd.DataFrame:
    frame = ws.meta.copy()
    for j, name in enumerate(ws.target_names):
        frame[f"{name}_true"] = ws.y_c[:, j]
        frame[f"{name}_pred"] = y_pred_c[:, j]
    return frame


class _KeepBest(keras.callbacks.Callback):
    """Remember the weights of the epoch with the lowest validation loss."""

    def __init__(self) -> None:
        super().__init__()
        self.best = np.inf
        self.best_epoch = 0
        self.weights = None

    def on_epoch_end(self, epoch, logs=None):
        value = (logs or {}).get("val_loss")
        if value is not None and value < self.best:
            self.best, self.best_epoch = float(value), epoch + 1
            self.weights = self.model.get_weights()


# --------------------------------------------------------------------------- #
# Centralised learning (CL): all training windows pooled in one place
# --------------------------------------------------------------------------- #
def train_centralized(ws: WindowSet, variant: str, cfg: dict, seed: int, verbose: int = 0) -> RunResult:
    set_seed(seed)
    groups = groups_for_variant(variant, list(ws.X))
    train, val, test = ws.part("train"), ws.part("val"), ws.part("test")
    model = build_model(variant, ws.group_dims, cfg["windows"]["seq_len"], len(ws.target_names), cfg)
    keep = _KeepBest()
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=int(cfg["centralized"]["patience"]))
    start = time.perf_counter()
    hist = model.fit(
        train.inputs(groups),
        train.y,
        validation_data=(val.inputs(groups), val.y),
        epochs=int(cfg["centralized"]["epochs"]),
        batch_size=int(cfg["model"]["batch_size"]),
        shuffle=True,
        callbacks=[keep, stop],
        verbose=verbose,
    )
    seconds = time.perf_counter() - start
    model.set_weights(keep.weights)
    y_pred = _predict(model, test, groups, cfg)
    history = [
        {"epoch": i + 1, "loss": float(loss), "val_loss": float(v)}
        for i, (loss, v) in enumerate(zip(hist.history["loss"], hist.history["val_loss"], strict=True))
    ]
    return RunResult(
        regime="centralized",
        variant=variant,
        seed=seed,
        metrics=regression_metrics(test.y_c, y_pred, ws.target_names),
        best_val_loss=keep.best,
        best_step=keep.best_epoch,
        train_seconds=seconds,
        history=history,
        predictions=_prediction_frame(test, y_pred),
    )


# --------------------------------------------------------------------------- #
# Federated learning (FL) simulation: FedAvg, FedPer, or local-only training
# --------------------------------------------------------------------------- #
def round_seed(seed: int, server_round: int, client: int) -> int:
    """Seed for one client's local training in one round (shared with the Flower app)."""
    return 1000 * int(seed) + 10 * int(server_round) + int(client)


def fedavg(updates: list[dict[str, np.ndarray]], sizes: list[int]) -> dict[str, np.ndarray]:
    """Weighted average f(w) = sum_k (n_k / n) F_k(w), Eq. (1) of the paper."""
    total = float(sum(sizes))
    return {key: sum(u[key] * (n / total) for u, n in zip(updates, sizes, strict=True)) for key in updates[0]}


def train_federated(
    ws: WindowSet, variant: str, cfg: dict, seed: int, strategy: str | None = None, verbose: int = 0
) -> RunResult:
    fcfg = cfg["federated"]
    strategy = strategy or fcfg["strategy"]
    set_seed(seed)
    groups = groups_for_variant(variant, list(ws.X))
    client_ids = sorted(int(c) for c in ws.meta["client"].unique())
    parts = {
        c: {
            s: ws.subset((ws.meta["client"] == c).to_numpy() & (ws.meta["split"] == s).to_numpy())
            for s in ("train", "val", "test")
        }
        for c in client_ids
    }
    model = build_model(variant, ws.group_dims, cfg["windows"]["seq_len"], len(ws.target_names), cfg)
    personal = personal_layers(model, strategy)
    shared, local_init = split_weights(get_named_weights(model), personal)
    local = {c: copy.deepcopy(local_init) for c in client_ids}

    best = {"val": np.inf, "round": 0, "shared": copy.deepcopy(shared), "local": copy.deepcopy(local)}
    history: list[dict] = []
    start = time.perf_counter()
    for rnd in range(1, int(fcfg["rounds"]) + 1):
        round_start = time.perf_counter()
        updates, sizes, train_losses = [], [], []
        for c in client_ids:
            train = parts[c]["train"]
            if len(train) == 0:
                continue
            set_named_weights(model, {**shared, **local[c]})
            reset_optimizer(model)  # fresh optimiser state, as a stateless Flower client
            set_seed(round_seed(seed, rnd, c))  # same batch order as the Flower ClientApp
            hist = model.fit(
                train.inputs(groups),
                train.y,
                epochs=int(fcfg["local_epochs"]),
                batch_size=int(cfg["model"]["batch_size"]),
                shuffle=True,
                verbose=0,
            )
            new_shared, local[c] = split_weights(get_named_weights(model), personal)
            updates.append(new_shared)
            sizes.append(len(train))
            train_losses.append(hist.history["loss"][-1])
        if shared:
            shared = fedavg(updates, sizes)

        val_losses, val_sizes = [], []
        for c in client_ids:
            val = parts[c]["val"]
            if len(val) == 0:
                continue
            set_named_weights(model, {**shared, **local[c]})
            val_losses.append(_mse(model, val, groups))
            val_sizes.append(len(val))
        val_loss = float(np.average(val_losses, weights=val_sizes))
        history.append(
            {
                "round": rnd,
                "train_loss": float(np.average(train_losses, weights=sizes)),
                "val_loss": val_loss,
                "seconds": time.perf_counter() - round_start,
            }
        )
        if verbose:
            print(f"[{strategy}] round {rnd:3d}  train {history[-1]['train_loss']:.4f}  val {val_loss:.4f}")
        if val_loss < best["val"]:
            best = {"val": val_loss, "round": rnd, "shared": copy.deepcopy(shared), "local": copy.deepcopy(local)}
    seconds = time.perf_counter() - start

    frames, y_true, y_pred = [], [], []
    for c in client_ids:
        test = parts[c]["test"]
        if len(test) == 0:
            continue
        set_named_weights(model, {**best["shared"], **best["local"][c]})
        pred = _predict(model, test, groups, cfg)
        frames.append(_prediction_frame(test, pred))
        y_true.append(test.y_c)
        y_pred.append(pred)
    y_true_all, y_pred_all = np.concatenate(y_true), np.concatenate(y_pred)
    return RunResult(
        regime=strategy if strategy != "local" else "local",
        variant=variant,
        seed=seed,
        metrics=regression_metrics(y_true_all, y_pred_all, ws.target_names),
        best_val_loss=best["val"],
        best_step=best["round"],
        train_seconds=seconds,
        history=history,
        predictions=pd.concat(frames, ignore_index=True),
    )


def run_regime(ws: WindowSet, variant: str, regime: str, cfg: dict, seed: int, verbose: int = 0) -> RunResult:
    if regime == "centralized":
        return train_centralized(ws, variant, cfg, seed, verbose)
    if regime in ("local", "fedavg", "fedper"):
        return train_federated(ws, variant, cfg, seed, strategy=regime, verbose=verbose)
    raise ValueError(f"unknown regime {regime!r}; choose from {REGIMES}")


# --------------------------------------------------------------------------- #
# Reference predictors without learning
# --------------------------------------------------------------------------- #
def naive_baselines(ws: WindowSet, cfg: dict) -> dict[str, dict[str, float]]:
    """``train_mean``: mean training target per zone.
    ``cabin_temperature`` (Griffiths target only): predict the current cabin
    temperature, i.e. assume every occupant is thermally neutral."""
    train, test = ws.part("train"), ws.part("test")
    out = {
        "train_mean": regression_metrics(
            test.y_c, np.broadcast_to(train.y_c.mean(axis=0), test.y_c.shape), ws.target_names
        )
    }
    if cfg["target"]["mode"] == "griffiths":
        cabin = np.repeat(test.meta["cabin_temp_c"].to_numpy()[:, None], len(ws.target_names), axis=1)
        out["cabin_temperature"] = regression_metrics(test.y_c, cabin, ws.target_names)
    return out
