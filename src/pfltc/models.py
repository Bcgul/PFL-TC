"""LSTM models for the three configurations of the paper (Fig. 3, Sec. IV-C).

* ``basic``                 : environmental inputs -> LSTM(50) -> Dense
* ``personalized_baseline`` : environmental + all zone inputs, concatenated
                              ("undifferentiated") -> LSTM(50) -> Dense
* ``personalized_model``    : one LSTM branch per input group,
                              LSTM(10) for the environment and LSTM(20) per
                              zone, concatenated -> Dense

All models predict one value per target zone. Layer names are fixed, because
federated personalisation (FedPer) decides by name which layers stay local.
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras  # noqa: E402
import numpy as np  # noqa: E402
from keras import layers  # noqa: E402

from .data import groups_for_variant  # noqa: E402


def build_model(variant: str, group_dims: dict[str, int], seq_len: int, n_outputs: int, cfg: dict):
    mcfg = cfg["model"]
    groups = groups_for_variant(variant, list(group_dims))
    inputs = {g: keras.Input(shape=(seq_len, group_dims[g]), name=g) for g in groups}

    if variant in ("basic", "personalized_baseline"):
        tensors = [inputs[g] for g in groups]
        x = tensors[0] if len(tensors) == 1 else layers.Concatenate(axis=-1, name="concat_inputs")(tensors)
        x = layers.LSTM(int(mcfg["lstm_units"]), name="lstm")(x)
    else:  # personalized_model
        branches = [layers.LSTM(int(mcfg["base_units"]), name="base_lstm")(inputs["basic"])]
        for g in groups:
            if g != "basic":
                branches.append(layers.LSTM(int(mcfg["zone_units"]), name=f"{g}_lstm")(inputs[g]))
        x = branches[0] if len(branches) == 1 else layers.Concatenate(name="concat_branches")(branches)

    outputs = layers.Dense(n_outputs, name="head")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name=variant)
    compile_model(model, cfg)
    return model


def compile_model(model, cfg: dict) -> None:
    """(Re)compile with a fresh Adam optimiser and MSE loss."""
    model.compile(optimizer=keras.optimizers.Adam(float(cfg["model"]["learning_rate"])), loss="mse")


def reset_optimizer(model) -> None:
    """Zero the optimiser state (step counter and Adam moments).

    Equivalent to starting from a fresh optimiser, as a stateless Flower client
    does every round, but without re-tracing the training graph."""
    optimizer = model.optimizer
    if not getattr(optimizer, "built", False):
        return
    for var in optimizer.variables:
        if var.path.endswith("learning_rate"):
            continue
        var.assign(np.zeros(var.shape, dtype=var.dtype))


def forward(model, inputs: dict[str, np.ndarray]) -> np.ndarray:
    """Inference without ``model.predict`` (avoids re-tracing on small arrays)."""
    return np.asarray(model(inputs, training=False))


# --------------------------------------------------------------------------- #
# Named weights (used for federated aggregation)
# --------------------------------------------------------------------------- #
def get_named_weights(model) -> dict[str, np.ndarray]:
    """All trainable state as ``{"<layer>/<index>": array}``."""
    out: dict[str, np.ndarray] = {}
    for layer in model.layers:
        for i, w in enumerate(layer.get_weights()):
            out[f"{layer.name}/{i}"] = np.asarray(w)
    return out


def set_named_weights(model, named: dict[str, np.ndarray], strict: bool = True) -> None:
    """Load weights by name; layers absent from ``named`` keep their values
    unless ``strict`` is True."""
    for layer in model.layers:
        current = layer.get_weights()
        if not current:
            continue
        keys = [f"{layer.name}/{i}" for i in range(len(current))]
        present = [k in named for k in keys]
        if all(present):
            layer.set_weights([named[k] for k in keys])
        elif any(present) or strict:
            missing = [k for k, p in zip(keys, present, strict=True) if not p]
            raise KeyError(f"missing weights for layer {layer.name!r}: {missing}")


def personal_layers(model, strategy: str) -> set[str]:
    """Layers that stay on the client.

    ``fedavg``: none, everything is averaged.
    ``fedper``: the output head and every zone branch (FedPer, Arivazhagan et
                al., 2019); the environmental part is shared.
    ``local`` : everything (no federation).
    """
    names = {layer.name for layer in model.layers if layer.get_weights()}
    if strategy == "fedavg":
        return set()
    if strategy == "fedper":
        return {n for n in names if n == "head" or (n.startswith("zone") and n.endswith("_lstm"))}
    if strategy == "local":
        return names
    raise ValueError(f"unknown strategy {strategy!r}")


def split_weights(named: dict[str, np.ndarray], personal: set[str]):
    """Split named weights into (shared, personal) dictionaries."""
    shared = {k: v for k, v in named.items() if k.split("/")[0] not in personal}
    local = {k: v for k, v in named.items() if k.split("/")[0] in personal}
    return shared, local
