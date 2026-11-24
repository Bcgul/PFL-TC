"""Helpers shared by the Flower ServerApp and ClientApp."""

from __future__ import annotations

import functools
import json

import numpy as np
from flwr.app import Array, ArrayRecord

from ..config import load_config, resolve_path
from ..data import WindowSet, feature_groups, groups_for_variant, prepare
from ..models import build_model, personal_layers
from ..train import set_seed

PERSONAL_STATE_KEY = "personal-weights"


def config_from_run(run_config: dict) -> dict:
    """Packaged defaults + optional YAML + the Flower run config."""
    cfg = load_config(run_config.get("config-file") or None)
    if run_config.get("data-path"):
        cfg["data"]["path"] = str(run_config["data-path"])
    cfg["federated"]["rounds"] = int(run_config["num-server-rounds"])
    cfg["federated"]["local_epochs"] = int(run_config["local-epochs"])
    cfg["federated"]["strategy"] = str(run_config["strategy"])
    cfg["seed"] = int(run_config["seed"])
    cfg["_variant"] = str(run_config["variant"])
    return cfg


def group_dims(cfg: dict) -> dict[str, int]:
    """Input dimensions known from the configuration alone (no data needed)."""
    return {g: len(cols) for g, cols in feature_groups(cfg).items()}


def n_targets(cfg: dict) -> int:
    t = cfg["target"]
    return len(t["zones"]) if t["mode"] == "griffiths" else len(t["columns"])


def make_model(cfg: dict):
    """Model with a seeded initialisation, identical on server and clients."""
    set_seed(cfg["seed"])
    variant = cfg["_variant"]
    model = build_model(variant, group_dims(cfg), cfg["windows"]["seq_len"], n_targets(cfg), cfg)
    groups = groups_for_variant(variant, list(group_dims(cfg)))
    return model, groups, personal_layers(model, cfg["federated"]["strategy"])


@functools.lru_cache(maxsize=2)
def _windows(cfg_json: str) -> WindowSet:
    return prepare(json.loads(cfg_json))


def client_partition(cfg: dict, partition_id: int, num_partitions: int) -> dict[str, WindowSet]:
    """Train/val/test windows of one client (vehicle)."""
    path = resolve_path(cfg["data"]["path"])
    if not path.is_file():
        raise FileNotFoundError(
            f"data file {path} not found. Pass its absolute path, e.g. "
            "flwr run . --run-config \"data-path='$PWD/data/cabin_comfort.csv'\""
        )
    ws = _windows(json.dumps({k: v for k, v in cfg.items() if not k.startswith("_")}, sort_keys=True))
    n_clients = int(ws.meta["client"].nunique())
    if num_partitions != n_clients:
        raise ValueError(
            f"partition '{cfg['federated']['partition']}' defines {n_clients} clients but the "
            f"federation has {num_partitions} SuperNodes; set num-supernodes = {n_clients}"
        )
    part = ws.subset(ws.meta["client"].to_numpy() == partition_id)
    return {s: part.part(s) for s in ("train", "val", "test")}


def to_record(named: dict[str, np.ndarray]) -> ArrayRecord:
    return ArrayRecord({k: Array(np.ascontiguousarray(v)) for k, v in named.items()})


def from_record(record: ArrayRecord) -> dict[str, np.ndarray]:
    return {k: v.numpy() for k, v in record.items()}
