"""Model construction, weight handling, FedAvg and a short end-to-end run."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pfltc.config import load_config
from pfltc.data import VARIANTS, prepare
from pfltc.models import build_model, get_named_weights, personal_layers, set_named_weights, split_weights
from pfltc.train import fedavg, run_regime

DATA = Path(__file__).resolve().parents[1] / "data" / "cabin_comfort.csv"
DIMS = {"basic": 4, "zone1": 2, "zone2": 2}


@pytest.fixture(scope="module")
def cfg():
    return load_config(
        overrides=[
            f"data.path={DATA}",
            "windows.seq_len=5",
            "centralized.epochs=2",
            "federated.rounds=2",
            "federated.local_epochs=1",
        ]
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_model_shapes(cfg, variant):
    model = build_model(variant, DIMS, 5, 2, cfg)
    x = {g: np.zeros((3, 5, d), dtype=np.float32) for g, d in DIMS.items()}
    inputs = {"basic": x["basic"]} if variant == "basic" else x
    assert model(inputs).shape == (3, 2)


def test_personalized_model_matches_paper_units(cfg):
    model = build_model("personalized_model", DIMS, 5, 2, cfg)
    units = {layer.name: layer.units for layer in model.layers if hasattr(layer, "units") and "lstm" in layer.name}
    assert units == {"base_lstm": 10, "zone1_lstm": 20, "zone2_lstm": 20}  # 10 + 20 + 20 = 50


def test_personal_layers(cfg):
    model = build_model("personalized_model", DIMS, 5, 2, cfg)
    assert personal_layers(model, "fedavg") == set()
    assert personal_layers(model, "fedper") == {"zone1_lstm", "zone2_lstm", "head"}
    assert "base_lstm" in personal_layers(model, "local")


def test_named_weight_roundtrip(cfg):
    a = build_model("personalized_model", DIMS, 5, 2, cfg)
    b = build_model("personalized_model", DIMS, 5, 2, cfg)
    set_named_weights(b, get_named_weights(a))
    for k, v in get_named_weights(a).items():
        np.testing.assert_array_equal(v, get_named_weights(b)[k])
    shared, local = split_weights(get_named_weights(a), {"head"})
    assert set(local) == {"head/0", "head/1"} and "base_lstm/0" in shared


def test_fedavg_is_sample_weighted():
    out = fedavg([{"w": np.array([0.0])}, {"w": np.array([10.0])}], [3, 1])
    np.testing.assert_allclose(out["w"], [2.5])


@pytest.mark.parametrize("regime", ["centralized", "fedavg", "fedper", "local"])
def test_short_run(cfg, regime):
    ws = prepare(cfg)
    result = run_regime(ws, "personalized_model", regime, cfg, seed=0)
    assert np.isfinite(result.metrics["mae"]) and result.metrics["mae"] < 10
    assert len(result.predictions) == len(ws.part("test"))


def test_runs_are_reproducible(cfg):
    ws = prepare(cfg)
    a = run_regime(ws, "basic", "fedavg", cfg, seed=1)
    b = run_regime(ws, "basic", "fedavg", cfg, seed=1)
    assert a.metrics["mae"] == pytest.approx(b.metrics["mae"], abs=1e-6)
