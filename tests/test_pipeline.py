"""Fast checks of configuration, data handling, models and aggregation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pfltc.config import load_config
from pfltc.data import add_targets, build_windows, clean, feature_groups, load_dataset, prepare
from pfltc.metrics import regression_metrics

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cabin_comfort.csv"


@pytest.fixture(scope="module")
def cfg():
    return load_config(overrides=[f"data.path={DATA}", "windows.seq_len=5"])


@pytest.fixture(scope="module")
def ws(cfg):
    return prepare(cfg)


# --------------------------------------------------------------------------- #
def test_config_rejects_vote_inputs_with_griffiths_target():
    with pytest.raises(ValueError, match="leaks the label"):
        load_config(overrides=["features.include_vote=true"])


@pytest.mark.parametrize("path", sorted((ROOT / "configs").glob("*.yaml")), ids=lambda p: p.name)
def test_example_configs_load(path):
    load_config(path)


def test_config_override_types():
    cfg = load_config(overrides=["windows.seq_len=7", "target.zones=['1']"])
    assert cfg["windows"]["seq_len"] == 7 and cfg["target"]["zones"] == ["1"]
    with pytest.raises(KeyError):
        load_config(overrides=["windows.unknown=1"])


# --------------------------------------------------------------------------- #
def test_dataset_has_no_personal_metadata():
    df = pd.read_csv(DATA)
    assert len(df) == 2115
    assert not {"name", "age", "gender"} & {c.lower() for c in df.columns}
    assert df["drive_id"].nunique() == 4 and df["day"].nunique() == 2


def test_clean_masks_implausible_humidity(cfg):
    df = load_dataset(DATA)
    assert (df["z3_rh_pct"] < 10).sum() == 1  # one sensor glitch (6 %) in the raw data
    cleaned, report = clean(df, cfg)
    assert report == {"z3_rh_pct": 1}
    assert cleaned["z3_rh_pct"].min() >= 10


def test_griffiths_target(cfg):
    df, cols = add_targets(load_dataset(DATA), cfg)
    assert cols == ["target_z1_c", "target_z2_c"]
    row = df.iloc[0]
    assert row["target_z1_c"] == pytest.approx(row["cabin_temp_c"] - row["z1_vote"] / 0.5)


def test_windows_never_straddle_split_parts(cfg, ws):
    df, target_cols = add_targets(clean(load_dataset(DATA), cfg)[0], cfg)
    labels = {}
    seq_len = cfg["windows"]["seq_len"]
    # Every window's first row must be in the same drive and split as its last row.
    for split in ("train", "val", "test"):
        part = ws.part(split)
        assert len(part) > 0
        starts = part.meta["row"].to_numpy() - (seq_len - 1)
        assert (df.loc[starts, "drive_id"].to_numpy() == part.meta["drive_id"].to_numpy()).all()
        labels[split] = part
    # Chronology: within every drive, train < val < test in time.
    for drive in ws.meta["drive_id"].unique():
        t = {s: labels[s].meta.loc[labels[s].meta["drive_id"] == drive, "timestamp"] for s in labels}
        assert t["train"].max() < t["val"].min() and t["val"].max() < t["test"].min()


def test_window_contents_match_source(cfg, ws):
    df, _ = add_targets(clean(load_dataset(DATA), cfg)[0], cfg)
    i = len(ws) // 2
    end = int(ws.meta.loc[i, "row"])
    expected = df.loc[end, ["target_z1_c", "target_z2_c"]].to_numpy(dtype=float)
    np.testing.assert_allclose(ws.y_c[i], expected)
    # Last time step of the basic group = cabin temperature at the window end (normalised).
    np.testing.assert_allclose(ws.X["basic"][i, -1, 1], (df.loc[end, "cabin_temp_c"] - 20.0) / 10.0, rtol=1e-5)


def test_target_never_an_input(cfg):
    groups = feature_groups(cfg)
    inputs = {c for cols in groups.values() for c in cols}
    assert not any(c.endswith("_vote") for c in inputs)
    bad = load_config(overrides=[f"data.path={DATA}", "target.mode=column", "target.columns=['cabin_temp_c']"])
    with pytest.raises(ValueError, match="also used as inputs"):
        prepare(bad)


def test_partitions(cfg):
    ws_day = prepare(cfg)
    assert sorted(ws_day.meta.groupby("client")["day"].unique().map(tuple)) == [(1,), (2,)]
    iid = load_config(overrides=[f"data.path={DATA}", "windows.seq_len=5", "federated.partition=iid"])
    a, b = prepare(iid, seed=3), prepare(iid, seed=3)
    assert (a.meta["client"] == b.meta["client"]).all()


def test_random_split_is_available(cfg):
    rnd = load_config(overrides=[f"data.path={DATA}", "windows.seq_len=5", "split.mode=random"])
    ws_r = prepare(rnd)
    fractions = ws_r.meta["split"].value_counts(normalize=True)
    assert fractions["train"] == pytest.approx(0.70, abs=0.01)


# --------------------------------------------------------------------------- #
def test_metrics_on_known_values():
    y = np.array([[20.0, 22.0], [24.0, 18.0]])
    p = np.array([[21.0, 22.0], [24.0, 20.0]])
    m = regression_metrics(y, p, ["a", "b"])
    assert m["mae"] == pytest.approx(0.75)
    assert m["mse"] == pytest.approx(1.25)
    assert m["a/mape"] == pytest.approx(100 * (1 / 20) / 2)
    assert m["b/bias"] == pytest.approx(1.0)


def test_windows_build_is_deterministic(cfg):
    df, cols = add_targets(clean(load_dataset(DATA), cfg)[0], cfg)
    a = build_windows(df, feature_groups(cfg), cols, cfg, seed=0)
    b = build_windows(df, feature_groups(cfg), cols, cfg, seed=0)
    np.testing.assert_array_equal(a.X["zone1"], b.X["zone1"])
