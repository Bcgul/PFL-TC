"""Data handling: loading, cleaning, targets, windowing, splitting, partitioning.

The released recordings (``data/cabin_comfort.csv``) contain one row per sensor
reading (~3.7 s apart) from four drives on two days. Each row has cabin-level
environmental measurements, per-zone skin temperature and relative humidity,
and the comfort vote of every occupant (ASHRAE 7-point scale, recorded by hand).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import resolve_path

# Seat (zone) identifiers used in the configuration.
ZONES = ("1", "2", "3l", "3r")
# The rear bench (zone 3) has a single skin-temperature / RH sensor but one
# comfort vote per seat.
ZONE_SENSOR_PREFIX = {"1": "z1", "2": "z2", "3l": "z3", "3r": "z3"}
ZONE_VOTE_COLUMN = {"1": "z1_vote", "2": "z2_vote", "3l": "z3l_vote", "3r": "z3r_vote"}

META_COLUMNS = ["timestamp", "day", "drive_id", "driver_id", "elapsed_s"]
SENSOR_COLUMNS = [
    "ambient_temp_c",
    "cabin_temp_c",
    "cabin_rh_pct",
    "light_intensity",
    "z1_skin_temp_c",
    "z1_rh_pct",
    "z2_skin_temp_c",
    "z2_rh_pct",
    "z3_skin_temp_c",
    "z3_rh_pct",
]
VOTE_COLUMNS = ["z1_vote", "z2_vote", "z3l_vote", "z3r_vote"]

VARIANTS = ("basic", "personalized_baseline", "personalized_model")


# --------------------------------------------------------------------------- #
# Loading and cleaning
# --------------------------------------------------------------------------- #
def load_dataset(path: str | Path) -> pd.DataFrame:
    """Read the released CSV and sort it by drive and time."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    missing = [c for c in META_COLUMNS + SENSOR_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df.sort_values(["drive_id", "timestamp"], kind="stable").reset_index(drop=True)


def _valid_range(col: str, ranges: dict) -> tuple[float, float] | None:
    if col.endswith("skin_temp_c"):
        return tuple(ranges["skin_temp_c"])
    if col.endswith("_temp_c"):
        return tuple(ranges["temp_c"])
    if col.endswith("_rh_pct"):
        return tuple(ranges["rh_pct"])
    return None


def clean(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict[str, int]]:
    """Mask physically implausible sensor values and interpolate short gaps.

    Returns the cleaned frame and the number of masked values per column.
    Interpolation is linear, stays inside each drive and only fills gaps that
    are enclosed by valid readings.
    """
    df = df.copy()
    ranges = cfg["data"]["valid_ranges"]
    report: dict[str, int] = {}
    sensor_cols = [c for c in df.columns if _valid_range(c, ranges) is not None]
    for col in sensor_cols:
        lo, hi = _valid_range(col, ranges)
        bad = df[col].notna() & ~df[col].between(lo, hi)
        if bad.any():
            report[col] = int(bad.sum())
            df.loc[bad, col] = np.nan
    cols = sensor_cols + [c for c in ("light_intensity",) if c in df.columns]
    gap = int(cfg["data"]["max_interp_gap"])
    df[cols] = df.groupby("drive_id", sort=False)[cols].transform(
        lambda s: s.interpolate(limit=gap, limit_area="inside")
    )
    return df, report


# --------------------------------------------------------------------------- #
# Targets and features
# --------------------------------------------------------------------------- #
def add_targets(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    """Add the regression target column(s) and return their names.

    ``griffiths``: neutral temperature of each occupant,
    ``T_n = T_cabin - vote / G`` (Griffiths' method). A vote of -1 ("slightly
    cool") at 21 degC with G = 0.5 1/K gives a preferred temperature of 23 degC.
    ``column``: recorded target columns, e.g. per-zone HVAC set temperatures.
    """
    tcfg = cfg["target"]
    df = df.copy()
    if tcfg["mode"] == "griffiths":
        slope = float(tcfg["griffiths_slope"])
        cols = []
        for zone in tcfg["zones"]:
            col = f"target_z{zone}_c"
            df[col] = df["cabin_temp_c"] - df[ZONE_VOTE_COLUMN[str(zone)]] / slope
            cols.append(col)
        return df, cols
    cols = list(tcfg["columns"])
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"target columns not found in data: {missing}")
    return df, cols


def feature_groups(cfg: dict) -> dict[str, list[str]]:
    """Input groups as in Fig. 3: ``basic`` plus one group per zone."""
    fcfg = cfg["features"]
    groups = {"basic": list(fcfg["basic"])}
    for zone in fcfg["zones"]:
        zone = str(zone)
        cols = [f"{ZONE_SENSOR_PREFIX[zone]}_{name}" for name in fcfg["zone"]]
        if fcfg["include_vote"]:
            cols.append(ZONE_VOTE_COLUMN[zone])
        groups[f"zone{zone}"] = cols
    return groups


def groups_for_variant(variant: str, groups: list[str]) -> list[str]:
    """Input groups a model variant consumes."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
    return ["basic"] if variant == "basic" else list(groups)


def check_leakage(groups: dict[str, list[str]], target_cols: list[str], cfg: dict) -> None:
    """Refuse configurations in which a target (or its source) is an input."""
    inputs = {c for cols in groups.values() for c in cols}
    overlap = inputs & set(target_cols)
    if overlap:
        raise ValueError(f"target column(s) {sorted(overlap)} are also used as inputs")
    if cfg["target"]["mode"] == "griffiths":
        votes = sorted(c for c in inputs if c.endswith("_vote"))
        if votes:
            raise ValueError(
                f"inputs {votes} are comfort votes, but the target is derived from the votes (target.mode = griffiths)"
            )


# --------------------------------------------------------------------------- #
# Normalisation (fixed constants, see default.yaml)
# --------------------------------------------------------------------------- #
def _norm_spec(col: str, norm_cfg: dict) -> tuple[float, float, str | None]:
    if col in norm_cfg:
        spec = norm_cfg[col]
    else:
        matches = [k for k in norm_cfg if k.startswith("_") and col.endswith(k)]
        if not matches:
            raise KeyError(f"no normalisation defined for column {col!r}; add it to `normalization`")
        spec = norm_cfg[max(matches, key=len)]
    func = spec[2] if len(spec) > 2 else None
    if func not in (None, "log1p"):
        raise ValueError(f"unknown normalisation function {func!r} for {col!r}")
    return float(spec[0]), float(spec[1]), func


def normalize(values: np.ndarray, col: str, norm_cfg: dict) -> np.ndarray:
    centre, scale, func = _norm_spec(col, norm_cfg)
    x = np.asarray(values, dtype=np.float64)
    if func == "log1p":
        x = np.log1p(np.clip(x, 0.0, None))
    return ((x - centre) / scale).astype(np.float32)


def normalize_target(y_c: np.ndarray, norm_cfg: dict) -> np.ndarray:
    centre, scale = (float(v) for v in norm_cfg["target"][:2])
    return ((np.asarray(y_c, dtype=np.float64) - centre) / scale).astype(np.float32)


def denormalize_target(y: np.ndarray, norm_cfg: dict) -> np.ndarray:
    centre, scale = (float(v) for v in norm_cfg["target"][:2])
    return np.asarray(y, dtype=np.float64) * scale + centre


# --------------------------------------------------------------------------- #
# Windows
# --------------------------------------------------------------------------- #
@dataclass
class WindowSet:
    """LSTM input windows; one row of ``meta`` per window (its last sample)."""

    X: dict[str, np.ndarray]  # group -> (N, seq_len, n_features), normalised
    y: np.ndarray  # (N, n_targets), normalised
    y_c: np.ndarray  # (N, n_targets), degC
    meta: pd.DataFrame
    target_names: list[str]

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, selector) -> WindowSet:
        idx = np.asarray(selector)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        return WindowSet(
            X={k: v[idx] for k, v in self.X.items()},
            y=self.y[idx],
            y_c=self.y_c[idx],
            meta=self.meta.iloc[idx].reset_index(drop=True),
            target_names=self.target_names,
        )

    def part(self, split: str) -> WindowSet:
        return self.subset(self.meta["split"].to_numpy() == split)

    def inputs(self, groups: list[str]) -> dict[str, np.ndarray]:
        return {g: self.X[g] for g in groups}

    @property
    def group_dims(self) -> dict[str, int]:
        return {k: int(v.shape[-1]) for k, v in self.X.items()}


def _row_splits(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    scfg = cfg["split"]
    labels = np.full(len(df), "pool", dtype=object)
    if scfg["mode"] != "chronological":
        return labels
    for idx in df.groupby("drive_id", sort=False).indices.values():
        idx = np.sort(idx)
        n = len(idx)
        n_train = int(round(n * scfg["train"]))
        n_val = int(round(n * scfg["val"]))
        labels[idx[:n_train]] = "train"
        labels[idx[n_train : n_train + n_val]] = "val"
        labels[idx[n_train + n_val :]] = "test"
    return labels


def _runs(df: pd.DataFrame, max_gap_s: float) -> np.ndarray:
    dt = df.groupby("drive_id", sort=False)["timestamp"].diff().dt.total_seconds()
    return (dt.isna() | (dt > max_gap_s)).cumsum().to_numpy()


def build_windows(
    df: pd.DataFrame, groups: dict[str, list[str]], target_cols: list[str], cfg: dict, seed: int
) -> WindowSet:
    """Cut every continuous run into overlapping windows of ``seq_len`` samples.

    A window is kept only if all its samples are valid and belong to the same
    split part, so no input sample is shared between train, val and test.
    """
    seq_len = int(cfg["windows"]["seq_len"])
    stride = int(cfg["windows"]["stride"])
    norm = cfg["normalization"]

    feats = {g: np.stack([normalize(df[c].to_numpy(), c, norm) for c in cols], axis=1) for g, cols in groups.items()}
    y_c_all = df[target_cols].to_numpy(dtype=np.float64)
    valid = np.isfinite(y_c_all).all(axis=1)
    for f in feats.values():
        valid &= np.isfinite(f).all(axis=1)

    labels = _row_splits(df, cfg)
    runs = _runs(df, float(cfg["data"]["max_gap_s"]))
    # Position of every row inside its (drive, split part) block. Validation and
    # test windows must end at least `min_context` rows into their block, so
    # different seq_len values are scored on the same target samples.
    block = (pd.Series(labels) != pd.Series(labels).shift()) | (df["drive_id"] != df["drive_id"].shift()).to_numpy()
    pos_in_block = pd.Series(np.arange(len(df))).groupby(block.cumsum().to_numpy()).cumcount().to_numpy()
    min_context = max(int(cfg["split"].get("min_context", 0)), seq_len - 1)

    ends: list[int] = []
    for run in np.unique(runs):
        idx = np.flatnonzero(runs == run)  # contiguous rows of one run
        for e in range(seq_len - 1, len(idx), stride):
            rows = idx[e - seq_len + 1 : e + 1]
            end = rows[-1]
            if not (valid[rows].all() and (labels[rows] == labels[end]).all()):
                continue
            if labels[end] in ("val", "test") and pos_in_block[end] < min_context:
                continue
            ends.append(int(end))
    if not ends:
        raise ValueError("no valid windows; check seq_len and the data")
    ends_arr = np.asarray(ends)

    X = {}
    for g, f in feats.items():
        view = np.lib.stride_tricks.sliding_window_view(f, seq_len, axis=0)  # (n-L+1, d, L)
        X[g] = np.ascontiguousarray(view[ends_arr - (seq_len - 1)].transpose(0, 2, 1))

    meta_cols = [c for c in META_COLUMNS + ["cabin_temp_c"] if c in df.columns]
    meta = df.iloc[ends_arr][meta_cols].reset_index(drop=True)
    meta["row"] = ends_arr
    meta["split"] = labels[ends_arr]
    if cfg["split"]["mode"] == "random":
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(meta))
        n_train = int(round(len(meta) * cfg["split"]["train"]))
        n_val = int(round(len(meta) * cfg["split"]["val"]))
        split = np.empty(len(meta), dtype=object)
        split[order[:n_train]] = "train"
        split[order[n_train : n_train + n_val]] = "val"
        split[order[n_train + n_val :]] = "test"
        meta["split"] = split

    y_c = y_c_all[ends_arr]
    return WindowSet(X=X, y=normalize_target(y_c, norm), y_c=y_c, meta=meta, target_names=target_cols)


# --------------------------------------------------------------------------- #
# Federated partitioning
# --------------------------------------------------------------------------- #
def client_ids(meta: pd.DataFrame, cfg: dict, seed: int) -> np.ndarray:
    """Client index of every window.

    ``by_day``  : one client per recording day (2 clients, default)
    ``by_drive``: one client per drive (4 clients)
    ``iid``     : windows assigned uniformly at random to ``num_clients``
    """
    scheme = cfg["federated"]["partition"]
    if scheme == "iid":
        rng = np.random.default_rng(seed)
        return rng.integers(0, int(cfg["federated"]["num_clients"]), len(meta))
    key = {"by_day": "day", "by_drive": "drive_id"}[scheme]
    mapping = {k: i for i, k in enumerate(sorted(meta[key].unique()))}
    return meta[key].map(mapping).to_numpy()


def prepare(cfg: dict, seed: int | None = None, data_root: str | Path | None = None) -> WindowSet:
    """Load, clean, add targets, window and partition the data in one call.

    The same windows are produced for every model variant, so variants are
    compared on identical samples; each variant then selects its input groups.
    """
    seed = cfg["seed"] if seed is None else seed
    df = load_dataset(resolve_path(cfg["data"]["path"], data_root))
    df, _ = clean(df, cfg)
    df, target_cols = add_targets(df, cfg)
    groups = feature_groups(cfg)
    check_leakage(groups, target_cols, cfg)
    ws = build_windows(df, groups, target_cols, cfg, seed)
    ws.meta["client"] = client_ids(ws.meta, cfg, seed)
    return ws
