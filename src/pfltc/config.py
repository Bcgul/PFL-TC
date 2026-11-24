"""Configuration loading: packaged defaults, optional YAML file, CLI overrides."""

from __future__ import annotations

import copy
from importlib import resources
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def default_config() -> dict[str, Any]:
    text = resources.files("pfltc").joinpath("default.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text)


def apply_override(cfg: dict, assignment: str) -> dict:
    """Apply one ``key.subkey=value`` override; the value is parsed as YAML."""
    if "=" not in assignment:
        raise ValueError(f"Override must look like key.subkey=value, got {assignment!r}")
    dotted, raw = assignment.split("=", 1)
    keys = dotted.strip().split(".")
    node = cfg
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            raise KeyError(f"Unknown config section {key!r} in {dotted!r}")
        node = node[key]
    if keys[-1] not in node:
        raise KeyError(f"Unknown config key {dotted!r}")
    node[keys[-1]] = yaml.safe_load(raw)
    return cfg


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    cfg = default_config()
    if path:
        with open(path, encoding="utf-8") as fh:
            cfg = _deep_merge(cfg, yaml.safe_load(fh) or {})
    for assignment in overrides or []:
        cfg = apply_override(cfg, assignment)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    split = cfg["split"]
    total = split["train"] + split["val"] + split["test"]
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1, got {total}")
    if split["mode"] not in {"chronological", "random"}:
        raise ValueError(f"unknown split.mode {split['mode']!r}")
    if cfg["target"]["mode"] not in {"griffiths", "column"}:
        raise ValueError(f"unknown target.mode {cfg['target']['mode']!r}")
    if cfg["target"]["mode"] == "column" and not cfg["target"]["columns"]:
        raise ValueError("target.mode = column needs target.columns")
    if cfg["target"]["mode"] == "griffiths" and cfg["features"]["include_vote"]:
        raise ValueError(
            "features.include_vote must be false when target.mode = griffiths: "
            "the target is computed from the comfort vote, so using the vote as an "
            "input leaks the label."
        )
    if cfg["federated"]["partition"] not in {"by_day", "by_drive", "iid"}:
        raise ValueError(f"unknown federated.partition {cfg['federated']['partition']!r}")
    if cfg["federated"]["strategy"] not in {"fedavg", "fedper", "local"}:
        raise ValueError(f"unknown federated.strategy {cfg['federated']['strategy']!r}")
    if cfg["windows"]["seq_len"] < 1:
        raise ValueError("windows.seq_len must be >= 1")


def resolve_path(path: str | Path, root: str | Path | None = None) -> Path:
    """Resolve a data path.

    Absolute paths are returned as they are. Relative paths are looked up in
    ``root`` (if given), the current directory and the repository root (for an
    editable install), in this order; the first existing file wins.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    candidates = [Path(root)] if root else []
    candidates += [Path.cwd(), Path(__file__).resolve().parents[2]]
    for base in candidates:
        if (base / p).is_file():
            return base / p
    return candidates[0] / p
