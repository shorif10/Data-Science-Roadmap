"""Load the YAML config, merge optional overlays, resolve paths."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict: `overlay` values win, nested dicts are merged recursively."""
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path, overlay: str | Path | dict | None = None) -> dict[str, Any]:
    """Read `path` (YAML). `overlay` may be a dict or a YAML file (e.g. DVC's params.yaml).

    The project root is the parent of the folder holding the config file; it is stored
    under the key `_root` and used by `resolve` to turn relative paths into absolute ones.
    """
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if overlay is not None:
        if not isinstance(overlay, dict):
            overlay = yaml.safe_load(Path(overlay).read_text(encoding="utf-8")) or {}
        cfg = deep_merge(cfg, overlay)
    cfg["_root"] = str(path.parent.parent)
    return cfg


def resolve(cfg: dict[str, Any], p: str | Path) -> Path:
    """Absolute path for a config path (relative paths are relative to the project root)."""
    p = Path(p)
    return p if p.is_absolute() else Path(cfg.get("_root", ".")) / p
