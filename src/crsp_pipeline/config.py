"""配置加载：configs/default.yaml 为模板，configs/local.yaml（不进 git）覆盖本机路径。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(
    default_path: str | Path = "configs/default.yaml",
    local_path: str | Path | None = "configs/local.yaml",
) -> dict[str, Any]:
    with open(default_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if local_path is not None and Path(local_path).exists():
        with open(local_path, encoding="utf-8") as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})
    return cfg
