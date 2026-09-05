"""封存目录的路径解析与逐折口径核对（G0）。

**这是全仓库唯一知道封存目录命名的模块。** 其余脚本（含 `exp11` / `ict` 探针）
一律通过解封入口写出的**未封存输出树**取分数，不得写死封存目录。

判据相关的硬约束（先于结果落笔）：

* 每折读前核对 ``SEALED_MANIFEST.json``：``scores_sha256``
  （`crsp_pipeline.sealed.sha256_file`）与 ``config`` 五项
  （lookback 90 / predict 6 / sample_count 5 / amp bf16 / batch_size 128）。
  **任何一项失配即中止整个运行**（`CLAUDE.md` §八：口径必须可机器核对；
  不同口径的读数不得并列比较）。
* 折号必须落在 :data:`unseal.config.CONFIRM_FOLDS` ∪
  :data:`unseal.config.CLEAN_WINDOW_FOLDS` 内；不提供任何绕过开关。
* 折窗口一律由 :mod:`unseal.folds` 机械生成，本模块只负责**核对**清单里的
  ``val_window`` 与机械窗口一致，不接受手写窗口。
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from crsp_pipeline.sealed import MANIFEST, assert_readable, sha256_file

from . import config as C

__all__ = [
    "SEALED_FOLDS",
    "SealedConfigMismatch",
    "sealed_eval_dir",
    "sealed_scores_path",
    "read_manifest",
    "verify_fold",
    "tree_sealed_dir",
    "verify_tree_fold",
]

#: 封存折号：05–35（确认集）+ 44–45（干净窗）。**不放宽
#: `signals.kronos_adapter.ALLOWED_FOLDS`**——那份白名单只管已消耗的开发折。
SEALED_FOLDS: frozenset[int] = frozenset(C.CONFIRM_FOLDS + C.CLEAN_WINDOW_FOLDS)

_SEALED_DIR_TEMPLATE: dict[str, str] = {
    # 目录名由 scripts/run_sealed_confirm_queue.ps1 的 --tag 决定（sealed_{arm}_fold{NN}）。
    # 本模块在 tests/test_sealed_mode.py 的 MARKER_ALLOWLIST 内，是唯一有正当理由
    # 写出这两个目录名的地方。
    "ft": "fold{f:02d}_lb90_s0_poolB_universe/eval_sealed_ft_fold{f:02d}",
    "zs": "zeroshot_base/eval_sealed_zs_fold{f:02d}",
}


class SealedConfigMismatch(RuntimeError):
    """封存清单的哈希或打分口径与冻结口径不符——中止整个运行。"""


def _check(fold_id: int, arm: str) -> tuple[int, str]:
    f = int(fold_id)
    if f not in SEALED_FOLDS:
        raise ValueError(
            f"折 {f:02d} 不在封存折集合 {sorted(SEALED_FOLDS)} 内；"
            f"开发折请走 signals.kronos_adapter（CLAUDE.md §四）。")
    if arm not in C.ARMS:
        raise ValueError(f"arm 必须是 {C.ARMS} 之一，收到 {arm!r}")
    return f, arm


def sealed_eval_dir(fold_id: int, arm: str, root: Path | str) -> Path:
    """该折该臂的封存评估目录（不检查是否存在）。"""
    f, a = _check(fold_id, arm)
    return Path(root) / _SEALED_DIR_TEMPLATE[a].format(f=f)


def sealed_scores_path(fold_id: int, arm: str, root: Path | str) -> Path:
    return sealed_eval_dir(fold_id, arm, root) / "scores.parquet"


def read_manifest(fold_id: int, arm: str, root: Path | str) -> dict:
    """读清单。清单只含哈希与口径，**不含任何绩效量**，读它不算读结果。"""
    path = sealed_eval_dir(fold_id, arm, root) / MANIFEST
    if not path.is_file():
        raise SealedConfigMismatch(f"缺封存清单：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_fold(fold_id: int, arm: str, root: Path | str,
                val_window: tuple[pd.Timestamp, pd.Timestamp],
                *, check_scores_hash: bool = True) -> dict:
    """核对一折一臂的封存清单；任何失配抛 :class:`SealedConfigMismatch`。

    返回一个只含**口径事实**的小字典（不含任何分数或绩效量），供审计落盘。
    ``val_window`` 必须来自 :mod:`unseal.folds` 的机械生成结果。
    """
    f, a = _check(fold_id, arm)
    man = read_manifest(f, a, root)
    d = sealed_eval_dir(f, a, root)
    scores = d / "scores.parquet"

    cfg = man.get("config") or {}
    bad = {k: (cfg.get(k), v) for k, v in C.REQUIRED_SCORING_CONFIG.items()
           if cfg.get(k) != v}
    if bad:
        raise SealedConfigMismatch(
            f"折 {f:02d} / {a}: 打分口径失配 {bad}（要求 "
            f"{C.REQUIRED_SCORING_CONFIG}）。CLAUDE.md §八：不同口径的读数不得并列比较。")

    want = [str(pd.Timestamp(x).date()) for x in val_window]
    got = [str(pd.Timestamp(x).date()) for x in (man.get("val_window") or [None, None])]
    if got != want:
        raise SealedConfigMismatch(
            f"折 {f:02d} / {a}: 清单 val_window={got} 与机械生成窗口 {want} 不符；"
            f"折窗口一律由 crsp_pipeline.splits.walk_forward_folds 生成，不得手写。")

    digest = None
    if check_scores_hash:
        if not scores.is_file():
            raise SealedConfigMismatch(f"折 {f:02d} / {a}: 缺 {scores}")
        assert_readable(scores, unseal=True)
        digest = sha256_file(scores)
        if digest != man.get("scores_sha256"):
            raise SealedConfigMismatch(
                f"折 {f:02d} / {a}: scores.parquet 的 sha256 {digest} 与清单记的 "
                f"{man.get('scores_sha256')} 不符——封存后被改动过，中止整个运行。")

    return {
        "fold": f"fold{f:02d}",
        "arm": a,
        "eval_dir": str(d),
        "val_window": got,
        "scoring_config": {k: cfg.get(k) for k in C.REQUIRED_SCORING_CONFIG},
        "scores_sha256": digest if digest is not None else man.get("scores_sha256"),
        "scores_sha256_checked": bool(check_scores_hash),
        "scores_rows": man.get("scores_rows"),
        "snapshot_id": man.get("snapshot_id"),
        "code_sha256": man.get("code_sha256"),
    }


# ------------------------------------------------------------------ H3 树基线

def tree_sealed_dir(fold_id: int, root: Path | str) -> Path:
    """冻结树基线（XGBoost）的封存目录（2026-09-05「计算专用」授权的产物）。"""
    f, _ = _check(fold_id, "ft")
    return Path(root) / C.H3_TREE_DIR_TEMPLATE.format(f=f)


def verify_tree_fold(fold_id: int, root: Path | str,
                     val_window: tuple[pd.Timestamp, pd.Timestamp],
                     *, check_scores_hash: bool = True) -> dict:
    """核对树基线一折的封存清单。失配抛 :class:`SealedConfigMismatch`。

    树基线的口径字段与 Kronos 不同（没有 lookback / sample_count / amp / batch_size），
    改核 **``config_sha256`` + ``jkp_snapshot_sha256`` + ``scores_sha256`` + ``val_window``**
    （2026-09-05 授权行 (iii) 列的清单字段）。跨折一致性由调用方汇总核对。
    """
    f, _ = _check(fold_id, "ft")
    d = tree_sealed_dir(f, root)
    man_path = d / MANIFEST
    if not man_path.is_file():
        raise SealedConfigMismatch(
            f"折 {f:02d}: 缺树基线封存清单 {man_path}（树基线队列可能尚未跑到该折）")
    man = json.loads(man_path.read_text(encoding="utf-8"))
    for key in ("config_sha256", "jkp_snapshot_sha256", "scores_sha256"):
        if not man.get(key):
            raise SealedConfigMismatch(f"折 {f:02d}: 树基线清单缺 {key}")

    want = [str(pd.Timestamp(x).date()) for x in val_window]
    got = [str(pd.Timestamp(x).date()) for x in (man.get("val_window") or [None, None])]
    if got != want:
        raise SealedConfigMismatch(
            f"折 {f:02d} 树基线: 清单 val_window={got} 与机械生成窗口 {want} 不符")

    digest = None
    scores = d / "scores.parquet"
    if check_scores_hash:
        if not scores.is_file():
            raise SealedConfigMismatch(f"折 {f:02d}: 缺 {scores}")
        assert_readable(scores, unseal=True)
        digest = sha256_file(scores)
        if digest != man["scores_sha256"]:
            raise SealedConfigMismatch(
                f"折 {f:02d} 树基线: scores.parquet 的 sha256 {digest} 与清单记的 "
                f"{man['scores_sha256']} 不符——封存后被改动过，中止整个运行。")

    return {
        "fold": f"fold{f:02d}",
        "model": man.get("model"),
        "eval_dir": str(d),
        "val_window": got,
        "config_sha256": man["config_sha256"],
        "jkp_snapshot_sha256": man["jkp_snapshot_sha256"],
        "scores_sha256": digest if digest is not None else man["scores_sha256"],
        "scores_sha256_checked": bool(check_scores_hash),
        "seeds": man.get("seeds"),
        "snapshot_id": man.get("snapshot_id"),
    }
