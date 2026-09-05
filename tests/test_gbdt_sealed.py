"""树基线『计算专用』模式的硬断言（2026-09-05 授权：计算 != 读取）。

盯死的东西与 ``test_sealed_mode.py`` 同源，只是换成树基线的目录形状
``outputs/<exp>/<model>/sealed/foldNN/``：

1. 封存目录里只有分数、哨兵、清单、日志与内层选参记录 —— 不得出现
   ``labels.parquet`` / ``metrics.json`` / ``daily_ic*.parquet`` / ``fold_summary.json``；
2. ``--sealed`` 这条路径在物理上进不了 label / 指标分支（把四个入口打成地雷验证）；
3. 清单字段齐全（snapshot_id / code_sha256 / config_sha256 / jkp_snapshot_sha256 /
   scores_sha256 / val_window / seeds / model / sealed_utc）；
4. 折表只能来自 ``scripts/emit_folds.py`` 的 JSON，不能手写；
5. 登记簿里封存条目不含任何绩效字段。

端到端那一段需要 xgboost（只在 ``.venv-gbdt``，而它没有 pytest），故实现放在
``scripts/gbdt_sealed_smoke.py``：``.venv`` 下自动跳过，``.venv-gbdt`` 下直接跑该脚本。
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from crsp_pipeline.sealed import (  # noqa: E402
    FORBIDDEN_FILES,
    SealedReadError,
    assert_readable,
    audit_dir,
)


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gbdt = _load("gbdt_baseline_sealed", "scripts/gbdt_baseline.py")
smoke = _load("gbdt_sealed_smoke", "scripts/gbdt_sealed_smoke.py")

LEDGER = REPO / "experiments" / "ledger.md"
METRIC_TOKENS = re.compile(r"RankIC|rank_ic|\bt\s*=|IC=|夏普|sharpe|年化|decile|收益",
                           re.IGNORECASE)
BANNED_NAMES = smoke.BANNED_NAMES
BANNED_PATTERNS = smoke.BANNED_PATTERNS
SEEDS = smoke.SEEDS


# ---------------------------------------------------------------- 单元断言

def test_fold_table_must_come_from_emit_folds(tmp_path: Path) -> None:
    path = tmp_path / "folds.json"
    path.write_text(
        json.dumps([{"n": "fold05", "ts": "2002-01-03", "te": "2004-12-22",
                     "vs": "2005-01-03", "ve": "2005-07-01"}]),
        encoding="utf-8")
    folds = gbdt._load_folds_json(path)
    assert folds[0] == gbdt.Fold("fold05", "2002-01-03", "2004-12-22",
                                 "2005-01-03", "2005-07-01")
    path.write_text(json.dumps([]), encoding="utf-8")
    with pytest.raises(ValueError):
        gbdt._load_folds_json(path)


def test_default_fold_table_is_still_the_seven_frozen_development_folds() -> None:
    assert gbdt.ACTIVE_FOLDS == gbdt.FOLDS
    assert [fold.name for fold in gbdt.FOLDS] == [f"fold{n}" for n in range(36, 43)]
    assert gbdt.FOLDS[0] == gbdt.Fold("fold36", "2017-07-03", "2020-06-22",
                                      "2020-07-01", "2020-12-31")
    assert gbdt.FOLDS[-1] == gbdt.Fold("fold42", "2020-07-01", "2023-06-22",
                                       "2023-07-03", "2023-12-29")


def test_sealed_artifacts_live_in_their_own_subtree() -> None:
    fold = gbdt.FOLDS[0]
    out = Path("outputs") / "exp"
    assert gbdt._fold_dir(out, "xgboost", fold, False) == out / "xgboost" / "fold36"
    assert gbdt._fold_dir(out, "xgboost", fold, True) == out / "xgboost" / "sealed" / "fold36"


def test_sealed_mode_refuses_a_hand_written_fold_table(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["gbdt_baseline.py", "--sealed"])
    with pytest.raises(ValueError, match="emit_folds"):
        gbdt.main()


def test_sealed_mode_refuses_to_summarize(monkeypatch, tmp_path: Path) -> None:
    table = tmp_path / "folds.json"
    table.write_text(json.dumps([{"n": "fold05", "ts": "2002-01-03", "te": "2004-12-22",
                                  "vs": "2005-01-03", "ve": "2005-07-01"}]), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["gbdt_baseline.py", "--sealed", "--summarize-only",
                                      "--folds-json", str(table)])
    with pytest.raises(ValueError, match="computes metrics"):
        gbdt.main()


def test_sealed_allowlist_covers_exactly_the_frozen_seed_artifacts() -> None:
    allowed = gbdt._sealed_allowed([11, 29, 47])
    assert allowed == {
        "tuning.json",
        "scores_seed11.parquet", "scores_seed11.json",
        "scores_seed29.parquet", "scores_seed29.json",
        "scores_seed47.parquet", "scores_seed47.json",
    }
    assert not allowed & set(FORBIDDEN_FILES)
    assert not allowed & BANNED_NAMES


# ---------------------------------------------------------------- 端到端冒烟

@pytest.fixture(scope="module")
def sealed_run(tmp_path_factory) -> Path:
    pytest.importorskip("xgboost", reason="xgboost 只在 .venv-gbdt")
    return smoke.run(tmp_path_factory.mktemp("gbdt_sealed"))


def test_sealed_run_publishes_only_allowed_artifacts(sealed_run: Path) -> None:
    report = audit_dir(sealed_run, extra_allowed=gbdt._sealed_allowed(SEEDS))
    assert report["clean"], report
    names = {p.name for p in sealed_run.iterdir()}
    assert {"scores.parquet", "SEALED", "SEALED_MANIFEST.json"} <= names
    assert not names & BANNED_NAMES, sorted(names & BANNED_NAMES)
    assert not [n for n in names for pat in BANNED_PATTERNS if pat in n]


def test_sealed_scores_are_the_frozen_three_seed_mean(sealed_run: Path) -> None:
    scores = pd.read_parquet(sealed_run / "scores.parquet")
    assert list(scores.columns) == ["PERMNO", "signal_date", "score"]
    assert len(scores) and not scores[["PERMNO", "signal_date"]].duplicated().any()
    stacked = np.column_stack([
        pd.read_parquet(sealed_run / f"scores_seed{seed}.parquet")["score"].to_numpy()
        for seed in SEEDS
    ])
    np.testing.assert_allclose(scores["score"].to_numpy(), stacked.mean(axis=1), rtol=1e-9)


def test_sealed_manifest_carries_every_required_field(sealed_run: Path) -> None:
    manifest = json.loads((sealed_run / "SEALED_MANIFEST.json").read_text(encoding="utf-8"))
    for key in ("snapshot_id", "code_sha256", "config_sha256", "jkp_snapshot_sha256",
                "scores_sha256", "val_window", "seeds", "model", "sealed_utc",
                "authorisation", "n_rows"):
        assert key in manifest, key
    assert manifest["model"] == "xgboost"
    assert manifest["seeds"] == SEEDS
    assert set(manifest["code_sha256"]) == {"gbdt_baseline.py", "config"}
    assert len(manifest["code_sha256"]["gbdt_baseline.py"]) == 64
    assert manifest["scores_sha256"] == gbdt._sha256(sealed_run / "scores.parquet")
    assert len(manifest["val_window"]) == 2
    prose = ("authorisation", "no_metrics", "inner_tuning_artifact", "scoring_config")
    payload = json.dumps({k: v for k, v in manifest.items() if k not in prose},
                         ensure_ascii=False)
    assert not METRIC_TOKENS.search(payload), payload


def test_read_guard_covers_the_sealed_fold_directory(sealed_run: Path) -> None:
    with pytest.raises(SealedReadError):
        assert_readable(sealed_run / "scores.parquet")


# ---------------------------------------------------------------- 真实产物

def _real_sealed_dirs() -> list[Path]:
    out = REPO / "outputs"
    if not out.exists():
        return []
    return sorted(p for p in out.glob("*/*/sealed/fold*") if p.is_dir())


@pytest.mark.parametrize("sub", _real_sealed_dirs(),
                         ids=lambda p: f"{p.parents[1].name}-{p.name}")
def test_real_sealed_dirs_never_hold_forbidden_artifacts(sub: Path) -> None:
    """对所有树基线封存目录都成立，包括正在写入的。"""
    names = {p.name for p in sub.iterdir()}
    assert not names & BANNED_NAMES, f"封存目录出现禁止产物 {sorted(names & BANNED_NAMES)}: {sub}"
    assert not [n for n in names for pat in BANNED_PATTERNS if pat in n], f"{sub}: {sorted(names)}"


@pytest.mark.parametrize(
    "sub", [p for p in _real_sealed_dirs() if (p / "SEALED_MANIFEST.json").exists()],
    ids=lambda p: f"{p.parents[1].name}-{p.name}")
def test_completed_real_sealed_dirs_are_clean(sub: Path) -> None:
    manifest = json.loads((sub / "SEALED_MANIFEST.json").read_text(encoding="utf-8"))
    report = audit_dir(sub, extra_allowed=gbdt._sealed_allowed(manifest["seeds"]))
    assert report["clean"], report
    for key in ("snapshot_id", "code_sha256", "config_sha256", "jkp_snapshot_sha256",
                "scores_sha256", "val_window", "seeds", "model", "sealed_utc"):
        assert key in manifest, f"清单缺字段 {key}: {sub}"
    assert manifest["scores_sha256"] == gbdt._sha256(sub / "scores.parquet")


def test_sealed_feature_cache_is_itself_sealed() -> None:
    """年度缓存带训练目标列 y，目录必须自带哨兵。"""
    for cache in (REPO / "outputs").glob("*/cache_sealed"):
        assert (cache / "SEALED").exists(), f"含标签的缓存没有哨兵: {cache}"
        with pytest.raises(SealedReadError):
            assert_readable(cache / "base")


def test_sealed_ledger_lines_carry_no_metrics() -> None:
    """只卡真正的 `| sealed-compute |` 流水行。

    ``test_sealed_mode.py`` 用的是子串匹配，会误伤散文里提到 sealed-compute 的
    授权条目（2026-09-05 那条就被误伤了）；这里按字段分隔符匹配。
    """
    if not LEDGER.exists():
        pytest.skip("no ledger")
    bad = [line for line in LEDGER.read_text(encoding="utf-8").splitlines()
           if "| sealed-compute |" in line and METRIC_TOKENS.search(line)]
    assert not bad, f"封存登记条目里出现绩效字段: {bad}"
