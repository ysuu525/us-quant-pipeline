"""消融臂配对检验（scripts/compare_arms.py）：块自助与判据逻辑。"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compare_arms import block_bootstrap_mean, compare, load_daily_ic  # noqa: E402


def _series(vals, start="2003-01-02"):
    idx = pd.bdate_range(start, periods=len(vals))
    return pd.Series(vals, index=idx, dtype=float)


def test_block_bootstrap_recovers_mean():
    x = np.arange(200, dtype=float)
    boot = block_bootstrap_mean([x], n_boot=500, block_len=10)
    assert len(boot) == 500
    # 自助均值应围绕样本均值分布
    assert abs(boot.mean() - x.mean()) < 5.0
    assert boot.std() > 0


def test_block_bootstrap_deterministic():
    x = np.random.default_rng(0).normal(size=120)
    a = block_bootstrap_mean([x], n_boot=200, seed=7)
    b = block_bootstrap_mean([x], n_boot=200, seed=7)
    assert np.array_equal(a, b)
    c = block_bootstrap_mean([x], n_boot=200, seed=8)
    assert not np.array_equal(a, c)


def test_block_bootstrap_respects_fold_boundaries():
    # 两折差异极大；块不跨折 → 自助均值不会跑到两折均值之外太远
    f1 = np.full(50, 0.0)
    f2 = np.full(50, 1.0)
    boot = block_bootstrap_mean([f1, f2], n_boot=300, block_len=10)
    assert boot.min() >= -0.05 and boot.max() <= 1.05
    assert abs(boot.mean() - 0.5) < 0.05    # 每折各占一半长度


def test_block_bootstrap_short_fold_ok():
    # 折长 < 块长时不得崩（L 收缩到折长）
    boot = block_bootstrap_mean([np.array([1.0, 2.0, 3.0])], n_boot=50, block_len=10)
    assert len(boot) == 50 and np.isfinite(boot).all()


def test_compare_detects_clear_win():
    # B 每天都比 A 高 0.02 → 99% CI 应全体 > 0，四折全为正
    a = {f"f{i}": _series(np.random.default_rng(i).normal(0, 0.01, 120),
                          start=f"200{3+i}-01-02") for i in range(4)}
    b = {k: v + 0.02 for k, v in a.items()}
    r = compare(a, b)
    assert r["n_folds"] == 4 and r["n_days_total"] == 480
    assert r["mean_diff"] == pytest.approx(0.02, abs=1e-9)
    assert r["n_folds_positive"] == 4
    assert r["ci99_all_positive"] is True
    assert r["p_boot_le_zero"] == 0.0


def test_compare_detects_tie():
    # 两臂同分布 → CI 应跨 0
    rng = np.random.default_rng(3)
    a = {f"f{i}": _series(rng.normal(0, 0.05, 120), start=f"200{3+i}-01-02")
         for i in range(4)}
    b = {k: _series(rng.normal(0, 0.05, 120), start=str(v.index[0].date()))
         for k, v in a.items()}
    r = compare(a, b)
    assert r["ci99_all_positive"] is False
    assert r["ci99"][0] < 0 < r["ci99"][1]


def test_compare_pairs_by_date_inner_join():
    # B 少了几天 → 只在交集上配对，不静默错位
    a = {"f1": _series(np.ones(10) * 0.01)}
    b = {"f1": _series(np.ones(10) * 0.03).iloc[3:]}
    r = compare(a, b)
    assert r["n_days_total"] == 7
    assert r["mean_diff"] == pytest.approx(0.02, abs=1e-12)


def test_load_daily_ic_roundtrip(tmp_path):
    s = _series([0.01, np.nan, 0.03])
    s.rename("rank_ic").to_frame().to_parquet(tmp_path / "daily_ic.parquet")
    out = load_daily_ic(tmp_path)
    assert len(out) == 2                      # NaN 被丢弃
    assert out.index.is_monotonic_increasing
    with pytest.raises(FileNotFoundError):
        load_daily_ic(tmp_path / "nope")


def test_multi_seed_averaging(tmp_path, capsys, monkeypatch):
    """同一 (臂, 折) 给多个目录 = 多 seed → 逐日 IC 按 seed 平均，不得覆盖。"""
    import runpy
    import sys as _sys

    def _write(d, vals, start="2003-01-02"):
        d.mkdir(parents=True, exist_ok=True)
        _series(vals, start).rename("rank_ic").to_frame().to_parquet(d / "daily_ic.parquet")
        return str(d)

    a_s0 = _write(tmp_path / "a0", [0.00, 0.00, 0.00])
    b_s0 = _write(tmp_path / "b0", [0.02, 0.02, 0.02])
    b_s1 = _write(tmp_path / "b1", [0.04, 0.04, 0.04])   # 同折第二个 seed

    argv = ["compare_arms.py", "--arm", f"A={a_s0}",
            "--arm", f"B={b_s0}", "--arm", f"B={b_s1}"]
    monkeypatch.setattr(_sys, "argv", argv)
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "compare_arms.py"),
                   run_name="__main__")
    out = capsys.readouterr().out
    assert "2 个 seed 逐日平均" in out
    # B 的两 seed 均值 = 0.03 → 配对差 = +0.03（若是覆盖则会是 0.02 或 0.04）
    assert "+0.03000" in out
