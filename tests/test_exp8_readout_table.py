"""实验 8 汇总表脚本的最小测试（合成数据、CPU、秒级）。

只验证汇总逻辑本身：折等权 vs 合并日两个口径分得开、oracle 逐格带
「不可用于判定」标注、必填列齐全、抄自登记簿的行原样保留。
不碰任何真实产物、不读任何面板。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import exp8_readout_table as T  # noqa: E402
from crsp_pipeline.signal_eval import newey_west_tstat  # noqa: E402


def _make_eval_dir(root: Path, name: str, ics: np.ndarray) -> Path:
    d = root / name
    d.mkdir(parents=True)
    nw = newey_west_tstat(pd.Series(ics), T.NW_LAGS)
    (d / "metrics.json").write_text(json.dumps({
        "tag": name, "val_window": ["2020-07-01", "2020-12-31"],
        "n_obs": 1000, "n_days": len(ics),
        "scoring_config": {"amp": "bf16", "batch_size": 256, "sample_count": 0,
                           "lookback": 90, "predict": 6, "mode": "rank_head"},
        "head_training": {"best_epoch": 3, "best_val_rank_ic": 0.02},
        "nw_raw": nw,
    }, ensure_ascii=False), encoding="utf-8")
    pd.Series(ics, name="rank_ic").to_frame().to_parquet(d / "daily_ic.parquet")
    return d


@pytest.fixture()
def synth(tmp_path):
    rng = np.random.default_rng(0)
    a = rng.normal(0.02, 0.05, 60)      # 正折
    b = rng.normal(-0.01, 0.05, 40)     # 负折
    dirs = {"fold36": _make_eval_dir(tmp_path, "f36", a),
            "fold37": _make_eval_dir(tmp_path, "f37", b)}
    ridge_json = tmp_path / "ridge.json"
    ridge_json.write_text(json.dumps({
        "fold36": {"alpha": 1e6, "rank_ic": 0.01, "t": 1.0, "n_days": 60,
                   "pool": "mean", "inner_rank_ic": 0.02, "oracle_rank_ic": 0.03},
        "fold37": {"alpha": 1e9, "rank_ic": -0.004, "t": -0.4, "n_days": 40,
                   "pool": "mean", "inner_rank_ic": 0.01, "oracle_rank_ic": 0.011},
        "_summary": {"ignored": True},
    }, ensure_ascii=False), encoding="utf-8")
    dic = tmp_path / "daily"
    dic.mkdir()
    for name, arr in (("fold36", a * 0.5), ("fold37", b * 0.5)):
        pd.DataFrame({"signal_date": pd.date_range("2020-07-01", periods=len(arr)),
                      "rank_ic": arr}).to_parquet(dic / f"{name}.parquet", index=False)
    return {"dirs": dirs, "a": a, "b": b, "ridge_json": ridge_json, "daily": dic}


def test_mlp_equal_fold_and_pooled_are_distinct(synth):
    out = T.collect_mlp(synth["dirs"])
    a, b = synth["a"], synth["b"]
    assert out["n_folds"] == 2
    assert out["n_days_total"] == len(a) + len(b)
    assert out["n_positive"] == 1
    assert out["mean_rank_ic_equal_fold"] == pytest.approx((a.mean() + b.mean()) / 2)
    pooled_expected = newey_west_tstat(pd.Series(np.concatenate([a, b])), T.NW_LAGS)
    assert out["pooled"]["mean"] == pytest.approx(pooled_expected["mean"])
    assert out["pooled"]["t"] == pytest.approx(pooled_expected["t"])
    # 折等权与合并日是两个口径，天数不同的折存在时不应相等
    assert out["mean_rank_ic_equal_fold"] != pytest.approx(out["pooled"]["mean"])
    assert out["scoring_config_all_equal"] is True


def test_mlp_matches_recorded_metrics(synth):
    out = T.collect_mlp(synth["dirs"])
    for fold, rec in out["per_fold"].items():
        assert rec["rank_ic"] == pytest.approx(rec["recorded_rank_ic"]), fold
        assert rec["t"] == pytest.approx(rec["recorded_t"]), fold


def test_ridge_oracle_is_fold_mean(synth):
    out = T.collect_ridge(synth["ridge_json"], synth["daily"], ["fold36", "fold37"])
    assert out["n_folds"] == 2
    assert out["mean_rank_ic_equal_fold"] == pytest.approx((0.01 - 0.004) / 2)
    assert out["mean_oracle_rank_ic"] == pytest.approx((0.03 + 0.011) / 2)
    assert out["n_positive"] == 1
    assert out["pooled"] is not None


def test_ridge_without_daily_dir_has_no_pooled(synth):
    out = T.collect_ridge(synth["ridge_json"], None, ["fold36", "fold37"])
    assert out["pooled"] is None


def test_markdown_has_required_columns_and_oracle_tag(synth):
    mlp = T.collect_mlp(synth["dirs"])
    ridge = T.collect_ridge(synth["ridge_json"], synth["daily"], ["fold36", "fold37"])
    rows = T.build_rows(mlp, ridge, "cmd", "fold36–fold37")
    md = T.rows_to_markdown(rows, "# 测试")
    for col in ["方法", "折号", "折数", "天数", "内层选参(是/否)", "scoring_config",
                "全池 RankIC", "NW(5) t", "正折数"]:
        assert col in md
    # oracle 必须逐格标注
    for line in md.splitlines():
        if "+0.0205" in line or "+0.01951" in line:
            pass
    assert md.count(T.ORACLE_TAG) >= 2      # 表头 + 至少一格
    oracle_cells = [T._oracle_cell(r.get("oracle")) for r in rows]
    for cell in oracle_cells:
        assert cell == "不适用" or T.ORACLE_TAG in cell or cell == T.UNCHECKED


def test_ledger_rows_are_copied_verbatim(synth):
    mlp = T.collect_mlp(synth["dirs"])
    ridge = T.collect_ridge(synth["ridge_json"], synth["daily"], ["fold36", "fold37"])
    rows = T.build_rows(mlp, ridge, "cmd", "fold36–fold37")
    copied = [r for r in rows if str(r["来源"]).startswith("抄自登记簿")]
    assert len(copied) == len(T.LEDGER_ROWS)
    by_src = {(r["方法"], r["折号"]): r for r in copied}
    assert by_src[("生成式 零样本(ZS)", "36–42")]["全池 RankIC"] == 0.01919
    assert by_src[("最佳树基线 XGBoost", "36–42")]["全池 RankIC（合并日）"] == 0.006280
    assert by_src[("MLP 排序头", "fold40（单折）")]["全池 RankIC"] == 0.00753
    # 五折行与七折行必须分开成行，不得合并
    assert any(r["折数"] == 5 for r in copied) and any(r["折数"] == 7 for r in copied)
