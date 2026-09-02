"""信号层的硬断言：分解恒等式、窗口规则、前视自查、规格指纹、折号守卫。

全部用**内存合成数据**，不读任何真实面板、不 import torch。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from signals import (  # noqa: E402
    OvernightIntradaySignal,
    SignalSpec,
    assert_no_lookahead,
    decompose,
)
from signals import kronos_adapter as ka  # noqa: E402
from signals import overnight_intraday as oi  # noqa: E402


# ---------------------------------------------------------------- 合成夹具


def make_panel(dates, permnos=(1, 2, 3), seed=7) -> pd.DataFrame:
    """按 open/close 恒等式生成面板：close_t = close_{t-1}(1+on)(1+intra)。"""
    rng = np.random.default_rng(seed)
    dates = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    rows = []
    for p in permnos:
        prev = 30.0
        for d in dates:
            on = rng.normal(0.0, 0.01)
            intra = rng.normal(0.0, 0.015)
            op = prev * (1.0 + on)
            cl = op * (1.0 + intra)
            rows.append((p, d, op, cl, (1.0 + on) * (1.0 + intra) - 1.0,
                         float(rng.lognormal(15.0, 1.0))))
            prev = cl
    return pd.DataFrame(rows, columns=["PERMNO", "DlyCalDt", "DlyOpen",
                                       "DlyClose", "DlyRet", "DlyPrcVol"])


def bdays(n, start="2021-01-04") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


# ---------------------------------------------------------------- 分解恒等式


def test_decomposition_identity_holds():
    """(1 + overnight)(1 + intraday) = 1 + DlyRet —— 分解不得凭空造收益。"""
    d = decompose(make_panel(bdays(40)))
    lhs = (1.0 + d["overnight"]) * (1.0 + d["intraday"])
    rhs = 1.0 + d["DlyRet"]
    ok = lhs.notna() & rhs.notna()
    assert ok.sum() > 100
    assert np.allclose(lhs[ok], rhs[ok], rtol=0, atol=1e-12)


def test_intraday_uses_absolute_prices():
    """CRSP 负价 = 买卖中点标记，价格有效，取绝对值（两端都取）。"""
    dts = bdays(3)
    panel = pd.DataFrame({
        "PERMNO": [1, 1, 1],
        "DlyCalDt": dts,
        "DlyOpen": [10.0, -10.0, -10.0],
        "DlyClose": [11.0, 11.0, -11.0],
        "DlyRet": [0.1, 0.1, 0.1],
    })
    got = decompose(panel)["intraday"].to_numpy()
    assert np.allclose(got, [0.1, 0.1, 0.1])


@pytest.mark.parametrize("bad_open", [0.0, np.nan, np.inf, -0.0])
def test_zero_or_missing_open_gives_nan(bad_open):
    dts = bdays(2)
    panel = pd.DataFrame({
        "PERMNO": [1, 1], "DlyCalDt": dts,
        "DlyOpen": [10.0, bad_open], "DlyClose": [11.0, 11.0],
        "DlyRet": [0.1, 0.1],
    })
    d = decompose(panel)
    assert np.isfinite(d["intraday"].iloc[0])
    assert np.isnan(d["intraday"].iloc[1])
    assert np.isnan(d["overnight"].iloc[1])   # intraday 缺 → overnight 也缺


def test_missing_ret_gives_nan_overnight_but_keeps_intraday():
    dts = bdays(2)
    panel = pd.DataFrame({
        "PERMNO": [1, 1], "DlyCalDt": dts,
        "DlyOpen": [10.0, 10.0], "DlyClose": [11.0, 11.0],
        "DlyRet": [0.1, np.nan],
    })
    d = decompose(panel)
    assert np.isfinite(d["intraday"].iloc[1])
    assert np.isnan(d["overnight"].iloc[1])


# ---------------------------------------------------------------- 窗口规则


def test_score_needs_full_21_row_window():
    panel = make_panel(bdays(25), permnos=(1,))
    out = OvernightIntradaySignal().compute(panel).sort_values("signal_date")
    assert out["score"].iloc[:oi.LOOKBACK - 1].isna().all()   # 前 20 行不出分
    assert out["score"].iloc[oi.LOOKBACK - 1:].notna().all()


def test_score_equals_plain_sum_of_21_intraday():
    panel = make_panel(bdays(21), permnos=(1,))
    d = decompose(panel)
    out = OvernightIntradaySignal().compute(panel)
    assert np.isclose(out["score"].iloc[-1], d["intraday"].sum(), atol=1e-12)
    assert np.isclose(out["score_overnight"].iloc[-1], d["overnight"].sum(),
                      atol=1e-12)
    assert int(out["n_valid"].iloc[-1]) == oi.LOOKBACK


@pytest.mark.parametrize("n_nan,expect_nan", [(6, False), (7, True)])
def test_min_valid_threshold(n_nan, expect_nan):
    """窗口内有效值 < MIN_VALID(15) → NaN；恰好 15 → 出分。"""
    dts = bdays(21)
    panel = make_panel(dts, permnos=(1,))
    panel.loc[panel.index[:n_nan], "DlyOpen"] = np.nan
    out = OvernightIntradaySignal().compute(panel).sort_values("signal_date")
    last = out.iloc[-1]
    assert int(last["n_valid"]) == oi.LOOKBACK - n_nan
    assert bool(pd.isna(last["score"])) is expect_nan


def test_span_over_31_calendar_days_is_nan():
    """21 行齐、有效值齐，但窗口首末跨度 > 31 自然日（缺行/停牌）→ NaN。"""
    base = list(bdays(20))
    normal = base + [base[-1] + pd.Timedelta(days=3)]     # 正常续接
    gapped = base + [base[-1] + pd.Timedelta(days=30)]    # 中间缺了一大段

    ok = OvernightIntradaySignal().compute(make_panel(normal, permnos=(1,)))
    bad = OvernightIntradaySignal().compute(make_panel(gapped, permnos=(1,)))
    assert (pd.Timestamp(normal[-1]) - pd.Timestamp(normal[0])).days <= oi.MAX_SPAN_DAYS
    assert (pd.Timestamp(gapped[-1]) - pd.Timestamp(gapped[0])).days > oi.MAX_SPAN_DAYS
    assert np.isfinite(ok["score"].iloc[-1])
    assert int(bad["n_valid"].iloc[-1]) == oi.LOOKBACK      # 有效值并不缺
    assert pd.isna(bad["score"].iloc[-1])                   # 但跨度规则否掉


def test_permnos_do_not_bleed_into_each_other():
    """滚动必须分组：另一只票的行不得进入本票窗口。"""
    dts = bdays(21)
    two = make_panel(dts, permnos=(1, 2))
    one = two[two["PERMNO"] == 1]
    a = OvernightIntradaySignal().compute(two)
    a = a[a["PERMNO"] == 1].sort_values("signal_date")["score"].to_numpy()
    b = OvernightIntradaySignal().compute(one).sort_values(
        "signal_date")["score"].to_numpy()
    assert np.allclose(a, b, equal_nan=True)


# ---------------------------------------------------------------- 前视自查


def test_no_lookahead_truncation_invariance():
    panel = make_panel(bdays(80), permnos=(1, 2, 3))
    cutoff = sorted(panel["DlyCalDt"].unique())[55]
    assert_no_lookahead(OvernightIntradaySignal(), panel, cutoff)


def test_no_lookahead_under_future_mutation():
    """更严的一版：把 t 之后的数据整体改掉，t 及之前的输出必须一字不变。"""
    sig = OvernightIntradaySignal()
    panel = make_panel(bdays(80), permnos=(1, 2, 3))
    cutoff = sorted(panel["DlyCalDt"].unique())[55]

    mutated = panel.copy()
    future = mutated["DlyCalDt"] > cutoff
    assert future.sum() > 0
    rng = np.random.default_rng(99)
    n = int(future.sum())
    mutated.loc[future, "DlyOpen"] = rng.uniform(1, 500, n)
    mutated.loc[future, "DlyClose"] = rng.uniform(1, 500, n)
    mutated.loc[future, "DlyRet"] = rng.normal(0, 1, n)

    a = OvernightIntradaySignal().compute(panel)
    b = sig.compute(mutated)
    keys = ["signal_date", "PERMNO"]
    a = a[a["signal_date"] <= cutoff].sort_values(keys).reset_index(drop=True)
    b = b[b["signal_date"] <= cutoff].sort_values(keys).reset_index(drop=True)
    assert a["signal_date"].equals(b["signal_date"])
    for col in ("score", "score_overnight", "n_valid"):
        assert np.allclose(a[col].to_numpy(dtype=float),
                           b[col].to_numpy(dtype=float), equal_nan=True), col


def test_row_order_does_not_matter():
    panel = make_panel(bdays(30), permnos=(1, 2))
    shuffled = panel.sample(frac=1.0, random_state=3)
    keys = ["signal_date", "PERMNO"]
    a = OvernightIntradaySignal().compute(panel).sort_values(keys).reset_index(drop=True)
    b = OvernightIntradaySignal().compute(shuffled).sort_values(keys).reset_index(drop=True)
    assert np.allclose(a["score"].to_numpy(dtype=float),
                       b["score"].to_numpy(dtype=float), equal_nan=True)


# ---------------------------------------------------------------- 规格指纹


def test_spec_hash_is_stable_and_parameter_sensitive():
    s1 = SignalSpec(name="x", version="v1", horizon_days=6,
                    params=(("a", 1), ("b", 2)), source_ref="ref")
    s2 = SignalSpec(name="x", version="v1", horizon_days=6,
                    params=(("a", 1), ("b", 2)), source_ref="ref")
    assert s1.spec_hash() == s2.spec_hash()
    assert len(s1.spec_hash()) == 64

    changed = [
        SignalSpec("x", "v1", 6, (("a", 1), ("b", 3)), "ref"),   # 值变
        SignalSpec("x", "v1", 6, (("a", 1),), "ref"),            # 少一个参数
        SignalSpec("x", "v1", 6, (("b", 2), ("a", 1)), "ref"),   # 顺序变
        SignalSpec("x", "v2", 6, (("a", 1), ("b", 2)), "ref"),   # 版本变
        SignalSpec("y", "v1", 6, (("a", 1), ("b", 2)), "ref"),   # 名字变
        SignalSpec("x", "v1", 5, (("a", 1), ("b", 2)), "ref"),   # 视野变
        SignalSpec("x", "v1", 6, (("a", 1), ("b", 2)), "other"),  # 出处变
    ]
    for c in changed:
        assert c.spec_hash() != s1.spec_hash(), c


def test_spec_is_frozen_and_hashable():
    s = oi.SPEC
    with pytest.raises(Exception):
        s.name = "nope"          # frozen dataclass
    assert isinstance(hash(s), int)
    assert isinstance(json.loads(s.canonical_json()), dict)


def test_frozen_constants_are_registered_in_spec():
    """常量与规格必须同源——改了常量却没改规格哈希是最阴的一类漂移。"""
    assert oi.SPEC.param("lookback") == oi.LOOKBACK == 21
    assert oi.SPEC.param("min_valid") == oi.MIN_VALID == 15
    assert oi.SPEC.param("max_span_days") == oi.MAX_SPAN_DAYS == 31
    assert oi.SPEC.param("winsorize") is False
    assert oi.SPEC.param("cross_sectional_zscore") is False
    assert "Lou-Polk-Skouras 2019" in oi.SPEC.source_ref
    assert "Barardehi-Bogousslavsky-Muravyev 2026" in oi.SPEC.source_ref


def test_missing_columns_raise():
    panel = make_panel(bdays(3), permnos=(1,)).drop(columns=["DlyRet"])
    with pytest.raises(KeyError):
        OvernightIntradaySignal().compute(panel)


# ---------------------------------------------------------------- Kronos 适配器


def _fake_scores(tmp: Path, fold: int, arm: str, n=3) -> Path:
    d = ka.eval_dir(fold, arm, root=tmp)
    d.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "PERMNO": np.arange(1, n + 1, dtype="int64"),
        "signal_date": pd.to_datetime(["2021-06-01"] * n),
        "score": np.linspace(0.0, 1.0, n),
        "extra": ["x"] * n,
    })
    df.to_parquet(d / "scores.parquet", index=False)
    return d / "scores.parquet"


@pytest.mark.parametrize("fold", [0, 5, 20, 35, 43, 99, -1])
def test_rejects_folds_outside_the_consumed_dev_set(fold):
    with pytest.raises(ValueError):
        ka.load_kronos_scores([fold], "ft")
    with pytest.raises(ValueError):
        ka.eval_dir(fold, "zs")


def test_allowed_fold_set_is_exactly_the_consumed_dev_folds():
    assert set(ka.ALLOWED_FOLDS) == {1, 2, 3, 4, 36, 37, 38, 39, 40, 41, 42}


def test_rejects_unknown_arm():
    with pytest.raises(ValueError):
        ka.load_kronos_scores([36], "gbdt")


@pytest.mark.parametrize("fold,arm,rel", [
    (36, "ft", "fold36_lb90_s0_poolB_universe/eval_amp_lb90_fold36"),
    (42, "ft", "fold42_lb90_s0_poolB_universe/eval_amp_lb90_fold42"),
    (36, "zs", "zeroshot_base/eval_zeroshot_fold36"),
    (1, "ft", "fold01_lb90_s0_poolB_universe/eval_poolB_universe"),
    (2, "ft", "fold02_lb90_s0_poolB_universe/eval_poolB_universe_fold02"),
    (4, "ft", "fold04_lb90_s0_poolB_universe/eval_poolB_universe_fold04"),
    (1, "zs", "zeroshot_base/eval_zs_fold01"),
    (3, "zs", "zeroshot_base/eval_zs_fold03"),
])
def test_path_layout_matches_k8_ensemble(fold, arm, rel, tmp_path):
    """路径必须与 scripts/k8_ensemble.py 逐字一致，否则读的是别的臂。"""
    got = ka.scores_path(fold, arm, root=tmp_path)
    assert got == tmp_path.joinpath(*rel.split("/")) / "scores.parquet"
    assert ka.labels_path(fold, "ft", root=tmp_path).name == "labels.parquet"


def test_read_guard_is_called_for_every_file(tmp_path, monkeypatch):
    seen: list[Path] = []
    monkeypatch.setattr(ka, "assert_readable", lambda p, *a, **k: seen.append(Path(p)))
    for f in (36, 37):
        _fake_scores(tmp_path, f, "ft")
    out = ka.load_kronos_scores([36, 37], "ft", root=tmp_path)

    assert len(seen) == 2, f"守卫调用次数不对: {seen}"
    assert all(p.name == "scores.parquet" for p in seen)
    assert list(out.columns) == ["signal_date", "PERMNO", "score"]
    assert len(out) == 6
    assert pd.api.types.is_datetime64_any_dtype(out["signal_date"])


def test_fold_validation_happens_before_touching_disk(tmp_path, monkeypatch):
    seen: list[Path] = []
    monkeypatch.setattr(ka, "assert_readable", lambda p, *a, **k: seen.append(Path(p)))
    _fake_scores(tmp_path, 36, "ft")
    with pytest.raises(ValueError):
        ka.load_kronos_scores([36, 7], "ft", root=tmp_path)
    assert seen == [], "非法折号必须在打开任何文件之前就被拒绝"


def test_load_labels_keeps_only_ok_status(tmp_path, monkeypatch):
    monkeypatch.setattr(ka, "assert_readable", lambda p, *a, **k: None)
    d = ka.eval_dir(36, "ft", root=tmp_path)
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "PERMNO": [1, 2, 3],
        "signal_date": pd.to_datetime(["2021-06-01"] * 3),
        "label": [0.01, 0.02, 0.03],
        "status": ["ok", "unfillable", "invalid"],
    }).to_parquet(d / "labels.parquet", index=False)
    out = ka.load_labels([36], root=tmp_path)
    assert len(out) == 1 and out["status"].tolist() == ["ok"]


# ---------------------------------------------------------------- 读数脚本 dry-run


@pytest.fixture(scope="module")
def readout():
    path = REPO / "scripts" / "signal2_devfold_readout.py"
    spec = importlib.util.spec_from_file_location("signal2_devfold_readout", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parse_folds(readout):
    assert readout.parse_folds("36-42") == [36, 37, 38, 39, 40, 41, 42]
    assert readout.parse_folds("1,2,36-38") == [1, 2, 36, 37, 38]
    assert readout.parse_folds("40,40") == [40]
    with pytest.raises(ValueError):
        readout.parse_folds("42-36")


def test_readout_refuses_unconsumed_folds(readout):
    with pytest.raises(SystemExit):
        readout.main(["--folds", "5-9", "--dry-run"])


def test_dry_run_end_to_end(readout, tmp_path, monkeypatch):
    """合成数据走通全流程并产出结构完整的 JSON（不碰任何真实数据）。"""
    root = tmp_path / "dryrun_root"
    root.mkdir()
    monkeypatch.setattr(readout.tempfile, "mkdtemp", lambda *a, **k: str(root))
    out = tmp_path / "readout.json"

    rc = readout.main(["--dry-run", "--folds", "36-37", "--arm", "both",
                       "--out", str(out)])
    assert rc == 0
    rep = json.loads(out.read_text(encoding="utf-8"))

    meta = rep["meta"]
    assert meta["mode"] == "dry-run"
    assert meta["spec_hash"] == oi.SPEC.spec_hash()
    assert meta["spec"]["name"] == "overnight_intraday"
    assert set(meta["code_sha256"]) >= {"scripts/signal2_devfold_readout.py",
                                        "src/signals/overnight_intraday.py"}
    assert [f["fold"] for f in meta["folds"]] == ["fold36", "fold37"]
    assert meta["arms"] == ["ft", "zs"]
    assert meta["elapsed_sec"] >= 0

    assert set(rep["per_fold"]) == {"fold36", "fold37"}
    for tag, r in rep["per_fold"].items():
        assert r["n_days_scored"] > 0 and r["n_obs_with_label"] > 0
        for col in ("score", "score_overnight"):
            for pool in ("all", "top500"):
                blk = r["ic"][col][pool]
                assert blk["n_days"] > 0
                assert len(blk["dates"]) == len(blk["values"]) == blk["n_days"]
                assert np.isfinite(blk["mean"])
                t = r["turnover_top_decile"][col][pool]
                assert 0.0 <= t["mean"] <= 1.0
        for arm in ("ft", "zs"):
            assert r["kronos_rank_corr"][arm]["n_days"] > 0

    s = rep["summary"]
    for col in ("score", "score_overnight"):
        for pool in ("all", "top500"):
            blk = s["ic"][col][pool]
            assert blk["nw_lag"] == 5
            assert blk["n_days"] > 0
            assert np.isfinite(blk["nw_t"])
            assert 0 <= blk["folds_positive"] <= 2
            assert set(blk["per_fold_mean"]) == {"fold36", "fold37"}
    for arm in ("ft", "zs"):
        assert np.isfinite(s["kronos_rank_corr"][arm]["pooled_mean"])
    assert s["turnover_top_decile"]["score"]["top500"]["n_days"] > 0

    # dry-run 绝不落到 outputs/，也绝不去碰真实面板
    assert not (REPO / "outputs" / "signal2_devfold_readout.json").exists()
