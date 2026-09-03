"""`portfolio` 包的测试。核心是**逐位一致性**。

`_reference_arm_returns` 是 `scripts/compare_arms_money.py` 第 62–122 行的
`arm_returns()` 的**原样复制**（2026-09-03 版本）。只改了三处对外部状态的
依赖，其余逻辑一个字符没动：

| 原行 | 原文 | 本文件 |
|---|---|---|
| 62 | `def arm_returns(lb, ret, oc, adv, topn, cost_bp)` | 加了 `sc_by_fold, folds, NT, EXIT_PCT` 形参，去掉没用到的 `lb` |
| 65 | `for fold in FOLDS:` | `for fold in folds:` |
| 66–68 | `d = OUT / ...` + `sc = pd.read_parquet(d / "scores.parquet", columns=[...]).dropna()` | `sc = sc_by_fold[fold].copy().dropna()` |

`NT` / `EXIT_PCT` 原本是模块级全局常量，这里降成默认值相同的形参（6 / 0.30），
函数体内的写法完全没变。

判据：`np.array_equal`（**不是 `allclose`**）。构造是项目的目标函数，一个
tie-break 都不能变——见 `src/portfolio/construction.py` docstring 的
「浮点逐位一致的三个脆弱点」。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio.combine import rank_sum_equal_weight, slow_filter, spearman_by_day
from portfolio.construction import frozen_long_only_returns, scores_frame_to_by_day


# ------------------------------------------------------------------ 参考实现
# 原样复制自 scripts/compare_arms_money.py 第 62–122 行（见模块 docstring）。

def _reference_arm_returns(sc_by_fold, ret, oc, adv, topn, cost_bp,
                           folds, NT=6, EXIT_PCT=0.30):
    """返回 DataFrame(index=date, columns=[long, fold])。"""
    rows = []
    for fold in folds:
        sc = sc_by_fold[fold].copy().dropna()
        sc["signal_date"] = pd.to_datetime(sc["signal_date"])
        by_day = {day: dict(zip(g["PERMNO"], g["score"]))
                  for day, g in sc.groupby("signal_date") if len(g) >= 50}
        del sc
        days = sorted(by_day)
        book = [None] * NT
        for i, day in enumerate(days):
            s, a = by_day[day], adv.get(day, {})
            elig = [p for p in s if p in a and np.isfinite(a[p])]
            if topn and len(elig) > topn:
                elig = sorted(elig, key=lambda p: -a[p])[:topn]
            elig = set(elig)
            s = {p: v for p, v in s.items() if p in elig}
            n = len(s)
            if n < 50:
                continue
            pct = (pd.Series(s).rank() / n).to_dict()
            k = max(1, n // 10)
            order = sorted(pct, key=lambda p: -pct[p])
            j = i % NT
            prev = book[j]
            if prev is None:
                nb, fresh = list(order[:k]), set(order[:k])
                turn = 0.0
            else:
                keep = [p for p in prev if p in pct and pct[p] >= 1 - EXIT_PCT][:k]
                held = set(keep)
                add = [p for p in order if p not in held][:k - len(keep)]
                nb, fresh = keep + add, set(add)
                turn = (k - len(keep)) / k
            book[j] = nb
            if i + 1 >= len(days) or i < NT:
                continue
            nd = days[i + 1]
            rm, om = ret.get(nd, {}), oc.get(nd, {})
            if not rm:
                continue
            cost = 2.0 * (cost_bp / 1e4) * turn / NT
            vals = []
            for t in range(NT):
                nm = book[t]
                if not nm:
                    continue
                rs = [(om.get(p) if (t == j and p in fresh) else rm.get(p)) for p in nm]
                rs = [v for v in rs if v is not None and np.isfinite(v)]
                if rs:
                    vals.append(np.mean(rs))
            if not vals:
                continue
            bench = float(np.mean([rm[p] for p in pct if p in rm]))
            rows.append((nd, float(np.mean(vals)) - bench - cost, fold))
    return pd.DataFrame(rows, columns=["date", "r", "fold"]).set_index("date").sort_index()


# ------------------------------------------------------------------ 合成数据

UNIVERSE = np.arange(10001, 10901, dtype=np.int64)   # 900 个 PERMNO
N_FOLDS, N_DAYS = 3, 40


def _fold_days(f: int) -> pd.DatetimeIndex:
    """三段互不重叠、按时间递增的交易日块（真实折设计也是这样）。"""
    start = pd.Timestamp("2021-01-04") + pd.DateOffset(months=6 * f)
    return pd.bdate_range(start, periods=N_DAYS)


def _make_scores(rng: np.random.Generator, days: pd.DatetimeIndex) -> pd.DataFrame:
    """列序与原脚本 read_parquet(columns=[...]) 一致；含并列、含缺失、含短日。"""
    parts = []
    for d in days:
        # 8% 的日子名字数不足 50 → 触发 by_day 构造处的 len(g) >= 50 过滤
        m = 45 if rng.random() < 0.08 else int(rng.integers(120, 601))
        names = rng.choice(UNIVERSE, size=m, replace=False)
        # 保留两位小数 → 大量并列，逼出 rank(method='average') 与 sorted 稳定性
        score = np.round(rng.normal(size=m), 2)
        score[rng.random(m) < 0.02] = np.nan      # 触发 dropna
        parts.append(pd.DataFrame({"PERMNO": names, "signal_date": d, "score": score}))
    return pd.concat(parts, ignore_index=True)


def _make_prices(rng: np.random.Generator, days: pd.DatetimeIndex):
    ret, oc, adv = {}, {}, {}
    for d in days:
        if rng.random() < 0.05:       # 5% 的日子没有价格 → 触发 `if not rm: continue`
            ret[d], oc[d], adv[d] = {}, {}, {}
            continue
        # 每天只覆盖名字宇宙的一部分 → 触发 `rm.get(p) is None` 的路径。
        # `DlyRet` 本身不放 NaN：`load_prices()` 已 `dropna(subset=["DlyRet"])`，
        # 真实 `ret` 字典里的值必然有限（否则 bench 会整天变 NaN，逐位比较就失效了）。
        k = int(rng.integers(700, 901))
        names = rng.choice(UNIVERSE, size=k, replace=False)
        r = rng.normal(0.0, 0.02, k)
        o = rng.normal(0.0, 0.015, k)
        # 取整到千位 → ADV 大量并列，逼出 top500 截断处 sorted 的稳定性
        a = np.round(rng.lognormal(15.0, 1.0, k), -3)
        o[rng.random(k) < 0.03] = np.nan      # DlyOpen 缺失 → oc 为 NaN
        a[rng.random(k) < 0.02] = np.nan      # rolling min_periods 不足 → ADV 为 NaN
        ret[d] = dict(zip(names, r))
        oc[d] = dict(zip(names, o))
        adv[d] = dict(zip(names, a))
    return ret, oc, adv


def _synthetic(seed: int):
    rng = np.random.default_rng(seed)
    folds = [f"fold{36 + f}" for f in range(N_FOLDS)]
    day_blocks = {f: _fold_days(i) for i, f in enumerate(folds)}
    sc_by_fold = {f: _make_scores(rng, day_blocks[f]) for f in folds}
    all_days = pd.DatetimeIndex(sorted(set().union(*[set(v) for v in day_blocks.values()])))
    ret, oc, adv = _make_prices(rng, all_days)
    return folds, sc_by_fold, ret, oc, adv


# ------------------------------------------------------------------ 逐位一致性

SEEDS = [20260903, 7, 424242, 1, 99991]


@pytest.mark.parametrize("seed", SEEDS)
def test_bitwise_identical_to_reference_per_fold(seed):
    """逐折对拍：`book` 在原实现里也是逐折重置的，所以一折就是一个独立单元。"""
    folds, sc_by_fold, ret, oc, adv = _synthetic(seed)
    for fold in folds:
        ref = _reference_arm_returns({fold: sc_by_fold[fold]}, ret, oc, adv,
                                     500, 8.0, [fold])
        new = frozen_long_only_returns(scores_frame_to_by_day(sc_by_fold[fold]),
                                       ret, oc, adv)
        assert list(ref.index) == list(new.index), f"{fold}: 成交日序列不一致"
        # 先确认没有 NaN——np.array_equal 下 NaN != NaN，若 r 全是 NaN
        # 这条断言会「因为都不相等而失败」或（若用 equal_nan）变成空测试。
        assert np.isfinite(ref["r"].to_numpy()).all(), f"{fold}: 参考实现产出了 NaN"
        assert np.array_equal(ref["r"].to_numpy(), new["r"].to_numpy()), \
            f"{fold}: r 列非逐位相同"
        assert len(ref) > 20, f"{fold}: 合成数据产出的天数太少，测试没覆盖到"


@pytest.mark.parametrize("seed", SEEDS)
def test_bitwise_identical_to_reference_all_folds(seed):
    """三折拼起来对拍（调用方自己贴 fold 列的那种用法）。"""
    folds, sc_by_fold, ret, oc, adv = _synthetic(seed)
    ref = _reference_arm_returns(sc_by_fold, ret, oc, adv, 500, 8.0, folds)
    parts = []
    for fold in folds:
        one = frozen_long_only_returns(scores_frame_to_by_day(sc_by_fold[fold]),
                                       ret, oc, adv)
        parts.append(one.assign(fold=fold))
    new = pd.concat(parts).sort_index()
    assert list(ref.index) == list(new.index)
    assert list(ref["fold"]) == list(new["fold"])
    assert np.array_equal(ref["r"].to_numpy(), new["r"].to_numpy())


@pytest.mark.parametrize("topn,cost_bp,exit_pct,nt", [
    (200, 8.0, 0.30, 6),      # top500 截断绑得更紧
    (500, 22.0, 0.30, 6),     # breakeven 档成本
    (500, 8.0, 0.50, 6),      # 更宽的无交易带
    (500, 8.0, 0.30, 3),      # 更少的套袖
    (0, 8.0, 0.30, 6),        # topn=0 → 不截断（原实现 `if topn and ...` 的短路）
])
def test_bitwise_identical_off_default_params(topn, cost_bp, exit_pct, nt):
    """非默认参数下也要逐位一致——k9 成本前沿会扫这些档。"""
    folds, sc_by_fold, ret, oc, adv = _synthetic(20260903)
    ref = _reference_arm_returns(sc_by_fold, ret, oc, adv, topn, cost_bp, folds,
                                 NT=nt, EXIT_PCT=exit_pct)
    parts = [frozen_long_only_returns(scores_frame_to_by_day(sc_by_fold[f]),
                                      ret, oc, adv, topn=topn, cost_bp=cost_bp,
                                      exit_pct=exit_pct, nt=nt)
             for f in folds]
    new = pd.concat(parts).sort_index()
    assert np.array_equal(ref["r"].to_numpy(), new["r"].to_numpy())


def test_prefiltered_input_is_idempotent():
    """`scores_by_day` 已过滤 / 未过滤，结果必须一样（两处 50 都保留）。"""
    folds, sc_by_fold, ret, oc, adv = _synthetic(20260903)
    sc = sc_by_fold[folds[0]]
    filtered = scores_frame_to_by_day(sc, min_names=50)
    raw = scores_frame_to_by_day(sc, min_names=0)          # 不在构造处过滤
    assert len(raw) > len(filtered), "合成数据里应当有名字数 < 50 的日子"
    a = frozen_long_only_returns(filtered, ret, oc, adv)
    b = frozen_long_only_returns(raw, ret, oc, adv)         # 由本函数补做过滤
    assert list(a.index) == list(b.index)
    assert np.array_equal(a["r"].to_numpy(), b["r"].to_numpy())


def test_diagnostic_columns_are_sane():
    folds, sc_by_fold, ret, oc, adv = _synthetic(20260903)
    out = frozen_long_only_returns(scores_frame_to_by_day(sc_by_fold[folds[0]]),
                                   ret, oc, adv)
    assert list(out.columns) == ["r", "turn", "n_names"]
    assert out["turn"].between(0.0, 1.0).all()
    assert (out["n_names"] > 0).all()
    # 六套袖等权、每袖约 n//10 只；n <= 500 → 每袖 <= 50 只，总去重 <= 300。
    assert out["n_names"].max() <= 300


# ------------------------------------------------------------------ combine

def _frame(day_to_scores: dict) -> pd.DataFrame:
    rows = []
    for d, mapping in day_to_scores.items():
        for p, v in mapping.items():
            rows.append((pd.Timestamp(d), np.int64(p), float(v)))
    return pd.DataFrame(rows, columns=["signal_date", "PERMNO", "score"])


# ---- rank_sum_equal_weight：v2 §4a 语义 --------------------------------
#
# 2026-09-04 起本组测试按 `experiments/signal2_prereg_v2.md` §4a.1 / §4a.2 重写。
# v2 §4a.6 明写：旧实现「任一侧缺失的名字丢弃」与 §4a.1 写死的「缺失/陈旧 →
# 退回 Kronos 单臂分」**不一致**，处置是「实现侧必须把 rank_sum_equal_weight
# 扩展到 §4a.1 + §4a.2 的语义并补测试」。因此下面三条旧断言按 v2 改：
#
# | 旧测试 | 旧断言 | v2 依据 | 现在 |
# |---|---|---|---|
# | `test_rank_sum_is_symmetric` | `f(a,b) == f(b,a)` 恒成立 | §4a.2：`a` 定义 `N_t`、`b` 只在 `V_t` 上排秩 → 函数不再对称 | 拆成「名字集相同时仍对称」+「名字集不同时不对称」 |
# | `test_rank_sum_uses_intersection_only` | 只留交集、单侧日整天丢 | §4a.1：`i ∈ N_t \ V_t → score = rk_kronos`；§4a.2 性质 2「不因缺失被驱逐出可持有集」 | 改为回退断言 |
# | `test_rank_sum_ranks_after_intersection` | 交集外的名字不影响秩 | §4a.2：`V_t ⊆ N_t`，`b` 里 `N_t` 之外的名字本就不进 `V_t` | 断言不变，理由换成 §4a.2 |
#
# 旧语义不是被推翻，是被**推广**：v2 §4a.2 性质 1 要求 `V_t = N_t` 时逐位退化
# 回旧式，由 `test_rank_sum_degenerates_to_v1_bitwise` 钉死。


def _reference_rank_sum_v1(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """`combine.rank_sum_equal_weight` 的**旧实现原样复制**（2026-09-03 版）。

    只用于 `V_t = N_t` 时的逐位退化对拍（v2 §4a.2 性质 1）。一个字符没改，
    包括 `0.5 * (ra.to_numpy() + rb.to_numpy())` 这个浮点表达式的写法。
    """
    from portfolio.combine import COLS as _COLS, _day_series, _normalize
    a, b = _normalize(a), _normalize(b)
    bmap = {day: g for day, g in b.groupby("signal_date")}
    frames = []
    for day, ga in a.groupby("signal_date"):
        gb = bmap.get(day)
        if gb is None:
            continue
        sa, sb = _day_series(ga), _day_series(gb)
        common = sorted(set(sa.index) & set(sb.index))
        if not common:
            continue
        ra = sa.loc[common].rank(pct=True)
        rb = sb.loc[common].rank(pct=True)
        frames.append(pd.DataFrame({
            "signal_date": day,
            "PERMNO": common,
            "score": 0.5 * (ra.to_numpy() + rb.to_numpy()),
        }))
    if not frames:
        return pd.DataFrame(columns=_COLS)
    return pd.concat(frames, ignore_index=True)


def test_rank_sum_is_symmetric_when_name_sets_match():
    """名字集完全相同 → `V_t = N_t`，两边同一把尺，IEEE 加法可交换 → 仍对称。"""
    rng = np.random.default_rng(11)
    a = _frame({"2021-01-04": dict(zip(range(1, 21), rng.normal(size=20))),
                "2021-01-05": dict(zip(range(1, 21), rng.normal(size=20)))})
    b = _frame({"2021-01-04": dict(zip(range(1, 21), rng.normal(size=20))),
                "2021-01-05": dict(zip(range(1, 21), rng.normal(size=20)))})
    ab = rank_sum_equal_weight(a, b)
    ba = rank_sum_equal_weight(b, a)
    pd.testing.assert_frame_equal(ab, ba, check_exact=True)


def test_rank_sum_is_asymmetric_when_name_sets_differ():
    """v2 §4a.2：`a` 定义 `N_t`、`b` 只在有效子集 `V_t` 上排秩 → **故意不对称**。

    旧 docstring 承诺的无条件对称性随 v2 §4a 作废；这条测试把作废钉死，
    免得日后有人「修好」这个不对称。
    """
    a = _frame({"2021-01-04": {1: 0.1, 2: 0.2, 3: 0.3}})
    b = _frame({"2021-01-04": {2: 9.0, 3: 8.0, 4: 7.0}})
    ab = rank_sum_equal_weight(a, b)
    ba = rank_sum_equal_weight(b, a)
    assert set(ab["PERMNO"]) == {1, 2, 3}       # N_t 由 a 定义
    assert set(ba["PERMNO"]) == {2, 3, 4}       # N_t 由 b 定义
    assert len(ab) == len(ba) == 3
    assert not ab["score"].to_numpy().tolist() == ba["score"].to_numpy().tolist()


def test_rank_sum_missing_signal2_falls_back_to_single_arm():
    """v2 §4a.1：`i ∈ N_t \\ V_t → score_combo = pct_rank_t(score_kronos)`。

    旧语义把这些名字丢弃（v2 §4a.6 明列为与预注册**不一致**的实现缺陷）。
    """
    a = _frame({"2021-01-04": {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}})
    b = _frame({"2021-01-04": {2: 9.0, 3: 8.0}})      # 只有 2、3 有信号#2
    out = rank_sum_equal_weight(a, b)
    assert list(out.columns) == ["signal_date", "PERMNO", "score", "used_signal2"]
    assert set(out["PERMNO"]) == {1, 2, 3, 4}, "N_t 的名字一个都不许丢"
    used = dict(zip(out["PERMNO"], out["used_signal2"]))
    assert used == {1: False, 2: True, 3: True, 4: False}
    s = dict(zip(out["PERMNO"], out["score"]))
    # rk_a 在 N_t={1,2,3,4} 上 = .25/.50/.75/1.00
    # rk_b 在 V_t={2,3} 上：b 里 2 的分更高 → 2→1.0、3→0.5
    assert s[1] == pytest.approx(0.25)                 # 回退单臂分
    assert s[4] == pytest.approx(1.00)                 # 回退单臂分
    assert s[2] == pytest.approx(0.5 * (0.50 + 1.0))
    assert s[3] == pytest.approx(0.5 * (0.75 + 0.5))


def test_rank_sum_carries_signal2_forward_then_falls_back_whole_day():
    """信号#2 只在部分日子有观测 → 31 日内前推使用，超期整天回退单臂分。

    - `2021-01-05`：距 b 的 01-04 只有 1 个日历日 → 前推使用（`used=True`）；
    - `2021-03-01`：距 01-04 有 56 个日历日 → 整天回退（`used=False`），
      **但名字一个不丢**（旧语义会整天丢弃）；
    - `b` 有而 `a` 无的 `2021-06-30`：仍然丢——`N_t` 由 `a` 定义。
    """
    a = _frame({"2021-01-04": {1: 0.1, 2: 0.2, 3: 0.3},
                "2021-01-05": {1: 0.5, 2: 0.4, 3: 0.3},
                "2021-03-01": {1: 0.5, 2: 0.4, 3: 0.3}})
    b = _frame({"2021-01-04": {1: 9.0, 2: 8.0, 3: 7.0},
                "2021-06-30": {1: 1.0, 2: 2.0, 3: 3.0}})
    out = rank_sum_equal_weight(a, b)
    assert set(out["signal_date"]) == {pd.Timestamp("2021-01-04"),
                                       pd.Timestamp("2021-01-05"),
                                       pd.Timestamp("2021-03-01")}
    got = {d: g["used_signal2"].tolist()
           for d, g in out.groupby("signal_date")}
    assert got[pd.Timestamp("2021-01-04")] == [True, True, True]
    assert got[pd.Timestamp("2021-01-05")] == [True, True, True]   # 前推 1 日
    assert got[pd.Timestamp("2021-03-01")] == [False, False, False]  # 56 日，陈旧
    d3 = out[out["signal_date"] == pd.Timestamp("2021-03-01")]
    assert np.allclose(d3["score"].to_numpy(), [1.0, 2 / 3, 1 / 3])  # = rk_a


def test_rank_sum_stale_gate_is_31_calendar_days():
    """v2 §4a.1 的陈旧门：**最近一次有效值距 signal_date 不超过 31 个日历日**。

    31 是预注册常数（v2 §5 不做清单：「不调 … 陈旧门 31 日」），这里只打边界，
    不是在调它。闭区间：恰好 31 日仍有效。
    """
    b = _frame({"2021-01-01": {1: 4.0, 2: 3.0, 3: 2.0, 4: 1.0}})
    names = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}
    # a 与 b 的名次完全相反 → 有效时合成分全等于 0.625（秩和守恒）
    for day, valid in [("2021-01-31", True),    # 30 个日历日 → 有效
                       ("2021-02-01", True),    # 31 个日历日 → 边界，仍有效
                       ("2021-02-02", False)]:  # 32 个日历日 → 陈旧
        out = rank_sum_equal_weight(_frame({day: names}), b)
        assert set(out["PERMNO"]) == {1, 2, 3, 4}, f"{day}: 名字被丢了"
        assert bool(out["used_signal2"].all()) is valid, f"{day}: 陈旧门判错"
        assert bool(out["used_signal2"].any()) is valid, f"{day}: 陈旧门判错"
        if valid:
            assert np.allclose(out["score"].to_numpy(), 0.625)
        else:
            # 回退单臂分 = a 在 N_t 上的百分位秩
            assert np.allclose(out["score"].to_numpy(), [0.25, 0.5, 0.75, 1.0])


def test_rank_sum_stale_days_is_a_parameter_not_a_hardcode():
    """`stale_days` 是关键字参数（测试要能打边界），但默认值 31 是预注册常数。"""
    import inspect

    from portfolio.combine import rank_sum_equal_weight as f
    sig = inspect.signature(f)
    assert sig.parameters["stale_days"].default == 31
    assert sig.parameters["stale_days"].kind is inspect.Parameter.KEYWORD_ONLY
    # 前两个参数仍是位置参数 → 签名向后兼容
    assert [p for p, v in sig.parameters.items()
            if v.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD] == ["a", "b"]
    b = _frame({"2021-01-01": {1: 4.0, 2: 3.0, 3: 2.0, 4: 1.0}})
    a = _frame({"2021-02-02": {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}})   # 32 个日历日
    assert not rank_sum_equal_weight(a, b)["used_signal2"].any()
    assert rank_sum_equal_weight(a, b, stale_days=32)["used_signal2"].all()


def test_rank_sum_opposite_signals_cancel():
    """完全相反的两条信号 → 合成分数全相等（秩和守恒）。"""
    a = _frame({"2021-01-04": {10: 1.0, 20: 2.0, 30: 3.0}})
    b = _frame({"2021-01-04": {10: 3.0, 20: 2.0, 30: 1.0}})
    out = rank_sum_equal_weight(a, b)
    assert out["used_signal2"].all()
    assert np.allclose(out["score"].to_numpy(), 2.0 / 3.0)


def test_rank_sum_ranks_only_within_the_valid_subset():
    """v2 §4a.2：`V_t ⊆ N_t`——`b` 里 `N_t` 之外的名字不进 `V_t`，也不影响秩。

    （旧测试名为 `..._ranks_after_intersection`，断言不变，理由换成 §4a.2。）
    """
    a = _frame({"2021-01-04": {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}})
    b_small = _frame({"2021-01-04": {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}})
    b_big = _frame({"2021-01-04": {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0,
                                   99: -5.0, 98: 99.0}})
    o1 = rank_sum_equal_weight(a, b_small)
    o2 = rank_sum_equal_weight(a, b_big)
    pd.testing.assert_frame_equal(o1, o2, check_exact=True)


@pytest.mark.parametrize("seed", [20260904, 3, 777])
def test_rank_sum_degenerates_to_v1_bitwise(seed):
    """v2 §4a.2 性质 1：`V_t = N_t` 时**逐位退化**为旧语义。

    判据 `np.array_equal`（不是 `allclose`）——新式必须是旧式的严格推广，
    合成分不许动最后一位，否则此前所有基于旧式的读数都不可比。
    """
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2021-01-04", periods=12)
    names = list(range(101, 141))
    # 同名字集、同日集 → V_t = N_t。分数取两位小数 → 大量并列，逼出
    # rank(method='average') 的并列处理。
    a = _frame({d: dict(zip(names, np.round(rng.normal(size=len(names)), 2)))
                for d in days})
    b = _frame({d: dict(zip(names, np.round(rng.normal(size=len(names)), 2)))
                for d in days})
    new = rank_sum_equal_weight(a, b)
    ref = _reference_rank_sum_v1(a, b)
    assert new["used_signal2"].all(), "V_t = N_t 时不该有任何回退行"
    assert list(new["signal_date"]) == list(ref["signal_date"])
    assert list(new["PERMNO"]) == list(ref["PERMNO"])
    assert np.array_equal(new["score"].to_numpy(), ref["score"].to_numpy()), \
        "score 列非逐位相同"


def test_rank_sum_empty_inputs():
    empty = _frame({})
    a = _frame({"2021-01-04": {1: 1.0, 2: 2.0}})
    out = rank_sum_equal_weight(a, empty)
    assert list(out.columns) == ["signal_date", "PERMNO", "score", "used_signal2"]
    assert not out["used_signal2"].any()
    assert np.allclose(out["score"].to_numpy(), [0.5, 1.0])   # 全部回退单臂分
    assert rank_sum_equal_weight(empty, a).empty


def test_rank_sum_output_feeds_construction_unchanged():
    """多出来的 `used_signal2` 列不得影响下游构造（三列契约）。"""
    rng = np.random.default_rng(5)
    names = list(range(1, 61))
    a = _frame({"2021-01-04": dict(zip(names, rng.normal(size=len(names))))})
    b = _frame({"2021-01-04": dict(zip(names[:30], rng.normal(size=30)))})
    out = rank_sum_equal_weight(a, b)
    by_day = scores_frame_to_by_day(out, min_names=50)
    assert len(by_day[pd.Timestamp("2021-01-04")]) == 60


def test_slow_filter_keep_frac_boundary():
    """`>=` 边界：4 个名字、keep_frac=0.5 → pct 秩 .25/.5/.75/1.0，留 3 个。"""
    fast = _frame({"2021-01-04": {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}})
    slow = _frame({"2021-01-04": {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}})
    out = slow_filter(fast, slow)
    assert set(out["PERMNO"]) == {2, 3, 4}
    # keep_frac=0.25 → 只留 pct 秩 >= 0.75 的两个
    assert set(slow_filter(fast, slow, keep_frac=0.25)["PERMNO"]) == {3, 4}
    # keep_frac=1.0 → 全留
    assert set(slow_filter(fast, slow, keep_frac=1.0)["PERMNO"]) == {1, 2, 3, 4}


def test_slow_filter_staleness():
    """陈旧线是「> stale_days 个交易日」：第 21 天留、第 22 天丢。"""
    days = pd.bdate_range("2021-01-04", periods=30)
    fast = _frame({d: {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0} for d in days})
    slow = _frame({days[0]: {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}})
    out = slow_filter(fast, slow, stale_days=21)
    kept_days = set(out["signal_date"])
    assert days[21] in kept_days
    assert days[22] not in kept_days
    assert kept_days == set(days[:22])


def test_slow_filter_no_prior_slow_period_drops_day():
    days = pd.bdate_range("2021-01-04", periods=5)
    fast = _frame({d: {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0} for d in days})
    slow = _frame({days[2]: {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}})
    out = slow_filter(fast, slow)
    assert set(out["signal_date"]) == set(days[2:])       # 前两天没有 <= d 的慢期


def test_slow_filter_preserves_fast_scores_and_k_override():
    fast = _frame({"2021-01-04": {1: 0.11, 2: 0.22, 3: 0.33, 4: 0.44}})
    slow = _frame({"2021-01-04": {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}})
    out = slow_filter(fast, slow)
    assert "k_override" not in out.columns             # 默认不带
    assert dict(zip(out["PERMNO"], out["score"])) == {2: 0.22, 3: 0.33, 4: 0.44}
    out2 = slow_filter(fast, slow, hold_count=50)
    assert (out2["k_override"] == 50).all()
    assert out2.attrs["k_override"] == 50
    # 构造层默认不读 k_override：三列契约不变，能直接喂进 by_day 构造。
    assert scores_frame_to_by_day(out2, min_names=1)


def test_spearman_by_day_known_cases():
    a = _frame({"2021-01-04": {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0},
                "2021-01-05": {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0},
                "2021-01-06": {1: 1.0, 2: 2.0}})
    b = _frame({"2021-01-04": {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0},
                "2021-01-05": {1: 4.0, 2: 3.0, 3: 2.0, 4: 1.0},
                "2021-01-06": {1: 1.0, 2: 2.0}})
    s = spearman_by_day(a, b)
    assert s.loc[pd.Timestamp("2021-01-04")] == pytest.approx(1.0)
    assert s.loc[pd.Timestamp("2021-01-05")] == pytest.approx(-1.0)
    assert np.isnan(s.loc[pd.Timestamp("2021-01-06")])   # 共同名字 < 3


def test_spearman_zero_variance_is_nan():
    a = _frame({"2021-01-04": {1: 1.0, 2: 2.0, 3: 3.0}})
    b = _frame({"2021-01-04": {1: 5.0, 2: 5.0, 3: 5.0}})
    assert np.isnan(spearman_by_day(a, b).iloc[0])
