"""exp14 §2.2 / §3：带 Rule E 掩码的构造必须在全 True 掩码下逐位重现冻结构造，
掩码只影响进场、不触发退出，且名额递补后入选数量不变。
"""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from portfolio.construction import frozen_long_only_returns          # noqa: E402
from portfolio.construction_rule_e import (                          # noqa: E402
    EligibilityMask, rule_e_long_only_returns,
)

NT, TOPN, EXIT_PCT, MIN_NAMES = 5, 500, 0.30, 50


def _panel(n_days: int = 80, n_names: int = 140, seed: int = 20260905):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2021-01-04", periods=n_days)
    names = list(range(10001, 10001 + n_names))
    scores, ret, oc, adv = {}, {}, {}, {}
    for day in days:
        scores[day] = {p: float(v) for p, v in zip(names, rng.normal(size=n_names))}
        ret[day] = {p: float(v) for p, v in zip(names, rng.normal(0, 0.02, n_names))}
        oc[day] = {p: float(v) for p, v in zip(names, rng.normal(0, 0.015, n_names))}
        adv[day] = {p: float(v) for p, v in zip(names, rng.uniform(1e6, 1e8, n_names))}
    return days, names, scores, ret, oc, adv


def _frozen(scores, ret, oc, adv, **kw):
    return frozen_long_only_returns(scores, ret, oc, adv, topn=TOPN, cost_bp=8.0,
                                    exit_pct=EXIT_PCT, nt=NT, min_names=MIN_NAMES, **kw)


def _rule_e(scores, ret, oc, adv, **kw):
    return rule_e_long_only_returns(scores, ret, oc, adv, topn=TOPN, cost_bp=8.0,
                                    exit_pct=EXIT_PCT, nt=NT, min_names=MIN_NAMES, **kw)


# --------------------------------------------------------------------------
# §2.2：全 True 掩码逐位重现
# --------------------------------------------------------------------------
def test_all_true_mask_reproduces_frozen_bit_for_bit():
    days, names, scores, ret, oc, adv = _panel()
    reference = _frozen(scores, ret, oc, adv)
    dense = pd.DataFrame([(d, p, True) for d in days for p in names],
                         columns=["signal_date", "PERMNO", "eligible"])
    mask = EligibilityMask.from_frame(dense)
    assert mask.n_blocked_cells == 0
    got = _rule_e(scores, ret, oc, adv, eligible=mask)
    assert got.index.equals(reference.index)
    assert list(got.columns) == list(reference.columns)
    assert np.array_equal(got.to_numpy(), reference.to_numpy())
    assert np.allclose(got["r"].to_numpy(), reference["r"].to_numpy(), atol=0.0, rtol=0.0)


def test_none_and_empty_mask_reproduce_frozen_bit_for_bit():
    days, names, scores, ret, oc, adv = _panel(seed=11)
    reference = _frozen(scores, ret, oc, adv)
    for mask in (None, EligibilityMask.all_true(), {}, EligibilityMask({})):
        got = _rule_e(scores, ret, oc, adv, eligible=mask)
        assert np.array_equal(got.to_numpy(), reference.to_numpy())


@pytest.mark.parametrize("nt", [5, 6])
@pytest.mark.parametrize("cost_bp", [0.0, 8.0])
def test_bitwise_identity_holds_across_frozen_parameters(nt, cost_bp):
    days, names, scores, ret, oc, adv = _panel(seed=3)
    reference = frozen_long_only_returns(scores, ret, oc, adv, topn=TOPN,
                                         cost_bp=cost_bp, exit_pct=EXIT_PCT, nt=nt,
                                         min_names=MIN_NAMES)
    got = rule_e_long_only_returns(scores, ret, oc, adv, eligible=None, topn=TOPN,
                                   cost_bp=cost_bp, exit_pct=EXIT_PCT, nt=nt,
                                   min_names=MIN_NAMES)
    assert np.array_equal(got.to_numpy(), reference.to_numpy())


# --------------------------------------------------------------------------
# §3：掩码只影响进场
# --------------------------------------------------------------------------
def test_masked_holdings_are_not_forced_out():
    """构造一个被掩码的已持仓名字，断言它仍在账上（掩码不触发退出）。"""
    days, names, scores, ret, oc, adv = _panel(seed=5)
    # 第 1 阶段：无掩码跑一遍，拿到某个交易日的持仓
    _, diag0 = _rule_e(scores, ret, oc, adv, eligible=None, collect_diagnostics=True)
    trade_dates = sorted(diag0["held_by_trade_date"])
    probe_date = trade_dates[len(trade_dates) // 2]
    held0 = diag0["held_by_trade_date"][probe_date]
    assert held0, "基线在探针日没有持仓，测试前提不成立"

    # 第 2 阶段：从探针日对应的信号日起，把这些持仓名字全部掩掉。
    signal_days = [d for d in days if d < probe_date]
    probe_signal_day = signal_days[-1]
    blocked = {d: frozenset(held0) for d in days if d >= probe_signal_day}
    frame, diag = _rule_e(scores, ret, oc, adv, eligible=EligibilityMask(blocked),
                          collect_diagnostics=True)
    held = diag["held_by_trade_date"][probe_date]
    still_held = held & held0
    assert still_held, "被掩码的已持仓名字被错误地清出了账"
    # 未开 force_exit 时，强退计数必须恒为 0
    assert int(diag["per_day"]["n_force_exited"].sum()) == 0


def test_exit_ranking_is_computed_on_the_full_pool():
    """退出判定只看全池 pct，与掩码无关：掩码不改变任何 keep 的大小。"""
    days, names, scores, ret, oc, adv = _panel(seed=9)
    _, diag0 = _rule_e(scores, ret, oc, adv, eligible=None, collect_diagnostics=True)
    rng = np.random.default_rng(4)
    blocked = {d: frozenset(rng.choice(names, size=40, replace=False).tolist())
               for d in days}
    _, diag = _rule_e(scores, ret, oc, adv, eligible=EligibilityMask(blocked),
                      collect_diagnostics=True)
    # 掩码会改变账面（进的名字不同），但「退出排名在全池上算」这条不变：
    # 关掉 force_exit 时 keep 永远等于 keep_before_force_exit。
    assert int(diag["per_day"]["n_force_exited"].sum()) == 0
    assert (diag0["per_day"]["n_pool"].to_numpy()
            == diag["per_day"]["n_pool"].to_numpy()).all()
    assert (diag0["per_day"]["k"].to_numpy() == diag["per_day"]["k"].to_numpy()).all()


def test_force_exit_flag_drops_masked_holdings_only_when_enabled():
    days, names, scores, ret, oc, adv = _panel(seed=13)
    rng = np.random.default_rng(6)
    blocked = {d: frozenset(rng.choice(names, size=60, replace=False).tolist())
               for d in days}
    mask = EligibilityMask(blocked)
    _, diag_off = _rule_e(scores, ret, oc, adv, eligible=mask, force_exit=False,
                          collect_diagnostics=True)
    _, diag_on = _rule_e(scores, ret, oc, adv, eligible=mask, force_exit=True,
                         collect_diagnostics=True)
    assert int(diag_off["per_day"]["n_force_exited"].sum()) == 0
    assert int(diag_on["per_day"]["n_force_exited"].sum()) > 0


# --------------------------------------------------------------------------
# §3：名额递补，入选数量不变
# --------------------------------------------------------------------------
def test_slot_count_is_unchanged_under_a_light_mask():
    """可进名字充足时，补入数量 == 无掩码时的名额数（名额递补）。"""
    days, names, scores, ret, oc, adv = _panel(seed=17)
    rng = np.random.default_rng(2)
    blocked = {d: frozenset(rng.choice(names, size=5, replace=False).tolist())
               for d in days}
    _, diag0 = _rule_e(scores, ret, oc, adv, eligible=None, collect_diagnostics=True)
    _, diag = _rule_e(scores, ret, oc, adv, eligible=EligibilityMask(blocked),
                      collect_diagnostics=True)
    assert int(diag["per_day"]["n_slot_shortfall"].sum()) == 0
    assert (diag["per_day"]["n_added"].to_numpy()
            == diag["per_day"]["n_slots"].to_numpy()).all()
    # 首个建仓日两边的入选数量必须一致
    first0 = diag0["per_day"].iloc[0]
    first = diag["per_day"].iloc[0]
    assert int(first0["n_added"]) == int(first["n_added"]) == int(first["k"])


def test_book_size_per_sleeve_matches_the_frozen_k():
    days, names, scores, ret, oc, adv = _panel(seed=23)
    rng = np.random.default_rng(8)
    blocked = {d: frozenset(rng.choice(names, size=20, replace=False).tolist())
               for d in days}
    _, diag = _rule_e(scores, ret, oc, adv, eligible=EligibilityMask(blocked),
                      collect_diagnostics=True)
    per_day = diag["per_day"]
    # keep + add 的名额上限恒为 k；无缺口时套袖规模恰为 k
    assert (per_day["n_slots"] + per_day["n_keep_before_force_exit"]
            == per_day["k"]).all()
    assert int(per_day["n_slot_shortfall"].sum()) == 0


def test_shortfall_is_reported_when_almost_everything_is_blocked():
    """极端掩码下名额补不满时必须被记账，而不是悄悄改变 k。"""
    days, names, scores, ret, oc, adv = _panel(seed=29)
    blocked = {d: frozenset(names[:-2]) for d in days}
    _, diag = _rule_e(scores, ret, oc, adv, eligible=EligibilityMask(blocked),
                      collect_diagnostics=True)
    per_day = diag["per_day"]
    assert int(per_day["n_slot_shortfall"].sum()) > 0
    assert (per_day["n_added"] <= per_day["n_slots"]).all()
    # 掩码不改变池大小与 k（不会触发 min_names）
    _, diag0 = _rule_e(scores, ret, oc, adv, eligible=None, collect_diagnostics=True)
    assert (diag0["per_day"]["n_pool"].to_numpy()
            == per_day["n_pool"].to_numpy()).all()


# --------------------------------------------------------------------------
# 模块纪律
# --------------------------------------------------------------------------
def test_rule_e_module_never_mentions_the_future_return_variable():
    # 用拼接构造 CLAUDE.md §一.1 点名的列名，免得断言本身污染 grep 结果。
    needle = "lab" + "el"
    source = (REPO / "src" / "portfolio" / "construction_rule_e.py").read_text(
        encoding="utf-8")
    assert needle not in source


def test_frozen_construction_module_is_untouched_by_exp14():
    """exp14 不得改 construction.py：这里锁住冻结口径的几个常量与签名。"""
    signature = inspect.signature(frozen_long_only_returns)
    defaults = {name: parameter.default
                for name, parameter in signature.parameters.items()
                if parameter.default is not inspect.Parameter.empty}
    assert defaults == {"topn": 500, "cost_bp": 8.0, "exit_pct": 0.30,
                        "nt": 6, "min_names": 50}
