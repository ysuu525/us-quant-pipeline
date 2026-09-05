"""exp15 的 B1 配额构造与 B2 事件套袖：零倾斜逐位重现、配额只碰进场、成本记账可手算。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from portfolio.construction import frozen_long_only_returns          # noqa: E402
from portfolio.construction_announcer import (                       # noqa: E402
    AnnouncerFlags, announcer_tilt_long_only_returns,
    combine_active_returns, event_sleeve_returns,
)


def synthetic(n_days: int = 80, n_names: int = 130, seed: int = 20260905):
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


DAYS, NAMES, SCORES, RET, OC, ADV = synthetic()


def random_flags(seed: int = 11, size: int = 20) -> AnnouncerFlags:
    rng = np.random.default_rng(seed)
    return AnnouncerFlags({day: set(rng.choice(NAMES, size=size, replace=False).tolist())
                           for day in DAYS})


# --------------------------------------------------------------------------
# 1. 零倾斜逐位重现冻结输出
# --------------------------------------------------------------------------
@pytest.mark.parametrize("nt", [5, 6])
@pytest.mark.parametrize("cost_bp", [0.0, 8.0])
@pytest.mark.parametrize("entry", ["none", "empty_object", "empty_mapping"])
def test_zero_tilt_reproduces_frozen_bitwise(nt, cost_bp, entry):
    frozen = frozen_long_only_returns(SCORES, RET, OC, ADV, topn=120,
                                      cost_bp=cost_bp, nt=nt)
    announcers = {"none": None,
                  "empty_object": AnnouncerFlags.none(),
                  "empty_mapping": {}}[entry]
    got = announcer_tilt_long_only_returns(SCORES, RET, OC, ADV, topn=120,
                                           cost_bp=cost_bp, nt=nt,
                                           announcers=announcers)
    assert got.index.equals(frozen.index)
    for column in ("r", "turn", "n_names"):
        assert np.array_equal(got[column].to_numpy(), frozen[column].to_numpy())


def test_apply_quota_false_reproduces_frozen_even_with_flags():
    """诊断开关：旗标非空但 apply_quota=False 时输出仍逐位等于冻结构造。"""
    frozen = frozen_long_only_returns(SCORES, RET, OC, ADV, topn=120, cost_bp=8.0, nt=5)
    got, diag = announcer_tilt_long_only_returns(
        SCORES, RET, OC, ADV, topn=120, cost_bp=8.0, nt=5,
        announcers=random_flags(), apply_quota=False, collect_diagnostics=True)
    assert np.array_equal(got["r"].to_numpy(), frozen["r"].to_numpy())
    assert np.array_equal(got["turn"].to_numpy(), frozen["turn"].to_numpy())
    # 但诊断仍数出了簿里的公告者（否则拿不到 E0 的对照量）
    assert diag["per_day"]["n_book_announcers"].sum() > 0


def test_days_without_flags_are_bitwise_frozen_even_when_other_days_are_flagged():
    """只有一半日子有旗标：无旗标那些重平衡日的进场名单与冻结完全一致。"""
    flagged_days = list(DAYS[::2])
    rng = np.random.default_rng(3)
    flags = AnnouncerFlags({day: set(rng.choice(NAMES, 15, replace=False).tolist())
                            for day in flagged_days})
    _, tilted = announcer_tilt_long_only_returns(
        SCORES, RET, OC, ADV, topn=120, cost_bp=0.0, nt=5, announcers=flags,
        collect_diagnostics=True)
    per_day = tilted["per_day"]
    clean = per_day[~per_day["signal_date"].isin(flagged_days)]
    assert len(clean) > 0
    assert (clean["n_announcers_in_pool"] == 0).all()
    assert (clean["k_E"] == 0).all()
    assert (clean["need_announcers"] == 0).all()


# --------------------------------------------------------------------------
# 2. 配额只影响进场，不影响退出；入选数量不变
# --------------------------------------------------------------------------
def _deterministic_panel(n_days: int = 4, n_names: int = 50):
    """名次写死的小面板：分数 = −序号 ⇒ `order` 恒为 [1, 2, ..., n]。"""
    days = pd.bdate_range("2022-03-01", periods=n_days)
    names = list(range(1, n_names + 1))
    scores = {day: {p: float(-p) for p in names} for day in days}
    ret = {day: {p: 0.001 for p in names} for day in days}
    oc = {day: {p: 0.0005 for p in names} for day in days}
    adv = {day: {p: 1e7 for p in names} for day in days}
    return days, names, scores, ret, oc, adv


def test_kept_announcers_over_quota_are_never_sold():
    """写死的极端格：簿里 5 只全是公告者、配额 k_E = 0（s_t = 0.1、k = 5）。
    配额只有进场一个动作 ⇒ 5 只**一个都不许被卖**，`n_keep` 仍是冻结规则给的 5。"""
    days, names, scores, ret, oc, adv = _deterministic_panel()
    # 第 0 天没有旗标 ⇒ 簿 = order 的前 5 名；之后这 5 只全成了公告者
    flags = AnnouncerFlags({day: set(names[:5]) for day in days[1:]})
    _, diag = announcer_tilt_long_only_returns(
        scores, ret, oc, adv, topn=None, cost_bp=0.0, nt=1, min_names=50,
        announcers=flags, collect_diagnostics=True)
    per_day = diag["per_day"]
    assert (per_day["k"] == 5).all()
    later = per_day.iloc[1:]                    # 第 0 天 prev is None
    assert (later["n_announcers_in_pool"] == 5).all()
    assert (later["k_E"] == 0).all()            # round(5 * 0.1) = 0（银行家舍入）
    assert (later["n_keep"] == 5).all()         # 冻结退出规则：pct >= 0.7 全留
    assert (later["n_keep_announcers"] == 5).all()
    assert (later["n_slots"] == 0).all()
    assert (later["n_book_announcers"] == 5).all()   # 超配额也不强退


def test_quota_leaves_keep_untouched_on_the_first_flagged_day():
    """只在中间某一天挂旗标：该日之前逐位等于冻结构造，该日的 `keep` / 名额数
    与关配额时完全相同（退出不受影响），只有进场的**成分**变了。"""
    flag_day = DAYS[40]
    rng = np.random.default_rng(21)
    flags = AnnouncerFlags({flag_day: set(rng.choice(NAMES, 30, replace=False).tolist())})
    _, tilted = announcer_tilt_long_only_returns(
        SCORES, RET, OC, ADV, topn=120, cost_bp=0.0, nt=5, announcers=flags,
        collect_diagnostics=True)
    _, base = announcer_tilt_long_only_returns(
        SCORES, RET, OC, ADV, topn=120, cost_bp=0.0, nt=5, announcers=flags,
        apply_quota=False, collect_diagnostics=True)
    t_rows, b_rows = tilted["per_day"], base["per_day"]
    before = t_rows["signal_date"] < flag_day
    for column in ("n_pool", "k", "n_keep", "n_slots", "n_added"):
        assert np.array_equal(t_rows.loc[before, column].to_numpy(),
                              b_rows.loc[before, column].to_numpy())
    t_flag = t_rows.loc[t_rows["signal_date"] == flag_day].iloc[0]
    b_flag = b_rows.loc[b_rows["signal_date"] == flag_day].iloc[0]
    assert t_flag["n_keep"] == b_flag["n_keep"]          # 退出未被触碰
    assert t_flag["n_slots"] == b_flag["n_slots"]        # 名额未变
    assert t_flag["n_added"] == b_flag["n_added"]        # 入选数量未变
    assert t_flag["n_added_announcers"] != b_flag["n_added_announcers"]


def test_full_pool_of_announcers_reproduces_frozen_bitwise():
    """全池都是公告者 ⇒ s_t = 1 ⇒ k_E = k ⇒ 进场全从公告者取 ⇒ 与冻结逐位相同。"""
    flags_all = AnnouncerFlags({day: set(NAMES) for day in DAYS})
    frozen = frozen_long_only_returns(SCORES, RET, OC, ADV, topn=120, cost_bp=0.0, nt=5)
    got, full = announcer_tilt_long_only_returns(
        SCORES, RET, OC, ADV, topn=120, cost_bp=0.0, nt=5, announcers=flags_all,
        collect_diagnostics=True)
    assert np.array_equal(got["r"].to_numpy(), frozen["r"].to_numpy())
    assert np.array_equal(got["turn"].to_numpy(), frozen["turn"].to_numpy())
    assert (full["per_day"]["k_E"] == full["per_day"]["k"]).all()


def test_entry_count_is_unchanged_by_the_quota():
    _, diag = announcer_tilt_long_only_returns(
        SCORES, RET, OC, ADV, topn=120, cost_bp=0.0, nt=5,
        announcers=random_flags(), collect_diagnostics=True)
    per_day = diag["per_day"]
    assert (per_day["n_added"] == per_day["n_slots"]).all()
    assert int(per_day["n_slot_shortfall"].sum()) == 0
    assert (per_day["n_keep"] + per_day["n_added"] <= per_day["k"]).all()


def test_quota_targets_round_k_times_pool_share():
    _, diag = announcer_tilt_long_only_returns(
        SCORES, RET, OC, ADV, topn=120, cost_bp=0.0, nt=5,
        announcers=random_flags(), collect_diagnostics=True)
    per_day = diag["per_day"]
    expected = [int(round(k * s)) for k, s in
                zip(per_day["k"], per_day["announcer_share_of_pool"])]
    assert list(per_day["k_E"]) == expected
    # 需要补的公告者数恰为 min(max(k_E − 留仓公告者, 0), 名额数)
    need = np.minimum(np.maximum(per_day["k_E"] - per_day["n_keep_announcers"], 0),
                      per_day["n_slots"])
    assert np.array_equal(per_day["need_announcers"].to_numpy(), need.to_numpy())
    # 无缺口时进场的公告者数恰等于 need；簿内公告者 = 留仓公告者 + 进场公告者
    assert int(per_day["announcer_shortfall"].sum()) == 0
    assert np.array_equal(per_day["n_added_announcers"].to_numpy(), need.to_numpy())
    assert np.array_equal(
        per_day["n_book_announcers"].to_numpy(),
        (per_day["n_keep_announcers"] + per_day["n_added_announcers"]).to_numpy())


def test_quota_moves_the_book_exposure_toward_the_pool_share():
    flags = random_flags(seed=13, size=25)
    _, tilted = announcer_tilt_long_only_returns(
        SCORES, RET, OC, ADV, topn=120, cost_bp=0.0, nt=5, announcers=flags,
        collect_diagnostics=True)
    _, base = announcer_tilt_long_only_returns(
        SCORES, RET, OC, ADV, topn=120, cost_bp=0.0, nt=5, announcers=flags,
        apply_quota=False, collect_diagnostics=True)
    pool = float(tilted["per_day"]["n_announcers_in_pool"].sum()
                 / tilted["per_day"]["n_pool"].sum())
    tilted_share = float(tilted["per_day"]["n_book_announcers"].sum()
                         / tilted["per_day"]["k"].sum())
    base_share = float(base["per_day"]["n_book_announcers"].sum()
                       / base["per_day"]["k"].sum())
    # 三个量都在同一量级；本测试只断言配额确实改变了簿的构成（方向由数据决定）
    assert 0.0 < pool < 1.0
    assert tilted_share != base_share


# --------------------------------------------------------------------------
# 3. B2：窗口与成本记账（合成数据，手算可核）
# --------------------------------------------------------------------------
def test_event_sleeve_eats_open_to_close_on_entry_day_and_full_days_after():
    trade_dates = list(DAYS[2:10])
    bench = {day: 0.0 for day in DAYS}
    events = pd.DataFrame({"PERMNO": [NAMES[0]], "d_entry": [DAYS[4]],
                           "d_event": [DAYS[5]], "d_exit": [DAYS[6]]})
    sleeve, per_trade = event_sleeve_returns(
        events, RET, OC, bench_by_trade_date=bench, trade_dates=trade_dates, cost_bp=8.0)
    assert sleeve.loc[DAYS[4], "gross"] == pytest.approx(OC[DAYS[4]][NAMES[0]])
    assert sleeve.loc[DAYS[5], "gross"] == pytest.approx(RET[DAYS[5]][NAMES[0]])
    assert sleeve.loc[DAYS[6], "gross"] == pytest.approx(RET[DAYS[6]][NAMES[0]])
    assert sleeve.loc[DAYS[3], "gross"] == 0.0 and sleeve.loc[DAYS[3], "n_names"] == 0
    assert sleeve.loc[DAYS[7], "gross"] == 0.0
    # 成本：进出各一次，各在首末贡献日
    assert sleeve.loc[DAYS[4], "cost"] == pytest.approx(8e-4)
    assert sleeve.loc[DAYS[5], "cost"] == pytest.approx(0.0)
    assert sleeve.loc[DAYS[6], "cost"] == pytest.approx(8e-4)
    total_cost = float(sleeve["cost"].sum())
    assert total_cost == pytest.approx(2 * 8e-4)
    # 逐笔复利
    manual = (1 + OC[DAYS[4]][NAMES[0]]) * (1 + RET[DAYS[5]][NAMES[0]]) \
        * (1 + RET[DAYS[6]][NAMES[0]]) - 1.0
    assert per_trade.loc[0, "gross_compounded"] == pytest.approx(manual)
    assert per_trade.loc[0, "n_days"] == 3


def test_event_sleeve_is_equal_weighted_across_names_each_day():
    trade_dates = list(DAYS[2:12])
    bench = {day: 0.0 for day in DAYS}
    events = pd.DataFrame({
        "PERMNO": [NAMES[0], NAMES[1], NAMES[2]],
        "d_entry": [DAYS[4], DAYS[4], DAYS[5]],
        "d_event": [DAYS[5], DAYS[5], DAYS[6]],
        "d_exit": [DAYS[6], DAYS[6], DAYS[7]]})
    sleeve, _ = event_sleeve_returns(
        events, RET, OC, bench_by_trade_date=bench, trade_dates=trade_dates, cost_bp=8.0)
    expected_day5 = np.mean([RET[DAYS[5]][NAMES[0]], RET[DAYS[5]][NAMES[1]],
                             OC[DAYS[5]][NAMES[2]]])
    assert sleeve.loc[DAYS[5], "gross"] == pytest.approx(expected_day5)
    assert sleeve.loc[DAYS[5], "n_names"] == 3
    # 第 5 天：1 个名字进（NAMES[2]），0 个出 ⇒ cost_coef = 1/3
    assert sleeve.loc[DAYS[5], "cost_coef"] == pytest.approx(1 / 3)
    assert sleeve.loc[DAYS[4], "n_entries"] == 2
    assert sleeve.loc[DAYS[6], "n_exits"] == 2


def test_event_sleeve_active_subtracts_the_same_pool_benchmark():
    trade_dates = list(DAYS[2:10])
    bench = {day: 0.001 for day in DAYS}
    events = pd.DataFrame({"PERMNO": [NAMES[0]], "d_entry": [DAYS[4]],
                           "d_event": [DAYS[5]], "d_exit": [DAYS[6]]})
    sleeve, per_trade = event_sleeve_returns(
        events, RET, OC, bench_by_trade_date=bench, trade_dates=trade_dates, cost_bp=8.0)
    assert sleeve.loc[DAYS[5], "active"] == pytest.approx(
        RET[DAYS[5]][NAMES[0]] - 0.001)
    # 空仓日：毛收益 0（现金），主动收益 = −基准（任务书 §4.2 的字面口径）
    assert sleeve.loc[DAYS[3], "active"] == pytest.approx(-0.001)
    assert per_trade.loc[0, "bench_compounded"] == pytest.approx(1.001 ** 3 - 1.0)


def test_event_sleeve_drops_legs_outside_the_trade_date_index():
    trade_dates = list(DAYS[5:8])          # 进场日 DAYS[4] 不在里面
    bench = {day: 0.0 for day in DAYS}
    events = pd.DataFrame({"PERMNO": [NAMES[0]], "d_entry": [DAYS[4]],
                           "d_event": [DAYS[5]], "d_exit": [DAYS[6]]})
    sleeve, per_trade = event_sleeve_returns(
        events, RET, OC, bench_by_trade_date=bench, trade_dates=trade_dates, cost_bp=8.0)
    assert sleeve.attrs["truncated_legs"] == 1
    assert per_trade.loc[0, "n_days"] == 2
    # 成本仍是进出各一次，落在实际贡献的首末日
    assert sleeve.loc[DAYS[5], "cost"] == pytest.approx(8e-4)
    assert sleeve.loc[DAYS[6], "cost"] == pytest.approx(8e-4)


def test_combine_reproduces_the_direct_net_combination_at_any_cost():
    trade_dates = list(DAYS[2:12])
    bench = {day: 0.0005 for day in DAYS}
    events = pd.DataFrame({"PERMNO": [NAMES[0], NAMES[3]],
                           "d_entry": [DAYS[4], DAYS[6]],
                           "d_event": [DAYS[5], DAYS[7]],
                           "d_exit": [DAYS[6], DAYS[8]]})
    sleeve, _ = event_sleeve_returns(
        events, RET, OC, bench_by_trade_date=bench, trade_dates=trade_dates, cost_bp=8.0)
    index = pd.DatetimeIndex(trade_dates)
    base_gross = pd.Series(np.linspace(-0.001, 0.002, len(index)), index=index)
    base_turn = pd.Series(np.linspace(0.2, 0.8, len(index)), index=index)
    for w in (0.05, 0.10, 0.20):
        combo = combine_active_returns(base_gross, base_turn, sleeve, w_event=w, nt=5)
        sleeve_at_8 = sleeve["gross"] - sleeve["bench"] - 8.0 / 1e4 * sleeve["cost_coef"]
        direct = ((1 - w) * (base_gross - 8.0 / 1e4 * 2.0 * base_turn / 5)
                  + w * sleeve_at_8)
        derived = combo["gross"] - 8.0 / 1e4 * 2.0 * combo["turn"] / 5
        assert np.allclose(direct.to_numpy(), derived.to_numpy(), atol=1e-15)
    combo0 = combine_active_returns(base_gross, base_turn, sleeve, w_event=0.0, nt=5)
    assert np.array_equal(combo0["gross"].to_numpy(), base_gross.to_numpy())
    assert np.array_equal(combo0["turn"].to_numpy(), base_turn.to_numpy())


def test_combine_rejects_missing_sleeve_days_and_bad_weights():
    index = pd.DatetimeIndex(list(DAYS[2:6]))
    base_gross = pd.Series(0.001, index=index)
    base_turn = pd.Series(0.4, index=index)
    sleeve = pd.DataFrame({"gross": [0.0], "bench": [0.0], "cost_coef": [0.0]},
                          index=pd.DatetimeIndex([DAYS[2]]))
    with pytest.raises(ValueError):
        combine_active_returns(base_gross, base_turn, sleeve, w_event=0.1, nt=5)
    full = pd.DataFrame({"gross": 0.0, "bench": 0.0, "cost_coef": 0.0}, index=index)
    with pytest.raises(ValueError):
        combine_active_returns(base_gross, base_turn, full, w_event=1.5, nt=5)


def test_announcer_flags_container_semantics():
    flags = AnnouncerFlags({DAYS[0]: {1, 2}, DAYS[1]: set()})
    assert flags.announcers_on(DAYS[0]) == frozenset({1, 2})
    assert flags.announcers_on(DAYS[1]) == frozenset()
    assert flags.announcers_on(DAYS[5]) == frozenset()
    assert flags.is_announcer(DAYS[0], 1) and not flags.is_announcer(DAYS[0], 3)
    assert flags.n_flag_cells == 2 and flags.n_days_with_flag == 1
    frame = pd.DataFrame({"signal_date": [DAYS[0], DAYS[0], DAYS[2]],
                          "PERMNO": [7, 8, 9]})
    built = AnnouncerFlags.from_frame(frame)
    assert built.announcers_on(DAYS[0]) == frozenset({7, 8})
    assert built.n_flag_cells == 3
    with pytest.raises(TypeError):
        announcer_tilt_long_only_returns(SCORES, RET, OC, ADV, announcers=object())


def test_counterfactual_diagnostics_count_the_names_the_quota_swapped():
    """同日反事实：`n_quota_promoted` / `n_quota_displaced` 必须等长（名额不变），
    且关掉配额时恒为 0（进场就是冻结的 add）。"""
    flags = random_flags(seed=17, size=25)
    _, tilted = announcer_tilt_long_only_returns(
        SCORES, RET, OC, ADV, topn=120, cost_bp=0.0, nt=5, announcers=flags,
        collect_diagnostics=True)
    _, base = announcer_tilt_long_only_returns(
        SCORES, RET, OC, ADV, topn=120, cost_bp=0.0, nt=5, announcers=flags,
        apply_quota=False, collect_diagnostics=True)
    t_rows = tilted["per_day"]
    assert np.array_equal(t_rows["n_quota_promoted"].to_numpy(),
                          t_rows["n_quota_displaced"].to_numpy())
    assert int(t_rows["n_quota_promoted"].sum()) > 0
    assert int(base["per_day"]["n_quota_promoted"].sum()) == 0
    assert int(base["per_day"]["n_quota_displaced"].sum()) == 0
    # 关配额时「冻结进场里的公告者数」就是实际进场的公告者数
    assert np.array_equal(base["per_day"]["n_added_announcers"].to_numpy(),
                          base["per_day"]["n_added_announcers_frozen"].to_numpy())
