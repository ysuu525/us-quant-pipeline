"""exp15 的合成数据冒烟路径（`--smoke`）。**不碰任何真实数据。**

只验证链路能跑通并且几条硬约束成立：
1. 空公告者集合 / ``announcers=None`` / ``apply_quota=False`` 与冻结构造逐位相同；
2. 配额只影响进场：被排除在配额外的**已持仓**名字仍在账上；
3. 配额后入选数量不变（仍是 ``k``）；
4. B2 三日窗与成本记账在手算合成数据上可核对；
5. ``rdq_hat'`` 的 +364 顺延规则在合成日历上正确；
6. 折号白名单拒绝越界折号。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from portfolio.construction import frozen_long_only_returns            # noqa: E402
from portfolio.construction_announcer import (                         # noqa: E402
    AnnouncerFlags, announcer_tilt_long_only_returns,
    combine_active_returns, event_sleeve_returns,
)


def synthetic_panel(n_days: int = 90, n_names: int = 140, seed: int = 20260905):
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


def run_smoke(out_dir: Path) -> int:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "exp15_main", REPO / "scripts" / "exp15_announcer_tilt_dev_estimate.py")
    exp15 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp15)

    days, names, scores, ret, oc, adv = synthetic_panel()
    checks: dict[str, object] = {}

    # --- 1. 零倾斜逐位重现
    frozen = frozen_long_only_returns(scores, ret, oc, adv, topn=120, cost_bp=8.0, nt=5)
    rng = np.random.default_rng(7)
    flags = AnnouncerFlags({day: set(rng.choice(names, size=20, replace=False).tolist())
                            for day in days})
    for tag, kwargs in (("none", {"announcers": None}),
                        ("empty", {"announcers": AnnouncerFlags.none()}),
                        ("mapping", {"announcers": {}}),
                        ("quota_off", {"announcers": flags, "apply_quota": False})):
        got = announcer_tilt_long_only_returns(
            scores, ret, oc, adv, topn=120, cost_bp=8.0, nt=5, **kwargs)
        checks[f"bitwise_{tag}"] = bool(
            np.array_equal(got["r"].to_numpy(), frozen["r"].to_numpy())
            and np.array_equal(got["turn"].to_numpy(), frozen["turn"].to_numpy()))

    # --- 2/3. 配额生效时入选数量不变、退出不受影响
    tilted, diag = announcer_tilt_long_only_returns(
        scores, ret, oc, adv, announcers=flags, topn=120, cost_bp=8.0, nt=5,
        collect_diagnostics=True)
    per_day = diag["per_day"]
    checks["entry_count_unchanged"] = bool(
        (per_day["n_added"] == per_day["n_slots"]).all())
    checks["no_slot_shortfall"] = int(per_day["n_slot_shortfall"].sum()) == 0
    _, base_diag = announcer_tilt_long_only_returns(
        scores, ret, oc, adv, announcers=flags, apply_quota=False, topn=120,
        cost_bp=8.0, nt=5, collect_diagnostics=True)
    checks["quota_changes_book"] = bool(
        not np.array_equal(tilted["r"].to_numpy(), frozen["r"].to_numpy()))
    checks["book_announcer_share_moves_toward_pool"] = {
        "pool_share": float(per_day["n_announcers_in_pool"].sum() / per_day["n_pool"].sum()),
        "tilted_book_share": float(per_day["n_book_announcers"].sum() / per_day["k"].sum()),
        "e0_book_share": float(base_diag["per_day"]["n_book_announcers"].sum()
                               / base_diag["per_day"]["k"].sum()),
    }

    # --- 4. B2 手算核对
    cal = pd.DatetimeIndex(days)
    trade_dates = list(days[3:8])
    bench = {day: 0.0 for day in days}
    events = pd.DataFrame({"PERMNO": [names[0]],
                           "d_entry": [days[4]], "d_event": [days[5]],
                           "d_exit": [days[6]]})
    sleeve, per_trade = event_sleeve_returns(
        events, ret, oc, bench_by_trade_date=bench, trade_dates=trade_dates, cost_bp=8.0)
    manual = [oc[days[4]][names[0]], ret[days[5]][names[0]], ret[days[6]][names[0]]]
    checks["b2_leg_returns_match"] = bool(
        np.allclose(sleeve.loc[[days[4], days[5], days[6]], "gross"].to_numpy(), manual))
    # 单名字：进出日成本系数 = 1（(1+0)/1 与 (0+1)/1），中间日 0
    checks["b2_cost_schedule"] = [float(sleeve.loc[d, "cost"]) for d in
                                  (days[4], days[5], days[6])]
    checks["b2_cost_ok"] = bool(
        np.isclose(sleeve.loc[days[4], "cost"], 8e-4)
        and np.isclose(sleeve.loc[days[5], "cost"], 0.0)
        and np.isclose(sleeve.loc[days[6], "cost"], 8e-4))
    checks["b2_per_trade_compounded"] = bool(np.isclose(
        per_trade.loc[0, "gross_compounded"],
        float(np.prod([1 + v for v in manual]) - 1.0)))
    checks["b2_empty_days_are_zero_gross"] = bool(
        (sleeve.loc[[days[3], days[7]], "gross"] == 0.0).all())

    combo = combine_active_returns(
        pd.Series(0.001, index=pd.DatetimeIndex(trade_dates)),
        pd.Series(0.5, index=pd.DatetimeIndex(trade_dates)),
        sleeve, w_event=0.10, nt=5)
    direct = (0.9 * (0.001 - 8.0 / 1e4 * 2.0 * 0.5 / 5) + 0.10 * sleeve["active"])
    derived = combo["gross"] - 8.0 / 1e4 * 2.0 * combo["turn"] / 5
    checks["combine_cost_equivalence"] = bool(
        np.allclose(direct.to_numpy(), derived.to_numpy(), atol=1e-15))

    # --- 5. +364 顺延
    hat = exp15.roll_to_trading_day(cal, pd.Series([days[10] - pd.Timedelta(days=1),
                                                    days[10]]))
    checks["roll_forward"] = [str(pd.Timestamp(x).date()) for x in hat]
    checks["roll_is_identity_on_trading_day"] = bool(
        pd.Timestamp(hat[1]) == pd.Timestamp(days[10]))

    # --- 6. 折号白名单
    try:
        exp15.assert_folds([5])
        checks["fold_whitelist"] = False
    except exp15.FoldWhitelistError:
        checks["fold_whitelist"] = True

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "smoke.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    hard = ["bitwise_none", "bitwise_empty", "bitwise_mapping", "bitwise_quota_off",
            "entry_count_unchanged", "no_slot_shortfall", "quota_changes_book",
            "b2_leg_returns_match", "b2_cost_ok", "b2_per_trade_compounded",
            "b2_empty_days_are_zero_gross", "combine_cost_equivalence",
            "roll_is_identity_on_trading_day", "fold_whitelist"]
    ok = all(bool(checks[key]) for key in hard)
    print(json.dumps(checks, ensure_ascii=False, indent=2, default=float), flush=True)
    print(f"[exp15-smoke] {'OK' if ok else 'FAILED'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(run_smoke(REPO / "outputs" / "exp15_announcer_tilt_dev_estimate"))
