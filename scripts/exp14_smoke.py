"""exp14 的合成数据冒烟路径（`--smoke`）。**不碰任何真实数据。**

只验证链路能跑通并且几条硬约束成立：
1. 全 True 掩码 / ``eligible=None`` 与冻结构造逐位相同；
2. 掩码只影响进场：被掩码的已持仓名字仍在账上；
3. 名额递补后入选数量不变；
4. Rule E 的窗口判定与逐日暴力实现一致；
5. 折号白名单拒绝越界折号。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from portfolio.construction import frozen_long_only_returns          # noqa: E402
from portfolio.construction_rule_e import (                          # noqa: E402
    EligibilityMask, rule_e_long_only_returns,
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


def brute_force_blocked(calendar, signal_day, event, slack_days, holding_days=6):
    """逐日暴力：窗口内是否存在交易日落在 [event − slack, event + slack]。"""
    if pd.isna(event):
        return False
    pos = int(np.searchsorted(calendar.to_numpy("datetime64[ns]"),
                              np.datetime64(pd.Timestamp(signal_day), "ns"), "left"))
    if pos + holding_days >= len(calendar):
        return False
    lo = pd.Timestamp(event) - pd.Timedelta(days=slack_days)
    hi = pd.Timestamp(event) + pd.Timedelta(days=slack_days)
    return any(lo <= calendar[pos + off] <= hi for off in range(1, holding_days + 1))


def run_smoke(out_dir: Path) -> int:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "exp14_main", REPO / "scripts" / "exp14_rule_e_dev_estimate.py")
    exp14 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp14)

    days, names, scores, ret, oc, adv = synthetic_panel()
    checks: dict[str, object] = {}

    # 1) 全 True 掩码逐位重现
    frozen = frozen_long_only_returns(scores, ret, oc, adv, topn=500, cost_bp=8.0,
                                      exit_pct=0.30, nt=5, min_names=50)
    dense = pd.DataFrame([(d, p, True) for d in days for p in names],
                         columns=["signal_date", "PERMNO", "eligible"])
    all_true = EligibilityMask.from_frame(dense)
    masked = rule_e_long_only_returns(scores, ret, oc, adv, eligible=all_true,
                                      topn=500, cost_bp=8.0, exit_pct=0.30, nt=5,
                                      min_names=50)
    none_mask = rule_e_long_only_returns(scores, ret, oc, adv, eligible=None,
                                         topn=500, cost_bp=8.0, exit_pct=0.30, nt=5,
                                         min_names=50)
    identical = (frozen.index.equals(masked.index)
                 and np.array_equal(frozen.to_numpy(), masked.to_numpy())
                 and np.array_equal(frozen.to_numpy(), none_mask.to_numpy()))
    checks["all_true_mask_bitwise_identical"] = bool(identical)
    if not identical:
        raise AssertionError("全 True 掩码未能逐位重现冻结构造")

    # 2) 掩码只影响进场：随机掩掉一半名字，账上仍存在被掩码的持仓名字
    rng = np.random.default_rng(7)
    blocked = {d: frozenset(rng.choice(names, size=n_names_blocked, replace=False).tolist())
               for d, n_names_blocked in zip(days, [len(names) // 2] * len(days))}
    _, diag = rule_e_long_only_returns(scores, ret, oc, adv,
                                       eligible=EligibilityMask(blocked),
                                       topn=500, cost_bp=8.0, exit_pct=0.30, nt=5,
                                       min_names=50, collect_diagnostics=True)
    held_blocked = 0
    for trade_date, held in diag["held_by_trade_date"].items():
        signal_day = max(d for d in days if d < trade_date)
        held_blocked += len(held & blocked[signal_day])
    checks["held_names_still_blocked_present"] = int(held_blocked)
    if held_blocked == 0:
        raise AssertionError("掩码疑似触发了退出：没有任何被掩码的名字仍在账上")
    per_day = diag["per_day"]
    checks["slot_shortfall_total"] = int(per_day["n_slot_shortfall"].sum())
    checks["force_exited_total"] = int(per_day["n_force_exited"].sum())
    if int(per_day["n_force_exited"].sum()) != 0:
        raise AssertionError("force_exit=False 时不应有强制退出")

    # 3) 名额递补：可进名字充足时 n_added == n_slots
    light = {d: frozenset(rng.choice(names, size=3, replace=False).tolist()) for d in days}
    _, diag_light = rule_e_long_only_returns(scores, ret, oc, adv,
                                             eligible=EligibilityMask(light),
                                             topn=500, cost_bp=8.0, exit_pct=0.30,
                                             nt=5, min_names=50,
                                             collect_diagnostics=True)
    shortfall = int(diag_light["per_day"]["n_slot_shortfall"].sum())
    checks["light_mask_slot_shortfall"] = shortfall
    if shortfall != 0:
        raise AssertionError("轻掩码下出现名额缺口，递补逻辑有问题")

    # 4) 窗口判定 vs 暴力
    calendar = pd.DatetimeIndex(pd.bdate_range("2020-12-01", periods=400))
    sample_days = list(days[:40])
    events = [pd.Timestamp(d) + pd.Timedelta(days=int(x))
              for d, x in zip(sample_days, rng.integers(-3, 20, len(sample_days)))]
    frame = pd.DataFrame({"signal_date": sample_days, "event": events})
    ok_all = True
    for slack in (0, 1, 2, 3, 5):
        vec = exp14.window_blocked(calendar, frame["signal_date"], frame["event"],
                                   slack_days=slack)
        brute = np.array([brute_force_blocked(calendar, d, e, slack)
                          for d, e in zip(frame["signal_date"], frame["event"])])
        ok_all &= bool(np.array_equal(vec, brute))
    checks["window_blocked_matches_bruteforce"] = bool(ok_all)
    if not ok_all:
        raise AssertionError("window_blocked 与暴力实现不一致")

    # 5) 折号白名单
    rejected = False
    try:
        exp14.assert_folds([36, 20])
    except exp14.FoldWhitelistError:
        rejected = True
    checks["fold_whitelist_rejects_out_of_range"] = rejected
    if not rejected:
        raise AssertionError("折号白名单未拦住越界折号")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "smoke.json").write_text(
        json.dumps({"synthetic_only": True, "checks": checks}, ensure_ascii=False,
                   indent=2), encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    print(f"[exp14-smoke] wrote {out_dir / 'smoke.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_smoke(REPO / "outputs" / "exp14_rule_e_dev_estimate"))
