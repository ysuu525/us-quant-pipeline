"""执行层测试（`src/execution/`）。

对应 `experiments/cost_pilot_protocol_v1_draft.md` §2--§6 与 `HANDOFF.md` §11.4a/b/c。

**全部用内存里的合成小数据**：不读 `F:\\quant\\processed\\` 下任何 parquet、
不读 `outputs/`、不 import torch / kronos_ft、不发任何网络请求。
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from execution import fees
from execution.alpaca_client import (AlpacaOpgClient, SubmitWindowError,
                                     assert_submit_window)
from execution.collision import (ARM_BOTH, ARM_FT, ARM_ZS, COLLISION_SEED,
                                 collision_winner, resolve_collisions)
from execution.daily_flow import HaltError, run_daily
from execution.orders import ORDER_COLUMNS, append_orders, load_orders
from execution.shadow_ledger import ShadowLedger, reconcile_arms

ET = ZoneInfo("America/New_York")
SRC = str(Path(__file__).resolve().parents[1] / "src")


def _orders(rows, arm):
    """构造一臂的理想订单表。rows = [(date_str, permno, side, notional), ...]"""
    return pd.DataFrame(
        [{"trade_date": pd.Timestamp(d), "PERMNO": p, "symbol": f"S{p}",
          "side": s, "notional": n, "arm": arm} for d, p, s, n in rows],
        columns=["trade_date", "PERMNO", "symbol", "side", "notional", "arm"])


# =====================================================================
# 1. 碰撞（协议 §4）
# =====================================================================

def test_no_collision_passes_through():
    """不同名字 / 不同日期 -> 两笔都原样通过，无让位。"""
    ft = _orders([("2026-01-05", 10001, "buy", 100.0)], ARM_FT)
    zs = _orders([("2026-01-05", 10002, "sell", 200.0)], ARM_ZS)
    ex, yl = resolve_collisions(ft, zs, seed=COLLISION_SEED)
    assert len(ex) == 2 and len(yl) == 0
    assert set(ex["collision"]) == {"none"}
    assert sorted(ex["arm"]) == [ARM_FT, ARM_ZS]
    ft_row = ex[ex.arm == ARM_FT].iloc[0]
    assert (ft_row.alloc_ft, ft_row.alloc_zs) == (1.0, 0.0)
    zs_row = ex[ex.arm == ARM_ZS].iloc[0]
    assert (zs_row.alloc_ft, zs_row.alloc_zs) == (0.0, 1.0)


def test_opposite_side_collision_yields_one_arm():
    """同日同名反向 -> 只提交一笔，另一臂进 yielded 表。"""
    day, permno = "2026-01-05", 10001
    ft = _orders([(day, permno, "buy", 100.0)], ARM_FT)
    zs = _orders([(day, permno, "sell", 150.0)], ARM_ZS)
    ex, yl = resolve_collisions(ft, zs, seed=COLLISION_SEED)

    assert len(ex) == 1 and len(yl) == 1
    winner = collision_winner(pd.Timestamp(day), permno, seed=COLLISION_SEED)
    assert winner in (ARM_FT, ARM_ZS)
    assert ex.iloc[0]["arm"] == winner
    assert ex.iloc[0]["collision"] == "won"
    assert yl.iloc[0]["arm"] == (ARM_ZS if winner == ARM_FT else ARM_FT)
    assert yl.iloc[0]["collision"] == "yielded"
    assert yl.iloc[0]["winner"] == winner
    # 获胜臂拿到自己的名义额与方向，没被对手改写
    expected_notional = 100.0 if winner == ARM_FT else 150.0
    assert ex.iloc[0]["notional"] == expected_notional


def test_collision_hash_matches_protocol_reference():
    """哈希实现与协议 §4.1 的参考代码逐字节一致。"""
    import hashlib

    for day, permno in [("2026-01-05", 10001), ("2024-07-01", 93436),
                        ("2020-12-31", 14593)]:
        msg = f"{COLLISION_SEED}|{day}|{int(permno)}".encode()
        ref = "FT" if (hashlib.sha256(msg).digest()[-1] & 1) == 0 else "ZS"
        assert collision_winner(pd.Timestamp(day), permno, seed=COLLISION_SEED) == ref


def test_collision_yield_is_symmetric_across_arms():
    """大样本下 FT / ZS 各约让位 50%（协议 §4.2：不得固定让同一臂）。"""
    days = pd.bdate_range("2020-01-01", "2023-12-31")   # 1044 个日期
    permnos = [10001, 10002, 10003, 10004]
    winners = [collision_winner(d, p, seed=COLLISION_SEED)
               for d in days for p in permnos]
    n = len(winners)
    share_ft = winners.count(ARM_FT) / n
    assert n > 4000
    # 二项 SE ≈ 0.5/sqrt(4176) ≈ 0.0077；±4 SE 宽度足够稳、又能抓住系统性偏斜
    assert abs(share_ft - 0.5) < 0.031, f"FT 获胜比例 {share_ft:.4f} 偏离 0.5 太远"


def test_collision_deterministic_across_processes():
    """同 seed 同输入，跨解释器进程逐行一致（协议 §4.1 禁用内置 hash()）。"""
    days = ["2026-01-05", "2026-01-06", "2026-01-07"]
    permnos = [10001, 10002, 10003, 14593, 93436]
    pairs = [(d, p) for d in days for p in permnos]
    local = [collision_winner(pd.Timestamp(d), p, seed=COLLISION_SEED)
             for d, p in pairs]

    code = textwrap.dedent(f"""
        import json, sys
        sys.path.insert(0, {SRC!r})
        import pandas as pd
        from execution.collision import collision_winner, COLLISION_SEED
        pairs = {pairs!r}
        print(json.dumps([collision_winner(pd.Timestamp(d), p, seed=COLLISION_SEED)
                          for d, p in pairs]))
    """)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout.strip().splitlines()[-1]) == local


def test_same_side_merge_and_pro_rata_allocation():
    """同日同名同向 -> 合并一笔、名义额相加、按名义额比例回分（协议 §4.3）。"""
    day, permno = "2026-01-05", 10001
    ft = _orders([(day, permno, "buy", 300.0)], ARM_FT)
    zs = _orders([(day, permno, "buy", 100.0)], ARM_ZS)
    ex, yl = resolve_collisions(ft, zs, seed=COLLISION_SEED)

    assert len(ex) == 1 and len(yl) == 0
    row = ex.iloc[0]
    assert row["arm"] == ARM_BOTH
    assert row["collision"] == "merged"
    assert row["side"] == "buy"
    assert row["notional"] == pytest.approx(400.0)
    assert row["alloc_ft"] == pytest.approx(0.75)
    assert row["alloc_zs"] == pytest.approx(0.25)
    assert row["alloc_ft"] + row["alloc_zs"] == pytest.approx(1.0)


def test_duplicate_same_arm_order_is_rejected():
    ft = _orders([("2026-01-05", 10001, "buy", 1.0),
                  ("2026-01-05", 10001, "sell", 1.0)], ARM_FT)
    with pytest.raises(ValueError, match="同日同名"):
        resolve_collisions(ft, _orders([], ARM_ZS), seed=COLLISION_SEED)


# =====================================================================
# 2. 影子账本（协议 §5）
# =====================================================================

def _synthetic_scores(days, permnos, seed=0):
    """合成分数：每日给每个 PERMNO 一个确定性的分。"""
    rng = np.random.default_rng(seed)
    return {d: {p: float(v) for p, v in zip(permnos, rng.normal(size=len(permnos)))}
            for d in days}


def _flat_adv(permnos):
    return {p: 1e6 + p for p in permnos}


def test_rebalance_touches_exactly_one_sleeve():
    """一次再平衡只动 day_index % NT 那个套袖。"""
    permnos = list(range(10001, 10081))          # n = 80 >= MIN_POOL
    days = list(pd.bdate_range("2026-01-05", periods=8))
    scores = _synthetic_scores(days, permnos, seed=1)
    led = ShadowLedger(ARM_FT, aum=3_000_000.0, nt=6)

    for i, day in enumerate(days[:6]):
        before = [None if s is None else list(s) for s in led.sleeves]
        led.target_from_scores(scores, day, i, _flat_adv(permnos))
        for j, (b, a) in enumerate(zip(before, led.sleeves)):
            if j == i % 6:
                assert a is not None and a != b
            else:
                assert a == b, f"套袖 {j} 在 day_index={i} 被动了"

    # 第 7 天（i=6）回到套袖 0，其余五个必须一字不动
    before = [list(s) for s in led.sleeves]
    led.target_from_scores(scores, days[6], 6, _flat_adv(permnos))
    for j in range(1, 6):
        assert led.sleeves[j] == before[j]


def test_target_size_and_unit_notional():
    """k = n//10、每槽位名义额 = AUM/(NT*k)。"""
    permnos = list(range(10001, 10081))          # n=80 -> k=8
    days = list(pd.bdate_range("2026-01-05", periods=1))
    scores = _synthetic_scores(days, permnos, seed=2)
    led = ShadowLedger(ARM_FT, aum=3_000_000.0, nt=6)
    tgt = led.target_from_scores(scores, days[0], 0, _flat_adv(permnos))

    assert len(led.sleeves[0]) == 8
    assert led.last_unit_notional == pytest.approx(3_000_000.0 / (6 * 8))
    assert len(tgt) == 8
    assert sum(tgt.values()) == pytest.approx(8 * 3_000_000.0 / 48)


def test_small_pool_returns_none_and_touches_nothing():
    permnos = list(range(10001, 10041))          # n = 40 < 50
    days = list(pd.bdate_range("2026-01-05", periods=1))
    scores = _synthetic_scores(days, permnos, seed=3)
    led = ShadowLedger(ARM_FT, aum=1e6, nt=6)
    assert led.target_from_scores(scores, days[0], 0, _flat_adv(permnos)) is None
    assert led.sleeves == [None] * 6


def test_topn_filter_uses_lagged_adv():
    """topn 按滞后 ADV20 降序截断；被截掉的名字不进目标。"""
    permnos = list(range(10001, 10101))          # 100 只
    days = list(pd.bdate_range("2026-01-05", periods=1))
    scores = _synthetic_scores(days, permnos, seed=4)
    adv = {p: float(p) for p in permnos}          # PERMNO 越大 ADV 越大
    led = ShadowLedger(ARM_FT, aum=1e6, nt=6)
    tgt = led.target_from_scores(scores, days[0], 0, adv, topn=60)
    kept = set(range(10041, 10101))               # ADV 最大的 60 只
    assert set(led.sleeves[0]) <= kept
    assert set(tgt) <= kept


def test_ideal_orders_are_target_minus_shadow_positions():
    led = ShadowLedger(ARM_FT, aum=1e6, nt=6)
    led.positions = {10001: 100.0, 10003: 50.0}
    orders = led.ideal_orders({10001: 300.0, 10002: 70.0},
                              trade_date="2026-01-05",
                              symbols={10001: "AAA", 10002: "BBB", 10003: "CCC"})
    got = {int(r.PERMNO): (r.side, r.notional) for r in orders.itertuples()}
    assert got == {10001: ("buy", 200.0), 10002: ("buy", 70.0),
                   10003: ("sell", 50.0)}
    assert set(orders["arm"]) == {ARM_FT}


def test_yielded_order_leaves_shadow_target_and_positions_untouched():
    """让位后：影子目标（sleeves）不变、影子持仓不变，次日从影子持仓重算。"""
    permnos = list(range(10001, 10081))
    days = list(pd.bdate_range("2026-01-05", periods=2))
    scores = _synthetic_scores(days, permnos, seed=5)
    led = ShadowLedger(ARM_ZS, aum=3_000_000.0, nt=6)
    tgt = led.target_from_scores(scores, days[0], 0, _flat_adv(permnos))
    sleeves_before = [None if s is None else list(s) for s in led.sleeves]
    positions_before = dict(led.positions)

    ideal = led.ideal_orders(tgt, trade_date=days[0])
    assert len(ideal) == 8
    # 该臂整日让位 = 一笔成交都没有
    led.apply_fills(pd.DataFrame(columns=["PERMNO", "side", "notional"]))

    assert led.sleeves == sleeves_before
    assert led.positions == positions_before
    # 次日重算的理想订单仍然从影子持仓（空）出发，缺口原样保留
    again = led.ideal_orders(tgt, trade_date=days[1])
    assert len(again) == len(ideal)
    assert again["notional"].sum() == pytest.approx(ideal["notional"].sum())


def test_apply_fills_updates_positions_and_drops_zeros():
    led = ShadowLedger(ARM_FT, aum=1e6, nt=6)
    led.apply_fills(pd.DataFrame([
        {"PERMNO": 10001, "side": "buy", "notional": 300.0},
        {"PERMNO": 10002, "side": "buy", "fill_price": 10.0, "fill_qty": 7,
         "notional": None},
    ]))
    assert led.positions == {10001: 300.0, 10002: 70.0}
    led.apply_fills(pd.DataFrame([{"PERMNO": 10001, "side": "sell",
                                   "notional": 300.0}]))
    assert 10001 not in led.positions          # 归零就从账本里消失


def test_reconcile_halts_on_mismatch():
    """协议 §5 第 6 步：两臂影子之和 vs 真实持仓，不平即 halt。"""
    ft = ShadowLedger(ARM_FT, aum=1e6, nt=6)
    zs = ShadowLedger(ARM_ZS, aum=1e6, nt=6)
    ft.positions = {10001: 100.0, 10002: 40.0}
    zs.positions = {10001: 60.0}

    ok = reconcile_arms([ft, zs], {10001: 160.0, 10002: 40.0}, tol_notional=1e-6)
    assert ok.halt is False and len(ok.diffs) == 0 and bool(ok) is True

    bad = reconcile_arms([ft, zs], {10001: 160.0, 10002: 39.0}, tol_notional=1e-6)
    assert bad.halt is True
    assert list(bad.diffs["PERMNO"]) == [10002]
    assert bad.diffs.iloc[0]["diff"] == pytest.approx(1.0)
    assert bad.max_abs_diff == pytest.approx(1.0)

    # 真实账户里有影子账本完全不知道的股票，也必须 halt
    ghost = reconcile_arms([ft, zs], {10001: 160.0, 10002: 40.0, 99999: 5.0},
                           tol_notional=1e-6)
    assert ghost.halt is True and 99999 in set(ghost.diffs["PERMNO"])


# =====================================================================
# 3. 费用（HANDOFF §11.4a）
# =====================================================================

def test_sell_100_shares_at_50():
    f = fees.order_fees("sell", 100, 5000.0, plan="all_in")
    assert f["sec"] == pytest.approx(0.103)          # 0.0000206 * 5000
    assert f["taf"] == pytest.approx(0.0195)         # 0.000195 * 100
    assert f["cat"] == pytest.approx(0.0003)         # 0.000003 * 100
    assert f["commission"] == pytest.approx(0.40)    # 0.0040 * 100
    assert f["total"] == pytest.approx(0.103 + 0.0195 + 0.0003 + 0.40)


def test_buy_100_shares_at_50_has_no_sell_side_regulatory_fees():
    f = fees.order_fees("buy", 100, 5000.0, plan="all_in")
    assert f["sec"] == 0.0
    assert f["taf"] == 0.0
    assert f["cat"] == pytest.approx(0.0003)
    assert f["commission"] == pytest.approx(0.40)
    assert f["total"] == pytest.approx(0.4003)


def test_cost_plus_commission():
    f = fees.order_fees("buy", 100, 5000.0, plan="cost_plus")
    assert f["commission"] == pytest.approx(0.25)    # 0.0025 * 100


def test_taf_cap():
    """TAF 上限 $9.79。9.79/0.000195 = 50205.13，故上限从 50,206 股起真正咬人。

    HANDOFF §11.4a 写的「50,205 股封顶」只有在把 TAF 四舍五入到分之后才成立
    （50,205 股的未舍入值 = 9.789975，round(.,2) = 9.79）。本模块不做分位舍入。
    """
    assert fees.finra_taf(50_205) == pytest.approx(9.789975)
    assert round(fees.finra_taf(50_205), 2) == pytest.approx(9.79)
    assert fees.finra_taf(50_206) == pytest.approx(9.79)      # 已封顶
    assert fees.finra_taf(1_000_000) == pytest.approx(9.79)
    assert fees.FINRA_TAF_CAP_SHARES == 50_206


def test_bad_side_rejected():
    with pytest.raises(ValueError):
        fees.order_fees("BUY_TO_OPEN", 100, 5000.0)
    with pytest.raises(ValueError):
        fees.commission_per_share("free")


def test_cost_bp_buy():
    """买 fill 50.05 / open 50.00 -> +10bp 执行偏差 + 费用。"""
    r = fees.cost_bp("buy", 50.05, 50.00, 100, 5005.0, plan="all_in")
    assert r["exec_bp"] == pytest.approx(10.0)
    expected_fee_bp = 0.4003 / 5005.0 * 1e4
    assert r["fees_bp"] == pytest.approx(expected_fee_bp)
    assert r["cost_bp"] == pytest.approx(10.0 + expected_fee_bp)
    assert np.isnan(r["drift_bp"])               # 未给 prev_close


def test_cost_bp_sell():
    """卖 fill 49.95 / open 50.00 -> +10bp 执行偏差（卖便宜也是成本）+ 费用。"""
    r = fees.cost_bp("sell", 49.95, 50.00, 100, 4995.0, plan="all_in")
    assert r["exec_bp"] == pytest.approx(10.0)
    f = fees.order_fees("sell", 100, 4995.0, plan="all_in")
    assert r["fees_bp"] == pytest.approx(f["total"] / 4995.0 * 1e4)
    assert r["cost_bp"] == pytest.approx(10.0 + r["fees_bp"])
    assert r["fee_usd_total"] == pytest.approx(f["total"])


def test_drift_is_diagnostic_only_and_never_enters_C():
    """决策收盘 -> 开盘漂移只作诊断，不进 C（协议 §0 / HANDOFF §11.2 第 6 条）。"""
    r = fees.cost_bp("buy", 50.05, 50.00, 100, 5005.0, prev_close=49.00)
    assert r["drift_bp"] == pytest.approx((50.00 - 49.00) / 49.00 * 1e4)
    assert r["cost_bp"] == pytest.approx(r["exec_bp"] + r["fees_bp"])
    # 无漂移的同一笔，C 必须完全一样
    r0 = fees.cost_bp("buy", 50.05, 50.00, 100, 5005.0, prev_close=50.00)
    assert r0["cost_bp"] == pytest.approx(r["cost_bp"])


def test_pure_moo_zero_deviation_is_fee_only():
    """fill == DlyOpen 时 C 恰好等于费用段。"""
    r = fees.cost_bp("buy", 50.0, 50.0, 100, 5000.0)
    assert r["exec_bp"] == pytest.approx(0.0)
    assert r["cost_bp"] == pytest.approx(r["fees_bp"])


# =====================================================================
# 4. 提交窗口（协议 §2）
# =====================================================================

@pytest.mark.parametrize("hh,mm,ss", [(19, 30, 0), (19, 0, 0), (23, 59, 59),
                                      (0, 0, 0), (8, 59, 0), (8, 59, 59)])
def test_submit_window_allows(hh, mm, ss):
    ts = datetime(2026, 1, 6, hh, mm, ss, tzinfo=ET)
    assert assert_submit_window(ts) == ts


@pytest.mark.parametrize("hh,mm,ss", [(9, 0, 0), (9, 28, 1), (9, 30, 0),
                                      (12, 0, 0), (18, 59, 59)])
def test_submit_window_rejects(hh, mm, ss):
    ts = datetime(2026, 1, 6, hh, mm, ss, tzinfo=ET)
    with pytest.raises(SubmitWindowError):
        assert_submit_window(ts)


def test_submit_window_converts_other_timezones():
    """00:30 UTC = 19:30 ET（冬令时）-> 允许。"""
    ts = pd.Timestamp("2026-01-07 00:30:00", tz="UTC")
    assert assert_submit_window(ts).hour == 19


def test_client_rejects_submission_outside_window():
    client = AlpacaOpgClient(dry_run=True, paper=True)
    orders = pd.DataFrame([{"arm": ARM_FT, "trade_date": pd.Timestamp("2026-01-06"),
                            "PERMNO": 10001, "symbol": "AAA", "side": "buy",
                            "qty": 10, "notional": 500.0, "collision": "none",
                            "alloc_ft": 1.0, "alloc_zs": 0.0}])
    with pytest.raises(SubmitWindowError):
        client.submit_opg_orders(orders, now_et=datetime(2026, 1, 6, 10, 0, tzinfo=ET))


def test_dry_run_client_is_offline():
    client = AlpacaOpgClient(dry_run=True)
    orders = pd.DataFrame([{"arm": ARM_FT, "trade_date": pd.Timestamp("2026-01-06"),
                            "PERMNO": 10001, "symbol": "AAA", "side": "buy",
                            "qty": 10, "notional": 500.0, "collision": "none",
                            "alloc_ft": 1.0, "alloc_zs": 0.0}])
    out = client.submit_opg_orders(orders,
                                   now_et=datetime(2026, 1, 5, 19, 30, tzinfo=ET))
    assert len(out) == 1
    assert out.iloc[0]["status"] == "accepted(dry-run)"
    assert out.iloc[0]["tif"] == "opg"
    assert str(out.iloc[0]["order_id"]).startswith("dry-")
    assert client.fetch_fills(pd.Timestamp("2026-01-06")).empty
    assert client.fetch_positions() == {}


def test_live_path_raises_without_sdk():
    client = AlpacaOpgClient(dry_run=False)
    orders = pd.DataFrame([{"arm": ARM_FT, "trade_date": pd.Timestamp("2026-01-06"),
                            "PERMNO": 10001, "symbol": "AAA", "side": "buy",
                            "qty": 10, "notional": 500.0, "collision": "none",
                            "alloc_ft": 1.0, "alloc_zs": 0.0}])
    with pytest.raises(RuntimeError):
        client.submit_opg_orders(orders,
                                 now_et=datetime(2026, 1, 5, 19, 30, tzinfo=ET))


# =====================================================================
# 5. 订单流水
# =====================================================================

def test_append_orders_does_not_overwrite(tmp_path):
    path = tmp_path / "orders.parquet"
    assert load_orders(path).empty
    assert list(load_orders(path).columns) == ORDER_COLUMNS

    first = pd.DataFrame([{"order_id": "a", "arm": ARM_FT,
                           "trade_date": pd.Timestamp("2026-01-06"),
                           "PERMNO": 10001, "side": "buy", "qty": 10,
                           "notional_target": 500.0, "tif": "opg",
                           "status": "accepted(dry-run)", "collision": "none"}])
    append_orders(path, first)
    second = first.copy()
    second["order_id"] = "b"
    append_orders(path, second)

    got = load_orders(path)
    assert len(got) == 2
    assert list(got["order_id"]) == ["a", "b"]
    assert list(got.columns) == ORDER_COLUMNS
    assert got["crsp_dly_open"].isna().all()      # 待回填
    assert got["prev_close"].isna().all()

    with pytest.raises(ValueError, match="order_id 重复"):
        append_orders(path, second)


# =====================================================================
# 6. 每日流程（协议 §5），全 dry-run
# =====================================================================

def _daily_fixture():
    permnos = list(range(10001, 10081))
    days = list(pd.bdate_range("2026-01-05", periods=3))
    ledgers = {ARM_FT: ShadowLedger(ARM_FT, aum=3_000_000.0, nt=6),
               ARM_ZS: ShadowLedger(ARM_ZS, aum=3_000_000.0, nt=6)}
    return (permnos, days, ledgers,
            _synthetic_scores(days, permnos, seed=11),
            _synthetic_scores(days, permnos, seed=22))


def test_run_daily_dry_run_writes_orders(tmp_path):
    permnos, days, ledgers, s_ft, s_zs = _daily_fixture()
    client = AlpacaOpgClient(dry_run=True, paper=True)
    path = tmp_path / "orders.parquet"
    adv = _flat_adv(permnos)
    prices = {p: 50.0 for p in permnos}
    symbols = {p: f"S{p}" for p in permnos}
    now = datetime(2026, 1, 4, 19, 30, tzinfo=ET)

    results = []
    for i, day in enumerate(days):
        results.append(run_daily(day, i, s_ft, s_zs, adv, ledgers, client,
                                 seed=COLLISION_SEED, orders_path=path,
                                 prices=prices, symbols=symbols, now_et=now))

    for res in results:
        assert res.reconcile is not None and res.reconcile.halt is False
        assert len(res.ideal[ARM_FT]) > 0 and len(res.ideal[ARM_ZS]) > 0
        # 可执行 + 让位 == 两臂理想订单之和 − 合并省下的那一笔
        n_ideal = len(res.ideal[ARM_FT]) + len(res.ideal[ARM_ZS])
        n_merged = int((res.exec_orders["collision"] == "merged").sum())
        assert len(res.exec_orders) + len(res.yielded) + n_merged == n_ideal

    got = load_orders(path)
    assert len(got) == sum(len(r.orders_frame) for r in results)
    assert list(got.columns) == ORDER_COLUMNS
    assert set(got["tif"].dropna()) == {"opg"}
    assert set(got["status"].dropna()) <= {"accepted(dry-run)", "yielded"}
    assert set(got["arm"].dropna()) <= {ARM_FT, ARM_ZS, ARM_BOTH}
    assert got["order_id"].is_unique
    assert (got["qty"].dropna() > 0).all()
    # dry-run 没有成交 -> 两臂影子持仓仍为空，核对自然对平
    assert ledgers[ARM_FT].positions == {} and ledgers[ARM_ZS].positions == {}


class _FillingClient(AlpacaOpgClient):
    """dry-run 客户端 + 注入的全额成交与真实持仓（仍然不联网、不读任何文件）。"""

    def __init__(self, real=None):
        super().__init__(dry_run=True, paper=True)
        self._last = pd.DataFrame()
        self.real: dict[int, float] = dict(real or {})

    def submit_opg_orders(self, orders, *, now_et=None, enforce_window=True):
        out = super().submit_opg_orders(orders, now_et=now_et,
                                        enforce_window=enforce_window)
        self._last = out
        return out

    def fetch_fills(self, trade_date):
        rows = []
        for r in self._last.to_dict("records"):
            notional = float(r["notional_target"])
            rows.append({"order_id": r["order_id"], "PERMNO": r["PERMNO"],
                         "symbol": r["symbol"], "side": r["side"],
                         "fill_price": 50.0, "fill_qty": notional / 50.0,
                         "fill_ts": pd.Timestamp(trade_date), "status": "filled"})
            p = int(r["PERMNO"])
            self.real[p] = self.real.get(p, 0.0) + (notional if r["side"] == "buy"
                                                    else -notional)
            if abs(self.real[p]) <= 1e-9:
                self.real.pop(p)
        return pd.DataFrame(rows)

    def fetch_positions(self):
        return dict(self.real)


def test_run_daily_allocates_merged_fill_pro_rata(tmp_path):
    """同向合并单的成交按名义额比例回分到两臂影子账本（协议 §4.3 + §5 第 5 步）。

    两臂给**同一份分数**、但 AUM 3:1 -> 当日 8 个名字全部同向合并，
    每笔 alloc_ft 恒为 0.75。
    """
    permnos = list(range(10001, 10081))          # n=80 -> k=8
    day = pd.Timestamp("2026-01-05")
    scores = {day: {p: float(p) for p in permnos}}
    ledgers = {ARM_FT: ShadowLedger(ARM_FT, aum=3_000_000.0, nt=6),
               ARM_ZS: ShadowLedger(ARM_ZS, aum=1_000_000.0, nt=6)}
    client = _FillingClient()

    res = run_daily(day, 0, scores, scores, _flat_adv(permnos), ledgers, client,
                    seed=COLLISION_SEED, orders_path=tmp_path / "orders.parquet",
                    prices={p: 50.0 for p in permnos}, now_et=datetime(
                        2026, 1, 4, 19, 30, tzinfo=ET), tol_notional=1e-6)

    assert res.reconcile.halt is False
    assert len(res.yielded) == 0
    assert len(res.exec_orders) == 8
    assert set(res.exec_orders["collision"]) == {"merged"}
    assert set(res.exec_orders["arm"]) == {ARM_BOTH}

    unit_ft, unit_zs = 3_000_000.0 / 48, 1_000_000.0 / 48
    for row in res.exec_orders.to_dict("records"):
        p = int(row["PERMNO"])
        assert row["notional"] == pytest.approx(unit_ft + unit_zs)
        assert row["alloc_ft"] == pytest.approx(0.75)
        assert row["alloc_zs"] == pytest.approx(0.25)
        assert ledgers[ARM_FT].positions[p] == pytest.approx(unit_ft)
        assert ledgers[ARM_ZS].positions[p] == pytest.approx(unit_zs)


def test_run_daily_yield_leaves_losing_arm_position_unchanged(tmp_path):
    """反向碰撞下落败臂当日站开：不提交、影子持仓不变，但核对仍然对平。

    构造：FT 分数 = +PERMNO（买 top-8 = 10073..10080）；ZS 分数 = −PERMNO
    （买 10001..10008），且 ZS 预先持有 10080 -> ZS 要卖 10080、FT 要买 10080。
    """
    permnos = list(range(10001, 10081))
    day = pd.Timestamp("2026-01-05")
    s_ft = {day: {p: float(p) for p in permnos}}
    s_zs = {day: {p: -float(p) for p in permnos}}

    ledgers = {ARM_FT: ShadowLedger(ARM_FT, aum=3_000_000.0, nt=6),
               ARM_ZS: ShadowLedger(ARM_ZS, aum=3_000_000.0, nt=6)}
    ledgers[ARM_ZS].positions = {10080: 5_000.0}
    client = _FillingClient(real={10080: 5_000.0})

    res = run_daily(day, 0, s_ft, s_zs, _flat_adv(permnos), ledgers, client,
                    seed=COLLISION_SEED, orders_path=tmp_path / "orders.parquet",
                    prices={p: 50.0 for p in permnos}, now_et=datetime(
                        2026, 1, 4, 19, 30, tzinfo=ET), tol_notional=1e-6)

    assert res.reconcile.halt is False              # 让位之后账本仍然对平
    assert len(res.yielded) == 1
    yl = res.yielded.iloc[0]
    assert int(yl["PERMNO"]) == 10080
    winner = collision_winner(day, 10080, seed=COLLISION_SEED)
    assert yl["winner"] == winner
    assert yl["arm"] == (ARM_ZS if winner == ARM_FT else ARM_FT)

    unit = 3_000_000.0 / 48
    if winner == ARM_FT:
        # ZS 的卖单没提交 -> ZS 在 10080 上的影子持仓原封不动
        assert ledgers[ARM_ZS].positions[10080] == pytest.approx(5_000.0)
        assert ledgers[ARM_FT].positions[10080] == pytest.approx(unit)
    else:
        # FT 的买单没提交 -> FT 在 10080 上仍然是零
        assert 10080 not in ledgers[ARM_FT].positions
        assert 10080 not in ledgers[ARM_ZS].positions       # 卖光了

    # 让位单进了流水，状态是 yielded、没有成交价
    got = load_orders(tmp_path / "orders.parquet")
    row = got[got["status"] == "yielded"]
    assert len(row) == 1
    assert int(row.iloc[0]["PERMNO"]) == 10080
    assert row.iloc[0]["collision"] == "yielded"
    assert pd.isna(row.iloc[0]["fill_price"])


def test_run_daily_halts_when_books_do_not_reconcile(tmp_path):
    permnos, days, ledgers, s_ft, s_zs = _daily_fixture()
    path = tmp_path / "orders.parquet"
    now = datetime(2026, 1, 4, 19, 30, tzinfo=ET)

    class _GhostPositionClient(AlpacaOpgClient):
        def fetch_positions(self):
            return {99999: 12345.0}          # 影子账本里根本没有的股票

    with pytest.raises(HaltError) as exc:
        run_daily(days[0], 0, s_ft, s_zs, _flat_adv(permnos), ledgers,
                  _GhostPositionClient(dry_run=True), seed=COLLISION_SEED,
                  orders_path=path, now_et=now, tol_notional=1e-6)

    assert exc.value.result.halt is True
    assert 99999 in set(exc.value.result.diffs["PERMNO"])
    # 停机之前当日流水已经落盘
    assert len(load_orders(path)) > 0
