"""协议 §5 的每日流程（不得简化）。

::

    1. 两臂各自的影子账本 + 当日分数  ->  各自的再平衡目标
    2. 目标 − 影子持仓                ->  两臂的「理想订单」
    3. 反向碰撞检测 -> §4.1 哈希让位；同向同名 -> §4.3 合并
    4.                                ->  「可执行订单」集合，提交
    5. 成交回报 -> 按 §4.3 比例规则回分至各自影子账本
    6. sum(两臂影子持仓) == 真实账户持仓  —— 逐股核对

**第 6 步不符即当日停机**（:class:`HaltError`），查清后方可继续；
停机日不计入协议 §6 的观察日数。

落盘约定：订单流水是 append-only 且 `order_id` 唯一，所以**每日只 append 一次**，
一行携带该日全部六步的结果（碰撞标记 / 提交状态 / 成交）。append 发生在第 6 步
**之前**，保证停机时当日流水已经在盘上。各步的中间表在 :class:`DailyResult` 里返回。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

from execution.collision import ARM_FT, ARM_ZS, COLLISION_SEED, resolve_collisions
from execution.orders import (ORDER_COLUMNS, STATUS_YIELDED, TIF_OPG,
                              append_orders, coerce_orders)
from execution.shadow_ledger import ReconcileResult, reconcile_arms


class HaltError(RuntimeError):
    """协议 §5 第 6 步逐股核对不平 —— 当日停机。"""

    def __init__(self, message: str, result: ReconcileResult):
        super().__init__(message)
        self.result = result


@dataclass
class DailyResult:
    """一个交易日六步流程的全部中间产物。"""

    day: pd.Timestamp
    day_index: int
    targets: dict = field(default_factory=dict)
    ideal: dict = field(default_factory=dict)
    exec_orders: pd.DataFrame = field(default_factory=pd.DataFrame)
    yielded: pd.DataFrame = field(default_factory=pd.DataFrame)
    submitted: pd.DataFrame = field(default_factory=pd.DataFrame)
    fills: pd.DataFrame = field(default_factory=pd.DataFrame)
    orders_frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    reconcile: ReconcileResult | None = None


def _yield_order_id(row: dict) -> str:
    day = pd.Timestamp(row["trade_date"]).date().isoformat()
    msg = f"yield|{row['arm']}|{day}|{int(row['PERMNO'])}|{row['side']}".encode("utf-8")
    return "yld-" + hashlib.sha256(msg).hexdigest()[:16]


def _qty_from_notional(notional: float, price: float | None) -> int | None:
    """名义额 -> **整股**股数（协议 §2：零股/小数股不参与竞价，禁止使用）。

    协议只冻结了「整股」，没有冻结取整方式；这里用**就近取整**，
    并保留 `notional_target` 原值以便事后核对。真钱之前需要在 ledger 里定死。
    """
    if price is None or not np.isfinite(float(price)) or float(price) <= 0:
        return None
    return int(round(float(notional) / float(price)))


def run_daily(day, day_index: int, scores_ft, scores_zs, adv_today,
              ledgers: Mapping[str, "object"], client, seed: str = COLLISION_SEED,
              orders_path=None, *, prices: dict | None = None,
              symbols: dict | None = None, now_et=None,
              tol_notional: float = 1e-6, topn: int = 500,
              exit_pct: float = 0.30) -> DailyResult:
    """跑通协议 §5 的一天。全 dry-run 可跑（`client.dry_run=True` 时不联网）。

    Parameters
    ----------
    day, day_index
        交易日与它在该臂日序列里的序号（``day_index % NT`` 决定动哪个套袖）。
    scores_ft, scores_zs
        ``{day: {PERMNO: score}}``，t 日收盘算出的分数。
    adv_today
        ``{PERMNO: 滞后 ADV20}``，严格 ``shift(1)``。
    ledgers
        ``{"FT": ShadowLedger, "ZS": ShadowLedger}``。
    client
        :class:`execution.alpaca_client.AlpacaOpgClient`。
    seed
        碰撞哈希 seed，默认 :data:`execution.collision.COLLISION_SEED`。
    orders_path
        订单流水 parquet；None 则不落盘。
    prices
        ``{PERMNO: 预估价}``，用来把名义额换成整股股数；缺则 `qty` 留空。

    Raises
    ------
    HaltError
        第 6 步逐股核对不平。当日流水已经落盘。
    """
    day = pd.Timestamp(day)
    ledger_ft, ledger_zs = ledgers[ARM_FT], ledgers[ARM_ZS]
    prices = prices or {}
    symbols = symbols or {}
    res = DailyResult(day=day, day_index=int(day_index))

    # --- 1. 两臂影子目标 -------------------------------------------------
    tgt_ft = ledger_ft.target_from_scores(scores_ft, day, day_index, adv_today,
                                          topn=topn, exit_pct=exit_pct,
                                          nt=ledger_ft.nt)
    tgt_zs = ledger_zs.target_from_scores(scores_zs, day, day_index, adv_today,
                                          topn=topn, exit_pct=exit_pct,
                                          nt=ledger_zs.nt)
    res.targets = {ARM_FT: tgt_ft, ARM_ZS: tgt_zs}

    # --- 2. 理想订单 ------------------------------------------------------
    ideal_ft = ledger_ft.ideal_orders(tgt_ft or {}, trade_date=day, symbols=symbols)
    ideal_zs = ledger_zs.ideal_orders(tgt_zs or {}, trade_date=day, symbols=symbols)
    res.ideal = {ARM_FT: ideal_ft, ARM_ZS: ideal_zs}

    # --- 3. 碰撞哈希让位 / 同向合并 --------------------------------------
    exec_orders, yielded = resolve_collisions(ideal_ft, ideal_zs, seed=seed)
    res.exec_orders, res.yielded = exec_orders, yielded

    # --- 4. 合成可执行订单并提交 -----------------------------------------
    if len(exec_orders):
        exec_orders = exec_orders.copy()
        exec_orders["qty"] = [
            _qty_from_notional(n, prices.get(int(p)))
            for n, p in zip(exec_orders["notional"], exec_orders["PERMNO"])]
    submitted = client.submit_opg_orders(exec_orders, now_et=now_et)
    res.submitted = submitted

    # --- 5. 成交回报 -> 按比例回分至各自影子账本 --------------------------
    fills = client.fetch_fills(day)
    res.fills = fills
    per_arm: dict[str, list[dict]] = {ARM_FT: [], ARM_ZS: []}
    if fills is not None and len(fills) and len(submitted):
        alloc = submitted[["order_id", "alloc_ft", "alloc_zs"]]
        for row in fills.merge(alloc, on="order_id", how="left").to_dict("records"):
            notional = float(row["fill_price"]) * float(row["fill_qty"])
            for arm, key in ((ARM_FT, "alloc_ft"), (ARM_ZS, "alloc_zs")):
                frac = row.get(key)
                if frac is None or not np.isfinite(float(frac)) or float(frac) == 0.0:
                    continue
                per_arm[arm].append({
                    "PERMNO": int(row["PERMNO"]),
                    "side": str(row["side"]).strip().lower(),
                    "notional": notional * float(frac),
                })
    ledger_ft.apply_fills(pd.DataFrame(per_arm[ARM_FT]))
    ledger_zs.apply_fills(pd.DataFrame(per_arm[ARM_ZS]))

    # --- 落盘（在核对之前，保证停机时流水已在盘上）-----------------------
    frame = _build_orders_frame(day, exec_orders, submitted, yielded, fills)
    res.orders_frame = frame
    if orders_path is not None and len(frame):
        append_orders(orders_path, frame)

    # --- 6. 逐股核对 ------------------------------------------------------
    real = client.fetch_positions()
    rec = reconcile_arms([ledger_ft, ledger_zs], real, tol_notional)
    res.reconcile = rec
    if rec.halt:
        raise HaltError(
            f"{day.date()} 协议 §5 第 6 步核对不平：{len(rec.diffs)} 只股票有差异，"
            f"max|diff|={rec.max_abs_diff:.6g} > tol={tol_notional:g} —— 当日停机，"
            "停机日不计入 §6 的观察日数。", rec)
    return res


def _build_orders_frame(day, exec_orders, submitted, yielded, fills) -> pd.DataFrame:
    """把可执行订单 + 提交状态 + 成交 + 让位单拼成当日的订单流水。"""
    rows: list[dict] = []

    if len(submitted):
        fill_map = {}
        if fills is not None and len(fills):
            fill_map = {r["order_id"]: r for r in fills.to_dict("records")}
        for row in submitted.to_dict("records"):
            f = fill_map.get(row["order_id"], {})
            rows.append({
                "order_id": row["order_id"], "arm": row["arm"],
                "trade_date": day, "submit_ts": row["submit_ts"],
                "PERMNO": row["PERMNO"], "symbol": row["symbol"], "side": row["side"],
                "qty": row["qty"], "notional_target": row["notional_target"],
                "tif": TIF_OPG, "status": f.get("status", row["status"]),
                "fill_price": f.get("fill_price"), "fill_qty": f.get("fill_qty"),
                "fill_ts": f.get("fill_ts"),
                "crsp_dly_open": None, "prev_close": None,   # 待回填
                "collision": row["collision"],
                "alloc_ft": row["alloc_ft"], "alloc_zs": row["alloc_zs"],
            })

    for row in yielded.to_dict("records") if len(yielded) else []:
        rows.append({
            "order_id": _yield_order_id(row), "arm": row["arm"],
            "trade_date": day, "submit_ts": None,
            "PERMNO": row["PERMNO"], "symbol": row["symbol"], "side": row["side"],
            "qty": None, "notional_target": row["notional"],
            "tif": TIF_OPG, "status": STATUS_YIELDED,
            "fill_price": None, "fill_qty": None, "fill_ts": None,
            "crsp_dly_open": None, "prev_close": None,
            "collision": row["collision"],
            "alloc_ft": 1.0 if row["arm"] == ARM_FT else 0.0,
            "alloc_zs": 1.0 if row["arm"] == ARM_ZS else 0.0,
        })

    return coerce_orders(pd.DataFrame(rows, columns=ORDER_COLUMNS))
