"""协议 §4：逐碰撞确定性哈希让位 + 同向同名合并。

背景（HANDOFF §11.4b / K14）：Alpaca 的市价单**总是**被同标的反向的市价/限价/
止损/止损限价单挡下（HTTP 403，paper 账户同样执行）。2,270 笔反向碰撞落在
729/832 = 87.6% 的交易日上（均值 2.7 笔/日、p90 5、max 15），**不是边缘情况**。

规则（预注册，第一笔成交前写死，此后不得更改）：

1. **反向碰撞** —— 对 ``f"{SEED}|{YYYY-MM-DD}|{PERMNO}"`` 取 SHA-256，
   最后一字节最低位 == 0 -> FT 获胜，== 1 -> ZS 获胜。
   获胜臂照常提交；落败臂当日在该标的上**站开、不提交**（单边无反向单，403 不触发）。
2. **同向同名** —— 合并成一笔提交、成交后按名义额比例分配到两臂，两臂记同一成交价。
3. 其余原样通过。

`SEED` 与哈希实现取自协议 §4.1 的代码块（`scripts/k14_collision_and_sizing.py`
只做方案比价与定档，**并未实现让位函数**）。

**禁用 Python 内置 `hash()`**（进程间随机化，协议 §4.1 明令禁止）。
"""
from __future__ import annotations

import hashlib

import pandas as pd

# 协议 §4.1：第一笔成交前写死，此后不得更改。
COLLISION_SEED = "k14-collision-yield-v1"

ARM_FT = "FT"
ARM_ZS = "ZS"
ARM_BOTH = "BOTH"

COLLISION_NONE = "none"
COLLISION_WON = "won"
COLLISION_YIELDED = "yielded"
COLLISION_MERGED = "merged"

#: `resolve_collisions` 的输入表必须至少含这些列。
ORDER_INPUT_COLUMNS = ["trade_date", "PERMNO", "symbol", "side", "notional", "arm"]
#: 可执行订单表的列。
EXEC_COLUMNS = ORDER_INPUT_COLUMNS + ["collision", "alloc_ft", "alloc_zs"]
#: 让位流水表的列（协议 §4.2 要求最终报告给出让位订单的计数与画像）。
YIELDED_COLUMNS = ORDER_INPUT_COLUMNS + ["collision", "winner", "winner_side"]


def collision_message(trade_date, permno, *, seed: str = COLLISION_SEED) -> bytes:
    """协议 §4.1 的待哈希字节串：``f"{SEED}|{date.isoformat()}|{int(PERMNO)}"``。"""
    day = pd.Timestamp(trade_date).date().isoformat()
    return f"{seed}|{day}|{int(permno)}".encode("utf-8")


def collision_winner(trade_date, permno, *, seed: str = COLLISION_SEED) -> str:
    """返回该 ``(trade_date, PERMNO)`` 碰撞的获胜臂：``"FT"`` 或 ``"ZS"``。

    与协议 §4.1 逐字一致：``digest()[-1] & 1``，0 -> FT，1 -> ZS。
    跨进程确定性（SHA-256，不用内置 `hash()`）。
    """
    return ARM_FT if (hashlib.sha256(
        collision_message(trade_date, permno, seed=seed)).digest()[-1] & 1) == 0 else ARM_ZS


def _normalise(orders, arm: str) -> dict[tuple, dict]:
    """把一臂的订单表转成 {(trade_date, PERMNO): row}，并校验列与唯一性。"""
    if orders is None:
        return {}
    missing = [c for c in ORDER_INPUT_COLUMNS if c not in orders.columns and c != "arm"]
    if missing:
        raise ValueError(f"{arm} 订单表缺列：{missing}")
    out: dict[tuple, dict] = {}
    for row in orders.to_dict("records"):
        day = pd.Timestamp(row["trade_date"])
        permno = int(row["PERMNO"])
        key = (day, permno)
        if key in out:
            raise ValueError(
                f"{arm} 同日同名出现多笔订单：{key}。冻结构造下每臂每名每日只有一笔，"
                "请先在上游合并。")
        side = str(row["side"]).strip().lower()
        if side not in ("buy", "sell"):
            raise ValueError(f"side 只能是 'buy'/'sell'，收到 {row['side']!r}")
        out[key] = {
            "trade_date": day,
            "PERMNO": permno,
            "symbol": row.get("symbol"),
            "side": side,
            "notional": float(row["notional"]),
            "arm": arm,
        }
    return out


def _alloc(arm: str) -> tuple[float, float]:
    return (1.0, 0.0) if arm == ARM_FT else (0.0, 1.0)


def resolve_collisions(orders_ft: pd.DataFrame, orders_zs: pd.DataFrame, *,
                       seed: str = COLLISION_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """协议 §4 的碰撞解析。

    输入两臂的「理想订单」表，列至少含
    ``trade_date, PERMNO, symbol, side ('buy'|'sell'), notional, arm``
    （`arm` 列的内容被忽略，以参数位置为准）。

    返回 ``(exec_orders, yielded_log)``：

    - ``exec_orders`` —— 可执行订单集合，列见 :data:`EXEC_COLUMNS`。
      `arm` 为 ``"FT"``/``"ZS"``/``"BOTH"``（合并单）；
      ``alloc_ft + alloc_zs == 1.0``，是**该笔成交回分到各臂的比例**
      （非合并单为 1/0 或 0/1；合并单按名义额比例）。
    - ``yielded_log`` —— 落败臂当日站开的订单，列见 :data:`YIELDED_COLUMNS`。

    确定性：结果按 ``(trade_date, PERMNO)`` 排序，哈希用 SHA-256，
    **同 seed 同输入跨进程逐行一致**。
    """
    ft = _normalise(orders_ft, ARM_FT)
    zs = _normalise(orders_zs, ARM_ZS)

    exec_rows: list[dict] = []
    yielded_rows: list[dict] = []

    for key in sorted(set(ft) | set(zs)):
        f, z = ft.get(key), zs.get(key)

        if f is not None and z is not None and f["side"] != z["side"]:
            # 反向碰撞 -> 哈希让位（协议 §4.1）
            winner = collision_winner(key[0], key[1], seed=seed)
            win, lose = (f, z) if winner == ARM_FT else (z, f)
            a_ft, a_zs = _alloc(winner)
            exec_rows.append({**win, "collision": COLLISION_WON,
                              "alloc_ft": a_ft, "alloc_zs": a_zs})
            yielded_rows.append({**lose, "collision": COLLISION_YIELDED,
                                 "winner": winner, "winner_side": win["side"]})
            continue

        if f is not None and z is not None:
            # 同向同名 -> 合并提交、按名义额比例回分（协议 §4.3）
            total = f["notional"] + z["notional"]
            if total > 0:
                a_ft = f["notional"] / total
            else:  # 退化情况：两笔名义额都是 0，比例无定义，对半记账
                a_ft = 0.5
            exec_rows.append({
                "trade_date": key[0], "PERMNO": key[1],
                "symbol": f["symbol"] if f["symbol"] is not None else z["symbol"],
                "side": f["side"], "notional": total, "arm": ARM_BOTH,
                "collision": COLLISION_MERGED,
                "alloc_ft": a_ft, "alloc_zs": 1.0 - a_ft,
            })
            continue

        # 无碰撞 -> 原样通过
        only = f if f is not None else z
        a_ft, a_zs = _alloc(only["arm"])
        exec_rows.append({**only, "collision": COLLISION_NONE,
                          "alloc_ft": a_ft, "alloc_zs": a_zs})

    exec_orders = pd.DataFrame(exec_rows, columns=EXEC_COLUMNS)
    yielded_log = pd.DataFrame(yielded_rows, columns=YIELDED_COLUMNS)
    return exec_orders, yielded_log
