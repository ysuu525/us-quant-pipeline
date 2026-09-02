"""订单 / 成交流水的 schema 与追加落盘。

列名是**固定常量**（:data:`ORDER_COLUMNS`），协议 §8 要求最终报告能给出逐笔明细、
逐日序列、以及「实际费用 vs (fill − DlyOpen)」的分解，全部从这张表出。

`crsp_dly_open` 与 `prev_close` **提交时留空、事后回填**（CRSP 日行数据 T+N 才到）；
在回填之前 :mod:`execution.fees` 的 `cost_bp` 不得计算。
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

TIF_OPG = "opg"  # 协议 §2：type=market, time_in_force=opg, 整股；零股/小数股禁用

# 状态取值
STATUS_NEW = "new"
STATUS_ACCEPTED_DRY_RUN = "accepted(dry-run)"
STATUS_ACCEPTED = "accepted"
STATUS_FILLED = "filled"
STATUS_PARTIAL = "partially_filled"
STATUS_REJECTED = "rejected"
STATUS_YIELDED = "yielded"  # 协议 §4.1：落败臂当日站开，根本没提交

# 碰撞标记（与 execution.collision 的常量一致）
COLLISION_VALUES = ("none", "won", "yielded", "merged")

#: 订单/成交流水的固定列顺序。
ORDER_COLUMNS = [
    "order_id",
    "arm",
    "trade_date",
    "submit_ts",
    "PERMNO",
    "symbol",
    "side",
    "qty",
    "notional_target",
    "tif",
    "status",
    "fill_price",
    "fill_qty",
    "fill_ts",
    "crsp_dly_open",   # 待回填
    "prev_close",      # 待回填
    "collision",
    "alloc_ft",
    "alloc_zs",
]

#: 各列的 parquet / pandas dtype，保证多次 append 后 schema 稳定。
ORDER_DTYPES = {
    "order_id": "string",
    "arm": "string",
    "trade_date": "datetime64[ns]",
    "submit_ts": "datetime64[ns]",
    "PERMNO": "Int64",
    "symbol": "string",
    "side": "string",
    "qty": "Int64",
    "notional_target": "float64",
    "tif": "string",
    "status": "string",
    "fill_price": "float64",
    "fill_qty": "Int64",
    "fill_ts": "datetime64[ns]",
    "crsp_dly_open": "float64",
    "prev_close": "float64",
    "collision": "string",
    "alloc_ft": "float64",
    "alloc_zs": "float64",
}

#: `AlpacaOpgClient.fetch_fills` 返回表的列。
FILL_COLUMNS = ["order_id", "PERMNO", "symbol", "side", "fill_price", "fill_qty",
                "fill_ts", "status"]


@dataclass
class OrderRecord:
    """一笔订单在流水表里的一行。"""

    order_id: str
    arm: str
    trade_date: pd.Timestamp
    PERMNO: int
    symbol: str | None = None
    side: str = "buy"
    qty: int | None = None
    notional_target: float = float("nan")
    submit_ts: pd.Timestamp | None = None
    tif: str = TIF_OPG
    status: str = STATUS_NEW
    fill_price: float | None = None
    fill_qty: int | None = None
    fill_ts: pd.Timestamp | None = None
    crsp_dly_open: float | None = None
    prev_close: float | None = None
    collision: str = "none"
    alloc_ft: float = float("nan")
    alloc_zs: float = float("nan")

    def to_dict(self) -> dict:
        return {c: asdict(self)[c] for c in ORDER_COLUMNS}


@dataclass
class FillRecord:
    """一笔成交回报。`fill_qty` 是**该订单的**总成交量；分臂比例在 `alloc_*` 里。"""

    order_id: str
    PERMNO: int
    side: str
    fill_price: float
    fill_qty: int
    fill_ts: pd.Timestamp | None = None
    symbol: str | None = None
    status: str = STATUS_FILLED

    def to_dict(self) -> dict:
        d = asdict(self)
        return {c: d.get(c) for c in FILL_COLUMNS}


def empty_orders() -> pd.DataFrame:
    """空的订单流水表（列与 dtype 都对）。"""
    return coerce_orders(pd.DataFrame(columns=ORDER_COLUMNS))


def coerce_orders(df: pd.DataFrame) -> pd.DataFrame:
    """补齐缺列、丢弃多余列、统一 dtype，返回列序为 :data:`ORDER_COLUMNS` 的新表。"""
    out = pd.DataFrame(index=pd.RangeIndex(len(df)))
    for col in ORDER_COLUMNS:
        src = df[col].to_numpy() if col in df.columns else None
        s = pd.Series(src, index=out.index, dtype="object") if src is not None \
            else pd.Series([pd.NA] * len(out), index=out.index, dtype="object")
        dtype = ORDER_DTYPES[col]
        if dtype.startswith("datetime64"):
            out[col] = pd.to_datetime(s, errors="coerce")
        elif dtype == "Int64":
            # 整股：先 round 再转，避免 2500.0000000001 这类浮点噪声炸掉 astype
            out[col] = (pd.to_numeric(s, errors="coerce")
                        .astype("float64").round().astype("Int64"))
        elif dtype == "float64":
            out[col] = pd.to_numeric(s, errors="coerce").astype("float64")
        else:
            out[col] = s.astype("string")
    return out


def load_orders(path) -> pd.DataFrame:
    """读订单流水；文件不存在时返回空表（不建文件）。"""
    p = Path(path)
    if not p.exists():
        return empty_orders()
    return coerce_orders(pd.read_parquet(p))


def append_orders(path, df: pd.DataFrame) -> Path:
    """**追加**写订单流水，绝不覆盖已有行。

    parquet 没有原生 append，这里读旧表 + concat + 原子换名重写；旧行逐字保留。
    调用方负责 `order_id` 唯一（本函数只在发现重复时报错，不做去重）。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = coerce_orders(df)
    old = load_orders(p)
    merged = pd.concat([old, new], ignore_index=True) if len(old) else new

    dup = merged["order_id"].dropna()
    dup = dup[dup.duplicated()]
    if len(dup):
        raise ValueError(
            f"order_id 重复：{sorted(set(dup.tolist()))[:5]}（共 {dup.nunique()} 个）。"
            "订单流水是 append-only 的，一个 order_id 只能落一行。")

    tmp = p.with_suffix(p.suffix + ".tmp")
    merged.to_parquet(tmp, index=False)
    os.replace(tmp, p)
    return p
