"""Alpaca OPG（开盘竞价）客户端 —— 默认 dry-run，不联网。

协议 §2 的提交窗口::

    t 日 19:00 ET 之后  ->  t+1 日 09:00 ET 之前

- Alpaca：19:00 ET 后排队至次日开盘；**09:28 ET 之后、19:00 之前提交会被拒**；
- Nasdaq 市价单进 cross 的截止就是 09:28，09:28:01 之后进的单拿不到 NOOP；
- 09:00 这个**内部**截止留 28 分钟缓冲，协议明写「不得压缩」。

订单类型冻结为 ``type=market, time_in_force=opg``、**整股**
（零股与小数股不参与竞价，禁止使用）。

密钥只从环境变量 ``APCA_API_KEY_ID`` / ``APCA_API_SECRET_KEY`` 读，
**绝不写进文件或日志**。本环境没有 `alpaca-py`、也没有密钥，一切都是 dry-run。
"""
from __future__ import annotations

import hashlib
import os
from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd

from execution.orders import (FILL_COLUMNS, STATUS_ACCEPTED_DRY_RUN, TIF_OPG)

ET = ZoneInfo("America/New_York")

#: 内部提交窗口：[19:00, 24:00) ∪ [00:00, 09:00)
SUBMIT_WINDOW_OPEN = time(19, 0)     # 含
SUBMIT_WINDOW_CLOSE = time(9, 0)     # 不含（09:00:00 整已经拒）
#: 券商 / 交易所的硬截止，仅作报错信息里的说明，判定用不到它。
BROKER_HARD_CUTOFF = time(9, 28)

SUBMIT_RESULT_COLUMNS = ["order_id", "arm", "trade_date", "PERMNO", "symbol", "side",
                         "qty", "notional_target", "tif", "status", "submit_ts",
                         "collision", "alloc_ft", "alloc_zs"]


class SubmitWindowError(RuntimeError):
    """在允许的提交窗口之外提交 OPG 单。"""


def _as_et(now_et) -> datetime:
    if now_et is None:
        return datetime.now(ET)
    ts = pd.Timestamp(now_et)
    if ts.tzinfo is None:
        return ts.to_pydatetime().replace(tzinfo=ET)
    return ts.tz_convert(ET).to_pydatetime()


def in_submit_window(now_et) -> bool:
    """当前是否落在协议 §2 的提交窗口内。naive 时间按 ET 解释。"""
    t = _as_et(now_et).time()
    return t >= SUBMIT_WINDOW_OPEN or t < SUBMIT_WINDOW_CLOSE


def assert_submit_window(now_et) -> datetime:
    """不在窗口内就抛 :class:`SubmitWindowError`；在窗口内返回换算到 ET 的时刻。"""
    t = _as_et(now_et)
    if not in_submit_window(t):
        raise SubmitWindowError(
            f"{t.isoformat()} 不在 OPG 提交窗口内。协议 §2：只允许 "
            f"{SUBMIT_WINDOW_OPEN.strftime('%H:%M')} ET 之后到次日 "
            f"{SUBMIT_WINDOW_CLOSE.strftime('%H:%M')} ET 之前提交；"
            f"券商/交易所硬截止是 {BROKER_HARD_CUTOFF.strftime('%H:%M')} ET，"
            "09:00 的内部截止是 28 分钟缓冲，不得压缩。")
    return t


def _dry_run_order_id(row: dict) -> str:
    """确定性的假 order_id，便于跨进程复现 dry-run 流水。"""
    day = pd.Timestamp(row["trade_date"]).date().isoformat()
    msg = f"dry|{row['arm']}|{day}|{int(row['PERMNO'])}|{row['side']}".encode("utf-8")
    return "dry-" + hashlib.sha256(msg).hexdigest()[:16]


class AlpacaOpgClient:
    """提交 / 查询 OPG 订单。``dry_run=True`` 时**完全不联网**。

    Parameters
    ----------
    dry_run : bool
        True（默认）= 不联网、不需要 `alpaca-py`、不需要密钥。
    paper : bool
        走 paper endpoint。注意协议 §7：**paper 阶段的成交价不得用于任何成本估计**
        （paper 撮合不是真实竞价），paper 只用于验收那五项。
    """

    def __init__(self, dry_run: bool = True, paper: bool = True):
        self.dry_run = bool(dry_run)
        self.paper = bool(paper)
        self._client = None

    # ----------------------------------------------------------- 真实路径依赖
    def _require_sdk(self):
        try:
            import alpaca  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("alpaca-py 未安装；本环境只能 dry-run") from exc
        key = os.environ.get("APCA_API_KEY_ID")
        secret = os.environ.get("APCA_API_SECRET_KEY")
        if not key or not secret:
            # 只报缺失，绝不回显任何取值。
            raise RuntimeError(
                "缺少环境变量 APCA_API_KEY_ID / APCA_API_SECRET_KEY；密钥不得写进文件或日志")
        return alpaca, key, secret

    def _live(self):
        if self._client is None:
            self._require_sdk()
            raise RuntimeError(
                "真实提交路径尚未接线：协议 §1 的 B1--B4 全部未填，"
                "§7 的 paper 预演也未验收，此路径不得启用")
        return self._client

    # --------------------------------------------------------------- 提 交
    def submit_opg_orders(self, orders: pd.DataFrame, *, now_et=None,
                          enforce_window: bool = True) -> pd.DataFrame:
        """提交整股 MOO（``market`` + ``opg``）订单，返回带 `order_id` / `status` 的表。

        输入至少要有 ``arm, trade_date, PERMNO, side``；`qty` / `symbol` /
        `notional_target` / `collision` / `alloc_ft` / `alloc_zs` 有就带上。
        """
        if enforce_window:
            submit_ts = assert_submit_window(now_et)
        else:
            submit_ts = _as_et(now_et)

        if orders is None or not len(orders):
            print(f"[AlpacaOpgClient] dry_run={self.dry_run} 无可提交订单")
            return pd.DataFrame(columns=SUBMIT_RESULT_COLUMNS)

        if not self.dry_run:
            self._live()  # 必然抛错：SDK 缺失 or 未接线

        rows = []
        for row in orders.to_dict("records"):
            qty = row.get("qty")
            if qty is not None and pd.notna(qty) and int(qty) <= 0:
                continue  # 零股不参与竞价，协议 §2 禁用
            rows.append({
                "order_id": _dry_run_order_id(row),
                "arm": row.get("arm"),
                "trade_date": pd.Timestamp(row["trade_date"]),
                "PERMNO": int(row["PERMNO"]),
                "symbol": row.get("symbol"),
                "side": str(row["side"]).strip().lower(),
                "qty": qty,
                "notional_target": row.get("notional", row.get("notional_target")),
                "tif": TIF_OPG,
                "status": STATUS_ACCEPTED_DRY_RUN,
                "submit_ts": pd.Timestamp(submit_ts).tz_localize(None),
                "collision": row.get("collision"),
                "alloc_ft": row.get("alloc_ft"),
                "alloc_zs": row.get("alloc_zs"),
            })
        out = pd.DataFrame(rows, columns=SUBMIT_RESULT_COLUMNS)
        print(f"[AlpacaOpgClient] dry_run={self.dry_run} paper={self.paper} "
              f"submit_ts={submit_ts.isoformat()} tif={TIF_OPG} n_orders={len(out)} "
              f"（未联网）")
        return out

    # --------------------------------------------------------------- 查 询
    def fetch_fills(self, trade_date) -> pd.DataFrame:
        """取某交易日的成交回报。dry-run 返回空表（列齐全）。"""
        if self.dry_run:
            return pd.DataFrame(columns=FILL_COLUMNS)
        self._live()
        raise AssertionError("unreachable")  # pragma: no cover

    def fetch_positions(self) -> dict[int, float]:
        """取真实账户持仓 ``{PERMNO: 名义额}``。dry-run 返回空 dict。"""
        if self.dry_run:
            return {}
        self._live()
        raise AssertionError("unreachable")  # pragma: no cover
