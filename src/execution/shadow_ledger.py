"""协议 §5：双影子账本。

单账户下真实持仓是**执行**的真相，但 FT 与 ZS 各自保有独立的冻结目标与虚拟持仓
账本——否则发生让位（§4.1）后，次日无从判断哪一臂本来该交易什么。

**影子账本记录的是实际分配到的成交，不是理想订单。** 让位会让该臂的影子账本与其
模型目标产生漂移，这是预期内的；次日从**影子持仓**重算目标即可。

冻结构造（HANDOFF §11.3 / 协议 §2）：``TOPN=500``（按滞后 ADV20）、``NT=6`` 套袖、
每日只有 ``day_index % NT`` 那个套袖再平衡、``k = n // 10``（进前 10%）、
``EXIT_PCT=0.30``（仍在前 30% 则保留）、等权 ``1/k``。

.. warning::
   :meth:`ShadowLedger.target_from_scores` 的再平衡逻辑是
   ``scripts/compare_arms_money.py`` 第 76--98 行的**独立重写**（只读该段抄逻辑，
   不 import 它，以免把面板加载与折级读数拖进执行层）。
   ``src/portfolio/construction.py``（另一位工程师同时在写）是同一构造的另一份实现，
   **两者必须最终对拍一致**：同一份 ``scores_by_day`` + ``adv`` 输入下，逐日 sleeve
   成员集合必须逐元素相等。本模块先独立实现，对拍前不得把任一方当作权威。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# 冻结常量（协议 §2，不在本协议内可选）
TOPN = 500
NT = 6
EXIT_PCT = 0.30
MIN_POOL = 50          # compare_arms_money.py:83 的 `if n < 50: continue`

SIDE_BUY = "buy"
SIDE_SELL = "sell"

IDEAL_ORDER_COLUMNS = ["trade_date", "PERMNO", "symbol", "side", "notional", "arm"]


def _is_finite(v) -> bool:
    try:
        return bool(np.isfinite(float(v)))
    except (TypeError, ValueError):
        return False


@dataclass
class ReconcileResult:
    """协议 §5 第 6 步的逐股核对结果。"""

    halt: bool
    n_checked: int
    max_abs_diff: float
    tol_notional: float
    diffs: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(
        columns=["PERMNO", "shadow", "real", "diff"]))

    def __bool__(self) -> bool:  # 便于 `if result:` 读作「对平了吗」
        return not self.halt


class ShadowLedger:
    """一臂的影子账本（FT 与 ZS 各一个）。

    Parameters
    ----------
    arm : str
        ``"FT"`` 或 ``"ZS"``，只作标签。
    aum : float
        该臂的 AUM。**协议 §1 B1 是阻塞字段**：只有当该数确实是计划部署规模时
        才可写死；本类不校验它，调用方负责。
    nt : int
        套袖数，冻结为 6。
    """

    def __init__(self, arm: str, aum: float, nt: int = NT):
        if nt < 1:
            raise ValueError("nt 必须 >= 1")
        self.arm = str(arm)
        self.aum = float(aum)
        self.nt = int(nt)
        #: PERMNO -> 名义额（美元）。只记录**实际分配到的成交**。
        self.positions: dict[int, float] = {}
        #: 每个套袖的目标名字（模型目标，与成交无关）；None = 该套袖尚未初始化。
        self.sleeves: list[list[int]] = [None] * self.nt  # type: ignore[list-item]
        #: 最近一次 `target_from_scores` 的单笔名义额 = aum / (nt * k)。
        self.last_unit_notional: float = float("nan")

    # ------------------------------------------------------------- 目标构造
    def target_from_scores(self, scores_by_day, day, day_index: int, adv_today,
                           *, topn: int = TOPN, exit_pct: float = EXIT_PCT,
                           nt: int = NT) -> dict[int, float] | None:
        """单日再平衡，返回目标账本 ``{PERMNO: 名义额}``；池子不足时返回 ``None``。

        逐行对应 ``scripts/compare_arms_money.py`` 第 76--98 行：

        =========================================  ====================================
        compare_arms_money.py                       本方法
        =========================================  ====================================
        ``s, a = by_day[day], adv.get(day, {})``    ``scores_by_day[day]`` / ``adv_today``
        ``elig = [p for p in s if p in a and ...]`` 同（滞后 ADV20 必须有限）
        ``elig = sorted(..., -a[p])[:topn]``        同（按 ADV 降序取 top500）
        ``s = {p: v for ... if p in elig}``         同（**保持原 dict 顺序**，rank 的 tie 依赖它）
        ``if n < 50: continue``                     返回 ``None``，且**不动任何套袖**
        ``pct = (pd.Series(s).rank() / n)``         同（average 法排名）
        ``k = max(1, n // 10)``                     同
        ``order = sorted(pct, key=-pct[p])``        同（stable sort，tie 保持 dict 顺序）
        ``j = i % NT``                              ``j = day_index % nt``
        ``prev is None -> nb = order[:k]``          同（冷启动，整袖建仓）
        ``keep = [p for p in prev if pct[p] >= 1-EXIT_PCT][:k]``  同（跌出前 30% 才卖）
        ``add = [p for p in order if p not in held][:k-len(keep)]``  同
        ``book[j] = keep + add``                    ``self.sleeves[j] = keep + add``
        =========================================  ====================================

        目标名义额：等权 ``1/k``、``NT`` 套袖平分 -> 每个 (套袖, 名字) 槽位
        ``aum / (nt * k)``；一个名字同时落在多个套袖里则名义额相加。

        .. important::
           **调用方契约：`day_index` 必须在池子不足的日子上照常递增。**
           原实现里 ``if n < 50: continue`` 位于 ``for i, day in enumerate(days)``
           内部，被跳过的日子**照样吃掉一个 `i`**、只是不重平衡 ``i % NT``。
           本方法在这种日子返回 ``None`` 且不动任何套袖，与之等价——
           但前提是调用方不要把 `day_index` 停住，否则套袖相位会整体错开。
           （另注：``days`` 本身来自已经过滤掉 ``len(g) < 50`` 的分数表，
           这里的 ``n`` 是**再过一遍 ADV/topn 之后**的池子大小。）
        """
        if nt != self.nt:
            raise ValueError(f"nt={nt} 与本账本的 nt={self.nt} 不一致；套袖数是冻结常量")

        s_raw = scores_by_day.get(day) if hasattr(scores_by_day, "get") else scores_by_day[day]
        if not s_raw:
            return None
        a = adv_today or {}

        elig = [p for p in s_raw if p in a and _is_finite(a[p])]
        if topn and len(elig) > topn:
            elig = sorted(elig, key=lambda p: -float(a[p]))[:topn]
        elig = set(elig)
        s = {p: v for p, v in s_raw.items() if p in elig}
        n = len(s)
        if n < MIN_POOL:
            return None

        pct = (pd.Series(s).rank() / n).to_dict()
        k = max(1, n // 10)
        order = sorted(pct, key=lambda p: -pct[p])

        j = day_index % nt
        prev = self.sleeves[j]
        if prev is None:
            nb = list(order[:k])
        else:
            keep = [p for p in prev if p in pct and pct[p] >= 1 - exit_pct][:k]
            held = set(keep)
            add = [p for p in order if p not in held][:k - len(keep)]
            nb = keep + add
        self.sleeves[j] = nb

        unit = self.aum / (nt * k)
        self.last_unit_notional = unit
        target: dict[int, float] = {}
        for sleeve in self.sleeves:
            if not sleeve:
                continue
            for p in sleeve:
                target[int(p)] = target.get(int(p), 0.0) + unit
        return target

    # ------------------------------------------------------------- 理想订单
    def ideal_orders(self, target_book: dict[int, float], *, trade_date=None,
                     symbols: dict[int, str] | None = None,
                     tol_notional: float = 1e-9) -> pd.DataFrame:
        """目标账本 − 当前**影子持仓** -> 该臂的理想订单（协议 §5 第 2 步）。

        注意基准是影子持仓，不是真实持仓：让位造成的漂移就是这样被带到次日的。
        """
        target_book = target_book or {}
        symbols = symbols or {}
        rows = []
        for permno in sorted(set(map(int, target_book)) | set(map(int, self.positions))):
            delta = float(target_book.get(permno, 0.0)) - float(self.positions.get(permno, 0.0))
            if abs(delta) <= tol_notional:
                continue
            rows.append({
                "trade_date": pd.Timestamp(trade_date) if trade_date is not None else pd.NaT,
                "PERMNO": permno,
                "symbol": symbols.get(permno),
                "side": SIDE_BUY if delta > 0 else SIDE_SELL,
                "notional": abs(delta),
                "arm": self.arm,
            })
        return pd.DataFrame(rows, columns=IDEAL_ORDER_COLUMNS)

    # ------------------------------------------------------------- 成交回写
    def apply_fills(self, fills: pd.DataFrame, *, tol_notional: float = 1e-9) -> None:
        """按**实际分配到该臂的成交**回写影子持仓（协议 §5 第 5 步）。

        `fills` 至少要有 ``PERMNO, side``，以及 ``notional``（该臂分到的名义额）
        或 ``fill_price`` + ``fill_qty``（此时名义额 = 价 x 量，量已按 alloc 比例分过）。

        **让位后「没成交」的情况**：该笔根本不出现在 fills 里，于是影子持仓不变、
        影子目标（sleeves）也不变——次日照常从影子持仓重算，漂移被显式带走。
        """
        if fills is None or not len(fills):
            return
        for row in fills.to_dict("records"):
            permno = int(row["PERMNO"])
            if "notional" in row and row["notional"] is not None and _is_finite(row["notional"]):
                notional = float(row["notional"])
            else:
                notional = float(row["fill_price"]) * float(row["fill_qty"])
            if not _is_finite(notional) or notional == 0.0:
                continue
            side = str(row["side"]).strip().lower()
            if side == SIDE_BUY:
                signed = notional
            elif side == SIDE_SELL:
                signed = -notional
            else:
                raise ValueError(f"side 只能是 'buy'/'sell'，收到 {row['side']!r}")
            new = self.positions.get(permno, 0.0) + signed
            if abs(new) <= tol_notional:
                self.positions.pop(permno, None)
            else:
                self.positions[permno] = new

    # --------------------------------------------------------------- 核 对
    def reconcile(self, real_positions: dict[int, float],
                  tol_notional: float) -> ReconcileResult:
        """把**本臂**影子持仓与给定的真实持仓逐股核对。

        协议 §5 第 6 步要求核对的是 ``sum(两臂影子持仓) == 真实账户持仓``；
        单账户下请用模块级的 :func:`reconcile_arms`。本方法是单臂版本
        （双账户 B4 分叉、或只跑一臂时适用），语义一致：**不平即 halt**。
        """
        return _reconcile_books(self.positions, real_positions, tol_notional)

    # ------------------------------------------------------------------ misc
    def snapshot(self) -> dict:
        return {
            "arm": self.arm,
            "aum": self.aum,
            "nt": self.nt,
            "n_positions": len(self.positions),
            "gross_notional": float(sum(abs(v) for v in self.positions.values())),
            "sleeve_sizes": [None if s is None else len(s) for s in self.sleeves],
            "last_unit_notional": self.last_unit_notional,
        }


def _reconcile_books(shadow: dict[int, float], real: dict[int, float],
                     tol_notional: float) -> ReconcileResult:
    real = {int(k): float(v) for k, v in (real or {}).items()}
    shadow = {int(k): float(v) for k, v in (shadow or {}).items()}
    keys = sorted(set(shadow) | set(real))
    rows, max_abs = [], 0.0
    for permno in keys:
        sv, rv = shadow.get(permno, 0.0), real.get(permno, 0.0)
        d = sv - rv
        max_abs = max(max_abs, abs(d))
        if abs(d) > tol_notional:
            rows.append({"PERMNO": permno, "shadow": sv, "real": rv, "diff": d})
    diffs = pd.DataFrame(rows, columns=["PERMNO", "shadow", "real", "diff"])
    return ReconcileResult(halt=bool(len(diffs)), n_checked=len(keys),
                           max_abs_diff=max_abs, tol_notional=float(tol_notional),
                           diffs=diffs)


def combined_positions(ledgers) -> dict[int, float]:
    """两臂影子持仓逐股相加。`ledgers` 可以是 ShadowLedger 的序列或映射。"""
    books = ledgers.values() if hasattr(ledgers, "values") else ledgers
    total: dict[int, float] = {}
    for led in books:
        for permno, notional in led.positions.items():
            total[int(permno)] = total.get(int(permno), 0.0) + float(notional)
    return total


def reconcile_arms(ledgers, real_positions: dict[int, float],
                   tol_notional: float) -> ReconcileResult:
    """协议 §5 第 6 步：``sum(两臂影子持仓)`` 与真实总持仓**逐股**核对。

    不平即 ``halt=True``，调用方必须当日停机（停机日不计入 §6 的观察日数）。
    """
    return _reconcile_books(combined_positions(ledgers), real_positions, tol_notional)
