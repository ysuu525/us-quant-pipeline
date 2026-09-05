"""折窗口的机械生成（**不得手写窗口**）。

边界完全来自 ``crsp_pipeline.splits.walk_forward_folds``（训 3 年 / 验 6 个月 /
滚动 6 个月 / purge 6 交易日），与 ``scripts/emit_folds.py`` 用同一条调用：
``walk_forward_folds(cal, "2000-01-03", cal.dates[-1])``，第 i 折即 ``fold{i:02d}``。
不传 ``oos_start``，以便同时得到 2024 年后的干净窗折。

用途限制：本模块只产生**窗口边界**，不读任何分数、标签或收益；生成的窗口随后
由 :func:`unseal.paths.verify_fold` 与封存清单的 ``val_window`` 逐折对账，
对不上即中止整个运行。

**仓库内并存两张折表，不可混用**（v4.1 附录 §5；登记簿 2026-09-05 ``open-issue``）
---------------------------------------------------------------------------------
* **表 B = ``CONFIRM_FOLDS``（本包用的这张）**：``walk_forward_folds(cal,
  "2000-01-03", …)``，即 ``scripts/emit_folds.py:34`` 那一条。封存确认集的**全部**
  产物按此：Kronos FT 封存 fold05 val 2005-01-03..2005-07-01、fold35 ..2020-07-02、
  fold44 2024-07-03..2025-01-02；XGBoost 树基线封存同表。
* **表 A = ``DEV_FOLDS``（本包不用）**：``walk_forward_folds(cal, "2000-01-01", …)``，
  开发折 36–42 的全部 Kronos 产物、``gbdt_baseline.py::FOLDS``、
  ``ridge_probe_folds.py::FOLDS``、HANDOFF 的「42 折」按此。

**两表不可混用**：同一折号在两表下的窗口相差 1–3 个交易日。封存集内部一致
（FT 与树基线都按表 B），故 H3 配对不错位；本轮**不改** ``scripts/emit_folds.py``。

两表交界处的一项剔除见 :data:`EXCLUDED_SIGNAL_DATES`。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from crsp_pipeline.calendar import TradingCalendar
from crsp_pipeline.splits import walk_forward_folds

__all__ = ["FoldWindow", "load_calendar", "fold_windows", "parse_folds", "era_of",
           "EXCLUDED_SIGNAL_DATES", "EXCLUDED_SIGNAL_DATES_FOLD",
           "excluded_dates_for"]

_EMIT_START = "2000-01-03"   # 表 B：与 scripts/emit_folds.py:34 逐字一致
_DEV_START = "2000-01-01"    # 表 A：开发折 36–42 用的那张，**本包不用**

#: v4.1 附录 §5 第 2 条的读取前剔除。
#:
#: 表 B 的 **fold35 止于 2020-07-02**，与表 A 的开发 **fold36 起于 2020-07-01**
#: 重叠 **2 个交易日**；这两个信号日的 Kronos 分数与标签**在开发阶段已被读取**。
#: 为使折 05–35 严格为「未消耗」，读取时把它们从 **fold35** 的一切统计中剔除
#: （H1 / H2 / H2-era / H3 / H6 / E 全部，且在标签计算之前就剔除），
#: **FT 与树基线两侧同样剔除**。影响约 2/3900 日，量级可忽略，纯度优先。
#:
#: **只作用于 fold35**：其余折与这两张表的交界无关，不得连带剔除。
EXCLUDED_SIGNAL_DATES: frozenset[str] = frozenset({"2020-07-01", "2020-07-02"})
#: 上述剔除只适用于这一折（表 A / 表 B 的唯一重叠处）。
EXCLUDED_SIGNAL_DATES_FOLD: int = 35


def excluded_dates_for(fold: int) -> frozenset[pd.Timestamp]:
    """该折需要剔除的信号日（v4.1 附录 §5）。非 fold35 一律为空集。"""
    if int(fold) != EXCLUDED_SIGNAL_DATES_FOLD:
        return frozenset()
    return frozenset(pd.Timestamp(d) for d in EXCLUDED_SIGNAL_DATES)


@dataclass(frozen=True)
class FoldWindow:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp

    @property
    def name(self) -> str:
        return f"fold{self.fold:02d}"


def load_calendar(processed: Path | str) -> TradingCalendar:
    idx = pd.read_parquet(Path(processed) / "market_index.parquet")
    return TradingCalendar.from_market_index(idx, "caldt")


def fold_windows(cal: TradingCalendar,
                 folds: Iterable[int] | None = None) -> dict[int, FoldWindow]:
    """机械生成折窗口。``folds`` 为 None 时返回全部。"""
    generated = walk_forward_folds(cal, _EMIT_START, cal.dates[-1])
    out: dict[int, FoldWindow] = {}
    for i, f in enumerate(generated, start=1):
        out[i] = FoldWindow(i, f.train_start, f.train_end, f.val_start, f.val_end)
    if folds is None:
        return out
    wanted = [int(x) for x in folds]
    missing = [f for f in wanted if f not in out]
    if missing:
        raise ValueError(
            f"机械滚动规则做不出折 {missing}（日历止于 {cal.dates[-1].date()}）；"
            f"须等更新的快照，**不得手写窗口**。")
    return {f: out[f] for f in wanted}


def parse_folds(spec: str) -> tuple[int, ...]:
    """``"5-35"`` / ``"5,6,7"`` / ``"05-35,44"`` → 折号元组（升序、去重）。"""
    got: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            got.update(range(int(lo), int(hi) + 1))
        else:
            got.add(int(part))
    return tuple(sorted(got))


def era_of(window: FoldWindow, cut: str) -> str:
    """按 ``val_start`` 与写死的切点二分。**只切一次，脚本内无分年循环。**"""
    return "early" if window.val_start < pd.Timestamp(cut) else "late"


def two_year_blocks(windows: Sequence[FoldWindow], *,
                    warmup_years: int = 1) -> list[dict]:
    """把折按「验证窗所在自然年」分组，每块 = 该年 + 前 ``warmup_years`` 年。

    给 ``exp11``（hi52 需 252 个观测、mom_12_1 需 231 个）与 ICT 探针
    （回看余量 > 45 交易日）用；每块跨度 ≤ 2 个日历年，符合 `CLAUDE.md` §七
    「列裁剪 + 日期下推 + 分块」。
    """
    by_year: dict[int, list[FoldWindow]] = {}
    for w in windows:
        by_year.setdefault(int(w.val_start.year), []).append(w)
    blocks = []
    for year in sorted(by_year):
        group = sorted(by_year[year], key=lambda w: w.fold)
        blocks.append({
            "lo": f"{year - warmup_years}-01-01",
            "hi": str(max(w.val_end for w in group).date()),
            "folds": tuple(w.fold for w in group),
        })
    return blocks
