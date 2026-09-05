"""exp14：财报日回避规则（Rule E）在冻结 NT=5 构造上的开发折估计。

任务书 `docs/任务书_exp14_RuleE_开发折估计_2026-09-05.md`
（冻结哈希 sha256 c790cb27a7ed55c9935453af398b436d1dd7ec428c55c7ed1519d98249568ba8）。
下面 §2–§5 是任务书原文全文抄录，先于读第一行数据落笔（CLAUDE.md §二）。

================================================================================
## 2. 硬禁令

1. 只读折 36–42 的 `scores.parquet`，FT 与 ZS 两臂（k=2 两臂都在实盘候选内），走
   `signals.kronos_adapter.scores_path` 与 `crsp_pipeline.sealed.assert_readable`；
   折号白名单断言。任何对折 05–35、fold44–45、封存窗、`*sealed*` 的读取即全部作废。
2. **不改** `src/portfolio/construction.py`、`scripts/nt5_baseline_readout.py`、
   `scripts/compare_arms_money.py`、`scripts/exp13_compustat_dev_diag.py`、
   `scripts/build_compustat_link.py`、v4。带过滤的构造写成**新模块**
   `src/portfolio/construction_rule_e.py`，函数签名 = 冻结函数 + 一个 `eligible` 掩码；
   **掩码全 True 时必须逐位重现冻结函数输出**（测试断言，`np.array_equal` 或
   `allclose(atol=0)`）。
3. 全脚本无 `label`；交付附 `grep -n label` 原始输出。
4. 不 `git add`、不 `git commit`、不追加 `experiments/ledger.md`。
5. 内存：复用 exp13 派生 parquet，不再碰 CSV；价格加载照 `nt5_baseline_readout.py`
   （复用 `compare_arms_money.load_prices`）。峰值提交内存增量 < 6 GB。不用 GPU。
6. 不得编数字；没核到写「未核」。

## 3. Rule E 的实现口径（先写死）

**定义**（抄自 `docs/设计_财报日回避规则_2026-09-05.md` §2，本任务不得改）：信号日 t，
若名字 i 的预期下次公告日 `rdq_hat_i` 满足
`rdq_hat_i − 1 日历日 ≤ d ≤ rdq_hat_i + 1 日历日` 且 `d ∈ [t+1, t+6]`（交易日），
则 i 在 t 日**不得新建仓**；名额由下一名递补；已持仓不强制退出；无 `rdq_hat` 者视为可进。

**注入点**（agent 先读 `construction.py`，在报告里写明冻结函数的进场与退出逻辑各在哪几行，
再按下列要求实现）：
- 掩码**只作用于当日新建仓的候选选择**：入选集合 = 按分数排名、在**可进**名字中取，
  **入选数量与无掩码时相同**（名额递补）。
- **退出排名在全池上算**，与冻结逻辑完全一致；被掩码的已持仓名字**不因掩码而退出**。
- `rdq_hat` 的时点：t 日只能用 t 日及以前已知的公告日（上一财年同季的 `rdq` ≤ t）。
  与 exp13 §4.2 同源，**优先 import exp13 的 `build_quarter_features` /
  `assign_point_in_time` / `_signed_trading_distance`**，不可 import 则逐字抄并注明行号。
- 交易日与日历日：`[t+1, t+6]` 按 `processed` 交易日历取；slack 的 ±1 按日历日。

## 4. 交付（全部估计交付；每个量报点估计，配对量另报 NW(5) 95% CI 与 7 折方向）

### 4.1 E0 基线
FT / ZS 各一条：冻结 NT=5、无掩码。**必须与 `outputs/nt5_baseline_readout.json` 逐位一致**
（毛年化、净 8bp 年化、夏普、MDD、年单边交易），否则停下报告。

### 4.2 E1 主规格：Rule E 如定义（`rdq_hat`，slack ±1，只禁进）
每臂报：毛年化、成本网格 2/4/8/12/16/22bp 净年化、每 bp 拖累、BE 单边成本、
年单边交易次数、净(8bp) 年化波动 / 夏普 / MDD / 最长水下、毛日收益 NW(5) t、
逐折毛年化与正折数（全部沿 `nt5_baseline_readout.summarize` 的口径）。
**配对量 vs E0**：逐日净(8bp)收益差的均值 + NW(5) 95% CI + 7 折正负；
逐日净收益**方差比**（E1/E0）合并与逐折；净夏普差（只报点估计）。
**过滤统计**：每日被掩码的入选候选占比、被掩码而递补的名额占比、
持仓-日中「若无 Rule E 本会持有」的份额、无 `rdq_hat` 的名字占候选池比例。

### 4.3 E2 预测误差（高功效描述量，唯一可据以修订 slack 的读数）
样本：折 36–42 验证窗内、top500 候选池名字的全部真实公告 `rdq`（沿 exp13 的 permno-季度表）。
对每个 `rdq` 找对应的 `rdq_hat`（同财季上一年 + 1 年）：
- `err = rdq − rdq_hat`（日历日）的分布：均值、中位数、P5/P25/P75/P95；
  `|err| ≤ 0 / 1 / 2 / 3 / 5 / 7 / 14` 的累计占比；`rdq_hat` 缺失占比；逐折与合并；
- **规则视角的命中率**：真实 `rdq` 落在某个 [t+1, t+6] 窗内的事件中，
  被 Rule E（slack ±1）在同一 t 拦住的比例（= 事件层面的召回）；
  以及被拦住但真实 `rdq` 不在窗内的比例（误拦）。
  **这两个数是 slack 是否「机械上够用」的直接证据。**

### 4.4 E3 上界与敏感性（只报不选）
表格，每格 = 4.2 的读数子集（净 8bp 年化、夏普、MDD、方差比、配对均值差 + CI、
7 折方向、年单边交易）：
- 日历 ∈ {`rdq_hat`（代理，下界）, `rdq_real`（真实公告日，**事后信息**，完美前瞻日历的上界）}；
- slack ∈ {0, 1, 2, 3, 5} 日历日；
- 动作 ∈ {只禁进, 禁进 + 强制退出（持仓名字若下一窗含公告日则在该档轮到时卖出）}。
共 2×5×2 = 20 格 + E0。**报告不得对任一格使用「最优 / 更好 / 建议」措辞**；
每格标明是否用了事后信息。

### 4.5 措辞模板
> 在开发折 36–42（已消耗，方向性证据）上，冻结 NT=5 构造加入 Rule E（`rdq_hat`，
> slack ±1，只禁进）后，FT 臂净(8bp) 年化由 a 变为 b，净夏普由 c 变为 d，
> 净日收益方差比 e，配对净收益差 f bp/日（NW(5) 95% CI [·,·]，N/7 折正）；
> 被拦住的入选候选占比 g。**不作判定。**

## 5. 判据（先写死）
- **无门槛、无 PASS/FAIL、无「有效 / 无效 / 更好」措辞。**
- E2 的读数**可以**成为修订 slack 的方法学依据（例：±1 日只拦住 30% 的真实公告 ⇒
  slack 机械不足）；但修订本身须由用户裁定并登记「修订发生在看到结果之后」。
- E1 / E3 的收益、夏普、方差**不得**成为修订任何固定项的依据。
- MDE 未算，本任务不是检验。
================================================================================

实现说明（不属于任务书，属交付时必须披露的实现选择）
----------------------------------------------------
* 折号白名单：本脚本只接受折 36–42，读任何文件之前先断言（:func:`assert_folds`）。
  分数路径走 ``signals.kronos_adapter.scores_path``（该函数自身也有 ``ALLOWED_FOLDS``
  守卫）；所有 parquet 读取前调用 ``crsp_pipeline.sealed.assert_readable``。
* 构造：``portfolio.construction_rule_e.rule_e_long_only_returns``（新模块），
  掩码全 True 时与 ``portfolio.construction.frozen_long_only_returns`` 逐位相同。
* 口径复用：``summarize`` / ``max_drawdown_and_underwater`` / ``COST_GRID_BP`` / ``NT``
  直接 import ``scripts/nt5_baseline_readout.py``（只 import，不改）；
  价格加载 import ``scripts/compare_arms_money.py::load_prices``。
* ``rdq_hat``：import ``scripts/exp13_compustat_dev_diag.py`` 的
  ``load_fundq`` / ``load_link`` / ``build_quarter_features`` / ``broadcast_to_permno`` /
  ``assign_point_in_time`` / ``load_trading_calendar``（只 import，不改）。
  ``assign_point_in_time`` 返回的 ``rdq_hat`` 列已带前视自查（由 t 日之前已公告的
  ``rdq`` 推出，见 exp13:423-426）。
* ``rdq_real``（E3 的事后日历上界）：exp13 只算 ``ea_real`` 的交易日距离、不返回日期，
  故本脚本自写 :func:`nearest_future_rdq_real`，逻辑镜像 exp13:430-438（同一 forward
  ``merge_asof``、``allow_exact_matches=True``），差别只是返回日期而非距离。
* 无未来收益变量：本脚本全文无 CLAUDE.md §一.1 点名的那个变量名
  （交付附 grep 原始输出）。收益只在 t+1 及以后作为结果进入冻结构造。
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from crsp_pipeline.sealed import assert_readable          # noqa: E402
from crsp_pipeline.signal_eval import newey_west_tstat    # noqa: E402
from portfolio.construction import scores_frame_to_by_day  # noqa: E402
from portfolio.construction_rule_e import (               # noqa: E402
    EligibilityMask, rule_e_long_only_returns,
)
from signals.kronos_adapter import scores_path            # noqa: E402

TASK_DOC = "docs/任务书_exp14_RuleE_开发折估计_2026-09-05.md"
TASK_DOC_SHA256 = "c790cb27a7ed55c9935453af398b436d1dd7ec428c55c7ed1519d98249568ba8"

PROCESSED = Path(r"F:\quant\processed\crsp_ciz_2026-08-24_20260825T130601Z")
DERIVED = Path(r"F:\quant\external\compustat\derived")
OUT_DIR = REPO / "outputs" / "exp14_rule_e_dev_estimate"
BASELINE_JSON = REPO / "outputs" / "nt5_baseline_readout.json"

#: 折号白名单（任务书 §2.1；CLAUDE.md §四）。只有已消耗的现代开发折。
ALLOWED_FOLDS_EXP14: frozenset[int] = frozenset(range(36, 43))
FOLDS: tuple[int, ...] = tuple(range(36, 43))
ARMS: tuple[str, ...] = ("ft", "zs")

#: 冻结 NT=5 构造（与 nt5_baseline_readout.py:45-46 同）
TOPN, EXIT_PCT, MIN_NAMES = 500, 0.30, 50
NET_COST_BP = 8.0            # 配对量与 E3 表的净收益口径
NW_LAG = 5
Z95 = 1.959963984540054

#: §3 的固定项：窗口 = 冻结持有期 6 个交易日；主规格 slack = ±1 日历日
HOLDING_DAYS = 6
MAIN_SLACK_DAYS = 1
#: §4.4 敏感性网格
GRID_CALENDARS: tuple[str, ...] = ("rdq_hat", "rdq_real")
GRID_SLACKS: tuple[int, ...] = (0, 1, 2, 3, 5)
GRID_ACTIONS: tuple[str, ...] = ("entry_only", "entry_and_exit")
#: E1 主规格在网格里的格名
MAIN_CONFIG = "rdq_hat|slack1|entry_only"
BASELINE_CONFIG = "E0"

#: E2 的 |err| 累计门槛（日历日）
ERR_THRESHOLDS: tuple[int, ...] = (0, 1, 2, 3, 5, 7, 14)

#: 只在这个日期区间内保留季度行（只为省内存 / 提速；rdq_hat 已在
#: build_quarter_features 里算完，过滤不改任何 rdq_hat 取值）
QUARTER_KEEP_LO = pd.Timestamp("2019-01-01")
QUARTER_KEEP_HI = pd.Timestamp("2025-12-31")

MEMORY_LIMIT_GB_DEFAULT = 56.0


class FoldWhitelistError(RuntimeError):
    """折号越出 exp14 白名单。"""


def assert_folds(folds: Iterable[int]) -> tuple[int, ...]:
    """读任何文件之前调用。任务书 §2.1。"""
    got = tuple(int(f) for f in folds)
    bad = sorted(set(got) - ALLOWED_FOLDS_EXP14)
    if bad:
        raise FoldWhitelistError(
            f"exp14 只允许折 {sorted(ALLOWED_FOLDS_EXP14)}；收到越界折号 {bad}。"
            "折 05–35 / fold44–45 / 封存窗的读取即全部作废（任务书 §2.1）。"
        )
    if not got:
        raise FoldWhitelistError("折号为空")
    return got


def log(message: str) -> None:
    print(message, flush=True)


def _load_script_module(name: str, path: Path):
    """照 exp13:153-160；只 import，不改被引脚本。"""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# 内存
# ---------------------------------------------------------------------------
def committed_memory_gb() -> float:
    """Windows 提交内存（照 exp11:142-168）。"""
    if sys.platform == "win32":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        return float(status.ullTotalPageFile - status.ullAvailPageFile) / 2**30
    values: dict[str, float] = {}
    with Path("/proc/meminfo").open(encoding="ascii") as handle:
        for line in handle:
            key, value = line.split(":", 1)
            values[key] = float(value.strip().split()[0]) * 1024
    return values["Committed_AS"] / 2**30


# ---------------------------------------------------------------------------
# 候选池（逐字复现 construction.py:151-159 的池构造，只读诊断用）
# ---------------------------------------------------------------------------
def daily_pools(scores_by_day: dict, adv_by_day: dict, *, topn: int = TOPN,
                min_names: int = MIN_NAMES) -> dict:
    """``{day: set(PERMNO)}``：当日进入 `pct`/`k` 计算的池，与冻结构造同一取法。

    逐字复现 `src/portfolio/construction.py:151-159`（ADV 有限性过滤 → 按 lagged
    ADV20 取前 topn → 当日 < min_names 的天整天不产出）。只用于过滤统计与 E2 的
    (t, PERMNO) 对，不进任何浮点运算。
    """
    by_day = {d: s for d, s in scores_by_day.items() if len(s) >= min_names}
    pools: dict = {}
    for day in sorted(by_day):
        s, a = by_day[day], adv_by_day.get(day, {})
        elig = [p for p in s if p in a and np.isfinite(a[p])]
        if topn and len(elig) > topn:
            elig = sorted(elig, key=lambda p: -a[p])[:topn]
        elig = set(elig)
        if len({p for p in s if p in elig}) < min_names:
            continue
        pools[day] = elig
    return pools


# ---------------------------------------------------------------------------
# rdq_real（事后日历）：镜像 exp13:430-438，差别只是返回日期
# ---------------------------------------------------------------------------
def nearest_future_rdq_real(keys: pd.DataFrame, permno_quarters: pd.DataFrame) -> pd.Series:
    """每个 (PERMNO, signal_date) 的最近一次未来实际 `rdq`（>= t）。

    逻辑镜像 `scripts/exp13_compustat_dev_diag.py:430-438`（同一 forward
    ``merge_asof``、``by="PERMNO"``、``allow_exact_matches=True``）；exp13 把它折成
    ``ea_real`` 的交易日距离，本函数保留日期本身，供 E3 的事后日历格使用。
    **这是事后信息**（真实公告日在 t 日尚不可知），只作上界敏感性。
    """
    left = keys[["PERMNO", "signal_date"]].copy()
    left["PERMNO"] = pd.to_numeric(left["PERMNO"], errors="raise").astype("int64")
    left["signal_date"] = pd.to_datetime(left["signal_date"]).astype("datetime64[ns]")
    left = left.sort_values(["signal_date", "PERMNO"], kind="mergesort").reset_index(drop=True)
    reals = permno_quarters.dropna(subset=["rdq"])[["PERMNO", "rdq"]].drop_duplicates()
    reals = reals.sort_values(["rdq", "PERMNO"], kind="mergesort").reset_index(drop=True)
    fwd = pd.merge_asof(left, reals, left_on="signal_date", right_on="rdq", by="PERMNO",
                        direction="forward", allow_exact_matches=True)
    return fwd.set_index(["PERMNO", "signal_date"])["rdq"].rename("rdq_real_next")


# ---------------------------------------------------------------------------
# Rule E 的窗口判定
# ---------------------------------------------------------------------------
def window_blocked(calendar: pd.DatetimeIndex, signal_date: pd.Series,
                   event_date: pd.Series, *, slack_days: int,
                   holding_days: int = HOLDING_DAYS) -> np.ndarray:
    """Rule E 的触发判定（向量化）。

    定义（任务书 §3）：存在交易日 d，使
    ``event − slack 日历日 <= d <= event + slack 日历日`` 且 ``d ∈ [t+1, t+holding]``
    （交易日，按 `processed` 的 CRSP 交易日历）。

    实现：窗口 = 日历上 t 之后的第 1..holding 个交易日（位置区间
    ``[t_pos+1, t_pos+holding]``）。设 ``lo = event − slack``、``hi = event + slack``，
    ``p_lo`` = 日历上 ``>= lo`` 的首个交易日的位置，
    ``start = max(t_pos+1, p_lo)`` = **窗口内**第一个 ``>= lo`` 的交易日的位置。
    窗口内存在满足条件的 d **当且仅当** ``start <= t_pos+holding`` 且
    ``calendar[start] <= hi``（窗口内的交易日按日历升序，第一个 ``>= lo`` 的若已
    超过 ``hi``，后面的都超过）。

    事件日缺失 ⇒ 不触发（§3「无 rdq_hat 者视为可进」）。
    """
    cal = calendar.to_numpy(dtype="datetime64[ns]")
    t_values = pd.to_datetime(signal_date).to_numpy(dtype="datetime64[ns]")
    t_pos = np.searchsorted(cal, t_values, side="left").astype(np.int64)
    if (t_pos >= len(cal)).any() or (cal[np.clip(t_pos, 0, len(cal) - 1)] != t_values).any():
        raise ValueError("有信号日不在 CRSP 交易日历上")
    events = pd.to_datetime(event_date).to_numpy(dtype="datetime64[ns]")
    known = ~pd.isna(events)
    slack = np.timedelta64(int(slack_days), "D")
    lo = events - slack          # NaT − timedelta = NaT
    hi = events + slack
    p_lo = np.searchsorted(cal, lo, side="left").astype(np.int64)
    window_ok = (t_pos + holding_days) < len(cal)      # 窗口必须在日历内完整存在
    hit = np.zeros(len(t_pos), dtype=bool)
    idx = np.where(known & window_ok)[0]
    if len(idx):
        start = np.maximum(t_pos[idx] + 1, p_lo[idx])
        ok = start <= t_pos[idx] + holding_days        # start 必落在日历内
        sel = np.where(ok, start, 0)
        hit[idx] = ok & (cal[sel] <= hi[idx])
    return hit


def window_bounds(calendar: pd.DatetimeIndex, signal_date: pd.Series,
                  *, holding_days: int = HOLDING_DAYS) -> tuple[np.ndarray, np.ndarray]:
    """每个 t 的持有窗首尾交易日 (d1, d_holding)；窗口不完整时置 NaT。"""
    cal = calendar.to_numpy(dtype="datetime64[ns]")
    t_values = pd.to_datetime(signal_date).to_numpy(dtype="datetime64[ns]")
    t_pos = np.searchsorted(cal, t_values, side="left").astype(np.int64)
    ok = (t_pos + holding_days) < len(cal)
    first = np.full(len(t_pos), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    last = np.full(len(t_pos), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    first[ok] = cal[t_pos[ok] + 1]
    last[ok] = cal[t_pos[ok] + holding_days]
    return first, last


def build_masks(pit: pd.DataFrame, calendar: pd.DatetimeIndex) -> dict:
    """(日历, slack) → :class:`EligibilityMask`。

    ``pit`` 列：PERMNO / signal_date / rdq_hat / rdq_real_next。
    """
    masks: dict = {}
    for cal_name, column in (("rdq_hat", "rdq_hat"), ("rdq_real", "rdq_real_next")):
        for slack in GRID_SLACKS:
            hit = window_blocked(calendar, pit["signal_date"], pit[column],
                                 slack_days=slack)
            blocked = pit.loc[hit, ["signal_date", "PERMNO"]]
            masks[(cal_name, slack)] = EligibilityMask.from_blocked_frame(
                blocked, meta={"calendar": cal_name, "slack_days": slack,
                               "holding_days": HOLDING_DAYS,
                               "ex_post_information": cal_name == "rdq_real",
                               "blocked_cells": int(hit.sum()),
                               "cells": int(len(pit))})
    return masks


def config_name(calendar: str, slack: int, action: str) -> str:
    return f"{calendar}|slack{slack}|{action}"


def grid_configs() -> list[tuple[str, str, int, str]]:
    """[(name, calendar, slack, action)]，共 20 格（E0 单列）。"""
    out = []
    for calendar in GRID_CALENDARS:
        for slack in GRID_SLACKS:
            for action in GRID_ACTIONS:
                out.append((config_name(calendar, slack, action), calendar, slack, action))
    return out


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------
def net_series(frame: pd.DataFrame, cost_bp: float = NET_COST_BP,
               nt: int = 5) -> pd.Series:
    """净收益 = 毛 − cost_bp 的拖累（与 nt5_baseline_readout.summarize:83,102 同式）。"""
    return frame["r"] - cost_bp / 1e4 * 2.0 * frame["turn"] / nt


def nw_mean_ci(series: pd.Series, lags: int = NW_LAG) -> dict[str, float]:
    stat = newey_west_tstat(pd.Series(series), lags)
    mean, se = float(stat["mean"]), float(stat["se"])
    return {"n": int(stat["n"]), "mean": mean, "se": se, "t": float(stat["t"]),
            "ci95_lo": mean - Z95 * se, "ci95_hi": mean + Z95 * se}


def paired_stats(net_a: pd.Series, net_b: pd.Series, fold: pd.Series) -> dict[str, Any]:
    """配对逐日净收益差（a − b）。均值 bp/日 + NW(5) 95% CI + 逐折方向 + 方差比。"""
    frame = pd.DataFrame({"a": net_a.to_numpy(dtype=float),
                          "b": net_b.to_numpy(dtype=float),
                          "fold": fold.to_numpy()})
    frame["d"] = frame["a"] - frame["b"]
    stat = nw_mean_ci(frame["d"])
    per_fold = {}
    for f, group in frame.groupby("fold"):
        per_fold[str(f)] = {
            "mean_diff_bp": float(group["d"].mean() * 1e4),
            "var_ratio": (float(group["a"].var(ddof=1) / group["b"].var(ddof=1))
                          if group["b"].var(ddof=1) > 0 else float("nan")),
        }
    var_a = float(frame["a"].var(ddof=1))
    var_b = float(frame["b"].var(ddof=1))
    return {
        "n_days": int(len(frame)),
        "mean_diff_bp_per_day": float(stat["mean"] * 1e4),
        "nw5_se_bp": float(stat["se"] * 1e4),
        "nw5_t": float(stat["t"]),
        "ci95_bp": [float(stat["ci95_lo"] * 1e4), float(stat["ci95_hi"] * 1e4)],
        "folds_positive": int(sum(v["mean_diff_bp"] > 0 for v in per_fold.values())),
        "folds_total": len(per_fold),
        "var_ratio_pooled": (var_a / var_b) if var_b > 0 else float("nan"),
        "per_fold": per_fold,
    }


# ---------------------------------------------------------------------------
# E2
# ---------------------------------------------------------------------------
def e2_error_distribution(events: pd.DataFrame) -> dict[str, Any]:
    """`err = rdq − rdq_hat`（日历日）的分布。`events` 列：rdq / rdq_hat。"""
    n = int(len(events))
    has_hat = events["rdq_hat"].notna()
    err = (events.loc[has_hat, "rdq"] - events.loc[has_hat, "rdq_hat"]).dt.days
    err = err.astype("float64")
    out: dict[str, Any] = {
        "n_events": n,
        "n_with_rdq_hat": int(has_hat.sum()),
        "rdq_hat_missing_share": float(1.0 - has_hat.mean()) if n else float("nan"),
    }
    if len(err):
        out.update({
            "err_mean_days": float(err.mean()),
            "err_median_days": float(err.median()),
            "err_p5_days": float(err.quantile(0.05)),
            "err_p25_days": float(err.quantile(0.25)),
            "err_p75_days": float(err.quantile(0.75)),
            "err_p95_days": float(err.quantile(0.95)),
            "abs_err_cum_share": {
                str(threshold): float((err.abs() <= threshold).mean())
                for threshold in ERR_THRESHOLDS
            },
            # 以全部真实公告为分母（含 rdq_hat 缺失者）
            "abs_err_cum_share_of_all_events": {
                str(threshold): float((err.abs() <= threshold).sum() / n) if n else float("nan")
                for threshold in ERR_THRESHOLDS
            },
        })
    else:
        out.update({"err_mean_days": float("nan"), "err_median_days": float("nan"),
                    "err_p5_days": float("nan"), "err_p25_days": float("nan"),
                    "err_p75_days": float("nan"), "err_p95_days": float("nan"),
                    "abs_err_cum_share": {}, "abs_err_cum_share_of_all_events": {}})
    return out


def e2_rule_recall(pairs: pd.DataFrame) -> dict[str, Any]:
    """事件层召回 / 误拦。`pairs` 列：real_in_window / blocked（布尔）。

    - 召回 = P(blocked | real_in_window)
    - 误拦 = P(not real_in_window | blocked)
    """
    n = int(len(pairs))
    real = pairs["real_in_window"].to_numpy(dtype=bool)
    blocked = pairs["blocked"].to_numpy(dtype=bool)
    n_real, n_blocked = int(real.sum()), int(blocked.sum())
    return {
        "n_stock_days": n,
        "n_real_in_window": n_real,
        "real_in_window_share": float(real.mean()) if n else float("nan"),
        "n_blocked": n_blocked,
        "blocked_share": float(blocked.mean()) if n else float("nan"),
        "n_blocked_and_real": int((blocked & real).sum()),
        "recall_blocked_given_real": (float((blocked & real).sum() / n_real)
                                      if n_real else float("nan")),
        "false_block_share_of_blocked": (float((blocked & ~real).sum() / n_blocked)
                                         if n_blocked else float("nan")),
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run(processed: Path, derived: Path, outputs_root: Path, *,
        folds: Iterable[int] = FOLDS,
        memory_limit_gb: float = MEMORY_LIMIT_GB_DEFAULT) -> dict[str, Any]:
    use_folds = assert_folds(folds)
    start_committed = committed_memory_gb()
    if start_committed > memory_limit_gb:
        raise RuntimeError(f"起始提交内存 {start_committed:.2f} GB > 上限 {memory_limit_gb} GB")
    log(f"[exp14] 起始 committed={start_committed:.2f} GB")
    peak_committed = start_committed

    exp13 = _load_script_module("exp14_exp13", REPO / "scripts" / "exp13_compustat_dev_diag.py")
    nt5 = _load_script_module("exp14_nt5", REPO / "scripts" / "nt5_baseline_readout.py")
    cam = _load_script_module("exp14_cam", REPO / "scripts" / "compare_arms_money.py")
    if nt5.NT != 5:
        raise AssertionError("nt5_baseline_readout.NT 不是 5，口径已变")
    nt_value = int(nt5.NT)

    calendar = exp13.load_trading_calendar(processed)
    fundq = exp13.load_fundq(derived)
    link = exp13.load_link(derived)
    quarters, quarter_stats = exp13.build_quarter_features(fundq)
    del fundq
    gc.collect()
    permno_quarters = exp13.broadcast_to_permno(quarters, link)
    n_pq_full = int(len(permno_quarters))
    keep = (
        permno_quarters["rdq"].between(QUARTER_KEEP_LO, QUARTER_KEEP_HI)
        | permno_quarters["rdq_hat"].between(QUARTER_KEEP_LO, QUARTER_KEEP_HI)
    )
    permno_quarters = permno_quarters.loc[keep].reset_index(drop=True)
    log(f"[exp14] 季度特征 {len(quarters):,} 行 -> permno 层 {n_pq_full:,} 行 -> "
        f"日期过滤后 {len(permno_quarters):,} 行")
    del quarters
    gc.collect()

    assert_readable(cam.P)
    log("[exp14] 加载价格/成交额（复用 compare_arms_money.load_prices，同口径）...")
    ret, oc, adv = cam.load_prices()
    peak_committed = max(peak_committed, committed_memory_gb())
    log(f"[exp14] 价格加载后 committed={committed_memory_gb():.2f} GB")

    configs = [(BASELINE_CONFIG, None, None, None)] + grid_configs()
    daily: dict[str, dict[str, list]] = {
        arm: {name: [] for name, *_ in configs} for arm in ARMS
    }
    filter_stats: dict[str, dict[str, list]] = {
        arm: {name: [] for name, *_ in configs} for arm in ARMS
    }
    held_share_rows: dict[str, list] = {arm: [] for arm in ARMS}
    pool_coverage: list[dict[str, Any]] = []
    e2_events: list[pd.DataFrame] = []
    e2_pairs: list[pd.DataFrame] = []
    arm_key_identical: dict[int, bool] = {}

    for fold in use_folds:
        log(f"[exp14] === fold{fold} ===")
        by_day_arm: dict[str, dict] = {}
        keys_arm: dict[str, pd.DataFrame] = {}
        for arm in ARMS:
            path = scores_path(fold, arm, root=outputs_root)
            assert_readable(path)
            frame = pd.read_parquet(path, columns=["PERMNO", "signal_date", "score"])
            by_day_arm[arm] = scores_frame_to_by_day(frame, min_names=MIN_NAMES)
            clean = frame[["PERMNO", "signal_date", "score"]].dropna()
            clean = clean.assign(signal_date=pd.to_datetime(clean["signal_date"]))
            keys_arm[arm] = (clean[["PERMNO", "signal_date"]]
                             .drop_duplicates()
                             .sort_values(["signal_date", "PERMNO"], kind="mergesort")
                             .reset_index(drop=True))
            del frame, clean

        same_keys = keys_arm["ft"].equals(keys_arm["zs"])
        arm_key_identical[fold] = bool(same_keys)
        pit_by_arm: dict[str, pd.DataFrame] = {}
        masks_by_arm: dict[str, dict] = {}
        for arm in ARMS:
            if same_keys and arm == "zs":
                pit_by_arm["zs"] = pit_by_arm["ft"]
                masks_by_arm["zs"] = masks_by_arm["ft"]
                continue
            matched, _pit_stats = exp13.assign_point_in_time(
                keys_arm[arm], permno_quarters, calendar)
            pit = matched[["PERMNO", "signal_date", "rdq_hat"]].copy()
            real = nearest_future_rdq_real(keys_arm[arm], permno_quarters)
            pit = pit.merge(real.reset_index(), on=["PERMNO", "signal_date"],
                            how="left", validate="one_to_one")
            pit_by_arm[arm] = pit
            masks_by_arm[arm] = build_masks(pit, calendar)
            del matched
            gc.collect()

        fold_pool_names: set = set()
        for arm in ARMS:
            by_day = by_day_arm[arm]
            pools = daily_pools(by_day, adv)
            pit = pit_by_arm[arm]
            masks = masks_by_arm[arm]

            # --- 候选池覆盖（无 rdq_hat 的名字占比）
            pool_frame = pd.DataFrame(
                [(day, permno) for day, names in pools.items() for permno in names],
                columns=["signal_date", "PERMNO"])
            pool_frame["signal_date"] = pd.to_datetime(
                pool_frame["signal_date"]).astype("datetime64[ns]")
            pool_frame["PERMNO"] = pd.to_numeric(
                pool_frame["PERMNO"], errors="raise").astype("int64")
            fold_pool_names |= set(pool_frame["PERMNO"].tolist())
            pool_frame = pool_frame.merge(pit, on=["signal_date", "PERMNO"],
                                          how="left", validate="one_to_one")
            pool_coverage.append({
                "fold": fold, "arm": arm,
                "pool_stock_days": int(len(pool_frame)),
                "pool_days": int(len(pools)),
                "rdq_hat_missing_share_of_pool": float(pool_frame["rdq_hat"].isna().mean()),
                "rdq_real_missing_share_of_pool": float(pool_frame["rdq_real_next"].isna().mean()),
            })

            # --- E0 与网格
            baseline_df = None
            baseline_diag = None
            for name, calendar_name, slack, action in configs:
                if name == BASELINE_CONFIG:
                    mask, force_exit = None, False
                else:
                    mask = masks[(calendar_name, slack)]
                    force_exit = action == "entry_and_exit"
                frame, diag = rule_e_long_only_returns(
                    by_day, ret, oc, adv, eligible=mask, force_exit=force_exit,
                    topn=TOPN, cost_bp=0.0, exit_pct=EXIT_PCT, nt=nt_value,
                    min_names=MIN_NAMES, collect_diagnostics=True)
                frame = frame.copy()
                frame["fold"] = fold
                daily[arm][name].append(frame)
                per_day = diag["per_day"]
                filter_stats[arm][name].append({
                    "fold": fold,
                    "rebalance_days": int(len(per_day)),
                    "sum_slots": int(per_day["n_slots"].sum()),
                    "sum_added": int(per_day["n_added"].sum()),
                    "sum_blocked_candidates": int(per_day["n_blocked_candidates"].sum()),
                    "sum_promoted": int(per_day["n_promoted"].sum()),
                    "sum_slot_shortfall": int(per_day["n_slot_shortfall"].sum()),
                    "sum_force_exited": int(per_day["n_force_exited"].sum()),
                    "sum_keep_before_force_exit": int(per_day["n_keep_before_force_exit"].sum()),
                    "sum_reentered_after_exit": int(per_day["n_reentered_after_exit"].sum()),
                    "sum_blocked_in_pool": int(per_day["n_blocked_in_pool"].sum()),
                    "sum_pool": int(per_day["n_pool"].sum()),
                    "days_with_shortfall": int((per_day["n_slot_shortfall"] > 0).sum()),
                    "mask_blocked_cells": int(diag["mask_blocked_cells"]),
                })
                if name == BASELINE_CONFIG:
                    baseline_df, baseline_diag = frame, diag
                if name == MAIN_CONFIG:
                    held0 = baseline_diag["held_by_trade_date"]
                    held1 = diag["held_by_trade_date"]
                    common = sorted(set(held0) & set(held1))
                    dropped = sum(len(held0[d] - held1[d]) for d in common)
                    base_total = sum(len(held0[d]) for d in common)
                    rule_total = sum(len(held1[d]) for d in common)
                    held_share_rows[arm].append({
                        "fold": fold, "trade_days": len(common),
                        "baseline_position_days": base_total,
                        "rule_e_position_days": rule_total,
                        "position_days_dropped_vs_baseline": dropped,
                    })
                del diag
            del baseline_df, baseline_diag

            # --- E2 的 (t, PERMNO) 对（只需算一次；两臂键相同则只用 ft）
            if arm == "ft" or not same_keys:
                first, last = window_bounds(calendar, pool_frame["signal_date"])
                real_next = pool_frame["rdq_real_next"].to_numpy(dtype="datetime64[ns]")
                real_in_window = (~pd.isna(real_next) & ~pd.isna(first)
                                  & (real_next >= first) & (real_next <= last))
                blocked = window_blocked(calendar, pool_frame["signal_date"],
                                         pool_frame["rdq_hat"], slack_days=MAIN_SLACK_DAYS)
                e2_pairs.append(pd.DataFrame({
                    "fold": fold, "arm": arm,
                    "real_in_window": real_in_window, "blocked": blocked}))
            del pool_frame, pools
            gc.collect()

        # --- E2 的事件表：折窗内、池内名字的全部真实 rdq
        fold_lo, fold_hi = _fold_window(by_day_arm["ft"])
        events = permno_quarters.loc[
            permno_quarters["rdq"].notna()
            & permno_quarters["rdq"].between(fold_lo, fold_hi)
            & permno_quarters["PERMNO"].isin(fold_pool_names),
            ["PERMNO", "fidx", "rdq", "rdq_hat"]].drop_duplicates(["PERMNO", "fidx"])
        events = events.assign(fold=fold)
        e2_events.append(events)
        log(f"  fold{fold}: 池内名字 {len(fold_pool_names):,}，窗内真实公告 {len(events):,}")

        del by_day_arm, keys_arm, pit_by_arm, masks_by_arm
        gc.collect()
        peak_committed = max(peak_committed, committed_memory_gb())

    # ---------------- 汇总 ----------------
    log("[exp14] 汇总读数 ...")
    readouts: dict[str, dict[str, Any]] = {arm: {} for arm in ARMS}
    net_by_config: dict[str, dict[str, pd.DataFrame]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for name, *_ in configs:
            allf = pd.concat(daily[arm][name]).sort_index()
            readouts[arm][name] = nt5.summarize(allf["r"], allf["turn"], allf["fold"])
            net_by_config[arm][name] = pd.DataFrame({
                "net": net_series(allf, NET_COST_BP, nt_value),
                "fold": allf["fold"].to_numpy(),
            })

    paired: dict[str, dict[str, Any]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        base = net_by_config[arm][BASELINE_CONFIG]
        for name, *_ in configs:
            if name == BASELINE_CONFIG:
                continue
            cfg = net_by_config[arm][name]
            joined = cfg.join(base["net"].rename("net_base"), how="inner")
            if len(joined) != len(cfg) or len(joined) != len(base):
                raise AssertionError(f"{arm}/{name}: 配对日集合与 E0 不一致")
            paired[arm][name] = paired_stats(joined["net"], joined["net_base"],
                                             joined["fold"])
            paired[arm][name]["sharpe_diff"] = (
                readouts[arm][name]["net_by_cost_bp"]["8"]["sharpe"]
                - readouts[arm][BASELINE_CONFIG]["net_by_cost_bp"]["8"]["sharpe"])

    # ---------------- E0 一致性 ----------------
    baseline_check = check_baseline(readouts, BASELINE_JSON, use_folds)

    # ---------------- E2 ----------------
    events_all = pd.concat(e2_events, ignore_index=True)
    pairs_all = pd.concat(e2_pairs, ignore_index=True)
    e2 = {
        "definition": {
            "sample": "折 36–42 各自验证窗内、top500 候选池名字的全部真实 rdq",
            "err": "rdq − rdq_hat，日历日",
            "real_in_window": "最近一次未来实际 rdq 落在持有窗首尾交易日的日历闭区间 [d1, d6] 内",
            "blocked": "Rule E（rdq_hat，slack ±1 日历日，窗口 [t+1, t+6] 交易日）触发",
            "unit_recall": "(signal_date, PERMNO) 股票-日，限当日 top500 候选池",
        },
        "pooled_error": e2_error_distribution(events_all),
        "per_fold_error": {
            str(fold): e2_error_distribution(group)
            for fold, group in events_all.groupby("fold")
        },
        "pooled_recall": e2_rule_recall(pairs_all),
        "per_fold_recall": {
            str(fold): e2_rule_recall(group)
            for fold, group in pairs_all.groupby("fold")
        },
        "per_arm_recall": {
            str(arm): e2_rule_recall(group)
            for arm, group in pairs_all.groupby("arm")
        },
    }

    # ---------------- 过滤统计 ----------------
    filter_summary: dict[str, dict[str, Any]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for name, *_ in configs:
            frame = pd.DataFrame(filter_stats[arm][name])
            slots = int(frame["sum_slots"].sum())
            filter_summary[arm][name] = {
                "rebalance_days": int(frame["rebalance_days"].sum()),
                "entry_slots": slots,
                "entries_filled": int(frame["sum_added"].sum()),
                "blocked_entry_candidates": int(frame["sum_blocked_candidates"].sum()),
                "blocked_share_of_entry_slots": (
                    float(frame["sum_blocked_candidates"].sum() / slots) if slots else float("nan")),
                "promoted_names": int(frame["sum_promoted"].sum()),
                "promoted_share_of_entry_slots": (
                    float(frame["sum_promoted"].sum() / slots) if slots else float("nan")),
                "slot_shortfall": int(frame["sum_slot_shortfall"].sum()),
                "days_with_shortfall": int(frame["days_with_shortfall"].sum()),
                "force_exited": int(frame["sum_force_exited"].sum()),
                "keep_before_force_exit": int(frame["sum_keep_before_force_exit"].sum()),
                "reentered_after_exit_same_day": int(frame["sum_reentered_after_exit"].sum()),
                "blocked_share_of_pool": (
                    float(frame["sum_blocked_in_pool"].sum() / frame["sum_pool"].sum())
                    if frame["sum_pool"].sum() else float("nan")),
                "mask_blocked_cells": int(frame["mask_blocked_cells"].sum()),
                "per_fold": frame.to_dict(orient="records"),
            }
        held = pd.DataFrame(held_share_rows[arm])
        filter_summary[arm]["_held_vs_baseline"] = {
            "baseline_position_days": int(held["baseline_position_days"].sum()),
            "rule_e_position_days": int(held["rule_e_position_days"].sum()),
            "position_days_dropped_vs_baseline": int(
                held["position_days_dropped_vs_baseline"].sum()),
            "dropped_share_of_baseline_position_days": float(
                held["position_days_dropped_vs_baseline"].sum()
                / held["baseline_position_days"].sum()),
            "per_fold": held.to_dict(orient="records"),
        }

    end_committed = committed_memory_gb()
    peak_committed = max(peak_committed, end_committed)
    return {
        "meta": {
            "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "task_doc": TASK_DOC,
            "task_doc_sha256": TASK_DOC_SHA256,
            "folds": list(use_folds),
            "fold_whitelist": sorted(ALLOWED_FOLDS_EXP14),
            "arms": list(ARMS),
            "construction": {"NT": nt_value, "TOPN": TOPN, "EXIT_PCT": EXIT_PCT,
                             "MIN_NAMES": MIN_NAMES, "entry_pct": 0.10,
                             "execution": "t close score -> t+1 open, fresh names eat open->close"},
            "rule_e": {"holding_days": HOLDING_DAYS, "main_slack_days": MAIN_SLACK_DAYS,
                       "main_calendar": "rdq_hat", "main_action": "entry_only",
                       "missing_rdq_hat": "eligible"},
            "grid": {"calendars": list(GRID_CALENDARS), "slacks": list(GRID_SLACKS),
                     "actions": list(GRID_ACTIONS)},
            "net_cost_bp": NET_COST_BP,
            "nw_lag": NW_LAG,
            "quarter_stats": quarter_stats,
            "permno_quarter_rows_full": n_pq_full,
            "permno_quarter_rows_kept": int(len(permno_quarters)),
            "arm_score_keys_identical_by_fold": arm_key_identical,
            "committed_memory_gb": {"start": start_committed, "peak": peak_committed,
                                    "end": end_committed,
                                    "delta_peak": peak_committed - start_committed},
            "use_restriction": (
                "开发折 36–42 已消耗，读数为方向性证据；估计交付、无门槛、不判定。"
                "E1 / E3 的收益、夏普、方差不得成为修订任何固定项的依据（任务书 §5）。"
            ),
        },
        "E0_baseline_check": baseline_check,
        "readouts": readouts,
        "paired_vs_E0": paired,
        "filter_stats": filter_summary,
        "pool_coverage": pool_coverage,
        "E2": e2,
    }


def _fold_window(by_day: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    days = sorted(by_day)
    return pd.Timestamp(days[0]), pd.Timestamp(days[-1])


BASELINE_FIELDS = ("gross_annual_pct", "oneway_trades_per_year", "drag_per_bp_annual_pct",
                   "breakeven_oneway_bp", "gross_daily_nw_t", "n_days", "folds_positive")
BASELINE_NET_FIELDS = ("net_annual_pct", "net_geo_annual_pct", "vol_annual_pct",
                       "sharpe", "max_drawdown_pct", "longest_underwater_days")


def check_baseline(readouts: dict, baseline_json: Path,
                   folds: tuple[int, ...] = FOLDS) -> dict[str, Any]:
    """E0 与 `outputs/nt5_baseline_readout.json` 逐位一致核对（任务书 §4.1）。"""
    if not baseline_json.exists():
        return {"status": "baseline_json_missing", "path": str(baseline_json)}
    if tuple(folds) != FOLDS:
        return {"status": "partial_run_not_comparable", "folds": list(folds),
                "note": "基线 JSON 是七折合并读数，只有全折运行才可逐位核对"}
    reference = json.loads(baseline_json.read_text(encoding="utf-8"))
    rows = []
    ok = True
    for arm in ARMS:
        got = readouts[arm][BASELINE_CONFIG]
        want = reference["arms"][arm]
        for field in BASELINE_FIELDS:
            same = got[field] == want[field]
            ok &= bool(same)
            rows.append({"arm": arm, "field": field, "exp14": got[field],
                         "nt5_baseline_readout": want[field], "bitwise_equal": bool(same)})
        for bp in want["net_by_cost_bp"]:
            for field in BASELINE_NET_FIELDS:
                a = got["net_by_cost_bp"][bp][field]
                b = want["net_by_cost_bp"][bp][field]
                same = a == b
                ok &= bool(same)
                rows.append({"arm": arm, "field": f"net_{bp}bp.{field}", "exp14": a,
                             "nt5_baseline_readout": b, "bitwise_equal": bool(same)})
        for fold, value in want["per_fold_gross_annual_pct"].items():
            a = got["per_fold_gross_annual_pct"][fold]
            same = a == value
            ok &= bool(same)
            rows.append({"arm": arm, "field": f"per_fold_gross.{fold}", "exp14": a,
                         "nt5_baseline_readout": value, "bitwise_equal": bool(same)})
    return {"status": "identical" if ok else "MISMATCH",
            "all_bitwise_equal": bool(ok),
            "n_fields_checked": len(rows),
            "mismatches": [r for r in rows if not r["bitwise_equal"]],
            "rows": rows}


# ---------------------------------------------------------------------------
# 报告表格（叙述部分由 report.md 手写；本函数只产机器可核对的表）
# ---------------------------------------------------------------------------
def render_tables(report: dict[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append
    add("<!-- 由 scripts/exp14_rule_e_dev_estimate.py 自动生成，勿手改 -->")
    add("")
    add("## E0 一致性核对（对 outputs/nt5_baseline_readout.json）")
    add("")
    check = report["E0_baseline_check"]
    add(f"状态：`{check['status']}`；核对字段数 {check.get('n_fields_checked', 0)}；"
        f"不一致 {len(check.get('mismatches', []))} 项。")
    add("")
    add("| 臂 | 字段 | exp14 E0 | nt5_baseline_readout.json | 逐位相同 |")
    add("|---|---|---|---|---|")
    keep = {"gross_annual_pct", "net_8bp.net_annual_pct", "net_8bp.sharpe",
            "net_8bp.max_drawdown_pct", "oneway_trades_per_year", "n_days",
            "folds_positive", "gross_daily_nw_t", "breakeven_oneway_bp"}
    for row in check.get("rows", []):
        if row["field"] in keep:
            add(f"| {row['arm'].upper()} | {row['field']} | {row['exp14']!r} | "
                f"{row['nt5_baseline_readout']!r} | {row['bitwise_equal']} |")
    for row in check.get("mismatches", []):
        if row["field"] not in keep:
            add(f"| {row['arm'].upper()} | {row['field']} | {row['exp14']!r} | "
                f"{row['nt5_baseline_readout']!r} | **False** |")
    add("")

    add("## E1 主规格（rdq_hat，slack ±1，只禁进）——两臂全读数")
    add("")
    add("| 量 | FT E0 | FT E1 | ZS E0 | ZS E1 |")
    add("|---|---|---|---|---|")

    def cell(arm: str, config: str, getter) -> str:
        return getter(report["readouts"][arm][config])

    rows_spec = [
        ("毛年化 %", lambda s: f"{s['gross_annual_pct']:.4f}"),
        ("毛日收益 NW(5) t", lambda s: f"{s['gross_daily_nw_t']:.4f}"),
        ("每 bp 拖累 %/年", lambda s: f"{s['drag_per_bp_annual_pct']:.4f}"),
        ("BE 单边成本 bp", lambda s: f"{s['breakeven_oneway_bp']:.4f}"),
        ("年单边交易次数", lambda s: f"{s['oneway_trades_per_year']:.4f}"),
        ("净 2bp 年化 %", lambda s: f"{s['net_by_cost_bp']['2']['net_annual_pct']:.4f}"),
        ("净 4bp 年化 %", lambda s: f"{s['net_by_cost_bp']['4']['net_annual_pct']:.4f}"),
        ("净 8bp 年化 %", lambda s: f"{s['net_by_cost_bp']['8']['net_annual_pct']:.4f}"),
        ("净 12bp 年化 %", lambda s: f"{s['net_by_cost_bp']['12']['net_annual_pct']:.4f}"),
        ("净 16bp 年化 %", lambda s: f"{s['net_by_cost_bp']['16']['net_annual_pct']:.4f}"),
        ("净 22bp 年化 %", lambda s: f"{s['net_by_cost_bp']['22']['net_annual_pct']:.4f}"),
        ("净 8bp 年化波动 %", lambda s: f"{s['net_by_cost_bp']['8']['vol_annual_pct']:.4f}"),
        ("净 8bp 夏普", lambda s: f"{s['net_by_cost_bp']['8']['sharpe']:.4f}"),
        ("净 8bp MDD %", lambda s: f"{s['net_by_cost_bp']['8']['max_drawdown_pct']:.4f}"),
        ("净 8bp 最长水下（日）", lambda s: f"{s['net_by_cost_bp']['8']['longest_underwater_days']}"),
        ("正折数 / 7", lambda s: f"{s['folds_positive']}"),
        ("交易日数", lambda s: f"{s['n_days']}"),
    ]
    for caption, getter in rows_spec:
        add(f"| {caption} | {cell('ft', BASELINE_CONFIG, getter)} | {cell('ft', MAIN_CONFIG, getter)} "
            f"| {cell('zs', BASELINE_CONFIG, getter)} | {cell('zs', MAIN_CONFIG, getter)} |")
    add("")
    add("### 逐折毛年化 %（E0 → E1）")
    add("")
    add("| 折 | FT E0 | FT E1 | ZS E0 | ZS E1 |")
    add("|---|---|---|---|---|")
    for fold in FOLDS:
        key = str(fold)
        add("| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
            fold,
            report["readouts"]["ft"][BASELINE_CONFIG]["per_fold_gross_annual_pct"][key],
            report["readouts"]["ft"][MAIN_CONFIG]["per_fold_gross_annual_pct"][key],
            report["readouts"]["zs"][BASELINE_CONFIG]["per_fold_gross_annual_pct"][key],
            report["readouts"]["zs"][MAIN_CONFIG]["per_fold_gross_annual_pct"][key]))
    add("")
    add("### 配对量 vs E0（逐日净 8bp 收益差 = E1 − E0）")
    add("")
    add("| 臂 | 配对日数 | 均值差 bp/日 | NW(5) SE bp | t | 95% CI bp | 7 折为正 | 方差比 E1/E0 | 净夏普差 |")
    add("|---|---|---|---|---|---|---|---|---|")
    for arm in ARMS:
        p = report["paired_vs_E0"][arm][MAIN_CONFIG]
        add(f"| {arm.upper()} | {p['n_days']} | {p['mean_diff_bp_per_day']:+.4f} | "
            f"{p['nw5_se_bp']:.4f} | {p['nw5_t']:+.3f} | "
            f"[{p['ci95_bp'][0]:+.4f}, {p['ci95_bp'][1]:+.4f}] | "
            f"{p['folds_positive']}/{p['folds_total']} | {p['var_ratio_pooled']:.4f} | "
            f"{p['sharpe_diff']:+.4f} |")
    add("")
    add("### 逐折配对差与方差比（E1 − E0）")
    add("")
    add("| 折 | FT 均值差 bp/日 | FT 方差比 | ZS 均值差 bp/日 | ZS 方差比 |")
    add("|---|---|---|---|---|")
    for fold in FOLDS:
        key = str(fold)
        f_ = report["paired_vs_E0"]["ft"][MAIN_CONFIG]["per_fold"][key]
        z_ = report["paired_vs_E0"]["zs"][MAIN_CONFIG]["per_fold"][key]
        add(f"| {fold} | {f_['mean_diff_bp']:+.4f} | {f_['var_ratio']:.4f} | "
            f"{z_['mean_diff_bp']:+.4f} | {z_['var_ratio']:.4f} |")
    add("")
    add("### E1 过滤统计")
    add("")
    add("| 量 | FT | ZS |")
    add("|---|---|---|")
    fs = report["filter_stats"]
    specs = [
        ("建仓名额总数（E1）", lambda s: f"{s['entry_slots']}"),
        ("实际补入名字数", lambda s: f"{s['entries_filled']}"),
        ("被掩码的入选候选数", lambda s: f"{s['blocked_entry_candidates']}"),
        ("被掩码的入选候选 / 建仓名额", lambda s: f"{s['blocked_share_of_entry_slots']:.6f}"),
        ("因掩码递补进来的名字数", lambda s: f"{s['promoted_names']}"),
        ("递补名字 / 建仓名额", lambda s: f"{s['promoted_share_of_entry_slots']:.6f}"),
        ("名额缺口（可进名字不足）", lambda s: f"{s['slot_shortfall']}"),
        ("出现名额缺口的档-日数", lambda s: f"{s['days_with_shortfall']}"),
        ("被掩码格 / 候选池股票-日", lambda s: f"{s['blocked_share_of_pool']:.6f}"),
        ("同日既退出又进场的名字数", lambda s: f"{s['reentered_after_exit_same_day']}"),
        ("强制退出数（主规格恒为 0）", lambda s: f"{s['force_exited']}"),
    ]
    for caption, getter in specs:
        add(f"| {caption} | {getter(fs['ft'][MAIN_CONFIG])} | {getter(fs['zs'][MAIN_CONFIG])} |")
    for caption, key in (("E0 持仓-日", "baseline_position_days"),
                       ("E1 持仓-日", "rule_e_position_days"),
                       ("「若无 Rule E 本会持有」的持仓-日", "position_days_dropped_vs_baseline"),
                       ("上一行 / E0 持仓-日", "dropped_share_of_baseline_position_days")):
        vals = []
        for arm in ARMS:
            v = fs[arm]["_held_vs_baseline"][key]
            vals.append(f"{v:.6f}" if isinstance(v, float) else f"{v}")
        add(f"| {caption} | {vals[0]} | {vals[1]} |")
    add("")
    add("| 量 | FT | ZS |")
    add("|---|---|---|")
    cov = pd.DataFrame(report["pool_coverage"])
    row_hat, row_real = [], []
    for arm in ARMS:
        sub = cov[cov["arm"] == arm]
        w = sub["pool_stock_days"].to_numpy(dtype=float)
        row_hat.append(float(np.average(sub["rdq_hat_missing_share_of_pool"], weights=w)))
        row_real.append(float(np.average(sub["rdq_real_missing_share_of_pool"], weights=w)))
    add(f"| 无 rdq_hat 的名字占候选池（股票-日加权） | {row_hat[0]:.6f} | {row_hat[1]:.6f} |")
    add(f"| 无未来实际 rdq 的名字占候选池 | {row_real[0]:.6f} | {row_real[1]:.6f} |")
    add("")

    add("## E2 预测误差与规则命中")
    add("")
    e2 = report["E2"]
    pooled = e2["pooled_error"]
    add(f"合并样本：折窗内、候选池名字的真实公告 {pooled['n_events']:,} 个；"
        f"其中有 `rdq_hat` 的 {pooled['n_with_rdq_hat']:,}；"
        f"`rdq_hat` 缺失占比 {pooled['rdq_hat_missing_share']:.6f}。")
    add("")
    add("| 量 | 合并 | " + " | ".join(f"fold{f}" for f in FOLDS) + " |")
    add("|---" * (2 + len(FOLDS)) + "|")

    def e2_row(caption: str, getter) -> None:
        cells = [getter(pooled)]
        for fold in FOLDS:
            cells.append(getter(e2["per_fold_error"][str(fold)]))
        add(f"| {caption} | " + " | ".join(cells) + " |")

    e2_row("真实公告数", lambda s: f"{s['n_events']:,}")
    e2_row("rdq_hat 缺失占比", lambda s: f"{s['rdq_hat_missing_share']:.4f}")
    e2_row("err 均值（日历日）", lambda s: f"{s['err_mean_days']:+.3f}")
    e2_row("err 中位数", lambda s: f"{s['err_median_days']:+.1f}")
    e2_row("err P5", lambda s: f"{s['err_p5_days']:+.1f}")
    e2_row("err P25", lambda s: f"{s['err_p25_days']:+.1f}")
    e2_row("err P75", lambda s: f"{s['err_p75_days']:+.1f}")
    e2_row("err P95", lambda s: f"{s['err_p95_days']:+.1f}")
    for threshold in ERR_THRESHOLDS:
        e2_row(f"|err| <= {threshold}（有 rdq_hat 者为分母）",
               lambda s, k=str(threshold): f"{s['abs_err_cum_share'][k]:.4f}")
    for threshold in ERR_THRESHOLDS:
        e2_row(f"|err| <= {threshold}（全部公告为分母）",
               lambda s, k=str(threshold): f"{s['abs_err_cum_share_of_all_events'][k]:.4f}")
    add("")
    add("### 规则视角：召回与误拦（slack ±1，单位 = 候选池股票-日）")
    add("")
    add("| 量 | 合并 | " + " | ".join(f"fold{f}" for f in FOLDS) + " |")
    add("|---" * (2 + len(FOLDS)) + "|")

    def recall_row(caption: str, getter) -> None:
        cells = [getter(e2["pooled_recall"])]
        for fold in FOLDS:
            cells.append(getter(e2["per_fold_recall"][str(fold)]))
        add(f"| {caption} | " + " | ".join(cells) + " |")

    recall_row("股票-日数", lambda s: f"{s['n_stock_days']:,}")
    recall_row("真实 rdq 落在 [t+1,t+6] 的占比", lambda s: f"{s['real_in_window_share']:.4f}")
    recall_row("被 Rule E 拦住的占比", lambda s: f"{s['blocked_share']:.4f}")
    recall_row("召回 P(拦住 | 真实在窗内)", lambda s: f"{s['recall_blocked_given_real']:.4f}")
    recall_row("误拦 P(真实不在窗内 | 拦住)", lambda s: f"{s['false_block_share_of_blocked']:.4f}")
    add("")

    add("## E3 敏感性表（只报不选；每格标是否用事后信息）")
    add("")
    for arm in ARMS:
        add(f"### {arm.upper()} 臂")
        add("")
        add("| 格 | 事后信息 | 净8bp年化 % | 净8bp夏普 | 净8bp MDD % | 方差比 E/E0 | "
            "配对均值差 bp/日 | 95% CI bp | 7折为正 | 年单边交易 |")
        add("|---|---|---|---|---|---|---|---|---|---|")
        base = report["readouts"][arm][BASELINE_CONFIG]
        b8 = base["net_by_cost_bp"]["8"]
        add(f"| E0（无 Rule E） | 否 | {b8['net_annual_pct']:.4f} | {b8['sharpe']:.4f} | "
            f"{b8['max_drawdown_pct']:.4f} | 1.0000 | — | — | — | "
            f"{base['oneway_trades_per_year']:.2f} |")
        for name, calendar_name, slack, action in grid_configs():
            s = report["readouts"][arm][name]
            n8 = s["net_by_cost_bp"]["8"]
            p = report["paired_vs_E0"][arm][name]
            ex_post = "是" if calendar_name == "rdq_real" else "否"
            act = "只禁进" if action == "entry_only" else "禁进+强退"
            add(f"| {calendar_name} / slack{slack} / {act} | {ex_post} | "
                f"{n8['net_annual_pct']:.4f} | {n8['sharpe']:.4f} | "
                f"{n8['max_drawdown_pct']:.4f} | {p['var_ratio_pooled']:.4f} | "
                f"{p['mean_diff_bp_per_day']:+.4f} | "
                f"[{p['ci95_bp'][0]:+.4f}, {p['ci95_bp'][1]:+.4f}] | "
                f"{p['folds_positive']}/{p['folds_total']} | "
                f"{s['oneway_trades_per_year']:.2f} |")
        add("")
    add("### E3 各格的过滤强度（被掩码格 / 候选池股票-日）")
    add("")
    add("| 格 | FT | ZS |")
    add("|---|---|---|")
    for name, calendar_name, slack, action in grid_configs():
        act = "只禁进" if action == "entry_only" else "禁进+强退"
        add(f"| {calendar_name} / slack{slack} / {act} | "
            f"{report['filter_stats']['ft'][name]['blocked_share_of_pool']:.6f} | "
            f"{report['filter_stats']['zs'][name]['blocked_share_of_pool']:.6f} |")
    add("")
    add("### E3 各格的名额缺口与强退次数")
    add("")
    add("| 格 | FT 名额缺口 | FT 强退 | FT 同日进出 | ZS 名额缺口 | ZS 强退 | ZS 同日进出 |")
    add("|---|---|---|---|---|---|---|")
    for name, calendar_name, slack, action in grid_configs():
        act = "只禁进" if action == "entry_only" else "禁进+强退"
        f_ = report["filter_stats"]["ft"][name]
        z_ = report["filter_stats"]["zs"][name]
        add(f"| {calendar_name} / slack{slack} / {act} | {f_['slot_shortfall']} | "
            f"{f_['force_exited']} | {f_['reentered_after_exit_same_day']} | "
            f"{z_['slot_shortfall']} | {z_['force_exited']} | "
            f"{z_['reentered_after_exit_same_day']} |")
    add("")
    return "\n".join(lines) + "\n"


def _write_outputs(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    (out_dir / "report_tables.md").write_text(render_tables(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="exp14 Rule E 开发折估计")
    parser.add_argument("--processed", type=Path, default=PROCESSED)
    parser.add_argument("--derived", type=Path, default=DERIVED)
    parser.add_argument("--outputs-root", type=Path, default=REPO / "outputs")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--memory-limit-gb", type=float, default=MEMORY_LIMIT_GB_DEFAULT)
    parser.add_argument("--folds", type=int, nargs="+", default=list(FOLDS),
                        help="只供试跑；正式读数必须是全部七折 36–42（白名单仍生效）")
    parser.add_argument("--smoke", action="store_true",
                        help="用合成数据跑通全链路（不碰任何真实数据）")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.smoke:
        from exp14_smoke import run_smoke  # noqa: PLC0415  仅冒烟路径导入
        return run_smoke(args.out_dir)
    report = run(args.processed.resolve(), args.derived.resolve(),
                 args.outputs_root.resolve(), folds=args.folds,
                 memory_limit_gb=args.memory_limit_gb)
    _write_outputs(report, args.out_dir)
    check = report["E0_baseline_check"]
    log(f"[exp14] E0 一致性：{check['status']}（{check.get('n_fields_checked', 0)} 字段，"
        f"{len(check.get('mismatches', []))} 项不一致）")
    for arm in ARMS:
        base = report["readouts"][arm][BASELINE_CONFIG]["net_by_cost_bp"]["8"]
        main_cfg = report["readouts"][arm][MAIN_CONFIG]["net_by_cost_bp"]["8"]
        p = report["paired_vs_E0"][arm][MAIN_CONFIG]
        log(f"[{arm.upper()}] 净8bp年化 {base['net_annual_pct']:.4f}% -> "
            f"{main_cfg['net_annual_pct']:.4f}%；夏普 {base['sharpe']:.4f} -> "
            f"{main_cfg['sharpe']:.4f}；方差比 {p['var_ratio_pooled']:.4f}；"
            f"配对差 {p['mean_diff_bp_per_day']:+.4f} bp/日 "
            f"CI [{p['ci95_bp'][0]:+.4f}, {p['ci95_bp'][1]:+.4f}]，"
            f"{p['folds_positive']}/{p['folds_total']} 折为正")
    log(f"[exp14] wrote {args.out_dir / 'summary.json'}")
    if check["status"] == "partial_run_not_comparable":
        log("[exp14] 试跑（非全折），未做 E0 逐位核对；正式读数必须跑满折 36–42。")
        return 0
    if check["status"] != "identical":
        log("[exp14] **E0 与 nt5_baseline_readout.json 不一致 —— 按任务书 §4.1 停下报告**")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
