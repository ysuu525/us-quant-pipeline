"""exp15：公告者倾斜（选项 B）在冻结 NT=5 构造上的开发折估计。

任务书 `docs/任务书_exp15_公告者倾斜_开发折估计_2026-09-05.md`
（冻结哈希 sha256 6f961db688861e9bc4a7914e0305da12d67771542293bb2e2f1516e6d8564484）。
下面 §2–§5 是任务书原文全文抄录，先于读第一行数据落笔（CLAUDE.md §二）。

================================================================================
## 2. 硬禁令

1. 只读折 36–42 两臂 `scores.parquet`（`scores_path` + `assert_readable` + 折号白名单断言）。
   绝不读折 05–35、fold44–45、`*sealed*`。
2. **不改** `src/portfolio/construction.py`、`construction_rule_e.py`、
   `scripts/nt5_baseline_readout.py`、`compare_arms_money.py`、`exp13_*`、`exp14_*`、
   v4、ledger、HANDOFF。新构造写新模块 `src/portfolio/construction_announcer.py`；
   **倾斜为零时须逐位重现冻结输出**（测试断言）。
3. 全脚本无 `label`；附 `grep -n label` 原始输出。
4. 不 `git add`、不 `git commit`、不追加登记簿。
5. 复用 exp13 派生 parquet 与 exp14 的 `rdq_hat` / 日历工具（import 优先）；不碰 CSV；
   峰值提交内存增量 < 6 GB；无 GPU。
6. 不得编数字；没核到写「未核」。

## 3. 预期公告日的对齐修订（**修订发生在看到 exp14 E2 之后**，理由是方法学的，须原样披露）

- exp14 E2：`rdq_hat = 去年同季 rdq + 365 天` 有 14.97% 落在非交易日、err 众数 −1，
  机制是日历年平移星期几。
- **本任务改为 `rdq_hat' = 去年同季 rdq + 364 天`（52 周，同星期几）；
  落在非交易日则顺延到下一交易日。** 零自由参数；选择依据是日期准确率而非任何收益量。
- 报告 `+365` 与 `+364` 两种规则的 |err| ≤ 0/1/2/3 累计占比与非交易日占比并列（同一样本），
  **只披露，不再改**。
- 时点：t 日只用 ≤ t 已知的公告日（沿 exp14）。

## 4. 设计（先写死；两个变体 × 两种日历）

### 4.1 B1 公告者中性（零自由参数）
- 定义：每个套袖重平衡日 t，池内「公告者」= `rdq_hat'` ±1 日历日落在 [t+1, t+6] 交易日的名字；
  池内公告者占比 `s_t`。
- 进场配额 `k_E = round(k × s_t)`；进场集合 = Kronos 秩最高的 `k_E` 个公告者 +
  秩最高的 `k − k_E` 个非公告者（各组内按冻结 `order`）。**入选数量不变。**
- 退出规则、权重、时点、成本全部沿冻结构造。`s_t = 0` 或全池无公告者时逐位等于冻结输出。
- 含义：把 Kronos 簿的公告者暴露从「系统性偏低」拉回池子的自然比例，**不超配**。

### 4.2 B2 事件套袖（用户的「单独处理」）
- 定义：名字 i 的事件日 `e_i = rdq_hat'`（交易日）。**持有窗 = [e_i − 1, e_i + 1] 三个交易日**：
  在 `e_i − 1` 开盘买入（该日只吃 open→close，`CLAUDE.md` §一.4），`e_i + 1` 收盘卖出。
  信号日 = `e_i − 2`（t 日收盘决定、t+1 开盘成交）。
- 套袖内等权；每日套袖收益 = 当日在持有窗内的名字的等权收益；成本 8bp 单边，进出各一次；
  当日无名字则为 0（现金）。
- **资金占比 `w_E = 10%` 先验固定**；组合 = `0.9 × E0 主动收益 + 0.1 × 套袖主动收益`，
  主动收益均为「− 同池等权基准」口径（与冻结读数一致）。敏感性 `w_E ∈ {5%, 20%}` 只报不选。
- **不设止损、不看 Kronos 分数、不看公告后走势**——「不傻拿」由固定 3 日窗体现。
- 候选池 = 当日 ADV 前 500（同冻结池）。

### 4.3 日历
每个变体各跑两种：`rdq_hat'`（t 日可知，现实下界）与 `rdq_real`（真实公告日，**事后信息**，
完美前瞻日历上界）。共 B1×2 + B2×2 + B2 敏感性 2×2 = 8 格 + E0。

### 4.4 诊断 D：财报后漂移 / 回撤（描述性，允许用事后 rdq）
- 样本：折 36–42 验证窗内、top500 池内全部真实公告事件。
- 对每个事件：事件日收益 `r_0`（含前一日盘后：取 [e−1 close → e close]，
  若 rdq 无时间信息则如此近似并注明）；随后累计收益 `CR(+1..+h)`，h = 1, 3, 5 交易日，
  相对同池等权基准。
- 分组：按 `r_0` 符号，及 `r_0` 截面五分位（当日）。报每组 `CR(+h)` 的逐日截面均值时间序列的
  均值 + NW(5) 95% CI + 7 折方向。
- 另报套袖视角：[e−1 open → e+1 close] 主动收益的均值 + CI + 7 折方向（= B2 每笔的毛收益）。
- **读法限定**：只报方向与量级；不得据此加止损或改窗口。

## 5. 判据（先写死）
- 无门槛、无 PASS/FAIL、无「有效 / 更好 / 建议」措辞；每个量报点估计 + NW(5) 95% CI + 7 折方向。
- MDE 未算（不是检验）；报 exp14 同法外推的 31 折 SE 供披露。
- 变体之间**不得**按收益 / 夏普择优；`rdq_real` 格只作上界。
- 措辞模板：「在开发折 36–42（已消耗，方向性证据）上，B2（`rdq_hat'`，w_E=10%）使 FT
  净(8bp) 年化由 a 变为 b，净夏普 c → d，方差比 e，配对差 f bp/日（CI [·,·]，N/7 折正），
  套袖单笔毛主动收益均值 g bp（CI [·,·]）。不作判定。」
================================================================================

实现说明（不属于任务书，属交付时必须披露的实现选择）
----------------------------------------------------
* 折号白名单：本脚本只接受折 36–42，读任何文件之前先断言（:func:`assert_folds`）。
  分数路径走 ``signals.kronos_adapter.scores_path``（该函数自身也有 ``ALLOWED_FOLDS``
  守卫）；所有 parquet 读取前调用 ``crsp_pipeline.sealed.assert_readable``。
* 构造：``portfolio.construction_announcer.announcer_tilt_long_only_returns``（新模块），
  公告者集合为空时与 ``portfolio.construction.frozen_long_only_returns`` 逐位相同。
* 口径复用：``summarize`` / ``max_drawdown_and_underwater`` / ``COST_GRID_BP`` / ``NT``
  直接 import ``scripts/nt5_baseline_readout.py``（只 import，不改）；
  价格加载 import ``scripts/compare_arms_money.py::load_prices``；
  ``window_blocked`` / ``daily_pools`` / ``nearest_future_rdq_real`` / ``paired_stats`` /
  ``nw_mean_ci`` / ``net_series`` / ``committed_memory_gb`` / ``check_baseline`` 直接 import
  ``scripts/exp14_rule_e_dev_estimate.py``（只 import，不改）。
* ``rdq_hat'``（§3）：在 exp13 的季度表上把 ``rdq_hat`` 列换成
  ``rdq_source + 364 天``（``rdq_source`` = 同财季上一财年的实际公告日，exp13:344 生成），
  非交易日顺延到下一交易日，再交给 ``exp13.assign_point_in_time`` 做时点匹配——
  该函数自带前视闸门（``rdq_source >= signal_date`` 的格置空，exp13:423-426）。
* ``rdq_real``：``exp14.nearest_future_rdq_real``（B1 的旗标）与季度表的 ``rdq``（B2 的事件表），
  两者都按 §4.2「事件日是交易日」顺延到下一交易日。**这是事后信息**，只作上界。
* 无未来收益变量：本脚本代码区全文无 CLAUDE.md §一.1 点名的那个变量名；上面任务书
  §2.3 抄录里的那一处是**禁令原文**，有测试剥掉 docstring 后断言代码区 0 命中。
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

from crsp_pipeline.sealed import assert_readable            # noqa: E402
from portfolio.construction import scores_frame_to_by_day   # noqa: E402
from portfolio.construction_announcer import (              # noqa: E402
    AnnouncerFlags, announcer_tilt_long_only_returns,
    combine_active_returns, event_sleeve_returns,
)
from signals.kronos_adapter import scores_path              # noqa: E402

TASK_DOC = "docs/任务书_exp15_公告者倾斜_开发折估计_2026-09-05.md"
TASK_DOC_SHA256 = "6f961db688861e9bc4a7914e0305da12d67771542293bb2e2f1516e6d8564484"

PROCESSED = Path(r"F:\quant\processed\crsp_ciz_2026-08-24_20260825T130601Z")
DERIVED = Path(r"F:\quant\external\compustat\derived")
OUT_DIR = REPO / "outputs" / "exp15_announcer_tilt_dev_estimate"
BASELINE_JSON = REPO / "outputs" / "nt5_baseline_readout.json"

#: 折号白名单（任务书 §2.1；CLAUDE.md §四）。只有已消耗的现代开发折。
ALLOWED_FOLDS_EXP15: frozenset[int] = frozenset(range(36, 43))
FOLDS: tuple[int, ...] = tuple(range(36, 43))
ARMS: tuple[str, ...] = ("ft", "zs")

#: 冻结 NT=5 构造（与 nt5_baseline_readout.py:45-46 同）
TOPN, EXIT_PCT, MIN_NAMES = 500, 0.30, 50
NET_COST_BP = 8.0
NW_LAG = 5
Z95 = 1.959963984540054

#: §4.1 的固定项：公告者窗口 = [t+1, t+6] 交易日、±1 日历日 slack
HOLDING_DAYS = 6
SLACK_DAYS = 1
#: §3 的固定项：+364 天（52 周）、非交易日顺延；对照规则是 exp14 的 +365 天
HAT_OFFSET_DAYS = 364
EXP14_HAT_OFFSET_DAYS = 365
#: §4.2 的固定项：持有窗 [e−1, e+1]，资金占比 10%，敏感性 5% / 20% 只报不选
EVENT_WINDOW = (-1, +1)
W_EVENT_MAIN = 0.10
W_EVENT_SENSITIVITY: tuple[float, ...] = (0.05, 0.20)
#: §4.3 的两种日历
CALENDARS: tuple[str, ...] = ("rdq_hat364", "rdq_real")
EX_POST_CALENDARS: frozenset[str] = frozenset({"rdq_real"})
#: §4.4 诊断 D 的持有期
DRIFT_HORIZONS: tuple[int, ...] = (1, 3, 5)
#: §3 报告的 |err| 累计门槛
ERR_THRESHOLDS: tuple[int, ...] = (0, 1, 2, 3)

BASELINE_CONFIG = "E0"

#: 31 折外推（ledger.md:307/:430 的方法：SE31 = 开发折 NW(5) SE × √(n_dev/3900)）
FOLDS31_TRADING_DAYS = 3900

QUARTER_KEEP_LO = pd.Timestamp("2019-01-01")
QUARTER_KEEP_HI = pd.Timestamp("2025-12-31")

MEMORY_LIMIT_GB_DEFAULT = 56.0


class FoldWhitelistError(RuntimeError):
    """折号越出 exp15 白名单。"""


def assert_folds(folds: Iterable[int]) -> tuple[int, ...]:
    """读任何文件之前调用。任务书 §2.1。"""
    got = tuple(int(f) for f in folds)
    bad = sorted(set(got) - ALLOWED_FOLDS_EXP15)
    if bad:
        raise FoldWhitelistError(
            f"exp15 只允许折 {sorted(ALLOWED_FOLDS_EXP15)}；收到越界折号 {bad}。"
            "折 05–35 / fold44–45 / 封存窗的读取即全部作废（任务书 §2.1）。"
        )
    if not got:
        raise FoldWhitelistError("折号为空")
    return got


def log(message: str) -> None:
    print(message, flush=True)


def _load_script_module(name: str, path: Path):
    """照 exp13:153-160 / exp14:199-206；只 import，不改被引脚本。"""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# §3 的对齐修订：+364 天 + 非交易日顺延
# ---------------------------------------------------------------------------
def roll_to_trading_day(calendar: pd.DatetimeIndex, dates) -> np.ndarray:
    """把日期顺延到 **>= 自身的首个交易日**；已是交易日则不动，越界置 NaT。"""
    cal = calendar.to_numpy(dtype="datetime64[ns]")
    values = pd.to_datetime(pd.Series(dates)).to_numpy(dtype="datetime64[ns]")
    pos = np.searchsorted(cal, values, side="left").astype(np.int64)
    out = np.full(len(values), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    ok = (pos < len(cal)) & ~pd.isna(values)
    out[ok] = cal[pos[ok]]
    return out


def with_hat364(permno_quarters: pd.DataFrame,
                calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """在 exp13 的 permno-季度表上加 `rdq_hat364_raw` / `rdq_hat364`（顺延后）两列。

    §3 的规则：``rdq_hat' = rdq_source + 364 天``（52 周，同星期几），
    非交易日顺延到下一交易日。``rdq_source`` 是 exp13:344 生成的「同财季上一财年的
    实际公告日」，**只用过去的 rdq**，本函数不引入任何新的时点自由度。
    """
    frame = permno_quarters.copy()
    source = frame["rdq_source"]
    raw = source + pd.Timedelta(days=HAT_OFFSET_DAYS)
    frame["rdq_hat364_raw"] = pd.to_datetime(raw).astype("datetime64[ns]")
    frame["rdq_hat364"] = roll_to_trading_day(calendar, frame["rdq_hat364_raw"])
    return frame


def alignment_accuracy(events: pd.DataFrame,
                       calendar: pd.DatetimeIndex) -> dict[str, Any]:
    """§3 要求的两规则准确率对照（同一样本）。

    ``events`` 列：``rdq``（真实）、``rdq_hat``（+365，exp14 规则）、
    ``rdq_hat364_raw``（+364 未顺延）、``rdq_hat364``（+364 顺延后）。
    """
    trading = set(pd.DatetimeIndex(calendar))
    n = int(len(events))
    out: dict[str, Any] = {"n_events": n}
    rules = (("plus365_exp14", "rdq_hat"),
             ("plus364_raw", "rdq_hat364_raw"),
             ("plus364_rolled", "rdq_hat364"))
    for rule_name, column in rules:
        has = events[column].notna()
        err = (events.loc[has, "rdq"] - events.loc[has, column]).dt.days.astype("float64")
        predicted = events.loc[has, column]
        non_trading = float((~predicted.isin(trading)).mean()) if len(predicted) else float("nan")
        entry: dict[str, Any] = {
            "n_with_prediction": int(has.sum()),
            "missing_share": float(1.0 - has.mean()) if n else float("nan"),
            "non_trading_day_share": non_trading,
        }
        if len(err):
            entry.update({
                "err_mean_days": float(err.mean()),
                "err_median_days": float(err.median()),
                "err_mode_days": float(err.mode().iloc[0]),
                "abs_err_cum_share": {
                    str(threshold): float((err.abs() <= threshold).mean())
                    for threshold in ERR_THRESHOLDS},
                "abs_err_cum_share_of_all_events": {
                    str(threshold): float((err.abs() <= threshold).sum() / n)
                    for threshold in ERR_THRESHOLDS},
            })
        else:
            entry.update({"err_mean_days": float("nan"), "err_median_days": float("nan"),
                          "err_mode_days": float("nan"), "abs_err_cum_share": {},
                          "abs_err_cum_share_of_all_events": {}})
        out[rule_name] = entry
    # 真实公告日本身落在非交易日的占比（参照量）
    real = events["rdq"].dropna()
    out["rdq_real_non_trading_day_share"] = (
        float((~real.isin(trading)).mean()) if len(real) else float("nan"))
    return out


# ---------------------------------------------------------------------------
# 事件表（B2 与诊断 D）
# ---------------------------------------------------------------------------
def build_event_windows(events: pd.DataFrame, calendar: pd.DatetimeIndex,
                        *, event_col: str) -> pd.DataFrame:
    """给每个事件日补 ``d_entry`` / ``d_event`` / ``d_exit`` 三个交易日。

    ``d_event`` 必须已是交易日（调用方已顺延）；``d_entry`` = 前一交易日、
    ``d_exit`` = 后一交易日（§4.2 的 [e−1, e+1] 窗）。日历端点处置 NaT。
    """
    cal = calendar.to_numpy(dtype="datetime64[ns]")
    values = pd.to_datetime(events[event_col]).to_numpy(dtype="datetime64[ns]")
    pos = np.searchsorted(cal, values, side="left").astype(np.int64)
    exact = (pos < len(cal)) & ~pd.isna(values)
    exact[exact] &= cal[pos[exact]] == values[exact]
    ok = exact & (pos >= 1) & (pos + 1 < len(cal))
    out = events.copy()
    entry = np.full(len(values), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    exit_ = np.full(len(values), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    event = np.full(len(values), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    entry[ok] = cal[pos[ok] - 1]
    event[ok] = cal[pos[ok]]
    exit_[ok] = cal[pos[ok] + 1]
    out["d_entry"], out["d_event"], out["d_exit"] = entry, event, exit_
    return out.loc[ok].reset_index(drop=True)


def screen_events_to_pool(events: pd.DataFrame, pool_by_trade_date: dict,
                          signal_day_of_trade_date: dict,
                          *, require_source_before_signal: bool) -> tuple[pd.DataFrame, dict]:
    """把事件表筛到「进场日 `e−1` 的冻结池内」，并按 §3 做时点闸门。

    * 池：名字必须在 ``d_entry`` 这个成交日对应的冻结池里（该池由 ``d_entry`` 的
      前一个信号日收盘时的 lagged ADV20 决定，`t` 日可知）。
    * 时点（只对 ``rdq_hat'`` 日历）：推出事件日的那次实际公告 ``rdq_source``
      必须**早于**信号日 ``e−2``（= ``signal_day_of_trade_date[d_entry]``），
      与 exp13 §3.4 / exp14 偏离第 1 条同口径（严格 `<`）。
    """
    stats = {"n_in": int(len(events))}
    frame = events.copy()
    frame["in_index"] = frame["d_entry"].isin(set(pool_by_trade_date))
    frame = frame.loc[frame["in_index"]].drop(columns=["in_index"])
    stats["n_entry_day_in_fold_index"] = int(len(frame))
    if not len(frame):
        stats.update({"n_in_pool": 0, "n_after_pit": 0, "n_voided_by_pit": 0})
        return frame, stats
    keep = [permno in pool_by_trade_date[day]
            for permno, day in zip(frame["PERMNO"], frame["d_entry"])]
    frame = frame.loc[keep]
    stats["n_in_pool"] = int(len(frame))
    if require_source_before_signal:
        signal_days = pd.Series(
            [signal_day_of_trade_date[day] for day in frame["d_entry"]],
            index=frame.index, dtype="datetime64[ns]")
        frame = frame.assign(signal_day=signal_days)
        known = frame["rdq_source"].notna() & (frame["rdq_source"] < frame["signal_day"])
        stats["n_voided_by_pit"] = int((~known).sum())
        frame = frame.loc[known]
    else:
        stats["n_voided_by_pit"] = 0
    stats["n_after_pit"] = int(len(frame))
    return frame.reset_index(drop=True), stats


# ---------------------------------------------------------------------------
# 诊断 D
# ---------------------------------------------------------------------------
def drift_diagnostics(events: pd.DataFrame, ret_by_day: dict,
                      bench_by_trade_date: dict, pool_by_trade_date: dict,
                      calendar: pd.DatetimeIndex,
                      horizons: Iterable[int] = DRIFT_HORIZONS) -> pd.DataFrame:
    """逐事件的 `r_0` 与 `CR(+h)`（相对同池等权基准，复利口径）。

    ``r_0`` = 事件日 `e` 的 `DlyRet`，即 **[e−1 收盘 → e 收盘]**（任务书 §4.4；
    `rdq` 无时间信息，故只能如此近似——`rdq` 是否含盘后信息见 §7 未核项）。
    ``CR(+h)`` = ``prod_{d=e+1..e+h}(1+r_d) − prod(1+bench_d)``；任一天缺失即置缺。
    另附 ``r0_pool_pct``：`r_0` 在**当日池内全部名字的 `DlyRet` 截面**中的百分位。
    """
    cal = calendar.to_numpy(dtype="datetime64[ns]")
    horizons = tuple(horizons)
    max_h = max(horizons)
    rows: list[dict] = []
    pool_rank_cache: dict = {}
    for row in events.itertuples(index=False):
        permno, d_event = row.PERMNO, pd.Timestamp(row.d_event)
        rm = ret_by_day.get(d_event, {})
        r0 = rm.get(permno)
        if r0 is None or not np.isfinite(r0):
            continue
        record: dict[str, Any] = {"PERMNO": permno, "d_event": d_event,
                                  "fold": getattr(row, "fold", None), "r0": float(r0)}
        # 当日池内截面百分位（池 = 该成交日的冻结池）
        pool = pool_by_trade_date.get(d_event)
        if pool is not None:
            cached = pool_rank_cache.get(d_event)
            if cached is None:
                values = np.array([rm[p] for p in pool if p in rm and np.isfinite(rm[p])],
                                  dtype=float)
                values.sort()
                cached = values
                pool_rank_cache[d_event] = cached
            if len(cached):
                record["r0_pool_pct"] = float(
                    np.searchsorted(cached, float(r0), side="right") / len(cached))
        pos = int(np.searchsorted(cal, np.datetime64(d_event, "ns"), side="left"))
        cum_r, cum_b, ok = 1.0, 1.0, True
        for offset in range(1, max_h + 1):
            if not ok:
                break
            if pos + offset >= len(cal):
                ok = False
                break
            day = pd.Timestamp(cal[pos + offset])
            value = ret_by_day.get(day, {}).get(permno)
            if day not in bench_by_trade_date or value is None or not np.isfinite(value):
                ok = False
                break
            cum_r *= 1.0 + float(value)
            cum_b *= 1.0 + float(bench_by_trade_date[day])
            if offset in horizons:
                record[f"cr{offset}"] = cum_r - cum_b
        rows.append(record)
    return pd.DataFrame(rows)


def group_series_stats(frame: pd.DataFrame, value_col: str, exp14) -> dict[str, Any]:
    """逐日截面均值时间序列 → 均值 + NW(5) 95% CI + 逐折方向。"""
    valid = frame[["d_event", "fold", value_col]].dropna()
    if not len(valid):
        return {"n_events": 0, "n_days": 0, "mean_bp": float("nan"),
                "nw5_se_bp": float("nan"), "nw5_t": float("nan"),
                "ci95_bp": [float("nan"), float("nan")],
                "folds_positive": 0, "folds_total": 0, "per_fold_mean_bp": {}}
    daily = valid.groupby(["fold", "d_event"])[value_col].mean().reset_index()
    stat = exp14.nw_mean_ci(daily[value_col])
    per_fold = {str(f): float(g[value_col].mean() * 1e4)
                for f, g in daily.groupby("fold")}
    return {
        "n_events": int(len(valid)),
        "n_days": int(len(daily)),
        "mean_bp": float(stat["mean"] * 1e4),
        "nw5_se_bp": float(stat["se"] * 1e4),
        "nw5_t": float(stat["t"]),
        "ci95_bp": [float(stat["ci95_lo"] * 1e4), float(stat["ci95_hi"] * 1e4)],
        "folds_positive": int(sum(v > 0 for v in per_fold.values())),
        "folds_total": len(per_fold),
        "per_fold_mean_bp": per_fold,
    }


def quintile_bins(values: pd.Series) -> pd.Series:
    """当日截面五分位（1 = 最低）；不足 5 个观测的日子置缺。

    实现：稳定名次（并列按出现次序）→ 百分位 ``rank / n`` → ``ceil(pct * 5)``，
    裁到 [1, 5]。等价于等计数分箱，且不依赖 pandas 的分箱边界处理。
    """
    n = len(values)
    if n < 5:
        return pd.Series([np.nan] * n, index=values.index)
    pct = values.rank(method="first") / n
    return np.ceil(pct * 5).clip(1, 5).astype(float)


def extrapolate_se_31folds(se: float, n_dev_days: int) -> float:
    """31 折外推（`ledger.md:307`/:430 同法）：SE31 = 开发折 SE × √(n_dev / 3900)。

    **假设**：日方差与序列相关结构在 31 折上与开发折相同；只按天数缩放。
    """
    if not np.isfinite(se) or n_dev_days <= 0:
        return float("nan")
    return float(se * np.sqrt(n_dev_days / FOLDS31_TRADING_DAYS))


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run(processed: Path, derived: Path, outputs_root: Path, *,
        folds: Iterable[int] = FOLDS,
        memory_limit_gb: float = MEMORY_LIMIT_GB_DEFAULT) -> dict[str, Any]:
    use_folds = assert_folds(folds)

    exp13 = _load_script_module("exp15_exp13", REPO / "scripts" / "exp13_compustat_dev_diag.py")
    exp14 = _load_script_module("exp15_exp14", REPO / "scripts" / "exp14_rule_e_dev_estimate.py")
    nt5 = _load_script_module("exp15_nt5", REPO / "scripts" / "nt5_baseline_readout.py")
    cam = _load_script_module("exp15_cam", REPO / "scripts" / "compare_arms_money.py")
    if nt5.NT != 5:
        raise AssertionError("nt5_baseline_readout.NT 不是 5，口径已变")
    nt_value = int(nt5.NT)

    start_committed = exp14.committed_memory_gb()
    if start_committed > memory_limit_gb:
        raise RuntimeError(f"起始提交内存 {start_committed:.2f} GB > 上限 {memory_limit_gb} GB")
    log(f"[exp15] 起始 committed={start_committed:.2f} GB")
    peak_committed = start_committed

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
    permno_quarters = with_hat364(permno_quarters, calendar)
    log(f"[exp15] 季度特征 {len(quarters):,} 行 -> permno 层 {n_pq_full:,} 行 -> "
        f"日期过滤后 {len(permno_quarters):,} 行（含 rdq_hat364）")
    del quarters
    gc.collect()

    # §3 的 rdq_hat' 时点匹配用的季度表：把 rdq_hat 列换成 +364 顺延后的日期
    pq_hat364 = permno_quarters.copy()
    pq_hat364["rdq_hat"] = permno_quarters["rdq_hat364"]

    assert_readable(cam.P)
    log("[exp15] 加载价格/成交额（复用 compare_arms_money.load_prices，同口径）...")
    ret, oc, adv = cam.load_prices()
    peak_committed = max(peak_committed, exp14.committed_memory_gb())
    log(f"[exp15] 价格加载后 committed={exp14.committed_memory_gb():.2f} GB")

    config_names = [BASELINE_CONFIG]
    for calendar_name in CALENDARS:
        config_names.append(f"B1|{calendar_name}")
        for w in (W_EVENT_MAIN, *W_EVENT_SENSITIVITY):
            config_names.append(f"B2|{calendar_name}|w{int(round(w * 100))}")

    daily: dict[str, dict[str, list]] = {arm: {c: [] for c in config_names} for arm in ARMS}
    b1_stats: dict[str, dict[str, list]] = {
        arm: {f"B1|{c}": [] for c in CALENDARS} for arm in ARMS}
    sleeve_stats: dict[str, dict[str, list]] = {
        arm: {c: [] for c in CALENDARS} for arm in ARMS}
    per_trade_rows: dict[str, dict[str, list]] = {
        arm: {c: [] for c in CALENDARS} for arm in ARMS}
    event_screen: list[dict[str, Any]] = []
    align_events: list[pd.DataFrame] = []
    drift_rows: list[pd.DataFrame] = []
    arm_key_identical: dict[int, bool] = {}

    for fold in use_folds:
        log(f"[exp15] === fold{fold} ===")
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
        for arm in ARMS:
            if same_keys and arm == "zs":
                pit_by_arm["zs"] = pit_by_arm["ft"]
                continue
            matched, _stats = exp13.assign_point_in_time(keys_arm[arm], pq_hat364, calendar)
            pit = matched[["PERMNO", "signal_date", "rdq_hat"]].copy()
            pit = pit.rename(columns={"rdq_hat": "rdq_hat364"})
            real = exp14.nearest_future_rdq_real(keys_arm[arm], permno_quarters)
            pit = pit.merge(real.reset_index(), on=["PERMNO", "signal_date"],
                            how="left", validate="one_to_one")
            pit["rdq_real"] = roll_to_trading_day(calendar, pit["rdq_real_next"])
            pit_by_arm[arm] = pit
            del matched
            gc.collect()

        fold_pool_names: set = set()
        fold_ref: dict[str, Any] = {}
        for arm in ARMS:
            by_day = by_day_arm[arm]
            pit = pit_by_arm[arm]

            # ---- E0（零倾斜；须与冻结构造逐位一致）
            e0_df, e0_diag = announcer_tilt_long_only_returns(
                by_day, ret, oc, adv, announcers=None, topn=TOPN, cost_bp=0.0,
                exit_pct=EXIT_PCT, nt=nt_value, min_names=MIN_NAMES,
                collect_diagnostics=True)
            bench_by_date = e0_diag["bench_by_trade_date"]
            pool_by_date = e0_diag["pool_by_trade_date"]
            signal_day_of = e0_diag["signal_day_of_trade_date"]
            trade_dates = list(e0_df.index)
            e0_df = e0_df.copy()
            e0_df["fold"] = fold
            daily[arm][BASELINE_CONFIG].append(e0_df)
            for names in pool_by_date.values():
                fold_pool_names |= set(names)
            if arm == "ft":
                fold_ref = {"bench": bench_by_date, "pool": pool_by_date,
                            "signal_day": signal_day_of, "trade_dates": trade_dates}

            for calendar_name in CALENDARS:
                column = "rdq_hat364" if calendar_name == "rdq_hat364" else "rdq_real"
                # ---- B1：公告者旗标（±1 日历日落在 [t+1, t+6] 交易日）
                hit = exp14.window_blocked(calendar, pit["signal_date"], pit[column],
                                           slack_days=SLACK_DAYS,
                                           holding_days=HOLDING_DAYS)
                flags = AnnouncerFlags.from_frame(
                    pit.loc[hit, ["signal_date", "PERMNO"]],
                    meta={"calendar": calendar_name, "slack_days": SLACK_DAYS,
                          "holding_days": HOLDING_DAYS,
                          "ex_post_information": calendar_name in EX_POST_CALENDARS,
                          "flag_cells": int(hit.sum()), "cells": int(len(pit))})
                b1_df, b1_diag = announcer_tilt_long_only_returns(
                    by_day, ret, oc, adv, announcers=flags, topn=TOPN, cost_bp=0.0,
                    exit_pct=EXIT_PCT, nt=nt_value, min_names=MIN_NAMES,
                    collect_diagnostics=True)
                b1_df = b1_df.copy()
                b1_df["fold"] = fold
                daily[arm][f"B1|{calendar_name}"].append(b1_df)
                per_day = b1_diag["per_day"]
                b1_stats[arm][f"B1|{calendar_name}"].append({
                    "fold": fold,
                    "rebalance_days": int(len(per_day)),
                    "sum_pool": int(per_day["n_pool"].sum()),
                    "sum_announcers_in_pool": int(per_day["n_announcers_in_pool"].sum()),
                    "sum_k": int(per_day["k"].sum()),
                    "sum_k_E": int(per_day["k_E"].sum()),
                    "sum_slots": int(per_day["n_slots"].sum()),
                    "sum_added": int(per_day["n_added"].sum()),
                    "sum_added_announcers": int(per_day["n_added_announcers"].sum()),
                    "sum_added_announcers_frozen": int(
                        per_day["n_added_announcers_frozen"].sum()),
                    "sum_quota_promoted": int(per_day["n_quota_promoted"].sum()),
                    "sum_quota_displaced": int(per_day["n_quota_displaced"].sum()),
                    "days_quota_changed_entry": int((per_day["n_quota_promoted"] > 0).sum()),
                    "sum_keep_announcers": int(per_day["n_keep_announcers"].sum()),
                    "sum_book_announcers": int(per_day["n_book_announcers"].sum()),
                    "sum_announcer_shortfall": int(per_day["announcer_shortfall"].sum()),
                    "sum_slot_shortfall": int(per_day["n_slot_shortfall"].sum()),
                    "days_quota_binding": int((per_day["need_announcers"] > 0).sum()),
                    "flag_cells": int(flags.n_flag_cells),
                })
                # 同一旗标、关掉配额（apply_quota=False）：输出须逐位等于 E0，
                # 诊断给出「E0 的簿里有几个公告者」——B1「拉回自然比例」的对照量。
                e0_flag_df, e0_flag_diag = announcer_tilt_long_only_returns(
                    by_day, ret, oc, adv, announcers=flags, apply_quota=False,
                    topn=TOPN, cost_bp=0.0, exit_pct=EXIT_PCT, nt=nt_value,
                    min_names=MIN_NAMES, collect_diagnostics=True)
                if not (np.array_equal(e0_flag_df["r"].to_numpy(), e0_df["r"].to_numpy())
                        and np.array_equal(e0_flag_df["turn"].to_numpy(),
                                           e0_df["turn"].to_numpy())):
                    raise AssertionError(
                        f"{arm}/{calendar_name}: apply_quota=False 未逐位重现 E0")
                e0_per_day = e0_flag_diag["per_day"]
                b1_stats[arm][f"B1|{calendar_name}"][-1].update({
                    "sum_book_announcers_E0": int(e0_per_day["n_book_announcers"].sum()),
                    "sum_added_announcers_E0": int(e0_per_day["n_added_announcers"].sum()),
                    "sum_k_E0": int(e0_per_day["k"].sum()),
                    "sum_slots_E0": int(e0_per_day["n_slots"].sum()),
                })
                del e0_flag_df, e0_flag_diag, e0_per_day, b1_diag

                # ---- B2：事件套袖
                if calendar_name == "rdq_hat364":
                    raw = permno_quarters.loc[
                        permno_quarters["rdq_hat364"].notna(),
                        ["PERMNO", "rdq_hat364", "rdq_source"]].rename(
                            columns={"rdq_hat364": "event_date"})
                    require_pit = True
                else:
                    raw = permno_quarters.loc[
                        permno_quarters["rdq"].notna(), ["PERMNO", "rdq"]].copy()
                    raw["event_date"] = roll_to_trading_day(calendar, raw["rdq"])
                    raw["rdq_source"] = pd.NaT
                    raw = raw[["PERMNO", "event_date", "rdq_source", "rdq"]]
                    require_pit = False
                raw = raw.dropna(subset=["event_date"])
                n_before_dedup = int(len(raw))
                raw = raw.drop_duplicates(["PERMNO", "event_date"], keep="first")
                windows = build_event_windows(raw, calendar, event_col="event_date")
                screened, screen = screen_events_to_pool(
                    windows, pool_by_date, signal_day_of,
                    require_source_before_signal=require_pit)
                screen.update({"fold": fold, "arm": arm, "calendar": calendar_name,
                               "n_raw": n_before_dedup,
                               "n_dedup_dropped": n_before_dedup - len(raw)})
                sleeve_daily, per_trade = event_sleeve_returns(
                    screened, ret, oc, bench_by_trade_date=bench_by_date,
                    trade_dates=trade_dates, cost_bp=NET_COST_BP)
                screen["truncated_legs"] = int(sleeve_daily.attrs["truncated_legs"])
                screen["dropped_events"] = int(sleeve_daily.attrs["dropped_events"])
                screen["n_trades"] = int(len(per_trade))
                event_screen.append(screen)
                sleeve_stats[arm][calendar_name].append({
                    "fold": fold,
                    "n_days": int(len(sleeve_daily)),
                    "n_empty_days": int((sleeve_daily["n_names"] == 0).sum()),
                    "mean_names": float(sleeve_daily["n_names"].mean()),
                    "median_names": float(sleeve_daily["n_names"].median()),
                    "max_names": int(sleeve_daily["n_names"].max()),
                    "n_trades": int(len(per_trade)),
                    "mean_cost_coef": float(sleeve_daily["cost_coef"].mean()),
                    "sleeve_gross_active_annual_pct": float(
                        (sleeve_daily["gross"] - sleeve_daily["bench"]).mean() * 252 * 100),
                    "sleeve_net_active_annual_pct": float(
                        sleeve_daily["active"].mean() * 252 * 100),
                    "sleeve_net_active_vol_annual_pct": float(
                        sleeve_daily["active"].std(ddof=1) * np.sqrt(252) * 100),
                })
                if len(per_trade):
                    pt = per_trade.copy()
                    pt["fold"] = fold
                    per_trade_rows[arm][calendar_name].append(pt)

                for w in (W_EVENT_MAIN, *W_EVENT_SENSITIVITY):
                    combo = combine_active_returns(
                        e0_df["r"], e0_df["turn"], sleeve_daily, w_event=w, nt=nt_value)
                    frame = pd.DataFrame({"r": combo["gross"], "turn": combo["turn"]})
                    frame["n_names"] = e0_df["n_names"].to_numpy()
                    frame["fold"] = fold
                    # 自检：summarize 的 8bp 折算必须等于直算的组合净收益
                    direct = ((1.0 - w) * (e0_df["r"] - NET_COST_BP / 1e4 * 2.0
                                           * e0_df["turn"] / nt_value)
                              + w * sleeve_daily["active"])
                    derived = frame["r"] - NET_COST_BP / 1e4 * 2.0 * frame["turn"] / nt_value
                    if not np.allclose(direct.to_numpy(), derived.to_numpy(), atol=1e-15):
                        raise AssertionError(
                            f"{arm}/{calendar_name}/w{w}: 等效换手折算与直算净收益不一致")
                    daily[arm][f"B2|{calendar_name}|w{int(round(w * 100))}"].append(frame)

                del sleeve_daily, per_trade, screened, windows, raw
                gc.collect()

            del e0_diag, e0_df
            gc.collect()

        # ---- §3 对齐样本 + 诊断 D（每折算一次；两臂键相同 ⇒ 池与基准相同，用 ft 的）
        fold_lo, fold_hi = min(fold_ref["trade_dates"]), max(fold_ref["trade_dates"])
        events = permno_quarters.loc[
            permno_quarters["rdq"].notna()
            & permno_quarters["rdq"].between(fold_lo, fold_hi)
            & permno_quarters["PERMNO"].isin(fold_pool_names),
            ["PERMNO", "fidx", "rdq", "rdq_hat", "rdq_hat364_raw",
             "rdq_hat364"]].drop_duplicates(["PERMNO", "fidx"])
        align_events.append(events.assign(fold=fold))
        log(f"  fold{fold}: 池内名字 {len(fold_pool_names):,}，窗内真实公告 {len(events):,}")

        d_raw = permno_quarters.loc[
            permno_quarters["rdq"].notna(), ["PERMNO", "rdq"]].copy()
        d_raw["event_date"] = roll_to_trading_day(calendar, d_raw["rdq"])
        d_raw["rdq_source"] = pd.NaT
        d_raw = d_raw.dropna(subset=["event_date"]).drop_duplicates(
            ["PERMNO", "event_date"], keep="first")
        d_windows = build_event_windows(d_raw, calendar, event_col="event_date")
        d_screened, _d_screen = screen_events_to_pool(
            d_windows, fold_ref["pool"], fold_ref["signal_day"],
            require_source_before_signal=False)
        d_screened = d_screened.assign(fold=fold)
        drift = drift_diagnostics(d_screened, ret, fold_ref["bench"],
                                  fold_ref["pool"], calendar)
        if len(drift):
            drift_rows.append(drift)
        del d_raw, d_windows, d_screened, drift, events

        del by_day_arm, keys_arm, pit_by_arm, fold_ref
        gc.collect()
        peak_committed = max(peak_committed, exp14.committed_memory_gb())

    # ---------------- 汇总 ----------------
    log("[exp15] 汇总读数 ...")
    readouts: dict[str, dict[str, Any]] = {arm: {} for arm in ARMS}
    net_by_config: dict[str, dict[str, pd.DataFrame]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for name in config_names:
            allf = pd.concat(daily[arm][name]).sort_index()
            readouts[arm][name] = nt5.summarize(allf["r"], allf["turn"], allf["fold"])
            net_by_config[arm][name] = pd.DataFrame({
                "net": exp14.net_series(allf, NET_COST_BP, nt_value),
                "fold": allf["fold"].to_numpy(),
            })

    paired: dict[str, dict[str, Any]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        base = net_by_config[arm][BASELINE_CONFIG]
        n_dev_days = int(len(base))
        for name in config_names:
            if name == BASELINE_CONFIG:
                continue
            cfg = net_by_config[arm][name]
            joined = cfg.join(base["net"].rename("net_base"), how="inner")
            if len(joined) != len(cfg) or len(joined) != len(base):
                raise AssertionError(f"{arm}/{name}: 配对日集合与 E0 不一致")
            stat = exp14.paired_stats(joined["net"], joined["net_base"], joined["fold"])
            stat["sharpe_diff"] = (
                readouts[arm][name]["net_by_cost_bp"]["8"]["sharpe"]
                - readouts[arm][BASELINE_CONFIG]["net_by_cost_bp"]["8"]["sharpe"])
            stat["nw5_se_bp_31folds_extrapolated"] = extrapolate_se_31folds(
                stat["nw5_se_bp"], n_dev_days)
            paired[arm][name] = stat

    baseline_check = exp14.check_baseline(
        {arm: {exp14.BASELINE_CONFIG: readouts[arm][BASELINE_CONFIG]} for arm in ARMS},
        BASELINE_JSON, use_folds)

    # ---------------- §3 对齐修订 ----------------
    events_all = pd.concat(align_events, ignore_index=True)
    alignment = {
        "definition": {
            "sample": "折 36–42 各自成交日区间内、top500 候选池名字的全部真实 rdq",
            "plus365_exp14": "exp13/exp14 的 rdq_hat = 上一财年同季 rdq + DateOffset(years=1)",
            "plus364_raw": "本任务 §3 的 rdq_source + 364 天，未顺延",
            "plus364_rolled": "上一行顺延到 >= 自身的首个交易日（§3 的正式规则）",
            "err": "rdq − 预测日，日历日",
        },
        "pooled": alignment_accuracy(events_all, calendar),
        "per_fold": {str(fold): alignment_accuracy(group, calendar)
                     for fold, group in events_all.groupby("fold")},
    }

    # ---------------- B2 逐笔与套袖统计 ----------------
    sleeve_summary: dict[str, dict[str, Any]] = {arm: {} for arm in ARMS}
    per_trade_summary: dict[str, dict[str, Any]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for calendar_name in CALENDARS:
            frame = pd.DataFrame(sleeve_stats[arm][calendar_name])
            sleeve_summary[arm][calendar_name] = {
                "n_days": int(frame["n_days"].sum()),
                "n_empty_days": int(frame["n_empty_days"].sum()),
                "empty_day_share": float(frame["n_empty_days"].sum() / frame["n_days"].sum()),
                "mean_names_per_day": float(
                    np.average(frame["mean_names"], weights=frame["n_days"])),
                "max_names_per_day": int(frame["max_names"].max()),
                "n_trades": int(frame["n_trades"].sum()),
                "mean_cost_coef": float(
                    np.average(frame["mean_cost_coef"], weights=frame["n_days"])),
                "per_fold": frame.to_dict(orient="records"),
            }
            parts = per_trade_rows[arm][calendar_name]
            if parts:
                pt = pd.concat(parts, ignore_index=True)
                stats = group_series_stats(
                    pt.rename(columns={"gross_active": "value"}), "value", exp14)
                stats["n_trades"] = int(len(pt))
                stats["mean_holding_days"] = float(pt["n_days"].mean())
                stats["nw5_se_bp_31folds_extrapolated"] = extrapolate_se_31folds(
                    stats["nw5_se_bp"], stats["n_days"])
                per_trade_summary[arm][calendar_name] = stats
            else:
                per_trade_summary[arm][calendar_name] = {"n_trades": 0}

    # ---------------- 诊断 D ----------------
    drift_all = pd.concat(drift_rows, ignore_index=True) if drift_rows else pd.DataFrame()
    diagnostic_d = build_diagnostic_d(drift_all, exp14)
    diagnostic_d["definition"] = {
        "sample": "折 36–42 成交日区间内、进场日 e−1 落在冻结 top500 池内的全部真实公告事件"
                  "（与 B2 rdq_real 格同一筛选，允许事后 rdq；任务书 §4.4）",
        "r0": "事件日 e 的 DlyRet，即 [e−1 收盘 → e 收盘]（rdq 无时间信息，近似）",
        "cr_h": "prod_{d=e+1..e+h}(1+r_d) − prod_{d=e+1..e+h}(1+bench_d)，复利口径",
        "quintile_event": "当日全部事件的 r_0 五分位（1 = 最低）",
        "quintile_pool": "r_0 在当日冻结池全部名字 DlyRet 截面中的百分位，再切五分位",
        "reading_limit": "只报方向与量级；不得据此加止损或改窗口（任务书 §4.4）",
    }

    end_committed = exp14.committed_memory_gb()
    peak_committed = max(peak_committed, end_committed)
    return {
        "meta": {
            "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "task_doc": TASK_DOC,
            "task_doc_sha256": TASK_DOC_SHA256,
            "folds": list(use_folds),
            "fold_whitelist": sorted(ALLOWED_FOLDS_EXP15),
            "arms": list(ARMS),
            "construction": {"NT": nt_value, "TOPN": TOPN, "EXIT_PCT": EXIT_PCT,
                             "MIN_NAMES": MIN_NAMES, "entry_pct": 0.10,
                             "execution": "t close score -> t+1 open, fresh names eat open->close"},
            "b1": {"holding_days": HOLDING_DAYS, "slack_days": SLACK_DAYS,
                   "quota": "k_E = round(k * s_t)，簿层配额；只经进场实现，从不强制退出",
                   "rounding": "Python 内建 round（银行家舍入）"},
            "b2": {"window": list(EVENT_WINDOW), "w_event_main": W_EVENT_MAIN,
                   "w_event_sensitivity": list(W_EVENT_SENSITIVITY),
                   "cost_bp_one_way": NET_COST_BP,
                   "empty_day": "套袖毛收益 0（现金），主动收益 = 0 − 基准"},
            "calendars": list(CALENDARS),
            "ex_post_calendars": sorted(EX_POST_CALENDARS),
            "hat_offset_days": HAT_OFFSET_DAYS,
            "exp14_hat_offset_days": EXP14_HAT_OFFSET_DAYS,
            "net_cost_bp": NET_COST_BP,
            "nw_lag": NW_LAG,
            "folds31_trading_days": FOLDS31_TRADING_DAYS,
            "quarter_stats": quarter_stats,
            "permno_quarter_rows_full": n_pq_full,
            "permno_quarter_rows_kept": int(len(permno_quarters)),
            "arm_score_keys_identical_by_fold": arm_key_identical,
            "committed_memory_gb": {"start": start_committed, "peak": peak_committed,
                                    "end": end_committed,
                                    "delta_peak": peak_committed - start_committed},
            "use_restriction": (
                "开发折 36–42 已消耗，读数为方向性证据；估计交付、无门槛、不判定。"
                "变体之间不得按收益 / 夏普择优；rdq_real 格只作上界（任务书 §5）。"
            ),
        },
        "config_names": config_names,
        "E0_baseline_check": baseline_check,
        "readouts": readouts,
        "paired_vs_E0": paired,
        "alignment_revision": alignment,
        "b1_stats": {arm: {name: _reduce_b1(rows) for name, rows in per_arm.items()}
                     for arm, per_arm in b1_stats.items()},
        "b2_sleeve": sleeve_summary,
        "b2_per_trade": per_trade_summary,
        "b2_event_screen": event_screen,
        "diagnostic_D": diagnostic_d,
    }


def _reduce_b1(rows: list[dict]) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    slots = int(frame["sum_slots"].sum())
    pool = int(frame["sum_pool"].sum())
    return {
        "rebalance_days": int(frame["rebalance_days"].sum()),
        "announcer_share_of_pool": float(frame["sum_announcers_in_pool"].sum() / pool)
        if pool else float("nan"),
        "book_announcer_share": float(frame["sum_book_announcers"].sum()
                                      / frame["sum_k"].sum()) if frame["sum_k"].sum() else float("nan"),
        "quota_share_of_k": float(frame["sum_k_E"].sum() / frame["sum_k"].sum())
        if frame["sum_k"].sum() else float("nan"),
        "entry_slots": slots,
        "entries_filled": int(frame["sum_added"].sum()),
        "entries_that_are_announcers": int(frame["sum_added_announcers"].sum()),
        "announcer_entry_share_of_slots": float(frame["sum_added_announcers"].sum() / slots)
        if slots else float("nan"),
        "kept_announcers": int(frame["sum_keep_announcers"].sum()),
        # 同日反事实：配额换掉了多少个进场名字
        "entries_that_are_announcers_frozen": int(frame["sum_added_announcers_frozen"].sum()),
        "quota_promoted_names": int(frame["sum_quota_promoted"].sum()),
        "quota_displaced_names": int(frame["sum_quota_displaced"].sum()),
        "quota_promoted_share_of_slots": float(frame["sum_quota_promoted"].sum() / slots)
        if slots else float("nan"),
        "days_quota_changed_entry": int(frame["days_quota_changed_entry"].sum()),
        "days_quota_binding": int(frame["days_quota_binding"].sum()),
        "announcer_shortfall": int(frame["sum_announcer_shortfall"].sum()),
        "slot_shortfall": int(frame["sum_slot_shortfall"].sum()),
        "flag_cells": int(frame["flag_cells"].sum()),
        # 同旗标、关配额（= E0 的簿）的对照量
        "sum_book_announcers_E0": int(frame["sum_book_announcers_E0"].sum()),
        "sum_added_announcers_E0": int(frame["sum_added_announcers_E0"].sum()),
        "sum_k_E0": int(frame["sum_k_E0"].sum()),
        "sum_slots_E0": int(frame["sum_slots_E0"].sum()),
        "book_announcer_share_E0": float(frame["sum_book_announcers_E0"].sum()
                                         / frame["sum_k_E0"].sum())
        if frame["sum_k_E0"].sum() else float("nan"),
        "per_fold": frame.to_dict(orient="records"),
    }


def build_diagnostic_d(drift: pd.DataFrame, exp14) -> dict[str, Any]:
    """诊断 D 的分组读数（描述性；不得据此加止损或改窗口）。"""
    if not len(drift):
        return {"n_events": 0}
    frame = drift.copy()
    frame["quintile_event"] = (frame.groupby("d_event")["r0"]
                               .transform(quintile_bins))
    if "r0_pool_pct" in frame.columns:
        frame["quintile_pool"] = np.ceil(
            frame["r0_pool_pct"].clip(lower=1e-12) * 5).clip(1, 5)
    else:
        frame["quintile_pool"] = np.nan
    out: dict[str, Any] = {
        "n_events": int(len(frame)),
        "n_days": int(frame["d_event"].nunique()),
        "r0_mean_bp": float(frame["r0"].mean() * 1e4),
        "r0_std_bp": float(frame["r0"].std(ddof=1) * 1e4),
        "r0_positive_share": float((frame["r0"] > 0).mean()),
        "r0_zero_count": int((frame["r0"] == 0).sum()),
        "by_horizon_all": {}, "by_sign": {}, "by_quintile_event": {},
        "by_quintile_pool": {},
    }
    out["r0_itself"] = group_series_stats(frame.rename(columns={"r0": "value"}),
                                          "value", exp14)
    for h in DRIFT_HORIZONS:
        column = f"cr{h}"
        if column not in frame.columns:
            continue
        out["by_horizon_all"][str(h)] = group_series_stats(
            frame.rename(columns={column: "value"}), "value", exp14)
        out["by_sign"][str(h)] = {}
        for name, mask in (("r0_positive", frame["r0"] > 0),
                           ("r0_negative", frame["r0"] < 0)):
            out["by_sign"][str(h)][name] = group_series_stats(
                frame.loc[mask].rename(columns={column: "value"}), "value", exp14)
        for key, group_col in (("by_quintile_event", "quintile_event"),
                               ("by_quintile_pool", "quintile_pool")):
            out[key][str(h)] = {}
            for q in (1, 2, 3, 4, 5):
                mask = frame[group_col] == q
                out[key][str(h)][str(q)] = group_series_stats(
                    frame.loc[mask].rename(columns={column: "value"}), "value", exp14)
    return out


# ---------------------------------------------------------------------------
# 报告表格（叙述部分由 report.md 手写；本函数只产机器可核对的表）
# ---------------------------------------------------------------------------
def _b1_name(calendar_name: str) -> str:
    return f"B1|{calendar_name}"


def _b2_name(calendar_name: str, w: float) -> str:
    return f"B2|{calendar_name}|w{int(round(w * 100))}"


def _cal_name(calendar_name: str) -> str:
    return "rdq_hat'（t 日可知）" if calendar_name == "rdq_hat364" else "rdq_real（事后）"


READOUT_ROWS = [
    ("毛年化 %", lambda s: f"{s['gross_annual_pct']:.4f}"),
    ("毛日收益 NW(5) t", lambda s: f"{s['gross_daily_nw_t']:.4f}"),
    ("每 bp 拖累 %/年", lambda s: f"{s['drag_per_bp_annual_pct']:.4f}"),
    ("BE 单边成本 bp", lambda s: f"{s['breakeven_oneway_bp']:.4f}"),
    ("年单边交易次数（等效）", lambda s: f"{s['oneway_trades_per_year']:.4f}"),
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


def _paired_cells(p: dict) -> str:
    return (f"{p['mean_diff_bp_per_day']:+.4f} | {p['nw5_se_bp']:.4f} | "
            f"{p['nw5_t']:+.3f} | [{p['ci95_bp'][0]:+.4f}, {p['ci95_bp'][1]:+.4f}] | "
            f"{p['nw5_se_bp_31folds_extrapolated']:.4f} | "
            f"{p['folds_positive']}/{p['folds_total']} | "
            f"{p['var_ratio_pooled']:.4f} | {p['sharpe_diff']:+.4f}")


PAIRED_HEADER = ("| 格 | 均值差 bp/日 | NW(5) SE bp | t | 95% CI bp | "
                 "31 折外推 SE bp | 7 折为正 | 方差比 vs E0 | 净夏普差 |")
PAIRED_RULE = "|---|---|---|---|---|---|---|---|---|"


def render_tables(report: dict[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append
    add("<!-- 由 scripts/exp15_announcer_tilt_dev_estimate.py 自动生成，勿手改 -->")
    add("")

    # ---- §3
    add("## §3 对齐修订：两规则准确率对照（同一样本）")
    add("")
    align = report["alignment_revision"]
    pooled = align["pooled"]
    add(f"合并样本：折 36–42 成交日区间内、top500 池内名字的真实公告 **{pooled['n_events']:,}** 个。")
    add(f"真实 `rdq` 本身落在非交易日的占比 **{pooled['rdq_real_non_trading_day_share']:.6f}**。")
    add("")
    add("| 量 | +365（exp14 规则） | +364 未顺延 | +364 顺延（§3 正式规则） |")
    add("|---|---|---|---|")
    rules = ("plus365_exp14", "plus364_raw", "plus364_rolled")

    def align_row(caption: str, getter) -> None:
        add(f"| {caption} | " + " | ".join(getter(pooled[r]) for r in rules) + " |")

    align_row("有预测日的事件数", lambda s: f"{s['n_with_prediction']:,}")
    align_row("预测缺失占比", lambda s: f"{s['missing_share']:.6f}")
    align_row("**预测日落在非交易日的占比**", lambda s: f"**{s['non_trading_day_share']:.6f}**")
    align_row("err 均值（日历日）", lambda s: f"{s['err_mean_days']:+.4f}")
    align_row("err 中位数", lambda s: f"{s['err_median_days']:+.1f}")
    align_row("err 众数", lambda s: f"{s['err_mode_days']:+.1f}")
    for threshold in ERR_THRESHOLDS:
        align_row(f"\\|err\\| <= {threshold}（分母 = 有预测者）",
                  lambda s, k=str(threshold): f"{s['abs_err_cum_share'][k]:.4f}")
    for threshold in ERR_THRESHOLDS:
        align_row(f"\\|err\\| <= {threshold}（分母 = 全部公告）",
                  lambda s, k=str(threshold): f"{s['abs_err_cum_share_of_all_events'][k]:.4f}")
    add("")
    add("### 逐折 |err| <= 1 与非交易日占比")
    add("")
    add("| 折 | 事件数 | +365 \\|err\\|<=1 | +364raw \\|err\\|<=1 | +364roll \\|err\\|<=1 | "
        "+365 非交易日 | +364raw 非交易日 |")
    add("|---|---|---|---|---|---|---|")
    for fold in report["meta"]["folds"]:
        s = align["per_fold"].get(str(fold))
        if s is None:
            continue
        add(f"| {fold} | {s['n_events']:,} | "
            f"{s['plus365_exp14']['abs_err_cum_share']['1']:.4f} | "
            f"{s['plus364_raw']['abs_err_cum_share']['1']:.4f} | "
            f"{s['plus364_rolled']['abs_err_cum_share']['1']:.4f} | "
            f"{s['plus365_exp14']['non_trading_day_share']:.4f} | "
            f"{s['plus364_raw']['non_trading_day_share']:.4f} |")
    add("")

    # ---- E0
    add("## E0 一致性核对（对 outputs/nt5_baseline_readout.json）")
    add("")
    check = report["E0_baseline_check"]
    add(f"状态：`{check['status']}`；核对字段数 {check.get('n_fields_checked', 0)}；"
        f"不一致 {len(check.get('mismatches', []))} 项。")
    add("")
    add("| 臂 | 字段 | exp15 E0 | nt5_baseline_readout.json | 逐位相同 |")
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

    # ---- B1
    add("## B1 公告者中性（两日历 × 两臂）")
    add("")
    for arm in ARMS:
        add(f"### {arm.upper()} 臂全读数（沿 nt5_baseline_readout.summarize 口径）")
        add("")
        add("| 量 | E0 | B1 rdq_hat' | B1 rdq_real（事后） |")
        add("|---|---|---|---|")
        for caption, getter in READOUT_ROWS:
            cells = [getter(report["readouts"][arm][BASELINE_CONFIG])]
            for calendar_name in CALENDARS:
                cells.append(getter(report["readouts"][arm][_b1_name(calendar_name)]))
            add(f"| {caption} | " + " | ".join(cells) + " |")
        add("")
    add("### B1 配对量 vs E0（逐日净 8bp 主动收益差）")
    add("")
    for arm in ARMS:
        add(f"**{arm.upper()} 臂**")
        add("")
        add(PAIRED_HEADER)
        add(PAIRED_RULE)
        for calendar_name in CALENDARS:
            p = report["paired_vs_E0"][arm][_b1_name(calendar_name)]
            add(f"| B1 / {_cal_name(calendar_name)} | {_paired_cells(p)} |")
        add("")
    add("### B1 逐折毛年化 %")
    add("")
    add("| 折 | FT E0 | FT B1 hat' | FT B1 real | ZS E0 | ZS B1 hat' | ZS B1 real |")
    add("|---|---|---|---|---|---|---|")
    for fold in report["meta"]["folds"]:
        key = str(fold)
        cells = []
        for arm in ARMS:
            for name in (BASELINE_CONFIG, *[_b1_name(c) for c in CALENDARS]):
                value = report["readouts"][arm][name]["per_fold_gross_annual_pct"].get(key)
                cells.append("—" if value is None else f"{value:.4f}")
        add(f"| {fold} | " + " | ".join(cells) + " |")
    add("")
    add("### B1 配额统计")
    add("")
    add("| 量 | FT hat' | FT real | ZS hat' | ZS real |")
    add("|---|---|---|---|---|")
    b1_specs = [
        ("重平衡档-日数", lambda s: f"{s['rebalance_days']}"),
        ("池内公告者占比 s_t（股票-日加权）", lambda s: f"{s['announcer_share_of_pool']:.6f}"),
        ("配额 k_E / k（加权）", lambda s: f"{s['quota_share_of_k']:.6f}"),
        ("**B1 簿内公告者 / k**", lambda s: f"**{s['book_announcer_share']:.6f}**"),
        ("**E0 簿内公告者 / k**（同旗标，关配额）",
         lambda s: f"**{s['sum_book_announcers_E0'] / s['sum_k_E0']:.6f}**"),
        ("建仓名额总数", lambda s: f"{s['entry_slots']}"),
        ("进场名字中的公告者数", lambda s: f"{s['entries_that_are_announcers']}"),
        ("同日反事实：冻结进场里的公告者数",
         lambda s: f"{s['entries_that_are_announcers_frozen']}"),
        ("E0 路径进场名字中的公告者数", lambda s: f"{s['sum_added_announcers_E0']}"),
        ("进场公告者 / 建仓名额", lambda s: f"{s['announcer_entry_share_of_slots']:.6f}"),
        ("**配额换进来的名字数（同日反事实）**", lambda s: f"**{s['quota_promoted_names']}**"),
        ("**配额换掉的名字 / 建仓名额**",
         lambda s: f"**{s['quota_promoted_share_of_slots']:.6f}**"),
        ("进场名单被配额改动的档-日数", lambda s: f"{s['days_quota_changed_entry']}"),
        ("配额有约束力（need > 0）的档-日数", lambda s: f"{s['days_quota_binding']}"),
        ("公告者不足以补满配额的次数", lambda s: f"{s['announcer_shortfall']}"),
        ("名额缺口（结构上应为 0）", lambda s: f"{s['slot_shortfall']}"),
        ("旗标格数（公告者股票-日）", lambda s: f"{s['flag_cells']}"),
    ]
    for caption, getter in b1_specs:
        cells = []
        for arm in ARMS:
            for calendar_name in CALENDARS:
                cells.append(getter(report["b1_stats"][arm][_b1_name(calendar_name)]))
        add(f"| {caption} | " + " | ".join(cells) + " |")
    add("")

    # ---- B2
    add("## B2 事件套袖（两日历 × 两臂 + w_E 敏感性）")
    add("")
    for arm in ARMS:
        add(f"### {arm.upper()} 臂全读数")
        add("")
        header = ["E0"]
        configs = [BASELINE_CONFIG]
        for calendar_name in CALENDARS:
            for w in (W_EVENT_MAIN, *W_EVENT_SENSITIVITY):
                tag = "hat'" if calendar_name == "rdq_hat364" else "real"
                header.append(f"{tag} w{int(round(w * 100))}%")
                configs.append(_b2_name(calendar_name, w))
        add("| 量 | " + " | ".join(header) + " |")
        add("|---" * (1 + len(header)) + "|")
        for caption, getter in READOUT_ROWS:
            cells = [getter(report["readouts"][arm][c]) for c in configs]
            add(f"| {caption} | " + " | ".join(cells) + " |")
        add("")
    add("### B2 配对量 vs E0（逐日净 8bp 主动收益差）")
    add("")
    for arm in ARMS:
        add(f"**{arm.upper()} 臂**")
        add("")
        add(PAIRED_HEADER)
        add(PAIRED_RULE)
        for calendar_name in CALENDARS:
            for w in (W_EVENT_MAIN, *W_EVENT_SENSITIVITY):
                p = report["paired_vs_E0"][arm][_b2_name(calendar_name, w)]
                mark = "主规格" if w == W_EVENT_MAIN else "敏感性"
                add(f"| B2 / {_cal_name(calendar_name)} / w_E={int(round(w * 100))}% "
                    f"（{mark}） | {_paired_cells(p)} |")
        add("")
    add("### 套袖本身的统计")
    add("")
    add("| 量 | FT hat' | FT real | ZS hat' | ZS real |")
    add("|---|---|---|---|---|")
    sleeve_specs = [
        ("套袖交易日数", lambda s: f"{s['n_days']}"),
        ("**日均名字数**", lambda s: f"**{s['mean_names_per_day']:.4f}**"),
        ("单日最多名字数", lambda s: f"{s['max_names_per_day']}"),
        ("**空仓日占比**", lambda s: f"**{s['empty_day_share']:.6f}**"),
        ("笔数（进出各一次）", lambda s: f"{s['n_trades']}"),
        ("日均成本系数 (进+出)/名字数", lambda s: f"{s['mean_cost_coef']:.6f}"),
    ]
    for caption, getter in sleeve_specs:
        cells = []
        for arm in ARMS:
            for calendar_name in CALENDARS:
                cells.append(getter(report["b2_sleeve"][arm][calendar_name]))
        add(f"| {caption} | " + " | ".join(cells) + " |")
    add("")
    add("### 套袖单笔毛主动收益（[e−1 开盘 → e+1 收盘] − 同期同池等权基准）")
    add("")
    add("| 臂 / 日历 | 笔数 | 有效日数 | 均值 bp | NW(5) SE bp | t | 95% CI bp | "
        "31 折外推 SE bp | 7 折为正 |")
    add("|---|---|---|---|---|---|---|---|---|")
    for arm in ARMS:
        for calendar_name in CALENDARS:
            s = report["b2_per_trade"][arm][calendar_name]
            if not s.get("n_trades"):
                add(f"| {arm.upper()} / {_cal_name(calendar_name)} | 0 | — | — | — | — | — | — | — |")
                continue
            add(f"| {arm.upper()} / {_cal_name(calendar_name)} | {s['n_trades']:,} | "
                f"{s['n_days']} | {s['mean_bp']:+.4f} | {s['nw5_se_bp']:.4f} | "
                f"{s['nw5_t']:+.3f} | [{s['ci95_bp'][0]:+.4f}, {s['ci95_bp'][1]:+.4f}] | "
                f"{s['nw5_se_bp_31folds_extrapolated']:.4f} | "
                f"{s['folds_positive']}/{s['folds_total']} |")
    add("")
    add("### 事件表筛选（FT 臂七折合并；ZS 臂键集合相同）")
    add("")
    screen = pd.DataFrame(report["b2_event_screen"])
    add("| 日历 | 原始事件行 | 去重丢弃 | 进场日在折内 | 池内 | 时点闸门后 | "
        "时点闸门砍掉 | 成笔数 | 截断的持有日腿 |")
    add("|---|---|---|---|---|---|---|---|---|")
    for calendar_name in CALENDARS:
        sub = screen[(screen["calendar"] == calendar_name) & (screen["arm"] == "ft")]
        add(f"| {_cal_name(calendar_name)} | {int(sub['n_raw'].sum()):,} | "
            f"{int(sub['n_dedup_dropped'].sum()):,} | "
            f"{int(sub['n_entry_day_in_fold_index'].sum()):,} | "
            f"{int(sub['n_in_pool'].sum()):,} | {int(sub['n_after_pit'].sum()):,} | "
            f"{int(sub['n_voided_by_pit'].sum()):,} | {int(sub['n_trades'].sum()):,} | "
            f"{int(sub['truncated_legs'].sum()):,} |")
    add("")

    # ---- 诊断 D
    add("## 诊断 D：财报后漂移 / 回撤（描述性，允许事后 rdq）")
    add("")
    d = report["diagnostic_D"]
    add(f"样本：**{d['n_events']:,}** 个真实公告事件，落在 **{d['n_days']}** 个事件日；"
        f"`r_0` 均值 **{d['r0_mean_bp']:+.2f} bp**、截面标准差 **{d['r0_std_bp']:.2f} bp**、"
        f"为正占比 **{d['r0_positive_share']:.4f}**（恰为 0 的 {d['r0_zero_count']} 个）。")
    add("")
    add("| 量 | 均值 bp | NW(5) SE bp | t | 95% CI bp | 7 折为正 | 事件数 | 日数 |")
    add("|---|---|---|---|---|---|---|---|")

    def d_row(caption: str, s: dict) -> None:
        if not s.get("n_events"):
            add(f"| {caption} | — | — | — | — | — | 0 | 0 |")
            return
        add(f"| {caption} | {s['mean_bp']:+.4f} | {s['nw5_se_bp']:.4f} | "
            f"{s['nw5_t']:+.3f} | [{s['ci95_bp'][0]:+.4f}, {s['ci95_bp'][1]:+.4f}] | "
            f"{s['folds_positive']}/{s['folds_total']} | {s['n_events']:,} | {s['n_days']} |")

    d_row("`r_0` 本身（[e−1 收盘 → e 收盘]，未减基准）", d["r0_itself"])
    for h in DRIFT_HORIZONS:
        d_row(f"CR(+{h}) 全样本", d["by_horizon_all"][str(h)])
    add("")
    add("### 按 `r_0` 符号分组")
    add("")
    add("| 组 | h | 均值 bp | NW(5) SE bp | t | 95% CI bp | 7 折为正 | 事件数 |")
    add("|---|---|---|---|---|---|---|---|")
    for name, caption in (("r0_positive", "r_0 > 0"), ("r0_negative", "r_0 < 0")):
        for h in DRIFT_HORIZONS:
            s = d["by_sign"][str(h)][name]
            if not s.get("n_events"):
                add(f"| {caption} | +{h} | — | — | — | — | — | 0 |")
                continue
            add(f"| {caption} | +{h} | {s['mean_bp']:+.4f} | {s['nw5_se_bp']:.4f} | "
                f"{s['nw5_t']:+.3f} | [{s['ci95_bp'][0]:+.4f}, {s['ci95_bp'][1]:+.4f}] | "
                f"{s['folds_positive']}/{s['folds_total']} | {s['n_events']:,} |")
    add("")
    for key, caption in (("by_quintile_event", "当日**事件**截面五分位（1 = 最低）"),
                         ("by_quintile_pool", "当日**冻结池**截面五分位（1 = 最低）")):
        add(f"### 按 {caption} 分组")
        add("")
        add("| 五分位 | h | 均值 bp | NW(5) SE bp | t | 95% CI bp | 7 折为正 | 事件数 |")
        add("|---|---|---|---|---|---|---|---|")
        for q in (1, 2, 3, 4, 5):
            for h in DRIFT_HORIZONS:
                s = d[key][str(h)][str(q)]
                if not s.get("n_events"):
                    add(f"| Q{q} | +{h} | — | — | — | — | — | 0 |")
                    continue
                add(f"| Q{q} | +{h} | {s['mean_bp']:+.4f} | {s['nw5_se_bp']:.4f} | "
                    f"{s['nw5_t']:+.3f} | [{s['ci95_bp'][0]:+.4f}, {s['ci95_bp'][1]:+.4f}] | "
                    f"{s['folds_positive']}/{s['folds_total']} | {s['n_events']:,} |")
        add("")
    return "\n".join(lines) + "\n"


def _write_outputs(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    (out_dir / "report_tables.md").write_text(render_tables(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="exp15 公告者倾斜开发折估计")
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
        from exp15_smoke import run_smoke  # noqa: PLC0415  仅冒烟路径导入
        return run_smoke(args.out_dir)
    report = run(args.processed.resolve(), args.derived.resolve(),
                 args.outputs_root.resolve(), folds=args.folds,
                 memory_limit_gb=args.memory_limit_gb)
    _write_outputs(report, args.out_dir)
    check = report["E0_baseline_check"]
    log(f"[exp15] E0 一致性：{check['status']}（{check.get('n_fields_checked', 0)} 字段，"
        f"{len(check.get('mismatches', []))} 项不一致）")
    for arm in ARMS:
        base = report["readouts"][arm][BASELINE_CONFIG]["net_by_cost_bp"]["8"]
        log(f"[{arm.upper()}] E0 净8bp {base['net_annual_pct']:.4f}% 夏普 {base['sharpe']:.4f}")
        for calendar_name in CALENDARS:
            for name in (_b1_name(calendar_name), _b2_name(calendar_name, W_EVENT_MAIN)):
                s = report["readouts"][arm][name]["net_by_cost_bp"]["8"]
                p = report["paired_vs_E0"][arm][name]
                log(f"       {name}: 净8bp {s['net_annual_pct']:.4f}% "
                    f"夏普 {s['sharpe']:.4f} 方差比 {p['var_ratio_pooled']:.4f} "
                    f"配对差 {p['mean_diff_bp_per_day']:+.4f} bp/日 "
                    f"CI [{p['ci95_bp'][0]:+.4f}, {p['ci95_bp'][1]:+.4f}] "
                    f"{p['folds_positive']}/{p['folds_total']} 折为正")
    log(f"[exp15] wrote {args.out_dir / 'summary.json'}")
    if check["status"] == "partial_run_not_comparable":
        log("[exp15] 试跑（非全折），未做 E0 逐位核对；正式读数必须跑满折 36–42。")
        return 0
    if check["status"] != "identical":
        log("[exp15] **E0 与 nt5_baseline_readout.json 不一致 —— 按任务书 §4.1 停下报告**")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
