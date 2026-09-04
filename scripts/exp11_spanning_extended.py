"""实验 11：在 K6b 上扩充短期价量张成控制集（只读开发折 36–42）。

判据 / 用途限制（先写死；CLAUDE.md §二：判据必须先于结果）

先把新控制集的完整定义与本段判据抄进 scripts/exp11_spanning_extended.py 的 docstring，
再跑第一行数据。
门槛照 K6b 原样，不得改（ledger:191/192）：量级门槛 保留率 ≥ 75%；
一致性门槛 逐折（固定载荷残差）alpha > 0 ≥ 5/7。两个门槛同时满足才写「非翻版」。
估计交付 + 机械判读：报保留率与逐折。不得因为加了控制而回头调 K6b 的门槛。
措辞必须带 K6b 已有的限定：保留率 >100% 时附「正向暴露于本样本内亏钱的因子」，
且不得写成 "survives spanning" 式的无限定断言（C11）。
只在开发折 36–42 上做；H1b（张成是否进确认集）由用户裁定，不在本实验范围。

K6b 三层限定（本实验完整继承）
-------------------------------
1. 控制因子用毛收益（成本 = 0），策略用净收益（8bp），刻意让控制集更强。
2. 控制因子与策略走完全相同的 top500 / 六档错位 / 进前10% / 跌出前30%
   才卖 / t+1 开盘建仓管道；只替换 score，控制掉共同的变现构造。
3. 控制因子取多空价差腿；该管道对因子方向精确对称，避免按赚钱方向选符号。
   每天候选池固定为该折 Kronos scores 中当天有分的名字。

新增控制集（固定，不扫描）
--------------------------
* turnover：DlyPrcVol / DlyCap 的 20 日均值，再在当日候选池内取截面秩；
* turnover_x_rev1 / turnover_x_rev5：turnover 秩与对应短期反转秩各自去均值后相乘；
* hi52：复权 close / 截至当日的 252 日 rolling_max(复权 close)，严格需 252 个观测；
* industry neutral：从 security_info_history 逐日按生效区间匹配 SIC2，信号和因子
  均先在当日候选池的 SIC2 内去均值；区间失效或缺 SIC 的记录不进入行业规格。

三个规格全部预先固定并全部报告：
* S-T：K6b 8 因子 + turnover + turnover_x_rev1 + turnover_x_rev5 + 市场；
* S-H：K6b 8 因子 + hi52 + 市场；
* S-TH-ind：K6b 8 因子 + 上述 turnover 三项 + hi52 + 市场，且信号与全部
  个股因子先做 point-in-time SIC2 行业内去均值。

明确缺口：财报日虚拟需要外部财报日历，SUE 需要 Compustat；本地 CRSP 无字段，
本次均不构造、不用代理替代。所有价量特征只用信号日收盘及以前的信息；收益侧
只由 K6b 的冻结管道在 t+1 及以后作为结果使用。大面板按不超过两个日历年的块
读取，严格列裁剪并在 parquet 层日期下推；绝不读取未开放折。
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from crsp_pipeline.sealed import assert_readable  # noqa: E402
from signals.kronos_adapter import scores_path  # noqa: E402

PROCESSED = Path(r"F:\quant\processed\crsp_ciz_2026-08-24_20260825T130601Z")
JKP = Path(r"F:\quant\external\jkp")
OUTPUTS = REPO / "outputs"
OUT_JSON = OUTPUTS / "exp11_spanning_extended.json"
MEMORY_LIMIT_GB = 40.0
FOLDS: tuple[int, ...] = tuple(range(36, 43))
FOLD_WINDOWS: dict[int, tuple[str, str]] = {
    36: ("2020-07-01", "2020-12-31"),
    37: ("2021-01-04", "2021-06-30"),
    38: ("2021-07-01", "2021-12-31"),
    39: ("2022-01-03", "2022-06-30"),
    40: ("2022-07-01", "2022-12-30"),
    41: ("2023-01-03", "2023-06-30"),
    42: ("2023-07-03", "2023-12-29"),
}
# 每块最多两个日历年；前一年提供 hi52 / K6b 动量所需的历史。
BLOCKS: tuple[dict[str, Any], ...] = (
    {"lo": "2019-01-01", "hi": "2020-12-31", "folds": (36,)},
    {"lo": "2020-01-01", "hi": "2021-12-31", "folds": (37, 38)},
    {"lo": "2021-01-01", "hi": "2022-12-31", "folds": (39, 40)},
    {"lo": "2022-01-01", "hi": "2023-12-31", "folds": (41, 42)},
)
TURNOVER_WINDOW = 20
HI52_WINDOW = 252
RETENTION_THRESHOLD_PCT = 75.0
CONSISTENCY_THRESHOLD = 5
RAW_COLUMNS: tuple[str, ...] = (
    "PERMNO", "DlyCalDt", "DlyOpen", "DlyHigh", "DlyLow", "DlyClose",
    "DlyVol", "DlyPrcVol", "DlyRet", "DlyCap",
)
ADJUSTED_COLUMNS: tuple[str, ...] = ("PERMNO", "DlyCalDt", "DlyClose")
EXTENDED_COLUMNS: tuple[str, ...] = (
    "turnover", "turnover_x_rev1", "turnover_x_rev5", "hi52",
)
SPEC_COLUMNS: dict[str, tuple[str, ...]] = {}


class MemoryGateError(RuntimeError):
    """系统提交内存高于冻结门槛。"""


def _load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


K6B = _load_script_module("exp11_k6b", REPO / "scripts" / "k6b_spanning.py")
CORE: tuple[str, ...] = tuple(K6B.CORE)
SPEC_COLUMNS.update(
    {
        "S-T": CORE + ("turnover", "turnover_x_rev1", "turnover_x_rev5"),
        "S-H": CORE + ("hi52",),
        "S-TH-ind": CORE + EXTENDED_COLUMNS,
    }
)


def log(message: str) -> None:
    print(message, flush=True)


def committed_memory_gb() -> float:
    """Windows 提交内存；非 Windows 测试环境退回 /proc/meminfo。"""
    if sys.platform == "win32":
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


def enforce_memory_gate(limit_gb: float = MEMORY_LIMIT_GB) -> float:
    committed = committed_memory_gb()
    if committed > limit_gb:
        raise MemoryGateError(
            f"committed memory {committed:.2f} GB > {limit_gb:.2f} GB；"
            "等待 GPU/其它任务释放内存后再运行实验 11"
        )
    return committed


def _date_filters(lo: str, hi: str) -> list[tuple[str, str, pd.Timestamp]]:
    return [
        ("DlyCalDt", ">=", pd.Timestamp(lo)),
        ("DlyCalDt", "<=", pd.Timestamp(hi)),
    ]


def load_raw_block(processed: Path, lo: str, hi: str) -> pd.DataFrame:
    path = processed / "panel_raw.parquet"
    assert_readable(path)
    frame = pd.read_parquet(path, columns=list(RAW_COLUMNS), filters=_date_filters(lo, hi))
    frame["DlyCalDt"] = pd.to_datetime(frame["DlyCalDt"])
    return (
        frame.dropna(subset=["DlyRet"])
        .sort_values(["PERMNO", "DlyCalDt"], kind="mergesort")
        .reset_index(drop=True)
    )


def load_adjusted_block(processed: Path, lo: str, hi: str) -> pd.DataFrame:
    path = processed / "panel_kronos_adj.parquet"
    assert_readable(path)
    frame = pd.read_parquet(
        path, columns=list(ADJUSTED_COLUMNS), filters=_date_filters(lo, hi)
    )
    frame["DlyCalDt"] = pd.to_datetime(frame["DlyCalDt"])
    return frame.sort_values(["PERMNO", "DlyCalDt"], kind="mergesort").reset_index(drop=True)


def turnover_history(raw: pd.DataFrame) -> pd.DataFrame:
    """逐名字 20 日 DlyPrcVol/DlyCap 均值；不做任何前移。"""
    ordered = raw.sort_values(["PERMNO", "DlyCalDt"], kind="mergesort").copy()
    ratio = ordered["DlyPrcVol"] / ordered["DlyCap"].where(ordered["DlyCap"] > 0)
    ordered["turnover_raw"] = (
        ratio.groupby(ordered["PERMNO"], sort=False)
        .rolling(TURNOVER_WINDOW, min_periods=TURNOVER_WINDOW)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return ordered[["PERMNO", "DlyCalDt", "turnover_raw"]]


def hi52_history(adjusted: pd.DataFrame) -> pd.DataFrame:
    """复权 close / 截至当日的 252 日最高复权 close。"""
    ordered = adjusted.sort_values(["PERMNO", "DlyCalDt"], kind="mergesort").copy()
    close = ordered["DlyClose"].where(ordered["DlyClose"] > 0)
    maximum = (
        close.groupby(ordered["PERMNO"], sort=False)
        .rolling(HI52_WINDOW, min_periods=HI52_WINDOW)
        .max()
        .reset_index(level=0, drop=True)
    )
    ordered["hi52"] = close / maximum.where(maximum > 0)
    return ordered[["PERMNO", "DlyCalDt", "hi52"]]


def build_block_factors(raw: pd.DataFrame, adjusted: pd.DataFrame, processed: Path) -> pd.DataFrame:
    """复用 K6b 的 CORE 构造，并只增加冻结的历史量。"""
    assert_readable(processed / "market_index.parquet")
    # K6b.build_factors 从自身冻结 P 读取市场指数；默认运行要求同一 processed 快照。
    if processed.resolve() != Path(K6B.P).resolve():
        raise ValueError(f"processed 必须与 K6b 冻结快照一致：{K6B.P}")
    turnover = turnover_history(raw)
    hi52 = hi52_history(adjusted)
    base = K6B.build_factors(raw)
    merged = base.merge(turnover, on=["PERMNO", "DlyCalDt"], how="left", validate="one_to_one")
    merged = merged.merge(hi52, on=["PERMNO", "DlyCalDt"], how="left", validate="one_to_one")
    return merged


def load_sic_history(processed: Path) -> pd.DataFrame:
    path = processed / "security_info_history.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"缺 point-in-time 行业历史：{path}")
    assert_readable(path)
    history = pd.read_parquet(
        path, columns=["permno", "secinfostartdt", "secinfoenddt", "siccd"]
    )
    history = history.dropna(subset=["siccd"]).rename(columns={"permno": "PERMNO"})
    # Parquet commonly returns ``permno`` as nullable Int64 while score files use
    # numpy int64.  merge_asof requires the by-key dtypes to match exactly.
    history["PERMNO"] = pd.to_numeric(history["PERMNO"], errors="raise").astype("int64")
    history["secinfostartdt"] = pd.to_datetime(history["secinfostartdt"]).astype("datetime64[ns]")
    history["secinfoenddt"] = pd.to_datetime(history["secinfoenddt"]).fillna(
        pd.Timestamp("2100-01-01")
    ).astype("datetime64[ns]")
    numeric = pd.to_numeric(history["siccd"], errors="coerce")
    history = history.loc[numeric.notna()].copy()
    history["sic2"] = numeric.loc[numeric.notna()].astype(int).astype(str).str.zfill(4).str[:2]
    if history.duplicated(["PERMNO", "secinfostartdt"]).any():
        raise ValueError("security_info_history 存在重复 (PERMNO, secinfostartdt)")
    return history.sort_values(["secinfostartdt", "PERMNO"], kind="mergesort").reset_index(drop=True)


def point_in_time_sic2(keys: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """照 xsec_context_probe.py:126-150 逐日区间匹配；失效后保持缺失。"""
    left = keys[["PERMNO", "signal_date"]].copy()
    left["PERMNO"] = pd.to_numeric(left["PERMNO"], errors="raise").astype("int64")
    left["signal_date"] = pd.to_datetime(left["signal_date"]).astype("datetime64[ns]")
    if left.duplicated(["PERMNO", "signal_date"]).any():
        raise ValueError("行业匹配键重复")
    left = left.sort_values(["signal_date", "PERMNO"], kind="mergesort")
    right = history[["PERMNO", "secinfostartdt", "secinfoenddt", "sic2"]].copy()
    right["PERMNO"] = pd.to_numeric(right["PERMNO"], errors="raise").astype("int64")
    right["secinfostartdt"] = pd.to_datetime(right["secinfostartdt"]).astype("datetime64[ns]")
    right["secinfoenddt"] = pd.to_datetime(right["secinfoenddt"]).astype("datetime64[ns]")
    right = right.sort_values(["secinfostartdt", "PERMNO"], kind="mergesort")
    matched = pd.merge_asof(
        left,
        right,
        left_on="signal_date",
        right_on="secinfostartdt",
        by="PERMNO",
        direction="backward",
    )
    expired = matched["secinfoenddt"].notna() & (
        matched["signal_date"] > matched["secinfoenddt"]
    )
    matched.loc[expired, "sic2"] = np.nan
    return matched.drop(columns=["secinfostartdt", "secinfoenddt"])


def add_candidate_extended_columns(candidate: pd.DataFrame) -> pd.DataFrame:
    """在当日 Kronos 候选池内取秩并机械构造两个交互。"""
    out = candidate.copy()
    group = out.groupby("signal_date", sort=False)
    out["turnover"] = group["turnover_raw"].rank(pct=True)
    for source, target in (("rev1", "turnover_x_rev1"), ("rev5", "turnover_x_rev5")):
        rank = group[source].rank(pct=True)
        centered_rank = rank - rank.groupby(out["signal_date"], sort=False).transform("mean")
        centered_turnover = out["turnover"] - group["turnover"].transform("mean")
        out[target] = centered_turnover * centered_rank
    return out


def industry_demean(candidate: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """缺 SIC 不回填；仅在当日候选池的有效 SIC2 内去均值。"""
    out = candidate.copy()
    valid = out["sic2"].notna()
    for column in columns:
        out[f"ind_{column}"] = np.nan
        group_mean = out.loc[valid].groupby(
            ["signal_date", "sic2"], sort=False
        )[column].transform("mean")
        centered = out.loc[valid, column] - group_mean
        out.loc[valid, f"ind_{column}"] = centered
    return out


def _by_day(frame: pd.DataFrame, value_column: str, minimum_names: int = 50) -> dict:
    result = {}
    for day, group in frame.groupby("signal_date", sort=True):
        values = group[["PERMNO", value_column]].dropna()
        values = values[np.isfinite(values[value_column])]
        if len(values) >= minimum_names:
            result[day] = dict(zip(values["PERMNO"], values[value_column]))
    return result


def _panel_maps(factors: pd.DataFrame) -> tuple[dict, dict, dict]:
    grouped = factors.groupby("DlyCalDt", sort=True)
    ret = {day: dict(zip(group["PERMNO"], group["DlyRet"])) for day, group in grouped}
    grouped = factors.groupby("DlyCalDt", sort=True)
    oc = {day: dict(zip(group["PERMNO"], group["oc"])) for day, group in grouped}
    grouped = factors.groupby("DlyCalDt", sort=True)
    adv = {day: dict(zip(group["PERMNO"], group["adv20"])) for day, group in grouped}
    return ret, oc, adv


def _fold_candidate(
    fold: int, factors: pd.DataFrame, sic_history: pd.DataFrame, outputs_root: Path
) -> tuple[pd.DataFrame, dict]:
    path = scores_path(fold, "ft", root=outputs_root)
    assert_readable(path)
    scores = pd.read_parquet(path, columns=["PERMNO", "signal_date", "score"]).dropna()
    scores["signal_date"] = pd.to_datetime(scores["signal_date"])
    lo, hi = map(pd.Timestamp, FOLD_WINDOWS[fold])
    if scores["signal_date"].min() < lo or scores["signal_date"].max() > hi:
        raise ValueError(f"fold{fold} scores 日期越过冻结窗口 {lo.date()}..{hi.date()}")
    if scores.duplicated(["PERMNO", "signal_date"]).any():
        raise ValueError(f"fold{fold} scores 键重复")
    keyed = factors.rename(columns={"DlyCalDt": "signal_date"})
    candidate = scores.merge(keyed, on=["PERMNO", "signal_date"], how="left", validate="one_to_one")
    sic = point_in_time_sic2(scores[["PERMNO", "signal_date"]], sic_history)
    candidate = candidate.merge(sic, on=["PERMNO", "signal_date"], how="left", validate="one_to_one")
    candidate = add_candidate_extended_columns(candidate)
    all_industry_columns = ("score",) + CORE + EXTENDED_COLUMNS
    candidate = industry_demean(candidate, all_industry_columns)
    raw_scores = _by_day(scores, "score")
    return candidate, raw_scores


def collect_fold_daily(
    fold: int,
    factors: pd.DataFrame,
    sic_history: pd.DataFrame,
    outputs_root: Path,
) -> dict[str, dict[str, Any]]:
    """复用 K6b.run_pipeline；所有规格参数只来自本文件的冻结常量。"""
    candidate, raw_scores = _fold_candidate(fold, factors, sic_history, outputs_root)
    days = sorted(raw_scores)
    # K6b 的管道只会索引 signal days（及 `days[i+1]`，仍在同一集合）；
    # 先裁到本折日期再建 Python 字典，避免把整个两年块复制成三套巨型映射。
    fold_panel = factors[factors["DlyCalDt"].isin(days)]
    ret, oc, adv = _panel_maps(fold_panel)
    raw_strategy = K6B.run_pipeline(raw_scores, days, ret, oc, adv, K6B.COST_BP)
    ind_scores = _by_day(candidate, "ind_score")
    industry_strategy = K6B.run_pipeline(
        ind_scores, days, ret, oc, adv, K6B.COST_BP
    )
    out: dict[str, dict[str, Any]] = {}
    for spec, columns in SPEC_COLUMNS.items():
        is_industry = spec == "S-TH-ind"
        spec_days = days
        strategy = industry_strategy if is_industry else raw_strategy
        controls = {}
        for column in columns:
            value_column = f"ind_{column}" if is_industry else column
            factor_scores = _by_day(candidate, value_column)
            controls[column] = K6B.run_pipeline(
                factor_scores, spec_days, ret, oc, adv, 0.0
            )["ls"].rename(column)
        strategy = strategy.copy()
        strategy["fold"] = f"fold{fold}"
        out[spec] = {"strategy": strategy, "controls": controls}
    return out


def load_market_return(jkp_root: Path) -> pd.Series:
    path = jkp_root / "usa_mkt_daily_vw_cap.csv"
    assert_readable(path)
    market = pd.read_csv(path)
    market["date"] = pd.to_datetime(market["date"])
    market = market.set_index("date")
    candidates = [column for column in market.columns if column.lower() in ("ret", "mkt", "mktrf")]
    if len(candidates) != 1:
        raise ValueError(f"市场收益列不唯一：{candidates}")
    return market[candidates[0]].rename("market")


def mechanical_readout(
    spec: str,
    strategy: pd.DataFrame,
    controls: pd.DataFrame,
    market: pd.Series,
) -> dict[str, Any]:
    regressors = pd.concat([controls, market], axis=1, sort=True)
    data = pd.concat([strategy["long"].rename("y"), regressors], axis=1, sort=True).dropna()
    if len(data) < 40:
        raise ValueError(f"{spec} 完整回归日不足：{len(data)}")
    matrix = data.drop(columns="y")
    design = np.column_stack([np.ones(len(data)), matrix.to_numpy(dtype=np.float64)])
    y = data["y"].to_numpy(dtype=np.float64)
    beta, tstat = K6B.nw_ols(y, design, 5)
    raw_ann = float(y.mean() * 252 * 100)
    alpha_ann = float(beta[0] * 252 * 100)
    retention = float(100 * alpha_ann / raw_ann) if raw_ann else math.nan
    residual = y - matrix.to_numpy(dtype=np.float64) @ beta[1:]
    residual_series = pd.Series(residual, index=data.index)
    per_fold: dict[str, float] = {}
    for fold in (f"fold{number}" for number in FOLDS):
        dates = strategy.index[strategy["fold"] == fold]
        selected = residual_series.loc[residual_series.index.isin(dates)]
        if len(selected) < 40:
            continue
        per_fold[fold] = float(selected.mean() * 252 * 100)
    positive = sum(value > 0 for value in per_fold.values())
    if positive < CONSISTENCY_THRESHOLD:
        conclusion = "不稳定，不可判定（固定载荷残差 alpha>0 少于 5/7）"
    elif retention >= RETENTION_THRESHOLD_PCT:
        conclusion = "非翻版（仅限本控制集、本开发样本与冻结构造）"
    elif retention >= 50.0:
        conclusion = "部分重叠（仅限本控制集、本开发样本与冻结构造）"
    else:
        conclusion = "已知短期价量效应的翻版（仅限本控制集、本开发样本与冻结构造）"
    caveat = (
        "保留率 >100% 仅表示正向暴露于本样本内亏钱的因子；"
        "不得写成无限定的 survives spanning。"
        if retention > 100.0
        else "判读仅限固定控制集、已消耗开发折与冻结构造。"
    )
    return {
        "n_days": int(len(data)),
        "n_regressors": int(matrix.shape[1]),
        "regressors": list(matrix.columns),
        "raw_ann_pct": raw_ann,
        "alpha_ann_pct": alpha_ann,
        "retention_pct": retention,
        "nw5_t_alpha": float(tstat[0]),
        "folds_alpha_positive": int(positive),
        "per_fold_fixed_loading_residual_alpha_ann_pct": per_fold,
        "thresholds": {
            "retention_pct_gte": RETENTION_THRESHOLD_PCT,
            "folds_alpha_positive_gte": CONSISTENCY_THRESHOLD,
        },
        "mechanical_conclusion": conclusion,
        "required_caveat": caveat,
        "betas": {
            column: {"coefficient": float(value), "nw5_t": float(tvalue)}
            for column, value, tvalue in zip(matrix.columns, beta[1:], tstat[1:])
        },
    }


def run(processed: Path, jkp_root: Path, outputs_root: Path) -> dict[str, Any]:
    start_committed = enforce_memory_gate()
    sic_history = load_sic_history(processed)
    collected = {
        spec: {"strategy": [], "controls": {column: [] for column in columns}}
        for spec, columns in SPEC_COLUMNS.items()
    }
    block_meta = []
    for block in BLOCKS:
        before = enforce_memory_gate()
        lo, hi = block["lo"], block["hi"]
        log(f"读取块 {lo}..{hi}；committed={before:.2f} GB")
        raw = load_raw_block(processed, lo, hi)
        adjusted = load_adjusted_block(processed, lo, hi)
        factors = build_block_factors(raw, adjusted, processed)
        del raw, adjusted
        for fold in block["folds"]:
            fold_daily = collect_fold_daily(fold, factors, sic_history, outputs_root)
            for spec, payload in fold_daily.items():
                collected[spec]["strategy"].append(payload["strategy"])
                for column, series in payload["controls"].items():
                    collected[spec]["controls"][column].append(series)
            log(f"  fold{fold} 完成")
        block_meta.append(
            {"lo": lo, "hi": hi, "folds": list(block["folds"]), "rows_after_k6b": len(factors)}
        )
        del factors
        gc.collect()
    market = load_market_return(jkp_root)
    results = {}
    for spec, payload in collected.items():
        strategy = pd.concat(payload["strategy"]).sort_index()
        controls = pd.concat(
            [pd.concat(payload["controls"][column]).sort_index() for column in SPEC_COLUMNS[spec]],
            axis=1,
        )
        results[spec] = mechanical_readout(spec, strategy, controls, market)
    return {
        "meta": {
            "run_utc": datetime.now(timezone.utc).isoformat(),
            "folds": list(FOLDS),
            "blocks": block_meta,
            "starting_committed_memory_gb": start_committed,
            "memory_limit_gb": MEMORY_LIMIT_GB,
            "k6b_reused": ["CORE", "build_factors", "run_pipeline", "nw_ols"],
            "k6b_frozen_construction": {
                "nt": K6B.NT,
                "topn": K6B.TOPN,
                "exit_pct": K6B.EXIT_PCT,
                "strategy_cost_bp": K6B.COST_BP,
                "control_cost_bp": 0.0,
            },
            "use_restriction": (
                "estimate + mechanical readout on dev folds only; do not retune K6b thresholds; "
                "H1b inclusion remains a user decision"
            ),
        },
        "fixed_specs": {name: list(columns) + ["market"] for name, columns in SPEC_COLUMNS.items()},
        "missing_controls": {
            "earnings_announcement_day": "requires external earnings calendar; absent from local CRSP; not run",
            "SUE": "requires Compustat; absent from project data; not run",
        },
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=PROCESSED)
    parser.add_argument("--jkp", type=Path, default=JKP)
    parser.add_argument("--outputs-root", type=Path, default=OUTPUTS)
    parser.add_argument("--out", type=Path, default=OUT_JSON)
    return parser.parse_args()


def print_table(report: dict[str, Any]) -> None:
    print("| spec | alpha_ann_pct | retention_pct | NW5 t | positive folds | mechanical readout |")
    print("|---|---:|---:|---:|---:|---|")
    for spec in ("S-T", "S-H", "S-TH-ind"):
        result = report["results"][spec]
        print(
            f"| {spec} | {result['alpha_ann_pct']:+.4f} | "
            f"{result['retention_pct']:.2f} | {result['nw5_t_alpha']:+.3f} | "
            f"{result['folds_alpha_positive']}/7 | {result['mechanical_conclusion']} |"
        )
        print(f"  限定：{result['required_caveat']}")
    print("缺口：财报日虚拟需外部日历；SUE 需 Compustat；本次均未构造。")


def main() -> int:
    args = parse_args()
    try:
        report = run(args.processed.resolve(), args.jkp.resolve(), args.outputs_root.resolve())
    except MemoryGateError as exc:
        print(f"[exp11] WAIT: {exc}", file=sys.stderr)
        return 3
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_table(report)
    print(f"[exp11] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
