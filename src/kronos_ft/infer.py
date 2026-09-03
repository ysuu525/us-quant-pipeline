"""推理打分（规范 §4/§6）。

score = predOpen(t+6) / predOpen(t+1) − 1 —— 模型只见 OHLCV，score 是
价格收益预测；标签是总收益，口径差已记入文档（§4）。

采样：官方 KronosPredictor.predict_batch，多路径在其内部取均值
（sample_count 路径平均）。默认采样参数取官方 backtest 配置
（finetune/config.py：T=0.6, top_p=0.9, top_k=0, sample_count=5），
冻结前如需改动记入试验登记簿。

y 时间戳：验证期推理用真实交易日历的后 6 个交易日；日历尾部（实盘当日）
不足 6 天时用工作日外推补齐——时间特征只进 stamp 嵌入，节假日错位影响
可忽略，但外推的发生次数会被统计返回。
"""

from __future__ import annotations

import contextlib

import numpy as np
import pandas as pd
import torch

from crsp_pipeline.calendar import CalendarError, TradingCalendar

from . import import_kronos
from .dataset import FEATURE_COLS
from .models import pick_device

PRED_COLS = ["open", "high", "low", "close", "volume", "amount"]
TIME_COLS = ["minute", "hour", "weekday", "day", "month"]
# 官方 backtest 采样参数（finetune/config.py）
SAMPLING_DEFAULTS = dict(T=0.6, top_p=0.9, top_k=0, sample_count=5)


def _stamp_matrix(dates: pd.DatetimeIndex) -> np.ndarray:
    """官方 calc_time_stamps 的等价物（同序同义，float32）。

    官方对**每个观测**都重算一遍；但打分窗口定义在交易日历上，同一 anchor
    日的窗口日期跨股票完全相同，故可按 anchor 缓存（fast 路径用）。"""
    d = pd.DatetimeIndex(dates)
    return np.column_stack([
        d.minute.to_numpy(), d.hour.to_numpy(), d.weekday.to_numpy(),
        d.day.to_numpy(), d.month.to_numpy(),
    ]).astype(np.float32)


def score_from_pred(pred_df: pd.DataFrame, predict: int = 6) -> float:
    """pred_df 行 = t+1 … t+predict 的预测。score = open(t+6)/open(t+1) − 1。"""
    if len(pred_df) < predict:
        return np.nan
    o1, o6 = float(pred_df["open"].iloc[0]), float(pred_df["open"].iloc[predict - 1])
    if not np.isfinite(o1) or o1 <= 0 or not np.isfinite(o6):
        return np.nan
    return o6 / o1 - 1.0


def future_sessions(calendar: TradingCalendar, anchor, n: int) -> tuple[pd.DatetimeIndex, bool]:
    """anchor 之后 n 个交易日；日历不够时工作日外推。返回 (日期, 是否外推)。"""
    try:
        return pd.DatetimeIndex([calendar.shift(anchor, k) for k in range(1, n + 1)]), False
    except CalendarError:
        pass
    tail = calendar.dates[calendar.dates > pd.Timestamp(anchor)]
    need = n - len(tail)
    start = (tail[-1] if len(tail) else pd.Timestamp(anchor)) + pd.offsets.BDay(1)
    ext = pd.bdate_range(start, periods=need)
    return tail.append(ext)[:n], True


def run_scoring(
    tokenizer,
    model,
    panel: pd.DataFrame,
    scoring_index: pd.DataFrame,
    calendar: TradingCalendar,
    lookback: int,
    predict: int = 6,
    batch_size: int = 128,
    device: str | None = None,
    amp: str | None = None,
    max_context: int = 512,
    permno_col: str = "PERMNO",
    date_col: str = "DlyCalDt",
    verbose: bool = False,
    fast: bool = True,
    **sampling,
) -> pd.DataFrame:
    """对 scoring_index（windows.build_scoring_index 产物，lookback 侧已保证
    连续有效）逐观测打分。输出 (signal_date, PERMNO, score, extrapolated)。

    fast=True（默认）：绕开 ``predict_batch`` 的逐序列 pandas 预处理，改用
    预抽 numpy + 按 anchor 缓存时间特征，直接调 ``predictor.generate``。
    **与官方路径逐位一致**（同样的 float32 数组、同样的 per-series
    mean/std/clip、同样的批次组成与顺序 → 同样的 RNG 消耗），由
    tests/test_kronos_train_infer.py::test_fast_path_bitwise_identical 盯死。
    实测：官方路径 CPU 侧 584 obs/s（占打分墙钟约 18%），fast 路径消除其中
    绝大部分。fast=False 走官方原路，用于对拍。

    amp：None（默认，纯 fp32，与既有全部读数同口径）/ "bf16" / "fp16"。
    开 amp 会改变数值与 RNG 之外的算子精度 → 分数与 fp32 不逐位相同，
    **跨口径的读数不可直接比较**（登记簿 2026-08-27 冻结项，粗筛定案后解禁）。
    """
    _, _, KronosPredictor = import_kronos()
    params = {**SAMPLING_DEFAULTS, **sampling}
    dev = pick_device(device)
    predictor = KronosPredictor(model, tokenizer,
                                device=str(dev),
                                max_context=max_context)
    if amp and dev.type == "cuda":
        _dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[amp]
        def _amp_ctx():
            return torch.autocast(device_type="cuda", dtype=_dtype)
    else:
        _amp_ctx = contextlib.nullcontext

    df = panel.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    rename = {v: k for k, v in FEATURE_COLS.items()}
    rename["DlyVol"] = "volume"
    rename["DlyPrcVol"] = "amount"
    by_pn = {
        pn: g.sort_values(date_col).set_index(date_col).rename(columns=rename)
        for pn, g in df.groupby(permno_col)
    }
    if fast:
        # 每只股票一份 (n_sessions, 6) float32 + 日期→行号；切片走位置索引
        feats = {pn: sec[PRED_COLS].to_numpy(np.float32) for pn, sec in by_pn.items()}
        pos = {pn: {d: i for i, d in enumerate(sec.index)} for pn, sec in by_pn.items()}
        stamp_cache: dict[pd.Timestamp, tuple[np.ndarray, np.ndarray, bool]] = {}

    def _stamps_for(anchor: pd.Timestamp, win_dates: pd.DatetimeIndex):
        """(x_stamp, y_stamp, extrapolated)，按 anchor 缓存——同一 anchor 的
        窗口日期与未来日期跨股票相同（窗口定义在交易日历上）。"""
        hit = stamp_cache.get(anchor)
        if hit is None:
            fut, ex = future_sessions(calendar, anchor, predict)
            hit = (_stamp_matrix(win_dates), _stamp_matrix(fut), ex)
            stamp_cache[anchor] = hit
        return hit

    rows = []
    idx = scoring_index.reset_index(drop=True)
    for lo in range(0, len(idx), batch_size):
        chunk = idx.iloc[lo:lo + batch_size]
        meta = []
        if fast:
            xs, xst, yst = [], [], []
            for pn, anchor, start in chunk[[permno_col, "anchor", "start"]].itertuples(index=False):
                anchor, start = pd.Timestamp(anchor), pd.Timestamp(start)
                sec_idx = by_pn[pn].index
                i0, i1 = pos[pn][start], pos[pn][anchor]
                x = feats[pn][i0:i1 + 1]
                x_stamp, y_stamp, extrap = _stamps_for(anchor, sec_idx[i0:i1 + 1])
                if not np.isfinite(x).all():
                    raise ValueError(f"打分窗含非有限值: PERMNO={pn} anchor={anchor.date()}")
                # 官方 predict_batch 的逐序列归一化，逐行对应
                mean, std = np.mean(x, axis=0), np.std(x, axis=0)
                xn = np.clip((x - mean) / (std + 1e-5), -predictor.clip, predictor.clip)
                xs.append(xn.astype(np.float32))
                xst.append(x_stamp)
                yst.append(y_stamp)
                meta.append((pn, anchor, extrap, mean, std))
            with _amp_ctx():
                preds_arr = predictor.generate(
                    np.stack(xs, 0), np.stack(xst, 0), np.stack(yst, 0), predict,
                    params["T"], params["top_k"], params["top_p"],
                    params["sample_count"], verbose,
                )
            for i, (pn, anchor, extrap, mean, std) in enumerate(meta):
                pred = pd.DataFrame(preds_arr[i] * (std + 1e-5) + mean, columns=PRED_COLS)
                rows.append({
                    permno_col: pn, "signal_date": anchor,
                    "score": score_from_pred(pred, predict),
                    "extrapolated": extrap,
                })
            continue

        df_list, x_ts, y_ts = [], [], []
        for pn, anchor, start in chunk[[permno_col, "anchor", "start"]].itertuples(index=False):
            sec = by_pn[pn]
            win = sec.loc[pd.Timestamp(start):pd.Timestamp(anchor), PRED_COLS]
            fut, extrap = future_sessions(calendar, anchor, predict)
            df_list.append(win.reset_index(drop=True))
            x_ts.append(pd.Series(win.index))
            y_ts.append(pd.Series(fut))
            meta.append((pn, pd.Timestamp(anchor), extrap))
        with _amp_ctx():
            preds = predictor.predict_batch(
                df_list, x_ts, y_ts, pred_len=predict,
                T=params["T"], top_k=params["top_k"], top_p=params["top_p"],
                sample_count=params["sample_count"], verbose=verbose,
            )
        for (pn, anchor, extrap), pred in zip(meta, preds):
            rows.append({
                permno_col: pn, "signal_date": anchor,
                "score": score_from_pred(pred, predict),
                "extrapolated": extrap,
            })
    return pd.DataFrame(rows)
