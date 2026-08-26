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

import numpy as np
import pandas as pd

from crsp_pipeline.calendar import CalendarError, TradingCalendar

from . import import_kronos
from .dataset import FEATURE_COLS
from .models import pick_device

PRED_COLS = ["open", "high", "low", "close", "volume", "amount"]
# 官方 backtest 采样参数（finetune/config.py）
SAMPLING_DEFAULTS = dict(T=0.6, top_p=0.9, top_k=0, sample_count=5)


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
    max_context: int = 512,
    permno_col: str = "PERMNO",
    date_col: str = "DlyCalDt",
    verbose: bool = False,
    **sampling,
) -> pd.DataFrame:
    """对 scoring_index（windows.build_scoring_index 产物，lookback 侧已保证
    连续有效）逐观测打分。输出 (signal_date, PERMNO, score, extrapolated)。"""
    _, _, KronosPredictor = import_kronos()
    params = {**SAMPLING_DEFAULTS, **sampling}
    predictor = KronosPredictor(model, tokenizer,
                                device=str(pick_device(device)),
                                max_context=max_context)

    df = panel.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    rename = {v: k for k, v in FEATURE_COLS.items()}
    rename["DlyVol"] = "volume"
    rename["DlyPrcVol"] = "amount"
    by_pn = {
        pn: g.sort_values(date_col).set_index(date_col).rename(columns=rename)
        for pn, g in df.groupby(permno_col)
    }

    rows = []
    idx = scoring_index.reset_index(drop=True)
    for lo in range(0, len(idx), batch_size):
        chunk = idx.iloc[lo:lo + batch_size]
        df_list, x_ts, y_ts, meta = [], [], [], []
        for pn, anchor, start in chunk[[permno_col, "anchor", "start"]].itertuples(index=False):
            sec = by_pn[pn]
            win = sec.loc[pd.Timestamp(start):pd.Timestamp(anchor), PRED_COLS]
            fut, extrap = future_sessions(calendar, anchor, predict)
            df_list.append(win.reset_index(drop=True))
            x_ts.append(pd.Series(win.index))
            y_ts.append(pd.Series(fut))
            meta.append((pn, pd.Timestamp(anchor), extrap))
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
