"""torch Dataset：CRSP 面板窗口 → Kronos 训练样本。

数据契约与官方 finetune/dataset.py 完全一致：

- 特征 6 列，顺序 [open, high, low, close, vol, amt]（amt = DlyPrcVol）；
- 时间特征 5 列 [minute, hour, weekday, day, month]（日频数据 minute=hour=0）；
- 归一化：mean/std **只在 lookback 段**上计算（防未来泄漏），作用于整个
  窗口，eps=1e-5，clip ±5（官方 dataset.py 第 109–117 行的逐行复刻）；
- 返回 (x [win, 6] float32, x_stamp [win, 5] float32)。

与官方的差别只有采样方式：官方每 epoch 随机重抽 n_iter 个窗口，本实现
枚举全部窗口、由 DataLoader(shuffle) 决定顺序（可复现，样本量按折固定）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from crsp_pipeline.calendar import TradingCalendar

FEATURE_COLS = {
    "open": "DlyOpen", "high": "DlyHigh", "low": "DlyLow",
    "close": "DlyClose", "vol": "DlyVol", "amt": "DlyPrcVol",
}
N_FEATURES = 6
N_TIME_FEATURES = 5


def _time_stamp_matrix(dates: pd.DatetimeIndex) -> np.ndarray:
    """[minute, hour, weekday, day, month]，与官方 QlibDataset 同序同义。"""
    return np.column_stack([
        dates.minute.to_numpy(),
        dates.hour.to_numpy(),
        dates.weekday.to_numpy(),
        dates.day.to_numpy(),
        dates.month.to_numpy(),
    ]).astype(np.float32)


class KronosWindowDataset(Dataset):
    """把 (windows.build_window_index 的索引 + 面板) 变成官方契约样本。

    初始化时每只股票 reindex 到交易日历、抽成 numpy 数组；__getitem__ 只做
    切片 + 归一化，无 pandas 开销，num_workers=0 也够快（Windows spawn 安全）。
    """

    def __init__(
        self,
        panel: pd.DataFrame,
        window_index: pd.DataFrame,
        calendar: TradingCalendar,
        lookback: int,
        predict: int,
        clip: float = 5.0,
        permno_col: str = "PERMNO",
        date_col: str = "DlyCalDt",
    ):
        self.lookback = int(lookback)
        self.window = int(lookback + predict)
        self.clip = float(clip)

        df = panel.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        self._feat: dict = {}    # permno -> (n_sessions, 6) float32
        self._stamp: dict = {}   # permno -> (n_sessions, 5) float32
        self._pos: dict = {}     # permno -> {date: 行号}
        cols = list(FEATURE_COLS.values())
        for pn, g in df.groupby(permno_col):
            g = g.sort_values(date_col)
            sessions = calendar.sessions(g[date_col].iloc[0], g[date_col].iloc[-1])
            s = g.set_index(date_col)[cols].reindex(sessions)
            self._feat[pn] = s.to_numpy(dtype=np.float32)
            self._stamp[pn] = _time_stamp_matrix(sessions)
            self._pos[pn] = {d: i for i, d in enumerate(sessions)}

        # 索引展开成 (permno, 起始行号)；窗口有效性由 windows.py 保证
        self._items = [
            (row[0], self._pos[row[0]][pd.Timestamp(row[1])])
            for row in window_index[[permno_col, "start"]].itertuples(index=False)
        ]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, i: int):
        pn, s = self._items[i]
        x = self._feat[pn][s:s + self.window].copy()
        stamp = self._stamp[pn][s:s + self.window]

        past = x[: self.lookback]
        mean = past.mean(axis=0)
        std = past.std(axis=0)
        x = (x - mean) / (std + 1e-5)
        x = np.clip(x, -self.clip, self.clip)

        return torch.from_numpy(x), torch.from_numpy(stamp)
