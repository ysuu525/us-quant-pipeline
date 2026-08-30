"""torch Dataset：CRSP 面板窗口 → Kronos 训练样本。

数据契约与官方 finetune/dataset.py 完全一致：

- 特征 6 列，顺序 [open, high, low, close, vol, amt]（amt = DlyPrcVol）；
- 时间特征 5 列 [minute, hour, weekday, day, month]（日频数据 minute=hour=0）；
- 归一化：mean/std **只在 lookback 段**上计算（防未来泄漏），作用于整个
  窗口，eps=1e-5，clip ±5（官方 dataset.py 第 109–117 行的逐行复刻）；
- 返回 (x [win, 6] float32, x_stamp [win, 5] float32)。

采样方式与官方一致（2026-08-27 改回，登记簿有记录）：官方每 epoch 用
``seed + epoch`` 播种的 rng 从全部窗口池**有放回**随机抽 n_iter 个
（finetune/dataset.py 的 set_epoch_seed / __getitem__）。真实数据一折约
350 万个窗口，此前的全量枚举实测 27 分钟/epoch，不可行。

- ``samples_per_epoch=None``（默认）：枚举全部窗口（合成数据测试用）；
- ``samples_per_epoch=N``：每 epoch 抽 min(N, 池大小) 个；每个 epoch 开始
  前调用 ``set_epoch_seed(epoch)``（训练侧传 epoch；**内层验证侧固定传
  同一值**，使早停指标逐 epoch 可比——这是对官方的唯一刻意偏离）。
"""

from __future__ import annotations

import random

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
        samples_per_epoch: int | None = None,
        seed: int = 0,
    ):
        self.lookback = int(lookback)
        self.window = int(lookback + predict)
        self.clip = float(clip)
        self.samples_per_epoch = samples_per_epoch
        self._seed = int(seed)
        self._rng = random.Random(self._seed)

        # 只预载窗口索引实际引用的股票（batch 内容不受影响；
        # 全量池 2.5 万只 → 22GB，universe 池只需其中一小部分）
        needed = set(window_index[permno_col].unique())
        df = panel[panel[permno_col].isin(needed)].copy()
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

    def set_epoch_seed(self, epoch: int) -> None:
        """官方 set_epoch_seed 的对应物：每个 epoch（或每次评估）开始前调用。"""
        self._rng.seed(self._seed + int(epoch))

    def __len__(self) -> int:
        if self.samples_per_epoch is None:
            return len(self._items)
        return min(int(self.samples_per_epoch), len(self._items))

    def __getitem__(self, i: int):
        if self.samples_per_epoch is None:
            pn, s = self._items[i]
        else:
            # 官方契约：忽略 i，从全池有放回随机抽（rng 由 set_epoch_seed 控制）
            pn, s = self._items[self._rng.randint(0, len(self._items) - 1)]
        x = self._feat[pn][s:s + self.window].copy()
        stamp = self._stamp[pn][s:s + self.window]

        past = x[: self.lookback]
        mean = past.mean(axis=0)
        std = past.std(axis=0)
        x = (x - mean) / (std + 1e-5)
        x = np.clip(x, -self.clip, self.clip)

        return torch.from_numpy(x), torch.from_numpy(stamp)
