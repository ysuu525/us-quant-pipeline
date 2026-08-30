"""KronosWindowDataset：官方归一化契约（lookback 段统计量、clip、时间特征）。"""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from kronos_ft.dataset import KronosWindowDataset  # noqa: E402
from kronos_ft.windows import build_window_index  # noqa: E402


def make_panel(cal, n, permno=1, seed=0):
    rng = np.random.default_rng(seed)
    d = cal.dates[:n]
    close = 50 + np.cumsum(rng.normal(0, 1, n))
    close = np.abs(close) + 1
    return pd.DataFrame({
        "PERMNO": permno, "DlyCalDt": d,
        "DlyOpen": close * 1.01, "DlyHigh": close * 1.02, "DlyLow": close * 0.99,
        "DlyClose": close, "DlyVol": rng.uniform(1e5, 1e6, n),
        "DlyPrcVol": close * 1e5,
    })


def test_shapes_and_normalization_golden(cal):
    L, P = 8, 4
    p = make_panel(cal, 30)
    idx = build_window_index(p, cal, L, P)
    ds = KronosWindowDataset(p, idx, cal, L, P)
    assert len(ds) == len(idx)

    x, stamp = ds[0]
    assert x.shape == (L + P, 6) and stamp.shape == (L + P, 5)
    assert x.dtype == torch.float32

    # 手算归一化：统计量只来自 lookback 段（官方契约，防未来泄漏）
    cols = ["DlyOpen", "DlyHigh", "DlyLow", "DlyClose", "DlyVol", "DlyPrcVol"]
    raw = p.iloc[: L + P][cols].to_numpy(np.float32)
    past = raw[:L]
    expected = (raw - past.mean(0)) / (past.std(0) + 1e-5)
    expected = np.clip(expected, -5.0, 5.0)
    assert np.allclose(x.numpy(), expected, atol=1e-5)


def test_future_leak_absent(cal):
    # 篡改窗口最后一行（未来行）不得改变 lookback 段的归一化结果
    L, P = 8, 4
    p1 = make_panel(cal, 30)
    p2 = p1.copy()
    p2.loc[L + P - 1, ["DlyOpen", "DlyHigh", "DlyLow", "DlyClose"]] *= 100.0
    idx1 = build_window_index(p1, cal, L, P)
    idx2 = build_window_index(p2, cal, L, P)
    x1, _ = KronosWindowDataset(p1, idx1, cal, L, P)[0]
    x2, _ = KronosWindowDataset(p2, idx2, cal, L, P)[0]
    assert np.allclose(x1.numpy()[:L], x2.numpy()[:L], atol=1e-6)


def test_time_features(cal):
    L, P = 5, 2
    p = make_panel(cal, 12)
    idx = build_window_index(p, cal, L, P)
    _, stamp = KronosWindowDataset(p, idx, cal, L, P)[0]
    d0 = cal.dates[0]
    # [minute, hour, weekday, day, month]（日频 → minute=hour=0）
    assert stamp[0].tolist() == [0.0, 0.0, float(d0.weekday()), float(d0.day), float(d0.month)]


def test_sampling_len_capped_and_deterministic(cal):
    # 官方 n_iter 契约：len = min(samples_per_epoch, 池大小)；
    # 同 seed 同 epoch → 同一抽样序列；不同 epoch → 不同序列
    L, P = 8, 4
    p = make_panel(cal, 40)
    idx = build_window_index(p, cal, L, P)
    assert len(idx) > 3

    ds = KronosWindowDataset(p, idx, cal, L, P, samples_per_epoch=3, seed=7)
    assert len(ds) == 3
    big = KronosWindowDataset(p, idx, cal, L, P, samples_per_epoch=10**6, seed=7)
    assert len(big) == len(idx)  # 上限=池大小

    ds.set_epoch_seed(1)
    seq_a = [ds[i][0].numpy().copy() for i in range(3)]
    ds.set_epoch_seed(1)
    seq_b = [ds[i][0].numpy().copy() for i in range(3)]
    for a, b in zip(seq_a, seq_b):
        assert np.array_equal(a, b)

    ds.set_epoch_seed(2)
    seq_c = [ds[i][0].numpy() for i in range(3)]
    assert any(not np.array_equal(a, c) for a, c in zip(seq_a, seq_c))


def test_sampling_none_keeps_enumeration(cal):
    L, P = 8, 4
    p = make_panel(cal, 30)
    idx = build_window_index(p, cal, L, P)
    ds = KronosWindowDataset(p, idx, cal, L, P)          # 默认：枚举
    assert len(ds) == len(idx)
    x0a, _ = ds[0]
    ds.set_epoch_seed(99)                                 # 枚举模式下无效果
    x0b, _ = ds[0]
    assert np.array_equal(x0a.numpy(), x0b.numpy())
