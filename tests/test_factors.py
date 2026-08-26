"""自建归因因子（§1）：暴露滚动、多空组合分组用滞后特征、动量窗口边界。"""

import numpy as np
import pandas as pd
import pytest

from crsp_pipeline import factors as F


def _panel(cal, permno, rets, start=0):
    d = cal.dates[start:start + len(rets)]
    return pd.DataFrame({"PERMNO": permno, "DlyCalDt": d, "DlyRet": rets})


def test_market_factor():
    idx = pd.DataFrame({"caldt": ["2020-01-06", "2020-01-07"], "vwretd": [0.01, -0.02]})
    s = F.market_factor(idx)
    assert s.iloc[1] == -0.02 and s.index[0] == pd.Timestamp("2020-01-06")


def test_rolling_realized_vol(cal):
    p = _panel(cal, 1, [0.01] * 10)
    v = F.rolling_realized_vol(p, cal, window=5, min_obs=3)
    assert np.isnan(v["vol"].iloc[1])          # 观测不足
    assert v["vol"].iloc[4] == pytest.approx(0.0)  # 常数收益 → 波动 0

    p2 = _panel(cal, 1, [0.01, -0.01] * 5)
    v2 = F.rolling_realized_vol(p2, cal, window=5, min_obs=5)
    assert v2["vol"].iloc[-1] > 0


def test_rolling_beta_two_x_market(cal):
    mkt = pd.Series(np.sin(np.arange(20)) * 0.01 + 0.001, index=cal.dates[:20])
    p = _panel(cal, 1, (2.0 * mkt).tolist())
    b = F.rolling_beta(p, mkt, cal, window=10, min_obs=5)
    assert b["beta"].iloc[-1] == pytest.approx(2.0, rel=1e-9)


def test_long_short_factor_uses_lagged_char(cal):
    # 6 只股票、2 天。分组用第 1 天的特征，收益取第 2 天。
    d1, d2 = cal.dates[0], cal.dates[1]
    rows = []
    caps = [1, 2, 3, 4, 5, 6]
    rets_d2 = [0.01, 0.03, 0.0, 0.0, 0.02, 0.04]
    for i in range(6):
        rows.append({"PERMNO": i + 1, "DlyCalDt": d1, "DlyCap": caps[i], "DlyRet": 0.0})
        # 第 2 天特征故意换成反序——若误用当日特征，结果会不同
        rows.append({"PERMNO": i + 1, "DlyCalDt": d2, "DlyCap": caps[5 - i], "DlyRet": rets_d2[i]})
    panel = pd.DataFrame(rows)
    f = F.size_factor(panel, n_groups=3, min_names_per_leg=1)
    # 小组 {1,2} 均值 0.02，大组 {5,6} 均值 0.03 → 小减大 = −0.01
    assert np.isnan(f.loc[d1])  # 首日无滞后特征
    assert f.loc[d2] == pytest.approx(-0.01)


def test_momentum_char_window_and_skip(cal):
    # formation=5, skip=1 → 窗口 [t−4, t−1]，跳过最近 1 日
    n = 10
    p = _panel(cal, 1, [0.01] * n)
    m = F.momentum_char(p, cal, formation_sessions=5, skip_sessions=1, min_obs=4)
    assert m["mom"].iloc[-1] == pytest.approx(1.01 ** 4 - 1, rel=1e-12)
    assert np.isnan(m["mom"].iloc[3])  # 历史不足

    # 跳过项确实生效：最后一天收益极端也不影响 t 日动量
    p2 = _panel(cal, 1, [0.01] * (n - 2) + [5.0, 0.01])
    m2 = F.momentum_char(p2, cal, formation_sessions=5, skip_sessions=1, min_obs=4)
    # t = 最后一日：窗口 [t−4, t−1] 含 +500% 那天 → 巨大
    assert m2["mom"].iloc[-1] > 1
    # t = 倒数第二日（+500% 当日）：其窗口只到 t−1，不含当日
    assert m2["mom"].iloc[-2] == pytest.approx(1.01 ** 4 - 1, rel=1e-12)


def test_momentum_factor_winners_minus_losers(cal):
    # 两只股票：W 一路 +1%，L 一路 −1%；动量特征滞后已由窗口保证
    n = 12
    w = _panel(cal, 1, [0.01] * n)
    l = _panel(cal, 2, [-0.01] * n)
    panel = pd.concat([w, l], ignore_index=True)
    mom = F.momentum_char(panel, cal, formation_sessions=5, skip_sessions=1, min_obs=4)
    merged = panel.merge(mom, on=["PERMNO", "DlyCalDt"])
    f = F.momentum_factor(merged, n_groups=2, min_names_per_leg=1)
    # 赢家日收益 0.01，输家 −0.01 → 赢减输 = 0.02
    assert f.dropna().iloc[-1] == pytest.approx(0.02)
