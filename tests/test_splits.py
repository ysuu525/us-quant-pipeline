"""切分纪律（§7）：purge off-by-one、walk-forward 滚动、封存 OOS。"""

import pandas as pd
import pytest

from crsp_pipeline.calendar import TradingCalendar
from crsp_pipeline.splits import (
    Fold,
    PurgeViolation,
    SealedOOSError,
    assert_purged,
    sealed_oos_window,
    walk_forward_folds,
)


@pytest.fixture
def cal10y():
    return TradingCalendar(pd.bdate_range("2015-01-01", "2026-12-31"))


def test_assert_purged_off_by_one(cal10y):
    cal = cal10y
    train_end = pd.Timestamp("2018-06-29")  # 周五
    train = cal.sessions("2018-01-01", train_end)

    # 标签最后触及 shift(train_end, 6)。val 从 shift(train_end, 7) 开始 → 恰好合法
    ok_start = cal.shift(train_end, 7)
    assert_purged(train, [ok_start], cal, label_horizon=6)

    # val 从 shift(train_end, 6) 开始 → 标签日 == 验证首日，必须报错（严格 <）
    bad_start = cal.shift(train_end, 6)
    with pytest.raises(PurgeViolation):
        assert_purged(train, [bad_start], cal, label_horizon=6)


def test_walk_forward_fold_geometry(cal10y):
    cal = cal10y
    folds = walk_forward_folds(cal, "2015-01-01", "2022-12-31",
                               train_years=3, val_months=6, step_months=6,
                               label_horizon=6)
    assert len(folds) > 3
    for f in folds:
        # purge 恰好卡在边界：train_end + 6 个交易日 = val_start 前一交易日
        assert cal.shift(f.train_end, 6) == cal.shift(f.val_start, -1)
        assert f.train_start < f.train_end < f.val_start <= f.val_end
    # 滚动步长 6 个月
    assert folds[1].val_start.month in {(folds[0].val_start.month + 6 - 1) % 12 + 1}


def test_oos_never_visible(cal10y):
    oos = "2021-01-01"
    folds = walk_forward_folds(cal10y, "2015-01-01", "2026-12-31",
                               oos_start=oos)
    assert len(folds) > 0
    for f in folds:
        assert f.val_end < pd.Timestamp(oos)

    with pytest.raises(SealedOOSError):
        sealed_oos_window(cal10y, oos, "2026-12-31")
    s, e = sealed_oos_window(cal10y, oos, "2026-12-31", unseal=True)
    assert s == pd.Timestamp("2021-01-01") and e <= pd.Timestamp("2026-12-31")
