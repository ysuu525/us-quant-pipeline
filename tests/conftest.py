import numpy as np
import pandas as pd
import pytest

from crsp_pipeline.calendar import TradingCalendar


@pytest.fixture
def cal():
    """合成交易日历：2020–2021 的工作日（周末即「休市日不生成行」）。"""
    return TradingCalendar(pd.bdate_range("2020-01-01", "2021-12-31"))


def make_sec(dates, opens=None, closes=None, rets=None, delflg=None, **extra):
    """构造单只证券的日行 DataFrame（index=日期）。缺省列填 NaN/''。"""
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    n = len(idx)

    def _col(v, fill):
        if v is None:
            return [fill] * n
        return list(v)

    df = pd.DataFrame(
        {
            "DlyOpen": _col(opens, np.nan),
            "DlyClose": _col(closes, np.nan),
            "DlyRet": _col(rets, np.nan),
            "DlyDelFlg": _col(delflg, ""),
        },
        index=idx,
    )
    for k, v in extra.items():
        df[k] = _col(v, np.nan)
    return df
