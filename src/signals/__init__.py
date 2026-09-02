"""信号层（信号 #1 = Kronos 之外的**其它**横截面信号）。

结构：

- :mod:`signals.base`               —— 冻结规格 ``SignalSpec`` + 抽象基类 ``Signal`` + 前视自查；
- :mod:`signals.overnight_intraday` —— 信号 #2「隔夜/日内分解」（LPS 2019 / BBM 2026 口径）；
- :mod:`signals.kronos_adapter`     —— 只读已消耗开发折的 Kronos 分数与标签（按需 import，
  会拉起 ``crsp_pipeline.sealed`` 的读取守卫）。

``kronos_adapter`` 不在此处导入，保持本包在纯合成数据上的可测性。
"""
from __future__ import annotations

from .base import (
    OUTPUT_COLUMNS,
    REQUIRED_PANEL_COLUMNS,
    Signal,
    SignalSpec,
    assert_no_lookahead,
)
from .overnight_intraday import (
    LOOKBACK,
    MAX_SPAN_DAYS,
    MIN_VALID,
    SPEC as OVERNIGHT_INTRADAY_SPEC,
    OvernightIntradaySignal,
    decompose,
)

__all__ = [
    "OUTPUT_COLUMNS",
    "REQUIRED_PANEL_COLUMNS",
    "Signal",
    "SignalSpec",
    "assert_no_lookahead",
    "LOOKBACK",
    "MIN_VALID",
    "MAX_SPAN_DAYS",
    "OVERNIGHT_INTRADAY_SPEC",
    "OvernightIntradaySignal",
    "decompose",
]
