"""美股 DL 量化管线（规范 v1.3）——数据/标签/切分层。

模块对应规范章节：
- calendar  交易日历（§7/§9 的日期算术基础）
- universe  选股面板：静态 CIZ 筛选 + 流动性条件（§2/§3）
- labels    execution-return 标签引擎（§4）
- adjust    事件累计复权 + DlyFacPrc 双路验证（§5）
- cleaning  BA 统计、lookback 缺口排除（§9）
- splits    walk-forward + purge 断言 + 封存 OOS（§7）
- factors   自建归因因子 + beta/波动率暴露（§1/§7.5）
- signal_eval    信号层评估：RankIC / NW t / 十分位价差 / 中性化（§7.5）
- costs     分段成本模型 + 通道预设（§8）
- execution_sim  执行层模拟 + Monte Carlo 选择噪声带（§7.5/§8）
"""

from . import (  # noqa: F401
    adjust, calendar, cleaning, config, costs, execution_sim, factors,
    labels, signal_eval, splits, universe,
)

__all__ = [
    "adjust", "calendar", "cleaning", "config", "costs", "execution_sim",
    "factors", "labels", "signal_eval", "splits", "universe",
]
