"""美股 DL 量化管线（规范 v1.3）——数据/标签/切分层。

模块对应规范章节：
- calendar  交易日历（§7/§9 的日期算术基础）
- universe  选股面板：静态 CIZ 筛选 + 流动性条件（§2/§3）
- labels    execution-return 标签引擎（§4）
- adjust    事件累计复权 + DlyFacPrc 双路验证（§5）
- cleaning  BA 统计、lookback 缺口排除（§9）
- splits    walk-forward + purge 断言 + 封存 OOS（§7）
"""

from . import adjust, calendar, cleaning, config, labels, splits, universe  # noqa: F401

__all__ = ["adjust", "calendar", "cleaning", "config", "labels", "splits", "universe"]
