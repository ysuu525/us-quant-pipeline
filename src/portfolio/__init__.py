"""组合构造层：把冻结的变现构造从脚本里抽出来，供多信号合成复用。

- `construction.frozen_long_only_returns`——与
  `scripts/compare_arms_money.py::arm_returns`（第 62–122 行）**逐位一致**的
  唯一权威实现。CLAUDE.md §五：组合构造必须先冻结再比臂。
- `combine`——两条零自由参数、先验固定的合成规则（秩相加 / 慢信号筛子）。

本包不引入任何新统计量，也不做任何判读；判据在预注册文件里。
"""
from portfolio.combine import rank_sum_equal_weight, slow_filter, spearman_by_day
from portfolio.construction import frozen_long_only_returns, scores_frame_to_by_day

__all__ = [
    "frozen_long_only_returns",
    "scores_frame_to_by_day",
    "rank_sum_equal_weight",
    "slow_filter",
    "spearman_by_day",
]
