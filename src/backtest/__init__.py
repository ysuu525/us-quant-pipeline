"""回测读数的薄包装层。统计量一律复用 `crsp_pipeline.signal_eval`，不另起炉灶。"""
from backtest.money import daily_rank_ic, fold_sign_count, nw_summary, run_frozen_money

__all__ = ["daily_rank_ic", "nw_summary", "fold_sign_count", "run_frozen_money"]
