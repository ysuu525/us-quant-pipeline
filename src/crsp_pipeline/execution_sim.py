"""执行层模拟（规范 §7.5 执行层 + §8）——$3000 约束下是否可实施。

全约束模拟：全部成本（costs.py 分段函数 + 滑点档）、仓位数上限、缓冲区
替换、score 改善阈值触发、t+1 成交、unfillable/退市处理。指标：净值对
VWRETD 的超额、最大回撤、年换手、成本分解。

**不做显著性声明**（§7.5 预承诺）：执行层结论只看实际路径是否落在
Monte Carlo 选择噪声带内且带中位为正。小广度、周频、短 OOS 的统计功效
不足以区分运气与边际，此局限预先写入。

时序近似（Phase 2 冻结口径，Phase 5 用真实 open 精化）：t 日收盘出信号，
t+1 开盘成交。模拟中交易记在 t+1 日**当日收益之前**——买入腿享有 t+1 全天
close-to-close 收益，卖出腿不享有；open 与前收之间的隔夜价差并入滑点档
（§8：碎股为盘初市价单，滑点档以开盘价 vs 开盘后 VWAP 校准）。此近似与
§4 标签的 open 口径差在报告中单独量化。

换手：缓冲区替换制（换一只 = 卖一单 + 买一单），不做全组合等权配平（§8）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .costs import FeeSchedule, RegulatoryFees, order_cost


# ---------------------------------------------------------------- 绩效工具

def max_drawdown(nav: pd.Series) -> float:
    """最大回撤（正数，如 0.35 = 回撤 35%）。"""
    running_max = nav.cummax()
    return float((1.0 - nav / running_max).max())


def excess_total_return(nav: pd.Series, market_ret: pd.Series) -> float:
    """区间总收益对基准（VWRETD 复合）的超额：(1+R_p)/(1+R_m) − 1。"""
    m = market_ret.reindex(nav.index).fillna(0.0)
    bench = float((1.0 + m).prod())
    port = float(nav.iloc[-1] / nav.iloc[0])
    return port / bench - 1.0


# ---------------------------------------------------------------- 模拟器

@dataclass
class SimResult:
    nav: pd.Series
    orders: pd.DataFrame                 # exec_date/PERMNO/side/value/成本分解
    cost_totals: dict
    final_positions: dict
    unpriced_days: int = 0               # 持仓日收益缺失按 0 处理的次数（报告用）
    turnover_annual: float = np.nan

    def metrics(self, market_ret: pd.Series | None = None) -> dict:
        out = {
            "total_return": float(self.nav.iloc[-1] / self.nav.iloc[0] - 1.0),
            "max_drawdown": max_drawdown(self.nav),
            "turnover_annual": self.turnover_annual,
            "cost_totals": self.cost_totals,
            "unpriced_days": self.unpriced_days,
        }
        if market_ret is not None:
            out["excess_vs_benchmark"] = excess_total_return(self.nav, market_ret)
        return out


def simulate_portfolio(
    scores: pd.DataFrame,
    daily_returns: pd.DataFrame,
    rebalance_dates,
    n_positions: int,
    schedule: FeeSchedule,
    buffer_rank: int | None = None,
    score_improvement_threshold: float = 0.0,
    initial_capital: float = 3000.0,
    slippage_bp: float = 30.0,
    regulatory: RegulatoryFees | None = None,
    assumed_price: float = 50.0,
    prices: pd.DataFrame | None = None,
    permno_col: str = "PERMNO",
    date_col: str = "signal_date",
    score_col: str = "score",
) -> SimResult:
    """缓冲区替换制组合模拟。

    Parameters
    ----------
    scores : (signal_date, PERMNO, score) 长表，只需含各调仓信号日的行；
        信号日不在表中的持仓视为「已出 universe」→ 强制卖出。
    daily_returns : 宽表 (index=交易日, columns=PERMNO) 的 ``DlyRet``，来自
        **未过滤全量面板**（退市终值收益已在其记录内，之后为 NaN）。
    rebalance_dates : 信号日列表（须为 daily_returns.index 中的日期）。
    buffer_rank : 持仓名次跌出该值才成为替换候选，默认 2×n_positions。
    score_improvement_threshold : 新股 score − 持仓 score 超过该值才触发替换。
    prices / assumed_price : 计算按股佣金所需股数；无价格表时用 assumed_price。
    """
    if buffer_rank is None:
        buffer_rank = 2 * n_positions
    dates = daily_returns.index
    rebalance_set = {pd.Timestamp(t) for t in rebalance_dates}
    scores = scores.copy()
    scores[date_col] = pd.to_datetime(scores[date_col])
    scores_by_date = {t: g.set_index(permno_col)[score_col].sort_values(ascending=False)
                      for t, g in scores.groupby(date_col)}
    last_valid = daily_returns.apply(lambda c: c.last_valid_index())

    cash = float(initial_capital)
    positions: dict = {}          # permno -> 市值
    pending: list | None = None   # 信号日排好的交易，下一交易日执行
    nav_out, orders, unpriced = {}, [], 0
    cost_totals = {"commission": 0.0, "regulatory": 0.0, "slippage": 0.0, "total": 0.0}
    total_sell_value = 0.0

    def _shares(pn, d, value):
        px = assumed_price
        if prices is not None and pn in prices.columns:
            p = prices.loc[:d, pn].dropna()
            if len(p):
                px = float(p.iloc[-1])
        return value / px

    def _execute(trades, d):
        nonlocal cash, total_sell_value
        # 先全部卖出，再等分买入（换一只=卖一单+买一单；仅新腿交易，不配平）
        buys = [t for t in trades if t[0] == "buy"]
        for side, pn in trades:
            if side != "sell":
                continue
            value = positions.pop(pn)
            c = order_cost(schedule, value, _shares(pn, d, value), "sell",
                           slippage_bp, regulatory)
            cash += value - c["total"]
            total_sell_value += value
            _log(d, pn, "sell", value, c)
        if buys:
            alloc = cash / len(buys)
            for side, pn in buys:
                c = order_cost(schedule, alloc, _shares(pn, d, alloc), "buy",
                               slippage_bp, regulatory)
                value = alloc - c["total"]
                if value <= 0:
                    continue
                positions[pn] = value
                cash -= alloc
                _log(d, pn, "buy", value, c)

    def _log(d, pn, side, value, c):
        for k in cost_totals:
            cost_totals[k] += c[k]
        orders.append({"exec_date": d, "PERMNO": pn, "side": side, "value": value, **c})

    def _plan(t):
        ranked = scores_by_date.get(t)
        if ranked is None:
            return [("sell", pn) for pn in list(positions)]  # 无信号 → 全平（不应发生，防御）
        rank_of = {pn: i + 1 for i, pn in enumerate(ranked.index)}
        trades = []
        held = set(positions)
        # 1) 出 universe / 无 score → 强制卖出
        forced = [pn for pn in held if pn not in rank_of]
        trades += [("sell", pn) for pn in forced]
        held -= set(forced)
        # 2) 缓冲区替换：名次跌出 buffer 且最优未持有候选改善超阈值
        candidates = [pn for pn in ranked.index if pn not in held]
        for pn in sorted(held, key=lambda p: -rank_of[p]):  # 最差的先看
            if rank_of[pn] <= buffer_rank or not candidates:
                continue
            best = candidates[0]
            if ranked[best] - ranked[pn] > score_improvement_threshold:
                trades += [("sell", pn), ("buy", best)]
                held.discard(pn)
                held.add(best)
                candidates.pop(0)
        # 3) 补空位（含初始建仓与强制卖出后的空位）
        n_vac = n_positions - len(held)
        for pn in candidates[:max(n_vac, 0)]:
            trades.append(("buy", pn))
            held.add(pn)
        return trades

    for d in dates:
        if pending is not None:
            _execute(pending, d)
            pending = None
        # 当日收益（交易日的买入腿享有全天收益，见模块 docstring 的时序近似）
        for pn in list(positions):
            lv = last_valid.get(pn)
            if lv is not None and d > lv:
                # 退市终值已在其记录日复合进市值 → 转现金，腾出仓位
                cash += positions.pop(pn)
                continue
            r = daily_returns.at[d, pn] if pn in daily_returns.columns else np.nan
            if pd.isna(r):
                unpriced += 1
            else:
                positions[pn] *= 1.0 + float(r)
        nav_out[d] = cash + sum(positions.values())
        if d in rebalance_set:
            pending = _plan(d)

    nav = pd.Series(nav_out).sort_index()
    years = max(len(dates) / 252.0, 1e-9)
    turnover = total_sell_value / max(nav.mean(), 1e-9) / years
    return SimResult(
        nav=nav,
        orders=pd.DataFrame(orders),
        cost_totals=cost_totals,
        final_positions=dict(positions),
        unpriced_days=unpriced,
        turnover_annual=turnover,
    )


# ---------------------------------------------------------------- 噪声带

def _jitter_scores_one_date(ranked: pd.Series, jitter: float, rng) -> pd.Series:
    """邻近名次随机重排：名次加 U(0, jitter) 噪声后重排序，score 多重集不变
    （每只股票取其扰动后名次位置上的原 score）。"""
    k = len(ranked)
    noisy = np.arange(k) + rng.uniform(0.0, jitter, size=k)
    new_order = ranked.index.to_numpy()[np.argsort(noisy, kind="stable")]
    return pd.Series(ranked.to_numpy(), index=new_order)


def selection_noise_band(
    scores: pd.DataFrame,
    n_draws: int = 200,
    rank_jitter: float = 5.0,
    seed: int = 0,
    market_ret: pd.Series | None = None,
    permno_col: str = "PERMNO",
    date_col: str = "signal_date",
    score_col: str = "score",
    band_quantiles: tuple[float, ...] = (0.05, 0.5, 0.95),
    **sim_kwargs,
) -> dict:
    """选择噪声带（§7.5）：每个调仓日在 score 邻近名次内随机重抽持仓，
    ≥200 次 Monte Carlo，报告净值分布带。

    结论口径（预承诺）：只看 (a) 实际路径终值在噪声带内的分位、
    (b) 带中位终值（给了 market_ret 则为对基准的超额）是否为正。
    不做显著性声明。
    """
    rng = np.random.default_rng(seed)
    base = simulate_portfolio(scores, permno_col=permno_col, date_col=date_col,
                              score_col=score_col, **sim_kwargs)

    sc = scores.copy()
    sc[date_col] = pd.to_datetime(sc[date_col])
    paths = {}
    for i in range(n_draws):
        parts = []
        for t, g in sc.groupby(date_col):
            ranked = g.set_index(permno_col)[score_col].sort_values(ascending=False)
            jit = _jitter_scores_one_date(ranked, rank_jitter, rng)
            parts.append(pd.DataFrame({
                date_col: t, permno_col: jit.index, score_col: jit.to_numpy(),
            }))
        draw_scores = pd.concat(parts, ignore_index=True)
        r = simulate_portfolio(draw_scores, permno_col=permno_col,
                               date_col=date_col, score_col=score_col, **sim_kwargs)
        paths[i] = r.nav
    paths = pd.DataFrame(paths)

    finals = paths.iloc[-1]
    band = paths.quantile(band_quantiles, axis=1).T  # index=日期, columns=分位
    actual_final = base.nav.iloc[-1]
    pct = float((finals < actual_final).mean())

    if market_ret is not None:
        m = market_ret.reindex(paths.index).fillna(0.0)
        bench = float((1.0 + m).prod())
        median_final_metric = float(finals.median() / paths.iloc[0].median() / bench - 1.0)
    else:
        median_final_metric = float(finals.median() / paths.iloc[0].median() - 1.0)

    return {
        "actual": base,
        "paths": paths,
        "band": band,
        "actual_final_percentile": pct,
        "actual_in_band": bool(band_quantiles[0] <= pct <= band_quantiles[-1]),
        "median_final_metric": median_final_metric,
        "median_positive": bool(median_final_metric > 0),
    }
