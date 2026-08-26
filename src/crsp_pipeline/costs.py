"""成本模型（规范 §8）：分段函数，非统一 bp。

    Cost = fee_schedule(order_value, order_type) + 监管结算费 + 价差/滑点档

- fee_schedule 按账户后台实际费率表写成分段函数；下方通道预设取自 v1.3
  §8 的初查数字（2026-08-25 网页核实），**开户后以账户后台费率表原文为准
  重新配置**（§8 行动项）；
- 监管结算费（SEC fee + FINRA TAF，卖出侧）费率随时间调整，默认值为公开
  档位，同样以实际扣费为准可配置；moomoo 碎股豁免过手费（passthrough_exempt）；
- 滑点三档 5/15/30bp（§8）；碎股不参与开盘竞价 → 执行假设为「t+1 open +
  滑点档」的盘初市价单，滑点档后续用开盘价 vs 盘后 5 分钟 VWAP 校准；
- 换手假设写死：缓冲区替换制（换一只 = 卖一单 + 买一单），不做全组合配平。
"""

from __future__ import annotations

from dataclasses import dataclass

SLIPPAGE_TIERS_BP = (5.0, 15.0, 30.0)


@dataclass(frozen=True)
class FeeSchedule:
    """佣金分段函数。kind:

    - 'fixed'        整股每单固定费（moomoo AU 整股 US$0.99/单）
    - 'value_capped' 按金额计费带上限：min(rate×value, cap)（moomoo AU 碎股）
    - 'per_share'    按股计费：min(max(per_share×shares, min_fee), max_pct×value)
                     （IBKR AU tiered，含碎股）
    - 'free'         零佣金（Alpaca）
    """
    name: str
    kind: str
    fixed: float = 0.0
    rate: float = 0.0
    cap: float | None = None
    per_share: float = 0.0
    min_fee: float = 0.0
    max_pct: float | None = None
    passthrough_exempt: bool = False  # 豁免 SEC/TAF 等过手费

    def commission(self, value: float, shares: float) -> float:
        if self.kind == "fixed":
            return self.fixed
        if self.kind == "value_capped":
            fee = self.rate * value
            return min(fee, self.cap) if self.cap is not None else fee
        if self.kind == "per_share":
            fee = max(self.per_share * shares, self.min_fee)
            if self.max_pct is not None:
                fee = min(fee, self.max_pct * value)
            return fee
        if self.kind == "free":
            return 0.0
        raise ValueError(f"unknown fee kind: {self.kind}")


@dataclass(frozen=True)
class RegulatoryFees:
    """卖出侧监管过手费。默认值为公开档位（随监管调整，以实际扣费为准配置）：
    SEC fee 按卖出金额、FINRA TAF 按卖出股数（设单笔上限）。"""
    sec_rate: float = 27.80e-6       # $27.80 / $1m 卖出金额档
    taf_per_share: float = 0.000166  # $/股（卖出）
    taf_cap: float = 8.30            # 单笔上限

    def fee(self, value: float, shares: float, side: str) -> float:
        if side != "sell":
            return 0.0
        return self.sec_rate * value + min(self.taf_per_share * shares, self.taf_cap)


# v1.3 §8 通道预设（开户后以费率表原文重配）
CHANNELS: dict[str, FeeSchedule] = {
    "moomoo_au_whole": FeeSchedule("moomoo_au_whole", "fixed", fixed=0.99),
    "moomoo_au_frac": FeeSchedule("moomoo_au_frac", "value_capped",
                                  rate=0.0099, cap=0.99, passthrough_exempt=True),
    "ibkr_au_tiered": FeeSchedule("ibkr_au_tiered", "per_share",
                                  per_share=0.0035, min_fee=0.35, max_pct=0.01),
    "alpaca": FeeSchedule("alpaca", "free"),
}


def order_cost(
    schedule: FeeSchedule,
    value: float,
    shares: float,
    side: str,
    slippage_bp: float = 30.0,
    regulatory: RegulatoryFees | None = None,
) -> dict:
    """单笔订单总成本分解：佣金 + 监管结算费（卖出侧，可豁免）+ 滑点。

    返回 {'commission', 'regulatory', 'slippage', 'total'}，单位美元。
    """
    if side not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'")
    reg = regulatory if regulatory is not None else RegulatoryFees()
    commission = schedule.commission(value, shares)
    reg_fee = 0.0 if schedule.passthrough_exempt else reg.fee(value, shares, side)
    slippage = value * slippage_bp / 1e4
    return {
        "commission": commission,
        "regulatory": reg_fee,
        "slippage": slippage,
        "total": commission + reg_fee + slippage,
    }


def replacement_cost(
    schedule: FeeSchedule,
    sell_value: float,
    sell_shares: float,
    buy_value: float,
    buy_shares: float,
    slippage_bp: float = 30.0,
    regulatory: RegulatoryFees | None = None,
) -> dict:
    """缓冲区替换一次的成本：卖一单 + 买一单（§8 换手假设，写死）。"""
    s = order_cost(schedule, sell_value, sell_shares, "sell", slippage_bp, regulatory)
    b = order_cost(schedule, buy_value, buy_shares, "buy", slippage_bp, regulatory)
    return {k: s[k] + b[k] for k in s}
