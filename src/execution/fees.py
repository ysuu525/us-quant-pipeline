"""费用会计与主指标 C。

费率来源：`HANDOFF.md` §11.4a（Alpaca Elite + 监管费透传），
协议 `experiments/cost_pilot_protocol_v1_draft.md` 附录 A 同表。

**这里算的是公示费率下的推算值，只能用于事前预算。**
协议 §3 明确规定：最终判读必须用成交确认与账户流水上的**实际全包费用**，
不得用本模块的推算值代替。本模块给 §7 的 paper 预演、订单预算、
以及 fill 到齐前的占位使用。

主指标（HANDOFF §11.2 裁定 6）::

    C = sign * (fill - CRSP_DlyOpen) / CRSP_DlyOpen * 1e4 + fees / notional * 1e4
    sign = +1 (买) / -1 (卖)          # 买贵为正成本、卖便宜为正成本

**决策收盘 -> 开盘的漂移不进 C**（回测也吃了这一段，计入即重复相加），
只作诊断字段 `drift_bp` 单独返回。
"""
from __future__ import annotations

import math
from typing import Literal

# ---------------------------------------------------------------- 费率常量
# Alpaca Elite（`time_in_force=opg` 只对 Elite Smart Router 开放，佣金无法回避）
COMMISSION_ALL_IN_PER_SHARE = 0.0040       # 默认方案：确定数，含各类场费/回扣，不含监管费
COMMISSION_COST_PLUS_PER_SHARE = 0.0025    # 可选方案：期望更低，但竞价场费/回扣未知

PLAN_ALL_IN = "all_in"
PLAN_COST_PLUS = "cost_plus"
COMMISSION_PER_SHARE = {
    PLAN_ALL_IN: COMMISSION_ALL_IN_PER_SHARE,
    PLAN_COST_PLUS: COMMISSION_COST_PLUS_PER_SHARE,
}

# 监管费（透传，双方案相同）
SEC_SECTION31_RATE = 0.0000206             # x 成交额，仅卖出（= 0.206bp）
FINRA_TAF_PER_SHARE = 0.000195             # /股，仅卖出
FINRA_TAF_CAP = 9.79                       # 单笔上限
FINRA_CAT_PER_SHARE = 0.000003             # /股，买卖双边

# 上限首次真正咬人的股数：ceil(9.79 / 0.000195) = 50206。
# HANDOFF §11.4a 写的「50,205 股封顶」只有在把 TAF 四舍五入到分之后才成立
# （50,205 股的未舍入 TAF = 9.789975，round(.,2) = 9.79）。本模块**不做分位舍入**，
# 因此上限从 50,206 股起生效。见报告中的口径说明。
FINRA_TAF_CAP_SHARES = 50_206

SIDE_BUY = "buy"
SIDE_SELL = "sell"
Side = Literal["buy", "sell"]

FEE_KEYS = ("commission", "sec", "taf", "cat", "total")


def _norm_side(side: str) -> str:
    s = str(side).strip().lower()
    if s not in (SIDE_BUY, SIDE_SELL):
        raise ValueError(f"side 只能是 {SIDE_BUY!r}/{SIDE_SELL!r}，收到 {side!r}")
    return s


def commission_per_share(plan: str = PLAN_ALL_IN) -> float:
    """按方案取每股佣金。协议 §9：方案必须在第一笔成交前定死，不得中途切换。"""
    try:
        return COMMISSION_PER_SHARE[plan]
    except KeyError:
        raise ValueError(
            f"plan 只能是 {PLAN_ALL_IN!r}/{PLAN_COST_PLUS!r}，收到 {plan!r}") from None


def finra_taf(shares: float) -> float:
    """FINRA TAF：$0.000195/股，仅卖出，单笔上限 $9.79。"""
    return min(FINRA_TAF_PER_SHARE * float(shares), FINRA_TAF_CAP)


def order_fees(side: str, shares: float, notional: float,
               plan: str = PLAN_ALL_IN) -> dict[str, float]:
    """单笔订单的公示费率全包，返回 {commission, sec, taf, cat, total}（美元）。

    - 佣金：按股，买卖双边；
    - SEC Section 31：按成交额，**仅卖出**；
    - FINRA TAF：按股，**仅卖出**，单笔上限 $9.79；
    - FINRA CAT：按股，**买卖双边**。
    """
    s = _norm_side(side)
    shares = float(shares)
    notional = float(notional)
    if shares < 0:
        raise ValueError("shares 不得为负")
    if notional < 0:
        raise ValueError("notional 不得为负")

    is_sell = s == SIDE_SELL
    commission = commission_per_share(plan) * shares
    sec = SEC_SECTION31_RATE * notional if is_sell else 0.0
    taf = finra_taf(shares) if is_sell else 0.0
    cat = FINRA_CAT_PER_SHARE * shares
    return {
        "commission": commission,
        "sec": sec,
        "taf": taf,
        "cat": cat,
        "total": commission + sec + taf + cat,
    }


def fees_bp(side: str, shares: float, notional: float,
            plan: str = PLAN_ALL_IN) -> float:
    """全包费用占名义额的 bp。notional<=0 时返回 nan（无法定义）。"""
    fees = order_fees(side, shares, notional, plan)
    if notional <= 0:
        return math.nan
    return fees["total"] / float(notional) * 1e4


def drift_bp(crsp_dly_open: float, prev_close: float) -> float:
    """诊断字段：决策收盘 -> 开盘的漂移。

    **不进 C**（协议 §0「不进入 C 的东西」、HANDOFF §11.2 裁定 6）：
    回测的建仓价就是 `DlyOpen`，这一段回测同样吃了，计入即重复相加。
    """
    if prev_close is None or not math.isfinite(float(prev_close)) or float(prev_close) == 0.0:
        return math.nan
    return (float(crsp_dly_open) - float(prev_close)) / float(prev_close) * 1e4


def cost_bp(side: str, fill_price: float, crsp_dly_open: float,
            shares: float, notional: float, plan: str = PLAN_ALL_IN,
            *, prev_close: float | None = None) -> dict[str, float]:
    """逐笔主指标 C（bp）及其分解。

    返回键：

    - ``exec_bp``   —— `sign * (fill - DlyOpen)/DlyOpen * 1e4`，执行偏差段；
    - ``fees_bp``   —— 全包费用 / 名义额 * 1e4；
    - ``cost_bp``   —— 主指标 C = exec_bp + fees_bp；
    - ``drift_bp``  —— 诊断，**不含在 C 内**；未给 `prev_close` 时为 nan；
    - ``fee_usd_*`` —— 逐项费用（美元），便于按协议 §8 做「费用 vs 执行偏差」分解。

    `sign = +1` 买 / `-1` 卖：买贵为正成本，卖便宜为正成本。
    """
    s = _norm_side(side)
    sign = 1.0 if s == SIDE_BUY else -1.0
    open_px = float(crsp_dly_open)
    if not math.isfinite(open_px) or open_px == 0.0:
        raise ValueError("crsp_dly_open 必须是非零有限数（回测建仓价）")

    exec_bp = sign * (float(fill_price) - open_px) / open_px * 1e4
    fees = order_fees(s, shares, notional, plan)
    f_bp = math.nan if float(notional) <= 0 else fees["total"] / float(notional) * 1e4

    return {
        "exec_bp": exec_bp,
        "fees_bp": f_bp,
        "cost_bp": exec_bp + f_bp,
        "drift_bp": drift_bp(open_px, prev_close) if prev_close is not None else math.nan,
        "fee_usd_commission": fees["commission"],
        "fee_usd_sec": fees["sec"],
        "fee_usd_taf": fees["taf"],
        "fee_usd_cat": fees["cat"],
        "fee_usd_total": fees["total"],
    }
