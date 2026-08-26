"""成本模型（§8）：分段费率、监管过手费、滑点、替换成本。数字对应 v1.3 §8 表。"""

import pytest

from crsp_pipeline.costs import CHANNELS, RegulatoryFees, order_cost, replacement_cost

NO_REG = RegulatoryFees(sec_rate=0.0, taf_per_share=0.0, taf_cap=0.0)


def test_moomoo_whole_fixed_fee():
    # $600/单 → 0.99 固定费 = 16.5bp（§8 表）
    c = order_cost(CHANNELS["moomoo_au_whole"], 600.0, 12.0, "buy", slippage_bp=0, regulatory=NO_REG)
    assert c["commission"] == pytest.approx(0.99)
    assert c["commission"] / 600.0 == pytest.approx(16.5e-4)


def test_moomoo_frac_capped_and_exempt():
    # $185/单：min(0.99% × 185, 0.99) = 0.99；豁免 SEC/TAF 过手费
    reg = RegulatoryFees()  # 非零监管费率
    c = order_cost(CHANNELS["moomoo_au_frac"], 185.0, 3.7, "sell", slippage_bp=0, regulatory=reg)
    assert c["commission"] == pytest.approx(0.99)
    assert c["regulatory"] == 0.0
    # 小单未触上限：$50 → 0.495
    c2 = order_cost(CHANNELS["moomoo_au_frac"], 50.0, 1.0, "buy", slippage_bp=0, regulatory=reg)
    assert c2["commission"] == pytest.approx(0.495)


def test_ibkr_tiered_min_and_cap():
    fs = CHANNELS["ibkr_au_tiered"]
    # min $0.35
    assert fs.commission(3000.0, 50.0) == pytest.approx(0.35)
    # 按股：2000 股 → $7
    assert fs.commission(50000.0, 2000.0) == pytest.approx(7.0)
    # 上限 1%：$300 的单，2000 股 → min(7, 3) = 3
    assert fs.commission(300.0, 2000.0) == pytest.approx(3.0)


def test_alpaca_free_but_sell_side_regulatory():
    reg = RegulatoryFees(sec_rate=27.80e-6, taf_per_share=0.000166, taf_cap=8.30)
    buy = order_cost(CHANNELS["alpaca"], 1000.0, 20.0, "buy", slippage_bp=0, regulatory=reg)
    assert buy["total"] == 0.0
    sell = order_cost(CHANNELS["alpaca"], 1000.0, 20.0, "sell", slippage_bp=0, regulatory=reg)
    assert sell["commission"] == 0.0
    assert sell["regulatory"] == pytest.approx(27.80e-6 * 1000 + 0.000166 * 20)


def test_taf_cap_binds():
    reg = RegulatoryFees(sec_rate=0.0, taf_per_share=0.000166, taf_cap=8.30)
    c = order_cost(CHANNELS["alpaca"], 1e6, 100_000.0, "sell", slippage_bp=0, regulatory=reg)
    assert c["regulatory"] == pytest.approx(8.30)


def test_slippage_tier():
    c = order_cost(CHANNELS["alpaca"], 1000.0, 20.0, "buy", slippage_bp=30, regulatory=NO_REG)
    assert c["slippage"] == pytest.approx(3.0)
    assert c["total"] == pytest.approx(3.0)


def test_replacement_is_sell_plus_buy():
    # 缓冲区替换制：换一只 = 卖一单 + 买一单（§8 写死）
    r = replacement_cost(CHANNELS["moomoo_au_whole"], 600.0, 12.0, 600.0, 12.0,
                         slippage_bp=0, regulatory=NO_REG)
    assert r["commission"] == pytest.approx(2 * 0.99)
    assert r["total"] == pytest.approx(2 * 0.99)
