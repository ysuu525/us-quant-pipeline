"""`scripts/ict_pattern_probe.py::compute_patterns` 的逐条定义验证（全合成 K 线）。

预注册 `experiments/ict_pattern_probe_prereg_v1.md` §3 的 6 个标记 × 看涨/看跌，
每个至少一正例一反例；外加一条「不得用未来信息」的断言：把 t+1 及之后的 K 线
整段替换，t 日及之前的所有标记必须逐位不变。

固定参数（不得在测试里改，改了就是改预注册）：L=20、ATR14、k=1.5、
订单块回看 5、有效期 20、实体占比 0.6、OTE [0.21, 0.38]。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "ict_pattern_probe", REPO / "scripts" / "ict_pattern_probe.py")
ICT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ICT)

compute_patterns = ICT.compute_patterns
L = ICT.L_SWING          # 20
K = ICT.K_DISP           # 1.5
OBLB = ICT.OB_LOOKBACK   # 5
VAL = ICT.VALIDITY       # 20
MINH = ICT.MIN_HIST_P5P6  # 45


# ----------------------------------------------------------------- 造 K 线工具


def flat(n: int, price: float = 100.0, half: float = 0.5):
    """n 根「窄幅横盘」K 线：O=C=price，H=price+half，L=price−half。"""
    o = np.full(n, price)
    c = np.full(n, price)
    h = np.full(n, price + half)
    lo = np.full(n, price - half)
    return o, h, lo, c


def cat(*bars):
    return tuple(np.concatenate([b[i] for b in bars]) for i in range(4))


def one(o, h, lo, c):
    return (np.array([o], float), np.array([h], float),
            np.array([lo], float), np.array([c], float))


def base(n=60):
    """够长的横盘前缀，保证 H20/L20/ATR14 与 MIN_HIST_P5P6 全部可算。"""
    return flat(n)


# --------------------------------------------------------------------- P1 扫流动性


def test_p1_bull_positive_and_negative():
    pre = base(60)
    # 正例：插针跌破 L20(=99.5) 后收在其上方
    good = one(100.0, 100.3, 98.0, 100.1)
    r = compute_patterns(*cat(pre, good))
    assert r["P1_bull"][-1] and r["computable"][-1]
    # 反例甲：跌破后收在下方（未收回）
    bad1 = one(100.0, 100.3, 98.0, 98.5)
    assert not compute_patterns(*cat(pre, bad1))["P1_bull"][-1]
    # 反例乙：根本没跌破 L20
    bad2 = one(100.0, 100.3, 99.7, 100.1)
    assert not compute_patterns(*cat(pre, bad2))["P1_bull"][-1]


def test_p1_bear_positive_and_negative():
    pre = base(60)
    good = one(100.0, 102.0, 99.8, 100.1)      # 上破 H20(=100.5) 后收回其下
    r = compute_patterns(*cat(pre, good))
    assert r["P1_bear"][-1]
    bad1 = one(100.0, 102.0, 99.8, 101.5)      # 上破后收在上方
    assert not compute_patterns(*cat(pre, bad1))["P1_bear"][-1]
    bad2 = one(100.0, 100.4, 99.8, 100.1)      # 未上破
    assert not compute_patterns(*cat(pre, bad2))["P1_bear"][-1]


def test_p1_needs_full_lookback():
    """前 L 根没有 H20/L20 → 不可触发、也不可算。"""
    o, h, lo, c = base(10)
    r = compute_patterns(o, h, lo, c)
    assert not r["P1_bull"].any() and not r["P1_bear"].any()
    assert not r["computable"].any()


# --------------------------------------------------------------------- P2 结构突破


def test_p2_bull_and_bear():
    pre = base(60)
    up = one(100.0, 101.5, 99.9, 101.2)        # C > H20 = 100.5
    dn = one(100.0, 100.1, 98.0, 98.4)         # C < L20 = 99.5
    non = one(100.0, 100.4, 99.6, 100.0)       # 夹在中间
    assert compute_patterns(*cat(pre, up))["P2_bull"][-1]
    assert not compute_patterns(*cat(pre, up))["P2_bear"][-1]
    assert compute_patterns(*cat(pre, dn))["P2_bear"][-1]
    assert not compute_patterns(*cat(pre, dn))["P2_bull"][-1]
    assert not compute_patterns(*cat(pre, non))["P2_bull"][-1]
    assert not compute_patterns(*cat(pre, non))["P2_bear"][-1]


def test_p2_boundary_is_strict():
    """C == H20 不算突破（定义是严格 >）。"""
    pre = base(60)
    eq = one(100.0, 100.6, 99.9, 100.5)        # C 恰等于 H20
    assert not compute_patterns(*cat(pre, eq))["P2_bull"][-1]


# ------------------------------------------------------------------- P3 FVG 失衡


def test_p3_bull_and_bear():
    pre = base(60)                              # 末三根都是 H=100.5 / L=99.5
    # 看涨：Low_t > High_{t-2}
    mid = one(100.0, 100.5, 99.5, 100.0)
    good = one(101.5, 102.0, 101.0, 101.8)      # L=101.0 > H_{t-2}=100.5
    r = compute_patterns(*cat(pre, mid, good))
    assert r["P3_bull"][-1] and not r["P3_bear"][-1]
    bad = one(100.4, 100.9, 100.4, 100.6)       # L=100.4 < 100.5
    assert not compute_patterns(*cat(pre, mid, bad))["P3_bull"][-1]
    # 看跌：High_t < Low_{t-2}
    gdn = one(98.5, 99.0, 98.0, 98.2)           # H=99.0 < L_{t-2}=99.5
    r2 = compute_patterns(*cat(pre, mid, gdn))
    assert r2["P3_bear"][-1] and not r2["P3_bull"][-1]
    bdn = one(99.3, 99.6, 99.0, 99.2)           # H=99.6 > 99.5
    assert not compute_patterns(*cat(pre, mid, bdn))["P3_bear"][-1]


# ---------------------------------------------------------------- P4 位移 K 线


def test_p4_bull_needs_both_range_and_body():
    pre = base(60)                              # 每根 TR = 1.0 → ATR14 = 1.0
    # TR = H−L = 3.0 > 1.5×1.0；实体占比 (C−O)/(H−L) = 2.7/3.0 = 0.9 ≥ 0.6
    good = one(99.0, 102.0, 99.0, 101.7)
    r = compute_patterns(*cat(pre, good))
    assert r["P4_bull"][-1] and not r["P4_bear"][-1]
    # 反例甲：幅度够大但实体占比只有 0.1（长影线）
    bad_body = one(99.0, 102.0, 99.0, 99.3)
    assert not compute_patterns(*cat(pre, bad_body))["P4_bull"][-1]
    # 反例乙：实体占满但幅度不够（TR = 1.0，不 > 1.5）
    bad_rng = one(99.6, 100.4, 99.6, 100.4)
    assert not compute_patterns(*cat(pre, bad_rng))["P4_bull"][-1]


def test_p4_bear_and_zero_range_bar():
    pre = base(60)
    good = one(102.0, 102.0, 99.0, 99.3)        # 阴线，实体占比 0.9
    assert compute_patterns(*cat(pre, good))["P4_bear"][-1]
    # H == L 的一字板：实体占比未定义 → 最简读法为「不触发」（不得抛异常）
    dead = one(100.0, 100.0, 100.0, 100.0)
    r = compute_patterns(*cat(pre, dead))
    assert not r["P4_bull"][-1] and not r["P4_bear"][-1]


# ------------------------------------------------------------ P5 订单块回测


def _p5_bull_series():
    """构造：横盘 → 一根阴线（订单块）→ P2 看涨突破 → 回落触及订单块。"""
    pre = base(60)
    ob = one(100.2, 100.4, 99.6, 99.7)          # 阴线：C<O，[L,H] = [99.6, 100.4]
    brk = one(100.0, 102.0, 99.9, 101.8)        # P2 看涨（C=101.8 > H20）
    return pre, ob, brk


def test_p5_bull_triggers_on_retest():
    pre, ob, brk = _p5_bull_series()
    # 回测日：Low ≤ OB_high(100.4) 且 Close > OB_low(99.6)
    retest = one(101.0, 101.2, 100.0, 100.9)
    r = compute_patterns(*cat(pre, ob, brk, retest))
    assert r["P5_bull"][-1]
    # 反例：一直没回到订单块上沿（Low 始终 > 100.4）
    far = one(101.5, 102.5, 101.4, 102.0)
    assert not compute_patterns(*cat(pre, ob, brk, far))["P5_bull"][-1]
    # 反例：触及了但收盘跌破订单块下沿
    thru = one(100.2, 100.3, 99.0, 99.2)
    assert not compute_patterns(*cat(pre, ob, brk, thru))["P5_bull"][-1]


def test_p5_bull_expires_after_validity_and_fires_once():
    pre, ob, brk = _p5_bull_series()
    # 有效期内第 VAL 根仍算，第 VAL+1 根不算
    far = flat(VAL, price=102.0, half=0.2)      # 全程不回测
    late = one(101.0, 101.2, 100.0, 100.9)
    r_in = compute_patterns(*cat(pre, ob, brk, flat(VAL - 1, 102.0, 0.2), late))
    assert r_in["P5_bull"][-1]
    r_out = compute_patterns(*cat(pre, ob, brk, far, late))
    assert not r_out["P5_bull"][-1]
    # 一个 t0 只触发一次：连续两根都满足回测条件，只有第一根为 True
    r2 = compute_patterns(*cat(pre, ob, brk,
                               one(101.0, 101.2, 100.0, 100.9),
                               one(100.9, 101.1, 100.1, 100.8)))
    assert r2["P5_bull"][-2] and not r2["P5_bull"][-1]


def test_p5_bear_mirror():
    pre = base(60)
    ob = one(99.8, 100.4, 99.6, 100.3)          # 阳线：C>O，[L,H] = [99.6, 100.4]
    brk = one(100.0, 100.1, 98.0, 98.3)         # P2 看跌（C < L20 = 99.5）
    retest = one(99.0, 99.8, 98.9, 99.1)        # High ≥ OB_low(99.6) 且 Close < OB_high
    r = compute_patterns(*cat(pre, ob, brk, retest))
    assert r["P5_bear"][-1]
    far = one(98.0, 98.4, 97.5, 97.9)           # 从未回到 99.6
    assert not compute_patterns(*cat(pre, ob, brk, far))["P5_bear"][-1]


def test_p5_needs_an_order_block_in_lookback():
    """t0 前 5 根内没有阴线 → 看涨 P5 永不触发。"""
    pre = flat(60, 100.0, 0.5)
    # 把 t0 前 5 根全部改成阳线（C > O）
    o, h, lo, c = pre
    o = o.copy(); c = c.copy()
    o[-OBLB:] = 99.8
    c[-OBLB:] = 100.2
    brk = one(100.0, 102.0, 99.9, 101.8)
    retest = one(101.0, 101.2, 99.9, 100.9)
    r = compute_patterns(*cat((o, h, lo, c), brk, retest))
    assert r["P2_bull"][len(o)]
    assert not r["P5_bull"].any()


# --------------------------------------------------------------------- P6 OTE


def test_p6_bull_zone_and_leg_break():
    pre = base(60)                               # L20 = 99.5
    brk = one(100.0, 110.0, 99.9, 109.0)         # P2 看涨；腿 = [99.5, 110.0]，长 10.5
    # 区间 = [99.5+0.21*10.5, 99.5+0.38*10.5] = [101.705, 103.49]
    hit = one(104.0, 104.5, 102.5, 103.0)        # Low 落在区间内，Close ≥ 区间下沿
    r = compute_patterns(*cat(pre, brk, hit))
    assert r["P6_bull"][-1]
    # 反例：回撤太浅（Low 高于区间上沿）
    shallow = one(106.0, 106.5, 105.0, 106.2)
    assert not compute_patterns(*cat(pre, brk, shallow))["P6_bull"][-1]
    # 反例：回撤太深（Low 低于区间下沿）—— 且已跌破腿低 → 作废
    deep = one(101.0, 101.5, 99.0, 99.2)
    assert not compute_patterns(*cat(pre, brk, deep))["P6_bull"][-1]
    # 破腿低之后即使再回到区间也不触发
    r2 = compute_patterns(*cat(pre, brk, deep, hit))
    assert not r2["P6_bull"].any()


def test_p6_bull_close_must_hold_zone_low():
    pre = base(60)
    brk = one(100.0, 110.0, 99.9, 109.0)
    # Low 进区间，但收盘跌回区间下沿之下 → 不触发
    weak = one(103.0, 103.2, 102.0, 101.0)
    assert not compute_patterns(*cat(pre, brk, weak))["P6_bull"][-1]


def test_p6_bear_mirror():
    pre = base(60)                               # H20 = 100.5
    brk = one(100.0, 100.1, 90.0, 90.5)          # P2 看跌；腿 = [90.0, 100.5]，长 10.5
    # 区间 = [100.5−0.38*10.5, 100.5−0.21*10.5] = [96.51, 98.295]
    hit = one(95.5, 97.5, 95.0, 96.8)            # High 落区间内，Close ≤ 区间上沿
    assert compute_patterns(*cat(pre, brk, hit))["P6_bear"][-1]
    shallow = one(92.0, 93.0, 91.5, 92.5)        # 反弹太浅
    assert not compute_patterns(*cat(pre, brk, shallow))["P6_bear"][-1]
    over = one(99.0, 101.0, 98.5, 100.5)         # 破腿高 → 作废
    assert not compute_patterns(*cat(pre, brk, over))["P6_bear"][-1]


def test_p6_expires_after_validity():
    pre = base(60)
    brk = one(100.0, 110.0, 99.9, 109.0)
    hit = one(104.0, 104.5, 102.5, 103.0)
    hold = flat(VAL, price=106.0, half=0.3)       # 有效期内不进区间
    assert compute_patterns(*cat(pre, brk, flat(VAL - 1, 106.0, 0.3), hit))["P6_bull"][-1]
    assert not compute_patterns(*cat(pre, brk, hold, hit))["P6_bull"][-1]


# ------------------------------------------------------- 可算掩码 / 缺失值


def test_computable_requires_min_history():
    o, h, lo, c = base(MINH + 5)
    r = compute_patterns(o, h, lo, c)
    assert not r["computable"][:MINH].any()
    assert r["computable"][MINH:].all()


def test_nan_bar_is_not_computable_and_never_triggers():
    o, h, lo, c = base(60)
    o, h, lo, c = o.copy(), h.copy(), lo.copy(), c.copy()
    h[-1] = np.nan
    r = compute_patterns(o, h, lo, c)
    assert not r["computable"][-1]
    for k, _ in ICT.MARKERS:
        assert not r[k][-1], k


# ------------------------------------------- 关键断言：不得使用 t+1 及之后的信息


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_no_lookahead_future_bars_cannot_change_past_flags(seed):
    """把 t+1..T 整段替换成完全不同的走势，t 日及之前的 12 个标记必须逐位不变。

    这条断言直接对应 CLAUDE.md §一.1（前视）与预注册 §7.1。
    """
    rng = np.random.default_rng(seed)
    n = 200
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    o = c * (1 + rng.normal(0, 0.005, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.01, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.01, n)))

    full = compute_patterns(o, h, lo, c)
    cut = 140                                   # t = cut-1 是最后一根「过去」K 线

    # 变体甲：截断（完全没有未来）
    trunc = compute_patterns(o[:cut], h[:cut], lo[:cut], c[:cut])
    # 变体乙：未来整段换成另一条随机路径
    o2, h2, lo2, c2 = o.copy(), h.copy(), lo.copy(), c.copy()
    c2[cut:] = 100 * np.exp(np.cumsum(rng.normal(0.01, 0.05, n - cut)))
    o2[cut:] = c2[cut:] * 1.01
    h2[cut:] = np.maximum(o2[cut:], c2[cut:]) * 1.05
    lo2[cut:] = np.minimum(o2[cut:], c2[cut:]) * 0.95
    alt = compute_patterns(o2, h2, lo2, c2)
    # 变体丙：未来全部置为缺失
    o3, h3, lo3, c3 = o.copy(), h.copy(), lo.copy(), c.copy()
    for a in (o3, h3, lo3, c3):
        a[cut:] = np.nan
    miss = compute_patterns(o3, h3, lo3, c3)

    keys = [k for k, _ in ICT.MARKERS] + ["computable"]
    for k in keys:
        np.testing.assert_array_equal(full[k][:cut], trunc[k], err_msg=f"{k} 截断不一致")
        np.testing.assert_array_equal(full[k][:cut], alt[k][:cut], err_msg=f"{k} 换未来后变了")
        np.testing.assert_array_equal(full[k][:cut], miss[k][:cut], err_msg=f"{k} 未来缺失后变了")
    # 该随机路径确实触发了标记，否则这条断言是空的
    assert sum(int(full[k][:cut].sum()) for k, _ in ICT.MARKERS) > 0


def test_frozen_parameters_match_prereg():
    """参数一律照预注册，不改、不加变体（预注册 §3 / §7.2）。"""
    assert (L, ICT.ATR_WIN, K, OBLB, VAL) == (20, 14, 1.5, 5, 20)
    assert ICT.BODY_FRAC == 0.6
    assert (ICT.OTE_LO_FRAC, ICT.OTE_HI_FRAC) == (0.21, 0.38)
    assert ICT.MIN_DAY_TRIG == 5 and ICT.MIN_FOLD_EVENTS == 20
    assert ICT.NW_LAG == 6 and ICT.SESOI_BP == 10.0 and ICT.MDE_MULT == 2.80
    assert MINH == VAL + L + OBLB == 45
