"""信号 #2：隔夜 / 日内收益分解（LPS 2019 / BBM 2026 口径，**零改动、参数不可调**）。

为什么是它（`docs/思路整理_2026-09-03.md` §4.1）
------------------------------------------------
Kronos 的输入与它的 8 因子张成控制**全是收盘到收盘口径，从没切过日夜**，
所以隔夜/日内分解的低相关有机制支撑（先验估 ρ 0.15–0.35，**必须实测**）。
只需 CRSP 日线的 open/close，不需要 Compustat；月频构造，年换手 8–12×。

冻结定义
--------
逐票（按 ``PERMNO`` 分组、按 ``DlyCalDt`` 升序）：

    intraday_t  = |DlyClose_t| / |DlyOpen_t| - 1
    overnight_t = (1 + DlyRet_t) / (1 + intraday_t) - 1
    score(t)            = Σ_{s = t-20 .. t} intraday_s      （21 个交易日，含 t）
    score_overnight(t)  = Σ_{s = t-20 .. t} overnight_s     （诊断列，不作主分）

有效性三条（任一不满足 → NaN）：

1. ``DlyOpen`` 与 ``DlyClose`` 的绝对值都 > 0 且有限，否则该日 ``intraday`` 为 NaN；
   ``DlyRet`` 或 ``intraday`` 缺失则该日 ``overnight`` 为 NaN；
2. 窗口内有效值 < :data:`MIN_VALID` (=15)；
3. 窗口首末日期跨度 > :data:`MAX_SPAN_DAYS` (=31)。

**负价取绝对值**：CRSP 里负价是「无成交、用买卖中点填充」的标记，价格本身有效。
与 ``scripts/compare_arms_money.py`` 的 ``oc`` 列同一处理（该脚本只对 Open 取绝对值，
因为它的分子 Close 已在别处过滤；此处两端都取绝对值，口径更严，
对同时为负的行不再产生符号翻转的伪收益）。

跨度检查用的是**自然日**，不是交易日（在两种许可实现里选了这一种）
--------------------------------------------------------------
``span_days(t) = DlyCalDt_t - DlyCalDt_{t-20}``（同一 PERMNO 内按**行**回退 20 行），
要求 ``<= 31`` 自然日。理由：

- 21 个连续交易日的自然日跨度下界是 28 天（4 个周末 = 8 个非交易日），
  含 1–3 个假日时为 29–31 天。**> 31 天必然意味着该票在窗口内缺行**
  （停牌、退市后重新上市、数据缺失），这时 21 行的和会把陈旧观测混进来。
- 不依赖外部交易日历，面板本身自洽；``crsp_pipeline.calendar.TradingCalendar``
  的交易日版本需要额外读 ``market_index.parquet``，在逐折读取的内存预算下不划算。
- 副作用（有意保留）：每只票前 20 行的 ``shift(20)`` 为 NaT，跨度检查判为无效，
  因此**窗口不满 21 行的行一律不出分**——这正是「21 个交易日含 t」的字面要求。

不做的事
--------
截面上**不做** winsorize、不做 z-score、不做行业/市值中性化。评估用 Spearman 秩，
对单调变换免疫，加这些只会引入未登记的自由参数（调研 B 隐性污染渠道 ④）。
这三条「不做」已写进 :data:`SPEC` 的 params，改动即改哈希。

前视自查
--------
score(t) 只用 ``DlyCalDt <= t`` 的行（rolling 后视窗口 + 同组 shift 正向）。
可执行检查见 ``signals.base.assert_no_lookahead``，测试里另有「改动 t+1 之后的数值，
t 及之前的输出不变」的更严版本。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Signal, SignalSpec

# ---------------------------------------------------------------- 冻结常量
# 三个常量均为**文献 / 日历先验**，不可调、不得在任何折上搜索。
# LOOKBACK=21：LPS 2019 的「过去 1 个月日内累计收益」，1 个月 = 21 个交易日；
# MIN_VALID=15：容许约 1/4 的缺失（停牌、半日市），低于此则和不可比；
# MAX_SPAN_DAYS=31：21 个交易日的自然日跨度上界（4 个周末 + 至多 3 个假日）。
LOOKBACK: int = 21
MIN_VALID: int = 15
MAX_SPAN_DAYS: int = 31

SOURCE_REF = ("Lou-Polk-Skouras 2019 JFE; "
              "Barardehi-Bogousslavsky-Muravyev 2026 RFS")

SPEC = SignalSpec(
    name="overnight_intraday",
    version="v1",
    horizon_days=6,          # 与 Kronos 并列读数：一律在冻结的 6 日执行收益标签上评估
    params=(
        ("lookback", LOOKBACK),
        ("min_valid", MIN_VALID),
        ("max_span_days", MAX_SPAN_DAYS),
        ("span_unit", "calendar_days"),
        ("component", "intraday_sum"),
        ("abs_price", True),          # 负价 = 买卖中点标记，取绝对值
        ("winsorize", False),
        ("cross_sectional_zscore", False),
        ("neutralise", "none"),
        ("tunable", False),           # 参数不可调；改动必须升 version
    ),
    source_ref=SOURCE_REF,
)


def decompose(panel: pd.DataFrame) -> pd.DataFrame:
    """逐日分解，返回带 ``intraday`` / ``overnight`` 两列的排序后副本。

    不做任何滚动，只做当日恒等式；单独暴露是为了让测试能直接盯死
    ``(1 + overnight)(1 + intraday) = 1 + DlyRet``。
    """
    Signal.require_columns(panel)
    df = panel.loc[:, ["PERMNO", "DlyCalDt", "DlyOpen", "DlyClose", "DlyRet"]].copy()
    df["DlyCalDt"] = pd.to_datetime(df["DlyCalDt"])
    df = (df.sort_values(["PERMNO", "DlyCalDt"], kind="mergesort")
            .reset_index(drop=True))

    o = pd.to_numeric(df["DlyOpen"], errors="coerce").abs()
    c = pd.to_numeric(df["DlyClose"], errors="coerce").abs()
    r = pd.to_numeric(df["DlyRet"], errors="coerce")

    ov, cv = o.to_numpy(dtype="float64"), c.to_numpy(dtype="float64")
    ok = np.isfinite(ov) & np.isfinite(cv) & (ov > 0.0) & (cv > 0.0)
    ratio = np.divide(cv, ov, out=np.full(len(df), np.nan), where=ok)
    intraday = pd.Series(ratio - 1.0, index=df.index)
    # o > 0 且 c > 0 ⇒ 1 + intraday = c/o > 0，分母不会为零
    overnight = (1.0 + r) / (1.0 + intraday) - 1.0

    df["intraday"] = intraday
    df["overnight"] = overnight.where(intraday.notna() & r.notna())
    return df


class OvernightIntradaySignal(Signal):
    """信号 #2 的实现。构造无参数——参数在 :data:`SPEC` 里，且不可调。"""

    spec = SPEC

    def compute(self, panel: pd.DataFrame) -> pd.DataFrame:
        """返回 ``signal_date, PERMNO, score, score_overnight, n_valid``。

        输出**保留全部输入行**（含 score 为 NaN 的行），由下游自行 dropna——
        这样「哪些行因为哪条规则被判无效」在 ``n_valid`` 上仍可核查。
        """
        df = decompose(panel)
        if df.empty:
            return pd.DataFrame(
                {"signal_date": pd.Series(dtype="datetime64[ns]"),
                 "PERMNO": pd.Series(dtype=df["PERMNO"].dtype),
                 "score": pd.Series(dtype="float64"),
                 "score_overnight": pd.Series(dtype="float64"),
                 "n_valid": pd.Series(dtype="int64")})

        g = df.groupby("PERMNO", sort=False)

        def _roll_sum(col: str) -> pd.Series:
            return (g[col].rolling(LOOKBACK, min_periods=MIN_VALID).sum()
                    .reset_index(level=0, drop=True))

        score = _roll_sum("intraday")
        score_on = _roll_sum("overnight")

        # 有效值计数：对 notna 指示量做同窗口滚动和（min_periods=1，恒有数值）
        df["_ok"] = df["intraday"].notna().astype("float64")
        n_valid = (df.groupby("PERMNO", sort=False)["_ok"]
                     .rolling(LOOKBACK, min_periods=1).sum()
                     .reset_index(level=0, drop=True))

        # 跨度：同组内回退 LOOKBACK-1 行的日期与当前日期之差（自然日）
        prev_date = g["DlyCalDt"].shift(LOOKBACK - 1)
        span = (df["DlyCalDt"] - prev_date).dt.days
        window_ok = span.notna() & (span <= MAX_SPAN_DAYS) & (n_valid >= MIN_VALID)

        out = pd.DataFrame({
            "signal_date": df["DlyCalDt"],
            "PERMNO": df["PERMNO"],
            "score": score.where(window_ok),
            "score_overnight": score_on.where(window_ok),
            "n_valid": n_valid.astype("int64"),
        })
        return out.reset_index(drop=True)
