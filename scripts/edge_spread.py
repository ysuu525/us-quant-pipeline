"""EDGE 有效价差估计器 —— 逐股逐月，2000-2025 全池。

Ardia, Guidotti & Kroencke (JFE 2024), "Efficient estimation of bid-ask spreads
from open, high, low, and close prices", doi:10.1016/j.jfineco.2024.103916。
本实现逐行对照官方参考实现 github.com/eguidotti/bidask (python/bidask/edge.py)，
向量化到 (PERMNO, 年月) 分组上；**不安装该包**（CLAUDE.md §七：不往 .venv 装东西）。

为什么要它
----------
① 本项目的成本模型至今是**全池统一 8bp** 的外生假设，从未实测；
② K6b 查出持仓在 top500 池内系统性偏向低价/不流动/高波动一端
   （corr(1/价格)=+0.382、Amihud=+0.326、特质波动=+0.317、价差代理=+0.291），
   所以池均值成本必然**低估**持仓的真实成本；
③ 容量测算显示零售规模（<=$1M）下冲击项 <=1.3bp，故 **成本 ≈ 价差 + 费用**，
   而 Alpaca 免佣 → **EDGE 基本就是本项目成本模型的全部**；
④ A.1 的次终点需要**年代适配**的成本，而 2003 年的有效价差远宽于今日。

口径说明
--------
* 用 `panel_kronos_adj.parquet`（复权 OHLC）——EDGE 用相邻日关系，未复权价
  在拆股日会产生巨大伪跳变；
* lag 在 **PERMNO × 年** 内取（跨月边界正确），聚合按 **PERMNO × 年月**；
* 输出 `edge` 是**相对价差**（0.01 = 1%）。单边成本 = edge / 2；
* CRSP 用负价表示买卖价均值（无成交），一律取绝对值；非正价与缺失剔除。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs"
P = Path(r"F:\quant\processed\crsp_ciz_2026-08-24_20260825T130601Z")
Y0, Y1 = 2000, 2025


def log(m):
    print(m, flush=True)


def edge_grouped(df: pd.DataFrame, gcols) -> pd.DataFrame:
    """向量化 EDGE。df 需含 o/h/l/c 的对数价与分组键，且已按 (PERMNO, 日期) 排序。

    与参考实现的对应关系逐行标注。
    """
    g_p = df.groupby("PERMNO", sort=False)
    # 参考实现的 shift：h1,l1,c1,m1 = h[:-1],...；o,h,l,c,m = [1:]
    # 这里改为在 PERMNO 内 shift(1)，等价且跨月边界正确（首行 lag 为 NaN 自然丢弃）
    h1, l1, c1, m1 = (g_p[k].shift(1) for k in ("h", "l", "c", "m"))
    o, h, l, c, m = df.o, df.h, df.l, df.c, df.m

    # 参考实现把数组整体前移一位（o=o[1:] …），因此其所有 nanmean 都在 n-1 行上算。
    # 本实现保留首行，而 r1 = m-o 不含 lag、不会自动变 NaN —— 若不显式屏蔽，
    # mean(r1)/mean(r3)/mean(r5) 会比参考多算一行。故先按 lag 是否可用建 mask。
    lagged = m1.notna() & c1.notna()
    r1 = (m - o).where(lagged)
    r2 = o - m1
    r3 = m - c1
    r4 = c1 - m1
    r5 = o - c1

    nan = h.isna() | l.isna() | c1.isna()
    tau = np.where(nan, np.nan, ((h != l) | (l != c1)).astype(float))
    tau = pd.Series(tau, index=df.index)
    po1 = tau * np.where(o.isna() | h.isna(), np.nan, (o != h).astype(float))
    po2 = tau * np.where(o.isna() | l.isna(), np.nan, (o != l).astype(float))
    pc1 = tau * np.where(c1.isna() | h1.isna(), np.nan, (c1 != h1).astype(float))
    pc2 = tau * np.where(c1.isna() | l1.isna(), np.nan, (c1 != l1).astype(float))

    w = df[gcols].copy()
    for nm, v in (("tau", tau), ("po1", po1), ("po2", po2), ("pc1", pc1), ("pc2", pc2),
                  ("r1", r1), ("r2", r2), ("r3", r3), ("r4", r4), ("r5", r5)):
        w[nm] = v
    gg = w.groupby(gcols, sort=False)

    pt = gg["tau"].transform("mean")
    po = gg["po1"].transform("mean") + gg["po2"].transform("mean")
    pc = gg["pc1"].transform("mean") + gg["pc2"].transform("mean")

    # de-meaned log-returns（参考实现：d1 = r1 - nanmean(r1)/pt*tau）
    d1 = w.r1 - gg["r1"].transform("mean") / pt * w.tau
    d3 = w.r3 - gg["r3"].transform("mean") / pt * w.tau
    d5 = w.r5 - gg["r5"].transform("mean") / pt * w.tau

    w["x1"] = -4.0 / po * d1 * w.r2 + -4.0 / pc * d3 * w.r4
    w["x2"] = -4.0 / po * d1 * w.r5 + -4.0 / pc * d5 * w.r4
    w["x1sq"], w["x2sq"] = w.x1 ** 2, w.x2 ** 2

    a = w.groupby(gcols, sort=False).agg(
        n=("x1", "size"), tau_sum=("tau", "sum"),
        po_m1=("po1", "mean"), po_m2=("po2", "mean"),
        pc_m1=("pc1", "mean"), pc_m2=("pc2", "mean"),
        e1=("x1", "mean"), e2=("x2", "mean"),
        m1s=("x1sq", "mean"), m2s=("x2sq", "mean"))
    a["po"] = a.po_m1 + a.po_m2
    a["pc"] = a.pc_m1 + a.pc_m2
    v1 = a.m1s - a.e1 ** 2
    v2 = a.m2s - a.e2 ** 2
    vt = v1 + v2
    s2 = np.where(vt > 0, (v2 * a.e1 + v1 * a.e2) / vt.replace(0, np.nan),
                  (a.e1 + a.e2) / 2.0)
    # 参考实现的三道 guard：nobs<3 / nansum(tau)<2 / po==0 or pc==0 → NaN
    bad = (a.n < 3) | (a.tau_sum < 2) | (a.po == 0) | (a.pc == 0)
    s = np.sqrt(np.abs(s2))
    a["edge"] = np.where(bad, np.nan, s)
    return a[["n", "edge"]].reset_index()


def main():
    parts = []
    for y in range(Y0, Y1 + 1):
        df = pd.read_parquet(
            P / "panel_kronos_adj.parquet",
            columns=["PERMNO", "DlyCalDt", "DlyOpen", "DlyHigh", "DlyLow", "DlyClose"],
            filters=[("DlyCalDt", ">=", pd.Timestamp(f"{y}-01-01")),
                     ("DlyCalDt", "<=", pd.Timestamp(f"{y}-12-31"))])
        if df.empty:
            continue
        df["DlyCalDt"] = pd.to_datetime(df["DlyCalDt"])
        for k in ("DlyOpen", "DlyHigh", "DlyLow", "DlyClose"):
            df[k] = df[k].abs()                      # CRSP 负价 = 买卖价均值
            df.loc[df[k] <= 0, k] = np.nan
        df = df.dropna(subset=["DlyOpen", "DlyHigh", "DlyLow", "DlyClose"])
        df = df.sort_values(["PERMNO", "DlyCalDt"]).reset_index(drop=True)
        df["o"] = np.log(df.DlyOpen); df["h"] = np.log(df.DlyHigh)
        df["l"] = np.log(df.DlyLow);  df["c"] = np.log(df.DlyClose)
        df["m"] = (df.h + df.l) / 2.0
        df["ym"] = df.DlyCalDt.dt.to_period("M").astype(str)
        a = edge_grouped(df, ["PERMNO", "ym"])
        parts.append(a)
        ok = a.edge.notna()
        log(f"  {y}: {len(a):>7,} 个 (股,月)  有效 {ok.mean():.1%}  "
            f"中位有效价差 {a.edge[ok].median()*1e4:6.1f}bp")
        del df
    E = pd.concat(parts, ignore_index=True)
    E.to_parquet(OUT / "edge_monthly.parquet", index=False)
    log(f"\n写入 outputs/edge_monthly.parquet  ({len(E):,} 行)")


if __name__ == "__main__":
    main()
