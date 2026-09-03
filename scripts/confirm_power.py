"""确认集（折 05-35）的功效分析 —— 在花掉统计资本之前算清楚它能买到什么。

背景
----
主终点已从「Patton-Timmermann 单调性检验」改掉：外部复核指出 MR 的有效时间样本
仍是约 3900 个交易日（截面相关使 500 名字 × 3900 天**不是**独立样本），
且「统计单调但经济幅度为零」也能通过。改用的三道门槛（codex 提议的默认值）：

  ① 毛主动收益 >= 5%/年
  ② HAC t >= 1.96
  ③ 31 折中至少 21 折为正

**三道门槛不独立**（同一批折同时决定 ①②③），故用联合模拟而非分别算。

参数来自近代七折实测：毛超额 +10.48%/年、波动 9.20%/年。
折长 0.5 年 -> 单折 SE = 9.20/sqrt(0.5) = 13.01%/年；31 折 15.5 年 -> 池化 SE = 2.34%/年。

**这个脚本不消耗任何评估折**：它只用已公开的近代读数做参数。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "outputs"
G_MOD, VOL_MOD = 10.48, 9.20       # 近代毛超额与波动（%/年）
NFOLD, FOLD_YR = 31, 0.5
THR_G, THR_T, THR_K = 5.0, 1.96, 21
NSIM, SEED = 200_000, 20260831


def log(m):
    print(m, flush=True)


def power(g_true, vol, rng, nfold=NFOLD, thr_g=THR_G, thr_t=THR_T, thr_k=THR_K):
    se_f = vol / np.sqrt(FOLD_YR)
    x = rng.normal(g_true, se_f, size=(NSIM, nfold))
    gbar = x.mean(1)
    se_p = vol / np.sqrt(nfold * FOLD_YR)
    t = gbar / se_p
    kpos = (x > 0).sum(1)
    return dict(p_g=float((gbar >= thr_g).mean()),
                p_t=float((t >= thr_t).mean()),
                p_k=float((kpos >= thr_k).mean()),
                p_all=float(((gbar >= thr_g) & (t >= thr_t) & (kpos >= thr_k)).mean()),
                e_kpos=float(kpos.mean()))


def main():
    rng = np.random.default_rng(SEED)
    res = {"params": {"gross_modern_pct": G_MOD, "vol_modern_pct": VOL_MOD,
                      "n_folds": NFOLD, "thresholds": {"gross": THR_G, "t": THR_T,
                                                       "k_positive": THR_K}}}

    log("=" * 78)
    log("A. 联合功效：真实毛超额取近代的不同比例（波动同近代 9.20%）")
    log("=" * 78)
    log(f"  {'强度':>8}{'真毛%':>8}{'P(毛>=5%)':>11}{'P(t>=1.96)':>12}"
        f"{'P(>=21/31折)':>14}{'期望正折':>10}{'**联合功效**':>13}")
    res["by_strength"] = {}
    for frac in (1.00, 0.75, 0.60, 0.50, 0.40, 0.30, 0.00):
        g = G_MOD * frac
        r = power(g, VOL_MOD, rng)
        res["by_strength"][f"{frac:.2f}"] = {"gross_true": g, **r}
        log(f"  {frac:>7.0%}{g:>8.2f}{r['p_g']:>11.3f}{r['p_t']:>12.3f}"
            f"{r['p_k']:>14.3f}{r['e_kpos']:>9.1f}{r['p_all']:>13.3f}")
    log("\n  最后一行（强度 0）= 三道门槛联合的**第一类错误率**（真信号为零时误判通过的概率）")

    log("\n" + "=" * 78)
    log("B. 哪道门槛是瓶颈")
    log("=" * 78)
    r50 = res["by_strength"]["0.50"]
    log(f"  半强度（真毛 {G_MOD*0.5:.2f}%/年）下：")
    log(f"    毛>=5% 单独通过率   {r50['p_g']:.3f}")
    log(f"    t>=1.96 单独通过率  {r50['p_t']:.3f}")
    log(f"    >=21/31 折单独通过率 {r50['p_k']:.3f}   <- 期望正折数只有 {r50['e_kpos']:.1f}")
    log(f"    **联合 {r50['p_all']:.3f}**")
    worst = min(("毛>=5%", r50["p_g"]), ("t>=1.96", r50["p_t"]),
                ("折数>=21", r50["p_k"]), key=lambda x: x[1])
    log(f"  -> 瓶颈是 **{worst[0]}**（单独通过率 {worst[1]:.3f}）")

    log("\n" + "=" * 78)
    log("C. 折数门槛该定在哪（半强度下，其余两道不变）")
    log("=" * 78)
    log(f"  {'门槛 k':>8}{'半强度功效':>12}{'零信号误判率':>14}")
    res["k_threshold"] = {}
    for k in (16, 17, 18, 19, 20, 21, 22, 23):
        a = power(G_MOD * 0.5, VOL_MOD, rng, thr_k=k)["p_all"]
        b = power(0.0, VOL_MOD, rng, thr_k=k)["p_all"]
        res["k_threshold"][k] = {"power_half": a, "size_null": b}
        log(f"  {k:>8}{a:>12.3f}{b:>14.4f}")

    log("\n" + "=" * 78)
    log("D. 波动更高的敏感性（2005-2020 含 2008，波动大概率高于近代）")
    log("=" * 78)
    log(f"  {'波动%':>8}{'全强度联合':>12}{'半强度联合':>12}")
    res["vol_sensitivity"] = {}
    for v in (9.2, 11.0, 13.0, 15.0):
        a = power(G_MOD, v, rng)["p_all"]
        b = power(G_MOD * 0.5, v, rng)["p_all"]
        res["vol_sensitivity"][v] = {"power_full": a, "power_half": b}
        log(f"  {v:>8.1f}{a:>12.3f}{b:>12.3f}")

    log("\n" + "=" * 78)
    log("E. ADV 梯度方向（K7b 之后的主科学问题）")
    log("=" * 78)
    log("  统计量 Δ = IC(top500) − IC(全池)。实测：2003-04 = -0.00620(NW t=-2.23, 504 天)、")
    log("            2020H2-23 = +0.00517(t=1.89, 881 天)。")
    for nd, nm in [(3900, "折05-35（31 折 ≈ 3900 天）")]:
        for d_true, lbl in [(0.00517, "与近代同向同幅"), (-0.00620, "与早期同向同幅"),
                            (0.0026, "近代的一半")]:
            se = abs(0.00517 / 1.89) * np.sqrt(881 / nd)
            t = d_true / se
            log(f"  {nm}：若真 Δ = {d_true:+.5f}（{lbl}） -> 期望 |t| = {abs(t):.2f}")
    log("  -> 梯度方向这一问，确认集的功效很足（|t| 3~6），是三个终点里最可靠的一个。")
    res["adv_gradient"] = {"se_at_3900d": float(abs(0.00517 / 1.89) * np.sqrt(881 / 3900))}

    (OUT / "confirm_power.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    log("\n写入 outputs/confirm_power.json")


if __name__ == "__main__":
    main()
