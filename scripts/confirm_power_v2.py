"""确认集功效分析 v2 —— 修正三处（外部复核 codex 提出，2026-08-31）。

相对 v1 的改动
--------------
1. **波动敏感性改用 k=18**（v1 的表还是 k=21，已过期）；
2. **改四态判读**，不再是通过/失败二分；
3. **ΔADV 独立成一个终点**，给出精确估计式、CI、量级门槛与**它自己的**联合功效
   （折数门槛不可从毛收益终点移植——两者是不同的估计量、不同的 SE）。

四态判读（预注册）
------------------
  确认        ：毛 >= 5% 且 HAC t >= 1.96 且 正折 >= 18/31
  低于经济目标：毛收益单边 95% CI 上界仍 < 5%
  信号基本死亡：毛收益 95% CI 上界 <= 0
  不确定      ：其余全部（尤其点估计尚可但波动过高的情形）

ΔADV（科学主终点，与经济主终点并列、不合成总 PASS）
----------------------------------------------------
  ΔADV = mean_t [ IC_t(top500) − IC_t(全池) ]
  实测：2003-04 = −0.00620（NW t=−2.23，504 天）；2020H2-23 = +0.00517（t=+1.89，881 天）

  现代型      ：点估计 >= +0.0026 且双侧 95% CI 下界 > 0
  早期型      ：点估计 <= −0.0031 且双侧 95% CI 上界 < 0
  经济上近零  ：90% CI 完全落入 [−0.0031, +0.0026]
  不确定      ：其余

⚠ 本脚本用 iid 正态折收益 + 已知波动，**是设计近似，不是实际 HAC 检验的严格 size**。
  报出的误判率应按「设计参考值」而非「真实第一类错误率」引用。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "outputs"
G_MOD, VOL_MOD = 10.48, 9.20
NFOLD, FOLD_YR = 31, 0.5
THR_G, THR_T, THR_K = 5.0, 1.96, 18          # k 由 21 降到 18（v1 §C 的结论）
NSIM, SEED = 500_000, 20260831

# ΔADV 的实测输入
D_MOD, T_MOD, N_MOD = 0.00517, 1.89, 881
D_EAR, T_EAR, N_EAR = -0.00620, -2.23, 504
D_HI, D_LO = 0.0026, -0.0031                  # 量级门槛（各取实测的一半）
N_CONFIRM = 3900                              # 折05-35 ≈ 31 折 × 125 天


def log(m):
    print(m, flush=True)


def states(g_true, vol, rng, nfold=NFOLD, thr_k=THR_K):
    """返回四态概率。"""
    se_f = vol / np.sqrt(FOLD_YR)
    se_p = vol / np.sqrt(nfold * FOLD_YR)
    x = rng.normal(g_true, se_f, size=(NSIM, nfold))
    gbar = x.mean(1)
    t = gbar / se_p
    kpos = (x > 0).sum(1)
    ci_hi = gbar + 1.96 * se_p                # 单边/双侧上界（同一表达式）
    confirm = (gbar >= THR_G) & (t >= THR_T) & (kpos >= thr_k)
    dead = (~confirm) & (ci_hi <= 0)
    below = (~confirm) & (~dead) & (ci_hi < THR_G)
    unclear = ~(confirm | dead | below)
    return dict(confirm=float(confirm.mean()), below=float(below.mean()),
                unclear=float(unclear.mean()), dead=float(dead.mean()),
                e_kpos=float(kpos.mean()))


def main():
    rng = np.random.default_rng(SEED)
    res = {"note": "iid 正态折收益 + 已知波动的设计近似，非严格 HAC size",
           "thresholds": {"gross": THR_G, "t": THR_T, "k": THR_K}}

    log("=" * 84)
    log(f"A. 四态判读概率（k={THR_K}，波动 = 近代 {VOL_MOD}%）")
    log("=" * 84)
    log(f"  {'强度':>7}{'真毛%':>8}{'确认':>9}{'低于目标':>11}{'不确定':>10}{'死亡':>9}{'期望正折':>10}")
    res["by_strength"] = {}
    for frac in (1.00, 0.75, 0.60, 0.50, 0.40, 0.30, 0.00):
        r = states(G_MOD * frac, VOL_MOD, rng)
        res["by_strength"][f"{frac:.2f}"] = {"gross_true": G_MOD * frac, **r}
        log(f"  {frac:>6.0%}{G_MOD*frac:>8.2f}{r['confirm']:>9.3f}{r['below']:>11.3f}"
            f"{r['unclear']:>10.3f}{r['dead']:>9.3f}{r['e_kpos']:>10.1f}")
    log("  （强度 0 那行的『确认』= 设计参考误判率）")

    log("\n" + "=" * 84)
    log(f"B. 波动敏感性（**已改用 k={THR_K}**，v1 的表用的是 k=21，已过期）")
    log("=" * 84)
    log(f"  {'主动波动%':>10}{'全强度确认':>12}{'半强度确认':>12}{'零信号误判':>12}"
        f"{'全强度不确定':>14}")
    res["vol_sensitivity"] = {}
    for v in (9.2, 11.0, 13.0, 15.0):
        a = states(G_MOD, v, rng)
        b = states(G_MOD * 0.5, v, rng)
        c = states(0.0, v, rng)
        res["vol_sensitivity"][v] = {"full": a, "half": b, "null": c}
        log(f"  {v:>10.1f}{a['confirm']:>12.3f}{b['confirm']:>12.3f}"
            f"{c['confirm']:>12.3f}{a['unclear']:>14.3f}")
    log("  -> 波动升高时，失败主要转入『不确定』而非『死亡』——这正是四态判读的意义。")

    log("\n" + "=" * 84)
    log("C. ΔADV（科学主终点，与经济主终点并列，**不合成总 PASS**）")
    log("=" * 84)
    se_day_mod = abs(D_MOD / T_MOD)                    # 近代 NW SE（881 天）
    se_c = se_day_mod * np.sqrt(N_MOD / N_CONFIRM)     # 折05-35 上的 SE
    se_fold = se_day_mod * np.sqrt(N_MOD / (N_CONFIRM / NFOLD))
    log(f"  估计式  ΔADV = mean_t [ IC_t(top500) − IC_t(全池) ]")
    log(f"  实测    2003-04 = {D_EAR:+.5f} (NW t={T_EAR:+.2f}, {N_EAR} 天)")
    log(f"          2020H2-23 = {D_MOD:+.5f} (NW t={T_MOD:+.2f}, {N_MOD} 天)")
    log(f"  折05-35 上的预期 SE = {se_c:.5f}（约 {N_CONFIRM} 天）；单折 SE = {se_fold:.5f}")
    log(f"  量级门槛：现代型 >= {D_HI:+.4f}   早期型 <= {D_LO:+.4f}（各取实测一半）\n")

    hw95, hw90 = 1.96 * se_c, 1.645 * se_c
    log(f"  {'真 ΔADV':>12}{'现代型':>9}{'早期型':>9}{'近零':>9}{'不确定':>10}{'期望正折':>10}")
    res["adv"] = {"se_confirm": float(se_c), "se_fold": float(se_fold),
                  "thresholds": {"modern": D_HI, "early": D_LO}, "by_truth": {}}
    for d_true, lbl in ((D_MOD, "近代同幅"), (D_HI, "近代一半"), (0.0, "零"),
                        (D_LO, "早期一半"), (D_EAR, "早期同幅")):
        dh = rng.normal(d_true, se_c, NSIM)
        xf = rng.normal(d_true, se_fold, size=(NSIM, NFOLD))
        modern = (dh >= D_HI) & (dh - hw95 > 0)
        early = (dh <= D_LO) & (dh + hw95 < 0)
        near0 = (~modern) & (~early) & (dh - hw90 >= D_LO) & (dh + hw90 <= D_HI)
        unc = ~(modern | early | near0)
        res["adv"]["by_truth"][f"{d_true:+.5f}"] = {
            "label": lbl, "modern": float(modern.mean()), "early": float(early.mean()),
            "near_zero": float(near0.mean()), "unclear": float(unc.mean()),
            "e_folds_pos": float((xf > 0).sum(1).mean())}
        log(f"  {d_true:>+12.5f}{modern.mean():>9.3f}{early.mean():>9.3f}"
            f"{near0.mean():>9.3f}{unc.mean():>10.3f}{(xf>0).sum(1).mean():>10.1f}"
            f"   ({lbl})")
    log("\n  ⚠ 『经济上近零』这一态功效不足：真值为零时也只有约"
        f" {res['adv']['by_truth']['+0.00000']['near_zero']:.0%} 概率被判为近零，"
        "其余落入『不确定』。")

    log("\n  ΔADV 若要附加折数门槛，须用**它自己的**单折 SE 模拟（不可移植毛收益的 18/31）：")
    log(f"  {'真 ΔADV':>12}{'期望正折':>10}{'P(>=18/31)':>13}{'P(>=21/31)':>13}")
    for d_true, lbl in ((D_MOD, "近代同幅"), (D_HI, "近代一半"), (0.0, "零")):
        xf = rng.normal(d_true, se_fold, size=(NSIM, NFOLD))
        k = (xf > 0).sum(1)
        log(f"  {d_true:>+12.5f}{k.mean():>10.1f}{float((k>=18).mean()):>13.3f}"
            f"{float((k>=21).mean()):>13.3f}   ({lbl})")

    log("\n" + "=" * 84)
    log("D. 结果矩阵（两个主终点交叉，不合成总 PASS）")
    log("=" * 84)
    for a, b, c in (("正", "通过", "top500 机制跨年代成立且可交易"),
                    ("负", "失败", "近代 top500 优势不具普遍性"),
                    ("正", "失败", "机制存在，但收益幅度/风险不够"),
                    ("负", "通过", "策略赚钱，但不是靠近代观察到的流动性机制")):
        log(f"  ΔADV {a} × 毛收益 {b}  ->  {c}")

    (OUT / "confirm_power_v2.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    log("\n写入 outputs/confirm_power_v2.json")


if __name__ == "__main__":
    main()
