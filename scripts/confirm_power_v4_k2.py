"""确认集 S 端（H2 / ΔADV）在 **k=2 双臂 + z=2.2414 + 去量级门** 口径下的功效重算。

出处 / 为什么跑这一条
--------------------
`experiments/confirmation_protocol_v4_draft_2026-09-05.md` §2.5 与 §8「未核」#2：
研究计划书 v0.2（`docs/研究计划书_v0.2_2026-09-04.md:305-315`）引审稿信写「k=2 口径
功效 92.2% / 34.3%」，并自陈在登记簿检索不到这两个数；v4 草案要求
「须由 `confirm_power_v3_joint` 在 k=2、z=2.2414、去量级门条件下重算一次并登记」。

判据 / 用途限制（**先于运行写死**，CLAUDE.md §二）
-------------------------------------------------
1. 本脚本是**设计期蒙特卡洛**，不读取任何折 05–35、eval_sealed_*、或 ≥2024-01-01 的
   分数 / 标签 / 收益。全部输入是**开发折 36–42 已落盘的汇总统计**（K11 审计 JSON）
   加解析外推，来源逐条写在 `INPUT_PROVENANCE` 里。
2. **估计交付**：输出是功效与 MDE，**不作任何判定**（不 PASS/FAIL、不选臂、不选门槛）。
3. 折数门 **18/31 已由 `ledger.md:283` 事前选定并写入 v4 §2.5，本脚本不重选**。
   输出里的 k 网格（16..22）只作审计展示，**不得据以更换门槛**。
4. 量级门（ΔADV ≥ +0.0026）的有无是 v4 修订 3 已作出的**协议改动**，不是本脚本的
   自由参数；两种口径都算，是为了与历史读数对账，**不是在两者间挑好看的**。
5. z 的两个取值（1.9600 单臂 / 2.2414 双臂 Bonferroni）同样来自既有裁定
   （`ledger.md:409` k=2 决定；`outputs/k12_confirmation_config_power.json`
   `design.z_both_bonferroni`），不是本脚本选的。
6. MDE80 的两种算法（解析 z 门 vs 含全部门的模拟）都报，主报**含全部门的模拟值**，
   因为 v4 §2.5 的判据是三门并列。

口径（CLAUDE.md §八）
--------------------
与 `scripts/k12_confirmation_config_power.py` 的 S 端完全一致：
- 单位是**逐日 NW(5) 标准误**外推到 31 折 3900 个交易日（`sqrt(881/3900)`）；
- 折内相关按 4 维（ΔADV_FT, ΔADV_ZS, IC500_FT, IC500_ZS）实测相关矩阵，
  敏感性 corr_scale ∈ {0, 0.5, 1.0}；
- 31 折按折间独立抽样（与 v3_joint / k12 同一近似；FT 臂相邻折训练窗重叠 83%，
  故折数门的真实 size 被低估，此限定须随数字一起披露）。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
K11_PATH = OUT / "k11_zs_ft_candidate_audit.json"
K11 = json.loads(K11_PATH.read_text(encoding="utf-8"))

ARMS = ("FT", "ZS")
NFOLD, N_CONFIRM_DAYS, N_IC_DAYS = 31, 3900, 881
NSIM, BATCH, SEED = 500_000, 25_000, 20260905
NSIM_MDE = 200_000

Z_SINGLE = 1.959963984540054          # k=1，v3 口径
Z_BOTH = 2.241402727604947            # k=2 Bonferroni，k12 design.z_both_bonferroni
Z_POWER80 = 0.8416212335729143        # Phi^-1(0.80)

KPOS = 18                             # 已冻结，ledger.md:283
KPOS_GRID = tuple(range(16, 23))      # 仅审计展示
DELTA_FLOOR = 0.0026                  # v3 量级门；v4 修订 3 已删

REFERENCE_DELTA = K11["by_arm"]["FT"]["delta_adv"]["mean"]
REFERENCE_TOP = K11["by_arm"]["FT"]["ic_top500"]["mean"]
HALF_DELTA, HALF_TOP = 0.0026, REFERENCE_TOP / 2.0
COMMON_TOP_NO_AMP = 0.020667959127347774   # k12:28（共同信号但无 ADV 放大）

INPUT_PROVENANCE = {
    "se_source_file": "outputs/k11_zs_ft_candidate_audit.json",
    "se_delta_FT_daily_nw5": K11["by_arm"]["FT"]["delta_adv"]["nw_se"],
    "se_delta_ZS_daily_nw5": K11["by_arm"]["ZS"]["delta_adv"]["nw_se"],
    "se_top_FT_daily_nw5": K11["by_arm"]["FT"]["ic_top500"]["nw_se"],
    "se_top_ZS_daily_nw5": K11["by_arm"]["ZS"]["ic_top500"]["nw_se"],
    "scaling": "sqrt(881/3900)，同 scripts/confirm_power_v3_joint.py:20-21 与 scripts/k12_confirmation_config_power.py:157",
    "correlation_source": "scripts/k12_confirmation_config_power.py:72-77（K11 开发折 36–42 逐日实测）",
    "kpos_gate_source": "experiments/ledger.md:283（18/31 事前选定）",
    "z_both_source": "outputs/k12_confirmation_config_power.json design.z_both_bonferroni",
    "reference_point_source": "experiments/ledger.md:280（FT ΔADV=+0.00517、top500 IC=+0.02584）",
    "sealed_or_unconsumed_data_read": False,
}

# 与 k12 完全相同的 4 维实测相关（顺序 delta_FT, delta_ZS, top_FT, top_ZS）。
S_CORR = np.array([
    [1.000000, -0.131230, 0.657950, -0.132427],
    [-0.131230, 1.000000, -0.174403, 0.692799],
    [0.657950, -0.174403, 1.000000, -0.244019],
    [-0.132427, 0.692799, -0.244019, 1.000000],
])
CORR_SCALES = (0.0, 0.5, 1.0)

SCENARIOS = {
    "both_reference_full": np.array(
        [REFERENCE_DELTA, REFERENCE_DELTA, REFERENCE_TOP, REFERENCE_TOP]),
    "both_reference_half": np.array([HALF_DELTA, HALF_DELTA, HALF_TOP, HALF_TOP]),
    "only_FT_reference_full": np.array([REFERENCE_DELTA, 0.0, REFERENCE_TOP, 0.0]),
    "only_ZS_reference_full": np.array([0.0, REFERENCE_DELTA, 0.0, REFERENCE_TOP]),
    "development_points_directional": np.array([
        K11["by_arm"]["FT"]["delta_adv"]["mean"],
        K11["by_arm"]["ZS"]["delta_adv"]["mean"],
        K11["by_arm"]["FT"]["ic_top500"]["mean"],
        K11["by_arm"]["ZS"]["ic_top500"]["mean"],
    ]),
    "both_null": np.zeros(4),
    "common_signal_no_adv_amplification": np.array(
        [0.0, 0.0, COMMON_TOP_NO_AMP, COMMON_TOP_NO_AMP]),
}

# 四种口径：v3（含量级门）/ v4（去量级门）× k=1 / k=2。
GATE_VARIANTS = {
    "v3_magnitude_z1.960": {"magnitude": True, "z": Z_SINGLE},
    "v3_magnitude_z2.2414": {"magnitude": True, "z": Z_BOTH},
    "v4_no_magnitude_z1.960": {"magnitude": False, "z": Z_SINGLE},
    "v4_no_magnitude_z2.2414": {"magnitude": False, "z": Z_BOTH},
}


def se_confirm() -> np.ndarray:
    scale = math.sqrt(N_IC_DAYS / N_CONFIRM_DAYS)
    return np.array([
        K11["by_arm"]["FT"]["delta_adv"]["nw_se"] * scale,
        K11["by_arm"]["ZS"]["delta_adv"]["nw_se"] * scale,
        K11["by_arm"]["FT"]["ic_top500"]["nw_se"] * scale,
        K11["by_arm"]["ZS"]["ic_top500"]["nw_se"] * scale,
    ])


def covariance(corr_scale: float, sd_fold: np.ndarray):
    corr = np.eye(4) + corr_scale * (S_CORR - np.eye(4))
    eig = float(np.linalg.eigvalsh(corr).min())
    if eig < -1e-10:
        raise RuntimeError(f"相关矩阵非半正定：{eig}")
    return corr * np.outer(sd_fold, sd_fold), eig


def gates(delta_mean, top_mean, kpos, se, magnitude, z, kpos_gate):
    passed = (delta_mean / se[:2] >= z) & (kpos >= kpos_gate) & (top_mean > 0)
    if magnitude:
        passed = passed & (delta_mean >= DELTA_FLOOR)
    return passed


def run_scenarios(se: np.ndarray, sd_fold: np.ndarray) -> dict:
    out = {}
    for corr_scale in CORR_SCALES:
        cov, eig = covariance(corr_scale, sd_fold)
        rng = np.random.default_rng(SEED + int(round(1000 * corr_scale)))
        counts = {
            v: {n: {"FT": 0, "ZS": 0, "any": 0} for n in SCENARIOS} for v in GATE_VARIANTS
        }
        kgrid = {
            v: {n: {k: 0 for k in KPOS_GRID} for n in SCENARIOS} for v in GATE_VARIANTS
        }
        done = 0
        while done < NSIM:
            n = min(BATCH, NSIM - done)
            noise = rng.multivariate_normal(np.zeros(4), cov, size=(n, NFOLD))
            noise_delta = noise[:, :, :2]
            noise_delta_mean = noise_delta.mean(axis=1)
            noise_top_mean = noise[:, :, 2:].mean(axis=1)
            for name, truth in SCENARIOS.items():
                delta_mean = noise_delta_mean + truth[:2]
                top_mean = noise_top_mean + truth[2:]
                kpos = (noise_delta > -truth[:2]).sum(axis=1)
                for variant, cfg in GATE_VARIANTS.items():
                    p = gates(delta_mean, top_mean, kpos, se,
                              cfg["magnitude"], cfg["z"], KPOS)
                    counts[variant][name]["FT"] += int(p[:, 0].sum())
                    counts[variant][name]["ZS"] += int(p[:, 1].sum())
                    counts[variant][name]["any"] += int((p[:, 0] | p[:, 1]).sum())
                    for k in KPOS_GRID:
                        pk = gates(delta_mean, top_mean, kpos, se,
                                   cfg["magnitude"], cfg["z"], k)
                        kgrid[variant][name][k] += int((pk[:, 0] | pk[:, 1]).sum())
            done += n
        out[f"corr_scale_{corr_scale:.1f}"] = {
            "correlation_scale": corr_scale,
            "correlation_eigen_min": eig,
            "variants": {
                variant: {
                    name: {
                        "power_FT_marginal": counts[variant][name]["FT"] / NSIM,
                        "power_ZS_marginal": counts[variant][name]["ZS"] / NSIM,
                        "power_any_arm": counts[variant][name]["any"] / NSIM,
                        "power_any_arm_by_kpos_audit_only": {
                            str(k): kgrid[variant][name][k] / NSIM for k in KPOS_GRID
                        },
                    }
                    for name in SCENARIOS
                }
                for variant in GATE_VARIANTS
            },
        }
        print(f"  corr_scale={corr_scale:.1f} 完成", flush=True)
    return out


class MdeSampler:
    """共同随机数：一次抽好噪声，再对不同真值平移，保证功效关于真值单调。

    情景族：两臂同为真值 delta_true，top500 IC 按参考幅度等比缩放
    （top = delta_true / REFERENCE_DELTA * REFERENCE_TOP），与 `both_reference_half`
    的构造法一致。这一等比假设须与数字一起披露。
    """

    def __init__(self, se: np.ndarray, sd_fold: np.ndarray, corr_scale: float):
        cov, _ = covariance(corr_scale, sd_fold)
        rng = np.random.default_rng(SEED + 7000 + int(round(1000 * corr_scale)))
        noise = rng.multivariate_normal(np.zeros(4), cov, size=(NSIM_MDE, NFOLD))
        self.se = se
        self.noise_delta = noise[:, :, :2]
        self.noise_delta_mean = self.noise_delta.mean(axis=1)
        self.noise_top_mean = noise[:, :, 2:].mean(axis=1)

    def power(self, delta_true: float, magnitude: bool, z: float) -> np.ndarray:
        top_true = delta_true / REFERENCE_DELTA * REFERENCE_TOP
        delta_mean = self.noise_delta_mean + delta_true
        top_mean = self.noise_top_mean + top_true
        kpos = (self.noise_delta > -delta_true).sum(axis=1)
        p = gates(delta_mean, top_mean, kpos, self.se, magnitude, z, KPOS)
        return p.mean(axis=0)


def mde80(se: np.ndarray, sd_fold: np.ndarray, corr_scale: float) -> dict:
    sampler = MdeSampler(se, sd_fold, corr_scale)
    result = {}
    for variant, cfg in GATE_VARIANTS.items():
        analytic = (cfg["z"] + Z_POWER80) * se[:2]
        if cfg["magnitude"]:
            analytic = np.maximum(analytic, DELTA_FLOOR)
        sim = []
        for arm_idx in range(2):
            lo, hi = 0.0, 0.020
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                if sampler.power(mid, cfg["magnitude"], cfg["z"])[arm_idx] < 0.80:
                    lo = mid
                else:
                    hi = mid
            sim.append(0.5 * (lo + hi))
        result[variant] = {
            "mde80_analytic_z_gate_only": dict(zip(ARMS, analytic.tolist())),
            "mde80_simulated_all_gates": dict(zip(ARMS, sim)),
            "note": (
                "解析值只含 z 门；模拟值含 z 门 + 18/31 折数门 + top500 符号门，故略高。"
                "量级门口径下解析值取 max(z 门, 0.0026)。"
            ),
        }
    return result


def analytic_z_only(se: np.ndarray) -> dict:
    """只含 CI（z）门的解析功效：Phi(mu/SE - z)。

    历史读数对账用：`ledger.md:412` 的「真值 0.0026 时 FT 51.5% / ZS 45.5%」与
    v4 草案 §2.5 的「97.8% / 约 52%」都是这一口径（单臂 z=1.96、无折数门、无符号门），
    **不是**含三门的联合功效。
    """
    from math import erf

    def phi(x: float) -> float:
        return 0.5 * (1.0 + erf(x / math.sqrt(2.0)))

    out = {}
    for label, z in (("z1.960", Z_SINGLE), ("z2.2414", Z_BOTH)):
        out[label] = {}
        for truth_label, delta_true in (
            ("reference_full_0.005174", REFERENCE_DELTA),
            ("reference_half_0.0026", HALF_DELTA),
        ):
            out[label][truth_label] = {
                arm: phi(delta_true / se[i] - z) for i, arm in enumerate(ARMS)
            }
    return out


def main() -> None:
    se = se_confirm()
    sd_fold = se * math.sqrt(NFOLD)
    print("SE_confirm:", dict(zip(
        ("delta_FT", "delta_ZS", "top_FT", "top_ZS"), se.round(7).tolist())), flush=True)

    scenarios = run_scenarios(se, sd_fold)
    print("情景模拟完成，开始 MDE80 二分…", flush=True)
    mde = {f"corr_scale_{cs:.1f}": mde80(se, sd_fold, cs) for cs in CORR_SCALES}

    result = {
        "purpose": (
            "v4 §8「未核」#2：H2 在 k=2、z=2.2414、去量级门条件下的功效与 MDE80 重算。"
            "估计交付，不作判定。"
        ),
        "design": {
            "n_folds": NFOLD,
            "n_simulations": NSIM,
            "n_simulations_mde": NSIM_MDE,
            "n_confirm_days": N_CONFIRM_DAYS,
            "z_single": Z_SINGLE,
            "z_both_bonferroni": Z_BOTH,
            "positive_fold_gate_frozen": KPOS,
            "positive_fold_gate_grid_audit_only": list(KPOS_GRID),
            "delta_magnitude_floor_v3_only": DELTA_FLOOR,
            "monte_carlo_max_se_probability": math.sqrt(0.25 / NSIM),
            "se_confirm": dict(zip(
                ("delta_FT", "delta_ZS", "top_FT", "top_ZS"), se.tolist())),
            "fold_independence_caveat": (
                "31 折按独立抽样；FT 臂相邻折训练窗重叠约 83%，折数门真实 size 被低估。"
            ),
        },
        "input_provenance": INPUT_PROVENANCE,
        "use_restriction": "估计交付，不作判定；不得用于选臂、选门槛或改变任何已冻结口径。",
        "analytic_ci_gate_only_power": analytic_z_only(se),
        "reconciliation": {
            "92.2_and_34.3": (
                "已溯源：= outputs/k12_confirmation_config_power.json "
                "scientific.corr_scale_1.0.scenarios.both_reference_full.both_ZS_marginal=0.922094 "
                "与 .both_reference_half.both_ZS_marginal=0.343534，即 **ZS 臂在 k=2 Bonferroni "
                "z=2.2414 下的边际功效**（含 v3 量级门、含 18/31 折数门与 top500 符号门）。"
                "本脚本 v3_magnitude_z2.2414 变体独立复现同值。"
            ),
            "97.8_and_52": (
                "= 单臂 z=1.96、**只含 CI 门**的解析值 Phi(0.005174/0.0013023-1.96)=0.9779 与 "
                "Phi(0.0026/0.0013023-1.96)=0.5145，见 analytic_ci_gate_only_power。"
                "含三门的模拟值为 0.975 / 0.504。"
            ),
            "51.5_and_45.5": (
                "= ledger.md:412，同为只含 CI 门的解析值，真值 0.0026 处 "
                "FT Phi(1.9964-1.96)=0.5145、ZS Phi(1.8471-1.96)=0.4551。"
            ),
            "magnitude_gate_is_redundant_at_k2": (
                "z=2.2414 时 CI 门要求 ΔADV_mean >= 2.2414*SE = 0.002919(FT) / 0.003155(ZS)，"
                "均严于 v3 量级门 0.0026，故**在 k=2 口径下删量级门对功效数字无任何影响**"
                "（v3_magnitude_z2.2414 与 v4_no_magnitude_z2.2414 逐位相同）。"
                "量级门只在 z=1.96 且 SE 较小的 FT 臂上有微弱约束。"
            ),
        },
        "scenarios": scenarios,
        "mde80": mde,
    }
    path = OUT / "confirm_power_v4_k2.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写入 {path}", flush=True)


if __name__ == "__main__":
    main()
