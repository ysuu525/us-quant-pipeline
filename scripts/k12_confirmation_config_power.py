"""K12: joint design power for one versus two confirmation configurations.

This is a design-only Monte Carlo.  It reads K11 summary parameters, never any
unconsumed fold, and evaluates E and S as separate claim families.  The full
pre-registration and decision rule are in experiments/ledger.md (2026-09-01).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
K11 = json.loads((OUT / "k11_zs_ft_candidate_audit.json").read_text(encoding="utf-8"))

ARMS = ("FT", "ZS")
NFOLD, FOLD_YEARS, N_CONFIRM_DAYS = 31, 0.5, 3900
NSIM, BATCH, SEED = 500_000, 25_000, 20260901
Z_SINGLE, Z_BOTH = 1.959963984540054, 2.241402727604947
KPOS, ECON_FLOOR, DELTA_FLOOR = 18, 5.0, 0.0026
REFERENCE_E = 10.48
REFERENCE_DELTA, REFERENCE_TOP = 0.005174221824799312, 0.025842180952147086
HALF_E, HALF_DELTA, HALF_TOP = 5.24, 0.0026, REFERENCE_TOP / 2.0
COMMON_TOP_NO_AMP = 0.020667959127347774
E_RHOS = (0.0, 0.21459392755247544, 0.5)
S_CORR_SCALES = (0.0, 0.5, 1.0)

E_SCENARIOS = {
    "both_reference_full": np.array([REFERENCE_E, REFERENCE_E]),
    "both_reference_half": np.array([HALF_E, HALF_E]),
    "only_FT_reference_full": np.array([REFERENCE_E, 0.0]),
    "only_ZS_reference_full": np.array([0.0, REFERENCE_E]),
    "development_points_directional": np.array([
        K11["by_arm"]["FT"]["gross_annual_arithmetic_pct"],
        K11["by_arm"]["ZS"]["gross_annual_arithmetic_pct"],
    ]),
    "both_null": np.zeros(2),
}

S_SCENARIOS = {
    # Vector order: delta_FT, delta_ZS, top_FT, top_ZS.
    "both_reference_full": np.array([
        REFERENCE_DELTA, REFERENCE_DELTA, REFERENCE_TOP, REFERENCE_TOP
    ]),
    "both_reference_half": np.array([
        HALF_DELTA, HALF_DELTA, HALF_TOP, HALF_TOP
    ]),
    "only_FT_reference_full": np.array([
        REFERENCE_DELTA, 0.0, REFERENCE_TOP, 0.0
    ]),
    "only_ZS_reference_full": np.array([
        0.0, REFERENCE_DELTA, 0.0, REFERENCE_TOP
    ]),
    "development_points_directional": np.array([
        K11["by_arm"]["FT"]["delta_adv"]["mean"],
        K11["by_arm"]["ZS"]["delta_adv"]["mean"],
        K11["by_arm"]["FT"]["ic_top500"]["mean"],
        K11["by_arm"]["ZS"]["ic_top500"]["mean"],
    ]),
    "both_null": np.zeros(4),
    "common_signal_no_adv_amplification": np.array([
        0.0, 0.0, COMMON_TOP_NO_AMP, COMMON_TOP_NO_AMP
    ]),
}

# Daily correlation from the consumed K11 window.  Sensitivity scales every
# off-diagonal toward zero, preserving positive semidefiniteness by convexity.
S_CORR = np.array([
    [1.000000, -0.131230, 0.657950, -0.132427],
    [-0.131230, 1.000000, -0.174403, 0.692799],
    [0.657950, -0.174403, 1.000000, -0.244019],
    [-0.132427, 0.692799, -0.244019, 1.000000],
])


def fresh_counts() -> dict[str, int]:
    return {
        "single_FT": 0,
        "single_ZS": 0,
        "both_any": 0,
        "both_FT_marginal": 0,
        "both_ZS_marginal": 0,
    }


def add_counts(counts: dict[str, int], p1: np.ndarray, p2: np.ndarray,
               q1: np.ndarray, q2: np.ndarray) -> None:
    """p uses single-config z; q uses two-config Bonferroni z."""
    counts["single_FT"] += int(p1.sum())
    counts["single_ZS"] += int(p2.sum())
    counts["both_any"] += int((q1 | q2).sum())
    counts["both_FT_marginal"] += int(q1.sum())
    counts["both_ZS_marginal"] += int(q2.sum())


def probabilities(counts: dict[str, int]) -> dict[str, float]:
    return {key: value / NSIM for key, value in counts.items()}


def economic_power() -> dict:
    # Mandatory HAC scaling: K11 stores the NW(5) SE of the daily mean over
    # 832 consumed days.  Preserve its long-run variance and scale the mean SE
    # to the planned 3,900-day confirmation window, then annualize to percent.
    n_development = min(K11["by_arm"][arm]["n_money_days"] for arm in ARMS)
    se_development_daily_bp = np.array([
        K11["by_arm"][arm]["gross_nw_se_daily_bp"] for arm in ARMS
    ])
    se_confirm = (
        se_development_daily_bp / 1e4
        * math.sqrt(n_development / N_CONFIRM_DAYS)
        * 252 * 100
    )
    sd_fold = se_confirm * math.sqrt(NFOLD)
    output = {}
    for rho in E_RHOS:
        corr = np.array([[1.0, rho], [rho, 1.0]])
        covariance = corr * np.outer(sd_fold, sd_fold)
        rng = np.random.default_rng(SEED + 1000 + int(round(1000 * rho)))
        counts = {name: fresh_counts() for name in E_SCENARIOS}
        completed = 0
        while completed < NSIM:
            n = min(BATCH, NSIM - completed)
            noise = rng.multivariate_normal(np.zeros(2), covariance, size=(n, NFOLD))
            for name, truth in E_SCENARIOS.items():
                folds = noise + truth
                means = folds.mean(axis=1)
                kpos = (folds > 0).sum(axis=1)
                pass_single = (
                    (means >= ECON_FLOOR)
                    & (means / se_confirm >= Z_SINGLE)
                    & (kpos >= KPOS)
                )
                pass_both = (
                    (means >= ECON_FLOOR)
                    & (means / se_confirm >= Z_BOTH)
                    & (kpos >= KPOS)
                )
                add_counts(
                    counts[name], pass_single[:, 0], pass_single[:, 1],
                    pass_both[:, 0], pass_both[:, 1]
                )
            completed += n
        output[f"rho_{rho:.6f}"] = {
            "rho": rho,
            "se_source": "K11 NW(5) daily-mean SE scaled sqrt(832/3900)",
            "se_confirm_annual_pct": dict(zip(ARMS, se_confirm.tolist())),
            "scenarios": {name: probabilities(value) for name, value in counts.items()},
        }
    return output


def scientific_power() -> dict:
    scale = math.sqrt(881 / N_CONFIRM_DAYS)
    se_confirm = np.array([
        K11["by_arm"]["FT"]["delta_adv"]["nw_se"] * scale,
        K11["by_arm"]["ZS"]["delta_adv"]["nw_se"] * scale,
        K11["by_arm"]["FT"]["ic_top500"]["nw_se"] * scale,
        K11["by_arm"]["ZS"]["ic_top500"]["nw_se"] * scale,
    ])
    sd_fold = se_confirm * math.sqrt(NFOLD)
    output = {}
    for corr_scale in S_CORR_SCALES:
        corr = np.eye(4) + corr_scale * (S_CORR - np.eye(4))
        eigen_min = float(np.linalg.eigvalsh(corr).min())
        if eigen_min < -1e-10:
            raise RuntimeError(f"Scientific correlation matrix is not PSD: {eigen_min}")
        covariance = corr * np.outer(sd_fold, sd_fold)
        rng = np.random.default_rng(SEED + 2000 + int(round(1000 * corr_scale)))
        counts = {name: fresh_counts() for name in S_SCENARIOS}
        completed = 0
        while completed < NSIM:
            n = min(BATCH, NSIM - completed)
            noise = rng.multivariate_normal(np.zeros(4), covariance, size=(n, NFOLD))
            for name, truth in S_SCENARIOS.items():
                folds = noise + truth
                delta, top = folds[:, :, :2], folds[:, :, 2:]
                delta_mean, top_mean = delta.mean(axis=1), top.mean(axis=1)
                kpos = (delta > 0).sum(axis=1)
                pass_single = (
                    (delta_mean >= DELTA_FLOOR)
                    & (delta_mean / se_confirm[:2] >= Z_SINGLE)
                    & (kpos >= KPOS)
                    & (top_mean > 0)
                )
                pass_both = (
                    (delta_mean >= DELTA_FLOOR)
                    & (delta_mean / se_confirm[:2] >= Z_BOTH)
                    & (kpos >= KPOS)
                    & (top_mean > 0)
                )
                add_counts(
                    counts[name], pass_single[:, 0], pass_single[:, 1],
                    pass_both[:, 0], pass_both[:, 1]
                )
            completed += n
        output[f"corr_scale_{corr_scale:.1f}"] = {
            "correlation_scale": corr_scale,
            "correlation_eigen_min": eigen_min,
            "se_confirm": {
                "delta_FT": float(se_confirm[0]),
                "delta_ZS": float(se_confirm[1]),
                "top_FT": float(se_confirm[2]),
                "top_ZS": float(se_confirm[3]),
            },
            "scenarios": {name: probabilities(value) for name, value in counts.items()},
        }
    return output


def decision_audit(economic: dict, scientific: dict) -> dict:
    false_success = []
    oracle_losses = []
    rescue_gains = []
    details = []

    for family, collection, false_scenario in (
        ("E", economic, "both_null"),
        ("S", scientific, "common_signal_no_adv_amplification"),
    ):
        for sensitivity, block in collection.items():
            scenarios = block["scenarios"]
            false_success.append(scenarios[false_scenario]["both_any"])
            for strong_arm, wrong_arm in (("FT", "ZS"), ("ZS", "FT")):
                scenario = scenarios[f"only_{strong_arm}_reference_full"]
                oracle = scenario[f"single_{strong_arm}"]
                wrong = scenario[f"single_{wrong_arm}"]
                both = scenario["both_any"]
                loss = oracle - both
                rescue = both - wrong
                oracle_losses.append(loss)
                rescue_gains.append(rescue)
                details.append({
                    "family": family,
                    "sensitivity": sensitivity,
                    "strong_arm": strong_arm,
                    "oracle_single_power": oracle,
                    "wrong_single_power": wrong,
                    "both_any_power": both,
                    "oracle_loss": loss,
                    "rescue_gain": rescue,
                })

    max_false = max(false_success)
    max_loss = max(oracle_losses)
    min_rescue = min(rescue_gains)
    criteria = {
        "familywise_false_success_le_5pct": max_false <= 0.05,
        "max_oracle_power_loss_le_5pp": max_loss <= 0.05,
        "min_wrong_single_rescue_ge_20pp": min_rescue >= 0.20,
    }
    return {
        "thresholds": {
            "max_false_success": 0.05,
            "max_oracle_power_loss": 0.05,
            "min_wrong_single_rescue": 0.20,
        },
        "worst_case": {
            "max_false_success": max_false,
            "max_oracle_power_loss": max_loss,
            "min_wrong_single_rescue": min_rescue,
        },
        "criteria": criteria,
        "recommendation": (
            "preregister_both_ZS_and_FT"
            if all(criteria.values()) else "single_ZS_by_B_layer_prior"
        ),
        "details": details,
    }


def main() -> None:
    print("K12 economic family...", flush=True)
    economic = economic_power()
    print("K12 scientific family...", flush=True)
    scientific = scientific_power()
    decision = decision_audit(economic, scientific)
    result = {
        "design": {
            "n_simulations": NSIM,
            "n_folds": NFOLD,
            "fold_years": FOLD_YEARS,
            "z_single": Z_SINGLE,
            "z_both_bonferroni": Z_BOTH,
            "positive_fold_gate": KPOS,
            "economic_floor_annual_pct": ECON_FLOOR,
            "delta_adv_floor": DELTA_FLOOR,
            "monte_carlo_max_se_probability": math.sqrt(0.25 / NSIM),
            "claims_are_separate": True,
        },
        "economic": economic,
        "scientific": scientific,
        "decision_audit": decision,
    }
    path = OUT / "k12_confirmation_config_power.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nK12 decision audit", flush=True)
    for key, value in decision["worst_case"].items():
        print(f"  {key}: {value:.4f}", flush=True)
    for key, value in decision["criteria"].items():
        print(f"  {key}: {value}", flush=True)
    print(f"  recommendation: {decision['recommendation']}", flush=True)
    print(f"写入 {path}", flush=True)


if __name__ == "__main__":
    main()
