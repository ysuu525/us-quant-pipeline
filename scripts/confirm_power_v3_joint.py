"""Direct joint power for the frozen confirmation-set scientific endpoint.

Unlike confirm_power_v2.py, this draws 31 fold estimates and evaluates the
magnitude, CI, fold-count, and top500-sign gates on the same simulated sample.
No confirmation or sealed observations are read.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


OUT = Path(__file__).resolve().parents[1] / "outputs"
NFOLD, NSIM, BATCH, SEED = 31, 500_000, 25_000, 20260901
DELTA_THRESHOLD = 0.0026
KPOS_THRESHOLD = 21
KPOS_GRID = tuple(range(18, 24))
SE_CONFIRM_DELTA = 0.0027400190490196826 * np.sqrt(881 / 3900)
SE_CONFIRM_TOP = 0.007218217988527544 * np.sqrt(881 / 3900)
SE_FOLD_DELTA = SE_CONFIRM_DELTA * np.sqrt(NFOLD)
SE_FOLD_TOP = SE_CONFIRM_TOP * np.sqrt(NFOLD)
RHOS = (0.0, 0.65795, 0.905576)
SCENARIOS = {
    "modern_full": {"delta": 0.005174221824799312, "top": 0.025842180952147086},
    "modern_half": {"delta": 0.0026, "top": 0.012921090476073543},
    "common_signal_no_adv_amplification": {"delta": 0.0, "top": 0.020667959127347774},
    "no_signal": {"delta": 0.0, "top": 0.0},
}


def simulate(delta_true: float, top_true: float, rho: float, rng: np.random.Generator) -> dict:
    total = modern = ci_and_size = fold_gate = sign_gate = 0
    modern_by_k = {k: 0 for k in KPOS_GRID}
    delta_sum = k_sum = 0.0
    covariance = np.array([
        [SE_FOLD_DELTA ** 2, rho * SE_FOLD_DELTA * SE_FOLD_TOP],
        [rho * SE_FOLD_DELTA * SE_FOLD_TOP, SE_FOLD_TOP ** 2],
    ])
    mean = np.array([delta_true, top_true])
    while total < NSIM:
        n = min(BATCH, NSIM - total)
        draws = rng.multivariate_normal(mean, covariance, size=(n, NFOLD))
        delta = draws[:, :, 0]
        top = draws[:, :, 1]
        delta_bar = delta.mean(axis=1)
        top_bar = top.mean(axis=1)
        kpos = (delta > 0).sum(axis=1)
        gate_ci_size = (delta_bar >= DELTA_THRESHOLD) & (
            delta_bar - 1.959963984540054 * SE_CONFIRM_DELTA > 0
        )
        gate_fold = kpos >= KPOS_THRESHOLD
        gate_sign = top_bar > 0
        passed = gate_ci_size & gate_fold & gate_sign
        modern += int(passed.sum())
        for k in KPOS_GRID:
            modern_by_k[k] += int((gate_ci_size & (kpos >= k) & gate_sign).sum())
        ci_and_size += int(gate_ci_size.sum())
        fold_gate += int(gate_fold.sum())
        sign_gate += int(gate_sign.sum())
        delta_sum += float(delta_bar.sum())
        k_sum += float(kpos.sum())
        total += n
    return {
        "joint_modern_probability": modern / total,
        "joint_modern_probability_by_positive_fold_threshold": {
            str(k): modern_by_k[k] / total for k in KPOS_GRID
        },
        "ci_and_magnitude_probability": ci_and_size / total,
        "fold_gate_probability": fold_gate / total,
        "top500_positive_sign_probability": sign_gate / total,
        "mean_simulated_delta": delta_sum / total,
        "mean_positive_folds": k_sum / total,
    }


def main() -> None:
    result = {
        "design": {
            "n_folds": NFOLD,
            "n_simulations": NSIM,
            "delta_threshold": DELTA_THRESHOLD,
            "positive_fold_threshold": KPOS_THRESHOLD,
            "positive_fold_threshold_grid": list(KPOS_GRID),
            "se_confirm_delta": float(SE_CONFIRM_DELTA),
            "se_confirm_top500_ic": float(SE_CONFIRM_TOP),
            "rho_sensitivity": list(RHOS),
            "note": "Normal half-year fold design approximation; not a claim of exact HAC size.",
        },
        "scenarios": {},
    }
    for rho in RHOS:
        rng = np.random.default_rng(SEED + int(round(rho * 1000)))
        key = f"rho_{rho:.6f}"
        result["scenarios"][key] = {}
        print(f"rho={rho:.3f}", flush=True)
        for name, truth in SCENARIOS.items():
            row = simulate(truth["delta"], truth["top"], rho, rng)
            result["scenarios"][key][name] = {"truth": truth, **row}
            print(
                f"  {name:<35} joint={row['joint_modern_probability']:.3f} "
                f"CI+size={row['ci_and_magnitude_probability']:.3f} "
                f"k={row['fold_gate_probability']:.3f} sign={row['top500_positive_sign_probability']:.3f}",
                flush=True,
            )
    (OUT / "confirm_power_v3_joint.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("写入 outputs/confirm_power_v3_joint.json", flush=True)


if __name__ == "__main__":
    main()
