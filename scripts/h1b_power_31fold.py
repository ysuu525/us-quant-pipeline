"""H1b（扩充控制集张成后的 alpha）在 31 折确认集上的 SE 与 MDE80 外推。

出处 / 为什么跑这一条
--------------------
`experiments/confirmation_protocol_v4_draft_2026-09-05.md` §2.4 与 §8「未核」#5：
`ledger.md:483` 自陈「H1b 在 31 折上的 SE / MDE80 未算」，而 [待裁定 C] 的 C-1 版
写明「前置：冻结前须先在开发折 SE 上做 31 折外推、把 MDE80 算出来并登记
（方法同 `ledger.md:430`），否则 §7 第 10 条不得打勾」。本脚本只补这一个数。

判据 / 用途限制（**先于运行写死**，CLAUDE.md §二）
-------------------------------------------------
1. **估计交付**，不作判定：输出是 SE₃₁ 与 MDE80，**不产生 PASS/FAIL**，
   不改任何冻结口径，不据此决定 H1b 是否进入确认集（那是用户裁定 C）。
2. **主规格先验固定为 S-TH-ind**（v4 [待裁定 C] C-1 原文：自由参数最少 + 唯一做了
   行业中性 = B 层优先序 ②④）。三规格都报，但**不得按结果换主规格**。
3. 推断单位是**逐日**：`exp11` 的 `nw5_t_alpha` 是逐日 NW(5) OLS 截距 t，
   故 SE_dev = alpha_ann_pct / nw5_t_alpha 是**精确复原**，不是近似。
   逐折 alpha（7 个）只作**一致性指标**，不作标准误的推断单位（CLAUDE.md §二）。
4. 外推同 `ledger.md:307` / `:430`：`SE₃₁ = SE_dev × sqrt(n_dev / n_31)`
   （保长期方差，按均值样本量缩放）。n_31 主口径取项目惯用的 **3900 交易日**，
   并列给出按**实际交易日历**算出的版本。
5. **不读取**折 05–35 / eval_sealed_* / 任何 ≥2024-01-01 的分数、标签、收益。
   本脚本对 `processed/market_index.parquet` **只取 `caldt` 一列**（交易日历日期），
   用于机械生成折边界并数交易日；不读任何收益列。
6. H1b **无预注册 SESOI**，故本脚本只报 MDE，**不作「MDE ≤ SESOI」判定**。

口径（CLAUDE.md §八）
--------------------
exp11 的冻结构造是 **K6b / NT=6**（`outputs/exp11_spanning_extended.json`
`meta.k6b_frozen_construction.nt = 6`），与 09-03 已改的 NT=5 生产口径不同；
本读数只在 exp11 自身口径内有效，**不得与 NT=5 的读数并列比较**。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from crsp_pipeline.calendar import TradingCalendar  # noqa: E402
from crsp_pipeline.splits import walk_forward_folds  # noqa: E402

OUT = REPO / "outputs"
EXP11 = json.loads((OUT / "exp11_spanning_extended.json").read_text(encoding="utf-8"))

SPECS = ("S-T", "S-H", "S-TH-ind")
PRIMARY_SPEC = "S-TH-ind"          # 先验固定，见 docstring 第 2 条
DEV_FOLDS = tuple(range(36, 43))
CONFIRM_FOLDS = tuple(range(5, 36))
N_CONFIRM_DAYS_CONVENTION = 3900   # ledger.md:307/:430 惯用值
Z_SINGLE, Z_BOTH, Z_POWER80 = 1.959963984540054, 2.241402727604947, 0.8416212335729143
MDE80_MULT_SINGLE = Z_SINGLE + Z_POWER80          # 2.8016
MDE80_MULT_BOTH = Z_BOTH + Z_POWER80              # 3.0830
FOLD_CALENDAR_START = "2000-01-03"                # 同 scripts/emit_folds.py:34


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def fold_session_counts(processed: Path) -> dict:
    market = pd.read_parquet(processed / "market_index.parquet", columns=["caldt"])
    cal = TradingCalendar.from_market_index(market, "caldt")
    folds = walk_forward_folds(cal, FOLD_CALENDAR_START, cal.dates[-1])
    per_fold = {}
    for i, f in enumerate(folds, start=1):
        per_fold[i] = {
            "val_start": str(f.val_start.date()),
            "val_end": str(f.val_end.date()),
            "sessions": int(len(cal.sessions(f.val_start, f.val_end))),
        }
    return per_fold


def main() -> None:
    processed = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("F:/quant/processed")
    per_fold = fold_session_counts(processed)

    dev_sessions = sum(per_fold[i]["sessions"] for i in DEV_FOLDS)
    confirm_sessions = sum(per_fold[i]["sessions"] for i in CONFIRM_FOLDS)

    results = {}
    for spec in SPECS:
        block = EXP11["results"][spec]
        alpha = float(block["alpha_ann_pct"])
        t = float(block["nw5_t_alpha"])
        n_dev_regression_days = int(block["n_days"])
        se_dev = alpha / t

        # 主口径：n_31 = 3900（项目惯例）。
        se31_convention = se_dev * math.sqrt(n_dev_regression_days / N_CONFIRM_DAYS_CONVENTION)
        # 并列口径：按实际交易日历。折 05–35 的可用回归日按开发折实测比例折算
        #（exp11 的 832 个回归日 / 开发折日历交易日），保守起见也给纯日历版本。
        usable_ratio = n_dev_regression_days / dev_sessions
        n31_calendar = confirm_sessions
        n31_expected_regression = confirm_sessions * usable_ratio
        se31_calendar = se_dev * math.sqrt(n_dev_regression_days / n31_calendar)
        se31_expected = se_dev * math.sqrt(n_dev_regression_days / n31_expected_regression)

        per_fold_alpha = block["per_fold_fixed_loading_residual_alpha_ann_pct"]
        fold_values = np.array(list(per_fold_alpha.values()), dtype=float)
        se_fold_level = float(fold_values.std(ddof=1) / math.sqrt(len(fold_values)))

        results[spec] = {
            "is_primary_spec_by_prior": spec == PRIMARY_SPEC,
            "dev_alpha_ann_pct": alpha,
            "dev_nw5_t_alpha": t,
            "dev_n_regression_days": n_dev_regression_days,
            "dev_retention_pct": float(block["retention_pct"]),
            "dev_folds_alpha_positive": int(block["folds_alpha_positive"]),
            "se_dev_ann_pct_nw5": se_dev,
            "se31_ann_pct": {
                "convention_3900d": se31_convention,
                "actual_calendar_sessions": se31_calendar,
                "expected_usable_regression_days": se31_expected,
            },
            "mde80_ann_pct_z1.960": {
                "convention_3900d": MDE80_MULT_SINGLE * se31_convention,
                "actual_calendar_sessions": MDE80_MULT_SINGLE * se31_calendar,
                "expected_usable_regression_days": MDE80_MULT_SINGLE * se31_expected,
            },
            "mde80_ann_pct_z2.2414_k2": {
                "convention_3900d": MDE80_MULT_BOTH * se31_convention,
                "actual_calendar_sessions": MDE80_MULT_BOTH * se31_calendar,
                "expected_usable_regression_days": MDE80_MULT_BOTH * se31_expected,
            },
            # ledger.md 2026-09-03 correction 的教训：报功效必须写明是「对 SESOI」
            # 还是「对预期效应」。H1b 无 SESOI，故以下只是**对预期效应**的功效，
            # 不能替代功效关判定。折扣系数沿用 ledger.md:430 的 E 端惯例。
            "power_at_expected_effect_not_at_sesoi": {
                "note": (
                    "描述性，不是功效关判定；H1b 无预注册 SESOI。"
                    "折扣 0.75 = 选择偏差、0.751 = K15 σ_cs 年代比（ledger.md:430 惯例）。"
                ),
                "truths_ann_pct": {
                    "dev_point_estimate": alpha,
                    "x0.75_selection": alpha * 0.75,
                    "x0.75x0.751_selection_and_era": alpha * 0.75 * 0.751,
                },
                "power_z1.960_convention_3900d": {
                    "dev_point_estimate": _phi(alpha / se31_convention - Z_SINGLE),
                    "x0.75_selection": _phi(alpha * 0.75 / se31_convention - Z_SINGLE),
                    "x0.75x0.751_selection_and_era":
                        _phi(alpha * 0.75 * 0.751 / se31_convention - Z_SINGLE),
                },
                "power_z2.2414_convention_3900d": {
                    "dev_point_estimate": _phi(alpha / se31_convention - Z_BOTH),
                    "x0.75_selection": _phi(alpha * 0.75 / se31_convention - Z_BOTH),
                    "x0.75x0.751_selection_and_era":
                        _phi(alpha * 0.75 * 0.751 / se31_convention - Z_BOTH),
                },
            },
            "fold_level_cross_check_only": {
                "se_of_mean_over_7_folds_ann_pct": se_fold_level,
                "implied_t": alpha / se_fold_level if se_fold_level else None,
                "caveat": (
                    "折级 SE 只作一致性交叉核对，**不是推断单位**（CLAUDE.md §二 K7a 教训）；"
                    "正式数字用逐日 NW(5)。"
                ),
            },
        }

    payload = {
        "purpose": (
            "v4 §8「未核」#5 / [待裁定 C] 前置：H1b 三规格在 31 折上的 SE 与 MDE80 外推。"
            "估计交付，不作判定。"
        ),
        "primary_spec_fixed_by_prior": PRIMARY_SPEC,
        "use_restriction": (
            "估计交付；不得据此判定 H1b 是否进入确认集（用户裁定 C），"
            "不得按结果更换主规格，不得与 NT=5 口径读数并列比较。"
        ),
        "input_provenance": {
            "dev_readout_file": "outputs/exp11_spanning_extended.json",
            "dev_readout_ledger_line": "experiments/ledger.md:483（[gpt-exp 11]，折 36–42）",
            "se_recovery": "SE_dev = alpha_ann_pct / nw5_t_alpha（exp11 的 t 是逐日 NW(5) OLS 截距 t，scripts/exp11_spanning_extended.py:420-455）",
            "extrapolation_rule": "SE31 = SE_dev * sqrt(n_dev / n_31)，同 experiments/ledger.md:307 与 :430",
            "mde80_rule": "MDE80 = (z + 0.8416) * SE31；z=1.96 单臂、z=2.2414 为 k=2 Bonferroni",
            "calendar_source": "processed/market_index.parquet 仅 caldt 列 + crsp_pipeline.splits.walk_forward_folds（同 scripts/emit_folds.py）",
            "construction_caliber": EXP11["meta"]["k6b_frozen_construction"],
            "sealed_or_unconsumed_data_read": False,
        },
        "day_counts": {
            "dev_folds": list(DEV_FOLDS),
            "dev_calendar_sessions": dev_sessions,
            "dev_regression_days_in_exp11": int(EXP11["results"][PRIMARY_SPEC]["n_days"]),
            "confirm_folds": list(CONFIRM_FOLDS),
            "confirm_calendar_sessions": confirm_sessions,
            "confirm_expected_usable_regression_days": (
                confirm_sessions
                * EXP11["results"][PRIMARY_SPEC]["n_days"] / dev_sessions
            ),
            "convention_n_confirm_days": N_CONFIRM_DAYS_CONVENTION,
            "confirm_fold_windows": {
                f"fold{i:02d}": per_fold[i] for i in CONFIRM_FOLDS
            },
        },
        "sesoi": None,
        "sesoi_note": (
            "H1b 无预注册 SESOI（v4 §2.4 未给），故只报 MDE，不作「MDE ≤ SESOI」判定。"
            "若日后要把 H1b 升为确认性检验，须先独立地写下 SESOI。"
        ),
        "results": results,
    }
    path = OUT / "h1b_power_31fold.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"开发折日历交易日 {dev_sessions}，exp11 回归日 "
          f"{EXP11['results'][PRIMARY_SPEC]['n_days']}", flush=True)
    print(f"折 05–35 日历交易日 {confirm_sessions}，预期可用回归日约 "
          f"{payload['day_counts']['confirm_expected_usable_regression_days']:.0f}", flush=True)
    for spec in SPECS:
        r = results[spec]
        print(f"{spec:<10} alpha={r['dev_alpha_ann_pct']:.4f}%  t={r['dev_nw5_t_alpha']:.4f}  "
              f"SE_dev={r['se_dev_ann_pct_nw5']:.4f}%  "
              f"SE31(3900d)={r['se31_ann_pct']['convention_3900d']:.4f}%  "
              f"MDE80(z1.96)={r['mde80_ann_pct_z1.960']['convention_3900d']:.4f}%  "
              f"MDE80(z2.2414)={r['mde80_ann_pct_z2.2414_k2']['convention_3900d']:.4f}%",
          flush=True)
    print(f"写入 {path}", flush=True)


if __name__ == "__main__":
    main()
