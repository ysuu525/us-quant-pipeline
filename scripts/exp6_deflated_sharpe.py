"""实验 6：登记簿实测试验次数、Deflated Sharpe 与 haircut。

本实验是估计交付，不是检验。没有任何门槛，不得据其判定任何假设通过或失败。
不得用它去改 H1/H2/H3 的判据、SESOI 或功效计算（那些另有既定流程）。
第 5 步的收缩区间一经写下并登记，在读取折 05–35 之前不得修订；解封后只报
「实现收缩比」以及它落在区间内还是外，不回头改区间，也不据此改任何假设的结论。
全部输入来自已消耗的开发折与登记簿文本，不读任何封存产物。

FT / NT=5 / 8bp 的逐日净收益严格 import
``scripts/nt5_baseline_readout.py`` 的 ``_load_cam``、``scores_path``、
``scores_frame_to_by_day`` 与 ``frozen_long_only_returns``（源文件第 58、150--156 行
附近），不修改该脚本、不另写构造。登记簿字节在进程内只读取一次；计数、哈希、
类型分布、IC t 与跨试验 Sharpe 方差均从这一个不可变字节快照派生。

Haircut 的 Holm/BHY 需要目标统计量在整个检验族中的秩，不能只由 N 和目标 p 唯一
确定。本脚本为可复核地补足该输入：每条登记记录算一个 p；若一行有多个显式 t，
取其中最小的双侧 p；没有显式 t 的行记 p=1。目标统计量替换族内最弱的一个 p，
使族大小仍严格等于 N。Bonferroni、Holm step-down 与 BHY step-up 均对同一向量算。
同时报告目标秩、可解析 t 的行数与这一机械规则，避免把秩假装成已知量。
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

LEDGER = REPO / "experiments" / "ledger.md"
NT5_JSON = REPO / "outputs" / "nt5_baseline_readout.json"
OUT_JSON = REPO / "outputs" / "exp6_deflated_sharpe.json"
TARGET_TYPES = ("eval", "ablation-read", "DECISION")
TYPE_RE = re.compile(
    rb"^- (?P<date>\d{4}-\d{2}-\d{2}(?:T[^ ]*)?) \| (?P<type>[^|]+?) \|",
    re.MULTILINE,
)
T_RE = re.compile(
    r"(?<![A-Za-z])t(?:\([^)]*\))?(?:\s*[=:]\s*|\s+)([+-]?\d+(?:\.\d+)?)"
)
SHARPE_RE = re.compile(r"夏普\s*([+-]?\d+(?:\.\d+)?)")
EULER_GAMMA = 0.5772156649
TARGET_IC = 0.0206679591
PREDICTION_CENTER = 0.0207 * 0.75
FALLBACK_PAD = 0.0027
NORMAL = NormalDist()


@dataclass(frozen=True)
class LedgerLine:
    line_number: int
    date: str
    entry_type: str
    text: str


def _json_number(x: float) -> float | None:
    return float(x) if math.isfinite(float(x)) else None


def read_ledger_once(path: Path = LEDGER) -> tuple[bytes, list[LedgerLine]]:
    """Read ledger bytes exactly once and parse entry headers from that snapshot."""
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig")
    lines: list[LedgerLine] = []
    for number, line in enumerate(text.splitlines(), 1):
        match = re.match(
            r"^- (?P<date>\d{4}-\d{2}-\d{2}(?:T[^ ]*)?) \| (?P<type>[^|]+?) \|",
            line,
        )
        if match:
            lines.append(
                LedgerLine(number, match.group("date"), match.group("type").strip(), line)
            )
    return raw, lines


def ledger_snapshot(raw: bytes, lines: list[LedgerLine]) -> dict:
    distribution: dict[str, int] = {}
    for row in lines:
        distribution[row.entry_type] = distribution.get(row.entry_type, 0) + 1
    selected = {}
    for entry_type in TARGET_TYPES:
        subset = [row for row in lines if row.entry_type == entry_type]
        selected[entry_type] = {
            "count": len(subset),
            "first_date": subset[0].date if subset else None,
            "last_date": subset[-1].date if subset else None,
        }
    n_low = selected["eval"]["count"]
    n_high = n_low + selected["ablation-read"]["count"]
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "line_count": len(raw.decode("utf-8-sig").splitlines()),
        "selected_counts": selected,
        "type_distribution": dict(sorted(distribution.items())),
        "N_low": n_low,
        "N_high": n_high,
    }


def parse_same_family_sharpes(lines: list[LedgerLine]) -> dict:
    """Mechanically take the first explicitly labelled Sharpe per relevant readout line."""
    used = []
    for row in lines:
        if row.entry_type != "ablation-read":
            continue
        # These three textual markers define the return/backtest family before inspection.
        if "夏普" not in row.text or "8bp" not in row.text or "纯多" not in row.text:
            continue
        match = SHARPE_RE.search(row.text)
        if match:
            used.append({"line": row.line_number, "annualized_sharpe": float(match.group(1))})
    annualized = np.asarray([x["annualized_sharpe"] for x in used], dtype=float)
    if annualized.size >= 5:
        # DSR works on daily Sharpe, so annual Sharpe variance is divided by 252.
        variance = float(np.var(annualized / math.sqrt(252.0), ddof=1))
        method = "ledger_empirical_sample_variance"
    else:
        variance = math.nan
        method = "analytic_fallback_required"
    return {
        "rule": "ablation-read line contains 8bp, 纯多 and 夏普; first explicitly labelled 夏普",
        "entries": used,
        "n": int(annualized.size),
        "annualized_values": annualized.tolist(),
        "daily_sharpe_variance": _json_number(variance),
        "method": method,
    }


def parse_ic_t(lines: list[LedgerLine]) -> dict:
    """Copy the frozen 7-fold full-pool FT t from the ledger; fail closed if absent."""
    candidates = []
    pattern = re.compile(r"全池\s*\+?0\.02067\s*\(t\s*=\s*([+-]?\d+(?:\.\d+)?)\)")
    for row in lines:
        match = pattern.search(row.text)
        if match:
            candidates.append((row.line_number, float(match.group(1))))
    if not candidates:
        raise RuntimeError("ledger has no FT/lb90/sc=5/full-pool 0.02067 t; recomputation required")
    unique = {value for _, value in candidates}
    if len(unique) != 1:
        raise RuntimeError(f"conflicting ledger IC t values: {candidates}")
    return {"mean": TARGET_IC, "t": candidates[0][1], "source_lines": [x[0] for x in candidates],
            "t_source": "copied_from_ledger_not_recomputed"}


def committed_memory_gb() -> float | None:
    """Return Windows commit charge (page-file total minus available), if available."""
    if sys.platform != "win32":
        return None

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return float(status.ullTotalPageFile - status.ullAvailPageFile) / 2**30


def _load_nt5_module():
    spec = importlib.util.spec_from_file_location(
        "nt5_baseline_readout_for_exp6", REPO / "scripts" / "nt5_baseline_readout.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def reconstruct_ft_nt5_net_returns(summary_path: Path = NT5_JSON) -> pd.Series:
    """Reconstruct only the already-consumed FT dev-fold NT=5/8bp daily series."""
    nt5 = _load_nt5_module()
    cam = nt5._load_cam()
    nt5.assert_readable(cam.P)
    ret, oc, adv = cam.load_prices()
    parts = []
    for fold in nt5.FOLDS:
        score_file = nt5.scores_path(fold, "ft")
        nt5.assert_readable(score_file)
        scores = pd.read_parquet(score_file, columns=["PERMNO", "signal_date", "score"])
        by_day = nt5.scores_frame_to_by_day(scores, min_names=nt5.MIN_NAMES)
        del scores
        frame = nt5.frozen_long_only_returns(
            by_day, ret, oc, adv, topn=nt5.TOPN, cost_bp=8.0,
            exit_pct=nt5.EXIT_PCT, nt=nt5.NT, min_names=nt5.MIN_NAMES,
        )
        if frame.index.duplicated().any():
            raise AssertionError(f"fold{fold}: duplicate return dates")
        parts.append(frame["r"].rename(f"fold{fold}"))
    daily = pd.concat(parts).sort_index()
    if daily.index.duplicated().any():
        raise AssertionError("overlapping fold dates in NT=5 daily returns")
    prior = json.loads(summary_path.read_text(encoding="utf-8"))["arms"]["ft"]
    if len(daily) != int(prior["n_days"]):
        raise AssertionError(f"NT=5 day count changed: {len(daily)} != {prior['n_days']}")
    annual = float(daily.mean() * 252 * 100)
    vol = float(daily.std(ddof=1) * math.sqrt(252) * 100)
    if not math.isclose(annual, prior["net_by_cost_bp"]["8"]["net_annual_pct"], abs_tol=1e-12):
        raise AssertionError("reconstructed NT=5 annual return does not match baseline JSON")
    if not math.isclose(vol, prior["net_by_cost_bp"]["8"]["vol_annual_pct"], abs_tol=1e-12):
        raise AssertionError("reconstructed NT=5 volatility does not match baseline JSON")
    return daily


def return_moments(daily: pd.Series) -> dict:
    values = pd.Series(daily).dropna().to_numpy(dtype=float)
    if len(values) < 3 or np.std(values, ddof=1) <= 0:
        raise ValueError("at least three nonconstant daily returns are required")
    sr_daily = float(np.mean(values) / np.std(values, ddof=1))
    return {
        "T": len(values),
        "mean_daily": float(np.mean(values)),
        "std_daily": float(np.std(values, ddof=1)),
        "SR_daily": sr_daily,
        "SR_annualized": sr_daily * math.sqrt(252.0),
        "skewness_bias_corrected": float(skew(values, bias=False)),
        "kurtosis_nonexcess_bias_corrected": float(kurtosis(values, fisher=False, bias=False)),
    }


def expected_max_sharpe(n_trials: int, variance: float) -> float:
    if n_trials < 1 or variance < 0:
        raise ValueError("n_trials must be >=1 and variance must be nonnegative")
    if n_trials == 1 or variance == 0:
        return 0.0
    a = NORMAL.inv_cdf(1.0 - 1.0 / n_trials)
    b = NORMAL.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(variance) * ((1.0 - EULER_GAMMA) * a + EULER_GAMMA * b)


def deflated_sharpe_probability(moments: dict, variance: float, n_trials: int) -> dict:
    sr = moments["SR_daily"]
    sr0 = expected_max_sharpe(n_trials, variance)
    denominator_sq = (
        1.0 - moments["skewness_bias_corrected"] * sr
        + (moments["kurtosis_nonexcess_bias_corrected"] - 1.0) * sr * sr / 4.0
    )
    if denominator_sq <= 0:
        raise ValueError("nonpositive DSR moment correction denominator")
    z = (sr - sr0) * math.sqrt(moments["T"] - 1.0) / math.sqrt(denominator_sq)
    return {"N": n_trials, "SR0_daily": sr0, "z": z, "DSR": NORMAL.cdf(z)}


def maximum_n_at_dsr(moments: dict, variance: float, target: float = 0.95) -> int | None:
    """Largest integer N meeting target; None means it still passes at the 1e12 audit cap."""
    if deflated_sharpe_probability(moments, variance, 1)["DSR"] < target:
        return 0
    low, high, cap = 1, 2, 10**12
    while high < cap and deflated_sharpe_probability(moments, variance, high)["DSR"] >= target:
        low, high = high, min(cap, high * 2)
    if high == cap and deflated_sharpe_probability(moments, variance, high)["DSR"] >= target:
        return None
    while low + 1 < high:
        mid = (low + high) // 2
        if deflated_sharpe_probability(moments, variance, mid)["DSR"] >= target:
            low = mid
        else:
            high = mid
    return low


def two_sided_p(t_value: float) -> float:
    return 2.0 * (1.0 - NORMAL.cdf(abs(float(t_value))))


def ledger_trial_p_values(lines: list[LedgerLine], allowed_types: set[str]) -> tuple[list[float], int]:
    values, parsed = [], 0
    for row in lines:
        if row.entry_type not in allowed_types:
            continue
        t_values = [float(x) for x in T_RE.findall(row.text)]
        if t_values:
            values.append(min(two_sided_p(x) for x in t_values))
            parsed += 1
        else:
            values.append(1.0)
    return values, parsed


def adjusted_p_values(raw_p: list[float]) -> dict[str, np.ndarray]:
    p = np.clip(np.asarray(raw_p, dtype=float), 0.0, 1.0)
    n = len(p)
    if n == 0:
        raise ValueError("empty p-value family")
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    bonf = np.minimum(1.0, p * n)
    holm_ranked = np.minimum(1.0, np.maximum.accumulate(ranked * (n - np.arange(n))))
    c_n = float(np.sum(1.0 / np.arange(1, n + 1)))
    bhy_raw = ranked * n * c_n / np.arange(1, n + 1)
    bhy_ranked = np.minimum(1.0, np.minimum.accumulate(bhy_raw[::-1])[::-1])
    holm = np.empty(n); holm[order] = holm_ranked
    bhy = np.empty(n); bhy[order] = bhy_ranked
    return {"bonferroni": bonf, "holm": holm, "bhy": bhy}


def haircut_for_family(t_value: float, effect: float, family_p: list[float]) -> dict:
    """Replace the weakest family p with the focal p, preserving the disclosed N."""
    if not family_p:
        raise ValueError("empty p-value family")
    focal_p = two_sided_p(t_value)
    p = list(family_p)
    target_index = int(np.argmax(p))
    p[target_index] = focal_p
    corrections = adjusted_p_values(p)
    rank = int(np.argsort(np.argsort(np.asarray(p), kind="stable"), kind="stable")[target_index] + 1)
    out = {}
    for method, adjusted in corrections.items():
        p_adj = float(adjusted[target_index])
        t_adj = 0.0 if p_adj >= 1.0 else NORMAL.inv_cdf(1.0 - p_adj / 2.0)
        haircut = min(1.0, max(0.0, 1.0 - t_adj / abs(t_value)))
        out[method] = {
            "raw_p_two_sided": focal_p, "adjusted_p": p_adj,
            "adjusted_abs_t": t_adj, "haircut": haircut,
            "adjusted_effect": effect * (1.0 - haircut),
        }
    return {"N": len(p), "target_rank_by_raw_p": rank, "methods": out}


def prediction_interval(ic_haircuts: dict) -> dict:
    methods = ic_haircuts["methods"]
    values = {name: float(item["adjusted_effect"]) for name, item in methods.items()}
    strict = values["bonferroni"]
    loose = values["bhy"]
    same_side = all(x < PREDICTION_CENTER for x in values.values()) or all(
        x > PREDICTION_CENTER for x in values.values()
    )
    if same_side:
        lower = min(*values.values(), PREDICTION_CENTER) - FALLBACK_PAD
        upper = max(*values.values(), PREDICTION_CENTER) + FALLBACK_PAD
    else:
        if strict > loose:
            raise ValueError("BHY adjusted IC is below Bonferroni; stated interval ordering fails")
        lower, upper = strict, loose
    return {
        "target_definition": (
            "folds 05-35; FT arm; lb90; sample_count=5; full pool; "
            "pooled daily RankIC mean"
        ),
        "center": PREDICTION_CENTER,
        "haircut_values": values,
        "fallback_triggered": same_side,
        "fallback_pad": FALLBACK_PAD,
        "lower": lower,
        "upper": upper,
    }


def analyse(raw: bytes, lines: list[LedgerLine], daily: pd.Series) -> dict:
    ledger = ledger_snapshot(raw, lines)
    moments = return_moments(daily)
    variance_info = parse_same_family_sharpes(lines)
    variance = variance_info["daily_sharpe_variance"]
    if variance is None:
        variance = (1.0 + moments["SR_daily"] ** 2 / 2.0) / moments["T"]
        variance_info["method"] = "analytic_fallback"
        variance_info["warning"] = "fewer than 5 extractable entries; fallback understates SR0"
        variance_info["daily_sharpe_variance"] = variance
    dsr = {}
    maximum_n = maximum_n_at_dsr(moments, variance)
    for name in ("N_low", "N_high"):
        dsr[name] = deflated_sharpe_probability(moments, variance, ledger[name])
        dsr[name]["maximum_N_with_DSR_at_least_0.95"] = maximum_n

    ic = parse_ic_t(lines)
    years = moments["T"] / 252.0
    sharpe_t = moments["SR_annualized"] * math.sqrt(years)
    haircuts = {}
    for name, allowed in (
        ("N_low", {"eval"}), ("N_high", {"eval", "ablation-read"})
    ):
        family, parsed = ledger_trial_p_values(lines, allowed)
        if len(family) != ledger[name]:
            raise AssertionError(f"{name} p-value family size mismatch")
        haircuts[name] = {
            "family_t_parse": {
                "entries": len(family), "entries_with_explicit_t": parsed,
                "missing_t_entries_assigned_p": 1.0,
                "multi_t_entry_rule": "minimum two-sided p on the line",
                "target_insertion_rule": "replace family maximum p so N is unchanged",
            },
            "sharpe": haircut_for_family(sharpe_t, moments["SR_annualized"], family),
            "ic": haircut_for_family(ic["t"], ic["mean"], family),
        }
    prereg = prediction_interval(haircuts["N_high"]["ic"])
    return {
        "meta": {
            "run_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "consumed dev folds 36-42 only; no sealed artifacts",
            "returns_source": "imported nt5_baseline_readout.py construction, FT, NT=5, 8bp",
        },
        "ledger": ledger,
        "return_moments": moments,
        "sharpe_t": sharpe_t,
        "ic": ic,
        "var_sr_n": variance_info,
        "deflated_sharpe": dsr,
        "haircuts": haircuts,
        "preregistered_prediction": prereg,
    }


def print_report(result: dict) -> None:
    ledger = result["ledger"]
    print("=== ledger disclosure ===")
    print(f"sha256={ledger['sha256']} bytes={ledger['bytes']}")
    for kind, item in ledger["selected_counts"].items():
        print(f"{kind:14s} n={item['count']:3d}  {item['first_date']} .. {item['last_date']}")
    print("types: " + ", ".join(f"{k}={v}" for k, v in ledger["type_distribution"].items()))
    print(f"N_low={ledger['N_low']} N_high={ledger['N_high']}")
    m = result["return_moments"]
    print("\n=== FT NT=5 / 8bp net daily returns ===")
    print(f"T={m['T']} SR_daily={m['SR_daily']:.8f} SR_ann={m['SR_annualized']:.4f} "
          f"skew={m['skewness_bias_corrected']:.4f} kurt={m['kurtosis_nonexcess_bias_corrected']:.4f}")
    var = result["var_sr_n"]
    print(f"Var(SR_n)={var['daily_sharpe_variance']:.10g} method={var['method']} lines="
          f"{[x['line'] for x in var['entries']]}")
    for name, item in result["deflated_sharpe"].items():
        print(f"{name}: N={item['N']} SR0={item['SR0_daily']:.6f} DSR={item['DSR']:.6f} "
              f"maxN@95={item['maximum_N_with_DSR_at_least_0.95']}")
    print("\n=== haircut (N_high preregistration family) ===")
    high = result["haircuts"]["N_high"]
    for endpoint in ("sharpe", "ic"):
        print(endpoint.upper())
        for method, item in high[endpoint]["methods"].items():
            print(f"  {method:11s} p_adj={item['adjusted_p']:.6g} haircut={item['haircut']:.2%} "
                  f"adjusted={item['adjusted_effect']:.8f}")
    pred = result["preregistered_prediction"]
    print("\n=== preregistered shrinkage prediction ===")
    print(f"center={pred['center']:.8f} interval=[{pred['lower']:.8f}, {pred['upper']:.8f}] "
          f"fallback={pred['fallback_triggered']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--nt5-summary", type=Path, default=NT5_JSON)
    parser.add_argument("--out", type=Path, default=OUT_JSON)
    parser.add_argument("--max-committed-gb", type=float, default=40.0)
    args = parser.parse_args()

    raw, lines = read_ledger_once(args.ledger)
    memory = committed_memory_gb()
    if memory is not None and memory > args.max_committed_gb:
        print(
            f"REFUSE heavy reconstruction: committed memory {memory:.2f}GB exceeds "
            f"{args.max_committed_gb:.2f}GB",
            file=sys.stderr,
        )
        return 75
    daily = reconstruct_ft_nt5_net_returns(args.nt5_summary)
    result = analyse(raw, lines, daily)
    result["meta"]["committed_memory_gb_before_reconstruction"] = memory
    result["meta"]["committed_memory_limit_gb"] = args.max_committed_gb
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(args.out)
    print_report(result)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
