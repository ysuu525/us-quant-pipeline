"""消融臂配对检验（预注册 §2 的判据实现）。

    python scripts/compare_arms.py \
        --arm A=outputs/fold01_lb90_s0/eval_poolA_full \
        --arm A=outputs/fold02_lb90_s0/eval_poolA_full_fold02 \
        --arm B=outputs/fold01_lb90_s0_poolB_universe/eval_poolB_universe \
        --arm B=outputs/fold02_..._poolB_universe/eval_..._fold02 \
        --out outputs/ablation_pool_lb90.md

判据（预注册 §2 第 1/2 轮，逐字实现）：

    淘汰档 T ⟺ 逐日配对差（最优档 − T）在 4 折合并序列上的 block
    bootstrap（块长 10 交易日、10000 次）99% CI 全体 > 0（第二轮 95%），
    且 4 折中 ≥3 折的折内均值差 > 0。

**为什么配对**：同一天两臂打分的是同一批股票、对同一批标签，折级共同噪声
（当天市场好不好做）在做差时被消掉，功效远高于各自算均值再比。

**块自助的实现选择**：块在**折内**抽取，不跨折边界（跨折的"相邻"日在
日历上相隔数月，拼在一个块里没有意义）；每折重采样回原长度后拼接，
再算总体均值。块长 10 交易日覆盖 6 日标签视野造成的重叠自相关。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from kronos_ft.train import append_ledger  # noqa: E402

BLOCK_LEN = 10
N_BOOT = 10000
BOOT_SEED = 20260827


def load_daily_ic(eval_dir: Path) -> pd.Series:
    """读单次评估的逐日 RankIC（evaluate_fold.py 的 daily_ic.parquet）。"""
    p = eval_dir / "daily_ic.parquet"
    if not p.is_file():
        raise FileNotFoundError(f"缺 daily_ic.parquet: {eval_dir}")
    df = pd.read_parquet(p)
    s = df["rank_ic"] if "rank_ic" in df.columns else df.iloc[:, 0]
    s.index = pd.to_datetime(df.index)
    return s.dropna().sort_index()


def block_bootstrap_mean(
    per_fold: list[np.ndarray], n_boot: int = N_BOOT,
    block_len: int = BLOCK_LEN, seed: int = BOOT_SEED,
) -> np.ndarray:
    """折内移动块自助，返回 n_boot 个总体均值。

    per_fold : 每折一条配对差序列（按日期排序）。块只在折内抽，不跨折。
    """
    rng = np.random.default_rng(seed)
    folds = [np.asarray(x, dtype=float) for x in per_fold if len(x) > 0]
    if not folds:
        return np.array([])
    out = np.empty(n_boot)
    for b in range(n_boot):
        parts = []
        for x in folds:
            n = len(x)
            L = min(block_len, n)
            n_blocks = int(np.ceil(n / L))
            starts = rng.integers(0, n - L + 1, size=n_blocks)
            parts.append(np.concatenate([x[s:s + L] for s in starts])[:n])
        out[b] = float(np.concatenate(parts).mean())
    return out


def compare(a_folds: dict[str, pd.Series], b_folds: dict[str, pd.Series]) -> dict:
    """两臂配对比较。键 = 折标识；同折按日期内连接（两臂应同日同票）。"""
    common_folds = sorted(set(a_folds) & set(b_folds))
    diffs, per_fold_mean, per_fold_n = [], {}, {}
    for f in common_folds:
        joined = pd.concat([a_folds[f].rename("a"), b_folds[f].rename("b")],
                           axis=1, join="inner").dropna()
        d = (joined["b"] - joined["a"]).to_numpy()
        diffs.append(d)
        per_fold_mean[f] = float(d.mean()) if len(d) else float("nan")
        per_fold_n[f] = int(len(d))

    all_d = np.concatenate(diffs) if diffs else np.array([])
    boot = block_bootstrap_mean(diffs)
    n_folds_positive = sum(1 for v in per_fold_mean.values() if v > 0)
    res = {
        "folds": common_folds,
        "n_days_total": int(len(all_d)),
        "n_days_by_fold": per_fold_n,
        "mean_diff": float(all_d.mean()) if len(all_d) else float("nan"),
        "mean_diff_by_fold": per_fold_mean,
        "n_folds_positive": n_folds_positive,
        "n_folds": len(common_folds),
        "block_len": BLOCK_LEN, "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
    }
    if len(boot):
        for lvl, lo_q, hi_q in [("99", 0.5, 99.5), ("95", 2.5, 97.5)]:
            lo, hi = np.percentile(boot, [lo_q, hi_q])
            res[f"ci{lvl}"] = [float(lo), float(hi)]
            res[f"ci{lvl}_all_positive"] = bool(lo > 0)
            res[f"ci{lvl}_all_negative"] = bool(hi < 0)
        res["p_boot_le_zero"] = float((boot <= 0).mean())
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="消融臂配对检验（预注册 §2 判据）")
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=EVAL_DIR",
                    help="重复给出，如 --arm A=outputs/.../eval_x --arm B=...")
    ap.add_argument("--baseline", default=None,
                    help="基准臂名（默认取第一个出现的臂）；报告 其他臂 − 基准")
    ap.add_argument("--out", default=None, help="报告输出路径（.md）")
    ap.add_argument("--ledger", action="store_true",
                    help="把结论追加到 experiments/ledger.md")
    args = ap.parse_args()

    # 同一 (臂, 折) 给多个目录 = 多个 seed → 逐日 IC 按 seed 平均（预注册 §2
    # 第二轮"补 seeds"的聚合口径：降低 seed 噪声后再做配对，不是把 seed 当
    # 独立观测堆进序列）。
    raw: dict[str, dict[str, list[pd.Series]]] = defaultdict(lambda: defaultdict(list))
    order: list[str] = []
    for spec in args.arm:
        if "=" not in spec:
            ap.error(f"--arm 需要 NAME=EVAL_DIR 形式: {spec}")
        name, path = spec.split("=", 1)
        s = load_daily_ic(Path(path))
        # 折标识：用该评估的日期区间（同折各臂各 seed 必然相同）
        fold_key = f"{s.index.min():%Y-%m-%d}..{s.index.max():%Y-%m-%d}"
        raw[name][fold_key].append(s)
        if name not in order:
            order.append(name)
        print(f"载入 {name} {fold_key}: {len(s)} 日, 均值 {s.mean():+.5f}", flush=True)

    arms: dict[str, dict[str, pd.Series]] = {}
    for name, folds in raw.items():
        arms[name] = {}
        for fk, series_list in folds.items():
            if len(series_list) == 1:
                arms[name][fk] = series_list[0]
            else:
                merged = pd.concat(series_list, axis=1).mean(axis=1)
                arms[name][fk] = merged
                print(f"  {name} {fk}: {len(series_list)} 个 seed 逐日平均 "
                      f"→ 均值 {merged.mean():+.5f}", flush=True)
    n_seeds = {n: {fk: len(v) for fk, v in f.items()} for n, f in raw.items()}
    if len({s for f in n_seeds.values() for s in f.values()}) > 1:
        print(f"⚠ 各臂/折的 seed 数不一致: {n_seeds}——配对功效不均，"
              f"结论慎读", flush=True)

    base = args.baseline or order[0]
    if base not in arms:
        ap.error(f"基准臂 {base} 不在给出的臂中: {order}")

    lines = ["# 消融臂配对检验（预注册 §2 判据）", "",
             f"生成时间：{datetime.now(timezone.utc).isoformat(timespec='seconds')}",
             f"基准臂：**{base}**；块自助：块长 {BLOCK_LEN} 交易日（折内抽样）、"
             f"{N_BOOT} 次、seed {BOOT_SEED}", ""]
    for name, folds in arms.items():
        overall = pd.concat(folds.values())
        lines.append(f"- 臂 **{name}**：{len(folds)} 折 / {len(overall)} 日，"
                     f"逐日 RankIC 均值 {overall.mean():+.5f}")
    lines.append("")

    results = {}
    for name in order:
        if name == base:
            continue
        r = compare(arms[base], arms[name])
        results[name] = r
        lines += [
            f"## {name} − {base}", "",
            f"- 配对日数 {r['n_days_total']}（{r['n_folds']} 折）；"
            f"**平均配对差 {r['mean_diff']:+.5f}**",
            f"- 折内均值差：" + "，".join(
                f"{k.split('..')[0]} {v:+.5f}" for k, v in r["mean_diff_by_fold"].items()),
            f"- 折内为正：**{r['n_folds_positive']}/{r['n_folds']}**"
            f"（预注册要求 ≥3/4 同向）",
            f"- 99% CI [{r['ci99'][0]:+.5f}, {r['ci99'][1]:+.5f}]"
            f" → 全体 > 0：**{r['ci99_all_positive']}**",
            f"- 95% CI [{r['ci95'][0]:+.5f}, {r['ci95'][1]:+.5f}]"
            f" → 全体 > 0：**{r['ci95_all_positive']}**",
            f"- 自助分布中 ≤0 的比例：{r['p_boot_le_zero']:.4f}",
            "",
        ]
        # 预注册判据（第一轮 99%、第二轮 95%；均要求 ≥3/4 折同向）
        gate = r["n_folds_positive"] >= max(3, int(np.ceil(0.75 * r["n_folds"])))
        verdict1 = r["ci99_all_positive"] and gate
        verdict2 = r["ci95_all_positive"] and gate
        lines += [
            f"**判据结论**：第一轮（99% + ≥3/4 折同向）= "
            f"{'胜出' if verdict1 else '不可分'}；"
            f"第二轮（95% + ≥3/4 折同向）= {'胜出' if verdict2 else '不可分'}。", "",
        ]
        r["verdict_round1"] = verdict1
        r["verdict_round2"] = verdict2

    text = "\n".join(lines)
    print("\n" + text)
    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(text, encoding="utf-8")
        outp.with_suffix(".json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False, default=float),
            encoding="utf-8")
        print(f"\n写入 {outp}")
    if args.ledger:
        for name, r in results.items():
            append_ledger(
                f"ablation-test | {name}−{base} folds={r['n_folds']} "
                f"days={r['n_days_total']} mean_diff={r['mean_diff']:+.5f} "
                f"folds_pos={r['n_folds_positive']}/{r['n_folds']} "
                f"CI99=[{r['ci99'][0]:+.5f},{r['ci99'][1]:+.5f}] "
                f"round1={'WIN' if r['verdict_round1'] else 'TIE'} "
                f"round2={'WIN' if r['verdict_round2'] else 'TIE'}"
            )


if __name__ == "__main__":
    main()
