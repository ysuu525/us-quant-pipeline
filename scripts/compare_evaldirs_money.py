"""按净收益比较任意两臂（通用版，输入是 evaluate_fold 的产物目录）。

与 compare_arms.py（预注册 §2，比较逐日 RankIC）的关系：
**同一套配对块自助判据，只是被检验的量从"逐日 RankIC 差"换成"逐日组合净收益差"。**
动机见 ledger 2026-08-31：RankIC 与实际收益只有 corr≈0.87 的松散关系，
逐折可差 2.7 倍；本项目的训练池、lookback、合成取舍此前全部按 IC 做出。

组合构造在此**冻结**（不得随臂调整，否则是拿同一批数据同时优化信号与构造）：
top-N 流动性过滤 / 十分位 / 六档错位 / 进入前10%、跌出前30%才卖 /
t+1 开盘建仓 / 单边成本 c / 纯多超额相对同池等权基准。

用法：
  python scripts/compare_evaldirs_money.py \
      --arm A=outputs/fold01_lb90_s0/eval_poolA_full \
      --arm A=outputs/fold02_lb90_s0/eval_poolA_full_fold02 \
      --arm B=outputs/fold01_lb90_s0_poolB_universe/eval_poolB_universe \
      --baseline B
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from crsp_pipeline.signal_eval import newey_west_tstat  # noqa: E402

P = Path(r"F:\quant\processed\crsp_ciz_2026-08-24_20260825T130601Z")
EXIT_PCT, NT = 0.30, 6
BLOCK_LEN, N_BOOT, BOOT_SEED = 10, 10000, 20260827


def log(m):
    print(m, flush=True)


def load_prices(lo, hi):
    df = pd.read_parquet(
        P / "panel_raw.parquet",
        columns=["PERMNO", "DlyCalDt", "DlyOpen", "DlyClose", "DlyRet", "DlyPrcVol"],
        filters=[("DlyCalDt", ">=", pd.Timestamp(lo)), ("DlyCalDt", "<=", pd.Timestamp(hi))])
    df["DlyCalDt"] = pd.to_datetime(df["DlyCalDt"])
    df = df.dropna(subset=["DlyRet"]).sort_values(["PERMNO", "DlyCalDt"])
    df["oc"] = np.where(df["DlyOpen"].abs() > 0,
                        df["DlyClose"] / df["DlyOpen"].abs() - 1.0, np.nan)
    df["adv20"] = df.groupby("PERMNO")["DlyPrcVol"].transform(
        lambda s: s.rolling(20, min_periods=10).mean().shift(1))
    out = ({d: dict(zip(g["PERMNO"], g["DlyRet"])) for d, g in df.groupby("DlyCalDt")},
           {d: dict(zip(g["PERMNO"], g["oc"])) for d, g in df.groupby("DlyCalDt")},
           {d: dict(zip(g["PERMNO"], g["adv20"])) for d, g in df.groupby("DlyCalDt")})
    del df
    return out


def fold_returns(eval_dir: Path, ret, oc, adv, topn, cost_bp):
    sc = pd.read_parquet(eval_dir / "scores.parquet",
                         columns=["PERMNO", "signal_date", "score"]).dropna()
    sc["signal_date"] = pd.to_datetime(sc["signal_date"])
    by_day = {day: dict(zip(g["PERMNO"], g["score"]))
              for day, g in sc.groupby("signal_date") if len(g) >= 50}
    del sc
    days = sorted(by_day)
    book = [None] * NT
    rows = []
    for i, day in enumerate(days):
        s, a = by_day[day], adv.get(day, {})
        elig = [p for p in s if p in a and np.isfinite(a[p])]
        if topn and len(elig) > topn:
            elig = sorted(elig, key=lambda p: -a[p])[:topn]
        elig = set(elig)
        s = {p: v for p, v in s.items() if p in elig}
        n = len(s)
        if n < 50:
            continue
        pct = (pd.Series(s).rank() / n).to_dict()
        k = max(1, n // 10)
        order = sorted(pct, key=lambda p: -pct[p])
        j = i % NT
        prev = book[j]
        if prev is None:
            nb, fresh, turn = list(order[:k]), set(order[:k]), 0.0
        else:
            keep = [p for p in prev if p in pct and pct[p] >= 1 - EXIT_PCT][:k]
            held = set(keep)
            add = [p for p in order if p not in held][:k - len(keep)]
            nb, fresh = keep + add, set(add)
            turn = (k - len(keep)) / k
        book[j] = nb
        if i + 1 >= len(days) or i < NT:
            continue
        nd = days[i + 1]
        rm, om = ret.get(nd, {}), oc.get(nd, {})
        if not rm:
            continue
        cost = 2.0 * (cost_bp / 1e4) * turn / NT
        vals = []
        for t in range(NT):
            nm = book[t]
            if not nm:
                continue
            rs = [(om.get(p) if (t == j and p in fresh) else rm.get(p)) for p in nm]
            rs = [v for v in rs if v is not None and np.isfinite(v)]
            if rs:
                vals.append(np.mean(rs))
        if not vals:
            continue
        bench = float(np.mean([rm[p] for p in pct if p in rm]))
        rows.append((nd, float(np.mean(vals)) - bench - cost))
    return pd.Series(dict(rows)).sort_index()


def stats(r):
    eq = (1 + r).cumprod()
    n = len(r)
    ann = eq.iloc[-1] ** (252 / n) - 1
    vol = r.std() * np.sqrt(252)
    dd = eq / eq.cummax() - 1
    return float(ann), float(vol), float(ann / vol) if vol else np.nan, float(dd.min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="NAME=EVAL_DIR")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--topn", type=int, default=500)
    ap.add_argument("--cost-bp", type=float, default=8.0)
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    arms = defaultdict(dict)
    for spec in args.arm:
        name, path = spec.split("=", 1)
        d = Path(path)
        sc = pd.read_parquet(d / "scores.parquet", columns=["signal_date"])
        key = (f"{pd.to_datetime(sc['signal_date']).min():%Y-%m-%d}"
               f"..{pd.to_datetime(sc['signal_date']).max():%Y-%m-%d}")
        arms[name][key] = d
        del sc
    lo = min(k.split("..")[0] for a in arms.values() for k in a)
    hi = max(k.split("..")[1] for a in arms.values() for k in a)
    log(f"载入 {len(arms)} 臂；折窗 {lo} .. {hi}")

    log("读取价格面板（列裁剪 + 日期下推）...")
    ret, oc, adv = load_prices(pd.Timestamp(lo) - pd.Timedelta(days=60),
                               pd.Timestamp(hi) + pd.Timedelta(days=20))

    series = {}
    for name, folds in arms.items():
        parts = {}
        for key, d in folds.items():
            parts[key] = fold_returns(d, ret, oc, adv, args.topn, args.cost_bp)
        series[name] = parts
        allr = pd.concat(parts.values()).sort_index()
        ann, vol, sh, mdd = stats(allr)
        nw = newey_west_tstat(allr, 5)
        log(f"  臂 {name}: {len(parts)} 折 / {len(allr)} 日  年化 {ann*100:+6.2f}%  "
            f"波动 {vol*100:5.2f}%  夏普 {sh:5.2f}  回撤 {mdd*100:7.2f}%  NW t {nw['t']:+.2f}")

    base = args.baseline
    res = {}
    rng = np.random.default_rng(BOOT_SEED)
    for name in series:
        if name == base:
            continue
        log(f"\n## {name} minus {base}（净收益口径，top{args.topn} / {args.cost_bp}bp）")
        diffs = {}
        for key in series[base]:
            if key not in series[name]:
                continue
            a, b = series[name][key], series[base][key]
            idx = a.index.intersection(b.index)
            diffs[key] = (a.loc[idx] - b.loc[idx])
        all_d = pd.concat(diffs.values()).sort_index()
        nw = newey_west_tstat(all_d, 5)
        pos = sum(1 for v in diffs.values() if v.mean() > 0)
        draws = np.empty(N_BOOT)
        arrs = [v.to_numpy() for v in diffs.values() if len(v) >= BLOCK_LEN]
        for i in range(N_BOOT):
            acc = []
            for arr in arrs:
                nblk = max(1, len(arr) // BLOCK_LEN)
                st = rng.integers(0, len(arr) - BLOCK_LEN + 1, nblk)
                acc.append(np.concatenate([arr[s:s + BLOCK_LEN] for s in st]))
            draws[i] = np.concatenate(acc).mean()
        ci99 = (float(np.quantile(draws, 0.005)), float(np.quantile(draws, 0.995)))
        ci95 = (float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)))
        gate = pos >= max(3, int(np.ceil(0.75 * len(diffs))))
        log(f"  配对日数 {len(all_d)}（{len(diffs)} 折）；逐日差均值 "
            f"{nw['mean']*1e4:+.3f}bp/日  NW t {nw['t']:+.2f}")
        log(f"  折内为正 {pos}/{len(diffs)}（判据要求 "
            f"≥{max(3, int(np.ceil(0.75*len(diffs))))}）")
        log(f"  99% CI [{ci99[0]*1e4:+.3f}, {ci99[1]*1e4:+.3f}] bp/日 → 全体>0 {ci99[0]>0}")
        log(f"  95% CI [{ci95[0]*1e4:+.3f}, {ci95[1]*1e4:+.3f}] bp/日 → 全体>0 {ci95[0]>0}")
        log(f"  自助分布中 ≤0 的比例 {float((draws<=0).mean()):.4f}")
        log(f"  **判据**：第一轮 99%+同向 = {'胜出' if (ci99[0]>0 and gate) else '不可分'}；"
            f"第二轮 95%+同向 = {'胜出' if (ci95[0]>0 and gate) else '不可分'}")
        res[name] = {"mean_diff_bp": float(nw["mean"] * 1e4), "t": float(nw["t"]),
                     "folds_pos": pos, "n_folds": len(diffs),
                     "ci99_bp": [c * 1e4 for c in ci99], "ci95_bp": [c * 1e4 for c in ci95],
                     "p_boot_le_zero": float((draws <= 0).mean()),
                     "round1": bool(ci99[0] > 0 and gate), "round2": bool(ci95[0] > 0 and gate)}
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(res, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
        log(f"\n写入 {args.out_json}")


if __name__ == "__main__":
    main()
