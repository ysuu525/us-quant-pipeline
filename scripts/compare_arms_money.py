"""按**净收益**而非 RankIC 比较臂，并给出显著性。

动机（2026-08-31 发现）：RankIC 与实际赚到的钱只有 corr≈0.87 的松散关系，
逐折可差 2.7 倍（fold40 IC 更高却只赚 fold37 的 1/2.7；fold41 IC 为正但毛价差
为零）。原因是 **收益 ≈ IC × 当日截面离散度**，而 IC 不含离散度这个乘数；
且 IC 看全截面、组合只吃两端。本项目此前的关键选择（训练池、lookback、
合成取舍）**全部按 IC 做出**，与真正的目标函数不一致，须按钱重算一遍。

显著性同样必须给：一个更高的净收益若落在噪声范围内，就不构成选择理由。
判据沿用预注册 §2 的配对块自助（块长 10 交易日、折内抽样、10000 次），
只是把被检验的量从"逐日 RankIC 差"换成"逐日组合净收益差"。

口径：七折 36–42 / top500 流动性过滤 / 进入前10%、跌出前30%才卖 /
六档错位 / t+1 开盘建仓 / 8bp 单边 / 纯多超额（相对同池等权基准）。
限定：折 36–42 已被反复查看，本读数为**方向性证据**，不是无偏估计。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from crsp_pipeline.signal_eval import newey_west_tstat  # noqa: E402

P = Path(r"F:\quant\processed\crsp_ciz_2026-08-24_20260825T130601Z")
OUT = REPO_ROOT / "outputs"
FOLDS = ["fold36", "fold37", "fold38", "fold39", "fold40", "fold41", "fold42"]
LO, HI = "2020-06-01", "2024-01-05"
EXIT_PCT, NT = 0.30, 6
BLOCK_LEN, N_BOOT, BOOT_SEED = 10, 10000, 20260827


def log(m):
    print(m, flush=True)


def load_prices():
    df = pd.read_parquet(
        P / "panel_raw.parquet",
        columns=["PERMNO", "DlyCalDt", "DlyOpen", "DlyClose", "DlyRet", "DlyPrcVol"],
        filters=[("DlyCalDt", ">=", pd.Timestamp(LO)), ("DlyCalDt", "<=", pd.Timestamp(HI))])
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


def arm_returns(lb, ret, oc, adv, topn, cost_bp):
    """返回 DataFrame(index=date, columns=[long, fold])。"""
    rows = []
    for fold in FOLDS:
        d = OUT / f"{fold}_lb{lb}_s0_poolB_universe" / f"eval_amp_lb{lb}_{fold}"
        sc = pd.read_parquet(d / "scores.parquet",
                             columns=["PERMNO", "signal_date", "score"]).dropna()
        sc["signal_date"] = pd.to_datetime(sc["signal_date"])
        by_day = {day: dict(zip(g["PERMNO"], g["score"]))
                  for day, g in sc.groupby("signal_date") if len(g) >= 50}
        del sc
        days = sorted(by_day)
        book = [None] * NT
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
                nb, fresh = list(order[:k]), set(order[:k])
                turn = 0.0
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
            rows.append((nd, float(np.mean(vals)) - bench - cost, fold))
    return pd.DataFrame(rows, columns=["date", "r", "fold"]).set_index("date").sort_index()


def block_bootstrap_ci(diff_by_fold, rng):
    """折内块自助（块长 10），与预注册 §2 的 IC 判据同法，只是量换成净收益差。"""
    draws = np.empty(N_BOOT)
    arrs = [v.to_numpy() for v in diff_by_fold.values() if len(v) >= BLOCK_LEN]
    for b in range(N_BOOT):
        acc = []
        for a in arrs:
            nblk = max(1, len(a) // BLOCK_LEN)
            starts = rng.integers(0, len(a) - BLOCK_LEN + 1, nblk)
            acc.append(np.concatenate([a[s:s + BLOCK_LEN] for s in starts]))
        draws[b] = np.concatenate(acc).mean()
    return draws


def stats(r):
    eq = (1 + r).cumprod()
    n = len(r)
    ann = eq.iloc[-1] ** (252 / n) - 1
    vol = r.std() * np.sqrt(252)
    dd = eq / eq.cummax() - 1
    return ann, vol, (ann / vol if vol else np.nan), dd.min()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topn", type=int, default=500)
    ap.add_argument("--cost-bp", type=float, default=8.0)
    ap.add_argument("--out-json", default=str(OUT / "compare_arms_money.json"))
    args = ap.parse_args()

    log("读取价格面板...")
    ret, oc, adv = load_prices()
    series = {}
    for lb in (90, 200):
        log(f"构造 lb{lb} 的日频净收益（top{args.topn} / {args.cost_bp}bp）...")
        series[lb] = arm_returns(lb, ret, oc, adv, args.topn, args.cost_bp)

    log(f"\n=== 按净收益比较（七折 / top{args.topn} / {args.cost_bp}bp / 纯多超额）===")
    res = {}
    for lb in (90, 200):
        s = series[lb]
        ann, vol, sh, mdd = stats(s["r"])
        nw = newey_west_tstat(s["r"], 5)
        res[f"lb{lb}"] = {"ann": float(ann), "vol": float(vol), "sharpe": float(sh),
                          "maxdd": float(mdd), "t_daily": float(nw["t"]),
                          "n_days": int(len(s))}
        log(f"  lb{lb}: 年化 {ann*100:+6.2f}%  波动 {vol*100:5.2f}%  夏普 {sh:5.2f}  "
            f"最大回撤 {mdd*100:7.2f}%  日收益 NW t {nw['t']:+.2f}  ({len(s)} 天)")

    # 配对：只在两臂都有的交易日上比
    a, b = series[90], series[200]
    j = a.join(b["r"].rename("r200"), how="inner")
    j["diff"] = j["r"] - j["r200"]
    log(f"\n  配对日数 {len(j)}（lb90 minus lb200）")
    nwd = newey_west_tstat(j["diff"], 5)
    ann_diff = (1 + j["r"]).prod() ** (252 / len(j)) - (1 + j["r200"]).prod() ** (252 / len(j))
    by_fold = {f: g["diff"] for f, g in j.groupby("fold")}
    pos = sum(1 for v in by_fold.values() if v.mean() > 0)
    rng = np.random.default_rng(BOOT_SEED)
    draws = block_bootstrap_ci(by_fold, rng)
    ci99 = (np.quantile(draws, 0.005), np.quantile(draws, 0.995))
    ci95 = (np.quantile(draws, 0.025), np.quantile(draws, 0.975))
    log(f"  年化差 {ann_diff*100:+.2f}pp   逐日差均值 {nwd['mean']*1e4:+.3f}bp/日  "
        f"NW t {nwd['t']:+.2f}")
    log(f"  折内 lb90 更优: {pos}/{len(by_fold)}")
    log(f"  块自助 99% CI [{ci99[0]*1e4:+.3f}, {ci99[1]*1e4:+.3f}] bp/日 → "
        f"全体>0: {ci99[0] > 0}")
    log(f"  块自助 95% CI [{ci95[0]*1e4:+.3f}, {ci95[1]*1e4:+.3f}] bp/日 → "
        f"全体>0: {ci95[0] > 0}")
    log(f"  自助分布中 ≤0 的比例 {float((draws <= 0).mean()):.4f}")
    gate = pos >= max(3, int(np.ceil(0.75 * len(by_fold))))
    log(f"\n  **判据（同预注册 §2）**：第一轮 99% + ≥3/4 折同向 = "
        f"{'胜出' if (ci99[0] > 0 and gate) else '不可分'}；"
        f"第二轮 95% = {'胜出' if (ci95[0] > 0 and gate) else '不可分'}")

    res["paired"] = {"n_days": int(len(j)), "ann_diff_pp": float(ann_diff * 100),
                     "mean_diff_bp": float(nwd["mean"] * 1e4), "t": float(nwd["t"]),
                     "folds_lb90_better": pos, "n_folds": len(by_fold),
                     "ci99_bp": [float(ci99[0] * 1e4), float(ci99[1] * 1e4)],
                     "ci95_bp": [float(ci95[0] * 1e4), float(ci95[1] * 1e4)],
                     "p_boot_le_zero": float((draws <= 0).mean())}
    Path(args.out_json).write_text(json.dumps(res, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
    log(f"\n写入 {args.out_json}")
    log("限定：折 36–42 已被反复查看，本读数为方向性证据，非无偏估计。")


if __name__ == "__main__":
    main()
