"""E4：评估指标的「贴合度–功效」图（探索性诊断；预注册用途限制见下）。

问题：全截面 RankIC 每天用数百只股票算、测得准，但策略只吃分数前 10%；
「前 50 名收益」贴合钱但每天只有 50 个观测、很吵。哪个指标既跟钱走得近、又测得准？
（背景：Kharitonov 等 2017 WSDM「在方向一致约束下最大化敏感度」；Sakai 的辨别力
一脉；本项目实测 IC 与钱相关 0.87、桥公式 spread ≈ 3.51×IC×σ_cs、η=0.90。）

**用途限制（先于运行写死）**：
* 本图只用于 (a) 选择实盘监控用哪个指标、(b) 论文方法节。
* **不得**用于在 Kronos 配置 / 构造参数 / 合成规则之间做选择（那是 B 层）。
* 只读已消耗的开发折 36–42（FT 臂），经白名单与 `assert_readable`。
* 无通过/不通过判据；只报每个指标的两个量与其自助区间。

定义（逐日 t，名字集 = 当日有分数且有 label 的 top500-by-ADV 名字）：
  候选指标 M_t：
    ic_full        全打分池 Spearman(score, label)
    ic_top500      top500 内 Spearman(score, label)
    ic_top30       按分数前 30% 的名字内部 Spearman(score, label)（退出线所依赖的排序）
    decile_spread  前 10% 名字 label 均值 − 池均值（bp）
    ic_x_sigma_expost   ic_top500 × 当日 label 的截面标准差（ex post，仅诊断；含未来信息，
                        不可作实盘特征——CLAUDE.md §一.1）
    ic_x_sigma_trail    ic_top500 × 过去 21 日 label 截面标准差的均值（滞后 6 日以避免前视，
                        即用 t−27..t−7 的 label；可实盘计算）
    prec50         前 50 名中 label > 当日池中位数 的比例 − 0.5
  真效用 U_t：NT=5 冻结构造（cost=0）的日毛超额 r_t；对齐用
    U_fwd_t = Σ_{k=1..6} r_{t+k}（t 日选中的名字在 t+1..t+6 持有）
  贴合度：corr(M_t, U_fwd_t)、sign agreement = P(sign M_t == sign U_fwd_t)
          （两者的 stationary bootstrap 95% CI，块长 10）
  功效：  t_M = mean(M_t)/NW5-SE，以及 t_M/√(n/252)（每年化 t，便于跨样本长度比较）
输出：outputs/e4_metric_alignment_power.json（+ 若有 matplotlib 则输出 PNG）。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from crsp_pipeline.sealed import assert_readable          # noqa: E402
from crsp_pipeline.signal_eval import newey_west_tstat    # noqa: E402
from portfolio.construction import (                      # noqa: E402
    frozen_long_only_returns, scores_frame_to_by_day,
)
from signals.kronos_adapter import labels_path, scores_path  # noqa: E402

NT, TOPN, EXIT_PCT, MIN_NAMES = 5, 500, 0.30, 50
FOLDS = list(range(36, 43))
ARM = "ft"
NW_LAG = 5
BOOT_B, BOOT_BLOCK, BOOT_SEED = 5000, 10, 20260903
OUT_JSON = REPO / "outputs" / "e4_metric_alignment_power.json"
OUT_PNG = REPO / "outputs" / "e4_metric_alignment_power.png"


def log(m: str) -> None:
    print(m, flush=True)


def _load_cam():
    spec = importlib.util.spec_from_file_location(
        "compare_arms_money", REPO / "scripts" / "compare_arms_money.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 10:
        return np.nan
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / d) if d > 0 else np.nan


def daily_metrics(sc: pd.DataFrame, lb: pd.DataFrame, adv: dict) -> pd.DataFrame:
    """逐日指标。sc: [PERMNO, signal_date, score]; lb: [PERMNO, signal_date, label, status]."""
    df = sc.merge(lb[lb["status"] == "ok"][["PERMNO", "signal_date", "label"]],
                  on=["PERMNO", "signal_date"], how="inner").dropna()
    rows = []
    for day, g in df.groupby("signal_date", sort=True):
        s_all, y_all = g["score"].to_numpy(), g["label"].to_numpy()
        ic_full = _spearman(s_all, y_all)
        a = adv.get(day, {})
        m = g[g["PERMNO"].map(lambda p: p in a and np.isfinite(a[p]))].copy()
        if len(m) > TOPN:
            m["_adv"] = m["PERMNO"].map(a)
            m = m.nlargest(TOPN, "_adv")
        if len(m) < MIN_NAMES:
            continue
        s, y = m["score"].to_numpy(), m["label"].to_numpy()
        ic_top = _spearman(s, y)
        order = np.argsort(-s)
        n = len(s)
        k10, k30 = max(1, n // 10), max(10, int(round(0.30 * n)))
        top10, top30 = order[:k10], order[:k30]
        decile_spread_bp = float((y[top10].mean() - y.mean()) * 1e4)
        ic_top30 = _spearman(s[top30], y[top30])
        med = np.median(y)
        prec50 = float((y[order[:50]] > med).mean() - 0.5)
        sigma_cs = float(y.std(ddof=1))
        rows.append((day, ic_full, ic_top, ic_top30, decile_spread_bp, prec50, sigma_cs))
    out = pd.DataFrame(rows, columns=["date", "ic_full", "ic_top500", "ic_top30",
                                      "decile_spread", "prec50", "sigma_cs"]).set_index("date")
    out["ic_x_sigma_expost"] = out["ic_top500"] * out["sigma_cs"]
    # 滞后 6 日再取 21 日均值：t−27..t−7 的 σ_cs（label 6 日期，t−7 的 label 到 t−1 已实现）
    trail = out["sigma_cs"].shift(7).rolling(21, min_periods=15).mean()
    out["ic_x_sigma_trail"] = out["ic_top500"] * trail
    return out


def stationary_bootstrap_idx(n: int, rng: np.random.Generator, block: int) -> np.ndarray:
    idx = np.empty(n, dtype=int)
    i = 0
    while i < n:
        start = rng.integers(0, n)
        L = rng.geometric(1.0 / block)
        for j in range(L):
            if i >= n:
                break
            idx[i] = (start + j) % n
            i += 1
    return idx


def alignment(m: np.ndarray, u: np.ndarray, rng: np.random.Generator) -> dict:
    ok = np.isfinite(m) & np.isfinite(u)
    m, u = m[ok], u[ok]
    n = len(m)
    corr = float(np.corrcoef(m, u)[0, 1])
    sign = float((np.sign(m) == np.sign(u)).mean())
    cs, ss = [], []
    for _ in range(BOOT_B):
        idx = stationary_bootstrap_idx(n, rng, BOOT_BLOCK)
        mm, uu = m[idx], u[idx]
        cs.append(np.corrcoef(mm, uu)[0, 1])
        ss.append((np.sign(mm) == np.sign(uu)).mean())
    return {"n": n, "corr": corr, "corr_ci95": [float(np.percentile(cs, 2.5)), float(np.percentile(cs, 97.5))],
            "sign_agree": sign, "sign_ci95": [float(np.percentile(ss, 2.5)), float(np.percentile(ss, 97.5))]}


def main() -> None:
    cam = _load_cam()
    assert_readable(cam.P)
    log("加载价格/成交额...")
    ret, oc, adv = cam.load_prices()

    metrics_parts, util_parts = [], []
    for f in FOLDS:
        sp, lp = scores_path(f, ARM), labels_path(f, ARM)
        assert_readable(sp); assert_readable(lp)
        sc = pd.read_parquet(sp, columns=["PERMNO", "signal_date", "score"]).dropna()
        sc["signal_date"] = pd.to_datetime(sc["signal_date"])
        lb = pd.read_parquet(lp, columns=["PERMNO", "signal_date", "label", "status"])
        lb["signal_date"] = pd.to_datetime(lb["signal_date"])
        metrics_parts.append(daily_metrics(sc, lb, adv))
        by_day = scores_frame_to_by_day(sc, min_names=MIN_NAMES)
        u = frozen_long_only_returns(by_day, ret, oc, adv, topn=TOPN, cost_bp=0.0,
                                     exit_pct=EXIT_PCT, nt=NT, min_names=MIN_NAMES)
        u["fold"] = f
        util_parts.append(u[["r", "fold"]])
        del sc, lb, by_day
        log(f"  fold{f} 完成")
    M = pd.concat(metrics_parts).sort_index()
    U = pd.concat(util_parts).sort_index()
    # U_fwd：t 之后 6 个交易日（折内）的构造毛超额之和
    U["u_fwd"] = U.groupby("fold")["r"].transform(
        lambda s: s.shift(-1).rolling(6, min_periods=6).sum().shift(-5))
    joined = M.join(U[["u_fwd", "r"]], how="inner")

    rng = np.random.default_rng(BOOT_SEED)
    cand = ["ic_full", "ic_top500", "ic_top30", "decile_spread", "prec50",
            "ic_x_sigma_expost", "ic_x_sigma_trail"]
    years = len(joined) / 252.0
    res = {"meta": {"run_utc": datetime.now(timezone.utc).isoformat(), "arm": ARM,
                    "folds": FOLDS, "n_days": int(len(joined)), "years": years,
                    "construction": {"NT": NT, "TOPN": TOPN, "EXIT_PCT": EXIT_PCT},
                    "use_restriction": "monitoring-metric choice & paper only; NOT for model/construction/combination selection"},
           "metrics": {}}
    for c in cand:
        x = joined[c].to_numpy(dtype=float)
        nw = newey_west_tstat(pd.Series(x), NW_LAG)
        al = alignment(x, joined["u_fwd"].to_numpy(dtype=float), rng)
        al_same = alignment(x, joined["r"].to_numpy(dtype=float), rng)
        res["metrics"][c] = {
            "power": {"mean": float(nw["mean"]), "nw_se": float(nw["se"]), "t": float(nw["t"]),
                      "t_per_sqrt_year": float(nw["t"] / np.sqrt(years))},
            "alignment_fwd6": al,
            "alignment_sameday": {"corr": al_same["corr"], "sign_agree": al_same["sign_agree"]},
        }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    log(f"\n=== E4 贴合度–功效（折 36–42，{len(joined)} 天，U = NT=5 构造毛超额的前向 6 日和）===")
    log(f"{'指标':<20}{'corr(M,U_fwd)':>16}{'符号一致':>10}{'t':>8}{'t/√年':>8}")
    for c in cand:
        r = res["metrics"][c]
        log(f"{c:<20}{r['alignment_fwd6']['corr']:>16.3f}{r['alignment_fwd6']['sign_agree']:>10.3f}"
            f"{r['power']['t']:>8.2f}{r['power']['t_per_sqrt_year']:>8.2f}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        for c in cand:
            r = res["metrics"][c]
            x, y = r["alignment_fwd6"]["corr"], r["power"]["t_per_sqrt_year"]
            ax.scatter(x, y)
            ax.annotate(c, (x, y), fontsize=8, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("alignment: corr(metric_t, construction excess over next 6 days)")
        ax.set_ylabel("power: NW t per sqrt(year)")
        ax.set_title("E4 metric alignment vs power (dev folds 36-42, FT, NT=5)")
        ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(OUT_PNG, dpi=130)
        log(f"图已写 {OUT_PNG}")
    except Exception as e:  # noqa: BLE001
        log(f"(未画图: {e})")
    log(f"已写 {OUT_JSON}")


if __name__ == "__main__":
    main()
