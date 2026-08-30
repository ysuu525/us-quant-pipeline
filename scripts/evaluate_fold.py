"""单折验证窗评估驱动（§7.5 信号层子集；池子/lookback 消融的读数来源）。

    python scripts/evaluate_fold.py --model-dir outputs/fold01_lb90_s0 \
        --processed F:/quant/processed/<快照ID> \
        --val-start 2003-01-02 --val-end 2003-06-30 --lookback 90 \
        --tag poolA_full

产出（--model-dir/eval_<tag>/）：scores.parquet（逐日全 universe 打分）、
labels.parquet（§4 标签 + 状态）、metrics.json、report.md。
每次运行自动在 experiments/ledger.md 记一笔验证窗查看（预注册 §3）。

口径（与预注册一致）：
- 打分对象 = anchor 日在 §2 universe 内、lookback 侧连续有效的股票；
- 输入蜡烛 = §5 复权后面板（与训练分布一致）；标签 = §4 execution-return
  （未复权全量面板计算）；
- 主指标 = 逐日横截面 RankIC 均值 + Newey-West t（lags=5）；
- 分层诊断 = 市值（DlyCap）与 ADV20 五分位内的 RankIC；
- 退出日现金股息 = distributions 中 disdivamt>0 的行（过夜持有人实得现金，
  2026-08-27 冻结）；业绩类退市码未冻结 → 不插补（DlyRet 缺失的退市记
  INVALID，各臂同口径，比较不受影响）。

内存注意：面板按需切片（评估窗前 lookback+缓冲 起）。退市原因细分
（delisted vs halted）在切片下可能把窗前已退市误报为 halted——只影响
unfillable 原因统计，不影响标签值。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from crsp_pipeline.calendar import TradingCalendar  # noqa: E402
from crsp_pipeline.labels import compute_labels  # noqa: E402
from crsp_pipeline.signal_eval import (  # noqa: E402
    daily_rank_ic,
    decile_spread,
    newey_west_tstat,
    winsorized_rank_ic,
)
from kronos_ft.infer import run_scoring  # noqa: E402
from kronos_ft.models import load_pretrained  # noqa: E402
from kronos_ft.train import append_ledger  # noqa: E402
from kronos_ft.windows import build_scoring_index, filter_index_by_universe  # noqa: E402


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def stratified_ic(df: pd.DataFrame, strat_col: str, n_groups: int = 5,
                  nw_lags: int = 5) -> dict:
    """按 strat_col 逐日五分位分层，各层内逐日 RankIC → NW 统计。"""
    d = df.dropna(subset=[strat_col, "score", "label"]).copy()
    d["_q"] = d.groupby("signal_date")[strat_col].transform(
        lambda s: pd.qcut(s.rank(method="first"), n_groups, labels=False)
        if len(s) >= n_groups else pd.Series(np.nan, index=s.index)
    )
    out = {}
    for q in range(n_groups):
        sub = d[d["_q"] == q]
        ic = daily_rank_ic(sub) if len(sub) else pd.Series(dtype=float)
        out[f"q{q + 1}"] = newey_west_tstat(ic, nw_lags)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="单折验证窗评估（打分 + RankIC）")
    ap.add_argument("--model-dir", required=True,
                    help="含 tokenizer_final / predictor_final 的训练输出目录")
    ap.add_argument("--processed", required=True, help="prepare_data.py 的输出目录")
    ap.add_argument("--val-start", required=True)
    ap.add_argument("--val-end", required=True)
    ap.add_argument("--lookback", type=int, required=True)
    ap.add_argument("--tag", required=True, help="登记簿与输出目录标识，如 poolA_full")
    ap.add_argument("--predict", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--sample-count", type=int, default=5)
    ap.add_argument("--amp", choices=["bf16", "fp16"], default=None,
                    help="GPU 侧混合精度（默认关=fp32）。开启会改变数值 → "
                         "与 fp32 口径的读数不可直接比较，须整批同口径")
    ap.add_argument("--device", default=None, help="cpu / cuda；默认自动")
    ap.add_argument("--limit-obs", type=int, default=0,
                    help="只打前 N 个观测（冒烟验证代码路径用；正式评估勿用）")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    P = Path(args.processed)
    out = model_dir / f"eval_{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    val_start, val_end = pd.Timestamp(args.val_start), pd.Timestamp(args.val_end)

    idxdf = pd.read_parquet(P / "market_index.parquet")
    cal = TradingCalendar.from_market_index(idxdf, "caldt")
    # 面板切片：评估窗前留 lookback+30 个交易日（打分窗口 + ADV20 计算）
    lo = cal.shift(cal.snap_forward(val_start), -(args.lookback + 30))
    end_idx = min(cal.index_of(cal.snap_back(val_end)) + args.predict + 2, len(cal) - 1)
    hi = cal.dates[end_idx]

    log(f"评估窗 [{val_start.date()} .. {val_end.date()}]  切片 [{lo.date()} .. {hi.date()}]")
    adj = pd.read_parquet(P / "panel_kronos_adj.parquet")
    adj = adj[(adj["DlyCalDt"] >= lo) & (adj["DlyCalDt"] <= hi)]
    uni = pd.read_parquet(P / "universe.parquet",
                          columns=["PERMNO", "DlyCalDt", "in_universe"])

    log("构建打分索引（lookback 侧连续有效 + anchor 在 universe 内）...")
    sidx = build_scoring_index(adj, cal, args.lookback)
    sidx = sidx[(sidx["anchor"] >= val_start) & (sidx["anchor"] <= val_end)]
    n0 = len(sidx)
    sidx = filter_index_by_universe(sidx, uni)
    log(f"打分观测: {n0:,} -> universe 内 {len(sidx):,}")
    if args.limit_obs:
        sidx = sidx.head(args.limit_obs)
        log(f"--limit-obs {args.limit_obs}（冒烟模式，结果不作判定用）")

    log("加载微调模型...")
    tokenizer, model = load_pretrained(str(model_dir / "tokenizer_final"),
                                       str(model_dir / "predictor_final"))
    log("推理打分（多路径均值，官方采样参数）...")
    need_pn = set(sidx["PERMNO"].unique())
    adj_scoring = adj[adj["PERMNO"].isin(need_pn)]
    scores = run_scoring(tokenizer, model, adj_scoring, sidx, cal,
                         args.lookback, predict=args.predict,
                         batch_size=args.batch_size, device=args.device,
                         amp=args.amp, sample_count=args.sample_count)
    scores.to_parquet(out / "scores.parquet", index=False)
    log(f"scores: {len(scores):,} 行, NaN {scores['score'].isna().mean():.4f}, "
        f"外推 {scores['extrapolated'].mean():.4f}")

    log("§4 标签（未复权全量面板）...")
    raw = pd.read_parquet(
        P / "panel_raw.parquet",
        columns=["PERMNO", "DlyCalDt", "DlyOpen", "DlyClose", "DlyRet",
                 "DlyDelFlg", "DlyCap", "DlyPrcVol"])
    raw = raw[(raw["DlyCalDt"] >= lo) & (raw["DlyCalDt"] <= hi)]
    dist = pd.read_parquet(P / "distributions.parquet",
                           columns=["permno", "disexdt", "disdivamt"])
    cash = dist[dist["disdivamt"].fillna(0) > 0].rename(
        columns={"permno": "PERMNO", "disexdt": "ex_date", "disdivamt": "div_cash"})
    raw_needed = raw[raw["PERMNO"].isin(need_pn)]
    labels = compute_labels(raw_needed, scores, cal, cash_dividends=cash,
                            predict_window=args.predict)
    labels.to_parquet(out / "labels.parquet", index=False)
    ok_share = float((labels["status"] == "ok").mean())
    log(f"labels: ok 占比 {ok_share:.4f}")

    df = scores.merge(labels[["PERMNO", "signal_date", "status", "label"]],
                      on=["PERMNO", "signal_date"])
    df = df[(df["status"] == "ok") & df["score"].notna()]
    # 分层变量：t 日市值；ADV20 = t 日止 20 日 DlyPrcVol 均值
    cap = raw_needed[["PERMNO", "DlyCalDt", "DlyCap"]].rename(
        columns={"DlyCalDt": "signal_date"})
    df = df.merge(cap, on=["PERMNO", "signal_date"], how="left")
    adv = raw_needed.sort_values(["PERMNO", "DlyCalDt"]).copy()
    adv["ADV20"] = adv.groupby("PERMNO")["DlyPrcVol"].transform(
        lambda s: s.rolling(20, min_periods=15).mean())
    df = df.merge(adv[["PERMNO", "DlyCalDt", "ADV20"]].rename(
        columns={"DlyCalDt": "signal_date"}), on=["PERMNO", "signal_date"], how="left")

    log("指标（RankIC / NW / 十分位 / 分层）...")
    ic = daily_rank_ic(df)
    metrics = {
        "tag": args.tag,
        "model_dir": str(model_dir),
        "val_window": [str(val_start.date()), str(val_end.date())],
        "scoring_config": {"amp": args.amp, "batch_size": args.batch_size,
                           "sample_count": args.sample_count,
                           "lookback": args.lookback, "predict": args.predict},
        "n_obs": int(len(df)),
        "n_days": int(ic.notna().sum()),
        "label_ok_share": ok_share,
        "nw_raw": newey_west_tstat(ic, 5),
        "nw_winsorized": newey_west_tstat(winsorized_rank_ic(df), 5),
        "decile_spread_net_mean": (
            float(np.nanmean(sp)) if len(df) and (sp := decile_spread(df)["net"]).notna().any()
            else None),
        "ic_by_cap_quintile": stratified_ic(df, "DlyCap"),
        "ic_by_adv_quintile": stratified_ic(df, "ADV20"),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")
    ic.rename("rank_ic").to_frame().to_parquet(out / "daily_ic.parquet")

    nw = metrics["nw_raw"]
    lines = [
        f"# 单折评估 {args.tag}", "",
        f"- 验证窗 {val_start.date()} .. {val_end.date()}；观测 {len(df):,}；"
        f"有效日 {metrics['n_days']}",
        f"- RankIC 均值 {nw['mean']:.5f}，NW t = {nw['t']:.2f}（lags=5）",
        f"- 十分位价差（扣 30bp×4 成本）均值 "
        + (f"{metrics['decile_spread_net_mean']:.5f}"
           if metrics['decile_spread_net_mean'] is not None else "n/a（有效日不足）"),
        f"- 市值分层 IC（q1 小 → q5 大）: "
        + ", ".join(f"{k}={v['mean']:.4f}" for k, v in metrics["ic_by_cap_quintile"].items()),
        f"- ADV 分层 IC: "
        + ", ".join(f"{k}={v['mean']:.4f}" for k, v in metrics["ic_by_adv_quintile"].items()),
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    smoke = f" LIMIT_OBS={args.limit_obs}（冒烟，不作判定）" if args.limit_obs else ""
    append_ledger(
        f"eval | tag={args.tag} val=[{val_start.date()}..{val_end.date()}] "
        f"n={len(df)} RankIC={nw['mean']:.5f} t={nw['t']:.2f} "
        f"（验证窗查看：RankIC/分层IC/十分位）{smoke}"
    )
    log(f"完成 → {out}")


if __name__ == "__main__":
    main()
