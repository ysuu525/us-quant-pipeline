"""冻结主干 + 横截面排序头：三项改造的落地入口。

三项改造（依据见 experiments/ledger.md 2026-08-30/31 条目与两份调研）
  ① 标签：官方 execution-return **在当日截面内转秩**（去掉市场共同成分）
  ② 损失：直接优化 RankIC 的可微形式（替代生成式逐 token loss）
  ③ 输出：确定性排序头（替代 5 路径采样 → 消除实测约 0.25 的采样噪声）
早停指标同步改为内层验证 RankIC（此前用生成 loss，与目标脱钩）。

主干默认**冻结且不微调**（依据：epoch 探针 e1≈e30、E12 零样本≈微调），
只前向一次抽表示；也可 --backbone-dir 指向任一微调产物做对照。

输出与 evaluate_fold 同构（scores.parquet / labels.parquet / metrics.json /
daily_ic.parquet），下游全部分析脚本可直接复用。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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
from kronos_ft.models import load_pretrained, pick_device  # noqa: E402
from kronos_ft.rank_head import train_head  # noqa: E402
from kronos_ft.represent import extract_representations  # noqa: E402
from kronos_ft.train import append_ledger  # noqa: E402
from kronos_ft.windows import build_scoring_index, filter_index_by_universe  # noqa: E402


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def stratified_ic(df: pd.DataFrame, col: str, q: int = 5) -> dict:
    out = {}
    d = df.dropna(subset=[col])
    if not len(d):
        return out
    grp = d.groupby("signal_date")[col].transform(
        lambda s: pd.qcut(s.rank(method="first"), q, labels=False, duplicates="drop"))
    for i in range(q):
        sub = d[grp == i]
        out[f"q{i + 1}"] = newey_west_tstat(daily_rank_ic(sub), 5) if len(sub) else {}
    return out


def build_block(cal: TradingCalendar, adj: pd.DataFrame, uni: pd.DataFrame,
                lookback: int, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    sidx = build_scoring_index(adj, cal, lookback)
    sidx = sidx[(sidx["anchor"] >= start) & (sidx["anchor"] <= end)]
    return filter_index_by_universe(sidx, uni).reset_index(drop=True)


def labels_for(P: Path, cal: TradingCalendar, idx: pd.DataFrame, lo, hi, predict: int):
    raw = pd.read_parquet(P / "panel_raw.parquet",
                          columns=["PERMNO", "DlyCalDt", "DlyOpen", "DlyClose",
                                   "DlyRet", "DlyDelFlg", "DlyCap", "DlyPrcVol"])
    raw = raw[(raw["DlyCalDt"] >= lo) & (raw["DlyCalDt"] <= hi)]
    raw = raw[raw["PERMNO"].isin(set(idx["PERMNO"].unique()))]
    dist = pd.read_parquet(P / "distributions.parquet",
                           columns=["permno", "disexdt", "disdivamt"])
    cash = dist[dist["disdivamt"].fillna(0) > 0].rename(
        columns={"permno": "PERMNO", "disexdt": "ex_date", "disdivamt": "div_cash"})
    obs = idx[["PERMNO", "anchor"]].rename(columns={"anchor": "signal_date"})
    labels = compute_labels(raw, obs, cal, cash_dividends=cash, predict_window=predict)
    return labels, raw


def to_daily_rank(df: pd.DataFrame, col: str = "label") -> np.ndarray:
    """改造①：当日截面内转秩到 [0,1]，等价于剔除市场共同成分。"""
    return df.groupby("signal_date")[col].rank(pct=True).to_numpy(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description="冻结主干 + 横截面排序头")
    ap.add_argument("--processed", required=True)
    ap.add_argument("--backbone-dir", default="outputs/zeroshot_base",
                    help="含 tokenizer_final/predictor_final；默认未微调的预训练底座")
    ap.add_argument("--train-start", required=True)
    ap.add_argument("--train-end", required=True)
    ap.add_argument("--val-start", required=True)
    ap.add_argument("--val-end", required=True)
    ap.add_argument("--lookback", type=int, default=90)
    ap.add_argument("--predict", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--amp", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--loss", choices=["ic", "topk"], default="ic")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inner-months", type=int, default=6)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    P = Path(args.processed)
    out = Path(args.out) / f"eval_{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    ts, te = pd.Timestamp(args.train_start), pd.Timestamp(args.train_end)
    vs, ve = pd.Timestamp(args.val_start), pd.Timestamp(args.val_end)

    cal = TradingCalendar.from_market_index(
        pd.read_parquet(P / "market_index.parquet"), "caldt")
    lo = cal.shift(cal.snap_forward(ts), -(args.lookback + 30))
    hi = cal.dates[min(cal.index_of(cal.snap_back(ve)) + args.predict + 2, len(cal) - 1)]
    log(f"训练窗 [{ts.date()}..{te.date()}]  验证窗 [{vs.date()}..{ve.date()}]  "
        f"面板切片 [{lo.date()}..{hi.date()}]")

    adj = pd.read_parquet(P / "panel_kronos_adj.parquet")
    adj = adj[(adj["DlyCalDt"] >= lo) & (adj["DlyCalDt"] <= hi)]
    uni = pd.read_parquet(P / "universe.parquet",
                          columns=["PERMNO", "DlyCalDt", "in_universe"])

    idx_tr = build_block(cal, adj, uni, args.lookback, ts, te)
    idx_va = build_block(cal, adj, uni, args.lookback, vs, ve)
    log(f"样本：训练 {len(idx_tr):,}  验证 {len(idx_va):,}")

    log(f"加载主干（冻结）：{args.backbone_dir}")
    tok, mdl = load_pretrained(f"{args.backbone_dir}/tokenizer_final",
                               f"{args.backbone_dir}/predictor_final")
    dev = pick_device(args.device)

    log("抽表示（确定性单次前向，无采样）...")
    need = set(idx_tr["PERMNO"]) | set(idx_va["PERMNO"])
    adj_use = adj[adj["PERMNO"].isin(need)]
    ix_tr, E_tr = extract_representations(tok, mdl, adj_use, idx_tr, cal, args.lookback,
                                          batch_size=args.batch_size, device=args.device,
                                          amp=args.amp)
    log(f"  训练表示 {E_tr.shape}")
    ix_va, E_va = extract_representations(tok, mdl, adj_use, idx_va, cal, args.lookback,
                                          batch_size=args.batch_size, device=args.device,
                                          amp=args.amp)
    log(f"  验证表示 {E_va.shape}")
    del adj_use
    torch.cuda.empty_cache()

    log("§4 标签（官方 execution-return）...")
    lab_tr, _ = labels_for(P, cal, idx_tr, lo, hi, args.predict)
    lab_va, raw_va = labels_for(P, cal, idx_va, lo, hi, args.predict)

    def prep(ix, E, lab):
        m = ix.merge(lab[["PERMNO", "signal_date", "status", "label"]],
                     on=["PERMNO", "signal_date"], how="left")
        keep = ((m["status"] == "ok") & m["label"].notna()).to_numpy()
        return m[keep].reset_index(drop=True), E[keep]

    m_tr, E_tr = prep(ix_tr, E_tr, lab_tr)
    m_va, E_va = prep(ix_va, E_va, lab_va)
    log(f"有效样本：训练 {len(m_tr):,}  验证 {len(m_va):,}")

    y_tr = to_daily_rank(m_tr)
    y_va = to_daily_rank(m_va)
    day_tr = m_tr["signal_date"].to_numpy().astype("datetime64[D]").astype(np.int64)
    day_va = m_va["signal_date"].to_numpy().astype("datetime64[D]").astype(np.int64)

    inner_start = cal.snap_forward(
        m_tr["signal_date"].max() - pd.DateOffset(months=args.inner_months))
    inner_cut = cal.shift(inner_start, -(args.predict + 1))
    mask_in = (m_tr["signal_date"] >= inner_start).to_numpy()
    mask_out = (m_tr["signal_date"] <= inner_cut).to_numpy()
    log(f"内层验证窗 [{inner_start.date()}..{m_tr['signal_date'].max().date()}]，"
        f"外层截至 {inner_cut.date()}（purge {args.predict + 1} 日）")

    log(f"训练排序头（损失={args.loss}，早停指标=内层 RankIC）...")
    head, info = train_head(
        E_tr[mask_out], day_tr[mask_out], y_tr[mask_out],
        E_tr[mask_in], day_tr[mask_in], y_tr[mask_in],
        d_in=E_tr.shape[1], device=dev, seed=args.seed, loss_kind=args.loss)

    head.eval()
    with torch.no_grad():
        score = head(torch.from_numpy(E_va.astype(np.float32)).to(dev)).cpu().numpy()
    scores = pd.DataFrame({"PERMNO": m_va["PERMNO"].to_numpy(),
                           "signal_date": m_va["signal_date"].to_numpy(),
                           "score": score.astype(np.float64),
                           "extrapolated": False})
    scores.to_parquet(out / "scores.parquet", index=False)
    lab_va.to_parquet(out / "labels.parquet", index=False)
    torch.save(head.state_dict(), out / "rank_head.pt")

    df = scores.merge(lab_va[["PERMNO", "signal_date", "status", "label"]],
                      on=["PERMNO", "signal_date"])
    df = df[(df["status"] == "ok") & df["score"].notna()]
    cap = raw_va[["PERMNO", "DlyCalDt", "DlyCap"]].rename(
        columns={"DlyCalDt": "signal_date"})
    df = df.merge(cap, on=["PERMNO", "signal_date"], how="left")
    advd = raw_va.sort_values(["PERMNO", "DlyCalDt"]).copy()
    advd["ADV20"] = advd.groupby("PERMNO")["DlyPrcVol"].transform(
        lambda s: s.rolling(20, min_periods=15).mean())
    df = df.merge(advd[["PERMNO", "DlyCalDt", "ADV20"]].rename(
        columns={"DlyCalDt": "signal_date"}), on=["PERMNO", "signal_date"], how="left")

    ic = daily_rank_ic(df)
    sp = decile_spread(df)["net"]
    nw = newey_west_tstat(ic, 5)
    metrics = {
        "tag": args.tag,
        "model_dir": args.backbone_dir,
        "val_window": [str(vs.date()), str(ve.date())],
        "n_obs": int(len(df)),
        "n_days": int(ic.notna().sum()),
        "label_ok_share": float((lab_va["status"] == "ok").mean()),
        "scoring_config": {"amp": args.amp, "batch_size": args.batch_size,
                           "sample_count": 0, "lookback": args.lookback,
                           "predict": args.predict, "mode": "rank_head",
                           "loss": args.loss, "seed": args.seed,
                           "backbone": args.backbone_dir, "frozen_backbone": True},
        "head_training": {k: v for k, v in info.items() if k != "history"},
        "nw_raw": nw,
        "nw_winsorized": newey_west_tstat(winsorized_rank_ic(df), 5),
        "decile_spread_net_mean": float(np.nanmean(sp)) if sp.notna().any() else None,
        "ic_by_cap_quintile": stratified_ic(df, "DlyCap"),
        "ic_by_adv_quintile": stratified_ic(df, "ADV20"),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "head_history.json").write_text(
        json.dumps(info["history"], indent=2), encoding="utf-8")
    ic.to_frame("rank_ic").to_parquet(out / "daily_ic.parquet")

    log(f"RankIC {nw['mean']:+.5f}  NW t {nw['t']:.2f}  "
        f"(内层最优 {info['best_val_rank_ic']:+.5f} @ epoch {info['best_epoch']})")
    append_ledger(
        f"eval | tag={args.tag} val=[{vs.date()}..{ve.date()}] n={len(df)} "
        f"RankIC={nw['mean']:.5f} t={nw['t']:.2f} 【排序头·冻结主干·确定性输出】"
        f" 内层最优 epoch={info['best_epoch']}")


if __name__ == "__main__":
    main()
