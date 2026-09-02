"""信号 #2（隔夜/日内分解）在**已消耗开发折**上的预注册读数执行器。

这份脚本是 ``experiments/signal2_prereg_v1.md`` 的执行器，仅此而已。

**判读阈值不在本脚本里。** 本脚本只产出读数（RankIC、与 Kronos 的秩相关、换手率），
不做任何判定、不打印任何「通过 / 不通过」。阈值与处置规则写在预注册文档里，
且必须在看到本脚本输出**之前**落笔（CLAUDE.md §二：判据先于结果）。

**不得用于比较合成规则的收益。** 等权秩相加 vs 慢信号筛子属 B 层
（MDE > SESOI，本样本量下不可回答，CLAUDE.md §二「比较分层」），
只能按写死的先验优先序决定，不能靠本脚本的数字挑。
本脚本也不算净值——CLAUDE.md §五：RankIC 已降级为诊断，钱的读数走
``scripts/compare_evaldirs_money.py``，且组合构造必须先冻结。

样本地位
--------
折 01–04 与 36–42 已被数十次决策读数消耗（CLAUDE.md §四）。本读数因此是
**方向性证据，不是无偏估计**；折 05–35 与 2024-07 起的窗口不经本脚本触碰
（``signals.kronos_adapter`` 硬拒非法折号，面板按折下推日期过滤）。

前视（CLAUDE.md §一.1）
----------------------
``label`` 在本脚本中**只作为 IC 的被解释变量**出现，从不进入任何特征构造；
标签直接读开发折已有的 ``labels.parquet``（不重算），保证与 Kronos 同一口径。
``adv20`` 一律 ``rolling(20).mean().shift(1)``，与 ``scripts/compare_arms_money.py`` 一致。
提交前自查：``grep -n label scripts/signal2_devfold_readout.py``。

用法
----
    # 今晚只跑这个（纯合成、内存内、不碰任何真实数据）
    python scripts/signal2_devfold_readout.py --dry-run

    # 队列结束后的真实读数（逐折读面板，折间 del + gc）
    python scripts/signal2_devfold_readout.py --folds 36-42 --arm both \
        --out outputs/signal2_devfold_readout.json

产出 JSON 结构
--------------
``meta``（规格哈希 / 代码哈希 / 折表 / 运行时间）、``per_fold``（逐折读数与逐日序列）、
``summary``（跨折拼接后的 NW(lag=5) t、折内均值 > 0 的折数、与 Kronos 的秩相关均值）。
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from crsp_pipeline.calendar import CalendarError, TradingCalendar  # noqa: E402
from crsp_pipeline.sealed import assert_readable  # noqa: E402
from crsp_pipeline.signal_eval import daily_rank_ic, newey_west_tstat  # noqa: E402
from crsp_pipeline.splits import walk_forward_folds  # noqa: E402
from signals.kronos_adapter import (  # noqa: E402
    ALLOWED_FOLDS,
    eval_dir,
    load_kronos_scores,
    load_labels,
)
from signals.overnight_intraday import SPEC, OvernightIntradaySignal  # noqa: E402

# ---------------------------------------------------------------- 冻结常量
# 真实数据路径：今晚**不执行**，只作为常量登记。
PROCESSED = Path(r"F:\quant\processed\crsp_ciz_2026-08-24_20260825T130601Z")
OUTPUTS = REPO / "outputs"
PANEL_FILE = "panel_raw.parquet"
MARKET_INDEX_FILE = "market_index.parquet"
# 列裁剪是硬要求（CLAUDE.md §七：禁止整表读 panel_raw，4980 万行）
PANEL_COLUMNS = ["PERMNO", "DlyCalDt", "DlyOpen", "DlyClose", "DlyRet", "DlyPrcVol"]

FOLD_CALENDAR_START = "2000-01-03"   # 与 scripts/emit_folds.py 一致
WARMUP_SESSIONS = 60                 # 验证窗前推的 warmup（>= LOOKBACK=21，留足缺失余量）
TOPN = 500                           # top500 流动性池，与 compare_arms_money.py 同口径
ADV_WINDOW, ADV_MIN = 20, 10
NW_LAG = 5                           # 6 日重叠标签 → 至少 5（signal_eval 的冻结口径）
MIN_NAMES_PER_DAY = 50               # 当日名字数低于此不计入逐日序列
TOP_DECILE = 10                      # 换手率看 top 1/10
DEFAULT_OUT = OUTPUTS / "signal2_devfold_readout.json"

CODE_FILES = (
    "scripts/signal2_devfold_readout.py",
    "src/signals/base.py",
    "src/signals/overnight_intraday.py",
    "src/signals/kronos_adapter.py",
)


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------- 折号解析


def parse_folds(spec: str) -> list[int]:
    """``"36-42"`` / ``"1,2,36-38"`` → 有序去重折号表。"""
    out: list[int] = []
    for chunk in str(spec).replace(" ", "").split(","):
        if not chunk:
            continue
        if "-" in chunk.lstrip("-"):
            a, b = chunk.split("-", 1)
            lo, hi = int(a), int(b)
            if hi < lo:
                raise ValueError(f"折区间反了：{chunk}")
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(chunk))
    seen: dict[int, None] = {}
    for f in out:
        seen.setdefault(f, None)
    return sorted(seen)


# ---------------------------------------------------------------- 逐日统计


def _daily_series(s: pd.Series) -> dict:
    s = s.dropna()
    return {
        "n_days": int(len(s)),
        "mean": float(s.mean()) if len(s) else float("nan"),
        "dates": [str(pd.Timestamp(d).date()) for d in s.index],
        "values": [float(v) for v in s.to_numpy()],
    }


def _rank_ic(df: pd.DataFrame, score_col: str, other_col: str) -> pd.Series:
    """逐日横截面 Spearman。复用 signal_eval.daily_rank_ic 保证口径一致；
    ``other_col`` 既可以是 label（IC），也可以是另一个信号的分数（秩相关）。"""
    sub = df[["signal_date", score_col, other_col]].dropna()
    if sub.empty:
        return pd.Series(dtype="float64")
    cnt = sub.groupby("signal_date")[score_col].transform("size")
    sub = sub[cnt >= MIN_NAMES_PER_DAY]
    if sub.empty:
        return pd.Series(dtype="float64")
    return daily_rank_ic(sub, score_col=score_col, label_col=other_col,
                         date_col="signal_date")


def top_decile_turnover(df: pd.DataFrame, score_col: str = "score") -> pd.Series:
    """逐日 top-decile 换手率 = 今日 top10% 中不在昨日 top10% 的比例。

    「昨日」指该池里上一个有效算分日（不是自然日前一天）。首日无前值，不出数。
    """
    sub = df[["signal_date", "PERMNO", score_col]].dropna()
    prev: set | None = None
    dates, vals = [], []
    for day, g in sub.groupby("signal_date", sort=True):
        n = len(g)
        if n < MIN_NAMES_PER_DAY:
            continue
        k = max(1, n // TOP_DECILE)
        top = set(g.nlargest(k, score_col)["PERMNO"])
        if prev is not None:
            dates.append(day)
            vals.append(len(top - prev) / len(top))
        prev = top
    return pd.Series(vals, index=pd.DatetimeIndex(dates), dtype="float64")


def _pool_top500(df: pd.DataFrame) -> pd.DataFrame:
    """当日按 adv20 取前 TOPN（adv20 = 20 日均量 shift(1)，t 日收盘可得）。"""
    sub = df[df["adv20"].notna()]
    if sub.empty:
        return sub
    rank = sub.groupby("signal_date")["adv20"].rank(ascending=False, method="first")
    return sub[rank <= TOPN]


# ---------------------------------------------------------------- 单折读数


def analyse_fold(fold_id: int, val_start, val_end, panel: pd.DataFrame,
                 labels: pd.DataFrame, kronos: dict[str, pd.DataFrame]) -> dict:
    """一折的全部读数。panel 需已含 adv20 且覆盖 [val_start - warmup, val_end]。"""
    vs, ve = pd.Timestamp(val_start), pd.Timestamp(val_end)
    sig = OvernightIntradaySignal().compute(panel)

    adv = (panel[["PERMNO", "DlyCalDt", "adv20"]]
           .rename(columns={"DlyCalDt": "signal_date"}))
    sig = sig.merge(adv, on=["signal_date", "PERMNO"], how="left")
    sig = sig[(sig["signal_date"] >= vs) & (sig["signal_date"] <= ve)]
    scored = sig[sig["score"].notna()].copy()

    lab = labels[(labels["signal_date"] >= vs) & (labels["signal_date"] <= ve)]
    merged = scored.merge(lab[["signal_date", "PERMNO", "label"]],
                          on=["signal_date", "PERMNO"], how="inner")
    pools = {"all": merged, "top500": _pool_top500(merged)}
    turn_pools = {"all": scored, "top500": _pool_top500(scored)}

    res: dict = {
        "fold": f"fold{fold_id:02d}",
        "val_start": str(vs.date()),
        "val_end": str(ve.date()),
        "n_days_scored": int(scored["signal_date"].nunique()),
        "n_obs_scored": int(len(scored)),
        "n_obs_with_label": int(len(merged)),
        "names_per_day_mean": (float(merged.groupby("signal_date").size().mean())
                               if len(merged) else float("nan")),
        "ic": {},
        "kronos_rank_corr": {},
        "turnover_top_decile": {},
    }

    for col in ("score", "score_overnight"):
        res["ic"][col] = {p: _daily_series(_rank_ic(g, col, "label"))
                          for p, g in pools.items()}
        res["turnover_top_decile"][col] = {
            p: _daily_series(top_decile_turnover(g, col))
            for p, g in turn_pools.items()}

    for arm, k in kronos.items():
        if k is None or k.empty:
            res["kronos_rank_corr"][arm] = {"n_days": 0, "mean": float("nan"),
                                            "dates": [], "values": []}
            continue
        kk = k[(k["signal_date"] >= vs) & (k["signal_date"] <= ve)]
        m = scored.merge(kk.rename(columns={"score": "kronos"}),
                         on=["signal_date", "PERMNO"], how="inner")
        res["kronos_rank_corr"][arm] = _daily_series(_rank_ic(m, "score", "kronos"))
    return res


# ---------------------------------------------------------------- 汇总


def _concat_daily(per_fold: list[dict], path: list[str]) -> pd.Series:
    """把各折的逐日序列按日期拼起来（验证窗互不重叠，直接拼）。"""
    frames = []
    for r in per_fold:
        node = r
        for key in path:
            node = node.get(key, {}) if isinstance(node, dict) else {}
        if not node:
            continue
        frames.append(pd.Series(node.get("values", []),
                                index=pd.DatetimeIndex(
                                    pd.to_datetime(node.get("dates", []))),
                                dtype="float64"))
    if not frames:
        return pd.Series(dtype="float64")
    return pd.concat(frames).sort_index()


def _pooled(per_fold: list[dict], path: list[str]) -> dict:
    s = _concat_daily(per_fold, path)
    nw = newey_west_tstat(s, NW_LAG) if len(s) else {
        "mean": float("nan"), "se": float("nan"), "t": float("nan"), "n": 0}
    per_fold_mean = []
    for r in per_fold:
        node = r
        for key in path:
            node = node.get(key, {}) if isinstance(node, dict) else {}
        per_fold_mean.append((r["fold"], float(node.get("mean", float("nan")))
                              if node else float("nan")))
    pos = sum(1 for _, m in per_fold_mean if np.isfinite(m) and m > 0)
    n_ok = sum(1 for _, m in per_fold_mean if np.isfinite(m))
    return {
        "pooled_mean": float(nw["mean"]),
        "nw_se": float(nw["se"]),
        "nw_t": float(nw["t"]),
        "n_days": int(nw["n"]),
        "nw_lag": NW_LAG,
        "folds_positive": pos,
        "folds_with_readout": n_ok,
        "per_fold_mean": dict(per_fold_mean),
    }


def summarise(per_fold: list[dict], arms: list[str]) -> dict:
    out: dict = {"ic": {}, "turnover_top_decile": {}, "kronos_rank_corr": {}}
    for col in ("score", "score_overnight"):
        out["ic"][col] = {p: _pooled(per_fold, ["ic", col, p])
                          for p in ("all", "top500")}
        out["turnover_top_decile"][col] = {
            p: _pooled(per_fold, ["turnover_top_decile", col, p])
            for p in ("all", "top500")}
    for arm in arms:
        out["kronos_rank_corr"][arm] = _pooled(per_fold, ["kronos_rank_corr", arm])
    return out


# ---------------------------------------------------------------- 数据装载


def add_adv20(df: pd.DataFrame) -> pd.DataFrame:
    """adv20 = 20 日 DlyPrcVol 均值再 shift(1)（t 日收盘可得，无前视）。"""
    df = df.sort_values(["PERMNO", "DlyCalDt"], kind="mergesort")
    df["adv20"] = df.groupby("PERMNO")["DlyPrcVol"].transform(
        lambda s: s.rolling(ADV_WINDOW, min_periods=ADV_MIN).mean().shift(1))
    return df


def load_panel_window(processed: Path, lo, hi) -> pd.DataFrame:
    """列裁剪 + 日期下推的逐折读取。**唯一**允许触碰 panel_raw 的入口。"""
    path = Path(processed) / PANEL_FILE
    assert_readable(path)
    df = pd.read_parquet(
        path, columns=PANEL_COLUMNS,
        filters=[("DlyCalDt", ">=", pd.Timestamp(lo)),
                 ("DlyCalDt", "<=", pd.Timestamp(hi))])
    df["DlyCalDt"] = pd.to_datetime(df["DlyCalDt"])
    return add_adv20(df)


def build_calendar(processed: Path) -> TradingCalendar:
    path = Path(processed) / MARKET_INDEX_FILE
    assert_readable(path)
    return TradingCalendar.from_market_index(
        pd.read_parquet(path, columns=["caldt"]), "caldt")


def fold_windows(cal: TradingCalendar, fold_ids: list[int]) -> dict[int, tuple]:
    """折号 → (warmup_start, val_start, val_end)。边界只来自 walk_forward_folds。"""
    folds = walk_forward_folds(cal, FOLD_CALENDAR_START, cal.dates[-1])
    out = {}
    for f in fold_ids:
        if f < 1 or f > len(folds):
            raise ValueError(f"折 {f} 超出日历能产出的折数 {len(folds)}")
        fd = folds[f - 1]
        try:
            warm = cal.shift(fd.val_start, -WARMUP_SESSIONS)
        except CalendarError:
            warm = cal.dates[0]
        out[f] = (warm, fd.val_start, fd.val_end)
    return out


# ---------------------------------------------------------------- 合成夹具


def make_synthetic_root(fold_ids: list[int], arms: list[str], root: Path,
                        n_permno: int = 60, seed: int = 20260903) -> Path:
    """造一套**纯合成**的迷你数据树，目录/列名与真实布局一致。

    只为把 dry-run 走成端到端（面板 → 信号 → 标签 → Kronos 分数 → JSON），
    数值本身没有任何解释力。合成面板按 open/close 恒等式生成，
    保证 ``(1+overnight)(1+intraday) = 1+DlyRet`` 成立。
    """
    rng = np.random.default_rng(seed)
    root = Path(root)
    proc, outs = root / "processed", root / "outputs"
    proc.mkdir(parents=True, exist_ok=True)

    cal_dates = pd.bdate_range("2000-01-03", "2024-12-31")
    pd.DataFrame({"caldt": cal_dates}).to_parquet(
        proc / MARKET_INDEX_FILE, index=False)
    cal = TradingCalendar(cal_dates)
    windows = fold_windows(cal, fold_ids)

    permnos = np.arange(10001, 10001 + n_permno, dtype="int64")
    frames, lab_frames = [], []
    for f, (warm, vs, ve) in windows.items():
        days = cal.sessions(warm, ve)
        n_d, n_p = len(days), len(permnos)
        on = rng.normal(0.0, 0.010, size=(n_d, n_p))
        intr = rng.normal(0.0, 0.015, size=(n_d, n_p))
        prev_close = np.full(n_p, 30.0)
        rows_o, rows_c, rows_r = [], [], []
        for i in range(n_d):
            op = prev_close * (1.0 + on[i])
            cl = op * (1.0 + intr[i])
            rows_o.append(op)
            rows_c.append(cl)
            rows_r.append((1.0 + on[i]) * (1.0 + intr[i]) - 1.0)
            prev_close = cl
        blk = pd.DataFrame({
            "PERMNO": np.tile(permnos, n_d),
            "DlyCalDt": np.repeat(days.to_numpy(), n_p),
            "DlyOpen": np.concatenate(rows_o),
            "DlyClose": np.concatenate(rows_c),
            "DlyRet": np.concatenate(rows_r),
            "DlyPrcVol": rng.lognormal(15.0, 1.0, size=n_d * n_p),
        })
        frames.append(blk)

        val = blk[blk["DlyCalDt"] >= vs]
        lab = pd.DataFrame({
            "PERMNO": val["PERMNO"].to_numpy(),
            "signal_date": val["DlyCalDt"].to_numpy(),
            # 合成标签：与当日 intraday 弱相关 + 噪声（纯为让 Spearman 非退化）
            "label": (0.3 * (val["DlyClose"] / val["DlyOpen"] - 1.0).to_numpy()
                      + rng.normal(0.0, 0.05, size=len(val))),
            "status": np.where(rng.random(len(val)) < 0.95, "ok", "unfillable"),
        })
        lab_frames.append((f, lab))

    panel = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["PERMNO", "DlyCalDt"])
    panel.to_parquet(proc / PANEL_FILE, index=False)

    for f, lab in lab_frames:
        for arm in set(arms) | {"ft"}:
            d = eval_dir(f, arm, root=outs)
            d.mkdir(parents=True, exist_ok=True)
            sc = lab[["PERMNO", "signal_date"]].copy()
            sc["score"] = rng.normal(size=len(sc))
            sc.to_parquet(d / "scores.parquet", index=False)
            if arm == "ft":
                lab.to_parquet(d / "labels.parquet", index=False)
    return root


# ---------------------------------------------------------------- 主流程


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while blk := fh.read(1 << 20):
            h.update(blk)
    return h.hexdigest()


def run(fold_ids: list[int], arms: list[str], processed: Path, outputs: Path,
        out_path: Path, mode: str) -> dict:
    t0 = time.time()
    cal = build_calendar(processed)
    windows = fold_windows(cal, fold_ids)

    per_fold = []
    for f in fold_ids:
        warm, vs, ve = windows[f]
        log(f"[fold{f:02d}] val=[{vs.date()}..{ve.date()}] warmup<={warm.date()}")
        panel = load_panel_window(processed, warm, ve)
        labels = load_labels([f], root=outputs)          # 只读 status == "ok"
        kron = {arm: load_kronos_scores([f], arm, root=outputs) for arm in arms}
        per_fold.append(analyse_fold(f, vs, ve, panel, labels, kron))
        del panel, labels, kron
        gc.collect()

    report = {
        "meta": {
            "script": "scripts/signal2_devfold_readout.py",
            "prereg": "experiments/signal2_prereg_v1.md",
            "mode": mode,
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "elapsed_sec": round(time.time() - t0, 2),
            "spec": SPEC.as_dict(),
            "spec_hash": SPEC.spec_hash(),
            "code_sha256": {rel: sha256_file(REPO / rel)
                            for rel in CODE_FILES if (REPO / rel).exists()},
            "arms": arms,
            "folds": [{"fold": f"fold{f:02d}",
                       "val_start": str(windows[f][1].date()),
                       "val_end": str(windows[f][2].date()),
                       "warmup_start": str(windows[f][0].date())}
                      for f in fold_ids],
            "params": {"topn": TOPN, "adv_window": ADV_WINDOW,
                       "adv_min_periods": ADV_MIN, "nw_lag": NW_LAG,
                       "warmup_sessions": WARMUP_SESSIONS,
                       "min_names_per_day": MIN_NAMES_PER_DAY,
                       "top_decile_denom": TOP_DECILE},
            "sample_status": ("折 01-04 / 36-42 已消耗，读数为方向性证据、非无偏估计"
                              "（CLAUDE.md §四）"),
            "caveat": "本脚本不含判读阈值，也不得用于比较合成规则的收益",
        },
        "per_fold": {r["fold"]: r for r in per_fold},
        "summary": summarise(per_fold, arms),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    log(f"写出 {out_path}（{len(per_fold)} 折，{report['meta']['elapsed_sec']}s）")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--folds", default="36-42", help="折号，如 36-42 或 1,2,36-38")
    ap.add_argument("--include-early", action="store_true",
                    help="额外加折 01-04 作年代诊断")
    ap.add_argument("--arm", choices=["ft", "zs", "both"], default="both")
    ap.add_argument("--dry-run", action="store_true",
                    help="用合成数据走通全流程，写到临时目录；不碰任何真实数据")
    ap.add_argument("--out", default=None,
                    help=f"输出 JSON（默认 {DEFAULT_OUT}；--dry-run 时默认写临时目录）")
    args = ap.parse_args(argv)

    fold_ids = parse_folds(args.folds)
    if args.include_early:
        fold_ids = sorted(set(fold_ids) | {1, 2, 3, 4})
    bad = [f for f in fold_ids if f not in ALLOWED_FOLDS]
    if bad:
        raise SystemExit(
            f"折 {bad} 不在已消耗的开发折集合内；折 05-35 与 2024-07 起的窗口"
            f"须用户另行授权（CLAUDE.md §四）")
    arms = ["ft", "zs"] if args.arm == "both" else [args.arm]

    if args.dry_run:
        root = Path(tempfile.mkdtemp(prefix="signal2_dryrun_"))
        log(f"[dry-run] 合成数据树 → {root}（不读 {PROCESSED}，不读 {OUTPUTS}）")
        make_synthetic_root(fold_ids, arms, root)
        out = Path(args.out) if args.out else root / "signal2_devfold_readout.json"
        run(fold_ids, arms, root / "processed", root / "outputs", out, "dry-run")
        return 0

    out = Path(args.out) if args.out else DEFAULT_OUT
    run(fold_ids, arms, PROCESSED, OUTPUTS, out, "real")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
