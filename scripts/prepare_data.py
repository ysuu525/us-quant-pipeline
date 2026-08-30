"""Phase 5 数据准备：CRSP 快照原始层 → 训练/评估输入（RUNBOOK §3 的那条命令）。

    python scripts/prepare_data.py --snapshot <快照目录> --out <输出目录>

产出（--out 目录）：

    market_index.parquet        市场指数（caldt/vwretd/...，交易日历来源，
                                train 的 --index-parquet 直接吃它）
    panel_raw.parquet           未过滤全量面板（§3 收益面板：规范 §1 列，零复权，
                                labels / universe / signal_eval 的输入）
    panel_kronos_adj.parquet    训练蜡烛面板（§5 复权后的 OHLCV+amt，
                                train 的 --panel 直接吃它）
    split_events.parquet        复权事件表（拆股+股票股利，factor=新/旧）
    universe.parquet            选股面板逐日标志（§2：static/price/adv/age/cap/in_universe）
    audit/audit.json|report.md  §10 coverage audit + §5 双路验证 + §9 统计
    prep_manifest.json          §11 可复现性：输入快照 ID、复权锚、行数、代码版本

2026-08-26 真实数据核对后冻结的口径（探查记录见 audit/report.md）：

- 复权事件 = ``distype='FRS'``（disdetailtype ∈ {STKSPL 拆股, STKDIV 股票股利}），
  factor = 1 + disfacshr（CIZ 沿用 legacy 约定 facshr=(新−旧)/旧；反向拆股
  facshr<0，factor∈(0,1)）。现金股息/分拆/配股不写入 OHLC（§5）。
- ``DlyFacPrc`` 语义 = **当期事件因子**（AAPL 2020-08-31 / NVDA 2024-06-10
  双路验证均判 event）；管线仍走事件累计路径（§5），DlyFacPrc 只用于交叉审计。
- ``DlyCap`` 单位 = $千（AAPL 2020-08-31 → $2.21T 量级校验通过）。
- ``DlyPrcVol`` = DlyClose × DlyVol（精确成立）。

当前快照拆两处的过渡用法（08-24 快照日线完整但缺小表）：

    python scripts/prepare_data.py --snapshot <含日线的快照> \
        --events-snapshot <含小表的快照>

补齐后（downloader --resume）用单一快照即可，prep_manifest 会记录实际来源。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from crsp_pipeline.adjust import (  # noqa: E402
    SPLIT_DISTYPE,
    adjust_panel,
    dual_path_report,
    split_events_from_distributions,
)
from crsp_pipeline.calendar import TradingCalendar  # noqa: E402
from crsp_pipeline.cleaning import ba_flag_stats, exclusion_report, lookback_usable_mask  # noqa: E402
from crsp_pipeline.snapshot import (  # noqa: E402
    load_daily,
    load_manifest,
    load_market_index,
    load_table,
)
from crsp_pipeline.universe import selection_panel, static_eligible_intervals  # noqa: E402

# §5 验证锚点：已知拆股 (ticker, permno, ex_date, 年份)
DUAL_PATH_CASES = [
    ("AAPL", 14593, "2020-08-31", 2020),
    ("NVDA", 86580, "2024-06-10", 2024),
]


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def audit_panel(panel: pd.DataFrame, events: pd.DataFrame, dist: pd.DataFrame,
                calendar: TradingCalendar, lookbacks: list[int]) -> dict:
    """§10 coverage audit + §5 双路验证 + §9 统计。返回 JSON 可序列化 dict。"""
    rep: dict = {}
    years = panel["DlyCalDt"].dt.year

    # -- 唯一性与总量（§10 不变量）
    dup = panel.duplicated(["PERMNO", "DlyCalDt"]).sum()
    rep["rows"] = int(len(panel))
    rep["permnos"] = int(panel["PERMNO"].nunique())
    rep["date_range"] = [str(panel["DlyCalDt"].min().date()), str(panel["DlyCalDt"].max().date())]
    rep["duplicate_permno_date_rows"] = int(dup)

    # -- 累计 distinct PERMNO 非递减（按构造必然成立，报告首末年计数做量级检查）
    per_year_new = panel.groupby(years)["PERMNO"].nunique()
    rep["permnos_per_year"] = {int(k): int(v) for k, v in per_year_new.items()}

    # -- DlyOpen 缺失率（逐年）
    open_missing = panel["DlyOpen"].isna() | (panel["DlyOpen"] <= 0)
    rep["dlyopen_missing_rate_by_year"] = {
        int(k): round(float(v), 6) for k, v in open_missing.groupby(years).mean().items()
    }

    # -- BA 报价中点占比（§9，逐年）
    ba = ba_flag_stats(panel)
    rep["ba_share_by_year"] = {int(r["year"]): round(float(r["ba_share"]), 6) for _, r in ba.iterrows()}

    # -- OHLC 一致性：low ≤ min(open,close) 且 max(open,close) ≤ high（有效行上）
    valid = panel[["DlyOpen", "DlyHigh", "DlyLow", "DlyClose"]].notna().all(axis=1)
    v = panel[valid]
    viol = ((v["DlyLow"] > v[["DlyOpen", "DlyClose"]].min(axis=1)) |
            (v["DlyHigh"] < v[["DlyOpen", "DlyClose"]].max(axis=1)))
    rep["ohlc_violation_rate"] = round(float(viol.mean()), 8)
    rep["ohlc_violation_rows"] = int(viol.sum())

    # -- DlyCap 覆盖率与量级（AAPL）
    rep["dlycap_coverage"] = round(float(panel["DlyCap"].notna().mean()), 6)
    aapl = panel[(panel["PERMNO"] == 14593)
                 & (panel["DlyCalDt"] == pd.Timestamp("2020-08-31"))]
    if len(aapl):
        cap = float(aapl["DlyCap"].iloc[0])
        rep["dlycap_aapl_2020-08-31"] = cap
        rep["dlycap_unit_check"] = "thousands_usd_ok" if 1.5e9 < cap < 3e9 else "UNEXPECTED"

    # -- 退市终值记录（§10：每年 > 0）
    delist_rows = panel["DlyDelFlg"].astype(str).str.upper().eq("Y")
    per_year_delist = delist_rows.groupby(years).sum()
    rep["delist_rows_by_year"] = {int(k): int(v) for k, v in per_year_delist.items()}
    rep["delist_every_year_positive"] = bool((per_year_delist > 0).all())

    # -- §5 交叉审计：DlyFacPrc（当期事件因子）vs 事件表 1+disfacshr
    fac = panel["DlyFacPrc"]
    fac_rows = panel[fac.notna() & (fac != 1.0)][["PERMNO", "DlyCalDt", "DlyFacPrc"]]
    merged = fac_rows.merge(events, left_on=["PERMNO", "DlyCalDt"],
                            right_on=["PERMNO", "ex_date"], how="left")
    matched = np.isclose(merged["DlyFacPrc"], merged["factor"], rtol=1e-4)
    rep["facprc_event_rows"] = int(len(fac_rows))
    rep["facprc_matched_by_split_events"] = int(matched.sum())
    rep["facprc_unmatched_rows"] = int((~matched).sum())
    rep["facprc_match_rate"] = round(float(matched.mean()), 6) if len(merged) else None
    # 不匹配行的解释：DlyFacPrc 是价格因子，还会反映分拆（SP/SEC*）等事件；
    # §5 冻结规则刻意不把这些写入蜡烛。按事件类型归类以证实无遗漏的拆股。
    unmatched = merged[~matched][["PERMNO", "DlyCalDt"]]
    d = dist[["permno", "disexdt", "distype"]].copy()
    d["disexdt"] = pd.to_datetime(d["disexdt"])
    um = unmatched.merge(d, left_on=["PERMNO", "DlyCalDt"],
                         right_on=["permno", "disexdt"], how="left")
    rep["facprc_unmatched_by_distype"] = (
        um["distype"].fillna("NO_DIST_EVENT").value_counts().to_dict()
    )
    # 反向核对：事件表里有、但当日 DlyFacPrc 不等的（含当日无行的退市/停牌）
    ev_merged = events.merge(panel[["PERMNO", "DlyCalDt", "DlyFacPrc"]],
                             left_on=["PERMNO", "ex_date"],
                             right_on=["PERMNO", "DlyCalDt"], how="left")
    ev_ok = np.isclose(ev_merged["DlyFacPrc"].astype(float), ev_merged["factor"], rtol=1e-4)
    rep["events_total"] = int(len(events))
    rep["events_reflected_in_facprc"] = int(np.nansum(ev_ok))

    # -- §5 双路验证（AAPL / NVDA golden case）
    dual = {}
    for name, pn, ex_date, year in DUAL_PATH_CASES:
        sec = panel[panel["PERMNO"] == pn].set_index("DlyCalDt").sort_index()
        if len(sec) == 0:
            dual[name] = "PERMNO_NOT_IN_PANEL"
            continue
        ev = events[events["PERMNO"] == pn][["ex_date", "factor"]]
        r = dual_path_report(sec, ev, anchor=str(sec.index.max().date()))
        dual[name] = {
            "ex_date": ex_date,
            "matches_event_semantics": bool(r["matches_event_semantics"]),
            "matches_cumulative_semantics": bool(r["matches_cumulative_semantics"]),
            "conclusion": r["conclusion"],
        }
    rep["dual_path"] = dual

    # -- §9 lookback 缺口排除率（重计算，逐年；交易所维度待接 PrimaryExch）
    excl = {}
    for lb in lookbacks:
        log(f"  §9 排除率 lookback={lb}（全量逐股滚动，较慢）...")
        usable = lookback_usable_mask(panel, calendar, lb)
        er = exclusion_report(usable, panel, exch_col=None)
        excl[str(lb)] = {int(r["year"]): round(float(r["exclusion_rate"]), 6)
                         for _, r in er.iterrows()}
    rep["lookback_exclusion_rate_by_year"] = excl
    return rep


def write_report_md(audit: dict, out: Path) -> None:
    lines = ["# Phase 5 数据审计报告（§10 / §5 / §9）", "",
             f"生成时间：{datetime.now(timezone.utc).isoformat(timespec='seconds')}", ""]
    lines += [
        f"- 面板：{audit['rows']:,} 行 / {audit['permnos']:,} 只证券 / "
        f"{audit['date_range'][0]} … {audit['date_range'][1]}",
        f"- PERMNO+date 重复行：{audit['duplicate_permno_date_rows']}（要求 0）",
        f"- OHLC 违反率（low≤min(o,c) ∧ max(o,c)≤high）：{audit['ohlc_violation_rate']}"
        f"（{audit['ohlc_violation_rows']} 行）",
        f"- DlyCap 覆盖率：{audit['dlycap_coverage']}；量级校验：{audit.get('dlycap_unit_check', 'n/a')}",
        f"- 每年退市终值记录均 > 0：{audit['delist_every_year_positive']}",
        f"- DlyFacPrc 事件行 {audit['facprc_event_rows']} 中被拆股事件表解释：",
        f"  {audit['facprc_matched_by_split_events']}（匹配率 {audit['facprc_match_rate']}；"
        "不匹配行为分拆等价格因子事件，§5 刻意不写入蜡烛，见下）",
        f"- 不匹配行按事件类型：{audit['facprc_unmatched_by_distype']}",
        f"- 事件表 {audit['events_total']} 条中当日 DlyFacPrc 一致：{audit['events_reflected_in_facprc']}",
        "",
        "## §5 双路验证（DlyFacPrc 语义）", "",
    ]
    for name, r in audit["dual_path"].items():
        lines.append(f"- {name}: {r}")
    lines += ["", "## DlyOpen 缺失率（逐年）", ""]
    for y, v in audit["dlyopen_missing_rate_by_year"].items():
        lines.append(f"- {y}: {v}")
    lines += ["", "## DlyPrcFlg='BA' 占比（逐年）", ""]
    for y, v in audit["ba_share_by_year"].items():
        lines.append(f"- {y}: {v}")
    lines += ["", "## §9 lookback 缺口排除率（逐年）", ""]
    for lb, by_year in audit["lookback_exclusion_rate_by_year"].items():
        lines.append(f"### lookback={lb}")
        for y, v in by_year.items():
            lines.append(f"- {y}: {v}")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="CRSP 快照 → 训练/评估输入（Phase 5）")
    ap.add_argument("--snapshot", required=True, help="含 raw/daily 的快照目录")
    ap.add_argument("--events-snapshot", default=None,
                    help="小表（distributions/delists/security_info/indexes）所在快照；"
                         "默认与 --snapshot 相同。过渡用，快照补齐后不需要")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--anchor", default=None,
                    help="复权锚定基准日（§5，固定不漂移）；默认 = manifest 的 actual_max_trading_date")
    ap.add_argument("--years", default=None,
                    help="只处理这些年份（冒烟用），如 2020,2021；默认全部")
    ap.add_argument("--stages", default="index,events,panel,audit,universe",
                    help="要跑的阶段，逗号分隔")
    ap.add_argument("--exclusion-lookbacks", default="90",
                    help="§9 排除率报告的 lookback 档，逗号分隔（全档 60,90,200,400 较慢）")
    args = ap.parse_args()

    stages = set(args.stages.split(","))
    daily_snap = Path(args.snapshot)
    event_snap = Path(args.events_snapshot) if args.events_snapshot else daily_snap
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit").mkdir(exist_ok=True)

    daily_manifest = load_manifest(daily_snap, allow_in_progress=True)
    event_manifest = load_manifest(event_snap, allow_in_progress=True)
    anchor = pd.Timestamp(args.anchor or daily_manifest["actual_max_trading_date"])
    years = [int(y) for y in args.years.split(",")] if args.years else None
    log(f"复权锚 anchor={anchor.date()}  日线快照={daily_snap.name}  小表快照={event_snap.name}")

    # ---- 1. 市场指数 / 交易日历
    log("市场指数 → market_index.parquet")
    idx = load_market_index(event_snap)
    idx.to_parquet(out / "market_index.parquet", index=False)
    calendar = TradingCalendar.from_market_index(idx, "caldt")

    # ---- 2. 事件表
    dist = load_table(event_snap, "distributions")
    events = split_events_from_distributions(dist)
    if "events" in stages:
        log(f"事件表：distributions {len(dist):,} 行 → 复权事件 {len(events):,} 条")
        events.to_parquet(out / "split_events.parquet", index=False)
        for name in ("distributions", "delists", "security_info_history"):
            load_table(event_snap, name).to_parquet(out / f"{name}.parquet", index=False)

    # ---- 3. 面板
    log("读全量日线（这一步最吃内存）...")
    panel = load_daily(daily_snap, years=years)
    log(f"panel_raw: {len(panel):,} 行 / {panel['PERMNO'].nunique():,} 只")
    if "panel" in stages:
        panel.to_parquet(out / "panel_raw.parquet", index=False)
        log("§5 复权 → panel_kronos_adj.parquet")
        adj = adjust_panel(panel, events, anchor)
        adj.to_parquet(out / "panel_kronos_adj.parquet", index=False)
        del adj

    # ---- 4. 审计
    if "audit" in stages:
        log("审计（§10 / §5 / §9）...")
        lookbacks = [int(x) for x in args.exclusion_lookbacks.split(",") if x]
        audit = audit_panel(panel, events, dist, calendar, lookbacks)
        (out / "audit" / "audit.json").write_text(
            json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
        write_report_md(audit, out / "audit")
        dual = {k: (v["conclusion"] if isinstance(v, dict) else v)
                for k, v in audit["dual_path"].items()}
        log(f"审计完成：重复行={audit['duplicate_permno_date_rows']} "
            f"OHLC违反率={audit['ohlc_violation_rate']} 双路={dual}")

    # ---- 5. 选股面板（§2）
    if "universe" in stages:
        log("universe 标志（§2，全量逐股滚动，最慢的一步）...")
        info = load_table(event_snap, "security_info_history")
        intervals = static_eligible_intervals(info)
        beg = pd.to_datetime(info["securitybegdt"], errors="coerce")
        first_trade = (
            pd.DataFrame({"permno": info["permno"], "beg": beg})
            .dropna().groupby("permno")["beg"].min()
        )
        sel = selection_panel(panel, calendar, intervals=intervals,
                              first_trade_dates=first_trade)
        flags = sel[["PERMNO", "DlyCalDt", "static_ok", "price_ok", "adv_ok",
                     "age_ok", "cap_ok", "in_universe"]]
        flags.to_parquet(out / "universe.parquet", index=False)
        n_by_day = sel[sel["in_universe"]].groupby("DlyCalDt").size()
        log(f"universe: 日均入选 {n_by_day.mean():.0f} 只 "
            f"(min {n_by_day.min()}, max {n_by_day.max()})")

    # ---- 6. §11 prep manifest
    prep_manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "daily_snapshot_id": daily_manifest["snapshot_id"],
        "events_snapshot_id": event_manifest["snapshot_id"],
        "daily_snapshot_complete": (daily_snap / "metadata" / "snapshot_manifest.json").is_file(),
        "anchor_date": str(anchor.date()),
        "years": years,
        "stages": sorted(stages),
        "split_event_rule": f"distype=='{SPLIT_DISTYPE}', factor=1+disfacshr（2026-08-26 冻结）",
        "repo_commit": git_head(),
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
    }
    (out / "prep_manifest.json").write_text(
        json.dumps(prep_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"完成 → {out}")


if __name__ == "__main__":
    main()
