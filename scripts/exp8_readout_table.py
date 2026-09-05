"""实验 8：读出方案对比表（统一到折 36–42 的同一口径）。

判据 / 用途限制（抄自 docs/实验指令_2026-09-04.md 实验 8，先写后跑）
- **估计交付**。**不用于选读出**：读出方案属 B 层（按先验，CLAUDE.md §二），
  本实验只把表格口径补齐，不重开选择。
- 超参照原冻结不动；**内层选参口径与线性探针一致**（训练窗尾部内层验证窗，
  外层验证窗**绝不用于调参**）；**不做任何网格扩张**。
- 同时报诚实口径与事后最优（oracle）上界，且 oracle 列必须逐格标注
  「**不可用于判定**」。
- **凡是折数或天数不同的两个数，表里必须分行，不得并列成一句结论。**

本脚本只做汇总，不训练、不打分、不碰任何标签：
- **本次新跑 / 复用**的行（MLP 排序头七折、线性探针七折）从各自产物目录读
  `metrics.json` 与逐日 RankIC 序列，用
  `crsp_pipeline.signal_eval.newey_west_tstat(series, lags=5)` 出合并日读数；
- **抄自登记簿**的行（生成式 ZS / FT、树基线、ledger:141 的 MLP 单折）
  逐字抄 `experiments/ledger.md`，**不重算**，登记簿没写的一律记「未核」。

两个 RankIC 列的区别（CLAUDE.md §八：不同口径不得并列）
- `全池 RankIC`：**折等权**均值（各折逐日均值再等权平均）——登记簿里 ZS/FT/探针
  的七折与五折均值都是这个口径；
- `全池 RankIC（合并日）`：把各折的逐日序列拼起来后按**天**等权——树基线
  ledger:186 的 “统一 881 日合并 RankIC” 是这个口径。
两列都填时才可横向比；只填其一的行不得与另一口径的行并列成一句结论。

路径一律显式拼装到**本实验自己的产物目录**与折 36–42 的既有目录，
不对 outputs/ 做任何通配。
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

from crsp_pipeline.signal_eval import newey_west_tstat  # noqa: E402

NW_LAGS = 5
UNCHECKED = "未核"
ORACLE_TAG = "不可用于判定"

COLUMNS = [
    "方法", "折号", "折数", "天数", "内层选参(是/否)", "scoring_config",
    "全池 RankIC", "全池 RankIC（合并日）", "NW(5) t", "正折数",
    f"事后最优(oracle) RankIC【{ORACLE_TAG}】", "来源", "备注",
]

# --------------------------------------------------------------- 抄自登记簿
# 逐字抄 experiments/ledger.md，**不重算**。登记簿没写的字段一律 UNCHECKED。
LEDGER_ROWS = [
    {
        "方法": "线性探针（内层选参）", "折号": "36–40", "折数": 5,
        "天数": UNCHECKED, "内层选参(是/否)": "是（内层选 alpha）",
        "scoring_config": "mean 池化、缓存表示；其余字段登记簿未给",
        "全池 RankIC": 0.01256, "全池 RankIC（合并日）": UNCHECKED,
        "NW(5) t": UNCHECKED, "正折数": "4/5",
        "oracle": None,
        "来源": "抄自登记簿 ledger.md:164",
        "备注": "五折读数，不得与七折行并列成一句结论",
    },
    {
        "方法": "线性探针（事后最优 oracle）", "折号": "36–40", "折数": 5,
        "天数": UNCHECKED, "内层选参(是/否)": "否（在外层验证窗上取最优 alpha）",
        "scoring_config": "mean 池化、缓存表示；其余字段登记簿未给",
        "全池 RankIC": UNCHECKED, "全池 RankIC（合并日）": UNCHECKED,
        "NW(5) t": UNCHECKED, "正折数": UNCHECKED,
        "oracle": 0.01951,
        "来源": "抄自登记簿 ledger.md:164",
        "备注": f"上界，{ORACLE_TAG}；与内层选参行之差 +0.0070 即选参偏差",
    },
    {
        "方法": "生成式 零样本(ZS)", "折号": "36–40", "折数": 5,
        "天数": UNCHECKED, "内层选参(是/否)": "否（零样本，无可选参数）",
        "scoring_config": "登记簿该行未给（同折 FT 行为 bf16/bs128/sc5/lb90/predict6）",
        "全池 RankIC": 0.02193, "全池 RankIC（合并日）": UNCHECKED,
        "NW(5) t": UNCHECKED, "正折数": UNCHECKED,
        "oracle": None,
        "来源": "抄自登记簿 ledger.md:164",
        "备注": "五折读数，不得与七折行并列成一句结论",
    },
    {
        "方法": "生成式 微调(FT)", "折号": "36–40", "折数": 5,
        "天数": UNCHECKED, "内层选参(是/否)": "是（内层早停）",
        "scoring_config": "登记簿该行未给（同折七折行记为 bf16/bs128/sc5/lb90）",
        "全池 RankIC": 0.02544, "全池 RankIC（合并日）": UNCHECKED,
        "NW(5) t": UNCHECKED, "正折数": UNCHECKED,
        "oracle": None,
        "来源": "抄自登记簿 ledger.md:164",
        "备注": "五折读数，不得与七折行并列成一句结论",
    },
    {
        "方法": "生成式 零样本(ZS)", "折号": "36–42", "折数": 7,
        "天数": UNCHECKED, "内层选参(是/否)": "否（零样本，无可选参数）",
        "scoring_config": "bf16 / bs128 / sample_count=5 / lb90（ledger:162 原文「同折同口径 bf16/bs128/lb90」）",
        "全池 RankIC": 0.01919, "全池 RankIC（合并日）": UNCHECKED,
        "NW(5) t": UNCHECKED, "正折数": "7/7（据 ledger:162 逐折数字全为正）",
        "oracle": None,
        "来源": "抄自登记簿 ledger.md:162",
        "备注": "逐折 36→42：+0.02652/+0.02313/+0.01067/+0.03432/+0.01499/+0.00684/+0.01789",
    },
    {
        "方法": "生成式 微调(FT)", "折号": "36–42", "折数": 7,
        "天数": UNCHECKED, "内层选参(是/否)": "是（内层早停）",
        "scoring_config": "bf16 / bs128 / sample_count=5 / lb90（ledger:162 原文）",
        "全池 RankIC": 0.02071, "全池 RankIC（合并日）": UNCHECKED,
        "NW(5) t": UNCHECKED,
        "正折数": "7/7（IC 为正的折数，据 ledger:162 逐折数字）；FT 优于 ZS 仅 4/7 折",
        "oracle": None,
        "来源": "抄自登记簿 ledger.md:162",
        "备注": "逐折 36→42：+0.02549/+0.02408/+0.00803/+0.04308/+0.02650/+0.01053/+0.00730",
    },
    {
        "方法": "最佳树基线 XGBoost", "折号": "36–42", "折数": 7,
        "天数": 881, "内层选参(是/否)": "是（内层选参，冻结配置 configs/gbdt_strong_v2.json）",
        "scoring_config": "登记簿未按 scoring_config 字段记录（树模型无该字段）",
        "全池 RankIC": 0.006373, "全池 RankIC（合并日）": 0.006280,
        "NW(5) t": 0.688, "正折数": "4/7",
        "oracle": None,
        "来源": "抄自登记簿 ledger.md:186",
        "备注": "折等权 +0.006373、合并 881 日 +0.006280；三种子均值 +0.005806±0.002667",
    },
    {
        "方法": "MLP 排序头", "折号": "fold40（单折）", "折数": 1,
        "天数": UNCHECKED, "内层选参(是/否)": "是（内层早停选轮次）",
        "scoring_config": "登记簿该行未给（见本表同一次运行的复用行）",
        "全池 RankIC": 0.00753, "全池 RankIC（合并日）": 0.00753,
        "NW(5) t": UNCHECKED, "正折数": "1/1",
        "oracle": None,
        "来源": "抄自登记簿 ledger.md:141",
        "备注": "单折读数（CLAUDE.md §三：少于七折的结论不算数），与本表复用行为同一次运行",
    },
]


# --------------------------------------------------------------- 读产物
def read_daily_ic(eval_dir: Path) -> pd.Series:
    """读 evaluate/排序头 产物里的逐日 RankIC 序列（由 daily_rank_ic 生成）。"""
    df = pd.read_parquet(eval_dir / "daily_ic.parquet")
    col = "rank_ic" if "rank_ic" in df.columns else df.columns[0]
    s = pd.Series(df[col].to_numpy(dtype=float))
    return s.dropna()


def collect_mlp(fold_dirs: dict[str, Path]) -> dict:
    per_fold, series, configs = {}, [], {}
    for fold, d in sorted(fold_dirs.items()):
        metrics = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        s = read_daily_ic(d)
        nw = newey_west_tstat(s, NW_LAGS)
        recorded = metrics.get("nw_raw", {})
        per_fold[fold] = {
            "eval_dir": str(d).replace("\\", "/"),
            "val_window": metrics.get("val_window"),
            "n_days": int(nw["n"]),
            "n_obs": metrics.get("n_obs"),
            "rank_ic": float(nw["mean"]),
            "t": float(nw["t"]),
            "recorded_rank_ic": recorded.get("mean"),
            "recorded_t": recorded.get("t"),
            "best_epoch": metrics.get("head_training", {}).get("best_epoch"),
            "best_inner_rank_ic": metrics.get("head_training", {}).get("best_val_rank_ic"),
        }
        configs[fold] = metrics.get("scoring_config")
        series.append(s)
    return _summarise(per_fold, series, configs)


def collect_ridge(probe_json: Path, daily_ic_dir: Path | None,
                  fold_names: list[str]) -> dict:
    raw = json.loads(probe_json.read_text(encoding="utf-8"))
    per_fold, series = {}, []
    for fold in fold_names:
        if fold not in raw:
            continue
        r = raw[fold]
        per_fold[fold] = {
            "alpha": r["alpha"], "rank_ic": float(r["rank_ic"]), "t": float(r["t"]),
            "n_days": int(r["n_days"]), "pool": r["pool"],
            "inner_rank_ic": r.get("inner_rank_ic"),
            "oracle_rank_ic": r.get("oracle_rank_ic"),
        }
        if daily_ic_dir is not None:
            f = daily_ic_dir / f"{fold}.parquet"
            if f.exists():
                series.append(pd.read_parquet(f)["rank_ic"].astype(float).dropna())
    out = _summarise(per_fold, series, {f: {"pool": per_fold[f]["pool"]}
                                        for f in per_fold})
    orc = [per_fold[f]["oracle_rank_ic"] for f in per_fold
           if per_fold[f]["oracle_rank_ic"] is not None]
    out["mean_oracle_rank_ic"] = float(np.mean(orc)) if orc else None
    return out


def _summarise(per_fold: dict, series: list[pd.Series], configs: dict) -> dict:
    vals = [v["rank_ic"] for v in per_fold.values()]
    days = [v["n_days"] for v in per_fold.values()]
    pooled = None
    if series:
        cat = pd.concat(series, ignore_index=True).dropna()
        pooled = newey_west_tstat(cat, NW_LAGS)
    uniq = {json.dumps(c, sort_keys=True, ensure_ascii=False) for c in configs.values()}
    return {
        "per_fold": per_fold,
        "n_folds": len(vals),
        "n_days_total": int(sum(days)),
        "mean_rank_ic_equal_fold": float(np.mean(vals)) if vals else None,
        "n_positive": int(sum(v > 0 for v in vals)),
        "pooled": pooled,
        "scoring_config": json.loads(next(iter(uniq))) if len(uniq) == 1 else None,
        "scoring_config_all_equal": len(uniq) == 1,
        "scoring_config_variants": sorted(uniq) if len(uniq) != 1 else None,
    }


# --------------------------------------------------------------- 出表
def _fmt(v, digits: int = 5) -> str:
    if v is None:
        return UNCHECKED
    if isinstance(v, str):
        return v
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{float(v):+.{digits}f}"


def _oracle_cell(v) -> str:
    if v is None or isinstance(v, str):
        return "不适用" if v is None else v
    return f"{float(v):+.5f}（{ORACLE_TAG}）"


def build_rows(mlp: dict, ridge: dict, ridge_cmd: str, folds_label: str) -> list[dict]:
    rows: list[dict] = []
    if mlp["n_folds"]:
        pooled = mlp["pooled"] or {}
        rows.append({
            "方法": "MLP 排序头（冻结主干 + 头）", "折号": folds_label,
            "折数": mlp["n_folds"], "天数": mlp["n_days_total"],
            "内层选参(是/否)": "是（内层验证窗早停选轮次；外层验证窗未参与）",
            "scoring_config": json.dumps(mlp["scoring_config"], ensure_ascii=False)
            if mlp["scoring_config_all_equal"] else "各折不一致！" ,
            "全池 RankIC": mlp["mean_rank_ic_equal_fold"],
            "全池 RankIC（合并日）": pooled.get("mean"),
            "NW(5) t": pooled.get("t"),
            "正折数": f"{mlp['n_positive']}/{mlp['n_folds']}",
            "oracle": None,
            "来源": "本次新跑（fold40 复用同参数同窗口的既有产物）",
            "备注": "NW t 为合并日口径；折等权均值与合并日均值天数不同，不得并列成一句结论",
        })
    if ridge["n_folds"]:
        pooled = ridge["pooled"] or {}
        rows.append({
            "方法": "线性探针（岭回归，内层选 alpha）", "折号": folds_label,
            "折数": ridge["n_folds"], "天数": ridge["n_days_total"],
            "内层选参(是/否)": "是（训练窗尾部 6 个月内层窗选 alpha，purge 7 日）",
            "scoring_config": ridge_cmd,
            "全池 RankIC": ridge["mean_rank_ic_equal_fold"],
            "全池 RankIC（合并日）": pooled.get("mean"),
            "NW(5) t": pooled.get("t"),
            "正折数": f"{ridge['n_positive']}/{ridge['n_folds']}",
            "oracle": ridge.get("mean_oracle_rank_ic"),
            "来源": "本次新跑（折 36–40 用既有表示缓存重算，折 41–42 新抽表示）",
            "备注": "逐日 IC 口径与 MLP 行略异：探针按 argsort 秩且丢弃当日 <50 只的交易日",
        })
    rows.extend(LEDGER_ROWS)
    return rows


def rows_to_markdown(rows: list[dict], header_note: str) -> str:
    out = [header_note, "", "| " + " | ".join(COLUMNS) + " |",
           "|" + "|".join(["---"] * len(COLUMNS)) + "|"]
    for r in rows:
        cells = []
        for c in COLUMNS:
            if c.startswith("事后最优"):
                cells.append(_oracle_cell(r.get("oracle")))
            elif c in ("全池 RankIC", "全池 RankIC（合并日）", "NW(5) t"):
                v = r.get(c)
                cells.append(_fmt(v, 2 if c == "NW(5) t" else 5))
            else:
                v = r.get(c, UNCHECKED)
                cells.append(str(v) if v is not None else UNCHECKED)
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def per_fold_markdown(mlp: dict, ridge: dict) -> str:
    """附：本次新跑两行的逐折明细（只作透明度，不改上表任何口径）。"""
    out = ["## 附：本次新跑两行的逐折明细（透明度用，不作并列比较）", ""]
    if mlp["n_folds"]:
        out += ["### MLP 排序头（冻结主干 + 头）", "",
                "| 折 | 验证窗 | 天数 | RankIC | NW(5) t | 内层早停 epoch | 产物 |",
                "|---|---|---|---|---|---|---|"]
        for f, v in mlp["per_fold"].items():
            w = v["val_window"]
            out.append(f"| {f} | {w[0]}..{w[1]} | {v['n_days']} | {v['rank_ic']:+.5f} | "
                       f"{v['t']:+.2f} | {v['best_epoch']} | `{v['eval_dir']}` |")
        out.append("")
    if ridge["n_folds"]:
        out += ["### 线性探针（岭回归，内层选 alpha）", "",
                f"| 折 | 天数 | 内层选定 alpha | RankIC | NW(5) t | "
                f"事后最优 oracle【{ORACLE_TAG}】 |",
                "|---|---|---|---|---|---|"]
        for f, v in ridge["per_fold"].items():
            out.append(f"| {f} | {v['n_days']} | {v['alpha']:.0e} | {v['rank_ic']:+.5f} | "
                       f"{v['t']:+.2f} | {v['oracle_rank_ic']:+.5f}（{ORACLE_TAG}） |")
        out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="实验 8 读出对比表")
    ap.add_argument("--folds-json", default="outputs/exp8_folds_36_42.json")
    ap.add_argument("--mlp-root", default="outputs")
    ap.add_argument("--mlp-fold40-dir",
                    default="outputs/rankhead_fold40/eval_rh_zeroshot_fold40",
                    help="fold40 复用的既有产物（同参数、同窗口）")
    ap.add_argument("--ridge-json", default="outputs/exp8_ridge_probe_7fold.json")
    ap.add_argument("--ridge-daily-ic-dir", default="outputs/exp8_ridge_daily_ic")
    ap.add_argument("--ridge-cmd-file", default="outputs/exp8_ridge_probe_cmd.txt")
    ap.add_argument("--out-json", default="outputs/exp8_readout_table.json")
    ap.add_argument("--out-md", default="outputs/exp8_readout_table.md")
    args = ap.parse_args()

    payload = json.loads(Path(args.folds_json).read_text(encoding="utf-8"))
    fold_names = [f["n"] for f in payload["folds"]]
    fold_dirs = {}
    for name in fold_names:
        d = (Path(args.mlp_fold40_dir) if name == "fold40"
             else Path(args.mlp_root) / f"exp8_mlp_{name}" / f"eval_exp8_mlp_{name}")
        if (d / "metrics.json").exists():
            fold_dirs[name] = d

    mlp = collect_mlp(fold_dirs)
    ridge_dir = Path(args.ridge_daily_ic_dir)
    ridge = (collect_ridge(Path(args.ridge_json),
                           ridge_dir if ridge_dir.exists() else None, fold_names)
             if Path(args.ridge_json).exists()
             else {"n_folds": 0, "per_fold": {}})
    cmd_file = Path(args.ridge_cmd_file)
    ridge_cmd = (" ".join(cmd_file.read_text(encoding="utf-8").split())
                 if cmd_file.exists() else UNCHECKED)

    label = f"{fold_names[0]}–{fold_names[-1]}" if fold_names else UNCHECKED
    rows = build_rows(mlp, ridge, ridge_cmd, label)
    note = (
        f"# 实验 8：读出方案对比表（{label}，同口径）\n\n"
        f"- 生成时间（UTC）：{datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
        "- **用途限制**：估计交付，**不用于选读出**（读出方案属 B 层，按先验定，"
        "见 CLAUDE.md §二）。\n"
        f"- `全池 RankIC` = **折等权**均值；`全池 RankIC（合并日）` = 各折逐日序列拼接后"
        f"按天等权；`NW(5) t` = `newey_west_tstat(lags={NW_LAGS})` 打在**合并日**序列上。\n"
        "- **折数或天数不同的两行不得并列成一句结论**（CLAUDE.md §三、§八）。\n"
        f"- `事后最优(oracle)` 列每一格都标注「{ORACLE_TAG}」。\n"
        f"- 「{UNCHECKED}」= 登记簿 / 产物里没有该数字，本次也没算。\n"
    )
    md = rows_to_markdown(rows, note) + "\n" + per_fold_markdown(mlp, ridge)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(md, encoding="utf-8")
    Path(args.out_json).write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": "实验 8：把读出方案对比表的折数口径补齐到七折 36–42（估计交付，不用于选读出）",
        "fold_table": payload,
        "columns": COLUMNS,
        "rows": rows,
        "mlp": mlp,
        "ridge": ridge,
        "ridge_cmd": ridge_cmd,
    }, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(rows_to_markdown(rows, note))


if __name__ == "__main__":
    main()
