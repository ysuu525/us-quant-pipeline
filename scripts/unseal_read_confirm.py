"""确认集解封读取的**唯一入口**（折 05–35 + fold44–45）。

    .venv\\Scripts\\python.exe scripts\\unseal_read_confirm.py --smoke --out <目录>
    .venv\\Scripts\\python.exe scripts\\unseal_read_confirm.py ^
        --out outputs\\unseal_<UTC> --clean-window --authorised-by-user ledger:<行号>

依据：**冻结版** `experiments/confirmation_protocol_v4.md`
（tag ``prereg-2026-09-05-v4``；其 sha256 在每次运行时现算并写进 ``run_meta.json``
与报告页头，文件一改哈希即变）、`docs/解封读取_任务书_2026-09-05.md`、`CLAUDE.md`。
**冻结版与任务书冲突时一律以冻结版为准。** 冻结版不存在即拒跑。

硬禁令（违反即全部读数作废）
============================
1. **禁止自动 ``append_ledger``**。本脚本与 `src/unseal/*` 一律不 import、
   不调用它——`evaluate_fold.py:356` 的自动追加会把逐折 IC 写进 append-only 登记簿，
   越过 v4 §3 释放清单，并把实验 6 的试验数 N 从 148 推高、连带改动自己要兑现的
   预测区间。**登记条目由主会话按报告手写。**
2. **不得计算、不得输出 FT−ZS 差值**（v4 §3）。两臂各自的量可报，差值与其显著性
   不算不看——该比较已由 `CLAUDE.md` §二定为 B 层（正交臂淘汰门槛 1.10–1.51×
   信号本身，永不触发）。
3. **释放清单严格限于 v4 §3**；报告里**只有合并量 + 正折数 + 逐折符号计数**，
   不得出现逐折 IC 表。
4. **fold44–45 是独立交付**（干净窗），同口径、同脚本、**独立输出目录，
   不并入 31 折合并统计**。
5. **一次运行**，唯一输出目录，含配置哈希、代码快照 sha256、工作树 porcelain。
6. 措辞按 `CLAUDE.md` §二：B 层与估计交付不写「不可分 / 无效」，
   写「在本样本量下该问题不可回答」。

判据（先于结果落笔；正本见 `src/unseal/config.py` 的模块 docstring）
====================================================================
* **H1**：全池逐日 RankIC 的区间估计，**不做二元 PASS**。点估计 + NW(5) 95% CI +
  正折数（/31）+ 实现收缩比与主预测区间（FT ``[0.0088812, 0.0182250]``、
  ZS ``[0.0080529, 0.0170925]``）的相交关系；敏感性预测 ``[0.0149, 0.0166]``
  只披露。**H4 读取闸门**（该臂 H1 的 95% CI 下界 > 0）只记录、不触发任何动作。
* **H2**：三门 —— CI 下界 > 0（z = 2.2414）∧ 正折数 ≥ 18/31 ∧
  ``mean RankIC(top500) > 0``；**没有量级门**。三门未全过不得写成「机制不存在」。
* **H2-era**：切点写死 **2013-01-01**（按 **val_start** 归段），**只切一次**，
  估计交付、无门槛。早期段 fold05–20（16 折）、晚期段 fold21–35（15 折）；
  fold20 末端越切点 2 个交易日仍归早期段，报告须写明。交互项 MDE80 原样引用
  v4 §2.5.1（ΔADV FT 0.00730 / ZS 0.00789；top500 IC FT 0.01923 / ZS 0.01955）。
* **H1b**：v4 §2.4 **裁定 C-1 已采纳**，估计交付，**默认开启**（``--no-h1b`` 关）；
  **只在 FT 臂交付**，主规格先验固定 S-TH-ind、S-T / S-H 次规格，
  口径沿 K6b 的 **NT=6**，与 NT=5 读数不并列；无门槛、无 SESOI、不作功效关判定。
* **H6**：ICT P1 探索性估计交付，只跑 ``P1_bull`` / ``P1_bear``，只算 Q1 与 Q2(b)，
  NW lag = 6，Q2(b) 五分位用 **FT 封存分数（sc=5）**；不进 H1–H4 判定、不作部署依据。
* **E**：估计交付。毛年化 + NW(5) CI + 正折数、单边换手、``BE ± 95% CI``、
  成本网格 2/4/8/12/16/22bp；部署线 ``C ≤ BE_dev × (1 − h) − 6bp`` **只报不判**
  （**h = 0.42 主判** → FT ``C ≤ 7.28bp``、ZS ``C ≤ 3.51bp``；敏感性
  h ∈ [0.399, 0.440]；旧线 ``×0.75`` **已作废，只作一行对照披露**）。
* **读取前剔除（v4.1 附录 §5）**：表 B 的 ``fold35``（止于 2020-07-02）与表 A 的
  开发 ``fold36``（起于 2020-07-01）重叠 **2 个交易日**（2020-07-01、2020-07-02），
  这两个信号日在开发阶段已被读过。读取时从 **fold35** 的一切统计中剔除
  （FT 与树基线两侧，且**在标签计算之前**），使折 05–35 严格为「未消耗」；
  剔除行数写进 ``run_meta.json`` / ``summary.json`` 与报告页头。

* **H3 不在本次范围**（树基线需封存窗内标签作训练目标，超出 2026-09-01 计算授权）；
  **H4 空置**（信号 #2 未过准入，`ledger:477`）。

用途限制
========
一次性、不可重开：31 折读后即永久视为已消耗。读数不得回头修改任何预注册区间、
门槛、构造参数或配置选择。fold44–45 只作洁净复核与效应估计，**不承担最终二元裁决**。

释放清单
========
E / H1 / H2 / H2-era / H6（+ 取 C-1 时的 H1b）各自在 v4 §3 列明的量，两臂各一份。
**运行后禁止查看或生成**：逐年曲线、ADV 五档、最优年代段、单股贡献、替代 universe、
替代阈值、任何臂间比较、K8 合成的追加比较、清单以外的一切量。

口径核对（`CLAUDE.md` §八）
===========================
每折读分数前核对 ``SEALED_MANIFEST.json`` 的 ``scores_sha256`` 与 config 五项
（lookback 90 / predict 6 / sample_count 5 / amp bf16 / batch_size 128）；
**任何一项失配即中止整个运行**。折窗口一律由
``crsp_pipeline.splits.walk_forward_folds`` 机械生成，**不得手写**，
并与清单的 ``val_window`` 逐折对账。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from unseal import aggregate, config as C, h1b as H1B, h3 as H3, h6 as H6, paths as UP  # noqa: E402
from unseal.folds import (  # noqa: E402
    EXCLUDED_SIGNAL_DATES, EXCLUDED_SIGNAL_DATES_FOLD,
    fold_windows, load_calendar, parse_folds,
)
from unseal.perfold import run_fold  # noqa: E402
from unseal.report import render_report  # noqa: E402

DEFAULT_PROCESSED = Path(r"F:\quant\processed\crsp_ciz_2026-08-24_20260825T130601Z")
DEFAULT_JKP = Path(r"F:\quant\external\jkp")

#: 参与代码快照哈希的文件（相对仓库根）。
CODE_SNAPSHOT_FILES = (
    "scripts/unseal_read_confirm.py",
    "scripts/exp11_spanning_extended.py",
    "scripts/ict_pattern_probe.py",
    "src/unseal/__init__.py", "src/unseal/config.py", "src/unseal/paths.py",
    "src/unseal/folds.py", "src/unseal/perfold.py", "src/unseal/aggregate.py",
    "src/unseal/h1b.py", "src/unseal/h3.py", "src/unseal/h6.py",
    "src/unseal/report.py",
    "src/unseal/smoke.py",
    "src/crsp_pipeline/sealed.py", "src/crsp_pipeline/labels.py",
    "src/crsp_pipeline/signal_eval.py", "src/crsp_pipeline/splits.py",
    "src/portfolio/construction.py", "src/backtest/money.py",
    "src/signals/kronos_adapter.py",
)


#: ``--clean-window-only`` 时写进 ``run_meta.json`` 的口径说明。
CONFIRM_NOT_READ_NOTE = (
    "确认段（折 05–35）未在本次运行中读取：本次只跑干净窗 fold44–45，"
    "未打开确认集任何一折的封存分数或标签，确认段仍为「未消耗」。")
#: ``--no-h3`` 时干净窗报告页头的强制披露（v4 §6）。
CLEAN_WINDOW_NO_H3_NOTE = (
    "H3 未在干净窗交付（v4 §6 干净窗无 H3 终点；树基线封存仅覆盖折 05–35）")


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while blk := f.read(1 << 20):
            h.update(blk)
    return h.hexdigest()


def frozen_config_snapshot() -> dict:
    """冻结常量的机器可读快照（配置哈希的原料）。"""
    return {
        "confirm_folds": list(C.CONFIRM_FOLDS),
        "clean_window_folds": list(C.CLEAN_WINDOW_FOLDS),
        "arms_in_read_order": list(C.ARMS),
        "arms_default": list(C.ARMS_DEFAULT),
        "required_scoring_config": C.REQUIRED_SCORING_CONFIG,
        "construction": {"nt": C.NT, "topn": C.TOPN, "exit_pct": C.EXIT_PCT,
                         "min_names": C.MIN_NAMES, "entry_pct": 0.10,
                         "predict_window": C.PREDICT_WINDOW,
                         "execution": "t close score -> t+1 open"},
        "inference": {"nw_lag": C.NW_LAG, "h6_nw_lag": C.H6_NW_LAG,
                      "z95": C.Z95, "z_bonferroni_k2": C.Z_BONFERRONI_K2,
                      "power_mult_80": C.POWER_MULT_80,
                      "h2_positive_folds_gate": C.H2_POSITIVE_FOLDS_GATE},
        "h1_dev_base": C.H1_DEV_BASE,
        "h1_main_prediction": {k: list(v) for k, v in C.H1_MAIN_PREDICTION.items()},
        "h1_sensitivity_prediction": list(C.H1_SENSITIVITY_PREDICTION),
        "excluded_signal_dates": {
            "dates": sorted(EXCLUDED_SIGNAL_DATES),
            "fold": EXCLUDED_SIGNAL_DATES_FOLD,
            "rule": ("v4.1 附录 §5：表 B 的 fold35（止于 2020-07-02）与表 A 的开发 "
                     "fold36（起于 2020-07-01）重叠 2 个交易日，这两个信号日在开发阶段"
                     "已被读过；读取时从 fold35 的一切统计中剔除（FT 与树基线两侧），"
                     "且在标签计算之前剔除，使折 05–35 严格为「未消耗」"),
        },
        "era": {"cut": C.ERA_CUT,
                "early_folds": list(C.ERA_EARLY_FOLDS),
                "late_folds": list(C.ERA_LATE_FOLDS),
                "mde80_interaction": C.H2_ERA_MDE80_INTERACTION},
        "cost_grid_bp": list(C.COST_GRID_BP),
        "be_dev_bp": C.BE_DEV_BP,
        "deployment": {"h_main": C.DEPLOY_HAIRCUT_MAIN,
                       "h_sensitivity": list(C.DEPLOY_HAIRCUT_RANGE),
                       "h_legacy_deprecated": C.DEPLOY_HAIRCUT_LEGACY,
                       "reserve_bp": C.DEPLOY_RESERVE_BP,
                       "c_stop_bp": C.C_STOP_BP},
        "h1b": {"default_enabled": C.H1B_DEFAULT_ENABLED,
                "primary_spec": C.H1B_PRIMARY_SPEC, "specs": list(C.H1B_SPECS),
                "arms": list(C.H1B_ARMS), "nt": C.H1B_NT},
        "h3": {"arm": C.H3_ARM, "tree_dir_template": C.H3_TREE_DIR_TEMPLATE,
               "mde80_dev7": C.H3_MDE80_DEV7, "mde80_31fold": C.H3_MDE80_31FOLD,
               "sesoi_candidates": list(C.H3_SESOI_CANDIDATES)},
        "h6": {"markers": list(C.H6_MARKERS), "questions": list(C.H6_QUESTIONS),
               "sesoi_bp": C.H6_SESOI_BP},
    }


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                              text=True, timeout=60).stdout.strip()
    except Exception as exc:                      # pragma: no cover - 环境无 git
        return f"<git unavailable: {exc}>"


def build_run_meta(out: Path, args: argparse.Namespace) -> dict:
    cfg = frozen_config_snapshot()
    cfg_json = json.dumps(cfg, ensure_ascii=False, sort_keys=True)
    code = {rel: _sha256_path(REPO_ROOT / rel) for rel in CODE_SNAPSHOT_FILES
            if (REPO_ROOT / rel).is_file()}
    porcelain = _git("status", "--porcelain")
    (out / "git_porcelain.txt").write_text(porcelain + "\n", encoding="utf-8")
    protocol = C.PROTOCOL_PATH
    if not protocol.is_file():
        raise SystemExit(
            f"拒绝运行：找不到冻结协议 {protocol}。解封必须依据冻结版 v4，"
            f"不得依据草案（v4 §7 第 7–9 条）。")
    addendum = C.ADDENDUM_PATH
    meta = {
        "protocol_path": protocol.relative_to(REPO_ROOT).as_posix(),
        "protocol_sha256": _sha256_path(protocol),
        "protocol_tag": C.PROTOCOL_TAG,
        "addendum_path": (addendum.relative_to(REPO_ROOT).as_posix()
                          if addendum.is_file() else None),
        "addendum_sha256": _sha256_path(addendum) if addendum.is_file() else None,
        "run_id": out.name,
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "smoke" if args.smoke else "real",
        "git_commit": _git("rev-parse", "HEAD"),
        "git_porcelain_lines": len([x for x in porcelain.splitlines() if x.strip()]),
        "config_sha256": _sha256_text(cfg_json),
        "frozen_config": cfg,
        "code_sha256": code,
        "code_sha256_of_snapshot": _sha256_text(
            json.dumps(code, ensure_ascii=False, sort_keys=True)),
        "python": sys.version,
        "argv": sys.argv[1:],
    }
    return meta


def _forbidden_keys(obj, prefix: str = "") -> list[str]:
    """机器可核对的 FT−ZS 差值禁令：任何键名不得含差值片段。"""
    bad: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = str(k)
            low = key.lower()
            if any(frag in low for frag in C.FORBIDDEN_KEY_FRAGMENTS):
                bad.append(f"{prefix}{key}")
            bad += _forbidden_keys(v, f"{prefix}{key}.")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            bad += _forbidden_keys(v, f"{prefix}[{i}].")
    return bad


def run_scope(*, scope_name: str, folds: tuple[int, ...], out: Path,
              processed: Path, outputs_root: Path, jkp: Path,
              run_meta: dict, args: argparse.Namespace,
              arms: tuple[str, ...] = C.ARMS) -> dict:
    """跑一个交付范围（31 折确认集，或独立的 fold44–45 干净窗）。

    ``arms`` 的顺序**先验固定为 FT → ZS**（v4 §1 minimax-regret），调用方不得重排。
    """
    out.mkdir(parents=True, exist_ok=True)
    cal = load_calendar(processed)
    windows = fold_windows(cal, folds)               # 机械生成，不得手写
    log(f"[{scope_name}] 折窗口机械生成完毕：{len(windows)} 折 "
        f"[{min(w.val_start for w in windows.values()).date()} .. "
        f"{max(w.val_end for w in windows.values()).date()}]")

    # ---- 阶段 0：**先把全部折 × 臂的口径核对做完**，一处失配即中止整个运行
    verified = []
    for arm in arms:
        for f in folds:
            verified.append(UP.verify_fold(f, arm, outputs_root,
                                           (windows[f].val_start, windows[f].val_end)))
    log(f"[{scope_name}] 口径核对通过：{len(verified)} 个（折 × 臂）")
    (out / "caliber_check.json").write_text(
        json.dumps(verified, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 阶段 1：逐折标签 + 指标 + 逐日序列（读取顺序固定 FT → ZS）
    per_fold = []
    for arm in arms:
        for f in folds:
            t0 = time.perf_counter()
            info = run_fold(processed, outputs_root, windows[f], arm, out, cal=cal)
            per_fold.append(info)
            log(f"[{scope_name}] {arm} 折 {f:02d}: 观测 {info['n_obs']:,}、"
                f"有效日 {info['n_days']}、钱日 {info['n_money_days']}、"
                f"{time.perf_counter() - t0:.1f}s")

    # ---- 阶段 2：合并统计（两臂各一份，**不算任何差值**）
    n_excluded = sum(int(x.get("n_rows_excluded", 0)) for x in per_fold)
    log(f"[{scope_name}] v4.1 §5 剔除：{n_excluded} 行"
        f"（信号日 {sorted(EXCLUDED_SIGNAL_DATES)}，只作用于 "
        f"fold{EXCLUDED_SIGNAL_DATES_FOLD:02d}）")
    payload: dict = {"run_meta": run_meta | {"n_verified": len(verified),
                                             "n_rows_excluded": n_excluded},
                     "scope": scope_name, "folds_count": len(folds),
                     "arms": list(arms),
                     "excluded_signal_dates": sorted(EXCLUDED_SIGNAL_DATES),
                     "excluded_signal_dates_fold": EXCLUDED_SIGNAL_DATES_FOLD,
                     "n_rows_excluded": n_excluded,
                     "h1": {}, "h2": {}, "h2_era": {}, "e": {}}
    for arm in arms:
        ic = aggregate.load_daily(out, arm, folds, "daily_ic")
        money = aggregate.load_daily(out, arm, folds, "daily_money")
        payload["h1"][arm] = aggregate.h1_summary(ic, arm)
        payload["h2"][arm] = aggregate.h2_summary(ic, arm)
        payload["h2_era"][arm] = aggregate.h2_era_summary(ic, windows, arm)
        payload["e"][arm] = aggregate.e_summary(money, arm)

    # ---- 阶段 3a：H3（Kronos FT vs 冻结树基线的配对 ΔIC；B 层估计交付）
    if args.h3:
        log(f"[{scope_name}] H3：Kronos {C.H3_ARM.upper()} vs 冻结树基线 XGBoost 的配对 ΔIC...")
        payload["h3"] = H3.run_h3(outputs_root, out, list(windows.values()),
                                  arm=C.H3_ARM)

    # ---- 阶段 3b：H6（ICT P1，探索性估计交付；分数用 FT 臂）
    if args.h6:
        log(f"[{scope_name}] H6：ICT P1（Q1 + Q2b，NW lag 6）...")
        payload["h6"] = H6.run_h6(processed, out / "ft", list(windows.values()),
                                  out / "h6_ict_p1", force=True,
                                  per_block=args.h6_block_folds)

    # ---- 阶段 4：H1b（默认不跑；v4 [待裁定 C] 未裁定期间取 C-2）
    if args.h1b:
        log(f"[{scope_name}] H1b：扩充控制集张成（K6b 口径 NT=6）...")
        payload["h1b"] = H1B.run_h1b(
            processed, jkp, out / "ft", list(windows.values()),
            out / "h1b_spanning.json", outputs_root=outputs_root,
            memory_limit_gb=args.memory_limit_gb,
            allow_alt_processed=args.smoke)

    # ---- 释放闸门：任何差值键都不许出现
    bad = _forbidden_keys(payload)
    if bad:
        raise RuntimeError(f"输出里出现被禁止的臂间差值键：{bad}（v4 §3）")

    (out / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")
    (out / "per_fold_index.json").write_text(
        json.dumps(per_fold, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="确认集解封读取（唯一入口）", formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", required=True, help="唯一输出目录，如 outputs/unseal_<UTC>")
    ap.add_argument("--smoke", action="store_true",
                    help="合成假数据全链路（v4 §7 第 3 条）；不碰任何真实封存目录")
    ap.add_argument("--folds", default=None,
                    help="折号，缺省 05-35（确认集）；如 5-35 或 5,6,7")
    ap.add_argument("--clean-window", action="store_true",
                    help="另跑 fold44–45（干净窗），**独立输出目录、独立报告，"
                         "不并入 31 折合并统计**")
    ap.add_argument("--clean-window-only", action="store_true",
                    help="**只**跑干净窗 fold44–45，跳过确认段（折 05–35）：隐含 "
                         "--clean-window，不调用确认段 run_scope、不生成 report.md。"
                         "与 --folds 互斥（确认段的折号在本模式下无意义）。"
                         "--authorised-by-user 闸门照旧生效。")
    ap.add_argument("--processed", type=Path, default=DEFAULT_PROCESSED)
    ap.add_argument("--outputs-root", type=Path, default=REPO_ROOT / "outputs")
    ap.add_argument("--jkp", type=Path, default=DEFAULT_JKP)
    ap.add_argument("--no-h1b", dest="h1b", action="store_false",
                    default=True,
                    help="跳过 H1b（v4 §2.4 裁定 C-1 已采纳，默认交付；只在 FT 臂）")
    ap.add_argument("--h1b", dest="h1b", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("--no-h3", dest="h3", action="store_false", default=True,
                    help="跳过 H3（默认交付；树基线队列未跑完该折会直接报错中止）")
    ap.add_argument("--no-h6", dest="h6", action="store_false", default=True,
                    help="跳过 H6（默认跑；H6 必须与 H1–H4 同一时刻读完）")
    ap.add_argument("--h6-block-folds", type=int, default=4,
                    help="H6 面板块内的折数（纯工程参数，不改任何数值口径）")
    ap.add_argument("--arms", default=",".join(C.ARMS_DEFAULT),
                    help="读取哪些臂，顺序先验固定 FT → ZS。**缺省 `ft`**——v4.1 附录"
                         "（用户裁定抛弃 ZS 臂，冻结后 / 读取前的设计变更）；ZS 分支代码"
                         "保留，传 `--arms ft,zs` 可恢复 v4 §1 的 k=2 双臂。"
                         "**z 维持 2.2414 不放宽。**")
    ap.add_argument("--memory-limit-gb", type=float, default=40.0,
                    help="H1b 的提交内存闸门（CLAUDE.md §七）")
    ap.add_argument("--smoke-permnos", type=int, default=150)
    ap.add_argument("--authorised-by-user", metavar="LEDGER_REF", default=None,
                    help="真数据运行的硬闸门，**必须填登记簿凭据**：用户对冻结版 v4 "
                         "书面「读」那一行的定位，形如 `ledger:517` 或 `2026-09-05`。"
                         "该串原样写进 run_meta.json 与报告页头，缺失或不含数字即拒跑。")
    return ap


def check_authorisation(ref: str | None) -> str:
    """闸门：必须给出可定位到登记簿某一行的凭据（行号或日期），不接受裸开关。"""
    ref = (ref or "").strip()
    if not ref or not any(ch.isdigit() for ch in ref):
        raise SystemExit(
            "拒绝运行：真数据解封须先在登记簿留下用户对冻结版 v4 的书面「读」，"
            "并把那一行的凭据填进 --authorised-by-user（如 `ledger:517` 或 `2026-09-05`）。"
            "（v4 §7 第 5、9 条：授权须逐项覆盖 (a) 打开折 05–35 的 scores 与 labels、"
            "(b) §3 释放清单的确切范围、(c) 一次性不可重开的确认。）")
    return ref


def main() -> int:
    args = build_parser().parse_args()
    if args.clean_window_only:
        if args.folds is not None:
            raise SystemExit(
                "--clean-window-only 只跑干净窗 "
                f"fold{min(C.CLEAN_WINDOW_FOLDS):02d}–{max(C.CLEAN_WINDOW_FOLDS):02d}，"
                "确认段（折 05–35）本次不读，--folds 无处可用：两者不得同时给出。")
        args.clean_window = True          # 隐含
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    # 授权闸门在**建目录之前**：被拒的运行不留下任何痕迹
    if not args.smoke:
        check_authorisation(args.authorised_by_user)
    out.mkdir(parents=True, exist_ok=True)

    arms = tuple(a.strip().lower() for a in args.arms.split(",") if a.strip())
    if not arms or any(a not in C.ARMS for a in arms):
        raise SystemExit(f"--arms 只能取 {C.ARMS} 的子集，收到 {args.arms!r}")
    arms = tuple(a for a in C.ARMS if a in arms)      # 顺序先验固定 FT → ZS
    processed, outputs_root, jkp = args.processed, args.outputs_root, args.jkp
    folds = C.CONFIRM_FOLDS if args.folds is None else parse_folds(args.folds)
    clean_folds = C.CLEAN_WINDOW_FOLDS

    if args.smoke:
        from unseal.smoke import DEFAULT_SMOKE_FOLDS, build_workspace
        log("合成假数据全链路（读数无任何含义，不得记入登记簿）...")
        if args.folds is None:
            folds = DEFAULT_SMOKE_FOLDS
        if args.clean_window_only:
            folds = ()                    # 确认段不跑，合成工作区也不造那些折
        ws_folds = tuple(folds) + (tuple(clean_folds) if args.clean_window else ())
        ws = build_workspace(out / "_smoke_workspace", n_permno=args.smoke_permnos,
                             folds=ws_folds)
        processed, outputs_root, jkp = ws["processed"], ws["outputs"], ws["jkp"]
        authorisation = "N/A（--smoke，未碰任何真实封存目录）"
    else:
        authorisation = check_authorisation(args.authorised_by_user)

    t0 = time.perf_counter()
    scope_mode = ("clean_window_only" if args.clean_window_only
                  else ("confirm+clean_window" if args.clean_window else "confirm"))
    run_meta = build_run_meta(out, args) | {"authorisation_ref": authorisation,
                                        "scope_mode": scope_mode,
                                        "scope_note": (
                                            CONFIRM_NOT_READ_NOTE
                                            if args.clean_window_only else None),
                                        "arms_read": list(arms),
                                        "zs_dropped": "zs" not in arms,
                                        "excluded_signal_dates":
                                            sorted(EXCLUDED_SIGNAL_DATES),
                                        "excluded_signal_dates_fold":
                                            EXCLUDED_SIGNAL_DATES_FOLD}
    (out / "run_meta.json").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.clean_window_only:
        log(f"--clean-window-only：{CONFIRM_NOT_READ_NOTE}")
    else:
        payload = run_scope(scope_name="confirm", folds=tuple(folds), out=out,
                            processed=processed, outputs_root=outputs_root, jkp=jkp,
                            run_meta=run_meta, args=args, arms=arms)
        lo = min(folds)
        hi = max(folds)
        (out / "report.md").write_text(
            render_report(payload, title="解封读取报告 —— 确认集",
                          scope=f"折 {lo:02d}–{hi:02d}（{len(folds)} 折）",
                          n_folds=len(folds)),
            encoding="utf-8")
        log(f"确认集报告 → {out / 'report.md'}")

    if args.clean_window and clean_folds:
        cw = out / "clean_window"
        payload_cw = run_scope(scope_name="clean_window", folds=tuple(clean_folds),
                               out=cw, processed=processed, outputs_root=outputs_root,
                               jkp=jkp, run_meta=run_meta, args=args, arms=arms)
        (out / "report_fold44_45.md").write_text(
            render_report(
                payload_cw,
                title="解封读取报告 —— 干净窗（独立交付，不并入确认集）",
                scope=f"折 {min(clean_folds):02d}–{max(clean_folds):02d}"
                      "（2024-07 起，合计 1 年；预期 t / 非中心参数约 0.65，"
                      "只作洁净复核与效应估计，**不承担最终二元裁决**）",
                n_folds=len(clean_folds),
                notes=() if args.h3 else (CLEAN_WINDOW_NO_H3_NOTE,)),
            encoding="utf-8")
        log(f"干净窗报告 → {out / 'report_fold44_45.md'}")
        if args.clean_window_only:
            # 本次运行只有干净窗一段：顶层 summary.json 也只含 clean_window 部分
            (out / "summary.json").write_text(
                json.dumps(payload_cw, ensure_ascii=False, indent=2, default=float),
                encoding="utf-8")

    log(f"完成，总耗时 {(time.perf_counter() - t0) / 60:.1f} 分钟 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
