"""实验 8 驱动：把 MLP 排序头从 fold40 单折补齐到七折 36–42。

判据 / 用途限制（抄自 docs/实验指令_2026-09-04.md 实验 8，先写后跑）
- **估计交付**。**不用于选读出**：读出方案属 B 层（按先验，CLAUDE.md §二），
  本实验只把表格口径补齐，不重开选择。
- 超参照原冻结不动；**内层选参口径与线性探针一致**（训练窗尾部内层验证窗，
  外层验证窗**绝不用于调参**）；**不做任何网格扩张**。
- 同时报诚实口径与事后最优（oracle）上界，且 oracle 列必须逐格标注
  「**不可用于判定**」。
- **凡是折数或天数不同的两个数，表里必须分行，不得并列成一句结论。**

折窗口**不得手写**
------------------
本脚本用 `crsp_pipeline.splits.walk_forward_folds` 机械生成折表，并与
`scripts/ridge_probe_folds.py::FOLDS`（线性探针实际使用、且折 36–40 的表示缓存
与 ledger:164 读数都建立在其上的那张表）**逐位断言一致**；不一致直接报错退出。
这样 MLP 行与线性探针行落在同一批日期上，满足 CLAUDE.md §八。

`--fold-calendar-start` 的默认值 `2000-01-01` 是本项目**既有全部产物**所用的口径：
`scripts/gbdt_baseline.py::FOLDS`、`scripts/ridge_probe_folds.py::FOLDS`，以及
折 36–42 的 Kronos eval 产物里 `metrics.json.val_window` 全部与之逐位一致
（例：fold36 val=[2020-07-01..2020-12-31]，fold42 val=[2023-07-03..2023-12-29]）。
**注意**：`scripts/emit_folds.py` 硬写的是 `2000-01-03`，它给出的是**另一张表**
（fold36 val 起 2020-07-06、fold42 val 止 2024-01-02）。两者的分歧已在交回报告里
单列，须由主会话裁定；在裁定前本脚本按「与被比较对象同口径」取默认值，
并把两张表同时落盘存证。

超参溯源（一字不改，出处见交回报告）
- `--lookback 90 --predict 6 --amp bf16 --batch-size 256 --loss ic --seed 0
   --inner-months 6 --backbone-dir outputs/zeroshot_base`
  取自 `outputs/rankhead_fold40/eval_rh_zeroshot_fold40/metrics.json`
  的 `scoring_config` 与 `train_rank_head.py` 的脚本默认值。

工程约束（CLAUDE.md §七）
- 每折启动前查系统提交内存，超过 `--mem-gate-gb`（默认 42 GB）就等 60 秒再查；
- 互斥锁 + 失败重试；已有 `metrics.json` 的折幂等跳过；
- 子进程 `OMP_NUM_THREADS=2 / MKL_NUM_THREADS=2`（另有 CPU 串行队列在跑）；
- GPU 任务严格串行，绝不并发。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from crsp_pipeline.calendar import TradingCalendar  # noqa: E402
from crsp_pipeline.splits import walk_forward_folds  # noqa: E402

LOCK_PATH = REPO_ROOT / "outputs" / "exp8_mlp_folds.lock"


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def build_fold_table(processed: Path, start: str, first: int, last: int) -> list[dict]:
    """机械生成折表（与 scripts/emit_folds.py 同一函数、同一参数化）。"""
    cal = TradingCalendar.from_market_index(
        pd.read_parquet(processed / "market_index.parquet"), "caldt")
    folds = walk_forward_folds(cal, start, cal.dates[-1])
    out = []
    for i, f in enumerate(folds, start=1):
        if not (first <= i <= last):
            continue
        out.append({"n": f"fold{i:02d}",
                    "ts": str(f.train_start.date()), "te": str(f.train_end.date()),
                    "vs": str(f.val_start.date()), "ve": str(f.val_end.date())})
    return out


def assert_matches_ridge_probe(table: list[dict]) -> None:
    """与线性探针实际使用的折表逐位比对；不一致就停，不许自己挑。"""
    import ridge_probe_folds  # noqa: PLC0415

    ref = {name: (ts, te, vs, ve) for name, ts, te, vs, ve in ridge_probe_folds.FOLDS}
    bad = []
    for row in table:
        got = (row["ts"], row["te"], row["vs"], row["ve"])
        want = ref.get(row["n"])
        if want is None:
            bad.append(f"{row['n']}: 线性探针折表里没有这一折")
        elif got != want:
            bad.append(f"{row['n']}: 生成 {got} != 线性探针 {want}")
    if bad:
        raise SystemExit(
            "折表与 scripts/ridge_probe_folds.py::FOLDS 不一致，拒绝运行：\n  "
            + "\n  ".join(bad))


def committed_gb() -> float:
    cmd = ["powershell", "-NoProfile", "-Command",
           "(Get-Counter '\\Memory\\Committed Bytes').CounterSamples[0].CookedValue/1GB"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return float(out.strip().splitlines()[-1].replace(",", "."))


def wait_for_memory(gate_gb: float, poll_s: int = 60, max_waits: int = 120) -> float:
    for _ in range(max_waits):
        cur = committed_gb()
        if cur <= gate_gb:
            return cur
        log(f"  系统提交内存 {cur:.1f} GB > 门槛 {gate_gb:.1f} GB，等待 {poll_s}s")
        time.sleep(poll_s)
    raise SystemExit(f"提交内存长期高于 {gate_gb} GB，放弃启动")


def run_fold(python: str, row: dict, processed: Path, out_root: Path, log_dir: Path,
             retries: int, gate_gb: float, dry_run: bool) -> str:
    tag = f"exp8_mlp_{row['n']}"
    out_dir = out_root / tag
    done_marker = out_dir / f"eval_{tag}" / "metrics.json"
    if done_marker.exists():
        log(f"{row['n']}: 已有 {done_marker}，幂等跳过")
        return "skipped"

    cmd = [python, str(REPO_ROOT / "scripts" / "exp8_mlp_one_fold.py"),
           "--processed", str(processed),
           "--backbone-dir", "outputs/zeroshot_base",
           "--train-start", row["ts"], "--train-end", row["te"],
           "--val-start", row["vs"], "--val-end", row["ve"],
           "--lookback", "90", "--predict", "6",
           "--amp", "bf16", "--batch-size", "256",
           "--loss", "ic", "--seed", "0", "--inner-months", "6",
           "--out", str(out_dir), "--tag", tag]
    log(f"{row['n']}: {' '.join(cmd)}")
    if dry_run:
        return "dry-run"

    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                "PYTHONIOENCODING": "utf-8"})
    log_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        cur = wait_for_memory(gate_gb)
        log(f"{row['n']}: 第 {attempt}/{retries} 次尝试（提交内存 {cur:.1f} GB）")
        log_file = log_dir / f"{tag}.attempt{attempt}.log"
        with open(log_file, "w", encoding="utf-8") as fh:
            rc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env,
                                stdout=fh, stderr=subprocess.STDOUT).returncode
        if rc == 0 and done_marker.exists():
            log(f"{row['n']}: 完成（日志 {log_file}）")
            return "ok"
        log(f"{row['n']}: 失败 rc={rc}（日志 {log_file}）")
    return "failed"


def main() -> None:
    ap = argparse.ArgumentParser(description="实验 8：MLP 排序头补七折")
    ap.add_argument("--processed", required=True)
    ap.add_argument("--fold-calendar-start", default="2000-01-01",
                    help="walk_forward_folds 的起点；默认 = 既有全部产物的口径")
    ap.add_argument("--first", type=int, default=36)
    ap.add_argument("--last", type=int, default=42)
    ap.add_argument("--skip-folds", default="fold40",
                    help="逗号分隔；默认跳过 fold40（同参数同窗口的产物已存在，复用）")
    ap.add_argument("--out-root", default="outputs")
    ap.add_argument("--log-dir", default="outputs/exp8_logs")
    ap.add_argument("--folds-json", default="outputs/exp8_folds_36_42.json")
    ap.add_argument("--python", default=str(REPO_ROOT / ".venv" / "Scripts" / "python.exe"))
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--mem-gate-gb", type=float, default=42.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    processed = Path(args.processed)
    table = build_fold_table(processed, args.fold_calendar_start, args.first, args.last)
    assert_matches_ridge_probe(table)
    other = build_fold_table(processed, "2000-01-03", args.first, args.last)

    payload = {"fold_calendar_start": args.fold_calendar_start,
               "folds": table,
               "emit_folds_py_start_2000_01_03": other,
               "note": "两张表的分歧须由主会话裁定；本次按与被比较对象同口径取前者。"}
    Path(args.folds_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.folds_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
    log(f"折表已落盘 {args.folds_json}")

    skip = {s.strip() for s in args.skip_folds.split(",") if s.strip()}
    if args.dry_run:
        for row in table:
            if row["n"] in skip:
                log(f"{row['n']}: --skip-folds 指定跳过")
                continue
            run_fold(args.python, row, processed, Path(args.out_root),
                     Path(args.log_dir), args.retries, args.mem_gate_gb, True)
        return

    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(f"已有实例在跑（锁 {LOCK_PATH}）；确认无进程后手工删除该文件")
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    status = {}
    try:
        for row in table:
            if row["n"] in skip:
                log(f"{row['n']}: --skip-folds 指定跳过（复用既有产物）")
                status[row["n"]] = "reused"
                continue
            status[row["n"]] = run_fold(args.python, row, processed, Path(args.out_root),
                                        Path(args.log_dir), args.retries,
                                        args.mem_gate_gb, False)
    finally:
        LOCK_PATH.unlink(missing_ok=True)
    log("逐折状态：" + json.dumps(status, ensure_ascii=False))
    if any(v == "failed" for v in status.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
