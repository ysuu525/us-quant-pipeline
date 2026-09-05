"""实验 8：单折运行 `scripts/train_rank_head.py`，**不改其任何超参与算法**。

存在的唯一理由：`train_rank_head.main()` 在结尾调用 `kronos_ft.train.append_ledger`，
会直接往 append-only 的 `experiments/ledger.md` 追加一行。实验 8 的任务书要求
**不改 `experiments/ledger.md`**（草稿行另写文件，由主会话审后合并），
所以本包装器在**调用前**把 `train_rank_head.append_ledger` 重定向到
`outputs/exp8_run_ledger_lines.log`。除此之外一行不动：
参数解析、表示抽取、内层选参、早停、指标计算全部走原脚本。

用法（参数原样透传给 train_rank_head.py 的 argparse）：

    .venv\\Scripts\\python.exe scripts/exp8_mlp_one_fold.py \\
        --processed <P> --train-start ... --train-end ... \\
        --val-start ... --val-end ... --out outputs/exp8_mlp_fold36 \\
        --tag exp8_mlp_fold36

判据 / 用途限制（抄自 docs/实验指令_2026-09-04.md 实验 8，先写后跑）
- **估计交付**。**不用于选读出**：读出方案属 B 层（按先验，CLAUDE.md §二），
  本实验只把表格口径补齐，不重开选择。
- 超参照原冻结不动；**内层选参口径与线性探针一致**（训练窗尾部内层验证窗，
  外层验证窗**绝不用于调参**）；**不做任何网格扩张**。
- 同时报诚实口径与事后最优（oracle）上界，且 oracle 列必须逐格标注
  「**不可用于判定**」。
- **凡是折数或天数不同的两个数，表里必须分行，不得并列成一句结论。**
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import train_rank_head  # noqa: E402

LOG_PATH = REPO_ROOT / "outputs" / "exp8_run_ledger_lines.log"


def _redirected_append_ledger(line: str) -> None:
    """原本会写进 experiments/ledger.md 的那一行，改写到本实验自己的日志。"""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(f"- {stamp} | {line}\n")


def main() -> None:
    train_rank_head.append_ledger = _redirected_append_ledger
    train_rank_head.main()


if __name__ == "__main__":
    main()
