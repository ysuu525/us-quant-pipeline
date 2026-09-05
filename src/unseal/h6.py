"""G8：H6（ICT P1「扫流动性后收回」）在折 05–35 上的驱动。

判据与地位（v4 §2.8，先于结果落笔）
-----------------------------------
**探索性估计交付。** 三条写死，任何一条被违反即本节读数作废：

1. **不进入 H1–H4 的判定**：不进 family、不进固定序贯链、不参与任何 α 分配；
2. **不作部署依据**：E 端的部署规则与动作表不引用 H6；
3. **与 H1–H4 在同一解封时刻一次读完**，不得为「先看看 P1 行不行」提前打开
   折 05–35 的任何目录。

只跑 ``P1_bull`` / ``P1_bear``，只算 **Q1** 与 **Q2(b)**（不做 Q2(a)）；
NW lag = **6**；SESOI = **10 bp / 6 日持有**；标记定义与全部固定参数一字不改
（L=20、H20/L20 用 t-20..t-1）。Q2(b) 的五分位用 **Kronos FT 封存分数
（sample_count=5）**——口径核对由解封入口在读分数前完成（`CLAUDE.md` §八）。

层级（v4 §2.8.4，本文推算的 31 折外推）：Q1 的 MDE80₃₁ ≈ 9.9–10.1bp ≈ SESOI
→ **A 层（边界）**，但仍不产生 PASS/FAIL；Q2(b) 的 MDE80₃₁ ≈ 22bp > SESOI
→ **B 层，只报数不判定**。CI 覆盖 0 时按措辞强制写「在本样本量下该问题不可回答」，
**不得**写「P1 无效」。

实现
----
按文件路径加载 `scripts/ict_pattern_probe.py` 并调用它的 :func:`configure`，
把折表、面板块、标记与问题注入进去；分数与标签从解封入口写出的**未封存输出树**
读取（`<out>/ft/fold<NN>/{scores,labels}.parquet` + `metrics.json`），
探针因此不知道封存目录的命名。面板块回看余量 > 45 个交易日（``MIN_HIST_P5P6``）。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from . import config as C
from .folds import FoldWindow

__all__ = ["load_ict", "build_blocks", "run_h6"]

REPO = Path(__file__).resolve().parents[2]
#: 面板块的回看余量（日历日）。> 45 个交易日 = MIN_HIST_P5P6。
LOOKBACK_CALENDAR_DAYS = 150


def load_ict():
    spec = importlib.util.spec_from_file_location(
        "ict_pattern_probe", REPO / "scripts" / "ict_pattern_probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_blocks(windows: Sequence[FoldWindow], *, per_block: int = 4) -> dict[str, dict]:
    """把折按顺序切成若干块；每块面板窗 = [首折 val_start − 150 日历日, 末折 val_end]。"""
    ordered = sorted(windows, key=lambda w: w.fold)
    blocks: dict[str, dict] = {}
    for i in range(0, len(ordered), per_block):
        group = ordered[i:i + per_block]
        lo = min(w.val_start for w in group) - pd.Timedelta(days=LOOKBACK_CALENDAR_DAYS)
        hi = max(w.val_end for w in group)
        name = f"b{i // per_block + 1:02d}"
        blocks[name] = {
            "folds": [w.name for w in group],
            "panel_lo": str(lo.date()),
            "panel_hi": str(hi.date()),
            "label": f"折 {group[0].fold:02d}–{group[-1].fold:02d}，统一口径（sc=5）",
        }
    return blocks


def run_h6(processed: Path, eval_root: Path, windows: Sequence[FoldWindow],
           out_dir: Path, *, force: bool = False,
           per_block: int = 4) -> dict[str, Any]:
    """跑 P1 的 Q1 与 Q2(b)；返回 stage3 的 results 字典（不写探针自己的报告）。"""
    ict = load_ict()
    processed, eval_root = Path(processed), Path(eval_root)
    blocks = build_blocks(windows, per_block=per_block)
    ict.configure(
        folds={w.name: w.name for w in sorted(windows, key=lambda x: x.fold)},
        blocks=blocks,
        markers=C.H6_MARKERS,
        questions=C.H6_QUESTIONS,
        eval_root=eval_root,
        panel=processed / "panel_kronos_adj.parquet",
        universe=processed / "universe.parquet",
        primary_block=next(iter(blocks)),
    )
    results = ict.run_probe(Path(out_dir), force=force, stages=("1", "2", "3"),
                            report=False)
    results = dict(results or {})
    results["unseal"] = {
        "endpoint": "H6",
        "form": "探索性估计交付；不产生 PASS/FAIL",
        "markers": list(C.H6_MARKERS),
        "questions": list(C.H6_QUESTIONS),
        "nw_lag": C.H6_NW_LAG,
        "sesoi_bp": C.H6_SESOI_BP,
        "score_arm": "ft",
        "use_restriction": (
            "不进 H1–H4 判定、不作部署依据、不得据以修改构造 / 退出线 / universe / "
            "部署规则；CI 覆盖 0 时写「" + C.UNANSWERABLE + "」，不得写「P1 无效」"
        ),
        "alternative_explanation": (
            "开发折主块 6 个看涨标记 5 个均值为负、看跌镜像为正，符号整体与短期反转一致、"
            "与 ICT 顺势处方相反；「t+1 开盘建仓吃跳空」是本设计分辨不了的替代解释"
            "（ledger:473）。"
        ),
    }
    return results
