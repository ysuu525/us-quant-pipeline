"""G6：H1b（扩充控制集张成后的 alpha）在折 05–35 上的驱动。

判据与地位（先于结果落笔）
--------------------------
* v4 §2.4 **裁定 C-1 已采纳**：H1b 以**估计交付**进入确认集，默认运行
  （``--no-h1b`` 可关）。不设门槛、不做 PASS/FAIL、不进 H1 判定、不参与任何
  α 分配、不作部署依据，**不得改变 E / H1 / H2 / H2-era / H6 的任何结论**。
* **只在 FT 臂交付**（v4 §2.4 第 4 条）：开发折读数与 31 折 MDE 外推都只有 FT 臂，
  ZS 臂没有可外推的 SE；**不得据确认结果回溯改臂或补做 ZS 臂**。
* **无 SESOI，故不作功效关判定**；本协议生效期内不得为 H1b 补写 SESOI 并据以判定。
* 主规格**先验固定为 S-TH-ind**（自由参数最少 + 唯一做了行业中性 = B 层优先序
  ②④）；S-T / S-H 并列作**次规格**，三者一并报告，**不得事后择优**。
* 口径沿 K6b 冻结构造（**NT=6**、top500、进前 10%、跌出前 30% 才卖、
  控制腿毛收益 / 策略腿 8bp）。**与 NT=5 的 E 端读数不并列比较**
  （`CLAUDE.md` §八），报告页头必须写明这一点。
* 保留率 > 100% 时必须附 K6b 的限定原句：「只表示正向暴露于本样本内亏钱的因子，
  **不得写成无限定的 survives spanning**」。
* 缺口照旧披露：财报日虚拟缺外部日历、SUE 缺 Compustat，**均未做**。

实现
----
不复制 `scripts/exp11_spanning_extended.py` 的任何一行，只按文件路径加载它并注入
``folds`` / ``fold_windows`` / ``blocks`` / ``scores_root``。分数从解封入口写出的
**未封存输出树**取（`<out>/ft/fold<NN>/scores.parquet`），exp11 因此不知道封存
目录的命名。折窗口由 :mod:`unseal.folds` 机械生成，**不得手写**。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Sequence

from . import config as C
from .folds import FoldWindow, two_year_blocks

__all__ = ["load_exp11", "run_h1b"]

REPO = Path(__file__).resolve().parents[2]


def load_exp11():
    spec = importlib.util.spec_from_file_location(
        "exp11_spanning_extended", REPO / "scripts" / "exp11_spanning_extended.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_h1b(processed: Path, jkp: Path, scores_root: Path,
            windows: Sequence[FoldWindow], out_json: Path, *,
            outputs_root: Path | None = None,
            memory_limit_gb: float | None = None,
            allow_alt_processed: bool = False) -> dict[str, Any]:
    """在给定折上跑三个冻结规格；返回 exp11 的报告字典（已加本模块的限定）。"""
    exp11 = load_exp11()
    folds = tuple(w.fold for w in windows)
    fold_windows = {w.fold: (str(w.val_start.date()), str(w.val_end.date()))
                    for w in windows}
    blocks = [{"lo": b["lo"], "hi": b["hi"], "folds": tuple(b["folds"])}
              for b in two_year_blocks(windows)]
    report = exp11.run(
        Path(processed).resolve(), Path(jkp).resolve(),
        Path(outputs_root or (REPO / "outputs")).resolve(),
        folds=folds, fold_windows=fold_windows, blocks=blocks,
        scores_root=Path(scores_root).resolve(),
        memory_limit_gb=(exp11.MEMORY_LIMIT_GB if memory_limit_gb is None
                         else memory_limit_gb),
        allow_alt_processed=allow_alt_processed,
    )
    report["meta"]["unseal"] = {
        "endpoint": "H1b",
        "status": "v4 §2.4 裁定 C-1 已采纳：以估计交付进入确认集",
        "form": "估计交付，无门槛，不产生 PASS/FAIL；无 SESOI，不作功效关判定",
        "arms": list(C.H1B_ARMS),
        "arm_note": ("只在 FT 臂交付（ZS 臂无可外推的开发折 SE）；"
                     "不得据确认结果回溯改臂或补做 ZS 臂"),
        "primary_spec": C.H1B_PRIMARY_SPEC,
        "secondary_specs": [s for s in C.H1B_SPECS if s != C.H1B_PRIMARY_SPEC],
        "mde80_disclosed_nt6": C.H1B_MDE80_NT6,
        "nt": C.H1B_NT,
        "caliber_warning": (
            f"沿 K6b 冻结构造 NT={C.H1B_NT}；**与 E 端 NT={C.NT} 的读数不并列比较**"
            "（CLAUDE.md §八）。"
        ),
        "use_restriction": "不得改变 E / H1 / H2 / H2-era / H6 的任何结论",
    }
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    import json
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float),
                        encoding="utf-8")
    return report
