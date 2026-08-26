"""Kronos 微调层（规范 §6 + docs/预注册_v1.md）。

官方仓库以 submodule 钉死在 third_party/kronos（commit 见 .gitmodules /
git submodule status）。本包不改官方代码：目标函数、归一化、采样参数
逐行沿用官方 finetune/ 实现；只替换数据管道（parquet 面板 → 窗口，
不走 Qlib）与训练循环（单卡、无 DDP/comet，加入预注册的内层验证早停、
SWA-K 权重平均、seed 管理与试验登记簿）。

模块：
- windows   训练/推理窗口索引构造 + 内层验证切分（purge 同 §7 口径）
- dataset   torch Dataset：官方归一化契约（lookback 段统计量、clip ±5）
- models    预训练加载 / 冒烟用小模型工厂 / device 自动降级
- train     单卡两阶段微调（tokenizer → predictor），--smoke 冒烟
- infer     多路径采样推理 → score = predOpen(t+6)/predOpen(t+1) − 1（§4）
"""

from __future__ import annotations

import sys
from pathlib import Path

KRONOS_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "kronos"


def import_kronos():
    """把官方仓库根加入 sys.path 并返回其 model 包（KronosTokenizer/Kronos/
    KronosPredictor）。集中在一处，避免各文件散落 path hack。"""
    p = str(KRONOS_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)
    from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: PLC0415
    return KronosTokenizer, Kronos, KronosPredictor
