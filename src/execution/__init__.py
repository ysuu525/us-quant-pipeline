"""成交成本小试的执行层（`experiments/cost_pilot_protocol_v1_draft.md`）。

模块划分：

- `fees`         —— HANDOFF §11.4a 的费率表与主指标 `C` 的 bp 计算；
- `collision`    —— 协议 §4 的反向碰撞哈希让位 / 同向合并；
- `shadow_ledger`—— 协议 §5 的双影子账本与逐股核对；
- `orders`       —— 订单/成交流水的列定义与追加落盘；
- `alpaca_client`—— OPG 提交窗口检查与 dry-run 客户端；
- `daily_flow`   —— 协议 §5 的每日六步流程。

**本包不导入 torch / kronos_ft，也不读取任何 parquet 面板。**
"""
from __future__ import annotations

__all__ = [
    "alpaca_client",
    "collision",
    "daily_flow",
    "fees",
    "orders",
    "shadow_ledger",
]
