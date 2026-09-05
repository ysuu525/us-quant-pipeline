"""解封读取（折 05–35 + fold44–45）的实现包。**只在解封授权后运行。**

判据、用途限制、释放清单的正本抄在 :mod:`unseal.config` 的模块 docstring 与
``scripts/unseal_read_confirm.py`` 的 docstring 里，本包每个模块的 docstring
再抄一份与自己有关的那部分（任务书 §0：判据必须先于结果落笔）。

三条贯穿全包的硬禁令（违反即全部读数作废）：

1. **禁止 ``append_ledger``**。本包与入口脚本一律不 import、不调用
   ``kronos_ft.train.append_ledger``；登记条目由主会话按报告手写
   （`evaluate_fold.py:356` 的自动追加会把逐折 IC 写进 append-only 登记簿，
   越过确认协议 v4 §3 的释放清单）。
2. **禁止计算或输出 FT−ZS 差值**（v4 §3）。两臂各自的预注册量可以并列呈现，
   差值与其显著性一律不算不看；本包任何 JSON / 报告都不得出现差值键。
3. **绝不写进封存目录**。所有产物写到 ``--out`` 指定的新目录；封存目录里
   一旦出现 ``labels.parquet`` 等禁止产物，``crsp_pipeline.sealed.write_seal``
   与 ``audit_dir`` 就会失败（`tests/test_sealed_mode.py`）。
"""
from __future__ import annotations

__all__ = ["config", "paths", "folds", "perfold", "aggregate", "h1b", "h6", "report", "smoke"]
