"""冻结的纯多变现构造——**唯一权威实现**。

来源
----
逐字搬自 `scripts/compare_arms_money.py` 的 `arm_returns()`（第 62–122 行）。
`scripts/k9_cost_alpha_frontier.py` 的 q=1 等权路径与之逐位一致（该脚本
docstring 第 186–187 行自述）。CLAUDE.md §五：**组合构造必须先冻结再比臂**，
否则是拿同一批数据同时优化「信号」与「怎么用信号」。

因此本模块的任何改动都会让此前所有臂间读数失效：排序方式、tie-break、
`rank` 的 method、`sorted` 的稳定性依赖、成本公式，一个字符都不能动。
`tests/test_portfolio.py` 用原样复制的参考实现断言 `np.array_equal`（不是
`allclose`），改动会立刻红灯。

冻结口径
--------
- 池：当日 lagged ADV20 前 `topn`（500）只；ADV 缺失或非有限的名字先剔除。
- 进入：分数百分位秩前 10%（`k = max(1, n // 10)`）。
- 退出：跌出前 30%（`exit_pct`）才卖（无交易带）。
- 套袖：`nt`=6 档错位，第 i 个交易日只重平衡第 `i % nt` 个套袖。
- 权重：套袖内等权，套袖间等权（对有效套袖取均值）。
- 时点：t 日收盘算分 → t+1 开盘成交。当日新买入的名字只吃 open→close 段
  (`oc`)，其余名字吃全日 `DlyRet`（CLAUDE.md §一.4）。
- 成本：`2 * cost_bp/1e4 * turn / nt`（双边；每天只有一个套袖在换手）。
- 基准：同日同池（`pct` 的名字集）等权。输出 = 组合 − 基准 − 成本。
- 预热：`i < nt` 的日子不产出收益（套袖尚未铺满）。

原脚本行号 → 本文件行号
------------------------
| `compare_arms_money.py` | 语句 | `construction.py` |
|---|---|---|
| 62 | `def arm_returns(...)` | 107（`frozen_long_only_returns`） |
| 64 | `rows = []` | 143 |
| 65–66 | `for fold in FOLDS` / 目录拼接 | —（本函数只处理一折，折循环交给调用方） |
| 67–71 | 读 parquet + 构造 `by_day`（含 `len(g) >= 50`） | 见 `scores_frame_to_by_day`（第 87 行）；本函数第 141 行再兜一次同一过滤 |
| 73 | `days = sorted(by_day)` | 144 |
| 74 | `book = [None] * NT` | 145 |
| 75 | `for i, day in enumerate(days)` | 146 |
| 76 | `s, a = by_day[day], adv.get(day, {})` | 147 |
| 77 | `elig = [...]` ADV 有限性过滤 | 148 |
| 78–79 | `topn` 截断（`sorted` 稳定，tie 按 `s` 键序） | 149–150 |
| 80 | `elig = set(elig)` | 151 |
| 81 | `s = {...}` 限制到池内 | 152 |
| 82 | `n = len(s)` | 153 |
| 83–84 | `if n < 50: continue` | 154–155 |
| 85 | `pct = (pd.Series(s).rank() / n).to_dict()` | 156 |
| 86 | `k = max(1, n // 10)` | 157 |
| 87 | `order = sorted(pct, key=lambda p: -pct[p])` | 158 |
| 88–89 | `j = i % NT` / `prev = book[j]` | 159–160 |
| 90–92 | 首次建仓分支 | 161–163 |
| 93–98 | 保留/补入分支与 `turn` | 164–169 |
| 99 | `book[j] = nb` | 170 |
| 100–101 | `if i + 1 >= len(days) or i < NT: continue` | 171–172 |
| 102 | `nd = days[i + 1]` | 173 |
| 103 | `rm, om = ret.get(nd, {}), oc.get(nd, {})` | 174 |
| 104–105 | `if not rm: continue` | 175–176 |
| 106 | `cost = 2.0 * (cost_bp / 1e4) * turn / NT` | 177 |
| 107–115 | 套袖收益聚合（新买入吃 `om`，其余吃 `rm`） | 178–186 |
| 116–117 | `if not vals: continue` | 187–188 |
| 118 | `bench = ...` | 189 |
| 119 | `rows.append(...)` | 191（多带 `turn`、`n_names` 两个诊断列） |
| 120 | `DataFrame(...).set_index("date").sort_index()` | 192–194 |

新增的 `turn` / `n_names` 只是诊断列，写在 `rows` 元组尾部，**不参与任何
浮点运算**，因此 `r` 列与原实现逐位相同。原实现的 `fold` 列由调用方在折
循环里自己贴（本函数不知道折）。

浮点逐位一致的三个脆弱点（改动前必读）
----------------------------------------
1. `pd.Series(s)` 的 index 顺序 = `s` 的键插入顺序 = `sc.groupby` 后的行顺序。
   `rank()` 用默认 `method='average'`；换成 `'first'` 或 `'min'` 会改变并列
   名字的百分位，进而改变进出场名单。
2. `sorted(pct, key=lambda p: -pct[p])` 是 Python 的**稳定**排序：并列分数
   按 `pct` 的键序（即上面那个插入顺序）决定谁先进场。改用
   `Series.sort_values` 或 `argsort` 会换 tie-break。
3. `np.mean(list)` 的成对求和顺序由列表顺序决定；`rs` 的顺序来自 `nm`
   （= `keep + add`）。改成 `sum(...)/len(...)` 或先排序都会动最后几位。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["frozen_long_only_returns", "scores_frame_to_by_day"]


def scores_frame_to_by_day(df: pd.DataFrame, min_names: int = 50) -> dict:
    """复现 `compare_arms_money.py` 第 67–70 行的 `by_day` 构造。

    输入 `DataFrame[PERMNO, signal_date, score]`（多余列会被丢弃，与原脚本
    `pd.read_parquet(columns=[...])` 的效果一致），输出
    `dict[Timestamp, dict[PERMNO, score]]`。

    与原脚本一致的三处约定：
    - 先取那三列再 `dropna()`（原脚本先按列读 parquet 再 dropna）；
    - `groupby("signal_date")` 默认 `sort=True`，组内保持原行序；
    - `len(g) >= min_names`（50）的日子才进 `by_day`——**这不是可有可无的
      过滤**：被丢掉的日子会整个从 `days` 里消失，从而改变 `i % NT` 的套袖
      编号与 `days[i + 1]` 的下一日，不能挪到循环内部去做。
    """
    sc = df[["PERMNO", "signal_date", "score"]].dropna()
    sc = sc.assign(signal_date=pd.to_datetime(sc["signal_date"]))
    return {day: dict(zip(g["PERMNO"], g["score"]))
            for day, g in sc.groupby("signal_date") if len(g) >= min_names}


def frozen_long_only_returns(
    scores_by_day: dict,
    ret_by_day: dict,
    oc_by_day: dict,
    adv_by_day: dict,
    *,
    topn: int = 500,
    cost_bp: float = 8.0,
    exit_pct: float = 0.30,
    nt: int = 6,
    min_names: int = 50,
) -> pd.DataFrame:
    """一折的冻结纯多超额日收益。`r` 列与 `arm_returns` 的 `r` 列逐位相同。

    参数
    ----
    scores_by_day : ``{Timestamp: {PERMNO: score}}``，与 `arm_returns` 内部的
        `by_day` 同构。若尚未按「当日 ≥ min_names 个名字」过滤，本函数会补做
        （见下方第 141 行），与原脚本第 71 行的 `if len(g) >= 50` 等价。
    ret_by_day / oc_by_day / adv_by_day : 与 `load_prices()` 返回的
        `ret / oc / adv` 同构，`{Timestamp: {PERMNO: float}}`。
        `oc = DlyClose / |DlyOpen| - 1`；`adv` 是 **lagged** ADV20。
    topn, cost_bp, exit_pct, nt, min_names : 冻结默认值 500 / 8.0 / 0.30 / 6 / 50。
        暴露为参数只为成本敏感性扫描（k9 前沿）与测试；**比臂时一律用默认值**。

    返回
    ----
    `DataFrame(index=date, columns=["r", "turn", "n_names"])`，`date` 是成交日
    （t+1），已 `sort_index()`。`turn` 是当日重平衡套袖的换手率（进成本公式的
    那个数），`n_names` 是当日全部套袖持有的去重名字数——两者都是诊断，不进
    任何浮点运算。
    """
    ret, oc, adv = ret_by_day, oc_by_day, adv_by_day
    NT, EXIT_PCT = nt, exit_pct
    by_day = {d: s for d, s in scores_by_day.items() if len(s) >= min_names}

    rows = []
    days = sorted(by_day)
    book = [None] * NT
    for i, day in enumerate(days):
        s, a = by_day[day], adv.get(day, {})
        elig = [p for p in s if p in a and np.isfinite(a[p])]
        if topn and len(elig) > topn:
            elig = sorted(elig, key=lambda p: -a[p])[:topn]
        elig = set(elig)
        s = {p: v for p, v in s.items() if p in elig}
        n = len(s)
        if n < min_names:
            continue
        pct = (pd.Series(s).rank() / n).to_dict()
        k = max(1, n // 10)
        order = sorted(pct, key=lambda p: -pct[p])
        j = i % NT
        prev = book[j]
        if prev is None:
            nb, fresh = list(order[:k]), set(order[:k])
            turn = 0.0
        else:
            keep = [p for p in prev if p in pct and pct[p] >= 1 - EXIT_PCT][:k]
            held = set(keep)
            add = [p for p in order if p not in held][:k - len(keep)]
            nb, fresh = keep + add, set(add)
            turn = (k - len(keep)) / k
        book[j] = nb
        if i + 1 >= len(days) or i < NT:
            continue
        nd = days[i + 1]
        rm, om = ret.get(nd, {}), oc.get(nd, {})
        if not rm:
            continue
        cost = 2.0 * (cost_bp / 1e4) * turn / NT
        vals = []
        for t in range(NT):
            nm = book[t]
            if not nm:
                continue
            rs = [(om.get(p) if (t == j and p in fresh) else rm.get(p)) for p in nm]
            rs = [v for v in rs if v is not None and np.isfinite(v)]
            if rs:
                vals.append(np.mean(rs))
        if not vals:
            continue
        bench = float(np.mean([rm[p] for p in pct if p in rm]))
        n_names = len({p for sleeve in book if sleeve for p in sleeve})
        rows.append((nd, float(np.mean(vals)) - bench - cost, turn, n_names))
    return (pd.DataFrame(rows, columns=["date", "r", "turn", "n_names"])
            .set_index("date")
            .sort_index())
