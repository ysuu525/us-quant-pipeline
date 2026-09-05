"""冻结纯多构造 + Rule E 进场过滤（exp14 专用）。

**本模块不改 `construction.py`**（任务书 `docs/任务书_exp14_RuleE_开发折估计_2026-09-05.md`
§2.2）。它把 :func:`portfolio.construction.frozen_long_only_returns` 逐行复制一份，
只在**两处**插入掩码钩子，其余一个字符都不动。

签名 = 冻结函数 + 一个 ``eligible`` 掩码（按 signal_date × PERMNO 的布尔表，
:class:`EligibilityMask` 内部稀疏存储被禁止的名字）。

注入点（行号对应 `src/portfolio/construction.py` 的冻结实现）
------------------------------------------------------------
| 冻结函数行号 | 语句 | 本模块的处理 |
|---|---|---|
| 150–162 | 池构造 / `n` / `pct` / `k` / `order` | **完全不动**：掩码在 `n`、`pct`、`k` 之后才生效，所以 `min_names`（158–159）与 benchmark（193）都不受掩码影响 |
| 165–167 | `prev is None` 首次建仓分支 | **注入①**：`nb` 改为「在可进名字中按 `order` 取前 `k` 个」 |
| 169 | `keep = [p for p in prev if p in pct and pct[p] >= 1 - EXIT_PCT][:k]` | 退出排名**仍在全池 `pct` 上算**；默认不动。仅当 `force_exit=True`（E3 的「禁进+强退」格）才额外剔除被掩码的持仓名字 |
| 171 | `add = [p for p in order if p not in held][:k - len(keep)]` | **注入②**：追加 `and p not in blocked`；`[:k - len(keep)]` 的名额上限不变 ⇒ **名额递补、入选数量与无掩码时相同**（可进名字够时） |
| 173 | `turn = (k - len(keep)) / k` | 不动 |
| 175–195 | 收益聚合 / 成本 / 基准 / 输出 | 不动 |

**掩码全 True 时逐位重现冻结函数输出**：空的 blocked 集合让注入① 退化为
``[p for p in order if p not in frozenset()][:k]``（== ``list(order[:k])``），
注入② 退化为原式，`keep` 不变 ⇒ `nb`、`fresh`、`turn`、`vals` 的元素与顺序
全部相同 ⇒ `r`/`turn`/`n_names` 三列逐位相同。
`tests/test_exp14_construction_rule_e.py` 用 ``np.array_equal`` 断言。

为什么掩码只影响进场
--------------------
规则定义（`docs/设计_财报日回避规则_2026-09-05.md` §2 第 4 条）只有「不买」一个
动作，已持仓不强制退出。因此掩码只出现在建仓候选的取用处（注入①②），不出现在
`keep` 的筛选式里；被掩码的已持仓名字照常按「跌出前 30% 才卖」判定，
不因掩码而退出。E3 的敏感性格需要「禁进 + 强退」时，用显式的 ``force_exit=True``
打开第二个动作，并在读数里单独标注。
"""
from __future__ import annotations

from typing import AbstractSet, Any, Mapping

import numpy as np
import pandas as pd

__all__ = ["EligibilityMask", "rule_e_long_only_returns"]

_EMPTY: frozenset = frozenset()


class EligibilityMask:
    """按 (signal_date, PERMNO) 的布尔掩码；``True`` = 当日可新建仓。

    内部只存**被禁止**的名字（稀疏），因为 Rule E 的触发率是个位数百分比，
    稠密表在 7 折 × ~500 名字 × ~880 日上没有必要。语义上等价于一张
    「signal_date × PERMNO 的布尔表，未出现的格子为 True」。
    """

    __slots__ = ("_blocked", "meta")

    def __init__(self, blocked_by_day: Mapping[Any, AbstractSet] | None = None,
                 meta: dict | None = None) -> None:
        self._blocked: dict = {
            pd.Timestamp(day): frozenset(names)
            for day, names in (blocked_by_day or {}).items()
            if len(names) > 0
        }
        self.meta: dict = dict(meta or {})

    # -- 构造 ---------------------------------------------------------------
    @classmethod
    def all_true(cls) -> "EligibilityMask":
        """全 True 掩码；与 ``eligible=None`` 等价，用于逐位重现测试。"""
        return cls({})

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, *, day_col: str = "signal_date",
                   name_col: str = "PERMNO", value_col: str = "eligible",
                   meta: dict | None = None) -> "EligibilityMask":
        """由稠密布尔表构造：列 = (signal_date, PERMNO, eligible)。"""
        need = {day_col, name_col, value_col}
        missing = need - set(frame.columns)
        if missing:
            raise ValueError(f"掩码表缺列 {sorted(missing)}")
        bad = frame.loc[~frame[value_col].astype(bool)]
        blocked: dict = {}
        for day, group in bad.groupby(day_col, sort=False):
            blocked[pd.Timestamp(day)] = frozenset(group[name_col].tolist())
        return cls(blocked, meta=meta)

    @classmethod
    def from_blocked_frame(cls, frame: pd.DataFrame, *, day_col: str = "signal_date",
                           name_col: str = "PERMNO",
                           meta: dict | None = None) -> "EligibilityMask":
        """由「只列出被禁止格子」的稀疏表构造（本模块的常用入口）。"""
        blocked: dict = {}
        if len(frame):
            for day, group in frame.groupby(day_col, sort=False):
                blocked[pd.Timestamp(day)] = frozenset(group[name_col].tolist())
        return cls(blocked, meta=meta)

    # -- 查询 ---------------------------------------------------------------
    def blocked_on(self, day) -> frozenset:
        return self._blocked.get(day, _EMPTY)

    def is_eligible(self, day, name) -> bool:
        return name not in self._blocked.get(day, _EMPTY)

    @property
    def n_blocked_cells(self) -> int:
        return int(sum(len(v) for v in self._blocked.values()))

    @property
    def n_days_with_block(self) -> int:
        return len(self._blocked)

    def __repr__(self) -> str:  # pragma: no cover - 诊断用
        return (f"EligibilityMask(days_with_block={self.n_days_with_block}, "
                f"blocked_cells={self.n_blocked_cells}, meta={self.meta})")


def _coerce_mask(eligible) -> EligibilityMask:
    if eligible is None:
        return EligibilityMask.all_true()
    if isinstance(eligible, EligibilityMask):
        return eligible
    if isinstance(eligible, Mapping):
        return EligibilityMask(eligible)
    raise TypeError("eligible 必须是 None、EligibilityMask 或 {day: set(PERMNO)} 映射")


def rule_e_long_only_returns(
    scores_by_day: dict,
    ret_by_day: dict,
    oc_by_day: dict,
    adv_by_day: dict,
    *,
    eligible=None,
    force_exit: bool = False,
    topn: int = 500,
    cost_bp: float = 8.0,
    exit_pct: float = 0.30,
    nt: int = 6,
    min_names: int = 50,
    collect_diagnostics: bool = False,
):
    """冻结构造 + Rule E 掩码。``eligible=None`` / 全 True ⇒ 与冻结函数逐位相同。

    参数
    ----
    scores_by_day / ret_by_day / oc_by_day / adv_by_day / topn / cost_bp /
    exit_pct / nt / min_names
        与 :func:`portfolio.construction.frozen_long_only_returns` 完全一致。
    eligible
        :class:`EligibilityMask` 或 ``{day: set(PERMNO)}``（被禁止的名字）或 ``None``。
        **只作用于当日新建仓的候选选择。**
    force_exit
        ``False``（Rule E 的定义，只禁进）时被掩码的已持仓名字**不因掩码退出**。
        ``True`` 只供 E3 的「禁进+强退」敏感性格；退出排名仍在全池上算，
        force_exit 只是在 `keep` 之后额外剔除被掩码的持仓名字。
    collect_diagnostics
        为 ``True`` 时额外返回过滤统计（被拦候选数、递补数、名额缺口、
        强退数、持仓名单等）。诊断量**不参与任何浮点运算**。

    返回
    ----
    ``DataFrame(index=date, columns=["r", "turn", "n_names"])``；
    ``collect_diagnostics=True`` 时返回 ``(df, diagnostics)``。
    """
    mask = _coerce_mask(eligible)
    ret, oc, adv = ret_by_day, oc_by_day, adv_by_day
    NT, EXIT_PCT = nt, exit_pct
    by_day = {d: s for d, s in scores_by_day.items() if len(s) >= min_names}

    rows = []
    diag_rows: list[dict] = []
    held_by_trade_date: dict = {}
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
        blocked = mask.blocked_on(day)
        j = i % NT
        prev = book[j]
        if prev is None:
            # 注入①：入选集合在可进名字中按分数排名取；名额上限仍是 k。
            nb = [p for p in order if p not in blocked][:k]
            fresh = set(nb)
            turn = 0.0
            keep = []
            n_keep_raw = n_forced = 0
            add_nomask: list = list(order[:k])
            n_slots = k
            add_used = nb
        else:
            keep_all = [p for p in prev if p in pct and pct[p] >= 1 - EXIT_PCT][:k]
            n_keep_raw = len(keep_all)
            if force_exit:
                keep = [p for p in keep_all if p not in blocked]
            else:
                keep = keep_all
            n_forced = n_keep_raw - len(keep)
            held = set(keep)
            n_slots = k - len(keep)
            # 诊断：同一 keep 下、无掩码时会被补入的名字（不进任何浮点运算）
            add_nomask = [p for p in order if p not in held][:n_slots]
            # 注入②：只在可进名字中递补，名额上限 k - len(keep) 不变。
            add = [p for p in order if p not in held and p not in blocked][:n_slots]
            nb, fresh = keep + add, set(add)
            turn = (k - len(keep)) / k
            add_used = add
        book[j] = nb
        if collect_diagnostics:
            nomask_set = set(add_nomask)
            used_set = set(add_used)
            prev_set = set(prev) if prev is not None else set()
            diag_rows.append({
                "signal_date": day,
                "sleeve": j,
                "n_pool": n,
                "k": k,
                "n_blocked_in_pool": len(blocked & elig) if blocked else 0,
                "n_keep_before_force_exit": n_keep_raw,
                "n_force_exited": n_forced,
                "n_slots": n_slots,
                "n_added": len(add_used),
                "n_slot_shortfall": n_slots - len(add_used),
                "n_blocked_candidates": sum(1 for p in add_nomask if p in blocked),
                "n_promoted": len(used_set - nomask_set),
                # §7.3：同一天既退出（不在 keep 里）又被递补进场的名字
                "n_reentered_after_exit": len(used_set & (prev_set - set(keep))),
                "n_book_names": len({p for sleeve in book if sleeve for p in sleeve}),
            })
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
        if collect_diagnostics:
            held_by_trade_date[nd] = frozenset(
                p for sleeve in book if sleeve for p in sleeve)
    out = (pd.DataFrame(rows, columns=["date", "r", "turn", "n_names"])
           .set_index("date")
           .sort_index())
    if not collect_diagnostics:
        return out
    diagnostics = {
        "per_day": pd.DataFrame(diag_rows),
        "held_by_trade_date": held_by_trade_date,
        "mask_meta": dict(mask.meta),
        "mask_blocked_cells": mask.n_blocked_cells,
        "mask_days_with_block": mask.n_days_with_block,
    }
    return out, diagnostics
