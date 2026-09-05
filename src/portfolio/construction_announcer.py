"""冻结纯多构造 + 公告者倾斜（exp15 专用）：B1 进场配额钩子、B2 事件套袖、组合合成。

**本模块不改 `construction.py`、不改 `construction_rule_e.py`**
（任务书 `docs/任务书_exp15_公告者倾斜_开发折估计_2026-09-05.md` §2.2）。
:func:`announcer_tilt_long_only_returns` 把
:func:`portfolio.construction.frozen_long_only_returns` 逐行复制一份，只在**进场
候选的取用处**插入配额钩子，其余一个字符都不动；注入方式沿用
`construction_rule_e.py` 的做法（该模块的表格式注入点说明是本模块的样板）。

注入点（行号对应 `src/portfolio/construction.py` 的冻结实现）
------------------------------------------------------------
| 冻结函数行号 | 语句 | 本模块的处理 |
|---|---|---|
| 150–162 | 池构造 / `n` / `pct` / `k` / `order` | **完全不动**：配额在 `n`、`pct`、`k`、`order` 之后才生效，故 `min_names`（158–159）与 benchmark（193）都不受配额影响 |
| 165–167 | `prev is None` 首次建仓分支 | 与 166 行的 `keep`/`add` 分支**合并为一条路径**（`prev is None` ⇒ `keep = []`、`n_slots = k`、`turn = 0.0`），空公告者集合下 `add == list(order[:k])`，与冻结逐位相同 |
| 169 | `keep = [p for p in prev if p in pct and pct[p] >= 1 - EXIT_PCT][:k]` | **一字未改**。退出排名仍在全池 `pct` 上算；公告者身份**从不**使已持仓退出 |
| 171 | `add = [p for p in order if p not in held][:k - len(keep)]` | **注入**：名额 `n_slots = k - len(keep)` 不变，但先按配额取 `need_ann` 个公告者、其余取非公告者，再**按 `order` 次序归并**成 `add` |
| 173 | `turn = (k - len(keep)) / k` | 不动（`keep` 未被配额触碰 ⇒ 同一 `prev` 下 `turn` 逐位相同） |
| 175–195 | 收益聚合 / 成本 / 基准 / 输出 | 不动 |

配额的定义（任务书 §4.1，固定项，本模块不得放宽）
--------------------------------------------------
信号日 t、套袖 j：池内「公告者」占比 ``s_t = n_ann / n``，配额
``k_E = round(k * s_t)``——**这是簿层配额**（`k` 是该档的持仓数），
不是名额层配额。已持仓里已经有 ``n_keep_ann`` 个公告者，故本次进场需要补的
公告者个数为 ``need_ann = min(max(k_E - n_keep_ann, 0), n_slots)``：

* ``k_E`` 用 `k` 而不是 `n_slots`，照任务书 §4.1 的字面公式；
* 已持仓的公告者**计入配额**，因为配额约束的是簿的公告者暴露（任务书 §4.1
  「把 Kronos 簿的公告者暴露从『系统性偏低』拉回池子的自然比例，**不超配**」）；
* ``max(..., 0)``：若 ``n_keep_ann > k_E``（配额已被留仓占满）则本次不补公告者，
  **但绝不强制卖出**——退出规则沿冻结（任务书 §4.1「退出规则、权重、时点、成本
  全部沿冻结构造」）。这是「不超配」在只有进场一个动作时能做到的全部。

`round` 是 Python 内建的**银行家舍入**（.5 取偶）。这是实现选择，须披露；
`k * s_t` 恰为半整数的情形在真实数据上罕见，诊断列 `k_E` 可核。

**公告者集合为空时逐位重现冻结输出**：``s_t = 0`` ⇒ ``k_E = 0`` ⇒
``need_ann = 0`` ⇒ ``add = [p for p in order if p not in held][:n_slots]``
（== 冻结第 171 行；``prev is None`` 时 == ``list(order[:k])``，即冻结第 166 行），
``keep``、``turn``、``fresh``、``vals`` 的元素与顺序全部相同 ⇒
`r`/`turn`/`n_names` 三列逐位相同。`tests/test_exp15_construction_announcer.py`
用 ``np.array_equal`` 断言。

为什么 `add` 要按 `order` 归并
------------------------------
两组（公告者 / 非公告者）各自沿冻结 `order` 取，合并次序任务书未指定。
本模块按**全局 `order` 位置**归并，理由有二：(a) 公告者集合为空时退化为冻结的
`add` 切片，逐位重现才成立；(b) `nb = keep + add` 的列表顺序进入
``np.mean`` 的成对求和顺序（`construction.py` docstring 「浮点逐位一致的三个
脆弱点」第 3 条），按 `order` 归并是与冻结实现最接近的次序。

B2 事件套袖（任务书 §4.2）
--------------------------
:func:`event_sleeve_returns` 与冻结构造**没有共享状态**：它是一条独立的资金腿，
只借用冻结构造给出的「当日池」与「当日同池等权基准」（由
:func:`announcer_tilt_long_only_returns` 的 ``collect_diagnostics`` 返回）。
持有窗 = ``[e-1, e+1]`` 三个交易日；``e-1`` 开盘买入（该日只吃 open→close，
`CLAUDE.md` §一.4）、``e+1`` 收盘卖出；套袖内**逐日等权**；成本 8bp 单边、
每个名字进出各一次。
"""
from __future__ import annotations

from typing import AbstractSet, Any, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "AnnouncerFlags",
    "announcer_tilt_long_only_returns",
    "event_sleeve_returns",
    "combine_active_returns",
]

_EMPTY: frozenset = frozenset()


class AnnouncerFlags:
    """按 (signal_date, PERMNO) 的「公告者」标记；未出现的格子 = 非公告者。

    内部只存**是公告者**的名字（稀疏）：exp14 E2 实测触发率约 11%，稠密表在
    7 折 × ~500 名字 × ~880 日上没有必要。
    """

    __slots__ = ("_flags", "meta")

    def __init__(self, announcers_by_day: Mapping[Any, AbstractSet] | None = None,
                 meta: dict | None = None) -> None:
        self._flags: dict = {
            pd.Timestamp(day): frozenset(names)
            for day, names in (announcers_by_day or {}).items()
            if len(names) > 0
        }
        self.meta: dict = dict(meta or {})

    @classmethod
    def none(cls) -> "AnnouncerFlags":
        """全 False；与 ``announcers=None`` 等价，用于逐位重现测试。"""
        return cls({})

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, *, day_col: str = "signal_date",
                   name_col: str = "PERMNO", meta: dict | None = None) -> "AnnouncerFlags":
        """由「只列出公告者格子」的稀疏表构造（本模块的常用入口）。"""
        flags: dict = {}
        if len(frame):
            for day, group in frame.groupby(day_col, sort=False):
                flags[pd.Timestamp(day)] = frozenset(group[name_col].tolist())
        return cls(flags, meta=meta)

    def announcers_on(self, day) -> frozenset:
        return self._flags.get(day, _EMPTY)

    def is_announcer(self, day, name) -> bool:
        return name in self._flags.get(day, _EMPTY)

    @property
    def n_flag_cells(self) -> int:
        return int(sum(len(v) for v in self._flags.values()))

    @property
    def n_days_with_flag(self) -> int:
        return len(self._flags)

    def __repr__(self) -> str:  # pragma: no cover - 诊断用
        return (f"AnnouncerFlags(days={self.n_days_with_flag}, "
                f"cells={self.n_flag_cells}, meta={self.meta})")


def _coerce_flags(announcers) -> AnnouncerFlags:
    if announcers is None:
        return AnnouncerFlags.none()
    if isinstance(announcers, AnnouncerFlags):
        return announcers
    if isinstance(announcers, Mapping):
        return AnnouncerFlags(announcers)
    raise TypeError("announcers 必须是 None、AnnouncerFlags 或 {day: set(PERMNO)} 映射")


def announcer_tilt_long_only_returns(
    scores_by_day: dict,
    ret_by_day: dict,
    oc_by_day: dict,
    adv_by_day: dict,
    *,
    announcers=None,
    apply_quota: bool = True,
    topn: int = 500,
    cost_bp: float = 8.0,
    exit_pct: float = 0.30,
    nt: int = 6,
    min_names: int = 50,
    collect_diagnostics: bool = False,
):
    """冻结构造 + B1 公告者进场配额。``announcers=None`` / 全空 ⇒ 与冻结函数逐位相同。

    参数
    ----
    scores_by_day / ret_by_day / oc_by_day / adv_by_day / topn / cost_bp /
    exit_pct / nt / min_names
        与 :func:`portfolio.construction.frozen_long_only_returns` 完全一致。
    announcers
        :class:`AnnouncerFlags` 或 ``{day: set(PERMNO)}``（该日的公告者）或 ``None``。
        **只作用于当日新建仓的候选选择。**
    apply_quota
        ``False`` 时**只统计不倾斜**：进场沿冻结第 166/171 行，输出无论旗标如何
        都与冻结函数逐位相同，但诊断里仍记录「E0 的簿里有几个公告者」——
        这是 B1「把暴露拉回自然比例」的对照量，**纯诊断开关**。
    collect_diagnostics
        为 ``True`` 时额外返回 ``(df, diagnostics)``；诊断量（含每个成交日的
        **同池等权基准**与**池名单**，供 B2 套袖使用）**不参与任何浮点运算**。

    返回
    ----
    ``DataFrame(index=date, columns=["r", "turn", "n_names"])``；
    ``collect_diagnostics=True`` 时返回 ``(df, diagnostics)``。
    """
    flags = _coerce_flags(announcers)
    ret, oc, adv = ret_by_day, oc_by_day, adv_by_day
    NT, EXIT_PCT = nt, exit_pct
    by_day = {d: s for d, s in scores_by_day.items() if len(s) >= min_names}

    rows = []
    diag_rows: list[dict] = []
    bench_by_trade_date: dict = {}
    pool_by_trade_date: dict = {}
    signal_day_of_trade_date: dict = {}
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
        ann = flags.announcers_on(day)
        # 池内公告者占比 s_t 与簿层配额 k_E（任务书 §4.1）
        n_ann = sum(1 for p in order if p in ann) if ann else 0
        s_t = n_ann / n
        k_E = int(round(k * s_t))
        j = i % NT
        prev = book[j]
        if prev is None:
            keep = []
            turn = 0.0
        else:
            # 冻结第 169 行，一字未改
            keep = [p for p in prev if p in pct and pct[p] >= 1 - EXIT_PCT][:k]
            turn = (k - len(keep)) / k
        held = set(keep)
        n_slots = k - len(keep)
        n_keep_ann = sum(1 for p in keep if p in ann) if ann else 0
        need_ann = min(max(k_E - n_keep_ann, 0), n_slots) if apply_quota else 0
        if not ann or not apply_quota:
            # 无公告者（或纯诊断模式）⇒ 退化为冻结第 171 行
            # （prev is None 时即冻结第 166 行）
            add = [p for p in order if p not in held][:n_slots]
            n_add_ann = sum(1 for p in add if p in ann) if ann else 0
            shortfall_ann = 0
        else:
            add_ann = [p for p in order if p not in held and p in ann][:need_ann]
            shortfall_ann = need_ann - len(add_ann)
            n_rest = n_slots - len(add_ann)
            add_non = [p for p in order if p not in held and p not in ann][:n_rest]
            chosen = set(add_ann) | set(add_non)
            if len(chosen) < n_slots:
                # 两组都取不满（结构上不应发生；留守卫并计入诊断）
                extra = [p for p in order if p not in held and p not in chosen][
                    :n_slots - len(chosen)]
                chosen |= set(extra)
            add = [p for p in order if p in chosen]
            n_add_ann = len(add_ann)
        nb, fresh = keep + add, set(add)
        book[j] = nb
        if collect_diagnostics:
            # 同日反事实：同一个 `keep` 下，冻结第 171 行本会补入的名字。
            # 纯诊断，不参与任何浮点运算（照 construction_rule_e.py 的 add_nomask）。
            add_nomask = [p for p in order if p not in held][:n_slots]
            nomask_set, used_set = set(add_nomask), set(add)
            diag_rows.append({
                "signal_date": day,
                "sleeve": j,
                "n_pool": n,
                "k": k,
                "n_announcers_in_pool": n_ann,
                "announcer_share_of_pool": s_t,
                "k_E": k_E,
                "n_keep": len(keep),
                "n_keep_announcers": n_keep_ann,
                "n_slots": n_slots,
                "need_announcers": need_ann,
                "n_added": len(add),
                "n_added_announcers": n_add_ann,
                # 同日反事实（冻结进场 vs 配额进场）
                "n_added_announcers_frozen": (sum(1 for p in add_nomask if p in ann)
                                              if ann else 0),
                "n_quota_promoted": len(used_set - nomask_set),
                "n_quota_displaced": len(nomask_set - used_set),
                "announcer_shortfall": shortfall_ann,
                "n_slot_shortfall": n_slots - len(add),
                "n_book_announcers": sum(1 for p in nb if p in ann) if ann else 0,
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
            bench_by_trade_date[nd] = bench
            pool_by_trade_date[nd] = frozenset(pct)
            signal_day_of_trade_date[nd] = day
            held_by_trade_date[nd] = frozenset(
                p for sleeve in book if sleeve for p in sleeve)
    out = (pd.DataFrame(rows, columns=["date", "r", "turn", "n_names"])
           .set_index("date")
           .sort_index())
    if not collect_diagnostics:
        return out
    diagnostics = {
        "per_day": pd.DataFrame(diag_rows),
        "bench_by_trade_date": bench_by_trade_date,
        "pool_by_trade_date": pool_by_trade_date,
        "signal_day_of_trade_date": signal_day_of_trade_date,
        "held_by_trade_date": held_by_trade_date,
        "flags_meta": dict(flags.meta),
        "flag_cells": flags.n_flag_cells,
        "days_with_flag": flags.n_days_with_flag,
    }
    return out, diagnostics


# ---------------------------------------------------------------------------
# B2 事件套袖（任务书 §4.2）
# ---------------------------------------------------------------------------
def event_sleeve_returns(
    events: pd.DataFrame,
    ret_by_day: dict,
    oc_by_day: dict,
    *,
    bench_by_trade_date: Mapping,
    trade_dates: Sequence,
    cost_bp: float = 8.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """三日事件套袖的逐日收益与逐笔毛主动收益。

    参数
    ----
    events
        列 ``PERMNO / d_entry / d_event / d_exit``（三个交易日；调用方负责
        保证 ``d_entry`` 是 ``d_event`` 的前一交易日、``d_exit`` 是后一交易日，
        并已做过池内 / 时点筛选）。**必须已按 (PERMNO, d_event) 去重。**
    ret_by_day / oc_by_day
        与冻结构造同构 ``{Timestamp: {PERMNO: float}}``。
    bench_by_trade_date
        每个成交日的同池等权基准（来自
        :func:`announcer_tilt_long_only_returns` 的诊断）。
    trade_dates
        套袖产出收益的日期集合（= E0 的成交日索引）。落在其外的持有日不计入
        逐日序列，调用方按返回的 ``truncated_legs`` 披露截断量。
    cost_bp
        单边成本；每个名字在**实际贡献的第一天**扣一次、**最后一天**再扣一次。

    返回
    ----
    ``(daily, per_trade)``：

    * ``daily``：index=date，列
      ``gross``（当日等权毛收益，无名字为 0）、``bench``、``n_names``、
      ``n_entries``、``n_exits``、``cost_coef``（= ``(n_entries+n_exits)/n_names``，
      每 1bp 的日拖累 = ``cost_coef/1e4``）、``cost``（``cost_bp`` 下的拖累）、
      ``active``（= ``gross - bench - cost``）。
    * ``per_trade``：每笔 ``PERMNO / d_event / d_first / d_last / n_days /
      gross_compounded / bench_compounded / gross_active``（复利口径，
      ``[e-1 开盘 → e+1 收盘]`` 相对同期同池等权基准）。
    """
    valid_days = set(pd.Timestamp(d) for d in trade_dates)
    legs: dict = {}          # date -> list of (PERMNO, value)
    per_trade_rows: list[dict] = []
    first_day: dict = {}     # date -> count of names entering
    last_day: dict = {}
    truncated_legs = 0
    dropped_events = 0

    for row in events.itertuples(index=False):
        permno = row.PERMNO
        window = ((pd.Timestamp(row.d_entry), "oc"),
                  (pd.Timestamp(row.d_event), "ret"),
                  (pd.Timestamp(row.d_exit), "ret"))
        contributions: list[tuple[pd.Timestamp, float]] = []
        for day, kind in window:
            if day not in valid_days:
                truncated_legs += 1
                continue
            source = oc_by_day if kind == "oc" else ret_by_day
            value = source.get(day, {}).get(permno)
            if value is None or not np.isfinite(value):
                truncated_legs += 1
                continue
            contributions.append((day, float(value)))
        if not contributions:
            dropped_events += 1
            continue
        for day, value in contributions:
            legs.setdefault(day, []).append((permno, value))
        d_first, d_last = contributions[0][0], contributions[-1][0]
        first_day[d_first] = first_day.get(d_first, 0) + 1
        last_day[d_last] = last_day.get(d_last, 0) + 1
        gross_c = float(np.prod([1.0 + v for _, v in contributions]) - 1.0)
        bench_c = float(np.prod([1.0 + float(bench_by_trade_date[d])
                                 for d, _ in contributions]) - 1.0)
        per_trade_rows.append({
            "PERMNO": permno,
            "d_event": pd.Timestamp(row.d_event),
            "d_first": d_first,
            "d_last": d_last,
            "n_days": len(contributions),
            "gross_compounded": gross_c,
            "bench_compounded": bench_c,
            "gross_active": gross_c - bench_c,
        })

    rows = []
    for day in sorted(valid_days):
        members = legs.get(day, [])
        m = len(members)
        n_in = first_day.get(day, 0)
        n_out = last_day.get(day, 0)
        gross = float(np.mean([v for _, v in members])) if m else 0.0
        cost_coef = (n_in + n_out) / m if m else 0.0
        bench = float(bench_by_trade_date[day])
        cost = cost_bp / 1e4 * cost_coef
        rows.append((day, gross, bench, m, n_in, n_out, cost_coef, cost,
                     gross - bench - cost))
    daily = (pd.DataFrame(rows, columns=["date", "gross", "bench", "n_names",
                                         "n_entries", "n_exits", "cost_coef",
                                         "cost", "active"])
             .set_index("date").sort_index())
    daily.attrs["truncated_legs"] = int(truncated_legs)
    daily.attrs["dropped_events"] = int(dropped_events)
    per_trade = pd.DataFrame(per_trade_rows)
    return daily, per_trade


def combine_active_returns(
    base_gross: pd.Series,
    base_turn: pd.Series,
    sleeve_daily: pd.DataFrame,
    *,
    w_event: float,
    nt: int,
) -> pd.DataFrame:
    """``(1 - w) × E0 主动 + w × 套袖主动``，并折算成 `summarize` 能吃的 (gross, turn)。

    `nt5_baseline_readout.summarize` 用 ``drag(bp) = 2 * bp/1e4 * turn / nt``
    把毛收益折成任意成本档的净收益。组合腿的每 bp 拖累是

        ``(1 - w) * 2 * turn_E0 / (1e4 * nt) + w * cost_coef / 1e4``

    故等效换手 ``turn_eff = (1 - w) * turn_E0 + w * nt / 2 * cost_coef``
    让 `summarize` 的成本网格对组合腿**恒等成立**（调用方用 8bp 直算值对拍）。

    返回列：``gross``（组合毛主动）、``turn``（等效换手）、``sleeve_active_gross``、
    ``base_gross``、``cost_coef``。
    """
    if not (0.0 <= w_event <= 1.0):
        raise ValueError("w_event 必须在 [0, 1]")
    joined = pd.DataFrame({
        "base_gross": base_gross.astype(float),
        "base_turn": base_turn.astype(float),
    }).join(sleeve_daily[["gross", "bench", "cost_coef"]], how="left")
    if joined[["gross", "bench", "cost_coef"]].isna().any().any():
        raise ValueError("套袖逐日序列缺少 E0 成交日；两者的日期集合必须一致")
    sleeve_active_gross = joined["gross"] - joined["bench"]
    out = pd.DataFrame({
        "gross": (1.0 - w_event) * joined["base_gross"] + w_event * sleeve_active_gross,
        "turn": ((1.0 - w_event) * joined["base_turn"]
                 + w_event * nt / 2.0 * joined["cost_coef"]),
        "sleeve_active_gross": sleeve_active_gross,
        "base_gross": joined["base_gross"],
        "cost_coef": joined["cost_coef"],
    })
    return out
