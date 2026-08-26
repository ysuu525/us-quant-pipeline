"""标签引擎：execution-return engine（规范 §4，总财富收益）。

标签 = t+1 开盘建仓、t+6 开盘退出的 total return，五段复合：

1. 建仓：t+1 实际 ``DlyOpen``（未复权原始价）。t+1 无有效 open →
   unfillable，按原因归类（停牌 vs 退市），不删除、不假设成交；
2. 首日段：``DlyClose(t+1)/DlyOpen(t+1) − 1``（建仓在 ex 之后，不含 t+1 股息）；
3. 中段：t+2 … t+5 逐日复合 CIZ ``DlyRet``（close-to-close 含息收益）；
4. 退出段：t+5 close → t+6 open 隔夜价格差；t+6 为 ex-date 时
   ``(DlyOpen(t+6) + Div(t+6)) / DlyClose(t+5) − 1``；
5. 退市接管：持有期内出现 ``DlyDelFlg='Y'`` → 从该点起复合至退市终值记录的
   ``DlyRet``，期末即为终值，不再要求 t+6 open 存在。

硬约束（§3 双面板分离）：**本模块的输入必须是未过滤的全量收益面板**。
选股面板的 active 过滤会滤掉 ``DlyDelFlg='Y'`` 的退市终值记录，用过滤后
面板算标签会系统性丢失退市收益。

CIZ 语义（§9）：退市记录上的 ``DlyRet`` 已并入退市收益，禁止再套
``(1+RET)(1+DLRET)−1``。``DlyRet`` 缺失的业绩类退市按 Shumway 插补
（−30%/−55%）作敏感性假设，插补只允许发生在业绩类退市码上。

未定义即报告：中段 ``DlyRet`` 缺失且非退市（长停牌等）、退出段价格缺失且
非退市——这些情形规范未定义填补规则，一律标 INVALID 并给出 reason，
由调用方按年报告占比，禁止静默填补。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from .calendar import CalendarError, TradingCalendar

# ---------------------------------------------------------------- 状态与原因

STATUS_OK = "ok"
STATUS_UNFILLABLE = "unfillable"
STATUS_INVALID = "invalid"

# unfillable 原因（§4 第 1 条：停牌 vs 退市）
REASON_HALTED = "halted"                    # t+1 无有效 open，但证券未退市
REASON_DELISTED = "delisted"                # t+1 时证券已退市 / 已无记录

# invalid 原因（可计算性问题，按年报告，不进训练集）
REASON_CALENDAR_END = "calendar_end"        # t+6 超出日历（数据截止附近的信号日）
REASON_MISSING_ENTRY_CLOSE = "missing_entry_close"
REASON_MISSING_MID_RET = "missing_mid_ret"  # 中段 DlyRet 缺失且非退市记录
REASON_MISSING_DELIST_RET = "missing_delist_ret"  # 退市终值 DlyRet 缺失且不允许插补
REASON_MISSING_EXIT_PRICE = "missing_exit_price"  # t+5 close 或 t+6 open 缺失且非退市


@dataclass
class LabelResult:
    permno: int
    signal_date: pd.Timestamp
    status: str
    reason: str | None = None
    label: float = np.nan
    # 诊断字段
    entry_date: pd.Timestamp | None = None
    entry_open: float = np.nan
    delist_takeover: bool = False        # 是否走了退市接管路径
    delist_date: pd.Timestamp | None = None
    imputed: bool = False                # 退市终值收益是否用了 Shumway 插补
    exit_dividend: float = 0.0           # 退出段计入的 t+6 现金股息（每股）
    segments: dict = field(default_factory=dict)


def _valid_price(x) -> bool:
    return x is not None and pd.notna(x) and x > 0


def _is_delist_row(row: pd.Series) -> bool:
    return str(row.get("DlyDelFlg", "")).upper() == "Y"


# ---------------------------------------------------------------- 单样本引擎

def compute_label(
    sec: pd.DataFrame,
    signal_date,
    calendar: TradingCalendar,
    dividends: Mapping[pd.Timestamp, float] | None = None,
    predict_window: int = 6,
    delist_imputation: float | None = None,
    is_performance_delist: Callable[[pd.Series], bool] | None = None,
    permno: int = -1,
) -> LabelResult:
    """对单只证券的单个信号日 t 计算 execution-return 标签。

    Parameters
    ----------
    sec : 该 PERMNO 在**未过滤全量面板**中的全部日行，index 为 DlyCalDt
        （DatetimeIndex），至少含列 DlyOpen / DlyClose / DlyRet / DlyDelFlg。
    dividends : {ex_date -> 每股现金股息}。仅用于退出段 t+6 为 ex-date 的情形。
    delist_imputation : Shumway 敏感性档位（如 -0.30 / -0.55）。None = 不插补，
        业绩类退市且 DlyRet 缺失时标 INVALID。
    is_performance_delist : 判定退市记录是否属于业绩类退市码的谓词
        （入参为该退市日的整行）。插补只发生在业绩类退市码上（§10 不变量）；
        未提供谓词时一律不插补。CIZ 具体退市码集合待真实数据核对后在调用方冻结。
    """
    t = pd.Timestamp(signal_date)
    dividends = dividends or {}

    def _res(**kw) -> LabelResult:
        return LabelResult(permno=permno, signal_date=t, **kw)

    # ---- 持有期日期：t+1 .. t+predict_window，按交易日历显式偏移 ----
    try:
        hold = [calendar.shift(t, k) for k in range(1, predict_window + 1)]
    except CalendarError:
        return _res(status=STATUS_INVALID, reason=REASON_CALENDAR_END)
    t1, t_exit = hold[0], hold[-1]

    # ---- 1. 建仓：t+1 实际 DlyOpen ----
    row1 = sec.loc[t1] if t1 in sec.index else None
    if row1 is None or not _valid_price(row1.get("DlyOpen")):
        # unfillable：区分停牌 vs 退市。退市 = t+1 时已出现退市记录，或已无任何后续记录
        delisted_before = any(
            _is_delist_row(r) for d, r in sec.iterrows() if d <= t1
        ) or (len(sec.index) > 0 and sec.index.max() < t1)
        reason = REASON_DELISTED if delisted_before else REASON_HALTED
        return _res(status=STATUS_UNFILLABLE, reason=reason, entry_date=t1)

    entry_open = float(row1["DlyOpen"])

    # ---- 2. 首日段：t+1 open → t+1 close（价格收益，不含 t+1 股息） ----
    if not _valid_price(row1.get("DlyClose")):
        return _res(
            status=STATUS_INVALID, reason=REASON_MISSING_ENTRY_CLOSE,
            entry_date=t1, entry_open=entry_open,
        )
    seg_day1 = float(row1["DlyClose"]) / entry_open - 1.0
    gross = 1.0 + seg_day1
    segments = {"day1": seg_day1, "mid": [], "exit": None}

    # t+1 本身是退市终值记录：open→close 已复合到终值，到此为止。
    # 注：假定退市记录的 DlyClose 即终值（与 DlyRet 口径一致）；此语义待真实
    # golden fixture（§10 Lehman 2008）核对，若不一致在此处修正。
    if _is_delist_row(row1):
        return _res(
            status=STATUS_OK, label=gross - 1.0, entry_date=t1, entry_open=entry_open,
            delist_takeover=True, delist_date=t1, segments=segments,
        )

    # ---- 3. 中段：t+2 … t+5 逐日复合 DlyRet；5. 退市接管 ----
    for d in hold[1:-1]:
        row = sec.loc[d] if d in sec.index else None
        if row is None:
            # 无行且未退市（长停牌/数据缺口）：规范未定义填补规则 → INVALID
            return _res(
                status=STATUS_INVALID, reason=REASON_MISSING_MID_RET,
                entry_date=t1, entry_open=entry_open, segments=segments,
            )
        ret = row.get("DlyRet")
        if _is_delist_row(row):
            ret, imputed = _delist_ret(
                row, ret, delist_imputation, is_performance_delist
            )
            if ret is None:
                return _res(
                    status=STATUS_INVALID, reason=REASON_MISSING_DELIST_RET,
                    entry_date=t1, entry_open=entry_open,
                    delist_date=d, segments=segments,
                )
            gross *= 1.0 + float(ret)
            segments["mid"].append(float(ret))
            return _res(
                status=STATUS_OK, label=gross - 1.0, entry_date=t1,
                entry_open=entry_open, delist_takeover=True, delist_date=d,
                imputed=imputed, segments=segments,
            )
        if pd.isna(ret):
            return _res(
                status=STATUS_INVALID, reason=REASON_MISSING_MID_RET,
                entry_date=t1, entry_open=entry_open, segments=segments,
            )
        gross *= 1.0 + float(ret)
        segments["mid"].append(float(ret))

    # ---- 4. 退出段：t+5 close → t+6 open（或 t+6 退市接管） ----
    t5 = hold[-2]
    row5 = sec.loc[t5]  # 中段循环已保证该行存在
    row6 = sec.loc[t_exit] if t_exit in sec.index else None

    if row6 is not None and _is_delist_row(row6):
        # 退市接管发生在 t+6：复合该记录的 DlyRet（close(t+5)→终值，含退市收益）
        ret, imputed = _delist_ret(
            row6, row6.get("DlyRet"), delist_imputation, is_performance_delist
        )
        if ret is None:
            return _res(
                status=STATUS_INVALID, reason=REASON_MISSING_DELIST_RET,
                entry_date=t1, entry_open=entry_open,
                delist_date=t_exit, segments=segments,
            )
        gross *= 1.0 + float(ret)
        segments["exit"] = float(ret)
        return _res(
            status=STATUS_OK, label=gross - 1.0, entry_date=t1,
            entry_open=entry_open, delist_takeover=True, delist_date=t_exit,
            imputed=imputed, segments=segments,
        )

    if (
        row6 is None
        or not _valid_price(row6.get("DlyOpen"))
        or not _valid_price(row5.get("DlyClose"))
    ):
        return _res(
            status=STATUS_INVALID, reason=REASON_MISSING_EXIT_PRICE,
            entry_date=t1, entry_open=entry_open, segments=segments,
        )

    # t+6 为 ex-date：过夜持仓有权获得该股息（v1.2 新增，取自 distribution 表）
    div = float(dividends.get(t_exit, 0.0))
    seg_exit = (float(row6["DlyOpen"]) + div) / float(row5["DlyClose"]) - 1.0
    gross *= 1.0 + seg_exit
    segments["exit"] = seg_exit

    return _res(
        status=STATUS_OK, label=gross - 1.0, entry_date=t1, entry_open=entry_open,
        exit_dividend=div, segments=segments,
    )


def _delist_ret(row, ret, imputation, is_performance_delist):
    """退市终值记录的收益：优先用 CIZ DlyRet（已并入退市收益）；缺失时仅当
    (a) 提供了插补档位且 (b) 该记录判定为业绩类退市码，才用 Shumway 插补。
    返回 (ret 或 None, 是否插补)。"""
    if pd.notna(ret):
        return float(ret), False
    if imputation is not None and is_performance_delist is not None and is_performance_delist(row):
        return float(imputation), True
    return None, False


# ---------------------------------------------------------------- 批量接口

def compute_labels(
    full_panel: pd.DataFrame,
    signal_obs: pd.DataFrame,
    calendar: TradingCalendar,
    cash_dividends: pd.DataFrame | None = None,
    predict_window: int = 6,
    delist_imputation: float | None = None,
    is_performance_delist: Callable[[pd.Series], bool] | None = None,
    permno_col: str = "PERMNO",
    date_col: str = "DlyCalDt",
) -> pd.DataFrame:
    """批量计算标签。

    Parameters
    ----------
    full_panel : **未过滤全量收益面板**（§3），含 PERMNO / DlyCalDt /
        DlyOpen / DlyClose / DlyRet / DlyDelFlg（+退市码列）。
    signal_obs : 两列 (PERMNO, signal_date)，来自选股面板「t 日入选」的观测。
    cash_dividends : 现金股息表，列 (PERMNO, ex_date, div_cash)。由调用方从
        CIZ distribution 表按现金股息事件类型预先筛好（具体 CIZ 事件码待
        真实数据核对后冻结）；本引擎不做事件类型判断。
    """
    div_by_permno: dict[int, dict[pd.Timestamp, float]] = {}
    if cash_dividends is not None and len(cash_dividends) > 0:
        cd = cash_dividends.copy()
        cd["ex_date"] = pd.to_datetime(cd["ex_date"])
        for pn, g in cd.groupby(permno_col):
            div_by_permno[pn] = g.groupby("ex_date")["div_cash"].sum().to_dict()

    results = []
    panel = full_panel.copy()
    panel[date_col] = pd.to_datetime(panel[date_col])
    by_permno = {pn: g.set_index(date_col).sort_index() for pn, g in panel.groupby(permno_col)}

    for pn, t in signal_obs[[permno_col, "signal_date"]].itertuples(index=False):
        sec = by_permno.get(pn)
        if sec is None:
            sec = pd.DataFrame(columns=panel.columns)
        r = compute_label(
            sec, t, calendar,
            dividends=div_by_permno.get(pn),
            predict_window=predict_window,
            delist_imputation=delist_imputation,
            is_performance_delist=is_performance_delist,
            permno=pn,
        )
        results.append(r)

    return pd.DataFrame(
        {
            permno_col: [r.permno for r in results],
            "signal_date": [r.signal_date for r in results],
            "status": [r.status for r in results],
            "reason": [r.reason for r in results],
            "label": [r.label for r in results],
            "delist_takeover": [r.delist_takeover for r in results],
            "imputed": [r.imputed for r in results],
            "exit_dividend": [r.exit_dividend for r in results],
        }
    )


def unfillable_report(labels: pd.DataFrame) -> pd.DataFrame:
    """unfillable / invalid 样本按年 × 原因统计占比（§4 / §10：按年报告）。"""
    df = labels.copy()
    df["year"] = pd.to_datetime(df["signal_date"]).dt.year
    total = df.groupby("year").size().rename("n_total")
    bad = (
        df[df["status"] != STATUS_OK]
        .groupby(["year", "status", "reason"])
        .size()
        .rename("n")
        .reset_index()
        .merge(total, on="year")
    )
    bad["share"] = bad["n"] / bad["n_total"]
    return bad
