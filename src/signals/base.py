"""信号层的公共契约：规格（可冻结、可哈希）与抽象基类。

设计目标只有一个——**让「这个信号到底是怎么算的」在登记簿里可核对**。
本项目已实测：看数据选参的偏差 +0.0070，比信号本身的三分之一还大
（CLAUDE.md §一.2）。对策不是「少调参」，是**把参数在看数据之前写死并留指纹**。
:class:`SignalSpec` 就是那枚指纹：预注册时把 ``spec_hash()`` 抄进
``experiments/ledger.md``，事后任何一处参数改动都会让哈希对不上。

前视禁令（CLAUDE.md §一.1）
--------------------------
``Signal.compute`` 的输出，**每一行只能用 ≤ signal_date 的数据**。

自查方法（截断不变性，truncation invariance）——本模块提供可执行版
:func:`assert_no_lookahead`：

1. 取一个截止日 ``T``（落在样本中段）；
2. 在**完整面板**上算一次 ``compute``；
3. 把面板裁到 ``DlyCalDt <= T`` 再算一次；
4. 两次结果在 ``signal_date <= T`` 的行上必须**逐格相同**（NaN 视为相同）。

若不同，说明某处用到了 T 之后的信息（典型来源：整列 ``transform`` 的截面统计量、
``bfill``、``rolling(center=True)``、先 dropna 再滚动导致窗口跨越未来行）。

更严的一版（测试里用）：不裁剪，而是把 ``T`` 之后的行**改成别的数**，
``signal_date <= T`` 的输出仍必须一字不变。改数比裁剪更能抓住「用了未来行但不改变行数」
的错误。
"""
from __future__ import annotations

import abc
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

# 长表面板的最小列集合（§4 未复权原始面板的列名口径）
REQUIRED_PANEL_COLUMNS: tuple[str, ...] = (
    "PERMNO", "DlyCalDt", "DlyOpen", "DlyClose", "DlyRet",
)
# compute() 输出的最小列集合
OUTPUT_COLUMNS: tuple[str, ...] = ("signal_date", "PERMNO", "score")


def _freeze_params(params: Any) -> tuple[tuple[str, object], ...]:
    """把 dict / 序列统一成有序且可哈希的 ``((k, v), ...)``。"""
    if params is None:
        return ()
    if isinstance(params, Mapping):
        items: Iterable = params.items()
    else:
        items = params
    out: list[tuple[str, object]] = []
    for it in items:
        if not isinstance(it, Sequence) or isinstance(it, (str, bytes)) or len(it) != 2:
            raise TypeError(f"params 项必须是 (name, value) 二元组，收到 {it!r}")
        k, v = it
        if not isinstance(k, str):
            raise TypeError(f"params 的键必须是 str，收到 {k!r}")
        if isinstance(v, (list, set)):
            v = tuple(sorted(v)) if isinstance(v, set) else tuple(v)
        out.append((k, v))
    names = [k for k, _ in out]
    if len(set(names)) != len(names):
        raise ValueError(f"params 出现重名键：{names}")
    return tuple(out)


@dataclass(frozen=True)
class SignalSpec:
    """一个信号的**冻结规格**。改任何一个字段都会改变 :meth:`spec_hash`。

    Attributes
    ----------
    name : str
        信号标识（登记簿里的名字，稳定不改）。
    version : str
        规格版本。口径变更必须**升版本**，不得原地改 v1。
    horizon_days : int
        该信号登记的预测视野（交易日）。本项目冻结标签为 6 日执行收益，
        因此与 Kronos 并列读数的信号一律登记 ``horizon_days=6``——
        即使信号本身是月频构造，评估口径必须同为 6 日标签（调研 B 的稀释门槛
        就是在 6 日标签上算的）。
    params : tuple[tuple[str, object], ...]
        有序、可哈希的参数表。**所有**影响数值的常量都必须在此登记，
        包括「零参数」宣称下的隐性选择（winsorize 与否、缺失处理、是否 z-score）——
        调研 B 隐性污染渠道 ④：「零参数」通常是假的。
    source_ref : str
        文献出处。用于区分「先验固定」与「在本数据上调出来」。
    """

    name: str
    version: str
    horizon_days: int
    params: tuple[tuple[str, object], ...] = field(default=())
    source_ref: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", _freeze_params(self.params))
        if not self.name:
            raise ValueError("SignalSpec.name 不得为空")
        if not self.version:
            raise ValueError("SignalSpec.version 不得为空")
        if int(self.horizon_days) <= 0:
            raise ValueError("SignalSpec.horizon_days 必须为正整数")
        object.__setattr__(self, "horizon_days", int(self.horizon_days))

    # ------------------------------------------------------------------ 序列化
    def as_dict(self) -> dict:
        """规格的 JSON 友好视图（params 保序，序列化为 [k, v] 列表）。"""
        return {
            "name": self.name,
            "version": self.version,
            "horizon_days": self.horizon_days,
            "params": [[k, v] for k, v in self.params],
            "source_ref": self.source_ref,
        }

    def canonical_json(self) -> str:
        """规范 JSON：外层键排序、无多余空白、非 JSON 类型退化为 str。

        params 保持**登记顺序**（不排序）——顺序本身是规格的一部分，
        换序视为换规格。
        """
        return json.dumps(
            self.as_dict(), sort_keys=True, ensure_ascii=False,
            separators=(",", ":"), default=str,
        )

    def spec_hash(self) -> str:
        """规格的 SHA-256 指纹（十六进制）。预注册时抄进登记簿。"""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def param(self, key: str, default: object = None) -> object:
        for k, v in self.params:
            if k == key:
                return v
        return default

    def __str__(self) -> str:  # pragma: no cover - 便于日志
        return f"{self.name}/{self.version}#{self.spec_hash()[:12]}"


class Signal(abc.ABC):
    """横截面信号的抽象基类。

    契约
    ----
    输入 ``panel``：长表，列至少含 :data:`REQUIRED_PANEL_COLUMNS`
    （``PERMNO, DlyCalDt, DlyOpen, DlyClose, DlyRet``），**未复权原始面板**口径。
    允许多带列（如 ``DlyPrcVol``），实现方按需取用。

    输出：长表，列至少含 ``signal_date, PERMNO, score``，可带额外诊断列。
    ``signal_date`` 为**算分日 t**（t 日收盘后可得），执行在 t+1 开盘（CLAUDE.md §一.4）。

    **前视禁令**：输出第 ``t`` 行只能用 ``DlyCalDt <= t`` 的数据。
    自查用 :func:`assert_no_lookahead`（模块 docstring 有完整说明）。
    """

    #: 子类必须提供的冻结规格
    spec: SignalSpec

    @abc.abstractmethod
    def compute(self, panel: pd.DataFrame) -> pd.DataFrame:
        """在长表面板上算分，返回 ``signal_date, PERMNO, score[, ...]``。"""
        raise NotImplementedError

    # ------------------------------------------------------------------ 工具
    @staticmethod
    def require_columns(panel: pd.DataFrame,
                        columns: Sequence[str] = REQUIRED_PANEL_COLUMNS) -> None:
        missing = [c for c in columns if c not in panel.columns]
        if missing:
            raise KeyError(f"面板缺列 {missing}；至少需要 {list(columns)}")

    def __repr__(self) -> str:  # pragma: no cover - 便于日志
        return f"<{type(self).__name__} {self.spec}>"


# ---------------------------------------------------------------- 前视自查


def _frame_equal_on(a: pd.DataFrame, b: pd.DataFrame, keys: Sequence[str],
                    cols: Sequence[str], atol: float) -> list[str]:
    """按 keys 对齐后比较 cols，返回不一致的列名（NaN 与 NaN 视为相等）。"""
    left = a.set_index(list(keys)).sort_index()
    right = b.set_index(list(keys)).sort_index()
    if not left.index.equals(right.index):
        return ["<index>"]
    bad = []
    for c in cols:
        x, y = left[c], right[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            xv = x.to_numpy(dtype="float64")
            yv = y.to_numpy(dtype="float64")
            same = (np.isnan(xv) & np.isnan(yv)) | np.isclose(
                xv, yv, rtol=0.0, atol=atol, equal_nan=True)
            if not bool(same.all()):
                bad.append(c)
        else:
            if not x.equals(y):
                bad.append(c)
    return bad


def assert_no_lookahead(signal: "Signal", panel: pd.DataFrame, cutoff,
                        date_col: str = "DlyCalDt",
                        value_cols: Sequence[str] | None = None,
                        atol: float = 1e-12) -> None:
    """截断不变性检查：裁掉 ``cutoff`` 之后的行，之前的输出必须一字不变。

    抓不到「用了未来行但不改行数」的错误——那一类用测试里的**改数版**
    （把 t+1 之后的数值整体改掉再比对）。两者都便宜，建议都跑。

    Raises
    ------
    AssertionError
        若截断前后在 ``signal_date <= cutoff`` 的行上出现任何差异。
    """
    cut = pd.Timestamp(cutoff)
    full = signal.compute(panel)
    part = signal.compute(panel[pd.to_datetime(panel[date_col]) <= cut])

    full = full[pd.to_datetime(full["signal_date"]) <= cut].copy()
    part = part[pd.to_datetime(part["signal_date"]) <= cut].copy()
    if value_cols is None:
        value_cols = [c for c in full.columns if c not in ("signal_date", "PERMNO")]
    bad = _frame_equal_on(full, part, ("signal_date", "PERMNO"), value_cols, atol)
    if bad:
        raise AssertionError(
            f"前视自查失败：截断 {cut.date()} 后以下列发生变化 {bad}——"
            f"说明 signal_date <= {cut.date()} 的输出用到了之后的数据")
