"""exp13：Compustat 开发折诊断（SUE / 财报临近的张成诊断 + 财报日方差贡献）。

任务书 `docs/任务书_exp13_Compustat_开发折诊断_2026-09-05.md`
（冻结哈希 sha256 f8d1ec2bc1be4627fbcb5d5ccddab739eb0c17d59aa0d1838bf6e898f3e1acd9）。
下面 §3–§5 是任务书原文全文抄录，先于读第一行 Compustat 数据落笔（CLAUDE.md §二）。

================================================================================
## 3. 数据处理规则（先写死）

### 3.1 季度表筛选
`indfmt=INDL, datafmt=STD, consol=C`（验收显示全表已如此，脚本仍显式过滤并断言）。`fic` **不筛**（CRSP 宇宙决定样本）；`curcdq` 不筛，但报告 top500 内非 USD 占比。

### 3.2 重复键（1999–2025 有 599 组 `gvkey+datadate`，每组 2 行，`fyr` 均不同）
成因【推断，未核】为财年变更时新旧财年口径并存。规则：**同组保留 `fyearq` 最大者；再平则保留 `rdq` 非缺者；再平则 `fqtr` 最大者。** 零参数、确定性。交付时报告该规则在 top500 宇宙内实际触及的股票-季度数。

### 3.3 GVKEY → PERMNO（CCM 缺失下的替代链接）
- 键：`fundq.cusip`（9 位，Compustat 当前 header CUSIP【推断，须查 PDF 核】）↔ `security_info_history.hdrcusip9`（CRSP header CUSIP；【本项目实测 09-05】25,331 个 permno 全部非空、一一对应、无一对多）。
- 一个 gvkey ↔ 多个 permno（多股份类别）：全部保留，季度数据广播到每个 permno；报告数量。
- 一个 permno ↔ 多个 gvkey：视为链接冲突，**该 permno 整段置缺**并报告数量，不做挑选。
- 已知劣势（原样写进报告）：header 对 header 不是历史时点匹配；CUSIP 变更后两库 header 可能不同步；退市多年的公司两库 header 可能指向不同证券。**若日后拿到 `ccmxpf_lnkhist`，本节整体替换，D1–D3 全部重跑并登记两版差异。**

### 3.4 时点规则（前视禁令的核心，`CLAUDE.md` §一.1）
- 季度 q 的任何数值，**可用起点 = `rdq_q` 之后的首个交易日**（rdq 当日不用：公告可能在收盘后【推断】）。
- `rdq` 缺失的季度**不可用**。不用 `datadate + 固定滞后` 回填——两套时点混用等于给不同名字不同的信息集。
- 陈旧上限：信号日 t 距最近可用季度的 `datadate` **≤ 180 日历日**，超过置缺（B 层先验：Fama–French 年度 6 个月滞后惯例的季度类比；唯一常数）。
- 不做任何重述处理。fundq 数值是否按首次披露保留：**未核**。agent 须在 `documentation/Comp_Quarterly6126.pdf` 查 `epspxq` / `ibq` / `ajexq` 条目并原文摘录，标「有证据 / 未核」。

## 4. 构造定义（先写死；常数全部取文献默认，B 层优先序 ①②）

### 4.1 SUE（Livnat–Mendenhall 2006 口径【文献】）
- `E_q = epspxq / ajexq`（基本每股收益扣非常项目，按累计调整因子复权）。
- `SUE_q = (E_q − E_{q−4}) / σ_q`，`σ_q` = 前 8 个季度（q−7..q）季节差 `(E_j − E_{j−4})` 的样本标准差；**非缺 < 6 个置缺；σ = 0 置缺**。
- `q−4` 按 `(fyearq, fqtr)` 对齐，**不是按行位移**。
- 信号日 t 取 §3.4 规则下最近可用季度的 SUE；截面上照 exp11 的 `rank(pct=True)` 在当日候选池内秩化。

### 4.2 EA（财报临近；Frazzini–Lamont 2007 公告溢价因子的零前视版【文献 + 推断】）
- 预期公告日 `rdq_hat_q` = 同 `fqtr` **上一财年**的实际 `rdq` + 1 年（无则置缺）。**只用过去的 rdq**。
- `ea_prox_t = −min(交易日数(t → 最近的未来 rdq_hat), 63)`；无可用 `rdq_hat` 置缺。63 = 一个季度的交易日数，唯一常数。分数越高 = 越临近公告。
- 敏感性 `ea_real_t`：用**实际** `rdq` 计 `−min(交易日数(t → 最近的未来 rdq), 63)`。**这是事后日历**（现实中提前 2–4 周知晓），只作敏感性并标注。

### 4.3 张成规格（在 S-TH-ind 冻结定义上追加，不改 S-TH-ind 本身）

| 规格 | 控制集 |
|---|---|
| S-TH-ind-SUE | S-TH-ind + `sue` |
| S-TH-ind-EA | S-TH-ind + `ea_prox` |
| S-TH-ind-SUE-EA | S-TH-ind + `sue` + `ea_prox` |
| S-TH-ind-EAreal（敏感性） | S-TH-ind + `ea_real` |

- 因子走 K6b 同管道（top500 / 六档错位 **NT=6** / 进前 10% / 跌出前 30% 才卖 / t+1 开盘 / 多空价差腿 / 毛收益）。**与 exp11 相同的 NT=6 口径，读数不得与 NT=5 并列**（`CLAUDE.md` §八）。
- `sue` / `ea_prox` 与 S-TH-ind 其它因子一样做 point-in-time SIC2 行业内去均值（逻辑照 `scripts/xsec_context_probe.py:126-150`，逐日区间匹配 + `secinfoenddt` 失效判定）。
- 候选池 = 当天有分的名字；因子缺失的名字**不进该因子的候选池**。agent 须先核对 exp11 `:237-320` 对 `hi52` 缺失的实际处理并照抄，交付时写明。
- 三个主规格**全部报告，不得择优**；S-TH-ind 原读数并列作基线行。

### 4.4 D3 财报日方差（描述统计，允许用事后 rdq）
- 样本：折 36–42 各自验证窗，top500 宇宙（同 K6b 候选池），每个股票-日 d 的日收益 r（`panel_raw` 口径，含退市收益）。
- `ea_day_d = 1` 若 `rdq ∈ {d−1, d, d+1}`。
- 报告：(a) `mean(r² | ea=1) / mean(r² | ea=0)`；(b) `mean(r | ea=1) − mean(r | ea=0)` 与 NW(5) 95% CI；(c) 以 6 日持有窗（t+1..t+6）为单位：含 ≥1 个 `ea_day` 的窗占全部窗的比例，及其收益方差占全部窗方差的份额；(d) 以上逐折 + 合并。

## 5. 判据（先写死）

- **无门槛、无 PASS/FAIL、无「有价值 / 无价值」措辞。**
- 每个量报点估计 + NW(5) 95% CI + 7 折正折数；D2 另报各追加因子的载荷 β 及其 CI。
- **MDE 未算，因为本任务不是检验。** 若日后要把任何一项升为检验，须先按 `CLAUDE.md` §二独立写 SESOI / MDE / 功效，并按比较分层归 A 层或 B 层。
- 措辞模板：「在开发折 36–42（已消耗，方向性证据）上，加入 X 后 S-TH-ind 的 alpha 由 a 变为 b（保留率 c → d，N/7 正）；β_X = e（CI [·,·]）。不作判定。」
- exp11 的限定原句照抄：「仅限本控制集、本开发样本与冻结构造；保留率 >100% 只表示正向暴露于本样本内亏钱的因子，**不得写成无限定的 survives spanning**。」
================================================================================

实现说明（不属于任务书，属交付时必须披露的实现选择）
----------------------------------------------------
* 折号白名单：本脚本只接受折 36–42，读任何文件之前先断言（见 :func:`assert_folds`）。
  分数路径走 ``signals.kronos_adapter.scores_path(fold, "ft")``，该函数自身也有
  ``ALLOWED_FOLDS`` 守卫；所有 parquet 读取前调用 ``crsp_pipeline.sealed.assert_readable``。
* 管道复用：``scripts/exp11_spanning_extended.py`` 与 ``scripts/k6b_spanning.py``
  只 import、不修改（任务书 §2.2）。块划分、块内因子构造、SIC2 逐日匹配、行业内去均值、
  候选池取法全部调 exp11 的函数本体。唯一逐字抄的是 K6b 的 ``nw_ols``
  （``scripts/k6b_spanning.py:93-104``），抄的原因是原函数只返回 (b, t)，
  而任务书 §5 要求 95% CI，需要 se；抄件 :func:`nw_ols_se` 与原函数逐位一致
  （``tests/test_exp13_readout.py`` 断言两者的 b/t 完全相同）。
* 因子缺失的处理照抄 exp11：``hi52`` 在 ``exp11_spanning_extended.py:223-234`` 里
  不足 252 个观测即为 NaN，经 ``build_block_factors:252-253`` 的 left merge 进候选表，
  ``industry_demean:322-333`` 对 NaN 不回填，最后由 ``_by_day:336-343`` 的
  ``dropna()`` 把缺失名字剔出该因子当日的候选池，且当日有效名字 < 50 时整天不进该因子。
  ``sue`` / ``ea_prox`` / ``ea_real`` 走同一条路径，不做任何特殊处理。
* 无未来收益变量：本脚本不读任何未来收益文件，全文无 CLAUDE.md §一.1 点名的那个变量名
  （交付时附 grep 原始输出）。收益只在 t+1 及以后由 K6b 的
  冻结管道作为结果使用（``run_pipeline`` 的 ret/oc 映射）。
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from crsp_pipeline.sealed import assert_readable  # noqa: E402
from signals.kronos_adapter import scores_path  # noqa: E402

TASK_DOC = "docs/任务书_exp13_Compustat_开发折诊断_2026-09-05.md"
TASK_DOC_SHA256 = "f8d1ec2bc1be4627fbcb5d5ccddab739eb0c17d59aa0d1838bf6e898f3e1acd9"

PROCESSED = Path(r"F:\quant\processed\crsp_ciz_2026-08-24_20260825T130601Z")
JKP = Path(r"F:\quant\external\jkp")
DERIVED = Path(r"F:\quant\external\compustat\derived")
OUT_DIR = REPO / "outputs" / "exp13_compustat_dev_diag"

#: 折号白名单（任务书 §2.1；CLAUDE.md §四）。只有已消耗的现代开发折。
ALLOWED_FOLDS_EXP13: frozenset[int] = frozenset(range(36, 43))

# --- 任务书里出现的全部常数，一处定义，不得在别处硬写 -------------------------
SUE_LOOKBACK_QUARTERS = 8      # §4.1 σ 的窗口 q-7..q
SUE_MIN_OBS = 6                # §4.1 非缺 < 6 置缺
SUE_SEASONAL_LAG = 4           # §4.1 q-4，按 (fyearq, fqtr) 对齐
STALENESS_DAYS = 180           # §3.4 唯一常数
EA_CAP_TRADING_DAYS = 63       # §4.2 唯一常数
EA_DAY_HALF_WINDOW = 1         # §4.4 rdq ∈ {d-1, d, d+1}
HOLDING_DAYS = 6               # §4.4 6 日持有窗 t+1..t+6
NW_LAGS = 5                    # §5 NW(5)
Z95 = 1.959963984540054        # 正态 95% 双侧分位
MIN_REGRESSION_DAYS = 40       # 照 exp11:449 / :464

EXTRA_FACTORS: tuple[str, ...] = ("sue", "ea_prox", "ea_real")


class FoldWhitelistError(RuntimeError):
    """折号越出 exp13 白名单。"""


def assert_folds(folds: Iterable[int]) -> tuple[int, ...]:
    """读任何文件之前调用。任务书 §2.1。"""
    got = tuple(int(f) for f in folds)
    bad = sorted(set(got) - ALLOWED_FOLDS_EXP13)
    if bad:
        raise FoldWhitelistError(
            f"exp13 只允许折 {sorted(ALLOWED_FOLDS_EXP13)}；收到越界折号 {bad}。"
            "折 05–35 / fold44–45 / 封存窗的读取即全部作废（任务书 §2.1）。"
        )
    if not got:
        raise FoldWhitelistError("折号为空")
    return got


def _load_script_module(name: str, path: Path):
    """照 exp11:118-124；只 import，不改被引脚本。"""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXP11 = _load_script_module("exp13_exp11", REPO / "scripts" / "exp11_spanning_extended.py")
K6B = EXP11.K6B
CORE: tuple[str, ...] = tuple(EXP11.CORE)
EXTENDED_COLUMNS: tuple[str, ...] = tuple(EXP11.EXTENDED_COLUMNS)
BASE_CONTROLS: tuple[str, ...] = CORE + EXTENDED_COLUMNS       # = exp11 的 S-TH-ind

#: §4.3 四个规格 + 一个复现校验行（不是新规格，用来证明本脚本重跑出 exp11 的 S-TH-ind）。
SPEC_COLUMNS: dict[str, tuple[str, ...]] = {
    "S-TH-ind-repro": BASE_CONTROLS,
    "S-TH-ind-SUE": BASE_CONTROLS + ("sue",),
    "S-TH-ind-EA": BASE_CONTROLS + ("ea_prox",),
    "S-TH-ind-SUE-EA": BASE_CONTROLS + ("sue", "ea_prox"),
    "S-TH-ind-EAreal": BASE_CONTROLS + ("ea_real",),
}
ALL_CONTROL_COLUMNS: tuple[str, ...] = BASE_CONTROLS + EXTRA_FACTORS


def log(message: str) -> None:
    print(message, flush=True)


# ---------------------------------------------------------------------------
# 统计：NW(5) OLS，逐字抄自 scripts/k6b_spanning.py:93-104，只多返回 se
# ---------------------------------------------------------------------------
def nw_ols_se(y, X, lags: int = NW_LAGS):
    """与 k6b_spanning.py:93-104 的 ``nw_ols`` 逐位一致，另返回标准误。

    抄写而非 import 的原因：原函数只返回 (b, t)，任务书 §5 要求 95% CI，需要 se。
    tests/test_exp13_readout.py 断言本函数与 ``K6B.nw_ols`` 的 (b, t) 完全相同。
    """
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ (X.T @ y)
    e = y - X @ b
    S = (X * e[:, None]).T @ (X * e[:, None])
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        A = (X[L:] * e[L:, None]).T @ (X[:-L] * e[:-L, None])
        S += w * (A + A.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(V), 0))
    return b, np.where(se > 0, b / se, np.nan), se


def nw_mean_ci(series: pd.Series, lags: int = NW_LAGS) -> dict[str, Any]:
    """序列均值的 NW(lags) 点估计 / t / 95% CI（对常数回归）。"""
    values = pd.Series(series).dropna().to_numpy(dtype=np.float64)
    if len(values) < 2:
        return {"n": int(len(values)), "mean": float("nan"), "nw5_se": float("nan"),
                "nw5_t": float("nan"), "ci95": [float("nan"), float("nan")]}
    design = np.ones((len(values), 1), dtype=np.float64)
    beta, tstat, se = nw_ols_se(values, design, lags)
    return {
        "n": int(len(values)),
        "mean": float(beta[0]),
        "nw5_se": float(se[0]),
        "nw5_t": float(tstat[0]),
        "ci95": [float(beta[0] - Z95 * se[0]), float(beta[0] + Z95 * se[0])],
    }


# ---------------------------------------------------------------------------
# 交易日历
# ---------------------------------------------------------------------------
def load_trading_calendar(processed: Path) -> pd.DatetimeIndex:
    """CRSP 市场指数的 caldt 即交易日历（K6b 用同一文件取 vwretd）。"""
    path = processed / "market_index.parquet"
    assert_readable(path)
    frame = pd.read_parquet(path, columns=["caldt"])
    days = pd.DatetimeIndex(pd.to_datetime(frame["caldt"]).unique()).sort_values()
    return days.astype("datetime64[ns]")


def _position_of_first_ge(calendar: pd.DatetimeIndex, dates: pd.Series) -> np.ndarray:
    """每个日期在交易日历里 >= 自身的首个交易日的位置；越界为 -1。"""
    values = pd.to_datetime(dates).to_numpy(dtype="datetime64[ns]")
    pos = np.searchsorted(calendar.to_numpy(dtype="datetime64[ns]"), values, side="left")
    pos = pos.astype(np.int64)
    pos[pos >= len(calendar)] = -1
    pos[pd.isna(values)] = -1
    return pos


def _position_exact(calendar: pd.DatetimeIndex, dates: pd.Series) -> np.ndarray:
    """信号日必须正好落在交易日历上；否则报错（口径不一致的早期信号）。"""
    values = pd.to_datetime(dates).to_numpy(dtype="datetime64[ns]")
    cal = calendar.to_numpy(dtype="datetime64[ns]")
    pos = np.searchsorted(cal, values, side="left").astype(np.int64)
    if (pos >= len(cal)).any() or (cal[np.clip(pos, 0, len(cal) - 1)] != values).any():
        raise ValueError("有信号日不在 CRSP 交易日历上")
    return pos


# ---------------------------------------------------------------------------
# §4.1 SUE / §4.2 rdq_hat：gvkey 层构造
# ---------------------------------------------------------------------------
def load_fundq(derived: Path) -> pd.DataFrame:
    path = derived / "fundq_slim.parquet"
    assert_readable(path)
    frame = pd.read_parquet(path)
    # parquet 可能回 datetime64[us]；全流程统一到 [ns]，否则 merge_asof 会因单位不同报错
    frame["datadate"] = pd.to_datetime(frame["datadate"]).astype("datetime64[ns]")
    frame["rdq"] = pd.to_datetime(frame["rdq"]).astype("datetime64[ns]")
    return frame


def load_link(derived: Path) -> pd.DataFrame:
    path = derived / "gvkey_permno_link.parquet"
    assert_readable(path)
    link = pd.read_parquet(path)
    link["permno"] = pd.to_numeric(link["permno"], errors="raise").astype("int64")
    if link.duplicated(["gvkey", "permno"]).any():
        raise AssertionError("链接表重复 (gvkey, permno)")
    if link.groupby("permno")["gvkey"].nunique().gt(1).any():
        raise AssertionError("§3.3 冲突 permno 未被剔除（链接表已被改动？）")
    return link


def build_quarter_features(fundq: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """gvkey 层：E_q、季节差、σ_q、SUE_q、rdq_hat_q。q-4 按 (fyearq, fqtr) 对齐。"""
    frame = fundq.copy()
    n_in = len(frame)
    valid = (
        frame["fqtr"].isin([1, 2, 3, 4])
        & frame["fyearq"].between(1900, 2100)
        & frame["gvkey"].notna()
    )
    dropped_bad_fiscal = int((~valid).sum())
    frame = frame.loc[valid].copy()
    frame["fyearq"] = frame["fyearq"].astype("int64")
    frame["fqtr"] = frame["fqtr"].astype("int64")
    # 财季序号：连续整数，q-4 就是 fidx-4（§4.1「不是按行位移」）
    frame["fidx"] = frame["fyearq"] * 4 + frame["fqtr"] - 1

    # (gvkey, fyearq, fqtr) 的残余重复：§3.2 只保证 (gvkey, datadate) 唯一。
    # 沿用 §3.2 的精神（rdq 非缺优先），再取 datadate 最大者；确定性，无自由参数。
    frame["_rdq_present"] = frame["rdq"].notna().astype("int8")
    dup_fidx_rows = int(frame.duplicated(["gvkey", "fidx"], keep=False).sum())
    dup_fidx_groups = int(
        frame.loc[frame.duplicated(["gvkey", "fidx"], keep=False), ["gvkey", "fidx"]]
        .drop_duplicates().shape[0]
    )
    frame = (
        frame.sort_values(["gvkey", "fidx", "_rdq_present", "datadate"],
                          ascending=[True, True, False, False], kind="mergesort")
        .drop_duplicates(["gvkey", "fidx"], keep="first")
        .drop(columns=["_rdq_present"])
        .sort_values(["gvkey", "fidx"], kind="mergesort")
        .reset_index(drop=True)
    )

    # E_q = epspxq / ajexq
    ajex = frame["ajexq"].where(frame["ajexq"] > 0)
    frame["E"] = frame["epspxq"] / ajex

    # 补齐每个 gvkey 的连续财季网格，缺口才会被当成缺失（否则 shift 会跨过缺口）
    span = frame.groupby("gvkey")["fidx"].agg(["min", "max"])
    lengths = (span["max"] - span["min"] + 1).to_numpy(dtype=np.int64)
    if (lengths > 500).any():
        raise ValueError("某 gvkey 的财季跨度超过 500 个季度，疑似 fyearq 异常")
    gvkeys = np.repeat(span.index.to_numpy(), lengths)
    starts = np.repeat(span["min"].to_numpy(dtype=np.int64), lengths)
    offsets = (np.arange(lengths.sum(), dtype=np.int64)
               - np.repeat(np.concatenate([[0], np.cumsum(lengths)[:-1]]), lengths))
    grid = pd.DataFrame({"gvkey": gvkeys, "fidx": starts + offsets})
    grid = grid.merge(
        frame[["gvkey", "fidx", "datadate", "rdq", "curcdq", "dedup_group_rows", "E"]],
        on=["gvkey", "fidx"], how="left", validate="one_to_one",
    ).sort_values(["gvkey", "fidx"], kind="mergesort").reset_index(drop=True)

    group = grid.groupby("gvkey", sort=False)
    grid["E_lag4"] = group["E"].shift(SUE_SEASONAL_LAG)
    grid["seasonal_diff"] = grid["E"] - grid["E_lag4"]
    sigma = (
        grid.groupby("gvkey", sort=False)["seasonal_diff"]
        .rolling(SUE_LOOKBACK_QUARTERS, min_periods=SUE_MIN_OBS)
        .std()
        .reset_index(level=0, drop=True)
    )
    grid["sigma"] = sigma
    grid["sue"] = grid["seasonal_diff"] / grid["sigma"].where(grid["sigma"] > 0)

    # §4.2：rdq_hat_q = 同 fqtr 上一财年的实际 rdq + 1 年（只用过去的 rdq）
    grid["rdq_source"] = grid.groupby("gvkey", sort=False)["rdq"].shift(SUE_SEASONAL_LAG)
    has_source = grid["rdq_source"].notna()
    grid["rdq_hat"] = pd.Series(pd.NaT, index=grid.index, dtype="datetime64[ns]")
    grid.loc[has_source, "rdq_hat"] = (
        grid.loc[has_source, "rdq_source"] + pd.DateOffset(years=1)
    )

    out = grid.loc[grid["datadate"].notna()].copy()
    for column in ("datadate", "rdq", "rdq_hat", "rdq_source"):
        out[column] = pd.to_datetime(out[column]).astype("datetime64[ns]")
    stats = {
        "fundq_rows_in": n_in,
        "dropped_bad_fiscal_tag": dropped_bad_fiscal,
        "duplicate_gvkey_fiscal_quarter_rows": dup_fidx_rows,
        "duplicate_gvkey_fiscal_quarter_groups": dup_fidx_groups,
        "quarter_rows": int(len(out)),
        "quarters_with_rdq": int(out["rdq"].notna().sum()),
        "quarters_with_sue": int(out["sue"].notna().sum()),
        "quarters_with_rdq_hat": int(out["rdq_hat"].notna().sum()),
        "sigma_zero_quarters": int((out["sigma"] == 0).sum()),
    }
    return out[[
        "gvkey", "fidx", "datadate", "rdq", "curcdq", "dedup_group_rows",
        "E", "seasonal_diff", "sigma", "sue", "rdq_hat", "rdq_source",
    ]], stats


def broadcast_to_permno(quarters: pd.DataFrame, link: pd.DataFrame) -> pd.DataFrame:
    """§3.3：季度数据广播到 gvkey 链上的每个 permno。"""
    merged = quarters.merge(link[["gvkey", "permno"]], on="gvkey", how="inner")
    merged = merged.rename(columns={"permno": "PERMNO"})
    return merged.sort_values(["PERMNO", "rdq", "datadate"], kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------------------
# §3.4 时点 + §4.2 交易日距离：把季度量落到 (PERMNO, signal_date)
# ---------------------------------------------------------------------------
def assign_point_in_time(keys: pd.DataFrame, permno_quarters: pd.DataFrame,
                         calendar: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict[str, Any]]:
    """keys = (PERMNO, signal_date)。返回 sue / ea_prox / ea_real 三列 + 诊断。

    时点：可用起点 = rdq 之后的首个交易日。信号日 t 是交易日，故
    「首个交易日(rdq) <= t」等价于「rdq < t」，用 allow_exact_matches=False 的
    backward asof 实现（rdq 当日不可用，rdq 缺失的季度不可用）。
    """
    left = keys[["PERMNO", "signal_date"]].copy()
    left["PERMNO"] = pd.to_numeric(left["PERMNO"], errors="raise").astype("int64")
    left["signal_date"] = pd.to_datetime(left["signal_date"]).astype("datetime64[ns]")
    if left.duplicated(["PERMNO", "signal_date"]).any():
        raise ValueError("时点匹配键重复")
    left = left.sort_values(["signal_date", "PERMNO"], kind="mergesort").reset_index(drop=True)

    # --- SUE：最近可用季度（按 rdq 排序取最后一个 rdq < t）
    right = permno_quarters.dropna(subset=["rdq"]).copy()
    right = right[["PERMNO", "rdq", "datadate", "curcdq", "dedup_group_rows", "sue"]]
    right = right.sort_values(["rdq", "PERMNO", "datadate"], kind="mergesort").reset_index(drop=True)
    matched = pd.merge_asof(
        left, right, left_on="signal_date", right_on="rdq", by="PERMNO",
        direction="backward", allow_exact_matches=False,
    )
    age_days = (matched["signal_date"] - matched["datadate"]).dt.days
    stale = matched["datadate"].notna() & (age_days > STALENESS_DAYS)
    matched["quarter_age_days"] = age_days.where(~stale)
    matched["stale"] = stale
    matched["sue"] = matched["sue"].where(~stale)
    matched["curcdq"] = matched["curcdq"].where(~stale)
    matched["dedup_group_rows"] = matched["dedup_group_rows"].where(~stale)
    matched["sue_source_rdq"] = matched["rdq"].where(~stale)
    matched["sue_source_datadate"] = matched["datadate"].where(~stale)
    matched = matched.drop(columns=["rdq", "datadate"])

    # --- ea_prox：最近的未来 rdq_hat（>= t），只用过去的 rdq 推出来的那些
    hats = permno_quarters.dropna(subset=["rdq_hat"])[["PERMNO", "rdq_hat", "rdq_source"]]
    hats = hats.sort_values(["rdq_hat", "PERMNO"], kind="mergesort").reset_index(drop=True)
    fwd = pd.merge_asof(
        left, hats, left_on="signal_date", right_on="rdq_hat", by="PERMNO",
        direction="forward", allow_exact_matches=True,
    )
    # 前视自查：被选中的 rdq_hat 必须由**信号日之前**已公告的 rdq 推出
    source_future = fwd["rdq_source"].notna() & (fwd["rdq_source"] >= fwd["signal_date"])
    n_source_future = int(source_future.sum())
    fwd.loc[source_future, "rdq_hat"] = pd.NaT
    matched["ea_prox"] = _signed_trading_distance(calendar, fwd["signal_date"], fwd["rdq_hat"])
    matched["rdq_hat"] = fwd["rdq_hat"]

    # --- ea_real（敏感性，事后日历）：最近的未来实际 rdq（>= t）
    reals = permno_quarters.dropna(subset=["rdq"])[["PERMNO", "rdq"]].drop_duplicates()
    reals = reals.sort_values(["rdq", "PERMNO"], kind="mergesort").reset_index(drop=True)
    fwd_real = pd.merge_asof(
        left, reals, left_on="signal_date", right_on="rdq", by="PERMNO",
        direction="forward", allow_exact_matches=True,
    )
    matched["ea_real"] = _signed_trading_distance(calendar, fwd_real["signal_date"],
                                                 fwd_real["rdq"])
    stats = {
        "rows": int(len(matched)),
        "ea_prox_rows_voided_source_not_yet_announced": n_source_future,
        "rows_stale_beyond_180d": int(stale.sum()),
    }
    return matched, stats


def _signed_trading_distance(calendar: pd.DatetimeIndex, signal_date: pd.Series,
                             event_date: pd.Series) -> np.ndarray:
    """−min(交易日数(t → 事件日), 63)；事件日缺失置缺。

    交易日数 = 事件日（若非交易日则取其后首个交易日）在日历上的位置减 t 的位置；
    事件日 <= t 记 0；事件日超出日历右端按上限 63 处理（距离必然 > 63）。
    """
    t_pos = _position_exact(calendar, signal_date)
    e_pos = _position_of_first_ge(calendar, event_date)
    out = np.full(len(t_pos), np.nan, dtype=np.float64)
    known = pd.notna(pd.to_datetime(event_date)).to_numpy()
    beyond = known & (e_pos < 0)                       # 事件日在日历右端之外
    inside = known & (e_pos >= 0)
    distance = np.zeros(len(t_pos), dtype=np.float64)
    distance[inside] = np.maximum(e_pos[inside] - t_pos[inside], 0)
    distance[beyond] = EA_CAP_TRADING_DAYS
    out[known] = -np.minimum(distance[known], EA_CAP_TRADING_DAYS)
    return out


# ---------------------------------------------------------------------------
# D2：折内候选表 + 控制腿
# ---------------------------------------------------------------------------
def fold_candidate(fold: int, factors: pd.DataFrame, sic_history: pd.DataFrame,
                   outputs_root: Path, permno_quarters: pd.DataFrame,
                   calendar: pd.DatetimeIndex, *,
                   fold_windows: dict[int, tuple[str, str]],
                   scores_root: Path | None) -> tuple[pd.DataFrame, dict, dict[str, Any]]:
    """照 exp11:356-383 组候选表，再并入 sue / ea_prox / ea_real 并一起行业内去均值。"""
    if scores_root is None:
        path = scores_path(fold, "ft", root=outputs_root)
    else:
        path = Path(scores_root) / f"fold{fold:02d}" / "scores.parquet"
    assert_readable(path)
    scores = pd.read_parquet(path, columns=["PERMNO", "signal_date", "score"]).dropna()
    scores["signal_date"] = pd.to_datetime(scores["signal_date"])
    lo, hi = map(pd.Timestamp, fold_windows[fold])
    if scores["signal_date"].min() < lo or scores["signal_date"].max() > hi:
        raise ValueError(f"fold{fold} scores 日期越过冻结窗口 {lo.date()}..{hi.date()}")
    if scores.duplicated(["PERMNO", "signal_date"]).any():
        raise ValueError(f"fold{fold} scores 键重复")
    keyed = factors.rename(columns={"DlyCalDt": "signal_date"})
    candidate = scores.merge(keyed, on=["PERMNO", "signal_date"], how="left",
                             validate="one_to_one")
    sic = EXP11.point_in_time_sic2(scores[["PERMNO", "signal_date"]], sic_history)
    candidate = candidate.merge(sic, on=["PERMNO", "signal_date"], how="left",
                                validate="one_to_one")
    candidate = EXP11.add_candidate_extended_columns(candidate)

    pit, pit_stats = assign_point_in_time(scores[["PERMNO", "signal_date"]],
                                          permno_quarters, calendar)
    candidate = candidate.merge(
        pit[["PERMNO", "signal_date", "sue", "ea_prox", "ea_real", "curcdq",
             "dedup_group_rows", "quarter_age_days"]],
        on=["PERMNO", "signal_date"], how="left", validate="one_to_one",
    )
    # §4.1：sue 在当日候选池内 rank(pct=True)（照 exp11 对 turnover 的做法 :313）。
    # ea_prox / ea_real 任务书未要求秩化，保持原值（见 report.md 的偏离/读法说明）。
    candidate["sue"] = candidate.groupby("signal_date", sort=False)["sue"].rank(pct=True)
    all_industry_columns = ("score",) + BASE_CONTROLS + EXTRA_FACTORS
    candidate = EXP11.industry_demean(candidate, all_industry_columns)
    raw_scores = EXP11._by_day(scores, "score")
    return candidate, raw_scores, pit_stats


def collect_fold_daily(fold: int, candidate: pd.DataFrame, raw_scores: dict,
                       factors: pd.DataFrame) -> dict[str, Any]:
    """一次算齐策略腿与 15 个控制腿；各规格再从中取列（与逐规格重算逐位相同）。"""
    days = sorted(raw_scores)
    fold_panel = factors[factors["DlyCalDt"].isin(days)]
    ret, oc, adv = EXP11._panel_maps(fold_panel)
    ind_scores = EXP11._by_day(candidate, "ind_score")
    strategy = K6B.run_pipeline(ind_scores, days, ret, oc, adv, K6B.COST_BP)
    strategy = strategy.copy()
    strategy["fold"] = f"fold{fold}"
    controls: dict[str, pd.Series] = {}
    for column in ALL_CONTROL_COLUMNS:
        factor_scores = EXP11._by_day(candidate, f"ind_{column}")
        controls[column] = K6B.run_pipeline(
            factor_scores, days, ret, oc, adv, 0.0
        )["ls"].rename(column)
    return {"strategy": strategy, "controls": controls, "days": days,
            "ret": ret, "oc": oc, "adv": adv}


def readout(spec: str, strategy: pd.DataFrame, controls: pd.DataFrame,
            market: pd.Series, folds: Sequence[int]) -> dict[str, Any]:
    """估计交付：alpha / 保留率 / NW(5) t 与 95% CI / 逐折固定载荷残差 / 载荷 CI。

    回归设计与 exp11:439-502 逐位相同（同 y、同回归元顺序、同 dropna、同 NW(5)），
    只是**不输出任何结论字符串**——任务书 §5：无门槛、无 PASS/FAIL、无「有价值」措辞。
    """
    regressors = pd.concat([controls, market], axis=1, sort=True)
    data = pd.concat([strategy["long"].rename("y"), regressors], axis=1, sort=True).dropna()
    if len(data) < MIN_REGRESSION_DAYS:
        raise ValueError(f"{spec} 完整回归日不足：{len(data)}")
    matrix = data.drop(columns="y")
    design = np.column_stack([np.ones(len(data)), matrix.to_numpy(dtype=np.float64)])
    y = data["y"].to_numpy(dtype=np.float64)
    beta, tstat, se = nw_ols_se(y, design, NW_LAGS)
    raw_ann = float(y.mean() * 252 * 100)
    alpha_ann = float(beta[0] * 252 * 100)
    retention = float(100 * alpha_ann / raw_ann) if raw_ann else float("nan")
    residual = y - matrix.to_numpy(dtype=np.float64) @ beta[1:]
    residual_series = pd.Series(residual, index=data.index)
    per_fold: dict[str, float] = {}
    for fold in (f"fold{number}" for number in folds):
        dates = strategy.index[strategy["fold"] == fold]
        selected = residual_series.loc[residual_series.index.isin(dates)]
        if len(selected) < MIN_REGRESSION_DAYS:
            continue
        per_fold[fold] = float(selected.mean() * 252 * 100)
    positive = int(sum(value > 0 for value in per_fold.values()))
    return {
        "n_days": int(len(data)),
        "n_regressors": int(matrix.shape[1]),
        "regressors": list(matrix.columns),
        "raw_ann_pct": raw_ann,
        "alpha_ann_pct": alpha_ann,
        "alpha_ann_pct_ci95": [float((beta[0] - Z95 * se[0]) * 252 * 100),
                               float((beta[0] + Z95 * se[0]) * 252 * 100)],
        "retention_pct": retention,
        "nw5_t_alpha": float(tstat[0]),
        "folds_alpha_positive": positive,
        "folds_evaluated": len(per_fold),
        "per_fold_fixed_loading_residual_alpha_ann_pct": per_fold,
        "betas": {
            column: {
                "coefficient": float(value),
                "nw5_t": float(tvalue),
                "ci95": [float(value - Z95 * error), float(value + Z95 * error)],
            }
            for column, value, tvalue, error in zip(
                matrix.columns, beta[1:], tstat[1:], se[1:])
        },
        "required_caveat": (
            "仅限本控制集、本开发样本与冻结构造；保留率 >100% 只表示正向暴露于"
            "本样本内亏钱的因子，不得写成无限定的 survives spanning。"
        ),
    }


# ---------------------------------------------------------------------------
# D1：覆盖审计
# ---------------------------------------------------------------------------
def top500_universe(candidate: pd.DataFrame) -> pd.DataFrame:
    """K6b 的候选池：当日有分、adv20 有限，按 adv20 取前 500（k6b_spanning.py:196-199）。

    并列在第 500 名上的取舍与 run_pipeline 的 Python ``sorted`` 稳定序可能不同；
    这只影响边缘一两个名字，不影响 D1 占比与 D3 方差统计的量级（已披露）。
    """
    frame = candidate[np.isfinite(candidate["adv20"].to_numpy(dtype="float64"))].copy()
    frame = frame.sort_values(["signal_date", "adv20", "PERMNO"],
                              ascending=[True, False, True], kind="mergesort")
    frame["_rank"] = frame.groupby("signal_date", sort=False).cumcount()
    return frame[frame["_rank"] < K6B.TOPN].drop(columns="_rank")


def coverage_stats(fold: int, universe: pd.DataFrame) -> dict[str, Any]:
    n = int(len(universe))
    curcdq = universe["curcdq"].astype("string")
    return {
        "fold": fold,
        "stock_days_in_top500": n,
        "distinct_permnos": int(universe["PERMNO"].nunique()),
        "distinct_days": int(universe["signal_date"].nunique()),
        "sue_non_missing_share": float(universe["sue"].notna().mean()) if n else float("nan"),
        "ea_prox_non_missing_share": float(universe["ea_prox"].notna().mean()) if n else float("nan"),
        "ea_real_non_missing_share": float(universe["ea_real"].notna().mean()) if n else float("nan"),
        "quarter_matched_share": float(curcdq.notna().mean()) if n else float("nan"),
        "non_usd_share_of_matched": (
            float((curcdq.dropna() != "USD").mean()) if curcdq.notna().any() else float("nan")
        ),
        "non_usd_share_of_all": (
            float((curcdq.notna() & (curcdq != "USD")).mean()) if n else float("nan")
        ),
        "dedup_touched_stock_quarter_cells": int(
            (universe["dedup_group_rows"].fillna(1) > 1).sum()
        ),
        "median_quarter_age_days": (
            float(universe["quarter_age_days"].median())
            if universe["quarter_age_days"].notna().any() else float("nan")
        ),
    }


# ---------------------------------------------------------------------------
# D3：财报日方差
# ---------------------------------------------------------------------------
def load_returns_slice(processed: Path, lo: str, hi: str) -> pd.DataFrame:
    """列裁剪 + 日期下推，只取 3 列（给 D3 的 t+1..t+6 窗补块尾之后的收益）。"""
    path = processed / "panel_raw.parquet"
    assert_readable(path)
    frame = pd.read_parquet(
        path, columns=["PERMNO", "DlyCalDt", "DlyRet"],
        filters=[("DlyCalDt", ">=", pd.Timestamp(lo)), ("DlyCalDt", "<=", pd.Timestamp(hi))],
    )
    frame["DlyCalDt"] = pd.to_datetime(frame["DlyCalDt"])
    return frame.dropna(subset=["DlyRet"]).reset_index(drop=True)


def earnings_day_pairs(permno_quarters: pd.DataFrame, calendar: pd.DatetimeIndex,
                       lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DataFrame:
    """§4.4：ea_day_d = 1 若 rdq ∈ {d−1, d, d+1}（日历日），d 取交易日。"""
    events = permno_quarters.dropna(subset=["rdq"])[["PERMNO", "rdq"]].drop_duplicates()
    events = events[(events["rdq"] >= lo - pd.Timedelta(days=EA_DAY_HALF_WINDOW + 5))
                    & (events["rdq"] <= hi + pd.Timedelta(days=EA_DAY_HALF_WINDOW + 5))]
    pieces = []
    for offset in range(-EA_DAY_HALF_WINDOW, EA_DAY_HALF_WINDOW + 1):
        piece = events.copy()
        # rdq ∈ {d-1, d, d+1}  <=>  d ∈ {rdq-1, rdq, rdq+1}
        piece["day"] = piece["rdq"] + pd.Timedelta(days=offset)
        pieces.append(piece[["PERMNO", "day"]])
    pairs = pd.concat(pieces, ignore_index=True).drop_duplicates()
    pairs = pairs[pairs["day"].isin(calendar)]
    return pairs.reset_index(drop=True)


def d3_fold(fold: int, universe: pd.DataFrame, returns: pd.DataFrame,
            ea_pairs: pd.DataFrame, calendar: pd.DatetimeIndex) -> dict[str, Any]:
    """(a) 方差比、(b) 均值差 + NW(5) CI、(c) 6 日窗占比与方差份额。"""
    days = pd.DatetimeIndex(sorted(universe["signal_date"].unique()))
    ea_set = set(map(tuple, ea_pairs[["PERMNO", "day"]].to_numpy()))

    # --- (a)(b)：股票-日层面
    cells = universe[["PERMNO", "signal_date", "DlyRet"]].dropna(subset=["DlyRet"]).copy()
    keys = list(zip(cells["PERMNO"].to_numpy(), cells["signal_date"].to_numpy()))
    cells["ea"] = np.fromiter((k in ea_set for k in keys), dtype=bool, count=len(cells))
    r = cells["DlyRet"].to_numpy(dtype=np.float64)
    ea = cells["ea"].to_numpy()
    mean_sq_ea = float(np.mean(r[ea] ** 2)) if ea.any() else float("nan")
    mean_sq_non = float(np.mean(r[~ea] ** 2)) if (~ea).any() else float("nan")
    daily = cells.groupby("signal_date")
    diff = (daily.apply(lambda g: g.loc[g["ea"], "DlyRet"].mean()
                        - g.loc[~g["ea"], "DlyRet"].mean(), include_groups=False)
            if len(cells) else pd.Series(dtype=float))
    diff = pd.Series(diff).dropna()

    # --- (c)：6 日持有窗 t+1..t+6
    window = returns[returns["PERMNO"].isin(universe["PERMNO"].unique())].copy()
    window["lr"] = np.log1p(window["DlyRet"].clip(lower=-0.99))
    grid_days = calendar[(calendar >= days.min())
                         & (calendar <= days.max() + pd.Timedelta(days=40))]
    matrix = (window.pivot_table(index="DlyCalDt", columns="PERMNO", values="lr",
                                 aggfunc="first")
              .reindex(index=grid_days))
    forward = matrix.rolling(HOLDING_DAYS, min_periods=HOLDING_DAYS).sum().shift(-HOLDING_DAYS)
    ea_array = np.zeros((len(grid_days), matrix.shape[1]), dtype=np.float64)
    valid_pairs = ea_pairs[ea_pairs["PERMNO"].isin(matrix.columns)
                           & ea_pairs["day"].isin(grid_days)]
    if len(valid_pairs):
        ea_array[
            grid_days.get_indexer(pd.DatetimeIndex(valid_pairs["day"])),
            matrix.columns.get_indexer(valid_pairs["PERMNO"]),
        ] = 1.0
    ea_matrix = pd.DataFrame(ea_array, index=grid_days, columns=matrix.columns)
    ea_forward = ea_matrix.rolling(HOLDING_DAYS, min_periods=HOLDING_DAYS).sum().shift(-HOLDING_DAYS)

    pairs = universe[["PERMNO", "signal_date"]]
    row_idx = grid_days.get_indexer(pd.DatetimeIndex(pairs["signal_date"]))
    col_idx = matrix.columns.get_indexer(pairs["PERMNO"])
    ok = (row_idx >= 0) & (col_idx >= 0)
    win_lr = np.full(len(pairs), np.nan)
    win_ea = np.full(len(pairs), np.nan)
    fwd_values = forward.to_numpy(dtype=np.float64)
    ea_values = ea_forward.to_numpy(dtype=np.float64)
    win_lr[ok] = fwd_values[row_idx[ok], col_idx[ok]]
    win_ea[ok] = ea_values[row_idx[ok], col_idx[ok]]
    complete = np.isfinite(win_lr) & np.isfinite(win_ea)
    window_ret = np.expm1(win_lr[complete])
    window_ea = win_ea[complete] > 0
    n_windows = int(complete.sum())
    if n_windows:
        centred = window_ret - window_ret.mean()
        total_ss = float(np.sum(centred ** 2))
        ea_ss = float(np.sum(centred[window_ea] ** 2))
        window_share = float(window_ea.mean())
        variance_share = float(ea_ss / total_ss) if total_ss > 0 else float("nan")
        var_ea = float(np.var(window_ret[window_ea], ddof=1)) if window_ea.sum() > 1 else float("nan")
        var_non = float(np.var(window_ret[~window_ea], ddof=1)) if (~window_ea).sum() > 1 else float("nan")
    else:
        total_ss = ea_ss = window_share = variance_share = var_ea = var_non = float("nan")
    return {
        "fold": fold,
        "stock_days": int(len(cells)),
        "stock_days_ea": int(ea.sum()),
        "ea_day_share": float(ea.mean()) if len(ea) else float("nan"),
        "mean_r2_ea": mean_sq_ea,
        "mean_r2_non_ea": mean_sq_non,
        "mean_r2_ratio": (mean_sq_ea / mean_sq_non) if mean_sq_non else float("nan"),
        "daily_mean_return_diff": nw_mean_ci(diff),
        "windows_complete": n_windows,
        "windows_dropped_incomplete": int(len(pairs) - n_windows),
        "window_share_with_ea_day": window_share,
        "window_variance_share_ea": variance_share,
        "window_var_ea": var_ea,
        "window_var_non_ea": var_non,
        "_pooled_arrays": {
            "n_ea": int(ea.sum()), "sum_sq_ea": float(np.sum(r[ea] ** 2)),
            "n_non": int((~ea).sum()), "sum_sq_non": float(np.sum(r[~ea] ** 2)),
            "n_windows_ea": int(window_ea.sum()) if n_windows else 0,
            "n_windows": n_windows,
            "sum_window_ret": float(window_ret.sum()) if n_windows else 0.0,
            "sum_window_ret_sq": float((window_ret ** 2).sum()) if n_windows else 0.0,
            "sum_window_ret_ea": float(window_ret[window_ea].sum()) if n_windows else 0.0,
            "sum_window_ret_sq_ea": float((window_ret[window_ea] ** 2).sum()) if n_windows else 0.0,
        },
        "_daily_diff": diff,
    }


def d3_pool(fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    agg = {k: 0.0 for k in ("sum_sq_ea", "sum_sq_non", "sum_window_ret",
                            "sum_window_ret_sq", "sum_window_ret_ea", "sum_window_ret_sq_ea")}
    counts = {k: 0 for k in ("n_ea", "n_non", "n_windows_ea", "n_windows")}
    diffs = []
    for result in fold_results:
        arrays = result["_pooled_arrays"]
        for key in agg:
            agg[key] += arrays[key]
        for key in counts:
            counts[key] += arrays[key]
        diffs.append(result["_daily_diff"])
    mean_sq_ea = agg["sum_sq_ea"] / counts["n_ea"] if counts["n_ea"] else float("nan")
    mean_sq_non = agg["sum_sq_non"] / counts["n_non"] if counts["n_non"] else float("nan")
    n_w = counts["n_windows"]
    if n_w:
        mean_w = agg["sum_window_ret"] / n_w
        total_ss = agg["sum_window_ret_sq"] - n_w * mean_w ** 2
        n_ea_w = counts["n_windows_ea"]
        ea_ss = (agg["sum_window_ret_sq_ea"]
                 - 2 * mean_w * agg["sum_window_ret_ea"] + n_ea_w * mean_w ** 2)
        window_share = n_ea_w / n_w
        variance_share = ea_ss / total_ss if total_ss > 0 else float("nan")
    else:
        window_share = variance_share = float("nan")
    pooled_diff = pd.concat(diffs).sort_index() if diffs else pd.Series(dtype=float)
    return {
        "stock_days_ea": counts["n_ea"],
        "stock_days_non_ea": counts["n_non"],
        "ea_day_share": (counts["n_ea"] / (counts["n_ea"] + counts["n_non"])
                         if counts["n_ea"] + counts["n_non"] else float("nan")),
        "mean_r2_ea": mean_sq_ea,
        "mean_r2_non_ea": mean_sq_non,
        "mean_r2_ratio": (mean_sq_ea / mean_sq_non) if mean_sq_non else float("nan"),
        "daily_mean_return_diff": nw_mean_ci(pooled_diff),
        "windows_complete": n_w,
        "window_share_with_ea_day": window_share,
        "window_variance_share_ea": variance_share,
        "folds_ratio_gt_1": int(sum(r["mean_r2_ratio"] > 1 for r in fold_results)),
        "folds_diff_positive": int(sum(r["daily_mean_return_diff"]["mean"] > 0
                                       for r in fold_results)),
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run(processed: Path, jkp_root: Path, derived: Path, outputs_root: Path, *,
        folds: Iterable[int] | None = None,
        fold_windows: dict[int, tuple[str, str]] | None = None,
        blocks: Iterable[dict[str, Any]] | None = None,
        scores_root: Path | None = None,
        memory_limit_gb: float = 50.0,
        allow_alt_processed: bool = False) -> dict[str, Any]:
    use_folds = assert_folds(EXP11.FOLDS if folds is None else folds)
    use_blocks = tuple(EXP11.BLOCKS if blocks is None else blocks)
    use_windows = EXP11.FOLD_WINDOWS if fold_windows is None else fold_windows
    for block in use_blocks:
        assert_folds(block["folds"])
    start_committed = EXP11.enforce_memory_gate(memory_limit_gb)
    log(f"[exp13] 起始 committed={start_committed:.2f} GB")

    calendar = load_trading_calendar(processed)
    fundq = load_fundq(derived)
    link = load_link(derived)
    quarters, quarter_stats = build_quarter_features(fundq)
    del fundq
    permno_quarters = broadcast_to_permno(quarters, link)
    log(f"[exp13] 季度特征 {len(quarters):,} 行 -> permno 层 {len(permno_quarters):,} 行")
    del quarters

    sic_history = EXP11.load_sic_history(processed)
    collected_strategy: list[pd.DataFrame] = []
    collected_controls: dict[str, list[pd.Series]] = {c: [] for c in ALL_CONTROL_COLUMNS}
    coverage: list[dict[str, Any]] = []
    d3_results: list[dict[str, Any]] = []
    pit_stats_all: dict[str, Any] = {}
    block_meta: list[dict[str, Any]] = []

    for block in use_blocks:
        before = EXP11.enforce_memory_gate(memory_limit_gb)
        lo, hi = block["lo"], block["hi"]
        log(f"[exp13] 读取块 {lo}..{hi}；committed={before:.2f} GB")
        raw = EXP11.load_raw_block(processed, lo, hi)
        adjusted = EXP11.load_adjusted_block(processed, lo, hi)
        factors = EXP11.build_block_factors(raw, adjusted, processed,
                                            allow_alt_processed=allow_alt_processed)
        del raw, adjusted
        gc.collect()
        # D3 的 6 日窗要越过块尾；补一小段只含 3 列的收益切片
        tail = load_returns_slice(processed, hi,
                                  str((pd.Timestamp(hi) + pd.Timedelta(days=40)).date()))
        base_returns = factors[["PERMNO", "DlyCalDt", "DlyRet"]]
        returns = pd.concat([base_returns, tail], ignore_index=True).drop_duplicates(
            ["PERMNO", "DlyCalDt"], keep="first")
        del tail
        for fold in block["folds"]:
            candidate, raw_scores, pit_stats = fold_candidate(
                fold, factors, sic_history, outputs_root, permno_quarters, calendar,
                fold_windows=use_windows, scores_root=scores_root)
            pit_stats_all[f"fold{fold}"] = pit_stats
            daily = collect_fold_daily(fold, candidate, raw_scores, factors)
            collected_strategy.append(daily["strategy"])
            for column, series in daily["controls"].items():
                collected_controls[column].append(series)
            universe = top500_universe(candidate)
            coverage.append(coverage_stats(fold, universe))
            fold_lo, fold_hi = map(pd.Timestamp, use_windows[fold])
            ea_pairs = earnings_day_pairs(
                permno_quarters, calendar, fold_lo,
                fold_hi + pd.Timedelta(days=40))
            d3_results.append(d3_fold(fold, universe, returns, ea_pairs, calendar))
            log(f"  fold{fold} 完成：策略 {len(daily['strategy'])} 天，"
                f"top500 股票-日 {coverage[-1]['stock_days_in_top500']:,}")
            del candidate, daily, universe, ea_pairs
            gc.collect()
        block_meta.append({"lo": lo, "hi": hi, "folds": list(block["folds"]),
                           "rows_after_k6b": int(len(factors))})
        del factors, returns, base_returns
        gc.collect()

    market = EXP11.load_market_return(jkp_root)
    strategy = pd.concat(collected_strategy).sort_index()
    control_frame = pd.concat(
        [pd.concat(collected_controls[column]).sort_index()
         for column in ALL_CONTROL_COLUMNS], axis=1)
    results = {
        spec: readout(spec, strategy, control_frame[list(columns)], market, use_folds)
        for spec, columns in SPEC_COLUMNS.items()
    }
    pooled = d3_pool(d3_results)
    for result in d3_results:
        result.pop("_pooled", None)
        result.pop("_pooled_arrays", None)
        result.pop("_daily_diff", None)
    end_committed = EXP11.committed_memory_gb()
    return {
        "meta": {
            "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "task_doc": TASK_DOC,
            "task_doc_sha256": TASK_DOC_SHA256,
            "folds": list(use_folds),
            "fold_whitelist": sorted(ALLOWED_FOLDS_EXP13),
            "blocks": block_meta,
            "scores_arm": "ft",
            "scores_root": (None if scores_root is None else str(scores_root)),
            "derived_root": str(derived),
            "committed_memory_gb": {"start": start_committed, "end": end_committed,
                                    "limit": memory_limit_gb},
            "k6b_frozen_construction": {
                "nt": K6B.NT, "topn": K6B.TOPN, "exit_pct": K6B.EXIT_PCT,
                "strategy_cost_bp": K6B.COST_BP, "control_cost_bp": 0.0,
            },
            "constants": {
                "sue_lookback_quarters": SUE_LOOKBACK_QUARTERS,
                "sue_min_obs": SUE_MIN_OBS,
                "staleness_days": STALENESS_DAYS,
                "ea_cap_trading_days": EA_CAP_TRADING_DAYS,
                "holding_days": HOLDING_DAYS,
                "nw_lags": NW_LAGS,
            },
            "use_restriction": (
                "开发折 36–42 已消耗，读数为方向性证据；估计交付、无门槛、不判定。"
                "不得据此改 v4、不得据此挑规格、不得进入任何判定、不得作为部署依据。"
            ),
        },
        "quarter_stats": quarter_stats,
        "point_in_time_stats": pit_stats_all,
        "D1_coverage": coverage,
        "D2_spanning": results,
        "D3_earnings_day_variance": {"per_fold": d3_results, "pooled": pooled},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="exp13 Compustat 开发折诊断")
    parser.add_argument("--processed", type=Path, default=PROCESSED)
    parser.add_argument("--jkp", type=Path, default=JKP)
    parser.add_argument("--derived", type=Path, default=DERIVED)
    parser.add_argument("--outputs-root", type=Path, default=REPO / "outputs")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--memory-limit-gb", type=float, default=50.0)
    parser.add_argument("--smoke", action="store_true",
                        help="用合成数据跑通全链路（不碰任何真实数据）")
    return parser.parse_args()


def _write_outputs(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    (out_dir / "link_coverage.json").write_text(
        json.dumps({"meta": report["meta"], "quarter_stats": report["quarter_stats"],
                    "point_in_time_stats": report["point_in_time_stats"],
                    "D1_coverage": report["D1_coverage"]},
                   ensure_ascii=False, indent=2, default=float), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.smoke:
        from exp13_smoke import run_smoke  # noqa: PLC0415  仅冒烟路径导入
        return run_smoke(args.out_dir)
    report = run(args.processed.resolve(), args.jkp.resolve(), args.derived.resolve(),
                 args.outputs_root.resolve(), memory_limit_gb=args.memory_limit_gb)
    _write_outputs(report, args.out_dir)
    for spec, result in report["D2_spanning"].items():
        print(f"[{spec}] n={result['n_days']} alpha={result['alpha_ann_pct']:+.4f}% "
              f"retention={result['retention_pct']:.2f}% t={result['nw5_t_alpha']:+.3f} "
              f"pos={result['folds_alpha_positive']}/{result['folds_evaluated']}")
    print(f"[exp13] wrote {args.out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
