"""实验 10 / P3：无标签 NLL membership gap（污染探针）。

判据 / 用途限制（**先写死，跑第一行数据之前抄进本 docstring**；来源
``docs/实验指令_2026-09-04.md`` 实验 10 与主会话 2026-09-05 任务书）
-------------------------------------------------------------------
- **诊断，无阈值，不作判定**。不得据此宣称「没有污染」或「有污染」，
  只能报分布差异的**方向与量级**，并把混淆项一并列出。
- 干净窗一侧**只读输入面板的 OHLCV 与 universe 过滤所需列**，一行 label 都不生成。
- fold44–45 恰是 Kronos 作者自己的测试期，**checkpoint 级的开发者选择不外生**——
  不许把干净窗当成完全外生的样本。
- 本脚本只在**微调（FT）臂**上做。**ZS 臂已按用户 2026-09-05 裁定整体抛弃**
  （`experiments/confirmation_protocol_v4.1_addendum_2026-09-05.md`，该文件只读不改），
  故 P3 不跑 `outputs/zeroshot_base`——**不是漏跑**。代价是一个新混淆项：
  FT 模型除预训练语料外还见过**自己的训练窗**，污染侧的窗口紧邻其训练窗、
  干净侧则远离，二者的差异因此同时含「预训练记忆」与「微调近因」两个来源。

用户对读取口径的许可（2026-09-05 书面给出，原话逐字）
-----------------------------------------------------
    「干净窗（折 44–45）的原始 OHLCV 面板可以读；禁止的是那两折的分数、
      标签与任何指标。」

这**不是解封读取授权**：折 05–35 与 44–45 的分数、标签、指标，以及 outputs/ 下
任何封存目录一律不得读、不得写。本脚本用 :func:`assert_read_ok` /
:func:`assert_write_ok` 把这条做成可执行断言（黑名单 + 白名单双向）。

「原始 OHLCV 面板」在本实验解释为**模型输入侧的价量面板**，即
``panel_kronos_adj.parquet``（规范 §5 复权后 OHLCV）。理由：Kronos 的输入面板
全项目一律是它（训练 `kronos_ft/train.py`、打分 `scripts/evaluate_fold.py:191`
都用这一份），两侧用同一份才可比；`panel_raw.parquet` 是**标签**侧面板
（含 DlyRet / DlyDelFlg），本脚本**一次都不读**，并由断言禁死。

NLL 口径
--------
逐窗 NLL 抄自 ``src/kronos_ft/train.py`` 第 99–115 行 ``_stage_loss`` 的
``predictor`` 分支（**不改 train.py**）：

    with torch.no_grad():
        s0, s1 = tokenizer.encode(x, half=True)
    logits = model(s0[:, :-1], s1[:, :-1], stamp[:, :-1, :])
    loss, _, _ = model.head.compute_loss(logits[0], logits[1], s0[:, 1:], s1[:, 1:])

两处**刻意的、已披露的**偏离：

1. ``DualHead.compute_loss``（``third_party/kronos/model/module.py:494-507``）
   是 batch 均值；本脚本要**逐窗** NLL，故用 ``F.cross_entropy(..., reduction="none")``
   自己 reshape 回 ``[B, T-1]`` 再对时间维取均值。**逐窗均值的 batch 平均与
   compute_loss 完全相等**（所有窗口等长），故口径不变。
2. ``Kronos.forward`` 默认 ``use_teacher_forcing=False``，会用 ``torch.multinomial``
   采样 s1 来构造 s2 的 sibling embedding（``kronos.py:266-274``）——训练时的
   蒙特卡洛替身，**带随机性**。任务书要求「确定性 NLL、无采样」，故主口径改用
   ``use_teacher_forcing=True, s1_targets=s0[:, 1:]``，即精确的自回归分解
   ``log p(s1_t | ctx) + log p(s2_t | ctx, s1_t)``。为量化这一步的影响，脚本另在
   一个小子样本上跑 train.py 原样的采样路径（``--sampled-check N``）并报差值。

窗口构造两侧完全同一套：``kronos_ft.windows.build_scoring_index``（lookback 侧
连续有效，**predict=0，不取任何未来行**）+ ``filter_index_by_universe``，
lookback=90。归一化 = 逐窗 mean/std + clip 5.0，与
``kronos_ft/dataset.py:121-125`` 和 ``kronos_ft/infer.py:158-159`` 逐行一致。

折窗口一律由 ``scripts/emit_folds.py`` 机械生成（子进程调用），**不手写**。

产物
----
``outputs/exp10_contamination_probes.json`` + ``outputs/exp10_p3_nll.svg``（纯
python 画，`.venv` 无 matplotlib，勿安装）。**一切产物与中间文件只写
``outputs/exp10_*``**；写路径由 :func:`assert_write_ok` 钉死。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from crsp_pipeline.calendar import TradingCalendar  # noqa: E402
from crsp_pipeline.sealed import (  # noqa: E402
    FORBIDDEN_FILES,
    MANIFEST,
    SENTINEL,
    assert_readable,
)
from kronos_ft.dataset import FEATURE_COLS  # noqa: E402
from kronos_ft.models import load_pretrained, pick_device  # noqa: E402
from kronos_ft.windows import build_scoring_index, filter_index_by_universe  # noqa: E402

# --------------------------------------------------------------------------
# 读写边界（可执行断言）
# --------------------------------------------------------------------------

#: 允许作为**模型来源**的开发折（已消耗；CLAUDE.md §四）。
ALLOWED_MODEL_FOLDS = (36, 37, 38, 39, 40, 41, 42)
#: 干净窗（Kronos 预训练截止 2024-06 之后）；**只读输入面板**，不读任何产物。
CLEAN_FOLDS = (44, 45)

#: 路径里出现即拒绝（fold43 已被用户 2026-09-02 裁定移除，不计算也不读）。
DENIED_PATH_TOKENS = ("fold43", "fold44", "fold45", "zeroshot_base")
#: 文件名出现即拒绝：封存/评估产物与标签侧面板。
DENIED_FILE_NAMES = frozenset(
    set(FORBIDDEN_FILES) | {"scores.parquet", MANIFEST, SENTINEL, "panel_raw.parquet"}
)
#: 面板列白名单——标签相关列（DlyRet / DlyDelFlg / DlyCap …）一律不得请求。
ALLOWED_PANEL_COLUMNS = frozenset(
    {"PERMNO", "DlyCalDt", "in_universe"} | set(FEATURE_COLS.values())
)
#: 本实验唯一允许的写路径前缀（相对仓库根）。
WRITE_PREFIX = "outputs/exp10_"
#: 允许从模型目录读取的文件名（只有权重与训练摘要，没有任何分数/指标）。
MODEL_ARTIFACT_NAMES = ("model.safetensors", "config.json", "README.md")


class BoundaryError(RuntimeError):
    """越过本实验读取 / 写入边界。"""


def _rel(path) -> str:
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def assert_read_ok(path) -> Path:
    """读守卫：任何 parquet / 权重 / json 打开之前调用。

    四道闸：(1) 封存哨兵（``crsp_pipeline.sealed.assert_readable``，沿路径向上找
    ``SEALED``）；(2) 路径 token 黑名单（fold43/44/45 的**产物目录**、ZS 基座）；
    (3) 文件名黑名单（labels/metrics/scores/清单/panel_raw）；(4) outputs/ 下
    白名单——只允许 ``outputs/fold{36..42}_lb90_s0_poolB_universe`` 里的模型权重
    与训练摘要，以及本实验自己的 ``outputs/exp10_*``。
    """
    p = Path(path).resolve()
    assert_readable(p)
    rel = _rel(p)
    parts = [s.lower() for s in p.parts]
    for token in DENIED_PATH_TOKENS:
        if any(token in s for s in parts):
            raise BoundaryError(
                f"拒绝读取 {rel}：路径含禁用标识 {token!r}。"
                "折 44–45 只允许读输入面板（panel_kronos_adj / universe / market_index），"
                "其分数、标签、指标与任何 outputs 产物一律不得读。")
    if p.name in DENIED_FILE_NAMES:
        raise BoundaryError(f"拒绝读取 {rel}：文件名在黑名单内（分数/标签/指标/清单/标签侧面板）。")
    out_root = (REPO_ROOT / "outputs").resolve()
    if out_root in p.parents or p == out_root:
        try:
            first = p.relative_to(out_root).parts[0]
        except ValueError:  # pragma: no cover - 上面已保证是子路径
            first = ""
        if first.startswith("exp10_"):
            return p
        ok = any(first == f"fold{f:02d}_lb90_s0_poolB_universe"
                 for f in ALLOWED_MODEL_FOLDS)
        if not ok:
            raise BoundaryError(
                f"拒绝读取 {rel}：outputs/ 下只允许折 {ALLOWED_MODEL_FOLDS} 的"
                "微调模型目录与本实验自己的 outputs/exp10_*。")
        tail = p.relative_to(out_root).parts[1:]
        allowed_tail = (
            (len(tail) == 1 and tail[0].endswith("_summary.json"))
            or (len(tail) == 1 and tail[0] == "run_summary.json")
            or (len(tail) == 1 and tail[0].endswith(".pt"))
            or (len(tail) == 2 and tail[0] in ("tokenizer_final", "predictor_final")
                and tail[1] in MODEL_ARTIFACT_NAMES)
        )
        if not allowed_tail:
            raise BoundaryError(
                f"拒绝读取 {rel}：模型目录里只允许读权重（*_final/*、*.pt）与训练摘要。")
    return p


def assert_write_ok(path) -> Path:
    """写守卫：一切产物与中间文件只能落在 ``outputs/exp10_*``。

    封存审计会把封存目录里多出的文件判为违规、目录哈希也已钉住，因此
    ``outputs/fold44_*``、``outputs/fold45_*``、``outputs/zeroshot_base/*``
    以及任何带哨兵的目录**连一个字节都不许写**。仓库 outputs/ 之外的路径
    （临时目录、测试 tmp_path）不受此限。
    """
    p = Path(path).resolve()
    rel = _rel(p)
    parts = [s.lower() for s in p.parts]
    for token in DENIED_PATH_TOKENS:
        if any(token in s for s in parts):
            raise BoundaryError(f"拒绝写入 {rel}：路径含禁用标识 {token!r}。")
    if any((cand / SENTINEL).exists() for cand in [p, *p.parents] if cand.is_dir()):
        raise BoundaryError(f"拒绝写入 {rel}：目标位于封存目录之下。")
    out_root = (REPO_ROOT / "outputs").resolve()
    if out_root in p.parents:
        if not _rel(p).startswith(WRITE_PREFIX):
            raise BoundaryError(
                f"拒绝写入 {rel}：本实验只允许写 {WRITE_PREFIX}*。")
    return p


def assert_columns_ok(columns) -> list[str]:
    """列守卫：只允许 OHLCV + 键 + universe 标志；标签相关列一个都不许请求。"""
    cols = list(columns)
    bad = [c for c in cols if c not in ALLOWED_PANEL_COLUMNS]
    if bad:
        raise BoundaryError(
            f"拒绝读取列 {bad}：只允许 {sorted(ALLOWED_PANEL_COLUMNS)}（无标签相关列）。")
    return cols


def install_label_tripwire() -> list[str]:
    """把 ``crsp_pipeline.labels`` 的每个公开可调用对象换成会抛错的绊线。

    ``crsp_pipeline/__init__.py`` 无条件 ``from . import labels``，所以「labels
    不在 sys.modules」这种断言在本仓库里恒假、毫无保护力。真正需要保证的是
    **一次都不调用**：绊线把 ``compute_labels`` 等换掉，本实验若有任何代码路径
    想生成 label 会立刻炸掉，而不是安静地算出来。返回被绊住的名字。
    """
    import crsp_pipeline.labels as L

    tripped = []
    for name in dir(L):
        if name.startswith("_"):
            continue
        obj = getattr(L, name)
        if callable(obj) and getattr(obj, "__module__", "") == L.__name__:
            def _boom(*_a, __n=name, **_k):
                raise BoundaryError(
                    f"crsp_pipeline.labels.{__n} 被调用：本实验一行 label 都不生成"
                    "（干净窗只读输入侧 OHLCV）。")
            setattr(L, name, _boom)
            tripped.append(name)
    return tripped


def read_parquet_guarded(path, columns=None, filters=None) -> pd.DataFrame:
    p = assert_read_ok(path)
    cols = assert_columns_ok(columns) if columns is not None else None
    return pd.read_parquet(p, columns=cols, filters=filters)


# --------------------------------------------------------------------------
# 运行环境闸门
# --------------------------------------------------------------------------

def committed_memory_gb() -> float | None:
    """系统提交内存（GB）。取不到返回 None（不阻塞，交回时标注）。"""
    cmd = ["powershell", "-NoProfile", "-Command",
           "(Get-Counter '\\Memory\\Committed Bytes').CounterSamples[0].CookedValue/1GB"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return float(out.stdout.strip().replace(",", "."))
    except Exception:
        return None


def wait_memory_gate(limit_gb: float = 42.0, sleep_s: int = 60,
                     max_waits: int = 10) -> float | None:
    """每折（每个窗口集 / 每个模型）开跑前查提交内存；超过 limit 等 60 秒再查。

    **已披露的偏离**：等待有上限（默认 10 次 = 10 分钟）。本机上别的会话的作业
    （gbdt 封存队列、实验 8）把系统提交内存长期顶在 42 GB 附近，无上限等待会把
    本实验拖成数小时；本进程自身峰值只有几 GB。超时即继续，并在日志里明说。
    """
    for _ in range(max_waits):
        gb = committed_memory_gb()
        if gb is None or gb <= limit_gb:
            if gb is not None:
                log(f"提交内存 {gb:.2f} GB <= {limit_gb} GB，允许启动")
            return gb
        log(f"提交内存 {gb:.2f} GB > {limit_gb} GB，等待 {sleep_s} 秒")
        time.sleep(sleep_s)
    gb = committed_memory_gb()
    log(f"内存闸门等待超时（{max_waits} 次），当前 {gb} GB —— 本进程占用只有几 GB，"
        f"按已披露的偏离继续")
    return gb


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# --------------------------------------------------------------------------
# 折窗口（机械生成，不手写）
# --------------------------------------------------------------------------

def emit_folds(processed: Path, first: int, last: int) -> dict[int, dict]:
    """子进程调用 ``scripts/emit_folds.py``，返回 {折号: {ts, te, vs, ve}}。"""
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "emit_folds.py"),
           "--processed", str(processed), "--first", str(first), "--last", str(last)]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT),
                         check=True, encoding="utf-8")
    rows = json.loads(out.stdout.strip().splitlines()[-1])
    return {int(r["n"].replace("fold", "")): r for r in rows}


# --------------------------------------------------------------------------
# 窗口集：索引 + 特征 + 混淆项
# --------------------------------------------------------------------------

PANEL_COLS = ["PERMNO", "DlyCalDt"] + list(FEATURE_COLS.values())
STAMP_COLS = ["minute", "hour", "weekday", "day", "month"]


def _stamp_matrix(dates: pd.DatetimeIndex) -> np.ndarray:
    """与 ``kronos_ft/infer.py:36-45`` / ``dataset.py:41-49`` 同序同义。"""
    d = pd.DatetimeIndex(dates)
    return np.column_stack([
        d.minute.to_numpy(), d.hour.to_numpy(), d.weekday.to_numpy(),
        d.day.to_numpy(), d.month.to_numpy(),
    ]).astype(np.float32)


class WindowSet:
    """一个评估窗的全部打分窗口 + 逐窗特征切片 + 混淆项统计。

    只持有每只股票的 (n_sessions, 6) float32 数组与日期→行号映射；批次组装时
    才做切片与归一化（照 ``kronos_ft/infer.py`` 的 fast 路径）。
    """

    def __init__(self, name: str, panel: pd.DataFrame, universe: pd.DataFrame,
                 calendar: TradingCalendar, lookback: int,
                 val_start: pd.Timestamp, val_end: pd.Timestamp,
                 limit_windows: int = 0, seed: int = 20260905):
        self.name = name
        self.lookback = int(lookback)
        self.val_start, self.val_end = val_start, val_end

        idx = build_scoring_index(panel, calendar, lookback)
        idx = idx[(idx["anchor"] >= val_start) & (idx["anchor"] <= val_end)]
        self.n_before_universe = int(len(idx))
        idx = filter_index_by_universe(idx, universe)
        idx = idx.sort_values(["anchor", "PERMNO"]).reset_index(drop=True)
        if limit_windows and len(idx) > limit_windows:
            rng = np.random.default_rng(seed)
            take = np.sort(rng.choice(len(idx), size=limit_windows, replace=False))
            idx = idx.iloc[take].reset_index(drop=True)
        self.index = idx

        need = set(idx["PERMNO"].unique())
        df = panel[panel["PERMNO"].isin(need)].copy()
        df["DlyCalDt"] = pd.to_datetime(df["DlyCalDt"])
        cols = list(FEATURE_COLS.values())
        self._feat: dict = {}
        self._pos: dict = {}
        self._sessions: dict = {}
        for pn, g in df.groupby("PERMNO"):
            g = g.sort_values("DlyCalDt")
            sessions = calendar.sessions(g["DlyCalDt"].iloc[0], g["DlyCalDt"].iloc[-1])
            s = g.set_index("DlyCalDt")[cols].reindex(sessions)
            self._feat[pn] = s.to_numpy(dtype=np.float32)
            self._pos[pn] = {d: i for i, d in enumerate(sessions)}
            self._sessions[pn] = sessions
        self._stamp_cache: dict = {}
        self.confounders = self._confounders(df)

    # ---- 批次组装 -------------------------------------------------------
    def __len__(self) -> int:
        return len(self.index)

    def batch(self, lo: int, hi: int, clip: float = 5.0):
        """返回 (x [B,L,6] 归一化后, stamp [B,L,5])，与训练/打分口径逐行一致。"""
        chunk = self.index.iloc[lo:hi]
        xs, sts = [], []
        for pn, anchor, start in chunk[["PERMNO", "anchor", "start"]].itertuples(index=False):
            anchor, start = pd.Timestamp(anchor), pd.Timestamp(start)
            i0, i1 = self._pos[pn][start], self._pos[pn][anchor]
            x = self._feat[pn][i0:i1 + 1]
            if not np.isfinite(x).all():
                raise ValueError(f"窗含非有限值: PERMNO={pn} anchor={anchor.date()}")
            # kronos_ft/dataset.py:121-125（训练）与 infer.py:158-159（打分）同款
            mean, std = x.mean(axis=0), x.std(axis=0)
            xn = np.clip((x - mean) / (std + 1e-5), -clip, clip).astype(np.float32)
            xs.append(xn)
            st = self._stamp_cache.get(anchor)
            if st is None:
                st = _stamp_matrix(self._sessions[pn][i0:i1 + 1])
                self._stamp_cache[anchor] = st
            sts.append(st)
        return np.stack(xs, 0), np.stack(sts, 0)

    # ---- 混淆项 ---------------------------------------------------------
    def _confounders(self, df: pd.DataFrame) -> dict:
        """两侧必须并列的混淆项。全部只用 anchor 日及以前的信息，无前视、无 label。"""
        idx = self.index
        if len(idx) == 0:
            return {"n_windows": 0}
        px = df[["PERMNO", "DlyCalDt", "DlyClose", "DlyPrcVol"]].sort_values(
            ["PERMNO", "DlyCalDt"]).copy()
        # r_t = Close_t / Close_{t-1} - 1：t 日收盘即可算，非未来收益、非 label
        px["ret_past"] = px.groupby("PERMNO")["DlyClose"].pct_change()
        m = idx.merge(px.rename(columns={"DlyCalDt": "anchor"}),
                      on=["PERMNO", "anchor"], how="left")
        xs_vol = m.groupby("anchor")["ret_past"].std()
        dv = m["DlyPrcVol"].to_numpy(dtype=float)
        dv = dv[np.isfinite(dv) & (dv > 0)]
        # 逐窗已实现波动（lookback 段日对数收益的 SD），窗内信息
        rv = []
        for pn, anchor, start in idx[["PERMNO", "anchor", "start"]].itertuples(index=False):
            i0, i1 = self._pos[pn][pd.Timestamp(start)], self._pos[pn][pd.Timestamp(anchor)]
            c = self._feat[pn][i0:i1 + 1, 3]
            with np.errstate(divide="ignore", invalid="ignore"):
                lr = np.diff(np.log(np.where(c > 0, c, np.nan)))
            lr = lr[np.isfinite(lr)]
            if lr.size > 1:
                rv.append(float(np.std(lr, ddof=1)))
        rv = np.asarray(rv, dtype=float)
        return {
            "n_windows": int(len(idx)),
            "n_windows_before_universe_filter": self.n_before_universe,
            "n_names": int(idx["PERMNO"].nunique()),
            "n_days": int(idx["anchor"].nunique()),
            "windows_per_day_median": float(idx.groupby("anchor").size().median()),
            "xs_vol_daily_mean": float(xs_vol.mean()),
            "xs_vol_daily_median": float(xs_vol.median()),
            "window_realized_vol_mean": float(np.mean(rv)) if rv.size else None,
            "window_realized_vol_median": float(np.median(rv)) if rv.size else None,
            "dollar_vol_p10": float(np.percentile(dv, 10)) if dv.size else None,
            "dollar_vol_median": float(np.median(dv)) if dv.size else None,
            "dollar_vol_p90": float(np.percentile(dv, 90)) if dv.size else None,
            "log10_dollar_vol_mean": float(np.mean(np.log10(dv))) if dv.size else None,
            "val_window": [str(self.val_start.date()), str(self.val_end.date())],
        }


def load_window_set(processed: Path, calendar: TradingCalendar, name: str,
                    val_start: str, val_end: str, lookback: int,
                    limit_windows: int = 0) -> WindowSet:
    vs, ve = pd.Timestamp(val_start), pd.Timestamp(val_end)
    lo = calendar.shift(calendar.snap_forward(vs), -(lookback + 30))
    hi = calendar.snap_back(ve)
    log(f"[{name}] 面板切片 [{lo.date()} .. {hi.date()}]（打分窗 predict=0，不取未来行）")
    filters = [("DlyCalDt", ">=", lo), ("DlyCalDt", "<=", hi)]
    panel = read_parquet_guarded(processed / "panel_kronos_adj.parquet",
                                 columns=PANEL_COLS, filters=filters)
    uni = read_parquet_guarded(processed / "universe.parquet",
                               columns=["PERMNO", "DlyCalDt", "in_universe"],
                               filters=filters)
    ws = WindowSet(name, panel, uni, calendar, lookback, vs, ve, limit_windows)
    log(f"[{name}] 窗口 {ws.n_before_universe:,} -> universe 内 {len(ws):,}"
        f"（名字 {ws.confounders['n_names']}，交易日 {ws.confounders['n_days']}）")
    del panel, uni
    return ws


# --------------------------------------------------------------------------
# 逐窗 NLL
# --------------------------------------------------------------------------

def _per_window_nll(model, s0: torch.Tensor, s1: torch.Tensor,
                    stamp: torch.Tensor, teacher_forcing: bool) -> torch.Tensor:
    """train.py:111-115 的逐窗版（reduction='none'）。返回 [B] 的 NLL。"""
    t1, t2 = s0[:, 1:], s1[:, 1:]
    if teacher_forcing:
        logits = model(s0[:, :-1], s1[:, :-1], stamp[:, :-1, :],
                       use_teacher_forcing=True, s1_targets=t1)
    else:
        logits = model(s0[:, :-1], s1[:, :-1], stamp[:, :-1, :])
    l1, l2 = logits[0].float(), logits[1].float()
    b, t = t1.shape
    ce1 = F.cross_entropy(l1.reshape(b * t, -1), t1.reshape(-1),
                          reduction="none").view(b, t)
    ce2 = F.cross_entropy(l2.reshape(b * t, -1), t2.reshape(-1),
                          reduction="none").view(b, t)
    return ((ce1 + ce2) / 2).mean(dim=1)


def score_nll(tokenizer, model, ws: WindowSet, device, amp: str | None,
              batch_size: int, sampled_check: int = 0,
              seed: int = 20260905) -> dict:
    """对一个窗口集算逐窗 NLL。返回 {nll: np.ndarray, anchor: np.ndarray, ...}。"""
    tokenizer.to(device).eval()
    model.to(device).eval()
    if amp and device.type == "cuda":
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[amp]

        def ctx():
            return torch.autocast(device_type="cuda", dtype=dtype)
    else:
        ctx = contextlib.nullcontext

    out_nll, out_sampled = [], []
    n = len(ws)
    t0 = time.perf_counter()
    # 采样路径复算（train.py 原样口径）用得到 RNG；主口径是教师强制，与它无关
    torch.manual_seed(seed)
    with torch.no_grad():
        for lo in range(0, n, batch_size):
            hi = min(lo + batch_size, n)
            x_np, st_np = ws.batch(lo, hi)
            x = torch.from_numpy(x_np).to(device)
            stamp = torch.from_numpy(st_np).to(device)
            with ctx():
                s0, s1 = tokenizer.encode(x, half=True)
                nll = _per_window_nll(model, s0, s1, stamp, teacher_forcing=True)
                out_nll.append(nll.float().cpu().numpy())
                if sampled_check and lo < sampled_check:
                    ns = _per_window_nll(model, s0, s1, stamp, teacher_forcing=False)
                    out_sampled.append(ns.float().cpu().numpy())
            if lo and (lo // batch_size) % 50 == 0:
                done = hi / max(n, 1)
                log(f"  [{ws.name}] {hi:,}/{n:,}（{done:.0%}），"
                    f"用时 {time.perf_counter() - t0:.0f}s")
    nll = np.concatenate(out_nll) if out_nll else np.array([])
    res = {
        "nll": nll,
        "anchor": ws.index["anchor"].to_numpy() if len(ws) else np.array([]),
        "permno": ws.index["PERMNO"].to_numpy() if len(ws) else np.array([]),
        "seconds": time.perf_counter() - t0,
    }
    if out_sampled:
        k = int(sum(a.size for a in out_sampled))
        res["nll_sampled_head"] = np.concatenate(out_sampled)
        res["nll_teacher_head"] = nll[:k]
    return res


# --------------------------------------------------------------------------
# 统计
# --------------------------------------------------------------------------

def describe(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if values.size > 1 else None,
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
    }


def _cluster_draws(values: np.ndarray, dates: np.ndarray, n_boot: int,
                   rng: np.random.Generator) -> np.ndarray:
    """按日期聚类的自助：重抽**日期**（不是窗口），返回 n_boot 个均值。"""
    order = np.argsort(dates, kind="stable")
    v, d = values[order], dates[order]
    uniq, starts = np.unique(d, return_index=True)
    ends = np.append(starts[1:], len(d))
    day_sum = np.add.reduceat(v, starts)
    day_cnt = (ends - starts).astype(float)
    k = len(uniq)
    picks = rng.integers(0, k, size=(n_boot, k))
    return day_sum[picks].sum(axis=1) / day_cnt[picks].sum(axis=1)


def cluster_bootstrap_gap(clean: dict, contam: dict, n_boot: int = 2000,
                          seed: int = 20260905) -> dict:
    """两侧各自按日期聚类自助，取差 (clean − contaminated) 的 95% 分位区间。

    两侧日期不重叠（污染窗 ≤ 2024-01，干净窗 ≥ 2024-07），故独立重抽是对的；
    **不把上万个窗当独立样本**（同日窗口高度相关）。
    """
    if clean["nll"].size == 0 or contam["nll"].size == 0:
        return {"n_boot": 0}
    rng = np.random.default_rng(seed)
    a = _cluster_draws(clean["nll"], clean["anchor"], n_boot, rng)
    b = _cluster_draws(contam["nll"], contam["anchor"], n_boot, rng)
    diff = a - b
    point = float(np.mean(clean["nll"]) - np.mean(contam["nll"]))
    return {
        "gap_clean_minus_contaminated": point,
        "ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
        "boot_sd": float(np.std(diff, ddof=1)),
        "n_boot": int(n_boot),
        "n_days_clean": int(np.unique(clean["anchor"]).size),
        "n_days_contaminated": int(np.unique(contam["anchor"]).size),
    }


def yearly_series(res: dict) -> dict:
    if res["nll"].size == 0:
        return {}
    years = pd.DatetimeIndex(res["anchor"]).year
    s = pd.Series(res["nll"]).groupby(years).agg(["count", "mean", "std"])
    return {str(int(y)): {"n": int(r["count"]), "mean": float(r["mean"]),
                          "sd": (None if not np.isfinite(r["std"]) else float(r["std"]))}
            for y, r in s.iterrows()}


# --------------------------------------------------------------------------
# SVG（纯 python，无第三方绘图库）
# --------------------------------------------------------------------------

def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def write_svg(path: Path, payload: dict) -> None:
    """两栏图：左 = 各模型两侧 NLL 的 p10/p50/p90 + 均值；右 = 按年均值折线。"""
    rows = []
    for model, entry in sorted(payload["by_model"].items()):
        for side, d in entry["sides"].items():
            if d.get("n"):
                rows.append((model, side, d))
    if not rows:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg' width='10' height='10'/>",
                        encoding="utf-8")
        return
    lo = min(d["p10"] for _, _, d in rows)
    hi = max(d["p90"] for _, _, d in rows)
    pad = 0.05 * max(hi - lo, 1e-6)
    lo, hi = lo - pad, hi + pad

    W, H = 1180, 520
    L, R, T, B = 70, 30, 56, 90
    pw = 560
    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' "
             f"viewBox='0 0 {W} {H}' font-family='DejaVu Sans,Arial,sans-serif'>",
             f"<rect width='{W}' height='{H}' fill='white'/>",
             f"<text x='{W/2}' y='24' font-size='16' text-anchor='middle'>"
             f"{_esc(payload['title'])}</text>",
             f"<text x='{W/2}' y='42' font-size='11' fill='#666' text-anchor='middle'>"
             f"诊断图，无阈值，不作判定；FT 臂（ZS 臂已按 2026-09-05 裁定抛弃）</text>"]

    # ---- 左：分布 ----
    def y_of(v):
        return H - B - (v - lo) / (hi - lo) * (H - T - B)
    parts.append(f"<line x1='{L}' y1='{T}' x2='{L}' y2='{H-B}' stroke='#333'/>")
    for i in range(6):
        v = lo + (hi - lo) * i / 5
        y = y_of(v)
        parts.append(f"<line x1='{L-4}' y1='{y:.1f}' x2='{L}' y2='{y:.1f}' stroke='#333'/>")
        parts.append(f"<text x='{L-8}' y='{y+4:.1f}' font-size='10' text-anchor='end'>"
                     f"{v:.3f}</text>")
    parts.append(f"<text x='{L-52}' y='{T-10}' font-size='11' fill='#333'>逐窗 NLL</text>")
    step = pw / max(len(rows), 1)
    colors = {"contaminated": "#c0392b", "fold44": "#2980b9", "fold45": "#16a085"}
    for i, (model, side, d) in enumerate(rows):
        x = L + step * (i + 0.5)
        c = colors.get(side, "#7f8c8d")
        parts.append(f"<line x1='{x:.1f}' y1='{y_of(d['p10']):.1f}' x2='{x:.1f}' "
                     f"y2='{y_of(d['p90']):.1f}' stroke='{c}' stroke-width='2'/>")
        parts.append(f"<rect x='{x-7:.1f}' y='{y_of(d['p50'])-1.5:.1f}' width='14' "
                     f"height='3' fill='{c}'/>")
        parts.append(f"<circle cx='{x:.1f}' cy='{y_of(d['mean']):.1f}' r='3.2' "
                     f"fill='white' stroke='{c}' stroke-width='2'/>")
        parts.append(f"<text x='{x:.1f}' y='{H-B+14}' font-size='9' "
                     f"text-anchor='middle' transform='rotate(35 {x:.1f} {H-B+14})'>"
                     f"{_esc(model)}/{_esc(side)}</text>")

    # ---- 右：按年 ----
    X0 = L + pw + 70
    XW = W - R - X0
    years = sorted({int(y) for e in payload["by_model"].values()
                    for d in e["yearly"].values() for y in d})
    if years:
        allv = [v["mean"] for e in payload["by_model"].values()
                for d in e["yearly"].values() for v in d.values()]
        ylo, yhi = min(allv), max(allv)
        ypad = 0.05 * max(yhi - ylo, 1e-6)
        ylo, yhi = ylo - ypad, yhi + ypad

        def yy(v):
            return H - B - (v - ylo) / (yhi - ylo) * (H - T - B)

        def xx(y):
            if len(years) == 1:
                return X0 + XW / 2
            return X0 + (years.index(y)) / (len(years) - 1) * XW
        parts.append(f"<line x1='{X0}' y1='{T}' x2='{X0}' y2='{H-B}' stroke='#333'/>")
        parts.append(f"<line x1='{X0}' y1='{H-B}' x2='{X0+XW}' y2='{H-B}' stroke='#333'/>")
        for i in range(5):
            v = ylo + (yhi - ylo) * i / 4
            parts.append(f"<text x='{X0-8}' y='{yy(v)+4:.1f}' font-size='10' "
                         f"text-anchor='end'>{v:.3f}</text>")
        for y in years:
            parts.append(f"<text x='{xx(y):.1f}' y='{H-B+16}' font-size='10' "
                         f"text-anchor='middle'>{y}</text>")
        palette = ["#c0392b", "#2980b9", "#16a085", "#8e44ad", "#d35400",
                   "#2c3e50", "#7f8c8d"]
        for k, (model, e) in enumerate(sorted(payload["by_model"].items())):
            pts = {}
            for d in e["yearly"].values():
                for y, v in d.items():
                    pts[int(y)] = v["mean"]
            seq = [(xx(y), yy(pts[y])) for y in sorted(pts) if y in years]
            if not seq:
                continue
            c = palette[k % len(palette)]
            path_d = " ".join(("M" if i == 0 else "L") + f"{a:.1f},{b:.1f}"
                              for i, (a, b) in enumerate(seq))
            parts.append(f"<path d='{path_d}' fill='none' stroke='{c}' stroke-width='1.8'/>")
            for a, b in seq:
                parts.append(f"<circle cx='{a:.1f}' cy='{b:.1f}' r='2.6' fill='{c}'/>")
            parts.append(f"<text x='{X0+6}' y='{T+14*(k+1)}' font-size='10' fill='{c}'>"
                         f"{_esc(model)}</text>")
        parts.append(f"<text x='{X0}' y='{T-10}' font-size='11' fill='#333'>"
                     f"按年逐窗 NLL 均值（各 FT 模型）</text>")
    parts.append("</svg>")
    assert_write_ok(path).write_text("\n".join(parts), encoding="utf-8")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

USER_PERMISSION = (
    "干净窗（折 44–45）的原始 OHLCV 面板可以读；"
    "禁止的是那两折的分数、标签与任何指标。"
)

CRITERION = (
    "诊断，无阈值，不作判定。只报分布差异的方向与量级并列出混淆项；"
    "不得据此宣称『没有污染』或『有污染』。"
)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="实验 10 / P3：无标签 NLL membership gap")
    ap.add_argument("--processed", required=True)
    ap.add_argument("--model-folds", default="36,39,42",
                    help="用作 FT 模型（同时提供污染侧窗口）的折号，逗号分隔")
    ap.add_argument("--lookback", type=int, default=90)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--amp", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--device", default=None)
    ap.add_argument("--gpu-mem-fraction", type=float, default=0.20)
    ap.add_argument("--limit-windows", type=int, default=0,
                    help="每个窗口集随机抽样上限（0=全量；仅冒烟/计时用，会写进 JSON）")
    ap.add_argument("--sampled-check", type=int, default=2560,
                    help="额外用 train.py 原样的采样路径复算的前 N 个窗（口径差量化）")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--memory-limit-gb", type=float, default=42.0)
    ap.add_argument("--out", default="outputs/exp10_contamination_probes.json")
    ap.add_argument("--svg", default="outputs/exp10_p3_nll.svg")
    ap.add_argument("--skip-clean", action="store_true",
                    help="只跑污染侧（计时校准用；不产出 gap）")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    tripped = install_label_tripwire()
    log(f"label 绊线已装（一次都不许调用）：{tripped}")
    torch.set_num_threads(2)

    processed = Path(args.processed)
    model_folds = [int(s) for s in args.model_folds.split(",") if s.strip()]
    bad = [f for f in model_folds if f not in ALLOWED_MODEL_FOLDS]
    if bad:
        raise BoundaryError(f"折 {bad} 不在允许的模型折白名单 {ALLOWED_MODEL_FOLDS} 内。")

    folds = emit_folds(processed, 36, 45)
    log("emit_folds.py（机械生成，未手写）："
        + json.dumps({k: folds[k] for k in sorted(folds)}, ensure_ascii=False))

    cal = TradingCalendar.from_market_index(
        read_parquet_guarded(processed / "market_index.parquet"), "caldt")

    dev = pick_device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(args.gpu_mem_fraction))
        log(f"显存封顶 {args.gpu_mem_fraction}（工程参数，不改数值）")

    # ---- 窗口集 ----
    wait_memory_gate(args.memory_limit_gb)
    sets: dict[str, WindowSet] = {}
    if not args.skip_clean:
        for f in CLEAN_FOLDS:
            wait_memory_gate(args.memory_limit_gb)
            sets[f"fold{f}"] = load_window_set(
                processed, cal, f"fold{f}", folds[f]["vs"], folds[f]["ve"],
                args.lookback, args.limit_windows)
    for f in model_folds:
        wait_memory_gate(args.memory_limit_gb)
        sets[f"fold{f}"] = load_window_set(
            processed, cal, f"fold{f}", folds[f]["vs"], folds[f]["ve"],
            args.lookback, args.limit_windows)

    # ---- 逐模型打 NLL ----
    by_model: dict[str, dict] = {}
    total_seconds = 0.0
    for f in model_folds:
        mdir = REPO_ROOT / "outputs" / f"fold{f:02d}_lb90_s0_poolB_universe"
        assert_read_ok(mdir / "tokenizer_final" / "model.safetensors")
        assert_read_ok(mdir / "predictor_final" / "model.safetensors")
        wait_memory_gate(args.memory_limit_gb)
        log(f"加载 FT 模型 fold{f} …")
        tokenizer, model = load_pretrained(str(mdir / "tokenizer_final"),
                                           str(mdir / "predictor_final"))
        entry = {"model_dir": _rel(mdir), "sides": {}, "yearly": {}, "gap": {},
                 "raw_seconds": {}}
        names = [f"fold{f}"] + ([] if args.skip_clean
                                else [f"fold{c}" for c in CLEAN_FOLDS])
        raw: dict[str, dict] = {}
        for nm in names:
            ws = sets[nm]
            log(f"NLL：模型 fold{f} × 窗口集 {nm}（{len(ws):,} 窗）…")
            r = score_nll(tokenizer, model, ws, dev, args.amp, args.batch_size,
                          sampled_check=args.sampled_check)
            raw[nm] = r
            total_seconds += r["seconds"]
            side = "contaminated" if nm == f"fold{f}" else nm
            entry["sides"][side] = describe(r["nll"]) | {
                "window_set": nm,
                "val_window": ws.confounders.get("val_window"),
                "confounders": ws.confounders,
            }
            entry["yearly"][side] = yearly_series(r)
            entry["raw_seconds"][side] = r["seconds"]
            if "nll_sampled_head" in r:
                a, b = r["nll_teacher_head"], r["nll_sampled_head"]
                entry["sides"][side]["sampled_path_check"] = {
                    "n": int(a.size),
                    "mean_teacher_forced": float(np.mean(a)),
                    "mean_sampled_train_py_path": float(np.mean(b)),
                    "mean_diff": float(np.mean(b - a)),
                    "corr": (float(np.corrcoef(a, b)[0, 1]) if a.size > 2 else None),
                }
        if not args.skip_clean:
            for c in CLEAN_FOLDS:
                entry["gap"][f"fold{c}"] = cluster_bootstrap_gap(
                    raw[f"fold{c}"], raw[f"fold{f}"], n_boot=args.n_boot)
        by_model[f"fold{f}"] = entry
        del tokenizer, model, raw
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "experiment": "exp10-P3 无标签 NLL membership gap",
        "title": "P3：FT 模型在污染窗 vs 干净窗的逐窗 NLL",
        "criterion": CRITERION,
        "user_permission_verbatim": USER_PERMISSION,
        "zero_shot_arm": ("ZS 臂已按用户 2026-09-05 裁定整体抛弃，故 P3 只在 FT 臂上做；"
                          "非漏跑。口径见 experiments/"
                          "confirmation_protocol_v4.1_addendum_2026-09-05.md（只读不改）。"),
        "qualifications": [
            "fold44–45 恰是 Kronos 作者自己的测试期，checkpoint 级的开发者选择不外生——"
            "不得把干净窗当成完全外生的样本。",
            "FT 模型除预训练语料外还见过自己的训练窗；污染侧窗口紧邻训练窗、干净侧远离，"
            "两侧差异同时含『预训练记忆』与『微调近因』两个来源，本设计无法分开。",
            "Kronos 输入端逐窗 z-score + clip（third_party/kronos/model/kronos.py:544-547 与 "
            ":629-631），归一化本身已压掉水平差异；NLL 差只能反映形状层面的差异。",
            "两侧不是随机分配：年代、波动率、成交额、名字集全都不同（见 confounders）。",
        ],
        "config": {
            "arm": "FT only",
            "lookback": args.lookback, "predict": 0,
            "amp": args.amp, "batch_size": args.batch_size,
            "device": str(dev), "gpu_mem_fraction": args.gpu_mem_fraction,
            "limit_windows": args.limit_windows,
            "sampled_check": args.sampled_check,
            "n_boot": args.n_boot,
            "input_panel": "panel_kronos_adj.parquet（§5 复权后 OHLCV，与训练/打分一致）",
            "nll_source": "src/kronos_ft/train.py:99-115 predictor 分支（逐窗化 + 教师强制）",
            "normalisation": "逐窗 mean/std + clip 5.0（dataset.py:121-125 / infer.py:158-159）",
        },
        "folds_from_emit_folds": {str(k): folds[k] for k in sorted(folds)},
        "by_model": by_model,
        "total_gpu_seconds": total_seconds,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out = assert_write_ok(REPO_ROOT / args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float),
                   encoding="utf-8")
    write_svg(REPO_ROOT / args.svg, payload)
    log(f"完成 → {_rel(out)}；GPU 合计 {total_seconds/60:.1f} 分钟")


if __name__ == "__main__":
    main()
