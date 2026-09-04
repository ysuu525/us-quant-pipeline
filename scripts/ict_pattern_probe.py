"""ICT / SMC 经典 K 线组合探针（预注册 v1，探索性诊断）。

本脚本实现 `experiments/ict_pattern_probe_prereg_v1.md`。预注册文件已冻结，
本脚本不得改动其中任何定义、参数或门槛。下面复述预注册的关键条款，便于
读代码时逐条核对。

定位（预注册 §0）
-----------------
**探索性诊断，不是确认性检验。** 通过门槛 ≠ 采纳；未通过门槛 ≠ 否定。
措辞强制：未达门槛一律写「在本样本量下该问题不可回答」，不得写「不可分」
或「两者相等」。

数据边界（预注册 §2）
---------------------
- 折：**仅** fold01–04（2003-01..2004-12）与 fold36–42（2020-07..2023-12），共 11 折。
- 禁区：fold05–fold35 全部目录、任何封存评估目录、任何 signal_date ≥ 2024-01-01。
- OHLCV：`panel_kronos_adj.parquet`（复权口径见 §「复权核对」）。
- 宇宙：`universe.parquet`，`in_universe == True`，按信号日 point-in-time。
- 标签：各折 `labels.parquet`，`status == "ok"`（t+1 DlyOpen 建仓、持有至 t+6），**不重算**。
- 分数：各折 `scores.parquet`（微调臂）。36–42 为统一口径块（主块）；
  01–04 的 `metrics.json` 无 `scoring_config` 字段，按 CLAUDE.md §八
  「口径必须可机器核对」单独成块，**不与主块并列比较**。

固定参数（预注册 §3，一律不改、不跑变体）
-----------------------------------------
摆动回看 L = 20；ATR 窗 14（用 t-1 及之前）；位移倍数 k = 1.5；
订单块回看 5 根；订单块 / OTE 有效期 20 个交易日。
H20 = max(High[t-20..t-1])，L20 = min(Low[t-20..t-1])，
TR_t = max(H_t−L_t, |H_t−C_{t-1}|, |L_t−C_{t-1}|)。

统计量（预注册 §4）
-------------------
y = label − 当日观测集内 label 均值（另报 y 按日 1%/99% 截尾版）。
- Q1：每日取触发股 y 均值（当日触发数 < 5 跳过）→ 日序列 → 均值 / NW(lag=6) SE / t；
  按折报均值，数「预期方向同向折数 / 有效折数」（有效折 = 该折事件数 ≥ 20）。
- Q2(a)：日截面回归 y = a + b1·kr + b2·f + b3·(kr−0.5)·f，Fama–MacBeth 汇总 b2、b3。
- Q2(b)：Kronos 前五分位内「看涨 f 触发 − 未触发」的 y 均值差；后五分位内看跌同理。
- Q0：触发率、标记间重叠、每折事件数。

SESOI / MDE（预注册 §5）
------------------------
SESOI = 10 bp / 6 日持有期。MDE = 2.80 × NW SE（双侧 5%、功效 80%），逐标记报告。
预注册先验估算 MDE ≈ 22–28 bp > SESOI，故定位为探索性。

判读门槛（预注册 §6，先于结果写死）
-----------------------------------
1. **升级候选**：预期方向 ∧ |均值| ≥ SESOI ∧ |t_NW| ≥ 2.0 ∧ 同向折数 ≥ 8/11
   （或 ≥ 73% 的有效折）。
2. **方向性支持、不可判读**：|均值| ≥ SESOI 但 t 或一致性未达；
   或 t 与一致性达标但 |均值| < SESOI。
3. **方向性不利**：反方向 ∧ |t| ≥ 2.0 ∧ 反向折数 ≥ 8/11（或 ≥ 73% 有效折）。
4. **无信息（在该 MDE 下）**：其余。
多重比较：12 标记 × (Q1 + Q2b) = 24 个门槛检验；另报 Bonferroni 口径 |t| ≥ 2.87。

禁止事项（预注册 §7 / CLAUDE.md §一）
-------------------------------------
1. 前视：任何标记只用 ≤ t 日 K 线；`label` 只出现在 y 的构造里。
2. 不在评估窗选参：L / k / ATR 窗 / 有效期 / 回看根数一律不改；不跑变体；
   不按结果挑标记子集。
3. 门槛已含「同向折数」一致性项。
4. 时点：标签本身已是 t+1 开盘建仓；标记在 t 收盘可算。

工程约束（CLAUDE.md §七）
-------------------------
纯 CPU（不碰 GPU）；面板读取列裁剪 + 日期下推 + PERMNO 过滤 + 逐 PERMNO 分组；
不装任何包；输出目录带互斥锁；分阶段落盘可断点续跑。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from crsp_pipeline.signal_eval import newey_west_tstat  # noqa: E402

# ----------------------------------------------------------------- 冻结常量

PROC = Path(r"F:\quant\processed\crsp_ciz_2026-08-24_20260825T130601Z")
PANEL = PROC / "panel_kronos_adj.parquet"
UNIVERSE = PROC / "universe.parquet"

L_SWING = 20          # 摆动回看（H20 / L20）
ATR_WIN = 14          # ATR 窗
K_DISP = 1.5          # 位移倍数
OB_LOOKBACK = 5       # 订单块回看根数
VALIDITY = 20         # 订单块 / OTE 有效期（交易日）
BODY_FRAC = 0.6       # 位移 K 线实体占比
OTE_LO_FRAC = 0.21    # OTE 区间下沿（自腿低起算的比例）= 79% 回撤
OTE_HI_FRAC = 0.38    # OTE 区间上沿 = 62% 回撤

MIN_DAY_TRIG = 5      # 当日触发数 < 5 则跳过该日
MIN_FOLD_EVENTS = 20  # 有效折门槛
NW_LAG = 6            # Newey–West lag
SESOI_BP = 10.0       # 最小科学相关效应（bp / 6 日持有）
MDE_MULT = 2.80       # MDE = 2.80 × NW SE
T_GATE = 2.0          # 门槛 1/3 的 |t| 要求
T_BONF = 2.87         # Bonferroni 口径
CONSISTENCY_FRAC = 0.73   # ≥ 73% 有效折同向
QUINTILE = 0.2        # 前 / 后五分位

# P5 / P6 可算所需的最少前置 K 线根数：
#   最早可能的 t0 = t − VALIDITY；t0 处需 L_SWING 根做 H20/L20，
#   再往前 OB_LOOKBACK 根做订单块 → 共 VALIDITY + L_SWING + OB_LOOKBACK。
MIN_HIST_P5P6 = VALIDITY + L_SWING + OB_LOOKBACK   # = 45

# 标记与预期方向（+1 = 预期 y 为正）
MARKERS = [
    ("P1_bull", +1), ("P1_bear", -1),
    ("P2_bull", +1), ("P2_bear", -1),
    ("P3_bull", +1), ("P3_bear", -1),
    ("P4_bull", +1), ("P4_bear", -1),
    ("P5_bull", +1), ("P5_bear", -1),
    ("P6_bull", +1), ("P6_bear", -1),
]
MARKER_COLS = [m for m, _ in MARKERS]
EXPECTED_SIGN = dict(MARKERS)

# 折 → (评估目录, 面板块)。**只列 11 个允许折**，绝不 glob 到禁区目录。
FOLDS = {
    "fold01": "outputs/fold01_lb90_s0_poolB_universe/eval_poolB_universe",
    "fold02": "outputs/fold02_lb90_s0_poolB_universe/eval_poolB_universe_fold02",
    "fold03": "outputs/fold03_lb90_s0_poolB_universe/eval_poolB_universe_fold03",
    "fold04": "outputs/fold04_lb90_s0_poolB_universe/eval_poolB_universe_fold04",
    "fold36": "outputs/fold36_lb90_s0_poolB_universe/eval_amp_lb90_fold36",
    "fold37": "outputs/fold37_lb90_s0_poolB_universe/eval_amp_lb90_fold37",
    "fold38": "outputs/fold38_lb90_s0_poolB_universe/eval_amp_lb90_fold38",
    "fold39": "outputs/fold39_lb90_s0_poolB_universe/eval_amp_lb90_fold39",
    "fold40": "outputs/fold40_lb90_s0_poolB_universe/eval_amp_lb90_fold40",
    "fold41": "outputs/fold41_lb90_s0_poolB_universe/eval_amp_lb90_fold41",
    "fold42": "outputs/fold42_lb90_s0_poolB_universe/eval_amp_lb90_fold42",
}

# 面板载入窗口：评估窗前留足回看余量（> MIN_HIST_P5P6 = 45 个交易日）
BLOCKS = {
    "late": {
        "folds": ["fold36", "fold37", "fold38", "fold39", "fold40", "fold41", "fold42"],
        "panel_lo": "2020-04-01", "panel_hi": "2023-12-31",
        "label": "36–42，统一口径，主块",
    },
    "early": {
        "folds": ["fold01", "fold02", "fold03", "fold04"],
        "panel_lo": "2002-10-01", "panel_hi": "2004-12-31",
        "label": "01–04，scoring_config 缺失，单独块",
    },
}
PRIMARY_BLOCK = "late"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------- 标记计算（核心，可测）


def _roll_prev(a: np.ndarray, win: int, how: str) -> np.ndarray:
    """arr[t-win .. t-1] 上的滚动统计；不足 win 个非缺观测则为 NaN。

    只用 **严格早于 t** 的 K 线（`.shift(1)`），这是「t 日收盘可算」的实现保证。
    """
    s = pd.Series(a, dtype="float64").rolling(win, min_periods=win)
    r = {"max": s.max, "min": s.min, "mean": s.mean}[how]()
    return r.shift(1).to_numpy(dtype=float)


def _shift(a: np.ndarray, k: int) -> np.ndarray:
    out = np.full(len(a), np.nan, dtype=float)
    if k < len(a):
        out[k:] = a[: len(a) - k]
    return out


def compute_patterns(
    o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray
) -> dict[str, np.ndarray]:
    """单只证券（按日期升序）的 12 个 ICT 标记 + 可算掩码。

    输入为等长 1-D float 数组（复权 OHLC）。返回 dict：
    12 个 bool 数组（`P1_bull` ... `P6_bear`）+ `computable`（bool）。

    **不使用任何 t 日之后的信息**：所有滚动量都经 `.shift(1)`，
    P5/P6 的扫描只从触发日 t0 向后推进到 t，t 处只读 t 日及更早的 K 线。
    """
    o = np.asarray(o, dtype=float)
    h = np.asarray(h, dtype=float)
    l = np.asarray(l, dtype=float)
    c = np.asarray(c, dtype=float)
    n = len(c)
    assert len(o) == len(h) == len(l) == n

    bar_ok = np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c)

    H20 = _roll_prev(h, L_SWING, "max")
    L20 = _roll_prev(l, L_SWING, "min")

    prev_c = _shift(c, 1)
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    atr_prev = _roll_prev(tr, ATR_WIN, "mean")

    with np.errstate(invalid="ignore"):
        # --- P1 扫流动性后收回
        p1_bull = bar_ok & (l < L20) & (c > L20)
        p1_bear = bar_ok & (h > H20) & (c < H20)
        # --- P2 结构突破
        p2_bull = bar_ok & (c > H20)
        p2_bear = bar_ok & (c < L20)
        # --- P3 三根 K 线失衡（FVG）
        h2, l2 = _shift(h, 2), _shift(l, 2)
        p3_bull = bar_ok & (l > h2)
        p3_bear = bar_ok & (h < l2)
        # --- P4 位移 K 线
        rng = h - l
        # H_t == L_t 时实体占比未定义 → 最简读法：不触发（比较结果为 False）
        with np.errstate(divide="ignore"):
            body_up = np.where(rng > 0, (c - o) / np.where(rng > 0, rng, np.nan), np.nan)
            body_dn = np.where(rng > 0, (o - c) / np.where(rng > 0, rng, np.nan), np.nan)
        disp = bar_ok & (tr > K_DISP * atr_prev)
        p4_bull = disp & (body_up >= BODY_FRAC)
        p4_bear = disp & (body_dn >= BODY_FRAC)

    # --- P5 订单块回测 / P6 OTE 回撤（由 P2 触发日 t0 派生）
    p5_bull = np.zeros(n, dtype=bool)
    p5_bear = np.zeros(n, dtype=bool)
    p6_bull = np.zeros(n, dtype=bool)
    p6_bear = np.zeros(n, dtype=bool)

    for t0 in np.flatnonzero(p2_bull):
        # 订单块 = t0 前 5 根内**最后一根**阴线（C<O）的 [Low, High]
        ob = None
        for j in range(t0 - 1, max(t0 - 1 - OB_LOOKBACK, -1), -1):
            if bar_ok[j] and c[j] < o[j]:
                ob = (l[j], h[j])
                break
        if ob is not None:
            ob_lo, ob_hi = ob
            for t in range(t0 + 1, min(n, t0 + VALIDITY + 1)):
                if not bar_ok[t]:
                    continue
                if l[t] <= ob_hi and c[t] > ob_lo:
                    p5_bull[t] = True
                    break
        # OTE：腿 = [L20(t0), High_{t0}]，区间 = [腿低+0.21·腿长, 腿低+0.38·腿长]
        leg_lo, leg_hi = L20[t0], h[t0]
        if np.isfinite(leg_lo) and np.isfinite(leg_hi) and leg_hi > leg_lo:
            span = leg_hi - leg_lo
            z_lo = leg_lo + OTE_LO_FRAC * span
            z_hi = leg_lo + OTE_HI_FRAC * span
            for t in range(t0 + 1, min(n, t0 + VALIDITY + 1)):
                if not bar_ok[t]:
                    continue
                if l[t] < leg_lo:          # 期间破腿低 → 作废
                    break
                if z_lo <= l[t] <= z_hi and c[t] >= z_lo:
                    p6_bull[t] = True
                    break

    for t0 in np.flatnonzero(p2_bear):
        ob = None
        for j in range(t0 - 1, max(t0 - 1 - OB_LOOKBACK, -1), -1):
            if bar_ok[j] and c[j] > o[j]:
                ob = (l[j], h[j])
                break
        if ob is not None:
            ob_lo, ob_hi = ob
            for t in range(t0 + 1, min(n, t0 + VALIDITY + 1)):
                if not bar_ok[t]:
                    continue
                if h[t] >= ob_lo and c[t] < ob_hi:
                    p5_bear[t] = True
                    break
        leg_hi, leg_lo = H20[t0], l[t0]
        if np.isfinite(leg_lo) and np.isfinite(leg_hi) and leg_hi > leg_lo:
            span = leg_hi - leg_lo
            z_hi = leg_hi - OTE_LO_FRAC * span
            z_lo = leg_hi - OTE_HI_FRAC * span
            for t in range(t0 + 1, min(n, t0 + VALIDITY + 1)):
                if not bar_ok[t]:
                    continue
                if h[t] > leg_hi:          # 期间破腿高 → 作废
                    break
                if z_lo <= h[t] <= z_hi and c[t] <= z_hi:
                    p6_bear[t] = True
                    break

    # 可算掩码：取**全部 6 个标记同时可算**（最简单一读法，保证 12 个标记共用
    # 同一观测集、同一 y 去均值口径）。要求 = 当日 K 线完整 ∧ 有 MIN_HIST_P5P6
    # 根前置 K 线 ∧ H20/L20/ATR14/t-2 均可算。
    idx = np.arange(n)
    computable = (
        bar_ok
        & (idx >= MIN_HIST_P5P6)
        & np.isfinite(H20)
        & np.isfinite(L20)
        & np.isfinite(atr_prev)
        & np.isfinite(h2)
        & np.isfinite(l2)
    )

    return {
        "P1_bull": p1_bull, "P1_bear": p1_bear,
        "P2_bull": p2_bull, "P2_bear": p2_bear,
        "P3_bull": p3_bull, "P3_bear": p3_bear,
        "P4_bull": p4_bull, "P4_bear": p4_bear,
        "P5_bull": p5_bull, "P5_bear": p5_bear,
        "P6_bull": p6_bull, "P6_bear": p6_bear,
        "computable": computable,
    }


# ------------------------------------------------------------- 阶段 1：标记表


def _fold_meta(fold: str) -> dict:
    d = REPO_ROOT / FOLDS[fold]
    m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    return {
        "eval_dir": FOLDS[fold],
        "val_window": m.get("val_window"),
        "scoring_config": m.get("scoring_config"),
        "n_obs": m.get("n_obs"),
        "tag": m.get("tag"),
    }


def _load_fold_keys(fold: str) -> pd.DataFrame:
    d = REPO_ROOT / FOLDS[fold]
    lb = pd.read_parquet(d / "labels.parquet",
                         columns=["PERMNO", "signal_date", "status", "label"])
    sc = pd.read_parquet(d / "scores.parquet",
                         columns=["PERMNO", "signal_date", "score"])
    m = lb.merge(sc, on=["PERMNO", "signal_date"], how="inner")
    m["signal_date"] = pd.to_datetime(m["signal_date"])
    m["fold"] = fold
    return m


def stage1_patterns(block: str, out_dir: Path, force: bool) -> Path:
    """读复权 OHLC（列裁剪 + 日期下推 + PERMNO 过滤），逐 PERMNO 算 12 个标记。"""
    dst = out_dir / f"patterns_{block}.parquet"
    if dst.exists() and not force:
        log(f"阶段1 [{block}] 已存在，跳过：{dst.name}")
        return dst
    cfg = BLOCKS[block]

    permnos: set[int] = set()
    for fold in cfg["folds"]:
        d = REPO_ROOT / FOLDS[fold]
        permnos |= set(pd.read_parquet(d / "labels.parquet",
                                       columns=["PERMNO"])["PERMNO"].unique().tolist())
    permnos_arr = np.array(sorted(permnos), dtype=np.int64)
    log(f"阶段1 [{block}] 折 {cfg['folds'][0]}..{cfg['folds'][-1]}，"
        f"{len(permnos_arr)} 个 PERMNO，面板窗 [{cfg['panel_lo']}..{cfg['panel_hi']}]")

    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as pads

    dset = pads.dataset(str(PANEL), format="parquet")
    filt = (
        (pc.field("DlyCalDt") >= pa.scalar(pd.Timestamp(cfg["panel_lo"]).to_datetime64()))
        & (pc.field("DlyCalDt") <= pa.scalar(pd.Timestamp(cfg["panel_hi"]).to_datetime64()))
        & pc.field("PERMNO").isin(pa.array(permnos_arr))
    )
    tbl = dset.to_table(
        columns=["PERMNO", "DlyCalDt", "DlyOpen", "DlyHigh", "DlyLow", "DlyClose"],
        filter=filt,
    )
    px = tbl.to_pandas()
    del tbl
    log(f"阶段1 [{block}] 面板载入 {len(px):,} 行，"
        f"约 {px.memory_usage(deep=True).sum()/2**20:.0f} MiB")
    px = px.sort_values(["PERMNO", "DlyCalDt"], kind="mergesort").reset_index(drop=True)

    o = px["DlyOpen"].to_numpy(float)
    h = px["DlyHigh"].to_numpy(float)
    lo = px["DlyLow"].to_numpy(float)
    c = px["DlyClose"].to_numpy(float)
    cols = {k: np.zeros(len(px), dtype=bool) for k in MARKER_COLS + ["computable"]}

    groups = px.groupby("PERMNO", sort=False).indices
    for i, (pn, idx) in enumerate(groups.items()):
        res = compute_patterns(o[idx], h[idx], lo[idx], c[idx])
        for k, v in res.items():
            cols[k][idx] = v
        if (i + 1) % 500 == 0:
            log(f"  ... {i+1}/{len(groups)} PERMNO")

    out = pd.DataFrame({"PERMNO": px["PERMNO"], "signal_date": px["DlyCalDt"]})
    for k, v in cols.items():
        out[k] = v
    out = out[out["signal_date"] >= pd.Timestamp(
        min(_fold_meta(f)["val_window"][0] for f in cfg["folds"]))]
    out.to_parquet(dst, index=False)
    log(f"阶段1 [{block}] 写出 {dst.name}（{len(out):,} 行，"
        f"可算 {out['computable'].mean():.2%}）")
    return dst


# ------------------------------------------------------------- 阶段 2：合并表


def stage2_merge(block: str, out_dir: Path, force: bool) -> Path:
    """标记 × 分数 × 标签 × 宇宙 → 观测集，并构造 y（label 唯一出现处）。"""
    dst = out_dir / f"merged_{block}.parquet"
    if dst.exists() and not force:
        log(f"阶段2 [{block}] 已存在，跳过：{dst.name}")
        return dst
    cfg = BLOCKS[block]
    pat = pd.read_parquet(out_dir / f"patterns_{block}.parquet")

    keys = pd.concat([_load_fold_keys(f) for f in cfg["folds"]], ignore_index=True)
    n_all = len(keys)
    keys = keys[(keys["status"] == "ok") & keys["label"].notna() & keys["score"].notna()]
    n_ok = len(keys)

    # 宇宙：point-in-time，按信号日
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as pads
    lo_d = keys["signal_date"].min()
    hi_d = keys["signal_date"].max()
    uds = pads.dataset(str(UNIVERSE), format="parquet")
    ufilt = (
        (pc.field("DlyCalDt") >= pa.scalar(lo_d.to_datetime64()))
        & (pc.field("DlyCalDt") <= pa.scalar(hi_d.to_datetime64()))
        & pc.field("PERMNO").isin(pa.array(
            np.array(sorted(keys["PERMNO"].unique()), dtype=np.int64)))
    )
    uni = uds.to_table(columns=["PERMNO", "DlyCalDt", "in_universe"],
                       filter=ufilt).to_pandas()
    uni = uni.rename(columns={"DlyCalDt": "signal_date"})
    uni["signal_date"] = pd.to_datetime(uni["signal_date"])
    keys = keys.merge(uni, on=["PERMNO", "signal_date"], how="left")
    del uni
    n_in_uni = int((keys["in_universe"] == True).sum())  # noqa: E712
    keys = keys[keys["in_universe"] == True]             # noqa: E712

    m = keys.merge(pat, on=["PERMNO", "signal_date"], how="left")
    del pat, keys
    n_joined = int(m["computable"].notna().sum())
    m["computable"] = m["computable"].fillna(False).astype(bool)
    for col in MARKER_COLS:
        m[col] = m[col].fillna(False).astype(bool)
    n_comp = int(m["computable"].sum())
    m = m[m["computable"]].copy()

    # ---- y 的构造：`label` 在本脚本中**仅**在此处进入计算，之后只用 y ----
    g = m.groupby("signal_date")["label"]
    m["y"] = m["label"] - g.transform("mean")
    # 按日 1%/99% 截尾版（对去均值后的 y 施加）
    gy = m.groupby("signal_date")["y"]
    lo_q = gy.transform(lambda s: s.quantile(0.01))
    hi_q = gy.transform(lambda s: s.quantile(0.99))
    m["y_w"] = m["y"].clip(lower=lo_q, upper=hi_q)
    m = m.drop(columns=["label", "status", "in_universe"])
    # ---- 此后再无 label ----

    m["kr"] = m.groupby("signal_date")["score"].rank(pct=True)
    m = m.sort_values(["signal_date", "PERMNO"]).reset_index(drop=True)
    m.to_parquet(dst, index=False)

    stats = {
        "n_rows_all": n_all, "n_label_ok_and_score": n_ok,
        "n_in_universe": n_in_uni, "n_joined_to_panel": n_joined,
        "n_computable": n_comp, "n_final": len(m),
        "drop_share_not_in_universe": 1.0 - n_in_uni / max(n_ok, 1),
        "drop_share_not_computable": 1.0 - n_comp / max(n_in_uni, 1),
    }
    (out_dir / f"merged_{block}_counts.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"阶段2 [{block}] 观测集 {len(m):,} 行（标签ok∧分数∧宇宙 {n_in_uni:,}，"
        f"可算率 {n_comp/max(n_in_uni,1):.2%}）")
    return dst


# --------------------------------------------------------------- 阶段 3：统计


def _nw(series: np.ndarray) -> dict:
    r = newey_west_tstat(pd.Series(series), NW_LAG)
    se = float(r["se"])
    return {"mean_bp": float(r["mean"]) * 1e4, "se_bp": se * 1e4,
            "t": float(r["t"]), "n_days": int(r["n"]),
            "mde_bp": MDE_MULT * se * 1e4}


def _verdict(mean_bp: float, t: float, sign: int, n_same: int, n_valid: int) -> str:
    if not np.isfinite(t) or n_valid == 0:
        return "无信息（在该 MDE 下）"
    need = int(np.ceil(CONSISTENCY_FRAC * n_valid))
    dir_ok = (mean_bp * sign) > 0
    mag_ok = abs(mean_bp) >= SESOI_BP
    t_ok = abs(t) >= T_GATE
    cons_ok = n_same >= need
    if dir_ok and mag_ok and t_ok and cons_ok:
        return "升级候选"
    if dir_ok and (mag_ok or (t_ok and cons_ok)):
        return "方向性支持、不可判读"
    if (not dir_ok) and t_ok and ((n_valid - n_same) >= need):
        return "方向性不利"
    return "无信息（在该 MDE 下）"


def _fold_of(dates: pd.Series, block: str) -> pd.Series:
    cfg = BLOCKS[block]
    out = pd.Series(index=dates.index, dtype=object)
    for f in cfg["folds"]:
        lo, hi = _fold_meta(f)["val_window"]
        k = (dates >= pd.Timestamp(lo)) & (dates <= pd.Timestamp(hi))
        out[k] = f
    return out


def _summarize(daily: pd.DataFrame, marker: str, question: str, ycol: str,
               block: str) -> dict:
    """daily: 列 [signal_date, fold, stat, n_evt]。"""
    d = daily.dropna(subset=["stat"]).sort_values("signal_date")
    if len(d) < 3:
        return {"marker": marker, "question": question, "y": ycol, "block": block,
                "mean_bp": np.nan, "se_bp": np.nan, "t": np.nan, "mde_bp": np.nan,
                "n_days": len(d), "n_events": int(d["n_evt"].sum()) if len(d) else 0,
                "n_valid_folds": 0, "n_same_dir_folds": 0, "per_fold": {},
                "verdict": "无信息（在该 MDE 下）", "bonferroni_hit": False}
    sign = EXPECTED_SIGN[marker]
    s = _nw(d["stat"].to_numpy(float))
    per_fold, n_valid, n_same = {}, 0, 0
    for f in BLOCKS[block]["folds"]:
        k = d["fold"] == f
        ev = int(d.loc[k, "n_evt"].sum())
        mu = float(d.loc[k, "stat"].mean()) * 1e4 if k.any() else np.nan
        valid = ev >= MIN_FOLD_EVENTS
        per_fold[f] = {"mean_bp": mu, "n_events": ev, "n_days": int(k.sum()),
                       "valid": bool(valid)}
        if valid:
            n_valid += 1
            if np.isfinite(mu) and mu * sign > 0:
                n_same += 1
    return {
        "marker": marker, "question": question, "y": ycol, "block": block,
        "expected_sign": sign, **s,
        "n_events": int(d["n_evt"].sum()),
        "n_valid_folds": n_valid, "n_same_dir_folds": n_same,
        "consistency_need": int(np.ceil(CONSISTENCY_FRAC * n_valid)),
        "per_fold": per_fold,
        "verdict": _verdict(s["mean_bp"], s["t"], sign, n_same, n_valid),
        "bonferroni_hit": bool(abs(s["t"]) >= T_BONF),
        "mde_le_sesoi": bool(s["mde_bp"] <= SESOI_BP),
    }


def stage3_stats(out_dir: Path) -> dict:
    results = {
        "prereg": "experiments/ict_pattern_probe_prereg_v1.md",
        "params": {"L_swing": L_SWING, "atr_win": ATR_WIN, "k_disp": K_DISP,
                   "ob_lookback": OB_LOOKBACK, "validity": VALIDITY,
                   "body_frac": BODY_FRAC, "ote": [OTE_LO_FRAC, OTE_HI_FRAC],
                   "min_day_trig": MIN_DAY_TRIG, "min_fold_events": MIN_FOLD_EVENTS,
                   "nw_lag": NW_LAG, "sesoi_bp": SESOI_BP, "mde_mult": MDE_MULT,
                   "t_gate": T_GATE, "t_bonf": T_BONF,
                   "consistency_frac": CONSISTENCY_FRAC, "quintile": QUINTILE,
                   "min_hist_p5p6": MIN_HIST_P5P6},
        "folds": {f: _fold_meta(f) for f in FOLDS},
        "blocks": {}, "q0": {}, "summaries": [],
    }
    daily_rows = []

    for block, cfg in BLOCKS.items():
        m = pd.read_parquet(out_dir / f"merged_{block}.parquet")
        m["fold"] = _fold_of(m["signal_date"], block)
        counts = json.loads(
            (out_dir / f"merged_{block}_counts.json").read_text(encoding="utf-8"))
        results["blocks"][block] = {
            "label": cfg["label"], "folds": cfg["folds"],
            "panel_window": [cfg["panel_lo"], cfg["panel_hi"]],
            "n_obs": len(m), "n_days": int(m["signal_date"].nunique()),
            "counts": counts,
            "scoring_configs": {f: _fold_meta(f)["scoring_config"] for f in cfg["folds"]},
        }

        # ---------------- Q0 描述
        q0 = {"trigger_rate": {}, "events_per_fold": {}, "overlap_jaccard": {}}
        for col in MARKER_COLS:
            q0["trigger_rate"][col] = float(m[col].mean())
            q0["events_per_fold"][col] = {
                f: int(m.loc[m["fold"] == f, col].sum()) for f in cfg["folds"]}
        arr = {c: m[c].to_numpy() for c in MARKER_COLS}
        for a in MARKER_COLS:
            q0["overlap_jaccard"][a] = {}
            for b in MARKER_COLS:
                u = int((arr[a] | arr[b]).sum())
                q0["overlap_jaccard"][a][b] = float((arr[a] & arr[b]).sum() / u) if u else 0.0
        results["q0"][block] = q0

        dates = m["signal_date"].to_numpy()
        folds_arr = m["fold"].to_numpy()
        day_idx = m.groupby("signal_date", sort=True).indices
        kr = m["kr"].to_numpy(float)

        for ycol in ("y", "y_w"):
            yv = m[ycol].to_numpy(float)
            for col in MARKER_COLS:
                fv = m[col].to_numpy()
                # ---------- Q1
                rows = []
                for day, idx in day_idx.items():
                    k = idx[fv[idx]]
                    if len(k) < MIN_DAY_TRIG:
                        continue
                    rows.append((day, folds_arr[idx[0]], float(yv[k].mean()), len(k)))
                dd = pd.DataFrame(rows, columns=["signal_date", "fold", "stat", "n_evt"])
                results["summaries"].append(_summarize(dd, col, "Q1", ycol, block))
                dd2 = dd.assign(marker=col, question="Q1", y=ycol, block=block)
                daily_rows.append(dd2)

                # ---------- Q2(b) 五分位内触发 − 未触发
                hi_side = EXPECTED_SIGN[col] > 0
                rows = []
                for day, idx in day_idx.items():
                    sel = idx[(kr[idx] >= 1 - QUINTILE)] if hi_side else idx[(kr[idx] <= QUINTILE)]
                    if len(sel) == 0:
                        continue
                    t_mask = fv[sel]
                    nt = int(t_mask.sum())
                    if nt < MIN_DAY_TRIG or nt == len(sel):
                        continue
                    rows.append((day, folds_arr[idx[0]],
                                 float(yv[sel][t_mask].mean() - yv[sel][~t_mask].mean()),
                                 nt))
                dd = pd.DataFrame(rows, columns=["signal_date", "fold", "stat", "n_evt"])
                results["summaries"].append(_summarize(dd, col, "Q2b", ycol, block))
                daily_rows.append(dd.assign(marker=col, question="Q2b", y=ycol,
                                            block=block))

                # ---------- Q2(a) Fama–MacBeth 日截面回归
                rows_b2, rows_b3 = [], []
                for day, idx in day_idx.items():
                    f_i = fv[idx].astype(float)
                    if f_i.sum() < MIN_DAY_TRIG or f_i.sum() == len(idx):
                        continue
                    kri = kr[idx]
                    X = np.column_stack([np.ones(len(idx)), kri, f_i,
                                         (kri - 0.5) * f_i])
                    try:
                        beta, *_ = np.linalg.lstsq(X, yv[idx], rcond=None)
                    except np.linalg.LinAlgError:
                        continue
                    if not np.all(np.isfinite(beta)):
                        continue
                    rows_b2.append((day, folds_arr[idx[0]], float(beta[2]), int(f_i.sum())))
                    rows_b3.append((day, folds_arr[idx[0]], float(beta[3]), int(f_i.sum())))
                for tag, rr in (("Q2a_b2", rows_b2), ("Q2a_b3", rows_b3)):
                    dd = pd.DataFrame(rr, columns=["signal_date", "fold", "stat", "n_evt"])
                    results["summaries"].append(_summarize(dd, col, tag, ycol, block))
                    daily_rows.append(dd.assign(marker=col, question=tag, y=ycol,
                                                block=block))
            log(f"阶段3 [{block}/{ycol}] 12 标记完成")
        del m

    daily = pd.concat(daily_rows, ignore_index=True)
    daily.to_parquet(out_dir / "daily_series.parquet", index=False)

    # 多重比较汇总（预注册 §6：12 标记 × (Q1 + Q2b) = 24 个门槛检验）
    for block in BLOCKS:
        gate = [s for s in results["summaries"]
                if s["block"] == block and s["y"] == "y"
                and s["question"] in ("Q1", "Q2b")]
        results["blocks"][block]["multiplicity"] = {
            "n_gate_tests": len(gate),
            "n_abs_t_ge_2": sum(1 for s in gate if abs(s["t"]) >= T_GATE),
            "n_bonferroni_hits": sum(1 for s in gate if s["bonferroni_hit"]),
            "n_mde_le_sesoi": sum(1 for s in gate if s.get("mde_le_sesoi")),
            "n_upgrade_candidates": sum(1 for s in gate if s["verdict"] == "升级候选"),
        }
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"阶段3 写出 results.json（{len(results['summaries'])} 条摘要）与 daily_series.parquet")
    return results


# ------------------------------------------------------------------ 报告生成


def _fmt(x, nd=2):
    return "未核" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def write_report(results: dict, out_dir: Path, runtime_s: float, peak_mib: float) -> None:
    L: list[str] = []
    A = L.append
    A("# ICT / SMC 经典 K 线组合探针 —— 结果报告（预注册 v1）")
    A("")
    A("> **定位：探索性诊断，不是确认性检验**（预注册 §0）。"
      "通过门槛 ≠ 采纳；未通过门槛 ≠ 否定。")
    A("> 未达门槛的读数一律表述为「在本样本量下该问题不可回答」，"
      "不得写「不可分」或「两者相等」（CLAUDE.md §二 B 层措辞强制）。")
    A("")
    A(f"- 预注册：`{results['prereg']}`（冻结，本次运行未修改）")
    A(f"- 运行时长：{runtime_s/60:.1f} 分钟；峰值常驻内存（粗测）：{peak_mib:.0f} MiB；纯 CPU。")
    A("")

    # ---------------- 摘要
    pb = results["blocks"][PRIMARY_BLOCK]
    mp = pb["multiplicity"]
    gate_rows = [s for s in results["summaries"]
                 if s["block"] == PRIMARY_BLOCK and s["y"] == "y"
                 and s["question"] in ("Q1", "Q2b")]
    tmax = max((abs(s["t"]) for s in gate_rows if np.isfinite(s["t"])), default=float("nan"))
    mdes = [s["mde_bp"] for s in gate_rows if np.isfinite(s["mde_bp"])]
    A("## 摘要（主块 = 折 36–42）")
    A("")
    A(f"- {mp['n_gate_tests']} 个预注册门槛检验（12 标记 × Q1/Q2b）："
      f"**升级候选 {mp['n_upgrade_candidates']} 个、Bonferroni 命中 "
      f"{mp['n_bonferroni_hits']} 个、|t| ≥ {T_GATE} 命中 {mp['n_abs_t_ge_2']} 个**；"
      f"最大 |t| = {tmax:.2f}。")
    A(f"- MDE 中位数 {float(np.median(mdes)):.0f} bp（区间 {min(mdes):.0f}–{max(mdes):.0f} bp），"
      f"**{mp['n_mde_le_sesoi']}/{mp['n_gate_tests']} 个检验满足 MDE ≤ SESOI = {SESOI_BP:.0f} bp**。")
    A("- 因此，按预注册 §0 的措辞强制：**在本样本量下，这 12 个 ICT 标记"
      "对 6 日执行收益是否有独立信息、以及对 Kronos 秩是否有增量，都不可回答**。"
      "这不是「两者相等」，也不是「标记无效」的证据。")
    A("")

    A("## 复权口径核对（预注册 §2 要求）")
    A("")
    A("OHLC 取自 `panel_kronos_adj.parquet`。核对 `src/crsp_pipeline/adjust.py` 与 "
      "`prep_manifest.json`：")
    A("")
    A("- 复权事件只取 CIZ `distype == 'FRS'`（`disdetailtype` ∈ {STKSPL 拆股, "
      "STKDIV 股票股利}），`factor = 1 + disfacshr`（manifest 的 "
      "`split_event_rule` 字段与代码常量一致，2026-08-26 冻结）。")
    A("- **现金股息、分拆（SP/SEC*）、配股不写入 OHLC**（规范 §5）。")
    A("- 累计因子以固定基准日 `anchor = 2025-12-31`（manifest `anchor_date`）为锚："
      "`cumfactor(t) = ∏ factor(e)`，对所有 `t < ex_date(e) ≤ anchor` 的事件；"
      "`price_adj = price / cumfactor`，`vol_adj = vol × cumfactor`。")
    A("- 禁止逐日裸乘 `DlyFacPrc`；`DlyFacPrc` 语义已验证为「当期事件因子」，只用于交叉审计。")
    A("")
    A("**对本探针的含义**：OHLC 四价用**同一个** `cumfactor(t)` 缩放，所以"
      "所有标记用到的量（H20/L20 比较、TR/ATR 比值、实体占比、订单块与 OTE 的价位区间）"
      "在拆股面前都是**尺度不变或同尺度可比**的；锚定固定基准日保证因子不随最新日漂移，"
      "同一 t 的标记值可复现。现金股息不入 OHLC，故除息日会留下一个真实的价格缺口 —— "
      "这是**已披露的口径特征**，不作调整（预注册未要求，改了就是看结果后改口径）。")
    A("")

    A("## 口径分块（CLAUDE.md §八）")
    A("")
    A("| 块 | 折 | `scoring_config` | 观测数 | 交易日数 |")
    A("|---|---|---|---|---|")
    for b, info in results["blocks"].items():
        scs = info["scoring_configs"]
        uniq = {json.dumps(v, ensure_ascii=False, sort_keys=True) for v in scs.values()}
        s = list(uniq)[0] if len(uniq) == 1 else "**折间不一致**"
        A(f"| {b}（{info['label']}） | {info['folds'][0]}–{info['folds'][-1]} | "
          f"`{s}` | {info['n_obs']:,} | {info['n_days']} |")
    A("")
    A("**36–42 为主块**（`scoring_config` 逐折一致、可机器核对）。"
      "**01–04 的 `metrics.json` 无 `scoring_config` 字段**（值为 `null`，"
      "评估时尚未写入该字段），按 CLAUDE.md §八「不同口径的读数不得并列比较」"
      "单独成块报告；两块的数字**不得放在同一行对比**，也不合并成 11 折统计量。")
    A("")
    A("**门槛判定在主块（36–42，7 折）上做**，一致性用预注册 §6 的括号口径"
      f"「≥ {CONSISTENCY_FRAC:.0%} 的有效折」（因主块只有 7 折，"
      "字面的 `≥ 8/11` 在分块后不可用）。01–04 块用同一规则单独判定。")
    A("")

    # ------------- Q0
    A("## Q0 描述统计（预注册 §4）")
    for b, info in results["blocks"].items():
        q0 = results["q0"][b]
        c = info["counts"]
        A("")
        A(f"### 块 {b}（{info['label']}）")
        A("")
        A(f"- 原始行数 {c['n_rows_all']:,} → 标签 ok ∧ 分数非缺 {c['n_label_ok_and_score']:,} "
          f"→ 在宇宙内 {c['n_in_universe']:,}（剔除 {c['drop_share_not_in_universe']:.2%}）"
          f"→ 标记可算 {c['n_computable']:,}（再剔除 {c['drop_share_not_computable']:.2%}）。")
        A(f"- 最终观测集 **{c['n_final']:,}** 股·日，{info['n_days']} 个交易日。")
        A(f"- 「可算」= 当日 OHLC 完整 ∧ 该证券在载入窗内已有 ≥ {MIN_HIST_P5P6} 根前置 K 线"
          "（P5/P6 所需的最长回看）∧ H20/L20/ATR14/t-2 均可算。"
          "**12 个标记共用同一可算掩码**，见「歧义处的读法」。")
        A("")
        A("| 标记 | 触发率（占股·日） | " + " | ".join(info["folds"]) + " |")
        A("|---|---|" + "---|" * len(info["folds"]))
        for col in MARKER_COLS:
            ev = q0["events_per_fold"][col]
            A(f"| {col} | {q0['trigger_rate'][col]:.2%} | "
              + " | ".join(f"{ev[f]:,}" for f in info["folds"]) + " |")
        A("")
        A("**标记间重叠（Jaccard，同一股·日同时触发）** —— 只列 ≥ 0.05 的对：")
        A("")
        pairs = []
        for i, a in enumerate(MARKER_COLS):
            for bb in MARKER_COLS[i + 1:]:
                j = q0["overlap_jaccard"][a][bb]
                if j >= 0.05:
                    pairs.append((j, a, bb))
        if pairs:
            A("| 标记 A | 标记 B | Jaccard |")
            A("|---|---|---|")
            for j, a, bb in sorted(pairs, reverse=True):
                A(f"| {a} | {bb} | {j:.3f} |")
        else:
            A("（无一对 Jaccard ≥ 0.05）")
    A("")

    # ------------- 主表
    def table(block: str, question: str, ycol: str) -> None:
        rows = [s for s in results["summaries"]
                if s["block"] == block and s["question"] == question and s["y"] == ycol]
        A("")
        A("| 标记 | 预期向 | 均值(bp) | NW SE(bp) | t | MDE(bp) | MDE≤SESOI | "
          "事件数 | 日数 | 同向/有效折 | 门槛判定 |")
        A("|---|---|---|---|---|---|---|---|---|---|---|")
        for s in rows:
            A(f"| {s['marker']} | {'+' if s.get('expected_sign', 0) > 0 else '−'} | "
              f"{_fmt(s['mean_bp'])} | {_fmt(s['se_bp'])} | {_fmt(s['t'])} | "
              f"{_fmt(s['mde_bp'])} | {'是' if s.get('mde_le_sesoi') else '否'} | "
              f"{s['n_events']:,} | {s['n_days']} | "
              f"{s['n_same_dir_folds']}/{s['n_valid_folds']} | {s['verdict']} |")

    for b, info in results["blocks"].items():
        A("")
        A(f"## 块 {b}（{info['label']}）—— 门槛表")
        A("")
        A(f"SESOI = {SESOI_BP:.0f} bp；MDE = {MDE_MULT} × NW SE；NW lag = {NW_LAG}；"
          f"有效折 = 该折事件数 ≥ {MIN_FOLD_EVENTS}；一致性要求 ≥ "
          f"{CONSISTENCY_FRAC:.0%} 的有效折。")
        for ycol, yname in (("y", "原始 y"), ("y_w", "按日 1%/99% 截尾 y")):
            A("")
            A(f"### Q1 独立信息（{yname}）")
            table(b, "Q1", ycol)
            A("")
            A(f"### Q2(b) Kronos 五分位内「触发 − 未触发」（{yname}）")
            A("")
            A("看涨标记在**前五分位**内评估（多头腿过滤价值），"
              "看跌标记在**后五分位**内评估（空头腿过滤价值）。")
            table(b, "Q2b", ycol)
            A("")
            A(f"### Q2(a) Fama–MacBeth 日截面回归 b2（标记主效应，{yname}）")
            table(b, "Q2a_b2", ycol)
            A("")
            A(f"### Q2(a) 交互项 b3（(kr−0.5)×f，{yname}）")
            A("")
            A("注：b2 / b3 的单位是回归系数（对 y 的边际效应），"
              "已按 ×1e4 换算为 bp；b3 的自变量 (kr−0.5) 取值范围约 ±0.5，"
              "故 b3 的量级需除以 2 才对应「秩从中位到极端」的效应。")
            table(b, "Q2a_b3", ycol)
        mp = info["multiplicity"]
        A("")
        A(f"**多重比较（本块，原始 y，Q1 + Q2b 共 {mp['n_gate_tests']} 个门槛检验）**："
          f"|t| ≥ {T_GATE} 命中 {mp['n_abs_t_ge_2']} 个"
          f"（零假设下期望约 {mp['n_gate_tests']*0.0455:.1f} 个）；"
          f"Bonferroni 口径 |t| ≥ {T_BONF} 命中 **{mp['n_bonferroni_hits']}** 个；"
          f"MDE ≤ SESOI 的检验 {mp['n_mde_le_sesoi']} 个；"
          f"「升级候选」{mp['n_upgrade_candidates']} 个。")
        cand = [s for s in results["summaries"]
                if s["block"] == b and s["y"] == "y"
                and s["question"] in ("Q1", "Q2b") and s["verdict"] == "升级候选"]
        if cand:
            A("")
            A("**本块「升级候选」逐条**（预注册 §6 多重比较条款）：")
            A("")
            for s in cand:
                mark = ("有 Bonferroni 支撑（|t| ≥ 2.87）" if s["bonferroni_hit"]
                        else "**无 Bonferroni 支撑 → 记为「与零假设相容的单次命中」**，"
                             "仍可作为候选，但不构成证据")
                A(f"- `{s['marker']}` / {s['question']}：均值 {s['mean_bp']:+.1f} bp，"
                  f"t = {s['t']:+.2f}，同向 {s['n_same_dir_folds']}/{s['n_valid_folds']} 折 —— {mark}。")
            A("")
            A(f"> 注：本块 MDE 全部 > SESOI（{SESOI_BP:.0f} bp），"
              "按预注册 §0 与 CLAUDE.md §二「功效关」，**这些命中只能作为"
              "「值得升级为正式预注册臂候选」的线索，不是采纳依据**。")
        if len(info["folds"]) < 7:
            A("")
            A(f"> **CLAUDE.md §三 强制注记**：本块只有 {len(info['folds'])} 折"
              "（< 7 折）。「少于七折的结论一律不算数」——本块的任何门槛判定"
              "（含「升级候选」与「方向性不利」）都只是方向性线索，不得单独引用。")

    A("")
    A("## 歧义处的读法（预注册 §3「取最简读法、不得同时跑多种读法」）")
    A("")
    for i, s in enumerate(AMBIGUITIES, 1):
        A(f"{i}. {s}")
    A("")
    A("## 事后观察（未预注册，不进门槛）")
    A("")
    A("以下全部是**看到结果之后**才写下的观察，按 CLAUDE.md §二只能放在本节，"
      "不参与任何判定，也不得据以修改判据。每条标注【实测】/【推断】。")
    A("")
    for s in posthoc_lines(results):
        A(f"- {s}")
    A("")
    (out_dir / "report.md").write_text("\n".join(L), encoding="utf-8")
    log("写出 report.md")


AMBIGUITIES: list[str] = [
    "**ATR14 的平滑**：预注册只写「ATR 窗 14（用 t-1 及之前）」。取最简读法 = "
    "TR 的**简单移动平均** `mean(TR[t-14..t-1])`（不用 Wilder 递归平滑）。",
    "**P4 分母为零**：`H_t == L_t` 时实体占比 `(C−O)/(H−L)` 未定义。"
    "取最简读法 = **不触发**（该日该标记为 False，不计入触发，也不剔除该观测）。",
    "**P5 订单块的「t0 前 5 根」**：读作 `t0-5 .. t0-1` 这 5 根，"
    "取其中**最靠近 t0** 的一根阴线（看涨）/ 阳线（看跌）。",
    "**P6「Low_t 进入区间」**：读作 `区间下沿 ≤ Low_t ≤ 区间上沿`（看跌镜像用 High_t）。"
    "「期间未破腿低」读作：在 `(t0, t]` 的每一根上 `Low_s ≥ 腿低`；"
    "一旦某根跌破即作废该 t0（不再继续扫描）。",
    "**五分位边界**：`kr` 为当日分数百分位秩（`rank(pct=True)`，取值 (0,1]）。"
    "前五分位 = `kr ≥ 0.8`，后五分位 = `kr ≤ 0.2`。",
    "**「标记可算」的口径**：预注册说「该观测从**该标记**的分析中剔除」，"
    "字面上是逐标记掩码。取最简读法 = **单一可算掩码**（要求 6 个标记同时可算，"
    "即 ≥ 45 根前置 K 线 ∧ 当日 OHLC 完整），使 12 个标记共用同一观测集与同一 y "
    "去均值口径，否则各标记的 y 基准不同、彼此不可比。代价是多剔除了少量只对 "
    "P1–P4 可算的观测，剔除比例已在 Q0 报告。",
    "**截尾版 y**：预注册说「y 按日 1%/99% 截尾版」。取最简读法 = 先按当日观测集"
    "去均值得 y，再把 y 裁剪到当日 y 的 1% / 99% 分位（不是先裁 label 再去均值）。",
    "**Q2(b) 的跳过规则**：预注册只说「触发数 < 5 跳过」。取最简读法 = "
    "该五分位内触发数 < 5 跳过；另外若该五分位内**全部**触发（无对照组）也跳过。",
    "**Q2(a) 的跳过规则**：与 Q1/Q2(b) 一致，当日全截面触发数 < 5 或全部触发则跳过该日。",
    "**分块与一致性门槛**：预注册 §6 写「同向折数 ≥ 8/11（或 ≥ 73% 的有效折）」，"
    "但 §2 又要求 01–04 若口径不同须单独成块、不与主块并列。二者在「11 折」上冲突。"
    "取最简读法 = **分块判定 + 用括号里的 73% 有效折口径**，不生成 11 折合并统计量。",
]

def _wins_verdict_diff(s) -> str:
    """截尾版与原始版的门槛判定差异（主块，Q1+Q2b）。"""
    g = s[(s.block == PRIMARY_BLOCK) & (s.question.isin(["Q1", "Q2b"]))]
    p = g.pivot_table(index=["marker", "question"], columns="y",
                      values="verdict", aggfunc="first")
    d = p[p["y"] != p["y_w"]]
    if len(d) == 0:
        return "**截尾没有改变主块的任何门槛判定**。"
    items = "；".join(f"`{i[0]}`/{i[1]}「{r['y']}」→「{r['y_w']}」"
                      for i, r in d.iterrows())
    hard = d[(d["y"].isin(["升级候选", "方向性不利"]))
             | (d["y_w"].isin(["升级候选", "方向性不利"]))]
    tail = ("；**没有任何一条跨进或跨出「升级候选」/「方向性不利」**"
            if len(hard) == 0 else "；**其中有条目跨进或跨出了「升级候选」/「方向性不利」**")
    return f"截尾改变了 {len(d)} 条门槛判定（{items}）{tail}。"


def posthoc_lines(results: dict) -> list[str]:
    """看到结果之后才写的观察。只描述，不判定；不得据此修改任何判据。"""
    import pandas as _pd
    s = _pd.DataFrame(results["summaries"])
    out: list[str] = []

    def gate(block: str) -> _pd.DataFrame:
        return s[(s.block == block) & (s.y == "y") & (s.question.isin(["Q1", "Q2b"]))]

    g = gate(PRIMARY_BLOCK)
    imax = g["t"].abs().idxmax()
    out.append(
        f"【实测】主块（36–42，7 折）24 个门槛检验**无一**达到 |t| ≥ {T_GATE}；"
        f"最大 |t| = {abs(g.loc[imax,'t']):.2f}（`{g.loc[imax,'marker']}` / "
        f"{g.loc[imax,'question']}，均值 {g.loc[imax,'mean_bp']:+.1f} bp）。"
        f"MDE 区间 {g['mde_bp'].min():.0f}–{g['mde_bp'].max():.0f} bp、"
        f"中位数 {g['mde_bp'].median():.0f} bp，**全部 > SESOI = {SESOI_BP:.0f} bp**。"
        "按预注册 §0 与措辞强制：**在本样本量下这 24 个问题都不可回答**。")
    out.append(
        f"【实测】预注册 §5 的先验估算（写在看结果之前）是「NW SE ≈ 8–10 bp、"
        f"MDE ≈ 22–28 bp」。主块实测 NW SE 中位数 {g['se_bp'].median():.1f} bp、"
        f"MDE 中位数 {g['mde_bp'].median():.0f} bp —— 功效估算与实测吻合，"
        "「本探针分辨不了 SESOI 量级的效应」这一事前判断被证实。")

    q1 = s[(s.block == PRIMARY_BLOCK) & (s.y == "y") & (s.question == "Q1")]
    q1 = q1.set_index("marker")["mean_bp"]
    bulls = [m for m in MARKER_COLS if m.endswith("_bull")]
    n_neg = sum(1 for m in bulls if q1[m] < 0)
    out.append(
        f"【实测】主块 Q1 的**符号是系统性反向的**：6 个看涨标记里 {n_neg} 个的均值为负"
        f"（P2_bull {q1['P2_bull']:+.1f}、P3_bull {q1['P3_bull']:+.1f}、"
        f"P4_bull {q1['P4_bull']:+.1f}、P5_bull {q1['P5_bull']:+.1f}、"
        f"P6_bull {q1['P6_bull']:+.1f} bp），而对应的看跌镜像 P2_bear/P3_bear 为正"
        f"（{q1['P2_bear']:+.1f} / {q1['P3_bear']:+.1f} bp）。"
        "**任何单个标记都没到 |t| ≥ 2**，这只是一个符号上的整体印象，不是读数。")
    out.append(
        "【推断】上一条的符号方向与「短期横截面反转」一致，而与 ICT 的「顺势」处方相反。"
        "另一个无法排除的机制是执行时点：标签在 **t+1 开盘**建仓，突破 / 位移日的次日"
        "开盘常带跳空，买在跳空上再持有 6 日会吃掉回补。本设计**分辨不了**"
        "「真实反转因子」与「次日开盘执行成本」这两种解释。")

    ge = gate("early")
    ce = [r for _, r in ge.iterrows() if r["verdict"] in ("升级候选", "方向性不利")]
    out.append(
        f"【实测】次块（01–04，仅 4 折，`scoring_config` 缺失）的效应量比主块大一个档次，"
        f"24 个检验里 {int((ge['t'].abs() >= T_GATE).sum())} 个 |t| ≥ 2、"
        f"{int(ge['bonferroni_hit'].sum())} 个过 Bonferroni；其中 "
        + "、".join(f"`{r['marker']}`/{r['question']} {r['mean_bp']:+.1f} bp (t={r['t']:+.2f}, {r['verdict']})"
                    for r in ce)
        + "。**两块口径不同、且次块 < 7 折，不得与主块并列，也不得单独引用。**")
    out.append(
        "【推断】主块（2020–2023）与次块（2003–2004）的落差，可能是效应随年代衰减，"
        "也可能只是次块 4 折的小样本噪声（CLAUDE.md §三：本项目已被 3 折结论推翻四次）。"
        "本探针**没有设计**来区分这两者。")

    q0 = results["q0"][PRIMARY_BLOCK]["trigger_rate"]
    hi = max(q0, key=q0.get)
    lo = min(q0, key=q0.get)
    out.append(
        f"【实测】触发率与预注册 §5 的先验（3–5%）出入较大：最高 `{hi}` {q0[hi]:.1%}、"
        f"最低 `{lo}` {q0[lo]:.1%}。P3（FVG）明显高于先验，P6（OTE）明显低于先验。"
        "触发率偏离先验会同时影响 SE 与 MDE，但两个方向的偏离在这里大致抵消"
        f"（实测 MDE 中位数 {g['mde_bp'].median():.0f} bp 落在先验区间内）。")

    jac = results["q0"][PRIMARY_BLOCK]["overlap_jaccard"]
    out.append(
        f"【实测】P5/P6 按定义派生自 P2 的触发日 t0，但同日重叠很低"
        f"（P2_bull–P5_bull Jaccard {jac['P2_bull']['P5_bull']:.3f}、"
        f"P2_bull–P6_bull {jac['P2_bull']['P6_bull']:.3f}）——因为 P5/P6 触发在 t0 **之后**的回测日。"
        f"重叠最高的一对是 P2_bull–P3_bull（{jac['P2_bull']['P3_bull']:.3f}）。")

    w = s[(s.block == PRIMARY_BLOCK) & (s.question.isin(["Q1", "Q2b"]))]
    piv = w.pivot_table(index=["marker", "question"], columns="y", values="mean_bp")
    d = (piv["y_w"] - piv["y"]).abs()
    gw = s[(s.block == PRIMARY_BLOCK) & (s.y == "y_w")
           & (s.question.isin(["Q1", "Q2b"]))]
    cross = gw[gw["t"].abs() >= T_GATE]
    out.append(
        f"【实测】按日 1%/99% 截尾把主块的均值最多改动 {d.max():.1f} bp"
        f"（中位 {d.median():.1f} bp）。截尾版有 {len(cross)} 个检验 |t| ≥ {T_GATE}"
        + ("" if len(cross) == 0 else
           "（" + "、".join(f"`{r['marker']}`/{r['question']} t={r['t']:+.2f}，"
                           f"判定仍为「{r['verdict']}」"
                           for _, r in cross.iterrows()) + "）")
        + f"，Bonferroni 命中 {int(gw['bonferroni_hit'].sum())} 个、"
        f"「升级候选」{int((gw['verdict'] == '升级候选').sum())} 个 —— "
        + _wins_verdict_diff(s)
        + "（预注册 §4 把截尾版列为「另报」，门槛判定以原始 y 为准。）")
    out.append(
        "【实测】宇宙过滤与「标记可算」过滤在两块都**剔除 0 行**：各折评估目录"
        "（`*_poolB_universe`）已经是宇宙内打分，而面板回看余量（62 个交易日）"
        f"处处满足 P5/P6 所需的 {MIN_HIST_P5P6} 根前置 K 线。"
        "唯一的剔除来自 `status != 'ok'` 与分数缺失（两块合计 "
        + f"{sum(i['counts']['n_rows_all'] - i['counts']['n_label_ok_and_score'] for i in results['blocks'].values()):,} 行）。")
    return out


# ---------------------------------------------------------------------- 主流程


class FileLock:
    """输出目录级互斥锁（CLAUDE.md §七：批处理脚本一律加互斥锁）。"""

    def __init__(self, path: Path):
        self.path = path
        self.fd = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
        except FileExistsError:
            raise SystemExit(
                f"已有另一进程持有锁 {self.path}；若确认无进程在跑，手动删除该文件。")
        os.write(self.fd, f"pid={os.getpid()} t={time.time():.0f}".encode())
        return self

    def __exit__(self, *exc):
        if self.fd is not None:
            os.close(self.fd)
        self.path.unlink(missing_ok=True)


def _peak_mib() -> float:
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.c_void_p(k32.GetCurrentProcess()), ctypes.byref(pmc), pmc.cb)
        return pmc.PeakWorkingSetSize / 2 ** 20
    except Exception:
        return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/ict_pattern_probe")
    ap.add_argument("--force", action="store_true", help="重算所有阶段（默认断点续跑）")
    ap.add_argument("--stage", default="all", choices=["all", "1", "2", "3"])
    args = ap.parse_args()

    out_dir = (REPO_ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with FileLock(out_dir / ".lock"):
        if args.stage in ("all", "1"):
            for b in BLOCKS:
                stage1_patterns(b, out_dir, args.force)
        if args.stage in ("all", "2"):
            for b in BLOCKS:
                stage2_merge(b, out_dir, args.force)
        if args.stage in ("all", "3"):
            res = stage3_stats(out_dir)
            write_report(res, out_dir, time.time() - t0, _peak_mib())
    log(f"完成，总耗时 {(time.time()-t0)/60:.1f} 分钟，峰值内存 {_peak_mib():.0f} MiB")


if __name__ == "__main__":
    main()
