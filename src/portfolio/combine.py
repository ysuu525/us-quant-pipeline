"""多信号合成规则——**零自由参数，先验固定**。

背景：`docs/思路整理_2026-09-03.md` §6.1 与 `docs/调研_合成与验证设计_2026-09-03.md`
§四。两份调研在合成规则上有分歧（调研 B 推「慢信号筛子」，调研 C 的换手削减
估计基于「50/50 秩相加」），因此**两条规则都实现，由先验拍板选一条，不在
数据上比**。CLAUDE.md §二.B 层：变体 A vs 变体 B 在本样本量下不可回答，
只能按写死的优先序选，选择记为**披露的设计假设**，不是发现。

本模块出现的所有常数（`keep_frac=0.50`、`slow_filter` 的 `stale_days=21`、
`rank_sum_equal_weight` 的 `stale_days=31`）都是
**预注册值，不是可调参数**：

- `keep_frac=0.50`——「一半」是无参考点默认（1/N 家族），调研 B §四 明列为
  「唯一常数 50%」；
- `slow_filter(stale_days=21)`——一个日历月的**交易日**数，慢信号按月频更新时
  「上一期还没到就算陈旧」的自然边界；
- `rank_sum_equal_weight(stale_days=31)`——**日历日**，`experiments/
  signal2_prereg_v2.md` §4a.1 写死的缺失/陈旧门（与信号#2 规格里的
  `MAX_SPAN_DAYS=31` 同一个日历先验）。**这两个 `stale_days` 单位不同、
  出处不同，不可互换**；
- 秩相加的权重 0.5/0.5——上游默认 1/N。调研 B 实测：ρ=0.2 时等权相对最优
  权重只损失 0.03–3.3%，而本项目「看数据估权」的偏差是 +0.0070，比等权的
  损失大一个量级。

**这几个数不得在任何评估折上调**（开发折 36–42 也不行——它们已被消耗，
在上面调参会把已经 20–28% 的选择偏差账继续放大）。若将来必须改，走预注册
修订流程并在 `experiments/ledger.md` 记「修订发生在看到结果之后」。

数据契约
--------
所有函数收发同一张长表：`DataFrame[signal_date, PERMNO, score]`。
`signal_date` 会被 `pd.to_datetime` 规范化；同一 (signal_date, PERMNO) 的
重复行只保留第一条（保留顺序 = 输入行序）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["rank_sum_equal_weight", "slow_filter", "spearman_by_day"]

COLS = ["signal_date", "PERMNO", "score"]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """取三列、规范日期、丢重复 (日, 名字)。不改任何数值。"""
    out = df[COLS].dropna()
    out = out.assign(signal_date=pd.to_datetime(out["signal_date"]))
    return out.drop_duplicates(subset=["signal_date", "PERMNO"], keep="first")


def _day_series(g: pd.DataFrame) -> pd.Series:
    return pd.Series(g["score"].to_numpy(), index=pd.Index(g["PERMNO"].to_numpy(), name="PERMNO"))


def rank_sum_equal_weight(
    a: pd.DataFrame,
    b: pd.DataFrame,
    *,
    stale_days: int = 31,
) -> pd.DataFrame:
    """逐日秩相加合成（等权）+ 缺失/陈旧回退单臂分。**零自由参数**。

    口径来源：`experiments/signal2_prereg_v2.md` §4a.1 / §4a.2（已裁定，
    不由数据选）。**本函数是不对称的**：`a` 是**基准臂 / Kronos**（定义当日
    名字全集，也是回退时用的那一臂），`b` 是**信号#2**（只在其有效子集上排秩）。

    规则（先验固定，不在评估折上调）
    --------------------------------
    记 `N_t` = 当日 `a` 有分数的名字全集（现状口径：top500 且有 Kronos 分数），
    `V_t ⊆ N_t` = 其中信号#2 **有效**的名字。信号#2 在名字 i 于日 t 上有效 ⟺
    存在日期 `s <= t` 使 `b` 在 (s, i) 上有非 NaN 分数，且 `(t − s)` 不超过
    `stale_days` 个**日历日**（取最近的那个 `s`，其分数前推到 t 使用）。

    1. `pct_rank_t(a)` 在 **`N_t`** 上算（全体名字同一把尺）；
       `pct_rank_t(b)` 在 **`V_t`** 上算。两者都用 `rank(pct=True)`
       （默认 `method='average'`，并列取平均秩），取值落在 (0, 1]。
    2. `i ∈ V_t`：`score = 0.5 * (rk_a^{N_t}[i] + rk_b^{V_t}[i])`；
       `i ∈ N_t \\ V_t`：`score = rk_a^{N_t}[i]`（**回退单臂分**）。
       后者等价于「信号#2 恰好与 `a` 的百分位秩一致」的中性处理：缺失的名字
       既不被驱逐出可持有集，也不因缺失获得排名优势（v2 §4a.2 性质 2）。
    3. 权重 0.5/0.5 是上游默认 1/N，**是预注册常数，不是拟合出来的**。

    `stale_days=31`（**日历日**）是 `experiments/signal2_prereg_v2.md` §4a.1 的
    **预注册常数，不得调**（v2 §5 不做清单：「不调 …/ 陈旧门 31 日」）。
    注意与 `slow_filter` 的 `stale_days=21` **不是同一个量**：那个是**交易日**，
    出处是调研 B §四，两者不可互换。做成关键字参数只为让测试能打边界，
    **不是给调用方调的旋钮**。

    退化与兼容
    ----------
    - `V_t = N_t`（无缺失、无陈旧）时，本式**逐位退化**为旧语义
      「先取交集再排秩、`0.5*(ra+rb)`」——旧实现是本实现的特例
      （v2 §4a.2 性质 1，`tests/test_portfolio.py::test_rank_sum_degenerates_...`）。
    - 签名向后兼容：`a`/`b` 仍为前两个位置参数，`stale_days` 是新增关键字参数。
    - **不再对称**：`b` 只在 `V_t` 上排秩、`a` 定义 `N_t`，所以
      `f(a, b) != f(b, a)`。旧 docstring 承诺的对称性随 v2 §4a 作废。
    - **只在一侧出现的日子**：`a` 有而 `b` 无（或 `b` 全部陈旧）的日子**保留**，
      整天回退成 `a` 的秩（旧语义是整天丢弃）；`b` 有而 `a` 无的日子仍丢弃
      （`N_t` 由 `a` 定义，没有 `a` 就没有名字集）。

    返回
    ----
    `DataFrame[signal_date, PERMNO, score, used_signal2]`，按
    (signal_date, PERMNO) 排序。`used_signal2` 为 bool：该行是否真的用上了
    信号#2（即 `i ∈ V_t`）。下游 `scores_frame_to_by_day` 只取前三列，多出来
    的诊断列不影响构造。

    注意：合成分数落在 [0, 1]，与单臂分数不同量纲。下游只用它的**名次**
    （`frozen_long_only_returns` 只看 `rank`），所以量纲无关；但不要拿它和
    单臂分数做数值比较。

    调研 B 的止损点（`docs/调研_合成与验证设计_2026-09-03.md` §四）：秩相加
    要不稀释，需 IC₂ > IC₁ × (√(2(1+ρ)) − 1) ≈ 0.0085（IC₁=0.0155, ρ=0.2），
    且必须是 **6 日标签上的** IC。本函数不做这个检查——v2 §4.2 已把量级门
    从准入规则里删除，0.0085 降为披露中的参考数字。
    """
    a, b = _normalize(a), _normalize(b)
    out_cols = COLS + ["used_signal2"]
    if a.empty:
        return a.assign(used_signal2=pd.Series(dtype=bool))[out_cols].reset_index(drop=True)

    left = a.sort_values("signal_date", kind="stable").reset_index(drop=True)
    if b.empty:
        left["_s2"] = pd.Series(np.nan, index=left.index, dtype=float)
    else:
        right = (b.rename(columns={"score": "_s2"})
                  .sort_values("signal_date", kind="stable")
                  .reset_index(drop=True))
        # backward + tolerance = 「最近一次有效值距 signal_date 不超过
        # stale_days 个日历日」；pandas 的 tolerance 是闭区间（<=）。
        left = pd.merge_asof(
            left, right, on="signal_date", by="PERMNO",
            direction="backward", tolerance=pd.Timedelta(days=int(stale_days)),
        )

    used = left["_s2"].notna().to_numpy()
    rk_a = left.groupby("signal_date")["score"].rank(pct=True).to_numpy()
    rk_b = np.full(len(left), np.nan)
    if used.any():
        sub = left.loc[used]
        rk_b[used] = sub.groupby("signal_date")["_s2"].rank(pct=True).to_numpy()
    # 0.5*(ra+rb) 与旧实现逐字相同的浮点表达式；回退行直接取 rk_a。
    score = np.where(used, 0.5 * (rk_a + rk_b), rk_a)

    out = pd.DataFrame({
        "signal_date": left["signal_date"].to_numpy(),
        "PERMNO": left["PERMNO"].to_numpy(),
        "score": score,
        "used_signal2": used,
    })
    return out.sort_values(["signal_date", "PERMNO"], kind="stable").reset_index(drop=True)


def slow_filter(
    fast: pd.DataFrame,
    slow: pd.DataFrame,
    *,
    keep_frac: float = 0.50,
    stale_days: int = 21,
    hold_count: int | None = None,
) -> pd.DataFrame:
    """慢信号做可持有集，快信号在集内出分。**零自由参数**。

    调研 B §四 的推荐方案：
    > 慢信号#2 的截面秩前 50% 构成可持有集；在可持有集内按信号#1 执行现有
    > 冻结规则。信号#2 缺失或陈旧超过 21 个交易日的名字视为不在可持有集。

    实现
    ----
    对每个 fast 的 `signal_date` d：
    1. 取 slow 中 `signal_date <= d` 且最近的一期 `sd`；不存在则该日无可持有
       集（整天丢弃）。
    2. 若 `sd` 与 `d` 相隔 **> stale_days 个交易日**则整期陈旧，该日无可持有集。
       交易日距离按 fast 与 slow 全部 `signal_date` 的排序并集数格子（fast 是
       日频信号，其日期集就是交易日历）。
    3. 可持有集 = 该期 slow 分数 `rank(pct=True) >= 1 - keep_frac` 的名字。
       用 `>=` 而非 `>`：并列在边界上的名字保留（与
       `construction.py` 的退出判据 `pct[p] >= 1 - EXIT_PCT` 同向，宁松勿紧）。
    4. 返回 fast 的行，限制在可持有集内，`score` 原封不动。

    未决问题（`docs/思路整理_2026-09-03.md` §6.1，本函数**不解决**）
    ----------------------------------------------------------------
    可持有集减半后，下游 `frozen_long_only_returns` 仍按 `k = n // 10` 取名次
    前 10%，而 `n` 已经减半 → **持仓从约 50 只掉到约 25 只，特质波动翻倍**。
    另一条路是把快信号放宽到前 20% 以保住 50 只，但十分位曲线第 9 档只有第 10
    档的 43%，放宽会损失锐度。§6.1 把这条列为「必须在看数据之前写死」的处置。

    本函数**只做筛选，不改构造**。`hold_count` 只是把接口留出来：给定时，
    返回的 DataFrame 会带 `k_override` 列与 `.attrs["k_override"]`，
    **构造层默认不读它**（`frozen_long_only_returns` 只认 `k = max(1, n // 10)`）。
    要真的固定持仓数，必须显式在构造层加一条读 `k_override` 的分支，并当作
    口径变更走预注册。

    常数声明：`keep_frac=0.50`、`stale_days=21` 是预注册值，不得在评估折上调。
    """
    fast, slow = _normalize(fast), _normalize(slow)
    if fast.empty or slow.empty:
        out = fast.iloc[:0].copy()
        return _attach_k(out, hold_count)

    fast_days = pd.DatetimeIndex(sorted(fast["signal_date"].unique()))
    slow_days = pd.DatetimeIndex(sorted(slow["signal_date"].unique()))
    calendar = pd.DatetimeIndex(sorted(set(fast_days) | set(slow_days)))
    cal_pos = {d: i for i, d in enumerate(calendar)}

    holdable: dict[pd.Timestamp, set] = {}
    for sd, g in slow.groupby("signal_date"):
        s = _day_series(g)
        r = s.rank(pct=True)
        holdable[sd] = set(r.index[r.to_numpy() >= 1.0 - keep_frac])

    # 每个 fast 日对应的 slow 期：slow_days 中 <= d 的最后一个。
    pos = slow_days.searchsorted(fast_days, side="right") - 1
    keep_sets: dict[pd.Timestamp, set] = {}
    for d, p in zip(fast_days, pos):
        if p < 0:
            continue
        sd = slow_days[p]
        if cal_pos[d] - cal_pos[sd] > stale_days:
            continue
        keep_sets[d] = holdable[sd]

    mask = np.fromiter(
        (p in keep_sets.get(d, ()) for d, p in zip(fast["signal_date"], fast["PERMNO"])),
        dtype=bool, count=len(fast))
    return _attach_k(fast.loc[mask].reset_index(drop=True), hold_count)


def _attach_k(df: pd.DataFrame, hold_count: int | None) -> pd.DataFrame:
    """把 `k_override` 挂上去（列 + attrs）。构造层默认不读。"""
    if hold_count is None:
        return df
    df = df.copy()
    df["k_override"] = int(hold_count)
    df.attrs["k_override"] = int(hold_count)
    return df


def spearman_by_day(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    """逐日同名字集 Spearman 秩相关，index = signal_date。

    用途：`docs/思路整理_2026-09-03.md` §6.1 的处置表要读「信号#2 与 Kronos 的
    秩相关 ρ」，ρ ≥ 0.3 直接终止。这是**信号自身的诊断**，不是合成规则的
    选择依据。

    约定：只在两侧当日都有分数的名字上算；共同名字 < 3 或任一侧零方差 → NaN
    （与 `crsp_pipeline.signal_eval._spearman` 同一约定）。
    """
    a, b = _normalize(a), _normalize(b)
    bmap = {day: g for day, g in b.groupby("signal_date")}
    idx, vals = [], []
    for day, ga in a.groupby("signal_date"):
        gb = bmap.get(day)
        if gb is None:
            continue
        sa, sb = _day_series(ga), _day_series(gb)
        common = sorted(set(sa.index) & set(sb.index))
        idx.append(day)
        if len(common) < 3:
            vals.append(np.nan)
            continue
        ra = sa.loc[common].rank().to_numpy(dtype=float)
        rb = sb.loc[common].rank().to_numpy(dtype=float)
        if ra.std() == 0 or rb.std() == 0:
            vals.append(np.nan)
            continue
        vals.append(float(np.corrcoef(ra, rb)[0, 1]))
    return pd.Series(vals, index=pd.DatetimeIndex(idx, name="signal_date"), dtype=float)
