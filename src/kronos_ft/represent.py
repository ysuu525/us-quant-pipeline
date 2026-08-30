"""表示提取（改造 ③ 的基础）：把冻结主干对 lookback 窗的隐层表示抽出来。

为什么要这一层
--------------
现行打分路径是"采样 5 条价格路径 → 取均值 → 压成 predOpen(t+6)/predOpen(t+1)-1
→ 排序"。实测该路径的分数信度仅约 0.75（同模型换 batch 重打两次的相关性），
即四分之一到一半是采样噪声；且生成目标与横截面排序目标脱钩（epoch 探针：
e1 的 RankIC 已接近 e30；E12：零样本 ≈ 微调）。

本模块只做一次确定性前向：tokenizer.encode → transformer → norm，取 anchor 位
的隐层向量（d_model=512）。**无采样、无自回归**，输出完全确定，
下游用轻量排序头把表示映射成分数。

口径一致性
----------
归一化（逐序列 mean/std + clip）、时间戳构造、窗口切片与 infer.run_scoring 的
fast 路径逐行对应，保证"表示"与"生成式分数"看到的是同一份输入。
"""

from __future__ import annotations

import contextlib

import numpy as np
import pandas as pd
import torch

from crsp_pipeline.calendar import TradingCalendar

from . import import_kronos
from .dataset import FEATURE_COLS
from .infer import PRED_COLS, _stamp_matrix
from .models import pick_device


def extract_representations(
    tokenizer,
    model,
    panel: pd.DataFrame,
    scoring_index: pd.DataFrame,
    calendar: TradingCalendar,
    lookback: int,
    batch_size: int = 256,
    device: str | None = None,
    amp: str | None = None,
    max_context: int = 512,
    clip: float = 5.0,
    pool: str = "last",
    permno_col: str = "PERMNO",
    date_col: str = "DlyCalDt",
) -> tuple[pd.DataFrame, np.ndarray]:
    """返回 (索引 DataFrame[PERMNO, signal_date], 表示矩阵 float16 [n, d_model])。

    pool: "last" 取 anchor 位；"mean" 取窗口内均值（两者维度相同）。
    """
    dev = pick_device(device)
    tokenizer = tokenizer.to(dev).eval()
    model = model.to(dev).eval()
    if amp and dev.type == "cuda":
        _dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[amp]

        def _amp_ctx():
            return torch.autocast(device_type="cuda", dtype=_dtype)
    else:
        _amp_ctx = contextlib.nullcontext

    df = panel.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    rename = {v: k for k, v in FEATURE_COLS.items()}
    rename["DlyVol"] = "volume"
    rename["DlyPrcVol"] = "amount"
    by_pn = {
        pn: g.sort_values(date_col).set_index(date_col).rename(columns=rename)
        for pn, g in df.groupby(permno_col)
    }
    feats = {pn: sec[PRED_COLS].to_numpy(np.float32) for pn, sec in by_pn.items()}
    pos = {pn: {d: i for i, d in enumerate(sec.index)} for pn, sec in by_pn.items()}
    stamp_cache: dict[pd.Timestamp, np.ndarray] = {}

    idx = scoring_index.reset_index(drop=True)
    metas: list[tuple] = []
    embs: list[np.ndarray] = []

    for lo in range(0, len(idx), batch_size):
        chunk = idx.iloc[lo:lo + batch_size]
        xs, xst, meta = [], [], []
        for pn, anchor, start in chunk[[permno_col, "anchor", "start"]].itertuples(index=False):
            anchor, start = pd.Timestamp(anchor), pd.Timestamp(start)
            sec_idx = by_pn[pn].index
            i0, i1 = pos[pn][start], pos[pn][anchor]
            x = feats[pn][i0:i1 + 1]
            if not np.isfinite(x).all():
                raise ValueError(f"窗含非有限值: PERMNO={pn} anchor={anchor.date()}")
            hit = stamp_cache.get(anchor)
            if hit is None:
                hit = _stamp_matrix(sec_idx[i0:i1 + 1])
                stamp_cache[anchor] = hit
            mean, std = np.mean(x, axis=0), np.std(x, axis=0)
            xn = np.clip((x - mean) / (std + 1e-5), -clip, clip)
            xs.append(xn.astype(np.float32))
            xst.append(hit)
            meta.append((pn, anchor))

        xt = torch.from_numpy(np.stack(xs, 0)).to(dev)
        st = torch.from_numpy(np.stack(xst, 0).astype(np.float32)).to(dev)
        if xt.size(1) > max_context:
            xt, st = xt[:, -max_context:], st[:, -max_context:]
        with torch.no_grad(), _amp_ctx():
            s1, s2 = tokenizer.encode(xt, half=True)
            _, ctx = model.decode_s1(s1, s2, st)
            h = ctx[:, -1, :] if pool == "last" else ctx.mean(dim=1)
        embs.append(h.float().cpu().numpy().astype(np.float16))
        metas.extend(meta)

    index = pd.DataFrame(metas, columns=[permno_col, "signal_date"])
    return index, np.concatenate(embs, 0) if embs else np.zeros((0, model.d_model), np.float16)
