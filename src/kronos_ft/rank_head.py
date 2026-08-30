"""横截面排序头（改造 ① + ②）。

改造内容与依据
--------------
① **标签改截面去均值**：一只股票 6 日收益里占方差大头的是市场共同成分，
   而它在横截面排序里恰好完全抵消、贡献为零。生成式训练却会大力奖励模型
   预测这部分。把标签在当日截面内去均值（或转成截面秩），等于把这个"定义上
   无用但占比最大"的成分从目标里删掉。依据：Blitz 等残差动量（残差收益的
   风险调整收益约为总收益的两倍）、spurious predictability 的 beta confound。

② **损失改直接优化 RankIC**：现行训练目标是逐 token 生成损失，与横截面排序
   没有单调关系（实测：内层生成 loss 单调下降 30 轮，RankIC 在 e1 已见顶）。
   本模块直接把"当日截面内标准化的预测 × 标准化的标签秩"的均值作为目标，
   它就是 Pearson-on-ranks，即 RankIC 的可微形式。

③ 头是确定性的：无采样 → 消除实测约 0.25 的采样噪声（信度 0.75）。

早停指标同样改为内层验证 RankIC（此前用生成 loss，与目标脱钩）。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class RankHead(nn.Module):
    """冻结主干之上的小头：LayerNorm → MLP → 标量分数。"""

    def __init__(self, d_in: int = 512, d_hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _standardize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (v - v.mean()) / (v.std() + eps)


def ic_loss(pred: torch.Tensor, target_rank: torch.Tensor) -> torch.Tensor:
    """单日截面的负 IC。target_rank 已是当日截面秩（0-1），此处再标准化。

    最大化 corr(pred, rank(label)) 等价于最小化 -mean(z(pred) * z(rank))。
    """
    return -(_standardize(pred) * _standardize(target_rank)).mean()


def topk_weighted_ic_loss(pred: torch.Tensor, target_rank: torch.Tensor,
                          k_frac: float = 0.1, w_top: float = 3.0) -> torch.Tensor:
    """给头部样本加权的 IC 损失。

    动机（调研 2026-08-31）：变现是"只做多 top-K"，而全截面 RankIC 与 top-K
    组合表现并不同向（同一实验里 RankNet 拿最高 IC 却不是最高夏普）。给真实
    标签排在前 k_frac 的样本更高权重，让损失更贴近变现方式。
    """
    zp, zt = _standardize(pred), _standardize(target_rank)
    w = torch.ones_like(zt)
    w[target_rank >= 1.0 - k_frac] = w_top
    w = w / w.mean()
    return -(w * zp * zt).mean()


def daily_rank_ic(pred: np.ndarray, label: np.ndarray) -> float:
    if len(pred) < 2:
        return float("nan")
    a = np.argsort(np.argsort(pred)).astype(np.float64)
    b = np.argsort(np.argsort(label)).astype(np.float64)
    return float(np.corrcoef(a, b)[0, 1])


def train_head(
    emb_tr: np.ndarray, day_tr: np.ndarray, y_tr: np.ndarray,
    emb_va: np.ndarray, day_va: np.ndarray, y_va: np.ndarray,
    d_in: int, device: torch.device,
    epochs: int = 60, lr: float = 1e-3, weight_decay: float = 1e-4,
    dropout: float = 0.2, patience: int = 8, seed: int = 0,
    loss_kind: str = "ic", min_names: int = 50, verbose: bool = True,
) -> tuple[RankHead, dict]:
    """按"日"成批训练；早停指标 = 内层验证逐日 RankIC 均值（与目标对齐）。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    head = RankHead(d_in, dropout=dropout).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    def _by_day(emb, day, y):
        order = np.argsort(day, kind="stable")
        emb, day, y = emb[order], day[order], y[order]
        bounds = np.flatnonzero(np.diff(day)) + 1
        groups = np.split(np.arange(len(day)), bounds)
        return emb, y, [g for g in groups if len(g) >= min_names]

    Etr, Ytr, Gtr = _by_day(emb_tr, day_tr, y_tr)
    Eva, Yva, Gva = _by_day(emb_va, day_va, y_va)
    Etr_t = torch.from_numpy(Etr.astype(np.float32)).to(device)
    Ytr_t = torch.from_numpy(Ytr.astype(np.float32)).to(device)
    Eva_t = torch.from_numpy(Eva.astype(np.float32)).to(device)

    fn = ic_loss if loss_kind == "ic" else topk_weighted_ic_loss
    best, best_state, best_ep, hist = -np.inf, None, 0, []
    for ep in range(1, epochs + 1):
        head.train()
        perm = np.random.permutation(len(Gtr))
        tot = 0.0
        for gi in perm:
            g = Gtr[gi]
            idx = torch.from_numpy(g).to(device)
            loss = fn(head(Etr_t[idx]), Ytr_t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach())
        head.eval()
        with torch.no_grad():
            pv = head(Eva_t).cpu().numpy()
        ics = [daily_rank_ic(pv[g], Yva[g]) for g in Gva]
        vic = float(np.nanmean(ics)) if ics else float("nan")
        hist.append({"epoch": ep, "train_loss": tot / max(1, len(Gtr)), "val_rank_ic": vic})
        if verbose:
            print(f"  [head] epoch {ep:>3}: loss {tot/max(1,len(Gtr)):+.4f}  "
                  f"内层 RankIC {vic:+.5f}", flush=True)
        if vic > best:
            best, best_ep = vic, ep
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        elif ep - best_ep >= patience:
            if verbose:
                print(f"  [head] 早停于 epoch {ep}（最优 {best_ep}，内层 RankIC {best:+.5f}）")
            break
    if best_state is not None:
        head.load_state_dict(best_state)
    return head, {"best_epoch": best_ep, "best_val_rank_ic": best, "history": hist}
