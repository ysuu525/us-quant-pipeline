"""单卡两阶段微调（tokenizer → predictor），实现 docs/预注册_v1.md §1。

与官方 finetune/ 的关系：目标函数、优化器超参、梯度裁剪、OneCycle 调度
逐行沿用（train_tokenizer.py / train_predictor.py）；去掉 DDP 与 comet_ml
（单卡 4080 / Mac 冒烟），加入预注册机制：

- 内层验证：训练窗尾部 6 个月（purge 6 交易日），早停与选点**只看内层
  生成式 loss**；
- SWA：早停最优 epoch 前最后 K=3 个 checkpoint 权重平均为该折唯一模型；
- seed 显式传入（每配置 3 seeds 由外层调度）；
- 试验登记簿：每次运行追加一行到 experiments/ledger.md。

冒烟模式（--smoke）：合成随机游走面板 + 随机初始化小模型，CPU/MPS 上
两阶段各跑 2 个 epoch，验证代码路径、checkpoint 存读与 SWA——Windows 上
第一次跑不该是这条代码路径的首次执行。
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from crsp_pipeline.calendar import TradingCalendar

from .dataset import KronosWindowDataset
from .models import build_tiny, load_pretrained, pick_device, swa_average
from .windows import build_window_index, filter_anchors, inner_split

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = REPO_ROOT / "experiments" / "ledger.md"


@dataclass
class TrainConfig:
    lookback: int = 90
    predict: int = 6
    clip: float = 5.0
    batch_size: int = 32
    max_epochs: int = 30
    patience: int = 5          # 内层 loss 连续 patience 个 epoch 未创新低 → 停
    swa_k: int = 3             # 预注册：最优 epoch 前最后 K 个 ckpt 权重平均
    inner_months: int = 6      # 预注册：内层验证窗
    seed: int = 0
    # 官方超参（finetune/config.py / train_*.py 原值）
    tokenizer_lr: float = 2e-4
    predictor_lr: float = 4e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    weight_decay: float = 0.1
    grad_clip_tokenizer: float = 2.0
    grad_clip_predictor: float = 3.0
    onecycle_pct_start: float = 0.03
    onecycle_div_factor: float = 10.0
    num_workers: int = 0       # Windows spawn 安全默认；调优在训练机上做
    device: str | None = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_loader(ds, cfg: TrainConfig, shuffle: bool) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(cfg.seed)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                      num_workers=cfg.num_workers, generator=g if shuffle else None,
                      drop_last=shuffle)


def _stage_loss(stage: str, model, tokenizer, x, stamp):
    """官方目标函数，逐行对应 train_tokenizer.py / train_predictor.py。"""
    if stage == "tokenizer":
        (z_pre, z), bsq_loss, _, _ = model(x)
        recon = F.mse_loss(z_pre, x) + F.mse_loss(z, x)
        return (recon + bsq_loss) / 2
    with torch.no_grad():
        s0, s1 = tokenizer.encode(x, half=True)
    logits = model(s0[:, :-1], s1[:, :-1], stamp[:, :-1, :])
    loss, _, _ = model.head.compute_loss(logits[0], logits[1], s0[:, 1:], s1[:, 1:])
    return loss


def _epoch(stage, model, tokenizer, loader, device, cfg,
           optimizer=None, scheduler=None) -> float:
    training = optimizer is not None
    model.train() if training else model.eval()
    losses = []
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for x, stamp in loader:
            x, stamp = x.to(device), stamp.to(device)
            loss = _stage_loss(stage, model, tokenizer, x, stamp)
            if training:
                optimizer.zero_grad()
                loss.backward()
                clip = (cfg.grad_clip_tokenizer if stage == "tokenizer"
                        else cfg.grad_clip_predictor)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
                optimizer.step()
                scheduler.step()
            losses.append(float(loss.detach()))
    return float(np.mean(losses)) if losses else float("nan")


def train_stage(stage: str, model, tokenizer, train_ds, inner_ds,
                cfg: TrainConfig, out_dir: Path) -> dict:
    """一个阶段的完整训练：早停（内层 loss）→ SWA → save_pretrained。

    返回 {best_epoch, stopped_epoch, best_inner_loss, swa_inner_loss, history}。
    """
    assert stage in ("tokenizer", "predictor")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(cfg.device)
    model.to(device)
    if tokenizer is not None:
        tokenizer.to(device).eval()

    train_loader = _make_loader(train_ds, cfg, shuffle=True)
    inner_loader = _make_loader(inner_ds, cfg, shuffle=False)
    if len(train_loader) == 0:
        raise ValueError(f"[{stage}] empty train loader (batch>{len(train_ds)} samples?)")

    lr = cfg.tokenizer_lr if stage == "tokenizer" else cfg.predictor_lr
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr,
        betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(train_loader),
        epochs=cfg.max_epochs, pct_start=cfg.onecycle_pct_start,
        div_factor=cfg.onecycle_div_factor,
    )

    history, ckpts = [], {}
    best_epoch, best_inner = 0, float("inf")
    for epoch in range(1, cfg.max_epochs + 1):
        train_loss = _epoch(stage, model, tokenizer, train_loader, device, cfg,
                            optimizer, scheduler)
        inner_loss = _epoch(stage, model, tokenizer, inner_loader, device, cfg)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "inner_loss": inner_loss})
        ckpt_path = out_dir / f"{stage}_epoch{epoch:03d}.pt"
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, ckpt_path)
        ckpts[epoch] = ckpt_path
        print(f"[{stage}] epoch {epoch}: train {train_loss:.4f}  inner {inner_loss:.4f}")

        if inner_loss < best_inner:
            best_inner, best_epoch = inner_loss, epoch
        elif epoch - best_epoch >= cfg.patience:
            print(f"[{stage}] early stop at epoch {epoch} (best {best_epoch})")
            break

    # SWA：最优 epoch 前最后 K 个 ckpt（含最优）等权平均
    lo = max(1, best_epoch - cfg.swa_k + 1)
    swa_epochs = list(range(lo, best_epoch + 1))
    sds = [torch.load(ckpts[e], weights_only=True) for e in swa_epochs]
    model.load_state_dict(swa_average(sds))
    model.to(device)
    swa_inner = _epoch(stage, model, tokenizer, inner_loader, device, cfg)

    final_dir = out_dir / f"{stage}_final"
    model.save_pretrained(str(final_dir))
    summary = {
        "stage": stage, "best_epoch": best_epoch,
        "stopped_epoch": history[-1]["epoch"],
        "best_inner_loss": best_inner, "swa_epochs": swa_epochs,
        "swa_inner_loss": swa_inner, "history": history,
        "final_dir": str(final_dir),
    }
    (out_dir / f"{stage}_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run(panel: pd.DataFrame, calendar: TradingCalendar, train_start, train_end,
        cfg: TrainConfig, out_dir: Path, stage: str = "both",
        pretrained: tuple[str, str] | None = None, tiny: bool = False,
        ledger: bool = True) -> dict:
    """一折 × 一 seed × 一 lookback 的完整微调入口。

    pretrained = (tokenizer_path, predictor_path)，HF id 或本地目录；
    tiny=True 用随机初始化小模型（冒烟）。
    """
    set_seed(cfg.seed)
    if tiny:
        tokenizer, model = build_tiny()
    else:
        if pretrained is None:
            raise ValueError("pretrained paths required unless tiny=True")
        tokenizer, model = load_pretrained(*pretrained)

    index = build_window_index(panel, calendar, cfg.lookback, cfg.predict)
    index = filter_anchors(index, train_start, train_end)
    tr_idx, iv_idx = inner_split(index, calendar, cfg.inner_months, cfg.predict)
    if len(tr_idx) == 0 or len(iv_idx) == 0:
        raise ValueError(
            f"empty split: {len(tr_idx)} train / {len(iv_idx)} inner-val windows"
        )
    print(f"windows: {len(tr_idx)} train / {len(iv_idx)} inner-val "
          f"(lookback={cfg.lookback}, predict={cfg.predict}, seed={cfg.seed})")

    def _ds(idx):
        return KronosWindowDataset(panel, idx, calendar, cfg.lookback,
                                   cfg.predict, cfg.clip)

    train_ds, inner_ds = _ds(tr_idx), _ds(iv_idx)
    results = {"config": asdict(cfg), "n_train": len(tr_idx), "n_inner": len(iv_idx)}

    if stage in ("tokenizer", "both"):
        results["tokenizer"] = train_stage("tokenizer", tokenizer, None,
                                           train_ds, inner_ds, cfg, out_dir)
    if stage in ("predictor", "both"):
        results["predictor"] = train_stage("predictor", model, tokenizer,
                                           train_ds, inner_ds, cfg, out_dir)

    if ledger:
        best = {k: round(v["best_inner_loss"], 5) for k, v in results.items()
                if isinstance(v, dict) and "best_inner_loss" in v}
        append_ledger(
            f"train | stage={stage} lookback={cfg.lookback} seed={cfg.seed} "
            f"window=[{pd.Timestamp(train_start).date()}..{pd.Timestamp(train_end).date()}] "
            f"best_inner={best} out={out_dir}"
        )
    (out_dir / "run_summary.json").write_text(json.dumps(results, indent=2, default=str))
    return results


def append_ledger(line: str) -> None:
    """试验登记簿（预注册 §3）：append-only。"""
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LEDGER_PATH.exists():
        LEDGER_PATH.write_text("# 试验登记簿（append-only，预注册 §3）\n\n")
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(f"- {pd.Timestamp.now(tz='UTC').isoformat(timespec='seconds')} | {line}\n")


# ---------------------------------------------------------------- 冒烟

def make_smoke_panel(n_permno: int = 2, n_sessions: int = 240, seed: int = 0):
    """合成随机游走 OHLCV 面板 + 对应交易日历。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-06", periods=n_sessions)
    rows = []
    for pn in range(1, n_permno + 1):
        close = 50.0 * np.exp(np.cumsum(rng.normal(0, 0.02, n_sessions)))
        opn = close * np.exp(rng.normal(0, 0.005, n_sessions))
        hi = np.maximum(opn, close) * (1 + np.abs(rng.normal(0, 0.004, n_sessions)))
        lo = np.minimum(opn, close) * (1 - np.abs(rng.normal(0, 0.004, n_sessions)))
        vol = rng.uniform(5e5, 2e6, n_sessions)
        rows.append(pd.DataFrame({
            "PERMNO": pn, "DlyCalDt": dates, "DlyOpen": opn, "DlyHigh": hi,
            "DlyLow": lo, "DlyClose": close, "DlyVol": vol,
            "DlyPrcVol": vol * close,
        }))
    return pd.concat(rows, ignore_index=True), TradingCalendar(dates)


def smoke(out_dir: Path | None = None) -> dict:
    panel, cal = make_smoke_panel()
    cfg = TrainConfig(lookback=30, predict=6, batch_size=16, max_epochs=2,
                      patience=2, swa_k=2, inner_months=2, num_workers=0)
    out = out_dir or (REPO_ROOT / "outputs" / "smoke")
    res = run(panel, cal, cal.dates[0], cal.dates[-10], cfg, Path(out),
              stage="both", tiny=True, ledger=False)
    for st in ("tokenizer", "predictor"):
        h = res[st]["history"]
        assert all(np.isfinite(e["train_loss"]) and np.isfinite(e["inner_loss"]) for e in h), st
    print("SMOKE PASS:", {st: res[st]["swa_inner_loss"] for st in ("tokenizer", "predictor")})
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Kronos 微调（单卡，预注册规则）")
    ap.add_argument("--smoke", action="store_true", help="合成数据冒烟")
    ap.add_argument("--panel", help="日频面板 parquet（未过滤全量或已裁剪训练区）")
    ap.add_argument("--index-parquet", help="市场指数 parquet（交易日历来源）")
    ap.add_argument("--train-start")
    ap.add_argument("--train-end")
    ap.add_argument("--stage", default="both", choices=["tokenizer", "predictor", "both"])
    ap.add_argument("--lookback", type=int, default=90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pretrained-tokenizer", default="NeoQuasar/Kronos-Tokenizer-base")
    ap.add_argument("--pretrained-predictor", default="NeoQuasar/Kronos-small")
    ap.add_argument("--out", default="outputs/run")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-epochs", type=int, default=30)
    args = ap.parse_args()

    if args.smoke:
        smoke()
        return

    if not (args.panel and args.index_parquet and args.train_start and args.train_end):
        ap.error("real run needs --panel --index-parquet --train-start --train-end")
    panel = pd.read_parquet(args.panel)
    idx_df = pd.read_parquet(args.index_parquet)
    date_col = "caldt" if "caldt" in idx_df.columns else idx_df.columns[0]
    cal = TradingCalendar.from_market_index(idx_df, date_col)
    cfg = TrainConfig(lookback=args.lookback, seed=args.seed,
                      batch_size=args.batch_size, max_epochs=args.max_epochs)
    run(panel, cal, args.train_start, args.train_end, cfg, Path(args.out),
        stage=args.stage,
        pretrained=(args.pretrained_tokenizer, args.pretrained_predictor))


if __name__ == "__main__":
    main()
