"""把某个训练中途的 epoch 检查点物化成 predictor_final 形状的目录，供 evaluate_fold 打分。

用途：检验早停指标（内层生成 loss）与真正关心的指标（RankIC）是否对齐——
若 RankIC 在生成 loss 见顶之后仍继续上升，则现行早停会系统性地欠训练。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from kronos_ft.models import load_pretrained  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="把 epoch 检查点物化为可打分的模型目录")
    ap.add_argument("--model-dir", required=True, help="训练产物目录（含 *_final 与 epoch 检查点）")
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--out", required=True, help="输出目录（会写 tokenizer_final / predictor_final）")
    args = ap.parse_args()

    md, out = Path(args.model_dir), Path(args.out)
    ckpt = md / f"predictor_epoch{args.epoch:03d}.pt"
    if not ckpt.exists():
        raise SystemExit(f"检查点不存在: {ckpt}")

    out.mkdir(parents=True, exist_ok=True)
    # tokenizer 沿用最终版（本实验只动 predictor 的训练轮数）
    if not (out / "tokenizer_final").exists():
        shutil.copytree(md / "tokenizer_final", out / "tokenizer_final")

    _, model = load_pretrained(str(md / "tokenizer_final"), str(md / "predictor_final"))
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[warn] missing={len(missing)} unexpected={len(unexpected)}")
    model.save_pretrained(str(out / "predictor_final"))
    print(f"已物化 epoch {args.epoch} → {out / 'predictor_final'}")


if __name__ == "__main__":
    main()
