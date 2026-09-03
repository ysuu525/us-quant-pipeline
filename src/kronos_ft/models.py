"""模型加载与 device 选择。

- 正式训练：HF Hub 或本地路径 from_pretrained（NeoQuasar/Kronos-Tokenizer-base
  + NeoQuasar/Kronos-small 等，路径进 experiment 配置）；
- 冒烟：随机初始化的小模型（构造参数取自官方 finetune_csv 的默认表，
  维度缩到秒级可训），不需要网络与权重下载；
- device 自动降级 cuda → mps → cpu（4080 训练 / Mac 冒烟 同一份代码）。
"""

from __future__ import annotations

import torch

from . import import_kronos


def pick_device(preference: str | None = None) -> torch.device:
    if preference:
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_pretrained(tokenizer_path: str, predictor_path: str):
    KronosTokenizer, Kronos, _ = import_kronos()
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
    model = Kronos.from_pretrained(predictor_path)
    return tokenizer, model


# 冒烟用小模型：结构性参数（s1/s2_bits 等）与官方默认同义，只缩尺寸。
TINY_TOKENIZER_KW = dict(
    d_in=6, d_model=32, n_heads=2, ff_dim=64, n_enc_layers=1, n_dec_layers=1,
    ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
    s1_bits=4, s2_bits=4, beta=0.05, gamma0=1.0, gamma=1.1, zeta=0.05,
    group_size=2,
)
TINY_MODEL_KW = dict(
    s1_bits=4, s2_bits=4, n_layers=2, d_model=32, n_heads=2, ff_dim=64,
    ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
    token_dropout_p=0.0, learn_te=True,
)


def build_tiny():
    KronosTokenizer, Kronos, _ = import_kronos()
    return KronosTokenizer(**TINY_TOKENIZER_KW), Kronos(**TINY_MODEL_KW)


def swa_average(state_dicts: list[dict]) -> dict:
    """K 个 checkpoint 的等权权重平均（预注册 §1.2；模型为 LayerNorm 架构，
    无 BatchNorm 统计量需要重估）。非浮点张量（如整型 buffer）取最后一个。"""
    if not state_dicts:
        raise ValueError("no state dicts to average")
    out = {}
    for k in state_dicts[-1]:
        v = state_dicts[-1][k]
        if torch.is_floating_point(v):
            out[k] = torch.stack([sd[k].float() for sd in state_dicts]).mean(0).to(v.dtype)
        else:
            out[k] = v.clone()
    return out
