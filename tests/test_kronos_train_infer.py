"""训练/推理冒烟：SWA 单元、score 公式、两阶段 tiny 训练 + 打分端到端。

端到端用随机初始化小模型跑真实代码路径（官方 tokenizer/model/predictor），
CPU 上秒级。Windows 拉下来后 `python -m kronos_ft.train --smoke` 跑的
就是同一条路径。
"""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from kronos_ft.infer import future_sessions, run_scoring, score_from_pred  # noqa: E402
from kronos_ft.models import build_tiny, swa_average  # noqa: E402
from kronos_ft.train import TrainConfig, make_smoke_panel, run  # noqa: E402
from kronos_ft.windows import build_scoring_index  # noqa: E402


def test_swa_average_is_elementwise_mean():
    a = {"w": torch.tensor([1.0, 3.0]), "n": torch.tensor([2], dtype=torch.long)}
    b = {"w": torch.tensor([3.0, 5.0]), "n": torch.tensor([7], dtype=torch.long)}
    avg = swa_average([a, b])
    assert torch.allclose(avg["w"], torch.tensor([2.0, 4.0]))
    assert avg["n"].item() == 7  # 非浮点取最后一个


def test_score_from_pred():
    pred = pd.DataFrame({"open": [100.0, 101, 102, 103, 104, 110]})
    assert score_from_pred(pred, 6) == pytest.approx(0.10)
    assert np.isnan(score_from_pred(pred.iloc[:4], 6))  # 行数不足


def test_future_sessions_extrapolates(cal):
    d, ex = future_sessions(cal, cal.dates[10], 6)
    assert not ex and list(d) == list(cal.dates[11:17])
    d2, ex2 = future_sessions(cal, cal.dates[-3], 6)
    assert ex2 and len(d2) == 6 and d2[0] == cal.dates[-2]


def test_end_to_end_smoke(tmp_path):
    panel, cal = make_smoke_panel(n_permno=2, n_sessions=140, seed=1)
    cfg = TrainConfig(lookback=20, predict=6, batch_size=16, max_epochs=2,
                      patience=2, swa_k=2, inner_months=2, num_workers=0,
                      device="cpu")
    res = run(panel, cal, cal.dates[0], cal.dates[-8], cfg, tmp_path,
              stage="both", tiny=True, ledger=False)
    for st in ("tokenizer", "predictor"):
        assert all(np.isfinite(e["inner_loss"]) for e in res[st]["history"])
        assert (tmp_path / f"{st}_final").exists()

    # 同一份 tiny 模型直接打分几个 anchor
    tokenizer, model = build_tiny()
    sidx = build_scoring_index(panel, cal, cfg.lookback).groupby("PERMNO").tail(2)
    scores = run_scoring(tokenizer, model, panel, sidx, cal, cfg.lookback,
                         predict=6, batch_size=4, device="cpu",
                         sample_count=2, verbose=False)
    assert len(scores) == len(sidx)
    assert scores["score"].notna().all()
    assert np.isfinite(scores["score"]).all()
