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


def test_index_cache_roundtrip_and_invalidation(cal, tmp_path):
    import json
    import pandas as pd
    from kronos_ft.train import TrainConfig, _load_or_build_index, make_smoke_panel

    panel, c = make_smoke_panel(n_permno=2, n_sessions=60)
    cfg = TrainConfig(lookback=8, predict=4)
    cache = tmp_path / "idx.parquet"

    idx1 = _load_or_build_index(panel, c, cfg, False, cache)
    assert cache.exists() and cache.with_suffix(".meta.json").exists()
    idx2 = _load_or_build_index(panel, c, cfg, False, cache)   # 命中
    pd.testing.assert_frame_equal(idx1, idx2)

    # 面板行数变化 → meta 不匹配 → 重建
    panel3 = panel.iloc[:-1]
    idx3 = _load_or_build_index(panel3, c, cfg, False, cache)
    meta = json.loads(cache.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["panel_rows"] == len(panel3)
    assert len(idx3) <= len(idx1)


def test_dataset_preload_trims_to_index(cal):
    import pandas as pd
    from kronos_ft.dataset import KronosWindowDataset
    from kronos_ft.train import make_smoke_panel
    from kronos_ft.windows import build_window_index

    panel, c = make_smoke_panel(n_permno=3, n_sessions=40)
    idx = build_window_index(panel, c, 8, 4)
    only1 = idx[idx["PERMNO"] == 1].reset_index(drop=True)
    ds = KronosWindowDataset(panel, only1, c, 8, 4)
    assert set(ds._feat.keys()) == {1}          # 只预载被引用的股票
    x, _ = ds[0]
    assert x.shape == (12, 6)


def test_resume_bitwise_identical(tmp_path, monkeypatch):
    """断点续训：epoch 3 训练中"崩溃"→ 重跑同命令自动接续，
    最终 history 与不间断跑逐位一致；状态文件在阶段完成后被清除。"""
    import kronos_ft.train as T

    panel, cal = make_smoke_panel(n_permno=2, n_sessions=140, seed=1)
    cfg = TrainConfig(lookback=20, predict=6, batch_size=16, max_epochs=3,
                      patience=10, swa_k=2, inner_months=2, num_workers=0,
                      device="cpu")

    def _run(out_dir):
        return run(panel, cal, cal.dates[0], cal.dates[-8], cfg, out_dir,
                   stage="tokenizer", tiny=True, ledger=False)

    # 基准：不间断跑完 3 epoch
    clean = _run(tmp_path / "clean")["tokenizer"]

    # 模拟崩溃：第 3 个"训练 pass"（即 epoch 3）抛异常
    orig_epoch = T._epoch
    calls = {"train": 0}

    def bomb(stage, model, tokenizer, loader, device, cfg_, optimizer=None,
             scheduler=None):
        if optimizer is not None:
            calls["train"] += 1
            if calls["train"] == 3:
                raise RuntimeError("simulated crash")
        return orig_epoch(stage, model, tokenizer, loader, device, cfg_,
                          optimizer, scheduler)

    crash_dir = tmp_path / "crash"
    monkeypatch.setattr(T, "_epoch", bomb)
    with pytest.raises(RuntimeError, match="simulated crash"):
        _run(crash_dir)
    monkeypatch.setattr(T, "_epoch", orig_epoch)

    state = crash_dir / "tokenizer_resume_state.pt"
    assert state.exists()                       # 崩溃时留有 epoch 2 的状态
    assert torch.load(state, weights_only=False)["epoch"] == 2

    resumed = _run(crash_dir)["tokenizer"]      # 同命令重跑 → 从 epoch 3 接续
    assert not state.exists()                   # 完成后状态清除

    assert len(resumed["history"]) == len(clean["history"]) == 3
    for a, b in zip(resumed["history"], clean["history"]):
        assert a["epoch"] == b["epoch"]
        assert a["train_loss"] == pytest.approx(b["train_loss"], abs=1e-9)
        assert a["inner_loss"] == pytest.approx(b["inner_loss"], abs=1e-9)
    assert resumed["swa_inner_loss"] == pytest.approx(clean["swa_inner_loss"], abs=1e-9)


def test_resume_config_mismatch_starts_fresh(tmp_path):
    """状态文件配置指纹不符 → 拒绝续训、全新开始（防跨配置误接）。"""
    import kronos_ft.train as T

    panel, cal = make_smoke_panel(n_permno=2, n_sessions=140, seed=1)
    cfg = TrainConfig(lookback=20, predict=6, batch_size=16, max_epochs=2,
                      patience=10, swa_k=2, inner_months=2, num_workers=0,
                      device="cpu")
    out = tmp_path / "o"
    out.mkdir()
    torch.save({"key": {"stage": "tokenizer", "lookback": 999}, "epoch": 1,
                "history": [], "best_epoch": 1, "best_inner": 0.0,
                "optimizer": {}, "scheduler": {},
                "rng_cpu": torch.get_rng_state(), "rng_cuda": None},
               out / "tokenizer_resume_state.pt")
    res = run(panel, cal, cal.dates[0], cal.dates[-8], cfg, out,
              stage="tokenizer", tiny=True, ledger=False)["tokenizer"]
    assert [e["epoch"] for e in res["history"]] == [1, 2]   # 从 epoch 1 全新开始


def test_completed_stage_skipped_on_rerun(tmp_path, monkeypatch):
    """崩溃自愈：tokenizer 已完成后重跑 --stage both，不得重训 tokenizer。"""
    import kronos_ft.train as T

    panel, cal = make_smoke_panel(n_permno=2, n_sessions=140, seed=1)
    cfg = TrainConfig(lookback=20, predict=6, batch_size=16, max_epochs=2,
                      patience=10, swa_k=2, inner_months=2, num_workers=0,
                      device="cpu")
    out = tmp_path / "o"
    run(panel, cal, cal.dates[0], cal.dates[-8], cfg, out,
        stage="tokenizer", tiny=True, ledger=False)
    assert (out / "tokenizer_final").exists()

    orig = T.train_stage

    def guard(stage, *a, **k):
        assert stage != "tokenizer", "已完成的 tokenizer 不应被重训"
        return orig(stage, *a, **k)

    monkeypatch.setattr(T, "train_stage", guard)
    res = run(panel, cal, cal.dates[0], cal.dates[-8], cfg, out,
              stage="both", tiny=True, ledger=False)
    assert "best_inner_loss" in res["tokenizer"]      # 复用的 summary
    assert (out / "predictor_final").exists()          # predictor 正常训练


def test_fast_path_bitwise_identical(cal):
    """fast 路径必须与官方 predict_batch 路径逐位一致（同 RNG 种子下）。

    fast 只改 CPU 侧喂数方式（numpy 切片 + 按 anchor 缓存时间特征），
    喂给 generate 的数组、批次组成与顺序均不变 → 采样结果必须完全相同。
    """
    import numpy as np
    import torch

    from kronos_ft.models import build_tiny
    from kronos_ft.train import make_smoke_panel
    from kronos_ft.windows import build_scoring_index

    panel, c = make_smoke_panel(n_permno=3, n_sessions=120, seed=5)
    tokenizer, model = build_tiny()
    sidx = build_scoring_index(panel, c, 20).groupby("PERMNO").tail(3).reset_index(drop=True)
    assert len(sidx) >= 6

    def _run(fast):
        torch.manual_seed(1234)
        return run_scoring(tokenizer, model, panel, sidx, c, lookback=20,
                           predict=6, batch_size=4, device="cpu",
                           sample_count=2, verbose=False, fast=fast)

    slow = _run(False)
    fast = _run(True)
    assert list(slow["PERMNO"]) == list(fast["PERMNO"])
    assert list(slow["signal_date"]) == list(fast["signal_date"])
    assert list(slow["extrapolated"]) == list(fast["extrapolated"])
    np.testing.assert_array_equal(slow["score"].to_numpy(), fast["score"].to_numpy())


def test_stamp_matrix_matches_official(cal):
    """_stamp_matrix 与官方 calc_time_stamps 同值同序。"""
    import sys

    import numpy as np

    from kronos_ft import import_kronos
    from kronos_ft.infer import _stamp_matrix

    import_kronos()  # 确保 third_party/kronos 在 sys.path 上
    from model.kronos import calc_time_stamps  # noqa: E402

    dates = cal.dates[:30]
    official = calc_time_stamps(pd.Series(dates)).values.astype(np.float32)
    np.testing.assert_array_equal(_stamp_matrix(dates), official)
