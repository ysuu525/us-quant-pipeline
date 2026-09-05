"""实验 10（P3/P4）的最小测试：合成数据、CPU、秒级。

重点是**读写边界的可执行断言**（任务书硬约束 1 与协调会话的补充 ①）：
折 44–45 / fold43 / ZS 基座的产物一律不得读；一切产物只能写 ``outputs/exp10_*``；
面板列白名单不含任何标签相关列；``crsp_pipeline.labels`` 的公开函数被绊线换掉，
一次都不许被调用（该包的 ``__init__`` 无条件导入 labels，所以「不在 sys.modules」
这种断言恒假、毫无保护力）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import exp10_contamination_probes as P3  # noqa: E402
import exp10_p4_adaptation as P4  # noqa: E402
from crsp_pipeline.calendar import TradingCalendar  # noqa: E402
from crsp_pipeline.sealed import write_seal  # noqa: E402


# ---------------------------------------------------------------- 边界断言

@pytest.mark.parametrize("rel", [
    "outputs/fold44_lb90_s0_poolB_universe/scores.parquet",
    "outputs/fold45_lb90_s0_poolB_universe/anything.parquet",
    "outputs/fold43_lb90_s0_poolB_universe/scores.parquet",
    "outputs/zeroshot_base/eval_zeroshot_fold36/scores.parquet",
])
def test_read_guard_blocks_clean_and_zs_outputs(rel):
    with pytest.raises(P3.BoundaryError):
        P3.assert_read_ok(REPO / rel)


@pytest.mark.parametrize("rel", [
    "outputs/fold36_lb90_s0_poolB_universe/eval_amp_lb90_fold36/scores.parquet",
    "outputs/fold36_lb90_s0_poolB_universe/eval_amp_lb90_fold36/metrics.json",
    "outputs/fold36_lb90_s0_poolB_universe/eval_amp_lb90_fold36/labels.parquet",
])
def test_read_guard_blocks_scores_labels_metrics_even_on_dev_folds(rel):
    """开发折的模型权重可读，但它的分数 / 标签 / 指标在本实验里也不该被打开。"""
    with pytest.raises(P3.BoundaryError):
        P3.assert_read_ok(REPO / rel)


def test_read_guard_blocks_label_side_panel(tmp_path):
    with pytest.raises(P3.BoundaryError):
        P3.assert_read_ok(tmp_path / "panel_raw.parquet")


def test_read_guard_allows_model_weights_and_own_outputs():
    ok = [
        "outputs/fold36_lb90_s0_poolB_universe/tokenizer_final/model.safetensors",
        "outputs/fold42_lb90_s0_poolB_universe/predictor_final/config.json",
        "outputs/fold39_lb90_s0_poolB_universe/predictor_epoch003.pt",
        "outputs/fold39_lb90_s0_poolB_universe/predictor_summary.json",
        "outputs/exp10_contamination_probes.json",
    ]
    for rel in ok:
        assert P3.assert_read_ok(REPO / rel)


def test_read_guard_blocks_other_sessions_outputs():
    with pytest.raises(P3.BoundaryError):
        P3.assert_read_ok(REPO / "outputs/gbdt_strong_jkp_v2/whatever.json")


def test_read_guard_blocks_dir_with_sentinel(tmp_path):
    d = tmp_path / "somewhere"
    d.mkdir()
    (d / "scores.parquet").write_bytes(b"x")
    write_seal(d, {"fold_tag": "unit-test"})
    from crsp_pipeline.sealed import SealedReadError
    with pytest.raises(SealedReadError):
        P3.assert_read_ok(d / "whatever.parquet")


@pytest.mark.parametrize("rel", [
    "outputs/fold44_lb90_s0_poolB_universe/exp10_note.json",
    "outputs/fold36_lb90_s0_poolB_universe/exp10_note.json",
    "outputs/zeroshot_base/exp10_note.json",
    "outputs/some_other_experiment/out.json",
])
def test_write_guard_blocks_everything_outside_exp10(rel):
    with pytest.raises(P3.BoundaryError):
        P3.assert_write_ok(REPO / rel)


def test_write_guard_allows_exp10_paths_and_paths_outside_outputs(tmp_path):
    assert P3.assert_write_ok(REPO / "outputs/exp10_contamination_probes.json")
    assert P3.assert_write_ok(REPO / "outputs/exp10_p3_nll.svg")
    assert P3.assert_write_ok(tmp_path / "scratch.json")


def test_column_guard_rejects_label_related_columns():
    assert P3.assert_columns_ok(["PERMNO", "DlyCalDt", "DlyOpen", "DlyClose"])
    for bad in (["DlyRet"], ["DlyDelFlg"], ["DlyCap"], ["label"]):
        with pytest.raises(P3.BoundaryError):
            P3.assert_columns_ok(["PERMNO", "DlyCalDt", *bad])


def test_label_tripwire_makes_label_generation_impossible():
    """本实验一行 label 都不生成：装上绊线后调用 labels.* 必须炸。

    注意 ``crsp_pipeline/__init__.py`` 无条件 ``from . import labels``，所以
    「不在 sys.modules」这种断言恒假；能保证的是**一次都不调用**。
    """
    import importlib

    import crsp_pipeline.labels as L
    try:
        tripped = P3.install_label_tripwire()
        assert "compute_labels" in tripped
        with pytest.raises(P3.BoundaryError):
            L.compute_labels(None, None, None)
    finally:
        importlib.reload(L)


def test_model_fold_whitelist_excludes_sealed_and_clean_folds():
    assert set(P3.ALLOWED_MODEL_FOLDS) == {36, 37, 38, 39, 40, 41, 42}
    assert set(P3.CLEAN_FOLDS) == {44, 45}
    assert not set(P3.ALLOWED_MODEL_FOLDS) & set(range(5, 36))
    assert not set(P3.ALLOWED_MODEL_FOLDS) & set(P3.CLEAN_FOLDS)


# ---------------------------------------------------------------- 数值路径

def _panel(n_permno=4, n_sessions=140, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_sessions)
    rows = []
    for pn in range(1, n_permno + 1):
        close = 40.0 * np.exp(np.cumsum(rng.normal(0, 0.02, n_sessions)))
        opn = close * np.exp(rng.normal(0, 0.005, n_sessions))
        hi = np.maximum(opn, close) * (1 + abs(rng.normal(0, 0.004, n_sessions)))
        lo = np.minimum(opn, close) * (1 - abs(rng.normal(0, 0.004, n_sessions)))
        vol = rng.uniform(5e5, 2e6, n_sessions)
        rows.append(pd.DataFrame({
            "PERMNO": pn, "DlyCalDt": dates, "DlyOpen": opn, "DlyHigh": hi,
            "DlyLow": lo, "DlyClose": close, "DlyVol": vol,
            "DlyPrcVol": vol * close}))
    panel = pd.concat(rows, ignore_index=True)
    uni = panel[["PERMNO", "DlyCalDt"]].copy()
    uni["in_universe"] = True
    return panel, uni, TradingCalendar(pd.DatetimeIndex(dates))


def test_window_set_builds_and_normalises_like_training():
    panel, uni, cal = _panel()
    lookback = 30
    ws = P3.WindowSet("synthetic", panel, uni, cal, lookback,
                      pd.Timestamp("2021-05-03"), pd.Timestamp("2021-06-30"))
    assert len(ws) > 0
    x, stamp = ws.batch(0, min(8, len(ws)))
    assert x.shape[1:] == (lookback, 6)
    assert stamp.shape[1:] == (lookback, 5)
    assert np.isfinite(x).all()
    assert x.min() >= -5.0 - 1e-6 and x.max() <= 5.0 + 1e-6
    # 逐窗 z-score：每个窗每列均值约 0（clip 未触发时）
    assert abs(float(x[0].mean())) < 0.5
    c = ws.confounders
    for k in ("n_windows", "n_names", "n_days", "xs_vol_daily_mean",
              "dollar_vol_median", "window_realized_vol_mean"):
        assert k in c


def test_per_window_nll_matches_official_batch_mean():
    """逐窗 NLL 的 batch 平均 == DualHead.compute_loss 的 batch 均值（口径不变）。"""
    import torch
    from kronos_ft.models import build_tiny

    torch.manual_seed(0)
    tk, md = build_tiny()
    tk.eval()
    md.eval()
    b, t = 3, 12
    x = torch.randn(b, t, 6)
    stamp = torch.zeros(b, t, 5)
    with torch.no_grad():
        s0, s1 = tk.encode(x, half=True)
        per = P3._per_window_nll(md, s0, s1, stamp, teacher_forcing=True)
        logits = md(s0[:, :-1], s1[:, :-1], stamp[:, :-1, :],
                    use_teacher_forcing=True, s1_targets=s0[:, 1:])
        ref, _, _ = md.head.compute_loss(logits[0], logits[1], s0[:, 1:], s1[:, 1:])
    assert per.shape == (b,)
    assert float(per.mean()) == pytest.approx(float(ref), rel=1e-5, abs=1e-6)


def _clustered(start, n_days, per_day, level, day_sd, win_sd, rng):
    """带**日效应**的合成 NLL：同日窗口共享一个日水平（真实数据就是这样）。"""
    days = pd.date_range(start, periods=n_days).to_numpy()
    day_eff = rng.normal(0.0, day_sd, n_days)
    vals = np.concatenate([level + day_eff[i] + rng.normal(0, win_sd, per_day)
                           for i in range(n_days)])
    return {"nll": vals, "anchor": np.repeat(days, per_day)}


def test_cluster_bootstrap_uses_dates_not_windows():
    rng = np.random.default_rng(0)
    clean = _clustered("2024-07-01", 20, 50, 3.2, 0.10, 0.02, rng)
    contam = _clustered("2023-07-03", 20, 50, 3.0, 0.10, 0.02, rng)
    out = P3.cluster_bootstrap_gap(clean, contam, n_boot=400)
    assert out["n_days_clean"] == 20 and out["n_days_contaminated"] == 20
    assert out["ci95"][0] < out["gap_clean_minus_contaminated"] < out["ci95"][1]
    # 同日 50 个窗不得被当成独立观测：聚类 SE 必须远大于朴素 SE
    naive = np.sqrt(np.var(clean["nll"], ddof=1) / clean["nll"].size
                    + np.var(contam["nll"], ddof=1) / contam["nll"].size)
    assert out["boot_sd"] > 3 * naive


def test_describe_and_yearly_series():
    v = np.array([1.0, 2.0, 3.0, 4.0])
    d = P3.describe(v)
    assert d["n"] == 4 and d["mean"] == pytest.approx(2.5)
    res = {"nll": v, "anchor": pd.to_datetime(
        ["2024-07-01", "2024-08-01", "2025-01-02", "2025-02-03"]).to_numpy()}
    ys = P3.yearly_series(res)
    assert set(ys) == {"2024", "2025"}
    assert ys["2024"]["mean"] == pytest.approx(1.5)


def test_svg_is_written_only_under_exp10(tmp_path):
    payload = {"title": "t", "by_model": {"fold36": {
        "sides": {"contaminated": {"n": 3, "mean": 3.0, "p10": 2.9, "p50": 3.0, "p90": 3.1},
                  "fold44": {"n": 3, "mean": 3.2, "p10": 3.1, "p50": 3.2, "p90": 3.3}},
        "yearly": {"contaminated": {"2023": {"n": 3, "mean": 3.0, "sd": 0.1}},
                   "fold44": {"2024": {"n": 3, "mean": 3.2, "sd": 0.1}}}}}}
    out = tmp_path / "exp10_p3_nll.svg"
    P3.write_svg(out, payload)
    assert out.read_text(encoding="utf-8").startswith("<svg")
    with pytest.raises(P3.BoundaryError):
        P3.write_svg(REPO / "outputs/fold44_x/exp10.svg", payload)


# ---------------------------------------------------------------- P4

def test_p4_layer_grouping_covers_both_stages():
    assert P4._group_of("predictor", "transformer.0.attn.q.weight") == "transformer"
    assert P4._group_of("predictor", "head.proj_s1.weight") == "head"
    assert P4._group_of("predictor", "embedding.emb_s1.weight") == "embedding"
    assert P4._group_of("tokenizer", "encoder.0.ffn.weight") == "encoder"
    assert P4._group_of("tokenizer", "decoder.1.ffn.weight") == "decoder"
    assert P4._group_of("tokenizer", "quant_embed.weight") == "quant_embed"


def test_p4_norms_are_zero_against_itself_and_positive_after_perturbation():
    import torch
    from kronos_ft.models import build_tiny

    torch.manual_seed(0)
    _, md = build_tiny()
    base = {k: v.clone() for k, v in md.state_dict().items()}
    same = P4._norms("predictor", base, base)
    assert same["__total__"]["delta_l2"] == pytest.approx(0.0, abs=1e-9)
    moved = {k: (v + 0.01 if torch.is_floating_point(v) else v)
             for k, v in base.items()}
    out = P4._norms("predictor", moved, base)
    assert out["__total__"]["delta_l2"] > 0
    assert out["__total__"]["rel"] > 0
    assert "transformer" in out and "head" in out


def test_p4_analyse_dir_on_tiny_training(tmp_path):
    """tiny 冒烟：真跑两个 epoch 的 tokenizer 阶段，再用 P4 读出曲线。"""
    import torch
    from kronos_ft.models import build_tiny
    from kronos_ft.train import TrainConfig, make_smoke_panel, run, set_seed

    panel, cal = make_smoke_panel(n_permno=2, n_sessions=140, seed=1)
    cfg = TrainConfig(lookback=20, predict=6, batch_size=16, max_epochs=2,
                      patience=5, swa_k=2, inner_months=2, num_workers=0,
                      device="cpu")
    out = tmp_path / "tinyrun"
    run(panel, cal, cal.dates[0], cal.dates[-8], cfg, out,
        stage="tokenizer", tiny=True, ledger=False)

    set_seed(cfg.seed)
    tk, md = build_tiny()
    base = {"tokenizer": {k: v.cpu() for k, v in tk.state_dict().items()},
            "predictor": {k: v.cpu() for k, v in md.state_dict().items()}}
    entry = P4.analyse_dir(out, base, guard=False)
    st = entry["stages"]["tokenizer"]
    assert st["n_epochs_on_disk"] == 2
    eps = st["epochs"]
    assert [e["epoch"] for e in eps] == [1, 2]
    assert all(e["train_loss"] is not None and e["inner_loss"] is not None for e in eps)
    assert all(e["norms"]["__total__"]["delta_l2"] > 0 for e in eps)
    # tiny 配置 n_enc_layers=1 → encoder/decoder 的 ModuleList 为空（真实模型
    # n_enc_layers=4 才有 encoder.0..2），故这里只断言分层字典非空且键都被识别
    assert set(eps[0]["norms"]) - {"__total__"}
    assert "other" not in eps[0]["norms"], eps[0]["norms"].keys()
    assert {"embed", "head", "quant_embed"} <= set(eps[0]["norms"])
    del torch


def test_p4_rejects_non_dev_folds():
    with pytest.raises(P3.BoundaryError):
        P3.assert_read_ok(REPO / "outputs/fold44_lb90_s0_poolB_universe/tokenizer_final/model.safetensors")
