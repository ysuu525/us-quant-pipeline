"""解封读取入口的合成数据全链路测试（任务书 §1「测试」）。

盯死的不是数值，而是**纪律**：

* **G0** 唯一入口跑通；逐折口径核对（``scores_sha256`` + config 五项 + 机械窗口）
  在任何重活之前完成，**任一项失配即中止整个运行**；
* **G1–G8** 各自的产物都在；
* **绝不写进封存目录**（封存目录仍只含 scores + 清单 + 哨兵 + 日志）；
* **不调用 ``append_ledger``**（源码里没有，跑完登记簿逐字节不变）；
* **不产生任何 FT−ZS 差值键**；
* **报告里没有逐折 IC 表**（连折号都不出现，只有合并量 + 正折数 + 符号计数）；
* fold44–45 是**独立输出目录、独立报告**，不并入 31 折合并统计。

合成读数没有任何含义，本测试也不断言任何读数的方向或大小。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from crsp_pipeline.sealed import FORBIDDEN_FILES, audit_dir  # noqa: E402
from unseal import aggregate  # noqa: E402
from unseal import config as C  # noqa: E402
from unseal.folds import fold_windows, load_calendar, parse_folds  # noqa: E402
from unseal.paths import (  # noqa: E402
    SEALED_FOLDS,
    SealedConfigMismatch,
    sealed_eval_dir,
    verify_fold,
)
from unseal.perfold import _delta_adv_daily  # noqa: E402
from unseal.report import render_report  # noqa: E402
from unseal.smoke import build_workspace  # noqa: E402

ENTRY = REPO / "scripts" / "unseal_read_confirm.py"
LEDGER = REPO / "experiments" / "ledger.md"
UNSEAL_SOURCES = [ENTRY] + sorted((REPO / "src" / "unseal").glob("*.py"))

SMOKE_FOLDS = (5, 21)          # 一早一晚，跨 2013-01-01 的年代切点
SMOKE_PERMNOS = 120
SMOKE_ARMS = C.ARMS_DEFAULT          # v4.1 附录：默认只读 FT
PROCESSED = Path(r"F:\quant\processed\crsp_ciz_2026-08-24_20260825T130601Z")


def _fingerprint(path: Path) -> bytes | None:
    return path.read_bytes() if path.is_file() else None


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    """跑一次 ``--smoke`` 全链路（含 --clean-window 与 --h1b）。"""
    out = tmp_path_factory.mktemp("unseal") / "unseal_smoke_selftest"
    ledger_before = _fingerprint(LEDGER)
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    proc = subprocess.run(
        [sys.executable, str(ENTRY), "--smoke", "--out", str(out),
         "--smoke-permnos", str(SMOKE_PERMNOS),
         "--folds", ",".join(str(f) for f in SMOKE_FOLDS),
         "--clean-window", "--memory-limit-gb", "1000000"],
        cwd=REPO, capture_output=True, text=True, timeout=3600, env=env,
        encoding="utf-8", errors="replace",
    )
    assert proc.returncode == 0, (proc.stdout or "")[-4000:] + (proc.stderr or "")[-4000:]
    return {"out": out, "ledger_before": ledger_before}


# ------------------------------------------------------------------ G0

def test_entry_writes_run_identity_and_caliber_check(smoke_run):
    out = smoke_run["out"]
    meta = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
    for key in ("run_id", "run_utc", "config_sha256", "code_sha256",
                "code_sha256_of_snapshot", "git_commit", "git_porcelain_lines",
                "protocol_path", "protocol_sha256", "protocol_tag",
                "authorisation_ref"):
        assert key in meta, key
    # 报告依据的是**冻结版** v4，哈希运行时现算
    assert meta["protocol_path"] == "experiments/confirmation_protocol_v4.md"
    assert meta["protocol_tag"] == "prereg-2026-09-05-v4"
    assert meta["protocol_sha256"] == hashlib.sha256(
        (REPO / meta["protocol_path"]).read_bytes()).hexdigest()
    for rel in ("report.md", "report_fold44_45.md"):
        head = (out / rel).read_text(encoding="utf-8")[:2000]
        assert meta["protocol_sha256"] in head and meta["protocol_tag"] in head
    assert (out / "git_porcelain.txt").is_file()
    # 口径核对在任何重活之前完成，且逐条记录了 config 五项
    checks = json.loads((out / "caliber_check.json").read_text(encoding="utf-8"))
    assert len(checks) == len(SMOKE_FOLDS) * len(SMOKE_ARMS)
    assert meta["arms_read"] == list(SMOKE_ARMS) == ["ft"]
    assert meta["zs_dropped"] is True
    assert meta["addendum_path"].endswith("v4.1_addendum_2026-09-05.md")
    for row in checks:
        assert row["scoring_config"] == C.REQUIRED_SCORING_CONFIG
        assert row["scores_sha256_checked"] is True


def test_caliber_mismatch_aborts(tmp_path):
    """哈希 / 口径 / 窗口任一失配都必须抛，且是在读分数之前抛。"""
    ws = build_workspace(tmp_path / "ws", n_permno=40, folds=(5,))
    w = ws["windows"][5]
    good = verify_fold(5, "ft", ws["outputs"], (w.val_start, w.val_end))
    assert good["scoring_config"] == C.REQUIRED_SCORING_CONFIG

    # (a) 窗口对不上机械生成的边界
    with pytest.raises(SealedConfigMismatch):
        verify_fold(5, "ft", ws["outputs"],
                    (w.val_start + pd.Timedelta(days=1), w.val_end))

    d = sealed_eval_dir(5, "ft", ws["outputs"])
    man = json.loads((d / "SEALED_MANIFEST.json").read_text(encoding="utf-8"))

    # (b) 打分口径失配（sample_count 被改成 20）
    bad = dict(man)
    bad["config"] = dict(man["config"]) | {"sample_count": 20}
    (d / "SEALED_MANIFEST.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(SealedConfigMismatch):
        verify_fold(5, "ft", ws["outputs"], (w.val_start, w.val_end))

    # (c) scores.parquet 的哈希对不上清单
    bad = dict(man)
    bad["scores_sha256"] = "0" * 64
    (d / "SEALED_MANIFEST.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(SealedConfigMismatch):
        verify_fold(5, "ft", ws["outputs"], (w.val_start, w.val_end))


def test_sealed_folds_do_not_widen_the_development_allowlist():
    from signals.kronos_adapter import (
        ALLOWED_FOLDS, FoldNotAllowedError, SEALED_FOLDS as ADAPTER_SEALED,
        scores_path,
    )
    assert ALLOWED_FOLDS == frozenset((1, 2, 3, 4, *range(36, 43)))
    assert ADAPTER_SEALED == SEALED_FOLDS
    assert not (ALLOWED_FOLDS & ADAPTER_SEALED)
    with pytest.raises(FoldNotAllowedError):
        scores_path(5, "ft")                      # 默认路径必须拒绝封存折
    p = scores_path(5, "ft", root=REPO / "outputs", unseal=True)
    assert p.name == "scores.parquet" and "fold05" in str(p)
    with pytest.raises(ValueError):
        scores_path(36, "ft", unseal=True)        # 开发折不走解封路径


def test_fold_windows_are_generated_mechanically_not_handwritten():
    """折窗口必须来自 walk_forward_folds，并与封存清单的 val_window 对得上。

    只读 ``market_index.parquet`` 与 ``SEALED_MANIFEST.json``（清单只含哈希与口径），
    **不读任何封存分数**。
    """
    if not (PROCESSED / "market_index.parquet").is_file():
        pytest.skip("no processed snapshot")
    cal = load_calendar(PROCESSED)
    want = (5, 20, 35, 44, 45)
    windows = fold_windows(cal, want)
    for f in want:
        man_path = sealed_eval_dir(f, "ft", REPO / "outputs") / "SEALED_MANIFEST.json"
        if not man_path.is_file():
            continue
        man = json.loads(man_path.read_text(encoding="utf-8"))
        assert man["val_window"] == [str(windows[f].val_start.date()),
                                     str(windows[f].val_end.date())]


def test_parse_folds_defaults_cover_the_confirmation_set():
    assert parse_folds("5-35") == C.CONFIRM_FOLDS
    assert parse_folds("44,45") == C.CLEAN_WINDOW_FOLDS
    assert len(C.CONFIRM_FOLDS) == 31


# ------------------------------------------------------------------ G1–G8 产物

def test_per_fold_artifacts_exist(smoke_run):
    out = smoke_run["out"]
    for arm in SMOKE_ARMS:
        for f in SMOKE_FOLDS:
            d = out / arm / f"fold{f:02d}"
            for name in ("labels.parquet", "metrics.json", "daily_ic.parquet",
                         "daily_money.parquet", "scores.parquet", "verify.json"):
                assert (d / name).is_file(), f"{d / name} 缺失"
            m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
            assert m["scoring_config"] == C.REQUIRED_SCORING_CONFIG
            ic = pd.read_parquet(d / "daily_ic.parquet")
            assert {"signal_date", "ic_full_h1", "ic_top500", "delta_adv"} <= set(ic.columns)


def test_summary_covers_every_delivered_endpoint(smoke_run):
    s = json.loads((smoke_run["out"] / "summary.json").read_text(encoding="utf-8"))
    for endpoint in ("h1", "h2", "h2_era", "e"):
        assert set(s[endpoint]) == set(SMOKE_ARMS), endpoint
        assert "zs" not in s[endpoint], f"{endpoint} 不应产出 ZS 量（v4.1 附录）"
    assert s["h6"]["unseal"]["markers"] == list(C.H6_MARKERS)
    assert s["h6"]["unseal"]["questions"] == list(C.H6_QUESTIONS)
    # H1b：C-1 已采纳 → 默认跑；只在 FT 臂
    assert s["h1b"]["meta"]["unseal"]["primary_spec"] == C.H1B_PRIMARY_SPEC
    assert s["h1b"]["meta"]["unseal"]["arms"] == ["ft"] == list(C.H1B_ARMS)
    assert C.H1B_DEFAULT_ENABLED is True
    for arm in SMOKE_ARMS:
        assert s["h2"][arm]["gates"].keys() == {
            "ci_low_gt_0",
            f"folds_positive_ge_{C.H2_POSITIVE_FOLDS_GATE}_of_{C.H2_TOTAL_FOLDS}",
            "mean_ic_top500_gt_0"}
        assert s["h2_era"][arm]["cut"] == C.ERA_CUT
        assert set(s["h2_era"][arm]["segments"]) == {"early", "late"}
        assert s["e"][arm]["deployment_line"]["verdict"].startswith("C 未实测")


def test_h6_only_runs_p1_with_q1_and_q2b(smoke_run):
    s = json.loads((smoke_run["out"] / "summary.json").read_text(encoding="utf-8"))
    got = {(row["marker"], row["question"]) for row in s["h6"]["summaries"]}
    assert {m for m, _ in got} == set(C.H6_MARKERS)
    assert {q for _, q in got} == set(C.H6_QUESTIONS)
    assert s["h6"]["params"]["nw_lag"] == C.H6_NW_LAG
    assert s["h6"]["params"]["sesoi_bp"] == C.H6_SESOI_BP


# ------------------------------------------------------------------ 硬禁令

def test_never_writes_into_sealed_directories(smoke_run):
    outputs = smoke_run["out"] / "_smoke_workspace" / "outputs"
    dirs = sorted(outputs.glob("*/eval_sealed_*"))
    assert dirs, "冒烟工作区里应当有封存目录"
    for d in dirs:
        rep = audit_dir(d)
        assert rep["clean"], rep
        assert not (set(p.name for p in d.iterdir()) & set(FORBIDDEN_FILES))


def test_ledger_gains_nothing_from_the_unseal_run(smoke_run):
    """登记簿是 append-only，且**本次解封运行不得往里写任何一行**。

    别的会话可能在并行往登记簿追加（例如封存打分队列），所以断言写成两条：
    历史前缀逐字节不变，且新增部分与本次解封运行无关。
    """
    before = smoke_run["ledger_before"]
    if before is None:
        pytest.skip("no ledger")
    after = LEDGER.read_bytes()
    assert after.startswith(before), "登记簿的历史被改写（append-only 被破坏）"
    appended = after[len(before):].decode("utf-8", errors="replace")
    run_id = smoke_run["out"].name
    assert "unseal" not in appended.lower(), f"本次解封运行往登记簿写了行: {appended}"
    assert run_id not in appended, f"本次解封运行往登记簿写了行: {appended}"


def test_sources_never_call_or_import_append_ledger():
    """按 AST 查**调用与导入**（docstring 里写「禁止 append_ledger」不算违规）。"""
    import ast

    bad = []
    for path in UNSEAL_SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                name = getattr(fn, "id", None) or getattr(fn, "attr", None)
                if name == "append_ledger":
                    bad.append(f"{path.name}: call")
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if any(a.name.split(".")[-1] == "append_ledger" for a in node.names):
                    bad.append(f"{path.name}: import")
    assert not bad, f"解封链路不得调用 append_ledger（登记条目由主会话手写）: {bad}"


def _walk_keys(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield f"{prefix}{k}"
            yield from _walk_keys(v, f"{prefix}{k}.")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_keys(v, f"{prefix}[{i}].")


def test_no_arm_difference_key_anywhere(smoke_run):
    out = smoke_run["out"]
    for rel in ("summary.json", "clean_window/summary.json"):
        payload = json.loads((out / rel).read_text(encoding="utf-8"))
        bad = [k for k in _walk_keys(payload)
               if any(frag in k.lower() for frag in C.FORBIDDEN_KEY_FRAGMENTS)]
        assert not bad, f"{rel} 出现臂间差值键: {bad}"


#: 报告里唯一允许出现折号的地方：v4 §2.5.1 强制披露的年代归段边界事实。
_ALLOWED_FOLD_MENTION = ("边界事实", "归段")


def _fold_mentions(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if re.search(r"fold\s?\d{2}", ln, re.I)]


def test_reports_contain_no_per_fold_ic_table(smoke_run):
    """报告只许有合并量 + 正折数 + 逐折符号计数；逐折数值留在 metrics.json。"""
    out = smoke_run["out"]
    for rel in ("report.md", "report_fold44_45.md"):
        text = (out / rel).read_text(encoding="utf-8")
        for line in _fold_mentions(text):
            assert any(k in line for k in _ALLOWED_FOLD_MENTION), (
                f"{rel} 出现非豁免的折号（疑似逐折表）: {line}")
            # 即便是豁免行，也不许带 IC / 收益量级的小数
            assert not re.search(r"[+-]\d*\.\d{4,}", line), (
                f"{rel} 的折号行带了读数: {line}")
        assert re.search(r"\d+/\d+", text), f"{rel} 应含正折数计数"
        assert "【实测】" in text and "【推断】" in text
        assert C.UNANSWERABLE in text


def test_clean_window_is_a_separate_delivery(smoke_run):
    out = smoke_run["out"]
    confirm = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    clean = json.loads((out / "clean_window" / "summary.json").read_text(encoding="utf-8"))
    assert confirm["scope"] == "confirm" and clean["scope"] == "clean_window"
    assert confirm["folds_count"] == len(SMOKE_FOLDS)
    assert clean["folds_count"] == len(C.CLEAN_WINDOW_FOLDS)
    # 确认集的输出树里没有干净窗那两折
    assert clean["arms"] == list(SMOKE_ARMS)
    for f in C.CLEAN_WINDOW_FOLDS:
        assert not (out / "ft" / f"fold{f:02d}").exists()
        assert (out / "clean_window" / "ft" / f"fold{f:02d}").is_dir()
    assert (out / "report_fold44_45.md").is_file()


# ------------------------------------------------------------------ 统计口径

def test_delta_adv_uses_lagged_adv_top500():
    """名字数 > 500 时 ΔADV 才有意义：top500 必须按滞后 ADV 选，且与全池不同。"""
    rng = np.random.default_rng(0)
    n = 700
    frame = pd.DataFrame({
        "signal_date": pd.Timestamp("2010-03-01"),
        "score": rng.normal(size=n),
        "label": rng.normal(size=n),
        "adv_lag": np.arange(n, dtype=float),
    })
    got = _delta_adv_daily(frame)
    assert len(got) == 1
    top = frame.nlargest(C.TOPN, "adv_lag")
    expect_top = top["score"].rank().corr(top["label"].rank())
    expect_full = frame["score"].rank().corr(frame["label"].rank())
    assert got["ic_top500"].iloc[0] == pytest.approx(expect_top)
    assert got["ic_full_adv"].iloc[0] == pytest.approx(expect_full)
    assert got["delta_adv"].iloc[0] == pytest.approx(expect_top - expect_full)
    assert got["n_top"].iloc[0] == C.TOPN

    # ADV 缺失的名字先被剔除
    frame2 = frame.copy()
    frame2.loc[frame2.index[:100], "adv_lag"] = np.nan
    assert _delta_adv_daily(frame2)["n_adv"].iloc[0] == n - 100


def _synthetic_daily_ic(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for k, fold in enumerate(range(5, 9)):
        days = pd.bdate_range("2005-01-03", periods=120) + pd.DateOffset(years=2 * k)
        parts.append(pd.DataFrame({
            "signal_date": days,
            "ic_full_h1": rng.normal(0.02, 0.05, len(days)),
            "ic_full_adv": rng.normal(0.02, 0.05, len(days)),
            "ic_top500": rng.normal(0.025, 0.05, len(days)),
            "delta_adv": rng.normal(0.005, 0.01, len(days)),
            "fold": fold,
        }))
    return pd.concat(parts, ignore_index=True)


def test_aggregate_summaries_are_well_formed():
    ic = _synthetic_daily_ic()
    h1 = aggregate.h1_summary(ic, "ft")
    assert h1["rank_ic"]["nw_lag"] == C.NW_LAG
    assert h1["rank_ic"]["z"] == C.Z95
    assert h1["folds_total"] == 4
    assert C.H1_DEV_BASE == {"ft": 0.0207, "zs": 0.01919}      # v4 §3 第 2 项
    assert h1["realized_shrinkage_ratio"] == pytest.approx(
        h1["rank_ic"]["mean"] / 0.0207)
    assert h1["main_prediction_interval"] == [0.0088812, 0.0182250]
    assert h1["h4_read_gate_ci_low_gt_0"] in (True, False)

    h2 = aggregate.h2_summary(ic, "ft")
    assert h2["delta_adv"]["z"] == C.Z_BONFERRONI_K2      # k=2 配置族 Bonferroni
    assert "没有量级门" in h2["form"] or "无量级门" in h2["form"]
    # 判据里没有任何量级门（v3 的 >= +0.0026 已删）
    assert not any("0.0026" in k or "ge_half" in k for k in h2["gates"])

    windows = fold_windows(load_calendar(PROCESSED), range(5, 9)) if (
        PROCESSED / "market_index.parquet").is_file() else None
    if windows is None:
        pytest.skip("no processed snapshot for era split")
    era = aggregate.h2_era_summary(ic, windows, "ft")
    assert era["cut"] == "2013-01-01"
    assert era["segments"]["early"]["n_folds"] + era["segments"]["late"]["n_folds"] == 4
    # v4 §2.5.1 的 MDE80 披露常量原样引用，不重算
    assert era["interaction_late_minus_early_delta_adv"]["mde80_disclosed"] == 0.00730
    assert era["interaction_late_minus_early_ic_top500"]["mde80_disclosed"] == 0.01923
    assert "fold20" in era["boundary_note"]


def test_era_split_matches_the_protocol_fold_membership():
    """v4 §2.5.1：早期 fold05–20（16 折）、晚期 fold21–35（15 折），按 val_start 归段。"""
    if not (PROCESSED / "market_index.parquet").is_file():
        pytest.skip("no processed snapshot")
    cal = load_calendar(PROCESSED)
    windows = fold_windows(cal, C.CONFIRM_FOLDS)
    cut = pd.Timestamp(C.ERA_CUT)
    early = tuple(sorted(f for f, w in windows.items() if w.val_start < cut))
    late = tuple(sorted(f for f, w in windows.items() if w.val_start >= cut))
    assert early == C.ERA_EARLY_FOLDS and len(early) == 16
    assert late == C.ERA_LATE_FOLDS and len(late) == 15
    # fold20 的验证窗末端越过切点，仍归早期段（协议写死的边界事实）
    assert windows[20].val_start < cut < windows[20].val_end


def test_e_summary_breakeven_and_deployment_line():
    rng = np.random.default_rng(3)
    money = pd.DataFrame({
        "date": pd.bdate_range("2005-01-03", periods=400),
        "gross": rng.normal(0.0004, 0.006, 400),
        "turn": np.full(400, 0.5),
        "fold": np.repeat([5, 6, 7, 8], 100),
    })
    e = aggregate.e_summary(money, "ft")
    drag = 2.0 * (1.0 / 1e4) * 0.5 / C.NT * 252 * 100
    assert e["drag_per_bp_annual_pct"] == pytest.approx(drag)
    assert e["breakeven_oneway_bp"] == pytest.approx(e["gross_annual_pct"] / drag)
    assert e["oneway_trades_per_year"] == pytest.approx(2 * 252 * 0.5 / C.NT)
    assert set(e["net_annual_pct_by_cost_bp"]) == {str(b) for b in C.COST_GRID_BP}
    d = e["deployment_line"]
    assert d["c_max_bp_main"] == pytest.approx(
        C.BE_DEV_BP["ft"] * (1 - C.DEPLOY_HAIRCUT_MAIN) - C.DEPLOY_RESERVE_BP)
    assert d["legacy_line"]["c_max_bp"] == pytest.approx(
        C.BE_DEV_BP["ft"] * (1 - C.DEPLOY_HAIRCUT_LEGACY) - C.DEPLOY_RESERVE_BP)
    # v4 §2.2 的主判值必须逐位对上：FT 7.28bp / ZS 3.51bp
    assert d["c_max_bp_main"] == pytest.approx(7.282, abs=5e-3)
    assert aggregate.e_summary(money, "zs")["deployment_line"]["c_max_bp_main"]         == pytest.approx(3.512, abs=5e-3)
    assert e["folds_total"] == 4


def test_render_report_never_leaks_fold_ids():
    ic = _synthetic_daily_ic()
    payload = {
        "run_meta": {"run_id": "t", "run_utc": "u", "n_verified": 8},
        "h1": {a: aggregate.h1_summary(ic, a) for a in C.ARMS},
        "h2": {a: aggregate.h2_summary(ic, a) for a in C.ARMS},
        "h2_era": {}, "e": {},
    }
    text = render_report(payload, title="t", scope="折 05–35（31 折）", n_folds=31)
    for line in _fold_mentions(text):
        assert any(k in line for k in _ALLOWED_FOLD_MENTION), line
    assert "H1b" in text and "裁定 C-1" in text        # v4 §2.4：C-1 已采纳


# ------------------------------------------------------------------ 被注入脚本的默认不变

def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ict_probe_defaults_unchanged_and_configure_restricts():
    ict = _load("ict_probe_for_test", "scripts/ict_pattern_probe.py")
    assert len(ict.FOLDS) == 11 and len(ict.MARKER_COLS) == 12
    assert ict.QUESTIONS == ("Q1", "Q2b", "Q2a") and ict.SKIP_P5P6 is False
    assert ict.NW_LAG == 6 and ict.SESOI_BP == 10.0 and ict.L_SWING == 20
    ict.configure(markers=C.H6_MARKERS, questions=C.H6_QUESTIONS)
    assert ict.MARKER_COLS == list(C.H6_MARKERS)
    assert ict.SKIP_P5P6 is True          # 只跑 P1 时可跳过 P5/P6 的派生循环
    with pytest.raises(ValueError):
        ict.configure(markers=("P9_bull",))


def test_ict_skip_p5p6_leaves_p1_to_p4_and_computable_bitwise_identical():
    ict = _load("ict_probe_bitwise", "scripts/ict_pattern_probe.py")
    rng = np.random.default_rng(11)
    n = 400
    close = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    opn = close * np.exp(rng.normal(0, 0.005, n))
    high = np.maximum(opn, close) * (1 + abs(rng.normal(0, 0.004, n)))
    low = np.minimum(opn, close) * (1 - abs(rng.normal(0, 0.004, n)))
    full = ict.compute_patterns(opn, high, low, close)
    fast = ict.compute_patterns(opn, high, low, close, skip_p5p6=True)
    for key in ("P1_bull", "P1_bear", "P2_bull", "P2_bear", "P3_bull", "P3_bear",
                "P4_bull", "P4_bear", "computable"):
        assert np.array_equal(full[key], fast[key]), key


def test_authorisation_gate_requires_a_ledger_reference():
    """闸门不接受裸开关：必须给出能定位到登记簿某一行的凭据（行号或日期）。"""
    entry = _load("unseal_entry_for_test", "scripts/unseal_read_confirm.py")
    for bad in (None, "", "   ", "yes", "authorised"):
        with pytest.raises(SystemExit):
            entry.check_authorisation(bad)
    assert entry.check_authorisation("ledger:517") == "ledger:517"
    assert entry.check_authorisation("  2026-09-05  ") == "2026-09-05"


def test_arms_default_is_the_frozen_k2_and_order_is_fixed():
    """v4 §1 是 k=2 双臂、顺序 FT → ZS；v4.1 附录裁定抛弃 ZS，故默认只读 FT。

    ZS 分支代码保留（附录第 4 节：生效前 R-build 仍保留双臂能力），
    传 `--arms ft,zs` 即可恢复 k=2。**z 维持 2.2414 不放宽。**
    """
    entry = _load("unseal_entry_arms", "scripts/unseal_read_confirm.py")
    ns = entry.build_parser().parse_args(["--out", "x"])
    assert ns.arms == "ft" == ",".join(C.ARMS_DEFAULT)      # v4.1 附录：默认抛弃 ZS
    assert C.ARMS == ("ft", "zs")                            # 双臂能力保留


def test_entry_refuses_to_run_without_the_frozen_protocol(monkeypatch, tmp_path):
    entry = _load("unseal_entry_missing_protocol", "scripts/unseal_read_confirm.py")
    monkeypatch.setattr(entry.C, "PROTOCOL_PATH", tmp_path / "nope.md")
    with pytest.raises(SystemExit):
        entry.build_run_meta(tmp_path, entry.build_parser().parse_args(
            ["--out", str(tmp_path), "--smoke"]))


def test_exp11_defaults_unchanged_by_injection_points():
    exp11 = _load("exp11_for_test", "scripts/exp11_spanning_extended.py")
    assert exp11.FOLDS == tuple(range(36, 43))
    assert exp11.MEMORY_LIMIT_GB == 40.0
    assert set(exp11.SPEC_COLUMNS) == set(C.H1B_SPECS)
    assert exp11.K6B.NT == C.H1B_NT              # H1b 沿 K6b 的 NT=6
    assert exp11.RETENTION_THRESHOLD_PCT == 75.0
    assert exp11.CONSISTENCY_THRESHOLD == 5


# ------------------------------------------------------------------ H3

def test_h3_paired_delta_ic_is_computed_on_the_common_observation_set():
    """ΔIC = Kronos − 树，两侧在同一观测集（标签 ok ∧ 两边分数非缺）上算。"""
    from unseal.h3 import daily_paired_ic

    day = pd.Timestamp("2010-03-01")
    n = 40
    rng = np.random.default_rng(5)
    permnos = np.arange(1, n + 1)
    kronos = pd.DataFrame({"PERMNO": permnos, "signal_date": day,
                           "score": rng.normal(size=n)})
    tree = pd.DataFrame({"PERMNO": permnos, "signal_date": day,
                         "score": rng.normal(size=n)})
    labels = pd.DataFrame({"PERMNO": permnos, "signal_date": day,
                           "status": ["ok"] * n, "label": rng.normal(size=n)})
    labels.loc[labels.index[:5], "status"] = "invalid"   # 非 ok 的行必须被剔除
    tree.loc[tree.index[5:8], "score"] = np.nan          # 树侧缺分的行也剔除
    got = daily_paired_ic(kronos, tree, labels)
    assert len(got) == 1
    assert got["n_obs"].iloc[0] == n - 5 - 3
    assert got["delta_ic"].iloc[0] == pytest.approx(
        got["ic_kronos"].iloc[0] - got["ic_tree"].iloc[0])


def test_h3_aborts_when_tree_config_or_jkp_hash_drifts_across_folds(tmp_path):
    """跨折 config_sha256 / jkp_snapshot_sha256 必须唯一，漂移即中止整个运行。"""
    from unseal import paths as UP
    from unseal.h3 import verify_tree_folds

    ws = build_workspace(tmp_path / "ws", n_permno=40, folds=(5, 6))
    windows = [ws["windows"][5], ws["windows"][6]]
    rows = verify_tree_folds(ws["outputs"], windows)
    assert len(rows) == 2 and rows[0]["config_sha256"] == rows[1]["config_sha256"]

    d = UP.tree_sealed_dir(6, ws["outputs"])
    man = json.loads((d / "SEALED_MANIFEST.json").read_text(encoding="utf-8"))
    man["jkp_snapshot_sha256"] = "f" * 64
    (d / "SEALED_MANIFEST.json").write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(SealedConfigMismatch):
        verify_tree_folds(ws["outputs"], windows)


def test_h3_summary_is_estimate_only_and_discloses_the_worse_power(smoke_run):
    s = json.loads((smoke_run["out"] / "summary.json").read_text(encoding="utf-8"))
    h3 = s["h3"]
    assert h3["arm"] == C.H3_ARM == "ft"          # v4.1 附录连带变更
    assert h3["mde80_dev7_disclosed"] == C.H3_MDE80_DEV7 == 0.0329
    assert h3["mde80_31fold_extrapolated"] == C.H3_MDE80_31FOLD == 0.0156
    assert all(x < h3["mde80_31fold_extrapolated"] for x in h3["sesoi_candidates"])
    assert C.UNANSWERABLE in h3["reading"]
    # 估计交付：不得有任何 PASS/FAIL 形态的判据键（门 / 通过 / 合格）
    verdict_keys = [k for k in _walk_keys(h3)
                    if any(w in k.lower() for w in
                           ("gate", "pass", "eligible", "verdict", "reject"))]
    assert not verdict_keys, verdict_keys
    assert "不产生 PASS/FAIL" in h3["form"]
    assert len(h3["caliber_check"]) == len(SMOKE_FOLDS)
    for row in h3["caliber_check"]:
        assert row["scores_sha256_checked"] is True
