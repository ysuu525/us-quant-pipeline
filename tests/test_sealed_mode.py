"""封存模式的硬断言（2026-09-02 授权：计算 != 读取）。

这些测试盯死三件事：
1. 封存目录里**只有** scores.parquet / SEALED / SEALED_MANIFEST.json / sealed_run.log；
2. 绝不出现 labels.parquet / metrics.json / daily_ic.parquet / report.md；
3. 守卫确实拒绝读取带哨兵的目录；ledger 里的封存条目不含任何 RankIC / t / 收益字段。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from crsp_pipeline.sealed import (  # noqa: E402
    ALLOWED_FILES,
    FORBIDDEN_FILES,
    SealedReadError,
    assert_readable,
    audit_dir,
    write_seal,
)

SEALED_ROOT = REPO / "outputs" / "sealed_confirm"
LEDGER = REPO / "experiments" / "ledger.md"
# 封存条目里绝不允许出现的绩效字段
METRIC_TOKENS = re.compile(r"RankIC|rank_ic|\bt\s*=|IC=|夏普|sharpe|年化|decile|收益",
                           re.IGNORECASE)


def test_guard_blocks_sealed_dir(tmp_path):
    d = tmp_path / "foldXX"
    d.mkdir()
    (d / "scores.parquet").write_bytes(b"x")
    write_seal(d, {"fold_tag": "unit-test"})
    with pytest.raises(SealedReadError):
        assert_readable(d / "scores.parquet")
    # 显式解封授权后放行
    assert_readable(d / "scores.parquet", unseal=True)


def test_guard_allows_unsealed_dir(tmp_path):
    d = tmp_path / "plain"
    d.mkdir()
    (d / "scores.parquet").write_bytes(b"x")
    assert_readable(d / "scores.parquet")


def test_write_seal_refuses_when_forbidden_file_present(tmp_path):
    d = tmp_path / "dirty"
    d.mkdir()
    (d / "scores.parquet").write_bytes(b"x")
    (d / "labels.parquet").write_bytes(b"x")
    with pytest.raises(RuntimeError):
        write_seal(d, {"fold_tag": "unit-test"})


def _sealed_dirs() -> list[Path]:
    """真实的封存目录是 outputs/*/eval_sealed_*，不是 outputs/sealed_confirm/
    （后者只放 _logs 与队列锁）。早期版本扫错了地方，属空跑通过。"""
    out = REPO / "outputs"
    if not out.exists():
        return []
    return sorted(p for p in out.glob("*/eval_sealed_*") if p.is_dir())


@pytest.mark.parametrize("sub", _sealed_dirs(), ids=lambda p: p.name)
def test_no_sealed_dir_ever_holds_forbidden_artifacts(sub):
    """对所有封存目录都成立，包括正在写入的——禁止产物一个都不许有。"""
    names = {p.name for p in sub.iterdir()}
    bad = names & set(FORBIDDEN_FILES)
    assert not bad, f"封存目录出现禁止产物 {sorted(bad)}: {sub}"


@pytest.mark.parametrize(
    "sub", [p for p in _sealed_dirs() if (p / "SEALED_MANIFEST.json").exists()],
    ids=lambda p: p.name)
def test_each_completed_sealed_dir_is_clean(sub):
    """已封存完成的目录（有清单的）必须完整且只含允许文件。"""
    rep = audit_dir(sub)
    assert not rep["forbidden_present"], f"封存目录出现禁止产物: {rep}"
    assert not rep["unexpected"], f"封存目录出现意外文件: {rep}"
    assert not rep["missing"], f"封存目录缺文件: {rep}"
    man = json.loads((sub / "SEALED_MANIFEST.json").read_text(encoding="utf-8"))
    for key in ("snapshot_id", "code_sha256", "scores_sha256", "val_window", "config"):
        assert key in man, f"清单缺字段 {key}: {sub}"
    for key in ("lookback", "sample_count", "amp", "batch_size"):
        assert key in man["config"], f"清单 config 缺 {key}: {sub}"


# ---- 轻量纪律：普通分析脚本不得引用封存产物（替代批量改 18 个脚本）----

SEALED_MARKERS = ("sealed_confirm", "eval_sealed")
# 只有这些文件有正当理由提到封存标识
MARKER_ALLOWLIST = {
    "src/crsp_pipeline/sealed.py",
    "tests/test_sealed_mode.py",
}
# unseal 是解封开关；splits.py 的 sealed_oos_window 是既有 API
UNSEAL_ALLOWLIST = MARKER_ALLOWLIST | {
    "src/crsp_pipeline/splits.py",
    "tests/test_splits.py",
}


def _py_sources():
    for pat in ("scripts/*.py", "src/**/*.py", "tests/*.py"):
        for f in REPO.glob(pat):
            yield f, f.relative_to(REPO).as_posix()


def test_no_ordinary_script_references_sealed_outputs():
    """封存靠流程纪律，不靠加密。这条测试是那条纪律的可执行版本：
    任何普通分析脚本都不得写死封存目录或封存 tag 前缀。"""
    bad = []
    for f, rel in _py_sources():
        if rel in MARKER_ALLOWLIST:
            continue
        txt = f.read_text(encoding="utf-8", errors="ignore")
        hits = [m for m in SEALED_MARKERS if m in txt]
        if hits:
            bad.append(f"{rel}: {hits}")
    assert not bad, (
        "以下脚本引用了封存产物标识；封存结果的读取须用户另行授权: " + "; ".join(bad))


def test_unseal_switch_is_not_used_casually():
    bad = []
    for f, rel in _py_sources():
        if rel in UNSEAL_ALLOWLIST:
            continue
        if "unseal" in f.read_text(encoding="utf-8", errors="ignore"):
            bad.append(rel)
    assert not bad, f"以下脚本使用了 unseal 解封开关，须先取得读取授权: {bad}"


def test_sealed_ledger_lines_carry_no_metrics():
    if not LEDGER.exists():
        pytest.skip("no ledger")
    bad = [ln for ln in LEDGER.read_text(encoding="utf-8").splitlines()
           if "sealed-compute" in ln and METRIC_TOKENS.search(ln)]
    assert not bad, f"封存登记条目里出现绩效字段: {bad}"


def test_allowed_and_forbidden_sets_are_disjoint():
    assert not (ALLOWED_FILES & set(FORBIDDEN_FILES))
