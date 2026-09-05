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
# 只有这些文件有正当理由提到封存标识。``xxx/*`` 是前缀通配（整个子目录）。
MARKER_ALLOWLIST = {
    "src/crsp_pipeline/sealed.py",
    "tests/test_sealed_mode.py",
    # 2026-09-05 解封读取（任务书 §1 G0）：唯一入口与其实现包。
    # src/unseal/paths.py 是全仓库唯一知道封存目录命名的模块。
    "scripts/unseal_read_confirm.py",
    "src/unseal/*",
    "tests/test_unseal_read_confirm.py",
}
# unseal 是解封开关；splits.py 的 sealed_oos_window 是既有 API
UNSEAL_ALLOWLIST = MARKER_ALLOWLIST | {
    "src/crsp_pipeline/splits.py",
    "tests/test_splits.py",
    # 显式的 unseal 参数：只把封存折号的路径解析转交 unseal.paths，
    # **不放宽 ALLOWED_FOLDS**（见该模块 docstring）。
    "src/signals/kronos_adapter.py",
}


def _in_allowlist(rel: str, allowlist: set[str]) -> bool:
    if rel in allowlist:
        return True
    return any(entry.endswith("/*") and rel.startswith(entry[:-1])
               for entry in allowlist)


def _py_sources():
    for pat in ("scripts/*.py", "src/**/*.py", "tests/*.py"):
        for f in REPO.glob(pat):
            yield f, f.relative_to(REPO).as_posix()


def test_no_ordinary_script_references_sealed_outputs():
    """封存靠流程纪律，不靠加密。这条测试是那条纪律的可执行版本：
    任何普通分析脚本都不得写死封存目录或封存 tag 前缀。"""
    bad = []
    for f, rel in _py_sources():
        if _in_allowlist(rel, MARKER_ALLOWLIST):
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
        if _in_allowlist(rel, UNSEAL_ALLOWLIST):
            continue
        if "unseal" in f.read_text(encoding="utf-8", errors="ignore"):
            bad.append(rel)
    assert not bad, f"以下脚本使用了 unseal 解封开关，须先取得读取授权: {bad}"


def _ledger_type(line: str) -> str:
    """登记行的类型字段 = 第二个「|」分隔字段（append_ledger 的固定格式）。"""
    parts = line.split("|")
    return parts[1].strip() if len(parts) >= 3 else ""


def test_sealed_ledger_lines_carry_no_metrics():
    """只查**类型字段为 sealed-compute** 的行。

    早先版本用子串匹配，会把正文里提到「sealed-compute 行」的授权 / 诊断条目
    （如 2026-09-05 的 `authorisation-and-sealed-mode`）一并卷进来误判。
    登记簿是 append-only、改不得，因此收严的是判据本身。
    """
    if not LEDGER.exists():
        pytest.skip("no ledger")
    bad = [ln for ln in LEDGER.read_text(encoding="utf-8").splitlines()
           if _ledger_type(ln) == "sealed-compute" and METRIC_TOKENS.search(ln)]
    assert not bad, f"封存登记条目里出现绩效字段: {bad}"


def test_ledger_type_field_matcher_is_narrow_but_still_catches_real_lines():
    """收严后的判据必须仍抓真条目、且不再误伤正文提及。"""
    real = ("- 2026-09-01T07:52:56+00:00 | sealed-compute | tag=sealed_ft_fold05 "
            "val=[2005-01-03..2005-07-01] rows=185441 （封存打分：未生成 labels）")
    mention = ("- 2026-09-05 | authorisation-and-sealed-mode | 登记簿只允许追加"
               "不含指标的 `sealed-compute` 行；不得计算任何 IC / 收益 / 分层统计。")
    assert _ledger_type(real) == "sealed-compute"
    assert _ledger_type(mention) == "authorisation-and-sealed-mode"
    # 正文提及确实带绩效词——旧的子串判据正是被它误判的
    assert METRIC_TOKENS.search(mention)
    if LEDGER.exists():
        lines = LEDGER.read_text(encoding="utf-8").splitlines()
        assert sum(1 for ln in lines if _ledger_type(ln) == "sealed-compute") > 0


def test_allowed_and_forbidden_sets_are_disjoint():
    assert not (ALLOWED_FILES & set(FORBIDDEN_FILES))
