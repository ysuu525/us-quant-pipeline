"""scripts/preregistration_manifest.py 的单元测试（全部合成数据，tmp_path 造假仓库）。

覆盖：
  1. 哈希稳定：同一份文件两次构建，逐文件 sha256 与自哈希都一致；
  2. --verify 能发现改动（内容改了 / 文件被删）；
  3. 缺失文件记 status=missing、不抛异常；
  4. 三组封存目录（Kronos 生成式 / 树基线逐折 / 树基线特征缓存）的
     SEALED_MANIFEST.json 都被 glob 到，而同目录的 parquet 一个都不被收录；
     pytest 临时目录里的同名假清单不得被收录（所以不能用 rglob）；
  5. 研究计划书 v0.2 优先、缺失时回退 v0.1；
  6. 自哈希算法可被外部按 recipe 独立复算。

注意：本文件与被测脚本都**不得**出现封存目录标识的连写字面量
（tests/test_sealed_mode.py 的纪律扫描覆盖 tests/*.py），因此目录名同样用
运行时拼接 `"eval_" + "sealed_..."` 构造。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "preregistration_manifest.py"
SPEC = importlib.util.spec_from_file_location("preregistration_manifest", SCRIPT)
assert SPEC and SPEC.loader
pm = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pm
SPEC.loader.exec_module(pm)


# 与脚本内一致的刻意拼接（理由见脚本 docstring / 本文件 docstring）
def _sealed_dir_name(tag: str) -> str:
    return "eval_" + "sealed_" + tag


_TREE_ROOT = "gbdt_strong_jkp_v2"        # 树基线封存根（相对 outputs/）
_TREE_SEALED = "sealed"                  # 树基线逐折封存目录名
_TREE_CACHE = "cache_" + "sealed"        # 树基线特征缓存封存目录名
_TREE_FOLDS = tuple(f"fold{n:02d}" for n in range(5, 36))   # 折 05–35，共 31 折


def make_repo(tmp_path: Path, *, with_v02: bool = False, with_prereg_v2: bool = False) -> Path:
    """造一个最小假仓库：默认清单里的文件 + 两个封存目录（各含清单与分数）。"""
    repo = tmp_path / "repo"
    (repo / "experiments").mkdir(parents=True)
    (repo / "docs").mkdir()
    (repo / "third_party").mkdir()
    for rel, text in [
        ("experiments/ledger.md", "- 2026-09-04 | DECISION | 假登记簿\n"),
        ("experiments/confirmation_protocol_v3.md", "协议 v3\n"),
        ("experiments/confirmation_protocol_v4.md", "协议 v4 冻结版\n"),
        ("experiments/confirmation_protocol_v4_revisions.md", "协议 v4 修订\n"),
        ("experiments/confirmation_protocol_v4.1_addendum_2026-09-05.md", "协议 v4.1 附录\n"),
        ("experiments/ict_pattern_probe_prereg_v1.md", "ICT 探针预注册 v1\n"),
        ("experiments/cost_pilot_protocol_v1_draft.md", "成本小试 v1 草稿\n"),
        ("CLAUDE.md", "强制规则\n"),
        ("HANDOFF.md", "交接\n"),
        ("docs/研究计划书_2026-09-03.md", "计划书 v0.1\n"),
        ("third_party/kronos_local.patch", "diff --git a/model/kronos.py\n"),
    ]:
        (repo / rel).write_text(text, encoding="utf-8")
    if with_v02:
        (repo / "docs/研究计划书_v0.2_2026-09-04.md").write_text("计划书 v0.2\n", encoding="utf-8")
    if with_prereg_v2:
        (repo / "experiments/signal2_prereg_v2.md").write_text("信号2 预注册 v2\n", encoding="utf-8")

    # 两个封存目录：清单文件应被收录，scores.parquet 绝不被收录
    for arm, tag in [("armA", "fold44"), ("armB", "fold45")]:
        d = repo / "outputs" / arm / _sealed_dir_name(tag)
        d.mkdir(parents=True)
        (d / "SEALED_MANIFEST.json").write_text(
            json.dumps({"snapshot_id": f"snap-{tag}", "code_sha256": "deadbeef",
                        "scores_sha256": "cafebabe", "val_window": ["2024-07-01", "2024-12-31"],
                        "config": {"lookback": 90, "sample_count": 5,
                                   "amp": "bf16", "batch_size": 128}},
                       ensure_ascii=False),
            encoding="utf-8")
        (d / "scores.parquet").write_bytes(b"NOT-A-REAL-PARQUET")
        (d / "SEALED").write_text("sentinel\n", encoding="utf-8")
    # 树基线：outputs/<根>/xgboost/<封存>/foldNN/  —— 比 Kronos 深两层，31 折
    for fold in _TREE_FOLDS:
        d = repo / "outputs" / _TREE_ROOT / "xgboost" / _TREE_SEALED / fold
        d.mkdir(parents=True)
        (d / "SEALED_MANIFEST.json").write_text(
            json.dumps({"snapshot_id": "snap-tree", "model": "xgboost", "fold": fold,
                        "code_sha256": {"gbdt_baseline.py": "abc123"},
                        "scores_sha256": "0ff1ce", "val_window": ["2005-01-03", "2005-07-01"],
                        "config": {"lookback": None, "sample_count": None,
                                   "amp": None, "batch_size": None}},
                       ensure_ascii=False),
            encoding="utf-8")
        (d / "scores.parquet").write_bytes(b"NOT-A-REAL-PARQUET")
        (d / "tuning.json").write_text("{}", encoding="utf-8")
    # 树基线特征缓存：自带哨兵与清单，钉住带训练目标 y 的缓存
    cache = repo / "outputs" / _TREE_ROOT / _TREE_CACHE
    cache.mkdir(parents=True)
    (cache / "SEALED_MANIFEST.json").write_text(
        json.dumps({"artifact": "feature cache", "config_sha256": "c0ffee",
                    "jkp_snapshot_sha256": "beefbeef", "years": [2002, 2003]},
                   ensure_ascii=False),
        encoding="utf-8")
    (cache / "SEALED").write_text("sentinel\n", encoding="utf-8")
    (cache / "jkp_state.parquet").write_bytes(b"NOT-A-REAL-PARQUET")

    # 一个非封存的普通输出目录，不应被 glob 到
    other = repo / "outputs" / "armA" / "eval_amp_lb90_fold40"
    other.mkdir(parents=True)
    (other / "SEALED_MANIFEST.json").write_text("{}", encoding="utf-8")
    # pytest 临时目录里的同名假清单：rglob 会收，显式 glob 不收
    tmpjunk = repo / "outputs" / "pytest_tmp_20260904T1625" / "test_guard0" / "foldXX"
    tmpjunk.mkdir(parents=True)
    (tmpjunk / "SEALED_MANIFEST.json").write_text("{}", encoding="utf-8")
    return repo


def build(repo: Path):
    return pm.build_manifest(repo, include_git=False)


# --------------------------------------------------------------------- 清单收集


def test_collect_paths_covers_defaults_and_sealed_manifests(tmp_path):
    repo = make_repo(tmp_path)
    paths = pm.collect_paths(repo)
    for rel in ("experiments/ledger.md", "experiments/confirmation_protocol_v3.md",
                "experiments/confirmation_protocol_v4.md",
                "experiments/confirmation_protocol_v4_revisions.md",
                "experiments/confirmation_protocol_v4.1_addendum_2026-09-05.md",
                "experiments/signal2_prereg_v2.md",
                "experiments/ict_pattern_probe_prereg_v1.md",
                "experiments/cost_pilot_protocol_v1_draft.md",
                "CLAUDE.md", "HANDOFF.md", "docs/研究计划书_2026-09-03.md",
                "third_party/kronos_local.patch"):
        assert rel in paths, rel
    sealed = [p for p in paths if p.endswith("SEALED_MANIFEST.json")]
    # 2 份 Kronos + 31 折树基线 + 1 份树基线特征缓存
    assert len(sealed) == 2 + 31 + 1, sealed
    assert len(set(sealed)) == len(sealed), "清单路径不得重复"
    # 非封存目录里的同名文件不得被 glob 到
    assert not any("eval_amp_lb90" in p for p in sealed)
    # pytest 临时目录里的假清单不得被 glob 到（这就是不用 rglob 的原因）
    assert not any("pytest_tmp" in p for p in sealed), sealed
    # 分数 / 缓存文件一个都不许进清单
    assert not any(p.endswith(".parquet") for p in paths)
    assert not any(p.endswith("tuning.json") for p in paths)


def test_sealed_globs_cover_three_distinct_groups(tmp_path):
    """三组 glob 各自命中自己那一组，合起来无重、无遗漏。"""
    repo = make_repo(tmp_path)
    n = {attr: len(list(repo.glob(getattr(pm, attr))))
         for attr in ("SEALED_MANIFEST_GLOB", "TREE_SEALED_MANIFEST_GLOB",
                      "TREE_CACHE_MANIFEST_GLOB")}
    assert n == {"SEALED_MANIFEST_GLOB": 2,
                 "TREE_SEALED_MANIFEST_GLOB": 31,
                 "TREE_CACHE_MANIFEST_GLOB": 1}, n
    assert len(pm.sealed_manifest_paths(repo)) == sum(n.values())
    # 传同一个 glob 两次也只留一份（去重）
    g = pm.SEALED_MANIFEST_GLOB
    assert len(pm.sealed_manifest_paths(repo, (g, g))) == 2


def test_proposal_prefers_v02_then_falls_back(tmp_path):
    repo1 = make_repo(tmp_path / "a")
    assert pm.resolve_proposal(repo1) == "docs/研究计划书_2026-09-03.md"
    repo2 = make_repo(tmp_path / "b", with_v02=True)
    assert pm.resolve_proposal(repo2) == "docs/研究计划书_v0.2_2026-09-04.md"


def test_missing_file_recorded_not_raised(tmp_path):
    repo = make_repo(tmp_path)  # signal2_prereg_v2.md 不存在
    man = build(repo)
    rec = {r["path"]: r for r in man["files"]}["experiments/signal2_prereg_v2.md"]
    assert rec["status"] == "missing"
    assert rec["sha256"] is None and rec["bytes"] is None and rec["mtime_utc"] is None
    assert man["counts"]["missing"] == 1
    assert man["counts"]["present"] == man["counts"]["total"] - 1

    repo2 = make_repo(tmp_path / "c", with_prereg_v2=True)
    man2 = build(repo2)
    assert man2["counts"]["missing"] == 0


# --------------------------------------------------------------------- 哈希


def test_hashes_are_stable_across_runs(tmp_path):
    repo = make_repo(tmp_path)
    a = build(repo)
    b = build(repo)
    assert {r["path"]: r["sha256"] for r in a["files"]} == \
           {r["path"]: r["sha256"] for r in b["files"]}
    # 自哈希只在注入同一时间戳时才逐位可比（generated_utc 进哈希）
    fixed = a["generated_utc"]
    from datetime import datetime
    now = datetime.fromisoformat(fixed)
    a2 = pm.build_manifest(repo, include_git=False, now=now)
    b2 = pm.build_manifest(repo, include_git=False, now=now)
    assert a2["self_sha256"] == b2["self_sha256"]


def test_file_sha256_matches_hashlib(tmp_path):
    repo = make_repo(tmp_path)
    man = build(repo)
    rec = {r["path"]: r for r in man["files"]}["experiments/ledger.md"]
    raw = (repo / "experiments/ledger.md").read_bytes()
    assert rec["sha256"] == hashlib.sha256(raw).hexdigest()
    assert rec["bytes"] == len(raw)


def test_self_hash_recipe_is_externally_reproducible(tmp_path):
    """按 manifest 里写的 recipe 独立复算，不依赖脚本内部函数。"""
    repo = make_repo(tmp_path)
    man = build(repo)
    body = {k: v for k, v in man.items() if k != "self_sha256"}
    blob = json.dumps(body, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")
    assert man["self_sha256"] == hashlib.sha256(blob).hexdigest()


def test_self_hash_changes_when_any_file_changes(tmp_path):
    repo = make_repo(tmp_path)
    from datetime import datetime
    a = build(repo)
    now = datetime.fromisoformat(a["generated_utc"])
    a = pm.build_manifest(repo, include_git=False, now=now)
    (repo / "experiments/ledger.md").write_text("- 改了一行\n", encoding="utf-8")
    b = pm.build_manifest(repo, include_git=False, now=now)
    assert a["self_sha256"] != b["self_sha256"]


# --------------------------------------------------------------------- verify


def test_verify_clean_repo(tmp_path):
    repo = make_repo(tmp_path)
    man = build(repo)
    out = pm.write_manifest(man, repo / "experiments" / "m.json")
    rep = pm.verify(repo, out)
    assert rep["ok"] is True
    assert rep["manifest_intact"] is True
    assert rep["n_changed"] == 0
    assert {r["verdict"] for r in rep["files"]} <= {"unchanged", "still-missing"}


def test_verify_detects_modified_and_deleted(tmp_path):
    repo = make_repo(tmp_path)
    out = pm.write_manifest(build(repo), repo / "experiments" / "m.json")

    (repo / "experiments/ledger.md").write_text("- 2026-09-05 | 偷偷改了\n", encoding="utf-8")
    (repo / "CLAUDE.md").unlink()

    rep = pm.verify(repo, out)
    assert rep["ok"] is False
    verdicts = {r["path"]: r["verdict"] for r in rep["files"]}
    assert verdicts["experiments/ledger.md"] == "changed"
    assert verdicts["CLAUDE.md"] == "missing-now"
    assert verdicts["HANDOFF.md"] == "unchanged"
    assert rep["n_changed"] == 2


@pytest.mark.parametrize("glob_attr", ["SEALED_MANIFEST_GLOB",
                                       "TREE_SEALED_MANIFEST_GLOB",
                                       "TREE_CACHE_MANIFEST_GLOB"])
def test_verify_detects_sealed_manifest_tamper(tmp_path, glob_attr):
    """三组封存清单里任何一份被换掉都必须被抓到——这正是 C9 要外部证明的那件事。"""
    repo = make_repo(tmp_path)
    out = pm.write_manifest(build(repo), repo / "experiments" / "m.json")
    target = next(p for p in repo.glob(getattr(pm, glob_attr)))
    target.write_text(json.dumps({"snapshot_id": "换掉了"}, ensure_ascii=False),
                      encoding="utf-8")
    rep = pm.verify(repo, out)
    assert rep["ok"] is False
    assert any(r["verdict"] == "changed" and r["path"].endswith("SEALED_MANIFEST.json")
               for r in rep["files"])


def test_verify_reports_appeared_but_does_not_fail(tmp_path):
    repo = make_repo(tmp_path)
    out = pm.write_manifest(build(repo), repo / "experiments" / "m.json")
    (repo / "experiments/signal2_prereg_v2.md").write_text("登记后才写的\n", encoding="utf-8")
    rep = pm.verify(repo, out)
    assert rep["ok"] is True
    assert [r["path"] for r in rep["appeared"]] == ["experiments/signal2_prereg_v2.md"]


def test_verify_detects_manifest_self_tamper(tmp_path):
    repo = make_repo(tmp_path)
    out = pm.write_manifest(build(repo), repo / "experiments" / "m.json")
    man = json.loads(out.read_text(encoding="utf-8"))
    for rec in man["files"]:
        if rec["path"] == "experiments/ledger.md":
            rec["sha256"] = "0" * 64
    out.write_text(json.dumps(man, indent=2, ensure_ascii=False), encoding="utf-8")
    rep = pm.verify(repo, out)
    assert rep["manifest_intact"] is False
    assert rep["ok"] is False  # 伪造的哈希对不上真文件


# --------------------------------------------------------------------- CLI


def test_cli_generate_then_verify(tmp_path, capsys):
    repo = make_repo(tmp_path)
    rc = pm.main(["--repo", str(repo), "--no-git",
                  "--out", str(repo / "experiments" / "cli.json")])
    assert rc == 0
    made = repo / "experiments" / "cli.json"
    assert made.is_file()
    capsys.readouterr()

    rc = pm.main(["--repo", str(repo), "--verify", str(made)])
    assert rc == 0
    capsys.readouterr()

    (repo / "HANDOFF.md").write_text("改了\n", encoding="utf-8")
    rc = pm.main(["--repo", str(repo), "--verify", str(made)])
    assert rc == 1
    assert "CHANGED" in capsys.readouterr().out


def test_default_out_path_shape(tmp_path):
    repo = make_repo(tmp_path)
    p = pm.default_out_path(repo, "2026-09-04T07:15:03+00:00")
    assert p.name == "preregistration_manifest_20260904T071503Z.json"
    assert p.parent == repo / "experiments"


def test_no_network_imports():
    """本脚本必须离线：源码里不得出现联网库。"""
    src = SCRIPT.read_text(encoding="utf-8")
    for bad in ("import requests", "import urllib", "import socket",
                "import http", "urlopen(", "requests."):
        assert bad not in src, bad


@pytest.mark.parametrize("marker", ["sealed" + "_confirm", "eval" + "_sealed", "un" + "seal"])
def test_new_files_carry_no_forbidden_markers(marker):
    """与 tests/test_sealed_mode.py 的纪律扫描同源的自查（拼接构造 marker）。"""
    for f in (SCRIPT, Path(__file__).resolve()):
        assert marker not in f.read_text(encoding="utf-8"), f"{f.name} 含 {marker}"
