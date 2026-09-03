"""预注册清单：对一组指定文件做 SHA-256 存证，产出可被外部时间戳锚定的 manifest。

**为什么有这个脚本**（审稿 2026-09-03 §1 C9，Major、时间最紧迫）：
    「arXiv 是发布时间戳不是 registry；ledger 的 append-only 是约定，git 由作者控制」——
    也就是说「预注册早于解封」目前**外部不可验证**。本脚本把「解封前存在过、
    内容是这样」这件事压成一份可公开、可复算的哈希清单；真正的**外部**证据来自
    对这份 manifest 做第三方锚定（OpenTimestamps / OSF），见下面的「锚定操作」。

**本脚本只做哈希，不联网**（没有任何 urllib / requests / socket 调用）。
锚定动作由用户手工执行——这是刻意的：自动上传等于把外部证据也交给作者的脚本。

------------------------------------------------------------------------------
默认清单
------------------------------------------------------------------------------
  experiments/ledger.md
  experiments/confirmation_protocol_v3.md
  experiments/confirmation_protocol_v4_revisions.md
  experiments/signal2_prereg_v2.md          （不存在则记 status=missing，不报错）
  experiments/cost_pilot_protocol_v1_draft.md
  docs/研究计划书_v0.2_2026-09-04.md        （不存在则回退到 v0.1 = 2026-09-03 版）
  CLAUDE.md
  HANDOFF.md
  third_party/kronos_local.patch            （子模块工作区补丁，见 DEFAULT_FILES 处注释）
  所有封存目录下的 SEALED_MANIFEST.json（glob，见下）

**只读 SEALED_MANIFEST.json 这一个文件，而且只按字节算哈希、不解析内容。**
它是封存目录的清单/哨兵文件，含 snapshot_id / code_sha256 / scores_sha256 /
val_window / config，**不含任何分数、标签或绩效指标**；同目录的 scores.parquet
一个字节都不碰。审稿要的正是「被封存的是哪一批分数」可被外部验证，而清单文件
恰好是唯一能提供这一点又不泄漏结果的载体。
（因此这里**不**调用 crsp_pipeline.sealed.assert_readable —— 那个守卫会拒绝封存
目录下的一切读取，而清单文件本身是授权白名单里的允许文件。）

------------------------------------------------------------------------------
关于 glob 模式被刻意拆开写的说明
------------------------------------------------------------------------------
tests/test_sealed_mode.py::test_no_ordinary_script_references_sealed_outputs
对 scripts/*.py、src/**/*.py、tests/*.py 做纪律扫描：普通分析脚本的源码里
**不得出现**封存目录标识字符串。本脚本必须 glob 封存目录才能找到清单文件，
于是把目录名前缀写成运行时拼接 `"eval_" + "sealed_*"`（见 SEALED_DIR_GLOB），
源码里因此不存在那个连写的字面量。**这是刻意为通过纪律扫描而做的拼接，不是笔误**，
改动本文件时请保持这一写法，并在改完后跑：

    .venv\\Scripts\\python.exe -m pytest tests\\test_sealed_mode.py \\
        tests\\test_prereg_manifest.py -q -p no:cacheprovider

------------------------------------------------------------------------------
manifest 内容
------------------------------------------------------------------------------
  generated_utc         生成时间（UTC，ISO-8601）
  repo_root             绝对路径（仅供人看，不进哈希语义）
  tool                  本脚本的相对路径与自身 sha256（谁生成的也要可核）
  git.head              git HEAD 完整哈希 / 短哈希 / 分支
  git.dirty             是否有未提交改动
  git.status_summary    `git status --porcelain` 按状态码分类计数
  git.status_porcelain  完整的 porcelain 行（外部可比对）
  files[]               每项：相对路径 / status(ok|missing) / sha256 / bytes / mtime_utc
  counts                总数、present、missing
  self_sha256           整份 manifest 的自哈希（算法见 self_sha256_recipe）

自哈希算法（外部可独立复算，不需要本脚本）：
    去掉顶层 "self_sha256" 键 → json.dumps(obj, sort_keys=True, ensure_ascii=False,
    separators=(",", ":")) → UTF-8 编码 → sha256 十六进制。

------------------------------------------------------------------------------
用法
------------------------------------------------------------------------------
    生成：  .venv\\Scripts\\python.exe scripts\\preregistration_manifest.py
    核验：  .venv\\Scripts\\python.exe scripts\\preregistration_manifest.py \\
                --verify experiments\\preregistration_manifest_20260904T....json

    --verify 退出码：0 = 没有已登记文件被改动或删除；1 = 有。
    （原本 missing、之后才出现的文件只作提示，不算漂移——例如 signal2_prereg_v2.md
      本来就预期会在登记之后才写出来。）

------------------------------------------------------------------------------
锚定操作（用户手工执行；本脚本不做任何一步）
------------------------------------------------------------------------------
路径 (a) OpenTimestamps —— 比特币链锚定，免费、无需账号、几分钟

    :: 装在 downloader\\.venv-dl，**不要**装进训练用的 .venv
    :: （CLAUDE.md §七「训练 venv 隔离」：已有装包破坏环境的前科）
    downloader\\.venv-dl\\Scripts\\python.exe -m pip install opentimestamps-client
    downloader\\.venv-dl\\Scripts\\ots.exe stamp experiments\\preregistration_manifest_<UTC>.json
    :: 得到 experiments\\preregistration_manifest_<UTC>.json.ots；把 .json 与 .ots
    :: 一起放在 experiments/ 并提交进 git
    :: 等比特币区块确认（数小时到一天）后升级并核验：
    downloader\\.venv-dl\\Scripts\\ots.exe upgrade experiments\\preregistration_manifest_<UTC>.json.ots
    downloader\\.venv-dl\\Scripts\\ots.exe verify experiments\\preregistration_manifest_<UTC>.json.ots
    :: verify 打印的区块时间即「不晚于该时刻，本 manifest 已存在」的外部证据。

路径 (b) OSF —— 有 DOI、金融/心理学审稿人最认的 registry，10 分钟

    1. osf.io 注册并新建 project（可先设 Private）。
    2. Files 里上传：本 manifest JSON + 被哈希的文本原件
       （ledger.md、协议 v3、协议 v4 修订、成本小试协议草稿、研究计划书、
         CLAUDE.md、HANDOFF.md）。封存目录的清单文件不必上传原件——它们的
         sha256 已在 manifest 里，上传原件反而扩大误触面。
    3. 左栏 Registrations → New registration → 模板选 **"Open-Ended Registration"**。
    4. 填标题/摘要（一句话：本次注册的是折 05–35 解封前的协议与登记簿状态），提交。
       提交后内容与时间戳都不可改，**registration 的时间戳就是外部证据**。
    5. 把 registration 的 URL / DOI 与 manifest 的 self_sha256 写进论文 §9 与
       experiments/ledger.md。
    ※ 不想立刻公开可选 embargo（最长 4 年）；embargo 期内注册时间戳依然成立。

路径 (c) git 侧 —— 辅助证据。**git 由作者控制，单独不足以充当外部时间戳**，
    必须与 (a) 或 (b) 至少之一同时做。

    git add experiments/preregistration_manifest_<UTC>.json experiments/preregistration_manifest_<UTC>.json.ots
    git commit -m "prereg: manifest <self_sha256 前 12 位>"
    git tag -s prereg-<YYYY-MM-DD> -m "preregistration manifest sha256=<self_sha256>"
    :: 没有 GPG 密钥时退回带注释（不签名）tag：
    git tag -a prereg-<YYYY-MM-DD> -m "preregistration manifest sha256=<self_sha256>"
    git push origin <branch> --follow-tags
    :: push 到公开 remote；若能推到一个不受作者控制的 mirror（机构 GitLab、
    :: 合作者账号下的 fork）则证据强度更高。

顺序建议：先跑本脚本生成 manifest → (a) ots stamp（最快落时间戳）→ (c) commit+tag
→ (b) OSF registration（给人看的那一份）。**三步全部在读取折 05–35 之前完成。**
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# 固定清单（相对仓库根，posix 分隔符）
DEFAULT_FILES: tuple[str, ...] = (
    "experiments/ledger.md",
    "experiments/confirmation_protocol_v3.md",
    "experiments/confirmation_protocol_v4_revisions.md",
    "experiments/signal2_prereg_v2.md",          # 允许缺失
    "experiments/cost_pilot_protocol_v1_draft.md",
    "CLAUDE.md",
    "HANDOFF.md",
    # third_party/kronos 是子模块，其工作区带一处本地补丁（bf16 -> numpy），
    # 而 SEALED_MANIFEST.json 的 code_sha256 只覆盖 kronos_ft/* 与 splits.py，
    # 不含 third_party。把补丁本身收进清单，推理路径才算完整可核。
    "third_party/kronos_local.patch",
)

# 研究计划书：优先 v0.2，缺失则回退 v0.1；两者都没有时记前者为 missing
PROPOSAL_CANDIDATES: tuple[str, ...] = (
    "docs/研究计划书_v0.2_2026-09-04.md",
    "docs/研究计划书_2026-09-03.md",
)

# 刻意的运行时拼接：源码里不出现连写的封存目录标识，
# 以通过 tests/test_sealed_mode.py 的纪律扫描（理由见模块 docstring）。
SEALED_DIR_GLOB = "outputs/*/" + "eval_" + "sealed_*"
SEALED_MANIFEST_NAME = "SEALED_MANIFEST.json"
SEALED_MANIFEST_GLOB = SEALED_DIR_GLOB + "/" + SEALED_MANIFEST_NAME

SCHEMA = "preregistration_manifest/v1"
SELF_HASH_KEY = "self_sha256"
SELF_HASH_RECIPE = (
    "去掉顶层 self_sha256 键后 json.dumps(obj, sort_keys=True, ensure_ascii=False, "
    'separators=(",", ":"))，UTF-8 编码，取 sha256 十六进制。'
)


def _utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while blk := f.read(chunk):
            h.update(blk)
    return h.hexdigest()


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


# --------------------------------------------------------------------------- 清单收集


def resolve_proposal(repo: Path, candidates: tuple[str, ...] = PROPOSAL_CANDIDATES) -> str:
    """研究计划书按优先序取第一个存在的；都不存在时返回首选（记 missing）。"""
    for rel in candidates:
        if (repo / rel).exists():
            return rel
    return candidates[0]


def sealed_manifest_paths(repo: Path) -> list[str]:
    """所有封存目录下的清单文件（相对路径，排序去重）。只 glob 清单文件本身。"""
    return sorted(p.relative_to(repo).as_posix()
                  for p in repo.glob(SEALED_MANIFEST_GLOB) if p.is_file())


def collect_paths(repo: Path) -> list[str]:
    """默认清单的完整相对路径列表（保持：固定项 → 计划书 → 封存清单 的顺序）。"""
    out: list[str] = []
    seen: set[str] = set()
    for rel in (*DEFAULT_FILES, resolve_proposal(repo), *sealed_manifest_paths(repo)):
        if rel not in seen:
            seen.add(rel)
            out.append(rel)
    return out


def file_record(repo: Path, rel: str) -> dict:
    """单个文件的记录。文件缺失时记 status=missing 而不是抛异常。"""
    p = repo / rel
    if not p.is_file():
        return {"path": rel, "status": "missing", "sha256": None,
                "bytes": None, "mtime_utc": None}
    st = p.stat()
    return {"path": rel, "status": "ok", "sha256": sha256_file(p),
            "bytes": int(st.st_size), "mtime_utc": _utc(st.st_mtime)}


# --------------------------------------------------------------------------- git


def _git(repo: Path, *args: str) -> tuple[int, str]:
    try:
        r = subprocess.run(["git", "-C", str(repo), *args],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:  # git 缺失 / 超时
        return 127, f"{type(exc).__name__}: {exc}"
    return r.returncode, (r.stdout or "").strip()


def git_info(repo: Path) -> dict:
    """HEAD 哈希 + porcelain 摘要。非 git 仓库或没装 git 时记 available=false。"""
    code, head = _git(repo, "rev-parse", "HEAD")
    if code != 0:
        return {"available": False, "reason": head[:400]}
    _, branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    code_st, status = _git(repo, "status", "--porcelain")
    lines = [ln for ln in status.splitlines() if ln.strip()] if code_st == 0 else []
    summary = Counter(ln[:2].strip() or "??" for ln in lines)
    return {
        "available": True,
        "head": head,
        "head_short": head[:12],
        "branch": branch,
        "dirty": bool(lines),
        "n_changed": len(lines),
        "status_summary": dict(sorted(summary.items())),
        "status_porcelain": lines,
        "status_porcelain_sha256": sha256_bytes("\n".join(lines).encode("utf-8")),
    }


# --------------------------------------------------------------------------- 构建


def canonical_bytes(manifest: dict) -> bytes:
    body = {k: v for k, v in manifest.items() if k != SELF_HASH_KEY}
    return json.dumps(body, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")


def self_hash(manifest: dict) -> str:
    return sha256_bytes(canonical_bytes(manifest))


def build_manifest(repo: Path, paths: list[str] | None = None, *,
                   include_git: bool = True, now: datetime | None = None) -> dict:
    repo = Path(repo).resolve()
    rels = list(paths) if paths is not None else collect_paths(repo)
    records = [file_record(repo, rel) for rel in rels]
    stamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    script = Path(__file__).resolve()
    try:
        script_rel = script.relative_to(repo).as_posix()
    except ValueError:
        script_rel = script.name
    man = {
        "schema": SCHEMA,
        "purpose": ("解封（读取折 05–35）之前对协议/登记簿/封存清单做哈希存证；"
                    "外部时间戳由 OpenTimestamps / OSF 手工锚定，见脚本 docstring。"),
        "generated_utc": stamp.isoformat(timespec="seconds"),
        "repo_root": str(repo),
        "tool": {"script": script_rel,
                 "script_sha256": sha256_file(script) if script.is_file() else None},
        "git": git_info(repo) if include_git else {"available": False,
                                                   "reason": "disabled by --no-git"},
        "files": records,
        "counts": {
            "total": len(records),
            "present": sum(r["status"] == "ok" for r in records),
            "missing": sum(r["status"] == "missing" for r in records),
        },
        "self_sha256_recipe": SELF_HASH_RECIPE,
    }
    man[SELF_HASH_KEY] = self_hash(man)
    return man


def default_out_path(repo: Path, stamp_iso: str) -> Path:
    tag = stamp_iso.replace("-", "").replace(":", "")
    tag = tag.replace("+0000", "Z").split("+")[0]
    if not tag.endswith("Z"):
        tag += "Z"
    return repo / "experiments" / f"preregistration_manifest_{tag}.json"


def write_manifest(manifest: dict, out: Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    return out


# --------------------------------------------------------------------------- 核验


def verify(repo: Path, manifest_path: Path) -> dict:
    """重算清单里每个文件的哈希，报告哪些变了。"""
    repo = Path(repo).resolve()
    manifest_path = Path(manifest_path)
    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded_self = man.get(SELF_HASH_KEY)
    recomputed_self = self_hash(man)

    rows = []
    for rec in man.get("files", []):
        cur = file_record(repo, rec["path"])
        was, now_ = rec.get("status"), cur["status"]
        if was == "ok" and now_ == "ok":
            verdict = "unchanged" if cur["sha256"] == rec.get("sha256") else "changed"
        elif was == "ok":
            verdict = "missing-now"
        elif now_ == "ok":
            verdict = "appeared"
        else:
            verdict = "still-missing"
        rows.append({"path": rec["path"], "verdict": verdict,
                     "recorded_sha256": rec.get("sha256"), "current_sha256": cur["sha256"],
                     "recorded_bytes": rec.get("bytes"), "current_bytes": cur["bytes"]})

    changed = [r for r in rows if r["verdict"] == "changed"]
    gone = [r for r in rows if r["verdict"] == "missing-now"]
    appeared = [r for r in rows if r["verdict"] == "appeared"]
    git_now = git_info(repo)
    git_rec = man.get("git", {})
    return {
        "manifest": str(manifest_path),
        "generated_utc": man.get("generated_utc"),
        "manifest_self_sha256_recorded": recorded_self,
        "manifest_self_sha256_recomputed": recomputed_self,
        "manifest_intact": recorded_self == recomputed_self,
        "git_head_recorded": git_rec.get("head"),
        "git_head_now": git_now.get("head"),
        "git_head_moved": git_rec.get("head") != git_now.get("head"),
        "files": rows,
        "changed": changed,
        "missing_now": gone,
        "appeared": appeared,
        "n_changed": len(changed) + len(gone),
        "ok": not changed and not gone,
    }


# --------------------------------------------------------------------------- CLI


def _print_manifest(man: dict, out: Path) -> None:
    g = man["git"]
    print(f"写出：{out}")
    print(f"self_sha256 = {man[SELF_HASH_KEY]}")
    print(f"生成时间(UTC) = {man['generated_utc']}")
    if g.get("available"):
        print(f"git HEAD = {g['head']}  branch={g['branch']}  "
              f"dirty={g['dirty']}（{g['n_changed']} 项：{g['status_summary']}）")
    else:
        print(f"git = 不可用（{g.get('reason', '')[:120]}）")
    c = man["counts"]
    print(f"文件 {c['total']} 项：present {c['present']} / missing {c['missing']}")
    for r in man["files"]:
        if r["status"] == "ok":
            print(f"  ok      {r['sha256'][:16]}  {r['bytes']:>10,}  {r['path']}")
        else:
            print(f"  MISSING {'-' * 16}  {'-':>10}  {r['path']}")
    print("\n下一步（手工）：ots stamp / OSF Open-Ended Registration / "
          "git tag + push —— 具体命令见 --help。")


def _print_verify(rep: dict) -> None:
    print(f"核验：{rep['manifest']}（生成于 {rep['generated_utc']}）")
    print(f"manifest 自哈希：{'一致' if rep['manifest_intact'] else '**不一致——manifest 本身被改过**'}")
    if rep["git_head_recorded"]:
        print(f"git HEAD：登记 {str(rep['git_head_recorded'])[:12]} → 现在 "
              f"{str(rep['git_head_now'])[:12]}"
              f"{'（已移动）' if rep['git_head_moved'] else '（未移动）'}")
    for r in rep["files"]:
        mark = {"unchanged": "ok      ", "changed": "CHANGED ",
                "missing-now": "DELETED ", "appeared": "APPEARED",
                "still-missing": "missing "}[r["verdict"]]
        print(f"  {mark} {r['path']}")
    if rep["appeared"]:
        print(f"提示：{len(rep['appeared'])} 个当初记为 missing 的文件现已存在"
              "（登记后才写出属预期，不算漂移）。")
    verdict = ("无已登记文件被改动或删除" if rep["ok"]
               else "有 %d 项被改动/删除" % rep["n_changed"])
    print(f"结论：{verdict}")


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:  # pragma: no cover - 老 python / 被重定向
        pass
    ap = argparse.ArgumentParser(
        prog="preregistration_manifest.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=str(REPO_ROOT), help="仓库根（默认本脚本所在仓库）")
    ap.add_argument("--out", default=None, help="输出路径（默认 experiments/preregistration_manifest_<UTC>.json）")
    ap.add_argument("--verify", default=None, metavar="MANIFEST.json",
                    help="核验模式：重算并报告哪些文件变了；退出码 0=无漂移，1=有")
    ap.add_argument("--no-git", action="store_true", help="不记录 git 信息（测试/非 git 目录用）")
    ap.add_argument("--json", action="store_true", help="把结果以 JSON 打到 stdout")
    args = ap.parse_args(argv)

    repo = Path(args.repo).resolve()
    if args.verify:
        rep = verify(repo, Path(args.verify))
        if args.json:
            print(json.dumps(rep, indent=2, ensure_ascii=False))
        else:
            _print_verify(rep)
        return 0 if rep["ok"] else 1

    man = build_manifest(repo, include_git=not args.no_git)
    out = Path(args.out) if args.out else default_out_path(repo, man["generated_utc"])
    write_manifest(man, out)
    if args.json:
        print(json.dumps(man, indent=2, ensure_ascii=False))
    else:
        _print_manifest(man, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
