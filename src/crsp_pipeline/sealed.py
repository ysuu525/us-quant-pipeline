"""封存产物的哨兵、清单与读取守卫（2026-09-02 授权口径）。

授权范围（用户 2026-09-02，写得很窄）：
  允许 —— FT 固定配置训练；FT/ZS 生成预测分数。
  不允许 —— 生成或读取验证窗 labels；计算任何 IC / 收益 / 分层 / 年度 / 组合指标；
            人工打开 scores。
  **本次计算不视为消耗确认集；任何结果读取仍须另行授权。**

因此封存目录里**只有** ``scores.parquet``、``SEALED_MANIFEST.json``、
``SEALED``（哨兵）与封存日志。**不写 labels.parquet** —— labels 是解封时才需要的，
提前生成只增加误触风险。

守卫：任何分析脚本读取路径前调用 :func:`assert_readable`。沿路径向上找哨兵，
找到即拒绝，除非显式 ``unseal=True``（未来的解封授权才允许传）。
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

SENTINEL = "SEALED"
MANIFEST = "SEALED_MANIFEST.json"
SEALED_LOG = "sealed_run.log"
ALLOWED_FILES = {SENTINEL, MANIFEST, SEALED_LOG, "scores.parquet"}

FORBIDDEN_FILES = ("labels.parquet", "metrics.json", "daily_ic.parquet", "report.md")


class SealedReadError(RuntimeError):
    """试图读取封存目录而未取得解封授权。"""


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while blk := f.read(chunk):
            h.update(blk)
    return h.hexdigest()


def find_sentinel(path: Path) -> Path | None:
    """沿 path 向上（含自身若为目录）寻找哨兵文件。"""
    p = Path(path).resolve()
    for cand in [p, *p.parents]:
        if not cand.is_dir():
            continue
        if (cand / SENTINEL).exists():
            return cand / SENTINEL
        # 不越过仓库根
        if (cand / "CLAUDE.md").exists():
            break
    return None


def assert_readable(path, unseal: bool = False) -> None:
    """读取守卫。分析脚本在打开任何 parquet 之前调用。"""
    if unseal:
        return
    hit = find_sentinel(Path(path))
    if hit is not None:
        raise SealedReadError(
            f"拒绝读取：{path} 位于封存目录 {hit.parent} 之下。\n"
            f"该目录由 2026-09-02 的『计算授权 != 读取授权』裁定封存。\n"
            f"读取结果需要用户另行授权；获授权后显式传 unseal=True。"
        )


def write_seal(out_dir: Path, manifest: dict) -> None:
    """写哨兵与清单。调用方必须已经确保目录内没有任何禁止产物。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for bad in FORBIDDEN_FILES:
        if (out_dir / bad).exists():
            raise RuntimeError(
                f"封存失败：{out_dir / bad} 不应存在。封存模式绝不产出该文件。")
    manifest = dict(manifest)
    manifest["sealed_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    # 调用方可以带上自己那次授权的原文（如树基线的 2026-09-05 计算专用授权）；
    # 不带时沿用 Kronos 队列的 2026-09-02 口径。
    manifest.setdefault(
        "authorisation",
        "2026-09-02 用户授权：仅允许 FT 固定配置训练与 FT/ZS 打分；"
        "不得生成/读取 labels，不得计算任何 IC/收益/分层/年度/组合指标，"
        "不得人工打开 scores。计算不视为消耗确认集；读取须另行授权。",
    )
    (out_dir / MANIFEST).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    (out_dir / SENTINEL).write_text(
        "SEALED — 计算授权 != 读取授权（2026-09-02）\n"
        "本目录仅含模型分数。禁止读取、绘图、统计或并入任何分析，\n"
        "直到用户就『读取折 05-35 / 2024-07 起封存窗的结果』另行授权。\n"
        "守卫：crsp_pipeline.sealed.assert_readable()\n",
        encoding="utf-8")


def audit_dir(out_dir: Path, extra_allowed: set[str] | None = None) -> dict:
    """列目录并断言只有允许的文件。冒烟与自动测试用。

    ``extra_allowed`` 供别的封存生产者（如树基线：三个 seed 分数文件与内层
    选参记录）声明它自己的白名单；不传时行为与旧版逐字节一致。
    """
    out_dir = Path(out_dir)
    names = {p.name for p in out_dir.iterdir()}
    allowed = ALLOWED_FILES | set(extra_allowed or ())
    extra = names - allowed
    missing = {SENTINEL, MANIFEST, "scores.parquet"} - names
    forbidden = names & set(FORBIDDEN_FILES)
    return {"dir": str(out_dir), "files": sorted(names),
            "unexpected": sorted(extra), "missing": sorted(missing),
            "forbidden_present": sorted(forbidden),
            "clean": not extra and not missing and not forbidden}
