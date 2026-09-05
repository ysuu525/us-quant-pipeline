"""Compustat 季度宽表 -> 裁剪 parquet + 去重 + GVKEY/PERMNO 链接表（exp13 的数据准备层）。

本脚本只做 `docs/任务书_exp13_Compustat_开发折诊断_2026-09-05.md` §3.1–§3.3 三件事，
**不构造任何特征、不读任何折的分数、不读任何未来收益文件**。特征构造在
`scripts/exp13_compustat_dev_diag.py`。

任务书 §3.1 季度表筛选
----------------------
``indfmt=INDL, datafmt=STD, consol=C``（验收显示全表已如此，脚本仍显式过滤并断言）。
``fic`` **不筛**（CRSP 宇宙决定样本）；``curcdq`` 不筛，但报告 top500 内非 USD 占比
（占比在 exp13 里算，本脚本只保留 ``curcdq`` 列）。

任务书 §3.2 重复键
------------------
1999–2025 有 599 组 ``gvkey+datadate``，每组 2 行，``fyr`` 均不同。成因【推断，未核】
为财年变更时新旧财年口径并存。规则：**同组保留 ``fyearq`` 最大者；再平则保留 ``rdq``
非缺者；再平则 ``fqtr`` 最大者。** 零参数、确定性。交付时报告该规则在 top500 宇宙内
实际触及的股票-季度数（在 exp13 里算）。

任务书 §3.3 GVKEY -> PERMNO（CCM 缺失下的替代链接）
---------------------------------------------------
* 键：``fundq.cusip``（9 位，Compustat 当前 header CUSIP【推断，须查 PDF 核】）
  <-> ``security_info_history.hdrcusip9``（CRSP header CUSIP；【本项目实测 09-05】
  25,331 个 permno 全部非空、一一对应、无一对多）。
* 一个 gvkey <-> 多个 permno（多股份类别）：全部保留，季度数据广播到每个 permno；报告数量。
* 一个 permno <-> 多个 gvkey：视为链接冲突，**该 permno 整段置缺**并报告数量，不做挑选。
* 已知劣势（原样写进报告）：header 对 header 不是历史时点匹配；CUSIP 变更后两库 header
  可能不同步；退市多年的公司两库 header 可能指向不同证券。**若日后拿到
  ``ccmxpf_lnkhist``，本节整体替换，D1–D3 全部重跑并登记两版差异。**

工程约束（CLAUDE.md §七 / 任务书 §2.6）
---------------------------------------
季度 CSV 解压后 4.57 GB。必须 ``usecols`` + ``chunksize``，只转一次 parquet，
此后不再碰 CSV；转换峰值提交内存 < 8 GB。本脚本不用 GPU。
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RULE_VERSION = "exp13-rules-2026-09-05"  # 任务书 §3.1-§3.3，冻结哈希 f8d1ec2b...

DEFAULT_ZIP = Path(
    r"F:\quant\external\compustat\received_20260905\raw\quarterly\Comp_Quarterly6126.csv.zip"
)
DEFAULT_MEMBER = "nyisktqrssfruytt.csv"
DEFAULT_PROCESSED = Path(r"F:\quant\processed\crsp_ciz_2026-08-24_20260825T130601Z")
DEFAULT_DERIVED = Path(r"F:\quant\external\compustat\derived")

# 只保留 §4 所需列（epspxq / ajexq）+ 键 + rdq/fyearq/fqtr/fyr/datacqtr/datafqtr/cusip/curcdq
KEEP_COLUMNS: tuple[str, ...] = (
    "gvkey", "datadate", "rdq", "fyearq", "fqtr", "fyr",
    "datacqtr", "datafqtr", "cusip", "curcdq", "epspxq", "ajexq",
)
FORMAT_COLUMNS: tuple[str, ...] = ("indfmt", "datafmt", "consol")
REQUIRED_FORMAT = ("INDL", "STD", "C")
STRING_COLUMNS = ("gvkey", "cusip", "curcdq", "datacqtr", "datafqtr") + FORMAT_COLUMNS
NUMERIC_COLUMNS = ("fyearq", "fqtr", "fyr", "epspxq", "ajexq")
DATE_COLUMNS = ("datadate", "rdq")
CHUNK_ROWS = 500_000


def log(message: str) -> None:
    print(message, flush=True)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def committed_memory_gb() -> float:
    """Windows 提交内存；非 Windows 退回 /proc/meminfo（照 exp11:142-168）。"""
    if sys.platform == "win32":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        return float(status.ullTotalPageFile - status.ullAvailPageFile) / 2**30
    values: dict[str, float] = {}
    with Path("/proc/meminfo").open(encoding="ascii") as handle:
        for line in handle:
            key, value = line.split(":", 1)
            values[key] = float(value.strip().split()[0]) * 1024
    return values["Committed_AS"] / 2**30


def _coerce_chunk(chunk: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """§3.1 显式过滤 INDL/STD/C 并返回被丢弃的行数。"""
    mask = (
        (chunk["indfmt"] == REQUIRED_FORMAT[0])
        & (chunk["datafmt"] == REQUIRED_FORMAT[1])
        & (chunk["consol"] == REQUIRED_FORMAT[2])
    )
    dropped = int((~mask).sum())
    chunk = chunk.loc[mask].drop(columns=list(FORMAT_COLUMNS))
    for column in DATE_COLUMNS:
        chunk[column] = pd.to_datetime(chunk[column], format="%Y-%m-%d", errors="coerce")
    for column in NUMERIC_COLUMNS:
        chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
    for column in ("gvkey", "cusip", "curcdq", "datacqtr", "datafqtr"):
        chunk[column] = chunk[column].astype("string").str.strip()
    return chunk.reset_index(drop=True), dropped


def read_quarterly(zip_path: Path, member: str, *, chunk_rows: int = CHUNK_ROWS,
                   max_chunks: int | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """usecols + chunksize 读季度宽表；只保留 KEEP_COLUMNS。"""
    frames: list[pd.DataFrame] = []
    rows_in = 0
    dropped_format = 0
    peak_committed = committed_memory_gb()
    with zipfile.ZipFile(zip_path) as archive:
        names = [item.filename for item in archive.infolist()]
        if member not in names:
            raise FileNotFoundError(f"ZIP 内无成员 {member!r}；实际成员：{names}")
        with archive.open(member) as handle:
            text = io.TextIOWrapper(handle, encoding="utf-8", newline="")
            reader = pd.read_csv(
                text,
                usecols=list(KEEP_COLUMNS + FORMAT_COLUMNS),
                dtype={column: "string" for column in STRING_COLUMNS},
                chunksize=chunk_rows,
                na_values=["", "."],
                keep_default_na=True,
                low_memory=False,
            )
            for index, chunk in enumerate(reader):
                rows_in += len(chunk)
                coerced, dropped = _coerce_chunk(chunk)
                dropped_format += dropped
                frames.append(coerced)
                peak_committed = max(peak_committed, committed_memory_gb())
                if (index + 1) % 5 == 0:
                    log(f"  已读 {rows_in:,} 行；committed={peak_committed:.2f} GB")
                if max_chunks is not None and index + 1 >= max_chunks:
                    break
    frame = pd.concat(frames, ignore_index=True)
    del frames
    stats = {
        "rows_in_csv": rows_in,
        "rows_dropped_format_filter": dropped_format,
        "rows_after_format_filter": int(len(frame)),
        "peak_committed_gb_during_read": round(peak_committed, 3),
    }
    return frame, stats


def deduplicate(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """§3.2：同 (gvkey, datadate) 组保留 fyearq 最大 -> rdq 非缺 -> fqtr 最大。"""
    work = frame.copy()
    work["_rdq_present"] = work["rdq"].notna().astype("int8")
    work["_orig_order"] = np.arange(len(work), dtype=np.int64)
    # 诊断列：该 (gvkey, datadate) 原始组的行数。任务书 §6 要求报告去重规则在 top500
    # 宇宙内实际触及的股票-季度数，只有把组大小随行保留才算得出来。
    work["dedup_group_rows"] = (
        work.groupby(["gvkey", "datadate"], dropna=False)["_orig_order"]
        .transform("size").astype("int16")
    )
    duplicated_mask = work.duplicated(["gvkey", "datadate"], keep=False)
    groups_total = int(
        work.loc[duplicated_mask, ["gvkey", "datadate"]].drop_duplicates().shape[0]
    )
    in_window = duplicated_mask & work["datadate"].between("1999-01-01", "2025-12-31")
    groups_1999_2025 = int(
        work.loc[in_window, ["gvkey", "datadate"]].drop_duplicates().shape[0]
    )
    ordered = work.sort_values(
        ["gvkey", "datadate", "fyearq", "_rdq_present", "fqtr", "_orig_order"],
        ascending=[True, True, False, False, False, True],
        kind="mergesort",
        na_position="last",
    )
    # 残余平局：三个键全相等时只能落回文件原序（稳定排序），必须披露。
    tie_keys = ["gvkey", "datadate", "fyearq", "_rdq_present", "fqtr"]
    tie_mask = ordered.duplicated(tie_keys, keep=False)
    unresolved_ties = int(
        ordered.loc[tie_mask, ["gvkey", "datadate"]].drop_duplicates().shape[0]
    )
    kept = ordered.drop_duplicates(["gvkey", "datadate"], keep="first")
    dropped_rows = int(len(work) - len(kept))
    kept = (
        kept.drop(columns=["_rdq_present", "_orig_order"])
        .sort_values(["gvkey", "datadate"], kind="mergesort")
        .reset_index(drop=True)
    )
    if kept.duplicated(["gvkey", "datadate"]).any():
        raise AssertionError("§3.2 去重后仍有重复 (gvkey, datadate)")
    stats = {
        "rule": "fyearq max -> rdq non-missing -> fqtr max (stable, file order as last resort)",
        "duplicate_groups_full_table": groups_total,
        "duplicate_groups_1999_2025": groups_1999_2025,
        "rows_dropped_by_dedup": dropped_rows,
        "unresolved_tie_groups_full_table": unresolved_ties,
        "rows_after_dedup": int(len(kept)),
    }
    return kept, stats


def _normalise_cusip9(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip().str.upper()
    ok = cleaned.str.len() == 9
    return cleaned.where(ok)


def build_link(fundq: pd.DataFrame, processed: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """§3.3：fundq.cusip(9) <-> security_info_history.hdrcusip9。"""
    sec_path = processed / "security_info_history.parquet"
    if not sec_path.is_file():
        raise FileNotFoundError(f"缺 {sec_path}")
    sec = pd.read_parquet(sec_path, columns=["permno", "hdrcusip9"])
    sec["permno"] = pd.to_numeric(sec["permno"], errors="raise").astype("int64")
    sec["cusip9"] = _normalise_cusip9(sec["hdrcusip9"])
    sec_pairs = sec.dropna(subset=["cusip9"])[["permno", "cusip9"]].drop_duplicates()
    permno_multi_hdr = int(
        (sec_pairs.groupby("permno")["cusip9"].nunique() > 1).sum()
    )

    fundq = fundq.copy()
    fundq["cusip9"] = _normalise_cusip9(fundq["cusip"])
    bad_cusip_rows = int(fundq["cusip9"].isna().sum())
    gvkey_pairs = (
        fundq.dropna(subset=["cusip9"])[["gvkey", "cusip9"]].drop_duplicates()
    )
    gvkey_multi_cusip = int((gvkey_pairs.groupby("gvkey")["cusip9"].nunique() > 1).sum())

    merged = gvkey_pairs.merge(sec_pairs, on="cusip9", how="inner")
    gvkeys_linked = int(merged["gvkey"].nunique())
    permnos_linked = int(merged["permno"].nunique())
    gvkey_to_many_permno = int((merged.groupby("gvkey")["permno"].nunique() > 1).sum())
    permno_gvkey_counts = merged.groupby("permno")["gvkey"].nunique()
    conflict_permnos = permno_gvkey_counts.index[permno_gvkey_counts > 1]
    permno_to_many_gvkey = int(len(conflict_permnos))
    clean = merged[~merged["permno"].isin(conflict_permnos)].copy()
    clean = clean.sort_values(["gvkey", "permno"], kind="mergesort").reset_index(drop=True)
    if clean.duplicated(["gvkey", "permno"]).any():
        raise AssertionError("链接表存在重复 (gvkey, permno)")
    if clean.groupby("permno")["gvkey"].nunique().gt(1).any():
        raise AssertionError("§3.3 冲突 permno 未被剔除")
    stats = {
        "security_info_history_rows": int(len(sec)),
        "security_info_history_permnos": int(sec["permno"].nunique()),
        "security_info_history_hdrcusip9_null": int(sec["hdrcusip9"].isna().sum()),
        "permno_with_multiple_hdrcusip9": permno_multi_hdr,
        "fundq_rows_with_unusable_cusip": bad_cusip_rows,
        "gvkey_with_multiple_cusip9": gvkey_multi_cusip,
        "gvkeys_linked": gvkeys_linked,
        "permnos_linked_before_conflict_drop": permnos_linked,
        "gvkey_to_many_permno": gvkey_to_many_permno,
        "permno_to_many_gvkey_dropped": permno_to_many_gvkey,
        "link_rows_final": int(len(clean)),
        "permnos_final": int(clean["permno"].nunique()),
        "gvkeys_final": int(clean["gvkey"].nunique()),
        "known_weaknesses": (
            "header 对 header 不是历史时点匹配；CUSIP 变更后两库 header 可能不同步；"
            "退市多年的公司两库 header 可能指向不同证券。拿到 crsp.ccmxpf_lnkhist 后"
            "本节整体替换，D1-D3 全部重跑并登记两版差异。"
        ),
    }
    return clean, stats


def run(zip_path: Path, member: str, processed: Path, derived: Path,
        *, chunk_rows: int = CHUNK_ROWS, max_chunks: int | None = None,
        skip_source_hash: bool = False) -> dict[str, Any]:
    start_committed = committed_memory_gb()
    log(f"[build_link] 起始 committed={start_committed:.2f} GB")
    derived.mkdir(parents=True, exist_ok=True)
    log(f"[build_link] 读 {zip_path}::{member}（usecols+chunksize={chunk_rows}）")
    frame, read_stats = read_quarterly(zip_path, member, chunk_rows=chunk_rows,
                                       max_chunks=max_chunks)
    log(f"[build_link] 格式过滤后 {len(frame):,} 行")
    slim, dedup_stats = deduplicate(frame)
    del frame
    log(f"[build_link] 去重后 {len(slim):,} 行"
        f"（重复组 {dedup_stats['duplicate_groups_full_table']}，"
        f"1999-2025 {dedup_stats['duplicate_groups_1999_2025']}）")
    slim_path = derived / "fundq_slim.parquet"
    slim.to_parquet(slim_path, index=False)
    link, link_stats = build_link(slim, processed)
    link_path = derived / "gvkey_permno_link.parquet"
    link.to_parquet(link_path, index=False)
    end_committed = committed_memory_gb()
    manifest = {
        "rule_version": RULE_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task_doc": "docs/任务书_exp13_Compustat_开发折诊断_2026-09-05.md",
        "task_doc_sha256": "f8d1ec2bc1be4627fbcb5d5ccddab739eb0c17d59aa0d1838bf6e898f3e1acd9",
        "source_zip": str(zip_path),
        "source_zip_sha256": (None if skip_source_hash else sha256_file(zip_path)),
        "source_member": member,
        "generator_script": str(Path(__file__).resolve()),
        "generator_script_sha256": sha256_file(Path(__file__).resolve()),
        "python": sys.version,
        "pandas": pd.__version__,
        "columns_kept": list(KEEP_COLUMNS) + ["dedup_group_rows"],
        "columns_kept_note": (
            "dedup_group_rows 是本脚本加的诊断列（原 (gvkey,datadate) 组的行数），"
            "任务书 §6 的『去重规则触及的股票-季度数』需要它；不是 Compustat 字段。"
        ),
        "format_filter": dict(zip(FORMAT_COLUMNS, REQUIRED_FORMAT)),
        "read": read_stats,
        "dedup": dedup_stats,
        "link": link_stats,
        "outputs": {
            "fundq_slim": str(slim_path),
            "gvkey_permno_link": str(link_path),
            "fundq_slim_sha256": sha256_file(slim_path),
            "gvkey_permno_link_sha256": sha256_file(link_path),
        },
        "committed_memory_gb": {
            "start": round(start_committed, 3),
            "peak_during_read": read_stats["peak_committed_gb_during_read"],
            "end": round(end_committed, 3),
        },
        "not_in_git": True,
    }
    (derived / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"[build_link] 写出 {slim_path}")
    log(f"[build_link] 写出 {link_path}")
    log(f"[build_link] 写出 {derived / 'MANIFEST.json'}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compustat 季度表 -> 裁剪 parquet + 链接表")
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--member", default=DEFAULT_MEMBER)
    parser.add_argument("--processed", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--derived", type=Path, default=DEFAULT_DERIVED)
    parser.add_argument("--chunk-rows", type=int, default=CHUNK_ROWS)
    parser.add_argument("--max-chunks", type=int, default=None,
                        help="仅冒烟用：只读前 N 块")
    parser.add_argument("--skip-source-hash", action="store_true",
                        help="仅冒烟用：跳过源 ZIP 的 sha256（749 MB）")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run(args.zip, args.member, args.processed, args.derived,
                   chunk_rows=args.chunk_rows, max_chunks=args.max_chunks,
                   skip_source_hash=args.skip_source_hash)
    print(json.dumps(
        {"dedup": manifest["dedup"], "link": manifest["link"],
         "read": manifest["read"]},
        ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
