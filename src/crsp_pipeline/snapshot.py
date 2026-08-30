"""快照原始层加载（Phase 5 数据准备，规范 §1 / §11）。

下载器（downloader/crsp_download.py）产出的快照目录：

    <snapshot>/raw/daily/year=YYYY/part-*.parquet     每日行情（hive 分区）
    <snapshot>/raw/security_info_history/             证券属性历史
    <snapshot>/raw/distributions/                     分红派息事件
    <snapshot>/raw/delists/                           退市事件
    <snapshot>/raw/daily_market_indexes/              市场指数收益率
    <snapshot>/metadata/snapshot_manifest.json        完工记录（§11 引用它的 snapshot_id）

本模块只做「读 + 改列名」：原始层零变换（§1），列名从数据库小写映射到
规范 §1 的 CIZ 官方大小写（PERMNO / DlyCalDt / DlyOpen ...），数值不动。
复权、清洗、universe 归各自模块。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow.dataset as ds

# 数据库小写列名 -> 规范 §1 官方大小写。只列管线实际消费的列；
# 其余列按需用 extra_columns 原名透传。
DAILY_RAW_TO_SPEC: dict[str, str] = {
    "permno": "PERMNO",
    "dlycaldt": "DlyCalDt",
    "dlyopen": "DlyOpen",
    "dlyhigh": "DlyHigh",
    "dlylow": "DlyLow",
    "dlyclose": "DlyClose",
    "dlyprcflg": "DlyPrcFlg",
    "dlyvol": "DlyVol",
    "dlyprcvol": "DlyPrcVol",
    "dlyret": "DlyRet",
    "dlyretmissflg": "DlyRetMissFlg",
    "dlyfacprc": "DlyFacPrc",
    "dlydelflg": "DlyDelFlg",
    "dlycap": "DlyCap",
}

# 未过滤全量面板（§3 收益面板兼训练蜡烛原料）默认携带的列。
PANEL_COLUMNS: tuple[str, ...] = tuple(DAILY_RAW_TO_SPEC.values())


class SnapshotError(RuntimeError):
    pass


def load_manifest(snapshot_dir: str | Path, allow_in_progress: bool = False) -> dict:
    """读快照 manifest。默认只接受完工快照（§11：实验引用完整快照 ID）。

    allow_in_progress=True 允许读 in_progress manifest——仅限数据核对/开发，
    正式训练前必须用 downloader 的 --resume 把快照补完。
    """
    metadata = Path(snapshot_dir) / "metadata"
    final = metadata / "snapshot_manifest.json"
    if final.is_file():
        return json.loads(final.read_text(encoding="utf-8"))
    partial = metadata / "snapshot_manifest.in_progress.json"
    if partial.is_file():
        if not allow_in_progress:
            raise SnapshotError(
                f"快照未完工（只有 in_progress manifest）: {snapshot_dir}\n"
                "  用 downloader 的 --resume 补完，或显式传 allow_in_progress=True"
                "（仅限开发/核对）。"
            )
        return json.loads(partial.read_text(encoding="utf-8"))
    raise SnapshotError(f"不是快照目录（缺 metadata/snapshot_manifest*.json）: {snapshot_dir}")


def load_daily(
    snapshot_dir: str | Path,
    columns: Iterable[str] | None = None,
    years: Iterable[int] | None = None,
    extra_columns: Iterable[str] = (),
) -> pd.DataFrame:
    """读每日行情，返回规范命名的 DataFrame，按 (PERMNO, DlyCalDt) 排序。

    columns : 规范名（PANEL_COLUMNS 的子集）；None = 全部 PANEL_COLUMNS。
    years : 只读这些年份分区；None = 全部。
    extra_columns : 额外透传的原始列（保持数据库小写原名，如 'siccd'）。
    """
    root = Path(snapshot_dir) / "raw" / "daily"
    if not root.is_dir():
        raise SnapshotError(f"快照缺 raw/daily: {snapshot_dir}")
    spec_cols = list(columns) if columns is not None else list(PANEL_COLUMNS)
    spec_to_raw = {v: k for k, v in DAILY_RAW_TO_SPEC.items()}
    unknown = [c for c in spec_cols if c not in spec_to_raw]
    if unknown:
        raise SnapshotError(f"未知面板列（不在 DAILY_RAW_TO_SPEC 中）: {unknown}")
    raw_cols = [spec_to_raw[c] for c in spec_cols] + list(extra_columns)

    dataset = ds.dataset(root, format="parquet", partitioning="hive")
    filt = ds.field("year").isin(list(years)) if years is not None else None
    table = dataset.to_table(columns=raw_cols, filter=filt)
    df = table.to_pandas()
    df = df.rename(columns=DAILY_RAW_TO_SPEC)
    df["DlyCalDt"] = pd.to_datetime(df["DlyCalDt"])
    return df.sort_values(["PERMNO", "DlyCalDt"], ignore_index=True)


def load_table(snapshot_dir: str | Path, name: str) -> pd.DataFrame:
    """读事件表 / 指数表（列名保持数据库小写原名，universe.py 等按小写消费）。

    name ∈ {security_info_history, distributions, delists, daily_market_indexes}
    """
    root = Path(snapshot_dir) / "raw" / name
    if not root.is_dir():
        raise SnapshotError(f"快照缺 raw/{name}: {snapshot_dir}")
    return ds.dataset(root, format="parquet").to_table().to_pandas()


def load_market_index(snapshot_dir: str | Path) -> pd.DataFrame:
    """市场指数表 → (caldt, vwretd, ewretd, vwretx, ewretx)，按日排序。

    列名 dlycaldt 改为 caldt：TradingCalendar.from_market_index 与
    kronos_ft.train 的 --index-parquet 都以 caldt 为日期列。
    """
    df = load_table(snapshot_dir, "daily_market_indexes")
    df = df.rename(columns={"dlycaldt": "caldt"})
    df["caldt"] = pd.to_datetime(df["caldt"])
    return df.sort_values("caldt", ignore_index=True)
