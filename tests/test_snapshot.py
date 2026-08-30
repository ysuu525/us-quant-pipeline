"""快照原始层加载（Phase 5）：列名映射、hive 年分区、manifest 完工门禁。"""

import json

import pandas as pd
import pytest

from crsp_pipeline import snapshot as S


@pytest.fixture
def snap(tmp_path):
    """两年日线 + 指数表 + in_progress manifest 的最小快照。"""
    for year, dates in [(2020, pd.bdate_range("2020-01-06", periods=3)),
                        (2021, pd.bdate_range("2021-01-04", periods=3))]:
        part = tmp_path / "raw" / "daily" / f"year={year}"
        part.mkdir(parents=True)
        pd.DataFrame({
            "permno": [10001, 10002, 10001],
            "dlycaldt": [dates[1], dates[0], dates[0]],  # 乱序，验证排序
            "dlyopen": 10.0, "dlyhigh": 11.0, "dlylow": 9.0, "dlyclose": 10.5,
            "dlyprcflg": "TR", "dlyvol": 1000.0, "dlyprcvol": 10500.0,
            "dlyret": 0.01, "dlyretmissflg": None, "dlyfacprc": 1.0,
            "dlydelflg": "N", "dlycap": 5.0e6,
        }).to_parquet(part / "part-00000.parquet", index=False)
    idx_dir = tmp_path / "raw" / "daily_market_indexes"
    idx_dir.mkdir(parents=True)
    pd.DataFrame({
        "dlycaldt": pd.bdate_range("2020-01-06", periods=3)[::-1],  # 乱序
        "vwretd": [0.01, -0.02, 0.005],
    }).to_parquet(idx_dir / "part-00000.parquet", index=False)
    meta = tmp_path / "metadata"
    meta.mkdir()
    (meta / "snapshot_manifest.in_progress.json").write_text(
        json.dumps({"snapshot_id": "test_snap", "actual_max_trading_date": "2021-01-06"}),
        encoding="utf-8",
    )
    return tmp_path


def test_load_daily_renames_and_sorts(snap):
    df = S.load_daily(snap)
    assert list(df.columns) == list(S.PANEL_COLUMNS)
    # 按 (PERMNO, DlyCalDt) 排好
    assert df[["PERMNO", "DlyCalDt"]].equals(
        df[["PERMNO", "DlyCalDt"]].sort_values(["PERMNO", "DlyCalDt"], ignore_index=True))
    assert len(df) == 6


def test_load_daily_year_filter_and_columns(snap):
    df = S.load_daily(snap, columns=["PERMNO", "DlyCalDt", "DlyClose"], years=[2020])
    assert list(df.columns) == ["PERMNO", "DlyCalDt", "DlyClose"]
    assert df["DlyCalDt"].dt.year.eq(2020).all()


def test_load_daily_unknown_column_rejected(snap):
    with pytest.raises(S.SnapshotError):
        S.load_daily(snap, columns=["PERMNO", "NotAColumn"])


def test_manifest_in_progress_gated(snap):
    with pytest.raises(S.SnapshotError):
        S.load_manifest(snap)  # 未完工快照默认拒绝（§11）
    m = S.load_manifest(snap, allow_in_progress=True)
    assert m["snapshot_id"] == "test_snap"


def test_manifest_final_preferred(snap):
    (snap / "metadata" / "snapshot_manifest.json").write_text(
        json.dumps({"snapshot_id": "final"}), encoding="utf-8")
    assert S.load_manifest(snap)["snapshot_id"] == "final"


def test_load_market_index_renames_and_sorts(snap):
    idx = S.load_market_index(snap)
    assert "caldt" in idx.columns
    assert idx["caldt"].is_monotonic_increasing
