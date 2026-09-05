"""exp13 数据准备层：§3.1 格式过滤、§3.2 去重确定性、§3.3 链接一对一、无未来收益变量名。"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILD = _load("exp13_build_link", "scripts/build_compustat_link.py")


def _row(gvkey, datadate, fyearq, fqtr, rdq=None, fyr=12, eps=1.0):
    return {
        "gvkey": gvkey, "datadate": pd.Timestamp(datadate), "rdq": rdq,
        "fyearq": float(fyearq), "fqtr": float(fqtr), "fyr": float(fyr),
        "datacqtr": f"{fyearq}Q{fqtr}", "datafqtr": f"{fyearq}Q{fqtr}",
        "cusip": "12345678" + "9", "curcdq": "USD", "epspxq": eps, "ajexq": 1.0,
    }


def test_dedup_keeps_max_fyearq_then_rdq_present_then_max_fqtr():
    frame = pd.DataFrame([
        # 组 A：fyearq 决胜（2021 胜 2020），即使 2020 那行有 rdq
        _row("A", "2021-03-31", 2020, 4, rdq=pd.Timestamp("2021-05-01"), eps=1.0),
        _row("A", "2021-03-31", 2021, 1, rdq=None, eps=2.0),
        # 组 B：fyearq 相同，rdq 非缺决胜
        _row("B", "2021-03-31", 2021, 1, rdq=None, eps=3.0),
        _row("B", "2021-03-31", 2021, 2, rdq=pd.Timestamp("2021-05-01"), eps=4.0),
        # 组 C：fyearq 与 rdq 状态相同，fqtr 最大决胜
        _row("C", "2021-03-31", 2021, 1, rdq=pd.Timestamp("2021-05-01"), eps=5.0),
        _row("C", "2021-03-31", 2021, 3, rdq=pd.Timestamp("2021-05-02"), eps=6.0),
    ])
    kept, stats = BUILD.deduplicate(frame)
    got = dict(zip(kept["gvkey"], kept["epspxq"]))
    assert got == {"A": 2.0, "B": 4.0, "C": 6.0}
    assert stats["duplicate_groups_full_table"] == 3
    assert stats["rows_dropped_by_dedup"] == 3
    assert stats["unresolved_tie_groups_full_table"] == 0
    assert (kept["dedup_group_rows"] == 2).all()


def test_dedup_is_order_invariant_and_flags_unresolved_ties():
    frame = pd.DataFrame([
        _row("A", "2021-03-31", 2021, 1, rdq=pd.Timestamp("2021-05-01"), eps=1.0),
        _row("A", "2021-03-31", 2021, 1, rdq=pd.Timestamp("2021-05-02"), eps=2.0),
    ])
    kept, stats = BUILD.deduplicate(frame)
    # 三个键全平：规则本身分不出，必须被计入 unresolved_ties 并如实披露
    assert stats["unresolved_tie_groups_full_table"] == 1
    assert len(kept) == 1
    # 而非平局的情形与输入行序无关
    good = pd.DataFrame([
        _row("B", "2021-03-31", 2020, 4, eps=1.0),
        _row("B", "2021-03-31", 2021, 1, eps=2.0),
    ])
    forward, _ = BUILD.deduplicate(good)
    backward, _ = BUILD.deduplicate(good.iloc[::-1].reset_index(drop=True))
    assert float(forward["epspxq"].iloc[0]) == float(backward["epspxq"].iloc[0]) == 2.0


def test_format_filter_drops_non_indl_std_c():
    chunk = pd.DataFrame([
        {**_row("A", "2021-03-31", 2021, 1), "indfmt": "INDL", "datafmt": "STD", "consol": "C"},
        {**_row("B", "2021-03-31", 2021, 1), "indfmt": "FS", "datafmt": "STD", "consol": "C"},
        {**_row("C", "2021-03-31", 2021, 1), "indfmt": "INDL", "datafmt": "SUMM_STD", "consol": "C"},
    ])
    kept, dropped = BUILD._coerce_chunk(chunk)
    assert dropped == 2
    assert list(kept["gvkey"]) == ["A"]
    assert not set(BUILD.FORMAT_COLUMNS) & set(kept.columns)


def test_link_is_one_to_one_and_drops_conflicting_permnos(tmp_path):
    fundq = pd.DataFrame([
        {"gvkey": "A", "cusip": "111111111"},
        {"gvkey": "B", "cusip": "222222222"},
        {"gvkey": "C", "cusip": "222222222"},   # 两个 gvkey 指向同一 permno -> 冲突
        {"gvkey": "D", "cusip": "333333333"},   # CRSP 里没有 -> 落空
        {"gvkey": "E", "cusip": "short"},       # 非 9 位 -> 不可用
    ])
    processed = tmp_path / "processed"
    processed.mkdir()
    pd.DataFrame({
        "permno": [10001, 10002, 10003],
        "hdrcusip9": ["111111111", "222222222", "444444444"],
    }).to_parquet(processed / "security_info_history.parquet", index=False)

    link, stats = BUILD.build_link(fundq, processed)
    assert list(link["gvkey"]) == ["A"]
    assert list(link["permno"]) == [10001]
    assert stats["permno_to_many_gvkey_dropped"] == 1
    assert stats["fundq_rows_with_unusable_cusip"] == 1
    assert not link.duplicated(["gvkey", "permno"]).any()
    assert link.groupby("permno")["gvkey"].nunique().le(1).all()


def test_gvkey_with_multiple_permnos_is_broadcast_not_dropped(tmp_path):
    fundq = pd.DataFrame([{"gvkey": "A", "cusip": "111111111"}])
    processed = tmp_path / "processed"
    processed.mkdir()
    pd.DataFrame({
        "permno": [10001, 10002],
        "hdrcusip9": ["111111111", "111111111"],
    }).to_parquet(processed / "security_info_history.parquet", index=False)
    link, stats = BUILD.build_link(fundq, processed)
    assert sorted(link["permno"]) == [10001, 10002]
    assert stats["gvkey_to_many_permno"] == 1
    assert stats["permno_to_many_gvkey_dropped"] == 0


@pytest.mark.parametrize("relative", [
    "scripts/build_compustat_link.py",
    "scripts/exp13_compustat_dev_diag.py",
    "scripts/exp13_smoke.py",
])
def test_scripts_contain_no_lookahead_target_variable(relative):
    """CLAUDE.md §一.1：未来收益只能作目标或结果，绝不能进特征。"""
    text = (REPO / relative).read_text(encoding="utf-8")
    forbidden = "lab" + "el"
    assert not re.search(forbidden, text), f"{relative} 出现被禁的目标变量名"
