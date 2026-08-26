#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CRSP CIZ 数据下载器（命令行版，pipeline specification v1.2）

用法示例：
    python crsp_download.py --username YOUR_WRDS_ID
    python crsp_download.py --start 2000-01-01 --cut 2026-08-13 --out ./crsp_ciz_snapshots
    python crsp_download.py --check-only          # 只验证表和列，不下载
    python crsp_download.py --resume              # 接着上次没下完的继续

与原版 crsp_ciz_download_v1_2.py 的数据层差异（输出目录结构与 manifest 兼容）：
  1. secinfoenddt / securityenddt 不再用 pandas parse_dates 解析——CRSP 可能用
     9999-12-31 之类的远期哨兵日期，超出 pandas datetime64[ns] 上限会被静默转成
     NaT。现按原样以 date32 存入 parquet。
  2. security_info 的 WHERE 条件允许 secinfoenddt 为 NULL（NULL 表示“至今有效”，
     原版会把这些行整批丢掉）。
  3. 每个数据集的 parquet schema 由数据库列类型统一确定，各分块文件类型完全一致，
     避免“某块全为空值导致类型漂移”的问题。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import textwrap
import traceback
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

# 第三方依赖在 require_deps() 里导入，这样 --help 在没装依赖时也能用
pa = None
pq = None
wrds = None


def require_deps() -> None:
    global pa, pq, wrds
    try:
        import pyarrow as _pa
        import pyarrow.parquet as _pq
        import wrds as _wrds
    except ImportError as exc:
        sys.exit(
            f"缺少依赖: {exc.name}\n"
            f"请先安装:  {sys.executable} -m pip install -U wrds pandas pyarrow"
        )
    pa, pq, wrds = _pa, _pq, _wrds


# --resume 不带值时的占位符（表示“自动挑最近一个没跑完的”）
RESUME_AUTO = "__auto__"

CRSP_SCHEMA = "crsp"
CHUNK_SIZE = 250_000

DAILY_REQUIRED = [
    "permno",
    "dlycaldt",
    "dlydelflg",
    "dlyprc",
    "dlyprcflg",
    "dlycap",
    "dlyret",
    "dlyretmissflg",
    "dlyfacprc",
    "dlyvol",
    "dlyopen",
    "dlyhigh",
    "dlylow",
    "dlyclose",
    "dlyprcvol",
]

DAILY_OPTIONAL = [
    "hdrcusip",
    "permco",
    "siccd",
    "nasdissuno",
    "yyyymmdd",
    "dlycapflg",
    "dlyprevprc",
    "dlyprevprcflg",
    "dlyprevdt",
    "dlyprevcap",
    "dlyprevcapflg",
    "dlyretx",
    "dlyreti",
    "dlyretdurflg",
    "dlyorddivamt",
    "dlynonorddivamt",
    "dlydistretflg",
    "dlybid",
    "dlyask",
    "dlynumtrd",
    "dlymmcnt",
    "disfacpr",
    "disfacshr",
    "disexdt",
    # legacy 口径，仅作审计对照；市值一律用 dlycap（规范 §1），下游禁止取用
    "shrout",
    # 仅作为对事件表构建的复权序列的审计对照，绝不作为其来源
    "dlycumfacprc",
    "dlycumfacshr",
]

INDEX_REQUIRED = ["dlycaldt", "vwretd", "ewretd"]
INDEX_OPTIONAL = ["vwretx", "ewretx"]

# 这些列可能含 9999-12-31 之类的远期哨兵日期，不能交给 pandas 解析成 datetime64
SENTINEL_DATE_COLS = {"secinfoenddt", "securityenddt"}

# v1.2 规范对事件表的硬性要求列
EVENT_REQUIRED_COLUMNS = {
    "security_info": [
        "permno",
        "secinfostartdt",
        "secinfoenddt",
        "securitybegdt",
        "securityenddt",
        "sharetype",
        "securitytype",
        "securitysubtype",
        "usincflg",
        "issuertype",
        "primaryexch",
        "conditionaltype",
        "tradingstatusflg",
    ],
    "distributions": [
        "permno",
        "disexdt",
        "distype",
        "disdivamt",
        "disfacpr",
        "disfacshr",
    ],
    "delists": [
        "permno",
        "delistingdt",
        "delactiontype",
        "delstatustype",
        "delreasontype",
        "delpaymenttype",
        "delret",
        "delretmisstype",
    ],
}


def log(message: str = "") -> None:
    print(message, flush=True)


def parse_iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid ISO date: {value}") from exc


def safe_identifier(value: str) -> str:
    """只放行来自常量表 / WRDS 清单的合法标识符，防 SQL 注入。"""
    if not re.fullmatch(r"[a-z_][a-z0-9_]*", value):
        raise ValueError(f"unsafe SQL identifier: {value!r}")
    return value


def canonical_query_hash(sql: str, params: dict[str, Any]) -> str:
    payload = {
        "sql": textwrap.dedent(sql).strip(),
        "params": {key: str(value) for key, value in sorted(params.items())},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def get_table_inventory(db, schema: str) -> list[str]:
    return sorted(str(name).lower() for name in db.list_tables(library=schema))


def get_columns_with_types(db, schema: str, table: str) -> list[tuple[str, str]]:
    frame = db.raw_sql(
        """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = %(schema)s AND table_name = %(table)s
        ORDER BY ordinal_position
        """,
        params={"schema": schema, "table": table},
        chunksize=None,
    )
    return [
        (str(name).lower(), str(dtype).lower())
        for name, dtype in zip(frame["column_name"], frame["data_type"])
    ]


def resolve_table(inventory: Iterable[str], candidates: Iterable[str], label: str) -> str:
    available = set(inventory)
    for candidate in candidates:
        if candidate.lower() in available:
            return safe_identifier(candidate.lower())
    raise RuntimeError(
        f"找不到「{label}」。已尝试的表名: {list(candidates)}。"
        " 可能是 WRDS 改了表名，或账号权限不完整；"
        " 详见输出目录下的 metadata/table_inventory.json。"
    )


def select_columns(
    actual: Iterable[str], required: Iterable[str], optional: Iterable[str], label: str
) -> tuple[list[str], list[str]]:
    actual_set = set(actual)
    missing = [name for name in required if name not in actual_set]
    if missing:
        raise RuntimeError(f"表 {label} 缺少必需列: {missing}")
    selected = [name for name in [*required, *optional] if name in actual_set]
    omitted_optional = [name for name in optional if name not in actual_set]
    return selected, omitted_optional


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def arrow_type_for(pg_type: str, parsed_as_datetime: bool):
    t = pg_type.lower()
    if t == "date":
        return pa.timestamp("ns") if parsed_as_datetime else pa.date32()
    if t.startswith("timestamp"):
        return pa.timestamp("ns")
    if t in ("smallint", "integer", "bigint"):
        return pa.int64()
    if t in ("numeric", "double precision", "real"):
        return pa.float64()
    if t == "boolean":
        return pa.bool_()
    return pa.string()


def build_arrow_schema(
    colnames: list[str], pg_types: dict[str, str], parsed_date_cols: Iterable[str]
):
    parsed = set(parsed_date_cols)
    return pa.schema(
        pa.field(name, arrow_type_for(pg_types.get(name, ""), name in parsed))
        for name in colnames
    )


def chunk_to_arrow(chunk, schema):
    # 把每一块都强制转成同一套类型，各 parquet 分块 schema 才能完全一致
    for field in schema:
        name = field.name
        if pa.types.is_integer(field.type):
            chunk[name] = chunk[name].astype("Int64")
        elif pa.types.is_floating(field.type):
            chunk[name] = chunk[name].astype("float64")
        elif pa.types.is_string(field.type):
            chunk[name] = chunk[name].astype("string")
        elif pa.types.is_boolean(field.type):
            chunk[name] = chunk[name].astype("boolean")
        # 日期列不动: parse_dates 已产出 datetime64, 哨兵日期列保持原样交给 date32
    frame = chunk[[field.name for field in schema]]
    return pa.Table.from_pandas(frame, schema=schema, preserve_index=False)


def stream_query_to_parquet(
    db,
    *,
    dataset_name: str,
    label: str,
    sql: str,
    params: dict[str, Any],
    date_cols: list[str],
    schema,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    iterator = db.raw_sql(
        textwrap.dedent(sql).strip(),
        params=params,
        date_cols=date_cols,
        chunksize=CHUNK_SIZE,
        return_iter=True,
    )

    rows = 0
    parts: list[str] = []
    observed_min: dict[str, Any] = {name: None for name in date_cols}
    observed_max: dict[str, Any] = {name: None for name in date_cols}

    for part_number, chunk in enumerate(iterator):
        chunk.columns = [str(name).lower() for name in chunk.columns]
        table = chunk_to_arrow(chunk, schema)
        part_path = output_dir / f"part-{part_number:05d}.parquet"
        pq.write_table(table, part_path, compression="zstd")
        parts.append(part_path.name)
        rows += len(chunk)

        for name in date_cols:
            if name not in chunk or not chunk[name].notna().any():
                continue
            part_min = chunk[name].min()
            part_max = chunk[name].max()
            if observed_min[name] is None or part_min < observed_min[name]:
                observed_min[name] = part_min
            if observed_max[name] is None or part_max > observed_max[name]:
                observed_max[name] = part_max

        log(f"    {label}: {rows:,} 行")

    if rows == 0:
        log(f"    {label}: 0 行（该区间无数据）")

    result = {
        "dataset": dataset_name,
        "rows": rows,
        "parts": parts,
        "date_min": observed_min,
        "date_max": observed_max,
        "sql": textwrap.dedent(sql).strip(),
        "params": {key: str(value) for key, value in params.items()},
        "sql_sha256": canonical_query_hash(sql, params),
    }
    write_json(output_dir / "_query_manifest.json", result)
    return result


def iter_year_ranges(start: date, inclusive_cut: date):
    for year in range(start.year, inclusive_cut.year + 1):
        lower = max(start, date(year, 1, 1))
        upper = min(inclusive_cut + timedelta(days=1), date(year + 1, 1, 1))
        if lower < upper:
            yield year, lower, upper


def table_schema_signature(columns: list[str]) -> str:
    return hashlib.sha256("\n".join(columns).encode()).hexdigest()


# ---------------------------------------------------------------- 断点续传


def is_in_progress(snapshot_dir: Path) -> bool:
    """写了开工记录、但还没写完工记录 —— 说明是个没跑完的快照。"""
    metadata_dir = snapshot_dir / "metadata"
    return (metadata_dir / "snapshot_manifest.in_progress.json").is_file() and not (
        metadata_dir / "snapshot_manifest.json"
    ).is_file()


def find_resumable(out_root: Path) -> Path:
    candidates = sorted(
        path
        for path in out_root.glob("crsp_ciz_*")
        if path.is_dir() and is_in_progress(path)
    )
    if not candidates:
        raise RuntimeError(
            f"在 {out_root} 里没找到没跑完的快照，没有东西可以续传。"
            " 去掉 --resume 就是开一次全新的下载。"
        )
    # 目录名里带时间戳, 字典序最大的就是最近一次
    chosen = candidates[-1]
    if len(candidates) > 1:
        log(f"找到 {len(candidates)} 个没跑完的快照，选最近的一个。")
        log("（要指定别的就写 --resume 那个目录的路径）")
    return chosen


def load_resume_target(args, out_root: Path) -> Path:
    if args.resume == RESUME_AUTO:
        return find_resumable(out_root)
    snapshot_dir = Path(args.resume).expanduser().resolve()
    if not snapshot_dir.is_dir():
        raise RuntimeError(f"--resume 指向的目录不存在: {snapshot_dir}")
    if (snapshot_dir / "metadata" / "snapshot_manifest.json").is_file():
        raise RuntimeError(
            f"这个快照已经是完整的了，不需要续传: {snapshot_dir}"
        )
    if not (snapshot_dir / "metadata" / "snapshot_manifest.in_progress.json").is_file():
        raise RuntimeError(
            f"这不像一个下载到一半的快照（缺 metadata/snapshot_manifest.in_progress.json）: {snapshot_dir}"
        )
    return snapshot_dir


def check_resume_compatible(
    prior: dict[str, Any],
    prior_inventory: dict[str, Any],
    *,
    args,
    schema_signatures: dict[str, str],
    actual_max: Any,
) -> None:
    """续传前确认这次的条件和上次完全一样, 否则一个快照里会混进两批数据。"""
    for label, flag, was, now in [
        ("起始日期", "--start", prior.get("requested_start_date"), str(args.start)),
        ("截止日期", "--cut", prior.get("data_cut_date"), str(args.cut)),
    ]:
        if was != now:
            raise RuntimeError(
                f"续传失败: 这个快照的{label}是 {was}，这次跑的是 {now}。\n"
                f"  要么把 {flag} 改回 {was}，要么去掉 --resume 重新下一份。"
            )

    prior_signatures = prior_inventory.get("schema_sha256") or {}
    changed = [
        key
        for key, value in schema_signatures.items()
        if prior_signatures.get(key) != value
    ]
    if changed:
        raise RuntimeError(
            f"续传失败: 数据库里这些表的列结构变了 {changed}。\n"
            "  接着下会让同一个快照里前后两批数据对不上。请去掉 --resume 重新下一份。"
        )

    if prior.get("actual_max_trading_date") != str(actual_max):
        raise RuntimeError(
            f"续传失败: 上次看到的最新交易日是 {prior.get('actual_max_trading_date')}，"
            f"现在是 {actual_max}。\n"
            "  说明 CRSP 期间更新过数据，接着下会前后不一致。请去掉 --resume 重新下一份。"
        )


def load_or_download(db, *, label: str, output_dir: Path, **kwargs) -> dict[str, Any]:
    """已经下完的直接读记录跳过；上次写到一半的整份删掉重下。"""
    done_marker = output_dir / "_query_manifest.json"
    if done_marker.is_file():
        result = json.loads(done_marker.read_text(encoding="utf-8"))
        log(f"    {label}: 跳过，上次已下完（{result['rows']:,} 行）")
        return result
    if output_dir.exists():
        # 没有完工标记 = 上次写到一半就断了。part 文件的行顺序不保证可重现,
        # 不能接着往后写, 只能整份重来。
        shutil.rmtree(output_dir)
        log(f"    {label}: 上次没写完，删掉重下")
    return stream_query_to_parquet(db, label=label, output_dir=output_dir, **kwargs)


def connect(username: str | None):
    log("连接 WRDS ...")
    kwargs = {"wrds_username": username} if username else {}
    return wrds.Connection(**kwargs)


def run(args) -> int:
    out_root = args.out.expanduser().resolve()
    resume_dir = load_resume_target(args, out_root) if args.resume else None

    db = connect(args.username or os.getenv("WRDS_USERNAME"))
    total_rows = 0
    try:
        libraries = {str(name).lower() for name in db.list_libraries()}
        if CRSP_SCHEMA not in libraries:
            raise RuntimeError(
                "该 WRDS 账号看不到 crsp 库（订阅 / 权限问题，需要联系 WRDS 管理员）。"
            )

        crsp_inventory = get_table_inventory(db, CRSP_SCHEMA)
        tables = {
            "daily": resolve_table(
                crsp_inventory, ["dsf_v2", "stkdlysecuritydata"], "每日股票数据表"
            ),
            "security_info": resolve_table(
                crsp_inventory, ["stksecurityinfohist"], "证券属性历史表"
            ),
            "distributions": resolve_table(
                crsp_inventory, ["stkdistributions"], "分红派息事件表"
            ),
            "delists": resolve_table(crsp_inventory, ["stkdelists"], "退市事件表"),
            "index": resolve_table(
                crsp_inventory, ["wrds_dailyindexret_query"], "每日市场指数表"
            ),
        }

        table_columns: dict[str, list[str]] = {}
        table_types: dict[str, dict[str, str]] = {}
        for key, table in tables.items():
            cols = get_columns_with_types(db, CRSP_SCHEMA, table)
            table_columns[key] = [name for name, _ in cols]
            table_types[key] = dict(cols)

        daily_columns, daily_omitted = select_columns(
            table_columns["daily"], DAILY_REQUIRED, DAILY_OPTIONAL, tables["daily"]
        )
        index_columns, index_omitted = select_columns(
            table_columns["index"], INDEX_REQUIRED, INDEX_OPTIONAL, tables["index"]
        )
        for key, required in EVENT_REQUIRED_COLUMNS.items():
            missing = [name for name in required if name not in table_columns[key]]
            if missing:
                raise RuntimeError(f"表 {tables[key]} 缺少必需列: {missing}")

        schema_signatures = {
            key: table_schema_signature(value) for key, value in table_columns.items()
        }
        inventory_record = {
            "crsp_tables": crsp_inventory,
            "resolved_tables": tables,
            "columns": table_columns,
            "column_types": table_types,
            "daily_selected_columns": daily_columns,
            "daily_optional_columns_not_exposed": daily_omitted,
            "index_selected_columns": index_columns,
            "index_optional_columns_not_exposed": index_omitted,
            "schema_sha256": schema_signatures,
        }

        max_date_frame = db.raw_sql(
            f"""
            SELECT MAX(dlycaldt) AS actual_max_trading_date
            FROM {CRSP_SCHEMA}.{tables['daily']}
            WHERE dlycaldt <= %(cut)s
            """,
            params={"cut": args.cut},
            date_cols=["actual_max_trading_date"],
            chunksize=None,
        )
        actual_max = max_date_frame.iloc[0, 0]

        if resume_dir is not None:
            snapshot_dir = resume_dir
            metadata_dir = snapshot_dir / "metadata"
            prior = json.loads(
                (metadata_dir / "snapshot_manifest.in_progress.json").read_text(
                    encoding="utf-8"
                )
            )
            prior_inventory_path = metadata_dir / "table_inventory.json"
            prior_inventory = (
                json.loads(prior_inventory_path.read_text(encoding="utf-8"))
                if prior_inventory_path.is_file()
                else {}
            )
            check_resume_compatible(
                prior,
                prior_inventory,
                args=args,
                schema_signatures=schema_signatures,
                actual_max=actual_max,
            )
            snapshot_id = prior["snapshot_id"]
            # 快照的身份来自第一次开跑的时间, 续传不改它
            query_time = prior["query_timestamp_utc"]
            resumed_at = list(prior.get("resumed_at_utc") or [])
            resumed_at.append(datetime.now(timezone.utc))
            log(f"续传快照: {snapshot_dir}")
        else:
            query_time = datetime.now(timezone.utc)
            snapshot_id = f"crsp_ciz_{args.cut.isoformat()}_{query_time:%Y%m%dT%H%M%SZ}"
            snapshot_dir = out_root / snapshot_id
            metadata_dir = snapshot_dir / "metadata"
            metadata_dir.mkdir(parents=True, exist_ok=False)
            resumed_at = []

        write_json(metadata_dir / "table_inventory.json", inventory_record)

        snapshot_manifest: dict[str, Any] = {
            "snapshot_id": snapshot_id,
            "requested_start_date": args.start,
            "data_cut_date": args.cut,
            "actual_max_trading_date": actual_max,
            "query_timestamp_utc": query_time,
            "resumed_at_utc": resumed_at,
            "wrds_schema": CRSP_SCHEMA,
            "resolved_tables": tables,
            # WRDS PostgreSQL 不稳定暴露 CRSP release ID, 用 schema 哈希和
            # actual_max_trading_date 代替
            "wrds_release_id": None,
            "inventory_only": args.check_only,
            "datasets": [],
        }

        def record(result: dict[str, Any]) -> int:
            """记下一份下完的数据, 并把进度落盘, 这样断了之后知道进行到哪。"""
            snapshot_manifest["datasets"].append(result)
            write_json(
                metadata_dir / "snapshot_manifest.in_progress.json", snapshot_manifest
            )
            return result["rows"]

        write_json(metadata_dir / "snapshot_manifest.in_progress.json", snapshot_manifest)

        log(f"表和列检查通过。输出目录: {snapshot_dir}")
        log(f"截止日前最新交易日: {actual_max}")

        if args.check_only:
            snapshot_manifest["completed_timestamp_utc"] = datetime.now(timezone.utc)
            write_json(metadata_dir / "snapshot_manifest.json", snapshot_manifest)
            log("--check-only：已完成检查，未下载数据。")
            return 0

        year_ranges = list(iter_year_ranges(args.start, args.cut))
        log("")
        log(f"[1/3] 每日股票数据，共 {len(year_ranges)} 年")

        daily_date_cols = [
            name for name in ["dlycaldt", "dlyprevdt", "disexdt"] if name in daily_columns
        ]
        daily_schema = build_arrow_schema(
            daily_columns, table_types["daily"], daily_date_cols
        )
        daily_select = ",\n       ".join(daily_columns)
        for year, lower, upper in year_ranges:
            sql = f"""
                SELECT {daily_select}
                FROM {CRSP_SCHEMA}.{tables['daily']}
                WHERE dlycaldt >= %(start)s
                  AND dlycaldt <  %(end)s
            """
            total_rows += record(
                load_or_download(
                    db,
                    dataset_name=f"daily_{year}",
                    label=f"{year}",
                    sql=sql,
                    params={"start": lower, "end": upper},
                    date_cols=daily_date_cols,
                    schema=daily_schema,
                    output_dir=snapshot_dir / "raw" / "daily" / f"year={year}",
                )
            )

        log("")
        log("[2/3] 事件表（证券属性 / 分红 / 退市）")

        # 事件表保留全部列, 只按时间窗口过滤。
        # security_info: secinfoenddt 为 NULL 表示"至今有效", 必须保留;
        # 哨兵日期列(见 SENTINEL_DATE_COLS)不交给 pandas 解析, 原样存为 date32。
        event_jobs = [
            (
                "security_info",
                "security_info_history",
                "证券属性历史",
                f"""
                    SELECT {{cols}}
                    FROM {CRSP_SCHEMA}.{tables['security_info']}
                    WHERE (secinfoenddt >= %(start)s OR secinfoenddt IS NULL)
                      AND secinfostartdt < %(end)s
                """,
                ["secinfostartdt", "secinfoenddt", "securitybegdt", "securityenddt"],
                snapshot_dir / "raw" / "security_info_history",
            ),
            (
                "distributions",
                "distributions",
                "分红派息事件",
                f"""
                    SELECT {{cols}}
                    FROM {CRSP_SCHEMA}.{tables['distributions']}
                    WHERE disexdt >= %(start)s
                      AND disexdt < %(end)s
                """,
                ["disexdt", "disdeclaredt", "disrecorddt", "dispaydt"],
                snapshot_dir / "raw" / "distributions",
            ),
            (
                "delists",
                "delists",
                "退市事件",
                f"""
                    SELECT {{cols}}
                    FROM {CRSP_SCHEMA}.{tables['delists']}
                    WHERE delistingdt >= %(start)s
                      AND delistingdt < %(end)s
                """,
                ["delistingdt", "delnextdt", "delamtdt", "deldlydt"],
                snapshot_dir / "raw" / "delists",
            ),
        ]
        common_params = {"start": args.start, "end": args.cut + timedelta(days=1)}
        for key, name, label, sql_template, candidate_date_cols, output_dir in event_jobs:
            colnames = table_columns[key]
            parsed_date_cols = [
                value
                for value in candidate_date_cols
                if value in colnames and value not in SENTINEL_DATE_COLS
            ]
            schema = build_arrow_schema(colnames, table_types[key], parsed_date_cols)
            total_rows += record(
                load_or_download(
                    db,
                    dataset_name=name,
                    label=label,
                    sql=sql_template.format(cols=", ".join(colnames)),
                    params=common_params,
                    date_cols=parsed_date_cols,
                    schema=schema,
                    output_dir=output_dir,
                )
            )

        log("")
        log("[3/3] 市场指数")
        index_schema = build_arrow_schema(
            index_columns, table_types["index"], ["dlycaldt"]
        )
        index_result = load_or_download(
            db,
            dataset_name="daily_market_indexes",
            label="市场指数",
            sql=f"""
                SELECT {", ".join(index_columns)}
                FROM {CRSP_SCHEMA}.{tables['index']}
                WHERE dlycaldt >= %(start)s
                  AND dlycaldt <  %(end)s
            """,
            params=common_params,
            date_cols=["dlycaldt"],
            schema=index_schema,
            output_dir=snapshot_dir / "raw" / "daily_market_indexes",
        )
        total_rows += record(index_result)

        snapshot_manifest["completed_timestamp_utc"] = datetime.now(timezone.utc)
        write_json(metadata_dir / "snapshot_manifest.json", snapshot_manifest)

        log("")
        log(f"完成，共 {total_rows:,} 行 -> {snapshot_dir}")
        return 0
    finally:
        db.close()


def get_args():
    parser = argparse.ArgumentParser(
        description="CRSP CIZ 数据下载器（pipeline specification v1.2）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--start", type=parse_iso_date, default=date(2000, 1, 1), help="起始日期 YYYY-MM-DD"
    )
    parser.add_argument(
        "--cut",
        type=parse_iso_date,
        default=None,
        help="截止日期 YYYY-MM-DD（含），默认昨天",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("crsp_ciz_snapshots"), help="输出根目录"
    )
    parser.add_argument(
        "--username", default=None, help="WRDS 用户名，也可用环境变量 WRDS_USERNAME"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只验证表 / 列并写 metadata，不下载数据",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const=RESUME_AUTO,
        default=None,
        metavar="快照目录",
        help="接着上次没下完的继续。不带值 = 自动挑 --out 里最近一个没跑完的快照；"
        "也可以直接写那个快照目录的路径",
    )
    args = parser.parse_args()

    if args.cut is None:
        args.cut = date.today() - timedelta(days=1)
    if args.start > args.cut:
        parser.error("--start must not be after --cut")
    if args.resume and args.check_only:
        parser.error("--resume 和 --check-only 不能一起用")
    return args


def main() -> int:
    args = get_args()
    require_deps()
    try:
        return run(args)
    except KeyboardInterrupt:
        log("\n已中断，本次快照不完整。")
        return 130
    except RuntimeError as exc:
        log(f"\n错误: {exc}")
        return 1
    except Exception:
        log("")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
