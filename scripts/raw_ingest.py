from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional

_log = logging.getLogger("gpdm.raw_ingest")

_EXT_TEXT = (".csv", ".tsv", ".psv", ".txt", ".dat")
_EXT_EXCEL = (".xlsx", ".xlsm", ".xls")
_EXT_JSON = (".json", ".ndjson", ".jsonl")
_EXT_PARQUET = (".parquet",)

_FAST_BYTES = int(os.environ.get("GPDM_FAST_BYTES_THRESHOLD", str(50 * 1024 * 1024)))

_SQLITE_RESERVED_NAMES = frozenset({
    "sqlite_master", "sqlite_sequence", "sqlite_stat1", "sqlite_stat2",
    "sqlite_stat3", "sqlite_stat4", "sqlite_temp_master",
    "sqlite_schema", "sqlite_temp_schema",
})


def _sanitize_user_table_name(name: str) -> Optional[str]:
    import re as _re
    if not name:
        return None
    c = _re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_")
    if not c:
        return None
    if not _re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", c):
        c = "_" + c
    c = c[:60]
    if c.lower() in _SQLITE_RESERVED_NAMES or c.lower().startswith("sqlite_"):
        _log.warning("Filename '%s' maps to reserved SQLite name; "
                     "rerouting to 'usr_%s'", name, c)
        c = ("usr_" + c)[:60]
    return c


_CONFIG_BLACKLIST = {
    "catalog_config.json", "sso_config.json", "connections.yaml",
    "column_map.json",
}


def _repo_root() -> str:
    return (os.environ.get('GPDM_BASE_DIR', '') or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _default_raw_dir() -> str:
    return os.path.join(_repo_root(), "data", "raw")


def _default_db_path() -> str:
    prod = os.path.join(_repo_root(), "data", "healthcare_production.db")
    if os.path.exists(prod):
        return prod
    return os.path.join(_repo_root(), "data", "healthcare_demo.db")


def _import(name: str):
    try:
        return __import__(f"scripts.{name}", fromlist=[name])
    except Exception:
        try:
            return __import__(name)
        except Exception as e:
            _log.warning("Could not import %s: %s", name, e)
            return None


def _snapshot_tables(conn: sqlite3.Connection) -> set:
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        return {r[0] for r in rows}
    except Exception:
        return set()


def _run_text(raw_dir: str, conn: sqlite3.Connection) -> List[str]:
    mod = _import("text_loader")
    if mod is None:
        return []
    before = _snapshot_tables(conn)
    mod.load_text_to_sqlite(raw_dir, conn)
    return sorted(_snapshot_tables(conn) - before)


def _run_csv(raw_dir: str, conn: sqlite3.Connection) -> List[str]:
    import csv as _csv
    import re as _re
    before = _snapshot_tables(conn)
    if not os.path.isdir(raw_dir):
        return []
    _VALID = _re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    def _sanit(name: str) -> Optional[str]:
        if not name:
            return None
        c = _re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_")
        if not c:
            return None
        if not _VALID.match(c):
            c = "_" + c
        return c[:60]

    for fname in sorted(os.listdir(raw_dir)):
        if not fname.lower().endswith(".csv"):
            continue
        fpath = os.path.join(raw_dir, fname)
        base = os.path.splitext(fname)[0]
        table = _sanitize_user_table_name(base)
        if table is None:
            continue
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace", newline="") as f:
                reader = _csv.reader(f)
                headers = next(reader, None)
                if not headers:
                    continue
                safe_cols, seen = [], {}
                for i, h in enumerate(headers):
                    c = _sanit(h.strip()) or f"col_{i}"
                    if c in seen:
                        seen[c] += 1
                        c = f"{c}_{seen[c]}"
                    else:
                        seen[c] = 0
                    safe_cols.append(c)
                col_defs = ", ".join(f'"{c}" TEXT' for c in safe_cols)
                conn.execute(f'DROP TABLE IF EXISTS "{table}"')
                conn.execute(f'CREATE TABLE "{table}" ({col_defs})')
                ph = ", ".join(["?"] * len(safe_cols))
                insert_sql = f'INSERT INTO "{table}" VALUES ({ph})'
                batch = []
                for row in reader:
                    if len(row) < len(safe_cols):
                        row = row + [None] * (len(safe_cols) - len(row))
                    elif len(row) > len(safe_cols):
                        row = row[: len(safe_cols)]
                    batch.append(tuple(row))
                    if len(batch) >= 1000:
                        conn.executemany(insert_sql, batch)
                        batch.clear()
                if batch:
                    conn.executemany(insert_sql, batch)
            conn.commit()
        except Exception as e:
            _log.warning("CSV ingest failed for %s: %s", fname, e)
    return sorted(_snapshot_tables(conn) - before)


def _run_excel(raw_dir: str, conn: sqlite3.Connection) -> List[str]:
    mod = _import("excel_loader")
    if mod is None:
        return []
    before = _snapshot_tables(conn)
    mod.load_excel_to_sqlite(raw_dir, conn)
    return sorted(_snapshot_tables(conn) - before)


def _run_json(raw_dir: str, conn: sqlite3.Connection) -> List[str]:
    mod = _import("json_loader")
    if mod is None:
        return []
    before = _snapshot_tables(conn)
    mod.load_json_to_sqlite(raw_dir, conn)
    return sorted(_snapshot_tables(conn) - before)


def _run_parquet(raw_dir: str, conn: sqlite3.Connection) -> List[str]:
    mod = _import("parquet_loader")
    if mod is None:
        return []
    before = _snapshot_tables(conn)
    try:
        mod.load_parquet_to_sqlite(raw_dir, conn)
    except Exception as e:
        _log.warning("parquet loader failed: %s", e)
    return sorted(_snapshot_tables(conn) - before)


def _fast_path_available() -> bool:
    try:
        fp = _import("fastpath")
        return fp is not None and fp._has("pyarrow")
    except Exception:
        return False


def _run_fast_csv(raw_dir: str, conn: sqlite3.Connection,
                  per_file_results: List[Dict[str, Any]]) -> List[str]:
    arrow = _import("arrow_loaders")
    fp = _import("fastpath")
    if arrow is None or fp is None or not arrow.arrow_available():
        return []
    before = _snapshot_tables(conn)
    pdir = fp.parquet_dir()
    os.makedirs(pdir, exist_ok=True)
    for fname in sorted(os.listdir(raw_dir)):
        low = fname.lower()
        if not low.endswith((".csv", ".tsv", ".psv")):
            continue
        fpath = os.path.join(raw_dir, fname)
        try:
            size = os.path.getsize(fpath)
        except OSError:
            continue
        if size < _FAST_BYTES:
            continue
        base = os.path.splitext(fname)[0]
        out_parquet = os.path.join(pdir, f"{fp.sanitize_id(base)}.parquet")
        res = arrow.load_csv_arrow(
            fpath, out_parquet=out_parquet, sqlite_conn=conn,
        )
        per_file_results.append({
            "file": fname, "engine": "arrow_csv",
            "rows": res.get("rows"),
            "parquet": res.get("parquet"),
        })
    return sorted(_snapshot_tables(conn) - before)


def _run_fast_ndjson(raw_dir: str, conn: sqlite3.Connection,
                     per_file_results: List[Dict[str, Any]]) -> List[str]:
    sj = _import("streaming_json")
    arrow = _import("arrow_loaders")
    fp = _import("fastpath")
    if fp is None:
        return []
    before = _snapshot_tables(conn)
    pdir = fp.parquet_dir()
    os.makedirs(pdir, exist_ok=True)
    for fname in sorted(os.listdir(raw_dir)):
        low = fname.lower()
        if not low.endswith((".ndjson", ".jsonl")):
            continue
        fpath = os.path.join(raw_dir, fname)
        try:
            size = os.path.getsize(fpath)
        except OSError:
            continue
        if size < _FAST_BYTES:
            continue
        base = os.path.splitext(fname)[0]
        out_parquet = os.path.join(pdir, f"{fp.sanitize_id(base)}.parquet")
        res: Dict[str, Any] = {}
        if arrow is not None and arrow.arrow_available():
            res = arrow.load_json_arrow(
                fpath, out_parquet=out_parquet, sqlite_conn=conn,
            )
            if "error" in res and sj is not None:
                res = sj.stream_ndjson(
                    fpath, sqlite_conn=conn, parquet_out=out_parquet,
                )
        elif sj is not None:
            res = sj.stream_ndjson(
                fpath, sqlite_conn=conn, parquet_out=out_parquet,
            )
        per_file_results.append({
            "file": fname, "engine": res.get("engine", "ndjson_stream"),
            "rows": res.get("rows"),
            "parquet": res.get("parquet"),
        })
    return sorted(_snapshot_tables(conn) - before)


def _run_profiler(db_path: str, source: str) -> Optional[Dict[str, Any]]:
    mod = _import("column_profiler")
    if mod is None:
        return None
    try:
        result = mod.run(db_path=db_path, source=source)
        return {
            "source": result.get("source"),
            "draft_path": result.get("draft_path"),
            "tables_mapped": list(result.get("mapping", {}).keys()),
        }
    except Exception as e:
        _log.warning("Profiler run failed: %s", e)
        return {"error": str(e)}


def _refresh_registry() -> str:
    mod = _import("catalog_registry")
    if mod is None:
        return "skipped_no_module"
    try:
        reg = mod.CatalogRegistry.get_instance()
        reg._initialized = False
        reg.environments = {}
        reg._concept_index.clear() if hasattr(reg._concept_index, "clear") else None
        reg._table_index.clear() if hasattr(reg._table_index, "clear") else None
        cfg = os.path.join(_repo_root(), "data", "catalog_config.json")
        reg.initialize(config_path=cfg if os.path.exists(cfg) else None,
                       auto_discover=False)
        return "reloaded"
    except Exception as e:
        _log.warning("Registry refresh failed: %s", e)
        return f"error:{e}"


def _mask_phi_in_tables(conn: sqlite3.Connection,
                        tables: List[str]) -> Dict[str, Any]:
    import time as _t
    t0 = _t.time()
    try:
        from phi_tokenizer import get_tokenizer
        from phi_lineage import get_lineage
    except ImportError:
        try:
            from phi_tokenizer import get_tokenizer
            from phi_lineage import get_lineage
        except ImportError as e:
            _log.warning("PHI modules unavailable: %s", e)
            return {"error": str(e), "skipped": True}

    tok = get_tokenizer()
    lineage = get_lineage()

    def _sql_token(phi_type: str, val: Any) -> Any:
        if val is None:
            return None
        return tok.tokenize(val, phi_type=phi_type)

    conn.create_function("phi_token", 2, _sql_token)

    out: Dict[str, Any] = {
        "tables": {},
        "total_cols_masked": 0,
        "total_rows_updated": 0,
        "ephemeral": tok.ephemeral,
    }

    for tbl in tables:
        try:
            cols = conn.execute(f'PRAGMA table_info("{tbl}")').fetchall()
        except sqlite3.Error as e:
            out.setdefault("errors", []).append(f"{tbl}:{e}")
            continue
        sensitive: Dict[str, str] = {}
        for row in cols:
            col_name = row[1]
            phi_type = tok.classify_column(col_name)
            if phi_type:
                sensitive[col_name] = phi_type
        if not sensitive:
            continue

        try:
            nrows = int(conn.execute(f'SELECT COUNT(*) FROM "{tbl}"').fetchone()[0])
        except sqlite3.Error:
            nrows = 0

        for col, phi_type in sensitive.items():
            try:
                sample = conn.execute(
                    f'SELECT "{col}" FROM "{tbl}" '
                    f'WHERE "{col}" IS NOT NULL LIMIT 1'
                ).fetchone()
            except sqlite3.Error:
                sample = None
            preview = ""
            if sample and sample[0] is not None:
                sval = str(sample[0])
                preview = "*" * max(0, len(sval) - 4) + sval[-4:]

            try:
                conn.execute(
                    f'UPDATE "{tbl}" SET "{col}" = phi_token(?, "{col}") '
                    f'WHERE "{col}" IS NOT NULL',
                    (phi_type,),
                )
                conn.commit()
                out["total_rows_updated"] += nrows
                out["total_cols_masked"] += 1
            except sqlite3.Error as e:
                out.setdefault("errors", []).append(f"{tbl}.{col}:{e}")
                continue

            try:
                lineage.record(
                    source_env="raw_drop",
                    source_table=tbl,
                    source_col=col,
                    masked_table=tbl,
                    masked_col=col,
                    phi_type=phi_type,
                    masking_fn="hmac_sha256",
                    masking_version="v1",
                    sample_preview=preview,
                )
            except Exception as e:
                out.setdefault("errors", []).append(
                    f"lineage:{tbl}.{col}:{e}"
                )

        out["tables"][tbl] = sensitive

    try:
        tok.flush_vault()
    except Exception:
        pass

    out["elapsed_ms"] = int((_t.time() - t0) * 1000)

    try:
        from phi_audit import get_audit_log
    except ImportError:
        try:
            from phi_audit import get_audit_log
        except ImportError:
            get_audit_log = None
    if get_audit_log is not None:
        try:
            al = get_audit_log()
            al.record(
                user_id="system",
                role="ingest",
                question_raw="",
                question_masked="<phi_mask_sweep>",
                resolved_sql="UPDATE ... SET col = phi_token(col)",
                tables_touched=list(out["tables"].keys()),
                phi_cols_touched=[
                    f"{t}.{c}" for t, cols in out["tables"].items() for c in cols
                ],
                phi_tokens_in_result=out["total_rows_updated"],
                rows_returned=out["total_rows_updated"],
                elapsed_ms=out["elapsed_ms"],
                source_ip="",
                outcome="ok",
            )
        except Exception as e:
            _log.warning("PHI audit record failed: %s", e)

    return out


def ingest_raw_folder(
    raw_dir: Optional[str] = None,
    db_path: Optional[str] = None,
    source: str = "raw",
    run_profiler: bool = True,
    refresh_registry: bool = True,
) -> Dict[str, Any]:
    t0 = time.time()
    raw_dir = raw_dir or os.environ.get("GPDM_RAW_DIR") or _default_raw_dir()
    db_path = db_path or _default_db_path()

    summary: Dict[str, Any] = {
        "raw_dir": raw_dir,
        "db_path": db_path,
        "source": source,
        "files": [],
        "tables_created": [],
        "profiler": None,
        "registry": "skipped",
        "errors": [],
        "elapsed_ms": 0,
    }

    if not os.path.isdir(raw_dir):
        summary["errors"].append(f"raw_dir not found: {raw_dir}")
        return summary

    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        existing = _snapshot_tables(conn)

        for fname in sorted(os.listdir(raw_dir)):
            if fname in _CONFIG_BLACKLIST:
                continue
            low = fname.lower()
            fpath = os.path.join(raw_dir, fname)
            if not os.path.isfile(fpath):
                continue
            if low.endswith(".csv"):
                summary["files"].append({"file": fname, "route": "csv"})
            elif low.endswith(_EXT_TEXT):
                summary["files"].append({"file": fname, "route": "text"})
            elif low.endswith(_EXT_EXCEL):
                summary["files"].append({"file": fname, "route": "excel"})
            elif low.endswith(_EXT_JSON):
                summary["files"].append({"file": fname, "route": "json"})
            elif low.endswith(_EXT_PARQUET):
                summary["files"].append({"file": fname, "route": "parquet"})
            else:
                summary["files"].append({"file": fname, "route": "skipped"})

        fp = _import("fastpath")
        if fp is not None:
            summary["engine"] = fp.describe_engine_selection()

        fast_results: List[Dict[str, Any]] = []
        try:
            added_csv = _run_fast_csv(raw_dir, conn, fast_results)
            added_ndj = _run_fast_ndjson(raw_dir, conn, fast_results)
            for t in (added_csv or []) + (added_ndj or []):
                summary["tables_created"].append(t)
        except Exception as e:
            _log.warning("fast path runners failed: %s", e)
            summary["errors"].append(f"fast_path:{e}")
        if fast_results:
            summary["fast_path"] = fast_results

        for runner, name in (
            (_run_csv,     "csv"),
            (_run_text,    "text"),
            (_run_excel,   "excel"),
            (_run_json,    "json"),
            (_run_parquet, "parquet"),
        ):
            try:
                added = runner(raw_dir, conn)
                if added:
                    summary["tables_created"].extend(added)
                    _log.info("%s loader created tables: %s", name, added)
            except Exception as e:
                _log.warning("%s loader failed: %s", name, e)
                summary["errors"].append(f"{name}:{e}")

        summary["tables_created"] = sorted(
            set(summary["tables_created"]) - existing
        )

        rows_per_table: Dict[str, int] = {}
        total_rows = 0
        for tbl in summary["tables_created"]:
            try:
                n = conn.execute(f'SELECT COUNT(*) FROM "{tbl}"').fetchone()[0]
                rows_per_table[tbl] = int(n)
                total_rows += int(n)
            except Exception as e:
                rows_per_table[tbl] = -1
                summary["errors"].append(f"count:{tbl}:{e}")
        summary["rows_per_table"] = rows_per_table
        summary["rows_ingested"] = total_rows

        if os.environ.get("GPDM_PHI_MASK", "1").lower() in ("1", "true", "yes"):
            try:
                mask_summary = _mask_phi_in_tables(conn, summary["tables_created"])
                summary["phi_masking"] = mask_summary
            except Exception as e:
                _log.warning("PHI masking sweep failed: %s", e)
                summary["errors"].append(f"phi_masking:{e}")
    finally:
        conn.close()

    try:
        _g = sqlite3.connect(db_path)
        try:
            _ic = _g.execute("PRAGMA integrity_check").fetchone()
            if not _ic or _ic[0] != "ok":
                summary["errors"].append(f"post_ingest_integrity:{_ic}")
                _log.error("Post-ingest integrity_check FAILED on %s: %s",
                           db_path, _ic)
            _bad = _g.execute(
                "SELECT COUNT(*) FROM sqlite_master "
                "WHERE type NOT IN ('table','index','view','trigger')"
            ).fetchone()[0]
            if _bad:
                summary["errors"].append(
                    f"post_ingest_schema_poison:{_bad}_rows")
                _log.error("Post-ingest: sqlite_master contains %d rows "
                           "with invalid type — schema poisoned on %s",
                           _bad, db_path)
        finally:
            _g.close()
    except Exception as _ige:
        summary["errors"].append(f"post_ingest_integrity_exception:{_ige}")
        _log.warning("Post-ingest integrity check raised: %s", _ige)

    if run_profiler and summary["tables_created"]:
        summary["profiler"] = _run_profiler(db_path, source=source)

    if refresh_registry:
        summary["registry"] = _refresh_registry()

    summary["elapsed_ms"] = int((time.time() - t0) * 1000)
    return summary


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(
        description="Dispatch any supported file in data/raw/ into the GPDM DB."
    )
    ap.add_argument("--dir", default=None, help="raw drop folder")
    ap.add_argument("--db", default=None, help="target SQLite DB path")
    ap.add_argument("--source", default="raw",
                    help="label used for the profiler's draft column map")
    ap.add_argument("--no-profiler", action="store_true")
    ap.add_argument("--no-registry", action="store_true")
    args = ap.parse_args()

    result = ingest_raw_folder(
        raw_dir=args.dir,
        db_path=args.db,
        source=args.source,
        run_profiler=not args.no_profiler,
        refresh_registry=not args.no_registry,
    )
    print(json.dumps(result, indent=2, default=str))
