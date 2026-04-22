from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("load_data")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
DEFAULT_RAW = DATA_DIR / "raw"
DEFAULT_DB = DATA_DIR / "healthcare_production.db"
MODEL_DIR = DATA_DIR / "models"

os.environ.setdefault("GPDM_BASE_DIR", str(SCRIPT_DIR))
os.environ.setdefault("GPDM_ML_ENABLED", "1")

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


SUPPORTED_EXTENSIONS = {
    ".csv", ".tsv", ".psv",
    ".json", ".ndjson", ".jsonl",
    ".xlsx", ".xlsm",
    ".parquet",
}


def detect_files(raw_dir: str) -> Dict[str, Any]:
    raw = Path(raw_dir)
    if not raw.is_dir():
        return {"found": 0, "files": [], "error": f"Directory not found: {raw_dir}"}

    files = []
    for f in sorted(raw.rglob("*")):
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
            size_mb = f.stat().st_size / (1024 * 1024)
            files.append({
                "path": str(f),
                "name": f.name,
                "type": f.suffix.lower().lstrip("."),
                "size_mb": round(size_mb, 1),
            })

    total_mb = sum(f["size_mb"] for f in files)
    log.info("Found %d file(s) totaling %.1f MB in %s", len(files), total_mb, raw_dir)
    for f in files:
        log.info("  %-40s  %7.1f MB  (%s)", f["name"], f["size_mb"], f["type"])

    return {"found": len(files), "total_mb": round(total_mb, 1), "files": files}


def load_data(raw_dir: str, db_path: str) -> Dict[str, Any]:
    log.info("Loading data from %s → %s", raw_dir, db_path)
    t0 = time.time()

    try:
        from raw_ingest import ingest_raw_folder
        result = ingest_raw_folder(
            raw_dir=raw_dir,
            db_path=db_path,
            run_profiler=True,
            refresh_registry=True,
        )
        elapsed = time.time() - t0

        tables = result.get("tables_created", [])
        files = result.get("files", [])
        total_rows = sum(f.get("rows", 0) for f in files if isinstance(f, dict))
        errors = result.get("errors", [])

        log.info("Loaded %d file(s) → %d table(s), %s rows in %.1fs",
                 len(files), len(tables), f"{total_rows:,}" if total_rows else "?", elapsed)

        if errors:
            for e in errors:
                log.warning("  Load error: %s", e)

        return {
            "status": "ok" if not errors else "partial",
            "files_processed": len(files),
            "tables_created": tables,
            "total_rows": total_rows,
            "errors": errors,
            "elapsed_seconds": round(elapsed, 1),
        }
    except Exception as e:
        log.error("Data loading failed: %s", e)
        return {"status": "error", "error": str(e), "elapsed_seconds": round(time.time() - t0, 1)}


def rebuild_summaries(db_path: str) -> Dict[str, Any]:
    log.info("Rebuilding summary tables...")
    t0 = time.time()
    conn = sqlite3.connect(db_path)
    results = {}

    try:
        conn.execute("DROP TABLE IF EXISTS _gpdm_summary_monthly_backup")
        try:
            conn.execute("ALTER TABLE _gpdm_summary_monthly RENAME TO _gpdm_summary_monthly_backup")
        except Exception:
            pass

        conn.execute("""
            CREATE TABLE IF NOT EXISTS _gpdm_summary_monthly AS
            SELECT
                substr(service_date, 1, 7) as month,
                COALESCE(region, 'Unknown') as region,
                COALESCE(status, 'Unknown') as claim_status,
                COALESCE(encounter_type, 'ALL') as visit_type,
                COUNT(*) as claim_count,
                ROUND(SUM(CAST(paid_amount AS REAL)), 2) as total_paid,
                ROUND(SUM(CAST(billed_amount AS REAL)), 2) as total_billed,
                ROUND(AVG(CAST(paid_amount AS REAL)), 2) as avg_paid,
                ROUND(AVG(CAST(billed_amount AS REAL)), 2) as avg_billed,
                COUNT(DISTINCT member_id) as unique_members,
                COUNT(DISTINCT provider_id) as unique_providers,
                SUM(CASE WHEN UPPER(status) = 'DENIED' THEN 1 ELSE 0 END) as denied_count,
                SUM(CASE WHEN denial_reason IS NOT NULL AND denial_reason != '' THEN 1 ELSE 0 END) as has_denial_reason
            FROM claims_4m
            WHERE service_date IS NOT NULL
            GROUP BY month, region, claim_status, visit_type
        """)
        conn.execute("DROP TABLE IF EXISTS _gpdm_summary_monthly_backup")
        cnt = conn.execute("SELECT COUNT(*) FROM _gpdm_summary_monthly").fetchone()[0]
        results["summary_monthly"] = cnt
        log.info("  _gpdm_summary_monthly: %s rows", f"{cnt:,}")

        conn.execute("DELETE FROM _gpdm_kpi_facts")
        conn.execute("""
            INSERT INTO _gpdm_kpi_facts
                (total_claims, total_members, total_providers, total_paid, total_billed,
                 avg_paid, total_denied, denial_rate, earliest_date, latest_date, total_months)
            SELECT
                COUNT(*),
                (SELECT COUNT(*) FROM members),
                COUNT(DISTINCT provider_id),
                ROUND(SUM(CAST(paid_amount AS REAL)), 2),
                ROUND(SUM(CAST(billed_amount AS REAL)), 2),
                ROUND(AVG(CAST(paid_amount AS REAL)), 2),
                SUM(CASE WHEN UPPER(status)='DENIED' THEN 1 ELSE 0 END),
                ROUND(SUM(CASE WHEN UPPER(status)='DENIED' THEN 1 ELSE 0 END)*100.0
                      / NULLIF(COUNT(*), 0), 2),
                MIN(service_date),
                MAX(service_date),
                COUNT(DISTINCT substr(service_date, 1, 7))
            FROM claims_4m
        """)
        results["kpi_facts"] = "rebuilt"
        log.info("  _gpdm_kpi_facts: rebuilt")

        conn.execute("DROP TABLE IF EXISTS _gpdm_summary_region")
        conn.execute("""
            CREATE TABLE _gpdm_summary_region AS
            SELECT
                COALESCE(region, 'Unknown') as region,
                COUNT(*) as claim_count,
                SUM(CASE WHEN UPPER(status)='DENIED' THEN 1 ELSE 0 END) as denied_count,
                ROUND(SUM(CASE WHEN UPPER(status)='DENIED' THEN 1.0 ELSE 0 END)*100.0
                      / NULLIF(COUNT(*), 0), 2) as denial_rate,
                ROUND(AVG(CAST(paid_amount AS REAL)), 2) as avg_paid,
                COUNT(DISTINCT member_id) as unique_members
            FROM claims_4m
            GROUP BY region
        """)
        cnt = conn.execute("SELECT COUNT(*) FROM _gpdm_summary_region").fetchone()[0]
        results["summary_region"] = cnt
        log.info("  _gpdm_summary_region: %s rows", cnt)

        conn.execute("DROP TABLE IF EXISTS _gpdm_summary_provider")
        conn.execute("""
            CREATE TABLE _gpdm_summary_provider AS
            SELECT
                provider_id,
                COUNT(*) as claim_count,
                SUM(CASE WHEN UPPER(status)='DENIED' THEN 1 ELSE 0 END) as denied_count,
                ROUND(SUM(CASE WHEN UPPER(status)='DENIED' THEN 1.0 ELSE 0 END)*100.0
                      / NULLIF(COUNT(*), 0), 2) as denial_rate,
                ROUND(AVG(CAST(paid_amount AS REAL)), 2) as avg_paid,
                COUNT(DISTINCT member_id) as unique_members,
                COUNT(DISTINCT encounter_type) as encounter_types
            FROM claims_4m
            WHERE provider_id IS NOT NULL
            GROUP BY provider_id
        """)
        cnt = conn.execute("SELECT COUNT(*) FROM _gpdm_summary_provider").fetchone()[0]
        results["summary_provider"] = cnt
        log.info("  _gpdm_summary_provider: %s rows", f"{cnt:,}")

        conn.execute("DROP TABLE IF EXISTS _gpdm_summary_status")
        conn.execute("""
            CREATE TABLE _gpdm_summary_status AS
            SELECT
                COALESCE(status, 'Unknown') as claim_status,
                COUNT(*) as claim_count,
                ROUND(AVG(CAST(paid_amount AS REAL)), 2) as avg_paid,
                ROUND(SUM(CAST(paid_amount AS REAL)), 2) as total_paid
            FROM claims_4m
            GROUP BY claim_status
        """)
        results["summary_status"] = "rebuilt"

        conn.execute("DROP TABLE IF EXISTS _gpdm_summary_visit_type")
        conn.execute("""
            CREATE TABLE _gpdm_summary_visit_type AS
            SELECT
                COALESCE(encounter_type, 'Unknown') as visit_type,
                COUNT(*) as claim_count,
                SUM(CASE WHEN UPPER(status)='DENIED' THEN 1 ELSE 0 END) as denied_count,
                ROUND(AVG(CAST(paid_amount AS REAL)), 2) as avg_paid,
                COUNT(DISTINCT member_id) as unique_members
            FROM claims_4m
            GROUP BY visit_type
        """)
        results["summary_visit_type"] = "rebuilt"

        conn.execute("DROP TABLE IF EXISTS _gpdm_summary_denial")
        conn.execute("""
            CREATE TABLE _gpdm_summary_denial AS
            SELECT
                COALESCE(denial_reason, 'None') as denial_reason,
                COUNT(*) as claim_count,
                ROUND(AVG(CAST(billed_amount AS REAL)), 2) as avg_billed,
                ROUND(AVG(CAST(paid_amount AS REAL)), 2) as avg_paid
            FROM claims_4m
            WHERE UPPER(status) = 'DENIED'
            GROUP BY denial_reason
        """)
        results["summary_denial"] = "rebuilt"

        conn.commit()
        elapsed = time.time() - t0
        log.info("All summaries rebuilt in %.1fs", elapsed)
        results["elapsed_seconds"] = round(elapsed, 1)
        results["status"] = "ok"

    except Exception as e:
        conn.rollback()
        log.error("Summary rebuild failed: %s", e)
        results["status"] = "error"
        results["error"] = str(e)
    finally:
        conn.close()

    return results


def train_models(db_path: str, sample_size: int = 500000) -> Dict[str, Any]:
    log.info("Training models (sample size: %s)...", f"{sample_size:,}")
    t0 = time.time()

    try:
        from ml_pretrain import MLModelStore
        store = MLModelStore()
        manifest = store.train_all(db_path=db_path, sample_size=sample_size)
        elapsed = time.time() - t0

        models = manifest.get("models", {})
        trained = [k for k, v in models.items() if isinstance(v, dict) and v.get("status") == "trained"]
        failed = [k for k, v in models.items() if isinstance(v, dict) and v.get("status") == "error"]

        log.info("Training complete: %d models trained, %d failed in %.1fs",
                 len(trained), len(failed), elapsed)
        for m in trained:
            log.info("  ✓ %s", m)
        for m in failed:
            log.warning("  ✗ %s", m)

        return {
            "status": "ok" if not failed else "partial",
            "models_trained": trained,
            "models_failed": failed,
            "elapsed_seconds": round(elapsed, 1),
        }
    except Exception as e:
        log.error("Model training failed: %s", e)
        return {"status": "error", "error": str(e), "elapsed_seconds": round(time.time() - t0, 1)}


def verify(db_path: str) -> Dict[str, Any]:
    log.info("Verifying system health...")
    conn = sqlite3.connect(db_path)
    checks = {}

    try:
        for table in ["claims_4m", "members", "providers", "encounters"]:
            cnt = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            checks[table] = f"{cnt:,} rows"
            status = "OK" if cnt > 0 else "EMPTY"
            log.info("  %-25s %10s rows  [%s]", table, f"{cnt:,}", status)

        for table in ["_gpdm_summary_monthly", "_gpdm_kpi_facts", "_gpdm_summary_region",
                       "_gpdm_summary_provider"]:
            try:
                cnt = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                checks[table] = f"{cnt:,} rows"
            except Exception:
                checks[table] = "MISSING"
                log.warning("  %-25s MISSING", table)

        try:
            kpi = conn.execute(
                "SELECT total_claims, denial_rate, avg_paid, total_members FROM _gpdm_kpi_facts"
            ).fetchone()
            if kpi:
                log.info("  KPI snapshot: %s claims | %.1f%% denial rate | $%s avg paid | %s members",
                         f"{int(kpi[0]):,}", float(kpi[1]), f"{float(kpi[2]):,.0f}", f"{int(kpi[3]):,}")
                checks["kpi_snapshot"] = {
                    "total_claims": int(kpi[0]),
                    "denial_rate": float(kpi[1]),
                    "avg_paid": float(kpi[2]),
                    "total_members": int(kpi[3]),
                }
        except Exception as e:
            checks["kpi_snapshot"] = f"error: {e}"

        model_dir = Path(db_path).parent / "models"
        if model_dir.is_dir():
            manifest_file = model_dir / "manifest.json"
            if manifest_file.exists():
                with open(manifest_file) as f:
                    manifest = json.load(f)
                models = manifest.get("models", {})
                trained = [k for k, v in models.items() if isinstance(v, dict) and v.get("status") == "trained"]
                checks["models"] = f"{len(trained)} trained"
                log.info("  Models: %d trained", len(trained))
            else:
                checks["models"] = "no manifest"
                log.warning("  Models: no manifest found")
        else:
            checks["models"] = "no model directory"

        checks["status"] = "ready"
        log.info("System is READY")

    except Exception as e:
        checks["status"] = "error"
        checks["error"] = str(e)
        log.error("Verification failed: %s", e)
    finally:
        conn.close()

    return checks


def run_full_pipeline(
    raw_dir: Optional[str] = None,
    db_path: Optional[str] = None,
    skip_training: bool = False,
    rebuild_only: bool = False,
    sample_size: int = 500000,
) -> Dict[str, Any]:
    raw_dir = raw_dir or str(DEFAULT_RAW)
    db_path = db_path or str(DEFAULT_DB)

    t0 = time.time()
    pipeline = {"steps": {}}

    log.info("=" * 60)
    log.info("GPDM Data Load Pipeline")
    log.info("=" * 60)
    log.info("  Source:   %s", raw_dir)
    log.info("  Database: %s", db_path)
    log.info("  Mode:     %s", "rebuild-only" if rebuild_only else
             ("load + summaries" if skip_training else "full pipeline"))
    log.info("=" * 60)

    if not rebuild_only:
        log.info("\n── STEP 1: Detecting files ──")
        detection = detect_files(raw_dir)
        pipeline["steps"]["detect"] = detection

        if detection["found"] == 0:
            log.warning("No files found in %s — nothing to load", raw_dir)
            if not Path(db_path).exists():
                log.error("Database doesn't exist either. Nothing to do.")
                pipeline["status"] = "error"
                pipeline["error"] = "No files to load and no existing database"
                return pipeline

            log.info("Proceeding with existing database data...")
        else:
            log.info("\n── STEP 2: Loading data ──")
            load_result = load_data(raw_dir, db_path)
            pipeline["steps"]["load"] = load_result

            if load_result.get("status") == "error":
                log.error("Loading failed. Stopping.")
                pipeline["status"] = "error"
                return pipeline

    log.info("\n── STEP 3: Rebuilding summaries ──")
    summary_result = rebuild_summaries(db_path)
    pipeline["steps"]["summaries"] = summary_result

    if not skip_training:
        log.info("\n── STEP 4: Training models ──")
        train_result = train_models(db_path, sample_size=sample_size)
        pipeline["steps"]["training"] = train_result
    else:
        log.info("\n── STEP 4: Training SKIPPED (--skip-training) ──")
        pipeline["steps"]["training"] = {"status": "skipped"}

    log.info("\n── STEP 5: Verifying system ──")
    verify_result = verify(db_path)
    pipeline["steps"]["verify"] = verify_result

    elapsed = time.time() - t0
    pipeline["total_elapsed_seconds"] = round(elapsed, 1)
    pipeline["status"] = verify_result.get("status", "unknown")

    log.info("\n" + "=" * 60)
    log.info("PIPELINE COMPLETE in %.1fs", elapsed)
    log.info("Status: %s", pipeline["status"].upper())
    log.info("=" * 60)

    return pipeline


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="GPDM Data Loader — drop files, system does the rest.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python load_data.py                          # Load from data/raw/, train everything
  python load_data.py --dir /tmp/dump          # Load from custom directory
  python load_data.py --skip-training          # Load + summaries only (fast)
  python load_data.py --rebuild-only           # No new files, just rebuild from existing data
  python load_data.py --sample 100000          # Smaller training sample (faster)
        """,
    )
    ap.add_argument("--dir", default=None, help="Directory containing the data dump")
    ap.add_argument("--db", default=None, help="Target database path")
    ap.add_argument("--skip-training", action="store_true", help="Skip model training")
    ap.add_argument("--rebuild-only", action="store_true", help="Rebuild from existing data (no file loading)")
    ap.add_argument("--sample", type=int, default=500000, help="Training sample size (default: 500K)")
    ap.add_argument("--json", action="store_true", help="Output results as JSON")
    args = ap.parse_args()

    result = run_full_pipeline(
        raw_dir=args.dir,
        db_path=args.db,
        skip_training=args.skip_training,
        rebuild_only=args.rebuild_only,
        sample_size=args.sample,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
