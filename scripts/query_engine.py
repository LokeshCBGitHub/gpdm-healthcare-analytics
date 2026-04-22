"""
Query Engine — Central orchestrator for the MTP Healthcare NL→SQL pipeline.

Pipeline:
    question → input validation → session context (follow-ups, duplicates) →
    catalog discovery → conversational/metadata early exit →
    enrichment engines (ML, NLP, Scale) → DynamicSQLEngine → SQL execution →
    result caching + audit logging → response

Public API:
    process_question(question, cfg, catalog=None, session=None) → dict

All external module imports are wrapped in try/except for graceful degradation.
Only stdlib is used directly; optional engines enhance accuracy when present.
"""

import os
import json
import re
import sqlite3
import csv
import time
import threading
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════════════════════
# GRACEFUL IMPORTS — Optional engines degrade gracefully when absent
# ═══════════════════════════════════════════════════════════════════════════════

# ML intent classification + entity extraction
try:
    from ml_integration import MLEngine
except ImportError:
    MLEngine = None

# NumPy-powered TF-IDF + cosine similarity
try:
    from nlp_engine import HybridNLPEngine
except ImportError:
    HybridNLPEngine = None

# Knowledge graph, vector search, query cache
try:
    from graph_vector_engine import ScaleEngine, DatabricksDialect
except ImportError:
    ScaleEngine = None
    DatabricksDialect = None

# Data lineage tracking
try:
    from lineage_tracker import LineageTracker
except ImportError:
    LineageTracker = None

# Analytics recommendations
try:
    from analytics_advisor import AnalyticsAdvisor
except ImportError:
    AnalyticsAdvisor = None

# Production infrastructure: DB pool, logging, validation, audit
try:
    from production import (
        DatabasePool, get_logger, audit_log, validate_question,
        validate_sql_output, resolve_config, record_query_metric,
        setup_logging,
    )
    HAS_PRODUCTION = True
except ImportError:
    HAS_PRODUCTION = False

# Databricks connectivity
try:
    from databricks_connector import DataSourceManager, DatabricksConfig
    HAS_DATABRICKS = True
except ImportError:
    HAS_DATABRICKS = False


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════════

def auto_discover_config() -> Optional[str]:
    """Auto-discover mtp_chatbot.cfg in ../paramset/ relative to this script."""
    config_path = Path(__file__).parent.parent / "paramset" / "mtp_chatbot.cfg"
    return str(config_path) if config_path.exists() else None


def read_config(path: str) -> Dict[str, str]:
    """
    Parse a KEY=VALUE config file with ${VAR} interpolation.

    Supports:
        - Comments (lines starting with #)
        - Quoted values (single or double quotes stripped)
        - Nested variable references resolved in up to 5 passes
    """
    config = {}
    if not os.path.exists(path):
        print(f"[WARNING] Config file not found: {path}")
        return config

    try:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip().strip('"').strip("'")
    except Exception as e:
        print(f"[ERROR] Failed to read config: {e}")

    # Resolve ${VAR} references (up to 5 passes for nested refs)
    for _ in range(5):
        changed = False
        for key, val in config.items():
            if '${' not in val:
                continue
            new_val = re.sub(
                r'\$\{(\w+)\}',
                lambda m: config.get(m.group(1), m.group(0)),
                val,
            )
            if new_val != val:
                config[key] = new_val
                changed = True
        if not changed:
            break

    return config


# ═══════════════════════════════════════════════════════════════════════════════
# SEMANTIC CATALOG — Table/column metadata from JSON profiles
# ═══════════════════════════════════════════════════════════════════════════════

class SemanticCatalog:
    """
    Loads table profiles and relationship definitions from the semantic_catalog/
    directory. Provides keyword-based table resolution and column lookup.
    """

    def __init__(self, catalog_dir: str = None):
        if catalog_dir is None:
            catalog_dir = str(Path(__file__).parent.parent / "semantic_catalog")

        self.catalog_dir = catalog_dir
        self.tables: Dict[str, dict] = {}
        self.relationships: list = []
        self.column_index: Dict[str, List[str]] = defaultdict(list)

        self._load_catalog()

    def _load_catalog(self):
        """Load table profiles and relationships from JSON files."""
        self._load_tables()
        self._load_relationships()

    def _load_tables(self):
        """Load table profiles, normalizing dict-format columns."""
        tables_dir = os.path.join(self.catalog_dir, "tables")
        if not os.path.exists(tables_dir):
            return

        for filename in os.listdir(tables_dir):
            if not filename.endswith('.json'):
                continue
            try:
                with open(os.path.join(tables_dir, filename), 'r') as f:
                    data = json.load(f)

                table_name = data.get('table_name', data.get('name', filename[:-5]))
                data['table_name'] = table_name

                # ── Normalize columns: dict-keyed → list-of-dicts ──
                # Some catalog JSONs store columns as {"COL_NAME": {props}}
                # rather than [{"column_name": "COL_NAME", ...}].
                cols_raw = data.get('columns', [])
                if isinstance(cols_raw, dict):
                    cols_raw = [{'column_name': k, **v} for k, v in cols_raw.items()]
                    data['columns'] = cols_raw

                self.tables[table_name] = data

                # Build reverse index: column_name → [table_names]
                for col in data.get('columns', []):
                    col_name = col.get('column_name', col.get('name', ''))
                    if col_name:
                        self.column_index[col_name.lower()].append(table_name)

            except Exception as e:
                print(f"[WARNING] Failed to load table {filename}: {e}")

    def _load_relationships(self):
        """Load relationship definitions from JSON files."""
        rel_dir = os.path.join(self.catalog_dir, "relationships")
        if not os.path.exists(rel_dir):
            return

        for filename in os.listdir(rel_dir):
            if not filename.endswith('.json'):
                continue
            try:
                with open(os.path.join(rel_dir, filename), 'r') as f:
                    data = json.load(f)

                if isinstance(data, list):
                    self.relationships = data
                elif isinstance(data, dict) and 'relationships' in data:
                    self.relationships = data['relationships']
                else:
                    self.relationships.append(data)

            except Exception as e:
                print(f"[WARNING] Failed to load relationship {filename}: {e}")

    # ── Table resolution ──────────────────────────────────────────────────

    # Keyword → table mapping for column-based inference when no direct match
    _COL_TABLE_MAP = {
        'paid': 'claims', 'billed': 'claims', 'claim': 'claims',
        'copay': 'claims', 'cost': 'claims', 'denied': 'claims',
        'amount': 'claims', 'denial': 'claims',
        'encounter': 'encounters', 'visit': 'encounters',
        'admit': 'encounters', 'discharge': 'encounters',
        'member': 'members', 'patient': 'members', 'enroll': 'members',
        'mrn': 'members', 'dob': 'members', 'birth': 'members',
        'provider': 'providers', 'npi': 'providers', 'doctor': 'providers',
        'specialty': 'providers', 'panel': 'providers',
        'diagnos': 'diagnoses', 'icd': 'diagnoses', 'hcc': 'diagnoses',
        'chronic': 'diagnoses',
        'prescri': 'prescriptions', 'medication': 'prescriptions',
        'rx': 'prescriptions', 'pharmacy': 'prescriptions', 'drug': 'prescriptions',
        'referral': 'referrals', 'refer': 'referrals',
        'appointment': 'appointments', 'appt': 'appointments',
        'cpt': 'cpt_codes',
    }

    def find_tables(self, question: str) -> List[str]:
        """
        Find relevant tables for a question using layered matching:
        1. Direct table name / alias match
        2. FuzzyMatcher abbreviation resolution
        3. Column-keyword inference
        """
        q = question.lower()
        relevant = set()

        # Layer 1: direct table name or alias
        for table_name, table_info in self.tables.items():
            keywords = [table_name.lower()]
            keywords.extend(table_info.get('aliases', []))
            keywords.extend(table_info.get('synonyms', []))
            if any(kw in q for kw in keywords):
                relevant.add(table_name)

        # Layer 2: fuzzy abbreviation match
        if not relevant:
            for word in q.split():
                resolved = FuzzyMatcher.resolve_table(word, self)
                if resolved:
                    relevant.add(resolved)

        # Layer 3: column-keyword inference
        if not relevant:
            for kw, tbl in self._COL_TABLE_MAP.items():
                if kw in q and tbl in self.tables:
                    relevant.add(tbl)
                    break

        return list(relevant)

    def find_column_by_name(self, name: str) -> List[Tuple[str, dict]]:
        """Find columns by name across all tables. Returns [(table, col_info)]."""
        name_lower = name.lower()
        results = []
        for table_name in self.column_index.get(name_lower, []):
            if table_name not in self.tables:
                continue
            for col in self.tables[table_name].get('columns', []):
                cn = col.get('column_name', col.get('name', ''))
                if cn.lower() == name_lower:
                    results.append((table_name, col))
                    break
        return results

    def get_all_table_names(self) -> List[str]:
        """Return all loaded table names."""
        return list(self.tables.keys())

    def get_all_column_names(self) -> List[str]:
        """Return all column names across all tables (for ML training)."""
        return list(self.column_index.keys())

    def get_context(self, question: str) -> Dict[str, Any]:
        """Get full context for a question: tables, details, relationships."""
        tables = self.find_tables(question)
        return {
            'tables': tables,
            'table_details': [self.tables.get(t, {}) for t in tables],
            'relationships': self.relationships if isinstance(self.relationships, list)
                             else list(self.relationships.values()) if isinstance(self.relationships, dict)
                             else [],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# FUZZY MATCHER — Healthcare abbreviations and synonym resolution
# ═══════════════════════════════════════════════════════════════════════════════

class FuzzyMatcher:
    """Healthcare-specific fuzzy matching with domain abbreviations and synonyms."""

    ABBREVIATIONS = {
        'enc': 'encounters', 'encounter': 'encounters',
        'clm': 'claims', 'claim': 'claims',
        'mbr': 'members', 'member': 'members',
        'dx': 'diagnoses', 'diagnosis': 'diagnoses',
        'rx': 'prescriptions', 'prescription': 'prescriptions',
        'ref': 'referrals', 'referral': 'referrals',
        'prov': 'providers', 'provider': 'providers',
        'pts': 'members', 'pt': 'members', 'patient': 'members',
        'appt': 'appointments', 'appointment': 'appointments',
        'ip': 'inpatient', 'inpt': 'inpatient',
        'op': 'outpatient', 'opt': 'outpatient',
        'avg': 'average', 'cnt': 'count', 'amt': 'amount',
        'dt': 'date', 'yr': 'year', 'mo': 'month',
    }

    SYNONYMS = {
        'patient': ['pt', 'member', 'mbr', 'individual'],
        'encounter': ['visit', 'appointment', 'admission'],
        'claim': ['medical_claim', 'insurance_claim'],
        'diagnosis': ['condition', 'dx', 'icd', 'disease'],
        'provider': ['physician', 'doctor', 'clinician', 'facility'],
        'cost': ['charge', 'amount', 'expense', 'price', 'paid_amount'],
    }

    @staticmethod
    def fuzzy_score(term: str, target: str) -> float:
        """Simple fuzzy scoring: exact=1.0, prefix=0.9, substring=0.7, else=0.0."""
        t, g = term.lower().strip(), target.lower().strip()
        if t == g:
            return 1.0
        if g.startswith(t):
            return 0.9
        if t in g:
            return 0.7
        return 0.0

    @staticmethod
    def resolve_table(term: str, catalog: 'SemanticCatalog') -> Optional[str]:
        """Resolve abbreviated or synonymous term to an actual table name."""
        term_lower = term.lower()
        all_tables = catalog.get_all_table_names()

        # Abbreviation lookup
        if term_lower in FuzzyMatcher.ABBREVIATIONS:
            resolved = FuzzyMatcher.ABBREVIATIONS[term_lower]
            for t in all_tables:
                if FuzzyMatcher.fuzzy_score(resolved, t) > 0.7:
                    return t

        # Direct fuzzy match
        for t in all_tables:
            if FuzzyMatcher.fuzzy_score(term, t) > 0.8:
                return t

        # Synonym match
        for canonical, syns in FuzzyMatcher.SYNONYMS.items():
            if term_lower in syns or term_lower == canonical:
                for t in all_tables:
                    if FuzzyMatcher.fuzzy_score(canonical, t) > 0.7:
                        return t

        return None

    @staticmethod
    def resolve_column_name(term: str, catalog: 'SemanticCatalog') -> Optional[str]:
        """Resolve abbreviated term to actual column name."""
        term_lower = term.lower()

        if term_lower in FuzzyMatcher.ABBREVIATIONS:
            resolved = FuzzyMatcher.ABBREVIATIONS[term_lower]
            for col_entries in catalog.column_index.values():
                for table_name in col_entries:
                    if FuzzyMatcher.fuzzy_score(resolved, table_name) > 0.7:
                        return table_name

        if term_lower in catalog.column_index:
            return term_lower

        return None


# ═══════════════════════════════════════════════════════════════════════════════
# CONVERSATIONAL HANDLER — Greetings, help, thanks, goodbye
# ═══════════════════════════════════════════════════════════════════════════════

_CONVERSATIONAL_PATTERNS = [
    (r'\bhello\b|\bhi\b|\bhey\b', 'greeting'),
    (r'\bhelp\b|\bwhat can you do\b|\bcapabilities\b', 'help'),
    (r'\bthank\b|\bthanks\b|\bappreciate\b', 'thanks'),
    (r'\bbye\b|\bgoodbye\b|\bsee you\b', 'bye'),
]

_CONVERSATIONAL_RESPONSES = {
    'greeting': (
        "Hi there! I'm your MTP healthcare chatbot. I can help you explore "
        "patient encounters, claims, diagnoses, procedures, referrals, and more. "
        "Ask me about patient trends, costs, readmissions, or any healthcare metrics!"
    ),
    'help': (
        "I can help with:\n"
        "1. **Counts & Aggregates**: 'How many patients...?' 'Total cost?'\n"
        "2. **Trends**: 'Show trends over time' 'YoY comparison'\n"
        "3. **Breakdowns**: 'Break down by age, gender, provider'\n"
        "4. **Top N**: 'Top 10 providers by cost'\n"
        "5. **Journeys**: 'Patient journey' 'Referral chains'\n"
        "6. **Comparisons**: 'Compare costs between...'\n"
        "7. **Metadata**: Ask about tables, columns, relationships"
    ),
    'thanks': "You're welcome! Anything else I can help with?",
    'bye': "Goodbye! Have a great day!",
}


def _check_conversational(question: str) -> Optional[str]:
    """If the question is conversational, return the response string. Else None."""
    q = question.lower()
    for pattern, key in _CONVERSATIONAL_PATTERNS:
        if re.search(pattern, q):
            return _CONVERSATIONAL_RESPONSES.get(key)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# METADATA ENGINE — Answers about tables, columns, relationships, PHI
# ═══════════════════════════════════════════════════════════════════════════════

_METADATA_PATTERNS = [
    r'\bwhat column\b|\bwhat field\b',
    r'\bwhat table\b',
    r'\brelationship\b',
    r'\bprimary key\b|\bforeign key\b',
    r'\bdata quality\b|\bnull\b',
    r'\bphi\b|\bsensitive\b|\bprivacy\b',
    r'\bhow many column\b|\bhow many table\b',
    r'\bcolumns in\b',
]


def _is_metadata_question(question: str) -> bool:
    """Check if question is about schema metadata rather than data."""
    q = question.lower()
    return any(re.search(p, q) for p in _METADATA_PATTERNS)


def _answer_metadata(question: str, catalog: SemanticCatalog) -> str:
    """Answer a metadata question without generating SQL."""
    q = question.lower()

    if re.search(r'\bwhat table\b', q):
        return f"Available tables: {', '.join(catalog.get_all_table_names())}"

    if re.search(r'\bcolumns in\b', q):
        for tname in catalog.get_all_table_names():
            if tname.lower() in q:
                cols = [c.get('column_name', c.get('name', ''))
                        for c in catalog.tables.get(tname, {}).get('columns', [])]
                return f"Columns in {tname}: {', '.join(c for c in cols if c)}"
        return "Could not identify table in your question."

    if re.search(r'\bphi\b|\bsensitive\b', q):
        return (
            "PHI (Protected Health Information) fields typically include: "
            "patient_id, name, ssn, dob, address, phone, email. "
            "These should be handled per HIPAA compliance rules."
        )

    if re.search(r'\brelationship\b', q):
        rels = catalog.relationships
        if rels and isinstance(rels, list):
            lines = []
            for r in rels[:5]:
                if isinstance(r, dict):
                    lines.append(
                        f"  {r.get('from_table', '?')} → {r.get('to_table', '?')} "
                        f"(on {r.get('join_key', r.get('on', '?'))})"
                    )
            return f"Key relationships:\n" + "\n".join(lines) if lines else "No relationships defined."
        return "No relationships defined."

    return "Please ask more specifically about a table, column, or relationship."


# ═══════════════════════════════════════════════════════════════════════════════
# JOURNEY BUILDER — Multi-table JOIN chains for clinical pathways
# ═══════════════════════════════════════════════════════════════════════════════

# Journey type → regex patterns
_JOURNEY_PATTERNS = {
    'patient_journey': [r'\bpatient journey\b', r'\bpatient flow\b', r'\bpatient timeline\b'],
    'referral_chain':  [r'\breferral chain\b', r'\breferral path\b'],
    'claim_lifecycle': [r'\bclaim lifecycle\b', r'\bclaim journey\b', r'\bfrom visit to payment\b'],
    'readmission':     [r'\breadmission\b', r'\breturn visit\b', r'\bwithin \d+ days\b'],
    'cost_analysis':   [r'\bcost analysis\b', r'\bcost by\b', r'\bspend\b'],
}

# Journey type → table chain (only referencing actual schema tables)
_JOURNEY_TABLES = {
    'patient_journey': ['members', 'encounters', 'diagnoses', 'claims'],
    'referral_chain':  ['referrals', 'encounters', 'providers'],
    'claim_lifecycle': ['encounters', 'claims', 'cpt_codes'],
    'readmission':     ['members', 'encounters', 'diagnoses'],
    'cost_analysis':   ['encounters', 'claims', 'providers'],
}


def _detect_journey(question: str) -> Optional[str]:
    """Detect journey type from question. Returns journey key or None."""
    q = question.lower()
    for jtype, patterns in _JOURNEY_PATTERNS.items():
        for p in patterns:
            if re.search(p, q):
                return jtype
    return None


def _get_journey_tables(journey_type: str) -> List[str]:
    """Get the table chain for a journey type."""
    return _JOURNEY_TABLES.get(journey_type, [])


# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE PATH RESOLUTION
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_db_path(cfg: Dict[str, str]) -> Optional[str]:
    """Resolve the path to the production database file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    # Try configured path first, then standard location
    for candidate in [
        cfg.get('DB_PATH', ''),
        os.path.join(base_dir, 'data', 'healthcare_production.db'),
        os.path.join(base_dir, 'data', 'healthcare_demo.db'),
    ]:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# SQL EXECUTION — Production DB pool or legacy CSV fallback
# ═══════════════════════════════════════════════════════════════════════════════

def _execute_sql_local(sql: str, cfg: Dict[str, str],
                       timeout: int = 30) -> Tuple[bool, List, str]:
    """
    Execute SQL against the persistent database (production mode) or
    fall back to in-memory CSV loading (legacy mode).

    Production mode uses DatabasePool singleton — data loaded ONCE at startup.
    Legacy mode rebuilds from CSVs on each call (backward compatibility only).

    Returns:
        (success, rows_as_dicts, error_message)
    """
    logger = get_logger('mtp.sql') if HAS_PRODUCTION else None

    # ── Databricks routing (if connected) ──
    if HAS_DATABRICKS:
        try:
            ds_mgr = DataSourceManager.get_instance()
            if ds_mgr.is_databricks:
                def _local_fallback(sql_fb, cfg_fb):
                    return _execute_sql_impl(sql_fb, cfg_fb, timeout, logger)
                return ds_mgr.execute_query(sql, local_fallback_fn=_local_fallback, cfg=cfg)
        except Exception as e:
            if logger:
                logger.warning("DataSourceManager error, falling back: %s", e)

    return _execute_sql_impl(sql, cfg, timeout, logger)


def _execute_sql_impl(sql: str, cfg: Dict[str, str],
                      timeout: int = 30, logger=None) -> Tuple[bool, List, str]:
    """Internal: execute SQL against local persistent DB or in-memory CSV."""

    # ── Production mode: persistent DatabasePool ──
    if HAS_PRODUCTION:
        try:
            pool = DatabasePool.get_instance()
            if not pool._loaded:
                pool.initialize(resolve_config(cfg))

            # Validate SQL before execution
            allowed = pool.get_table_names()
            is_valid, err = validate_sql_output(sql, allowed)
            if not is_valid:
                if logger:
                    logger.warning("SQL validation failed: %s — %s", err, sql[:200])
                return False, [], f"SQL validation: {err}"

            return pool.execute_query(sql, timeout)

        except Exception as e:
            if logger:
                logger.error("Production DB error, falling back to legacy: %s", e)
            # Fall through to legacy mode

    # ── Legacy mode: in-memory CSV loading ──
    return _execute_sql_legacy(sql, cfg, logger)


def _execute_sql_legacy(sql: str, cfg: Dict[str, str],
                        logger=None) -> Tuple[bool, List, str]:
    """Legacy CSV-to-SQLite execution for backward compatibility."""
    try:
        conn = sqlite3.connect(':memory:')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        base_dir = cfg.get('BASE_DIR', str(Path(__file__).parent.parent))
        data_path = cfg.get('RAW_DIR', os.path.join(base_dir, 'data', 'raw'))

        if not os.path.exists(data_path):
            return False, [], f"Data directory not found: {data_path}"

        loaded = set()
        for filename in os.listdir(data_path):
            if not filename.endswith('.csv'):
                continue
            table_name = filename[:-4]
            try:
                with open(os.path.join(data_path, filename), 'r',
                          encoding='utf-8', errors='replace') as f:
                    reader = csv.DictReader(f)
                    columns = reader.fieldnames
                    if not columns:
                        continue

                    col_defs = ', '.join(f'"{c}" TEXT' for c in columns)
                    cursor.execute(f'CREATE TABLE "{table_name}" ({col_defs})')

                    placeholders = ', '.join('?' for _ in columns)
                    insert = f'INSERT INTO "{table_name}" VALUES ({placeholders})'
                    batch = []
                    for row in reader:
                        batch.append([row.get(c, '') for c in columns])
                        if len(batch) >= 5000:
                            cursor.executemany(insert, batch)
                            batch = []
                    if batch:
                        cursor.executemany(insert, batch)

                    loaded.add(table_name)
                    conn.commit()
            except Exception as e:
                if logger:
                    logger.warning("Failed to load CSV %s: %s", filename, e)

        if not loaded:
            return False, [], f"No CSV files loaded from {data_path}"

        cursor.execute(sql)
        raw_rows = cursor.fetchall()
        rows = [dict(r) for r in raw_rows] if raw_rows and hasattr(raw_rows[0], 'keys') else raw_rows
        conn.close()
        return True, rows, ""

    except sqlite3.Error as e:
        return False, [], f"SQL Error: {e}"
    except Exception as e:
        return False, [], f"Execution Error: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# LINEAGE & ANALYTICS WRAPPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _get_lineage_info(question: str, tables_used: List[str]) -> Dict[str, Any]:
    """Track data lineage if LineageTracker is available."""
    if LineageTracker is None:
        return {'source_tables': tables_used, 'lineage': f"Question → {' → '.join(tables_used)}"}
    try:
        return LineageTracker().track(question, tables_used)
    except Exception as e:
        return {'error': str(e), 'source_tables': tables_used}


def _get_analytics_recommendations(question: str, tables_used: List[str]) -> Dict[str, Any]:
    """Get analytics recommendations if AnalyticsAdvisor is available."""
    if AnalyticsAdvisor is None:
        return {'recommendations': []}
    try:
        return AnalyticsAdvisor().recommend(question, tables_used)
    except Exception as e:
        return {'error': str(e), 'recommendations': []}


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE SINGLETON CACHE — Expensive engines cached as module-level singletons
# ═══════════════════════════════════════════════════════════════════════════════

_engine_cache: Dict[str, Any] = {}
_engine_cache_lock = threading.Lock()


def clear_engine_cache():
    """Clear all cached engines. Call on dashboard restart or data source change."""
    with _engine_cache_lock:
        for engine in _engine_cache.values():
            if hasattr(engine, 'query_cache'):
                try:
                    engine.query_cache.cache.clear()
                except Exception:
                    pass
        _engine_cache.clear()


def _get_cached_engine(key: str, factory, *args, **kwargs):
    """Get or create a cached engine instance (thread-safe, double-checked locking)."""
    if key not in _engine_cache:
        with _engine_cache_lock:
            if key not in _engine_cache:
                try:
                    _engine_cache[key] = factory(*args, **kwargs)
                except Exception:
                    return None
    return _engine_cache.get(key)


# ═══════════════════════════════════════════════════════════════════════════════
# PROCESS QUESTION — Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def _empty_result(question: str) -> Dict[str, Any]:
    """Create an empty result template."""
    return {
        'success': False,
        'question': question,
        'sql': None,
        'answer': None,
        'tables_used': [],
        'intent': None,
        'engine_mode': None,
        'is_metadata': False,
        'confidence': 0.0,
    }


def _init_session_context(question: str, session: dict, result: dict):
    """
    Initialize session context: duplicate detection, conversation chaining,
    table hierarchy, catalog discovery.

    Mutates `result` with context info and may return a modified question
    (merged with follow-up context).

    Returns:
        (modified_question, session_mgr, learning_engine, catalog_registry)
    """
    _session_mgr = None
    _learning_engine = None
    _catalog_registry = None

    try:
        from session_context import (
            SessionContextManager, ConversationChainer,
            DuplicateDetector, TableHierarchyResolver,
        )
        _session_mgr = SessionContextManager()
    except ImportError:
        pass

    try:
        from catalog_registry import CatalogRegistry, ContextualLearningEngine
        _catalog_registry = CatalogRegistry.get_instance()
        if not _catalog_registry._initialized:
            cat_cfg = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                '..', 'data', 'catalog_config.json',
            )
            _catalog_registry.initialize(config_path=cat_cfg)
        _learning_engine = ContextualLearningEngine()
    except ImportError:
        pass

    user_id = session.get('user_id', 'anonymous') if session else 'anonymous'
    session_id = session.get('session_id', 'default') if session else 'default'

    if _session_mgr:
        try:
            recent = _session_mgr.get_recent_turns(user_id, session_id, count=10)

            # Duplicate detection
            dup = DuplicateDetector.check(question, recent, threshold=0.85)
            if dup:
                result['duplicate_warning'] = dup

            # Conversation chaining — follow-up detection
            chain = ConversationChainer.analyze(question, recent)
            if chain.get('is_followup') and chain.get('carry_forward'):
                result['followup_info'] = chain
                merged = ConversationChainer.merge_context(
                    question, chain['carry_forward'], chain['followup_type'],
                )
                if merged and merged != question:
                    result['merged_question'] = merged
                    question = merged

            # Table hierarchy detection
            hierarchy = TableHierarchyResolver.detect_from_question(question)
            if hierarchy and hierarchy.get('filter'):
                result['hierarchy_match'] = hierarchy

            # Active UI selections
            active = _session_mgr.get_state(user_id, session_id, 'active_selections')
            if active:
                result['active_selections'] = active
        except Exception:
            pass

    return question, _session_mgr, _learning_engine, _catalog_registry


def _run_enrichment_engines(question: str, catalog: SemanticCatalog,
                            result: dict) -> dict:
    """
    Run optional ML, NLP, and Scale engines for enrichment.
    Returns scale_meta dict and mutates result with ml_meta, nlp_meta, scale_meta.

    If Scale engine has a cache hit, returns the cached result directly
    (caller should check result['engine_mode'] == 'cached').
    """
    # ── ML Engine: ensemble intent classification ──
    if MLEngine:
        try:
            def _init_ml():
                ml = MLEngine()
                ml.initialize(
                    column_names=catalog.get_all_column_names(),
                    enable_llm=False,
                )
                return ml

            ml = _get_cached_engine('ml_engine', _init_ml)
            if ml:
                ml_intent, ml_conf, ml_details = ml.classify_intent(question)
                result['ml_meta'] = {
                    'ml_intent': ml_intent,
                    'ml_confidence': ml_conf,
                    'ml_models_used': list(ml_details.get('models', {}).keys()),
                    'ml_method': ml_details.get('method', 'unknown'),
                }
                if not result['intent']:
                    result['intent'] = ml_intent
        except Exception:
            pass

    # ── NLP Engine: TF-IDF + cosine similarity ──
    nlp_inst = None
    if HybridNLPEngine:
        try:
            cat_dir = catalog.catalog_dir if hasattr(catalog, 'catalog_dir') else None
            nlp_inst = _get_cached_engine('nlp_engine', HybridNLPEngine, cat_dir)
            if nlp_inst:
                analysis = nlp_inst.analyze_question(question)
                nlp_meta = {
                    'nlp_intent': analysis.get('intent', ''),
                    'nlp_confidence': analysis.get('intent_confidence', 0),
                    'nlp_tables': analysis.get('relevant_tables', []),
                    'nlp_top_column': '',
                    'nlp_method': analysis.get('method', 'numpy_hybrid'),
                }
                cols = analysis.get('relevant_columns', [])
                if cols:
                    top = cols[0]
                    nlp_meta['nlp_top_column'] = f"{top['table']}.{top['column']}"
                result['nlp_meta'] = nlp_meta
                if not result['intent']:
                    result['intent'] = analysis.get('intent', '')
        except Exception:
            pass

    # ── Scale Engine: Knowledge Graph, Vector Search, Query Cache ──
    scale_meta = {}
    if ScaleEngine:
        try:
            def _init_scale():
                s = ScaleEngine(catalog.catalog_dir if hasattr(catalog, 'catalog_dir') else None)
                s._init_stats = s.initialize()
                return s

            scale = _get_cached_engine('scale_engine', _init_scale)
            if not scale:
                raise RuntimeError("Scale engine init failed")

            stats = getattr(scale, '_init_stats', {})

            # Cache hit check (semantic near-duplicate)
            cached = scale.query_cache.get(question)
            if cached and isinstance(cached.get('answer'), list) and cached['answer']:
                result.update(cached)
                result['engine_mode'] = 'cached'
                result['success'] = True
                result['scale_meta'] = {'cache_hit': True}
                return {'_scale': scale, '_nlp': nlp_inst, 'cache_hit': True}

            # Vector search + knowledge graph
            schema_hits = scale.search_schema(question, k=5)
            query_words = [w for w in question.lower().split() if len(w) > 2]
            subgraph = scale.get_relevant_subgraph(query_words)

            scale_meta = {
                'vector_search_hits': len(schema_hits),
                'top_vector_hit': schema_hits[0]['id'] if schema_hits else None,
                'subgraph_nodes': subgraph.get('nodes', 0),
                'subgraph_tables': subgraph.get('tables', []),
                'cache_hit': False,
                'kg_nodes': stats.get('knowledge_graph', {}).get('total_nodes', 0),
                'kg_edges': stats.get('knowledge_graph', {}).get('total_edges', 0),
            }
            result['scale_meta'] = scale_meta
        except Exception:
            pass

    return {'_scale': _engine_cache.get('scale_engine'), '_nlp': nlp_inst, 'cache_hit': False}


def _classify_intent_from_dynamic(dyn_result: dict) -> str:
    """Derive intent label from DynamicSQLEngine output."""
    agg_func = dyn_result.get('agg_info', {}).get('agg_func')
    filters = dyn_result.get('filters', [])
    top_n = dyn_result.get('agg_info', {}).get('top_n')

    if agg_func == 'COUNT':
        return 'count'
    if agg_func in ('AVG', 'SUM', 'MAX', 'MIN'):
        return 'aggregate'
    if any('DENIED' in str(f) or 'denied' in str(f) for f in filters):
        return 'negation'
    if top_n:
        return 'top_n'
    return 'lookup'


def _post_process_result(result: dict, question: str, rows: list,
                         engines: dict, dyn_result: dict,
                         _learning_engine, session: dict, _t0: float):
    """
    Post-process after successful SQL execution:
    - NLP validation
    - Scale engine caching + Databricks translation
    - Contextual learning log
    """
    nlp_inst = engines.get('_nlp')
    scale_inst = engines.get('_scale')

    # NLP result validation
    if nlp_inst and rows:
        try:
            validation = nlp_inst.validate_results(rows, question, result['sql'])
            if 'nlp_meta' in result:
                result['nlp_meta']['validation'] = validation
        except Exception:
            pass

    # Scale engine: Databricks SQL + cache + pattern mining
    if scale_inst:
        try:
            if DatabricksDialect:
                result['databricks_sql'] = DatabricksDialect.translate(
                    result['sql'], tables=result['tables_used'], use_catalog=True,
                )
            if isinstance(rows, list) and rows:
                scale_inst.query_cache.put(question, {
                    'sql': result['sql'], 'answer': result['answer'],
                    'tables_used': result['tables_used'],
                    'intent': result['intent'], 'success': True,
                })
            scale_inst.pattern_miner.log_query(
                question, result['sql'],
                dyn_result.get('agg_info', {}), result['tables_used'],
            )
        except Exception:
            pass

    # Contextual learning log
    _log_to_learning_engine(_learning_engine, session, question, result, _t0)


def _log_to_learning_engine(learning_engine, session, question, result, _t0):
    """Log query to contextual learning engine (if available)."""
    if not learning_engine:
        return
    try:
        user_id = session.get('user_id', 'anonymous') if session else 'api'
        username = session.get('username', 'anonymous') if session else 'api'
        row_cnt = len(result.get('answer', [])) if isinstance(result.get('answer'), list) else 0
        learning_engine.log_query(
            user_id=user_id, username=username, question=question,
            sql=result.get('sql', ''), tables=result.get('tables_used', []),
            intent=result.get('intent', ''), success=result.get('success', False),
            result_count=row_cnt,
            execution_time_ms=int((time.time() - _t0) * 1000),
        )
    except Exception:
        pass


def _audit_and_finalize(result: dict, question: str, session: dict,
                        _t0: float, _learning_engine,
                        similar_queries: list):
    """Final step: attach suggestions, record audit metrics."""
    elapsed_ms = (time.time() - _t0) * 1000

    # Attach similar queries
    if similar_queries:
        result['similar_queries'] = similar_queries

    # Attach user suggestions
    if _learning_engine:
        try:
            uid = session.get('user_id', 'anonymous') if session else 'api'
            result['suggestions'] = _learning_engine.get_suggestions(user_id=uid, limit=5)
        except Exception:
            pass

    # Production audit log + metrics
    if HAS_PRODUCTION:
        row_count = len(result.get('answer', [])) if isinstance(result.get('answer'), list) else 0
        record_query_metric(result['success'], elapsed_ms,
                            error=result.get('_dynamic_error'))
        audit_log(
            event='QUERY',
            user=session.get('username', 'anonymous') if session else 'api',
            details={
                'intent': result.get('intent', ''),
                'engine_mode': result.get('engine_mode', ''),
                'tables_used': ','.join(result.get('tables_used', [])),
                'row_count': row_count,
                'success': result['success'],
                'response_time_ms': round(elapsed_ms, 1),
                'confidence': result.get('confidence', 0),
                'question_length': len(question),
            },
        )


def process_question(question: str, cfg: Dict[str, str],
                     catalog: 'SemanticCatalog' = None,
                     session: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Main entry point: process a natural language question end-to-end.

    Pipeline:
        Input validation → Session context → Catalog discovery →
        Conversational/metadata early exit → Enrichment engines (ML, NLP, Scale) →
        DynamicSQLEngine SQL generation → SQL execution → Post-processing →
        Audit logging → Response

    Args:
        question: Natural language question
        cfg:      Configuration dict (data_dir, etc.)
        catalog:  SemanticCatalog instance (auto-created if None)
        session:  Optional session dict with user_id, username, history, etc.

    Returns:
        Dict with success, question, sql, answer, tables_used, intent,
        engine_mode, is_metadata, confidence, and optional enrichment keys.
    """
    _t0 = time.time()
    _logger = get_logger('mtp.engine') if HAS_PRODUCTION else None

    # ── 1. INPUT VALIDATION ──
    if HAS_PRODUCTION:
        is_valid, sanitized, err_msg = validate_question(question)
        if not is_valid:
            if _logger:
                _logger.warning("Invalid input rejected: %s", err_msg)
            r = _empty_result(question)
            r.update({'answer': f'Invalid query: {err_msg}', 'engine_mode': 'validation_error'})
            return r
        question = sanitized

    if catalog is None:
        catalog = SemanticCatalog()

    result = _empty_result(question)

    # ── 2. SESSION CONTEXT (follow-ups, duplicates, hierarchy) ──
    question, _session_mgr, _learning_engine, _catalog_registry = \
        _init_session_context(question, session, result)

    # ── 3. CATALOG DISCOVERY (multi-catalog disambiguation) ──
    skip_discovery = session.get('skip_catalog_discovery', False) if session else False
    if _catalog_registry and _catalog_registry._initialized and not skip_discovery:
        discovery = _catalog_registry.discover_for_query(question)
        if discovery.get('needs_clarification') and discovery.get('clarification_prompt'):
            result.update({
                'answer': discovery['clarification_prompt'],
                'success': True,
                'engine_mode': 'catalog_discovery',
                'is_metadata': True,
                'catalog_discovery': discovery,
            })
            return result

    # ── 4. SIMILAR QUERIES (for suggestions) ──
    similar_queries = []
    if _learning_engine:
        try:
            similar_queries = _learning_engine.get_similar_queries(question, limit=3)
        except Exception:
            pass

    # ── 5. CONVERSATIONAL EARLY EXIT ──
    conv_response = _check_conversational(question)
    if conv_response:
        result.update({'answer': conv_response, 'success': True, 'engine_mode': 'conversational'})
        return result

    # ── 6. METADATA EARLY EXIT ──
    if _is_metadata_question(question):
        result.update({
            'answer': _answer_metadata(question, catalog),
            'success': True,
            'engine_mode': 'metadata',
            'is_metadata': True,
        })
        return result

    # ── 7. LINEAGE & ANALYTICS (enrichment, non-blocking) ──
    lineage_info = _get_lineage_info(question, [])
    if lineage_info.get('source_tables'):
        result['lineage'] = lineage_info
    recommendations = _get_analytics_recommendations(question, [])
    if recommendations.get('recommendations'):
        result['analytics_recommendations'] = recommendations

    # ── 8. JOURNEY DETECTION ──
    journey_type = _detect_journey(question)
    if journey_type:
        result['intent'] = journey_type
        result['engine_mode'] = 'journey'

    # ── 9. ENRICHMENT ENGINES (ML, NLP, Scale) ──
    engines = _run_enrichment_engines(question, catalog, result)
    if engines.get('cache_hit'):
        return result  # Scale engine cache hit — already populated

    # ── 10. SEMANTIC SQL ENGINE (primary) → DYNAMIC SQL ENGINE (fallback) ──
    # The SemanticSQLEngine uses TF-IDF, cosine similarity, intent classification,
    # knowledge graphs, and schema auto-discovery to generate SQL intelligently.
    # It falls back to the hardcoded DynamicSQLEngine for low-confidence queries.
    try:
        # Try the intelligence-driven semantic engine first
        semantic_layer = None
        dyn_result = None
        try:
            from semantic_sql_engine import SemanticSQLEngine
            db_path = _resolve_db_path(cfg)
            if db_path and os.path.exists(db_path):
                sem_engine = SemanticSQLEngine(db_path, catalog.catalog_dir)
                dyn_result = sem_engine.generate(question)
                # Keep reference to semantic layer for validation later
                semantic_layer = sem_engine.semantic
        except Exception:
            pass  # Graceful degradation

        # Fallback: use the hardcoded DynamicSQLEngine directly
        if dyn_result is None:
            from dynamic_sql_engine import DynamicSQLEngine
            dyn_engine = DynamicSQLEngine(catalog.catalog_dir)
            dyn_result = dyn_engine.generate(question)

            # Still try to initialize semantic layer for validation
            try:
                from semantic_layer import SemanticLayer
                db_path = _resolve_db_path(cfg)
                if db_path and os.path.exists(db_path):
                    semantic_layer = SemanticLayer(db_path)
                    semantic_layer.initialize()
                    # Enrich fallback result with semantic metadata
                    sem_intent = semantic_layer.classify_intent(question)
                    dyn_result['semantic_intent'] = sem_intent.get('intent')
                    dyn_result['semantic_confidence'] = sem_intent.get('confidence', 0)
                    sem_cols = semantic_layer.match_columns(question, top_k=5)
                    sem_vals = semantic_layer.match_values(question, top_k=5)
                    dyn_result['semantic_columns'] = sem_cols
                    dyn_result['semantic_values'] = sem_vals
                    sql = dyn_result.get('sql', '')
                    tables = dyn_result.get('tables_used', [])
                    if sql and tables:
                        validation = semantic_layer.validate(sql, tables)
                        if validation.get('warnings'):
                            dyn_result['data_warnings'] = validation['warnings']
                            dyn_result['data_suggestions'] = validation.get('suggestions', [])
            except Exception:
                pass

        # Handle multi-question results
        if dyn_result.get('multi_query') and dyn_result.get('all_results'):
            return _handle_multi_query(dyn_result, result, cfg)

        # Single query
        result['sql'] = dyn_result.get('sql')
        result['tables_used'] = dyn_result.get('tables_used', [])
        result['confidence'] = dyn_result.get('confidence', 0.5)
        result['intent'] = _classify_intent_from_dynamic(dyn_result)

        # Pass semantic enrichment to result
        if dyn_result.get('data_warnings'):
            result['data_warnings'] = dyn_result['data_warnings']
            result['data_suggestions'] = dyn_result.get('data_suggestions', [])

        # Execute SQL
        if result['sql'] and not result['sql'].startswith('--'):
            success, rows, error = _execute_sql_local(result['sql'], cfg)
            if success:
                # Check for empty results with semantic validation
                if not rows and semantic_layer:
                    try:
                        has_data, msg = semantic_layer.check_empty(result['sql'])
                        if not has_data:
                            result['answer'] = []
                            result['success'] = True
                            result['engine_mode'] = 'dynamic'
                            result['no_data_message'] = msg
                            _audit_and_finalize(
                                result, question, session, _t0,
                                _learning_engine, similar_queries,
                            )
                            return result
                    except Exception:
                        pass

                result['answer'] = rows if rows else []
                result['success'] = True
                result['engine_mode'] = 'dynamic'

                _post_process_result(
                    result, question, rows, engines, dyn_result,
                    _learning_engine, session, _t0,
                )
                _audit_and_finalize(
                    result, question, session, _t0,
                    _learning_engine, similar_queries,
                )
                return result

            # Execution failed — record error, fall through
            result['_dynamic_error'] = error

    except Exception as e:
        result['_dynamic_error'] = str(e)

    # ── 11. FINAL LOGGING + AUDIT (for fallthrough/error cases) ──
    _log_to_learning_engine(_learning_engine, session, question, result, _t0)
    _audit_and_finalize(result, question, session, _t0, _learning_engine, similar_queries)

    return result


def _handle_multi_query(dyn_result: dict, result: dict,
                        cfg: dict) -> Dict[str, Any]:
    """Execute multiple sub-queries from a multi-question parse."""
    all_sub = dyn_result['all_results']
    combined_answers = []
    combined_sql = []
    all_tables = []

    for i, sub in enumerate(all_sub):
        sub_sql = sub.get('sql', '')
        combined_sql.append(sub_sql)
        if sub_sql and not sub_sql.startswith('--'):
            success, rows, error = _execute_sql_local(sub_sql, cfg)
            combined_answers.append({
                'question_number': i + 1,
                'sql': sub_sql,
                'data': rows if success and rows else [],
                'tables': sub.get('tables_used', []),
            })
            for t in sub.get('tables_used', []):
                if t not in all_tables:
                    all_tables.append(t)

    result.update({
        'sql': ';\n'.join(combined_sql),
        'tables_used': all_tables,
        'confidence': min(r.get('confidence', 0.5) for r in all_sub),
        'intent': 'multi_query',
        'answer': combined_answers,
        'success': True,
        'engine_mode': 'dynamic',
        'multi_query': True,
    })
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE ENTRY POINT (interactive CLI)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    config_path = auto_discover_config()
    if config_path:
        _cfg = read_config(config_path)
        print(f"[INFO] Loaded config from {config_path}")
    else:
        _cfg = {'data_dir': '../data'}
        print("[WARNING] Config not found, using defaults")

    _catalog = SemanticCatalog()
    _session = {'username': 'cli', 'history': []}

    print("\n=== MTP Healthcare Chatbot (CLI) ===")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break

            r = process_question(user_input, _cfg, _catalog, _session)
            _session['history'].append(user_input)

            print(f"\nBot (Intent: {r['intent']}, Mode: {r['engine_mode']})")
            if isinstance(r['answer'], list) and r['answer']:
                rows = r['answer']
                print(f"Results ({len(rows)} rows):")
                if isinstance(rows[0], dict):
                    cols = list(rows[0].keys())
                    widths = {c: min(max(len(c), max((len(str(row.get(c, ''))[:20]) for row in rows[:10]), default=4)), 20) for c in cols}
                    print('  ' + '  '.join(c.ljust(widths[c]) for c in cols))
                    print('  ' + '  '.join('-' * widths[c] for c in cols))
                    for row in rows[:15]:
                        print('  ' + '  '.join(str(row.get(c, ''))[:20].ljust(widths[c]) for c in cols))
                    if len(rows) > 15:
                        print(f"  ... and {len(rows) - 15} more rows")
            elif isinstance(r['answer'], str):
                print(r['answer'])

            if r['sql']:
                print(f"\nSQL: {r['sql']}")
            print()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            traceback.print_exc()
