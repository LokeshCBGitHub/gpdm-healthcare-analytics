"""
NL-to-SQL Query Engine - Central Brain of MTP Healthcare Chatbot

This module integrates all from-scratch components:
- Information Retrieval (TF-IDF, BM25) via ir_engine.py
- ML Models (Intent Classifier, Entity Extractor, Similarity Scorer) via ml_models.py
- Mini Transformer LLM via scratch_llm.py
- Semantic catalog, fuzzy matching, journey building, and SQL generation

Architecture:
    Question → ConversationalHandler → LineageTracker → AnalyticsAdvisor →
    MetadataEngine → JourneyBuilder → [IR Engine + ML Models + LLM] →
    SQL Generator → Local SQLite Execution → Result

All imports wrapped in try/except for graceful fallback.
"""

import os
import json
import re
import sqlite3
import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict
import traceback

# ==================== GRACEFUL IMPORTS ====================

try:
    from ir_engine import IREngine
except ImportError:
    IREngine = None

try:
    from ml_models import IntentClassifier, EntityExtractor, SimilarityScorer
except ImportError:
    IntentClassifier = None
    EntityExtractor = None
    SimilarityScorer = None

try:
    from scratch_llm import DeepSeekLM, BPETokenizer
except ImportError:
    DeepSeekLM = None
    BPETokenizer = None

try:
    from ml_integration import MLEngine
except ImportError:
    MLEngine = None

try:
    from lineage_tracker import LineageTracker
except ImportError:
    LineageTracker = None

try:
    from analytics_advisor import AnalyticsAdvisor
except ImportError:
    AnalyticsAdvisor = None

try:
    from graph_vector_engine import ScaleEngine, DatabricksDialect
except ImportError:
    ScaleEngine = None
    DatabricksDialect = None

try:
    from nlp_engine import HybridNLPEngine
except ImportError:
    HybridNLPEngine = None

try:
    from production import (
        DatabasePool, get_logger, audit_log, validate_question,
        validate_sql_output, resolve_config, record_query_metric,
        setup_logging
    )
    HAS_PRODUCTION = True
except ImportError:
    HAS_PRODUCTION = False

try:
    from databricks_connector import DataSourceManager, DatabricksConfig
    HAS_DATABRICKS = True
except ImportError:
    HAS_DATABRICKS = False


# ==================== CONFIG DISCOVERY ====================

def auto_discover_config() -> Optional[str]:
    """
    Auto-discovers mtp_chatbot.cfg in ../paramset/ relative to script location.

    Returns:
        Path to config file if found, None otherwise.
    """
    script_dir = Path(__file__).parent
    paramset_dir = script_dir.parent / "paramset"
    config_path = paramset_dir / "mtp_chatbot.cfg"

    if config_path.exists():
        return str(config_path)
    return None


def read_config(path: str) -> Dict[str, str]:
    """
    Parses KEY=VALUE config file.

    Args:
        path: Path to config file

    Returns:
        Dict of configuration key-value pairs
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
                    value = value.strip().strip('"').strip("'")
                    config[key.strip()] = value
    except Exception as e:
        print(f"[ERROR] Failed to read config: {e}")

    # Resolve ${VAR} references (up to 5 passes for nested refs)
    for _ in range(5):
        changed = False
        for key, val in config.items():
            if '${' in val:
                import re as _re
                def _repl(m):
                    return config.get(m.group(1), m.group(0))
                new_val = _re.sub(r'\$\{(\w+)\}', _repl, val)
                if new_val != val:
                    config[key] = new_val
                    changed = True
        if not changed:
            break

    return config


# ==================== SEMANTIC CATALOG ====================

class SemanticCatalog:
    """
    Loads and manages semantic catalog data: table profiles and relationships.
    Provides methods to search for tables, columns, and relationships.
    """

    def __init__(self, catalog_dir: str = None):
        """
        Initialize semantic catalog from JSON files.

        Args:
            catalog_dir: Path to semantic_catalog directory (auto-discovery from ../semantic_catalog/)
        """
        if catalog_dir is None:
            script_dir = Path(__file__).parent
            catalog_dir = str(script_dir.parent / "semantic_catalog")

        self.catalog_dir = catalog_dir
        self.tables = {}
        self.relationships = {}
        self.column_index = defaultdict(list)  # column_name -> [table_names]

        self._load_catalog()

    def _load_catalog(self):
        """Load table profiles and relationships from JSON files."""
        tables_dir = os.path.join(self.catalog_dir, "tables")
        relationships_dir = os.path.join(self.catalog_dir, "relationships")

        # Load table profiles
        if os.path.exists(tables_dir):
            for filename in os.listdir(tables_dir):
                if filename.endswith('.json'):
                    try:
                        path = os.path.join(tables_dir, filename)
                        with open(path, 'r') as f:
                            data = json.load(f)
                            table_name = data.get('table_name', data.get('name', filename.replace('.json', '')))
                            data['table_name'] = table_name
                            # Normalize columns: if dict, convert to list of dicts
                            cols_raw = data.get('columns', {})
                            if isinstance(cols_raw, dict):
                                cols_list = list(cols_raw.values())
                                data['columns'] = cols_list
                            self.tables[table_name] = data
                            # Build column index
                            for col in data.get('columns', []):
                                col_name = col.get('column_name', col.get('name', ''))
                                if col_name:
                                    self.column_index[col_name.lower()].append(table_name)
                    except Exception as e:
                        print(f"[WARNING] Failed to load table {filename}: {e}")

        # Load relationships
        if os.path.exists(relationships_dir):
            for filename in os.listdir(relationships_dir):
                if filename.endswith('.json'):
                    try:
                        path = os.path.join(relationships_dir, filename)
                        with open(path, 'r') as f:
                            data = json.load(f)
                            # Handle both list format and nested dict format
                            if isinstance(data, list):
                                self.relationships = data
                            elif isinstance(data, dict) and 'relationships' in data:
                                self.relationships = data['relationships']
                            else:
                                rel_id = data.get('id', filename.replace('.json', ''))
                                if not isinstance(self.relationships, list):
                                    self.relationships = []
                                self.relationships.append(data)
                    except Exception as e:
                        print(f"[WARNING] Failed to load relationship {filename}: {e}")

    def find_tables(self, question: str) -> List[str]:
        """
        Find relevant tables for a question using simple keyword matching.

        Args:
            question: Natural language question

        Returns:
            List of relevant table names
        """
        question_lower = question.lower()
        relevant = set()

        # Direct table name match
        for table_name, table_info in self.tables.items():
            table_keywords = [table_name.lower()]
            table_keywords.extend(table_info.get('aliases', []))
            table_keywords.extend(table_info.get('synonyms', []))
            for keyword in table_keywords:
                if keyword in question_lower:
                    relevant.add(table_name)
                    break

        # If no direct match, try fuzzy: match via FuzzyMatcher abbreviations
        if not relevant:
            for word in question_lower.split():
                resolved = FuzzyMatcher.resolve_table(word, self)
                if resolved:
                    relevant.add(resolved[0])

        # If still no match, try column-based: find which table has columns mentioned
        if not relevant:
            col_keywords = {
                'paid': 'claims', 'billed': 'claims', 'claim': 'claims', 'copay': 'claims',
                'cost': 'claims', 'denied': 'claims', 'amount': 'claims',
                'encounter': 'encounters', 'visit': 'encounters', 'admit': 'encounters',
                'discharge': 'encounters', 'chief complaint': 'encounters',
                'member': 'members', 'patient': 'members', 'enroll': 'members',
                'mrn': 'members', 'dob': 'members', 'birth': 'members',
                'provider': 'providers', 'npi': 'providers', 'doctor': 'providers',
                'specialty': 'providers', 'panel': 'providers',
                'diagnos': 'diagnoses', 'icd': 'diagnoses', 'hcc': 'diagnoses',
                'chronic': 'diagnoses',
                'prescri': 'prescriptions', 'medication': 'prescriptions', 'rx': 'prescriptions',
                'pharmacy': 'prescriptions', 'drug': 'prescriptions',
                'referral': 'referrals', 'refer': 'referrals',
            }
            for kw, tbl in col_keywords.items():
                if kw in question_lower and tbl in self.tables:
                    relevant.add(tbl)
                    break

        return list(relevant)

    def find_column_by_name(self, name: str) -> List[Tuple[str, str]]:
        """
        Find columns by name across all tables.

        Args:
            name: Column name (case-insensitive)

        Returns:
            List of (table_name, column_info) tuples
        """
        name_lower = name.lower()
        results = []

        for table_name in self.column_index.get(name_lower, []):
            if table_name in self.tables:
                for col in self.tables[table_name].get('columns', []):
                    cn = col.get('column_name', col.get('name', ''))
                    if cn.lower() == name_lower:
                        results.append((table_name, col))
                        break

        return results

    def get_all_table_names(self) -> List[str]:
        """Get all available table names."""
        return list(self.tables.keys())

    def get_context(self, question: str) -> Dict[str, Any]:
        """
        Get full context for a question: relevant tables, columns, relationships.

        Args:
            question: Natural language question

        Returns:
            Dict with tables, columns, relationships metadata
        """
        tables = self.find_tables(question)
        context = {
            'tables': tables,
            'table_details': [self.tables.get(t, {}) for t in tables],
            'relationships': self.relationships if isinstance(self.relationships, list) else list(self.relationships.values()) if isinstance(self.relationships, dict) else [],
        }
        return context


# ==================== FUZZY MATCHER ====================

class FuzzyMatcher:
    """
    Healthcare-specific fuzzy matching with domain abbreviations and synonyms.
    """

    # Healthcare abbreviations
    ABBREVIATIONS = {
        'enc': 'encounters', 'encounter': 'encounters',
        'clm': 'claims', 'claim': 'claims',
        'mbr': 'members', 'member': 'members',
        'dx': 'diagnoses', 'diagnosis': 'diagnoses',
        'rx': 'prescriptions', 'prescription': 'prescriptions',
        'ref': 'referrals', 'referral': 'referrals',
        'prov': 'providers', 'provider': 'providers',
        'pts': 'patients', 'pt': 'patients', 'patient': 'patients',
        'srv': 'services', 'service': 'services',
        'proc': 'procedures', 'procedure': 'procedures',
        'admit': 'admissions', 'admission': 'admissions',
        'ed': 'emergency_dept', 'emergency': 'emergency_dept',
        'ip': 'inpatient', 'inpt': 'inpatient',
        'op': 'outpatient', 'opt': 'outpatient',
        'px': 'procedures',
        'hx': 'history',
        'sx': 'symptoms',
        'tx': 'treatment',
        'max': 'maximum', 'min': 'minimum',
        'avg': 'average', 'mean': 'average',
        'cnt': 'count', 'cnt': 'count',
        'amt': 'amount',
        'qty': 'quantity',
        'dt': 'date',
        'tm': 'time',
        'yr': 'year',
        'mo': 'month',
        'wk': 'week',
        'dy': 'day',
        'id': 'identifier',
        'desc': 'description',
        'nm': 'name',
    }

    # Synonym groups
    SYNONYMS = {
        'patient': ['pt', 'member', 'mbr', 'individual'],
        'encounter': ['visit', 'appointment', 'admission', 'claim'],
        'claim': ['medical_claim', 'insurance_claim', 'service_claim'],
        'diagnosis': ['condition', 'dx', 'icd', 'disease'],
        'procedure': ['proc', 'px', 'treatment', 'service'],
        'provider': ['physician', 'doctor', 'clinician', 'facility'],
        'cost': ['charge', 'amount', 'expense', 'price', 'paid_amount'],
        'quantity': ['count', 'number', 'cnt', 'qty'],
        'recent': ['latest', 'last', 'current'],
    }

    @staticmethod
    def fuzzy_score(term: str, target: str) -> float:
        """
        Simple fuzzy scoring: exact match > prefix > substring.

        Args:
            term: Input term
            target: Target term to match

        Returns:
            Score 0.0-1.0
        """
        term_lower = term.lower().strip()
        target_lower = target.lower().strip()

        if term_lower == target_lower:
            return 1.0
        elif target_lower.startswith(term_lower):
            return 0.9
        elif term_lower in target_lower:
            return 0.7
        else:
            return 0.0

    @staticmethod
    def resolve_table(term: str, catalog: SemanticCatalog) -> Optional[str]:
        """
        Resolve abbreviated/synonymous term to actual table name.

        Args:
            term: Input term (may be abbreviation)
            catalog: SemanticCatalog instance

        Returns:
            Resolved table name or None
        """
        term_lower = term.lower()

        # Check abbreviations first
        if term_lower in FuzzyMatcher.ABBREVIATIONS:
            resolved = FuzzyMatcher.ABBREVIATIONS[term_lower]
            # Find matching table
            for table_name in catalog.get_all_table_names():
                if FuzzyMatcher.fuzzy_score(resolved, table_name) > 0.7:
                    return table_name

        # Direct match
        for table_name in catalog.get_all_table_names():
            if FuzzyMatcher.fuzzy_score(term, table_name) > 0.8:
                return table_name

        # Check synonyms
        for canonical, synonyms in FuzzyMatcher.SYNONYMS.items():
            if term_lower in synonyms or term_lower == canonical:
                for table_name in catalog.get_all_table_names():
                    if FuzzyMatcher.fuzzy_score(canonical, table_name) > 0.7:
                        return table_name

        return None

    @staticmethod
    def resolve_column_name(term: str, catalog: SemanticCatalog) -> Optional[str]:
        """
        Resolve abbreviated/synonymous term to actual column name.

        Args:
            term: Input term (may be abbreviation)
            catalog: SemanticCatalog instance

        Returns:
            Resolved column name or None
        """
        term_lower = term.lower()

        # Check abbreviations
        if term_lower in FuzzyMatcher.ABBREVIATIONS:
            resolved = FuzzyMatcher.ABBREVIATIONS[term_lower]
            # Find matching column
            for col_entries in catalog.column_index.values():
                for table_name in col_entries:
                    if FuzzyMatcher.fuzzy_score(resolved, table_name) > 0.7:
                        return table_name

        # Direct lookup
        if term_lower in catalog.column_index:
            return term_lower

        return None


# ==================== CONVERSATIONAL HANDLER ====================

class ConversationalHandler:
    """
    Handles non-SQL conversational requests: greetings, help, capabilities.
    """

    @staticmethod
    def is_conversational(question: str) -> bool:
        """Check if question is conversational."""
        patterns = [
            r'\bhello\b|\bhi\b|\bhey\b',
            r'\bhelp\b|\bwhat can you do\b|\bcapabilities\b',
            r'\bthank\b|\bthanks\b|\bappreciate\b',
            r'\bbye\b|\bgoodbye\b|\bsee you\b',
        ]
        for pattern in patterns:
            if re.search(pattern, question.lower()):
                return True
        return False

    @staticmethod
    def handle(question: str) -> str:
        """Handle conversational request."""
        question_lower = question.lower()

        if re.search(r'\bhello\b|\bhi\b|\bhey\b', question_lower):
            return (
                "Hi there! I'm your MTP healthcare chatbot. I can help you explore "
                "patient encounters, claims, diagnoses, procedures, referrals, and more. "
                "Ask me about patient trends, costs, readmissions, or any healthcare metrics!"
            )

        if re.search(r'\bhelp\b|\bwhat can you do\b|\bcapabilities\b', question_lower):
            return (
                "I can help with:\n"
                "1. **Counts & Aggregates**: 'How many patients...?' 'What's the total cost?'\n"
                "2. **Trends**: 'Show trends over time' 'YoY comparison'\n"
                "3. **Breakdowns**: 'Break down by age, gender, provider'\n"
                "4. **Top N**: 'Top 10 providers by cost'\n"
                "5. **Journeys**: 'Patient journey' 'Referral chains' 'Readmission patterns'\n"
                "6. **Comparisons**: 'Compare costs between...'\n"
                "7. **Metadata**: Ask about tables, columns, relationships, data quality\n"
                "Commands: tables, relationships, describe <table>, history, lineage, analytics"
            )

        if re.search(r'\bthank\b|\bthanks\b', question_lower):
            return "You're welcome! Anything else I can help with?"

        if re.search(r'\bbye\b|\bgoodbye\b', question_lower):
            return "Goodbye! Have a great day!"

        return "I'm here to help. Try asking me a healthcare question or type 'help'."


# ==================== METADATA ENGINE ====================

class MetadataEngine:
    """
    Answers metadata questions about tables, columns, relationships, DQ, PHI without SQL.
    """

    @staticmethod
    def is_metadata_question(question: str) -> bool:
        """Check if question is about metadata."""
        patterns = [
            r'\bwhat column\b|\bwhat field\b',
            r'\bwhat table\b',
            r'\brelationship\b',
            r'\bprimary key\b|\bforeign key\b',
            r'\bdata quality\b|\bnull\b',
            r'\bphi\b|\bsensitive\b|\bprivacy\b',
            r'\bhow many column\b|\bhow many table\b',
            r'\bcolumns in\b',
        ]
        for pattern in patterns:
            if re.search(pattern, question.lower()):
                return True
        return False

    @staticmethod
    def answer(question: str, catalog: SemanticCatalog) -> str:
        """Answer metadata question without generating SQL."""
        question_lower = question.lower()

        if re.search(r'\bwhat table\b', question_lower):
            tables = catalog.get_all_table_names()
            return f"Available tables: {', '.join(tables)}"

        if re.search(r'\bcolumns in\b', question_lower):
            # Extract table name
            for table_name in catalog.get_all_table_names():
                if table_name.lower() in question_lower:
                    table_info = catalog.tables.get(table_name, {})
                    columns = [c.get('name') for c in table_info.get('columns', [])]
                    return f"Columns in {table_name}: {', '.join(columns)}"
            return "Could not identify table in your question."

        if re.search(r'\bphi\b|\bsensitive\b', question_lower):
            return (
                "PHI (Protected Health Information) fields typically include: "
                "patient_id, name, ssn, dob, address, phone, email. "
                "These should be handled per HIPAA compliance rules."
            )

        if re.search(r'\brelationship\b', question_lower):
            rels = catalog.relationships
            if rels:
                rel_text = "\n".join([
                    f"  {r.get('from_table')} (PK: {r.get('from_key')}) → "
                    f"{r.get('to_table')} (FK: {r.get('to_key')})"
                    for r in list(rels.values())[:5]
                ])
                return f"Key relationships:\n{rel_text}"
            return "No relationships defined."

        return "Please ask more specifically about a table, column, or relationship."


# ==================== JOURNEY BUILDER ====================

class JourneyBuilder:
    """
    Builds multi-table JOIN chains for 6 journey types.
    """

    JOURNEY_PATTERNS = {
        'patient_journey': [
            r'\bpatient journey\b|\bpatient flow\b',
            r'\bpatient timeline\b',
        ],
        'referral_chain': [
            r'\breferral chain\b|\breferral path\b',
            r'\bhow patient get referred\b',
        ],
        'initiation': [
            r'\binitiation\b|\bfirst visit\b|\bonboarding\b',
        ],
        'claim_lifecycle': [
            r'\bclaim lifecycle\b|\bclaim journey\b',
            r'\bfrom visit to payment\b',
        ],
        'readmission': [
            r'\breadmission\b|\breturn visit\b',
            r'\bwithin \d+ days\b',
        ],
        'cost_analysis': [
            r'\bcost analysis\b|\bcost by\b',
            r'\bspend\b',
        ],
    }

    @staticmethod
    def detect_journey(question: str) -> Optional[str]:
        """Detect journey type from question."""
        for journey_type, patterns in JourneyBuilder.JOURNEY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, question.lower()):
                    return journey_type
        return None

    @staticmethod
    def build_journey(journey_type: str, catalog: SemanticCatalog) -> List[str]:
        """
        Build JOIN chain for journey type.

        Args:
            journey_type: One of the 6 journey types
            catalog: SemanticCatalog

        Returns:
            List of tables in JOIN order
        """
        journeys = {
            'patient_journey': ['members', 'encounters', 'services', 'diagnoses'],
            'referral_chain': ['referrals', 'encounters', 'providers'],
            'initiation': ['members', 'encounters', 'claims'],
            'claim_lifecycle': ['encounters', 'claims', 'services', 'payments'],
            'readmission': ['members', 'encounters', 'diagnoses'],
            'cost_analysis': ['encounters', 'claims', 'services', 'providers'],
        }

        return journeys.get(journey_type, [])


# ==================== INTENT PARSER ====================

class IntentParser:
    """
    Parses user intent using ML when available, falls back to regex.
    Extracts intent type, time filters, aggregations, thresholds.
    """

    INTENT_PATTERNS = {
        # ORDER MATTERS: more specific intents must come before general ones
        'negation': [r'\bdenied\b', r'\brejected\b', r'\bfailed\b', r'\bunpaid\b', r'\bnot paid\b', r'\bwithout\b', r'\bexclude\b'],
        'count': [r'\bhow many\b', r'\bcount\b', r'\btotal number\b'],
        'aggregate': [r'\btotal\b', r'\bsum\b', r'\baverage\b', r'\bmax\b', r'\bmin\b'],
        'trend': [r'\btrend\b', r'\bover time\b', r'\byear over year\b', r'\byoy\b'],
        'breakdown': [r'\bbreak down\b', r'\bby age\b', r'\bby gender\b', r'\bby provider\b', r'\bby region\b', r'\bby status\b', r'\bby type\b', r'\bby facility\b', r'\bby specialty\b', r'\bper\b'],
        'filter': [r'\bwhere\b', r'\bfor patient\b', r'\bin state\b'],
        'top_n': [r'\btop \d+\b', r'\brank\b', r'\bhighest\b', r'\blowest\b'],
        'join': [r'\bcompare\b', r'\bvs\b', r'\bagainst\b'],
        'distinct': [r'\bhow many different\b', r'\bunique\b'],
        'journey': [r'\bjourney\b', r'\bpath\b', r'\bflow\b'],
        'metadata': [r'\bwhat column\b', r'\bwhat table\b', r'\brelationship\b'],
        'threshold': [r'\bmore than\b', r'\bless than\b', r'\bgreater than\b', r'\bexceed\b'],
        'anomaly': [r'\banomal\b', r'\boutlier\b', r'\bunusual\b'],
        'percentage': [r'\bpercent\b', r'\b%\b', r'\bshare\b', r'\bportion\b'],
        'latest': [r'\blatest\b', r'\bmost recent\b', r'\blast\b'],
        'lookup': [r'\bwhat\b', r'\blookup\b', r'\bfind\b', r'\bshow\b', r'\blist\b', r'\bdisplay\b', r'\bgive me\b', r'\bget\b'],
    }

    TIME_PATTERNS = {
        'last_30_days': r'\blast 30 days\b|\blast month\b',
        'last_quarter': r'\blast quarter\b|\blast 3 months\b',
        'this_year': r'\bthis year\b|\bcurrent year\b',
        'year_over_year': r'\byear over year\b|\byoy\b',
        'last_year': r'\blast year\b',
        'last_week': r'\blast week\b|\blast 7 days\b',
    }

    def __init__(self, intent_classifier=None):
        """Initialize with optional ML intent classifier."""
        self.intent_classifier = intent_classifier

    def parse(self, question: str) -> Dict[str, Any]:
        """
        Parse question for intent and metadata.

        Args:
            question: Natural language question

        Returns:
            Dict with intent, time_filter, aggregation, threshold, etc.
        """
        result = {
            'intent': self._detect_intent(question),
            'time_filter': self._extract_time_filter(question),
            'aggregation': self._extract_aggregation(question),
            'threshold': self._extract_threshold(question),
            'negation': self._detect_negation(question),
            'top_n': self._extract_top_n(question),
        }
        return result

    def _detect_intent(self, question: str) -> str:
        """Detect primary intent using ML or regex."""
        if self.intent_classifier:
            try:
                intent = self.intent_classifier.predict(question)
                return intent
            except Exception:
                # Intent classifier failed; fall back to regex patterns
                pass

        # Regex fallback
        for intent_type, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, question.lower()):
                    return intent_type

        return 'lookup'  # default

    def _extract_time_filter(self, question: str) -> Optional[str]:
        """Extract time filter."""
        for time_type, pattern in self.TIME_PATTERNS.items():
            if re.search(pattern, question.lower()):
                return time_type
        return None

    def _extract_aggregation(self, question: str) -> Optional[str]:
        """Extract aggregation function."""
        agg_patterns = {
            'SUM': r'\bsum\b|\btotal\b|\bcombined\b',
            'AVG': r'\baverage\b|\bmean\b',
            'MAX': r'\bmaximum\b|\bmax\b|\bhighest\b',
            'MIN': r'\bminimum\b|\bmin\b|\blowest\b',
            'COUNT': r'\bcount\b|\bhow many\b',
        }
        for agg, pattern in agg_patterns.items():
            if re.search(pattern, question.lower()):
                return agg
        return None

    def _extract_threshold(self, question: str) -> Optional[str]:
        """Extract threshold condition."""
        match = re.search(r'(more|greater|less|over|under|above|below|than|exceed)\s+(\d+)', question.lower())
        if match:
            return match.group(0)
        return None

    def _detect_negation(self, question: str) -> bool:
        """Detect negation."""
        patterns = [r'\bnot\b', r'\bwithout\b', r'\bexclude\b', r'\bno\b']
        for pattern in patterns:
            if re.search(pattern, question.lower()):
                return True
        return False

    def _extract_top_n(self, question: str) -> Optional[int]:
        """Extract TOP N value."""
        match = re.search(r'\btop\s+(\d+)\b', question.lower())
        if match:
            return int(match.group(1))
        return None


# ==================== SQL GENERATOR ====================

class SQLGenerator:
    """
    Generates SQL for all intent types.
    Uses IR engine and Entity Extractor to identify tables and columns.
    """

    def __init__(self, ir_engine=None, entity_extractor=None, catalog=None):
        """Initialize with optional IR engine and entity extractor."""
        self.ir_engine = ir_engine
        self.entity_extractor = entity_extractor
        self.catalog = catalog

    def generate(self, question: str, intent_info: Dict[str, Any],
                 catalog: SemanticCatalog) -> Dict[str, Any]:
        """
        Generate SQL based on intent and question.

        Args:
            question: Natural language question
            intent_info: Output from IntentParser.parse()
            catalog: SemanticCatalog instance

        Returns:
            Dict with sql, tables_used, confidence
        """
        intent = intent_info.get('intent', 'lookup')

        # Default tables if not found
        tables_used = catalog.find_tables(question)
        if not tables_used and catalog.get_all_table_names():
            tables_used = [catalog.get_all_table_names()[0]]

        sql = ""

        try:
            if intent == 'count':
                sql = self._generate_count(question, tables_used)
            elif intent == 'aggregate':
                sql = self._generate_aggregate(question, intent_info, tables_used)
            elif intent == 'trend':
                sql = self._generate_trend(question, tables_used, intent_info)
            elif intent == 'breakdown':
                sql = self._generate_breakdown(question, tables_used)
            elif intent == 'top_n':
                sql = self._generate_top_n(question, tables_used, intent_info)
            elif intent == 'distinct':
                sql = self._generate_distinct(question, tables_used)
            elif intent == 'negation':
                sql = self._generate_negation(question, tables_used)
            elif intent == 'threshold':
                sql = self._generate_threshold(question, tables_used, intent_info)
            elif intent == 'anomaly':
                sql = self._generate_anomaly(question, tables_used)
            elif intent == 'percentage':
                sql = self._generate_percentage(question, tables_used)
            elif intent == 'latest':
                sql = self._generate_latest(question, tables_used)
            else:
                sql = self._generate_lookup(question, tables_used)

        except Exception as e:
            sql = f"-- Error generating SQL: {e}\nSELECT 'Error: {str(e)}' as error;"

        return {
            'sql': sql,
            'tables_used': tables_used,
            'intent': intent,
            'confidence': 0.8 if sql and not sql.startswith('--') else 0.3,
        }

    def _get_columns(self, table):
        """Get actual column info from catalog for a table."""
        profile = self.catalog.tables.get(table, {}) if self.catalog else {}
        return profile.get('columns', [])

    def _find_col(self, table, *hints):
        """Find actual column name matching hints (keyword matching)."""
        cols = self._get_columns(table)
        for hint in hints:
            hint_lower = hint.lower()
            for c in cols:
                cn = (c.get('column_name') or c.get('name') or '').lower()
                if hint_lower == cn or hint_lower in cn:
                    return c.get('column_name', c.get('name', ''))
            for c in cols:
                if (c.get('healthcare_type') or '').lower() == hint_lower:
                    return c.get('column_name', c.get('name', ''))
            for c in cols:
                if (c.get('semantic_type') or '').lower() == hint_lower:
                    return c.get('column_name', c.get('name', ''))
        return None

    def _find_numeric_col(self, table):
        """Find a numeric/currency column for aggregation."""
        cols = self._get_columns(table)
        for c in cols:
            st = (c.get('semantic_type') or '').lower()
            cn = (c.get('column_name') or '').lower()
            if st in ('currency', 'float', 'integer', 'numeric') or \
               any(k in cn for k in ['amount', 'cost', 'paid', 'billed', 'price', 'charge', 'score']):
                return c.get('column_name', c.get('name', ''))
        return None

    def _find_date_col(self, table):
        """Find a date column."""
        return self._find_col(table, 'service_date', 'date', 'admit_date', 'encounter_date',
                              'prescription_date', 'referral_date', 'enrollment_date')

    def _find_category_col(self, table, question=''):
        """Find best category column based on question keywords."""
        q = question.lower()
        # Try specific matches first
        if 'member' in q or 'patient' in q:
            c = self._find_col(table, 'member_id', 'mrn')
            if c: return c
        if 'region' in q:
            c = self._find_col(table, 'kp_region', 'region')
            if c: return c
        if 'provider' in q or 'npi' in q or 'doctor' in q:
            c = self._find_col(table, 'rendering_npi', 'npi', 'provider')
            if c: return c
        if 'status' in q:
            c = self._find_col(table, 'status', 'claim_status', 'encounter_status')
            if c: return c
        if 'type' in q or 'visit' in q:
            c = self._find_col(table, 'visit_type', 'claim_type', 'plan_type', 'type')
            if c: return c
        if 'facility' in q or 'hospital' in q:
            c = self._find_col(table, 'facility')
            if c: return c
        if 'department' in q or 'specialty' in q:
            c = self._find_col(table, 'department', 'specialty')
            if c: return c
        if 'gender' in q:
            c = self._find_col(table, 'gender')
            if c: return c
        if 'diagnos' in q or 'icd' in q:
            c = self._find_col(table, 'icd10', 'primary_diagnosis', 'diagnosis')
            if c: return c
        # Default: first string column with reasonable cardinality
        cols = self._get_columns(table)
        for c in cols:
            st = (c.get('semantic_type') or '').lower()
            ht = (c.get('healthcare_type') or '').lower()
            cn = (c.get('column_name') or c.get('name') or '').lower()
            if st in ('string', 'category') and ht not in ('member_id', 'mrn') and \
               'id' not in cn and 'date' not in cn and 'description' not in cn:
                return c.get('column_name', c.get('name', ''))
        return 'KP_REGION'

    def _extract_number(self, question, default=10):
        """Extract a number from question."""
        m = re.search(r'\b(\d+)\b', question)
        return int(m.group(1)) if m else default

    def _generate_count(self, question: str, tables: List[str]) -> str:
        table = tables[0] if tables else 'encounters'
        # Check if counting by a category
        cat = self._find_category_col(table, question)
        q = question.lower()
        if 'by' in q or 'per' in q or 'each' in q:
            return f"SELECT {cat}, COUNT(*) as count FROM {table} GROUP BY {cat} ORDER BY count DESC;"
        return f"SELECT COUNT(*) as total_count FROM {table};"

    def _generate_aggregate(self, question: str, intent_info: Dict[str, Any],
                           tables: List[str]) -> str:
        table = tables[0] if tables else 'claims'
        q = question.lower()
        # Detect aggregation function
        agg = 'AVG'
        if 'sum' in q or 'total' in q: agg = 'SUM'
        elif 'max' in q or 'highest' in q or 'maximum' in q: agg = 'MAX'
        elif 'min' in q or 'lowest' in q or 'minimum' in q: agg = 'MIN'
        elif 'average' in q or 'avg' in q or 'mean' in q: agg = 'AVG'
        # Find numeric column from question keywords
        num_col = None
        for word in q.split():
            c = self._find_col(table, word)
            if c:
                cols = self._get_columns(table)
                for col in cols:
                    if col.get('column_name', '') == c and col.get('semantic_type', '') in ('currency', 'float', 'integer', 'numeric'):
                        num_col = c
                        break
            if num_col:
                break
        if not num_col:
            num_col = self._find_numeric_col(table) or 'PAID_AMOUNT'
        # Check for GROUP BY
        cat = self._find_category_col(table, question)
        if 'by' in q or 'per' in q:
            return f"SELECT {cat}, {agg}(CAST({num_col} AS REAL)) as {agg.lower()}_{num_col.lower()} FROM {table} GROUP BY {cat} ORDER BY {agg.lower()}_{num_col.lower()} DESC;"
        return f"SELECT {agg}(CAST({num_col} AS REAL)) as {agg.lower()}_{num_col.lower()} FROM {table};"

    def _generate_trend(self, question: str, tables: List[str],
                       intent_info: Dict[str, Any]) -> str:
        table = tables[0] if tables else 'encounters'
        date_col = self._find_date_col(table) or 'SERVICE_DATE'
        return (
            f"SELECT SUBSTR({date_col}, 1, 7) as month, COUNT(*) as count "
            f"FROM {table} WHERE {date_col} IS NOT NULL AND {date_col} != '' "
            f"GROUP BY SUBSTR({date_col}, 1, 7) ORDER BY month;"
        )

    def _generate_breakdown(self, question: str, tables: List[str]) -> str:
        table = tables[0] if tables else 'encounters'
        cat = self._find_category_col(table, question)
        return f"SELECT {cat}, COUNT(*) as count FROM {table} GROUP BY {cat} ORDER BY count DESC;"

    def _generate_top_n(self, question: str, tables: List[str],
                       intent_info: Dict[str, Any]) -> str:
        table = tables[0] if tables else 'claims'
        n = self._extract_number(question, 10)
        q = question.lower()
        cat = self._find_category_col(table, question)
        num_col = self._find_numeric_col(table)
        if num_col and ('amount' in q or 'cost' in q or 'paid' in q or 'billed' in q):
            return (f"SELECT {cat}, SUM(CAST({num_col} AS REAL)) as total "
                    f"FROM {table} GROUP BY {cat} ORDER BY total DESC LIMIT {n};")
        return (f"SELECT {cat}, COUNT(*) as count "
                f"FROM {table} GROUP BY {cat} ORDER BY count DESC LIMIT {n};")

    def _generate_distinct(self, question: str, tables: List[str]) -> str:
        table = tables[0] if tables else 'encounters'
        cat = self._find_category_col(table, question)
        return f"SELECT DISTINCT {cat}, COUNT(*) as count FROM {table} GROUP BY {cat} ORDER BY count DESC;"

    def _generate_negation(self, question: str, tables: List[str]) -> str:
        table = tables[0] if tables else 'claims'
        q = question.lower()
        # Try to find what's being negated
        if 'denied' in q or 'rejected' in q:
            status_col = self._find_col(table, 'claim_status', 'status')
            if status_col:
                return (f"SELECT CLAIM_ID, MEMBER_ID, SERVICE_DATE, CPT_CODE, CPT_DESCRIPTION, "
                        f"BILLED_AMOUNT, PAID_AMOUNT, {status_col}, DENIAL_REASON, KP_REGION "
                        f"FROM {table} WHERE {status_col} = 'DENIED' ORDER BY SERVICE_DATE DESC LIMIT 50;")
        if 'not paid' in q or 'unpaid' in q or 'failed' in q:
            status_col = self._find_col(table, 'claim_status', 'status')
            if status_col:
                return (f"SELECT CLAIM_ID, MEMBER_ID, SERVICE_DATE, CPT_CODE, CPT_DESCRIPTION, "
                        f"BILLED_AMOUNT, PAID_AMOUNT, {status_col}, DENIAL_REASON, KP_REGION "
                        f"FROM {table} WHERE {status_col} != 'PAID' ORDER BY SERVICE_DATE DESC LIMIT 50;")
        if 'null' in q or 'missing' in q or 'empty' in q:
            cat = self._find_category_col(table, question)
            return f"SELECT COUNT(*) as null_count FROM {table} WHERE {cat} IS NULL OR {cat} = '';"
        return f"SELECT * FROM {table} WHERE 1=1 LIMIT 100;"

    def _generate_threshold(self, question: str, tables: List[str],
                           intent_info: Dict[str, Any]) -> str:
        table = tables[0] if tables else 'claims'
        num_col = self._find_numeric_col(table) or 'PAID_AMOUNT'
        cat = self._find_category_col(table, question)
        threshold = self._extract_number(question, 1000)
        return (f"SELECT {cat}, COUNT(*) as count, SUM(CAST({num_col} AS REAL)) as total "
                f"FROM {table} GROUP BY {cat} HAVING COUNT(*) > {threshold} ORDER BY total DESC;")

    def _generate_anomaly(self, question: str, tables: List[str]) -> str:
        table = tables[0] if tables else 'claims'
        num_col = self._find_numeric_col(table) or 'PAID_AMOUNT'
        return (
            f"SELECT *, CAST({num_col} AS REAL) as val "
            f"FROM {table} "
            f"WHERE CAST({num_col} AS REAL) > "
            f"(SELECT AVG(CAST({num_col} AS REAL)) + 2 * "
            f"CASE WHEN COUNT(*) > 1 THEN "
            f"SQRT(SUM((CAST({num_col} AS REAL) - (SELECT AVG(CAST({num_col} AS REAL)) FROM {table})) * "
            f"(CAST({num_col} AS REAL) - (SELECT AVG(CAST({num_col} AS REAL)) FROM {table}))) / (COUNT(*)-1)) "
            f"ELSE 0 END FROM {table}) "
            f"LIMIT 50;"
        )

    def _generate_percentage(self, question: str, tables: List[str]) -> str:
        table = tables[0] if tables else 'encounters'
        cat = self._find_category_col(table, question)
        return (
            f"SELECT {cat}, COUNT(*) as count, "
            f"ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM {table}), 2) as pct "
            f"FROM {table} GROUP BY {cat} ORDER BY pct DESC;"
        )

    def _generate_latest(self, question: str, tables: List[str]) -> str:
        table = tables[0] if tables else 'encounters'
        date_col = self._find_date_col(table) or 'SERVICE_DATE'
        return f"SELECT * FROM {table} ORDER BY {date_col} DESC LIMIT 1;"

    def _generate_lookup(self, question: str, tables: List[str]) -> str:
        table = tables[0] if tables else 'encounters'
        # Select key columns instead of SELECT * for readability
        key_cols = {
            'claims': 'CLAIM_ID, MEMBER_ID, SERVICE_DATE, CPT_CODE, CPT_DESCRIPTION, BILLED_AMOUNT, PAID_AMOUNT, CLAIM_STATUS, KP_REGION',
            'members': 'MEMBER_ID, FIRST_NAME, LAST_NAME, DATE_OF_BIRTH, GENDER, KP_REGION, PLAN_TYPE',
            'providers': 'NPI, PROVIDER_FIRST_NAME, PROVIDER_LAST_NAME, SPECIALTY, DEPARTMENT, KP_REGION',
            'encounters': 'ENCOUNTER_ID, MEMBER_ID, SERVICE_DATE, VISIT_TYPE, DEPARTMENT, FACILITY, PRIMARY_DIAGNOSIS',
            'diagnoses': 'MEMBER_ID, ICD10_CODE, ICD10_DESCRIPTION, DIAGNOSIS_TYPE, DIAGNOSIS_DATE, SEVERITY',
            'prescriptions': '*',
            'referrals': '*',
        }
        cols = key_cols.get(table, '*')
        return f"SELECT {cols} FROM {table} LIMIT 20;"


# ==================== LOCAL SQL EXECUTION ====================

def _execute_sql_local(sql: str, cfg: Dict[str, str], timeout: int = 30) -> Tuple[bool, List, str]:
    """
    Execute SQL against the persistent database (production mode) or
    fall back to in-memory CSV loading (legacy mode).

    Production mode uses DatabasePool singleton — data loaded ONCE at startup.
    Legacy mode rebuilds from CSVs on each call (for backward compatibility).

    Args:
        sql: SQL query string (SELECT only)
        cfg: Configuration dict with data_dir
        timeout: Query timeout in seconds

    Returns:
        Tuple of (success, rows_as_dicts, error_message)
    """
    logger = get_logger('mtp.sql') if HAS_PRODUCTION else None

    # ── DATABRICKS MODE: Route through DataSourceManager if Databricks connected ──
    if HAS_DATABRICKS:
        try:
            ds_mgr = DataSourceManager.get_instance()
            if ds_mgr.is_databricks:
                def _local_fallback(sql_fb, cfg_fb):
                    """Local execution fallback when Databricks fails."""
                    return _execute_sql_local_impl(sql_fb, cfg_fb, timeout, logger)

                return ds_mgr.execute_query(sql, local_fallback_fn=_local_fallback, cfg=cfg)
        except Exception as e:
            if logger:
                logger.warning("DataSourceManager error, falling back: %s", e)

    return _execute_sql_local_impl(sql, cfg, timeout, logger)


def _execute_sql_local_impl(sql: str, cfg: Dict[str, str], timeout: int = 30,
                             logger=None) -> Tuple[bool, List, str]:
    """Internal: execute SQL against local persistent DB or in-memory CSV."""

    # ── PRODUCTION MODE: Use persistent DatabasePool ──
    if HAS_PRODUCTION:
        try:
            pool = DatabasePool.get_instance()
            if not pool._loaded:
                resolved_cfg = resolve_config(cfg)
                pool.initialize(resolved_cfg)

            # Validate SQL before execution
            allowed_tables = pool.get_table_names()
            is_valid, err_msg = validate_sql_output(sql, allowed_tables)
            if not is_valid:
                if logger:
                    logger.warning("SQL validation failed: %s — %s", err_msg, sql[:200])
                return False, [], f"SQL validation: {err_msg}"

            return pool.execute_query(sql, timeout)

        except Exception as e:
            if logger:
                logger.error("Production DB error, falling back to legacy: %s", e)
            # Fall through to legacy mode

    # ── LEGACY MODE: In-memory CSV loading (backward compatibility) ──
    try:
        conn = sqlite3.connect(':memory:')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        base_dir = cfg.get('BASE_DIR', str(Path(__file__).parent.parent))
        data_path = cfg.get('RAW_DIR', os.path.join(base_dir, 'data', 'raw'))

        if not os.path.exists(data_path):
            return False, [], f"Data directory not found: {data_path}"

        loaded_tables = set()
        for filename in os.listdir(data_path):
            if filename.endswith('.csv'):
                table_name = filename.replace('.csv', '')
                try:
                    csv_path = os.path.join(data_path, filename)
                    with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
                        reader = csv.DictReader(f)
                        columns = reader.fieldnames
                        if not columns:
                            continue

                        col_defs = ', '.join([f'"{col}" TEXT' for col in columns])
                        cursor.execute(f'CREATE TABLE "{table_name}" ({col_defs})')

                        placeholders = ', '.join(['?' for _ in columns])
                        insert_sql = f'INSERT INTO "{table_name}" VALUES ({placeholders})'
                        batch = []
                        for row in reader:
                            batch.append([row.get(col, '') for col in columns])
                            if len(batch) >= 5000:
                                cursor.executemany(insert_sql, batch)
                                batch = []
                        if batch:
                            cursor.executemany(insert_sql, batch)

                        loaded_tables.add(table_name)
                        conn.commit()

                except Exception as e:
                    if logger:
                        logger.warning("Failed to load CSV %s: %s", filename, e)

        if not loaded_tables:
            return False, [], f"No CSV files loaded from {data_path}"

        cursor.execute(sql)
        raw_rows = cursor.fetchall()

        if raw_rows and hasattr(raw_rows[0], 'keys'):
            rows = [dict(r) for r in raw_rows]
        else:
            rows = raw_rows

        conn.close()
        return True, rows, ""

    except sqlite3.Error as e:
        return False, [], f"SQL Error: {str(e)}"
    except Exception as e:
        return False, [], f"Execution Error: {str(e)}"


# ==================== PREVIEW DATA GENERATION ====================

def _generate_preview_data(sql: str, catalog: SemanticCatalog, tables_used: List[str]) -> str:
    """
    Generate preview data from catalog metadata.

    Args:
        sql: Generated SQL
        catalog: SemanticCatalog instance
        tables_used: List of tables used

    Returns:
        CSV string of preview data
    """
    if not tables_used or not catalog:
        return "No tables available for preview"

    table_name = tables_used[0]
    table_info = catalog.tables.get(table_name, {})
    columns = table_info.get('columns', [])

    if not columns:
        return f"No column metadata for {table_name}"

    # Generate mock rows
    rows = []
    for i in range(3):
        row = {}
        for col in columns[:5]:  # First 5 columns
            col_name = col.get('column_name', col.get('name', 'col'))
            col_type = col.get('semantic_type', col.get('type', 'string')).lower()
            if col_type in ('int', 'integer', 'numeric'):
                row[col_name] = str(100 + i)
            elif col_type in ('float', 'decimal', 'currency'):
                row[col_name] = str(99.99 + i)
            else:
                tvs = col.get('top_values', [])
                if tvs and i < len(tvs):
                    row[col_name] = str(tvs[i].get('value', f"Value_{i}"))
                else:
                    row[col_name] = f"Value_{i}"
        rows.append(row)

    # Format as CSV
    col_names = [c.get('column_name', c.get('name', 'col')) for c in columns[:5]]
    col_names = [c for c in col_names if c]  # Filter None
    csv_lines = [','.join(col_names)]
    for row in rows:
        csv_lines.append(','.join([str(row.get(c, '')) for c in col_names]))

    return '\n'.join(csv_lines)


# ==================== LINEAGE TRACKER WRAPPER ====================

def _get_lineage_info(question: str, tables_used: List[str]) -> Dict[str, Any]:
    """Wrapper to use LineageTracker if available."""
    if LineageTracker is None:
        return {
            'source_tables': tables_used,
            'lineage': f"Question → {' → '.join(tables_used)}"
        }

    try:
        tracker = LineageTracker()
        lineage = tracker.track(question, tables_used)
        return lineage
    except Exception as e:
        return {'error': str(e), 'source_tables': tables_used}


# ==================== ANALYTICS ADVISOR WRAPPER ====================

def _get_analytics_recommendations(question: str, tables_used: List[str]) -> Dict[str, Any]:
    """Wrapper to use AnalyticsAdvisor if available."""
    if AnalyticsAdvisor is None:
        return {'recommendations': []}

    try:
        advisor = AnalyticsAdvisor()
        recommendations = advisor.recommend(question, tables_used)
        return recommendations
    except Exception as e:
        return {'error': str(e), 'recommendations': []}


# ==================== ENGINE SINGLETON CACHE ====================
# Engines are expensive to initialize (model training, TF-IDF fitting, graph building).
# Cache them as module-level singletons for reuse across queries.

_engine_cache: Dict[str, Any] = {}
_engine_cache_lock = __import__('threading').Lock()


def clear_engine_cache():
    """Clear all cached engines (ML, NLP, Scale) and their query caches.
    Call this on dashboard restart or when data source changes."""
    global _engine_cache
    with _engine_cache_lock:
        # Clear query caches inside engines before discarding
        for key, engine in _engine_cache.items():
            if hasattr(engine, 'query_cache'):
                try:
                    engine.query_cache.cache.clear()
                except Exception:
                    pass
        _engine_cache.clear()


def _get_cached_engine(key: str, factory, *args, **kwargs):
    """Get or create a cached engine instance (thread-safe)."""
    if key not in _engine_cache:
        with _engine_cache_lock:
            if key not in _engine_cache:
                try:
                    _engine_cache[key] = factory(*args, **kwargs)
                except Exception:
                    return None
    return _engine_cache.get(key)


# ==================== MAIN QUESTION PROCESSING ====================

def process_question(question: str, cfg: Dict[str, str],
                    catalog: 'SemanticCatalog' = None,
                    session: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Main entry point: process natural language question end-to-end.

    Production pipeline:
        Input Validation → Question → ConversationalHandler → LineageTracker →
        AnalyticsAdvisor → MetadataEngine → JourneyBuilder →
        [IR + ML + NLP + LLM] → SQL Generation → SQL Validation →
        Persistent DB Execution → Result Validation → Audit Log

    Args:
        question: Natural language question
        cfg: Configuration dict
        catalog: SemanticCatalog instance (auto-created if None)
        session: Optional session dict for history/state

    Returns:
        Dict with:
            - success: bool
            - question: str (original)
            - sql: str
            - answer: str or List (result rows)
            - tables_used: List[str]
            - intent: str
            - engine_mode: str (conversational|metadata|journey|sql)
            - is_metadata: bool
    """
    _t0 = time.time()
    _logger = get_logger('mtp.engine') if HAS_PRODUCTION else None

    # ── INPUT VALIDATION (production mode) ──
    if HAS_PRODUCTION:
        is_valid, sanitized, err_msg = validate_question(question)
        if not is_valid:
            if _logger:
                _logger.warning("Invalid input rejected: %s", err_msg)
            return {
                'success': False, 'question': question, 'sql': None,
                'answer': f'Invalid query: {err_msg}', 'tables_used': [],
                'intent': None, 'engine_mode': 'validation_error',
                'is_metadata': False, 'confidence': 0.0,
            }
        question = sanitized

    if catalog is None:
        catalog = SemanticCatalog()

    result = {
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

    # 0b. SESSION CONTEXT — conversation chaining, duplicate detection, hierarchy
    _session_mgr = None
    _learning_engine = None
    _catalog_registry = None
    _context_info = {}  # will hold followup/duplicate info for the result
    _original_question = question  # preserve original before any merging

    try:
        from session_context import (
            SessionContextManager, ConversationChainer,
            DuplicateDetector, TableHierarchyResolver
        )
        _session_mgr = SessionContextManager()
    except ImportError:
        pass

    try:
        from catalog_registry import CatalogRegistry, ContextualLearningEngine
        _catalog_registry = CatalogRegistry.get_instance()
        if not _catalog_registry._initialized:
            _cat_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', 'data', 'catalog_config.json')
            _catalog_registry.initialize(config_path=_cat_cfg)
        _learning_engine = ContextualLearningEngine()
    except ImportError:
        pass

    # Session IDs from the session dict
    _user_id = session.get('user_id', 'anonymous') if session else 'anonymous'
    _session_id = session.get('session_id', 'default') if session else 'default'

    if _session_mgr:
        try:
            # Get recent turns for this session
            recent_turns = _session_mgr.get_recent_turns(_user_id, _session_id, count=10)

            # DUPLICATE DETECTION — have they asked this before?
            dup = DuplicateDetector.check(question, recent_turns, threshold=0.85)
            if dup:
                result['duplicate_warning'] = dup
                _context_info['duplicate'] = dup

            # CONVERSATION CHAINING — is this a follow-up?
            chain = ConversationChainer.analyze(question, recent_turns)
            if chain.get('is_followup') and chain.get('carry_forward'):
                _context_info['followup'] = chain
                result['followup_info'] = chain
                # Merge the question with context for a complete standalone query
                merged = ConversationChainer.merge_context(
                    question, chain['carry_forward'], chain['followup_type']
                )
                if merged and merged != question:
                    result['merged_question'] = merged
                    question = merged  # Use merged question for SQL generation

            # TABLE HIERARCHY — detect specific table type from question
            hierarchy_match = TableHierarchyResolver.detect_from_question(question)
            if hierarchy_match and hierarchy_match.get('filter'):
                _context_info['hierarchy_match'] = hierarchy_match
                result['hierarchy_match'] = hierarchy_match

            # ACTIVE SELECTIONS — check if user has made table selections in the UI
            active_selections = _session_mgr.get_state(_user_id, _session_id, 'active_selections')
            if active_selections:
                _context_info['active_selections'] = active_selections
                result['active_selections'] = active_selections
        except Exception:
            pass

    # Check if catalog discovery is needed (multi-catalog disambiguation)
    # Skip catalog discovery when session says so (e.g., dashboard always uses local)
    _skip_discovery = session.get('skip_catalog_discovery', False) if session else False
    if _catalog_registry and _catalog_registry._initialized and not _skip_discovery:
        discovery = _catalog_registry.discover_for_query(question)
        if discovery.get('needs_clarification') and discovery.get('clarification_prompt'):
            prompt = discovery['clarification_prompt']
            result['answer'] = prompt
            result['success'] = True
            result['engine_mode'] = 'catalog_discovery'
            result['is_metadata'] = True
            result['catalog_discovery'] = discovery
            return result

    # 0c. CONTEXTUAL SUGGESTIONS — attach similar queries from other users
    _similar_queries = []
    if _learning_engine:
        try:
            _similar_queries = _learning_engine.get_similar_queries(question, limit=3)
        except Exception:
            pass

    # 1. CONVERSATIONAL HANDLER
    if ConversationalHandler.is_conversational(question):
        result['answer'] = ConversationalHandler.handle(question)
        result['success'] = True
        result['engine_mode'] = 'conversational'
        return result

    # 2. METADATA ENGINE
    if MetadataEngine.is_metadata_question(question):
        result['answer'] = MetadataEngine.answer(question, catalog)
        result['success'] = True
        result['engine_mode'] = 'metadata'
        result['is_metadata'] = True
        return result

    # 3. LINEAGE TRACKING (if available)
    lineage_info = _get_lineage_info(question, [])

    # 4. ANALYTICS RECOMMENDATIONS (if available)
    recommendations = _get_analytics_recommendations(question, [])

    # 5. JOURNEY BUILDER
    journey_type = JourneyBuilder.detect_journey(question)
    tables_for_journey = []
    if journey_type:
        tables_for_journey = JourneyBuilder.build_journey(journey_type, catalog)
        result['intent'] = journey_type
        result['engine_mode'] = 'journey'

    # 5b. ML ENGINE — Ensemble intent classification + semantic column resolution
    ml_meta = {}
    if MLEngine:
        try:
            def _init_ml():
                ml = MLEngine()
                ml.initialize(
                    column_names=catalog.get_all_column_names() if hasattr(catalog, 'get_all_column_names') else [],
                    enable_llm=False
                )
                return ml

            _ml = _get_cached_engine('ml_engine', _init_ml)
            if _ml:
                ml_intent, ml_conf, ml_details = _ml.classify_intent(question)
                ml_meta = {
                    'ml_intent': ml_intent,
                    'ml_confidence': ml_conf,
                    'ml_models_used': list(ml_details.get('models', {}).keys()),
                    'ml_method': ml_details.get('method', 'unknown'),
                }
                if not result['intent']:
                    result['intent'] = ml_intent
                result['ml_meta'] = ml_meta
        except Exception:
            pass

    # 5b-numpy. HYBRID NLP ENGINE — NumPy-powered TF-IDF + cosine similarity
    nlp_meta = {}
    if HybridNLPEngine:
        try:
            _catalog_dir = catalog.catalog_dir if hasattr(catalog, 'catalog_dir') else None
            _nlp = _get_cached_engine('nlp_engine', HybridNLPEngine, _catalog_dir)
            if not _nlp:
                _nlp = HybridNLPEngine(_catalog_dir)
            nlp_analysis = _nlp.analyze_question(question)

            nlp_meta = {
                'nlp_intent': nlp_analysis.get('intent', ''),
                'nlp_confidence': nlp_analysis.get('intent_confidence', 0),
                'nlp_tables': nlp_analysis.get('relevant_tables', []),
                'nlp_top_column': '',
                'nlp_method': nlp_analysis.get('method', 'numpy_hybrid'),
            }
            cols = nlp_analysis.get('relevant_columns', [])
            if cols:
                top = cols[0]
                nlp_meta['nlp_top_column'] = f"{top['table']}.{top['column']}"

            # Use NLP intent as secondary signal when ML engine had low confidence
            if not result['intent']:
                result['intent'] = nlp_analysis['intent']

            result['nlp_meta'] = nlp_meta
        except Exception:
            pass

    # 5c. SCALE ENGINE — Knowledge Graph, Vector Search, Query Cache, Databricks
    scale_meta = {}
    if ScaleEngine:
        try:
            def _init_scale():
                s = ScaleEngine(catalog.catalog_dir if hasattr(catalog, 'catalog_dir') else None)
                s._init_stats = s.initialize()
                return s

            _scale = _get_cached_engine('scale_engine', _init_scale)
            if not _scale:
                raise RuntimeError("Scale engine init failed")
            _scale_stats = getattr(_scale, '_init_stats', {})

            # Check query cache first (semantic near-duplicate detection)
            cached = _scale.query_cache.get(question)
            if cached:
                cached_answer = cached.get('answer', [])
                # Only use cache if it has actual data rows
                if isinstance(cached_answer, list) and len(cached_answer) > 0:
                    result.update(cached)
                    result['engine_mode'] = 'cached'
                    result['success'] = True
                    result['scale_meta'] = {'cache_hit': True}
                    return result
                # Stale/empty cache entry — skip it

            # Vector search for relevant schema elements
            schema_hits = _scale.search_schema(question, k=5)

            # Knowledge graph reasoning
            query_words = [w for w in question.lower().split() if len(w) > 2]
            subgraph = _scale.get_relevant_subgraph(query_words)

            # Get optimization suggestions
            opt_suggestions = []

            scale_meta = {
                'vector_search_hits': len(schema_hits),
                'top_vector_hit': schema_hits[0]['id'] if schema_hits else None,
                'subgraph_nodes': subgraph.get('nodes', 0),
                'subgraph_tables': subgraph.get('tables', []),
                'cache_hit': False,
                'kg_nodes': _scale_stats['knowledge_graph']['total_nodes'],
                'kg_edges': _scale_stats['knowledge_graph']['total_edges'],
            }
            result['scale_meta'] = scale_meta
        except Exception:
            pass

    # 6. DYNAMIC SQL ENGINE (primary) — schema-aware, multi-table, auto-JOIN
    try:
        from dynamic_sql_engine import DynamicSQLEngine
        dyn_engine = DynamicSQLEngine(catalog.catalog_dir)
        dyn_result = dyn_engine.generate(question)

        # Handle multi-question results
        is_multi = dyn_result.get('multi_query', False)
        all_sub_results = dyn_result.get('all_results', [])

        if is_multi and all_sub_results:
            # Execute each sub-query separately and combine results
            combined_answers = []
            combined_sql_parts = []
            all_tables = []
            for i, sub_r in enumerate(all_sub_results):
                sub_sql = sub_r.get('sql', '')
                combined_sql_parts.append(sub_sql)
                if sub_sql and not sub_sql.startswith('--'):
                    success, rows, error = _execute_sql_local(sub_sql, cfg)
                    sub_answer = {
                        'question_number': i + 1,
                        'sql': sub_sql,
                        'data': rows if success and rows else [],
                        'tables': sub_r.get('tables_used', []),
                    }
                    combined_answers.append(sub_answer)
                    for t in sub_r.get('tables_used', []):
                        if t not in all_tables:
                            all_tables.append(t)

            result['sql'] = ';\n'.join(combined_sql_parts)
            result['tables_used'] = all_tables
            result['confidence'] = min(r.get('confidence', 0.5) for r in all_sub_results)
            result['intent'] = 'multi_query'
            result['answer'] = combined_answers
            result['success'] = True
            result['engine_mode'] = 'dynamic'
            result['multi_query'] = True
            return result
        else:
            result['sql'] = dyn_result.get('sql')
            result['tables_used'] = dyn_result.get('tables_used', [])
            result['confidence'] = dyn_result.get('confidence', 0.5)
            result['intent'] = dyn_result.get('agg_info', {}).get('agg_func', 'lookup')

            # Determine intent label from dynamic engine results
            agg_func = dyn_result.get('agg_info', {}).get('agg_func')
            filters = dyn_result.get('filters', [])
            if agg_func == 'COUNT':
                result['intent'] = 'count'
            elif agg_func in ('AVG', 'SUM', 'MAX', 'MIN'):
                result['intent'] = 'aggregate'
            elif any('DENIED' in f or 'denied' in f for f in filters):
                result['intent'] = 'negation'
            elif dyn_result.get('agg_info', {}).get('top_n'):
                result['intent'] = 'top_n'
            else:
                result['intent'] = 'lookup'

            # Execute the generated SQL
            if result['sql'] and not result['sql'].startswith('--'):
                success, rows, error = _execute_sql_local(result['sql'], cfg)
                if success:
                    result['answer'] = rows if rows else []
                    result['success'] = True
                    result['engine_mode'] = 'dynamic'

                # NLP Engine: validate results using pandas
                if HybridNLPEngine and nlp_meta and rows:
                    try:
                        validation = _nlp.validate_results(rows, question, result['sql'])
                        nlp_meta['validation'] = validation
                        result['nlp_meta'] = nlp_meta
                    except Exception:
                        pass

                # Scale Engine: cache result + generate Databricks SQL
                if ScaleEngine and scale_meta:
                    try:
                        if DatabricksDialect:
                            result['databricks_sql'] = DatabricksDialect.translate(
                                result['sql'], tables=result['tables_used'], use_catalog=True
                            )
                        # Only cache if we have actual data rows
                        if isinstance(rows, list) and len(rows) > 0:
                            _scale.query_cache.put(question, {
                                'sql': result['sql'], 'answer': result['answer'],
                                'tables_used': result['tables_used'],
                                'intent': result['intent'], 'success': True,
                            })
                        _scale.pattern_miner.log_query(
                            question, result['sql'], dyn_result.get('agg_info', {}),
                            result['tables_used']
                        )
                    except Exception:
                        pass

                # Log to contextual learning engine before returning
                if _learning_engine:
                    try:
                        _u_id = session.get('user_id', 'anonymous') if session else 'api'
                        _u_name = session.get('username', 'anonymous') if session else 'api'
                        _row_cnt = len(result.get('answer', [])) if isinstance(result.get('answer'), list) else 0
                        _learning_engine.log_query(
                            user_id=_u_id, username=_u_name, question=question,
                            sql=result.get('sql', ''), tables=result.get('tables_used', []),
                            intent=result.get('intent', ''), success=result.get('success', False),
                            result_count=_row_cnt, execution_time_ms=int((time.time() - _t0) * 1000),
                        )
                    except Exception:
                        pass
                # Attach similar queries
                if _similar_queries:
                    result['similar_queries'] = _similar_queries

                return result
            # If dynamic SQL execution failed (DB error), fall through to legacy
            if not success:
                result['_dynamic_error'] = error

    except Exception as dyn_err:
        result['_dynamic_error'] = str(dyn_err)

    # 7. LEGACY FALLBACK — template-based SQL generation
    intent_classifier = IntentClassifier if IntentClassifier else None
    intent_parser = IntentParser(intent_classifier)
    intent_info = intent_parser.parse(question)

    if not result['intent']:
        result['intent'] = intent_info.get('intent', 'lookup')

    tables_used = catalog.find_tables(question) or tables_for_journey
    if not tables_used and catalog.get_all_table_names():
        tables_used = [catalog.get_all_table_names()[0]]

    result['tables_used'] = tables_used

    entity_extractor = EntityExtractor if EntityExtractor else None
    ir_engine = None
    if IREngine:
        try:
            ir_engine = IREngine()
            ir_engine.load_catalog(catalog.catalog_dir)
        except Exception:
            ir_engine = None

    sql_generator = SQLGenerator(ir_engine, entity_extractor, catalog)
    sql_result = sql_generator.generate(question, intent_info, catalog)

    result['sql'] = sql_result.get('sql')
    result['confidence'] = sql_result.get('confidence', 0.5)

    # Execute legacy SQL
    if result['sql'] and not result['sql'].startswith('--'):
        success, rows, error = _execute_sql_local(result['sql'], cfg)
        if success:
            result['answer'] = rows
            result['success'] = True
            result['engine_mode'] = 'legacy'
        else:
            result['answer'] = f"Execution Error: {error}"
            result['success'] = False
    else:
        preview = _generate_preview_data(result['sql'], catalog, tables_used)
        result['answer'] = preview
        result['success'] = True
        result['engine_mode'] = 'preview'

    # ── CONTEXTUAL LEARNING: Log query + attach suggestions ──
    elapsed_ms = (time.time() - _t0) * 1000
    if _learning_engine:
        try:
            _user_id = session.get('user_id', 'anonymous') if session else 'api'
            _username = session.get('username', 'anonymous') if session else 'api'
            row_count_learn = len(result.get('answer', [])) if isinstance(result.get('answer'), list) else 0
            _learning_engine.log_query(
                user_id=_user_id,
                username=_username,
                question=question,
                sql=result.get('sql', ''),
                tables=result.get('tables_used', []),
                intent=result.get('intent', ''),
                success=result.get('success', False),
                result_count=row_count_learn,
                execution_time_ms=int(elapsed_ms),
            )
        except Exception:
            pass

    # Attach similar queries and suggestions to result
    if _similar_queries:
        result['similar_queries'] = _similar_queries
    if _learning_engine:
        try:
            _user_id = session.get('user_id', 'anonymous') if session else 'api'
            result['suggestions'] = _learning_engine.get_suggestions(
                user_id=_user_id, limit=5
            )
        except Exception:
            pass

    # ── PRODUCTION: Audit log + metrics ──
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
            }
        )

    return result


# ==================== INTERACTIVE MODE ====================

def interactive_mode(cfg: Dict[str, str]):
    """
    Interactive REPL for chatbot.

    Commands:
        tables - list all tables
        relationships - list relationships
        describe <table> - describe table
        history - show question history
        lineage - show data lineage
        analytics - show analytics recommendations
        recommend - get recommendations
        feedback - provide feedback
        quit - exit

    Args:
        cfg: Configuration dict
    """
    catalog = SemanticCatalog()
    session = {
        'username': input("Enter your name: ").strip() or "User",
        'history': [],
        'start_time': datetime.now(),
    }

    # Initialize advanced components
    try:
        from advanced_ds import LRUCache, BloomFilter, Trie, MetricsCollector, EventStore, CircuitBreaker
        query_cache = LRUCache(capacity=128, ttl_seconds=300)
        table_filter = BloomFilter(expected_items=500)
        sql_trie = Trie()
        metrics = MetricsCollector()
        events = EventStore()
        sql_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30)

        # Populate bloom filter and trie with catalog info
        for tname in catalog.get_all_table_names():
            table_filter.add(tname)
            sql_trie.insert(tname, 'table')
        for kw in ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'COUNT', 'AVG', 'SUM', 'MAX', 'MIN', 'JOIN', 'HAVING', 'LIMIT', 'DISTINCT', 'BETWEEN', 'LIKE', 'IN', 'AND', 'OR']:
            sql_trie.insert(kw.lower(), 'keyword')
        has_advanced = True
    except ImportError:
        has_advanced = False
        query_cache = None

    # Initialize recommendation engine
    try:
        from recommendation_engine import RecommendationEngine, load_csv_to_sqlite
        raw_dir = cfg.get('RAW_DIR', os.path.join(cfg.get('DATA_DIR', '../data'), 'raw'))
        rec_conn = load_csv_to_sqlite(raw_dir)
        rec_engine = RecommendationEngine(db_conn=rec_conn)
        has_recommendations = True
    except (ImportError, Exception):
        has_recommendations = False
        rec_engine = None

    # Initialize dashboard generators
    try:
        import dashboard_generator as dash
        has_dashboard = True
    except ImportError:
        has_dashboard = False

    try:
        import dashboard_server as dash_server
        has_dash_server = True
    except ImportError:
        has_dash_server = False

    print(f"\n=== MTP Healthcare Chatbot ===")
    print(f"Welcome, {session['username']}!")
    advanced_str = " | Advanced DS: ON" if has_advanced else ""
    rec_str = " | Recommendations: ON" if has_recommendations else ""
    dash_str = " | Dashboards: ON" if has_dashboard else ""
    print(f"Type 'help' for commands, 'quit' to exit.{advanced_str}{rec_str}{dash_str}")
    if has_dash_server:
        print(f"  Type 'dashboard' to open the full interactive dashboard in your browser.\n")
    else:
        print()

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() == 'quit':
                print("Goodbye!")
                break

            if user_input.lower() == 'dashboard' and has_dash_server:
                print("  Launching interactive dashboard in browser...")
                dash_server.launch_dashboard(cfg)
                print("  Dashboard is running! You can keep querying here too.")
                continue

            if user_input.lower() == 'help':
                print(
                    "Commands:\n"
                    "  tables        - List available tables\n"
                    "  relationships - Show table relationships\n"
                    "  describe <t>  - Show table schema\n"
                    "  history       - Show question history\n"
                    "  lineage       - Show data lineage info\n"
                    "  analytics     - Show analytics recommendations\n"
                    "  recommend     - Get query recommendations\n"
                    "  kpis          - Show all KPI metrics\n"
                    "  revenue       - Revenue improvement recommendations\n"
                    "  retention     - Member retention recommendations\n"
                    "  acquisition   - New customer acquisition insights\n"
                    "  operations    - Operational efficiency recommendations\n"
                    "  actions       - Top 5 prioritized actions across all areas\n"
                    "  metrics       - System performance metrics\n"
                    "  complete <p>  - SQL autocomplete for prefix\n"
                    "  dashboard     - Open full interactive dashboard in browser\n"
                    "  quit          - Exit\n"
                )
                continue

            if user_input.lower() == 'tables':
                tables = catalog.get_all_table_names()
                print(f"Available tables: {', '.join(tables)}\n")
                continue

            if user_input.lower() == 'relationships':
                rels = catalog.relationships
                if rels:
                    if isinstance(rels, list):
                        for rel in rels[:10]:
                            if isinstance(rel, dict):
                                print(f"  {rel.get('from_table', '?')} → {rel.get('to_table', '?')} (on {rel.get('join_key', rel.get('on', '?'))})")
                            else:
                                print(f"  {rel}")
                    elif isinstance(rels, dict):
                        for rel_id, rel in list(rels.items())[:10]:
                            print(f"  {rel.get('from_table', '?')} → {rel.get('to_table', '?')}")
                else:
                    print("No relationships defined.")
                print()
                continue

            if user_input.lower().startswith('describe '):
                table = user_input.replace('describe ', '').strip()
                if table in catalog.tables:
                    info = catalog.tables[table]
                    print(f"\n{table}:")
                    for col in info.get('columns', [])[:5]:
                        print(f"  {col.get('name')} ({col.get('type', 'unknown')})")
                else:
                    print(f"Table not found: {table}")
                print()
                continue

            if user_input.lower() == 'history':
                print("\nRecent questions:")
                for q in session['history'][-5:]:
                    print(f"  - {q}")
                print()
                continue

            # --- Recommendation Engine Commands ---
            if user_input.lower() in ('kpis', 'kpi') and has_recommendations:
                kpi_summary = rec_engine.kpi_summary()
                if has_dashboard:
                    path = dash.kpi_dashboard(kpi_summary)
                    print(f"\n  Dashboard opened in browser: {path}")
                else:
                    print("\n=== KPI Dashboard ===")
                    for cat, kpis in kpi_summary.get('by_category', {}).items():
                        print(f"\n  [{cat.upper()}]")
                        for k in kpis[:3]:
                            val = k.get('value', 'N/A')
                            if isinstance(val, float):
                                val = "{:.2f}{}".format(val, k.get('unit', ''))
                            alert = " *** ALERT ***" if k.get('alert') else ""
                            print(f"    {k['kpi']}: {val}{alert}")
                    alerts = kpi_summary.get('alerts', [])
                    if alerts:
                        print(f"\n  ALERTS: {len(alerts)} KPIs need attention!")
                print()
                continue

            if user_input.lower() == 'revenue' and has_recommendations:
                recs = rec_engine.revenue_recommendations()
                if has_dashboard:
                    path = dash.revenue_dashboard(recs)
                    print(f"\n  Dashboard opened in browser: {path}")
                else:
                    print("\n=== Revenue Improvement Recommendations ===")
                    for r in recs:
                        print(f"\n  [{r['priority']}] {r['title']}")
                        print(f"    Metric: {r.get('metric', 'N/A')}")
                        print(f"    Insight: {r.get('insight', '')[:100]}")
                        for a in r.get('actions', [])[:2]:
                            print(f"    -> {a}")
                        if r.get('expected_impact'):
                            print(f"    Impact: {r['expected_impact']}")
                print()
                continue

            if user_input.lower() == 'retention' and has_recommendations:
                recs = rec_engine.retention_recommendations()
                if has_dashboard:
                    path = dash.retention_dashboard(recs)
                    print(f"\n  Dashboard opened in browser: {path}")
                else:
                    print("\n=== Member Retention Recommendations ===")
                    for r in recs:
                        print(f"\n  [{r['priority']}] {r['title']}")
                        print(f"    Metric: {r.get('metric', 'N/A')}")
                        print(f"    Insight: {r.get('insight', '')[:100]}")
                        for a in r.get('actions', [])[:2]:
                            print(f"    -> {a}")
                print()
                continue

            if user_input.lower() == 'acquisition' and has_recommendations:
                recs = rec_engine.acquisition_recommendations()
                if has_dashboard:
                    path = dash.acquisition_dashboard(recs)
                    print(f"\n  Dashboard opened in browser: {path}")
                else:
                    print("\n=== Customer Acquisition Recommendations ===")
                    for r in recs:
                        print(f"\n  [{r['priority']}] {r['title']}")
                        print(f"    Metric: {r.get('metric', 'N/A')}")
                        for a in r.get('actions', [])[:2]:
                            print(f"    -> {a}")
                print()
                continue

            if user_input.lower() == 'operations' and has_recommendations:
                recs = rec_engine.operational_recommendations()
                if has_dashboard:
                    path = dash.operations_dashboard(recs)
                    print(f"\n  Dashboard opened in browser: {path}")
                else:
                    print("\n=== Operational Efficiency Recommendations ===")
                    for r in recs:
                        print(f"\n  [{r['priority']}] {r['title']}")
                        print(f"    Metric: {r.get('metric', 'N/A')}")
                        for a in r.get('actions', [])[:2]:
                            print(f"    -> {a}")
                print()
                continue

            if user_input.lower() == 'actions' and has_recommendations:
                actions = rec_engine.prioritized_actions()
                if has_dashboard:
                    path = dash.actions_dashboard(actions)
                    print(f"\n  Dashboard opened in browser: {path}")
                else:
                    print("\n=== TOP 5 PRIORITIZED ACTIONS ===")
                    for i, a in enumerate(actions, 1):
                        print(f"  {i}. [{a['priority']}] {a['action']}")
                        print(f"     First Step: {a['first_step']}")
                print()
                continue

            if user_input.lower() == 'metrics' and has_advanced:
                cache_stats = query_cache.stats()
                perf_stats = metrics.get_dashboard()
                if has_dashboard:
                    path = dash.metrics_dashboard(cache_stats, perf_stats)
                    print(f"\n  Dashboard opened in browser: {path}")
                else:
                    print("\n=== System Performance Metrics ===")
                    print(f"  Query Cache: {cache_stats}")
                    print(f"  Performance: {perf_stats}")
                print()
                continue

            if user_input.lower().startswith('complete ') and has_advanced:
                prefix = user_input[9:].strip()
                completions = sql_trie.starts_with(prefix, limit=10)
                print(f"\n  Completions for '{prefix}':")
                for word, kind in completions:
                    print(f"    {word} ({kind})")
                print()
                continue

            # --- Cache check (before expensive processing) ---
            t_start = time.time()
            if has_advanced:
                cached = query_cache.get(user_input.lower())
                if cached is not None:
                    elapsed_ms = (time.time() - t_start) * 1000
                    metrics.record_query(elapsed_ms, True)
                    events.emit('QUERY_CACHED', {'question': user_input, 'latency_ms': elapsed_ms})
                    print(f"\n[CACHED] (hit rate: {query_cache.hit_rate*100:.0f}%)")
                    result = cached
                    session['history'].append(user_input)
                    print(f"Bot (Intent: {result['intent']}, Mode: {result['engine_mode']})")
                    if isinstance(result['answer'], str):
                        print(result['answer'])
                    if result.get('sql'):
                        print(f"\nSQL: {result['sql'][:100]}...")
                    print()
                    continue

            # Process as question
            result = process_question(user_input, cfg, catalog, session)

            # Record metrics and cache result
            elapsed_ms = (time.time() - t_start) * 1000
            if has_advanced:
                metrics.record_query(elapsed_ms, result.get('success', True))
                query_cache.put(user_input.lower(), result)
                events.emit('QUERY_EXECUTED', {
                    'question': user_input,
                    'intent': result.get('intent', ''),
                    'latency_ms': elapsed_ms,
                    'success': result.get('success', True),
                })

            session['history'].append(user_input)

            print(f"\nBot (Intent: {result['intent']}, Mode: {result['engine_mode']})")

            # Open dashboard for query results
            if has_dashboard and (isinstance(result['answer'], list) or result.get('sql')):
                path = dash.query_results_dashboard(
                    user_input,
                    result.get('sql', ''),
                    result.get('answer', []),
                    result.get('intent', ''),
                    result.get('engine_mode', ''))
                print(f"  Dashboard opened in browser: {path}")

            if isinstance(result['answer'], list) and result['answer']:
                rows = result['answer']
                print(f"Results ({len(rows)} rows):\n")
                # Pretty-print as aligned table
                if isinstance(rows[0], dict):
                    cols = list(rows[0].keys())
                    # Compute column widths (capped at 20 chars)
                    widths = {}
                    for c in cols:
                        vals = [str(r.get(c, ''))[:20] for r in rows[:10]]
                        widths[c] = min(max(len(c), max((len(v) for v in vals), default=4)), 20)
                    # Header
                    header = '  '.join(c.ljust(widths[c]) for c in cols)
                    print(f"  {header}")
                    print(f"  {'  '.join('-' * widths[c] for c in cols)}")
                    # Rows (show up to 15)
                    for row in rows[:15]:
                        line = '  '.join(str(row.get(c, ''))[:20].ljust(widths[c]) for c in cols)
                        print(f"  {line}")
                    if len(rows) > 15:
                        print(f"\n  ... and {len(rows) - 15} more rows (see dashboard for full results)")
                else:
                    for row in rows[:10]:
                        print(f"  {row}")
            elif isinstance(result['answer'], str):
                print(result['answer'])

            if result['sql']:
                print(f"\nSQL: {result['sql']}")
            print()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            traceback.print_exc()


# ==================== MAIN ====================

if __name__ == '__main__':
    # Auto-discover config
    config_path = auto_discover_config()
    if config_path:
        cfg = read_config(config_path)
        print(f"[INFO] Loaded config from {config_path}")
    else:
        cfg = {'data_dir': '../data'}
        print("[WARNING] Config not found, using defaults")

    # Run interactive mode
    interactive_mode(cfg)
