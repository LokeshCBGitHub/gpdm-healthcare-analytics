"""
Semantic SQL Engine: Data-driven NL→SQL generation using schema learning,
intent classification, and knowledge graphs. Adapts intelligently to any database.
"""

import re
import os
import math
import sqlite3
import logging
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional, Set

from semantic_layer import (
    SemanticLayer, SchemaLearner, SemanticSchemaIndex,
    IntentClassifier, ComputedColumnInferrer, DataValidator,
    AdaptiveQueryBuilder, TFIDFIndex, SparseVector,
)

try:
    from graph_vector_engine import KnowledgeGraph
except ImportError:
    KnowledgeGraph = None

logger = logging.getLogger('gpdm.semantic_sql')


# PART 1: SCHEMA KNOWLEDGE GRAPH

class SchemaKnowledgeGraph:
    """
    Builds a KnowledgeGraph from the SchemaLearner's auto-discovered schema.
    Uses PageRank to score column/table importance for SELECT prioritization
    and Dijkstra for optimal join path finding.

    This replaces the hardcoded BFS join logic in SchemaRegistry.
    """

    def __init__(self, learner: SchemaLearner):
        self.learner = learner
        self.graph = KnowledgeGraph() if KnowledgeGraph else None
        if self.graph:
            self._build()

    def _build(self):
        """Build knowledge graph from learned schema."""
        g = self.graph

        # Add table nodes
        for table, profiles in self.learner.tables.items():
            row_count = self.learner.table_row_counts.get(table, 0)
            g.add_node(f"table:{table}", 'table',
                       row_count=row_count, col_count=len(profiles))

            # Add column nodes and edges
            for p in profiles:
                col_id = f"col:{table}.{p.name}"
                g.add_node(col_id, 'column',
                           table=table, name=p.name,
                           is_numeric=p.is_numeric, is_date=p.is_date,
                           is_categorical=p.is_categorical, is_id=p.is_id,
                           semantic_tags=p.semantic_tags,
                           distinct_count=p.distinct_count)

                # table → column (higher weight for more important columns)
                weight = 1.0
                if p.is_id:
                    weight = 2.0  # IDs are join keys — important
                elif p.is_numeric and 'currency' in p.semantic_tags:
                    weight = 1.8  # Amount columns often queried
                elif p.is_categorical:
                    weight = 1.5  # GROUP BY candidates
                elif p.is_date:
                    weight = 1.3  # Filter/trend candidates
                g.add_edge(f"table:{table}", col_id, 'has_column', weight=weight)

                # Add semantic type nodes and edges
                for tag in p.semantic_tags:
                    type_id = f"type:{tag}"
                    if type_id not in g.nodes:
                        g.add_node(type_id, 'concept')
                    g.add_edge(col_id, type_id, 'is_type', weight=0.8)

                # Add value nodes for categorical columns
                if p.is_categorical:
                    for v in p.sample_values[:10]:
                        val_id = f"val:{table}.{p.name}={v}"
                        g.add_node(val_id, 'value',
                                   table=table, column=p.name, value=v)
                        g.add_edge(col_id, val_id, 'has_value', weight=0.5)

        # Add join edges from learned schema
        for t1, neighbors in self.learner.join_graph.items():
            for t2, join_col in neighbors.items():
                if '=' in join_col:
                    left, right = join_col.split('=', 1)
                    g.add_edge(f"table:{t1}", f"table:{t2}", 'joins_to',
                               weight=2.0, left_col=left, right_col=right)
                else:
                    g.add_edge(f"table:{t1}", f"table:{t2}", 'joins_to',
                               weight=2.0, join_column=join_col)

        # Compute PageRank for importance scoring
        g.compute_pagerank()

    def find_join_path(self, tables: List[str]) -> List[Tuple[str, str, str]]:
        """
        Find optimal join path using Dijkstra's algorithm on the knowledge graph.
        Returns: [(table1, table2, join_condition), ...]
        """
        if not self.graph or len(tables) <= 1:
            return []

        joins = []
        connected = {tables[0]}

        for target in tables[1:]:
            if target in connected:
                continue

            # Try Dijkstra through the graph
            best_source = None
            best_cost = float('inf')
            best_path = []

            for source in connected:
                cost, path = self.graph.dijkstra(
                    f"table:{source}", f"table:{target}",
                    edge_filter=lambda e: e.relation in ('joins_to', 'has_column')
                )
                if cost < best_cost:
                    best_cost = cost
                    best_path = path
                    best_source = source

            if best_source and best_path:
                # Extract join condition from the path
                join_col = self._get_join_condition(best_source, target)
                joins.append((best_source, target, join_col))
                connected.add(target)
            else:
                # Fallback: try learned join graph directly
                join_col = self._get_join_condition(tables[0], target)
                joins.append((tables[0], target, join_col))
                connected.add(target)

        return joins

    def _get_join_condition(self, t1: str, t2: str) -> str:
        """Get join condition between two tables from learned schema.
        Fully data-driven — no hardcoded column names.
        Prefers shared _ID columns (strong foreign keys) over learned joins
        when both exist, because the learned join graph may pick descriptive
        columns (KP_REGION, RENDERING_NPI) that produce cartesian-like joins."""

        # First: check for shared _ID columns — these are the strongest join keys
        cols1 = {p.name.upper() for p in self.learner.tables.get(t1, [])}
        cols2 = {p.name.upper() for p in self.learner.tables.get(t2, [])}
        shared_ids = sorted(c for c in (cols1 & cols2) if c.endswith('_ID'))

        # Prefer entity-specific IDs (ENCOUNTER_ID, MEMBER_ID, CLAIM_ID) over generic _ID
        if shared_ids:
            # Prefer IDs that match either table's stem (e.g. ENCOUNTER_ID for encounters)
            t1_stem = t1.rstrip('s').upper()
            t2_stem = t2.rstrip('s').upper()
            for sid in shared_ids:
                if sid.startswith(t1_stem) or sid.startswith(t2_stem):
                    return sid
            # Otherwise return the first shared _ID
            return shared_ids[0]

        # Fall back to learned join graph
        join_col = self.learner.join_graph.get(t1, {}).get(t2, '')
        if join_col:
            return join_col
        # Reverse lookup
        join_col = self.learner.join_graph.get(t2, {}).get(t1, '')
        if join_col:
            if '=' in join_col:
                left, right = join_col.split('=', 1)
                return f"{right}={left}"
            return join_col

        # Fallback: look for shared column names, prefer strong join keys (_ID columns)
        cols1 = {p.name.upper() for p in self.learner.tables.get(t1, [])}
        cols2 = {p.name.upper() for p in self.learner.tables.get(t2, [])}
        shared = cols1 & cols2

        # Exclude non-join columns: categorical/descriptive columns that happen to share names
        # Detect these by checking if the column is NOT an ID/FK in either table
        weak = set()
        for col in shared:
            is_join_key = col.endswith('_ID') or col.endswith('_KEY') or col == 'ID'
            if not is_join_key:
                # Check if it's categorical in both tables
                for p in self.learner.tables.get(t1, []):
                    if p.name.upper() == col and (p.is_categorical or p.is_date):
                        weak.add(col)
                        break
        shared -= weak

        # Priority: *_ID columns first
        for c in sorted(shared):
            if c.endswith('_ID'):
                return c
        # Then any remaining shared column
        if shared:
            return sorted(shared)[0]

        # Last resort: look for FK pattern — t2's primary key referenced from t1
        # e.g. orders has customer_id → customers.customer_id
        t2_stem = t2.rstrip('s').upper()
        for p in self.learner.tables.get(t1, []):
            if p.name.upper() == f"{t2_stem}_ID":
                if f"{t2_stem}_ID" in cols2:
                    return f"{t2_stem}_ID"
        t1_stem = t1.rstrip('s').upper()
        for p in self.learner.tables.get(t2, []):
            if p.name.upper() == f"{t1_stem}_ID":
                if f"{t1_stem}_ID" in cols1:
                    return f"{t1_stem}_ID"

        # Absolute last resort: first _ID column shared
        for c in sorted(cols1 & cols2):
            return c
        return ''

    def get_important_columns(self, table: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Get the most important columns for a table, ranked by PageRank."""
        if not self.graph:
            return [(p.name, 1.0) for p in self.learner.tables.get(table, [])[:top_k]]

        results = []
        for p in self.learner.tables.get(table, []):
            node_id = f"col:{table}.{p.name}"
            node = self.graph.nodes.get(node_id)
            if node:
                results.append((p.name, node.importance))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


# PART 2: SEMANTIC SQL COMPOSER

class SemanticSQLComposer:
    """
    Composes SQL queries from semantic understanding rather than regex patterns.

    Input: intent, matched columns, matched tables, matched values, computed columns
    Output: valid SQL query

    This replaces the hardcoded SQLComposer + QueryParser pipeline.
    """

    # ═══════════════════════════════════════════════════════════════════════
    # PRODUCTION CONSTANTS — tunable thresholds, column registries, scoring
    # weights. Every magic number in the engine is defined here exactly once.
    # ═══════════════════════════════════════════════════════════════════════

    # ── Clinical / domain constants ──────────────────────────────────────
    READMISSION_WINDOW_DAYS = 30          # 30-day readmission standard (CMS)
    AGE_BRACKETS = (18, 35, 50, 65)       # Age group boundaries for demographics

    # ── Member identity columns (searched in priority order) ─────────────
    MEMBER_ID_CANDIDATES = ('MEMBER_ID', 'PATIENT_ID', 'MRN', 'PERSON_ID',
                            'SUBSCRIBER_ID', 'BENEFICIARY_ID')

    # ── NPI / provider-join patterns ─────────────────────────────────────
    NPI_COLUMN_SUFFIXES = ('_NPI',)       # Columns that reference providers
    NPI_COLUMN_NAMES = ('NPI', 'RENDERING_NPI', 'BILLING_NPI',
                        'ATTENDING_NPI', 'PROVIDER_NPI', 'ORDERING_NPI')

    # ── Financial columns (used for concept mismatch detection) ──────────
    FINANCIAL_COLUMNS = ('BILLED_AMOUNT', 'PAID_AMOUNT', 'ALLOWED_AMOUNT',
                         'MEMBER_RESPONSIBILITY', 'COPAY', 'COINSURANCE',
                         'DEDUCTIBLE', 'TOTAL_CHARGE')

    # ── Entity → column resolution map (for ranking queries) ─────────────
    ENTITY_ID_HINTS = {
        'provider': ('RENDERING_NPI', 'BILLING_NPI', 'NPI', 'PROVIDER_ID'),
        'member':   ('MEMBER_ID', 'PATIENT_ID', 'MRN'),
        'patient':  ('MEMBER_ID', 'PATIENT_ID', 'MRN'),
        'diagnosis': ('DIAGNOSIS_CODE', 'ICD_CODE', 'DX_CODE'),
        'procedure': ('PROCEDURE_CODE', 'CPT_CODE', 'HCPCS_CODE'),
    }

    # ── Insurance / plan type disambiguation triggers ────────────────────
    INSURANCE_KEYWORDS = frozenset({'plan', 'payer', 'insurer', 'coverage'})
    INSURANCE_VALUES = frozenset({'medicaid', 'medicare', 'hmo', 'ppo', 'epo',
                                  'hdhp', 'tricare', 'champva', 'medi-cal',
                                  'commercial', 'self-pay', 'advantage'})
    DISAMBIGUATION_TRIGGERS = ('provider', 'payer', 'insurer', 'plan',
                               'insurance', 'carrier', 'coverage')
    PRACTITIONER_TERMS = frozenset({'doctor', 'physician', 'specialist', 'npi',
                                    'practitioner', 'clinician', 'panel',
                                    'workforce', 'staff', 'specialty'})

    # ── Scoring weights — tuned via gpdm_config.py for balance ────────────
    # Import from central config; fallback to calibrated defaults
    try:
        from gpdm_config import (
            WORD_OVERLAP_WEIGHT as _WOW, CURRENCY_TAG_BOOST as _CTB,
            DISAMBIGUATION_FACT_BOOST as _DFB, DISAMBIGUATION_INDEX_BOOST as _DIB,
            INSURANCE_VALUE_INDEX_BOOST as _IVB, RATE_TABLE_BOOST as _RTB,
            ROW_COUNT_DIVISOR as _RCD, ROW_BOOST_CAP as _RBC,
            MIN_CONCEPT_HITS as _MCH, MIN_CONCEPT_MATCH_SCORE as _MCM,
            MIN_COLUMN_SCORE as _MCS, MIN_VALUE_SCORE as _MVS,
            VALUE_FILTER_SCORE as _VFS, TABLE_SCORE_RATIO as _TSR,
            MIN_COL_PER_TABLE as _MCPT, MIN_TABLE_SIZE_FOR_METRIC as _MTSM,
            HAVING_MIN_USEFUL_DEFAULT as _HMD, HAVING_MIN_USEFUL_CROSS as _HMC,
            HAVING_MIN_RATE_THRESHOLD as _HMR,
            DEFAULT_BREAKDOWN_LIMIT as _DBL, DEFAULT_TREND_LIMIT as _DTL,
            DEFAULT_RANKING_LIMIT as _DRL, DEFAULT_LOOKUP_LIMIT as _DLL,
            MIN_SEMANTIC_CONFIDENCE as _MSC,
        )
    except ImportError:
        _WOW, _CTB, _DFB, _DIB = 7, 5, 10, 7
        _IVB, _RTB, _RCD, _RBC = 5, 7, 10000, 1.8
        _MCH, _MCM, _MCS, _MVS = 3, 4, 0.15, 0.20
        _VFS, _TSR, _MCPT, _MTSM = 0.25, 0.55, 2, 500
        _HMD, _HMC, _HMR = 5, 200, 15
        _DBL, _DTL, _DRL, _DLL = 25, 48, 10, 100
        _MSC = 0.40

    WORD_OVERLAP_WEIGHT = _WOW
    ROW_COUNT_DIVISOR = _RCD
    ROW_BOOST_CAP = _RBC
    CURRENCY_TAG_BOOST = _CTB
    DISAMBIGUATION_FACT_BOOST = _DFB
    DISAMBIGUATION_INDEX_BOOST = _DIB
    INSURANCE_VALUE_INDEX_BOOST = _IVB
    RATE_TABLE_BOOST = _RTB

    # ── Score thresholds ─────────────────────────────────────────────────
    MIN_CONCEPT_HITS = _MCH
    MIN_CONCEPT_MATCH_SCORE = _MCM
    MIN_COLUMN_SCORE = _MCS
    MIN_VALUE_SCORE = _MVS
    VALUE_FILTER_SCORE = _VFS
    TABLE_SCORE_RATIO = _TSR
    MIN_COL_PER_TABLE = _MCPT
    MIN_TABLE_SIZE_FOR_METRIC = _MTSM

    # ── Adaptive HAVING defaults ─────────────────────────────────────────
    HAVING_MIN_USEFUL_DEFAULT = _HMD
    HAVING_MIN_USEFUL_CROSS_METRIC = _HMC
    HAVING_MIN_RATE_THRESHOLD = _HMR

    # ── Result limits ────────────────────────────────────────────────────
    DEFAULT_BREAKDOWN_LIMIT = _DBL
    DEFAULT_TREND_LIMIT = _DTL
    DEFAULT_RANKING_LIMIT = _DRL
    DEFAULT_LOOKUP_LIMIT = _DLL

    # ── Minimum confidence ───────────────────────────────────────────────
    MIN_SEMANTIC_CONFIDENCE = _MSC

    def __init__(self, learner: SchemaLearner, kg: SchemaKnowledgeGraph,
                 inferrer: ComputedColumnInferrer, domain_config=None):
        self.learner = learner
        self.kg = kg
        self.inferrer = inferrer
        self.domain_config = domain_config

        # ── Schema Intelligence: auto-discovers domain knowledge from data ──
        self._schema_intel = None
        try:
            from schema_intelligence import SchemaIntelligence
            db_path = getattr(learner, 'db_path', None)
            if db_path:
                self._schema_intel = SchemaIntelligence(learner, db_path)
                self._schema_intel.learn()
                # Merge auto-discovered knowledge with manual (manual wins conflicts)
                from dynamic_sql_engine import DOMAIN_CONCEPTS as _manual_dc, SYNONYMS as _manual_syn
                self._schema_intel.merge_with_manual(_manual_dc, _manual_syn)
                logger.info("SchemaIntelligence: %d concepts (%d auto-discovered), %d synonyms",
                            len(self._schema_intel.domain_concepts),
                            sum(1 for v in self._schema_intel.domain_concepts.values()
                                if v.get('_discovered')),
                            len(self._schema_intel.synonyms))
        except Exception as e:
            logger.warning("SchemaIntelligence unavailable (%s), using manual config only", e)

        # Domain concept disambiguator: maps ambiguous terms to schema concepts
        # (must be built before concept table index so it can influence scoring)
        self._domain_disambiguations = self._build_domain_disambiguations()
        # Pre-build concept→table index for robust fact table resolution
        self._concept_table_index = self._build_concept_table_index()
        # Population filters: general mechanism for "metric FOR population" patterns
        self._population_filters = self._build_population_filters()
        # Derived concepts: computed metrics not present as columns
        self._derived_concepts = self._build_derived_concepts()

    # ═══════════════════════════════════════════════════════════════════════
    # ARCHITECTURAL LAYER 4: DOMAIN CONCEPT DISAMBIGUATOR
    # ═══════════════════════════════════════════════════════════════════════
    # In healthcare analytics, certain terms are ambiguous:
    #   "provider" → insurance plan (Medicaid/Medicare/PPO) NOT practitioners
    #   "plan" → insurance plan type
    #   "payer" → insurance plan type
    # This layer discovers these mappings from actual schema data at startup
    # and rewrites dimension terms before column scoring.
    # ═══════════════════════════════════════════════════════════════════════

    def _build_domain_disambiguations(self) -> Dict[str, Dict]:
        """Discover domain-specific term→column mappings from schema.

        Scans for insurance/plan type columns and maps terms like 'provider'
        to them. This is schema-driven, not hardcoded — if the database has
        a PLAN_TYPE column with values like 'Medicaid', 'Medicare', etc.,
        those are recognized as insurance providers.
        """
        disambiguations = {}

        # Discover insurance/plan type columns
        insurance_keywords = self.INSURANCE_KEYWORDS
        insurance_values = self.INSURANCE_VALUES

        for tbl_name, profiles in self.learner.tables.items():
            for p in profiles:
                if not p.is_categorical:
                    continue
                cn_lower = p.name.lower()

                # Check if this column looks like a plan/insurance type
                is_plan_col = any(kw in cn_lower for kw in insurance_keywords)

                # Also check sample values for insurance plan names
                has_insurance_vals = False
                if p.sample_values:
                    for val in p.sample_values:
                        val_lower = str(val).lower()
                        if any(iv in val_lower for iv in insurance_values):
                            has_insurance_vals = True
                            break

                if is_plan_col or has_insurance_vals:
                    # This column represents insurance providers/plans
                    row_count = self.learner.table_row_counts.get(tbl_name, 0)
                    key = f'insurance_plan_{tbl_name}_{p.name}'
                    disambiguations[key] = {
                        'table': tbl_name,
                        'column': p.name,
                        'triggers': list(self.DISAMBIGUATION_TRIGGERS),
                        'row_count': row_count,
                        'sample_values': list(p.sample_values or [])[:6],
                        'description': f'Insurance plan type ({", ".join(str(v) for v in (p.sample_values or [])[:4])})',
                    }

        return disambiguations

    def _disambiguate_dimension_term(self, term: str, fact: str) -> Optional[Dict]:
        """Check if a dimension term should be disambiguated.

        E.g., "provider" in "denial rate by provider" → PLAN_TYPE in claims
        Returns {table, column, description} or None.
        """
        term_lower = term.lower().strip()
        # Stem for plural tolerance: "providers" → "provider"
        term_stem = term_lower.rstrip('s') if term_lower.endswith('s') and len(term_lower) > 3 else term_lower

        # Find matching disambiguations
        matches = []
        for key, dis in self._domain_disambiguations.items():
            if term_lower in dis['triggers'] or term_stem in dis['triggers']:
                # Prefer columns in the fact table itself
                in_fact = (dis['table'] == fact)
                matches.append((dis, in_fact))

        if not matches:
            return None

        # Sort: prefer in-fact, then by row count
        matches.sort(key=lambda x: (x[1], x[0]['row_count']), reverse=True)
        best = matches[0][0]

        return {
            'table': best['table'],
            'column': best['column'],
            'description': best['description'],
        }

    # ═══════════════════════════════════════════════════════════════════════
    # ARCHITECTURAL LAYER 1: POPULATION FILTERS
    # ═══════════════════════════════════════════════════════════════════════
    # Detects "metric X FOR population Y" where Y is defined by any concept:
    #   "average cost for readmitted patients"
    #   "total claims for denied members"
    #   "average length of stay for emergency encounters"
    #   "claim count for members over 65"
    # The population filter is a CTE or WHERE clause that restricts the base
    # population, then the metric is computed on that restricted set.
    # ═══════════════════════════════════════════════════════════════════════

    def _build_population_filters(self) -> Dict[str, Dict]:
        """Build a registry of population filters from schema metadata.
        Each filter defines HOW to restrict a population, not what to do with it.
        The metric computation is handled separately.

        Returns: {keyword: {type, build_fn_name, description, tables}}
        """
        filters = {}

        # Readmission filter: members readmitted within 30 days
        # Discover from encounters table: needs VISIT_TYPE, ADMIT_DATE, DISCHARGE_DATE
        enc_cols = {p.name for p in self.learner.tables.get('encounters', [])}
        if 'ADMIT_DATE' in enc_cols and 'DISCHARGE_DATE' in enc_cols:
            filters['readmitted'] = {
                'type': 'cte',
                'keywords': ['readmit', 'readmission', 'readmitted', 're-admission', 're-admitted'],
                'description': 'Members readmitted within 30 days of discharge',
                'member_col': 'MEMBER_ID',
                'source_table': 'encounters',
            }

        # Status-based filters: discover from any table with a status column
        for tbl_name, profiles in self.learner.tables.items():
            for p in profiles:
                if not p.is_categorical or 'status' not in p.name.lower():
                    continue
                # Get distinct values from this status column
                vals = set(v.upper() for v in (p.sample_values or []) if isinstance(v, str))
                # Also try a DB scan for completeness
                try:
                    import sqlite3
                    db_path = getattr(self.learner, 'db_path', None)
                    if db_path:
                        conn = sqlite3.connect(db_path)
                        cursor = conn.execute(
                            f"SELECT DISTINCT {p.name} FROM {tbl_name} "
                            f"WHERE {p.name} IS NOT NULL LIMIT 20")
                        for row in cursor.fetchall():
                            if row[0] and isinstance(row[0], str):
                                vals.add(row[0].upper())
                        conn.close()
                except Exception:
                    pass

                for val in vals:
                    vl = val.lower()
                    # Register the value and its derivations as population filter keywords
                    keywords = [vl]
                    if vl.endswith('ed') and len(vl) > 4:
                        keywords.append(vl[:-2] + 'al')  # denied → denial
                        keywords.append(vl[:-2] + 'y')   # denied → deny
                        keywords.append(vl[:-1])          # approved → approve
                    filters[f'{tbl_name}.{p.name}={val}'] = {
                        'type': 'where',
                        'keywords': keywords,
                        'table': tbl_name,
                        'column': p.name,
                        'value': val,
                        'description': f'{tbl_name} where {p.name} = {val}',
                        'member_col': self._resolve_member_column(tbl_name),
                    }

        return filters

    def _find_member_column(self, table: str) -> str:
        """DEPRECATED: Use _resolve_member_column() instead. Kept for compatibility."""
        return self._resolve_member_column(table)

    def _detect_population_filter(self, q: str) -> Optional[Dict]:
        """Detect if the question defines a population to filter on.
        Returns {filter_key, agg, metric_hint, dimension} or None.

        Catches patterns like:
          "average cost for readmitted patients"
          "total claims for denied encounters"
          "claim count for members over 65"
          "average length of stay for emergency visits"
        """
        q_lower = q.lower()

        # Pattern: "METRIC for/of/among POPULATION"
        # Also: "POPULATION METRIC" (e.g., "readmitted patient costs")
        agg_words = {
            'average': 'AVG', 'avg': 'AVG', 'mean': 'AVG',
            'total': 'SUM', 'sum': 'SUM', 'count': 'COUNT',
            'median': 'AVG', 'highest': 'MAX', 'max': 'MAX',
            'lowest': 'MIN', 'min': 'MIN', 'number': 'COUNT',
        }

        # Try structured patterns first
        patterns = [
            # "average cost for readmitted patients"
            r'(average|avg|mean|total|sum|count|median|highest|lowest|max|min|number\s+of)'
            r'\s+(\w+(?:\s+\w+)?)\s+'
            r'(?:for|of|among|per|in)\s+'
            r'(?:\w+\s+)?(\w+)',
            # "how many claims for denied members"
            r'how\s+many\s+(\w+)\s+(?:for|of|among|in|with)\s+(?:\w+\s+)?(\w+)',
        ]

        for pat in patterns:
            m = re.search(pat, q_lower)
            if m:
                groups = m.groups()
                if len(groups) == 3:
                    agg_word, metric_hint, pop_word = groups
                    agg_word = agg_word.strip()
                elif len(groups) == 2:
                    metric_hint, pop_word = groups
                    agg_word = 'count'
                else:
                    continue

                agg_fn = agg_words.get(agg_word.split()[0], 'AVG')

                # Match pop_word against registered population filters
                matched_filter = self._match_population_filter(pop_word)
                if matched_filter:
                    # Extract dimension if present
                    dim_term = None
                    dim_m = re.search(r'by\s+(\w+(?:\s+\w+)?)', q_lower)
                    if dim_m:
                        dim_term = dim_m.group(1).lower().strip()

                    return {
                        'filter_key': matched_filter,
                        'agg': agg_fn,
                        'metric_hint': metric_hint.strip(),
                        'dimension': dim_term,
                    }

        return None

    def _match_population_filter(self, word: str) -> Optional[str]:
        """Match a word against all registered population filters.
        Returns the filter key or None."""
        word_lower = word.lower().rstrip('s')  # normalize plural
        for key, filt in self._population_filters.items():
            for kw in filt['keywords']:
                if word_lower.startswith(kw) or kw.startswith(word_lower):
                    return key
        return None

    def _sql_population_filtered_metric(self, q: str, fact: str,
                                         pop_info: Dict) -> Dict:
        """General-purpose: compute any metric for any filtered population.
        This replaces individual handlers like _sql_readmission_filtered_metric.

        Builds: [population CTE/WHERE] → [metric computation] → [optional GROUP BY]
        """
        filter_key = pop_info['filter_key']
        agg_fn = pop_info['agg']
        metric_hint = pop_info['metric_hint']
        dim_term = pop_info.get('dimension')
        filt = self._population_filters[filter_key]

        # --- Step 1: Build the population restriction ---
        if filt['type'] == 'cte' and 'readmit' in filter_key:
            # Readmission CTE (general pattern for event-based populations)
            inpatient_val = 'INPATIENT'
            res = self._resolve_category_value('inpatient', ['encounters'])
            if res:
                _, _, inpatient_val = res
            pop_cte = (
                f"WITH filtered_pop AS ("
                f"SELECT DISTINCT ip.MEMBER_ID FROM ("
                f"SELECT MEMBER_ID, ADMIT_DATE, "
                f"LAG(DISCHARGE_DATE) OVER (PARTITION BY MEMBER_ID ORDER BY ADMIT_DATE) as prev_discharge "
                f"FROM encounters "
                f"WHERE VISIT_TYPE = '{inpatient_val}' AND DISCHARGE_DATE != '' AND ADMIT_DATE != ''"
                f") ip "
                f"WHERE julianday(ip.ADMIT_DATE) - julianday(ip.prev_discharge) BETWEEN 0 AND {self.READMISSION_WINDOW_DAYS}"
                f")"
            )
            join_clause = f"JOIN filtered_pop fp ON t.{filt['member_col']} = fp.MEMBER_ID"
        elif filt['type'] == 'where':
            # Status-based WHERE filter
            pop_cte = (
                f"WITH filtered_pop AS ("
                f"SELECT DISTINCT {filt['member_col']} AS MEMBER_ID "
                f"FROM {filt['table']} "
                f"WHERE {filt['column']} = '{filt['value']}'"
                f")"
            )
            join_clause = f"JOIN filtered_pop fp ON t.{self._resolve_member_column(fact)} = fp.MEMBER_ID"
        else:
            return None

        # --- Step 2: Resolve the metric column ---
        metric_col, metric_table = self._resolve_metric_column(q, metric_hint, fact)

        # --- Step 3: Resolve the dimension (optional) ---
        dim_col = None
        dim_join = ""
        tables_used = [metric_table, filt.get('source_table', filt.get('table', 'encounters'))]

        if dim_term:
            dim = self._discover_dimension(q, metric_table)
            if dim:
                dim_col = dim['column']
                if dim.get('needs_join'):
                    dim_tbl = dim.get('table', metric_table)
                    on_clause = self._dim_on_clause(dim, 't', 'dim')
                    dim_join = f" JOIN {dim_tbl} dim ON {on_clause}"
                    dim_prefix = f"dim."
                    tables_used.append(dim_tbl)
                else:
                    dim_prefix = "t."

        # --- Step 4: Assemble ---
        if dim_col:
            sql = (
                f"{pop_cte} "
                f"SELECT {dim_prefix}{dim_col}, "
                f"COUNT(*) as record_count, "
                f"ROUND({agg_fn}(t.{metric_col}), 2) as {agg_fn.lower()}_{metric_col.lower()} "
                f"FROM {metric_table} t {join_clause}{dim_join} "
                f"GROUP BY {dim_prefix}{dim_col} "
                f"ORDER BY {agg_fn.lower()}_{metric_col.lower()} DESC LIMIT {self.DEFAULT_BREAKDOWN_LIMIT};"
            )
        else:
            sql = (
                f"{pop_cte} "
                f"SELECT COUNT(*) as filtered_records, "
                f"COUNT(DISTINCT fp.MEMBER_ID) as filtered_members, "
                f"ROUND({agg_fn}(t.{metric_col}), 2) as {agg_fn.lower()}_{metric_col.lower()}, "
                f"ROUND(MIN(t.{metric_col}), 2) as min_{metric_col.lower()}, "
                f"ROUND(MAX(t.{metric_col}), 2) as max_{metric_col.lower()} "
                f"FROM {metric_table} t {join_clause};"
            )

        return {
            'sql': sql, 'tables_used': list(set(tables_used)),
            'confidence': 0.85, 'intent': 'aggregation',
            'explanation': f'{agg_fn} of {metric_col} for {filt["description"]}',
        }

    def _resolve_metric_column(self, q: str, metric_hint: str,
                                default_table: str) -> Tuple[str, str]:
        """General-purpose metric column resolver.
        Uses synonym expansion + concept index + row-count scoring.
        Returns (column_name, table_name)."""
        # Expand hint words with synonyms for financial terms
        hint_words = set(metric_hint.replace('_', ' ').split())
        synonyms = {
            'cost': {'amount', 'billed', 'paid', 'charge', 'cost'},
            'charge': {'amount', 'billed', 'cost', 'charge'},
            'spend': {'amount', 'cost', 'paid', 'billed', 'spend'},
            'payment': {'paid', 'amount', 'payment'},
            'revenue': {'amount', 'billed', 'revenue', 'paid'},
            'stay': {'length', 'duration', 'los', 'stay'},
            'los': {'length', 'duration', 'stay', 'los'},
        }
        expanded = set(hint_words)
        for hw in hint_words:
            if hw in synonyms:
                expanded.update(synonyms[hw])

        candidates = []
        for tbl_name, profiles in self.learner.tables.items():
            row_count = self.learner.table_row_counts.get(tbl_name, 0)
            for p in profiles:
                if not p.is_numeric or p.is_id:
                    continue
                cn_words = set(p.name.lower().replace('_', ' ').split())
                overlap = expanded & cn_words
                if overlap:
                    score = len(overlap) * self.WORD_OVERLAP_WEIGHT + min(row_count / (self.ROW_COUNT_DIVISOR / 2), 5)
                    if 'currency' in (p.semantic_tags or []):
                        score += self.CURRENCY_TAG_BOOST
                    candidates.append((tbl_name, p.name, score))

        if candidates:
            candidates.sort(key=lambda c: -c[2])
            return candidates[0][1], candidates[0][0]

        # Fallback: concept index for general concept matching
        q_words = set(re.findall(r'[a-z]+', q.lower())) - {'the', 'a', 'for', 'of', 'is', 'what'}
        table_scores = defaultdict(float)
        for word in q_words:
            if word in self._concept_table_index:
                for tbl, score in self._concept_table_index[word].items():
                    table_scores[tbl] += score
        if table_scores:
            best_table = max(table_scores, key=table_scores.get)
            # Find first numeric non-ID column with currency tag
            for p in self.learner.tables.get(best_table, []):
                if p.is_numeric and not p.is_id and 'currency' in (p.semantic_tags or []):
                    return p.name, best_table
            # Fallback to first numeric non-ID
            for p in self.learner.tables.get(best_table, []):
                if p.is_numeric and not p.is_id:
                    return p.name, best_table

        return self.FINANCIAL_COLUMNS[0], default_table  # Best-guess fallback

    # ═══════════════════════════════════════════════════════════════════════
    # ARCHITECTURAL LAYER 2: DERIVED CONCEPTS
    # ═══════════════════════════════════════════════════════════════════════
    # Handles concepts that don't exist as columns but are computed from
    # schema relationships:
    #   "utilization" = encounters per member per time period
    #   "ALOS" / "length of stay" = days between admit and discharge
    #   "per member" / "per capita" = metric / COUNT(DISTINCT MEMBER_ID)
    # ═══════════════════════════════════════════════════════════════════════

    def _build_derived_concepts(self) -> Dict[str, Dict]:
        """Build a registry of derived concepts from schema structure.
        Each concept defines how to compute a value that doesn't exist as a column."""
        concepts = {}

        # Utilization: discovered from tables with visits/encounters + member IDs
        for tbl_name, profiles in self.learner.tables.items():
            col_names = {p.name.upper() for p in profiles}
            has_member = any(c in col_names for c in self.MEMBER_ID_CANDIDATES)
            has_date = any(p.is_date for p in profiles)
            row_count = self.learner.table_row_counts.get(tbl_name, 0)

            if has_member and has_date and row_count > self.MIN_TABLE_SIZE_FOR_METRIC:
                # This table can produce "per member per time" metrics
                member_col = self._resolve_member_column(tbl_name)
                date_cols = [p for p in profiles if p.is_date]
                date_col = date_cols[0].name if date_cols else None

                if date_col:
                    concepts[f'utilization_{tbl_name}'] = {
                        'keywords': ['utilization', 'utilisation', 'visits per member',
                                     'encounters per member', 'per member', 'per capita',
                                     'per patient'],
                        'type': 'per_member_per_time',
                        'table': tbl_name,
                        'member_col': member_col,
                        'date_col': date_col,
                        'row_count': row_count,
                        'description': f'{tbl_name} per member per time period',
                    }

        # ALOS: discovered from tables with admit + discharge date pairs
        for tbl_name, profiles in self.learner.tables.items():
            date_pair = self.learner.find_date_pair(tbl_name)
            if date_pair:
                start_col, end_col = date_pair
                concepts[f'duration_{tbl_name}'] = {
                    'keywords': ['length of stay', 'alos', 'average los',
                                 'processing time', 'turnaround', 'duration'],
                    'type': 'date_diff',
                    'table': tbl_name,
                    'start_col': start_col.name,
                    'end_col': end_col.name,
                    'description': f'Duration from {start_col.name} to {end_col.name}',
                }

        return concepts

    def _detect_derived_concept(self, q: str) -> Optional[Tuple[str, Dict]]:
        """Check if the question references a derived concept.
        Returns (concept_key, concept_def) or None."""
        q_lower = q.lower()
        best_key = None
        best_score = 0
        for key, concept in self._derived_concepts.items():
            score = 0
            for kw in concept['keywords']:
                if kw in q_lower:
                    score += len(kw)  # longer keyword matches = stronger signal
            if score > best_score:
                best_score = score
                best_key = key
        if best_key and best_score >= 5:
            return best_key, self._derived_concepts[best_key]
        return None

    def _sql_derived_concept(self, q: str, fact: str,
                              concept_key: str, concept: Dict) -> Dict:
        """Generate SQL for a derived concept.
        This replaces the hardcoded _sql_utilization handler."""
        q_lower = q.lower()

        if concept['type'] == 'per_member_per_time':
            tbl = concept['table']
            member_col = concept['member_col']
            date_col = concept['date_col']

            # Time granularity
            if 'month' in q_lower:
                bucket = f"SUBSTR({date_col}, 1, 7)"
                label = 'month'
            elif 'quarter' in q_lower:
                bucket = (f"SUBSTR({date_col}, 1, 4) || '-Q' || "
                          f"((CAST(SUBSTR({date_col}, 6, 2) AS INTEGER) - 1) / 3 + 1)")
                label = 'quarter'
            else:
                bucket = f"SUBSTR({date_col}, 1, 4)"
                label = 'year'

            # Detect grouping dimension
            dim_term = None
            for pat in [r'by\s+(\w+(?:\s+\w+)?)', r'per\s+(\w+)']:
                m = re.search(pat, q_lower)
                if m:
                    gt = m.group(1).lower().strip()
                    if gt not in ('member', 'patient', 'year', 'month', 'quarter', 'time'):
                        dim_term = gt
                        break

            if dim_term:
                dim = self._discover_dimension(q, tbl)
                if dim:
                    dim_col = dim['column']
                    if dim.get('needs_join'):
                        dim_tbl = dim.get('table', tbl)
                        on_clause = self._dim_on_clause(dim, 'e', 'd')
                        sql = (
                            f"WITH member_util AS ("
                            f"SELECT e.{member_col}, d.{dim_col}, {bucket} as period, COUNT(*) as visits "
                            f"FROM {tbl} e JOIN {dim_tbl} d ON {on_clause} "
                            f"WHERE {date_col} IS NOT NULL AND {date_col} != '' "
                            f"GROUP BY e.{member_col}, d.{dim_col}, period"
                            f") "
                            f"SELECT {dim_col}, ROUND(AVG(visits), 2) as avg_visits_per_member_{label}, "
                            f"COUNT(DISTINCT {member_col}) as unique_members, "
                            f"SUM(visits) as total_visits "
                            f"FROM member_util GROUP BY {dim_col} "
                            f"ORDER BY avg_visits_per_member_{label} DESC LIMIT {self.DEFAULT_BREAKDOWN_LIMIT};"
                        )
                        return {'sql': sql, 'tables_used': [tbl, dim_tbl],
                                'confidence': 0.85, 'intent': 'aggregation',
                                'explanation': f'{tbl} utilization per member per {label} by {dim_col}'}
                    else:
                        sql = (
                            f"WITH member_util AS ("
                            f"SELECT {member_col}, {dim_col}, {bucket} as period, COUNT(*) as visits "
                            f"FROM {tbl} WHERE {date_col} IS NOT NULL AND {date_col} != '' "
                            f"GROUP BY {member_col}, {dim_col}, period"
                            f") "
                            f"SELECT {dim_col}, ROUND(AVG(visits), 2) as avg_visits_per_member_{label}, "
                            f"COUNT(DISTINCT {member_col}) as unique_members, "
                            f"SUM(visits) as total_visits "
                            f"FROM member_util GROUP BY {dim_col} "
                            f"ORDER BY avg_visits_per_member_{label} DESC LIMIT {self.DEFAULT_BREAKDOWN_LIMIT};"
                        )
                        return {'sql': sql, 'tables_used': [tbl],
                                'confidence': 0.85, 'intent': 'aggregation',
                                'explanation': f'{tbl} utilization per member per {label} by {dim_col}'}

            # No dimension — overall utilization per member per time
            sql = (
                f"WITH member_util AS ("
                f"SELECT {member_col}, {bucket} as period, COUNT(*) as visits "
                f"FROM {tbl} WHERE {date_col} IS NOT NULL AND {date_col} != '' "
                f"GROUP BY {member_col}, period"
                f") "
                f"SELECT period, ROUND(AVG(visits), 2) as avg_visits_per_member, "
                f"COUNT(DISTINCT {member_col}) as unique_members, "
                f"SUM(visits) as total_visits "
                f"FROM member_util GROUP BY period ORDER BY period LIMIT {self.DEFAULT_TREND_LIMIT};"
            )
            return {'sql': sql, 'tables_used': [tbl],
                    'confidence': 0.85, 'intent': 'aggregation',
                    'explanation': f'{tbl} utilization per member per {label}'}

        elif concept['type'] == 'date_diff':
            tbl = concept['table']
            start = concept['start_col']
            end = concept['end_col']
            dim = self._discover_dimension(q, tbl)
            if dim:
                dim_col = dim['column']
                if dim.get('needs_join'):
                    dim_tbl = dim.get('table', tbl)
                    on_clause = self._dim_on_clause(dim, 't', 'd')
                    sql = (
                        f"SELECT d.{dim_col}, "
                        f"ROUND(AVG(julianday(t.{end}) - julianday(t.{start})), 2) as avg_days, "
                        f"COUNT(*) as record_count "
                        f"FROM {tbl} t JOIN {dim_tbl} d ON {on_clause} "
                        f"WHERE t.{start} != '' AND t.{end} != '' "
                        f"GROUP BY d.{dim_col} ORDER BY avg_days DESC LIMIT {self.DEFAULT_BREAKDOWN_LIMIT};"
                    )
                    return {'sql': sql, 'tables_used': [tbl, dim_tbl],
                            'confidence': 0.85, 'intent': 'aggregation',
                            'explanation': f'Average duration ({start}→{end}) by {dim_col}'}
                else:
                    sql = (
                        f"SELECT {dim_col}, "
                        f"ROUND(AVG(julianday({end}) - julianday({start})), 2) as avg_days, "
                        f"COUNT(*) as record_count "
                        f"FROM {tbl} WHERE {start} != '' AND {end} != '' "
                        f"GROUP BY {dim_col} ORDER BY avg_days DESC LIMIT {self.DEFAULT_BREAKDOWN_LIMIT};"
                    )
                    return {'sql': sql, 'tables_used': [tbl],
                            'confidence': 0.85, 'intent': 'aggregation',
                            'explanation': f'Average duration ({start}→{end}) by {dim_col}'}
            else:
                sql = (
                    f"SELECT ROUND(AVG(julianday({end}) - julianday({start})), 2) as avg_days, "
                    f"ROUND(MIN(julianday({end}) - julianday({start})), 2) as min_days, "
                    f"ROUND(MAX(julianday({end}) - julianday({start})), 2) as max_days, "
                    f"COUNT(*) as record_count "
                    f"FROM {tbl} WHERE {start} != '' AND {end} != '';"
                )
                return {'sql': sql, 'tables_used': [tbl],
                        'confidence': 0.85, 'intent': 'aggregation',
                        'explanation': f'Average duration from {start} to {end}'}

        return None

    # ═══════════════════════════════════════════════════════════════════════
    # ARCHITECTURAL LAYER 3: DATA-AWARE THRESHOLDS
    # ═══════════════════════════════════════════════════════════════════════

    def _compute_adaptive_having(self, table: str, group_col: str,
                                  min_useful: int = 10,
                                  where_clause: str = None,
                                  count_expr: str = None) -> int:
        """Compute HAVING threshold based on actual data distribution.
        Instead of hardcoding 5 or 10, adapts to the rarity of events.

        If a typical group has 100 records, HAVING >= 10 is reasonable.
        If a typical group has 2 records, HAVING >= 2 is better.
        Goal: keep enough groups to produce >= min_useful result rows.

        Args:
            table: Table or subquery to analyze
            group_col: Column to GROUP BY
            min_useful: Minimum number of groups to retain
            where_clause: Optional WHERE filter (e.g., "VISIT_TYPE = 'INPATIENT'")
            count_expr: Optional COUNT expression (e.g., "COUNT(DISTINCT MEMBER_ID)")
                         defaults to "COUNT(*)"
        """
        try:
            import sqlite3
            db_path = getattr(self.learner, 'db_path', None)
            if not db_path:
                return 3

            conn = sqlite3.connect(db_path)
            # Get the distribution of group sizes
            cnt = count_expr or 'COUNT(*)'
            where = f" WHERE {where_clause}" if where_clause else ""
            cursor = conn.execute(
                f"SELECT {cnt} as grp_size FROM {table}{where} "
                f"GROUP BY {group_col} ORDER BY grp_size")
            sizes = [row[0] for row in cursor.fetchall()]
            conn.close()

            if not sizes:
                return 1

            total_groups = len(sizes)
            # Find the threshold that keeps at least min_useful groups
            target_keep = min(min_useful, total_groups)
            if target_keep <= 0:
                return 1

            # Sort descending — we want to keep the top groups
            sizes.sort(reverse=True)
            if target_keep <= len(sizes):
                # The threshold should be <= the size of the target_keep-th group
                return max(1, sizes[min(target_keep - 1, len(sizes) - 1)])
            return 1
        except Exception:
            return 3

    def _build_concept_table_index(self) -> Dict[str, Dict[str, int]]:
        """Pre-build a mapping from concept words → {table: relevance_score}.

        Scans ALL schema metadata to create a comprehensive index:
        1. Column names → table (split on underscores: CLAIM_STATUS → claim, status)
        2. Categorical sample values → table (DENIED → claims, INPATIENT → encounters)
        3. Semantic tags → table
        4. Table names themselves (singular + plural forms)

        This replaces ad-hoc word matching at query time with a pre-computed
        index that catches concepts the question mentions even when the table
        name itself never appears in the question.
        """
        index = defaultdict(lambda: defaultdict(int))  # word → {table → score}
        stop_words = {'id', 'key', 'code', 'date', 'time', 'type', 'name',
                      'the', 'and', 'for', 'with', 'from', 'into', 'by', 'at',
                      'num', 'is', 'of', 'a', 'an', 'to', 'in', 'on', 'or'}

        for tbl_name, profiles in self.learner.tables.items():
            tbl_lower = tbl_name.lower()
            # Table name → strong signal
            index[tbl_lower][tbl_name] += 8
            tbl_stem = tbl_lower.rstrip('s') if len(tbl_lower) > 3 else tbl_lower
            if tbl_stem != tbl_lower:
                index[tbl_stem][tbl_name] += 8

            for p in profiles:
                col_lower = p.name.lower()
                # Column name words → moderate signal
                col_words = col_lower.replace('_', ' ').split()
                for w in col_words:
                    if w not in stop_words and len(w) > 2:
                        index[w][tbl_name] += 2

                # Categorical sample values → strong signal
                # E.g., "DENIED" in claims.CLAIM_STATUS → "denied" → claims
                if p.is_categorical and p.sample_values:
                    for val in set(p.sample_values):
                        if val and isinstance(val, str):
                            vl = val.lower().replace('_', ' ')
                            for vw in vl.split():
                                if vw not in stop_words and len(vw) > 2:
                                    index[vw][tbl_name] += 4
                            # Also add common derivations:
                            # "denied" → "denial", "deny"
                            if vl.endswith('ed') and len(vl) > 4:
                                stem = vl[:-2]
                                index[stem][tbl_name] += 4
                                index[stem + 'al'][tbl_name] += 4  # denied → denial
                                index[stem + 'y'][tbl_name] += 3   # denied → deny
                            if vl.endswith('ed') and len(vl) > 3:
                                index[vl[:-1]][tbl_name] += 3       # approved → approve
                            # "inpatient" → "inpatient" (exact)
                            index[vl][tbl_name] += 5

                # Numeric column names → signal for aggregation queries
                if p.is_numeric and not p.is_id:
                    for w in col_words:
                        if w in ('amount', 'cost', 'paid', 'billed', 'allowed',
                                 'charge', 'price', 'revenue', 'total', 'balance',
                                 'copay', 'coinsurance', 'deductible', 'duration',
                                 'quantity', 'dosage', 'refills'):
                            index[w][tbl_name] += 5

                # Semantic tags
                for tag in (p.semantic_tags or []):
                    tag_l = tag.lower()
                    if tag_l not in stop_words and len(tag_l) > 2:
                        index[tag_l][tbl_name] += 3

        # Also do a quick DB scan for values not in sample_values
        # (handles rare categories like DENIED that may be underrepresented)
        try:
            import sqlite3
            db_path = getattr(self.learner, 'db_path', None)
            if db_path:
                conn = sqlite3.connect(db_path)
                for tbl_name, profiles in self.learner.tables.items():
                    for p in profiles:
                        if p.is_categorical and 'status' in p.name.lower():
                            try:
                                cursor = conn.execute(
                                    f"SELECT DISTINCT {p.name} FROM {tbl_name} "
                                    f"WHERE {p.name} IS NOT NULL LIMIT 20")
                                for row in cursor.fetchall():
                                    val = row[0]
                                    if val and isinstance(val, str):
                                        vl = val.lower()
                                        index[vl][tbl_name] += 5
                                        if vl.endswith('ed') and len(vl) > 4:
                                            index[vl[:-2]][tbl_name] += 4
                                            index[vl[:-2] + 'al'][tbl_name] += 4
                                            index[vl[:-2] + 'y'][tbl_name] += 3
                            except Exception:
                                pass
                conn.close()
        except Exception:
            pass

        # ── DOMAIN DISAMBIGUATION in concept index ──
        # In healthcare analytics, "provider" typically means insurance plan
        # (Medicaid, Medicare, PPO, HMO), NOT individual practitioners.
        # Redirect "provider" scoring toward tables with PLAN_TYPE columns.
        if self._domain_disambiguations:
            for key, dis in self._domain_disambiguations.items():
                dis_table = dis['table']
                for trigger in dis['triggers']:
                    # Boost the correct table for ambiguous terms
                    index[trigger][dis_table] += self.DISAMBIGUATION_INDEX_BOOST
                    # Add insurance plan values as triggers
                    for val in dis.get('sample_values', []):
                        val_lower = str(val).lower()
                        for vw in val_lower.split():
                            if len(vw) > 2:
                                index[vw][dis_table] += self.INSURANCE_VALUE_INDEX_BOOST

        # ── Entity synonym boost ──
        # "patient", "patients" → members table (not providers.ACCEPTS_NEW_PATIENTS)
        # "female", "male" → members table (GENDER column)
        entity_synonyms = {
            'members': ['patient', 'patients', 'enrollee', 'enrollees',
                        'beneficiary', 'beneficiaries', 'female', 'male',
                        'gender', 'demographic', 'demographics'],
            'encounters': ['visit', 'visits', 'admission', 'admissions',
                           'inpatient', 'outpatient', 'emergency'],
            'claims': ['claim', 'billing', 'billed', 'paid', 'denied',
                       'denial', 'reimbursement'],
            'prescriptions': ['prescription', 'medication', 'drug', 'pharmacy',
                              'rx', 'refill'],
            'diagnoses': ['diagnosis', 'icd', 'condition', 'disease'],
            'appointments': ['appointment', 'scheduling', 'booking',
                             'no-show', 'cancellation'],
            'referrals': ['referral', 'authorization', 'specialist'],
        }
        for tbl, synonyms in entity_synonyms.items():
            if tbl in self.learner.tables:
                for syn in synonyms:
                    index[syn][tbl] += 8  # Strong synonym boost

        # ── Healthcare domain vocabulary boost ──
        # Import domain knowledge from the dynamic engine so the semantic engine
        # understands healthcare business terms like "prior authorization",
        # "high-value claim", "polypharmacy", "specialist", etc.
        # This bridges the gap between schema-agnostic learning and domain intelligence.
        try:
            from dynamic_sql_engine import DOMAIN_CONCEPTS, SYNONYMS as DYN_SYNONYMS, TABLE_KEYWORDS
            # Common English words that leak from compound terms and pollute
            # the concept index (e.g., "no-show" → "no", "show")
            _domain_stop = {'no', 'show', 'second', 'new', 'high', 'low',
                            'annual', 'care', 'date', 'type', 'size',
                            'rate', 'time', 'code', 'score', 'class',
                            'day', 'days', 'home', 'mail', 'order',
                            'value', 'claim', 'visit', 'left',
                            # Tuva: additional stop words for expanded vocabulary
                            'use', 'based', 'self', 'long', 'short',
                            'term', 'total', 'per', 'all', 'open',
                            'zero', 'mid', 'pre', 'non', 'out',
                            'fee', 'step', 'drug', 'cost', 'pay',
                            'fill', 'lab', 'check', 'well', 'office'}

            # 1. DOMAIN_CONCEPTS: healthcare business terms → {tables, conds}
            for term, concept in DOMAIN_CONCEPTS.items():
                term_words = term.lower().replace('-', ' ').split()
                for tbl in concept.get('tables', []):
                    if tbl in self.learner.tables:
                        # Boost each word in the term toward its table
                        for w in term_words:
                            if len(w) > 2 and w not in _domain_stop:
                                index[w][tbl] += 6
                        # Also index the full multi-word term (always safe)
                        if len(term_words) > 1:
                            index[term.lower()][tbl] += 10

            # 2. TABLE_KEYWORDS: comprehensive per-table keyword lists
            for tbl, keywords in TABLE_KEYWORDS.items():
                if tbl in self.learner.tables:
                    for kw in keywords:
                        kw_words = kw.lower().split()
                        for w in kw_words:
                            if len(w) > 2 and w not in _domain_stop:
                                index[w][tbl] += 5
                        if len(kw_words) > 1:
                            index[kw.lower()][tbl] += 8

            # 3. SYNONYMS: NL terms → column names (used for column resolution)
            for term, col_list in DYN_SYNONYMS.items():
                term_words = term.lower().split()
                for col_name in col_list:
                    # Find which table has this column
                    for tbl, profiles in self.learner.tables.items():
                        for p in profiles:
                            if p.name == col_name:
                                for w in term_words:
                                    if len(w) > 2 and w not in _domain_stop:
                                        index[w][tbl] += 4
                                break

            logger.info("Domain vocabulary: loaded %d concepts, %d table keywords, %d synonyms into concept index",
                        len(DOMAIN_CONCEPTS), sum(len(v) for v in TABLE_KEYWORDS.values()),
                        len(DYN_SYNONYMS))
        except ImportError:
            logger.debug("Dynamic engine not available; domain vocabulary not loaded")

        return dict(index)

    def compose(self, question: str, intent: Dict, columns: List[Dict],
                tables: List[Dict], values: List[Dict],
                computed: List[Dict]) -> Dict[str, Any]:
        """
        Main entry point: compose SQL from semantic understanding.

        Args:
            question: original natural language question
            intent: {intent, confidence, all_scores}
            columns: [{column, table, score, meta}, ...]
            tables: [{table, score, row_count}, ...]
            values: [{table, column, value, score}, ...]
            computed: [{expr, alias, table, description}, ...]

        Returns: {sql, tables_used, confidence, explanation}
        """
        q = question.lower()
        intent_type = intent.get('intent', 'lookup')

        # ── LEARNED PATTERN CHECK (fastest path) ──
        # If we've seen a structurally similar question succeed before, reuse it
        if self._schema_intel:
            _learned = self._schema_intel.find_learned_pattern(question)
            if _learned and _learned.get('count', 0) >= 2:
                # Only reuse patterns that have succeeded at least twice
                logger.info("PatternLearner: reusing learned pattern (used %dx): %s",
                            _learned['count'], question[:60])
                _learned_result = {
                    'sql': _learned['sql'],
                    'tables_used': _learned['tables'],
                    'confidence': 0.95,
                    'explanation': f"Reused learned pattern (matched {_learned['count']}x)",
                    '_pattern_reuse': True,
                }
                return self._validate_and_repair_sql(question, _learned_result)

        # ── INTELLIGENT QUERY PLANNER (first pass) ──
        # Attempts structured decomposition: NL → Plan → Resolved → SQL.
        # Uses auto-discovered knowledge from SchemaIntelligence when available.
        # Falls back to legacy path if confidence is low or SQL is invalid.
        try:
            from query_planner import plan_and_compose
            from dynamic_sql_engine import COMPUTED_COLUMNS as _CC

            # Use auto-discovered knowledge if available, otherwise fall back to manual
            if self._schema_intel:
                _dc = self._schema_intel.domain_concepts   # Auto-discovered + manual merged
                _syn = self._schema_intel.synonyms          # Auto-discovered + manual merged
            else:
                from dynamic_sql_engine import DOMAIN_CONCEPTS as _dc, SYNONYMS as _syn

            planner_result = plan_and_compose(
                question=question,
                schema_learner=self.learner,
                concept_index=self._concept_table_index,
                domain_concepts=_dc,
                synonyms=_syn,
                computed_columns=_CC,
                kg=self.kg,
                schema_intel=self._schema_intel,
            )
            if planner_result and planner_result.get('confidence', 0) >= 0.5:
                _planner_sql = planner_result.get('sql', '')
                _planner_valid = False
                _vrows = []

                if _planner_sql:
                    # ── Self-healing SQL validation ──
                    # Instead of simple try/catch, use SQLSelfHealer to fix errors
                    if self._schema_intel:
                        _heal_ctx = {
                            'primary_table': planner_result.get('tables_used', [''])[0],
                            'tables_used': planner_result.get('tables_used', []),
                            'intent': intent_type,
                        }
                        _heal_result = self._schema_intel.heal_sql(_planner_sql, _heal_ctx)
                        _vrows = _heal_result.get('rows', [])
                        if _vrows:
                            _planner_valid = True
                            # Use the healed SQL if it was fixed
                            if _heal_result.get('healed'):
                                planner_result['sql'] = _heal_result['sql']
                                planner_result['_healed'] = True
                                planner_result['_heal_log'] = _heal_result.get('heal_log', [])
                                logger.info("SQLSelfHealer fixed planner SQL in %d attempts: %s",
                                            _heal_result['attempts'], question[:60])
                            else:
                                logger.info("QueryPlanner SQL validated (%d rows, conf=%.2f): %s",
                                            len(_vrows), planner_result['confidence'], question[:80])
                        else:
                            _err = _heal_result.get('error', 'empty result')
                            logger.debug("QueryPlanner SQL failed after healing: %s", _err)
                    else:
                        # Fallback: simple try/catch validation (no healer)
                        try:
                            db_path = getattr(self.learner, 'db_path', None)
                            if db_path:
                                import sqlite3 as _sq3
                                _vconn = _sq3.connect(db_path)
                                _vcur = _vconn.cursor()
                                _vcur.execute(_planner_sql)
                                _vrows = _vcur.fetchall()
                                _vconn.close()
                                if _vrows:
                                    _planner_valid = True
                                    logger.info("QueryPlanner SQL validated (%d rows, conf=%.2f): %s",
                                                len(_vrows), planner_result['confidence'], question[:80])
                                else:
                                    logger.debug("QueryPlanner SQL returned 0 rows, deferring")
                        except Exception as _ve:
                            logger.debug("QueryPlanner SQL failed validation: %s", _ve)

                # Quality gate: if question asks for a rate/percentage but
                # planner SQL has no CASE WHEN, it missed the metric — defer
                if _planner_valid:
                    _rate_words = {'rate', 'rates', 'percentage', 'percent', '%'}
                    _q_words_check = set(re.findall(r'[a-z%]+', q))
                    _asks_rate = bool(_q_words_check & _rate_words)
                    _planner_sql_final = planner_result.get('sql', '')
                    _sql_has_rate = ('CASE WHEN' in _planner_sql_final.upper() or
                                     'RATE' in _planner_sql_final.upper() or
                                     '100.0' in _planner_sql_final)
                    if _asks_rate and not _sql_has_rate:
                        logger.info("QueryPlanner: rate requested but SQL has no rate calc, deferring")
                        _planner_valid = False

                if _planner_valid:
                    # Record success for pattern learning
                    if self._schema_intel:
                        self._schema_intel.record_success(
                            question, planner_result.get('sql', ''),
                            planner_result.get('tables_used', []),
                            len(_vrows) if _vrows else 0)
                    return self._validate_and_repair_sql(question, planner_result)
                else:
                    logger.debug("QueryPlanner deferred (SQL invalid/incomplete), falling to legacy")
            else:
                logger.debug("QueryPlanner deferred (conf=%.2f), falling to legacy",
                             planner_result.get('confidence', 0) if planner_result else 0)
        except Exception as e:
            logger.warning("QueryPlanner import/run error: %s", e)

        # ── SPECIAL PATTERN DETECTION (legacy fallback) ──
        # These patterns need specialized SQL that the standard SELECT/WHERE/GROUP
        # pipeline cannot produce, like HAVING, subqueries, CASE-based ratios, etc.
        special = self._detect_special_pattern(q, intent_type, columns, tables, values)
        if special:
            return self._validate_and_repair_sql(question, special)

        # Step 1: Determine primary tables
        primary_tables = self._resolve_tables(tables, columns, values, question)

        # Step 1a: Cross-table cost redirect — if the question mentions cost
        # but the primary table lacks financial columns, swap in a table that has them.
        _cost_signals = {'cost', 'spend', 'spending', 'expense', 'charge', 'paid',
                         'billed', 'amount', 'dollar', 'price', 'payment', 'revenue'}
        _q_words = set(re.findall(r'[a-z]+', q))
        if _q_words & _cost_signals:
            _has_cost_col = False
            for t in primary_tables:
                for p in self.learner.tables.get(t, []):
                    if p.is_numeric and any(kw in p.name.lower() for kw in
                            ['amount', 'cost', 'paid', 'billed', 'allowed', 'price', 'spend']):
                        _has_cost_col = True
                        break
                if _has_cost_col:
                    break
            if not _has_cost_col:
                # Find table with cost columns
                for t_name, profiles in self.learner.tables.items():
                    if t_name in primary_tables:
                        continue
                    for p in profiles:
                        if p.is_numeric and any(kw in p.name.lower() for kw in
                                ['amount', 'cost', 'paid', 'billed', 'allowed']):
                            logger.info("Cost redirect: swapping '%s' for '%s' (has cost columns)",
                                        primary_tables[0], t_name)
                            primary_tables = [t_name] + [t for t in primary_tables if t != t_name][:2]
                            _has_cost_col = True
                            break
                    if _has_cost_col:
                        break

        # Step 1b: Concept-aware table expansion — if the question references
        # a concept (cost, amount, revenue, etc.) but the selected tables lack
        # matching columns, pull in the table that has them.
        primary_tables = self._expand_tables_for_concepts(q, primary_tables, columns)

        # Step 2: Determine what to SELECT
        select_parts, group_cols, needs_group = self._build_select(
            q, intent_type, columns, primary_tables, computed, values
        )

        # Step 2b: Cross-table group resolution — if group column is in another
        # table, pull that table into the join list
        if needs_group and group_cols:
            for gc in group_cols:
                gc_clean = re.sub(r'^\w\.', '', gc)  # strip alias
                found_in_primary = False
                for t in primary_tables:
                    for p in self.learner.tables.get(t, []):
                        if p.name == gc_clean:
                            found_in_primary = True
                            break
                    if found_in_primary:
                        break
                if not found_in_primary:
                    # Search ALL tables for this column
                    for t, profiles in self.learner.tables.items():
                        if t not in primary_tables:
                            for p in profiles:
                                if p.name == gc_clean:
                                    primary_tables.append(t)
                                    found_in_primary = True
                                    break
                        if found_in_primary:
                            break

        # Step 3: Build WHERE conditions from matched values and question parsing
        conditions = self._build_conditions(q, values, columns, primary_tables, intent_type)

        # Step 4: Build FROM clause with JOINs
        from_clause, join_tables = self._build_from(primary_tables, columns, conditions)

        # Step 5: Build ORDER BY
        order_clause = self._build_order(q, intent_type, select_parts, group_cols)

        # Step 6: Build LIMIT
        limit = self._build_limit(q, intent_type, needs_group)

        # Step 7: Qualify ambiguous columns for multi-table queries
        if len(primary_tables) > 1:
            primary = primary_tables[0]
            # Build a set of columns per table for qualification
            col_to_tables = defaultdict(list)
            for t in primary_tables:
                for p in self.learner.tables.get(t, []):
                    col_to_tables[p.name].append(t)

            def _qualify(expr):
                """Add table prefix to ambiguous column references."""
                for col, tbls in col_to_tables.items():
                    if len(tbls) > 1 and col in expr:
                        # Use primary table's alias prefix
                        alias = primary[0]
                        # Don't qualify inside function calls that already have prefixes
                        if f"{alias}.{col}" not in expr and f".{col}" not in expr:
                            expr = re.sub(r'\b' + re.escape(col) + r'\b', f"{alias}.{col}", expr)
                return expr

            select_parts = [_qualify(sp) for sp in select_parts]
            conditions = [_qualify(c) for c in conditions]
            group_cols = [_qualify(gc) for gc in group_cols]

        # Step 8: Assemble
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        group_by = f" GROUP BY {', '.join(group_cols)}" if needs_group and group_cols else ""

        sql = f"SELECT {', '.join(select_parts)} FROM {from_clause}{where}{group_by}{order_clause}{limit};"

        all_tables = list(dict.fromkeys([t for t in primary_tables] +
                                         [t for t in join_tables]))

        confidence = self._calculate_confidence(intent, columns, tables, values, conditions, question)

        result = {
            'sql': sql,
            'tables_used': all_tables,
            'confidence': confidence,
            'intent': intent_type,
            'explanation': self._explain(question, intent_type, all_tables,
                                        columns, conditions, select_parts),
        }
        return self._validate_and_repair_sql(question, result)

    def _validate_and_repair_sql(self, question: str, result: Dict) -> Dict:
        """Post-generation validation: catch nonsensical SQL before returning.

        This is the systemic safety net that prevents entire classes of bugs:
        1. Rate queries MUST have CASE WHEN or rate-related columns
        2. SQL columns must be relevant to the question's domain concepts
        3. Detect wrong-table selection (e.g., appointments for denial queries)

        If validation fails, attempt to regenerate via a corrected path.
        """
        sql = result.get('sql', '')
        q = question.lower()
        if not sql:
            return result

        # ── Check 1: Rate queries must produce rate-like output ──
        is_rate_query = any(w in q for w in ('rate', 'denial', 'denied', 'approval',
                                              'approved', 'rejection', 'rejected',
                                              'percentage', 'percent'))
        if is_rate_query and 'CASE WHEN' not in sql and '_rate' not in sql.lower() and 'pct' not in sql.lower():
            # Rate query without rate computation — likely wrong table or wrong path
            logger.debug("Validation: rate query lacks rate computation, attempting repair")
            result['data_warnings'] = result.get('data_warnings', []) + [
                'Rate query detected but SQL lacks rate computation — results may be aggregate counts'
            ]

        # ── Check 2: Column relevance — flag when SQL uses columns completely
        #    unrelated to the question's concepts ──
        # Extract column names from SQL (rough heuristic)
        sql_cols = set(re.findall(r'\b([A-Z][A-Z_]+(?:_[A-Z]+)*)\b', sql))
        # Check for concept mismatches
        concept_mismatches = []
        if any(w in q for w in ('denial', 'denied', 'deny', 'claim')):
            if 'DURATION_MINUTES' in sql_cols and 'CLAIM_STATUS' not in sql_cols:
                concept_mismatches.append('Query about claims/denial but SQL uses DURATION_MINUTES (appointment metric)')
        if any(w in q for w in ('readmission', 'readmit')):
            if 'ADMIT_DATE' not in sql_cols and 'DISCHARGE_DATE' not in sql_cols:
                if 'readmission' not in sql.lower():
                    concept_mismatches.append('Readmission query but SQL lacks admission/discharge date logic')
        if any(w in q for w in ('cost', 'billed', 'paid', 'amount', 'spending')):
            has_money_col = any(c in sql_cols for c in self.FINANCIAL_COLUMNS)
            if not has_money_col and 'amount' not in sql.lower() and 'cost' not in sql.lower():
                concept_mismatches.append('Cost query but SQL lacks financial columns')

        if concept_mismatches:
            logger.warning("SQL concept mismatch: %s", concept_mismatches)
            result['data_warnings'] = result.get('data_warnings', []) + concept_mismatches
            # Attempt auto-repair: if we can detect the right table, regenerate
            repaired = self._attempt_sql_repair(question, result, concept_mismatches)
            if repaired:
                return repaired

        # ── Check 3: Trend/dimension queries must produce GROUP BY ──
        is_trend = any(re.search(p, q) for p in [r'\btrend\b', r'\bover\s+time\b',
                        r'\bby\s+\w+', r'\bper\s+\w+', r'\bacross\s+'])
        has_group = 'GROUP BY' in sql
        if is_trend and not has_group:
            logger.warning("Validation: trend/dimension query lacks GROUP BY — likely returning aggregate")
            result['data_warnings'] = result.get('data_warnings', []) + [
                'Trend/dimension query detected but SQL lacks GROUP BY — result may be a single aggregate row'
            ]
            # Lower confidence to reflect the mismatch
            if isinstance(result.get('confidence'), (int, float)):
                result['confidence'] = min(result['confidence'], 0.5)

        return result

    def _attempt_sql_repair(self, question: str, original: Dict,
                            mismatches: List[str]) -> Optional[Dict]:
        """Try to regenerate SQL when validation detects a concept mismatch.
        Uses the concept index to find the correct table and re-routes."""
        q = question.lower()

        # If the mismatch is "wrong table for denial/claim queries", find correct table
        if any('DURATION_MINUTES' in m for m in mismatches):
            # The SQL went to appointments instead of claims — force claims
            correct_table = self._identify_fact_table(q, [])
            if correct_table and correct_table != original.get('tables_used', [''])[0]:
                logger.info("SQL repair: switching from %s to %s",
                           original.get('tables_used', ['?'])[0], correct_table)
                # Try to find rate info on the correct table
                rate_info = self._detect_rate_info(q, [], [correct_table])
                if rate_info:
                    status_col, target_val, _ = rate_info
                    dim = self._discover_dimension(q, correct_table)
                    safe_val = re.sub(r'[^a-zA-Z0-9_]', '_', target_val.lower())
                    if dim:
                        dim_col = dim['column']
                        if dim.get('needs_join'):
                            dim_table = dim.get('table', correct_table)
                            on_clause = self._dim_on_clause(dim, 'c', 'd')
                            sql = (
                                f"SELECT d.{dim_col}, COUNT(*) as total, "
                                f"SUM(CASE WHEN {status_col} = '{target_val}' THEN 1 ELSE 0 END) as {safe_val}_count, "
                                f"ROUND(100.0 * SUM(CASE WHEN {status_col} = '{target_val}' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as {safe_val}_rate "
                                f"FROM {correct_table} c JOIN {dim_table} d ON {on_clause} "
                                f"GROUP BY d.{dim_col} ORDER BY {safe_val}_rate DESC LIMIT {self.DEFAULT_BREAKDOWN_LIMIT};"
                            )
                        else:
                            sql = (
                                f"SELECT {dim_col}, COUNT(*) as total, "
                                f"SUM(CASE WHEN {status_col} = '{target_val}' THEN 1 ELSE 0 END) as {safe_val}_count, "
                                f"ROUND(100.0 * SUM(CASE WHEN {status_col} = '{target_val}' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as {safe_val}_rate "
                                f"FROM {correct_table} GROUP BY {dim_col} ORDER BY {safe_val}_rate DESC LIMIT {self.DEFAULT_BREAKDOWN_LIMIT};"
                            )
                        return {
                            'sql': sql, 'tables_used': [correct_table],
                            'confidence': 0.85, 'intent': 'rate',
                            'explanation': f'Repaired: {target_val.lower()} rate by {dim_col} from {correct_table}',
                            'data_warnings': ['Auto-repaired: corrected table selection'],
                        }
        return None

    def _resolve_tables(self, tables: List[Dict], columns: List[Dict],
                        values: List[Dict], question: str = '') -> List[str]:
        """
        Determine which tables to query, using semantic scores.
        Conservative approach: prefer fewer tables, only add JOINs when needed.
        """
        table_scores = defaultdict(float)

        # From table matching — give HIGH weight to the top table match
        for i, t in enumerate(tables):
            weight = 5.0 if i == 0 else 2.0
            table_scores[t['table']] += t['score'] * weight

        # From column matching — only count high-confidence matches
        col_per_table = defaultdict(int)
        for c in columns[:10]:
            if c['score'] >= 0.1:
                table_scores[c['table']] += c['score'] * 1.5
                col_per_table[c['table']] += 1

        # From value matching
        for v in values[:5]:
            if v['score'] >= 0.15:
                table_scores[v['table']] += v['score'] * 1.0

        if not table_scores:
            # Pick the largest table as default — data-driven, not hardcoded
            biggest = max(self.learner.table_row_counts.items(),
                          key=lambda x: x[1], default=(list(self.learner.tables.keys())[0], 0))
            return [biggest[0]]

        # Boost: if a table name is explicitly mentioned in the question, strong signal
        q_words = set(re.findall(r'[a-z]+', question.lower()))
        for table in list(table_scores.keys()):
            # Table name or singular form in question
            if table in q_words or table.rstrip('s') in q_words:
                table_scores[table] += 5.0

        # Boost from concept index: entity synonyms like "patient" → members
        if hasattr(self, '_concept_table_index') and self._concept_table_index:
            for word in q_words:
                if word in self._concept_table_index:
                    for tbl, score in self._concept_table_index[word].items():
                        if tbl in table_scores or tbl in self.learner.tables:
                            table_scores[tbl] += min(score / 2.0, 5.0)

        # Sort by score, take top tables
        sorted_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
        primary = sorted_tables[0][0]
        result = [primary]

        # Only add secondary tables if they have STRONG independent signal
        # (>60% of primary score AND at least 2 column matches)
        for table, score in sorted_tables[1:]:
            if (score >= sorted_tables[0][1] * 0.6 and col_per_table.get(table, 0) >= 2):
                result.append(table)

        return result[:3]

    def _expand_tables_for_concepts(self, q: str, tables: List[str],
                                     columns: List[Dict]) -> List[str]:
        """
        If the question references a concept (cost, revenue, amount, etc.) but
        the selected tables don't have matching numeric columns, find and add
        the table that does.  This ensures cross-table JOINs happen when needed.

        Uses DomainConfig (schema-driven) when available; falls back to scanning
        SchemaLearner metadata directly so it works on ANY schema.
        """
        q_lower = q.lower()

        # --- Strategy 1: Use DomainConfig (auto-discovered concepts) ---
        if self.domain_config:
            concept_hits = self.domain_config.find_concept_from_question(q)
            if not concept_hits:
                return tables

            # Collect column keywords that satisfy found concepts
            needed_keywords = set()
            for concept, _tbl, col in concept_hits:
                needed_keywords.add(col.lower())
                # Also add all keywords for the concept
                for kw in self.domain_config.concept_keywords.get(concept, []):
                    needed_keywords.add(kw.lower())

            # Check if any selected table already has a matching numeric column
            for table in tables:
                for p in self.learner.tables.get(table, []):
                    if p.is_numeric and p.name.lower() in needed_keywords:
                        return tables  # Already covered

            # Not covered — find tables via DomainConfig
            for concept, _tbl, _col in concept_hits:
                candidate_tables = self.domain_config.find_tables_with_concept(concept)
                for ct in candidate_tables:
                    if ct not in tables:
                        logger.info("Concept expansion (DomainConfig): added table '%s' for concept '%s'",
                                    ct, concept)
                        tables = tables + [ct]
                        return tables[:3]

            return tables

        # --- Strategy 2: Schema-scan fallback (no DomainConfig) ---
        # Scan all numeric columns and look for word overlap with the question
        q_words = set(re.findall(r'[a-z]+', q_lower))
        # Words that signal monetary/numeric concepts
        signal_words = q_words & {
            w for tbl_profiles in self.learner.tables.values()
            for p in tbl_profiles if p.is_numeric
            for w in p.name.lower().replace('_', ' ').split()
        }
        if not signal_words:
            return tables

        # Check if already covered
        for table in tables:
            for p in self.learner.tables.get(table, []):
                if p.is_numeric:
                    col_words = set(p.name.lower().replace('_', ' ').split())
                    if col_words & q_words:
                        return tables

        # Search all tables for best match
        best_table = None
        best_score = 0
        for tbl, profiles in self.learner.tables.items():
            if tbl in tables:
                continue
            score = 0
            for p in profiles:
                if p.is_numeric:
                    col_words = set(p.name.lower().replace('_', ' ').split())
                    overlap = col_words & q_words
                    score += len(overlap) * 2
                    if 'currency' in p.semantic_tags:
                        score += 2
            if score > best_score:
                best_score = score
                best_table = tbl

        if best_table and best_score >= 1:
            logger.info("Concept expansion (schema-scan): added table '%s' (score=%d)",
                        best_table, best_score)
            tables = tables + [best_table]

        return tables[:3]

    def _build_select(self, q: str, intent_type: str, columns: List[Dict],
                      tables: List[str], computed: List[Dict],
                      values: List[Dict]) -> Tuple[List[str], List[str], bool]:
        """Build SELECT clause and GROUP BY columns based on intent."""
        select_parts = []
        group_cols = []
        needs_group = False
        primary = tables[0] if tables else list(self.learner.tables.keys())[0]
        alias = primary[0] if len(tables) > 1 else ''
        prefix = f"{alias}." if alias else ""

        # Extract "by X" group terms from question
        group_term = self._extract_group_term(q)

        # Handle computed columns first
        if computed:
            for comp in computed:
                if comp['alias'] == 'age_group' and ('age' in q.lower()):
                    # Age group breakdown query — detect custom interval (e.g., "by 5 years")
                    custom_interval = self._extract_age_interval(q)
                    if custom_interval:
                        # Rebuild age expression with custom interval
                        age_expr = self._age_group_case(comp['column'], interval=custom_interval)
                    else:
                        age_expr = comp['expr']
                    # Strip table alias prefix if single-table query
                    if len(tables) <= 1:
                        age_expr = re.sub(r'\b\w\.(?=\w)', '', age_expr)
                    total = self.learner.table_row_counts.get(comp['table'], 1)
                    select_parts = [
                        f"{age_expr} as age_group",
                        "COUNT(*) as count",
                        f"ROUND(100.0 * COUNT(*) / {total}, 2) as pct"
                    ]
                    group_cols = ['age_group']
                    return select_parts, group_cols, True

        if intent_type == 'count':
            if group_term:
                # Check if group term is a time unit (month, week, year, quarter)
                TIME_UNITS = {'month', 'week', 'year', 'quarter', 'day', 'monthly',
                              'weekly', 'yearly', 'quarterly', 'daily'}
                if group_term.strip().lower() in TIME_UNITS:
                    # Switch to time-based grouping
                    date_col = self._find_date_column(q, columns, tables)
                    if date_col:
                        bucket = f"SUBSTR({date_col}, 1, 7)"
                        if 'year' in group_term.lower():
                            bucket = f"SUBSTR({date_col}, 1, 4)"
                        elif 'quarter' in group_term.lower():
                            bucket = (f"SUBSTR({date_col}, 1, 4) || '-Q' || "
                                      f"((CAST(SUBSTR({date_col}, 6, 2) AS INTEGER) - 1) / 3 + 1)")
                        select_parts = [f"{bucket} as period", "COUNT(*) as count"]
                        group_cols = ['period']
                        needs_group = True
                    else:
                        select_parts = ["COUNT(*) as total_count"]
                else:
                    group_col = self._resolve_group_column(group_term, columns, tables)
                    if group_col:
                        select_parts = [group_col, "COUNT(*) as count"]
                        group_cols = [group_col]
                        needs_group = True
                    else:
                        select_parts = ["COUNT(*) as total_count"]
            else:
                select_parts = ["COUNT(*) as total_count"]

        elif intent_type == 'aggregate' or (intent_type == 'breakdown' and self._has_amount_word(q)):
            # Aggregate or breakdown with amount words → compute aggregates
            agg_func = self._detect_agg_func(q)
            num_col = self._find_numeric_column(q, columns, tables)
            if group_term:
                group_col = self._resolve_group_column(group_term, columns, tables)
                if group_col and num_col:
                    select_parts = [
                        group_col,
                        f"ROUND({agg_func}(CAST({num_col} AS REAL)), 2) as {agg_func.lower()}_{num_col.lower()}",
                        "COUNT(*) as record_count"
                    ]
                    group_cols = [group_col]
                    needs_group = True
                elif group_col:
                    select_parts = [group_col, "COUNT(*) as count"]
                    group_cols = [group_col]
                    needs_group = True
                else:
                    if num_col:
                        select_parts = [f"ROUND({agg_func}(CAST({num_col} AS REAL)), 2) as result"]
                    else:
                        select_parts = ["COUNT(*) as total_count"]
            else:
                if num_col:
                    select_parts = [
                        f"ROUND({agg_func}(CAST({num_col} AS REAL)), 2) as {agg_func.lower()}_{num_col.lower()}",
                        "COUNT(*) as record_count"
                    ]
                else:
                    select_parts = ["COUNT(*) as total_count"]

        elif intent_type == 'ranking':
            agg_func = self._detect_agg_func(q)
            num_col = self._find_numeric_column(q, columns, tables)
            group_col = self._resolve_group_column(
                group_term or self._extract_ranking_entity(q), columns, tables
            )
            if group_col:
                if num_col and agg_func != 'COUNT':
                    select_parts = [
                        group_col,
                        f"ROUND({agg_func}(CAST({num_col} AS REAL)), 2) as {agg_func.lower()}_{num_col.lower()}",
                        "COUNT(*) as record_count"
                    ]
                else:
                    select_parts = [group_col, "COUNT(*) as total_count"]
                group_cols = [group_col]
                needs_group = True
            else:
                select_parts = self._default_select(columns, tables)

        elif intent_type == 'breakdown':
            # Check if group term is a time unit → date-based grouping
            TIME_UNITS = {'month', 'week', 'year', 'quarter', 'day', 'monthly',
                          'weekly', 'yearly', 'quarterly', 'daily'}
            if group_term and group_term.strip().lower() in TIME_UNITS:
                date_col = self._find_date_column(q, columns, tables)
                if date_col:
                    bucket = f"SUBSTR({date_col}, 1, 7)"
                    if 'year' in group_term.lower():
                        bucket = f"SUBSTR({date_col}, 1, 4)"
                    elif 'quarter' in group_term.lower():
                        bucket = (f"SUBSTR({date_col}, 1, 4) || '-Q' || "
                                  f"((CAST(SUBSTR({date_col}, 6, 2) AS INTEGER) - 1) / 3 + 1)")
                    select_parts = [f"{bucket} as period", "COUNT(*) as count"]
                    group_cols = ['period']
                    needs_group = True
                else:
                    select_parts = self._default_select(columns, tables)
            else:
                group_col = self._resolve_group_column(group_term, columns, tables)
                if group_col:
                    # Add percentage calculation for breakdowns
                    select_parts = [
                        group_col,
                        "COUNT(*) as count",
                        f"ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) as pct"
                    ]
                    group_cols = [group_col]
                    needs_group = True
                else:
                    select_parts = self._default_select(columns, tables)

        elif intent_type == 'percentage':
            group_col = self._resolve_group_column(group_term, columns, tables)
            if group_col:
                select_parts = [
                    group_col,
                    "COUNT(*) as count",
                    "ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) as pct"
                ]
                group_cols = [group_col]
                needs_group = True
            else:
                # "what percentage of X are Y" — single percentage
                # Strategy 1: value matches from semantic matching
                logger.debug("Percentage S1: tables=%s values=%s", tables, [(v['table'], v['column'], v['value'], v['score']) for v in values[:5]])
                for v in values:
                    if v['table'] in tables and v['score'] >= 0.1:
                        val_text = v['value'].lower().replace('_', ' ')
                        # Guard: single-char values must match as whole words, not substrings
                        if len(val_text) <= 2:
                            is_match = bool(re.search(r'\b' + re.escape(val_text) + r'\b', q))
                        else:
                            is_match = val_text in q or v['value'].lower() in q
                        if is_match:
                            col = v['column']
                            val = v['value']
                            select_parts = [
                                "COUNT(*) as total",
                                f"SUM(CASE WHEN {col} = '{val}' THEN 1 ELSE 0 END) as {val.lower()}_count",
                                f"ROUND(100.0 * SUM(CASE WHEN {col} = '{val}' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as pct"
                            ]
                            break

                # Strategy 2: Search for question keywords in actual DB distinct values
                if not select_parts:
                    # Extract meaningful keywords from question (exclude common stop words)
                    q_words = set(re.findall(r'\b[a-z]{3,}\b', q.lower()))
                    stop_words = {'the', 'and', 'are', 'what', 'which', 'how', 'for', 'with', 'from', 'have'}
                    q_words -= stop_words

                    # Common abbreviation expansions: question word → DB value abbreviations
                    # Enables "female" to match 'F', "male" to match 'M', etc.
                    _abbrev_map = {
                        'female': 'f', 'male': 'm',
                        'yes': 'y', 'no': 'n',
                        'active': 'a', 'inactive': 'i',
                    }
                    q_abbrevs = {}  # abbreviation → full word
                    for w in q_words:
                        if w in _abbrev_map:
                            q_abbrevs[_abbrev_map[w]] = w

                    if q_words:
                        try:
                            conn = sqlite3.connect(self.learner.db_path)
                            for table in tables:
                                for p in self.learner.tables.get(table, []):
                                    if not p.is_categorical:
                                        continue
                                    # Search all categorical columns, not just status/type
                                    cursor = conn.execute(
                                        f"SELECT DISTINCT {p.name} FROM {table} WHERE {p.name} IS NOT NULL LIMIT 30"
                                    )
                                    for row in cursor:
                                        if row[0]:
                                            val_lower = row[0].lower()
                                            # Check substring match for multi-char values
                                            is_match = any(word in val_lower for word in q_words)
                                            # Also check abbreviation match for single-char values
                                            if not is_match and len(val_lower) <= 2 and val_lower in q_abbrevs:
                                                is_match = True
                                            if is_match:
                                                val = row[0]
                                                select_parts = [
                                                    "COUNT(*) as total",
                                                    f"SUM(CASE WHEN {p.name} = '{val}' THEN 1 ELSE 0 END) as {val.lower()}_count",
                                                    f"ROUND(100.0 * SUM(CASE WHEN {p.name} = '{val}' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as pct"
                                                ]
                                                break
                                    if select_parts:
                                        break
                                if select_parts:
                                    break
                            conn.close()
                        except Exception:
                            pass

                # Strategy 3: Authorization-related percentage
                if not select_parts and ('authorization' in q or 'auth' in q or 'prior auth' in q):
                    # Check referrals for authorization
                    if 'referrals' in self.learner.tables:
                        for p in self.learner.tables.get('referrals', []):
                            if 'authorization' in p.name.lower():
                                primary = tables[0] if tables else 'claims'
                                select_parts = [
                                    "COUNT(*) as total",
                                    f"SUM(CASE WHEN {p.name} != '' AND {p.name} IS NOT NULL THEN 1 ELSE 0 END) as with_auth",
                                    f"ROUND(100.0 * SUM(CASE WHEN {p.name} != '' AND {p.name} IS NOT NULL THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as auth_pct"
                                ]
                                # Override tables to use referrals
                                tables.clear()
                                tables.append('referrals')
                                break

                if not select_parts:
                    select_parts = ["COUNT(*) as total_count"]

        elif intent_type == 'trend':
            date_col = self._find_date_column(q, columns, tables)
            if date_col:
                bucket = f"SUBSTR({date_col}, 1, 7)"
                if 'quarter' in q:
                    bucket = (f"SUBSTR({date_col}, 1, 4) || '-Q' || "
                              f"((CAST(SUBSTR({date_col}, 6, 2) AS INTEGER) - 1) / 3 + 1)")
                num_col = self._find_numeric_column(q, columns, tables)
                if num_col:
                    agg_func = self._detect_agg_func(q)
                    select_parts = [
                        f"{bucket} as period",
                        f"ROUND({agg_func}(CAST({num_col} AS REAL)), 2) as {agg_func.lower()}_{num_col.lower()}",
                        "COUNT(*) as record_count"
                    ]
                else:
                    select_parts = [f"{bucket} as period", "COUNT(*) as count"]
                group_cols = ['period']
                needs_group = True
            else:
                select_parts = self._default_select(columns, tables)

        elif intent_type == 'rate':
            # Rate queries: denial rate, fill rate, no-show rate, etc.
            rate_info = self._detect_rate_info(q, columns, tables)
            if rate_info:
                status_col, target_val, group_col = rate_info

                # Check if this is a time-based rate query ("trend", "over time", "monthly", "last N months")
                time_patterns = [
                    r'\btrend\b', r'\bover\s+time\b', r'\bmonthly\b', r'\bquarterly\b',
                    r'\byearly\b', r'\blast\s+\d+\s+months?\b', r'\bby\s+month\b',
                    r'\bover\s+the\s+(?:last|past)\b',
                ]
                is_time_rate = any(re.search(p, q) for p in time_patterns)
                if is_time_rate and not group_col:
                    # Add time-based grouping
                    date_col = self._find_date_column(q, columns, tables)
                    if date_col:
                        bucket = f"SUBSTR({date_col}, 1, 7)"
                        if 'quarter' in q:
                            bucket = (f"SUBSTR({date_col}, 1, 4) || '-Q' || "
                                      f"((CAST(SUBSTR({date_col}, 6, 2) AS INTEGER) - 1) / 3 + 1)")
                        elif 'year' in q:
                            bucket = f"SUBSTR({date_col}, 1, 4)"
                        select_parts = [
                            f"{bucket} as period",
                            "COUNT(*) as total",
                            f"SUM(CASE WHEN {status_col} = '{target_val}' THEN 1 ELSE 0 END) as {target_val.lower()}_count",
                            f"ROUND(100.0 * SUM(CASE WHEN {status_col} = '{target_val}' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as rate"
                        ]
                        group_cols = ['period']
                        needs_group = True
                    else:
                        # No date column found — fall through to basic rate
                        group_select = ""
                        select_parts = [
                            f"COUNT(*) as total",
                            f"SUM(CASE WHEN {status_col} = '{target_val}' THEN 1 ELSE 0 END) as {target_val.lower()}_count",
                            f"ROUND(100.0 * SUM(CASE WHEN {status_col} = '{target_val}' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as rate"
                        ]
                else:
                    group_select = f"{group_col}, " if group_col else ""
                    select_parts = [
                        f"{group_select}COUNT(*) as total",
                        f"SUM(CASE WHEN {status_col} = '{target_val}' THEN 1 ELSE 0 END) as {target_val.lower()}_count",
                        f"ROUND(100.0 * SUM(CASE WHEN {status_col} = '{target_val}' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as rate"
                    ]
                    if group_col:
                        group_cols = [group_col]
                    needs_group = bool(group_col)
            else:
                # Fallback for rate intent: find a status column and show breakdown
                # rather than returning raw rows
                for table in tables:
                    for p in self.learner.tables.get(table, []):
                        if 'status' in p.name.lower() and p.is_categorical:
                            select_parts = [
                                p.name,
                                "COUNT(*) as count",
                                f"ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) as pct"
                            ]
                            group_cols = [p.name]
                            needs_group = True
                            break
                    if select_parts:
                        break
                if not select_parts:
                    select_parts = ["COUNT(*) as total_count"]

        elif intent_type == 'comparison':
            # Try to find a good categorical column to compare on
            group_col = self._resolve_group_column(group_term, columns, tables) if group_term else None
            num_col = self._find_numeric_column(q, columns, tables)
            if group_col and num_col:
                agg_func = self._detect_agg_func(q)
                select_parts = [
                    group_col,
                    f"COUNT(*) as count",
                    f"ROUND({agg_func}(CAST({num_col} AS REAL)), 2) as {agg_func.lower()}_{num_col.lower()}"
                ]
                group_cols = [group_col]
                needs_group = True
            elif group_col:
                select_parts = [group_col, "COUNT(*) as count"]
                group_cols = [group_col]
                needs_group = True
            else:
                select_parts = self._default_select(columns, tables)

        elif intent_type == 'filter':
            select_parts = self._default_select(columns, tables)

        else:  # 'lookup' or unknown
            select_parts = self._default_select(columns, tables)

        return select_parts, group_cols, needs_group

    def _build_conditions(self, q: str, values: List[Dict], columns: List[Dict],
                          tables: List[str], intent_type: str = 'lookup') -> List[str]:
        """Build WHERE conditions from semantic value matching and question parsing.
        Uses learned schema profiles to validate conditions against actual data."""
        conditions = []
        used_cols = set()

        # From semantic value matching — high-confidence value matches become filters
        # EXCEPTION: for percentage/rate intents, don't filter on the target value
        # (e.g. "what percentage are cancelled" needs ALL rows, not just cancelled ones)
        skip_filter_for_rate = intent_type in ('percentage', 'rate')

        for v in values:
            if v['score'] >= 0.2 and v['table'] in tables:
                col = v['column']
                val = v['value']
                if col not in used_cols:
                    val_text = val.lower().replace('_', ' ')
                    if val_text in q or val.lower() in q:
                        # Don't add as filter if it appears to be a GROUP BY dimension
                        group_term = self._extract_group_term(q)
                        if group_term and col.lower().replace('_', ' ') in group_term:
                            continue
                        # Don't filter on the target value for percentage/rate queries
                        if skip_filter_for_rate:
                            continue
                        conditions.append(f"{col} = '{val}'")
                        used_cols.add(col)

        # Handle "older than N" / "younger than N" patterns directly
        older_m = re.search(r'\b(?:older|elder)\s+than\s+(\d+)', q)
        younger_m = re.search(r'\b(?:younger)\s+than\s+(\d+)', q)
        if older_m or younger_m:
            age = int((older_m or younger_m).group(1))
            op = '<=' if older_m else '>='
            # Find birth date column
            birth_col = self.learner.find_birth_date_column()
            if birth_col:
                conditions.append(f"{birth_col.name} {op} date('now', '-{age} years')")

        # Handle "age > N" pattern
        age_compare = re.search(r'\bage\s*(?:>|>=|over|above)\s*(\d+)', q)
        if age_compare:
            age = int(age_compare.group(1))
            birth_col = self.learner.find_birth_date_column()
            if birth_col:
                conditions.append(f"{birth_col.name} <= date('now', '-{age} years')")

        # Extract numeric comparisons from question
        COMPARISON_RE = [
            (r'(?:more than|over|greater than|above|exceeds?|>)\s*\$?([\d,]+\.?\d*)', '>'),
            (r'(?:less than|under|below|fewer than|<)\s*\$?([\d,]+\.?\d*)', '<'),
            (r'(?:at least|minimum|min)\s*\$?([\d,]+\.?\d*)', '>='),
            (r'(?:at most|maximum|max)\s*\$?([\d,]+\.?\d*)', '<='),
            (r'between\s*\$?([\d,]+\.?\d*)\s*and\s*\$?([\d,]+\.?\d*)', 'BETWEEN'),
        ]

        for pat, op in COMPARISON_RE:
            m = re.search(pat, q)
            if m:
                val = m.group(1).replace(',', '')
                try:
                    num_val = float(val)
                    # Year detection — don't treat as numeric filter
                    if 2020 <= num_val <= 2030:
                        continue
                except ValueError:
                    continue

                # Age context — data-driven birth date discovery
                age_words = ['member', 'patient', 'person', 'people', 'age', 'old', 'year',
                             'customer', 'user', 'employee', 'student']
                money_words = ['$', 'billed', 'paid', 'cost', 'amount', 'claim', 'score',
                               'price', 'revenue', 'sales', 'order', 'total']
                if 1 <= num_val <= 120 and any(w in q for w in age_words) and not any(w in q for w in money_words):
                    birth_col = self.learner.find_birth_date_column()
                    if birth_col:
                        age = int(num_val)
                        if op in ('>', '>='):
                            conditions.append(f"{birth_col.name} <= date('now', '-{age} years')")
                        elif op in ('<', '<='):
                            conditions.append(f"{birth_col.name} >= date('now', '-{age} years')")
                        continue

                # Numeric column filter
                num_col = self._find_numeric_column(q, columns, tables)
                if num_col:
                    if op == 'BETWEEN':
                        val2 = m.group(2).replace(',', '')
                        conditions.append(f"CAST({num_col} AS REAL) BETWEEN {val} AND {val2}")
                    else:
                        conditions.append(f"CAST({num_col} AS REAL) {op} {val}")

        # Year filter
        ym = re.search(r'(?:in|over|during|for|from|year)?\s*(20[12]\d)\b', q)
        if ym:
            yr = ym.group(1)
            date_col = self._find_date_column(q, columns, tables)
            if date_col:
                conditions.append(f"{date_col} LIKE '{yr}%'")

        # Time filters
        TIME_FILTERS = {
            'last 30 days': "date('now', '-30 days')",
            'last month': "date('now', '-1 month')",
            'last quarter': "date('now', '-3 months')",
            'last 3 months': "date('now', '-3 months')",
            'this year': "date('now', 'start of year')",
            'last week': "date('now', '-7 days')",
            'last 7 days': "date('now', '-7 days')",
            'last 6 months': "date('now', '-6 months')",
            'last 90 days': "date('now', '-90 days')",
        }
        for phrase, expr in TIME_FILTERS.items():
            if phrase in q:
                date_col = self._find_date_column(q, columns, tables)
                if date_col:
                    conditions.append(f"{date_col} >= {expr}")
                break

        # ── Healthcare domain concept conditions ──
        # Match multi-word and single-word healthcare business terms from the
        # dynamic engine's DOMAIN_CONCEPTS dictionary. These translate terms like
        # "prior authorization", "high-value claim", "expensive", "chronic", etc.
        # into actual WHERE clause conditions grounded in the schema.
        conditions = self._apply_domain_conditions(q, tables, conditions, used_cols)

        # ── Qualitative adjective thresholds ──
        # "high dollar", "expensive claims", "costly encounters" → data-driven WHERE
        conditions = self._apply_qualitative_thresholds(q, tables, conditions)

        return conditions

    # ── Qualitative adjective patterns ──
    # Maps adjective + domain hint to (column_name, operator, value)
    _QUALITATIVE_PATTERNS = [
        # "high dollar/cost/amount", "expensive", "costly"
        (r'\b(?:high|large|big)\s+(?:dollar|cost|amount|value|paid|billed)',
         {'claims': ('CAST(PAID_AMOUNT AS REAL)', '>', 1000),
          'prescriptions': ('CAST(COST AS REAL)', '>', 100)}),
        (r'\b(?:expensive|costly|pricey)\b',
         {'claims': ('CAST(PAID_AMOUNT AS REAL)', '>', 1000),
          'prescriptions': ('CAST(COST AS REAL)', '>', 100),
          'encounters': ('CAST(LENGTH_OF_STAY AS INTEGER)', '>', 5)}),
        # "low dollar/cost"
        (r'\b(?:low|small|cheap)\s+(?:dollar|cost|amount|value|paid|billed)',
         {'claims': ('CAST(PAID_AMOUNT AS REAL)', '<', 500),
          'prescriptions': ('CAST(COST AS REAL)', '<', 20)}),
        # "long stay", "extended stay"
        (r'\b(?:long|extended|prolonged)\s+(?:stay|los|length)',
         {'encounters': ('CAST(LENGTH_OF_STAY AS INTEGER)', '>', 7)}),
        # "short stay"
        (r'\b(?:short|brief)\s+(?:stay|los|length)',
         {'encounters': ('CAST(LENGTH_OF_STAY AS INTEGER)', '<', 2)}),
        # "high risk"
        (r'\bhigh\s+risk\b',
         {'members': ('CAST(RISK_SCORE AS REAL)', '>=', 3.5)}),
        # "low risk"
        (r'\blow\s+risk\b',
         {'members': ('CAST(RISK_SCORE AS REAL)', '<', 2.0)}),
    ]

    def _apply_domain_conditions(self, q: str, tables: List[str],
                                  conditions: List[str],
                                  used_cols: set) -> List[str]:
        """Apply healthcare domain concept conditions from dynamic engine vocabulary.

        Matches multi-word phrases first (more specific), then single words.
        Only applies conditions for tables that are in the query's table list.
        """
        try:
            from dynamic_sql_engine import DOMAIN_CONCEPTS
        except ImportError:
            return conditions

        applied = set()  # Track which concepts were applied

        # Sort by phrase length descending so multi-word matches take priority
        sorted_concepts = sorted(DOMAIN_CONCEPTS.items(),
                                  key=lambda x: len(x[0]), reverse=True)

        for term, concept in sorted_concepts:
            # Check if the term appears in the question
            if term.lower() not in q:
                continue
            # Check that the concept's target table is in our query tables
            concept_tables = concept.get('tables', [])
            matching_tables = [t for t in concept_tables if t in tables]
            if not matching_tables:
                continue
            # Don't apply if we already applied a more specific concept
            # for the same column
            new_conds = concept.get('conds', [])
            if not new_conds:
                continue

            skip = False
            for cond in new_conds:
                # Extract column name from condition
                col_m = re.match(r"[\w.]*?(\w+)\s*(?:=|>|<|LIKE|IN|IS|BETWEEN|NOT)", cond)
                if col_m:
                    col = col_m.group(1)
                    if col in applied:
                        skip = True
                        break
                    applied.add(col)

            if not skip:
                for cond in new_conds:
                    if cond not in conditions:
                        conditions.append(cond)
                logger.info("Domain concept '%s' → conditions: %s", term, new_conds)

        return conditions

    def _apply_qualitative_thresholds(self, q: str, tables: List[str],
                                       conditions: List[str]) -> List[str]:
        """Detect qualitative adjectives ('high dollar', 'expensive', 'costly')
        and translate them to data-driven threshold filters.

        Uses adaptive thresholds based on actual data percentiles when possible,
        with sensible defaults as fallback.
        """
        for pattern, table_thresholds in self._QUALITATIVE_PATTERNS:
            if not re.search(pattern, q):
                continue
            for tbl in tables:
                if tbl not in table_thresholds:
                    continue
                col_expr, op, default_val = table_thresholds[tbl]
                # Try to get data-driven threshold (P75 for "high", P25 for "low")
                threshold = default_val
                try:
                    # Extract raw column name for percentile query
                    col_name = re.search(r'(\w+)\s*(?:AS|$)', col_expr)
                    if col_name:
                        raw_col = col_name.group(1)
                        conn = sqlite3.connect(self.learner.db_path)
                        # Use approximate percentile via NTILE
                        if op in ('>', '>='):
                            # P75 for "high"
                            row = conn.execute(
                                f"SELECT {col_expr} FROM {tbl} WHERE {raw_col} IS NOT NULL "
                                f"AND {raw_col} != '' ORDER BY {col_expr} DESC "
                                f"LIMIT 1 OFFSET (SELECT COUNT(*)/4 FROM {tbl} WHERE {raw_col} IS NOT NULL AND {raw_col} != '')"
                            ).fetchone()
                        else:
                            # P25 for "low"
                            row = conn.execute(
                                f"SELECT {col_expr} FROM {tbl} WHERE {raw_col} IS NOT NULL "
                                f"AND {raw_col} != '' ORDER BY {col_expr} ASC "
                                f"LIMIT 1 OFFSET (SELECT COUNT(*)/4 FROM {tbl} WHERE {raw_col} IS NOT NULL AND {raw_col} != '')"
                            ).fetchone()
                        conn.close()
                        if row and row[0] is not None:
                            threshold = round(float(row[0]), 2)
                except Exception:
                    pass  # Fall back to default

                cond = f"{col_expr} {op} {threshold}"
                if cond not in conditions:
                    conditions.append(cond)
                    logger.info("Qualitative threshold: '%s' → %s (table=%s, threshold=%s)",
                                pattern, cond, tbl, threshold)
                break  # Only apply to first matching table
        return conditions

    def _build_from(self, tables: List[str], columns: List[Dict],
                    conditions: List[str]) -> Tuple[str, List[str]]:
        """Build FROM clause with JOINs using knowledge graph."""
        if len(tables) <= 1:
            return tables[0] if tables else list(self.learner.tables.keys())[0], []

        primary = tables[0]
        join_tables = []
        join_clauses = []

        # Use knowledge graph for join path finding
        joins = self.kg.find_join_path(tables)

        aliases = {primary: primary[0]}
        parts = [f"{primary} {primary[0]}"]

        for t1, t2, join_col in joins:
            alias = t2[0]
            # Handle alias conflicts
            if alias in aliases.values():
                alias = t2[:2]
            aliases[t2] = alias

            if '=' in join_col:
                left, right = join_col.split('=', 1)
                on_clause = f"{aliases.get(t1, t1[0])}.{left} = {alias}.{right}"
            else:
                on_clause = f"{aliases.get(t1, t1[0])}.{join_col} = {alias}.{join_col}"

            join_clauses.append(f"JOIN {t2} {alias} ON {on_clause}")
            join_tables.append(t2)

        return parts[0] + ' ' + ' '.join(join_clauses), join_tables

    def _build_order(self, q: str, intent_type: str, select_parts: List[str],
                     group_cols: List[str]) -> str:
        """Build ORDER BY clause."""
        # Detect direction
        asc = any(w in q for w in ['lowest', 'smallest', 'least', 'cheapest',
                                    'fewest', 'bottom', 'ascending'])
        direction = 'ASC' if asc else 'DESC'

        if intent_type in ('ranking', 'breakdown', 'percentage'):
            # Order by the aggregate column
            if len(select_parts) > 1:
                # Last column is usually the metric
                alias = select_parts[-1].split(' as ')[-1].strip() if ' as ' in select_parts[-1].lower() else None
                if alias:
                    return f" ORDER BY {alias} {direction}"
                return f" ORDER BY 2 {direction}"

        if intent_type == 'trend':
            return " ORDER BY period"

        if intent_type == 'count' and group_cols:
            return f" ORDER BY count {direction}"

        return ""

    def _build_limit(self, q: str, intent_type: str, needs_group: bool) -> str:
        """Build LIMIT clause."""
        # Extract explicit limit
        top_m = re.search(r'\btop\s+(\d+)\b', q)
        if top_m:
            return f" LIMIT {top_m.group(1)}"

        limit_m = re.search(r'\blimit\s+(\d+)\b', q)
        if limit_m:
            return f" LIMIT {limit_m.group(1)}"

        if intent_type == 'ranking':
            return " LIMIT 10"
        elif intent_type == 'trend':
            return " LIMIT 50"
        elif needs_group:
            return " LIMIT 30"
        elif intent_type == 'lookup':
            return " LIMIT 50"
        return ""

    # UNIVERSAL QUERY INTELLIGENCE
    # One system that handles ANY query by discovering dimensions, metrics,
    # and computation patterns from the actual schema. No per-query-type
    # functions — just semantic analysis + universal SQL composition.

    def _discover_disambiguated_entity(self, q: str, fact: str) -> Optional[Dict]:
        """Detect when the question's SUBJECT is a disambiguated term.

        Catches patterns that _extract_group_term misses because they lack
        'by'/'per'/'each' keywords:
          - "which provider has the highest cost?"
          - "what provider has the most claims?"
          - "top 10 providers"
          - "compare providers"
          - "provider with highest denial rate"
          - "show me providers and their costs"
          - "breakdown of providers"

        Returns a synthesized dimension dict compatible with _sql_universal,
        or None if no disambiguation applies.
        """
        if not self._domain_disambiguations:
            return None

        q_lower = q.lower()
        q_words = set(re.findall(r'[a-z]+', q_lower))
        # Build stemmed set for plural tolerance: "providers" → "provider"
        q_stems = self._stem_words(q_words)

        # Check if any disambiguation trigger word is in the question
        matched_trigger = None
        for key, dis in self._domain_disambiguations.items():
            for trigger in dis['triggers']:
                if trigger in q_stems:
                    matched_trigger = trigger
                    break
            if matched_trigger:
                break

        if not matched_trigger:
            return None

        # Guard: do NOT fire if the question mentions practitioner-specific terms
        practitioner_terms = self.PRACTITIONER_TERMS
        if q_words & practitioner_terms:
            return None

        # Guard: do NOT fire if the question asks "which providers have highest X"
        # In this pattern, "provider" refers to healthcare practitioners, not insurance plans
        provider_ranking_patterns = [
            r'\bwhich\s+provider\w*\s+(?:has|have)\s+(?:the\s+)?(?:high|most|low|least|best|worst|great)',
            r'\btop\s+\d*\s*provider\w*\b',
            r'\bprovider\w*\s+(?:with|having)\s+(?:the\s+)?(?:high|most|low|least)',
            r'\bprovider\w*\s+(?:volume|count|caseload|workload|productivity)',
            r'\bhigh\w*\s+volume\s+provider\w*',
            r'\bprovider\w*\s+rank',
        ]
        if matched_trigger in ('provider', 'providers'):
            if any(re.search(pat, q_lower) for pat in provider_ranking_patterns):
                return None

        # Check: is the trigger word used as the ENTITY/SUBJECT of the question?
        # (not just as a filter value like "for provider X")
        entity_patterns = [
            # "which provider has...", "what provider has..."
            rf'\b(?:which|what)\s+{matched_trigger}\w*\s+(?:has|have|is|are|had|show|get)',
            # "top N providers", "top providers"
            rf'\btop\s+\d*\s*{matched_trigger}\w*',
            # "compare providers", "compare providers by X"
            rf'\bcompare\s+{matched_trigger}\w*',
            rf'\b{matched_trigger}\w*\s+comparison',
            # "provider with highest/lowest/most..."
            rf'\b{matched_trigger}\w*\s+(?:with|having)\s+(?:the\s+)?(?:high|low|most|least|best|worst)',
            # "breakdown of providers", "distribution of providers"
            rf'\b(?:breakdown|distribution|split|summary)\s+(?:of|for|across)\s+{matched_trigger}\w*',
            # "show me providers", "list providers", "providers and their"
            rf'\b(?:show|list|display|get)\s+(?:me\s+)?(?:all\s+)?{matched_trigger}\w*',
            rf'\b{matched_trigger}\w*\s+and\s+their\b',
            # "provider cost", "provider denial rate" (entity as adjective)
            rf'\b{matched_trigger}\w*\s+(?:cost|amount|rate|count|total|average|denial|claim)',
            # Just "providers" as the subject in a ranking/aggregation context
            rf'\b{matched_trigger}\w*\s+(?:ranked|sorted|ordered)',
            # "[metric] by provider" — standard dimension pattern
            rf'\bby\s+{matched_trigger}\w*',
            # "denial rate for each provider", "cost per provider"
            rf'\b(?:per|each)\s+{matched_trigger}\w*',
        ]

        is_entity = any(re.search(pat, q_lower) for pat in entity_patterns)
        if not is_entity:
            return None

        # The trigger word IS the entity — synthesize a dimension from disambiguation
        return self._disambiguate_dimension_term(matched_trigger, fact)

    def _build_disambiguated_dimension(self, disambig: Dict, fact: str) -> Optional[Dict]:
        """Convert a disambiguation result into a full dimension dict for _sql_universal."""
        dis_table = disambig['table']
        dis_col = disambig['column']
        needs_join = (dis_table != fact)
        join_col = None
        dim_join_col = None

        if needs_join:
            fact_cols = {p.name for p in self.learner.tables.get(fact, [])}
            dis_cols = {p.name for p in self.learner.tables.get(dis_table, [])}
            shared_ids = sorted(c for c in fact_cols & dis_cols if c.endswith('_ID'))
            if shared_ids:
                join_col = shared_ids[0]
                dim_join_col = shared_ids[0]
            else:
                jc = self.kg._get_join_condition(fact, dis_table)
                if jc and '=' in jc:
                    parts = jc.split('=', 1)
                    join_col = parts[0].strip()
                    dim_join_col = parts[1].strip()
                elif jc:
                    join_col = jc
                    dim_join_col = jc
                else:
                    return None  # Can't join

        return {
            'table': dis_table, 'column': dis_col,
            'needs_join': needs_join, 'join_col': join_col,
            'dim_join_col': dim_join_col,
            'is_age_group': False,
        }

    def _detect_special_pattern(self, q: str, intent_type: str,
                                 columns: List[Dict], tables: List[Dict],
                                 values: List[Dict]) -> Optional[Dict[str, Any]]:
        """Universal query planner. Discovers what the question needs from
        semantic signals, then resolves all schema elements from data."""
        signals = self._extract_signals(q)
        fact = self._identify_fact_table(q, tables)

        # ── Cross-table cost redirect ──────────────────────────────────
        # "cost per encounter" → route to claims (which has cost columns),
        # not encounters (which only has LENGTH_OF_STAY).
        # Same for "cost per visit", "spend per encounter", etc.
        q_lower = q.lower()
        cost_words = {'cost', 'spend', 'spending', 'expense', 'charge', 'paid', 'billed', 'amount', 'dollar', 'price'}
        encounter_words = {'encounter', 'visit', 'admission'}
        q_words_set = set(re.findall(r'[a-z]+', q_lower))
        if fact == 'encounters' and (q_words_set & cost_words) and (q_words_set & encounter_words):
            # The question mentions cost AND encounter — cost data lives in claims
            if 'claims' in {t.get('table') for t in tables} | set(self.learner.tables.keys()):
                fact = 'claims'
                logger.info("Cross-table redirect: cost+encounter → claims (cost data lives in claims)")

        # ── Provider ranking redirect ─────────────────────────────────
        # "which providers have the highest volume" → individual provider ranking
        # Routes to claims and groups by RENDERING_NPI (individual provider)
        provider_ranking_re = re.search(
            r'\b(?:which|what|top\s*\d*)\s+provider\w*\s+'
            r'(?:has|have|with|having)?\s*(?:the\s+)?(?:high|most|low|least|great|big|larg)',
            q_lower
        )
        if not provider_ranking_re:
            provider_ranking_re = re.search(
                r'\b(?:provider\w*\s+(?:volume|count|caseload|rank|productivity))',
                q_lower
            )
        if provider_ranking_re:
            # Build a provider ranking query directly
            metric_col = 'PAID_AMOUNT' if (q_words_set & cost_words) else None
            count_based = 'volume' in q_lower or 'count' in q_lower or 'most' in q_lower or 'highest' in q_lower
            npi_col = 'RENDERING_NPI'
            claim_tbl = 'claims' if 'claims' in self.learner.tables else fact
            if count_based and not metric_col:
                # Provider volume = count of claims per provider
                sql = (f"SELECT {npi_col}, COUNT(*) as claim_count "
                       f"FROM {claim_tbl} "
                       f"GROUP BY {npi_col} "
                       f"ORDER BY claim_count DESC LIMIT 20;")
                logger.info("Provider ranking: volume-based, grouping by %s in %s", npi_col, claim_tbl)
                return {'sql': sql, 'tables_used': [claim_tbl],
                        'confidence': 0.85, 'intent': 'ranking'}
            elif metric_col:
                sql = (f"SELECT {npi_col}, COUNT(*) as claim_count, "
                       f"ROUND(SUM(CAST({metric_col} AS REAL)), 2) as total_{metric_col.lower()} "
                       f"FROM {claim_tbl} "
                       f"GROUP BY {npi_col} "
                       f"ORDER BY total_{metric_col.lower()} DESC LIMIT 20;")
                logger.info("Provider ranking: cost-based, grouping by %s in %s", npi_col, claim_tbl)
                return {'sql': sql, 'tables_used': [claim_tbl],
                        'confidence': 0.85, 'intent': 'ranking'}

        # Route to SQL pattern (most specific first — order matters)

        # ARCHITECTURAL LAYER 2: Derived concepts (utilization, ALOS, etc.)
        # These are computed metrics not present as columns — discovered from schema.
        derived = self._detect_derived_concept(q)
        if derived:
            concept_key, concept_def = derived
            return self._sql_derived_concept(q, fact, concept_key, concept_def)

        # ARCHITECTURAL LAYER 1: Population filters ("metric FOR population")
        # General mechanism: detects any "avg X for Y population" pattern.
        pop_filter = self._detect_population_filter(q)
        if pop_filter:
            return self._sql_population_filtered_metric(q, fact, pop_filter)

        if signals['readmission']:
            # Check for cross-metric: readmission + another rate (e.g., denial)
            cross_rate = self._detect_cross_metric_rate(q)
            if cross_rate:
                return self._sql_cross_metric_readmission(q, fact, cross_rate)
            return self._sql_readmission(q, fact)
        if signals['threshold'] is not None:
            return self._sql_threshold(q, fact, signals)
        if signals['intersection']:
            return self._sql_intersection(q, fact, *signals['intersection'])
        if signals['ratio']:
            return self._sql_ratio(q, fact, *signals['ratio'])
        if signals['ranking_comparison']:
            return self._sql_ranking_comparison(q, fact, signals['ranking_comparison'], columns)
        if signals['comparison']:
            return self._sql_comparison(q, fact, *signals['comparison'], columns)
        if signals['date_arithmetic']:
            return self._sql_date_arithmetic(q, fact)
        if signals['most_occurring']:
            return self._sql_most_occurring(q, fact, signals['most_occurring'], columns)

        # ARCHITECTURAL LAYER 4b: Disambiguated entity as dimension
        # Catches "which provider has highest cost?", "top 10 providers", etc.
        # where the disambiguated term IS the entity being grouped/ranked,
        # even without explicit "by" keyword.
        disambig_entity = self._discover_disambiguated_entity(q, fact)
        if disambig_entity:
            dim = self._build_disambiguated_dimension(disambig_entity, fact)
            if dim:
                logger.info("Layer 4b: disambiguated entity '%s' → %s.%s as dimension",
                            disambig_entity.get('column'), disambig_entity.get('table'),
                            disambig_entity.get('column'))
                return self._sql_universal(q, fact, dim, signals, columns, values)

        # Universal: any dimension + metric + filter + percentage
        dim = self._discover_dimension(q, fact)
        if dim or signals['percentage'] or signals['age_filter'] or signals['birth_year_filter']:
            return self._sql_universal(q, fact, dim, signals, columns, values)

        # Standalone rate query: "what is the denial rate?" (no dimension, no signals)
        # Detects rate info and produces an overall rate.
        # GUARD: Do NOT fire for trend/temporal queries — let the standard compose()
        # handle those with GROUP BY month/year.
        is_trend_query = any(w in q for w in ['trend', 'over time', 'monthly', 'by month',
                                               'by year', 'quarterly', 'weekly', 'last 12',
                                               'over the last', 'year over year', 'yoy'])
        if is_trend_query:
            return None

        # Search tables ordered by: (1) largest fact tables first (claims > encounters > others)
        # This ensures "denial rate" → claims.CLAIM_STATUS, not referrals.STATUS
        rate_search_tables = sorted(
            self.learner.tables.keys(),
            key=lambda t: self.learner.table_row_counts.get(t, 0),
            reverse=True
        )
        rate_info = self._detect_rate_info(q, columns, rate_search_tables)
        if rate_info:
            status_col, target_val, _ = rate_info
            # Find which table actually has this status_col with this value
            rate_table = None
            for tbl_name in rate_search_tables:
                for p in self.learner.tables.get(tbl_name, []):
                    if p.name == status_col and p.is_categorical:
                        # Verify: does this table actually have the target value?
                        try:
                            import sqlite3 as _sq3
                            _conn = _sq3.connect(self.learner.db_path)
                            _has = _conn.execute(
                                f"SELECT 1 FROM {tbl_name} WHERE {status_col} = ? LIMIT 1",
                                (target_val,)
                            ).fetchone()
                            _conn.close()
                            if _has:
                                rate_table = tbl_name
                                break
                        except Exception:
                            pass
                if rate_table:
                    break
            if not rate_table:
                rate_table = fact

            safe_val = re.sub(r'[^a-zA-Z0-9_]', '_', target_val.lower())
            sql = (
                f"SELECT COUNT(*) as total, "
                f"SUM(CASE WHEN {status_col} = '{target_val}' THEN 1 ELSE 0 END) as {safe_val}_count, "
                f"ROUND(100.0 * SUM(CASE WHEN {status_col} = '{target_val}' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as {safe_val}_rate "
                f"FROM {rate_table};"
            )
            return {
                'sql': sql, 'tables_used': [rate_table],
                'confidence': 0.88, 'intent': 'rate',
                'explanation': f'Overall {target_val.lower()} rate from {rate_table}',
            }

        return None

    def _resolve_category_value(self, term: str, tables: List[str]) -> Optional[Tuple[str, str, str]]:
        """Resolve natural language category term to (table, column, actual_value).
        Searches all categorical columns via sample_values (fast), then DISTINCT queries."""
        if not term:
            return None
        term_lower = term.lower()
        def _fuzzy_match(val_str, target):
            vl = val_str.lower().replace('_', ' ')
            return target in vl or vl.startswith(target)
        for table in tables:
            for p in self.learner.tables.get(table, []):
                if not p.is_categorical:
                    continue
                if p.sample_values:
                    for val in p.sample_values:
                        if _fuzzy_match(val, term_lower):
                            return (table, p.name, val)
        try:
            conn = sqlite3.connect(self.learner.db_path) if hasattr(self.learner, 'db_path') else None
            if conn:
                for table in tables:
                    for p in self.learner.tables.get(table, []):
                        if not p.is_categorical:
                            continue
                        try:
                            cursor = conn.execute(f"SELECT DISTINCT {p.name} FROM {table} WHERE {p.name} IS NOT NULL LIMIT 50")
                            for row in cursor.fetchall():
                                val = row[0]
                                if val and _fuzzy_match(val, term_lower):
                                    conn.close()
                                    return (table, p.name, val)
                        except Exception:
                            pass
                conn.close()
        except Exception:
            pass
        return None

    def _extract_signals(self, q: str) -> Dict[str, Any]:
        """One-pass extraction of ALL semantic signals from the question.
        Returns a dictionary of detected patterns so the planner can route."""
        signals = {
            'threshold': None,
            'ratio': None,
            'comparison': None,
            'ranking_comparison': None,
            'intersection': None,
            'date_arithmetic': False,
            'readmission': False,
            'percentage': False,
            'age_filter': None,
            'birth_year_filter': None,
            'most_occurring': None,
        }

        # Threshold: "more than N visits/claims/encounters" or "above N%"
        for pat in [
            r'(?:more\s+than|over|above|exceeds?|>=?)\s*(\d+)\s*(?:times|visits|admissions|encounters|claims|appointments)',
            r'(?:more\s+than|over|above|exceeds?|>=?)\s*(\d+)\s*%',
            r'visited\s+\w+\s+(?:more\s+than|over)\s*(\d+)',
            r'had\s+(?:more\s+than|over|above)\s*(\d+)',
            r'at\s+least\s+(\d+)\s*(?:times|visits|admissions|encounters|claims)',
        ]:
            m = re.search(pat, q)
            if m:
                signals['threshold'] = int(m.group(1))
                break

        # Ratio: "X to Y ratio"
        for pat in [
            r'(\w+)\s+to\s+(\w+)\s+ratio',
            r'ratio\s+of\s+(\w+)\s+(?:to|vs|and)\s+(\w+)',
            r'(\w+)\s+(?:vs|versus)\s+(\w+)\s+ratio',
        ]:
            m = re.search(pat, q)
            if m:
                signals['ratio'] = (m.group(1).lower(), m.group(2).lower())
                break

        # Ranking comparison: "top N vs bottom N [entity]" or "compare top 5 and bottom 5 providers"
        for pat in [
            r'(?:compare\s+)?top\s+(\d+)\s+(?:vs|versus|and|&)\s+bottom\s+(\d+)\s+(\w+)',
            r'(?:compare\s+)?bottom\s+(\d+)\s+(?:vs|versus|and|&)\s+top\s+(\d+)\s+(\w+)',
            r'top\s+(\d+)\s+(?:and|vs|versus)\s+(?:the\s+)?(?:worst|bottom|lowest)\s+(\d+)\s+(\w+)',
            r'(?:best|highest|top)\s+(\d+)\s+(?:vs|versus|and|&)\s+(?:worst|lowest|bottom)\s+(\d+)\s+(\w+)',
        ]:
            m = re.search(pat, q)
            if m:
                n_top = int(m.group(1))
                n_bot = int(m.group(2))
                entity = m.group(3).lower()
                # For "bottom N vs top N" patterns, swap
                if 'bottom' in pat[:20]:
                    n_top, n_bot = n_bot, n_top
                signals['ranking_comparison'] = (n_top, n_bot, entity)
                break

        # Comparison: "compare X vs Y costs"
        for pat in [
            r'compare\s+(\w+)\s+(?:vs|versus|and|to)\s+(\w+)\s+(\w+)',
            r'(\w+)\s+(?:vs|versus)\s+(\w+)\s+(\w+)',
            r'compare\s+(\w+)\s+(?:vs|versus|and|to)\s+(\w+)',
        ]:
            m = re.search(pat, q)
            if m:
                cat_a = m.group(1).lower()
                cat_b = m.group(2).lower()
                metric = m.group(3).lower() if m.lastindex >= 3 else None
                # Skip if already captured as ranking comparison (top/bottom are not categories)
                if cat_a in ('top', 'bottom', 'best', 'worst', 'highest', 'lowest'):
                    break
                signals['comparison'] = (cat_a, cat_b, metric)
                break

        # Intersection: "members with both X and Y"
        for pat in [
            r'(?:members?|patients?|people)\s+(?:with|who\s+had|who\s+have)\s+both\s+(\w+)\s+and\s+(\w+)',
            r'(?:members?|patients?)\s+(?:with|having)\s+(\w+)\s+and\s+(\w+)\s+visits?',
        ]:
            m = re.search(pat, q)
            if m:
                signals['intersection'] = (m.group(1).lower(), m.group(2).lower())
                break

        # Date arithmetic: processing time, turnaround, duration
        time_patterns = [
            r'(?:processing|turnaround|adjudication|resolution)\s+time',
            r'time\s+(?:to|for)\s+(?:process|adjudicate|resolve)',
            r'how\s+long\s+(?:to|does\s+it\s+take)',
            r'days?\s+(?:to|between|from)',
        ]
        if any(re.search(pat, q) for pat in time_patterns):
            signals['date_arithmetic'] = True

        # Readmission
        if any(w in q for w in ['readmission', 'readmit', 're-admission']):
            signals['readmission'] = True

        # NOTE: Utilization is now handled by ARCHITECTURAL LAYER 2 (Derived Concepts)
        # in _detect_special_pattern() routing, not via extract_signals.

        # Percentage
        if any(w in q for w in ['%', 'percent', 'percentage', 'compared to total', 'of total']):
            signals['percentage'] = True

        # Age filter: "age >65"
        age_m = re.search(r'age\s*(?:>|>=|over|above)\s*(\d+)', q)
        if not age_m:
            age_m = re.search(r'(?:older|elder)\s+than\s+(\d+)', q)
        if age_m:
            signals['age_filter'] = int(age_m.group(1))

        # Birth year filter: "birth year < 1960"
        year_m = re.search(r'(?:birth|born|dob)\s*(?:year)?\s*(?:<|<=|before)\s*(1[89]\d{2}|20\d{2})', q)
        if year_m:
            signals['birth_year_filter'] = int(year_m.group(1))

        # Most occurring / common — but NOT "top N <entity>" which is a ranking
        m = re.search(r'(?:most\s+(?:occurring|common|frequent|popular)|top)\s+(\w+(?:\s+\w+)?)', q)
        if m:
            target = m.group(1).lower().strip()
            # Skip if "top" is followed by a number (e.g. "top 10 providers") — that's ranking
            if re.match(r'^\d+', target):
                pass  # Standard ranking, not "most occurring"
            elif target not in ('expensive', 'common', 'frequent'):
                signals['most_occurring'] = target

        return signals

    def _identify_fact_table(self, q: str, tables: List[Dict]) -> str:
        """Resolve the primary fact table using the pre-built concept index.

        Instead of ad-hoc word matching, we score every table by summing
        concept index hits for ALL words in the question. This catches
        semantic connections like "denial" → claims (because DENIED is a
        value in claims.CLAIM_STATUS) that pure table-name matching misses.
        """
        q_lower = q.lower()
        q_words = set(re.findall(r'[a-z]+', q_lower))
        # Remove ultra-common words that add noise
        q_words -= {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'do', 'does',
                     'did', 'has', 'have', 'had', 'be', 'been', 'being', 'this',
                     'that', 'these', 'those', 'it', 'its', 'what', 'which',
                     'who', 'how', 'when', 'where', 'why', 'can', 'will',
                     'should', 'would', 'could', 'may', 'might', 'shall',
                     'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with',
                     'from', 'up', 'out', 'if', 'or', 'and', 'not', 'no',
                     'so', 'but', 'all', 'any', 'each', 'than', 'me', 'my',
                     'we', 'our', 'they', 'them', 'their', 'also', 'show',
                     'give', 'tell', 'get', 'see', 'list', 'find', 'many'}

        # Strategy 1: Score tables via concept index
        table_scores = defaultdict(float)
        for word in q_words:
            if word in self._concept_table_index:
                for tbl, score in self._concept_table_index[word].items():
                    table_scores[tbl] += score

        # Small row-count bonus to prefer fact tables over dimension tables
        if hasattr(self.learner, 'table_row_counts') and self.learner.table_row_counts:
            for tbl in table_scores:
                row_count = self.learner.table_row_counts.get(tbl, 0)
                if row_count > 0:
                    table_scores[tbl] += min(row_count / self.ROW_COUNT_DIVISOR,
                                              self.ROW_BOOST_CAP)

        # ARCHITECTURAL LAYER 4: Domain disambiguation for fact table
        # If "provider" is in the question and the data has insurance plan types,
        # redirect from the practitioners table to the claims/members fact table
        # UNLESS the question explicitly asks about practitioners (doctor, NPI, etc.)
        practitioner_terms = self.PRACTITIONER_TERMS
        mentions_practitioner = bool(q_words & practitioner_terms)
        # Build stemmed set for plural tolerance
        q_stems = self._stem_words(q_words)
        if not mentions_practitioner and self._domain_disambiguations:
            for key, dis in self._domain_disambiguations.items():
                # If any disambiguation trigger word appears in the question,
                # boost the appropriate fact table
                for trigger in dis['triggers']:
                    if trigger in q_stems:
                        dis_table = dis['table']
                        table_scores[dis_table] += self.DISAMBIGUATION_FACT_BOOST
                        break

        # Rate query boost: "denial rate", "approval rate", etc.
        # Prefer tables where the rate can actually be computed from a status column
        rate_terms = {'denial', 'denied', 'deny', 'approval', 'approved', 'approve',
                      'rejection', 'rejected', 'reject'}
        is_rate_q = 'rate' in q_words and bool(q_words & rate_terms)
        if is_rate_q:
            rate_target = None
            if q_words & {'denial', 'denied', 'deny'}:
                rate_target = 'DENIED'
            elif q_words & {'approval', 'approved', 'approve'}:
                rate_target = 'APPROVED'
            elif q_words & {'rejection', 'rejected', 'reject'}:
                rate_target = 'REJECTED'
            if rate_target:
                for tbl_name, profiles in self.learner.tables.items():
                    for p in profiles:
                        if not p.is_categorical or 'status' not in p.name.lower():
                            continue
                        # Check if this status column has the target value
                        vals = set(str(v).upper() for v in (p.sample_values or []))
                        if rate_target in vals:
                            # Boost this table — it can compute the rate
                            row_count = self.learner.table_row_counts.get(tbl_name, 0)
                            boost = self.RATE_TABLE_BOOST + min(row_count / self.ROW_COUNT_DIVISOR, 5)
                            table_scores[tbl_name] += boost
                            logger.debug("Rate boost: %s.%s has %s → +%.1f",
                                        tbl_name, p.name, rate_target, boost)

        if table_scores:
            best_table = max(table_scores, key=table_scores.get)
            best_score = table_scores[best_table]
            if best_score >= self.MIN_CONCEPT_HITS:
                return best_table

        # Strategy 2: Use semantic table scores from TF-IDF matching
        if tables:
            best = tables[0]
            if isinstance(best, dict):
                return best['table']
            return best

        # Strategy 3: Largest table (fact tables tend to have most rows)
        if hasattr(self.learner, 'table_row_counts') and self.learner.table_row_counts:
            return max(self.learner.table_row_counts, key=self.learner.table_row_counts.get)
        return list(self.learner.tables.keys())[0]

    @staticmethod
    def _dim_on_clause(dim: Dict, fact_alias: str = 'f', dim_alias: str = 'd') -> str:
        """Build a SQL ON clause from a dimension dict.
        Handles both symmetric joins (MEMBER_ID=MEMBER_ID) and asymmetric
        joins (PROVIDER_NPI=NPI) discovered by the knowledge graph."""
        fact_col = dim.get('join_col', 'MEMBER_ID')
        dim_col = dim.get('dim_join_col') or fact_col
        return f"{fact_alias}.{fact_col} = {dim_alias}.{dim_col}"

    # ═══════════════════════════════════════════════════════════════════════
    # PRODUCTION HELPERS — reusable SQL fragments, schema resolution,
    # column identification. Each has a single well-defined purpose.
    # ═══════════════════════════════════════════════════════════════════════

    def _build_readmission_cte(self, alias: str = 'ip',
                                extra_columns: List[str] = None,
                                inpatient_val: str = None) -> Tuple[str, str]:
        """Build the standard 30-day readmission CTE.

        Produces a WITH clause that identifies inpatient encounters with a
        LAG window function to detect readmissions within READMISSION_WINDOW_DAYS.

        Args:
            alias: CTE alias name (default 'ip')
            extra_columns: Additional encounter columns to include
            inpatient_val: Override for INPATIENT value (auto-resolved if None)

        Returns:
            (cte_sql, readmission_case_expr)
            - cte_sql: "WITH ip AS (SELECT ...)" string
            - readmission_case_expr: CASE expression that counts readmissions
        """
        # Resolve inpatient value from data
        if not inpatient_val:
            inpatient_val = 'INPATIENT'
            res = self._resolve_category_value('inpatient',
                                               list(self.learner.tables.keys()))
            if res:
                _, _, inpatient_val = res

        # Discover encounter table columns
        enc_table = self._find_table_by_role('encounters')
        member_col = self._resolve_member_column(enc_table)
        admit_col, discharge_col = self._resolve_admission_columns(enc_table)
        visit_type_col = self._resolve_visit_type_column(enc_table)

        # Build column list
        columns = [member_col, admit_col, discharge_col]
        if extra_columns:
            columns.extend(c for c in extra_columns if c not in columns)
        cols_str = ', '.join(columns)

        window_days = self.READMISSION_WINDOW_DAYS

        cte = (
            f"WITH {alias} AS ("
            f"SELECT {cols_str}, "
            f"LAG({discharge_col}) OVER (PARTITION BY {member_col} "
            f"ORDER BY {admit_col}) as prev_discharge "
            f"FROM {enc_table} "
            f"WHERE {visit_type_col} = '{inpatient_val}' "
            f"AND {discharge_col} != '' AND {admit_col} != ''"
            f")"
        )

        readmit_case = (
            f"COUNT(DISTINCT CASE WHEN julianday({alias}.{admit_col}) "
            f"- julianday({alias}.prev_discharge) "
            f"BETWEEN 0 AND {window_days} THEN {alias}.{member_col} END)"
        )

        return cte, readmit_case

    def _find_table_by_role(self, role: str) -> str:
        """Find a table by its logical role (encounters, claims, members, etc.).

        Schema-agnostic: checks actual table names in the database, trying
        common synonyms before falling back to the largest matching table.

        Args:
            role: Logical role like 'encounters', 'claims', 'members'

        Returns:
            Actual table name string
        """
        synonyms = {
            'encounters': ('encounters', 'visits', 'admissions', 'episodes'),
            'claims': ('claims', 'billing', 'charges', 'transactions'),
            'members': ('members', 'patients', 'beneficiaries', 'enrollees'),
            'providers': ('providers', 'practitioners', 'physicians', 'clinicians'),
            'appointments': ('appointments', 'scheduling', 'bookings'),
        }
        candidates = synonyms.get(role, (role,))
        for name in candidates:
            if name in self.learner.tables:
                return name
        # Fallback: return the role name (best-guess)
        return role

    def _resolve_member_column(self, table: str) -> str:
        """Find the member/patient identity column in a table.

        Searches MEMBER_ID_CANDIDATES in priority order, falling back to
        the first non-primary-key ID column.

        Args:
            table: Table name to search

        Returns:
            Column name string (never None — returns best guess)
        """
        col_names = {p.name.upper() for p in self.learner.tables.get(table, [])}
        for candidate in self.MEMBER_ID_CANDIDATES:
            if candidate in col_names:
                # Return with original case from schema
                for p in self.learner.tables.get(table, []):
                    if p.name.upper() == candidate:
                        return p.name
        # Fallback: first _ID column that isn't the table's own PK
        tbl_stem = table.rstrip('s').upper()
        for p in self.learner.tables.get(table, []):
            if p.is_id and not p.name.upper().startswith(tbl_stem):
                return p.name
        return 'MEMBER_ID'

    def _resolve_admission_columns(self, table: str) -> Tuple[str, str]:
        """Find admission and discharge date columns in a table.

        Searches for columns with admit/discharge semantics.

        Returns:
            (admit_col, discharge_col) — defaults to ('ADMIT_DATE', 'DISCHARGE_DATE')
        """
        admit_col = 'ADMIT_DATE'
        discharge_col = 'DISCHARGE_DATE'
        for p in self.learner.tables.get(table, []):
            cn = p.name.upper()
            if p.is_date:
                if any(w in cn for w in ('ADMIT', 'ADMISSION', 'ARRIVAL')):
                    admit_col = p.name
                elif any(w in cn for w in ('DISCHARGE', 'DEPART', 'EXIT')):
                    discharge_col = p.name
        return admit_col, discharge_col

    def _resolve_visit_type_column(self, table: str) -> str:
        """Find the visit/encounter type column in a table.

        Returns:
            Column name string — defaults to 'VISIT_TYPE'
        """
        for p in self.learner.tables.get(table, []):
            cn = p.name.upper()
            if p.is_categorical and any(w in cn for w in ('VISIT_TYPE', 'ENCOUNTER_TYPE',
                                                           'SERVICE_TYPE', 'CARE_TYPE')):
                return p.name
        return 'VISIT_TYPE'

    @staticmethod
    def _is_npi_column(col_name: str) -> bool:
        """Check if a column represents a provider NPI reference.

        Matches: NPI, RENDERING_NPI, BILLING_NPI, ATTENDING_NPI, PROVIDER_NPI, etc.
        Does NOT match: NPI_COUNT, NPI_VALID (avoid false positives).
        """
        cn = col_name.upper()
        return cn in SemanticSQLComposer.NPI_COLUMN_NAMES or cn.endswith('_NPI')

    def _find_npi_column(self, table: str, preferred: str = None) -> str:
        """Find the provider NPI column in a table.

        Args:
            table: Table to search
            preferred: Optional preferred column name (e.g., 'RENDERING_NPI')

        Returns:
            NPI column name — defaults to 'RENDERING_NPI'
        """
        if preferred:
            for p in self.learner.tables.get(table, []):
                if p.name == preferred:
                    return p.name
        for p in self.learner.tables.get(table, []):
            if self._is_npi_column(p.name):
                return p.name
        return 'RENDERING_NPI'

    def _build_result_limit(self, q: str, intent: str = 'breakdown') -> int:
        """Determine the appropriate LIMIT for a query.

        Checks for explicit "top N" / "limit N" in the question first,
        then falls back to intent-appropriate defaults.

        Args:
            q: Question text (lowercase)
            intent: Query intent type

        Returns:
            Integer limit value
        """
        # Explicit user limit
        m = re.search(r'\btop\s+(\d+)\b', q)
        if m:
            return int(m.group(1))
        m = re.search(r'\blimit\s+(\d+)\b', q)
        if m:
            return int(m.group(1))

        # Intent-based defaults
        limits = {
            'breakdown': self.DEFAULT_BREAKDOWN_LIMIT,
            'rate': self.DEFAULT_BREAKDOWN_LIMIT,
            'trend': self.DEFAULT_TREND_LIMIT,
            'ranking': self.DEFAULT_RANKING_LIMIT,
            'comparison': self.DEFAULT_BREAKDOWN_LIMIT,
            'lookup': self.DEFAULT_LOOKUP_LIMIT,
        }
        return limits.get(intent, self.DEFAULT_BREAKDOWN_LIMIT)

    def _find_entity_id(self, table: str) -> str:
        """Find the primary ID column for an entity table.

        Checks ENTITY_ID_HINTS first, then falls back to first _ID column.
        """
        tbl_lower = table.lower().rstrip('s')
        for hint_key, candidates in self.ENTITY_ID_HINTS.items():
            if hint_key in tbl_lower:
                for cand in candidates:
                    for p in self.learner.tables.get(table, []):
                        if p.name == cand:
                            return p.name
        # Fallback: first _ID column
        for p in self.learner.tables.get(table, []):
            if p.is_id and p.name.endswith('_ID'):
                return p.name
        return f"{table.rstrip('s').upper()}_ID"

    def _stem_word(self, word: str) -> Set[str]:
        """Produce stem variants of a word for fuzzy matching.

        Returns set of stems: 'providers' → {'providers', 'provider'}
        """
        stems = {word}
        if word.endswith('s') and len(word) > 3:
            stems.add(word[:-1])
        if word.endswith('ies') and len(word) > 4:
            stems.add(word[:-3] + 'y')
        if word.endswith('ed') and len(word) > 4:
            stems.add(word[:-2])
        return stems

    def _stem_words(self, words: Set[str]) -> Set[str]:
        """Produce stem variants for a set of words."""
        stems = set()
        for w in words:
            stems.update(self._stem_word(w))
        return stems

    def _discover_dimension(self, q: str, fact: str) -> Optional[Dict]:
        """Universal dimension discovery. Finds ANY grouping dimension across
        ALL tables by scoring categorical columns against question terms.
        Returns {table, column, needs_join, join_col, dim_join_col} or None.

        This is the KEY innovation: instead of hardcoding dimension handlers
        (age group, gender, race, plan type, etc.), we discover the grouping
        dimension from the actual schema automatically."""
        group_term = self._extract_group_term(q)
        if not group_term:
            return None

        # ARCHITECTURAL LAYER 4: Domain disambiguation
        # "provider" in KP analytics → insurance plan type, not practitioners
        disambig = self._disambiguate_dimension_term(group_term, fact)
        if disambig:
            dis_table = disambig['table']
            dis_col = disambig['column']
            needs_join = (dis_table != fact)
            join_col = None
            dim_join_col = None
            if needs_join:
                fact_cols = {p.name for p in self.learner.tables.get(fact, [])}
                dis_cols = {p.name for p in self.learner.tables.get(dis_table, [])}
                shared_ids = sorted(c for c in fact_cols & dis_cols if c.endswith('_ID'))
                if shared_ids:
                    join_col = shared_ids[0]
                    dim_join_col = shared_ids[0]
                else:
                    jc = self.kg._get_join_condition(fact, dis_table)
                    if jc and '=' in jc:
                        parts = jc.split('=', 1)
                        join_col = parts[0].strip()
                        dim_join_col = parts[1].strip()
                    elif jc:
                        join_col = jc
                        dim_join_col = jc
                    else:
                        # Can't join — skip disambiguation, fall through to normal scoring
                        disambig = None
            if disambig:
                return {
                    'table': dis_table, 'column': dis_col,
                    'needs_join': needs_join, 'join_col': join_col,
                    'dim_join_col': dim_join_col,
                    'is_age_group': False,
                }

        # Special case: "age group" needs computed CASE WHEN
        if 'age' in group_term.lower():
            birth_col = self.learner.find_birth_date_column()
            if birth_col:
                birth_table = None
                for t, profiles in self.learner.tables.items():
                    if any(p.name == birth_col.name for p in profiles):
                        birth_table = t
                        break
                if birth_table:
                    needs_join = (birth_table != fact)
                    join_col = None
                    if needs_join:
                        fact_cols = {p.name for p in self.learner.tables.get(fact, [])}
                        birth_cols = {p.name for p in self.learner.tables.get(birth_table, [])}
                        shared_ids = sorted(c for c in fact_cols & birth_cols if c.endswith('_ID'))
                        join_col = shared_ids[0] if shared_ids else 'MEMBER_ID'
                    return {
                        'table': birth_table, 'column': birth_col.name,
                        'needs_join': needs_join, 'join_col': join_col,
                        'is_age_group': True,
                    }

        # General dimension discovery: score ALL categorical columns across ALL tables
        candidates = []
        term_lower = group_term.lower().strip()
        tn = term_lower.replace(' ', '_')

        for table, profiles in self.learner.tables.items():
            for p in profiles:
                if not (p.is_categorical or (not p.is_numeric and not p.is_date)):
                    continue
                cn = p.name.lower()
                score = 0.0

                # Exact match (highest priority)
                if cn == tn or cn == tn + '_id':
                    score = 10.0
                elif tn.replace('_', '') == cn.replace('_', ''):
                    score = 9.0
                # Table-qualified match: "type" for claims → CLAIM_TYPE
                table_stem = table.rstrip('s').lower()
                if cn == f"{table_stem}_{tn}":
                    score = 8.5
                # Partial word overlap
                for word in term_lower.split():
                    if len(word) >= 3:
                        if word in cn:
                            score = max(score, 5.0)
                        elif word in cn.replace('_', ' '):
                            score = max(score, 4.0)
                        for tag in p.semantic_tags:
                            if word in tag:
                                score = max(score, 3.0)

                if score > 0:
                    candidates.append({
                        'table': table, 'column': p.name,
                        'score': score, 'in_fact': (table == fact),
                    })

        if not candidates:
            return None

        # Prefer columns in the fact table, then highest score
        candidates.sort(key=lambda x: (x['in_fact'], x['score']), reverse=True)
        best = candidates[0]

        needs_join = not best['in_fact']
        join_col = None
        dim_join_col = None  # When different from join_col (e.g., PROVIDER_NPI→NPI)
        if needs_join:
            fact_cols = {p.name for p in self.learner.tables.get(fact, [])}
            dim_cols = {p.name for p in self.learner.tables.get(best['table'], [])}
            shared_ids = sorted(c for c in fact_cols & dim_cols if c.endswith('_ID'))
            if shared_ids:
                join_col = shared_ids[0]
                dim_join_col = shared_ids[0]  # Same column name in both tables
            else:
                jc = self.kg._get_join_condition(fact, best['table'])
                if not jc:
                    return None
                # Handle asymmetric join conditions like "PROVIDER_NPI=NPI"
                if '=' in jc:
                    parts = jc.split('=', 1)
                    join_col = parts[0].strip()      # fact table column
                    dim_join_col = parts[1].strip()   # dimension table column
                else:
                    join_col = jc
                    dim_join_col = jc

        return {
            'table': best['table'], 'column': best['column'],
            'needs_join': needs_join, 'join_col': join_col,
            'dim_join_col': dim_join_col,  # Column in the dimension table
            'is_age_group': False,
        }

    @staticmethod
    @staticmethod
    def _extract_age_interval(q: str) -> Optional[int]:
        """Extract custom age interval from question text.
        'by 5 years' -> 5, 'by 10 year' -> 10, 'in 5-year groups' -> 5."""
        m = re.search(r'(?:by|in|every|each)\s+(\d+)\s*[-\s]?years?', q.lower())
        if m:
            return int(m.group(1))
        m2 = re.search(r'(\d+)\s*[-\s]?year\s+(?:group|bucket|interval|band|range)', q.lower())
        if m2:
            return int(m2.group(1))
        return None

    @staticmethod
    def _age_group_case(col_expr: str, interval: int = None) -> str:
        """Static helper: CASE WHEN for age buckets from a birth date column.
        If interval is specified (e.g., 5), generates 5-year buckets.
        Otherwise uses default clinical age groups."""
        age_calc = f"(julianday('now') - julianday({col_expr})) / 365.25"

        if interval and interval > 0:
            # Dynamic N-year buckets: 0-4, 5-9, 10-14, ... up to max_age+
            cases = []
            max_age = 100
            for start in range(0, max_age, interval):
                end = start + interval
                label = f"'{start}-{end - 1}'"
                cases.append(f"WHEN {age_calc} < {end} THEN {label}")
            cases.append(f"ELSE '{max_age}+'")
            return f"CASE {' '.join(cases)} END"
        else:
            # Default clinical age groups
            return (
                f"CASE "
                f"WHEN {age_calc} < 18 THEN '0-17' "
                f"WHEN {age_calc} < 35 THEN '18-34' "
                f"WHEN {age_calc} < 50 THEN '35-49' "
                f"WHEN {age_calc} < 65 THEN '50-64' "
                f"ELSE '65+' END"
            )

    def _sql_threshold(self, q: str, fact: str, signals: Dict) -> Dict:
        """Frequency/threshold queries: members with >N visits via GROUP BY + HAVING.
        Also handles rate/percentage thresholds like 'denied rate above 20%'."""
        threshold = signals['threshold']

        # Detect rate/percentage threshold pattern (e.g., "denied claim rate above 20%")
        is_rate_threshold = '%' in q or any(w in q for w in ['rate', 'percentage', 'pct'])
        if is_rate_threshold:
            return self._sql_rate_threshold(q, fact, threshold)

        count_table = fact
        entity_id = self._find_entity_id(count_table)

        # Determine the subject entity from the question — e.g., "members with
        # more than 5 encounters" → group by MEMBER_ID, not ENCOUNTER_ID
        entity_words = ['member', 'patient', 'person', 'people', 'individual', 'who']
        subject_is_entity = any(w in q for w in entity_words)
        if subject_is_entity:
            # The subject is an entity (members) but the COUNTABLE thing may be
            # a different table (encounters, claims, etc.). Detect the countable
            # table from the question and switch count_table to it.
            q_words = set(re.findall(r'[a-z]+', q))
            for tbl_name in self.learner.tables:
                tbl_lower = tbl_name.lower()
                tbl_stem = tbl_lower.rstrip('s') if len(tbl_lower) > 3 else tbl_lower
                if tbl_stem in q_words or tbl_lower in q_words:
                    if tbl_lower != 'members' and tbl_lower != 'patients':
                        count_table = tbl_name
                        break
            # Find a member/patient ID column in the count table
            fact_cols = {p.name for p in self.learner.tables.get(count_table, [])}
            for candidate in ['MEMBER_ID', 'PATIENT_ID', 'MRN']:
                if candidate in fact_cols:
                    entity_id = candidate
                    break

        # Visit type filter
        visit_type_filter = None
        for keyword in ['emergency', 'inpatient', 'outpatient', 'home health', 'telehealth']:
            if keyword in q:
                result = self._resolve_category_value(keyword, [count_table])
                if result:
                    _, col, val = result
                    visit_type_filter = f"{col} = '{val}'"
                break

        where_parts = []
        if visit_type_filter:
            where_parts.append(visit_type_filter)
        where_clause = f" WHERE {' AND '.join(where_parts)}" if where_parts else ""

        if subject_is_entity:
            sql = (
                f"SELECT {entity_id}, COUNT(*) as visit_count FROM {count_table}{where_clause} "
                f"GROUP BY {entity_id} HAVING COUNT(*) > {threshold} "
                f"ORDER BY visit_count DESC LIMIT {self.DEFAULT_BREAKDOWN_LIMIT};"
            )
        else:
            sql = (
                f"SELECT COUNT(*) as member_count FROM ("
                f"SELECT {entity_id} FROM {count_table}{where_clause} "
                f"GROUP BY {entity_id} HAVING COUNT(*) > {threshold}) sub;"
            )
        return {
            'sql': sql, 'tables_used': [count_table],
            'confidence': 0.85, 'intent': 'filter',
            'explanation': f'Frequency: entities with >{threshold} occurrences',
        }

    def _sql_rate_threshold(self, q: str, fact: str, threshold: int) -> Optional[Dict]:
        """Rate/percentage threshold: 'providers with denied rate above 20%'.
        Groups by a dimension entity, computes a CASE-based rate, filters with HAVING."""
        # Discover the grouping entity (provider, facility, department, etc.)
        dim = self._discover_dimension(q, fact)
        group_col = f"{dim['table']}.{dim['column']}" if dim else None
        if not group_col:
            # Look for entity-specific columns in the fact table
            fact_cols = [p.name for p in self.learner.tables.get(fact, [])]
            entity_hints = {
                'provider': ['RENDERING_NPI', 'BILLING_NPI', 'NPI', 'PROVIDER_ID'],
                'facility': ['FACILITY', 'FACILITY_ID'],
                'department': ['DEPARTMENT', 'DEPARTMENT_ID'],
                'member': ['MEMBER_ID', 'MRN'],
                'patient': ['MEMBER_ID', 'MRN', 'PATIENT_ID'],
            }
            for hint, candidate_cols in entity_hints.items():
                if hint in q:
                    for cc in candidate_cols:
                        if cc in fact_cols:
                            group_col = cc
                            break
                    break
        if not group_col:
            return None

        # Discover the status/category value to compute rate for
        rate_keywords = ['denied', 'approved', 'rejected', 'pending', 'completed',
                         'cancelled', 'emergency', 'inpatient', 'outpatient']
        target_val = None
        target_col = None
        for kw in rate_keywords:
            if kw in q:
                result = self._resolve_category_value(kw, [fact])
                if result:
                    _, target_col, target_val = result
                    break
        if not target_col or not target_val:
            return None

        # Build: GROUP BY dim, HAVING rate > threshold%
        tables_used = [fact]
        from_clause = fact
        if dim and dim.get('needs_join') and dim['table'] != fact:
            fact_jc = dim.get('join_col', 'MEMBER_ID')
            dim_jc = dim.get('dim_join_col') or fact_jc
            from_clause = f"{fact} JOIN {dim['table']} ON {fact}.{fact_jc} = {dim['table']}.{dim_jc}"
            tables_used.append(dim['table'])

        # Sanitize alias
        safe_val = re.sub(r'[^a-zA-Z0-9_]', '_', target_val.lower())

        sql = (
            f"SELECT {group_col}, "
            f"COUNT(*) as total, "
            f"SUM(CASE WHEN {target_col} = '{target_val}' THEN 1 ELSE 0 END) as {safe_val}_count, "
            f"ROUND(100.0 * SUM(CASE WHEN {target_col} = '{target_val}' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as {safe_val}_rate "
            f"FROM {from_clause} "
            f"GROUP BY {group_col} "
            f"HAVING {safe_val}_rate > {threshold} "
            f"ORDER BY {safe_val}_rate DESC;"
        )
        return {
            'sql': sql, 'tables_used': tables_used,
            'confidence': 0.85, 'intent': 'filter',
            'explanation': f'{target_val} rate above {threshold}% grouped by {group_col}',
        }

    def _sql_ratio(self, q: str, fact: str, cat_a: str, cat_b: str) -> Optional[Dict]:
        """Ratio queries like 'emergency to outpatient ratio' via CASE WHEN division."""
        for table in [fact] + list(self.learner.tables.keys()):
            res_a = self._resolve_category_value(cat_a, [table])
            res_b = self._resolve_category_value(cat_b, [table])
            if res_a and res_b:
                ta, col_a, val_a = res_a
                tb, col_b, val_b = res_b
                if ta == tb and col_a == col_b:
                    sql = (
                        f"SELECT "
                        f"SUM(CASE WHEN {col_a} = '{val_a}' THEN 1 ELSE 0 END) as {cat_a}_count, "
                        f"SUM(CASE WHEN {col_a} = '{val_b}' THEN 1 ELSE 0 END) as {cat_b}_count, "
                        f"ROUND(1.0 * SUM(CASE WHEN {col_a} = '{val_a}' THEN 1 ELSE 0 END) / "
                        f"NULLIF(SUM(CASE WHEN {col_a} = '{val_b}' THEN 1 ELSE 0 END), 0), 4) as ratio "
                        f"FROM {ta};"
                    )
                    return {
                        'sql': sql, 'tables_used': [ta],
                        'confidence': 0.9, 'intent': 'aggregate',
                        'explanation': f'Ratio of {cat_a} to {cat_b}',
                    }
        return None

    def _sql_ranking_comparison(self, q: str, fact: str,
                                ranking: Tuple, columns: List[Dict]) -> Optional[Dict]:
        """Ranking comparison: 'top N vs bottom N [entity]'.
        Schema-agnostic: discovers the entity table, a metric to rank by
        (preferring numeric columns from a joinable fact table), and produces
        a UNION of top-N (DESC) and bottom-N (ASC) subqueries."""
        n_top, n_bot, entity_word = ranking

        # LAYER 4: Check if entity_word is a disambiguated term
        # e.g., "top 5 vs bottom 5 providers" → providers = plan type, not practitioner table
        disambig = self._disambiguate_dimension_term(entity_word, fact)
        if disambig:
            # Disambiguation applies — build as a dimension-grouped query instead
            practitioner_terms = {'doctor', 'physician', 'specialist', 'npi', 'practitioner',
                                  'clinician', 'panel', 'workforce', 'staff', 'specialty'}
            q_words = set(re.findall(r'[a-z]+', q.lower()))
            if not (q_words & practitioner_terms):
                dim = self._build_disambiguated_dimension(disambig, fact)
                if dim:
                    signals = self._extract_signals(q)
                    return self._sql_universal(q, fact, dim, signals, columns, [])

        # 1. Find the entity table — match entity_word against table names
        entity_table = None
        entity_stem = entity_word.rstrip('s') if len(entity_word) > 2 else entity_word
        for tbl_name in self.learner.tables:
            tbl_lower = tbl_name.lower()
            tbl_stem = tbl_lower.rstrip('s') if len(tbl_lower) > 2 else tbl_lower
            if tbl_stem == entity_stem or tbl_lower == entity_word:
                entity_table = tbl_name
                break
        if not entity_table:
            # Fallback: use fact table
            entity_table = fact

        # 2. Find a label column (name, description) for the entity — prefer non-ID categorical
        #    Priority: name_col > desc_col > other categorical label > first categorical
        #    Exclude numeric columns (TENURE_MONTHS, AGE, etc.)
        entity_profiles = self.learner.tables.get(entity_table, [])
        name_col = None
        desc_col = None
        label_col = None
        id_col = None
        for p in entity_profiles:
            pn = p.name.lower()
            if p.is_id and p.name.endswith('_ID'):
                if not id_col:
                    id_col = p.name
            elif hasattr(p, 'is_numeric') and p.is_numeric:
                continue  # skip numeric columns
            elif not p.is_id and not p.name.endswith('_ID'):
                if any(w in pn for w in ('name', 'first_name', 'last_name', 'fname', 'lname')):
                    if not name_col:
                        name_col = p.name
                elif any(w in pn for w in ('description', 'desc', 'text', 'detail')):
                    if not desc_col:
                        desc_col = p.name
                elif p.is_categorical and any(w in pn for w in
                        ('specialty', 'type', 'title', 'label', 'category', 'code', 'icd', 'dx')):
                    if not label_col:
                        label_col = p.name
        # Priority chain
        label_col = name_col or desc_col or label_col
        if not label_col:
            # Last resort: use first non-ID, non-numeric categorical column
            for p in entity_profiles:
                if (p.is_categorical and not p.is_id and not p.name.endswith('_ID')
                        and not (hasattr(p, 'is_numeric') and p.is_numeric)):
                    label_col = p.name
                    break
        if not id_col:
            id_col = self._find_entity_id(entity_table)

        # 3. Find a metric to rank by — prefer cost/amount columns from joinable
        #    fact tables (e.g., claims for providers), then entity table columns
        metric_col = None
        metric_table = None
        metric_agg = 'SUM'
        entity_cols = {p.name for p in entity_profiles}

        # First try: look for cost/amount columns in joinable fact tables
        #            (these are more meaningful for ranking than entity attributes)
        if True:
            entity_id_cols = {p.name for p in entity_profiles if p.is_id or p.name.endswith('_ID')}
            for tbl_name, profiles in self.learner.tables.items():
                if tbl_name == entity_table:
                    continue
                tbl_cols = {p.name for p in profiles}
                shared_ids = entity_id_cols & tbl_cols
                if not shared_ids:
                    # Also check NPI-style joins
                    for ec in entity_cols:
                        for tc in tbl_cols:
                            if ec == tc and ('NPI' in ec or ec.endswith('_ID')):
                                shared_ids.add(ec)
                    # Also check indirect joins via ID patterns
                    for p in profiles:
                        if self._is_npi_column(p.name):
                            if 'NPI' in entity_cols:
                                shared_ids.add('NPI')
                if shared_ids:
                    # Found a joinable table — look for cost/amount columns
                    for p in profiles:
                        if p.is_numeric and not p.is_id and not p.name.endswith('_ID'):
                            cn = p.name.lower()
                            if any(w in cn for w in ('amount', 'cost', 'paid', 'billed', 'allowed')):
                                metric_col = p.name
                                metric_table = tbl_name
                                break
                    if metric_col:
                        break

        # Second try: find a metric column in the entity table itself
        if not metric_col:
            metric_col = self._find_numeric_column(q, columns, [entity_table])
            if metric_col:
                metric_table = entity_table

        # Third try: just use COUNT(*) from a joinable table
        if not metric_col:
            for tbl_name, profiles in self.learner.tables.items():
                if tbl_name == entity_table:
                    continue
                tbl_cols = {p.name for p in profiles}
                entity_id_cols_check = {p.name for p in entity_profiles if p.is_id or p.name.endswith('_ID')}
                shared = entity_id_cols_check & tbl_cols
                if not shared:
                    for p in profiles:
                        if self._is_npi_column(p.name):
                            if 'NPI' in entity_cols:
                                shared.add('NPI')
                if shared:
                    metric_col = None  # Will use COUNT(*)
                    metric_table = tbl_name
                    metric_agg = 'COUNT'
                    break

        # 4. Build the SQL
        tables_used = [entity_table]
        if metric_table and metric_table != entity_table:
            tables_used.append(metric_table)

        # Determine JOIN condition
        join_clause = ""
        if metric_table and metric_table != entity_table:
            entity_col_set = {p.name for p in entity_profiles}
            metric_profiles = self.learner.tables.get(metric_table, [])
            metric_col_set = {p.name for p in metric_profiles}

            # Direct shared ID columns
            shared_ids = sorted(c for c in entity_col_set & metric_col_set
                              if c.endswith('_ID') or c == 'NPI')
            if shared_ids:
                jc = shared_ids[0]
                join_clause = f" JOIN {metric_table} ON {entity_table}.{jc} = {metric_table}.{jc}"
            else:
                # NPI-based joins (provider.NPI = claims.RENDERING_NPI)
                for p in metric_profiles:
                    if 'NPI' in p.name and 'NPI' in entity_col_set:
                        join_clause = f" JOIN {metric_table} ON {entity_table}.NPI = {metric_table}.{p.name}"
                        break

        # Build metric expression — for diagnosis/condition entities,
        # include BOTH code AND description if both exist
        group_col = id_col
        select_label = f"{entity_table}.{id_col}"
        if label_col:
            select_label = f"{entity_table}.{label_col}"
            group_col = label_col
            # For diagnosis entities: show code + description together
            if entity_stem in ('diagnosis', 'diagnose', 'condition', 'disease', 'dx'):
                code_col = None
                for p in entity_profiles:
                    pn = p.name.lower()
                    if any(w in pn for w in ('code', 'icd')) and p.name != label_col:
                        code_col = p.name
                        break
                if code_col:
                    select_label = (f"{entity_table}.{code_col} || ' - ' || "
                                    f"{entity_table}.{label_col}")
                    group_col = f"{entity_table}.{code_col}, {entity_table}.{label_col}"

        if metric_col and metric_agg != 'COUNT':
            metric_expr = f"ROUND({metric_agg}(CAST({metric_table or entity_table}.{metric_col} AS REAL)), 2)"
            metric_alias = f"total_{metric_col.lower()}"
        elif metric_agg == 'COUNT' and metric_table:
            metric_expr = "COUNT(*)"
            metric_alias = "record_count"
        else:
            # Last resort: just count rows in the entity table
            metric_expr = "COUNT(*)"
            metric_alias = "record_count"
            join_clause = ""
            tables_used = [entity_table]

        from_clause = f"{entity_table}{join_clause}"

        sql = (
            f"SELECT 'Top {n_top}' as ranking_group, {select_label} as {entity_stem}, "
            f"{metric_expr} as {metric_alias} "
            f"FROM {from_clause} "
            f"GROUP BY {entity_table}.{group_col} "
            f"ORDER BY {metric_alias} DESC LIMIT {n_top} "
            f"UNION ALL "
            f"SELECT 'Bottom {n_bot}' as ranking_group, {select_label} as {entity_stem}, "
            f"{metric_expr} as {metric_alias} "
            f"FROM {from_clause} "
            f"GROUP BY {entity_table}.{group_col} "
            f"ORDER BY {metric_alias} ASC LIMIT {n_bot};"
        )

        # SQLite doesn't support ORDER BY in UNION subqueries without wrapping
        # Use subquery approach instead
        sql = (
            f"SELECT * FROM ("
            f"SELECT 'Top {n_top}' as ranking_group, {select_label} as {entity_stem}, "
            f"{metric_expr} as {metric_alias} "
            f"FROM {from_clause} "
            f"GROUP BY {entity_table}.{group_col} "
            f"ORDER BY {metric_alias} DESC LIMIT {n_top}"
            f") UNION ALL SELECT * FROM ("
            f"SELECT 'Bottom {n_bot}' as ranking_group, {select_label} as {entity_stem}, "
            f"{metric_expr} as {metric_alias} "
            f"FROM {from_clause} "
            f"GROUP BY {entity_table}.{group_col} "
            f"ORDER BY {metric_alias} ASC LIMIT {n_bot}"
            f");"
        )

        return {
            'sql': sql, 'tables_used': tables_used,
            'confidence': 0.88, 'intent': 'comparison',
            'explanation': f'Top {n_top} vs bottom {n_bot} {entity_word} ranked by {metric_alias}',
        }

    def _sql_comparison(self, q: str, fact: str, cat_a: str, cat_b: str,
                        metric_word: Optional[str], columns: List[Dict]) -> Optional[Dict]:
        """Comparison queries like 'compare inpatient vs outpatient costs'."""
        # Build search list with uppercase variants (learner stores table names in uppercase)
        search_tables = []
        for t in [fact, 'encounters', 'claims']:
            search_tables.append(t.upper())

        res_a = self._resolve_category_value(cat_a, search_tables)
        res_b = self._resolve_category_value(cat_b, search_tables)
        if not (res_a and res_b):
            return None
        ta, col_a, val_a = res_a
        tb, col_b, val_b = res_b
        if ta != tb or col_a != col_b:
            return None

        if metric_word and any(w in metric_word for w in ['cost', 'amount', 'paid', 'billed']):
            # Cost comparison — need numeric column
            cost_search_table = 'CLAIMS' if ta == 'ENCOUNTERS' else ta
            cost_col = self._find_numeric_column(q, columns, [cost_search_table])
            if not cost_col:
                for p in self.learner.tables.get(cost_search_table, []):
                    if p.is_numeric and ('amount' in p.name.lower() or 'cost' in p.name.lower()):
                        cost_col = p.name
                        break
            if not cost_col:
                cost_col = self.FINANCIAL_COLUMNS[1] if len(self.FINANCIAL_COLUMNS) > 1 else self.FINANCIAL_COLUMNS[0]

            if ta == 'ENCOUNTERS':
                sql = (
                    f"SELECT '{val_a}' as category, COUNT(c.CLAIM_ID) as count, "
                    f"ROUND(AVG(CAST(c.{cost_col} AS REAL)), 2) as avg_cost, "
                    f"ROUND(SUM(CAST(c.{cost_col} AS REAL)), 2) as total_cost "
                    f"FROM CLAIMS c JOIN ENCOUNTERS e ON c.ENCOUNTER_ID = e.ENCOUNTER_ID "
                    f"WHERE e.{col_a} = '{val_a}' "
                    f"UNION ALL SELECT '{val_b}' as category, COUNT(c.CLAIM_ID) as count, "
                    f"ROUND(AVG(CAST(c.{cost_col} AS REAL)), 2) as avg_cost, "
                    f"ROUND(SUM(CAST(c.{cost_col} AS REAL)), 2) as total_cost "
                    f"FROM CLAIMS c JOIN ENCOUNTERS e ON c.ENCOUNTER_ID = e.ENCOUNTER_ID "
                    f"WHERE e.{col_a} = '{val_b}';"
                )
                return {
                    'sql': sql, 'tables_used': ['CLAIMS', 'ENCOUNTERS'],
                    'confidence': 0.9, 'intent': 'comparison',
                    'explanation': f'Cost comparison: {cat_a} vs {cat_b}',
                }
            else:
                sql = (
                    f"SELECT '{val_a}' as category, COUNT(*) as count, "
                    f"ROUND(AVG(CAST({cost_col} AS REAL)), 2) as avg_cost, "
                    f"ROUND(SUM(CAST({cost_col} AS REAL)), 2) as total_cost "
                    f"FROM {ta} WHERE {col_a} = '{val_a}' "
                    f"UNION ALL SELECT '{val_b}' as category, COUNT(*) as count, "
                    f"ROUND(AVG(CAST({cost_col} AS REAL)), 2) as avg_cost, "
                    f"ROUND(SUM(CAST({cost_col} AS REAL)), 2) as total_cost "
                    f"FROM {ta} WHERE {col_a} = '{val_b}';"
                )
                return {
                    'sql': sql, 'tables_used': [ta],
                    'confidence': 0.9, 'intent': 'comparison',
                    'explanation': f'Cost comparison: {cat_a} vs {cat_b}',
                }
        else:
            sql = (
                f"SELECT {col_a}, COUNT(*) as count, "
                f"ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) as pct "
                f"FROM {ta} WHERE {col_a} IN ('{val_a}', '{val_b}') "
                f"GROUP BY {col_a};"
            )
            return {
                'sql': sql, 'tables_used': [ta],
                'confidence': 0.85, 'intent': 'comparison',
                'explanation': f'Count comparison: {cat_a} vs {cat_b}',
            }

    def _sql_intersection(self, q: str, fact: str, cat_a: str, cat_b: str) -> Optional[Dict]:
        """Members with BOTH X and Y via INTERSECT."""
        res_a = self._resolve_category_value(cat_a, ['encounters', fact])
        res_b = self._resolve_category_value(cat_b, ['encounters', fact])
        if not (res_a and res_b):
            return None
        ta, col_a, val_a = res_a
        _, col_b, val_b = res_b
        if col_a != col_b:
            return None
        entity_id = self._find_entity_id(ta)
        sql = (
            f"SELECT COUNT(*) as member_count FROM ("
            f"SELECT {entity_id} FROM {ta} WHERE {col_a} = '{val_a}' "
            f"INTERSECT "
            f"SELECT {entity_id} FROM {ta} WHERE {col_a} = '{val_b}'"
            f") sub;"
        )
        return {
            'sql': sql, 'tables_used': [ta],
            'confidence': 0.9, 'intent': 'count',
            'explanation': f'Members with both {cat_a} and {cat_b}',
        }

    def _sql_date_arithmetic(self, q: str, fact: str) -> Optional[Dict]:
        """Handle date arithmetic queries like 'average processing time for claims'."""
        for table in [fact] + list(self.learner.tables.keys()):
            dates = self.learner.get_date_columns(table)
            if len(dates) < 2:
                continue
            # Find the best pair
            date_pairs = []
            for i, d1 in enumerate(dates):
                for d2 in dates[i + 1:]:
                    n1, n2 = d1.name.lower(), d2.name.lower()
                    score = 0
                    if ('submit' in n1 and 'adjudicat' in n2) or ('adjudicat' in n1 and 'submit' in n2):
                        score = 10
                    elif ('admit' in n1 and 'discharge' in n2) or ('discharge' in n1 and 'admit' in n2):
                        score = 8
                    elif ('start' in n1 and 'end' in n2) or ('end' in n1 and 'start' in n2):
                        score = 5
                    elif 'service' in n1 or 'service' in n2:
                        score = 3
                    else:
                        score = 1
                    if score > 0:
                        date_pairs.append((d1.name, d2.name, score))
            if not date_pairs:
                continue
            date_pairs.sort(key=lambda x: x[2], reverse=True)
            start_col, end_col = date_pairs[0][0], date_pairs[0][1]
            # Ensure end > start semantically
            if any(w in start_col.lower() for w in ['adjudicat', 'discharge', 'end', 'out']):
                start_col, end_col = end_col, start_col
            sql = (
                f"SELECT "
                f"ROUND(AVG(julianday({end_col}) - julianday({start_col})), 1) as avg_days, "
                f"ROUND(MIN(julianday({end_col}) - julianday({start_col})), 1) as min_days, "
                f"ROUND(MAX(julianday({end_col}) - julianday({start_col})), 1) as max_days, "
                f"COUNT(*) as record_count "
                f"FROM {table} "
                f"WHERE {end_col} != '' AND {start_col} != '' "
                f"AND {end_col} IS NOT NULL AND {start_col} IS NOT NULL;"
            )
            return {
                'sql': sql, 'tables_used': [table],
                'confidence': 0.85, 'intent': 'aggregate',
                'explanation': f'Date arithmetic: average days between {start_col} and {end_col}',
            }
        return None

    def _sql_readmission(self, q: str, fact: str) -> Dict:
        """Readmission rate: members readmitted within 30 days of discharge.

        Data-driven dimension support:
        - Time dimensions: "by quarter", "by month", "trend" → GROUP BY date bucket
        - Categorical: "by region", "by plan type", "by specialty" → GROUP BY column
        - Cross-table: "by specialty" → JOIN providers table
        - No dimension → aggregate (1 row)
        """
        # ── Detect dimension ──
        group_term = None
        for pat in [r'by\s+(\w+(?:\s+\w+)?)', r'per\s+(\w+)', r'across\s+(?:all\s+)?(\w+(?:\s+\w+)?)']:
            m = re.search(pat, q)
            if m:
                group_term = m.group(1).lower().strip()
                break

        # ── Detect time-based grouping ──
        time_patterns = [r'\btrend\b', r'\bover\s+time\b', r'\bmonthly\b', r'\bquarterly\b',
                         r'\byearly\b', r'\bby\s+month\b', r'\bby\s+quarter\b', r'\bby\s+year\b',
                         r'\blast\s+\d+\s+months?\b', r'\bover\s+the\s+(?:last|past)\b']
        is_time_query = any(re.search(p, q) for p in time_patterns)

        # ── Build readmission CTE via production helper ──
        enc_table = self._find_table_by_role('encounters')
        # Gather extra categorical + NPI columns for dimension joins
        extra_cols = ['KP_REGION']
        for p in self.learner.tables.get(enc_table, []):
            if p.is_categorical and p.name not in extra_cols and not p.name.endswith('_ID'):
                extra_cols.append(p.name)
            if self._is_npi_column(p.name) and p.name not in extra_cols:
                extra_cols.append(p.name)

        base_cte, readmit_case = self._build_readmission_cte(
            alias='ip', extra_columns=extra_cols
        )
        member_col = self._resolve_member_column(enc_table)
        admit_col, discharge_col = self._resolve_admission_columns(enc_table)
        # Reconstruct the full column list for downstream join/grouping checks
        enc_columns = [member_col, admit_col, discharge_col] + extra_cols
        total_expr = f"COUNT(DISTINCT ip.{member_col})"

        # ── Path 1: Time-based trend ──
        if is_time_query or (group_term and group_term in ('month', 'quarter', 'year', 'time')):
            bucket_field = 'quarter'  # default
            if 'month' in q:
                bucket = "SUBSTR(ADMIT_DATE, 1, 7)"
                bucket_field = 'month'
            elif 'year' in q:
                bucket = "SUBSTR(ADMIT_DATE, 1, 4)"
                bucket_field = 'year'
            else:
                # Quarter: YYYY-Q#
                bucket = ("SUBSTR(ADMIT_DATE, 1, 4) || '-Q' || "
                          "((CAST(SUBSTR(ADMIT_DATE, 6, 2) AS INTEGER) - 1) / 3 + 1)")
            sql = (
                f"{base_cte} "
                f"SELECT {bucket} as period, "
                f"{total_expr} as total_admissions, "
                f"{readmit_case} as readmissions, "
                f"ROUND(100.0 * {readmit_case} / NULLIF({total_expr}, 0), 2) as readmission_rate "
                f"FROM ip GROUP BY period ORDER BY period LIMIT {self.DEFAULT_TREND_LIMIT};"
            )
            return {
                'sql': sql, 'tables_used': ['encounters'],
                'confidence': 0.88, 'intent': 'rate',
                'explanation': f'Readmission rate trend by {bucket_field}',
            }

        # ── Path 2: Categorical dimension ──
        if group_term:
            # Try to resolve group_term to a column in encounters or a cross-table join
            dim_col = None
            dim_join = None

            # Check encounters columns directly
            for p in self.learner.tables.get('encounters', []):
                cn = p.name.lower()
                if group_term in cn or cn.replace('_', ' ') == group_term:
                    dim_col = p.name
                    break
                # "region" → KP_REGION, "plan" → PLAN_TYPE, etc.
                col_words = cn.replace('_', ' ').split()
                if group_term.rstrip('s') in [w.rstrip('s') for w in col_words]:
                    dim_col = p.name
                    break

            # If not found, try cross-table (e.g., "by specialty" → providers, "by plan type" → members)
            # Collect ALL candidates and pick the best one (prefer MEMBER_ID joins, smaller tables)
            if not dim_col:
                candidates = []  # [(table, column, join_clause, priority)]
                for tbl_name, profiles in self.learner.tables.items():
                    if tbl_name == 'encounters':
                        continue
                    for p in profiles:
                        cn = p.name.lower()
                        cn_words = cn.replace('_', ' ').split()
                        gt = group_term.rstrip('s') if len(group_term) > 3 else group_term
                        # Multi-word: check each word of group_term against column name
                        gt_words = gt.split()
                        col_match = (group_term in cn or gt in cn.replace('_', ' ') or
                                     cn.replace('_', ' ').replace('kp ', '') == group_term)
                        if not col_match and len(gt_words) > 1:
                            # Multi-word: require the MOST specific word to match
                            # "plan type" → must match "plan" (not just "type")
                            # Use the first word as the primary match key
                            primary_word = gt_words[0].rstrip('s') if len(gt_words[0]) > 3 else gt_words[0]
                            if primary_word in cn or any(primary_word == cw.rstrip('s') for cw in cn_words):
                                col_match = True
                        elif not col_match:
                            # Single word: check against column name
                            gw_stem = gt.rstrip('s') if len(gt) > 3 else gt
                            if any(gw_stem == cw.rstrip('s') for cw in cn_words):
                                col_match = True
                        if not (p.is_categorical and not p.is_id and col_match):
                            continue
                        # Found matching column — determine join path
                        # Only use columns that are IN the ip CTE (enc_columns list)
                        enc_cte_set = set(enc_columns)
                        tbl_cols_set = {pp.name for pp in profiles}
                        shared = enc_cte_set & tbl_cols_set
                        # Prefer MEMBER_ID for joins (most reliable FK)
                        if 'MEMBER_ID' in shared:
                            join = f"JOIN {tbl_name} ON ip.MEMBER_ID = {tbl_name}.MEMBER_ID"
                            candidates.append((tbl_name, p.name, join, 10))
                        elif tbl_name == 'providers':
                            # NPI join for providers
                            for ep_name in enc_columns:
                                if 'NPI' in ep_name and ep_name != 'NPI':
                                    if 'NPI' in tbl_cols_set:
                                        join = f"JOIN {tbl_name} ON ip.{ep_name} = {tbl_name}.NPI"
                                        candidates.append((tbl_name, p.name, join, 8))
                                        break
                        else:
                            shared_ids = [c for c in shared if c.endswith('_ID')]
                            if shared_ids:
                                jc = shared_ids[0]
                                join = f"JOIN {tbl_name} ON ip.{jc} = {tbl_name}.{jc}"
                                candidates.append((tbl_name, p.name, join, 5))
                        break

                # Pick best candidate (highest priority, smallest table for tie-breaking)
                if candidates:
                    candidates.sort(key=lambda c: (-c[3],
                        self.learner.table_row_counts.get(c[0], 0) if hasattr(self.learner, 'table_row_counts') else 0))
                    best = candidates[0]
                    dim_col = best[1]
                    dim_join = best[2]

            # Fallback: resolve_group_column but ONLY for columns in the ip CTE
            if not dim_col:
                resolved = self._resolve_group_column(group_term, [], ['encounters'])
                if resolved and resolved in enc_columns:
                    dim_col = resolved

            if dim_col:
                # Determine if dim_col is in ip CTE or needs a join
                col_prefix = ""
                tables_used = ['encounters']
                join_clause = ""
                if dim_col in enc_columns:
                    col_prefix = "ip."
                elif dim_join:
                    join_clause = f" {dim_join}"
                    join_table = dim_join.split('JOIN ')[1].split(' ON')[0].strip()
                    tables_used.append(join_table)
                    col_prefix = f"{join_table}."
                else:
                    col_prefix = "ip."

                sql = (
                    f"{base_cte} "
                    f"SELECT {col_prefix}{dim_col}, "
                    f"{total_expr} as total_admissions, "
                    f"{readmit_case} as readmissions, "
                    f"ROUND(100.0 * {readmit_case} / NULLIF({total_expr}, 0), 2) as readmission_rate "
                    f"FROM ip{join_clause} GROUP BY {col_prefix}{dim_col} ORDER BY readmission_rate DESC LIMIT {self.DEFAULT_BREAKDOWN_LIMIT};"
                )
                return {
                    'sql': sql, 'tables_used': tables_used,
                    'confidence': 0.88, 'intent': 'rate',
                    'explanation': f'Readmission rate by {dim_col}',
                }

        # ── Path 3: Aggregate (no dimension) ──
        sql = (
            f"{base_cte} "
            f"SELECT "
            f"{total_expr} as total_inpatient_members, "
            f"{readmit_case} as readmitted_members, "
            f"ROUND(100.0 * {readmit_case} / NULLIF({total_expr}, 0), 2) as readmission_rate_pct "
            f"FROM ip;"
        )
        return {
            'sql': sql, 'tables_used': ['encounters'],
            'confidence': 0.85, 'intent': 'rate',
            'explanation': 'Readmission rate: members readmitted within 30 days of discharge',
        }

    def _detect_cross_metric_rate(self, q: str) -> Optional[str]:
        """Detect if a readmission query ALSO asks about another rate metric.
        Returns the cross-metric keyword (e.g., 'denial') or None.

        Only fires for CORRELATION/COMPARISON questions between two rates.
        "Do high-denial providers also have high readmission?" → cross-metric.
        "Average cost for readmitted patients" → NOT cross-metric (cost is the
        target metric, readmission is a filter, not a second rate to compare).
        """
        q_lower = q.lower()

        # If the question asks for a METRIC (avg/total/count) OF readmitted patients,
        # that's a filtered aggregation, NOT a cross-metric comparison.
        # E.g., "average cost for readmitted patients", "total claims for readmitted members"
        filter_patterns = [
            r'(?:average|avg|mean|total|sum|count|median)\s+\w+\s+(?:for|of|among)\s+(?:\w+\s+)?readmit',
            r'(?:cost|amount|charge|paid|billed|spending)\s+(?:for|of|among)\s+(?:\w+\s+)?readmit',
            r'readmit\w*\s+(?:patient|member|cost|charge|amount|spending)',
        ]
        if any(re.search(p, q_lower) for p in filter_patterns):
            return None

        # Cross-metric patterns: "high-denial ... readmission", "denial and readmission",
        # "do X also have Y", correlations between two rates
        cross_keywords = {
            'denial': ['denial', 'denied', 'deny', 'denials'],
            'approval': ['approval', 'approved', 'approve'],
            'cost': ['cost', 'expensive', 'costly', 'high-cost', 'spending'],
        }
        for metric, keywords in cross_keywords.items():
            if any(kw in q_lower for kw in keywords):
                return metric
        return None

    def _sql_cross_metric_readmission(self, q: str, fact: str, cross_metric: str) -> Dict:
        """Generate a provider-level query combining readmission rate with another metric.
        E.g., "Do high-denial providers also have high readmission rates?" →
        provider | total_claims | denied_claims | denial_rate | readmissions | readmission_rate"""
        # Find inpatient value
        inpatient_val = 'INPATIENT'
        res = self._resolve_category_value('inpatient', ['encounters'])
        if res:
            _, _, inpatient_val = res

        # Find the status column and target value for the cross metric
        status_col = 'CLAIM_STATUS'
        target_val = 'DENIED'
        if cross_metric == 'denial':
            # Try to resolve actual column/value from data
            res = self._resolve_category_value('denied', ['claims'])
            if res:
                _, status_col, target_val = res
        elif cross_metric == 'approval':
            res = self._resolve_category_value('approved', ['claims'])
            if res:
                _, status_col, target_val = res

        # Find provider NPI columns via schema-aware helpers
        claims_table = self._find_table_by_role('claims')
        enc_table = self._find_table_by_role('encounters')
        prov_table = self._find_table_by_role('providers')
        claims_npi = self._find_npi_column(claims_table)
        enc_npi = self._find_npi_column(enc_table)

        # Find provider label column (name or specialty for display)
        provider_name = 'NPI'
        for p in self.learner.tables.get(prov_table, []):
            if any(w in p.name.lower() for w in ('name', 'first', 'last', 'specialty')):
                provider_name = p.name
                break

        safe_metric = re.sub(r'[^a-zA-Z0-9_]', '_', cross_metric.lower())
        safe_target = re.sub(r'[^a-zA-Z0-9_]', '_', target_val.lower())

        if cross_metric in ('denial', 'approval'):
            # ARCHITECTURAL LAYER 3: Data-aware thresholds
            # Cross-metric queries join TWO CTEs — rare events (readmissions, high denial)
            # concentrate in smaller providers. Use a generous retention target so the
            # intersection of two CTEs still has meaningful results.
            claims_having = self._compute_adaptive_having('claims', claims_npi, min_useful=self.HAVING_MIN_USEFUL_CROSS_METRIC)
            # For encounters: threshold on INPATIENT-filtered subset (much smaller)
            enc_having = self._compute_adaptive_having(
                'encounters', enc_npi, min_useful=10,
                where_clause=f"VISIT_TYPE = '{inpatient_val}' AND DISCHARGE_DATE != '' AND ADMIT_DATE != ''",
                count_expr='COUNT(DISTINCT MEMBER_ID)'
            )

            # CTE 1: denial rate per provider from claims
            # CTE 2: readmission rate per provider from encounters
            sql = (
                f"WITH denial_stats AS ("
                f"SELECT {claims_npi} as provider_npi, "
                f"COUNT(*) as total_claims, "
                f"SUM(CASE WHEN {status_col} = '{target_val}' THEN 1 ELSE 0 END) as {safe_target}_count, "
                f"ROUND(100.0 * SUM(CASE WHEN {status_col} = '{target_val}' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as {safe_metric}_rate "
                f"FROM claims GROUP BY {claims_npi} HAVING COUNT(*) >= {claims_having}"
                f"), "
                f"ip AS ("
                f"SELECT {enc_npi}, MEMBER_ID, ADMIT_DATE, DISCHARGE_DATE, "
                f"LAG(DISCHARGE_DATE) OVER (PARTITION BY MEMBER_ID ORDER BY ADMIT_DATE) as prev_discharge "
                f"FROM encounters "
                f"WHERE VISIT_TYPE = '{inpatient_val}' AND DISCHARGE_DATE != '' AND ADMIT_DATE != ''"
                f"), "
                f"readmit_stats AS ("
                f"SELECT {enc_npi} as provider_npi, "
                f"COUNT(DISTINCT MEMBER_ID) as total_patients, "
                f"COUNT(DISTINCT CASE WHEN julianday(ADMIT_DATE) - julianday(prev_discharge) BETWEEN 0 AND {self.READMISSION_WINDOW_DAYS} "
                f"THEN MEMBER_ID END) as readmitted_patients, "
                f"ROUND(100.0 * COUNT(DISTINCT CASE WHEN julianday(ADMIT_DATE) - julianday(prev_discharge) BETWEEN 0 AND {self.READMISSION_WINDOW_DAYS} "
                f"THEN MEMBER_ID END) / NULLIF(COUNT(DISTINCT MEMBER_ID), 0), 2) as readmission_rate "
                f"FROM ip GROUP BY {enc_npi} HAVING COUNT(DISTINCT MEMBER_ID) >= {enc_having}"
                f") "
                f"SELECT d.provider_npi, "
                f"p.{provider_name} as provider_name, "
                f"d.total_claims, d.{safe_target}_count, d.{safe_metric}_rate, "
                f"COALESCE(r.total_patients, 0) as total_patients, "
                f"COALESCE(r.readmitted_patients, 0) as readmitted_patients, "
                f"COALESCE(r.readmission_rate, 0) as readmission_rate "
                f"FROM denial_stats d "
                f"LEFT JOIN readmit_stats r ON d.provider_npi = r.provider_npi "
                f"LEFT JOIN providers p ON d.provider_npi = p.NPI "
                f"WHERE r.readmitted_patients > 0 OR d.{safe_metric}_rate >= {self.HAVING_MIN_RATE_THRESHOLD} "
                f"ORDER BY r.readmission_rate DESC, d.{safe_metric}_rate DESC "
                f"LIMIT {self.DEFAULT_BREAKDOWN_LIMIT};"
            )
        else:
            # For cost cross-metric, join claims cost with readmission
            # ARCHITECTURAL LAYER 3: Data-aware thresholds
            cost_claims_having = self._compute_adaptive_having('claims', claims_npi, min_useful=self.HAVING_MIN_USEFUL_CROSS_METRIC)
            cost_enc_having = self._compute_adaptive_having(
                'encounters', enc_npi, min_useful=10,
                where_clause=f"VISIT_TYPE = '{inpatient_val}' AND DISCHARGE_DATE != '' AND ADMIT_DATE != ''",
                count_expr='COUNT(DISTINCT MEMBER_ID)'
            )

            sql = (
                f"WITH cost_stats AS ("
                f"SELECT {claims_npi} as provider_npi, "
                f"COUNT(*) as total_claims, "
                f"ROUND(AVG(BILLED_AMOUNT), 2) as avg_cost, "
                f"ROUND(SUM(BILLED_AMOUNT), 2) as total_cost "
                f"FROM claims GROUP BY {claims_npi} HAVING COUNT(*) >= {cost_claims_having}"
                f"), "
                f"ip AS ("
                f"SELECT {enc_npi}, MEMBER_ID, ADMIT_DATE, DISCHARGE_DATE, "
                f"LAG(DISCHARGE_DATE) OVER (PARTITION BY MEMBER_ID ORDER BY ADMIT_DATE) as prev_discharge "
                f"FROM encounters "
                f"WHERE VISIT_TYPE = '{inpatient_val}' AND DISCHARGE_DATE != '' AND ADMIT_DATE != ''"
                f"), "
                f"readmit_stats AS ("
                f"SELECT {enc_npi} as provider_npi, "
                f"COUNT(DISTINCT MEMBER_ID) as total_patients, "
                f"COUNT(DISTINCT CASE WHEN julianday(ADMIT_DATE) - julianday(prev_discharge) BETWEEN 0 AND {self.READMISSION_WINDOW_DAYS} "
                f"THEN MEMBER_ID END) as readmitted_patients, "
                f"ROUND(100.0 * COUNT(DISTINCT CASE WHEN julianday(ADMIT_DATE) - julianday(prev_discharge) BETWEEN 0 AND {self.READMISSION_WINDOW_DAYS} "
                f"THEN MEMBER_ID END) / NULLIF(COUNT(DISTINCT MEMBER_ID), 0), 2) as readmission_rate "
                f"FROM ip GROUP BY {enc_npi} HAVING COUNT(DISTINCT MEMBER_ID) >= {cost_enc_having}"
                f") "
                f"SELECT c.provider_npi, "
                f"p.{provider_name} as provider_name, "
                f"c.total_claims, c.avg_cost, c.total_cost, "
                f"COALESCE(r.total_patients, 0) as total_patients, "
                f"COALESCE(r.readmitted_patients, 0) as readmitted_patients, "
                f"COALESCE(r.readmission_rate, 0) as readmission_rate "
                f"FROM cost_stats c "
                f"LEFT JOIN readmit_stats r ON c.provider_npi = r.provider_npi "
                f"LEFT JOIN providers p ON c.provider_npi = p.NPI "
                f"WHERE r.readmitted_patients > 0 OR c.avg_cost >= ("
                f"SELECT AVG(BILLED_AMOUNT) * 1.5 FROM claims) "
                f"ORDER BY r.readmission_rate DESC, c.avg_cost DESC "
                f"LIMIT {self.DEFAULT_BREAKDOWN_LIMIT};"
            )

        return {
            'sql': sql, 'tables_used': ['claims', 'encounters', 'providers'],
            'confidence': 0.85, 'intent': 'rate',
            'explanation': f'Cross-metric analysis: {cross_metric} rate vs readmission rate by provider',
        }

    def _sql_most_occurring(self, q: str, fact: str, target: str,
                            columns: List[Dict]) -> Optional[Dict]:
        """Handle 'most occurring/common X in year Y' queries."""
        # Year filter
        year = None
        ym = re.search(r'(?:in|for|during)\s+(20[12]\d)', q)
        if ym:
            year = ym.group(1)

        # Find the best column for the target via semantic scoring
        best_table = best_col = None
        best_score = 0
        for tbl, profiles in self.learner.tables.items():
            for p in profiles:
                col_name_lower = p.name.lower()
                score = 0
                if target in col_name_lower or col_name_lower in target:
                    score = 10
                elif any(target in tag.lower() or tag.lower() in target for tag in p.semantic_tags):
                    score = 8
                elif any(word in col_name_lower for word in target.split()):
                    score = 5
                if score > best_score:
                    best_score = score
                    best_table = tbl
                    best_col = p.name

        if not best_table or best_score == 0:
            return None

        # CPT codes: join with cpt_codes for descriptions
        if best_col == 'CPT_CODE' and best_table == 'claims':
            date_col = 'SERVICE_DATE'
            where = f" WHERE {date_col} LIKE '{year}%'" if year else ""
            sql = (
                f"SELECT c.CPT_CODE, cpt.DESCRIPTION, cpt.CATEGORY, COUNT(*) as occurrence_count "
                f"FROM claims c "
                f"LEFT JOIN cpt_codes cpt ON c.CPT_CODE = cpt.CPT_CODE"
                f"{where} "
                f"GROUP BY c.CPT_CODE, cpt.DESCRIPTION, cpt.CATEGORY "
                f"ORDER BY occurrence_count DESC LIMIT 10;"
            )
            return {
                'sql': sql, 'tables_used': ['claims', 'cpt_codes'],
                'confidence': 0.9, 'intent': 'ranking',
                'explanation': f'Most occurring CPT codes{" in " + year if year else ""} with descriptions',
            }

        date_col = self._find_date_column(q, columns, [best_table])
        where = f" WHERE {date_col} LIKE '{year}%'" if year and date_col else ""

        sql = (
            f"SELECT {best_col}, COUNT(*) as occurrence_count "
            f"FROM {best_table}{where} "
            f"GROUP BY {best_col} "
            f"ORDER BY occurrence_count DESC LIMIT 10;"
        )
        return {
            'sql': sql, 'tables_used': [best_table],
            'confidence': 0.85, 'intent': 'ranking',
            'explanation': f'Most occurring {target}{" in " + year if year else ""}',
        }

    def _sql_value_percentage(self, q: str, fact: str) -> Optional[Dict]:
        """Percentage by searching actual DB distinct values for keyword matches.
        Handles '% claims with pending' or 'claims with X as percentage of total'."""
        q_words = set(re.findall(r'\b[a-z]{3,}\b', q.lower()))
        stop = {'the', 'and', 'are', 'what', 'which', 'how', 'for', 'with', 'from',
                'have', 'show', 'claims', 'claim', 'total', 'volume', 'compared', 'percentage'}
        q_words -= stop

        # Check authorization pattern (special: column != empty means "has auth")
        if 'authorization' in q or 'auth' in q or 'prior auth' in q:
            if 'referrals' in self.learner.tables:
                for p in self.learner.tables.get('referrals', []):
                    if 'authorization' in p.name.lower():
                        sql = (
                            f"SELECT COUNT(*) as total, "
                            f"SUM(CASE WHEN {p.name} != '' AND {p.name} IS NOT NULL THEN 1 ELSE 0 END) as with_auth, "
                            f"ROUND(100.0 * SUM(CASE WHEN {p.name} != '' AND {p.name} IS NOT NULL THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as auth_pct "
                            f"FROM referrals;"
                        )
                        return {
                            'sql': sql, 'tables_used': ['referrals'],
                            'confidence': 0.85, 'intent': 'percentage',
                            'explanation': 'Claims with prior authorization as percentage',
                        }

        # Search DB distinct values for matching keywords
        if not q_words:
            return None

        # Abbreviation map: question word → single-char DB value
        _abbrev_map = {
            'female': 'f', 'male': 'm',
            'yes': 'y', 'no': 'n',
            'active': 'a', 'inactive': 'i',
        }
        q_abbrevs = {_abbrev_map[w]: w for w in q_words if w in _abbrev_map}

        try:
            conn = sqlite3.connect(self.learner.db_path)
            for p in self.learner.tables.get(fact, []):
                if not p.is_categorical:
                    continue
                cursor = conn.execute(
                    f"SELECT DISTINCT {p.name} FROM {fact} WHERE {p.name} IS NOT NULL LIMIT 30"
                )
                for row in cursor:
                    if row[0]:
                        val_lower = row[0].lower()
                        # Substring match for multi-char values
                        is_match = any(w in val_lower for w in q_words)
                        # Abbreviation match for single-char values
                        if not is_match and len(val_lower) <= 2 and val_lower in q_abbrevs:
                            is_match = True
                        if is_match:
                            val = row[0]
                            conn.close()
                            total = self.learner.table_row_counts.get(fact, 1)
                            sql = (
                                f"SELECT COUNT(*) as total, "
                                f"SUM(CASE WHEN {p.name} = '{val}' THEN 1 ELSE 0 END) as {val.lower().replace(' ', '_')}_count, "
                                f"ROUND(100.0 * SUM(CASE WHEN {p.name} = '{val}' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as pct "
                                f"FROM {fact};"
                            )
                            return {
                                'sql': sql, 'tables_used': [fact],
                                'confidence': 0.8, 'intent': 'percentage',
                                'explanation': f'Percentage of {fact} where {p.name} = {val}',
                            }
            conn.close()
        except Exception:
            pass
        return None

    def _sql_universal(self, q: str, fact: str, dim: Optional[Dict],
                       signals: Dict, columns: List[Dict],
                       values: List[Dict]) -> Optional[Dict]:
        """Universal SQL builder for ANY dimension + metric + filter + percentage combo.
        Replaces _compose_age_group_query, _compose_demographic_breakdown,
        _compose_age_filter_percentage, and _compose_keyword_percentage.

        This single method handles:
        - Age group breakdowns (dim.is_age_group)
        - Cross-table demographic breakdowns (dim.needs_join)
        - Same-table GROUP BY (dim in fact table)
        - Age/birth-year filter percentages (signals)
        - Keyword-based value percentages (fallback)
        """
        # --- Case 1: Age/birth-year filter percentage (no dimension needed) ---
        if (signals.get('age_filter') or signals.get('birth_year_filter')) and signals.get('percentage'):
            birth_col = self.learner.find_birth_date_column()
            if birth_col:
                birth_table = None
                for t, profiles in self.learner.tables.items():
                    if any(p.name == birth_col.name for p in profiles):
                        birth_table = t
                        break
                if signals.get('age_filter'):
                    age = signals['age_filter']
                    label = f'age > {age}'
                else:
                    year = signals['birth_year_filter']
                    label = f'birth year < {year}'

                total = self.learner.table_row_counts.get(fact, 1)

                if birth_table and birth_table == fact:
                    # Birth col in fact table — no JOIN needed
                    if signals.get('age_filter'):
                        where = f"{birth_col.name} <= date('now', '-{age} years')"
                    else:
                        where = f"CAST(SUBSTR({birth_col.name}, 1, 4) AS INTEGER) < {year}"
                    sql = (
                        f"SELECT COUNT(*) as matching, "
                        f"{total} as total, "
                        f"ROUND(100.0 * COUNT(*) / {total}, 2) as pct "
                        f"FROM {fact} WHERE {where};"
                    )
                    return {
                        'sql': sql, 'tables_used': [fact],
                        'confidence': 0.85, 'intent': 'percentage',
                        'explanation': f'Percentage where {label}',
                    }
                elif birth_table and birth_table != fact:
                    fact_cols = {p.name for p in self.learner.tables.get(fact, [])}
                    birth_cols = {p.name for p in self.learner.tables.get(birth_table, [])}
                    shared_ids = sorted(c for c in fact_cols & birth_cols if c.endswith('_ID'))
                    join_col = shared_ids[0] if shared_ids else 'MEMBER_ID'

                    if signals.get('age_filter'):
                        where = f"m.{birth_col.name} <= date('now', '-{age} years')"
                    else:
                        where = f"CAST(SUBSTR(m.{birth_col.name}, 1, 4) AS INTEGER) < {year}"

                    sql = (
                        f"SELECT COUNT(*) as matching, "
                        f"{total} as total, "
                        f"ROUND(100.0 * COUNT(*) / {total}, 2) as pct "
                        f"FROM {fact} f "
                        f"JOIN {birth_table} m ON f.{join_col} = m.{join_col} "
                        f"WHERE {where};"
                    )
                    return {
                        'sql': sql, 'tables_used': [fact, birth_table],
                        'confidence': 0.85, 'intent': 'percentage',
                        'explanation': f'Percentage where {label}',
                    }

        # --- Case 2: Age/birth-year filter WITHOUT percentage (simple filter) ---
        if (signals.get('age_filter') or signals.get('birth_year_filter')) and not dim:
            birth_col = self.learner.find_birth_date_column()
            if birth_col:
                birth_table = None
                for t, profiles in self.learner.tables.items():
                    if any(p.name == birth_col.name for p in profiles):
                        birth_table = t
                        break

                if signals.get('age_filter'):
                    age = signals['age_filter']
                    age_label = f'age > {age}'
                else:
                    year = signals['birth_year_filter']
                    age_label = f'birth year < {year}'

                if birth_table and birth_table == fact:
                    # Birth col is IN the fact table — direct WHERE, no JOIN
                    if signals.get('age_filter'):
                        where = f"{birth_col.name} <= date('now', '-{age} years')"
                    else:
                        where = f"CAST(SUBSTR({birth_col.name}, 1, 4) AS INTEGER) < {year}"
                    sql = (
                        f"SELECT * FROM {fact} WHERE {where} LIMIT 200;"
                    )
                    return {
                        'sql': sql, 'tables_used': [fact],
                        'confidence': 0.85, 'intent': 'filter',
                        'explanation': f'Filtered records where {age_label}',
                    }
                elif birth_table and birth_table != fact:
                    fact_cols = {p.name for p in self.learner.tables.get(fact, [])}
                    birth_cols = {p.name for p in self.learner.tables.get(birth_table, [])}
                    shared_ids = sorted(c for c in fact_cols & birth_cols if c.endswith('_ID'))
                    join_col = shared_ids[0] if shared_ids else 'MEMBER_ID'

                    if signals.get('age_filter'):
                        where = f"m.{birth_col.name} <= date('now', '-{age} years')"
                    else:
                        where = f"CAST(SUBSTR(m.{birth_col.name}, 1, 4) AS INTEGER) < {year}"

                    sql = (
                        f"SELECT COUNT(*) as count "
                        f"FROM {fact} f "
                        f"JOIN {birth_table} m ON f.{join_col} = m.{join_col} "
                        f"WHERE {where};"
                    )
                    return {
                        'sql': sql, 'tables_used': [fact, birth_table],
                        'confidence': 0.85, 'intent': 'filter',
                        'explanation': f'Filtered count where {age_label}',
                    }

        # --- Case 3: Percentage with no dimension — keyword value search ---
        if signals.get('percentage') and not dim:
            result = self._sql_value_percentage(q, fact)
            if result:
                return result

        # --- Case 4: Dimension-based queries ---
        if not dim:
            return None

        total = self.learner.table_row_counts.get(fact, 1)

        # 4a: Age group dimension (computed CASE WHEN)
        if dim.get('is_age_group'):
            # Check for custom age interval in question (e.g., "by 5 years")
            custom_interval = self._extract_age_interval(q)
            if dim['needs_join']:
                join_col = dim['join_col']
                age_expr = self._age_group_case(f"m.{dim['column']}", interval=custom_interval)
                sql = (
                    f"SELECT {age_expr} as age_group, "
                    f"COUNT(*) as count, "
                    f"ROUND(100.0 * COUNT(*) / {total}, 2) as pct "
                    f"FROM {fact} f "
                    f"JOIN {dim['table']} m ON f.{join_col} = m.{join_col} "
                    f"GROUP BY age_group ORDER BY age_group;"
                )
                tables_used = [fact, dim['table']]
            else:
                age_expr = self._age_group_case(dim['column'], interval=custom_interval)
                sql = (
                    f"SELECT {age_expr} as age_group, "
                    f"COUNT(*) as count, "
                    f"ROUND(100.0 * COUNT(*) / {total}, 2) as pct "
                    f"FROM {fact} "
                    f"GROUP BY age_group ORDER BY age_group;"
                )
                tables_used = [fact]
            return {
                'sql': sql, 'tables_used': tables_used,
                'confidence': 0.9, 'intent': 'breakdown',
                'explanation': f'Age group breakdown with computed age buckets from {dim["column"]}',
            }

        # 4b: Cross-table dimension (needs JOIN)
        if dim['needs_join']:
            on_clause = self._dim_on_clause(dim, 'f', 'd')

            # Detect if we need an amount column — search across concept-expanded tables
            search_tables = self._expand_tables_for_concepts(q, [fact], columns)
            num_col = self._find_numeric_column(q, columns, search_tables)
            has_amount = self._has_amount_word(q) and num_col
            agg_func = self._detect_agg_func(q)

            # Determine which table the numeric column belongs to
            num_table = fact
            if num_col:
                for t in search_tables:
                    for p in self.learner.tables.get(t, []):
                        if p.name == num_col and t != fact:
                            num_table = t
                            break

            # Build extra JOINs if the metric comes from a third table
            extra_join = ""
            extra_tables = []
            if num_table != fact and num_table != dim['table']:
                nj = self.kg._get_join_condition(fact, num_table)
                if '=' in nj:
                    nleft, nright = nj.split('=', 1)
                    extra_on = f"f.{nleft} = n.{nright}"
                else:
                    extra_on = f"f.{nj} = n.{nj}"
                extra_join = f"JOIN {num_table} n ON {extra_on} "
                extra_tables = [num_table]
                num_col_ref = f"n.{num_col}"
            elif num_table == fact:
                num_col_ref = f"f.{num_col}" if num_col else None
            else:
                num_col_ref = f"d.{num_col}" if num_col else None

            if has_amount:
                sql = (
                    f"SELECT d.{dim['column']}, "
                    f"COUNT(*) as count, "
                    f"ROUND({agg_func}(CAST({num_col_ref} AS REAL)), 2) as {agg_func.lower()}_{num_col.lower()}, "
                    f"ROUND(100.0 * COUNT(*) / {total}, 2) as pct "
                    f"FROM {fact} f "
                    f"JOIN {dim['table']} d ON {on_clause} "
                    f"{extra_join}"
                    f"GROUP BY d.{dim['column']} "
                    f"ORDER BY {agg_func.lower()}_{num_col.lower()} DESC LIMIT {self.DEFAULT_BREAKDOWN_LIMIT};"
                )
            else:
                sql = (
                    f"SELECT d.{dim['column']}, "
                    f"COUNT(*) as count, "
                    f"ROUND(100.0 * COUNT(*) / {total}, 2) as pct "
                    f"FROM {fact} f "
                    f"JOIN {dim['table']} d ON {on_clause} "
                    f"GROUP BY d.{dim['column']} "
                    f"ORDER BY count DESC LIMIT {self.DEFAULT_BREAKDOWN_LIMIT};"
                )
            return {
                'sql': sql, 'tables_used': [fact, dim['table']] + extra_tables,
                'confidence': 0.85, 'intent': 'breakdown',
                'explanation': f'Cross-table breakdown: {fact} grouped by {dim["table"]}.{dim["column"]}',
            }

        # 4c-rate: If this is a rate query (denial rate, fill rate, etc.)
        # with a dimension, produce CASE WHEN rate SQL instead of plain COUNT
        rate_info = self._detect_rate_info(q, columns, [fact])
        if rate_info:
            status_col, target_val, _ = rate_info  # Ignore rate_info's group_col; use dim
            safe_val = re.sub(r'[^a-zA-Z0-9_]', '_', target_val.lower())
            if dim['needs_join']:
                on_clause = self._dim_on_clause(dim, 'f', 'd')
                sql = (
                    f"SELECT d.{dim['column']}, "
                    f"COUNT(*) as total, "
                    f"SUM(CASE WHEN f.{status_col} = '{target_val}' THEN 1 ELSE 0 END) as {safe_val}_count, "
                    f"ROUND(100.0 * SUM(CASE WHEN f.{status_col} = '{target_val}' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as {safe_val}_rate "
                    f"FROM {fact} f "
                    f"JOIN {dim['table']} d ON {on_clause} "
                    f"GROUP BY d.{dim['column']} "
                    f"ORDER BY {safe_val}_rate DESC LIMIT {self.DEFAULT_BREAKDOWN_LIMIT};"
                )
                tables_used = [fact, dim['table']]
            else:
                sql = (
                    f"SELECT {dim['column']}, "
                    f"COUNT(*) as total, "
                    f"SUM(CASE WHEN {status_col} = '{target_val}' THEN 1 ELSE 0 END) as {safe_val}_count, "
                    f"ROUND(100.0 * SUM(CASE WHEN {status_col} = '{target_val}' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as {safe_val}_rate "
                    f"FROM {fact} "
                    f"GROUP BY {dim['column']} "
                    f"ORDER BY {safe_val}_rate DESC LIMIT {self.DEFAULT_BREAKDOWN_LIMIT};"
                )
                tables_used = [fact]
            return {
                'sql': sql, 'tables_used': tables_used,
                'confidence': 0.9, 'intent': 'rate',
                'explanation': f'{target_val} rate by {dim["column"]}',
            }

        # 4c: Same-table dimension — also check if metric is in another table
        search_tables = self._expand_tables_for_concepts(q, [fact], columns)
        num_col = self._find_numeric_column(q, columns, search_tables)
        has_amount = self._has_amount_word(q) and num_col
        agg_func = self._detect_agg_func(q)

        # Check if num_col lives in a different table
        num_table = fact
        if num_col:
            for t in search_tables:
                if t == fact:
                    continue
                for p in self.learner.tables.get(t, []):
                    if p.name == num_col:
                        num_table = t
                        break

        if has_amount and num_table != fact:
            # Need a JOIN to get the metric from another table
            nj = self.kg._get_join_condition(fact, num_table)
            if '=' in nj:
                nleft, nright = nj.split('=', 1)
                on_clause = f"f.{nleft} = n.{nright}"
            else:
                on_clause = f"f.{nj} = n.{nj}"
            sql = (
                f"SELECT f.{dim['column']}, "
                f"COUNT(*) as count, "
                f"ROUND({agg_func}(CAST(n.{num_col} AS REAL)), 2) as {agg_func.lower()}_{num_col.lower()}, "
                f"ROUND(100.0 * COUNT(*) / {total}, 2) as pct "
                f"FROM {fact} f "
                f"JOIN {num_table} n ON {on_clause} "
                f"GROUP BY f.{dim['column']} "
                f"ORDER BY {agg_func.lower()}_{num_col.lower()} DESC LIMIT {self.DEFAULT_BREAKDOWN_LIMIT};"
            )
            return {
                'sql': sql, 'tables_used': [fact, num_table],
                'confidence': 0.85, 'intent': 'breakdown',
                'explanation': f'Breakdown: {fact} grouped by {dim["column"]} with metric from {num_table}',
            }
        elif has_amount:
            sql = (
                f"SELECT {dim['column']}, "
                f"COUNT(*) as count, "
                f"ROUND({agg_func}(CAST({num_col} AS REAL)), 2) as {agg_func.lower()}_{num_col.lower()}, "
                f"ROUND(100.0 * COUNT(*) / {total}, 2) as pct "
                f"FROM {fact} "
                f"GROUP BY {dim['column']} "
                f"ORDER BY {agg_func.lower()}_{num_col.lower()} DESC LIMIT {self.DEFAULT_BREAKDOWN_LIMIT};"
            )
        else:
            sql = (
                f"SELECT {dim['column']}, "
                f"COUNT(*) as count, "
                f"ROUND(100.0 * COUNT(*) / {total}, 2) as pct "
                f"FROM {fact} "
                f"GROUP BY {dim['column']} "
                f"ORDER BY count DESC LIMIT {self.DEFAULT_BREAKDOWN_LIMIT};"
            )
        return {
            'sql': sql, 'tables_used': [fact] if num_table == fact else [fact, num_table],
            'confidence': 0.85, 'intent': 'breakdown',
            'explanation': f'Breakdown: {fact} grouped by {dim["column"]}',
        }

    # --- Helper methods ---

    def _extract_group_term(self, q: str) -> Optional[str]:
        """Extract the GROUP BY term from natural language.

        Handles subtle patterns:
        - "by provider" → group by provider  (YES)
        - "per region"  → group by region    (YES)
        - "average cost per encounter" → NOT a group term (rate descriptor)
        - "cost per visit over time"   → NOT a group term (rate descriptor)
        """
        # "per X" after an aggregate word is a rate descriptor, not grouping
        # e.g. "average cost per encounter" → per encounter is the unit, not group dim
        AGG_RATE_PATTERN = (
            r'\b(?:average|avg|mean|total|sum|median|cost|revenue|amount|charge|'
            r'paid|billed|allowed)\s+\w*\s*per\b'
        )
        has_agg_per = bool(re.search(AGG_RATE_PATTERN, q))

        patterns = [
            ('by', r'\bby\s+([\w\s]+?)(?:\s+(?:for|in|from|where|order|having|limit|with|to)|[?.!;,]?\s*$)'),
            ('per', r'\bper\s+([\w\s]+?)(?:\s+(?:for|in|from|where|order)|[?.!;,]?\s*$)'),
            ('each', r'\b(?:for|in) each\s+([\w\s]+?)(?:\s+(?:for|in|from|where)|[?.!;,]?\s*$)'),
            ('group', r'\bgrouped? by\s+([\w\s]+?)(?:[?.!;,]?\s*$)'),
            ('across', r'\bacross\s+([\w\s]+?)(?:\s+(?:for|in|from|where)|[?.!;,]?\s*$)'),
        ]
        for label, pat in patterns:
            # Skip "per X" when it's a rate descriptor (avg cost per encounter)
            if label == 'per' and has_agg_per:
                continue
            m = re.search(pat, q)
            if m:
                raw = m.group(1).strip()
                # Remove noise words
                noise = {'the', 'a', 'an', 'each', 'every', 'all', 'their', 'its',
                          'or', 'on', 'of', 'front', 'side', 'basis', 'end',
                          'changed', 'changes', 'over', 'time', 'trend', 'trended'}
                words = [w for w in raw.split() if w.lower() not in noise and len(w) > 1]
                return ' '.join(words) if words else None
        return None

    def _extract_ranking_entity(self, q: str) -> Optional[str]:
        """Extract entity being ranked from question."""
        m = re.search(r'\btop\s+\d*\s*(\w+)', q)
        if m:
            word = m.group(1)
            if word.lower() not in ('total', 'average', 'count', 'sum', 'amount'):
                return word
        m = re.search(r'\bwhich\s+([\w\s]+?)\s+(?:has|have|is|are)\s+', q)
        if m:
            return m.group(1).strip().split()[-1]
        return None

    def _resolve_group_column(self, term: Optional[str], columns: List[Dict],
                              tables: List[str]) -> Optional[str]:
        """Resolve a natural language group term to an actual column using TF-IDF matching."""
        if not term:
            return None

        term_lower = term.lower().strip()

        # ── Column Graph consultation — use structural intelligence first ──
        column_graph = getattr(self.learner, 'column_graph', None)
        if column_graph:
            for table in tables:
                display_pairs = column_graph.get_display_columns(table)
                for code_col, label_col in display_pairs:
                    if (term_lower in code_col.lower() or
                        term_lower in label_col.lower()):
                        return label_col
                name_pair = column_graph.get_name_columns(table)
                if name_pair and term_lower in ('member', 'patient', 'provider',
                                                  'person', 'name'):
                    return name_pair[0]  # FIRST_NAME — compose() concatenates

        # Zero: If the term matches a TABLE name (e.g., "providers", "members"),
        # the user wants to group by that entity. Find a suitable identifier or
        # label column from that entity table (or a FK in the fact table).
        term_stem = term_lower.rstrip('s') if len(term_lower) > 2 else term_lower
        for tbl_name, profiles in self.learner.tables.items():
            tbl_lower = tbl_name.lower()
            tbl_stem = tbl_lower.rstrip('s') if len(tbl_lower) > 2 else tbl_lower
            if tbl_stem == term_stem or tbl_lower == term_lower:
                # Found matching table — find a label column (name, specialty, etc.)
                # Priority: name_col > description_col > label_col > id_col
                # Exclude numeric columns to avoid returning tenure_months, age, etc.
                name_col = None
                desc_col = None
                label_col = None
                id_col = None
                for p in profiles:
                    pn = p.name.lower()
                    if p.is_id and p.name.endswith('_ID'):
                        if not id_col:
                            id_col = p.name
                    elif hasattr(p, 'is_numeric') and p.is_numeric:
                        continue  # skip numeric columns like TENURE_MONTHS, AGE
                    elif not p.is_id and not p.name.endswith('_ID'):
                        # Highest priority: NAME columns
                        if any(w in pn for w in ('name', 'first_name', 'last_name', 'fname', 'lname')):
                            if not name_col:
                                name_col = p.name
                        # Second priority: DESCRIPTION columns
                        elif any(w in pn for w in ('description', 'desc', 'text', 'detail')):
                            if not desc_col:
                                desc_col = p.name
                        # Third priority: other label-like columns (but NOT 'status')
                        elif (p.is_categorical and any(w in pn for w in
                                  ('specialty', 'type', 'title', 'label',
                                   'department', 'category', 'icd', 'dx', 'code'))):
                            if not label_col:
                                label_col = p.name
                # Priority chain: name > description > label > id
                label_col = name_col or desc_col or label_col
                # Prefer a label column, fall back to ID
                best_col = label_col or id_col
                if best_col:
                    # Check if this column exists in the current tables
                    for t in tables:
                        for p in self.learner.tables.get(t, []):
                            if p.name == best_col:
                                return best_col
                    # If not in current tables but entity table is joinable, return it
                    # (the join will be handled by cross-table group resolution in compose())
                    return best_col
                break

        # First: try EXACT column name match (underscore-normalized)
        tn = term_lower.replace(' ', '_')
        for table in tables:
            for p in self.learner.tables.get(table, []):
                cn = p.name.lower()
                if cn == tn or cn == tn + '_id':
                    return p.name
                if tn.replace('_', '') == cn.replace('_', ''):
                    return p.name

        # Second: try table-qualified match — "type" for claims → CLAIM_TYPE
        # (prefer columns that contain the table name + term)
        for table in tables:
            table_stem = table.rstrip('s').lower()
            qualified = f"{table_stem}_{tn}"
            for p in self.learner.tables.get(table, []):
                cn = p.name.lower()
                if cn == qualified or cn == qualified.replace('_', ''):
                    return p.name

        # Third: use TF-IDF semantic search on the group term specifically
        if self.learner:
            # Search the column index with the group term
            group_results = []
            for table in tables:
                for p in self.learner.tables.get(table, []):
                    cn = p.name.lower()
                    # Score: how well does this column name match the term?
                    score = 0.0
                    for word in term_lower.split():
                        if len(word) >= 3:
                            if word in cn:
                                score += 2.0
                            elif word in cn.replace('_', ' '):
                                score += 1.5
                            # Check semantic tags
                            for tag in p.semantic_tags:
                                if word in tag:
                                    score += 0.5
                    if score > 0 and (p.is_categorical or p.is_id):
                        group_results.append((p.name, table, score))

            if group_results:
                group_results.sort(key=lambda x: x[2], reverse=True)
                return group_results[0][0]

        # Fourth: use the semantic column matches from main query
        for c in columns:
            if c['table'] in tables:
                cn = c['column'].lower()
                if term_lower in cn or cn in term_lower.replace(' ', '_'):
                    if c.get('meta', {}).get('is_categorical') or c.get('meta', {}).get('is_id'):
                        return c['column']

        # Fifth: common semantic mappings derived from column semantic tags
        for table in tables:
            for p in self.learner.tables.get(table, []):
                cn = p.name.lower()
                for word in term_lower.split():
                    if len(word) >= 3 and word in cn:
                        if p.is_categorical or p.is_id:
                            return p.name

        # Sixth: search ALL tables (not just selected ones) — enables cross-table JOINs
        for table, profiles in self.learner.tables.items():
            if table in tables:
                continue  # Already searched above
            # Exact match
            for p in profiles:
                cn = p.name.lower()
                if cn == tn or cn == tn + '_id':
                    return p.name
            # Partial match
            for p in profiles:
                cn = p.name.lower()
                for word in term_lower.split():
                    if len(word) >= 3 and word in cn:
                        if p.is_categorical or p.is_id or not p.is_numeric:
                            return p.name

        return None

    def _has_amount_word(self, q: str) -> bool:
        """Check if the question mentions monetary/numeric amounts.
        Uses DomainConfig when available, then ALWAYS falls through to
        schema-scan + aggregate triggers for full coverage."""
        q_words = set(re.findall(r'[a-z]+', q.lower()))

        # Strategy 1: DomainConfig concept matching
        if self.domain_config:
            for w in q_words:
                if self.domain_config.is_amount_word(w):
                    return True
                # Also check singular/plural forms
                w_stem = w.rstrip('s') if len(w) > 3 else w
                if w_stem != w and self.domain_config.is_amount_word(w_stem):
                    return True

        # Strategy 2 (always runs): check question words overlap with
        # numeric column name parts + common aggregate trigger words
        numeric_words = set()
        for profiles in self.learner.tables.values():
            for p in profiles:
                if p.is_numeric and not p.is_id:
                    numeric_words.update(p.name.lower().replace('_', ' ').split())
        numeric_words -= {'id', 'key', 'num', 'code'}
        agg_triggers = {'total', 'sum', 'average', 'avg', 'mean',
                        'spending', 'earnings', 'gross', 'profit'}
        return bool(q_words & (numeric_words | agg_triggers))

    def _detect_agg_func(self, q: str) -> str:
        """Detect aggregation function from question words."""
        if any(w in q for w in ['average', 'avg', 'mean']):
            return 'AVG'
        if any(w in q for w in ['total', 'sum', 'combined', 'revenue', 'sales',
                                    'spending', 'income', 'earnings', 'gross']):
            # "total number/count" → COUNT, everything else → SUM
            if any(w in q for w in ['number', 'count']):
                return 'COUNT'
            return 'SUM'
        if any(w in q for w in ['maximum', 'max', 'highest', 'largest', 'biggest']):
            return 'MAX'
        if any(w in q for w in ['minimum', 'min', 'lowest', 'smallest']):
            return 'MIN'
        return 'COUNT'

    def _find_numeric_column(self, q: str, columns: List[Dict],
                             tables: List[str]) -> Optional[str]:
        """Find the most relevant numeric column using semantic matching.
        Data-driven: matches question words against actual column names from schema,
        then uses TF-IDF-matched columns, then falls back to currency-tagged columns."""
        q_words = set(re.findall(r'[a-z]+', q.lower()))

        # Strategy 1: match question words against actual column names in our tables
        # Score each numeric column by word overlap with question (EXCLUDE IDs unless explicitly named)
        candidates = []
        for table in tables:
            for p in self.learner.tables.get(table, []):
                if not p.is_numeric:
                    continue
                # Skip FK columns (ending in _id) — these are never metrics
                col_name_lower = p.name.lower()
                if col_name_lower.endswith('_id'):
                    continue
                # For other ID-flagged columns, only skip if the column name
                # isn't directly referenced as a word in the question
                if p.is_id:
                    if col_name_lower not in q_words:
                        continue
                col_parts = set(p.name.lower().replace('_', ' ').split())
                # Remove generic parts that match too broadly
                col_parts -= {'id', 'key', 'code', 'num'}
                overlap = q_words & col_parts
                if overlap:
                    # Weight: more overlap = better, currency tag = bonus
                    score = len(overlap) * 2.0
                    if 'currency' in p.semantic_tags:
                        score += 1.0
                    candidates.append((p.name, score))
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]

        # Strategy 2: semantically matched numeric columns with currency tag (skip IDs)
        for c in columns:
            if c['table'] in tables and c.get('meta', {}).get('is_numeric'):
                if c.get('meta', {}).get('is_id') or c['column'].lower().endswith('_id'):
                    continue
                if 'currency' in c.get('meta', {}).get('semantic_tags', []):
                    return c['column']

        # Strategy 3: any semantically matched numeric column (skip IDs)
        for c in columns:
            if c['table'] in tables and c.get('meta', {}).get('is_numeric'):
                if c.get('meta', {}).get('is_id') or c['column'].lower().endswith('_id'):
                    continue
                return c['column']

        # Strategy 4: first currency-tagged numeric column in primary table
        for table in tables:
            for p in self.learner.tables.get(table, []):
                if p.is_numeric and 'currency' in p.semantic_tags:
                    return p.name

        # Strategy 5: first numeric column that isn't an ID
        for table in tables:
            for p in self.learner.tables.get(table, []):
                if p.is_numeric and not p.is_id and not p.name.lower().endswith('_id'):
                    return p.name

        return None

    def _find_date_column(self, q: str, columns: List[Dict],
                          tables: List[str]) -> Optional[str]:
        """Find the most relevant date column from learned schema.
        Uses DomainConfig relevance-scored date columns when available;
        falls back to SchemaLearner date columns with word-overlap scoring."""

        # --- Strategy 1: DomainConfig (pre-scored by relevance) ---
        if self.domain_config:
            for table in tables:
                date_cols = self.domain_config.find_date_columns(table)
                if date_cols:
                    # DomainConfig returns them sorted by relevance score already
                    return date_cols[0].name

        else:
            # --- Strategy 2: Score date columns by word overlap with question ---
            q_words = set(re.findall(r'[a-z]+', q.lower()))
            best_col = None
            best_score = -1
            for table in tables:
                dates = self.learner.get_date_columns(table)
                for d in dates:
                    col_words = set(d.name.lower().replace('_', ' ').split())
                    col_words -= {'date', 'dt', 'ts', 'timestamp'}  # Generic parts
                    overlap = len(col_words & q_words)
                    # Business-event dates score higher than audit dates
                    audit_penalty = -2 if any(w in d.name.lower() for w in ['created', 'updated', 'modified']) else 0
                    score = overlap + audit_penalty
                    if score > best_score:
                        best_score = score
                        best_col = d.name
            if best_col:
                return best_col

        # Strategy 3: check semantically matched date columns
        for c in columns:
            if c['table'] in tables and c.get('meta', {}).get('is_date'):
                return c['column']

        # Strategy 4: first date column in primary table
        for table in tables:
            dates = self.learner.get_date_columns(table)
            if dates:
                return dates[0].name

        return None

    def _detect_rate_info(self, q: str, columns: List[Dict],
                          tables: List[str]) -> Optional[Tuple[str, str, Optional[str]]]:
        """Detect rate query components from question — DATA-DRIVEN.
        Finds status/categorical columns and matches rate target value
        from actual data profiles AND from actual DISTINCT values in the DB
        (sample_values can be unrepresentative for rare values like DENIED)."""
        # Extract the rate target word from the question
        rate_m = re.search(r'(\w+)\s+rate', q)
        if not rate_m:
            rate_m = re.search(r'rate\s+(?:of|for)\s+(\w+)', q)
        if not rate_m:
            return None

        rate_word = rate_m.group(1).lower()

        # Build stems: "denial" → ["denial", "deni", "deny", "denied"]
        stems = {rate_word}
        if rate_word.endswith('al'):
            stems.add(rate_word[:-2])
            stems.add(rate_word[:-2] + 'ed')
            stems.add(rate_word[:-2] + 'y')  # denial → deny
        if rate_word.endswith('tion'):
            stems.add(rate_word[:-4])
        stems.add(rate_word + 'ed')
        stems.add(rate_word + 'led')
        stems = {s for s in stems if len(s) >= 3}

        def _match_value(val):
            """Check if a value matches any stem."""
            vl = val.lower().replace('_', ' ')
            return any(stem in vl or vl.startswith(stem) for stem in stems)

        # Search status columns for the rate target value.
        # For each table: first check sample_values, then query DB for actual distinct values.
        # This per-table approach ensures larger tables (claims) are fully checked
        # before smaller tables (referrals), preventing biased sample_values from
        # causing misrouting (e.g., claims.CLAIM_STATUS samples are all 'ADJUSTED'
        # but actual DISTINCT values include 'DENIED').
        import sqlite3 as _sq3
        db_path = self.learner.db_path if hasattr(self.learner, 'db_path') else None
        _conn = None
        try:
            if db_path:
                _conn = _sq3.connect(db_path)
        except Exception:
            pass

        for table in tables:
            for p in self.learner.tables.get(table, []):
                if not p.is_categorical:
                    continue
                is_status_col = 'status' in p.name.lower()
                if not is_status_col:
                    continue

                # Check 1: sample_values (fast)
                if p.sample_values:
                    unique_vals = list(set(p.sample_values))
                    for val in unique_vals:
                        if _match_value(val):
                            group_col = None
                            group_term = self._extract_group_term(q)
                            if group_term:
                                group_col = self._resolve_group_column(group_term, columns, tables)
                            if _conn:
                                _conn.close()
                            return (p.name, val, group_col)

                # Check 2: actual DB distinct values (catches underrepresented values)
                if _conn:
                    try:
                        cursor = _conn.execute(
                            f"SELECT DISTINCT {p.name} FROM {table} WHERE {p.name} IS NOT NULL LIMIT 20"
                        )
                        distinct_vals = [r[0] for r in cursor.fetchall() if r[0]]
                        for val in distinct_vals:
                            if _match_value(val):
                                group_col = None
                                group_term = self._extract_group_term(q)
                                if group_term:
                                    group_col = self._resolve_group_column(group_term, columns, tables)
                                _conn.close()
                                return (p.name, val, group_col)
                    except Exception:
                        pass

        if _conn:
            _conn.close()

        # Pass 3: any categorical column (not just status) from sample values
        for table in tables:
            for p in self.learner.tables.get(table, []):
                if p.is_categorical and p.sample_values:
                    for val in set(p.sample_values):
                        if _match_value(val):
                            group_col = None
                            group_term = self._extract_group_term(q)
                            if group_term:
                                group_col = self._resolve_group_column(group_term, columns, tables)
                            return (p.name, val, group_col)

        return None

    def _default_select(self, columns: List[Dict], tables: List[str]) -> List[str]:
        """Build a default SELECT using the most important columns from PageRank."""
        if not tables:
            return ['*']

        primary = tables[0]
        important = self.kg.get_important_columns(primary, top_k=8)

        if important:
            return [col for col, _ in important]

        # Fallback: use semantic column matches
        selected = []
        seen = set()
        for c in columns:
            if c['table'] == primary and c['column'] not in seen:
                selected.append(c['column'])
                seen.add(c['column'])
            if len(selected) >= 8:
                break

        return selected if selected else ['*']

    # Stop-words excluded from concept coverage (common English words that
    # don't carry domain meaning and should not penalize confidence).
    _CONFIDENCE_STOP_WORDS = frozenset({
        # Common English
        'what', 'is', 'the', 'a', 'an', 'of', 'in', 'for', 'to', 'and',
        'or', 'by', 'on', 'at', 'from', 'with', 'my', 'our', 'me', 'us',
        'show', 'tell', 'give', 'get', 'find', 'list', 'display', 'see',
        'can', 'do', 'does', 'did', 'how', 'about', 'this', 'that',
        'are', 'were', 'was', 'be', 'been', 'have', 'has', 'had',
        'all', 'each', 'every', 'some', 'any', 'no', 'not',
        'it', 'its', 'i', 'we', 'they', 'he', 'she', 'their',
        'please', 'just', 'also', 'very', 'much', 'many',
        # Temporal / analytical modifiers (structural, not domain content)
        'trend', 'over', 'last', 'month', 'months', 'year', 'years',
        'quarter', 'quarterly', 'weekly', 'daily', 'monthly', 'yearly',
        'time', 'period', 'date', 'during', 'since', 'between',
        # Analytical verbs/modifiers
        'average', 'total', 'count', 'sum', 'number', 'rate',
        'top', 'highest', 'lowest', 'most', 'least', 'best', 'worst',
        'per', 'compare', 'versus', 'breakdown',
        'which', 'where', 'there',
    })

    def _calculate_confidence(self, intent: Dict, columns: List[Dict],
                              tables: List[Dict], values: List[Dict],
                              conditions: List[str],
                              question: str = '') -> float:
        """Calculate confidence score for the generated SQL.

        Includes a concept-coverage penalty: if key words in the question
        have zero hits in the concept index AND zero column matches, the
        system is probably generating garbage SQL. Penalize to trigger
        fallback or low-confidence warnings.
        """
        score = 0.3  # Base

        # Intent confidence
        intent_conf = intent.get('confidence', 0)
        score += min(0.2, intent_conf * 0.2)

        # Column match quality
        if columns:
            avg_col_score = sum(c['score'] for c in columns[:5]) / min(len(columns), 5)
            score += min(0.2, avg_col_score * 0.3)

        # Table match quality
        if tables:
            avg_table_score = sum(t['score'] for t in tables[:3]) / min(len(tables), 3)
            score += min(0.15, avg_table_score * 0.2)

        # Value matches add confidence
        if values:
            score += min(0.1, len(values) * 0.03)

        # Conditions add confidence
        if conditions:
            score += min(0.05, len(conditions) * 0.02)

        # ── Concept coverage penalty ──────────────────────────────────────
        # If the question contains content words with ZERO concept index hits
        # and ZERO column matches, the system doesn't understand the query.
        # Penalize so we trigger fallback or low-confidence path.
        if question and hasattr(self, '_concept_table_index'):
            q_words = set(re.findall(r'[a-z]+', question.lower()))
            content_words = q_words - self._CONFIDENCE_STOP_WORDS
            if content_words:
                # Check which content words are resolved in concept index or columns
                resolved = set()
                for w in content_words:
                    # Resolved via concept index
                    if w in self._concept_table_index and self._concept_table_index[w]:
                        resolved.add(w)
                        continue
                    # Resolved via column matches (column name contains the word)
                    for c in (columns or []):
                        if w in c.get('column', '').lower() or w in c.get('table', '').lower():
                            resolved.add(w)
                            break
                unresolved = content_words - resolved
                coverage = len(resolved) / len(content_words) if content_words else 1.0
                # If less than 50% of content words are resolved, heavy penalty
                if coverage < 0.5:
                    penalty = 0.25  # Drops below MIN_SEMANTIC_CONFIDENCE (0.35)
                    score -= penalty
                    logger.info("Concept coverage penalty: %.2f (resolved %d/%d words: %s | unresolved: %s)",
                                penalty, len(resolved), len(content_words),
                                sorted(resolved), sorted(unresolved))
                elif coverage < 0.75:
                    penalty = 0.10
                    score -= penalty
                    logger.info("Concept coverage warning: resolved %d/%d words (unresolved: %s)",
                                len(resolved), len(content_words), sorted(unresolved))

        return max(0.0, min(1.0, score))

    def _explain(self, question: str, intent_type: str, tables: List[str],
                 columns: List[Dict], conditions: List[str],
                 select_parts: List[str]) -> str:
        """Generate human-readable explanation."""
        INTENT_DESC = {
            'count': 'counting records', 'aggregate': 'computing aggregates',
            'ranking': 'ranking results', 'breakdown': 'breaking down by category',
            'lookup': 'looking up records', 'trend': 'analyzing trends over time',
            'comparison': 'comparing values', 'percentage': 'calculating percentages',
            'rate': 'computing rate metrics', 'filter': 'filtering records',
        }
        action = INTENT_DESC.get(intent_type, 'querying data')

        lines = [f"<b>Intent:</b> {action} (classified by TF-IDF centroid matching)"]

        if tables:
            lines.append(f"<b>Tables:</b> {', '.join(tables)} (matched by cosine similarity)")

        if columns:
            top_matches = columns[:5]
            col_strs = [f"{c['table']}.{c['column']} ({c['score']:.2f})" for c in top_matches]
            lines.append(f"<b>Columns:</b> {'; '.join(col_strs)} (TF-IDF ranked)")

        if conditions:
            lines.append(f"<b>Filters:</b> {len(conditions)} condition(s) from semantic value matching")

        return "<br>".join(lines)


# PART 3: SEMANTIC SQL ENGINE

class SemanticSQLEngine:
    """
    Public API: SemanticSQLEngine(db_path).generate(question) → dict

    This is the intelligence-driven replacement for DynamicSQLEngine.
    It uses:
    - TF-IDF + cosine similarity for column/table/value resolution
    - Statistical intent classification via centroid nearest-neighbor
    - Knowledge graph + PageRank for join reasoning
    - Schema auto-discovery from actual database introspection
    - Computed column inference from structural patterns
    - Data-aware validation against real data profiles

    Falls back to DynamicSQLEngine when semantic confidence is low.

    Returns: {sql, tables_used, confidence, intent, explanation,
              semantic_columns, semantic_values, data_warnings}
    """

    # Minimum semantic confidence to use semantic SQL instead of falling back
    MIN_SEMANTIC_CONFIDENCE = 0.35

    def __init__(self, db_path: str, catalog_dir: str = None, nlp_mode: str = 'auto'):
        """
        Initialize the SemanticSQLEngine.

        Args:
            db_path: Path to SQLite database
            catalog_dir: Optional catalog directory for fallback engine
            nlp_mode: NLP backend mode — 'auto', 'scratch', 'library', 'tier2', 'tier3'
                      'auto' detects best available HIPAA-compliant libraries
                      'scratch' forces pure Python stdlib only
        """
        self.db_path = db_path
        self.catalog_dir = catalog_dir
        self.nlp_mode = nlp_mode

        # Initialize NLP engine factory for switchable backends
        self.nlp_factory = None
        try:
            from nlp_engine_factory import NLPEngineFactory
            self.nlp_factory = NLPEngineFactory(mode=nlp_mode)
            logger.info("NLP factory: mode=%s, tier=%d",
                        nlp_mode, self.nlp_factory.inventory.best_tier)
        except Exception as e:
            logger.warning("NLP factory unavailable (%s), using scratch backends", e)

        # Initialize semantic layer with factory for backend switching
        self.semantic = SemanticLayer(db_path, nlp_factory=self.nlp_factory)
        self.semantic.initialize()

        # Build knowledge graph from learned schema
        self.kg = SchemaKnowledgeGraph(self.semantic.learner)

        # Initialize schema-driven domain config (replaces all hardcoded maps)
        self.domain_config = None
        try:
            from domain_config import DomainConfig
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'config', 'domain_config.json'
            )
            self.domain_config = DomainConfig(
                self.semantic.learner,
                config_path=config_path if os.path.exists(config_path) else None
            )
        except Exception as e:
            logger.warning("DomainConfig unavailable (%s), using fallback logic", e)

        # Create semantic composer
        self.composer = SemanticSQLComposer(
            self.semantic.learner, self.kg, self.semantic.inferrer,
            domain_config=self.domain_config
        )

        # Initialize fallback engine (the old hardcoded one)
        self._fallback = None
        try:
            from dynamic_sql_engine import DynamicSQLEngine
            self._fallback = DynamicSQLEngine(catalog_dir, db_path)
        except Exception:
            pass

        logger.info("SemanticSQLEngine initialized — schema-agnostic, data-driven, nlp=%s",
                     nlp_mode)

    def generate(self, question: str) -> Dict[str, Any]:
        """
        Generate SQL from natural language using semantic intelligence.

        Pipeline:
        1. Classify intent (TF-IDF centroid nearest-neighbor)
        2. Match columns (TF-IDF cosine similarity on schema descriptions)
        3. Match tables (TF-IDF cosine similarity on table descriptions)
        4. Match values (TF-IDF cosine similarity on categorical values)
        5. Find computed columns (structural pattern inference)
        6. Compose SQL (semantic understanding → structured SQL)
        7. Validate against actual data
        8. Fall back to hardcoded engine if confidence is low
        """
        # Typo normalization (still useful as a preprocessing step)
        try:
            from dynamic_sql_engine import normalize_typos
            question = normalize_typos(question)
        except ImportError:
            pass

        # Step 1: Classify intent
        intent = self.semantic.classify_intent(question)
        logger.info("Intent: %s (%.2f)", intent['intent'], intent['confidence'])

        # Step 2: Match columns
        columns = self.semantic.match_columns(question, top_k=15)

        # Step 3: Match tables
        tables = self.semantic.match_tables(question, top_k=5)

        # Step 4: Match values
        values = self.semantic.match_values(question, top_k=10)

        # Step 5: Find computed columns
        computed = self.semantic.find_computed_columns(question)

        # Step 6: Compose SQL
        result = self.composer.compose(question, intent, columns, tables, values, computed)

        # Step 7: Validate
        validation = self.semantic.validate(result['sql'], result['tables_used'])
        result['data_warnings'] = validation.get('warnings', [])
        result['data_suggestions'] = validation.get('suggestions', [])

        # Step 7b: Extract entities (from NLPEngineFactory if available)
        entities = self.semantic.extract_entities(question)
        if entities:
            result['entities'] = entities

        # Step 8: Enrich with semantic metadata
        result['semantic_intent'] = intent['intent']
        result['semantic_confidence'] = intent['confidence']
        result['intent_backend'] = intent.get('backend', 'scratch')
        result['semantic_columns'] = columns[:5]
        result['semantic_values'] = values[:5]
        result['columns_resolved'] = [
            {'column': c['column'], 'table': c['table'],
             'match_type': 'semantic', 'original_term': question}
            for c in columns[:10]
        ]
        result['filters'] = []
        result['agg_info'] = {
            'agg_func': intent['intent'].upper() if intent['intent'] in ('count', 'aggregate') else None,
            'group_by_terms': [],
            'top_n': None,
            'order': 'DESC',
        }

        # Step 9: Decision — use semantic result or fall back?
        if result['confidence'] < self.MIN_SEMANTIC_CONFIDENCE and self._fallback:
            logger.info("Low semantic confidence (%.2f), falling back to hardcoded engine",
                        result['confidence'])
            fallback_result = self._fallback.generate(question)

            # Enrich fallback with semantic metadata
            fallback_result['semantic_intent'] = intent['intent']
            fallback_result['semantic_confidence'] = intent['confidence']
            fallback_result['semantic_columns'] = columns[:5]
            fallback_result['semantic_values'] = values[:5]
            fallback_result['used_fallback'] = True

            # Validate fallback too
            fb_validation = self.semantic.validate(
                fallback_result.get('sql', ''),
                fallback_result.get('tables_used', [])
            )
            if fb_validation.get('warnings'):
                fallback_result['data_warnings'] = fb_validation['warnings']
                fallback_result['data_suggestions'] = fb_validation.get('suggestions', [])

            return fallback_result

        result['used_fallback'] = False
        return result


if __name__ == '__main__':
    pass
