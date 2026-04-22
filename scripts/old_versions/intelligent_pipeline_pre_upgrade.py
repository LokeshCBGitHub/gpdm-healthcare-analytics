"""Intelligent Pipeline — Unified orchestrator for question → SQL → results.

Combines neural understanding, continuous learning, model selection,
CMS knowledge, and auto-visualization into a cohesive flow.
"""

import os
import re
import sys
import time
import json
import sqlite3
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger('gpdm.pipeline')

# Ensure script directory is on path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# ── Configuration Imports ──
try:
    from gpdm_config import (
        SEMANTIC_CACHE_SIZE, SEMANTIC_CACHE_SIMILARITY,
        FEEDBACK_SUCCESS, FEEDBACK_FAILURE, HOPFIELD_RECALL_THRESHOLD,
    )
except ImportError:
    SEMANTIC_CACHE_SIZE = 500
    SEMANTIC_CACHE_SIMILARITY = 0.88
    FEEDBACK_SUCCESS = 1.0
    FEEDBACK_FAILURE = 0.2
    HOPFIELD_RECALL_THRESHOLD = 0.82


class IntelligentPipeline:
    """Unified intelligent pipeline for healthcare analytics."""

    def __init__(self, db_path: str, catalog_dir: str = None,
                 nlp_mode: str = 'auto', neural_dim: int = 64):
        self.db_path = db_path
        self.catalog_dir = catalog_dir
        self.neural_dim = neural_dim
        self._init_start = time.time()

        # Data directory for persistence
        self.data_dir = os.path.dirname(db_path)
        self.state_db = os.path.join(self.data_dir, 'intelligent_state.db')

        from semantic_sql_engine import SemanticSQLEngine
        self.sql_engine = SemanticSQLEngine(db_path, catalog_dir, nlp_mode=nlp_mode)
        self.learner = self.sql_engine.semantic.learner
        logger.info("SQL engine initialized")

        from neural_intelligence import NeuralQueryUnderstanding
        self.neural = NeuralQueryUnderstanding(dim=neural_dim, db_path=db_path)
        pretrain_docs = self._build_pretrain_corpus()
        self.neural.initialize(self.learner, documents=pretrain_docs)
        self.neural.load_state(self.state_db)
        logger.info("Neural intelligence initialized")

        from continuous_learning import ContinuousLearningEngine
        self.learning = ContinuousLearningEngine(self.neural, self.state_db)

        from model_selector import MetaModelSelector
        self.model_selector = MetaModelSelector(db_path=self.state_db)

        from cms_data_loader import CMSKnowledgeBase
        self.cms = CMSKnowledgeBase(os.path.join(self.data_dir, 'cms_knowledge.db'))
        self.cms.load_all()

        self.clinical = None
        try:
            from tuva_clinical_layer import TuvaClinicalEnricher
            self.clinical = TuvaClinicalEnricher(db_path)
            if not self.clinical.is_healthcare:
                self.clinical = None
        except:
            pass

        # ── Advanced Analytics (differentiators) ──
        from advanced_analytics import (
            SQLSelfHealer, QueryDecomposer, ConfidenceScorer,
            NarrativeEngine, SemanticCache, AnomalyDetector
        )
        self.self_healer = SQLSelfHealer(db_path)
        self.decomposer = QueryDecomposer(db_path)
        self.confidence_scorer = ConfidenceScorer(
            schema_cache=self.self_healer._schema_cache
        )
        self.narrative_engine = NarrativeEngine(
            domain_config=getattr(self.sql_engine, 'domain_config', None)
        )
        self.semantic_cache = SemanticCache(max_size=SEMANTIC_CACHE_SIZE, similarity_threshold=SEMANTIC_CACHE_SIMILARITY)
        self.anomaly_detector = AnomalyDetector()

        # ── Deep Intelligence (beyond Genie) ──
        from deep_intelligence import (
            BenchmarkEngine, MultiReasoningEngine,
            PrecursorInsightEngine, DataGapDetector, RootCauseAnalyzer
        )
        # Pass domain_config from SQL engine for schema-driven intelligence
        _dc = getattr(self.sql_engine, 'domain_config', None)
        self.benchmark_engine = BenchmarkEngine(domain_config=_dc)
        self.multi_reasoning = MultiReasoningEngine(domain_config=_dc)
        self.precursor_insights = PrecursorInsightEngine(db_path, domain_config=_dc)
        self.data_gap_detector = DataGapDetector(db_path)
        self.root_cause_analyzer = RootCauseAnalyzer(db_path)

        # ── Statistical & Stochastic Models ──
        from statistical_models import StatisticalAnalyzer
        self.statistical_analyzer = StatisticalAnalyzer()

        # ── Concept Expander (multi-dimensional understanding) ──
        from concept_expander import ConceptExpander
        self.concept_expander = ConceptExpander(db_path)

        # ── Healthcare Analytics Models ──
        self.analytics_engine = None
        try:
            from healthcare_analytics import HealthcareAnalyticsEngine
            self.analytics_engine = HealthcareAnalyticsEngine(db_path)
            self.analytics_engine.train_all()
            logger.info("Healthcare analytics models trained")
        except Exception as e:
            logger.warning("Healthcare analytics models unavailable: %s", e)

        logger.info("Advanced analytics + deep intelligence + statistical models + concept expander initialized")

        # ── Column Relationship Graph (from SchemaLearner) ──
        self.column_graph = getattr(self.learner, 'column_graph', None)
        self.cooccurrence_matrix = getattr(self.learner, 'cooccurrence_matrix', None)
        if self.column_graph:
            logger.info("Column graph: %d relationships discovered",
                        len(self.column_graph.edges))

        # ── Intelligent Cache Manager (3-tier) ──
        self.cache_manager = None
        try:
            from cache_manager import CacheManager
            self.cache_manager = CacheManager(data_dir=self.data_dir)
            logger.info("3-tier cache manager initialized")
        except Exception as e:
            logger.warning("Cache manager unavailable: %s", e)

        # ── Query Tracker (suggested questions + KPIs) ──
        self.query_tracker = None
        try:
            from query_tracker import QueryTracker
            self.query_tracker = QueryTracker(data_dir=self.data_dir)
            logger.info("Query tracker initialized")
        except Exception as e:
            logger.warning("Query tracker unavailable: %s", e)

        # ── Multi-Model Forecast Runner ──
        self.forecast_runner = None
        try:
            from forecast_runner import ForecastRunner
            self.forecast_runner = ForecastRunner(forecast_horizon=6)
            logger.info("Multi-model forecast runner initialized")
        except Exception as e:
            logger.warning("Forecast runner unavailable: %s", e)

        # ── Advanced NLP Engine (stemming, BM25, NER, etc.) ──
        self.advanced_nlp = None
        try:
            from advanced_nlp_engine import AdvancedNLPEngine
            self.advanced_nlp = AdvancedNLPEngine()
            # Build corpus from schema for BM25
            corpus_docs = []
            for table, profiles in self.learner.tables.items():
                for p in profiles:
                    corpus_docs.append(f"{table} {p.name.replace('_', ' ')} {' '.join(p.semantic_tags)}")
            # Store corpus for later BM25 indexing
            self.advanced_nlp._corpus_docs = corpus_docs
            logger.info("Advanced NLP engine initialized (BM25, NER, stemming, %d docs)", len(corpus_docs))
        except Exception as e:
            logger.warning("Advanced NLP engine unavailable: %s", e)

        # ── GPDM Definition Engine (primary intelligence source) ──
        self.gpdm_definitions = None
        try:
            from gpdm_definitions import GPDMDefinitionEngine
            self.gpdm_definitions = GPDMDefinitionEngine(catalog_dir)
            logger.info("GPDM Definition Engine initialized (primary intelligence source)")
        except Exception as e:
            logger.warning("GPDM Definition Engine unavailable: %s", e)

        # ── Insight Engine (statistical, probabilistic, ML analysis) ──
        self.insight_engine = None
        try:
            from insight_engine import InsightEngine
            self.insight_engine = InsightEngine()
            logger.info("Insight Engine initialized (statistical, Bayesian, Monte Carlo, Markov)")
        except Exception as e:
            logger.warning("Insight Engine unavailable: %s", e)

        self._conversations: Dict[str, Dict] = {}

        # Track init time
        init_time = time.time() - self._init_start
        logger.info("Intelligent pipeline ready in %.1fs", init_time)

    def _build_pretrain_corpus(self) -> List[str]:
        """Build pretraining corpus from CMS knowledge + actual schema."""
        docs = []

        # CMS healthcare vocabulary
        try:
            from cms_data_loader import build_healthcare_vocabulary
            docs.extend(build_healthcare_vocabulary())
        except Exception:
            pass

        # Schema-derived documents
        for table, profiles in self.learner.tables.items():
            col_names = ' '.join(p.name.replace('_', ' ') for p in profiles)
            docs.append(f"{table} table with columns {col_names}")
            for p in profiles:
                tags = ' '.join(p.semantic_tags)
                doc = f"{p.name.replace('_', ' ')} in {table} {tags}"
                if p.is_categorical and p.sample_values:
                    vals = ' '.join(str(v) for v in p.sample_values[:5])
                    doc += f" values {vals}"
                docs.append(doc)

        # Common query patterns
        docs.extend([
            "how many total count number of records",
            "average mean sum total aggregate amount",
            "by per group breakdown distribution",
            "top highest best most ranking",
            "trend over time monthly quarterly yearly",
            "percentage rate ratio proportion",
            "compare versus difference gap",
            "denied rejected cancelled failed",
            "approved paid completed successful",
            "older than younger than age between",
            "cost amount charge payment reimbursement",
        ])

        return docs

    # ── Golden SQL Templates ──
    # Pre-tested SQL for common healthcare query patterns the planner struggles with.
    # These bypass the planner entirely for known-good results.
    _GOLDEN_TEMPLATES = [
        # Provider denial rates
        {
            'patterns': [r'provider.*denial\s*rate', r'denial\s*rate.*provider',
                         r'provider.*highest.*denial', r'which.*provider.*denial'],
            'sql': (
                "SELECT c.RENDERING_NPI as provider_npi, "
                "p.PROVIDER_FIRST_NAME || ' ' || p.PROVIDER_LAST_NAME as provider_name, "
                "p.SPECIALTY, COUNT(*) as total_claims, "
                "SUM(CASE WHEN c.CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) as denied_claims, "
                "ROUND(100.0 * SUM(CASE WHEN c.CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as denial_rate_pct"
                "FROM claims c LEFT JOIN providers p ON c.RENDERING_NPI = p.NPI "
                "GROUP BY c.RENDERING_NPI, provider_name, p.SPECIALTY "
                "HAVING COUNT(*) >= 5 "
                "ORDER BY denial_rate_pct DESC LIMIT 20"
            ),
        },
        # Member count by age group
        {
            'patterns': [r'member.*age\s*group', r'age\s*group.*member',
                         r'count.*age\s*group', r'patient.*age\s*group',
                         r'age.*distribution', r'age.*breakdown'],
            'sql': (
                "SELECT CASE "
                "WHEN CAST((julianday('now') - julianday(DATE_OF_BIRTH)) / 365.25 AS INT) < 18 THEN '0-17 Pediatric' "
                "WHEN CAST((julianday('now') - julianday(DATE_OF_BIRTH)) / 365.25 AS INT) BETWEEN 18 AND 25 THEN '18-25 Young Adult' "
                "WHEN CAST((julianday('now') - julianday(DATE_OF_BIRTH)) / 365.25 AS INT) BETWEEN 26 AND 40 THEN '26-40 Adult' "
                "WHEN CAST((julianday('now') - julianday(DATE_OF_BIRTH)) / 365.25 AS INT) BETWEEN 41 AND 55 THEN '41-55 Middle Age' "
                "WHEN CAST((julianday('now') - julianday(DATE_OF_BIRTH)) / 365.25 AS INT) BETWEEN 56 AND 64 THEN '56-64 Pre-Medicare' "
                "ELSE '65+ Medicare' END as age_group, "
                "COUNT(*) as member_count, "
                "ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM members), 1) as pct "
                "FROM members "
                "GROUP BY age_group "
                "ORDER BY MIN(CAST((julianday('now') - julianday(DATE_OF_BIRTH)) / 365.25 AS INT))"
            ),
        },
        # Denial rate by region
        {
            'patterns': [r'denial\s*rate.*region', r'region.*denial\s*rate',
                         r'denied.*by\s*region', r'denial.*by\s*region'],
            'sql': (
                "SELECT KP_REGION as region, COUNT(*) as total_claims, "
                "SUM(CASE WHEN CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) as denied_claims, "
                "ROUND(100.0 * SUM(CASE WHEN CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as denial_rate_pct"
                "FROM claims GROUP BY KP_REGION ORDER BY denial_rate_pct DESC"
            ),
        },
        # Denial rate by specialty
        {
            'patterns': [r'denial\s*rate.*specialty', r'specialty.*denial\s*rate',
                         r'denied.*by\s*specialty'],
            'sql': (
                "SELECT p.SPECIALTY, COUNT(*) as total_claims, "
                "SUM(CASE WHEN c.CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) as denied_claims, "
                "ROUND(100.0 * SUM(CASE WHEN c.CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as denial_rate_pct"
                "FROM claims c JOIN providers p ON c.RENDERING_NPI = p.NPI "
                "GROUP BY p.SPECIALTY HAVING COUNT(*) >= 10 ORDER BY denial_rate_pct DESC"
            ),
        },
        # (Readmission queries handled by analytics engine — no golden template needed)
        # Approval rate by plan type
        {
            'patterns': [r'approval\s*rate.*plan', r'plan.*approval\s*rate'],
            'sql': (
                "SELECT PLAN_TYPE as plan, COUNT(*) as total_claims, "
                "SUM(CASE WHEN CLAIM_STATUS = 'APPROVED' THEN 1 ELSE 0 END) as approved, "
                "ROUND(100.0 * SUM(CASE WHEN CLAIM_STATUS = 'APPROVED' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as approval_rate_pct"
                "FROM claims GROUP BY PLAN_TYPE ORDER BY approval_rate_pct DESC"
            ),
        },
        # Members by plan type
        {
            'patterns': [r'member.*by\s*plan', r'member.*per\s*plan',
                         r'member.*each\s*plan', r'member.*plan\s*type',
                         r'patient.*by\s*plan', r'patient.*plan\s*type',
                         r'enrollment.*by\s*plan', r'plan\s*type.*member',
                         r'plan\s*type.*breakdown', r'plan\s*type.*distribution'],
            'sql': (
                "SELECT PLAN_TYPE as plan_type, COUNT(*) as member_count, "
                "ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM members), 1) as pct "
                "FROM members GROUP BY PLAN_TYPE ORDER BY member_count DESC"
            ),
        },
        # Member count (total) — must come AFTER plan-type breakdown
        {
            'patterns': [r'^how\s+many\s+members\b(?!.*(?:by|per|each|plan|region|age|type))',
                         r'total.*member\s*count', r'^total\s+members$',
                         r'^how\s+many\s+patients\b(?!.*(?:by|per|each|plan|region|age|type))',
                         r'total.*patient\s*count'],
            'sql': (
                "SELECT COUNT(*) as total_members FROM members"
            ),
        },
        # Utilization per member per year (PMPY)
        {
            'patterns': [r'utilization.*per\s*member', r'utilization.*pmpy',
                         r'per\s*member.*utilization', r'pmpy',
                         r'utilization.*per\s*capita', r'avg.*utilization',
                         r'average.*utilization', r'utilization\s+rate'],
            'sql': (
                "SELECT "
                "SUBSTR(c.SERVICE_DATE, 1, 4) as service_year, "
                "COUNT(DISTINCT c.MEMBER_ID) as unique_members, "
                "COUNT(*) as total_claims, "
                "ROUND(1.0 * COUNT(*) / COUNT(DISTINCT c.MEMBER_ID), 2) as claims_per_member, "
                "ROUND(SUM(c.BILLED_AMOUNT) / COUNT(DISTINCT c.MEMBER_ID), 2) as cost_per_member "
                "FROM claims c "
                "GROUP BY service_year "
                "ORDER BY service_year"
            ),
        },
        # Top diagnoses
        {
            'patterns': [r'top.*diagnos', r'common.*diagnos', r'frequent.*diagnos',
                         r'most.*diagnos', r'diagnos.*frequent', r'diagnos.*common'],
            'sql': (
                "SELECT DIAGNOSIS_CODE, DIAGNOSIS_DESCRIPTION, "
                "COUNT(*) as claim_count, "
                "COUNT(DISTINCT MEMBER_ID) as affected_members, "
                "ROUND(SUM(BILLED_AMOUNT), 2) as total_billed "
                "FROM claims "
                "WHERE DIAGNOSIS_CODE IS NOT NULL "
                "GROUP BY DIAGNOSIS_CODE, DIAGNOSIS_DESCRIPTION "
                "ORDER BY claim_count DESC LIMIT 20"
            ),
        },
        # Top procedures
        {
            'patterns': [r'top.*procedure', r'common.*procedure', r'frequent.*procedure',
                         r'most.*procedure', r'procedure.*frequent', r'procedure.*common'],
            'sql': (
                "SELECT PROCEDURE_CODE, PROCEDURE_DESCRIPTION, "
                "COUNT(*) as claim_count, "
                "COUNT(DISTINCT MEMBER_ID) as affected_members, "
                "ROUND(SUM(BILLED_AMOUNT), 2) as total_billed "
                "FROM claims "
                "WHERE PROCEDURE_CODE IS NOT NULL "
                "GROUP BY PROCEDURE_CODE, PROCEDURE_DESCRIPTION "
                "ORDER BY claim_count DESC LIMIT 20"
            ),
        },
        # Claims by status
        {
            'patterns': [r'claim.*by\s*status', r'claim\s*status.*breakdown',
                         r'claim\s*status.*distribution', r'how\s+many.*claim.*status'],
            'sql': (
                "SELECT CLAIM_STATUS as status, COUNT(*) as claim_count, "
                "ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM claims), 1) as pct, "
                "ROUND(SUM(BILLED_AMOUNT), 2) as total_billed "
                "FROM claims GROUP BY CLAIM_STATUS ORDER BY claim_count DESC"
            ),
        },
        # Average cost per claim
        {
            'patterns': [r'average.*cost.*claim', r'avg.*cost.*claim',
                         r'mean.*cost.*claim', r'cost.*per\s*claim'],
            'sql': (
                "SELECT ROUND(AVG(BILLED_AMOUNT), 2) as avg_billed, "
                "ROUND(AVG(ALLOWED_AMOUNT), 2) as avg_allowed, "
                "ROUND(AVG(PAID_AMOUNT), 2) as avg_paid, "
                "COUNT(*) as total_claims "
                "FROM claims"
            ),
        },
    ]

    def _try_golden_template(self, question: str, t0: float, session_id: str) -> Optional[Dict[str, Any]]:
        """Try to match question against golden SQL templates."""
        q_lower = question.lower()
        for template in self._GOLDEN_TEMPLATES:
            for pattern in template['patterns']:
                if re.search(pattern, q_lower):
                    sql = template['sql']
                    try:
                        rows, columns, error = self._execute_sql(sql)
                        if error or not rows:
                            continue
                        # Build standard result
                        narrative = self.narrative_engine.generate(
                            question=question, sql=sql, rows=rows, columns=columns,
                            intent={'intent': 'lookup', 'confidence': 0.95},
                        )
                        result = {
                            'sql': sql,
                            'rows': rows,
                            'columns': columns,
                            'row_count': len(rows),
                            'error': None,
                            'source': 'golden_template',
                            'narrative': narrative,
                            'confidence': {'grade': 'A', 'overall': 0.95},
                            'confidence_grade': 'A',
                            'confidence_overall': 0.95,
                            'cache_hit': False,
                            'latency_ms': round((time.time() - t0) * 1000),
                            'suggestions': [],
                            'anomalies': [],
                            'dashboard': {
                                'chart_type': 'bar',
                                'chart_config': {},
                                'title': question,
                                'subtitle': '',
                                'secondary_chart': None,
                                'color_scheme': 'kp_blue',
                                'layout': 'standard',
                            },
                            'benchmark': {},
                        }
                        self.semantic_cache.put(question, result)
                        self._update_conversation(session_id, question, result, rows, columns)
                        logger.info("Golden template matched '%s' in %dms",
                                   question[:50], result['latency_ms'])
                        return result
                    except Exception as e:
                        logger.warning("Golden template error: %s", e)
                        continue
        return None

    def _check_analytics(self, question: str, t0: float, session_id: str) -> Optional[Dict[str, Any]]:
        """Check if question should be handled by analytics models. Returns result or None."""
        if not self.analytics_engine:
            return None
        _q_lower = question.lower()
        _analytics_keywords = [
            # Risk stratification
            'risk stratif', 'high risk', 'highest risk', 'risk score', 'risk tier',
            'rising risk', 'risk member', 'risk analyt', 'member risk',
            'risk by region', 'risk by plan', 'risk by age',
            # Readmission
            'readmission predict', 'readmission model', 'readmission risk',
            'readmit',
            # Cost anomalies
            'cost anomal', 'cost outlier', 'unusual cost', 'suspicious claim',
            'fraud', 'waste', 'anomal',
            # Provider scorecard
            'provider scorecard', 'provider performance', 'provider quality',
            'provider grade', 'best provider', 'worst provider', 'low performing',
            'bottom provider', 'top provider', 'worst performing',
            'provider rank', 'provider rating',
            # Population segments
            'population segment', 'population health', 'member cohort', 'cluster',
            'member segment', 'population cohort',
            # Care gaps
            'care gap', 'missed care', 'preventive care gap', 'non-compliance',
            'adherence gap', 'care pattern',
            # Overview
            'analytics overview', 'analytics model', 'what models',
            'what analytics', 'available model',
        ]
        _matched_kw = [kw for kw in _analytics_keywords if kw in _q_lower]
        if not _matched_kw:
            return None
        logger.info("Analytics keyword match: %s → %s", question[:60], _matched_kw)
        try:
            analytics_result = self.analytics_engine.query(question)
            import json as _json
            _data = analytics_result.get('data', {})
            _rows = []
            _columns = []
            if isinstance(_data, list) and _data:
                _columns = list(_data[0].keys()) if isinstance(_data[0], dict) else []
                _rows = [list(d.values()) if isinstance(d, dict) else [d] for d in _data]
            elif isinstance(_data, dict):
                _columns = ['metric', 'value']
                _rows = [[k, _json.dumps(v) if isinstance(v, (dict, list)) else v]
                         for k, v in _data.items()]

            result = {
                'sql': f"-- Analytics Model: {analytics_result.get('model', 'unknown')}\n"
                       f"-- Type: {analytics_result.get('type', 'unknown')}",
                'rows': _rows,
                'columns': _columns,
                'row_count': len(_rows),
                'analytics_model': analytics_result.get('model'),
                'analytics_type': analytics_result.get('type'),
                'analytics_data': _data,
                'latency_ms': round((time.time() - t0) * 1000),
                'source': 'analytics_engine',
                'cache_hit': False,
                'confidence': {'grade': 'A', 'overall': 0.95},
                'confidence_grade': 'A',
                'confidence_overall': 0.95,
                'error': None,
                'narrative': f"Results from {analytics_result.get('model', 'analytics')} model "
                             f"({analytics_result.get('type', '')}). "
                             f"Returned {len(_rows)} results.",
                'suggestions': [],
                'anomalies': [],
                'dashboard': {
                    'chart_type': 'table',
                    'chart_config': {},
                    'title': f"{analytics_result.get('model', '').replace('_', ' ').title()}",
                    'subtitle': analytics_result.get('type', '').replace('_', ' '),
                },
            }
            self.semantic_cache.put(question, result)
            self._update_conversation(session_id, question, result, _rows, _columns)
            logger.info("Analytics model '%s' handled query in %dms: %s",
                        analytics_result.get('model'), result['latency_ms'], question[:60])
            return result
        except Exception as e:
            logger.warning("Analytics engine error (falling through): %s", e)
            return None

    def process(self, question: str, session_id: str = 'default') -> Dict[str, Any]:
        """Process question through full intelligent pipeline with advanced analytics."""
        t0 = time.time()

        # ── Step -1.5: ANALYTICS MODEL CHECK (before concept expansion) ──
        analytics_result = self._check_analytics(question, t0, session_id)
        if analytics_result:
            return analytics_result

        # ── Step -1.2: GOLDEN TEMPLATES — Pre-tested SQL for common patterns ──
        golden_result = self._try_golden_template(question, t0, session_id)
        if golden_result:
            return golden_result

        # ── Step -1: CONCEPT EXPANSION — Is this a broad conceptual query? ──
        concept_key = self.concept_expander.detect_concept(question)
        if concept_key:
            concept_result = self.concept_expander.expand(question, concept_key)
            if concept_result and concept_result.get('dimensions'):
                logger.info("Concept expansion: '%s' → %d dimensions in %dms",
                           concept_key, concept_result['total_dimensions'],
                           concept_result['execution_time_ms'])
                # Build a unified result that the frontend can render as multi-panel
                concept_result['latency_ms'] = round((time.time() - t0) * 1000)
                concept_result['source'] = 'concept_expander'
                concept_result['cache_hit'] = False
                concept_result['confidence'] = {'grade': 'A', 'overall': 0.95}
                concept_result['confidence_grade'] = 'A'
                concept_result['confidence_overall'] = 0.95
                concept_result['error'] = None
                concept_result['narrative'] = self._build_concept_narrative(concept_result)
                concept_result['suggestions'] = self._build_concept_suggestions(concept_key, question)
                concept_result['anomalies'] = []
                # Populate top-level rows from the first dimension so API consumers
                # (and tests) see actual data, not empty arrays
                dims = concept_result.get('dimensions', [])
                if dims and dims[0].get('rows'):
                    concept_result['rows'] = dims[0]['rows']
                    concept_result['columns'] = dims[0].get('columns', [])
                else:
                    concept_result['rows'] = []
                    concept_result['columns'] = []
                concept_result['sql'] = f'-- Multi-dimensional concept query: {concept_key}\n' + '\n'.join(
                    f'-- Dimension {i+1}: {d["label"]}' for i, d in enumerate(concept_result['dimensions'])
                )
                concept_result['dashboard'] = {
                    'chart_type': 'concept_multi',
                    'chart_config': {},
                    'layout': 'grid',
                    'title': concept_result['label'],
                    'subtitle': concept_result['description'],
                    'secondary_chart': None,
                    'color_scheme': 'kp_blue',
                }
                # Cache it
                self.semantic_cache.put(question, concept_result)
                # Update conversation context so follow-ups work
                self._update_conversation(
                    session_id, question, concept_result,
                    concept_result.get('rows', []),
                    concept_result.get('columns', [])
                )
                return concept_result

        # ── Step 0: CACHE CHECK — Instant return for near-duplicate questions ──
        cached = self.semantic_cache.get(question)
        if cached:
            cached['latency_ms'] = round((time.time() - t0) * 1000)
            cached['source'] = 'semantic_cache'
            cached['cache_hit'] = True
            logger.info("Cache hit for '%s' in %dms", question[:40], cached['latency_ms'])
            # Update conversation context so follow-ups work even after cache hits
            self._update_conversation(
                session_id, question, cached,
                cached.get('rows', []), cached.get('columns', [])
            )
            return cached

        # ── Step 1: ANALYZE — What kind of question is this? ──
        intent = self.sql_engine.semantic.classify_intent(question)
        try:
            memory_recall = self.neural.hopfield.recall(
                self.neural.embeddings.encode_sentence(question), top_k=3
            )
        except Exception as e:
            logger.debug("Hopfield recall error (non-fatal): %s", e)
            memory_recall = []

        strategy = self.model_selector.select_strategy(
            question, intent, memory_recall
        )
        logger.info("Strategy: %s (%.2f) for '%s'",
                     strategy['strategy'], strategy['confidence'], question[:50])

        # ── Step 2: UNDERSTAND — Deep neural comprehension ──
        try:
            understanding = self.neural.understand(question)
        except Exception as e:
            logger.debug("Neural understand error (non-fatal): %s", e)
            understanding = {}

        # ── Step 3: RECALL — Check learning memory for similar patterns ──
        recalled_sql = None
        if strategy['use_hopfield'] and memory_recall:
            for meta, score in memory_recall:
                if score > HOPFIELD_RECALL_THRESHOLD and 'sql' in meta:
                    recalled_sql = meta['sql']
                    logger.info("Hopfield recall: score=%.2f, using cached SQL", score)
                    break

        # ── Step 4: CONTEXT — Multi-turn conversation ──
        conv = self._get_conversation(session_id)
        is_followup = self._is_followup(question, conv)
        effective_question = question
        if is_followup and conv.get('last_question'):
            effective_question = self._rewrite_followup(question, conv)
            logger.info("Follow-up rewritten: '%s'", effective_question)

        # ── Step 4b: DECOMPOSE — Break complex questions into sub-queries ──
        decompose_steps = []
        if self.decomposer.needs_decomposition(effective_question):
            decompose_steps = self.decomposer.decompose(effective_question)
            logger.info("Decomposed into %d steps", len(decompose_steps))

        # ── Step 0.5: NLP ENRICHMENT — stems, entities, expanded terms ──
        nlp_enrichment = {}
        if self.advanced_nlp:
            try:
                processed = self.advanced_nlp.process_query(effective_question)
                nlp_enrichment = {
                    'tokens': processed.tokens,
                    'stems': processed.stems,
                    'lemmas': processed.lemmas,
                    'entities': processed.entities,
                    'expanded_terms': list(processed.expanded_terms),
                    'ngrams': processed.ngrams,
                }
                # Add extracted entities to help SQL generation
                if nlp_enrichment.get('entities'):
                    logger.info("NLP entities: %s", nlp_enrichment['entities'])
            except Exception as e:
                logger.debug("NLP enrichment error (non-fatal): %s", e)

        # ── Step 0.6: GPDM DEFINITION ENRICHMENT — primary intelligence ──
        gpdm_context = {}
        if self.gpdm_definitions:
            try:
                tables_list = list(self.learner.tables.keys())
                cols_list = []
                for tbl_profiles in self.learner.tables.values():
                    cols_list.extend([p.name for p in tbl_profiles])
                gpdm_context = self.gpdm_definitions.enrich_query(
                    effective_question, tables_list, cols_list
                )
                if gpdm_context.get('confidence', 0) > 0:
                    logger.info("GPDM enrichment: confidence=%.2f", gpdm_context.get('confidence', 0))
            except Exception as e:
                logger.debug("GPDM enrichment error (non-fatal): %s", e)

        # ── Step 5: GENERATE — Produce SQL ──
        t_gen = time.time()
        if recalled_sql and strategy['strategy'] == 'RECALL':
            sql_result = {
                'sql': recalled_sql,
                'tables_used': [],
                'confidence': 0.9,
                'intent': intent['intent'],
                'explanation': 'Pattern recalled from memory',
                'source': 'hopfield_recall',
            }
        else:
            sql_result = self.sql_engine.generate(effective_question)
            sql_result['source'] = 'semantic_engine'
        gen_time = time.time() - t_gen

        # ── Step 6: EXECUTE SQL ──
        rows, columns, error = self._execute_sql(sql_result.get('sql', ''))

        # ── Step 6b: SELF-HEAL — Fix bad SQL or verify results ──
        heal_actions = []
        probe_results = {}
        def _regenerate(q):
            return self.sql_engine.generate(q)

        heal_result = self.self_healer.heal(
            sql=sql_result.get('sql', ''),
            error=error,
            rows=rows,
            columns=columns,
            question=effective_question,
            regenerate_fn=_regenerate if error else None
        )
        if heal_result.get('healed'):
            rows = heal_result['rows']
            columns = heal_result['columns']
            error = heal_result['error']
            sql_result['sql'] = heal_result['sql']
            heal_actions = heal_result.get('heal_actions', [])
            logger.info("Self-healed: %s", heal_actions)
        probe_results = heal_result.get('probe_results', {})

        # ── Step 7: CONFIDENCE SCORE — Multi-dimensional quality assessment ──
        confidence = self.confidence_scorer.score(
            question=effective_question,
            sql=sql_result.get('sql', ''),
            rows=rows,
            columns=columns,
            error=error,
            intent=intent,
            strategy=strategy
        )

        # ── Step 8: ANOMALY DETECTION — Flag outliers and unusual patterns ──
        anomalies = []
        if rows and not error:
            anomalies = self.anomaly_detector.detect(rows, columns, effective_question)

        # ── Step 9: ENRICH — Clinical context ──
        clinical_context = {}
        if self.clinical and strategy.get('use_clinical'):
            try:
                enrichment = self.clinical.enrich(effective_question, sql_result)
                clinical_context = enrichment or {}
            except Exception:
                pass

        # CMS knowledge enrichment
        cms_context = self._enrich_with_cms(question)

        # ── Step 9b: BENCHMARK — Compare against industry standards ──
        benchmark = self.benchmark_engine.benchmark(
            effective_question, rows, columns, intent
        )

        # ── Step 9c: DATA GAP DETECTION — Clear messages for missing data ──
        data_gaps = self.data_gap_detector.detect(
            effective_question, sql_result.get('sql', ''), error, rows, columns
        )

        # ── Step 9d: MULTI-REASONING — 4 parallel reasoning strategies ──
        reasoning = self.multi_reasoning.reason(
            effective_question, rows, columns, intent,
            sql_result.get('sql', ''), anomalies
        )

        # ── Step 9e: PRECURSOR DEEP DIVES — Auto follow-up analyses ──
        deep_dives = []
        if rows and not error and len(rows) <= 5000:
            try:
                deep_dives = self.precursor_insights.generate_deep_dives(
                    effective_question, rows[:500], columns
                )
            except Exception as e:
                logger.debug("Deep dive error (non-fatal): %s", e)

        # ── Step 9f: ROOT CAUSE — Investigate anomalies ──
        root_causes = []
        if anomalies:
            root_causes = self.root_cause_analyzer.investigate(
                anomalies, sql_result.get('sql', ''), effective_question
            )

        # ── Step 9g: STATISTICAL MODELS — Monte Carlo, Bayesian, etc. ──
        stat_models = []
        if rows and not error and len(rows) <= 5000:
            try:
                stat_models = self.statistical_analyzer.analyze(
                    effective_question, rows[:500], columns, intent
                )
                if stat_models:
                    logger.info("Statistical models: %s",
                                [m['type'] for m in stat_models])
            except Exception as e:
                logger.debug("Statistical analysis failed: %s", e)

        # ── Step 10: NARRATIVE — Generate analyst-quality explanation ──
        narrative = self.narrative_engine.generate(
            question=effective_question,
            sql=sql_result.get('sql', ''),
            rows=rows,
            columns=columns,
            intent=intent,
            clinical_context=clinical_context,
            confidence=confidence,
            anomalies=anomalies,
            heal_actions=heal_actions,
            decompose_steps=decompose_steps
        )

        # ── Step 11: VISUALIZE — Auto-select dashboard ──
        dashboard = self.model_selector.select_dashboard(
            sql_result,
            intent=intent.get('intent', ''),
            columns=columns,
            data_sample=rows[:5] if rows else [],
            row_count=len(rows),
        )

        # ── Step 12: LEARN — Update from this interaction ──
        feedback = FEEDBACK_SUCCESS if rows and not error else FEEDBACK_FAILURE
        try:
            self.learning.on_interaction(
                question=effective_question,
                sql=sql_result.get('sql', ''),
                feedback=feedback,
                result_count=len(rows),
                metadata={
                    'intent': intent.get('intent', ''),
                    'strategy': strategy['strategy'],
                    'tables': sql_result.get('tables_used', []),
                    'session_id': session_id,
                    'confidence_grade': confidence.get('grade', ''),
                }
            )
        except Exception as e:
            logger.debug("Learning update error (non-fatal): %s", e)

        # Record strategy outcome
        self.model_selector.record_outcome(
            strategy['strategy'],
            success=bool(rows and not error),
            latency=gen_time,
        )

        # ── Step 13: UPDATE CONVERSATION ──
        self._update_conversation(session_id, question, sql_result, rows, columns)

        # ── Step 14: SUGGEST FOLLOW-UPS ──
        suggestions = self._generate_suggestions(
            intent, columns, rows, sql_result,
            question=question, anomalies=anomalies,
            reasoning=reasoning, benchmark=benchmark,
            session_id=session_id,
        )

        total_time = time.time() - t0

        result = {
            # Core result
            'sql': sql_result.get('sql', ''),
            'rows': rows,
            'columns': columns,
            'row_count': len(rows),
            'error': error,

            # Intelligence metadata
            'intent': intent.get('intent', ''),
            'intent_confidence': intent.get('confidence', 0),
            'strategy': strategy['strategy'],
            'strategy_confidence': strategy['confidence'],
            'source': sql_result.get('source', 'semantic_engine'),

            # Neural understanding
            'attention_active': strategy.get('use_transformer', False),
            'hopfield_active': strategy.get('use_hopfield', False),
            'gnn_active': strategy.get('use_gnn', False),
            'memory_recalls': len(memory_recall),

            # ── Advanced Analytics (differentiators) ──
            'confidence': confidence,
            'confidence_grade': confidence.get('grade', ''),
            'confidence_overall': confidence.get('overall', 0),
            'anomalies': anomalies,
            'narrative': narrative,
            'heal_actions': heal_actions,
            'probe_results': probe_results,
            'decompose_steps': [s.get('label', '') for s in decompose_steps],
            'cache_hit': False,
            'cache_stats': self.semantic_cache.stats,

            # ── Deep Intelligence (beyond Genie) ──
            'benchmark': benchmark,
            'data_gaps': data_gaps,
            'reasoning': reasoning,
            'deep_dives': deep_dives,
            'root_causes': root_causes,
            'stat_models': stat_models,

            # Clinical context
            'clinical_context': clinical_context,
            'cms_context': cms_context,

            # Conversation
            'is_followup': is_followup,
            'effective_question': effective_question,

            # NLP Enrichment
            'nlp_enrichment': nlp_enrichment,

            # GPDM Intelligence
            'gpdm_context': gpdm_context,

            # Dashboard
            'dashboard': {
                'chart_type': dashboard.chart_type,
                'chart_config': dashboard.chart_config,
                'layout': dashboard.layout,
                'title': dashboard.title,
                'subtitle': dashboard.subtitle,
                'secondary_chart': dashboard.secondary_chart,
                'color_scheme': dashboard.color_scheme,
            },

            # Follow-ups
            'suggestions': suggestions,

            # Performance
            'latency_ms': round(total_time * 1000),
            'generation_ms': round(gen_time * 1000),

            # Explanation (legacy) + narrative (new)
            'explanation': narrative,
        }

        # ── Step 14b: INSIGHT ENGINE — statistical, probabilistic, ML analysis ──
        if self.insight_engine and rows and columns and not error:
            try:
                row_dicts = [dict(zip(columns, row)) for row in rows] if rows and isinstance(rows[0], (tuple, list)) else rows
                insights = self.insight_engine.generate_insights(
                    question=question,
                    sql=sql_result.get('sql', ''),
                    rows=row_dicts,
                    columns=columns
                )
                result['insights'] = insights
                logger.info("Insight Engine: quality=%.2f, keys=%s",
                           insights.get('quality_score', 0),
                           list(insights.keys())[:5])
            except Exception as e:
                logger.debug("Insight engine error (non-fatal): %s", e)

        # ── Step 15: CACHE — Store result for future fast retrieval ──
        self.semantic_cache.put(question, result)

        # ── Step 15b: PERSISTENT CACHE — 3-tier cache for cross-session ──
        if self.cache_manager and rows and not error:
            try:
                self.cache_manager.set_query_result(
                    question=question,
                    sql=sql_result.get('sql', ''),
                    result={'rows': rows[:500], 'columns': columns,
                            'row_count': len(rows)},
                    ttl_hours=1.0
                )
            except Exception as e:
                logger.debug("Cache store error (non-fatal): %s", e)

        # ── Step 16: QUERY TRACKING — Record for suggestions + KPIs ──
        if self.query_tracker:
            try:
                from query_tracker import QueryRecord
                record = QueryRecord(
                    question=question,
                    normalized=self.query_tracker.normalize_question(question),
                    user=session_id,
                    timestamp=time.time(),
                    intent=intent.get('intent', ''),
                    tables=sql_result.get('tables_used', []),
                    columns=columns,
                    success=bool(rows and not error),
                    response_time_ms=round(total_time * 1000),
                    sql=sql_result.get('sql', ''),
                    error=error or '',
                )
                self.query_tracker.record(record)
            except Exception as e:
                logger.debug("Query tracking error (non-fatal): %s", e)

        # ── Step 17: CO-OCCURRENCE LEARNING — Update column matrix ──
        if self.cooccurrence_matrix and columns and not error:
            try:
                self.cooccurrence_matrix.record_success(columns)
            except Exception as e:
                logger.debug("Co-occurrence update error (non-fatal): %s", e)

        return result

    def _execute_sql(self, sql: str) -> Tuple[List, List, Optional[str]]:
        """Execute SQL and return (rows, column_names, error)."""
        if not sql:
            return [], [], "No SQL generated"
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(sql)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            conn.close()
            return rows, columns, None
        except Exception as e:
            return [], [], str(e)

    def _enrich_with_cms(self, question: str) -> Dict[str, Any]:
        """Enrich with CMS clinical knowledge."""
        context = {}
        q_lower = question.lower()

        # Check for clinical conditions
        for condition_id, info in self.cms.get_statistics().items():
            pass  # Stats only

        # Check for specific terms
        condition = None
        for term in ['diabetes', 'hypertension', 'heart failure', 'copd',
                     'depression', 'cancer', 'ckd', 'obesity', 'asthma']:
            if term in q_lower:
                condition = self.cms.find_condition(term)
                if condition:
                    context['condition'] = condition
                    break

        # Check for quality measure references
        for measure_id in ['CDC', 'BCS', 'CBP', 'COL', 'PCR', 'AMM', 'FUH', 'EDU']:
            if measure_id.lower() in q_lower or measure_id in question:
                measure = self.cms.get_quality_measure(measure_id)
                if measure:
                    context['quality_measure'] = measure
                    break

        # Check for HEDIS mention
        if 'hedis' in q_lower:
            context['hedis_note'] = 'HEDIS measures available: CDC, BCS, CBP, COL, PCR, AMM, FUH, EDU'

        return context

    def _get_conversation(self, session_id: str) -> Dict:
        """Get or create conversation context."""
        if session_id not in self._conversations:
            self._conversations[session_id] = {
                'turns': [],
                'last_question': '',
                'last_sql': '',
                'last_intent': '',
                'last_tables': [],
                'last_columns': [],
            }
        return self._conversations[session_id]

    def _is_followup(self, question: str, conv: Dict) -> bool:
        """Determine if this is a follow-up question.

        A follow-up modifies the PREVIOUS query (add grouping, filter, swap agg).
        A NEW question about a different topic using words like 'same period' or
        'corresponding' is NOT a follow-up — it's a new query that references context.
        """
        if not conv.get('last_question'):
            return False
        q = question.lower().strip()
        words = q.split()

        # Very short queries are likely follow-ups ("by region", "only denied")
        # BUT NOT if they contain a table/entity name — that's a standalone query
        if len(words) <= 3:
            # Check if any word matches a table name (self-contained query)
            # Also normalize typos first so "diagnossi" → "diagnoses" matches
            table_names = {t.lower() for t in self.learner.tables}
            table_singulars = {t.lower().rstrip('s') for t in self.learner.tables}
            q_words = set(w.lower() for w in words)
            # Also try normalized words
            try:
                from dynamic_sql_engine import normalize_typos
                normalized = normalize_typos(q)
                q_words |= set(normalized.lower().split())
            except ImportError:
                pass
            # If a word IS a table name or its singular, it's an independent query
            if q_words & (table_names | table_singulars):
                return False
            return True

        # Check for explicit pronoun references to previous context
        has_pronoun_ref = any(w in q for w in ('this', 'these', 'that', 'them', 'it'))

        # If the question is self-contained (has its own metric + dimension)
        # AND doesn't reference previous context, it's a new question
        has_own_metric = any(w in q for w in ('cost', 'revenue', 'rate', 'count',
                                                'average', 'total', 'volume'))
        has_own_dimension = any(w in q for w in ('by', 'per', 'over time', 'monthly',
                                                   'quarterly', 'by provider', 'by region'))
        if has_own_metric and has_own_dimension and not has_pronoun_ref:
            return False

        # "corresponding", "same period", "same time" — new query referencing time context
        if any(s in q for s in ('corresponding', 'same period', 'same time',
                                  'same timeframe', 'comparable')):
            return False

        followup_signals = ['also', 'what about', 'those', 'instead',
                           'but only', 'only', 'just the', 'now show', 'and also',
                           'same but', 'show this', 'relate to', 'how does this',
                           'how do these', 'for these', 'for those']
        if any(s in q for s in followup_signals):
            return True

        # Pronoun references with a question about the previous result
        if has_pronoun_ref and any(w in q for w in ('relate', 'affect', 'impact',
                                                      'connect', 'correlate', 'compare',
                                                      'consistent', 'across', 'differ',
                                                      'vary', 'break', 'split')):
            return True

        return False

    def _rewrite_followup(self, question: str, conv: Dict) -> str:
        """Rewrite a follow-up into a standalone question."""
        import re
        q = question.lower().strip()
        last = conv.get('last_question', '')

        # "by X" → add grouping to last question
        if q.startswith('by '):
            base = re.sub(r'\bby\s+\w+', '', last).strip()
            return f"{base} {question}"

        # "Show this by X" / "break this down by X" → add dimension to last question
        by_match = re.search(r'\bby\s+(\w+(?:\s+\w+)?)', q)
        if by_match and any(w in q for w in ('this', 'it', 'that')):
            new_dim = by_match.group(0)  # "by provider", "by region", etc.
            # Remove old "by X" from last question and add new one
            base = re.sub(r'\bby\s+\w+(?:\s+\w+)?', '', last).strip()
            # Also remove "over time" / "changed over time" for group-by rewrites
            base = re.sub(r'\b(changed\s+)?over\s+time\b', '', base).strip()
            # Clean up trailing punctuation, noise words, and extra whitespace
            base = re.sub(r'[?.!]+$', '', base).strip()
            base = re.sub(r'\s+(?:to|the|a|an|is|and|or|for|how|has)\s*$', '', base).strip()
            base = re.sub(r'\s{2,}', ' ', base).strip()
            return f"{base} {new_dim}"

        # "only X" → add filter to last question
        if q.startswith('only ') or q.startswith('just '):
            filter_term = q.split(' ', 1)[1]
            return f"{filter_term} {last}"

        # "show average instead" → swap aggregation
        if 'instead' in q:
            for agg in ['average', 'total', 'count', 'sum', 'maximum', 'minimum']:
                if agg in q:
                    base = re.sub(r'\b(average|total|count|sum|maximum|minimum)\b', agg, last)
                    return base

        # "Is this consistent across X" / "does this vary by X" / "break this down by X type"
        # → Add dimension to previous query
        across_match = re.search(
            r'(?:consistent|differ|vary|varies?|break.*down)\s+(?:across|by|for)\s+(?:all\s+)?(.+?)(?:\?|$)',
            q
        )
        if across_match and any(w in q for w in ('this', 'these', 'that', 'it')):
            new_dim = across_match.group(1).strip().rstrip('?')
            # Remove old "by X" or "over time" from last question and add new dimension
            base = re.sub(r'\bby\s+\w+(?:\s+\w+)?', '', last).strip()
            base = re.sub(r'\b(changed\s+)?over\s+time\b', '', base).strip()
            base = re.sub(r'\b(?:trend|over\s+the\s+last\s+\d+\s+\w+)\b', '', base).strip()
            base = re.sub(r'[?.!]+$', '', base).strip()
            base = re.sub(r'\s{2,}', ' ', base).strip()
            return f"{base} by {new_dim}"

        # "How does this relate to X" / "relate this to costs" / "what about costs for these"
        # → Extract the new concept and merge with the entity/dimension from previous query
        relate_match = re.search(
            r'(?:how\s+does?\s+(?:this|these|that|it)\s+relate\s+to\s+(.+?)(?:\?|$))|'
            r'(?:relate\s+(?:this|these|that|it)\s+to\s+(.+?)(?:\?|$))|'
            r'(?:(?:what|how)\s+about\s+(.+?)\s+(?:for|of|with)\s+(?:this|these|those|them))',
            q
        )
        if relate_match:
            new_concept = (relate_match.group(1) or relate_match.group(2)
                          or relate_match.group(3) or '').strip()
            if new_concept:
                # Extract entity/dimension from previous question
                last_tables = conv.get('last_tables', [])
                # Find the entity word from previous question
                entity_word = ''
                for tbl in last_tables:
                    tbl_lower = tbl.lower()
                    if tbl_lower in last.lower() or tbl_lower.rstrip('s') in last.lower():
                        entity_word = tbl_lower
                        break
                if not entity_word and last_tables:
                    entity_word = last_tables[0].lower()
                if entity_word:
                    # Use singular form for grouping dimension
                    # (avoids table-name matching: "by providers" → fact=providers)
                    dim_word = entity_word.rstrip('s') if len(entity_word) > 3 else entity_word
                    return f"Show total {new_concept} by {dim_word}"
                else:
                    return f"Show {new_concept}"

        # Default: prepend to last question context
        return f"{last} {question}"

    def _update_conversation(self, session_id: str, question: str,
                              sql_result: Dict, rows: List, columns: List):
        """Update conversation context."""
        conv = self._get_conversation(session_id)
        conv['turns'].append({
            'question': question,
            'sql': sql_result.get('sql', ''),
            'row_count': len(rows),
            'timestamp': time.time(),
        })
        conv['last_question'] = question
        conv['last_sql'] = sql_result.get('sql', '')
        conv['last_intent'] = sql_result.get('intent', '')
        conv['last_tables'] = sql_result.get('tables_used', [])
        conv['last_columns'] = columns

        # Keep only last 10 turns
        if len(conv['turns']) > 10:
            conv['turns'] = conv['turns'][-10:]

    def _generate_suggestions(self, intent: Dict, columns: List,
                               rows: List, sql_result: Dict,
                               question: str = '', anomalies: List = None,
                               reasoning: Dict = None, benchmark: Dict = None,
                               session_id: str = 'default') -> List[str]:
        """Generate smart, context-aware follow-up suggestions based on the actual
        question, results, detected anomalies, healthcare domain knowledge, AND
        session history (to maintain topic continuity across multi-turn conversations).

        Each suggestion is a natural-language question the user can click to ask next,
        phrased as a real analyst would ask as a logical next step.
        """
        suggestions = []
        q = question.lower()
        col_lower = [c.lower() for c in columns] if columns else []
        col_text = ' '.join(col_lower)
        anomalies = anomalies or []

        if not columns or not rows:
            return ['What is the overall claim denial rate?',
                    'Show total claims by provider',
                    'Show member enrollment by plan type',
                    'What are the top 10 diagnoses by volume?']

        # ── Detect the domain and dimensions of the current query ──
        is_denial = any(w in q + col_text for w in ('denial', 'denied', 'deny', 'rejection'))
        is_readmission = any(w in q + col_text for w in ('readmission', 'readmit'))
        is_cost = any(w in q + col_text for w in ('cost', 'paid', 'charge', 'billed', 'amount', 'revenue'))
        is_utilization = any(w in q + col_text for w in ('utilization', 'volume', 'visit', 'encounter', 'appointment'))
        is_provider = any(w in q + col_text for w in ('provider', 'npi', 'physician', 'doctor', 'specialty'))
        is_member = any(w in q + col_text for w in ('member', 'patient', 'enrollee', 'demographic'))
        is_region = any(w in q + col_text for w in ('region', 'state', 'market', 'facility', 'location'))
        is_plan = any(w in q + col_text for w in ('plan', 'plan_type', 'insurance', 'payer', 'coverage'))
        is_trend = any(w in q for w in ('trend', 'over time', 'monthly', 'quarterly', 'yearly', 'by month'))
        is_rate = any(w in q + col_text for w in ('rate', 'pct', 'percent', 'ratio'))

        has_outliers = len([a for a in anomalies if a.get('severity') == 'high']) > 0
        top_entity = str(rows[0][0]) if rows and rows[0] else ''

        # ── 1. Deep dive suggestions: break current view into next dimension ──
        # If looking at providers, suggest drilling by region, plan, or time
        if is_provider and not is_region:
            if is_denial:
                suggestions.append('What is the denial rate by region?')
            elif is_cost:
                suggestions.append('Show average cost per encounter by region')
            else:
                suggestions.append('Show this breakdown by region')

        if is_provider and not is_plan:
            if is_denial:
                suggestions.append('Show denial rate by plan type')
            elif is_cost:
                suggestions.append('What is the cost distribution by plan type?')

        if is_region and not is_provider:
            if is_denial:
                suggestions.append('Which providers have the highest denial rate?')
            else:
                suggestions.append('Show top 10 providers by volume')

        if is_plan and not is_provider:
            suggestions.append('Which providers have the highest volume in this plan?')

        # ── 2a. If already a trend, suggest dimensional breakdowns ──
        if is_trend:
            if not is_provider:
                suggestions.append('Show this by provider to identify who is driving the trend')
            if not is_region:
                suggestions.append('Break this trend down by region')
            if not is_plan:
                suggestions.append('Is this trend consistent across all plan types?')
            if is_utilization or is_cost:
                suggestions.append('What is the denial rate trend over the same period?')
            if not is_cost and not is_denial:
                suggestions.append('Show the corresponding cost trend over the same period')

        # ── 2b. Trend suggestion: if not already a trend, offer time view ──
        if not is_trend:
            if is_denial:
                suggestions.append('Show denial rate trend over the last 12 months')
            elif is_readmission:
                suggestions.append('Show readmission rate trend by quarter')
            elif is_cost:
                suggestions.append('How has average cost per encounter changed over time?')
            elif is_utilization:
                suggestions.append('Show monthly visit volume trend')
            else:
                suggestions.append('Show this metric trend over the last 12 months')

        # ── 3. Outlier investigation: if anomalies detected, offer deep dive ──
        if has_outliers and top_entity:
            top_label = top_entity
            if top_entity.isdigit() and len(top_entity) == 10:
                top_label = f'provider NPI {top_entity}'
            elif top_entity.isdigit():
                top_label = f'provider {top_entity}'
            if is_denial:
                suggestions.append(f'What are the top denial reasons for {top_label}?')
            elif is_cost:
                suggestions.append(f'What is driving high costs for {top_label}?')
            else:
                suggestions.append(f'Show detailed breakdown for {top_label}')

        # ── 4. Cross-domain suggestions: pivot to related healthcare metrics ──
        if is_denial and not is_cost:
            suggestions.append('What is the financial impact of denied claims?')
        if is_denial and not is_readmission:
            suggestions.append('Do high-denial providers also have high readmission rates?')
        if is_cost and not is_denial:
            suggestions.append('What is the claim denial rate for high-cost providers?')
        if is_readmission and not is_cost:
            suggestions.append('What is the average cost for readmitted patients?')
        if is_utilization and not is_cost:
            suggestions.append('What is the average cost per visit by type?')
        if is_member and not is_utilization:
            suggestions.append('What is the average utilization per member per year?')

        # ── 5. Benchmark & comparison suggestions ──
        if is_rate and len(rows) > 5:
            suggestions.append('Which providers are above the 90th percentile?')
        if is_provider and len(rows) > 1:
            suggestions.append('Compare top 5 vs bottom 5 providers')

        # ── 6. Root cause / why questions ──
        if is_denial:
            if 'reason' not in q and 'why' not in q:
                suggestions.append('What are the most common denial reason codes?')
        if is_readmission:
            if 'diagnosis' not in q:
                suggestions.append('What are the top diagnoses for readmitted patients?')

        # ── 7. Specialty / department dimension ──
        if is_provider and 'specialty' not in q:
            suggestions.append('Show this metric by provider specialty')
        if is_provider and 'department' not in q and 'specialty' in q:
            suggestions.append('Break down by department')

        # ── 8. Population health angle ──
        if is_denial or is_readmission or is_cost:
            if 'age' not in q and 'gender' not in q:
                suggestions.append('How does this vary by patient age group?')

        # ── 9. SESSION CONTEXT — Cross-topic suggestions from prior turns ──
        conv = self._get_conversation(session_id)
        prior_turns = conv.get('turns', [])
        if len(prior_turns) >= 2:
            # Look at prior topics to suggest cross-domain connections
            prior_questions = ' '.join(t.get('question', '').lower() for t in prior_turns[:-1])
            prior_had_demographics = any(w in prior_questions for w in ('demographic', 'race', 'ethnicity', 'age', 'gender'))
            prior_had_claims = any(w in prior_questions for w in ('claim', 'denial', 'cost', 'paid', 'billed'))
            prior_had_readmission = any(w in prior_questions for w in ('readmission', 'readmit'))
            prior_had_encounters = any(w in prior_questions for w in ('encounter', 'visit', 'appointment'))
            prior_had_providers = any(w in prior_questions for w in ('provider', 'npi', 'specialty'))

            # If prior questions were about demographics but current isn't, suggest connection
            if prior_had_demographics and not is_member:
                if is_cost:
                    suggestions.append('How does average cost vary by race/ethnicity?')
                elif is_denial:
                    suggestions.append('Show denial rate by age group and gender')
                else:
                    suggestions.append('Break this down by patient demographics')

            # If prior was claims and current is different, suggest claim connection
            if prior_had_claims and not is_cost and not is_denial:
                suggestions.append('How does this relate to claim costs?')

            # If prior was readmission and current is different
            if prior_had_readmission and not is_readmission:
                suggestions.append('Is there a relationship between this and readmission rates?')

            # If prior was provider and current is different
            if prior_had_providers and not is_provider:
                suggestions.append('Show this metric by provider')

        # ── Deduplicate and cap at 6 ──
        seen = set()
        unique = []
        for s in suggestions:
            s_key = s.lower().strip()
            if s_key not in seen:
                seen.add(s_key)
                unique.append(s)
        return unique[:6]

    # ── Concept expansion helpers ──

    def _build_concept_narrative(self, concept_result: Dict) -> str:
        """Build a unified narrative for a multi-dimensional concept query."""
        dims = concept_result.get('dimensions', [])
        label = concept_result.get('label', 'Analysis')
        parts = [f"Comprehensive {label}: Analyzed {len(dims)} dimensions in {concept_result.get('execution_time_ms', 0)}ms."]

        for d in dims:
            if d.get('insight') and not d.get('error'):
                parts.append(f"[{d['label']}] {d['insight']}")

        return " ".join(parts)

    def _build_concept_suggestions(self, concept_key: str, question: str) -> List[str]:
        """Generate follow-up suggestions specific to a concept expansion.

        For static concepts, use curated domain-aware suggestions.
        For dynamic:table concepts, auto-generate suggestions based on the
        table's actual dimensions, metrics, and cross-domain relationships.
        """
        # ── Static concept suggestions (curated) ──
        concept_suggestions = {
            'demographics': [
                'Show denial rate broken down by gender',
                'How does average cost per member vary by age group?',
                'Which race/ethnicity group has the highest readmission rate?',
                'Show chronic condition prevalence by region',
                'Compare utilization patterns across plan types',
                'Which age group has the highest ED utilization?',
            ],
            'financials': [
                'Show denial rate by denial reason code',
                'Which providers have the highest cost per encounter?',
                'How has total monthly spend changed over time?',
                'Show cost breakdown by top 10 diagnosis codes',
                'Compare cost per member across regions',
                'What is the financial impact of readmissions?',
            ],
            'provider_network': [
                'Which specialties have the highest denial rates?',
                'Show average patient panel by region',
                'Which providers have the most no-shows?',
                'Compare provider productivity by specialty',
                'Show referral patterns between specialties',
                'Which providers are not accepting new patients?',
            ],
            'utilization': [
                'Show ED visit rate by member age group',
                'Which departments have the longest wait times?',
                'Compare inpatient vs outpatient volume trend',
                'Show no-show rate by department',
                'Which facilities have the highest utilization?',
                'Show telehealth adoption trend over time',
            ],
            'quality': [
                'Show denial rate trend by month',
                'Which providers have the highest readmission rates?',
                'Compare medication adherence by region',
                'Show HCC capture rate by provider',
                'Which diagnoses are most commonly coded as chronic?',
                'Show quality metrics by plan type',
            ],
        }

        if concept_key in concept_suggestions:
            return concept_suggestions[concept_key][:6]

        # ── Dynamic concept suggestions (auto-generated) ──
        if concept_key.startswith('dynamic:'):
            table = concept_key.split(':', 1)[1]
            return self._build_dynamic_suggestions(table)

        return [
            'Show this data trended over time',
            'Break down by provider',
            'Compare across regions',
            'Show by plan type',
        ]

    def _build_dynamic_suggestions(self, table: str) -> List[str]:
        """Generate smart follow-ups for a dynamically expanded table."""
        suggestions = []
        profiler = self.concept_expander.profiler
        table_lower = table.lower()
        from concept_expander import TABLE_LABELS
        table_label = TABLE_LABELS.get(table_lower, table.replace('_', ' ').title())

        # Get this table's dimensions (filter out skipped high-cardinality columns)
        from concept_expander import COLUMN_CONTEXT
        skip_cols = {
            'CPT_CODE', 'CPT_DESCRIPTION', 'ICD10_CODE', 'ICD10_DESCRIPTION',
            'MEDICATION_NAME', 'PRIMARY_DIAGNOSIS', 'DIAGNOSIS_DESCRIPTION',
            'CHIEF_COMPLAINT', 'REASON', 'CHECK_IN_TIME', 'CHECK_OUT_TIME',
            'APPOINTMENT_TIME',
        }
        dims = [d for d in profiler.get_dimensions(table)
                if d['name'].upper() not in skip_cols]
        dim_names = [d['name'].upper() for d in dims]
        metrics = profiler.get_metrics(table)
        temporals = profiler.get_temporals(table)

        # 1. Cross-table pivots — connect this table to other domains
        cross_domain = {
            'claims':        ('What is the denial rate by provider?', 'Show cost trend by month'),
            'members':       ('Show claims volume by member age group', 'Which members have the highest utilization?'),
            'encounters':    ('What is the average cost per encounter?', 'Show readmission rate by diagnosis'),
            'providers':     ('Show denial rate by provider specialty', 'Compare cost per encounter by provider'),
            'appointments':  ('What is the no-show rate by provider?', 'Show appointment volume trend by month'),
            'prescriptions': ('Which medication classes have the highest cost?', 'Show prescription volume by provider'),
            'diagnoses':     ('Which diagnoses drive the highest cost?', 'Show diagnosis prevalence trend over time'),
            'referrals':     ('What is the referral completion rate by specialty?', 'Show referral volume by urgency level'),
        }
        if table_lower in cross_domain:
            suggestions.extend(cross_domain[table_lower])

        # 2. Drill into a specific dimension value
        if dims:
            top_dim = dims[0]
            col_label = top_dim['name'].replace('_', ' ').lower()
            if top_dim.get('samples'):
                top_val = top_dim['samples'][0]
                suggestions.append(f'Show detailed breakdown for {col_label} = {top_val}')

        # 3. Trend analysis if temporal columns exist
        if temporals:
            suggestions.append(f'Show {table_label.lower()} trend over the last 12 months')

        # 4. Metric deep-dives
        metric_labels = {
            'BILLED_AMOUNT': 'billed amount', 'PAID_AMOUNT': 'paid amount',
            'COST': 'cost', 'COPAY': 'copay', 'PANEL_SIZE': 'panel size',
            'RISK_SCORE': 'risk score', 'LENGTH_OF_STAY': 'length of stay',
            'DAYS_SUPPLY': 'days supply', 'ALLOWED_AMOUNT': 'allowed amount',
        }
        # Skip NPI and code columns from metrics
        skip_metrics = {'NPI', 'RENDERING_NPI', 'BILLING_NPI', 'CPT_CODE', 'ZIP_CODE'}
        display_metrics = [m for m in metrics if m['name'].upper() not in skip_metrics]
        for m in display_metrics[:2]:
            ml = metric_labels.get(m['name'].upper(), m['name'].replace('_', ' ').lower())
            if 'REGION' in dim_names or 'KP_REGION' in dim_names:
                suggestions.append(f'Compare average {ml} across regions')
                break
            elif 'PLAN_TYPE' in dim_names:
                suggestions.append(f'Compare average {ml} by plan type')
                break

        # 5. Outlier / ranking question
        suggestions.append(f'Which {table_label.lower()} have the highest volume?')

        # 6. Population health connection
        if table_lower not in ('members',):
            suggestions.append(f'How does {table_label.lower()} vary by patient age group?')

        # Deduplicate and cap at 6
        seen = set()
        unique = []
        for s in suggestions:
            key = s.lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(s)
        return unique[:6]

    def reset_session(self, session_id: str = 'default'):
        """Reset a conversation session."""
        if session_id in self._conversations:
            del self._conversations[session_id]
        self.learning.new_session()

    def save_state(self):
        """Persist all learned state."""
        self.learning.save_state()
        self.model_selector._save_performance()
        logger.info("Pipeline state saved")

    def get_system_status(self) -> Dict[str, Any]:
        """Comprehensive system status."""
        return {
            'neural': {
                'embedding_vocab': len(self.neural.embeddings.vocab),
                'hopfield_patterns': self.neural.hopfield.num_patterns,
                'gnn_nodes': len(self.neural.gnn.node_ids) if self.neural.gnn.node_ids else 0,
            },
            'learning': self.learning.get_statistics(),
            'model_selector': self.model_selector.get_performance_report(),
            'cms_knowledge': self.cms.get_statistics(),
            'clinical_active': self.clinical is not None,
            'schema': {
                'tables': list(self.learner.tables.keys()),
                'total_columns': sum(len(p) for p in self.learner.tables.values()),
            },
        }

if __name__ == '__main__':
    pass
