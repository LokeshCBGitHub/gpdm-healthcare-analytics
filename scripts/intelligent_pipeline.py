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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

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

    def __init__(self, db_path: str, catalog_dir: str = None,
                 nlp_mode: str = 'auto', neural_dim: int = 64):
        self.db_path = db_path
        self.catalog_dir = catalog_dir
        self.neural_dim = neural_dim
        self._init_start = time.time()

        self.data_dir = os.path.dirname(db_path)
        self.state_db = os.path.join(self.data_dir, 'intelligent_state.db')

        from semantic_sql_engine import SemanticSQLEngine
        self.sql_engine = SemanticSQLEngine(db_path, catalog_dir, nlp_mode=nlp_mode)
        self.learner = self.sql_engine.semantic.learner
        logger.info("SQL engine initialized")

        try:
            _idx_conn = sqlite3.connect(db_path)
            _idx_conn.execute('CREATE INDEX IF NOT EXISTS idx_diagnoses_member_encounter ON diagnoses(MEMBER_ID, ENCOUNTER_ID)')
            _idx_conn.execute('CREATE INDEX IF NOT EXISTS idx_claims_member_encounter ON claims(MEMBER_ID, ENCOUNTER_ID)')
            _idx_conn.commit()
            _idx_conn.close()
            logger.info("Cross-table JOIN indexes verified")
        except Exception as e:
            logger.debug("Index creation error (non-fatal): %s", e)

        from neural_intelligence import NeuralQueryUnderstanding
        self.neural = NeuralQueryUnderstanding(dim=neural_dim, db_path=db_path)
        state_loaded = self.neural.load_state(self.state_db)
        if state_loaded and self.neural.embeddings._trained:
            self.neural.gnn.build_from_schema(self.learner, self.neural.embeddings)
            self.neural.gnn.propagate()
            self.neural._initialized = True
            logger.info("Neural intelligence loaded from state: vocab=%d, dim=%d",
                        len(self.neural.embeddings.vocab), neural_dim)
        else:
            pretrain_docs = self._build_pretrain_corpus()
            self.neural.initialize(self.learner, documents=pretrain_docs)
            self.neural.save_state(self.state_db)
            logger.info("Neural intelligence trained fresh: vocab=%d, dim=%d",
                        len(self.neural.embeddings.vocab), neural_dim)
        logger.info("Neural intelligence initialized")

        from continuous_learning import ContinuousLearningEngine
        self.learning = ContinuousLearningEngine(self.neural, self.state_db)

        from model_selector import MetaModelSelector
        self.model_selector = MetaModelSelector(db_path=self.state_db)

        from cms_data_loader import CMSKnowledgeBase
        self.cms = CMSKnowledgeBase(os.path.join(self.data_dir, 'cms_knowledge.db'))
        self.cms.load_all()

        try:
            from reasoning_chain import ReasoningChain
            self.reasoning_chain_available = True
            logger.info("ReasoningChain available for transparency tracking")
        except ImportError:
            self.reasoning_chain_available = False
            logger.debug("ReasoningChain not available (optional)")

        try:
            from demographic_analytics import DemographicAnalytics
            self.demographic_analytics = DemographicAnalytics(self.db_path)
            logger.info("DemographicAnalytics engine initialized")
        except Exception as e:
            self.demographic_analytics = None
            logger.debug("DemographicAnalytics not available: %s", e)

        self.knowledge_graph = None
        try:
            from knowledge_graph import KnowledgeGraph
            self.knowledge_graph = KnowledgeGraph(self.db_path)
            self.knowledge_graph.discover_schema()
            self.knowledge_graph.discover_relationships()
            self.knowledge_graph.classify_column_semantics()
            kg_metrics = self.knowledge_graph.compute_metrics()
            logger.info("KnowledgeGraph initialized: coverage=%.1f%%, %d relationships, %d gaps",
                        kg_metrics.get('knowledge_coverage', 0) * 100,
                        kg_metrics.get('total_relationships', 0),
                        kg_metrics.get('gap_count', 0))
        except Exception as e:
            logger.warning("KnowledgeGraph unavailable: %s", e)

        self.reasoning_pre_validator = None
        try:
            from reasoning_chain import ReasoningPreValidator
            if self.knowledge_graph:
                self.reasoning_pre_validator = ReasoningPreValidator(self.knowledge_graph)
                logger.info("ReasoningPreValidator active — validates query direction before SQL generation")
            else:
                logger.debug("ReasoningPreValidator skipped — requires KnowledgeGraph")
        except Exception as e:
            logger.debug("ReasoningPreValidator unavailable: %s", e)

        self.learning_scorer = None
        try:
            from learning_scorer import LearningScorer
            self.learning_scorer = LearningScorer(self.data_dir)
            logger.info("LearningScorer initialized — tracking accuracy, gaps, retrain triggers")
        except Exception as e:
            logger.warning("LearningScorer unavailable: %s", e)

        self.executive_dashboard_engine = None
        try:
            from executive_dashboards import ExecutiveDashboardEngine
            self.executive_dashboard_engine = ExecutiveDashboardEngine(self.db_path)
            logger.info("ExecutiveDashboardEngine initialized — 7 KP Medicare dashboard types")
        except Exception as e:
            logger.warning("ExecutiveDashboardEngine unavailable: %s", e)

        self.self_healing_engine = None
        try:
            from self_healing import SelfHealingEngine
            self.self_healing_engine = SelfHealingEngine(self.db_path)
            gt = self.self_healing_engine.get_ground_truth()
            logger.info("SelfHealingEngine active — %d ground truth metrics, auto-correction enabled", len(gt))
        except Exception as e:
            logger.warning("SelfHealingEngine unavailable: %s", e)

        self.deep_query_understanding = None
        try:
            from deep_query_understanding import DeepQueryUnderstanding
            self.deep_query_understanding = DeepQueryUnderstanding()
            logger.info("DeepQueryUnderstanding active — semantic intent + entity extraction + SQL hints")
        except Exception as e:
            logger.warning("DeepQueryUnderstanding unavailable: %s", e)

        self.revenue_optimization = None
        try:
            from revenue_optimization import RevenueOptimizationEngine
            self.revenue_optimization = RevenueOptimizationEngine(self.db_path)
            logger.info("RevenueOptimizationEngine active — churn, PMPM, denial, HCC, retention analytics")
        except Exception as e:
            logger.warning("RevenueOptimizationEngine unavailable: %s", e)

        self.dashboard_renderer = None
        try:
            from dashboard_frontend import DashboardFrontendRenderer
            self.dashboard_renderer = DashboardFrontendRenderer(db_path=self.db_path)
            logger.info("DashboardFrontendRenderer active — visual HTML dashboard generation with deep insights")
        except Exception as e:
            logger.warning("DashboardFrontendRenderer unavailable: %s", e)

        self.anticipation_engine = None
        self.uncertainty_handler = None
        try:
            from anticipation_engine import AnticipationEngine, UncertaintyHandler
            self.anticipation_engine = AnticipationEngine(self.db_path)
            self.uncertainty_handler = UncertaintyHandler(self.db_path)
            logger.info("AnticipationEngine + UncertaintyHandler active — predictive suggestions, data confidence, disambiguation")
        except Exception as e:
            logger.warning("Anticipation/Uncertainty engines unavailable: %s", e)

        self.advanced_ml = None
        try:
            from advanced_ml_engine import AdvancedMLEngine
            self.advanced_ml = AdvancedMLEngine(self.db_path)
            logger.info("AdvancedMLEngine active — 8 state-of-art models (GBM, ARIMA, K-Means, Isolation Forest, Bayes, Survival, PCA, Ensemble)")
        except Exception as e:
            logger.warning("AdvancedMLEngine unavailable: %s", e)

        self.source_protect = None
        try:
            from source_protect import SourceProtect
            self.source_protect = SourceProtect()
            logger.info("SourceProtect active — dev tools disabled, code obfuscated")
        except Exception as e:
            logger.debug("SourceProtect unavailable: %s", e)

        self.clinical = None
        try:
            from tuva_clinical_layer import TuvaClinicalEnricher
            self.clinical = TuvaClinicalEnricher(db_path)
            if not self.clinical.is_healthcare:
                self.clinical = None
        except:
            pass

        self.query_intelligence = None
        try:
            from query_intelligence import QueryIntelligence
            self.query_intelligence = QueryIntelligence(db_path)
            logger.info("QueryIntelligence active — schema linking + intent decomposition + SQL assembly")
        except Exception as e:
            logger.warning("QueryIntelligence unavailable: %s", e)

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

        from deep_intelligence import (
            BenchmarkEngine, MultiReasoningEngine,
            PrecursorInsightEngine, DataGapDetector, RootCauseAnalyzer
        )
        _dc = getattr(self.sql_engine, 'domain_config', None)
        self.benchmark_engine = BenchmarkEngine(domain_config=_dc)
        self.multi_reasoning = MultiReasoningEngine(domain_config=_dc)
        self.precursor_insights = PrecursorInsightEngine(db_path, domain_config=_dc)
        self.data_gap_detector = DataGapDetector(db_path)
        self.root_cause_analyzer = RootCauseAnalyzer(db_path)

        from statistical_models import StatisticalAnalyzer
        self.statistical_analyzer = StatisticalAnalyzer()

        from concept_expander import ConceptExpander
        self.concept_expander = ConceptExpander(db_path)

        self.analytics_engine = None
        try:
            from healthcare_analytics import HealthcareAnalyticsEngine
            self.analytics_engine = HealthcareAnalyticsEngine(db_path)
            self.analytics_engine.train_all()
            logger.info("Healthcare analytics models trained")
        except Exception as e:
            logger.warning("Healthcare analytics models unavailable: %s", e)

        logger.info("Advanced analytics + deep intelligence + statistical models + concept expander initialized")

        self.orchestrator = None
        try:
            from analytics_orchestrator import AnalyticsOrchestrator
            self.orchestrator = AnalyticsOrchestrator(db_path)
            engines = self.orchestrator.get_available_engines()
            ok_engines = [k for k, v in engines.items() if v]
            logger.info("Analytics orchestrator initialized: %d/%d engines available (%s)",
                        len(ok_engines), len(engines), ', '.join(ok_engines))
        except Exception as e:
            logger.warning("Analytics orchestrator unavailable: %s", e)

        self.analytical_intelligence = None
        try:
            from analytical_intelligence import AnalyticalIntelligence
            self.analytical_intelligence = AnalyticalIntelligence(db_path)
            logger.info("AnalyticalIntelligence active — multi-query decomposition + insight generation + dashboards")
        except Exception as e:
            logger.warning("AnalyticalIntelligence unavailable: %s", e)

        self.column_graph = getattr(self.learner, 'column_graph', None)
        self.cooccurrence_matrix = getattr(self.learner, 'cooccurrence_matrix', None)
        if self.column_graph:
            logger.info("Column graph: %d relationships discovered",
                        len(self.column_graph.edges))

        self.cache_manager = None
        try:
            from cache_manager import CacheManager
            self.cache_manager = CacheManager(data_dir=self.data_dir, db_path=db_path)
            logger.info("3-tier cache manager initialized")
        except Exception as e:
            logger.warning("Cache manager unavailable: %s", e)

        self.query_tracker = None
        try:
            from query_tracker import QueryTracker
            self.query_tracker = QueryTracker(data_dir=self.data_dir)
            logger.info("Query tracker initialized")
        except Exception as e:
            logger.warning("Query tracker unavailable: %s", e)

        self.forecast_runner = None
        try:
            from forecast_runner import ForecastRunner
            self.forecast_runner = ForecastRunner(forecast_horizon=6)
            logger.info("Multi-model forecast runner initialized")
        except Exception as e:
            logger.warning("Forecast runner unavailable: %s", e)

        self.advanced_nlp = None
        try:
            from advanced_nlp_engine import AdvancedNLPEngine
            self.advanced_nlp = AdvancedNLPEngine()
            corpus_docs = []
            for table, profiles in self.learner.tables.items():
                for p in profiles:
                    corpus_docs.append(f"{table} {p.name.replace('_', ' ')} {' '.join(p.semantic_tags)}")
            self.advanced_nlp._corpus_docs = corpus_docs
            logger.info("Advanced NLP engine initialized (BM25, NER, stemming, %d docs)", len(corpus_docs))
        except Exception as e:
            logger.warning("Advanced NLP engine unavailable: %s", e)

        self.builtin_nlu = None
        try:
            from builtin_nlu import BuiltInNLU
            self.builtin_nlu = BuiltInNLU(self.learner, db_path=db_path)
            logger.info("Built-in NLU ACTIVE: own embedder + MLP classifier + "
                        "schema entity extractor — no external LLMs needed")
        except Exception as e:
            logger.warning("Built-in NLU unavailable: %s", e)

        self.semantic_graph = None
        self.intent_parser = None
        self.sql_constructor = None
        self.conversation_context = None
        self.healthcare_llm_pipeline = None
        try:
            from schema_graph import SemanticSchemaGraph
            from sql_constructor import SQLConstructor
            self.semantic_graph = SemanticSchemaGraph(db_path)

            llm_config = {}
            if hasattr(self, 'cfg') and self.cfg:
                llm_config = {k: v for k, v in self.cfg.items()
                              if k.startswith('LLM_')}
            try:
                from healthcare_llm_pipeline import create_healthcare_llm_pipeline
                self.healthcare_llm_pipeline = create_healthcare_llm_pipeline(
                    db_path=db_path,
                    schema_graph=self.semantic_graph,
                    config=llm_config,
                )
                _llm_status = "active" if self.healthcare_llm_pipeline.llm_available else "built-in intelligence"
                logger.info("Healthcare LLM Pipeline: %s (NLU + SQL + Narrative)", _llm_status)
            except Exception as llm_pipe_e:
                logger.info("Healthcare LLM Pipeline not available (%s)", llm_pipe_e)

            try:
                from local_llm_engine import create_intent_parser, ConversationContext
                self.intent_parser = create_intent_parser(
                    self.semantic_graph, config=llm_config
                )
                self.conversation_context = ConversationContext()
                _parser_type = type(self.intent_parser).__name__
                logger.info("Intent parser: %s (LLM toggle available)", _parser_type)
            except Exception as llm_e:
                logger.info("Local LLM not available (%s), using rule-based parser", llm_e)
                from intent_parser import IntentParser
                self.intent_parser = IntentParser(self.semantic_graph)

            self.sql_constructor = SQLConstructor(self.semantic_graph, db_path)
            _n_tables = len(self.semantic_graph.tables)
            _n_cols = sum(len(v) for v in self.semantic_graph.columns.values())
            _n_joins = len(self.semantic_graph.join_paths)
            logger.info("Semantic Intelligence ACTIVE: %d tables, %d columns, %d JOIN paths — "
                        "true NLU + schema graph + SQL reasoning", _n_tables, _n_cols, _n_joins)
        except Exception as e:
            logger.warning("Semantic Intelligence unavailable: %s", e)

        self.semantic_embedder = None
        try:
            from semantic_embedder import SemanticEmbedder, get_embedder
            self.semantic_embedder = get_embedder()
            logger.info("SemanticEmbedder ACTIVE: %d-dim embeddings, %d concept clusters — "
                        "replaces TF-IDF for all similarity matching",
                        self.semantic_embedder.embed_dim, len(self.semantic_embedder._cluster_centers))
        except Exception as e:
            logger.warning("SemanticEmbedder unavailable: %s", e)

        self.language_brain = None
        try:
            if self.semantic_graph and self.intent_parser:
                from language_brain import create_intelligent_parser
                _intelligent = create_intelligent_parser(
                    self.semantic_graph,
                    embedder=self.semantic_embedder
                )
                self._regex_parser_backup = self.intent_parser
                self.intent_parser = _intelligent
                self.language_brain = _intelligent.brain
                logger.info("LanguageBrain ACTIVE: trained semantic understanding — "
                            "%d intent centroids, %d phrase→column mappings, "
                            "runs BEFORE regex as primary NLU",
                            len(_intelligent.brain._intent_centroids),
                            len(_intelligent.brain._phrase_column_index))
        except Exception as e:
            logger.warning("LanguageBrain unavailable: %s — using regex parser only", e)

        self.healthcare_analyst = None
        try:
            from healthcare_analyst import HealthcareFinancialAnalyst, get_analyst
            self.healthcare_analyst = get_analyst()
            logger.info("HealthcareFinancialAnalyst ACTIVE: %d analytical concepts — "
                        "domain-specific reasoning for healthcare/finance queries",
                        len(self.healthcare_analyst.concepts))
        except Exception as e:
            logger.warning("HealthcareFinancialAnalyst unavailable: %s", e)

        self.domain_intelligence = None
        try:
            from domain_intelligence import create_domain_intelligence
            self.domain_intelligence = create_domain_intelligence()
            logger.info("Domain Intelligence ACTIVE: 99 jargon terms, 9 concept nodes, "
                        "8 KPI definitions, 7 benchmarks, causal reasoning")
        except Exception as e:
            logger.info("Domain Intelligence not available: %s", e)

        self.healthcare_transformer = None
        try:
            from healthcare_transformer import HealthcareTransformerEngine
            self.healthcare_transformer = HealthcareTransformerEngine(
                db_path=db_path,
                schema_graph=self.semantic_graph if self.semantic_graph else None,
            )
            param_count = self.healthcare_transformer.transformer.get_param_count()
            logger.info("Healthcare Transformer ACTIVE: %dK params, "
                        "d_model=128, n_heads=8, n_layers=3 (Llama 3 / Phi-3 architecture)",
                        param_count // 1000)
        except Exception as e:
            logger.info("Healthcare Transformer not available: %s", e)

        self.local_llm = None
        self.intelligent_nlu = None
        logger.info("Air-gap mode: ALL intelligence runs in-process — zero network calls")

        self.gpdm_definitions = None
        try:
            from gpdm_definitions import GPDMDefinitionEngine
            self.gpdm_definitions = GPDMDefinitionEngine(catalog_dir)
            logger.info("GPDM Definition Engine initialized (primary intelligence source)")
        except Exception as e:
            logger.warning("GPDM Definition Engine unavailable: %s", e)

        self.insight_engine = None
        try:
            from insight_engine import InsightEngine
            self.insight_engine = InsightEngine()
            logger.info("Insight Engine initialized (statistical, Bayesian, Monte Carlo, Markov)")
        except Exception as e:
            logger.warning("Insight Engine unavailable: %s", e)


        self._perf_report = None
        try:
            from perf_indexes import optimize_db, apply_runtime_pragmas
            self._perf_report = optimize_db(db_path)
            idx_ok = len(self._perf_report.get('indexes_created', []))
            idx_skip = len(self._perf_report.get('indexes_skipped', []))
            logger.info("perf_indexes: %d indexes created, %d skipped, WAL+mmap active",
                        idx_ok, idx_skip)
        except Exception as e:
            logger.warning("perf_indexes unavailable (DB still works, just slower): %s", e)

        self.live_cache = None
        try:
            from live_cache import LiveCache
            self.live_cache = LiveCache(default_ttl=120.0)
            logger.info("LiveCache initialized (TTL=120s, data-version invalidation)")
        except Exception as e:
            logger.warning("LiveCache unavailable: %s", e)

        self.kpi_cube = None
        self._kpi_cube_stats = None
        try:
            from kpi_cube import refresh_cube
            self._kpi_cube_stats = refresh_cube(db_path, full=False)
            self.kpi_cube = True
            rows_built = self._kpi_cube_stats.get('rows_in_cube', 0)
            logger.info("KPI cube refreshed: %d member-month rows", rows_built)
        except Exception as e:
            logger.warning("KPI cube unavailable (KPIs still work, just slower): %s", e)

        self._executive_kpis = None
        try:
            from executive_kpis import compute_executive_kpis
            if self.live_cache:
                self._executive_kpis = self.live_cache.get_or_compute(
                    'executive_kpis', (db_path,),
                    lambda: compute_executive_kpis(db_path),
                    db_path=db_path,
                    tables=['encounters', 'claims', 'members'],
                    ttl=300.0,
                )
            else:
                self._executive_kpis = compute_executive_kpis(db_path)
            kpi_count = len(self._executive_kpis.get('kpis', {})) if isinstance(self._executive_kpis, dict) else 0
            logger.info("Executive KPIs computed: %d metrics", kpi_count)
        except Exception as e:
            logger.warning("Executive KPIs unavailable: %s", e)

        self.answer_cache = None
        try:
            from answer_cache import AnswerCache
            self.answer_cache = AnswerCache(data_dir=self.data_dir)
            logger.info("AnswerCache initialized (similarity threshold=%.2f, max=%d entries)",
                        self.answer_cache.cfg.threshold, self.answer_cache.cfg.max_entries)
        except Exception as e:
            logger.warning("AnswerCache unavailable (in-memory cache still works): %s", e)

        self._intent_context_available = False
        try:
            from intent_context import classify_intent as _ic_test
            self._intent_context_available = True
            logger.info("IntentContext available (persona + anticipation + intent classification)")
        except Exception as e:
            logger.warning("IntentContext unavailable: %s", e)

        self.retrieval_index = None
        try:
            from retrieval import get_index
            self.retrieval_index = get_index(db_path)
            logger.info("Retrieval index initialized")
        except Exception as e:
            logger.warning("Retrieval index unavailable: %s", e)

        self.reranker = None
        try:
            from reranker import Reranker
            self.reranker = Reranker()
            logger.info("Reranker initialized (backend=%s)", getattr(self.reranker, '_backend_name', 'unknown'))
        except Exception as e:
            logger.warning("Reranker unavailable: %s", e)

        self.nl2sql_grounder = None
        try:
            from nl2sql_grounded import get_grounder
            schema_reg = getattr(self.sql_engine, 'semantic', None)
            if schema_reg:
                self.nl2sql_grounder = get_grounder(schema_reg)
                logger.info("NL2SQL grounder initialized (schema-aware table/column grounding)")
        except Exception as e:
            logger.warning("NL2SQL grounder unavailable: %s", e)

        self._business_insights_available = False
        try:
            from business_insights import rising_risk_cohort as _bi_test
            self._business_insights_available = True
            logger.info("BusinessInsights available (rising_risk, readmit_watchlist, hedis_gaps, network_perf)")
        except Exception as e:
            logger.warning("BusinessInsights unavailable: %s", e)

        self.kpi_discovery = None
        try:
            from kpi_discovery import get_discovery
            self.kpi_discovery = get_discovery(self.data_dir)
            logger.info("KPI Discovery initialized")
        except Exception as e:
            logger.warning("KPI Discovery unavailable: %s", e)

        self._calibration_available = False
        try:
            from calibration import calibrate_risk_from_sqlite as _cal_test
            self._calibration_available = True
            logger.info("Calibration available (Platt, isotonic, temperature scaling)")
        except Exception as e:
            logger.warning("Calibration unavailable: %s", e)

        self._explainability_available = False
        try:
            from explainability import permutation_importance as _expl_test
            self._explainability_available = True
            logger.info("Explainability available (integrated gradients, SHAP, permutation importance)")
        except Exception as e:
            logger.warning("Explainability unavailable: %s", e)

        self._drift_monitor_available = False
        try:
            from drift_monitor import should_retrain as _dm_test
            self._drift_monitor_available = True
            logger.info("DriftMonitor available (PSI, KS, chi-square, Jensen-Shannon)")
        except Exception as e:
            logger.warning("DriftMonitor unavailable: %s", e)

        self.forecast_adapter = None
        try:
            from forecast_adapters import get_backend
            self.forecast_adapter = get_backend()
            logger.info("ForecastAdapter initialized (backend=%s, available=%s)",
                        getattr(self.forecast_adapter, 'name', 'unknown'),
                        getattr(self.forecast_adapter, 'available', False))
        except Exception as e:
            logger.warning("ForecastAdapter unavailable: %s", e)


        self.deep_engine = None
        try:
            from deep_understanding import DeepUnderstandingEngine
            self.deep_engine = DeepUnderstandingEngine(
                schema_learner=self.learner,
                db_path=db_path,
                data_dir=self.data_dir,
            )
            logger.info("Deep Understanding Engine: vocab=%d, dim=%d, concepts=%d — "
                        "FULLY IN-PROCESS, zero network calls",
                        self.deep_engine.vocab_size,
                        self.deep_engine.embedding_dim,
                        self.deep_engine.concept_count)
        except Exception as e:
            logger.warning("Deep Understanding Engine unavailable: %s", e)

        shared_embedder = None
        if self.deep_engine and hasattr(self.deep_engine, '_embedder'):
            shared_embedder = self.deep_engine._embedder
        self.sql_reasoner = None
        try:
            from sql_reasoning import SQLReasoningEngine
            self.sql_reasoner = SQLReasoningEngine(
                schema_learner=self.learner,
                db_path=db_path,
                embedder=shared_embedder,
            )
            logger.info("SQL Reasoning Engine: %d patterns, %d templates — in-process",
                        self.sql_reasoner.pattern_count,
                        self.sql_reasoner.template_count)
        except Exception as e:
            logger.warning("SQL Reasoning Engine unavailable: %s", e)

        self.pre_sql_validator = None
        try:
            from pre_sql_validator import PreSQLValidator
            self.pre_sql_validator = PreSQLValidator(db_path=db_path)
            logger.info("Pre-SQL Validator initialized — %d tables profiled",
                        len(self.pre_sql_validator._table_columns))
        except Exception as e:
            logger.warning("Pre-SQL Validator unavailable: %s", e)

        self.quality_gate = None
        try:
            from sql_quality_gate import SQLQualityGate
            self.quality_gate = SQLQualityGate(db_path=db_path, schema_learner=self.learner)
            logger.info("SQL Quality Gate initialized — %d table profiles, %d learned corrections",
                        len(self.quality_gate._table_columns),
                        len(self.quality_gate._learned_corrections))
        except Exception as e:
            logger.warning("SQL Quality Gate unavailable: %s", e)

        self.narrative_intelligence = None
        try:
            from narrative_intelligence import NarrativeIntelligence
            self.narrative_intelligence = NarrativeIntelligence(db_path=db_path)
            logger.info("Narrative Intelligence initialized — persona-aware, benchmark-driven")
        except Exception as e:
            logger.warning("Narrative Intelligence unavailable: %s", e)


        self.schema_mapper = None
        try:
            from schema_mapper import SchemaMapper
            self.schema_mapper = SchemaMapper(llm_backend=None)
            logger.info("Schema Mapper initialized (air-gapped, pattern-based)")
        except Exception as e:
            logger.warning("Schema Mapper unavailable: %s", e)

        self.schema_merger = None
        try:
            from schema_merger import SchemaMerger
            self.schema_merger = SchemaMerger(db_path)
            logger.info("Schema Merger initialized")
        except Exception as e:
            logger.warning("Schema Merger unavailable: %s", e)

        self.hot_reload = None
        try:
            from hot_reload import HotReloadController
            self.hot_reload = HotReloadController(self)
            logger.info("Hot Reload Controller initialized — no restart needed for schema changes")
        except Exception as e:
            logger.warning("Hot Reload Controller unavailable: %s", e)


        self._conversations: Dict[str, Dict] = {}

        init_time = time.time() - self._init_start
        wired_count = sum([
            self._perf_report is not None,
            self.live_cache is not None,
            self.kpi_cube is not None,
            self._executive_kpis is not None,
            self.answer_cache is not None,
            self._intent_context_available,
            self.retrieval_index is not None,
            self.reranker is not None,
            self.nl2sql_grounder is not None,
            self._business_insights_available,
            self.kpi_discovery is not None,
            self._calibration_available,
            self._explainability_available,
            self._drift_monitor_available,
            self.forecast_adapter is not None,
            self.deep_engine is not None,
            self.sql_reasoner is not None,
            self.narrative_intelligence is not None,
            self.schema_mapper is not None,
            self.schema_merger is not None,
            self.hot_reload is not None,
        ])
        logger.info("Intelligent pipeline ready in %.1fs — %d/21 intelligence modules wired "
                     "(FULLY AIR-GAPPED — zero network calls)",
                     init_time, wired_count)

    def _build_pretrain_corpus(self) -> List[str]:
        docs = []

        try:
            from cms_data_loader import build_healthcare_vocabulary
            docs.extend(build_healthcare_vocabulary())
        except Exception:
            pass

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

    _GOLDEN_TEMPLATES = [
        {
            'patterns': [r'^(?:how\s+many|total(?:\s+number\s+of)?|count(?:\s+of)?)\s+claims(?:\s+are\s+there)?[\s?]*$',
                         r'^(?:how\s+many|total)\s+claims\s+(?:do\s+we\s+have|exist|in\s+(?:the\s+)?(?:system|database|db))'],
            'sql': "SELECT COUNT(*) AS total FROM claims",
        },
        {
            'patterns': [r'^(?:how\s+many|total(?:\s+number\s+of)?|count(?:\s+of)?)\s+members(?:\s+are\s+there)?[\s?]*$',
                         r'^(?:how\s+many|total)\s+members\s+(?:do\s+we\s+have|exist|in\s+(?:the\s+)?(?:system|database|db))'],
            'sql': "SELECT COUNT(*) AS total FROM members",
        },
        {
            'patterns': [r'^(?:how\s+many|total(?:\s+number\s+of)?|count(?:\s+of)?)\s+encounters(?:\s+are\s+there)?[\s?]*$',
                         r'^(?:how\s+many|total)\s+encounters\s+(?:do\s+we\s+have|exist|in\s+(?:the\s+)?(?:system|database|db))'],
            'sql': "SELECT COUNT(*) AS total FROM encounters",
        },
        {
            'patterns': [r'(?:primary|main|top|common|biggest)\s+reason.*(?:reject|denied|denial)',
                         r'reason.*claim.*(?:reject|denied|denial)',
                         r'why.*claim.*(?:reject|denied|get\s+denied)',
                         r'(?:reject|denied|denial).*reason'],
            'sql': (
                "SELECT ICD10_CODE, ICD10_DESCRIPTION, PLAN_TYPE, "
                "COUNT(*) as denied_count "
                "FROM claims "
                "WHERE CLAIM_STATUS = 'DENIED' AND ICD10_CODE IS NOT NULL "
                "GROUP BY ICD10_CODE, ICD10_DESCRIPTION, PLAN_TYPE "
                "ORDER BY denied_count DESC LIMIT 20"
            ),
        },
        {
            'patterns': [r'(?:how\s+frequent|how\s+often).*(?:reject|denied).*(?:by\s+(?:the\s+)?plan|plan)',
                         r'(?:reject|denied|denial).*(?:rate|frequency).*(?:by\s+plan|per\s+plan|across.*plan)',
                         r'submission.*(?:reject|denied).*(?:by\s+(?:the\s+)?plan|plan)'],
            'sql': (
                "SELECT PLAN_TYPE, COUNT(*) AS total_claims, "
                "SUM(CASE WHEN CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) AS denied_count, "
                "ROUND(100.0 * SUM(CASE WHEN CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) AS denial_rate_pct "
                "FROM claims GROUP BY PLAN_TYPE ORDER BY denial_rate_pct DESC"
            ),
        },
        {
            'patterns': [r'^denial\s*rate$', r'^what.*denial\s*rate$', r'^overall.*denial\s*rate',
                         r'^claim.*denial\s*rate$', r'^what.*(?:is|are).*claim.*denied',
                         r'^how\s+many.*claims?\s+(?:are|were|have been)\s+denied$',
                         r'^percent.*denied\s*claims'],
            'sql': (
                "SELECT COUNT(*) as total_claims, "
                "SUM(CASE WHEN CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) as denied_claims, "
                "ROUND(100.0 * SUM(CASE WHEN CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as denial_rate_pct, "
                "ROUND(100.0 * SUM(CASE WHEN CLAIM_STATUS = 'PAID' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as paid_rate_pct, "
                "ROUND(100.0 * SUM(CASE WHEN CLAIM_STATUS = 'PENDING' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as pending_rate_pct "
                "FROM claims"
            ),
        },
        {
            'patterns': [r'average.*paid\s*amount', r'avg.*paid\s*amount',
                         r'mean.*paid\s*amount', r'average.*payment',
                         r'avg.*payment', r'average.*reimbursement'],
            'sql': (
                "SELECT ROUND(AVG(PAID_AMOUNT), 2) as avg_paid_amount, "
                "ROUND(AVG(BILLED_AMOUNT), 2) as avg_billed_amount, "
                "ROUND(AVG(ALLOWED_AMOUNT), 2) as avg_allowed_amount, "
                "COUNT(*) as total_claims "
                "FROM claims"
            ),
        },
        {
            'patterns': [r'average.*billed\s*amount', r'avg.*billed\s*amount',
                         r'mean.*billed\s*amount', r'average.*charge'],
            'sql': (
                "SELECT ROUND(AVG(BILLED_AMOUNT), 2) as avg_billed_amount, "
                "ROUND(AVG(ALLOWED_AMOUNT), 2) as avg_allowed_amount, "
                "ROUND(AVG(PAID_AMOUNT), 2) as avg_paid_amount, "
                "COUNT(*) as total_claims "
                "FROM claims"
            ),
        },
        {
            'patterns': [r'total.*billed.*paid', r'total.*billed.*allowed',
                         r'billed.*vs.*paid', r'billed.*versus.*paid',
                         r'sum.*billed.*paid', r'compare.*billed.*paid'],
            'sql': (
                "SELECT ROUND(SUM(BILLED_AMOUNT), 2) as total_billed, "
                "ROUND(SUM(ALLOWED_AMOUNT), 2) as total_allowed, "
                "ROUND(SUM(PAID_AMOUNT), 2) as total_paid, "
                "ROUND(SUM(MEMBER_RESPONSIBILITY), 2) as total_member_responsibility, "
                "COUNT(*) as total_claims "
                "FROM claims"
            ),
        },
        {
            'patterns': [r'^(?!.*(?:specialty|most expensive|most costly|provider cost|age\s*group|by\s*region|highest|lowest)).*average.*cost.*per\s*member(?!.*(?:chronic|specialty|provider|age|region))',
                         r'^(?!.*(?:specialty|most expensive|most costly|provider cost|age\s*group|by\s*region|highest|lowest)).*avg.*cost.*per\s*member(?!.*(?:chronic|specialty|provider|age|region))',
                         r'^(?!.*(?:specialty|most expensive|most costly|provider cost|age\s*group|by\s*region|highest|lowest|which\s*region)).*cost.*per\s*member(?!.*(?:chronic|specialty|provider|age|region))',
                         r'pmpm', r'per\s*member.*per\s*month',
                         r'^(?!.*(?:specialty|provider|age|region)).*per\s*member.*cost(?!.*(?:chronic|specialty|provider|age|region))',
                         r'^(?!.*(?:specialty|provider|age|region)).*spend.*per\s*member(?!.*(?:chronic|specialty|provider|age|region))'],
            'sql': (
                "SELECT ROUND(SUM(PAID_AMOUNT) * 1.0 / NULLIF(COUNT(DISTINCT MEMBER_ID), 0), 2) as cost_per_member, "
                "ROUND(SUM(BILLED_AMOUNT) * 1.0 / NULLIF(COUNT(DISTINCT MEMBER_ID), 0), 2) as billed_per_member, "
                "COUNT(DISTINCT MEMBER_ID) as unique_members, "
                "COUNT(*) as total_claims "
                "FROM claims"
            ),
        },
        {
            'patterns': [r'cost.*per\s*member.*chronic', r'cost.*member.*chronic\s*condition',
                         r'spend.*by\s*chronic', r'cost.*by\s*chronic',
                         r'chronic.*cost.*per\s*member'],
            'sql': (
                "SELECT m.CHRONIC_CONDITIONS, "
                "COUNT(DISTINCT m.MEMBER_ID) as member_count, "
                "ROUND(AVG(c.PAID_AMOUNT), 2) as avg_paid_per_claim, "
                "ROUND(SUM(c.PAID_AMOUNT) * 1.0 / NULLIF(COUNT(DISTINCT m.MEMBER_ID), 0), 2) as cost_per_member "
                "FROM claims c JOIN members m ON c.MEMBER_ID = m.MEMBER_ID "
                "GROUP BY m.CHRONIC_CONDITIONS "
                "ORDER BY m.CHRONIC_CONDITIONS"
            ),
        },
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
        {
            'patterns': [r'sum.*deductible', r'total.*deductible',
                         r'deductible.*sum', r'deductible.*total'],
            'sql': (
                "SELECT ROUND(SUM(CAST(DEDUCTIBLE AS REAL)), 2) as total_deductibles, "
                "ROUND(AVG(CAST(DEDUCTIBLE AS REAL)), 2) as avg_deductible, "
                "COUNT(*) as total_claims "
                "FROM claims"
            ),
        },
        {
            'patterns': [r'average.*copay.*plan', r'avg.*copay.*plan',
                         r'copay.*by\s*plan', r'copay.*per\s*plan'],
            'sql': (
                "SELECT PLAN_TYPE, ROUND(AVG(COPAY), 2) as avg_copay, "
                "ROUND(AVG(COINSURANCE), 2) as avg_coinsurance, "
                "ROUND(AVG(DEDUCTIBLE), 2) as avg_deductible, "
                "COUNT(*) as total_claims "
                "FROM claims GROUP BY PLAN_TYPE ORDER BY avg_copay DESC"
            ),
        },
        {
            'patterns': [r'yield\s*rate', r'payment.*yield', r'paid.*allowed.*ratio',
                         r'reimbursement.*rate'],
            'sql': (
                "SELECT PLAN_TYPE, "
                "ROUND(SUM(PAID_AMOUNT) * 100.0 / NULLIF(SUM(ALLOWED_AMOUNT), 0), 2) as yield_rate_pct, "
                "ROUND(SUM(PAID_AMOUNT), 2) as total_paid, "
                "ROUND(SUM(ALLOWED_AMOUNT), 2) as total_allowed "
                "FROM claims GROUP BY PLAN_TYPE ORDER BY yield_rate_pct DESC"
            ),
        },

        {
            'patterns': [r'provider.*denial\s*rate', r'denial\s*rate.*provider',
                         r'provider.*highest.*denial', r'which.*provider.*denial'],
            'sql': (
                "SELECT c.RENDERING_NPI as provider_npi, "
                "p.PROVIDER_FIRST_NAME || ' ' || p.PROVIDER_LAST_NAME as provider_name, "
                "p.SPECIALTY, COUNT(*) as total_claims, "
                "SUM(CASE WHEN c.CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) as denied_claims, "
                "ROUND(100.0 * SUM(CASE WHEN c.CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as denial_rate_pct "
                "FROM claims c LEFT JOIN providers p ON c.RENDERING_NPI = p.NPI "
                "GROUP BY c.RENDERING_NPI, provider_name, p.SPECIALTY "
                "HAVING COUNT(*) >= 5 "
                "ORDER BY denial_rate_pct DESC LIMIT 20"
            ),
        },
        {
            'patterns': [r'denial\s*rate.*region', r'region.*denial\s*rate',
                         r'denied.*by\s*region', r'denial.*by\s*region'],
            'sql': (
                "SELECT KP_REGION as region, COUNT(*) as total_claims, "
                "SUM(CASE WHEN CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) as denied_claims, "
                "ROUND(100.0 * SUM(CASE WHEN CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as denial_rate_pct "
                "FROM claims GROUP BY KP_REGION ORDER BY denial_rate_pct DESC"
            ),
        },
        {
            'patterns': [r'denial\s*rate.*specialty', r'specialty.*denial\s*rate',
                         r'denied.*by\s*specialty'],
            'sql': (
                "SELECT p.SPECIALTY, COUNT(*) as total_claims, "
                "SUM(CASE WHEN c.CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) as denied_claims, "
                "ROUND(100.0 * SUM(CASE WHEN c.CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as denial_rate_pct "
                "FROM claims c JOIN providers p ON c.RENDERING_NPI = p.NPI "
                "GROUP BY p.SPECIALTY HAVING COUNT(*) >= 10 ORDER BY denial_rate_pct DESC"
            ),
        },
        {
            'patterns': [r'denied.*by\s*reason', r'denied.*denial\s*reason',
                         r'denial\s*reason.*breakdown', r'why.*denied',
                         r'reason.*denied', r'denial\s*reason'],
            'sql': (
                "SELECT DENIAL_REASON, COUNT(*) as denied_count, "
                "ROUND(100.0 * COUNT(*) / NULLIF((SELECT COUNT(*) FROM claims WHERE CLAIM_STATUS = 'DENIED'), 0), 1) as pct "
                "FROM claims WHERE CLAIM_STATUS = 'DENIED' AND DENIAL_REASON IS NOT NULL "
                "GROUP BY DENIAL_REASON ORDER BY denied_count DESC"
            ),
        },
        {
            'patterns': [r'approval\s*rate.*plan', r'plan.*approval\s*rate',
                         r'^(?!.*referral).*claim.*approval\s*rate',
                         r'^(?!.*referral).*approval\s*rate.*(?:claim|plan|by\s*plan)'],
            'sql': (
                "SELECT PLAN_TYPE as plan, COUNT(*) as total_claims, "
                "SUM(CASE WHEN CLAIM_STATUS = 'PAID' THEN 1 ELSE 0 END) as approved, "
                "ROUND(100.0 * SUM(CASE WHEN CLAIM_STATUS = 'PAID' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as approval_rate_pct "
                "FROM claims GROUP BY PLAN_TYPE ORDER BY approval_rate_pct DESC"
            ),
        },

        {
            'patterns': [r'^(?!.*cost).*member.*age\s*group', r'age\s*group.*member(?!.*cost)',
                         r'count.*age\s*group', r'^(?!.*cost).*patient.*age\s*group',
                         r'^(?!.*cost).*age.*distribution', r'^(?!.*cost).*age.*breakdown'],
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
        {
            'patterns': [r'^how\s+many\s+members\b(?!.*(?:by|per|each|plan|region|age|type|die|died|death|visit|doctor|encounter|never|zero|disenroll|leave|churn))',
                         r'total.*member\s*count', r'^total\s+members$',
                         r'^how\s+many\s+patients\b(?!.*(?:by|per|each|plan|region|age|type|die|died|death|visit|doctor|encounter|never|zero|disenroll|leave|churn))',
                         r'total.*patient\s*count'],
            'sql': (
                "SELECT COUNT(*) as total_members FROM members"
            ),
        },
        {
            'patterns': [r'member.*by\s*gender', r'member.*gender.*breakdown',
                         r'gender.*distribution', r'gender.*demographics',
                         r'patient.*by\s*gender', r'demographic.*gender'],
            'sql': (
                "SELECT GENDER, COUNT(*) as member_count, "
                "ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM members), 1) as pct "
                "FROM members GROUP BY GENDER ORDER BY member_count DESC"
            ),
        },
        {
            'patterns': [r'risk\s*score.*plan', r'risk.*by\s*plan',
                         r'risk\s*score.*by\s*plan'],
            'sql': (
                "SELECT PLAN_TYPE, "
                "ROUND(AVG(RISK_SCORE), 3) as avg_risk_score, "
                "ROUND(MIN(RISK_SCORE), 2) as min_risk, "
                "ROUND(MAX(RISK_SCORE), 2) as max_risk, "
                "COUNT(*) as member_count "
                "FROM members GROUP BY PLAN_TYPE ORDER BY avg_risk_score DESC"
            ),
        },
        {
            'patterns': [r'risk\s*score.*region', r'risk.*by\s*region',
                         r'risk\s*score.*by\s*region', r'region.*risk\s*score'],
            'sql': (
                "SELECT KP_REGION, "
                "ROUND(AVG(RISK_SCORE), 3) as avg_risk_score, "
                "ROUND(MIN(RISK_SCORE), 2) as min_risk, "
                "ROUND(MAX(RISK_SCORE), 2) as max_risk, "
                "COUNT(*) as member_count "
                "FROM members GROUP BY KP_REGION ORDER BY avg_risk_score DESC"
            ),
        },
        {
            'patterns': [r'^(?!.*by\s).*(?:avg|average).*risk\s*score$',
                         r'^(?!.*by\s).*risk\s*score.*(?:avg|average)$'],
            'sql': (
                "SELECT ROUND(AVG(RISK_SCORE), 3) as avg_risk_score, "
                "ROUND(MIN(RISK_SCORE), 2) as min_risk, "
                "ROUND(MAX(RISK_SCORE), 2) as max_risk, "
                "COUNT(*) as member_count "
                "FROM members"
            ),
        },
        {
            'patterns': [r'member.*chronic\s*condition', r'chronic\s*condition.*count',
                         r'chronic\s*condition.*breakdown', r'chronic\s*condition.*distribution',
                         r'member.*by\s*chronic', r'patient.*chronic\s*condition'],
            'sql': (
                "SELECT CHRONIC_CONDITIONS, COUNT(*) as member_count, "
                "ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM members), 1) as pct, "
                "ROUND(AVG(RISK_SCORE), 3) as avg_risk_score "
                "FROM members GROUP BY CHRONIC_CONDITIONS ORDER BY CHRONIC_CONDITIONS"
            ),
        },

        {
            'patterns': [r'(?:average|avg|mean).*cost.*(?:diabet|hypertens|chronic|asthma)',
                         r'(?:diabet|hypertens|chronic|asthma).*(?:average|avg|mean).*cost',
                         r'(?:average|avg|mean).*(?:claim|paid|billed).*(?:diabet|hypertens)',
                         r'(?:diabet|hypertens).*patient.*(?:cost|spend|paid|claim)',
                         r'cost.*(?:diabet|hypertens).*patient'],
            'sql': (
                "SELECT ROUND(AVG(c.PAID_AMOUNT), 2) as avg_paid, "
                "ROUND(AVG(c.BILLED_AMOUNT), 2) as avg_billed, "
                "COUNT(*) as total_claims, "
                "COUNT(DISTINCT c.MEMBER_ID) as unique_members "
                "FROM claims c "
                "JOIN diagnoses d ON c.MEMBER_ID = d.MEMBER_ID AND c.ENCOUNTER_ID = d.ENCOUNTER_ID "
                "WHERE d.ICD10_DESCRIPTION LIKE '%diabet%' "
                "OR d.ICD10_CODE LIKE 'E10%' OR d.ICD10_CODE LIKE 'E11%' "
                "OR d.ICD10_CODE LIKE 'E13%'"
            ),
        },
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
        {
            'patterns': [r'^(?!.*(?:cost|paid|billed|average\s*cost|avg\s*cost)).*encounter.*by\s*visit\s*type',
                         r'^(?!.*(?:cost|paid|billed|average\s*cost|avg\s*cost)).*encounter.*visit\s*type',
                         r'visit\s*type.*breakdown', r'visit\s*type.*distribution',
                         r'visit\s*type.*count'],
            'sql': (
                "SELECT VISIT_TYPE, COUNT(*) as encounter_count, "
                "ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM encounters), 1) as pct "
                "FROM encounters GROUP BY VISIT_TYPE ORDER BY encounter_count DESC"
            ),
        },
        {
            'patterns': [r'average.*length.*stay.*inpatient', r'avg.*length.*stay.*inpatient',
                         r'inpatient.*average.*length.*stay', r'inpatient.*avg.*length.*stay',
                         r'average.*los\b.*inpatient', r'avg.*los\b.*inpatient',
                         r'inpatient.*length.*stay', r'average.*bed\s*days',
                         r'alos.*inpatient', r'inpatient.*alos'],
            'sql': (
                "SELECT ROUND(AVG(LENGTH_OF_STAY), 2) as avg_los_days, "
                "ROUND(MIN(LENGTH_OF_STAY), 0) as min_los, "
                "ROUND(MAX(LENGTH_OF_STAY), 0) as max_los, "
                "COUNT(*) as inpatient_count "
                "FROM encounters WHERE VISIT_TYPE = 'INPATIENT' AND LENGTH_OF_STAY > 0"
            ),
        },
        {
            'patterns': [r'chronic.*diagnos', r'number.*chronic', r'count.*chronic\s*diagnos',
                         r'how\s+many.*chronic'],
            'sql': (
                "SELECT IS_CHRONIC, COUNT(*) as diagnosis_count, "
                "ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM diagnoses), 1) as pct "
                "FROM diagnoses GROUP BY IS_CHRONIC ORDER BY diagnosis_count DESC"
            ),
        },

        {
            'patterns': [r'top.*diagnos(?!.*(?:by\s*(?:claim\s*)?cost|by\s*paid|by\s*spend))',
                         r'common.*diagnos', r'frequent.*diagnos',
                         r'most.*diagnos(?!.*(?:cost|expens|paid))',
                         r'diagnos.*frequent', r'diagnos.*common'],
            'sql': (
                "SELECT ICD10_CODE, ICD10_DESCRIPTION, "
                "COUNT(*) as diagnosis_count, "
                "COUNT(DISTINCT MEMBER_ID) as affected_members "
                "FROM diagnoses "
                "WHERE ICD10_CODE IS NOT NULL "
                "GROUP BY ICD10_CODE, ICD10_DESCRIPTION "
                "ORDER BY diagnosis_count DESC LIMIT 20"
            ),
        },
        {
            'patterns': [r'top.*procedure', r'common.*procedure', r'frequent.*procedure',
                         r'most.*procedure', r'procedure.*frequent', r'procedure.*common',
                         r'top.*cpt', r'common.*cpt', r'frequent.*cpt'],
            'sql': (
                "SELECT CPT_CODE, CPT_DESCRIPTION, "
                "COUNT(*) as claim_count, "
                "COUNT(DISTINCT MEMBER_ID) as affected_members, "
                "ROUND(SUM(BILLED_AMOUNT), 2) as total_billed "
                "FROM claims "
                "WHERE CPT_CODE IS NOT NULL "
                "GROUP BY CPT_CODE, CPT_DESCRIPTION "
                "ORDER BY claim_count DESC LIMIT 20"
            ),
        },

        {
            'patterns': [r'top.*medication', r'common.*medication', r'frequent.*medication',
                         r'most.*medication', r'medication.*frequent', r'medication.*common',
                         r'top.*prescription', r'most.*prescribed', r'top.*drug',
                         r'common.*drug', r'prescription.*count', r'medication.*count'],
            'sql': (
                "SELECT MEDICATION_NAME, MEDICATION_CLASS, "
                "COUNT(*) as rx_count, "
                "COUNT(DISTINCT MEMBER_ID) as unique_members, "
                "ROUND(SUM(COST), 2) as total_cost, "
                "ROUND(AVG(COST), 2) as avg_cost "
                "FROM prescriptions "
                "GROUP BY MEDICATION_NAME, MEDICATION_CLASS "
                "ORDER BY rx_count DESC LIMIT 20"
            ),
        },

        {
            'patterns': [r'most\s+expensive\s+(?:medication|drug|prescription|rx)',
                         r'costliest\s+(?:medication|drug|prescription|rx)',
                         r'priciest\s+(?:medication|drug)',
                         r'highest\s+cost\s+(?:medication|drug)'],
            'sql': (
                "SELECT MEDICATION_NAME, MEDICATION_CLASS, "
                "ROUND(MAX(CAST(COST AS REAL)), 2) as max_cost, "
                "ROUND(AVG(CAST(COST AS REAL)), 2) as avg_cost, "
                "COUNT(*) as rx_count "
                "FROM prescriptions "
                "GROUP BY MEDICATION_NAME, MEDICATION_CLASS "
                "ORDER BY max_cost DESC LIMIT 20"
            ),
        },
        {
            'patterns': [r'total\s+cost.*prescription', r'total.*prescription.*cost',
                         r'total\s+cost.*(?:rx|medication|drug)',
                         r'(?:rx|medication|drug).*total\s+cost',
                         r'total.*(?:pharmacy|drug)\s+spend',
                         r'how\s+much.*(?:spend|spent).*prescription'],
            'sql': (
                "SELECT ROUND(SUM(CAST(COST AS REAL)), 2) as total_cost, "
                "COUNT(*) as total_prescriptions, "
                "ROUND(AVG(CAST(COST AS REAL)), 2) as avg_cost_per_rx "
                "FROM prescriptions"
            ),
        },
        {
            'patterns': [r'average\s+cost.*prescription', r'avg\s+cost.*prescription',
                         r'average.*(?:rx|prescription)\s+cost',
                         r'mean\s+cost.*prescription',
                         r'cost\s+per\s+prescription',
                         r'average.*(?:drug|medication)\s+cost'],
            'sql': (
                "SELECT ROUND(AVG(CAST(COST AS REAL)), 2) as avg_cost, "
                "ROUND(MIN(CAST(COST AS REAL)), 2) as min_cost, "
                "ROUND(MAX(CAST(COST AS REAL)), 2) as max_cost, "
                "COUNT(*) as total_prescriptions "
                "FROM prescriptions"
            ),
        },
        {
            'patterns': [r'total\s+prescriptions?\s+(?:written|filled|dispensed|issued)',
                         r'how\s+many\s+prescriptions?\s+(?:written|filled|dispensed|were|have)',
                         r'(?:total|number\s+of)\s+prescriptions?\s+(?:in|by|across|written|filled)',
                         r'prescriptions?\s+(?:written|filled).*(?:network|system|total)',
                         r'total\s+(?:number\s+of\s+)?prescriptions?\b(?!.*cost)(?!.*spend)(?!.*price)'],
            'sql': (
                "SELECT COUNT(*) as total_prescriptions, "
                "COUNT(DISTINCT MEDICATION_NAME) as unique_medications, "
                "COUNT(DISTINCT MEMBER_ID) as unique_patients "
                "FROM prescriptions"
            ),
        },
        {
            'patterns': [r'(?:average|avg|mean)\s+days?\s+supply',
                         r'days?\s+supply.*(?:average|avg)',
                         r'(?:average|avg).*supply\s+(?:days|duration)',
                         r'prescription.*(?:duration|supply)',
                         r'how\s+(?:many|long).*days?\s+supply'],
            'sql': (
                "SELECT ROUND(AVG(CAST(DAYS_SUPPLY AS REAL)), 2) as avg_days_supply, "
                "ROUND(MIN(CAST(DAYS_SUPPLY AS REAL)), 0) as min_days_supply, "
                "ROUND(MAX(CAST(DAYS_SUPPLY AS REAL)), 0) as max_days_supply, "
                "COUNT(*) as total_prescriptions "
                "FROM prescriptions"
            ),
        },

        {
            'patterns': [r'no.?show\s*rate', r'appointment.*no.?show',
                         r'no.?show.*appointment', r'missed.*appointment',
                         r'appointment.*miss', r'no.?show'],
            'sql': (
                "SELECT STATUS, COUNT(*) as appt_count, "
                "ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM appointments), 1) as pct "
                "FROM appointments GROUP BY STATUS ORDER BY appt_count DESC"
            ),
        },
        {
            'patterns': [r'referral.*completion', r'referral.*complete.*rate',
                         r'referral.*status', r'referral.*breakdown'],
            'sql': (
                "SELECT STATUS, COUNT(*) as referral_count, "
                "ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM referrals), 1) as pct "
                "FROM referrals GROUP BY STATUS ORDER BY referral_count DESC"
            ),
        },

        {
            'patterns': [r'(?:total|how\s+many)\s*(?:number\s*of\s*)?unique\s*provider',
                         r'(?:total|how\s+many)\s*(?:number\s*of\s*)?distinct\s*(?:provider|npi)',
                         r'count.*unique.*provider', r'unique.*provider.*count',
                         r'number.*unique.*provider',
                         r'distinct\s+npi', r'unique\s+npi',
                         r'how\s+many.*npi.*provider'],
            'sql': (
                "SELECT COUNT(DISTINCT NPI) as unique_providers FROM providers"
            ),
        },
        {
            'patterns': [r'provider.*by\s*specialty', r'provider.*specialty.*breakdown',
                         r'specialty.*count', r'specialty.*distribution',
                         r'how\s+many.*provider.*specialty'],
            'sql': (
                "SELECT SPECIALTY, COUNT(*) as provider_count, "
                "ROUND(AVG(PANEL_SIZE), 0) as avg_panel_size, "
                "SUM(CASE WHEN STATUS = 'Active' THEN 1 ELSE 0 END) as active_providers "
                "FROM providers GROUP BY SPECIALTY ORDER BY provider_count DESC"
            ),
        },

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
        {
            'patterns': [r'claim.*by\s*claim\s*type', r'claim\s*type.*breakdown',
                         r'claim\s*type.*distribution', r'professional.*institutional'],
            'sql': (
                "SELECT CLAIM_TYPE, COUNT(*) as claim_count, "
                "ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM claims), 1) as pct, "
                "ROUND(SUM(BILLED_AMOUNT), 2) as total_billed, "
                "ROUND(SUM(PAID_AMOUNT), 2) as total_paid "
                "FROM claims GROUP BY CLAIM_TYPE ORDER BY claim_count DESC"
            ),
        },
        {
            'patterns': [r'claim.*by\s*facility', r'facility.*claim.*count',
                         r'facility.*breakdown', r'facility.*distribution'],
            'sql': (
                "SELECT FACILITY, COUNT(*) as claim_count, "
                "ROUND(SUM(BILLED_AMOUNT), 2) as total_billed, "
                "ROUND(SUM(PAID_AMOUNT), 2) as total_paid "
                "FROM claims GROUP BY FACILITY ORDER BY claim_count DESC"
            ),
        },
        {
            'patterns': [r'claim.*by\s*region', r'region.*claim.*count',
                         r'region.*breakdown', r'claim.*region.*distribution'],
            'sql': (
                "SELECT KP_REGION as region, COUNT(*) as claim_count, "
                "ROUND(SUM(BILLED_AMOUNT), 2) as total_billed, "
                "ROUND(SUM(PAID_AMOUNT), 2) as total_paid, "
                "COUNT(DISTINCT MEMBER_ID) as unique_members "
                "FROM claims GROUP BY KP_REGION ORDER BY claim_count DESC"
            ),
        },
        {
            'patterns': [r'total.*paid.*by\s*plan', r'paid.*amount.*by\s*plan',
                         r'paid.*per\s*plan', r'payment.*by\s*plan'],
            'sql': (
                "SELECT PLAN_TYPE, ROUND(SUM(PAID_AMOUNT), 2) as total_paid, "
                "ROUND(SUM(BILLED_AMOUNT), 2) as total_billed, "
                "COUNT(*) as claim_count, "
                "COUNT(DISTINCT MEMBER_ID) as unique_members "
                "FROM claims GROUP BY PLAN_TYPE ORDER BY total_paid DESC"
            ),
        },
        {
            'patterns': [r'claim.*by\s*severity', r'claim.*severity.*count',
                         r'claim.*count.*severity', r'severity.*claim',
                         r'severity.*breakdown', r'severity.*distribution',
                         r'total.*claim.*severity', r'count.*claim.*severity'],
            'sql': (
                "SELECT d.SEVERITY, "
                "COUNT(DISTINCT c.CLAIM_ID) as claim_count, "
                "ROUND(100.0 * COUNT(DISTINCT c.CLAIM_ID) / (SELECT COUNT(*) FROM claims), 1) as pct, "
                "ROUND(AVG(c.PAID_AMOUNT), 2) as avg_paid_amount, "
                "COUNT(DISTINCT c.MEMBER_ID) as unique_members "
                "FROM claims c "
                "JOIN diagnoses d ON c.ENCOUNTER_ID = d.ENCOUNTER_ID "
                "WHERE d.SEVERITY IS NOT NULL "
                "GROUP BY d.SEVERITY "
                "ORDER BY claim_count DESC"
            ),
        },
        {
            'patterns': [r'cost.*by\s*severity', r'paid.*by\s*severity',
                         r'severity.*cost', r'severity.*paid',
                         r'spend.*by\s*severity', r'amount.*severity'],
            'sql': (
                "SELECT d.SEVERITY, "
                "COUNT(DISTINCT c.CLAIM_ID) as claim_count, "
                "ROUND(SUM(c.PAID_AMOUNT) / NULLIF(COUNT(DISTINCT c.CLAIM_ID), 0), 2) as avg_paid_per_claim, "
                "ROUND(SUM(c.BILLED_AMOUNT) / NULLIF(COUNT(DISTINCT c.CLAIM_ID), 0), 2) as avg_billed_per_claim, "
                "COUNT(DISTINCT c.MEMBER_ID) as unique_members "
                "FROM claims c "
                "JOIN diagnoses d ON c.ENCOUNTER_ID = d.ENCOUNTER_ID "
                "WHERE d.SEVERITY IS NOT NULL "
                "GROUP BY d.SEVERITY "
                "ORDER BY avg_paid_per_claim DESC"
            ),
        },
        {
            'patterns': [r'diagnos.*by\s*severity', r'diagnos.*severity.*count',
                         r'diagnos.*count.*severity', r'severity.*diagnos'],
            'sql': (
                "SELECT SEVERITY, COUNT(*) as diagnosis_count, "
                "ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM diagnoses), 1) as pct, "
                "SUM(CASE WHEN IS_CHRONIC = 1 THEN 1 ELSE 0 END) as chronic_count, "
                "COUNT(DISTINCT MEMBER_ID) as unique_members "
                "FROM diagnoses "
                "WHERE SEVERITY IS NOT NULL "
                "GROUP BY SEVERITY "
                "ORDER BY diagnosis_count DESC"
            ),
        },

        {
            'patterns': [r'(?:compare|comparison).*inpatient.*outpatient',
                         r'inpatient.*vs.*outpatient',
                         r'inpatient.*versus.*outpatient',
                         r'outpatient.*vs.*inpatient',
                         r'inpatient.*outpatient.*cost'],
            'sql': (
                "SELECT e.VISIT_TYPE, COUNT(*) as encounter_count, "
                "ROUND(AVG(c.PAID_AMOUNT), 2) as avg_cost, "
                "ROUND(SUM(c.PAID_AMOUNT), 2) as total_cost, "
                "ROUND(AVG(c.BILLED_AMOUNT), 2) as avg_billed, "
                "COUNT(DISTINCT e.MEMBER_ID) as unique_members "
                "FROM encounters e "
                "JOIN claims c ON e.ENCOUNTER_ID = c.ENCOUNTER_ID "
                "WHERE e.VISIT_TYPE IN ('INPATIENT', 'OUTPATIENT') "
                "GROUP BY e.VISIT_TYPE "
                "ORDER BY avg_cost DESC"
            ),
        },
        {
            'patterns': [r'billed.*vs.*paid', r'billed.*versus.*paid',
                         r'compare.*billed.*paid', r'paid.*vs.*billed'],
            'sql': (
                "SELECT PLAN_TYPE, "
                "ROUND(AVG(BILLED_AMOUNT), 2) as avg_billed, "
                "ROUND(AVG(PAID_AMOUNT), 2) as avg_paid, "
                "ROUND(AVG(ALLOWED_AMOUNT), 2) as avg_allowed, "
                "ROUND(AVG(BILLED_AMOUNT) - AVG(PAID_AMOUNT), 2) as avg_gap, "
                "COUNT(*) as claims "
                "FROM claims GROUP BY PLAN_TYPE ORDER BY avg_gap DESC"
            ),
        },


        {
            'patterns': [r'compare.*(?:ncal|scal|region).*(?:kpi|all|across|metric)',
                         r'(?:ncal|scal).*vs.*(?:ncal|scal)',
                         r'compare.*region.*(?:across|all)', r'region.*comparison.*kpi',
                         r'rank.*region.*(?:performance|overall)', r'region.*(?:rank|ranking)'],
            'sql': (
                "SELECT c.KP_REGION AS region, "
                "COUNT(DISTINCT m.MEMBER_ID) AS members, "
                "ROUND(SUM(c.PAID_AMOUNT) * 1.0 / NULLIF(COUNT(DISTINCT m.MEMBER_ID || strftime('%Y-%m', c.SERVICE_DATE)), 0), 2) AS pmpm, "
                "ROUND(100.0 * SUM(c.PAID_AMOUNT) / NULLIF(SUM(c.BILLED_AMOUNT), 0), 2) AS mlr_pct, "
                "ROUND(100.0 * SUM(CASE WHEN c.CLAIM_STATUS='DENIED' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) AS denial_rate, "
                "ROUND(100.0 * SUM(CASE WHEN c.CLAIM_STATUS='PAID' THEN c.PAID_AMOUNT ELSE 0 END) / NULLIF(SUM(c.BILLED_AMOUNT), 0), 2) AS collection_rate, "
                "ROUND(AVG(m.RISK_SCORE), 3) AS avg_risk, "
                "COUNT(*) AS total_claims "
                "FROM claims c JOIN members m ON c.MEMBER_ID = m.MEMBER_ID "
                "GROUP BY c.KP_REGION ORDER BY pmpm DESC"
            ),
        },

        {
            'patterns': [r'profit\s*margin', r'net\s*(?:margin|income|revenue)',
                         r'revenue.*vs.*cost', r'billed.*minus.*paid',
                         r'how.*much.*(?:profit|earn|net)', r'total.*revenue.*total.*cost'],
            'sql': (
                "SELECT ROUND(SUM(BILLED_AMOUNT), 2) AS total_billed, "
                "ROUND(SUM(PAID_AMOUNT), 2) AS total_paid, "
                "ROUND(SUM(BILLED_AMOUNT) - SUM(PAID_AMOUNT), 2) AS margin, "
                "ROUND(100.0 * (SUM(BILLED_AMOUNT) - SUM(PAID_AMOUNT)) / NULLIF(SUM(BILLED_AMOUNT), 0), 2) AS margin_pct, "
                "ROUND(SUM(ALLOWED_AMOUNT), 2) AS total_allowed, "
                "COUNT(*) AS total_claims "
                "FROM claims WHERE CLAIM_STATUS = 'PAID'"
            ),
        },

        {
            'patterns': [r'money.*(?:on|leave|left).*table', r'(?:uncollect|unrecover|lost).*revenue',
                         r'denial.*(?:revenue|cost|impact|recovery)', r'how.*much.*(?:losing|lost).*denial',
                         r'potential.*recovery', r'denial.*savings',
                         r'revenue.*(?:recover|recoup).*denied', r'recover.*denied.*claim'],
            'sql': (
                "SELECT "
                "SUM(CASE WHEN CLAIM_STATUS='DENIED' THEN BILLED_AMOUNT ELSE 0 END) AS denied_revenue, "
                "SUM(CASE WHEN CLAIM_STATUS='PAID' THEN BILLED_AMOUNT - PAID_AMOUNT ELSE 0 END) AS underpayment_gap, "
                "SUM(CASE WHEN CLAIM_STATUS='DENIED' THEN BILLED_AMOUNT ELSE 0 END) + "
                "SUM(CASE WHEN CLAIM_STATUS='PAID' THEN BILLED_AMOUNT - PAID_AMOUNT ELSE 0 END) AS total_opportunity, "
                "COUNT(CASE WHEN CLAIM_STATUS='DENIED' THEN 1 END) AS denied_claims, "
                "ROUND(100.0 * SUM(CASE WHEN CLAIM_STATUS='DENIED' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) AS denial_pct, "
                "ROUND(AVG(CASE WHEN CLAIM_STATUS='DENIED' THEN BILLED_AMOUNT END), 2) AS avg_denied_amount "
                "FROM claims"
            ),
        },

        {
            'patterns': [r'(?:provider|specialty).*cost.*(?:per|each).*member',
                         r'(?:most|highest).*(?:expensive|costly).*(?:specialty|provider)',
                         r'specialty.*(?:rank|most|highest).*cost',
                         r'cost.*by.*specialty', r'spending.*by.*specialty'],
            'sql': (
                "SELECT p.SPECIALTY, "
                "COUNT(DISTINCT c.MEMBER_ID) AS unique_members, "
                "ROUND(SUM(c.PAID_AMOUNT), 2) AS total_paid, "
                "ROUND(SUM(c.PAID_AMOUNT) / NULLIF(COUNT(DISTINCT c.MEMBER_ID), 0), 2) AS cost_per_member, "
                "COUNT(*) AS claims, "
                "ROUND(AVG(c.PAID_AMOUNT), 2) AS avg_claim "
                "FROM claims c JOIN providers p ON c.RENDERING_NPI = p.NPI "
                "GROUP BY p.SPECIALTY ORDER BY cost_per_member DESC LIMIT 20"
            ),
        },

        {
            'patterns': [r'month.*(?:lost|loss|worst|most.*money|biggest.*loss)',
                         r'worst.*(?:month|period).*(?:financial|revenue)',
                         r'when.*(?:lost|lose).*(?:most|biggest)',
                         r'(?:highest|biggest).*(?:loss|deficit).*month',
                         r'biggest.*financial.*loss', r'when.*biggest.*(?:financial|loss)'],
            'sql': (
                "SELECT strftime('%Y-%m', SERVICE_DATE) AS month, "
                "ROUND(SUM(BILLED_AMOUNT), 2) AS billed, "
                "ROUND(SUM(PAID_AMOUNT), 2) AS paid, "
                "ROUND(SUM(BILLED_AMOUNT) - SUM(PAID_AMOUNT), 2) AS gap, "
                "COUNT(*) AS claims, "
                "SUM(CASE WHEN CLAIM_STATUS='DENIED' THEN 1 ELSE 0 END) AS denials "
                "FROM claims GROUP BY month ORDER BY gap DESC LIMIT 12"
            ),
        },

        {
            'patterns': [r'high.*cost.*(?:disenroll|leaving|churn)',
                         r'(?:disenroll|leaving).*high.*cost',
                         r'(?:expensive|costly).*member.*(?:leave|disenroll|churn)',
                         r'cost.*correlat.*(?:disenroll|retention|churn)'],
            'sql': (
                "SELECT "
                "CASE WHEN m.DISENROLLMENT_DATE != '' THEN 'Disenrolled' ELSE 'Active' END AS status, "
                "COUNT(*) AS member_count, "
                "ROUND(AVG(CAST(m.RISK_SCORE AS REAL)), 3) AS avg_risk_score, "
                "ROUND(AVG(total_cost), 2) AS avg_total_cost, "
                "ROUND(AVG(claim_count), 1) AS avg_claims "
                "FROM members m LEFT JOIN ("
                "  SELECT MEMBER_ID, SUM(PAID_AMOUNT) AS total_cost, COUNT(*) AS claim_count "
                "  FROM claims GROUP BY MEMBER_ID"
                ") c ON m.MEMBER_ID = c.MEMBER_ID "
                "GROUP BY status"
            ),
        },

        {
            'patterns': [r'member.*never.*(?:visit|doctor|encounter|seen)',
                         r'zero.*(?:utilization|visit|encounter)',
                         r'no.*(?:visit|encounter|appointment).*member',
                         r'member.*(?:0|zero).*(?:claim|encounter|visit)'],
            'sql': (
                "SELECT m.KP_REGION AS region, COUNT(*) AS members_never_visited, "
                "ROUND(AVG(CAST(m.RISK_SCORE AS REAL)), 3) AS avg_risk_score "
                "FROM members m LEFT JOIN encounters e ON m.MEMBER_ID = e.MEMBER_ID "
                "WHERE e.MEMBER_ID IS NULL "
                "GROUP BY m.KP_REGION ORDER BY members_never_visited DESC"
            ),
        },

        {
            'patterns': [r'(?:how\s+many|number\s+of).*(?:die|died|death|deceased|mortality)',
                         r'death.*(?:rate|count|number)', r'mortality.*rate'],
            'sql': None,
            '_narrative': (
                "Our database does not contain mortality/death data. The DISENROLLMENT_DATE field tracks "
                "when members leave the plan, but disenrollment reasons (including death) are not recorded. "
                "To answer mortality questions, you would need CMS death file linkage or claims-based mortality "
                "indicators (e.g., discharge disposition codes). What we CAN tell you: we have {active} active "
                "members and {disenrolled} disenrolled members."
            ),
        },

        {
            'patterns': [r'(?:which|top|first).*member.*(?:call|outreach|contact|retention)',
                         r'retention.*(?:outreach|target|priority|list)',
                         r'(?:at.risk|high.risk).*(?:disenroll|churn).*member',
                         r'member.*(?:likely|risk).*(?:leave|disenroll|churn)'],
            'sql': (
                "SELECT m.MEMBER_ID, m.KP_REGION AS region, m.RISK_SCORE, "
                "ROUND(JULIANDAY('now') - JULIANDAY(m.ENROLLMENT_DATE)) AS days_enrolled, "
                "COALESCE(c.total_claims, 0) AS total_claims, "
                "COALESCE(ROUND(c.total_paid, 2), 0) AS total_cost, "
                "COALESCE(c.denied_claims, 0) AS denied_claims, "
                "CASE WHEN m.RISK_SCORE > 2.0 THEN 'Critical' "
                "     WHEN m.RISK_SCORE > 1.5 THEN 'High' "
                "     WHEN m.RISK_SCORE > 1.0 THEN 'Moderate' ELSE 'Low' END AS risk_tier "
                "FROM members m LEFT JOIN ("
                "  SELECT MEMBER_ID, COUNT(*) AS total_claims, SUM(PAID_AMOUNT) AS total_paid, "
                "  SUM(CASE WHEN CLAIM_STATUS='DENIED' THEN 1 ELSE 0 END) AS denied_claims "
                "  FROM claims GROUP BY MEMBER_ID"
                ") c ON m.MEMBER_ID = c.MEMBER_ID "
                "WHERE m.DISENROLLMENT_DATE = '' "
                "ORDER BY m.RISK_SCORE DESC, c.total_paid DESC LIMIT 100"
            ),
        },

        {
            'patterns': [r'(?:average|avg).*cost.*(?:by\s*)?department',
                         r'cost.*per\s*department', r'department.*cost',
                         r'spend.*by\s*department'],
            'sql': (
                "SELECT e.DEPARTMENT, "
                "COUNT(*) as encounter_count, "
                "ROUND(AVG(CAST(c.PAID_AMOUNT AS REAL)), 2) as avg_cost, "
                "ROUND(SUM(CAST(c.PAID_AMOUNT AS REAL)), 2) as total_cost "
                "FROM encounters e JOIN claims c ON e.ENCOUNTER_ID = c.ENCOUNTER_ID "
                "WHERE e.DEPARTMENT IS NOT NULL "
                "GROUP BY e.DEPARTMENT ORDER BY avg_cost DESC"
            ),
        },
        {
            'patterns': [r'cost.*per\s*member.*age\s*group', r'cost.*member.*by\s*age',
                         r'age\s*group.*cost.*member', r'spend.*per\s*member.*age'],
            'sql': (
                "SELECT CASE "
                "WHEN CAST((julianday('now') - julianday(m.DATE_OF_BIRTH)) / 365.25 AS INT) < 18 THEN '0-17 Pediatric' "
                "WHEN CAST((julianday('now') - julianday(m.DATE_OF_BIRTH)) / 365.25 AS INT) BETWEEN 18 AND 25 THEN '18-25 Young Adult' "
                "WHEN CAST((julianday('now') - julianday(m.DATE_OF_BIRTH)) / 365.25 AS INT) BETWEEN 26 AND 40 THEN '26-40 Adult' "
                "WHEN CAST((julianday('now') - julianday(m.DATE_OF_BIRTH)) / 365.25 AS INT) BETWEEN 41 AND 55 THEN '41-55 Middle Age' "
                "WHEN CAST((julianday('now') - julianday(m.DATE_OF_BIRTH)) / 365.25 AS INT) BETWEEN 56 AND 64 THEN '56-64 Pre-Medicare' "
                "ELSE '65+ Medicare' END as age_group, "
                "COUNT(DISTINCT m.MEMBER_ID) as members, "
                "ROUND(SUM(CAST(c.PAID_AMOUNT AS REAL)) / NULLIF(COUNT(DISTINCT m.MEMBER_ID), 0), 2) as cost_per_member, "
                "COUNT(*) as total_claims "
                "FROM claims c JOIN members m ON c.MEMBER_ID = m.MEMBER_ID "
                "GROUP BY age_group "
                "ORDER BY MIN(CAST((julianday('now') - julianday(m.DATE_OF_BIRTH)) / 365.25 AS INT))"
            ),
        },
        {
            'patterns': [r'(?:highest|most).*cost.*per\s*member.*region',
                         r'region.*highest.*cost.*member',
                         r'which\s*region.*(?:highest|most).*cost.*member',
                         r'cost.*per\s*member.*by\s*region'],
            'sql': (
                "SELECT m.KP_REGION, "
                "COUNT(DISTINCT m.MEMBER_ID) as members, "
                "ROUND(SUM(CAST(c.PAID_AMOUNT AS REAL)) / NULLIF(COUNT(DISTINCT m.MEMBER_ID), 0), 2) as cost_per_member, "
                "ROUND(SUM(CAST(c.PAID_AMOUNT AS REAL)), 2) as total_paid, "
                "COUNT(*) as total_claims "
                "FROM claims c JOIN members m ON c.MEMBER_ID = m.MEMBER_ID "
                "GROUP BY m.KP_REGION ORDER BY cost_per_member DESC"
            ),
        },
        {
            'patterns': [r'top.*diagnos.*(?:by\s*(?:claim\s*)?cost|by\s*paid|by\s*spend)',
                         r'(?:costliest|expensive).*diagnos',
                         r'diagnos.*highest.*cost', r'diagnos.*most.*cost'],
            'sql': (
                "SELECT d.ICD10_DESCRIPTION, "
                "COUNT(DISTINCT c.CLAIM_ID) as claim_count, "
                "ROUND(SUM(CAST(c.PAID_AMOUNT AS REAL)), 2) as total_cost, "
                "ROUND(AVG(CAST(c.PAID_AMOUNT AS REAL)), 2) as avg_cost_per_claim, "
                "COUNT(DISTINCT d.MEMBER_ID) as affected_members "
                "FROM diagnoses d JOIN claims c ON d.MEMBER_ID = c.MEMBER_ID AND d.ENCOUNTER_ID = c.ENCOUNTER_ID "
                "WHERE d.ICD10_DESCRIPTION IS NOT NULL "
                "GROUP BY d.ICD10_DESCRIPTION ORDER BY total_cost DESC LIMIT 20"
            ),
        },
        {
            'patterns': [r'(?:average|avg).*cost.*(?:per\s*)?encounter.*(?:by\s*)?visit\s*type',
                         r'cost.*per\s*encounter.*visit\s*type',
                         r'visit\s*type.*cost.*per\s*encounter',
                         r'encounter.*cost.*by\s*visit\s*type'],
            'sql': (
                "SELECT e.VISIT_TYPE, "
                "COUNT(*) as encounter_count, "
                "ROUND(AVG(CAST(c.PAID_AMOUNT AS REAL)), 2) as avg_cost_per_encounter, "
                "ROUND(SUM(CAST(c.PAID_AMOUNT AS REAL)), 2) as total_cost, "
                "COUNT(DISTINCT e.MEMBER_ID) as unique_members "
                "FROM encounters e JOIN claims c ON e.ENCOUNTER_ID = c.ENCOUNTER_ID "
                "GROUP BY e.VISIT_TYPE ORDER BY avg_cost_per_encounter DESC"
            ),
        },

        {
            'patterns': [r'(?:improve|increase|boost|raise).*star.*rating',
                         r'biggest.*(?:impact|thing).*star', r'star.*(?:improve|increase)',
                         r'single.*biggest.*(?:improve|impact).*(?:star|quality)',
                         r'how.*(?:do|can).*(?:boost|improve|increase|raise).*star',
                         r'what.*(?:can|do).*(?:improve|boost|increase).*star',
                         r'(?:fastest|quickest|best).*(?:way|thing).*(?:improve|boost|raise).*star'],
            'sql': (
                "SELECT 'HEDIS Gap Closure' AS opportunity, "
                "'Target Breast Cancer Screening (BCS) and Colorectal Screening (COL) — largest gaps below benchmark' AS action, "
                "'These 2 measures carry ~20%% of Stars weight and have the widest gap from 4-star benchmark' AS why, "
                "'Focus on 21,549 members missing screenings — each 1%% improvement adds ~0.02 stars' AS impact "
                "UNION ALL "
                "SELECT 'CAHPS Experience' AS opportunity, "
                "'Improve Getting Needed Care and Customer Service scores through care coordination' AS action, "
                "'Patient experience measures account for ~30%% of overall rating' AS why, "
                "'Current 73.2%% vs 85%% benchmark — closing this gap could add 0.3-0.5 stars' AS impact "
                "UNION ALL "
                "SELECT 'Denial Rate Reduction' AS opportunity, "
                "'Reduce 9.5%% coding denials through provider education and pre-authorization automation' AS action, "
                "'Denied claims hurt D-SNP measures and Part D adherence scores' AS why, "
                "'Recovery of $2M+ in denied revenue while improving quality metrics' AS impact"
            ),
        },
        {
            'patterns': [r'claim.*processing\s*time.*trend', r'processing\s*time.*trend',
                         r'claim.*turnaround.*trend', r'adjudication.*time.*trend',
                         r'claim.*processing.*over\s*time', r'how\s*long.*process.*claim.*trend',
                         r'claim.*cycle\s*time.*trend'],
            'sql': (
                "SELECT SUBSTR(SUBMITTED_DATE, 1, 7) AS month, "
                "COUNT(*) AS total_claims, "
                "ROUND(AVG(JULIANDAY(ADJUDICATED_DATE) - JULIANDAY(SUBMITTED_DATE)), 1) AS avg_processing_days, "
                "ROUND(MIN(JULIANDAY(ADJUDICATED_DATE) - JULIANDAY(SUBMITTED_DATE)), 1) AS min_processing_days, "
                "ROUND(MAX(JULIANDAY(ADJUDICATED_DATE) - JULIANDAY(SUBMITTED_DATE)), 1) AS max_processing_days "
                "FROM CLAIMS "
                "WHERE SUBMITTED_DATE != '' AND ADJUDICATED_DATE != '' "
                "AND SUBMITTED_DATE IS NOT NULL AND ADJUDICATED_DATE IS NOT NULL "
                "GROUP BY SUBSTR(SUBMITTED_DATE, 1, 7) "
                "ORDER BY month"
            ),
        },
        {
            'patterns': [r'member.*claim.*(?:greater\s*than|more\s*than|over|above|exceed|\>)\s*\$?(\d[\d,]*)',
                         r'claim.*(?:greater\s*than|more\s*than|over|above|exceed|\>)\s*\$?(\d[\d,]*).*member',
                         r'(?:which|show|list|find).*member.*claim.*(?:greater|more|over|above|exceed|\>)\s*\$?(\d[\d,]*)',
                         r'member.*(?:high|expensive).*claim.*(?:greater|more|over|above|exceed|\>)\s*\$?(\d[\d,]*)'],
            'sql': (
                "SELECT m.MEMBER_ID, m.FIRST_NAME, m.LAST_NAME, m.KP_REGION, "
                "COUNT(c.CLAIM_ID) AS claim_count, "
                "ROUND(SUM(CAST(c.PAID_AMOUNT AS REAL)), 2) AS total_paid, "
                "ROUND(MAX(CAST(c.PAID_AMOUNT AS REAL)), 2) AS max_single_claim "
                "FROM MEMBERS m JOIN CLAIMS c ON m.MEMBER_ID = c.MEMBER_ID "
                "WHERE CAST(c.PAID_AMOUNT AS REAL) > {threshold} "
                "GROUP BY m.MEMBER_ID, m.FIRST_NAME, m.LAST_NAME, m.KP_REGION "
                "ORDER BY total_paid DESC LIMIT 50"
            ),
            '_dynamic_threshold': True,
        },
    ]

    _TYPO_CORRECTIONS = {
        'clams': 'claims', 'clam': 'claim', 'cliam': 'claim', 'clims': 'claims',
        'calims': 'claims', 'claism': 'claims', 'caims': 'claims', 'cliams': 'claims',
        'memebers': 'members', 'memebrs': 'members', 'membrs': 'members',
        'memers': 'members', 'mebmers': 'members', 'mmbers': 'members', 'menmbers': 'members',
        'membes': 'members', 'membres': 'members', 'membesr': 'members',
        'payd': 'paid', 'payed': 'paid', 'piad': 'paid', 'padi': 'paid',
        'amout': 'amount', 'ammount': 'amount', 'amoutn': 'amount', 'amonut': 'amount',
        'amnount': 'amount', 'amoint': 'amount', 'amunt': 'amount',
        'averge': 'average', 'avrage': 'average', 'averag': 'average', 'avergae': 'average',
        'avrge': 'average', 'averege': 'average', 'avreage': 'average', 'averagge': 'average',
        'deneid': 'denied', 'denyed': 'denied', 'deniied': 'denied', 'denide': 'denied',
        'deined': 'denied', 'deneied': 'denied', 'deniald': 'denial', 'denail': 'denial',
        'higest': 'highest', 'highets': 'highest', 'hihgest': 'highest', 'highst': 'highest',
        'hgihest': 'highest', 'higheest': 'highest',
        'lowets': 'lowest', 'lowset': 'lowest', 'lowst': 'lowest', 'loweest': 'lowest',
        'rist': 'risk', 'riks': 'risk', 'risc': 'risk', 'rsik': 'risk', 'rikss': 'risk',
        'lenght': 'length', 'leghth': 'length', 'lenghth': 'length', 'lentgh': 'length',
        'legnth': 'length', 'legth': 'length',
        'stey': 'stay', 'stya': 'stay', 'satay': 'stay', 'saty': 'stay',
        'prescriptons': 'prescriptions', 'presciptions': 'prescriptions', 'perscriptions': 'prescriptions',
        'prescritpions': 'prescriptions', 'prsecriptions': 'prescriptions', 'prescrptions': 'prescriptions',
        'prescirptions': 'prescriptions', 'presciption': 'prescription', 'perscription': 'prescription',
        'medicaton': 'medication', 'medciation': 'medication', 'medicaiton': 'medication',
        'mediation': 'medication', 'medicaitons': 'medications', 'mdication': 'medication',
        'hwomany': 'how many', 'howmany': 'how many', 'howany': 'how many', 'hwmany': 'how many',
        'toal': 'total', 'totla': 'total', 'ttoal': 'total', 'totall': 'total', 'ttal': 'total',
        'numbr': 'number', 'nubmer': 'number', 'numebr': 'number', 'numer': 'number', 'numbeer': 'number',
        'wats': "what's", 'whats': "what's", 'waht': 'what', 'wht': 'what', 'hwat': 'what',
        'provdier': 'provider', 'providr': 'provider', 'proivder': 'provider',
        'prvider': 'provider', 'porider': 'provider', 'provder': 'provider', 'prvoider': 'provider',
        'encoutner': 'encounter', 'enconter': 'encounter', 'ecnounter': 'encounter',
        'encountr': 'encounter', 'encunter': 'encounter', 'encoounter': 'encounter',
        'diagnsis': 'diagnosis', 'diagnsois': 'diagnosis', 'diagnois': 'diagnosis',
        'diagosis': 'diagnosis', 'diagnosos': 'diagnosis', 'dianosis': 'diagnosis',
        'digagnosis': 'diagnosis', 'diagnoisis': 'diagnosis', 'diagnossi': 'diagnosis',
        'refarral': 'referral', 'referal': 'referral', 'refferal': 'referral',
        'refferral': 'referral', 'reerral': 'referral', 'referall': 'referral',
        'regon': 'region', 'reigon': 'region', 'regin': 'region', 'rgion': 'region', 'reagion': 'region',
        'speciailty': 'specialty', 'speciality': 'specialty', 'specailty': 'specialty',
        'speicalty': 'specialty', 'specality': 'specialty', 'speclaty': 'specialty',
        'copya': 'copay', 'copa': 'copay', 'coopay': 'copay',
        'deductble': 'deductible', 'deductibl': 'deductible', 'deductibel': 'deductible',
        'dedcutible': 'deductible', 'deducitble': 'deductible',
        'coinsurnace': 'coinsurance', 'coinsurnce': 'coinsurance', 'coinsruance': 'coinsurance',
        'enrollmnet': 'enrollment', 'enrolment': 'enrollment', 'enrllment': 'enrollment', 'enrollemnt': 'enrollment',
        'utilizaton': 'utilization', 'utlization': 'utilization', 'utilzation': 'utilization',
        'utilizaiton': 'utilization', 'utlilization': 'utilization', 'utilisaton': 'utilization',
        'scroe': 'score', 'scor': 'score', 'socre': 'score', 'scroe': 'score', 'scroe': 'score',
        'infrmation': 'information', 'infomation': 'information', 'informaton': 'information',
        'dahboard': 'dashboard', 'dashbaord': 'dashboard', 'dashbord': 'dashboard', 'dahsboard': 'dashboard',
        'anlytics': 'analytics', 'analyitcs': 'analytics', 'analytcs': 'analytics', 'anlaytics': 'analytics',
        'finanical': 'financial', 'fincaual': 'financial', 'financail': 'financial', 'finacial': 'financial',
        'halthcare': 'healthcare', 'helathcare': 'healthcare', 'healtcare': 'healthcare',
        'healthacre': 'healthcare', 'heathcare': 'healthcare', 'healtchare': 'healthcare',
        'impelmented': 'implemented', 'implmented': 'implemented', 'impelment': 'implement',
        'particualr': 'particular', 'paticular': 'particular', 'particluar': 'particular',
        'segregation': 'segregation', 'segragation': 'segregation',
        'coempelt': 'complete', 'compelte': 'complete', 'comeplete': 'complete', 'compleet': 'complete',
        'holiistic': 'holistic', 'holisitc': 'holistic',
        'remarbalby': 'remarkably', 'remarkabyl': 'remarkably', 'remarbaly': 'remarkably',
        'misspelling': 'misspelling', 'mispelling': 'misspelling', 'misspeling': 'misspelling',
        'phamracy': 'pharmacy', 'pharamcy': 'pharmacy', 'pharamacy': 'pharmacy', 'pharmcy': 'pharmacy',
        'telhealth': 'telehealth', 'telehelath': 'telehealth', 'telehalth': 'telehealth',
        'readmision': 'readmission', 'readmisson': 'readmission', 'readmisison': 'readmission',
        'comorbiidty': 'comorbidity', 'comorbidiy': 'comorbidity', 'comorbitiy': 'comorbidity',
        'chartlson': 'charlson', 'charlon': 'charlson', 'carlson': 'charlson',
        'elxihauser': 'elixhauser', 'elixhuaser': 'elixhauser', 'elixhaser': 'elixhauser',
        'acturaial': 'actuarial', 'acutarial': 'actuarial', 'actaurial': 'actuarial',
        'herfindal': 'herfindahl', 'herfindal': 'herfindahl', 'herindahl': 'herfindahl',
        'stratfication': 'stratification', 'straitfication': 'stratification',
        'polyphamacy': 'polypharmacy', 'polyphramy': 'polypharmacy',
        'forecsat': 'forecast', 'forcast': 'forecast', 'forceast': 'forecast', 'forecat': 'forecast',
        'revnue': 'revenue', 'reveune': 'revenue', 'revenu': 'revenue', 'revneue': 'revenue',
        'qualiy': 'quality', 'quaility': 'quality', 'qaulity': 'quality', 'qualitiy': 'quality',
        'intiative': 'initiative', 'initaitive': 'initiative', 'initiaitve': 'initiative',
        'performace': 'performance', 'perfromance': 'performance', 'peformance': 'performance',
        'chrnoic': 'chronic', 'chroic': 'chronic', 'chronc': 'chronic', 'chronci': 'chronic',
        'burdn': 'burden', 'bruden': 'burden', 'burdne': 'burden',
        'gendre': 'gender', 'gneder': 'gender', 'gendr': 'gender', 'gedner': 'gender',
        'appoitment': 'appointment', 'apointment': 'appointment', 'appointmnet': 'appointment',
        'servcie': 'service', 'serivce': 'service', 'servce': 'service',
        'pateint': 'patient', 'patinet': 'patient', 'paitent': 'patient', 'patietn': 'patient',
        'soudlnt': 'should not', 'shoudl': 'should', 'shold': 'should', 'shoud': 'should',
        'missign': 'missing', 'misisng': 'missing', 'missin': 'missing',
        'wehre': 'where', 'whre': 'where', 'weer': 'where',
        'shoiw': 'show', 'shwo': 'show', 'shw': 'show', 'hsow': 'show',
        'dign': 'doing', 'diong': 'doing', 'doign': 'doing',
        'thign': 'thing', 'thinsg': 'things', 'htings': 'things',
        'beter': 'better', 'bettr': 'better', 'bettter': 'better',
        'mdoldes': 'models', 'modles': 'models', 'modls': 'models', 'moedls': 'models',
        'lok': 'look', 'loko': 'look', 'loook': 'look',
        'incdue': 'include', 'incldue': 'include', 'inlcude': 'include',
        'possibel': 'possible', 'possble': 'possible', 'posible': 'possible',
        'resutls': 'results', 'reuslts': 'results', 'reults': 'results',
        'explian': 'explain', 'expalin': 'explain', 'exlpain': 'explain',
        'anlaysis': 'analysis', 'anlysis': 'analysis', 'analyis': 'analysis', 'anaylsis': 'analysis',
        'breakdwon': 'breakdown', 'berakdown': 'breakdown', 'brekdown': 'breakdown',
        'distribtuion': 'distribution', 'distrubtion': 'distribution', 'distirbution': 'distribution',
        'overivew': 'overview', 'overvew': 'overview', 'ovrview': 'overview',
        'summray': 'summary', 'sumary': 'summary', 'sumarry': 'summary',
        'cmopare': 'compare', 'comapre': 'compare', 'comprae': 'compare',
        'rase': 'race', 'raec': 'race', 'rcae': 'race', 'rce': 'race',
        'condiiton': 'condition', 'condtion': 'condition', 'conditon': 'condition',
        'conditoin': 'condition', 'condiitons': 'conditions', 'condtions': 'conditions',
        'grup': 'group', 'gruop': 'group', 'gorup': 'group', 'grp': 'group',
        'cycl': 'cycle', 'cylce': 'cycle', 'cycke': 'cycle',
        'indx': 'index', 'idex': 'index', 'indxe': 'index', 'inex': 'index',
        'spned': 'spend', 'sepnd': 'spend', 'spnde': 'spend',
        'paterns': 'patterns', 'pattrns': 'patterns', 'pattens': 'patterns',
        'reate': 'rate', 'raet': 'rate', 'rte': 'rate',
        'tpye': 'type', 'tyep': 'type', 'tye': 'type', 'tpype': 'type',
        'risc': 'risk', 'rsk': 'risk', 'rsik': 'risk',
        'charlsn': 'charlson', 'cahrson': 'charlson', 'charslon': 'charlson',
        'readmison': 'readmission', 'readmisison': 'readmission', 'readmssion': 'readmission',
        'acutarial': 'actuarial', 'acturaial': 'actuarial', 'aturial': 'actuarial',
        'vizits': 'visits', 'visist': 'visits', 'vists': 'visits', 'viists': 'visits',
        'hospotal': 'hospital', 'hopsital': 'hospital', 'hosptial': 'hospital', 'hospitl': 'hospital',
        'mesures': 'measures', 'measurs': 'measures', 'measuers': 'measures', 'meausres': 'measures',
        'admisson': 'admission', 'admision': 'admission', 'admisison': 'admission',
        'dischareg': 'discharge', 'dischrage': 'discharge', 'dischare': 'discharge',
        'netwrok': 'network', 'newtork': 'network', 'networ': 'network',
        'complinace': 'compliance', 'complaince': 'compliance', 'complianec': 'compliance',
        'eligibilty': 'eligibility', 'eligiblity': 'eligibility', 'eligbility': 'eligibility',
        'authorizaton': 'authorization', 'authrization': 'authorization', 'authroization': 'authorization',
        'benfit': 'benefit', 'benifit': 'benefit', 'benefti': 'benefit',
        'coverge': 'coverage', 'covarage': 'coverage', 'coverag': 'coverage',
        'prediciton': 'prediction', 'predicton': 'prediction', 'predction': 'prediction',
        'projecton': 'projection', 'porjection': 'projection', 'projeciton': 'projection',
        'benchamrk': 'benchmark', 'benchmrak': 'benchmark', 'benchmrk': 'benchmark',
        'avoidble': 'avoidable', 'avoidabel': 'avoidable', 'avodable': 'avoidable',
        'helth': 'health', 'heatlh': 'health', 'hleath': 'health', 'healht': 'health',
        'equiyt': 'equity', 'eqiuty': 'equity', 'equty': 'equity', 'eqituy': 'equity',
        'scorcrd': 'scorecard', 'scoecard': 'scorecard', 'scorcard': 'scorecard', 'scoreacrd': 'scorecard',
        'hedsi': 'hedis', 'hesid': 'hedis', 'heids': 'hedis',
        'losign': 'losing', 'loosing': 'losing', 'losng': 'losing',
        'mony': 'money', 'moeny': 'money', 'monye': 'money',
    }

    _HEALTHCARE_VOCABULARY = [
        'claims', 'claim', 'members', 'member', 'paid', 'billed', 'amount', 'average',
        'denied', 'denial', 'highest', 'lowest', 'risk', 'length', 'stay',
        'prescriptions', 'prescription', 'medication', 'medications',
        'total', 'number', 'what', 'provider', 'encounter', 'encounters',
        'diagnosis', 'diagnoses', 'referral', 'referrals', 'region', 'regions',
        'specialty', 'copay', 'deductible', 'coinsurance', 'enrollment',
        'utilization', 'score', 'dashboard', 'analytics', 'financial',
        'healthcare', 'pharmacy', 'telehealth', 'readmission', 'comorbidity',
        'charlson', 'elixhauser', 'actuarial', 'herfindahl', 'stratification',
        'polypharmacy', 'forecast', 'revenue', 'quality', 'initiative',
        'performance', 'chronic', 'burden', 'gender', 'appointment',
        'service', 'patient', 'should', 'missing', 'where', 'show',
        'analysis', 'breakdown', 'distribution', 'overview', 'summary',
        'compare', 'cost', 'costs', 'expensive', 'spending', 'spend',
        'trend', 'trends', 'pattern', 'patterns', 'rate', 'rates',
        'count', 'counts', 'demographic', 'demographics', 'equity',
        'disparity', 'disparities', 'population', 'clinical', 'outcome',
        'outcomes', 'preventive', 'wellness', 'screening', 'appointment',
        'appointments', 'pmpm', 'hedis', 'drg', 'hcc', 'lace',
        'race', 'ethnicity', 'age', 'plan', 'type', 'status',
        'facility', 'department', 'severity', 'category', 'class',
        'scorecard', 'high', 'low', 'moderate', 'emergency', 'inpatient',
        'outpatient', 'avoidable', 'care', 'management', 'roi',
        'concentration', 'market', 'share', 'effectiveness',
        'gap', 'gaps', 'money', 'losing', 'collection', 'reimbursement',
        'write', 'off', 'disenrollment', 'attrition', 'retention',
        'no', 'show', 'cancel', 'cancelled', 'completed',
        'active', 'inactive', 'accepting', 'panel', 'size',
        'urgent', 'stat', 'external', 'internal', 'completion',
        'by', 'per', 'for', 'each', 'across', 'between', 'within',
        'top', 'bottom', 'how', 'many', 'much', 'which', 'the',
        'most', 'least', 'best', 'worst', 'all', 'every',
        'condition', 'conditions', 'group', 'cycle', 'index',
        'spend', 'rate', 'type', 'race', 'information',
        'implement', 'implemented', 'particular', 'complete',
        'holistic', 'remarkably', 'models', 'look', 'include',
        'possible', 'results', 'explain', 'better', 'doing',
        'thing', 'things', 'should', 'where', 'what',
        'visits', 'visit', 'hospital', 'hospitals', 'measure', 'measures',
        'admission', 'admissions', 'discharge', 'discharges',
        'network', 'compliance', 'access', 'diversion',
        'primary', 'secondary', 'tertiary', 'quarterly', 'annual',
        'monthly', 'weekly', 'daily', 'yearly',
        'denied', 'approved', 'pending', 'processed',
        'inpatient', 'outpatient', 'emergent', 'non-emergent',
        'physician', 'specialist', 'generalist', 'nurse',
        'billing', 'coding', 'adjudication', 'authorization',
        'formulary', 'generic', 'brand', 'prior',
        'benefit', 'benefits', 'coverage', 'eligible', 'eligibility',
        'capitation', 'fee-for-service', 'bundled',
        'benchmark', 'percentile', 'median', 'mean',
        'correlation', 'regression', 'prediction', 'projection',
        'providers', 'members', 'patients', 'claims', 'encounters',
        'diagnoses', 'referrals', 'appointments', 'prescriptions',
        'specialists', 'hospitals', 'facilities', 'departments',
        'scores', 'amounts', 'costs', 'expenses', 'payments',
        'membership', 'usage', 'case', 'mix', 'index', 'cycle', 'month',
        'supply', 'days', 'panel', 'new', 'old', 'current', 'previous',
        'next', 'last', 'first', 'second', 'third', 'annual', 'monthly',
        'weekly', 'daily', 'quarterly', 'yearly', 'recent', 'overall',
        'specific', 'general', 'particular', 'individual', 'aggregate',
        'average', 'maximum', 'minimum', 'median', 'count', 'sum',
        'loss', 'write', 'denial', 'approval', 'pending', 'paid',
        'billed', 'allowed', 'responsibility', 'copay', 'deductible',
        'coinsurance', 'premium', 'formulary', 'generic', 'brand',
        'prior', 'capitation', 'bundled', 'fee', 'star', 'stars',
        'hedis', 'ncqa', 'cms', 'hcc', 'lace', 'drg', 'pmpm',
        'actuarial', 'mlr', 'hhi', 'cci', 'herfindahl', 'charlson',
        'elixhauser', 'polypharmacy', 'readmission', 'comorbidity',
        'stratification', 'disenrollment', 'telehealth', 'telemedicine',
        'equity', 'disparity', 'disparities', 'scorecard',
        'avoidable', 'diversion', 'concentration', 'adequacy',
        'completeness', 'timeliness', 'accuracy', 'consistency',
        'validity', 'reliability', 'integrity', 'compliance',
        'adherence', 'concordance', 'discordance', 'variance',
        'deviation', 'outlier', 'anomaly', 'threshold', 'baseline',
        'target', 'goal', 'actual', 'expected', 'projected',
    ]

    _KEYBOARD_NEIGHBORS = {
        'q': 'wa', 'w': 'qeas', 'e': 'wrds', 'r': 'etdf', 't': 'ryfg',
        'y': 'tugh', 'u': 'yijh', 'i': 'uojk', 'o': 'iplk', 'p': 'ol',
        'a': 'qwsz', 's': 'awedxz', 'd': 'serfxc', 'f': 'drtgcv',
        'g': 'ftyhvb', 'h': 'gyujbn', 'j': 'huiknm', 'k': 'jiolm',
        'l': 'kop', 'z': 'asx', 'x': 'zsdc', 'c': 'xdfv',
        'v': 'cfgb', 'b': 'vghn', 'n': 'bhjm', 'm': 'njk',
    }

    _CONTEXT_AFTER_BY = {
        'race', 'ethnicity', 'gender', 'sex', 'region', 'age', 'plan',
        'specialty', 'facility', 'department', 'severity', 'category',
        'status', 'type', 'class', 'urgency', 'state', 'county', 'city',
        'zip', 'area', 'location', 'quarter', 'month', 'year',
    }

    _CONTEXT_METRICS = {
        'rate', 'cost', 'spend', 'amount', 'count', 'total', 'average',
        'score', 'index', 'ratio', 'percentage', 'volume', 'burden',
        'pattern', 'trend', 'utilization', 'performance',
    }

    _BIGRAM_BOOST = {
        ('chronic', 'condition'): 3.0, ('chronic', 'conditions'): 3.0,
        ('condition', 'burden'): 3.0, ('high', 'risk'): 3.0,
        ('risk', 'score'): 3.0, ('risk', 'stratification'): 3.0,
        ('paid', 'amount'): 3.0, ('billed', 'amount'): 3.0,
        ('length', 'of'): 2.0, ('of', 'stay'): 2.0,
        ('denial', 'rate'): 3.0, ('readmission', 'rate'): 3.0,
        ('pharmacy', 'spend'): 3.0, ('medication', 'class'): 3.0,
        ('plan', 'type'): 3.0, ('visit', 'type'): 3.0,
        ('claim', 'type'): 3.0, ('claim', 'status'): 3.0,
        ('age', 'group'): 3.0, ('member', 'id'): 2.0,
        ('emergency', 'room'): 2.0, ('emergency', 'department'): 2.0,
        ('care', 'gap'): 3.0, ('care', 'management'): 3.0,
        ('comorbidity', 'index'): 3.0, ('case', 'mix'): 3.0,
        ('medical', 'loss'): 2.0, ('loss', 'ratio'): 2.0,
        ('hcc', 'risk'): 3.0, ('hcc', 'category'): 3.0,
        ('hcc', 'code'): 3.0, ('lace', 'score'): 3.0,
        ('lace', 'readmission'): 3.0, ('drg', 'analysis'): 3.0,
        ('quality', 'initiative'): 3.0, ('quality', 'measure'): 3.0,
        ('population', 'health'): 3.0, ('clinical', 'outcome'): 3.0,
        ('cost', 'per'): 2.0, ('per', 'member'): 2.0,
        ('revenue', 'cycle'): 3.0, ('collection', 'rate'): 3.0,
        ('provider', 'performance'): 3.0, ('provider', 'network'): 3.0,
        ('appointment', 'access'): 3.0, ('no', 'show'): 2.0,
        ('preventive', 'care'): 3.0, ('wellness', 'visit'): 3.0,
        ('losing', 'money'): 3.0, ('avoidable', 'er'): 3.0,
        ('actuarial', 'pmpm'): 3.0, ('charlson', 'comorbidity'): 3.0,
        ('deep', 'dive'): 2.0, ('how', 'many'): 2.0,
        ('top', 'providers'): 2.0, ('top', 'diagnoses'): 2.0,
        ('referral', 'pattern'): 3.0, ('enrollment', 'membership'): 3.0,
        ('telehealth', 'usage'): 3.0, ('er', 'utilization'): 3.0,
        ('health', 'equity'): 3.0, ('equity', 'scorecard'): 3.0,
        ('polypharmacy', 'by'): 2.0, ('herfindahl', 'index'): 3.0,
        ('hedis', 'quality'): 3.0, ('star', 'rating'): 3.0,
        ('cost', 'effectiveness'): 3.0, ('value', 'based'): 2.0,
    }

    _SYNONYM_MAP = {
        'hemorrhaging cash': 'losing money',
        'bleeding money': 'losing money',
        'burning cash': 'losing money',
        'wasting money': 'losing money',
        'financial hemorrhage': 'losing money',
        'bed utilization': 'length of stay',
        'bed days': 'length of stay',
        'patient churn': 'disenrollment',
        'member churn': 'disenrollment',
        'member turnover': 'disenrollment',
        'doc visits': 'encounters',
        'doctor visits': 'encounters',
        'physician visits': 'encounters',
        'office visits': 'encounters',
        'meds': 'prescriptions',
        'drugs': 'prescriptions',
        'scripts': 'prescriptions',
        'er trips': 'emergency visits',
        'er usage': 'er utilization',
        'hospital stays': 'inpatient admissions',
        'hospital admissions': 'inpatient admissions',
        'drug interactions': 'polypharmacy',
        'pill burden': 'polypharmacy',
        'med count': 'polypharmacy',
        'readmit': 'readmission',
        'bounce back': 'readmission',
        'came back': 'readmission',
        'return visit': 'readmission',
        'missed appointments': 'no show',
        'skipped appointments': 'no show',
        'flaked': 'no show',
        'docs': 'providers',
        'physicians': 'providers',
        'doctors': 'providers',
        'specialists': 'providers',
        'enrollees': 'members',
        'patients': 'members',
        'beneficiaries': 'members',
        'subscribers': 'members',
        'insureds': 'members',
        'lives': 'members',
        'covered lives': 'members',
        'rejects': 'denials',
        'rejected claims': 'denied claims',
        'turned down': 'denied',
        'kicked back': 'denied',
        'unpaid claims': 'denied claims',
        'expensive patients': 'high cost members',
        'sickest patients': 'highest risk members',
        'sickest members': 'highest risk members',
        'healthiest members': 'lowest risk members',
        'healthiest patients': 'lowest risk members',
        'frequent flyers': 'high utilization members',
        'super utilizers': 'high utilization members',
        'heavy users': 'high utilization members',
        'wellness checks': 'annual wellness visits',
        'preventive checkups': 'preventive care',
        'health checks': 'preventive screening',
        'virtual visits': 'telehealth',
        'video visits': 'telehealth',
        'remote visits': 'telehealth',
        'online care': 'telehealth',
        'out of pocket': 'member responsibility',
        'oop': 'member responsibility',
        'copays': 'copay',
        'deductibles': 'deductible',
        'premium': 'plan cost',
        'disease burden': 'chronic condition burden',
        'illness burden': 'chronic condition burden',
        'comorbid load': 'comorbidity index',
        'risk profile': 'risk stratification',
        'risk level': 'risk stratification',
        'risk bucket': 'risk tier',
        'risk band': 'risk tier',
        'star scores': 'star rating',
        'cms stars': 'star rating',
        'quality scores': 'quality measure',
        'quality metrics': 'quality measure',
        'cost savings': 'cost effectiveness',
        'money saved': 'cost effectiveness',
        'value for money': 'cost effectiveness',
        'network adequacy': 'provider concentration',
        'provider access': 'provider concentration',
        'referral leakage': 'external referrals',
        'out of network referrals': 'external referrals',
    }

    _COMPOUND_WORDS = {
        'healthcare': 'healthcare', 'healthequity': 'health equity',
        'casemix': 'case mix', 'riskstratification': 'risk stratification',
        'riskscore': 'risk score', 'risktier': 'risk tier',
        'claimstatus': 'claim status', 'claimtype': 'claim type',
        'plantype': 'plan type', 'visittype': 'visit type',
        'lengthofstay': 'length of stay', 'readmissionrate': 'readmission rate',
        'denialrate': 'denial rate', 'noshow': 'no show', 'noshowrate': 'no show rate',
        'agegroup': 'age group', 'riskfactor': 'risk factor',
        'costper': 'cost per', 'permember': 'per member',
        'permonth': 'per month', 'memberresponsibility': 'member responsibility',
        'paidamount': 'paid amount', 'billedamount': 'billed amount',
        'allowedamount': 'allowed amount', 'medicalloss': 'medical loss',
        'lossratio': 'loss ratio', 'revenuecycle': 'revenue cycle',
        'caregap': 'care gap', 'caremanagement': 'care management',
        'populationhealth': 'population health', 'clinicaloutcome': 'clinical outcome',
        'providernetwork': 'provider network', 'providerconcentration': 'provider concentration',
        'ervisit': 'er visit', 'ervisits': 'er visits',
        'erutilization': 'er utilization', 'erdiversion': 'er diversion',
        'emergencyroom': 'emergency room', 'emergencydepartment': 'emergency department',
        'annualwellness': 'annual wellness', 'preventivecare': 'preventive care',
        'chroniccondition': 'chronic condition', 'chronicconditions': 'chronic conditions',
        'drugclass': 'drug class', 'medicationclass': 'medication class',
        'servicetype': 'service type', 'servicecategory': 'service category',
        'referralpattern': 'referral pattern', 'appointmentaccess': 'appointment access',
        'pharmacyspend': 'pharmacy spend', 'telehealthusage': 'telehealth usage',
        'qualityinitiative': 'quality initiative', 'hcccode': 'hcc code',
        'hcccategory': 'hcc category', 'lacescore': 'lace score',
        'drganalysis': 'drg analysis', 'costeffectiveness': 'cost effectiveness',
        'valuebased': 'value based',
    }

    _MERGE_PHRASES = {
        ('health', 'care'): 'healthcare',
        ('case', 'mix'): 'case mix',
        ('no', 'show'): 'no show',
        ('length', 'of', 'stay'): 'length of stay',
        ('per', 'member', 'per', 'month'): 'per member per month',
        ('risk', 'score'): 'risk score',
        ('risk', 'tier'): 'risk tier',
        ('paid', 'amount'): 'paid amount',
        ('billed', 'amount'): 'billed amount',
        ('denial', 'rate'): 'denial rate',
        ('claim', 'status'): 'claim status',
        ('claim', 'type'): 'claim type',
        ('plan', 'type'): 'plan type',
        ('visit', 'type'): 'visit type',
        ('age', 'group'): 'age group',
        ('care', 'gap'): 'care gap',
        ('care', 'management'): 'care management',
        ('loss', 'ratio'): 'loss ratio',
        ('star', 'rating'): 'star rating',
    }

    _IMPLICIT_BY_PATTERNS = [
        (r'\bacross\s+(race|ethnicity|gender|sex|region|area|location|age\s*group|age|plan\s*type|plan|'
         r'specialty|facility|department|severity|category|status|zip|state|county|city|'
         r'visit\s*type|encounter\s*type|claim\s*type|claim\s*status|service\s*type|'
         r'medication\s*class|drug\s*class|urgency)', r'by \1'),
        (r'\bper\s+(race|ethnicity|gender|sex|region|area|location|age\s*group|age|plan\s*type|plan|'
         r'specialty|facility|department|severity|category|status)', r'by \1'),
        (r'\bfor\s+each\s+(race|ethnicity|gender|sex|region|area|location|age\s*group|age|plan\s*type|plan|'
         r'specialty|facility|department|severity|category|status)', r'by \1'),
        (r'\bbroken\s+down\s+by\s+(race|ethnicity|gender|sex|region|area|location|age\s*group|age|plan\s*type|plan|'
         r'specialty|facility|department|severity|category|status)', r'by \1'),
        (r'\bsplit\s+by\s+(race|ethnicity|gender|sex|region|area|location|age\s*group|age|plan\s*type|plan|'
         r'specialty|facility|department|severity|category|status)', r'by \1'),
        (r'\bsegmented\s+by\s+(race|ethnicity|gender|sex|region|area|location|age\s*group|age|plan\s*type|plan|'
         r'specialty|facility|department|severity|category|status)', r'by \1'),
        (r'\bstratified\s+by\s+(race|ethnicity|gender|sex|region|area|location|age\s*group|age|plan\s*type|plan|'
         r'specialty|facility|department|severity|category|status)', r'by \1'),
        (r'\bcompare\s+(race|ethnicity|gender|sex|region|area|location|age\s*group|age|plan\s*type|plan)s?\b',
         r'by \1'),
        (r'\b(?:among|between)\s+(?:different\s+)?(race|ethnicity|gender|sex|region|area|location|'
         r'age\s*group|age|plan\s*type|plan)s?\b', r'by \1'),
        (r'\b(?:grouped|grouping)\s+by\s+(race|ethnicity|gender|sex|region|area|location|age\s*group|age|plan\s*type|plan|'
         r'specialty|facility|department|severity|category|status)', r'by \1'),
    ]

    _TRIGRAM_FREQ = {}

    @classmethod
    def _build_trigram_model(cls):
        if cls._TRIGRAM_FREQ:
            return
        healthcare_phrases = [
            'chronic condition burden by race', 'high risk members by region',
            'denial rate by plan type', 'claims cost by gender',
            'pharmacy spend by age group', 'readmission rate by severity',
            'er utilization by region', 'length of stay by facility',
            'risk stratification by age group', 'cost per member by plan type',
            'telehealth usage by gender', 'preventive care by race',
            'care gap by region', 'quality initiative by specialty',
            'revenue cycle by claim status', 'population health by race',
            'clinical outcomes by gender', 'appointment access by department',
            'referral pattern by specialty', 'enrollment membership by plan type',
            'medication class by age group', 'provider performance by specialty',
            'cms hcc risk by race', 'charlson comorbidity by age group',
            'lace readmission by region', 'drg analysis by facility',
            'actuarial pmpm by plan type', 'hedis quality by region',
            'cost effectiveness by gender', 'polypharmacy by age group',
            'avoidable er by race', 'losing money by region',
            'health equity scorecard', 'provider concentration by specialty',
            'risk score distribution by race', 'member responsibility by plan',
            'average paid amount by gender', 'total claims by region',
            'pmpm cost by plan type', 'hcc category distribution',
            'care management roi by region', 'claims severity by race',
        ]
        for phrase in healthcare_phrases:
            words = phrase.split()
            for i in range(len(words) - 2):
                tri = (words[i], words[i + 1], words[i + 2])
                cls._TRIGRAM_FREQ[tri] = cls._TRIGRAM_FREQ.get(tri, 0) + 1
            for i in range(len(words) - 1):
                bi = (words[i], words[i + 1])
                cls._TRIGRAM_FREQ[bi] = cls._TRIGRAM_FREQ.get(bi, 0) + 1

    @classmethod
    def _trigram_score(cls, words):
        cls._build_trigram_model()
        if len(words) < 2:
            return 0.0
        score = 0.0
        for i in range(len(words) - 2):
            tri = (words[i], words[i + 1], words[i + 2])
            if tri in cls._TRIGRAM_FREQ:
                score += cls._TRIGRAM_FREQ[tri] * 2.0
        for i in range(len(words) - 1):
            bi = (words[i], words[i + 1])
            if bi in cls._TRIGRAM_FREQ:
                score += cls._TRIGRAM_FREQ[bi] * 1.0
        return score

    class _BKTreeNode:
        __slots__ = ('word', 'children')
        def __init__(self, word):
            self.word = word
            self.children = {}

    _bk_tree_root = None

    @classmethod
    def _build_bk_tree(cls):
        if cls._bk_tree_root is not None:
            return
        vocab = cls._HEALTHCARE_VOCABULARY
        if not vocab:
            return
        cls._bk_tree_root = cls._BKTreeNode(vocab[0])
        for word in vocab[1:]:
            cls._bk_tree_insert(cls._bk_tree_root, word)

    @classmethod
    def _bk_tree_insert(cls, node, word):
        d = cls._damerau_levenshtein(node.word, word)
        if d == 0:
            return
        if d in node.children:
            cls._bk_tree_insert(node.children[d], word)
        else:
            node.children[d] = cls._BKTreeNode(word)

    @classmethod
    def _bk_tree_search(cls, node, target, max_dist):
        if node is None:
            return []
        results = []
        d = cls._damerau_levenshtein(node.word, target)
        if d <= max_dist:
            results.append((node.word, d))
        for i in range(max(1, d - max_dist), d + max_dist + 1):
            if i in node.children:
                results.extend(cls._bk_tree_search(node.children[i], target, max_dist))
        return results

    _VOCAB_SET = None

    @classmethod
    def _get_vocab_set(cls):
        if cls._VOCAB_SET is None:
            cls._VOCAB_SET = set(cls._HEALTHCARE_VOCABULARY)
        return cls._VOCAB_SET

    _learned_corrections = {}

    @staticmethod
    def _edit_distance(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return IntelligentPipeline._edit_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                ins = prev[j + 1] + 1
                dele = curr[j] + 1
                sub = prev[j] + (0 if c1 == c2 else 1)
                curr.append(min(ins, dele, sub))
            prev = curr
        return prev[len(s2)]

    @staticmethod
    def _damerau_levenshtein(s1: str, s2: str) -> int:
        d = {}
        len1 = len(s1)
        len2 = len(s2)
        for i in range(-1, len1 + 1):
            d[(i, -1)] = i + 1
        for j in range(-1, len2 + 1):
            d[(-1, j)] = j + 1
        for i in range(len1):
            for j in range(len2):
                cost = 0 if s1[i] == s2[j] else 1
                d[(i, j)] = min(
                    d[(i - 1, j)] + 1,
                    d[(i, j - 1)] + 1,
                    d[(i - 1, j - 1)] + cost,
                )
                if i > 0 and j > 0 and s1[i] == s2[j - 1] and s1[i - 1] == s2[j]:
                    d[(i, j)] = min(d[(i, j)], d[(i - 2, j - 2)] + cost)
        return d[(len1 - 1, len2 - 1)]

    @staticmethod
    def _metaphone(word: str) -> str:
        if not word:
            return ''
        w = word.upper()
        if w[:2] in ('AE', 'GN', 'KN', 'PN', 'WR'):
            w = w[1:]
        if w[0] == 'X':
            w = 'S' + w[1:]
        result = []
        i = 0
        while i < len(w) and len(result) < 6:
            c = w[i]
            if c in 'AEIOU':
                if i == 0:
                    result.append(c)
            elif c == 'B':
                if not (i > 0 and w[i - 1] == 'M'):
                    result.append('B')
            elif c == 'C':
                if i + 1 < len(w) and w[i + 1] in 'EIY':
                    result.append('S')
                else:
                    result.append('K')
            elif c == 'D':
                if i + 1 < len(w) and w[i + 1] in 'GEI':
                    result.append('J')
                else:
                    result.append('T')
            elif c in 'FLJMNR':
                if not result or result[-1] != c:
                    result.append(c)
            elif c == 'G':
                if i + 1 < len(w) and w[i + 1] in 'EIY':
                    result.append('J')
                elif not (i > 0 and w[i - 1] == 'G'):
                    result.append('K')
            elif c == 'H':
                if i + 1 < len(w) and w[i + 1] in 'AEIOU' and (i == 0 or w[i - 1] not in 'AEIOU'):
                    result.append('H')
            elif c == 'K':
                if not (i > 0 and w[i - 1] == 'C'):
                    result.append('K')
            elif c == 'P':
                if i + 1 < len(w) and w[i + 1] == 'H':
                    result.append('F')
                    i += 1
                else:
                    result.append('P')
            elif c == 'Q':
                result.append('K')
            elif c == 'S':
                if i + 1 < len(w) and w[i + 1] == 'H':
                    result.append('X')
                    i += 1
                elif i + 2 < len(w) and w[i:i + 3] == 'SIO':
                    result.append('X')
                elif i + 2 < len(w) and w[i:i + 3] == 'SIA':
                    result.append('X')
                else:
                    result.append('S')
            elif c == 'T':
                if i + 1 < len(w) and w[i + 1] == 'H':
                    result.append('0')
                    i += 1
                elif i + 2 < len(w) and w[i:i + 3] in ('TIA', 'TIO'):
                    result.append('X')
                else:
                    result.append('T')
            elif c == 'V':
                result.append('F')
            elif c in 'WY':
                if i + 1 < len(w) and w[i + 1] in 'AEIOU':
                    result.append(c)
            elif c == 'X':
                result.append('KS')
            elif c == 'Z':
                result.append('S')
            i += 1
        return ''.join(result)

    @staticmethod
    def _keyboard_distance(c1: str, c2: str) -> float:
        if c1 == c2:
            return 0.0
        neighbors = IntelligentPipeline._KEYBOARD_NEIGHBORS
        if c1 in neighbors and c2 in neighbors.get(c1, ''):
            return 0.5
        if c2 in neighbors and c1 in neighbors.get(c2, ''):
            return 0.5
        return 1.0

    def _score_candidate(self, typo: str, candidate: str, prev_word: str, next_word: str) -> float:
        dl_dist = self._damerau_levenshtein(typo, candidate)
        ed_dist = self._edit_distance(typo, candidate)
        dist = min(dl_dist, ed_dist)

        base_score = 10.0 - dist * 2.5

        if typo and candidate and typo[0] == candidate[0]:
            base_score += 1.5
        if typo and candidate and typo[:2] == candidate[:2]:
            base_score += 1.0

        if len(typo) > 1 and len(candidate) > 1:
            kb_sim = 0.0
            for i in range(min(len(typo), len(candidate))):
                if typo[i] != candidate[i]:
                    kb_sim += (1.0 - self._keyboard_distance(typo[i], candidate[i])) * 0.5
            base_score += kb_sim

        meta_typo = self._metaphone(typo)
        meta_cand = self._metaphone(candidate)
        if meta_typo and meta_cand:
            if meta_typo == meta_cand:
                base_score += 2.0
            elif meta_typo[:3] == meta_cand[:3]:
                base_score += 1.0

        if prev_word == 'by' and candidate in self._CONTEXT_AFTER_BY:
            base_score += 4.0
        if prev_word == 'by' and candidate in self._CONTEXT_METRICS:
            base_score -= 1.0

        if prev_word in ('denial', 'readmission', 'collection', 'no-show', 'wellness', 'avoidable'):
            if candidate == 'rate':
                base_score += 3.0
        if next_word in ('condition', 'conditions', 'disease'):
            if candidate == 'chronic':
                base_score += 3.0
        if prev_word in ('age', 'plan', 'visit', 'claim', 'medication', 'drug', 'service'):
            if candidate in ('group', 'type', 'status', 'class', 'category'):
                base_score += 3.0
        if prev_word in ('risk', 'lace', 'hcc', 'comorbidity'):
            if candidate == 'score':
                base_score += 3.0

        if prev_word:
            bigram_key = (prev_word, candidate)
            if bigram_key in self._BIGRAM_BOOST:
                base_score += self._BIGRAM_BOOST[bigram_key]

        if next_word:
            bigram_key_fwd = (candidate, next_word)
            if bigram_key_fwd in self._BIGRAM_BOOST:
                base_score += self._BIGRAM_BOOST[bigram_key_fwd]

        sorted_typo = ''.join(sorted(typo))
        sorted_cand = ''.join(sorted(candidate))
        if sorted_typo == sorted_cand:
            base_score += 2.0

        return base_score

    def _apply_synonyms(self, question: str) -> str:
        q = question.lower()
        for phrase, replacement in sorted(self._SYNONYM_MAP.items(), key=lambda x: -len(x[0])):
            if phrase in q:
                q = q.replace(phrase, replacement)
        return q

    def _apply_compound_words(self, question: str) -> str:
        words = question.lower().split()
        result = []
        i = 0
        while i < len(words):
            clean = words[i].strip('?.,!;:')

            if clean in self._COMPOUND_WORDS:
                expanded = self._COMPOUND_WORDS[clean]
                suffix = words[i][len(clean):] if len(words[i]) > len(clean) else ''
                result.append(expanded + suffix)
                i += 1
                continue

            merged = False
            for phrase_tuple, merged_form in self._MERGE_PHRASES.items():
                plen = len(phrase_tuple)
                if i + plen <= len(words):
                    candidate = tuple(words[j].strip('?.,!;:') for j in range(i, i + plen))
                    if candidate == phrase_tuple:
                        result.append(merged_form)
                        i += plen
                        merged = True
                        break
            if not merged:
                result.append(words[i])
                i += 1
        return ' '.join(result)

    def _apply_implicit_dimensions(self, question: str) -> str:
        q = question.lower()
        for pattern, replacement in self._IMPLICIT_BY_PATTERNS:
            q = re.sub(pattern, replacement, q)
        return q

    def _correct_typos(self, question: str) -> str:
        question = self._apply_synonyms(question)

        question = self._apply_compound_words(question)

        question = self._apply_implicit_dimensions(question)

        self._build_bk_tree()

        words = question.lower().split()
        corrected_words = []
        vocab_set = self._get_vocab_set()

        _stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                      'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall',
                      'can', 'could', 'may', 'might', 'must', 'i', 'me', 'my', 'we', 'our',
                      'you', 'your', 'it', 'its', 'to', 'of', 'in', 'on', 'at', 'and', 'or',
                      'not', 'no', 'so', 'if', 'up', 'out', 'off', 'but', 'this', 'that',
                      'with', 'from', 'into', 'about', 'as', 'than', 'then', 'also', 'very',
                      'just', 'too', 'now', 'here', 'there', 'am', 'get', 'got', 'go'}

        for idx, word in enumerate(words):
            clean = word.strip('?.,!;:')
            suffix = word[len(clean):] if len(word) > len(clean) else ''

            if clean in self._TYPO_CORRECTIONS:
                corrected_words.append(self._TYPO_CORRECTIONS[clean] + suffix)
                continue

            if clean in self._learned_corrections:
                corrected_words.append(self._learned_corrections[clean] + suffix)
                continue

            if len(clean) <= 2 or clean in _stopwords or clean.isdigit():
                corrected_words.append(word)
                continue

            if clean in vocab_set:
                corrected_words.append(word)
                continue

            prev_word = corrected_words[-1].strip('?.,!;:') if corrected_words else ''
            next_word = words[idx + 1].strip('?.,!;:') if idx + 1 < len(words) else ''

            max_dist = 1 if len(clean) <= 4 else (2 if len(clean) <= 7 else 3)

            bk_results = self._bk_tree_search(self._bk_tree_root, clean, max_dist)

            candidates = []
            for vocab_word, raw_dist in bk_results:
                score = self._score_candidate(clean, vocab_word, prev_word, next_word)
                candidates.append((vocab_word, raw_dist, score))

            if not candidates and len(clean) >= 4:
                meta_typo = self._metaphone(clean)
                wider_results = self._bk_tree_search(self._bk_tree_root, clean, max_dist + 2)
                for vocab_word, raw_dist in wider_results:
                    if len(vocab_word) >= 4 and self._metaphone(vocab_word) == meta_typo:
                        score = self._score_candidate(clean, vocab_word, prev_word, next_word)
                        candidates.append((vocab_word, raw_dist, score))

            if candidates:
                candidates.sort(key=lambda x: (-x[2], x[1]))
                best = candidates[0]

                if len(candidates) >= 2:
                    second = candidates[1]
                    if best[2] - second[2] < 1.0:
                        test_words_best = corrected_words[-2:] + [best[0]]
                        test_words_second = corrected_words[-2:] + [second[0]]
                        tri_best = self._trigram_score([w.strip('?.,!;:') for w in test_words_best])
                        tri_second = self._trigram_score([w.strip('?.,!;:') for w in test_words_second])
                        if tri_second > tri_best + 0.5:
                            best = second

                if best[2] >= 5.0:
                    corrected_words.append(best[0] + suffix)
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)

        result = ' '.join(corrected_words)
        return result

    def _learn_correction(self, typo: str, correction: str, success: bool):
        if success:
            self._learned_corrections[typo] = correction
            logger.debug("Learned correction: '%s' → '%s'", typo, correction)
        elif typo in self._learned_corrections and self._learned_corrections[typo] == correction:
            del self._learned_corrections[typo]
            logger.debug("Unlearned bad correction: '%s' → '%s'", typo, correction)

    _SUPERLATIVE_COLUMN_MAP = [
        ('paid amount', 'PAID_AMOUNT', 'claims', 10),
        ('paid', 'PAID_AMOUNT', 'claims', 3),
        ('billed amount', 'BILLED_AMOUNT', 'claims', 10),
        ('billed', 'BILLED_AMOUNT', 'claims', 5),
        ('allowed amount', 'ALLOWED_AMOUNT', 'claims', 10),
        ('allowed', 'ALLOWED_AMOUNT', 'claims', 5),
        ('cost claim', 'PAID_AMOUNT', 'claims', 8),
        ('cost', 'PAID_AMOUNT', 'claims', 2),
        ('risk score', 'RISK_SCORE', 'members', 10),
        ('risk', 'RISK_SCORE', 'members', 5),
        ('length of stay', 'LENGTH_OF_STAY', 'encounters', 10),
        ('los', 'LENGTH_OF_STAY', 'encounters', 8),
        ('copay', 'COPAY', 'claims', 10),
        ('coinsurance', 'COINSURANCE', 'claims', 10),
        ('deductible', 'DEDUCTIBLE', 'claims', 10),
        ('panel size', 'PANEL_SIZE', 'providers', 10),
        ('days supply', 'DAYS_SUPPLY', 'prescriptions', 10),
        ('rvu', 'RVU', 'cpt_codes', 10),
        ('medication cost', 'COST', 'prescriptions', 10),
        ('prescription cost', 'COST', 'prescriptions', 10),
        ('expensive medication', 'COST', 'prescriptions', 10),
        ('expensive prescription', 'COST', 'prescriptions', 10),
        ('medication', 'COST', 'prescriptions', 3),
        ('prescription', 'COST', 'prescriptions', 3),
    ]

    _GROUPING_INDICATORS = [
        'by ', 'per ', 'for each ', 'in each ', 'across ',
        'provider', 'specialty', 'region', 'plan', 'department',
        'facility', 'type', 'status', 'gender', 'age group',
        'category', 'class', 'medication name', 'drug name',
        'caseload', 'case load', 'docs', 'which doc',
        'which diagnos', 'which member', 'which patient',
        'which claim', 'which procedure', 'which condition',
        'diagnos', 'trend', 'over time', 'processing time',
        'greater than', 'more than', 'less than', 'above', 'below',
        'top ', 'top 5', 'top 10', 'top 20', 'ranking',
        'break down', 'breakdown', 'distribution',
        'drive ', 'driving', 'cause', 'reason',
    ]

    def _try_superlative_scalar(self, question: str, t0: float, session_id: str) -> Optional[Dict[str, Any]]:
        q_lower = question.lower()

        max_words = ['highest', 'maximum', 'max ', 'biggest', 'largest', 'most expensive', 'greatest']
        min_words = ['lowest', 'minimum', 'min ', 'smallest', 'least expensive', 'least costly', 'cheapest']
        is_max = any(w in q_lower for w in max_words)
        is_min = any(w in q_lower for w in min_words)
        if not is_max and not is_min:
            return None

        for gi in self._GROUPING_INDICATORS:
            if gi in q_lower:
                if gi in ('medication', 'prescription') and ('expensive' in q_lower or 'costly' in q_lower):
                    continue
                return None

        best_col = None
        best_table = None
        best_priority = -1
        for phrase, col, table, priority in self._SUPERLATIVE_COLUMN_MAP:
            if phrase in q_lower and priority > best_priority:
                best_col = col
                best_table = table
                best_priority = priority

        if not best_col:
            return None

        agg_fn = 'MAX' if is_max else 'MIN'
        sql = f"SELECT {agg_fn}(CAST({best_col} AS REAL)) AS {agg_fn.lower()}_{best_col.lower()} FROM {best_table}"

        try:
            rows, columns, error = self._execute_sql(sql)
            if error or not rows:
                return None
            narrative = self.narrative_engine.generate(
                question=question, sql=sql, rows=rows, columns=columns,
                intent={'intent': 'aggregation', 'confidence': 0.98},
            )
            result = {
                'sql': sql,
                'rows': rows,
                'columns': columns,
                'row_count': len(rows),
                'error': None,
                'source': 'superlative_scalar',
                'narrative': narrative,
                'confidence': {'grade': 'A', 'overall': 0.98},
                'confidence_grade': 'A',
                'confidence_overall': 0.98,
                'cache_hit': False,
                'latency_ms': round((time.time() - t0) * 1000),
                'suggestions': [],
                'anomalies': [],
                'dashboard': {
                    'chart_type': 'metric',
                    'chart_config': {},
                    'title': question,
                    'subtitle': '',
                    'secondary_chart': None,
                    'color_scheme': 'kp_blue',
                },
            }
            self._update_conversation(session_id, question, result, rows, columns)
            return result
        except Exception as e:
            logger.warning("Superlative scalar interceptor error: %s", e)
            return None

    def _validate_sql_matches_question(self, question: str, sql: str) -> bool:
        if not sql:
            return True
        _q = question.lower()
        _sql = sql.upper()
        _sql_lower = sql.lower()

        _rank_words = [
            'most expensive', 'costliest', 'busiest', 'longest', 'biggest',
            'most commonly', 'most frequently', 'primary reasons',
            'most denied', 'most revenue', 'most utilization',
            'top facility', 'top provider', 'top diagnos', 'top 5', 'top 10',
            'ranked', 'largest to smallest', 'smallest to largest',
        ]
        if any(w in _q for w in _rank_words) and 'ORDER BY' not in _sql:
            logger.info("SQL REJECTED — question has rank indicator but SQL missing ORDER BY")
            return False

        _denial_words = ['denied', 'denial', 'reject', 'rejected', 'rejection',
                         'clean claim', 'claim denial', 'denial rate', 'denial frequency']
        if any(w in _q for w in _denial_words):
            if 'DENIED' not in _sql and 'CLAIM_STATUS' not in _sql:
                logger.info("SQL REJECTED — question mentions denial but SQL missing DENIED/CLAIM_STATUS")
                return False

        import re as _re
        _groupby_q_patterns = [
            r'\bper\s+(?:insurance\s+)?(?:plan|region|specialty|department|type|category|race|age)',
            r'\bby\s+(?:plan\s+type|region|specialty|department|visit\s+type|category|race|age|diagnosis)',
            r'broken\s+(?:down|out)\s+by\b',
            r'stratified\s+by\b',
            r'segmented\s+by\b',
            r'break\s+down\s+\w+\s+by\b',
        ]
        if any(_re.search(p, _q) for p in _groupby_q_patterns) and 'GROUP BY' not in _sql:
            logger.info("SQL REJECTED — question has GROUP BY pattern but SQL missing GROUP BY")
            return False

        _COL_MAP = {
            'copay': 'COPAY', 'copayment': 'COPAY', 'out of pocket': 'COPAY',
            'coinsurance': 'COINSURANCE',
            'deductible': 'DEDUCTIBLE',
            'billed amount': 'BILLED_AMOUNT', 'billed charge': 'BILLED_AMOUNT',
            'allowed amount': 'ALLOWED_AMOUNT', 'contracted rate': 'ALLOWED_AMOUNT',
            'length of stay': 'LENGTH_OF_STAY', 'los ': 'LENGTH_OF_STAY',
            'stays': 'LENGTH_OF_STAY', 'stay duration': 'LENGTH_OF_STAY',
            'risk score': 'RISK_SCORE', 'acuity': 'RISK_SCORE', 'hcc score': 'RISK_SCORE',
            'complexity': 'RISK_SCORE',
            'panel size': 'PANEL_SIZE',
            'plan type': 'PLAN_TYPE', 'insurance plan': 'PLAN_TYPE',
            'medication class': 'MEDICATION_CLASS', 'drug class': 'MEDICATION_CLASS',
            'therapeutic class': 'MEDICATION_CLASS',
            'specialty': 'SPECIALTY', 'specialties': 'SPECIALTY',
            'race': 'RACE',
            'rvu': 'RVU',
            'days supply': 'DAYS_SUPPLY',
        }
        for kw, col in _COL_MAP.items():
            if kw in _q and col not in _sql:
                logger.info("SQL REJECTED — question mentions '%s' but SQL missing %s", kw, col)
                return False

        _count_phrases = ['claims count', 'total count', ' count asap', 'count of ',
                          'how many', 'number of', 'total number']
        if any(w in _q for w in _count_phrases):
            if 'SUM(' in _sql and 'COUNT(' not in _sql:
                logger.info("SQL REJECTED — question asks for COUNT but SQL has SUM")
                return False

        _ENTITY_MAP = {
            'diabetic': 'diagnos', 'diabetes': 'diagnos',
            'provider specialty': 'providers', 'provider specialties': 'providers',
            'medication class': 'prescriptions',
        }
        for kw, req in _ENTITY_MAP.items():
            if kw in _q and req not in _sql_lower:
                logger.info("SQL REJECTED — question mentions '%s' but SQL missing '%s'", kw, req)
                return False

        if 'compare' in _q or 'comparison' in _q:
            if 'paid amount' in _q and 'billed amount' in _q:
                if 'PAID_AMOUNT' not in _sql or 'BILLED_AMOUNT' not in _sql:
                    logger.info("SQL REJECTED — comparison question missing one of the compared columns")
                    return False

        if 'unique' in _q or 'distinct' in _q:
            if 'DISTINCT' not in _sql:
                logger.info("SQL REJECTED — question asks for unique/distinct but SQL missing DISTINCT")
                return False

        _diagnos_words = ['diagnosed', 'diagnosis', 'diagnoses', 'conditions across',
                          'commonly diagnosed', 'top 5 diagnos', 'top diagnos',
                          'diagnosis category']
        if any(w in _q for w in _diagnos_words):
            if 'diagnos' not in _sql_lower and 'icd' not in _sql_lower:
                logger.info("SQL REJECTED — question mentions diagnosis but SQL missing diagnoses/ICD")
                return False

        _avg_words = ['average cost', 'avg cost', 'average claim cost',
                      'average cost per member']
        if any(w in _q for w in _avg_words):
            if 'AVG(' not in _sql and 'avg' not in _sql_lower:
                logger.info("SQL REJECTED — question asks for AVG but SQL missing AVG function")
                return False

        if 'submission' in _q and ('rejected' in _q or 'reject' in _q):
            if 'DENIED' not in _sql and 'CLAIM_STATUS' not in _sql:
                logger.info("SQL REJECTED — submission rejection question missing DENIED/CLAIM_STATUS")
                return False

        _rx_words = ['prescription', 'medication', 'drug', 'rx']
        _rx_cost_words = ['cost of prescription', 'cost per prescription',
                          'expensive medication', 'medication cost',
                          'drug cost', 'prescription cost']
        if any(w in _q for w in _rx_cost_words):
            if 'COST' not in _sql or 'prescriptions' not in _sql_lower:
                logger.info("SQL REJECTED — prescription cost question missing COST/prescriptions")
                return False
        if 'days supply' in _q or 'day supply' in _q:
            if 'DAYS_SUPPLY' not in _sql:
                logger.info("SQL REJECTED — days supply question missing DAYS_SUPPLY column")
                return False

        import re as _re2
        _numeric_comparison_phrases = [
            (r'(?:greater\s+than|more\s+than|over|above|exceed(?:s|ing)?)\s+\$?([\d,]+)', '>'),
            (r'(?:less\s+than|under|below)\s+\$?([\d,]+)', '<'),
            (r'>\s*\$?([\d,]+)', '>'),
            (r'<\s*\$?([\d,]+)', '<'),
        ]
        for pattern, _op in _numeric_comparison_phrases:
            m = _re2.search(pattern, _q)
            if m:
                _num_str = m.group(1).replace(',', '')
                if _num_str.isdigit():
                    _num_val = _num_str
                    has_where_numeric = bool(_re2.search(r'WHERE\b.*(?:>|>=|<|<=|BETWEEN)\s*\d', _sql, _re2.IGNORECASE))
                    has_num_in_sql = _num_val in _sql
                    if not has_where_numeric and not has_num_in_sql:
                        logger.info("SQL REJECTED — question has numeric comparison '%s %s' but SQL has no WHERE with numeric filter", _op, _num_val)
                        return False

        if ('member' in _q or 'patient' in _q) and ('claim' in _q):
            _amount_words = ['greater than', 'more than', 'over', 'above', 'exceed', 'less than', 'under', 'below']
            if any(w in _q for w in _amount_words):
                _cost_cols = ['PAID_AMOUNT', 'BILLED_AMOUNT', 'ALLOWED_AMOUNT', 'MEMBER_RESPONSIBILITY']
                if not any(c in _sql for c in _cost_cols) and 'CLAIM_ID' not in _sql:
                    logger.info("SQL REJECTED — member+claims+amount comparison but SQL missing cost columns")
                    return False

        return True

    def _try_query_intelligence(self, question: str, t0: float, session_id: str) -> Optional[Dict[str, Any]]:
        if not self.query_intelligence:
            return None
        q_lower = question.lower()
        skip_patterns = [
            'dashboard', 'report', 'forecast', 'predict', 'trend',
            'compare.*vs', 'year over year', 'month over month',
            'what should we', 'recommend', 'suggest', 'strategy',
            'why are we', 'root cause', 'explain why',
            'mortality', 'died', 'death',
        ]
        for sp in skip_patterns:
            if re.search(sp, q_lower):
                return None
        try:
            qi_result = self.query_intelligence.generate_sql(question)
            sql = qi_result.get('sql', '')
            valid = qi_result.get('valid', False)
            confidence = qi_result.get('confidence', 0)
            if not sql or not valid or confidence < 0.5:
                logger.debug("QI: skipped (sql=%s, valid=%s, conf=%.2f)",
                           bool(sql), valid, confidence)
                return None
            if hasattr(self, '_validate_sql_matches_question'):
                try:
                    if not self._validate_sql_matches_question(question, sql):
                        logger.debug("QI: SQL failed semantic validation, falling through")
                        return None
                except Exception:
                    pass
            rows, cols, err = self._execute_sql(sql)
            if err or rows is None or len(rows) == 0:
                logger.debug("QI: SQL execution failed or empty: %s", err)
                return None
            elapsed = time.time() - t0
            result = {
                'answer': '',
                'sql': sql,
                'rows': rows,
                'results': rows,
                'columns': cols,
                'error': None,
                'row_count': len(rows),
                'confidence': 'A' if confidence >= 0.7 else 'B',
                'confidence_score': confidence,
                'source': 'query_intelligence',
                'approach': f'QI: schema_linking → intent_decomposition → sql_assembly (conf={confidence:.2f})',
                'latency_ms': int(elapsed * 1000),
                'session_id': session_id,
                '_qi_confidence': confidence,
            }
            try:
                _answer = self._generate_narrative(question, sql, rows, cols, result)
                if _answer:
                    result['answer'] = _answer
            except Exception:
                result['answer'] = f"Found {len(rows)} result(s)."
            logger.info("QI SUCCESS: %s → %d rows, conf=%.2f, %dms",
                       question[:60], len(rows), confidence, int(elapsed * 1000))
            return result
        except Exception as e:
            logger.warning("QI error (falling through to templates): %s", e)
            return None

    def _try_golden_template(self, question: str, t0: float, session_id: str) -> Optional[Dict[str, Any]]:
        if os.environ.get('GPDM_DISABLE_GOLDEN_TEMPLATES', '') == '1':
            logger.info("Golden templates DISABLED (GPDM_DISABLE_GOLDEN_TEMPLATES=1)")
            return None
        q_lower = question.lower()
        for template in self._GOLDEN_TEMPLATES:
            for pattern in template['patterns']:
                if re.search(pattern, q_lower):
                    sql = template['sql']

                    if sql is None and '_narrative' in template:
                        narrative_tmpl = template['_narrative']
                        try:
                            _rows, _, _ = self._execute_sql(
                                "SELECT SUM(CASE WHEN DISENROLLMENT_DATE='' THEN 1 ELSE 0 END) AS active, "
                                "SUM(CASE WHEN DISENROLLMENT_DATE!='' THEN 1 ELSE 0 END) AS disenrolled FROM members"
                            )
                            narrative = narrative_tmpl.format(
                                active=f"{_rows[0][0]:,}" if _rows else "unknown",
                                disenrolled=f"{_rows[0][1]:,}" if _rows else "unknown"
                            )
                        except Exception:
                            narrative = narrative_tmpl.replace('{active}', 'many').replace('{disenrolled}', 'some')
                        return {
                            'sql': '',
                            'rows': [],
                            'columns': [],
                            'row_count': 0,
                            'error': None,
                            'source': 'golden_template_data_gap',
                            'narrative': narrative,
                            'confidence': {'grade': 'A', 'overall': 0.95},
                            'confidence_grade': 'A',
                            'confidence_overall': 0.95,
                            'cache_hit': False,
                            'latency_ms': round((time.time() - t0) * 1000),
                            'suggestions': ['Show disenrollment trends', 'Active member count by region'],
                        }

                    if template.get('_dynamic_threshold'):
                        _threshold_match = re.search(r'(?:greater\s*than|more\s*than|over|above|exceed(?:s|ing)?|>)\s*\$?([\d,]+)', q_lower)
                        if _threshold_match:
                            _thresh_val = _threshold_match.group(1).replace(',', '')
                            sql = sql.replace('{threshold}', _thresh_val)
                        else:
                            sql = sql.replace('{threshold}', '10000')

                    if sql and not self._validate_sql_matches_question(question, sql):
                        logger.info("Golden template REJECTED for '%s' — SQL doesn't match intent", question[:50])
                        continue

                    try:
                        rows, columns, error = self._execute_sql(sql)
                        if error or not rows:
                            continue
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
        if not self.analytics_engine:
            return None
        _q_lower = question.lower()
        _analytics_keywords = [
            'risk stratif', 'high risk', 'highest risk', 'risk score', 'risk tier',
            'rising risk', 'risk member', 'risk analyt', 'member risk',
            'risk by region', 'risk by plan', 'risk by age',
            'readmission predict', 'readmission model', 'readmission risk',
            'readmit',
            'cost anomal', 'cost outlier', 'unusual cost', 'suspicious claim',
            'fraud', 'waste', 'anomal',
            'provider scorecard', 'provider performance', 'provider quality',
            'provider grade', 'best provider', 'worst provider', 'low performing',
            'bottom provider', 'top provider', 'worst performing',
            'provider rank', 'provider rating',
            'population segment', 'population health', 'member cohort', 'cluster',
            'member segment', 'population cohort',
            'care gap', 'missed care', 'preventive care gap', 'non-compliance',
            'adherence gap', 'care pattern',
            'analytics overview', 'analytics model', 'what models',
            'what analytics', 'available model',
        ]
        _matched_kw = [kw for kw in _analytics_keywords if kw in _q_lower]
        if not _matched_kw:
            return None
        _simple_agg_words = {'minimum', 'maximum', 'average', 'avg', 'total', 'sum',
                             'count', 'how many', 'highest', 'lowest', 'min ', 'max '}
        if any(w in _q_lower for w in _simple_agg_words):
            return None
        _casual_skip = ['whats the deal', 'what does our', 'what do our', 'show me the',
                        'tell me about', 'what about', 'deal with our',
                        'what is our average', 'what is the average',
                        'what is our mean', 'distribution']
        if any(w in _q_lower for w in _casual_skip):
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
        result = self._process_core(question, session_id)
        sql = result.get('sql', '')
        _source = result.get('source', '')
        _is_brain = _source == 'semantic_intelligence' and self.language_brain is not None
        if sql:
            try:
                sanitized = self._sanitize_sql(sql, question, _brain_generated=_is_brain)
                if sanitized != sql:
                    logger.info("Final sanitizer rewrote SQL for: %s", question[:60])
                    result['sql'] = sanitized
                    result['_sanitized'] = True
                    new_rows, new_cols, new_err = self._execute_sql(sanitized)
                    if not new_err:
                        result['rows'] = new_rows
                        result['results'] = new_rows
                        result['columns'] = new_cols
                        result['error'] = None
                        result['row_count'] = len(new_rows)
            except Exception as e:
                logger.debug("Final sanitizer error (non-fatal): %s", e)

        final_sql = result.get('sql', '') or ''
        q_low = question.lower()
        _needs_reexec = False
        if ('paid' in q_low and ('amount' in q_low or 'total' in q_low or 'top' in q_low)):
            if 'BILLED_AMOUNT' in final_sql.upper() and 'PAID_AMOUNT' not in final_sql.upper():
                final_sql = final_sql.replace('BILLED_AMOUNT', 'PAID_AMOUNT').replace('billed_amount', 'paid_amount')
                _needs_reexec = True
        if ('billed' in q_low and ('amount' in q_low or 'total' in q_low)):
            if 'PAID_AMOUNT' in final_sql.upper() and 'BILLED_AMOUNT' not in final_sql.upper():
                final_sql = final_sql.replace('PAID_AMOUNT', 'BILLED_AMOUNT').replace('paid_amount', 'billed_amount')
                _needs_reexec = True
        if _needs_reexec:
            result['sql'] = final_sql
            result['_sanitized'] = True
            try:
                new_rows, new_cols, new_err = self._execute_sql(final_sql)
                if not new_err:
                    result['rows'] = new_rows
                    result['results'] = new_rows
                    result['columns'] = new_cols
                    result['error'] = None
                    result['row_count'] = len(new_rows)
            except Exception:
                pass

        _final_sql = result.get('sql', '') or ''
        if (_final_sql and self.language_brain and self.intent_parser
                and self.sql_constructor
                and result.get('source') not in ('semantic_intelligence', 'superlative_scalar', 'query_intelligence')):
            try:
                if not self._validate_sql_matches_question(question, _final_sql):
                    logger.info("BRAIN FALLBACK: Final SQL failed validation for '%s', regenerating",
                               question[:60])
                    _fb_intent = self.intent_parser.parse(question)
                    if _fb_intent.confidence >= 0.10:
                        _fb_result = self.sql_constructor.construct(_fb_intent)
                        _fb_sql = _fb_result.get('sql', '')
                        if _fb_sql and self._validate_sql_matches_question(question, _fb_sql):
                            _fb_sql_san = self._sanitize_sql(_fb_sql, question, _brain_generated=True)
                            _fb_rows, _fb_cols, _fb_err = self._execute_sql(_fb_sql_san)
                            if not _fb_err and _fb_rows is not None and len(_fb_rows) > 0:
                                logger.info("BRAIN FALLBACK SUCCESS: %s → %d rows",
                                           question[:60], len(_fb_rows))
                                result['sql'] = _fb_sql_san
                                result['rows'] = _fb_rows
                                result['results'] = _fb_rows
                                result['columns'] = _fb_cols
                                result['error'] = None
                                result['row_count'] = len(_fb_rows)
                                result['source'] = 'semantic_intelligence'
                                result['approach'] = 'schema_graph_reasoning'
                                result['_brain_fallback'] = True
            except Exception as e:
                logger.debug("Brain fallback error (non-fatal): %s", e)

        return result

    def _is_structured_data_query(self, question: str) -> bool:
        q = question.lower()
        _by_dimension = bool(re.search(
            r'\bby\s+(?:race|ethnicity|gender|sex|region|area|location|age|age\s*group|plan\s*type|plan|'
            r'specialty|facility|department|provider|status|severity|category|zip|state|county|city|'
            r'visit\s*type|encounter\s*type|claim\s*type|claim\s*status|service\s*type|service\s*category|'
            r'medication\s*class|drug\s*class|urgency|priority|diagnosis\s*severity)',
            q
        ))
        _model_queries = bool(re.search(
            r'hcc\s+(?:category|code)\s+(?:distribution|breakdown|prevalence)|'
            r'top\s+hcc|hcc\s+prevalence|provider\s+concentration|herfindahl|'
            r'market\s+concentration|provider\s+market\s+share',
            q
        ))
        _comparative_structured = bool(re.search(
            r'which\s+(?:race|ethnicity|gender|region|age\s*group|plan\s*type|plan|specialty|facility)\s+'
            r'(?:has|have|had|shows?|is|are)\s+(?:the\s+)?'
            r'(?:highest|lowest|most|least|best|worst|greatest|fewest|biggest|smallest)',
            q
        ))
        if not _by_dimension and not _model_queries and not _comparative_structured:
            return False
        _structured_concepts = [
            'burden', 'high risk', 'utilization', 'scorecard',
            'cost', 'risk', 'claim', 'condition', 'chronic',
            'member', 'patient', 'encounter', 'visit',
            'denial', 'readmission', 'admission', 'average',
            'total', 'count', 'rate', 'pattern',
            'spend', 'paid', 'billed', 'expense',
            'pharmacy', 'medication', 'prescription', 'drug', 'rx ',
            'referral', 'appointment', 'scheduling', 'no show', 'no-show',
            'enrollment', 'membership', 'disenrollment', 'attrition', 'retention',
            'revenue', 'collection', 'reimbursement', 'write off', 'write-off',
            'telehealth', 'telemedicine', 'virtual visit', 'virtual care',
            'preventive', 'wellness', 'screening', 'annual wellness',
            'care gap', 'gap in care', 'missing care', 'unmet need',
            'losing money', 'financial loss', 'cost leakage', 'money loss',
            'quality', 'hedis', 'quality measure', 'quality initiative',
            'pmpm', 'per member per month', 'cost per member',
            'provider performance', 'provider network', 'provider productivity',
            'clinical outcome', 'disease prevalence', 'diagnosis pattern',
            'population health', 'population risk', 'population profile',
            'er ', 'emergency department', 'emergency room', 'ed visit',
            'care management', 'roi', 'return on investment',
            'severity', 'comorbid', 'multi-morbid', 'multimorbid',
            'cms hcc', 'hcc ', 'hierarchical condition', 'hcc risk', 'hcc category',
            'charlson', 'elixhauser', 'comorbidity index', 'cci ',
            'lace ', 'lace score', 'lace readmission', 'readmission risk',
            'drg', 'diagnosis related group', 'case mix', 'cmi ',
            'actuarial', 'medical loss ratio', 'mlr ',
            'risk stratification', 'risk tier', 'risk band', 'risk segment',
            'herfindahl', 'hhi ', 'market concentration', 'provider concentration',
            'hedis', 'ncqa', 'star rating', 'star measure',
            'cost effective', 'value based', 'cost per outcome',
            'polypharmacy', 'medication burden', 'drug burden',
            'avoidable er', 'unnecessary er', 'non-emergent', 'er diversion',
        ]
        has_concept = any(c in q for c in _structured_concepts)
        _explicit_narrative = any(p in q for p in [
            'deep dive', 'demographic overview', 'demographic analysis',
            'comprehensive demographic', 'full demographic', 'tell me about demographics',
            'demographic breakdown', 'demographic report',
        ])
        if _explicit_narrative:
            return False
        return has_concept

    def _route_structured_query(self, question: str, t0: float, session_id: str) -> Optional[Dict[str, Any]]:
        q_lower = question.lower()

        _concept_sql_map = {
            'chronic_condition_burden': {
                'triggers': [r'chronic\s+condition\s+burden', r'chronic\s+(?:disease|condition)\s+(?:load|count|prevalence)'],
                'sql_template': (
                    "SELECT m.{dim_col} AS dimension, "
                    "COUNT(DISTINCT m.MEMBER_ID) AS total_members, "
                    "COUNT(DISTINCT CASE WHEN d.IS_CHRONIC = 'Y' THEN m.MEMBER_ID END) AS members_with_chronic, "
                    "ROUND(100.0 * COUNT(DISTINCT CASE WHEN d.IS_CHRONIC = 'Y' THEN m.MEMBER_ID END) / "
                    "NULLIF(COUNT(DISTINCT m.MEMBER_ID), 0), 1) AS chronic_pct, "
                    "ROUND(AVG(CASE WHEN d.IS_CHRONIC = 'Y' THEN m.CHRONIC_CONDITIONS ELSE NULL END), 2) AS avg_chronic_conditions "
                    "FROM members m "
                    "LEFT JOIN diagnoses d ON m.MEMBER_ID = d.MEMBER_ID "
                    "GROUP BY m.{dim_col} ORDER BY chronic_pct DESC"
                ),
            },
            'high_risk_members': {
                'triggers': [r'high\s*risk\s+member', r'high\s*risk\s+patient', r'at[\s-]*risk\s+member'],
                'sql_template': (
                    "SELECT m.{dim_col} AS dimension, "
                    "COUNT(*) AS total_members, "
                    "SUM(CASE WHEN CAST(m.RISK_SCORE AS REAL) >= 2.0 THEN 1 ELSE 0 END) AS high_risk_members, "
                    "ROUND(100.0 * SUM(CASE WHEN CAST(m.RISK_SCORE AS REAL) >= 2.0 THEN 1 ELSE 0 END) / "
                    "NULLIF(COUNT(*), 0), 1) AS high_risk_pct, "
                    "ROUND(AVG(CAST(m.RISK_SCORE AS REAL)), 3) AS avg_risk_score "
                    "FROM members m "
                    "GROUP BY m.{dim_col} ORDER BY high_risk_pct DESC"
                ),
            },
            'utilization_patterns': {
                'triggers': [r'utilization\s+pattern', r'utilization\s+by\b', r'healthcare\s+utilization',
                             r'visit\s+pattern', r'encounter\s+pattern'],
                'sql_template': (
                    "SELECT m.{dim_col} AS dimension, "
                    "COUNT(DISTINCT m.MEMBER_ID) AS total_members, "
                    "COUNT(e.ENCOUNTER_ID) AS total_encounters, "
                    "ROUND(1.0 * COUNT(e.ENCOUNTER_ID) / NULLIF(COUNT(DISTINCT m.MEMBER_ID), 0), 2) AS encounters_per_member, "
                    "ROUND(AVG(CAST(e.LENGTH_OF_STAY AS REAL)), 1) AS avg_los, "
                    "COUNT(DISTINCT CASE WHEN UPPER(e.VISIT_TYPE) = 'EMERGENCY' THEN e.ENCOUNTER_ID END) AS er_visits "
                    "FROM members m "
                    "LEFT JOIN encounters e ON m.MEMBER_ID = e.MEMBER_ID "
                    "GROUP BY m.{dim_col} ORDER BY encounters_per_member DESC"
                ),
            },
            'health_equity_scorecard': {
                'triggers': [r'health\s+equity\s+scorecard', r'equity\s+scorecard', r'health\s+equity\s+score'],
                'sql_template': (
                    "SELECT m.RACE AS dimension, "
                    "COUNT(DISTINCT m.MEMBER_ID) AS total_members, "
                    "ROUND(AVG(CAST(m.RISK_SCORE AS REAL)), 3) AS avg_risk_score, "
                    "ROUND(AVG(CAST(COALESCE(c.PAID_AMOUNT, 0) AS REAL)), 2) AS avg_cost, "
                    "ROUND(1.0 * COUNT(e.ENCOUNTER_ID) / NULLIF(COUNT(DISTINCT m.MEMBER_ID), 0), 2) AS encounters_per_member, "
                    "ROUND(AVG(CAST(m.CHRONIC_CONDITIONS AS REAL)), 2) AS avg_chronic_conditions "
                    "FROM members m "
                    "LEFT JOIN claims c ON m.MEMBER_ID = c.MEMBER_ID "
                    "LEFT JOIN encounters e ON m.MEMBER_ID = e.MEMBER_ID "
                    "GROUP BY m.RACE ORDER BY total_members DESC"
                ),
                'fixed_dim': True,
            },
            'cost_risk_by': {
                'triggers': [r'cost\s*(?:&|and)\s*risk\s+by', r'cost\s+(?:and|&)\s+risk\s+(?:by|across)',
                             r'risk\s+(?:and|&)\s+cost\s+by'],
                'sql_template': (
                    "SELECT m.{dim_col} AS dimension, "
                    "COUNT(DISTINCT m.MEMBER_ID) AS member_count, "
                    "ROUND(AVG(CAST(COALESCE(c.PAID_AMOUNT, 0) AS REAL)), 2) AS avg_cost, "
                    "ROUND(AVG(CAST(m.RISK_SCORE AS REAL)), 3) AS avg_risk_score "
                    "FROM members m "
                    "LEFT JOIN claims c ON m.MEMBER_ID = c.MEMBER_ID "
                    "GROUP BY m.{dim_col} ORDER BY avg_cost DESC"
                ),
            },
            'denial_rate': {
                'triggers': [r'denial\s+rate\s+by', r'denied\s+claim.*by', r'claim\s+denial.*by',
                             r'denial\s+(?:pattern|breakdown|analysis)\s+by'],
                'sql_template': (
                    "SELECT {dim_src}.{dim_col} AS dimension, "
                    "COUNT(*) AS total_claims, "
                    "SUM(CASE WHEN c.CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) AS denied_claims, "
                    "ROUND(100.0 * SUM(CASE WHEN c.CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) AS denial_rate_pct, "
                    "ROUND(AVG(CASE WHEN c.CLAIM_STATUS = 'DENIED' THEN CAST(c.BILLED_AMOUNT AS REAL) ELSE NULL END), 2) AS avg_denied_amount "
                    "FROM claims c "
                    "LEFT JOIN members m ON c.MEMBER_ID = m.MEMBER_ID "
                    "GROUP BY {dim_src}.{dim_col} ORDER BY denial_rate_pct DESC"
                ),
                '_claim_dim': True,
            },
            'claims_cost': {
                'triggers': [r'claim\s+cost\s+by', r'claims?\s+(?:spend|spending|expense)\s+by',
                             r'paid\s+amount\s+by', r'billed\s+amount\s+by', r'average\s+cost\s+by',
                             r'total\s+cost\s+by', r'cost\s+(?:breakdown|analysis|distribution)\s+by'],
                'sql_template': (
                    "SELECT {dim_src}.{dim_col} AS dimension, "
                    "COUNT(*) AS total_claims, "
                    "ROUND(SUM(CAST(c.PAID_AMOUNT AS REAL)), 2) AS total_paid, "
                    "ROUND(AVG(CAST(c.PAID_AMOUNT AS REAL)), 2) AS avg_paid, "
                    "ROUND(SUM(CAST(c.BILLED_AMOUNT AS REAL)), 2) AS total_billed, "
                    "ROUND(AVG(CAST(c.MEMBER_RESPONSIBILITY AS REAL)), 2) AS avg_member_responsibility "
                    "FROM claims c "
                    "LEFT JOIN members m ON c.MEMBER_ID = m.MEMBER_ID "
                    "GROUP BY {dim_src}.{dim_col} ORDER BY total_paid DESC"
                ),
                '_claim_dim': True,
            },
            'claims_severity': {
                'triggers': [r'claim\s+severity\s+by', r'severity\s+(?:breakdown|distribution|analysis)\s+by',
                             r'diagnosis\s+severity\s+by'],
                'sql_template': (
                    "SELECT d.SEVERITY AS severity, m.{dim_col} AS dimension, "
                    "COUNT(*) AS diagnosis_count, "
                    "COUNT(DISTINCT d.MEMBER_ID) AS affected_members, "
                    "ROUND(AVG(CAST(COALESCE(c.PAID_AMOUNT, 0) AS REAL)), 2) AS avg_claim_cost "
                    "FROM diagnoses d "
                    "LEFT JOIN members m ON d.MEMBER_ID = m.MEMBER_ID "
                    "LEFT JOIN claims c ON d.MEMBER_ID = c.MEMBER_ID AND d.ENCOUNTER_ID = c.ENCOUNTER_ID "
                    "GROUP BY d.SEVERITY, m.{dim_col} ORDER BY d.SEVERITY, avg_claim_cost DESC"
                ),
            },
            'readmission_rate': {
                'triggers': [r'readmission\s+(?:rate|pattern)\s+by', r'readmit.*by',
                             r'(?:30|60|90)\s*day\s+readmission\s+by'],
                'sql_template': (
                    "SELECT m.{dim_col} AS dimension, "
                    "COUNT(DISTINCT e.ENCOUNTER_ID) AS total_admissions, "
                    "COUNT(DISTINCT CASE WHEN UPPER(e.VISIT_TYPE) IN ('INPATIENT','EMERGENCY') THEN e.ENCOUNTER_ID END) AS acute_admissions, "
                    "ROUND(AVG(CAST(e.LENGTH_OF_STAY AS REAL)), 1) AS avg_los, "
                    "ROUND(AVG(CAST(m.RISK_SCORE AS REAL)), 3) AS avg_risk_score "
                    "FROM members m "
                    "LEFT JOIN encounters e ON m.MEMBER_ID = e.MEMBER_ID "
                    "GROUP BY m.{dim_col} ORDER BY acute_admissions DESC"
                ),
            },
            'pharmacy_spend': {
                'triggers': [r'pharmacy\s+(?:spend|cost|expense)\s+by', r'medication\s+(?:cost|spend)\s+by',
                             r'prescription\s+(?:cost|spend|expense)\s+by', r'drug\s+(?:cost|spend)\s+by',
                             r'rx\s+(?:cost|spend)\s+by'],
                'sql_template': (
                    "SELECT m.{dim_col} AS dimension, "
                    "COUNT(DISTINCT m.MEMBER_ID) AS members_with_rx, "
                    "COUNT(p.RX_ID) AS total_prescriptions, "
                    "ROUND(SUM(CAST(p.COST AS REAL)), 2) AS total_rx_cost, "
                    "ROUND(AVG(CAST(p.COST AS REAL)), 2) AS avg_rx_cost, "
                    "ROUND(AVG(CAST(p.DAYS_SUPPLY AS REAL)), 1) AS avg_days_supply "
                    "FROM members m "
                    "LEFT JOIN prescriptions p ON m.MEMBER_ID = p.MEMBER_ID "
                    "GROUP BY m.{dim_col} ORDER BY total_rx_cost DESC"
                ),
            },
            'medication_class': {
                'triggers': [r'medication\s+class\s+by', r'drug\s+class\s+by',
                             r'medication\s+(?:type|category)\s+by', r'top\s+medication.*by',
                             r'prescription\s+(?:breakdown|pattern|distribution)\s+by'],
                'sql_template': (
                    "SELECT p.MEDICATION_CLASS AS medication_class, m.{dim_col} AS dimension, "
                    "COUNT(*) AS prescription_count, "
                    "COUNT(DISTINCT p.MEMBER_ID) AS unique_patients, "
                    "ROUND(SUM(CAST(p.COST AS REAL)), 2) AS total_cost, "
                    "ROUND(AVG(CAST(p.COST AS REAL)), 2) AS avg_cost_per_rx "
                    "FROM prescriptions p "
                    "LEFT JOIN members m ON p.MEMBER_ID = m.MEMBER_ID "
                    "GROUP BY p.MEDICATION_CLASS, m.{dim_col} ORDER BY total_cost DESC"
                ),
            },
            'provider_performance': {
                'triggers': [r'provider\s+(?:performance|productivity|volume)\s+by',
                             r'provider\s+network\s+by', r'physician\s+(?:performance|productivity)\s+by',
                             r'doctor\s+(?:performance|volume)\s+by'],
                'sql_template': (
                    "SELECT p.SPECIALTY AS specialty, p.KP_REGION AS region, "
                    "COUNT(DISTINCT p.NPI) AS provider_count, "
                    "ROUND(AVG(CAST(p.PANEL_SIZE AS REAL)), 0) AS avg_panel_size, "
                    "SUM(CASE WHEN p.ACCEPTS_NEW_PATIENTS = 'Y' THEN 1 ELSE 0 END) AS accepting_new, "
                    "SUM(CASE WHEN p.STATUS = 'ACTIVE' THEN 1 ELSE 0 END) AS active_providers "
                    "FROM providers p "
                    "GROUP BY p.SPECIALTY, p.KP_REGION ORDER BY provider_count DESC"
                ),
                'fixed_dim': True,
            },
            'referral_pattern': {
                'triggers': [r'referral\s+(?:pattern|volume|breakdown|analysis)\s+by',
                             r'referral\s+rate\s+by', r'referral\s+(?:type|reason)\s+by'],
                'sql_template': (
                    "SELECT m.{dim_col} AS dimension, "
                    "COUNT(r.REFERRAL_ID) AS total_referrals, "
                    "COUNT(DISTINCT r.MEMBER_ID) AS members_referred, "
                    "SUM(CASE WHEN r.URGENCY = 'URGENT' OR r.URGENCY = 'STAT' THEN 1 ELSE 0 END) AS urgent_referrals, "
                    "SUM(CASE WHEN r.REFERRAL_TYPE = 'EXTERNAL' THEN 1 ELSE 0 END) AS external_referrals, "
                    "ROUND(100.0 * SUM(CASE WHEN r.STATUS = 'COMPLETED' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) AS completion_rate_pct "
                    "FROM members m "
                    "LEFT JOIN referrals r ON m.MEMBER_ID = r.MEMBER_ID "
                    "GROUP BY m.{dim_col} ORDER BY total_referrals DESC"
                ),
            },
            'appointment_access': {
                'triggers': [r'appointment\s+(?:access|volume|pattern|breakdown)\s+by',
                             r'appointment\s+(?:no.?show|cancellation|status)\s+by',
                             r'scheduling\s+(?:pattern|access)\s+by', r'access\s+to\s+care\s+by'],
                'sql_template': (
                    "SELECT m.{dim_col} AS dimension, "
                    "COUNT(a.APPOINTMENT_ID) AS total_appointments, "
                    "SUM(CASE WHEN a.STATUS = 'COMPLETED' THEN 1 ELSE 0 END) AS completed, "
                    "SUM(CASE WHEN a.STATUS = 'NO_SHOW' THEN 1 ELSE 0 END) AS no_shows, "
                    "SUM(CASE WHEN a.STATUS = 'CANCELLED' THEN 1 ELSE 0 END) AS cancelled, "
                    "ROUND(100.0 * SUM(CASE WHEN a.STATUS = 'NO_SHOW' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) AS no_show_rate_pct "
                    "FROM members m "
                    "LEFT JOIN appointments a ON m.MEMBER_ID = a.MEMBER_ID "
                    "GROUP BY m.{dim_col} ORDER BY no_show_rate_pct DESC"
                ),
            },
            'enrollment_membership': {
                'triggers': [r'enrollment\s+(?:pattern|breakdown|status)\s+by', r'membership\s+(?:by|breakdown|distribution)',
                             r'member\s+(?:enrollment|disenrollment|attrition|retention)\s+by',
                             r'plan\s+(?:enrollment|membership)\s+by'],
                'sql_template': (
                    "SELECT m.{dim_col} AS dimension, "
                    "COUNT(*) AS total_members, "
                    "SUM(CASE WHEN m.DISENROLLMENT_DATE = '' OR m.DISENROLLMENT_DATE IS NULL THEN 1 ELSE 0 END) AS active_members, "
                    "SUM(CASE WHEN m.DISENROLLMENT_DATE != '' AND m.DISENROLLMENT_DATE IS NOT NULL THEN 1 ELSE 0 END) AS disenrolled, "
                    "ROUND(100.0 * SUM(CASE WHEN m.DISENROLLMENT_DATE != '' AND m.DISENROLLMENT_DATE IS NOT NULL THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) AS disenrollment_rate_pct, "
                    "ROUND(AVG(CAST(m.RISK_SCORE AS REAL)), 3) AS avg_risk_score "
                    "FROM members m "
                    "GROUP BY m.{dim_col} ORDER BY total_members DESC"
                ),
            },
            'revenue_cycle': {
                'triggers': [r'revenue\s+cycle\s+by', r'revenue\s+(?:analysis|breakdown)\s+by',
                             r'(?:collection|reimbursement)\s+rate\s+by', r'financial\s+(?:performance|analysis)\s+by'],
                'sql_template': (
                    "SELECT {dim_src}.{dim_col} AS dimension, "
                    "COUNT(*) AS total_claims, "
                    "ROUND(SUM(CAST(c.BILLED_AMOUNT AS REAL)), 2) AS total_billed, "
                    "ROUND(SUM(CAST(c.PAID_AMOUNT AS REAL)), 2) AS total_paid, "
                    "ROUND(SUM(CAST(c.ALLOWED_AMOUNT AS REAL)), 2) AS total_allowed, "
                    "ROUND(100.0 * SUM(CAST(c.PAID_AMOUNT AS REAL)) / NULLIF(SUM(CAST(c.BILLED_AMOUNT AS REAL)), 0), 1) AS collection_rate_pct, "
                    "ROUND(SUM(CAST(c.BILLED_AMOUNT AS REAL)) - SUM(CAST(c.PAID_AMOUNT AS REAL)), 2) AS write_off "
                    "FROM claims c "
                    "LEFT JOIN members m ON c.MEMBER_ID = m.MEMBER_ID "
                    "GROUP BY {dim_src}.{dim_col} ORDER BY total_billed DESC"
                ),
                '_claim_dim': True,
            },
            'population_health': {
                'triggers': [r'population\s+health\s+by', r'population\s+(?:risk|profile|overview)\s+by',
                             r'member\s+health\s+(?:status|profile)\s+by'],
                'sql_template': (
                    "SELECT m.{dim_col} AS dimension, "
                    "COUNT(*) AS total_members, "
                    "ROUND(AVG(CAST(m.RISK_SCORE AS REAL)), 3) AS avg_risk_score, "
                    "ROUND(AVG(CAST(m.CHRONIC_CONDITIONS AS REAL)), 2) AS avg_chronic_conditions, "
                    "SUM(CASE WHEN CAST(m.CHRONIC_CONDITIONS AS INTEGER) >= 3 THEN 1 ELSE 0 END) AS multi_morbid, "
                    "ROUND(100.0 * SUM(CASE WHEN CAST(m.CHRONIC_CONDITIONS AS INTEGER) >= 3 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) AS multi_morbid_pct "
                    "FROM members m "
                    "GROUP BY m.{dim_col} ORDER BY avg_risk_score DESC"
                ),
            },
            'clinical_outcomes': {
                'triggers': [r'clinical\s+outcome.*by', r'diagnosis\s+(?:pattern|distribution|breakdown)\s+by',
                             r'disease\s+(?:prevalence|pattern|distribution)\s+by',
                             r'condition\s+(?:pattern|distribution|breakdown)\s+by'],
                'sql_template': (
                    "SELECT m.{dim_col} AS dimension, "
                    "COUNT(DISTINCT d.DIAGNOSIS_ID) AS total_diagnoses, "
                    "COUNT(DISTINCT d.MEMBER_ID) AS affected_members, "
                    "SUM(CASE WHEN d.IS_CHRONIC = 'Y' THEN 1 ELSE 0 END) AS chronic_diagnoses, "
                    "SUM(CASE WHEN d.SEVERITY IN ('SEVERE','CRITICAL') THEN 1 ELSE 0 END) AS severe_critical, "
                    "ROUND(100.0 * SUM(CASE WHEN d.SEVERITY IN ('SEVERE','CRITICAL') THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) AS severe_pct "
                    "FROM diagnoses d "
                    "LEFT JOIN members m ON d.MEMBER_ID = m.MEMBER_ID "
                    "GROUP BY m.{dim_col} ORDER BY total_diagnoses DESC"
                ),
            },
            'er_utilization': {
                'triggers': [r'(?:er|emergency|ed)\s+(?:utilization|visit|volume|rate)\s+by',
                             r'emergency\s+(?:department|room)\s+by'],
                'sql_template': (
                    "SELECT m.{dim_col} AS dimension, "
                    "COUNT(DISTINCT m.MEMBER_ID) AS total_members, "
                    "COUNT(DISTINCT CASE WHEN UPPER(e.VISIT_TYPE) = 'EMERGENCY' THEN e.ENCOUNTER_ID END) AS er_visits, "
                    "ROUND(1000.0 * COUNT(DISTINCT CASE WHEN UPPER(e.VISIT_TYPE) = 'EMERGENCY' THEN e.ENCOUNTER_ID END) / NULLIF(COUNT(DISTINCT m.MEMBER_ID), 0), 1) AS er_per_1000, "
                    "COUNT(DISTINCT CASE WHEN UPPER(e.VISIT_TYPE) = 'INPATIENT' THEN e.ENCOUNTER_ID END) AS inpatient_admits, "
                    "ROUND(1000.0 * COUNT(DISTINCT CASE WHEN UPPER(e.VISIT_TYPE) = 'INPATIENT' THEN e.ENCOUNTER_ID END) / NULLIF(COUNT(DISTINCT m.MEMBER_ID), 0), 1) AS admits_per_1000 "
                    "FROM members m "
                    "LEFT JOIN encounters e ON m.MEMBER_ID = e.MEMBER_ID "
                    "GROUP BY m.{dim_col} ORDER BY er_per_1000 DESC"
                ),
            },
            'pmpm_cost': {
                'triggers': [r'pmpm\s+(?:cost|trend|analysis|spend)\s+by', r'per\s+member\s+per\s+month\s+by',
                             r'pmpm\s+by', r'cost\s+per\s+member\s+by'],
                'sql_template': (
                    "SELECT m.{dim_col} AS dimension, "
                    "COUNT(DISTINCT m.MEMBER_ID) AS total_members, "
                    "ROUND(SUM(CAST(c.PAID_AMOUNT AS REAL)), 2) AS total_paid, "
                    "ROUND(SUM(CAST(c.PAID_AMOUNT AS REAL)) / NULLIF(COUNT(DISTINCT m.MEMBER_ID), 0), 2) AS total_cost_per_member, "
                    "ROUND(SUM(CAST(c.PAID_AMOUNT AS REAL)) / NULLIF(COUNT(DISTINCT m.MEMBER_ID), 0) / 12, 2) AS pmpm, "
                    "ROUND(AVG(CAST(m.RISK_SCORE AS REAL)), 3) AS avg_risk "
                    "FROM members m "
                    "LEFT JOIN claims c ON m.MEMBER_ID = c.MEMBER_ID "
                    "GROUP BY m.{dim_col} ORDER BY pmpm DESC"
                ),
            },
            'telehealth_usage': {
                'triggers': [r'telehealth\s+(?:usage|utilization|adoption|pattern)\s+by',
                             r'virtual\s+(?:visit|care)\s+by', r'telemedicine\s+by'],
                'sql_template': (
                    "SELECT m.{dim_col} AS dimension, "
                    "COUNT(DISTINCT m.MEMBER_ID) AS total_members, "
                    "COUNT(DISTINCT CASE WHEN UPPER(e.VISIT_TYPE) = 'TELEHEALTH' THEN e.ENCOUNTER_ID END) AS telehealth_visits, "
                    "ROUND(100.0 * COUNT(DISTINCT CASE WHEN UPPER(e.VISIT_TYPE) = 'TELEHEALTH' THEN e.ENCOUNTER_ID END) / "
                    "NULLIF(COUNT(e.ENCOUNTER_ID), 0), 1) AS telehealth_pct, "
                    "COUNT(e.ENCOUNTER_ID) AS total_encounters "
                    "FROM members m "
                    "LEFT JOIN encounters e ON m.MEMBER_ID = e.MEMBER_ID "
                    "GROUP BY m.{dim_col} ORDER BY telehealth_pct DESC"
                ),
            },
            'preventive_care': {
                'triggers': [r'preventive\s+care\s+by', r'wellness\s+(?:visit|check)\s+by',
                             r'screening\s+(?:rate|pattern)\s+by', r'annual\s+wellness\s+by',
                             r'preventive\s+(?:screening|service)\s+by'],
                'sql_template': (
                    "SELECT m.{dim_col} AS dimension, "
                    "COUNT(DISTINCT m.MEMBER_ID) AS total_members, "
                    "COUNT(DISTINCT CASE WHEN a.APPOINTMENT_TYPE = 'ANNUAL_WELLNESS' THEN a.APPOINTMENT_ID END) AS wellness_visits, "
                    "ROUND(100.0 * COUNT(DISTINCT CASE WHEN a.APPOINTMENT_TYPE = 'ANNUAL_WELLNESS' THEN a.MEMBER_ID END) / "
                    "NULLIF(COUNT(DISTINCT m.MEMBER_ID), 0), 1) AS wellness_visit_pct, "
                    "COUNT(DISTINCT CASE WHEN a.APPOINTMENT_TYPE = 'LAB_WORK' THEN a.APPOINTMENT_ID END) AS lab_visits "
                    "FROM members m "
                    "LEFT JOIN appointments a ON m.MEMBER_ID = a.MEMBER_ID "
                    "GROUP BY m.{dim_col} ORDER BY wellness_visit_pct DESC"
                ),
            },
            'care_gap': {
                'triggers': [r'care\s+gap\s+by', r'gap\s+in\s+care\s+by', r'missing\s+care\s+by',
                             r'unmet\s+need\s+by'],
                'sql_template': (
                    "SELECT m.{dim_col} AS dimension, "
                    "COUNT(DISTINCT m.MEMBER_ID) AS total_members, "
                    "COUNT(DISTINCT CASE WHEN CAST(m.CHRONIC_CONDITIONS AS INTEGER) > 0 "
                    "AND NOT EXISTS (SELECT 1 FROM encounters e2 WHERE e2.MEMBER_ID = m.MEMBER_ID) "
                    "THEN m.MEMBER_ID END) AS chronic_no_encounter, "
                    "COUNT(DISTINCT CASE WHEN CAST(m.RISK_SCORE AS REAL) >= 2.0 "
                    "AND NOT EXISTS (SELECT 1 FROM appointments a2 WHERE a2.MEMBER_ID = m.MEMBER_ID AND a2.APPOINTMENT_TYPE = 'PCP_VISIT') "
                    "THEN m.MEMBER_ID END) AS high_risk_no_pcp, "
                    "ROUND(AVG(CAST(m.CHRONIC_CONDITIONS AS REAL)), 2) AS avg_chronic "
                    "FROM members m "
                    "GROUP BY m.{dim_col} ORDER BY chronic_no_encounter DESC"
                ),
            },
            'losing_money': {
                'triggers': [r'losing\s+money\s+by', r'where\s+(?:are\s+we|is\s+the)\s+losing',
                             r'financial\s+loss\s+by', r'cost\s+leakage\s+by', r'money\s+loss\s+by'],
                'sql_template': (
                    "SELECT m.{dim_col} AS dimension, "
                    "COUNT(*) AS total_claims, "
                    "ROUND(SUM(CAST(c.BILLED_AMOUNT AS REAL)) - SUM(CAST(c.PAID_AMOUNT AS REAL)), 2) AS revenue_gap, "
                    "ROUND(100.0 * (SUM(CAST(c.BILLED_AMOUNT AS REAL)) - SUM(CAST(c.PAID_AMOUNT AS REAL))) / "
                    "NULLIF(SUM(CAST(c.BILLED_AMOUNT AS REAL)), 0), 1) AS gap_pct, "
                    "SUM(CASE WHEN c.CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) AS denied_claims, "
                    "ROUND(SUM(CASE WHEN c.CLAIM_STATUS = 'DENIED' THEN CAST(c.BILLED_AMOUNT AS REAL) ELSE 0 END), 2) AS denied_amount "
                    "FROM claims c "
                    "LEFT JOIN members m ON c.MEMBER_ID = m.MEMBER_ID "
                    "GROUP BY m.{dim_col} ORDER BY revenue_gap DESC"
                ),
            },
            'quality_initiative': {
                'triggers': [r'quality\s+(?:initiative|measure|metric|score)\s+by',
                             r'hedis\s+(?:measure|score|performance)\s+by',
                             r'quality\s+(?:performance|improvement)\s+by'],
                'sql_template': (
                    "SELECT m.{dim_col} AS dimension, "
                    "COUNT(DISTINCT m.MEMBER_ID) AS total_members, "
                    "ROUND(100.0 * COUNT(DISTINCT CASE WHEN a.APPOINTMENT_TYPE = 'ANNUAL_WELLNESS' THEN a.MEMBER_ID END) / "
                    "NULLIF(COUNT(DISTINCT m.MEMBER_ID), 0), 1) AS wellness_rate, "
                    "ROUND(100.0 * SUM(CASE WHEN c.CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) / "
                    "NULLIF(COUNT(c.CLAIM_ID), 0), 1) AS denial_rate, "
                    "ROUND(AVG(CAST(m.CHRONIC_CONDITIONS AS REAL)), 2) AS avg_chronic, "
                    "ROUND(AVG(CAST(m.RISK_SCORE AS REAL)), 3) AS avg_risk "
                    "FROM members m "
                    "LEFT JOIN appointments a ON m.MEMBER_ID = a.MEMBER_ID "
                    "LEFT JOIN claims c ON m.MEMBER_ID = c.MEMBER_ID "
                    "GROUP BY m.{dim_col} ORDER BY wellness_rate DESC"
                ),
            },
            'care_management_roi': {
                'triggers': [r'care\s+management\s+(?:roi|return|value)\s+by',
                             r'roi\s+(?:of|for)\s+care\s+management',
                             r'care\s+management.*high[\s-]*risk'],
                'sql_template': (
                    "SELECT m.{dim_col} AS dimension, "
                    "COUNT(DISTINCT m.MEMBER_ID) AS total_members, "
                    "SUM(CASE WHEN CAST(m.RISK_SCORE AS REAL) >= 2.0 THEN 1 ELSE 0 END) AS high_risk_count, "
                    "ROUND(AVG(CASE WHEN CAST(m.RISK_SCORE AS REAL) >= 2.0 THEN CAST(COALESCE(c.PAID_AMOUNT, 0) AS REAL) END), 2) AS high_risk_avg_cost, "
                    "ROUND(AVG(CASE WHEN CAST(m.RISK_SCORE AS REAL) < 2.0 THEN CAST(COALESCE(c.PAID_AMOUNT, 0) AS REAL) END), 2) AS low_risk_avg_cost, "
                    "ROUND(AVG(CASE WHEN CAST(m.RISK_SCORE AS REAL) >= 2.0 THEN CAST(COALESCE(c.PAID_AMOUNT, 0) AS REAL) END) - "
                    "AVG(CASE WHEN CAST(m.RISK_SCORE AS REAL) < 2.0 THEN CAST(COALESCE(c.PAID_AMOUNT, 0) AS REAL) END), 2) AS cost_differential "
                    "FROM members m "
                    "LEFT JOIN claims c ON m.MEMBER_ID = c.MEMBER_ID "
                    "GROUP BY m.{dim_col} ORDER BY cost_differential DESC"
                ),
            },
            'cms_hcc_risk': {
                'triggers': [r'cms[\s-]*hcc\s+(?:risk|score|model|category|analysis)', r'hcc\s+(?:risk|score|category|model)\s+by',
                             r'hierarchical\s+condition\s+category', r'hcc\s+by\b', r'cms[\s-]*hcc\s+by'],
                'sql_template': (
                    "SELECT m.{dim_col} AS dimension, "
                    "COUNT(DISTINCT m.MEMBER_ID) AS total_members, "
                    "ROUND(AVG(CAST(m.RISK_SCORE AS REAL)), 3) AS avg_hcc_risk_score, "
                    "COUNT(DISTINCT d.HCC_CODE) AS unique_hcc_codes, "
                    "COUNT(DISTINCT d.HCC_CATEGORY) AS unique_hcc_categories, "
                    "ROUND(AVG(CAST(m.CHRONIC_CONDITIONS AS REAL)), 2) AS avg_chronic_conditions, "
                    "SUM(CASE WHEN CAST(m.RISK_SCORE AS REAL) >= 3.0 THEN 1 ELSE 0 END) AS very_high_risk, "
                    "SUM(CASE WHEN CAST(m.RISK_SCORE AS REAL) BETWEEN 2.0 AND 2.99 THEN 1 ELSE 0 END) AS high_risk, "
                    "SUM(CASE WHEN CAST(m.RISK_SCORE AS REAL) BETWEEN 1.0 AND 1.99 THEN 1 ELSE 0 END) AS moderate_risk, "
                    "SUM(CASE WHEN CAST(m.RISK_SCORE AS REAL) < 1.0 THEN 1 ELSE 0 END) AS low_risk "
                    "FROM members m "
                    "LEFT JOIN diagnoses d ON m.MEMBER_ID = d.MEMBER_ID "
                    "GROUP BY m.{dim_col} ORDER BY avg_hcc_risk_score DESC"
                ),
            },
            'hcc_category_distribution': {
                'triggers': [r'hcc\s+(?:category|code)\s+(?:distribution|breakdown|analysis)',
                             r'top\s+hcc\s+(?:categories|codes)', r'hcc\s+prevalence'],
                'sql_template': (
                    "SELECT d.HCC_CATEGORY AS hcc_category, d.HCC_CODE AS hcc_code, "
                    "COUNT(DISTINCT d.MEMBER_ID) AS affected_members, "
                    "COUNT(*) AS diagnosis_count, "
                    "ROUND(AVG(CAST(m.RISK_SCORE AS REAL)), 3) AS avg_risk_score, "
                    "ROUND(AVG(CAST(COALESCE(c.PAID_AMOUNT, 0) AS REAL)), 2) AS avg_cost "
                    "FROM diagnoses d "
                    "LEFT JOIN members m ON d.MEMBER_ID = m.MEMBER_ID "
                    "LEFT JOIN claims c ON d.MEMBER_ID = c.MEMBER_ID AND d.ENCOUNTER_ID = c.ENCOUNTER_ID "
                    "WHERE d.HCC_CODE IS NOT NULL AND d.HCC_CODE != '' "
                    "GROUP BY d.HCC_CATEGORY, d.HCC_CODE ORDER BY affected_members DESC LIMIT 25"
                ),
                'fixed_dim': True,
            },
            'charlson_comorbidity': {
                'triggers': [r'charlson\s+(?:comorbidity|index|score)', r'comorbidity\s+(?:index|score|burden)\s+by',
                             r'cci\s+(?:score|by|analysis)', r'elixhauser\s+(?:comorbidity|index|score)'],
                'sql_template': (
                    "SELECT m.{dim_col} AS dimension, "
                    "COUNT(DISTINCT m.MEMBER_ID) AS total_members, "
                    "ROUND(AVG(CAST(m.CHRONIC_CONDITIONS AS REAL)), 2) AS avg_comorbidity_index, "
                    "SUM(CASE WHEN CAST(m.CHRONIC_CONDITIONS AS INTEGER) = 0 THEN 1 ELSE 0 END) AS zero_comorbidity, "
                    "SUM(CASE WHEN CAST(m.CHRONIC_CONDITIONS AS INTEGER) BETWEEN 1 AND 2 THEN 1 ELSE 0 END) AS mild_1_2, "
                    "SUM(CASE WHEN CAST(m.CHRONIC_CONDITIONS AS INTEGER) BETWEEN 3 AND 4 THEN 1 ELSE 0 END) AS moderate_3_4, "
                    "SUM(CASE WHEN CAST(m.CHRONIC_CONDITIONS AS INTEGER) >= 5 THEN 1 ELSE 0 END) AS severe_5_plus, "
                    "ROUND(100.0 * SUM(CASE WHEN CAST(m.CHRONIC_CONDITIONS AS INTEGER) >= 5 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) AS severe_pct, "
                    "ROUND(AVG(CAST(m.RISK_SCORE AS REAL)), 3) AS avg_risk_score "
                    "FROM members m "
                    "GROUP BY m.{dim_col} ORDER BY avg_comorbidity_index DESC"
                ),
            },
            'lace_readmission': {
                'triggers': [r'lace\s+(?:score|index|readmission|model)', r'readmission\s+(?:risk|prediction|lace)',
                             r'lace\s+by\b'],
                'sql_template': (
                    "SELECT m.{dim_col} AS dimension, "
                    "COUNT(DISTINCT e.ENCOUNTER_ID) AS total_encounters, "
                    "ROUND(AVG(CAST(e.LENGTH_OF_STAY AS REAL)), 1) AS avg_length_of_stay, "
                    "COUNT(DISTINCT CASE WHEN UPPER(e.VISIT_TYPE) = 'EMERGENCY' THEN e.ENCOUNTER_ID END) AS er_arrivals, "
                    "ROUND(AVG(CAST(m.CHRONIC_CONDITIONS AS REAL)), 2) AS avg_comorbidities, "
                    "ROUND(AVG(CAST(m.RISK_SCORE AS REAL)), 3) AS avg_acuity_risk, "
                    "ROUND(("
                    "  CASE WHEN AVG(CAST(e.LENGTH_OF_STAY AS REAL)) < 1 THEN 0 "
                    "       WHEN AVG(CAST(e.LENGTH_OF_STAY AS REAL)) BETWEEN 1 AND 3 THEN 2 "
                    "       WHEN AVG(CAST(e.LENGTH_OF_STAY AS REAL)) BETWEEN 4 AND 6 THEN 4 "
                    "       ELSE 6 END + "
                    "  CASE WHEN AVG(CAST(m.CHRONIC_CONDITIONS AS REAL)) < 2 THEN 1 "
                    "       WHEN AVG(CAST(m.CHRONIC_CONDITIONS AS REAL)) BETWEEN 2 AND 4 THEN 3 "
                    "       ELSE 5 END + "
                    "  CASE WHEN AVG(CAST(m.RISK_SCORE AS REAL)) < 1.0 THEN 0 "
                    "       WHEN AVG(CAST(m.RISK_SCORE AS REAL)) BETWEEN 1.0 AND 2.0 THEN 2 "
                    "       ELSE 4 END"
                    "), 1) AS estimated_lace_score "
                    "FROM members m "
                    "LEFT JOIN encounters e ON m.MEMBER_ID = e.MEMBER_ID "
                    "GROUP BY m.{dim_col} ORDER BY estimated_lace_score DESC"
                ),
            },
            'drg_analysis': {
                'triggers': [r'drg\s+(?:analysis|weight|distribution|mix|index|by)', r'diagnosis\s+related\s+group',
                             r'case\s+mix\s+(?:index|by|analysis)', r'cmi\s+(?:by|analysis)'],
                'sql_template': (
                    "SELECT m.{dim_col} AS dimension, "
                    "COUNT(DISTINCT e.ENCOUNTER_ID) AS total_admissions, "
                    "COUNT(DISTINCT CASE WHEN UPPER(e.VISIT_TYPE) = 'INPATIENT' THEN e.ENCOUNTER_ID END) AS inpatient_cases, "
                    "ROUND(AVG(CAST(e.LENGTH_OF_STAY AS REAL)), 1) AS avg_length_of_stay, "
                    "ROUND(AVG(CAST(COALESCE(c.PAID_AMOUNT, 0) AS REAL)), 2) AS avg_cost_per_case, "
                    "ROUND(SUM(CAST(COALESCE(c.PAID_AMOUNT, 0) AS REAL)), 2) AS total_cost, "
                    "ROUND(AVG(CAST(m.RISK_SCORE AS REAL)), 3) AS avg_risk_case_mix, "
                    "COUNT(DISTINCT d.ICD10_CODE) AS unique_diagnosis_codes "
                    "FROM members m "
                    "LEFT JOIN encounters e ON m.MEMBER_ID = e.MEMBER_ID "
                    "LEFT JOIN claims c ON m.MEMBER_ID = c.MEMBER_ID AND e.ENCOUNTER_ID = c.ENCOUNTER_ID "
                    "LEFT JOIN diagnoses d ON m.MEMBER_ID = d.MEMBER_ID AND e.ENCOUNTER_ID = d.ENCOUNTER_ID "
                    "GROUP BY m.{dim_col} ORDER BY avg_cost_per_case DESC"
                ),
            },
            'actuarial_pmpm': {
                'triggers': [r'actuarial\s+(?:pmpm|analysis|cost|model)', r'actuarial\s+by',
                             r'medical\s+loss\s+ratio', r'mlr\s+(?:by|analysis)',
                             r'pmpm\s+(?:actuarial|trend|projection)'],
                'sql_template': (
                    "SELECT m.{dim_col} AS dimension, "
                    "COUNT(DISTINCT m.MEMBER_ID) AS total_members, "
                    "ROUND(SUM(CAST(c.PAID_AMOUNT AS REAL)), 2) AS total_medical_cost, "
                    "ROUND(SUM(CAST(c.PAID_AMOUNT AS REAL)) / NULLIF(COUNT(DISTINCT m.MEMBER_ID), 0), 2) AS total_cost_per_member, "
                    "ROUND(SUM(CAST(c.PAID_AMOUNT AS REAL)) / NULLIF(COUNT(DISTINCT m.MEMBER_ID), 0) / 12, 2) AS medical_pmpm, "
                    "ROUND(SUM(CAST(c.BILLED_AMOUNT AS REAL)), 2) AS total_billed, "
                    "ROUND(100.0 * SUM(CAST(c.PAID_AMOUNT AS REAL)) / NULLIF(SUM(CAST(c.BILLED_AMOUNT AS REAL)), 0), 1) AS medical_loss_ratio_pct, "
                    "ROUND(AVG(CAST(m.RISK_SCORE AS REAL)), 3) AS avg_risk_adjustment_factor "
                    "FROM members m "
                    "LEFT JOIN claims c ON m.MEMBER_ID = c.MEMBER_ID "
                    "GROUP BY m.{dim_col} ORDER BY medical_pmpm DESC"
                ),
            },
            'risk_stratification': {
                'triggers': [r'risk\s+stratification\s+by', r'risk\s+(?:tier|band|bucket|segment)\s+by',
                             r'risk\s+distribution\s+by', r'risk\s+(?:profile|segmentation)\s+by'],
                'sql_template': (
                    "SELECT m.{dim_col} AS dimension, "
                    "COUNT(*) AS total_members, "
                    "SUM(CASE WHEN CAST(m.RISK_SCORE AS REAL) < 0.5 THEN 1 ELSE 0 END) AS very_low_risk, "
                    "SUM(CASE WHEN CAST(m.RISK_SCORE AS REAL) BETWEEN 0.5 AND 0.99 THEN 1 ELSE 0 END) AS low_risk, "
                    "SUM(CASE WHEN CAST(m.RISK_SCORE AS REAL) BETWEEN 1.0 AND 1.49 THEN 1 ELSE 0 END) AS moderate_risk, "
                    "SUM(CASE WHEN CAST(m.RISK_SCORE AS REAL) BETWEEN 1.5 AND 1.99 THEN 1 ELSE 0 END) AS elevated_risk, "
                    "SUM(CASE WHEN CAST(m.RISK_SCORE AS REAL) BETWEEN 2.0 AND 2.99 THEN 1 ELSE 0 END) AS high_risk, "
                    "SUM(CASE WHEN CAST(m.RISK_SCORE AS REAL) >= 3.0 THEN 1 ELSE 0 END) AS very_high_risk, "
                    "ROUND(AVG(CAST(m.RISK_SCORE AS REAL)), 3) AS avg_risk, "
                    "ROUND(AVG(CAST(COALESCE(c.PAID_AMOUNT, 0) AS REAL)), 2) AS avg_cost "
                    "FROM members m "
                    "LEFT JOIN claims c ON m.MEMBER_ID = c.MEMBER_ID "
                    "GROUP BY m.{dim_col} ORDER BY avg_risk DESC"
                ),
            },
            'provider_concentration': {
                'triggers': [r'(?:provider|market)\s+concentration', r'herfindahl',
                             r'hhi\s+(?:by|index|analysis)', r'provider\s+(?:market\s+share|dominance)'],
                'sql_template': (
                    "SELECT p.SPECIALTY AS specialty, p.KP_REGION AS region, "
                    "COUNT(DISTINCT p.NPI) AS provider_count, "
                    "ROUND(AVG(CAST(p.PANEL_SIZE AS REAL)), 0) AS avg_panel, "
                    "MAX(CAST(p.PANEL_SIZE AS REAL)) AS max_panel, "
                    "MIN(CAST(p.PANEL_SIZE AS REAL)) AS min_panel, "
                    "ROUND(100.0 * MAX(CAST(p.PANEL_SIZE AS REAL)) / NULLIF(SUM(CAST(p.PANEL_SIZE AS REAL)), 0), 1) AS top_provider_share_pct "
                    "FROM providers p "
                    "WHERE p.STATUS = 'ACTIVE' "
                    "GROUP BY p.SPECIALTY, p.KP_REGION ORDER BY provider_count DESC"
                ),
                'fixed_dim': True,
            },
            'hedis_quality': {
                'triggers': [r'hedis\s+(?:quality|measure|metric|score|compliance|rate)', r'hedis\s+by\b',
                             r'ncqa\s+(?:quality|measure|metric)', r'star\s+(?:rating|measure)\s+by'],
                'sql_template': (
                    "SELECT m.{dim_col} AS dimension, "
                    "COUNT(DISTINCT m.MEMBER_ID) AS total_members, "
                    "ROUND(100.0 * COUNT(DISTINCT CASE WHEN a.APPOINTMENT_TYPE = 'ANNUAL_WELLNESS' THEN a.MEMBER_ID END) / "
                    "NULLIF(COUNT(DISTINCT m.MEMBER_ID), 0), 1) AS annual_wellness_visit_rate, "
                    "ROUND(100.0 * COUNT(DISTINCT CASE WHEN a.APPOINTMENT_TYPE IN ('LAB_WORK','SCREENING') THEN a.MEMBER_ID END) / "
                    "NULLIF(COUNT(DISTINCT m.MEMBER_ID), 0), 1) AS preventive_screening_rate, "
                    "ROUND(100.0 * SUM(CASE WHEN c.CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) / "
                    "NULLIF(COUNT(c.CLAIM_ID), 0), 1) AS denial_rate, "
                    "ROUND(AVG(CAST(m.CHRONIC_CONDITIONS AS REAL)), 2) AS avg_chronic_conditions, "
                    "ROUND(1000.0 * COUNT(DISTINCT CASE WHEN UPPER(e.VISIT_TYPE) = 'EMERGENCY' THEN e.ENCOUNTER_ID END) / "
                    "NULLIF(COUNT(DISTINCT m.MEMBER_ID), 0), 1) AS er_per_1000 "
                    "FROM members m "
                    "LEFT JOIN appointments a ON m.MEMBER_ID = a.MEMBER_ID "
                    "LEFT JOIN claims c ON m.MEMBER_ID = c.MEMBER_ID "
                    "LEFT JOIN encounters e ON m.MEMBER_ID = e.MEMBER_ID "
                    "GROUP BY m.{dim_col} ORDER BY annual_wellness_visit_rate DESC"
                ),
            },
            'cost_effectiveness': {
                'triggers': [r'cost\s+effective', r'value\s+based\s+(?:care|model|analysis)',
                             r'cost\s+per\s+(?:quality|outcome|encounter)\s+by',
                             r'efficiency\s+(?:score|metric|analysis)\s+by'],
                'sql_template': (
                    "SELECT m.{dim_col} AS dimension, "
                    "COUNT(DISTINCT m.MEMBER_ID) AS total_members, "
                    "ROUND(SUM(CAST(c.PAID_AMOUNT AS REAL)) / NULLIF(COUNT(e.ENCOUNTER_ID), 0), 2) AS cost_per_encounter, "
                    "ROUND(SUM(CAST(c.PAID_AMOUNT AS REAL)) / NULLIF(COUNT(DISTINCT m.MEMBER_ID), 0), 2) AS cost_per_member, "
                    "ROUND(1.0 * COUNT(e.ENCOUNTER_ID) / NULLIF(COUNT(DISTINCT m.MEMBER_ID), 0), 2) AS encounters_per_member, "
                    "ROUND(100.0 * COUNT(DISTINCT CASE WHEN UPPER(e.VISIT_TYPE) = 'EMERGENCY' THEN e.ENCOUNTER_ID END) / "
                    "NULLIF(COUNT(e.ENCOUNTER_ID), 0), 1) AS er_pct_of_encounters, "
                    "ROUND(AVG(CAST(m.RISK_SCORE AS REAL)), 3) AS avg_risk "
                    "FROM members m "
                    "LEFT JOIN claims c ON m.MEMBER_ID = c.MEMBER_ID "
                    "LEFT JOIN encounters e ON m.MEMBER_ID = e.MEMBER_ID "
                    "GROUP BY m.{dim_col} ORDER BY cost_per_member DESC"
                ),
            },
            'polypharmacy': {
                'triggers': [r'polypharmacy\s+by', r'medication\s+(?:burden|count|load)\s+by',
                             r'drug\s+(?:burden|interaction|count)\s+by',
                             r'prescription\s+(?:burden|count|load)\s+by'],
                'sql_template': (
                    "SELECT m.{dim_col} AS dimension, "
                    "COUNT(DISTINCT m.MEMBER_ID) AS total_members, "
                    "COUNT(DISTINCT p.RX_ID) AS total_prescriptions, "
                    "ROUND(1.0 * COUNT(DISTINCT p.RX_ID) / NULLIF(COUNT(DISTINCT m.MEMBER_ID), 0), 1) AS rx_per_member, "
                    "COUNT(DISTINCT p.MEDICATION_CLASS) AS unique_drug_classes, "
                    "SUM(CASE WHEN (SELECT COUNT(DISTINCT p2.MEDICATION_CLASS) FROM prescriptions p2 WHERE p2.MEMBER_ID = m.MEMBER_ID) >= 5 THEN 1 ELSE 0 END) AS polypharmacy_flag, "
                    "ROUND(AVG(CAST(p.COST AS REAL)), 2) AS avg_rx_cost "
                    "FROM members m "
                    "LEFT JOIN prescriptions p ON m.MEMBER_ID = p.MEMBER_ID "
                    "GROUP BY m.{dim_col} ORDER BY rx_per_member DESC"
                ),
            },
            'avoidable_er': {
                'triggers': [r'avoidable\s+(?:er|ed|emergency)', r'unnecessary\s+(?:er|ed|emergency)',
                             r'non[\s-]*emergent\s+(?:er|ed)\s+by', r'(?:er|ed)\s+diversion\s+by'],
                'sql_template': (
                    "SELECT m.{dim_col} AS dimension, "
                    "COUNT(DISTINCT m.MEMBER_ID) AS total_members, "
                    "COUNT(DISTINCT CASE WHEN UPPER(e.VISIT_TYPE) = 'EMERGENCY' THEN e.ENCOUNTER_ID END) AS total_er_visits, "
                    "COUNT(DISTINCT CASE WHEN UPPER(e.VISIT_TYPE) = 'EMERGENCY' AND d.SEVERITY IN ('MILD','MODERATE') THEN e.ENCOUNTER_ID END) AS potentially_avoidable_er, "
                    "ROUND(100.0 * COUNT(DISTINCT CASE WHEN UPPER(e.VISIT_TYPE) = 'EMERGENCY' AND d.SEVERITY IN ('MILD','MODERATE') THEN e.ENCOUNTER_ID END) / "
                    "NULLIF(COUNT(DISTINCT CASE WHEN UPPER(e.VISIT_TYPE) = 'EMERGENCY' THEN e.ENCOUNTER_ID END), 0), 1) AS avoidable_er_pct, "
                    "ROUND(AVG(CASE WHEN UPPER(e.VISIT_TYPE) = 'EMERGENCY' THEN CAST(COALESCE(c.PAID_AMOUNT, 0) AS REAL) END), 2) AS avg_er_cost "
                    "FROM members m "
                    "LEFT JOIN encounters e ON m.MEMBER_ID = e.MEMBER_ID "
                    "LEFT JOIN diagnoses d ON e.ENCOUNTER_ID = d.ENCOUNTER_ID AND e.MEMBER_ID = d.MEMBER_ID "
                    "LEFT JOIN claims c ON e.ENCOUNTER_ID = c.ENCOUNTER_ID AND e.MEMBER_ID = c.MEMBER_ID "
                    "GROUP BY m.{dim_col} ORDER BY avoidable_er_pct DESC"
                ),
            },
        }

        _dim_map = {
            'race': 'RACE', 'ethnicity': 'RACE', 'race/ethnicity': 'RACE',
            'gender': 'GENDER', 'sex': 'GENDER',
            'region': 'KP_REGION', 'area': 'KP_REGION', 'location': 'KP_REGION',
            'age group': 'AGE_GROUP', 'age': 'AGE_GROUP',
            'plan type': 'PLAN_TYPE', 'plan': 'PLAN_TYPE',
            'specialty': 'SPECIALTY', 'department': 'DEPARTMENT',
            'facility': 'FACILITY_NAME', 'facility name': 'FACILITY_NAME',
            'visit type': 'VISIT_TYPE', 'encounter type': 'VISIT_TYPE',
            'claim type': 'CLAIM_TYPE', 'claim status': 'CLAIM_STATUS',
            'service type': 'SERVICE_TYPE', 'service category': 'SERVICE_CATEGORY',
            'medication class': 'MEDICATION_CLASS', 'drug class': 'MEDICATION_CLASS',
            'severity': 'SEVERITY', 'diagnosis severity': 'SEVERITY',
            'urgency': 'URGENCY', 'priority': 'URGENCY',
            'zip': 'ZIP_CODE', 'zip code': 'ZIP_CODE',
            'state': 'STATE', 'county': 'COUNTY', 'city': 'CITY',
            'status': 'STATUS', 'category': 'CATEGORY',
        }

        # Which dim cols live on claims vs members vs other tables
        _claims_dims = {'CLAIM_TYPE', 'CLAIM_STATUS', 'SERVICE_TYPE', 'SERVICE_CATEGORY'}
        _encounter_dims = {'VISIT_TYPE', 'FACILITY_NAME', 'DEPARTMENT'}
        _provider_dims = {'SPECIALTY'}
        _diagnosis_dims = {'SEVERITY'}
        _referral_dims = {'URGENCY'}
        _prescription_dims = {'MEDICATION_CLASS'}

        dim_match = re.search(
            r'\bby\s+(race(?:/ethnicity)?|ethnicity|gender|sex|region|area|location|'
            r'age\s*group|age|plan\s*type|plan|specialty|department|facility(?:\s*name)?|'
            r'visit\s*type|encounter\s*type|claim\s*type|claim\s*status|service\s*type|'
            r'service\s*category|medication\s*class|drug\s*class|severity|diagnosis\s*severity|'
            r'urgency|priority|zip(?:\s*code)?|state|county|city|status|category)',
            q_lower
        )
        dim_col = 'RACE'
        if dim_match:
            dim_col = _dim_map.get(dim_match.group(1).strip(), 'RACE')

        for concept_key, concept in _concept_sql_map.items():
            for trigger in concept['triggers']:
                if re.search(trigger, q_lower):
                    sql = concept['sql_template']
                    if not concept.get('fixed_dim'):
                        sql = sql.replace('{dim_col}', dim_col)
                    else:
                        sql = sql.replace('{dim_col}', dim_col)

                    # Resolve {dim_src} for _claim_dim concepts
                    if '{dim_src}' in sql:
                        # Determine which table alias has the dim_col
                        if dim_col in _claims_dims:
                            dim_src = 'c'
                        elif dim_col in _encounter_dims:
                            dim_src = 'e'
                        elif dim_col in _provider_dims:
                            dim_src = 'p'
                        elif dim_col in _diagnosis_dims:
                            dim_src = 'd'
                        elif dim_col in _referral_dims:
                            dim_src = 'r'
                        elif dim_col in _prescription_dims:
                            dim_src = 'p'
                        else:
                            # Default: member dimensions come from members table
                            dim_src = 'm'
                        sql = sql.replace('{dim_src}', dim_src)

                    try:
                        rows, columns, error = self._execute_sql(sql)
                        if error or not rows:
                            logger.warning("Structured concept '%s' SQL failed: %s", concept_key, error)
                            continue

                        elapsed = time.time() - t0
                        narrative = ''
                        try:
                            narrative = self._generate_narrative(question, sql, rows, columns, {})
                        except Exception:
                            narrative = f"Query returned {len(rows)} result(s)."

                        result = {
                            'answer': narrative,
                            'sql': sql,
                            'rows': rows,
                            'results': rows,
                            'columns': columns,
                            'error': None,
                            'row_count': len(rows),
                            'confidence': {'grade': 'A', 'overall': 0.92},
                            'source': 'structured_intelligence',
                            'approach': f'Structured concept: {concept_key} | dim={dim_col} | intelligence_routed=True',
                            'latency_ms': int(elapsed * 1000),
                            'session_id': session_id,
                            'cache_hit': False,
                        }
                        self._update_conversation(session_id, question, result, rows, columns)
                        logger.info("STRUCTURED INTELLIGENCE: %s | concept=%s | dim=%s | %d rows | %dms",
                                   question[:60], concept_key, dim_col, len(rows), int(elapsed * 1000))
                        return result
                    except Exception as e:
                        logger.warning("Structured concept '%s' error: %s", concept_key, e)
                        continue

        if self.semantic_graph and self.intent_parser and self.sql_constructor:
            try:
                t_sem = time.time()
                parsed_intent = self.intent_parser.parse(question)
                logger.info("Structured fallback parsed: intent=%s, tables=%s, group_by=%s, conf=%.2f",
                            parsed_intent.intent, parsed_intent.tables,
                            parsed_intent.group_by, parsed_intent.confidence)

                if parsed_intent.confidence >= 0.30:
                    construct_result = self.sql_constructor.construct(parsed_intent)
                    sem_sql = construct_result.get('sql', '')
                    sem_conf = construct_result.get('confidence', 0)

                    if sem_sql and sem_conf >= 0.45:
                        _brain_conf = parsed_intent.confidence >= 0.60
                        sanitized = self._sanitize_sql(sem_sql, question, _brain_generated=_brain_conf)
                        rows, columns, error = self._execute_sql(sanitized)

                        if not error and rows is not None and len(rows) > 0:
                            elapsed = time.time() - t0
                            narrative = ''
                            try:
                                narrative = self._generate_narrative(question, sanitized, rows, columns, {})
                            except Exception:
                                narrative = f"Query returned {len(rows)} result(s)."

                            result = {
                                'answer': narrative,
                                'sql': sanitized,
                                'rows': rows,
                                'results': rows,
                                'columns': columns,
                                'error': None,
                                'row_count': len(rows),
                                'confidence': {'grade': 'A' if sem_conf >= 0.7 else 'B', 'overall': sem_conf},
                                'source': 'structured_semantic_intelligence',
                                'approach': f'Structured semantic: intent={parsed_intent.intent} | tables={parsed_intent.tables} | conf={sem_conf:.2f}',
                                'latency_ms': int(elapsed * 1000),
                                'session_id': session_id,
                                'cache_hit': False,
                            }
                            self._update_conversation(session_id, question, result, rows, columns)
                            logger.info("STRUCTURED SEMANTIC: %s | %d rows | conf=%.2f | %dms",
                                       question[:60], len(rows), sem_conf, int(elapsed * 1000))
                            return result
            except Exception as e:
                logger.warning("Structured semantic fallback error: %s", e)

        return None

    def _process_core(self, question: str, session_id: str = 'default') -> Dict[str, Any]:
        t0 = time.time()

        reasoning_chain_obj = None
        if self.reasoning_chain_available:
            try:
                from reasoning_chain import ReasoningChain
                reasoning_chain_obj = ReasoningChain(question)
            except Exception as e:
                logger.debug("ReasoningChain initialization error (non-fatal): %s", e)

        _corrected = self._correct_typos(question)
        if _corrected != question:
            logger.info("Typo correction: '%s' → '%s'", question[:60], _corrected[:60])
            question = _corrected

        _is_structured = self._is_structured_data_query(question)
        if _is_structured:
            logger.info("STRUCTURED DATA QUERY detected — routing to SQL intelligence path: %s", question[:80])
            _struct_result = self._route_structured_query(question, t0, session_id)
            if _struct_result:
                return _struct_result
            logger.info("Structured query path returned no result, falling through to standard routing")

        if self.demographic_analytics:
            _q_lower = question.lower()
            _demo_result = None
            _demo_label = None
            if any(k in _q_lower for k in ['member demographic', 'patient demographic', 'member profile', 'population profile', 'demographic analysis', 'demographic overview', 'demographic deep dive', 'demographic breakdown']):
                _demo_result = self.demographic_analytics.analyze_member_demographics()
                _demo_label = 'Member Demographics Deep-Dive'
            elif any(k in _q_lower for k in ['provider demographic', 'provider profile', 'specialty distribution', 'provider distribution']):
                _demo_result = self.demographic_analytics.analyze_provider_demographics()
                _demo_label = 'Provider Demographics Deep-Dive'
            elif any(k in _q_lower for k in ['health equity', 'equity score', 'disparity', 'disparities', 'health inequit']):
                _demo_result = self.demographic_analytics.get_health_equity_scorecard()
                _demo_label = 'Health Equity Scorecard'
            elif any(k in _q_lower for k in ['geographic demographic', 'regional demographic', 'place demographic', 'location demographic', 'city demographic', 'state demographic']):
                _demo_result = self.demographic_analytics.analyze_geographic_demographics()
                _demo_label = 'Geographic Demographics Deep-Dive'
            elif any(k in _q_lower for k in ['health status demographic', 'comorbidity analysis', 'disease burden analysis', 'chronic demographic analysis', 'severity distribution analysis']):
                _demo_result = self.demographic_analytics.analyze_health_status_demographics()
                _demo_label = 'Health Status Demographics Deep-Dive'
            elif any(k in _q_lower for k in ['socioeconomic', 'income proxy', 'cost burden', 'ses analysis', 'economic demographic']):
                _demo_result = self.demographic_analytics.analyze_socioeconomic_proxy()
                _demo_label = 'Socioeconomic Proxy Analysis'
            elif any(k in _q_lower for k in ['utilization demographic', 'utilization by race', 'utilization by gender', 'healthcare utilization pattern']):
                _demo_result = self.demographic_analytics.analyze_utilization_demographics()
                _demo_label = 'Utilization Demographics Deep-Dive'
            elif any(k in _q_lower for k in ['comprehensive demographic', 'full demographic report', 'complete demographic', 'all demographic']):
                _demo_result = self.demographic_analytics.get_comprehensive_demographic_report()
                _demo_label = 'Comprehensive Demographics Report'

            if _demo_result and _demo_label:
                _dimensions = []
                _dim_configs = [
                    ('age_stratification', 'tiers', 'Age Group', 'Members', 'Age Distribution',
                     'Breakdown of member population by age tier — Senior (65+), Middle Age (45-64), Adult (26-44), Young Adult (18-25), Pediatric (<18).',
                     'Age stratification is critical for risk adjustment, preventive care targeting, and benefit design. Senior and pediatric populations drive disproportionate cost and clinical complexity.'),
                    ('risk_stratification', 'tiers', 'Risk Tier', 'Members', 'Risk Stratification',
                     'Distribution of members across risk tiers based on HCC risk scores and chronic condition burden.',
                     'Risk stratification drives capitation rate setting, care management enrollment, and HEDIS/Stars performance. Very High risk members typically account for 60-80% of total cost.'),
                    ('gender_distribution', 'distribution', 'Gender', 'Members', 'Gender Distribution',
                     'Male vs Female member distribution across the population.',
                     'Gender balance affects utilization patterns, preventive screening rates (mammography, prostate), and maternity cost projections.'),
                    ('race_distribution', 'distribution', 'Race/Ethnicity', 'Members', 'Race & Ethnicity Distribution',
                     'Racial and ethnic composition of the member population.',
                     'Race/ethnicity data is foundational for health equity analysis, HEDIS stratified reporting, and targeted intervention design to reduce disparities.'),
                    ('language_distribution', 'distribution', 'Language', 'Members', 'Language Distribution',
                     'Primary language spoken by members.',
                     'Language data drives interpreter service capacity planning, multilingual material production, and culturally competent care program design.'),
                    ('region_distribution', 'distribution', 'Region', 'Members', 'Regional Distribution',
                     'Geographic distribution of members across service regions.',
                     'Regional distribution affects network adequacy, facility planning, and regional cost variation analysis.'),
                ]
                for _key, _sub_key, _col0, _col1, _label, _insight, _med_ctx in _dim_configs:
                    _strat = _demo_result.get(_key, {})
                    _data = _strat.get(_sub_key, {})
                    if _data and isinstance(_data, dict):
                        _sorted = sorted(_data.items(), key=lambda x: x[1], reverse=True)
                        _dim_rows = [[k, v] for k, v in _sorted]
                        _dimensions.append({
                            'label': _label,
                            'columns': [_col0, _col1],
                            'rows': _dim_rows,
                            'row_count': len(_dim_rows),
                            'chart_type': 'bar',
                            'insight': _insight,
                            'medical_context': _med_ctx,
                        })

                _equity = _demo_result.get('equity_metrics', {})
                for _eq_key, _eq_label, _eq_insight in [
                    ('gender', 'Cost & Risk by Gender', 'Average cost and risk score comparison across genders — key indicator for utilization equity.'),
                    ('race', 'Cost & Risk by Race/Ethnicity', 'Average cost and risk score by race — critical for identifying health disparities and equity gaps.'),
                    ('region', 'Cost & Risk by Region', 'Regional cost and risk variation — indicates network efficiency and population health differences.'),
                ]:
                    _eq_data = _equity.get(_eq_key, {})
                    if _eq_data and isinstance(_eq_data, dict):
                        _eq_rows = []
                        for _cat, _metrics in sorted(_eq_data.items()):
                            if isinstance(_metrics, dict):
                                _eq_rows.append([
                                    _cat,
                                    round(_metrics.get('avg_cost', 0), 0),
                                    round(_metrics.get('avg_risk_score', 0), 2),
                                    _metrics.get('member_count', 0),
                                ])
                        if _eq_rows:
                            _dimensions.append({
                                'label': _eq_label,
                                'columns': ['Category', 'Average Cost', 'Risk Score', 'Members'],
                                'rows': _eq_rows,
                                'row_count': len(_eq_rows),
                                'chart_type': 'bar',
                                'insight': _eq_insight,
                                'medical_context': 'Equity metrics are required for CMS Health Equity Index reporting and NCQA accreditation. Disparities exceeding 10% warrant targeted intervention programs.',
                            })

                _narrative = f"## {_demo_label}\n\n"
                _total = _demo_result.get('age_stratification', {}).get('total', 0)
                _narrative += f"Comprehensive demographic analysis across **{_total:,}** members, spanning {len(_dimensions)} analytical dimensions including age, risk, gender, race/ethnicity, language, region, and equity metrics.\n\n"

                _suggestions = [
                    'health equity scorecard',
                    'high risk members by region',
                    'cost analysis by age group',
                    'provider demographics',
                    'chronic condition burden by race',
                    'utilization patterns by gender',
                ]

                result = {
                    'answer': _narrative,
                    'sql': f'-- Demographics deep-dive: {_demo_label}',
                    'rows': [],
                    'columns': [],
                    'source': 'demographic_analytics',
                    'latency_ms': round((time.time() - t0) * 1000),
                    'confidence': {'grade': 'A', 'overall': 0.95},
                    'cache_hit': False,
                    'error': None,
                    'narrative': _narrative,
                    'demographics': _demo_result,
                    'is_concept': True,
                    'label': _demo_label,
                    'description': f'Comprehensive analysis across {len(_dimensions)} demographic dimensions for {_total:,} members',
                    'dimensions': _dimensions,
                    'total_dimensions': len(_dimensions),
                    'suggestions': _suggestions,
                }
                self._update_conversation(session_id, question, result, [], [])
                return result

        if self.analytical_intelligence:
            try:
                if self.analytical_intelligence.should_analyze(question):
                    logger.info("AnalyticalIntelligence triggered for: %s", question[:60])
                    _ai_result = self.analytical_intelligence.analyze(question)
                    if _ai_result and not _ai_result.get('error'):
                        _ai_result['latency_ms'] = round((time.time() - t0) * 1000)
                        _ai_result['answer'] = _ai_result.get('narrative', '')
                        _ai_result.setdefault('confidence', {'grade': 'A', 'overall': 0.95})
                        _ai_result.setdefault('cache_hit', False)
                        self._update_conversation(session_id, question, _ai_result,
                                                  _ai_result.get('rows', []),
                                                  _ai_result.get('columns', []))
                        logger.info("AnalyticalIntelligence returned %d queries, %d insights for: %s",
                                   _ai_result.get('query_count', 0),
                                   len(_ai_result.get('insights', [])),
                                   question[:60])
                        return _ai_result
            except Exception as e:
                logger.warning("AnalyticalIntelligence error (falling through): %s", e)

        _superlative_result = self._try_superlative_scalar(question, t0, session_id)
        if _superlative_result:
            logger.info("Superlative scalar interceptor matched: %s", question[:60])
            return _superlative_result

        qi_result = self._try_query_intelligence(question, t0, session_id)
        if qi_result:
            logger.info("QueryIntelligence PRIMARY match for: %s (confidence=%.2f)",
                       question[:60], qi_result.get('_qi_confidence', 0))
            return qi_result

        golden_result = self._try_golden_template(question, t0, session_id)
        if golden_result:
            logger.info("Golden template FALLBACK match for: %s", question[:60])
            return golden_result

        if self.executive_dashboard_engine:
            _q_lower = question.lower()
            _exec_result = None
            _exec_label = None
            _exec_methods = {
                ('member experience', 'cahps', 'satisfaction score', 'patient experience', 'voluntary termination'):
                    ('get_member_experience', 'Member Experience Dashboard'),
                ('star rating', 'stars measure', 'stars performance', 'hedis', 'cms star', '5 star', 'five star'):
                    ('get_stars_performance', 'Stars Measure Performance'),
                ('risk adjustment', 'rada', 'coding accuracy', 'hcc capture', 'risk score distribution'):
                    ('get_risk_adjustment_coding', 'Risk Adjustment & Coding Accuracy'),
                ('financial performance', 'pmpm financial', 'pmpm dashboard', 'medical loss ratio', 'revenue expense', 'financial dashboard', 'ytd financial'):
                    ('get_financial_performance', 'Financial Performance Dashboard'),
                ('membership market', 'market share', 'enrollment growth', 'growth driver', 'membership dashboard'):
                    ('get_membership_market_share', 'Membership & Market Share'),
                ('service utilization', 'utilization per 1000', 'util/1000', 'bed days', 'ed visit rate', 'utilization dashboard'):
                    ('get_service_utilization', 'Service Utilization Metrics'),
                ('executive summary', 'executive dashboard', 'kp scorecard', 'strategic priorities', 'performance scorecard', 'full dashboard'):
                    ('get_executive_summary', 'Executive Summary & Scorecard'),
            }
            for keywords, (method_name, label) in _exec_methods.items():
                if any(k in _q_lower for k in keywords):
                    try:
                        _method = getattr(self.executive_dashboard_engine, method_name)
                        _exec_result = _method()
                        _exec_label = label
                    except Exception as e:
                        logger.debug("ExecutiveDashboard %s error: %s", method_name, e)
                    break

            if _exec_result and _exec_label:
                if self.self_healing_engine:
                    try:
                        _exec_result = self.self_healing_engine.validate_dashboard(
                            _exec_result, _exec_label
                        )
                        _zeros = self.self_healing_engine.detect_suspicious_zeros(
                            _exec_result, _exec_label
                        )
                        if _zeros:
                            logger.warning("Self-healing flagged %d suspicious zeros in %s",
                                           len(_zeros), _exec_label)
                    except Exception as e:
                        logger.debug("Self-healing dashboard validation error: %s", e)

                _exec_narrative = f"## {_exec_label}\n\n"
                if isinstance(_exec_result, dict):
                    for k, v in list(_exec_result.items())[:8]:
                        if isinstance(v, (str, int, float)):
                            _exec_narrative += f"**{k}**: {v}\n\n"
                        elif isinstance(v, dict):
                            _exec_narrative += f"**{k}**: {json.dumps(v, default=str)[:200]}\n\n"
                result = {
                    'answer': _exec_narrative,
                    'sql': f'-- Executive Dashboard: {_exec_label}',
                    'rows': [[json.dumps(_exec_result, default=str)]],
                    'columns': ['executive_dashboard'],
                    'source': 'executive_dashboard_engine',
                    'latency_ms': round((time.time() - t0) * 1000),
                    'confidence': {'grade': 'A', 'overall': 0.95},
                    'cache_hit': False,
                    'error': None,
                    'narrative': _exec_narrative,
                    'executive_dashboard': _exec_result,
                    'dashboard_type': _exec_label,
                }
                self._update_conversation(session_id, question, result, result['rows'], result['columns'])
                return result

        if self.revenue_optimization:
            _q_lower = question.lower()
            _rev_result = None
            _rev_label = None
            if any(k in _q_lower for k in ['churn risk', 'attrition risk', 'member churn', 'disenrollment risk']):
                try:
                    _rev_result = self.revenue_optimization.get_churn_risk_analysis()
                    _rev_label = 'Churn Risk Analysis'
                except Exception as e:
                    logger.debug("Revenue churn error: %s", e)
            elif any(k in _q_lower for k in ['pmpm optim', 'cost optim', 'pmpm reduction', 'cost reduction opportunity']):
                try:
                    _rev_result = self.revenue_optimization.get_pmpm_optimization()
                    _rev_label = 'PMPM Optimization Analysis'
                except Exception as e:
                    logger.debug("Revenue PMPM error: %s", e)
            elif any(k in _q_lower for k in ['denial recovery', 'denial analysis', 'denied claims recovery', 'claim denial analysis', 'claim denial recovery']):
                try:
                    _rev_result = self.revenue_optimization.get_denial_recovery()
                    _rev_label = 'Denial Recovery Analysis'
                except Exception as e:
                    logger.debug("Revenue denial error: %s", e)
            elif any(k in _q_lower for k in ['hcc gap', 'coding gap', 'risk adjustment gap', 'hcc closure']):
                try:
                    _rev_result = self.revenue_optimization.get_hcc_gap_closure()
                    _rev_label = 'HCC Gap Closure Analysis'
                except Exception as e:
                    logger.debug("Revenue HCC error: %s", e)
            elif any(k in _q_lower for k in ['retention target', 'retention campaign', 'retention strategy', 'keep member']):
                try:
                    _rev_result = self.revenue_optimization.get_retention_targeting()
                    _rev_label = 'Retention Targeting Analysis'
                except Exception as e:
                    logger.debug("Revenue retention error: %s", e)
            elif any(k in _q_lower for k in ['revenue optim', 'revenue opportunity', 'revenue dashboard', 'total revenue opportunity']):
                try:
                    _rev_result = self.revenue_optimization.get_revenue_dashboard()
                    _rev_label = 'Revenue Optimization Dashboard'
                except Exception as e:
                    logger.debug("Revenue dashboard error: %s", e)

            if _rev_result and _rev_label:
                _rev_narrative = f"## {_rev_label}\n\n"
                if isinstance(_rev_result, dict):
                    for k, v in list(_rev_result.items())[:8]:
                        if isinstance(v, (str, int, float)):
                            _rev_narrative += f"**{k}**: {v}\n\n"
                        elif isinstance(v, dict):
                            _rev_narrative += f"**{k}**: {json.dumps(v, default=str)[:200]}\n\n"
                result = {
                    'answer': _rev_narrative,
                    'sql': f'-- Revenue Optimization: {_rev_label}',
                    'rows': [[json.dumps(_rev_result, default=str)]],
                    'columns': ['revenue_optimization'],
                    'source': 'revenue_optimization_engine',
                    'latency_ms': round((time.time() - t0) * 1000),
                    'confidence': {'grade': 'A', 'overall': 0.92},
                    'cache_hit': False,
                    'error': None,
                    'narrative': _rev_narrative,
                    'revenue_optimization': _rev_result,
                    'analytics_type': _rev_label,
                }
                self._update_conversation(session_id, question, result, result['rows'], result['columns'])
                return result

        t_analytics = time.time()
        analytics_result = self._check_analytics(question, t0, session_id)
        if analytics_result:
            return analytics_result
        logger.debug("Analytics check took %.1fms", (time.time() - t_analytics) * 1000)

        if self._business_insights_available:
            try:
                from business_insights import rising_risk_cohort, readmit_watchlist, hedis_gap_list, network_performance
                _q_lower = question.lower()
                _biz_result = None
                _biz_label = None
                if any(k in _q_lower for k in ['rising risk', 'risk cohort', 'high risk', 'at risk']):
                    _biz_result = rising_risk_cohort(self.db_path)
                    _biz_label = 'Rising Risk Cohort'
                elif any(k in _q_lower for k in ['readmit', 'readmission', 'bounce back']):
                    _biz_result = readmit_watchlist(self.db_path)
                    _biz_label = 'Readmission Watchlist'
                elif any(k in _q_lower for k in ['hedis', 'gap in care', 'care gap', 'quality gap']):
                    _biz_result = hedis_gap_list(self.db_path)
                    _biz_label = 'HEDIS Care Gaps'
                elif any(k in _q_lower for k in ['network perf', 'provider network', 'network score']):
                    _biz_result = network_performance(self.db_path)
                    _biz_label = 'Network Performance'

                if _biz_result and _biz_result.get('status') == 'ok':
                    _biz_rows = _biz_result.get('rows', [])
                    _biz_cols = _biz_result.get('columns', [])
                    if not _biz_rows and isinstance(_biz_result.get('data'), list):
                        _biz_rows = _biz_result['data']
                        _biz_cols = list(_biz_rows[0].keys()) if _biz_rows and isinstance(_biz_rows[0], dict) else []
                    result = {
                        'sql': f"-- Business Intelligence: {_biz_label}",
                        'rows': _biz_rows[:200] if isinstance(_biz_rows, list) else [],
                        'columns': _biz_cols,
                        'row_count': len(_biz_rows) if isinstance(_biz_rows, list) else 0,
                        'analytics_model': 'business_insights',
                        'analytics_type': _biz_label,
                        'analytics_data': _biz_result,
                        'latency_ms': round((time.time() - t0) * 1000),
                        'source': f'business_insights/{_biz_label}',
                        'cache_hit': False,
                        'confidence': {'grade': 'A', 'overall': 0.90},
                        'confidence_grade': 'A',
                        'confidence_overall': 0.90,
                        'error': None,
                        'narrative': _biz_result.get('narrative', f'{_biz_label} analysis complete.'),
                        'suggestions': _biz_result.get('suggestions', []),
                        'anomalies': [],
                        'dashboard': {
                            'chart_type': 'table',
                            'chart_config': {},
                            'title': _biz_label,
                            'subtitle': 'Business Intelligence Module',
                        },
                    }
                    self.semantic_cache.put(question, result)
                    if self.answer_cache:
                        try:
                            self.answer_cache.store(question, result)
                        except Exception:
                            pass
                    self._update_conversation(session_id, question, result,
                                              result['rows'], result['columns'])
                    logger.info("BusinessInsights '%s' handled query in %dms",
                                _biz_label, result['latency_ms'])
                    return result
            except Exception as e:
                logger.debug("BusinessInsights error (falling through): %s", e)

        if self.orchestrator:
            t_orch = time.time()
            try:
                orch_result = self.orchestrator.route(question)
                _orch_conf = orch_result.get('orchestrator', {}).get('confidence', 0) if orch_result else 0
                if orch_result and _orch_conf >= 0.50 and (
                                    orch_result.get('status') == 'ok'
                                    or orch_result.get('method')
                                    or orch_result.get('ate') is not None
                                    or orch_result.get('narrative')
                                    or orch_result.get('engine')):
                    engine_name = orch_result.get('engine', 'orchestrator')
                    orch_meta = orch_result.get('orchestrator', {})
                    _narrative = orch_result.get('narrative', '')
                    _data = {k: v for k, v in orch_result.items()
                             if k not in ('narrative', 'engine', 'orchestrator', 'dq_badge')}
                    _rows = []
                    _columns = ['metric', 'value']
                    for k, v in _data.items():
                        if isinstance(v, (str, int, float)):
                            _rows.append([k, v])

                    result = {
                        'sql': f"-- Advanced Analytics: {engine_name}\n"
                               f"-- Intent: {orch_meta.get('intent', 'unknown')} "
                               f"(confidence: {orch_meta.get('confidence', 0):.2f})",
                        'rows': _rows,
                        'columns': _columns,
                        'row_count': len(_rows),
                        'analytics_model': engine_name,
                        'analytics_type': orch_meta.get('intent', 'advanced'),
                        'analytics_data': _data,
                        'latency_ms': round((time.time() - t0) * 1000),
                        'source': f'orchestrator/{engine_name}',
                        'cache_hit': False,
                        'confidence': {'grade': 'A', 'overall': 0.92},
                        'confidence_grade': 'A',
                        'confidence_overall': 0.92,
                        'error': None,
                        'narrative': _narrative,
                        'dq_badge': orch_result.get('dq_badge', {}),
                        'suggestions': [
                            f"Ask about {h}" for h in
                            [e for e, _ in (orch_meta.get('all_intents', []))[1:3]]
                        ] if orch_meta.get('all_intents') else [],
                        'anomalies': [],
                        'dashboard': {
                            'chart_type': 'analytics_detail',
                            'chart_config': {'engine': engine_name},
                            'title': engine_name.replace('_', ' ').title(),
                            'subtitle': f"Advanced Analytics — {orch_meta.get('intent', '')}",
                        },
                    }
                    self.semantic_cache.put(question, result)
                    self._update_conversation(session_id, question, result, _rows, _columns)
                    logger.info("Orchestrator '%s' handled query in %dms: %s",
                                engine_name, result['latency_ms'], question[:60])
                    return result
            except Exception as e:
                logger.warning("Orchestrator error (falling through): %s", e)
            logger.debug("Orchestrator check took %.1fms", (time.time() - t_orch) * 1000)

        _skip_concept_expander = False
        _q_lower_ce = question.lower()
        _specific_indicators = [
            'sickest', 'highest risk', 'high risk', 'risk score',
            'denial rate', 'denied', 'denial',
            'diabetes', 'hypertension', 'asthma', 'copd', 'cancer',
            'hmo', 'ppo', 'plan type',
            'paid amount', 'billed amount', 'cost per',
            'length of stay', 'readmission',
            'ncal', 'scal', 'co', 'mid',
            'top 5', 'top 10', 'top 3',
            'cumulative', 'grand total', 'average', 'avg', 'mean',
            'maximum', 'minimum', 'sum of', 'total number',
            'how many', 'count', 'break down', 'broken down',
            'stratified', 'segmented',
            'costliest', 'busiest', 'most expensive', 'most commonly',
            'most frequently', 'biggest caseload', 'ranked',
            'largest to smallest', 'smallest to largest',
            'compare', 'compared to', 'versus', 'vs',
            'loss ratio', 'clean claim ratio',
            'revenue', 'visit type', 'visit types', 'most revenue',
            'generate the most', 'generates the most',
            'frequency', 'claim denial',
        ]
        if any(ind in _q_lower_ce for ind in _specific_indicators):
            _skip_concept_expander = True
            logger.info("Skipping concept_expander — question has specific analytical indicators: %s", question[:60])
        try:
            _dc = self.sql_engine.composer._detect_derived_concept(question)
            if _dc:
                _dc_key, _dc_def = _dc
                if _dc_def['type'] in ('profit_margin', 'denial_impact', 'zero_utilization',
                                        'disenrollment_analysis', 'temporal_financial',
                                        'specialty_cost'):
                    logger.info("Skipping concept_expander — derived concept '%s' will handle: %s",
                                _dc_key, question[:60])
                    _skip_concept_expander = True
        except Exception:
            pass
        t_concept = time.time()
        concept_key = self.concept_expander.detect_concept(question)
        if concept_key and not _skip_concept_expander:
            concept_result = self.concept_expander.expand(question, concept_key)
            if concept_result and concept_result.get('dimensions'):
                logger.info("Concept expansion: '%s' → %d dimensions in %dms",
                           concept_key, concept_result['total_dimensions'],
                           concept_result['execution_time_ms'])
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
                self.semantic_cache.put(question, concept_result)
                self._update_conversation(
                    session_id, question, concept_result,
                    concept_result.get('rows', []),
                    concept_result.get('columns', [])
                )
                return concept_result
        logger.debug("Concept expansion check took %.1fms", (time.time() - t_concept) * 1000)

        if self.answer_cache:
            try:
                t_ans_cache = time.time()
                ans_hit = self.answer_cache.lookup(question)
                if ans_hit:
                    cached_sql_check = (ans_hit.get('sql', '') or '').lower()
                    if any(p in cached_sql_check for p in self._INTERNAL_TABLE_PREFIXES):
                        logger.info("AnswerCache hit REJECTED — contains internal table reference")
                        ans_hit = None
                if ans_hit:
                    _ac_sql_lower = (ans_hit.get('sql', '') or '').lower()
                    _ac_q_lower = question.lower()
                    _filter_mismatch = False
                    _status_filters = {
                        "claim_status = 'denied'": ['denied', 'denial', 'reject'],
                        "claim_status = 'paid'": ['paid', 'approved', 'accepted'],
                        "claim_status = 'pending'": ['pending', 'in process', 'awaiting'],
                        "claim_status = 'adjusted'": ['adjusted', 'adjustment'],
                        "referral_status = 'denied'": ['denied', 'denial', 'reject'],
                        "referral_status = 'approved'": ['approved', 'accepted'],
                    }
                    for sql_filter, required_words in _status_filters.items():
                        if sql_filter in _ac_sql_lower and not any(w in _ac_q_lower for w in required_words):
                            _filter_mismatch = True
                            logger.info("AnswerCache hit REJECTED — SQL has filter '%s' but question doesn't mention %s",
                                        sql_filter, required_words)
                            break
                    if _filter_mismatch:
                        ans_hit = None
                if ans_hit:
                    _ac_sql = (ans_hit.get('sql', '') or '').upper()
                    _ac_q = question.lower()
                    _intent_mismatch = False
                    if any(w in _ac_q for w in ['how many', 'count of', 'number of', 'total number']):
                        if any(f in _ac_sql for f in ['SUM(', 'AVG(', 'MAX(', 'MIN(']) and 'COUNT(' not in _ac_sql:
                            _intent_mismatch = True
                            logger.info("AnswerCache REJECTED — question is COUNT but SQL has agg function")
                    _agg_signal_words = ['average', 'avg', 'mean', 'typical', 'expected',
                                         'total cost', 'total spend', 'total paid', 'total billed',
                                         'cumulative', 'grand total', 'tally up', 'add up',
                                         'sum of', 'combined',
                                         'peak', 'floor', 'ceiling',
                                         'highest', 'lowest', 'maximum', 'minimum',
                                         'per claim', 'per visit', 'per encounter', 'per member']
                    if any(w in _ac_q for w in _agg_signal_words):
                        if 'COUNT(*)' in _ac_sql and not any(f in _ac_sql for f in ['SUM(', 'AVG(', 'MAX(', 'MIN(']):
                            _intent_mismatch = True
                            logger.info("AnswerCache REJECTED — question is agg but SQL is plain COUNT")
                    COLUMN_KEYWORDS = {
                        'copay': 'COPAY', 'co-pay': 'COPAY', 'copayment': 'COPAY',
                        'out of pocket': 'COPAY',
                        'coinsurance': 'COINSURANCE', 'co-insurance': 'COINSURANCE',
                        'cost sharing': 'COINSURANCE',
                        'deductible': 'DEDUCTIBLE',
                        'billed': 'BILLED_AMOUNT', 'billed amount': 'BILLED_AMOUNT',
                        'charge': 'BILLED_AMOUNT', 'charges': 'BILLED_AMOUNT',
                        'submitted': 'BILLED_AMOUNT',
                        'allowed': 'ALLOWED_AMOUNT', 'allowed amount': 'ALLOWED_AMOUNT',
                        'contracted rate': 'ALLOWED_AMOUNT', 'negotiated rate': 'ALLOWED_AMOUNT',
                        'paid amount': 'PAID_AMOUNT', 'reimbursement': 'PAID_AMOUNT',
                        'reimbursing': 'PAID_AMOUNT',
                        'risk score': 'RISK_SCORE', 'acuity': 'RISK_SCORE',
                        'panel size': 'PANEL_SIZE', 'panel': 'PANEL_SIZE',
                        'physician panel': 'PANEL_SIZE', 'caseload': 'PANEL_SIZE',
                        'hcc score': 'RISK_SCORE', 'patient complexity': 'RISK_SCORE',
                        'length of stay': 'LENGTH_OF_STAY', 'los': 'LENGTH_OF_STAY',
                        'stay duration': 'LENGTH_OF_STAY', 'days admitted': 'LENGTH_OF_STAY',
                        'panel size': 'PANEL_SIZE', 'patient panel': 'PANEL_SIZE',
                        'caseload': 'PANEL_SIZE',
                        'specialty': 'SPECIALTY',
                        'medication class': 'MEDICATION_CLASS',
                        'drug class': 'MEDICATION_CLASS',
                        'therapeutic class': 'MEDICATION_CLASS',
                        'plan type': 'PLAN_TYPE',
                        'rvu': 'RVU',
                        'race': 'RACE', 'ethnicity': 'RACE',
                    }
                    AGG_KEYWORDS = {
                        'average': 'AVG', 'avg': 'AVG', 'mean': 'AVG', 'typical': 'AVG',
                        'floor': 'MIN', 'minimum': 'MIN', 'lowest': 'MIN',
                        'peak': 'MAX', 'maximum': 'MAX', 'ceiling': 'MAX',
                        'grand total': 'SUM', 'cumulative': 'SUM', 'tally up': 'SUM',
                    }
                    for kw, agg_fn in AGG_KEYWORDS.items():
                        if kw in _ac_q:
                            if f'{agg_fn}(' not in _ac_sql and 'COUNT(' in _ac_sql.upper():
                                _intent_mismatch = True
                                logger.info("AnswerCache REJECTED — question has '%s' (%s) but SQL has COUNT() instead", kw, agg_fn)
                                break
                    for kw, col_name in COLUMN_KEYWORDS.items():
                        if kw in _ac_q and col_name not in _ac_sql:
                            _intent_mismatch = True
                            logger.info("AnswerCache REJECTED — question mentions '%s' but SQL missing %s", kw, col_name)
                            break
                    if not _intent_mismatch:
                        ENTITY_TABLE_KEYWORDS = {
                            'physician': 'providers', 'doctor': 'providers',
                            'clinician': 'providers', 'provider': 'providers',
                            'headcount': 'providers', 'panel size': 'providers',
                            'physician panel': 'providers',
                            'admission': 'encounters', 'visit': 'encounters',
                            'hospitalization': 'encounters',
                            'authorization': 'referrals', 'referral': 'referrals',
                            'prescription': 'prescriptions', 'medication': 'prescriptions',
                            'drug': 'prescriptions', 'rx': 'prescriptions',
                            'script': 'prescriptions',
                            'diagnosis': 'diagnoses', 'condition': 'diagnoses',
                            'icd': 'diagnoses',
                        }
                        for kw, req_table in ENTITY_TABLE_KEYWORDS.items():
                            if kw in _ac_q and req_table not in _ac_sql_lower:
                                _intent_mismatch = True
                                logger.info("AnswerCache REJECTED — question mentions '%s' but SQL missing table '%s'", kw, req_table)
                                break
                    if _intent_mismatch:
                        ans_hit = None
                if ans_hit:
                    _cached_sql_text = ans_hit.get('sql', '') or ''
                    if _cached_sql_text and not self._validate_sql_matches_question(question, _cached_sql_text):
                        logger.info("AnswerCache hit REJECTED by shared validation for '%s'", question[:50])
                        ans_hit = None
                if ans_hit:
                    cached_sql = ans_hit.get('sql', '')
                    if cached_sql and self.quality_gate:
                        try:
                            validation = self.quality_gate.validate(
                                question, cached_sql,
                                understanding=None
                            )
                            if validation.violations:
                                logger.info("AnswerCache hit REJECTED due to validation failures: %s",
                                            validation.violations)
                            else:
                                _cached_sql = ans_hit.get('sql', '')
                                if _cached_sql:
                                    _sanitized = self._sanitize_sql(_cached_sql, question)
                                    if _sanitized != _cached_sql:
                                        logger.info("AnswerCache SQL sanitized for: %s", question[:40])
                                        ans_hit['sql'] = _sanitized
                                        _rows, _cols, _err = self._execute_sql(_sanitized)
                                        if not _err:
                                            ans_hit['rows'] = _rows
                                            ans_hit['results'] = _rows
                                            ans_hit['columns'] = _cols
                                ans_hit['latency_ms'] = round((time.time() - t0) * 1000)
                                ans_hit['source'] = 'answer_cache'
                                ans_hit['cache_hit'] = True
                                logger.info("AnswerCache HIT for '%s' in %dms (similarity-keyed, validated)",
                                            question[:40], ans_hit['latency_ms'])
                                self._update_conversation(
                                    session_id, question, ans_hit,
                                    ans_hit.get('rows', []), ans_hit.get('columns', [])
                                )
                                return ans_hit
                        except Exception as e:
                            logger.debug("AnswerCache validation error (non-fatal): %s", e)
                    else:
                        _cached_sql = ans_hit.get('sql', '')
                        if _cached_sql:
                            _sanitized = self._sanitize_sql(_cached_sql, question)
                            if _sanitized != _cached_sql:
                                logger.info("AnswerCache SQL sanitized (unvalidated) for: %s", question[:40])
                                ans_hit['sql'] = _sanitized
                                _rows, _cols, _err = self._execute_sql(_sanitized)
                                if not _err:
                                    ans_hit['rows'] = _rows
                                    ans_hit['results'] = _rows
                                    ans_hit['columns'] = _cols
                        ans_hit['latency_ms'] = round((time.time() - t0) * 1000)
                        ans_hit['source'] = 'answer_cache'
                        ans_hit['cache_hit'] = True
                        logger.info("AnswerCache HIT for '%s' in %dms (similarity-keyed, unvalidated)",
                                    question[:40], ans_hit['latency_ms'])
                        self._update_conversation(
                            session_id, question, ans_hit,
                            ans_hit.get('rows', []), ans_hit.get('columns', [])
                        )
                        return ans_hit
                logger.debug("AnswerCache lookup took %.1fms", (time.time() - t_ans_cache) * 1000)
            except Exception as e:
                logger.debug("AnswerCache lookup error (non-fatal): %s", e)

        t_cache = time.time()
        cached = self.semantic_cache.get(question)
        if cached:
            _cached_sql_lc = (cached.get('sql', '') or '').lower()
            if any(p in _cached_sql_lc for p in self._INTERNAL_TABLE_PREFIXES):
                logger.info("SemanticCache hit REJECTED — contains internal table reference")
                cached = None
        if cached:
            cached_sql = cached.get('sql', '')
            cache_valid = True
            if cached_sql and self.quality_gate and not cached_sql.startswith('--'):
                try:
                    _cv = self.quality_gate.validate(question, cached_sql)
                    if _cv.violations:
                        logger.info("SemanticCache hit REJECTED: %s", _cv.violations[:2])
                        cache_valid = False
                except Exception:
                    pass
            if cache_valid:
                _cached_sql = cached.get('sql', '')
                if _cached_sql:
                    _sanitized = self._sanitize_sql(_cached_sql, question)
                    if _sanitized != _cached_sql:
                        logger.info("SemanticCache SQL sanitized for: %s", question[:40])
                        cached['sql'] = _sanitized
                        _rows, _cols, _err = self._execute_sql(_sanitized)
                        if not _err:
                            cached['rows'] = _rows
                            cached['results'] = _rows
                            cached['columns'] = _cols
                cached['latency_ms'] = round((time.time() - t0) * 1000)
                cached['source'] = 'semantic_cache'
                cached['cache_hit'] = True
                logger.info("Cache hit for '%s' in %dms", question[:40], cached['latency_ms'])
                self._update_conversation(
                    session_id, question, cached,
                    cached.get('rows', []), cached.get('columns', [])
                )
                return cached
        logger.debug("Cache lookup took %.1fms", (time.time() - t_cache) * 1000)

        if self.healthcare_llm_pipeline and self.healthcare_llm_pipeline.llm_available:
            try:
                t_llm = time.time()
                llm_result = self.healthcare_llm_pipeline.process(question, session_id=session_id)
                llm_time = (time.time() - t_llm) * 1000

                if llm_result and llm_result.get('rows') and not llm_result.get('error'):
                    rows = llm_result['rows']
                    columns = llm_result.get('columns', [])
                    sql = llm_result.get('sql', '')
                    narrative = llm_result.get('narrative', '')

                    result = {
                        'answer': narrative or f"Found {len(rows)} results.",
                        'sql': sql,
                        'data': rows,
                        'rows': rows,
                        'results': rows,
                        'columns': columns,
                        'row_count': len(rows),
                        'error': None,
                        'source': f'healthcare_llm_pipeline/{llm_result.get("source", "")}',
                        'approach': 'llm_full_pipeline',
                        'cache_hit': False,
                        'latency_ms': llm_time,
                        'confidence': llm_result.get('confidence', 0.9),
                        'narrative': narrative,
                        'reasoning': f"LLM Pipeline: {llm_result.get('source', '')}",
                    }

                    try:
                        self.semantic_cache.put(question, result)
                    except:
                        pass

                    self._update_conversation(session_id, question, result, rows, columns)

                    logger.info("LLM Pipeline: %s | %d rows | %.0fms | source=%s",
                                question[:50], len(rows), llm_time, llm_result.get('source', ''))
                    return result

            except Exception as e:
                logger.warning("Healthcare LLM Pipeline failed, falling through: %s", e)

        if self.semantic_graph and self.intent_parser and self.sql_constructor:
            try:
                t_semantic = time.time()
                if self.conversation_context and self.conversation_context.is_followup(question):
                    logger.info("Follow-up question detected, enriching with context")
                parsed_intent = self.intent_parser.parse(question)
                if self.conversation_context and self.conversation_context.is_followup(question):
                    self.conversation_context.enrich_followup(question, parsed_intent)

                domain_analysis = None
                if self.domain_intelligence:
                    try:
                        domain_analysis = self.domain_intelligence.analyze_question(question)
                        domain_where = domain_analysis.get('sql_where_clause', '')
                        if domain_where and not parsed_intent.filters:
                            from intent_parser import ParsedFilter
                            parsed_intent.filters.append(ParsedFilter(
                                column='_DOMAIN_FILTER_',
                                operator='RAW',
                                value=domain_where,
                                table_hint='',
                                confidence=0.8,
                            ))
                    except Exception:
                        pass

                logger.info("SemanticNLU parsed: intent=%s, tables=%s, confidence=%.2f",
                            parsed_intent.intent, parsed_intent.tables, parsed_intent.confidence)

                analyst_insight = None
                if self.healthcare_analyst:
                    try:
                        intent_dict = {
                            'intent': parsed_intent.intent,
                            'agg_column': parsed_intent.agg_column,
                            'agg_function': parsed_intent.agg_function,
                            'group_by': parsed_intent.group_by,
                        }
                        analyst_insight = self.healthcare_analyst.enrich_intent(intent_dict, question)
                        if analyst_insight.get('analytical_concept'):
                            logger.info("HealthcareAnalyst: concept=%s (%s)",
                                       analyst_insight['analytical_concept'],
                                       analyst_insight.get('concept_description', '')[:60])
                    except Exception as analyst_e:
                        logger.debug("Healthcare analyst enrichment error: %s", analyst_e)

                if parsed_intent.confidence >= 0.30:
                    construct_result = self.sql_constructor.construct(parsed_intent)
                    sem_sql = construct_result.get('sql', '')
                    sem_confidence = construct_result.get('confidence', 0)
                    sem_reasoning = construct_result.get('reasoning', '')

                    logger.info("SemanticSQL constructed: confidence=%.2f, tables=%s, sql_len=%d",
                                sem_confidence, construct_result.get('tables_used', []), len(sem_sql))

                    if sem_sql and sem_confidence >= 0.55:
                        _brain_was_confident = (
                            hasattr(parsed_intent, 'agg_function') and parsed_intent.agg_function
                            and parsed_intent.confidence >= 0.60
                        )
                        if not _brain_was_confident and self.language_brain:
                            _brain_was_confident = parsed_intent.confidence >= 0.60

                        sanitized_sql = self._sanitize_sql(sem_sql, question, _brain_generated=_brain_was_confident)

                        rows, columns, error = self._execute_sql(sanitized_sql)

                        if not error and rows is not None:
                            sem_time_ms = round((time.time() - t0) * 1000)
                            logger.info("SemanticIntelligence SUCCESS in %dms: %s → %d rows",
                                        sem_time_ms, question[:60], len(rows))

                            narrative = ''
                            try:
                                narrative = self.narrative_engine.narrate(
                                    question, sanitized_sql, rows, columns)
                            except Exception:
                                narrative = f"Query returned {len(rows)} result{'s' if len(rows) != 1 else ''}."

                            conf_grade = 'A' if sem_confidence >= 0.8 else ('B' if sem_confidence >= 0.6 else 'C')
                            confidence = {
                                'grade': conf_grade,
                                'overall': sem_confidence,
                                'semantic_nlu': parsed_intent.confidence,
                                'sql_construction': sem_confidence,
                            }

                            result = {
                                'sql': sanitized_sql,
                                'rows': rows,
                                'results': rows,
                                'columns': columns,
                                'row_count': len(rows),
                                'error': None,
                                'source': 'semantic_intelligence',
                                'approach': 'schema_graph_reasoning',
                                'cache_hit': False,
                                'latency_ms': sem_time_ms,
                                'confidence': confidence,
                                'confidence_grade': conf_grade,
                                'confidence_overall': sem_confidence,
                                'narrative': narrative,
                                'reasoning_chain': sem_reasoning,
                                'intent_parsed': {
                                    'intent': parsed_intent.intent,
                                    'sub_intent': parsed_intent.sub_intent,
                                    'tables': parsed_intent.tables,
                                    'columns': parsed_intent.columns,
                                    'filters': [str(f) for f in parsed_intent.filters],
                                    'agg_function': parsed_intent.agg_function,
                                    'group_by': parsed_intent.group_by,
                                    'temporal': parsed_intent.temporal,
                                },
                                'suggestions': [],
                                'anomalies': [],
                                'dashboard': {
                                    'chart_type': 'auto',
                                    'chart_config': {},
                                    'title': question[:80],
                                    'subtitle': f'Semantic Intelligence (confidence: {sem_confidence:.0%})',
                                },
                            }

                            if analyst_insight:
                                result['analyst_insight'] = {
                                    'concept': analyst_insight.get('analytical_concept'),
                                    'description': analyst_insight.get('concept_description'),
                                    'unit': analyst_insight.get('result_unit'),
                                    'business_context': analyst_insight.get('business_context'),
                                    'warnings': analyst_insight.get('analytical_warnings', []),
                                }

                            if self.healthcare_transformer:
                                try:
                                    t_intent, t_conf, t_attn = self.healthcare_transformer.classify_intent(question)
                                    result['transformer_insight'] = {
                                        'intent': t_intent,
                                        'confidence': t_conf,
                                        'attention_focus': t_attn.get('top_attended_positions', []),
                                        'agreement': t_intent == parsed_intent.intent,
                                    }
                                    if t_intent == parsed_intent.intent and t_conf > 0.5:
                                        result['confidence']['ensemble_boost'] = True
                                        result['confidence']['overall'] = min(
                                            result['confidence']['overall'] * 1.1, 1.0
                                        )
                                except Exception:
                                    pass

                            try:
                                result['suggestions'] = self._generate_suggestions(
                                    {'intent': parsed_intent.intent, 'confidence': parsed_intent.confidence},
                                    columns, rows, construct_result, question=question,
                                    session_id=session_id,
                                )
                            except Exception:
                                result['suggestions'] = []

                            try:
                                self.semantic_cache.put(question, result)
                            except Exception:
                                pass
                            if self.answer_cache:
                                try:
                                    self.answer_cache.store(question, result)
                                except Exception:
                                    pass

                            try:
                                if hasattr(self, 'learning') and self.learning:
                                    self.learning.record_interaction(
                                        question, sanitized_sql, rows,
                                        columns, error, sem_confidence)
                            except Exception:
                                pass

                            self._update_conversation(session_id, question, result, rows, columns)

                            if self.conversation_context:
                                try:
                                    self.conversation_context.add_turn(
                                        question, parsed_intent,
                                        sql=sanitized_sql, result=rows
                                    )
                                except Exception:
                                    pass

                            return result
                        else:
                            logger.info("SemanticSQL execution failed (falling through): %s", error)
                    else:
                        logger.info("SemanticSQL low confidence %.2f (falling through to legacy)", sem_confidence)
                else:
                    logger.info("SemanticNLU low confidence %.2f (falling through to legacy)", parsed_intent.confidence)

                logger.debug("Semantic intelligence check took %.1fms", (time.time() - t_semantic) * 1000)
            except Exception as e:
                logger.warning("Semantic intelligence error (falling through to legacy): %s", e)

        logger.warning("LEGACY FALLBACK: Question fell through semantic intelligence to TF-IDF path: %s",
                       question[:100])
        t_intent = time.time()
        intent = self.sql_engine.semantic.classify_intent(question)

        if reasoning_chain_obj:
            try:
                reasoning_chain_obj.add_step(
                    stage='intent_detection',
                    decision=intent.get('intent', 'unknown'),
                    evidence=[f"confidence: {intent.get('confidence', 0):.2f}"],
                    confidence=intent.get('confidence', 0),
                    duration_ms=(time.time() - t_intent) * 1000,
                )
            except Exception as e:
                logger.debug("ReasoningChain intent logging error (non-fatal): %s", e)

        intent_context_tag = None
        user_persona = None
        if self._intent_context_available:
            try:
                from intent_context import classify_intent as ic_classify, detect_persona, anticipate
                intent_context_tag = ic_classify(question)
                if intent_context_tag.confidence > intent.get('confidence', 0):
                    intent['intent_context'] = intent_context_tag.to_dict()
                    intent['secondary_intents'] = intent_context_tag.secondary
                    logger.info("IntentContext: primary=%s (conf=%.2f), secondary=%s",
                                intent_context_tag.primary, intent_context_tag.confidence,
                                intent_context_tag.secondary)
                conv_history = self._get_conversation(session_id)
                recent_qs = [h.get('question', '') for h in conv_history.get('history', [])[-10:]]
                user_persona = detect_persona(None, recent_qs)
                if user_persona:
                    intent['persona'] = user_persona
                    logger.info("Persona detected: %s (lens=%s)",
                                user_persona.get('label', '?'), user_persona.get('lens', '?'))
            except Exception as e:
                logger.debug("IntentContext error (non-fatal): %s", e)

        deep_analysis = None
        if self.deep_query_understanding:
            try:
                conv_history = self._get_conversation(session_id) if session_id else {}
                prev_questions = [h.get('question', '') for h in conv_history.get('history', [])[-3:]]
                prev_q = prev_questions[-1] if prev_questions else None
                deep_analysis = self.deep_query_understanding.analyze(question, context={'previous_question': prev_q})
                if deep_analysis:
                    da = deep_analysis if isinstance(deep_analysis, dict) else (deep_analysis.__dict__ if hasattr(deep_analysis, '__dict__') else {})
                    intent['deep_analysis'] = da
                    sql_hints = self.deep_query_understanding.get_sql_hints(question)
                    if sql_hints:
                        intent['sql_hints'] = sql_hints if isinstance(sql_hints, dict) else (sql_hints.__dict__ if hasattr(sql_hints, '__dict__') else {})
                    _intent_val = da.get('intent', 'unknown')
                    if hasattr(_intent_val, 'value'):
                        _intent_val = _intent_val.value
                    logger.info("DeepQueryUnderstanding: intent=%s, entities=%d",
                                _intent_val,
                                len(da.get('entities', [])))
            except Exception as e:
                logger.debug("DeepQueryUnderstanding error (non-fatal): %s", e)

        is_simple_agg = any(token in question.lower()
                           for token in ['count', 'total', 'sum', 'average', 'how many', 'how much'])
        is_simple_lookup = intent.get('intent') in ('lookup', 'count', 'aggregation')

        memory_recall = []
        if not is_simple_agg:
            try:
                t_hopfield = time.time()
                memory_recall = self.neural.hopfield.recall(
                    self.neural.embeddings.encode_sentence(question), top_k=3
                )
                logger.debug("Hopfield recall took %.1fms", (time.time() - t_hopfield) * 1000)
            except Exception as e:
                logger.debug("Hopfield recall error (non-fatal): %s", e)
                memory_recall = []

        strategy = self.model_selector.select_strategy(
            question, intent, memory_recall
        )
        logger.info("Strategy: %s (%.2f) for '%s' [fast_path=%s]",
                     strategy['strategy'], strategy['confidence'], question[:50], is_simple_agg)
        logger.debug("Intent classification took %.1fms", (time.time() - t_intent) * 1000)

        understanding = {}
        if not is_simple_agg:
            try:
                t_neural = time.time()
                understanding = self.neural.understand(question)
                logger.debug("Neural understanding took %.1fms", (time.time() - t_neural) * 1000)
            except Exception as e:
                logger.debug("Neural understand error (non-fatal): %s", e)
                understanding = {}

        recalled_sql = None
        if strategy['use_hopfield'] and memory_recall:
            for meta, score in memory_recall:
                if score > HOPFIELD_RECALL_THRESHOLD and 'sql' in meta:
                    recalled_sql = meta['sql']
                    logger.info("Hopfield recall: score=%.2f, using cached SQL", score)
                    break

        t_context = time.time()
        conv = self._get_conversation(session_id)
        is_followup = self._is_followup(question, conv)
        effective_question = question
        if is_followup and conv.get('last_question'):
            effective_question = self._rewrite_followup(question, conv)
            logger.info("Follow-up rewritten: '%s'", effective_question)
        logger.debug("Context lookup took %.1fms", (time.time() - t_context) * 1000)

        decompose_steps = []
        if not is_simple_agg and self.decomposer.needs_decomposition(effective_question):
            t_decompose = time.time()
            decompose_steps = self.decomposer.decompose(effective_question)
            logger.info("Decomposed into %d steps in %.1fms", len(decompose_steps),
                       (time.time() - t_decompose) * 1000)

        nlp_enrichment = {}
        if not is_simple_agg and self.advanced_nlp:
            try:
                t_nlp = time.time()
                processed = self.advanced_nlp.process_query(effective_question)
                nlp_enrichment = {
                    'tokens': processed.tokens,
                    'stems': processed.stems,
                    'lemmas': processed.lemmas,
                    'entities': processed.entities,
                    'expanded_terms': list(processed.expanded_terms),
                    'ngrams': processed.ngrams,
                }
                if nlp_enrichment.get('entities'):
                    logger.info("NLP entities: %s", nlp_enrichment['entities'])
                logger.debug("NLP enrichment took %.1fms", (time.time() - t_nlp) * 1000)
            except Exception as e:
                logger.debug("NLP enrichment error (non-fatal): %s", e)

        gpdm_context = {}
        if not is_simple_agg and self.gpdm_definitions:
            try:
                t_gpdm = time.time()
                tables_list = list(self.learner.tables.keys())
                cols_list = []
                for tbl_profiles in self.learner.tables.values():
                    cols_list.extend([p.name for p in tbl_profiles])
                gpdm_context = self.gpdm_definitions.enrich_query(
                    effective_question, tables_list, cols_list
                )
                if gpdm_context.get('confidence', 0) > 0:
                    logger.info("GPDM enrichment: confidence=%.2f", gpdm_context.get('confidence', 0))
                logger.debug("GPDM enrichment took %.1fms", (time.time() - t_gpdm) * 1000)
            except Exception as e:
                logger.debug("GPDM enrichment error (non-fatal): %s", e)

        builtin_understanding = {}
        if not is_simple_agg and self.builtin_nlu:
            try:
                t_builtin = time.time()
                builtin_understanding = self.builtin_nlu.understand(effective_question)

                if (builtin_understanding.get('intent_confidence', 0) > 0.6 and
                        builtin_understanding.get('intent')):
                    bn_intent = builtin_understanding['intent']
                    bn_conf = builtin_understanding['intent_confidence']
                    if bn_conf > intent.get('confidence', 0):
                        logger.info("BuiltInNLU intent override: %s → %s (conf=%.2f)",
                                    intent.get('intent'), bn_intent, bn_conf)
                        intent['intent'] = bn_intent
                        intent['confidence'] = bn_conf
                        intent['source'] = 'builtin_nlu'

                for ent in builtin_understanding.get('entities', []):
                    if ent.get('source') == 'schema_lookup' and ent.get('confidence', 0) > 0.8:
                        ent_tuple = (ent['text'], ent.get('type', 'VALUE'))
                        existing = set(nlp_enrichment.get('entities', []))
                        if ent_tuple not in existing:
                            nlp_enrichment.setdefault('entities', []).append(ent_tuple)

                if builtin_understanding.get('metric'):
                    nlp_enrichment['builtin_metric'] = builtin_understanding['metric']
                if builtin_understanding.get('dimension'):
                    nlp_enrichment['builtin_dimension'] = builtin_understanding['dimension']

                logger.info("BuiltInNLU: intent=%s(%.2f), entities=%d, metric=%s, dim=%s, "
                            "ambig=%s in %dms",
                            builtin_understanding.get('intent', '?'),
                            builtin_understanding.get('intent_confidence', 0),
                            len(builtin_understanding.get('entities', [])),
                            builtin_understanding.get('metric', {}).get('column', 'none'),
                            builtin_understanding.get('dimension', {}).get('column', 'none'),
                            builtin_understanding.get('is_ambiguous', False),
                            builtin_understanding.get('latency_ms', 0))
                logger.debug("BuiltInNLU took %.1fms", (time.time() - t_builtin) * 1000)
            except Exception as e:
                logger.debug("BuiltInNLU error (non-fatal): %s", e)

        retrieval_context = []
        if not is_simple_agg and self.retrieval_index:
            try:
                t_retrieval = time.time()
                raw_hits = self.retrieval_index.search(effective_question, k=10)
                if self.reranker and raw_hits:
                    hit_docs = [h.text for h in raw_hits]
                    hit_scores = [h.score for h in raw_hits]
                    reranked = self.reranker.rerank(
                        effective_question, hit_docs, hit_scores, k=5
                    )
                    retrieval_context = reranked.results if hasattr(reranked, 'results') else raw_hits[:5]
                    logger.info("Retrieval+rerank: %d raw → %d reranked in %.1fms",
                                len(raw_hits), len(retrieval_context),
                                (time.time() - t_retrieval) * 1000)
                else:
                    retrieval_context = raw_hits[:5]
                    logger.debug("Retrieval (no reranker): %d hits in %.1fms",
                                 len(retrieval_context), (time.time() - t_retrieval) * 1000)
                if retrieval_context:
                    nlp_enrichment['retrieval_context'] = [
                        getattr(h, 'text', str(h)) for h in retrieval_context[:5]
                    ]
            except Exception as e:
                logger.debug("Retrieval+rerank error (non-fatal): %s", e)

        grounding = {}
        if not is_simple_agg and self.nl2sql_grounder:
            try:
                t_ground = time.time()
                grounding = self.nl2sql_grounder.ground(effective_question)
                if grounding.get('selected_tables'):
                    nlp_enrichment['grounded_tables'] = grounding['selected_tables']
                    nlp_enrichment['grounded_columns'] = grounding.get('selected_columns', [])
                    nlp_enrichment['grounded_join_path'] = grounding.get('join_path', [])
                    logger.info("NL2SQL grounding: tables=%s, cols=%d, joins=%d in %.1fms",
                                grounding['selected_tables'],
                                len(grounding.get('selected_columns', [])),
                                len(grounding.get('join_path', [])),
                                (time.time() - t_ground) * 1000)
                logger.debug("NL2SQL grounding took %.1fms", (time.time() - t_ground) * 1000)
            except Exception as e:
                logger.debug("NL2SQL grounding error (non-fatal): %s", e)

        deep_understanding = {}
        if not is_simple_agg and self.deep_engine:
            try:
                t_deep = time.time()
                deep_result = self.deep_engine.understand(effective_question)
                deep_understanding = deep_result.to_dict() if hasattr(deep_result, 'to_dict') else {}

                for ent in deep_understanding.get('entities', []):
                    if isinstance(ent, dict):
                        ent_tuple = (ent.get('text', ''), ent.get('type', 'CONCEPT'))
                        existing = set(nlp_enrichment.get('entities', []))
                        if ent_tuple not in existing:
                            nlp_enrichment.setdefault('entities', []).append(ent_tuple)

                if (deep_understanding.get('intent') and
                        deep_understanding.get('intent_confidence', 0) > 0.6):
                    deep_intent = deep_understanding['intent']
                    deep_conf = deep_understanding['intent_confidence']
                    if deep_conf > intent.get('confidence', 0):
                        logger.info("DeepEngine intent override: %s → %s (conf=%.2f)",
                                    intent.get('intent'), deep_intent, deep_conf)
                        intent['intent'] = deep_intent
                        intent['confidence'] = deep_conf
                        intent['source'] = 'deep_understanding'

                if deep_understanding.get('target_tables'):
                    nlp_enrichment['deep_tables'] = deep_understanding['target_tables']
                if deep_understanding.get('target_columns'):
                    nlp_enrichment['deep_columns'] = deep_understanding['target_columns']
                if deep_understanding.get('sql_template'):
                    nlp_enrichment['deep_sql_hint'] = deep_understanding['sql_template']
                if deep_understanding.get('metric_columns'):
                    nlp_enrichment['deep_metrics'] = deep_understanding['metric_columns']
                if deep_understanding.get('dimension_columns'):
                    nlp_enrichment['deep_dimensions'] = deep_understanding['dimension_columns']

                if deep_understanding.get('attention_weights'):
                    nlp_enrichment['attention_weights'] = deep_understanding['attention_weights']

                logger.info("DeepEngine: intent=%s(%.2f), entities=%d, tables=%s, "
                            "cols=%d, sql_hint=%s in %dms",
                            deep_understanding.get('intent', '?'),
                            deep_understanding.get('intent_confidence', 0),
                            len(deep_understanding.get('entities', [])),
                            deep_understanding.get('target_tables', []),
                            len(deep_understanding.get('target_columns', [])),
                            bool(deep_understanding.get('sql_template')),
                            deep_understanding.get('latency_ms', 0))
                logger.debug("DeepEngine took %.1fms", (time.time() - t_deep) * 1000)
            except Exception as e:
                logger.debug("DeepEngine error (non-fatal): %s", e)

        llm_understanding = {}

        pre_validation = None
        if self.reasoning_pre_validator and not is_simple_agg:
            try:
                t_preval = time.time()
                _candidate_tables = []
                _candidate_columns = []
                _agg_type = None
                if deep_understanding:
                    _candidate_tables = deep_understanding.get('target_tables', [])
                    _candidate_columns = [c.get('column', c) if isinstance(c, dict) else c
                                          for c in deep_understanding.get('target_columns', [])]
                if builtin_understanding:
                    if builtin_understanding.get('metric', {}).get('table'):
                        _t = builtin_understanding['metric']['table']
                        if _t not in _candidate_tables:
                            _candidate_tables.append(_t)
                    if builtin_understanding.get('dimension', {}).get('table'):
                        _t = builtin_understanding['dimension']['table']
                        if _t not in _candidate_tables:
                            _candidate_tables.append(_t)
                _intent_name = intent.get('intent', '').lower()
                if 'count' in _intent_name or 'how many' in effective_question.lower():
                    _agg_type = 'COUNT'
                elif 'sum' in _intent_name or 'total' in effective_question.lower():
                    _agg_type = 'SUM'
                elif 'average' in _intent_name or 'avg' in effective_question.lower():
                    _agg_type = 'AVG'

                pre_validation = self.reasoning_pre_validator.validate_query_direction(
                    question=effective_question,
                    detected_intent=intent,
                    candidate_tables=_candidate_tables,
                    candidate_columns=_candidate_columns,
                    aggregation_type=_agg_type,
                )
                if reasoning_chain_obj:
                    reasoning_chain_obj.add_step(
                        stage='pre_validation',
                        decision=f"validated={pre_validation.validated}, confidence={pre_validation.confidence:.2f}",
                        evidence=pre_validation.corrections[:3] if pre_validation.corrections else ['Direction validated'],
                        confidence=pre_validation.confidence,
                        duration_ms=(time.time() - t_preval) * 1000,
                    )
                logger.info("PreValidation: validated=%s, confidence=%.2f, corrections=%d in %.1fms",
                            pre_validation.validated, pre_validation.confidence,
                            len(pre_validation.corrections), (time.time() - t_preval) * 1000)
            except Exception as e:
                logger.debug("ReasoningPreValidator error (non-fatal): %s", e)

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
            sql_result = self.sql_engine.generate(
                effective_question,
                neural_context=understanding,
                nlp_enrichment=nlp_enrichment,
            )
            sql_result['source'] = 'semantic_engine'

            _raw_sql = (sql_result.get('sql', '') or '').upper().strip()
            if 'SELECT *' in _raw_sql and 'LIMIT 50' in _raw_sql:
                import re as _re
                _tbl_match = _re.search(r'FROM\s+(\w+)', _raw_sql)
                _tbl = _tbl_match.group(1).lower() if _tbl_match else 'claims'
                if _tbl.startswith('gpdm_') or _tbl.startswith('_'):
                    _tbl = 'claims'
                q_lower = effective_question.lower()

                if any(kw in q_lower for kw in ['how many', 'count', 'total number', 'number of']):
                    sql_result['sql'] = f"SELECT COUNT(*) AS total_count FROM {_tbl};"
                elif any(kw in q_lower for kw in ['summary', 'overview', 'summarize', 'give me']):
                    try:
                        _cols = [r[1] for r in self._execute_sql(f"PRAGMA table_info({_tbl})")[0]]
                        _numeric = [c for c in _cols if any(k in c.upper() for k in ['AMOUNT', 'COST', 'SCORE', 'COUNT', 'DAYS', 'DURATION', 'SIZE'])]
                        _categorical = [c for c in _cols if any(k in c.upper() for k in ['STATUS', 'TYPE', 'REGION', 'GENDER', 'CLASS', 'CATEGORY'])]
                        parts = [f"COUNT(*) AS total_records"]
                        for nc in _numeric[:3]:
                            parts.append(f"ROUND(AVG(CAST({nc} AS REAL)), 2) AS avg_{nc.lower()}")
                            parts.append(f"ROUND(SUM(CAST({nc} AS REAL)), 2) AS total_{nc.lower()}")
                        sql_result['sql'] = f"SELECT {', '.join(parts)} FROM {_tbl};"
                    except Exception:
                        sql_result['sql'] = f"SELECT COUNT(*) AS total_records FROM {_tbl};"
                elif any(kw in q_lower for kw in ['top', 'most', 'highest', 'best', 'biggest', 'largest']):
                    _grp_col = None
                    _dim_hints = {
                        'specialty': 'SPECIALTY', 'specialties': 'SPECIALTY',
                        'region': 'KP_REGION', 'regions': 'KP_REGION',
                        'provider': 'RENDERING_NPI', 'providers': 'RENDERING_NPI',
                        'facility': 'FACILITY', 'facilities': 'FACILITY',
                        'department': 'DEPARTMENT', 'departments': 'DEPARTMENT',
                        'plan': 'PLAN_TYPE', 'plan type': 'PLAN_TYPE',
                        'diagnosis': 'ICD10_DESCRIPTION', 'diagnoses': 'ICD10_DESCRIPTION',
                        'medication': 'MEDICATION_NAME', 'medications': 'MEDICATION_NAME',
                        'visit type': 'VISIT_TYPE',
                        'member': 'MEMBER_ID', 'members': 'MEMBER_ID',
                        'claim type': 'CLAIM_TYPE',
                    }
                    for hint_word, hint_col in _dim_hints.items():
                        if hint_word in q_lower:
                            try:
                                _tbl_cols = [r[1] for r in self._execute_sql(f"PRAGMA table_info({_tbl})")[0]]
                                if hint_col in _tbl_cols:
                                    _grp_col = hint_col
                                    break
                                for other_tbl in ['providers', 'claims', 'encounters', 'prescriptions', 'members']:
                                    if other_tbl == _tbl:
                                        continue
                                    _other_cols = [r[1] for r in self._execute_sql(f"PRAGMA table_info({other_tbl})")[0]]
                                    if hint_col in _other_cols:
                                        _join_cols = set(_tbl_cols) & set(_other_cols)
                                        if _join_cols:
                                            _jc = next(iter(_join_cols))
                                            _grp_col = f"{other_tbl[0]}.{hint_col}"
                                            _tbl = f"{_tbl} t JOIN {other_tbl} {other_tbl[0]} ON t.{_jc} = {other_tbl[0]}.{_jc}"
                                            break
                            except Exception:
                                pass
                            break
                    if _grp_col:
                        _limit_match = _re.search(r'top\s+(\d+)', q_lower)
                        _limit = int(_limit_match.group(1)) if _limit_match else 10
                        sql_result['sql'] = f"SELECT {_grp_col}, COUNT(*) AS cnt FROM {_tbl} GROUP BY {_grp_col} ORDER BY cnt DESC LIMIT {_limit};"
                    else:
                        sql_result['sql'] = f"SELECT COUNT(*) AS total_records FROM {_tbl.split()[0]};"
                elif 'per' in q_lower or 'each' in q_lower or 'by' in q_lower:
                    sql_result['sql'] = f"SELECT COUNT(*) AS total_records FROM {_tbl};"
                else:
                    sql_result['sql'] = f"SELECT COUNT(*) AS total_records FROM {_tbl};"
                sql_result['source'] = 'semantic_engine/intelligence_rescue'
                logger.info("Intelligence rescue: replaced SELECT * LIMIT 50 with smart summary for table %s", _tbl)

            _sql_check = (sql_result.get('sql', '') or '').upper()
            _q_check = effective_question.lower()
            _is_count_question = any(kw in _q_check for kw in ['how many', 'count of', 'number of', 'total number'])
            _has_count = 'COUNT(' in _sql_check
            _returns_many_rows = 'LIMIT 50' in _sql_check or ('LIMIT' not in _sql_check and 'GROUP BY' not in _sql_check and 'COUNT(' not in _sql_check)
            if _is_count_question and not _has_count and _sql_check and not _sql_check.startswith('--'):
                _inner_sql = sql_result['sql'].rstrip(';').strip()
                if 'SELECT' in _sql_check and 'FROM' in _sql_check:
                    import re as _re2
                    _from_match = _re2.search(r'FROM\s+(.+?)(?:ORDER|LIMIT|$)', _inner_sql, _re2.IGNORECASE | _re2.DOTALL)
                    if _from_match:
                        _from_clause = _from_match.group(1).strip().rstrip(';')
                        sql_result['sql'] = f"SELECT COUNT(*) AS total_count FROM {_from_clause};"
                        sql_result['source'] = sql_result.get('source', '') + '/count_enforced'
                        logger.info("Count enforcement: converted row query to COUNT(*) for 'how many' question")

            if sql_result.get('data_gap'):
                missing = sql_result.get('missing_data', [])
                explanation = sql_result.get('explanation', 'Data not available')
                narrative = (
                    f"Our database does not contain {', '.join(missing)} data. "
                    f"The DISENROLLMENT_DATE field tracks when members leave the plan, but "
                    f"disenrollment reasons (including death) are not recorded. "
                    f"To answer this question, you would need external data sources "
                    f"(e.g., CMS death file linkage or discharge disposition codes)."
                )
                try:
                    _rows, _, _ = self._execute_sql(
                        "SELECT SUM(CASE WHEN DISENROLLMENT_DATE='' THEN 1 ELSE 0 END) AS active, "
                        "SUM(CASE WHEN DISENROLLMENT_DATE!='' THEN 1 ELSE 0 END) AS disenrolled FROM members"
                    )
                    if _rows:
                        narrative += f" What we CAN tell you: we have {_rows[0][0]:,} active members and {_rows[0][1]:,} disenrolled members."
                except Exception:
                    pass
                return {
                    'sql': '', 'rows': [], 'columns': [], 'row_count': 0,
                    'error': None, 'source': 'semantic_engine_data_gap',
                    'narrative': narrative,
                    'confidence': {'grade': 'A', 'overall': 0.95},
                    'confidence_grade': 'A', 'confidence_overall': 0.95,
                    'cache_hit': False,
                    'latency_ms': round((time.time() - t0) * 1000),
                    'suggestions': ['Show disenrollment trends', 'Active member count by region'],
                }

        gen_time = time.time() - t_gen
        logger.info("SQL generation took %.1fms", gen_time * 1000)

        sql_reasoning_result = None
        is_derived_concept = sql_result.get('_derived_concept', False)
        if is_derived_concept:
            logger.info("Skipping SQLReasoner — semantic engine used derived concept: %s",
                        sql_result.get('_derived_concept_type', '?'))
        if not is_simple_agg and self.sql_reasoner and not is_derived_concept:
            try:
                t_reason = time.time()
                sql_reasoning_result = self.sql_reasoner.generate(
                    understanding=deep_understanding if deep_understanding else None,
                    question=effective_question,
                )
                if sql_reasoning_result and hasattr(sql_reasoning_result, 'sql') and sql_reasoning_result.sql:
                    primary_sql = sql_result.get('sql', '')
                    reasoner_confidence = sql_reasoning_result.confidence
                    use_reasoner = False

                    if not primary_sql:
                        use_reasoner = True
                        logger.info("SQLReasoner wins: primary produced no SQL")
                    elif reasoner_confidence >= 0.70:
                        primary_upper = primary_sql.upper()
                        reasoner_upper = sql_reasoning_result.sql.upper()

                        q_lower = effective_question.lower()
                        is_global_query = any(kw in q_lower for kw in [
                            'maximum', 'minimum', 'total', 'all ', 'overall', 'entire',
                            'how many', 'count of', 'count all', 'average ',
                            'highest', 'lowest', 'largest', 'smallest', 'biggest',
                        ])
                        primary_has_filter = 'WHERE' in primary_upper and 'WHERE 1' not in primary_upper
                        reasoner_no_filter = 'WHERE' not in reasoner_upper or 'WHERE 1' in reasoner_upper

                        if is_global_query and primary_has_filter and reasoner_no_filter:
                            use_reasoner = True
                            logger.info("SQLReasoner wins: global query but primary has restrictive WHERE")

                        if getattr(sql_reasoning_result, 'is_derived_metric', False):
                            use_reasoner = True
                            logger.info("SQLReasoner wins: derived metric")

                        if deep_understanding and deep_understanding.get('target_tables'):
                            expected_tables = set(t.lower() for t in deep_understanding['target_tables'])
                            primary_sql_lower = primary_sql.lower()
                            reasoner_sql_lower = sql_reasoning_result.sql.lower()
                            has_wrong_join = False
                            for wrong_tbl in ['prescriptions', 'diagnoses', 'cpt_codes', 'appointments', 'referrals']:
                                if (wrong_tbl not in expected_tables and
                                    f'join {wrong_tbl}' in primary_sql_lower and
                                    f'join {wrong_tbl}' not in reasoner_sql_lower):
                                    has_wrong_join = True
                                    break
                            if has_wrong_join:
                                use_reasoner = True
                                logger.info("SQLReasoner wins: primary has wrong JOIN, reasoner uses correct tables")

                    if use_reasoner:
                        sql_result['sql'] = sql_reasoning_result.sql
                        sql_result['source'] = 'sql_reasoning_engine'
                        sql_result['tables_used'] = sql_reasoning_result.tables_used
                        logger.info("SQLReasoner generated SQL: confidence=%.2f, approach=%s",
                                    sql_reasoning_result.confidence, sql_reasoning_result.approach)
                else:
                    if primary_sql := sql_result.get('sql'):
                        validation = self.sql_reasoner.validate_sql(primary_sql)
                        if hasattr(validation, 'errors') and validation.errors:
                            logger.info("SQLReasoner validation found %d issues: %s",
                                        len(validation.errors), validation.errors[:2])
                            if hasattr(validation, 'repaired_sql') and validation.repaired_sql:
                                try:
                                    test_rows, test_cols, test_err = self._execute_sql(validation.repaired_sql)
                                    if not test_err and test_rows:
                                        sql_result['sql'] = validation.repaired_sql
                                        sql_result['source'] = 'sql_reasoning_repaired'
                                        logger.info("SQLReasoner repair applied successfully")
                                except Exception:
                                    pass
                        sql_result['sql_validation'] = {
                            'valid': hasattr(validation, 'valid') and validation.valid,
                            'errors': getattr(validation, 'errors', []),
                            'warnings': getattr(validation, 'warnings', []),
                        }
                logger.debug("SQLReasoner took %.1fms", (time.time() - t_reason) * 1000)
            except Exception as e:
                logger.debug("SQLReasoner error (non-fatal): %s", e)

        if self.quality_gate and sql_result.get('sql') and not sql_result['sql'].startswith('--') and not sql_result.get('_derived_concept'):
            try:
                t_gate = time.time()
                gate_result = self.quality_gate.validate(
                    question=effective_question,
                    sql=sql_result['sql'],
                    understanding=deep_understanding if deep_understanding else None,
                )
                if not gate_result.passed:
                    logger.info("Quality Gate caught %d violations: %s",
                                len(gate_result.violations), gate_result.violations[:2])
                    if gate_result.corrected_sql != sql_result['sql']:
                        _test_rows, _test_cols, _test_err = self._execute_sql(gate_result.corrected_sql)
                        if _test_rows and not _test_err:
                            logger.info("Quality Gate correction applied: %s",
                                        gate_result.corrections[:2])
                            sql_result['sql'] = gate_result.corrected_sql
                            sql_result['quality_gate'] = {
                                'violations': gate_result.violations,
                                'corrections': gate_result.corrections,
                            }
                            sql_result['source'] = sql_result.get('source', '') + '/quality_gate_corrected'
                        else:
                            logger.debug("Quality Gate correction failed execution — keeping original")
                else:
                    logger.debug("Quality Gate: PASSED")
                    if reasoning_chain_obj:
                        try:
                            reasoning_chain_obj.add_step(
                                stage='quality_gate',
                                decision='Passed validation',
                                evidence=['All checks passed'],
                                confidence=0.95,
                                duration_ms=(time.time() - t_gate) * 1000,
                            )
                        except Exception as e:
                            logger.debug("ReasoningChain QG logging error (non-fatal): %s", e)

                if not gate_result.passed and reasoning_chain_obj:
                    try:
                        reasoning_chain_obj.add_step(
                            stage='quality_gate',
                            decision=f"Corrected: {gate_result.corrections[:1]}",
                            evidence=gate_result.violations[:2],
                            confidence=0.85,
                            duration_ms=(time.time() - t_gate) * 1000,
                        )
                    except Exception as e:
                        logger.debug("ReasoningChain QG error logging (non-fatal): %s", e)

                logger.debug("Quality Gate took %.1fms", (time.time() - t_gate) * 1000)
            except Exception as e:
                logger.debug("Quality Gate error (non-fatal): %s", e)

        _raw_sql = sql_result.get('sql', '')
        if _raw_sql:
            try:
                _sanitized = self._sanitize_sql(_raw_sql, question)
                if _sanitized != _raw_sql:
                    logger.info("SQL sanitizer rewrote query for: %s", question[:60])
                    sql_result['sql'] = _sanitized
                    sql_result['_sanitized'] = True
            except Exception as _san_err:
                logger.warning("SQL sanitizer error (non-fatal): %s", _san_err)

        if self.pre_sql_validator and sql_result.get('sql') and not sql_result['sql'].startswith('--'):
            try:
                t_presql = time.time()
                presql_result = self.pre_sql_validator.validate_and_correct(
                    question=effective_question,
                    sql=sql_result['sql'],
                )
                if presql_result.corrections:
                    _test_rows, _test_cols, _test_err = self._execute_sql(presql_result.corrected_sql)
                    if _test_rows is not None and not _test_err:
                        logger.info("Pre-SQL Validator corrected %d issues: %s",
                                    len(presql_result.corrections), presql_result.violations[:2])
                        sql_result['sql'] = presql_result.corrected_sql
                        sql_result['pre_sql_validation'] = {
                            'violations': presql_result.violations,
                            'corrections': presql_result.corrections,
                        }
                    else:
                        logger.debug("Pre-SQL correction failed execution — keeping original")
                logger.debug("Pre-SQL Validation took %.1fms", (time.time() - t_presql) * 1000)
            except Exception as e:
                logger.debug("Pre-SQL Validation error (non-fatal): %s", e)

        t_exec = time.time()
        rows, columns, error = self._execute_sql(sql_result.get('sql', ''))
        exec_time = time.time() - t_exec
        logger.info("SQL execution took %.1fms", exec_time * 1000)

        if rows and not error and sql_result.get('sql'):
            _sql_upper = sql_result['sql'].upper()
            _has_join = ' JOIN ' in _sql_upper
            _has_count = 'COUNT(*)' in _sql_upper or 'COUNT(C.' in _sql_upper
            _no_distinct = 'COUNT(DISTINCT' not in _sql_upper
            if _has_join and _has_count and _no_distinct:
                _known_limits = {
                    'claims': 60000, 'encounters': 50000, 'members': 25000,
                    'diagnoses': 20000, 'prescriptions': 12000, 'appointments': 10000,
                    'referrals': 5000, 'providers': 3000,
                }
                _primary_table = None
                for _tbl in _known_limits:
                    if f'FROM {_tbl.upper()}' in _sql_upper or f'FROM {_tbl}' in _sql_upper.replace('"', ''):
                        _primary_table = _tbl
                        break
                if _primary_table:
                    _max_expected = _known_limits[_primary_table]
                    for _ri, _row in enumerate(rows):
                        for _ci, _val in enumerate(_row):
                            if isinstance(_val, (int, float)) and _val > _max_expected * 10:
                                _col_name = columns[_ci] if _ci < len(columns) else ''
                                if 'count' in _col_name.lower() or 'record' in _col_name.lower():
                                    logger.warning(
                                        "JOIN inflation detected: %s=%s (max %s expected %s). "
                                        "Trying COUNT(DISTINCT) fix.",
                                        _col_name, _val, _primary_table, _max_expected
                                    )
                                    _id_col = _primary_table.upper().rstrip('S') + '_ID'
                                    if _primary_table == 'claims':
                                        _id_col = 'c.CLAIM_ID'
                                    elif _primary_table == 'encounters':
                                        _id_col = 'e.ENCOUNTER_ID'
                                    elif _primary_table == 'members':
                                        _id_col = 'm.MEMBER_ID'
                                    _fixed_sql = sql_result['sql'].replace(
                                        'COUNT(*)', f'COUNT(DISTINCT {_id_col})'
                                    )
                                    _fix_rows, _fix_cols, _fix_err = self._execute_sql(_fixed_sql)
                                    if _fix_rows and not _fix_err:
                                        rows = _fix_rows
                                        columns = _fix_cols
                                        sql_result['sql'] = _fixed_sql
                                        sql_result['join_inflation_fixed'] = True
                                        logger.info("JOIN inflation fixed with COUNT(DISTINCT %s)", _id_col)
                                    break
                        else:
                            continue
                        break

        if self.quality_gate and rows and not error and sql_result.get('sql') and not sql_result.get('_derived_concept'):
            try:
                post_gate = self.quality_gate.validate_post_execution(
                    question=effective_question,
                    sql=sql_result['sql'],
                    rows=rows,
                    columns=columns,
                    understanding=deep_understanding if deep_understanding else None,
                )
                if not post_gate.passed and post_gate.corrected_sql != sql_result['sql']:
                    _pg_rows, _pg_cols, _pg_err = self._execute_sql(post_gate.corrected_sql)
                    if _pg_rows and not _pg_err:
                        logger.info("Post-execution Quality Gate correction: %s",
                                    post_gate.corrections[:2])
                        rows = _pg_rows
                        columns = _pg_cols
                        sql_result['sql'] = post_gate.corrected_sql
                        sql_result['quality_gate_post'] = {
                            'violations': post_gate.violations,
                            'corrections': post_gate.corrections,
                        }
            except Exception as e:
                logger.debug("Post-execution Quality Gate error (non-fatal): %s", e)

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

        if (error or not rows) and sql_result.get('source') == 'semantic_engine':
            golden_result = self._try_golden_template(question, t0, session_id)
            if golden_result:
                golden_result['source'] = 'golden_template_fallback'
                logger.info("Golden template fallback activated for: %s", question[:60])
                return golden_result

        should_run_deep_analytics = (
            not is_simple_agg and
            rows and not error and
            len(rows) <= 100 and
            intent.get('intent') in ('trend', 'comparison', 'rate', 'analysis', 'anomaly')
        )
        logger.info("Deep analytics gate: %s (intent=%s, rows=%d, error=%s)",
                   should_run_deep_analytics, intent.get('intent'), len(rows) if rows else 0, bool(error))

        t_confidence = time.time()
        confidence = self.confidence_scorer.score(
            question=effective_question,
            sql=sql_result.get('sql', ''),
            rows=rows,
            columns=columns,
            error=error,
            intent=intent,
            strategy=strategy,
            neural_context=understanding,
            nlp_enrichment=nlp_enrichment,
            sql_result=sql_result,
        )
        logger.debug("Confidence scoring took %.1fms", (time.time() - t_confidence) * 1000)

        anomalies = []
        if should_run_deep_analytics:
            try:
                t_anom = time.time()
                anomalies = self.anomaly_detector.detect(rows, columns, effective_question)
                logger.debug("Anomaly detection took %.1fms", (time.time() - t_anom) * 1000)
            except Exception as e:
                logger.debug("Anomaly detection error (non-fatal): %s", e)

        t_clinical = time.time()
        clinical_context = {}
        if self.clinical and strategy.get('use_clinical'):
            try:
                enrichment = self.clinical.enrich(effective_question, sql_result)
                clinical_context = enrichment or {}
            except Exception:
                pass
        logger.debug("Clinical enrichment took %.1fms", (time.time() - t_clinical) * 1000)

        t_cms = time.time()
        cms_context = self._enrich_with_cms(question)
        logger.debug("CMS enrichment took %.1fms", (time.time() - t_cms) * 1000)

        t_bench = time.time()
        benchmark = self.benchmark_engine.benchmark(
            effective_question, rows, columns, intent
        )
        logger.debug("Benchmarking took %.1fms", (time.time() - t_bench) * 1000)

        t_gaps = time.time()
        data_gaps = self.data_gap_detector.detect(
            effective_question, sql_result.get('sql', ''), error, rows, columns
        )
        logger.debug("Data gap detection took %.1fms", (time.time() - t_gaps) * 1000)

        reasoning = {}
        if should_run_deep_analytics:
            try:
                t_reason = time.time()
                reasoning = self.multi_reasoning.reason(
                    effective_question, rows, columns, intent,
                    sql_result.get('sql', ''), anomalies
                )
                logger.debug("Multi-reasoning took %.1fms", (time.time() - t_reason) * 1000)
            except Exception as e:
                logger.debug("Multi-reasoning error (non-fatal): %s", e)

        deep_dives = []
        if should_run_deep_analytics and len(rows) <= 50:
            try:
                t_dives = time.time()
                deep_dives = self.precursor_insights.generate_deep_dives(
                    effective_question, rows[:500], columns
                )
                logger.debug("Deep dives took %.1fms", (time.time() - t_dives) * 1000)
            except Exception as e:
                logger.debug("Deep dive error (non-fatal): %s", e)

        root_causes = []
        if should_run_deep_analytics and anomalies:
            try:
                t_root = time.time()
                root_causes = self.root_cause_analyzer.investigate(
                    anomalies, sql_result.get('sql', ''), effective_question
                )
                logger.debug("Root cause analysis took %.1fms", (time.time() - t_root) * 1000)
            except Exception as e:
                logger.debug("Root cause error (non-fatal): %s", e)

        stat_models = []
        if should_run_deep_analytics and len(rows) <= 50:
            try:
                t_stat = time.time()
                stat_models = self.statistical_analyzer.analyze(
                    effective_question, rows[:500], columns, intent
                )
                if stat_models:
                    logger.info("Statistical models: %s", [m['type'] for m in stat_models])
                logger.debug("Statistical analysis took %.1fms", (time.time() - t_stat) * 1000)
            except Exception as e:
                logger.debug("Statistical analysis failed: %s", e)

        adapter_forecast = None
        _is_forecast_query = any(k in question.lower() for k in [
            'forecast', 'predict', 'projection', 'next quarter', 'next month',
            'next year', 'will be', 'expected', 'trending'
        ])
        if _is_forecast_query and self.forecast_adapter and getattr(self.forecast_adapter, 'available', False):
            try:
                t_adapter = time.time()
                time_cols = [i for i, c in enumerate(columns)
                             if any(t in c.lower() for t in ['month', 'year', 'date', 'period', 'quarter'])]
                value_cols = [i for i, c in enumerate(columns)
                              if isinstance(rows[0][i] if rows and i < len(rows[0]) else None, (int, float))
                              and i not in time_cols]
                if rows and len(rows) >= 4 and value_cols:
                    vc = value_cols[0]
                    values = np.array([float(r[vc]) for r in rows if isinstance(r[vc], (int, float))],
                                     dtype=float)
                    if len(values) >= 4:
                        adapter_forecast = self.forecast_adapter.forecast(values, periods=6)
                        if adapter_forecast:
                            adapter_forecast['target_column'] = columns[vc]
                            adapter_forecast['backend'] = getattr(self.forecast_adapter, 'name', 'unknown')
                            logger.info("ForecastAdapter: %s backend, %d periods, target=%s in %.1fms",
                                        adapter_forecast.get('model', '?'), 6, columns[vc],
                                        (time.time() - t_adapter) * 1000)
            except Exception as e:
                logger.debug("ForecastAdapter error (non-fatal): %s", e)

        t_narrative = time.time()
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
        logger.debug("Narrative generation took %.1fms", (time.time() - t_narrative) * 1000)

        deep_narrative = None
        llm_narrative = None
        if not is_simple_agg and self.narrative_intelligence and rows and not error:
            try:
                t_deep_narr = time.time()
                narr_result = self.narrative_intelligence.generate(
                    question=effective_question,
                    sql=sql_result.get('sql', ''),
                    rows=rows[:50],
                    columns=columns,
                    intent=intent.get('intent', 'unknown'),
                    persona=user_persona.get('label', 'auto') if user_persona else 'auto',
                    context={
                        'deep_understanding': deep_understanding,
                        'benchmark': benchmark,
                        'anomalies': anomalies,
                    },
                )
                if narr_result:
                    deep_narrative = narr_result
                    if hasattr(narr_result, 'narrative') and narr_result.narrative:
                        if len(narr_result.narrative) > len(narrative or ''):
                            narrative = narr_result.narrative
                    if hasattr(narr_result, 'summary'):
                        llm_narrative = narr_result.summary
                    logger.info("NarrativeIntelligence: severity=%s, insights=%d, recs=%d in %dms",
                                getattr(narr_result, 'severity', '?'),
                                len(getattr(narr_result, 'insights', [])),
                                len(getattr(narr_result, 'recommendations', [])),
                                round((time.time() - t_deep_narr) * 1000))
                logger.debug("NarrativeIntelligence took %.1fms", (time.time() - t_deep_narr) * 1000)
            except Exception as e:
                logger.debug("NarrativeIntelligence error (non-fatal): %s", e)

        t_dashboard = time.time()
        dashboard = self.model_selector.select_dashboard(
            sql_result,
            intent=intent.get('intent', ''),
            columns=columns,
            data_sample=rows[:5] if rows else [],
            row_count=len(rows),
        )
        logger.debug("Dashboard selection took %.1fms", (time.time() - t_dashboard) * 1000)

        t_learn = time.time()
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

        self.model_selector.record_outcome(
            strategy['strategy'],
            success=bool(rows and not error),
            latency=gen_time,
        )

        _learn_sql = (sql_result.get('sql', '') or '').strip()
        _learn_sql_upper = _learn_sql.upper()
        _learning_allowed = True
        _learning_rejection_reason = None

        if any(p in _learn_sql.lower() for p in self._INTERNAL_TABLE_PREFIXES):
            _learning_allowed = False
            _learning_rejection_reason = 'internal_table'

        elif 'SELECT *' in _learn_sql_upper:
            _learning_allowed = False
            _learning_rejection_reason = 'select_star'

        elif error:
            _learning_allowed = False
            _learning_rejection_reason = 'sql_error'

        elif not rows:
            _learning_allowed = False
            _learning_rejection_reason = 'no_results'

        elif len(_learn_sql) < 20:
            _learning_allowed = False
            _learning_rejection_reason = 'too_short'

        elif _learn_sql.startswith('--'):
            _learning_allowed = False
            _learning_rejection_reason = 'comment_only'

        elif confidence.get('overall', 0) < 0.4:
            _learning_allowed = False
            _learning_rejection_reason = 'low_confidence'

        if not _learning_allowed:
            logger.info("Learning REJECTED for '%s': %s", effective_question[:50], _learning_rejection_reason)

        if self.sql_reasoner and _learning_allowed and sql_result.get('sql'):
            try:
                self.sql_reasoner.learn(
                    question=effective_question,
                    sql=sql_result['sql'],
                    success=True,
                )
            except Exception as e:
                logger.debug("SQLReasoner learn error (non-fatal): %s", e)

        if self.deep_engine and _learning_allowed:
            try:
                if hasattr(self.deep_engine, 'learn'):
                    self.deep_engine.learn(
                        question=effective_question,
                        intent=intent.get('intent', ''),
                        tables=sql_result.get('tables_used', []),
                        columns=[c for c in columns] if columns else [],
                        sql=sql_result.get('sql', ''),
                    )
            except Exception as e:
                logger.debug("DeepEngine learn error (non-fatal): %s", e)

        logger.debug("Learning & strategy recording took %.1fms", (time.time() - t_learn) * 1000)

        t_conv_update = time.time()
        self._update_conversation(session_id, question, sql_result, rows, columns)
        logger.debug("Conversation update took %.1fms", (time.time() - t_conv_update) * 1000)

        t_suggest = time.time()
        suggestions = self._generate_suggestions(
            intent, columns, rows, sql_result,
            question=question, anomalies=anomalies,
            reasoning=reasoning, benchmark=benchmark,
            session_id=session_id,
        )
        logger.debug("Suggestion generation took %.1fms", (time.time() - t_suggest) * 1000)

        total_time = time.time() - t0

        result = {
            'sql': sql_result.get('sql', ''),
            'rows': rows,
            'columns': columns,
            'row_count': len(rows),
            'error': error,

            'intent': intent.get('intent', ''),
            'intent_confidence': intent.get('confidence', 0),
            'strategy': strategy['strategy'],
            'strategy_confidence': strategy['confidence'],
            'source': sql_result.get('source', 'semantic_engine'),

            'attention_active': strategy.get('use_transformer', False),
            'hopfield_active': strategy.get('use_hopfield', False),
            'gnn_active': strategy.get('use_gnn', False),
            'memory_recalls': len(memory_recall),
            'neural_schema_matches': len(understanding.get('schema_matches', [])),
            'intent_fusion': sql_result.get('intent_fusion'),
            'neural_boosted_columns': sql_result.get('neural_boosted_columns', []),
            'ontology_boosted_columns': sql_result.get('ontology_boosted_columns', []),
            'ner_injected_values': sql_result.get('ner_injected_values', []),

            'explanation_detail': sql_result.get('explanation_detail', {}),

            'builtin_nlu_active': self.builtin_nlu is not None,
            'builtin_nlu': {
                'intent': builtin_understanding.get('intent'),
                'intent_confidence': builtin_understanding.get('intent_confidence', 0),
                'entities_found': len(builtin_understanding.get('entities', [])),
                'metric': builtin_understanding.get('metric'),
                'dimension': builtin_understanding.get('dimension'),
                'is_ambiguous': builtin_understanding.get('is_ambiguous', False),
                'explanation': builtin_understanding.get('explanation', ''),
                'latency_ms': builtin_understanding.get('latency_ms', 0),
            } if builtin_understanding else None,

            'deep_engine_active': self.deep_engine is not None,
            'deep_understanding': {
                'intent': deep_understanding.get('intent'),
                'intent_confidence': deep_understanding.get('intent_confidence', 0),
                'entities_found': len(deep_understanding.get('entities', [])),
                'target_tables': deep_understanding.get('target_tables', []),
                'target_columns': deep_understanding.get('target_columns', []),
                'sql_approach': deep_understanding.get('sql_approach', ''),
                'reasoning_chain': deep_understanding.get('reasoning_chain', []),
                'attention_weights': deep_understanding.get('attention_weights', {}),
                'latency_ms': deep_understanding.get('latency_ms', 0),
            } if deep_understanding else None,
            'sql_reasoner_active': self.sql_reasoner is not None,
            'sql_validation': sql_result.get('sql_validation'),
            'narrative_intelligence_active': self.narrative_intelligence is not None,
            'deep_narrative': {
                'severity': getattr(deep_narrative, 'severity', None),
                'insights': getattr(deep_narrative, 'insights', []),
                'recommendations': getattr(deep_narrative, 'recommendations', []),
                'benchmark_comparison': getattr(deep_narrative, 'benchmark_comparison', {}),
                'persona_used': getattr(deep_narrative, 'persona_used', ''),
            } if deep_narrative else None,
            'llm_active': False,
            'llm_understanding': None,
            'llm_review': None,
            'llm_narrative': llm_narrative,
            'llm_fix_applied': False,

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

            'benchmark': benchmark,
            'data_gaps': data_gaps,
            'reasoning': reasoning,
            'deep_dives': deep_dives,
            'root_causes': root_causes,
            'stat_models': stat_models,
            'adapter_forecast': adapter_forecast,

            'retrieval_hits': len(retrieval_context),
            'grounding': {k: v for k, v in grounding.items()
                          if k in ('selected_tables', 'selected_columns')} if grounding else None,
            'intent_context': intent_context_tag.to_dict() if intent_context_tag else None,
            'persona': user_persona if user_persona else None,

            'clinical_context': clinical_context,
            'cms_context': cms_context,

            'is_followup': is_followup,
            'effective_question': effective_question,

            'nlp_enrichment': nlp_enrichment,

            'gpdm_context': gpdm_context,

            'dashboard': {
                'chart_type': dashboard.chart_type,
                'chart_config': dashboard.chart_config,
                'layout': dashboard.layout,
                'title': dashboard.title,
                'subtitle': dashboard.subtitle,
                'secondary_chart': dashboard.secondary_chart,
                'color_scheme': dashboard.color_scheme,
            },

            'suggestions': suggestions,

            'latency_ms': round(total_time * 1000),
            'generation_ms': round(gen_time * 1000),

            'explanation': narrative,
        }

        ml_trust = {}
        _is_risk_query = any(k in question.lower() for k in [
            'risk', 'predict', 'score', 'stratif', 'likelihood', 'probabilit'
        ])
        if _is_risk_query and rows and not error:
            if self._calibration_available:
                try:
                    from calibration import calibrate_risk_from_sqlite
                    if self.live_cache:
                        cal_result = self.live_cache.get_or_compute(
                            'calibration', (self.db_path, 'isotonic'),
                            lambda: calibrate_risk_from_sqlite(self.db_path, method='isotonic'),
                            db_path=self.db_path,
                            tables=['members', 'encounters'],
                            ttl=600.0,
                        )
                    else:
                        cal_result = calibrate_risk_from_sqlite(self.db_path, method='isotonic')
                    if cal_result:
                        ml_trust['calibration'] = cal_result
                        logger.info("Risk calibration: ECE=%.4f, method=%s",
                                    cal_result.get('ece', -1), cal_result.get('method', '?'))
                except Exception as e:
                    logger.debug("Calibration error (non-fatal): %s", e)

            if self._drift_monitor_available:
                try:
                    from drift_monitor import should_retrain
                    if self.live_cache:
                        drift_result = self.live_cache.get_or_compute(
                            'drift_monitor', (self.db_path,),
                            lambda: should_retrain(self.db_path),
                            db_path=self.db_path,
                            tables=['members', 'encounters', 'claims'],
                            ttl=900.0,
                        )
                    else:
                        drift_result = should_retrain(self.db_path)
                    if drift_result:
                        ml_trust['drift'] = drift_result
                        if drift_result.get('retrain'):
                            logger.warning("DRIFT DETECTED — retrain recommended: %s",
                                          drift_result.get('reasons', []))
                        else:
                            logger.info("Drift check: no retrain needed (PSI=%.4f)",
                                       drift_result.get('max_psi', 0))
                except Exception as e:
                    logger.debug("Drift monitor error (non-fatal): %s", e)

            if self._explainability_available and len(rows) <= 20:
                try:
                    from explainability import permutation_importance
                    numeric_cols = []
                    for ci, col in enumerate(columns):
                        if rows and isinstance(rows[0][ci] if ci < len(rows[0]) else None, (int, float)):
                            numeric_cols.append((ci, col))
                    if len(numeric_cols) >= 2:
                        import numpy as _np
                        X = _np.array([[row[ci] for ci, _ in numeric_cols] for row in rows
                                       if all(isinstance(row[ci] if ci < len(row) else None, (int, float))
                                              for ci, _ in numeric_cols)],
                                      dtype=float)
                        if X.shape[0] >= 5 and X.shape[1] >= 2:
                            feat_names = [col for _, col in numeric_cols]
                            importance = permutation_importance(
                                X[:, :-1], X[:, -1], feat_names[:-1]
                            )
                            ml_trust['explainability'] = {
                                'method': 'permutation_importance',
                                'features': importance,
                                'target_column': feat_names[-1],
                            }
                            logger.info("Explainability: %d features explained", len(importance))
                except Exception as e:
                    logger.debug("Explainability error (non-fatal): %s", e)

        if ml_trust:
            result['ml_trust'] = ml_trust

        if self.kpi_discovery and self.query_tracker:
            try:
                discovered = self.kpi_discovery.discover()
                if discovered:
                    result['discovered_kpis'] = [
                        {'label': k.label, 'support': k.support, 'frequency': k.frequency}
                        for k in discovered[:10]
                    ]
                    logger.debug("KPI Discovery: %d auto-discovered KPIs", len(discovered))
            except Exception as e:
                logger.debug("KPI Discovery error (non-fatal): %s", e)

        if self._executive_kpis and _is_risk_query:
            result['executive_kpis'] = self._executive_kpis

        if self._intent_context_available and intent_context_tag:
            try:
                from intent_context import anticipate
                follow_ups = anticipate(
                    intent_context_tag,
                    user_persona,
                    question,
                    {'row_count': len(rows) if rows else 0, 'columns': columns}
                )
                if follow_ups:
                    anticipated = [f.prompt if hasattr(f, 'prompt') else str(f)
                                   for f in follow_ups[:3]]
                    existing = result.get('suggestions', [])
                    result['suggestions'] = anticipated + [s for s in existing if s not in anticipated]
                    result['anticipated_follow_ups'] = anticipated
                    logger.info("Anticipated %d follow-up questions", len(anticipated))
            except Exception as e:
                logger.debug("Anticipation error (non-fatal): %s", e)

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

        _cache_sql = (result.get('sql', '') or '')
        _cache_allowed = True
        if any(p in _cache_sql.lower() for p in self._INTERNAL_TABLE_PREFIXES):
            _cache_allowed = False
        elif 'SELECT *' in _cache_sql.upper() and 'LIMIT 50' in _cache_sql.upper():
            _cache_allowed = False
        elif _cache_sql.startswith('--') and 'SELECT' not in _cache_sql.upper():
            _cache_allowed = False

        if _cache_allowed:
            self.semantic_cache.put(question, result)
        else:
            logger.info("Semantic cache REJECTED for '%s': bad SQL pattern", question[:50])

        _qg_had_violations = bool(result.get('quality_gate', {}).get('violations') or
                                  result.get('quality_gate_post', {}).get('violations'))
        if self.answer_cache and rows and not error and not _qg_had_violations and _cache_allowed:
            try:
                self.answer_cache.store(question, result)
            except Exception as e:
                logger.debug("AnswerCache store error (non-fatal): %s", e)

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

        if self.cooccurrence_matrix and columns and not error:
            try:
                self.cooccurrence_matrix.record_success(columns)
            except Exception as e:
                logger.debug("Co-occurrence update error (non-fatal): %s", e)

        if self.orchestrator and 'dq_badge' not in result:
            try:
                result = self.orchestrator.enrich_response(question, result)
            except Exception as e:
                logger.debug("DQ enrichment error (non-fatal): %s", e)

        logger.info(
            "PIPELINE SUMMARY [%.0fms total] | SQL gen: %.0fms, exec: %.0fms | "
            "intent=%s, rows=%d, fast_path=%s, deep_analytics=%s, cache=%s",
            total_time * 1000, gen_time * 1000, exec_time * 1000,
            intent.get('intent', '?'), len(rows) if rows else 0,
            is_simple_agg, should_run_deep_analytics, cached and 'HIT' or 'MISS'
        )

        if reasoning_chain_obj:
            try:
                reasoning_chain_obj.add_step(
                    stage='sql_generation',
                    decision=f"Generated: {result.get('sql', '')[:80]}",
                    evidence=[f"source: {result.get('source', '')}",
                              f"strategy: {result.get('strategy', '')}"],
                    confidence=result.get('strategy_confidence', 0.5),
                    duration_ms=result.get('generation_ms', 0),
                )
                reasoning_chain_obj.add_step(
                    stage='execution',
                    decision=f"Returned {result.get('row_count', 0)} rows",
                    evidence=[f"columns: {result.get('columns', [])}"],
                    confidence=result.get('confidence_overall', 0.5),
                )
                result['reasoning_chain'] = reasoning_chain_obj.to_dict()
                result['thought_process'] = reasoning_chain_obj.to_narrative()
            except Exception as e:
                logger.debug("ReasoningChain result attachment error (non-fatal): %s", e)

        if self.learning_scorer:
            try:
                _score = self.learning_scorer.score_query(
                    question=question,
                    intent=intent.get('intent', 'unknown'),
                    sql=result.get('sql', ''),
                    result=result,
                    latency=result.get('latency_ms', 0),
                )
                result['learning_score'] = _score.get('overall_score', 0) if isinstance(_score, dict) else 0
                logger.debug("LearningScorer: score=%.2f", result['learning_score'])
            except Exception as e:
                logger.debug("LearningScorer error (non-fatal): %s", e)

        if self.knowledge_graph:
            try:
                _success = not error and rows and len(rows) > 0
                self.knowledge_graph.learn_pattern(
                    question=question,
                    sql=result.get('sql', ''),
                    success=_success,
                    execution_time=result.get('latency_ms', 0),
                )
            except Exception as e:
                logger.debug("KnowledgeGraph learn error (non-fatal): %s", e)

        if self.self_healing_engine:
            try:
                result = self.self_healing_engine.validate_query_result(question, result)
                fix_log = self.self_healing_engine.get_fix_log()
                if fix_log:
                    result['_self_heal_count'] = len(fix_log)
            except Exception as e:
                logger.debug("Self-healing query validation error (non-fatal): %s", e)

        if self.anticipation_engine:
            try:
                _intent = result.get('intent', result.get('approach', 'general'))
                _kpi = result.get('kpi_area', 'general')
                _quality = result.get('confidence', 0.8)
                if isinstance(_quality, dict):
                    _quality = _quality.get('overall', 0.8)
                self.anticipation_engine.record_interaction(
                    session_id=session_id,
                    query=question,
                    intent=str(_intent),
                    kpi_area=str(_kpi),
                    response_quality=float(_quality) if _quality else 0.8
                )
                _queryCount = len(getattr(self.anticipation_engine, 'session_history', {}).get(session_id, []))
                _ctx = {
                    'current_intent': str(_intent),
                    'current_kpi': str(_kpi),
                    'depth_level': 'early' if _queryCount < 3 else ('mid' if _queryCount < 8 else 'deep'),
                }
                anticipated = self.anticipation_engine.anticipate_next(session_id, current_context=_ctx)
                if anticipated:
                    result['anticipated_questions'] = anticipated[:3]
                    logger.info("Anticipated %d follow-up questions", len(anticipated))
                session_ctx = self.anticipation_engine.get_session_context(session_id)
                if session_ctx:
                    result['session_context'] = {
                        'query_count': session_ctx.get('query_count', 0),
                        'depth_level': session_ctx.get('depth_level', 'overview'),
                        'focus_areas': session_ctx.get('focus_areas', [])
                    }
            except Exception as e:
                logger.debug("Anticipation engine error (non-fatal): %s", e)

        if self.uncertainty_handler and result.get('rows'):
            try:
                uncertainty = self.uncertainty_handler.assess_data_uncertainty(
                    result, result.get('sql', '')
                )
                if uncertainty:
                    result['data_confidence'] = uncertainty.get('overall_confidence', None)
                    if uncertainty.get('warnings'):
                        result['data_warnings'] = uncertainty['warnings'][:3]
                    if uncertainty.get('confidence_intervals'):
                        result['confidence_intervals'] = uncertainty['confidence_intervals']
            except Exception as e:
                logger.debug("Uncertainty handler error (non-fatal): %s", e)

        return result

    _INTERNAL_TABLE_PREFIXES = ('gpdm_', '_gpdm_', '_dq_', '_schema_', '_data_', '_audit_',
                                 'sqlite_', 'query_patterns')

    _TABLE_KEY_COLUMNS = {
        'claims': {'agg': 'PAID_AMOUNT', 'group': 'KP_REGION', 'count': 'CLAIM_ID'},
        'encounters': {'agg': 'LENGTH_OF_STAY', 'group': 'VISIT_TYPE', 'count': 'ENCOUNTER_ID'},
        'members': {'agg': 'RISK_SCORE', 'group': 'KP_REGION', 'count': 'MEMBER_ID'},
        'providers': {'agg': 'PANEL_SIZE', 'group': 'SPECIALTY', 'count': 'NPI'},
        'prescriptions': {'agg': 'COST', 'group': 'MEDICATION_NAME', 'count': 'RX_ID'},
        'referrals': {'agg': 'STATUS', 'group': 'SPECIALTY', 'count': 'REFERRAL_ID'},
        'appointments': {'agg': 'STATUS', 'group': 'DEPARTMENT', 'count': 'APPOINTMENT_ID'},
    }

    def _sanitize_sql(self, sql: str, question: str, _brain_generated: bool = False) -> str:
        import re as _re

        sql_upper = sql.upper()
        q_lower = question.lower()

        try:
            from dynamic_sql_engine import normalize_typos
            q_lower = normalize_typos(q_lower)
        except ImportError:
            pass

        if ('paid' in q_lower and ('amount' in q_lower or 'total' in q_lower or 'top' in q_lower)):
            if 'BILLED_AMOUNT' in sql_upper and 'PAID_AMOUNT' not in sql_upper:
                sql = sql.replace('BILLED_AMOUNT', 'PAID_AMOUNT').replace('billed_amount', 'paid_amount')
                sql_upper = sql.upper()
                logger.info("Sanitizer check 0: replaced BILLED_AMOUNT with PAID_AMOUNT")

        has_internal = False
        for prefix in self._INTERNAL_TABLE_PREFIXES:
            if prefix.upper() in sql_upper or prefix in sql.lower():
                has_internal = True
                break

        if has_internal:
            logger.warning("SQL sanitizer: internal table detected in SQL, rebuilding: %s", sql[:100])
            table_keywords = {
                'claims': ['claim', 'billed', 'paid', 'denied', 'copay', 'coinsurance', 'deductible'],
                'encounters': ['encounter', 'visit', 'admit', 'discharge', 'inpatient', 'emergency', 'telehealth'],
                'members': ['member', 'patient', 'enroll', 'disenroll', 'risk score'],
                'providers': ['provider', 'doctor', 'npi', 'specialty', 'specialit'],
                'prescriptions': ['medication', 'prescription', 'drug', 'pharmacy', 'rx', 'prescribe'],
                'referrals': ['referral', 'refer'],
                'appointments': ['appointment', 'no-show', 'schedule'],
                'diagnoses': ['diagnosis', 'diagnos', 'icd', 'condition'],
            }
            best_table = 'claims'
            best_score = 0
            for tbl, keywords in table_keywords.items():
                score = sum(1 for kw in keywords if kw in q_lower)
                if score > best_score:
                    best_score = score
                    best_table = tbl

            if any(w in q_lower for w in ['how many', 'count', 'total number']):
                return f"SELECT COUNT(*) AS total FROM {best_table}"
            elif any(w in q_lower for w in ['total amount', 'total billed', 'sum of']):
                col = self._TABLE_KEY_COLUMNS.get(best_table, {}).get('agg', '*')
                return f"SELECT ROUND(SUM(CAST({col} AS REAL)), 2) AS total FROM {best_table}"
            elif any(w in q_lower for w in ['average', 'avg', 'mean']):
                col = self._TABLE_KEY_COLUMNS.get(best_table, {}).get('agg', '*')
                return f"SELECT ROUND(AVG(CAST({col} AS REAL)), 2) AS average FROM {best_table}"
            elif any(w in q_lower for w in ['top', 'most', 'highest', 'best']):
                grp = self._TABLE_KEY_COLUMNS.get(best_table, {}).get('group', 'CLAIM_ID')
                return f"SELECT {grp}, COUNT(*) AS cnt FROM {best_table} GROUP BY {grp} ORDER BY cnt DESC LIMIT 10"
            elif 'per region' in q_lower or 'by region' in q_lower:
                if 'KP_REGION' in self._get_table_columns(best_table):
                    return f"SELECT KP_REGION, COUNT(*) AS cnt FROM {best_table} GROUP BY KP_REGION ORDER BY cnt DESC"
            elif 'trend' in q_lower or 'over time' in q_lower or 'per month' in q_lower:
                date_col = 'SERVICE_DATE'
                return f"SELECT SUBSTR({date_col}, 1, 7) AS period, COUNT(*) AS cnt FROM {best_table} WHERE {date_col} IS NOT NULL AND {date_col} != '' GROUP BY period ORDER BY period"
            else:
                grp = self._TABLE_KEY_COLUMNS.get(best_table, {}).get('group', '')
                if grp:
                    return f"SELECT {grp}, COUNT(*) AS cnt FROM {best_table} GROUP BY {grp} ORDER BY cnt DESC LIMIT 20"
                return f"SELECT COUNT(*) AS total FROM {best_table}"

        if not _brain_generated and (('total' in q_lower and 'billed' in q_lower) or 'sum of billed' in q_lower):
            _is_global = 'across all' in q_lower or ('per' not in q_lower and 'by' not in q_lower and 'each' not in q_lower)
            if _is_global:
                return "SELECT ROUND(SUM(CAST(BILLED_AMOUNT AS REAL)), 2) AS total_billed FROM claims"
            elif 'SUM' not in sql_upper and 'BILLED_AMOUNT' not in sql_upper:
                return "SELECT ROUND(SUM(CAST(BILLED_AMOUNT AS REAL)), 2) AS total_billed FROM claims"

        if _brain_generated:
            _group_by_patterns = []
        else:
            _group_by_patterns = [
            (r'per\s+region|by\s+region|each\s+region', 'KP_REGION'),
            (r'per\s+plan\s*type|for\s+each\s+plan\s*type|by\s+plan\s*type', 'PLAN_TYPE'),
            (r'per\s+specialty|by\s+specialty|each\s+specialty', 'SPECIALTY'),
            (r'per\s+facility|by\s+facility|each\s+facility', 'FACILITY'),
            (r'per\s+visit\s*type|by\s+visit\s*type|each\s+visit\s*type', 'VISIT_TYPE'),
            (r'per\s+department|by\s+department|each\s+department', 'DEPARTMENT'),
            (r'per\s+month|by\s+month|each\s+month', None),
        ]
        for pattern, col in _group_by_patterns:
            if _re.search(pattern, q_lower) and col and 'GROUP BY' not in sql_upper:
                _from_match = _re.search(r'FROM\s+(\w+)', sql, _re.IGNORECASE)
                primary = _from_match.group(1) if _from_match else 'claims'
                table_cols = self._get_table_columns(primary)
                if col in table_cols:
                    if 'count' in q_lower or 'how many' in q_lower:
                        return f"SELECT {col}, COUNT(*) AS cnt FROM {primary} GROUP BY {col} ORDER BY cnt DESC"
                    elif 'average' in q_lower or 'avg' in q_lower:
                        agg_col = self._TABLE_KEY_COLUMNS.get(primary, {}).get('agg', '*')
                        for c in table_cols:
                            if c.lower() in q_lower.replace(' ', '_'):
                                agg_col = c
                                break
                        return f"SELECT {col}, ROUND(AVG(CAST({agg_col} AS REAL)), 2) AS avg_{agg_col.lower()} FROM {primary} GROUP BY {col} ORDER BY avg_{agg_col.lower()} DESC"
                    elif 'total' in q_lower or 'sum' in q_lower:
                        agg_col = self._TABLE_KEY_COLUMNS.get(primary, {}).get('agg', '*')
                        return f"SELECT {col}, ROUND(SUM(CAST({agg_col} AS REAL)), 2) AS total_{agg_col.lower()} FROM {primary} GROUP BY {col} ORDER BY total_{agg_col.lower()} DESC"
                    else:
                        return f"SELECT {col}, COUNT(*) AS cnt FROM {primary} GROUP BY {col} ORDER BY cnt DESC"

        if not _brain_generated and ('per month' in q_lower or 'by month' in q_lower or 'each month' in q_lower) and 'GROUP BY' not in sql_upper:
            _from_match = _re.search(r'FROM\s+(\w+)', sql, _re.IGNORECASE)
            primary = _from_match.group(1) if _from_match else 'claims'
            date_col = 'SERVICE_DATE'
            year_filter = ''
            _year_match = _re.search(r'\b(20\d{2})\b', question)
            if _year_match:
                year_filter = f" WHERE {date_col} LIKE '{_year_match.group(1)}%'"
            return f"SELECT SUBSTR({date_col}, 1, 7) AS month, COUNT(*) AS cnt FROM {primary}{year_filter} GROUP BY month ORDER BY month"

        _year_match = _re.search(r'\bin\s+(20\d{2})\b', q_lower)
        if _year_match and _year_match.group(1) not in sql:
            year = _year_match.group(1)
            if 'WHERE' in sql_upper:
                sql = _re.sub(r'(WHERE\s+)', rf'\1SERVICE_DATE LIKE \'{year}%\' AND ', sql, count=1, flags=_re.IGNORECASE)
            else:
                sql = _re.sub(r'(FROM\s+\w+)', rf'\1 WHERE SERVICE_DATE LIKE \'{year}%\'', sql, count=1, flags=_re.IGNORECASE)

        if not _brain_generated and any(kw in q_lower for kw in ['top', 'most', 'highest', 'best', 'biggest']) and 'GROUP BY' not in sql_upper and 'MAX(' not in sql_upper and 'MIN(' not in sql_upper:
            _from_match = _re.search(r'FROM\s+(\w+)', sql, _re.IGNORECASE)
            primary = _from_match.group(1) if _from_match else 'claims'
            table_cols = self._get_table_columns(primary)
            _dim_keywords = {
                'specialty': 'SPECIALTY', 'specialties': 'SPECIALTY',
                'provider': 'RENDERING_NPI', 'region': 'KP_REGION',
                'facility': 'FACILITY', 'department': 'DEPARTMENT',
                'medication': 'MEDICATION_NAME', 'diagnosis': 'ICD10_DESCRIPTION',
                'member': 'MEMBER_ID', 'plan': 'PLAN_TYPE', 'visit': 'VISIT_TYPE',
            }
            grp_col = None
            for kw, col in _dim_keywords.items():
                if kw in q_lower:
                    if col in table_cols:
                        grp_col = col
                        break
                    if col == 'SPECIALTY':
                        prov_cols = self._get_table_columns('providers')
                        if 'SPECIALTY' in prov_cols:
                            if 'RENDERING_NPI' in table_cols or 'NPI' in table_cols:
                                join_col = 'RENDERING_NPI' if 'RENDERING_NPI' in table_cols else 'NPI'
                                _limit_match = _re.search(r'top\s+(\d+)', q_lower)
                                _limit = int(_limit_match.group(1)) if _limit_match else 10
                                return f"SELECT p.SPECIALTY, COUNT(*) AS cnt FROM {primary} c JOIN providers p ON c.{join_col} = p.NPI GROUP BY p.SPECIALTY ORDER BY cnt DESC LIMIT {_limit}"
            if grp_col:
                _limit_match = _re.search(r'top\s+(\d+)', q_lower)
                _limit = int(_limit_match.group(1)) if _limit_match else 10
                if any(kw in q_lower for kw in ['paid', 'amount', 'cost', 'spend', 'billed']):
                    agg_col = None
                    for c in table_cols:
                        if any(k in c.upper() for k in ['PAID', 'BILLED', 'COST', 'AMOUNT']):
                            agg_col = c
                            break
                    if agg_col:
                        return f"SELECT {grp_col}, ROUND(SUM(CAST({agg_col} AS REAL)), 2) AS total FROM {primary} GROUP BY {grp_col} ORDER BY total DESC LIMIT {_limit}"
                return f"SELECT {grp_col}, COUNT(*) AS cnt FROM {primary} GROUP BY {grp_col} ORDER BY cnt DESC LIMIT {_limit}"

        if _re.search(r'\bN\b', sql) and 'COUNT(N' not in sql_upper and 'IN (' not in sql_upper:
            _from_match = _re.search(r'FROM\s+(\w+)', sql, _re.IGNORECASE)
            primary = _from_match.group(1) if _from_match else 'claims'
            table_cols = self._get_table_columns(primary)
            if 'N' not in table_cols:
                if 'trend' in q_lower or 'over time' in q_lower:
                    return f"SELECT SUBSTR(SERVICE_DATE, 1, 7) AS period, COUNT(*) AS cnt FROM {primary} WHERE SERVICE_DATE IS NOT NULL GROUP BY period ORDER BY period"
                else:
                    sql = _re.sub(r'CAST\(N AS REAL\)', '1', sql)
                    sql = _re.sub(r'\bN\b,\s*', '', sql)

        if 'billed' in q_lower and 'BILLED_AMOUNT' not in sql_upper and 'COINSURANCE' in sql_upper:
            sql = sql.replace('COINSURANCE', 'BILLED_AMOUNT').replace('coinsurance', 'billed_amount').replace('sum_coinsurance', 'sum_billed_amount')

        if ('length of stay' in q_lower or 'los' in q_lower.split()) and 'LENGTH_OF_STAY' not in sql_upper:
            if 'encounters' in sql.lower() or 'inpatient' in q_lower or 'visit' in q_lower:
                _filter = ''
                if 'inpatient' in q_lower:
                    _filter = " WHERE VISIT_TYPE = 'INPATIENT'"
                if 'AVG' in sql_upper or 'average' in q_lower or 'avg' in q_lower or 'mean' in q_lower:
                    return f"SELECT ROUND(AVG(CAST(LENGTH_OF_STAY AS REAL)), 2) AS avg_length_of_stay FROM encounters{_filter}"
                elif 'MAX' in sql_upper or 'longest' in q_lower or 'maximum' in q_lower:
                    return f"SELECT MAX(CAST(LENGTH_OF_STAY AS REAL)) AS max_length_of_stay FROM encounters{_filter}"
                elif 'SUM' in sql_upper or 'total' in q_lower:
                    return f"SELECT ROUND(SUM(CAST(LENGTH_OF_STAY AS REAL)), 2) AS total_length_of_stay FROM encounters{_filter}"
                else:
                    return f"SELECT ROUND(AVG(CAST(LENGTH_OF_STAY AS REAL)), 2) AS avg_length_of_stay FROM encounters{_filter}"

        superlative_max = any(w in q_lower for w in ['highest', 'maximum', 'max ', 'biggest', 'largest'])
        superlative_min = any(w in q_lower for w in ['lowest', 'minimum', 'min ', 'smallest', 'least'])
        _ranking_context = any(kw in q_lower for kw in ['caseload', 'case load', 'docs', 'which doc'])
        if not _brain_generated and not _ranking_context and (superlative_max or superlative_min) and 'GROUP BY' not in sql_upper:
            agg_fn = 'MAX' if superlative_max else 'MIN'
            if agg_fn not in sql_upper:
                _from_match = _re.search(r'FROM\s+(\w+)', sql, _re.IGNORECASE)
                primary = _from_match.group(1) if _from_match else 'claims'
                target_col = None
                for word_col in [('paid amount', 'PAID_AMOUNT'), ('paid', 'PAID_AMOUNT'),
                                 ('billed', 'BILLED_AMOUNT'), ('cost', 'PAID_AMOUNT'),
                                 ('risk score', 'RISK_SCORE'), ('risk', 'RISK_SCORE'),
                                 ('length of stay', 'LENGTH_OF_STAY'), ('copay', 'COPAY'),
                                 ('panel size', 'PANEL_SIZE'), ('allowed', 'ALLOWED_AMOUNT'),
                                 ('days supply', 'DAYS_SUPPLY'), ('rvu', 'RVU')]:
                    if word_col[0] in q_lower:
                        target_col = word_col[1]
                        break
                if not target_col:
                    _order_match = _re.search(r'ORDER\s+BY\s+(\w+)', sql, _re.IGNORECASE)
                    if _order_match:
                        target_col = _order_match.group(1)
                if target_col:
                    return f"SELECT {agg_fn}(CAST({target_col} AS REAL)) AS {agg_fn.lower()}_{target_col.lower()} FROM {primary}"

        if ('paid' in q_lower and 'amount' in q_lower) or 'paid amount' in q_lower or 'total paid' in q_lower:
            if 'BILLED_AMOUNT' in sql_upper and 'PAID_AMOUNT' not in sql_upper:
                sql = sql.replace('BILLED_AMOUNT', 'PAID_AMOUNT').replace('billed_amount', 'paid_amount')
                sql = _re.sub(r'total_billed', 'total_paid', sql, flags=_re.IGNORECASE)
                return sql

        if 'hmo' in q_lower and 'HMO' not in sql_upper:
            _from_match = _re.search(r'FROM\s+(\w+)', sql, _re.IGNORECASE)
            primary = _from_match.group(1) if _from_match else 'claims'
            table_cols = self._get_table_columns(primary)
            if 'PLAN_TYPE' in table_cols:
                if 'WHERE' in sql_upper:
                    sql = _re.sub(r'(WHERE\s+)', r"\1PLAN_TYPE = 'HMO' AND ", sql, count=1, flags=_re.IGNORECASE)
                else:
                    sql = _re.sub(r'(FROM\s+\w+)', r"\1 WHERE PLAN_TYPE = 'HMO'", sql, count=1, flags=_re.IGNORECASE)
            else:
                if primary == 'encounters':
                    if 'average' in q_lower or 'avg' in q_lower or 'cost' in q_lower:
                        agg_col = 'PAID_AMOUNT' if 'paid' in q_lower or 'cost' in q_lower else 'BILLED_AMOUNT'
                        return f"SELECT ROUND(AVG(CAST(c.{agg_col} AS REAL)), 2) AS avg_cost FROM encounters e JOIN claims c ON e.ENCOUNTER_ID = c.ENCOUNTER_ID WHERE c.PLAN_TYPE = 'HMO'"
                    else:
                        return f"SELECT COUNT(*) AS total FROM encounters e JOIN claims c ON e.ENCOUNTER_ID = c.ENCOUNTER_ID WHERE c.PLAN_TYPE = 'HMO'"

        if ('denial' in q_lower or 'denied' in q_lower or 'deny' in q_lower) and 'DENIED' not in sql_upper and 'DENIAL' not in sql_upper:
            if 'rate' in q_lower or 'percentage' in q_lower or 'percent' in q_lower:
                if 'region' in q_lower or 'KP_REGION' in sql_upper:
                    return ("SELECT KP_REGION, "
                            "COUNT(*) AS total_claims, "
                            "SUM(CASE WHEN CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) AS denied_claims, "
                            "ROUND(100.0 * SUM(CASE WHEN CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) / COUNT(*), 2) AS denial_rate "
                            "FROM claims GROUP BY KP_REGION ORDER BY denial_rate DESC")
                elif 'highest' in q_lower or 'top' in q_lower or 'most' in q_lower:
                    return ("SELECT KP_REGION, "
                            "ROUND(100.0 * SUM(CASE WHEN CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) / COUNT(*), 2) AS denial_rate "
                            "FROM claims GROUP BY KP_REGION ORDER BY denial_rate DESC LIMIT 1")
                else:
                    return ("SELECT ROUND(100.0 * SUM(CASE WHEN CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) / COUNT(*), 2) AS denial_rate "
                            "FROM claims")
            else:
                if 'WHERE' not in sql_upper:
                    sql = _re.sub(r'(FROM\s+claims)', r"\1 WHERE CLAIM_STATUS = 'DENIED'", sql, count=1, flags=_re.IGNORECASE)

        _region_keywords = {
            'ncal': 'NCAL', 'scal': 'SCAL', 'colorado': 'CO', 'mid-atlantic': 'MAS',
            'northwest': 'NW', 'hawaii': 'HI', 'georgia': 'GA',
        }
        _region_val = None
        for _rk, _rv in _region_keywords.items():
            if _rk in q_lower and _rv not in sql_upper:
                _region_val = _rv
                break
        if _region_val:
            _from_match = _re.search(r'FROM\s+(\w+)', sql, _re.IGNORECASE)
            primary = _from_match.group(1) if _from_match else 'claims'
            table_cols = self._get_table_columns(primary)
            if 'KP_REGION' in table_cols:
                if 'WHERE' in sql_upper:
                    sql = _re.sub(r'(WHERE\s+)', rf"\1KP_REGION = '{_region_val}' AND ", sql, count=1, flags=_re.IGNORECASE)
                else:
                    sql = _re.sub(r'(FROM\s+\w+)', rf"\1 WHERE KP_REGION = '{_region_val}'", sql, count=1, flags=_re.IGNORECASE)
            else:
                if 'everything' in q_lower or 'summary' in q_lower or 'show' in q_lower or 'about' in q_lower:
                    return (f"SELECT COUNT(*) AS total_claims, "
                            f"ROUND(SUM(CAST(PAID_AMOUNT AS REAL)), 2) AS total_paid, "
                            f"ROUND(AVG(CAST(PAID_AMOUNT AS REAL)), 2) AS avg_paid, "
                            f"SUM(CASE WHEN CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) AS denied_claims "
                            f"FROM claims WHERE KP_REGION = '{_region_val}'")

        if ('sick' in q_lower or 'highest risk' in q_lower or 'high risk' in q_lower) and 'RISK_SCORE' not in sql_upper:
            return "SELECT MEMBER_ID, FIRST_NAME, LAST_NAME, RISK_SCORE, CHRONIC_CONDITIONS, KP_REGION FROM members ORDER BY CAST(RISK_SCORE AS REAL) DESC LIMIT 20"

        if any(d in q_lower for d in ['diabetes', 'hypertension', 'asthma', 'copd', 'cancer', 'heart failure', 'disease']) \
                and 'diagnoses' not in sql.lower() and 'ICD10' not in sql_upper:
            _condition_map = {
                'diabetes': 'diabetes', 'hypertension': 'hypertension',
                'asthma': 'asthma', 'copd': 'copd',
                'cancer': 'cancer', 'heart failure': 'heart failure',
            }
            _cond_filter = None
            for cond_name, cond_kw in _condition_map.items():
                if cond_name in q_lower:
                    _cond_filter = cond_kw
                    break
            if _cond_filter:
                if 'how many' in q_lower or 'count' in q_lower or 'number' in q_lower:
                    return f"SELECT COUNT(DISTINCT MEMBER_ID) AS member_count FROM diagnoses WHERE LOWER(ICD10_DESCRIPTION) LIKE '%{_cond_filter}%'"
                else:
                    return f"SELECT MEMBER_ID, ICD10_CODE, ICD10_DESCRIPTION, DIAGNOSIS_DATE FROM diagnoses WHERE LOWER(ICD10_DESCRIPTION) LIKE '%{_cond_filter}%' LIMIT 50"
            else:
                if 'how many' in q_lower or 'count' in q_lower:
                    return "SELECT COUNT(DISTINCT MEMBER_ID) AS member_count FROM diagnoses"

        return sql

    def _get_table_columns(self, table_name: str) -> list:
        try:
            conn = sqlite3.connect(self.db_path)
            cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table_name})").fetchall()]
            conn.close()
            return cols
        except Exception:
            return []

    def _execute_sql(self, sql: str, timeout_seconds: int = 30) -> Tuple[List, List, Optional[str]]:
        if not sql:
            return [], [], "No SQL generated"
        try:
            conn = sqlite3.connect(self.db_path)
            _start = time.time()
            def _check_timeout():
                if time.time() - _start > timeout_seconds:
                    return 1
                return 0
            conn.set_progress_handler(_check_timeout, 10000)

            cursor = conn.execute(sql)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            conn.close()
            return rows, columns, None
        except sqlite3.OperationalError as e:
            if 'interrupt' in str(e).lower():
                logger.warning("SQL execution timed out after %ds: %s...", timeout_seconds, sql[:80])
                return [], [], f"Query timed out after {timeout_seconds}s — try a more specific query"
            return [], [], str(e)
        except Exception as e:
            return [], [], str(e)

    def _enrich_with_cms(self, question: str) -> Dict[str, Any]:
        context = {}
        q_lower = question.lower()

        for condition_id, info in self.cms.get_statistics().items():
            pass

        condition = None
        for term in ['diabetes', 'hypertension', 'heart failure', 'copd',
                     'depression', 'cancer', 'ckd', 'obesity', 'asthma']:
            if term in q_lower:
                condition = self.cms.find_condition(term)
                if condition:
                    context['condition'] = condition
                    break

        for measure_id in ['CDC', 'BCS', 'CBP', 'COL', 'PCR', 'AMM', 'FUH', 'EDU']:
            if measure_id.lower() in q_lower or measure_id in question:
                measure = self.cms.get_quality_measure(measure_id)
                if measure:
                    context['quality_measure'] = measure
                    break

        if 'hedis' in q_lower:
            context['hedis_note'] = 'HEDIS measures available: CDC, BCS, CBP, COL, PCR, AMM, FUH, EDU'

        return context

    def _get_conversation(self, session_id: str) -> Dict:
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
        if not conv.get('last_question'):
            return False
        q = question.lower().strip()
        words = q.split()

        if len(words) <= 3:
            table_names = {t.lower() for t in self.learner.tables}
            table_singulars = {t.lower().rstrip('s') for t in self.learner.tables}
            q_words = set(w.lower() for w in words)
            try:
                from dynamic_sql_engine import normalize_typos
                normalized = normalize_typos(q)
                q_words |= set(normalized.lower().split())
            except ImportError:
                pass
            if q_words & (table_names | table_singulars):
                return False
            return True

        has_pronoun_ref = any(w in q for w in ('this', 'these', 'that', 'them', 'it'))

        has_own_metric = any(w in q for w in ('cost', 'revenue', 'rate', 'count',
                                                'average', 'total', 'volume'))
        has_own_dimension = any(w in q for w in ('by', 'per', 'over time', 'monthly',
                                                   'quarterly', 'by provider', 'by region'))
        if has_own_metric and has_own_dimension and not has_pronoun_ref:
            return False

        if any(s in q for s in ('corresponding', 'same period', 'same time',
                                  'same timeframe', 'comparable')):
            return False

        followup_signals = ['also', 'what about', 'those', 'instead',
                           'but only', 'only', 'just the', 'now show', 'and also',
                           'same but', 'show this', 'relate to', 'how does this',
                           'how do these', 'for these', 'for those']
        if any(s in q for s in followup_signals):
            return True

        if has_pronoun_ref and any(w in q for w in ('relate', 'affect', 'impact',
                                                      'connect', 'correlate', 'compare',
                                                      'consistent', 'across', 'differ',
                                                      'vary', 'break', 'split')):
            return True

        return False

    def _rewrite_followup(self, question: str, conv: Dict) -> str:
        import re
        q = question.lower().strip()
        last = conv.get('last_question', '')

        if q.startswith('by '):
            base = re.sub(r'\bby\s+\w+', '', last).strip()
            return f"{base} {question}"

        by_match = re.search(r'\bby\s+(\w+(?:\s+\w+)?)', q)
        if by_match and any(w in q for w in ('this', 'it', 'that')):
            new_dim = by_match.group(0)
            base = re.sub(r'\bby\s+\w+(?:\s+\w+)?', '', last).strip()
            base = re.sub(r'\b(changed\s+)?over\s+time\b', '', base).strip()
            base = re.sub(r'[?.!]+$', '', base).strip()
            base = re.sub(r'\s+(?:to|the|a|an|is|and|or|for|how|has)\s*$', '', base).strip()
            base = re.sub(r'\s{2,}', ' ', base).strip()
            return f"{base} {new_dim}"

        if q.startswith('only ') or q.startswith('just '):
            filter_term = q.split(' ', 1)[1]
            return f"{filter_term} {last}"

        if 'instead' in q:
            for agg in ['average', 'total', 'count', 'sum', 'maximum', 'minimum']:
                if agg in q:
                    base = re.sub(r'\b(average|total|count|sum|maximum|minimum)\b', agg, last)
                    return base

        across_match = re.search(
            r'(?:consistent|differ|vary|varies?|break.*down)\s+(?:across|by|for)\s+(?:all\s+)?(.+?)(?:\?|$)',
            q
        )
        if across_match and any(w in q for w in ('this', 'these', 'that', 'it')):
            new_dim = across_match.group(1).strip().rstrip('?')
            base = re.sub(r'\bby\s+\w+(?:\s+\w+)?', '', last).strip()
            base = re.sub(r'\b(changed\s+)?over\s+time\b', '', base).strip()
            base = re.sub(r'\b(?:trend|over\s+the\s+last\s+\d+\s+\w+)\b', '', base).strip()
            base = re.sub(r'[?.!]+$', '', base).strip()
            base = re.sub(r'\s{2,}', ' ', base).strip()
            return f"{base} by {new_dim}"

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
                last_tables = conv.get('last_tables', [])
                entity_word = ''
                for tbl in last_tables:
                    tbl_lower = tbl.lower()
                    if tbl_lower in last.lower() or tbl_lower.rstrip('s') in last.lower():
                        entity_word = tbl_lower
                        break
                if not entity_word and last_tables:
                    entity_word = last_tables[0].lower()
                if entity_word:
                    dim_word = entity_word.rstrip('s') if len(entity_word) > 3 else entity_word
                    return f"Show total {new_concept} by {dim_word}"
                else:
                    return f"Show {new_concept}"

        return f"{last} {question}"

    def _update_conversation(self, session_id: str, question: str,
                              sql_result: Dict, rows: List, columns: List):
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

        if len(conv['turns']) > 10:
            conv['turns'] = conv['turns'][-10:]

    def _generate_suggestions(self, intent: Dict, columns: List,
                               rows: List, sql_result: Dict,
                               question: str = '', anomalies: List = None,
                               reasoning: Dict = None, benchmark: Dict = None,
                               session_id: str = 'default') -> List[str]:
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

        if is_rate and len(rows) > 5:
            suggestions.append('Which providers are above the 90th percentile?')
        if is_provider and len(rows) > 1:
            suggestions.append('Compare top 5 vs bottom 5 providers')

        if is_denial:
            if 'reason' not in q and 'why' not in q:
                suggestions.append('What are the most common denial reason codes?')
        if is_readmission:
            if 'diagnosis' not in q:
                suggestions.append('What are the top diagnoses for readmitted patients?')

        if is_provider and 'specialty' not in q:
            suggestions.append('Show this metric by provider specialty')
        if is_provider and 'department' not in q and 'specialty' in q:
            suggestions.append('Break down by department')

        if is_denial or is_readmission or is_cost:
            if 'age' not in q and 'gender' not in q:
                suggestions.append('How does this vary by patient age group?')

        conv = self._get_conversation(session_id)
        prior_turns = conv.get('turns', [])
        if len(prior_turns) >= 2:
            prior_questions = ' '.join(t.get('question', '').lower() for t in prior_turns[:-1])
            prior_had_demographics = any(w in prior_questions for w in ('demographic', 'race', 'ethnicity', 'age', 'gender'))
            prior_had_claims = any(w in prior_questions for w in ('claim', 'denial', 'cost', 'paid', 'billed'))
            prior_had_readmission = any(w in prior_questions for w in ('readmission', 'readmit'))
            prior_had_encounters = any(w in prior_questions for w in ('encounter', 'visit', 'appointment'))
            prior_had_providers = any(w in prior_questions for w in ('provider', 'npi', 'specialty'))

            if prior_had_demographics and not is_member:
                if is_cost:
                    suggestions.append('How does average cost vary by race/ethnicity?')
                elif is_denial:
                    suggestions.append('Show denial rate by age group and gender')
                else:
                    suggestions.append('Break this down by patient demographics')

            if prior_had_claims and not is_cost and not is_denial:
                suggestions.append('How does this relate to claim costs?')

            if prior_had_readmission and not is_readmission:
                suggestions.append('Is there a relationship between this and readmission rates?')

            if prior_had_providers and not is_provider:
                suggestions.append('Show this metric by provider')

        seen = set()
        unique = []
        for s in suggestions:
            s_key = s.lower().strip()
            if s_key not in seen:
                seen.add(s_key)
                unique.append(s)
        return unique[:6]


    def _build_concept_narrative(self, concept_result: Dict) -> str:
        dims = concept_result.get('dimensions', [])
        label = concept_result.get('label', 'Analysis')
        parts = [f"Comprehensive {label}: Analyzed {len(dims)} dimensions in {concept_result.get('execution_time_ms', 0)}ms."]

        for d in dims:
            if d.get('insight') and not d.get('error'):
                parts.append(f"[{d['label']}] {d['insight']}")

        return " ".join(parts)

    def _build_concept_suggestions(self, concept_key: str, question: str) -> List[str]:
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
        suggestions = []
        profiler = self.concept_expander.profiler
        table_lower = table.lower()
        from concept_expander import TABLE_LABELS
        table_label = TABLE_LABELS.get(table_lower, table.replace('_', ' ').title())

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

        if dims:
            top_dim = dims[0]
            col_label = top_dim['name'].replace('_', ' ').lower()
            if top_dim.get('samples'):
                top_val = top_dim['samples'][0]
                suggestions.append(f'Show detailed breakdown for {col_label} = {top_val}')

        if temporals:
            suggestions.append(f'Show {table_label.lower()} trend over the last 12 months')

        metric_labels = {
            'BILLED_AMOUNT': 'billed amount', 'PAID_AMOUNT': 'paid amount',
            'COST': 'cost', 'COPAY': 'copay', 'PANEL_SIZE': 'panel size',
            'RISK_SCORE': 'risk score', 'LENGTH_OF_STAY': 'length of stay',
            'DAYS_SUPPLY': 'days supply', 'ALLOWED_AMOUNT': 'allowed amount',
        }
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

        suggestions.append(f'Which {table_label.lower()} have the highest volume?')

        if table_lower not in ('members',):
            suggestions.append(f'How does {table_label.lower()} vary by patient age group?')

        seen = set()
        unique = []
        for s in suggestions:
            key = s.lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(s)
        return unique[:6]

    def reset_session(self, session_id: str = 'default'):
        if session_id in self._conversations:
            del self._conversations[session_id]
        self.learning.new_session()

    def save_state(self):
        self.learning.save_state()
        self.model_selector._save_performance()
        logger.info("Pipeline state saved")

    def get_system_status(self) -> Dict[str, Any]:
        import sqlite3 as _sq3

        db_info = {}
        if self.db_path:
            try:
                _conn = _sq3.connect(self.db_path)
                _cur = _conn.cursor()
                _cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
                _tbls = [r[0] for r in _cur.fetchall()]
                _tbl_details = []
                _total_rows = 0
                for _t in _tbls:
                    _cur.execute(f'SELECT COUNT(*) FROM [{_t}]')
                    _cnt = _cur.fetchone()[0]
                    _total_rows += _cnt
                    _tbl_details.append({'name': _t, 'rows': _cnt})
                db_size_mb = os.path.getsize(self.db_path) / (1024 * 1024)
                db_info = {
                    'path': os.path.basename(self.db_path),
                    'size_mb': round(db_size_mb, 1),
                    'tables': len(_tbls),
                    'total_rows': _total_rows,
                    'table_details': _tbl_details,
                }
                _conn.close()
            except Exception as _dbe:
                db_info = {'error': str(_dbe)}

        data_sources = []
        _data_dir = os.path.dirname(self.db_path) if self.db_path else ''
        if _data_dir and os.path.isdir(_data_dir):
            for _sub in ['raw', 'raw_4m', 'parquet', 'raw_scale']:
                _sd = os.path.join(_data_dir, _sub)
                if os.path.isdir(_sd):
                    for _f in os.listdir(_sd):
                        _fp = os.path.join(_sd, _f)
                        if os.path.isfile(_fp):
                            data_sources.append({
                                'name': _f,
                                'folder': _sub,
                                'size_mb': round(os.path.getsize(_fp) / (1024 * 1024), 1),
                            })

        nlu_info = {}
        if self.builtin_nlu:
            nlu_info = {
                'active': True,
                'training_examples': getattr(self.builtin_nlu, '_train_count', 0) or
                    len(getattr(self.builtin_nlu, 'intent_classifier', {}).training_data if hasattr(getattr(self.builtin_nlu, 'intent_classifier', None), 'training_data') else []),
                'accuracy': getattr(getattr(self.builtin_nlu, 'intent_classifier', None), 'accuracy', 0),
                'entity_values': len(getattr(getattr(self.builtin_nlu, 'entity_extractor', None), 'entity_index', {})),
                'idf_terms': len(getattr(getattr(self.builtin_nlu, 'embedder', None), 'idf_weights', {})),
                'schema_terms': len(getattr(getattr(self.builtin_nlu, 'embedder', None), 'schema_boost', {})),
            }
        else:
            nlu_info = {'active': False}

        deep_intel_info = {
            'deep_engine': self.deep_engine is not None,
            'sql_reasoner': self.sql_reasoner is not None,
            'narrative_intelligence': self.narrative_intelligence is not None,
            'mode': 'air-gapped (zero network calls)',
        }
        if self.sql_reasoner:
            deep_intel_info['sql_patterns'] = self.sql_reasoner.pattern_count
            deep_intel_info['sql_templates'] = self.sql_reasoner.template_count
        llm_info = {
            'available': False,
            'type': 'none (air-gapped mode — all intelligence in-process)',
            'deep_intelligence': deep_intel_info,
        }

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
            'database': db_info,
            'data_sources': data_sources,
            'builtin_nlu': nlu_info,
            'llm': llm_info,
            'engines_active': {
                'intelligent_pipeline': True,
                'builtin_nlu': self.builtin_nlu is not None,
                'neural_engine': True,
                'semantic_sql': True,
                'concept_expander': self.concept_expander is not None,
                'anomaly_detector': True,
                'self_healer': True,
                'confidence_scorer': True,
                'narrative_engine': True,
                'clinical_context': self.clinical is not None,
                'cms_knowledge': True,
                'benchmark_engine': True,
                'forecast_runner': self.forecast_runner is not None,
                'advanced_nlp': self.advanced_nlp is not None,
                'deep_engine': self.deep_engine is not None,
                'sql_reasoner': self.sql_reasoner is not None,
                'narrative_intelligence': self.narrative_intelligence is not None,
                'gpdm_definitions': self.gpdm_definitions is not None,
                'self_healing_engine': self.self_healing_engine is not None,
                'healthcare_transformer': self.healthcare_transformer is not None,
            },
            'transformer': self.healthcare_transformer.get_model_info() if self.healthcare_transformer else {'active': False},
        }

    def run_self_healing_health_check(self) -> Dict:
        if not self.self_healing_engine or not self.executive_dashboard_engine:
            return {'error': 'Self-healing or dashboard engine not available'}
        return self.self_healing_engine.run_full_health_check(self.executive_dashboard_engine)

if __name__ == '__main__':
    pass
