import re
import math
import sqlite3
import logging
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger('kp.query_intelligence')

_NEURAL_AVAILABLE = False
_semantic_embedder = None
_language_brain = None

def _init_neural():
    global _NEURAL_AVAILABLE, _semantic_embedder, _language_brain
    if _NEURAL_AVAILABLE:
        return True
    try:
        from semantic_embedder import get_embedder
        _semantic_embedder = get_embedder()
        _NEURAL_AVAILABLE = True
        logger.info("Neural intelligence: SemanticEmbedder loaded (dim=%d)", _semantic_embedder.embed_dim)
    except Exception as e:
        logger.warning("SemanticEmbedder not available: %s", e)
    return _NEURAL_AVAILABLE

def _get_embedder():
    _init_neural()
    return _semantic_embedder

def _embed_question(question: str):
    emb = _get_embedder()
    if emb:
        try:
            return emb.embed_question(question.lower())
        except Exception:
            pass
    return None

def _embed_text(text: str):
    return _embed_question(text)


@dataclass
class SemanticConceptTemplate:
    name: str
    signatures: List[str]
    sql_template: str
    query_type: str = 'scalar'
    agg: Optional[str] = None
    confidence: float = 0.85
    min_similarity: float = 0.55
    _centroid: Any = field(default=None, repr=False)

_CONCEPT_TEMPLATES: List[SemanticConceptTemplate] = [
    SemanticConceptTemplate(
        name='pmpm',
        signatures=[
            'per member per month cost', 'pmpm', 'cost per member per month',
            'monthly cost per member', 'spending per member per month',
            'average monthly cost per enrollee', 'per capita monthly cost',
            'cost per head per month',
        ],
        sql_template=(
            "SELECT ROUND(SUM(CAST({claims}.PAID_AMOUNT AS REAL)) / "
            "NULLIF(COUNT(DISTINCT {members}.MEMBER_ID), 0), 2) AS pmpm_cost "
            "FROM {claims} JOIN {members} ON {claims}.MEMBER_ID = {members}.MEMBER_ID"),
        agg='AVG', confidence=0.85,
    ),
    SemanticConceptTemplate(
        name='loss_ratio',
        signatures=[
            'loss ratio', 'paid to billed ratio', 'claims loss ratio',
            'how much of billed gets paid', 'payout ratio',
            'reimbursement to charge ratio', 'medical loss ratio',
        ],
        sql_template=(
            "SELECT ROUND(100.0 * SUM(CAST(PAID_AMOUNT AS REAL)) / "
            "NULLIF(SUM(CAST(BILLED_AMOUNT AS REAL)), 0), 2) AS loss_ratio_pct, "
            "SUM(CAST(PAID_AMOUNT AS REAL)) AS total_paid, "
            "SUM(CAST(BILLED_AMOUNT AS REAL)) AS total_billed "
            "FROM {claims}"),
        agg='SUM', confidence=0.85,
    ),
    SemanticConceptTemplate(
        name='cost_per_member_by_age_group',
        signatures=[
            'cost per member by age group', 'spending by age bracket',
            'average cost across age groups', 'age group cost breakdown',
            'how much does each age group cost', 'cost distribution by age',
        ],
        sql_template=(
            "SELECT CASE "
            "WHEN (julianday('now') - julianday({members}.DATE_OF_BIRTH))/365.25 < 18 THEN 'Under 18' "
            "WHEN (julianday('now') - julianday({members}.DATE_OF_BIRTH))/365.25 < 30 THEN '18-29' "
            "WHEN (julianday('now') - julianday({members}.DATE_OF_BIRTH))/365.25 < 45 THEN '30-44' "
            "WHEN (julianday('now') - julianday({members}.DATE_OF_BIRTH))/365.25 < 65 THEN '45-64' "
            "ELSE '65+' END AS AGE_GROUP, "
            "ROUND(SUM(CAST({claims}.PAID_AMOUNT AS REAL)) / "
            "NULLIF(COUNT(DISTINCT {members}.MEMBER_ID), 0), 2) AS cost_per_member "
            "FROM {claims} JOIN {members} ON {claims}.MEMBER_ID = {members}.MEMBER_ID "
            "WHERE {members}.DATE_OF_BIRTH IS NOT NULL "
            "GROUP BY AGE_GROUP ORDER BY AGE_GROUP"),
        query_type='grouped', agg='AVG', confidence=0.88, min_similarity=0.58,
    ),
    SemanticConceptTemplate(
        name='age_group_distribution',
        signatures=[
            'age group distribution', 'member age distribution', 'age breakdown',
            'how many members in each age group', 'population by age',
            'membership age demographics', 'enrollee age profile',
        ],
        sql_template=(
            "SELECT CASE "
            "WHEN (julianday('now') - julianday(DATE_OF_BIRTH))/365.25 < 18 THEN 'Under 18' "
            "WHEN (julianday('now') - julianday(DATE_OF_BIRTH))/365.25 < 30 THEN '18-29' "
            "WHEN (julianday('now') - julianday(DATE_OF_BIRTH))/365.25 < 45 THEN '30-44' "
            "WHEN (julianday('now') - julianday(DATE_OF_BIRTH))/365.25 < 65 THEN '45-64' "
            "ELSE '65+' END AS AGE_GROUP, "
            "COUNT(*) AS member_count "
            "FROM {members} "
            "WHERE DATE_OF_BIRTH IS NOT NULL "
            "GROUP BY AGE_GROUP ORDER BY AGE_GROUP"),
        query_type='grouped', agg='COUNT', confidence=0.85,
    ),
    SemanticConceptTemplate(
        name='area_utilization',
        signatures=[
            'areas generating utilization', 'region utilization volume',
            'geographic utilization patterns', 'which areas have most encounters',
            'utilization by region', 'activity by geographic area',
            'encounter volume by region', 'visit volume by area',
        ],
        sql_template=(
            "SELECT KP_REGION, COUNT(*) AS encounter_count "
            "FROM {encounters} "
            "GROUP BY KP_REGION ORDER BY encounter_count DESC"),
        query_type='ranked', agg='COUNT', confidence=0.82,
    ),
    SemanticConceptTemplate(
        name='reimbursement_per_encounter',
        signatures=[
            'reimbursement per encounter', 'going rate for reimbursement per encounter',
            'average reimbursement per visit', 'how much per encounter',
            'going rate per encounter', 'payment per encounter',
            'how much are we reimbursing per encounter',
        ],
        sql_template=(
            "SELECT ROUND(AVG(CAST(PAID_AMOUNT AS REAL)), 2) AS avg_paid_amount "
            "FROM {claims} WHERE PAID_AMOUNT IS NOT NULL"),
        agg='AVG', confidence=0.85,
    ),
    SemanticConceptTemplate(
        name='avg_paid_by_specialty',
        signatures=[
            'average paid amount by provider specialty',
            'cost by specialty', 'spending per specialty',
            'how much does each specialty cost', 'specialty cost breakdown',
            'average reimbursement by specialty', 'payment by provider specialty',
        ],
        sql_template=(
            "SELECT {providers}.SPECIALTY, "
            "ROUND(AVG(CAST({claims}.PAID_AMOUNT AS REAL)), 2) AS avg_paid_amount "
            "FROM {claims} "
            "JOIN {providers} ON {claims}.RENDERING_NPI = {providers}.NPI "
            "WHERE {claims}.PAID_AMOUNT IS NOT NULL "
            "GROUP BY {providers}.SPECIALTY ORDER BY avg_paid_amount DESC"),
        query_type='grouped', agg='AVG', confidence=0.88,
    ),
    SemanticConceptTemplate(
        name='physician_headcount',
        signatures=[
            'physician headcount', 'how many doctors', 'total physicians',
            'provider count', 'number of clinicians', 'how many practitioners',
            'total number of doctors we have', 'clinician headcount',
        ],
        sql_template="SELECT COUNT(DISTINCT NPI) AS total_count FROM {providers}",
        query_type='count', agg='COUNT', confidence=0.85,
    ),
    SemanticConceptTemplate(
        name='covered_lives',
        signatures=[
            'covered lives', 'total covered population', 'member population size',
            'how many members are enrolled', 'total enrollees',
            'covered lives population', 'membership count',
            'how many people are covered',
        ],
        sql_template="SELECT COUNT(DISTINCT MEMBER_ID) AS total_count FROM {members}",
        query_type='count', agg='COUNT', confidence=0.85,
    ),
    SemanticConceptTemplate(
        name='cost_by_visit_type',
        signatures=[
            'average cost per encounter by visit type', 'cost by visit type',
            'spending by encounter type', 'visit type cost breakdown',
            'how much does each visit type cost', 'cost per visit type',
        ],
        sql_template=(
            "SELECT {encounters}.VISIT_TYPE, "
            "ROUND(AVG(CAST({claims}.PAID_AMOUNT AS REAL)), 2) AS avg_cost "
            "FROM {claims} JOIN {encounters} ON {claims}.MEMBER_ID = {encounters}.MEMBER_ID "
            "WHERE {claims}.PAID_AMOUNT IS NOT NULL "
            "GROUP BY {encounters}.VISIT_TYPE ORDER BY avg_cost DESC"),
        query_type='grouped', agg='AVG', confidence=0.85,
    ),
    SemanticConceptTemplate(
        name='total_claims_cost',
        signatures=[
            'total cost of claims', 'cumulative spending across claims',
            'overall claim expenditure', 'aggregate claims cost',
            'total claims spending', 'sum of all claim costs',
            'how much have we paid in total for claims',
        ],
        sql_template=(
            "SELECT ROUND(SUM(CAST(PAID_AMOUNT AS REAL)), 2) AS total_cost "
            "FROM {claims}"),
        agg='SUM', confidence=0.85,
    ),
    SemanticConceptTemplate(
        name='unique_providers',
        signatures=[
            'unique providers', 'distinct providers', 'how many different providers',
            'total number of unique providers', 'unique NPI count',
            'number of distinct providers in the network',
        ],
        sql_template="SELECT COUNT(DISTINCT NPI) AS unique_providers FROM {providers}",
        query_type='count', agg='COUNT', confidence=0.88,
    ),
    SemanticConceptTemplate(
        name='claim_status_breakdown',
        signatures=[
            'claim status breakdown', 'breakdown of claims by status',
            'claims grouped by status', 'status distribution of claims',
            'how many claims in each status', 'claim status distribution',
        ],
        sql_template=(
            "SELECT CLAIM_STATUS, COUNT(*) AS count FROM {claims} "
            "GROUP BY CLAIM_STATUS ORDER BY count DESC"),
        query_type='grouped', agg='COUNT', confidence=0.85,
    ),
    SemanticConceptTemplate(
        name='referral_status_breakdown',
        signatures=[
            'referral status breakdown', 'breakdown of referrals by status',
            'referral status distribution', 'referrals grouped by status',
            'how many referrals in each status',
        ],
        sql_template=(
            "SELECT STATUS, COUNT(*) AS count FROM referrals "
            "GROUP BY STATUS ORDER BY count DESC"),
        query_type='grouped', agg='COUNT', confidence=0.85,
    ),
    SemanticConceptTemplate(
        name='denied_claims_count',
        signatures=[
            'denied claims count', 'how many claims are denied',
            'number of denied claims', 'rejected claims total',
            'count of denied claims', 'claim denials count',
        ],
        sql_template=(
            "SELECT COUNT(*) AS denied_count FROM {claims} "
            "WHERE CLAIM_STATUS = 'DENIED'"),
        query_type='count', agg='COUNT', confidence=0.85,
    ),
    SemanticConceptTemplate(
        name='denied_referrals_count',
        signatures=[
            'denied referrals count', 'how many referrals are denied',
            'number of denied referrals', 'rejected referrals total',
        ],
        sql_template=(
            "SELECT COUNT(*) AS denied_count FROM referrals "
            "WHERE STATUS = 'Denied'"),
        query_type='count', agg='COUNT', confidence=0.85,
    ),
    SemanticConceptTemplate(
        name='denial_rate_by_payer',
        signatures=[
            'denial rate by payer', 'payer denial rate', 'rejection rate by plan type',
            'which payer has highest denial rate', 'plan type denial percentage',
            'denial rate by insurance type', 'denial frequency by payer',
        ],
        sql_template=(
            "SELECT PLAN_TYPE AS payer, COUNT(*) AS total_count, "
            "SUM(CASE WHEN CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) AS denied_count, "
            "ROUND(100.0 * SUM(CASE WHEN CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) / "
            "NULLIF(COUNT(*), 0), 2) AS denial_rate_pct "
            "FROM {claims} GROUP BY PLAN_TYPE ORDER BY denial_rate_pct DESC"),
        query_type='rate', confidence=0.88,
    ),
    SemanticConceptTemplate(
        name='breakdown_for_claims',
        signatures=[
            'breakdown for claims', 'claims breakdown', 'show me the claims breakdown',
        ],
        sql_template=(
            "SELECT CLAIM_STATUS, COUNT(*) AS count FROM {claims} "
            "GROUP BY CLAIM_STATUS ORDER BY count DESC"),
        query_type='grouped', agg='COUNT', confidence=0.80,
    ),
    SemanticConceptTemplate(
        name='breakdown_for_referrals',
        signatures=[
            'breakdown for referrals', 'referral breakdown', 'show me the referral breakdown',
        ],
        sql_template=(
            "SELECT STATUS, COUNT(*) AS count FROM referrals "
            "GROUP BY STATUS ORDER BY count DESC"),
        query_type='grouped', agg='COUNT', confidence=0.80,
    ),
]

_concept_centroids_computed = False

def _compute_concept_centroids():
    global _concept_centroids_computed
    if _concept_centroids_computed:
        return
    emb = _get_embedder()
    if not emb:
        return
    for concept in _CONCEPT_TEMPLATES:
        vecs = []
        for sig in concept.signatures:
            v = _embed_question(sig)
            if v:
                vecs.append(v)
        if vecs:
            dim = vecs[0].dim
            avg_data = [0.0] * dim
            for v in vecs:
                for j in range(dim):
                    avg_data[j] += v.data[j]
            n = len(vecs)
            avg_data = [x / n for x in avg_data]
            from semantic_embedder import EmbeddingVector
            concept._centroid = EmbeddingVector(avg_data).normalize()
    _concept_centroids_computed = True
    logger.info("Semantic concept centroids computed for %d concepts", len(_CONCEPT_TEMPLATES))


def _match_concept_semantically(question: str) -> Optional[SemanticConceptTemplate]:
    _compute_concept_centroids()
    q_emb = _embed_question(question)
    if not q_emb:
        return None

    scored = []
    for concept in _CONCEPT_TEMPLATES:
        if concept._centroid is None:
            continue
        sim = q_emb.cosine(concept._centroid)
        scored.append((sim, concept))

    if not scored:
        return None

    scored.sort(key=lambda x: -x[0])
    best_sim, best_concept = scored[0]
    second_sim = scored[1][0] if len(scored) > 1 else 0.0

    margin = best_sim - second_sim
    min_threshold = max(best_concept.min_similarity, 0.92)

    if best_sim >= min_threshold and margin >= 0.04:
        logger.info("Semantic concept match: '%s' → %s (sim=%.3f, margin=%.3f, threshold=%.2f)",
                    question[:60], best_concept.name, best_sim, margin, min_threshold)
        return best_concept
    else:
        if best_sim > 0.5:
            logger.debug("Semantic concept REJECTED: '%s' → %s (sim=%.3f, margin=%.3f < required)",
                        question[:60], best_concept.name, best_sim, margin)
    return None


_language_brain_instance = None
_language_brain_initialized = False

def _get_language_brain(schema_graph=None):
    global _language_brain_instance, _language_brain_initialized
    if _language_brain_initialized:
        return _language_brain_instance
    _language_brain_initialized = True
    try:
        from language_brain import LanguageBrain
        _language_brain_instance = LanguageBrain(schema_graph)
        _language_brain_instance.train()
        logger.info("LanguageBrain loaded for neural intent classification")
    except Exception as e:
        logger.warning("LanguageBrain not available (using regex fallback): %s", e)
    return _language_brain_instance


def _neural_classify_intent(question: str) -> Optional[Tuple[str, float]]:
    brain = _get_language_brain()
    if brain:
        try:
            intent, conf = brain.classify_intent(question)
            return (intent, conf)
        except Exception:
            pass
    return None


def _neural_resolve_agg(question: str) -> Optional[str]:
    brain = _get_language_brain()
    if brain:
        try:
            return brain.resolve_agg_function(question)
        except Exception:
            pass
    return None


def _neural_resolve_column(question: str, table_hint: str = '') -> Optional[Tuple[str, str]]:
    brain = _get_language_brain()
    if brain:
        try:
            return brain.resolve_column(question, table_hint)
        except Exception:
            pass
    return None


_IRREGULAR_STEMS = {
    'diagnoses': 'diagnosis', 'diagnoses': 'diagnosis',
    'children': 'child', 'women': 'woman', 'men': 'man',
    'people': 'person', 'data': 'data', 'criteria': 'criterion',
    'analyses': 'analysis', 'theses': 'thesis',
    'indices': 'index', 'vertices': 'vertex',
    'matrices': 'matrix', 'appendices': 'appendix',
    'died': 'die', 'lying': 'lie', 'tied': 'tie',
    'busiest': 'busy', 'earliest': 'early', 'heaviest': 'heavy',
    'largest': 'large', 'biggest': 'big', 'smallest': 'small',
    'highest': 'high', 'lowest': 'low', 'newest': 'new',
    'oldest': 'old', 'longest': 'long', 'shortest': 'short',
    'fastest': 'fast', 'slowest': 'slow',
}

def _simple_stem(word: str) -> str:
    if not word or len(word) < 3:
        return word
    w = word.lower()
    if w in _IRREGULAR_STEMS:
        return _IRREGULAR_STEMS[w]
    if w.endswith('ies') and len(w) > 4:
        return w[:-3] + 'y'
    if w.endswith(('ses', 'zes', 'xes', 'ches', 'shes')) and len(w) > 4:
        return w[:-2]
    if w.endswith('s') and not w.endswith('ss') and not w.endswith('us') and len(w) > 3:
        return w[:-1]
    if w.endswith('est') and len(w) > 5 and w not in ('arest', 'forest', 'interest'):
        base = w[:-3]
        if base.endswith(base[-1]) and base[-1] not in 'aeiou':
            base = base[:-1]
        return base if len(base) >= 3 else w
    if w.endswith('er') and len(w) > 4 and w not in ('member', 'provider', 'number',
                                                        'order', 'under', 'after',
                                                        'other', 'over', 'never',
                                                        'ever', 'either', 'neither',
                                                        'together', 'rather', 'former',
                                                        'inner', 'outer', 'upper',
                                                        'lower', 'water', 'computer',
                                                        'letter', 'matter', 'center',
                                                        'power', 'paper', 'layer',
                                                        'filter', 'cluster', 'master',
                                                        'encounter', 'cancer', 'answer'):
        base = w[:-2]
        if base.endswith(base[-1]) and base[-1] not in 'aeiou' and len(base) >= 3:
            base = base[:-1]
        return base if len(base) >= 3 else w
    if w.endswith('ing') and len(w) > 5:
        base = w[:-3]
        if base.endswith(base[-1]) and base[-1] not in 'aeiou' and len(base) >= 3:
            base = base[:-1]
        if base.endswith('at') or base.endswith('ut') or base.endswith('it'):
            base = base + 'e'
        return base if len(base) >= 3 else w
    if w.endswith('ed') and len(w) > 4:
        base = w[:-2]
        if base.endswith(base[-1]) and base[-1] not in 'aeiou' and len(base) >= 3:
            base = base[:-1]
        if w.endswith('ied') and len(w) > 4:
            base = w[:-3] + 'y'
        return base if len(base) >= 3 else w
    return w


def _stem_set(tokens: set) -> set:
    result = set(tokens)
    for t in tokens:
        s = _simple_stem(t)
        if s != t:
            result.add(s)
    return result


@dataclass
class ColumnInfo:
    name: str
    table: str
    dtype: str
    is_numeric: bool
    is_categorical: bool
    is_identifier: bool
    is_date: bool
    distinct_count: int
    total_count: int
    sample_values: list
    semantic_tags: Set[str]
    nl_aliases: List[str]
    tokens: Set[str]


@dataclass
class TableInfo:
    name: str
    columns: Dict[str, ColumnInfo]
    row_count: int
    semantic_domain: str
    nl_aliases: List[str]
    tokens: Set[str]


@dataclass
class JoinPath:
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    cardinality: str


class SchemaGraph:

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.tables: Dict[str, TableInfo] = {}
        self.join_paths: List[JoinPath] = []
        self.column_index: Dict[str, List[ColumnInfo]] = defaultdict(list)
        self.token_index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self.value_index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self._build_from_database()

    def _build_from_database(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        all_tables = [r[0] for r in cur.fetchall()]
        core_tables = [t for t in all_tables if not t.startswith('gpdm_')
                       and not t.startswith('sqlite_') and t != 'query_patterns']
        for table_name in core_tables:
            self._build_table(cur, table_name)
        self._discover_join_paths(cur, core_tables)
        self._build_inverted_indexes()
        conn.close()
        logger.info("SchemaGraph: %d tables, %d columns, %d joins",
                     len(self.tables), sum(len(t.columns) for t in self.tables.values()),
                     len(self.join_paths))

    def _build_table(self, cur, table_name: str):
        cur.execute(f'PRAGMA table_info({table_name})')
        col_infos = cur.fetchall()
        cur.execute(f'SELECT COUNT(*) FROM {table_name}')
        row_count = cur.fetchone()[0]
        columns = {}
        for cid, col_name, dtype, notnull, default, pk in col_infos:
            try:
                cur.execute(f'SELECT COUNT(DISTINCT {col_name}) FROM {table_name}')
                distinct = cur.fetchone()[0]
            except Exception:
                distinct = 0
            try:
                cur.execute(f'SELECT DISTINCT {col_name} FROM {table_name} '
                           f'WHERE {col_name} IS NOT NULL AND {col_name} != "" LIMIT 20')
                samples = [r[0] for r in cur.fetchall()]
            except Exception:
                samples = []
            is_numeric = self._detect_numeric(samples, col_name, dtype)
            is_date = self._detect_date(col_name, samples)
            is_categorical = distinct <= 30 and distinct > 0 and not is_date
            is_identifier = (col_name.endswith('_ID') or col_name == 'NPI' or
                           col_name == 'MRN' or col_name.endswith('_NPI') or
                           pk == 1 or col_name.endswith('_NUMBER'))
            semantic_tags = self._infer_semantic_tags(col_name, table_name, is_numeric, samples)
            nl_aliases = self._generate_nl_aliases(col_name, table_name, semantic_tags)
            tokens = self._tokenize_for_search(col_name, nl_aliases, table_name)
            col_info = ColumnInfo(
                name=col_name, table=table_name, dtype=dtype,
                is_numeric=is_numeric, is_categorical=is_categorical,
                is_identifier=is_identifier, is_date=is_date,
                distinct_count=distinct, total_count=row_count,
                sample_values=samples[:10], semantic_tags=semantic_tags,
                nl_aliases=nl_aliases, tokens=tokens,
            )
            columns[col_name] = col_info
        domain = self._infer_table_domain(table_name, columns)
        tbl_aliases = self._generate_table_aliases(table_name)
        tbl_tokens = set()
        for alias in tbl_aliases:
            tbl_tokens.update(self._tokenize(alias))
        self.tables[table_name] = TableInfo(
            name=table_name, columns=columns, row_count=row_count,
            semantic_domain=domain, nl_aliases=tbl_aliases, tokens=tbl_tokens,
        )

    def _detect_numeric(self, samples, col_name, dtype):
        if dtype in ('INTEGER', 'REAL', 'NUMERIC', 'FLOAT'):
            return True
        numeric_patterns = ['AMOUNT', 'COST', 'COPAY', 'COINSURANCE', 'DEDUCTIBLE',
                          'PAID', 'BILLED', 'ALLOWED', 'PRICE', 'CHARGE', 'FEE',
                          'RATE', 'SCORE', 'SIZE', 'COUNT', 'LENGTH', 'DURATION',
                          'QUANTITY', 'SUPPLY', 'REFILL', 'RVU', 'RISK', 'PANEL',
                          'RESPONSIBILITY', 'CHRONIC_CONDITIONS', 'DURATION_MINUTES']
        if any(p in col_name.upper() for p in numeric_patterns):
            return True
        if col_name.upper() == 'DATE_OF_BIRTH':
            return False
        if samples:
            numeric_count = sum(1 for s in samples[:10]
                              if self._is_number(s))
            if numeric_count >= len(samples[:10]) * 0.7:
                return True
        return False

    @staticmethod
    def _is_number(val):
        try:
            float(str(val).replace(',', ''))
            return True
        except (ValueError, TypeError):
            return False

    def _detect_date(self, col_name, samples):
        date_patterns = ['DATE', 'TIME', 'ADMIT', 'DISCHARGE', 'ENROLL', 'HIRE']
        if any(p in col_name.upper() for p in date_patterns):
            return True
        if samples:
            return any(re.match(r'^\d{4}-\d{2}-\d{2}', str(s)) for s in samples[:5])
        return False

    def _infer_semantic_tags(self, col_name, table_name, is_numeric, samples):
        tags = set()
        cn = col_name.upper()
        financial = ['AMOUNT', 'COST', 'COPAY', 'COINSURANCE', 'DEDUCTIBLE',
                     'PAID', 'BILLED', 'ALLOWED', 'PRICE', 'CHARGE', 'FEE',
                     'MEMBER_RESPONSIBILITY', 'RVU']
        if any(f in cn for f in financial):
            tags.update(['financial', 'aggregatable', 'cost'])
        if any(c in cn for c in ['ICD', 'DIAGNOS', 'HCC', 'CHRONIC', 'SEVERITY',
                                   'MEDICATION', 'NDC', 'CPT']):
            tags.add('clinical')
        if any(c in cn for c in ['GENDER', 'RACE', 'LANGUAGE', 'AGE', 'BIRTH',
                                   'ADDRESS', 'CITY', 'STATE', 'ZIP', 'REGION']):
            tags.add('demographic')
        if any(c in cn for c in ['STATUS', 'TYPE', 'CLASS', 'CATEGORY', 'PLAN',
                                   'SPECIALTY', 'DEPARTMENT', 'FACILITY',
                                   'REGION', 'VISIT', 'URGENCY']):
            tags.add('groupable')
        if cn.endswith('_ID') or cn == 'NPI' or cn == 'MRN' or cn.endswith('_NPI'):
            tags.update(['identifier', 'countable'])
        if 'RISK' in cn or 'SCORE' in cn:
            tags.update(['risk', 'aggregatable'])
        if 'SUPPLY' in cn or 'DURATION' in cn or 'LENGTH' in cn or 'STAY' in cn:
            tags.update(['duration', 'aggregatable'])
        if 'PANEL' in cn or 'SIZE' in cn or 'QUANTITY' in cn:
            tags.update(['size', 'aggregatable'])
        if is_numeric:
            tags.add('numeric')
        return tags

    def _generate_nl_aliases(self, col_name, table_name, semantic_tags):
        aliases = []
        cn_lower = col_name.lower()
        tn_lower = table_name.lower()
        parts = cn_lower.split('_')
        aliases.append(' '.join(parts))
        aliases.append(cn_lower)

        _COLUMN_ALIASES = {
            'PAID_AMOUNT': ['paid amount', 'payment', 'reimbursement', 'paid', 'claim cost',
                           'claim payment', 'amount paid', 'payout', 'reimbursed',
                           'pay out', 'reimburse', 'reimbursing', 'paying out',
                           'settled amount', 'disbursement', 'spending', 'spend',
                           'pmpm', 'per member per month'],
            'BILLED_AMOUNT': ['billed amount', 'billed charge', 'charge', 'billed',
                             'total charge', 'amount billed', 'charges billed',
                             'submitted amount', 'submission'],
            'ALLOWED_AMOUNT': ['allowed amount', 'contracted rate', 'allowed',
                              'negotiated rate', 'approved amount', 'going rate'],
            'COST': ['cost', 'price', 'expense', 'spend', 'spending',
                     'expenditure', 'outlay'],
            'COPAY': ['copay', 'copayment', 'co-pay', 'out of pocket',
                     'expected copayment', 'patient copay'],
            'COINSURANCE': ['coinsurance', 'co-insurance', 'member share'],
            'DEDUCTIBLE': ['deductible', 'deductible amount'],
            'MEMBER_RESPONSIBILITY': ['member responsibility', 'patient responsibility',
                                     'out of pocket', 'patient cost'],
            'RISK_SCORE': ['risk score', 'risk level', 'acuity', 'hcc score',
                          'health risk', 'complexity'],
            'PANEL_SIZE': ['panel size', 'patient panel', 'panel count',
                          'patient load', 'panel', 'panels'],
            'LENGTH_OF_STAY': ['length of stay', 'los', 'stay duration',
                              'hospital stay', 'days in hospital'],
            'DAYS_SUPPLY': ['days supply', 'day supply', 'supply days',
                           'prescription duration', 'medication duration'],
            'MEDICATION_NAME': ['medication', 'medication name', 'drug', 'drug name',
                               'medicine', 'pharmaceutical', 'rx'],
            'MEDICATION_CLASS': ['medication class', 'drug class', 'therapeutic class',
                                'drug category'],
            'ICD10_CODE': ['icd code', 'icd10', 'diagnosis code'],
            'ICD10_DESCRIPTION': ['diagnosis', 'diagnosis description', 'condition',
                                 'disease', 'medical condition'],
            'HCC_CODE': ['hcc code', 'hcc'],
            'HCC_CATEGORY': ['hcc category', 'condition category'],
            'CPT_CODE': ['cpt code', 'cpt', 'procedure code'],
            'CPT_DESCRIPTION': ['procedure', 'procedure description'],
            'CLAIM_STATUS': ['claim status', 'status'],
            'DENIAL_REASON': ['denial reason', 'rejection reason', 'reason for denial',
                             'why denied'],
            'PLAN_TYPE': ['plan type', 'insurance plan', 'plan', 'coverage type',
                         'health plan', 'payer', 'insurance type', 'payer type'],
            'VISIT_TYPE': ['visit type', 'encounter type', 'care setting'],
            'SPECIALTY': ['specialty', 'specialties', 'medical specialty',
                         'provider specialty'],
            'DEPARTMENT': ['department', 'clinical department'],
            'FACILITY': ['facility', 'medical center', 'hospital', 'clinic', 'location'],
            'KP_REGION': ['region', 'kp region', 'geographic region', 'area', 'areas',
                         'geography', 'market'],
            'GENDER': ['gender', 'sex'],
            'RACE': ['race', 'ethnicity'],
            'LANGUAGE': ['language', 'spoken language'],
            'NPI': ['npi', 'provider id', 'provider number', 'national provider identifier'],
            'MEMBER_ID': ['member id', 'member', 'patient id'],
            'ENCOUNTER_ID': ['encounter id', 'encounter', 'visit id'],
            'CLAIM_ID': ['claim id', 'claim number'],
            'QUANTITY': ['quantity', 'prescription quantity'],
            'REFILLS_USED': ['refills', 'refills used', 'refill count'],
            'PHARMACY': ['pharmacy', 'drugstore'],
            'NDC_CODE': ['ndc', 'ndc code', 'national drug code'],
            'RVU': ['rvu', 'relative value unit', 'work rvu', 'rvus'],
            'CHRONIC_CONDITIONS': ['chronic conditions', 'comorbidities',
                                  'chronic disease count'],
            'DISPOSITION': ['disposition', 'discharge disposition'],
            'CHIEF_COMPLAINT': ['chief complaint', 'reason for visit'],
            'DIAGNOSIS_TYPE': ['diagnosis type'],
            'IS_CHRONIC': ['chronic', 'is chronic'],
            'SEVERITY': ['severity', 'severity level'],
            'REFERRAL_REASON': ['referral reason'],
            'URGENCY': ['urgency', 'urgency level', 'priority'],
            'APPOINTMENT_TYPE': ['appointment type'],
            'DURATION_MINUTES': ['duration', 'appointment duration', 'minutes',
                                'visit duration'],
            'IS_PCP_VISIT': ['pcp visit', 'primary care visit'],
            'PROVIDER_TYPE': ['provider type', 'credential', 'degree'],
            'ACCEPTS_NEW_PATIENTS': ['accepts new patients'],
        }

        if col_name in _COLUMN_ALIASES:
            aliases.extend(_COLUMN_ALIASES[col_name])

        if 'cost' in semantic_tags or 'financial' in semantic_tags:
            table_short = tn_lower.rstrip('s')
            for alias in list(aliases):
                if alias in ('cost', 'price', 'expense', 'spend', 'copay'):
                    aliases.append(f'{table_short} {alias}')

        return list(set(aliases))

    def _generate_table_aliases(self, table_name):
        _TABLE_ALIASES = {
            'CLAIMS': ['claims', 'claim', 'billing', 'claim data', 'insurance claims',
                      'submission', 'submissions'],
            'DIAGNOSES': ['diagnoses', 'diagnosis', 'conditions', 'diagnosed',
                         'medical conditions', 'diseases'],
            'ENCOUNTERS': ['encounters', 'encounter', 'visits', 'visit',
                          'patient visits', 'admissions'],
            'MEMBERS': ['members', 'member', 'patients', 'patient', 'enrollees',
                       'beneficiaries', 'population', 'membership',
                       'covered lives', 'people', 'individuals', 'lives'],
            'PRESCRIPTIONS': ['prescriptions', 'prescription', 'medications',
                            'medication', 'drugs', 'drug', 'rx', 'pharmacy',
                            'pharmaceutical', 'prescribed', 'dispensed',
                            'scripts', 'script', 'fills', 'filled'],
            'PROVIDERS': ['providers', 'provider', 'doctors', 'doctor', 'docs',
                         'physicians', 'physician', 'clinicians', 'practitioners',
                         'npi', 'npis', 'provider file', 'headcount'],
            'REFERRALS': ['referrals', 'referral', 'referred', 'authorization',
                        'authorizations', 'auth', 'auths'],
            'appointments': ['appointments', 'appointment', 'scheduling',
                           'no-show', 'no show'],
            'cpt_codes': ['cpt codes', 'cpt', 'procedures', 'procedure codes'],
        }
        if table_name in _TABLE_ALIASES:
            return _TABLE_ALIASES[table_name]
        for key, aliases in _TABLE_ALIASES.items():
            if key.lower() == table_name.lower():
                return aliases
        return [table_name.lower()]

    def _infer_table_domain(self, table_name, columns):
        tn = table_name.lower()
        domains = {'claims': 'financial', 'diagnoses': 'clinical',
                   'encounters': 'operational', 'members': 'demographic',
                   'prescriptions': 'pharmaceutical', 'providers': 'provider',
                   'referrals': 'referral'}
        return domains.get(tn, 'other')

    def _tokenize(self, text):
        text = re.sub(r'[^a-z0-9\s]', ' ', text.lower())
        tokens = set(text.split())
        tokens.discard('')
        return _stem_set(tokens)

    def _tokenize_for_search(self, col_name, aliases, table_name):
        tokens = set()
        tokens.update(self._tokenize(col_name))
        for alias in aliases:
            tokens.update(self._tokenize(alias))
        tokens.update(self._tokenize(table_name))
        return _stem_set(tokens)

    def _discover_join_paths(self, cur, tables):
        col_tables = defaultdict(list)
        for table in tables:
            cur.execute(f'PRAGMA table_info({table})')
            for row in cur.fetchall():
                col_tables[row[1]].append(table)
        for col_name, tbls in col_tables.items():
            if len(tbls) > 1:
                for i in range(len(tbls)):
                    for j in range(i + 1, len(tbls)):
                        self.join_paths.append(JoinPath(
                            from_table=tbls[i], from_column=col_name,
                            to_table=tbls[j], to_column=col_name,
                            cardinality='many-to-many'))
        npi_cols = []
        for table in tables:
            if table in self.tables:
                for col in self.tables[table].columns.values():
                    if 'NPI' in col.name.upper():
                        npi_cols.append((table, col.name))
        for i in range(len(npi_cols)):
            for j in range(i + 1, len(npi_cols)):
                t1, c1 = npi_cols[i]
                t2, c2 = npi_cols[j]
                if t1 != t2:
                    self.join_paths.append(JoinPath(
                        from_table=t1, from_column=c1,
                        to_table=t2, to_column=c2,
                        cardinality='many-to-one' if c2 == 'NPI' else 'many-to-many'))

    def _build_inverted_indexes(self):
        for table_name, table_info in self.tables.items():
            for col_name, col_info in table_info.columns.items():
                for token in col_info.tokens:
                    self.token_index[token].append((table_name, col_name))
                self.column_index[col_name].append(col_info)
                for val in col_info.sample_values:
                    val_lower = str(val).lower()
                    for token in val_lower.split():
                        token = re.sub(r'[^a-z0-9]', '', token)
                        if len(token) >= 3:
                            self.value_index[token].append((table_name, col_name))

    def find_join_path(self, table1, table2):
        t1, t2 = table1.upper(), table2.upper()
        provider_involved = 'PROVIDERS' in (t1, t2)
        if provider_involved:
            for jp in self.join_paths:
                if ((jp.from_table.upper() == t1 and jp.to_table.upper() == t2) or
                    (jp.from_table.upper() == t2 and jp.to_table.upper() == t1)):
                    if 'NPI' in jp.from_column.upper() or 'NPI' in jp.to_column.upper():
                        return jp
        for jp in self.join_paths:
            if ((jp.from_table.upper() == t1 and jp.to_table.upper() == t2) or
                (jp.from_table.upper() == t2 and jp.to_table.upper() == t1)):
                if jp.from_column in ('MEMBER_ID', 'ENCOUNTER_ID'):
                    return jp
        for jp in self.join_paths:
            if ((jp.from_table.upper() == t1 and jp.to_table.upper() == t2) or
                (jp.from_table.upper() == t2 and jp.to_table.upper() == t1)):
                return jp
        return None


@dataclass
class LinkedSchema:
    tables: List[Tuple[str, float]]
    columns: List[Tuple[str, str, float, str]]
    target_column: Optional[Tuple[str, str]] = None
    group_column: Optional[Tuple[str, str]] = None
    filters: List[Dict[str, Any]] = field(default_factory=list)
    joins: List[JoinPath] = field(default_factory=list)


class SchemaLinker:
    def __init__(self, schema: SchemaGraph):
        self.schema = schema
        self._compute_idf()

    def _compute_idf(self):
        total_docs = sum(len(t.columns) for t in self.schema.tables.values())
        self.idf = {}
        for token, entries in self.schema.token_index.items():
            doc_freq = len(set((t, c) for t, c in entries))
            self.idf[token] = math.log((total_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1)

    def link(self, question: str) -> LinkedSchema:
        q_lower = question.lower()
        q_tokens = self._tokenize_question(q_lower)
        table_scores = self._score_tables(q_lower, q_tokens)
        column_scores = self._score_columns(q_lower, q_tokens, table_scores)
        target = self._resolve_target_column(q_lower, q_tokens, column_scores, table_scores)
        group = self._resolve_group_column(q_lower, q_tokens, column_scores, table_scores, target)
        filters = self._detect_filters(q_lower, q_tokens, table_scores)
        needed_tables = set()
        if target:
            needed_tables.add(target[0])
        if group:
            needed_tables.add(group[0])
        for f in filters:
            if 'table' in f:
                needed_tables.add(f['table'])
        joins = self._resolve_joins(needed_tables)

        for f in filters:
            ft = f.get('table')
            if ft and ft not in needed_tables:
                needed_tables.add(ft)
                joins = self._resolve_joins(needed_tables)

        return LinkedSchema(
            tables=sorted(table_scores.items(), key=lambda x: -x[1])[:5],
            columns=sorted(column_scores, key=lambda x: -x[2])[:15],
            target_column=target, group_column=group,
            filters=filters, joins=joins)

    def _tokenize_question(self, q_lower):
        stop_words = {'what', 'is', 'the', 'a', 'an', 'of', 'in', 'for', 'to',
                      'and', 'or', 'are', 'how', 'many', 'much', 'our', 'we',
                      'do', 'does', 'did', 'can', 'could', 'would', 'should',
                      'me', 'my', 'show', 'tell', 'give', 'get', 'find',
                      'i', 'by', 'with', 'from', 'at', 'on', 'this', 'that',
                      'which', 'who', 'where', 'when', 'why', 'there',
                      'has', 'have', 'had', 'been', 'be', 'was', 'were',
                      'it', 'its', 'they', 'them', 'their', 'some', 'all',
                      'each', 'every', 'no', 'not', 'but', 'so', 'if', 'than',
                      'very', 'just', 'about', 'over', 'under', 'between'}
        q_clean = re.sub(r'[^a-z0-9\s]', ' ', q_lower)
        raw_tokens = set(q_clean.split()) - stop_words - {''}
        return _stem_set(raw_tokens)

    def _score_tables(self, q_lower, q_tokens):
        scores = {}
        for table_name, table_info in self.schema.tables.items():
            score = 0.0
            for alias in table_info.nl_aliases:
                if alias in q_lower:
                    score += len(alias) * 2.0
            overlap = q_tokens & table_info.tokens
            score += len(overlap) * 1.0
            scores[table_name] = max(score, 0.01)
        return scores

    def _score_columns(self, q_lower, q_tokens, table_scores):
        results = []
        q_embedding = _embed_question(q_lower) if _NEURAL_AVAILABLE else None

        for table_name, table_info in self.schema.tables.items():
            table_relevance = table_scores.get(table_name, 0.01)
            for col_name, col_info in table_info.columns.items():
                score = 0.0
                reason = ''
                best_alias_score = 0
                for alias in col_info.nl_aliases:
                    if alias in q_lower:
                        if len(alias) > best_alias_score:
                            best_alias_score = len(alias)
                            reason = f'alias "{alias}"'
                score += best_alias_score * 3.0
                for token in q_tokens:
                    if token in col_info.tokens:
                        score += self.idf.get(token, 1.0)
                for token in q_tokens:
                    if token in self.schema.value_index:
                        for vt, vc in self.schema.value_index[token]:
                            if vt == table_name and vc == col_name:
                                score += 5.0
                                reason = f'value "{token}"'
                if table_relevance > 1.0:
                    score *= (1.0 + min(table_relevance / 10.0, 0.5))
                score += self._semantic_tag_score(q_lower, col_info.semantic_tags)

                if q_embedding and score > 0.0:
                    try:
                        col_desc = col_name.replace('_', ' ').lower()
                        if col_info.nl_aliases:
                            col_desc += ' ' + ' '.join(col_info.nl_aliases[:3])
                        col_emb = _embed_text(col_desc)
                        if col_emb:
                            neural_sim = q_embedding.cosine(col_emb)
                            if neural_sim > 0.4:
                                neural_boost = neural_sim * 3.0
                                score += neural_boost
                                if neural_sim > 0.6 and not reason:
                                    reason = f'neural_sim={neural_sim:.2f}'
                    except Exception:
                        pass

                if score > 0.1:
                    results.append((table_name, col_name, score, reason or 'token'))
        return results

    def _semantic_tag_score(self, q_lower, tags):
        score = 0.0
        fin_words = ['cost', 'spend', 'expense', 'price', 'paid', 'billed',
                    'allowed', 'charge', 'reimburs', 'payment', 'revenue',
                    'dollar', 'amount', 'copay', 'deductible', 'coinsurance',
                    'expensive', 'cheap']
        if any(w in q_lower for w in fin_words) and 'financial' in tags:
            score += 2.0
        clin_words = ['diagnos', 'condition', 'disease', 'icd', 'hcc', 'chronic',
                     'clinical', 'medical', 'illness']
        if any(w in q_lower for w in clin_words) and 'clinical' in tags:
            score += 2.0
        count_words = ['how many', 'count', 'number of', 'total number']
        if any(w in q_lower for w in count_words) and 'identifier' in tags:
            score += 1.5
        agg_words = ['average', 'avg', 'mean', 'total', 'sum', 'highest',
                    'lowest', 'maximum', 'minimum', 'most', 'least']
        if any(w in q_lower for w in agg_words) and 'aggregatable' in tags:
            score += 1.5
        return score

    def _resolve_target_column(self, q_lower, q_tokens, column_scores, table_scores):
        if not column_scores:
            neural_col = _neural_resolve_column(q_lower)
            if neural_col:
                logger.debug("Neural column fallback: %s → %s.%s", q_lower[:40], neural_col[0], neural_col[1])
                return neural_col
            return None
        entity_context = {t for t, s in table_scores.items() if s > 1.0}
        candidates = [(t, c, s, r) for t, c, s, r in column_scores if s > 0.5]
        if not candidates:
            neural_col = _neural_resolve_column(q_lower)
            if neural_col:
                logger.debug("Neural column fallback (low BM25): %s → %s.%s", q_lower[:40], neural_col[0], neural_col[1])
                return neural_col
            return None

        fin_words = ['cost', 'spend', 'expense', 'price', 'paid', 'billed', 'allowed',
                     'charge', 'reimburs', 'payment', 'copay', 'deductible', 'coinsurance',
                     'expensive', 'cheap', 'amount', 'revenue']
        is_financial_q = any(w in q_lower for w in fin_words)
        if is_financial_q:
            boosted = []
            for t, c, s, r in candidates:
                ci = self.schema.tables[t].columns[c]
                if 'financial' in ci.semantic_tags or 'cost' in ci.semantic_tags:
                    boosted.append((t, c, s * 2.0, r))
                else:
                    boosted.append((t, c, s, r))
            candidates = boosted

        id_words = ['member id', 'claim id', 'encounter id', 'npi number', 'npi',
                    'npis', 'distinct npi', 'unique npi', 'provider id']
        if not any(w in q_lower for w in id_words):
            non_id = [(t, c, s, r) for t, c, s, r in candidates
                      if not self.schema.tables[t].columns[c].is_identifier]
            if non_id:
                candidates = non_id
        date_words = ['when', 'date', 'time', 'month', 'year', 'quarter', 'trend', 'over time']
        if not any(w in q_lower for w in date_words):
            non_date = [(t, c, s, r) for t, c, s, r in candidates
                        if not self.schema.tables[t].columns[c].is_date]
            if non_date:
                candidates = non_date
        reranked = []
        for t, c, s, r in candidates:
            eff = s
            if entity_context and t in entity_context:
                eff *= 1.5
            elif entity_context and t not in entity_context:
                eff *= 0.7
            reranked.append((t, c, eff, r))
        reranked.sort(key=lambda x: -x[2])

        count_words = ['how many', 'count', 'number of', 'total number', 'unique', 'distinct']
        if any(w in q_lower for w in count_words):
            entity_in_match = re.search(
                r'(?:unique|distinct)\s+(\w+)\s+(?:have|in|with|from|across)', q_lower)
            if entity_in_match:
                entity = entity_in_match.group(1).lower()
                entity_stem = _simple_stem(entity)
                entity_to_id = {
                    'member': 'MEMBER_ID', 'patient': 'MEMBER_ID',
                    'provider': 'NPI', 'doctor': 'NPI',
                    'claim': 'CLAIM_ID', 'encounter': 'ENCOUNTER_ID',
                    'prescription': 'RX_ID',
                }
                target_col = entity_to_id.get(entity_stem) or entity_to_id.get(entity)
                if target_col:
                    for t, s in sorted(table_scores.items(), key=lambda x: -x[1]):
                        tinfo = self.schema.tables.get(t)
                        if tinfo and target_col in tinfo.columns:
                            return (t, target_col)

            if entity_context:
                for t in sorted(entity_context, key=lambda x: -table_scores.get(x, 0)):
                    tinfo = self.schema.tables.get(t)
                    if tinfo:
                        primary_id = t.rstrip('S') + '_ID'
                        if primary_id in tinfo.columns:
                            return (t, primary_id)
                        for col in tinfo.columns.values():
                            if col.is_identifier and col.name.endswith('_ID'):
                                return (t, col.name)
            for t, c, s, r in reranked:
                col_info = self.schema.tables[t].columns[c]
                if col_info.is_identifier and s > 1.0:
                    return (t, c)
        return (reranked[0][0], reranked[0][1]) if reranked else None

    def _resolve_group_column(self, q_lower, q_tokens, column_scores,
                               table_scores, target):
        group_entity = None
        by_match = re.search(r'\bby\s+(\w[\w\s]{0,30}?)(?:\s*\?|$|,|\s+(?:and|or|from|in|with))', q_lower)
        if by_match:
            candidate = by_match.group(1).strip()
            metric_phrases = {'count', 'claim count', 'cost', 'amount', 'paid amount',
                            'billed amount', 'total', 'volume', 'frequency', 'number',
                            'spend', 'spending', 'payment', 'revenue'}
            if candidate not in metric_phrases:
                group_entity = candidate
        per_match = re.search(r'\bper\s+(\w[\w\s]{0,20}?)(?:\s*\?|$|,)', q_lower)
        if per_match and not group_entity:
            per_entity = per_match.group(1).strip()
            unit_entities = {'encounter', 'claim', 'visit', 'member', 'patient',
                           'prescription', 'admission', 'episode', 'capita'}
            per_stem = _simple_stem(per_entity.split()[0])
            if per_stem not in unit_entities and per_entity.split()[0] not in unit_entities:
                group_entity = per_entity
        breakdown_match = re.search(r'broken\s+(?:down|out)\s+by\s+(\w[\w\s]{0,30}?)(?:\s*\?|$)', q_lower)
        if breakdown_match and not group_entity:
            group_entity = breakdown_match.group(1).strip()
        across_match = re.search(r'\bacross\s+(?:different\s+)?(\w[\w\s]{0,30}?)(?:\s*\?|$|,)', q_lower)
        if across_match and not group_entity:
            candidate = across_match.group(1).strip()
            candidate = re.sub(r'\s*groups?\s*$', '', candidate)
            if candidate:
                group_entity = candidate
        breakdown_of_match = re.search(r'\bbreakdown\s+(?:for|of)\s+(\w+)', q_lower)
        if breakdown_of_match and not group_entity:
            entity = breakdown_of_match.group(1).strip()
            group_entity = 'status'
        if 'breakdown' in q_lower and 'status' in q_lower and not group_entity:
            group_entity = 'status'
        if not group_entity:
            return None
        best_match, best_score = None, 0
        for table_name, table_info in self.schema.tables.items():
            for col_name, col_info in table_info.columns.items():
                if not col_info.is_categorical and 'groupable' not in col_info.semantic_tags:
                    continue
                score = 0
                ge = group_entity.lower()
                for alias in col_info.nl_aliases:
                    if alias in ge or ge in alias:
                        score += len(min(alias, ge, key=len)) * 3
                ge_tokens = self._tokenize_question(ge)
                score += len(ge_tokens & col_info.tokens) * 2
                if target and table_name == target[0]:
                    score *= 1.3
                if score > best_score:
                    best_score = score
                    best_match = (table_name, col_name)
        return best_match

    def _detect_filters(self, q_lower, q_tokens, table_scores):
        filters = []
        status_patterns = {
            'denied': {'table': 'CLAIMS', 'column': 'CLAIM_STATUS', 'value': 'DENIED', 'op': '='},
            'rejected': {'table': 'CLAIMS', 'column': 'CLAIM_STATUS', 'value': 'DENIED', 'op': '='},
            'pending': {'table': 'CLAIMS', 'column': 'CLAIM_STATUS', 'value': 'PENDING', 'op': '='},
            'no-show': {'table': 'appointments', 'column': 'STATUS', 'value': 'No-Show', 'op': '='},
            'no show': {'table': 'appointments', 'column': 'STATUS', 'value': 'No-Show', 'op': '='},
            'chronic': {'table': 'DIAGNOSES', 'column': 'IS_CHRONIC', 'value': 'Y', 'op': '='},
            'inpatient': {'table': 'ENCOUNTERS', 'column': 'VISIT_TYPE', 'value': 'INPATIENT', 'op': '='},
            'outpatient': {'table': 'ENCOUNTERS', 'column': 'VISIT_TYPE', 'value': 'OUTPATIENT', 'op': '='},
            'emergency': {'table': 'ENCOUNTERS', 'column': 'VISIT_TYPE', 'value': 'EMERGENCY', 'op': '='},
            'telehealth': {'table': 'ENCOUNTERS', 'column': 'VISIT_TYPE', 'value': 'TELEHEALTH', 'op': '='},
        }
        for phrase, filt in status_patterns.items():
            if phrase in q_lower:
                filters.append(dict(filt))
        diag_patterns = {
            'diabet': {'table': 'DIAGNOSES', 'column': 'ICD10_DESCRIPTION',
                      'op': 'LIKE', 'value': '%diabet%',
                      'icd_prefix': ['E10', 'E11', 'E13']},
            'hypertens': {'table': 'DIAGNOSES', 'column': 'ICD10_DESCRIPTION',
                         'op': 'LIKE', 'value': '%hypertens%',
                         'icd_prefix': ['I10', 'I11', 'I12', 'I13']},
            'asthma': {'table': 'DIAGNOSES', 'column': 'ICD10_DESCRIPTION',
                      'op': 'LIKE', 'value': '%asthma%', 'icd_prefix': ['J45']},
        }
        for phrase, filt in diag_patterns.items():
            if phrase in q_lower:
                filters.append(dict(filt))
        return filters

    def _resolve_joins(self, needed_tables):
        if len(needed_tables) <= 1:
            return []
        joins = []
        tables_list = list(needed_tables)
        connected = {tables_list[0]}
        for table in tables_list[1:]:
            if table in connected:
                continue
            for ct in connected:
                jp = self.schema.find_join_path(table, ct)
                if jp:
                    joins.append(jp)
                    connected.add(table)
                    break
        return joins


@dataclass
class DecomposedIntent:
    query_type: str
    aggregation: Optional[str]
    is_distinct: bool = False
    limit: Optional[int] = None
    order_direction: Optional[str] = None
    is_rate: bool = False
    is_comparison: bool = False
    confidence: float = 0.0


class IntentDecomposer:
    def __init__(self, schema: SchemaGraph = None):
        self.schema = schema

    def decompose(self, question: str, linked: LinkedSchema) -> DecomposedIntent:
        q = question.lower()

        neural_intent = _neural_classify_intent(question)
        neural_agg = _neural_resolve_agg(question)

        agg = self._detect_aggregation(q, linked)
        if neural_agg and not agg:
            agg = neural_agg
            logger.debug("Neural agg override: %s for '%s'", agg, question[:40])

        query_type = self._detect_query_type(q, linked, agg)
        if neural_intent and neural_intent[1] > 0.65:
            n_intent, n_conf = neural_intent
            _intent_map = {
                'count': 'count', 'aggregate': 'scalar', 'rank': 'ranked',
                'rate': 'rate', 'trend': 'trend', 'compare': 'compare',
                'list': 'grouped',
            }
            mapped = _intent_map.get(n_intent)
            if mapped and query_type == 'scalar' and mapped != 'scalar':
                query_type = mapped
                logger.debug("Neural intent refinement: %s→%s (conf=%.2f) for '%s'",
                           n_intent, mapped, n_conf, question[:40])

        is_distinct = bool(re.search(r'\b(?:unique|distinct|different)\b', q))
        limit = self._detect_limit(q)
        order_dir = self._detect_order(q)
        is_rate = bool(re.search(r'\b(?:rate|percentage|percent|proportion|ratio|frequency)\b', q))

        confidence = 0.5
        if agg: confidence += 0.15
        if linked.target_column: confidence += 0.15
        if query_type in ('scalar', 'count', 'rate'): confidence += 0.1
        if neural_intent and neural_intent[1] > 0.6:
            confidence += 0.05
        return DecomposedIntent(
            query_type=query_type, aggregation=agg, is_distinct=is_distinct,
            limit=limit, order_direction=order_dir, is_rate=is_rate,
            confidence=min(confidence, 0.98))

    def _detect_aggregation(self, q, linked):
        if re.search(r'\b(?:going rate|contracted rate)\b', q):
            return 'AVG'
        if re.search(r'\b(?:average|avg|mean|typical|expected|per capita)\b', q):
            return 'AVG'
        if re.search(r'\b(?:on average|typically)\b', q):
            return 'AVG'
        if re.search(r'\breimburs\w*\s+per\s+', q):
            return 'AVG'
        if re.search(r'\b(?:highest|maximum|max\b|greatest|largest|biggest|busiest|peak|ceiling)\b', q):
            return 'MAX'
        if re.search(r'\b(?:most expensive|costliest|priciest)\b', q):
            return 'MAX'
        if re.search(r'\b(?:lowest|minimum|min\b|smallest|least expensive|cheapest|fewest|floor\b|bottom)\b', q):
            return 'MIN'
        if re.search(r'\b(?:total|combined|sum|overall|aggregate|cumulative|tally|tally up|gross|net)\b', q):
            if re.search(r'total\s+(?:count|number)', q):
                return 'COUNT'
            if re.search(r'total\s+(?:member|claim|prescription|encounter|provider|patient|referral|appointment|physician|doctor)s?\b', q):
                if not re.search(r'total\s+\w+\s+(?:cost|spend|expense|amount|paid|billed)', q):
                    return 'COUNT'
            return 'SUM'
        if re.search(r'\b(?:how many|count of|number of|total number|headcount|head count|tally of)\b', q):
            return 'COUNT'
        if re.search(r'\w+\s+count\b', q) and not re.search(r'count\s+(?:of|the|by|for|per)', q):
            return 'COUNT'
        if re.search(r'\bpopulation\b', q) and not re.search(r'population\s+(?:health|risk|cost)', q):
            return 'COUNT'
        if 'how much' in q:
            if re.search(r'how much\b.*\bper\s+', q):
                return 'AVG'
            return 'SUM'
        if re.search(r'\b(?:pay out|reimburse?|reimbursing|spending)\b', q):
            if re.search(r'\b(?:average|typically|per|each)\b', q):
                return 'AVG'
            return 'SUM'
        return None

    def _detect_query_type(self, q, linked, agg):
        if re.search(r'\b(?:share|proportion)\s+of\b', q):
            return 'rate'
        if re.search(r'\b(?:rate|percentage|percent|proportion)\b', q):
            if re.search(r'\b(?:going rate|contracted rate|negotiated rate|reimbursement rate)\b', q):
                pass
            else:
                return 'rate'
        if re.search(r'\b(?:top\s+\d+|bottom\s+\d+|ranked|ranking)\b', q):
            return 'ranked'
        if re.search(r'\b(?:most\s+(?:common|frequent|popular|expensive|prescribed))\b', q):
            return 'ranked'
        if re.search(r'\b(?:least\s+(?:common|frequent|popular|expensive))\b', q):
            return 'ranked'
        if re.search(r'(?:largest|smallest|biggest|highest|lowest)\s+to\s+(?:largest|smallest|biggest|highest|lowest)', q):
            return 'ranked'

        superlative_entity = re.search(
            r'\b(?:largest|biggest|smallest|highest|lowest|busiest|most expensive|costliest|cheapest|'
            r'most prescribed|most common|most frequent)\s+(\w+)', q)
        if superlative_entity:
            entity = superlative_entity.group(1).lower()
            entity_stem = _simple_stem(entity)
            entity_words = {
                'panel', 'medication', 'drug', 'provider', 'doctor', 'facility',
                'hospital', 'region', 'specialty', 'plan', 'department', 'diagnosis',
                'condition', 'member', 'patient', 'claim', 'prescription', 'encounter',
                'procedure', 'referral', 'appointment',
            }
            if entity_stem in entity_words or entity in entity_words:
                return 'ranked'
            if self.schema:
                for tname, tinfo in self.schema.tables.items():
                    aliases_flat = set()
                    for a in tinfo.nl_aliases:
                        for w in a.split():
                            aliases_flat.add(w)
                            aliases_flat.add(_simple_stem(w))
                    if entity_stem in aliases_flat or entity in aliases_flat:
                        return 'ranked'

        if re.search(r'which\s+\w+\s+has\s+the\s+(?:most|highest|lowest|largest|biggest|smallest|fewest|least)', q):
            return 'ranked'

        if re.search(r'(?:highest|lowest|largest|smallest|biggest)\s+(?:average|avg|total|mean)', q):
            if re.search(r'\b(?:by|per|for each)\b', q) or linked.group_column:
                return 'ranked'

        if agg in ('MAX', 'MIN') and not linked.group_column:
            return 'scalar'
        if linked.group_column:
            return 'grouped'
        if re.search(r'\b(?:by|per|for each|broken down|breakdown|distribution|across\s+(?:different\s+)?(?:\w+\s+)?(?:group|type|category|region|plan|segment|age))', q):
            return 'grouped'
        if re.search(r'\b(?:unique|distinct)\b', q):
            return 'count'
        if agg == 'COUNT':
            return 'count'
        if agg:
            return 'scalar'
        return 'list'

    def _detect_limit(self, q):
        m = re.search(r'\btop\s+(\d+)', q)
        if m: return int(m.group(1))
        if re.search(r'\btop\b', q): return 10
        return None

    def _detect_order(self, q):
        if re.search(r'\b(?:highest|most|largest|biggest|top|greatest|expensive|busiest)\b', q):
            return 'DESC'
        if re.search(r'\b(?:lowest|least|smallest|bottom|cheapest|fewest)\b', q):
            return 'ASC'
        return None


class SQLAssembler:
    def __init__(self, schema: SchemaGraph):
        self.schema = schema

    def assemble(self, linked: LinkedSchema, intent: DecomposedIntent, question: str) -> str:
        q_lower = question.lower()
        primary_table = self._primary_table(linked, intent)
        if not primary_table:
            return ""
        select = self._build_select(linked, intent, primary_table, q_lower)
        from_cl = self._build_from(linked, primary_table)
        where = self._build_where(linked, intent, primary_table, q_lower)
        group = self._build_group_by(linked, intent)
        order = self._build_order_by(linked, intent, q_lower)
        limit = self._build_limit(intent)
        sql = f"SELECT {select} FROM {from_cl}"
        if where: sql += f" WHERE {where}"
        if group: sql += f" GROUP BY {group}"
        if order: sql += f" ORDER BY {order}"
        if limit: sql += f" LIMIT {limit}"
        return sql

    def _primary_table(self, linked, intent):
        if linked.target_column: return linked.target_column[0]
        if linked.group_column: return linked.group_column[0]
        if linked.tables: return linked.tables[0][0]
        return None

    def _build_select(self, linked, intent, primary_table, q_lower):
        target = linked.target_column
        group = linked.group_column
        agg = intent.aggregation

        if intent.query_type == 'rate':
            return self._build_rate_select(linked, intent, primary_table, q_lower)

        if intent.query_type == 'count':
            parts = []
            if group:
                parts.append(self._col_ref(group, linked))
            if intent.is_distinct and target:
                parts.append(f"COUNT(DISTINCT {self._col_ref(target, linked)}) AS unique_count")
            elif target:
                ci = self._get_ci(target)
                if ci and ci.is_identifier:
                    parts.append(f"COUNT(DISTINCT {self._col_ref(target, linked)}) AS total_count")
                else:
                    parts.append("COUNT(*) AS total_count")
            else:
                parts.append("COUNT(*) AS total_count")
            return ', '.join(parts)

        if intent.query_type == 'scalar':
            if agg and target:
                ci = self._get_ci(target)
                cr = self._col_ref(target, linked)
                if ci and ci.is_numeric:
                    cast = f"CAST({cr} AS REAL)"
                    return f"ROUND({agg}({cast}), 2) AS {agg.lower()}_{target[1].lower()}"
                return f"{agg}({cr}) AS {agg.lower()}_value"
            if agg:
                return f"{agg}(*) AS total"
            return '*'

        if intent.query_type == 'ranked':
            return self._build_ranked_select(linked, intent, primary_table, q_lower)

        if intent.query_type == 'grouped':
            return self._build_grouped_select(linked, intent, primary_table, q_lower)

        if target:
            return self._col_ref(target, linked)
        return '*'

    def _build_ranked_select(self, linked, intent, primary_table, q_lower):
        parts = []
        target = linked.target_column
        group = linked.group_column
        agg = intent.aggregation or 'SUM'

        if group:
            parts.append(self._col_ref(group, linked))
        else:
            gc = self._infer_rank_group(q_lower, primary_table)
            if gc:
                parts.append(self._col_ref(gc, linked))
                linked.group_column = gc

        metric_col = None
        if target:
            ci = self._get_ci(target)
            if ci and ci.is_numeric and not ci.is_identifier:
                metric_col = target
            elif ci and ci.is_identifier:
                cr = self._col_ref(target, linked)
                alias = f"count_{target[1].lower()}"
                parts.append(f"COUNT(DISTINCT {cr}) AS {alias}")
                self._last_rank_alias = alias
                return ', '.join(parts) if parts else '*'
            elif ci and not ci.is_numeric:
                if not linked.group_column:
                    linked.group_column = target
                    if len(parts) == 0 or self._col_ref(target, linked) not in parts[0]:
                        parts.insert(0, self._col_ref(target, linked))
                metric_col = self._infer_rank_metric(q_lower, linked)

        if not metric_col:
            metric_col = self._infer_rank_metric(q_lower, linked)

        if metric_col:
            ci = self._get_ci(metric_col)
            cr = self._col_ref(metric_col, linked)
            if ci and ci.is_numeric:
                if re.search(r'\b(?:common|frequent|popular|prescribed)\b', q_lower) and agg in ('MAX', 'SUM'):
                    agg = 'COUNT'
                alias = f"{agg.lower()}_{metric_col[1].lower()}"
                if agg == 'COUNT':
                    parts.append(f"COUNT(*) AS count")
                    self._last_rank_alias = 'count'
                else:
                    parts.append(f"ROUND({agg}(CAST({cr} AS REAL)), 2) AS {alias}")
                    self._last_rank_alias = alias
            else:
                parts.append("COUNT(*) AS count")
                self._last_rank_alias = 'count'
        else:
            parts.append("COUNT(*) AS count")
            self._last_rank_alias = 'count'
        return ', '.join(parts) if parts else '*'

    def _build_grouped_select(self, linked, intent, primary_table, q_lower):
        parts = []
        target = linked.target_column
        group = linked.group_column
        agg = intent.aggregation or 'COUNT'

        if group:
            parts.append(self._col_ref(group, linked))
        if target:
            ci = self._get_ci(target)
            cr = self._col_ref(target, linked)
            if ci and ci.is_numeric and agg in ('SUM', 'AVG', 'MAX', 'MIN'):
                parts.append(f"ROUND({agg}(CAST({cr} AS REAL)), 2) AS {agg.lower()}_{target[1].lower()}")
            elif ci and ci.is_identifier:
                parts.append(f"COUNT(DISTINCT {cr}) AS count_{target[1].lower()}")
            else:
                parts.append("COUNT(*) AS count")
        else:
            parts.append("COUNT(*) AS count")
        return ', '.join(parts)

    def _build_rate_select(self, linked, intent, primary_table, q_lower):
        if 'denial' in q_lower or 'denied' in q_lower or 'reject' in q_lower:
            cond = "CLAIM_STATUS = 'DENIED'"
            label = 'denial_rate_pct'
        elif 'no-show' in q_lower or 'no show' in q_lower:
            cond = "STATUS = 'No-Show'"
            label = 'no_show_rate_pct'
        elif 'approv' in q_lower or 'clean claim' in q_lower:
            cond = "CLAIM_STATUS = 'APPROVED'"
            label = 'approval_rate_pct'
        else:
            cond = "1=1"
            label = 'rate_pct'
        parts = []
        if linked.group_column:
            parts.append(self._col_ref(linked.group_column, linked))
        parts.append("COUNT(*) AS total_count")
        parts.append(f"SUM(CASE WHEN {cond} THEN 1 ELSE 0 END) AS condition_count")
        parts.append(f"ROUND(100.0 * SUM(CASE WHEN {cond} THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) AS {label}")
        return ', '.join(parts)

    def _infer_rank_group(self, q_lower, primary_table):
        entity_map = {
            'medication': ('prescriptions', 'MEDICATION_NAME'),
            'drug': ('prescriptions', 'MEDICATION_NAME'),
            'diagnosis': ('diagnoses', 'ICD10_DESCRIPTION'),
            'condition': ('diagnoses', 'ICD10_DESCRIPTION'),
            'provider': ('providers', 'NPI'),
            'doctor': ('providers', 'NPI'),
            'facility': ('encounters', 'FACILITY'),
            'hospital': ('encounters', 'FACILITY'),
            'region': ('claims', 'KP_REGION'),
            'specialty': ('providers', 'SPECIALTY'),
            'plan': ('claims', 'PLAN_TYPE'),
            'department': ('encounters', 'DEPARTMENT'),
            'procedure': ('claims', 'CPT_DESCRIPTION'),
            'panel': ('providers', 'NPI'),
            'encounter': ('encounters', 'FACILITY'),
            'area': ('claims', 'KP_REGION'),
            'claim': ('claims', 'PLAN_TYPE'),
            'member': ('members', 'MEMBER_ID'),
            'patient': ('members', 'MEMBER_ID'),
            'pharmacy': ('prescriptions', 'PHARMACY'),
        }
        q_tokens = set(re.sub(r'[^a-z0-9\s]', ' ', q_lower).split())
        q_stems = {_simple_stem(t) for t in q_tokens}
        for entity, (table, col) in entity_map.items():
            if entity in q_lower or entity in q_stems:
                real_table = self._resolve_table_name(table)
                return (real_table, col)
        return None

    def _resolve_table_name(self, name):
        if name in self.schema.tables:
            return name
        for t in self.schema.tables:
            if t.lower() == name.lower():
                return t
        return name

    def _infer_rank_metric(self, q_lower, linked):
        cost_words = ['expensive', 'costly', 'costliest', 'priciest', 'cheapest',
                     'cost', 'spend', 'revenue']
        if any(w in q_lower for w in cost_words):
            group = linked.group_column
            if group:
                tinfo = self.schema.tables.get(group[0])
                if tinfo:
                    for col in tinfo.columns.values():
                        if 'cost' in col.semantic_tags or 'financial' in col.semantic_tags:
                            return (group[0], col.name)
                claims = self.schema.tables.get('CLAIMS')
                if claims:
                    return ('CLAIMS', 'PAID_AMOUNT')

        if 'panel' in q_lower or 'patient load' in q_lower:
            return ('PROVIDERS', 'PANEL_SIZE')

        if 'stay' in q_lower or 'length of stay' in q_lower or 'los' in q_lower:
            return ('ENCOUNTERS', 'LENGTH_OF_STAY')

        if 'risk' in q_lower or 'acuity' in q_lower:
            return ('MEMBERS', 'RISK_SCORE')

        return None

    def _build_from(self, linked, primary_table):
        if not linked.joins:
            return primary_table
        parts = [primary_table]
        joined = {primary_table.upper()}
        for jp in linked.joins:
            if jp.from_table.upper() in joined and jp.to_table.upper() not in joined:
                parts.append(f"JOIN {jp.to_table} ON {jp.from_table}.{jp.from_column} = {jp.to_table}.{jp.to_column}")
                joined.add(jp.to_table.upper())
            elif jp.to_table.upper() in joined and jp.from_table.upper() not in joined:
                parts.append(f"JOIN {jp.from_table} ON {jp.from_table}.{jp.from_column} = {jp.to_table}.{jp.to_column}")
                joined.add(jp.from_table.upper())
        return ' '.join(parts)

    def _build_where(self, linked, intent, primary_table, q_lower):
        conditions = []
        has_joins = bool(linked.joins)
        for f in linked.filters:
            t = f.get('table', primary_table)
            prefix = f"{t}." if has_joins else ""
            col, val, op = f['column'], f['value'], f.get('op', '=')
            if op == 'LIKE':
                cond = f"{prefix}{col} LIKE '{val}'"
                if 'icd_prefix' in f:
                    icds = ' OR '.join(f"{prefix}ICD10_CODE LIKE '{p}%'" for p in f['icd_prefix'])
                    cond = f"({cond} OR {icds})"
                conditions.append(cond)
            elif val == '':
                conditions.append(f"({prefix}{col} = '' OR {prefix}{col} IS NULL)")
            else:
                conditions.append(f"{prefix}{col} {op} '{val}'")
        if linked.target_column:
            ci = self._get_ci(linked.target_column)
            if ci and ci.is_numeric and intent.aggregation in ('AVG', 'MAX', 'MIN'):
                t, c = linked.target_column
                prefix = f"{t}." if has_joins else ""
                conditions.append(f"{prefix}{c} IS NOT NULL")
        return ' AND '.join(conditions)

    def _build_group_by(self, linked, intent):
        if intent.query_type in ('grouped', 'ranked', 'rate') and linked.group_column:
            return self._col_ref(linked.group_column, linked)
        return ''

    def _build_order_by(self, linked, intent, q_lower):
        d = intent.order_direction or 'DESC'
        if intent.query_type == 'ranked':
            alias = getattr(self, '_last_rank_alias', 'count')
            return f"{alias} {d}"
        if intent.query_type in ('grouped', 'rate') and intent.order_direction:
            alias = getattr(self, '_last_rank_alias', None)
            if alias:
                return f"{alias} {d}"
            if linked.target_column:
                agg = intent.aggregation or 'COUNT'
                return f"{agg.lower()}_{linked.target_column[1].lower()} {d}"
            return f"count {d}"
        return ''

    def _build_limit(self, intent):
        if intent.limit: return str(intent.limit)
        if intent.query_type == 'ranked': return '20'
        return ''

    def _col_ref(self, tc, linked):
        if isinstance(tc, tuple):
            t, c = tc
        else:
            return str(tc)
        if linked.joins:
            return f"{t}.{c}"
        return c

    def _get_ci(self, tc):
        if not tc: return None
        t, c = tc
        tinfo = self.schema.tables.get(t)
        if not tinfo: return None
        return tinfo.columns.get(c)


class QueryIntelligence:

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.schema = SchemaGraph(db_path)
        self.linker = SchemaLinker(self.schema)
        self.decomposer = IntentDecomposer(self.schema)
        self.assembler = SQLAssembler(self.schema)

        _init_neural()
        _get_language_brain(self.schema)
        _compute_concept_centroids()

        logger.info("QueryIntelligence: %d tables, %d columns, %d joins, neural=%s",
                     len(self.schema.tables),
                     sum(len(t.columns) for t in self.schema.tables.values()),
                     len(self.schema.join_paths),
                     'ACTIVE' if _NEURAL_AVAILABLE else 'fallback')

    def generate_sql(self, question: str) -> Dict[str, Any]:

        concept = _match_concept_semantically(question)
        if concept:
            sql = self._render_concept_sql(concept)
            valid, issues = self._validate_sql(sql, question, None, None)
            if valid:
                return {
                    'sql': sql,
                    'confidence': concept.confidence,
                    'valid': valid,
                    'issues': issues,
                    'linked_schema': None,
                    'intent': DecomposedIntent(
                        query_type=concept.query_type,
                        aggregation=concept.agg,
                        confidence=concept.confidence),
                    'source': 'query_intelligence',
                    '_neural_match': concept.name,
                }

        composite = self._resolve_composite_concept(question)
        if composite:
            sql = composite['sql']
            valid, issues = self._validate_sql(sql, question, None, None)
            return {
                'sql': sql,
                'confidence': composite.get('confidence', 0.85),
                'valid': valid,
                'issues': issues,
                'linked_schema': None,
                'intent': DecomposedIntent(
                    query_type=composite.get('type', 'scalar'),
                    aggregation=composite.get('agg'),
                    confidence=composite.get('confidence', 0.85)),
                'source': 'query_intelligence',
            }

        linked = self.linker.link(question)
        intent = self.decomposer.decompose(question, linked)
        sql = self.assembler.assemble(linked, intent, question)
        valid, issues = self._validate_sql(sql, question, linked, intent)
        return {
            'sql': sql,
            'confidence': intent.confidence,
            'valid': valid,
            'issues': issues,
            'linked_schema': linked,
            'intent': intent,
            'source': 'query_intelligence',
        }

    def _render_concept_sql(self, concept: SemanticConceptTemplate) -> str:
        members_t = self.assembler._resolve_table_name('members')
        claims_t = self.assembler._resolve_table_name('claims')
        encounters_t = self.assembler._resolve_table_name('encounters')
        providers_t = self.assembler._resolve_table_name('providers')
        return concept.sql_template.format(
            members=members_t, claims=claims_t,
            encounters=encounters_t, providers=providers_t,
        )

    def _resolve_composite_concept(self, question: str) -> Optional[Dict]:
        q = question.lower()

        members_t = self.assembler._resolve_table_name('members')
        claims_t = self.assembler._resolve_table_name('claims')
        encounters_t = self.assembler._resolve_table_name('encounters')

        if re.search(r'\bpmpm\b|per member per month', q):
            return {
                'sql': (f"SELECT ROUND(SUM(CAST({claims_t}.PAID_AMOUNT AS REAL)) / "
                        f"NULLIF(COUNT(DISTINCT {members_t}.MEMBER_ID), 0), 2) AS pmpm_cost "
                        f"FROM {claims_t} JOIN {members_t} ON {claims_t}.MEMBER_ID = {members_t}.MEMBER_ID"),
                'type': 'scalar', 'agg': 'AVG', 'confidence': 0.85,
            }

        if re.search(r'\bloss ratio\b', q):
            return {
                'sql': (f"SELECT ROUND(100.0 * SUM(CAST(PAID_AMOUNT AS REAL)) / "
                        f"NULLIF(SUM(CAST(BILLED_AMOUNT AS REAL)), 0), 2) AS loss_ratio_pct, "
                        f"SUM(CAST(PAID_AMOUNT AS REAL)) AS total_paid, "
                        f"SUM(CAST(BILLED_AMOUNT AS REAL)) AS total_billed "
                        f"FROM {claims_t}"),
                'type': 'scalar', 'agg': 'SUM', 'confidence': 0.85,
            }

        if re.search(r'\bcost\b.*\bage\s*group\b', q) or re.search(r'\bage\s*group\b.*\bcost\b', q):
            return {
                'sql': (f"SELECT CASE "
                        f"WHEN (julianday('now') - julianday({members_t}.DATE_OF_BIRTH))/365.25 < 18 THEN 'Under 18' "
                        f"WHEN (julianday('now') - julianday({members_t}.DATE_OF_BIRTH))/365.25 < 30 THEN '18-29' "
                        f"WHEN (julianday('now') - julianday({members_t}.DATE_OF_BIRTH))/365.25 < 45 THEN '30-44' "
                        f"WHEN (julianday('now') - julianday({members_t}.DATE_OF_BIRTH))/365.25 < 65 THEN '45-64' "
                        f"ELSE '65+' END AS AGE_GROUP, "
                        f"ROUND(SUM(CAST({claims_t}.PAID_AMOUNT AS REAL)) / "
                        f"NULLIF(COUNT(DISTINCT {members_t}.MEMBER_ID), 0), 2) AS cost_per_member "
                        f"FROM {claims_t} JOIN {members_t} ON {claims_t}.MEMBER_ID = {members_t}.MEMBER_ID "
                        f"WHERE {members_t}.DATE_OF_BIRTH IS NOT NULL "
                        f"GROUP BY AGE_GROUP ORDER BY AGE_GROUP"),
                'type': 'grouped', 'agg': 'AVG', 'confidence': 0.88,
            }

        if re.search(r'\bage\s*group', q) or re.search(r'\bdistribution\b.*\bage\b', q):
            return {
                'sql': (f"SELECT CASE "
                        f"WHEN (julianday('now') - julianday(DATE_OF_BIRTH))/365.25 < 18 THEN 'Under 18' "
                        f"WHEN (julianday('now') - julianday(DATE_OF_BIRTH))/365.25 < 30 THEN '18-29' "
                        f"WHEN (julianday('now') - julianday(DATE_OF_BIRTH))/365.25 < 45 THEN '30-44' "
                        f"WHEN (julianday('now') - julianday(DATE_OF_BIRTH))/365.25 < 65 THEN '45-64' "
                        f"ELSE '65+' END AS AGE_GROUP, "
                        f"COUNT(*) AS member_count "
                        f"FROM {members_t} "
                        f"WHERE DATE_OF_BIRTH IS NOT NULL "
                        f"GROUP BY AGE_GROUP ORDER BY AGE_GROUP"),
                'type': 'grouped', 'agg': 'COUNT', 'confidence': 0.85,
            }

        if re.search(r'\b(?:area|region)s?\b.*\b(?:utiliz|generating|volume|activity)', q) or \
           re.search(r'\butiliz\w*\b.*\b(?:area|region)s?\b', q):
            return {
                'sql': (f"SELECT KP_REGION, COUNT(*) AS encounter_count "
                        f"FROM {encounters_t} "
                        f"GROUP BY KP_REGION ORDER BY encounter_count DESC"),
                'type': 'ranked', 'agg': 'COUNT', 'confidence': 0.82,
            }

        if re.search(r'\breimburs\w*\b.*\bper\s+encounter\b', q) or \
           re.search(r'\bgoing rate\b.*\breimburs\w*\b.*\bper\s+encounter\b', q) or \
           re.search(r'\bgoing rate\b.*\bper\s+encounter\b', q):
            return {
                'sql': (f"SELECT ROUND(AVG(CAST(PAID_AMOUNT AS REAL)), 2) AS avg_paid_amount "
                        f"FROM {claims_t} WHERE PAID_AMOUNT IS NOT NULL"),
                'type': 'scalar', 'agg': 'AVG', 'confidence': 0.85,
            }

        if re.search(r'\b(?:avg|average)\s+paid\b.*\bspecialty\b', q) or \
           re.search(r'\bspecialty\b.*\b(?:avg|average)\s+paid\b', q):
            providers_t = self.assembler._resolve_table_name('providers')
            return {
                'sql': (f"SELECT {providers_t}.SPECIALTY, "
                        f"ROUND(AVG(CAST({claims_t}.PAID_AMOUNT AS REAL)), 2) AS avg_paid_amount "
                        f"FROM {claims_t} "
                        f"JOIN {providers_t} ON {claims_t}.RENDERING_NPI = {providers_t}.NPI "
                        f"WHERE {claims_t}.PAID_AMOUNT IS NOT NULL "
                        f"GROUP BY {providers_t}.SPECIALTY ORDER BY avg_paid_amount DESC"),
                'type': 'grouped', 'agg': 'AVG', 'confidence': 0.88,
            }

        if re.search(r'\b(?:physician|doctor|clinician|practitioner)\b.*\b(?:headcount|head count|count|total|number|how many)\b', q) or \
           re.search(r'\b(?:headcount|head count|count|total|number|how many)\b.*\b(?:physician|doctor|clinician|practitioner)\b', q):
            providers_t = self.assembler._resolve_table_name('providers')
            return {
                'sql': f"SELECT COUNT(DISTINCT NPI) AS total_count FROM {providers_t}",
                'type': 'count', 'agg': 'COUNT', 'confidence': 0.85,
            }

        if re.search(r'\bcovered\s+lives\b', q) or \
           (re.search(r'\bpopulation\b', q) and re.search(r'\b(?:member|patient|enrollee|covered|live)\b', q)):
            return {
                'sql': f"SELECT COUNT(DISTINCT MEMBER_ID) AS total_count FROM {members_t}",
                'type': 'count', 'agg': 'COUNT', 'confidence': 0.85,
            }

        if re.search(r'\bcost\b.*\bvisit\s*type\b', q) or re.search(r'\bvisit\s*type\b.*\bcost\b', q):
            return {
                'sql': (f"SELECT {encounters_t}.VISIT_TYPE, "
                        f"ROUND(AVG(CAST({claims_t}.PAID_AMOUNT AS REAL)), 2) AS avg_cost "
                        f"FROM {claims_t} JOIN {encounters_t} ON {claims_t}.MEMBER_ID = {encounters_t}.MEMBER_ID "
                        f"WHERE {claims_t}.PAID_AMOUNT IS NOT NULL "
                        f"GROUP BY {encounters_t}.VISIT_TYPE ORDER BY avg_cost DESC"),
                'type': 'grouped', 'agg': 'AVG', 'confidence': 0.85,
            }

        if re.search(r'\b(?:total|sum|overall|cumulative)\s+(?:cost|spend\w*)\s+(?:of|for|across)\s+(?:all\s+)?claims\b', q):
            return {
                'sql': (f"SELECT ROUND(SUM(CAST(PAID_AMOUNT AS REAL)), 2) AS total_cost "
                        f"FROM {claims_t}"),
                'type': 'scalar', 'agg': 'SUM', 'confidence': 0.85,
            }

        if re.search(r'\bunique\s+providers?\b', q) or re.search(r'\bdistinct\s+providers?\b', q):
            providers_t = self.assembler._resolve_table_name('providers')
            return {
                'sql': f"SELECT COUNT(DISTINCT NPI) AS unique_providers FROM {providers_t}",
                'type': 'count', 'agg': 'COUNT', 'confidence': 0.88,
            }

        if re.search(r'\bstatus\s+breakdown\b|\bbreakdown\b.*\bstatus\b', q):
            if 'referral' in q:
                return {
                    'sql': (f"SELECT STATUS, COUNT(*) AS count FROM referrals "
                            f"GROUP BY STATUS ORDER BY count DESC"),
                    'type': 'grouped', 'agg': 'COUNT', 'confidence': 0.85,
                }
            else:
                return {
                    'sql': (f"SELECT CLAIM_STATUS, COUNT(*) AS count FROM {claims_t} "
                            f"GROUP BY CLAIM_STATUS ORDER BY count DESC"),
                    'type': 'grouped', 'agg': 'COUNT', 'confidence': 0.85,
                }

        if re.search(r'\bbreakdown\s+for\s+(\w+)', q):
            entity = re.search(r'\bbreakdown\s+for\s+(\w+)', q).group(1)
            if 'referral' in entity:
                return {
                    'sql': (f"SELECT STATUS, COUNT(*) AS count FROM referrals "
                            f"GROUP BY STATUS ORDER BY count DESC"),
                    'type': 'grouped', 'agg': 'COUNT', 'confidence': 0.85,
                }
            elif 'claim' in entity:
                return {
                    'sql': (f"SELECT CLAIM_STATUS, COUNT(*) AS count FROM {claims_t} "
                            f"GROUP BY CLAIM_STATUS ORDER BY count DESC"),
                    'type': 'grouped', 'agg': 'COUNT', 'confidence': 0.85,
                }

        if re.search(r'\bdenied\s+(?:claims?|referrals?)\s+count\b', q):
            if 'referral' in q:
                return {
                    'sql': (f"SELECT COUNT(*) AS denied_count FROM referrals "
                            f"WHERE STATUS = 'Denied'"),
                    'type': 'count', 'agg': 'COUNT', 'confidence': 0.85,
                }
            else:
                return {
                    'sql': (f"SELECT COUNT(*) AS denied_count FROM {claims_t} "
                            f"WHERE CLAIM_STATUS = 'DENIED'"),
                    'type': 'count', 'agg': 'COUNT', 'confidence': 0.85,
                }

        if re.search(r'\bdenial\s+rate\b.*\bpayer\b', q) or re.search(r'\bpayer\b.*\bdenial\s+rate\b', q):
            return {
                'sql': (f"SELECT PLAN_TYPE AS payer, COUNT(*) AS total_count, "
                        f"SUM(CASE WHEN CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) AS denied_count, "
                        f"ROUND(100.0 * SUM(CASE WHEN CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) / "
                        f"NULLIF(COUNT(*), 0), 2) AS denial_rate_pct "
                        f"FROM {claims_t} GROUP BY PLAN_TYPE ORDER BY denial_rate_pct DESC"),
                'type': 'rate', 'agg': None, 'confidence': 0.88,
            }

        return None

    def _validate_sql(self, sql, question, linked=None, intent=None):
        issues = []
        if not sql:
            issues.append("No SQL generated")
            return False, issues
        try:
            conn = sqlite3.connect(self.db_path)
            conn.cursor().execute(f"EXPLAIN QUERY PLAN {sql}")
            conn.close()
        except Exception as e:
            issues.append(f"SQL error: {e}")
            return False, issues
        return True, issues
