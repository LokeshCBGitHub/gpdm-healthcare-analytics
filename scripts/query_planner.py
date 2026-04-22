import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Set
from collections import defaultdict

from tokenizer_utils import tokenize_query, get_mentioned_columns

logger = logging.getLogger('gpdm.planner')


@dataclass
class MetricSpec:
    concept: str
    aggregation: str
    target: Optional[str] = None
    denominator: Optional[str] = None
    qualifier: Optional[str] = None

@dataclass
class DimensionSpec:
    concept: str
    temporal: bool = False
    granularity: Optional[str] = None

@dataclass
class FilterSpec:
    concept: str
    operator: str = '='
    value: Optional[str] = None
    temporal: bool = False
    temporal_range: Optional[str] = None

@dataclass
class SortSpec:
    direction: str = 'desc'
    by: Optional[str] = None
    limit: Optional[int] = None

@dataclass
class QueryPlan:
    intent: str = 'lookup'

    metrics: List[MetricSpec] = field(default_factory=list)

    dimensions: List[DimensionSpec] = field(default_factory=list)

    filters: List[FilterSpec] = field(default_factory=list)

    sort: Optional[SortSpec] = None

    percentage: bool = False
    cumulative: bool = False
    comparison_mode: Optional[str] = None

    entity_focus: Optional[str] = None

    raw_question: str = ''

    confidence: float = 0.0


class QueryPlanner:

    _INTENT_PATTERNS = [
        (r'\b(?:trend|over\s+time|monthly|by\s+month|by\s+year|quarterly|'
         r'over\s+the\s+last|year\s+over\s+year|yoy|time\s+series|'
         r'changed|changes?|evolution|trajectory)\b', 'trend'),
        (r'\b(?:denial\s+rate|approval\s+rate|readmission\s+rate|'
         r'rate\s+of|what\s+(?:percent|percentage|%)|'
         r'how\s+many\s+(?:percent|%))\b', 'rate'),
        (r'\b(?:top\s+\d+|bottom\s+\d+|highest|lowest|most|least|'
         r'rank|ranking|best|worst|which\s+\w+\s+has?\s+the)\b', 'ranking'),
        (r'\b(?:compare|comparison|versus|vs\.?|differ|difference|'
         r'contrast)\b', 'comparison'),
        (r'\b(?:breakdown|distribution|split|composition|proportion|'
         r'share|makeup|breakout)\b', 'distribution'),
        (r'\b(?:show\s+(?:me\s+)?(?:all\s+)?|list\s+(?:all\s+)?|display\s+(?:all\s+)?|'
         r'get\s+(?:all\s+)?|view\s+(?:all\s+)?|give\s+me\s+(?:all\s+)?|'
         r'pull\s+(?:up\s+)?(?:all\s+)?|find\s+(?:all\s+)?|fetch\s+(?:all\s+)?)'
         r'(?:the\s+)?(\w+)', 'lookup'),
        (r'\b(?:average|avg|total|sum|count|how\s+many|number\s+of|'
         r'mean|median|minimum|maximum)\b', 'aggregation'),
    ]

    _AGG_PATTERNS = [
        (r'\baverage\b|\bavg\b|\bmean\b', 'avg'),
        (r'\bcount\b|\bhow\s+many\b|\bnumber\s+of\b|\bvolume\b', 'count'),
        (r'\btotal\b|\bsum\b|\bcombined\b|\baggregate\b', 'sum'),
        (r'\bmaximum\b|\bmax\b|\bhighest\b|\blargest\b|\bbiggest\b|\bmost\b', 'max'),
        (r'\bminimum\b|\bmin\b|\blowest\b|\bsmallest\b|\bleast\b', 'min'),
        (r'\bdistinct\b|\bunique\b', 'count_distinct'),
    ]

    _TEMPORAL_GRANULARITY = [
        (r'\b(?:daily|by\s+day|per\s+day|each\s+day)\b', 'day'),
        (r'\b(?:weekly|by\s+week|per\s+week|each\s+week)\b', 'week'),
        (r'\b(?:monthly|by\s+month|per\s+month|each\s+month)\b', 'month'),
        (r'\b(?:quarterly|by\s+quarter|per\s+quarter|each\s+quarter)\b', 'quarter'),
        (r'\b(?:yearly|annually|by\s+year|per\s+year|each\s+year|annual)\b', 'year'),
    ]

    _TIME_FILTER_PATTERNS = [
        (r'\blast\s+(\d+)\s+months?\b', 'months'),
        (r'\blast\s+(\d+)\s+years?\b', 'years'),
        (r'\blast\s+(\d+)\s+days?\b', 'days'),
        (r'\blast\s+(\d+)\s+weeks?\b', 'weeks'),
        (r'\blast\s+(\d+)\s+quarters?\b', 'quarters'),
        (r'\b(?:last|past)\s+month\b', 'last_month'),
        (r'\b(?:last|past)\s+quarter\b', 'last_quarter'),
        (r'\b(?:last|past)\s+year\b', 'last_year'),
        (r'\bthis\s+year\b', 'this_year'),
        (r'\bthis\s+month\b', 'this_month'),
        (r'\bthis\s+quarter\b', 'this_quarter'),
        (r'\bytd\b|\byear\s+to\s+date\b', 'ytd'),
    ]

    _SORT_PATTERNS = [
        (r'\btop\s+(\d+)\b', 'desc'),
        (r'\bbottom\s+(\d+)\b', 'asc'),
        (r'\bhighest\b|\bmost\b|\blargest\b|\bbiggest\b|\bgreatest\b', 'desc'),
        (r'\blowest\b|\bleast\b|\bsmallest\b|\bfewest\b', 'asc'),
    ]

    _DIMENSION_SIGNALS = [
        r'\bby\s+(\w+(?:\s+\w+)?)\b',
        r'\bper\s+(\w+)\b',
        r'\beach\s+(\w+)\b',
        r'\bacross\s+(\w+(?:\s+\w+)?)\b',
        r'\bfor\s+each\s+(\w+)\b',
        r'\bgrouped?\s+by\s+(\w+(?:\s+\w+)?)\b',
    ]

    _RATE_PATTERNS = [
        (r'(\w+)\s+rate', None),
        (r'(?:percent|percentage|%)\s+(?:of\s+)?(\w+)', None),
        (r'what\s+(?:percent|percentage|%)\s+(?:of\s+)?(?:\w+\s+)?(?:are|is|were|was)\s+(\w+)', None),
    ]

    _THRESHOLD_PATTERNS = [
        r'(?:more\s+than|over|above|exceeds?|>=?)\s*(\d+)',
        r'(?:less\s+than|under|below|<=?)\s*(\d+)',
        r'(?:at\s+least|minimum)\s+(\d+)',
        r'(?:at\s+most|maximum)\s+(\d+)',
        r'(?:between)\s+(\d+)\s+and\s+(\d+)',
    ]

    _PERCENTAGE_SIGNALS = frozenset({
        'percent', 'percentage', '%', 'proportion', 'share', 'fraction',
        'what percentage', 'what percent',
    })

    def __init__(self, concept_index: Dict[str, Dict[str, float]] = None,
                 domain_concepts: Dict = None, synonyms: Dict = None,
                 schema_learner=None):
        self.concept_index = concept_index or {}
        self.domain_concepts = domain_concepts or {}
        self.synonyms = synonyms or {}
        self.learner = schema_learner

        self._metric_concepts = self._build_metric_concepts()

    def _build_metric_concepts(self) -> Dict[str, List[str]]:
        metrics = defaultdict(list)

        if not self.learner:
            for term, cols in self.synonyms.items():
                for col in cols:
                    if any(kw in col.lower() for kw in
                           ['amount', 'cost', 'count', 'score', 'rate', 'size',
                            'days', 'quantity', 'duration', 'rvu', 'supply',
                            'copay', 'coinsurance', 'deductible', 'panel',
                            'refill', 'responsibility', 'length']):
                        metrics[term].append(col)
            return dict(metrics)

        numeric_cols = set()
        for tbl_name, profiles in self.learner.tables.items():
            for p in profiles:
                if p.is_numeric and not p.is_id and not p.name.lower().endswith('_id'):
                    numeric_cols.add(p.name.upper())

        for term, cols in self.synonyms.items():
            for col in cols:
                if col.upper() in numeric_cols:
                    metrics[term].append(col)

        for tbl_name, profiles in self.learner.tables.items():
            for p in profiles:
                if p.is_numeric and not p.is_id and not p.name.lower().endswith('_id'):
                    col_words = p.name.lower().replace('_', ' ').split()
                    for word in col_words:
                        if len(word) >= 3 and word not in ('the', 'and', 'for'):
                            if word not in metrics:
                                metrics[word] = []
                            if p.name not in metrics[word]:
                                metrics[word].append(p.name)
                    full_name = p.name.lower().replace('_', ' ')
                    if full_name not in metrics:
                        metrics[full_name] = []
                    if p.name not in metrics[full_name]:
                        metrics[full_name].append(p.name)

        return dict(metrics)

    def decompose(self, question: str) -> QueryPlan:
        q = question.lower().strip()
        plan = QueryPlan(raw_question=question)

        plan.intent = self._detect_intent(q)

        if plan.intent == 'lookup':
            plan.entity_focus = self._extract_entity_focus(q)

        plan.metrics = self._extract_metrics(q, plan.intent)

        plan.dimensions = self._extract_dimensions(q, plan.intent)

        plan.filters = self._extract_filters(q)

        plan.sort = self._extract_sort(q)

        plan.percentage = self._detect_percentage(q)
        plan.comparison_mode = self._detect_comparison_mode(q)

        plan = self._infer_missing(plan, q)

        plan.confidence = self._calculate_plan_confidence(plan)

        logger.info("QueryPlan: intent=%s, metrics=%d, dims=%d, filters=%d, conf=%.2f",
                     plan.intent, len(plan.metrics), len(plan.dimensions),
                     len(plan.filters), plan.confidence)

        return plan


    def _detect_intent(self, q: str) -> str:
        for pattern, intent in self._INTENT_PATTERNS:
            if re.search(pattern, q):
                return intent

        if '?' in q and re.search(r'\b(?:what|which|who|how)\b', q):
            return 'lookup'

        return 'aggregation'

    def _extract_entity_focus(self, q: str) -> Optional[str]:
        m = re.search(
            r'\b(?:show\s+(?:me\s+)?(?:all\s+)?(?:the\s+)?|'
            r'list\s+(?:all\s+)?(?:the\s+)?|'
            r'display\s+(?:all\s+)?(?:the\s+)?|'
            r'get\s+(?:all\s+)?(?:the\s+)?|'
            r'view\s+(?:all\s+)?(?:the\s+)?|'
            r'give\s+me\s+(?:all\s+)?(?:the\s+)?|'
            r'pull\s+(?:up\s+)?(?:all\s+)?(?:the\s+)?|'
            r'find\s+(?:all\s+)?(?:the\s+)?|'
            r'fetch\s+(?:all\s+)?(?:the\s+)?)'
            r'(\w+)', q)
        if m:
            return m.group(1).lower()
        return None


    def _extract_metrics(self, q: str, intent: str) -> List[MetricSpec]:
        metrics = []

        agg = 'count'
        for pattern, agg_type in self._AGG_PATTERNS:
            if re.search(pattern, q):
                agg = agg_type
                break

        rate_match = re.search(r'(\w+)\s+rate\b', q)
        if rate_match:
            target = rate_match.group(1)
            metrics.append(MetricSpec(
                concept=f'{target} rate',
                aggregation='rate',
                target=target,
            ))
            return metrics

        pct_match = re.search(
            r'(?:what\s+)?(?:percent|percentage|%)\s+(?:of\s+)?(\w+)\s+'
            r'(?:are|is|were|was|have|has)\s+(\w+)', q)
        if pct_match:
            population = pct_match.group(1)
            target = pct_match.group(2)
            metrics.append(MetricSpec(
                concept=f'percentage of {population} that are {target}',
                aggregation='rate',
                target=target,
                denominator=population,
            ))
            return metrics

        if agg == 'count':
            metrics.append(MetricSpec(concept='count', aggregation='count'))
            return metrics

        agg_metric_patterns = [
            (r'\b(?:average|avg|mean)\s+(\w+(?:\s+\w+)?)', 'avg'),
            (r'\b(?:total|sum\s+of|sum)\s+(\w+(?:\s+\w+)?)', 'sum'),
            (r'\b(?:max|maximum|highest)\s+(\w+(?:\s+\w+)?)', 'max'),
            (r'\b(?:min|minimum|lowest)\s+(\w+(?:\s+\w+)?)', 'min'),
        ]
        for pattern, agg_type in agg_metric_patterns:
            agg_metric_match = re.search(pattern, q)
            if agg_metric_match:
                concept = agg_metric_match.group(1).strip()
                if concept in ('by', 'per', 'across', 'for', 'of'):
                    continue
                concept = re.sub(r'\s+(?:by|per|across|for|of)\b.*', '', concept).strip()
                if concept and concept in self._metric_concepts:
                    metrics.append(MetricSpec(concept=concept, aggregation=agg_type))
                    return metrics
                if self.learner:
                    concept_col = concept.upper().replace(' ', '_')
                    for tbl_name, profiles in self.learner.tables.items():
                        for p in profiles:
                            if p.name == concept_col and p.is_numeric:
                                metrics.append(MetricSpec(concept=concept, aggregation=agg_type))
                                return metrics

        ratio_match = re.search(r'(\w+)\s+to\s+(\w+)\s+ratio', q)
        if ratio_match:
            metrics.append(MetricSpec(
                concept=f'{ratio_match.group(1)} to {ratio_match.group(2)} ratio',
                aggregation='ratio',
                target=ratio_match.group(1),
                denominator=ratio_match.group(2),
            ))
            return metrics

        q_words = tokenize_query(q)
        metric_candidates = []
        for word in q_words:
            if word in self._metric_concepts:
                metric_candidates.append(word)
        for term in self.synonyms:
            if ' ' in term and term in q:
                if term in self._metric_concepts:
                    metric_candidates.append(term)

        if metric_candidates:
            best = max(metric_candidates, key=len)
            metrics.append(MetricSpec(concept=best, aggregation=agg))
        elif intent in ('aggregation', 'trend', 'ranking', 'distribution'):
            metrics.append(MetricSpec(concept='count', aggregation='count'))

        if 'volume' in q and not any(m.aggregation == 'count' for m in metrics):
            metrics.append(MetricSpec(concept='volume', aggregation='count'))

        return metrics


    def _extract_dimensions(self, q: str, intent: str) -> List[DimensionSpec]:
        dims = []

        if intent == 'trend' or re.search(r'\bover\s+time\b|\btrend\b|\bmonthly\b', q):
            granularity = 'month'
            for pattern, gran in self._TEMPORAL_GRANULARITY:
                if re.search(pattern, q):
                    granularity = gran
                    break
            dims.append(DimensionSpec(
                concept='time', temporal=True, granularity=granularity))

        for pattern in self._DIMENSION_SIGNALS:
            m = re.search(pattern, q)
            if m:
                dim_term = m.group(1).strip()
                if dim_term in ('month', 'year', 'quarter', 'week', 'day',
                                'time', 'date', 'period'):
                    continue
                if dim_term in ('count', 'total', 'amount', 'cost', 'average'):
                    continue
                dims.append(DimensionSpec(concept=dim_term))
                break

        if intent == 'ranking' and not any(not d.temporal for d in dims):
            subject = self._extract_ranking_subject(q)
            if subject:
                dims.append(DimensionSpec(concept=subject))

        if intent == 'distribution' and not any(not d.temporal for d in dims):
            dist_target = self._extract_distribution_target(q)
            if dist_target:
                dims.append(DimensionSpec(concept=dist_target))

        return dims

    def _extract_ranking_subject(self, q: str) -> Optional[str]:
        patterns = [
            r'\bwhich\s+(\w+)',
            r'\btop\s+\d*\s*(\w+)',
            r'\b(\w+)\s+with\s+(?:the\s+)?(?:high|low|most|least)',
        ]
        for pat in patterns:
            m = re.search(pat, q)
            if m:
                subject = m.group(1).strip().rstrip('s')
                if subject not in ('the', 'are', 'is', 'has', 'have', 'do', 'does'):
                    return subject
        return None

    def _extract_distribution_target(self, q: str) -> Optional[str]:
        m = re.search(r'(?:breakdown|distribution|split)\s+(?:of\s+)?(\w+)', q)
        if m:
            return m.group(1).strip()
        return None


    def _extract_filters(self, q: str) -> List[FilterSpec]:
        filters = []

        q_lower = q.lower()
        q_words = set(re.findall(r'\b\w+\b', q_lower))

        sorted_concepts = sorted(self.domain_concepts.items(),
                                 key=lambda x: len(x[0]), reverse=True)
        matched_terms = set()

        _rate_match = re.search(r'(\w+)\s+rate\b', q_lower)
        _rate_target = _rate_match.group(1) if _rate_match else None

        for term, concept in sorted_concepts:
            t_lower = term.lower()

            if len(t_lower) <= 2:
                continue

            t_words = t_lower.split()
            if len(t_words) == 1:
                if t_lower not in q_words:
                    continue
            else:
                if t_lower not in q_lower:
                    continue

            if _rate_target and t_lower in (_rate_target, _rate_target + 'ed',
                                            _rate_target + 'd', _rate_target + 'ial',
                                            _rate_target + 'al'):
                continue

            overlap = False
            for mt in matched_terms:
                if t_lower in mt or mt in t_lower:
                    overlap = True
                    break
            if overlap:
                continue
            matched_terms.add(t_lower)
            conds = concept.get('conds', [])
            if conds:
                filters.append(FilterSpec(
                    concept=term,
                    operator='domain',
                    value=term,
                ))

        for pattern, period_type in self._TIME_FILTER_PATTERNS:
            m = re.search(pattern, q_lower)
            if m:
                if m.groups():
                    filters.append(FilterSpec(
                        concept='time',
                        temporal=True,
                        temporal_range=m.group(0),
                    ))
                else:
                    filters.append(FilterSpec(
                        concept='time',
                        temporal=True,
                        temporal_range=period_type,
                    ))
                break

        for pattern in self._THRESHOLD_PATTERNS:
            m = re.search(pattern, q_lower)
            if m:
                val = m.group(1)
                if 'more than' in q_lower or 'over' in q_lower or 'above' in q_lower or '>=' in q_lower:
                    op = '>='
                elif 'less than' in q_lower or 'under' in q_lower or 'below' in q_lower:
                    op = '<='
                elif 'between' in q_lower:
                    op = 'BETWEEN'
                else:
                    op = '>='
                filters.append(FilterSpec(
                    concept='threshold',
                    operator=op,
                    value=val,
                ))
                break

        return filters


    def _extract_sort(self, q: str) -> Optional[SortSpec]:
        sort = SortSpec()
        found = False

        for pattern, direction in self._SORT_PATTERNS:
            m = re.search(pattern, q)
            if m:
                sort.direction = direction
                if m.groups():
                    try:
                        sort.limit = int(m.group(1))
                    except (ValueError, IndexError):
                        pass
                found = True
                break

        return sort if found else None


    def _detect_percentage(self, q: str) -> bool:
        return any(sig in q for sig in self._PERCENTAGE_SIGNALS)

    def _detect_comparison_mode(self, q: str) -> Optional[str]:
        if re.search(r'\bvs\.?\b|\bversus\b', q):
            return 'vs'
        if re.search(r'\bcompare\b|\bcomparison\b', q):
            return 'compare'
        if re.search(r'\bdifference\b|\bdelta\b', q):
            return 'difference'
        return None


    def _infer_missing(self, plan: QueryPlan, q: str) -> QueryPlan:
        if plan.intent == 'trend' and not any(d.temporal for d in plan.dimensions):
            plan.dimensions.insert(0, DimensionSpec(
                concept='time', temporal=True, granularity='month'))

        if plan.intent == 'ranking' and not plan.sort:
            plan.sort = SortSpec(direction='desc', limit=20)

        if plan.dimensions and not plan.metrics:
            plan.metrics.append(MetricSpec(concept='count', aggregation='count'))

        if plan.intent == 'rate' and not any(m.aggregation == 'rate' for m in plan.metrics):
            rate_m = re.search(r'(\w+)\s+rate', q)
            if rate_m:
                plan.metrics.insert(0, MetricSpec(
                    concept=f'{rate_m.group(1)} rate',
                    aggregation='rate',
                    target=rate_m.group(1),
                ))

        if plan.percentage and not any(m.aggregation == 'rate' for m in plan.metrics):
            pct_m = re.search(r'(?:are|is|were|was)\s+(\w+)', q)
            if pct_m:
                plan.metrics.insert(0, MetricSpec(
                    concept=f'percentage {pct_m.group(1)}',
                    aggregation='rate',
                    target=pct_m.group(1),
                ))

        has_threshold = any(f.concept == 'threshold' for f in plan.filters)
        has_non_temporal_dim = any(not d.temporal for d in plan.dimensions)
        if has_threshold and not has_non_temporal_dim:
            entity_match = re.search(
                r'(\w+(?:s)?)\s+(?:with|having|where)\s+(?:more\s+than|over|at\s+least|greater\s+than|>=?)\s*\d+',
                q)
            if entity_match:
                entity = entity_match.group(1).rstrip('s')
                plan.dimensions.append(DimensionSpec(concept=entity))
            else:
                n_entity = re.search(r'\d+\s+(\w+)', q)
                if n_entity:
                    counted = n_entity.group(1).rstrip('s')
                    subject = re.search(r'^(\w+)', q)
                    if subject and subject.group(1).lower() not in (
                        'what', 'which', 'how', 'show', 'find', 'list', 'get'):
                        plan.dimensions.append(DimensionSpec(concept=subject.group(1).rstrip('s')))

        return plan


    def _calculate_plan_confidence(self, plan: QueryPlan) -> float:
        score = 0.0

        if plan.intent == 'lookup' and plan.entity_focus:
            score += 0.5
        elif plan.intent != 'lookup':
            score += 0.2

        if plan.metrics:
            score += 0.3

        if plan.dimensions or plan.filters:
            score += 0.2

        if any(m.concept != 'count' for m in plan.metrics):
            score += 0.15

        if any(d.temporal for d in plan.dimensions) or \
           any(f.temporal for f in plan.filters):
            score += 0.15

        return min(1.0, score)


@dataclass
class ResolvedMetric:
    table: str
    column: str
    aggregation: str
    expression: Optional[str] = None
    alias: str = ''

@dataclass
class ResolvedDimension:
    table: str
    column: str
    expression: Optional[str] = None
    alias: str = ''
    needs_join: bool = False
    join_on: Optional[str] = None

@dataclass
class ResolvedFilter:
    table: str
    condition: str

@dataclass
class ResolvedPlan:
    primary_table: str
    metrics: List[ResolvedMetric] = field(default_factory=list)
    dimensions: List[ResolvedDimension] = field(default_factory=list)
    filters: List[ResolvedFilter] = field(default_factory=list)
    join_tables: List[Tuple[str, str]] = field(default_factory=list)
    sort_direction: str = 'desc'
    sort_column: Optional[str] = None
    limit: Optional[int] = None
    percentage: bool = False
    confidence: float = 0.0
    intent: str = 'aggregation'
    raw_question: str = ''


class SchemaResolver:

    def __init__(self, schema_learner, concept_index: Dict,
                 domain_concepts: Dict = None, synonyms: Dict = None,
                 computed_columns: Dict = None, kg=None, schema_intel=None):
        self.learner = schema_learner
        self.concept_index = concept_index
        self.domain_concepts = domain_concepts or {}
        self.synonyms = synonyms or {}
        self.computed_columns = computed_columns or {}
        self.kg = kg
        self._schema_intel = schema_intel

        self._numeric_cols = {}
        self._categorical_cols = {}
        self._date_cols = {}
        self._id_cols = {}
        for tbl, profiles in self.learner.tables.items():
            self._numeric_cols[tbl] = {p.name: p for p in profiles if p.is_numeric}
            self._categorical_cols[tbl] = {p.name: p for p in profiles if p.is_categorical}
            self._date_cols[tbl] = {p.name: p for p in profiles if p.is_date}
            self._id_cols[tbl] = {p.name: p for p in profiles if p.name.endswith('_ID') or p.name == 'NPI' or p.name == 'MRN'}

    def resolve(self, plan: QueryPlan) -> ResolvedPlan:
        resolved = ResolvedPlan(
            primary_table='',
            confidence=plan.confidence,
            intent=plan.intent,
            percentage=plan.percentage,
            raw_question=plan.raw_question,
        )

        resolved.primary_table = self._resolve_primary_table(plan)

        resolved.metrics = self._resolve_metrics(plan, resolved.primary_table)

        resolved.dimensions = self._resolve_dimensions(plan, resolved.primary_table)

        resolved.filters = self._resolve_filters(plan, resolved.primary_table)

        resolved = self._reconsider_primary_table(resolved, plan)

        resolved.join_tables = self._resolve_joins(resolved)

        if plan.sort:
            resolved.sort_direction = plan.sort.direction
            resolved.limit = plan.sort.limit
        elif plan.intent == 'ranking':
            resolved.sort_direction = 'desc'
            resolved.limit = 20

        resolved = self._validate_and_fix_tables(resolved)

        logger.info("ResolvedPlan: table=%s, metrics=%d, dims=%d, filters=%d, joins=%d",
                     resolved.primary_table, len(resolved.metrics),
                     len(resolved.dimensions), len(resolved.filters),
                     len(resolved.join_tables))

        return resolved

    def _resolve_primary_table(self, plan: QueryPlan) -> str:
        if plan.entity_focus:
            entity = plan.entity_focus.lower()
            for tbl in self.learner.tables:
                tbl_lower = tbl.lower()
                if (entity == tbl_lower or
                    entity.rstrip('s') == tbl_lower.rstrip('s') or
                    entity == tbl_lower.rstrip('s') or
                    tbl_lower == entity.rstrip('s')):
                    logger.info("Entity focus '%s' → table %s", entity, tbl)
                    return tbl

            if entity in self.concept_index:
                best_tbl = max(self.concept_index[entity].items(),
                               key=lambda x: x[1], default=(None, 0))
                if best_tbl[0]:
                    logger.info("Entity focus '%s' → table %s (via concept_index)",
                                entity, best_tbl[0])
                    return best_tbl[0]

        _INTERNAL_TABLE_PREFIXES = ('gpdm_', '_gpdm_', '_dq_', '_schema_', '_data_', '_audit_',
                                     'sqlite_', 'query_patterns')
        table_scores = defaultdict(float)
        q_lower = plan.raw_question.lower()
        q_words = set(re.findall(r'[a-z][a-z0-9_]*', q_lower))

        explicit_matches = []
        for tbl in self.learner.tables:
            if tbl.lower() in q_lower:
                explicit_matches.append((tbl, len(tbl)))
                table_scores[tbl] += 100
        if len(explicit_matches) > 1:
            explicit_matches.sort(key=lambda x: x[1], reverse=True)
            longest = explicit_matches[0][0]
            table_scores[longest] += 200

        for word in q_words:
            for tbl in self.learner.tables:
                tbl_lower = tbl.lower()
                if (word == tbl_lower or
                    word.rstrip('s') == tbl_lower.rstrip('s') or
                    word == tbl_lower.rstrip('s')):
                    table_scores[tbl] += 50

        for word in q_words:
            if word in self.concept_index:
                for tbl, score in self.concept_index[word].items():
                    table_scores[tbl] += score

        for metric in plan.metrics:
            concept_words = tokenize_query(metric.concept)
            for word in concept_words:
                if word in self.synonyms:
                    for col in self.synonyms[word]:
                        for tbl in self.learner.tables:
                            if any(p.name == col for p in self.learner.tables[tbl]):
                                table_scores[tbl] += 5

        for dim in plan.dimensions:
            if not dim.temporal:
                concept_words = tokenize_query(dim.concept)
                for word in concept_words:
                    if word in self.concept_index:
                        for tbl, score in self.concept_index[word].items():
                            table_scores[tbl] += score * 0.5

        for filt in plan.filters:
            if filt.concept in self.domain_concepts:
                for tbl in self.domain_concepts[filt.concept].get('tables', []):
                    table_scores[tbl] += 10

        for tbl in list(table_scores.keys()):
            if any(tbl.lower().startswith(p) for p in _INTERNAL_TABLE_PREFIXES):
                del table_scores[tbl]

        if not table_scores:
            real_tables = {t: c for t, c in self.learner.table_row_counts.items()
                          if not any(t.lower().startswith(p) for p in _INTERNAL_TABLE_PREFIXES)}
            return max(real_tables, key=real_tables.get) if real_tables else 'claims'

        return max(table_scores, key=lambda t: (table_scores[t],
                   self.learner.table_row_counts.get(t, 0)))

    def _reconsider_primary_table(self, resolved: 'ResolvedPlan', plan: 'QueryPlan') -> 'ResolvedPlan':
        if not resolved.dimensions:
            logger.debug("_reconsider: no dimensions, skipping")
            return resolved

        dim_tables = set()
        all_dims_external = True
        for dim in resolved.dimensions:
            logger.info("_reconsider: dim col=%s table=%s (primary=%s)",
                       dim.column, dim.table, resolved.primary_table)
            if dim.table == resolved.primary_table:
                all_dims_external = False
                break
            if dim.table:
                dim_tables.add(dim.table)

        if not all_dims_external or len(dim_tables) != 1:
            logger.info("_reconsider: dims_external=%s, dim_tables=%s — not switching",
                       all_dims_external, dim_tables)
            return resolved

        candidate_table = dim_tables.pop()

        _INTERNAL_PREFIXES = ('gpdm_', '_gpdm_', '_dq_', '_schema_', '_data_', '_audit_',
                               'sqlite_', 'query_patterns')
        if any(candidate_table.lower().startswith(p) for p in _INTERNAL_PREFIXES):
            logger.info("_reconsider: candidate %s is internal table — not switching", candidate_table)
            return resolved

        metrics_compatible = True
        for m in resolved.metrics:
            logger.info("_reconsider: metric col=%s agg=%s table=%s", m.column, m.aggregation, m.table)
            if m.aggregation == 'COUNT' and m.column == '*':
                continue
            candidate_cols = {p.name.lower() for p in self.learner.tables.get(candidate_table, [])}
            if m.column.lower() not in candidate_cols:
                logger.info("_reconsider: metric %s NOT in %s cols, not switching", m.column, candidate_table)
                metrics_compatible = False
                break

        if not metrics_compatible:
            return resolved

        primary_rows = self.learner.table_row_counts.get(resolved.primary_table, 0)
        candidate_rows = self.learner.table_row_counts.get(candidate_table, 0)

        if primary_rows > 10000 and candidate_rows < primary_rows * 5:
            old_primary = resolved.primary_table
            resolved.primary_table = candidate_table

            new_metrics = []
            for m in resolved.metrics:
                if m.column == '*':
                    new_metrics.append(ResolvedMetric(
                        column=m.column, aggregation=m.aggregation,
                        table=candidate_table, alias=m.alias,
                    ))
                elif m.table == old_primary:
                    candidate_cols = {p.name for p in self.learner.tables.get(candidate_table, [])}
                    if m.column in candidate_cols:
                        new_metrics.append(ResolvedMetric(
                            column=m.column, aggregation=m.aggregation,
                            table=candidate_table, alias=m.alias,
                        ))
                    else:
                        new_metrics.append(m)
                else:
                    new_metrics.append(m)
            resolved.metrics = new_metrics

            resolved._switched_from_table = old_primary

            logger.info("Primary table switched: %s → %s (avoiding expensive JOIN, "
                       "%d×%d rows → %d rows)", old_primary, candidate_table,
                       primary_rows, candidate_rows, candidate_rows)

        return resolved

    def _resolve_metrics(self, plan: QueryPlan, primary_table: str) -> List[ResolvedMetric]:
        resolved = []

        for metric in plan.metrics:
            if metric.aggregation == 'rate':
                rm = self._resolve_rate_metric(metric, primary_table)
                if rm:
                    resolved.append(rm)
                continue

            if metric.aggregation == 'count' and metric.concept in ('count', 'volume'):
                resolved.append(ResolvedMetric(
                    table=primary_table,
                    column='*',
                    aggregation='COUNT',
                    alias='record_count',
                ))
                continue

            if metric.aggregation == 'count_distinct':
                col = self._find_id_column(primary_table, metric.concept)
                if col:
                    resolved.append(ResolvedMetric(
                        table=primary_table,
                        column=col,
                        aggregation='COUNT_DISTINCT',
                        alias=f'unique_{col.lower()}',
                    ))
                continue

            col, col_table = self._find_metric_column(metric.concept, primary_table)
            if col:
                agg_sql = {'avg': 'AVG', 'sum': 'SUM', 'count': 'COUNT',
                           'min': 'MIN', 'max': 'MAX'}.get(metric.aggregation, 'AVG')
                resolved.append(ResolvedMetric(
                    table=col_table,
                    column=col,
                    aggregation=agg_sql,
                    alias=f'{metric.aggregation}_{col.lower()}',
                ))

        if not resolved:
            resolved.append(ResolvedMetric(
                table=primary_table,
                column='*',
                aggregation='COUNT',
                alias='record_count',
            ))

        return resolved

    def _resolve_rate_metric(self, metric: MetricSpec, primary_table: str) -> Optional[ResolvedMetric]:
        target = metric.target
        if not target:
            return None

        target_lower = target.lower()
        target_upper = target.upper()

        _morphs = {target_lower}
        _morphs.add(metric.concept.lower())
        if target_lower.endswith('al'):
            _morphs.add(target_lower[:-2] + 'ed')
        elif target_lower.endswith('ion'):
            _morphs.add(target_lower[:-3] + 'ted')
            _morphs.add(target_lower[:-3] + 'ed')
        elif target_lower.endswith('ment'):
            _morphs.add(target_lower[:-4] + 'ed')
        else:
            _morphs.add(f'{target_lower}ed')
            _morphs.add(f'{target_lower}d')
        for concept_key in _morphs:
            if concept_key in self.domain_concepts:
                dc = self.domain_concepts[concept_key]
                conds = dc.get('conds', [])
                tables = dc.get('tables', [])
                if conds:
                    numerator_cond = conds[0]
                    tbl = tables[0] if tables and tables[0] in self.learner.tables else primary_table
                    expr = (f"ROUND(100.0 * SUM(CASE WHEN {numerator_cond} "
                            f"THEN 1 ELSE 0 END) / COUNT(*), 2)")
                    return ResolvedMetric(
                        table=tbl,
                        column=numerator_cond.split('=')[0].strip() if '=' in numerator_cond else '*',
                        aggregation='RATE',
                        expression=expr,
                        alias=f'{target_lower}_rate_pct',
                    )

        rate_keys = [f'{target_lower}_rate', f'{target_lower}ion_rate', metric.concept.replace(' ', '_')]
        for rk in rate_keys:
            if rk in self.computed_columns:
                cc = self.computed_columns[rk]
                return ResolvedMetric(
                    table=cc.get('table', primary_table),
                    column=rk,
                    aggregation='RATE',
                    expression=cc['expr'],
                    alias=cc.get('alias', f'{target_lower}_rate_pct'),
                )

        db_path = getattr(self.learner, 'db_path', None)
        if db_path:
            import sqlite3 as _sq3
            try:
                _conn = _sq3.connect(db_path)
                _cur = _conn.cursor()
                for tbl_name in [primary_table] + list(self.learner.tables.keys()):
                    for p in self.learner.tables.get(tbl_name, []):
                        if not p.is_categorical:
                            continue
                        col_lower = p.name.lower()
                        if not any(kw in col_lower for kw in ['status', 'type', 'disposition',
                                                               'result', 'outcome', 'flag']):
                            continue
                        _cur.execute(f"SELECT DISTINCT {p.name} FROM {tbl_name} LIMIT 50")
                        distinct_vals = [r[0] for r in _cur.fetchall() if r[0]]
                        for val in distinct_vals:
                            val_str = str(val).upper()
                            if target_upper in val_str or val_str.startswith(target_upper[:4]):
                                expr = (f"ROUND(100.0 * SUM(CASE WHEN {p.name} = '{val}' "
                                        f"THEN 1 ELSE 0 END) / COUNT(*), 2)")
                                _conn.close()
                                return ResolvedMetric(
                                    table=tbl_name,
                                    column=p.name,
                                    aggregation='RATE',
                                    expression=expr,
                                    alias=f'{target_lower}_rate_pct',
                                )
                _conn.close()
            except Exception as e:
                logger.debug("Rate metric DB lookup failed: %s", e)

        for tbl_name in [primary_table] + list(self.learner.tables.keys()):
            for p in self.learner.tables.get(tbl_name, []):
                if p.is_categorical and p.sample_values:
                    for val in p.sample_values:
                        if target_upper in str(val).upper() or \
                           str(val).upper().startswith(target_upper[:4]):
                            expr = (f"ROUND(100.0 * SUM(CASE WHEN {p.name} = '{val}' "
                                    f"THEN 1 ELSE 0 END) / COUNT(*), 2)")
                            return ResolvedMetric(
                                table=tbl_name,
                                column=p.name,
                                aggregation='RATE',
                                expression=expr,
                                alias=f'{target_lower}_rate_pct',
                            )

        return None

    def _find_metric_column(self, concept: str, primary_table: str) -> Tuple[Optional[str], str]:
        concept_words = tokenize_query(concept)

        schema_columns = {tbl: [p.name for p in self.learner.tables.get(tbl, [])]
                         for tbl in self.learner.tables}
        exact_cols = get_mentioned_columns(concept, schema_columns, target_table=primary_table)
        for match in exact_cols:
            col_name = match.name
            if col_name in self._numeric_cols.get(match.source_table, {}):
                return col_name, match.source_table

        if not exact_cols:
            exact_cols_all = get_mentioned_columns(concept, schema_columns)
            for match in exact_cols_all:
                col_name = match.name
                if col_name in self._numeric_cols.get(match.source_table, {}):
                    return col_name, match.source_table

        for word in concept_words:
            if word in self.synonyms:
                for col_name in self.synonyms[word]:
                    if col_name in self._numeric_cols.get(primary_table, {}):
                        return col_name, primary_table
                    for tbl in self.learner.tables:
                        if col_name in self._numeric_cols.get(tbl, {}):
                            return col_name, tbl

        best_col, best_table, best_score = None, primary_table, 0
        for tbl_name, num_cols in self._numeric_cols.items():
            for col_name in num_cols:
                col_words = set(col_name.lower().replace('_', ' ').split())
                overlap = len(concept_words & col_words)
                tbl_bonus = 2 if tbl_name == primary_table else 0
                score = overlap + tbl_bonus
                if score > best_score:
                    best_score = score
                    best_col = col_name
                    best_table = tbl_name

        if best_col:
            return best_col, best_table

        if self._schema_intel:
            result = self._schema_intel.find_metric(concept, preferred_table=primary_table)
            if result:
                logger.info("SchemaResolver: semantic fallback found metric '%s' → %s.%s",
                            concept, result[0], result[1])
                return result[1], result[0]

        if self._numeric_cols.get(primary_table):
            first = next(iter(self._numeric_cols[primary_table]))
            return first, primary_table

        return None, primary_table

    def _find_id_column(self, table: str, concept: str) -> Optional[str]:
        concept_lower = concept.lower()
        ids = self._id_cols.get(table, {})
        for col_name in ids:
            if concept_lower in col_name.lower():
                return col_name
        for col_name in ids:
            if col_name == 'MEMBER_ID':
                return col_name
        return next(iter(ids), None) if ids else None

    def _resolve_dimensions(self, plan: QueryPlan, primary_table: str) -> List[ResolvedDimension]:
        resolved = []

        for dim in plan.dimensions:
            if dim.temporal:
                rd = self._resolve_temporal_dimension(dim, primary_table)
                if rd:
                    resolved.append(rd)
            else:
                rd = self._resolve_categorical_dimension(dim, primary_table)
                if rd:
                    resolved.append(rd)

        return resolved

    def _resolve_temporal_dimension(self, dim: DimensionSpec,
                                     primary_table: str) -> Optional[ResolvedDimension]:
        try:
            from dynamic_sql_engine import TABLE_DATE_COL
            date_col = TABLE_DATE_COL.get(primary_table)
        except ImportError:
            date_col = None

        date_table = primary_table

        if not date_col:
            date_cols = self._date_cols.get(primary_table, {})
            if date_cols:
                for preferred in ['SERVICE_DATE', 'ADMIT_DATE', 'ENROLLMENT_DATE',
                                  'PRESCRIPTION_DATE', 'APPOINTMENT_DATE']:
                    if preferred in date_cols:
                        date_col = preferred
                        break
                if not date_col:
                    date_col = next(iter(date_cols))

        if not date_col:
            for tbl in self.learner.tables:
                if tbl == primary_table:
                    continue
                tbl_dates = self._date_cols.get(tbl, {})
                if tbl_dates:
                    date_col = next(iter(tbl_dates))
                    date_table = tbl
                    break

        if not date_col:
            return None

        tbl_alias = date_table[0]
        qualified_date = f"{tbl_alias}.{date_col}"

        gran = dim.granularity or 'month'
        if gran == 'month':
            expr = f"SUBSTR({qualified_date}, 1, 7)"
        elif gran == 'quarter':
            expr = (f"SUBSTR({qualified_date}, 1, 4) || '-Q' || "
                    f"CAST((CAST(SUBSTR({qualified_date}, 6, 2) AS INTEGER) + 2) / 3 AS TEXT)")
        elif gran == 'year':
            expr = f"SUBSTR({qualified_date}, 1, 4)"
        elif gran == 'week':
            expr = f"SUBSTR({qualified_date}, 1, 10)"
        elif gran == 'day':
            expr = f"SUBSTR({qualified_date}, 1, 10)"
        else:
            expr = f"SUBSTR({qualified_date}, 1, 7)"

        return ResolvedDimension(
            table=primary_table,
            column=date_col,
            expression=expr,
            alias='period',
        )

    def _resolve_categorical_dimension(self, dim: DimensionSpec,
                                        primary_table: str) -> Optional[ResolvedDimension]:
        concept = dim.concept.lower().strip()

        schema_columns = {tbl: [p.name for p in self.learner.tables.get(tbl, [])]
                         for tbl in self.learner.tables}

        exact_matches = get_mentioned_columns(dim.concept, schema_columns, target_table=primary_table)
        for match in exact_matches:
            for tbl_name in [match.source_table]:
                for p in self.learner.tables.get(tbl_name, []):
                    if p.name == match.name:
                        return ResolvedDimension(
                            table=tbl_name,
                            column=p.name,
                            alias=p.name.lower(),
                            needs_join=(tbl_name != primary_table),
                        )

        if not exact_matches:
            exact_matches_all = get_mentioned_columns(dim.concept, schema_columns)
            for match in exact_matches_all:
                for tbl_name in [match.source_table]:
                    for p in self.learner.tables.get(tbl_name, []):
                        if p.name == match.name:
                            return ResolvedDimension(
                                table=tbl_name,
                                column=p.name,
                                alias=p.name.lower(),
                                needs_join=(tbl_name != primary_table),
                            )

        concept_words = tokenize_query(concept)

        for comp_name, comp_def in self.computed_columns.items():
            comp_words = set(comp_name.lower().replace('_', ' ').split())
            if concept_words & comp_words:
                return ResolvedDimension(
                    table=comp_def.get('table', primary_table),
                    column=comp_name,
                    expression=comp_def['expr'],
                    alias=comp_def.get('alias', comp_name),
                    needs_join=(comp_def.get('table', primary_table) != primary_table),
                )

        for word in [concept] + list(concept_words):
            if word in self.synonyms:
                for col_name in self.synonyms[word]:
                    if col_name in self._categorical_cols.get(primary_table, {}):
                        return ResolvedDimension(
                            table=primary_table,
                            column=col_name,
                            alias=col_name.lower(),
                        )
                    for tbl in self.learner.tables:
                        if col_name in self._categorical_cols.get(tbl, {}):
                            return ResolvedDimension(
                                table=tbl,
                                column=col_name,
                                alias=col_name.lower(),
                                needs_join=(tbl != primary_table),
                            )
                    for tbl in self.learner.tables:
                        if col_name in self._id_cols.get(tbl, {}):
                            return ResolvedDimension(
                                table=tbl,
                                column=col_name,
                                alias=col_name.lower(),
                                needs_join=(tbl != primary_table),
                            )

        best_col, best_table, best_score = None, primary_table, 0
        for tbl_name, cat_cols in self._categorical_cols.items():
            for col_name in cat_cols:
                col_words = set(col_name.lower().replace('_', ' ').split())
                overlap = len(concept_words & col_words)
                tbl_bonus = 2 if tbl_name == primary_table else 0
                score = overlap + tbl_bonus
                if score > best_score:
                    best_score = score
                    best_col = col_name
                    best_table = tbl_name

        if best_col and best_score > 0:
            return ResolvedDimension(
                table=best_table,
                column=best_col,
                alias=best_col.lower(),
                needs_join=(best_table != primary_table),
            )

        if self._schema_intel:
            result = self._schema_intel.find_dimension(concept, preferred_table=primary_table)
            if result:
                logger.info("SchemaResolver: semantic fallback found dimension '%s' → %s.%s",
                            concept, result[0], result[1])
                return ResolvedDimension(
                    table=result[0],
                    column=result[1],
                    alias=result[1].lower(),
                    needs_join=(result[0] != primary_table),
                )

        return None

    def _resolve_filters(self, plan: QueryPlan, primary_table: str) -> List[ResolvedFilter]:
        resolved = []

        for filt in plan.filters:
            if filt.temporal:
                rf = self._resolve_temporal_filter(filt, primary_table)
                if rf:
                    resolved.append(rf)
            elif filt.operator == 'domain':
                rfs = self._resolve_domain_filter(filt, primary_table)
                resolved.extend(rfs)
            elif filt.concept == 'threshold':
                resolved.append(ResolvedFilter(
                    table=primary_table,
                    condition=f'__THRESHOLD__{filt.operator}__{filt.value}',
                ))
            else:
                rf = self._resolve_generic_filter(filt, primary_table)
                if rf:
                    resolved.append(rf)

        return resolved

    def _resolve_temporal_filter(self, filt: FilterSpec, primary_table: str) -> Optional[ResolvedFilter]:
        try:
            from dynamic_sql_engine import TABLE_DATE_COL, TIME_FILTERS
        except ImportError:
            return None

        date_col = TABLE_DATE_COL.get(primary_table, 'SERVICE_DATE')
        time_range = filt.temporal_range

        if time_range in TIME_FILTERS:
            val = TIME_FILTERS[time_range]
            if isinstance(val, tuple):
                return ResolvedFilter(
                    table=primary_table,
                    condition=f"{date_col} BETWEEN {val[0]} AND {val[1]}",
                )
            return ResolvedFilter(
                table=primary_table,
                condition=f"{date_col} >= {val}",
            )

        m = re.match(r'last\s+(\d+)\s+(months?|years?|days?|weeks?|quarters?)', time_range or '')
        if m:
            n = int(m.group(1))
            unit = m.group(2).rstrip('s')
            if unit == 'quarter':
                n *= 3
                unit = 'month'
            return ResolvedFilter(
                table=primary_table,
                condition=f"{date_col} >= date('now', '-{n} {unit}s')",
            )

        return None

    def _resolve_domain_filter(self, filt: FilterSpec, primary_table: str) -> List[ResolvedFilter]:
        concept = filt.concept
        if concept not in self.domain_concepts:
            concept = concept.lower()
        if concept not in self.domain_concepts:
            return []

        conds = self.domain_concepts[concept].get('conds', [])
        tables = self.domain_concepts[concept].get('tables', [])

        tbl = primary_table
        for t in tables:
            if t in self.learner.tables:
                tbl = t
                break

        return [ResolvedFilter(table=tbl, condition=c) for c in conds]

    def _resolve_generic_filter(self, filt: FilterSpec, primary_table: str) -> Optional[ResolvedFilter]:
        concept = filt.concept.lower()
        value = filt.value

        for tbl_name in [primary_table] + list(self.learner.tables.keys()):
            for p in self.learner.tables.get(tbl_name, []):
                if p.is_categorical and p.sample_values:
                    for sv in p.sample_values:
                        if concept in str(sv).lower() or str(sv).lower() in concept:
                            return ResolvedFilter(
                                table=tbl_name,
                                condition=f"{p.name} = '{sv}'",
                            )

        return None

    def _resolve_joins(self, resolved: ResolvedPlan) -> List[Tuple[str, str]]:
        needed_tables = set()
        needed_tables.add(resolved.primary_table)

        for m in resolved.metrics:
            needed_tables.add(m.table)
        for d in resolved.dimensions:
            if d.needs_join:
                needed_tables.add(d.table)
        for f in resolved.filters:
            needed_tables.add(f.table)

        _switched_from = getattr(resolved, '_switched_from_table', None)
        if resolved.raw_question:
            from tokenizer_utils import get_mentioned_tables
            mentioned = get_mentioned_tables(resolved.raw_question, list(self.learner.tables.keys()))
            for tbl_name, _ in mentioned:
                if tbl_name != resolved.primary_table and tbl_name != _switched_from:
                    jc = self._find_join_condition(resolved.primary_table, tbl_name)
                    if jc:
                        needed_tables.add(tbl_name)

        join_tables = needed_tables - {resolved.primary_table}
        joins = []

        for tbl in join_tables:
            join_cond = self._find_join_condition(resolved.primary_table, tbl)
            if join_cond:
                joins.append((tbl, join_cond))

        return joins

    def _find_join_condition(self, tbl_a: str, tbl_b: str) -> Optional[str]:
        if self.kg:
            jc = self.kg._get_join_condition(tbl_a, tbl_b)
            if jc:
                return jc

        cols_a = {p.name for p in self.learner.tables.get(tbl_a, [])}
        cols_b = {p.name for p in self.learner.tables.get(tbl_b, [])}
        shared_ids = sorted(c for c in cols_a & cols_b if c.endswith('_ID') or c in ('NPI', 'MRN'))
        if shared_ids:
            a_stem = tbl_a.rstrip('s').upper()
            b_stem = tbl_b.rstrip('s').upper()
            for sid in shared_ids:
                if sid.upper().startswith(a_stem) or sid.upper().startswith(b_stem):
                    return f"{tbl_a}.{sid} = {tbl_b}.{sid}"
            return f"{tbl_a}.{shared_ids[0]} = {tbl_b}.{shared_ids[0]}"

        join_col = self.learner.join_graph.get(tbl_a, {}).get(tbl_b, '')
        if join_col:
            if '=' in join_col:
                return join_col
            return f"{tbl_a}.{join_col} = {tbl_b}.{join_col}"
        join_col = self.learner.join_graph.get(tbl_b, {}).get(tbl_a, '')
        if join_col:
            if '=' in join_col:
                left, right = join_col.split('=', 1)
                return f"{right.strip()} = {left.strip()}"
            return f"{tbl_a}.{join_col} = {tbl_b}.{join_col}"

        shared = sorted(cols_a & cols_b)
        if shared:
            strong = [c for c in shared if not c.upper().startswith(('STATUS', 'TYPE', 'NAME', 'DATE'))]
            if strong:
                return f"{tbl_a}.{strong[0]} = {tbl_b}.{strong[0]}"

        return None

    def _validate_and_fix_tables(self, resolved: ResolvedPlan) -> ResolvedPlan:
        for metric in resolved.metrics:
            if metric.column == '*':
                continue
            primary_cols = {p.name for p in self.learner.tables.get(resolved.primary_table, [])}
            if metric.column not in primary_cols:
                for tbl, profiles in self.learner.tables.items():
                    if any(p.name == metric.column for p in profiles):
                        if tbl != resolved.primary_table:
                            metric.table = tbl
                            if not any(t == tbl for t, _ in resolved.join_tables):
                                join_cond = self._find_join_condition(resolved.primary_table, tbl)
                                if join_cond:
                                    resolved.join_tables.append((tbl, join_cond))
                        break

        return resolved


class SQLComposer:

    def __init__(self, table_row_counts: Dict[str, int] = None):
        self.table_row_counts = table_row_counts or {}

    def compose(self, plan: ResolvedPlan) -> Dict[str, Any]:
        if (plan.intent == 'lookup' and
            not plan.dimensions and
            (not plan.metrics or
             (len(plan.metrics) == 1 and plan.metrics[0].aggregation == 'COUNT'
              and plan.metrics[0].column == '*'))):
            tbl = plan.primary_table
            limit = plan.limit or 50
            where_parts = []
            for f in plan.filters:
                if f.expression:
                    where_parts.append(f.expression)
            sql = f"SELECT * FROM {tbl}"
            if where_parts:
                sql += f" WHERE {' AND '.join(where_parts)}"
            sql += f" LIMIT {limit};"
            return {
                'sql': sql,
                'tables_used': [tbl],
                'confidence': max(0.7, plan.confidence),
                'intent': 'lookup',
            }

        aliases = self._assign_aliases(plan)

        select_parts = self._build_select(plan, aliases)

        from_clause = self._build_from(plan, aliases)

        where_parts, having_parts = self._build_conditions(plan, aliases)

        group_parts = self._build_group_by(plan, aliases)

        order_clause = self._build_order_by(plan, aliases, select_parts)

        limit = plan.limit or (50 if not group_parts else 30)
        limit_clause = f" LIMIT {limit}"

        if having_parts and not group_parts:
            for dim in plan.dimensions:
                if dim.expression:
                    group_parts.append(dim.alias)
                elif dim.column:
                    multi_table = len(aliases) > 1
                    col = self._qualify(dim.column, dim.table, aliases) if multi_table else dim.column
                    group_parts.append(col)
            if not group_parts:
                for sp in select_parts:
                    if not any(agg in sp.upper() for agg in ['COUNT(', 'SUM(', 'AVG(', 'MIN(', 'MAX(', 'ROUND(']):
                        group_parts.append(sp.split(' as ')[0].strip())
                        break

        sql = f"SELECT {', '.join(select_parts)}"
        sql += f" FROM {from_clause}"
        if where_parts:
            sql += f" WHERE {' AND '.join(where_parts)}"
        if group_parts:
            sql += f" GROUP BY {', '.join(group_parts)}"
        if having_parts:
            sql += f" HAVING {' AND '.join(having_parts)}"
        sql += order_clause
        sql += limit_clause + ";"

        tables_used = [plan.primary_table] + [t for t, _ in plan.join_tables]

        return {
            'sql': sql,
            'tables_used': list(dict.fromkeys(tables_used)),
            'confidence': max(0.6, plan.confidence),
            'intent': plan.intent,
        }

    def _assign_aliases(self, plan: ResolvedPlan) -> Dict[str, str]:
        aliases = {plan.primary_table: plan.primary_table[0]}
        used = {plan.primary_table[0]}
        for tbl, _ in plan.join_tables:
            alias = tbl[0]
            if alias in used:
                alias = tbl[:2]
            if alias in used:
                alias = tbl[:3]
            aliases[tbl] = alias
            used.add(alias)
        return aliases

    def _qualify(self, col: str, table: str, aliases: Dict[str, str]) -> str:
        alias = aliases.get(table, table[0] if table else '')
        if '(' in col or '.' in col or ' ' in col:
            return col
        return f"{alias}.{col}"

    def _build_select(self, plan: ResolvedPlan, aliases: Dict) -> List[str]:
        parts = []

        for dim in plan.dimensions:
            if dim.expression:
                parts.append(f"{dim.expression} as {dim.alias}")
            else:
                col = self._qualify(dim.column, dim.table, aliases)
                parts.append(col)

        for metric in plan.metrics:
            if metric.expression:
                parts.append(f"{metric.expression} as {metric.alias}")
            elif metric.aggregation == 'COUNT' and metric.column == '*':
                parts.append(f"COUNT(*) as {metric.alias}")
            elif metric.aggregation == 'COUNT_DISTINCT':
                col = self._qualify(metric.column, metric.table, aliases)
                parts.append(f"COUNT(DISTINCT {col}) as {metric.alias}")
            else:
                col = self._qualify(metric.column, metric.table, aliases)
                cast_col = f"CAST({col} AS REAL)"
                parts.append(f"ROUND({metric.aggregation}({cast_col}), 2) as {metric.alias}")

        if plan.metrics and not any(m.column == '*' for m in plan.metrics):
            parts.append("COUNT(*) as record_count")

        if plan.percentage and plan.dimensions:
            total = self.table_row_counts.get(plan.primary_table, 1)
            parts.append(f"ROUND(100.0 * COUNT(*) / {total}, 2) as pct")

        return parts

    def _build_from(self, plan: ResolvedPlan, aliases: Dict) -> str:
        primary_alias = aliases.get(plan.primary_table, plan.primary_table[0])
        from_parts = f"{plan.primary_table} {primary_alias}"

        for tbl, join_cond in plan.join_tables:
            tbl_alias = aliases.get(tbl, tbl[0])

            aliased_cond = join_cond
            for full_name, alias in aliases.items():
                aliased_cond = aliased_cond.replace(f"{full_name}.", f"{alias}.")

            if '=' not in aliased_cond and '.' not in aliased_cond:
                col = aliased_cond.strip()
                aliased_cond = f"{primary_alias}.{col} = {tbl_alias}.{col}"
            elif '=' in aliased_cond and '.' not in aliased_cond:
                parts = aliased_cond.split('=')
                if len(parts) == 2:
                    left = parts[0].strip()
                    right = parts[1].strip()
                    aliased_cond = f"{primary_alias}.{left} = {tbl_alias}.{right}"

            from_parts += f" JOIN {tbl} {tbl_alias} ON {aliased_cond}"

        return from_parts

    def _build_conditions(self, plan: ResolvedPlan, aliases: Dict) -> Tuple[List[str], List[str]]:
        where_parts = []
        having_parts = []

        for f in plan.filters:
            cond = f.condition
            if cond.startswith('__THRESHOLD__'):
                parts = cond.split('__')
                op = parts[2]
                val = parts[3]
                having_parts.append(f"COUNT(*) {op} {val}")
            else:
                alias = aliases.get(f.table, f.table[0])
                cond = self._qualify_condition(cond, alias)
                where_parts.append(cond)

        return where_parts, having_parts

    def _qualify_condition(self, condition: str, alias: str) -> str:
        if '.' in condition.split('=')[0].split('>')[0].split('<')[0]:
            return condition

        parts = re.split(r"('(?:[^'\\]|\\.)*')", condition)
        result_parts = []
        for i, part in enumerate(parts):
            if i % 2 == 1:
                result_parts.append(part)
            else:
                def _add_alias(m):
                    col = m.group(0)
                    if col in ('AND', 'OR', 'NOT', 'NULL', 'LIKE', 'IN', 'IS', 'BETWEEN',
                               'THEN', 'ELSE', 'END', 'WHEN', 'CASE', 'AS', 'REAL', 'INTEGER',
                               'TEXT', 'CAST', 'ROUND', 'SUM', 'AVG', 'COUNT', 'MAX', 'MIN',
                               'SELECT', 'FROM', 'WHERE', 'GROUP', 'ORDER', 'LIMIT', 'JOIN',
                               'SUBSTR', 'COALESCE', 'NULLIF', 'HAVING', 'DESC', 'ASC'):
                        return col
                    return f"{alias}.{col}"
                result_parts.append(re.sub(r'\b([A-Z][A-Z_]+(?:_[A-Z]+)*)\b', _add_alias, part))
        return ''.join(result_parts)

    def _build_group_by(self, plan: ResolvedPlan, aliases: Dict) -> List[str]:
        if not plan.dimensions and not any(
            m.aggregation in ('AVG', 'SUM', 'COUNT', 'MIN', 'MAX', 'COUNT_DISTINCT', 'RATE')
            for m in plan.metrics
        ):
            return []

        parts = []
        for dim in plan.dimensions:
            if dim.expression:
                parts.append(dim.alias)
            else:
                col = self._qualify(dim.column, dim.table, aliases)
                parts.append(col)

        return parts

    def _build_order_by(self, plan: ResolvedPlan, aliases: Dict,
                         select_parts: List[str]) -> str:
        if not select_parts:
            return ""

        if plan.intent == 'trend':
            for dim in plan.dimensions:
                if dim.expression and dim.alias:
                    return f" ORDER BY {dim.alias}"
            return f" ORDER BY 1"

        if plan.intent in ('ranking', 'distribution', 'aggregation'):
            for metric in plan.metrics:
                if metric.alias:
                    return f" ORDER BY {metric.alias} {plan.sort_direction}"
            return f" ORDER BY COUNT(*) {plan.sort_direction}"

        return " ORDER BY 1"


def plan_and_compose(question: str, schema_learner, concept_index: Dict,
                     domain_concepts: Dict = None, synonyms: Dict = None,
                     computed_columns: Dict = None, kg=None,
                     schema_intel=None) -> Optional[Dict]:
    try:
        planner = QueryPlanner(
            concept_index=concept_index,
            domain_concepts=domain_concepts or {},
            synonyms=synonyms or {},
            schema_learner=schema_learner,
        )
        plan = planner.decompose(question)

        if plan.confidence < 0.3:
            logger.info("QueryPlanner: low confidence %.2f, deferring to legacy path",
                        plan.confidence)
            return None

        resolver = SchemaResolver(
            schema_learner=schema_learner,
            concept_index=concept_index,
            domain_concepts=domain_concepts or {},
            synonyms=synonyms or {},
            computed_columns=computed_columns or {},
            kg=kg,
            schema_intel=schema_intel,
        )
        resolved = resolver.resolve(plan)

        composer = SQLComposer(
            table_row_counts=schema_learner.table_row_counts,
        )
        result = composer.compose(resolved)

        result['plan'] = {
            'intent': plan.intent,
            'metrics': [{'concept': m.concept, 'agg': m.aggregation} for m in plan.metrics],
            'dimensions': [{'concept': d.concept, 'temporal': d.temporal} for d in plan.dimensions],
            'filters': [{'concept': f.concept} for f in plan.filters],
        }

        return result

    except Exception as e:
        logger.warning("QueryPlanner error: %s", e, exc_info=True)
        return None
