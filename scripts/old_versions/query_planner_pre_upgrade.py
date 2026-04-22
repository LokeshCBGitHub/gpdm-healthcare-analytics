"""Query Planner: Decomposes natural language into structured query plans.

Architecture:
    NL Question → QueryPlanner.decompose() → QueryPlan
    QueryPlan   → SchemaResolver.resolve()  → ResolvedPlan
    ResolvedPlan → SQLComposer.compose()    → SQL string

The planner does NOT hardcode SQL templates. It identifies WHAT the question
asks for (metric, dimension, filter, temporal, etc.) and lets the resolver
and composer handle HOW to translate that into SQL against the actual schema.

Domain knowledge (SYNONYMS, DOMAIN_CONCEPTS, etc.) stays declarative in
dynamic_sql_engine.py. This module contains NO healthcare-specific logic.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Set
from collections import defaultdict

logger = logging.getLogger('gpdm.planner')


# ─────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────

@dataclass
class MetricSpec:
    """What to measure."""
    concept: str           # NL concept: "cost", "denial rate", "count", "volume"
    aggregation: str       # "avg", "sum", "count", "count_distinct", "min", "max", "rate", "ratio"
    target: Optional[str] = None    # For rate/ratio: what to count (e.g., "denied")
    denominator: Optional[str] = None  # For rate: total population
    qualifier: Optional[str] = None  # "per encounter", "per member", "per month"

@dataclass
class DimensionSpec:
    """What to group by."""
    concept: str           # NL concept: "region", "provider", "plan type", "age group"
    temporal: bool = False  # Is this a time-based dimension?
    granularity: Optional[str] = None  # For temporal: "month", "quarter", "year", "week"

@dataclass
class FilterSpec:
    """What to filter on."""
    concept: str           # NL concept: "diabetes", "emergency", "high risk"
    operator: str = '='    # =, !=, >, <, >=, <=, LIKE, IN, BETWEEN, IS_NOT_NULL
    value: Optional[str] = None  # Explicit value if provided
    temporal: bool = False  # Is this a time filter?
    temporal_range: Optional[str] = None  # "last 12 months", "this year"

@dataclass
class SortSpec:
    """How to order results."""
    direction: str = 'desc'  # "asc" or "desc"
    by: Optional[str] = None  # What to sort by (if different from metric)
    limit: Optional[int] = None  # Top N

@dataclass
class QueryPlan:
    """Complete decomposition of a natural language question.

    This is the universal intermediate representation. Every question,
    regardless of complexity, is decomposed into these components.
    The schema resolver then maps each component to actual columns/tables.
    """
    # Core intent
    intent: str = 'lookup'  # lookup, aggregation, trend, ranking, comparison, rate, distribution

    # What to measure (can be multiple for complex queries)
    metrics: List[MetricSpec] = field(default_factory=list)

    # What to group/slice by
    dimensions: List[DimensionSpec] = field(default_factory=list)

    # What to filter on
    filters: List[FilterSpec] = field(default_factory=list)

    # Ordering / limiting
    sort: Optional[SortSpec] = None

    # Special modifiers
    percentage: bool = False    # Show as percentage of total
    cumulative: bool = False    # Running total
    comparison_mode: Optional[str] = None  # "vs", "difference", "ratio"

    # Entity focus: when the user says "show me members" → "members"
    # Used for lookup intent to identify the target table directly
    entity_focus: Optional[str] = None

    # Raw question for fallback
    raw_question: str = ''

    # Confidence in the decomposition
    confidence: float = 0.0


# ─────────────────────────────────────────────────────────────────────
# QUERY PLANNER — NL → QueryPlan
# ─────────────────────────────────────────────────────────────────────

class QueryPlanner:
    """Decomposes natural language questions into structured QueryPlans.

    This is schema-agnostic — it understands the STRUCTURE of questions
    (what is being asked, what is being measured, what is being grouped)
    without knowing the specific database schema. Domain vocabulary is
    injected via the concept_index parameter.
    """

    # ── Intent patterns (structural, not domain-specific) ──────────
    _INTENT_PATTERNS = [
        # Trend: temporal analysis
        (r'\b(?:trend|over\s+time|monthly|by\s+month|by\s+year|quarterly|'
         r'over\s+the\s+last|year\s+over\s+year|yoy|time\s+series|'
         r'changed|changes?|evolution|trajectory)\b', 'trend'),
        # Rate: percentage/ratio of a category
        (r'\b(?:denial\s+rate|approval\s+rate|readmission\s+rate|'
         r'rate\s+of|what\s+(?:percent|percentage|%)|'
         r'how\s+many\s+(?:percent|%))\b', 'rate'),
        # Ranking: top/bottom/highest/lowest
        (r'\b(?:top\s+\d+|bottom\s+\d+|highest|lowest|most|least|'
         r'rank|ranking|best|worst|which\s+\w+\s+has?\s+the)\b', 'ranking'),
        # Comparison: A vs B
        (r'\b(?:compare|comparison|versus|vs\.?|differ|difference|'
         r'contrast)\b', 'comparison'),
        # Distribution: breakdown, split
        (r'\b(?:breakdown|distribution|split|composition|proportion|'
         r'share|makeup|breakout)\b', 'distribution'),
        # Lookup: show me / list / display / get (all) X
        (r'\b(?:show\s+(?:me\s+)?(?:all\s+)?|list\s+(?:all\s+)?|display\s+(?:all\s+)?|'
         r'get\s+(?:all\s+)?|view\s+(?:all\s+)?|give\s+me\s+(?:all\s+)?|'
         r'pull\s+(?:up\s+)?(?:all\s+)?|find\s+(?:all\s+)?|fetch\s+(?:all\s+)?)'
         r'(?:the\s+)?(\w+)', 'lookup'),
        # Aggregation: avg, total, count
        (r'\b(?:average|avg|total|sum|count|how\s+many|number\s+of|'
         r'mean|median|minimum|maximum)\b', 'aggregation'),
    ]

    # ── Aggregation detection ──────────────────────────────────────
    _AGG_PATTERNS = [
        (r'\baverage\b|\bavg\b|\bmean\b', 'avg'),
        (r'\btotal\b|\bsum\b|\bcombined\b|\baggregate\b', 'sum'),
        (r'\bcount\b|\bhow\s+many\b|\bnumber\s+of\b|\bvolume\b', 'count'),
        (r'\bmaximum\b|\bmax\b|\bhighest\b|\blargest\b|\bbiggest\b|\bmost\b', 'max'),
        (r'\bminimum\b|\bmin\b|\blowest\b|\bsmallest\b|\bleast\b', 'min'),
        (r'\bdistinct\b|\bunique\b', 'count_distinct'),
    ]

    # ── Temporal granularity ───────────────────────────────────────
    _TEMPORAL_GRANULARITY = [
        (r'\b(?:daily|by\s+day|per\s+day|each\s+day)\b', 'day'),
        (r'\b(?:weekly|by\s+week|per\s+week|each\s+week)\b', 'week'),
        (r'\b(?:monthly|by\s+month|per\s+month|each\s+month)\b', 'month'),
        (r'\b(?:quarterly|by\s+quarter|per\s+quarter|each\s+quarter)\b', 'quarter'),
        (r'\b(?:yearly|annually|by\s+year|per\s+year|each\s+year|annual)\b', 'year'),
    ]

    # ── Time filter patterns ───────────────────────────────────────
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

    # ── Sort/limit patterns ────────────────────────────────────────
    _SORT_PATTERNS = [
        (r'\btop\s+(\d+)\b', 'desc'),
        (r'\bbottom\s+(\d+)\b', 'asc'),
        (r'\bhighest\b|\bmost\b|\blargest\b|\bbiggest\b|\bgreatest\b', 'desc'),
        (r'\blowest\b|\bleast\b|\bsmallest\b|\bfewest\b', 'asc'),
    ]

    # ── Dimension signal words ─────────────────────────────────────
    _DIMENSION_SIGNALS = [
        r'\bby\s+(\w+(?:\s+\w+)?)\b',
        r'\bper\s+(\w+)\b',
        r'\beach\s+(\w+)\b',
        r'\bacross\s+(\w+(?:\s+\w+)?)\b',
        r'\bfor\s+each\s+(\w+)\b',
        r'\bgrouped?\s+by\s+(\w+(?:\s+\w+)?)\b',
    ]

    # ── Rate patterns ──────────────────────────────────────────────
    _RATE_PATTERNS = [
        (r'(\w+)\s+rate', None),          # "denial rate", "readmission rate"
        (r'(?:percent|percentage|%)\s+(?:of\s+)?(\w+)', None),  # "percentage of denied"
        (r'what\s+(?:percent|percentage|%)\s+(?:of\s+)?(?:\w+\s+)?(?:are|is|were|was)\s+(\w+)', None),
    ]

    # ── Threshold patterns ─────────────────────────────────────────
    _THRESHOLD_PATTERNS = [
        r'(?:more\s+than|over|above|exceeds?|>=?)\s*(\d+)',
        r'(?:less\s+than|under|below|<=?)\s*(\d+)',
        r'(?:at\s+least|minimum)\s+(\d+)',
        r'(?:at\s+most|maximum)\s+(\d+)',
        r'(?:between)\s+(\d+)\s+and\s+(\d+)',
    ]

    # ── Percentage signals ─────────────────────────────────────────
    _PERCENTAGE_SIGNALS = frozenset({
        'percent', 'percentage', '%', 'proportion', 'share', 'fraction',
        'what percentage', 'what percent',
    })

    def __init__(self, concept_index: Dict[str, Dict[str, float]] = None,
                 domain_concepts: Dict = None, synonyms: Dict = None):
        """
        Args:
            concept_index: word → {table: score} mapping from schema
            domain_concepts: healthcare term → {conds, tables} from dynamic_sql_engine
            synonyms: NL term → [column_names] from dynamic_sql_engine
        """
        self.concept_index = concept_index or {}
        self.domain_concepts = domain_concepts or {}
        self.synonyms = synonyms or {}

        # Build reverse index: concept word → possible metric columns
        self._metric_concepts = self._build_metric_concepts()

    def _build_metric_concepts(self) -> Dict[str, List[str]]:
        """Build mapping from NL metric words to possible column names."""
        metrics = defaultdict(list)
        for term, cols in self.synonyms.items():
            for col in cols:
                # Only numeric-sounding columns are metrics
                if any(kw in col.lower() for kw in
                       ['amount', 'cost', 'count', 'score', 'rate', 'size',
                        'days', 'quantity', 'duration', 'rvu', 'supply']):
                    metrics[term].append(col)
        return dict(metrics)

    def decompose(self, question: str) -> QueryPlan:
        """Decompose a natural language question into a structured QueryPlan.

        This is the main entry point. It identifies:
        1. Intent — what type of question is this?
        2. Metrics — what is being measured?
        3. Dimensions — what is being grouped by?
        4. Filters — what population/subset?
        5. Sort/limit — ordering and top-N
        6. Modifiers — percentage, cumulative, comparison
        """
        q = question.lower().strip()
        plan = QueryPlan(raw_question=question)

        # Step 1: Detect intent (and entity focus for lookup)
        plan.intent = self._detect_intent(q)

        # Step 1b: Extract entity focus for lookup intent
        if plan.intent == 'lookup':
            plan.entity_focus = self._extract_entity_focus(q)

        # Step 2: Extract metrics (what to measure)
        plan.metrics = self._extract_metrics(q, plan.intent)

        # Step 3: Extract dimensions (what to group by)
        plan.dimensions = self._extract_dimensions(q, plan.intent)

        # Step 4: Extract filters (what to filter on)
        plan.filters = self._extract_filters(q)

        # Step 5: Extract sort/limit
        plan.sort = self._extract_sort(q)

        # Step 6: Detect modifiers
        plan.percentage = self._detect_percentage(q)
        plan.comparison_mode = self._detect_comparison_mode(q)

        # Step 7: Infer missing components
        plan = self._infer_missing(plan, q)

        # Step 8: Calculate plan confidence
        plan.confidence = self._calculate_plan_confidence(plan)

        logger.info("QueryPlan: intent=%s, metrics=%d, dims=%d, filters=%d, conf=%.2f",
                     plan.intent, len(plan.metrics), len(plan.dimensions),
                     len(plan.filters), plan.confidence)

        return plan

    # ── Step 1: Intent Detection ───────────────────────────────────

    def _detect_intent(self, q: str) -> str:
        """Detect the question's intent from structural patterns.

        Priority order matters — more specific intents match first.
        """
        for pattern, intent in self._INTENT_PATTERNS:
            if re.search(pattern, q):
                return intent

        # Fallback heuristics
        if '?' in q and re.search(r'\b(?:what|which|who|how)\b', q):
            return 'lookup'

        return 'aggregation'  # Default: most questions want some aggregate

    def _extract_entity_focus(self, q: str) -> Optional[str]:
        """Extract the entity being looked up (e.g., 'members' from 'show me all members')."""
        # Match "show me (all) (the) X", "list (all) X", etc.
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

    # ── Step 2: Metric Extraction ──────────────────────────────────

    def _extract_metrics(self, q: str, intent: str) -> List[MetricSpec]:
        """Extract what is being measured from the question."""
        metrics = []

        # Detect aggregation type
        agg = 'count'  # Default
        for pattern, agg_type in self._AGG_PATTERNS:
            if re.search(pattern, q):
                agg = agg_type
                break

        # Rate detection: "denial rate", "approval rate", etc.
        rate_match = re.search(r'(\w+)\s+rate\b', q)
        if rate_match:
            target = rate_match.group(1)
            metrics.append(MetricSpec(
                concept=f'{target} rate',
                aggregation='rate',
                target=target,
            ))
            return metrics

        # Percentage detection: "what percentage of patients are female"
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

        # Ratio detection: "X to Y ratio"
        ratio_match = re.search(r'(\w+)\s+to\s+(\w+)\s+ratio', q)
        if ratio_match:
            metrics.append(MetricSpec(
                concept=f'{ratio_match.group(1)} to {ratio_match.group(2)} ratio',
                aggregation='ratio',
                target=ratio_match.group(1),
                denominator=ratio_match.group(2),
            ))
            return metrics

        # Extract concept words that could be metrics
        # Look for words that map to numeric columns via synonyms
        q_words = set(re.findall(r'[a-z]+', q))
        metric_candidates = []
        for word in q_words:
            if word in self._metric_concepts:
                metric_candidates.append(word)
            # Also check multi-word synonyms
        for term in self.synonyms:
            if ' ' in term and term in q:
                if term in self._metric_concepts:
                    metric_candidates.append(term)

        # Pick the best metric candidate
        if metric_candidates:
            # Prefer the most specific (longest) match
            best = max(metric_candidates, key=len)
            metrics.append(MetricSpec(concept=best, aggregation=agg))
        elif intent in ('aggregation', 'trend', 'ranking', 'distribution'):
            # No explicit metric found — use count as default
            metrics.append(MetricSpec(concept='count', aggregation='count'))

        # For "volume" questions, ensure count
        if 'volume' in q and not any(m.aggregation == 'count' for m in metrics):
            metrics.append(MetricSpec(concept='volume', aggregation='count'))

        return metrics

    # ── Step 3: Dimension Extraction ───────────────────────────────

    def _extract_dimensions(self, q: str, intent: str) -> List[DimensionSpec]:
        """Extract what to group by from the question."""
        dims = []

        # Check for temporal dimension (trend queries)
        if intent == 'trend' or re.search(r'\bover\s+time\b|\btrend\b|\bmonthly\b', q):
            granularity = 'month'  # Default
            for pattern, gran in self._TEMPORAL_GRANULARITY:
                if re.search(pattern, q):
                    granularity = gran
                    break
            dims.append(DimensionSpec(
                concept='time', temporal=True, granularity=granularity))

        # Extract explicit dimension from "by X", "per X", "across X", etc.
        for pattern in self._DIMENSION_SIGNALS:
            m = re.search(pattern, q)
            if m:
                dim_term = m.group(1).strip()
                # Skip temporal terms (already handled)
                if dim_term in ('month', 'year', 'quarter', 'week', 'day',
                                'time', 'date', 'period'):
                    continue
                # Skip aggregation terms that aren't dimensions
                if dim_term in ('count', 'total', 'amount', 'cost', 'average'):
                    continue
                dims.append(DimensionSpec(concept=dim_term))
                break  # Only one explicit dimension per question typically

        # For ranking intent without explicit dimension, infer from subject
        if intent == 'ranking' and not any(not d.temporal for d in dims):
            subject = self._extract_ranking_subject(q)
            if subject:
                dims.append(DimensionSpec(concept=subject))

        # For distribution without explicit dimension, infer from context
        if intent == 'distribution' and not any(not d.temporal for d in dims):
            dist_target = self._extract_distribution_target(q)
            if dist_target:
                dims.append(DimensionSpec(concept=dist_target))

        return dims

    def _extract_ranking_subject(self, q: str) -> Optional[str]:
        """Extract what entity is being ranked.
        E.g., "which providers have the highest volume" → "provider"
        """
        patterns = [
            r'\bwhich\s+(\w+)',
            r'\btop\s+\d*\s*(\w+)',
            r'\b(\w+)\s+with\s+(?:the\s+)?(?:high|low|most|least)',
        ]
        for pat in patterns:
            m = re.search(pat, q)
            if m:
                subject = m.group(1).strip().rstrip('s')  # Singularize
                if subject not in ('the', 'are', 'is', 'has', 'have', 'do', 'does'):
                    return subject
        return None

    def _extract_distribution_target(self, q: str) -> Optional[str]:
        """Extract what to show distribution of.
        E.g., "breakdown of claims by region" → "region"
        """
        m = re.search(r'(?:breakdown|distribution|split)\s+(?:of\s+)?(\w+)', q)
        if m:
            return m.group(1).strip()
        return None

    # ── Step 4: Filter Extraction ──────────────────────────────────

    def _extract_filters(self, q: str) -> List[FilterSpec]:
        """Extract all filters from the question.

        Filters come from:
        1. Domain concepts (e.g., "diabetic" → ICD10 filter)
        2. Explicit values (e.g., "in the emergency department")
        3. Temporal ranges (e.g., "last 12 months")
        4. Thresholds (e.g., "more than 10 claims")
        """
        filters = []

        # 1. Domain concept filters
        q_lower = q.lower()
        q_words = set(re.findall(r'\b\w+\b', q_lower))

        # Sort by length (longest first) to match multi-word terms first
        sorted_concepts = sorted(self.domain_concepts.items(),
                                 key=lambda x: len(x[0]), reverse=True)
        matched_terms = set()

        # If this is a rate query, remember the rate target so we don't
        # add it as a WHERE filter too (that would pre-filter the denominator)
        _rate_match = re.search(r'(\w+)\s+rate\b', q_lower)
        _rate_target = _rate_match.group(1) if _rate_match else None

        for term, concept in sorted_concepts:
            t_lower = term.lower()

            # Skip very short terms (2 chars or less) — too ambiguous
            # e.g., 'do' matches PROVIDER_TYPE='DO' but also appears in
            # "how many members do we have?"
            if len(t_lower) <= 2:
                continue

            # Word boundary check: for single-word terms, require exact word match
            # For multi-word terms, require substring match (already length-gated)
            t_words = t_lower.split()
            if len(t_words) == 1:
                if t_lower not in q_words:
                    continue
            else:
                if t_lower not in q_lower:
                    continue

            # Skip domain concept filters that match the rate target
            # e.g., for "denial rate", don't add WHERE CLAIM_STATUS='DENIED'
            # because the rate expression already handles it in CASE WHEN
            if _rate_target and t_lower in (_rate_target, _rate_target + 'ed',
                                            _rate_target + 'd', _rate_target + 'ial',
                                            _rate_target + 'al'):
                continue

            # Avoid overlapping matches
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
                    operator='domain',  # Signals: use domain condition
                    value=term,
                ))

        # 2. Temporal filters
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
                break  # Only one time filter

        # 3. Threshold filters
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

    # ── Step 5: Sort Extraction ────────────────────────────────────

    def _extract_sort(self, q: str) -> Optional[SortSpec]:
        """Extract sort direction and limit from the question."""
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

    # ── Step 6: Modifier Detection ─────────────────────────────────

    def _detect_percentage(self, q: str) -> bool:
        """Detect if the question asks for percentage/proportion."""
        return any(sig in q for sig in self._PERCENTAGE_SIGNALS)

    def _detect_comparison_mode(self, q: str) -> Optional[str]:
        """Detect if this is a comparison question and what type."""
        if re.search(r'\bvs\.?\b|\bversus\b', q):
            return 'vs'
        if re.search(r'\bcompare\b|\bcomparison\b', q):
            return 'compare'
        if re.search(r'\bdifference\b|\bdelta\b', q):
            return 'difference'
        return None

    # ── Step 7: Inference ──────────────────────────────────────────

    def _infer_missing(self, plan: QueryPlan, q: str) -> QueryPlan:
        """Infer missing plan components from context.

        If the question says "trend" but no temporal dimension was found,
        add one. If it says "by region" but no metric, add count. Etc.
        """
        # Trend without temporal dimension → add month granularity
        if plan.intent == 'trend' and not any(d.temporal for d in plan.dimensions):
            plan.dimensions.insert(0, DimensionSpec(
                concept='time', temporal=True, granularity='month'))

        # Ranking without sort → add desc sort
        if plan.intent == 'ranking' and not plan.sort:
            plan.sort = SortSpec(direction='desc', limit=20)

        # Has dimensions but no metric → default to count
        if plan.dimensions and not plan.metrics:
            plan.metrics.append(MetricSpec(concept='count', aggregation='count'))

        # Rate intent without rate metric → look for rate in question
        if plan.intent == 'rate' and not any(m.aggregation == 'rate' for m in plan.metrics):
            rate_m = re.search(r'(\w+)\s+rate', q)
            if rate_m:
                plan.metrics.insert(0, MetricSpec(
                    concept=f'{rate_m.group(1)} rate',
                    aggregation='rate',
                    target=rate_m.group(1),
                ))

        # Percentage flag without rate metric → synthesize one
        if plan.percentage and not any(m.aggregation == 'rate' for m in plan.metrics):
            pct_m = re.search(r'(?:are|is|were|was)\s+(\w+)', q)
            if pct_m:
                plan.metrics.insert(0, MetricSpec(
                    concept=f'percentage {pct_m.group(1)}',
                    aggregation='rate',
                    target=pct_m.group(1),
                ))

        # Threshold filter without dimension → infer entity to group by
        # "members with more than 5 encounters" → GROUP BY member, COUNT encounters
        # "providers with more than 10 claims" → GROUP BY provider, COUNT claims
        has_threshold = any(f.concept == 'threshold' for f in plan.filters)
        has_non_temporal_dim = any(not d.temporal for d in plan.dimensions)
        if has_threshold and not has_non_temporal_dim:
            # Find the entity being counted from the question
            entity_match = re.search(
                r'(\w+(?:s)?)\s+(?:with|having|where)\s+(?:more\s+than|over|at\s+least|greater\s+than|>=?)\s*\d+',
                q)
            if entity_match:
                entity = entity_match.group(1).rstrip('s')  # "members" → "member"
                plan.dimensions.append(DimensionSpec(concept=entity))
            else:
                # Fallback: look for "N entities" pattern
                n_entity = re.search(r'\d+\s+(\w+)', q)
                if n_entity:
                    # The table being counted is likely the fact table dimension
                    counted = n_entity.group(1).rstrip('s')
                    # The GROUP BY entity is the SUBJECT, not the counted thing
                    # "members with more than 5 encounters" → group by member
                    subject = re.search(r'^(\w+)', q)
                    if subject and subject.group(1).lower() not in (
                        'what', 'which', 'how', 'show', 'find', 'list', 'get'):
                        plan.dimensions.append(DimensionSpec(concept=subject.group(1).rstrip('s')))

        return plan

    # ── Step 8: Confidence ─────────────────────────────────────────

    def _calculate_plan_confidence(self, plan: QueryPlan) -> float:
        """Estimate confidence in the decomposition."""
        score = 0.0

        # Has an intent → +0.2
        if plan.intent == 'lookup' and plan.entity_focus:
            # Lookup with entity focus is a strong, clear intent
            score += 0.5
        elif plan.intent != 'lookup':
            score += 0.2

        # Has metrics → +0.3
        if plan.metrics:
            score += 0.3

        # Has dimensions or filters → +0.2
        if plan.dimensions or plan.filters:
            score += 0.2

        # Metrics have specific concepts (not just "count") → +0.15
        if any(m.concept != 'count' for m in plan.metrics):
            score += 0.15

        # Has temporal component → +0.15
        if any(d.temporal for d in plan.dimensions) or \
           any(f.temporal for f in plan.filters):
            score += 0.15

        return min(1.0, score)


# ─────────────────────────────────────────────────────────────────────
# SCHEMA RESOLVER — QueryPlan → ResolvedPlan
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ResolvedMetric:
    """A metric resolved to actual schema elements."""
    table: str
    column: str              # Actual column name (e.g., "PAID_AMOUNT")
    aggregation: str         # SQL aggregation: AVG, SUM, COUNT, etc.
    expression: Optional[str] = None  # Full SQL expression if complex
    alias: str = ''          # AS alias

@dataclass
class ResolvedDimension:
    """A dimension resolved to actual schema elements."""
    table: str
    column: str              # Actual column name (e.g., "KP_REGION")
    expression: Optional[str] = None  # For computed dims (age_group, time bucket)
    alias: str = ''
    needs_join: bool = False
    join_on: Optional[str] = None  # JOIN condition

@dataclass
class ResolvedFilter:
    """A filter resolved to actual SQL condition."""
    table: str
    condition: str           # Full SQL condition (e.g., "CLAIM_STATUS = 'DENIED'")

@dataclass
class ResolvedPlan:
    """A fully resolved query plan with actual schema references."""
    primary_table: str
    metrics: List[ResolvedMetric] = field(default_factory=list)
    dimensions: List[ResolvedDimension] = field(default_factory=list)
    filters: List[ResolvedFilter] = field(default_factory=list)
    join_tables: List[Tuple[str, str]] = field(default_factory=list)  # [(table, join_cond)]
    sort_direction: str = 'desc'
    sort_column: Optional[str] = None
    limit: Optional[int] = None
    percentage: bool = False
    confidence: float = 0.0
    intent: str = 'aggregation'


class SchemaResolver:
    """Resolves abstract QueryPlan concepts to actual database schema elements.

    Uses the SchemaLearner's metadata (tables, columns, types, sample values,
    join paths) to map "cost" → PAID_AMOUNT in claims, "region" → KP_REGION, etc.
    """

    def __init__(self, schema_learner, concept_index: Dict,
                 domain_concepts: Dict = None, synonyms: Dict = None,
                 computed_columns: Dict = None, kg=None, schema_intel=None):
        """
        Args:
            schema_learner: SchemaLearner with table metadata
            concept_index: word → {table: score} mapping
            domain_concepts: healthcare term → {conds, tables}
            synonyms: NL term → [column_names]
            computed_columns: name → {expr, alias, table, description}
            kg: KnowledgeGraph for join paths
            schema_intel: Optional SchemaIntelligence for TF-IDF semantic fallback
        """
        self.learner = schema_learner
        self.concept_index = concept_index
        self.domain_concepts = domain_concepts or {}
        self.synonyms = synonyms or {}
        self.computed_columns = computed_columns or {}
        self.kg = kg
        self._schema_intel = schema_intel  # Semantic search fallback

        # Pre-build column type index
        self._numeric_cols = {}  # table → {col_name: profile}
        self._categorical_cols = {}
        self._date_cols = {}
        self._id_cols = {}
        for tbl, profiles in self.learner.tables.items():
            self._numeric_cols[tbl] = {p.name: p for p in profiles if p.is_numeric}
            self._categorical_cols[tbl] = {p.name: p for p in profiles if p.is_categorical}
            self._date_cols[tbl] = {p.name: p for p in profiles if p.is_date}
            self._id_cols[tbl] = {p.name: p for p in profiles if p.name.endswith('_ID') or p.name == 'NPI' or p.name == 'MRN'}

    def resolve(self, plan: QueryPlan) -> ResolvedPlan:
        """Resolve all abstract concepts in a QueryPlan to concrete schema elements."""
        resolved = ResolvedPlan(
            primary_table='',
            confidence=plan.confidence,
            intent=plan.intent,
            percentage=plan.percentage,
        )

        # Step 1: Identify primary table from metrics + dimensions + filters
        resolved.primary_table = self._resolve_primary_table(plan)

        # Step 2: Resolve metrics
        resolved.metrics = self._resolve_metrics(plan, resolved.primary_table)

        # Step 3: Resolve dimensions
        resolved.dimensions = self._resolve_dimensions(plan, resolved.primary_table)

        # Step 4: Resolve filters
        resolved.filters = self._resolve_filters(plan, resolved.primary_table)

        # Step 5: Determine needed joins
        resolved.join_tables = self._resolve_joins(resolved)

        # Step 6: Sort and limit
        if plan.sort:
            resolved.sort_direction = plan.sort.direction
            resolved.limit = plan.sort.limit
        elif plan.intent == 'ranking':
            resolved.sort_direction = 'desc'
            resolved.limit = 20

        # Step 7: Cross-table validation — ensure metric columns exist in resolved tables
        resolved = self._validate_and_fix_tables(resolved)

        logger.info("ResolvedPlan: table=%s, metrics=%d, dims=%d, filters=%d, joins=%d",
                     resolved.primary_table, len(resolved.metrics),
                     len(resolved.dimensions), len(resolved.filters),
                     len(resolved.join_tables))

        return resolved

    def _resolve_primary_table(self, plan: QueryPlan) -> str:
        """Determine the primary/fact table from plan components."""
        # Fast path: if the plan has an entity_focus, match directly to a table
        if plan.entity_focus:
            entity = plan.entity_focus.lower()
            # Direct table name match (singular or plural)
            for tbl in self.learner.tables:
                tbl_lower = tbl.lower()
                if (entity == tbl_lower or
                    entity.rstrip('s') == tbl_lower.rstrip('s') or
                    entity == tbl_lower.rstrip('s') or
                    tbl_lower == entity.rstrip('s')):
                    logger.info("Entity focus '%s' → table %s", entity, tbl)
                    return tbl

            # Check concept index
            if entity in self.concept_index:
                best_tbl = max(self.concept_index[entity].items(),
                               key=lambda x: x[1], default=(None, 0))
                if best_tbl[0]:
                    logger.info("Entity focus '%s' → table %s (via concept_index)",
                                entity, best_tbl[0])
                    return best_tbl[0]

        table_scores = defaultdict(float)
        q_words = set(re.findall(r'[a-z]+', plan.raw_question.lower()))

        # Strong boost: if a word IS a table name (or singular/plural variant)
        for word in q_words:
            for tbl in self.learner.tables:
                tbl_lower = tbl.lower()
                if (word == tbl_lower or
                    word.rstrip('s') == tbl_lower.rstrip('s') or
                    word == tbl_lower.rstrip('s')):
                    table_scores[tbl] += 50  # Dominant signal

        # Score from concept index
        for word in q_words:
            if word in self.concept_index:
                for tbl, score in self.concept_index[word].items():
                    table_scores[tbl] += score

        # Boost from metrics
        for metric in plan.metrics:
            concept_words = set(re.findall(r'[a-z]+', metric.concept.lower()))
            for word in concept_words:
                if word in self.synonyms:
                    for col in self.synonyms[word]:
                        for tbl in self.learner.tables:
                            if any(p.name == col for p in self.learner.tables[tbl]):
                                table_scores[tbl] += 5

        # Boost from dimensions
        for dim in plan.dimensions:
            if not dim.temporal:
                concept_words = set(re.findall(r'[a-z]+', dim.concept.lower()))
                for word in concept_words:
                    if word in self.concept_index:
                        for tbl, score in self.concept_index[word].items():
                            table_scores[tbl] += score * 0.5

        # Boost from filters (domain concepts specify tables)
        for filt in plan.filters:
            if filt.concept in self.domain_concepts:
                for tbl in self.domain_concepts[filt.concept].get('tables', []):
                    table_scores[tbl] += 10

        if not table_scores:
            # Fallback: largest table
            return max(self.learner.table_row_counts, key=self.learner.table_row_counts.get) \
                if self.learner.table_row_counts else 'claims'

        # Pick highest-scoring table, with tie-break on row count
        return max(table_scores, key=lambda t: (table_scores[t],
                   self.learner.table_row_counts.get(t, 0)))

    def _resolve_metrics(self, plan: QueryPlan, primary_table: str) -> List[ResolvedMetric]:
        """Resolve abstract metric specs to actual columns."""
        resolved = []

        for metric in plan.metrics:
            # Rate metrics → CASE WHEN expression
            if metric.aggregation == 'rate':
                rm = self._resolve_rate_metric(metric, primary_table)
                if rm:
                    resolved.append(rm)
                continue

            # Count metric → COUNT(*)
            if metric.aggregation == 'count' and metric.concept in ('count', 'volume'):
                resolved.append(ResolvedMetric(
                    table=primary_table,
                    column='*',
                    aggregation='COUNT',
                    alias='record_count',
                ))
                continue

            # Count distinct
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

            # Standard aggregation → find the numeric column
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

        # If no metrics resolved, add a default COUNT(*)
        if not resolved:
            resolved.append(ResolvedMetric(
                table=primary_table,
                column='*',
                aggregation='COUNT',
                alias='record_count',
            ))

        return resolved

    def _resolve_rate_metric(self, metric: MetricSpec, primary_table: str) -> Optional[ResolvedMetric]:
        """Resolve a rate metric (denial rate, readmission rate, etc.).

        Strategy order:
        1. Domain concepts — "denied" → CLAIM_STATUS = 'DENIED' (from dynamic_sql_engine)
        2. Computed columns — "denial_rate" pre-defined expression
        3. DB query — find actual distinct values in status-like columns
        4. Sample values fallback
        """
        target = metric.target
        if not target:
            return None

        target_lower = target.lower()
        target_upper = target.upper()

        # ── Strategy 1: Domain concepts ──
        # Check if domain_concepts has a direct mapping for the target or morphological variants
        # "denial" → "denied", "approval" → "approved", "readmission" → "readmitted"
        _morphs = {target_lower}
        _morphs.add(metric.concept.lower())
        # Generate common verb forms from noun: denial→denied, approval→approved
        if target_lower.endswith('al'):
            _morphs.add(target_lower[:-2] + 'ed')  # denial→denied, approval→approved
        elif target_lower.endswith('ion'):
            _morphs.add(target_lower[:-3] + 'ted')  # readmission→readmitted (close enough)
            _morphs.add(target_lower[:-3] + 'ed')   # admission→admitted
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
                    # Use the domain condition directly in a CASE WHEN
                    # Take the first condition as the rate numerator
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

        # ── Strategy 2: Computed columns ──
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

        # ── Strategy 3: DB query for actual distinct values ──
        # Find status-like categorical columns and query for real values
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
                        # Only check status-like columns
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

        # ── Strategy 4: Sample values fallback ──
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
        """Find the best numeric column matching a concept.

        Search order:
        1. Synonym mapping (NL term → column names)
        2. Column name word overlap
        3. Cross-table search
        """
        concept_words = set(re.findall(r'[a-z]+', concept.lower()))

        # Strategy 1: Synonym lookup
        for word in concept_words:
            if word in self.synonyms:
                for col_name in self.synonyms[word]:
                    # Check primary table first
                    if col_name in self._numeric_cols.get(primary_table, {}):
                        return col_name, primary_table
                    # Then other tables
                    for tbl in self.learner.tables:
                        if col_name in self._numeric_cols.get(tbl, {}):
                            return col_name, tbl

        # Strategy 2: Column name overlap
        best_col, best_table, best_score = None, primary_table, 0
        for tbl_name, num_cols in self._numeric_cols.items():
            for col_name in num_cols:
                col_words = set(col_name.lower().replace('_', ' ').split())
                overlap = len(concept_words & col_words)
                # Prefer primary table
                tbl_bonus = 2 if tbl_name == primary_table else 0
                score = overlap + tbl_bonus
                if score > best_score:
                    best_score = score
                    best_col = col_name
                    best_table = tbl_name

        if best_col:
            return best_col, best_table

        # Strategy 3: TF-IDF semantic search via SchemaIntelligence
        if self._schema_intel:
            result = self._schema_intel.find_metric(concept, preferred_table=primary_table)
            if result:
                logger.info("SchemaResolver: semantic fallback found metric '%s' → %s.%s",
                            concept, result[0], result[1])
                return result[1], result[0]

        # Strategy 4: Return first numeric column in primary table
        if self._numeric_cols.get(primary_table):
            first = next(iter(self._numeric_cols[primary_table]))
            return first, primary_table

        return None, primary_table

    def _find_id_column(self, table: str, concept: str) -> Optional[str]:
        """Find an ID column for count distinct queries."""
        concept_lower = concept.lower()
        ids = self._id_cols.get(table, {})
        for col_name in ids:
            if concept_lower in col_name.lower():
                return col_name
        # Return primary-looking ID
        for col_name in ids:
            if col_name == 'MEMBER_ID':
                return col_name
        return next(iter(ids), None) if ids else None

    def _resolve_dimensions(self, plan: QueryPlan, primary_table: str) -> List[ResolvedDimension]:
        """Resolve abstract dimensions to actual columns."""
        resolved = []

        for dim in plan.dimensions:
            if dim.temporal:
                # Time dimension → find best date column and apply bucketing
                rd = self._resolve_temporal_dimension(dim, primary_table)
                if rd:
                    resolved.append(rd)
            else:
                # Categorical dimension → find best categorical column
                rd = self._resolve_categorical_dimension(dim, primary_table)
                if rd:
                    resolved.append(rd)

        return resolved

    def _resolve_temporal_dimension(self, dim: DimensionSpec,
                                     primary_table: str) -> Optional[ResolvedDimension]:
        """Resolve a time dimension to a date column + bucketing expression.

        The date column is qualified with the table's first letter alias
        since we always use aliases in the composer.
        """
        # Import TABLE_DATE_COL from domain config
        try:
            from dynamic_sql_engine import TABLE_DATE_COL
            date_col = TABLE_DATE_COL.get(primary_table)
        except ImportError:
            date_col = None

        # Determine which table owns this date column
        date_table = primary_table

        # Fallback: find any date column in primary table
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

        # If primary table has no date col, search other tables
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

        # Qualify date column with table alias (first letter)
        tbl_alias = date_table[0]
        qualified_date = f"{tbl_alias}.{date_col}"

        # Build bucketing expression with qualified column
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
        """Resolve a categorical dimension (region, specialty, etc.) to actual column."""
        concept = dim.concept.lower().strip()
        concept_words = set(re.findall(r'[a-z]+', concept))

        # Strategy 1: Check computed columns (age_group, risk_tier, etc.)
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

        # Strategy 2: Synonym lookup
        for word in [concept] + list(concept_words):
            if word in self.synonyms:
                for col_name in self.synonyms[word]:
                    # Find which table has this categorical column
                    if col_name in self._categorical_cols.get(primary_table, {}):
                        return ResolvedDimension(
                            table=primary_table,
                            column=col_name,
                            alias=col_name.lower(),
                        )
                    # Check other tables
                    for tbl in self.learner.tables:
                        if col_name in self._categorical_cols.get(tbl, {}):
                            return ResolvedDimension(
                                table=tbl,
                                column=col_name,
                                alias=col_name.lower(),
                                needs_join=(tbl != primary_table),
                            )
                    # Also check ID columns for entity-based dimensions (provider, member)
                    for tbl in self.learner.tables:
                        if col_name in self._id_cols.get(tbl, {}):
                            return ResolvedDimension(
                                table=tbl,
                                column=col_name,
                                alias=col_name.lower(),
                                needs_join=(tbl != primary_table),
                            )

        # Strategy 3: Column name scoring
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

        # Strategy 4: TF-IDF semantic search via SchemaIntelligence
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
        """Resolve abstract filters to actual SQL conditions."""
        resolved = []

        for filt in plan.filters:
            if filt.temporal:
                # Time filter → date range condition
                rf = self._resolve_temporal_filter(filt, primary_table)
                if rf:
                    resolved.append(rf)
            elif filt.operator == 'domain':
                # Domain concept → use DOMAIN_CONCEPTS conditions
                rfs = self._resolve_domain_filter(filt, primary_table)
                resolved.extend(rfs)
            elif filt.concept == 'threshold':
                # Threshold → HAVING clause (handled by composer)
                resolved.append(ResolvedFilter(
                    table=primary_table,
                    condition=f'__THRESHOLD__{filt.operator}__{filt.value}',
                ))
            else:
                # Generic filter → try to match column + value
                rf = self._resolve_generic_filter(filt, primary_table)
                if rf:
                    resolved.append(rf)

        return resolved

    def _resolve_temporal_filter(self, filt: FilterSpec, primary_table: str) -> Optional[ResolvedFilter]:
        """Resolve a temporal filter to a date range condition."""
        try:
            from dynamic_sql_engine import TABLE_DATE_COL, TIME_FILTERS
        except ImportError:
            return None

        date_col = TABLE_DATE_COL.get(primary_table, 'SERVICE_DATE')
        time_range = filt.temporal_range

        # Check if it matches a predefined filter
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

        # Parse "last N months/years/days"
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
        """Resolve a domain concept filter to SQL conditions."""
        concept = filt.concept
        if concept not in self.domain_concepts:
            # Try lowercase
            concept = concept.lower()
        if concept not in self.domain_concepts:
            return []

        conds = self.domain_concepts[concept].get('conds', [])
        tables = self.domain_concepts[concept].get('tables', [])

        # Use the concept's specified table if it matches available tables
        tbl = primary_table
        for t in tables:
            if t in self.learner.tables:
                tbl = t
                break

        return [ResolvedFilter(table=tbl, condition=c) for c in conds]

    def _resolve_generic_filter(self, filt: FilterSpec, primary_table: str) -> Optional[ResolvedFilter]:
        """Resolve a generic filter by matching concept to column values."""
        concept = filt.concept.lower()
        value = filt.value

        # Search for a categorical column that has this value
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
        """Determine what tables need to be JOINed and how."""
        needed_tables = set()
        needed_tables.add(resolved.primary_table)

        # Collect all tables referenced by metrics, dimensions, filters
        for m in resolved.metrics:
            needed_tables.add(m.table)
        for d in resolved.dimensions:
            if d.needs_join:
                needed_tables.add(d.table)
        for f in resolved.filters:
            needed_tables.add(f.table)

        # Remove primary
        join_tables = needed_tables - {resolved.primary_table}
        joins = []

        for tbl in join_tables:
            join_cond = self._find_join_condition(resolved.primary_table, tbl)
            if join_cond:
                joins.append((tbl, join_cond))

        return joins

    def _find_join_condition(self, tbl_a: str, tbl_b: str) -> Optional[str]:
        """Find the JOIN condition between two tables."""
        # Try knowledge graph
        if self.kg:
            jc = self.kg._get_join_condition(tbl_a, tbl_b)
            if jc:
                return jc

        # Find shared columns
        cols_a = {p.name for p in self.learner.tables.get(tbl_a, [])}
        cols_b = {p.name for p in self.learner.tables.get(tbl_b, [])}
        shared = sorted(c for c in cols_a & cols_b if c.endswith('_ID') or c in ('NPI', 'MRN'))
        if shared:
            return f"{tbl_a}.{shared[0]} = {tbl_b}.{shared[0]}"

        return None

    def _validate_and_fix_tables(self, resolved: ResolvedPlan) -> ResolvedPlan:
        """Ensure metric columns actually exist in the resolved tables.

        If a metric references a column that doesn't exist in the primary table,
        find the table that has it and add a join.
        """
        for metric in resolved.metrics:
            if metric.column == '*':
                continue
            # Check if column exists in primary table
            primary_cols = {p.name for p in self.learner.tables.get(resolved.primary_table, [])}
            if metric.column not in primary_cols:
                # Find table with this column
                for tbl, profiles in self.learner.tables.items():
                    if any(p.name == metric.column for p in profiles):
                        if tbl != resolved.primary_table:
                            metric.table = tbl
                            # Add join if not already present
                            if not any(t == tbl for t, _ in resolved.join_tables):
                                join_cond = self._find_join_condition(resolved.primary_table, tbl)
                                if join_cond:
                                    resolved.join_tables.append((tbl, join_cond))
                        break

        return resolved


# ─────────────────────────────────────────────────────────────────────
# SQL COMPOSER — ResolvedPlan → SQL
# ─────────────────────────────────────────────────────────────────────

class SQLComposer:
    """Composes SQL from a fully resolved query plan.

    No domain knowledge here — just SQL generation from resolved schema refs.
    """

    def __init__(self, table_row_counts: Dict[str, int] = None):
        self.table_row_counts = table_row_counts or {}

    def compose(self, plan: ResolvedPlan) -> Dict[str, Any]:
        """Generate SQL from a resolved plan."""
        # Fast path: lookup intent with no aggregation → SELECT * FROM table
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

        # Assign table aliases
        aliases = self._assign_aliases(plan)

        # Build SELECT clause
        select_parts = self._build_select(plan, aliases)

        # Build FROM clause
        from_clause = self._build_from(plan, aliases)

        # Build WHERE clause
        where_parts, having_parts = self._build_conditions(plan, aliases)

        # Build GROUP BY
        group_parts = self._build_group_by(plan, aliases)

        # Build ORDER BY
        order_clause = self._build_order_by(plan, aliases, select_parts)

        # Build LIMIT
        limit = plan.limit or (50 if not group_parts else 30)
        limit_clause = f" LIMIT {limit}"

        # Safety: if HAVING exists but no GROUP BY, infer GROUP BY from dimensions
        if having_parts and not group_parts:
            # Find the first non-aggregate, non-expression column in SELECT
            for dim in plan.dimensions:
                if dim.expression:
                    group_parts.append(dim.alias)
                elif dim.column:
                    multi_table = len(aliases) > 1
                    col = self._qualify(dim.column, dim.table, aliases) if multi_table else dim.column
                    group_parts.append(col)
            # If still empty, use the first SELECT column that isn't an aggregate
            if not group_parts:
                for sp in select_parts:
                    if not any(agg in sp.upper() for agg in ['COUNT(', 'SUM(', 'AVG(', 'MIN(', 'MAX(', 'ROUND(']):
                        group_parts.append(sp.split(' as ')[0].strip())
                        break

        # Assemble
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
        """Assign short aliases to tables."""
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
        """Qualify a column with its table alias. Always qualifies."""
        alias = aliases.get(table, table[0] if table else '')
        if '(' in col or '.' in col or ' ' in col:
            return col  # Already qualified or is an expression
        return f"{alias}.{col}"

    def _build_select(self, plan: ResolvedPlan, aliases: Dict) -> List[str]:
        """Build SELECT column list. Always qualifies columns with table aliases."""
        parts = []

        # Add dimensions first (they appear in GROUP BY)
        for dim in plan.dimensions:
            if dim.expression:
                parts.append(f"{dim.expression} as {dim.alias}")
            else:
                col = self._qualify(dim.column, dim.table, aliases)
                parts.append(col)

        # Add metrics
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

        # Add total count if we have other metrics (useful context)
        if plan.metrics and not any(m.column == '*' for m in plan.metrics):
            parts.append("COUNT(*) as record_count")

        # Add percentage of total if requested
        if plan.percentage and plan.dimensions:
            total = self.table_row_counts.get(plan.primary_table, 1)
            parts.append(f"ROUND(100.0 * COUNT(*) / {total}, 2) as pct")

        return parts

    def _build_from(self, plan: ResolvedPlan, aliases: Dict) -> str:
        """Build FROM + JOIN clause. Always uses aliases for consistency.

        Handles three join condition formats:
        1. Full table names: claims.MEMBER_ID = members.MEMBER_ID → c.MEMBER_ID = m.MEMBER_ID
        2. Bare column: MEMBER_ID → primary_alias.MEMBER_ID = join_alias.MEMBER_ID
        3. Partial: NPI=SUPERVISING_NPI → primary_alias.NPI = join_alias.SUPERVISING_NPI
        """
        primary_alias = aliases.get(plan.primary_table, plan.primary_table[0])
        from_parts = f"{plan.primary_table} {primary_alias}"

        for tbl, join_cond in plan.join_tables:
            tbl_alias = aliases.get(tbl, tbl[0])

            # Rewrite full table names to aliases first
            aliased_cond = join_cond
            for full_name, alias in aliases.items():
                aliased_cond = aliased_cond.replace(f"{full_name}.", f"{alias}.")

            # If join condition is just a bare column name (no = sign, no dots),
            # expand it to primary.COL = join.COL
            if '=' not in aliased_cond and '.' not in aliased_cond:
                col = aliased_cond.strip()
                aliased_cond = f"{primary_alias}.{col} = {tbl_alias}.{col}"
            elif '=' in aliased_cond and '.' not in aliased_cond:
                # Format like "COL1=COL2" — qualify both sides
                parts = aliased_cond.split('=')
                if len(parts) == 2:
                    left = parts[0].strip()
                    right = parts[1].strip()
                    aliased_cond = f"{primary_alias}.{left} = {tbl_alias}.{right}"

            from_parts += f" JOIN {tbl} {tbl_alias} ON {aliased_cond}"

        return from_parts

    def _build_conditions(self, plan: ResolvedPlan, aliases: Dict) -> Tuple[List[str], List[str]]:
        """Build WHERE and HAVING conditions. Always qualifies with aliases."""
        where_parts = []
        having_parts = []

        for f in plan.filters:
            cond = f.condition
            if cond.startswith('__THRESHOLD__'):
                # Parse threshold: __THRESHOLD__>=__10
                parts = cond.split('__')
                op = parts[2]
                val = parts[3]
                having_parts.append(f"COUNT(*) {op} {val}")
            else:
                # Always qualify condition with table alias
                alias = aliases.get(f.table, f.table[0])
                cond = self._qualify_condition(cond, alias)
                where_parts.append(cond)

        return where_parts, having_parts

    def _qualify_condition(self, condition: str, alias: str) -> str:
        """Add table alias to column references in a condition.

        Carefully avoids qualifying values inside string literals.
        """
        # Skip if already qualified or is a function call
        if '.' in condition.split('=')[0].split('>')[0].split('<')[0]:
            return condition

        # Split condition on quoted strings to avoid qualifying inside them
        # Pattern: alternate between outside-quotes and inside-quotes segments
        parts = re.split(r"('(?:[^'\\]|\\.)*')", condition)
        result_parts = []
        for i, part in enumerate(parts):
            if i % 2 == 1:
                # Inside quotes — leave as-is
                result_parts.append(part)
            else:
                # Outside quotes — qualify column references
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
        """Build GROUP BY clause. Always qualifies with aliases."""
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
        """Build ORDER BY clause."""
        if not select_parts:
            return ""

        # For trend queries, order by the temporal dimension
        if plan.intent == 'trend':
            for dim in plan.dimensions:
                if dim.expression and dim.alias:
                    return f" ORDER BY {dim.alias}"
            return f" ORDER BY 1"  # First column

        # For rankings, order by the first metric descending
        if plan.intent in ('ranking', 'distribution', 'aggregation'):
            # Find the metric column/alias to sort by
            for metric in plan.metrics:
                if metric.alias:
                    return f" ORDER BY {metric.alias} {plan.sort_direction}"
            return f" ORDER BY COUNT(*) {plan.sort_direction}"

        return " ORDER BY 1"


# ─────────────────────────────────────────────────────────────────────
# UNIFIED ENTRY POINT
# ─────────────────────────────────────────────────────────────────────

def plan_and_compose(question: str, schema_learner, concept_index: Dict,
                     domain_concepts: Dict = None, synonyms: Dict = None,
                     computed_columns: Dict = None, kg=None,
                     schema_intel=None) -> Optional[Dict]:
    """Full pipeline: NL question → QueryPlan → ResolvedPlan → SQL.

    Args:
        schema_intel: Optional SchemaIntelligence for semantic fallback resolution

    Returns dict with {sql, tables_used, confidence, intent} or None if
    the planner can't handle this question (caller should fall back).
    """
    try:
        planner = QueryPlanner(
            concept_index=concept_index,
            domain_concepts=domain_concepts or {},
            synonyms=synonyms or {},
        )
        plan = planner.decompose(question)

        # If the plan has very low confidence, let the old path handle it
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

        # Add plan metadata for debugging
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
