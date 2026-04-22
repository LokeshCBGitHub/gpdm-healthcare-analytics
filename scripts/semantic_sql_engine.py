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

from tokenizer_utils import tokenize_query, tokenize_no_stopwords, get_mentioned_columns

logger = logging.getLogger('gpdm.semantic_sql')


class SchemaKnowledgeGraph:

    def __init__(self, learner: SchemaLearner):
        self.learner = learner
        self.graph = KnowledgeGraph() if KnowledgeGraph else None
        if self.graph:
            self._build()

    def _build(self):
        g = self.graph

        for table, profiles in self.learner.tables.items():
            row_count = self.learner.table_row_counts.get(table, 0)
            g.add_node(f"table:{table}", 'table',
                       row_count=row_count, col_count=len(profiles))

            for p in profiles:
                col_id = f"col:{table}.{p.name}"
                g.add_node(col_id, 'column',
                           table=table, name=p.name,
                           is_numeric=p.is_numeric, is_date=p.is_date,
                           is_categorical=p.is_categorical, is_id=p.is_id,
                           semantic_tags=p.semantic_tags,
                           distinct_count=p.distinct_count)

                weight = 1.0
                if p.is_id:
                    weight = 2.0
                elif p.is_numeric and 'currency' in p.semantic_tags:
                    weight = 1.8
                elif p.is_categorical:
                    weight = 1.5
                elif p.is_date:
                    weight = 1.3
                g.add_edge(f"table:{table}", col_id, 'has_column', weight=weight)

                for tag in p.semantic_tags:
                    type_id = f"type:{tag}"
                    if type_id not in g.nodes:
                        g.add_node(type_id, 'concept')
                    g.add_edge(col_id, type_id, 'is_type', weight=0.8)

                if p.is_categorical:
                    for v in p.sample_values[:10]:
                        val_id = f"val:{table}.{p.name}={v}"
                        g.add_node(val_id, 'value',
                                   table=table, column=p.name, value=v)
                        g.add_edge(col_id, val_id, 'has_value', weight=0.5)

        for t1, neighbors in self.learner.join_graph.items():
            for t2, join_col in neighbors.items():
                if '=' in join_col:
                    left, right = join_col.split('=', 1)
                    g.add_edge(f"table:{t1}", f"table:{t2}", 'joins_to',
                               weight=2.0, left_col=left, right_col=right)
                else:
                    g.add_edge(f"table:{t1}", f"table:{t2}", 'joins_to',
                               weight=2.0, join_column=join_col)

        g.compute_pagerank()

    def find_join_path(self, tables: List[str]) -> List[Tuple[str, str, str]]:
        if not self.graph or len(tables) <= 1:
            return []

        joins = []
        connected = {tables[0]}

        for target in tables[1:]:
            if target in connected:
                continue

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
                join_col = self._get_join_condition(best_source, target)
                joins.append((best_source, target, join_col))
                connected.add(target)
            else:
                join_col = self._get_join_condition(tables[0], target)
                joins.append((tables[0], target, join_col))
                connected.add(target)

        return joins

    def _get_join_condition(self, t1: str, t2: str) -> str:

        cols1 = {p.name.upper() for p in self.learner.tables.get(t1, [])}
        cols2 = {p.name.upper() for p in self.learner.tables.get(t2, [])}
        shared_ids = sorted(c for c in (cols1 & cols2) if c.endswith('_ID'))

        if shared_ids:
            t1_stem = t1.rstrip('s').upper()
            t2_stem = t2.rstrip('s').upper()
            for sid in shared_ids:
                if sid.startswith(t1_stem) or sid.startswith(t2_stem):
                    return sid
            return shared_ids[0]

        join_col = self.learner.join_graph.get(t1, {}).get(t2, '')
        if join_col:
            return join_col
        join_col = self.learner.join_graph.get(t2, {}).get(t1, '')
        if join_col:
            if '=' in join_col:
                left, right = join_col.split('=', 1)
                return f"{right}={left}"
            return join_col

        cols1 = {p.name.upper() for p in self.learner.tables.get(t1, [])}
        cols2 = {p.name.upper() for p in self.learner.tables.get(t2, [])}
        shared = cols1 & cols2

        weak = set()
        for col in shared:
            is_join_key = col.endswith('_ID') or col.endswith('_KEY') or col == 'ID'
            if not is_join_key:
                for p in self.learner.tables.get(t1, []):
                    if p.name.upper() == col and (p.is_categorical or p.is_date):
                        weak.add(col)
                        break
        shared -= weak

        for c in sorted(shared):
            if c.endswith('_ID'):
                return c
        if shared:
            return sorted(shared)[0]

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

        for c in sorted(cols1 & cols2):
            return c
        return ''

    def get_important_columns(self, table: str, top_k: int = 10) -> List[Tuple[str, float]]:
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


class SemanticSQLComposer:


    READMISSION_WINDOW_DAYS = 30
    AGE_BRACKETS = (18, 35, 50, 65)

    MEMBER_ID_CANDIDATES = ('MEMBER_ID', 'PATIENT_ID', 'MRN', 'PERSON_ID',
                            'SUBSCRIBER_ID', 'BENEFICIARY_ID')

    NPI_COLUMN_SUFFIXES = ('_NPI',)
    NPI_COLUMN_NAMES = ('NPI', 'RENDERING_NPI', 'BILLING_NPI',
                        'ATTENDING_NPI', 'PROVIDER_NPI', 'ORDERING_NPI')

    FINANCIAL_COLUMNS = ('BILLED_AMOUNT', 'PAID_AMOUNT', 'ALLOWED_AMOUNT',
                         'MEMBER_RESPONSIBILITY', 'COPAY', 'COINSURANCE',
                         'DEDUCTIBLE', 'TOTAL_CHARGE')

    ENTITY_ID_HINTS = {
        'provider': ('RENDERING_NPI', 'BILLING_NPI', 'NPI', 'PROVIDER_ID'),
        'member':   ('MEMBER_ID', 'PATIENT_ID', 'MRN'),
        'patient':  ('MEMBER_ID', 'PATIENT_ID', 'MRN'),
        'diagnosis': ('DIAGNOSIS_CODE', 'ICD_CODE', 'DX_CODE'),
        'procedure': ('PROCEDURE_CODE', 'CPT_CODE', 'HCPCS_CODE'),
    }

    INSURANCE_KEYWORDS = frozenset({'plan', 'payer', 'insurer', 'coverage'})
    INSURANCE_VALUES = frozenset({'medicaid', 'medicare', 'hmo', 'ppo', 'epo',
                                  'hdhp', 'tricare', 'champva', 'medi-cal',
                                  'commercial', 'self-pay', 'advantage'})
    DISAMBIGUATION_TRIGGERS = ('provider', 'payer', 'insurer', 'plan',
                               'insurance', 'carrier', 'coverage')
    PRACTITIONER_TERMS = frozenset({'doctor', 'physician', 'specialist', 'npi',
                                    'practitioner', 'clinician', 'panel',
                                    'workforce', 'staff', 'specialty'})

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

    MIN_CONCEPT_HITS = _MCH
    MIN_CONCEPT_MATCH_SCORE = _MCM
    MIN_COLUMN_SCORE = _MCS
    MIN_VALUE_SCORE = _MVS
    VALUE_FILTER_SCORE = _VFS
    TABLE_SCORE_RATIO = _TSR
    MIN_COL_PER_TABLE = _MCPT
    MIN_TABLE_SIZE_FOR_METRIC = _MTSM

    HAVING_MIN_USEFUL_DEFAULT = _HMD
    HAVING_MIN_USEFUL_CROSS_METRIC = _HMC
    HAVING_MIN_RATE_THRESHOLD = _HMR

    DEFAULT_BREAKDOWN_LIMIT = _DBL
    DEFAULT_TREND_LIMIT = _DTL
    DEFAULT_RANKING_LIMIT = _DRL
    DEFAULT_LOOKUP_LIMIT = _DLL

    MIN_SEMANTIC_CONFIDENCE = _MSC

    def __init__(self, learner: SchemaLearner, kg: SchemaKnowledgeGraph,
                 inferrer: ComputedColumnInferrer, domain_config=None):
        self.learner = learner
        self.kg = kg
        self.inferrer = inferrer
        self.domain_config = domain_config

        self._schema_intel = None
        try:
            from schema_intelligence import SchemaIntelligence
            db_path = getattr(learner, 'db_path', None)
            if db_path:
                self._schema_intel = SchemaIntelligence(learner, db_path)
                self._schema_intel.learn_or_load()
                from dynamic_sql_engine import DOMAIN_CONCEPTS as _manual_dc, SYNONYMS as _manual_syn
                self._schema_intel.merge_with_manual(_manual_dc, _manual_syn)
                logger.info("SchemaIntelligence: %d concepts (%d auto-discovered), %d synonyms",
                            len(self._schema_intel.domain_concepts),
                            sum(1 for v in self._schema_intel.domain_concepts.values()
                                if v.get('_discovered')),
                            len(self._schema_intel.synonyms))
        except Exception as e:
            logger.warning("SchemaIntelligence unavailable (%s), using manual config only", e)

        self._domain_disambiguations = self._build_domain_disambiguations()
        self._concept_table_index = self._build_concept_table_index()
        self._population_filters = self._build_population_filters()
        self._derived_concepts = self._build_derived_concepts()


    def _build_domain_disambiguations(self) -> Dict[str, Dict]:
        disambiguations = {}

        insurance_keywords = self.INSURANCE_KEYWORDS
        insurance_values = self.INSURANCE_VALUES

        for tbl_name, profiles in self.learner.tables.items():
            for p in profiles:
                if not p.is_categorical:
                    continue
                cn_lower = p.name.lower()

                is_plan_col = any(kw in cn_lower for kw in insurance_keywords)

                has_insurance_vals = False
                if p.sample_values:
                    for val in p.sample_values:
                        val_lower = str(val).lower()
                        if any(iv in val_lower for iv in insurance_values):
                            has_insurance_vals = True
                            break

                if is_plan_col or has_insurance_vals:
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
        term_lower = term.lower().strip()
        term_stem = term_lower.rstrip('s') if term_lower.endswith('s') and len(term_lower) > 3 else term_lower

        matches = []
        for key, dis in self._domain_disambiguations.items():
            if term_lower in dis['triggers'] or term_stem in dis['triggers']:
                in_fact = (dis['table'] == fact)
                matches.append((dis, in_fact))

        if not matches:
            return None

        matches.sort(key=lambda x: (x[1], x[0]['row_count']), reverse=True)
        best = matches[0][0]

        return {
            'table': best['table'],
            'column': best['column'],
            'description': best['description'],
        }


    def _build_population_filters(self) -> Dict[str, Dict]:
        filters = {}

        enc_cols = {p.name for p in self.learner.tables.get('encounters', [])}
        if 'ADMIT_DATE' in enc_cols and 'DISCHARGE_DATE' in enc_cols:
            filters['readmitted'] = {
                'type': 'cte',
                'keywords': ['readmit', 'readmission', 'readmitted', 're-admission', 're-admitted'],
                'description': 'Members readmitted within 30 days of discharge',
                'member_col': 'MEMBER_ID',
                'source_table': 'encounters',
            }

        for tbl_name, profiles in self.learner.tables.items():
            for p in profiles:
                if not p.is_categorical or 'status' not in p.name.lower():
                    continue
                vals = set()
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
                                vals.add(row[0])
                        conn.close()
                except Exception:
                    pass
                if not vals:
                    for v in (p.sample_values or []):
                        if isinstance(v, str):
                            vals.add(v)

                for val in vals:
                    vl = val.lower()
                    keywords = [vl]
                    if vl.endswith('ed') and len(vl) > 4:
                        keywords.append(vl[:-2] + 'al')
                        keywords.append(vl[:-2] + 'y')
                        keywords.append(vl[:-1])
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
        return self._resolve_member_column(table)

    def _detect_population_filter(self, q: str) -> Optional[Dict]:
        q_lower = q.lower()

        agg_words = {
            'average': 'AVG', 'avg': 'AVG', 'mean': 'AVG',
            'total': 'SUM', 'sum': 'SUM', 'count': 'COUNT',
            'median': 'AVG', 'highest': 'MAX', 'max': 'MAX',
            'lowest': 'MIN', 'min': 'MIN', 'number': 'COUNT',
        }

        patterns = [
            r'(average|avg|mean|total|sum|count|median|highest|lowest|max|min|number\s+of)'
            r'\s+(\w+(?:\s+\w+)?)\s+'
            r'(?:for|of|among|per|in)\s+'
            r'(?:\w+\s+)?(\w+)',
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

                matched_filter = self._match_population_filter(pop_word)
                if matched_filter:
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
        word_lower = word.lower().rstrip('s')
        for key, filt in self._population_filters.items():
            for kw in filt['keywords']:
                if word_lower.startswith(kw) or kw.startswith(word_lower):
                    return key
        return None

    def _sql_population_filtered_metric(self, q: str, fact: str,
                                         pop_info: Dict) -> Dict:
        filter_key = pop_info['filter_key']
        agg_fn = pop_info['agg']
        metric_hint = pop_info['metric_hint']
        dim_term = pop_info.get('dimension')
        filt = self._population_filters[filter_key]

        if filt['type'] == 'cte' and 'readmit' in filter_key:
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

        metric_col, metric_table = self._resolve_metric_column(q, metric_hint, fact)

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

        q_words = tokenize_no_stopwords(q, extra_stops={'what'})
        table_scores = defaultdict(float)
        for word in q_words:
            if word in self._concept_table_index:
                for tbl, score in self._concept_table_index[word].items():
                    table_scores[tbl] += score
        if table_scores:
            best_table = max(table_scores, key=table_scores.get)
            for p in self.learner.tables.get(best_table, []):
                if p.is_numeric and not p.is_id and 'currency' in (p.semantic_tags or []):
                    return p.name, best_table
            for p in self.learner.tables.get(best_table, []):
                if p.is_numeric and not p.is_id:
                    return p.name, best_table

        return self.FINANCIAL_COLUMNS[0], default_table


    def _build_derived_concepts(self) -> Dict[str, Dict]:
        concepts = {}

        for tbl_name, profiles in self.learner.tables.items():
            col_names = {p.name.upper() for p in profiles}
            has_member = any(c in col_names for c in self.MEMBER_ID_CANDIDATES)
            has_date = any(p.is_date for p in profiles)
            row_count = self.learner.table_row_counts.get(tbl_name, 0)

            if has_member and has_date and row_count > self.MIN_TABLE_SIZE_FOR_METRIC:
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

        for tbl_name, profiles in self.learner.tables.items():
            col_names_upper = {p.name.upper() for p in profiles}
            col_map = {p.name.upper(): p for p in profiles}

            if 'PAID_AMOUNT' in col_names_upper and 'ALLOWED_AMOUNT' in col_names_upper:
                concepts[f'yield_rate_{tbl_name}'] = {
                    'keywords': ['yield rate', 'yield', 'payment yield',
                                 'paid to allowed', 'reimbursement rate',
                                 'collection rate', 'paid vs allowed ratio'],
                    'type': 'financial_ratio',
                    'table': tbl_name,
                    'numerator': 'PAID_AMOUNT',
                    'denominator': 'ALLOWED_AMOUNT',
                    'label': 'yield_rate',
                    'description': f'Yield rate (paid/allowed) from {tbl_name}',
                }

            has_member = any(c in col_names_upper for c in self.MEMBER_ID_CANDIDATES)
            has_paid = 'PAID_AMOUNT' in col_names_upper
            if has_member and has_paid:
                member_col = self._resolve_member_column(tbl_name)
                concepts[f'pmpm_{tbl_name}'] = {
                    'keywords': ['pmpm', 'cost per member', 'per member per month',
                                 'average cost per member', 'spend per member',
                                 'paid per member', 'pmpm cost', 'our pmpm',
                                 'per member monthly', 'monthly per member',
                                 'member spend', 'member cost',
                                 'cost per patient', 'spend per patient',
                                 'what is our pmpm', 'calculate pmpm'],
                    'type': 'per_member_cost',
                    'table': tbl_name,
                    'cost_col': 'PAID_AMOUNT',
                    'member_col': member_col,
                    'label': 'cost_per_member',
                    'description': f'Cost per member from {tbl_name}',
                }

            financial_cols = [c for c in ['BILLED_AMOUNT', 'PAID_AMOUNT', 'ALLOWED_AMOUNT']
                              if c in col_names_upper]
            if len(financial_cols) >= 2:
                concepts[f'financial_summary_{tbl_name}'] = {
                    'keywords': ['billed vs paid', 'paid vs allowed',
                                 'billed vs paid vs allowed', 'financial summary',
                                 'total billed', 'financial comparison',
                                 'billed paid allowed'],
                    'type': 'multi_column_sum',
                    'table': tbl_name,
                    'columns': financial_cols,
                    'label': 'financial_summary',
                    'description': f'Financial column totals from {tbl_name}',
                }


        for tbl_name, profiles in self.learner.tables.items():
            col_names_upper = {p.name.upper() for p in profiles}
            if 'BILLED_AMOUNT' in col_names_upper and 'PAID_AMOUNT' in col_names_upper:
                concepts[f'profit_margin_{tbl_name}'] = {
                    'keywords': ['profit margin', 'net margin', 'net revenue', 'net income',
                                 'revenue vs cost', 'revenue versus cost', 'total revenue total cost',
                                 'how much profit', 'how much earn', 'margin percent',
                                 'billed minus paid', 'revenue after cost'],
                    'type': 'profit_margin',
                    'table': tbl_name,
                    'revenue_col': 'BILLED_AMOUNT',
                    'cost_col': 'PAID_AMOUNT',
                    'description': f'Profit margin (billed - paid) from {tbl_name}',
                }

        for tbl_name, profiles in self.learner.tables.items():
            col_names_upper = {p.name.upper() for p in profiles}
            if 'CLAIM_STATUS' in col_names_upper and 'BILLED_AMOUNT' in col_names_upper:
                concepts[f'denial_impact_{tbl_name}'] = {
                    'keywords': ['money on the table', 'money left on table', 'money leaving',
                                 'uncollected revenue', 'unrecovered revenue', 'lost revenue',
                                 'denial impact', 'denial revenue', 'denial recovery',
                                 'revenue recover denied', 'recover denied claims',
                                 'potential recovery', 'denial savings'],
                    'type': 'denial_impact',
                    'table': tbl_name,
                    'status_col': 'CLAIM_STATUS',
                    'amount_col': 'BILLED_AMOUNT',
                    'paid_col': 'PAID_AMOUNT' if 'PAID_AMOUNT' in col_names_upper else None,
                    'description': f'Revenue lost to denials from {tbl_name}',
                }

        member_tables = [t for t, profs in self.learner.tables.items()
                         if any(p.name.upper() in self.MEMBER_ID_CANDIDATES for p in profs)
                         and any(p.name.upper() in ('ENROLLMENT_DATE', 'DATE_OF_BIRTH') for p in profs)]
        visit_tables = [t for t, profs in self.learner.tables.items()
                        if any(p.name.upper() in self.MEMBER_ID_CANDIDATES for p in profs)
                        and any(p.is_date for p in profs)
                        and t not in member_tables
                        and self.learner.table_row_counts.get(t, 0) > 100]
        for mt in member_tables:
            mt_member_col = self._resolve_member_column(mt)
            for vt in visit_tables:
                vt_member_col = self._resolve_member_column(vt)
                if mt_member_col and vt_member_col:
                    concepts[f'zero_utilization_{mt}_{vt}'] = {
                        'keywords': ['never visited', 'never visit', 'zero utilization',
                                     'zero encounter', 'zero visit', 'no visit', 'no encounter',
                                     'members never seen', 'never seen doctor',
                                     'zero claims', 'no claims'],
                        'type': 'zero_utilization',
                        'member_table': mt,
                        'visit_table': vt,
                        'member_col': mt_member_col,
                        'visit_member_col': vt_member_col,
                        'description': f'Members in {mt} with no records in {vt}',
                    }

        for tbl_name, profiles in self.learner.tables.items():
            col_names_upper = {p.name.upper() for p in profiles}
            if 'DISENROLLMENT_DATE' in col_names_upper:
                has_risk = 'RISK_SCORE' in col_names_upper
                has_member = any(c in col_names_upper for c in self.MEMBER_ID_CANDIDATES)
                if has_member:
                    concepts[f'disenrollment_{tbl_name}'] = {
                        'keywords': ['disenrolling', 'disenrollment', 'disenroll', 'churn',
                                     'leaving', 'members leaving', 'retention outreach',
                                     'high cost disenroll', 'cost disenroll correlation',
                                     'likely to disenroll', 'at risk disenroll',
                                     'members call outreach', 'retention target',
                                     'retention priority'],
                        'type': 'disenrollment_analysis',
                        'table': tbl_name,
                        'member_col': self._resolve_member_column(tbl_name),
                        'disenroll_col': 'DISENROLLMENT_DATE',
                        'risk_col': 'RISK_SCORE' if has_risk else None,
                        'description': f'Disenrollment analysis from {tbl_name}',
                    }

        for tbl_name, profiles in self.learner.tables.items():
            col_names_upper = {p.name.upper() for p in profiles}
            date_cols = [p for p in profiles if p.is_date and p.name.upper() == 'SERVICE_DATE']
            if date_cols and 'BILLED_AMOUNT' in col_names_upper and 'PAID_AMOUNT' in col_names_upper:
                concepts[f'temporal_financial_{tbl_name}'] = {
                    'keywords': ['month lost', 'month loss', 'worst month', 'biggest loss month',
                                 'biggest financial loss', 'when lost most', 'when lose most',
                                 'worst financial month', 'highest loss month',
                                 'monthly financial', 'financial by month',
                                 'lose the most money', 'lost the most money',
                                 'month did we lose', 'biggest financial',
                                 'when did we have the biggest financial',
                                 'month did we have the biggest'],
                    'type': 'temporal_financial',
                    'table': tbl_name,
                    'date_col': date_cols[0].name,
                    'revenue_col': 'BILLED_AMOUNT',
                    'cost_col': 'PAID_AMOUNT',
                    'description': f'Financial performance over time from {tbl_name}',
                }

        for tbl_name, profiles in self.learner.tables.items():
            col_names_upper = {p.name.upper() for p in profiles}
            if 'KP_REGION' in col_names_upper and 'PAID_AMOUNT' in col_names_upper:
                has_member = any(c in col_names_upper for c in self.MEMBER_ID_CANDIDATES)
                has_status = 'CLAIM_STATUS' in col_names_upper
                if has_member:
                    concepts[f'regional_comparison_{tbl_name}'] = {
                        'keywords': ['compare region', 'region comparison', 'rank region',
                                     'region performance', 'region kpi', 'ncal vs scal',
                                     'scal vs ncal', 'regions across kpi',
                                     'compare ncal', 'compare scal', 'all regions'],
                        'type': 'regional_comparison',
                        'table': tbl_name,
                        'region_col': 'KP_REGION',
                        'member_col': self._resolve_member_column(tbl_name),
                        'cost_col': 'PAID_AMOUNT',
                        'billed_col': 'BILLED_AMOUNT' if 'BILLED_AMOUNT' in col_names_upper else None,
                        'status_col': 'CLAIM_STATUS' if has_status else None,
                        'description': f'Regional comparison from {tbl_name}',
                    }

        provider_tables = [t for t, profs in self.learner.tables.items()
                           if any(p.name.upper() == 'SPECIALTY' for p in profs)]
        for tbl_name, profiles in self.learner.tables.items():
            col_names_upper = {p.name.upper() for p in profiles}
            if 'PAID_AMOUNT' in col_names_upper:
                npi_cols = [p for p in profiles if 'NPI' in p.name.upper()]
                for pt in provider_tables:
                    pt_npi_cols = [p for p in self.learner.tables[pt] if 'NPI' in p.name.upper()]
                    if npi_cols and pt_npi_cols:
                        has_member = any(c in col_names_upper for c in self.MEMBER_ID_CANDIDATES)
                        concepts[f'specialty_cost_{tbl_name}_{pt}'] = {
                            'keywords': ['specialty cost', 'most expensive specialty',
                                         'most costly specialty', 'specialty cost per member',
                                         'cost by specialty', 'spending by specialty',
                                         'provider specialty cost', 'provider cost per member'],
                            'type': 'specialty_cost',
                            'fact_table': tbl_name,
                            'dim_table': pt,
                            'fact_npi_col': npi_cols[0].name,
                            'dim_npi_col': pt_npi_cols[0].name,
                            'cost_col': 'PAID_AMOUNT',
                            'member_col': self._resolve_member_column(tbl_name) if has_member else None,
                            'description': f'Cost by specialty ({tbl_name} → {pt})',
                        }
                        break

        all_cols = set()
        for tbl_name, profiles in self.learner.tables.items():
            for p in profiles:
                all_cols.add(p.name.upper())
        missing_data = []
        if 'DEATH_DATE' not in all_cols and 'DATE_OF_DEATH' not in all_cols:
            missing_data.append('mortality')
        if 'RACE' not in all_cols and 'ETHNICITY' not in all_cols:
            missing_data.append('race_ethnicity')
        if missing_data:
            concepts['data_gaps'] = {
                'keywords': ['members died', 'death', 'mortality', 'mortality rate',
                             'how many died', 'number died', 'death rate',
                             'deceased', 'passed away', 'died last year',
                             'who died', 'member deaths', 'patient deaths'],
                'type': 'data_gap',
                'missing': missing_data,
                'description': 'Data that does not exist in this database',
            }

        return concepts

    def _detect_derived_concept(self, q: str) -> Optional[Tuple[str, Dict]]:
        q_lower = q.lower()
        best_key = None
        best_score = 0
        for key, concept in self._derived_concepts.items():
            score = 0
            for kw in concept['keywords']:
                if kw in q_lower:
                    score += len(kw)
            if score > best_score:
                best_score = score
                best_key = key
        if best_key and best_score >= 5:
            return best_key, self._derived_concepts[best_key]
        return None

    def _sql_derived_concept(self, q: str, fact: str,
                              concept_key: str, concept: Dict) -> Dict:
        q_lower = q.lower()

        if concept['type'] == 'per_member_per_time':
            tbl = concept['table']
            member_col = concept['member_col']
            date_col = concept['date_col']

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

        elif concept['type'] == 'financial_ratio':
            tbl = concept['table']
            num_col = concept['numerator']
            den_col = concept['denominator']
            label = concept.get('label', 'ratio')

            dim = self._discover_dimension(q, tbl)
            if dim:
                dim_col = dim['column']
                dim_tbl = dim.get('table', tbl) if dim.get('needs_join') else tbl
                if dim.get('needs_join'):
                    on_clause = self._dim_on_clause(dim, 't', 'd')
                    sql = (
                        f"SELECT d.{dim_col}, "
                        f"ROUND(100.0 * SUM(CAST(t.{num_col} AS REAL)) / "
                        f"NULLIF(SUM(CAST(t.{den_col} AS REAL)), 0), 2) as {label}, "
                        f"COUNT(*) as record_count "
                        f"FROM {tbl} t JOIN {dim_tbl} d ON {on_clause} "
                        f"GROUP BY d.{dim_col} ORDER BY {label} DESC "
                        f"LIMIT {self.DEFAULT_BREAKDOWN_LIMIT};"
                    )
                    return {'sql': sql, 'tables_used': [tbl, dim_tbl],
                            'confidence': 0.90, 'intent': 'aggregation',
                            'explanation': f'{label} ({num_col}/{den_col}) by {dim_col}'}
                else:
                    sql = (
                        f"SELECT {dim_col}, "
                        f"ROUND(100.0 * SUM(CAST({num_col} AS REAL)) / "
                        f"NULLIF(SUM(CAST({den_col} AS REAL)), 0), 2) as {label}, "
                        f"COUNT(*) as record_count "
                        f"FROM {tbl} "
                        f"GROUP BY {dim_col} ORDER BY {label} DESC "
                        f"LIMIT {self.DEFAULT_BREAKDOWN_LIMIT};"
                    )
                    return {'sql': sql, 'tables_used': [tbl],
                            'confidence': 0.90, 'intent': 'aggregation',
                            'explanation': f'{label} ({num_col}/{den_col}) by {dim_col}'}
            else:
                sql = (
                    f"SELECT ROUND(100.0 * SUM(CAST({num_col} AS REAL)) / "
                    f"NULLIF(SUM(CAST({den_col} AS REAL)), 0), 2) as {label}, "
                    f"ROUND(SUM(CAST({num_col} AS REAL)), 2) as total_{num_col.lower()}, "
                    f"ROUND(SUM(CAST({den_col} AS REAL)), 2) as total_{den_col.lower()}, "
                    f"COUNT(*) as record_count "
                    f"FROM {tbl};"
                )
                return {'sql': sql, 'tables_used': [tbl],
                        'confidence': 0.90, 'intent': 'aggregation',
                        'explanation': f'{label}: {num_col}/{den_col}'}

        elif concept['type'] == 'per_member_cost':
            tbl = concept['table']
            cost_col = concept['cost_col']
            member_col = concept['member_col']
            label = concept.get('label', 'cost_per_member')

            dim = self._discover_dimension(q, tbl)
            if dim:
                dim_col = dim['column']
                if dim.get('needs_join'):
                    dim_tbl = dim.get('table', tbl)
                    on_clause = self._dim_on_clause(dim, 't', 'd')
                    sql = (
                        f"SELECT d.{dim_col}, "
                        f"ROUND(SUM(CAST(t.{cost_col} AS REAL)) / "
                        f"NULLIF(COUNT(DISTINCT t.{member_col}), 0), 2) as {label}, "
                        f"COUNT(DISTINCT t.{member_col}) as unique_members, "
                        f"ROUND(SUM(CAST(t.{cost_col} AS REAL)), 2) as total_cost "
                        f"FROM {tbl} t JOIN {dim_tbl} d ON {on_clause} "
                        f"GROUP BY d.{dim_col} ORDER BY {label} DESC "
                        f"LIMIT {self.DEFAULT_BREAKDOWN_LIMIT};"
                    )
                    return {'sql': sql, 'tables_used': [tbl, dim_tbl],
                            'confidence': 0.90, 'intent': 'aggregation',
                            'explanation': f'{label} by {dim_col}'}
                else:
                    sql = (
                        f"SELECT {dim_col}, "
                        f"ROUND(SUM(CAST({cost_col} AS REAL)) / "
                        f"NULLIF(COUNT(DISTINCT {member_col}), 0), 2) as {label}, "
                        f"COUNT(DISTINCT {member_col}) as unique_members, "
                        f"ROUND(SUM(CAST({cost_col} AS REAL)), 2) as total_cost "
                        f"FROM {tbl} "
                        f"GROUP BY {dim_col} ORDER BY {label} DESC "
                        f"LIMIT {self.DEFAULT_BREAKDOWN_LIMIT};"
                    )
                    return {'sql': sql, 'tables_used': [tbl],
                            'confidence': 0.90, 'intent': 'aggregation',
                            'explanation': f'{label} by {dim_col}'}
            else:
                sql = (
                    f"SELECT ROUND(SUM(CAST({cost_col} AS REAL)) / "
                    f"NULLIF(COUNT(DISTINCT {member_col}), 0), 2) as {label}, "
                    f"COUNT(DISTINCT {member_col}) as unique_members, "
                    f"ROUND(SUM(CAST({cost_col} AS REAL)), 2) as total_cost "
                    f"FROM {tbl};"
                )
                return {'sql': sql, 'tables_used': [tbl],
                        'confidence': 0.90, 'intent': 'aggregation',
                        'explanation': f'{label}: SUM({cost_col})/COUNT(DISTINCT {member_col})'}

        elif concept['type'] == 'multi_column_sum':
            tbl = concept['table']
            cols = concept['columns']
            label = concept.get('label', 'summary')

            select_parts = [
                f"ROUND(SUM(CAST({col} AS REAL)), 2) as total_{col.lower()}"
                for col in cols
            ]
            select_parts.append("COUNT(*) as record_count")

            dim = self._discover_dimension(q, tbl)
            if dim:
                dim_col = dim['column']
                sql = (
                    f"SELECT {dim_col}, {', '.join(select_parts)} "
                    f"FROM {tbl} "
                    f"GROUP BY {dim_col} ORDER BY total_{cols[0].lower()} DESC "
                    f"LIMIT {self.DEFAULT_BREAKDOWN_LIMIT};"
                )
                return {'sql': sql, 'tables_used': [tbl],
                        'confidence': 0.90, 'intent': 'aggregation',
                        'explanation': f'{label}: {", ".join(cols)} by {dim_col}'}
            else:
                sql = f"SELECT {', '.join(select_parts)} FROM {tbl};"
                return {'sql': sql, 'tables_used': [tbl],
                        'confidence': 0.90, 'intent': 'aggregation',
                        'explanation': f'{label}: totals for {", ".join(cols)}'}

        elif concept['type'] == 'profit_margin':
            tbl = concept['table']
            rev = concept['revenue_col']
            cost = concept['cost_col']
            dim = self._discover_dimension(q, tbl)
            if dim:
                dim_col = dim['column']
                dc = f"d.{dim_col}" if dim.get('needs_join') else dim_col
                from_clause = f"{tbl} t"
                if dim.get('needs_join'):
                    dim_tbl = dim.get('table', tbl)
                    on_clause = self._dim_on_clause(dim, 't', 'd')
                    from_clause += f" JOIN {dim_tbl} d ON {on_clause}"
                    prefix = 't.'
                else:
                    prefix = ''
                sql = (
                    f"SELECT {dc}, "
                    f"ROUND(SUM(CAST({prefix}{rev} AS REAL)), 2) AS total_revenue, "
                    f"ROUND(SUM(CAST({prefix}{cost} AS REAL)), 2) AS total_cost, "
                    f"ROUND(SUM(CAST({prefix}{rev} AS REAL)) - SUM(CAST({prefix}{cost} AS REAL)), 2) AS margin, "
                    f"ROUND(100.0 * (SUM(CAST({prefix}{rev} AS REAL)) - SUM(CAST({prefix}{cost} AS REAL))) / "
                    f"NULLIF(SUM(CAST({prefix}{rev} AS REAL)), 0), 2) AS margin_pct, "
                    f"COUNT(*) AS claims "
                    f"FROM {from_clause} "
                    f"GROUP BY {dc} ORDER BY margin DESC "
                    f"LIMIT {self.DEFAULT_BREAKDOWN_LIMIT};"
                )
                return {'sql': sql, 'tables_used': [tbl],
                        'confidence': 0.92, 'intent': 'aggregation',
                        'explanation': f'Profit margin ({rev} - {cost}) by {dim_col}'}
            else:
                sql = (
                    f"SELECT ROUND(SUM(CAST({rev} AS REAL)), 2) AS total_revenue, "
                    f"ROUND(SUM(CAST({cost} AS REAL)), 2) AS total_cost, "
                    f"ROUND(SUM(CAST({rev} AS REAL)) - SUM(CAST({cost} AS REAL)), 2) AS margin, "
                    f"ROUND(100.0 * (SUM(CAST({rev} AS REAL)) - SUM(CAST({cost} AS REAL))) / "
                    f"NULLIF(SUM(CAST({rev} AS REAL)), 0), 2) AS margin_pct, "
                    f"COUNT(*) AS total_claims "
                    f"FROM {tbl} WHERE {cost} > 0;"
                )
                return {'sql': sql, 'tables_used': [tbl],
                        'confidence': 0.92, 'intent': 'aggregation',
                        'explanation': f'Profit margin: ({rev} - {cost}) / {rev}'}

        elif concept['type'] == 'denial_impact':
            tbl = concept['table']
            status_col = concept['status_col']
            amount_col = concept['amount_col']
            paid_col = concept.get('paid_col')
            underpayment = ""
            if paid_col:
                underpayment = (
                    f"SUM(CASE WHEN {status_col}='PAID' THEN CAST({amount_col} AS REAL) - CAST({paid_col} AS REAL) ELSE 0 END) AS underpayment_gap, "
                )
            sql = (
                f"SELECT "
                f"SUM(CASE WHEN {status_col}='DENIED' THEN CAST({amount_col} AS REAL) ELSE 0 END) AS denied_revenue, "
                f"{underpayment}"
                f"COUNT(CASE WHEN {status_col}='DENIED' THEN 1 END) AS denied_claims, "
                f"ROUND(100.0 * SUM(CASE WHEN {status_col}='DENIED' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) AS denial_pct, "
                f"ROUND(AVG(CASE WHEN {status_col}='DENIED' THEN CAST({amount_col} AS REAL) END), 2) AS avg_denied_amount, "
                f"COUNT(*) AS total_claims "
                f"FROM {tbl};"
            )
            return {'sql': sql, 'tables_used': [tbl],
                    'confidence': 0.92, 'intent': 'aggregation',
                    'explanation': f'Revenue impact of denied claims from {tbl}'}

        elif concept['type'] == 'zero_utilization':
            mt = concept['member_table']
            vt = concept['visit_table']
            mc = concept['member_col']
            vc = concept['visit_member_col']
            dim = self._discover_dimension(q, mt)
            if dim and not dim.get('needs_join'):
                dim_col = dim['column']
                sql = (
                    f"SELECT m.{dim_col}, COUNT(*) AS members_never_visited "
                    f"FROM {mt} m LEFT JOIN {vt} v ON m.{mc} = v.{vc} "
                    f"WHERE v.{vc} IS NULL "
                    f"GROUP BY m.{dim_col} ORDER BY members_never_visited DESC;"
                )
            else:
                region_col = None
                for p in self.learner.tables.get(mt, []):
                    if 'REGION' in p.name.upper():
                        region_col = p.name
                        break
                if region_col:
                    sql = (
                        f"SELECT m.{region_col} AS region, COUNT(*) AS members_never_visited, "
                        f"ROUND(AVG(CAST(m.RISK_SCORE AS REAL)), 3) AS avg_risk "
                        f"FROM {mt} m LEFT JOIN {vt} v ON m.{mc} = v.{vc} "
                        f"WHERE v.{vc} IS NULL "
                        f"GROUP BY m.{region_col} ORDER BY members_never_visited DESC;"
                    )
                else:
                    sql = (
                        f"SELECT COUNT(*) AS members_never_visited "
                        f"FROM {mt} m LEFT JOIN {vt} v ON m.{mc} = v.{vc} "
                        f"WHERE v.{vc} IS NULL;"
                    )
            return {'sql': sql, 'tables_used': [mt, vt],
                    'confidence': 0.90, 'intent': 'aggregation',
                    'explanation': f'Members in {mt} with zero records in {vt}'}

        elif concept['type'] == 'disenrollment_analysis':
            tbl = concept['table']
            mc = concept['member_col']
            dc = concept['disenroll_col']
            rc = concept.get('risk_col')
            q_lower = q.lower()

            if any(kw in q_lower for kw in ['outreach', 'call', 'target', 'priority', 'likely to disenroll', 'most likely', 'at risk']):
                risk_expr = f"m.{rc}" if rc else "'N/A'"
                risk_order = f"m.{rc} DESC, " if rc else ""
                sql = (
                    f"SELECT m.{mc}, {risk_expr} AS risk_score, "
                    f"COALESCE(c.total_claims, 0) AS total_claims, "
                    f"COALESCE(ROUND(c.total_paid, 2), 0) AS total_cost "
                    f"FROM {tbl} m LEFT JOIN ("
                    f"  SELECT MEMBER_ID, COUNT(*) AS total_claims, SUM(PAID_AMOUNT) AS total_paid "
                    f"  FROM claims GROUP BY MEMBER_ID"
                    f") c ON m.{mc} = c.MEMBER_ID "
                    f"WHERE m.{dc} = '' "
                    f"ORDER BY {risk_order}c.total_paid DESC LIMIT 100;"
                )
                return {'sql': sql, 'tables_used': [tbl, 'claims'],
                        'confidence': 0.90, 'intent': 'ranking',
                        'explanation': f'Active members ranked by disenrollment risk'}

            risk_expr = f"ROUND(AVG(CAST(m.{rc} AS REAL)), 3) AS avg_risk," if rc else ""
            sql = (
                f"SELECT CASE WHEN m.{dc} != '' THEN 'Disenrolled' ELSE 'Active' END AS status, "
                f"COUNT(*) AS member_count, "
                f"{risk_expr} "
                f"ROUND(AVG(COALESCE(c.total_cost, 0)), 2) AS avg_total_cost "
                f"FROM {tbl} m LEFT JOIN ("
                f"  SELECT MEMBER_ID, SUM(PAID_AMOUNT) AS total_cost FROM claims GROUP BY MEMBER_ID"
                f") c ON m.{mc} = c.MEMBER_ID "
                f"GROUP BY status;"
            )
            return {'sql': sql, 'tables_used': [tbl, 'claims'],
                    'confidence': 0.90, 'intent': 'comparison',
                    'explanation': f'Active vs disenrolled member cost comparison'}

        elif concept['type'] == 'temporal_financial':
            tbl = concept['table']
            date_col = concept['date_col']
            rev = concept['revenue_col']
            cost = concept['cost_col']
            sql = (
                f"SELECT strftime('%Y-%m', {date_col}) AS month, "
                f"ROUND(SUM(CAST({rev} AS REAL)), 2) AS billed, "
                f"ROUND(SUM(CAST({cost} AS REAL)), 2) AS paid, "
                f"ROUND(SUM(CAST({rev} AS REAL)) - SUM(CAST({cost} AS REAL)), 2) AS gap, "
                f"COUNT(*) AS claims "
                f"FROM {tbl} WHERE {date_col} IS NOT NULL AND {date_col} != '' "
                f"GROUP BY month ORDER BY gap DESC LIMIT 12;"
            )
            return {'sql': sql, 'tables_used': [tbl],
                    'confidence': 0.92, 'intent': 'trend',
                    'explanation': f'Monthly financial gap ({rev} - {cost}) from {tbl}'}

        elif concept['type'] == 'regional_comparison':
            tbl = concept['table']
            rc = concept['region_col']
            mc = concept['member_col']
            cost_col = concept['cost_col']
            billed_col = concept.get('billed_col')
            status_col = concept.get('status_col')
            select_parts = [
                f"c.{rc} AS region",
                f"COUNT(DISTINCT m.MEMBER_ID) AS members",
            ]
            if billed_col:
                select_parts.append(
                    f"ROUND(SUM(CAST(c.{cost_col} AS REAL)) / "
                    f"NULLIF(COUNT(DISTINCT m.MEMBER_ID || strftime('%Y-%m', c.SERVICE_DATE)), 0), 2) AS pmpm"
                )
                select_parts.append(
                    f"ROUND(100.0 * SUM(CAST(c.{cost_col} AS REAL)) / "
                    f"NULLIF(SUM(CAST(c.{billed_col} AS REAL)), 0), 2) AS mlr_pct"
                )
            if status_col:
                select_parts.append(
                    f"ROUND(100.0 * SUM(CASE WHEN c.{status_col}='DENIED' THEN 1 ELSE 0 END) / "
                    f"NULLIF(COUNT(*), 0), 2) AS denial_rate"
                )
            select_parts.append(f"ROUND(AVG(CAST(m.RISK_SCORE AS REAL)), 3) AS avg_risk")
            select_parts.append("COUNT(*) AS total_claims")
            sql = (
                f"SELECT {', '.join(select_parts)} "
                f"FROM {tbl} c JOIN members m ON c.{mc} = m.MEMBER_ID "
                f"GROUP BY c.{rc} ORDER BY pmpm DESC;"
            )
            return {'sql': sql, 'tables_used': [tbl, 'members'],
                    'confidence': 0.92, 'intent': 'comparison',
                    'explanation': f'Regional KPI comparison from {tbl}'}

        elif concept['type'] == 'specialty_cost':
            fact_tbl = concept['fact_table']
            dim_tbl = concept['dim_table']
            fact_npi = concept['fact_npi_col']
            dim_npi = concept['dim_npi_col']
            cost_col = concept['cost_col']
            mc = concept.get('member_col')
            if mc:
                sql = (
                    f"SELECT p.SPECIALTY, "
                    f"COUNT(DISTINCT c.{mc}) AS unique_members, "
                    f"ROUND(SUM(CAST(c.{cost_col} AS REAL)), 2) AS total_paid, "
                    f"ROUND(SUM(CAST(c.{cost_col} AS REAL)) / NULLIF(COUNT(DISTINCT c.{mc}), 0), 2) AS cost_per_member, "
                    f"COUNT(*) AS claims "
                    f"FROM {fact_tbl} c JOIN {dim_tbl} p ON c.{fact_npi} = p.{dim_npi} "
                    f"GROUP BY p.SPECIALTY ORDER BY cost_per_member DESC LIMIT 20;"
                )
            else:
                sql = (
                    f"SELECT p.SPECIALTY, "
                    f"ROUND(SUM(CAST(c.{cost_col} AS REAL)), 2) AS total_paid, "
                    f"ROUND(AVG(CAST(c.{cost_col} AS REAL)), 2) AS avg_claim, "
                    f"COUNT(*) AS claims "
                    f"FROM {fact_tbl} c JOIN {dim_tbl} p ON c.{fact_npi} = p.{dim_npi} "
                    f"GROUP BY p.SPECIALTY ORDER BY total_paid DESC LIMIT 20;"
                )
            return {'sql': sql, 'tables_used': [fact_tbl, dim_tbl],
                    'confidence': 0.92, 'intent': 'ranking',
                    'explanation': f'Cost by provider specialty ({fact_tbl} → {dim_tbl})'}

        elif concept['type'] == 'data_gap':
            missing = concept['missing']
            return {
                'sql': '',
                'tables_used': [],
                'confidence': 0.95,
                'intent': 'data_gap',
                'data_gap': True,
                'missing_data': missing,
                'explanation': f'Data gap: {", ".join(missing)} not available in this database',
            }

        return None


    def _compute_adaptive_having(self, table: str, group_col: str,
                                  min_useful: int = 10,
                                  where_clause: str = None,
                                  count_expr: str = None) -> int:
        try:
            import sqlite3
            db_path = getattr(self.learner, 'db_path', None)
            if not db_path:
                return 3

            conn = sqlite3.connect(db_path)
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
            target_keep = min(min_useful, total_groups)
            if target_keep <= 0:
                return 1

            sizes.sort(reverse=True)
            if target_keep <= len(sizes):
                return max(1, sizes[min(target_keep - 1, len(sizes) - 1)])
            return 1
        except Exception:
            return 3

    def _build_concept_table_index(self) -> Dict[str, Dict[str, int]]:
        index = defaultdict(lambda: defaultdict(int))
        stop_words = {'id', 'key', 'code', 'date', 'time', 'type', 'name',
                      'the', 'and', 'for', 'with', 'from', 'into', 'by', 'at',
                      'num', 'is', 'of', 'a', 'an', 'to', 'in', 'on', 'or'}

        for tbl_name, profiles in self.learner.tables.items():
            tbl_lower = tbl_name.lower()
            index[tbl_lower][tbl_name] += 8
            tbl_stem = tbl_lower.rstrip('s') if len(tbl_lower) > 3 else tbl_lower
            if tbl_stem != tbl_lower:
                index[tbl_stem][tbl_name] += 8

            for p in profiles:
                col_lower = p.name.lower()
                col_words = col_lower.replace('_', ' ').split()
                for w in col_words:
                    if w not in stop_words and len(w) > 2:
                        index[w][tbl_name] += 2

                if p.is_categorical and p.sample_values:
                    for val in set(p.sample_values):
                        if val and isinstance(val, str):
                            vl = val.lower().replace('_', ' ')
                            for vw in vl.split():
                                if vw not in stop_words and len(vw) > 2:
                                    index[vw][tbl_name] += 4
                            if vl.endswith('ed') and len(vl) > 4:
                                stem = vl[:-2]
                                index[stem][tbl_name] += 4
                                index[stem + 'al'][tbl_name] += 4
                                index[stem + 'y'][tbl_name] += 3
                            if vl.endswith('ed') and len(vl) > 3:
                                index[vl[:-1]][tbl_name] += 3
                            index[vl][tbl_name] += 5

                if p.is_numeric and not p.is_id:
                    for w in col_words:
                        if w in ('amount', 'cost', 'paid', 'billed', 'allowed',
                                 'charge', 'price', 'revenue', 'total', 'balance',
                                 'copay', 'coinsurance', 'deductible', 'duration',
                                 'quantity', 'dosage', 'refills'):
                            index[w][tbl_name] += 5

                for tag in (p.semantic_tags or []):
                    tag_l = tag.lower()
                    if tag_l not in stop_words and len(tag_l) > 2:
                        index[tag_l][tbl_name] += 3

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

        if self._domain_disambiguations:
            for key, dis in self._domain_disambiguations.items():
                dis_table = dis['table']
                for trigger in dis['triggers']:
                    index[trigger][dis_table] += self.DISAMBIGUATION_INDEX_BOOST
                    for val in dis.get('sample_values', []):
                        val_lower = str(val).lower()
                        for vw in val_lower.split():
                            if len(vw) > 2:
                                index[vw][dis_table] += self.INSURANCE_VALUE_INDEX_BOOST

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
                    index[syn][tbl] += 8

        try:
            from dynamic_sql_engine import DOMAIN_CONCEPTS, SYNONYMS as DYN_SYNONYMS, TABLE_KEYWORDS
            _domain_stop = {'no', 'show', 'second', 'new', 'high', 'low',
                            'annual', 'care', 'date', 'type', 'size',
                            'rate', 'time', 'code', 'score', 'class',
                            'day', 'days', 'home', 'mail', 'order',
                            'value', 'claim', 'visit', 'left',
                            'use', 'based', 'self', 'long', 'short',
                            'term', 'total', 'per', 'all', 'open',
                            'zero', 'mid', 'pre', 'non', 'out',
                            'fee', 'step', 'drug', 'cost', 'pay',
                            'fill', 'lab', 'check', 'well', 'office'}

            for term, concept in DOMAIN_CONCEPTS.items():
                term_words = term.lower().replace('-', ' ').split()
                for tbl in concept.get('tables', []):
                    if tbl in self.learner.tables:
                        for w in term_words:
                            if len(w) > 2 and w not in _domain_stop:
                                index[w][tbl] += 6
                        if len(term_words) > 1:
                            index[term.lower()][tbl] += 10

            for tbl, keywords in TABLE_KEYWORDS.items():
                if tbl in self.learner.tables:
                    for kw in keywords:
                        kw_words = kw.lower().split()
                        for w in kw_words:
                            if len(w) > 2 and w not in _domain_stop:
                                index[w][tbl] += 5
                        if len(kw_words) > 1:
                            index[kw.lower()][tbl] += 8

            for term, col_list in DYN_SYNONYMS.items():
                term_words = term.lower().split()
                for col_name in col_list:
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
        q = question.lower()
        intent_type = intent.get('intent', 'lookup')

        if self._schema_intel:
            _learned = self._schema_intel.find_learned_pattern(question)
            if _learned and _learned.get('count', 0) >= 2:
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

        derived = self._detect_derived_concept(q)
        if derived:
            concept_key, concept_def = derived
            if concept_def['type'] in ('financial_ratio', 'per_member_cost', 'multi_column_sum',
                                        'profit_margin', 'denial_impact', 'zero_utilization',
                                        'disenrollment_analysis', 'temporal_financial',
                                        'regional_comparison', 'specialty_cost', 'data_gap',
                                        'per_member_per_time'):
                fact = self._identify_fact_table(q, tables)
                derived_result = self._sql_derived_concept(q, fact, concept_key, concept_def)
                if derived_result:
                    return self._validate_and_repair_sql(question, derived_result)

        try:
            from query_planner import plan_and_compose
            from dynamic_sql_engine import COMPUTED_COLUMNS as _CC

            if self._schema_intel:
                _dc = self._schema_intel.domain_concepts
                _syn = self._schema_intel.synonyms
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
                        try:
                            db_path = getattr(self.learner, 'db_path', None)
                            if db_path:
                                import sqlite3 as _sq3
                                import time as _vtime
                                _vconn = _sq3.connect(db_path)
                                _vstart = _vtime.time()
                                def _vtimeout():
                                    return 1 if _vtime.time() - _vstart > 15 else 0
                                _vconn.set_progress_handler(_vtimeout, 10000)
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

                if _planner_valid and _planner_sql_final:
                    _group_match = re.search(r'GROUP\s+BY\s+\w*\.?(\w+)', _planner_sql_final, re.IGNORECASE)
                    if _group_match:
                        _group_col = _group_match.group(1)
                        for _t in planner_result.get('tables_used', []):
                            for _p in self.learner.tables.get(_t, []):
                                if _p.name == _group_col and _p.is_id and _p.distinct_count:
                                    _tbl_rows = self.learner.table_row_counts.get(_t, 1)
                                    if _p.distinct_count / max(_tbl_rows, 1) > 0.5:
                                        logger.info("QueryPlanner: GROUP BY near-unique ID %s (%d/%d), deferring",
                                                    _group_col, _p.distinct_count, _tbl_rows)
                                        _planner_valid = False
                                        break

                if _planner_valid:
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

        special = self._detect_special_pattern(q, intent_type, columns, tables, values)
        if special:
            return self._validate_and_repair_sql(question, special)

        primary_tables = self._resolve_tables(tables, columns, values, question)

        _cost_signals = {'cost', 'spend', 'spending', 'expense', 'charge', 'paid',
                         'billed', 'amount', 'dollar', 'price', 'payment', 'revenue'}
        _q_words = tokenize_query(q)
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

        primary_tables = self._expand_tables_for_concepts(q, primary_tables, columns)

        select_parts, group_cols, needs_group = self._build_select(
            q, intent_type, columns, primary_tables, computed, values
        )

        if needs_group and group_cols:
            for gc in group_cols:
                gc_clean = re.sub(r'^\w\.', '', gc)
                found_in_primary = False
                for t in primary_tables:
                    for p in self.learner.tables.get(t, []):
                        if p.name == gc_clean:
                            found_in_primary = True
                            break
                    if found_in_primary:
                        break
                if not found_in_primary:
                    for t, profiles in self.learner.tables.items():
                        if t not in primary_tables:
                            for p in profiles:
                                if p.name == gc_clean:
                                    primary_tables.append(t)
                                    found_in_primary = True
                                    break
                        if found_in_primary:
                            break

        conditions = self._build_conditions(q, values, columns, primary_tables, intent_type)

        from_clause, join_tables = self._build_from(primary_tables, columns, conditions)

        order_clause = self._build_order(q, intent_type, select_parts, group_cols)

        limit = self._build_limit(q, intent_type, needs_group)

        if len(primary_tables) > 1:
            primary = primary_tables[0]
            col_to_tables = defaultdict(list)
            for t in primary_tables:
                for p in self.learner.tables.get(t, []):
                    col_to_tables[p.name].append(t)

            def _qualify(expr):
                for col, tbls in col_to_tables.items():
                    if len(tbls) > 1 and col in expr:
                        alias = primary[0]
                        if f"{alias}.{col}" not in expr and f".{col}" not in expr:
                            expr = re.sub(r'\b' + re.escape(col) + r'\b', f"{alias}.{col}", expr)
                return expr

            select_parts = [_qualify(sp) for sp in select_parts]
            conditions = [_qualify(c) for c in conditions]
            group_cols = [_qualify(gc) for gc in group_cols]

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
        sql = result.get('sql', '')
        q = question.lower()
        if not sql:
            return result

        is_rate_query = any(w in q for w in ('rate', 'denial', 'denied', 'approval',
                                              'approved', 'rejection', 'rejected',
                                              'percentage', 'percent'))
        if is_rate_query and 'CASE WHEN' not in sql and '_rate' not in sql.lower() and 'pct' not in sql.lower():
            logger.debug("Validation: rate query lacks rate computation, attempting repair")
            result['data_warnings'] = result.get('data_warnings', []) + [
                'Rate query detected but SQL lacks rate computation — results may be aggregate counts'
            ]

        sql_cols = set(re.findall(r'\b([A-Z][A-Z_]+(?:_[A-Z]+)*)\b', sql))
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
            repaired = self._attempt_sql_repair(question, result, concept_mismatches)
            if repaired:
                return repaired

        is_trend = any(re.search(p, q) for p in [r'\btrend\b', r'\bover\s+time\b',
                        r'\bby\s+\w+', r'\bper\s+\w+', r'\bacross\s+'])
        has_group = 'GROUP BY' in sql
        if is_trend and not has_group:
            logger.warning("Validation: trend/dimension query lacks GROUP BY — likely returning aggregate")
            result['data_warnings'] = result.get('data_warnings', []) + [
                'Trend/dimension query detected but SQL lacks GROUP BY — result may be a single aggregate row'
            ]
            if isinstance(result.get('confidence'), (int, float)):
                result['confidence'] = min(result['confidence'], 0.5)

        return result

    def _attempt_sql_repair(self, question: str, original: Dict,
                            mismatches: List[str]) -> Optional[Dict]:
        q = question.lower()

        if any('DURATION_MINUTES' in m for m in mismatches):
            correct_table = self._identify_fact_table(q, [])
            if correct_table and correct_table != original.get('tables_used', [''])[0]:
                logger.info("SQL repair: switching from %s to %s",
                           original.get('tables_used', ['?'])[0], correct_table)
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
        table_scores = defaultdict(float)

        for i, t in enumerate(tables):
            weight = 5.0 if i == 0 else 2.0
            table_scores[t['table']] += t['score'] * weight

        col_per_table = defaultdict(int)
        for c in columns[:10]:
            if c['score'] >= 0.1:
                table_scores[c['table']] += c['score'] * 1.5
                col_per_table[c['table']] += 1

        for v in values[:5]:
            if v['score'] >= 0.15:
                table_scores[v['table']] += v['score'] * 1.0

        if not table_scores:
            biggest = max(self.learner.table_row_counts.items(),
                          key=lambda x: x[1], default=(list(self.learner.tables.keys())[0], 0))
            return [biggest[0]]

        q_lower = question.lower()
        q_words = set(re.findall(r'[a-z][a-z0-9_]*', q_lower))
        for table in list(self.learner.tables.keys()):
            if table.lower() in q_lower:
                table_scores[table] = table_scores.get(table, 0) + 10.0
        for table in list(table_scores.keys()):
            if table in q_words or table.rstrip('s') in q_words:
                table_scores[table] += 5.0

        if hasattr(self, '_concept_table_index') and self._concept_table_index:
            for word in q_words:
                if word in self._concept_table_index:
                    for tbl, score in self._concept_table_index[word].items():
                        if tbl in table_scores or tbl in self.learner.tables:
                            table_scores[tbl] += min(score / 2.0, 5.0)

        sorted_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
        primary = sorted_tables[0][0]
        result = [primary]

        for table, score in sorted_tables[1:]:
            explicitly_mentioned = table.lower() in q_lower
            strong_signal = (score >= sorted_tables[0][1] * 0.5 and col_per_table.get(table, 0) >= 1)
            if explicitly_mentioned or strong_signal:
                result.append(table)

        return result[:3]

    def _expand_tables_for_concepts(self, q: str, tables: List[str],
                                     columns: List[Dict]) -> List[str]:
        q_lower = q.lower()

        if self.domain_config:
            concept_hits = self.domain_config.find_concept_from_question(q)
            if not concept_hits:
                return tables

            needed_keywords = set()
            for concept, _tbl, col in concept_hits:
                needed_keywords.add(col.lower())
                for kw in self.domain_config.concept_keywords.get(concept, []):
                    needed_keywords.add(kw.lower())

            for table in tables:
                for p in self.learner.tables.get(table, []):
                    if p.is_numeric and p.name.lower() in needed_keywords:
                        return tables

            for concept, _tbl, _col in concept_hits:
                candidate_tables = self.domain_config.find_tables_with_concept(concept)
                for ct in candidate_tables:
                    if ct not in tables:
                        logger.info("Concept expansion (DomainConfig): added table '%s' for concept '%s'",
                                    ct, concept)
                        tables = tables + [ct]
                        return tables[:3]

            return tables

        q_words = tokenize_query(q_lower)
        signal_words = q_words & {
            w for tbl_profiles in self.learner.tables.values()
            for p in tbl_profiles if p.is_numeric
            for w in p.name.lower().replace('_', ' ').split()
        }
        if not signal_words:
            return tables

        for table in tables:
            for p in self.learner.tables.get(table, []):
                if p.is_numeric:
                    col_words = set(p.name.lower().replace('_', ' ').split())
                    if col_words & q_words:
                        return tables

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
        select_parts = []
        group_cols = []
        needs_group = False
        primary = tables[0] if tables else list(self.learner.tables.keys())[0]
        alias = primary[0] if len(tables) > 1 else ''
        prefix = f"{alias}." if alias else ""

        group_term = self._extract_group_term(q)

        if computed:
            for comp in computed:
                if comp['alias'] == 'age_group' and ('age' in q.lower()):
                    custom_interval = self._extract_age_interval(q)
                    if custom_interval:
                        age_expr = self._age_group_case(comp['column'], interval=custom_interval)
                    else:
                        age_expr = comp['expr']
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
                TIME_UNITS = {'month', 'week', 'year', 'quarter', 'day', 'monthly',
                              'weekly', 'yearly', 'quarterly', 'daily'}
                if group_term.strip().lower() in TIME_UNITS:
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
                logger.debug("Percentage S1: tables=%s values=%s", tables, [(v['table'], v['column'], v['value'], v['score']) for v in values[:5]])
                for v in values:
                    if v['table'] in tables and v['score'] >= 0.1:
                        val_text = v['value'].lower().replace('_', ' ')
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

                if not select_parts:
                    q_words = tokenize_query(q, min_length=3)
                    stop_words = {'the', 'and', 'are', 'what', 'which', 'how', 'for', 'with', 'from', 'have'}
                    q_words -= stop_words

                    _abbrev_map = {
                        'female': 'f', 'male': 'm',
                        'yes': 'y', 'no': 'n',
                        'active': 'a', 'inactive': 'i',
                    }
                    q_abbrevs = {}
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
                                    cursor = conn.execute(
                                        f"SELECT DISTINCT {p.name} FROM {table} WHERE {p.name} IS NOT NULL LIMIT 30"
                                    )
                                    for row in cursor:
                                        if row[0]:
                                            val_lower = row[0].lower()
                                            is_match = any(word in val_lower for word in q_words)
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

                if not select_parts and ('authorization' in q or 'auth' in q or 'prior auth' in q):
                    if 'referrals' in self.learner.tables:
                        for p in self.learner.tables.get('referrals', []):
                            if 'authorization' in p.name.lower():
                                primary = tables[0] if tables else 'claims'
                                select_parts = [
                                    "COUNT(*) as total",
                                    f"SUM(CASE WHEN {p.name} != '' AND {p.name} IS NOT NULL THEN 1 ELSE 0 END) as with_auth",
                                    f"ROUND(100.0 * SUM(CASE WHEN {p.name} != '' AND {p.name} IS NOT NULL THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as auth_pct"
                                ]
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
            rate_info = self._detect_rate_info(q, columns, tables)
            if rate_info:
                status_col, target_val, group_col = rate_info

                time_patterns = [
                    r'\btrend\b', r'\bover\s+time\b', r'\bmonthly\b', r'\bquarterly\b',
                    r'\byearly\b', r'\blast\s+\d+\s+months?\b', r'\bby\s+month\b',
                    r'\bover\s+the\s+(?:last|past)\b',
                ]
                is_time_rate = any(re.search(p, q) for p in time_patterns)
                if is_time_rate and not group_col:
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

        else:
            select_parts = self._default_select(columns, tables)

        return select_parts, group_cols, needs_group

    def _build_conditions(self, q: str, values: List[Dict], columns: List[Dict],
                          tables: List[str], intent_type: str = 'lookup') -> List[str]:
        conditions = []
        used_cols = set()

        skip_filter_for_rate = intent_type in ('percentage', 'rate')

        for v in values:
            if v['score'] >= 0.2 and v['table'] in tables:
                col = v['column']
                val = v['value']
                if col not in used_cols:
                    val_text = val.lower().replace('_', ' ')
                    if val_text in q or val.lower() in q:
                        group_term = self._extract_group_term(q)
                        if group_term and col.lower().replace('_', ' ') in group_term:
                            continue
                        if skip_filter_for_rate:
                            continue
                        conditions.append(f"{col} = '{val}'")
                        used_cols.add(col)

        older_m = re.search(r'\b(?:older|elder)\s+than\s+(\d+)', q)
        younger_m = re.search(r'\b(?:younger)\s+than\s+(\d+)', q)
        if older_m or younger_m:
            age = int((older_m or younger_m).group(1))
            op = '<=' if older_m else '>='
            birth_col = self.learner.find_birth_date_column()
            if birth_col:
                conditions.append(f"{birth_col.name} {op} date('now', '-{age} years')")

        age_compare = re.search(r'\bage\s*(?:>|>=|over|above)\s*(\d+)', q)
        if age_compare:
            age = int(age_compare.group(1))
            birth_col = self.learner.find_birth_date_column()
            if birth_col:
                conditions.append(f"{birth_col.name} <= date('now', '-{age} years')")

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
                    if 2020 <= num_val <= 2030:
                        continue
                except ValueError:
                    continue

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

                num_col = self._find_numeric_column(q, columns, tables)
                if num_col:
                    if op == 'BETWEEN':
                        val2 = m.group(2).replace(',', '')
                        conditions.append(f"CAST({num_col} AS REAL) BETWEEN {val} AND {val2}")
                    else:
                        conditions.append(f"CAST({num_col} AS REAL) {op} {val}")

        ym = re.search(r'(?:in|over|during|for|from|year)?\s*(20[12]\d)\b', q)
        if ym:
            yr = ym.group(1)
            date_col = self._find_date_column(q, columns, tables)
            if date_col:
                conditions.append(f"{date_col} LIKE '{yr}%'")

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

        conditions = self._apply_domain_conditions(q, tables, conditions, used_cols)

        conditions = self._apply_qualitative_thresholds(q, tables, conditions)

        return conditions

    _QUALITATIVE_PATTERNS = [
        (r'\b(?:high|large|big)\s+(?:dollar|cost|amount|value|paid|billed)',
         {'claims': ('CAST(PAID_AMOUNT AS REAL)', '>', 1000),
          'prescriptions': ('CAST(COST AS REAL)', '>', 100)}),
        (r'\b(?:expensive|costly|pricey)\b',
         {'claims': ('CAST(PAID_AMOUNT AS REAL)', '>', 1000),
          'prescriptions': ('CAST(COST AS REAL)', '>', 100),
          'encounters': ('CAST(LENGTH_OF_STAY AS INTEGER)', '>', 5)}),
        (r'\b(?:low|small|cheap)\s+(?:dollar|cost|amount|value|paid|billed)',
         {'claims': ('CAST(PAID_AMOUNT AS REAL)', '<', 500),
          'prescriptions': ('CAST(COST AS REAL)', '<', 20)}),
        (r'\b(?:long|extended|prolonged)\s+(?:stay|los|length)',
         {'encounters': ('CAST(LENGTH_OF_STAY AS INTEGER)', '>', 7)}),
        (r'\b(?:short|brief)\s+(?:stay|los|length)',
         {'encounters': ('CAST(LENGTH_OF_STAY AS INTEGER)', '<', 2)}),
        (r'\bhigh\s+risk\b',
         {'members': ('CAST(RISK_SCORE AS REAL)', '>=', 3.5)}),
        (r'\blow\s+risk\b',
         {'members': ('CAST(RISK_SCORE AS REAL)', '<', 2.0)}),
    ]

    def _apply_domain_conditions(self, q: str, tables: List[str],
                                  conditions: List[str],
                                  used_cols: set) -> List[str]:
        try:
            from dynamic_sql_engine import DOMAIN_CONCEPTS
        except ImportError:
            return conditions

        applied = set()

        sorted_concepts = sorted(DOMAIN_CONCEPTS.items(),
                                  key=lambda x: len(x[0]), reverse=True)

        for term, concept in sorted_concepts:
            if term.lower() not in q:
                continue
            concept_tables = concept.get('tables', [])
            matching_tables = [t for t in concept_tables if t in tables]
            if not matching_tables:
                continue
            new_conds = concept.get('conds', [])
            if not new_conds:
                continue

            skip = False
            for cond in new_conds:
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
        for pattern, table_thresholds in self._QUALITATIVE_PATTERNS:
            if not re.search(pattern, q):
                continue
            for tbl in tables:
                if tbl not in table_thresholds:
                    continue
                col_expr, op, default_val = table_thresholds[tbl]
                threshold = default_val
                try:
                    col_name = re.search(r'(\w+)\s*(?:AS|$)', col_expr)
                    if col_name:
                        raw_col = col_name.group(1)
                        conn = sqlite3.connect(self.learner.db_path)
                        if op in ('>', '>='):
                            row = conn.execute(
                                f"SELECT {col_expr} FROM {tbl} WHERE {raw_col} IS NOT NULL "
                                f"AND {raw_col} != '' ORDER BY {col_expr} DESC "
                                f"LIMIT 1 OFFSET (SELECT COUNT(*)/4 FROM {tbl} WHERE {raw_col} IS NOT NULL AND {raw_col} != '')"
                            ).fetchone()
                        else:
                            row = conn.execute(
                                f"SELECT {col_expr} FROM {tbl} WHERE {raw_col} IS NOT NULL "
                                f"AND {raw_col} != '' ORDER BY {col_expr} ASC "
                                f"LIMIT 1 OFFSET (SELECT COUNT(*)/4 FROM {tbl} WHERE {raw_col} IS NOT NULL AND {raw_col} != '')"
                            ).fetchone()
                        conn.close()
                        if row and row[0] is not None:
                            threshold = round(float(row[0]), 2)
                except Exception:
                    pass

                cond = f"{col_expr} {op} {threshold}"
                if cond not in conditions:
                    conditions.append(cond)
                    logger.info("Qualitative threshold: '%s' → %s (table=%s, threshold=%s)",
                                pattern, cond, tbl, threshold)
                break
        return conditions

    def _build_from(self, tables: List[str], columns: List[Dict],
                    conditions: List[str]) -> Tuple[str, List[str]]:
        if len(tables) <= 1:
            return tables[0] if tables else list(self.learner.tables.keys())[0], []

        primary = tables[0]
        join_tables = []
        join_clauses = []

        joins = self.kg.find_join_path(tables)

        def _generate_alias(table_name: str, existing_aliases: set) -> str:
            for length in range(2, min(5, len(table_name) + 1)):
                candidate = table_name[:length].lower()
                if candidate not in existing_aliases:
                    return candidate
            return table_name.lower()

        alias_set = set()
        primary_alias = _generate_alias(primary, alias_set)
        alias_set.add(primary_alias)
        aliases = {primary: primary_alias}
        parts = [f"{primary} {primary_alias}"]

        for t1, t2, join_col in joins:
            alias = _generate_alias(t2, alias_set)
            alias_set.add(alias)
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
        asc = any(w in q for w in ['lowest', 'smallest', 'least', 'cheapest',
                                    'fewest', 'bottom', 'ascending'])
        direction = 'ASC' if asc else 'DESC'

        if intent_type in ('ranking', 'breakdown', 'percentage'):
            if len(select_parts) > 1:
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


    def _discover_disambiguated_entity(self, q: str, fact: str) -> Optional[Dict]:
        if not self._domain_disambiguations:
            return None

        q_lower = q.lower()
        q_words = tokenize_query(q_lower)
        q_stems = self._stem_words(q_words)

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

        practitioner_terms = self.PRACTITIONER_TERMS
        if q_words & practitioner_terms:
            return None

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

        entity_patterns = [
            rf'\b(?:which|what)\s+{matched_trigger}\w*\s+(?:has|have|is|are|had|show|get)',
            rf'\btop\s+\d*\s*{matched_trigger}\w*',
            rf'\bcompare\s+{matched_trigger}\w*',
            rf'\b{matched_trigger}\w*\s+comparison',
            rf'\b{matched_trigger}\w*\s+(?:with|having)\s+(?:the\s+)?(?:high|low|most|least|best|worst)',
            rf'\b(?:breakdown|distribution|split|summary)\s+(?:of|for|across)\s+{matched_trigger}\w*',
            rf'\b(?:show|list|display|get)\s+(?:me\s+)?(?:all\s+)?{matched_trigger}\w*',
            rf'\b{matched_trigger}\w*\s+and\s+their\b',
            rf'\b{matched_trigger}\w*\s+(?:cost|amount|rate|count|total|average|denial|claim)',
            rf'\b{matched_trigger}\w*\s+(?:ranked|sorted|ordered)',
            rf'\bby\s+{matched_trigger}\w*',
            rf'\b(?:per|each)\s+{matched_trigger}\w*',
        ]

        is_entity = any(re.search(pat, q_lower) for pat in entity_patterns)
        if not is_entity:
            return None

        return self._disambiguate_dimension_term(matched_trigger, fact)

    def _build_disambiguated_dimension(self, disambig: Dict, fact: str) -> Optional[Dict]:
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
                    return None

        return {
            'table': dis_table, 'column': dis_col,
            'needs_join': needs_join, 'join_col': join_col,
            'dim_join_col': dim_join_col,
            'is_age_group': False,
        }

    def _detect_special_pattern(self, q: str, intent_type: str,
                                 columns: List[Dict], tables: List[Dict],
                                 values: List[Dict]) -> Optional[Dict[str, Any]]:
        signals = self._extract_signals(q)
        fact = self._identify_fact_table(q, tables)

        q_lower = q.lower()
        cost_words = {'cost', 'spend', 'spending', 'expense', 'charge', 'paid', 'billed', 'amount', 'dollar', 'price'}
        encounter_words = {'encounter', 'visit', 'admission'}
        q_words_set = tokenize_query(q_lower)
        if fact == 'encounters' and (q_words_set & cost_words) and (q_words_set & encounter_words):
            if 'claims' in {t.get('table') for t in tables} | set(self.learner.tables.keys()):
                fact = 'claims'
                logger.info("Cross-table redirect: cost+encounter → claims (cost data lives in claims)")

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
            metric_col = 'PAID_AMOUNT' if (q_words_set & cost_words) else None
            count_based = 'volume' in q_lower or 'count' in q_lower or 'most' in q_lower or 'highest' in q_lower
            npi_col = 'RENDERING_NPI'
            claim_tbl = 'claims' if 'claims' in self.learner.tables else fact
            if count_based and not metric_col:
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


        derived = self._detect_derived_concept(q)
        if derived:
            concept_key, concept_def = derived
            return self._sql_derived_concept(q, fact, concept_key, concept_def)

        pop_filter = self._detect_population_filter(q)
        if pop_filter:
            return self._sql_population_filtered_metric(q, fact, pop_filter)

        if signals['readmission']:
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

        disambig_entity = self._discover_disambiguated_entity(q, fact)
        if disambig_entity:
            dim = self._build_disambiguated_dimension(disambig_entity, fact)
            if dim:
                logger.info("Layer 4b: disambiguated entity '%s' → %s.%s as dimension",
                            disambig_entity.get('column'), disambig_entity.get('table'),
                            disambig_entity.get('column'))
                return self._sql_universal(q, fact, dim, signals, columns, values)

        dim = self._discover_dimension(q, fact)
        if dim or signals['percentage'] or signals['age_filter'] or signals['birth_year_filter']:
            return self._sql_universal(q, fact, dim, signals, columns, values)

        is_trend_query = any(w in q for w in ['trend', 'over time', 'monthly', 'by month',
                                               'by year', 'quarterly', 'weekly', 'last 12',
                                               'over the last', 'year over year', 'yoy'])
        if is_trend_query:
            return None

        rate_search_tables = sorted(
            self.learner.tables.keys(),
            key=lambda t: self.learner.table_row_counts.get(t, 0),
            reverse=True
        )
        rate_info = self._detect_rate_info(q, columns, rate_search_tables)
        if rate_info:
            status_col, target_val, _ = rate_info
            rate_table = None
            for tbl_name in rate_search_tables:
                for p in self.learner.tables.get(tbl_name, []):
                    if p.name == status_col and p.is_categorical:
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

        for pat in [
            r'(\w+)\s+to\s+(\w+)\s+ratio',
            r'ratio\s+of\s+(\w+)\s+(?:to|vs|and)\s+(\w+)',
            r'(\w+)\s+(?:vs|versus)\s+(\w+)\s+ratio',
        ]:
            m = re.search(pat, q)
            if m:
                signals['ratio'] = (m.group(1).lower(), m.group(2).lower())
                break

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
                if 'bottom' in pat[:20]:
                    n_top, n_bot = n_bot, n_top
                signals['ranking_comparison'] = (n_top, n_bot, entity)
                break

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
                if cat_a in ('top', 'bottom', 'best', 'worst', 'highest', 'lowest'):
                    break
                signals['comparison'] = (cat_a, cat_b, metric)
                break

        for pat in [
            r'(?:members?|patients?|people)\s+(?:with|who\s+had|who\s+have)\s+both\s+(\w+)\s+and\s+(\w+)',
            r'(?:members?|patients?)\s+(?:with|having)\s+(\w+)\s+and\s+(\w+)\s+visits?',
        ]:
            m = re.search(pat, q)
            if m:
                signals['intersection'] = (m.group(1).lower(), m.group(2).lower())
                break

        time_patterns = [
            r'(?:processing|turnaround|adjudication|resolution)\s+time',
            r'time\s+(?:to|for)\s+(?:process|adjudicate|resolve)',
            r'how\s+long\s+(?:to|does\s+it\s+take)',
            r'days?\s+(?:to|between|from)',
        ]
        if any(re.search(pat, q) for pat in time_patterns):
            signals['date_arithmetic'] = True

        if any(w in q for w in ['readmission', 'readmit', 're-admission']):
            signals['readmission'] = True


        if any(w in q for w in ['%', 'percent', 'percentage', 'compared to total', 'of total']):
            signals['percentage'] = True

        age_m = re.search(r'age\s*(?:>|>=|over|above)\s*(\d+)', q)
        if not age_m:
            age_m = re.search(r'(?:older|elder)\s+than\s+(\d+)', q)
        if age_m:
            signals['age_filter'] = int(age_m.group(1))

        year_m = re.search(r'(?:birth|born|dob)\s*(?:year)?\s*(?:<|<=|before)\s*(1[89]\d{2}|20\d{2})', q)
        if year_m:
            signals['birth_year_filter'] = int(year_m.group(1))

        m = re.search(r'(?:most\s+(?:occurring|common|frequent|popular)|top)\s+(\w+(?:\s+\w+)?)', q)
        if m:
            target = m.group(1).lower().strip()
            if re.match(r'^\d+', target):
                pass
            elif target not in ('expensive', 'common', 'frequent'):
                signals['most_occurring'] = target

        return signals

    def _identify_fact_table(self, q: str, tables: List[Dict]) -> str:
        q_lower = q.lower()

        for tbl_name in self.learner.tables:
            if tbl_name.lower() in q_lower:
                pass

        explicit_tables = []
        for tbl_name in self.learner.tables:
            tl = tbl_name.lower()
            if tl in q_lower:
                explicit_tables.append((tbl_name, len(tl)))
        if explicit_tables:
            explicit_tables.sort(key=lambda x: x[1], reverse=True)
            best_explicit = explicit_tables[0][0]
            if len(explicit_tables) == 1 or explicit_tables[0][1] > explicit_tables[1][1]:
                return best_explicit

        q_words = set(re.findall(r'[a-z][a-z0-9_]*', q_lower))
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

        table_scores = defaultdict(float)
        for word in q_words:
            if word in self._concept_table_index:
                for tbl, score in self._concept_table_index[word].items():
                    table_scores[tbl] += score

        for tbl_name in self.learner.tables:
            if tbl_name.lower() in q_lower:
                table_scores[tbl_name] = table_scores.get(tbl_name, 0) + 50.0

        if hasattr(self.learner, 'table_row_counts') and self.learner.table_row_counts:
            for tbl in table_scores:
                row_count = self.learner.table_row_counts.get(tbl, 0)
                if row_count > 0:
                    table_scores[tbl] += min(row_count / self.ROW_COUNT_DIVISOR,
                                              self.ROW_BOOST_CAP)

        practitioner_terms = self.PRACTITIONER_TERMS
        mentions_practitioner = bool(q_words & practitioner_terms)
        q_stems = self._stem_words(q_words)
        if not mentions_practitioner and self._domain_disambiguations:
            for key, dis in self._domain_disambiguations.items():
                for trigger in dis['triggers']:
                    if trigger in q_stems:
                        dis_table = dis['table']
                        table_scores[dis_table] += self.DISAMBIGUATION_FACT_BOOST
                        break

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
                        vals = set(str(v).upper() for v in (p.sample_values or []))
                        if rate_target in vals:
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

        if tables:
            best = tables[0]
            if isinstance(best, dict):
                return best['table']
            return best

        if hasattr(self.learner, 'table_row_counts') and self.learner.table_row_counts:
            return max(self.learner.table_row_counts, key=self.learner.table_row_counts.get)
        return list(self.learner.tables.keys())[0]

    @staticmethod
    def _dim_on_clause(dim: Dict, fact_alias: str = 'f', dim_alias: str = 'd') -> str:
        fact_col = dim.get('join_col', 'MEMBER_ID')
        dim_col = dim.get('dim_join_col') or fact_col
        return f"{fact_alias}.{fact_col} = {dim_alias}.{dim_col}"


    def _build_readmission_cte(self, alias: str = 'ip',
                                extra_columns: List[str] = None,
                                inpatient_val: str = None) -> Tuple[str, str]:
        if not inpatient_val:
            inpatient_val = 'INPATIENT'
            res = self._resolve_category_value('inpatient',
                                               list(self.learner.tables.keys()))
            if res:
                _, _, inpatient_val = res

        enc_table = self._find_table_by_role('encounters')
        member_col = self._resolve_member_column(enc_table)
        admit_col, discharge_col = self._resolve_admission_columns(enc_table)
        visit_type_col = self._resolve_visit_type_column(enc_table)

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
        return role

    def _resolve_member_column(self, table: str) -> str:
        col_names = {p.name.upper() for p in self.learner.tables.get(table, [])}
        for candidate in self.MEMBER_ID_CANDIDATES:
            if candidate in col_names:
                for p in self.learner.tables.get(table, []):
                    if p.name.upper() == candidate:
                        return p.name
        tbl_stem = table.rstrip('s').upper()
        for p in self.learner.tables.get(table, []):
            if p.is_id and not p.name.upper().startswith(tbl_stem):
                return p.name
        return 'MEMBER_ID'

    def _resolve_admission_columns(self, table: str) -> Tuple[str, str]:
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
        for p in self.learner.tables.get(table, []):
            cn = p.name.upper()
            if p.is_categorical and any(w in cn for w in ('VISIT_TYPE', 'ENCOUNTER_TYPE',
                                                           'SERVICE_TYPE', 'CARE_TYPE')):
                return p.name
        return 'VISIT_TYPE'

    @staticmethod
    def _is_npi_column(col_name: str) -> bool:
        cn = col_name.upper()
        return cn in SemanticSQLComposer.NPI_COLUMN_NAMES or cn.endswith('_NPI')

    def _find_npi_column(self, table: str, preferred: str = None) -> str:
        if preferred:
            for p in self.learner.tables.get(table, []):
                if p.name == preferred:
                    return p.name
        for p in self.learner.tables.get(table, []):
            if self._is_npi_column(p.name):
                return p.name
        return 'RENDERING_NPI'

    def _build_result_limit(self, q: str, intent: str = 'breakdown') -> int:
        m = re.search(r'\btop\s+(\d+)\b', q)
        if m:
            return int(m.group(1))
        m = re.search(r'\blimit\s+(\d+)\b', q)
        if m:
            return int(m.group(1))

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
        tbl_lower = table.lower().rstrip('s')
        for hint_key, candidates in self.ENTITY_ID_HINTS.items():
            if hint_key in tbl_lower:
                for cand in candidates:
                    for p in self.learner.tables.get(table, []):
                        if p.name == cand:
                            return p.name
        for p in self.learner.tables.get(table, []):
            if p.is_id and p.name.endswith('_ID'):
                return p.name
        return f"{table.rstrip('s').upper()}_ID"

    def _stem_word(self, word: str) -> Set[str]:
        stems = {word}
        if word.endswith('s') and len(word) > 3:
            stems.add(word[:-1])
        if word.endswith('ies') and len(word) > 4:
            stems.add(word[:-3] + 'y')
        if word.endswith('ed') and len(word) > 4:
            stems.add(word[:-2])
        return stems

    def _stem_words(self, words: Set[str]) -> Set[str]:
        stems = set()
        for w in words:
            stems.update(self._stem_word(w))
        return stems

    def _discover_dimension(self, q: str, fact: str) -> Optional[Dict]:
        group_term = self._extract_group_term(q)
        if not group_term:
            return None

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
                        disambig = None
            if disambig:
                return {
                    'table': dis_table, 'column': dis_col,
                    'needs_join': needs_join, 'join_col': join_col,
                    'dim_join_col': dim_join_col,
                    'is_age_group': False,
                }

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

        candidates = []
        term_lower = group_term.lower().strip()
        tn = term_lower.replace(' ', '_')

        for table, profiles in self.learner.tables.items():
            for p in profiles:
                if not (p.is_categorical or (not p.is_numeric and not p.is_date)):
                    continue
                cn = p.name.lower()
                score = 0.0

                if cn == tn or cn == tn + '_id':
                    score = 10.0
                elif tn.replace('_', '') == cn.replace('_', ''):
                    score = 9.0
                table_stem = table.rstrip('s').lower()
                if cn == f"{table_stem}_{tn}":
                    score = 8.5
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

        candidates.sort(key=lambda x: (x['score'] >= 5.0, x['in_fact'], x['score']), reverse=True)
        best = candidates[0]

        needs_join = not best['in_fact']
        join_col = None
        dim_join_col = None
        if needs_join:
            fact_cols = {p.name for p in self.learner.tables.get(fact, [])}
            dim_cols = {p.name for p in self.learner.tables.get(best['table'], [])}
            shared_ids = sorted(c for c in fact_cols & dim_cols if c.endswith('_ID'))
            if shared_ids:
                join_col = shared_ids[0]
                dim_join_col = shared_ids[0]
            else:
                jc = self.kg._get_join_condition(fact, best['table'])
                if not jc:
                    return None
                if '=' in jc:
                    parts = jc.split('=', 1)
                    join_col = parts[0].strip()
                    dim_join_col = parts[1].strip()
                else:
                    join_col = jc
                    dim_join_col = jc

        return {
            'table': best['table'], 'column': best['column'],
            'needs_join': needs_join, 'join_col': join_col,
            'dim_join_col': dim_join_col,
            'is_age_group': False,
        }

    @staticmethod
    @staticmethod
    def _extract_age_interval(q: str) -> Optional[int]:
        m = re.search(r'(?:by|in|every|each)\s+(\d+)\s*[-\s]?years?', q.lower())
        if m:
            return int(m.group(1))
        m2 = re.search(r'(\d+)\s*[-\s]?year\s+(?:group|bucket|interval|band|range)', q.lower())
        if m2:
            return int(m2.group(1))
        return None

    @staticmethod
    def _age_group_case(col_expr: str, interval: int = None) -> str:
        age_calc = f"(julianday('now') - julianday({col_expr})) / 365.25"

        if interval and interval > 0:
            cases = []
            max_age = 100
            for start in range(0, max_age, interval):
                end = start + interval
                label = f"'{start}-{end - 1}'"
                cases.append(f"WHEN {age_calc} < {end} THEN {label}")
            cases.append(f"ELSE '{max_age}+'")
            return f"CASE {' '.join(cases)} END"
        else:
            return (
                f"CASE "
                f"WHEN {age_calc} < 18 THEN '0-17' "
                f"WHEN {age_calc} < 35 THEN '18-34' "
                f"WHEN {age_calc} < 50 THEN '35-49' "
                f"WHEN {age_calc} < 65 THEN '50-64' "
                f"ELSE '65+' END"
            )

    def _sql_threshold(self, q: str, fact: str, signals: Dict) -> Dict:
        threshold = signals['threshold']

        is_rate_threshold = '%' in q or any(w in q for w in ['rate', 'percentage', 'pct'])
        if is_rate_threshold:
            return self._sql_rate_threshold(q, fact, threshold)

        count_table = fact
        entity_id = self._find_entity_id(count_table)

        entity_words = ['member', 'patient', 'person', 'people', 'individual', 'who']
        subject_is_entity = any(w in q for w in entity_words)
        if subject_is_entity:
            q_words = tokenize_query(q)
            for tbl_name in self.learner.tables:
                tbl_lower = tbl_name.lower()
                tbl_stem = tbl_lower.rstrip('s') if len(tbl_lower) > 3 else tbl_lower
                if tbl_stem in q_words or tbl_lower in q_words:
                    if tbl_lower != 'members' and tbl_lower != 'patients':
                        count_table = tbl_name
                        break
            fact_cols = {p.name for p in self.learner.tables.get(count_table, [])}
            for candidate in ['MEMBER_ID', 'PATIENT_ID', 'MRN']:
                if candidate in fact_cols:
                    entity_id = candidate
                    break

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
        dim = self._discover_dimension(q, fact)
        group_col = f"{dim['table']}.{dim['column']}" if dim else None
        if not group_col:
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

        tables_used = [fact]
        from_clause = fact
        if dim and dim.get('needs_join') and dim['table'] != fact:
            fact_jc = dim.get('join_col', 'MEMBER_ID')
            dim_jc = dim.get('dim_join_col') or fact_jc
            from_clause = f"{fact} JOIN {dim['table']} ON {fact}.{fact_jc} = {dim['table']}.{dim_jc}"
            tables_used.append(dim['table'])

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
        n_top, n_bot, entity_word = ranking

        disambig = self._disambiguate_dimension_term(entity_word, fact)
        if disambig:
            practitioner_terms = {'doctor', 'physician', 'specialist', 'npi', 'practitioner',
                                  'clinician', 'panel', 'workforce', 'staff', 'specialty'}
            q_words = tokenize_query(q)
            if not (q_words & practitioner_terms):
                dim = self._build_disambiguated_dimension(disambig, fact)
                if dim:
                    signals = self._extract_signals(q)
                    return self._sql_universal(q, fact, dim, signals, columns, [])

        entity_table = None
        entity_stem = entity_word.rstrip('s') if len(entity_word) > 2 else entity_word
        for tbl_name in self.learner.tables:
            tbl_lower = tbl_name.lower()
            tbl_stem = tbl_lower.rstrip('s') if len(tbl_lower) > 2 else tbl_lower
            if tbl_stem == entity_stem or tbl_lower == entity_word:
                entity_table = tbl_name
                break
        if not entity_table:
            entity_table = fact

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
                continue
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
        label_col = name_col or desc_col or label_col
        if not label_col:
            for p in entity_profiles:
                if (p.is_categorical and not p.is_id and not p.name.endswith('_ID')
                        and not (hasattr(p, 'is_numeric') and p.is_numeric)):
                    label_col = p.name
                    break
        if not id_col:
            id_col = self._find_entity_id(entity_table)

        metric_col = None
        metric_table = None
        metric_agg = 'SUM'
        entity_cols = {p.name for p in entity_profiles}

        if True:
            entity_id_cols = {p.name for p in entity_profiles if p.is_id or p.name.endswith('_ID')}
            for tbl_name, profiles in self.learner.tables.items():
                if tbl_name == entity_table:
                    continue
                tbl_cols = {p.name for p in profiles}
                shared_ids = entity_id_cols & tbl_cols
                if not shared_ids:
                    for ec in entity_cols:
                        for tc in tbl_cols:
                            if ec == tc and ('NPI' in ec or ec.endswith('_ID')):
                                shared_ids.add(ec)
                    for p in profiles:
                        if self._is_npi_column(p.name):
                            if 'NPI' in entity_cols:
                                shared_ids.add('NPI')
                if shared_ids:
                    for p in profiles:
                        if p.is_numeric and not p.is_id and not p.name.endswith('_ID'):
                            cn = p.name.lower()
                            if any(w in cn for w in ('amount', 'cost', 'paid', 'billed', 'allowed')):
                                metric_col = p.name
                                metric_table = tbl_name
                                break
                    if metric_col:
                        break

        if not metric_col:
            metric_col = self._find_numeric_column(q, columns, [entity_table])
            if metric_col:
                metric_table = entity_table

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
                    metric_col = None
                    metric_table = tbl_name
                    metric_agg = 'COUNT'
                    break

        tables_used = [entity_table]
        if metric_table and metric_table != entity_table:
            tables_used.append(metric_table)

        join_clause = ""
        if metric_table and metric_table != entity_table:
            entity_col_set = {p.name for p in entity_profiles}
            metric_profiles = self.learner.tables.get(metric_table, [])
            metric_col_set = {p.name for p in metric_profiles}

            shared_ids = sorted(c for c in entity_col_set & metric_col_set
                              if c.endswith('_ID') or c == 'NPI')
            if shared_ids:
                jc = shared_ids[0]
                join_clause = f" JOIN {metric_table} ON {entity_table}.{jc} = {metric_table}.{jc}"
            else:
                for p in metric_profiles:
                    if 'NPI' in p.name and 'NPI' in entity_col_set:
                        join_clause = f" JOIN {metric_table} ON {entity_table}.NPI = {metric_table}.{p.name}"
                        break

        group_col = id_col
        select_label = f"{entity_table}.{id_col}"
        if label_col:
            select_label = f"{entity_table}.{label_col}"
            group_col = label_col
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
        for table in [fact] + list(self.learner.tables.keys()):
            dates = self.learner.get_date_columns(table)
            if len(dates) < 2:
                continue
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
        group_term = None
        for pat in [r'by\s+(\w+(?:\s+\w+)?)', r'per\s+(\w+)', r'across\s+(?:all\s+)?(\w+(?:\s+\w+)?)']:
            m = re.search(pat, q)
            if m:
                group_term = m.group(1).lower().strip()
                break

        time_patterns = [r'\btrend\b', r'\bover\s+time\b', r'\bmonthly\b', r'\bquarterly\b',
                         r'\byearly\b', r'\bby\s+month\b', r'\bby\s+quarter\b', r'\bby\s+year\b',
                         r'\blast\s+\d+\s+months?\b', r'\bover\s+the\s+(?:last|past)\b']
        is_time_query = any(re.search(p, q) for p in time_patterns)

        enc_table = self._find_table_by_role('encounters')
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
        enc_columns = [member_col, admit_col, discharge_col] + extra_cols
        total_expr = f"COUNT(DISTINCT ip.{member_col})"

        if is_time_query or (group_term and group_term in ('month', 'quarter', 'year', 'time')):
            bucket_field = 'quarter'
            if 'month' in q:
                bucket = "SUBSTR(ADMIT_DATE, 1, 7)"
                bucket_field = 'month'
            elif 'year' in q:
                bucket = "SUBSTR(ADMIT_DATE, 1, 4)"
                bucket_field = 'year'
            else:
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

        if group_term:
            dim_col = None
            dim_join = None

            for p in self.learner.tables.get('encounters', []):
                cn = p.name.lower()
                if group_term in cn or cn.replace('_', ' ') == group_term:
                    dim_col = p.name
                    break
                col_words = cn.replace('_', ' ').split()
                if group_term.rstrip('s') in [w.rstrip('s') for w in col_words]:
                    dim_col = p.name
                    break

            if not dim_col:
                candidates = []
                for tbl_name, profiles in self.learner.tables.items():
                    if tbl_name == 'encounters':
                        continue
                    for p in profiles:
                        cn = p.name.lower()
                        cn_words = cn.replace('_', ' ').split()
                        gt = group_term.rstrip('s') if len(group_term) > 3 else group_term
                        gt_words = gt.split()
                        col_match = (group_term in cn or gt in cn.replace('_', ' ') or
                                     cn.replace('_', ' ').replace('kp ', '') == group_term)
                        if not col_match and len(gt_words) > 1:
                            primary_word = gt_words[0].rstrip('s') if len(gt_words[0]) > 3 else gt_words[0]
                            if primary_word in cn or any(primary_word == cw.rstrip('s') for cw in cn_words):
                                col_match = True
                        elif not col_match:
                            gw_stem = gt.rstrip('s') if len(gt) > 3 else gt
                            if any(gw_stem == cw.rstrip('s') for cw in cn_words):
                                col_match = True
                        if not (p.is_categorical and not p.is_id and col_match):
                            continue
                        enc_cte_set = set(enc_columns)
                        tbl_cols_set = {pp.name for pp in profiles}
                        shared = enc_cte_set & tbl_cols_set
                        if 'MEMBER_ID' in shared:
                            join = f"JOIN {tbl_name} ON ip.MEMBER_ID = {tbl_name}.MEMBER_ID"
                            candidates.append((tbl_name, p.name, join, 10))
                        elif tbl_name == 'providers':
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

                if candidates:
                    candidates.sort(key=lambda c: (-c[3],
                        self.learner.table_row_counts.get(c[0], 0) if hasattr(self.learner, 'table_row_counts') else 0))
                    best = candidates[0]
                    dim_col = best[1]
                    dim_join = best[2]

            if not dim_col:
                resolved = self._resolve_group_column(group_term, [], ['encounters'])
                if resolved and resolved in enc_columns:
                    dim_col = resolved

            if dim_col:
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
        q_lower = q.lower()

        filter_patterns = [
            r'(?:average|avg|mean|total|sum|count|median)\s+\w+\s+(?:for|of|among)\s+(?:\w+\s+)?readmit',
            r'(?:cost|amount|charge|paid|billed|spending)\s+(?:for|of|among)\s+(?:\w+\s+)?readmit',
            r'readmit\w*\s+(?:patient|member|cost|charge|amount|spending)',
        ]
        if any(re.search(p, q_lower) for p in filter_patterns):
            return None

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
        inpatient_val = 'INPATIENT'
        res = self._resolve_category_value('inpatient', ['encounters'])
        if res:
            _, _, inpatient_val = res

        status_col = 'CLAIM_STATUS'
        target_val = 'DENIED'
        if cross_metric == 'denial':
            res = self._resolve_category_value('denied', ['claims'])
            if res:
                _, status_col, target_val = res
        elif cross_metric == 'approval':
            res = self._resolve_category_value('approved', ['claims'])
            if res:
                _, status_col, target_val = res

        claims_table = self._find_table_by_role('claims')
        enc_table = self._find_table_by_role('encounters')
        prov_table = self._find_table_by_role('providers')
        claims_npi = self._find_npi_column(claims_table)
        enc_npi = self._find_npi_column(enc_table)

        provider_name = 'NPI'
        for p in self.learner.tables.get(prov_table, []):
            if any(w in p.name.lower() for w in ('name', 'first', 'last', 'specialty')):
                provider_name = p.name
                break

        safe_metric = re.sub(r'[^a-zA-Z0-9_]', '_', cross_metric.lower())
        safe_target = re.sub(r'[^a-zA-Z0-9_]', '_', target_val.lower())

        if cross_metric in ('denial', 'approval'):
            claims_having = self._compute_adaptive_having('claims', claims_npi, min_useful=self.HAVING_MIN_USEFUL_CROSS_METRIC)
            enc_having = self._compute_adaptive_having(
                'encounters', enc_npi, min_useful=10,
                where_clause=f"VISIT_TYPE = '{inpatient_val}' AND DISCHARGE_DATE != '' AND ADMIT_DATE != ''",
                count_expr='COUNT(DISTINCT MEMBER_ID)'
            )

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
        year = None
        ym = re.search(r'(?:in|for|during)\s+(20[12]\d)', q)
        if ym:
            year = ym.group(1)

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
        q_words = tokenize_query(q, min_length=3)
        stop = {'the', 'and', 'are', 'what', 'which', 'how', 'for', 'with', 'from',
                'have', 'show', 'claims', 'claim', 'total', 'volume', 'compared', 'percentage'}
        q_words -= stop

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

        if not q_words:
            return None

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
                        is_match = any(w in val_lower for w in q_words)
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

        if signals.get('percentage') and not dim:
            result = self._sql_value_percentage(q, fact)
            if result:
                return result

        if not dim:
            return None

        total = self.learner.table_row_counts.get(fact, 1)

        if dim.get('is_age_group'):
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

        if dim['needs_join']:
            on_clause = self._dim_on_clause(dim, 'f', 'd')

            search_tables = self._expand_tables_for_concepts(q, [fact], columns)
            num_col = self._find_numeric_column(q, columns, search_tables)
            has_amount = self._has_amount_word(q) and num_col
            agg_func = self._detect_agg_func(q)

            num_table = fact
            if num_col:
                for t in search_tables:
                    for p in self.learner.tables.get(t, []):
                        if p.name == num_col and t != fact:
                            num_table = t
                            break

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

        rate_info = self._detect_rate_info(q, columns, [fact])
        if rate_info:
            status_col, target_val, _ = rate_info
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

        search_tables = self._expand_tables_for_concepts(q, [fact], columns)
        num_col = self._find_numeric_column(q, columns, search_tables)
        has_amount = self._has_amount_word(q) and num_col
        agg_func = self._detect_agg_func(q)

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


    def _extract_group_term(self, q: str) -> Optional[str]:
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
            if label == 'per' and has_agg_per:
                continue
            m = re.search(pat, q)
            if m:
                raw = m.group(1).strip()
                noise = {'the', 'a', 'an', 'each', 'every', 'all', 'their', 'its',
                          'or', 'on', 'of', 'front', 'side', 'basis', 'end',
                          'changed', 'changes', 'over', 'time', 'trend', 'trended'}
                words = [w for w in raw.split() if w.lower() not in noise and len(w) > 1]
                return ' '.join(words) if words else None
        return None

    def _extract_ranking_entity(self, q: str) -> Optional[str]:
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
        if not term:
            return None

        term_lower = term.lower().strip()

        schema_columns = {tbl: [p.name for p in self.learner.tables.get(tbl, [])]
                         for tbl in self.learner.tables}
        exact_matches = get_mentioned_columns(term, schema_columns, target_table=tables[0] if tables else None)
        for match in exact_matches:
            return match.name

        if not exact_matches and tables:
            exact_matches_all = get_mentioned_columns(term, schema_columns)
            for match in exact_matches_all:
                return match.name

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
                    return name_pair[0]

        term_stem = term_lower.rstrip('s') if len(term_lower) > 2 else term_lower
        for tbl_name, profiles in self.learner.tables.items():
            tbl_lower = tbl_name.lower()
            tbl_stem = tbl_lower.rstrip('s') if len(tbl_lower) > 2 else tbl_lower
            if tbl_stem == term_stem or tbl_lower == term_lower:
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
                        continue
                    elif not p.is_id and not p.name.endswith('_ID'):
                        if any(w in pn for w in ('name', 'first_name', 'last_name', 'fname', 'lname')):
                            if not name_col:
                                name_col = p.name
                        elif any(w in pn for w in ('description', 'desc', 'text', 'detail')):
                            if not desc_col:
                                desc_col = p.name
                        elif (p.is_categorical and any(w in pn for w in
                                  ('specialty', 'type', 'title', 'label',
                                   'department', 'category', 'icd', 'dx', 'code'))):
                            if not label_col:
                                label_col = p.name
                label_col = name_col or desc_col or label_col
                best_col = label_col or id_col
                if best_col:
                    for t in tables:
                        for p in self.learner.tables.get(t, []):
                            if p.name == best_col:
                                return best_col
                    return best_col
                break

        tn = term_lower.replace(' ', '_')
        for table in tables:
            for p in self.learner.tables.get(table, []):
                cn = p.name.lower()
                if cn == tn or cn == tn + '_id':
                    return p.name
                if tn.replace('_', '') == cn.replace('_', ''):
                    return p.name

        for table in tables:
            table_stem = table.rstrip('s').lower()
            qualified = f"{table_stem}_{tn}"
            for p in self.learner.tables.get(table, []):
                cn = p.name.lower()
                if cn == qualified or cn == qualified.replace('_', ''):
                    return p.name

        if self.learner:
            group_results = []
            for table in tables:
                row_count = self.learner.table_row_counts.get(table, 1)
                for p in self.learner.tables.get(table, []):
                    cn = p.name.lower()
                    score = 0.0
                    for word in term_lower.split():
                        if len(word) >= 3:
                            if word in cn:
                                score += 2.0
                            elif word in cn.replace('_', ' '):
                                score += 1.5
                            for tag in p.semantic_tags:
                                if word in tag:
                                    score += 0.5
                    if score > 0 and (p.is_categorical or p.is_id):
                        if p.is_id and p.distinct_count:
                            cardinality_ratio = p.distinct_count / max(row_count, 1)
                            if cardinality_ratio > 0.5:
                                score -= 10.0
                        if p.is_categorical and not p.is_id:
                            score += 3.0
                        if any(w in cn for w in ('name', 'description', 'class',
                                                  'type', 'category', 'specialty')):
                            score += 2.0
                        group_results.append((p.name, table, score))

            if group_results:
                group_results.sort(key=lambda x: x[2], reverse=True)
                if group_results[0][2] > 0:
                    return group_results[0][0]

        for c in columns:
            if c['table'] in tables:
                cn = c['column'].lower()
                if term_lower in cn or cn in term_lower.replace(' ', '_'):
                    if c.get('meta', {}).get('is_categorical') or c.get('meta', {}).get('is_id'):
                        return c['column']

        for table in tables:
            for p in self.learner.tables.get(table, []):
                cn = p.name.lower()
                for word in term_lower.split():
                    if len(word) >= 3 and word in cn:
                        if p.is_categorical or p.is_id:
                            return p.name

        for table, profiles in self.learner.tables.items():
            if table in tables:
                continue
            for p in profiles:
                cn = p.name.lower()
                if cn == tn or cn == tn + '_id':
                    return p.name
            for p in profiles:
                cn = p.name.lower()
                for word in term_lower.split():
                    if len(word) >= 3 and word in cn:
                        if p.is_categorical or p.is_id or not p.is_numeric:
                            return p.name

        return None

    def _has_amount_word(self, q: str) -> bool:
        q_words = tokenize_query(q)

        if self.domain_config:
            for w in q_words:
                if self.domain_config.is_amount_word(w):
                    return True
                w_stem = w.rstrip('s') if len(w) > 3 else w
                if w_stem != w and self.domain_config.is_amount_word(w_stem):
                    return True

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
        if any(w in q for w in ['average', 'avg', 'mean']):
            return 'AVG'
        if any(w in q for w in ['total', 'sum', 'combined', 'revenue', 'sales',
                                    'spending', 'income', 'earnings', 'gross']):
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
        schema_columns = {tbl: [p.name for p in self.learner.tables.get(tbl, [])]
                         for tbl in self.learner.tables}
        exact_matches = get_mentioned_columns(q, schema_columns, target_table=tables[0] if tables else None)
        for match in exact_matches:
            for p in self.learner.tables.get(match.source_table, []):
                if p.name == match.name and p.is_numeric:
                    return match.name

        if not exact_matches:
            exact_matches_all = get_mentioned_columns(q, schema_columns)
            for match in exact_matches_all:
                for p in self.learner.tables.get(match.source_table, []):
                    if p.name == match.name and p.is_numeric:
                        return match.name

        q_words = tokenize_query(q)

        candidates = []
        for table in tables:
            for p in self.learner.tables.get(table, []):
                if not p.is_numeric:
                    continue
                col_name_lower = p.name.lower()
                if col_name_lower.endswith('_id'):
                    continue
                if p.is_id:
                    if col_name_lower not in q_words:
                        continue
                col_parts = set(p.name.lower().replace('_', ' ').split())
                col_parts -= {'id', 'key', 'code', 'num'}
                overlap = q_words & col_parts
                if overlap:
                    score = len(overlap) * 2.0
                    if 'currency' in p.semantic_tags:
                        score += 1.0
                    candidates.append((p.name, score))
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]

        for c in columns:
            if c['table'] in tables and c.get('meta', {}).get('is_numeric'):
                if c.get('meta', {}).get('is_id') or c['column'].lower().endswith('_id'):
                    continue
                if 'currency' in c.get('meta', {}).get('semantic_tags', []):
                    return c['column']

        for c in columns:
            if c['table'] in tables and c.get('meta', {}).get('is_numeric'):
                if c.get('meta', {}).get('is_id') or c['column'].lower().endswith('_id'):
                    continue
                return c['column']

        for table in tables:
            for p in self.learner.tables.get(table, []):
                if p.is_numeric and 'currency' in p.semantic_tags:
                    return p.name

        for table in tables:
            for p in self.learner.tables.get(table, []):
                if p.is_numeric and not p.is_id and not p.name.lower().endswith('_id'):
                    return p.name

        return None

    def _find_date_column(self, q: str, columns: List[Dict],
                          tables: List[str]) -> Optional[str]:
        schema_columns = {tbl: [p.name for p in self.learner.tables.get(tbl, [])]
                         for tbl in self.learner.tables}
        exact_matches = get_mentioned_columns(q, schema_columns, target_table=tables[0] if tables else None)
        for match in exact_matches:
            for p in self.learner.tables.get(match.source_table, []):
                if p.name == match.name and p.is_date:
                    return match.name

        if not exact_matches:
            exact_matches_all = get_mentioned_columns(q, schema_columns)
            for match in exact_matches_all:
                for p in self.learner.tables.get(match.source_table, []):
                    if p.name == match.name and p.is_date:
                        return match.name

        if self.domain_config:
            for table in tables:
                date_cols = self.domain_config.find_date_columns(table)
                if date_cols:
                    return date_cols[0].name

        else:
            q_words = tokenize_query(q)
            best_col = None
            best_score = -1
            for table in tables:
                dates = self.learner.get_date_columns(table)
                for d in dates:
                    col_words = set(d.name.lower().replace('_', ' ').split())
                    col_words -= {'date', 'dt', 'ts', 'timestamp'}
                    overlap = len(col_words & q_words)
                    audit_penalty = -2 if any(w in d.name.lower() for w in ['created', 'updated', 'modified']) else 0
                    score = overlap + audit_penalty
                    if score > best_score:
                        best_score = score
                        best_col = d.name
            if best_col:
                return best_col

        for c in columns:
            if c['table'] in tables and c.get('meta', {}).get('is_date'):
                return c['column']

        for table in tables:
            dates = self.learner.get_date_columns(table)
            if dates:
                return dates[0].name

        return None

    def _detect_rate_info(self, q: str, columns: List[Dict],
                          tables: List[str]) -> Optional[Tuple[str, str, Optional[str]]]:
        rate_m = re.search(r'(\w+)\s+rate', q)
        if not rate_m:
            rate_m = re.search(r'rate\s+(?:of|for)\s+(\w+)', q)
        if not rate_m:
            return None

        rate_word = rate_m.group(1).lower()

        stems = {rate_word}
        if rate_word.endswith('al'):
            stems.add(rate_word[:-2])
            stems.add(rate_word[:-2] + 'ed')
            stems.add(rate_word[:-2] + 'y')
        if rate_word.endswith('tion'):
            stems.add(rate_word[:-4])
        stems.add(rate_word + 'ed')
        stems.add(rate_word + 'led')
        stems = {s for s in stems if len(s) >= 3}

        def _match_value(val):
            vl = val.lower().replace('_', ' ')
            return any(stem in vl or vl.startswith(stem) for stem in stems)

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
        if not tables:
            return ['*']

        primary = tables[0]
        important = self.kg.get_important_columns(primary, top_k=8)

        if important:
            return [col for col, _ in important]

        selected = []
        seen = set()
        for c in columns:
            if c['table'] == primary and c['column'] not in seen:
                selected.append(c['column'])
                seen.add(c['column'])
            if len(selected) >= 8:
                break

        return selected if selected else ['*']

    _CONFIDENCE_STOP_WORDS = frozenset({
        'what', 'is', 'the', 'a', 'an', 'of', 'in', 'for', 'to', 'and',
        'or', 'by', 'on', 'at', 'from', 'with', 'my', 'our', 'me', 'us',
        'show', 'tell', 'give', 'get', 'find', 'list', 'display', 'see',
        'can', 'do', 'does', 'did', 'how', 'about', 'this', 'that',
        'are', 'were', 'was', 'be', 'been', 'have', 'has', 'had',
        'all', 'each', 'every', 'some', 'any', 'no', 'not',
        'it', 'its', 'i', 'we', 'they', 'he', 'she', 'their',
        'please', 'just', 'also', 'very', 'much', 'many',
        'trend', 'over', 'last', 'month', 'months', 'year', 'years',
        'quarter', 'quarterly', 'weekly', 'daily', 'monthly', 'yearly',
        'time', 'period', 'date', 'during', 'since', 'between',
        'average', 'total', 'count', 'sum', 'number', 'rate',
        'top', 'highest', 'lowest', 'most', 'least', 'best', 'worst',
        'per', 'compare', 'versus', 'breakdown',
        'which', 'where', 'there',
    })

    def _calculate_confidence(self, intent: Dict, columns: List[Dict],
                              tables: List[Dict], values: List[Dict],
                              conditions: List[str],
                              question: str = '') -> float:
        score = 0.3

        intent_conf = intent.get('confidence', 0)
        score += min(0.2, intent_conf * 0.2)

        if columns:
            avg_col_score = sum(c['score'] for c in columns[:5]) / min(len(columns), 5)
            score += min(0.2, avg_col_score * 0.3)

        if tables:
            avg_table_score = sum(t['score'] for t in tables[:3]) / min(len(tables), 3)
            score += min(0.15, avg_table_score * 0.2)

        if values:
            score += min(0.1, len(values) * 0.03)

        if conditions:
            score += min(0.05, len(conditions) * 0.02)

        if question and hasattr(self, '_concept_table_index'):
            q_words = tokenize_query(question)
            content_words = q_words - self._CONFIDENCE_STOP_WORDS
            if content_words:
                resolved = set()
                for w in content_words:
                    if w in self._concept_table_index and self._concept_table_index[w]:
                        resolved.add(w)
                        continue
                    for c in (columns or []):
                        if w in c.get('column', '').lower() or w in c.get('table', '').lower():
                            resolved.add(w)
                            break
                unresolved = content_words - resolved
                coverage = len(resolved) / len(content_words) if content_words else 1.0
                if coverage < 0.5:
                    penalty = 0.25
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


class SemanticSQLEngine:

    MIN_SEMANTIC_CONFIDENCE = 0.35

    def __init__(self, db_path: str, catalog_dir: str = None, nlp_mode: str = 'auto'):
        self.db_path = db_path
        self.catalog_dir = catalog_dir
        self.nlp_mode = nlp_mode

        self.nlp_factory = None
        try:
            from nlp_engine_factory import NLPEngineFactory
            self.nlp_factory = NLPEngineFactory(mode=nlp_mode)
            logger.info("NLP factory: mode=%s, tier=%d",
                        nlp_mode, self.nlp_factory.inventory.best_tier)
        except Exception as e:
            logger.warning("NLP factory unavailable (%s), using scratch backends", e)

        self.semantic = SemanticLayer(db_path, nlp_factory=self.nlp_factory)
        self.semantic.initialize()

        self.kg = SchemaKnowledgeGraph(self.semantic.learner)

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

        self.composer = SemanticSQLComposer(
            self.semantic.learner, self.kg, self.semantic.inferrer,
            domain_config=self.domain_config
        )

        self._fallback = None
        try:
            from dynamic_sql_engine import DynamicSQLEngine
            self._fallback = DynamicSQLEngine(catalog_dir, db_path)
        except Exception:
            pass

        logger.info("SemanticSQLEngine initialized — schema-agnostic, data-driven, nlp=%s",
                     nlp_mode)

    def generate(self, question: str, neural_context: Dict[str, Any] = None,
                 nlp_enrichment: Dict[str, Any] = None) -> Dict[str, Any]:
        neural_context = neural_context or {}
        nlp_enrichment = nlp_enrichment or {}

        try:
            from dynamic_sql_engine import normalize_typos
            question = normalize_typos(question)
        except ImportError:
            pass

        intent = self.semantic.classify_intent(question)
        tfidf_intent = intent['intent']
        tfidf_conf = intent['confidence']

        recalled_intents = {}
        for meta, score in neural_context.get('recalled_patterns', []):
            if score > 0.3 and isinstance(meta, dict):
                ri = meta.get('intent', '')
                if ri:
                    recalled_intents[ri] = max(recalled_intents.get(ri, 0), score)

        if recalled_intents:
            best_recalled = max(recalled_intents, key=recalled_intents.get)
            recall_conf = recalled_intents[best_recalled]
            if best_recalled != tfidf_intent and recall_conf > tfidf_conf * 0.8:
                blended_scores = dict(intent.get('all_scores', {}))
                for ri, rs in recalled_intents.items():
                    blended_scores[ri] = blended_scores.get(ri, 0) * 0.6 + rs * 0.4
                best_fused = max(blended_scores, key=blended_scores.get)
                if best_fused != tfidf_intent:
                    logger.info("Intent fused: TF-IDF=%s(%.2f) + Hopfield=%s(%.2f) → %s",
                                tfidf_intent, tfidf_conf, best_recalled, recall_conf, best_fused)
                    intent['intent'] = best_fused
                    intent['confidence'] = blended_scores[best_fused]
                    intent['all_scores'] = blended_scores
                    intent['fusion'] = {'tfidf': tfidf_intent, 'hopfield': best_recalled,
                                        'method': 'weighted_blend'}

        logger.info("Intent: %s (%.2f) [fused=%s]", intent['intent'], intent['confidence'],
                     'fusion' in intent)

        columns = self.semantic.match_columns(question, top_k=15)

        schema_matches = neural_context.get('schema_matches', [])
        if schema_matches:
            neural_col_names = set()
            neural_tbl_names = set()
            for sm in schema_matches:
                node_id = sm.get('node_id', '')
                if '.' in node_id:
                    parts = node_id.split('.', 1)
                    neural_tbl_names.add(parts[0].upper())
                    neural_col_names.add(parts[1].upper())
                else:
                    neural_tbl_names.add(node_id.upper())

            for col in columns:
                col_name = col.get('column', '').upper()
                tbl_name = col.get('table', '').upper()
                if col_name in neural_col_names:
                    col['score'] = col.get('score', 0) + 0.15
                    col['neural_boosted'] = True
                if tbl_name in neural_tbl_names:
                    col['score'] = col.get('score', 0) + 0.05

            columns.sort(key=lambda c: c.get('score', 0), reverse=True)

        expanded_terms = set(t.upper().replace(' ', '_') for t in nlp_enrichment.get('expanded_terms', []))
        if expanded_terms:
            for col in columns:
                col_name = col.get('column', '').upper()
                for et in expanded_terms:
                    if et in col_name or col_name in et:
                        col['score'] = col.get('score', 0) + 0.10
                        col['ontology_boosted'] = True
                        break
            columns.sort(key=lambda c: c.get('score', 0), reverse=True)

        tables = self.semantic.match_tables(question, top_k=5)

        _INTERNAL_PREFIXES = ('gpdm_', '_gpdm_', '_dq_', '_schema_', '_data_', '_audit_',
                              'sqlite_', 'query_patterns')
        tables = [t for t in tables
                  if not any(t.get('table', '').lower().startswith(p) for p in _INTERNAL_PREFIXES)]

        if schema_matches:
            for tbl in tables:
                if tbl.get('table', '').upper() in neural_tbl_names:
                    tbl['score'] = tbl.get('score', 0) + 0.15
                    tbl['neural_boosted'] = True
            tables.sort(key=lambda t: t.get('score', 0), reverse=True)

        values = self.semantic.match_values(question, top_k=10)

        ner_entities = nlp_enrichment.get('entities', [])
        if ner_entities:
            for entity_text, entity_type in ner_entities:
                already_matched = any(
                    v.get('value', '').upper() == entity_text.upper()
                    for v in values
                )
                if not already_matched:
                    entity_col = self._resolve_entity_column(entity_text, entity_type)
                    if entity_col:
                        values.append({
                            'table': entity_col['table'],
                            'column': entity_col['column'],
                            'value': entity_text,
                            'score': 0.85,
                            'source': 'ner',
                            'entity_type': entity_type,
                        })
                        logger.info("NER entity injected: %s (%s) → %s.%s",
                                    entity_text, entity_type,
                                    entity_col['table'], entity_col['column'])

        computed = self.semantic.find_computed_columns(question)

        derived = self.composer._detect_derived_concept(question)
        if derived:
            concept_key, concept_def = derived
            logger.info("Derived concept matched: %s (type=%s) for: %s",
                        concept_key, concept_def['type'], question[:60])
            matched_tables = self.semantic.match_tables(question, top_k=3)
            matched_tables = [t for t in matched_tables
                              if not any(t.get('table', '').lower().startswith(p)
                                         for p in _INTERNAL_PREFIXES)]
            fact_table = matched_tables[0]['table'] if matched_tables else 'claims'
            derived_result = self.composer._sql_derived_concept(question, fact_table, concept_key, concept_def)
            if derived_result:
                derived_result.setdefault('semantic_intent', intent['intent'])
                derived_result.setdefault('semantic_confidence', intent['confidence'])
                derived_result.setdefault('intent_backend', intent.get('backend', 'scratch'))
                derived_result.setdefault('intent_fusion', intent.get('fusion', None))
                derived_result.setdefault('semantic_columns', columns[:5])
                derived_result.setdefault('semantic_values', values[:5])
                derived_result.setdefault('neural_boosted_columns', [])
                derived_result.setdefault('ontology_boosted_columns', [])
                derived_result.setdefault('ner_injected_values', [])
                derived_result.setdefault('columns_resolved', [])
                derived_result.setdefault('filters', [])
                derived_result.setdefault('agg_info', {})
                derived_result.setdefault('used_fallback', False)
                derived_result.setdefault('data_warnings', [])
                derived_result.setdefault('data_suggestions', [])
                derived_result['_derived_concept'] = True
                derived_result['_derived_concept_type'] = concept_def['type']
                validation = self.semantic.validate(derived_result['sql'], derived_result.get('tables_used', []))
                derived_result['data_warnings'] = validation.get('warnings', [])
                derived_result['data_suggestions'] = validation.get('suggestions', [])
                return derived_result

        result = self.composer.compose(question, intent, columns, tables, values, computed)

        validation = self.semantic.validate(result['sql'], result['tables_used'])
        result['data_warnings'] = validation.get('warnings', [])
        result['data_suggestions'] = validation.get('suggestions', [])

        entities = self.semantic.extract_entities(question)
        if entities:
            result['entities'] = entities

        result['explanation_detail'] = self._build_explanation(
            question, intent, columns[:5], tables[:3], values[:5],
            result.get('sql', ''), neural_context, nlp_enrichment
        )

        result['semantic_intent'] = intent['intent']
        result['semantic_confidence'] = intent['confidence']
        result['intent_backend'] = intent.get('backend', 'scratch')
        result['intent_fusion'] = intent.get('fusion', None)
        result['semantic_columns'] = columns[:5]
        result['semantic_values'] = values[:5]
        result['neural_boosted_columns'] = [c for c in columns[:10] if c.get('neural_boosted')]
        result['ontology_boosted_columns'] = [c for c in columns[:10] if c.get('ontology_boosted')]
        result['ner_injected_values'] = [v for v in values if v.get('source') == 'ner']
        result['columns_resolved'] = [
            {'column': c['column'], 'table': c['table'],
             'match_type': 'neural' if c.get('neural_boosted') else 'semantic',
             'original_term': question}
            for c in columns[:10]
        ]
        result['filters'] = []
        result['agg_info'] = {
            'agg_func': intent['intent'].upper() if intent['intent'] in ('count', 'aggregate') else None,
            'group_by_terms': [],
            'top_n': None,
            'order': 'DESC',
        }

        if result['confidence'] < self.MIN_SEMANTIC_CONFIDENCE and self._fallback:
            logger.info("Low semantic confidence (%.2f), falling back to hardcoded engine",
                        result['confidence'])
            fallback_result = self._fallback.generate(question)

            fallback_result['semantic_intent'] = intent['intent']
            fallback_result['semantic_confidence'] = intent['confidence']
            fallback_result['semantic_columns'] = columns[:5]
            fallback_result['semantic_values'] = values[:5]
            fallback_result['used_fallback'] = True

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

    def _resolve_entity_column(self, entity_text: str, entity_type: str) -> Optional[Dict]:
        type_patterns = {
            'ICD10': ['DIAGNOSIS_CODE', 'ICD_CODE', 'DX_CODE', 'PRIMARY_DIAGNOSIS'],
            'CPT': ['CPT_CODE', 'PROCEDURE_CODE', 'SERVICE_CODE'],
            'DRUG': ['MEDICATION_NAME', 'DRUG_NAME', 'NDC_CODE', 'MEDICATION'],
            'PROVIDER': ['PROVIDER_NAME', 'RENDERING_PROVIDER', 'PROVIDER_ID'],
            'FACILITY': ['FACILITY_NAME', 'FACILITY_TYPE', 'SERVICE_LOCATION'],
            'PLAN': ['PLAN_TYPE', 'PLAN_NAME', 'INSURANCE_PLAN'],
            'SPECIALTY': ['SPECIALTY', 'PROVIDER_SPECIALTY'],
            'STATE': ['STATE', 'MEMBER_STATE', 'PROVIDER_STATE'],
            'GENDER': ['GENDER', 'MEMBER_GENDER', 'SEX'],
        }
        patterns = type_patterns.get(entity_type, [])
        for tbl_name, profiles in self.semantic.learner.tables.items():
            for p in profiles:
                col_upper = p.name.upper()
                if col_upper in patterns:
                    return {'table': tbl_name, 'column': p.name}
                if entity_type.lower() in col_upper.lower():
                    return {'table': tbl_name, 'column': p.name}
        if self.semantic.learner:
            for tbl_name, profiles in self.semantic.learner.tables.items():
                for p in profiles:
                    vals = getattr(p, 'top_values', None) or getattr(p, 'sample_values', None)
                    if p.is_categorical and vals:
                        upper_vals = [str(v).upper() for v in vals]
                        if entity_text.upper() in upper_vals:
                            return {'table': tbl_name, 'column': p.name}
        return None

    def _build_explanation(self, question: str, intent: Dict,
                           columns: List[Dict], tables: List[Dict],
                           values: List[Dict], sql: str,
                           neural_context: Dict, nlp_enrichment: Dict) -> Dict[str, Any]:
        explanation = {
            'reasoning_steps': [],
            'confidence_factors': [],
            'data_sources': [],
        }

        intent_step = f"Classified as '{intent['intent']}' query"
        if intent.get('fusion'):
            fusion = intent['fusion']
            intent_step += (f" (fused TF-IDF '{fusion['tfidf']}' with "
                            f"memory recall '{fusion['hopfield']}')")
        intent_step += f" with {intent['confidence']:.0%} confidence"
        explanation['reasoning_steps'].append(intent_step)

        if tables:
            tbl_names = [t['table'] for t in tables[:3]]
            tbl_step = f"Selected table(s): {', '.join(tbl_names)}"
            neural_tbls = [t['table'] for t in tables[:3] if t.get('neural_boosted')]
            if neural_tbls:
                tbl_step += f" (neural GNN boosted: {', '.join(neural_tbls)})"
            explanation['reasoning_steps'].append(tbl_step)

        if columns:
            col_names = [f"{c['table']}.{c['column']}" for c in columns[:5]]
            col_step = f"Resolved columns: {', '.join(col_names)}"
            explanation['reasoning_steps'].append(col_step)

        if values:
            val_descriptions = []
            for v in values[:3]:
                src = f" (NER:{v['entity_type']})" if v.get('source') == 'ner' else ''
                val_descriptions.append(f"{v['column']}='{v['value']}'{src}")
            if val_descriptions:
                explanation['reasoning_steps'].append(
                    f"Applied filters: {', '.join(val_descriptions)}"
                )

        if nlp_enrichment.get('expanded_terms'):
            explanation['reasoning_steps'].append(
                f"Ontology expanded: {', '.join(list(nlp_enrichment['expanded_terms'])[:5])}"
            )

        if intent['confidence'] > 0.8:
            explanation['confidence_factors'].append('High intent confidence')
        if neural_context.get('schema_matches'):
            explanation['confidence_factors'].append(
                f"Neural schema context ({len(neural_context['schema_matches'])} matches)")
        if neural_context.get('recalled_patterns'):
            good_recalls = [s for _, s in neural_context['recalled_patterns'] if s > 0.3]
            if good_recalls:
                explanation['confidence_factors'].append(
                    f"Memory recall ({len(good_recalls)} similar past queries)")

        explanation['plain_english'] = self._sql_to_english(sql, intent, columns, tables, values)
        explanation['sql'] = sql

        return explanation

    def _sql_to_english(self, sql: str, intent: Dict, columns: List[Dict],
                         tables: List[Dict], values: List[Dict]) -> str:
        if not sql:
            return "No SQL generated."

        parts = []
        intent_name = intent.get('intent', 'lookup')

        sql_upper = sql.upper()
        if 'NULLIF' in sql_upper and '/' in sql_upper and ('100.0' in sql or 'RATE' in sql_upper):
            parts.append("Calculating the ratio/rate of")
        elif 'AVG(' in sql_upper:
            parts.append("Calculating the average of")
        elif 'SUM(' in sql_upper and 'COUNT' in sql_upper:
            parts.append("Computing aggregates for")
        elif 'SUM(' in sql_upper:
            parts.append("Summing up")
        elif 'COUNT(DISTINCT' in sql_upper:
            parts.append("Counting distinct")
        elif 'COUNT' in sql_upper:
            parts.append("Counting")
        elif 'MAX(' in sql_upper:
            parts.append("Finding the maximum")
        elif 'MIN(' in sql_upper:
            parts.append("Finding the minimum")
        elif 'CASE' in sql_upper and 'WHEN' in sql_upper:
            parts.append("Evaluating conditional logic for")
        else:
            parts.append("Looking up")

        if columns:
            col_names = [c['column'].replace('_', ' ').lower() for c in columns[:3]]
            parts.append(', '.join(col_names))

        if tables:
            parts.append(f"from the {tables[0]['table'].replace('_', ' ')} data")

        if 'WHERE' in sql.upper():
            parts.append("with specific filters applied")

        if 'GROUP BY' in sql.upper():
            import re
            gb_match = re.search(r'GROUP\s+BY\s+(\w+(?:\.\w+)?)', sql, re.IGNORECASE)
            if gb_match:
                gb_col = gb_match.group(1).split('.')[-1].replace('_', ' ').lower()
                parts.append(f"grouped by {gb_col}")

        if 'ORDER BY' in sql.upper():
            if 'DESC' in sql.upper():
                parts.append("sorted from highest to lowest")
            else:
                parts.append("sorted from lowest to highest")

        if 'LIMIT' in sql.upper():
            import re
            lim_match = re.search(r'LIMIT\s+(\d+)', sql, re.IGNORECASE)
            if lim_match:
                parts.append(f"showing top {lim_match.group(1)} results")

        return ' '.join(parts) + '.'


if __name__ == '__main__':
    pass
