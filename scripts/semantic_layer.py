import os
import re
import json
import math
import sqlite3
import logging
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Optional, Set


logger = logging.getLogger('gpdm.semantic')

try:
    from gpdm_config import TFIDF_MIN_SEARCH_SCORE
except ImportError:
    TFIDF_MIN_SEARCH_SCORE = 0.08


class SparseVector:
    __slots__ = ('indices', 'values', '_norm')

    def __init__(self, indices: List[int], values: List[float]):
        self.indices = indices
        self.values = values
        self._norm = math.sqrt(sum(v * v for v in values)) if values else 0.0

    def cosine(self, other: 'SparseVector') -> float:
        if self._norm == 0 or other._norm == 0:
            return 0.0
        dot = 0.0
        i, j = 0, 0
        while i < len(self.indices) and j < len(other.indices):
            if self.indices[i] == other.indices[j]:
                dot += self.values[i] * other.values[j]
                i += 1
                j += 1
            elif self.indices[i] < other.indices[j]:
                i += 1
            else:
                j += 1
        return dot / (self._norm * other._norm)


class TFIDFIndex:

    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_count: int = 0
        self.documents: Dict[str, SparseVector] = {}
        self.doc_metadata: Dict[str, Dict] = {}

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = text.lower()
        words = re.findall(r'[a-z][a-z0-9_]*', text)
        tokens = list(words)
        for i in range(len(words) - 1):
            tokens.append(f"{words[i]}_{words[i+1]}")
        return tokens

    def build(self, documents: List[Tuple[str, str, Dict]]) -> None:
        doc_freq = Counter()
        all_tokens = set()

        for doc_id, text, meta in documents:
            tokens = self._tokenize(text)
            unique = set(tokens)
            for t in unique:
                doc_freq[t] += 1
            all_tokens.update(unique)

        self.doc_count = len(documents)

        for i, token in enumerate(sorted(all_tokens)):
            self.vocab[token] = i

        for token, idx in self.vocab.items():
            df = doc_freq.get(token, 0)
            self.idf[token] = math.log((self.doc_count + 1) / (df + 1)) + 1.0

        for doc_id, text, meta in documents:
            vec = self._vectorize(text)
            self.documents[doc_id] = vec
            self.doc_metadata[doc_id] = meta

    def _vectorize(self, text: str) -> SparseVector:
        tokens = self._tokenize(text)
        tf = Counter(tokens)
        indices, values = [], []
        for token, count in sorted(tf.items()):
            if token in self.vocab:
                idx = self.vocab[token]
                tf_val = 1 + math.log(count) if count > 0 else 0
                idf_val = self.idf.get(token, 1.0)
                indices.append(idx)
                values.append(tf_val * idf_val)
        return SparseVector(indices, values)

    def search(self, query: str, top_k: int = 10, min_score: float = None) -> List[Tuple[str, float, Dict]]:
        if min_score is None:
            min_score = TFIDF_MIN_SEARCH_SCORE
        qvec = self._vectorize(query)
        results = []
        for doc_id, dvec in self.documents.items():
            score = qvec.cosine(dvec)
            if score >= min_score:
                results.append((doc_id, score, self.doc_metadata.get(doc_id, {})))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


class ColumnProfile:
    __slots__ = (
        'table', 'name', 'data_type', 'sample_values', 'distinct_count',
        'null_pct', 'is_numeric', 'is_date', 'is_categorical', 'is_id',
        'is_text', 'min_val', 'max_val', 'semantic_tags',
    )

    def __init__(self, table: str, name: str):
        self.table = table
        self.name = name
        self.data_type = 'text'
        self.sample_values: List[str] = []
        self.distinct_count = 0
        self.null_pct = 0.0
        self.is_numeric = False
        self.is_date = False
        self.is_categorical = False
        self.is_id = False
        self.is_text = False
        self.min_val = None
        self.max_val = None
        self.semantic_tags: List[str] = []


class SchemaLearner:

    _NAME_PATTERNS = [
        (r'(?:^|_)id$|_id$|^id_', 'identifier'),
        (r'date|_dt$|_ts$|timestamp|_at$|created|updated|modified', 'date'),
        (r'amount|price|cost|fee|charge|paid|billed|revenue|salary|total|balance', 'currency'),
        (r'count|num|qty|quantity|number|size|length', 'count'),
        (r'rate|ratio|percent|pct|score|index', 'rate'),
        (r'name|title|label|description|desc|text|comment|note|reason', 'text_descriptive'),
        (r'status|state|stage|phase|flag|type|category|class|kind|group|level|tier', 'categorical'),
        (r'code|key|ref|tag', 'code'),
        (r'email|phone|address|city|state|zip|country|street|url', 'contact'),
        (r'npi|ssn|mrn|ein|tin|license|dea', 'regulated_id'),
        (r'first_name|last_name|fname|lname|given_name|family_name', 'person_name'),
        (r'birth|dob|age', 'birth_date'),
        (r'region|facility|department|location|site|branch|office', 'location'),
        (r'gender|sex|race|ethnicity|language|religion', 'demographic'),
    ]

    _VALUE_PATTERNS = [
        (r'^\d{4}-\d{2}-\d{2}', 'date'),
        (r'^\d{4}-\d{2}-\d{2}T', 'datetime'),
        (r'^-?\d+\.?\d*$', 'numeric'),
        (r'^[A-Z][a-z]+$', 'proper_noun'),
        (r'^[A-Z]{2,6}$', 'code_upper'),
        (r'^\d{10}$', 'phone_or_npi'),
    ]

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.tables: Dict[str, List[ColumnProfile]] = {}
        self.table_row_counts: Dict[str, int] = {}
        self.join_graph: Dict[str, Dict[str, str]] = defaultdict(dict)
        self._column_lookup: Dict[str, List[ColumnProfile]] = defaultdict(list)

    def learn(self, sample_size: int = 500) -> Dict[str, Any]:
        conn = sqlite3.connect(self.db_path)
        try:
            tables = [row[0] for row in
                      conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()]

            for table in tables:
                self._profile_table(conn, table, sample_size)

            self._discover_joins()

            try:
                from column_graph import ColumnRelationshipGraph, ColumnCooccurrenceMatrix
                self.column_graph = ColumnRelationshipGraph(self)
                self.column_graph.discover()
                self.cooccurrence_matrix = ColumnCooccurrenceMatrix()
                logger.info("Column graph: %d edges, %d code-label pairs, %d name pairs",
                           len(self.column_graph.edges),
                           len(self.column_graph.code_label_pairs),
                           len(self.column_graph.name_pairs))
            except Exception as e:
                logger.warning("Column graph init failed (non-fatal): %s", e)
                self.column_graph = None
                self.cooccurrence_matrix = None

            return {
                'tables': len(self.tables),
                'total_columns': sum(len(cols) for cols in self.tables.values()),
                'total_rows': sum(self.table_row_counts.values()),
                'join_paths': sum(len(v) for v in self.join_graph.values()),
                'column_relationships': len(self.column_graph.edges) if self.column_graph else 0,
            }
        finally:
            conn.close()

    def _profile_table(self, conn: sqlite3.Connection, table: str, sample_size: int):
        try:
            row_count = conn.execute(f"SELECT COUNT(*) FROM [{table}]").fetchone()[0]
        except Exception:
            return
        self.table_row_counts[table] = row_count

        try:
            pragma = conn.execute(f"PRAGMA table_info([{table}])").fetchall()
        except Exception:
            return

        profiles = []
        for col_info in pragma:
            _, col_name, col_type, _, _, _ = col_info
            profile = ColumnProfile(table, col_name)
            profile.data_type = col_type or 'text'

            try:
                sample_rows = conn.execute(
                    f"SELECT [{col_name}] FROM [{table}] WHERE [{col_name}] IS NOT NULL "
                    f"AND [{col_name}] != '' LIMIT {sample_size}"
                ).fetchall()
                values = [str(r[0]) for r in sample_rows]
                profile.sample_values = values[:20]

                dist = conn.execute(
                    f"SELECT COUNT(DISTINCT [{col_name}]) FROM [{table}]"
                ).fetchone()[0]
                profile.distinct_count = dist

                null_count = conn.execute(
                    f"SELECT COUNT(*) FROM [{table}] WHERE [{col_name}] IS NULL OR [{col_name}] = ''"
                ).fetchone()[0]
                profile.null_pct = (null_count / row_count * 100) if row_count > 0 else 0

                if values:
                    profile.min_val = min(values)
                    profile.max_val = max(values)

            except Exception:
                pass

            self._infer_semantics(profile)

            profiles.append(profile)
            self._column_lookup[col_name.upper()].append(profile)

        self.tables[table] = profiles

    def _infer_semantics(self, profile: ColumnProfile):
        cn = profile.name.lower()

        for pattern, tag in self._NAME_PATTERNS:
            if re.search(pattern, cn):
                profile.semantic_tags.append(tag)

        profile.is_id = bool(cn.endswith('_id') or cn == 'id' or
                             'identifier' in profile.semantic_tags or
                             'regulated_id' in profile.semantic_tags)

        if profile.sample_values:
            date_count = sum(1 for v in profile.sample_values[:10]
                             if re.match(r'^\d{4}-\d{2}-\d{2}', v))
            if date_count >= 7:
                profile.is_date = True
                if 'date' not in profile.semantic_tags:
                    profile.semantic_tags.append('date')
                if 'birth' in cn or 'dob' in cn:
                    profile.semantic_tags.append('birth_date')

            num_count = sum(1 for v in profile.sample_values[:10]
                            if re.match(r'^-?\d+\.?\d*$', v))
            if num_count >= 7:
                profile.is_numeric = True
                if 'currency' not in profile.semantic_tags and 'count' not in profile.semantic_tags:
                    profile.semantic_tags.append('numeric')

            table_rows = self.table_row_counts.get(profile.table, 1)
            if profile.distinct_count <= 50 and table_rows > 100:
                profile.is_categorical = True
                if 'categorical' not in profile.semantic_tags:
                    profile.semantic_tags.append('categorical')

            if not profile.is_numeric and not profile.is_date:
                avg_len = sum(len(v) for v in profile.sample_values[:10]) / max(len(profile.sample_values[:10]), 1)
                if avg_len > 20 or profile.distinct_count > table_rows * 0.5:
                    profile.is_text = True

    def _discover_joins(self):
        col_tables: Dict[str, List[str]] = defaultdict(list)
        for table, profiles in self.tables.items():
            for p in profiles:
                col_tables[p.name.upper()].append(table)

        for col, tables in col_tables.items():
            if len(tables) >= 2 and not col.startswith('STATUS'):
                for i, t1 in enumerate(tables):
                    for t2 in tables[i + 1:]:
                        self.join_graph[t1][t2] = col
                        self.join_graph[t2][t1] = col

        npi_like = {}
        for table, profiles in self.tables.items():
            for p in profiles:
                cn = p.name.upper()
                if 'NPI' in cn:
                    npi_like.setdefault(table, []).append(cn)

        for t1, cols1 in npi_like.items():
            for t2, cols2 in npi_like.items():
                if t1 >= t2:
                    continue
                for c1 in cols1:
                    for c2 in cols2:
                        if c1 != c2 and (c1 == 'NPI' or c2 == 'NPI'):
                            base = c1 if c1 == 'NPI' else c2
                            foreign = c2 if c1 == 'NPI' else c1
                            base_t = t1 if c1 == 'NPI' else t2
                            foreign_t = t2 if c1 == 'NPI' else t1
                            self.join_graph[base_t][foreign_t] = f"{base}={foreign}"
                            self.join_graph[foreign_t][base_t] = f"{foreign}={base}"

    def get_column(self, col_name: str) -> Optional[ColumnProfile]:
        profiles = self._column_lookup.get(col_name.upper(), [])
        return profiles[0] if profiles else None

    def get_date_columns(self, table: str) -> List[ColumnProfile]:
        return [p for p in self.tables.get(table, []) if p.is_date]

    def get_categorical_columns(self, table: str) -> List[ColumnProfile]:
        return [p for p in self.tables.get(table, []) if p.is_categorical]

    def get_numeric_columns(self, table: str) -> List[ColumnProfile]:
        return [p for p in self.tables.get(table, []) if p.is_numeric]

    def get_id_columns(self, table: str) -> List[ColumnProfile]:
        return [p for p in self.tables.get(table, []) if p.is_id]

    def find_birth_date_column(self, table: str = None) -> Optional[ColumnProfile]:
        search_tables = [table] if table else list(self.tables.keys())
        for t in search_tables:
            for p in self.tables.get(t, []):
                if 'birth_date' in p.semantic_tags or 'birth' in p.name.lower() or 'dob' in p.name.lower():
                    return p
        return None

    def find_date_pair(self, table: str) -> Optional[Tuple[ColumnProfile, ColumnProfile]]:
        dates = self.get_date_columns(table)
        START_HINTS = ['submit', 'start', 'admit', 'enroll', 'create', 'open', 'begin', 'prescri', 'refer']
        END_HINTS = ['adjudicat', 'end', 'discharge', 'disenroll', 'close', 'complet', 'resolv', 'fill']
        for d1 in dates:
            for d2 in dates:
                if d1.name == d2.name:
                    continue
                cn1, cn2 = d1.name.lower(), d2.name.lower()
                is_start = any(h in cn1 for h in START_HINTS)
                is_end = any(h in cn2 for h in END_HINTS)
                if is_start and is_end:
                    return (d1, d2)
        return None

    def get_value_column(self, table: str, hints: List[str] = None) -> Optional[ColumnProfile]:
        numerics = self.get_numeric_columns(table)
        if hints:
            for n in numerics:
                cn = n.name.lower()
                if any(h in cn for h in hints):
                    return n
        for n in numerics:
            if 'currency' in n.semantic_tags:
                return n
        return numerics[0] if numerics else None


class SemanticSchemaIndex:

    def __init__(self, learner: SchemaLearner):
        self.learner = learner
        self.column_index = TFIDFIndex()
        self.table_index = TFIDFIndex()
        self.value_index = TFIDFIndex()
        self._build_indexes()

    def _build_indexes(self):
        col_docs = []
        for table, profiles in self.learner.tables.items():
            for p in profiles:
                desc_parts = [
                    p.name.lower().replace('_', ' '),
                    p.name.lower(),
                    table,
                ]

                desc_parts.extend(p.semantic_tags)

                cn = p.name.lower()
                if p.is_date:
                    desc_parts.extend(['date', 'when', 'time', 'period'])
                    if 'birth' in cn:
                        desc_parts.extend(['age', 'born', 'birthday', 'years old', 'birth date'])
                    elif 'enroll' in cn:
                        desc_parts.extend(['enrolled', 'joined', 'signed up', 'registration'])
                    elif 'discharge' in cn:
                        desc_parts.extend(['discharged', 'left', 'released'])
                    elif 'service' in cn:
                        desc_parts.extend(['service', 'visit', 'seen'])
                    elif 'submit' in cn:
                        desc_parts.extend(['submitted', 'filed', 'sent'])
                    elif 'adjudicat' in cn:
                        desc_parts.extend(['adjudicated', 'processed', 'decided', 'resolved'])
                if p.is_numeric:
                    desc_parts.extend(['amount', 'number', 'value', 'count', 'total'])
                    if 'currency' in p.semantic_tags:
                        desc_parts.extend(['cost', 'price', 'money', 'spend', 'payment',
                                          'dollars', 'revenue', 'expense', 'charge'])
                if p.is_categorical:
                    desc_parts.extend(['type', 'category', 'kind', 'group', 'class'])
                    for v in p.sample_values[:15]:
                        desc_parts.append(v.lower().replace('_', ' '))
                if 'person_name' in p.semantic_tags:
                    desc_parts.extend(['name', 'who', 'person', 'individual'])
                if 'location' in p.semantic_tags:
                    desc_parts.extend(['where', 'location', 'place', 'area', 'site'])
                if 'demographic' in p.semantic_tags:
                    desc_parts.extend(['demographic', 'population', 'characteristic'])
                if p.is_id:
                    desc_parts.extend(['identifier', 'key', 'reference'])

                doc_id = f"{table}.{p.name}"
                text = ' '.join(desc_parts)
                meta = {
                    'table': table, 'column': p.name,
                    'is_numeric': p.is_numeric, 'is_date': p.is_date,
                    'is_categorical': p.is_categorical, 'is_id': p.is_id,
                    'semantic_tags': p.semantic_tags,
                    'distinct_count': p.distinct_count,
                }
                col_docs.append((doc_id, text, meta))

        if col_docs:
            self.column_index.build(col_docs)

        table_docs = []
        for table, profiles in self.learner.tables.items():
            parts = [table, table.rstrip('s')]
            for p in profiles:
                parts.append(p.name.lower().replace('_', ' '))
                parts.extend(p.semantic_tags)
            row_count = self.learner.table_row_counts.get(table, 0)
            meta = {'table': table, 'row_count': row_count, 'col_count': len(profiles)}
            table_docs.append((table, ' '.join(parts), meta))

        if table_docs:
            self.table_index.build(table_docs)

        val_docs = []
        for table, profiles in self.learner.tables.items():
            for p in profiles:
                if p.is_categorical and p.sample_values:
                    for v in set(p.sample_values[:30]):
                        doc_id = f"{table}.{p.name}={v}"
                        text = f"{v.lower().replace('_', ' ')} {p.name.lower().replace('_', ' ')}"
                        meta = {'table': table, 'column': p.name, 'value': v}
                        val_docs.append((doc_id, text, meta))

        if val_docs:
            self.value_index.build(val_docs)

    def match_columns(self, question: str, top_k: int = 15) -> List[Dict]:
        results = self.column_index.search(question, top_k=top_k, min_score=TFIDF_MIN_SEARCH_SCORE)
        return [{'column': r[2]['column'], 'table': r[2]['table'],
                 'score': r[1], 'meta': r[2]} for r in results]

    def match_tables(self, question: str, top_k: int = 5) -> List[Dict]:
        results = self.table_index.search(question, top_k=top_k, min_score=TFIDF_MIN_SEARCH_SCORE)
        return [{'table': r[2]['table'], 'score': r[1],
                 'row_count': r[2].get('row_count', 0)} for r in results]

    def match_values(self, question: str, top_k: int = 10) -> List[Dict]:
        results = self.value_index.search(question, top_k=top_k, min_score=0.1)
        return [{'table': r[2]['table'], 'column': r[2]['column'],
                 'value': r[2]['value'], 'score': r[1]} for r in results]


class IntentClassifier:

    TRAINING_DATA = {
        'count': [
            'how many claims are there',
            'count of members',
            'total number of encounters',
            'number of providers',
            'how many patients',
            'volume of prescriptions',
        ],
        'aggregate': [
            'average paid amount',
            'total cost of claims',
            'sum of billed amounts',
            'mean length of stay',
            'maximum paid amount',
            'minimum cost per visit',
        ],
        'ranking': [
            'top 10 providers by volume',
            'which specialty has the most claims',
            'highest cost procedures',
            'most common diagnosis',
            'busiest facilities',
            'top medications prescribed',
        ],
        'breakdown': [
            'claims by type',
            'distribution of members by gender',
            'breakdown of encounters by visit type',
            'split by region',
            'claims per category',
            'members grouped by plan type',
        ],
        'lookup': [
            'show me the claims for this member',
            'list all providers in this region',
            'what is CPT code 99213',
            'find appointments for tomorrow',
            'get details for this encounter',
            'description of this procedure',
        ],
        'trend': [
            'claims trend over time',
            'monthly admission count',
            'quarterly revenue growth',
            'enrollment over the past year',
            'cost per month',
            'seasonal pattern in visits',
        ],
        'comparison': [
            'compare inpatient vs outpatient costs',
            'emergency versus telehealth ratio',
            'paid vs billed amount difference',
            'male compared to female',
            'professional versus institutional claims',
        ],
        'percentage': [
            'what percentage of claims are denied',
            'proportion of members that are female',
            'share of emergency visits',
            'claims percent by demographics',
            'what pct of encounters are inpatient',
        ],
        'rate': [
            'denial rate by region',
            'readmission rate',
            'fill rate for prescriptions',
            'no show rate',
            'processing time',
            'approval rate',
        ],
        'filter': [
            'members older than 65',
            'claims greater than 5000',
            'encounters with length of stay over 7',
            'patients with risk score above 3',
            'providers in California',
        ],
    }

    def __init__(self):
        self._index = TFIDFIndex()
        self._centroids: Dict[str, SparseVector] = {}
        self._train()

    def _train(self):
        docs = []
        for intent, questions in self.TRAINING_DATA.items():
            for i, q in enumerate(questions):
                docs.append((f"{intent}_{i}", q, {'intent': intent}))
        self._index.build(docs)

        for intent, questions in self.TRAINING_DATA.items():
            vectors = [self._index._vectorize(q) for q in questions]
            if not vectors:
                continue
            all_indices = set()
            for v in vectors:
                all_indices.update(v.indices)
            all_indices = sorted(all_indices)

            avg_indices, avg_values = [], []
            for idx in all_indices:
                total = 0.0
                for v in vectors:
                    for vi, vv in zip(v.indices, v.values):
                        if vi == idx:
                            total += vv
                            break
                avg_indices.append(idx)
                avg_values.append(total / len(vectors))

            self._centroids[intent] = SparseVector(avg_indices, avg_values)

    def classify(self, question: str) -> Dict[str, Any]:
        qvec = self._index._vectorize(question)

        scores = {}
        for intent, centroid in self._centroids.items():
            scores[intent] = qvec.cosine(centroid)

        q = question.lower()
        BOOSTS = {
            'count': ['how many', 'count', 'number of', 'total number', 'volume'],
            'aggregate': ['average', 'avg', 'sum', 'total', 'mean', 'maximum', 'minimum'],
            'ranking': ['top', 'highest', 'lowest', 'most', 'least', 'busiest', 'best', 'worst'],
            'breakdown': ['by', 'per', 'breakdown', 'distribution', 'split', 'grouped'],
            'lookup': ['show', 'list', 'find', 'get', 'what is', 'describe', 'detail'],
            'trend': ['trend', 'over time', 'monthly', 'quarterly', 'growth', 'time series'],
            'comparison': ['vs', 'versus', 'compare', 'comparison', 'differ'],
            'percentage': ['%', 'percent', 'proportion', 'share', 'pct'],
            'rate': ['rate', 'ratio'],
            'filter': ['greater than', 'less than', 'above', 'below', 'older', 'younger', 'more than'],
        }

        for intent, keywords in BOOSTS.items():
            for kw in keywords:
                if kw in q:
                    scores[intent] = scores.get(intent, 0) + 0.15
                    break

        max_score = max(scores.values()) if scores else 0
        if max_score > 0:
            scores = {k: v / max_score for k, v in scores.items()}

        best = max(scores, key=scores.get) if scores else 'lookup'
        return {
            'intent': best,
            'confidence': scores.get(best, 0),
            'all_scores': dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)),
        }


class ComputedColumnInferrer:

    def __init__(self, learner: SchemaLearner):
        self.learner = learner
        self.computed: Dict[str, Dict] = {}
        self._infer()

    def _infer(self):
        for table, profiles in self.learner.tables.items():
            birth_col = self.learner.find_birth_date_column(table)
            if birth_col:
                alias_prefix = table[0] if len(self.learner.tables) > 1 else ''
                col_ref = f"{alias_prefix}.{birth_col.name}" if alias_prefix else birth_col.name

                self.computed[f'{table}:age'] = {
                    'expr': f"CAST((julianday('now') - julianday({col_ref})) / 365.25 AS INTEGER)",
                    'alias': 'age',
                    'table': table,
                    'source_column': birth_col.name,
                    'description': f'Age computed from {birth_col.name}',
                    'triggers': ['age', 'years old', 'how old'],
                }
                self.computed[f'{table}:age_group'] = {
                    'expr': (f"CASE "
                             f"WHEN (julianday('now') - julianday({col_ref})) / 365.25 < 18 THEN '0-17' "
                             f"WHEN (julianday('now') - julianday({col_ref})) / 365.25 < 35 THEN '18-34' "
                             f"WHEN (julianday('now') - julianday({col_ref})) / 365.25 < 50 THEN '35-49' "
                             f"WHEN (julianday('now') - julianday({col_ref})) / 365.25 < 65 THEN '50-64' "
                             f"ELSE '65+' END"),
                    'alias': 'age_group',
                    'table': table,
                    'source_column': birth_col.name,
                    'description': 'Age group buckets',
                    'triggers': ['age group', 'age bracket', 'age range', 'age distribution'],
                }
            date_pair = self.learner.find_date_pair(table)
            if date_pair:
                start_col, end_col = date_pair
                alias_prefix = table[0] if len(self.learner.tables) > 1 else ''
                s_ref = f"{alias_prefix}.{start_col.name}" if alias_prefix else start_col.name
                e_ref = f"{alias_prefix}.{end_col.name}" if alias_prefix else end_col.name

                duration_name = 'duration_days'
                if 'process' in start_col.name.lower() or 'submit' in start_col.name.lower():
                    duration_name = 'processing_days'
                elif 'admit' in start_col.name.lower():
                    duration_name = 'length_of_stay'
                elif 'enroll' in start_col.name.lower():
                    duration_name = 'tenure_days'

                self.computed[f'{table}:{duration_name}'] = {
                    'expr': f"ROUND(julianday({e_ref}) - julianday({s_ref}), 1)",
                    'alias': duration_name,
                    'table': table,
                    'source_column': f"{start_col.name} → {end_col.name}",
                    'description': f'Days from {start_col.name} to {end_col.name}',
                    'triggers': [duration_name.replace('_', ' '), 'duration', 'time between',
                                 'processing time', 'turnaround'],
                }

    def find_computed(self, question: str) -> List[Dict]:
        q = question.lower()
        results = []
        for key, info in self.computed.items():
            if any(trigger in q for trigger in info.get('triggers', [])):
                results.append(info)
        return results

    def get_age_expr(self, table: str = None) -> Optional[Dict]:
        for key, info in self.computed.items():
            if info['alias'] == 'age' and (table is None or info['table'] == table):
                return info
        return None

    def get_age_group_expr(self, table: str = None) -> Optional[Dict]:
        for key, info in self.computed.items():
            if info['alias'] == 'age_group' and (table is None or info['table'] == table):
                return info
        return None


class DataValidator:

    def __init__(self, db_path: str, learner: SchemaLearner):
        self.db_path = db_path
        self.learner = learner
        self._date_ranges: Dict[str, Tuple[str, str]] = {}
        self._value_sets: Dict[str, Set[str]] = {}
        self._profile()

    def _profile(self):
        conn = sqlite3.connect(self.db_path)
        try:
            for table, profiles in self.learner.tables.items():
                for p in profiles:
                    if p.is_date and p.sample_values:
                        try:
                            result = conn.execute(
                                f"SELECT MIN([{p.name}]), MAX([{p.name}]) FROM [{table}] "
                                f"WHERE [{p.name}] IS NOT NULL AND [{p.name}] != ''"
                            ).fetchone()
                            if result and result[0]:
                                self._date_ranges[f"{table}.{p.name}"] = (result[0], result[1])
                        except Exception:
                            pass

                    if p.is_categorical and p.distinct_count <= 50:
                        try:
                            values = conn.execute(
                                f"SELECT DISTINCT [{p.name}] FROM [{table}] "
                                f"WHERE [{p.name}] IS NOT NULL AND [{p.name}] != '' LIMIT 100"
                            ).fetchall()
                            self._value_sets[f"{table}.{p.name}"] = {str(v[0]) for v in values}
                        except Exception:
                            pass
        finally:
            conn.close()

    def validate_sql(self, sql: str, tables: List[str]) -> Dict[str, Any]:
        warnings = []
        suggestions = []

        year_m = re.search(r"LIKE '(\d{4})%'", sql)
        if year_m:
            year = year_m.group(1)
            for table in tables:
                for key, (min_d, max_d) in self._date_ranges.items():
                    if key.startswith(table + '.'):
                        min_year = min_d[:4] if min_d else '0000'
                        max_year = max_d[:4] if max_d else '9999'
                        if year < min_year or year > max_year:
                            warnings.append(
                                f"Year {year} is outside the data range ({min_year}-{max_year}) "
                                f"for {key}."
                            )
                            suggestions.append(
                                f"Data is available from {min_year} to {max_year}."
                            )

        eq_matches = re.findall(r"(\w+)\s*=\s*'([^']+)'", sql)
        for col, val in eq_matches:
            for key, values in self._value_sets.items():
                if col.upper() in key.upper():
                    if val not in values and val.upper() not in {v.upper() for v in values}:
                        warnings.append(
                            f"Value '{val}' not found in {key}. "
                            f"Known values: {', '.join(sorted(list(values))[:10])}"
                        )

        return {
            'valid': len(warnings) == 0,
            'warnings': warnings,
            'suggestions': suggestions,
        }

    def check_result_empty(self, sql: str) -> Tuple[bool, str]:
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(sql)
            row = cursor.fetchone()
            if row is None:
                return False, "No data found matching your query criteria."
            if len(row) == 1 and row[0] == 0:
                return False, "The count returned zero — no matching records exist."
            return True, "Data found."
        except Exception as e:
            return False, f"Query error: {str(e)}"
        finally:
            conn.close()


class AdaptiveQueryBuilder:

    def __init__(self, learner: SchemaLearner, inferrer: ComputedColumnInferrer):
        self.learner = learner
        self.inferrer = inferrer

    def build_demographic_breakdown(self, fact_table: str, dim_table: str,
                                    join_col: str, question: str) -> Optional[str]:
        dim_profiles = self.learner.tables.get(dim_table, [])
        demo_cols = []
        for p in dim_profiles:
            if p.is_categorical and 'demographic' in p.semantic_tags:
                demo_cols.append(p)
            elif p.name.upper() in ('GENDER', 'SEX', 'RACE', 'ETHNICITY', 'LANGUAGE'):
                demo_cols.append(p)

        age_group = self.inferrer.get_age_group_expr(dim_table)

        if not demo_cols and not age_group:
            return None

        parts = []
        total_subq = f"SELECT COUNT(*) FROM {fact_table}"

        for col in demo_cols:
            parts.append(
                f"SELECT '{col.name}' as dimension, {dim_table[0]}.{col.name} as value, "
                f"COUNT(*) as count, "
                f"ROUND(100.0 * COUNT(*) / ({total_subq}), 2) as pct "
                f"FROM {fact_table} f JOIN {dim_table} {dim_table[0]} "
                f"ON f.{join_col} = {dim_table[0]}.{join_col} "
                f"GROUP BY {dim_table[0]}.{col.name}"
            )

        if age_group:
            parts.append(
                f"SELECT 'AGE_GROUP' as dimension, {age_group['expr']} as value, "
                f"COUNT(*) as count, "
                f"ROUND(100.0 * COUNT(*) / ({total_subq}), 2) as pct "
                f"FROM {fact_table} f JOIN {dim_table} {dim_table[0]} "
                f"ON f.{join_col} = {dim_table[0]}.{join_col} "
                f"GROUP BY value"
            )

        if parts:
            return ' UNION ALL '.join(parts) + ' ORDER BY dimension, count DESC;'
        return None

    def build_rate_query(self, table: str, status_col: str, target_value: str,
                         group_col: str = None) -> str:
        group_select = f"{group_col}, " if group_col else ""
        group_by = f" GROUP BY {group_col}" if group_col else ""
        order = f" ORDER BY rate DESC" if group_col else ""

        return (
            f"SELECT {group_select}"
            f"COUNT(*) as total, "
            f"SUM(CASE WHEN {status_col} = '{target_value}' THEN 1 ELSE 0 END) as {target_value.lower()}_count, "
            f"ROUND(100.0 * SUM(CASE WHEN {status_col} = '{target_value}' THEN 1 ELSE 0 END) "
            f"/ COUNT(*), 2) as rate "
            f"FROM {table}{group_by}{order} LIMIT 50;"
        )

    def build_age_group_query(self, table: str) -> Optional[str]:
        age_group = self.inferrer.get_age_group_expr(table)
        if not age_group:
            return None
        row_count = self.learner.table_row_counts.get(table, 0)
        return (
            f"SELECT {age_group['expr']} as age_group, "
            f"COUNT(*) as count, "
            f"ROUND(100.0 * COUNT(*) / {row_count}, 2) as pct "
            f"FROM {table} "
            f"GROUP BY age_group ORDER BY age_group;"
        )


class SemanticLayer:

    def __init__(self, db_path: str, nlp_factory=None):
        self.db_path = db_path
        self.nlp_factory = nlp_factory
        self.learner: Optional[SchemaLearner] = None
        self.index: Optional[SemanticSchemaIndex] = None
        self.classifier: Optional[IntentClassifier] = None
        self.inferrer: Optional[ComputedColumnInferrer] = None
        self.validator: Optional[DataValidator] = None
        self.builder: Optional[AdaptiveQueryBuilder] = None
        self.entity_extractor = None
        self._initialized = False

    def initialize(self) -> Dict[str, Any]:
        import time as _time
        _t0 = _time.time()

        self.learner = SchemaLearner(self.db_path)

        cache_path = os.path.join(os.path.dirname(self.db_path), '.schema_cache.json')
        try:
            from schema_persistence import load_schema_cache, save_schema_cache
            cached = load_schema_cache(cache_path, self.db_path, ColumnProfile)
            if cached:
                self.learner.tables = cached['tables']
                self.learner.table_row_counts = cached['table_row_counts']
                self.learner.join_graph = defaultdict(dict, {
                    k: dict(v) for k, v in cached['join_graph'].items()
                })
                self.learner._column_lookup = defaultdict(list)
                for tbl, profiles in self.learner.tables.items():
                    for p in profiles:
                        self.learner._column_lookup[p.name.lower()].append(p)
                if not self.learner.join_graph:
                    self.learner._discover_joins()
                schema_info = {
                    'tables': len(self.learner.tables),
                    'total_columns': sum(len(cols) for cols in self.learner.tables.values()),
                    'total_rows': sum(self.learner.table_row_counts.values()),
                    'join_paths': sum(len(v) for v in self.learner.join_graph.values()),
                    'source': 'cache',
                }
                logger.info("Schema loaded from cache in %.1fs: %s", _time.time() - _t0, schema_info)
            else:
                schema_info = self.learner.learn()
                save_schema_cache(self.learner, cache_path, self.db_path)
                schema_info['source'] = 'fresh'
                logger.info("Schema learned fresh in %.1fs: %s", _time.time() - _t0, schema_info)
        except Exception as e:
            logger.warning("Schema persistence failed (%s), learning fresh", e)
            schema_info = self.learner.learn()
            logger.info("Schema learned (fallback): %s", schema_info)

        self.index = SemanticSchemaIndex(self.learner)
        logger.info("Semantic indexes built")

        self.classifier = IntentClassifier()
        if self.nlp_factory:
            self._enhance_classifier()
        logger.info("Intent classifier trained (factory=%s)",
                     'yes' if self.nlp_factory else 'no')

        self.inferrer = ComputedColumnInferrer(self.learner)
        logger.info("Computed columns inferred: %d", len(self.inferrer.computed))

        self.validator = DataValidator(self.db_path, self.learner)
        logger.info("Data validator profiled")

        self.builder = AdaptiveQueryBuilder(self.learner, self.inferrer)

        if self.nlp_factory:
            try:
                self.entity_extractor = self.nlp_factory.get_entity_extractor()
                logger.info("Entity extractor: %s", self.entity_extractor.backend)
            except Exception as e:
                logger.warning("Entity extractor setup failed: %s", e)

        self._initialized = True
        nlp_info = {}
        if self.nlp_factory:
            nlp_info = self.nlp_factory.report()
        return {
            'schema': schema_info,
            'computed_columns': list(self.inferrer.computed.keys()),
            'nlp_backends': nlp_info.get('active_backends', {}),
            'status': 'initialized',
        }

    def _enhance_classifier(self):
        try:
            lib_clf = self.nlp_factory.get_classifier()
            if lib_clf.backend == 'scratch':
                return

            texts, labels = [], []
            for intent, questions in IntentClassifier.TRAINING_DATA.items():
                for q in questions:
                    texts.append(q)
                    labels.append(intent)
            lib_clf.fit(texts, labels)

            original_classify = self.classifier.classify

            def hybrid_classify(question: str) -> Dict[str, Any]:
                scratch_result = original_classify(question)
                try:
                    lib_label, lib_conf = lib_clf.predict(question)
                    if lib_conf > scratch_result['confidence']:
                        return {
                            'intent': lib_label,
                            'confidence': lib_conf,
                            'all_scores': scratch_result.get('all_scores', {}),
                            'backend': lib_clf.backend,
                        }
                except Exception:
                    pass
                scratch_result['backend'] = 'scratch'
                return scratch_result

            self.classifier.classify = hybrid_classify
            logger.info("Intent classifier enhanced with %s backend", lib_clf.backend)
        except Exception as e:
            logger.warning("Could not enhance classifier: %s", e)

    def extract_entities(self, text: str) -> List[Dict]:
        if self.entity_extractor:
            try:
                return self.entity_extractor.extract(text)
            except Exception:
                pass
        return []

    def classify_intent(self, question: str) -> Dict[str, Any]:
        if not self._initialized:
            return {'intent': 'lookup', 'confidence': 0.0}
        return self.classifier.classify(question)

    def match_columns(self, question: str, top_k: int = 15) -> List[Dict]:
        if not self._initialized:
            return []
        return self.index.match_columns(question, top_k)

    def match_tables(self, question: str, top_k: int = 5) -> List[Dict]:
        if not self._initialized:
            return []
        return self.index.match_tables(question, top_k)

    def match_values(self, question: str, top_k: int = 10) -> List[Dict]:
        if not self._initialized:
            return []
        return self.index.match_values(question, top_k)

    def find_computed_columns(self, question: str) -> List[Dict]:
        if not self._initialized:
            return []
        return self.inferrer.find_computed(question)

    def validate(self, sql: str, tables: List[str]) -> Dict[str, Any]:
        if not self._initialized:
            return {'valid': True, 'warnings': [], 'suggestions': []}
        return self.validator.validate_sql(sql, tables)

    def check_empty(self, sql: str) -> Tuple[bool, str]:
        if not self._initialized:
            return True, "Validation not available."
        return self.validator.check_result_empty(sql)

    def get_schema_summary(self) -> Dict[str, Any]:
        if not self._initialized:
            return {}
        summary = {}
        for table, profiles in self.learner.tables.items():
            summary[table] = {
                'rows': self.learner.table_row_counts.get(table, 0),
                'columns': len(profiles),
                'date_cols': [p.name for p in profiles if p.is_date],
                'numeric_cols': [p.name for p in profiles if p.is_numeric],
                'categorical_cols': [p.name for p in profiles if p.is_categorical],
                'id_cols': [p.name for p in profiles if p.is_id],
            }
        return summary


