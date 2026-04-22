import os
import re
import json
import math
import time
import sqlite3
import logging
import hashlib
import pickle
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter

import numpy as np

logger = logging.getLogger('gpdm.neural_engine')


class HealthcareEmbeddings:

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.embeddings: Optional[np.ndarray] = None
        self._char_ngram_index: Dict[str, Set[str]] = defaultdict(set)
        self._trained = False

    def train(self, documents: List[str], schema_terms: List[str]):
        t0 = time.time()

        all_words = set()
        word_counts = Counter()
        doc_freq = Counter()

        for doc in documents:
            words = set(self._tokenize(doc))
            all_words |= words
            word_counts.update(self._tokenize(doc))
            doc_freq.update(words)

        for term in schema_terms:
            for word in self._tokenize(term.replace('_', ' ')):
                all_words.add(word)
                word_counts[word] += 5

        self.vocab = {w: i for i, w in enumerate(sorted(all_words))}

        n_docs = max(len(documents), 1)
        self.idf = {w: math.log(n_docs / (1 + doc_freq.get(w, 0))) + 1.0 for w in self.vocab}

        for word in self.vocab:
            for ng in self._char_ngrams(word, 3):
                self._char_ngram_index[ng].add(word)

        np.random.seed(42)
        n_vocab = len(self.vocab)
        self.embeddings = np.random.randn(n_vocab, self.dim).astype(np.float32)

        self._align_semantic_groups(schema_terms)

        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.embeddings = self.embeddings / norms

        self._trained = True
        elapsed = (time.time() - t0) * 1000
        logger.info("HealthcareEmbeddings trained: %d vocab, %d dims in %.0fms",
                     n_vocab, self.dim, elapsed)

    def _align_semantic_groups(self, schema_terms: List[str]):
        groups = {
            'member_words': ['member', 'patient', 'enrollee', 'beneficiary', 'person'],
            'claim_words': ['claim', 'billing', 'charge', 'service', 'adjudication'],
            'money_words': ['cost', 'paid', 'amount', 'spend', 'billed', 'payment', 'copay', 'deductible', 'coinsurance', 'expense', 'revenue', 'price'],
            'provider_words': ['provider', 'doctor', 'physician', 'specialist', 'npi', 'clinician'],
            'diagnosis_words': ['diagnosis', 'condition', 'disease', 'icd', 'chronic', 'comorbidity'],
            'medication_words': ['medication', 'drug', 'prescription', 'rx', 'pharmacy', 'formulary', 'ndc'],
            'time_words': ['month', 'year', 'quarter', 'trend', 'time', 'date', 'period', 'monthly', 'yearly', 'quarterly', 'annually'],
            'count_words': ['count', 'total', 'number', 'many', 'how'],
            'rank_words': ['top', 'most', 'highest', 'best', 'worst', 'least', 'bottom', 'which'],
            'rate_words': ['rate', 'percentage', 'percent', 'ratio', 'proportion'],
            'region_words': ['region', 'ncal', 'scal', 'nw', 'co', 'facility', 'location'],
        }

        for group_name, words in groups.items():
            indices = [self.vocab[w] for w in words if w in self.vocab]
            if len(indices) < 2:
                continue
            centroid = self.embeddings[indices].mean(axis=0)
            for idx in indices:
                self.embeddings[idx] = 0.6 * self.embeddings[idx] + 0.4 * centroid

    def encode(self, text: str) -> np.ndarray:
        words = self._tokenize(text)
        if not words:
            return np.zeros(self.dim, dtype=np.float32)

        vec = np.zeros(self.dim, dtype=np.float32)
        weight_sum = 0.0

        for word in words:
            idx = self.vocab.get(word)
            if idx is None:
                corrected = self._nearest_word(word)
                idx = self.vocab.get(corrected)

            if idx is not None:
                w = self.idf.get(word, 1.0)
                vec += w * self.embeddings[idx]
                weight_sum += w

        if weight_sum > 0:
            vec /= weight_sum

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm

        return vec

    def similarity(self, text1: str, text2: str) -> float:
        v1 = self.encode(text1)
        v2 = self.encode(text2)
        return float(np.dot(v1, v2))

    def _nearest_word(self, word: str) -> Optional[str]:
        ngrams = self._char_ngrams(word, 3)
        candidates = Counter()
        for ng in ngrams:
            for w in self._char_ngram_index.get(ng, set()):
                candidates[w] += 1

        if not candidates:
            return None

        best = candidates.most_common(1)[0]
        min_overlap = max(1, len(ngrams) // 3)
        if best[1] >= min_overlap:
            return best[0]
        return None

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'[a-z0-9]+', text.lower())

    def _char_ngrams(self, word: str, n: int = 3) -> List[str]:
        padded = f'#{word}#'
        return [padded[i:i+n] for i in range(len(padded) - n + 1)]


class NeuralIntentClassifier:

    INTENT_LABELS = ['count', 'aggregate', 'rank', 'trend', 'rate',
                      'compare', 'list', 'summary', 'correlate', 'exists']

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = len(self.INTENT_LABELS)
        self.label_to_idx = {l: i for i, l in enumerate(self.INTENT_LABELS)}

        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self._trained = False

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 300, lr: float = 0.01):
        t0 = time.time()
        n_samples = X.shape[0]

        self.W1 = np.random.randn(self.input_dim, self.hidden_dim).astype(np.float32) * np.sqrt(2.0 / self.input_dim)
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float32)
        self.W2 = np.random.randn(self.hidden_dim, self.n_classes).astype(np.float32) * np.sqrt(2.0 / self.hidden_dim)
        self.b2 = np.zeros(self.n_classes, dtype=np.float32)

        Y = np.zeros((n_samples, self.n_classes), dtype=np.float32)
        for i, label in enumerate(y):
            Y[i, label] = 1.0

        best_loss = float('inf')
        for epoch in range(epochs):
            z1 = X @ self.W1 + self.b1
            h1 = np.maximum(0, z1)
            z2 = h1 @ self.W2 + self.b2

            exp_z = np.exp(z2 - z2.max(axis=1, keepdims=True))
            probs = exp_z / exp_z.sum(axis=1, keepdims=True)

            loss = -np.sum(Y * np.log(probs + 1e-10)) / n_samples

            dz2 = (probs - Y) / n_samples
            dW2 = h1.T @ dz2
            db2 = dz2.sum(axis=0)

            dh1 = dz2 @ self.W2.T
            dz1 = dh1 * (z1 > 0).astype(np.float32)
            dW1 = X.T @ dz1
            db1 = dz1.sum(axis=0)

            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            self.W2 -= lr * dW2
            self.b2 -= lr * db2

            if loss < best_loss:
                best_loss = loss

        preds = self.predict_batch(X)
        acc = np.mean(preds == y) * 100

        elapsed = (time.time() - t0) * 1000
        self._trained = True
        logger.info("IntentClassifier trained: %d examples, %d epochs, acc=%.1f%%, loss=%.4f in %.0fms",
                     n_samples, epochs, acc, best_loss, elapsed)

    def predict(self, x: np.ndarray) -> Tuple[str, float]:
        if not self._trained:
            return 'count', 0.0

        z1 = x @ self.W1 + self.b1
        h1 = np.maximum(0, z1)
        z2 = h1 @ self.W2 + self.b2

        exp_z = np.exp(z2 - z2.max())
        probs = exp_z / exp_z.sum()

        idx = np.argmax(probs)
        return self.INTENT_LABELS[idx], float(probs[idx])

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        z1 = X @ self.W1 + self.b1
        h1 = np.maximum(0, z1)
        z2 = h1 @ self.W2 + self.b2
        return np.argmax(z2, axis=1)


class HealthcareEntityExtractor:

    AGG_PATTERNS = {
        'average': ('AVG', None), 'avg': ('AVG', None), 'mean': ('AVG', None),
        'total': ('SUM', None), 'sum': ('SUM', None),
        'maximum': ('MAX', None), 'max': ('MAX', None), 'highest': ('MAX', None),
        'minimum': ('MIN', None), 'min': ('MIN', None), 'lowest': ('MIN', None),
        'count': ('COUNT', None), 'number': ('COUNT', None), 'how many': ('COUNT', None),
    }

    TEMPORAL_PATTERNS = [
        (r'per\s+month|monthly|by\s+month|each\s+month|month\s+over\s+month', 'month'),
        (r'per\s+quarter|quarterly|by\s+quarter|each\s+quarter', 'quarter'),
        (r'per\s+year|yearly|annually|by\s+year|each\s+year', 'year'),
        (r'per\s+day|daily|by\s+day', 'day'),
        (r'over\s+time|trend|time\s+series', 'month'),
    ]

    LIMIT_PATTERNS = [
        (r'top\s+(\d+)', None),
        (r'first\s+(\d+)', None),
        (r'(\d+)\s+(?:most|least|highest|lowest)', None),
    ]

    def __init__(self, schema_graph):
        self.graph = schema_graph
        self._build_entity_index()

    def _build_entity_index(self):
        self.table_synonyms: Dict[str, str] = {}
        self.column_index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self.value_index: Dict[str, Tuple[str, str]] = {}

        for table_name, concept in self.graph.tables.items():
            self.table_synonyms[table_name] = table_name
            self.table_synonyms[concept.concept] = table_name
            for syn in concept.synonyms:
                self.table_synonyms[syn.lower()] = table_name

            for col_name in self.graph.columns.get(table_name, {}):
                words = col_name.lower().replace('_', ' ').split()
                for word in words:
                    if len(word) >= 3:
                        self.column_index[word].append((table_name, col_name))

    def extract(self, question: str) -> Dict[str, Any]:
        q_lower = question.lower()
        result = {
            'tables': [], 'columns': [], 'agg_function': '',
            'agg_column': '', 'agg_table': '', 'temporal': False,
            'time_granularity': '', 'limit': None, 'filters': [],
            'order': '',
        }

        result['tables'] = self._extract_tables(q_lower)

        words = re.findall(r'[a-z0-9]+', q_lower)
        table_names = [t for t, _ in result['tables']]
        col_matches = self.graph.find_columns_for_words(
            words, table_names or None, raw_question=q_lower
        )
        result['columns'] = [(t, c, conf) for t, c, _, conf in col_matches[:10]]

        for pattern, (func, col) in self.AGG_PATTERNS.items():
            if pattern in q_lower:
                result['agg_function'] = func
                break

        if result['agg_function'] and result['columns']:
            for t, c, conf in result['columns']:
                sem = self.graph.columns.get(t, {}).get(c)
                if sem and (sem.aggregatable or sem.is_money):
                    result['agg_column'] = c
                    result['agg_table'] = t
                    break

        for pattern, granularity in self.TEMPORAL_PATTERNS:
            if re.search(pattern, q_lower):
                result['temporal'] = True
                result['time_granularity'] = granularity
                break

        for pattern, _ in self.LIMIT_PATTERNS:
            match = re.search(pattern, q_lower)
            if match:
                result['limit'] = int(match.group(1))
                break

        if any(w in q_lower for w in ['most', 'highest', 'top', 'best', 'largest']):
            result['order'] = 'desc'
        elif any(w in q_lower for w in ['least', 'lowest', 'bottom', 'fewest', 'smallest']):
            result['order'] = 'asc'

        result['filters'] = self._extract_filters(q_lower, original_question=question)

        return result

    def _extract_tables(self, q_lower: str) -> List[Tuple[str, float]]:
        scores: Dict[str, float] = defaultdict(float)

        for synonym, table in self.table_synonyms.items():
            if len(synonym) >= 3 and synonym in q_lower:
                pattern = r'\b' + re.escape(synonym) + r'\b'
                if re.search(pattern, q_lower):
                    scores[table] += 1.0
                else:
                    scores[table] += 0.5

        return sorted(scores.items(), key=lambda x: -x[1])

    def _extract_filters(self, q_lower: str, original_question: str = '') -> List[Dict]:
        filters = []

        breakdown_words = ['break down', 'breakdown', 'by status', 'per status', 'each status',
                           'by claim status', 'status breakdown', 'group by']
        is_breakdown = any(bw in q_lower for bw in breakdown_words)

        status_map = {
            'denied': ('CLAIM_STATUS', 'DENIED', 'claims'),
            'denial': ('CLAIM_STATUS', 'DENIED', 'claims'),
            'approved': ('CLAIM_STATUS', 'APPROVED', 'claims'),
            'approval': ('CLAIM_STATUS', 'APPROVED', 'claims'),
            'pending': ('CLAIM_STATUS', 'PENDING', 'claims'),
            'no-show': ('STATUS', 'NO_SHOW', 'appointments'),
            'no show': ('STATUS', 'NO_SHOW', 'appointments'),
            'chronic': ('IS_CHRONIC', 'Y', 'diagnoses'),
        }
        for keyword, (col, val, table) in status_map.items():
            if keyword in q_lower:
                if is_breakdown and keyword in ('denied', 'approved', 'pending'):
                    continue
                if keyword in ('denial', 'approval') and 'rate' in q_lower:
                    filters.append({'column': col, 'operator': '=', 'value': val, 'table': table})
                    continue
                filters.append({'column': col, 'operator': '=', 'value': val, 'table': table})

        compare_words = ['compare', 'vs', 'versus', 'by visit type', 'per visit type',
                         'each visit type', 'by type', 'per type', 'how many encounters by']
        is_visit_compare = any(cw in q_lower for cw in compare_words)

        visit_types = {
            'emergency': ('VISIT_TYPE', 'EMERGENCY', 'encounters'),
            'inpatient': ('VISIT_TYPE', 'INPATIENT', 'encounters'),
            'outpatient': ('VISIT_TYPE', 'OUTPATIENT', 'encounters'),
            'telehealth': ('VISIT_TYPE', 'TELEHEALTH', 'encounters'),
        }
        for keyword, (col, val, table) in visit_types.items():
            if keyword in q_lower:
                visit_type_count = sum(1 for vt in visit_types if vt in q_lower)
                if visit_type_count >= 2 or is_visit_compare:
                    continue
                filters.append({'column': col, 'operator': '=', 'value': val, 'table': table})

        for plan in ['HMO', 'PPO', 'POS']:
            pattern = r'\b' + re.escape(plan.lower()) + r'\b'
            if re.search(pattern, q_lower):
                filters.append({'column': 'PLAN_TYPE', 'operator': '=', 'value': plan, 'table': 'claims'})

        regions = ['NCAL', 'SCAL', 'NW', 'CO', 'MAS', 'HI', 'GA', 'WA', 'MID']
        region_false_positives = {
            'co': {'count', 'compare', 'copay', 'cost', 'common', 'component', 'code', 'column', 'correlation', 'condition', 'could'},
            'hi': {'high', 'highest', 'history', 'his', 'histogram', 'which', 'within'},
            'ga': {'gap', 'game', 'gave', 'gain', 'gather'},
            'wa': {'was', 'want', 'ward', 'wait', 'walk', 'watch'},
            'nw': {'new', 'now'},
        }

        for region in regions:
            r_lower = region.lower()
            pattern = r'\b' + re.escape(r_lower) + r'\b'
            if re.search(pattern, q_lower):
                if len(region) <= 3:
                    has_uppercase = bool(original_question and re.search(r'\b' + re.escape(region) + r'\b', original_question))
                    region_context = re.search(r'\b(region|regions|kp_region|area|territory)\b', q_lower)
                    if len(region) <= 2:
                        if not has_uppercase and not region_context:
                            continue
                    else:
                        if not has_uppercase and not region_context:
                            fp_words = region_false_positives.get(r_lower, set())
                            q_words = set(q_lower.split())
                            if not q_words.intersection({r_lower}) or q_words.intersection(fp_words):
                                continue

                filters.append({'column': 'KP_REGION', 'operator': '=', 'value': region, 'table': 'claims'})

        year_match = re.search(r'\b(20\d{2})\b', q_lower)
        if year_match:
            year = year_match.group(1)
            filters.append({'column': 'SERVICE_DATE', 'operator': 'LIKE', 'value': f'{year}%', 'table': ''})

        risk_match = re.search(r'risk\s+score\s+(?:above|over|greater|>)\s+(\d+(?:\.\d+)?)', q_lower)
        if risk_match:
            filters.append({'column': 'RISK_SCORE', 'operator': '>', 'value': risk_match.group(1), 'table': 'members'})

        return filters


class SmartSQLGenerator:

    def __init__(self, schema_graph, db_path: str):
        self.graph = schema_graph
        self.db_path = db_path
        self._sql_constructor = None
        try:
            from sql_constructor import SQLConstructor
            self._sql_constructor = SQLConstructor(schema_graph, db_path)
        except Exception as e:
            logger.warning("SQLConstructor not available: %s", e)

    def generate(self, intent: str, entities: Dict, question: str) -> Optional[str]:
        if self._sql_constructor:
            try:
                parsed = self._to_parsed_intent(intent, entities, question)
                result = self._sql_constructor.construct(parsed)
                sql = result.get('sql', '')
                if sql and self._validate_sql(sql):
                    return sql
            except Exception as e:
                logger.warning("SQL Constructor failed: %s", e)

        return None

    def _to_parsed_intent(self, intent: str, entities: Dict, question: str):
        from intent_parser import ParsedIntent, ParsedFilter, normalize_typos

        normalized = normalize_typos(question.lower().strip())
        parsed = ParsedIntent(
            original_question=question,
            normalized_question=normalized,
            intent=intent,
        )

        parsed.tables = [t for t, _ in entities.get('tables', [])]
        parsed.agg_function = entities.get('agg_function', '')
        parsed.agg_column = entities.get('agg_column', '')
        parsed.agg_table = entities.get('agg_table', '')
        parsed.temporal = entities.get('temporal', False)
        parsed.time_granularity = entities.get('time_granularity', '')
        parsed.limit = entities.get('limit')
        parsed.order_by = entities.get('order', '')
        parsed.columns = [(t, c) for t, c, _ in entities.get('columns', [])]

        q_lower = question.lower()
        if intent == 'compare':
            vs_match = re.search(r'(\w+)\s+(?:vs|versus|compared?\s+to|or)\s+(\w+)', q_lower)
            if vs_match:
                parsed.compare_values = [vs_match.group(1).strip(), vs_match.group(2).strip()]
                parsed.comparison = True
            elif 'compare' in q_lower:
                and_match = re.search(r'compare\s+(\w+)\s+(?:and|&)\s+(\w+)', q_lower)
                if and_match:
                    parsed.compare_values = [and_match.group(1).strip(), and_match.group(2).strip()]
                    parsed.comparison = True

        if 'pmpm' in q_lower or 'per member per month' in q_lower:
            parsed.sub_intent = 'per_unit'
            if not parsed.agg_column:
                parsed.agg_column = 'PAID_AMOUNT'
                parsed.agg_table = 'claims'
            if 'claims' not in parsed.tables:
                parsed.tables.append('claims')

        if intent == 'aggregate' and not parsed.agg_column:
            if not parsed.agg_function:
                parsed.agg_function = 'COUNT'

        group_patterns = [
            (r'(?:by|per|for each|in each|across)\s+(?:plan\s*type)', 'PLAN_TYPE', 'claims'),
            (r'(?:by|per|for each|in each|across)\s+(?:region|kp.region)', 'KP_REGION', 'claims'),
            (r'(?:by|per|for each|in each|across)\s+(?:visit\s*type)', 'VISIT_TYPE', 'encounters'),
            (r'(?:by|per|for each|in each|across)\s+(?:specialty|specialties)', 'SPECIALTY', 'providers'),
            (r'(?:by|per|for each|in each|across)\s+(?:department)', 'DEPARTMENT', 'encounters'),
            (r'(?:by|per|for each|in each|across)\s+(?:facility|facilities)', 'FACILITY', 'encounters'),
            (r'(?:by|per|for each|in each|across)\s+(?:medication.class)', 'MEDICATION_CLASS', 'prescriptions'),
            (r'(?:by|per|for each|in each|across)\s+(?:status|claim.status)', 'CLAIM_STATUS', 'claims'),
            (r'(?:by|per|for each|in each|across)\s+(?:gender)', 'GENDER', 'members'),
        ]
        for pattern, col, table in group_patterns:
            if re.search(pattern, q_lower):
                parsed.group_by.append((table, col))

        if intent == 'rank' and not parsed.group_by:
            which_patterns = [
                (r'which\s+(?:specialty|specialties)', 'SPECIALTY', 'providers'),
                (r'which\s+(?:department)', 'DEPARTMENT', 'encounters'),
                (r'which\s+(?:region)', 'KP_REGION', 'claims'),
                (r'which\s+(?:facility|facilities)', 'FACILITY', 'encounters'),
                (r'which\s+(?:provider)', 'NPI', 'providers'),
                (r'which\s+(?:medication)', 'MEDICATION_NAME', 'prescriptions'),
            ]
            for pattern, col, table in which_patterns:
                if re.search(pattern, q_lower):
                    parsed.group_by.append((table, col))
                    break

        for f in entities.get('filters', []):
            parsed.filters.append(ParsedFilter(
                column=f['column'],
                operator=f['operator'],
                value=f['value'],
                table_hint=f.get('table', ''),
                confidence=0.9,
            ))

        if intent == 'rate':
            has_status = any(f.column in ('CLAIM_STATUS', 'STATUS', 'ENCOUNTER_STATUS') for f in parsed.filters)
            if not has_status:
                if 'denial' in q_lower or 'denied' in q_lower:
                    parsed.filters.append(ParsedFilter(
                        column='CLAIM_STATUS', operator='=', value='DENIED',
                        table_hint='claims', confidence=0.9))
                elif 'approval' in q_lower or 'approved' in q_lower:
                    parsed.filters.append(ParsedFilter(
                        column='CLAIM_STATUS', operator='=', value='APPROVED',
                        table_hint='claims', confidence=0.9))
                elif 'no show' in q_lower or 'no-show' in q_lower:
                    parsed.filters.append(ParsedFilter(
                        column='STATUS', operator='=', value='NO_SHOW',
                        table_hint='appointments', confidence=0.9))
                    if 'appointments' not in parsed.tables:
                        parsed.tables.append('appointments')

        if 'provider' in q_lower and 'claim' in q_lower:
            if 'providers' not in parsed.tables:
                parsed.tables.append('providers')
            if 'claims' not in parsed.tables:
                parsed.tables.append('claims')

        if 'sick' in q_lower:
            if not parsed.agg_column:
                parsed.agg_column = 'RISK_SCORE'
                parsed.agg_table = 'members'
            if 'members' not in parsed.tables:
                parsed.tables.append('members')
            parsed.order_by = 'desc'

        parsed.confidence = 0.85
        return parsed

    def _validate_sql(self, sql: str) -> bool:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(f"SELECT * FROM ({sql}) AS _test LIMIT 0")
            conn.close()
            return True
        except:
            return False


class InsightNarrator:

    BENCHMARKS = {
        'denial_rate': {'good': 5.0, 'avg': 10.0, 'bad': 15.0, 'unit': '%'},
        'readmission_rate': {'good': 10.0, 'avg': 15.0, 'bad': 20.0, 'unit': '%'},
        'avg_los': {'good': 3.0, 'avg': 5.0, 'bad': 7.0, 'unit': 'days'},
        'no_show_rate': {'good': 5.0, 'avg': 12.0, 'bad': 20.0, 'unit': '%'},
        'pmpm': {'good': 350.0, 'avg': 500.0, 'bad': 750.0, 'unit': '$'},
    }

    def __init__(self, db_path: str = None):
        self.db_path = db_path
        self._total_members = None
        self._total_claims = None

    def _get_context_numbers(self):
        if self._total_members is not None:
            return
        try:
            conn = sqlite3.connect(self.db_path)
            self._total_members = conn.execute("SELECT COUNT(*) FROM members").fetchone()[0]
            self._total_claims = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
            conn.close()
        except:
            self._total_members = 0
            self._total_claims = 0

    def narrate(self, question: str, intent: str, rows: List, columns: List[str],
                entities: Dict = None) -> str:
        self._get_context_numbers()
        q_lower = question.lower()

        if not rows:
            return "No data found for this query. This may indicate a data gap or the criteria are too restrictive."

        if len(rows) == 1 and len(rows[0]) == 1:
            val = rows[0][0]
            return self._narrate_single_value(q_lower, val, intent)

        if len(rows) == 1:
            return self._narrate_single_row(q_lower, rows[0], columns, intent)

        return self._narrate_multi_row(q_lower, rows, columns, intent)

    def _narrate_single_value(self, question: str, value: Any, intent: str) -> str:
        if isinstance(value, float):
            formatted = f"{value:,.2f}"
        elif isinstance(value, int):
            formatted = f"{value:,}"
        else:
            formatted = str(value)

        if 'member' in question and intent == 'count':
            return (f"There are {formatted} members in the health plan. "
                    f"This represents the total enrolled population across all regions and plan types.")
        elif 'claim' in question and intent == 'count':
            return (f"The system contains {formatted} claims. "
                    f"This covers all claim types including professional, institutional, and pharmacy claims.")
        elif 'provider' in question and intent == 'count':
            return (f"There are {formatted} providers in the network, spanning multiple specialties and regions.")
        elif any(w in question for w in ['cost', 'paid', 'billed', 'amount', 'spend']):
            return (f"The total is ${formatted}. "
                    f"{'This is the aggregate across all claims in the system.' if float(value) > 10000 else ''}")
        elif any(w in question for w in ['average', 'avg', 'mean']):
            return f"The average is {formatted}."
        elif any(w in question for w in ['risk score']):
            rs = float(value) if isinstance(value, (int, float)) else 0
            risk_level = "low" if rs < 1.0 else "moderate" if rs < 2.0 else "high"
            return f"The average risk score is {formatted}, indicating a {risk_level}-risk population overall."

        return f"The result is {formatted}."

    def _narrate_single_row(self, question: str, row: tuple, columns: List[str], intent: str) -> str:
        parts = []
        for i, col in enumerate(columns):
            val = row[i]
            if val is None:
                continue
            col_display = col.lower().replace('_', ' ')
            if isinstance(val, float):
                if 'rate' in col_display or 'percent' in col_display:
                    parts.append(f"{col_display}: {val:.1f}%")
                elif any(w in col_display for w in ['amount', 'cost', 'paid', 'billed']):
                    parts.append(f"{col_display}: ${val:,.2f}")
                else:
                    parts.append(f"{col_display}: {val:,.2f}")
            elif isinstance(val, int):
                parts.append(f"{col_display}: {val:,}")
            else:
                parts.append(f"{col_display}: {val}")

        narrative = "; ".join(parts) + "."

        if intent == 'rate' and 'denial' in question:
            rate_val = None
            for i, col in enumerate(columns):
                if 'rate' in col.lower():
                    rate_val = row[i]
            if rate_val is not None:
                bench = self.BENCHMARKS.get('denial_rate', {})
                if rate_val < bench.get('good', 5):
                    narrative += f" This denial rate is below the {bench['good']}% benchmark — strong performance."
                elif rate_val > bench.get('bad', 15):
                    narrative += f" This exceeds the {bench['bad']}% threshold and warrants investigation into denial root causes."
                else:
                    narrative += " This is within the typical range for Medicare Advantage plans."

        return narrative

    def _narrate_multi_row(self, question: str, rows: List, columns: List[str], intent: str) -> str:
        n = len(rows)

        if intent == 'trend':
            first_period = rows[0][0] if rows[0] else '?'
            last_period = rows[-1][0] if rows[-1] else '?'
            if len(rows[0]) > 1 and len(rows[-1]) > 1:
                first_val = rows[0][-1]
                last_val = rows[-1][-1]
                if isinstance(first_val, (int, float)) and isinstance(last_val, (int, float)) and first_val > 0:
                    pct_change = ((last_val - first_val) / first_val) * 100
                    direction = "increased" if pct_change > 0 else "decreased"
                    return (f"Across {n} periods from {first_period} to {last_period}, "
                            f"the metric {direction} by {abs(pct_change):.1f}%. "
                            f"Starting at {first_val:,} and ending at {last_val:,}.")
            return f"Trend data spans {n} periods from {first_period} to {last_period}."

        if intent == 'rank':
            top = rows[0]
            top_name = top[0] if top else '?'
            top_val = top[-1] if len(top) > 1 else top[0]
            if isinstance(top_val, float):
                return f"The top result is {top_name} with {top_val:,.2f}. Found {n} results total."
            elif isinstance(top_val, int):
                return f"The top result is {top_name} with {top_val:,}. Found {n} results total."
            return f"The top result is {top_name}. Found {n} results total."

        if n <= 10:
            summary_parts = []
            for row in rows[:5]:
                name = row[0] if row else '?'
                val = row[-1] if len(row) > 1 else row[0]
                if isinstance(val, float):
                    summary_parts.append(f"{name}: {val:,.2f}")
                elif isinstance(val, int):
                    summary_parts.append(f"{name}: {val:,}")
                else:
                    summary_parts.append(f"{name}: {val}")
            return f"Breakdown across {n} categories: " + "; ".join(summary_parts) + "."

        return f"Query returned {n} results across {len(columns)} dimensions."


class TrainingDataGenerator:

    def __init__(self, schema_graph):
        self.graph = schema_graph

    def generate(self) -> List[Tuple[str, str]]:
        examples = []

        for table, concept in self.graph.tables.items():
            examples.extend([
                (f"how many {concept.concept} do we have", 'count'),
                (f"count of {table}", 'count'),
                (f"total number of {concept.concept}", 'count'),
                (f"how many {concept.concept} are there", 'count'),
            ])

        for table in ['claims', 'prescriptions', 'encounters']:
            money_cols = self.graph.get_money_columns(table)
            for col in money_cols:
                col_display = col.lower().replace('_', ' ')
                examples.extend([
                    (f"total {col_display}", 'aggregate'),
                    (f"average {col_display}", 'aggregate'),
                    (f"what is the sum of {col_display}", 'aggregate'),
                    (f"avg {col_display} per claim", 'aggregate'),
                ])

        rank_phrases = ['top', 'most', 'which', 'highest', 'best']
        for table, concept in self.graph.tables.items():
            for phrase in rank_phrases:
                examples.extend([
                    (f"{phrase} {concept.concept} by count", 'rank'),
                    (f"which {concept.concept} has the most", 'rank'),
                ])
        examples.extend([
            ("top 5 medications by cost", 'rank'),
            ("which specialty sees the most patients", 'rank'),
            ("top providers by claim count", 'rank'),
            ("most common diagnoses", 'rank'),
            ("which region has the highest denial rate", 'rank'),
            ("most prescribed medications", 'rank'),
        ])

        examples.extend([
            ("claims per month", 'trend'),
            ("monthly trend of encounters", 'trend'),
            ("how have claims changed over time", 'trend'),
            ("quarterly cost trend", 'trend'),
            ("claims by month in 2024", 'trend'),
            ("show me the trend of paid amounts", 'trend'),
            ("encounters per quarter", 'trend'),
            ("yearly claims volume", 'trend'),
            ("claims over time", 'trend'),
            ("monthly breakdown of prescriptions", 'trend'),
        ])

        examples.extend([
            ("what is the denial rate", 'rate'),
            ("percentage of claims denied", 'rate'),
            ("denial rate by region", 'rate'),
            ("what percent of claims are approved", 'rate'),
            ("no show rate for appointments", 'rate'),
            ("readmission rate", 'rate'),
            ("what is the approval rate", 'rate'),
        ])

        examples.extend([
            ("compare HMO vs PPO costs", 'compare'),
            ("telehealth vs inpatient visits", 'compare'),
            ("compare drugs vs hospital spending", 'compare'),
            ("NCAL vs SCAL claims", 'compare'),
        ])

        examples.extend([
            ("show me the members", 'list'),
            ("list all providers", 'list'),
            ("show denied claims", 'list'),
            ("who are our sickest patients", 'list'),
            ("list members with high risk", 'list'),
        ])

        examples.extend([
            ("give me a summary of claims", 'summary'),
            ("overview of the system", 'summary'),
            ("show me everything about NCAL", 'summary'),
            ("claims summary", 'summary'),
            ("executive summary", 'summary'),
        ])

        examples.extend([
            ("claims per month in 2024", 'trend'),
            ("how many claims per month", 'trend'),
            ("encounters per month", 'trend'),
            ("cost per quarter", 'trend'),
            ("paid amount per month", 'trend'),
            ("monthly claims count", 'trend'),
            ("monthly cost breakdown", 'trend'),
            ("trend of claims over time", 'trend'),
            ("show me claims by month", 'trend'),
            ("prescriptions per month", 'trend'),
            ("visits over time", 'trend'),
            ("monthly enrollment trend", 'trend'),
        ])

        examples.extend([
            ("compare telehealth vs inpatient visit counts", 'compare'),
            ("compare emergency vs outpatient", 'compare'),
            ("HMO versus PPO", 'compare'),
            ("compare NCAL and SCAL", 'compare'),
            ("telehealth vs inpatient", 'compare'),
        ])

        examples.extend([
            ("break down claims by plan type", 'rank'),
            ("claims by plan type", 'rank'),
            ("breakdown by region", 'rank'),
            ("breakdown by status", 'rank'),
            ("distribution by plan type", 'rank'),
            ("encounters by visit type", 'rank'),
            ("claims count per region", 'rank'),
            ("prescriptions by medication class", 'rank'),
            ("which department has the most encounters", 'rank'),
            ("copay by plan type", 'rank'),
            ("average copay for each plan type", 'rank'),
            ("what are the top 5 most common diagnoses", 'rank'),
            ("which facility handles the most volume", 'rank'),
        ])

        examples.extend([
            ("show me everything about NCAL", 'summary'),
            ("tell me about our claims", 'summary'),
            ("give me a claims overview", 'summary'),
        ])

        examples.extend([
            ("how much do we spend on prescriptions", 'aggregate'),
            ("what is the average age of our members", 'aggregate'),
            ("what is the average risk score of our members", 'aggregate'),
            ("average risk score", 'aggregate'),
            ("what is the avg risk score", 'aggregate'),
            ("prescription status breakdown", 'rank'),
            ("distribution of claim types", 'rank'),
            ("PMPM cost", 'aggregate'),
            ("what is our PMPM cost", 'aggregate'),
            ("average panel size", 'aggregate'),
            ("average length of stay", 'aggregate'),
            ("whats the avg paid amnt per claim", 'aggregate'),
            ("how many memebers in ncal region", 'count'),
            ("top specialites by clam count", 'rank'),
            ("what is the no show rate for appointments", 'rate'),
            ("no-show rate", 'rate'),
            ("what percentage are denied", 'rate'),
            ("denial rate by plan type", 'rate'),
            ("what is our denial rate", 'rate'),
        ])

        return examples


class HealthcareNeuralEngine:

    def __init__(self, db_path: str, schema_graph=None):
        self.db_path = db_path
        t0 = time.time()

        if schema_graph:
            self.graph = schema_graph
        else:
            from schema_graph import SemanticSchemaGraph
            self.graph = SemanticSchemaGraph(db_path)

        self.embeddings = HealthcareEmbeddings(dim=64)
        self.intent_classifier = NeuralIntentClassifier(input_dim=64)
        self.entity_extractor = HealthcareEntityExtractor(self.graph)
        self.sql_generator = SmartSQLGenerator(self.graph, db_path)
        self.narrator = InsightNarrator(db_path)

        self._train()

        elapsed = time.time() - t0
        logger.info("HealthcareNeuralEngine initialized in %.1fs", elapsed)

    def _apply_intent_overrides(self, intent: str, confidence: float,
                                entities: Dict, q_lower: str) -> str:
        temporal_signals = ['per month', 'per quarter', 'per year', 'over time',
                           'monthly', 'quarterly', 'yearly', 'by month', 'by quarter',
                           'trend', 'per week']
        if entities.get('temporal') and intent not in ('trend',):
            if any(ts in q_lower for ts in temporal_signals):
                logger.debug("Intent override: %s → trend (temporal signal)", intent)
                return 'trend'

        if 'vs' in q_lower or 'versus' in q_lower or 'compare' in q_lower:
            if intent not in ('compare', 'rate'):
                logger.debug("Intent override: %s → compare (vs/compare signal)", intent)
                return 'compare'

        if re.search(r'\btop\s+\d+\b', q_lower) or 'most common' in q_lower:
            if intent not in ('rank',):
                logger.debug("Intent override: %s → rank (top-N signal)", intent)
                return 'rank'

        if any(w in q_lower for w in ['denial rate', 'approval rate', 'readmission rate',
                                        'no-show rate', 'no show rate', 'percentage of',
                                        'percent of', 'what percent']):
            if intent not in ('rate',):
                logger.debug("Intent override: %s → rate (rate signal)", intent)
                return 'rate'

        if any(w in q_lower for w in ['everything about', 'summary of', 'overview of',
                                        'executive summary', 'tell me about']):
            if intent not in ('summary',):
                logger.debug("Intent override: %s → summary (summary signal)", intent)
                return 'summary'

        group_patterns = [r'\bby (plan type|region|visit type|specialty|department|medication|facility|status)\b',
                         r'\bper (plan type|region|visit type|specialty|department)\b',
                         r'\beach (plan type|region|visit type)\b',
                         r'\bbreak\s*down\b']
        if intent == 'aggregate' and not entities.get('temporal'):
            for gp in group_patterns:
                if re.search(gp, q_lower):
                    logger.debug("Intent override: aggregate → rank (group-by signal)")
                    return 'rank'

        if 'pmpm' in q_lower:
            return 'aggregate'

        return intent

    def _train(self):
        gen = TrainingDataGenerator(self.graph)
        training_pairs = gen.generate()

        documents = [q for q, _ in training_pairs]
        schema_terms = []
        for table_name, concept in self.graph.tables.items():
            schema_terms.append(table_name)
            schema_terms.append(concept.concept)
            schema_terms.extend(concept.synonyms)
            for col in self.graph.columns.get(table_name, {}):
                schema_terms.append(col)

        self.embeddings.train(documents, schema_terms)

        X = np.array([self.embeddings.encode(q) for q, _ in training_pairs])
        y = np.array([self.intent_classifier.label_to_idx.get(intent, 0) for _, intent in training_pairs])
        self.intent_classifier.train(X, y, epochs=500, lr=0.02)

    def process(self, question: str, session_id: str = '') -> Dict[str, Any]:
        t_start = time.time()
        result = {
            'sql': '', 'rows': [], 'columns': [], 'narrative': '',
            'source': 'healthcare_neural', 'confidence': 0.0,
            'intent': '', 'error': None, 'timing': {},
        }

        t_nlu = time.time()
        query_vec = self.embeddings.encode(question)
        intent, confidence = self.intent_classifier.predict(query_vec)
        result['timing']['nlu_ms'] = (time.time() - t_nlu) * 1000

        t_ent = time.time()
        entities = self.entity_extractor.extract(question)
        result['timing']['entity_ms'] = (time.time() - t_ent) * 1000

        q_lower = question.lower()
        intent = self._apply_intent_overrides(intent, confidence, entities, q_lower)
        result['intent'] = intent
        result['confidence'] = confidence

        t_sql = time.time()
        sql = self.sql_generator.generate(intent, entities, question)
        result['timing']['sql_ms'] = (time.time() - t_sql) * 1000

        if not sql:
            result['error'] = 'Failed to generate valid SQL'
            result['timing']['total_ms'] = (time.time() - t_start) * 1000
            return result

        result['sql'] = sql

        t_exec = time.time()
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description] if cur.description else []
            conn.close()
            result['rows'] = rows
            result['columns'] = columns
        except Exception as e:
            result['error'] = str(e)
            result['timing']['total_ms'] = (time.time() - t_start) * 1000
            return result
        result['timing']['exec_ms'] = (time.time() - t_exec) * 1000

        t_narr = time.time()
        result['narrative'] = self.narrator.narrate(
            question, intent, rows, columns, entities
        )
        result['timing']['narrative_ms'] = (time.time() - t_narr) * 1000

        result['timing']['total_ms'] = (time.time() - t_start) * 1000
        return result


def create_neural_engine(db_path: str, schema_graph=None) -> HealthcareNeuralEngine:
    return HealthcareNeuralEngine(db_path=db_path, schema_graph=schema_graph)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

    db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'healthcare_production.db')
    print("=" * 80)
    print("  HEALTHCARE NEURAL ENGINE — Self Test")
    print("=" * 80)

    engine = create_neural_engine(db_path)

    tests = [
        "how many members do we have?",
        "what is the denial rate?",
        "top 5 medications by cost",
        "claims per month",
        "average risk score",
        "which specialties see the most emergency cases",
        "how many denied claims",
        "total paid amount",
    ]

    for q in tests:
        result = engine.process(q)
        print(f"\nQ: {q}")
        print(f"  Intent: {result['intent']} (conf={result['confidence']:.2f})")
        print(f"  SQL: {result['sql'][:150]}")
        print(f"  Rows: {len(result['rows'])}")
        print(f"  Narrative: {result['narrative'][:120]}")
        if result['error']:
            print(f"  ERROR: {result['error']}")
        print(f"  Timing: {result['timing']}")
