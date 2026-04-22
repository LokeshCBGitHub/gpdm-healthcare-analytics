import json
import logging
import os
import re
import sqlite3
import time
import numpy as np
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger('gpdm.builtin_nlu')


class HealthcareEmbedder:

    DIM = 384

    def __init__(self):
        self._idf_weights: Dict[str, float] = {}
        self._synonyms: Dict[str, List[str]] = {}
        self._schema_terms: Set[str] = set()
        self._trained = False

    def train(self, documents: List[str], schema_terms: Set[str] = None,
              synonyms: Dict[str, List[str]] = None):
        self._schema_terms = schema_terms or set()
        self._synonyms = synonyms or {}

        doc_freq: Dict[str, int] = {}
        n_docs = len(documents)
        for doc in documents:
            terms = set(self._tokenize(doc))
            for t in terms:
                doc_freq[t] = doc_freq.get(t, 0) + 1

        for term, df in doc_freq.items():
            self._idf_weights[term] = np.log((n_docs + 1) / (df + 1)) + 1

        self._trained = True
        logger.info("HealthcareEmbedder trained: %d IDF terms, %d schema terms, %d synonyms",
                     len(self._idf_weights), len(self._schema_terms), len(self._synonyms))

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b[a-z][a-z0-9_]+\b', (text or '').lower())

    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        out = np.zeros((len(texts), self.DIM), dtype=np.float32)

        for i, text in enumerate(texts):
            t = (text or '').lower()

            expanded = t
            for syn_key, syn_list in self._synonyms.items():
                for syn in syn_list:
                    if syn in t:
                        expanded += f' {syn_key}'

            s = f'  {expanded}  '
            for j in range(len(s) - 2):
                ngram = s[j:j+3]
                h = (hash(ngram) & 0xFFFFFFFF)
                idx = h % self.DIM
                sign = 1.0 if (h >> 16) & 1 else -1.0

                weight = 1.0
                for term in self._tokenize(expanded):
                    if ngram.strip() in term:
                        weight = self._idf_weights.get(term, 1.0)
                        break

                for term in self._schema_terms:
                    if ngram.strip() in term.lower():
                        weight *= 1.5
                        break

                out[i, idx] += sign * weight

            words = self._tokenize(expanded)
            for word in words:
                h2 = (hash(f'__word__{word}') & 0xFFFFFFFF)
                idx2 = h2 % self.DIM
                sign2 = 1.0 if (h2 >> 16) & 1 else -1.0
                idf = self._idf_weights.get(word, 1.0)
                out[i, idx2] += sign2 * idf

        if normalize:
            norms = np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
            out = out / norms

        return out

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def similarity(self, text_a: str, text_b: str) -> float:
        va = self.encode_one(text_a)
        vb = self.encode_one(text_b)
        return float(np.dot(va, vb))


class BuiltInIntentClassifier:

    LABELS = [
        'aggregate', 'breakdown', 'comparison', 'count', 'filter',
        'lookup', 'percentage', 'ranking', 'rate', 'trend'
    ]

    def __init__(self, embedder: HealthcareEmbedder, hidden_dim: int = 128):
        self.embedder = embedder
        self.input_dim = embedder.DIM
        self.hidden_dim = hidden_dim
        self.n_classes = len(self.LABELS)

        rng = np.random.default_rng(42)
        scale1 = np.sqrt(2.0 / self.input_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        self.W1 = rng.normal(0, scale1, (self.input_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = rng.normal(0, scale2, (hidden_dim, self.n_classes)).astype(np.float32)
        self.b2 = np.zeros(self.n_classes, dtype=np.float32)
        self._trained = False

    def train(self, texts: List[str], labels: List[str],
              epochs: int = 100, lr: float = 0.01) -> Dict[str, Any]:
        t0 = time.time()

        X = self.embedder.encode(texts)
        label_to_idx = {l: i for i, l in enumerate(self.LABELS)}
        Y = np.array([label_to_idx.get(l, 0) for l in labels])

        Y_oh = np.zeros((len(Y), self.n_classes), dtype=np.float32)
        for i, y in enumerate(Y):
            Y_oh[i, y] = 1.0

        best_loss = float('inf')
        best_acc = 0.0
        best_weights = None

        mW1 = np.zeros_like(self.W1); vW1 = np.zeros_like(self.W1)
        mb1 = np.zeros_like(self.b1); vb1 = np.zeros_like(self.b1)
        mW2 = np.zeros_like(self.W2); vW2 = np.zeros_like(self.W2)
        mb2 = np.zeros_like(self.b2); vb2 = np.zeros_like(self.b2)
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for epoch in range(epochs):
            n = len(Y)
            batch_size = min(64, n)
            indices = np.random.permutation(n)

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                batch_idx = indices[start:start + batch_size]
                Xb = X[batch_idx]
                Yb = Y[batch_idx]
                Yb_oh = Y_oh[batch_idx]
                bs = len(batch_idx)

                z1 = Xb @ self.W1 + self.b1
                a1 = np.maximum(z1, 0)

                z2 = a1 @ self.W2 + self.b2
                exp_z = np.exp(z2 - z2.max(axis=1, keepdims=True))
                probs = exp_z / exp_z.sum(axis=1, keepdims=True)

                batch_loss = -np.mean(np.log(probs[np.arange(bs), Yb] + 1e-10))
                epoch_loss += batch_loss
                n_batches += 1

                dz2 = (probs - Yb_oh) / bs
                dW2 = a1.T @ dz2
                db2 = dz2.sum(axis=0)

                da1 = dz2 @ self.W2.T
                dz1 = da1 * (z1 > 0).astype(np.float32)
                dW1 = Xb.T @ dz1
                db1 = dz1.sum(axis=0)

                reg = 1e-4
                dW1 += reg * self.W1
                dW2 += reg * self.W2

                step = epoch * max(1, n // batch_size) + n_batches
                for param, grad, m, v in [
                    (self.W1, dW1, mW1, vW1), (self.b1, db1, mb1, vb1),
                    (self.W2, dW2, mW2, vW2), (self.b2, db2, mb2, vb2),
                ]:
                    m[:] = beta1 * m + (1 - beta1) * grad
                    v[:] = beta2 * v + (1 - beta2) * (grad ** 2)
                    m_hat = m / (1 - beta1 ** (step + 1))
                    v_hat = v / (1 - beta2 ** (step + 1))
                    param -= lr * m_hat / (np.sqrt(v_hat) + eps)

            avg_loss = epoch_loss / max(n_batches, 1)

            if epoch % 10 == 0 or epoch == epochs - 1:
                z1_full = X @ self.W1 + self.b1
                a1_full = np.maximum(z1_full, 0)
                z2_full = a1_full @ self.W2 + self.b2
                exp_z_full = np.exp(z2_full - z2_full.max(axis=1, keepdims=True))
                probs_full = exp_z_full / exp_z_full.sum(axis=1, keepdims=True)
                acc = (probs_full.argmax(axis=1) == Y).mean()

                if acc > best_acc or (acc == best_acc and avg_loss < best_loss):
                    best_loss = avg_loss
                    best_acc = acc
                    best_weights = (
                        self.W1.copy(), self.b1.copy(),
                        self.W2.copy(), self.b2.copy()
                    )
                if epoch % 50 == 0:
                    logger.debug("Epoch %d: loss=%.4f, acc=%.2f%%", epoch, avg_loss, acc * 100)

        if best_weights:
            self.W1, self.b1, self.W2, self.b2 = best_weights

        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(z1, 0)
        z2 = a1 @ self.W2 + self.b2
        exp_z = np.exp(z2 - z2.max(axis=1, keepdims=True))
        probs = exp_z / exp_z.sum(axis=1, keepdims=True)
        final_acc = (probs.argmax(axis=1) == Y).mean()

        self._trained = True
        elapsed = time.time() - t0

        logger.info("Intent classifier trained: %d examples, %d epochs, "
                     "acc=%.1f%%, loss=%.4f in %.1fs",
                     len(texts), epochs, final_acc * 100, best_loss, elapsed)

        return {
            'trained': True,
            'examples': len(texts),
            'accuracy': float(final_acc),
            'best_loss': float(best_loss),
            'elapsed_s': round(elapsed, 1),
        }

    def predict(self, text: str) -> Dict[str, Any]:
        x = self.embedder.encode_one(text)

        z1 = x @ self.W1 + self.b1
        a1 = np.maximum(z1, 0)
        z2 = a1 @ self.W2 + self.b2
        exp_z = np.exp(z2 - z2.max())
        probs = exp_z / exp_z.sum()

        idx = int(np.argmax(probs))
        all_scores = {self.LABELS[i]: float(probs[i]) for i in range(self.n_classes)}

        return {
            'intent': self.LABELS[idx],
            'confidence': float(probs[idx]),
            'all_scores': dict(sorted(all_scores.items(), key=lambda x: x[1], reverse=True)),
            'source': 'builtin_nlu',
        }


class SchemaEntityExtractor:

    def __init__(self, schema_learner):
        self.learner = schema_learner
        self._value_map: Dict[str, Dict] = {}
        self._build_value_index()

    def _build_value_index(self):
        for tbl_name, profiles in self.learner.tables.items():
            for p in profiles:
                if p.is_categorical and p.sample_values:
                    for val in p.sample_values:
                        val_str = str(val).strip()
                        if len(val_str) >= 2:
                            key = val_str.upper()
                            self._value_map[key] = {
                                'table': tbl_name,
                                'column': p.name,
                                'value': val_str,
                                'distinct_count': p.distinct_count or 0,
                            }
        logger.info("Entity index built: %d values from %d tables",
                     len(self._value_map), len(self.learner.tables))

    def extract(self, question: str) -> List[Dict[str, Any]]:
        entities = []
        q_upper = question.upper()

        sorted_values = sorted(self._value_map.keys(), key=len, reverse=True)
        matched_spans = set()

        for val_key in sorted_values:
            if val_key in q_upper:
                start = q_upper.find(val_key)
                end = start + len(val_key)
                span = (start, end)
                overlaps = any(s[0] < span[1] and s[1] > span[0] for s in matched_spans)
                if overlaps:
                    continue

                if len(val_key) <= 3:
                    before_ok = (start == 0 or not q_upper[start - 1].isalpha())
                    after_ok = (end >= len(q_upper) or not q_upper[end].isalpha())
                    if not (before_ok and after_ok):
                        continue

                info = self._value_map[val_key]
                entities.append({
                    'text': info['value'],
                    'type': self._infer_type(info['column']),
                    'column': info['column'],
                    'table': info['table'],
                    'source': 'schema_lookup',
                    'confidence': 0.95,
                })
                matched_spans.add(span)

        date_patterns = [
            (r'\b(20\d{2})\b', 'YEAR'),
            (r'\b(january|february|march|april|may|june|july|august|'
             r'september|october|november|december)\b', 'MONTH'),
            (r'\b(q[1-4]|quarter\s*[1-4])\b', 'QUARTER'),
            (r'\b(last\s+(?:month|year|quarter)|this\s+(?:month|year|quarter))\b', 'TIME_PERIOD'),
        ]
        for pattern, etype in date_patterns:
            m = re.search(pattern, question, re.IGNORECASE)
            if m:
                entities.append({
                    'text': m.group(0),
                    'type': etype,
                    'column': None,
                    'table': None,
                    'source': 'pattern',
                    'confidence': 0.85,
                })

        num_patterns = [
            (r'(?:greater|more|over|above|exceeds?|>)\s*(?:than\s+)?\$?([\d,]+(?:\.\d+)?)', 'THRESHOLD_GT'),
            (r'(?:less|under|below|fewer|<)\s*(?:than\s+)?\$?([\d,]+(?:\.\d+)?)', 'THRESHOLD_LT'),
            (r'(?:top|bottom|first|last)\s+(\d+)', 'TOP_N'),
        ]
        for pattern, etype in num_patterns:
            m = re.search(pattern, question, re.IGNORECASE)
            if m:
                entities.append({
                    'text': m.group(1).replace(',', ''),
                    'type': etype,
                    'column': None,
                    'table': None,
                    'source': 'pattern',
                    'confidence': 0.9,
                })

        return entities

    def _infer_type(self, column_name: str) -> str:
        cn = column_name.upper()
        if 'STATUS' in cn:
            return 'STATUS'
        if 'TYPE' in cn:
            return 'TYPE'
        if 'PLAN' in cn:
            return 'PLAN'
        if 'SPECIALTY' in cn or 'PROVIDER' in cn:
            return 'PROVIDER'
        if 'REGION' in cn or 'STATE' in cn:
            return 'LOCATION'
        if 'GENDER' in cn or 'SEX' in cn:
            return 'DEMOGRAPHIC'
        if 'MEDICATION' in cn or 'DRUG' in cn:
            return 'MEDICATION'
        if 'DIAGNOSIS' in cn or 'ICD' in cn or 'DX' in cn:
            return 'DIAGNOSIS'
        if 'CPT' in cn or 'PROCEDURE' in cn:
            return 'PROCEDURE'
        return 'VALUE'


class MetricDimensionResolver:

    AGG_PATTERNS = [
        (r'\b(?:average|avg|mean)\b', 'AVG'),
        (r'\b(?:total|sum|sum\s+of)\b', 'SUM'),
        (r'\b(?:count|number\s+of|how\s+many|volume)\b', 'COUNT'),
        (r'\b(?:max|maximum|highest|greatest)\b', 'MAX'),
        (r'\b(?:min|minimum|lowest|least|smallest)\b', 'MIN'),
        (r'\b(?:top|bottom)\b', 'COUNT'),
    ]

    DIM_PATTERNS = [
        r'\bby\s+(\w+(?:\s+\w+)?)',
        r'\bper\s+(\w+(?:\s+\w+)?)',
        r'\bacross\s+(\w+)',
        r'\bgrouped?\s+by\s+(\w+(?:\s+\w+)?)',
        r'\bfor\s+each\s+(\w+(?:\s+\w+)?)',
        r'\bbroken?\s+down\s+by\s+(\w+(?:\s+\w+)?)',
    ]

    def __init__(self, schema_learner, embedder: HealthcareEmbedder):
        self.learner = schema_learner
        self.embedder = embedder
        self._col_embeddings: Dict[str, np.ndarray] = {}
        self._col_info: Dict[str, Dict] = {}
        self._build_column_index()

    def _build_column_index(self):
        for tbl_name, profiles in self.learner.tables.items():
            for p in profiles:
                desc_parts = [
                    p.name.replace('_', ' ').lower(),
                    tbl_name.replace('_', ' ').lower(),
                ]
                if p.semantic_tags:
                    desc_parts.extend(p.semantic_tags)

                desc = ' '.join(desc_parts)
                key = f'{tbl_name}.{p.name}'
                self._col_embeddings[key] = self.embedder.encode_one(desc)
                self._col_info[key] = {
                    'table': tbl_name,
                    'column': p.name,
                    'is_numeric': p.is_numeric,
                    'is_categorical': p.is_categorical,
                    'is_date': p.is_date,
                    'is_id': p.is_id,
                    'distinct_count': p.distinct_count,
                    'row_count': self.learner.table_row_counts.get(tbl_name, 0),
                }

    def resolve_metric(self, question: str) -> Optional[Dict[str, Any]]:
        q = question.lower()

        agg = 'COUNT'
        for pattern, agg_type in self.AGG_PATTERNS:
            if re.search(pattern, q):
                agg = agg_type
                break

        metric_q = q
        for dim_p in self.DIM_PATTERNS:
            metric_q = re.sub(dim_p, '', metric_q)
        metric_q = re.sub(r'\b(?:average|avg|mean|total|sum|count|number|how\s+many|'
                          r'max|min|top|bottom|highest|lowest)\b', '', metric_q)
        metric_q = metric_q.strip()

        if not metric_q or len(metric_q) < 3:
            return None

        q_vec = self.embedder.encode_one(metric_q)
        best_score = -1.0
        best_col = None

        for key, col_vec in self._col_embeddings.items():
            info = self._col_info[key]
            if not info['is_numeric'] or info['is_id']:
                continue
            score = float(np.dot(q_vec, col_vec))
            if score > best_score:
                best_score = score
                best_col = info

        if best_col and best_score > 0.1:
            return {
                'column': best_col['column'],
                'table': best_col['table'],
                'aggregation': agg,
                'confidence': min(best_score + 0.3, 1.0),
            }
        return None

    def resolve_dimension(self, question: str) -> Optional[Dict[str, Any]]:
        q = question.lower()

        dim_phrase = None
        for pattern in self.DIM_PATTERNS:
            m = re.search(pattern, q)
            if m:
                dim_phrase = m.group(1).strip()
                dim_phrase = re.sub(r'\b(?:the|a|an|in|of|for)\b$', '', dim_phrase).strip()
                break

        if not dim_phrase:
            return None

        q_vec = self.embedder.encode_one(dim_phrase)
        best_score = -1.0
        best_col = None

        for key, col_vec in self._col_embeddings.items():
            info = self._col_info[key]
            if info['is_id']:
                continue
            score = float(np.dot(q_vec, col_vec))
            if info['is_categorical']:
                score += 0.2
            if info['distinct_count'] and info['row_count']:
                ratio = info['distinct_count'] / max(info['row_count'], 1)
                if ratio > 0.5:
                    score -= 0.3
            if score > best_score:
                best_score = score
                best_col = info

        if best_col and best_score > 0.1:
            return {
                'column': best_col['column'],
                'table': best_col['table'],
                'confidence': min(best_score + 0.2, 1.0),
            }
        return None


def generate_augmented_training_data(schema_learner) -> Tuple[List[str], List[str]]:
    texts: List[str] = []
    labels: List[str] = []

    tables = list(schema_learner.tables.keys())

    for tbl_name, profiles in schema_learner.tables.items():
        tbl_readable = tbl_name.replace('_', ' ')
        numeric_cols = [p for p in profiles if p.is_numeric and not p.is_id]
        categorical_cols = [p for p in profiles if p.is_categorical and not p.is_id]
        date_cols = [p for p in profiles if p.is_date]
        id_cols = [p for p in profiles if p.is_id]

        texts.extend([
            f'how many {tbl_readable} are there',
            f'count of {tbl_readable}',
            f'total number of {tbl_readable}',
            f'volume of {tbl_readable}',
            f'{tbl_readable} count',
        ])
        labels.extend(['count'] * 5)

        for nc in numeric_cols[:3]:
            col_readable = nc.name.replace('_', ' ').lower()

            texts.extend([
                f'average {col_readable}',
                f'total {col_readable}',
                f'sum of {col_readable}',
                f'mean {col_readable}',
                f'what is the average {col_readable}',
                f'total {col_readable} across all {tbl_readable}',
            ])
            labels.extend(['aggregate'] * 6)

            texts.extend([
                f'top 10 {tbl_readable} by {col_readable}',
                f'highest {col_readable}',
                f'lowest {col_readable}',
                f'which {tbl_readable} have the most {col_readable}',
            ])
            labels.extend(['ranking'] * 4)

            for cc in categorical_cols[:2]:
                cat_readable = cc.name.replace('_', ' ').lower()

                texts.extend([
                    f'{col_readable} by {cat_readable}',
                    f'average {col_readable} per {cat_readable}',
                    f'total {col_readable} grouped by {cat_readable}',
                    f'{col_readable} breakdown by {cat_readable}',
                ])
                labels.extend(['breakdown'] * 4)

                texts.extend([
                    f'compare {col_readable} across {cat_readable}',
                    f'{col_readable} difference by {cat_readable}',
                ])
                labels.extend(['comparison'] * 2)

        for cc in categorical_cols[:2]:
            cat_readable = cc.name.replace('_', ' ').lower()

            texts.extend([
                f'percentage of {tbl_readable} by {cat_readable}',
                f'what percent of {tbl_readable} are in each {cat_readable}',
                f'proportion of {tbl_readable} by {cat_readable}',
            ])
            labels.extend(['percentage'] * 3)

            if cc.sample_values:
                sample_val = str(list(cc.sample_values)[0])
                texts.extend([
                    f'{tbl_readable} where {cat_readable} is {sample_val}',
                    f'show {tbl_readable} with {cat_readable} = {sample_val}',
                    f'filter {tbl_readable} by {cat_readable}',
                ])
                labels.extend(['filter'] * 3)

        if date_cols:
            dc = date_cols[0]
            date_readable = dc.name.replace('_', ' ').lower()

            texts.extend([
                f'{tbl_readable} trend over time',
                f'{tbl_readable} by month',
                f'monthly {tbl_readable} count',
                f'{tbl_readable} over the past year',
                f'{tbl_readable} trend by {date_readable}',
            ])
            labels.extend(['trend'] * 5)

        texts.extend([
            f'show all {tbl_readable}',
            f'list {tbl_readable}',
            f'details of {tbl_readable}',
        ])
        labels.extend(['lookup'] * 3)

    healthcare_patterns = [
        ('denial rate', 'rate'), ('readmission rate', 'rate'),
        ('yield rate', 'rate'), ('no show rate', 'rate'),
        ('fill rate', 'rate'), ('approval rate', 'rate'),

        ('PMPM cost', 'aggregate'), ('cost per member', 'aggregate'),
        ('average length of stay', 'aggregate'), ('ALOS', 'aggregate'),
        ('cost per encounter', 'aggregate'),

        ('billed vs paid', 'comparison'), ('paid vs allowed', 'comparison'),
        ('inpatient vs outpatient', 'comparison'),

        ('top providers', 'ranking'), ('top diagnoses', 'ranking'),
        ('most prescribed medications', 'ranking'),
        ('highest cost procedures', 'ranking'),

        ('cost trend', 'trend'), ('enrollment over time', 'trend'),
        ('claim volume by quarter', 'trend'),

        ('claims by region', 'breakdown'), ('cost by specialty', 'breakdown'),
        ('members by plan type', 'breakdown'),

        ('denial percentage', 'percentage'),
        ('what percentage are denied', 'percentage'),

        ('member count', 'count'), ('provider count', 'count'),

        ('denied claims', 'filter'), ('claims over 10000', 'filter'),
        ('emergency encounters', 'filter'),

        ('show me member details', 'lookup'),
        ('what is CPT code 99213', 'lookup'),
    ]
    for text, label in healthcare_patterns:
        texts.append(text)
        labels.append(label)

    ambiguous_patterns = [
        ('which providers are bleeding money', 'ranking'),
        ('where are we losing on reimbursements', 'rate'),
        ('how are we doing on denials', 'rate'),
        ('give me the big picture on claims', 'breakdown'),
        ('what is going on with our costs', 'trend'),
        ('who are our best providers', 'ranking'),
        ('worst performing plans', 'ranking'),
        ('show me the money trail', 'aggregate'),
    ]
    for text, label in ambiguous_patterns:
        texts.append(text)
        labels.append(label)

    logger.info("Generated %d augmented training examples from %d tables",
                 len(texts), len(tables))
    return texts, labels


class BuiltInNLU:

    def __init__(self, schema_learner, db_path: str = None):
        t0 = time.time()
        self.learner = schema_learner
        self.db_path = db_path

        synonyms = self._build_synonyms()
        schema_terms = set()
        for tbl, profiles in schema_learner.tables.items():
            schema_terms.add(tbl.lower())
            for p in profiles:
                schema_terms.add(p.name.lower())
                schema_terms.update(p.semantic_tags)

        self.embedder = HealthcareEmbedder()
        corpus = self._build_corpus()
        self.embedder.train(corpus, schema_terms=schema_terms, synonyms=synonyms)

        texts, labels = generate_augmented_training_data(schema_learner)
        self.intent_classifier = BuiltInIntentClassifier(self.embedder)
        train_result = self.intent_classifier.train(texts, labels, epochs=300, lr=0.005)

        self.entity_extractor = SchemaEntityExtractor(schema_learner)

        self.metric_resolver = MetricDimensionResolver(schema_learner, self.embedder)

        elapsed = time.time() - t0
        logger.info("BuiltInNLU ready: %d training examples, %.1f%% accuracy, "
                     "%d entity values, trained in %.1fs",
                     train_result['examples'], train_result['accuracy'] * 100,
                     len(self.entity_extractor._value_map), elapsed)

    def _build_corpus(self) -> List[str]:
        docs = []
        for tbl, profiles in self.learner.tables.items():
            for p in profiles:
                docs.append(f"{tbl} {p.name.replace('_', ' ')} {' '.join(p.semantic_tags)}")
            docs.append(f"table {tbl} with {len(profiles)} columns")
        return docs

    def _build_synonyms(self) -> Dict[str, List[str]]:
        return {
            'paid_amount': ['paid', 'payment', 'reimbursement', 'cost'],
            'billed_amount': ['billed', 'charges', 'charge'],
            'allowed_amount': ['allowed', 'allowable'],
            'claim': ['claim', 'claims', 'medical claim'],
            'member': ['member', 'patient', 'enrollee', 'beneficiary'],
            'provider': ['provider', 'doctor', 'physician', 'clinician'],
            'encounter': ['encounter', 'visit', 'admission'],
            'prescription': ['prescription', 'rx', 'medication', 'drug'],
            'diagnosis': ['diagnosis', 'dx', 'condition', 'disease'],
            'copay': ['copay', 'copayment', 'co-pay'],
            'deductible': ['deductible', 'ded'],
            'coinsurance': ['coinsurance', 'co-insurance'],
            'denial': ['denial', 'denied', 'reject', 'rejected'],
            'specialty': ['specialty', 'speciality', 'spec'],
            'referral': ['referral', 'refer', 'referred'],
        }

    def understand(self, question: str) -> Dict[str, Any]:
        t0 = time.time()

        intent_result = self.intent_classifier.predict(question)

        entities = self.entity_extractor.extract(question)

        metric = self.metric_resolver.resolve_metric(question)

        dimension = self.metric_resolver.resolve_dimension(question)

        is_ambiguous = self._is_ambiguous(question)

        explanation = self._build_explanation(question, intent_result, entities, metric, dimension)

        elapsed_ms = round((time.time() - t0) * 1000)

        return {
            'intent': intent_result['intent'],
            'intent_confidence': intent_result['confidence'],
            'intent_all_scores': intent_result.get('all_scores', {}),
            'entities': entities,
            'metric': metric,
            'dimension': dimension,
            'is_ambiguous': is_ambiguous,
            'explanation': explanation,
            'understanding_source': 'builtin_nlu',
            'latency_ms': elapsed_ms,
        }

    def _is_ambiguous(self, question: str) -> bool:
        q = question.lower()
        patterns = [
            r'\b(bleeding|hemorrhaging|burning)\s+(money|cash)',
            r'\b(going on|happening)\s+with\b',
            r'\bgive me.*(picture|overview|snapshot)',
            r'\bhow.*(doing|performing|looking)\b',
            r'\bwhat.s wrong\b',
            r'\bwhere are we\b',
            r'\btell me about\b',
        ]
        return any(re.search(p, q) for p in patterns)

    def _build_explanation(self, question: str, intent: Dict,
                            entities: List, metric: Dict,
                            dimension: Dict) -> str:
        parts = [f"Understood as a '{intent['intent']}' query"]

        if metric:
            parts.append(f"computing {metric['aggregation']}({metric['column']}) "
                         f"from {metric['table']}")

        if dimension:
            parts.append(f"grouped by {dimension['column']}")

        if entities:
            ent_descs = [f"{e['text']} ({e['type']})" for e in entities[:3]]
            parts.append(f"filtering by {', '.join(ent_descs)}")

        parts.append(f"with {intent['confidence']:.0%} confidence")

        return ' — '.join(parts) + '.'
