from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger('gpdm.deep_understanding')


_WB_CACHE: Dict[str, re.Pattern] = {}

_BOUNDARY_REQUIRED = {
    'age', 'per', 'old', 'sex', 'by', 'ppo', 'pos', 'epo', 'vs', 'er',
    'md', 'do', 'pa', 'np', 'visit', 'plan', 'rate', 'date', 'total',
    'count', 'sum', 'mean', 'month', 'year', 'show', 'list', 'find',
    'top', 'best', 'worst', 'most', 'least', 'each', 'split',
    'patient', 'patients',
}


def _phrase_in_text(phrase: str, text: str) -> bool:
    words = phrase.split()

    if len(words) >= 2:
        return phrase in text

    word = phrase.lower()
    need_boundary = (len(word) <= 5) or (word in _BOUNDARY_REQUIRED)

    if need_boundary:
        if word not in _WB_CACHE:
            _WB_CACHE[word] = re.compile(r'(?<![a-z])' + re.escape(word) + r'(?![a-z])')
        return bool(_WB_CACHE[word].search(text))
    else:
        return word in text


@dataclass
class Entity:
    text: str
    type: str
    resolved_to: str = ''
    confidence: float = 0.0
    source: str = ''


@dataclass
class DeepUnderstanding:
    intent: str = 'unknown'
    intent_confidence: float = 0.0
    secondary_intents: List[str] = field(default_factory=list)

    entities: List[Entity] = field(default_factory=list)

    target_tables: List[str] = field(default_factory=list)
    target_columns: List[str] = field(default_factory=list)
    metric_columns: List[Tuple[str, str]] = field(default_factory=list)
    dimension_columns: List[str] = field(default_factory=list)
    filter_conditions: List[Dict] = field(default_factory=list)
    join_path: List[Tuple[str, str, str, str]] = field(default_factory=list)

    sql_template: str = ''
    sql_approach: str = ''

    reasoning_chain: List[str] = field(default_factory=list)
    attention_weights: Dict[str, float] = field(default_factory=dict)

    source: str = 'deep_understanding'
    latency_ms: int = 0
    history_match_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'intent': self.intent,
            'intent_confidence': self.intent_confidence,
            'secondary_intents': self.secondary_intents,
            'entities': [{'text': e.text, 'type': e.type,
                          'resolved_to': e.resolved_to, 'confidence': e.confidence}
                         for e in self.entities],
            'target_tables': self.target_tables,
            'target_columns': self.target_columns,
            'metric_columns': self.metric_columns,
            'dimension_columns': self.dimension_columns,
            'filter_conditions': self.filter_conditions,
            'sql_template': self.sql_template,
            'sql_approach': self.sql_approach,
            'reasoning_chain': self.reasoning_chain,
            'attention_weights': self.attention_weights,
            'source': self.source,
            'latency_ms': self.latency_ms,
        }


HEALTHCARE_CONCEPTS = {
    'cost': {
        'synonyms': ['cost', 'expense', 'spend', 'spending', 'paid', 'billed',
                     'charge', 'payment', 'reimbursement', 'amount', 'dollar',
                     'money', 'financial', 'expensive', 'costly', 'bleeding money',
                     'burning cash', 'hemorrhaging', 'revenue', 'income', 'total_cost'],
        'columns': ['paid_amount', 'billed_amount', 'allowed_amount', 'total_cost',
                    'net_payment', 'copay', 'deductible', 'coinsurance'],
        'agg': 'SUM',
        'type': 'METRIC',
        'semantic_group': 'financial',
    },

    'utilization': {
        'synonyms': ['utilization', 'usage', 'visits', 'encounters', 'admissions',
                     'er visits', 'inpatient', 'outpatient', 'volume', 'throughput',
                     'frequency', 'how many', 'count', 'total', 'visit'],
        'columns': ['encounter_id', 'visit_type', 'admission_type', 'visit_count'],
        'agg': 'COUNT',
        'type': 'METRIC',
        'semantic_group': 'utilization',
    },

    'denial': {
        'synonyms': ['denial', 'denied', 'reject', 'rejected', 'denial rate',
                     'not approved', 'claim denied', 'authorization denied',
                     'claim rejection', 'denied claims'],
        'columns': ['claim_status', 'adjudication_status'],
        'values': ['DENIED', 'REJECTED', 'DENY'],
        'agg': 'rate',
        'type': 'RATE',
        'semantic_group': 'quality',
    },

    'readmission': {
        'synonyms': ['readmission', 'readmit', 'readmitted', 'bounce back',
                     'return visit', 'came back', 'within 30 days', 'readmission rate',
                     '30-day readmission', 'return to hospital'],
        'columns': ['admit_date', 'discharge_date', 'member_id'],
        'tables': ['encounters'],
        'agg': 'rate',
        'type': 'RATE',
        'semantic_group': 'quality',
    },

    'pmpm': {
        'synonyms': ['pmpm', 'per member per month', 'member month cost',
                     'monthly cost per member', 'cost per member', 'per capita'],
        'columns': ['paid_amount', 'member_id'],
        'agg': 'PMPM',
        'type': 'DERIVED_METRIC',
        'semantic_group': 'financial',
    },

    'yield': {
        'synonyms': ['yield', 'yield rate', 'collection rate', 'paid vs billed',
                     'reimbursement rate', 'payment rate', 'how much collected',
                     'yield percentage'],
        'columns': ['paid_amount', 'billed_amount', 'allowed_amount'],
        'agg': 'ratio',
        'type': 'DERIVED_METRIC',
        'semantic_group': 'financial',
    },

    'los': {
        'synonyms': ['length of stay', 'los', 'alos', 'average length', 'days in hospital',
                     'stay duration', 'how long', 'inpatient days'],
        'columns': ['admit_date', 'discharge_date', 'los'],
        'agg': 'AVG',
        'type': 'DERIVED_METRIC',
        'semantic_group': 'utilization',
    },

    'risk': {
        'synonyms': ['risk', 'risk score', 'high risk', 'at risk', 'hcc', 'raf',
                     'acuity', 'severity', 'risk stratification', 'risk level',
                     'vulnerable', 'complex', 'chronically ill'],
        'columns': ['risk_score', 'hcc_score', 'acuity_score', 'frailty_score'],
        'agg': 'AVG',
        'type': 'METRIC',
        'semantic_group': 'clinical',
    },

    'quality': {
        'synonyms': ['quality', 'hedis', 'star rating', 'stars', 'measure',
                     'performance', 'compliance', 'gap in care', 'care gap',
                     'screening', 'preventive'],
        'columns': ['hedis_measure', 'quality_score', 'star_rating'],
        'agg': 'rate',
        'type': 'RATE',
        'semantic_group': 'quality',
    },

    'diagnosis': {
        'synonyms': ['diagnosis', 'diagnosed', 'condition', 'disease', 'icd', 'dx',
                     'comorbidity', 'chronic', 'acute', 'diabetes', 'hypertension',
                     'copd', 'cancer', 'heart failure', 'asthma', 'depression'],
        'columns': ['primary_dx', 'diagnosis_code', 'dx_code', 'icd_code'],
        'type': 'DIMENSION',
        'semantic_group': 'clinical',
    },

    'provider': {
        'synonyms': ['provider', 'doctor', 'physician', 'npi', 'specialist',
                     'primary care', 'pcp', 'facility', 'hospital', 'clinic',
                     'practice', 'practitioner'],
        'columns': ['provider_id', 'npi', 'rendering_npi', 'specialty', 'provider_name'],
        'type': 'DIMENSION',
        'semantic_group': 'organizational',
    },

    'member': {
        'synonyms': ['member', 'patient', 'enrollee', 'subscriber', 'beneficiary',
                     'person', 'individual', 'population'],
        'columns': ['member_id', 'patient_id', 'member_name'],
        'type': 'ENTITY',
        'semantic_group': 'entity',
    },

    'time': {
        'synonyms': ['month', 'year', 'quarter', 'weekly', 'daily', 'trend',
                     'over time', 'by month', 'by year', 'period', 'date',
                     'this year', 'last year', 'ytd', 'mtd', 'historical',
                     'temporal'],
        'columns': ['service_date', 'admit_date', 'claim_date', 'discharge_date'],
        'type': 'TIME_DIMENSION',
        'semantic_group': 'temporal',
    },

    'geography': {
        'synonyms': ['region', 'state', 'county', 'zip', 'market', 'area',
                     'location', 'where', 'geographic', 'by region', 'by state'],
        'columns': ['region', 'state', 'zip_code', 'service_area', 'county'],
        'type': 'DIMENSION',
        'semantic_group': 'organizational',
    },

    'plan': {
        'synonyms': ['plan', 'plan type', 'lob', 'line of business', 'commercial',
                     'medicare', 'medicaid', 'product', 'hmo', 'ppo', 'epo',
                     'pos', 'coverage', 'product line'],
        'columns': ['plan_type', 'lob', 'product_name', 'plan_name'],
        'type': 'DIMENSION',
        'semantic_group': 'organizational',
    },

    'demographics': {
        'synonyms': ['age', 'gender', 'sex', 'demographic', 'age group', 'male',
                     'female', 'elderly', 'pediatric', 'adult', 'young', 'old',
                     'senior', 'age bracket'],
        'columns': ['age', 'gender', 'date_of_birth', 'age_group'],
        'type': 'DIMENSION',
        'semantic_group': 'entity',
    },
}

INTENT_PATTERNS = {
    'count': {
        'triggers': ['how many', 'count', 'total number', 'number of', 'how much count'],
        'agg': 'COUNT',
        'needs_groupby': False,
        'sql_shape': 'SELECT COUNT(...)',
    },
    'aggregate': {
        'triggers': ['total', 'sum', 'average', 'mean', 'median', 'how much',
                     'overall', 'combined'],
        'agg': 'SUM/AVG',
        'needs_groupby': False,
        'sql_shape': 'SELECT AGG(...)',
    },
    'trend': {
        'triggers': ['trend', 'over time', 'by month', 'by year', 'monthly', 'growth',
                     'change', 'increase', 'decrease', 'trajectory', 'time series'],
        'agg': 'time_series',
        'needs_groupby': True,
        'sql_shape': 'SELECT time_period, AGG(...) GROUP BY time_period',
    },
    'comparison': {
        'triggers': ['compare', 'versus', 'vs', 'difference', 'better', 'worse',
                     'highest', 'lowest', 'rank', 'benchmark', 'relative'],
        'agg': 'compare',
        'needs_groupby': True,
        'sql_shape': 'GROUP BY dimension, show metrics for comparison',
    },
    'ranking': {
        'triggers': ['top', 'bottom', 'best', 'worst', 'highest', 'lowest', 'most',
                     'least', 'leading', 'biggest', 'smallest', 'sorted'],
        'agg': 'ORDER BY',
        'needs_groupby': True,
        'sql_shape': 'ORDER BY metric DESC/ASC LIMIT N',
    },
    'breakdown': {
        'triggers': ['breakdown', 'by', 'split', 'segment', 'group', 'distribution',
                     'per', 'each', 'across'],
        'agg': 'GROUP BY',
        'needs_groupby': True,
        'sql_shape': 'GROUP BY dimension',
    },
    'rate': {
        'triggers': ['rate', 'percentage', 'percent', 'ratio', 'proportion', 'share',
                     'fraction', 'yield', 'pct', '%'],
        'agg': 'rate',
        'needs_groupby': False,
        'sql_shape': 'COUNT(filtered) / COUNT(*)',
    },
    'detail': {
        'triggers': ['show', 'list', 'detail', 'which', 'who', 'find', 'give me',
                     'get', 'pull', 'display'],
        'agg': 'SELECT',
        'needs_groupby': False,
        'sql_shape': 'SELECT * FROM ...',
    },
    'executive': {
        'triggers': ['summary', 'overview', 'dashboard', 'executive', 'briefing',
                     'kpi', 'scorecard', 'health of', 'snapshot'],
        'agg': 'multi',
        'needs_groupby': False,
        'sql_shape': 'Multiple aggregations',
    },
}

BENCHMARKS = {
    'denial_rate': {'industry_avg': 0.08, 'good': 0.05, 'concerning': 0.15},
    'readmission_30d': {'industry_avg': 0.16, 'good': 0.12, 'concerning': 0.20},
    'pmpm_commercial': {'industry_avg': 450, 'good': 350, 'concerning': 600},
    'pmpm_medicare': {'industry_avg': 950, 'good': 800, 'concerning': 1200},
    'er_per_1000': {'industry_avg': 450, 'good': 350, 'concerning': 550},
    'yield_rate': {'industry_avg': 0.90, 'good': 0.95, 'concerning': 0.80},
    'alos': {'industry_avg': 4.5, 'good': 3.5, 'concerning': 6.0},
    'ar_days': {'industry_avg': 35, 'good': 25, 'concerning': 50},
}


class _SkipGramEmbedder:

    def __init__(self, dim: int = 384, epochs: int = 20, window: int = 4):
        self.dim = dim
        self.epochs = epochs
        self.window = window
        self.vocab: Dict[str, int] = {}
        self._vectors: Optional[np.ndarray] = None
        self._context_vecs: Optional[np.ndarray] = None
        self._vocab_freq: Dict[str, int] = {}
        self._trained = False

    @property
    def size(self) -> int:
        return len(self.vocab)

    def train(self, schema_learner, healthcare_concepts: Dict):
        self._build_vocab(schema_learner, healthcare_concepts)
        n = len(self.vocab)

        np.random.seed(42)
        self._vectors = np.random.randn(n, self.dim).astype(np.float32) * 0.01
        self._context_vecs = np.random.randn(n, self.dim).astype(np.float32) * 0.01

        training_pairs = self._build_training_pairs(healthcare_concepts)
        if not training_pairs:
            training_pairs = [(w, w) for w in list(self.vocab.keys())[:100]]

        self._train_negative_sampling(training_pairs)

        for v in [self._vectors, self._context_vecs]:
            if v is not None:
                norms = np.linalg.norm(v, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                v /= norms

        self._trained = True
        logger.info("Skip-gram embedder trained: vocab=%d, dim=%d, epochs=%d, pairs=%d",
                    n, self.dim, self.epochs, len(training_pairs))

    def _build_vocab(self, schema_learner, concepts: Dict):
        tokens = set()

        for concept, info in concepts.items():
            tokens.add(concept)
            for syn in info.get('synonyms', []):
                for word in self._split_words(syn):
                    if len(word) > 1:
                        tokens.add(word)
            for col in info.get('columns', []):
                for word in self._split_words(col):
                    if len(word) > 1:
                        tokens.add(word)

        if schema_learner:
            for table_name, profiles in schema_learner.tables.items():
                for word in self._split_words(table_name):
                    if len(word) > 1:
                        tokens.add(word)
                for p in profiles:
                    for word in self._split_words(p.name):
                        if len(word) > 1:
                            tokens.add(word)
                    for tag in getattr(p, 'semantic_tags', []):
                        for word in self._split_words(tag):
                            if len(word) > 1:
                                tokens.add(word)
                    if p.is_categorical and p.sample_values:
                        for v in p.sample_values[:10]:
                            for word in self._split_words(str(v)):
                                if len(word) > 1:
                                    tokens.add(word)

        for intent, info in INTENT_PATTERNS.items():
            tokens.add(intent)
            for trigger in info.get('triggers', []):
                for word in self._split_words(trigger):
                    if len(word) > 1:
                        tokens.add(word)

        tokens = sorted(t for t in tokens if len(t) > 1)
        self.vocab = {t: i for i, t in enumerate(tokens)}

        for concept, info in concepts.items():
            self._vocab_freq[concept] = 100
            for syn in info.get('synonyms', []):
                for word in self._split_words(syn):
                    if word in self.vocab:
                        self._vocab_freq[word] = self._vocab_freq.get(word, 0) + 5

    def _build_training_pairs(self, concepts: Dict) -> List[Tuple[str, str]]:
        pairs = []

        for concept, info in concepts.items():
            syns = info.get('synonyms', [])
            cols = info.get('columns', [])
            all_rel = [concept] + syns + cols

            for i, w1 in enumerate(all_rel):
                for w2 in all_rel[i+1:]:
                    for word1 in self._split_words(w1):
                        for word2 in self._split_words(w2):
                            if word1 in self.vocab and word2 in self.vocab:
                                pairs.append((word1, word2))
                                pairs.append((word2, word1))

        for intent, info in INTENT_PATTERNS.items():
            triggers = info.get('triggers', [])
            for t1 in triggers:
                for t2 in triggers:
                    if t1 != t2:
                        for w1 in self._split_words(t1):
                            for w2 in self._split_words(t2):
                                if w1 in self.vocab and w2 in self.vocab:
                                    pairs.append((w1, w2))

        return pairs

    def _train_negative_sampling(self, training_pairs: List[Tuple[str, str]]):
        unigram_dist = np.zeros(len(self.vocab), dtype=np.float64)
        for word, idx in self.vocab.items():
            freq = self._vocab_freq.get(word, 1)
            unigram_dist[idx] = freq ** 0.75

        unigram_dist /= unigram_dist.sum()

        learning_rate = 0.025
        min_lr = 0.0001
        lr_decay = (learning_rate - min_lr) / (self.epochs * len(training_pairs))

        neg_table_size = 100000
        neg_table = np.random.choice(len(self.vocab), size=neg_table_size, p=unigram_dist)
        neg_pos = 0

        for epoch in range(self.epochs):
            if epoch > 0 and epoch % 3 == 0:
                logger.debug("Skip-gram epoch %d/%d", epoch, self.epochs)

            for word, context in training_pairs:
                w_idx = self.vocab[word]
                c_idx = self.vocab[context]

                self._update_pair(w_idx, c_idx, 1, learning_rate)

                for _ in range(5):
                    neg_idx = int(neg_table[neg_pos % neg_table_size])
                    neg_pos += 1
                    if neg_idx != c_idx:
                        self._update_pair(w_idx, neg_idx, 0, learning_rate)

            learning_rate = max(min_lr, learning_rate - lr_decay * len(training_pairs))

    def _update_pair(self, word_idx: int, context_idx: int, label: int,
                     learning_rate: float):
        word_vec = self._vectors[word_idx]
        context_vec = self._context_vecs[context_idx]
        dot = float(np.dot(word_vec, context_vec))

        dot = np.clip(dot, -30, 30)
        pred = 1.0 / (1.0 + np.exp(-dot))

        error = (label - pred) * learning_rate

        grad = error * context_vec
        self._vectors[word_idx] += grad

        grad = error * word_vec
        self._context_vecs[context_idx] += grad

    def encode(self, text: str) -> np.ndarray:
        if not self._trained:
            return np.zeros(self.dim, dtype=np.float32)

        words = self._tokenize(text)
        if not words:
            return np.zeros(self.dim, dtype=np.float32)

        vecs = []
        for word in words:
            if word in self.vocab:
                vecs.append(self._vectors[self.vocab[word]])
            else:
                subword_vecs = []
                for char in word:
                    if char in self.vocab:
                        subword_vecs.append(self._vectors[self.vocab[char]])
                if subword_vecs:
                    vecs.append(np.mean(subword_vecs, axis=0))

        if not vecs:
            return np.zeros(self.dim, dtype=np.float32)

        result = np.mean(vecs, axis=0).astype(np.float32)
        norm = float(np.linalg.norm(result))
        if norm > 0:
            result /= norm
        return result

    def similarity(self, text1: str, text2: str) -> float:
        v1 = self.encode(text1)
        v2 = self.encode(text2)
        return float(np.dot(v1, v2))

    def nearest(self, text: str, candidates: List[str],
                top_k: int = 5) -> List[Tuple[str, float]]:
        q_vec = self.encode(text)
        if np.linalg.norm(q_vec) == 0:
            return []

        scored = []
        for cand in candidates:
            c_vec = self.encode(cand)
            sim = float(np.dot(q_vec, c_vec))
            scored.append((cand, sim))

        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = text.lower()
        tokens = re.findall(r'[a-z][a-z0-9_]*', text)
        expanded = []
        for t in tokens:
            expanded.append(t)
            parts = t.split('_')
            if len(parts) > 1:
                expanded.extend(p for p in parts if len(p) > 1)
        return expanded

    @staticmethod
    def _split_words(text: str) -> List[str]:
        text = text.lower()
        words = re.findall(r'[a-z][a-z0-9]*', text)
        return words


class _MultiHeadAttention:

    def __init__(self, embedder: _SkipGramEmbedder, dim: int = 384, heads: int = 4):
        self.embedder = embedder
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

        np.random.seed(42)
        self.W_q = [np.random.randn(dim, dim).astype(np.float32) * 0.01
                    for _ in range(heads)]
        self.W_k = [np.random.randn(dim, dim).astype(np.float32) * 0.01
                    for _ in range(heads)]
        self.W_v = [np.random.randn(dim, dim).astype(np.float32) * 0.01
                    for _ in range(heads)]
        self.W_o = np.random.randn(dim, dim).astype(np.float32) * 0.01

        self._pos_enc = self._build_position_encoding(512)

    def _build_position_encoding(self, max_len: int) -> np.ndarray:
        pe = np.zeros((max_len, self.dim), dtype=np.float32)
        for pos in range(max_len):
            for i in range(0, self.dim, 2):
                pe[pos, i] = np.sin(pos / (10000 ** (i / self.dim)))
                if i + 1 < self.dim:
                    pe[pos, i + 1] = np.cos(pos / (10000 ** ((i + 1) / self.dim)))
        return pe

    def attend(self, question: str, schema_learner) -> Dict[str, float]:
        tokens = self.embedder._tokenize(question)
        if not tokens:
            return {}

        token_vecs = np.array([self.embedder.encode(t) for t in tokens],
                              dtype=np.float32)

        for i in range(len(tokens)):
            if i < len(self._pos_enc):
                token_vecs[i] += self._pos_enc[i]

        norms = np.linalg.norm(token_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        token_vecs = token_vecs / norms

        schema_vecs = self._get_schema_context(schema_learner)

        head_weights = []
        for h in range(self.heads):
            Q = token_vecs @ self.W_q[h]
            K = schema_vecs @ self.W_k[h] if len(schema_vecs) > 0 else np.zeros(
                (len(tokens), self.dim), dtype=np.float32)
            V = schema_vecs @ self.W_v[h] if len(schema_vecs) > 0 else np.zeros(
                (len(tokens), self.dim), dtype=np.float32)

            scores = Q @ K.T / np.sqrt(self.head_dim + 1e-6)
            scores = np.clip(scores, -10, 10)
            attn_weights = self._softmax(scores[0])
            head_weights.append(attn_weights)

        avg_weights = np.mean(head_weights, axis=0)
        return {t: float(w) for t, w in zip(tokens, avg_weights)}

    def _get_schema_context(self, schema_learner) -> np.ndarray:
        if not schema_learner:
            return np.zeros((1, self.dim), dtype=np.float32)

        vecs = []
        for table_name, profiles in schema_learner.tables.items():
            vecs.append(self.embedder.encode(table_name))
            for p in profiles:
                vecs.append(self.embedder.encode(p.name))

        if not vecs:
            return np.zeros((1, self.dim), dtype=np.float32)

        return np.array(vecs, dtype=np.float32)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        exp_x = np.exp(np.clip(x, -30, 30))
        return exp_x / (exp_x.sum() + 1e-8)


class _SemanticRoleLabeler:

    def extract(self, question: str, embedder: _SkipGramEmbedder,
                healthcare_concepts: Dict) -> Dict[str, List[str]]:
        roles = {
            'MEASURE': [],
            'ENTITY': [],
            'DIMENSION': [],
            'FILTER': [],
            'TEMPORAL': [],
            'COMPARISON': [],
            'LIMIT': [],
        }

        q_lower = question.lower()

        for concept, info in healthcare_concepts.items():
            info_type = info.get('type', '').upper()
            if info_type in ('METRIC', 'RATE', 'DERIVED_METRIC'):
                for syn in info.get('synonyms', []):
                    if _phrase_in_text(syn, q_lower):
                        roles['MEASURE'].append(concept)
                        break

        for concept, info in healthcare_concepts.items():
            if info.get('type', '').upper() in ('ENTITY', 'DIMENSION'):
                for syn in info.get('synonyms', []):
                    if _phrase_in_text(syn, q_lower):
                        roles['ENTITY'].append(concept)
                        break

        for concept, info in healthcare_concepts.items():
            if info.get('type', '').upper() in ('DIMENSION', 'TIME_DIMENSION'):
                for syn in info.get('synonyms', []):
                    if 'by ' + syn in q_lower or ' per ' + syn in q_lower:
                        roles['DIMENSION'].append(concept)
                        break

        filter_patterns = [
            (r'(denied|denied claims|claim denied)', ['denial']),
            (r'(age|aged|older) (?:than|>|over) (\d+)', ['age_filter']),
            (r'(high|higher|highest) (cost|spend|spending)', ['high_cost']),
            (r'(low|lower|lowest) (cost|spend|spending)', ['low_cost']),
            (r'(in|from|with) (california|ca|ny|florida)', ['geography_filter']),
            (r'(commercial|medicare|medicaid) (members|plans)', ['plan_filter']),
        ]

        for pattern, labels in filter_patterns:
            if re.search(pattern, q_lower, re.IGNORECASE):
                roles['FILTER'].extend(labels)

        temporal_patterns = [
            (r'last (year|month|quarter)', 'past_period'),
            (r'(this|current) (year|month|quarter)', 'current_period'),
            (r'(january|february|march|april|may|june|july|august|'
             r'september|october|november|december)', 'month'),
            (r'(\d{4})', 'year'),
            (r'(ytd|mtd|rolling|trend|over time)', 'period_comparison'),
        ]

        for pattern, label in temporal_patterns:
            if re.search(pattern, q_lower, re.IGNORECASE):
                roles['TEMPORAL'].append(label)

        if re.search(r'(compare|versus|vs|difference|better|worse|relative)',
                     q_lower, re.IGNORECASE):
            roles['COMPARISON'].append('comparative')

        limit_match = re.search(r'(top|bottom|first|last|limit) (\d+)', q_lower)
        if limit_match:
            roles['LIMIT'].append(f"{limit_match.group(1).lower()} {limit_match.group(2)}")

        return {k: list(set(v)) for k, v in roles.items() if v}


class _ConceptGraph:

    def __init__(self):
        self.edges: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)

    def add_edge(self, source: str, target: str, relation: str, weight: float = 1.0):
        self.edges[source.lower()].append((target.lower(), relation, weight))

    def build(self, schema_learner, healthcare_concepts: Dict):
        for concept, info in healthcare_concepts.items():
            for col in info.get('columns', []):
                self.add_edge(concept, col, 'maps_to_column', 0.95)
                self.add_edge(col, concept, 'represents', 0.95)
            for syn in info.get('synonyms', []):
                self.add_edge(syn, concept, 'synonym_of', 0.98)

        if schema_learner:
            for table_name, profiles in schema_learner.tables.items():
                for p in profiles:
                    self.add_edge(table_name, p.name, 'has_column', 0.99)
                    self.add_edge(p.name, table_name, 'belongs_to', 0.99)

            join_graph = getattr(schema_learner, 'join_graph', None)
            if join_graph:
                for t1, targets in join_graph.items():
                    for t2, join_col in targets.items():
                        self.add_edge(t1, t2, f'joins_via_{join_col}', 0.85)
                        self.add_edge(t2, t1, f'joins_via_{join_col}', 0.85)

        logger.info("Concept graph: %d nodes", len(self.edges))

    def traverse(self, start_nodes: List[str], max_hops: int = 3,
                 min_weight: float = 0.3) -> Dict[str, float]:
        visited: Dict[str, float] = {}
        queue: List[Tuple[str, float, int]] = [(n.lower(), 1.0, 0)
                                               for n in start_nodes]

        while queue:
            node, score, hops = queue.pop(0)
            if hops > max_hops:
                continue
            if node in visited and visited[node] >= score:
                continue
            visited[node] = score

            for target, relation, weight in self.edges.get(node, []):
                new_score = score * weight * (0.8 ** hops)
                if new_score >= min_weight:
                    queue.append((target, new_score, hops + 1))

        return visited


class _QueryMemory:

    def __init__(self, embedder: _SkipGramEmbedder, data_dir: str):
        self._embedder = embedder
        self._db_path = os.path.join(data_dir, 'query_memory.db')
        self._memories: List[Dict] = []
        self._vectors: Optional[np.ndarray] = None
        self._init_db()
        self._load()

    def _init_db(self):
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS query_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    intent TEXT,
                    tables_used TEXT,
                    columns_used TEXT,
                    sql TEXT,
                    success INTEGER DEFAULT 1,
                    confidence REAL DEFAULT 0.0,
                    timestamp REAL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS ix_success
                ON query_memory(success)
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning("Query memory DB init failed: %s", e)

    def _load(self):
        try:
            conn = sqlite3.connect(self._db_path)
            rows = conn.execute(
                "SELECT question, intent, tables_used, columns_used, sql "
                "FROM query_memory WHERE success=1 ORDER BY timestamp DESC LIMIT 10000"
            ).fetchall()
            conn.close()

            self._memories = []
            for q, intent, tables, cols, sql in rows:
                self._memories.append({
                    'question': q,
                    'intent': intent,
                    'tables': json.loads(tables) if tables else [],
                    'columns': json.loads(cols) if cols else [],
                    'sql': sql,
                })

            if self._memories and self._embedder._trained:
                vecs = [self._embedder.encode(m['question']) for m in self._memories]
                self._vectors = np.array(vecs, dtype=np.float32)
            logger.info("Query memory loaded: %d patterns", len(self._memories))
        except Exception as e:
            logger.warning("Query memory load failed: %s", e)

    def store(self, question: str, intent: str, tables: List[str],
              columns: List[str], sql: str, success: bool):
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute(
                "INSERT INTO query_memory "
                "(question, intent, tables_used, columns_used, sql, success, timestamp) "
                "VALUES (?,?,?,?,?,?,?)",
                (question, intent, json.dumps(tables), json.dumps(columns), sql,
                 int(success), time.time())
            )
            conn.commit()
            conn.close()

            if success:
                self._memories.append({
                    'question': question,
                    'intent': intent,
                    'tables': tables,
                    'columns': columns,
                    'sql': sql,
                })
                if self._embedder._trained:
                    new_vec = self._embedder.encode(question).reshape(1, -1)
                    if self._vectors is not None:
                        self._vectors = np.vstack([self._vectors, new_vec])
                    else:
                        self._vectors = new_vec
        except Exception as e:
            logger.debug("Query memory store failed: %s", e)

    def recall(self, question: str, top_k: int = 3,
               min_sim: float = 0.65) -> List[Tuple[Dict, float]]:
        if not self._memories or self._vectors is None:
            return []

        q_vec = self._embedder.encode(question)
        if np.linalg.norm(q_vec) == 0:
            return []

        sims = self._vectors @ q_vec
        top_idx = np.argsort(-sims)[:top_k]

        results = []
        for idx in top_idx:
            sim = float(sims[idx])
            if sim >= min_sim:
                results.append((self._memories[idx], sim))

        return results


class DeepUnderstandingEngine:

    def __init__(self, schema_learner=None, db_path: str = '', data_dir: str = ''):
        self._schema = schema_learner
        self._db_path = db_path
        self._data_dir = data_dir or os.path.dirname(db_path)

        t0 = time.time()

        self._embedder = _SkipGramEmbedder(dim=384, epochs=8)
        self._embedder.train(schema_learner, HEALTHCARE_CONCEPTS)

        self._attention = _MultiHeadAttention(self._embedder, dim=384, heads=4)

        self._srl = _SemanticRoleLabeler()

        self._graph = _ConceptGraph()
        self._graph.build(schema_learner, HEALTHCARE_CONCEPTS)

        self._memory = _QueryMemory(self._embedder, self._data_dir)

        self._table_names = list(schema_learner.tables.keys()) if schema_learner else []
        self._all_columns = {}
        if schema_learner:
            for table, profiles in schema_learner.tables.items():
                for p in profiles:
                    self._all_columns[p.name.lower()] = (table, p)

        build_time = (time.time() - t0) * 1000
        logger.info(
            "Deep understanding engine built in %.0fms "
            "(vocab=%d, concepts=%d, memories=%d)",
            build_time, self._embedder.size, len(HEALTHCARE_CONCEPTS),
            len(self._memory._memories)
        )

    @property
    def vocab_size(self) -> int:
        return self._embedder.size

    @property
    def embedding_dim(self) -> int:
        return self._embedder.dim

    @property
    def concept_count(self) -> int:
        return len(HEALTHCARE_CONCEPTS)

    def understand(self, question: str) -> DeepUnderstanding:
        t0 = time.time()
        result = DeepUnderstanding()

        attention = self._attention.attend(question, self._schema)
        result.attention_weights = attention
        result.reasoning_chain.append(
            f"Attention: {sorted(attention.items(), key=lambda x: -x[1])[:5]}"
        )

        self._classify_intent(question, result)

        roles = self._srl.extract(question, self._embedder, HEALTHCARE_CONCEPTS)
        result.reasoning_chain.append(f"Semantic roles: {roles}")

        self._detect_concepts(question, result, roles)

        self._resolve_schema(question, result, attention)

        self._extract_filters(question, result, roles)

        self._match_history(question, result)

        self._generate_sql_hints(question, result, roles)

        result.latency_ms = round((time.time() - t0) * 1000)
        return result

    def _classify_intent(self, question: str, result: DeepUnderstanding):
        q_lower = question.lower()
        scores: Dict[str, float] = {}

        for intent, info in INTENT_PATTERNS.items():
            kw_score = 0.0
            for trigger in info.get('triggers', []):
                if _phrase_in_text(trigger, q_lower):
                    kw_score = max(kw_score, 0.5 + 0.05 * len(trigger.split()))

            sem_score = max(
                (self._embedder.similarity(question, trigger)
                 for trigger in info.get('triggers', [])),
                default=0.0
            )

            combined = kw_score * 0.35 + sem_score * 0.65
            if combined > 0.15:
                scores[intent] = combined

        if scores:
            sorted_intents = sorted(scores.items(), key=lambda x: -x[1])
            result.intent = sorted_intents[0][0]
            result.intent_confidence = min(sorted_intents[0][1], 0.99)
            result.secondary_intents = [i for i, s in sorted_intents[1:4]
                                        if s > 0.25]
            result.reasoning_chain.append(
                f"Intent: {result.intent} ({result.intent_confidence:.2f})"
            )
        else:
            result.intent = 'detail'
            result.intent_confidence = 0.2

    def _detect_concepts(self, question: str, result: DeepUnderstanding,
                         roles: Dict[str, List[str]]):
        q_lower = question.lower()
        concept_scores: Dict[str, float] = {}

        for concept, info in HEALTHCARE_CONCEPTS.items():
            for syn in info.get('synonyms', []):
                if _phrase_in_text(syn, q_lower):
                    score = 0.7 + 0.05 * len(syn.split())
                    concept_scores[concept] = max(concept_scores.get(concept, 0),
                                                  score)
                    result.entities.append(Entity(
                        text=syn,
                        type=info.get('type', 'CONCEPT'),
                        resolved_to=concept,
                        confidence=0.95,
                        source='synonym_match',
                    ))
                    break

            sim = self._embedder.similarity(
                question,
                ' '.join(info.get('synonyms', [])[:3])
            )
            if sim > 0.55:
                concept_scores[concept] = max(concept_scores.get(concept, 0), sim)

        column_candidates: Dict[str, float] = {}
        if concept_scores:
            reachable = self._graph.traverse(list(concept_scores.keys()), max_hops=2)
            for node, score in reachable.items():
                if node in self._all_columns and score > 0.50:
                    column_candidates[node] = score

        table_scores: Dict[str, float] = {}
        table_columns: Dict[str, List[str]] = {}
        for col_name, score in column_candidates.items():
            table, profile = self._all_columns[col_name]
            table_scores[table] = table_scores.get(table, 0) + score
            table_columns.setdefault(table, []).append(profile.name)

        top_tables = sorted(table_scores.items(), key=lambda x: -x[1])[:2]
        for tbl, _ in top_tables:
            if tbl not in result.target_tables:
                result.target_tables.append(tbl)
            for col in table_columns.get(tbl, []):
                if col not in result.target_columns:
                    result.target_columns.append(col)

        result.reasoning_chain.append(
            f"Concepts: {list(concept_scores.keys())}"
        )

    @staticmethod
    def _edit_distance(a: str, b: str) -> int:
        if len(a) < len(b):
            return DeepUnderstandingEngine._edit_distance(b, a)
        if not b:
            return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a):
            curr = [i + 1]
            for j, cb in enumerate(b):
                curr.append(min(prev[j + 1] + 1, curr[j] + 1,
                                prev[j] + (0 if ca == cb else 1)))
            prev = curr
        return prev[-1]

    def _correct_typos(self, question: str) -> str:
        CORRECTIONS = {
            'clams': 'claims', 'clam': 'claims', 'calims': 'claims',
            'cliams': 'claims', 'ckaims': 'claims',
            'deniel': 'denial', 'deniels': 'denials', 'denisl': 'denial',
            'membrs': 'members', 'memebers': 'members', 'memebrs': 'members',
            'presciption': 'prescription', 'prescrption': 'prescription',
            'perscription': 'prescription', 'presciptions': 'prescriptions',
            'averge': 'average', 'avrage': 'average', 'averag': 'average',
            'encountr': 'encounter', 'encoutner': 'encounter',
            'encountrs': 'encounters', 'encountes': 'encounters',
            'provders': 'providers', 'providrs': 'providers', 'proviers': 'providers',
            'referall': 'referral', 'refferal': 'referral', 'referals': 'referrals',
            'diagnoseis': 'diagnoses', 'dignoses': 'diagnoses', 'diagnossi': 'diagnoses',
            'appoitment': 'appointment', 'appointmnet': 'appointment',
            'appoitments': 'appointments',
            'utilzation': 'utilization', 'utlization': 'utilization',
            'performace': 'performance', 'preformance': 'performance',
            'readmision': 'readmission', 'readmisson': 'readmission',
            'specilty': 'specialty', 'speciality': 'specialty',
            'facilty': 'facility', 'faclity': 'facility',
        }
        words = question.split()
        corrected = []
        for word in words:
            lower = word.lower().strip('.,?!;:')
            if lower in CORRECTIONS:
                corrected.append(CORRECTIONS[lower])
            else:
                best_match = None
                best_dist = 3
                for correct_term in ['claims', 'members', 'encounters', 'providers',
                                      'prescriptions', 'referrals', 'diagnoses',
                                      'appointments', 'denial', 'facility', 'specialty',
                                      'department', 'medication', 'chronic', 'cost',
                                      'trend', 'breakdown', 'region', 'plan']:
                    dist = self._edit_distance(lower, correct_term)
                    if dist < best_dist and dist <= 2 and len(lower) > 3:
                        best_dist = dist
                        best_match = correct_term
                if best_match:
                    corrected.append(best_match)
                else:
                    corrected.append(word)
        return ' '.join(corrected)

    def _resolve_schema(self, question: str, result: DeepUnderstanding,
                       attention: Dict[str, float]):
        q_lower = self._correct_typos(question).lower()

        TABLE_ALIASES = {
            'claims': 'claims', 'claim': 'claims', 'denial': 'claims', 'denied': 'claims',
            'billed': 'claims', 'paid': 'claims', 'cost': 'claims', 'costs': 'claims',
            'reimbursement': 'claims', 'revenue': 'claims',
            'members': 'members', 'member': 'members', 'patient': 'members',
            'patients': 'members', 'enrolled': 'members', 'enrollment': 'members',
            'risk score': 'members', 'demographic': 'members',
            'encounters': 'encounters', 'encounter': 'encounters', 'visit': 'encounters',
            'visits': 'encounters', 'admission': 'encounters', 'admissions': 'encounters',
            'inpatient': 'encounters', 'outpatient': 'encounters', 'er': 'encounters',
            'emergency': 'encounters', 'length of stay': 'encounters', 'los': 'encounters',
            'diagnoses': 'diagnoses', 'diagnosis': 'diagnoses', 'icd': 'diagnoses',
            'chronic': 'diagnoses', 'condition': 'diagnoses', 'conditions': 'diagnoses',
            'prescriptions': 'prescriptions', 'prescription': 'prescriptions',
            'medication': 'prescriptions', 'medications': 'prescriptions',
            'drug': 'prescriptions', 'drugs': 'prescriptions', 'rx': 'prescriptions',
            'pharmacy': 'prescriptions',
            'providers': 'providers', 'provider': 'providers', 'doctor': 'providers',
            'doctors': 'providers', 'physician': 'providers', 'npi': 'providers',
            'specialty': 'providers',
            'appointments': 'appointments', 'appointment': 'appointments',
            'no-show': 'appointments', 'noshow': 'appointments', 'schedule': 'appointments',
            'referrals': 'referrals', 'referral': 'referrals',
            'cpt_codes': 'cpt_codes', 'cpt': 'cpt_codes', 'procedure': 'cpt_codes',
            'procedures': 'cpt_codes',
            'facility': 'claims', 'facilities': 'claims',
            'adjudicate': 'claims', 'adjudication': 'claims', 'processing time': 'claims',
        }
        direct_tables = []
        for alias, table in TABLE_ALIASES.items():
            if _phrase_in_text(alias, q_lower) and table not in direct_tables:
                direct_tables.append(table)

        if direct_tables:
            result.target_tables = direct_tables[:3]
        elif not result.target_tables:
            table_scores: Dict[str, float] = {}
            for token, weight in sorted(attention.items(),
                                        key=lambda x: -x[1])[:5]:
                if weight < 0.08:
                    continue
                matches = self._embedder.nearest(
                    token, list(self._all_columns.keys()), top_k=1)
                for col_name, sim in matches:
                    if sim > 0.60:
                        table, _ = self._all_columns[col_name]
                        table_scores[table] = table_scores.get(table, 0) + weight * sim
            if table_scores:
                best = sorted(table_scores.items(), key=lambda x: -x[1])[:2]
                result.target_tables = [t for t, _ in best]
            else:
                result.target_tables = ['claims']

        result.target_columns = []
        result.metric_columns = []
        result.dimension_columns = []

        needs_metric = result.intent in ('aggregate', 'count', 'rate', 'trend',
                                          'ranking', 'comparison', 'breakdown',
                                          'executive', 'root_cause')
        needs_dimension = result.intent in ('breakdown', 'trend', 'comparison',
                                             'ranking')

        relevant_cols: List[Tuple[str, str, float]] = []
        for token, weight in sorted(attention.items(), key=lambda x: -x[1])[:6]:
            if weight < 0.05:
                continue
            for col_lower, (table, profile) in self._all_columns.items():
                if table not in result.target_tables:
                    continue
                if token in col_lower or col_lower in token:
                    relevant_cols.append((profile.name, table, weight * 1.5))
                elif any(kw in col_lower for kw in token.split('_')):
                    relevant_cols.append((profile.name, table, weight))

        COL_HINTS = {
            'cost': ['PAID_AMOUNT', 'BILLED_AMOUNT', 'ALLOWED_AMOUNT', 'COST'],
            'denial': ['CLAIM_STATUS'], 'denied': ['CLAIM_STATUS'],
            'region': ['KP_REGION'], 'plan': ['PLAN_TYPE'],
            'status': ['CLAIM_STATUS', 'STATUS'], 'gender': ['GENDER'],
            'age': ['DATE_OF_BIRTH'], 'state': ['STATE'],
            'specialty': ['SPECIALTY'], 'department': ['DEPARTMENT'],
            'visit type': ['VISIT_TYPE'], 'medication': ['MEDICATION_NAME', 'MEDICATION_CLASS'],
            'severity': ['SEVERITY'], 'risk': ['RISK_SCORE'],
            'pmpm': ['PAID_AMOUNT', 'MEMBER_ID', 'SERVICE_DATE'],
            'trend': ['SERVICE_DATE', 'ENCOUNTER_DATE', 'PRESCRIPTION_DATE'],
            'monthly': ['SERVICE_DATE', 'ENCOUNTER_DATE'],
            'facility': ['FACILITY'], 'facilities': ['FACILITY'],
            'telehealth': ['VISIT_TYPE'], 'inpatient': ['VISIT_TYPE'],
            'outpatient': ['VISIT_TYPE'], 'emergency': ['VISIT_TYPE'],
            'denial reason': ['DENIAL_REASON'], 'chronic': ['CHRONIC_CONDITIONS'],
            'adjudicate': ['ADJUDICATED_DATE', 'SUBMITTED_DATE'],
            'processing': ['ADJUDICATED_DATE', 'SUBMITTED_DATE'],
        }
        for keyword, cols in COL_HINTS.items():
            if _phrase_in_text(keyword, q_lower):
                for col in cols:
                    for tbl in result.target_tables:
                        if col.lower() in self._all_columns:
                            _, profile = self._all_columns[col.lower()]
                            relevant_cols.append((col, tbl, 0.8))

        seen_cols = set()
        for col_name, table, score in sorted(relevant_cols, key=lambda x: -x[2]):
            if col_name in seen_cols:
                continue
            seen_cols.add(col_name)
            col_lower = col_name.lower()
            if col_lower not in self._all_columns:
                continue
            _, profile = self._all_columns[col_lower]
            if profile.name not in result.target_columns:
                result.target_columns.append(profile.name)

            if profile.is_numeric and not profile.is_id and needs_metric:
                agg = 'SUM'
                if 'count' in q_lower or 'how many' in q_lower:
                    agg = 'COUNT'
                elif 'average' in q_lower or 'avg' in q_lower or 'mean' in q_lower:
                    agg = 'AVG'
                elif 'max' in q_lower or 'highest' in q_lower or 'top' in q_lower:
                    agg = 'MAX'
                elif 'min' in q_lower or 'lowest' in q_lower or 'bottom' in q_lower:
                    agg = 'MIN'
                if len(result.metric_columns) < 3:
                    result.metric_columns.append((profile.name, agg))
            elif (profile.is_categorical or profile.is_id) and needs_dimension:
                if len(result.dimension_columns) < 2:
                    result.dimension_columns.append(profile.name)

        if needs_metric and not result.metric_columns:
            primary_table = result.target_tables[0]
            default_metrics = {
                'claims': [('PAID_AMOUNT', 'SUM')],
                'encounters': [('LENGTH_OF_STAY', 'AVG')],
                'members': [('RISK_SCORE', 'AVG')],
                'prescriptions': [('COST', 'SUM')],
                'appointments': [('APPOINTMENT_ID', 'COUNT')],
                'referrals': [('REFERRAL_ID', 'COUNT')],
                'diagnoses': [('DIAGNOSIS_ID', 'COUNT')],
                'providers': [('NPI', 'COUNT')],
            }
            result.metric_columns = default_metrics.get(primary_table,
                                                         [('*', 'COUNT')])

        if len(result.target_tables) > 1 and self._schema:
            join_graph = getattr(self._schema, 'join_graph', None)
            if join_graph:
                for t1, targets in join_graph.items():
                    for t2, join_col in targets.items():
                        if t1 in result.target_tables and t2 in result.target_tables:
                            if '=' in str(join_col):
                                c1, c2 = str(join_col).split('=', 1)
                            else:
                                c1 = c2 = str(join_col)
                            if (t1, c1, t2, c2) not in result.join_path:
                                result.join_path.append((t1, c1, t2, c2))

        result.reasoning_chain.append(
            f"Schema: {result.target_tables}, "
            f"metrics={result.metric_columns}, dims={result.dimension_columns}"
        )

    def _extract_filters(self, question: str, result: DeepUnderstanding,
                        roles: Dict[str, List[str]]):
        q_lower = question.lower()

        if _phrase_in_text('denied', q_lower):
            result.filter_conditions.append({
                'column': 'claim_status',
                'operator': 'IN',
                'values': ['DENIED', 'REJECTED'],
                'confidence': 0.95,
            })

        age_match = re.search(r'(age|aged|older) (?:than|>|over) (\d+)',
                              q_lower, re.IGNORECASE)
        if age_match:
            result.filter_conditions.append({
                'column': 'age',
                'operator': '>',
                'values': [int(age_match.group(2))],
                'confidence': 0.90,
            })

        state_match = re.search(r'\b(california|ca|ny|florida|tx|tx)\b',
                                q_lower, re.IGNORECASE)
        if state_match:
            result.filter_conditions.append({
                'column': 'state',
                'operator': '=',
                'values': [state_match.group(1).upper()],
                'confidence': 0.90,
            })

        for plan in ['commercial', 'medicare', 'medicaid']:
            if _phrase_in_text(plan, q_lower):
                result.filter_conditions.append({
                    'column': 'plan_type',
                    'operator': '=',
                    'values': [plan.upper()],
                    'confidence': 0.85,
                })
                break

    def _match_history(self, question: str, result: DeepUnderstanding):
        matches = self._memory.recall(question, top_k=2)
        if matches:
            best, score = matches[0]
            result.history_match_score = score
            result.reasoning_chain.append(
                f"History match: {score:.2f}"
            )
            if score > 0.78:
                for table in best.get('tables', []):
                    if table not in result.target_tables:
                        result.target_tables.append(table)

    def _generate_sql_hints(self, question: str, result: DeepUnderstanding,
                           roles: Dict[str, List[str]]):
        intent = result.intent
        pattern_info = INTENT_PATTERNS.get(intent, {})

        if intent == 'rate':
            result.sql_approach = (
                "COUNT(filter condition) * 100.0 / COUNT(*)"
            )
        elif intent == 'trend':
            result.sql_approach = (
                "GROUP BY date_trunc(), ORDER BY period"
            )
        elif intent == 'ranking':
            result.sql_approach = (
                "GROUP BY dimension, ORDER BY metric DESC, LIMIT N"
            )
        elif intent == 'comparison':
            result.sql_approach = (
                "GROUP BY comparison dimension, aggregate metrics"
            )
        elif intent == 'breakdown':
            result.sql_approach = (
                "GROUP BY " + (", ".join(result.dimension_columns) or "dimension")
            )
        else:
            result.sql_approach = pattern_info.get('sql_shape',
                                                   'SELECT with WHERE/GROUP BY')

    def learn(self, question: str, intent: str, tables: List[str],
              columns: List[str], sql: str, success: bool):
        self._memory.store(question, intent, tables, columns, sql, success)
