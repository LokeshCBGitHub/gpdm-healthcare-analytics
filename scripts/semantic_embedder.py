import math
import hashlib
import logging
import os
import json
import struct
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field

logger = logging.getLogger('gpdm.semantic_embedder')


EMBED_DIM = 128
N_GRAM_RANGE = (2, 5)
HASH_BUCKETS = 4096
N_ATTENTION_HEADS = 4
CONTEXT_WINDOW = 8


CONCEPT_CLUSTERS = {
    'diagnosis': ['diagnosis', 'diagnoses', 'icd10', 'icd', 'condition', 'disease',
                  'chronic', 'acute', 'comorbidity', 'hcc', 'severity'],
    'provider': ['provider', 'doctor', 'physician', 'clinician', 'specialist',
                 'npi', 'rendering', 'referring', 'prescribing', 'attending',
                 'surgeon', 'pcp', 'primary care'],
    'member': ['member', 'patient', 'enrollee', 'beneficiary', 'subscriber',
               'mrn', 'member_id', 'demographic', 'population'],
    'claim': ['claim', 'claims', 'billing', 'reimbursement', 'adjudication',
              'submission', 'processing', 'cpt', 'hcpcs', 'drg'],
    'encounter': ['encounter', 'visit', 'admission', 'discharge', 'inpatient',
                  'outpatient', 'emergency', 'ed', 'office visit', 'telehealth'],
    'prescription': ['prescription', 'medication', 'drug', 'pharmacy', 'rx',
                     'formulary', 'ndc', 'days supply', 'refill', 'generic',
                     'brand', 'therapeutic', 'medication class'],

    'cost': ['cost', 'amount', 'paid', 'billed', 'allowed', 'charge',
             'expense', 'spending', 'expenditure', 'reimbursement'],
    'copay': ['copay', 'co-pay', 'copayment', 'cost share', 'member responsibility',
              'out of pocket', 'oop'],
    'deductible': ['deductible', 'out of pocket max', 'annual deductible',
                   'plan deductible', 'family deductible'],
    'coinsurance': ['coinsurance', 'co-insurance', 'cost sharing',
                    'member share', 'plan share'],
    'denial': ['denial', 'denied', 'rejected', 'not approved', 'adverse',
               'denial reason', 'denial rate', 'rejection rate', 'appeal'],

    'aggregate': ['total', 'sum', 'average', 'mean', 'median', 'count',
                  'minimum', 'maximum', 'aggregate', 'statistics'],
    'rate': ['rate', 'percentage', 'percent', 'ratio', 'proportion',
             'frequency', 'incidence', 'prevalence'],
    'trend': ['trend', 'over time', 'monthly', 'quarterly', 'yearly',
              'growth', 'decline', 'trajectory', 'forecast'],
    'rank': ['top', 'highest', 'lowest', 'most', 'least', 'best', 'worst',
             'ranking', 'leading', 'outlier'],
    'compare': ['compare', 'versus', 'vs', 'difference', 'comparison',
                'relative', 'benchmark', 'against'],
    'pmpm': ['pmpm', 'per member per month', 'cost per member', 'utilization rate',
             'per capita', 'per enrollee', 'member months'],

    'region': ['region', 'kp_region', 'area', 'geography', 'location',
               'state', 'county', 'zip', 'market', 'service area'],
    'demographic': ['age', 'gender', 'race', 'ethnicity', 'language',
                    'socioeconomic', 'income', 'education', 'age group'],

    'quality': ['quality', 'hedis', 'stars', 'measure', 'metric', 'kpi',
                'performance', 'benchmark', 'gap', 'compliance'],
    'risk': ['risk', 'risk score', 'raf', 'hcc', 'risk adjustment',
             'stratification', 'acuity', 'severity', 'complexity'],
}

COLUMN_SEMANTICS = {
    'PAID_AMOUNT': ['cost', 'paid amount', 'reimbursement', 'payment'],
    'BILLED_AMOUNT': ['billed', 'charges', 'submitted amount'],
    'ALLOWED_AMOUNT': ['allowed', 'contracted rate', 'negotiated rate'],
    'COPAY': ['copay', 'copayment', 'member cost share'],
    'COINSURANCE': ['coinsurance', 'cost sharing percentage'],
    'DEDUCTIBLE': ['deductible', 'annual deductible'],
    'MEMBER_RESPONSIBILITY': ['member responsibility', 'out of pocket', 'oop'],
    'CLAIM_STATUS': ['status', 'claim status', 'approved denied'],
    'RENDERING_NPI': ['provider', 'rendering provider', 'treating doctor'],
    'MEMBER_ID': ['member', 'patient', 'enrollee'],
    'ICD10_CODE': ['diagnosis code', 'icd10', 'condition code'],
    'ICD10_DESCRIPTION': ['diagnosis', 'condition', 'disease name'],
    'CPT_CODE': ['procedure code', 'cpt', 'service code'],
    'CPT_DESCRIPTION': ['procedure', 'service', 'treatment'],
    'MEDICATION_NAME': ['medication', 'drug', 'prescription drug name'],
    'MEDICATION_CLASS': ['drug class', 'therapeutic class', 'drug category'],
    'KP_REGION': ['region', 'service area', 'geography', 'location'],
    'RISK_SCORE': ['risk score', 'risk level', 'acuity', 'hcc score'],
    'LENGTH_OF_STAY': ['length of stay', 'los', 'days admitted', 'stay duration'],
    'SPECIALTY': ['specialty', 'medical specialty', 'doctor type'],
    'PLAN_TYPE': ['plan type', 'insurance plan', 'coverage type'],
    'VISIT_TYPE': ['visit type', 'encounter type', 'service setting'],
    'NPI': ['npi', 'national provider identifier', 'provider id'],
    'PANEL_SIZE': ['panel size', 'patient panel', 'attributed members'],
    'CLAIM_TYPE': ['claim type', 'service category'],
    'DENIAL_REASON': ['denial reason', 'rejection cause', 'denial code'],
    'FACILITY': ['facility', 'hospital', 'clinic', 'location'],
}


def _stable_hash(s: str, buckets: int) -> int:
    h = 2166136261
    for c in s.encode('utf-8'):
        h ^= c
        h = (h * 16777619) & 0xFFFFFFFF
    return h % buckets


def _char_ngrams(text: str, lo: int = 2, hi: int = 5) -> List[str]:
    text = f'<{text.lower()}>'
    ngrams = []
    for n in range(lo, hi + 1):
        for i in range(len(text) - n + 1):
            ngrams.append(text[i:i + n])
    return ngrams


class EmbeddingVector:

    __slots__ = ('data', 'dim')

    def __init__(self, data: List[float]):
        self.data = data
        self.dim = len(data)

    def cosine(self, other: 'EmbeddingVector') -> float:
        if self.dim != other.dim:
            return 0.0
        dot = sum(a * b for a, b in zip(self.data, other.data))
        na = math.sqrt(sum(a * a for a in self.data))
        nb = math.sqrt(sum(b * b for b in other.data))
        if na < 1e-10 or nb < 1e-10:
            return 0.0
        return dot / (na * nb)

    def __add__(self, other: 'EmbeddingVector') -> 'EmbeddingVector':
        return EmbeddingVector([a + b for a, b in zip(self.data, other.data)])

    def scale(self, factor: float) -> 'EmbeddingVector':
        return EmbeddingVector([a * factor for a in self.data])

    def normalize(self) -> 'EmbeddingVector':
        norm = math.sqrt(sum(a * a for a in self.data))
        if norm < 1e-10:
            return self
        return EmbeddingVector([a / norm for a in self.data])


class SemanticEmbedder:

    def __init__(self, embed_dim: int = EMBED_DIM):
        self.embed_dim = embed_dim
        self._projection = None
        self._cluster_centers = {}
        self._column_embeddings = {}
        self._table_embeddings = {}
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return

        self._projection = self._build_projection_matrix()

        for cluster_name, terms in CONCEPT_CLUSTERS.items():
            vecs = [self._embed_single(t) for t in terms]
            centroid = self._average_vectors(vecs)
            self._cluster_centers[cluster_name] = centroid

        for col_name, descriptions in COLUMN_SEMANTICS.items():
            col_vec = self._embed_single(col_name.lower().replace('_', ' '))
            desc_vecs = [self._embed_single(d) for d in descriptions]
            all_vecs = [col_vec] + desc_vecs
            self._column_embeddings[col_name] = self._average_vectors(all_vecs)

        self._initialized = True
        logger.info("SemanticEmbedder initialized: %d clusters, %d column embeddings",
                    len(self._cluster_centers), len(self._column_embeddings))

    def _build_projection_matrix(self) -> List[List[float]]:
        matrix = []
        for i in range(HASH_BUCKETS):
            row = []
            for j in range(self.embed_dim):
                seed = _stable_hash(f"proj_{i}_{j}", 2**31)
                val = ((seed / 2**31) - 0.5) * 0.1
                row.append(val)
            matrix.append(row)

        cluster_directions = {}
        n_clusters = len(CONCEPT_CLUSTERS)
        for idx, (cluster_name, terms) in enumerate(CONCEPT_CLUSTERS.items()):
            direction = [0.0] * self.embed_dim
            base_dims = []
            for k in range(3):
                dim_idx = (idx * 3 + k) % self.embed_dim
                base_dims.append(dim_idx)
                direction[dim_idx] = 1.0 / math.sqrt(3)

            cluster_directions[cluster_name] = direction

            for term in terms:
                ngrams = _char_ngrams(term.lower())
                for ng in ngrams:
                    bucket = _stable_hash(ng, HASH_BUCKETS)
                    for dim_idx in base_dims:
                        matrix[bucket][dim_idx] += 0.05 / max(len(ngrams), 1)

        return matrix

    def _embed_single(self, text: str) -> EmbeddingVector:
        if not self._projection:
            self.initialize()

        text = text.lower().strip()
        if not text:
            return EmbeddingVector([0.0] * self.embed_dim)

        ngrams = _char_ngrams(text)
        if not ngrams:
            return EmbeddingVector([0.0] * self.embed_dim)

        embedding = [0.0] * self.embed_dim
        for ng in ngrams:
            bucket = _stable_hash(ng, HASH_BUCKETS)
            row = self._projection[bucket]
            for j in range(self.embed_dim):
                embedding[j] += row[j]

        scale = 1.0 / math.sqrt(len(ngrams))
        embedding = [v * scale for v in embedding]

        return EmbeddingVector(embedding).normalize()

    def embed_question(self, question: str) -> EmbeddingVector:
        if not self._initialized:
            self.initialize()

        words = question.lower().split()
        if not words:
            return EmbeddingVector([0.0] * self.embed_dim)

        word_vecs = [self._embed_single(w) for w in words]

        weights = []
        for wv in word_vecs:
            max_sim = 0.0
            for cluster_name, centroid in self._cluster_centers.items():
                sim = wv.cosine(centroid)
                if sim > max_sim:
                    max_sim = sim
            weights.append(math.exp(max_sim * 3.0))

        total_weight = sum(weights) + 1e-10
        weights = [w / total_weight for w in weights]

        result = [0.0] * self.embed_dim
        for i, (wv, weight) in enumerate(zip(word_vecs, weights)):
            for j in range(self.embed_dim):
                result[j] += wv.data[j] * weight

        return EmbeddingVector(result).normalize()

    def embed_column(self, column_name: str) -> EmbeddingVector:
        if not self._initialized:
            self.initialize()
        if column_name in self._column_embeddings:
            return self._column_embeddings[column_name]
        return self._embed_single(column_name.lower().replace('_', ' '))

    def find_best_column(self, question: str, candidates: List[str],
                         top_k: int = 3) -> List[Tuple[str, float]]:
        if not self._initialized:
            self.initialize()

        q_lower = question.lower()
        q_words = set(q_lower.split())
        q_vec = self.embed_question(question)

        scores = []
        for col in candidates:
            col_vec = self.embed_column(col)
            embed_sim = q_vec.cosine(col_vec)

            col_lower = col.lower().replace('_', ' ')
            col_words = set(col_lower.split())
            exact_boost = 0.0

            if col_lower in q_lower:
                exact_boost = 0.6
            elif any(cw in q_words for cw in col_words if len(cw) >= 3):
                exact_boost = 0.4

            if col in COLUMN_SEMANTICS:
                for synonym in COLUMN_SEMANTICS[col]:
                    syn_lower = synonym.lower()
                    if syn_lower in q_lower:
                        exact_boost = max(exact_boost, 0.5)
                        break
                    elif any(sw in q_words for sw in syn_lower.split() if len(sw) >= 4):
                        exact_boost = max(exact_boost, 0.3)

            combined = embed_sim * 0.4 + exact_boost * 0.6
            scores.append((col, combined))

        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    def find_best_concept(self, text: str) -> Tuple[str, float]:
        if not self._initialized:
            self.initialize()

        text_vec = self.embed_question(text)
        best_cluster = 'unknown'
        best_sim = -1.0
        for cluster_name, centroid in self._cluster_centers.items():
            sim = text_vec.cosine(centroid)
            if sim > best_sim:
                best_sim = sim
                best_cluster = cluster_name
        return best_cluster, best_sim

    def similarity(self, text_a: str, text_b: str) -> float:
        if not self._initialized:
            self.initialize()
        vec_a = self.embed_question(text_a)
        vec_b = self.embed_question(text_b)
        return vec_a.cosine(vec_b)

    def _average_vectors(self, vecs: List[EmbeddingVector]) -> EmbeddingVector:
        if not vecs:
            return EmbeddingVector([0.0] * self.embed_dim)
        result = [0.0] * self.embed_dim
        for v in vecs:
            for j in range(self.embed_dim):
                result[j] += v.data[j]
        n = len(vecs)
        result = [v / n for v in result]
        return EmbeddingVector(result).normalize()


class SemanticSearchIndex:

    def __init__(self, embedder: Optional[SemanticEmbedder] = None):
        self.embedder = embedder or SemanticEmbedder()
        self._documents = {}
        self._embeddings = {}
        self._clusters = {}

    def build(self, documents: Dict[str, str]):
        self.embedder.initialize()
        self._documents = dict(documents)

        for doc_id, text in documents.items():
            vec = self.embedder.embed_question(text)
            self._embeddings[doc_id] = vec

            cluster, sim = self.embedder.find_best_concept(text)
            if cluster not in self._clusters:
                self._clusters[cluster] = []
            self._clusters[cluster].append(doc_id)

        logger.debug("SemanticSearchIndex built: %d documents, %d clusters",
                     len(documents), len(self._clusters))

    def search(self, query: str, top_k: int = 5,
               min_score: float = 0.1) -> List[Tuple[str, float]]:
        q_vec = self.embedder.embed_question(query)

        scores = []
        for doc_id, doc_vec in self._embeddings.items():
            sim = q_vec.cosine(doc_vec)
            if sim >= min_score:
                scores.append((doc_id, sim))

        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    def add_document(self, doc_id: str, text: str):
        self._documents[doc_id] = text
        vec = self.embedder.embed_question(text)
        self._embeddings[doc_id] = vec

    def size(self) -> int:
        return len(self._documents)


class SemanticSchemaSearcher:

    def __init__(self, db_path: str, embedder: Optional[SemanticEmbedder] = None):
        self.db_path = db_path
        self.embedder = embedder or SemanticEmbedder()
        self._column_index = SemanticSearchIndex(self.embedder)
        self._table_index = SemanticSearchIndex(self.embedder)
        self._initialized = False

    def initialize(self, schema_info: Optional[Dict] = None):
        if self._initialized:
            return

        self.embedder.initialize()

        col_docs = {}
        for col_name, descriptions in COLUMN_SEMANTICS.items():
            col_text = f"{col_name.lower().replace('_', ' ')} {' '.join(descriptions)}"
            col_docs[col_name] = col_text
        self._column_index.build(col_docs)

        table_docs = {
            'claims': 'claims billing financial paid amount copay deductible denial status',
            'members': 'member patient enrollee demographic age gender region risk',
            'providers': 'provider doctor physician npi specialty department',
            'encounters': 'encounter visit admission discharge inpatient outpatient',
            'diagnoses': 'diagnosis condition icd10 chronic severity hcc',
            'prescriptions': 'prescription medication drug pharmacy therapeutic',
            'appointments': 'appointment scheduling no show cancellation',
            'referrals': 'referral referring specialist authorization',
        }
        self._table_index.build(table_docs)

        self._initialized = True
        logger.info("SemanticSchemaSearcher initialized")

    def find_columns(self, question: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if not self._initialized:
            self.initialize()
        return self._column_index.search(question, top_k=top_k)

    def find_tables(self, question: str, top_k: int = 3) -> List[Tuple[str, float]]:
        if not self._initialized:
            self.initialize()
        return self._table_index.search(question, top_k=top_k)

    def match_column_to_question(self, question: str,
                                  candidates: List[str]) -> List[Tuple[str, float]]:
        return self.embedder.find_best_column(question, candidates)


class SemanticSimilarityEngine:

    def __init__(self, embedder: Optional[SemanticEmbedder] = None):
        self.embedder = embedder or SemanticEmbedder()
        self.embedder.initialize()

    def question_similarity(self, q1: str, q2: str) -> float:
        return self.embedder.similarity(q1, q2)

    def batch_similarity(self, query: str,
                          candidates: List[str]) -> List[Tuple[str, float]]:
        q_vec = self.embedder.embed_question(query)
        results = []
        for c in candidates:
            c_vec = self.embedder.embed_question(c)
            sim = q_vec.cosine(c_vec)
            results.append((c, sim))
        results.sort(key=lambda x: -x[1])
        return results

    def is_semantically_similar(self, q1: str, q2: str,
                                 threshold: float = 0.75) -> bool:
        sim = self.question_similarity(q1, q2)
        return sim >= threshold


_GLOBAL_EMBEDDER: Optional[SemanticEmbedder] = None


def get_embedder() -> SemanticEmbedder:
    global _GLOBAL_EMBEDDER
    if _GLOBAL_EMBEDDER is None:
        _GLOBAL_EMBEDDER = SemanticEmbedder()
        _GLOBAL_EMBEDDER.initialize()
    return _GLOBAL_EMBEDDER


def get_similarity_engine() -> SemanticSimilarityEngine:
    return SemanticSimilarityEngine(get_embedder())
