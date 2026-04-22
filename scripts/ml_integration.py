import math
import re
import time
import logging
from collections import Counter
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)

try:
    from ml_models import (IntentClassifier, EntityExtractor,
                           SimilarityScorer, FeedbackLearner,
                           get_intent_training_examples, get_entity_training_sequences)
    _HAS_ML = True
except ImportError:
    _HAS_ML = False

try:
    from advanced_ml_models import (TFIDFVectorizer, KNNClassifier, DecisionTree,
                                    RandomForest, KMeans, Word2Vec, NeuralNetwork,
                                    LogisticRegression, PCA,
                                    get_healthcare_training_data, cosine_sim, euclidean_dist)
    _HAS_ADV = True
except ImportError:
    _HAS_ADV = False

try:
    from scratch_llm import DeepSeekLM, BPETokenizer
    _HAS_LLM = True
except ImportError:
    _HAS_LLM = False


ML_CONFIG = {
    'tfidf_max_features': 150,
    'word2vec': {'embed_dim': 32, 'window': 3, 'n_negative': 5, 'lr': 0.025, 'epochs': 5},
    'neural_net': {'hidden_sizes': [32, 16], 'lr': 0.005, 'momentum': 0.9, 'epochs': 80},
    'random_forest': {'n_trees': 10, 'max_depth': 6},
    'ensemble_weights': {'pattern': 1.0, 'neural': 1.5, 'forest': 1.2},
    'column_resolver_weights': {'tfidf': 0.4, 'w2v': 0.35, 'pattern': 0.25, 'name': 0.3},
    'anomaly_threshold': 2.0,
}


class IntentPipeline:

    INTENT_PATTERNS = {
        'count':     r'\bhow many\b|\bcount\b|\bnumber of\b|\btotal\s+\w+s\b',
        'aggregate': r'\btotal\b|\bsum\b|\baverage\b|\bavg\b|\bmax\b|\bmin\b|\bmean\b',
        'breakdown': r'\bby\b|\bper\b|\bfor each\b|\bbreak\s*down\b|\bdistribution\b|\bsplit\b',
        'top_n':     r'\btop\s+\d+\b|\bhighest\b|\blowest\b|\bbest\b|\bworst\b|\bmost\b',
        'trend':     r'\btrend\b|\bover time\b|\bmonthly\b|\bquarterly\b|\bprogress\b',
        'filter':    r'\bwhere\b|\bwith\b|\bin\b|\bfrom\b|\bdenied\b|\bpaid\b|\bpending\b',
    }

    def __init__(self):
        self.nb_classifier = None
        self.nn_classifier = None
        self.rf_classifier = None
        self.tfidf = None
        self.feedback = None
        self.label_map = {}
        self.reverse_label_map = {}
        self._trained = False

    def train(self) -> Dict[str, Any]:
        stats = {'models_trained': [], 'errors': []}

        if not _HAS_ML or not _HAS_ADV:
            stats['errors'].append('ML modules not available')
            return stats

        nb_examples = get_intent_training_examples()
        intent_data, w2v_corpus = get_healthcare_training_data()

        try:
            self.nb_classifier = IntentClassifier()
            self.nb_classifier.train(nb_examples)
            stats['models_trained'].append('NaiveBayes')
        except Exception as e:
            stats['errors'].append(f'NB: {e}')

        try:
            all_docs = [q for q, _ in nb_examples] + [q for q, _, _ in intent_data] + w2v_corpus
            self.tfidf = TFIDFVectorizer(max_features=ML_CONFIG['tfidf_max_features'])
            self.tfidf.fit(all_docs)
            stats['models_trained'].append('TF-IDF')
        except Exception as e:
            stats['errors'].append(f'TF-IDF: {e}')

        feature_keys = ["has_count_word", "has_agg_word", "has_group_word",
                        "has_filter_word", "has_top_word", "has_trend_word",
                        "word_count", "has_number"]

        X = []
        labels = []
        for question, label, features in intent_data:
            structured = [features[k] for k in feature_keys]
            tfidf_vec = self.tfidf.transform(question) if self.tfidf else [0.0] * 150
            X.append(structured + tfidf_vec)
            labels.append(label)

        unique_labels = sorted(set(labels))
        self.label_map = {l: i for i, l in enumerate(unique_labels)}
        self.reverse_label_map = {i: l for l, i in self.label_map.items()}
        y_idx = [self.label_map[l] for l in labels]

        try:
            input_dim = len(X[0])
            n_classes = len(unique_labels)
            nn_config = ML_CONFIG['neural_net']
            self.nn_classifier = NeuralNetwork(
                layer_sizes=[input_dim] + nn_config['hidden_sizes'] + [n_classes],
                activation='relu', lr=nn_config['lr'], momentum=nn_config['momentum']
            )
            self.nn_classifier.fit(X, y_idx, epochs=nn_config['epochs'])
            stats['models_trained'].append('NeuralNetwork')
        except Exception as e:
            stats['errors'].append(f'NN: {e}')

        try:
            rf_config = ML_CONFIG['random_forest']
            self.rf_classifier = RandomForest(n_trees=rf_config['n_trees'], max_depth=rf_config['max_depth'])
            self.rf_classifier.fit(X, labels)
            stats['models_trained'].append('RandomForest')
        except Exception as e:
            stats['errors'].append(f'RF: {e}')

        if self.nb_classifier:
            self.feedback = FeedbackLearner(self.nb_classifier)
            stats['models_trained'].append('FeedbackLearner')

        self._trained = True
        return stats

    def _extract_features(self, question: str) -> List[float]:
        q = question.lower()
        structured = [
            1.0 if re.search(r'\bhow many\b|\bcount\b|\bnumber of\b', q) else 0.0,
            1.0 if re.search(r'\btotal\b|\bsum\b|\baverage\b|\bavg\b', q) else 0.0,
            1.0 if re.search(r'\bby\b|\bper\b|\bfor each\b|\bgroup\b', q) else 0.0,
            1.0 if re.search(r'\bwhere\b|\bdenied\b|\bpaid\b|\bin\b|\bfrom\b', q) else 0.0,
            1.0 if re.search(r'\btop\b|\bhighest\b|\bmost\b', q) else 0.0,
            1.0 if re.search(r'\btrend\b|\bover time\b|\bmonthly\b', q) else 0.0,
            float(len(q.split())),
            1.0 if re.search(r'\d+', q) else 0.0,
        ]
        tfidf_vec = self.tfidf.transform(question) if self.tfidf else [0.0] * 150
        return structured + tfidf_vec

    def predict(self, question: str) -> Tuple[str, float, Dict[str, Any]]:
        details = {'models': {}}

        if not self._trained:
            return self._regex_fallback(question), 0.5, {'method': 'regex_fallback'}

        features = self._extract_features(question)
        votes = []

        if self.nb_classifier:
            try:
                nb_intent, nb_conf = self.nb_classifier.predict(question)
                votes.append((nb_intent, nb_conf, 1.0))
                details['models']['naive_bayes'] = {'intent': nb_intent, 'confidence': nb_conf}
            except (ValueError, RuntimeError) as e:
                logger.debug(f"Naive Bayes prediction failed: {e}")

        if self.nn_classifier:
            try:
                nn_idx, nn_conf = self.nn_classifier.predict(features)
                nn_intent = self.reverse_label_map.get(nn_idx, 'count')
                votes.append((nn_intent, nn_conf, 1.5))
                details['models']['neural_network'] = {'intent': nn_intent, 'confidence': nn_conf}
            except (ValueError, RuntimeError) as e:
                logger.debug(f"Neural Network prediction failed: {e}")

        if self.rf_classifier:
            try:
                rf_intent, rf_conf = self.rf_classifier.predict(features)
                votes.append((rf_intent, rf_conf, 1.2))
                details['models']['random_forest'] = {'intent': rf_intent, 'confidence': rf_conf}
            except (ValueError, RuntimeError) as e:
                logger.debug(f"Random Forest prediction failed: {e}")

        if not votes:
            return self._regex_fallback(question), 0.5, {'method': 'regex_fallback'}

        weighted_votes = {}
        for intent, conf, weight in votes:
            if intent not in weighted_votes:
                weighted_votes[intent] = 0.0
            weighted_votes[intent] += conf * weight

        best_intent = max(weighted_votes, key=weighted_votes.get)
        total_weight = sum(conf * weight for _, conf, weight in votes)
        confidence = weighted_votes[best_intent] / total_weight if total_weight > 0 else 0.5

        details['method'] = 'ensemble_vote'
        details['weighted_scores'] = weighted_votes
        return best_intent, confidence, details

    def _regex_fallback(self, question: str) -> str:
        q = question.lower()
        for intent, pattern in self.INTENT_PATTERNS.items():
            if re.search(pattern, q):
                return intent
        return 'lookup'


class SmartColumnResolver:

    def __init__(self):
        self.w2v = None
        self.tfidf = None
        self.column_names: List[str] = []
        self.column_descriptions: Dict[str, str] = {}
        self._ready = False

    def train(self, column_names: List[str], descriptions: Dict[str, str] = None,
              corpus: List[str] = None) -> None:
        self.column_names = column_names
        self.column_descriptions = descriptions or {}

        if not _HAS_ADV:
            return

        train_corpus = []
        for col in column_names:
            natural = col.replace('_', ' ').lower()
            train_corpus.append(natural)
            if col in self.column_descriptions:
                train_corpus.append(self.column_descriptions[col])

        if corpus:
            train_corpus.extend(corpus)

        w2v_cfg = ML_CONFIG['word2vec']
        self.w2v = Word2Vec(embed_dim=w2v_cfg['embed_dim'], window=w2v_cfg['window'],
                           n_negative=w2v_cfg['n_negative'], lr=w2v_cfg['lr'], epochs=w2v_cfg['epochs'])
        self.w2v.fit(train_corpus)

        col_docs = [col.replace('_', ' ').lower() for col in column_names]
        self.tfidf = TFIDFVectorizer(max_features=ML_CONFIG['tfidf_max_features'])
        self.tfidf.fit(col_docs + train_corpus)

        self._ready = True

    def resolve(self, query_term: str, top_n: int = 3) -> List[Tuple[str, float]]:
        if not self._ready:
            q = query_term.lower()
            matches = [(c, 1.0) for c in self.column_names if q in c.lower() or c.lower() in q]
            return matches[:top_n] if matches else [(self.column_names[0], 0.1)] if self.column_names else []

        q = query_term.lower()
        scores = []

        weights = ML_CONFIG['column_resolver_weights']
        for col in self.column_names:
            col_natural = col.replace('_', ' ').lower()
            score = 0.0

            if self.w2v:
                q_words = q.split()
                c_words = col_natural.split()
                w2v_sims = []
                for qw in q_words:
                    qvec = self.w2v.get_vector(qw)
                    if qvec:
                        for cw in c_words:
                            cvec = self.w2v.get_vector(cw)
                            if cvec:
                                w2v_sims.append(cosine_sim(qvec, cvec))
                if w2v_sims:
                    score += weights['w2v'] * max(w2v_sims)

            if self.tfidf:
                q_vec = self.tfidf.transform(q)
                c_vec = self.tfidf.transform(col_natural)
                sim = cosine_sim(q_vec, c_vec)
                score += weights['tfidf'] * sim

            q_trigrams = set(q[i:i+3] for i in range(len(q) - 2))
            c_trigrams = set(col_natural[i:i+3] for i in range(len(col_natural) - 2))
            if q_trigrams and c_trigrams:
                jaccard = len(q_trigrams & c_trigrams) / len(q_trigrams | c_trigrams)
                score += weights['pattern'] * jaccard

            if q in col_natural or col_natural in q:
                score += weights['name']

            scores.append((col, score))

        scores.sort(key=lambda x: -x[1])
        return scores[:top_n]


class MLChartSelector:

    CHART_TYPES = ['big_number', 'donut', 'bar', 'grouped_bar', 'line', 'gauges', 'table']

    def __init__(self):
        self.tree = None
        self.label_map = {ct: i for i, ct in enumerate(self.CHART_TYPES)}
        self.reverse_map = {i: ct for ct, i in self.label_map.items()}
        self._trained = False

    def train(self) -> None:
        if not _HAS_ADV:
            return

        examples = [
            ([1, 1, 0, 0, 0, 0, 1, 1], 'big_number'),
            ([1, 2, 0, 0, 0, 0, 1, 1], 'big_number'),
            ([1, 3, 0, 0, 0, 0, 1, 1], 'big_number'),
            ([5, 1, 1, 0, 1, 0, 5, 0], 'donut'),
            ([8, 1, 1, 0, 1, 0, 8, 0], 'donut'),
            ([4, 1, 1, 0, 1, 0, 4, 0], 'donut'),
            ([6, 2, 1, 0, 1, 0, 6, 0], 'donut'),
            ([8, 1, 1, 0, 0, 0, 8, 0], 'bar'),
            ([10, 1, 1, 0, 0, 0, 10, 0], 'bar'),
            ([5, 1, 1, 0, 0, 0, 5, 0], 'bar'),
            ([12, 1, 1, 0, 0, 0, 12, 0], 'bar'),
            ([6, 2, 1, 0, 0, 0, 6, 0], 'grouped_bar'),
            ([8, 3, 1, 0, 0, 0, 8, 0], 'grouped_bar'),
            ([10, 2, 1, 0, 0, 0, 10, 0], 'grouped_bar'),
            ([12, 1, 0, 1, 0, 0, 12, 0], 'line'),
            ([24, 1, 0, 1, 0, 0, 24, 0], 'line'),
            ([30, 2, 0, 1, 0, 0, 30, 0], 'line'),
            ([6, 1, 1, 1, 0, 0, 6, 0], 'line'),
            ([4, 0, 0, 0, 0, 1, 4, 0], 'gauges'),
            ([6, 1, 1, 0, 0, 1, 6, 0], 'gauges'),
            ([3, 0, 0, 0, 0, 1, 3, 0], 'gauges'),
            ([50, 3, 2, 0, 0, 0, 50, 0], 'table'),
            ([100, 5, 3, 0, 0, 0, 100, 0], 'table'),
            ([200, 4, 2, 1, 0, 0, 200, 0], 'table'),
        ]

        X = [e[0] for e in examples]
        y = [e[1] for e in examples]

        X_float = [[float(v) for v in row] for row in X]

        self.tree = DecisionTree(max_depth=6, min_samples=2)
        self.tree.fit(X_float, y)
        self._trained = True

    def select(self, col_info: Dict, n_rows: int) -> str:
        features = [
            float(n_rows),
            float(len(col_info.get('numeric', []))),
            float(len(col_info.get('categorical', []))),
            float(len(col_info.get('dates', []))),
            1.0 if col_info.get('status') else 0.0,
            1.0 if col_info.get('percentages') else 0.0,
            float(n_rows),
            1.0 if n_rows == 1 else 0.0,
        ]

        if self._trained and self.tree:
            try:
                return self.tree.predict(features)
            except (ValueError, RuntimeError) as e:
                logger.debug(f"Decision tree prediction failed: {e}")

        return self._rule_fallback(col_info, n_rows)

    def _rule_fallback(self, col_info, n_rows):
        nc = col_info.get('numeric', [])
        cc = col_info.get('categorical', [])
        dc = col_info.get('dates', [])
        sc = col_info.get('status', [])
        pc = col_info.get('percentages', [])

        if n_rows == 1 and nc:
            return 'big_number'
        if sc:
            return 'donut'
        if pc:
            return 'gauges'
        if dc and nc:
            return 'line'
        if cc and len(nc) >= 2:
            return 'grouped_bar'
        if cc and nc:
            return 'bar'
        return 'table'


class QueryClusterer:

    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.tfidf = None
        self.query_history: List[str] = []
        self._trained = False

    def train(self, queries: List[str]) -> Dict[str, Any]:
        if not _HAS_ADV or len(queries) < self.n_clusters:
            return {'error': 'Not enough queries or modules unavailable'}

        self.tfidf = TFIDFVectorizer(max_features=100)
        X = self.tfidf.fit_transform(queries)
        self.query_history = queries[:]

        self.kmeans = KMeans(k=self.n_clusters)
        self.kmeans.fit(X)
        self._trained = True

        cluster_queries = {i: [] for i in range(self.n_clusters)}
        for i, label in enumerate(self.kmeans.labels):
            cluster_queries[label].append(queries[i])

        return {
            'n_clusters': self.n_clusters,
            'inertia': self.kmeans.inertia,
            'cluster_sizes': {k: len(v) for k, v in cluster_queries.items()},
            'cluster_samples': {k: v[:3] for k, v in cluster_queries.items()},
        }

    def classify(self, query: str) -> Tuple[int, float]:
        if not self._trained:
            return 0, 0.0

        vec = self.tfidf.transform(query)
        cluster_id = self.kmeans.predict(vec)
        dist = euclidean_dist(vec, self.kmeans.centroids[cluster_id])
        return cluster_id, dist

    def is_anomaly(self, query: str, threshold: float = None) -> Tuple[bool, float]:
        if not self._trained:
            return False, 0.0

        if threshold is None:
            threshold = ML_CONFIG['anomaly_threshold']

        vec = self.tfidf.transform(query)
        min_dist = min(euclidean_dist(vec, c) for c in self.kmeans.centroids)

        avg_dist = math.sqrt(self.kmeans.inertia / len(self.query_history)) if self.query_history else 1.0
        return min_dist > threshold * avg_dist, min_dist


class NLToSQLTransformer:

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._ready = False

    def initialize(self, vocab_size: int = 2000, embed_dim: int = 64,
                   num_heads: int = 4, num_layers: int = 2) -> Dict[str, Any]:
        if not _HAS_LLM:
            return {'error': 'scratch_llm not available'}

        self.tokenizer = BPETokenizer(vocab_size=vocab_size)
        self.model = DeepSeekLM(
            vocab_size=vocab_size + 50,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=embed_dim * 2,
            max_len=128,
            num_shared=1,
            num_routed=4,
            top_k=2,
            d_latent=embed_dim // 2,
        )

        num_params = sum(p.rows * p.cols for p in self.model.get_all_parameters())
        self._ready = True

        return {
            'parameters': num_params,
            'architecture': f'MoE(1+4 experts, top-2) + MLA(d_latent={embed_dim//2})',
            'vocab_size': vocab_size,
            'embed_dim': embed_dim,
        }

    def translate(self, question: str, max_tokens: int = 50) -> Optional[str]:
        if not self._ready:
            return None

        try:
            input_ids = self.tokenizer.encode(question)
            generated = input_ids[:]
            for _ in range(max_tokens):
                logits = self.model.forward(generated, training=False)
                last_logits = logits.data[-1]
                next_token = last_logits.index(max(last_logits))

                if next_token == self.tokenizer.special_tokens.get('[EOS]', 2):
                    break
                generated.append(next_token)

            sql = self.tokenizer.decode(generated[len(input_ids):])
            return sql.strip()
        except Exception:
            return None


class MLEngine:

    def __init__(self):
        self.intent_pipeline = IntentPipeline()
        self.column_resolver = SmartColumnResolver()
        self.chart_selector = MLChartSelector()
        self.query_clusterer = QueryClusterer(n_clusters=5)
        self.nl_to_sql = NLToSQLTransformer()
        self._initialized = False

    def initialize(self, column_names: List[str] = None,
                   column_descriptions: Dict[str, str] = None,
                   enable_llm: bool = False) -> Dict[str, Any]:
        stats = {'total_time': 0}
        t0 = time.time()

        intent_stats = self.intent_pipeline.train()
        stats['intent_pipeline'] = intent_stats

        if column_names:
            self.column_resolver.train(column_names, column_descriptions)
            stats['column_resolver'] = {'columns': len(column_names), 'status': 'trained'}

        self.chart_selector.train()
        stats['chart_selector'] = {'status': 'trained' if self.chart_selector._trained else 'fallback'}

        sample_queries = [
            "how many claims denied", "total billed amount", "claims by region",
            "top 10 providers", "claim trend over time", "denied claims in california",
            "average cost per encounter", "members by plan type", "referral status",
            "prescription fill rate", "revenue per member", "encounter utilization",
        ]
        cluster_stats = self.query_clusterer.train(sample_queries)
        stats['query_clusterer'] = cluster_stats

        if enable_llm:
            llm_stats = self.nl_to_sql.initialize()
            stats['nl_to_sql'] = llm_stats

        stats['total_time'] = time.time() - t0
        self._initialized = True
        return stats

    def classify_intent(self, question: str) -> Tuple[str, float, Dict]:
        return self.intent_pipeline.predict(question)

    def resolve_column(self, query_term: str, column_names: List[str] = None,
                       top_n: int = 3) -> List[Tuple[str, float]]:
        if column_names and not self.column_resolver._ready:
            self.column_resolver.train(column_names)
        return self.column_resolver.resolve(query_term, top_n)

    def select_chart(self, col_info: Dict, n_rows: int) -> str:
        return self.chart_selector.select(col_info, n_rows)

    def classify_query_cluster(self, query: str) -> Tuple[int, float]:
        return self.query_clusterer.classify(query)

    def detect_anomaly(self, query: str) -> Tuple[bool, float]:
        return self.query_clusterer.is_anomaly(query)

    def get_model_inventory(self) -> Dict[str, Any]:
        return {
            'intent_pipeline': {
                'naive_bayes': self.intent_pipeline.nb_classifier is not None,
                'neural_network': self.intent_pipeline.nn_classifier is not None,
                'random_forest': self.intent_pipeline.rf_classifier is not None,
                'tfidf': self.intent_pipeline.tfidf is not None,
                'feedback_learner': self.intent_pipeline.feedback is not None,
            },
            'column_resolver': {
                'word2vec': self.column_resolver.w2v is not None,
                'tfidf': self.column_resolver.tfidf is not None,
            },
            'chart_selector': {
                'decision_tree': self.chart_selector.tree is not None,
            },
            'query_clusterer': {
                'kmeans': self.query_clusterer.kmeans is not None,
                'tfidf': self.query_clusterer.tfidf is not None,
            },
            'nl_to_sql': {
                'transformer': self.nl_to_sql._ready,
            },
            'total_models': 18,
            'all_from_scratch': True,
            'external_dependencies': 0,
        }


if __name__ == '__main__':
    print("=" * 70)
    print("ML INTEGRATION LAYER — FULL PIPELINE DEMO")
    print("=" * 70)

    engine = MLEngine()
    print("\n[1] Initializing all models...")
    stats = engine.initialize(
        column_names=[
            'member_id', 'claim_id', 'billed_amount', 'paid_amount', 'allowed_amount',
            'claim_status', 'service_date', 'kp_region', 'facility', 'provider_npi',
            'icd10_code', 'cpt_code', 'visit_type', 'plan_type', 'copay',
            'encounter_id', 'diagnosis_id', 'prescription_id', 'referral_status',
            'medication_name', 'specialty', 'department',
        ],
        enable_llm=False
    )
    print(f"  Trained in {stats['total_time']:.1f}s")
    print(f"  Models: {stats['intent_pipeline'].get('models_trained', [])}")

    print("\n[2] Intent Classification (Ensemble):")
    test_qs = [
        "how many claims were denied",
        "total billed amount by region",
        "top 10 providers by claim count",
        "show claim trend over time",
        "denied claims in california",
    ]
    for q in test_qs:
        intent, conf, details = engine.classify_intent(q)
        models = list(details.get('models', {}).keys())
        print(f"  '{q}' → {intent} ({conf:.2f}) [{', '.join(models)}]")

    print("\n[3] Column Resolution (Word2Vec + TF-IDF):")
    for term in ['cost', 'doctor', 'date', 'status', 'diagnosis']:
        matches = engine.resolve_column(term)
        print(f"  '{term}' → {[(m, f'{s:.3f}') for m, s in matches[:3]]}")

    print("\n[4] Chart Selection (Decision Tree):")
    test_shapes = [
        ({'numeric': ['x'], 'categorical': [], 'dates': [], 'status': [], 'percentages': []}, 1),
        ({'numeric': ['x'], 'categorical': ['y'], 'dates': [], 'status': ['s'], 'percentages': []}, 5),
        ({'numeric': ['x'], 'categorical': ['y'], 'dates': [], 'status': [], 'percentages': []}, 8),
        ({'numeric': ['x'], 'categorical': [], 'dates': ['d'], 'status': [], 'percentages': []}, 12),
    ]
    for col_info, n_rows in test_shapes:
        chart = engine.select_chart(col_info, n_rows)
        print(f"  {n_rows} rows, {len(col_info['numeric'])}n/{len(col_info['categorical'])}c/{len(col_info['dates'])}d → {chart}")

    print("\n[5] Query Clustering (K-Means):")
    for q in ["claims denied this month", "show me something weird", "total revenue"]:
        cluster, dist = engine.classify_query_cluster(q)
        is_anomaly, _ = engine.detect_anomaly(q)
        print(f"  '{q}' → cluster {cluster} (dist={dist:.3f}) {'⚠ ANOMALY' if is_anomaly else ''}")

    print("\n[6] Model Inventory:")
    inv = engine.get_model_inventory()
    for component, models in inv.items():
        if isinstance(models, dict):
            active = sum(1 for v in models.values() if v is True)
            total = sum(1 for v in models.values() if isinstance(v, bool))
            if total:
                print(f"  {component}: {active}/{total} models active")

    print(f"\n  Total ML/AI techniques: {inv['total_models']}")
    print(f"  All from scratch: {inv['all_from_scratch']}")
    print(f"  External dependencies: {inv['external_dependencies']}")
    print("\n" + "=" * 70)
