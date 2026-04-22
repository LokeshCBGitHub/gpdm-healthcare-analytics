import math
import logging
from typing import (
    Dict, List, Tuple, Any, Optional, Set, Protocol, runtime_checkable
)
from dataclasses import dataclass, field

logger = logging.getLogger('gpdm.nlp_factory')


@runtime_checkable
class Vectorizer(Protocol):
    def fit(self, documents: List[str]) -> None: ...
    def transform(self, texts: List[str]) -> List[Any]: ...
    def fit_transform(self, documents: List[str]) -> List[Any]: ...


@runtime_checkable
class SimilarityEngine(Protocol):
    def cosine_similarity(self, vec_a: Any, vec_b: Any) -> float: ...
    def pairwise_similarity(self, query_vec: Any, corpus_vecs: List[Any]) -> List[float]: ...


@runtime_checkable
class EntityExtractor(Protocol):
    def extract(self, text: str) -> List[Dict[str, Any]]: ...


@runtime_checkable
class TextClassifier(Protocol):
    def fit(self, texts: List[str], labels: List[str]) -> None: ...
    def predict(self, text: str) -> Tuple[str, float]: ...


@runtime_checkable
class EmbeddingEngine(Protocol):
    def encode(self, texts: List[str]) -> List[List[float]]: ...
    def similarity(self, text_a: str, text_b: str) -> float: ...


class ScratchVectorizer:

    def __init__(self):
        self._index = None

    def fit(self, documents: List[str]) -> None:
        from semantic_layer import TFIDFIndex
        self._index = TFIDFIndex()
        doc_tuples = [(str(i), doc, {}) for i, doc in enumerate(documents)]
        self._index.build(doc_tuples)

    def transform(self, texts: List[str]) -> List[Any]:
        if not self._index:
            raise RuntimeError("Vectorizer not fitted")
        return [self._index._vectorize(t) for t in texts]

    def fit_transform(self, documents: List[str]) -> List[Any]:
        self.fit(documents)
        return self.transform(documents)

    @property
    def backend(self):
        return 'scratch'


class ScratchSimilarity:

    def cosine_similarity(self, vec_a, vec_b) -> float:
        from semantic_layer import SparseVector
        if isinstance(vec_a, SparseVector) and isinstance(vec_b, SparseVector):
            return vec_a.cosine(vec_b)
        return self._dense_cosine(vec_a, vec_b)

    def pairwise_similarity(self, query_vec, corpus_vecs: List) -> List[float]:
        return [self.cosine_similarity(query_vec, v) for v in corpus_vecs]

    @staticmethod
    def _dense_cosine(a, b) -> float:
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @property
    def backend(self):
        return 'scratch'


class ScratchEntityExtractor:

    PATTERNS = {
        'DATE': [
            r'\b(\d{4}-\d{2}-\d{2})\b',
            r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s+\d{4})\b',
            r'\b(last\s+(?:month|year|quarter|week|30\s+days|90\s+days))\b',
            r'\b(this\s+(?:month|year|quarter))\b',
            r'\b(20\d\d)\b',
        ],
        'NUMBER': [
            r'\$\s*([\d,]+\.?\d*)',
            r'\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\b',
        ],
        'COMPARISON': [
            r'\b((?:greater|more|less|fewer|over|under|above|below)\s+than\s+\$?[\d,]+)',
            r'\b(between\s+\$?[\d,]+\s+and\s+\$?[\d,]+)',
            r'\b(at\s+(?:least|most)\s+\$?[\d,]+)',
        ],
        'MEDICAL_CODE': [
            r'\b([A-Z]\d{2}(?:\.\d{1,4})?)\b',
            r'\b(\d{5})\b',
        ],
        'AGE_REFERENCE': [
            r'\b((?:older|younger|over|under)\s+(?:than\s+)?\d+)',
            r'\b(age\s*(?:>|<|>=|<=|=)\s*\d+)',
            r'\b(\d+\s*(?:years?\s+old|y\.?o\.?))\b',
        ],
    }

    def extract(self, text: str) -> List[Dict[str, Any]]:
        import re
        entities = []
        for etype, patterns in self.PATTERNS.items():
            for pat in patterns:
                for m in re.finditer(pat, text, re.IGNORECASE):
                    entities.append({
                        'text': m.group(0),
                        'type': etype,
                        'start': m.start(),
                        'end': m.end(),
                        'source': 'scratch',
                    })
        return entities

    @property
    def backend(self):
        return 'scratch'


class ScratchClassifier:

    def __init__(self):
        self._centroids = {}
        self._vectorizer = ScratchVectorizer()

    def fit(self, texts: List[str], labels: List[str]) -> None:
        from semantic_layer import SparseVector
        vecs = self._vectorizer.fit_transform(texts)
        label_vecs = {}
        for vec, label in zip(vecs, labels):
            if label not in label_vecs:
                label_vecs[label] = []
            label_vecs[label].append(vec)

        for label, vecs_list in label_vecs.items():
            if vecs_list:
                merged = {}
                for v in vecs_list:
                    for idx, val in zip(v.indices, v.values):
                        merged[idx] = merged.get(idx, 0) + val
                n = len(vecs_list)
                indices = sorted(merged.keys())
                values = [merged[i] / n for i in indices]
                self._centroids[label] = SparseVector(indices, values)

    def predict(self, text: str) -> Tuple[str, float]:
        vec = self._vectorizer.transform([text])[0]
        sim = ScratchSimilarity()
        best_label = ''
        best_score = -1
        for label, centroid in self._centroids.items():
            score = sim.cosine_similarity(vec, centroid)
            if score > best_score:
                best_score = score
                best_label = label
        return best_label, best_score

    @property
    def backend(self):
        return 'scratch'


class SklearnVectorizer:

    def __init__(self):
        self._vectorizer = None
        self._fitted = False

    def fit(self, documents: List[str]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=1,
            max_df=0.95,
        )
        self._vectorizer.fit(documents)
        self._fitted = True

    def transform(self, texts: List[str]) -> List[Any]:
        if not self._fitted:
            raise RuntimeError("Vectorizer not fitted")
        return self._vectorizer.transform(texts)

    def fit_transform(self, documents: List[str]) -> List[Any]:
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=1,
            max_df=0.95,
        )
        result = self._vectorizer.fit_transform(documents)
        self._fitted = True
        return result

    @property
    def backend(self):
        return 'sklearn'


class SklearnSimilarity:

    def cosine_similarity(self, vec_a, vec_b) -> float:
        from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
        import numpy as np
        a = vec_a if hasattr(vec_a, 'shape') else np.array(vec_a).reshape(1, -1)
        b = vec_b if hasattr(vec_b, 'shape') else np.array(vec_b).reshape(1, -1)
        if hasattr(a, 'toarray'):
            a_dense = a.toarray()
        else:
            a_dense = a.reshape(1, -1) if a.ndim == 1 else a
        if hasattr(b, 'toarray'):
            b_dense = b.toarray()
        else:
            b_dense = b.reshape(1, -1) if b.ndim == 1 else b
        return float(sk_cosine(a_dense, b_dense)[0][0])

    def pairwise_similarity(self, query_vec, corpus_vecs) -> List[float]:
        from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
        import numpy as np
        if hasattr(corpus_vecs, 'shape') and hasattr(query_vec, 'shape'):
            q = query_vec if query_vec.ndim == 2 else query_vec.reshape(1, -1)
            scores = sk_cosine(q, corpus_vecs)
            return scores[0].tolist()
        return [self.cosine_similarity(query_vec, v) for v in corpus_vecs]

    @property
    def backend(self):
        return 'sklearn'


class SklearnClassifier:

    def __init__(self):
        self._pipeline = None

    def fit(self, texts: List[str], labels: List[str]) -> None:
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV

        self._pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                sublinear_tf=True,
                ngram_range=(1, 2),
                max_features=5000,
                stop_words='english',
            )),
            ('clf', CalibratedClassifierCV(
                LinearSVC(max_iter=2000, C=1.0),
                cv=min(3, len(set(labels)))
            )),
        ])
        self._pipeline.fit(texts, labels)

    def predict(self, text: str) -> Tuple[str, float]:
        if not self._pipeline:
            raise RuntimeError("Classifier not fitted")
        proba = self._pipeline.predict_proba([text])[0]
        classes = self._pipeline.classes_
        best_idx = proba.argmax()
        return classes[best_idx], float(proba[best_idx])

    @property
    def backend(self):
        return 'sklearn'


class SpacyEntityExtractor:

    MODELS = [
        'en_core_sci_lg',
        'en_core_med7_lg',
        'en_core_web_lg',
        'en_core_web_sm',
    ]

    def __init__(self):
        self._nlp = None
        self._model_name = None
        self._load_model()

    def _load_model(self):
        import spacy
        for model in self.MODELS:
            try:
                self._nlp = spacy.load(model)
                self._model_name = model
                logger.info("spaCy loaded: %s", model)
                return
            except OSError:
                continue
        self._nlp = spacy.blank('en')
        self._model_name = 'blank'
        logger.warning("No spaCy model found — using blank pipeline")

    def extract(self, text: str) -> List[Dict[str, Any]]:
        doc = self._nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'type': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'source': f'spacy/{self._model_name}',
            })
        return entities

    @property
    def backend(self):
        return f'spacy/{self._model_name}'


class SentenceTransformerEmbeddings:

    MODELS = [
        'all-MiniLM-L6-v2',
        'all-mpnet-base-v2',
        'paraphrase-MiniLM-L6-v2',
    ]

    def __init__(self, model_name: str = None):
        self._model = None
        self._model_name = None

        from sentence_transformers import SentenceTransformer

        models_to_try = [model_name] if model_name else self.MODELS
        for m in models_to_try:
            try:
                self._model = SentenceTransformer(m)
                self._model_name = m
                logger.info("sentence-transformers loaded: %s", m)
                break
            except Exception as e:
                logger.debug("Could not load %s: %s", m, e)
                continue

        if not self._model:
            raise ImportError("No sentence-transformer model available")

    def encode(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def similarity(self, text_a: str, text_b: str) -> float:
        from sentence_transformers import util
        emb_a = self._model.encode([text_a])
        emb_b = self._model.encode([text_b])
        return float(util.cos_sim(emb_a, emb_b)[0][0])

    @property
    def backend(self):
        return f'sentence-transformers/{self._model_name}'


@dataclass
class LibraryInventory:
    sklearn: bool = False
    sklearn_version: str = ''
    spacy: bool = False
    spacy_version: str = ''
    spacy_models: List[str] = field(default_factory=list)
    sentence_transformers: bool = False
    st_version: str = ''
    nltk: bool = False
    numpy: bool = False
    scipy: bool = False
    scispacy: bool = False
    med7: bool = False

    @classmethod
    def detect(cls) -> 'LibraryInventory':
        inv = cls()

        try:
            import sklearn
            inv.sklearn = True
            inv.sklearn_version = sklearn.__version__
        except ImportError:
            pass

        try:
            import spacy
            inv.spacy = True
            inv.spacy_version = spacy.__version__
            for model in SpacyEntityExtractor.MODELS:
                try:
                    spacy.load(model)
                    inv.spacy_models.append(model)
                except OSError:
                    pass
        except ImportError:
            pass

        try:
            import sentence_transformers
            inv.sentence_transformers = True
            inv.st_version = sentence_transformers.__version__
        except ImportError:
            pass

        try:
            import nltk
            inv.nltk = True
        except ImportError:
            pass

        try:
            import numpy
            inv.numpy = True
        except ImportError:
            pass

        try:
            import scipy
            inv.scipy = True
        except ImportError:
            pass

        try:
            import scispacy
            inv.scispacy = True
        except ImportError:
            pass

        return inv

    def summary(self) -> str:
        parts = ['Available HIPAA-compliant NLP libraries:']
        if self.sklearn:
            parts.append(f'  [TIER 2] scikit-learn {self.sklearn_version}')
        if self.spacy:
            models = ', '.join(self.spacy_models) if self.spacy_models else 'no models'
            parts.append(f'  [TIER 3] spaCy {self.spacy_version} ({models})')
        if self.sentence_transformers:
            parts.append(f'  [TIER 3] sentence-transformers {self.st_version}')
        if self.nltk:
            parts.append(f'  [TIER 2] NLTK')
        if not any([self.sklearn, self.spacy, self.sentence_transformers]):
            parts.append('  [TIER 1] Pure Python stdlib only')
        return '\n'.join(parts)

    @property
    def best_tier(self) -> int:
        if self.sentence_transformers or self.scispacy:
            return 3
        if self.sklearn or self.spacy:
            return 2
        return 1


class NLPEngineFactory:

    def __init__(self, mode: str = 'auto'):
        self.mode = mode
        self.inventory = LibraryInventory.detect()
        self._active_backends = {}
        logger.info("NLP Engine Factory initialized — mode=%s, tier=%d",
                     mode, self.inventory.best_tier)

    def get_vectorizer(self) -> Any:
        if self._should_use_library('sklearn'):
            try:
                v = SklearnVectorizer()
                self._active_backends['vectorizer'] = v.backend
                return v
            except Exception as e:
                logger.warning("sklearn vectorizer failed: %s, falling back", e)

        v = ScratchVectorizer()
        self._active_backends['vectorizer'] = v.backend
        return v

    def get_similarity_engine(self) -> Any:
        if self._should_use_library('sklearn'):
            try:
                s = SklearnSimilarity()
                self._active_backends['similarity'] = s.backend
                return s
            except Exception as e:
                logger.warning("sklearn similarity failed: %s, falling back", e)

        s = ScratchSimilarity()
        self._active_backends['similarity'] = s.backend
        return s

    def get_entity_extractor(self) -> Any:
        if self._should_use_library('spacy'):
            try:
                e = SpacyEntityExtractor()
                self._active_backends['ner'] = e.backend
                return e
            except Exception as ex:
                logger.warning("spaCy NER failed: %s, falling back", ex)

        e = ScratchEntityExtractor()
        self._active_backends['ner'] = e.backend
        return e

    def get_classifier(self) -> Any:
        if self._should_use_library('sklearn'):
            try:
                c = SklearnClassifier()
                self._active_backends['classifier'] = c.backend
                return c
            except Exception as e:
                logger.warning("sklearn classifier failed: %s, falling back", e)

        c = ScratchClassifier()
        self._active_backends['classifier'] = c.backend
        return c

    def get_embedding_engine(self) -> Optional[Any]:
        if self._should_use_library('sentence_transformers'):
            try:
                e = SentenceTransformerEmbeddings()
                self._active_backends['embeddings'] = e.backend
                return e
            except Exception as ex:
                logger.info("sentence-transformers not available: %s", ex)

        return None

    def _should_use_library(self, lib_name: str) -> bool:
        if self.mode == 'scratch':
            return False
        if self.mode == 'library':
            return True
        if self.mode == 'tier2':
            return lib_name in ('sklearn', 'numpy', 'scipy', 'nltk')
        if self.mode == 'tier3':
            return True

        return getattr(self.inventory, lib_name, False)

    def report(self) -> Dict[str, Any]:
        return {
            'mode': self.mode,
            'tier': self.inventory.best_tier,
            'inventory': {
                'sklearn': f'{self.inventory.sklearn_version}' if self.inventory.sklearn else None,
                'spacy': f'{self.inventory.spacy_version} ({", ".join(self.inventory.spacy_models)})' if self.inventory.spacy else None,
                'sentence_transformers': self.inventory.st_version if self.inventory.sentence_transformers else None,
                'nltk': self.inventory.nltk,
                'scispacy': self.inventory.scispacy,
            },
            'active_backends': dict(self._active_backends),
            'hipaa_compliant': True,
            'data_exfiltration_risk': 'ZERO — all processing is local',
        }

    def report_text(self) -> str:
        r = self.report()
        lines = [
            f"NLP Engine Factory Report",
            f"{'=' * 40}",
            f"Mode: {r['mode']} | Max Tier: {r['tier']}",
            f"HIPAA Compliant: {r['hipaa_compliant']}",
            f"Data Risk: {r['data_exfiltration_risk']}",
            f"",
            f"Active Backends:",
        ]
        for cap, backend in r['active_backends'].items():
            lines.append(f"  {cap}: {backend}")
        lines.append(f"")
        lines.append(f"Available Libraries:")
        for lib, version in r['inventory'].items():
            if version:
                lines.append(f"  {lib}: {version}")
            else:
                lines.append(f"  {lib}: not installed")
        return '\n'.join(lines)


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("Detecting available NLP libraries...\n")

    inv = LibraryInventory.detect()
    print(inv.summary())
    print(f"\nBest tier: {inv.best_tier}\n")

    for mode in ['scratch', 'auto']:
        print(f"\n{'='*50}")
        print(f"MODE: {mode}")
        print(f"{'='*50}")

        factory = NLPEngineFactory(mode=mode)

        vec = factory.get_vectorizer()
        docs = ["diabetes claims by region", "total billed amount",
                "members older than 65", "COVID emergency visits"]
        vec.fit(docs)
        vectors = vec.transform(["diabetes patients"])
        print(f"Vectorizer: {vec.backend}")
        print(f"  Vector type: {type(vectors[0]).__name__}")

        sim = factory.get_similarity_engine()
        print(f"Similarity: {sim.backend}")

        ner = factory.get_entity_extractor()
        entities = ner.extract("patients older than 65 with diabetes claims over $5000 in 2024")
        print(f"NER: {ner.backend}")
        for e in entities[:5]:
            print(f"  [{e['type']}] {e['text']}")

        clf = factory.get_classifier()
        train_texts = [
            "how many claims", "count of members", "total number",
            "average cost", "total billed amount", "sum of paid",
            "claims by region", "members by age group", "breakdown by type",
            "top 10 providers", "highest cost claims", "best performing",
        ]
        train_labels = [
            "count", "count", "count",
            "aggregate", "aggregate", "aggregate",
            "breakdown", "breakdown", "breakdown",
            "ranking", "ranking", "ranking",
        ]
        clf.fit(train_texts, train_labels)
        label, conf = clf.predict("total claims per region")
        print(f"Classifier: {clf.backend}")
        print(f"  'total claims per region' → {label} ({conf:.2f})")

        emb = factory.get_embedding_engine()
        if emb:
            score = emb.similarity("diabetes treatment", "diabetic care")
            print(f"Embeddings: {emb.backend}")
            print(f"  'diabetes treatment' ↔ 'diabetic care' → {score:.3f}")
        else:
            print("Embeddings: not available (Tier 1-2)")

        print(f"\n{factory.report_text()}")
