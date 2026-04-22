import numpy as np
import pandas as pd
import re
import os
import json
import sqlite3
import difflib
import logging
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Any

TFIDF_MAX_FEATURES = 5000
SIMILARITY_THRESHOLD = 0.01
NGRAM_RANGE = (1, 2)

logger = logging.getLogger('gpdm.nlp')

class NumpyTFIDF:

    def __init__(self, max_features: int = TFIDF_MAX_FEATURES, ngram_range: tuple = NGRAM_RANGE):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vocabulary_: Dict[str, int] = {}
        self.idf_: Optional[np.ndarray] = None
        self._fitted = False

    def _tokenize(self, text: str) -> List[str]:
        words = re.findall(r'[a-z_]+', text.lower())
        tokens = list(words)
        if self.ngram_range[1] >= 2:
            for i in range(len(words) - 1):
                tokens.append(f"{words[i]}_{words[i+1]}")
        if self.ngram_range[1] >= 3:
            for i in range(len(words) - 2):
                tokens.append(f"{words[i]}_{words[i+1]}_{words[i+2]}")
        return tokens

    def fit(self, documents: List[str]) -> 'NumpyTFIDF':
        n_docs = len(documents)

        df_counts = Counter()
        all_tokens = []
        for doc in documents:
            tokens = self._tokenize(doc)
            all_tokens.append(tokens)
            df_counts.update(set(tokens))

        most_common = df_counts.most_common(self.max_features)
        self.vocabulary_ = {term: idx for idx, (term, _) in enumerate(most_common)}

        n_terms = len(self.vocabulary_)
        self.idf_ = np.zeros(n_terms)
        for term, idx in self.vocabulary_.items():
            df = df_counts[term]
            self.idf_[idx] = np.log((1 + n_docs) / (1 + df)) + 1

        self._fitted = True
        return self

    def transform(self, documents: List[str]) -> np.ndarray:
        if not self._fitted:
            raise ValueError("Must call fit() first")

        n_docs = len(documents)
        n_terms = len(self.vocabulary_)
        matrix = np.zeros((n_docs, n_terms))

        for i, doc in enumerate(documents):
            tokens = self._tokenize(doc)
            term_counts = Counter(tokens)
            for term, count in term_counts.items():
                if term in self.vocabulary_:
                    idx = self.vocabulary_[term]
                    tf = 1 + np.log(count) if count > 0 else 0
                    matrix[i, idx] = tf * self.idf_[idx]

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        matrix = matrix / norms

        return matrix

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        return self.fit(documents).transform(documents)


class NumpyVectorIndex:

    def __init__(self):
        self.vectors: Optional[np.ndarray] = None
        self.ids: List[str] = []
        self.metadata: List[Dict] = []
        self._vectorizer: Optional[NumpyTFIDF] = None

    def build(self, items: List[Dict], text_key: str = 'text'):
        texts = [item.get(text_key, '') for item in items]
        self.ids = [item.get('id', str(i)) for i, item in enumerate(items)]
        self.metadata = items

        self._vectorizer = NumpyTFIDF(max_features=3000, ngram_range=NGRAM_RANGE)
        self.vectors = self._vectorizer.fit_transform(texts)

    def search(self, query: str, k: int = 5) -> List[Dict]:
        if self.vectors is None or self._vectorizer is None:
            return []

        q_vec = self._vectorizer.transform([query])

        similarities = (q_vec @ self.vectors.T).flatten()

        if len(similarities) > k:
            top_k_idx = np.argpartition(similarities, -k)[-k:]
            top_k_idx = top_k_idx[np.argsort(similarities[top_k_idx])[::-1]]
        else:
            top_k_idx = np.argsort(similarities)[::-1]

        results = []
        for idx in top_k_idx:
            score = float(similarities[idx])
            if score > SIMILARITY_THRESHOLD:
                results.append({
                    'id': self.ids[idx],
                    'score': score,
                    'metadata': self.metadata[idx],
                })

        return results


class SemanticClassifier:

    TRAINING_DATA = {
        'count': [
            'how many claims', 'count of members', 'total number of encounters',
            'how many patients', 'number of prescriptions', 'count referrals',
            'how many denied claims', 'total claims', 'count by region',
        ],
        'aggregate': [
            'average paid amount', 'total billed amount', 'sum of costs',
            'average length of stay', 'mean risk score', 'max billed',
            'minimum copay', 'average cost by region', 'total revenue',
        ],
        'top_n': [
            'top 5 providers', 'top 10 members', 'top 3 diagnoses',
            'bottom 5 regions', 'first 10 claims', 'highest volume providers',
            'most expensive medications', 'largest departments',
        ],
        'which_most': [
            'which member has the most claims', 'which provider has highest',
            'which region has most denied', 'which specialty has highest average',
            'which medication is most prescribed', 'who has the most encounters',
        ],
        'trend': [
            'claims trend over time', 'monthly volume', 'trend over 2024',
            'month by month encounters', 'time series claims', 'yearly trend',
        ],
        'rate': [
            'denial rate by region', 'approval rate', 'readmission rate',
            'fill rate by pharmacy', 'completion rate', 'denial percentage',
        ],
        'filter': [
            'show me denied claims', 'members with risk score above',
            'claims over 5000', 'emergency encounters', 'chronic conditions',
            'inpatient visits', 'claims in 2024',
        ],
        'comparison': [
            'compare billed vs paid', 'billed versus paid by region',
            'compare costs across departments', 'side by side analysis',
        ],
        'having': [
            'members with more than 5 claims', 'providers with over 10 encounters',
            'departments with fewer than 100 visits', 'medications prescribed more than 50 times',
        ],
        'lookup': [
            'show me patient details', 'list all providers', 'get member info',
            'display claims data', 'show encounters', 'find prescription records',
        ],
    }

    def __init__(self):
        self.vectorizer = NumpyTFIDF(max_features=2000, ngram_range=NGRAM_RANGE)
        self.label_vectors: Dict[str, np.ndarray] = {}
        self._trained = False

    def train(self):
        all_texts = []
        all_labels = []
        for label, examples in self.TRAINING_DATA.items():
            for text in examples:
                all_texts.append(text)
                all_labels.append(label)

        vectors = self.vectorizer.fit_transform(all_texts)

        for label in self.TRAINING_DATA:
            indices = [i for i, l in enumerate(all_labels) if l == label]
            label_vecs = vectors[indices]
            centroid = label_vecs.mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            self.label_vectors[label] = centroid

        self._trained = True

    def classify(self, question: str) -> Tuple[str, float, Dict]:
        if not self._trained:
            self.train()

        q_vec = self.vectorizer.transform([question]).flatten()

        scores = {}
        for label, centroid in self.label_vectors.items():
            score = float(np.dot(q_vec, centroid))
            scores[label] = score

        best_label = max(scores, key=scores.get)
        best_score = scores[best_label]

        sorted_scores = sorted(scores.values(), reverse=True)
        margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]

        return best_label, best_score, {
            'all_scores': scores,
            'margin': margin,
            'method': 'numpy_tfidf_cosine',
        }


class SmartColumnResolver:

    def __init__(self, catalog_dir: str = None):
        self.columns: List[Dict] = []
        self.column_texts: List[str] = []
        self.vector_index = NumpyVectorIndex()
        self._loaded = False

        if catalog_dir:
            self._load_catalog(catalog_dir)

    def _load_catalog(self, catalog_dir: str):
        tables_dir = os.path.join(catalog_dir, 'tables')
        if not os.path.exists(tables_dir):
            return

        for fname in os.listdir(tables_dir):
            if not fname.endswith('.json'):
                continue
            try:
                with open(os.path.join(tables_dir, fname)) as f:
                    data = json.load(f)
                table_name = data.get('table_name', fname.replace('.json', ''))
                cols = data.get('columns', {})
                if isinstance(cols, dict):
                    cols = list(cols.values())

                for col in cols:
                    col_name = col.get('column_name', col.get('name', ''))
                    if not col_name:
                        continue

                    parts = [
                        col_name.lower().replace('_', ' '),
                        col.get('semantic_type', ''),
                        col.get('healthcare_type', ''),
                        table_name,
                    ]
                    text = ' '.join(p for p in parts if p)

                    self.columns.append({
                        'name': col_name,
                        'table': table_name,
                        'semantic_type': col.get('semantic_type', ''),
                        'healthcare_type': col.get('healthcare_type', ''),
                    })
                    self.column_texts.append(text)
            except (ValueError, KeyError) as e:
                logger.debug('Catalog load error: %s', e)

        if self.columns:
            items = [{'id': f"{c['table']}.{c['name']}", 'text': t, **c}
                     for c, t in zip(self.columns, self.column_texts)]
            self.vector_index.build(items, text_key='text')
            self._loaded = True

    def resolve(self, question: str, top_k: int = 10) -> List[Dict]:
        if not self._loaded:
            return []

        hits = self.vector_index.search(question, k=top_k)

        results = []
        for hit in hits:
            meta = hit['metadata']
            results.append({
                'column': meta['name'],
                'table': meta['table'],
                'score': hit['score'],
                'match_type': 'vector_similarity',
                'semantic_type': meta.get('semantic_type', ''),
            })

        words = re.findall(r'[a-z]+', question.lower())
        all_col_names = [c['name'].lower().replace('_', '') for c in self.columns]

        for word in words:
            if len(word) < 4:
                continue
            matches = difflib.get_close_matches(word, all_col_names, n=2, cutoff=0.7)
            for match in matches:
                idx = all_col_names.index(match)
                col = self.columns[idx]
                key = f"{col['table']}.{col['name']}"
                if not any(r['column'] == col['name'] and r['table'] == col['table'] for r in results):
                    results.append({
                        'column': col['name'],
                        'table': col['table'],
                        'score': 0.7,
                        'match_type': 'fuzzy_match',
                        'original_term': word,
                    })

        return results


class QueryValidator:

    @staticmethod
    def validate(results: List[Dict], question: str, sql: str) -> Dict[str, Any]:
        if not results:
            return {'valid': True, 'warnings': ['No results returned'], 'score': 0.5}

        df = pd.DataFrame(results)
        warnings = []
        score = 1.0

        q = question.lower()
        n_rows = len(df)
        n_cols = len(df.columns)

        if any(w in q for w in ['how many', 'total number', 'count of']) and ' by ' not in q and ' per ' not in q:
            if n_rows > 1:
                warnings.append(f'Expected 1 row for count query, got {n_rows}')
                score -= 0.2

        if re.search(r'which\s+\w+\s+(?:has|have|is)\s+(?:the\s+)?(?:most|highest|lowest)', q):
            if n_rows > 1:
                warnings.append(f'Expected 1 row for "which...most" query, got {n_rows}')
                score -= 0.2

        if 'GROUP BY' in sql.upper():
            group_cols = []
            group_match = re.search(r'GROUP BY\s+(.+?)(?:\s+ORDER|\s+HAVING|\s+LIMIT|;|$)', sql, re.IGNORECASE)
            if group_match:
                group_expr = group_match.group(1)
                first_col = df.columns[0]
                if df[first_col].duplicated().any():
                    warnings.append(f'Duplicate values in grouping column {first_col}')
                    score -= 0.3

        for col in df.select_dtypes(include=[np.number]).columns:
            col_lower = col.lower()
            values = df[col].dropna()
            if len(values) == 0:
                continue

            if any(k in col_lower for k in ['amount', 'paid', 'billed', 'cost', 'copay']):
                if (values < 0).any():
                    warnings.append(f'{col} has negative values (unexpected for currency)')
                    score -= 0.1

            if col_lower == 'count' or col_lower.startswith('total_'):
                if (values <= 0).any():
                    warnings.append(f'{col} has non-positive values')
                    score -= 0.1

            if any(k in col_lower for k in ['rate', 'pct', 'percent']):
                if (values > 100).any() or (values < 0).any():
                    warnings.append(f'{col} has values outside 0-100% range')
                    score -= 0.1

        top_match = re.search(r'top\s+(\d+)', q)
        if top_match:
            expected = int(top_match.group(1))
            if n_rows > expected:
                warnings.append(f'Expected top {expected} but got {n_rows} rows')
                score -= 0.1

        return {
            'valid': score >= 0.5,
            'score': max(0, score),
            'warnings': warnings,
            'stats': {
                'rows': n_rows,
                'columns': n_cols,
                'numeric_cols': len(df.select_dtypes(include=[np.number]).columns),
                'null_pct': float(df.isnull().sum().sum() / (n_rows * n_cols) * 100) if n_rows > 0 else 0,
            }
        }


class DataProfiler:

    @staticmethod
    def profile_table(df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
        profile = {
            'table': table_name,
            'rows': len(df),
            'columns': len(df.columns),
            'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'column_profiles': {},
        }

        for col in df.columns:
            col_data = df[col]
            col_profile = {
                'dtype': str(col_data.dtype),
                'null_count': int(col_data.isnull().sum()),
                'null_pct': float(col_data.isnull().mean() * 100),
                'unique_count': int(col_data.nunique()),
                'unique_pct': float(col_data.nunique() / len(df) * 100) if len(df) > 0 else 0,
            }

            numeric_col = pd.to_numeric(col_data, errors='coerce')
            if numeric_col.notna().sum() > len(df) * 0.5:
                col_profile['is_numeric'] = True
                col_profile['mean'] = float(numeric_col.mean())
                col_profile['median'] = float(numeric_col.median())
                col_profile['std'] = float(numeric_col.std())
                col_profile['min'] = float(numeric_col.min())
                col_profile['max'] = float(numeric_col.max())
                col_profile['skew'] = float(numeric_col.skew())

                q1 = numeric_col.quantile(0.25)
                q3 = numeric_col.quantile(0.75)
                iqr = q3 - q1
                outlier_count = int(((numeric_col < q1 - 1.5 * iqr) | (numeric_col > q3 + 1.5 * iqr)).sum())
                col_profile['outlier_count'] = outlier_count
                col_profile['outlier_pct'] = float(outlier_count / len(df) * 100) if len(df) > 0 else 0
            else:
                col_profile['is_numeric'] = False
                top_5 = col_data.value_counts().head(5)
                col_profile['top_values'] = {str(k): int(v) for k, v in top_5.items()}

            profile['column_profiles'][col] = col_profile

        null_score = 1.0 - (df.isnull().sum().sum() / (len(df) * len(df.columns))) if len(df) > 0 else 0
        unique_score = np.mean([p['unique_pct'] / 100 for p in profile['column_profiles'].values()])
        profile['quality_score'] = float((null_score * 0.6 + unique_score * 0.4))

        return profile

    @staticmethod
    def profile_database(db_path: str) -> Dict[str, Any]:
        conn = sqlite3.connect(db_path)
        all_tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
        valid_table_names = set(all_tables['name'].tolist())

        profiles = {}
        for table_name in all_tables['name']:
            if table_name not in valid_table_names:
                continue
            df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 10000", conn)
            profiles[table_name] = DataProfiler.profile_table(df, table_name)

        conn.close()

        total_rows = sum(p['rows'] for p in profiles.values())
        total_cols = sum(p['columns'] for p in profiles.values())
        avg_quality = np.mean([p['quality_score'] for p in profiles.values()])

        return {
            'tables': profiles,
            'summary': {
                'total_tables': len(profiles),
                'total_rows': total_rows,
                'total_columns': total_cols,
                'avg_quality_score': float(avg_quality),
            }
        }


class HybridNLPEngine:

    def __init__(self, catalog_dir: str = None):
        self.classifier = SemanticClassifier()
        self.column_resolver = SmartColumnResolver(catalog_dir)
        self.validator = QueryValidator()
        self.profiler = DataProfiler()

        self.schema_index = NumpyVectorIndex()
        self._initialized = False

        if catalog_dir:
            self.initialize(catalog_dir)

    def initialize(self, catalog_dir: str = None) -> Dict[str, Any]:
        self.classifier.train()

        if catalog_dir:
            items = self._build_schema_items(catalog_dir)
            if items:
                self.schema_index.build(items, text_key='text')

        self._initialized = True

        return {
            'classifier_intents': len(self.classifier.TRAINING_DATA),
            'column_resolver_columns': len(self.column_resolver.columns),
            'schema_index_items': len(self.schema_index.ids),
            'vectorizer_features': len(self.classifier.vectorizer.vocabulary_) if self.classifier.vectorizer.vocabulary_ else 0,
        }

    def _build_schema_items(self, catalog_dir: str) -> List[Dict]:
        items = []
        tables_dir = os.path.join(catalog_dir, 'tables')
        if not os.path.exists(tables_dir):
            return items

        for fname in os.listdir(tables_dir):
            if not fname.endswith('.json'):
                continue
            try:
                with open(os.path.join(tables_dir, fname)) as f:
                    data = json.load(f)
                table_name = data.get('table_name', '')

                items.append({
                    'id': f'table:{table_name}',
                    'text': f"{table_name} table healthcare data",
                    'type': 'table',
                    'name': table_name,
                })

                cols = data.get('columns', {})
                if isinstance(cols, dict):
                    cols = list(cols.values())
                for col in cols:
                    col_name = col.get('column_name', col.get('name', ''))
                    sem_type = col.get('semantic_type', '')
                    hc_type = col.get('healthcare_type', '')
                    items.append({
                        'id': f'col:{table_name}.{col_name}',
                        'text': f"{col_name} {sem_type} {hc_type} {table_name}".replace('_', ' '),
                        'type': 'column',
                        'table': table_name,
                        'name': col_name,
                    })
            except Exception:
                pass

        return items

    def analyze_question(self, question: str) -> Dict[str, Any]:
        if not self._initialized:
            self.initialize()

        intent, confidence, intent_details = self.classifier.classify(question)

        columns = self.column_resolver.resolve(question, top_k=8)

        schema_hits = self.schema_index.search(question, k=10)

        relevant_tables = []
        for hit in schema_hits:
            meta = hit['metadata']
            if meta.get('type') == 'table':
                relevant_tables.append(meta['name'])
            elif meta.get('type') == 'column' and meta.get('table'):
                if meta['table'] not in relevant_tables:
                    relevant_tables.append(meta['table'])

        return {
            'intent': intent,
            'intent_confidence': confidence,
            'intent_details': intent_details,
            'relevant_columns': columns,
            'relevant_schema': schema_hits[:5],
            'relevant_tables': relevant_tables[:3],
            'method': 'numpy_hybrid',
        }

    def validate_results(self, results: List[Dict], question: str, sql: str) -> Dict[str, Any]:
        return self.validator.validate(results, question, sql)


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    catalog_dir = os.path.join(os.path.dirname(script_dir), 'semantic_catalog')
    db_path = os.path.join(os.path.dirname(script_dir), 'data', 'healthcare_demo.db')

    print("=" * 60)
    print("NumPy-Powered NLP Engine — Hybrid Mode")
    print("=" * 60)

    engine = HybridNLPEngine(catalog_dir)
    stats = engine.initialize(catalog_dir)
    print(f"\nInitialized: {stats}")

    test_questions = [
        "how many claims",
        "which member has the most claims",
        "top 5 providers by total billed",
        "average paid amount by region",
        "claims trend over 2024",
        "denial rate by specialty",
        "show me denied claims",
        "members with more than 5 claims",
        "compare billed vs paid by region",
    ]

    print("\n--- Intent Classification ---")
    for q in test_questions:
        result = engine.analyze_question(q)
        print(f"  Q: {q}")
        print(f"    Intent: {result['intent']} ({result['intent_confidence']:.2f})")
        if result['relevant_columns']:
            top_col = result['relevant_columns'][0]
            print(f"    Top column: {top_col['table']}.{top_col['column']} ({top_col['score']:.2f})")
        print()

    if os.path.exists(db_path):
        print("\n--- Database Profile ---")
        profile = engine.profiler.profile_database(db_path)
        for table, p in profile['tables'].items():
            print(f"  {table}: {p['rows']} rows, {p['columns']} cols, quality: {p['quality_score']:.2f}")
        print(f"  Overall: {profile['summary']['total_rows']} rows, quality: {profile['summary']['avg_quality_score']:.2f}")
