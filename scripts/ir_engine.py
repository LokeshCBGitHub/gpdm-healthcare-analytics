import json
import math
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional, Any


STOPWORDS: Set[str] = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an',
    'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before',
    'being', 'below', 'between', 'both', 'but', 'by', 'can', 'could',
    'did', 'do', 'does', 'doing', 'down', 'during', 'each', 'few', 'for',
    'from', 'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
    'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
    'into', 'is', 'it', 'its', 'itself', 'just', 'me', 'might', 'more',
    'most', 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once',
    'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own',
    'same', 'she', 'should', 'so', 'some', 'such', 'than', 'that', 'the',
    'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these',
    'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until',
    'up', 'very', 'was', 'we', 'were', 'what', 'when', 'where', 'which',
    'while', 'who', 'whom', 'why', 'with', 'you', 'your', 'yours',
    'yourself', 'yourselves', 'is', 'are', 'am', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'will',
    'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall',
    'healthcare', 'data', 'table', 'column', 'field', 'value', 'type',
    'name', 'id', 'description', 'patient', 'medical', 'clinical',
}


class PorterStemmer:

    @staticmethod
    def _ends_with(word: str, suffix: str) -> bool:
        return word.endswith(suffix)

    @staticmethod
    def _measure(word: str) -> int:
        vowels = set('aeiouy')
        measure = 0
        is_vowel = False

        for char in word.lower():
            char_is_vowel = char in vowels
            if char_is_vowel and not is_vowel:
                measure += 1
            is_vowel = char_is_vowel

        return measure

    @staticmethod
    def _contains_vowel(word: str) -> bool:
        vowels = set('aeiouy')
        return any(c in vowels for c in word.lower())

    @classmethod
    def stem(cls, word: str) -> str:
        word = word.lower()

        if len(word) <= 2:
            return word

        if word.endswith('sses'):
            word = word[:-2]
        elif word.endswith('ies'):
            word = word[:-2] + 'i'
        elif word.endswith('ss'):
            pass
        elif word.endswith('s'):
            word = word[:-1]

        if word.endswith('eed'):
            stem = word[:-3]
            if cls._measure(stem) > 0:
                word = stem + 'ee'
        elif word.endswith('ed'):
            stem = word[:-2]
            if cls._contains_vowel(stem):
                word = stem
        elif word.endswith('ing'):
            stem = word[:-3]
            if cls._contains_vowel(stem):
                word = stem

        if len(word) > 1 and word.endswith('y'):
            stem = word[:-1]
            if cls._contains_vowel(stem):
                word = stem + 'i'

        measure = cls._measure(word)
        if measure > 0:
            for suffix, replacement in [
                ('ational', 'ate'), ('tional', 'tion'), ('enci', 'ence'),
                ('anci', 'ance'), ('izer', 'ize'), ('bli', 'ble'), ('alli', 'al'),
                ('entli', 'ent'), ('eli', 'e'), ('ousli', 'ous'), ('ization', 'ize'),
                ('ation', 'ate'), ('ator', 'ate'), ('alism', 'al'), ('iveness', 'ive'),
                ('fulness', 'ful'), ('ousness', 'ous'), ('aliti', 'al'),
            ]:
                if word.endswith(suffix):
                    word = word[:-len(suffix)] + replacement
                    break

        if measure > 0:
            for suffix, replacement in [
                ('icate', 'ic'), ('ative', ''), ('alize', 'al'),
            ]:
                if word.endswith(suffix):
                    word = word[:-len(suffix)] + replacement
                    break

        if cls._measure(word) > 1:
            for suffix in [
                'al', 'ance', 'ence', 'er', 'ic', 'able', 'ible', 'ant',
                'ement', 'ment', 'ent', 'ou', 'ion', 'ou', 'ism', 'ate', 'iti',
                'ous', 'ive', 'ize',
            ]:
                if word.endswith(suffix):
                    word = word[:-len(suffix)]
                    break

        if word.endswith('e'):
            stem = word[:-1]
            if cls._measure(stem) > 1:
                word = stem
            elif cls._measure(stem) == 1:
                if len(stem) >= 3:
                    if not (stem[-1] in 'wxY'):
                        consonants = 'bcdfghjklmnpqrstvwxyz'
                        if (stem[-1] in consonants and
                            stem[-2] not in 'aeiouy' and
                            stem[-3] in 'aeiouy'):
                            pass
                        else:
                            word = stem

        if (cls._measure(word) > 1 and
            len(word) >= 2 and word[-1] == 'l' and word[-2] in 'bcdfghjklmnpqrstvwxyz'):
            word = word[:-1]

        return word


class Tokenizer:

    def __init__(self, stemmer: Optional[PorterStemmer] = None):
        self.stemmer = stemmer or PorterStemmer()

    def tokenize(self, text: str) -> List[str]:
        text = text.lower()

        words = []
        current_word = []
        for char in text:
            if char.isalnum() or char in "-_'":
                current_word.append(char)
            else:
                if current_word:
                    words.append(''.join(current_word))
                    current_word = []
        if current_word:
            words.append(''.join(current_word))

        tokens = []
        for word in words:
            if word and word not in STOPWORDS and len(word) > 1:
                tokens.append(self.stemmer.stem(word))

        return tokens


class InvertedIndex:

    def __init__(self):
        self.index: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.doc_lengths: Dict[int, int] = {}
        self.num_docs = 0

    def add_document(self, doc_id: int, tokens: List[str]):
        self.doc_lengths[doc_id] = len(tokens)
        self.num_docs = max(self.num_docs, doc_id + 1)

        for token in tokens:
            self.index[token][doc_id] += 1

    def get_postings(self, term: str) -> Dict[int, int]:
        return dict(self.index.get(term, {}))

    def get_document_frequency(self, term: str) -> int:
        return len(self.index.get(term, {}))


class TFIDFScorer:

    def __init__(self, inverted_index: InvertedIndex):
        self.index = inverted_index

    def get_idf(self, term: str) -> float:
        df = self.index.get_document_frequency(term)
        if df == 0:
            return 0.0
        return math.log(self.index.num_docs / df)

    def get_tf(self, term: str, doc_id: int) -> float:
        postings = self.index.get_postings(term)
        if doc_id not in postings:
            return 0.0

        term_count = postings[doc_id]
        doc_length = self.index.doc_lengths.get(doc_id, 1)
        return term_count / max(doc_length, 1)

    def score_documents(self, query_tokens: List[str]) -> Dict[int, float]:
        query_vector: Dict[str, float] = {}
        for token in query_tokens:
            idf = self.get_idf(token)
            query_vector[token] = query_vector.get(token, 0) + idf

        query_mag = math.sqrt(sum(v ** 2 for v in query_vector.values()))
        if query_mag == 0:
            return {}

        doc_scores: Dict[int, float] = defaultdict(float)
        for token, query_weight in query_vector.items():
            postings = self.index.get_postings(token)
            idf = self.get_idf(token)

            for doc_id, term_freq in postings.items():
                doc_length = self.index.doc_lengths.get(doc_id, 1)
                tf = term_freq / max(doc_length, 1)
                doc_scores[doc_id] += tf * idf * query_weight

        for doc_id in doc_scores:
            doc_scores[doc_id] /= query_mag

        return dict(doc_scores)


class BM25Scorer:

    def __init__(self, inverted_index: InvertedIndex, k1: float = 1.5, b: float = 0.75):
        self.index = inverted_index
        self.k1 = k1
        self.b = b
        self._compute_avgdl()

    def _compute_avgdl(self):
        if self.index.num_docs == 0:
            self.avgdl = 0
        else:
            total_length = sum(self.index.doc_lengths.values())
            self.avgdl = total_length / self.index.num_docs

    def get_idf(self, term: str) -> float:
        df = self.index.get_document_frequency(term)
        if df == 0:
            return 0.0

        idf = math.log((self.index.num_docs - df + 0.5) / (df + 0.5) + 1.0)
        return idf

    def score_documents(self, query_tokens: List[str]) -> Dict[int, float]:
        doc_scores: Dict[int, float] = defaultdict(float)

        for token in query_tokens:
            idf = self.get_idf(token)
            postings = self.index.get_postings(token)

            for doc_id, term_freq in postings.items():
                doc_length = self.index.doc_lengths.get(doc_id, 1)

                numerator = term_freq * (self.k1 + 1)
                denominator = (term_freq +
                              self.k1 * (1 - self.b + self.b * (doc_length / self.avgdl)))

                doc_scores[doc_id] += idf * (numerator / denominator)

        return dict(doc_scores)


class DocumentStore:

    def __init__(self):
        self.documents: Dict[int, Dict[str, Any]] = {}
        self.next_id = 0

    def add_document(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        doc_id = self.next_id
        self.documents[doc_id] = {
            'content': content,
            'metadata': metadata or {}
        }
        self.next_id += 1
        return doc_id

    def get_document(self, doc_id: int) -> Optional[Dict[str, Any]]:
        return self.documents.get(doc_id)

    def get_metadata(self, doc_id: int) -> Optional[Dict[str, Any]]:
        doc = self.documents.get(doc_id)
        return doc['metadata'] if doc else None


class IREngine:

    def __init__(self, catalog_path: Optional[str] = None):
        self.tokenizer = Tokenizer()
        self.inverted_index = InvertedIndex()
        self.document_store = DocumentStore()
        self.tfidf_scorer = TFIDFScorer(self.inverted_index)
        self.bm25_scorer = BM25Scorer(self.inverted_index)

        self.table_columns: Dict[int, List[int]] = defaultdict(list)
        self.table_metadata: Dict[int, Dict[str, Any]] = {}

        if catalog_path:
            self.load_catalog(catalog_path)

    def load_catalog(self, catalog_path: str):
        tables_dir = Path(catalog_path) / 'tables'
        if not tables_dir.exists():
            return

        for json_file in tables_dir.glob('*.json'):
            with open(json_file, 'r') as f:
                try:
                    table_data = json.load(f)
                except json.JSONDecodeError:
                    continue

            table_name = table_data.get('name', json_file.stem)
            table_description = table_data.get('description', '')
            table_type = table_data.get('healthcare_type', '')

            table_content = f"{table_name} {table_description} {table_type}"
            table_id = self.document_store.add_document(
                table_content,
                {
                    'type': 'table',
                    'name': table_name,
                    'description': table_description,
                    'healthcare_type': table_type,
                    'columns': []
                }
            )
            self.table_metadata[table_id] = self.document_store.get_metadata(table_id)

            table_tokens = self.tokenizer.tokenize(table_content)
            self.inverted_index.add_document(table_id, table_tokens)

            columns = table_data.get('columns', [])
            for col in columns:
                col_name = col.get('name', '')
                col_description = col.get('description', '')
                col_type = col.get('type', '')
                col_values = ' '.join(col.get('top_values', [])[:5])

                col_content = f"{col_name} {col_description} {col_type} {col_values}"
                col_id = self.document_store.add_document(
                    col_content,
                    {
                        'type': 'column',
                        'table_id': table_id,
                        'table_name': table_name,
                        'name': col_name,
                        'description': col_description,
                        'data_type': col_type,
                        'top_values': col.get('top_values', [])
                    }
                )

                col_tokens = self.tokenizer.tokenize(col_content)
                self.inverted_index.add_document(col_id, col_tokens)

                self.table_columns[table_id].append(col_id)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float, Dict[str, Any]]]:
        query_tokens = self.tokenizer.tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25_scorer.score_documents(query_tokens)

        if not scores:
            scores = self.tfidf_scorer.score_documents(query_tokens)

        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            (doc_id, score, self.document_store.get_metadata(doc_id))
            for doc_id, score in sorted_results
        ]

    def find_relevant_tables(self, question: str, top_k: int = 5) -> List[Tuple[str, float]]:
        results = self.search(question, top_k=top_k * 2)

        seen_tables = set()
        table_results = []

        for doc_id, score, metadata in results:
            if metadata and metadata.get('type') == 'table':
                table_name = metadata.get('name', '')
                if table_name not in seen_tables:
                    table_results.append((table_name, score))
                    seen_tables.add(table_name)
                    if len(table_results) >= top_k:
                        break

        return table_results

    def find_relevant_columns(
        self, question: str, table: Optional[str] = None, top_k: int = 5
    ) -> List[Tuple[str, str, float]]:
        results = self.search(question, top_k=top_k * 3)

        seen_columns = set()
        col_results = []

        for doc_id, score, metadata in results:
            if metadata and metadata.get('type') == 'column':
                if table and metadata.get('table_name') != table:
                    continue

                col_key = (metadata.get('table_name'), metadata.get('name'))
                if col_key not in seen_columns:
                    col_results.append((
                        metadata.get('table_name', ''),
                        metadata.get('name', ''),
                        score
                    ))
                    seen_columns.add(col_key)
                    if len(col_results) >= top_k:
                        break

        return col_results

    def get_context(self, question: str, max_tokens: int = 2000) -> str:
        results = self.search(question, top_k=10)

        context_lines = []
        token_count = 0

        for doc_id, score, metadata in results:
            if not metadata:
                continue

            if metadata.get('type') == 'table':
                line = f"TABLE: {metadata.get('name')} - {metadata.get('description')} (score: {score:.3f})"
            elif metadata.get('type') == 'column':
                line = (f"COLUMN: {metadata.get('table_name')}.{metadata.get('name')} - "
                       f"{metadata.get('description')} (type: {metadata.get('data_type')}) "
                       f"(score: {score:.3f})")
            else:
                continue

            line_tokens = len(line) // 4
            if token_count + line_tokens > max_tokens:
                break

            context_lines.append(line)
            token_count += line_tokens

        return '\n'.join(context_lines)


def discover_config_path() -> Optional[str]:
    script_dir = Path(__file__).parent
    paramset_dir = script_dir.parent / 'paramset'

    if paramset_dir.exists() and paramset_dir.is_dir():
        return str(paramset_dir)

    return None


def discover_catalog_path() -> Optional[str]:
    script_dir = Path(__file__).parent

    catalog_dir = script_dir.parent / 'semantic_catalog'
    if catalog_dir.exists():
        return str(catalog_dir)

    catalog_dir = script_dir.parent / 'catalog'
    if catalog_dir.exists():
        return str(catalog_dir)

    return None


if __name__ == '__main__':
    config_path = discover_config_path()
    print(f"Config path: {config_path}")

    catalog_path = discover_catalog_path()
    print(f"Catalog path: {catalog_path}")

    engine = IREngine(catalog_path=catalog_path)
    print(f"Indexed {engine.inverted_index.num_docs} documents\n")

    test_queries = [
        "What tables contain patient demographics?",
        "Show me all healthcare-related columns",
        "Find tables with medical encounter information",
        "Which columns describe clinical diagnoses?",
        "Search for patient identifier fields",
        "What tables have medication data?",
        "Find columns related to lab results",
        "Show tables with visit information",
        "Which columns contain patient vital signs?",
        "Find date-related columns for appointments",
    ]

    print("=" * 80)
    print("TEST QUERIES AND RESULTS")
    print("=" * 80)

    for i, query in enumerate(test_queries, 1):
        print(f"\n[Query {i}] {query}")
        print("-" * 80)

        results = engine.search(query, top_k=5)
        if results:
            for doc_id, score, metadata in results:
                if metadata:
                    if metadata.get('type') == 'table':
                        print(f"  TABLE: {metadata.get('name')} ({score:.4f})")
                    elif metadata.get('type') == 'column':
                        print(f"  COLUMN: {metadata.get('table_name')}.{metadata.get('name')} ({score:.4f})")
        else:
            print("  No results found")

    print("\n" + "=" * 80)
    print("CONTEXT EXAMPLE")
    print("=" * 80)
    example_query = "patient medical history"
    context = engine.get_context(example_query, max_tokens=500)
    print(f"Query: {example_query}")
    print(f"Context:\n{context}")
