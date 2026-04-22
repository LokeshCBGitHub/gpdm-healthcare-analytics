import math
import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass
import json


@dataclass
class ProcessedQuery:
    original: str
    tokens: List[str]
    stems: List[str]
    lemmas: List[str]
    entities: List[Tuple[str, str]]
    expanded_terms: Set[str]
    ngrams: Dict[int, List[str]]
    bm25_scores: Dict[str, float]


class PorterStemmer:

    def __init__(self):
        self.consonant_pattern = "[^aeiou]"
        self.vowel_pattern = "[aeiouy]"

    def _is_consonant(self, word: str, i: int) -> bool:
        if i < 0 or i >= len(word):
            return False
        return word[i] not in "aeiouy"

    def _measure(self, word: str) -> int:
        cv_sequence = ""
        for char in word:
            is_vowel = char in "aeiouy"
            if not cv_sequence or (is_vowel != (cv_sequence[-1] == 'v')):
                cv_sequence += 'v' if is_vowel else 'c'
        return cv_sequence.count('vc')

    def _ends_with_double(self, word: str, char: str) -> bool:
        return len(word) >= 2 and word[-2:] == char * 2

    def _is_short_syllable(self, word: str) -> bool:
        if len(word) == 2:
            return self._is_consonant(word, 0) and word[1] in "aeiouy"
        if len(word) == 3:
            return (self._is_consonant(word, 0) and word[1] in "aeiouy" and
                    self._is_consonant(word, 2))
        return False

    def _is_short_word(self, word: str) -> bool:
        return self._measure(word) == 1 and self._is_short_syllable(word)

    def stem(self, word: str) -> str:
        if len(word) <= 2:
            return word

        word = word.lower()

        if word.endswith("sses"):
            word = word[:-2]
        elif word.endswith("ies"):
            word = word[:-2] + "i"
        elif word.endswith("ss"):
            pass
        elif word.endswith("s") and len(word) > 1:
            if word[-2] not in "aeioulmnrst":
                word = word[:-1]

        removed_step1b = False
        if word.endswith("eed"):
            stem = word[:-3]
            if self._measure(stem) > 0:
                word = word[:-1]
        elif word.endswith("ed") and self._measure(word[:-2]) > 0:
            word = word[:-2]
            removed_step1b = True
        elif word.endswith("ing") and self._measure(word[:-3]) > 0:
            word = word[:-3]
            removed_step1b = True

        if removed_step1b:
            if word.endswith(("at", "bl", "iz")):
                word += "e"
            elif len(word) > 1 and self._ends_with_double(word, word[-1]):
                if word[-1] not in "lsz":
                    word = word[:-1]
            elif self._is_short_word(word):
                word += "e"

        if len(word) > 1 and word[-1] in "wy" and word[-2] not in "aeiouy":
            if word[-1] == "y":
                word = word[:-1] + "i"

        step2_rules = [
            ("ational", "ate"), ("tional", "tion"), ("enci", "ence"),
            ("anci", "ance"), ("izer", "ize"), ("bli", "ble"), ("alli", "al"),
            ("entli", "ent"), ("eli", "e"), ("ousli", "ous"), ("ization", "ize"),
            ("ation", "ate"), ("ator", "ate"), ("alism", "al"), ("iveness", "ive"),
            ("fulness", "ful"), ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"),
            ("biliti", "ble"), ("logi", "log")
        ]

        for ending, replacement in step2_rules:
            if word.endswith(ending):
                stem = word[:-len(ending)]
                if self._measure(stem) > 0:
                    word = stem + replacement
                break

        step3_rules = [
            ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
            ("ical", "ic"), ("ful", ""), ("ness", "")
        ]

        for ending, replacement in step3_rules:
            if word.endswith(ending):
                stem = word[:-len(ending)]
                if self._measure(stem) > 0:
                    word = stem + replacement
                break

        step4_rules = [
            "al", "ance", "ence", "er", "ic", "able", "ible", "ant",
            "ement", "ment", "ent", "ou", "ion", "ou", "ism", "ate", "iti",
            "ous", "ive", "ize"
        ]

        for ending in step4_rules:
            if word.endswith(ending):
                stem = word[:-len(ending)]
                measure = self._measure(stem)
                if ending == "ion":
                    if len(stem) > 0 and stem[-1] in "st" and measure > 0:
                        word = stem
                elif measure > 1:
                    word = stem
                break

        if word.endswith("e"):
            stem = word[:-1]
            measure = self._measure(stem)
            if measure > 1:
                word = stem
            elif measure == 1 and not self._is_short_syllable(stem):
                word = stem

        if self._measure(word) > 1 and word.endswith("ll"):
            word = word[:-1]

        return word


class RuleBasedLemmatizer:

    def __init__(self):
        self.irregular_forms = {
            "diagnoses": "diagnosis",
            "analyses": "analysis",
            "syntheses": "synthesis",
            "hypotheses": "hypothesis",
            "prognoses": "prognosis",
            "radiographies": "radiography",
            "therapies": "therapy",
            "surgeries": "surgery",
            "procedures": "procedure",
            "treatments": "treatment",
            "medications": "medication",
            "prescriptions": "prescription",
            "infections": "infection",
            "conditions": "condition",
            "symptoms": "symptom",
            "tests": "test",
            "examinations": "examination",
            "visits": "visit",
            "patients": "patient",
            "providers": "provider",
            "doctors": "doctor",
            "nurses": "nurse",
            "physician": "physician",
            "specialists": "specialist",
            "allergies": "allergy",
            "adverse": "adverse",
            "events": "event",
            "hospitalization": "hospitalization",
            "was": "be",
            "were": "be",
            "been": "be",
            "being": "be",
            "am": "be",
            "is": "be",
            "are": "be",
            "feet": "foot",
            "teeth": "tooth",
            "children": "child",
            "had": "have",
            "has": "have",
            "does": "do",
            "did": "do",
            "goes": "go",
            "went": "go",
        }
        self.suffix_rules = [
            ("ies", "y"), ("es", ""), ("s", ""),
            ("ed", ""), ("ing", ""), ("ly", "")
        ]

    def lemmatize(self, word: str) -> str:
        word_lower = word.lower()

        if word_lower in self.irregular_forms:
            return self.irregular_forms[word_lower]

        for suffix, replacement in self.suffix_rules:
            if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 1:
                return word_lower[:-len(suffix)] + replacement

        return word_lower


class StopWords:

    def __init__(self):
        self.words = {
            "a", "an", "and", "are", "as", "at", "be", "but", "by",
            "for", "if", "in", "into", "is", "it", "no", "not",
            "of", "on", "or", "such", "that", "the", "their", "then",
            "there", "these", "they", "this", "to", "was", "will",
            "with", "would", "about", "above", "after", "again",
            "against", "all", "am", "any", "before", "being", "below",
            "between", "both", "can", "could", "did", "do", "does",
            "down", "during", "each", "few", "from", "further", "had",
            "has", "have", "he", "her", "here", "hers", "herself", "him",
            "himself", "his", "how", "i", "me", "my", "myself", "nor",
            "our", "ours", "ourselves", "out", "over", "same", "should",
            "so", "some", "than", "too", "under", "until", "up", "very",
            "we", "were", "what", "when", "where", "which", "while",
            "who", "whom", "why", "you", "your", "yours", "yourself",
            "yourselves",
            "patient", "patients", "case", "cases", "study", "studies",
            "report", "reports", "finding", "findings", "result", "results",
            "data", "method", "methods", "procedure", "procedures",
            "treatment", "treatments", "therapy", "therapies", "test", "tests",
        }

    def is_stopword(self, word: str) -> bool:
        return word.lower() in self.words

    def remove(self, tokens: List[str]) -> List[str]:
        return [t for t in tokens if not self.is_stopword(t)]


class NGramGenerator:

    @staticmethod
    def generate(tokens: List[str], n: int) -> List[str]:
        if n > len(tokens) or n < 1:
            return []
        return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    @staticmethod
    def generate_all(tokens: List[str], max_n: int = 3) -> Dict[int, List[str]]:
        ngrams = {}
        for n in range(1, min(max_n + 1, len(tokens) + 1)):
            ngrams[n] = NGramGenerator.generate(tokens, n)
        return ngrams


class HealthcareNER:

    def __init__(self):
        self.icd10_pattern = re.compile(r'\b[A-Z]\d{2}(?:\.\d{1,2})?\b')
        self.medication_keywords = {
            "aspirin", "ibuprofen", "acetaminophen", "metformin", "lisinopril",
            "atorvastatin", "levothyroxine", "omeprazole", "amoxicillin",
            "penicillin", "insulin", "testosterone", "estrogen", "warfarin"
        }
        self.provider_keywords = {
            "dr.", "dr", "md", "phd", "rn", "pa", "np", "physician",
            "surgeon", "cardiologist", "neurologist", "specialist"
        }
        self.diagnosis_keywords = {
            "hypertension", "diabetes", "cancer", "infection", "pneumonia",
            "myocardial", "stroke", "asthma", "arrhythmia", "fibrillation",
            "hepatitis", "arthritis", "fracture", "dislocation", "edema"
        }
        self.date_pattern = re.compile(
            r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4})\b',
            re.IGNORECASE
        )
        self.dollar_pattern = re.compile(r'\$[\d,]+(?:\.\d{2})?')

    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        entities = []

        for match in self.icd10_pattern.finditer(text):
            entities.append((match.group(), "ICD_CODE"))

        for match in self.dollar_pattern.finditer(text):
            entities.append((match.group(), "DOLLAR_AMOUNT"))

        for match in self.date_pattern.finditer(text):
            entities.append((match.group(), "DATE"))

        tokens = text.lower().split()
        for token in tokens:
            clean_token = token.strip('.,;:!?')
            if clean_token in self.medication_keywords:
                entities.append((token, "MEDICATION"))
            elif clean_token in self.provider_keywords:
                entities.append((token, "PROVIDER"))
            elif clean_token in self.diagnosis_keywords:
                entities.append((token, "DIAGNOSIS"))

        return entities


class EditDistanceSpellCorrector:

    def __init__(self, vocabulary: Optional[Set[str]] = None):
        self.healthcare_vocab = {
            "hypertension", "diabetes", "medication", "diagnosis", "treatment",
            "patient", "provider", "hospital", "examination", "procedure",
            "symptoms", "infection", "therapy", "surgery", "prescription",
            "allergies", "arrhythmia", "pneumonia", "cardiologist", "surgeon",
            "anesthesia", "radiography", "ultrasound", "tomography", "biopsy",
            "pathology", "microbiology", "pharmacy", "neurology", "cardiology"
        }
        if vocabulary:
            self.vocabulary = vocabulary.union(self.healthcare_vocab)
        else:
            self.vocabulary = self.healthcare_vocab

    def _edits_one(self, word: str) -> Set[str]:
        letters = "abcdefghijklmnopqrstuvwxyz"
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def _edits_two(self, word: str) -> Set[str]:
        return set(e2 for e1 in self._edits_one(word) for e2 in self._edits_one(e1))

    def correct(self, word: str, max_edits: int = 2) -> Optional[str]:
        if word in self.vocabulary:
            return word

        candidates = set()

        if max_edits >= 1:
            candidates.update(self._edits_one(word) & self.vocabulary)

        if max_edits >= 2 and not candidates:
            candidates.update(self._edits_two(word) & self.vocabulary)

        if candidates:
            return min(candidates, key=lambda c: (len(c), c))

        return None


class BM25Scorer:

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.document_lengths = {}
        self.idf_scores = {}
        self.num_docs = 0
        self.avg_doc_length = 0.0

    def build_index(self, documents: Dict[str, List[str]]) -> None:
        self.num_docs = len(documents)
        self.document_lengths = {doc_id: len(tokens) for doc_id, tokens in documents.items()}
        self.avg_doc_length = sum(self.document_lengths.values()) / max(self.num_docs, 1)

        doc_frequencies = defaultdict(int)
        for tokens in documents.values():
            for token in set(tokens):
                doc_frequencies[token] += 1

        for term, freq in doc_frequencies.items():
            self.idf_scores[term] = math.log((self.num_docs - freq + 0.5) / (freq + 0.5) + 1.0)

    def score_document(self, document_id: str, tokens: List[str], query_tokens: List[str]) -> float:
        score = 0.0
        doc_length = self.document_lengths.get(document_id, 0)
        term_freqs = Counter(tokens)

        for term in query_tokens:
            if term not in self.idf_scores:
                continue

            idf = self.idf_scores[term]
            tf = term_freqs.get(term, 0)

            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))

            score += idf * (numerator / denominator)

        return score


class TFIDFScorer:

    def __init__(self):
        self.idf_scores = {}
        self.num_docs = 0

    def build_index(self, documents: Dict[str, List[str]]) -> None:
        self.num_docs = len(documents)
        doc_frequencies = defaultdict(int)

        for tokens in documents.values():
            for term in set(tokens):
                doc_frequencies[term] += 1

        for term, freq in doc_frequencies.items():
            self.idf_scores[term] = math.log(self.num_docs / freq) if freq > 0 else 0.0

    def score_term(self, term: str, term_freq: int) -> float:
        sublinear_tf = 1 + math.log(term_freq) if term_freq > 0 else 0
        idf = self.idf_scores.get(term, 0.0)
        return sublinear_tf * idf


class HealthcareOntology:

    def __init__(self):
        self.synonyms = {
            "hypertension": {"high blood pressure", "elevated bp", "htn"},
            "diabetes": {"hyperglycemia", "diabetes mellitus", "dm"},
            "myocardial": {"heart attack", "mi", "infarction"},
            "pneumonia": {"lung infection", "respiratory infection"},
            "arrhythmia": {"irregular heartbeat", "dysrhythmia"},
            "medication": {"drug", "pharmaceutical", "medicine"},
            "diagnosis": {"condition", "disease", "disorder"},
            "treatment": {"therapy", "intervention", "management"},
            "patient": {"individual", "subject", "person"},
            "provider": {"physician", "doctor", "clinician", "specialist"},
            "surgery": {"surgical procedure", "operation"},
            "infection": {"bacterial", "viral", "microbial", "pathogenic"},
            "examination": {"assessment", "evaluation", "check-up"},
        }

    def expand_query(self, terms: List[str]) -> Set[str]:
        expanded = set(terms)
        for term in terms:
            term_lower = term.lower()
            if term_lower in self.synonyms:
                expanded.update(self.synonyms[term_lower])
        return expanded


class RocchioRelevanceFeedback:

    def __init__(self, alpha: float = 1.0, beta: float = 0.75, gamma: float = 0.15):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def refine_query(
        self,
        original_query: List[str],
        relevant_docs: List[Dict[str, float]],
        non_relevant_docs: List[Dict[str, float]]
    ) -> List[Tuple[str, float]]:
        refined = defaultdict(float)

        for term in original_query:
            refined[term] += self.alpha

        if relevant_docs:
            for doc_vector in relevant_docs:
                for term, score in doc_vector.items():
                    refined[term] += self.beta * score / len(relevant_docs)

        if non_relevant_docs:
            for doc_vector in non_relevant_docs:
                for term, score in doc_vector.items():
                    refined[term] -= self.gamma * score / len(non_relevant_docs)

        return sorted(refined.items(), key=lambda x: x[1], reverse=True)


@dataclass
class QRel:
    query_id: str
    document_id: str
    relevance_score: int


class QueryRelevanceJudgments:

    def __init__(self):
        self.qrels: Dict[str, List[QRel]] = defaultdict(list)

    def add_judgment(self, query_id: str, document_id: str, relevance_score: int) -> None:
        qrel = QRel(query_id, document_id, relevance_score)
        self.qrels[query_id].append(qrel)

    def get_relevant_docs(self, query_id: str, threshold: int = 1) -> Set[str]:
        return {qrel.document_id for qrel in self.qrels.get(query_id, [])
                if qrel.relevance_score >= threshold}


class RetrievalMetrics:

    @staticmethod
    def precision(retrieved: List[str], relevant: Set[str]) -> float:
        if not retrieved:
            return 0.0
        return len(set(retrieved) & relevant) / len(retrieved)

    @staticmethod
    def recall(retrieved: List[str], relevant: Set[str]) -> float:
        if not relevant:
            return 0.0
        return len(set(retrieved) & relevant) / len(relevant)

    @staticmethod
    def f1_score(precision: float, recall: float) -> float:
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def average_precision(retrieved: List[str], relevant: Set[str]) -> float:
        if not relevant:
            return 0.0

        score = 0.0
        num_relevant = 0

        for i, doc in enumerate(retrieved):
            if doc in relevant:
                num_relevant += 1
                precision_at_i = num_relevant / (i + 1)
                score += precision_at_i

        return score / len(relevant)

    @staticmethod
    def ndcg(retrieved: List[str], relevant: Set[str], k: int = 10) -> float:
        retrieved_k = retrieved[:k]

        dcg = 0.0
        for i, doc in enumerate(retrieved_k):
            rel = 1 if doc in relevant else 0
            dcg += rel / math.log2(i + 2)

        idcg = 0.0
        for i in range(min(len(relevant), k)):
            idcg += 1 / math.log2(i + 2)

        return dcg / idcg if idcg > 0 else 0.0


class NaiveBayesClassifier:

    def __init__(self, smoothing: float = 1.0):
        self.smoothing = smoothing
        self.class_counts = defaultdict(int)
        self.term_class_counts = defaultdict(lambda: defaultdict(int))
        self.vocabulary = set()
        self.class_priors = {}
        self.num_docs = 0

    def train(self, documents: List[Tuple[List[str], str]]) -> None:
        self.num_docs = len(documents)

        for tokens, label in documents:
            self.class_counts[label] += 1
            for token in tokens:
                self.term_class_counts[label][token] += 1
                self.vocabulary.add(token)

        for label in self.class_counts:
            self.class_priors[label] = self.class_counts[label] / self.num_docs

    def predict(self, tokens: List[str]) -> Tuple[str, float]:
        if not self.class_priors:
            return "unknown", 0.0

        scores = {}
        for label in self.class_priors:
            score = math.log(self.class_priors[label])

            for token in tokens:
                term_count = self.term_class_counts[label].get(token, 0)
                class_total = sum(self.term_class_counts[label].values())
                vocab_size = len(self.vocabulary)

                probability = (term_count + self.smoothing) / (class_total + self.smoothing * vocab_size)
                score += math.log(probability)

            scores[label] = score

        best_label = max(scores, key=scores.get)
        confidence = math.exp(scores[best_label])
        return best_label, confidence


class LanguageModel:

    def __init__(self, lambda_param: float = 0.5):
        self.lambda_param = lambda_param
        self.term_counts = defaultdict(int)
        self.total_count = 0

    def build_model(self, tokens: List[str]) -> None:
        for token in tokens:
            self.term_counts[token] += 1
            self.total_count += 1

    def probability(self, term: str) -> float:
        if self.total_count == 0:
            return 0.0

        ml_prob = self.term_counts[term] / self.total_count
        collection_prob = 1.0 / len(self.term_counts) if self.term_counts else 0.0

        return self.lambda_param * ml_prob + (1 - self.lambda_param) * collection_prob


class DirichletSmoothing:

    def __init__(self, mu: float = 2000.0):
        self.mu = mu
        self.term_counts = defaultdict(int)
        self.collection_counts = defaultdict(int)
        self.collection_total = 0

    def probability(self, term: str, doc_length: int) -> float:
        term_count = self.term_counts[term]
        collection_count = self.collection_counts[term]

        if collection_count == 0 and term_count == 0:
            return 0.0

        numerator = term_count + self.mu * (collection_count / max(self.collection_total, 1))
        denominator = doc_length + self.mu

        return numerator / denominator


class MutualInformation:

    @staticmethod
    def calculate(
        term_class_counts: Dict[Tuple[str, str], int],
        term_counts: Dict[str, int],
        class_counts: Dict[str, int],
        total: int
    ) -> Dict[str, float]:
        mi_scores = {}

        for term in term_counts:
            mi = 0.0
            for cls in class_counts:
                count = term_class_counts.get((term, cls), 0)

                p_term = term_counts[term] / total
                p_class = class_counts[cls] / total
                p_term_class = count / total if count > 0 else 1e-10

                if p_term > 0 and p_class > 0 and p_term_class > 0:
                    mi += p_term_class * math.log(p_term_class / (p_term * p_class))

            mi_scores[term] = mi

        return mi_scores


class ChiSquareTest:

    @staticmethod
    def calculate(
        term: str,
        cls: str,
        term_class_count: int,
        term_count: int,
        class_count: int,
        total: int
    ) -> float:
        n11 = term_class_count
        n10 = term_count - term_class_count
        n01 = class_count - term_class_count
        n00 = total - term_count - class_count + term_class_count

        numerator = total * (n11 * n00 - n10 * n01) ** 2
        denominator = (n11 + n10) * (n11 + n01) * (n10 + n00) * (n01 + n00)

        if denominator == 0:
            return 0.0

        return numerator / denominator


class PointwiseMutualInformation:

    def __init__(self):
        self.bigram_counts = defaultdict(int)
        self.unigram_counts = defaultdict(int)
        self.total = 0

    def build_from_ngrams(self, bigrams: List[str]) -> None:
        for bigram in bigrams:
            self.bigram_counts[bigram] += 1
            self.total += 1

            terms = bigram.split()
            if len(terms) == 2:
                self.unigram_counts[terms[0]] += 1
                self.unigram_counts[terms[1]] += 1

    def calculate(self, term1: str, term2: str) -> float:
        if self.total == 0:
            return 0.0

        bigram = f"{term1} {term2}"
        p_bigram = self.bigram_counts[bigram] / self.total
        p_term1 = self.unigram_counts[term1] / self.total
        p_term2 = self.unigram_counts[term2] / self.total

        if p_bigram == 0 or p_term1 == 0 or p_term2 == 0:
            return 0.0

        return math.log(p_bigram / (p_term1 * p_term2))


class LogisticRegression:

    def __init__(self, learning_rate: float = 0.01, iterations: int = 100):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = {}
        self.bias = 0.0

    def _sigmoid(self, x: float) -> float:
        try:
            return 1 / (1 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    def train(self, features: List[Dict[str, float]], labels: List[int]) -> None:
        if not features or not labels:
            return

        all_keys = set()
        for feat_dict in features:
            all_keys.update(feat_dict.keys())

        self.weights = {key: 0.0 for key in all_keys}

        for _ in range(self.iterations):
            for feature_dict, label in zip(features, labels):
                logits = self.bias
                for key, value in feature_dict.items():
                    logits += self.weights[key] * value

                prediction = self._sigmoid(logits)
                error = prediction - label

                self.bias -= self.learning_rate * error
                for key, value in feature_dict.items():
                    self.weights[key] -= self.learning_rate * error * value

    def predict(self, feature_dict: Dict[str, float]) -> float:
        logits = self.bias
        for key, value in feature_dict.items():
            logits += self.weights.get(key, 0.0) * value
        return self._sigmoid(logits)


class DecisionTree:

    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth
        self.tree = None

    def _gini(self, labels: List[int]) -> float:
        if not labels:
            return 0.0
        counts = Counter(labels)
        total = len(labels)
        return 1.0 - sum((count / total) ** 2 for count in counts.values())

    def _find_best_split(
        self,
        features: List[Dict[str, float]],
        labels: List[int],
        feature_names: Set[str]
    ) -> Tuple[str, float, float]:
        best_gini = float('inf')
        best_feature = None
        best_threshold = None

        for feature in feature_names:
            values = sorted(set(f.get(feature, 0.0) for f in features))

            for i in range(len(values) - 1):
                threshold = (values[i] + values[i + 1]) / 2

                left_labels = [labels[j] for j, f in enumerate(features)
                              if f.get(feature, 0.0) <= threshold]
                right_labels = [labels[j] for j, f in enumerate(features)
                               if f.get(feature, 0.0) > threshold]

                if not left_labels or not right_labels:
                    continue

                gini = (len(left_labels) * self._gini(left_labels) +
                       len(right_labels) * self._gini(right_labels)) / len(labels)

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gini

    def train(self, features: List[Dict[str, float]], labels: List[int]) -> None:
        if not features or not labels:
            return

        feature_names = set()
        for feat_dict in features:
            feature_names.update(feat_dict.keys())

        self._build_tree(features, labels, feature_names, depth=0)

    def _build_tree(self, features, labels, feature_names, depth):
        if depth >= self.max_depth or len(set(labels)) <= 1:
            counts = Counter(labels)
            return max(counts, key=counts.get) if counts else 0

        best_feature, best_threshold, _ = self._find_best_split(features, labels, feature_names)

        if best_feature is None:
            counts = Counter(labels)
            return max(counts, key=counts.get) if counts else 0

        left_idx = [i for i, f in enumerate(features) if f.get(best_feature, 0.0) <= best_threshold]
        right_idx = [i for i, f in enumerate(features) if f.get(best_feature, 0.0) > best_threshold]

        left_tree = self._build_tree(
            [features[i] for i in left_idx],
            [labels[i] for i in left_idx],
            feature_names,
            depth + 1
        )

        right_tree = self._build_tree(
            [features[i] for i in right_idx],
            [labels[i] for i in right_idx],
            feature_names,
            depth + 1
        )

        self.tree = {
            "feature": best_feature,
            "threshold": best_threshold,
            "left": left_tree,
            "right": right_tree
        }

    def predict(self, feature_dict: Dict[str, float]) -> int:
        if not self.tree:
            return 0

        node = self.tree
        while isinstance(node, dict):
            if feature_dict.get(node["feature"], 0.0) <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]

        return node


class KMeansClustering:

    def __init__(self, k: int = 3, max_iterations: int = 100):
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = []
        self.clusters = []

    @staticmethod
    def _euclidean_distance(point1: Dict[str, float], point2: Dict[str, float]) -> float:
        all_keys = set(point1.keys()) | set(point2.keys())
        return math.sqrt(sum((point1.get(k, 0.0) - point2.get(k, 0.0)) ** 2 for k in all_keys))

    def train(self, points: List[Dict[str, float]]) -> None:
        if len(points) < self.k:
            self.k = len(points)

        import random
        self.centroids = random.sample(points, self.k)

        for _ in range(self.max_iterations):
            self.clusters = [[] for _ in range(self.k)]
            for point in points:
                distances = [self._euclidean_distance(point, centroid) for centroid in self.centroids]
                cluster_id = distances.index(min(distances))
                self.clusters[cluster_id].append(point)

            new_centroids = []
            for cluster in self.clusters:
                if not cluster:
                    new_centroids.append(random.choice(points))
                else:
                    all_keys = set()
                    for point in cluster:
                        all_keys.update(point.keys())

                    centroid = {}
                    for key in all_keys:
                        centroid[key] = sum(p.get(key, 0.0) for p in cluster) / len(cluster)
                    new_centroids.append(centroid)

            if all(self._euclidean_distance(old, new) < 1e-6 for old, new in zip(self.centroids, new_centroids)):
                break

            self.centroids = new_centroids

    def predict(self, point: Dict[str, float]) -> int:
        if not self.centroids:
            return 0
        distances = [self._euclidean_distance(point, centroid) for centroid in self.centroids]
        return distances.index(min(distances))


class CosineSimilarityMatrix:

    @staticmethod
    def _dot_product(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        return sum(vec1.get(k, 0.0) * vec2.get(k, 0.0) for k in set(vec1.keys()) & set(vec2.keys()))

    @staticmethod
    def _magnitude(vec: Dict[str, float]) -> float:
        return math.sqrt(sum(v ** 2 for v in vec.values()))

    @staticmethod
    def similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        dot_product = CosineSimilarityMatrix._dot_product(vec1, vec2)
        mag1 = CosineSimilarityMatrix._magnitude(vec1)
        mag2 = CosineSimilarityMatrix._magnitude(vec2)

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot_product / (mag1 * mag2)

    @staticmethod
    def matrix(vectors: List[Dict[str, float]]) -> List[List[float]]:
        n = len(vectors)
        matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(i, n):
                sim = CosineSimilarityMatrix.similarity(vectors[i], vectors[j])
                matrix[i][j] = sim
                matrix[j][i] = sim

        return matrix


class AdvancedNLPEngine:

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.porter_stemmer = PorterStemmer()
        self.lemmatizer = RuleBasedLemmatizer()
        self.stopwords = StopWords()
        self.ner = HealthcareNER()
        self.spell_corrector = EditDistanceSpellCorrector()
        self.bm25 = BM25Scorer(
            k1=self.config.get("bm25_k1", 1.5),
            b=self.config.get("bm25_b", 0.75)
        )
        self.tfidf = TFIDFScorer()
        self.ontology = HealthcareOntology()
        self.rocchio = RocchioRelevanceFeedback()
        self.qrels = QueryRelevanceJudgments()
        self.naive_bayes = NaiveBayesClassifier()
        self.language_model = LanguageModel()
        self.pmi = PointwiseMutualInformation()

    def process_query(self, text: str) -> ProcessedQuery:
        tokens = [t.lower() for t in re.findall(r'\b\w+\b', text)]

        stems = [self.porter_stemmer.stem(t) for t in tokens]
        lemmas = [self.lemmatizer.lemmatize(t) for t in tokens]

        entities = self.ner.extract_entities(text)

        filtered_tokens = self.stopwords.remove(tokens)

        expanded_terms = self.ontology.expand_query(filtered_tokens)

        ngrams = NGramGenerator.generate_all(filtered_tokens)

        bm25_scores = {term: 0.0 for term in filtered_tokens}

        return ProcessedQuery(
            original=text,
            tokens=tokens,
            stems=stems,
            lemmas=lemmas,
            entities=entities,
            expanded_terms=expanded_terms,
            ngrams=ngrams,
            bm25_scores=bm25_scores
        )

    def rank_columns(self, query: str, columns: Dict[str, List[str]]) -> List[Tuple[str, float]]:
        processed = self.process_query(query)
        query_tokens = self.stopwords.remove(processed.stems)

        self.bm25.build_index(columns)
        self.tfidf.build_index(columns)

        scores = {}
        for col_id, col_tokens in columns.items():
            bm25_score = self.bm25.score_document(col_id, col_tokens, query_tokens)

            tfidf_score = 0.0
            term_freqs = Counter(col_tokens)
            for term in query_tokens:
                tfidf_score += self.tfidf.score_term(term, term_freqs.get(term, 0))

            scores[col_id] = bm25_score + tfidf_score

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def evaluate_retrieval(
        self,
        query: str,
        results: List[str],
        qrels: Optional[QueryRelevanceJudgments] = None,
        query_id: str = "q1"
    ) -> Dict[str, float]:
        if not qrels:
            qrels = self.qrels

        relevant_docs = qrels.get_relevant_docs(query_id)

        precision = RetrievalMetrics.precision(results, relevant_docs)
        recall = RetrievalMetrics.recall(results, relevant_docs)
        f1 = RetrievalMetrics.f1_score(precision, recall)
        map_score = RetrievalMetrics.average_precision(results, relevant_docs)
        ndcg = RetrievalMetrics.ndcg(results, relevant_docs)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "map": map_score,
            "ndcg": ndcg
        }

    def classify_intent(self, query: str) -> Tuple[str, float]:
        processed = self.process_query(query)
        return self.naive_bayes.predict(processed.stems)

    def detect_collocations(self, corpus: List[str], threshold: float = 1.0) -> List[Tuple[str, float]]:
        bigrams = []
        for text in corpus:
            tokens = self.stopwords.remove([t.lower() for t in re.findall(r'\b\w+\b', text)])
            bigrams.extend(NGramGenerator.generate(tokens, 2))

        self.pmi.build_from_ngrams(bigrams)

        collocations = []
        for bigram in set(bigrams):
            terms = bigram.split()
            if len(terms) == 2:
                pmi_score = self.pmi.calculate(terms[0], terms[1])
                if pmi_score >= threshold:
                    collocations.append((bigram, pmi_score))

        return sorted(collocations, key=lambda x: x[1], reverse=True)

    def cluster_queries(self, queries: List[str], k: int = 3) -> Dict[int, List[str]]:
        all_terms = set()
        processed_queries = [self.process_query(q) for q in queries]

        for pq in processed_queries:
            all_terms.update(pq.stems)

        vectors = []
        for pq in processed_queries:
            term_freqs = Counter(pq.stems)
            vector = {term: term_freqs.get(term, 0) for term in all_terms}
            vectors.append(vector)

        kmeans = KMeansClustering(k=k)
        kmeans.train(vectors)

        clusters = defaultdict(list)
        for query, vector in zip(queries, vectors):
            cluster_id = kmeans.predict(vector)
            clusters[cluster_id].append(query)

        return dict(clusters)
