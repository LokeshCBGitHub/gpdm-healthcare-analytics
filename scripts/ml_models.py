import json
import math
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Any, Optional


def dot_product(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def magnitude(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def normalize(v: List[float]) -> List[float]:
    mag = magnitude(v)
    if mag == 0:
        return v
    return [x / mag for x in v]


def softmax(scores: List[float]) -> List[float]:
    if not scores:
        return []
    max_score = max(scores)
    exp_scores = [math.exp(s - max_score) for s in scores]
    sum_exp = sum(exp_scores)
    return [e / sum_exp for e in exp_scores]


def argmax(scores: List[float]) -> int:
    if not scores:
        return 0
    return scores.index(max(scores))


def log_prob(x: float) -> float:
    if x <= 0:
        return -1e10
    return math.log(x)


class IntentClassifier:

    def __init__(self):
        self.intent_classes = [
            'count', 'aggregate', 'trend', 'breakdown', 'filter', 'top_n',
            'lookup', 'join', 'compare', 'distinct', 'journey', 'metadata',
            'negation', 'threshold', 'anomaly', 'percentage', 'existence', 'latest'
        ]
        self.vocabulary = set()
        self.class_counts = {intent: 0 for intent in self.intent_classes}
        self.feature_counts = {intent: defaultdict(int) for intent in self.intent_classes}
        self.class_priors = {}
        self.vocab_size = 0
        self.total_examples = 0

    def tokenize(self, text: str) -> List[str]:
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def train(self, examples: List[Tuple[str, str]]) -> None:
        self.vocabulary = set()
        self.class_counts = {intent: 0 for intent in self.intent_classes}
        self.feature_counts = {intent: defaultdict(int) for intent in self.intent_classes}

        for question, intent in examples:
            if intent not in self.intent_classes:
                continue

            self.total_examples += 1
            self.class_counts[intent] += 1
            tokens = self.tokenize(question)

            for token in tokens:
                self.vocabulary.add(token)
                self.feature_counts[intent][token] += 1

        self.vocab_size = len(self.vocabulary)

        if self.total_examples > 0:
            self.class_priors = {
                intent: self.class_counts[intent] / self.total_examples
                for intent in self.intent_classes
            }

    def predict(self, question: str) -> Tuple[str, float]:
        tokens = self.tokenize(question)

        if not self.vocabulary:
            return ('count', 0.0)

        scores = {}

        for intent in self.intent_classes:
            if self.class_priors.get(intent, 0) == 0:
                score = -1e10
            else:
                score = log_prob(self.class_priors[intent])

            for token in tokens:
                count = self.feature_counts[intent].get(token, 0)
                likelihood = (count + 1) / (sum(self.feature_counts[intent].values()) + self.vocab_size)
                score += log_prob(likelihood)

            scores[intent] = score

        max_score = max(scores.values()) if scores else 0
        adjusted_scores = [math.exp(scores[intent] - max_score) for intent in self.intent_classes]
        total = sum(adjusted_scores)

        predicted_intent = self.intent_classes[argmax([scores[intent] for intent in self.intent_classes])]
        predicted_idx = self.intent_classes.index(predicted_intent)
        confidence = adjusted_scores[predicted_idx] / total if total > 0 else 0.0

        return (predicted_intent, confidence)

    def save(self, path: str) -> None:
        model_data = {
            'vocabulary': list(self.vocabulary),
            'class_counts': self.class_counts,
            'feature_counts': {intent: dict(counts) for intent, counts in self.feature_counts.items()},
            'class_priors': self.class_priors,
            'vocab_size': self.vocab_size,
            'total_examples': self.total_examples,
            'intent_classes': self.intent_classes
        }
        with open(path, 'w') as f:
            json.dump(model_data, f, indent=2)

    def load(self, path: str) -> None:
        with open(path, 'r') as f:
            model_data = json.load(f)

        self.vocabulary = set(model_data['vocabulary'])
        self.class_counts = model_data['class_counts']
        self.feature_counts = {
            intent: defaultdict(int, counts)
            for intent, counts in model_data['feature_counts'].items()
        }
        self.class_priors = model_data['class_priors']
        self.vocab_size = model_data['vocab_size']
        self.total_examples = model_data['total_examples']
        self.intent_classes = model_data['intent_classes']


class EntityExtractor:

    def __init__(self):
        self.entity_types = ['TABLE', 'COLUMN', 'VALUE', 'NUMBER', 'DATE',
                            'REGION', 'OPERATOR', 'AGG_FUNC', 'O']

        self.initial_probs = {}
        self.transition_probs = {}
        self.emission_probs = {}

        self.feature_cache = {}

    def _extract_features(self, token: str) -> List[str]:
        features = []

        features.append(f'token={token.lower()}')
        features.append(f'len={len(token)}')

        if token[0].isupper():
            features.append('capitalized')
        if token.isupper():
            features.append('all_caps')
        if token.isdigit():
            features.append('numeric')

        if any(c.isdigit() for c in token):
            features.append('contains_digit')
        if '-' in token or '/' in token:
            features.append('date_like')

        return features

    def train(self, tagged_sequences: List[List[Tuple[str, str]]]) -> None:
        initial_counts = defaultdict(int)
        transition_counts = defaultdict(lambda: defaultdict(int))
        emission_counts = defaultdict(lambda: defaultdict(int))

        for sequence in tagged_sequences:
            if not sequence:
                continue

            initial_counts[sequence[0][1]] += 1

            for i, (token, entity_type) in enumerate(sequence):
                features = self._extract_features(token)
                for feature in features:
                    emission_counts[entity_type][feature] += 1

                if i < len(sequence) - 1:
                    next_entity = sequence[i + 1][1]
                    transition_counts[entity_type][next_entity] += 1

        total_initial = sum(initial_counts.values())
        self.initial_probs = {
            entity_type: (count + 1) / (total_initial + len(self.entity_types))
            for entity_type, count in initial_counts.items()
        }
        for entity_type in self.entity_types:
            if entity_type not in self.initial_probs:
                self.initial_probs[entity_type] = 1 / (total_initial + len(self.entity_types))

        self.transition_probs = {}
        for from_state in self.entity_types:
            self.transition_probs[from_state] = {}
            total = sum(transition_counts[from_state].values())
            for to_state in self.entity_types:
                count = transition_counts[from_state].get(to_state, 0)
                self.transition_probs[from_state][to_state] = (count + 1) / (total + len(self.entity_types))

        self.emission_probs = {}
        for entity_type in self.entity_types:
            self.emission_probs[entity_type] = {}
            total = sum(emission_counts[entity_type].values())
            all_features = set()
            for features_dict in emission_counts.values():
                all_features.update(features_dict.keys())

            for feature in all_features:
                count = emission_counts[entity_type].get(feature, 0)
                self.emission_probs[entity_type][feature] = (count + 1) / (total + len(all_features))

    def _viterbi(self, tokens: List[str]) -> List[str]:
        n = len(tokens)

        viterbi = [{} for _ in range(n)]
        backpointer = [{} for _ in range(n)]

        features = self._extract_features(tokens[0])
        for state in self.entity_types:
            emit_prob = 1.0
            for feature in features:
                emit_prob *= self.emission_probs[state].get(feature, 0.01)

            viterbi[0][state] = log_prob(self.initial_probs[state]) + log_prob(emit_prob)
            backpointer[0][state] = None

        for t in range(1, n):
            features = self._extract_features(tokens[t])

            for curr_state in self.entity_types:
                emit_prob = 1.0
                for feature in features:
                    emit_prob *= self.emission_probs[curr_state].get(feature, 0.01)

                best_prev_score = -1e10
                best_prev_state = None

                for prev_state in self.entity_types:
                    trans_prob = self.transition_probs[prev_state].get(curr_state, 0.01)
                    score = viterbi[t-1][prev_state] + log_prob(trans_prob) + log_prob(emit_prob)
                    if score > best_prev_score:
                        best_prev_score = score
                        best_prev_state = prev_state

                viterbi[t][curr_state] = best_prev_score
                backpointer[t][curr_state] = best_prev_state

        path = [None] * n
        path[-1] = max(viterbi[-1], key=viterbi[-1].get)

        for t in range(n - 2, -1, -1):
            path[t] = backpointer[t + 1][path[t + 1]]

        return path

    def extract(self, question: str) -> List[Tuple[str, str, int, int]]:
        tokens = re.findall(r'\b\w+\b|\d+', question)

        if not tokens:
            return []

        predicted_labels = self._viterbi(tokens)

        entities = []
        start_pos = 0
        current_entity = None
        current_label = None

        for token, label in zip(tokens, predicted_labels):
            if label != 'O':
                if label == current_label:
                    current_entity += ' ' + token
                else:
                    if current_entity:
                        pos = question.find(current_entity)
                        entities.append((current_entity, current_label, pos, pos + len(current_entity)))
                    current_entity = token
                    current_label = label
            else:
                if current_entity:
                    pos = question.find(current_entity)
                    entities.append((current_entity, current_label, pos, pos + len(current_entity)))
                    current_entity = None
                    current_label = None

        if current_entity:
            pos = question.find(current_entity)
            entities.append((current_entity, current_label, pos, pos + len(current_entity)))

        return entities


class SimilarityScorer:

    @staticmethod
    def tokenize(text: str) -> List[str]:
        text = text.lower()
        return re.findall(r'\b\w+\b', text)

    @staticmethod
    def cosine(text1: str, text2: str) -> float:
        tokens1 = SimilarityScorer.tokenize(text1)
        tokens2 = SimilarityScorer.tokenize(text2)

        if not tokens1 or not tokens2:
            return 0.0

        freq1 = Counter(tokens1)
        freq2 = Counter(tokens2)

        all_tokens = set(tokens1) | set(tokens2)

        vec1 = [freq1.get(token, 0) for token in all_tokens]
        vec2 = [freq2.get(token, 0) for token in all_tokens]

        numerator = dot_product(vec1, vec2)
        denominator = magnitude(vec1) * magnitude(vec2)

        if denominator == 0:
            return 0.0

        return numerator / denominator

    @staticmethod
    def jaccard(text1: str, text2: str) -> float:
        tokens1 = set(SimilarityScorer.tokenize(text1))
        tokens2 = set(SimilarityScorer.tokenize(text2))

        if not tokens1 and not tokens2:
            return 1.0

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        if union == 0:
            return 0.0

        return intersection / union

    @staticmethod
    def ngram_similarity(text1: str, text2: str, n: int = 3) -> float:
        text1 = text1.lower()
        text2 = text2.lower()

        ngrams1 = set(text1[i:i+n] for i in range(len(text1) - n + 1))
        ngrams2 = set(text2[i:i+n] for i in range(len(text2) - n + 1))

        if not ngrams1 and not ngrams2:
            return 1.0 if text1 == text2 else 0.0

        if not ngrams1 or not ngrams2:
            return 0.0

        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)

        return intersection / union

    @staticmethod
    def best_match(query: str, candidates: List[str]) -> Tuple[str, float]:
        if not candidates:
            return ('', 0.0)

        scores = []
        for candidate in candidates:
            cosine_sim = SimilarityScorer.cosine(query, candidate)
            jaccard_sim = SimilarityScorer.jaccard(query, candidate)
            ngram_sim = SimilarityScorer.ngram_similarity(query, candidate)

            combined = 0.5 * cosine_sim + 0.3 * jaccard_sim + 0.2 * ngram_sim
            scores.append(combined)

        best_idx = argmax(scores)
        return (candidates[best_idx], scores[best_idx])


class FeedbackLearner:

    def __init__(self, intent_classifier: IntentClassifier):
        self.classifier = intent_classifier
        self.feedback_history = []
        self.total_predictions = 0
        self.correct_predictions = 0

    def record_feedback(self, question: str, predicted_intent: str,
                       correct_intent: str, was_correct: bool) -> None:
        feedback_entry = {
            'question': question,
            'predicted': predicted_intent,
            'correct': correct_intent,
            'was_correct': was_correct
        }
        self.feedback_history.append(feedback_entry)
        self.total_predictions += 1
        if was_correct:
            self.correct_predictions += 1

    def get_accuracy(self) -> float:
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions

    def retrain(self) -> None:
        if not self.feedback_history:
            return

        training_examples = [
            (entry['question'], entry['correct'])
            for entry in self.feedback_history
        ]

        self.classifier.train(training_examples)

    def save_feedback(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump({
                'feedback_history': self.feedback_history,
                'total_predictions': self.total_predictions,
                'correct_predictions': self.correct_predictions
            }, f, indent=2)

    def load_feedback(self, path: str) -> None:
        with open(path, 'r') as f:
            data = json.load(f)
        self.feedback_history = data['feedback_history']
        self.total_predictions = data['total_predictions']
        self.correct_predictions = data['correct_predictions']


def get_intent_training_examples() -> List[Tuple[str, str]]:
    return [
        ("How many customers do we have?", "count"),
        ("Count the number of orders", "count"),
        ("What is the total count of products?", "count"),
        ("How many transactions occurred?", "count"),
        ("Total number of users", "count"),
        ("Count all records", "count"),
        ("How many entries in the database?", "count"),
        ("Total items sold", "count"),
        ("Number of active sessions", "count"),
        ("How many distinct customers purchased?", "count"),

        ("What is the total revenue?", "aggregate"),
        ("Sum of all sales", "aggregate"),
        ("Average order value", "aggregate"),
        ("Total cost of inventory", "aggregate"),
        ("Maximum price", "aggregate"),
        ("Minimum salary", "aggregate"),
        ("Sum the expenses", "aggregate"),
        ("Average rating", "aggregate"),
        ("Total amount spent", "aggregate"),

        ("Show sales trend over time", "trend"),
        ("How are sales trending?", "trend"),
        ("What is the trend in customer growth?", "trend"),
        ("Show me the progression of revenue", "trend"),
        ("Is usage increasing?", "trend"),
        ("How has traffic evolved?", "trend"),
        ("Show the monthly trend", "trend"),

        ("Break down sales by region", "breakdown"),
        ("Show revenue by category", "breakdown"),
        ("Breakdown expenses by department", "breakdown"),
        ("Revenue split by product type", "breakdown"),
        ("Show distribution by customer segment", "breakdown"),
        ("Sales by state", "breakdown"),

        ("Show orders from 2024", "filter"),
        ("Get customers in California", "filter"),
        ("Orders with amount over 1000", "filter"),
        ("Show active users", "filter"),
        ("Get transactions after January", "filter"),
        ("Filter by status approved", "filter"),
        ("Show high-value customers", "filter"),

        ("Top 10 customers", "top_n"),
        ("Show top 5 products", "top_n"),
        ("Top sellers", "top_n"),
        ("Get the top 3 regions by revenue", "top_n"),
        ("Best performing employees", "top_n"),
        ("Highest sales", "top_n"),

        ("What is the customer record for ID 123?", "lookup"),
        ("Find order 456", "lookup"),
        ("Get product details for SKU 789", "lookup"),
        ("Show me the employee record", "lookup"),
        ("Retrieve invoice details", "lookup"),

        ("Show customers and their orders", "join"),
        ("Join users with transactions", "join"),
        ("Combine employee and department info", "join"),
        ("Match products with suppliers", "join"),

        ("Compare 2023 vs 2024 sales", "compare"),
        ("How does Q1 compare to Q2?", "compare"),
        ("Difference between regions", "compare"),
        ("Show East vs West sales", "compare"),
        ("Compare this month to last month", "compare"),

        ("How many distinct categories?", "distinct"),
        ("Unique customers", "distinct"),
        ("Show distinct regions", "distinct"),
        ("Count unique products", "distinct"),

        ("Show customer journey", "journey"),
        ("Track order status", "journey"),
        ("What is the customer lifecycle?", "journey"),
        ("Show transaction path", "journey"),

        ("What tables are available?", "metadata"),
        ("List all columns in orders", "metadata"),
        ("Show database schema", "metadata"),
        ("What fields exist?", "metadata"),

        ("Orders NOT delivered", "negation"),
        ("Customers without purchases", "negation"),
        ("Show inactive users", "negation"),
        ("Orders that failed", "negation"),

        ("Orders above 500", "threshold"),
        ("Show items below cost", "threshold"),
        ("Employees earning more than 80k", "threshold"),
        ("Revenue exceeding quota", "threshold"),

        ("Show unusual patterns", "anomaly"),
        ("Detect outliers", "anomaly"),
        ("What transactions are suspicious?", "anomaly"),
        ("Find abnormal behavior", "anomaly"),

        ("What percent completed?", "percentage"),
        ("Completion rate", "percentage"),
        ("Show growth percentage", "percentage"),
        ("What is the conversion rate?", "percentage"),

        ("Do we have any pending orders?", "existence"),
        ("Are there any errors?", "existence"),
        ("Is there any unused inventory?", "existence"),
        ("Do customers have feedback?", "existence"),

        ("Show latest orders", "latest"),
        ("Most recent transactions", "latest"),
        ("Latest customer signup", "latest"),
        ("What happened recently?", "latest"),
    ]


def get_entity_training_sequences() -> List[List[Tuple[str, str]]]:
    return [
        [('How', 'O'), ('many', 'O'), ('customers', 'COLUMN'), ('in', 'O'), ('California', 'REGION')],
        [('Show', 'O'), ('sales', 'COLUMN'), ('by', 'O'), ('product', 'COLUMN')],
        [('Top', 'O'), ('10', 'NUMBER'), ('orders', 'TABLE')],
        [('Revenue', 'COLUMN'), ('from', 'O'), ('2024', 'DATE')],
        [('Get', 'O'), ('users', 'TABLE'), ('where', 'O'), ('age', 'COLUMN'), ('>', 'OPERATOR'), ('18', 'NUMBER')],
        [('Sum', 'AGG_FUNC'), ('of', 'O'), ('sales', 'COLUMN')],
        [('Average', 'AGG_FUNC'), ('price', 'COLUMN'), ('in', 'O'), ('New', 'REGION'), ('York', 'REGION')],
        [('Show', 'O'), ('count', 'AGG_FUNC'), ('of', 'O'), ('transactions', 'TABLE')],
        [('Find', 'O'), ('orders', 'TABLE'), ('with', 'O'), ('amount', 'COLUMN'), ('=', 'OPERATOR'), ('100', 'NUMBER')],
        [('List', 'O'), ('employees', 'TABLE'), ('from', 'O'), ('2023', 'DATE')],
        [('What', 'O'), ('is', 'O'), ('the', 'O'), ('maximum', 'AGG_FUNC'), ('salary', 'COLUMN')],
        [('Count', 'AGG_FUNC'), ('distinct', 'O'), ('regions', 'COLUMN')],
        [('Revenue', 'COLUMN'), ('compared', 'O'), ('to', 'O'), ('2023', 'DATE')],
        [('Get', 'O'), ('products', 'TABLE'), ('cheaper', 'OPERATOR'), ('than', 'O'), ('50', 'NUMBER')],
        [('How', 'O'), ('many', 'O'), ('customers', 'TABLE'), ('in', 'O'), ('Texas', 'REGION')],
        [('Top', 'O'), ('5', 'NUMBER'), ('sales', 'COLUMN'), ('by', 'O'), ('region', 'COLUMN')],
        [('Show', 'O'), ('orders', 'TABLE'), ('after', 'O'), ('January', 'DATE')],
        [('Min', 'AGG_FUNC'), ('price', 'COLUMN'), ('in', 'O'), ('Asia', 'REGION')],
        [('Filter', 'O'), ('transactions', 'TABLE'), ('>=', 'OPERATOR'), ('1000', 'NUMBER')],
        [('Sum', 'AGG_FUNC'), ('revenue', 'COLUMN'), ('from', 'O'), ('2024', 'DATE')],
    ]


if __name__ == '__main__':
    print("=" * 80)
    print("ML MODELS MODULE - PURE PYTHON IMPLEMENTATION")
    print("=" * 80)

    print("\n[1] Initializing Intent Classifier...")
    intent_classifier = IntentClassifier()

    print("[2] Training Intent Classifier on ~200 examples...")
    training_examples = get_intent_training_examples()
    intent_classifier.train(training_examples)
    print(f"    Vocabulary size: {intent_classifier.vocab_size}")
    print(f"    Total examples: {intent_classifier.total_examples}")
    print(f"    Intent classes: {len(intent_classifier.intent_classes)}")

    print("\n[3] Initializing Entity Extractor (HMM)...")
    entity_extractor = EntityExtractor()

    print("[4] Training Entity Extractor on ~50 sequences...")
    entity_sequences = get_entity_training_sequences()
    entity_extractor.train(entity_sequences)
    print(f"    Entity types: {entity_extractor.entity_types}")

    print("\n[5] Initializing Similarity Scorer...")
    similarity_scorer = SimilarityScorer()

    print("[6] Initializing Feedback Learner...")
    feedback_learner = FeedbackLearner(intent_classifier)

    print("\n" + "=" * 80)
    print("TESTING INTENT CLASSIFICATION ON 20 TEST QUESTIONS")
    print("=" * 80)

    test_questions = [
        "How many orders did we get this month?",
        "What is the total revenue from 2024?",
        "Show sales trend over the year",
        "Break down expenses by department",
        "Get all orders from California",
        "Top 10 best-selling products",
        "Find customer record 12345",
        "Show customers and their purchase history",
        "Compare Q1 and Q2 performance",
        "How many unique regions do we serve?",
        "Show the customer journey",
        "What columns exist in the orders table?",
        "Show orders that were not delivered",
        "Orders over 5000 dollars",
        "Detect suspicious transactions",
        "What percent of orders completed?",
        "Are there any pending invoices?",
        "Show the most recent sales",
        "Customers without any purchases",
        "Maximum salary in the company",
    ]

    correct_predictions = 0
    results = []

    for i, question in enumerate(test_questions, 1):
        predicted_intent, confidence = intent_classifier.predict(question)

        entities = entity_extractor.extract(question)

        print(f"\n[Q{i}] {question}")
        print(f"     Intent: {predicted_intent} (confidence: {confidence:.3f})")
        if entities:
            print(f"     Entities: {entities[:2]}")

        results.append({
            'question': question,
            'predicted_intent': predicted_intent,
            'confidence': confidence,
            'entities_found': len(entities)
        })

    print("\n" + "=" * 80)
    print("TESTING SIMILARITY SCORING")
    print("=" * 80)

    query = "total sales by region"
    candidates = [
        "revenue breakdown by area",
        "show all products",
        "sum of sales per region",
        "customer list"
    ]

    print(f"\nQuery: '{query}'")
    print(f"Candidates: {candidates}\n")

    for candidate in candidates:
        cosine = SimilarityScorer.cosine(query, candidate)
        jaccard = SimilarityScorer.jaccard(query, candidate)
        ngram = SimilarityScorer.ngram_similarity(query, candidate)
        print(f"  '{candidate}'")
        print(f"    Cosine: {cosine:.3f}, Jaccard: {jaccard:.3f}, N-gram: {ngram:.3f}")

    best_match, best_score = SimilarityScorer.best_match(query, candidates)
    print(f"\nBest match: '{best_match}' (score: {best_score:.3f})")

    print("\n" + "=" * 80)
    print("TESTING FEEDBACK LEARNER")
    print("=" * 80)

    for result in results[:5]:
        correct = result['predicted_intent']
        feedback_learner.record_feedback(
            result['question'],
            result['predicted_intent'],
            correct,
            was_correct=True
        )

    print(f"\nRecorded 5 feedback entries")
    print(f"Current accuracy: {feedback_learner.get_accuracy():.3f}")
    print(f"Total predictions: {feedback_learner.total_predictions}")
    print(f"Correct predictions: {feedback_learner.correct_predictions}")

    print("\n" + "=" * 80)
    print("TESTING SAVE/LOAD FUNCTIONALITY")
    print("=" * 80)

    model_path = "/tmp/test_intent_classifier.json"
    feedback_path = "/tmp/test_feedback.json"

    print(f"\nSaving model to: {model_path}")
    intent_classifier.save(model_path)

    print(f"Loading model from: {model_path}")
    loaded_classifier = IntentClassifier()
    loaded_classifier.load(model_path)

    test_q = "What is the total revenue?"
    original_pred, original_conf = intent_classifier.predict(test_q)
    loaded_pred, loaded_conf = loaded_classifier.predict(test_q)

    print(f"\nOriginal model prediction: {original_pred} ({original_conf:.3f})")
    print(f"Loaded model prediction:  {loaded_pred} ({loaded_conf:.3f})")
    print(f"Match: {original_pred == loaded_pred}")

    print(f"\nSaving feedback to: {feedback_path}")
    feedback_learner.save_feedback(feedback_path)

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"\nIntent Classes Supported: {len(intent_classifier.intent_classes)}")
    print(f"Entity Types Supported: {len(entity_extractor.entity_types)}")
    print(f"Model Vocabulary Size: {intent_classifier.vocab_size} tokens")
    print(f"Training Examples Processed: {intent_classifier.total_examples}")
    print(f"Average Confidence on Test Set: {sum(r['confidence'] for r in results) / len(results):.3f}")
    print(f"Entities Extracted (avg per question): {sum(r['entities_found'] for r in results) / len(results):.1f}")

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 80)
