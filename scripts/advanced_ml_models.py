import math
import random
import json
import re
import time
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional, Any


def dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))

def vec_add(a: List[float], b: List[float]) -> List[float]:
    return [x + y for x, y in zip(a, b)]

def vec_sub(a: List[float], b: List[float]) -> List[float]:
    return [x - y for x, y in zip(a, b)]

def vec_scale(a: List[float], s: float) -> List[float]:
    return [x * s for x in a]

def vec_norm(a: List[float]) -> float:
    return math.sqrt(sum(x * x for x in a))

def vec_normalize(a: List[float]) -> List[float]:
    n = vec_norm(a)
    return [x / n for x in a] if n > 0 else a

def cosine_sim(a: List[float], b: List[float]) -> float:
    d = dot(a, b)
    na, nb = vec_norm(a), vec_norm(b)
    return d / (na * nb) if na > 0 and nb > 0 else 0.0

def euclidean_dist(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ez = math.exp(x)
        return ez / (1.0 + ez)

def relu(x: float) -> float:
    return max(0.0, x)

def relu_deriv(x: float) -> float:
    return 1.0 if x > 0 else 0.0

def tanh(x: float) -> float:
    return math.tanh(x)

def tanh_deriv(x: float) -> float:
    t = math.tanh(x)
    return 1.0 - t * t

def log_safe(x: float) -> float:
    return math.log(max(x, 1e-15))

def softmax(scores: List[float]) -> List[float]:
    mx = max(scores)
    exps = [math.exp(s - mx) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


class TFIDFVectorizer:

    def __init__(self, max_features: int = 500):
        self.max_features = max_features
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_count = 0

    def tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b[a-z]+\b', text.lower())

    def fit(self, documents: List[str]) -> 'TFIDFVectorizer':
        self.doc_count = len(documents)
        df = Counter()

        all_tf = []
        for doc in documents:
            tokens = self.tokenize(doc)
            tf = Counter(tokens)
            all_tf.append(tf)
            for term in set(tokens):
                df[term] += 1

        top_terms = [t for t, _ in df.most_common(self.max_features)]
        self.vocabulary = {term: idx for idx, term in enumerate(top_terms)}

        self.idf = {}
        for term in self.vocabulary:
            self.idf[term] = math.log(self.doc_count / (1 + df[term]))

        return self

    def transform(self, text: str) -> List[float]:
        tokens = self.tokenize(text)
        tf = Counter(tokens)
        total = len(tokens) if tokens else 1
        vec = [0.0] * len(self.vocabulary)

        for term, idx in self.vocabulary.items():
            if term in tf:
                vec[idx] = (tf[term] / total) * self.idf.get(term, 0.0)

        return vec

    def fit_transform(self, documents: List[str]) -> List[List[float]]:
        self.fit(documents)
        return [self.transform(doc) for doc in documents]

    def get_top_terms(self, text: str, n: int = 5) -> List[Tuple[str, float]]:
        vec = self.transform(text)
        idx_to_term = {v: k for k, v in self.vocabulary.items()}
        scored = [(idx_to_term[i], vec[i]) for i in range(len(vec)) if vec[i] > 0]
        scored.sort(key=lambda x: -x[1])
        return scored[:n]


class KNNClassifier:

    def __init__(self, k: int = 5):
        self.k = k
        self.X_train: List[List[float]] = []
        self.y_train: List[str] = []

    def fit(self, X: List[List[float]], y: List[str]) -> 'KNNClassifier':
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, x: List[float]) -> Tuple[str, float]:
        distances = []
        for i, x_i in enumerate(self.X_train):
            d = euclidean_dist(x, x_i)
            distances.append((d, self.y_train[i]))

        distances.sort(key=lambda t: t[0])
        neighbors = distances[:self.k]

        votes = Counter(label for _, label in neighbors)
        winner, count = votes.most_common(1)[0]
        confidence = count / self.k

        return winner, confidence

    def predict_batch(self, X: List[List[float]]) -> List[Tuple[str, float]]:
        return [self.predict(x) for x in X]


class DecisionTree:

    def __init__(self, max_depth: int = 10, min_samples: int = 2):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.tree = None

    @staticmethod
    def entropy(labels: List[str]) -> float:
        if not labels:
            return 0.0
        counts = Counter(labels)
        total = len(labels)
        ent = 0.0
        for c in counts.values():
            p = c / total
            if p > 0:
                ent -= p * math.log2(p)
        return ent

    def _best_split(self, X: List[List[float]], y: List[str]) -> Tuple[int, float, float]:
        best_gain = -1
        best_feat = 0
        best_thresh = 0.0
        parent_entropy = self.entropy(y)
        n = len(y)

        for feat_idx in range(len(X[0])):
            values = sorted(set(row[feat_idx] for row in X))
            for i in range(len(values) - 1):
                thresh = (values[i] + values[i + 1]) / 2.0

                left_y = [y[j] for j in range(n) if X[j][feat_idx] <= thresh]
                right_y = [y[j] for j in range(n) if X[j][feat_idx] > thresh]

                if not left_y or not right_y:
                    continue

                w_left = len(left_y) / n
                w_right = len(right_y) / n
                child_entropy = w_left * self.entropy(left_y) + w_right * self.entropy(right_y)
                gain = parent_entropy - child_entropy

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat_idx
                    best_thresh = thresh

        return best_feat, best_thresh, best_gain

    def _build_tree(self, X: List[List[float]], y: List[str], depth: int) -> Dict:
        if depth >= self.max_depth or len(y) < self.min_samples or len(set(y)) == 1:
            return {'leaf': True, 'class': Counter(y).most_common(1)[0][0],
                    'samples': len(y), 'distribution': dict(Counter(y))}

        feat_idx, thresh, gain = self._best_split(X, y)

        if gain <= 0:
            return {'leaf': True, 'class': Counter(y).most_common(1)[0][0],
                    'samples': len(y), 'distribution': dict(Counter(y))}

        left_idx = [i for i in range(len(y)) if X[i][feat_idx] <= thresh]
        right_idx = [i for i in range(len(y)) if X[i][feat_idx] > thresh]

        left_X = [X[i] for i in left_idx]
        left_y = [y[i] for i in left_idx]
        right_X = [X[i] for i in right_idx]
        right_y = [y[i] for i in right_idx]

        return {
            'leaf': False,
            'feature': feat_idx,
            'threshold': thresh,
            'info_gain': gain,
            'samples': len(y),
            'left': self._build_tree(left_X, left_y, depth + 1),
            'right': self._build_tree(right_X, right_y, depth + 1),
        }

    def fit(self, X: List[List[float]], y: List[str]) -> 'DecisionTree':
        self.tree = self._build_tree(X, y, 0)
        return self

    def _predict_one(self, x: List[float], node: Dict) -> str:
        if node['leaf']:
            return node['class']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        else:
            return self._predict_one(x, node['right'])

    def predict(self, x: List[float]) -> str:
        return self._predict_one(x, self.tree)

    def predict_batch(self, X: List[List[float]]) -> List[str]:
        return [self.predict(x) for x in X]

    def print_tree(self, node=None, indent="", feature_names=None):
        if node is None:
            node = self.tree
        if node['leaf']:
            print(f"{indent}→ {node['class']} (n={node['samples']})")
        else:
            fname = feature_names[node['feature']] if feature_names else f"f[{node['feature']}]"
            print(f"{indent}{fname} ≤ {node['threshold']:.3f}? (gain={node['info_gain']:.4f}, n={node['samples']})")
            print(f"{indent}  Yes:")
            self.print_tree(node['left'], indent + "    ", feature_names)
            print(f"{indent}  No:")
            self.print_tree(node['right'], indent + "    ", feature_names)


class RandomForest:

    def __init__(self, n_trees: int = 10, max_depth: int = 8,
                 max_features: Optional[int] = None, min_samples: int = 2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples = min_samples
        self.trees: List[Tuple[DecisionTree, List[int]]] = []

    def fit(self, X: List[List[float]], y: List[str]) -> 'RandomForest':
        n_samples = len(X)
        n_features = len(X[0])
        max_feat = self.max_features or max(1, int(math.sqrt(n_features)))

        self.trees = []
        for _ in range(self.n_trees):
            indices = [random.randint(0, n_samples - 1) for _ in range(n_samples)]
            X_boot = [X[i] for i in indices]
            y_boot = [y[i] for i in indices]

            feat_indices = sorted(random.sample(range(n_features), min(max_feat, n_features)))

            X_proj = [[row[f] for f in feat_indices] for row in X_boot]

            tree = DecisionTree(max_depth=self.max_depth, min_samples=self.min_samples)
            tree.fit(X_proj, y_boot)
            self.trees.append((tree, feat_indices))

        return self

    def predict(self, x: List[float]) -> Tuple[str, float]:
        votes = []
        for tree, feat_indices in self.trees:
            x_proj = [x[f] for f in feat_indices]
            votes.append(tree.predict(x_proj))

        counts = Counter(votes)
        winner, count = counts.most_common(1)[0]
        confidence = count / len(self.trees)
        return winner, confidence

    def predict_batch(self, X: List[List[float]]) -> List[Tuple[str, float]]:
        return [self.predict(x) for x in X]


class KMeans:

    def __init__(self, k: int = 3, max_iter: int = 100, seed: int = 42):
        self.k = k
        self.max_iter = max_iter
        self.seed = seed
        self.centroids: List[List[float]] = []
        self.labels: List[int] = []
        self.inertia: float = 0.0

    def fit(self, X: List[List[float]]) -> 'KMeans':
        random.seed(self.seed)
        n = len(X)
        dim = len(X[0])

        self.centroids = [X[random.randint(0, n - 1)][:]]
        for _ in range(1, self.k):
            dists = []
            for x in X:
                min_d = min(euclidean_dist(x, c) ** 2 for c in self.centroids)
                dists.append(min_d)
            total_d = sum(dists)
            r = random.uniform(0, total_d)
            cumulative = 0
            for i, d in enumerate(dists):
                cumulative += d
                if cumulative >= r:
                    self.centroids.append(X[i][:])
                    break

        for iteration in range(self.max_iter):
            new_labels = []
            for x in X:
                dists = [euclidean_dist(x, c) for c in self.centroids]
                new_labels.append(dists.index(min(dists)))

            if new_labels == self.labels:
                break
            self.labels = new_labels

            for c_idx in range(self.k):
                members = [X[i] for i in range(n) if self.labels[i] == c_idx]
                if members:
                    self.centroids[c_idx] = [
                        sum(m[d] for m in members) / len(members)
                        for d in range(dim)
                    ]

        self.inertia = sum(
            euclidean_dist(X[i], self.centroids[self.labels[i]]) ** 2
            for i in range(n)
        )
        return self

    def predict(self, x: List[float]) -> int:
        dists = [euclidean_dist(x, c) for c in self.centroids]
        return dists.index(min(dists))

    def predict_batch(self, X: List[List[float]]) -> List[int]:
        return [self.predict(x) for x in X]


class Word2Vec:

    def __init__(self, embed_dim: int = 32, window: int = 3,
                 n_negative: int = 5, lr: float = 0.025, epochs: int = 5):
        self.embed_dim = embed_dim
        self.window = window
        self.n_negative = n_negative
        self.lr = lr
        self.epochs = epochs
        self.vocab: Dict[str, int] = {}
        self.reverse_vocab: Dict[int, str] = {}
        self.W_in: List[List[float]] = []
        self.W_out: List[List[float]] = []
        self.word_freq: List[float] = []

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b[a-z]+\b', text.lower())

    def _init_embeddings(self, vocab_size: int):
        limit = math.sqrt(6.0 / (vocab_size + self.embed_dim))
        self.W_in = [[random.uniform(-limit, limit) for _ in range(self.embed_dim)]
                     for _ in range(vocab_size)]
        self.W_out = [[random.uniform(-limit, limit) for _ in range(self.embed_dim)]
                      for _ in range(vocab_size)]

    def _sample_negative(self) -> int:
        r = random.random() * self.total_freq_pow
        cumulative = 0.0
        for i, f in enumerate(self.word_freq_pow):
            cumulative += f
            if cumulative >= r:
                return i
        return len(self.word_freq_pow) - 1

    def fit(self, corpus: List[str]) -> 'Word2Vec':
        word_counts = Counter()
        for sentence in corpus:
            word_counts.update(self._tokenize(sentence))

        self.vocab = {w: i for i, (w, c) in enumerate(word_counts.most_common()) if c >= 1}
        self.reverse_vocab = {i: w for w, i in self.vocab.items()}
        vocab_size = len(self.vocab)

        if vocab_size < 2:
            return self

        self.word_freq = [0.0] * vocab_size
        for w, i in self.vocab.items():
            self.word_freq[i] = word_counts[w]
        self.word_freq_pow = [f ** 0.75 for f in self.word_freq]
        self.total_freq_pow = sum(self.word_freq_pow)

        self._init_embeddings(vocab_size)

        tokenized = []
        for sentence in corpus:
            tokens = self._tokenize(sentence)
            ids = [self.vocab[t] for t in tokens if t in self.vocab]
            if len(ids) >= 2:
                tokenized.append(ids)

        for epoch in range(self.epochs):
            total_loss = 0.0
            n_pairs = 0
            random.shuffle(tokenized)

            for ids in tokenized:
                for i, target_id in enumerate(ids):
                    start = max(0, i - self.window)
                    end = min(len(ids), i + self.window + 1)

                    for j in range(start, end):
                        if j == i:
                            continue
                        context_id = ids[j]

                        d = dot(self.W_in[target_id], self.W_out[context_id])
                        sig = sigmoid(d)
                        grad = (sig - 1.0) * self.lr
                        total_loss -= log_safe(sig)

                        for k in range(self.embed_dim):
                            g_in = grad * self.W_out[context_id][k]
                            g_out = grad * self.W_in[target_id][k]
                            self.W_in[target_id][k] -= g_in
                            self.W_out[context_id][k] -= g_out

                        for _ in range(self.n_negative):
                            neg_id = self._sample_negative()
                            if neg_id == context_id:
                                continue

                            d = dot(self.W_in[target_id], self.W_out[neg_id])
                            sig = sigmoid(d)
                            grad = sig * self.lr
                            total_loss -= log_safe(1.0 - sig)

                            for k in range(self.embed_dim):
                                g_in = grad * self.W_out[neg_id][k]
                                g_out = grad * self.W_in[target_id][k]
                                self.W_in[target_id][k] -= g_in
                                self.W_out[neg_id][k] -= g_out

                        n_pairs += 1

        return self

    def get_vector(self, word: str) -> Optional[List[float]]:
        word = word.lower()
        if word in self.vocab:
            return self.W_in[self.vocab[word]][:]
        return None

    def most_similar(self, word: str, top_n: int = 5) -> List[Tuple[str, float]]:
        vec = self.get_vector(word)
        if vec is None:
            return []

        similarities = []
        for w, idx in self.vocab.items():
            if w == word.lower():
                continue
            sim = cosine_sim(vec, self.W_in[idx])
            similarities.append((w, sim))

        similarities.sort(key=lambda x: -x[1])
        return similarities[:top_n]


class NeuralNetwork:

    def __init__(self, layer_sizes: List[int], activation: str = 'relu',
                 lr: float = 0.01, momentum: float = 0.9):
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.lr = lr
        self.momentum = momentum
        self.n_layers = len(layer_sizes) - 1

        self.weights: List[List[List[float]]] = []
        self.biases: List[List[float]] = []
        self.v_weights: List[List[List[float]]] = []
        self.v_biases: List[List[float]] = []

        random.seed(42)
        for i in range(self.n_layers):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            limit = math.sqrt(6.0 / (fan_in + fan_out))

            W = [[random.uniform(-limit, limit) for _ in range(fan_out)]
                 for _ in range(fan_in)]
            b = [0.0] * fan_out

            self.weights.append(W)
            self.biases.append(b)
            self.v_weights.append([[0.0] * fan_out for _ in range(fan_in)])
            self.v_biases.append([0.0] * fan_out)

    def _activate(self, x: float) -> float:
        if self.activation == 'relu':
            return relu(x)
        return tanh(x)

    def _activate_deriv(self, x: float) -> float:
        if self.activation == 'relu':
            return relu_deriv(x)
        return tanh_deriv(x)

    def forward(self, x: List[float]) -> Tuple[List[float], List[Any]]:
        cache = {'inputs': [], 'pre_activations': [], 'activations': []}
        a = x[:]

        for layer_idx in range(self.n_layers):
            cache['inputs'].append(a[:])
            W = self.weights[layer_idx]
            b = self.biases[layer_idx]

            z = []
            for j in range(len(b)):
                val = b[j]
                for i in range(len(a)):
                    val += a[i] * W[i][j]
                z.append(val)
            cache['pre_activations'].append(z[:])

            if layer_idx == self.n_layers - 1:
                a = softmax(z)
            else:
                a = [self._activate(v) for v in z]
            cache['activations'].append(a[:])

        return a, cache

    def backward(self, y_true_idx: int, cache: dict) -> None:
        output = cache['activations'][-1]
        delta = output[:]
        delta[y_true_idx] -= 1.0

        for layer_idx in range(self.n_layers - 1, -1, -1):
            a_in = cache['inputs'][layer_idx]
            W = self.weights[layer_idx]

            for i in range(len(a_in)):
                for j in range(len(delta)):
                    grad = a_in[i] * delta[j]
                    self.v_weights[layer_idx][i][j] = (
                        self.momentum * self.v_weights[layer_idx][i][j] - self.lr * grad
                    )
                    self.weights[layer_idx][i][j] += self.v_weights[layer_idx][i][j]

            for j in range(len(delta)):
                self.v_biases[layer_idx][j] = (
                    self.momentum * self.v_biases[layer_idx][j] - self.lr * delta[j]
                )
                self.biases[layer_idx][j] += self.v_biases[layer_idx][j]

            if layer_idx > 0:
                new_delta = [0.0] * len(a_in)
                for i in range(len(a_in)):
                    for j in range(len(delta)):
                        new_delta[i] += W[i][j] * delta[j]

                z = cache['pre_activations'][layer_idx - 1]
                delta = [new_delta[i] * self._activate_deriv(z[i]) for i in range(len(new_delta))]

    def fit(self, X: List[List[float]], y: List[int], epochs: int = 50,
            batch_size: int = 16, verbose: bool = False) -> List[float]:
        loss_history = []

        for epoch in range(epochs):
            indices = list(range(len(X)))
            random.shuffle(indices)

            epoch_loss = 0.0
            correct = 0

            for idx in indices:
                output, cache = self.forward(X[idx])
                epoch_loss -= log_safe(output[y[idx]])

                if output.index(max(output)) == y[idx]:
                    correct += 1

                self.backward(y[idx], cache)

            avg_loss = epoch_loss / len(X)
            acc = correct / len(X)
            loss_history.append(avg_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f}, Acc: {acc:.3f}")

        return loss_history

    def predict(self, x: List[float]) -> Tuple[int, float]:
        output, _ = self.forward(x)
        pred_idx = output.index(max(output))
        return pred_idx, output[pred_idx]


class LogisticRegression:

    def __init__(self, n_features: int, n_classes: int, lr: float = 0.01):
        self.n_features = n_features
        self.n_classes = n_classes
        self.lr = lr

        limit = math.sqrt(6.0 / (n_features + n_classes))
        random.seed(42)
        self.W = [[random.uniform(-limit, limit) for _ in range(n_classes)]
                  for _ in range(n_features)]
        self.b = [0.0] * n_classes

    def predict_proba(self, x: List[float]) -> List[float]:
        z = self.b[:]
        for j in range(self.n_classes):
            for i in range(self.n_features):
                z[j] += x[i] * self.W[i][j]
        return softmax(z)

    def predict(self, x: List[float]) -> Tuple[int, float]:
        probs = self.predict_proba(x)
        idx = probs.index(max(probs))
        return idx, probs[idx]

    def fit(self, X: List[List[float]], y: List[int], epochs: int = 100,
            verbose: bool = False) -> List[float]:
        loss_history = []
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            indices = list(range(len(X)))
            random.shuffle(indices)

            for idx in indices:
                probs = self.predict_proba(X[idx])
                total_loss -= log_safe(probs[y[idx]])
                if probs.index(max(probs)) == y[idx]:
                    correct += 1

                grad = probs[:]
                grad[y[idx]] -= 1.0

                for j in range(self.n_classes):
                    for i in range(self.n_features):
                        self.W[i][j] -= self.lr * grad[j] * X[idx][i]
                    self.b[j] -= self.lr * grad[j]

            avg_loss = total_loss / len(X)
            loss_history.append(avg_loss)

            if verbose and (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f}, Acc: {correct/len(X):.3f}")

        return loss_history


class PCA:

    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.components: List[List[float]] = []
        self.mean: List[float] = []
        self.explained_variance: List[float] = []

    def fit(self, X: List[List[float]]) -> 'PCA':
        n = len(X)
        dim = len(X[0])

        self.mean = [sum(X[i][d] for i in range(n)) / n for d in range(dim)]

        X_centered = [[X[i][d] - self.mean[d] for d in range(dim)] for i in range(n)]

        cov = [[0.0] * dim for _ in range(dim)]
        for i in range(dim):
            for j in range(i, dim):
                val = sum(X_centered[k][i] * X_centered[k][j] for k in range(n)) / (n - 1)
                cov[i][j] = val
                cov[j][i] = val

        self.components = []
        self.explained_variance = []

        for comp in range(self.n_components):
            random.seed(42 + comp)
            v = [random.gauss(0, 1) for _ in range(dim)]
            v = vec_normalize(v)

            for _ in range(200):
                v_new = [sum(cov[i][j] * v[j] for j in range(dim)) for i in range(dim)]
                eigenvalue = vec_norm(v_new)
                if eigenvalue > 0:
                    v_new = vec_normalize(v_new)
                v = v_new

            self.components.append(v)
            self.explained_variance.append(eigenvalue)

            for i in range(dim):
                for j in range(dim):
                    cov[i][j] -= eigenvalue * v[i] * v[j]

        return self

    def transform(self, X: List[List[float]]) -> List[List[float]]:
        result = []
        for x in X:
            centered = [x[d] - self.mean[d] for d in range(len(x))]
            projected = [dot(centered, comp) for comp in self.components]
            result.append(projected)
        return result

    def fit_transform(self, X: List[List[float]]) -> List[List[float]]:
        self.fit(X)
        return self.transform(X)


def get_healthcare_training_data():

    intent_data = [
        ("how many claims were denied", "count",
         {"has_count_word": 1, "has_agg_word": 0, "has_group_word": 0, "has_filter_word": 1, "has_top_word": 0, "has_trend_word": 0, "word_count": 5, "has_number": 0}),
        ("total billed amount", "aggregate",
         {"has_count_word": 0, "has_agg_word": 1, "has_group_word": 0, "has_filter_word": 0, "has_top_word": 0, "has_trend_word": 0, "word_count": 3, "has_number": 0}),
        ("show claims by region", "breakdown",
         {"has_count_word": 0, "has_agg_word": 0, "has_group_word": 1, "has_filter_word": 0, "has_top_word": 0, "has_trend_word": 0, "word_count": 4, "has_number": 0}),
        ("top 10 providers", "top_n",
         {"has_count_word": 0, "has_agg_word": 0, "has_group_word": 0, "has_filter_word": 0, "has_top_word": 1, "has_trend_word": 0, "word_count": 3, "has_number": 1}),
        ("claim trend over time", "trend",
         {"has_count_word": 0, "has_agg_word": 0, "has_group_word": 0, "has_filter_word": 0, "has_top_word": 0, "has_trend_word": 1, "word_count": 4, "has_number": 0}),
        ("denied claims in california", "filter",
         {"has_count_word": 0, "has_agg_word": 0, "has_group_word": 0, "has_filter_word": 1, "has_top_word": 0, "has_trend_word": 0, "word_count": 4, "has_number": 0}),
        ("average cost per encounter", "aggregate",
         {"has_count_word": 0, "has_agg_word": 1, "has_group_word": 1, "has_filter_word": 0, "has_top_word": 0, "has_trend_word": 0, "word_count": 4, "has_number": 0}),
        ("number of members by plan type", "breakdown",
         {"has_count_word": 1, "has_agg_word": 0, "has_group_word": 1, "has_filter_word": 0, "has_top_word": 0, "has_trend_word": 0, "word_count": 6, "has_number": 0}),
        ("top 5 diagnosis codes", "top_n",
         {"has_count_word": 0, "has_agg_word": 0, "has_group_word": 0, "has_filter_word": 0, "has_top_word": 1, "has_trend_word": 0, "word_count": 4, "has_number": 1}),
        ("how many encounters last month", "count",
         {"has_count_word": 1, "has_agg_word": 0, "has_group_word": 0, "has_filter_word": 1, "has_top_word": 0, "has_trend_word": 0, "word_count": 5, "has_number": 0}),
        ("total paid amount by provider", "breakdown",
         {"has_count_word": 0, "has_agg_word": 1, "has_group_word": 1, "has_filter_word": 0, "has_top_word": 0, "has_trend_word": 0, "word_count": 5, "has_number": 0}),
        ("prescription fill rate trend", "trend",
         {"has_count_word": 0, "has_agg_word": 0, "has_group_word": 0, "has_filter_word": 0, "has_top_word": 0, "has_trend_word": 1, "word_count": 4, "has_number": 0}),
        ("show referral status distribution", "breakdown",
         {"has_count_word": 0, "has_agg_word": 0, "has_group_word": 0, "has_filter_word": 0, "has_top_word": 0, "has_trend_word": 0, "word_count": 4, "has_number": 0}),
        ("claims with amount over 5000", "filter",
         {"has_count_word": 0, "has_agg_word": 0, "has_group_word": 0, "has_filter_word": 1, "has_top_word": 0, "has_trend_word": 0, "word_count": 5, "has_number": 1}),
        ("sum of copay amounts", "aggregate",
         {"has_count_word": 0, "has_agg_word": 1, "has_group_word": 0, "has_filter_word": 0, "has_top_word": 0, "has_trend_word": 0, "word_count": 4, "has_number": 0}),
        ("count of prescriptions by medication", "breakdown",
         {"has_count_word": 1, "has_agg_word": 0, "has_group_word": 1, "has_filter_word": 0, "has_top_word": 0, "has_trend_word": 0, "word_count": 5, "has_number": 0}),
        ("highest billed claims", "top_n",
         {"has_count_word": 0, "has_agg_word": 0, "has_group_word": 0, "has_filter_word": 0, "has_top_word": 1, "has_trend_word": 0, "word_count": 3, "has_number": 0}),
        ("revenue trend by quarter", "trend",
         {"has_count_word": 0, "has_agg_word": 0, "has_group_word": 1, "has_filter_word": 0, "has_top_word": 0, "has_trend_word": 1, "word_count": 4, "has_number": 0}),
        ("members with diabetes diagnosis", "filter",
         {"has_count_word": 0, "has_agg_word": 0, "has_group_word": 0, "has_filter_word": 1, "has_top_word": 0, "has_trend_word": 0, "word_count": 4, "has_number": 0}),
        ("total encounters this year", "count",
         {"has_count_word": 0, "has_agg_word": 1, "has_group_word": 0, "has_filter_word": 1, "has_top_word": 0, "has_trend_word": 0, "word_count": 4, "has_number": 0}),
    ]

    w2v_corpus = [
        "how many claims were denied in the northern region",
        "total billed amount for all paid claims",
        "show claim status distribution by region",
        "average cost per encounter for emergency visits",
        "top providers by claim count in southern california",
        "denied claims trend over the last six months",
        "number of members enrolled in hmo plan type",
        "prescription fill rate by medication category",
        "referral completion rate by specialty department",
        "total paid amount versus billed amount by provider",
        "encounters per member by age group and region",
        "diagnosis code frequency for diabetes and hypertension",
        "claim processing time trend by quarter",
        "revenue per member by plan type and region",
        "provider load balance across facilities",
        "member retention rate by enrollment year",
        "copay amount distribution by visit type",
        "highest cost diagnoses by department",
        "new member acquisition by region per month",
        "claim denial reasons by payer and provider",
        "average length of stay by diagnosis category",
        "medication adherence rate by patient age group",
        "emergency department utilization by region and time",
        "preventive care encounter frequency by member plan",
        "surgical procedure cost comparison across facilities",
    ]

    return intent_data, w2v_corpus


if __name__ == '__main__':
    print("=" * 70)
    print("ADVANCED ML MODELS — PURE PYTHON (ZERO DEPENDENCIES)")
    print("=" * 70)

    t0 = time.time()
    intent_data, w2v_corpus = get_healthcare_training_data()

    print("\n[1/9] TF-IDF Vectorizer")
    tfidf = TFIDFVectorizer(max_features=200)
    docs = [q for q, _, _ in intent_data] + w2v_corpus
    tfidf.fit(docs)
    print(f"  Vocabulary: {len(tfidf.vocabulary)} terms")
    print(f"  IDF range: [{min(tfidf.idf.values()):.3f}, {max(tfidf.idf.values()):.3f}]")

    test_q = "how many denied claims"
    top_terms = tfidf.get_top_terms(test_q, n=5)
    print(f"  Top terms for '{test_q}': {top_terms}")

    print("\n[2/9] Building feature vectors...")
    feature_keys = ["has_count_word", "has_agg_word", "has_group_word",
                    "has_filter_word", "has_top_word", "has_trend_word",
                    "word_count", "has_number"]
    X_structured = [[d[2][k] for k in feature_keys] for d in intent_data]

    X_tfidf = [tfidf.transform(d[0]) for d in intent_data]
    X_combined = [X_structured[i] + X_tfidf[i] for i in range(len(intent_data))]

    labels = [d[1] for d in intent_data]
    unique_labels = sorted(set(labels))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    y_idx = [label_to_idx[l] for l in labels]

    print(f"  Features per sample: {len(X_combined[0])} ({len(feature_keys)} structured + {len(X_tfidf[0])} TF-IDF)")
    print(f"  Classes: {unique_labels}")

    print("\n[3/9] K-Nearest Neighbors (k=3)")
    knn = KNNClassifier(k=3)
    knn.fit(X_combined, labels)
    test_vec = X_combined[0]
    pred, conf = knn.predict(test_vec)
    print(f"  Test prediction: '{pred}' (confidence: {conf:.2f})")

    print("\n[4/9] Decision Tree (ID3, max_depth=5)")
    dt = DecisionTree(max_depth=5)
    dt.fit(X_combined, labels)
    print("  Tree structure:")
    dt.print_tree(feature_names=feature_keys + [f"tfidf_{i}" for i in range(len(X_tfidf[0]))])
    dt_preds = dt.predict_batch(X_combined)
    dt_acc = sum(1 for p, t in zip(dt_preds, labels) if p == t) / len(labels)
    print(f"  Training accuracy: {dt_acc:.3f}")

    print("\n[5/9] Random Forest (10 trees, max_depth=6)")
    rf = RandomForest(n_trees=10, max_depth=6)
    rf.fit(X_combined, labels)
    rf_correct = 0
    for i in range(len(X_combined)):
        pred, _ = rf.predict(X_combined[i])
        if pred == labels[i]:
            rf_correct += 1
    print(f"  Training accuracy: {rf_correct/len(labels):.3f}")

    print("\n[6/9] K-Means Clustering (k=4)")
    km = KMeans(k=4)
    km.fit(X_combined)
    print(f"  Inertia: {km.inertia:.2f}")
    cluster_dist = Counter(km.labels)
    print(f"  Cluster sizes: {dict(cluster_dist)}")

    for c in range(4):
        members = [labels[i] for i in range(len(labels)) if km.labels[i] == c]
        print(f"    Cluster {c}: {Counter(members)}")

    print("\n[7/9] Word2Vec (Skip-gram, dim=32, window=3)")
    w2v = Word2Vec(embed_dim=32, window=3, n_negative=5, lr=0.025, epochs=3)
    w2v.fit(w2v_corpus)
    print(f"  Vocabulary: {len(w2v.vocab)} words")

    for test_word in ["claims", "region", "provider", "cost"]:
        similar = w2v.most_similar(test_word, top_n=3)
        if similar:
            print(f"  Similar to '{test_word}': {[(w, f'{s:.3f}') for w, s in similar]}")

    print("\n[8/9] Neural Network ({} -> 32 -> 16 -> {})".format(
        len(X_combined[0]), len(unique_labels)))
    nn = NeuralNetwork(
        layer_sizes=[len(X_combined[0]), 32, 16, len(unique_labels)],
        activation='relu', lr=0.005, momentum=0.9
    )
    loss_history = nn.fit(X_combined, y_idx, epochs=100, verbose=True)
    nn_correct = sum(1 for i in range(len(X_combined))
                     if nn.predict(X_combined[i])[0] == y_idx[i])
    print(f"  Final accuracy: {nn_correct/len(X_combined):.3f}")
    print(f"  Loss: {loss_history[0]:.4f} → {loss_history[-1]:.4f}")

    print("\n[9/9] Logistic Regression")
    lr_model = LogisticRegression(len(X_combined[0]), len(unique_labels), lr=0.01)
    lr_loss = lr_model.fit(X_combined, y_idx, epochs=100, verbose=True)
    lr_correct = sum(1 for i in range(len(X_combined))
                     if lr_model.predict(X_combined[i])[0] == y_idx[i])
    print(f"  Final accuracy: {lr_correct/len(X_combined):.3f}")

    print("\n[BONUS] PCA (2 components)")
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_combined)
    print(f"  Explained variance: {[f'{v:.2f}' for v in pca.explained_variance]}")
    print(f"  Sample projections: {[(f'{x[0]:.2f}', f'{x[1]:.2f}') for x in X_2d[:5]]}")

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("SUMMARY — 9 MODELS TRAINED IN {:.1f}s".format(elapsed))
    print("=" * 70)
    print("""
    Model                  | Type              | Technique
    ───────────────────────┼───────────────────┼──────────────────────────
    TF-IDF Vectorizer      | Feature Eng.      | Term freq × inverse doc freq
    KNN Classifier         | Supervised        | Euclidean distance, majority vote
    Decision Tree (ID3)    | Supervised        | Information gain, entropy splits
    Random Forest          | Ensemble          | Bagging + random feature subsets
    K-Means                | Unsupervised      | K-Means++ init, centroid iteration
    Word2Vec               | Embeddings        | Skip-gram, negative sampling, SGD
    Neural Network         | Deep Learning     | Backprop, momentum, cross-entropy
    Logistic Regression    | Supervised        | Softmax, gradient descent
    PCA                    | Dim. Reduction    | Power iteration, covariance eigen

    + From ml_models.py:
    Naive Bayes            | Supervised        | Laplace smoothing, log-space Bayes
    HMM Entity Extractor   | Sequence Model    | Viterbi decoding, emission/transition
    Similarity Scorer      | Unsupervised      | Cosine + Jaccard + char n-grams
    Feedback Learner       | Online Learning   | Incremental retraining

    + From scratch_llm.py:
    DeepSeek Transformer   | Generative LM     | MoE + MLA + BPE + full backprop
    BPE Tokenizer          | Tokenization      | Byte-pair encoding, merge rules
    AdamW Optimizer        | Optimization      | Adaptive LR, weight decay
    Cosine LR Scheduler    | Scheduling        | Warmup + cosine annealing
    Beam Search            | Decoding          | Width-first, log-prob scoring
    KV-Cache               | Inference         | Cached key/value for autoregressive

    Total: 18+ ML/AI techniques, ALL from scratch, ZERO dependencies
    """)
    print("=" * 70)
