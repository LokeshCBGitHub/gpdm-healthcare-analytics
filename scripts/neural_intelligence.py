import numpy as np
import math
import json
import sqlite3
import logging
import time
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger('gpdm.neural')

class LearnedEmbeddings:

    def __init__(self, dim: int = 64, window: int = 3):
        self.dim = dim
        self.window = window
        self.vocab: Dict[str, int] = {}
        self.inv_vocab: Dict[int, str] = {}
        self.embeddings: Optional[np.ndarray] = None
        self._cooccurrence: Optional[np.ndarray] = None
        self._word_counts: Dict[str, int] = {}
        self._trained = False

    def _tokenize(self, text: str) -> List[str]:
        import re
        return [t.lower() for t in re.findall(r'[a-z][a-z0-9_]*', text.lower()) if len(t) > 1]

    def build_vocab(self, documents: List[str], min_count: int = 1):
        from collections import Counter
        word_freq = Counter()
        for doc in documents:
            word_freq.update(self._tokenize(doc))

        self.vocab = {}
        self.inv_vocab = {}
        for word, count in word_freq.most_common():
            if count >= min_count:
                idx = len(self.vocab)
                self.vocab[word] = idx
                self.inv_vocab[idx] = word
                self._word_counts[word] = count

    def train(self, documents: List[str]):
        V = len(self.vocab)
        if V == 0:
            self.build_vocab(documents)
            V = len(self.vocab)

        if V < 2:
            self.embeddings = np.random.randn(max(V, 1), self.dim) * 0.1
            self._trained = True
            return

        cooc = np.zeros((V, V), dtype=np.float32)
        for doc in documents:
            tokens = self._tokenize(doc)
            indices = [self.vocab[t] for t in tokens if t in self.vocab]
            for i, idx_i in enumerate(indices):
                for j in range(max(0, i - self.window), min(len(indices), i + self.window + 1)):
                    if i != j:
                        idx_j = indices[j]
                        weight = 1.0 / max(1, abs(i - j))
                        cooc[idx_i, idx_j] += weight

        self._cooccurrence = cooc

        total = cooc.sum() + 1e-10
        row_sums = cooc.sum(axis=1, keepdims=True) + 1e-10
        col_sums = cooc.sum(axis=0, keepdims=True) + 1e-10

        pmi = np.log(cooc * total / (row_sums * col_sums) + 1e-10)
        ppmi = np.maximum(pmi, 0)

        actual_dim = min(self.dim, V - 1, ppmi.shape[0] - 1)
        if actual_dim < 1:
            self.embeddings = np.random.randn(V, self.dim) * 0.1
            self._trained = True
            return

        try:
            U, S, Vt = np.linalg.svd(ppmi, full_matrices=False)
            self.embeddings = U[:, :actual_dim] * np.sqrt(S[:actual_dim])
            if actual_dim < self.dim:
                padding = np.zeros((V, self.dim - actual_dim))
                self.embeddings = np.hstack([self.embeddings, padding])
        except np.linalg.LinAlgError:
            self.embeddings = np.random.randn(V, self.dim) * 0.1

        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10
        self.embeddings = self.embeddings / norms
        self._trained = True
        logger.info("Embeddings trained: vocab=%d, dim=%d", V, self.dim)

    def encode_word(self, word: str) -> np.ndarray:
        word = word.lower()
        if word in self.vocab and self._trained:
            return self.embeddings[self.vocab[word]]
        return self._oov_embedding(word)

    def _oov_embedding(self, word: str) -> np.ndarray:
        if not self._trained or self.embeddings is None:
            return np.zeros(self.dim)

        word_chars = set(word)
        similarities = []
        for v_word, idx in self.vocab.items():
            overlap = len(word_chars & set(v_word)) / max(len(word_chars | set(v_word)), 1)
            if overlap > 0.3:
                similarities.append((idx, overlap))

        if not similarities:
            return np.zeros(self.dim)

        total_weight = sum(s for _, s in similarities)
        result = np.zeros(self.dim)
        for idx, sim in similarities:
            result += self.embeddings[idx] * sim
        return result / (total_weight + 1e-10)

    def encode_sequence(self, text: str) -> np.ndarray:
        tokens = self._tokenize(text)
        if not tokens:
            return np.zeros((1, self.dim))
        return np.array([self.encode_word(t) for t in tokens])

    def encode_sentence(self, text: str) -> np.ndarray:
        seq = self.encode_sequence(text)
        return seq.mean(axis=0)

    def similarity(self, word_a: str, word_b: str) -> float:
        a = self.encode_word(word_a)
        b = self.encode_word(word_b)
        norm = (np.linalg.norm(a) * np.linalg.norm(b))
        if norm < 1e-10:
            return 0.0
        return float(np.dot(a, b) / norm)

    def update_incremental(self, new_documents: List[str]):
        for doc in new_documents:
            for token in self._tokenize(doc):
                if token not in self.vocab:
                    idx = len(self.vocab)
                    self.vocab[token] = idx
                    self.inv_vocab[idx] = token
                    self._word_counts[token] = 0
                self._word_counts[token] = self._word_counts.get(token, 0) + 1

        all_docs = new_documents
        if self.embeddings is not None:
            old_embeddings = self.embeddings.copy()
            old_vocab_size = old_embeddings.shape[0]
        else:
            old_embeddings = None
            old_vocab_size = 0

        self.train(all_docs)

        if old_embeddings is not None and self.embeddings is not None:
            blend_size = min(old_vocab_size, self.embeddings.shape[0])
            if blend_size > 0 and old_embeddings.shape[1] == self.embeddings.shape[1]:
                alpha = 0.3
                self.embeddings[:blend_size] = (
                    (1 - alpha) * old_embeddings[:blend_size] +
                    alpha * self.embeddings[:blend_size]
                )

class PositionalEncoding:

    def __init__(self, dim: int, max_len: int = 200):
        self.encoding = np.zeros((max_len, dim))
        positions = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, dim, 2) * -(math.log(10000.0) / dim))

        self.encoding[:, 0::2] = np.sin(positions * div_term)
        self.encoding[:, 1::2] = np.cos(positions * div_term[:dim // 2])

    def encode(self, seq_len: int) -> np.ndarray:
        return self.encoding[:seq_len]


class SelfAttention:

    def __init__(self, dim: int, num_heads: int = 4, dropout_rate: float = 0.0):
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout_rate = dropout_rate

        scale = np.sqrt(2.0 / dim)
        self.W_q = np.random.randn(dim, dim).astype(np.float32) * scale
        self.W_k = np.random.randn(dim, dim).astype(np.float32) * scale
        self.W_v = np.random.randn(dim, dim).astype(np.float32) * scale
        self.W_o = np.random.randn(dim, dim).astype(np.float32) * scale

        self.ln_gamma = np.ones(dim, dtype=np.float32)
        self.ln_beta = np.zeros(dim, dtype=np.float32)

        self.ff_W1 = np.random.randn(dim, dim * 4).astype(np.float32) * scale
        self.ff_b1 = np.zeros(dim * 4, dtype=np.float32)
        self.ff_W2 = np.random.randn(dim * 4, dim).astype(np.float32) * scale
        self.ff_b2 = np.zeros(dim, dtype=np.float32)

        self.ff_ln_gamma = np.ones(dim, dtype=np.float32)
        self.ff_ln_beta = np.zeros(dim, dtype=np.float32)

    def _layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True) + 1e-6
        return gamma * (x - mean) / std + beta

    def _gelu(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        x_max = x.max(axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / (exp_x.sum(axis=axis, keepdims=True) + 1e-10)

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        seq_len = x.shape[0]

        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        Q = Q.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)
        K = K.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)
        V = V.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)

        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(self.head_dim)

        if mask is not None:
            scores = scores + mask * (-1e9)

        attn_weights = self._softmax(scores, axis=-1)

        context = np.matmul(attn_weights, V)

        context = context.transpose(1, 0, 2).reshape(seq_len, self.dim)

        output = context @ self.W_o

        output = self._layer_norm(x + output, self.ln_gamma, self.ln_beta)

        ff_out = self._gelu(output @ self.ff_W1 + self.ff_b1) @ self.ff_W2 + self.ff_b2
        output = self._layer_norm(output + ff_out, self.ff_ln_gamma, self.ff_ln_beta)

        return output, attn_weights

    def get_attention_map(self, x: np.ndarray) -> np.ndarray:
        _, attn_weights = self.forward(x)
        return attn_weights.mean(axis=0)

    def update_weights(self, gradients: Dict[str, np.ndarray], lr: float = 0.001):
        for param_name, grad in gradients.items():
            param = getattr(self, param_name, None)
            if param is not None:
                setattr(self, param_name, param - lr * grad)


class TransformerEncoder:

    def __init__(self, dim: int = 64, num_layers: int = 2, num_heads: int = 4):
        self.dim = dim
        self.num_layers = num_layers
        self.layers = [SelfAttention(dim, num_heads) for _ in range(num_layers)]
        self.pos_encoding = PositionalEncoding(dim)

    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        seq_len = embeddings.shape[0]

        x = embeddings + self.pos_encoding.encode(seq_len)

        all_attention = []
        for layer in self.layers:
            x, attn = layer.forward(x)
            all_attention.append(attn)

        return x, all_attention

    def encode_pooled(self, embeddings: np.ndarray) -> np.ndarray:
        encoded, _ = self.encode(embeddings)
        weights = np.ones(encoded.shape[0]) / encoded.shape[0]
        return (encoded * weights[:, np.newaxis]).sum(axis=0)

    def save_weights(self) -> Dict[str, Any]:
        weights = {}
        for i, layer in enumerate(self.layers):
            prefix = f'layer_{i}'
            for attr in ['W_q', 'W_k', 'W_v', 'W_o', 'ln_gamma', 'ln_beta',
                         'ff_W1', 'ff_b1', 'ff_W2', 'ff_b2', 'ff_ln_gamma', 'ff_ln_beta']:
                weights[f'{prefix}.{attr}'] = getattr(layer, attr).tolist()
        return weights

    def load_weights(self, weights: Dict[str, Any]):
        for i, layer in enumerate(self.layers):
            prefix = f'layer_{i}'
            for attr in ['W_q', 'W_k', 'W_v', 'W_o', 'ln_gamma', 'ln_beta',
                         'ff_W1', 'ff_b1', 'ff_W2', 'ff_b2', 'ff_ln_gamma', 'ff_ln_beta']:
                key = f'{prefix}.{attr}'
                if key in weights:
                    setattr(layer, attr, np.array(weights[key], dtype=np.float32))

class ModernHopfieldNetwork:

    def __init__(self, dim: int, max_patterns: int = 10000,
                 beta: float = 8.0, decay_rate: float = 0.001):
        self.dim = dim
        self.max_patterns = max_patterns
        self.beta = beta
        self.decay_rate = decay_rate

        self.patterns: Optional[np.ndarray] = None
        self.pattern_metadata: List[Dict] = []
        self.pattern_strengths: Optional[np.ndarray] = None
        self.pattern_timestamps: List[float] = []
        self.pattern_access_counts: List[int] = []

        self.num_patterns = 0

    def store(self, pattern: np.ndarray, metadata: Dict[str, Any]):
        pattern = pattern.flatten()[:self.dim]
        if len(pattern) < self.dim:
            pattern = np.pad(pattern, (0, self.dim - len(pattern)))

        norm = np.linalg.norm(pattern)
        if norm > 1e-10:
            pattern = pattern / norm

        if self.patterns is None:
            self.patterns = pattern.reshape(1, -1)
            self.pattern_strengths = np.array([1.0])
        else:
            similarities = self.patterns @ pattern
            max_sim_idx = np.argmax(similarities)
            max_sim = similarities[max_sim_idx]

            if max_sim > 0.95:
                self.pattern_strengths[max_sim_idx] += 0.5
                self.pattern_access_counts[max_sim_idx] += 1
                self.pattern_timestamps[max_sim_idx] = time.time()
                old_meta = self.pattern_metadata[max_sim_idx]
                old_meta.setdefault('aliases', []).append(metadata.get('question', ''))
                return

            if self.num_patterns >= self.max_patterns:
                self._forget_weakest()

            self.patterns = np.vstack([self.patterns, pattern.reshape(1, -1)])
            self.pattern_strengths = np.append(self.pattern_strengths, 1.0)

        self.pattern_metadata.append(metadata)
        self.pattern_timestamps.append(time.time())
        self.pattern_access_counts.append(0)
        self.num_patterns = self.patterns.shape[0]

    def recall(self, query: np.ndarray, top_k: int = 5,
               min_similarity: float = 0.3) -> List[Tuple[Dict, float]]:
        if self.patterns is None or self.num_patterns == 0:
            return []

        query = query.flatten()[:self.dim]
        if len(query) < self.dim:
            query = np.pad(query, (0, self.dim - len(query)))

        norm = np.linalg.norm(query)
        if norm > 1e-10:
            query = query / norm

        similarities = self.patterns @ query

        energies = self.beta * similarities * self.pattern_strengths

        attn = self._softmax(energies)

        weighted_scores = attn * similarities

        top_indices = np.argsort(weighted_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= min_similarity:
                self.pattern_strengths[idx] += 0.1
                self.pattern_access_counts[idx] += 1
                self.pattern_timestamps[idx] = time.time()
                results.append((self.pattern_metadata[idx], score))

        return results

    def recall_and_complete(self, query: np.ndarray, num_steps: int = 3) -> np.ndarray:
        if self.patterns is None or self.num_patterns == 0:
            return query

        x = query.flatten()[:self.dim].copy()
        if len(x) < self.dim:
            x = np.pad(x, (0, self.dim - len(x)))

        norm = np.linalg.norm(x)
        if norm > 1e-10:
            x = x / norm

        for _ in range(num_steps):
            scores = self.beta * (self.patterns @ x)
            attn = self._softmax(scores)
            x_new = self.patterns.T @ attn

            norm = np.linalg.norm(x_new)
            if norm > 1e-10:
                x_new = x_new / norm

            if np.allclose(x, x_new, atol=1e-6):
                break
            x = x_new

        return x

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x_max = x.max()
        exp_x = np.exp(x - x_max)
        return exp_x / (exp_x.sum() + 1e-10)

    def _forget_weakest(self):
        now = time.time()
        scores = np.zeros(self.num_patterns)

        for i in range(self.num_patterns):
            age = now - self.pattern_timestamps[i]
            stability = 1.0 + self.pattern_access_counts[i] * 0.5
            retention = np.exp(-age / (stability * 86400))

            scores[i] = self.pattern_strengths[i] * retention

        weakest = np.argmin(scores)
        self.patterns = np.delete(self.patterns, weakest, axis=0)
        self.pattern_strengths = np.delete(self.pattern_strengths, weakest)
        self.pattern_metadata.pop(weakest)
        self.pattern_timestamps.pop(weakest)
        self.pattern_access_counts.pop(weakest)
        self.num_patterns -= 1

    def apply_forgetting_curve(self):
        if self.patterns is None:
            return

        now = time.time()
        keep_mask = []

        for i in range(self.num_patterns):
            age = now - self.pattern_timestamps[i]
            stability = 1.0 + self.pattern_access_counts[i] * 0.5
            retention = np.exp(-age / (stability * 86400))

            self.pattern_strengths[i] *= (0.99 + 0.01 * retention)

            keep_mask.append(self.pattern_strengths[i] > 0.1)

        keep = np.array(keep_mask)
        if not keep.all():
            self.patterns = self.patterns[keep]
            self.pattern_strengths = self.pattern_strengths[keep]
            self.pattern_metadata = [m for m, k in zip(self.pattern_metadata, keep_mask) if k]
            self.pattern_timestamps = [t for t, k in zip(self.pattern_timestamps, keep_mask) if k]
            self.pattern_access_counts = [c for c, k in zip(self.pattern_access_counts, keep_mask) if k]
            self.num_patterns = self.patterns.shape[0]
            logger.info("Memory consolidation: pruned to %d patterns", self.num_patterns)

    def save(self) -> Dict[str, Any]:
        return {
            'patterns': self.patterns.tolist() if self.patterns is not None else None,
            'strengths': self.pattern_strengths.tolist() if self.pattern_strengths is not None else None,
            'metadata': self.pattern_metadata,
            'timestamps': self.pattern_timestamps,
            'access_counts': self.pattern_access_counts,
            'beta': self.beta,
            'dim': self.dim,
        }

    def load(self, data: Dict[str, Any]):
        if data.get('patterns') is not None:
            self.patterns = np.array(data['patterns'], dtype=np.float32)
            self.pattern_strengths = np.array(data['strengths'], dtype=np.float32)
            self.pattern_metadata = data['metadata']
            self.pattern_timestamps = data['timestamps']
            self.pattern_access_counts = data['access_counts']
            self.num_patterns = self.patterns.shape[0]
            self.beta = data.get('beta', self.beta)

class SchemaGNN:

    def __init__(self, dim: int = 64, num_layers: int = 2):
        self.dim = dim
        self.num_layers = num_layers

        self.W_msg = [
            np.random.randn(dim, dim).astype(np.float32) * np.sqrt(2.0 / dim)
            for _ in range(num_layers)
        ]
        self.W_upd = [
            np.random.randn(dim * 2, dim).astype(np.float32) * np.sqrt(2.0 / (dim * 2))
            for _ in range(num_layers)
        ]
        self.b_upd = [np.zeros(dim, dtype=np.float32) for _ in range(num_layers)]

        self.node_ids: List[str] = []
        self.node_features: Optional[np.ndarray] = None
        self.adjacency: Optional[np.ndarray] = None
        self.edge_types: Dict[Tuple[int, int], str] = {}

    def build_from_schema(self, learner, embeddings: LearnedEmbeddings):
        self.node_ids = []
        node_texts = []

        table_indices = {}
        for table in learner.tables:
            idx = len(self.node_ids)
            table_indices[table] = idx
            self.node_ids.append(f"table:{table}")
            node_texts.append(table.replace('_', ' '))

        col_indices = {}
        for table, profiles in learner.tables.items():
            for p in profiles:
                idx = len(self.node_ids)
                col_id = f"{table}.{p.name}"
                col_indices[col_id] = idx
                self.node_ids.append(f"col:{col_id}")
                text = f"{p.name.replace('_', ' ')} {table} {' '.join(p.semantic_tags)}"
                node_texts.append(text)

        num_nodes = len(self.node_ids)

        self.node_features = np.zeros((num_nodes, self.dim), dtype=np.float32)
        for i, text in enumerate(node_texts):
            emb = embeddings.encode_sentence(text)
            self.node_features[i, :len(emb)] = emb[:self.dim]

        self.adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)

        for table, profiles in learner.tables.items():
            t_idx = table_indices[table]
            for p in profiles:
                c_idx = col_indices[f"{table}.{p.name}"]
                self.adjacency[t_idx, c_idx] = 1.0
                self.adjacency[c_idx, t_idx] = 1.0
                self.edge_types[(t_idx, c_idx)] = 'has_column'
                self.edge_types[(c_idx, t_idx)] = 'belongs_to'

        col_by_name = {}
        for col_id, idx in col_indices.items():
            col_name = col_id.split('.')[-1]
            if col_name not in col_by_name:
                col_by_name[col_name] = []
            col_by_name[col_name].append((col_id, idx))
        
        for col_name, col_list in col_by_name.items():
            if col_name.lower().endswith('_id') and len(col_list) > 1:
                for i in range(len(col_list)):
                    for j in range(i + 1, len(col_list)):
                        col_id1, idx1 = col_list[i]
                        col_id2, idx2 = col_list[j]
                        self.adjacency[idx1, idx2] = 2.0
                        self.adjacency[idx2, idx1] = 2.0
                        self.edge_types[(idx1, idx2)] = 'foreign_key'
                        self.edge_types[(idx2, idx1)] = 'foreign_key'

        logger.info("Schema GNN built: %d nodes, %d edges",
                     num_nodes, int((self.adjacency > 0).sum()))

    def propagate(self) -> np.ndarray:
        if self.node_features is None:
            return np.array([])

        h = self.node_features.copy()

        for l in range(self.num_layers):
            messages = h @ self.W_msg[l]

            aggregated = self.adjacency @ messages

            combined = np.hstack([h, aggregated])
            h_new = np.tanh(combined @ self.W_upd[l] + self.b_upd[l])

            h = h + h_new

            norms = np.linalg.norm(h, axis=1, keepdims=True) + 1e-10
            h = h / norms

        self.node_features = h
        return h

    def get_node_embedding(self, node_id: str) -> np.ndarray:
        if node_id in self.node_ids:
            idx = self.node_ids.index(node_id)
            return self.node_features[idx]
        return np.zeros(self.dim)

    def get_table_embedding(self, table: str) -> np.ndarray:
        embeddings = []
        for i, nid in enumerate(self.node_ids):
            if nid == f"table:{table}" or nid.startswith(f"col:{table}."):
                embeddings.append(self.node_features[i])
        if embeddings:
            return np.mean(embeddings, axis=0)
        return np.zeros(self.dim)

    def find_join_path(self, source_table: str, target_table: str,
                       max_hops: int = 3) -> List[str]:
        from collections import deque

        source_id = f"table:{source_table}"
        target_id = f"table:{target_table}"

        if source_id not in self.node_ids or target_id not in self.node_ids:
            return []

        s_idx = self.node_ids.index(source_id)
        t_idx = self.node_ids.index(target_id)

        queue = deque([(s_idx, [source_table])])
        visited = {s_idx}

        while queue:
            current, path = queue.popleft()
            if current == t_idx:
                return path
            if len(path) > max_hops:
                continue

            for neighbor in range(len(self.node_ids)):
                if self.adjacency[current, neighbor] > 0 and neighbor not in visited:
                    visited.add(neighbor)
                    nid = self.node_ids[neighbor]
                    new_path = path.copy()
                    if nid.startswith('table:'):
                        new_path.append(nid.replace('table:', ''))
                    queue.append((neighbor, new_path))

        return []

class NeuralQueryUnderstanding:

    def __init__(self, dim: int = 64, db_path: str = None):
        self.dim = dim
        self.db_path = db_path

        self.embeddings = LearnedEmbeddings(dim=dim)
        self.transformer = TransformerEncoder(dim=dim, num_layers=2, num_heads=4)
        self.hopfield = ModernHopfieldNetwork(dim=dim)
        self.gnn = SchemaGNN(dim=dim, num_layers=2)

        scale = np.sqrt(2.0 / dim)
        self.W_combine = np.random.randn(dim * 3, dim).astype(np.float32) * scale
        self.b_combine = np.zeros(dim, dtype=np.float32)

        self._initialized = False
        self._persistence_path = None

    def initialize(self, learner, documents: List[str] = None):
        if documents is None:
            documents = self._build_training_docs(learner)

        self.embeddings.build_vocab(documents)
        self.embeddings.train(documents)

        self.gnn.build_from_schema(learner, self.embeddings)
        self.gnn.propagate()

        self._initialized = True
        logger.info("Neural intelligence initialized: vocab=%d, dim=%d",
                     len(self.embeddings.vocab), self.dim)

    def _build_training_docs(self, learner) -> List[str]:
        docs = []
        for table, profiles in learner.tables.items():
            col_names = ' '.join(p.name.replace('_', ' ') for p in profiles)
            docs.append(f"{table} table with columns {col_names}")

            for p in profiles:
                tags = ' '.join(p.semantic_tags)
                doc = f"{p.name.replace('_', ' ')} column in {table} {tags}"
                if p.is_categorical and p.sample_values:
                    vals = ' '.join(str(v) for v in p.sample_values[:5])
                    doc += f" values include {vals}"
                docs.append(doc)

        query_patterns = [
            "how many total count number",
            "average mean sum total aggregate",
            "by per group breakdown distribution",
            "top highest best most ranking",
            "trend over time monthly quarterly yearly",
            "percentage rate ratio proportion",
            "greater than less than above below between",
            "compare versus difference gap",
        ]
        docs.extend(query_patterns)

        return docs

    def understand(self, question: str) -> Dict[str, Any]:
        if not self._initialized:
            return {'question_embedding': np.zeros(self.dim), 'tokens': []}

        word_embeddings = self.embeddings.encode_sequence(question)
        tokens = self.embeddings._tokenize(question)

        attended, attention_maps = self.transformer.encode(word_embeddings)
        question_vec = attended.mean(axis=0)

        recalled = self.hopfield.recall(question_vec, top_k=3)

        completed_vec = self.hopfield.recall_and_complete(question_vec)

        schema_matches = self._match_schema(question_vec)

        gnn_context = np.zeros(self.dim)
        if schema_matches:
            gnn_context = np.mean([m['embedding'] for m in schema_matches[:5]], axis=0)

        combined_input = np.concatenate([question_vec, completed_vec, gnn_context])
        combined = np.tanh(combined_input @ self.W_combine + self.b_combine)

        return {
            'question_embedding': combined,
            'transformer_embedding': question_vec,
            'hopfield_embedding': completed_vec,
            'attention_maps': attention_maps,
            'recalled_patterns': recalled,
            'schema_matches': schema_matches,
            'tokens': tokens,
            'word_embeddings': word_embeddings,
        }

    def _match_schema(self, question_vec: np.ndarray) -> List[Dict]:
        if self.gnn.node_features is None:
            return []

        similarities = self.gnn.node_features @ question_vec
        top_indices = np.argsort(similarities)[::-1][:10]

        matches = []
        for idx in top_indices:
            if similarities[idx] > 0.1:
                node_id = self.gnn.node_ids[idx]
                matches.append({
                    'node_id': node_id,
                    'similarity': float(similarities[idx]),
                    'embedding': self.gnn.node_features[idx],
                })
        return matches

    def learn_from_interaction(self, question: str, sql: str,
                                feedback: float = 1.0,
                                metadata: Dict = None):
        if not self._initialized:
            return

        word_embs = self.embeddings.encode_sequence(question)
        attended, _ = self.transformer.encode(word_embs)
        question_vec = attended.mean(axis=0)

        pattern_meta = {
            'question': question,
            'sql': sql,
            'feedback': feedback,
            'timestamp': time.time(),
            **(metadata or {}),
        }
        self.hopfield.store(question_vec, pattern_meta)

        self.embeddings.update_incremental([question, sql])

    def consolidate_memory(self):
        self.hopfield.apply_forgetting_curve()

    def save_state(self, db_path: str):
        conn = sqlite3.connect(db_path)
        conn.execute('''CREATE TABLE IF NOT EXISTS neural_state (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at REAL
        )''')

        hopfield_data = json.dumps(self.hopfield.save())
        conn.execute('''INSERT OR REPLACE INTO neural_state (key, value, updated_at)
                        VALUES (?, ?, ?)''', ('hopfield', hopfield_data, time.time()))

        transformer_data = json.dumps(self.transformer.save_weights())
        conn.execute('''INSERT OR REPLACE INTO neural_state (key, value, updated_at)
                        VALUES (?, ?, ?)''', ('transformer', transformer_data, time.time()))

        vocab_data = json.dumps({
            'vocab': self.embeddings.vocab,
            'embeddings': self.embeddings.embeddings.tolist() if self.embeddings.embeddings is not None else None,
        })
        conn.execute('''INSERT OR REPLACE INTO neural_state (key, value, updated_at)
                        VALUES (?, ?, ?)''', ('embeddings', vocab_data, time.time()))

        conn.commit()
        conn.close()
        logger.info("Neural state saved to %s", db_path)

    def load_state(self, db_path: str) -> bool:
        try:
            conn = sqlite3.connect(db_path)
            rows = conn.execute('SELECT key, value FROM neural_state').fetchall()
            conn.close()

            state = {k: json.loads(v) for k, v in rows}

            if 'hopfield' in state:
                self.hopfield.load(state['hopfield'])

            if 'transformer' in state:
                self.transformer.load_weights(state['transformer'])

            if 'embeddings' in state:
                emb_data = state['embeddings']
                self.embeddings.vocab = emb_data.get('vocab', {})
                self.embeddings.inv_vocab = {v: k for k, v in self.embeddings.vocab.items()}
                if emb_data.get('embeddings'):
                    self.embeddings.embeddings = np.array(emb_data['embeddings'], dtype=np.float32)
                    self.embeddings._trained = True

            logger.info("Neural state restored: %d Hopfield patterns, %d vocab words",
                         self.hopfield.num_patterns, len(self.embeddings.vocab))
            return True
        except Exception as e:
            logger.debug("Could not load neural state: %s", e)
            return False
