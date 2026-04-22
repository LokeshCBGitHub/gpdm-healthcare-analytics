import os
import re
import json
import math
import time
import sqlite3
import logging
import hashlib
import pickle
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter

import numpy as np

logger = logging.getLogger('gpdm.transformer')


class HealthcareBPETokenizer:

    PAD_TOKEN = '<pad>'
    UNK_TOKEN = '<unk>'
    BOS_TOKEN = '<bos>'
    EOS_TOKEN = '<eos>'
    SEP_TOKEN = '<sep>'

    def __init__(self, vocab_size: int = 2048):
        self.vocab_size = vocab_size
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self._trained = False

    def train(self, corpus: List[str], min_freq: int = 2):
        t0 = time.time()

        self.token_to_id = {
            self.PAD_TOKEN: 0, self.UNK_TOKEN: 1,
            self.BOS_TOKEN: 2, self.EOS_TOKEN: 3, self.SEP_TOKEN: 4,
        }

        all_chars = set()
        for text in corpus:
            all_chars.update(text.lower())
        for ch in sorted(all_chars):
            if ch not in self.token_to_id:
                self.token_to_id[ch] = len(self.token_to_id)

        healthcare_tokens = [
            'claim', 'claims', 'member', 'members', 'provider', 'providers',
            'encounter', 'encounters', 'diagnosis', 'diagnoses', 'prescription',
            'referral', 'appointment', 'denied', 'denial', 'approved', 'pending',
            'emergency', 'inpatient', 'outpatient', 'telehealth',
            'paid', 'billed', 'cost', 'amount', 'copay', 'deductible',
            'pmpm', 'per_member', 'revenue',
            'count', 'total', 'average', 'sum', 'rate', 'percentage',
            'trend', 'monthly', 'quarterly', 'yearly', 'compare',
            'top', 'highest', 'lowest', 'most', 'least',
            'by', 'per', 'for', 'each', 'group',
            'region', 'plan_type', 'visit_type', 'specialty', 'department',
            'facility', 'risk_score', 'medication', 'status',
            'hmo', 'ppo', 'ncal', 'scal',
            'select', 'from', 'where', 'group_by', 'order_by', 'having',
            'join', 'inner', 'left', 'count(*)', 'sum(', 'avg(', 'max(', 'min(',
        ]
        for token in healthcare_tokens:
            if token not in self.token_to_id and len(self.token_to_id) < self.vocab_size:
                self.token_to_id[token] = len(self.token_to_id)

        word_freqs = Counter()
        for text in corpus:
            words = text.lower().split()
            for word in words:
                word_freqs[' '.join(list(word)) + ' </w>'] += 1

        self.merges = []
        for merge_idx in range(min(500, self.vocab_size - len(self.token_to_id))):
            pair_freqs = Counter()
            for word, freq in word_freqs.items():
                symbols = word.split()
                for i in range(len(symbols) - 1):
                    pair_freqs[(symbols[i], symbols[i + 1])] += freq

            if not pair_freqs:
                break

            best_pair = max(pair_freqs, key=pair_freqs.get)
            if pair_freqs[best_pair] < min_freq:
                break

            merged = best_pair[0] + best_pair[1]
            self.merges.append(best_pair)

            if merged.replace('</w>', '') not in self.token_to_id and len(self.token_to_id) < self.vocab_size:
                self.token_to_id[merged.replace('</w>', '')] = len(self.token_to_id)

            new_word_freqs = Counter()
            for word, freq in word_freqs.items():
                new_word = word.replace(f'{best_pair[0]} {best_pair[1]}', merged)
                new_word_freqs[new_word] = freq
            word_freqs = new_word_freqs

        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self._trained = True

        elapsed = (time.time() - t0) * 1000
        logger.info("BPE tokenizer trained: vocab_size=%d, merges=%d in %.0fms",
                     len(self.token_to_id), len(self.merges), elapsed)

    def encode(self, text: str) -> List[int]:
        if not self._trained:
            return [1]

        tokens = [self.token_to_id[self.BOS_TOKEN]]
        words = text.lower().split()

        for word in words:
            if word in self.token_to_id:
                tokens.append(self.token_to_id[word])
                continue

            symbols = list(word)
            for left, right in self.merges:
                i = 0
                new_symbols = []
                while i < len(symbols):
                    if i < len(symbols) - 1:
                        pair = symbols[i] + symbols[i + 1]
                        merged_clean = (left + right).replace('</w>', '')
                        if pair == merged_clean:
                            new_symbols.append(pair)
                            i += 2
                            continue
                    new_symbols.append(symbols[i])
                    i += 1
                symbols = new_symbols

            for sym in symbols:
                if sym in self.token_to_id:
                    tokens.append(self.token_to_id[sym])
                else:
                    for ch in sym:
                        tokens.append(self.token_to_id.get(ch, 1))

        tokens.append(self.token_to_id[self.EOS_TOKEN])
        return tokens

    def decode(self, ids: List[int]) -> str:
        tokens = []
        for i in ids:
            tok = self.id_to_token.get(i, '')
            if tok in (self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.SEP_TOKEN):
                continue
            if tok == self.UNK_TOKEN:
                tokens.append('?')
            else:
                tokens.append(tok)
        return ' '.join(tokens)

    @property
    def actual_vocab_size(self) -> int:
        return len(self.token_to_id)


class RotaryPositionalEmbedding:

    def __init__(self, dim: int, max_seq_len: int = 512, base: float = 10000.0):
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        freqs = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))

        positions = np.arange(max_seq_len, dtype=np.float32)
        angles = np.outer(positions, freqs)

        self.cos_cache = np.cos(angles).astype(np.float32)
        self.sin_cache = np.sin(angles).astype(np.float32)

    def apply(self, x: np.ndarray, start_pos: int = 0) -> np.ndarray:
        if x.ndim == 2:
            seq_len, dim = x.shape
        else:
            _, seq_len, dim = x.shape

        half_dim = dim // 2
        cos = self.cos_cache[start_pos:start_pos + seq_len, :half_dim]
        sin = self.sin_cache[start_pos:start_pos + seq_len, :half_dim]

        if x.ndim == 2:
            x1 = x[:, :half_dim]
            x2 = x[:, half_dim:2 * half_dim]
        else:
            x1 = x[:, :, :half_dim]
            x2 = x[:, :, half_dim:2 * half_dim]

        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        if x.ndim == 2:
            result = np.concatenate([rotated_x1, rotated_x2], axis=-1)
            if dim > 2 * half_dim:
                result = np.concatenate([result, x[:, 2 * half_dim:]], axis=-1)
        else:
            result = np.concatenate([rotated_x1, rotated_x2], axis=-1)
            if dim > 2 * half_dim:
                result = np.concatenate([result, x[:, :, 2 * half_dim:]], axis=-1)

        return result.astype(np.float32)


class RMSNorm:

    def __init__(self, dim: int, eps: float = 1e-6):
        self.eps = eps
        self.weight = np.ones(dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return (x / rms) * self.weight


class MultiHeadAttention:

    def __init__(self, d_model: int, n_heads: int):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        scale_qk = np.sqrt(2.0 / (d_model + self.d_head))
        scale_v = np.sqrt(2.0 / (d_model + self.d_head))

        self.W_q = np.random.randn(d_model, d_model).astype(np.float32) * scale_qk
        self.W_k = np.random.randn(d_model, d_model).astype(np.float32) * scale_qk
        self.W_v = np.random.randn(d_model, d_model).astype(np.float32) * scale_v
        self.W_o = np.random.randn(d_model, d_model).astype(np.float32) * np.sqrt(2.0 / d_model)

        self.rope = RotaryPositionalEmbedding(self.d_head)

        self._attn_weights = None

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        seq_len = x.shape[0]

        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        Q = Q.reshape(seq_len, self.n_heads, self.d_head).transpose(1, 0, 2)
        K = K.reshape(seq_len, self.n_heads, self.d_head).transpose(1, 0, 2)
        V = V.reshape(seq_len, self.n_heads, self.d_head).transpose(1, 0, 2)

        for h in range(self.n_heads):
            Q[h] = self.rope.apply(Q[h])
            K[h] = self.rope.apply(K[h])

        scale = np.sqrt(self.d_head).astype(np.float32)
        attn_scores = np.matmul(Q, K.transpose(0, 2, 1)) / scale

        if mask is not None:
            attn_scores = attn_scores + mask[np.newaxis, :, :]

        attn_scores_max = np.max(attn_scores, axis=-1, keepdims=True)
        exp_scores = np.exp(attn_scores - attn_scores_max)
        attn_weights = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-10)

        self._attn_weights = attn_weights

        attn_output = np.matmul(attn_weights, V)

        attn_output = attn_output.transpose(1, 0, 2).reshape(seq_len, self.d_model)

        return attn_output @ self.W_o

    def get_attention_weights(self) -> Optional[np.ndarray]:
        return self._attn_weights


class SwiGLUFeedForward:

    def __init__(self, d_model: int, d_ff: int = None):
        self.d_model = d_model
        self.d_ff = d_ff or d_model * 4

        scale = np.sqrt(2.0 / (d_model + self.d_ff))
        self.W1 = np.random.randn(d_model, self.d_ff).astype(np.float32) * scale
        self.W3 = np.random.randn(d_model, self.d_ff).astype(np.float32) * scale
        self.W2 = np.random.randn(self.d_ff, d_model).astype(np.float32) * scale

    def forward(self, x: np.ndarray) -> np.ndarray:
        gate = x @ self.W1
        swish_gate = gate * (1.0 / (1.0 + np.exp(-gate.clip(-20, 20))))

        value = x @ self.W3

        hidden = swish_gate * value

        return hidden @ self.W2


class TransformerBlock:

    def __init__(self, d_model: int, n_heads: int, d_ff: int = None):
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.ffn = SwiGLUFeedForward(d_model, d_ff)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        normed = self.norm1.forward(x)
        attn_out = self.attention.forward(normed, mask)
        x = x + attn_out

        normed = self.norm2.forward(x)
        ff_out = self.ffn.forward(normed)
        x = x + ff_out

        return x


class HealthcareTransformer:

    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 8,
                 n_layers: int = 3, d_ff: int = 512, max_seq_len: int = 128,
                 n_intents: int = 10):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.n_intents = n_intents

        self.token_embeddings = np.random.randn(vocab_size, d_model).astype(np.float32) * 0.02

        self.blocks = [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]

        self.final_norm = RMSNorm(d_model)

        self.intent_head = np.random.randn(d_model, n_intents).astype(np.float32) * np.sqrt(2.0 / d_model)
        self.intent_bias = np.zeros(n_intents, dtype=np.float32)

        self.n_entity_tags = 9
        self.entity_head = np.random.randn(d_model, self.n_entity_tags).astype(np.float32) * np.sqrt(2.0 / d_model)
        self.entity_bias = np.zeros(self.n_entity_tags, dtype=np.float32)

        self.confidence_head = np.random.randn(d_model, 1).astype(np.float32) * np.sqrt(2.0 / d_model)

    def forward(self, token_ids: List[int]) -> Dict[str, np.ndarray]:
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len]

        seq_len = len(token_ids)

        x = np.array([self.token_embeddings[tid] for tid in token_ids], dtype=np.float32)
        x = x * np.sqrt(self.d_model).astype(np.float32)

        attention_weights = []
        for block in self.blocks:
            x = block.forward(x)
            attention_weights.append(block.attention.get_attention_weights())

        x = self.final_norm.forward(x)

        pooled = np.mean(x, axis=0)

        intent_logits = pooled @ self.intent_head + self.intent_bias

        entity_logits = x @ self.entity_head + self.entity_bias

        conf_raw = float(pooled @ self.confidence_head)
        confidence = 1.0 / (1.0 + np.exp(-conf_raw))

        return {
            'hidden': x,
            'pooled': pooled,
            'intent_logits': intent_logits,
            'entity_logits': entity_logits,
            'confidence': float(confidence),
            'attention_weights': attention_weights,
        }

    def train_step(self, token_ids: List[int], target_intent: int,
                   target_entities: Optional[List[int]] = None,
                   lr: float = 0.001) -> float:
        output = self.forward(token_ids)
        intent_logits = output['intent_logits']
        pooled = output['pooled']

        exp_logits = np.exp(intent_logits - intent_logits.max())
        probs = exp_logits / exp_logits.sum()

        loss = -np.log(probs[target_intent] + 1e-10)

        d_logits = probs.copy()
        d_logits[target_intent] -= 1.0

        d_intent_head = np.outer(pooled, d_logits)
        self.intent_head -= lr * d_intent_head
        self.intent_bias -= lr * d_logits

        d_pooled = d_logits @ self.intent_head.T

        seq_len = len(token_ids)
        d_per_token = d_pooled / seq_len

        for i, tid in enumerate(token_ids):
            self.token_embeddings[tid] -= lr * d_per_token

        for block in self.blocks:
            noise_scale = lr * loss * 0.01
            if noise_scale > 0.001:
                noise_scale = 0.001
            block.attention.W_q -= noise_scale * np.random.randn(*block.attention.W_q.shape).astype(np.float32)
            block.attention.W_k -= noise_scale * np.random.randn(*block.attention.W_k.shape).astype(np.float32)
            block.attention.W_v -= noise_scale * np.random.randn(*block.attention.W_v.shape).astype(np.float32)

        if target_entities is not None and len(target_entities) == seq_len:
            hidden = output['hidden']
            entity_logits = output['entity_logits']

            for t in range(seq_len):
                exp_e = np.exp(entity_logits[t] - entity_logits[t].max())
                e_probs = exp_e / exp_e.sum()
                d_e = e_probs.copy()
                d_e[target_entities[t]] -= 1.0

                self.entity_head -= lr * np.outer(hidden[t], d_e)
                self.entity_bias -= lr * d_e

        return float(loss)

    def get_param_count(self) -> int:
        count = self.token_embeddings.size
        for block in self.blocks:
            count += block.attention.W_q.size + block.attention.W_k.size
            count += block.attention.W_v.size + block.attention.W_o.size
            count += block.ffn.W1.size + block.ffn.W2.size + block.ffn.W3.size
            count += block.norm1.weight.size + block.norm2.weight.size
        count += self.final_norm.weight.size
        count += self.intent_head.size + self.intent_bias.size
        count += self.entity_head.size + self.entity_bias.size
        count += self.confidence_head.size
        return count


INTENT_LABELS = ['count', 'aggregate', 'rank', 'trend', 'rate',
                 'compare', 'list', 'summary', 'correlate', 'exists']

ENTITY_TAGS = ['O', 'T', 'C', 'V', 'A', 'F', 'G', 'L', 'M']


class TransformerTrainingData:

    def __init__(self, schema_graph):
        self.graph = schema_graph

    def generate(self) -> List[Dict]:
        examples = []

        intent_examples = self._generate_intent_examples()
        for question, intent in intent_examples:
            intent_idx = INTENT_LABELS.index(intent) if intent in INTENT_LABELS else 0
            examples.append({
                'question': question,
                'intent': intent,
                'intent_idx': intent_idx,
            })

        return examples

    def _generate_intent_examples(self) -> List[Tuple[str, str]]:
        examples = []

        for table, concept in self.graph.tables.items():
            examples.extend([
                (f"how many {concept.concept} do we have", 'count'),
                (f"count of {table}", 'count'),
                (f"total number of {concept.concept}", 'count'),
                (f"how many {concept.concept} are there", 'count'),
                (f"how many {table} in the system", 'count'),
            ])

        for table in ['claims', 'prescriptions', 'encounters']:
            money_cols = self.graph.get_money_columns(table)
            for col in money_cols:
                col_display = col.lower().replace('_', ' ')
                examples.extend([
                    (f"total {col_display}", 'aggregate'),
                    (f"average {col_display}", 'aggregate'),
                    (f"what is the sum of {col_display}", 'aggregate'),
                    (f"avg {col_display} per claim", 'aggregate'),
                    (f"what is the average {col_display}", 'aggregate'),
                    (f"maximum {col_display}", 'aggregate'),
                    (f"highest {col_display} on any single claim", 'aggregate'),
                ])

        examples.extend([
            ("what is the average risk score of our members", 'aggregate'),
            ("average risk score", 'aggregate'),
            ("average length of stay for inpatient visits", 'aggregate'),
            ("what is the average copay", 'aggregate'),
            ("total amount billed across all claims", 'aggregate'),
            ("how much do we spend on prescriptions", 'aggregate'),
            ("PMPM cost", 'aggregate'),
            ("what is our PMPM cost", 'aggregate'),
            ("what is the highest paid amount on any single claim", 'aggregate'),
        ])

        for table, concept in self.graph.tables.items():
            examples.extend([
                (f"top {concept.concept} by count", 'rank'),
                (f"which {concept.concept} has the most", 'rank'),
                (f"most common {concept.concept}", 'rank'),
            ])

        examples.extend([
            ("top 5 medications by cost", 'rank'),
            ("which specialty sees the most patients", 'rank'),
            ("top providers by claim count", 'rank'),
            ("most common diagnoses", 'rank'),
            ("which region has the highest denial rate", 'rank'),
            ("most prescribed medications", 'rank'),
            ("break down claims by plan type", 'rank'),
            ("claims by plan type", 'rank'),
            ("encounters by visit type", 'rank'),
            ("claims count per region", 'rank'),
            ("prescriptions by medication class", 'rank'),
            ("which department has the most encounters", 'rank'),
            ("which providers have the most claims", 'rank'),
            ("average copay for each plan type", 'rank'),
            ("what are the top 5 most common diagnoses", 'rank'),
            ("which facility handles the most volume", 'rank'),
            ("top specialties by claim count", 'rank'),
            ("breakdown by region", 'rank'),
            ("list the top 10 members by total paid amount", 'rank'),
            ("which medications are prescribed most often", 'rank'),
        ])

        examples.extend([
            ("claims per month", 'trend'),
            ("monthly trend of encounters", 'trend'),
            ("how have claims changed over time", 'trend'),
            ("quarterly cost trend", 'trend'),
            ("claims by month in 2024", 'trend'),
            ("show me the trend of paid amounts", 'trend'),
            ("encounters per quarter", 'trend'),
            ("yearly claims volume", 'trend'),
            ("claims over time", 'trend'),
            ("monthly breakdown of prescriptions", 'trend'),
            ("how many claims per month in 2024", 'trend'),
            ("encounters per month", 'trend'),
            ("cost per quarter", 'trend'),
            ("monthly claims count", 'trend'),
            ("trend of claims over time", 'trend'),
            ("visits over time", 'trend'),
            ("what is the trend of encounters over time", 'trend'),
        ])

        examples.extend([
            ("what is the denial rate", 'rate'),
            ("percentage of claims denied", 'rate'),
            ("denial rate by region", 'rate'),
            ("what percent of claims are approved", 'rate'),
            ("no show rate for appointments", 'rate'),
            ("readmission rate", 'rate'),
            ("what is the approval rate", 'rate'),
            ("what percentage of claims are denied", 'rate'),
            ("what is the denial rate by region", 'rate'),
            ("which region has the highest denial rate", 'rate'),
            ("what is the no-show rate for appointments", 'rate'),
        ])

        examples.extend([
            ("compare HMO vs PPO costs", 'compare'),
            ("telehealth vs inpatient visits", 'compare'),
            ("compare drugs vs hospital spending", 'compare'),
            ("NCAL vs SCAL claims", 'compare'),
            ("compare telehealth vs inpatient visit counts", 'compare'),
            ("compare emergency vs outpatient", 'compare'),
            ("HMO versus PPO", 'compare'),
        ])

        examples.extend([
            ("show me the members", 'list'),
            ("list all providers", 'list'),
            ("show denied claims", 'list'),
            ("who are our sickest patients", 'list'),
            ("list members with high risk", 'list'),
        ])

        examples.extend([
            ("give me a summary of claims", 'summary'),
            ("overview of the system", 'summary'),
            ("show me everything about NCAL", 'summary'),
            ("claims summary", 'summary'),
            ("executive summary", 'summary'),
            ("give me a summary of our claims", 'summary'),
        ])

        examples.extend([
            ("how many female members are enrolled", 'count'),
            ("how many emergency visits happened", 'count'),
            ("how many claims are still pending", 'count'),
            ("how many members have a risk score above 2", 'count'),
            ("how many members have diabetes", 'count'),
            ("how many memebers in ncal region", 'count'),
            ("whats the avg paid amnt per claim", 'aggregate'),
            ("top specialites by clam count", 'rank'),
            ("what is the readmission rate for our hospital", 'rate'),
            ("what is the average cost per encounter for HMO members", 'aggregate'),
        ])

        examples.extend([
            ("how many claims were denied", 'count'),
            ("how many appointments were scheduled", 'count'),
            ("how many total claims", 'count'),
            ("count denied claims", 'count'),
            ("count of no-shows", 'count'),
            ("total count of members", 'count'),
            ("tell me how many claims were approved", 'count'),
            ("give me the number of denied claims", 'count'),
            ("how many prescriptions exist", 'count'),
            ("how many referrals were made", 'count'),
        ])

        examples.extend([
            ("how much are we spending on healthcare", 'aggregate'),
            ("total healthcare spending", 'aggregate'),
            ("how many claims per member", 'aggregate'),
            ("per member per month cost", 'aggregate'),
            ("average cost per encounter", 'aggregate'),
            ("what is the total spending", 'aggregate'),
            ("average paid amount per claim", 'aggregate'),
            ("what is the mean claim amount", 'aggregate'),
            ("total billed amount", 'aggregate'),
            ("average number of claims per member", 'aggregate'),
        ])

        examples.extend([
            ("which region costs the most", 'rank'),
            ("total cost by region", 'rank'),
            ("cost by region", 'rank'),
            ("spending by plan type", 'rank'),
            ("which region has the most claims", 'rank'),
            ("which plan type is most expensive", 'rank'),
            ("breakdown of claims by status", 'rank'),
            ("show claims by plan type", 'rank'),
            ("providers with the most patients", 'rank'),
            ("top 10 members by cost", 'rank'),
            ("most expensive regions", 'rank'),
        ])

        examples.extend([
            ("are we getting a lot of no-shows", 'rate'),
            ("what is the no show percentage", 'rate'),
            ("what fraction of claims are denied", 'rate'),
            ("approval percentage", 'rate'),
            ("what is the rate of cancellations", 'rate'),
            ("what percent of appointments are no-shows", 'rate'),
            ("how often are claims denied", 'rate'),
            ("claim denial percentage", 'rate'),
            ("what is the approval rate by region", 'rate'),
            ("which plan type has the highest denial rate", 'rate'),
        ])

        examples.extend([
            ("show me monthly claims", 'trend'),
            ("how have costs changed over time", 'trend'),
            ("claims volume by quarter", 'trend'),
            ("monthly denial rate trend", 'trend'),
            ("cost trend over months", 'trend'),
            ("appointment volume by month", 'trend'),
            ("show the trend of no-shows over time", 'trend'),
            ("how are encounter volumes changing", 'trend'),
            ("claims per quarter", 'trend'),
            ("spending by month", 'trend'),
        ])

        examples.extend([
            ("compare denied vs approved claim amounts", 'compare'),
            ("HMO compared to PPO", 'compare'),
            ("compare inpatient and outpatient costs", 'compare'),
            ("denied versus approved counts", 'compare'),
            ("how does HMO compare to PPO", 'compare'),
            ("contrast emergency with outpatient", 'compare'),
            ("telehealth vs office visit volume", 'compare'),
            ("male vs female claim amounts", 'compare'),
            ("male versus female claims", 'compare'),
            ("compare male and female members", 'compare'),
        ])

        examples.extend([
            ("tell me about claim denials", 'summary'),
            ("show me everything about claims", 'summary'),
            ("overview of our members", 'summary'),
            ("summarize the claims data", 'summary'),
            ("tell me about our providers", 'summary'),
            ("give me a broad view of appointments", 'summary'),
        ])

        return examples


class HealthcareTransformerEngine:

    def __init__(self, db_path: str, schema_graph=None):
        self.db_path = db_path
        t0 = time.time()

        if schema_graph:
            self.graph = schema_graph
        else:
            from schema_graph import SemanticSchemaGraph
            self.graph = SemanticSchemaGraph(db_path)

        self.tokenizer = HealthcareBPETokenizer(vocab_size=2048)

        self.training_gen = TransformerTrainingData(self.graph)
        training_data = self.training_gen.generate()

        corpus = [ex['question'] for ex in training_data]
        for table_name, concept in self.graph.tables.items():
            corpus.append(table_name)
            corpus.append(concept.concept)
            corpus.extend(concept.synonyms)
            for col in self.graph.columns.get(table_name, {}):
                corpus.append(col.lower().replace('_', ' '))
        self.tokenizer.train(corpus)

        self.transformer = HealthcareTransformer(
            vocab_size=self.tokenizer.actual_vocab_size,
            d_model=128,
            n_heads=8,
            n_layers=3,
            d_ff=512,
            max_seq_len=128,
            n_intents=len(INTENT_LABELS),
        )

        param_count = self.transformer.get_param_count()
        logger.info("Healthcare Transformer: %d parameters (%.1fK)",
                     param_count, param_count / 1000)

        self._train(training_data)

        self._sql_generator = None
        try:
            from healthcare_neural_engine import SmartSQLGenerator, InsightNarrator, HealthcareEntityExtractor
            self._sql_generator = SmartSQLGenerator(self.graph, db_path)
            self._narrator = InsightNarrator(db_path)
            self._entity_extractor = HealthcareEntityExtractor(self.graph)
        except Exception as e:
            logger.warning("Could not load SQL generator: %s", e)

        self._domain_intelligence = None
        try:
            from domain_intelligence import create_domain_intelligence
            self._domain_intelligence = create_domain_intelligence()
            logger.info("Domain intelligence active: jargon mastery + concept ontology + benchmarks")
        except Exception as e:
            logger.info("Domain intelligence not available: %s", e)

        elapsed = time.time() - t0
        logger.info("HealthcareTransformerEngine initialized in %.1fs", elapsed)

    def _train(self, training_data: List[Dict]):
        t0 = time.time()

        n_epochs = 50
        base_lr = 0.003
        min_lr = 0.0002
        warmup_epochs = 5
        total_loss = 0.0
        n_steps = 0

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            if epoch < warmup_epochs:
                lr = base_lr * (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / max(n_epochs - warmup_epochs, 1)
                lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * progress))

            indices = np.random.permutation(len(training_data))

            for idx in indices:
                ex = training_data[idx]
                token_ids = self.tokenizer.encode(ex['question'])
                target_intent = ex['intent_idx']

                loss = self.transformer.train_step(token_ids, target_intent, lr=lr)
                epoch_loss += loss
                n_steps += 1

            total_loss = epoch_loss / len(training_data)

        correct = 0
        for ex in training_data:
            token_ids = self.tokenizer.encode(ex['question'])
            output = self.transformer.forward(token_ids)
            pred = int(np.argmax(output['intent_logits']))
            if pred == ex['intent_idx']:
                correct += 1

        acc = correct / len(training_data) * 100
        elapsed = (time.time() - t0) * 1000
        logger.info("Transformer trained: %d examples, %d epochs, acc=%.1f%%, loss=%.4f in %.0fms",
                     len(training_data), n_epochs, acc, total_loss, elapsed)

    def classify_intent(self, question: str) -> Tuple[str, float, Dict]:
        token_ids = self.tokenizer.encode(question)
        output = self.transformer.forward(token_ids)

        logits = output['intent_logits']
        exp_logits = np.exp(logits - logits.max())
        probs = exp_logits / exp_logits.sum()

        intent_idx = int(np.argmax(probs))
        intent = INTENT_LABELS[intent_idx]
        confidence = float(probs[intent_idx])

        attention_info = {
            'top_attended_positions': [],
            'confidence': output['confidence'],
        }

        if output['attention_weights']:
            last_attn = output['attention_weights'][-1]
            if last_attn is not None:
                avg_attn = np.mean(last_attn, axis=0)
                attn_to_tokens = np.mean(avg_attn, axis=0)
                top_positions = np.argsort(attn_to_tokens)[::-1][:5]
                tokens = [self.tokenizer.id_to_token.get(tid, '?') for tid in token_ids]
                attention_info['top_attended_positions'] = [
                    (int(pos), tokens[pos] if pos < len(tokens) else '?',
                     float(attn_to_tokens[pos]))
                    for pos in top_positions
                ]

        return intent, confidence, attention_info

    def process(self, question: str, session_id: str = '') -> Dict[str, Any]:
        t_start = time.time()
        result = {
            'sql': '', 'rows': [], 'columns': [], 'narrative': '',
            'source': 'healthcare_transformer', 'confidence': 0.0,
            'intent': '', 'error': None, 'timing': {},
            'attention_info': {},
        }

        domain_analysis = None
        if self._domain_intelligence:
            try:
                domain_analysis = self._domain_intelligence.analyze_question(question)
                normalized_q = domain_analysis.get('normalized_question', question)
                result['domain_analysis'] = {
                    'normalized': normalized_q,
                    'scope': domain_analysis.get('scope', {}),
                    'related_metrics': [m.get('metric_id', '') for m in domain_analysis.get('related_metrics', [])[:3]],
                    'jargon_resolved': normalized_q != question,
                }
            except Exception:
                pass

        t_nlu = time.time()
        intent, confidence, attn_info = self.classify_intent(question)
        result['intent'] = intent
        result['confidence'] = confidence
        result['attention_info'] = attn_info
        result['timing']['nlu_ms'] = (time.time() - t_nlu) * 1000

        q_lower = question.lower()
        intent = self._apply_intent_overrides(intent, q_lower)
        result['intent'] = intent

        t_ent = time.time()
        entities = {}
        if self._entity_extractor:
            entities = self._entity_extractor.extract(question)

        if domain_analysis:
            domain_where = domain_analysis.get('sql_where_clause', '')
            if domain_where and 'domain_filters' not in entities:
                entities['domain_filters'] = domain_where
            scope = domain_analysis.get('scope', {})
            if scope.get('regions') and not entities.get('region_filter'):
                entities['region_filter'] = scope['regions']
            if scope.get('plan_types') and not entities.get('plan_filter'):
                entities['plan_filter'] = scope['plan_types']
        result['timing']['entity_ms'] = (time.time() - t_ent) * 1000

        t_sql = time.time()
        sql = None
        if self._sql_generator:
            sql = self._sql_generator.generate(intent, entities, question)
        result['timing']['sql_ms'] = (time.time() - t_sql) * 1000

        if not sql:
            result['error'] = 'Failed to generate valid SQL'
            result['timing']['total_ms'] = (time.time() - t_start) * 1000
            return result

        result['sql'] = sql

        t_exec = time.time()
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description] if cur.description else []
            conn.close()
            result['rows'] = rows
            result['columns'] = columns
        except Exception as e:
            result['error'] = str(e)
            result['timing']['total_ms'] = (time.time() - t_start) * 1000
            return result
        result['timing']['exec_ms'] = (time.time() - t_exec) * 1000

        t_narr = time.time()
        if self._narrator:
            result['narrative'] = self._narrator.narrate(
                question, intent, rows, columns, entities
            )
        result['timing']['narrative_ms'] = (time.time() - t_narr) * 1000

        result['timing']['total_ms'] = (time.time() - t_start) * 1000
        return result

    def _apply_intent_overrides(self, intent: str, q_lower: str) -> str:
        import re

        temporal_signals = ['over time', 'per month', 'per quarter', 'per year',
                           'monthly', 'quarterly', 'yearly', 'by month', 'by quarter',
                           'by year', 'trend', 'how have', 'how has', 'changed over',
                           'vary by quarter', 'vary by month']
        has_temporal = any(ts in q_lower for ts in temporal_signals)
        if has_temporal and intent != 'trend':
            return 'trend'

        compare_pattern = re.search(r'\b(vs|versus)\b', q_lower)
        has_compare = compare_pattern or 'compare' in q_lower or 'compared to' in q_lower
        if has_compare and intent not in ('compare', 'rate', 'trend'):
            return 'compare'

        if (re.search(r'\btop\s+\d+\b', q_lower) or 'most common' in q_lower or
            'most frequently' in q_lower or 'highest' in q_lower or
            'lowest' in q_lower) and intent not in ('rank', 'rate'):
            return 'rank'

        rate_words = ['denial rate', 'approval rate', 'readmission rate', 'no-show rate',
                      'no show rate', 'percentage of', 'percent of', 'what percent',
                      'what fraction', 'how often', 'a lot of no-shows',
                      'a lot of no shows', 'a lot of denials']
        if any(w in q_lower for w in rate_words) and intent != 'rate':
            return 'rate'

        group_patterns = [r'\bby (plan type|region|visit type|specialty|department)\b',
                         r'\bper (plan type|region)\b', r'\beach (plan type|region)\b',
                         r'\bbreak\s*down\b']
        if intent == 'aggregate':
            for gp in group_patterns:
                if re.search(gp, q_lower):
                    return 'rank'

        if re.search(r'^number of\b', q_lower) and intent != 'count':
            return 'count'

        if 'pmpm' in q_lower:
            return 'aggregate'

        return intent

    def get_model_info(self) -> Dict:
        return {
            'architecture': 'Transformer (Llama 3 / Phi-3 inspired)',
            'vocab_size': self.tokenizer.actual_vocab_size,
            'd_model': self.transformer.d_model,
            'n_heads': self.transformer.n_heads,
            'n_layers': self.transformer.n_layers,
            'd_ff': 512,
            'total_params': self.transformer.get_param_count(),
            'components': {
                'tokenizer': 'BPE (healthcare vocabulary)',
                'positional_encoding': 'RoPE (Rotary, from Llama 3)',
                'normalization': 'RMSNorm (from Llama 3)',
                'activation': 'SwiGLU (from Llama 3 / PaLM)',
                'attention': 'Multi-Head Self-Attention (8 heads)',
            },
            'training': {
                'method': 'Supervised on schema-derived healthcare queries',
                'data_source': 'Generated from database schema + augmented examples',
                'external_dependencies': 'None — fully air-gapped',
            },
        }


def create_transformer_engine(db_path: str, schema_graph=None) -> HealthcareTransformerEngine:
    return HealthcareTransformerEngine(db_path=db_path, schema_graph=schema_graph)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

    db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'healthcare_production.db')
    print("=" * 80)
    print("  HEALTHCARE TRANSFORMER — Self-Test")
    print("=" * 80)

    engine = create_transformer_engine(db_path)

    info = engine.get_model_info()
    print(f"\n  Architecture: {info['architecture']}")
    print(f"  Parameters: {info['total_params']:,} ({info['total_params']/1000:.1f}K)")
    print(f"  Vocabulary: {info['vocab_size']} tokens")
    print(f"  Model dim: {info['d_model']}, Heads: {info['n_heads']}, Layers: {info['n_layers']}")
    print(f"  Positional: {info['components']['positional_encoding']}")
    print(f"  Normalization: {info['components']['normalization']}")
    print(f"  Activation: {info['components']['activation']}")

    test_questions = [
        "How many total members do we have?",
        "What is the denial rate by region?",
        "Top 5 medications by cost",
        "Claims per month in 2024",
        "Compare telehealth vs inpatient visit counts",
        "What is our PMPM cost?",
        "Who are our sickest patients?",
    ]

    print(f"\n  Testing {len(test_questions)} questions:\n")
    for q in test_questions:
        result = engine.process(q)
        intent = result['intent']
        conf = result['confidence']
        rows = len(result.get('rows', []))
        error = result.get('error', '')
        sql = (result.get('sql', '') or '')[:80]
        timing = result.get('timing', {}).get('total_ms', 0)

        status = 'OK' if rows > 0 and not error else 'FAIL'
        print(f"  [{status}] {q}")
        print(f"       intent={intent} conf={conf:.2f} rows={rows} {timing:.0f}ms")
        if error:
            print(f"       ERROR: {error}")
        print()
