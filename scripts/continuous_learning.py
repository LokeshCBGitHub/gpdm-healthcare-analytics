import numpy as np
import sqlite3
import json
import time
import math
import hashlib
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import deque

try:
    from gpdm_config import (
        MEMORY_STABILITY_ACCESS_WEIGHT, MEMORY_STABILITY_IMPORTANCE_BASE,
        MEMORY_STRENGTH_CAP, MEMORY_IMPORTANCE_EMA_OLD, MEMORY_IMPORTANCE_EMA_NEW,
        STABILITY_GAIN_EASY, STABILITY_GAIN_OPTIMAL, STABILITY_GAIN_HARD,
        STABILITY_GAIN_FORGOTTEN, FORGET_THRESHOLD, STM_RECALL_THRESHOLD,
        LTM_RECALL_THRESHOLD, WORKING_MEMORY_CAPACITY, WORKING_MEMORY_HALFLIFE,
        STM_CAPACITY, CONSOLIDATION_MIN_ACCESS, CONSOLIDATION_MIN_IMPORTANCE,
    )
except ImportError:
    MEMORY_STABILITY_ACCESS_WEIGHT = 0.25
    MEMORY_STABILITY_IMPORTANCE_BASE = 0.5
    MEMORY_STRENGTH_CAP = 4.0
    MEMORY_IMPORTANCE_EMA_OLD = 0.75
    MEMORY_IMPORTANCE_EMA_NEW = 0.25
    STABILITY_GAIN_EASY = 0.08
    STABILITY_GAIN_OPTIMAL = 0.25
    STABILITY_GAIN_HARD = 0.40
    STABILITY_GAIN_FORGOTTEN = 0.15
    FORGET_THRESHOLD = 0.08
    STM_RECALL_THRESHOLD = 0.35
    LTM_RECALL_THRESHOLD = 0.15
    WORKING_MEMORY_CAPACITY = 7
    WORKING_MEMORY_HALFLIFE = 300.0
    STM_CAPACITY = 80
    CONSOLIDATION_MIN_ACCESS = 2
    CONSOLIDATION_MIN_IMPORTANCE = 0.65

logger = logging.getLogger('gpdm.learning')

@dataclass
class MemoryItem:
    id: str
    content: Dict[str, Any]
    memory_type: str
    strength: float = 1.0
    stability: float = 1.0
    retrievability: float = 1.0
    created_at: float = 0.0
    last_accessed: float = 0.0
    access_count: int = 0
    importance: float = 0.5
    embedding: Optional[List[float]] = None
    tags: List[str] = field(default_factory=list)

    def compute_retrievability(self, now: float = None) -> float:
        if now is None:
            now = time.time()

        elapsed = now - self.last_accessed
        if elapsed <= 0:
            return 1.0

        effective_stability = self.stability * (1 + MEMORY_STABILITY_ACCESS_WEIGHT * self.access_count) * (MEMORY_STABILITY_IMPORTANCE_BASE + self.importance)
        stability_seconds = effective_stability * 86400

        self.retrievability = math.exp(-elapsed / stability_seconds)
        return self.retrievability

    def reinforce(self, feedback: float = 1.0) -> None:
        now = time.time()
        self.access_count += 1
        R_at_recall = self.compute_retrievability(now)

        if R_at_recall > 0.95:
            stability_gain = STABILITY_GAIN_EASY
        elif R_at_recall > 0.7:
            stability_gain = STABILITY_GAIN_OPTIMAL * feedback
        elif R_at_recall > 0.3:
            stability_gain = STABILITY_GAIN_HARD * feedback
        else:
            stability_gain = STABILITY_GAIN_FORGOTTEN * feedback

        self.stability += stability_gain
        self.strength = min(self.strength + 0.2 * feedback, MEMORY_STRENGTH_CAP)
        self.importance = MEMORY_IMPORTANCE_EMA_OLD * self.importance + MEMORY_IMPORTANCE_EMA_NEW * feedback
        self.last_accessed = now
        self.retrievability = 1.0

    def should_forget(self, threshold: float = None) -> bool:
        if threshold is None:
            threshold = FORGET_THRESHOLD
        return self.compute_retrievability() < threshold

class WorkingMemory:

    def __init__(self, capacity: int = None):
        if capacity is None:
            capacity = WORKING_MEMORY_CAPACITY
        self.capacity = capacity
        self.items: deque = deque(maxlen=capacity)
        self.active_entities: Dict[str, Any] = {}

    def push(self, item: Dict[str, Any]) -> None:
        self.items.append({
            **item,
            'timestamp': time.time(),
        })

    def get_context(self) -> List[Dict]:
        return list(reversed(self.items))

    def set_entity(self, name: str, value: Any) -> None:
        self.active_entities[name] = {
            'value': value,
            'set_at': time.time(),
        }

    def get_entity(self, name: str) -> Any:
        if name in self.active_entities:
            return self.active_entities[name]['value']
        return None

    def clear(self) -> None:
        self.items.clear()
        self.active_entities.clear()

    def decay(self, half_life: float = None) -> None:
        if half_life is None:
            half_life = WORKING_MEMORY_HALFLIFE
        now = time.time()
        surviving = deque(maxlen=self.capacity)
        for item in self.items:
            age = now - item['timestamp']
            if age < half_life * 3:
                surviving.append(item)
        self.items = surviving


class ShortTermMemory:

    def __init__(self, capacity: int = None):
        if capacity is None:
            capacity = STM_CAPACITY
        self.capacity = capacity
        self.items: List[MemoryItem] = []

    def store(self, item: MemoryItem) -> None:
        item.created_at = time.time()
        item.last_accessed = time.time()
        self.items.append(item)

        if len(self.items) > self.capacity:
            self._evict_weakest()

    def recall(self, query_embedding: np.ndarray, top_k: int = 5) -> List[MemoryItem]:
        if not self.items or query_embedding is None:
            return []

        scored = []
        for item in self.items:
            if item.embedding is not None:
                emb = np.array(item.embedding)
                sim = float(np.dot(query_embedding, emb) /
                           (np.linalg.norm(query_embedding) * np.linalg.norm(emb) + 1e-10))
                score = sim * item.compute_retrievability()
                scored.append((item, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        for item, score in scored[:top_k]:
            if score > STM_RECALL_THRESHOLD:
                item.reinforce(feedback=0.5)
        return [item for item, _ in scored[:top_k]]

    def get_consolidation_candidates(self) -> List[MemoryItem]:
        candidates = []
        for item in self.items:
            if item.access_count >= CONSOLIDATION_MIN_ACCESS or item.importance > CONSOLIDATION_MIN_IMPORTANCE:
                candidates.append(item)
        return candidates

    def _evict_weakest(self) -> None:
        if not self.items:
            return
        weakest_idx = min(range(len(self.items)),
                         key=lambda i: self.items[i].strength * self.items[i].retrievability)
        self.items.pop(weakest_idx)


class LongTermMemory:

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.memories: Dict[str, MemoryItem] = {}
        self._init_db()
        self._load()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute('''CREATE TABLE IF NOT EXISTS long_term_memory (
            id TEXT PRIMARY KEY,
            content TEXT,
            memory_type TEXT,
            strength REAL,
            stability REAL,
            retrievability REAL,
            created_at REAL,
            last_accessed REAL,
            access_count INTEGER,
            importance REAL,
            embedding TEXT,
            tags TEXT
        )''')
        conn.execute('''CREATE INDEX IF NOT EXISTS idx_ltm_type
                        ON long_term_memory(memory_type)''')
        conn.execute('''CREATE INDEX IF NOT EXISTS idx_ltm_strength
                        ON long_term_memory(strength)''')
        conn.commit()
        conn.close()

    def _load(self) -> None:
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute('''SELECT id, content, memory_type, strength, stability,
                               retrievability, created_at, last_accessed, access_count,
                               importance, embedding, tags FROM long_term_memory''').fetchall()
        conn.close()

        for row in rows:
            item = MemoryItem(
                id=row[0],
                content=json.loads(row[1]),
                memory_type=row[2],
                strength=row[3],
                stability=row[4],
                retrievability=row[5],
                created_at=row[6],
                last_accessed=row[7],
                access_count=row[8],
                importance=row[9],
                embedding=json.loads(row[10]) if row[10] else None,
                tags=json.loads(row[11]) if row[11] else [],
            )
            self.memories[item.id] = item

        logger.info("Long-term memory loaded: %d items", len(self.memories))

    def store(self, item: MemoryItem) -> None:
        self.memories[item.id] = item
        self._persist_item(item)

    def recall(self, query_embedding: np.ndarray = None,
               memory_type: str = None,
               tags: List[str] = None,
               top_k: int = 10,
               min_retrievability: float = 0.1) -> List[MemoryItem]:
        candidates = []

        for item in self.memories.values():
            if memory_type and item.memory_type != memory_type:
                continue
            if tags and not any(t in item.tags for t in tags):
                continue

            R = item.compute_retrievability()
            if R < min_retrievability:
                continue

            score = R * item.strength
            if query_embedding is not None and item.embedding is not None:
                emb = np.array(item.embedding)
                q = query_embedding
                sim = float(np.dot(q, emb) / (np.linalg.norm(q) * np.linalg.norm(emb) + 1e-10))
                score *= (1 + sim)

            candidates.append((item, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        result = []
        for item, score in candidates[:top_k]:
            item.reinforce(feedback=0.3)
            self._persist_item(item)
            result.append(item)

        return result

    def consolidate(self, episodes: List[MemoryItem]) -> None:
        if not episodes:
            return

        by_intent: Dict[str, List[MemoryItem]] = {}
        for ep in episodes:
            intent = ep.content.get('intent', 'unknown')
            by_intent.setdefault(intent, []).append(ep)

        for intent, group in by_intent.items():
            if len(group) < 2:
                continue

            questions = [ep.content.get('question', '') for ep in group]
            sqls = [ep.content.get('sql', '') for ep in group]

            semantic_item = MemoryItem(
                id=f"semantic:{intent}:{hashlib.md5(intent.encode()).hexdigest()[:8]}",
                content={
                    'intent': intent,
                    'pattern': f"Questions about '{intent}' typically involve: {', '.join(set(questions[:3]))}",
                    'example_sqls': list(set(sqls[:3])),
                    'frequency': len(group),
                },
                memory_type='semantic',
                strength=sum(ep.strength for ep in group) / len(group),
                stability=max(ep.stability for ep in group),
                importance=max(ep.importance for ep in group),
            )

            valid_embeddings = [np.array(ep.embedding) for ep in group
                               if ep.embedding is not None]
            if valid_embeddings:
                avg_emb = np.mean(valid_embeddings, axis=0)
                semantic_item.embedding = avg_emb.tolist()

            semantic_item.created_at = time.time()
            semantic_item.last_accessed = time.time()
            semantic_item.tags = [intent, 'consolidated']

            self.store(semantic_item)

        logger.info("Consolidated %d episodes into semantic memories", len(episodes))

    def apply_forgetting(self) -> None:
        now = time.time()
        to_forget = []

        for item_id, item in self.memories.items():
            R = item.compute_retrievability(now)
            if R < FORGET_THRESHOLD and item.importance < 0.5:
                to_forget.append(item_id)

        for item_id in to_forget:
            del self.memories[item_id]

        if to_forget:
            conn = sqlite3.connect(self.db_path)
            for item_id in to_forget:
                conn.execute('DELETE FROM long_term_memory WHERE id = ?', (item_id,))
            conn.commit()
            conn.close()
            logger.info("Forgetting: removed %d weak memories, %d remain",
                        len(to_forget), len(self.memories))

    def _persist_item(self, item: MemoryItem) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute('''INSERT OR REPLACE INTO long_term_memory
                        (id, content, memory_type, strength, stability, retrievability,
                         created_at, last_accessed, access_count, importance, embedding, tags)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
            item.id,
            json.dumps(item.content),
            item.memory_type,
            item.strength,
            item.stability,
            item.retrievability,
            item.created_at,
            item.last_accessed,
            item.access_count,
            item.importance,
            json.dumps(item.embedding) if item.embedding is not None else None,
            json.dumps(item.tags),
        ))
        conn.commit()
        conn.close()

    def get_statistics(self) -> Dict[str, Any]:
        by_type = {}
        total_strength = 0
        for item in self.memories.values():
            by_type[item.memory_type] = by_type.get(item.memory_type, 0) + 1
            total_strength += item.strength

        return {
            'total_memories': len(self.memories),
            'by_type': by_type,
            'avg_strength': total_strength / max(len(self.memories), 1),
        }

class ExperienceReplayBuffer:

    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        self.buffer: List[Dict] = []
        self.success_buffer: List[Dict] = []
        self.failure_buffer: List[Dict] = []

    def store(self, experience: Dict[str, Any]) -> None:
        exp = {
            **experience,
            'timestamp': time.time(),
        }

        self.buffer.append(exp)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

        feedback = experience.get('feedback', 0.5)
        if feedback >= 0.7:
            self.success_buffer.append(exp)
            if len(self.success_buffer) > self.capacity // 2:
                self.success_buffer.pop(0)
        elif feedback < 0.3:
            self.failure_buffer.append(exp)
            if len(self.failure_buffer) > self.capacity // 4:
                self.failure_buffer.pop(0)

    def sample_batch(self, batch_size: int = 16,
                     success_ratio: float = 0.7) -> List[Dict]:
        batch = []
        n_success = int(batch_size * success_ratio)
        n_failure = batch_size - n_success

        if self.success_buffer:
            indices = np.random.choice(
                len(self.success_buffer),
                size=min(n_success, len(self.success_buffer)),
                replace=False
            )
            batch.extend([self.success_buffer[i] for i in indices])

        if self.failure_buffer:
            indices = np.random.choice(
                len(self.failure_buffer),
                size=min(n_failure, len(self.failure_buffer)),
                replace=False
            )
            batch.extend([self.failure_buffer[i] for i in indices])

        return batch

    def get_statistics(self) -> Dict[str, Any]:
        return {
            'total': len(self.buffer),
            'successes': len(self.success_buffer),
            'failures': len(self.failure_buffer),
        }

class OnlineLearner:

    def __init__(self, neural_understanding, learning_rate: float = 0.01):
        self.neural = neural_understanding
        self.lr = learning_rate
        self.replay_buffer = ExperienceReplayBuffer()

        self.fisher_diagonal: Dict[str, np.ndarray] = {}
        self.ewc_lambda = 0.5

        self.total_updates = 0
        self.successful_queries = 0
        self.failed_queries = 0

    def learn_from_feedback(self, question: str, sql: str,
                            feedback: float, metadata: Dict = None) -> None:
        understanding = self.neural.understand(question)
        question_emb = understanding['question_embedding']

        self.replay_buffer.store({
            'question': question,
            'sql': sql,
            'feedback': feedback,
            'embedding': question_emb.tolist(),
            'intent': (metadata or {}).get('intent', 'unknown'),
            'tables': (metadata or {}).get('tables', []),
        })

        if feedback >= 0.5:
            self.neural.hopfield.store(question_emb, {
                'question': question,
                'sql': sql,
                'feedback': feedback,
                **(metadata or {}),
            })
            self.successful_queries += 1
        else:
            self.failed_queries += 1

        self.neural.embeddings.update_incremental([question])
        self.total_updates += 1

        if self.total_updates % 10 == 0:
            self._replay_learn()

    def _replay_learn(self) -> None:
        batch = self.replay_buffer.sample_batch(batch_size=8)
        if not batch:
            return

        for exp in batch:
            emb = np.array(exp.get('embedding', []))
            if emb.size > 0 and exp.get('feedback', 0) >= 0.7:
                self.neural.hopfield.store(emb, {
                    'question': exp['question'],
                    'sql': exp['sql'],
                    'replay': True,
                })

    def compute_fisher_information(self, experiences: List[Dict]) -> None:
        for param_name in ['W_q', 'W_k', 'W_v', 'W_o']:
            for layer in self.neural.transformer.layers:
                param = getattr(layer, param_name)
                fisher = np.zeros_like(param)
                for exp in experiences:
                    emb = np.array(exp.get('embedding', []))
                    if emb.size > 0 and exp.get('feedback', 0) >= 0.7:
                        fisher += np.outer(emb[:param.shape[0]], emb[:param.shape[1]]) ** 2
                fisher /= max(len(experiences), 1)
                key = f'{id(layer)}.{param_name}'
                self.fisher_diagonal[key] = fisher

    def get_learning_statistics(self) -> Dict[str, Any]:
        return {
            'total_updates': self.total_updates,
            'successful_queries': self.successful_queries,
            'failed_queries': self.failed_queries,
            'success_rate': self.successful_queries / max(self.total_updates, 1),
            'replay_buffer': self.replay_buffer.get_statistics(),
        }

class ContinuousLearningEngine:

    def __init__(self, neural_understanding, db_path: str):
        self.neural = neural_understanding
        self.db_path = db_path

        self.working_memory = WorkingMemory()
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory(db_path)
        self.online_learner = OnlineLearner(neural_understanding)

        self._last_consolidation = time.time()
        self._consolidation_interval = 3600

        self._session_start = time.time()
        self._interactions_this_session = 0

    def on_interaction(self, question: str, sql: str,
                       feedback: float = 1.0,
                       result_count: int = 0,
                       metadata: Dict = None) -> None:
        metadata = metadata or {}
        now = time.time()

        self.working_memory.push({
            'question': question,
            'sql': sql,
            'feedback': feedback,
            'result_count': result_count,
            'intent': metadata.get('intent', ''),
        })

        understanding = self.neural.understand(question)
        embedding = understanding['question_embedding']

        memory_id = f"ep:{hashlib.md5(f'{question}:{now}'.encode()).hexdigest()[:12]}"
        stm_item = MemoryItem(
            id=memory_id,
            content={
                'question': question,
                'sql': sql,
                'feedback': feedback,
                'result_count': result_count,
                **metadata,
            },
            memory_type='episodic',
            strength=feedback,
            importance=feedback,
            embedding=embedding.tolist(),
            tags=[metadata.get('intent', 'unknown')],
        )
        self.short_term.store(stm_item)

        self.online_learner.learn_from_feedback(question, sql, feedback, metadata)

        if feedback >= 0.8 and result_count > 0:
            proc_id = f"proc:{hashlib.md5(sql.encode()).hexdigest()[:12]}"
            proc_item = MemoryItem(
                id=proc_id,
                content={
                    'question_pattern': question,
                    'sql_template': sql,
                    'intent': metadata.get('intent', ''),
                    'tables': metadata.get('tables', []),
                },
                memory_type='procedural',
                strength=feedback,
                stability=1.0,
                importance=feedback,
                embedding=embedding.tolist(),
                tags=['sql_pattern', metadata.get('intent', 'unknown')],
            )
            self.long_term.store(proc_item)

        self._interactions_this_session += 1

        if now - self._last_consolidation > self._consolidation_interval:
            self.consolidate()

    def recall_context(self, question: str) -> Dict[str, Any]:
        understanding = self.neural.understand(question)
        embedding = understanding['question_embedding']

        wm_context = self.working_memory.get_context()
        stm_items = self.short_term.recall(embedding, top_k=3)
        ltm_procedural = self.long_term.recall(
            embedding, memory_type='procedural', top_k=3
        )
        ltm_semantic = self.long_term.recall(
            embedding, memory_type='semantic', top_k=3
        )
        hopfield_recalls = understanding.get('recalled_patterns', [])

        return {
            'working_memory': wm_context,
            'short_term': [{'content': item.content, 'strength': item.strength}
                          for item in stm_items],
            'long_term_procedural': [{'content': item.content, 'strength': item.strength}
                                    for item in ltm_procedural],
            'long_term_semantic': [{'content': item.content, 'strength': item.strength}
                                  for item in ltm_semantic],
            'hopfield_recalls': hopfield_recalls,
            'neural_understanding': understanding,
        }

    def consolidate(self) -> None:
        candidates = self.short_term.get_consolidation_candidates()

        if candidates:
            self.long_term.consolidate(candidates)

        self._last_consolidation = time.time()
        logger.info("Consolidation complete: %d candidates processed", len(candidates))

    def forget(self) -> None:
        self.working_memory.decay()
        self.long_term.apply_forgetting()
        self.neural.hopfield.apply_forgetting_curve()

    def new_session(self) -> None:
        self.consolidate()
        self.working_memory.clear()
        self._session_start = time.time()
        self._interactions_this_session = 0

    def get_statistics(self) -> Dict[str, Any]:
        return {
            'session': {
                'duration': time.time() - self._session_start,
                'interactions': self._interactions_this_session,
            },
            'working_memory': {
                'items': len(self.working_memory.items),
                'entities': len(self.working_memory.active_entities),
            },
            'short_term': {
                'items': len(self.short_term.items),
            },
            'long_term': self.long_term.get_statistics(),
            'learning': self.online_learner.get_learning_statistics(),
            'hopfield': {
                'patterns': self.neural.hopfield.num_patterns,
            },
        }

    def save_state(self) -> None:
        self.neural.save_state(self.db_path)

    def load_state(self) -> None:
        self.neural.load_state(self.db_path)
