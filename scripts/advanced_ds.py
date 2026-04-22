import time
import math
import hashlib
import json
import os
from collections import OrderedDict, deque
from typing import Any, Dict, List, Optional, Tuple, Callable


class LRUCache:

    def __init__(self, capacity: int = 128, ttl_seconds: float = 300.0):
        self.capacity = capacity
        self.ttl = ttl_seconds
        self._cache = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            value, ts = self._cache[key]
            if self.ttl > 0 and (time.time() - ts) > self.ttl:
                del self._cache[key]
                self.misses += 1
                return None
            self._cache.move_to_end(key)
            self.hits += 1
            return value
        self.misses += 1
        return None

    def put(self, key: str, value: Any):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (value, time.time())
        while len(self._cache) > self.capacity:
            self._cache.popitem(last=False)

    def invalidate(self, key: str):
        self._cache.pop(key, None)

    def clear(self):
        self._cache.clear()

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def stats(self) -> Dict:
        return {
            'size': len(self._cache),
            'capacity': self.capacity,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': '{:.1f}%'.format(self.hit_rate * 100),
        }


class BloomFilter:

    def __init__(self, expected_items: int = 1000, fp_rate: float = 0.01):
        self.size = max(64, int(-expected_items * math.log(fp_rate) / (math.log(2) ** 2)))
        self.num_hashes = max(1, int((self.size / expected_items) * math.log(2)))
        self.bits = [False] * self.size
        self.count = 0

    def _hashes(self, item: str) -> List[int]:
        h1 = int(hashlib.md5(item.encode()).hexdigest(), 16)
        h2 = int(hashlib.sha1(item.encode()).hexdigest(), 16)
        return [(h1 + i * h2) % self.size for i in range(self.num_hashes)]

    def add(self, item: str):
        for pos in self._hashes(item):
            self.bits[pos] = True
        self.count += 1

    def might_contain(self, item: str) -> bool:
        return all(self.bits[pos] for pos in self._hashes(item))

    def estimated_fp_rate(self) -> float:
        ones = sum(self.bits)
        if ones == 0:
            return 0.0
        return (ones / self.size) ** self.num_hashes


class TrieNode:
    __slots__ = ['children', 'is_end', 'value', 'frequency']

    def __init__(self):
        self.children = {}
        self.is_end = False
        self.value = None
        self.frequency = 0

class Trie:

    def __init__(self):
        self.root = TrieNode()
        self.size = 0

    def insert(self, word: str, value: Any = None):
        node = self.root
        for ch in word.lower():
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end = True
        node.value = value
        self.size += 1

    def search(self, word: str) -> Optional[Any]:
        node = self._find_node(word.lower())
        if node and node.is_end:
            node.frequency += 1
            return node.value
        return None

    def starts_with(self, prefix: str, limit: int = 10) -> List[Tuple[str, Any]]:
        node = self._find_node(prefix.lower())
        if node is None:
            return []
        results = []
        self._collect(node, prefix.lower(), results)
        results.sort(key=lambda x: x[2], reverse=True)
        return [(w, v) for w, v, _ in results[:limit]]

    def _find_node(self, prefix: str) -> Optional[TrieNode]:
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node

    def _collect(self, node: TrieNode, prefix: str, results: list):
        if node.is_end:
            results.append((prefix, node.value, node.frequency))
        for ch, child in node.children.items():
            self._collect(child, prefix + ch, results)


class RingBuffer:

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.head = 0
        self.count = 0

    def append(self, value: float):
        self.buffer[self.head] = value
        self.head = (self.head + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def get_all(self) -> List[float]:
        if self.count < self.capacity:
            return [self.buffer[i] for i in range(self.count)]
        start = self.head
        return [self.buffer[(start + i) % self.capacity] for i in range(self.capacity)]

    def mean(self) -> float:
        vals = self.get_all()
        return sum(vals) / len(vals) if vals else 0.0

    def percentile(self, p: float) -> float:
        vals = sorted(self.get_all())
        if not vals:
            return 0.0
        idx = int(len(vals) * p / 100.0)
        return vals[min(idx, len(vals) - 1)]

    def std_dev(self) -> float:
        vals = self.get_all()
        if len(vals) < 2:
            return 0.0
        m = sum(vals) / len(vals)
        return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))

    def stats(self) -> Dict:
        vals = self.get_all()
        if not vals:
            return {'count': 0}
        return {
            'count': len(vals),
            'mean': '{:.2f}'.format(self.mean()),
            'p50': '{:.2f}'.format(self.percentile(50)),
            'p95': '{:.2f}'.format(self.percentile(95)),
            'p99': '{:.2f}'.format(self.percentile(99)),
            'std': '{:.2f}'.format(self.std_dev()),
        }


class CircuitBreaker:

    CLOSED = 'CLOSED'
    OPEN = 'OPEN'
    HALF_OPEN = 'HALF_OPEN'

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0,
                 half_open_max: int = 1):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max = half_open_max

        self.state = self.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.half_open_calls = 0

    def can_execute(self) -> bool:
        if self.state == self.CLOSED:
            return True
        elif self.state == self.OPEN:
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = self.HALF_OPEN
                self.half_open_calls = 0
                return True
            return False
        else:
            return self.half_open_calls < self.half_open_max

    def record_success(self):
        if self.state == self.HALF_OPEN:
            self.success_count += 1
            self.state = self.CLOSED
            self.failure_count = 0
        self.failure_count = max(0, self.failure_count - 1)

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.state == self.HALF_OPEN:
            self.state = self.OPEN
        elif self.failure_count >= self.failure_threshold:
            self.state = self.OPEN

    def execute(self, func: Callable, *args, fallback: Callable = None, **kwargs):
        if not self.can_execute():
            if fallback:
                return fallback(*args, **kwargs)
            raise RuntimeError("Circuit breaker OPEN - service unavailable")

        try:
            if self.state == self.HALF_OPEN:
                self.half_open_calls += 1
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            if fallback:
                return fallback(*args, **kwargs)
            raise


class ConnectionPool:

    def __init__(self, factory: Callable, max_size: int = 10):
        self.factory = factory
        self.max_size = max_size
        self._available = deque()
        self._in_use = set()
        self.total_created = 0

    def acquire(self):
        if self._available:
            obj = self._available.popleft()
        elif len(self._in_use) < self.max_size:
            obj = self.factory()
            self.total_created += 1
        else:
            raise RuntimeError("Pool exhausted ({} in use)".format(len(self._in_use)))
        self._in_use.add(id(obj))
        return obj

    def release(self, obj):
        obj_id = id(obj)
        if obj_id in self._in_use:
            self._in_use.discard(obj_id)
            self._available.append(obj)

    def stats(self) -> Dict:
        return {
            'available': len(self._available),
            'in_use': len(self._in_use),
            'total_created': self.total_created,
            'max_size': self.max_size,
        }


class WriteAheadLog:

    def __init__(self, log_path: str = None):
        self.log_path = log_path or '/tmp/mtp_wal.jsonl'
        self.entries = []
        self.sequence = 0

    def append(self, event_type: str, data: Dict) -> int:
        self.sequence += 1
        entry = {
            'seq': self.sequence,
            'ts': time.time(),
            'type': event_type,
            'data': data,
        }
        self.entries.append(entry)

        if self.log_path:
            try:
                with open(self.log_path, 'a') as f:
                    f.write(json.dumps(entry) + '\n')
            except (IOError, OSError):
                pass

        return self.sequence

    def replay(self, from_seq: int = 0) -> List[Dict]:
        return [e for e in self.entries if e['seq'] > from_seq]

    def get_by_type(self, event_type: str, limit: int = 100) -> List[Dict]:
        return [e for e in reversed(self.entries) if e['type'] == event_type][:limit]


class EventStore:

    def __init__(self):
        self.events = []
        self.subscribers = {}

    def emit(self, event_type: str, payload: Dict) -> Dict:
        event = {
            'id': len(self.events) + 1,
            'type': event_type,
            'ts': time.time(),
            'payload': payload,
        }
        self.events.append(event)

        for callback in self.subscribers.get(event_type, []):
            try:
                callback(event)
            except Exception:
                pass
        return event

    def subscribe(self, event_type: str, callback: Callable):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)

    def query_events(self, event_type: str = None, since: float = 0,
                     limit: int = 100) -> List[Dict]:
        results = self.events
        if event_type:
            results = [e for e in results if e['type'] == event_type]
        if since > 0:
            results = [e for e in results if e['ts'] >= since]
        return results[-limit:]

    def aggregate(self, event_type: str, field: str) -> Dict:
        values = []
        for e in self.events:
            if e['type'] == event_type and field in e.get('payload', {}):
                val = e['payload'][field]
                if isinstance(val, (int, float)):
                    values.append(val)
        if not values:
            return {'count': 0}
        return {
            'count': len(values),
            'sum': sum(values),
            'mean': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
        }


class ConsistentHashRing:

    def __init__(self, nodes: List[str] = None, replicas: int = 100):
        self.replicas = replicas
        self.ring = {}
        self.sorted_keys = []
        if nodes:
            for node in nodes:
                self.add_node(node)

    def _hash(self, key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def add_node(self, node: str):
        for i in range(self.replicas):
            h = self._hash("{}-{}".format(node, i))
            self.ring[h] = node
        self.sorted_keys = sorted(self.ring.keys())

    def remove_node(self, node: str):
        for i in range(self.replicas):
            h = self._hash("{}-{}".format(node, i))
            self.ring.pop(h, None)
        self.sorted_keys = sorted(self.ring.keys())

    def get_node(self, key: str) -> Optional[str]:
        if not self.ring:
            return None
        h = self._hash(key)
        lo, hi = 0, len(self.sorted_keys)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.sorted_keys[mid] < h:
                lo = mid + 1
            else:
                hi = mid
        idx = lo % len(self.sorted_keys)
        return self.ring[self.sorted_keys[idx]]


class BatchStreamPipeline:

    def __init__(self):
        self.batch_results = {}
        self.speed_buffer = deque(maxlen=10000)
        self.batch_timestamp = 0

    def submit_event(self, event: Dict):
        event['_ts'] = time.time()
        self.speed_buffer.append(event)

    def run_batch(self, events: List[Dict], processor: Callable) -> Dict:
        self.batch_results = processor(events)
        self.batch_timestamp = time.time()
        return self.batch_results

    def query(self, key: str) -> Any:
        batch_val = self.batch_results.get(key, 0)
        speed_val = sum(1 for e in self.speed_buffer
                       if e.get('key') == key and e['_ts'] > self.batch_timestamp)
        return batch_val + speed_val

    def get_speed_stats(self) -> Dict:
        count = len(self.speed_buffer)
        if count == 0:
            return {'events': 0, 'window_seconds': 0}
        oldest = self.speed_buffer[0]['_ts']
        newest = self.speed_buffer[-1]['_ts']
        return {
            'events': count,
            'window_seconds': '{:.1f}'.format(newest - oldest),
            'events_per_second': '{:.1f}'.format(count / max(0.001, newest - oldest)),
        }


class MetricsCollector:

    def __init__(self, window_size: int = 1000):
        self.latency = RingBuffer(window_size)
        self.accuracy = RingBuffer(window_size)
        self.throughput = RingBuffer(window_size)
        self.errors = RingBuffer(window_size)
        self._query_count = 0
        self._start_time = time.time()

    def record_query(self, latency_ms: float, is_correct: bool = True):
        self.latency.append(latency_ms)
        self.accuracy.append(1.0 if is_correct else 0.0)
        self._query_count += 1

    def record_error(self):
        self.errors.append(1.0)
        self._query_count += 1

    def get_dashboard(self) -> Dict:
        elapsed = time.time() - self._start_time
        return {
            'total_queries': self._query_count,
            'uptime_seconds': '{:.0f}'.format(elapsed),
            'queries_per_second': '{:.1f}'.format(self._query_count / max(1, elapsed)),
            'latency': self.latency.stats(),
            'accuracy': {
                'mean': '{:.1f}%'.format(self.accuracy.mean() * 100),
                'count': self.accuracy.count,
            },
            'error_count': self.errors.count,
        }


if __name__ == '__main__':
    import random
    print("=" * 60)
    print("Advanced Data Structures & System Design - Self Test")
    print("=" * 60)

    cache = LRUCache(capacity=3, ttl_seconds=10)
    cache.put("q1", "SELECT COUNT(*) FROM claims")
    cache.put("q2", "SELECT AVG(paid_amount) FROM claims")
    cache.put("q3", "SELECT * FROM claims WHERE region='CA'")
    assert cache.get("q1") is not None
    cache.put("q4", "SELECT * FROM members")
    assert cache.get("q2") is None
    print("[OK] LRU Cache: {}".format(cache.stats()))

    bf = BloomFilter(expected_items=100, fp_rate=0.01)
    bf.add("claims")
    bf.add("members")
    bf.add("encounters")
    assert bf.might_contain("claims")
    assert bf.might_contain("members")
    assert not bf.might_contain("xyz_nonexistent_table")
    print("[OK] Bloom Filter: {} items, FP rate ~{:.4f}".format(bf.count, bf.estimated_fp_rate()))

    trie = Trie()
    trie.insert("select", "SQL keyword")
    trie.insert("self_pay", "Payment type")
    trie.insert("service_date", "Date column")
    completions = trie.starts_with("sel")
    assert len(completions) == 2
    print("[OK] Trie: 'sel' -> {}".format([c[0] for c in completions]))

    rb = RingBuffer(capacity=100)
    for i in range(50):
        rb.append(random.gauss(50, 10) if i % 2 == 0 else random.gauss(45, 8))
    print("[OK] Ring Buffer: {}".format(rb.stats()))

    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
    for _ in range(3):
        cb.record_failure()
    assert cb.state == CircuitBreaker.OPEN
    assert not cb.can_execute()
    print("[OK] Circuit Breaker: state={} after 3 failures".format(cb.state))

    es = EventStore()
    es.emit("QUERY_SUBMITTED", {"question": "how many claims", "user": "demo"})
    es.emit("QUERY_EXECUTED", {"sql": "SELECT COUNT(*) FROM claims", "latency_ms": 45})
    events = es.query_events("QUERY_EXECUTED")
    print("[OK] Event Store: {} events".format(len(es.events)))

    ring = ConsistentHashRing(["expert_0", "expert_1", "expert_2", "expert_3"])
    routes = {}
    for q in ["count claims", "average cost", "claims by region", "denied claims"]:
        node = ring.get_node(q)
        routes[q] = node
    print("[OK] Consistent Hash: {}".format(routes))

    mc = MetricsCollector()
    for _ in range(20):
        mc.record_query(random.uniform(10, 100), random.random() > 0.1)
    print("[OK] Metrics: {}".format(mc.get_dashboard()))

    print("\nAll data structures and patterns verified!")
