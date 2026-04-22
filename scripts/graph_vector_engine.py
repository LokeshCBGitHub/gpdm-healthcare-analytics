import math
import random
import hashlib
import json
import os
import time
from collections import defaultdict, Counter, deque
from typing import (
    Dict, List, Tuple, Optional, Set, Any, Callable, Union
)


class KnowledgeNode:
    __slots__ = ('id', 'type', 'properties', 'edges_out', 'edges_in', 'importance')

    def __init__(self, node_id: str, node_type: str, properties: Dict = None):
        self.id = node_id
        self.type = node_type
        self.properties = properties or {}
        self.edges_out: List['KnowledgeEdge'] = []
        self.edges_in: List['KnowledgeEdge'] = []
        self.importance = 0.0


class KnowledgeEdge:
    __slots__ = ('source', 'target', 'relation', 'weight', 'properties')

    def __init__(self, source: str, target: str, relation: str,
                 weight: float = 1.0, properties: Dict = None):
        self.source = source
        self.target = target
        self.relation = relation
        self.weight = weight
        self.properties = properties or {}


class KnowledgeGraph:

    def __init__(self):
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: List[KnowledgeEdge] = []
        self.adjacency: Dict[str, List[KnowledgeEdge]] = defaultdict(list)
        self.reverse_adj: Dict[str, List[KnowledgeEdge]] = defaultdict(list)
        self._pagerank_computed = False


    def add_node(self, node_id: str, node_type: str, **properties) -> KnowledgeNode:
        if node_id in self.nodes:
            self.nodes[node_id].properties.update(properties)
            return self.nodes[node_id]
        node = KnowledgeNode(node_id, node_type, properties)
        self.nodes[node_id] = node
        return node

    def add_edge(self, source: str, target: str, relation: str,
                 weight: float = 1.0, **properties) -> KnowledgeEdge:
        edge = KnowledgeEdge(source, target, relation, weight, properties)
        self.edges.append(edge)
        self.adjacency[source].append(edge)
        self.reverse_adj[target].append(edge)
        if source in self.nodes:
            self.nodes[source].edges_out.append(edge)
        if target in self.nodes:
            self.nodes[target].edges_in.append(edge)
        self._pagerank_computed = False
        return edge


    def bfs(self, start: str, max_depth: int = 5,
            edge_filter: Callable = None) -> List[Tuple[str, int, List[str]]]:
        visited = {start}
        queue = deque([(start, 0, [start])])
        results = []

        while queue:
            current, depth, path = queue.popleft()
            results.append((current, depth, path))

            if depth >= max_depth:
                continue

            for edge in self.adjacency.get(current, []):
                if edge.target not in visited:
                    if edge_filter and not edge_filter(edge):
                        continue
                    visited.add(edge.target)
                    queue.append((edge.target, depth + 1, path + [edge.target]))

        return results

    def dfs(self, start: str, max_depth: int = 5,
            edge_filter: Callable = None) -> List[Tuple[str, int, List[str]]]:
        visited = set()
        results = []

        def _dfs(node, depth, path):
            if node in visited or depth > max_depth:
                return
            visited.add(node)
            results.append((node, depth, path))
            for edge in self.adjacency.get(node, []):
                if edge_filter and not edge_filter(edge):
                    continue
                _dfs(edge.target, depth + 1, path + [edge.target])

        _dfs(start, 0, [start])
        return results

    def dijkstra(self, start: str, end: str,
                 edge_filter: Callable = None) -> Tuple[float, List[str]]:
        INF = float('inf')
        dist = defaultdict(lambda: INF)
        dist[start] = 0.0
        prev = {}
        visited = set()

        unvisited = {start}

        while unvisited:
            current = min(unvisited, key=lambda n: dist[n])
            if current == end:
                break
            unvisited.discard(current)
            visited.add(current)

            for edge in self.adjacency.get(current, []):
                if edge.target in visited:
                    continue
                if edge_filter and not edge_filter(edge):
                    continue
                cost = 1.0 / max(edge.weight, 0.001)
                new_dist = dist[current] + cost
                if new_dist < dist[edge.target]:
                    dist[edge.target] = new_dist
                    prev[edge.target] = current
                    unvisited.add(edge.target)

        if end not in prev and start != end:
            return INF, []
        path = []
        node = end
        while node != start:
            path.append(node)
            node = prev.get(node)
            if node is None:
                return INF, []
        path.append(start)
        return dist[end], list(reversed(path))


    def compute_pagerank(self, damping: float = 0.85, iterations: int = 50,
                         tolerance: float = 1e-6) -> None:
        n = len(self.nodes)
        if n == 0:
            return

        node_ids = list(self.nodes.keys())
        rank = {nid: 1.0 / n for nid in node_ids}

        for iteration in range(iterations):
            new_rank = {}
            max_delta = 0.0

            for nid in node_ids:
                incoming_sum = 0.0
                for edge in self.reverse_adj.get(nid, []):
                    src = edge.source
                    out_degree = len(self.adjacency.get(src, []))
                    if out_degree > 0:
                        incoming_sum += rank[src] / out_degree * edge.weight

                new_rank[nid] = (1 - damping) / n + damping * incoming_sum
                max_delta = max(max_delta, abs(new_rank[nid] - rank[nid]))

            rank = new_rank
            if max_delta < tolerance:
                break

        total = sum(rank.values())
        for nid in node_ids:
            self.nodes[nid].importance = rank[nid] / total if total > 0 else 0

        self._pagerank_computed = True


    def extract_relevant_subgraph(self, query_terms: List[str],
                                   max_hops: int = 3) -> 'KnowledgeGraph':
        seed_nodes = set()
        for term in query_terms:
            term_lower = term.lower()
            for nid, node in self.nodes.items():
                if term_lower in nid.lower():
                    seed_nodes.add(nid)
                elif term_lower in node.properties.get('description', '').lower():
                    seed_nodes.add(nid)
                elif any(term_lower in str(v).lower() for v in node.properties.values()):
                    seed_nodes.add(nid)

        relevant_nodes = set()
        for seed in seed_nodes:
            for node_id, depth, path in self.bfs(seed, max_depth=max_hops):
                relevant_nodes.add(node_id)

        subgraph = KnowledgeGraph()
        for nid in relevant_nodes:
            node = self.nodes[nid]
            subgraph.add_node(nid, node.type, **node.properties)
        for edge in self.edges:
            if edge.source in relevant_nodes and edge.target in relevant_nodes:
                subgraph.add_edge(edge.source, edge.target, edge.relation,
                                  edge.weight, **edge.properties)

        return subgraph


    def reason(self, start_concept: str, target_type: str,
               max_hops: int = 4) -> List[Dict]:
        results = []
        for node_id, depth, path in self.bfs(start_concept, max_depth=max_hops):
            node = self.nodes.get(node_id)
            if node and node.type == target_type:
                chain = []
                for i in range(len(path) - 1):
                    src, tgt = path[i], path[i + 1]
                    for edge in self.adjacency.get(src, []):
                        if edge.target == tgt:
                            chain.append({
                                'from': src,
                                'to': tgt,
                                'relation': edge.relation,
                                'weight': edge.weight,
                            })
                            break
                results.append({
                    'target': node_id,
                    'depth': depth,
                    'path': path,
                    'chain': chain,
                    'importance': node.importance,
                })

        results.sort(key=lambda r: (-r['importance'], r['depth']))
        return results


    @classmethod
    def from_catalog(cls, catalog_dir: str) -> 'KnowledgeGraph':
        graph = cls()

        tables_dir = os.path.join(catalog_dir, 'tables')
        rels_path = os.path.join(catalog_dir, 'relationships', 'relationship_map.json')

        DOMAIN_CONCEPTS = {
            'diabetes': ['E11', 'E10', 'E13', 'diabetes', 'diabetic', 'A1C', 'glucose'],
            'hypertension': ['I10', 'I11', 'hypertension', 'blood pressure', 'HTN'],
            'heart_disease': ['I20', 'I21', 'I25', 'cardiac', 'heart', 'coronary', 'CHF'],
            'emergency': ['emergency', 'ER', 'ED', 'urgent', 'trauma'],
            'chronic': ['chronic', 'long-term', 'ongoing', 'persistent'],
            'denied': ['denied', 'denial', 'rejected', 'not approved'],
            'high_cost': ['expensive', 'high cost', 'costly', 'outlier'],
            'readmission': ['readmit', 'readmission', 're-admission', 'return visit'],
            'preventive': ['preventive', 'screening', 'wellness', 'annual', 'routine'],
            'mental_health': ['psychiatric', 'mental', 'behavioral', 'depression', 'anxiety'],
        }

        for concept, terms in DOMAIN_CONCEPTS.items():
            graph.add_node(f'concept:{concept}', 'concept',
                           description=concept.replace('_', ' '),
                           terms=terms)

        if os.path.exists(tables_dir):
            for fname in sorted(os.listdir(tables_dir)):
                if not fname.endswith('.json'):
                    continue
                try:
                    with open(os.path.join(tables_dir, fname)) as f:
                        profile = json.load(f)
                except Exception:
                    continue

                table_name = profile.get('table_name', fname.replace('.json', ''))
                row_count = profile.get('total_rows', 0)
                col_count = profile.get('total_columns', 0)
                purpose = profile.get('table_purpose', '')

                graph.add_node(f'table:{table_name}', 'table',
                               row_count=row_count, col_count=col_count,
                               purpose=purpose, description=purpose)

                columns = profile.get('columns', {})
                if isinstance(columns, dict):
                    columns = list(columns.values())

                for col_info in columns:
                    col_name = col_info.get('column_name', col_info.get('name', ''))
                    if not col_name:
                        continue

                    data_type = col_info.get('data_type', 'text')
                    semantic_type = col_info.get('semantic_type', '')
                    healthcare_type = col_info.get('healthcare_type', '')
                    cardinality = col_info.get('cardinality', 0)
                    null_pct = col_info.get('null_percentage', 0)
                    top_vals = col_info.get('top_values', [])

                    col_id = f'col:{table_name}.{col_name}'
                    graph.add_node(col_id, 'column',
                                   table=table_name, column=col_name,
                                   data_type=data_type,
                                   semantic_type=semantic_type,
                                   healthcare_type=healthcare_type,
                                   cardinality=cardinality,
                                   null_pct=null_pct,
                                   description=f'{col_name} in {table_name}')

                    graph.add_edge(f'table:{table_name}', col_id, 'has_column',
                                   weight=1.0)

                    type_id = f'type:{data_type}'
                    graph.add_node(type_id, 'data_type', description=data_type)
                    graph.add_edge(col_id, type_id, 'is_type', weight=0.5)

                    if healthcare_type:
                        ht_id = f'hc_type:{healthcare_type}'
                        graph.add_node(ht_id, 'healthcare_type',
                                       description=healthcare_type)
                        graph.add_edge(col_id, ht_id, 'has_healthcare_type', weight=0.8)

                    col_lower = col_name.lower()
                    for concept, terms in DOMAIN_CONCEPTS.items():
                        for term in terms:
                            if term.lower() in col_lower:
                                graph.add_edge(col_id, f'concept:{concept}',
                                               'relates_to_concept', weight=0.7)
                                break

                    if top_vals and cardinality and cardinality < 50:
                        for val_entry in top_vals[:10]:
                            if isinstance(val_entry, dict):
                                val = val_entry.get('value', '')
                            else:
                                val = str(val_entry)
                            if val:
                                val_id = f'val:{table_name}.{col_name}={val}'
                                graph.add_node(val_id, 'value',
                                               table=table_name,
                                               column=col_name,
                                               value=val)
                                graph.add_edge(col_id, val_id, 'has_value', weight=0.3)

                                val_lower = val.lower()
                                for concept, terms in DOMAIN_CONCEPTS.items():
                                    for term in terms:
                                        if term.lower() in val_lower:
                                            graph.add_edge(val_id, f'concept:{concept}',
                                                           'instance_of', weight=0.6)
                                            break

        if os.path.exists(rels_path):
            try:
                with open(rels_path) as f:
                    rel_data = json.load(f)
                for rel in rel_data.get('relationships', []):
                    src_table = rel.get('source_table', '')
                    tgt_table = rel.get('target_table', '')
                    join_col = rel.get('join_column', '')
                    rel_type = rel.get('relationship_type', 'many_to_one')
                    is_fk = rel.get('is_fk', True)

                    weight = 1.0
                    if rel_type == 'one_to_many' or rel_type == 'many_to_one':
                        weight = 0.9
                    elif rel_type == 'many_to_many':
                        weight = 0.6
                    if is_fk:
                        weight += 0.2

                    graph.add_edge(f'table:{src_table}', f'table:{tgt_table}',
                                   'joins_to', weight=weight,
                                   join_column=join_col,
                                   relationship_type=rel_type)
                    graph.add_edge(f'table:{tgt_table}', f'table:{src_table}',
                                   'joins_to', weight=weight,
                                   join_column=join_col,
                                   relationship_type=rel_type)
            except Exception:
                pass

        METRICS = {
            'denial_rate': {'tables': ['claims'], 'columns': ['CLAIM_STATUS'],
                            'formula': 'DENIED / total', 'description': 'Claim denial rate'},
            'avg_cost': {'tables': ['claims'], 'columns': ['BILLED_AMOUNT'],
                         'formula': 'AVG(BILLED_AMOUNT)', 'description': 'Average claim cost'},
            'readmission_rate': {'tables': ['encounters'], 'columns': ['VISIT_TYPE'],
                                  'formula': 'readmits / total', 'description': 'Readmission rate'},
            'avg_los': {'tables': ['encounters'], 'columns': ['LENGTH_OF_STAY'],
                        'formula': 'AVG(LENGTH_OF_STAY)', 'description': 'Average length of stay'},
            'utilization': {'tables': ['encounters', 'members'],
                            'columns': ['ENCOUNTER_ID', 'MEMBER_ID'],
                            'formula': 'encounters / members', 'description': 'Service utilization'},
        }
        for metric_name, info in METRICS.items():
            m_id = f'metric:{metric_name}'
            graph.add_node(m_id, 'metric', **info)
            for t in info['tables']:
                graph.add_edge(m_id, f'table:{t}', 'computed_from', weight=0.8)
            for c in info['columns']:
                for t in info['tables']:
                    col_id = f'col:{t}.{c}'
                    if col_id in graph.nodes:
                        graph.add_edge(m_id, col_id, 'uses_column', weight=0.9)

        graph.compute_pagerank()

        return graph

    def get_stats(self) -> Dict:
        type_counts = Counter(n.type for n in self.nodes.values())
        rel_counts = Counter(e.relation for e in self.edges)
        return {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'node_types': dict(type_counts),
            'edge_types': dict(rel_counts),
            'top_important': sorted(
                [(nid, n.importance) for nid, n in self.nodes.items()],
                key=lambda x: -x[1]
            )[:10],
        }


class SparseVector:
    __slots__ = ('indices', 'values', 'dim')

    def __init__(self, indices: List[int], values: List[float], dim: int):
        self.indices = indices
        self.values = values
        self.dim = dim

    def dot(self, other: 'SparseVector') -> float:
        i, j = 0, 0
        result = 0.0
        while i < len(self.indices) and j < len(other.indices):
            if self.indices[i] == other.indices[j]:
                result += self.values[i] * other.values[j]
                i += 1
                j += 1
            elif self.indices[i] < other.indices[j]:
                i += 1
            else:
                j += 1
        return result

    def norm(self) -> float:
        return math.sqrt(sum(v * v for v in self.values))

    def cosine_similarity(self, other: 'SparseVector') -> float:
        n1, n2 = self.norm(), other.norm()
        if n1 == 0 or n2 == 0:
            return 0.0
        return self.dot(other) / (n1 * n2)


class VectorIndex:

    def __init__(self, num_planes: int = 8, num_tables: int = 4, dim: int = 500):
        self.num_planes = num_planes
        self.num_tables = num_tables
        self.dim = dim

        random.seed(42)
        self.hyperplanes = []
        for _ in range(num_tables):
            planes = []
            for _ in range(num_planes):
                plane = [random.gauss(0, 1) for _ in range(dim)]
                norm = math.sqrt(sum(x * x for x in plane))
                plane = [x / norm for x in plane]
                planes.append(plane)
            self.hyperplanes.append(planes)

        self.tables: List[Dict[int, List[Tuple[str, SparseVector]]]] = [
            defaultdict(list) for _ in range(num_tables)
        ]

        self.vectors: Dict[str, SparseVector] = {}

        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_count = 0

    def _hash_vector(self, vec: SparseVector, table_idx: int) -> int:
        hash_val = 0
        planes = self.hyperplanes[table_idx]
        for i, plane in enumerate(planes):
            dot = 0.0
            for idx, val in zip(vec.indices, vec.values):
                if idx < len(plane):
                    dot += val * plane[idx]
            if dot >= 0:
                hash_val |= (1 << i)
        return hash_val

    def build_vocabulary(self, documents: List[Tuple[str, str]]) -> None:
        doc_freq = Counter()
        all_tokens = set()

        for doc_id, text in documents:
            tokens = self._tokenize(text)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_freq[token] += 1
            all_tokens.update(unique_tokens)

        self.doc_count = len(documents)

        sorted_tokens = sorted(all_tokens, key=lambda t: -doc_freq[t])
        for i, token in enumerate(sorted_tokens[:self.dim]):
            self.vocabulary[token] = i

        for token, idx in self.vocabulary.items():
            df = doc_freq.get(token, 0)
            self.idf[token] = math.log((self.doc_count + 1) / (df + 1)) + 1

    def text_to_vector(self, text: str) -> SparseVector:
        tokens = self._tokenize(text)
        tf = Counter(tokens)

        indices = []
        values = []
        for token, count in tf.items():
            if token in self.vocabulary:
                idx = self.vocabulary[token]
                tf_val = 1 + math.log(count) if count > 0 else 0
                idf_val = self.idf.get(token, 1.0)
                indices.append(idx)
                values.append(tf_val * idf_val)

        if indices:
            pairs = sorted(zip(indices, values))
            indices, values = zip(*pairs)
            indices, values = list(indices), list(values)

        return SparseVector(indices, values, self.dim)

    def index_document(self, doc_id: str, text: str) -> None:
        vec = self.text_to_vector(text)
        self.vectors[doc_id] = vec

        for t in range(self.num_tables):
            hash_key = self._hash_vector(vec, t)
            self.tables[t][hash_key].append((doc_id, vec))

    def search(self, query_text: str, k: int = 5,
               num_probes: int = 2) -> List[Tuple[str, float]]:
        query_vec = self.text_to_vector(query_text)

        candidates = set()
        for t in range(self.num_tables):
            hash_key = self._hash_vector(query_vec, t)

            for doc_id, _ in self.tables[t].get(hash_key, []):
                candidates.add(doc_id)

            if num_probes > 1:
                for bit in range(self.num_planes):
                    probed_key = hash_key ^ (1 << bit)
                    for doc_id, _ in self.tables[t].get(probed_key, []):
                        candidates.add(doc_id)

        scored = []
        for doc_id in candidates:
            vec = self.vectors.get(doc_id)
            if vec:
                sim = query_vec.cosine_similarity(vec)
                scored.append((doc_id, sim))

        scored.sort(key=lambda x: -x[1])
        return scored[:k]

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        words = []
        current = []
        for ch in text:
            if ch.isalnum() or ch == '_':
                current.append(ch)
            else:
                if current:
                    words.append(''.join(current))
                current = []
        if current:
            words.append(''.join(current))

        trigrams = []
        for word in words:
            if len(word) >= 3:
                for i in range(len(word) - 2):
                    trigrams.append(f'__{word[i:i+3]}')

        return words + trigrams

    def get_stats(self) -> Dict:
        bucket_sizes = []
        for t in range(self.num_tables):
            for bucket in self.tables[t].values():
                bucket_sizes.append(len(bucket))
        return {
            'total_documents': len(self.vectors),
            'vocabulary_size': len(self.vocabulary),
            'num_tables': self.num_tables,
            'num_planes': self.num_planes,
            'total_buckets': sum(len(t) for t in self.tables),
            'avg_bucket_size': sum(bucket_sizes) / max(len(bucket_sizes), 1),
            'max_bucket_size': max(bucket_sizes) if bucket_sizes else 0,
        }


class SimHash:

    def __init__(self, num_bits: int = 64):
        self.num_bits = num_bits
        random.seed(12345)
        self.projections = []
        for _ in range(num_bits):
            self.projections.append(random.getrandbits(128))

    def hash(self, text: str) -> int:
        tokens = self._tokenize(text)
        if not tokens:
            return 0

        weights = [0.0] * self.num_bits

        for token in tokens:
            token_hash = int(hashlib.md5(token.encode()).hexdigest(), 16)

            for i in range(self.num_bits):
                if token_hash & (1 << (i % 128)):
                    weights[i] += 1
                else:
                    weights[i] -= 1

        fingerprint = 0
        for i in range(self.num_bits):
            if weights[i] > 0:
                fingerprint |= (1 << i)

        return fingerprint

    @staticmethod
    def hamming_distance(h1: int, h2: int) -> int:
        xor = h1 ^ h2
        count = 0
        while xor:
            count += xor & 1
            xor >>= 1
        return count

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        words = []
        current = []
        for ch in text:
            if ch.isalnum():
                current.append(ch)
            else:
                if current:
                    words.append(''.join(current))
                current = []
        if current:
            words.append(''.join(current))
        STOP = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                'would', 'could', 'should', 'may', 'might', 'can', 'shall',
                'of', 'in', 'to', 'for', 'with', 'on', 'at', 'from', 'by',
                'and', 'or', 'but', 'not', 'no', 'if', 'then', 'than',
                'that', 'this', 'it', 'its', 'i', 'me', 'my', 'we', 'our',
                'you', 'your', 'he', 'she', 'they', 'them', 'their',
                'what', 'which', 'who', 'whom', 'where', 'when', 'how',
                'there', 'here', 'give', 'show', 'tell', 'get', 'find'}
        return [w for w in words if w not in STOP and len(w) > 1]


class SemanticQueryCache:

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600,
                 similarity_threshold: int = 5):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.similarity_threshold = similarity_threshold

        self.simhash = SimHash(num_bits=64)

        self.cache: Dict[int, Dict] = {}
        self.lru_order: deque = deque()
        self.frequency: Counter = Counter()
        self.hits = 0
        self.misses = 0
        self.near_hits = 0

    def get(self, query: str) -> Optional[Dict]:
        fingerprint = self.simhash.hash(query)

        if fingerprint in self.cache:
            entry = self.cache[fingerprint]
            if time.time() - entry['timestamp'] <= self.ttl_seconds:
                self.hits += 1
                self.frequency[fingerprint] += 1
                self._promote(fingerprint)
                return entry['result']
            else:
                del self.cache[fingerprint]

        for cached_fp, entry in list(self.cache.items()):
            dist = SimHash.hamming_distance(fingerprint, cached_fp)
            if dist <= self.similarity_threshold:
                if time.time() - entry['timestamp'] <= self.ttl_seconds:
                    self.near_hits += 1
                    self.frequency[cached_fp] += 1
                    self._promote(cached_fp)
                    return entry['result']

        self.misses += 1
        return None

    def put(self, query: str, result: Dict):
        fingerprint = self.simhash.hash(query)

        while len(self.cache) >= self.max_size:
            self._evict()

        self.cache[fingerprint] = {
            'query': query,
            'result': result,
            'timestamp': time.time(),
        }
        self.lru_order.append(fingerprint)
        self.frequency[fingerprint] = 1

    def _promote(self, fingerprint: int):
        try:
            self.lru_order.remove(fingerprint)
        except ValueError:
            pass
        self.lru_order.append(fingerprint)

    def _evict(self):
        if not self.lru_order:
            return

        candidates = []
        for _ in range(min(10, len(self.lru_order))):
            fp = self.lru_order.popleft()
            candidates.append(fp)

        if candidates:
            victim = min(candidates, key=lambda fp: self.frequency.get(fp, 0))
            candidates.remove(victim)
            for fp in candidates:
                self.lru_order.appendleft(fp)
            self.cache.pop(victim, None)
            self.frequency.pop(victim, None)

    def warm(self, historical_queries: List[Tuple[str, Dict]]):
        for query, result in historical_queries:
            self.put(query, result)

    def get_stats(self) -> Dict:
        total = self.hits + self.near_hits + self.misses
        return {
            'cache_size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'near_hits': self.near_hits,
            'misses': self.misses,
            'hit_rate': (self.hits + self.near_hits) / max(total, 1),
            'total_queries': total,
        }


class TableStatistics:

    def __init__(self, table_name: str, row_count: int = 0):
        self.table_name = table_name
        self.row_count = row_count
        self.size_bytes = 0
        self.partitions: List[str] = []
        self.sort_columns: List[str] = []
        self.column_stats: Dict[str, Dict] = {}

    def add_column_stats(self, col_name: str, ndv: int = 0,
                         min_val: Any = None, max_val: Any = None,
                         null_fraction: float = 0.0,
                         histogram: List[Tuple[Any, int]] = None) -> None:
        self.column_stats[col_name] = {
            'ndv': ndv,
            'min': min_val,
            'max': max_val,
            'null_fraction': null_fraction,
            'histogram': histogram or [],
        }

    def selectivity(self, col_name: str, operator: str, value: Any) -> float:
        stats = self.column_stats.get(col_name, {})
        ndv = stats.get('ndv', 100)
        null_frac = stats.get('null_fraction', 0.0)

        if operator == '=':
            return (1.0 - null_frac) / max(ndv, 1)
        elif operator in ('<', '<=', '>', '>='):
            return (1.0 - null_frac) * 0.33
        elif operator == 'LIKE':
            return 0.1
        elif operator == 'IN':
            return min(1.0, 0.05)
        elif operator == 'IS NULL':
            return null_frac
        elif operator == 'IS NOT NULL':
            return 1.0 - null_frac
        else:
            return 0.5


class AdaptiveQueryPlanner:

    BROADCAST_THRESHOLD_BYTES = 100 * 1024 * 1024
    BROADCAST_THRESHOLD_ROWS = 100_000

    def __init__(self):
        self.table_stats: Dict[str, TableStatistics] = {}

    def register_table(self, stats: TableStatistics) -> None:
        self.table_stats[stats.table_name] = stats

    def estimate_join_cost(self, left_table: str, right_table: str,
                           join_col: str) -> Dict:
        left = self.table_stats.get(left_table)
        right = self.table_stats.get(right_table)

        if not left or not right:
            return {'cost': 1e9, 'strategy': 'shuffle_hash', 'output_rows': 0}

        left_rows = left.row_count
        right_rows = right.row_count

        left_ndv = left.column_stats.get(join_col, {}).get('ndv', left_rows)
        right_ndv = right.column_stats.get(join_col, {}).get('ndv', right_rows)
        join_selectivity = 1.0 / max(max(left_ndv, right_ndv), 1)
        output_rows = int(left_rows * right_rows * join_selectivity)

        if right_rows <= self.BROADCAST_THRESHOLD_ROWS:
            strategy = 'broadcast_right'
            cost = right_rows * 0.001 + left_rows * 0.01
        elif left_rows <= self.BROADCAST_THRESHOLD_ROWS:
            strategy = 'broadcast_left'
            cost = left_rows * 0.001 + right_rows * 0.01
        else:
            strategy = 'shuffle_hash'
            cost = (left_rows + right_rows) * 0.05

        return {
            'cost': cost,
            'strategy': strategy,
            'output_rows': output_rows,
            'left_rows': left_rows,
            'right_rows': right_rows,
        }

    def optimal_join_order(self, tables: List[str],
                           join_graph: Dict[str, Dict[str, str]]) -> List[Tuple[str, str, str]]:
        if len(tables) <= 1:
            return []
        if len(tables) == 2:
            t1, t2 = tables
            col = join_graph.get(t1, {}).get(t2, 'MEMBER_ID')
            return [(t1, t2, col)]

        if len(tables) <= 8:
            return self._dp_join_order(tables, join_graph)
        else:
            return self._greedy_join_order(tables, join_graph)

    def _dp_join_order(self, tables: List[str],
                       join_graph: Dict[str, Dict[str, str]]) -> List[Tuple[str, str, str]]:
        n = len(tables)

        dp = {}

        for t in tables:
            stats = self.table_stats.get(t)
            rows = stats.row_count if stats else 10000
            dp[frozenset([t])] = (rows * 0.01, [], rows)

        for size in range(2, n + 1):
            for subset in self._subsets_of_size(tables, size):
                fset = frozenset(subset)
                best = (float('inf'), [], 0)

                for left_set in self._subsets_of_size(list(subset), size - 1):
                    left_fset = frozenset(left_set)
                    right_fset = fset - left_fset

                    if len(right_fset) != 1:
                        continue

                    right_table = list(right_fset)[0]

                    if left_fset not in dp:
                        continue

                    left_cost, left_joins, left_rows = dp[left_fset]

                    join_col = None
                    join_from = None
                    for lt in left_set:
                        if right_table in join_graph.get(lt, {}):
                            join_col = join_graph[lt][right_table]
                            join_from = lt
                            break

                    if not join_col:
                        continue

                    join_info = self.estimate_join_cost(join_from, right_table, join_col)
                    total_cost = left_cost + join_info['cost']

                    if total_cost < best[0]:
                        new_joins = left_joins + [(join_from, right_table, join_col)]
                        best = (total_cost, new_joins, join_info['output_rows'])

                if best[0] < float('inf'):
                    dp[fset] = best

        full_set = frozenset(tables)
        if full_set in dp:
            return dp[full_set][1]
        else:
            return self._greedy_join_order(tables, join_graph)

    def _greedy_join_order(self, tables: List[str],
                           join_graph: Dict[str, Dict[str, str]]) -> List[Tuple[str, str, str]]:
        if not tables:
            return []

        remaining = set(tables)
        start = min(remaining, key=lambda t:
                    self.table_stats.get(t, TableStatistics(t, 10**9)).row_count)
        joined = {start}
        remaining.discard(start)
        joins = []

        while remaining:
            best_cost = float('inf')
            best_join = None

            for j_table in joined:
                for r_table in remaining:
                    col = join_graph.get(j_table, {}).get(r_table)
                    if col:
                        info = self.estimate_join_cost(j_table, r_table, col)
                        if info['cost'] < best_cost:
                            best_cost = info['cost']
                            best_join = (j_table, r_table, col)

            if best_join:
                joins.append(best_join)
                joined.add(best_join[1])
                remaining.discard(best_join[1])
            else:
                next_table = remaining.pop()
                joins.append((list(joined)[0], next_table, 'MEMBER_ID'))
                joined.add(next_table)

        return joins

    def _subsets_of_size(self, items: List, size: int) -> List[Tuple]:
        if size == 0:
            return [()]
        if size > len(items):
            return []
        if size == len(items):
            return [tuple(items)]

        with_first = self._subsets_of_size(items[1:], size - 1)
        with_first = [tuple([items[0]] + list(s)) for s in with_first]

        without_first = self._subsets_of_size(items[1:], size)

        return with_first + without_first

    def suggest_optimizations(self, tables: List[str], filters: List[str],
                              agg_func: str = None) -> List[Dict]:
        suggestions = []

        for t in tables:
            stats = self.table_stats.get(t)
            if not stats:
                continue

            if stats.partitions:
                for f in filters:
                    for pcol in stats.partitions:
                        if pcol.lower() in f.lower():
                            suggestions.append({
                                'type': 'partition_pruning',
                                'table': t,
                                'partition': pcol,
                                'filter': f,
                                'rationale': f'Filter on partition column {pcol} enables partition pruning, '
                                             f'scanning only relevant data partitions instead of full table.',
                                'estimated_speedup': '10-100x',
                            })

            if stats.row_count <= self.BROADCAST_THRESHOLD_ROWS:
                suggestions.append({
                    'type': 'broadcast_hint',
                    'table': t,
                    'row_count': stats.row_count,
                    'rationale': f'{t} is small enough ({stats.row_count:,} rows) for broadcast join, '
                                 f'avoiding expensive shuffle.',
                    'hint': f'/*+ BROADCAST({t}) */',
                })

            if stats.sort_columns:
                suggestions.append({
                    'type': 'zorder',
                    'table': t,
                    'columns': stats.sort_columns,
                    'rationale': f'Table {t} has ZORDER on {stats.sort_columns}, '
                                 f'enabling data skipping for filtered queries.',
                })

        if agg_func in ('COUNT', 'AVG') and any(
            self.table_stats.get(t, TableStatistics(t)).row_count > 1_000_000
            for t in tables
        ):
            suggestions.append({
                'type': 'approximate_aggregation',
                'rationale': 'For tables with >1M rows, consider APPROX_COUNT_DISTINCT '
                             'or PERCENTILE_APPROX for faster results with <2% error.',
                'alternatives': {
                    'COUNT(DISTINCT x)': 'APPROX_COUNT_DISTINCT(x)',
                    'PERCENTILE(x, 0.5)': 'PERCENTILE_APPROX(x, 0.5)',
                },
            })

        return suggestions

    @classmethod
    def from_catalog(cls, catalog_dir: str) -> 'AdaptiveQueryPlanner':
        planner = cls()
        tables_dir = os.path.join(catalog_dir, 'tables')

        if not os.path.exists(tables_dir):
            return planner

        PARTITION_STRATEGY = {
            'claims': ['SERVICE_DATE'],
            'encounters': ['SERVICE_DATE'],
            'members': ['KP_REGION'],
            'prescriptions': ['PRESCRIPTION_DATE'],
            'diagnoses': ['DIAGNOSIS_DATE'],
        }

        ZORDER_STRATEGY = {
            'claims': ['MEMBER_ID', 'RENDERING_NPI', 'CLAIM_STATUS'],
            'encounters': ['MEMBER_ID', 'RENDERING_NPI', 'VISIT_TYPE'],
            'members': ['MEMBER_ID', 'KP_REGION'],
            'providers': ['NPI', 'SPECIALTY'],
            'prescriptions': ['MEMBER_ID', 'PRESCRIBING_NPI'],
            'diagnoses': ['MEMBER_ID', 'ICD10_CODE'],
            'referrals': ['MEMBER_ID', 'REFERRING_NPI'],
        }

        for fname in os.listdir(tables_dir):
            if not fname.endswith('.json'):
                continue
            try:
                with open(os.path.join(tables_dir, fname)) as f:
                    profile = json.load(f)
            except Exception:
                continue

            table_name = profile.get('table_name', fname.replace('.json', ''))
            row_count = profile.get('total_rows', 0)

            stats = TableStatistics(table_name, row_count)
            stats.size_bytes = row_count * 500
            stats.partitions = PARTITION_STRATEGY.get(table_name, [])
            stats.sort_columns = ZORDER_STRATEGY.get(table_name, [])

            columns = profile.get('columns', {})
            if isinstance(columns, dict):
                columns = list(columns.values())

            for col_info in columns:
                col_name = col_info.get('column_name', col_info.get('name', ''))
                if not col_name:
                    continue
                stats.add_column_stats(
                    col_name,
                    ndv=col_info.get('cardinality', 0),
                    null_fraction=col_info.get('null_percentage', 0) / 100.0,
                )

            planner.register_table(stats)

        return planner


class DatabricksDialect:

    DEFAULT_CATALOG = 'healthcare_prod'
    DEFAULT_SCHEMA = 'analytics'

    DATE_TRANSLATIONS = {
        "SUBSTR(SERVICE_DATE, 1, 7)": "DATE_FORMAT(SERVICE_DATE, 'yyyy-MM')",
        "SUBSTR(SERVICE_DATE, 1, 4)": "YEAR(SERVICE_DATE)",
        "DATE('now')": "CURRENT_DATE()",
        "DATETIME('now')": "CURRENT_TIMESTAMP()",
    }

    TYPE_TRANSLATIONS = {
        'CAST(X AS REAL)': 'CAST(X AS DOUBLE)',
        'CAST(X AS INTEGER)': 'CAST(X AS INT)',
    }

    @classmethod
    def translate(cls, sqlite_sql: str, planner: AdaptiveQueryPlanner = None,
                  tables: List[str] = None,
                  use_catalog: bool = True,
                  approximate: bool = False) -> str:
        sql = sqlite_sql

        for sqlite_func, spark_func in cls.DATE_TRANSLATIONS.items():
            sql = sql.replace(sqlite_func, spark_func)

        import re
        sql = re.sub(
            r"SUBSTR\((\w+), 1, 7\)",
            r"DATE_FORMAT(\1, 'yyyy-MM')",
            sql
        )
        sql = re.sub(
            r"SUBSTR\((\w+), 1, 4\)",
            r"YEAR(\1)",
            sql
        )

        sql = sql.replace('AS REAL)', 'AS DOUBLE)')

        sql = re.sub(
            r"(\w+) LIKE '(\d{4})%'",
            r"YEAR(\1) = \2",
            sql
        )

        if use_catalog and tables:
            for table in tables:
                qualified = f"{cls.DEFAULT_CATALOG}.{cls.DEFAULT_SCHEMA}.{table}"
                sql = re.sub(
                    rf'(FROM|JOIN)\s+{table}\b',
                    rf'\1 {qualified}',
                    sql,
                    flags=re.IGNORECASE
                )

        if planner and tables:
            for table in tables:
                stats = planner.table_stats.get(table)
                if stats and stats.row_count <= planner.BROADCAST_THRESHOLD_ROWS:
                    if f'JOIN {table}' in sql or f'JOIN {cls.DEFAULT_CATALOG}' in sql:
                        hint = f'/*+ BROADCAST({table}) */'
                        sql = sql.replace('SELECT ', f'SELECT {hint} ', 1)

        if approximate:
            sql = re.sub(
                r'COUNT\(DISTINCT\s+(\w+)\)',
                r'APPROX_COUNT_DISTINCT(\1)',
                sql
            )

        sql = sql.rstrip(';').strip()

        return sql

    @classmethod
    def generate_window_query(cls, base_column: str, partition_by: str,
                              order_by: str, window_func: str = 'ROW_NUMBER',
                              table: str = 'claims') -> str:
        return (
            f"SELECT *, {window_func}() OVER ("
            f"PARTITION BY {partition_by} ORDER BY {order_by}"
            f") as {window_func.lower()}_rank "
            f"FROM {cls.DEFAULT_CATALOG}.{cls.DEFAULT_SCHEMA}.{table}"
        )

    @classmethod
    def generate_delta_optimize(cls, table: str, zorder_cols: List[str] = None) -> str:
        qualified = f"{cls.DEFAULT_CATALOG}.{cls.DEFAULT_SCHEMA}.{table}"
        sql = f"OPTIMIZE {qualified}"
        if zorder_cols:
            sql += f" ZORDER BY ({', '.join(zorder_cols)})"
        return sql

    @classmethod
    def generate_analyze(cls, table: str) -> str:
        qualified = f"{cls.DEFAULT_CATALOG}.{cls.DEFAULT_SCHEMA}.{table}"
        return f"ANALYZE TABLE {qualified} COMPUTE STATISTICS FOR ALL COLUMNS"


class ColumnLineage:

    def __init__(self):
        self.lineage: Dict[str, Dict] = {}
        self.reverse_lineage: Dict[str, List[str]] = defaultdict(list)

    def add_lineage(self, derived_col: str, source_columns: List[str],
                    transformation: str, freshness_ts: float = None) -> None:
        self.lineage[derived_col] = {
            'sources': source_columns,
            'transformation': transformation,
            'freshness': freshness_ts or time.time(),
            'created_at': time.time(),
        }
        for src in source_columns:
            self.reverse_lineage[src].append(derived_col)

    def impact_analysis(self, source_column: str) -> List[str]:
        affected = set()
        queue = [source_column]
        while queue:
            col = queue.pop(0)
            for derived in self.reverse_lineage.get(col, []):
                if derived not in affected:
                    affected.add(derived)
                    queue.append(derived)
        return sorted(affected)

    def get_sources(self, derived_col: str) -> List[str]:
        sources = set()
        queue = [derived_col]
        while queue:
            col = queue.pop(0)
            entry = self.lineage.get(col)
            if entry:
                for src in entry['sources']:
                    sources.add(src)
                    queue.append(src)
            else:
                sources.add(col)
        return sorted(sources)

    def check_freshness(self, derived_col: str, max_age_seconds: int = 86400) -> Dict:
        entry = self.lineage.get(derived_col, {})
        freshness = entry.get('freshness', 0)
        age = time.time() - freshness
        return {
            'column': derived_col,
            'last_updated': freshness,
            'age_seconds': age,
            'is_fresh': age <= max_age_seconds,
            'sources': entry.get('sources', []),
        }


class QueryPatternMiner:

    def __init__(self):
        self.patterns: Counter = Counter()
        self.query_log: List[Dict] = []

    def log_query(self, question: str, sql: str, agg_info: Dict,
                  tables: List[str], execution_time: float = 0) -> None:
        fingerprint = self._extract_pattern(agg_info, tables)
        self.patterns[fingerprint] += 1
        self.query_log.append({
            'question': question,
            'sql': sql,
            'pattern': fingerprint,
            'tables': tables,
            'execution_time': execution_time,
            'timestamp': time.time(),
        })

    def _extract_pattern(self, agg_info: Dict, tables: List[str]) -> str:
        agg = agg_info.get('agg_func', 'LOOKUP')
        group = 'GROUP' if agg_info.get('group_by_terms') else 'SCALAR'
        top = 'TOP' if agg_info.get('top_n') else ''
        num_tables = len(tables)
        join = f'{num_tables}JOIN' if num_tables > 1 else 'SINGLE'
        parts = [agg, group, join]
        if top:
            parts.append(top)
        return '_'.join(parts)

    def get_common_patterns(self, min_count: int = 2) -> List[Tuple[str, int]]:
        return [(p, c) for p, c in self.patterns.most_common() if c >= min_count]

    def suggest_materialized_views(self) -> List[Dict]:
        suggestions = []
        for pattern, count in self.patterns.most_common(10):
            if count >= 3:
                examples = [q for q in self.query_log if q['pattern'] == pattern]
                avg_time = sum(q['execution_time'] for q in examples) / max(len(examples), 1)

                if avg_time > 1.0:
                    suggestions.append({
                        'pattern': pattern,
                        'frequency': count,
                        'avg_execution_time': avg_time,
                        'example_sql': examples[0]['sql'] if examples else '',
                        'rationale': f'Pattern "{pattern}" executed {count} times '
                                     f'with avg {avg_time:.1f}s — good candidate for materialized view.',
                    })

        return suggestions

    def detect_anomalous_queries(self) -> List[Dict]:
        anomalies = []

        for q in self.query_log:
            if self.patterns[q['pattern']] == 1:
                anomalies.append({
                    'query': q['question'],
                    'pattern': q['pattern'],
                    'reason': 'Rare pattern — possible error or unusual request',
                })

            if q['execution_time'] > 30:
                anomalies.append({
                    'query': q['question'],
                    'execution_time': q['execution_time'],
                    'reason': 'Slow execution — may need optimization',
                })

        return anomalies


class ScaleEngine:

    def __init__(self, catalog_dir: str = None):
        if catalog_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            catalog_dir = os.path.join(os.path.dirname(script_dir), 'semantic_catalog')

        self.catalog_dir = catalog_dir

        self.knowledge_graph = None
        self.vector_index = None
        self.query_cache = None
        self.query_planner = None
        self.lineage = ColumnLineage()
        self.pattern_miner = QueryPatternMiner()

        self._initialized = False

    def initialize(self) -> Dict:
        stats = {}

        self.knowledge_graph = KnowledgeGraph.from_catalog(self.catalog_dir)
        kg_stats = self.knowledge_graph.get_stats()
        stats['knowledge_graph'] = kg_stats

        self.vector_index = VectorIndex(num_planes=8, num_tables=4, dim=300)
        documents = []
        for nid, node in self.knowledge_graph.nodes.items():
            text_parts = [nid.replace(':', ' ').replace('.', ' ')]
            for k, v in node.properties.items():
                if isinstance(v, str):
                    text_parts.append(v)
            documents.append((nid, ' '.join(text_parts)))
        self.vector_index.build_vocabulary(documents)
        for doc_id, text in documents:
            self.vector_index.index_document(doc_id, text)
        stats['vector_index'] = self.vector_index.get_stats()

        self.query_cache = SemanticQueryCache(max_size=500, ttl_seconds=3600)
        stats['query_cache'] = self.query_cache.get_stats()

        self.query_planner = AdaptiveQueryPlanner.from_catalog(self.catalog_dir)
        stats['query_planner'] = {
            'tables_registered': len(self.query_planner.table_stats),
            'tables': {t: {'rows': s.row_count, 'partitions': s.partitions,
                           'zorder': s.sort_columns}
                       for t, s in self.query_planner.table_stats.items()},
        }

        self._setup_lineage()

        self._initialized = True
        return stats

    def _setup_lineage(self):
        self.lineage.add_lineage(
            'denial_rate',
            ['claims.CLAIM_STATUS'],
            'ROUND(100.0 * SUM(CASE WHEN CLAIM_STATUS = \'DENIED\' THEN 1 ELSE 0 END) / COUNT(*), 2)'
        )
        self.lineage.add_lineage(
            'utilization_rate',
            ['encounters.ENCOUNTER_ID', 'members.MEMBER_ID'],
            'COUNT(DISTINCT ENCOUNTER_ID) / COUNT(DISTINCT MEMBER_ID)'
        )
        self.lineage.add_lineage(
            'avg_cost',
            ['claims.BILLED_AMOUNT'],
            'AVG(BILLED_AMOUNT)'
        )

    def search_schema(self, query: str, k: int = 10) -> List[Dict]:
        if not self.vector_index:
            return []

        results = self.vector_index.search(query, k=k)
        enriched = []
        for doc_id, similarity in results:
            node = self.knowledge_graph.nodes.get(doc_id) if self.knowledge_graph else None
            enriched.append({
                'id': doc_id,
                'type': node.type if node else 'unknown',
                'similarity': round(similarity, 4),
                'importance': round(node.importance, 6) if node else 0,
                'properties': node.properties if node else {},
            })
        return enriched

    def get_relevant_subgraph(self, query_terms: List[str]) -> Dict:
        if not self.knowledge_graph:
            return {}

        subgraph = self.knowledge_graph.extract_relevant_subgraph(query_terms)
        return {
            'nodes': len(subgraph.nodes),
            'edges': len(subgraph.edges),
            'tables': [nid for nid, n in subgraph.nodes.items() if n.type == 'table'],
            'columns': [nid for nid, n in subgraph.nodes.items() if n.type == 'column'],
            'concepts': [nid for nid, n in subgraph.nodes.items() if n.type == 'concept'],
        }

    def optimize_query(self, tables: List[str], filters: List[str],
                       agg_func: str = None) -> Dict:
        if not self.query_planner:
            return {}

        suggestions = self.query_planner.suggest_optimizations(tables, filters, agg_func)
        return {
            'suggestions': suggestions,
            'join_order': self.query_planner.optimal_join_order(
                tables, {t: {} for t in tables}
            ) if len(tables) > 1 else [],
        }

    def translate_to_databricks(self, sqlite_sql: str,
                                 tables: List[str] = None) -> str:
        return DatabricksDialect.translate(
            sqlite_sql,
            planner=self.query_planner,
            tables=tables,
            use_catalog=True,
        )

    def get_full_stats(self) -> Dict:
        return {
            'knowledge_graph': self.knowledge_graph.get_stats() if self.knowledge_graph else {},
            'vector_index': self.vector_index.get_stats() if self.vector_index else {},
            'query_cache': self.query_cache.get_stats() if self.query_cache else {},
            'query_planner': {
                'tables': len(self.query_planner.table_stats) if self.query_planner else 0,
            },
            'pattern_miner': {
                'patterns': len(self.pattern_miner.patterns),
                'queries_logged': len(self.pattern_miner.query_log),
            },
        }
