import sqlite3
import json
import logging
import os
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Optional, Any
import hashlib
import statistics


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KnowledgeGraph:

    def __init__(self, db_path: str, kg_db_path: str = None):
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Healthcare database not found: {db_path}")

        self.db_path = db_path

        if kg_db_path is None:
            data_dir = os.path.dirname(db_path)
            kg_db_path = os.path.join(data_dir, 'knowledge_graph.db')

        self.kg_db_path = kg_db_path

        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.kg_conn = sqlite3.connect(kg_db_path)

        self._init_kg_schema()

        self._schema_cache = {}
        self._relationships = {}
        self._column_semantics = {}

        logger.info(f"KnowledgeGraph initialized with db={db_path}, kg_db={kg_db_path}")

    def _init_kg_schema(self):
        cursor = self.kg_conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS schema_discovery (
                table_name TEXT PRIMARY KEY,
                column_count INTEGER,
                row_count INTEGER,
                columns_json TEXT,
                primary_keys TEXT,
                discovered_at TIMESTAMP,
                updated_at TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS relationships (
                relationship_id TEXT PRIMARY KEY,
                table_a TEXT NOT NULL,
                table_b TEXT NOT NULL,
                column_a TEXT NOT NULL,
                column_b TEXT NOT NULL,
                relationship_type TEXT,
                join_direction TEXT,
                join_clause TEXT,
                confidence REAL,
                frequency INTEGER,
                discovered_at TIMESTAMP,
                UNIQUE(table_a, table_b, column_a, column_b)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS column_semantics (
                semantic_id TEXT PRIMARY KEY,
                table_name TEXT NOT NULL,
                column_name TEXT NOT NULL,
                semantic_type TEXT,
                data_type TEXT,
                sample_values TEXT,
                value_distribution TEXT,
                is_nullable INTEGER,
                is_key INTEGER,
                discovered_at TIMESTAMP,
                UNIQUE(table_name, column_name)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learned_patterns (
                pattern_id TEXT PRIMARY KEY,
                pattern_type TEXT,
                tables_involved TEXT,
                columns_involved TEXT,
                aggregations TEXT,
                filters TEXT,
                join_paths TEXT,
                success_count INTEGER,
                failure_count INTEGER,
                success_rate REAL,
                avg_execution_time REAL,
                last_used TIMESTAMP,
                learned_at TIMESTAMP,
                examples_json TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS query_sql_mappings (
                mapping_id TEXT PRIMARY KEY,
                query_text TEXT,
                query_hash TEXT UNIQUE,
                sql_text TEXT,
                tables_used TEXT,
                pattern_id TEXT,
                success INTEGER,
                execution_time REAL,
                error_message TEXT,
                executed_at TIMESTAMP,
                FOREIGN KEY (pattern_id) REFERENCES learned_patterns(pattern_id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_gaps (
                gap_id TEXT PRIMARY KEY,
                gap_type TEXT,
                table_a TEXT,
                table_b TEXT,
                column_a TEXT,
                column_b TEXT,
                description TEXT,
                priority_score REAL,
                last_encountered TIMESTAMP,
                encounter_count INTEGER,
                detected_at TIMESTAMP,
                resolved_at TIMESTAMP,
                resolution_note TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                metric_id TEXT PRIMARY KEY,
                metric_name TEXT,
                metric_value REAL,
                timestamp TIMESTAMP,
                window_size INTEGER,
                context_json TEXT
            )
        ''')

        cursor.execute('CREATE INDEX IF NOT EXISTS idx_rel_tables ON relationships(table_a, table_b)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_col_sem_table ON column_semantics(table_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_patterns_type ON learned_patterns(pattern_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_query_hash ON query_sql_mappings(query_hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_gaps_priority ON knowledge_gaps(priority_score DESC)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(metric_name, timestamp DESC)')

        self.kg_conn.commit()
        logger.info("Knowledge graph schema initialized")

    def discover_schema(self, force_refresh: bool = False) -> Dict[str, Any]:
        if self._schema_cache and not force_refresh:
            return self._schema_cache

        logger.info("Starting schema discovery...")
        schema = {}

        cursor = self.conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]

        for table in tables:
            cursor.execute(f"PRAGMA table_info({table});")
            columns = cursor.fetchall()

            cursor.execute(f"SELECT COUNT(*) FROM {table};")
            row_count = cursor.fetchone()[0]

            cursor.execute(f"SELECT * FROM {table} LIMIT 5;")
            sample_rows = cursor.fetchall()

            column_info = []
            for col in columns:
                col_dict = {
                    'name': col[1],
                    'type': col[2],
                    'notnull': col[3],
                    'default': col[4],
                    'pk': col[5]
                }
                column_info.append(col_dict)

            schema[table] = {
                'columns': column_info,
                'row_count': row_count,
                'column_count': len(column_info),
                'sample_rows': len(sample_rows)
            }

            self._persist_schema_discovery(table, column_info, row_count)

        self._schema_cache = schema
        logger.info(f"Schema discovery complete: {len(tables)} tables discovered")
        return schema

    def _persist_schema_discovery(self, table_name: str, columns: List[Dict], row_count: int):
        cursor = self.kg_conn.cursor()

        columns_json = json.dumps(columns)
        now = datetime.utcnow().isoformat()

        cursor.execute('''
            INSERT OR REPLACE INTO schema_discovery
            (table_name, column_count, row_count, columns_json, discovered_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (table_name, len(columns), row_count, columns_json, now, now))

        self.kg_conn.commit()

    def discover_relationships(self, force_refresh: bool = False) -> Dict[Tuple[str, str], List[Dict]]:
        if self._relationships and not force_refresh:
            return self._relationships

        logger.info("Starting relationship discovery...")
        relationships = defaultdict(list)

        if not self._schema_cache:
            self.discover_schema()

        schema = self._schema_cache
        tables = list(schema.keys())

        cursor = self.conn.cursor()

        for i, table_a in enumerate(tables):
            cols_a = {col['name'] for col in schema[table_a]['columns']}

            for table_b in tables[i+1:]:
                cols_b = {col['name'] for col in schema[table_b]['columns']}

                common_cols = cols_a & cols_b

                for col in common_cols:
                    rel = {
                        'table_a': table_a,
                        'table_b': table_b,
                        'column_a': col,
                        'column_b': col,
                        'relationship_type': 'shared_column',
                        'join_direction': 'bidirectional'
                    }

                    cursor.execute(f"SELECT COUNT(DISTINCT {col}) FROM {table_a};")
                    count_a = cursor.fetchone()[0]
                    cursor.execute(f"SELECT COUNT(DISTINCT {col}) FROM {table_b};")
                    count_b = cursor.fetchone()[0]

                    if count_a < count_b:
                        rel['join_direction'] = f"{table_a}.{col} -> {table_b}.{col}"
                    elif count_b < count_a:
                        rel['join_direction'] = f"{table_b}.{col} -> {table_a}.{col}"

                    rel['join_clause'] = f"{table_a}.{col} = {table_b}.{col}"
                    rel['confidence'] = 0.9

                    relationships[(table_a, table_b)].append(rel)
                    relationships[(table_b, table_a)].append(rel)

                    self._persist_relationship(rel)

        self._relationships = dict(relationships)
        logger.info(f"Relationship discovery complete: {len(self._relationships)} relationship pairs found")
        return self._relationships

    def _persist_relationship(self, rel: Dict[str, Any]):
        cursor = self.kg_conn.cursor()

        rel_id = hashlib.sha256(
            f"{rel['table_a']}_{rel['table_b']}_{rel['column_a']}_{rel['column_b']}".encode()
        ).hexdigest()

        now = datetime.utcnow().isoformat()

        cursor.execute('''
            INSERT OR IGNORE INTO relationships
            (relationship_id, table_a, table_b, column_a, column_b, relationship_type,
             join_direction, join_clause, confidence, frequency, discovered_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            rel_id,
            rel['table_a'],
            rel['table_b'],
            rel['column_a'],
            rel['column_b'],
            rel.get('relationship_type', 'unknown'),
            rel.get('join_direction', 'unknown'),
            rel.get('join_clause', ''),
            rel.get('confidence', 0.5),
            1,
            now
        ))

        self.kg_conn.commit()

    def discover_column_semantics(self, force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
        if self._column_semantics and not force_refresh:
            return self._column_semantics

        logger.info("Starting column semantics discovery...")
        semantics = {}

        if not self._schema_cache:
            self.discover_schema()

        schema = self._schema_cache
        cursor = self.conn.cursor()

        semantic_patterns = {
            '_ID': 'identifier',
            '_NPI': 'provider_identifier',
            '_CODE': 'code',
            '_DATE': 'date',
            '_AMOUNT': 'amount',
            '_COUNT': 'count',
            'DESCRIPTION': 'text',
            'NAME': 'name',
            'STATUS': 'category',
            'TYPE': 'category',
            'REASON': 'text',
            'GENDER': 'category',
            'RACE': 'category'
        }

        for table, info in schema.items():
            for col_info in info['columns']:
                col_name = col_info['name']
                col_type = col_info['type'].upper()

                semantic_type = 'unknown'
                for pattern, sem_type in semantic_patterns.items():
                    if pattern in col_name.upper():
                        semantic_type = sem_type
                        break

                if semantic_type == 'unknown':
                    if 'INT' in col_type:
                        semantic_type = 'numeric'
                    elif 'REAL' in col_type or 'FLOAT' in col_type:
                        semantic_type = 'decimal'
                    elif 'TEXT' in col_type:
                        semantic_type = 'text'

                try:
                    cursor.execute(f"""
                        SELECT {col_name}, COUNT(*) as cnt
                        FROM {table}
                        WHERE {col_name} IS NOT NULL
                        GROUP BY {col_name}
                        ORDER BY cnt DESC
                        LIMIT 10
                    """)
                    value_dist = {row[0]: row[1] for row in cursor.fetchall()}
                except Exception as e:
                    value_dist = {}
                    logger.debug(f"Could not sample {table}.{col_name}: {e}")

                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE {col_name} IS NULL;")
                    null_count = cursor.fetchone()[0]
                except Exception:
                    null_count = 0

                sem_dict = {
                    'table': table,
                    'column': col_name,
                    'semantic_type': semantic_type,
                    'data_type': col_type,
                    'sample_values': list(value_dist.keys())[:5],
                    'value_distribution': value_dist,
                    'null_count': null_count,
                    'is_key': col_info['pk'] > 0,
                    'is_nullable': col_info['notnull'] == 0
                }

                key = f"{table}.{col_name}"
                semantics[key] = sem_dict

                self._persist_column_semantics(sem_dict)

        self._column_semantics = semantics
        logger.info(f"Column semantics discovery complete: {len(semantics)} columns analyzed")
        return semantics

    def _persist_column_semantics(self, sem_dict: Dict[str, Any]):
        cursor = self.kg_conn.cursor()

        sem_id = f"{sem_dict['table']}.{sem_dict['column']}"
        now = datetime.utcnow().isoformat()

        cursor.execute('''
            INSERT OR REPLACE INTO column_semantics
            (semantic_id, table_name, column_name, semantic_type, data_type,
             sample_values, value_distribution, is_nullable, is_key, discovered_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            sem_id,
            sem_dict['table'],
            sem_dict['column'],
            sem_dict['semantic_type'],
            sem_dict['data_type'],
            json.dumps(sem_dict['sample_values']),
            json.dumps(dict(list(sem_dict['value_distribution'].items())[:10])),
            int(sem_dict['is_nullable']),
            int(sem_dict['is_key']),
            now
        ))

        self.kg_conn.commit()

    def learn_pattern(self, question: str, sql: str, success: bool,
                     execution_time: float = 0.0, error_msg: str = None) -> str:
        query_hash = hashlib.sha256(question.encode()).hexdigest()
        mapping_id = hashlib.sha256(f"{query_hash}_{int(datetime.utcnow().timestamp())}".encode()).hexdigest()

        tables_used = self._extract_tables_from_sql(sql)

        pattern_type = self._detect_pattern_type(sql)

        extracted = self._extract_sql_components(sql, tables_used)

        pattern_id = self._get_or_create_pattern(
            pattern_type=pattern_type,
            tables_involved=tables_used,
            **extracted
        )

        cursor = self.kg_conn.cursor()
        now = datetime.utcnow().isoformat()

        cursor.execute('''
            INSERT OR REPLACE INTO query_sql_mappings
            (mapping_id, query_text, query_hash, sql_text, tables_used, pattern_id,
             success, execution_time, error_message, executed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            mapping_id,
            question,
            query_hash,
            sql,
            json.dumps(tables_used),
            pattern_id,
            int(success),
            execution_time,
            error_msg,
            now
        ))

        if success:
            cursor.execute('''
                UPDATE learned_patterns
                SET success_count = success_count + 1, last_used = ?
                WHERE pattern_id = ?
            ''', (now, pattern_id))
        else:
            cursor.execute('''
                UPDATE learned_patterns
                SET failure_count = failure_count + 1
                WHERE pattern_id = ?
            ''', (pattern_id,))

        cursor.execute('''
            SELECT success_count, failure_count FROM learned_patterns WHERE pattern_id = ?
        ''', (pattern_id,))
        row = cursor.fetchone()
        if row:
            success_count, failure_count = row
            total = success_count + failure_count
            success_rate = success_count / total if total > 0 else 0
            cursor.execute('''
                UPDATE learned_patterns SET success_rate = ? WHERE pattern_id = ?
            ''', (success_rate, pattern_id))

        self.kg_conn.commit()

        logger.info(f"Pattern learned: {pattern_id}, type={pattern_type}, success={success}")
        return pattern_id

    def _extract_tables_from_sql(self, sql: str) -> List[str]:
        tables = []
        sql_upper = sql.upper()

        from_start = sql_upper.find('FROM')
        if from_start != -1:
            after_from = sql[from_start + 4:].strip()
            table_name = after_from.split()[0].strip('(),;')
            if table_name:
                tables.append(table_name)

        import re
        join_pattern = r'(?:INNER\s+|LEFT\s+|RIGHT\s+|FULL\s+)?JOIN\s+(\w+)'
        matches = re.findall(join_pattern, sql_upper)
        tables.extend(matches)

        return list(set(tables))

    def _detect_pattern_type(self, sql: str) -> str:
        sql_upper = sql.upper()

        pattern_indicators = {
            'aggregation': ['COUNT(', 'SUM(', 'AVG(', 'MAX(', 'MIN(', 'GROUP BY'],
            'filter': ['WHERE', 'HAVING'],
            'join': ['JOIN', 'INNER', 'LEFT', 'RIGHT', 'FULL'],
            'sorting': ['ORDER BY'],
            'distinct': ['DISTINCT'],
            'limit': ['LIMIT'],
            'union': ['UNION'],
            'subquery': ['(SELECT']
        }

        detected = []
        for pattern_type, indicators in pattern_indicators.items():
            for indicator in indicators:
                if indicator in sql_upper:
                    detected.append(pattern_type)
                    break

        if not detected:
            detected = ['simple_select']

        return ','.join(sorted(set(detected)))

    def _extract_sql_components(self, sql: str, tables: List[str]) -> Dict[str, Any]:
        import re

        components = {
            'columns_involved': [],
            'aggregations': [],
            'filters': [],
            'joins': []
        }

        sql_upper = sql.upper()

        col_pattern = r'(\w+)\.(\w+)'
        col_matches = re.findall(col_pattern, sql)
        components['columns_involved'] = list(set([f"{t}.{c}" for t, c in col_matches]))

        agg_pattern = r'(COUNT|SUM|AVG|MAX|MIN|COUNT DISTINCT)\s*\('
        agg_matches = re.findall(agg_pattern, sql_upper)
        components['aggregations'] = list(set(agg_matches))

        where_match = re.search(r'WHERE\s+(.+?)(?:GROUP BY|HAVING|ORDER BY|$)', sql_upper)
        if where_match:
            components['filters'] = [where_match.group(1)[:100]]

        join_pattern = r'(?:INNER\s+|LEFT\s+|RIGHT\s+|FULL\s+)?(JOIN)\s+(\w+)'
        join_matches = re.findall(join_pattern, sql_upper)
        components['joins'] = [j[1] for j in join_matches]

        return components

    def _get_or_create_pattern(self, pattern_type: str, tables_involved: List[str],
                               columns_involved: List[str] = None,
                               aggregations: List[str] = None,
                               filters: List[str] = None,
                               joins: List[str] = None) -> str:
        pattern_key = f"{pattern_type}_{','.join(sorted(tables_involved))}"
        pattern_id = hashlib.sha256(pattern_key.encode()).hexdigest()

        cursor = self.kg_conn.cursor()

        cursor.execute('SELECT pattern_id FROM learned_patterns WHERE pattern_id = ?', (pattern_id,))
        if cursor.fetchone():
            return pattern_id

        now = datetime.utcnow().isoformat()

        cursor.execute('''
            INSERT INTO learned_patterns
            (pattern_id, pattern_type, tables_involved, columns_involved, aggregations,
             filters, join_paths, success_count, failure_count, success_rate,
             avg_execution_time, learned_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            pattern_id,
            pattern_type,
            json.dumps(tables_involved),
            json.dumps(columns_involved or []),
            json.dumps(aggregations or []),
            json.dumps(filters or []),
            json.dumps(joins or []),
            0,
            0,
            0.0,
            0.0,
            now
        ))

        self.kg_conn.commit()
        logger.info(f"New pattern created: {pattern_id}")

        return pattern_id

    def find_closest_pattern(self, question: str, min_confidence: float = 0.5) -> Optional[Dict[str, Any]]:
        cursor = self.kg_conn.cursor()

        cursor.execute('''
            SELECT pattern_id, pattern_type, tables_involved, columns_involved,
                   aggregations, filters, join_paths, success_rate
            FROM learned_patterns
            ORDER BY success_rate DESC
            LIMIT 20
        ''')

        patterns = cursor.fetchall()

        if not patterns:
            logger.warning("No learned patterns found")
            return None

        question_upper = question.upper()
        best_match = None
        best_score = 0

        for pattern in patterns:
            pattern_dict = {
                'pattern_id': pattern[0],
                'pattern_type': pattern[1],
                'tables_involved': json.loads(pattern[2]),
                'columns_involved': json.loads(pattern[3]),
                'aggregations': json.loads(pattern[4]),
                'filters': json.loads(pattern[5]),
                'join_paths': json.loads(pattern[6]),
                'success_rate': pattern[7]
            }

            score = pattern_dict['success_rate']

            pattern_keywords = pattern_dict['pattern_type'].split(',')
            for keyword in pattern_keywords:
                if keyword.upper() in question_upper:
                    score += 0.1

            for table in pattern_dict['tables_involved']:
                if table.upper() in question_upper:
                    score += 0.05

            if score > best_score:
                best_score = score
                best_match = pattern_dict

        if best_match and best_score >= min_confidence:
            logger.info(f"Best pattern found: {best_match['pattern_id']} (score={best_score})")
            return best_match

        logger.info(f"No pattern found above confidence threshold {min_confidence}")
        return None

    def generate_query_blueprint(self, question: str) -> Dict[str, Any]:
        blueprint = {
            'question': question,
            'confidence': 0.0,
            'tables': [],
            'joins': [],
            'select_columns': [],
            'aggregations': [],
            'filters': [],
            'group_by': [],
            'order_by': [],
            'limit': None,
            'join_paths': [],
            'notes': []
        }

        pattern = self.find_closest_pattern(question)

        if pattern:
            blueprint['confidence'] = pattern['success_rate']
            blueprint['tables'] = pattern['tables_involved']
            blueprint['filters'] = pattern['filters']
            blueprint['join_paths'] = pattern['join_paths']
            blueprint['notes'].append(f"Based on pattern: {pattern['pattern_id'][:8]}")

            question_upper = question.upper()
            q_aggs = []
            if 'COUNT' in question_upper or 'HOW MANY' in question_upper:
                q_aggs.append('COUNT')
            if 'TOTAL' in question_upper or 'SUM' in question_upper:
                q_aggs.append('SUM')
            if 'AVERAGE' in question_upper or 'AVG' in question_upper or 'MEAN' in question_upper:
                q_aggs.append('AVG')
            if 'MAX' in question_upper or 'HIGHEST' in question_upper or 'MAXIMUM' in question_upper:
                q_aggs.append('MAX')
            if 'MIN' in question_upper or 'LOWEST' in question_upper or 'MINIMUM' in question_upper:
                q_aggs.append('MIN')
            blueprint['aggregations'] = q_aggs if q_aggs else pattern['aggregations']
        else:
            question_upper = question.upper()

            if 'CLAIM' in question_upper:
                blueprint['tables'].append('CLAIMS')
            if 'MEMBER' in question_upper or 'PATIENT' in question_upper:
                blueprint['tables'].append('MEMBERS')
            if 'PROVIDER' in question_upper or 'DOCTOR' in question_upper or 'PHYSICIAN' in question_upper:
                blueprint['tables'].append('PROVIDERS')
            if 'ENCOUNTER' in question_upper or 'VISIT' in question_upper or 'APPOINTMENT' in question_upper:
                blueprint['tables'].append('ENCOUNTERS')
            if 'PRESCRIPTION' in question_upper or 'MEDICATION' in question_upper or 'DRUG' in question_upper:
                blueprint['tables'].append('PRESCRIPTIONS')
            if 'DIAGNOSIS' in question_upper or 'CONDITION' in question_upper:
                blueprint['tables'].append('DIAGNOSES')
            if 'REFERRAL' in question_upper:
                blueprint['tables'].append('REFERRALS')

            if not blueprint['tables']:
                blueprint['tables'] = ['CLAIMS']

            if 'COUNT' in question_upper or 'HOW MANY' in question_upper:
                blueprint['aggregations'].append('COUNT')
            if 'TOTAL' in question_upper or 'SUM' in question_upper:
                blueprint['aggregations'].append('SUM')
            if 'AVERAGE' in question_upper or 'AVG' in question_upper:
                blueprint['aggregations'].append('AVG')
            if 'MAX' in question_upper or 'HIGHEST' in question_upper or 'MAXIMUM' in question_upper:
                blueprint['aggregations'].append('MAX')
            if 'MIN' in question_upper or 'LOWEST' in question_upper or 'MINIMUM' in question_upper:
                blueprint['aggregations'].append('MIN')

            blueprint['notes'].append("Generated using heuristic analysis")
            blueprint['confidence'] = 0.3

        if len(blueprint['tables']) > 1:
            blueprint['join_paths'] = self._generate_join_paths(blueprint['tables'])

        logger.info(f"Blueprint generated for: {question[:50]}... (confidence={blueprint['confidence']})")
        return blueprint

    def _generate_join_paths(self, tables: List[str]) -> List[str]:
        if not self._relationships:
            self.discover_relationships()

        join_paths = []

        for i in range(len(tables) - 1):
            table_a = tables[i]
            table_b = tables[i + 1]

            key = (table_a, table_b)
            if key in self._relationships:
                for rel in self._relationships[key]:
                    join_paths.append(rel['join_clause'])

        return join_paths

    def detect_gaps(self) -> List[Dict[str, Any]]:
        gaps = []

        cursor = self.kg_conn.cursor()

        if self._relationships:
            for (table_a, table_b), rels in self._relationships.items():
                for rel in rels:
                    cursor.execute('''
                        SELECT COUNT(*) FROM query_sql_mappings
                        WHERE tables_used LIKE ?
                    ''', (f'%{table_a}%',))
                    count_a = cursor.fetchone()[0]

                    cursor.execute('''
                        SELECT COUNT(*) FROM query_sql_mappings
                        WHERE tables_used LIKE ?
                    ''', (f'%{table_b}%',))
                    count_b = cursor.fetchone()[0]

                    if count_a > 0 and count_b > 0:
                        cursor.execute('''
                            SELECT COUNT(*) FROM query_sql_mappings
                            WHERE tables_used LIKE ? AND tables_used LIKE ?
                        ''', (f'%{table_a}%', f'%{table_b}%'))
                        count_join = cursor.fetchone()[0]

                        if count_join == 0:
                            gap = {
                                'gap_type': 'uncovered_relationship',
                                'table_a': table_a,
                                'table_b': table_b,
                                'column_a': rel['column_a'],
                                'column_b': rel['column_b'],
                                'priority_score': 0.8 if count_a > 10 and count_b > 10 else 0.4,
                                'description': f'Relationship {table_a}-{table_b} on {rel["column_a"]} never queried'
                            }
                            gaps.append(gap)

        cursor.execute('''
            SELECT pattern_id, pattern_type, failure_count, success_count
            FROM learned_patterns
            WHERE failure_count > 0
            ORDER BY failure_count DESC
            LIMIT 10
        ''')

        for pattern_row in cursor.fetchall():
            pattern_id, pattern_type, failure_count, success_count = pattern_row
            total = failure_count + success_count
            failure_rate = failure_count / total if total > 0 else 0

            if failure_rate > 0.3:
                gap = {
                    'gap_type': 'low_confidence_pattern',
                    'pattern_id': pattern_id,
                    'priority_score': min(failure_rate, 1.0),
                    'description': f'Pattern {pattern_id[:8]} has {failure_rate:.1%} failure rate'
                }
                gaps.append(gap)

        for gap in gaps:
            self._persist_gap(gap)

        logger.info(f"Detected {len(gaps)} knowledge gaps")
        return gaps

    def _persist_gap(self, gap: Dict[str, Any]):
        cursor = self.kg_conn.cursor()

        gap_id = hashlib.sha256(
            f"{gap['gap_type']}_{gap.get('table_a', '')}_{gap.get('table_b', '')}".encode()
        ).hexdigest()

        now = datetime.utcnow().isoformat()

        cursor.execute('SELECT gap_id, encounter_count FROM knowledge_gaps WHERE gap_id = ?', (gap_id,))
        existing = cursor.fetchone()

        if existing:
            new_count = existing[1] + 1
            cursor.execute('''
                UPDATE knowledge_gaps
                SET encounter_count = ?, last_encountered = ?, priority_score = ?
                WHERE gap_id = ?
            ''', (new_count, now, gap['priority_score'], gap_id))
        else:
            cursor.execute('''
                INSERT INTO knowledge_gaps
                (gap_id, gap_type, table_a, table_b, column_a, column_b,
                 description, priority_score, last_encountered, encounter_count, detected_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                gap_id,
                gap['gap_type'],
                gap.get('table_a', ''),
                gap.get('table_b', ''),
                gap.get('column_a', ''),
                gap.get('column_b', ''),
                gap.get('description', ''),
                gap.get('priority_score', 0.5),
                now,
                1,
                now
            ))

        self.kg_conn.commit()

    def compute_metrics(self) -> Dict[str, float]:
        cursor = self.kg_conn.cursor()
        metrics = {}
        now = datetime.utcnow().isoformat()

        metrics['total_tables'] = len(self._schema_cache) if self._schema_cache else 0
        metrics['total_columns'] = sum(
            len(info.get('columns', []))
            for info in self._schema_cache.values()
        ) if self._schema_cache else 0
        metrics['total_relationships'] = len(self._relationships) if self._relationships else 0

        if self._column_semantics:
            total_columns = len(self._column_semantics)
            cursor.execute('''
                SELECT COUNT(DISTINCT cm.semantic_id)
                FROM column_semantics cm
                INNER JOIN query_sql_mappings qm ON qm.tables_used LIKE '%' || cm.table_name || '%'
            ''')
            covered_columns = cursor.fetchone()[0]
            coverage = (covered_columns / total_columns * 100) if total_columns > 0 else 0
            metrics['knowledge_coverage'] = coverage

        cursor.execute('SELECT COUNT(DISTINCT pattern_type) FROM learned_patterns')
        metrics['pattern_diversity'] = cursor.fetchone()[0] or 0

        cursor.execute('SELECT COUNT(*) FROM learned_patterns')
        metrics['total_patterns'] = cursor.fetchone()[0] or 0

        cursor.execute('SELECT COUNT(*) FROM knowledge_gaps WHERE resolved_at IS NULL')
        metrics['gap_count'] = cursor.fetchone()[0] or 0

        cursor.execute('SELECT COUNT(*) FROM query_sql_mappings')
        metrics['query_volume'] = cursor.fetchone()[0] or 0

        cursor.execute('''
            SELECT COUNT(*) FILTER (WHERE success = 1), COUNT(*)
            FROM (SELECT success FROM query_sql_mappings ORDER BY executed_at DESC LIMIT 50)
        ''')
        result = cursor.fetchone()
        if result and result[1] > 0:
            metrics['recent_accuracy'] = (result[0] / result[1] * 100)
        else:
            metrics['recent_accuracy'] = 0

        cursor.execute('SELECT COUNT(*) FILTER (WHERE success = 1), COUNT(*) FROM query_sql_mappings')
        result = cursor.fetchone()
        if result and result[1] > 0:
            metrics['overall_accuracy'] = (result[0] / result[1] * 100)
        else:
            metrics['overall_accuracy'] = 0

        cursor.execute('SELECT AVG(execution_time) FROM query_sql_mappings WHERE success = 1')
        avg_time = cursor.fetchone()[0]
        metrics['avg_execution_time'] = avg_time if avg_time else 0

        cursor.execute('SELECT MAX(learned_at) FROM learned_patterns')
        last_learn = cursor.fetchone()[0]
        if last_learn:
            last_learn_dt = datetime.fromisoformat(last_learn)
            days_old = (datetime.utcnow() - last_learn_dt).days
            metrics['staleness_days'] = days_old
        else:
            metrics['staleness_days'] = 999

        for metric_name, value in metrics.items():
            metric_id = hashlib.sha256(f"{metric_name}_{now}".encode()).hexdigest()
            cursor.execute('''
                INSERT INTO metrics (metric_id, metric_name, metric_value, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (metric_id, metric_name, value, now))

        self.kg_conn.commit()

        logger.info(f"Metrics computed: {len(metrics)} metrics")
        return metrics

    def should_retrain(self, accuracy_threshold: float = 0.75,
                      gap_threshold: int = 5,
                      staleness_days: int = 7) -> Tuple[bool, Dict[str, Any]]:
        metrics = self.compute_metrics()
        diagnosis = {}
        should_retrain = False

        recent_acc = metrics.get('recent_accuracy', 0)
        if recent_acc < accuracy_threshold * 100:
            diagnosis['accuracy_low'] = recent_acc
            should_retrain = True

        gaps = metrics.get('gap_count', 0)
        if gaps > gap_threshold:
            diagnosis['too_many_gaps'] = gaps
            should_retrain = True

        staleness = metrics.get('staleness_days', 999)
        if staleness > staleness_days:
            diagnosis['knowledge_stale'] = staleness
            should_retrain = True

        if should_retrain:
            logger.warning(f"Retrain triggered: {diagnosis}")
        else:
            logger.info("Knowledge graph is healthy, no retrain needed")

        return should_retrain, diagnosis

    def get_retrain_priorities(self) -> List[Dict[str, Any]]:
        cursor = self.kg_conn.cursor()

        cursor.execute('''
            SELECT gap_id, gap_type, table_a, table_b, column_a, column_b,
                   description, priority_score, encounter_count
            FROM knowledge_gaps
            WHERE resolved_at IS NULL
            ORDER BY priority_score DESC, encounter_count DESC
            LIMIT 20
        ''')

        priorities = []
        for row in cursor.fetchall():
            priority = {
                'gap_id': row[0],
                'gap_type': row[1],
                'table_a': row[2],
                'table_b': row[3],
                'column_a': row[4],
                'column_b': row[5],
                'description': row[6],
                'priority_score': row[7],
                'encounter_count': row[8]
            }
            priorities.append(priority)

        return priorities

    def get_learning_summary(self) -> Dict[str, Any]:
        metrics = self.compute_metrics()
        gaps = self.detect_gaps()
        should_retrain, diagnosis = self.should_retrain()

        cursor = self.kg_conn.cursor()

        cursor.execute('''
            SELECT pattern_id, pattern_type, success_rate, last_used
            FROM learned_patterns
            ORDER BY last_used DESC
            LIMIT 5
        ''')
        recent_patterns = [
            {'pattern_id': p[0][:8], 'type': p[1], 'success_rate': p[2], 'last_used': p[3]}
            for p in cursor.fetchall()
        ]

        summary = {
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': metrics,
            'health': {
                'should_retrain': should_retrain,
                'diagnosis': diagnosis
            },
            'patterns': {
                'total_learned': metrics.get('total_patterns', 0),
                'diversity': metrics.get('pattern_diversity', 0),
                'recent': recent_patterns
            },
            'gaps': {
                'total_unresolved': len(gaps),
                'top_priorities': [
                    {
                        'description': g['description'],
                        'priority': g.get('priority_score', 0),
                        'type': g['gap_type']
                    }
                    for g in gaps[:5]
                ]
            }
        }

        return summary

    def close(self):
        if self.conn:
            self.conn.close()
        if self.kg_conn:
            self.kg_conn.close()
        logger.info("Knowledge graph connections closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main():
    import os

    db_path = '/sessions/great-gallant-allen/mnt/chatbot/mtp_demo/data/healthcare_demo.db'

    kg = KnowledgeGraph(db_path)

    try:
        logger.info("=" * 80)
        logger.info("SCHEMA DISCOVERY")
        logger.info("=" * 80)
        schema = kg.discover_schema()
        logger.info(f"Discovered {len(schema)} tables")

        logger.info("\n" + "=" * 80)
        logger.info("RELATIONSHIP DISCOVERY")
        logger.info("=" * 80)
        relationships = kg.discover_relationships()
        logger.info(f"Discovered {len(relationships)} relationship pairs")

        logger.info("\n" + "=" * 80)
        logger.info("COLUMN SEMANTICS DISCOVERY")
        logger.info("=" * 80)
        semantics = kg.discover_column_semantics()
        logger.info(f"Analyzed {len(semantics)} column semantics")

        logger.info("\n" + "=" * 80)
        logger.info("PATTERN LEARNING")
        logger.info("=" * 80)

        test_queries = [
            (
                "How many claims were paid for each member?",
                "SELECT MEMBER_ID, COUNT(*) as claim_count FROM CLAIMS WHERE CLAIM_STATUS='PAID' GROUP BY MEMBER_ID"
            ),
            (
                "What is the total allowed amount by region?",
                "SELECT KP_REGION, SUM(CAST(ALLOWED_AMOUNT AS FLOAT)) as total FROM CLAIMS GROUP BY KP_REGION"
            ),
            (
                "Show members with encounters and their diagnoses",
                "SELECT m.MEMBER_ID, e.ENCOUNTER_ID, d.ICD10_CODE FROM MEMBERS m JOIN ENCOUNTERS e ON m.MEMBER_ID = e.MEMBER_ID JOIN DIAGNOSES d ON e.ENCOUNTER_ID = d.ENCOUNTER_ID"
            )
        ]

        for question, sql in test_queries:
            pattern_id = kg.learn_pattern(question, sql, success=True, execution_time=0.5)
            logger.info(f"Learned pattern: {pattern_id[:8]}")

        logger.info("\n" + "=" * 80)
        logger.info("PATTERN MATCHING")
        logger.info("=" * 80)
        test_question = "Find the count of paid claims grouped by member"
        pattern = kg.find_closest_pattern(test_question)
        if pattern:
            logger.info(f"Found pattern with score: {pattern['success_rate']}")

        logger.info("\n" + "=" * 80)
        logger.info("BLUEPRINT GENERATION")
        logger.info("=" * 80)
        blueprint = kg.generate_query_blueprint(test_question)
        logger.info(f"Blueprint tables: {blueprint['tables']}")
        logger.info(f"Blueprint confidence: {blueprint['confidence']}")

        logger.info("\n" + "=" * 80)
        logger.info("GAP DETECTION")
        logger.info("=" * 80)
        gaps = kg.detect_gaps()
        logger.info(f"Detected {len(gaps)} gaps")

        logger.info("\n" + "=" * 80)
        logger.info("METRICS")
        logger.info("=" * 80)
        metrics = kg.compute_metrics()
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.2f}" if isinstance(value, float) else f"{metric}: {value}")

        logger.info("\n" + "=" * 80)
        logger.info("RETRAIN CHECK")
        logger.info("=" * 80)
        should_retrain, diagnosis = kg.should_retrain()
        logger.info(f"Should retrain: {should_retrain}")
        logger.info(f"Diagnosis: {diagnosis}")

        logger.info("\n" + "=" * 80)
        logger.info("LEARNING SUMMARY")
        logger.info("=" * 80)
        summary = kg.get_learning_summary()
        logger.info(json.dumps(summary, indent=2, default=str))

    finally:
        kg.close()


if __name__ == '__main__':
    main()
