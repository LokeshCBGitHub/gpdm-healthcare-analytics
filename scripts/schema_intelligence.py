import re
import os
import math
import json
import time
import hashlib
import sqlite3
import logging
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, field, asdict
from tokenizer_utils import tokenize_query_list

logger = logging.getLogger('gpdm.schema_intel')


@dataclass
class DiscoveredConcept:
    term: str
    column: str
    value: str
    table: str
    frequency: int = 0
    total_rows: int = 0
    proportion: float = 0.0


class ValueConceptDiscovery:

    _MORPHS = {
        'ed': [('ed', ''), ('ed', 'al'), ('ed', 'ion')],
        'ing': [('ing', ''), ('ing', 'e')],
        'tion': [('tion', 't'), ('tion', 'te')],
        'ment': [('ment', ''), ('ment', 'e')],
    }

    def __init__(self, db_path: str, schema_learner):
        self.db_path = db_path
        self.learner = schema_learner
        self._concepts: Dict[str, DiscoveredConcept] = {}

    def discover(self) -> Dict[str, DiscoveredConcept]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for tbl_name, profiles in self.learner.tables.items():
            total_rows = self.learner.table_row_counts.get(tbl_name, 0)

            for p in profiles:
                if not p.is_categorical:
                    continue

                if p.name.endswith('_ID') or p.name in ('NPI', 'MRN'):
                    continue
                if getattr(p, 'distinct_count', 0) > 100:
                    continue

                try:
                    cursor.execute(
                        f"SELECT {p.name}, COUNT(*) as cnt FROM {tbl_name} "
                        f"WHERE {p.name} IS NOT NULL AND {p.name} != '' "
                        f"GROUP BY {p.name} ORDER BY cnt DESC LIMIT 50"
                    )
                    values = cursor.fetchall()
                except Exception:
                    continue

                col_lower = p.name.lower()
                is_flag_col = (col_lower.startswith('is_') or col_lower.startswith('has_')
                               or col_lower.startswith('flag_'))
                true_values = {'y', 'yes', 'true', '1', 't'}

                for val, count in values:
                    val_str = str(val).strip()
                    if not val_str or len(val_str) > 100:
                        continue

                    terms = self._generate_terms(val_str)

                    if is_flag_col and val_str.lower() in true_values:
                        flag_name = col_lower
                        for prefix in ('is_', 'has_', 'flag_'):
                            if flag_name.startswith(prefix):
                                flag_name = flag_name[len(prefix):]
                                break
                        terms.append(flag_name)
                        for w in flag_name.split('_'):
                            if len(w) >= 3:
                                terms.append(w)

                    proportion = count / total_rows if total_rows > 0 else 0

                    for term in terms:
                        if len(term) < 2:
                            continue
                        existing = self._concepts.get(term)
                        if existing and existing.frequency >= count:
                            continue
                        self._concepts[term] = DiscoveredConcept(
                            term=term,
                            column=p.name,
                            value=val_str,
                            table=tbl_name,
                            frequency=count,
                            total_rows=total_rows,
                            proportion=proportion,
                        )

        conn.close()
        logger.info("ValueConceptDiscovery: discovered %d concepts from %d tables",
                     len(self._concepts), len(self.learner.tables))
        return self._concepts

    def _generate_terms(self, value: str) -> List[str]:
        terms = []
        val_lower = value.lower().strip()
        terms.append(val_lower)

        _common_english = {
            'show', 'claim', 'claims', 'type', 'code', 'plan', 'date',
            'name', 'rate', 'status', 'visit', 'care', 'check', 'test',
            'level', 'class', 'group', 'order', 'result', 'record',
            'general', 'other', 'new', 'old', 'high', 'low', 'total',
            'office', 'follow', 'home', 'initial', 'standard',
            'primary', 'secondary', 'complex', 'simple', 'basic',
            'condition', 'medical', 'health', 'service', 'reason',
            'note', 'full', 'partial', 'complete', 'pending',
            'active', 'closed', 'open', 'review', 'need', 'risk',
            'major', 'minor', 'moderate', 'severe', 'normal',
        }
        words = val_lower.replace('_', ' ').split()
        if len(words) > 1:
            for w in words:
                if len(w) > 2 and w not in _common_english:
                    terms.append(w)
            for i in range(2, len(words)):
                terms.append(' '.join(words[:i]))

        for w in words:
            for suffix, replacements in self._MORPHS.items():
                if w.endswith(suffix):
                    for old_suffix, new_suffix in replacements:
                        variant = w[:-len(old_suffix)] + new_suffix
                        if len(variant) > 2:
                            terms.append(variant)

        if val_lower.endswith('ed'):
            stem = val_lower[:-2]
            terms.append(stem + 'al')
            terms.append(stem)
            if stem.endswith('i'):
                terms.append(stem[:-1] + 'y')
            if stem.endswith('t'):
                terms.append(stem + 'ion')
                terms.append(stem + 'ting')

        if val_lower.endswith('ing'):
            stem = val_lower[:-3]
            terms.append(stem)
            terms.append(stem + 'e')

        return list(set(t for t in terms if len(t) >= 2))

    def to_domain_concepts(self) -> Dict[str, Dict]:
        _blocked_terms = {
            'show', 'claim', 'claims', 'type', 'code', 'plan', 'date',
            'name', 'rate', 'status', 'visit', 'care', 'check', 'test',
            'level', 'class', 'group', 'order', 'result', 'record',
            'general', 'other', 'new', 'old', 'high', 'low', 'total',
            'office', 'follow', 'home', 'initial', 'standard', 'month',
            'primary', 'secondary', 'complex', 'simple', 'basic', 'year',
            'condition', 'medical', 'health', 'service', 'reason',
            'note', 'full', 'partial', 'complete', 'pending', 'time',
            'active', 'closed', 'open', 'review', 'need', 'risk',
            'major', 'minor', 'moderate', 'severe', 'normal',
            'average', 'many', 'most', 'volume', 'count', 'number',
            'member', 'members', 'patient', 'patients', 'provider',
            'providers', 'report', 'summary', 'trend', 'data',
        }
        result = {}
        for term, concept in self._concepts.items():
            if term.lower() in _blocked_terms:
                continue
            if len(term) <= 3 and ' ' not in term:
                continue
            result[term] = {
                'conds': [f"{concept.column} = '{concept.value}'"],
                'tables': [concept.table],
                '_discovered': True,
                '_frequency': concept.frequency,
                '_proportion': concept.proportion,
            }
        return result

    def get_rate_targets(self, table: str = None) -> Dict[str, DiscoveredConcept]:
        targets = {}
        for term, concept in self._concepts.items():
            col_lower = concept.column.lower()
            if not any(kw in col_lower for kw in ['status', 'type', 'disposition',
                                                    'result', 'outcome', 'flag', 'category']):
                continue
            if table and concept.table != table:
                continue
            targets[term] = concept
        return targets


class SemanticColumnIndex:

    def __init__(self, schema_learner, db_path: str = None):
        self.learner = schema_learner
        self.db_path = db_path
        self._idf: Dict[str, float] = {}
        self._vectors: Dict[str, Dict[str, float]] = {}
        self._column_meta: Dict[str, Tuple[str, str, Any]] = {}
        self._norms: Dict[str, float] = {}
        self._vocab: Set[str] = set()

    def build(self):
        documents = []

        for tbl_name, profiles in self.learner.tables.items():
            for p in profiles:
                col_id = f"{tbl_name}.{p.name}"
                text = self._column_to_text(tbl_name, p)
                self._column_meta[col_id] = (tbl_name, p.name, p)
                documents.append((col_id, text))

        if not documents:
            return

        N = len(documents)
        doc_freq = Counter()
        doc_tokens = {}

        for col_id, text in documents:
            tokens = self._tokenize(text)
            doc_tokens[col_id] = tokens
            for token in set(tokens):
                doc_freq[token] += 1

        self._idf = {
            token: math.log((N + 1) / (df + 1)) + 1.0
            for token, df in doc_freq.items()
        }
        self._vocab = set(self._idf.keys())

        for col_id, tokens in doc_tokens.items():
            tf = Counter(tokens)
            max_tf = max(tf.values()) if tf else 1
            vector = {}
            for token, count in tf.items():
                tf_score = 0.5 + 0.5 * (count / max_tf)
                vector[token] = tf_score * self._idf.get(token, 1.0)
            self._vectors[col_id] = vector
            self._norms[col_id] = math.sqrt(sum(v*v for v in vector.values()))

        logger.info("SemanticColumnIndex: indexed %d columns, %d vocab tokens",
                     len(self._vectors), len(self._vocab))

    def _column_to_text(self, table: str, profile) -> str:
        parts = []

        tbl_lower = table.lower()
        parts.append(tbl_lower)
        if tbl_lower.endswith('s'):
            parts.append(tbl_lower[:-1])
        else:
            parts.append(tbl_lower + 's')

        col_words = profile.name.lower().replace('_', ' ').split()
        parts.extend(col_words)

        if profile.is_numeric:
            parts.extend(['numeric', 'number', 'amount', 'quantity', 'measure', 'metric'])
            col_lower = profile.name.lower()
            if any(kw in col_lower for kw in ['amount', 'cost', 'paid', 'billed',
                                                'allowed', 'price', 'charge']):
                parts.extend(['cost', 'spend', 'spending', 'expense', 'financial',
                              'dollar', 'money', 'payment', 'revenue', 'price'])
            if any(kw in col_lower for kw in ['score', 'rate', 'pct', 'percent']):
                parts.extend(['score', 'rating', 'percentage', 'rate'])
            if any(kw in col_lower for kw in ['count', 'num', 'quantity']):
                parts.extend(['count', 'total', 'number', 'volume', 'how many'])
            if any(kw in col_lower for kw in ['days', 'duration', 'length', 'stay']):
                parts.extend(['duration', 'length', 'stay', 'days', 'time', 'los'])

        if profile.is_date:
            parts.extend(['date', 'time', 'when', 'period', 'temporal'])

        if profile.is_categorical:
            parts.extend(['category', 'type', 'group', 'classification'])

        if hasattr(profile, 'semantic_tags') and profile.semantic_tags:
            parts.extend(profile.semantic_tags)

        if hasattr(profile, 'sample_values') and profile.sample_values:
            for sv in profile.sample_values[:10]:
                sv_str = str(sv).lower()
                parts.extend(sv_str.replace('_', ' ').split())

        return ' '.join(parts)

    def _tokenize(self, text: str) -> List[str]:
        words = re.findall(r'[a-z][a-z0-9]*', text.lower())
        tokens = list(words)
        for i in range(len(words) - 1):
            tokens.append(f"{words[i]}_{words[i+1]}")
        return tokens

    def search(self, query: str, top_k: int = 10,
               column_type: str = None) -> List[Tuple[str, str, str, float]]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        qtf = Counter(query_tokens)
        max_qtf = max(qtf.values()) if qtf else 1
        q_vector = {}
        for token, count in qtf.items():
            if token in self._idf:
                tf_score = 0.5 + 0.5 * (count / max_qtf)
                q_vector[token] = tf_score * self._idf[token]

        if not q_vector:
            return []

        q_norm = math.sqrt(sum(v*v for v in q_vector.values()))
        if q_norm == 0:
            return []

        scores = []
        for col_id, col_vector in self._vectors.items():
            if column_type:
                _, _, profile = self._column_meta[col_id]
                if column_type == 'numeric' and not profile.is_numeric:
                    continue
                if column_type == 'categorical' and not profile.is_categorical:
                    continue
                if column_type == 'date' and not profile.is_date:
                    continue

            dot = sum(q_vector.get(t, 0) * v for t, v in col_vector.items()
                      if t in q_vector)

            col_norm = self._norms.get(col_id, 0)
            if col_norm > 0:
                sim = dot / (q_norm * col_norm)
                if sim > 0.01:
                    tbl, col_name, _ = self._column_meta[col_id]
                    scores.append((tbl, col_name, col_id, sim))

        scores.sort(key=lambda x: x[3], reverse=True)
        return scores[:top_k]

    def find_metric_column(self, concept: str, preferred_table: str = None) -> Optional[Tuple[str, str]]:
        results = self.search(concept, top_k=5, column_type='numeric')
        if not results:
            return None

        if preferred_table:
            for tbl, col, _, score in results:
                if tbl == preferred_table:
                    return (tbl, col)

        return (results[0][0], results[0][1])

    def find_dimension_column(self, concept: str, preferred_table: str = None) -> Optional[Tuple[str, str]]:
        results = self.search(concept, top_k=5, column_type='categorical')
        if not results:
            return None

        if preferred_table:
            for tbl, col, _, score in results:
                if tbl == preferred_table:
                    return (tbl, col)

        return (results[0][0], results[0][1])

    def find_date_column(self, preferred_table: str = None) -> Optional[Tuple[str, str]]:
        results = self.search('service date admission date enrollment',
                              top_k=5, column_type='date')
        if not results:
            return None
        if preferred_table:
            for tbl, col, _, score in results:
                if tbl == preferred_table:
                    return (tbl, col)
        return (results[0][0], results[0][1])

    def to_synonyms(self, threshold: float = 0.12) -> Dict[str, List[str]]:
        synonyms = defaultdict(list)

        col_derived_terms = set()
        for tbl_name, profiles in self.learner.tables.items():
            tbl_lower = tbl_name.lower()
            col_derived_terms.add(tbl_lower)
            if tbl_lower.endswith('s'):
                col_derived_terms.add(tbl_lower[:-1])
            else:
                col_derived_terms.add(tbl_lower + 's')

            for p in profiles:
                words = p.name.lower().replace('_', ' ').split()
                for w in words:
                    if len(w) > 2 and w not in ('the', 'and', 'for', 'not'):
                        col_derived_terms.add(w)

                if len(words) >= 2:
                    col_derived_terms.add(' '.join(words))

        _type_expansions = {
            'amount': ['cost', 'spend', 'spending', 'expense', 'charge', 'payment',
                       'revenue', 'price', 'dollar', 'money', 'financial'],
            'paid': ['payment', 'reimbursement', 'paid amount', 'paid cost'],
            'billed': ['billed cost', 'billed charge', 'bill', 'billing'],
            'allowed': ['allowed cost', 'allowable', 'contracted'],
            'date': ['when', 'time', 'period', 'temporal', 'day'],
            'service': ['service date', 'service time', 'dos'],
            'admit': ['admission', 'admit date', 'admitted'],
            'discharge': ['discharge date', 'discharged', 'left'],
            'enrollment': ['enrollment date', 'enrolled', 'start date'],
            'disenrollment': ['disenrollment date', 'end date', 'terminated'],
            'fill': ['fill date', 'dispensed', 'filled'],
            'member': ['patient', 'person', 'subscriber', 'beneficiary', 'enrollee'],
            'provider': ['doctor', 'physician', 'clinician', 'practitioner'],
            'npi': ['provider', 'doctor', 'physician', 'npi number'],
            'diagnosis': ['condition', 'disease', 'dx', 'icd'],
            'procedure': ['cpt', 'hcpcs', 'service', 'treatment'],
            'prescription': ['medication', 'drug', 'rx', 'pharmacy', 'medicine'],
            'referral': ['authorization', 'auth', 'referred'],
            'appointment': ['visit', 'schedule', 'booked', 'scheduled'],
            'region': ['area', 'location', 'geography', 'market', 'territory'],
            'specialty': ['speciality', 'department', 'discipline'],
            'status': ['state', 'outcome', 'result', 'disposition'],
            'type': ['kind', 'category', 'classification', 'class'],
            'plan': ['insurance', 'coverage', 'payer', 'insurer'],
            'gender': ['sex'],
            'age': ['years old', 'age group', 'age bracket'],
            'length': ['duration', 'stay', 'los', 'days'],
            'score': ['rating', 'metric', 'index'],
            'risk': ['acuity', 'severity', 'complexity'],
        }

        for col_word in list(col_derived_terms):
            if col_word in _type_expansions:
                for alias in _type_expansions[col_word]:
                    col_derived_terms.add(alias)

        for term in col_derived_terms:
            if len(term) < 2:
                continue
            results = self.search(term, top_k=3)
            for tbl, col, _, score in results:
                if score >= threshold:
                    if col not in synonyms[term]:
                        synonyms[term].append(col)

        logger.info("SemanticColumnIndex: generated %d synonym entries (auto-derived)",
                     len(synonyms))
        return dict(synonyms)


class SQLSelfHealer:

    MAX_RETRIES = 3

    def __init__(self, schema_learner, db_path: str):
        self.learner = schema_learner
        self.db_path = db_path
        self._col_to_tables: Dict[str, List[str]] = defaultdict(list)
        for tbl, profiles in self.learner.tables.items():
            for p in profiles:
                self._col_to_tables[p.name].append(tbl)

    def execute_with_healing(self, sql: str, context: Dict = None) -> Dict[str, Any]:
        context = context or {}
        primary_table = context.get('primary_table', '')
        attempts = 0
        current_sql = sql
        heal_log = []

        while attempts < self.MAX_RETRIES:
            attempts += 1
            try:
                conn = sqlite3.connect(self.db_path)
                import time as _time
                _exec_start = _time.time()
                def _timeout_check():
                    return 1 if _time.time() - _exec_start > 15 else 0
                conn.set_progress_handler(_timeout_check, 10000)

                cursor = conn.cursor()
                cursor.execute(current_sql)
                rows = cursor.fetchall()
                columns = [d[0] for d in cursor.description] if cursor.description else []
                conn.close()

                return {
                    'sql': current_sql,
                    'rows': rows,
                    'columns': columns,
                    'healed': attempts > 1,
                    'attempts': attempts,
                    'heal_log': heal_log,
                }

            except sqlite3.OperationalError as e:
                error_msg = str(e)
                try:
                    conn.close()
                except Exception:
                    pass
                if 'interrupt' in error_msg.lower():
                    return {
                        'sql': current_sql,
                        'rows': [],
                        'columns': [],
                        'error': 'Query timed out (likely expensive JOIN)',
                        'healed': False,
                        'attempts': attempts,
                        'heal_log': heal_log + ['Timed out after 15s'],
                    }

            except Exception as e:
                error_msg = str(e)
                conn.close()

                fixed_sql = self._diagnose_and_fix(current_sql, error_msg, context)
                if fixed_sql and fixed_sql != current_sql:
                    heal_log.append({
                        'attempt': attempts,
                        'error': error_msg,
                        'fix_applied': True,
                    })
                    logger.info("SQLSelfHealer: healed '%s' → retrying (attempt %d)",
                                error_msg[:60], attempts + 1)
                    current_sql = fixed_sql
                else:
                    heal_log.append({
                        'attempt': attempts,
                        'error': error_msg,
                        'fix_applied': False,
                    })
                    return {
                        'sql': current_sql,
                        'rows': [],
                        'columns': [],
                        'healed': False,
                        'attempts': attempts,
                        'error': error_msg,
                        'heal_log': heal_log,
                    }

        return {
            'sql': current_sql, 'rows': [], 'columns': [],
            'healed': False, 'attempts': attempts,
            'error': 'Max retries exceeded',
            'heal_log': heal_log,
        }

    def _diagnose_and_fix(self, sql: str, error: str, context: Dict) -> Optional[str]:
        error_lower = error.lower()

        m = re.search(r'ambiguous column name:\s*(\w+)', error, re.I)
        if m:
            return self._fix_ambiguous_column(sql, m.group(1), context)

        m = re.search(r'no such column:\s*(\S+)', error, re.I)
        if m:
            return self._fix_missing_column(sql, m.group(1), context)

        m = re.search(r'no such table:\s*(\w+)', error, re.I)
        if m:
            return self._fix_missing_table(sql, m.group(1))

        if 'group by' in error_lower and 'having' in error_lower:
            return self._fix_missing_group_by(sql, context)

        if 'aggregate' in error_lower and 'group by' in error_lower:
            return self._fix_aggregate_in_group_by(sql)

        return None

    def _fix_ambiguous_column(self, sql: str, col_name: str, context: Dict) -> Optional[str]:
        aliases = self._extract_aliases(sql)
        if not aliases:
            return None

        tables_with_col = self._col_to_tables.get(col_name, [])
        if not tables_with_col:
            return None

        primary_table = context.get('primary_table', '')
        best_table = None
        for tbl in tables_with_col:
            if tbl == primary_table:
                best_table = tbl
                break
            if tbl in aliases:
                best_table = tbl

        if not best_table:
            best_table = tables_with_col[0]

        alias = aliases.get(best_table, best_table[0])

        fixed = self._qualify_bare_column(sql, col_name, alias)
        return fixed if fixed != sql else None

    def _fix_missing_column(self, sql: str, col_ref: str, context: Dict) -> Optional[str]:
        if '.' in col_ref:
            alias, col_name = col_ref.split('.', 1)
            tables_with_col = self._col_to_tables.get(col_name, [])
            aliases = self._extract_aliases(sql)

            for tbl in tables_with_col:
                correct_alias = aliases.get(tbl)
                if correct_alias and correct_alias != alias:
                    fixed = sql.replace(f"{alias}.{col_name}", f"{correct_alias}.{col_name}")
                    return fixed

            if tables_with_col:
                needed_table = tables_with_col[0]
                if needed_table not in aliases:
                    return self._add_join(sql, needed_table, aliases, context)

        return None

    def _fix_missing_table(self, sql: str, table_name: str) -> Optional[str]:
        for tbl in self.learner.tables:
            if tbl.lower().startswith(table_name.lower()[:3]):
                return sql.replace(table_name, tbl)
        return None

    def _fix_missing_group_by(self, sql: str, context: Dict) -> Optional[str]:
        select_match = re.search(r'SELECT\s+(.+?)\s+FROM', sql, re.I | re.DOTALL)
        if not select_match:
            return None

        select_clause = select_match.group(1)
        parts = [p.strip() for p in select_clause.split(',')]
        group_cols = []
        for part in parts:
            if re.search(r'\b(COUNT|SUM|AVG|MIN|MAX|ROUND)\s*\(', part, re.I):
                continue
            alias_match = re.search(r'\bas\s+(\w+)\s*$', part, re.I)
            if alias_match:
                group_cols.append(alias_match.group(1))
            else:
                group_cols.append(part.split('.')[-1].strip())

        if not group_cols:
            return None

        having_pos = re.search(r'\bHAVING\b', sql, re.I)
        if having_pos:
            group_clause = f" GROUP BY {', '.join(group_cols)} "
            return sql[:having_pos.start()] + group_clause + sql[having_pos.start():]

        return None

    def _fix_aggregate_in_group_by(self, sql: str) -> Optional[str]:
        group_match = re.search(r'GROUP BY\s+(.+?)(?:\s+HAVING|\s+ORDER|\s+LIMIT|;|$)',
                                sql, re.I | re.DOTALL)
        if not group_match:
            return None

        group_clause = group_match.group(1)
        select_match = re.search(r'SELECT\s+(.+?)\s+FROM', sql, re.I | re.DOTALL)
        if not select_match:
            return None

        parts = [p.strip() for p in group_clause.split(',')]
        clean_parts = []
        for part in parts:
            if not re.search(r'\b(COUNT|SUM|AVG|MIN|MAX|ROUND)\s*\(', part, re.I):
                clean_parts.append(part)

        if clean_parts:
            new_group = ', '.join(clean_parts)
            return sql[:group_match.start()] + f"GROUP BY {new_group}" + sql[group_match.end():]

        return None

    def _extract_aliases(self, sql: str) -> Dict[str, str]:
        aliases = {}
        for m in re.finditer(r'(?:FROM|JOIN)\s+(\w+)\s+(\w)\b', sql, re.I):
            table_name = m.group(1)
            alias = m.group(2)
            aliases[table_name] = alias
        return aliases

    def _qualify_bare_column(self, sql: str, col_name: str, alias: str) -> str:
        parts = re.split(r"('(?:[^'\\]|\\.)*')", sql)
        result = []
        for i, part in enumerate(parts):
            if i % 2 == 1:
                result.append(part)
            else:
                part = re.sub(
                    r'(?<!\w\.)(?<!\w)(' + re.escape(col_name) + r')(?!\w)',
                    f'{alias}.{col_name}',
                    part
                )
                result.append(part)
        return ''.join(result)

    def _add_join(self, sql: str, needed_table: str, existing_aliases: Dict,
                  context: Dict) -> Optional[str]:
        needed_cols = {p.name for p in self.learner.tables.get(needed_table, [])}

        for existing_table, alias in existing_aliases.items():
            existing_cols = {p.name for p in self.learner.tables.get(existing_table, [])}
            shared = needed_cols & existing_cols
            join_cols = sorted(c for c in shared if c.endswith('_ID') or c in ('NPI', 'MRN'))
            if join_cols:
                new_alias = needed_table[0]
                if new_alias in existing_aliases.values():
                    new_alias = needed_table[:2]
                join_col = join_cols[0]
                join_clause = f" JOIN {needed_table} {new_alias} ON {alias}.{join_col} = {new_alias}.{join_col}"

                last_join = sql.rfind(' JOIN ')
                if last_join >= 0:
                    on_end = sql.find(' WHERE', last_join)
                    if on_end < 0:
                        on_end = sql.find(' GROUP', last_join)
                    if on_end < 0:
                        on_end = sql.find(' ORDER', last_join)
                    if on_end < 0:
                        on_end = sql.find(' LIMIT', last_join)
                    if on_end >= 0:
                        return sql[:on_end] + join_clause + sql[on_end:]
                else:
                    from_match = re.search(r'FROM\s+\w+\s+\w\b', sql, re.I)
                    if from_match:
                        pos = from_match.end()
                        return sql[:pos] + join_clause + sql[pos:]

        return None


class CompoundConceptInference:

    _ABBREVIATIONS = {
        'er': 'emergency', 'ed': 'emergency', 'ip': 'inpatient',
        'op': 'outpatient', 'icu': 'intensive', 'snf': 'skilled nursing',
        'rx': 'prescription', 'dx': 'diagnosis', 'hx': 'history',
        'tx': 'treatment', 'fx': 'fracture', 'sx': 'surgery',
        'pt': 'patient', 'dr': 'doctor', 'md': 'doctor',
        'chf': 'heart failure', 'copd': 'pulmonary', 'mi': 'myocardial',
        'afib': 'fibrillation', 'dm': 'diabetes', 'htn': 'hypertension',
        'uti': 'urinary tract', 'dvt': 'deep vein', 'pe': 'pulmonary embolism',
        'aki': 'kidney injury', 'ckd': 'kidney disease',
        'pcp': 'primary care', 'ama': 'against medical',
    }

    _PHRASE_PATTERNS = [
        (r'^active\s+(\w+)', '_status_active', "STATUS-like column = 'ACTIVE' or similar"),
        (r'^(?:high|expensive|costly)\s+(\w+)', '_threshold_high', 'numeric column > P75'),
        (r'^recent\s+(\w+)', '_temporal_recent', 'date column within last 6 months'),
        (r'^new\s+(\w+)', '_temporal_new', 'enrollment date within last 3 months'),
        (r'^readmiss', '_temporal_readmission', 'temporal re-encounter pattern'),
    ]

    def __init__(self, db_path: str, schema_learner, value_concepts: Dict[str, DiscoveredConcept]):
        self.db_path = db_path
        self.learner = schema_learner
        self.value_concepts = value_concepts
        self._compounds: Dict[str, Dict] = {}

    def discover(self) -> Dict[str, Dict]:
        self._discover_abbreviation_mappings()
        self._discover_multi_value_concepts()
        self._discover_statistical_thresholds()
        self._discover_temporal_patterns()

        logger.info("CompoundConceptInference: discovered %d compound concepts",
                     len(self._compounds))
        return self._compounds

    def _discover_abbreviation_mappings(self):
        for abbrev, expansion in self._ABBREVIATIONS.items():
            for term, concept in self.value_concepts.items():
                if (expansion in term or term.startswith(expansion[:4])):
                    self._compounds[abbrev] = {
                        'conds': [f"{concept.column} = '{concept.value}'"],
                        'tables': [concept.table],
                        '_discovered': True,
                        '_compound': True,
                        '_source': f'abbreviation:{abbrev}→{expansion}→{concept.value}',
                    }
                    break

    def _discover_multi_value_concepts(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for tbl_name, profiles in self.learner.tables.items():
            for p in profiles:
                if not p.is_categorical or p.name.endswith('_ID'):
                    continue
                if getattr(p, 'distinct_count', 0) > 50:
                    continue

                try:
                    cursor.execute(
                        f"SELECT DISTINCT {p.name} FROM {tbl_name} "
                        f"WHERE {p.name} IS NOT NULL LIMIT 30"
                    )
                    values = [r[0] for r in cursor.fetchall()]
                except Exception:
                    continue

                val_groups = self._group_related_values(values)
                for group_name, group_vals in val_groups.items():
                    if len(group_vals) >= 2:
                        in_list = ', '.join(f"'{v}'" for v in group_vals)
                        self._compounds[group_name] = {
                            'conds': [f"{p.name} IN ({in_list})"],
                            'tables': [tbl_name],
                            '_discovered': True,
                            '_compound': True,
                            '_source': f'multi_value:{p.name}→{group_vals}',
                        }

        conn.close()

    def _group_related_values(self, values: List[str]) -> Dict[str, List[str]]:
        groups = {}
        val_strs = [str(v) for v in values if v]

        prefix_groups = defaultdict(list)
        for v in val_strs:
            v_lower = v.lower().replace('_', ' ')
            if len(v_lower) >= 4:
                prefix = v_lower[:4].strip()
                prefix_groups[prefix].append(v)

        for prefix, vals in prefix_groups.items():
            if len(vals) >= 2 and len(vals) <= 10:
                group_name = f"{prefix} related"
                groups[group_name] = vals

        return groups

    def _discover_statistical_thresholds(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for tbl_name, profiles in self.learner.tables.items():
            for p in profiles:
                if not p.is_numeric or p.name.endswith('_ID'):
                    continue

                col_lower = p.name.lower()
                if not any(kw in col_lower for kw in ['amount', 'cost', 'paid', 'billed',
                                                        'charge', 'length', 'stay', 'days',
                                                        'duration', 'score', 'count']):
                    continue

                try:
                    cursor.execute(f"""
                        SELECT MIN(val), MAX(val) FROM (
                            SELECT {p.name} as val,
                                   NTILE(4) OVER (ORDER BY {p.name}) as quartile
                            FROM {tbl_name}
                            WHERE {p.name} IS NOT NULL AND {p.name} > 0
                        ) WHERE quartile = 4
                    """)
                    row = cursor.fetchone()
                    if row and row[0]:
                        p75 = row[0]
                        col_words = col_lower.replace('_', ' ')
                        for prefix in ['high', 'large', 'long', 'expensive']:
                            for word in col_words.split():
                                if len(word) > 3:
                                    key = f"{prefix} {word}"
                                    self._compounds[key] = {
                                        'conds': [f"{p.name} > {p75}"],
                                        'tables': [tbl_name],
                                        '_discovered': True,
                                        '_compound': True,
                                        '_source': f'threshold:{p.name}>P75({p75})',
                                    }
                except Exception:
                    continue

        conn.close()

    def _discover_temporal_patterns(self):
        for tbl_name, profiles in self.learner.tables.items():
            date_cols = [p.name for p in profiles if p.is_date]
            id_cols = [p.name for p in profiles if p.name.endswith('_ID')
                       and 'MEMBER' in p.name]

            if not date_cols:
                continue

            date_col = date_cols[0]
            tbl_lower = tbl_name.lower()

            entity = tbl_lower.rstrip('s')
            self._compounds[f'recent {entity}'] = {
                'conds': [f"{date_col} >= date('now', '-6 months')"],
                'tables': [tbl_name],
                '_discovered': True,
                '_compound': True,
                '_source': f'temporal:recent_{entity}',
            }

            if id_cols and entity in ('encounter', 'admission', 'visit'):
                member_col = id_cols[0]
                self._compounds[f'{entity} readmission'] = {
                    'conds': [
                        f"EXISTS (SELECT 1 FROM {tbl_name} t2 "
                        f"WHERE t2.{member_col} = {tbl_name}.{member_col} "
                        f"AND t2.{date_col} > {tbl_name}.{date_col} "
                        f"AND julianday(t2.{date_col}) - julianday({tbl_name}.{date_col}) <= 30)"
                    ],
                    'tables': [tbl_name],
                    '_discovered': True,
                    '_compound': True,
                    '_source': f'temporal:readmission_{tbl_name}',
                }


class QueryPatternLearner:

    def __init__(self):
        self._patterns: Dict[str, Dict] = {}
        self._signature_cache: Dict[str, str] = {}

    def record_success(self, question: str, sql: str, tables_used: List[str],
                        rows_returned: int = 0):
        sig = self._signature(question)
        template = self._templatize(sql)

        if sig in self._patterns:
            self._patterns[sig]['count'] += 1
            self._patterns[sig]['last_q'] = question
        else:
            self._patterns[sig] = {
                'sql': sql,
                'template': template,
                'tables': tables_used,
                'count': 1,
                'last_q': question,
                'rows': rows_returned,
            }

    def find_pattern(self, question: str, min_count: int = 1) -> Optional[Dict]:
        sig = self._signature(question)
        pattern = self._patterns.get(sig)
        if pattern and pattern['count'] >= min_count:
            return pattern
        return None

    def _signature(self, question: str) -> str:
        if question in self._signature_cache:
            return self._signature_cache[question]

        q = question.lower().strip()

        q = re.sub(r'\b20\d{2}\b', 'YEAR', q)
        q = re.sub(r'\b\d+\b', 'NUM', q)

        words = tokenize_query_list(q)

        _structural = {'show', 'what', 'which', 'how', 'many', 'much', 'is', 'are',
                       'the', 'for', 'in', 'with', 'and', 'or', 'of', 'to',
                       'from', 'between', 'each', 'per', 'all', 'total', 'top',
                       'more', 'than', 'over', 'under', 'at', 'least', 'most'}
        _agg_words = {'average', 'avg', 'sum', 'count', 'min', 'max', 'mean'}
        _rate_words = {'rate', 'rates', 'percentage', 'percent', 'ratio'}
        _temporal = {'month', 'monthly', 'year', 'yearly', 'quarter', 'quarterly',
                     'week', 'weekly', 'daily', 'trend', 'time'}

        sig_parts = []
        prev_was_by = False
        for w in words:
            if w in _rate_words:
                sig_parts.append('RATE')
            elif w in _agg_words:
                sig_parts.append(f'AGG_{w}')
            elif w in _temporal:
                sig_parts.append('TEMPORAL')
            elif w == 'by':
                sig_parts.append('by')
                prev_was_by = True
            elif prev_was_by:
                sig_parts.append(w)
                prev_was_by = False
            elif w in _structural:
                sig_parts.append(w)
            else:
                sig_parts.append(w)

        sig = '_'.join(sig_parts)
        self._signature_cache[question] = sig
        return sig

    def _templatize(self, sql: str) -> str:
        template = sql
        template = re.sub(r"'[^']*'", "'?'", template)
        template = re.sub(r'\b\d+\b', '?', template)
        return template

    @property
    def pattern_count(self) -> int:
        return len(self._patterns)

    @property
    def total_reuses(self) -> int:
        return sum(p['count'] - 1 for p in self._patterns.values() if p['count'] > 1)


class SchemaIntelligence:

    def __init__(self, schema_learner, db_path: str):
        self.learner = schema_learner
        self.db_path = db_path

        self.value_discovery = ValueConceptDiscovery(db_path, schema_learner)
        self.compound_inference = None
        self.column_index = SemanticColumnIndex(schema_learner, db_path)
        self.healer = SQLSelfHealer(schema_learner, db_path)
        self.pattern_learner = QueryPatternLearner()

        self.discovered_concepts: Dict[str, DiscoveredConcept] = {}
        self.domain_concepts: Dict[str, Dict] = {}
        self.synonyms: Dict[str, List[str]] = {}
        self._learned = False

    def learn(self):
        logger.info("SchemaIntelligence: starting auto-discovery...")

        self.discovered_concepts = self.value_discovery.discover()
        self.domain_concepts = self.value_discovery.to_domain_concepts()

        self.compound_inference = CompoundConceptInference(
            self.db_path, self.learner, self.discovered_concepts)
        compound_concepts = self.compound_inference.discover()
        for k, v in compound_concepts.items():
            if k not in self.domain_concepts:
                self.domain_concepts[k] = v

        self.column_index.build()
        self.synonyms = self.column_index.to_synonyms()

        self._learned = True
        auto_single = sum(1 for v in self.domain_concepts.values() if v.get('_discovered') and not v.get('_compound'))
        auto_compound = sum(1 for v in self.domain_concepts.values() if v.get('_compound'))
        logger.info("SchemaIntelligence: %d concepts (%d single-value, %d compound), %d synonyms",
                     len(self.domain_concepts), auto_single, auto_compound, len(self.synonyms))

        self._save_cache()


    def _cache_path(self) -> str:
        data_dir = os.path.dirname(self.db_path)
        return os.path.join(data_dir, 'intelligence_cache.json')

    def _db_fingerprint(self) -> str:
        try:
            from schema_persistence import compute_db_fingerprint
            return compute_db_fingerprint(self.db_path)
        except Exception:
            return ''

    def _save_cache(self):
        try:
            cache = {
                'fingerprint': self._db_fingerprint(),
                'saved_at': time.time(),
                'domain_concepts': self.domain_concepts,
                'synonyms': self.synonyms,
                'discovered_concepts': {
                    k: asdict(v) for k, v in self.discovered_concepts.items()
                },
                'column_index': {
                    'idf': self.column_index._idf,
                    'vectors': self.column_index._vectors,
                    'norms': self.column_index._norms,
                    'vocab': list(self.column_index._vocab),
                    'column_meta': {
                        k: (v[0], v[1]) for k, v in self.column_index._column_meta.items()
                    },
                },
            }
            with open(self._cache_path(), 'w') as f:
                json.dump(cache, f)
            logger.info("Intelligence cache saved: %d concepts, %d synonyms",
                        len(self.domain_concepts), len(self.synonyms))
        except Exception as e:
            logger.warning("Failed to save intelligence cache: %s", e)

    def _load_cache(self) -> bool:
        cp = self._cache_path()
        if not os.path.exists(cp):
            return False
        try:
            with open(cp) as f:
                cache = json.load(f)

            current_fp = self._db_fingerprint()
            cached_fp = cache.get('fingerprint', '')
            if current_fp != cached_fp:
                logger.info("Intelligence cache stale (fp %s != %s), will rebuild",
                            cached_fp[:8], current_fp[:8])
                return False

            self.discovered_concepts = {
                k: DiscoveredConcept(**v)
                for k, v in cache.get('discovered_concepts', {}).items()
            }

            self.domain_concepts = cache.get('domain_concepts', {})
            self.synonyms = cache.get('synonyms', {})

            ci = cache.get('column_index', {})
            self.column_index._idf = ci.get('idf', {})
            self.column_index._vectors = ci.get('vectors', {})
            self.column_index._norms = ci.get('norms', {})
            self.column_index._vocab = set(ci.get('vocab', []))
            saved_meta = ci.get('column_meta', {})
            for col_id, (tbl, col_name) in saved_meta.items():
                profile = None
                if tbl in self.learner.tables:
                    for p in self.learner.tables[tbl]:
                        if p.name == col_name:
                            profile = p
                            break
                self.column_index._column_meta[col_id] = (tbl, col_name, profile)

            self._learned = True
            age = time.time() - cache.get('saved_at', 0)
            logger.info("Intelligence cache loaded: %d concepts, %d synonyms, age=%.0fs",
                        len(self.domain_concepts), len(self.synonyms), age)
            return True
        except Exception as e:
            logger.warning("Failed to load intelligence cache: %s", e)
            return False

    def learn_or_load(self):
        if self._load_cache():
            return
        self.learn()

    def find_metric(self, concept: str, preferred_table: str = None) -> Optional[Tuple[str, str]]:
        return self.column_index.find_metric_column(concept, preferred_table)

    def find_dimension(self, concept: str, preferred_table: str = None) -> Optional[Tuple[str, str]]:
        return self.column_index.find_dimension_column(concept, preferred_table)

    def find_column(self, concept: str, column_type: str = None,
                    preferred_table: str = None) -> Optional[Tuple[str, str, float]]:
        results = self.column_index.search(concept, top_k=5, column_type=column_type)
        if not results:
            return None
        if preferred_table:
            for tbl, col, _, score in results:
                if tbl == preferred_table:
                    return (tbl, col, score)
        return (results[0][0], results[0][1], results[0][3])

    def heal_sql(self, sql: str, context: Dict = None) -> Dict[str, Any]:
        return self.healer.execute_with_healing(sql, context)

    def get_rate_targets(self, table: str = None) -> Dict[str, DiscoveredConcept]:
        return self.value_discovery.get_rate_targets(table)

    def record_success(self, question: str, sql: str, tables_used: List[str],
                        rows_returned: int = 0):
        self.pattern_learner.record_success(question, sql, tables_used, rows_returned)

    def find_learned_pattern(self, question: str) -> Optional[Dict]:
        return self.pattern_learner.find_pattern(question)

    def merge_with_manual(self, manual_concepts: Dict = None,
                          manual_synonyms: Dict = None):
        if manual_concepts:
            merged = dict(self.domain_concepts)
            merged.update(manual_concepts)
            self.domain_concepts = merged

        if manual_synonyms:
            merged = dict(self.synonyms)
            merged.update(manual_synonyms)
            self.synonyms = merged

    def intelligence_report(self) -> Dict[str, Any]:
        total_concepts = len(self.domain_concepts)
        auto_single = sum(1 for v in self.domain_concepts.values()
                          if v.get('_discovered') and not v.get('_compound'))
        auto_compound = sum(1 for v in self.domain_concepts.values()
                            if v.get('_compound'))
        manual = total_concepts - auto_single - auto_compound

        return {
            'total_concepts': total_concepts,
            'auto_discovered_single': auto_single,
            'auto_discovered_compound': auto_compound,
            'manual_only': manual,
            'auto_percentage': round(100 * (auto_single + auto_compound) / max(total_concepts, 1), 1),
            'total_synonyms': len(self.synonyms),
            'learned_patterns': self.pattern_learner.pattern_count,
            'pattern_reuses': self.pattern_learner.total_reuses,
        }
