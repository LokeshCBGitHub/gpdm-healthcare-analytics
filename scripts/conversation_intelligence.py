import re
import math
import json
import time
import sqlite3
import logging
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict

logger = logging.getLogger('mtp.conversation')


@dataclass
class QueryState:
    tables: List[str] = field(default_factory=list)
    select_columns: List[str] = field(default_factory=list)
    filters: List[str] = field(default_factory=list)
    group_by: List[str] = field(default_factory=list)
    order_by: str = ''
    limit: Optional[int] = None

    intent: str = ''
    agg_func: str = ''
    metric_column: str = ''
    time_column: str = ''
    time_filter: str = ''

    source_question: str = ''
    source_sql: str = ''
    result_count: int = 0
    turn_number: int = 0

    confidence: float = 0.0
    explanation: str = ''

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'QueryState':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ConversationContext:
    session_id: str = ''
    user_id: str = ''
    current_state: QueryState = field(default_factory=QueryState)
    turn_history: List[Dict] = field(default_factory=list)
    last_results: List[Any] = field(default_factory=list)
    last_result_columns: List[str] = field(default_factory=list)
    preferred_tables: Set[str] = field(default_factory=set)
    preferred_metrics: List[str] = field(default_factory=list)
    chain_depth: int = 0


class FollowUpClassifier:

    FOLLOWUP_PATTERNS = [
        (r'^(?:now\s+)?(?:break\s+(?:it\s+)?(?:down\s+)?)?by\s+(.+)$', 'add_groupby', 0.95),
        (r'^(?:group|grouped|split|broken down)\s+by\s+(.+)$', 'add_groupby', 0.95),
        (r'^per\s+(\w+)$', 'add_groupby', 0.9),

        (r'^(?:but\s+)?(?:only|just|filter(?:ed)?|restrict(?:ed)?)\s+(?:to\s+|for\s+)?(.+)$', 'add_filter', 0.9),
        (r'^(?:exclude|remove|without|except|not)\s+(.+)$', 'add_filter', 0.9),
        (r'^for\s+(\w+(?:\s+\w+)?)$', 'add_filter', 0.85),
        (r'^(?:where|when|if)\s+(.+)$', 'add_filter', 0.85),

        (r'^(?:same|that)\s+(?:but\s+)?(?:for|in|during)\s+(last\s+\w+|this\s+\w+|20\d\d|q[1-4])', 'temporal_shift', 0.95),
        (r'^(?:month|year|quarter|week)\s+(?:over|by)\s+(?:month|year|quarter|week)', 'temporal_shift', 0.9),
        (r'^(?:year|month)\s+over\s+year', 'temporal_shift', 0.9),
        (r'^(?:ytd|year.to.date|mtd|month.to.date)', 'temporal_shift', 0.9),

        (r'^(?:show|as|give|compute)\s+(?:the\s+)?(?:average|avg|sum|total|count|percentage|pct|%|rate|median)', 'refine_metric', 0.9),
        (r'^(?:average|avg|sum|total|percentage|rate)\s+(?:instead|of that|of those)', 'refine_metric', 0.9),

        (r'^(?:more\s+)?detail(?:s)?\s*(?:on|for|about)?', 'drill_down', 0.85),
        (r'^expand\s+(?:on\s+)?(?:that|those|it|this)', 'drill_down', 0.9),
        (r'^(?:drill|dig|dive)\s+(?:down|into|deeper)', 'drill_down', 0.9),
        (r'^show\s+me\s+(?:the\s+)?(?:details|breakdown|specifics)', 'drill_down', 0.85),

        (r'^(?:the|show|details?\s+(?:on|for))\s+(?:top|bottom|first|last)\s+\d+', 'result_reference', 0.9),
        (r'^(?:show|details?\s+(?:on|for))\s+(?:#|number|row)\s*\d+', 'result_reference', 0.9),
        (r'^(?:the|those)\s+(?:denied|approved|pending|completed|cancelled|active)', 'result_reference', 0.9),

        (r'^(?:what|how)\s+about\s+(.+?)[\?.]?$', 'context_switch', 0.85),
        (r'^(?:now|next|instead)\s+(?:show|do|try|look at)\s+(.+)$', 'context_switch', 0.85),
        (r'^switch\s+to\s+(.+)$', 'context_switch', 0.95),

        (r'^(?:but|and)\s+(?:only\s+)?(?:over|above|more than|greater than|>)\s+\$?[\d,]+', 'constraint_modify', 0.9),
        (r'^(?:but|and)\s+(?:only\s+)?(?:under|below|less than|fewer than|<)\s+\$?[\d,]+', 'constraint_modify', 0.9),
        (r'^(?:but|and)\s+(?:only\s+)?(?:between|from)\s+\$?[\d,]+\s+(?:and|to)\s+\$?[\d,]+', 'constraint_modify', 0.9),
        (r'^(?:but|and)\s+(?:in|from|at)\s+(.+)$', 'constraint_modify', 0.85),
    ]

    NEW_QUESTION_SIGNALS = [
        r'\bhow many\s+\w+\s+are\b',
        r'\bwhat is the (?:total|average|count)\b',
        r'\bshow me all\b',
        r'\btotal\s+(?:number|count)\s+of\b',
        r'\blist (?:all|every)\b',
        r'\bgive me (?:a|the)\b',
        r'\bfind (?:all|every)\b',
        r'\bwhich\s+\w+\s+(?:have|has|are|is|were)\b',
        r'\bcompare\s+\w+\s+(?:and|vs|versus|with|to)\b',
        r'\bcreate\s+a\b',
        r'\bgenerate\s+a\b',
    ]

    @classmethod
    def classify(cls, question: str, context: ConversationContext) -> Dict:
        q = question.lower().strip()
        q_words = set(q.split())
        word_count = len(q.split())

        if not context.turn_history:
            return cls._new_question_result()

        for pattern, ftype, conf in cls.FOLLOWUP_PATTERNS:
            m = re.match(pattern, q)
            if m:
                term = m.group(1) if m.lastindex else ''
                return {
                    'is_followup': True,
                    'followup_type': ftype,
                    'confidence': conf,
                    'matched_pattern': pattern,
                    'extracted_term': term.strip(),
                    'explanation': f'Pattern match: {ftype} ({conf:.0%})',
                }

        new_q_score = 0
        for pattern in cls.NEW_QUESTION_SIGNALS:
            if re.search(pattern, q):
                new_q_score += 0.4
                break

        if word_count >= 7:
            new_q_score += 0.2

        followup_score = 0

        if word_count <= 3:
            followup_score += 0.4
        elif word_count <= 5:
            followup_score += 0.2

        REFERENCE_WORDS = {'those', 'that', 'these', 'them', 'it', 'this',
                           'same', 'above', 'previous', 'last', 'again'}
        ref_overlap = q_words & REFERENCE_WORDS
        if ref_overlap:
            followup_score += 0.3 * len(ref_overlap)

        CONTINUATION_WORDS = {'also', 'and', 'but', 'now', 'instead', 'then',
                              'next', 'plus', 'additionally'}
        if q_words & CONTINUATION_WORDS:
            followup_score += 0.25

        if context.current_state.tables:
            prev_tables = set(t.lower() for t in context.current_state.tables)
            for t in prev_tables:
                if t in q or t.rstrip('s') in q:
                    followup_score += 0.15

        if context.current_state.source_question:
            prev_words = set(context.current_state.source_question.lower().split())
            STOP = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'of', 'in',
                     'for', 'to', 'by', 'and', 'or', 'how', 'what', 'show', 'me',
                     'many', 'much', 'all', 'total'}
            prev_meaningful = prev_words - STOP
            curr_meaningful = q_words - STOP
            if prev_meaningful and curr_meaningful:
                overlap = len(prev_meaningful & curr_meaningful)
                if overlap > 0:
                    followup_score += 0.1 * min(overlap, 3)

        net_score = followup_score - new_q_score

        if net_score >= 0.4:
            ftype = cls._infer_followup_type(q, context)
            return {
                'is_followup': True,
                'followup_type': ftype,
                'confidence': min(0.5 + net_score * 0.3, 0.85),
                'matched_pattern': 'heuristic',
                'extracted_term': q,
                'explanation': f'Heuristic: {ftype} (score={net_score:.2f})',
            }

        return cls._new_question_result()

    @classmethod
    def _infer_followup_type(cls, q: str, context: ConversationContext) -> str:
        if re.search(r'\bby\s+\w+', q):
            return 'add_groupby'
        if re.search(r'\b(denied|approved|pending|cancelled|completed)\b', q):
            return 'add_filter'
        if re.search(r'\b(20\d\d|last\s+(year|month|quarter)|this\s+(year|month))\b', q):
            return 'temporal_shift'
        if re.search(r'\b(detail|expand|drill|breakdown)\b', q):
            return 'drill_down'
        if re.search(r'\b(average|sum|total|percentage|rate)\b', q):
            return 'refine_metric'
        return 'continuation'

    @staticmethod
    def _new_question_result() -> Dict:
        return {
            'is_followup': False,
            'followup_type': 'new_question',
            'confidence': 0.0,
            'matched_pattern': '',
            'extracted_term': '',
            'explanation': 'New question — starting fresh context',
        }


class QueryRewriter:

    @classmethod
    def rewrite(cls, question: str, classification: Dict,
                context: ConversationContext) -> Dict:
        q = question.lower().strip()
        ftype = classification['followup_type']
        term = classification.get('extracted_term', '')
        state = context.current_state
        prev_q = state.source_question

        if not prev_q:
            return {
                'rewritten': question,
                'original': question,
                'modifications': [],
                'carry_forward': {},
            }

        mods = []
        carry = {
            'tables': list(state.tables),
            'filters': list(state.filters),
            'group_by': list(state.group_by),
            'intent': state.intent,
            'metric': state.metric_column,
            'previous_sql': state.source_sql,
        }

        if ftype == 'add_groupby':
            base = re.sub(r'\s+by\s+[\w\s]+$', '', prev_q)
            rewritten = f"{base} by {term}" if term else f"{base} by {q.split('by',1)[-1].strip()}"
            mods.append(f'Added grouping: {term}')

        elif ftype == 'add_filter':
            if re.match(r'^(?:exclude|remove|without|except|not)\b', q):
                rewritten = f"{prev_q} excluding {term}"
                mods.append(f'Added exclusion: {term}')
            else:
                by_match = re.search(r'\s+(by\s+.+)$', prev_q)
                if by_match:
                    base = prev_q[:by_match.start()]
                    by_clause = by_match.group(1)
                    table_names = [t.lower() for t in state.tables]
                    inserted = False
                    for tname in table_names:
                        for form in [tname, tname.rstrip('s'), tname + 's']:
                            idx = base.lower().find(form)
                            if idx >= 0:
                                rewritten = f"{base[:idx]}{term} {base[idx:]} {by_clause}"
                                inserted = True
                                break
                        if inserted:
                            break
                    if not inserted:
                        rewritten = f"{base} {term} {by_clause}"
                else:
                    rewritten = f"{prev_q} {term}"
                mods.append(f'Added filter: {term}')

        elif ftype == 'temporal_shift':
            base = re.sub(r'\b(?:in|for|during)\s+(?:20\d\d|last\s+\w+|this\s+\w+|q[1-4])\b',
                          '', prev_q).strip()
            time_expr = term if term else q
            time_m = re.search(r'(last\s+\w+|this\s+\w+|20\d\d|q[1-4]|ytd|year.to.date|'
                               r'month\s+over\s+month|year\s+over\s+year)', q)
            if time_m:
                time_expr = time_m.group(1)
            rewritten = f"{base} in {time_expr}"
            mods.append(f'Changed time window: {time_expr}')

        elif ftype == 'refine_metric':
            agg_m = re.search(r'(average|avg|sum|total|count|percentage|pct|rate|median)', q)
            if agg_m:
                new_agg = agg_m.group(1)
                agg_words = r'(average|avg|sum|total|count|percentage|pct|rate|number of)'
                if re.search(agg_words, prev_q):
                    rewritten = re.sub(agg_words, new_agg, prev_q, count=1)
                else:
                    rewritten = f"{new_agg} {prev_q}"
                mods.append(f'Changed metric to: {new_agg}')
            else:
                rewritten = prev_q

        elif ftype == 'drill_down':
            detail_base = re.sub(
                r'\b(total|average|avg|count|sum|number of|how many)\s+',
                '', prev_q
            )
            rewritten = f"show details for {detail_base}"
            mods.append('Drill-down to record details')

        elif ftype == 'result_reference':
            status_m = re.search(r'\b(denied|approved|pending|completed|cancelled|active|returned)\b', q)
            if status_m:
                rewritten = f"{prev_q} {status_m.group(1)}"
                mods.append(f'Filter to: {status_m.group(1)}')
            else:
                top_m = re.search(r'(?:top|bottom|first|last)\s+(\d+)', q)
                if top_m:
                    rewritten = f"top {top_m.group(1)} {prev_q}"
                    mods.append(f'Limited to top {top_m.group(1)}')
                else:
                    rewritten = prev_q

        elif ftype == 'context_switch':
            new_subject = term.rstrip('?').strip()

            prev_tables = set(t.lower() for t in state.tables)
            rewritten = prev_q
            replaced = False
            for old_table in prev_tables:
                for form in [old_table, old_table.rstrip('s'), old_table + 's']:
                    if form in rewritten.lower():
                        rewritten = re.sub(
                            re.escape(form), new_subject,
                            rewritten, count=1, flags=re.IGNORECASE
                        )
                        replaced = True
                        break
                if replaced:
                    break

            if not replaced:
                intent_word = ''
                if state.intent == 'count':
                    intent_word = 'count of'
                elif state.intent == 'aggregate':
                    intent_word = 'total'
                elif state.intent == 'breakdown':
                    intent_word = ''

                by_match = re.search(r'\s+(by\s+.+)$', prev_q)
                by_clause = by_match.group(1) if by_match else ''
                rewritten = f"{intent_word} {new_subject} {by_clause}".strip()

            mods.append(f'Switched context to: {new_subject}')
            carry['tables'] = []
            carry['filters'] = []

        elif ftype == 'constraint_modify':
            num_m = re.search(r'(over|above|more than|greater than|>|'
                              r'under|below|less than|fewer than|<|'
                              r'between|from)\s+\$?([\d,]+)', q)
            if num_m:
                constraint = q.lstrip('but ').lstrip('and ')
                rewritten = f"{prev_q} {constraint}"
                mods.append(f'Added constraint: {constraint}')
            else:
                rewritten = f"{prev_q} {q.lstrip('but ').lstrip('and ')}"
                mods.append(f'Added constraint: {q}')

        else:
            if q.startswith(('also ', 'and ', 'plus ')):
                rewritten = f"{prev_q} {q}"
            elif q.startswith(('but ', 'however ')):
                rewritten = f"{prev_q} {q}"
            else:
                rewritten = question
            mods.append('Continuation')

        return {
            'rewritten': rewritten.strip(),
            'original': question,
            'modifications': mods,
            'carry_forward': carry,
        }


class ConversationIntelligence:

    def __init__(self, db_path: str, catalog_dir: str = None, nlp_mode: str = 'auto'):
        self.db_path = db_path
        self.catalog_dir = catalog_dir
        self.nlp_mode = nlp_mode

        from semantic_sql_engine import SemanticSQLEngine
        self.engine = SemanticSQLEngine(db_path, catalog_dir, nlp_mode=nlp_mode)

        self.clinical = None
        try:
            from tuva_clinical_layer import TuvaClinicalEnricher
            self.clinical = TuvaClinicalEnricher(db_path)
            if self.clinical.is_healthcare:
                logger.info("Tuva clinical layer ACTIVE — healthcare schema detected")
            else:
                logger.info("Tuva clinical layer inactive — non-healthcare schema")
                self.clinical = None
        except ImportError:
            logger.info("Tuva clinical layer not available")

        self._contexts: Dict[str, ConversationContext] = {}

        self._db = None
        self._init_storage()

        logger.info("ConversationIntelligence initialized")

    def _init_storage(self):
        try:
            db_dir = os.path.dirname(self.db_path)
            ctx_db = os.path.join(db_dir, 'conversation_context.db')
            self._db = sqlite3.connect(ctx_db, check_same_thread=False)
            self._db.execute("""
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    turn_number INTEGER NOT NULL,
                    timestamp REAL NOT NULL,
                    original_question TEXT NOT NULL,
                    rewritten_question TEXT,
                    followup_type TEXT,
                    followup_confidence REAL,
                    sql_generated TEXT,
                    intent TEXT,
                    tables_used TEXT,
                    filters_applied TEXT,
                    group_by_cols TEXT,
                    result_count INTEGER,
                    confidence REAL,
                    modifications TEXT,
                    chain_depth INTEGER DEFAULT 0
                )
            """)
            self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_conv_session
                ON conversation_turns(session_id, turn_number)
            """)
            self._db.commit()
        except Exception as e:
            logger.warning("Could not init conversation storage: %s", e)
            self._db = None

    def get_context(self, session_id: str) -> ConversationContext:
        if session_id not in self._contexts:
            ctx = ConversationContext(session_id=session_id)
            if self._db:
                try:
                    rows = self._db.execute("""
                        SELECT original_question, rewritten_question, sql_generated,
                               intent, tables_used, filters_applied, group_by_cols,
                               result_count, turn_number, followup_type
                        FROM conversation_turns
                        WHERE session_id = ?
                        ORDER BY turn_number DESC LIMIT 10
                    """, (session_id,)).fetchall()
                    for row in reversed(rows):
                        ctx.turn_history.append({
                            'question': row[0],
                            'rewritten': row[1],
                            'sql': row[2],
                            'intent': row[3],
                            'tables_used': json.loads(row[4]) if row[4] else [],
                            'filters': json.loads(row[5]) if row[5] else [],
                            'group_by': json.loads(row[6]) if row[6] else [],
                            'result_count': row[7],
                            'turn_number': row[8],
                            'followup_type': row[9],
                        })
                    if rows:
                        last = rows[0]
                        ctx.current_state = QueryState(
                            tables=json.loads(last[4]) if last[4] else [],
                            filters=json.loads(last[5]) if last[5] else [],
                            group_by=json.loads(last[6]) if last[6] else [],
                            intent=last[3] or '',
                            source_question=last[0],
                            source_sql=last[2] or '',
                            result_count=last[7] or 0,
                            turn_number=last[8],
                        )
                except Exception as e:
                    logger.warning("Could not load conversation history: %s", e)

            self._contexts[session_id] = ctx
        return self._contexts[session_id]

    def process_turn(self, question: str, session_id: str = 'default',
                     execute: bool = True) -> Dict[str, Any]:
        ctx = self.get_context(session_id)

        classification = FollowUpClassifier.classify(question, ctx)
        is_followup = classification['is_followup']
        followup_type = classification['followup_type']

        if is_followup:
            rewrite_result = QueryRewriter.rewrite(question, classification, ctx)
            rewritten = rewrite_result['rewritten']
            modifications = rewrite_result['modifications']
            carry_forward = rewrite_result['carry_forward']
            ctx.chain_depth += 1
        else:
            rewritten = question
            modifications = []
            carry_forward = {}
            ctx.chain_depth = 0

        engine_result = self.engine.generate(rewritten)

        if self.clinical:
            engine_result = self.clinical.enrich(rewritten, engine_result)

        sql = engine_result.get('sql', '')
        intent = engine_result.get('intent', '')
        tables_used = engine_result.get('tables_used', [])
        confidence = engine_result.get('confidence', 0)

        results = []
        columns = []
        result_count = 0
        if execute and sql:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.execute(sql)
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                results = cursor.fetchall()
                result_count = len(results)
                conn.close()
            except Exception as e:
                logger.warning("SQL execution error: %s | SQL: %s", e, sql[:200])
                results = []
                result_count = 0

        filters_from_sql = self._extract_filters(sql)
        group_by_from_sql = self._extract_group_by(sql)

        ctx.current_state = QueryState(
            tables=tables_used,
            filters=filters_from_sql,
            group_by=group_by_from_sql,
            intent=intent,
            source_question=rewritten,
            source_sql=sql,
            result_count=result_count,
            turn_number=len(ctx.turn_history) + 1,
            confidence=confidence,
        )

        ctx.last_results = results[:50]
        ctx.last_result_columns = columns

        ctx.turn_history.append({
            'question': question,
            'rewritten': rewritten,
            'sql': sql,
            'intent': intent,
            'tables_used': tables_used,
            'filters': filters_from_sql,
            'group_by': group_by_from_sql,
            'result_count': result_count,
            'turn_number': ctx.current_state.turn_number,
            'followup_type': followup_type,
            'is_followup': is_followup,
            'modifications': modifications,
            'timestamp': time.time(),
        })

        if len(ctx.turn_history) > 20:
            ctx.turn_history = ctx.turn_history[-20:]

        if self._db:
            try:
                self._db.execute("""
                    INSERT INTO conversation_turns
                    (session_id, turn_number, timestamp, original_question,
                     rewritten_question, followup_type, followup_confidence,
                     sql_generated, intent, tables_used, filters_applied,
                     group_by_cols, result_count, confidence, modifications,
                     chain_depth)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    ctx.current_state.turn_number,
                    time.time(),
                    question,
                    rewritten,
                    followup_type,
                    classification['confidence'],
                    sql,
                    intent,
                    json.dumps(tables_used),
                    json.dumps(filters_from_sql),
                    json.dumps(group_by_from_sql),
                    result_count,
                    confidence,
                    json.dumps(modifications),
                    ctx.chain_depth,
                ))
                self._db.commit()
            except Exception as e:
                logger.warning("Could not persist turn: %s", e)

        context_summary = self._build_context_summary(ctx)

        return {
            'original_question': question,
            'rewritten_question': rewritten,
            'is_followup': is_followup,
            'followup_type': followup_type,
            'followup_confidence': classification['confidence'],
            'modifications': modifications,
            'sql': sql,
            'intent': intent,
            'tables_used': tables_used,
            'confidence': confidence,
            'explanation': engine_result.get('explanation', ''),
            'result_count': result_count,
            'results': results if execute else [],
            'columns': columns if execute else [],
            'chain_depth': ctx.chain_depth,
            'conversation_context': context_summary,
            'semantic_intent': engine_result.get('semantic_intent', ''),
            'semantic_columns': engine_result.get('semantic_columns', []),
            'semantic_values': engine_result.get('semantic_values', []),
            'data_warnings': engine_result.get('data_warnings', []),
            'used_fallback': engine_result.get('used_fallback', False),
            'clinical_context': engine_result.get('clinical_context', ''),
            'clinical_terms': engine_result.get('clinical_terms', []),
            'quality_measure': engine_result.get('quality_measure'),
        }

    def reset_context(self, session_id: str = 'default'):
        if session_id in self._contexts:
            del self._contexts[session_id]

    def get_conversation_history(self, session_id: str = 'default',
                                 limit: int = 10) -> List[Dict]:
        ctx = self.get_context(session_id)
        return ctx.turn_history[-limit:]

    def get_suggested_followups(self, session_id: str = 'default') -> List[str]:
        ctx = self.get_context(session_id)
        state = ctx.current_state
        suggestions = []

        if not state.tables:
            return suggestions

        if not state.group_by:
            for table in state.tables:
                for p in self.engine.semantic.learner.tables.get(table, []):
                    if p.is_categorical and not p.is_id:
                        suggestions.append(f"by {p.name.lower().replace('_', ' ')}")
                        if len(suggestions) >= 2:
                            break
                if len(suggestions) >= 2:
                    break

        if state.intent in ('count', 'aggregate', 'breakdown'):
            for table in state.tables:
                dates = self.engine.semantic.learner.get_date_columns(table)
                if dates:
                    suggestions.append("monthly trend")
                    break

        if state.intent in ('count',):
            suggestions.append("show as percentage")
        elif state.intent in ('aggregate',):
            suggestions.append("show the average instead")

        for table in state.tables:
            for p in self.engine.semantic.learner.tables.get(table, []):
                if p.is_categorical and p.sample_values and not p.is_id:
                    val = p.sample_values[0]
                    suggestions.append(f"only {val.lower().replace('_', ' ')}")
                    break
            break

        return suggestions[:5]


    def _extract_filters(self, sql: str) -> List[str]:
        m = re.search(r'WHERE\s+(.+?)(?:\s+GROUP\s+BY|\s+ORDER\s+BY|\s+LIMIT|;|$)',
                      sql, re.IGNORECASE)
        if m:
            where = m.group(1)
            return [c.strip() for c in re.split(r'\s+AND\s+', where, flags=re.IGNORECASE)]
        return []

    def _extract_group_by(self, sql: str) -> List[str]:
        m = re.search(r'GROUP\s+BY\s+(.+?)(?:\s+ORDER\s+BY|\s+HAVING|\s+LIMIT|;|$)',
                      sql, re.IGNORECASE)
        if m:
            return [c.strip() for c in m.group(1).split(',')]
        return []

    def _build_context_summary(self, ctx: ConversationContext) -> str:
        if not ctx.turn_history:
            return "New conversation"

        parts = []
        depth = ctx.chain_depth
        if depth > 0:
            parts.append(f"Follow-up chain: {depth} deep")

        state = ctx.current_state
        if state.tables:
            parts.append(f"Tables: {', '.join(state.tables)}")
        if state.filters:
            parts.append(f"Filters: {len(state.filters)}")
        if state.group_by:
            parts.append(f"Grouped by: {', '.join(state.group_by)}")
        if state.result_count:
            parts.append(f"Last result: {state.result_count} rows")

        return ' | '.join(parts) if parts else "Active conversation"


import os

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    db_path = os.path.join(os.path.dirname(script_dir), 'data', 'healthcare_production.db')
    if not os.path.exists(db_path):
        print(f"DB not found: {db_path}")
        exit(1)

    ci = ConversationIntelligence(db_path)

    CONVERSATIONS = [
        [
            "how many claims are there",
            "by region",
            "only denied",
            "same for 2024",
        ],
        [
            "total billed amount by claim type",
            "show average instead",
            "by region",
        ],
        [
            "count of members",
            "by age group",
            "what about encounters",
        ],
        [
            "claims greater than 5000",
            "by provider specialty",
            "but only for denied claims",
        ],
    ]

    for conv_idx, conversation in enumerate(CONVERSATIONS):
        session = f"test_{conv_idx}"
        ci.reset_context(session)
        print(f"\n{'='*70}")
        print(f"CONVERSATION {conv_idx + 1}")
        print(f"{'='*70}")

        for turn_idx, question in enumerate(conversation):
            result = ci.process_turn(question, session_id=session)
            fu = "FOLLOW-UP" if result['is_followup'] else "NEW"
            ftype = result['followup_type'] or ''
            print(f"\n  T{turn_idx+1}: \"{question}\"")
            print(f"      [{fu}:{ftype}] conf={result['followup_confidence']:.2f} "
                  f"chain={result['chain_depth']}")
            if result['is_followup']:
                print(f"      Rewritten: \"{result['rewritten_question']}\"")
            print(f"      SQL: {result['sql'][:100]}")
            print(f"      Rows: {result['result_count']}")
            if result['modifications']:
                print(f"      Mods: {result['modifications']}")

        suggestions = ci.get_suggested_followups(session)
        if suggestions:
            print(f"\n  Suggestions: {suggestions[:3]}")

    print(f"\n{'='*70}")
    print("CONVERSATION INTELLIGENCE TEST COMPLETE")
