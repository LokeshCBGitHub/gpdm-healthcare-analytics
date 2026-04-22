import re
import logging
from typing import Dict, List, Tuple, Optional, Any

from schema_graph import SemanticSchemaGraph, TABLE_CONCEPTS
from intent_parser import ParsedIntent, ParsedFilter

logger = logging.getLogger('gpdm.sql_constructor')


class SQLConstructor:

    CONCEPT_TABLE_MAP = {
        'drugs': ('prescriptions', 'COST'),
        'drug': ('prescriptions', 'COST'),
        'medications': ('prescriptions', 'COST'),
        'medication': ('prescriptions', 'COST'),
        'prescriptions': ('prescriptions', 'COST'),
        'pharmacy': ('prescriptions', 'COST'),
        'rx': ('prescriptions', 'COST'),
        'hospital': ('claims', 'PAID_AMOUNT'),
        'hospital stays': ('claims', 'PAID_AMOUNT'),
        'inpatient': ('claims', 'PAID_AMOUNT'),
        'hospitalization': ('claims', 'PAID_AMOUNT'),
        'encounters': ('claims', 'PAID_AMOUNT'),
        'visits': ('claims', 'PAID_AMOUNT'),
        'outpatient': ('claims', 'PAID_AMOUNT'),
        'claims': ('claims', 'PAID_AMOUNT'),
        'referrals': ('claims', 'PAID_AMOUNT'),
    }

    def __init__(self, schema_graph: SemanticSchemaGraph, db_path: str):
        self.graph = schema_graph
        self.db_path = db_path

    def construct(self, intent: ParsedIntent) -> Dict[str, Any]:
        reasoning_steps = []

        primary_table = self._select_primary_table(intent)
        reasoning_steps.append(f"Primary table: {primary_table}")

        needed_tables = self._determine_needed_tables(intent, primary_table)
        reasoning_steps.append(f"Tables needed: {needed_tables}")

        joins = self._build_joins(primary_table, needed_tables)
        reasoning_steps.append(f"JOINs: {len(joins)} paths")

        needed_before = set(needed_tables)
        select_clause = self._build_select(intent, primary_table, needed_tables)
        reasoning_steps.append(f"SELECT: {select_clause[:80]}")

        swap_primary = False
        if intent.intent == 'rank' and intent.group_by and intent.agg_table:
            grp_table = intent.group_by[0][0]
            if grp_table == intent.agg_table and grp_table != primary_table:
                primary_table = grp_table
                swap_primary = True
                reasoning_steps.append(f"Primary swapped to {primary_table} (group+agg colocated)")

        if set(needed_tables) != needed_before or swap_primary:
            if swap_primary:
                needed_tables_new = self._determine_needed_tables(intent, primary_table)
                needed_tables.clear()
                needed_tables.extend(needed_tables_new)
            joins = self._build_joins(primary_table, needed_tables)
            if swap_primary:
                select_clause = self._build_select(intent, primary_table, needed_tables)
            reasoning_steps.append(f"Tables updated after SELECT: {needed_tables}")

        where_clause = self._build_where(intent, primary_table, needed_tables)
        if where_clause:
            reasoning_steps.append(f"WHERE: {where_clause[:80]}")

        group_by = self._build_group_by(intent, primary_table, needed_tables)
        if group_by:
            reasoning_steps.append(f"GROUP BY: {group_by}")

        order_limit = self._build_order_limit(intent, select_clause)

        sql = self._assemble_sql(
            select_clause, primary_table, joins,
            where_clause, group_by, order_limit
        )

        reasoning_steps.append(f"Final SQL: {sql[:100]}")

        return {
            'sql': sql,
            'tables_used': needed_tables,
            'confidence': intent.confidence,
            'reasoning': ' → '.join(reasoning_steps),
        }

    def _select_primary_table(self, intent: ParsedIntent) -> str:
        if not intent.tables:
            return 'claims'

        if intent.intent == 'rate':
            for f in intent.filters:
                if f.column in ('STATUS', 'CLAIM_STATUS', 'ENCOUNTER_STATUS'):
                    hint = f.table_hint if f.table_hint else ''
                    if hint and hint in [t for t in intent.tables]:
                        return hint
            if 'appointments' in intent.tables:
                return 'appointments'
            return 'claims'

        if intent.agg_column in ('PAID_AMOUNT', 'BILLED_AMOUNT', 'ALLOWED_AMOUNT',
                                  'COPAY', 'COINSURANCE', 'DEDUCTIBLE',
                                  'MEMBER_RESPONSIBILITY'):
            return 'claims'
        if intent.agg_column == 'COST':
            return 'prescriptions'

        if intent.intent in ('count', 'list', 'exists', 'summary'):
            raw_q = getattr(intent, 'original_question', '') or getattr(intent, 'normalized_question', '')
            q_words = set(re.sub(r'[^\w\s]', '', w) for w in raw_q.lower().split() if w) if raw_q else set()
            q_words.discard('')
            check_terms = set()
            for v in intent.values:
                check_terms.add(v.lower() if isinstance(v, str) else str(v).lower())
            check_terms |= q_words

            for t in intent.tables:
                tc = self.graph.tables.get(t)
                if tc:
                    name_variants = {t, tc.concept} | set(tc.synonyms)
                    for variant in name_variants:
                        v_lower = variant.lower()
                        if v_lower in check_terms:
                            return t
                        for ct in check_terms:
                            if len(ct) >= 4 and len(v_lower) >= 4:
                                if ct.startswith(v_lower) or v_lower.startswith(ct):
                                    return t

            for gt, gc in intent.group_by:
                if gt in intent.tables:
                    return gt

            return intent.tables[0]

        if 'diagnoses' in intent.tables:
            raw_q = getattr(intent, 'original_question', '') or ''
            q_lower = raw_q.lower()
            tc = self.graph.tables.get('diagnoses')
            diag_names = {'diagnoses', 'diagnosis'} | (set(tc.synonyms) if tc else set())
            if any(n in q_lower for n in diag_names if len(n) >= 4):
                if intent.agg_column and 'AMOUNT' in (intent.agg_column or '').upper():
                    return 'claims'
                return 'diagnoses'

        if intent.intent in ('aggregate', 'rank') and intent.agg_column:
            if any(k in (intent.agg_column or '').upper() for k in ['AMOUNT', 'COST']):
                return 'claims' if intent.agg_column != 'COST' else 'prescriptions'

        if intent.intent == 'rank' and len(intent.tables) >= 2:
            raw_q = getattr(intent, 'original_question', '') or ''
            q_lower = raw_q.lower()
            for t in intent.tables:
                tc = self.graph.tables.get(t)
                if tc:
                    names = {t, tc.concept} | set(tc.synonyms)
                    for name in names:
                        if re.search(r'(?:most|highest|top|least|fewest)\s+' + re.escape(name), q_lower):
                            return t
            for t in intent.tables:
                tc = self.graph.tables.get(t)
                if tc:
                    names = {t, tc.concept} | set(tc.synonyms)
                    for name in names:
                        if re.search(r'(?:by|per)\s+' + re.escape(name) + r'\s+(?:count|volume|number)', q_lower):
                            return t

        if intent.intent == 'rank' and intent.group_by:
            grp_table, grp_col = intent.group_by[0]
            candidate = intent.tables[0]
            if grp_col not in self.graph.columns.get(candidate, {}):
                if grp_col in self.graph.columns.get(grp_table, {}):
                    return grp_table

        return intent.tables[0]

    def _determine_needed_tables(self, intent: ParsedIntent,
                                  primary: str) -> List[str]:
        needed = {primary}


        actually_needed_cols = set()

        if intent.agg_table and intent.agg_column:
            actually_needed_cols.add((intent.agg_table, intent.agg_column))

        for gt, gc in intent.group_by:
            actually_needed_cols.add((gt, gc))

        for f in intent.filters:
            if f.table_hint:
                actually_needed_cols.add((f.table_hint, f.column))

        explicit_group_tables = {gt for gt, gc in intent.group_by if gt in (intent.tables or [])}

        primary_cols = set(self.graph.columns.get(primary, {}).keys())
        for t, col in actually_needed_cols:
            if col in primary_cols and t not in explicit_group_tables:
                if intent.agg_table == t and intent.agg_column == col:
                    intent.agg_table = primary
                for i, (gt, gc) in enumerate(intent.group_by):
                    if gt == t and gc == col and col in primary_cols:
                        intent.group_by[i] = (primary, gc)
                for f in intent.filters:
                    if f.table_hint == t and f.column == col and col in primary_cols:
                        f.table_hint = primary
            else:
                needed.add(t)

        for f in intent.filters:
            if f.table_hint:
                needed.add(f.table_hint)
            else:
                tables_with_col = self.graph.column_to_tables.get(f.column, [])
                if tables_with_col:
                    for t in tables_with_col:
                        if t in needed:
                            f.table_hint = t
                            break
                    if not f.table_hint:
                        f.table_hint = tables_with_col[0]
                        needed.add(f.table_hint)

        for table, col in intent.group_by:
            needed.add(table)

        if intent.agg_table:
            needed.add(intent.agg_table)

        if intent.sub_intent == 'per_unit':
            needed.add('claims')
            needed.add('members')

        if intent.intent == 'correlate':
            if len(needed) < 2:
                if 'claims' not in needed:
                    needed.add('claims')

        return list(needed)

    def _build_joins(self, primary: str, needed: List[str]) -> List[str]:
        if len(needed) <= 1:
            return []

        join_clauses = []
        other_tables = [t for t in needed if t != primary]

        join_paths = self.graph.find_multi_table_join([primary] + other_tables)

        used_tables = {primary}
        for jp in join_paths:
            if jp.to_table in used_tables and jp.from_table in used_tables:
                continue

            src = jp.from_table
            tgt = jp.to_table

            if tgt not in used_tables:
                join_clauses.append(
                    f"{jp.join_type} JOIN {tgt} ON {src}.{jp.from_col} = {tgt}.{jp.to_col}"
                )
                used_tables.add(tgt)
            elif src not in used_tables:
                join_clauses.append(
                    f"{jp.join_type} JOIN {src} ON {tgt}.{jp.to_col} = {src}.{jp.from_col}"
                )
                used_tables.add(src)

        for t in other_tables:
            if t not in used_tables:
                jp = self.graph.find_join_path(primary, t)
                if jp:
                    join_clauses.append(
                        f"{jp.join_type} JOIN {t} ON {primary}.{jp.from_col} = {t}.{jp.to_col}"
                    )
                    used_tables.add(t)

        return join_clauses

    def _build_select(self, intent: ParsedIntent, primary: str,
                       needed: List[str]) -> str:
        parts = []

        if intent.intent == 'count':
            pk = self.graph.tables.get(primary, TABLE_CONCEPTS.get('claims'))
            count_col = pk.count_column if pk else '*'
            _has_distinct = getattr(intent, 'distinct', False)
            _specific_col = intent.agg_column if intent.agg_column and _has_distinct else None
            if _specific_col:
                prefix = f"{primary}." if len(needed) > 1 else ""
                parts.append(f"COUNT(DISTINCT {prefix}{_specific_col}) AS total")
            elif intent.distinct or len(needed) > 1:
                parts.append(f"COUNT(DISTINCT {primary}.{count_col}) AS total")
            else:
                parts.append("COUNT(*) AS total")

            for table, col in intent.group_by:
                prefix = f"{table}." if len(needed) > 1 else ""
                parts.insert(0, f"{prefix}{col}")

        elif intent.intent == 'aggregate':
            agg_col = intent.agg_column
            agg_func = intent.agg_function or 'SUM'
            agg_table = intent.agg_table or primary

            if intent.sub_intent == 'per_unit':
                parts.append(
                    f"ROUND(SUM(CAST({agg_table}.{agg_col} AS REAL)) / "
                    f"NULLIF(COUNT(DISTINCT {agg_table}.MEMBER_ID), 0), 2) AS per_member"
                )
                parts.append(f"COUNT(DISTINCT {agg_table}.MEMBER_ID) AS unique_members")
                parts.append(f"ROUND(SUM(CAST({agg_table}.{agg_col} AS REAL)), 2) AS total")
            else:
                prefix = f"{agg_table}." if len(needed) > 1 else ""
                if agg_col:
                    parts.append(
                        f"ROUND({agg_func}(CAST({prefix}{agg_col} AS REAL)), 2) AS {agg_func.lower()}_{agg_col.lower()}"
                    )
                else:
                    parts.append("COUNT(*) AS total")

            for table, col in intent.group_by:
                prefix = f"{table}." if len(needed) > 1 else ""
                parts.insert(0, f"{prefix}{col}")

        elif intent.intent == 'rate':
            status_filter = None
            for f in intent.filters:
                if f.column in ('CLAIM_STATUS', 'STATUS', 'ENCOUNTER_STATUS'):
                    status_filter = f
                    break

            if status_filter:
                status_val = status_filter.value
                prefix = f"{primary}." if len(needed) > 1 else ""
                parts.append(f"COUNT(*) AS total")
                parts.append(
                    f"SUM(CASE WHEN {prefix}{status_filter.column} = '{status_val}' THEN 1 ELSE 0 END) AS {status_val.lower()}_count"
                )
                parts.append(
                    f"ROUND(100.0 * SUM(CASE WHEN {prefix}{status_filter.column} = '{status_val}' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) AS {status_val.lower()}_rate"
                )
                intent.filters = [f for f in intent.filters if f is not status_filter]
            elif getattr(intent, 'rate_type', None) == 'denial':
                prefix = f"{primary}." if len(needed) > 1 else ""
                parts.append(f"COUNT(*) AS total")
                parts.append(
                    f"SUM(CASE WHEN {prefix}CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) AS denied_count"
                )
                parts.append(
                    f"ROUND(100.0 * SUM(CASE WHEN {prefix}CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) AS denied_rate"
                )
            else:
                prefix = f"{primary}." if len(needed) > 1 else ""
                raw_q = getattr(intent, 'original_question', '') or ''
                q_lower = raw_q.lower()
                if any(w in q_lower for w in ['denied', 'denial', 'reject', 'clean claim']):
                    parts.append(f"COUNT(*) AS total")
                    parts.append(
                        f"SUM(CASE WHEN {prefix}CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) AS denied_count"
                    )
                    parts.append(
                        f"ROUND(100.0 * SUM(CASE WHEN {prefix}CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) AS denied_rate"
                    )
                elif any(w in q_lower for w in ['approved', 'clean', 'acceptance']):
                    parts.append(f"COUNT(*) AS total")
                    parts.append(
                        f"SUM(CASE WHEN {prefix}CLAIM_STATUS != 'DENIED' THEN 1 ELSE 0 END) AS approved_count"
                    )
                    parts.append(
                        f"ROUND(100.0 * SUM(CASE WHEN {prefix}CLAIM_STATUS != 'DENIED' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) AS approved_rate"
                    )
                else:
                    parts.append("COUNT(*) AS total")

            for table, col in intent.group_by:
                prefix = f"{table}." if len(needed) > 1 else ""
                parts.insert(0, f"{prefix}{col}")

        elif intent.intent == 'compare':
            compare_cols = getattr(intent, 'compare_columns', None) or []
            prefix = f"{primary}." if len(needed) > 1 else ""
            if compare_cols:
                for tbl, col in compare_cols:
                    p = f"{tbl}." if len(needed) > 1 else ""
                    parts.append(f"ROUND(SUM(CAST({p}{col} AS REAL)), 2) AS total_{col.lower()}")
                    parts.append(f"ROUND(AVG(CAST({p}{col} AS REAL)), 2) AS avg_{col.lower()}")
                    if tbl not in needed:
                        needed.append(tbl)
            else:
                parts.append(f"ROUND(SUM(CAST({prefix}PAID_AMOUNT AS REAL)), 2) AS total_paid")
                parts.append(f"ROUND(SUM(CAST({prefix}BILLED_AMOUNT AS REAL)), 2) AS total_billed")
            parts.append("COUNT(*) AS claim_count")

        elif intent.intent == 'rank':
            if intent.group_by:
                grp_table, grp_col = intent.group_by[0]

                if grp_col == 'IS_CHRONIC':
                    if 'ICD10_DESCRIPTION' in self.graph.columns.get('diagnoses', {}):
                        grp_table = 'diagnoses'
                        grp_col = 'ICD10_DESCRIPTION'
                        intent.group_by[0] = (grp_table, grp_col)
                        has_chronic_filter = any(f.column == 'IS_CHRONIC' for f in intent.filters)
                        if not has_chronic_filter:
                            from intent_parser import ParsedFilter
                            intent.filters.append(ParsedFilter(
                                column='IS_CHRONIC', operator='=', value='Y',
                                table_hint='diagnoses', confidence=0.95
                            ))

                prefix = f"{grp_table}." if len(needed) > 1 else ""
                parts.append(f"{prefix}{grp_col}")
                if intent.agg_column:
                    agg_prefix = f"{intent.agg_table}." if intent.agg_table and len(needed) > 1 else ""
                    _rank_agg = intent.agg_function or 'SUM'
                    _alias = f"{_rank_agg.lower()}_{intent.agg_column.lower()}"
                    parts.append(
                        f"ROUND({_rank_agg}(CAST({agg_prefix}{intent.agg_column} AS REAL)), 2) AS {_alias}"
                    )
                else:
                    parts.append("COUNT(*) AS cnt")
            else:
                grp_col = None

                raw_q = getattr(intent, 'original_question', '') or getattr(intent, 'normalized_question', '')
                q_lower = raw_q.lower()
                OTHER_TABLE_FK_MAP = {
                    ('claims', 'providers'): 'RENDERING_NPI',
                    ('claims', 'members'): 'MEMBER_ID',
                    ('encounters', 'providers'): 'RENDERING_NPI',
                    ('encounters', 'members'): 'MEMBER_ID',
                    ('prescriptions', 'providers'): 'PRESCRIBING_NPI',
                    ('prescriptions', 'members'): 'MEMBER_ID',
                    ('diagnoses', 'members'): 'MEMBER_ID',
                    ('appointments', 'providers'): 'PROVIDER_NPI',
                    ('appointments', 'members'): 'MEMBER_ID',
                    ('referrals', 'providers'): 'REFERRING_NPI',
                    ('referrals', 'members'): 'MEMBER_ID',
                }
                for other_t in intent.tables:
                    if other_t == primary:
                        continue
                    fk_col = OTHER_TABLE_FK_MAP.get((primary, other_t))
                    if fk_col and fk_col in self.graph.columns.get(primary, {}):
                        tc_other = self.graph.tables.get(other_t)
                        if tc_other:
                            names = {other_t, tc_other.concept} | set(tc_other.synonyms)
                            if any(n in q_lower for n in names):
                                grp_col = (primary, fk_col)
                                break

                if not grp_col:
                    best_match = (None, 0)
                    for table, col in intent.columns:
                        sem = self.graph.columns.get(table, {}).get(col)
                        if sem and sem.groupable and not sem.is_identifier:
                            col_words = col.lower().replace('_', ' ').split()
                            subcategory_words = (sem.subcategory or '').replace('_', ' ').split()
                            all_words = set(col_words + subcategory_words)
                            match_count = sum(1 for w in all_words if len(w) >= 3 and w in q_lower)
                            if match_count > best_match[1]:
                                best_match = ((table, col), match_count)
                    if best_match[0]:
                        grp_col = best_match[0]

                if not grp_col:
                    for table, col in intent.columns:
                        sem = self.graph.columns.get(table, {}).get(col)
                        if sem and sem.groupable and not sem.is_identifier:
                            if col == 'IS_CHRONIC' and 'ICD10_DESCRIPTION' in self.graph.columns.get('diagnoses', {}):
                                grp_col = ('diagnoses', 'ICD10_DESCRIPTION')
                                has_chronic_filter = any(f.column == 'IS_CHRONIC' for f in intent.filters)
                                if not has_chronic_filter:
                                    from intent_parser import ParsedFilter
                                    intent.filters.append(ParsedFilter(
                                        column='IS_CHRONIC', operator='=', value='Y',
                                        table_hint='diagnoses', confidence=0.95
                                    ))
                            else:
                                grp_col = (table, col)
                            break
                if grp_col:
                    if grp_col[0] not in needed:
                        needed.append(grp_col[0])

                    if intent.agg_column and intent.agg_table != grp_col[0]:
                        grp_table_money = self.graph.get_money_columns(grp_col[0])
                        if grp_table_money:
                            raw_q = getattr(intent, 'original_question', '') or ''
                            q_lower = raw_q.lower()
                            for mc in grp_table_money:
                                mc_words = mc.lower().replace('_', ' ').split()
                                if any(w in q_lower for w in mc_words if len(w) >= 3):
                                    intent.agg_column = mc
                                    intent.agg_table = grp_col[0]
                                    break

                    prefix = f"{grp_col[0]}." if len(needed) > 1 else ""
                    parts.append(f"{prefix}{grp_col[1]}")
                    if intent.agg_column:
                        agg_prefix = f"{intent.agg_table}." if intent.agg_table and len(needed) > 1 else ""
                        _rank_agg2 = intent.agg_function or 'SUM'
                        parts.append(
                            f"ROUND({_rank_agg2}(CAST({agg_prefix}{intent.agg_column} AS REAL)), 2) AS total"
                        )
                    else:
                        parts.append("COUNT(*) AS cnt")
                    intent.group_by.append(grp_col)
                else:
                    RANK_DEFAULT_COLS = {
                        'diagnoses': 'ICD10_DESCRIPTION',
                        'providers': 'SPECIALTY',
                        'prescriptions': 'MEDICATION_NAME',
                        'members': 'PLAN_TYPE',
                        'encounters': 'VISIT_TYPE',
                        'claims': 'CPT_DESCRIPTION',
                        'referrals': 'REFERRAL_REASON',
                        'appointments': 'APPOINTMENT_TYPE',
                    }
                    rank_col = RANK_DEFAULT_COLS.get(primary)
                    if not rank_col:
                        tc = self.graph.tables.get(primary)
                        rank_col = tc.count_column if tc else '*'
                    parts.append(f"{rank_col}")
                    parts.append("COUNT(*) AS cnt")
                    intent.group_by.append((primary, rank_col))

        elif intent.intent == 'trend':
            tc = self.graph.tables.get(primary)
            date_col = tc.primary_date if tc else 'SERVICE_DATE'
            prefix = f"{primary}." if len(needed) > 1 else ""

            gran = intent.time_granularity or 'month'
            if gran == 'month':
                parts.append(f"SUBSTR({prefix}{date_col}, 1, 7) AS period")
            elif gran == 'quarter':
                parts.append(
                    f"SUBSTR({prefix}{date_col}, 1, 4) || '-Q' || "
                    f"((CAST(SUBSTR({prefix}{date_col}, 6, 2) AS INTEGER) - 1) / 3 + 1) AS period"
                )
            elif gran == 'year':
                parts.append(f"SUBSTR({prefix}{date_col}, 1, 4) AS period")
            else:
                parts.append(f"SUBSTR({prefix}{date_col}, 1, 7) AS period")

            if intent.agg_column:
                agg_prefix = f"{intent.agg_table}." if intent.agg_table and len(needed) > 1 else ""
                parts.append(
                    f"ROUND(SUM(CAST({agg_prefix}{intent.agg_column} AS REAL)), 2) AS total"
                )
            parts.append("COUNT(*) AS cnt")

            intent.group_by = []

        elif intent.intent == 'compare':
            if intent.compare_values:
                q_lower = ' '.join(intent.values).lower() if intent.values else ''
                is_cost_compare = any(w in (intent.agg_column or '').upper() for w in ['AMOUNT', 'COST']) or \
                                  any(w in q_lower for w in ['spend', 'cost', 'paid', 'expensive'])

                for val in intent.compare_values:
                    mapped = False
                    for keyword, (col, db_val) in self._get_value_map().items():
                        if val.lower() == keyword or val.lower() in keyword:
                            prefix = f"{primary}." if len(needed) > 1 else ""
                            if is_cost_compare:
                                money_col = intent.agg_column or 'PAID_AMOUNT'
                                agg_prefix = f"{intent.agg_table}." if intent.agg_table and len(needed) > 1 else prefix
                                parts.append(
                                    f"ROUND(SUM(CASE WHEN {prefix}{col} = '{db_val}' THEN CAST({agg_prefix}{money_col} AS REAL) ELSE 0 END), 2) AS {val.lower().replace(' ', '_')}_spend"
                                )
                            else:
                                parts.append(
                                    f"SUM(CASE WHEN {prefix}{col} = '{db_val}' THEN 1 ELSE 0 END) AS {val.lower().replace(' ', '_')}_count"
                                )
                            mapped = True
                            break
                    if not mapped:
                        concept_match = self.CONCEPT_TABLE_MAP.get(val.lower())
                        if concept_match:
                            matched_table, matched_col = concept_match
                            safe_name = val.lower().replace(' ', '_')
                            if is_cost_compare:
                                parts.append(
                                    f"(SELECT ROUND(SUM(CAST({matched_col} AS REAL)), 2) FROM {matched_table}) AS {safe_name}_spend"
                                )
                            else:
                                parts.append(f"(SELECT COUNT(*) FROM {matched_table}) AS {safe_name}_count")
                        else:
                            table_match = self.graph.find_tables_for_question(val)
                            if table_match and is_cost_compare:
                                matched_table = table_match[0][0]
                                money_cols = self.graph.get_money_columns(matched_table)
                                if money_cols:
                                    parts.append(
                                        f"(SELECT ROUND(SUM(CAST({money_cols[0]} AS REAL)), 2) FROM {matched_table}) AS {val.lower().replace(' ', '_')}_spend"
                                    )
                                else:
                                    parts.append(f"(SELECT COUNT(*) FROM {matched_table}) AS {val.lower().replace(' ', '_')}_count")
                            elif table_match:
                                parts.append(f"(SELECT COUNT(*) FROM {table_match[0][0]}) AS {val.lower().replace(' ', '_')}_count")
                            else:
                                parts.append(f"COUNT(*) AS {val.lower().replace(' ', '_')}_count")
            else:
                parts.append("COUNT(*) AS total")
                for table, col in intent.group_by:
                    prefix = f"{table}." if len(needed) > 1 else ""
                    parts.insert(0, f"{prefix}{col}")

        elif intent.intent == 'correlate':
            measures = []
            for table, col in intent.columns:
                sem = self.graph.columns.get(table, {}).get(col)
                if sem and sem.aggregatable:
                    measures.append((table, col))
            if len(measures) >= 2:
                t1, c1 = measures[0]
                t2, c2 = measures[1]
                p1 = f"{t1}." if len(needed) > 1 else ""
                p2 = f"{t2}." if len(needed) > 1 else ""
                parts.append(f"{primary}.MEMBER_ID")
                parts.append(f"ROUND(AVG(CAST({p1}{c1} AS REAL)), 2) AS avg_{c1.lower()}")
                parts.append(f"ROUND(SUM(CAST({p2}{c2} AS REAL)), 2) AS total_{c2.lower()}")
                intent.group_by = [(primary, 'MEMBER_ID')]
            elif intent.agg_column:
                parts.append(f"{primary}.MEMBER_ID")
                prefix = f"{intent.agg_table}." if intent.agg_table and len(needed) > 1 else ""
                parts.append(f"ROUND(SUM(CAST({prefix}{intent.agg_column} AS REAL)), 2) AS total")
                intent.group_by = [(primary, 'MEMBER_ID')]
            else:
                parts.append("COUNT(*) AS total")

        elif intent.intent in ('list', 'exists'):
            tc = self.graph.tables.get(primary)
            if tc:
                parts.append(f"{primary}.{tc.primary_key}")
                useful = self.graph.get_groupable_columns(primary)[:3]
                for col in useful:
                    if col != tc.primary_key:
                        parts.append(f"{primary}.{col}")
                agg_cols = self.graph.get_aggregatable_columns(primary)
                if agg_cols:
                    parts.append(f"{primary}.{agg_cols[0]}")
            else:
                parts.append("*")
            intent.limit = intent.limit or 50

        elif intent.intent == 'summary':
            prefix = f"{primary}." if len(needed) > 1 else ""
            parts.append("COUNT(*) AS total_records")
            money_cols = self.graph.get_money_columns(primary)
            for mc in money_cols[:2]:
                parts.append(f"ROUND(SUM(CAST({prefix}{mc} AS REAL)), 2) AS total_{mc.lower()}")
                parts.append(f"ROUND(AVG(CAST({prefix}{mc} AS REAL)), 2) AS avg_{mc.lower()}")
            agg_cols = self.graph.get_aggregatable_columns(primary)
            for ac in agg_cols[:2]:
                if ac not in money_cols:
                    parts.append(f"ROUND(AVG(CAST({prefix}{ac} AS REAL)), 2) AS avg_{ac.lower()}")

        if not parts:
            parts.append("COUNT(*) AS total")

        return ', '.join(parts)

    def _build_where(self, intent: ParsedIntent, primary: str,
                      needed: List[str]) -> str:
        conditions = []
        multi = len(needed) > 1

        for f in intent.filters:
            table = f.table_hint or primary
            prefix = f"{table}." if multi else ""

            if f.column not in self.graph.columns.get(table, {}):
                found = False
                for t in needed:
                    if f.column in self.graph.columns.get(t, {}):
                        prefix = f"{t}." if multi else ""
                        found = True
                        break
                if not found:
                    continue

            if f.operator == 'LIKE':
                conditions.append(f"LOWER({prefix}{f.column}) {f.operator} LOWER('{f.value}')")
            elif f.operator == 'IN':
                vals = ', '.join(f"'{v}'" for v in f.value)
                conditions.append(f"{prefix}{f.column} IN ({vals})")
            elif f.operator == 'BETWEEN':
                conditions.append(
                    f"CAST({prefix}{f.column} AS REAL) BETWEEN {f.value[0]} AND {f.value[1]}"
                )
            elif f.operator in ('>', '<', '>=', '<='):
                conditions.append(f"CAST({prefix}{f.column} AS REAL) {f.operator} {f.value}")
            else:
                conditions.append(f"{prefix}{f.column} = '{f.value}'")

        if intent.time_range:
            tc = self.graph.tables.get(primary)
            date_col = tc.primary_date if tc else 'SERVICE_DATE'
            prefix = f"{primary}." if multi else ""
            if date_col:
                if intent.time_range == 'last_year':
                    conditions.append(
                        f"{prefix}{date_col} >= date('now', '-1 year')"
                    )
                elif intent.time_range == 'this_year':
                    conditions.append(
                        f"{prefix}{date_col} >= date('now', 'start of year')"
                    )
                elif re.match(r'^\d{4}$', intent.time_range):
                    conditions.append(f"{prefix}{date_col} LIKE '{intent.time_range}%'")

        if intent.temporal or intent.trend:
            tc = self.graph.tables.get(primary)
            date_col = tc.primary_date if tc else 'SERVICE_DATE'
            prefix = f"{primary}." if multi else ""
            if date_col:
                conditions.append(f"{prefix}{date_col} IS NOT NULL")
                conditions.append(f"{prefix}{date_col} != ''")

        if intent.negation and len(needed) > 1:
            pass

        return ' AND '.join(conditions) if conditions else ''

    def _build_group_by(self, intent: ParsedIntent, primary: str,
                         needed: List[str]) -> str:
        multi = len(needed) > 1

        if intent.intent == 'trend':
            tc = self.graph.tables.get(primary)
            date_col = tc.primary_date if tc else 'SERVICE_DATE'
            prefix = f"{primary}." if multi else ""
            gran = intent.time_granularity or 'month'
            if gran == 'month':
                return f"SUBSTR({prefix}{date_col}, 1, 7)"
            elif gran == 'quarter':
                return (f"SUBSTR({prefix}{date_col}, 1, 4) || '-Q' || "
                        f"((CAST(SUBSTR({prefix}{date_col}, 6, 2) AS INTEGER) - 1) / 3 + 1)")
            elif gran == 'year':
                return f"SUBSTR({prefix}{date_col}, 1, 4)"

        group_cols = []
        for table, col in intent.group_by:
            prefix = f"{table}." if multi else ""
            group_cols.append(f"{prefix}{col}")

        return ', '.join(group_cols) if group_cols else ''

    def _build_order_limit(self, intent: ParsedIntent, select_clause: str) -> str:
        parts = []

        if intent.intent == 'trend':
            parts.append("ORDER BY period")
        elif intent.order_by:
            direction = 'DESC' if intent.order_by == 'desc' else 'ASC'
            for alias in ['total', 'cnt', 'count', 'avg_', 'sum_', 'rate',
                          'denied_rate', 'total_paid', 'per_member',
                          'avg_length', 'avg_paid', 'avg_copay', 'sum_paid']:
                if alias in select_clause.lower():
                    match = re.search(rf'AS\s+(\w*{alias}\w*)', select_clause, re.IGNORECASE)
                    if match:
                        parts.append(f"ORDER BY {match.group(1)} {direction}")
                        break
            if not parts and intent.group_by:
                parts.append(f"ORDER BY 2 {direction}")

        if intent.limit:
            parts.append(f"LIMIT {intent.limit}")
        elif intent.intent == 'list':
            parts.append("LIMIT 50")
        elif intent.intent == 'rank' and not intent.limit:
            parts.append("LIMIT 10")

        return ' '.join(parts)

    def _assemble_sql(self, select: str, primary: str, joins: List[str],
                       where: str, group_by: str, order_limit: str) -> str:
        sql_parts = [f"SELECT {select}"]

        all_subqueries = select.strip().startswith('(SELECT ') and ') AS ' in select

        if all_subqueries and not where and not group_by:
            sql_parts.append("FROM (SELECT 1) AS _dummy")
        else:
            sql_parts.append(f"FROM {primary}")
            for join in joins:
                sql_parts.append(join)

        if where:
            sql_parts.append(f"WHERE {where}")

        if group_by:
            sql_parts.append(f"GROUP BY {group_by}")

        if order_limit:
            sql_parts.append(order_limit)
        elif all_subqueries and not where and not group_by:
            sql_parts.append("LIMIT 1")

        return ' '.join(sql_parts)

    def _get_value_map(self):
        from intent_parser import IntentParser
        temp = IntentParser(self.graph)
        return temp.value_map
