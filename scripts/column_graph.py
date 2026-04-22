import logging
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
import json

logger = logging.getLogger('gpdm.column_graph')

try:
    from gpdm_config import (
        COL_CODE_LABEL_SUFFIX_CONFIDENCE, COL_CODE_LABEL_ID_CONFIDENCE,
        COL_CODE_LABEL_SEMANTIC_CONFIDENCE, COL_NAME_PAIR_CONFIDENCE,
        COL_PK_UNIQUENESS_RATIO, COL_FK_CONFIDENCE,
        COL_DIM_METRIC_CONFIDENCE, COL_DIM_MAX_CARDINALITY,
        COL_TEMPORAL_CONFIDENCE,
        COOCCURRENCE_DECAY_FACTOR, COOCCURRENCE_MIN_THRESHOLD,
    )
except ImportError:
    COL_CODE_LABEL_SUFFIX_CONFIDENCE = 0.92
    COL_CODE_LABEL_ID_CONFIDENCE = 0.85
    COL_CODE_LABEL_SEMANTIC_CONFIDENCE = 0.72
    COL_NAME_PAIR_CONFIDENCE = 0.93
    COL_PK_UNIQUENESS_RATIO = 0.95
    COL_FK_CONFIDENCE = 0.82
    COL_DIM_METRIC_CONFIDENCE = 0.80
    COL_DIM_MAX_CARDINALITY = 75
    COL_TEMPORAL_CONFIDENCE = 0.88
    COOCCURRENCE_DECAY_FACTOR = 0.85
    COOCCURRENCE_MIN_THRESHOLD = 2


class ColumnRelationshipGraph:

    CODE_LABEL = 'CODE_LABEL'
    NAME_PAIR = 'NAME_PAIR'
    PRIMARY_KEY = 'PRIMARY_KEY'
    FOREIGN_KEY = 'FOREIGN_KEY'
    DIMENSION_METRIC = 'DIMENSION_METRIC'
    TEMPORAL_ENTITY = 'TEMPORAL_ENTITY'

    def __init__(self, schema_learner):
        self.learner = schema_learner
        self.edges = []
        self.code_label_pairs = {}
        self.name_pairs = {}
        self.pk_fk_map = {}
        self.dimension_metrics = defaultdict(list)
        self.temporal_entities = defaultdict(list)
        logger.info("ColumnRelationshipGraph initialized")

    def discover(self):
        logger.info("Starting column relationship discovery")
        try:
            for table, profiles in self.learner.tables.items():
                logger.debug(f"Discovering relationships in table: {table}")
                self._discover_code_label_pairs(table, profiles)
                self._discover_name_pairs(table, profiles)
                self._discover_pk_fk(table, profiles)
                self._discover_dimension_metrics(table, profiles)
                self._discover_temporal_entities(table, profiles)

            self._discover_cross_table_fks()
            logger.info(f"Discovery complete. Found {len(self.edges)} relationships")
        except Exception as e:
            logger.error(f"Error during relationship discovery: {e}", exc_info=True)
            raise

    def _discover_code_label_pairs(self, table: str, profiles: List) -> None:
        profile_map = {p.name.upper(): p for p in profiles}

        for profile in profiles:
            col_upper = profile.name.upper()
            confidence = 0.0
            label_col = None

            if col_upper.endswith('_CODE'):
                base = col_upper[:-5]
                for suffix in ['_DESCRIPTION', '_DESC', '_NAME', '_TEXT', '_LABEL']:
                    candidate = base + suffix
                    if candidate in profile_map:
                        label_col = candidate
                        confidence = COL_CODE_LABEL_SUFFIX_CONFIDENCE
                        break

            elif col_upper.endswith('_ID') and profile.is_id:
                base = col_upper[:-3]
                for suffix in ['_NAME', '_DESCRIPTION', '_TITLE']:
                    candidate = base + suffix
                    if candidate in profile_map:
                        label_col = candidate
                        confidence = COL_CODE_LABEL_ID_CONFIDENCE
                        break

            if not label_col and 'code' in (profile.semantic_tags or []):
                col_parts = col_upper.split('_')
                for other in profiles:
                    if other.name.upper() == col_upper:
                        continue
                    if 'text_descriptive' in (other.semantic_tags or []):
                        other_parts = other.name.upper().split('_')
                        shared = set(col_parts) & set(other_parts)
                        shared -= {'ID', 'CODE', 'NAME', 'DESC', 'TEXT', 'TYPE', 'STATUS'}
                        if shared:
                            label_col = other.name.upper()
                            confidence = COL_CODE_LABEL_SEMANTIC_CONFIDENCE
                            break

            if label_col:
                self.code_label_pairs[col_upper] = label_col
                self.edges.append((col_upper, label_col, self.CODE_LABEL, confidence, table))
                logger.debug(f"Found CODE_LABEL: {col_upper} -> {label_col} (confidence: {confidence})")

    def _discover_name_pairs(self, table: str, profiles: List) -> None:
        profile_map = {p.name.upper(): p for p in profiles}
        col_upper_to_profile = {p.name.upper(): p for p in profiles}

        name_pair_patterns = [
            ('FIRST_NAME', 'LAST_NAME'),
            ('FNAME', 'LNAME'),
            ('GIVEN_NAME', 'FAMILY_NAME'),
        ]

        for first_pattern, last_pattern in name_pair_patterns:
            for col_upper, profile in col_upper_to_profile.items():
                if col_upper.endswith(first_pattern):
                    prefix = col_upper[:-len(first_pattern)]
                    last_candidate = prefix + last_pattern
                    if last_candidate in profile_map:
                        self.name_pairs[col_upper] = (col_upper, last_candidate)
                        self.edges.append((col_upper, last_candidate, self.NAME_PAIR, COL_NAME_PAIR_CONFIDENCE, table))
                        logger.debug(f"Found NAME_PAIR: {col_upper} + {last_candidate}")

    def _discover_pk_fk(self, table: str, profiles: List) -> None:
        row_count = self.learner.table_row_counts.get(table, 1)
        if row_count == 0:
            row_count = 1

        best_pk = None
        best_ratio = 0.0

        for profile in profiles:
            if profile.name.upper().endswith('_ID') and profile.is_id:
                if profile.distinct_count:
                    ratio = profile.distinct_count / row_count
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_pk = profile.name.upper()

        if best_pk:
            if table not in self.pk_fk_map:
                self.pk_fk_map[table] = {'pk': best_pk, 'fks': {}}
            else:
                self.pk_fk_map[table]['pk'] = best_pk
            self.edges.append((table, best_pk, self.PRIMARY_KEY, best_ratio, table))
            logger.debug(f"Found PRIMARY_KEY for {table}: {best_pk} (ratio: {best_ratio:.2f})")

    def _discover_dimension_metrics(self, table: str, profiles: List) -> None:
        categorical_cols = []
        numeric_cols = []

        for profile in profiles:
            tags = profile.semantic_tags or []
            if 'categorical' in tags or (profile.distinct_count and profile.distinct_count < COL_DIM_MAX_CARDINALITY):
                categorical_cols.append(profile)
            if profile.is_numeric:
                numeric_cols.append(profile)

        for cat_profile in categorical_cols:
            dim_col = cat_profile.name.upper()
            for num_profile in numeric_cols:
                metric_col = num_profile.name.upper()
                if dim_col != metric_col:
                    self.dimension_metrics[dim_col].append(metric_col)
                    self.edges.append((dim_col, metric_col, self.DIMENSION_METRIC, COL_DIM_METRIC_CONFIDENCE, table))

            if self.dimension_metrics[dim_col]:
                logger.debug(f"Found DIMENSION_METRIC: {dim_col} -> {self.dimension_metrics[dim_col]}")

    def _discover_temporal_entities(self, table: str, profiles: List) -> None:
        date_cols = []
        id_cols = []

        for profile in profiles:
            tags = profile.semantic_tags or []
            if 'date' in tags:
                date_cols.append(profile)
            if profile.is_id or profile.name.upper().endswith('_ID'):
                id_cols.append(profile)

        for date_profile in date_cols:
            date_col = date_profile.name.upper()
            for id_profile in id_cols:
                id_col = id_profile.name.upper()
                self.temporal_entities[date_col].append(id_col)
                self.edges.append((date_col, id_col, self.TEMPORAL_ENTITY, COL_TEMPORAL_CONFIDENCE, table))

            if self.temporal_entities[date_col]:
                logger.debug(f"Found TEMPORAL_ENTITY: {date_col} -> {self.temporal_entities[date_col]}")

    def _discover_cross_table_fks(self) -> None:
        col_to_tables = defaultdict(list)

        for table, profiles in self.learner.tables.items():
            for profile in profiles:
                col_upper = profile.name.upper()
                col_to_tables[col_upper].append((table, profile))

        for col_name, table_profile_list in col_to_tables.items():
            if len(table_profile_list) >= 2:
                for i, (source_table, source_profile) in enumerate(table_profile_list):
                    for target_table, target_profile in table_profile_list[i+1:]:
                        if source_profile.data_type == target_profile.data_type:
                            confidence = COL_FK_CONFIDENCE

                            if source_table not in self.pk_fk_map:
                                self.pk_fk_map[source_table] = {'pk': None, 'fks': {}}
                            if target_table not in self.pk_fk_map:
                                self.pk_fk_map[target_table] = {'pk': None, 'fks': {}}

                            self.pk_fk_map[source_table]['fks'][col_name] = target_table
                            self.edges.append((col_name, target_table, self.FOREIGN_KEY, confidence, f"{source_table}-{target_table}"))
                            logger.debug(f"Found FOREIGN_KEY: {col_name} in {source_table} -> {target_table}")

    def get_label_for(self, column_name: str) -> Optional[str]:
        return self.code_label_pairs.get(column_name.upper())

    def get_display_columns(self, entity_table: str) -> List[Tuple[str, str]]:
        if entity_table not in self.learner.tables:
            logger.warning(f"Table not found: {entity_table}")
            return []

        display_cols = []
        seen = set()
        profiles = self.learner.tables[entity_table]
        profile_names = {p.name.upper() for p in profiles}

        for profile in profiles:
            col_upper = profile.name.upper()
            if col_upper in self.name_pairs and col_upper not in seen:
                first, last = self.name_pairs[col_upper]
                display_cols.append((first, self.NAME_PAIR))
                display_cols.append((last, self.NAME_PAIR))
                seen.add(first)
                seen.add(last)

        for profile in profiles:
            col_upper = profile.name.upper()
            if col_upper in self.code_label_pairs and col_upper not in seen:
                label = self.code_label_pairs[col_upper]
                display_cols.append((col_upper, self.CODE_LABEL))
                display_cols.append((label, self.CODE_LABEL))
                seen.add(col_upper)
                seen.add(label)

        if entity_table in self.pk_fk_map:
            pk = self.pk_fk_map[entity_table].get('pk')
            if pk and pk not in seen:
                display_cols.append((pk, self.PRIMARY_KEY))

        return display_cols

    def get_name_columns(self, table: str) -> Optional[Tuple[str, str]]:
        if table not in self.learner.tables:
            return None

        profiles = self.learner.tables[table]
        for profile in profiles:
            col_upper = profile.name.upper()
            if col_upper in self.name_pairs:
                return self.name_pairs[col_upper]

        return None

    def get_primary_key(self, table: str) -> Optional[str]:
        if table in self.pk_fk_map:
            return self.pk_fk_map[table].get('pk')
        return None

    def get_join_columns(self, table_a: str, table_b: str) -> Optional[str]:
        if table_a in self.pk_fk_map:
            fks = self.pk_fk_map[table_a].get('fks', {})
            for fk_col, target_table in fks.items():
                if target_table == table_b:
                    return fk_col

        if table_b in self.pk_fk_map:
            fks = self.pk_fk_map[table_b].get('fks', {})
            for fk_col, target_table in fks.items():
                if target_table == table_a:
                    return fk_col

        return None

    def get_metrics_for_dimension(self, dimension_col: str) -> List[str]:
        dim_upper = dimension_col.upper()
        return self.dimension_metrics.get(dim_upper, [])

    def get_date_column_for(self, table: str) -> Optional[str]:
        if table not in self.learner.tables:
            return None

        profiles = self.learner.tables[table]
        date_candidates = []

        for profile in profiles:
            tags = profile.semantic_tags or []
            if 'date' in tags:
                col_upper = profile.name.upper()
                if col_upper in self.temporal_entities and self.temporal_entities[col_upper]:
                    date_candidates.append((col_upper, 1))
                else:
                    date_candidates.append((col_upper, 0))

        if date_candidates:
            date_candidates.sort(key=lambda x: x[1], reverse=True)
            return date_candidates[0][0]

        return None

    def to_dict(self) -> Dict:
        return {
            'edges': self.edges,
            'code_label_pairs': self.code_label_pairs,
            'name_pairs': self.name_pairs,
            'pk_fk_map': self.pk_fk_map,
            'dimension_metrics': dict(self.dimension_metrics),
            'temporal_entities': dict(self.temporal_entities),
        }

    @classmethod
    def from_dict(cls, data: Dict, schema_learner) -> 'ColumnRelationshipGraph':
        graph = cls(schema_learner)
        graph.edges = data.get('edges', [])
        graph.code_label_pairs = data.get('code_label_pairs', {})
        graph.name_pairs = data.get('name_pairs', {})
        graph.pk_fk_map = data.get('pk_fk_map', {})
        graph.dimension_metrics = defaultdict(list, data.get('dimension_metrics', {}))
        graph.temporal_entities = defaultdict(list, data.get('temporal_entities', {}))
        logger.info("ColumnRelationshipGraph deserialized from cache")
        return graph


class ColumnCooccurrenceMatrix:

    def __init__(self):
        self.matrix = {}
        self.query_count = 0
        logger.info("ColumnCooccurrenceMatrix initialized")

    def record_success(self, columns_used: List[str]) -> None:
        if not columns_used:
            return

        normalized = [col.upper() for col in columns_used]

        for i, col_a in enumerate(normalized):
            for col_b in normalized[i+1:]:
                key = tuple(sorted([col_a, col_b]))
                self.matrix[key] = self.matrix.get(key, 0) + 1

        self.query_count += 1
        if self.query_count % 100 == 0:
            logger.debug(f"Recorded {self.query_count} successful queries")

    def suggest_related_columns(self, column: str, top_k: int = 5) -> List[Tuple[str, int]]:
        col_upper = column.upper()
        suggestions = {}

        for (col_a, col_b), count in self.matrix.items():
            if col_a == col_upper:
                suggestions[col_b] = suggestions.get(col_b, 0) + count
            elif col_b == col_upper:
                suggestions[col_a] = suggestions.get(col_a, 0) + count

        sorted_suggestions = sorted(suggestions.items(), key=lambda x: x[1], reverse=True)
        return sorted_suggestions[:top_k]

    def get_pair_score(self, col_a: str, col_b: str) -> int:
        key = tuple(sorted([col_a.upper(), col_b.upper()]))
        return self.matrix.get(key, 0)

    def to_dict(self) -> Dict:
        return {
            'matrix': self.matrix,
            'query_count': self.query_count,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ColumnCooccurrenceMatrix':
        matrix = cls()
        matrix.matrix = data.get('matrix', {})
        matrix.query_count = data.get('query_count', 0)
        logger.info(f"ColumnCooccurrenceMatrix deserialized with {matrix.query_count} recorded queries")
        return matrix

    def decay(self, factor: float = None) -> None:
        if factor is None:
            factor = COOCCURRENCE_DECAY_FACTOR
        if factor <= 0 or factor >= 1:
            logger.warning(f"Invalid decay factor {factor}, must be between 0 and 1")
            return

        original_pairs = len(self.matrix)

        threshold = COOCCURRENCE_MIN_THRESHOLD
        decayed_matrix = {}

        for key, count in self.matrix.items():
            new_count = int(count * factor)
            if new_count >= threshold:
                decayed_matrix[key] = new_count

        self.matrix = decayed_matrix
        removed = original_pairs - len(self.matrix)

        logger.info(f"Applied decay factor {factor}: {original_pairs} pairs -> {len(self.matrix)} pairs (removed {removed})")
