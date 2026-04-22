import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict
from datetime import datetime
import os

try:
    from gpdm_config import (
        FUZZY_MATCH_EXACT,
        FUZZY_MATCH_PREFIX,
        FUZZY_MATCH_SUBSTRING,
        FUZZY_TABLE_THRESHOLD,
        FUZZY_LEVENSHTEIN_FLOOR,
        MIN_COLUMN_SCORE,
    )
except ImportError:
    FUZZY_MATCH_EXACT = 1.0
    FUZZY_MATCH_PREFIX = 0.88
    FUZZY_MATCH_SUBSTRING = 0.65
    FUZZY_TABLE_THRESHOLD = 0.55
    FUZZY_LEVENSHTEIN_FLOOR = 0.60
    MIN_COLUMN_SCORE = 0.15


logger = logging.getLogger(__name__)


class ColumnMetadata:

    def __init__(self, column_name: str, table_name: str, definition: Dict[str, Any]):
        self.column_name = column_name
        self.table_name = table_name
        self.data_type = definition.get('data_type', 'string')
        self.semantic_type = definition.get('semantic_type', 'text')
        self.healthcare_type = definition.get('healthcare_type')
        self.is_phi = definition.get('is_phi', False)
        self.cardinality = definition.get('cardinality', 0)
        self.null_percentage = definition.get('null_percentage', 0.0)
        self.description = definition.get('description', '')
        self.sample_values = definition.get('sample_values', [])
        self.top_values = definition.get('top_values', [])

    def __repr__(self) -> str:
        return f"Column({self.table_name}.{self.column_name}, {self.semantic_type})"


class TableDefinition:

    def __init__(self, table_name: str, definition: Dict[str, Any]):
        self.table_name = table_name
        self.purpose = definition.get('table_purpose', '')
        self.row_count = definition.get('total_rows', 0)
        self.column_count = definition.get('total_columns', 0)
        self.has_phi = definition.get('has_phi', False)
        self.dq_score = definition.get('dq_score', 0)
        self.profiled_at = definition.get('profiled_at', '')
        self.columns: Dict[str, ColumnMetadata] = {}

        for col_name, col_def in definition.get('columns', {}).items():
            self.columns[col_name.upper()] = ColumnMetadata(col_name.upper(), table_name, col_def)

    def get_column(self, col_name: str) -> Optional[ColumnMetadata]:
        return self.columns.get(col_name.upper())

    def __repr__(self) -> str:
        return f"Table({self.table_name}, {self.column_count} cols, {self.row_count} rows)"


class RelationshipPath:

    def __init__(self, source: str, target: str, join_col: str, rel_type: str, is_fk: bool):
        self.source = source
        self.target = target
        self.join_column = join_col
        self.relationship_type = rel_type
        self.is_foreign_key = is_fk

    def __repr__(self) -> str:
        return f"{self.source} --({self.join_column})--> {self.target}"


class GPDMDefinitionEngine:

    def __init__(self, catalog_path: Optional[str] = None):
        self.catalog_path = Path(catalog_path or self._find_catalog_path())
        self.tables: Dict[str, TableDefinition] = {}
        self.relationships: List[RelationshipPath] = []
        self.context = ""

        self.term_to_columns: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)
        self.healthcare_type_to_columns: Dict[str, List[str]] = defaultdict(list)
        self.semantic_type_to_columns: Dict[str, List[str]] = defaultdict(list)
        self.column_descriptions: Dict[str, str] = {}

        self.join_paths: Dict[Tuple[str, str], List[RelationshipPath]] = {}

        self._load_tables()
        self._load_relationships()
        self._load_context()
        self._build_indices()

        logger.info(f"GPDM Definition Engine initialized: {len(self.tables)} tables, "
                   f"{len(self.relationships)} relationships")

    def _find_catalog_path(self) -> str:
        if env_path := os.environ.get('GPDM_CATALOG_PATH'):
            return env_path

        candidates = [
            Path(__file__).parent.parent / 'semantic_catalog',
            Path.cwd() / 'semantic_catalog',
            Path.cwd() / 'mtp_demo' / 'semantic_catalog',
        ]

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        raise FileNotFoundError("semantic_catalog directory not found. "
                              "Set GPDM_CATALOG_PATH environment variable.")

    def _load_tables(self) -> None:
        tables_dir = self.catalog_path / 'tables'

        if not tables_dir.exists():
            logger.warning(f"Tables directory not found: {tables_dir}")
            return

        for table_file in tables_dir.glob('*.json'):
            try:
                with open(table_file, 'r') as f:
                    definition = json.load(f)

                table_name = definition.get('table_name', table_file.stem).lower()
                self.tables[table_name] = TableDefinition(table_name, definition)
                logger.debug(f"Loaded table: {table_name}")
            except Exception as e:
                logger.error(f"Failed to load table {table_file}: {e}")

    def _load_relationships(self) -> None:
        rel_file = self.catalog_path / 'relationships' / 'relationship_map.json'

        if not rel_file.exists():
            logger.warning(f"Relationship map not found: {rel_file}")
            return

        try:
            with open(rel_file, 'r') as f:
                rel_data = json.load(f)

            for rel in rel_data.get('relationships', []):
                path = RelationshipPath(
                    source=rel.get('source_table', '').lower(),
                    target=rel.get('target_table', '').lower(),
                    join_col=rel.get('join_column', ''),
                    rel_type=rel.get('relationship_type', ''),
                    is_fk=rel.get('is_fk', False)
                )
                self.relationships.append(path)

                key = (path.source, path.target)
                if key not in self.join_paths:
                    self.join_paths[key] = []
                self.join_paths[key].append(path)

            logger.info(f"Loaded {len(self.relationships)} relationships")
        except Exception as e:
            logger.error(f"Failed to load relationships: {e}")

    def _load_context(self) -> None:
        context_file = self.catalog_path / 'context' / 'full_context.txt'

        if context_file.exists():
            try:
                with open(context_file, 'r') as f:
                    self.context = f.read()
                logger.debug("Loaded full context")
            except Exception as e:
                logger.error(f"Failed to load context: {e}")

    def _build_indices(self) -> None:
        for table_name, table in self.tables.items():
            for col_name, col_meta in table.columns.items():
                if col_meta.healthcare_type:
                    key = col_meta.healthcare_type.lower()
                    self.healthcare_type_to_columns[key].append(f"{table_name}.{col_name}")

                if col_meta.semantic_type:
                    key = col_meta.semantic_type.lower()
                    self.semantic_type_to_columns[key].append(f"{table_name}.{col_name}")

                full_key = f"{table_name}.{col_name}"
                self.column_descriptions[full_key] = col_meta.description

                self._index_column_terms(table_name, col_name, col_meta)

    def _index_column_terms(self, table_name: str, col_name: str, col_meta: ColumnMetadata) -> None:
        terms = set()

        col_lower = col_name.lower()
        terms.add(col_lower)
        terms.update(col_lower.split('_'))

        if col_meta.semantic_type:
            terms.add(col_meta.semantic_type.lower())

        if col_meta.healthcare_type:
            terms.add(col_meta.healthcare_type.lower())

        if col_meta.description:
            desc_lower = col_meta.description.lower()
            for word in desc_lower.split():
                word_clean = word.strip('.,;:')
                if len(word_clean) > 2 and word_clean not in ['the', 'for', 'and', 'or']:
                    terms.add(word_clean)

        for sample in col_meta.sample_values[:3]:
            if isinstance(sample, str):
                terms.add(sample.lower())

        full_col = f"{table_name}.{col_name}"
        for term in terms:
            self.term_to_columns[term].append((full_col, col_meta.semantic_type, FUZZY_MATCH_EXACT))

    def resolve_column(self, term: str, context: Optional[Dict[str, Any]] = None) -> Optional[Tuple[str, float]]:
        term_lower = term.lower().strip()
        candidates = []

        if term_lower in self.term_to_columns:
            for col, sem_type, conf in self.term_to_columns[term_lower]:
                candidates.append((col, conf, 'exact_term'))

        if term_lower in self.healthcare_type_to_columns:
            for col in self.healthcare_type_to_columns[term_lower]:
                candidates.append((col, FUZZY_MATCH_EXACT, 'healthcare_type'))

        if term_lower in self.semantic_type_to_columns:
            for col in self.semantic_type_to_columns[term_lower]:
                candidates.append((col, FUZZY_MATCH_PREFIX, 'semantic_type'))

        for indexed_term, term_cols in self.term_to_columns.items():
            if term_lower in indexed_term and term_lower != indexed_term:
                for col, sem_type, _ in term_cols:
                    candidates.append((col, FUZZY_MATCH_SUBSTRING, 'substring'))
            elif indexed_term in term_lower:
                for col, sem_type, _ in term_cols:
                    candidates.append((col, FUZZY_MATCH_PREFIX, 'prefix'))

        if not candidates:
            return None

        unique_cols = {}
        for col, conf, match_type in candidates:
            if col not in unique_cols or unique_cols[col][0] < conf:
                unique_cols[col] = (conf, match_type)

        if context and 'tables' in context:
            filtered = {col: conf for col, (conf, _) in unique_cols.items()
                       if col.split('.')[0] in context['tables']}
            if filtered:
                unique_cols = {col: (conf, 'table_filtered') for col, (conf, _) in filtered.items()}

        best_col = max(unique_cols.items(), key=lambda x: x[1][0])
        return (best_col[0], best_col[1][0])

    def resolve_table(self, term: str) -> Optional[Tuple[str, float]]:
        term_lower = term.lower().strip()
        candidates = []

        for table_name in self.tables.keys():
            if term_lower == table_name:
                candidates.append((table_name, FUZZY_MATCH_EXACT))
            elif table_name.startswith(term_lower) or term_lower.startswith(table_name[:3]):
                candidates.append((table_name, FUZZY_MATCH_PREFIX))
            elif term_lower in table_name or table_name in term_lower:
                candidates.append((table_name, FUZZY_MATCH_SUBSTRING))

            if table_name in self.tables:
                purpose = self.tables[table_name].purpose.lower()
                if term_lower in purpose:
                    candidates.append((table_name, FUZZY_MATCH_PREFIX))

        if not candidates:
            return None

        best = max(candidates, key=lambda x: x[1])
        if best[1] >= FUZZY_TABLE_THRESHOLD:
            return best

        return None

    def get_join_path(self, table_a: str, table_b: str) -> Optional[RelationshipPath]:
        key = (table_a.lower(), table_b.lower())

        if key in self.join_paths:
            paths = self.join_paths[key]
            if paths:
                fk_paths = [p for p in paths if p.is_foreign_key]
                return fk_paths[0] if fk_paths else paths[0]

        reverse_key = (table_b.lower(), table_a.lower())
        if reverse_key in self.join_paths:
            paths = self.join_paths[reverse_key]
            if paths:
                path = paths[0]
                return RelationshipPath(path.target, path.source, path.join_column,
                                      path.relationship_type, path.is_foreign_key)

        return None

    def get_all_join_paths(self, table_a: str, table_b: str) -> List[RelationshipPath]:
        key = (table_a.lower(), table_b.lower())
        return self.join_paths.get(key, [])

    def get_column_metadata(self, table: str, column: str) -> Optional[Dict[str, Any]]:
        table_def = self.tables.get(table.lower())
        if not table_def:
            return None

        col_meta = table_def.get_column(column.upper())
        if not col_meta:
            return None

        return {
            'column_name': col_meta.column_name,
            'table_name': col_meta.table_name,
            'data_type': col_meta.data_type,
            'semantic_type': col_meta.semantic_type,
            'healthcare_type': col_meta.healthcare_type,
            'is_phi': col_meta.is_phi,
            'cardinality': col_meta.cardinality,
            'null_percentage': col_meta.null_percentage,
            'description': col_meta.description,
            'sample_values': col_meta.sample_values,
            'top_values': col_meta.top_values,
        }

    def get_all_tables(self) -> List[str]:
        return sorted(self.tables.keys())

    def get_table_columns(self, table: str) -> List[str]:
        table_def = self.tables.get(table.lower())
        if not table_def:
            return []
        return sorted(table_def.columns.keys())

    def get_table_purpose(self, table: str) -> str:
        table_def = self.tables.get(table.lower())
        if not table_def:
            return ""
        return table_def.purpose

    def get_table_info(self, table: str) -> Optional[Dict[str, Any]]:
        table_def = self.tables.get(table.lower())
        if not table_def:
            return None

        return {
            'table_name': table_def.table_name,
            'purpose': table_def.purpose,
            'row_count': table_def.row_count,
            'column_count': table_def.column_count,
            'has_phi': table_def.has_phi,
            'dq_score': table_def.dq_score,
            'profiled_at': table_def.profiled_at,
            'columns': sorted(table_def.columns.keys()),
        }

    def enrich_query(self, question: str, tables: List[str], columns: List[str]) -> Dict[str, Any]:
        enrichment = {
            'question': question,
            'tables': tables,
            'columns': columns,
            'table_purposes': {},
            'column_metadata': {},
            'join_paths': {},
            'domain_context': '',
            'phi_warning': False,
            'confidence': 0.0,
        }

        for table in tables:
            purpose = self.get_table_purpose(table)
            if purpose:
                enrichment['table_purposes'][table] = purpose

        for col_spec in columns:
            parts = col_spec.split('.')
            if len(parts) == 2:
                table, col = parts
                meta = self.get_column_metadata(table, col)
                if meta:
                    enrichment['column_metadata'][col_spec] = meta
                    if meta.get('is_phi'):
                        enrichment['phi_warning'] = True

        if len(tables) > 1:
            for i, t1 in enumerate(tables):
                for t2 in tables[i+1:]:
                    path = self.get_join_path(t1, t2)
                    if path:
                        enrichment['join_paths'][f"{t1}→{t2}"] = str(path)

        if self.context:
            enrichment['domain_context'] = self.context[:500]

        return enrichment

    def search_columns_by_healthcare_type(self, hc_type: str) -> List[str]:
        return self.healthcare_type_to_columns.get(hc_type.lower(), [])

    def search_columns_by_semantic_type(self, sem_type: str) -> List[str]:
        return self.semantic_type_to_columns.get(sem_type.lower(), [])

    def get_join_graph(self) -> Dict[str, List[str]]:
        graph = defaultdict(set)
        for rel in self.relationships:
            graph[rel.source].add(rel.target)
            graph[rel.target].add(rel.source)
        return {k: sorted(v) for k, v in graph.items()}


def integrate_gpdm_definitions(pipeline_instance: Any) -> GPDMDefinitionEngine:
    engine = GPDMDefinitionEngine()

    if pipeline_instance:
        pipeline_instance.gpdm = engine
        logger.info("GPDM Definition Engine injected into pipeline")

    return engine


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    engine = GPDMDefinitionEngine()

    print(f"\n=== GPDM Definition Engine Test ===\n")
    print(f"Loaded tables: {engine.get_all_tables()}")
    print(f"\nTable relationships graph:\n{engine.get_join_graph()}\n")

    result = engine.resolve_column("cost")
    print(f"resolve_column('cost'): {result}")

    result = engine.resolve_column("member")
    print(f"resolve_column('member'): {result}")

    result = engine.resolve_table("claims")
    print(f"resolve_table('claims'): {result}")

    path = engine.get_join_path("claims", "members")
    print(f"join_path(claims → members): {path}")

    enriched = engine.enrich_query(
        "How many claims per member?",
        ['claims', 'members'],
        ['claims.claim_id', 'members.member_id']
    )
    print(f"\nQuery enrichment keys: {list(enriched.keys())}")
