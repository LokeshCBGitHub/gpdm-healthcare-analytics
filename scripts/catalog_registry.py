import os
import json
import re
import secrets
import sqlite3
import time
import threading
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger('gpdm.catalog')


COLUMN_DISPLAY_LIMIT = 20
TABLE_DISPLAY_LIMIT = 10

MIN_WORD_LENGTH = 3
POPULARITY_THRESHOLD = 3

EXACT_MATCH_SCORE = 10
KEYWORD_MATCH_SCORE = 5
PARTIAL_MATCH_SCORE = 3


@dataclass
class TableInfo:
    name: str
    schema_name: str
    catalog_name: str
    environment: str
    source_system: str = ""
    table_type: str = "TABLE"
    description: str = ""
    row_count: Optional[int] = None
    columns: List[str] = field(default_factory=list)
    column_types: Dict[str, str] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    parent_concept: str = ""
    last_updated: str = ""
    is_active: bool = True

    @property
    def full_path(self) -> str:
        return f"{self.catalog_name}.{self.schema_name}.{self.name}"

    @property
    def display_path(self) -> str:
        return f"[{self.environment}] {self.full_path}"


@dataclass
class SchemaInfo:
    name: str
    catalog_name: str
    environment: str
    description: str = ""
    tables: Dict[str, TableInfo] = field(default_factory=dict)
    source_system: str = ""


@dataclass
class CatalogInfo:
    name: str
    environment: str
    description: str = ""
    schemas: Dict[str, SchemaInfo] = field(default_factory=dict)
    source_system: str = ""
    is_default: bool = False


@dataclass
class EnvironmentInfo:
    name: str
    description: str = ""
    catalogs: Dict[str, CatalogInfo] = field(default_factory=dict)
    connection_config: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True


CONCEPT_HIERARCHY = {
    'claims': {
        'description': 'Healthcare claims and billing',
        'subtypes': {
            'inpatient_institutional': {
                'description': 'Inpatient hospital/institutional claims (UB-04)',
                'keywords': ['inpatient', 'institutional', 'hospital', 'facility', 'ub04', 'ub-04'],
            },
            'outpatient_professional': {
                'description': 'Outpatient/professional claims (CMS-1500)',
                'keywords': ['outpatient', 'professional', 'physician', 'office', 'cms1500', 'ambulatory'],
            },
            'pharmacy': {
                'description': 'Pharmacy/drug claims',
                'keywords': ['pharmacy', 'drug', 'rx', 'prescription', 'medication', 'ndc'],
            },
            'dental': {
                'description': 'Dental claims',
                'keywords': ['dental', 'oral', 'tooth', 'teeth', 'orthodontic'],
            },
            'behavioral': {
                'description': 'Behavioral/mental health claims',
                'keywords': ['behavioral', 'mental', 'psychiatric', 'therapy', 'counseling'],
            },
            'all': {
                'description': 'All claim types combined',
                'keywords': ['all', 'total', 'combined', 'overall', 'general'],
            },
        },
    },
    'members': {
        'description': 'Member/patient demographics and enrollment',
        'subtypes': {
            'enrollment': {
                'description': 'Current and historical enrollment records',
                'keywords': ['enrollment', 'eligibility', 'coverage', 'plan'],
            },
            'demographics': {
                'description': 'Member demographic information',
                'keywords': ['demographics', 'demographic', 'personal', 'age', 'gender', 'race'],
            },
            'risk': {
                'description': 'Risk scores and stratification',
                'keywords': ['risk', 'hcc', 'raf', 'acuity', 'stratification'],
            },
        },
    },
    'encounters': {
        'description': 'Clinical encounters and visits',
        'subtypes': {
            'inpatient': {
                'description': 'Inpatient admissions',
                'keywords': ['inpatient', 'admission', 'hospital', 'stay'],
            },
            'outpatient': {
                'description': 'Outpatient visits',
                'keywords': ['outpatient', 'office', 'visit', 'ambulatory'],
            },
            'emergency': {
                'description': 'Emergency department visits',
                'keywords': ['emergency', 'er', 'ed', 'urgent'],
            },
            'telehealth': {
                'description': 'Telehealth/virtual visits',
                'keywords': ['telehealth', 'virtual', 'video', 'remote'],
            },
        },
    },
    'providers': {
        'description': 'Provider/physician information',
        'subtypes': {
            'network': {
                'description': 'In-network provider directory',
                'keywords': ['network', 'directory', 'contracted', 'in-network'],
            },
            'performance': {
                'description': 'Provider quality and performance metrics',
                'keywords': ['performance', 'quality', 'star', 'hedis', 'metrics'],
            },
        },
    },
    'diagnoses': {
        'description': 'Clinical diagnosis codes and conditions',
        'subtypes': {},
    },
    'prescriptions': {
        'description': 'Prescription and medication data',
        'subtypes': {},
    },
    'referrals': {
        'description': 'Care referrals and authorizations',
        'subtypes': {},
    },
}

SOURCE_SYSTEMS = {
    'dsw': {
        'full_name': 'Data Services Warehouse (DSW)',
        'description': 'KP primary data warehouse with claims, enrollment, and clinical data',
    },
    'clarity': {
        'full_name': 'Epic Clarity',
        'description': 'Epic EHR reporting database with clinical encounters and orders',
    },
    'apixio': {
        'full_name': 'Apixio HCC',
        'description': 'HCC risk coding and NLP-extracted diagnoses',
    },
    'hedis': {
        'full_name': 'HEDIS Analytics',
        'description': 'Healthcare quality measure data (HEDIS/STARS)',
    },
    'cdw': {
        'full_name': 'Clinical Data Warehouse',
        'description': 'Integrated clinical and administrative data',
    },
}


class CatalogRegistry:

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.environments: Dict[str, EnvironmentInfo] = {}
        self._concept_index: Dict[str, List[TableInfo]] = defaultdict(list)
        self._table_index: Dict[str, TableInfo] = {}
        self._keyword_index: Dict[str, List[TableInfo]] = defaultdict(list)
        self._active_context: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        self._initialized = False

    @classmethod
    def get_instance(cls) -> 'CatalogRegistry':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = CatalogRegistry()
        return cls._instance

    def initialize(self, config_path: Optional[str] = None, auto_discover: bool = True):
        with self._lock:
            if self._initialized:
                return

            if config_path and os.path.exists(config_path):
                self._load_from_config(config_path)

            self._discover_local_tables()

            if auto_discover:
                self._discover_databricks()

            self._rebuild_indexes()
            self._initialized = True

    def _load_from_config(self, path: str):
        try:
            with open(path, 'r') as f:
                config = json.load(f)

            for env_data in config.get('environments', []):
                env = EnvironmentInfo(
                    name=env_data['name'],
                    description=env_data.get('description', ''),
                    connection_config=env_data.get('connection', {}),
                    is_active=env_data.get('is_active', True),
                )

                for cat_data in env_data.get('catalogs', []):
                    catalog = CatalogInfo(
                        name=cat_data['name'],
                        environment=env.name,
                        description=cat_data.get('description', ''),
                        source_system=cat_data.get('source_system', ''),
                        is_default=cat_data.get('is_default', False),
                    )

                    for schema_data in cat_data.get('schemas', []):
                        schema = SchemaInfo(
                            name=schema_data['name'],
                            catalog_name=catalog.name,
                            environment=env.name,
                            description=schema_data.get('description', ''),
                            source_system=schema_data.get('source_system', catalog.source_system),
                        )

                        for tbl_data in schema_data.get('tables', []):
                            table = TableInfo(
                                name=tbl_data['name'],
                                schema_name=schema.name,
                                catalog_name=catalog.name,
                                environment=env.name,
                                source_system=tbl_data.get('source_system', schema.source_system),
                                table_type=tbl_data.get('table_type', 'TABLE'),
                                description=tbl_data.get('description', ''),
                                columns=tbl_data.get('columns', []),
                                column_types=tbl_data.get('column_types', {}),
                                tags=tbl_data.get('tags', []),
                                parent_concept=tbl_data.get('parent_concept', ''),
                                row_count=tbl_data.get('row_count'),
                                is_active=tbl_data.get('is_active', True),
                            )
                            schema.tables[table.name] = table

                        catalog.schemas[schema.name] = schema

                    env.catalogs[catalog.name] = catalog

                self.environments[env.name] = env
        except Exception as e:
            print(f"  Warning: Failed to load catalog config: {e}")

    def _discover_local_tables(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(os.path.dirname(script_dir), 'data', 'healthcare_production.db')

        if not os.path.exists(db_path):
            db_path = os.path.join(os.path.dirname(script_dir), 'data', 'healthcare_demo.db')

        if not os.path.exists(db_path):
            return

        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row

            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()

            env_name = 'local'
            if env_name not in self.environments:
                self.environments[env_name] = EnvironmentInfo(
                    name=env_name,
                    description='Local development environment (SQLite)',
                )

            env = self.environments[env_name]
            cat_name = 'local_catalog'
            if cat_name not in env.catalogs:
                env.catalogs[cat_name] = CatalogInfo(
                    name=cat_name,
                    environment=env_name,
                    description='Local healthcare demo data',
                    is_default=True,
                )

            catalog = env.catalogs[cat_name]
            schema_name = 'default'
            if schema_name not in catalog.schemas:
                catalog.schemas[schema_name] = SchemaInfo(
                    name=schema_name,
                    catalog_name=cat_name,
                    environment=env_name,
                )

            schema = catalog.schemas[schema_name]

            for row in tables:
                tbl_name = row['name']

                cols = conn.execute(f"PRAGMA table_info({tbl_name})").fetchall()
                col_names = [c['name'] for c in cols]
                col_types = {c['name']: c['type'] for c in cols}

                try:
                    count = conn.execute(f"SELECT COUNT(*) FROM {tbl_name}").fetchone()[0]
                except Exception:
                    count = None

                parent = tbl_name.lower().rstrip('s')
                for concept in CONCEPT_HIERARCHY:
                    if concept in tbl_name.lower() or tbl_name.lower().rstrip('s') == concept.rstrip('s'):
                        parent = concept
                        break

                table = TableInfo(
                    name=tbl_name,
                    schema_name=schema_name,
                    catalog_name=cat_name,
                    environment=env_name,
                    description=f"Local {tbl_name} table ({count or '?'} rows)",
                    row_count=count,
                    columns=col_names,
                    column_types=col_types,
                    parent_concept=parent,
                    tags=[tbl_name.lower()],
                    is_active=True,
                )
                schema.tables[tbl_name] = table

            conn.close()
        except Exception as e:
            print(f"  Warning: Failed to discover local tables: {e}")

    def _discover_databricks(self):
        try:
            from databricks_connector import DataSourceManager
            ds = DataSourceManager.get_instance()
            if not ds.is_databricks:
                return

            connector = ds._databricks
            config = ds._config

            env_name = 'prod'
            if env_name not in self.environments:
                self.environments[env_name] = EnvironmentInfo(
                    name=env_name,
                    description='Production Databricks environment',
                    connection_config={
                        'host': config.host,
                        'warehouse_id': config.warehouse_id,
                    },
                )

            env = self.environments[env_name]

            try:
                result = connector.execute_sql("SHOW CATALOGS")
                if result and result.get('rows'):
                    for row in result['rows']:
                        cat_name = row[0] if isinstance(row, (list, tuple)) else list(row.values())[0]
                        if cat_name not in env.catalogs:
                            env.catalogs[cat_name] = CatalogInfo(
                                name=cat_name,
                                environment=env_name,
                                is_default=(cat_name == config.catalog),
                            )

                        try:
                            schemas_result = connector.execute_sql(
                                f"SHOW SCHEMAS IN {cat_name}"
                            )
                            if schemas_result and schemas_result.get('rows'):
                                for s_row in schemas_result['rows']:
                                    s_name = s_row[0] if isinstance(s_row, (list, tuple)) else list(s_row.values())[0]
                                    cat = env.catalogs[cat_name]
                                    if s_name not in cat.schemas:
                                        cat.schemas[s_name] = SchemaInfo(
                                            name=s_name,
                                            catalog_name=cat_name,
                                            environment=env_name,
                                        )

                                    try:
                                        tables_result = connector.execute_sql(
                                            f"SHOW TABLES IN {cat_name}.{s_name}"
                                        )
                                        if tables_result and tables_result.get('rows'):
                                            for t_row in tables_result['rows']:
                                                t_name = t_row[1] if isinstance(t_row, (list, tuple)) else list(t_row.values())[1]

                                                parent = ''
                                                t_lower = t_name.lower()
                                                for concept in CONCEPT_HIERARCHY:
                                                    if concept in t_lower:
                                                        parent = concept
                                                        break

                                                tags = [w for w in t_lower.replace('_', ' ').split() if len(w) > MIN_WORD_LENGTH]

                                                table = TableInfo(
                                                    name=t_name,
                                                    schema_name=s_name,
                                                    catalog_name=cat_name,
                                                    environment=env_name,
                                                    parent_concept=parent,
                                                    tags=tags,
                                                )
                                                cat.schemas[s_name].tables[t_name] = table
                                    except (ImportError, ConnectionError) as e:
                                        logger.debug('Discovery skipped: %s', e)
                        except (ImportError, ConnectionError) as e:
                            logger.debug('Discovery skipped: %s', e)
            except (ImportError, ConnectionError) as e:
                logger.debug('Discovery skipped: %s', e)
        except ImportError as e:
            logger.debug('Discovery skipped: %s', e)

    def _rebuild_indexes(self):
        self._concept_index.clear()
        self._table_index.clear()
        self._keyword_index.clear()

        for env in self.environments.values():
            for catalog in env.catalogs.values():
                for schema in catalog.schemas.values():
                    for table in schema.tables.values():
                        self._table_index[table.full_path] = table
                        self._table_index[table.name.lower()] = table

                        if table.parent_concept:
                            self._concept_index[table.parent_concept].append(table)

                        for tag in table.tags:
                            self._keyword_index[tag.lower()].append(table)

                        for part in table.name.lower().replace('_', ' ').split():
                            if len(part) > MIN_WORD_LENGTH:
                                self._keyword_index[part].append(table)

    def get_environments(self) -> List[Dict]:
        return [
            {
                'name': env.name,
                'description': env.description,
                'is_active': env.is_active,
                'catalog_count': len(env.catalogs),
                'catalogs': list(env.catalogs.keys()),
            }
            for env in self.environments.values()
        ]

    def get_catalogs(self, environment: Optional[str] = None) -> List[Dict]:
        results = []
        for env in self.environments.values():
            if environment and env.name != environment:
                continue
            for cat in env.catalogs.values():
                schema_count = len(cat.schemas)
                table_count = sum(len(s.tables) for s in cat.schemas.values())
                results.append({
                    'name': cat.name,
                    'environment': env.name,
                    'description': cat.description,
                    'source_system': cat.source_system,
                    'is_default': cat.is_default,
                    'schema_count': schema_count,
                    'table_count': table_count,
                })
        return results

    def get_schemas(self, catalog: str, environment: Optional[str] = None) -> List[Dict]:
        results = []
        for env in self.environments.values():
            if environment and env.name != environment:
                continue
            if catalog in env.catalogs:
                cat = env.catalogs[catalog]
                for schema in cat.schemas.values():
                    results.append({
                        'name': schema.name,
                        'catalog': catalog,
                        'environment': env.name,
                        'table_count': len(schema.tables),
                        'tables': list(schema.tables.keys()),
                    })
        return results

    def get_tables(self, concept: Optional[str] = None,
                   catalog: Optional[str] = None,
                   schema: Optional[str] = None,
                   environment: Optional[str] = None,
                   source_system: Optional[str] = None) -> List[Dict]:
        results = []

        if concept:
            concept_lower = concept.lower().rstrip('s')
            for c_key, tables in self._concept_index.items():
                if concept_lower in c_key or c_key in concept_lower:
                    for t in tables:
                        if environment and t.environment != environment:
                            continue
                        if catalog and t.catalog_name != catalog:
                            continue
                        if schema and t.schema_name != schema:
                            continue
                        if source_system and t.source_system.lower() != source_system.lower():
                            continue
                        results.append(self._table_to_dict(t))
        else:
            for env in self.environments.values():
                if environment and env.name != environment:
                    continue
                for cat in env.catalogs.values():
                    if catalog and cat.name != catalog:
                        continue
                    for sch in cat.schemas.values():
                        if schema and sch.name != schema:
                            continue
                        for t in sch.tables.values():
                            if source_system and t.source_system.lower() != source_system.lower():
                                continue
                            results.append(self._table_to_dict(t))

        return results

    def search_tables(self, query: str) -> List[Dict]:
        query_lower = query.lower()
        words = query_lower.replace('_', ' ').split()

        scored: Dict[str, Tuple[float, TableInfo]] = {}

        for word in words:
            for concept, tables in self._concept_index.items():
                if word in concept or concept in word:
                    for t in tables:
                        key = t.full_path
                        score = scored.get(key, (0, t))[0] + EXACT_MATCH_SCORE
                        scored[key] = (score, t)

            for kw, tables in self._keyword_index.items():
                if word in kw or kw in word:
                    for t in tables:
                        key = t.full_path
                        score = scored.get(key, (0, t))[0] + KEYWORD_MATCH_SCORE
                        scored[key] = (score, t)

            for path, t in self._table_index.items():
                if word in path.lower():
                    key = t.full_path
                    score = scored.get(key, (0, t))[0] + PARTIAL_MATCH_SCORE
                    scored[key] = (score, t)

        ranked = sorted(scored.values(), key=lambda x: -x[0])
        return [self._table_to_dict(t) for _, t in ranked]

    def _table_to_dict(self, t: TableInfo) -> Dict:
        return {
            'name': t.name,
            'full_path': t.full_path,
            'display_path': t.display_path,
            'schema': t.schema_name,
            'catalog': t.catalog_name,
            'environment': t.environment,
            'source_system': t.source_system,
            'description': t.description,
            'row_count': t.row_count,
            'column_count': len(t.columns),
            'columns': t.columns[:COLUMN_DISPLAY_LIMIT],
            'parent_concept': t.parent_concept,
            'tags': t.tags,
            'is_active': t.is_active,
        }

    def discover_for_query(self, question: str) -> Dict:
        q_lower = question.lower()

        matched_concept = None
        for concept in CONCEPT_HIERARCHY:
            if concept in q_lower or concept.rstrip('s') in q_lower:
                matched_concept = concept
                break

        if not matched_concept:
            tables = self.search_tables(question)
            if tables:
                matched_concept = tables[0].get('parent_concept', '')

        if not matched_concept:
            return {
                'needs_clarification': False,
                'matched_concept': None,
                'available_catalogs': [],
                'available_subtypes': [],
                'available_sources': [],
                'suggested_table': None,
                'clarification_prompt': None,
            }

        matching_tables = self.get_tables(concept=matched_concept)

        concept_info = CONCEPT_HIERARCHY.get(matched_concept, {})
        matched_subtype = None
        for sub_name, sub_info in concept_info.get('subtypes', {}).items():
            for kw in sub_info.get('keywords', []):
                if kw in q_lower:
                    matched_subtype = sub_name
                    break
            if matched_subtype:
                break

        matched_env = None
        for env_name in self.environments:
            if env_name in q_lower:
                matched_env = env_name
                break

        matched_catalog = None
        for env in self.environments.values():
            for cat_name in env.catalogs:
                if cat_name.lower().replace('_', ' ') in q_lower or cat_name.lower() in q_lower:
                    matched_catalog = cat_name
                    matched_env = env.name
                    break

        matched_source = None
        for src_key in SOURCE_SYSTEMS:
            if src_key in q_lower:
                matched_source = src_key
                break

        available_catalogs = list(set(t['catalog'] for t in matching_tables))
        available_envs = list(set(t['environment'] for t in matching_tables))
        available_sources = list(set(t['source_system'] for t in matching_tables if t['source_system']))
        available_subtypes = list(concept_info.get('subtypes', {}).keys())

        if matched_env:
            matching_tables = [t for t in matching_tables if t['environment'] == matched_env]
        if matched_catalog:
            matching_tables = [t for t in matching_tables if t['catalog'] == matched_catalog]
        if matched_source:
            matching_tables = [t for t in matching_tables if t['source_system'].lower() == matched_source]

        if len(matching_tables) == 1:
            return {
                'needs_clarification': False,
                'matched_concept': matched_concept,
                'available_catalogs': available_catalogs,
                'available_subtypes': available_subtypes,
                'available_sources': available_sources,
                'suggested_table': matching_tables[0],
                'clarification_prompt': None,
            }

        catalog_trigger_words = {
            'catalog', 'schema', 'environment', 'env', 'prod', 'staging', 'dev',
            'enriched', 'dsw', 'clarity', 'apixio', 'hedis', 'cdw',
            'inpatient', 'outpatient', 'institutional', 'professional',
            'pharmacy', 'dental', 'behavioral',
            'which table', 'which catalog', 'which database', 'which source',
            'browse', 'explore', 'discover', 'list tables', 'list catalogs',
        }
        user_wants_catalog = any(tw in q_lower for tw in catalog_trigger_words)

        needs_clarification = user_wants_catalog and (
            len(available_catalogs) > 1 or
            (len(available_subtypes) > 1 and not matched_subtype) or
            len(available_sources) > 1
        )

        clarification_parts = []
        if matched_concept:
            clarification_parts.append(
                f"I found **{matched_concept}** data in the following locations:"
            )

        if len(available_envs) > 1 and not matched_env:
            clarification_parts.append(
                f"\n**Environments:** {', '.join(available_envs)}"
            )

        if len(available_catalogs) > 1 and not matched_catalog:
            clarification_parts.append(
                f"\n**Catalogs:** {', '.join(available_catalogs)}"
            )

        if available_subtypes and not matched_subtype:
            subtype_descs = []
            for st_name, st_info in concept_info.get('subtypes', {}).items():
                subtype_descs.append(f"  • **{st_name}** — {st_info['description']}")
            clarification_parts.append(
                f"\n**Available types:**\n" + '\n'.join(subtype_descs)
            )

        if available_sources and not matched_source:
            source_descs = []
            for src in available_sources:
                src_info = SOURCE_SYSTEMS.get(src.lower(), {})
                src_name = src_info.get('full_name', src)
                source_descs.append(f"  • **{src}** — {src_name}")
            clarification_parts.append(
                f"\n**Data sources:** \n" + '\n'.join(source_descs)
            )

        if matching_tables:
            table_list = []
            for t in matching_tables[:TABLE_DISPLAY_LIMIT]:
                table_list.append(f"  • `{t['display_path']}` ({t.get('row_count', '?')} rows)")
            clarification_parts.append(
                f"\n**Matching tables:**\n" + '\n'.join(table_list)
            )

        clarification_parts.append(
            "\nWhich would you like to query? You can specify the catalog, type, or source."
        )

        return {
            'needs_clarification': needs_clarification,
            'matched_concept': matched_concept,
            'available_catalogs': available_catalogs,
            'available_subtypes': available_subtypes,
            'available_sources': available_sources,
            'suggested_table': matching_tables[0] if len(matching_tables) == 1 else None,
            'matching_tables': matching_tables,
            'clarification_prompt': '\n'.join(clarification_parts) if needs_clarification else None,
        }

    def set_user_context(self, user_id: str, environment: Optional[str] = None,
                         catalog: Optional[str] = None, schema: Optional[str] = None):
        ctx = self._active_context.get(user_id, {})
        if environment:
            ctx['environment'] = environment
        if catalog:
            ctx['catalog'] = catalog
        if schema:
            ctx['schema'] = schema
        self._active_context[user_id] = ctx

    def get_user_context(self, user_id: str) -> Dict:
        return self._active_context.get(user_id, {})

    def clear_user_context(self, user_id: str):
        self._active_context.pop(user_id, None)

    def get_summary(self) -> Dict:
        total_tables = 0
        env_summaries = []
        for env in self.environments.values():
            cat_count = len(env.catalogs)
            tbl_count = sum(
                len(s.tables)
                for c in env.catalogs.values()
                for s in c.schemas.values()
            )
            total_tables += tbl_count
            env_summaries.append({
                'name': env.name,
                'catalogs': cat_count,
                'tables': tbl_count,
                'is_active': env.is_active,
            })

        return {
            'environments': len(self.environments),
            'total_tables': total_tables,
            'concepts': list(self._concept_index.keys()),
            'details': env_summaries,
        }


class ContextualLearningEngine:

    def __init__(self, db_path: Optional[str] = None):
        if not db_path:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(os.path.dirname(script_dir), 'data', 'learning.db')
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        conn = self._get_conn()
        try:
            conn.executescript("""
                -- Master query log: every query from every user
                CREATE TABLE IF NOT EXISTS query_log (
                    query_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    username TEXT,
                    question TEXT NOT NULL,
                    normalized_question TEXT,
                    sql_generated TEXT,
                    tables_used TEXT,       -- JSON array
                    columns_used TEXT,      -- JSON array
                    filters_used TEXT,      -- JSON array
                    intent TEXT,
                    catalog_used TEXT,
                    environment_used TEXT,
                    success INTEGER DEFAULT 1,
                    result_count INTEGER DEFAULT 0,
                    execution_time_ms INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now')),
                    INDEX_KEY TEXT           -- normalized form for grouping
                );

                -- Per-user conversation context
                CREATE TABLE IF NOT EXISTS user_context (
                    context_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    updated_at TEXT DEFAULT (datetime('now')),
                    UNIQUE(user_id, session_id, key)
                );

                -- Query patterns: aggregated patterns learned from queries
                CREATE TABLE IF NOT EXISTS query_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    pattern_text TEXT NOT NULL,     -- normalized query template
                    example_question TEXT,
                    tables_used TEXT,               -- JSON array
                    columns_used TEXT,              -- JSON array
                    intent TEXT,
                    frequency INTEGER DEFAULT 1,
                    success_rate REAL DEFAULT 1.0,
                    last_used TEXT DEFAULT (datetime('now')),
                    avg_execution_ms INTEGER DEFAULT 0
                );

                -- Suggestions: curated suggestions shown to users
                CREATE TABLE IF NOT EXISTS suggestions (
                    suggestion_id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    category TEXT,                  -- 'popular', 'trending', 'similar', 'recommended'
                    source TEXT,                    -- 'user_query', 'system', 'admin'
                    display_order INTEGER DEFAULT 0,
                    times_used INTEGER DEFAULT 0,
                    is_active INTEGER DEFAULT 1,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                -- User preferences: what each user prefers
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id TEXT NOT NULL,
                    preference_key TEXT NOT NULL,
                    preference_value TEXT,
                    updated_at TEXT DEFAULT (datetime('now')),
                    PRIMARY KEY (user_id, preference_key)
                );

                -- Create indexes
                CREATE INDEX IF NOT EXISTS idx_query_log_user ON query_log(user_id);
                CREATE INDEX IF NOT EXISTS idx_query_log_time ON query_log(created_at);
                CREATE INDEX IF NOT EXISTS idx_query_log_norm ON query_log(normalized_question);
                CREATE INDEX IF NOT EXISTS idx_query_log_key ON query_log(INDEX_KEY);
                CREATE INDEX IF NOT EXISTS idx_user_context_user ON user_context(user_id);
                CREATE INDEX IF NOT EXISTS idx_patterns_freq ON query_patterns(frequency DESC);
                CREATE INDEX IF NOT EXISTS idx_suggestions_cat ON suggestions(category, is_active);
            """)
            conn.commit()
        except Exception as e:
            print(f"  Warning: Failed to initialize learning DB: {e}")
        finally:
            conn.close()

    def log_query(self, user_id: str, username: str, question: str,
                  sql: str = "", tables: List[str] = None,
                  columns: List[str] = None, filters: List[str] = None,
                  intent: str = "", catalog: str = "", environment: str = "",
                  success: bool = True, result_count: int = 0,
                  execution_time_ms: int = 0):
        query_id = f"q_{secrets.token_hex(8)}"
        normalized = self._normalize_question(question)
        index_key = self._make_index_key(normalized)

        conn = self._get_conn()
        try:
            conn.execute("""
                INSERT INTO query_log (query_id, user_id, username, question,
                    normalized_question, sql_generated, tables_used, columns_used,
                    filters_used, intent, catalog_used, environment_used,
                    success, result_count, execution_time_ms, INDEX_KEY)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                query_id, user_id, username, question, normalized, sql,
                json.dumps(tables or []), json.dumps(columns or []),
                json.dumps(filters or []), intent, catalog, environment,
                1 if success else 0, result_count, execution_time_ms, index_key,
            ))

            self._update_pattern(conn, normalized, question, tables, columns,
                                intent, success, execution_time_ms)

            self._maybe_add_suggestion(conn, normalized, question)

            conn.commit()
        except Exception as e:
            conn.rollback()
        finally:
            conn.close()

    def _normalize_question(self, question: str) -> str:
        q = question.lower().strip()
        q = re.sub(r'\b\d+\b', '<N>', q)
        q = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', '<DATE>', q)
        q = re.sub(r'"[^"]*"', '<VALUE>', q)
        q = re.sub(r"'[^']*'", '<VALUE>', q)
        q = re.sub(r'\s+', ' ', q)
        return q

    def _make_index_key(self, normalized: str) -> str:
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'how',
                      'many', 'much', 'show', 'me', 'give', 'get', 'find', 'list',
                      'all', 'of', 'in', 'for', 'by', 'to', 'and', 'or', 'with'}
        words = [w for w in re.findall(r'[a-z]+', normalized) if w not in stop_words and len(w) > 2]
        return ' '.join(sorted(set(words)))

    def _update_pattern(self, conn, normalized: str, example: str,
                        tables: List[str], columns: List[str],
                        intent: str, success: bool, exec_ms: int):
        existing = conn.execute(
            "SELECT pattern_id, frequency, success_rate, avg_execution_ms FROM query_patterns WHERE pattern_text = ?",
            (normalized,)
        ).fetchone()

        if existing:
            freq = existing['frequency'] + 1
            old_rate = existing['success_rate']
            new_rate = (old_rate * (freq - 1) + (1.0 if success else 0.0)) / freq
            old_ms = existing['avg_execution_ms']
            new_ms = int((old_ms * (freq - 1) + exec_ms) / freq)
            conn.execute(
                "UPDATE query_patterns SET frequency=?, success_rate=?, avg_execution_ms=?, last_used=datetime('now') WHERE pattern_id=?",
                (freq, new_rate, new_ms, existing['pattern_id'])
            )
        else:
            conn.execute("""
                INSERT INTO query_patterns (pattern_id, pattern_text, example_question,
                    tables_used, columns_used, intent, frequency, success_rate, avg_execution_ms)
                VALUES (?,?,?,?,?,?,1,?,?)
            """, (
                f"pat_{secrets.token_hex(6)}", normalized, example,
                json.dumps(tables or []), json.dumps(columns or []),
                intent, 1.0 if success else 0.0, exec_ms,
            ))

    def _maybe_add_suggestion(self, conn, normalized: str, question: str):
        count = conn.execute(
            "SELECT COUNT(*) FROM query_log WHERE normalized_question = ?",
            (normalized,)
        ).fetchone()[0]

        if count >= POPULARITY_THRESHOLD:
            existing = conn.execute(
                "SELECT suggestion_id FROM suggestions WHERE question = ?",
                (question,)
            ).fetchone()
            if not existing:
                conn.execute("""
                    INSERT OR IGNORE INTO suggestions (suggestion_id, question, category, source, times_used)
                    VALUES (?,?,?,?,?)
                """, (f"sug_{secrets.token_hex(6)}", question, 'popular', 'user_query', count))

    def get_suggestions(self, user_id: Optional[str] = None,
                        category: Optional[str] = None,
                        limit: int = 10) -> List[Dict]:
        conn = self._get_conn()
        try:
            results = []

            popular = conn.execute("""
                SELECT question, COUNT(*) as cnt, MAX(created_at) as last
                FROM query_log
                WHERE success = 1
                GROUP BY normalized_question
                ORDER BY cnt DESC
                LIMIT ?
            """, (limit,)).fetchall()

            for row in popular:
                results.append({
                    'question': row['question'],
                    'category': 'popular',
                    'frequency': row['cnt'],
                    'last_asked': row['last'],
                })

            if user_id:
                recent = conn.execute("""
                    SELECT DISTINCT question, created_at
                    FROM query_log
                    WHERE user_id = ? AND success = 1
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (user_id, limit)).fetchall()

                for row in recent:
                    results.append({
                        'question': row['question'],
                        'category': 'recent',
                        'last_asked': row['created_at'],
                    })

            trending = conn.execute("""
                SELECT question, COUNT(*) as recent_cnt
                FROM query_log
                WHERE created_at > datetime('now', '-1 day')
                AND success = 1
                GROUP BY normalized_question
                HAVING recent_cnt >= 2
                ORDER BY recent_cnt DESC
                LIMIT ?
            """, (limit,)).fetchall()

            for row in trending:
                results.append({
                    'question': row['question'],
                    'category': 'trending',
                    'frequency': row['recent_cnt'],
                })

            curated = conn.execute("""
                SELECT question, category, times_used
                FROM suggestions
                WHERE is_active = 1
                ORDER BY display_order, times_used DESC
                LIMIT ?
            """, (limit,)).fetchall()

            for row in curated:
                results.append({
                    'question': row['question'],
                    'category': row['category'],
                    'frequency': row['times_used'],
                })

            seen = set()
            unique = []
            for r in results:
                q = r['question'].lower().strip()
                if q not in seen:
                    seen.add(q)
                    unique.append(r)

            return unique[:limit * 2]
        finally:
            conn.close()

    def get_similar_queries(self, question: str, limit: int = 5) -> List[Dict]:
        normalized = self._normalize_question(question)
        index_key = self._make_index_key(normalized)
        key_words = index_key.split()

        if not key_words:
            return []

        conn = self._get_conn()
        try:
            placeholders = ' OR '.join(['INDEX_KEY LIKE ?' for _ in key_words])
            params = [f'%{w}%' for w in key_words]

            rows = conn.execute(f"""
                SELECT DISTINCT question, username, tables_used, intent,
                    COUNT(*) as cnt, MAX(created_at) as last
                FROM query_log
                WHERE ({placeholders})
                AND question != ?
                AND success = 1
                GROUP BY normalized_question
                ORDER BY cnt DESC
                LIMIT ?
            """, params + [question, limit]).fetchall()

            return [
                {
                    'question': row['question'],
                    'asked_by': row['username'] or 'anonymous',
                    'tables_used': json.loads(row['tables_used']) if row['tables_used'] else [],
                    'intent': row['intent'],
                    'frequency': row['cnt'],
                    'last_asked': row['last'],
                }
                for row in rows
            ]
        finally:
            conn.close()

    def set_context(self, user_id: str, session_id: str, key: str, value: Any):
        conn = self._get_conn()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO user_context (context_id, user_id, session_id, key, value, updated_at)
                VALUES (?, ?, ?, ?, ?, datetime('now'))
            """, (f"ctx_{secrets.token_hex(6)}", user_id, session_id, key, json.dumps(value)))
            conn.commit()
        finally:
            conn.close()

    def get_context(self, user_id: str, session_id: str) -> Dict:
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT key, value FROM user_context WHERE user_id = ? AND session_id = ? ORDER BY updated_at DESC",
                (user_id, session_id)
            ).fetchall()
            return {row['key']: json.loads(row['value']) for row in rows}
        finally:
            conn.close()

    def get_conversation_history(self, user_id: str, limit: int = 20) -> List[Dict]:
        conn = self._get_conn()
        try:
            rows = conn.execute("""
                SELECT question, sql_generated, tables_used, intent,
                    result_count, created_at, catalog_used, environment_used
                FROM query_log
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (user_id, limit)).fetchall()

            return [
                {
                    'question': row['question'],
                    'sql': row['sql_generated'],
                    'tables': json.loads(row['tables_used']) if row['tables_used'] else [],
                    'intent': row['intent'],
                    'result_count': row['result_count'],
                    'time': row['created_at'],
                    'catalog': row['catalog_used'],
                    'environment': row['environment_used'],
                }
                for row in rows
            ]
        finally:
            conn.close()

    def set_preference(self, user_id: str, key: str, value: Any):
        conn = self._get_conn()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO user_preferences (user_id, preference_key, preference_value, updated_at)
                VALUES (?, ?, ?, datetime('now'))
            """, (user_id, key, json.dumps(value)))
            conn.commit()
        finally:
            conn.close()

    def get_preferences(self, user_id: str) -> Dict:
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT preference_key, preference_value FROM user_preferences WHERE user_id = ?",
                (user_id,)
            ).fetchall()
            return {row['preference_key']: json.loads(row['preference_value']) for row in rows}
        finally:
            conn.close()

    def get_learning_stats(self) -> Dict:
        conn = self._get_conn()
        try:
            total_queries = conn.execute("SELECT COUNT(*) FROM query_log").fetchone()[0]
            unique_users = conn.execute("SELECT COUNT(DISTINCT user_id) FROM query_log").fetchone()[0]
            unique_patterns = conn.execute("SELECT COUNT(*) FROM query_patterns").fetchone()[0]
            suggestions = conn.execute("SELECT COUNT(*) FROM suggestions WHERE is_active=1").fetchone()[0]

            top_tables_raw = conn.execute("""
                SELECT tables_used, COUNT(*) as cnt
                FROM query_log WHERE tables_used != '[]'
                GROUP BY tables_used ORDER BY cnt DESC LIMIT 5
            """).fetchall()

            top_intents = conn.execute("""
                SELECT intent, COUNT(*) as cnt
                FROM query_log WHERE intent != ''
                GROUP BY intent ORDER BY cnt DESC LIMIT 5
            """).fetchall()

            return {
                'total_queries': total_queries,
                'unique_users': unique_users,
                'unique_patterns': unique_patterns,
                'active_suggestions': suggestions,
                'top_tables': [
                    {'tables': json.loads(r['tables_used']), 'count': r['cnt']}
                    for r in top_tables_raw
                ],
                'top_intents': [
                    {'intent': r['intent'], 'count': r['cnt']}
                    for r in top_intents
                ],
            }
        finally:
            conn.close()
