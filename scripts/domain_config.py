import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict

from semantic_layer import SchemaLearner, ColumnProfile

logger = logging.getLogger('gpdm.domain_config')


class DomainConfig:

    def __init__(self, learner: SchemaLearner, config_path: Optional[str] = None):
        self.learner = learner
        self.config_path = config_path

        self.concept_keywords: Dict[str, List[str]] = {}
        self.entity_patterns: Dict[str, Dict[str, Any]] = {}
        self.time_dimension_registry: Dict[str, List[ColumnProfile]] = {}
        self.rate_candidates: Dict[str, List[Tuple[str, List[str]]]] = {}
        self.join_registry: Dict[Tuple[str, str], str] = {}

        self.overrides: Dict[str, Any] = {}
        self.benchmarks: Dict[str, Dict[str, Any]] = {}
        self.benchmark_keyword_map: Dict[str, str] = {}
        self.rate_context: Dict[str, Dict[str, Any]] = {}
        self.recommendations: Dict[str, List[str]] = {}

        self._column_to_concept: Dict[str, str] = {}
        self._column_to_entity: Dict[str, str] = {}

        self._time_units = {
            'year', 'month', 'week', 'day', 'hour', 'minute', 'second',
            'annual', 'monthly', 'weekly', 'daily', 'hourly',
            'yearly', 'quarterly', 'semi-annual',
        }

        self._auto_discover()
        self._load_overrides()
        self._build_reverse_lookups()

        logger.info(
            "DomainConfig initialized: %d concepts, %d entities, %d time dims, %d rate candidates",
            len(self.concept_keywords),
            len(self.entity_patterns),
            sum(len(v) for v in self.time_dimension_registry.values()),
            sum(len(v) for v in self.rate_candidates.values()),
        )

    def _auto_discover(self) -> None:
        logger.debug("Starting auto-discovery from SchemaLearner")

        self._discover_amount_concepts()
        self._discover_entity_types()
        self._discover_time_dimensions()
        self._discover_rate_candidates()
        self._discover_joins()

    def _discover_amount_concepts(self) -> None:
        logger.debug("Discovering amount concepts from numeric columns")

        concept_map: Dict[str, Set[str]] = defaultdict(set)

        for table, profiles in self.learner.tables.items():
            for profile in profiles:
                if not profile.is_numeric:
                    continue

                col_lower = profile.name.lower()

                if 'currency' in profile.semantic_tags:
                    if any(w in col_lower for w in ['paid', 'payment', 'reimburse', 'allowed']):
                        concept_map['paid_amount'].add(profile.name)
                    elif any(w in col_lower for w in ['billed', 'charge', 'bill']):
                        concept_map['billed_amount'].add(profile.name)
                    elif any(w in col_lower for w in ['cost', 'expense']):
                        concept_map['cost'].add(profile.name)
                    elif any(w in col_lower for w in ['revenue', 'income']):
                        concept_map['revenue'].add(profile.name)
                    elif any(w in col_lower for w in ['fee', 'copay', 'deductible']):
                        concept_map['fee'].add(profile.name)
                    else:
                        concept_map['amount'].add(profile.name)

                elif 'count' in profile.semantic_tags:
                    if any(w in col_lower for w in ['claim', 'encounter', 'visit']):
                        concept_map['visit_count'].add(profile.name)
                    elif any(w in col_lower for w in ['member', 'patient', 'provider']):
                        concept_map['entity_count'].add(profile.name)
                    elif any(w in col_lower for w in ['procedure', 'service']):
                        concept_map['procedure_count'].add(profile.name)
                    else:
                        concept_map['count'].add(profile.name)

                elif 'rate' in profile.semantic_tags or any(w in col_lower for w in ['rate', 'ratio', 'percent']):
                    if any(w in col_lower for w in ['denial', 'reject']):
                        concept_map['denial_rate'].add(profile.name)
                    elif any(w in col_lower for w in ['approval', 'approve']):
                        concept_map['approval_rate'].add(profile.name)
                    elif any(w in col_lower for w in ['readmit']):
                        concept_map['readmission_rate'].add(profile.name)
                    elif any(w in col_lower for w in ['no.?show', 'noshow']):
                        concept_map['no_show_rate'].add(profile.name)
                    else:
                        concept_map['rate'].add(profile.name)

                if not concept_map or profile.name not in [
                    col for cols in concept_map.values() for col in cols
                ]:
                    concept_map['numeric'].add(profile.name)

        self.concept_keywords = {
            concept: sorted(list(keywords))
            for concept, keywords in concept_map.items()
        }

        logger.debug("Discovered %d concepts: %s", len(self.concept_keywords),
                    ', '.join(self.concept_keywords.keys()))

    def _discover_entity_types(self) -> None:
        logger.debug("Discovering entity types from categorical columns")

        for table, profiles in self.learner.tables.items():
            for profile in profiles:
                if not profile.is_categorical:
                    continue

                col_lower = profile.name.lower()

                entity_type = 'category'

                if 'categorical' in profile.semantic_tags or 'status' in col_lower:
                    entity_type = 'status'
                    if any(w in col_lower for w in ['flag', 'indicator']):
                        entity_type = 'flag'

                elif 'demographic' in profile.semantic_tags:
                    entity_type = 'demographic'
                    if any(w in col_lower for w in ['gender', 'sex']):
                        entity_type = 'gender'
                    elif any(w in col_lower for w in ['race', 'ethnicity']):
                        entity_type = 'race_ethnicity'
                    elif any(w in col_lower for w in ['language']):
                        entity_type = 'language'

                elif 'location' in profile.semantic_tags:
                    entity_type = 'location'

                elif 'code' in profile.semantic_tags:
                    entity_type = 'code'
                    if any(w in col_lower for w in ['icd', 'diagnosis']):
                        entity_type = 'icd_code'
                    elif any(w in col_lower for w in ['cpt', 'procedure']):
                        entity_type = 'cpt_code'
                    elif any(w in col_lower for w in ['ndc', 'drug']):
                        entity_type = 'ndc_code'

                distinct_values = sorted(set(profile.sample_values[:50])) if profile.sample_values else []

                self.entity_patterns[f"{table}.{profile.name}"] = {
                    'table': table,
                    'column': profile.name,
                    'entity_type': entity_type,
                    'values': distinct_values,
                    'distinct_count': profile.distinct_count,
                    'semantic_tags': profile.semantic_tags,
                }

        logger.debug("Discovered %d entity patterns", len(self.entity_patterns))

    def _discover_time_dimensions(self) -> None:
        logger.debug("Discovering time dimensions from date columns")

        for table, profiles in self.learner.tables.items():
            date_columns = [p for p in profiles if p.is_date]
            if date_columns:
                self.time_dimension_registry[table] = sorted(
                    date_columns,
                    key=lambda p: self._score_date_column(p),
                    reverse=True
                )

        logger.debug("Discovered time dimensions in %d tables",
                    len(self.time_dimension_registry))

    def _score_date_column(self, profile: ColumnProfile) -> float:
        score = 0.0
        col_lower = profile.name.lower()

        if any(w in col_lower for w in ['service', 'encounter', 'visit', 'claim']):
            score += 10.0
        elif any(w in col_lower for w in ['submit', 'start', 'begin', 'open']):
            score += 8.0
        elif any(w in col_lower for w in ['end', 'close', 'complete', 'discharge']):
            score += 7.0
        elif any(w in col_lower for w in ['created', 'updated', 'modified']):
            score += 4.0
        elif any(w in col_lower for w in ['adjudic', 'process']):
            score += 6.0

        score += (100.0 - profile.null_pct) / 100.0

        return score

    def _discover_rate_candidates(self) -> None:
        logger.debug("Discovering rate computation candidates")

        for table, profiles in self.learner.tables.items():
            candidates = []

            for profile in profiles:
                if not profile.is_categorical:
                    continue

                col_lower = profile.name.lower()

                is_status = any(w in col_lower for w in [
                    'status', 'state', 'phase', 'stage', 'flag',
                    'outcome', 'result', 'decision', 'approval'
                ])

                if is_status and profile.distinct_count <= 20 and profile.sample_values:
                    values = [str(v).upper() for v in profile.sample_values]
                    candidates.append((profile.name, values))

            if candidates:
                self.rate_candidates[table] = candidates

        logger.debug("Discovered %d rate candidate tables",
                    len(self.rate_candidates))

    def _discover_joins(self) -> None:
        logger.debug("Discovering join relationships")

        for from_table, targets in self.learner.join_graph.items():
            for to_table, join_col in targets.items():
                if from_table < to_table:
                    self.join_registry[(from_table, to_table)] = join_col

        logger.debug("Discovered %d join paths", len(self.join_registry))

    def _load_overrides(self) -> None:
        if not self.config_path or not os.path.exists(self.config_path):
            logger.debug("No config file found at %s", self.config_path)
            return

        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)

            self.overrides = config

            if 'concepts' in config:
                for concept, keywords in config['concepts'].items():
                    self.concept_keywords[concept] = keywords

            if 'entities' in config:
                for entity_name, entity_config in config['entities'].items():
                    for col_key, pattern_data in entity_config.items():
                        if col_key in self.entity_patterns:
                            self.entity_patterns[col_key].update(pattern_data)

            if 'benchmarks' in config:
                self.benchmarks = config['benchmarks']

            if 'benchmark_keyword_map' in config:
                self.benchmark_keyword_map = config['benchmark_keyword_map']

            if 'rate_context' in config:
                self.rate_context = config['rate_context']

            if 'recommendations' in config:
                self.recommendations = config['recommendations']

            logger.info("Loaded config overrides from %s", self.config_path)

        except Exception as e:
            logger.warning("Failed to load config from %s: %s", self.config_path, e)

    def _build_reverse_lookups(self) -> None:
        for concept, keywords in self.concept_keywords.items():
            for keyword in keywords:
                self._column_to_concept[keyword.upper()] = concept

        for col_key, entity_info in self.entity_patterns.items():
            col_name = entity_info['column']
            self._column_to_entity[col_name.upper()] = entity_info['entity_type']


    def find_metric_tables(self, question: str) -> List[str]:
        question_lower = question.lower()
        relevant_tables = set()

        for concept, keywords in self.concept_keywords.items():
            for keyword in keywords:
                if keyword.lower() in question_lower:
                    for table, profiles in self.learner.tables.items():
                        for profile in profiles:
                            if profile.name.lower() == keyword.lower():
                                relevant_tables.add(table)

        return sorted(list(relevant_tables))

    def find_concept_from_question(self, question: str) -> List[Tuple[str, str, str]]:
        question_lower = question.lower()
        results = []

        import re
        words = set(re.findall(r'\b[a-z_]+\b', question_lower))

        for concept, keywords in self.concept_keywords.items():
            for keyword in keywords:
                keyword_lower = keyword.lower()

                if keyword_lower in question_lower:
                    for table, profiles in self.learner.tables.items():
                        for profile in profiles:
                            if profile.name.lower() == keyword_lower:
                                results.append((concept, table, profile.name))

        seen = set()
        unique_results = []
        for item in results:
            if item not in seen:
                seen.add(item)
                unique_results.append(item)

        return unique_results

    def find_entity_type(self, column_name: str) -> str:
        col_upper = column_name.upper()
        return self._column_to_entity.get(col_upper, "unknown")

    def find_date_columns(self, table: str) -> List[ColumnProfile]:
        return self.time_dimension_registry.get(table, [])

    def find_status_columns(self, table: str) -> List[Tuple[str, List[str]]]:
        return self.rate_candidates.get(table, [])

    def find_join_path(self, from_table: str, to_table: str) -> Optional[str]:
        key = tuple(sorted([from_table, to_table]))
        return self.join_registry.get(key)

    def get_benchmark(self, metric_key: str) -> Optional[Dict]:
        return self.benchmarks.get(metric_key)

    def get_amount_concepts(self) -> Dict[str, List[str]]:
        return self.concept_keywords.copy()

    def is_amount_word(self, word: str) -> bool:
        word_lower = word.lower()

        if word_lower in self.concept_keywords:
            return True

        for keywords in self.concept_keywords.values():
            if word_lower in [kw.lower() for kw in keywords]:
                return True

        return False

    def get_time_units(self) -> Set[str]:
        return self._time_units.copy()

    def get_concept_for_column(self, column_name: str) -> Optional[str]:
        col_upper = column_name.upper()
        return self._column_to_concept.get(col_upper)

    def get_entities_by_type(self, entity_type: str) -> List[Dict]:
        return [
            pattern for pattern in self.entity_patterns.values()
            if pattern['entity_type'] == entity_type
        ]

    def find_column_in_table(self, table: str, concept: str) -> Optional[ColumnProfile]:
        if concept not in self.concept_keywords:
            return None

        keywords = self.concept_keywords[concept]
        table_profiles = self.learner.tables.get(table, [])

        for profile in table_profiles:
            if profile.name.lower() in [kw.lower() for kw in keywords]:
                return profile

        return None

    def find_tables_with_concept(self, concept: str) -> List[str]:
        if concept not in self.concept_keywords:
            return []

        keywords = self.concept_keywords[concept]
        tables = set()

        for table, profiles in self.learner.tables.items():
            for profile in profiles:
                if profile.name.lower() in [kw.lower() for kw in keywords]:
                    tables.add(table)

        return sorted(list(tables))

    def get_schema_summary(self) -> Dict[str, Any]:
        return {
            'concepts': self.concept_keywords,
            'entity_types': {
                col_key: {
                    'type': info['entity_type'],
                    'distinct_count': info['distinct_count'],
                    'sample_values': info['values'][:10],
                }
                for col_key, info in self.entity_patterns.items()
            },
            'time_dimensions': {
                table: [p.name for p in cols]
                for table, cols in self.time_dimension_registry.items()
            },
            'rate_candidates': self.rate_candidates,
            'join_paths': {
                f"{t1}→{t2}": col
                for (t1, t2), col in self.join_registry.items()
            },
            'benchmarks': self.benchmarks,
        }

    def find_benchmark_for_question(self, question: str) -> Optional[Tuple[str, Dict]]:
        q_lower = question.lower()
        for keyword, bench_key in self.benchmark_keyword_map.items():
            if keyword in q_lower:
                bench = self.benchmarks.get(bench_key)
                if bench:
                    return (bench_key, bench)
        return None

    def get_rate_context(self, domain: str) -> Optional[Dict]:
        return self.rate_context.get(domain)

    def get_recommendations(self, category: str) -> List[str]:
        return self.recommendations.get(category, [])

    def classify_domain_from_columns(self, columns: List[str]) -> Set[str]:
        domains = set()
        col_text = ' '.join(c.lower().replace('_', ' ') for c in columns)

        for concept in self.concept_keywords:
            if any(kw.lower() in col_text for kw in self.concept_keywords[concept]):
                if concept in ('cost', 'paid_amount', 'billed_amount', 'revenue',
                               'fee', 'amount'):
                    domains.add('financial')

        for table, candidates in self.rate_candidates.items():
            for col_name, _values in candidates:
                if col_name.lower() in col_text:
                    domains.add('clinical')

        for _key, info in self.entity_patterns.items():
            if info['column'].lower() in col_text:
                if info['entity_type'] in ('demographic', 'gender', 'race_ethnicity'):
                    domains.add('clinical')
                elif info['entity_type'] in ('location',):
                    domains.add('operational')

        for table in self.time_dimension_registry:
            for dc in self.time_dimension_registry[table]:
                if dc.name.lower() in col_text:
                    domains.add('operational')

        if not domains:
            domains.add('general')

        return domains

    def __repr__(self) -> str:
        return (
            f"DomainConfig("
            f"concepts={len(self.concept_keywords)}, "
            f"entities={len(self.entity_patterns)}, "
            f"joins={len(self.join_registry)}"
            f")"
        )


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python domain_config.py <db_path> [config_path]")
        sys.exit(1)

    db_path = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) > 2 else None

    logging.basicConfig(
        level=logging.INFO,
        format='%(name)s - %(levelname)s - %(message)s'
    )

    learner = SchemaLearner(db_path)
    schema_info = learner.learn()
    print(f"Learned schema: {schema_info}")

    config = DomainConfig(learner, config_path)
    print(f"\nDomainConfig: {config}")

    summary = config.get_schema_summary()
    print(f"\nConcepts discovered: {list(summary['concepts'].keys())}")
    print(f"Entity types: {set(info['type'] for info in summary['entity_types'].values())}")
    print(f"Time dimensions: {len(summary['time_dimensions'])} tables")
    print(f"Join paths: {len(summary['join_paths'])}")

    print("\n--- API Examples ---")
    if summary['concepts']:
        first_concept = list(summary['concepts'].keys())[0]
        print(f"Tables with '{first_concept}': {config.find_tables_with_concept(first_concept)}")

    if list(config.learner.tables.keys()):
        first_table = list(config.learner.tables.keys())[0]
        print(f"Date columns in '{first_table}': {[p.name for p in config.find_date_columns(first_table)]}")
