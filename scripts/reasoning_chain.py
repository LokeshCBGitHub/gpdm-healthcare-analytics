import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from enum import Enum


class StageType(str, Enum):
    INTENT_DETECTION = "intent_detection"
    TABLE_SELECTION = "table_selection"
    COLUMN_RESOLUTION = "column_resolution"
    JOIN_PLANNING = "join_planning"
    AGGREGATION_SELECTION = "aggregation_selection"
    FILTER_DETECTION = "filter_detection"
    SQL_GENERATION = "sql_generation"
    QUALITY_GATE = "quality_gate"
    COMPETITION = "competition"
    EXECUTION = "execution"
    SELF_HEALING = "self_healing"
    ANOMALY_DETECTION = "anomaly_detection"
    POST_VALIDATION = "post_validation"


@dataclass
class ReasoningStep:
    stage: str
    decision: str
    evidence: List[str]
    confidence: float
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    rejected_reasons: List[str] = field(default_factory=list)
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ReasoningChain:

    def __init__(self, question: str):
        self.question = question
        self.steps: List[ReasoningStep] = []
        self.start_time = time.time()
        self._confidence_scores: Dict[str, float] = {}

    def add_step(self,
                 stage: str,
                 decision: str,
                 evidence: List[str],
                 confidence: float,
                 alternatives: Optional[List[Dict[str, Any]]] = None,
                 rejected_reasons: Optional[List[str]] = None,
                 duration_ms: float = 0.0,
                 notes: Optional[str] = None) -> None:
        step = ReasoningStep(
            stage=stage,
            decision=decision,
            evidence=evidence or [],
            confidence=confidence,
            alternatives=alternatives or [],
            rejected_reasons=rejected_reasons or [],
            duration_ms=duration_ms,
            timestamp=time.time(),
            notes=notes,
        )
        self.steps.append(step)
        self._confidence_scores[stage] = confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            'question': self.question,
            'total_duration_ms': (time.time() - self.start_time) * 1000,
            'step_count': len(self.steps),
            'steps': [step.to_dict() for step in self.steps],
            'confidence_breakdown': self.get_confidence_breakdown(),
            'overall_confidence': self.get_overall_confidence(),
        }

    def to_narrative(self) -> str:
        if not self.steps:
            return f"No reasoning steps captured for query: {self.question}"

        lines = [f"THOUGHT PROCESS for \"{self.question}\":\n"]

        for i, step in enumerate(self.steps, 1):
            stage_name = self._format_stage_name(step.stage)
            lines.append(f"{i}. {stage_name}:")

            lines.append(f"   Decision: {step.decision} (confidence: {step.confidence:.2f})")

            if step.evidence:
                evidence_text = ", ".join(step.evidence[:3])
                if len(step.evidence) > 3:
                    evidence_text += f", ... ({len(step.evidence)} total)"
                lines.append(f"   Evidence: {evidence_text}")

            if step.alternatives:
                alt_lines = []
                for alt in step.alternatives[:2]:
                    alt_name = alt.get('name', alt.get('intent', alt.get('option', '?')))
                    alt_conf = alt.get('confidence', alt.get('score', 0))
                    alt_lines.append(f"{alt_name} ({alt_conf:.2f})")
                if len(step.alternatives) > 2:
                    alt_lines.append(f"... {len(step.alternatives) - 2} others")
                lines.append(f"   Alternatives considered: {', '.join(alt_lines)}")

            if step.rejected_reasons:
                for reason in step.rejected_reasons[:2]:
                    lines.append(f"   Rejected: {reason}")

            lines.append("")

        lines.append("CONFIDENCE BREAKDOWN:")
        breakdown = self.get_confidence_breakdown()
        for stage, conf in breakdown.items():
            stage_name = self._format_stage_name(stage)
            bar = self._make_confidence_bar(conf)
            lines.append(f"  {stage_name:.<25} {bar} {conf:.2f}")

        lines.append(f"\nOverall Confidence: {self.get_overall_confidence():.2f}")
        lines.append(f"Total Duration: {(time.time() - self.start_time) * 1000:.0f}ms")

        return "\n".join(lines)

    def get_confidence_breakdown(self) -> Dict[str, float]:
        return dict(self._confidence_scores)

    def get_overall_confidence(self) -> float:
        weights = {
            'intent_detection': 0.20,
            'table_selection': 0.20,
            'column_resolution': 0.15,
            'sql_generation': 0.15,
            'quality_gate': 0.15,
            'filter_detection': 0.10,
            'aggregation_selection': 0.05,
            'execution': 0.05,
            'join_planning': 0.10,
            'competition': 0.05,
            'self_healing': 0.03,
            'anomaly_detection': 0.02,
        }

        total_weight = 0.0
        weighted_sum = 0.0

        for stage, conf in self._confidence_scores.items():
            weight = weights.get(stage, 0.05)
            weighted_sum += conf * weight
            total_weight += weight

        if total_weight == 0:
            return 0.5

        return min(1.0, weighted_sum / total_weight)

    def get_step(self, stage: str) -> Optional[ReasoningStep]:
        for step in self.steps:
            if step.stage == stage:
                return step
        return None

    def get_steps_by_type(self, stage_type: str) -> List[ReasoningStep]:
        return [s for s in self.steps if s.stage == stage_type]

    @staticmethod
    def _format_stage_name(stage: str) -> str:
        replacements = {
            'intent_detection': 'UNDERSTANDING',
            'table_selection': 'TABLE SELECTION',
            'column_resolution': 'COLUMN MAPPING',
            'join_planning': 'JOIN PLANNING',
            'aggregation_selection': 'AGGREGATION',
            'filter_detection': 'FILTER DETECTION',
            'sql_generation': 'SQL CONSTRUCTION',
            'quality_gate': 'VALIDATION',
            'competition': 'COMPETITION',
            'execution': 'EXECUTION',
            'self_healing': 'SELF-HEALING',
            'anomaly_detection': 'ANOMALY CHECK',
            'post_validation': 'POST-VALIDATION',
        }
        return replacements.get(stage, stage.replace('_', ' ').upper())

    @staticmethod
    def _make_confidence_bar(confidence: float, width: int = 10) -> str:
        filled = int(confidence * width)
        bar = '█' * filled + '░' * (width - filled)
        return bar


class ReasoningChainRecorder:

    def __init__(self, chain: ReasoningChain):
        self.chain = chain

    def intent_detection(self,
                        intent_obj: Dict[str, Any],
                        alternatives: Optional[List[tuple]] = None,
                        duration_ms: float = 0.0) -> None:
        decision = intent_obj.get('intent', '?')
        confidence = intent_obj.get('confidence', 0.5)

        evidence = []
        if 'keywords' in intent_obj:
            evidence.append(f"keywords: {', '.join(intent_obj['keywords'][:3])}")
        if 'explanation' in intent_obj:
            evidence.append(intent_obj['explanation'])

        alts = []
        if alternatives:
            for name, score in alternatives:
                alts.append({'intent': name, 'confidence': score})

        self.chain.add_step(
            stage='intent_detection',
            decision=decision,
            evidence=evidence or ['intent classification'],
            confidence=confidence,
            alternatives=alts,
            duration_ms=duration_ms,
        )

    def table_selection(self,
                       selected_table: str,
                       confidence: float,
                       alternatives: Optional[List[tuple]] = None,
                       evidence: Optional[List[str]] = None,
                       rejected_reasons: Optional[List[str]] = None,
                       duration_ms: float = 0.0) -> None:
        alts = []
        if alternatives:
            for name, score in alternatives:
                alts.append({'table': name, 'confidence': score})

        self.chain.add_step(
            stage='table_selection',
            decision=selected_table,
            evidence=evidence or [f'selected: {selected_table}'],
            confidence=confidence,
            alternatives=alts,
            rejected_reasons=rejected_reasons or [],
            duration_ms=duration_ms,
        )

    def column_resolution(self,
                         resolved_columns: Dict[str, str],
                         confidence: float,
                         evidence: Optional[List[str]] = None,
                         rejected_columns: Optional[List[tuple]] = None,
                         duration_ms: float = 0.0) -> None:
        rejected = []
        if rejected_columns:
            for col_name, score, reason in rejected_columns:
                rejected.append(f"{col_name} ({score:.1f}) — {reason}")

        column_str = ", ".join([f"{k}→{v}" for k, v in resolved_columns.items()])

        self.chain.add_step(
            stage='column_resolution',
            decision=column_str,
            evidence=evidence or [f'resolved {len(resolved_columns)} columns'],
            confidence=confidence,
            rejected_reasons=rejected,
            duration_ms=duration_ms,
        )

    def join_planning(self,
                     join_decisions: str,
                     confidence: float,
                     evidence: Optional[List[str]] = None,
                     rejected_joins: Optional[List[str]] = None,
                     duration_ms: float = 0.0) -> None:
        self.chain.add_step(
            stage='join_planning',
            decision=join_decisions,
            evidence=evidence or ['join strategy determined'],
            confidence=confidence,
            rejected_reasons=rejected_joins or [],
            duration_ms=duration_ms,
        )

    def aggregation_selection(self,
                             agg_function: str,
                             confidence: float,
                             evidence: Optional[List[str]] = None,
                             alternatives: Optional[List[tuple]] = None,
                             duration_ms: float = 0.0) -> None:
        alts = []
        if alternatives:
            for func_name, score in alternatives:
                alts.append({'function': func_name, 'confidence': score})

        self.chain.add_step(
            stage='aggregation_selection',
            decision=agg_function,
            evidence=evidence or [f'selected aggregation: {agg_function}'],
            confidence=confidence,
            alternatives=alts,
            duration_ms=duration_ms,
        )

    def filter_detection(self,
                        filter_clause: str,
                        confidence: float,
                        evidence: Optional[List[str]] = None,
                        duration_ms: float = 0.0) -> None:
        self.chain.add_step(
            stage='filter_detection',
            decision=filter_clause or 'no filters',
            evidence=evidence or ['filter analysis complete'],
            confidence=confidence,
            duration_ms=duration_ms,
        )

    def sql_generation(self,
                      sql: str,
                      confidence: float,
                      evidence: Optional[List[str]] = None,
                      engine_name: str = 'semantic',
                      duration_ms: float = 0.0) -> None:
        sql_preview = sql[:100].replace('\n', ' ') + ('...' if len(sql) > 100 else '')

        self.chain.add_step(
            stage='sql_generation',
            decision=sql_preview,
            evidence=evidence or [f'generated via {engine_name} engine'],
            confidence=confidence,
            duration_ms=duration_ms,
            notes=f'engine: {engine_name}',
        )

    def quality_gate(self,
                    result: str,
                    confidence: float,
                    checks_passed: int,
                    total_checks: int,
                    violations: Optional[List[str]] = None,
                    duration_ms: float = 0.0) -> None:
        decision = f"{checks_passed}/{total_checks} checks passed"
        evidence = [f"All validation checks completed"]

        self.chain.add_step(
            stage='quality_gate',
            decision=decision,
            evidence=evidence,
            confidence=confidence,
            rejected_reasons=violations or [],
            duration_ms=duration_ms,
            notes=result,
        )

    def competition(self,
                   winner: str,
                   winner_confidence: float,
                   loser: str,
                   loser_confidence: float,
                   reason: str,
                   duration_ms: float = 0.0) -> None:
        self.chain.add_step(
            stage='competition',
            decision=f"{winner} won",
            evidence=[reason],
            confidence=winner_confidence,
            alternatives=[{'engine': loser, 'confidence': loser_confidence}],
            rejected_reasons=[f"{loser} rejected: {reason}"],
            duration_ms=duration_ms,
        )

    def execution(self,
                 status: str,
                 row_count: int,
                 confidence: float,
                 evidence: Optional[List[str]] = None,
                 sanity_checks: Optional[List[str]] = None,
                 duration_ms: float = 0.0) -> None:
        self.chain.add_step(
            stage='execution',
            decision=f"{status} ({row_count} rows)",
            evidence=evidence or ['query executed successfully'],
            confidence=confidence,
            rejected_reasons=sanity_checks or [],
            duration_ms=duration_ms,
        )


@dataclass
class QueryValidationResult:
    validated: bool
    confidence: float
    tables_match_blueprint: bool
    aggregation_correct: bool
    corrections: List[str] = field(default_factory=list)
    mismatches: List[str] = field(default_factory=list)
    path_scores: Dict[str, float] = field(default_factory=dict)
    blueprint: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None


class ReasoningPreValidator:

    def __init__(self, knowledge_graph):
        self.kg = knowledge_graph
        self.validation_log: List[QueryValidationResult] = []

    def validate_query_direction(self,
                                 question: str,
                                 detected_intent: Dict[str, Any],
                                 candidate_tables: List[str],
                                 candidate_columns: Dict[str, str],
                                 aggregation_type: Optional[str] = None,
                                 confidence_threshold: float = 0.65) -> QueryValidationResult:
        start_time = time.time()

        blueprint = self.kg.generate_query_blueprint(question)

        tables_match = self._validate_tables(
            candidate_tables=candidate_tables,
            blueprint_tables=blueprint['tables']
        )

        aggregation_match = self._validate_aggregation(
            proposed_agg=aggregation_type,
            blueprint_aggs=blueprint['aggregations'],
            intent=detected_intent.get('intent', '')
        )

        columns_match = self._validate_columns(
            candidate_columns=candidate_columns,
            candidate_tables=candidate_tables,
            blueprint_tables=blueprint['tables']
        )

        mismatches = []
        corrections = []

        if not tables_match['valid']:
            mismatches.append(tables_match['reason'])
            corrections.extend(tables_match['corrections'])

        if not aggregation_match['valid']:
            mismatches.append(aggregation_match['reason'])
            corrections.extend(aggregation_match['corrections'])

        if not columns_match['valid']:
            mismatches.append(columns_match['reason'])
            corrections.extend(columns_match['corrections'])

        component_confidences = [
            tables_match['confidence'],
            aggregation_match['confidence'],
            columns_match['confidence'],
            blueprint['confidence']
        ]
        overall_confidence = sum(component_confidences) / len(component_confidences)

        validated = (
            overall_confidence >= confidence_threshold and
            len(mismatches) == 0
        )

        path_scores = self._score_alternative_paths(
            question=question,
            candidate_tables=candidate_tables,
            blueprint=blueprint
        )

        result = QueryValidationResult(
            validated=validated,
            confidence=overall_confidence,
            tables_match_blueprint=tables_match['valid'],
            aggregation_correct=aggregation_match['valid'],
            corrections=corrections,
            mismatches=mismatches,
            path_scores=path_scores,
            blueprint=blueprint,
            notes=f"Validation completed in {(time.time()-start_time)*1000:.1f}ms. "
                  f"Threshold: {confidence_threshold:.2f}, Result: {overall_confidence:.2f}"
        )

        self.validation_log.append(result)
        return result

    def _validate_tables(self, candidate_tables: List[str], blueprint_tables: List[str]) -> Dict[str, Any]:
        if not candidate_tables or not blueprint_tables:
            return {
                'valid': False,
                'confidence': 0.0,
                'reason': 'Missing table information',
                'corrections': ['Ensure tables are properly identified']
            }

        candidate_set = set(t.upper() for t in candidate_tables)
        blueprint_set = set(t.upper() for t in blueprint_tables)

        if candidate_set == blueprint_set:
            return {
                'valid': True,
                'confidence': 1.0,
                'reason': 'Tables match perfectly',
                'corrections': []
            }

        if blueprint_set.issubset(candidate_set):
            return {
                'valid': True,
                'confidence': 0.9,
                'reason': 'Candidate tables include all required tables',
                'corrections': []
            }

        if candidate_set.issubset(blueprint_set):
            missing = blueprint_set - candidate_set
            return {
                'valid': False,
                'confidence': 0.4,
                'reason': f'Missing tables: {", ".join(missing)}',
                'corrections': [f'Include table(s): {", ".join(missing)}']
            }

        overlap = candidate_set & blueprint_set
        if overlap:
            confidence = len(overlap) / len(blueprint_set)
            wrong_candidate = candidate_set - blueprint_set
            return {
                'valid': False,
                'confidence': confidence,
                'reason': f'Unexpected tables: {", ".join(wrong_candidate)}',
                'corrections': [
                    f'Use tables: {", ".join(blueprint_set)} instead of {", ".join(candidate_set)}'
                ]
            }

        return {
            'valid': False,
            'confidence': 0.0,
            'reason': f'No overlap between candidate {candidate_set} and expected {blueprint_set}',
            'corrections': [f'Use tables: {", ".join(blueprint_set)}']
        }

    def _validate_aggregation(self, proposed_agg: Optional[str],
                             blueprint_aggs: List[str],
                             intent: str) -> Dict[str, Any]:
        if intent != 'aggregation' and not proposed_agg:
            return {
                'valid': True,
                'confidence': 1.0,
                'reason': 'No aggregation required for this intent',
                'corrections': []
            }

        if intent == 'aggregation' and not proposed_agg:
            return {
                'valid': False,
                'confidence': 0.2,
                'reason': 'Aggregation intent detected but no aggregation function specified',
                'corrections': [f'Add aggregation function from: {", ".join(blueprint_aggs or ["COUNT", "SUM", "AVG", "MAX", "MIN"])}']
            }

        if not blueprint_aggs:
            return {
                'valid': True,
                'confidence': 0.6,
                'reason': f'Using proposed aggregation {proposed_agg} (blueprint had no suggestion)',
                'corrections': []
            }

        if proposed_agg and proposed_agg.upper() in [a.upper() for a in blueprint_aggs]:
            return {
                'valid': True,
                'confidence': 1.0,
                'reason': f'{proposed_agg} matches blueprint expectations',
                'corrections': []
            }

        if proposed_agg:
            return {
                'valid': False,
                'confidence': 0.3,
                'reason': f'{proposed_agg} does not match expected aggregations: {", ".join(blueprint_aggs)}',
                'corrections': [f'Use one of: {", ".join(blueprint_aggs)}']
            }

        return {
            'valid': True,
            'confidence': 0.5,
            'reason': 'Aggregation validation inconclusive',
            'corrections': []
        }

    def _validate_columns(self, candidate_columns: Dict[str, str],
                         candidate_tables: List[str],
                         blueprint_tables: List[str]) -> Dict[str, Any]:
        if not candidate_columns:
            return {
                'valid': True,
                'confidence': 0.8,
                'reason': 'No specific columns required',
                'corrections': []
            }

        if not candidate_tables:
            return {
                'valid': False,
                'confidence': 0.0,
                'reason': 'Cannot validate columns without table context',
                'corrections': ['Specify tables first']
            }

        mismatched = []
        if isinstance(candidate_columns, list):
            col_items = [(c, c) for c in candidate_columns]
        else:
            col_items = candidate_columns.items()

        valid_columns = set()
        if hasattr(self, 'knowledge_graph') and self.knowledge_graph and hasattr(self.knowledge_graph, '_schema_cache'):
            for table in candidate_tables:
                table_info = self.knowledge_graph._schema_cache.get(table, {})
                for col_info in table_info.get('columns', []):
                    valid_columns.add(col_info['name'].upper())

        for user_term, resolved_col in col_items:
            if resolved_col:
                col_name = resolved_col.split('.')[-1].upper() if '.' in resolved_col else resolved_col.upper()
                if valid_columns and col_name not in valid_columns:
                    mismatched.append(f'{resolved_col} (not found in {candidate_tables})')

        if mismatched:
            return {
                'valid': False,
                'confidence': 0.4,
                'reason': f'Columns not found in schema: {", ".join(mismatched)}',
                'corrections': [f'Check column names against actual schema']
            }

        return {
            'valid': True,
            'confidence': 0.85,
            'reason': f'Columns appear valid for tables {candidate_tables}',
            'corrections': []
        }

    def _score_alternative_paths(self, question: str, candidate_tables: List[str],
                                 blueprint: Dict[str, Any]) -> Dict[str, float]:
        path_scores = {}

        blueprint_path = " -> ".join(blueprint['tables'])
        path_scores[f"Blueprint: {blueprint_path}"] = blueprint['confidence']

        if candidate_tables != blueprint['tables']:
            candidate_path = " -> ".join(candidate_tables)
            path_scores[f"Current: {candidate_path}"] = max(0.3, blueprint['confidence'] - 0.2)

        question_upper = question.upper()
        fallback_tables = []

        if 'MEMBER' in question_upper and 'MEMBERS' not in blueprint['tables']:
            fallback_tables.append('MEMBERS')
        if 'CLAIM' in question_upper and 'CLAIMS' not in blueprint['tables']:
            fallback_tables.append('CLAIMS')
        if 'PROVIDER' in question_upper and 'PROVIDERS' not in blueprint['tables']:
            fallback_tables.append('PROVIDERS')

        for table in fallback_tables[:2]:
            path_scores[f"Fallback: {table}"] = 0.4

        return path_scores

    def suggest_correction(self, validation: QueryValidationResult) -> Dict[str, Any]:
        if validation.validated:
            return {
                'status': 'valid',
                'message': 'Query direction is correct, proceed with SQL generation'
            }

        suggestion = {
            'status': 'invalid',
            'confidence': validation.confidence,
            'threshold': 0.65,
            'reasons_for_rejection': validation.mismatches,
            'suggested_corrections': validation.corrections,
            'blueprint': {
                'expected_tables': validation.blueprint['tables'] if validation.blueprint else [],
                'expected_aggregations': validation.blueprint['aggregations'] if validation.blueprint else [],
            },
            'alternative_paths': validation.path_scores,
            'recommendation': self._generate_recommendation(validation)
        }

        return suggestion

    def _generate_recommendation(self, validation: QueryValidationResult) -> str:
        if validation.confidence < 0.3:
            return "ABORT: Confidence too low. Suggest rephrasing the question or clarifying what you're looking for."
        elif validation.confidence < 0.5:
            return "CAUTION: Low confidence. Consider the suggested corrections before proceeding."
        elif validation.confidence < 0.65:
            return "PROCEED WITH CAUTION: Apply one or more corrections for better accuracy."
        else:
            return "PROCEED: Query direction appears sound."

    def get_validation_summary(self) -> Dict[str, Any]:
        if not self.validation_log:
            return {'validations_run': 0, 'message': 'No validations performed yet'}

        total = len(self.validation_log)
        passed = sum(1 for v in self.validation_log if v.validated)
        avg_confidence = sum(v.confidence for v in self.validation_log) / total if total > 0 else 0

        return {
            'validations_run': total,
            'passed': passed,
            'failed': total - passed,
            'pass_rate': passed / total if total > 0 else 0,
            'average_confidence': avg_confidence,
            'last_validation': {
                'validated': self.validation_log[-1].validated,
                'confidence': self.validation_log[-1].confidence,
                'mismatches': self.validation_log[-1].mismatches,
                'corrections': self.validation_log[-1].corrections,
            } if self.validation_log else None
        }


