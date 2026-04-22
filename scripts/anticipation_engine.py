import sqlite3
import json
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Any
from statistics import mean, stdev, quantiles
import math


class AnticipationEngine:

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.session_history: Dict[str, List[Dict]] = defaultdict(list)
        self.intent_transitions: Dict[str, Counter] = defaultdict(Counter)
        self.time_patterns: Dict[str, List] = defaultdict(list)
        self.kpi_correlations: Dict[Tuple[str, str], float] = {}
        self._load_historical_patterns()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _load_historical_patterns(self) -> None:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            self.kpi_correlations[("mlr", "pmpm")] = 0.82
            self.kpi_correlations[("pmpm", "mlr")] = 0.78

            self.kpi_correlations[("denial_rate", "appeal_rate")] = 0.76
            self.kpi_correlations[("appeal_rate", "denial_rate")] = 0.71

            self.kpi_correlations[("risk_score", "high_risk_members")] = 0.88
            self.kpi_correlations[("high_risk_members", "referrals")] = 0.79

            self.kpi_correlations[("retention_rate", "disenrollment_reasons")] = 0.73
            self.kpi_correlations[("net_enrollment", "revenue_impact")] = 0.81

            conn.close()
        except Exception as e:
            print(f"Warning: Could not load historical patterns: {e}")

    def record_interaction(
        self,
        session_id: str,
        query: str,
        intent: str,
        kpi_area: str,
        response_quality: float
    ) -> None:
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'intent': intent,
            'kpi_area': kpi_area,
            'quality': response_quality
        }
        self.session_history[session_id].append(interaction)

        if len(self.session_history[session_id]) > 1:
            prev_intent = self.session_history[session_id][-2]['intent']
            self.intent_transitions[prev_intent][intent] += 1

        hour = datetime.now().hour
        self.time_patterns[kpi_area].append({
            'hour': hour,
            'quality': response_quality,
            'intent': intent
        })

    def anticipate_next(self, session_id: str, current_context: Optional[Dict] = None) -> List[Dict]:
        if current_context is None:
            current_context = {}

        suggestions = []
        current_intent = current_context.get('current_intent', 'unknown')
        current_kpi = current_context.get('current_kpi', 'mlr')
        depth_level = current_context.get('depth_level', 'early')
        time_spent = current_context.get('time_spent_minutes', 0)

        if current_intent in self.intent_transitions:
            next_intents = self.intent_transitions[current_intent].most_common(2)
            for next_intent, count in next_intents:
                confidence = min(0.95, count / (count + 3))

                intent_to_query = {
                    'explore_mlr': 'Show MLR breakdown by region and product',
                    'investigate_anomaly': 'Identify anomalies in claims processing',
                    'drill_to_region': 'Focus on region with worst KPI',
                    'member_risk_analysis': 'Analyze high-risk member population',
                    'referral_analysis': 'Review specialty referral patterns',
                    'cost_reduction': 'Identify top cost drivers by service category'
                }

                suggestions.append({
                    'suggestion': f"Explore {next_intent.replace('_', ' ')}",
                    'confidence': confidence,
                    'reasoning': f"After {current_intent}, {int(confidence*100)}% of sessions continue with {next_intent}",
                    'category': 'follow_up',
                    'priority': 1,
                    'quick_query': intent_to_query.get(next_intent, '')
                })

        hour = datetime.now().hour
        if 6 <= hour < 10:
            if depth_level != 'early':
                suggestions.append({
                    'suggestion': 'Generate executive summary of key metrics',
                    'confidence': 0.72,
                    'reasoning': 'Morning sessions typically focus on high-level KPI summaries',
                    'category': 'context_aware',
                    'priority': 2,
                    'quick_query': 'Show executive summary: MLR, PMPM, denial rate, member count by region'
                })
        elif 14 <= hour < 17:
            if depth_level == 'early':
                suggestions.append({
                    'suggestion': 'Dive deeper into regional breakdown',
                    'confidence': 0.68,
                    'reasoning': 'Afternoon sessions tend toward detailed regional analysis',
                    'category': 'context_aware',
                    'priority': 2,
                    'quick_query': 'Show detailed metrics by region: claims, encounters, referrals'
                })

        if current_kpi in dict(self.kpi_correlations):
            correlated_kpis = [
                (kpi2, conf) for (kpi1, kpi2), conf in self.kpi_correlations.items()
                if kpi1 == current_kpi
            ]
            for kpi2, correlation in sorted(correlated_kpis, key=lambda x: x[1], reverse=True)[:2]:
                suggestions.append({
                    'suggestion': f"Review {kpi2.replace('_', ' ')} (related to {current_kpi})",
                    'confidence': correlation,
                    'reasoning': f"{int(correlation*100)}% of users explore {kpi2} after reviewing {current_kpi}",
                    'category': 'kpi_correlation',
                    'priority': 1 if correlation > 0.8 else 2,
                    'quick_query': f'Show {kpi2} analysis'
                })

        if depth_level == 'early' and time_spent > 10:
            suggestions.append({
                'suggestion': 'Ready for deeper analysis? Explore member cohort segmentation',
                'confidence': 0.65,
                'reasoning': 'Session duration suggests user ready for advanced features',
                'category': 'progression',
                'priority': 3,
                'quick_query': 'Show member segments by risk, age, and chronic conditions'
            })

        suggestions.sort(key=lambda x: (x['priority'], -x['confidence']))
        return suggestions[:5]

    def get_session_context(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self.session_history:
            return {
                'queries_asked': [],
                'kpis_viewed': {},
                'intents': [],
                'time_spent_minutes': 0,
                'depth_level': 'early',
                'avg_response_quality': 0.0,
                'suggested_next_steps': []
            }

        history = self.session_history[session_id]
        queries = [h['query'] for h in history]
        kpis = Counter([h['kpi_area'] for h in history])
        intents = [h['intent'] for h in history]
        qualities = [h['quality'] for h in history]

        if len(history) > 1:
            first_time = datetime.fromisoformat(history[0]['timestamp'])
            last_time = datetime.fromisoformat(history[-1]['timestamp'])
            time_spent = (last_time - first_time).total_seconds() / 60
        else:
            time_spent = 0

        unique_kpis = len(kpis)
        if unique_kpis <= 2:
            depth = 'early'
        elif unique_kpis <= 4:
            depth = 'mid'
        else:
            depth = 'deep'

        context = {
            'queries_asked': queries,
            'kpis_viewed': dict(kpis),
            'intents': intents,
            'time_spent_minutes': time_spent,
            'depth_level': depth,
            'avg_response_quality': mean(qualities) if qualities else 0.0,
            'num_interactions': len(history),
            'last_kpi': history[-1]['kpi_area'] if history else None
        }

        context['suggested_next_steps'] = self.anticipate_next(
            session_id,
            {
                'current_intent': intents[-1] if intents else 'unknown',
                'current_kpi': context['last_kpi'],
                'depth_level': depth,
                'time_spent_minutes': time_spent
            }
        )

        return context

    def get_cross_session_patterns(self) -> Dict[str, Any]:
        all_intents = [h['intent'] for history in self.session_history.values() for h in history]
        all_kpis = [h['kpi_area'] for history in self.session_history.values() for h in history]
        all_qualities = [h['quality'] for history in self.session_history.values() for h in history]

        all_hours = []
        for patterns in self.time_patterns.values():
            all_hours.extend([p['hour'] for p in patterns])
        peak_hours = Counter(all_hours).most_common(3) if all_hours else []

        confusion_points = {}
        for intent in set(all_intents):
            quality_scores = [
                h['quality'] for history in self.session_history.values()
                for h in history if h['intent'] == intent
            ]
            if quality_scores:
                avg_quality = mean(quality_scores)
                if avg_quality < 0.6:
                    confusion_points[intent] = avg_quality

        common_paths = []
        for intent, next_intents in self.intent_transitions.items():
            if next_intents.most_common(1):
                path = f"{intent} → {next_intents.most_common(1)[0][0]}"
                common_paths.append(path)

        return {
            'total_sessions': len(self.session_history),
            'total_interactions': sum(len(h) for h in self.session_history.values()),
            'most_common_query_paths': common_paths[:5],
            'popular_kpis': dict(Counter(all_kpis).most_common(8)),
            'peak_usage_hours': [(hour, count) for hour, count in peak_hours],
            'common_confusion_points': confusion_points,
            'avg_session_quality': mean(all_qualities) if all_qualities else 0.0,
            'intent_distribution': dict(Counter(all_intents).most_common(10))
        }


class UncertaintyHandler:

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.schema = self._load_schema()
        self.disambiguation_rules = self._build_disambiguation_rules()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _load_schema(self) -> Dict[str, List[str]]:
        schema = {}
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            for table in tables:
                cursor.execute(f"PRAGMA table_info({table})")
                columns = [row[1] for row in cursor.fetchall()]
                schema[table] = columns

            conn.close()
        except Exception as e:
            print(f"Warning: Could not load schema: {e}")

        return schema

    def _build_disambiguation_rules(self) -> Dict[str, List[Tuple[str, float]]]:
        return {
            'rate': [
                ('medical_loss_ratio', 0.45),
                ('denial_rate', 0.30),
                ('collection_rate', 0.15),
                ('retention_rate', 0.10)
            ],
            'members': [
                ('total_member_count', 0.50),
                ('high_risk_members', 0.35),
                ('new_members', 0.15)
            ],
            'cost': [
                ('pmpm', 0.55),
                ('total_claims_value', 0.30),
                ('out_of_pocket_spend', 0.15)
            ],
            'claims': [
                ('claim_count', 0.40),
                ('denied_claims', 0.35),
                ('average_claim_amount', 0.25)
            ],
            'referrals': [
                ('specialty_referral_volume', 0.60),
                ('referral_approval_rate', 0.25),
                ('referral_turnaround_time', 0.15)
            ],
            'quality': [
                ('readmission_rate', 0.40),
                ('preventive_screening_rate', 0.35),
                ('chronic_disease_control', 0.25)
            ]
        }

    def assess_data_uncertainty(self, query_result: Dict[str, Any], sql: str) -> Dict[str, Any]:
        data = query_result.get('data', [])
        metric_name = query_result.get('metric_name', 'unknown')

        warnings = []
        confidence_intervals = {}
        data_quality_components = []

        if data and isinstance(data, list) and isinstance(data[0], dict):
            for col in data[0].keys():
                values = [row.get(col) for row in data if row.get(col) is not None]
                missing_pct = (len(data) - len(values)) / len(data) * 100 if data else 0

                if missing_pct > 15:
                    warnings.append({
                        'type': 'missing_data',
                        'detail': f"{missing_pct:.1f}% of '{col}' values are NULL",
                        'severity': 'high' if missing_pct > 30 else 'medium'
                    })

                numeric_values = [
                    float(v) for v in values
                    if isinstance(v, (int, float, str)) and str(v).replace('.', '', 1).isdigit()
                ]

                if len(numeric_values) >= 5:
                    ci_dict = self._compute_confidence_interval(numeric_values, col)
                    if ci_dict:
                        confidence_intervals.update(ci_dict)
                    data_quality_components.append(1.0 - missing_pct/100)

                if len(numeric_values) >= 10:
                    outlier_count = self._detect_outliers(numeric_values)
                    if outlier_count > len(numeric_values) * 0.10:
                        warnings.append({
                            'type': 'outliers_detected',
                            'detail': f"{outlier_count} outliers in '{col}' ({outlier_count/len(numeric_values)*100:.1f}%)",
                            'severity': 'medium'
                        })

        sample_size = len(data)
        if sample_size < 30:
            warnings.append({
                'type': 'small_sample',
                'detail': f"Only {sample_size} records; results may not be statistically significant",
                'severity': 'high' if sample_size < 10 else 'medium'
            })

        last_updated = query_result.get('last_updated')
        if last_updated:
            try:
                update_time = datetime.fromisoformat(last_updated)
                days_old = (datetime.now() - update_time).days
                if days_old > 7:
                    warnings.append({
                        'type': 'data_freshness',
                        'detail': f"Data is {days_old} days old; may not reflect recent activity",
                        'severity': 'low'
                    })
            except:
                pass

        completeness = mean(data_quality_components) if data_quality_components else 0.7
        sample_adequacy = min(1.0, sample_size / 100)
        outlier_presence = 1.0 - len([w for w in warnings if w['type'] == 'outliers_detected']) * 0.1

        overall_confidence = completeness * 0.5 + sample_adequacy * 0.3 + outlier_presence * 0.2
        overall_confidence = max(0.0, min(1.0, overall_confidence))

        recommendations = []
        if sample_size < 50:
            recommendations.append("Consider expanding time period or aggregating regions to increase sample size")
        if warnings and any(w['severity'] == 'high' for w in warnings):
            recommendations.append("Filter out incomplete records before drawing conclusions")
        if days_old > 14 if last_updated else False:
            recommendations.append("Request data refresh; results may be outdated")

        return {
            'overall_confidence': round(overall_confidence, 3),
            'data_quality_score': round(completeness, 3),
            'sample_size': sample_size,
            'warnings': warnings,
            'confidence_intervals': confidence_intervals,
            'recommendations': recommendations
        }

    def _compute_confidence_interval(self, values: List[float], column_name: str) -> Dict[str, Dict]:
        if len(values) < 2:
            return {}

        n = len(values)
        m = mean(values)
        s = stdev(values) if n > 1 else 0
        se = s / math.sqrt(n) if n > 0 else 0

        z_critical = 1.96 if n > 30 else 2.048

        margin = z_critical * se

        return {
            column_name: {
                'value': round(m, 2),
                'ci_lower': round(m - margin, 2),
                'ci_upper': round(m + margin, 2),
                'confidence': 0.95,
                'std_error': round(se, 3)
            }
        }

    def _detect_outliers(self, values: List[float]) -> int:
        if len(values) < 4:
            return 0

        try:
            sorted_vals = sorted(values)
            q1 = sorted_vals[len(values) // 4]
            q3 = sorted_vals[3 * len(values) // 4]
            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = sum(1 for v in values if v < lower_bound or v > upper_bound)
            return outliers
        except:
            return 0

    def disambiguate_query(self, query: str, possible_intents: Optional[List[str]] = None) -> Dict[str, Any]:
        query_lower = query.lower()

        matched_rules = {}
        for keyword, interpretations in self.disambiguation_rules.items():
            if keyword in query_lower:
                matched_rules[keyword] = interpretations

        if not matched_rules:
            return {
                'is_ambiguous': False,
                'ambiguity_type': None,
                'possible_interpretations': [],
                'best_guess': None,
                'suggested_response': None
            }

        all_interpretations = {}
        for keyword, interps in matched_rules.items():
            for intent, base_conf in interps:
                if intent not in all_interpretations:
                    all_interpretations[intent] = []
                all_interpretations[intent].append(base_conf)

        avg_confidences = {
            intent: mean(confs) for intent, confs in all_interpretations.items()
        }

        sorted_intents = sorted(avg_confidences.items(), key=lambda x: x[1], reverse=True)

        ambiguity_type = 'metric_reference'
        if 'region' in query_lower or 'area' in query_lower:
            ambiguity_type = 'region_scope'
        elif 'month' in query_lower or 'year' in query_lower or 'quarter' in query_lower:
            ambiguity_type = 'time_period'
        elif 'total' in query_lower or 'average' in query_lower or 'by' in query_lower:
            ambiguity_type = 'aggregation_level'

        clarifying_questions = {
            'medical_loss_ratio': 'Are you asking about the Medical Loss Ratio (claims divided by premiums)?',
            'denial_rate': 'Do you mean the percentage of claims denied?',
            'collection_rate': 'Are you asking about payment collection efficiency?',
            'retention_rate': 'Do you mean member retention/continuation rate?',
            'pmpm': 'Do you mean Per-Member-Per-Month cost?',
            'total_member_count': 'Are you asking for total active members?',
            'high_risk_members': 'Do you want to focus on high-risk members (risk score > 2.0)?'
        }

        interpretations = [
            {
                'intent': intent,
                'confidence': round(conf, 2),
                'clarifying_question': clarifying_questions.get(intent, f'Are you asking about {intent}?')
            }
            for intent, conf in sorted_intents[:3]
        ]

        best_guess = sorted_intents[0][0] if sorted_intents else None

        suggested_response = f"I found multiple metrics that could match. The most likely is {best_guess}. {interpretations[0]['clarifying_question']}" if interpretations else None

        return {
            'is_ambiguous': len(sorted_intents) > 1 and sorted_intents[0][1] < 0.7,
            'ambiguity_type': ambiguity_type,
            'possible_interpretations': interpretations,
            'best_guess': best_guess,
            'suggested_response': suggested_response
        }

    def handle_uncertain_forecast(
        self,
        values: List[float],
        model_name: str,
        forecast_models: Optional[Dict[str, List[float]]] = None
    ) -> Dict[str, Any]:
        if not forecast_models:
            forecast_models = {model_name: values}

        forecast_models = {k: list(v) for k, v in forecast_models.items()}

        all_forecasts = []
        for model_preds in forecast_models.values():
            all_forecasts.extend(model_preds)

        if not all_forecasts:
            return {
                'consensus_forecast': [],
                'model_agreement': 0.0,
                'prediction_intervals': {},
                'divergence_warning': None,
                'recommended_model': model_name,
                'reasoning': 'Insufficient data for ensemble'
            }

        sorted_forecasts = sorted(all_forecasts)
        n = len(sorted_forecasts)

        p10_idx = int(n * 0.10)
        p50_idx = int(n * 0.50)
        p90_idx = int(n * 0.90)

        prediction_intervals = {
            'p10': sorted_forecasts[p10_idx] if p10_idx < n else None,
            'p50': sorted_forecasts[p50_idx] if p50_idx < n else None,
            'p90': sorted_forecasts[p90_idx] if p90_idx < n else None
        }

        if len(forecast_models) > 1:
            model_stds = [stdev(preds) if len(preds) > 1 else 0 for preds in forecast_models.values()]
            avg_variation = mean(model_stds) if model_stds else 0
            overall_mean = mean(all_forecasts)
            cv = (avg_variation / overall_mean * 100) if overall_mean != 0 else 0

            model_agreement = max(0.0, 1.0 - (cv / 100))
        else:
            model_agreement = 0.8

        divergence_warning = None
        if model_agreement < 0.7:
            divergence_warning = f"Models diverge significantly (agreement: {model_agreement:.0%}). Forecast confidence is low."

        recommended_model = model_name
        reasoning = f"Using {model_name} as primary model"

        if model_agreement < 0.6:
            reasoning += "; high uncertainty suggests caution in decision-making"

        return {
            'consensus_forecast': [round(v, 2) for v in sorted_forecasts[:min(12, n)]],
            'model_agreement': round(model_agreement, 3),
            'prediction_intervals': {k: round(v, 2) if v else None for k, v in prediction_intervals.items()},
            'divergence_warning': divergence_warning,
            'recommended_model': recommended_model,
            'reasoning': reasoning,
            'num_models_evaluated': len(forecast_models)
        }


if __name__ == '__main__':
    db_path = 'data/healthcare_production.db'

    engine = AnticipationEngine(db_path)
    handler = UncertaintyHandler(db_path)

    engine.record_interaction(
        session_id='session_001',
        query='Show MLR by region',
        intent='explore_mlr',
        kpi_area='mlr',
        response_quality=0.88
    )

    engine.record_interaction(
        session_id='session_001',
        query='What about PMPM costs?',
        intent='explore_pmpm',
        kpi_area='pmpm',
        response_quality=0.92
    )

    context = engine.get_session_context('session_001')
    print("Session Context:", json.dumps(context, indent=2, default=str))

    sample_result = {
        'data': [{'pmpm': 847.23}, {'pmpm': 854.12}, {'pmpm': 839.45}],
        'metric_name': 'pmpm',
        'last_updated': datetime.now().isoformat()
    }
    assessment = handler.assess_data_uncertainty(sample_result, 'SELECT AVG(total_pmpm) FROM claims')
    print("Uncertainty Assessment:", json.dumps(assessment, indent=2, default=str))

    disambiguation = handler.disambiguate_query('What is the rate?')
    print("Disambiguation:", json.dumps(disambiguation, indent=2, default=str))
