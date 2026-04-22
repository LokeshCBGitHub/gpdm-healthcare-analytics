import numpy as np
import time
import logging
import json
import sqlite3
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger('gpdm.model_selector')


@dataclass
class QuestionFeatures:
    word_count: int = 0
    has_aggregation: bool = False
    has_groupby: bool = False
    has_filter: bool = False
    has_temporal: bool = False
    has_comparison: bool = False
    has_ranking: bool = False
    has_clinical_terms: bool = False
    has_follow_up_signal: bool = False
    num_entities: int = 0
    complexity: float = 0.0
    intent: str = ''
    intent_confidence: float = 0.0
    memory_match_score: float = 0.0
    memory_match_sql: str = ''
    domain: str = ''
    recommended_strategy: str = ''
    strategy_confidence: float = 0.0


class QuestionAnalyzer:
    AGG_WORDS = {'total', 'sum', 'average', 'avg', 'mean', 'count', 'maximum',
                 'minimum', 'max', 'min', 'median'}
    GROUP_WORDS = {'by', 'per', 'breakdown', 'distribution', 'grouped', 'split',
                   'each', 'every', 'across'}
    FILTER_WORDS = {'where', 'only', 'just', 'specifically', 'greater', 'less',
                    'above', 'below', 'between', 'older', 'younger', 'more',
                    'fewer', 'than', 'at least', 'at most', 'excluding'}
    TEMPORAL_WORDS = {'trend', 'over time', 'monthly', 'quarterly', 'yearly',
                      'annual', 'weekly', 'daily', 'growth', 'decline',
                      'increase', 'decrease', 'change', 'trajectory', 'forecast'}
    COMPARE_WORDS = {'compare', 'versus', 'vs', 'difference', 'gap',
                     'relative', 'against', 'benchmark', 'outperform'}
    RANK_WORDS = {'top', 'bottom', 'highest', 'lowest', 'most', 'least',
                  'best', 'worst', 'busiest', 'leading'}
    CLINICAL_WORDS = {'diabetes', 'hypertension', 'heart', 'cancer', 'copd',
                      'asthma', 'covid', 'mental health', 'depression',
                      'icd', 'cpt', 'hcc', 'hedis', 'readmission',
                      'a1c', 'screening', 'preventive', 'chronic',
                      'diagnosis', 'prescription', 'medication', 'drug',
                      'hospitalization', 'emergency', 'inpatient', 'outpatient'}
    FOLLOWUP_WORDS = {'also', 'what about', 'and', 'but', 'instead',
                      'same', 'those', 'it', 'them', 'this', 'that',
                      'only', 'just', 'now', 'more'}

    def analyze(self, question: str, intent: Optional[Dict] = None,
                memory_recall: Optional[List] = None) -> QuestionFeatures:
        q = question.lower()
        words = q.split()
        features = QuestionFeatures()

        features.word_count = len(words)
        features.has_aggregation = any(w in q for w in self.AGG_WORDS)
        features.has_groupby = any(w in q for w in self.GROUP_WORDS)
        features.has_filter = any(w in q for w in self.FILTER_WORDS)
        features.has_temporal = any(w in q for w in self.TEMPORAL_WORDS)
        features.has_comparison = any(w in q for w in self.COMPARE_WORDS)
        features.has_ranking = any(w in q for w in self.RANK_WORDS)
        features.has_clinical_terms = any(w in q for w in self.CLINICAL_WORDS)
        features.has_follow_up_signal = len(words) <= 4 or any(w in q for w in self.FOLLOWUP_WORDS)

        if intent:
            features.intent = intent.get('intent', '')
            features.intent_confidence = intent.get('confidence', 0)

        if memory_recall:
            best_recall = memory_recall[0] if memory_recall else (None, 0)
            if isinstance(best_recall, tuple) and len(best_recall) >= 2:
                meta, score = best_recall
                features.memory_match_score = score
                features.memory_match_sql = meta.get('sql', '') if isinstance(meta, dict) else ''

        complexity = 0.0
        if features.has_aggregation: complexity += 0.15
        if features.has_groupby: complexity += 0.15
        if features.has_filter: complexity += 0.15
        if features.has_temporal: complexity += 0.2
        if features.has_comparison: complexity += 0.2
        if features.has_ranking: complexity += 0.1
        if features.has_clinical_terms: complexity += 0.15
        if features.word_count > 10: complexity += 0.1
        if features.word_count > 20: complexity += 0.1
        features.complexity = min(complexity, 1.0)

        features.recommended_strategy, features.strategy_confidence = (
            self._select_strategy(features))

        return features

    def _select_strategy(self, f: QuestionFeatures) -> Tuple[str, float]:
        scores = {'RECALL': 0.0, 'STATISTICAL': 0.0, 'ATTENTION': 0.0,
                  'ANALYTICAL': 0.0, 'CLINICAL': 0.0, 'EXPLORATORY': 0.0}

        if f.memory_match_score > 0.85:
            scores['RECALL'] = 0.95
        elif f.memory_match_score > 0.7:
            scores['RECALL'] = 0.7

        if f.complexity < 0.3 and f.intent in ('count', 'aggregate', 'lookup'):
            scores['STATISTICAL'] = 0.8
        if f.word_count <= 6 and not f.has_temporal:
            scores['STATISTICAL'] += 0.2

        if f.complexity > 0.4:
            scores['ATTENTION'] = 0.3 + f.complexity * 0.5
        if f.word_count > 10:
            scores['ATTENTION'] += 0.2

        if f.has_temporal:
            scores['ANALYTICAL'] = 0.8
        if f.has_comparison:
            scores['ANALYTICAL'] = max(scores['ANALYTICAL'], 0.7)
        if f.intent in ('trend', 'comparison', 'rate', 'percentage'):
            scores['ANALYTICAL'] += 0.3

        if f.has_clinical_terms:
            scores['CLINICAL'] = 0.75
            if f.intent in ('rate', 'percentage', 'trend'):
                scores['CLINICAL'] += 0.15

        if f.complexity > 0.6 and f.word_count > 15:
            scores['EXPLORATORY'] = 0.6
        if f.has_comparison and f.has_temporal:
            scores['EXPLORATORY'] = max(scores['EXPLORATORY'], 0.5)

        best_strategy = max(scores, key=scores.get)
        return best_strategy, min(scores[best_strategy], 1.0)


@dataclass
class DashboardRecommendation:
    chart_type: str
    chart_config: Dict
    layout: str
    title: str
    subtitle: str = ''
    deep_dive_available: bool = False
    secondary_chart: str = ''
    color_scheme: str = 'healthcare'


class DashboardSelector:
    COLOR_SCHEMES = {
        'healthcare': {
            'primary': '#0066CC',
            'secondary': '#00A3E0',
            'accent': '#FF6B35',
            'success': '#28A745',
            'warning': '#FFC107',
            'danger': '#DC3545',
            'neutral': '#6C757D',
            'background': '#F8F9FA',
            'palette': ['#0066CC', '#00A3E0', '#FF6B35', '#28A745',
                       '#FFC107', '#6C757D', '#8B5CF6', '#EC4899'],
        },
        'professional': {
            'primary': '#1B2838',
            'secondary': '#2D4059',
            'accent': '#EA5455',
            'success': '#28C76F',
            'warning': '#FF9F43',
            'danger': '#EA5455',
            'neutral': '#82868B',
            'background': '#F4F5FA',
            'palette': ['#1B2838', '#2D4059', '#EA5455', '#28C76F',
                       '#FF9F43', '#7367F0', '#00CFE8', '#82868B'],
        },
    }

    def recommend(self, query_result: Dict[str, Any],
                  intent: str = '',
                  row_count: int = 0,
                  columns: Optional[List[str]] = None,
                  data_sample: Optional[List[Tuple]] = None) -> DashboardRecommendation:
        columns = columns or []
        data_sample = data_sample or []
        col_types = self._detect_column_types(columns, data_sample)
        num_cols = [c for c, t in col_types.items() if t == 'numeric']
        cat_cols = [c for c, t in col_types.items() if t == 'categorical']
        date_cols = [c for c, t in col_types.items() if t == 'temporal']

        if row_count == 1 and len(num_cols) <= 2:
            return self._recommend_kpi(columns, data_sample, intent)

        if date_cols and num_cols and intent in ('trend', 'breakdown', 'count'):
            return self._recommend_timeseries(columns, data_sample, intent, date_cols, num_cols)

        if row_count <= 10 and cat_cols:
            return self._recommend_categorical(columns, data_sample, intent,
                                               cat_cols, num_cols, row_count)

        if row_count <= 30 and cat_cols and num_cols:
            return DashboardRecommendation(
                chart_type='horizontal_bar',
                chart_config={
                    'x_axis': num_cols[0] if num_cols else columns[-1],
                    'y_axis': cat_cols[0] if cat_cols else columns[0],
                    'sort': 'descending',
                    'show_values': True,
                    'color_scheme': 'healthcare',
                },
                layout='full',
                title=self._auto_title(intent, cat_cols, num_cols),
                deep_dive_available=True,
            )

        if row_count > 30:
            return DashboardRecommendation(
                chart_type='data_table',
                chart_config={
                    'columns': columns,
                    'page_size': 20,
                    'sortable': True,
                    'filterable': True,
                    'export': True,
                    'highlight_rules': self._generate_highlight_rules(col_types),
                },
                layout='full',
                title=self._auto_title(intent, cat_cols, num_cols),
                secondary_chart='summary_stats' if num_cols else '',
                deep_dive_available=True,
            )

        return DashboardRecommendation(
            chart_type='data_table',
            chart_config={'columns': columns, 'page_size': 20},
            layout='full',
            title='Query Results',
        )

    def _recommend_kpi(self, columns: List[str], data_sample: List[Tuple],
                       intent: str) -> DashboardRecommendation:
        value = data_sample[0][0] if data_sample and data_sample[0] else 0
        label = columns[0] if columns else 'Result'

        if isinstance(value, float):
            if abs(value) >= 1_000_000:
                display = f"{value/1_000_000:.1f}M"
            elif abs(value) >= 1_000:
                display = f"{value/1_000:.1f}K"
            else:
                display = f"{value:.2f}"
        elif isinstance(value, int):
            display = f"{value:,}"
        else:
            display = str(value)

        return DashboardRecommendation(
            chart_type='kpi_card',
            chart_config={
                'value': display,
                'raw_value': value,
                'label': label.replace('_', ' ').title(),
                'icon': self._intent_icon(intent),
                'color': 'primary',
                'trend_indicator': None,
            },
            layout='quarter',
            title=label.replace('_', ' ').title(),
        )

    def _recommend_timeseries(self, columns: List[str], data_sample: List[Tuple],
                               intent: str, date_cols: List[str],
                               num_cols: List[str]) -> DashboardRecommendation:
        chart = 'area' if intent in ('trend', 'count') else 'line'
        return DashboardRecommendation(
            chart_type=chart,
            chart_config={
                'x_axis': date_cols[0],
                'y_axis': num_cols[0] if num_cols else columns[-1],
                'show_trend_line': True,
                'show_data_points': len(data_sample) <= 30,
                'fill_area': chart == 'area',
                'color_scheme': 'healthcare',
                'animate': True,
            },
            layout='full',
            title=self._auto_title(intent, date_cols, num_cols),
            subtitle='Trend over time',
            deep_dive_available=True,
            secondary_chart='summary_stats',
        )

    def _recommend_categorical(self, columns: List[str], data_sample: List[Tuple],
                                intent: str, cat_cols: List[str],
                                num_cols: List[str], row_count: int) -> DashboardRecommendation:
        if row_count <= 5 and intent in ('breakdown', 'percentage', 'rate'):
            chart = 'donut'
            layout = 'half'
        elif len(num_cols) >= 2:
            chart = 'grouped_bar'
            layout = 'full'
        else:
            chart = 'bar'
            layout = 'half' if row_count <= 5 else 'full'

        return DashboardRecommendation(
            chart_type=chart,
            chart_config={
                'x_axis': cat_cols[0] if cat_cols else columns[0],
                'y_axis': num_cols[0] if num_cols else columns[-1],
                'sort': 'descending',
                'show_values': True,
                'show_percentage': intent in ('breakdown', 'percentage', 'rate'),
                'color_scheme': 'healthcare',
                'animate': True,
            },
            layout=layout,
            title=self._auto_title(intent, cat_cols, num_cols),
            deep_dive_available=True,
        )

    def _detect_column_types(self, columns: List[str],
                              data_sample: List[Tuple]) -> Dict[str, str]:
        types = {}
        for i, col in enumerate(columns):
            col_lower = col.lower()

            if any(d in col_lower for d in ['date', 'time', 'month', 'year', 'period', 'quarter']):
                types[col] = 'temporal'
                continue

            if data_sample:
                val = data_sample[0][i] if i < len(data_sample[0]) else None
                if isinstance(val, (int, float)):
                    types[col] = 'numeric'
                elif isinstance(val, str):
                    if len(val) >= 7 and (val[4:5] == '-' or val[2:3] == '/'):
                        types[col] = 'temporal'
                    else:
                        types[col] = 'categorical'
                else:
                    types[col] = 'categorical'
            else:
                if any(n in col_lower for n in ['count', 'total', 'sum', 'avg', 'amount',
                                                 'rate', 'pct', 'percentage', 'num']):
                    types[col] = 'numeric'
                else:
                    types[col] = 'categorical'

        return types

    def _auto_title(self, intent: str, primary_cols: List[str],
                     secondary_cols: List[str]) -> str:
        primary = primary_cols[0].replace('_', ' ').title() if primary_cols else 'Data'
        secondary = secondary_cols[0].replace('_', ' ').title() if secondary_cols else ''

        templates = {
            'count': f"Count by {primary}",
            'aggregate': f"{secondary} by {primary}" if secondary else f"Aggregate by {primary}",
            'breakdown': f"Distribution by {primary}",
            'ranking': f"Top {primary}" if not secondary else f"Top {primary} by {secondary}",
            'trend': f"{secondary} Over Time" if secondary else "Trend Over Time",
            'percentage': f"Percentage by {primary}",
            'rate': f"Rate by {primary}",
            'comparison': f"{primary} Comparison",
        }
        return templates.get(intent, f"{primary} Analysis")

    def _intent_icon(self, intent: str) -> str:
        icons = {
            'count': 'hash',
            'aggregate': 'calculator',
            'breakdown': 'pie-chart',
            'ranking': 'trending-up',
            'trend': 'activity',
            'percentage': 'percent',
            'rate': 'bar-chart-2',
            'filter': 'filter',
            'lookup': 'search',
        }
        return icons.get(intent, 'bar-chart')

    def _generate_highlight_rules(self, col_types: Dict[str, str]) -> List[Dict]:
        rules = []
        for col, ctype in col_types.items():
            if ctype == 'numeric' and 'rate' in col.lower():
                rules.append({
                    'column': col,
                    'type': 'color_scale',
                    'min_color': '#DC3545',
                    'max_color': '#28A745',
                })
            elif ctype == 'numeric' and any(w in col.lower() for w in ['count', 'total', 'amount']):
                rules.append({
                    'column': col,
                    'type': 'data_bar',
                    'color': '#0066CC',
                })
        return rules


class MetaModelSelector:

    def __init__(self, db_path: str = None):
        self.analyzer = QuestionAnalyzer()
        self.dashboard_selector = DashboardSelector()
        self.db_path = db_path
        self.strategy_performance: Dict[str, Dict] = {
            'STATISTICAL': {'uses': 0, 'successes': 0, 'total_latency': 0},
            'ATTENTION': {'uses': 0, 'successes': 0, 'total_latency': 0},
            'RECALL': {'uses': 0, 'successes': 0, 'total_latency': 0},
            'ANALYTICAL': {'uses': 0, 'successes': 0, 'total_latency': 0},
            'CLINICAL': {'uses': 0, 'successes': 0, 'total_latency': 0},
            'EXPLORATORY': {'uses': 0, 'successes': 0, 'total_latency': 0},
        }
        if db_path:
            self._load_performance()

    def select_strategy(self, question: str, intent: Optional[Dict] = None,
                        memory_recall: Optional[List] = None) -> Dict[str, Any]:
        features = self.analyzer.analyze(question, intent, memory_recall)
        strategy = features.recommended_strategy
        confidence = features.strategy_confidence

        perf = self.strategy_performance.get(strategy, {})
        if perf.get('uses', 0) > 10:
            success_rate = perf['successes'] / perf['uses']
            if success_rate < 0.5:
                confidence *= 0.7

        return {
            'strategy': strategy,
            'confidence': confidence,
            'features': features,
            'use_transformer': strategy in ('ATTENTION', 'ANALYTICAL', 'EXPLORATORY'),
            'use_hopfield': strategy in ('RECALL', 'ATTENTION'),
            'use_gnn': strategy in ('ANALYTICAL', 'EXPLORATORY', 'CLINICAL'),
            'use_clinical': strategy == 'CLINICAL' or features.has_clinical_terms,
        }

    def select_dashboard(self, query_result: Dict,
                         intent: str = '',
                         columns: Optional[List[str]] = None,
                         data_sample: Optional[List[Tuple]] = None,
                         row_count: int = 0) -> DashboardRecommendation:
        return self.dashboard_selector.recommend(
            query_result, intent, row_count, columns, data_sample
        )

    def record_outcome(self, strategy: str, success: bool,
                       latency: float = 0):
        if strategy in self.strategy_performance:
            perf = self.strategy_performance[strategy]
            perf['uses'] += 1
            if success:
                perf['successes'] += 1
            perf['total_latency'] += latency

        if self.db_path:
            self._save_performance()

    def get_performance_report(self) -> Dict[str, Any]:
        report = {}
        for strategy, perf in self.strategy_performance.items():
            uses = perf['uses']
            report[strategy] = {
                'uses': uses,
                'success_rate': perf['successes'] / max(uses, 1),
                'avg_latency': perf['total_latency'] / max(uses, 1),
            }
        return report

    def _load_performance(self) -> None:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute('''CREATE TABLE IF NOT EXISTS strategy_performance (
                strategy TEXT PRIMARY KEY, data TEXT)''')
            rows = conn.execute('SELECT strategy, data FROM strategy_performance').fetchall()
            conn.close()
            for strategy, data in rows:
                self.strategy_performance[strategy] = json.loads(data)
        except Exception:
            pass

    def _save_performance(self) -> None:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute('''CREATE TABLE IF NOT EXISTS strategy_performance (
                strategy TEXT PRIMARY KEY, data TEXT)''')
            for strategy, perf in self.strategy_performance.items():
                conn.execute('''INSERT OR REPLACE INTO strategy_performance (strategy, data)
                                VALUES (?, ?)''', (strategy, json.dumps(perf)))
            conn.commit()
            conn.close()
        except Exception:
            pass
