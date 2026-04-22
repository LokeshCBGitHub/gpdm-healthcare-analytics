from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_TIMEOUT = 15


INTENT_KEYWORDS = {
    'survival': [
        'survival', 'time to event', 'readmission curve', 'hazard',
        'cox', 'readmit within', 'days until', 'time until readmission',
        'survival curve', 'survival analysis', 'kaplan meier', 'km curve',
        'time to readmit', 'readmission probability', 'event time',
        'how long until', 'when will they', 'probability of readmission',
    ],
    'fairness': [
        'fairness', 'equity', 'bias', 'disparity', 'discrimination',
        'health equity', 'demographic parity', 'equalized odds',
        'fair', 'unfair', 'racial disparity', 'gender disparity',
        'equitable', 'inequity', 'disparities across', 'bias audit',
        'is our model fair', 'are predictions biased', 'demographic bias',
    ],
    'causal': [
        'what if', 'causal', 'counterfactual', 'treatment effect',
        'would have', 'had we', 'if we enrolled', 'if we implemented',
        'impact of', 'effect of telehealth', 'effect of follow-up',
        'does .* reduce', 'does .* increase', 'did .* cause',
        'attributable', 'causal effect', 'intervention effect',
        'would readmission', 'what would happen if',
    ],
    'causal_forest': [
        'who benefits most', 'heterogeneous', 'treatment heterogeneity',
        'personalized treatment', 'which patients benefit',
        'subgroup effect', 'varying effect', 'cate',
        'conditional treatment', 'individual treatment effect',
    ],
    'dq_deep': [
        'data quality', 'data health', 'completeness', 'validity',
        'data integrity', 'data issues', 'data problems', 'missing data',
        'null values', 'data accuracy', 'quality score', 'dq score',
        'how clean is', 'data completeness', 'column quality',
    ],
    'executive': [
        'executive summary', 'cfo briefing', 'vp briefing',
        'board report', 'executive intelligence', 'kpi summary',
        'financial overview', 'operational overview', 'clinical overview',
        'leadership briefing', 'c-suite', 'executive dashboard',
    ],
    'preventive': [
        'preventive care', 'prevention', 'at-risk population',
        'cost of inaction', 'intervention roi', 'care gap',
        'high utilizer', 'frequent er', 'preventable',
        'wellness program', 'screening gap', 'chronic prevention',
        'pre-chronic', 'pre-diabetes', 'pre-hypertension',
    ],
    'chronic_risk': [
        'chronic risk', 'chronic prediction', 'will develop',
        'pre-chronic', 'risk of developing', 'future chronic',
        'chronic progression', 'disease progression', 'comorbidity risk',
    ],
    'business_insights': [
        'rising risk', 'risk cohort', 'readmission watchlist',
        'hedis gap', 'care gap', 'network performance',
        'network score', 'provider performance', 'quality gap',
        'high risk patients', 'at risk cohort', 'bounce back',
    ],
    'kpi_discovery': [
        'discover kpi', 'auto kpi', 'what kpis', 'suggested kpis',
        'important metrics', 'key metrics', 'most asked',
        'popular questions', 'trending questions', 'kpi recommendations',
    ],
}


class AnalyticsOrchestrator:

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._dq_cache: Dict[str, Dict] = {}
        self._dq_cache_time: float = 0
        self._dq_cache_ttl: float = 300

        self._dq_engine = None
        self._survival = None
        self._fairness = None
        self._causal = None
        self._causal_forest = None
        self._deep_analytics = None
        self._chronic_risk = None


    def _get_dq_engine(self):
        if self._dq_engine is None:
            try:
                from dq_engine import DQEngine
                self._dq_engine = DQEngine(self.db_path, sample_size=10000)
                logger.info("DQ Engine loaded")
            except Exception as e:
                logger.warning("DQ Engine unavailable: %s", e)
        return self._dq_engine

    def _get_survival(self):
        if self._survival is None:
            try:
                import survival
                self._survival = survival
                logger.info("Survival module loaded")
            except Exception as e:
                logger.warning("Survival module unavailable: %s", e)
        return self._survival

    def _get_fairness(self):
        if self._fairness is None:
            try:
                import fairness
                self._fairness = fairness
                logger.info("Fairness module loaded")
            except Exception as e:
                logger.warning("Fairness module unavailable: %s", e)
        return self._fairness

    def _get_causal(self):
        if self._causal is None:
            try:
                import causal_inference
                self._causal = causal_inference
                logger.info("Causal inference module loaded")
            except Exception as e:
                logger.warning("Causal inference module unavailable: %s", e)
        return self._causal

    def _get_causal_forest(self):
        if self._causal_forest is None:
            try:
                import causal_forest
                self._causal_forest = causal_forest
                logger.info("Causal forest module loaded")
            except Exception as e:
                logger.warning("Causal forest module unavailable: %s", e)
        return self._causal_forest

    def _get_deep_analytics(self):
        if self._deep_analytics is None:
            try:
                from deep_healthcare_analytics import DeepHealthcareAnalytics
                self._deep_analytics = DeepHealthcareAnalytics(self.db_path)
                logger.info("Deep healthcare analytics loaded")
            except Exception as e:
                logger.warning("Deep healthcare analytics unavailable: %s", e)
        return self._deep_analytics

    def _get_chronic_risk(self):
        if self._chronic_risk is None:
            try:
                from chronic_risk_predictor import ChronicRiskPredictor
                self._chronic_risk = ChronicRiskPredictor(self.db_path)
                logger.info("Chronic risk predictor loaded")
            except Exception as e:
                logger.warning("Chronic risk predictor unavailable: %s", e)
        return self._chronic_risk


    def detect_intent(self, question: str) -> List[Tuple[str, float]]:
        q = question.lower().strip()
        scores: Dict[str, float] = {}

        for engine, keywords in INTENT_KEYWORDS.items():
            score = 0.0
            matched = []
            for kw in keywords:
                if kw in q:
                    weight = len(kw.split()) * 0.15 + 0.1
                    score += weight
                    matched.append(kw)

            if matched:
                scores[engine] = min(score, 1.0)

        result = sorted(scores.items(), key=lambda x: -x[1])
        return result


    def get_dq_badge(self, table_name: str = None) -> Dict[str, Any]:
        now = time.time()
        cache_key = table_name or '__overall__'

        if cache_key in self._dq_cache and (now - self._dq_cache_time) < self._dq_cache_ttl:
            return self._dq_cache[cache_key]

        engine = self._get_dq_engine()
        if not engine:
            return {'status': 'unavailable', 'confidence': 0, 'grade': 'N/A'}

        try:
            if table_name:
                report = engine.check_table(table_name)
                score = report.get('dq_score', 0)
                grade = report.get('dq_grade', 'N/A')
                cc = report.get('column_checks', {})
                avg_comp = sum(c.get('completeness', 0) for c in cc.values()) / max(len(cc), 1)
                avg_valid = sum(c.get('validity', 0) for c in cc.values()) / max(len(cc), 1)
                avg_consist = sum(c.get('consistency', 0) for c in cc.values()) / max(len(cc), 1)
            else:
                report = engine.run()
                summary = report.get('summary', {})
                score = summary.get('average_dq_score', 0)
                grade = summary.get('overall_grade', 'N/A')
                avg_comp, avg_valid, avg_consist = 0, 0, 0
                for tbl_data in report.get('tables', {}).values():
                    cc = tbl_data.get('column_checks', {})
                    if cc:
                        avg_comp += sum(c.get('completeness', 0) for c in cc.values()) / len(cc)
                        avg_valid += sum(c.get('validity', 0) for c in cc.values()) / len(cc)
                        avg_consist += sum(c.get('consistency', 0) for c in cc.values()) / len(cc)
                n_tables = len(report.get('tables', {}))
                if n_tables:
                    avg_comp /= n_tables
                    avg_valid /= n_tables
                    avg_consist /= n_tables

            badge = {
                'status': 'ok',
                'confidence': round(score, 1),
                'grade': grade,
                'table': table_name,
                'details': {
                    'completeness': round(avg_comp, 1),
                    'validity': round(avg_valid, 1),
                    'consistency': round(avg_consist, 1),
                },
                'timestamp': now,
            }
            self._dq_cache[cache_key] = badge
            self._dq_cache_time = now
            return badge
        except Exception as e:
            logger.warning("DQ badge error for %s: %s", table_name, e)
            return {'status': 'error', 'confidence': 0, 'grade': 'N/A', 'error': str(e)}


    def run_survival_analysis(self, horizon_days: int = 90, method: str = 'cox') -> Dict[str, Any]:
        mod = self._get_survival()
        if not mod:
            return {'status': 'unavailable', 'engine': 'survival'}
        try:
            result = mod.readmission_survival_from_sqlite(
                self.db_path, horizon_days=horizon_days, method=method
            )
            if result.get('status') == 'ok':
                result['narrative'] = self._build_survival_narrative(result)
                result['engine'] = 'survival'
            return result
        except Exception as e:
            logger.error("Survival analysis error: %s\n%s", e, traceback.format_exc())
            return {'status': 'error', 'engine': 'survival', 'error': str(e)}

    def run_fairness_audit(self, threshold: float = 0.5) -> Dict[str, Any]:
        mod = self._get_fairness()
        if not mod:
            return {'status': 'unavailable', 'engine': 'fairness'}
        try:
            result = mod.audit_from_sqlite(self.db_path, threshold=threshold)
            if result.get('status') == 'ok':
                result['narrative'] = self._build_fairness_narrative(result)
                result['engine'] = 'fairness'
            return result
        except Exception as e:
            logger.error("Fairness audit error: %s\n%s", e, traceback.format_exc())
            return {'status': 'error', 'engine': 'fairness', 'error': str(e)}

    def run_causal_analysis(self, analysis_type: str = 'followup') -> Dict[str, Any]:
        mod = self._get_causal()
        if not mod:
            return {'status': 'unavailable', 'engine': 'causal_inference'}
        try:
            if analysis_type == 'telehealth':
                result = mod.estimate_telehealth_effect(self.db_path)
            else:
                result = mod.estimate_followup_effect(self.db_path)
            if isinstance(result, dict):
                result['narrative'] = self._build_causal_narrative(result, analysis_type)
                result['engine'] = 'causal_inference'
            return result
        except Exception as e:
            logger.error("Causal analysis error: %s\n%s", e, traceback.format_exc())
            return {'status': 'error', 'engine': 'causal_inference', 'error': str(e)}

    def run_causal_forest(self) -> Dict[str, Any]:
        mod = self._get_causal_forest()
        if not mod:
            return {'status': 'unavailable', 'engine': 'causal_forest'}
        try:
            result = mod.estimate_heterogeneous_followup_effect(self.db_path)
            if result.get('status') == 'ok':
                result['narrative'] = self._build_forest_narrative(result)
                result['engine'] = 'causal_forest'
            return result
        except Exception as e:
            logger.error("Causal forest error: %s\n%s", e, traceback.format_exc())
            return {'status': 'error', 'engine': 'causal_forest', 'error': str(e)}

    def run_dq_deep(self, tables: List[str] = None) -> Dict[str, Any]:
        engine = self._get_dq_engine()
        if not engine:
            return {'status': 'unavailable', 'engine': 'dq_engine'}
        try:
            raw = engine.run(tables=tables)
            summary = raw.get('summary', {})
            result = {
                'status': 'ok',
                'overall_score': summary.get('average_dq_score', 0),
                'overall_grade': summary.get('overall_grade', 'N/A'),
                'tables': {},
                'issues': summary.get('issues_by_type', {}),
                'table_scores': summary.get('table_scores', []),
            }
            for tname, tdata in raw.get('tables', {}).items():
                result['tables'][tname] = {
                    'table_score': tdata.get('dq_score', 0),
                    'grade': tdata.get('dq_grade', 'N/A'),
                    'rows': tdata.get('total_rows', 0),
                    'columns': tdata.get('column_checks', {}),
                }
            result['narrative'] = self._build_dq_narrative(result)
            result['engine'] = 'dq_engine'
            return result
        except Exception as e:
            logger.error("DQ deep scan error: %s\n%s", e, traceback.format_exc())
            return {'status': 'error', 'engine': 'dq_engine', 'error': str(e)}

    def run_executive_summary(self, role: str = 'cfo') -> Dict[str, Any]:
        analytics = self._get_deep_analytics()
        if not analytics:
            return {'status': 'unavailable', 'engine': 'executive'}
        try:
            result = analytics.get_executive_summary(role)
            result['engine'] = 'executive'
            return result
        except Exception as e:
            logger.error("Executive summary error: %s\n%s", e, traceback.format_exc())
            return {'status': 'error', 'engine': 'executive', 'error': str(e)}

    def run_preventive_analysis(self) -> Dict[str, Any]:
        analytics = self._get_deep_analytics()
        if not analytics:
            return {'status': 'unavailable', 'engine': 'preventive'}
        try:
            result = analytics.analyze_preventive_opportunities()
            result['engine'] = 'preventive'
            return result
        except Exception as e:
            logger.error("Preventive analysis error: %s\n%s", e, traceback.format_exc())
            return {'status': 'error', 'engine': 'preventive', 'error': str(e)}

    def run_chronic_risk(self) -> Dict[str, Any]:
        predictor = self._get_chronic_risk()
        if not predictor:
            return {'status': 'unavailable', 'engine': 'chronic_risk'}
        try:
            result = predictor.predict_all()
            result['engine'] = 'chronic_risk'
            return result
        except Exception as e:
            logger.error("Chronic risk error: %s\n%s", e, traceback.format_exc())
            return {'status': 'error', 'engine': 'chronic_risk', 'error': str(e)}

    def run_business_insights(self, question: str) -> Dict[str, Any]:
        try:
            from business_insights import rising_risk_cohort, readmit_watchlist, hedis_gap_list, network_performance
            q = question.lower()
            if 'readmit' in q or 'bounce back' in q:
                result = readmit_watchlist(self.db_path)
                label = 'Readmission Watchlist'
            elif 'hedis' in q or 'care gap' in q or 'quality gap' in q:
                result = hedis_gap_list(self.db_path)
                label = 'HEDIS Care Gaps'
            elif 'network' in q:
                result = network_performance(self.db_path)
                label = 'Network Performance'
            else:
                result = rising_risk_cohort(self.db_path)
                label = 'Rising Risk Cohort'
            result['engine'] = 'business_insights'
            result['narrative'] = result.get('narrative', f'{label} analysis complete.')
            return result
        except Exception as e:
            logger.error("Business insights error: %s\n%s", e, traceback.format_exc())
            return {'status': 'error', 'engine': 'business_insights', 'error': str(e)}

    def run_kpi_discovery(self) -> Dict[str, Any]:
        try:
            from kpi_discovery import get_discovery
            data_dir = os.path.dirname(self.db_path)
            disc = get_discovery(data_dir)
            discovered = disc.discover()
            stats = disc.stats()
            result = {
                'status': 'ok',
                'engine': 'kpi_discovery',
                'discovered_kpis': [
                    {'label': k.label, 'support': k.support,
                     'frequency': k.frequency, 'tokens': k.signal_tokens}
                    for k in discovered[:20]
                ],
                'stats': stats,
                'narrative': (f"Auto-discovered {len(discovered)} important KPIs from "
                              f"query patterns. Top signals: "
                              + ', '.join(k.label for k in discovered[:5])) if discovered else
                             'No KPIs discovered yet — more query history needed.',
            }
            return result
        except Exception as e:
            logger.error("KPI discovery error: %s\n%s", e, traceback.format_exc())
            return {'status': 'error', 'engine': 'kpi_discovery', 'error': str(e)}


    def route(self, question: str) -> Optional[Dict[str, Any]]:
        intents = self.detect_intent(question)
        if not intents:
            return None

        top_intent, confidence = intents[0]
        logger.info("Orchestrator routing '%s' → %s (confidence=%.2f)",
                     question[:60], top_intent, confidence)

        t0 = time.time()

        result = None
        q_lower = question.lower()

        if top_intent == 'survival':
            horizon = 90
            for n in [30, 60, 90, 180, 365]:
                if str(n) in q_lower and ('day' in q_lower or 'horizon' in q_lower):
                    horizon = n
                    break
            result = self.run_survival_analysis(horizon_days=horizon)

        elif top_intent == 'fairness':
            result = self.run_fairness_audit()

        elif top_intent == 'causal':
            atype = 'telehealth' if 'telehealth' in q_lower else 'followup'
            result = self.run_causal_analysis(analysis_type=atype)

        elif top_intent == 'causal_forest':
            result = self.run_causal_forest()

        elif top_intent == 'dq_deep':
            result = self.run_dq_deep()

        elif top_intent == 'executive':
            role = 'cfo'
            if 'vp' in q_lower or 'operations' in q_lower:
                role = 'vp_operations'
            elif 'clinical' in q_lower or 'medical' in q_lower:
                role = 'clinical_director'
            elif 'population' in q_lower:
                role = 'population_health'
            result = self.run_executive_summary(role=role)

        elif top_intent == 'preventive':
            result = self.run_preventive_analysis()

        elif top_intent == 'chronic_risk':
            result = self.run_chronic_risk()

        elif top_intent == 'business_insights':
            result = self.run_business_insights(question)

        elif top_intent == 'kpi_discovery':
            result = self.run_kpi_discovery()

        if result is None:
            return None

        result['orchestrator'] = {
            'intent': top_intent,
            'confidence': confidence,
            'all_intents': intents[:3],
            'latency_ms': round((time.time() - t0) * 1000),
        }

        result['dq_badge'] = self.get_dq_badge()

        return result


    def full_intelligence_scan(self) -> Dict[str, Any]:
        t0 = time.time()
        report: Dict[str, Any] = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'engines': {},
            'status': 'ok',
        }

        engine_runners = {
            'data_quality': lambda: self.run_dq_deep(),
            'survival': lambda: self.run_survival_analysis(),
            'fairness': lambda: self.run_fairness_audit(),
            'causal_followup': lambda: self.run_causal_analysis('followup'),
            'causal_telehealth': lambda: self.run_causal_analysis('telehealth'),
            'causal_forest': lambda: self.run_causal_forest(),
            'executive_cfo': lambda: self.run_executive_summary('cfo'),
            'preventive': lambda: self.run_preventive_analysis(),
            'business_insights': lambda: self.run_business_insights('rising risk cohort'),
            'kpi_discovery': lambda: self.run_kpi_discovery(),
        }

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            for name, runner in engine_runners.items():
                futures[name] = executor.submit(runner)

            for name, future in futures.items():
                try:
                    result = future.result(timeout=MODULE_TIMEOUT)
                    report['engines'][name] = result
                    logger.info("Engine %s completed: status=%s",
                                name, result.get('status', 'unknown'))
                except FuturesTimeout:
                    report['engines'][name] = {
                        'status': 'timeout',
                        'error': f'{name} exceeded {MODULE_TIMEOUT}s timeout'
                    }
                    logger.warning("Engine %s timed out", name)
                except Exception as e:
                    report['engines'][name] = {
                        'status': 'error',
                        'error': str(e)
                    }
                    logger.error("Engine %s failed: %s", name, e)

        ok_count = sum(1 for v in report['engines'].values()
                       if v.get('status') == 'ok')
        total = len(report['engines'])
        report['summary'] = {
            'engines_run': total,
            'engines_ok': ok_count,
            'engines_failed': total - ok_count,
            'total_latency_ms': round((time.time() - t0) * 1000),
        }

        report['narrative'] = self._build_scan_narrative(report)

        return report


    def enrich_response(self, question: str, base_result: Dict[str, Any]) -> Dict[str, Any]:
        sql = base_result.get('sql', '')
        primary_table = self._extract_primary_table(sql)
        base_result['dq_badge'] = self.get_dq_badge(primary_table)

        intents = self.detect_intent(question)
        if intents:
            top_intent, conf = intents[0]
            if conf >= 0.2:
                base_result['deeper_analytics_available'] = {
                    'engine': top_intent,
                    'confidence': conf,
                    'hint': self._get_analytics_hint(top_intent),
                }

        return base_result


    def _build_survival_narrative(self, result: Dict) -> str:
        n = result.get('n', 0)
        events = result.get('n_events', 0)
        c_idx = result.get('c_index', 0)
        horizon = result.get('horizon_days', 90)
        event_rate = result.get('event_rate', (events / n * 100) if n > 0 else 0)

        narrative = (
            f"READMISSION RISK ANALYSIS — {horizon}-Day Horizon\n\n"
            f"KEY FINDING: Of {n:,} discharge episodes, {events:,} resulted in readmission "
            f"({event_rate:.1f}% readmission rate). "
        )

        hazard_ratios = result.get('hazard_ratios', {})
        if hazard_ratios:
            sorted_hr = sorted(hazard_ratios.items(), key=lambda x: -abs(x[1] - 1.0))
            risk_factors = []
            protective_factors = []
            for name, hr in sorted_hr:
                if abs(hr - 1.0) < 0.01:
                    continue
                readable_name = name.replace('_', ' ').title()
                if hr > 1.05:
                    risk_factors.append((readable_name, hr))
                elif hr < 0.95:
                    protective_factors.append((readable_name, hr))

            if risk_factors:
                narrative += "\n\nRISK FACTORS (what increases readmission):\n"
                for name, hr in risk_factors[:5]:
                    pct_increase = (hr - 1.0) * 100
                    narrative += f"  • {name}: {pct_increase:+.1f}% higher readmission risk per unit increase\n"

            if protective_factors:
                narrative += "\nPROTECTIVE FACTORS (what reduces readmission):\n"
                for name, hr in protective_factors[:3]:
                    pct_decrease = (1.0 - hr) * 100
                    narrative += f"  • {name}: {pct_decrease:.1f}% lower readmission risk per unit increase\n"

        surv_probs = result.get('survival_probabilities', {})
        hr_probs = result.get('high_risk_survival', {})
        if surv_probs:
            narrative += "\nREADMISSION TIMELINE (average patient):\n"
            for day in sorted(surv_probs.keys()):
                p = surv_probs[day]
                readmit_pct = (1 - p) * 100
                narrative += f"  Day {day}: {readmit_pct:.1f}% cumulative readmission probability\n"

            if hr_probs:
                narrative += "\nHIGH-RISK PATIENTS (75th percentile):\n"
                for day in sorted(hr_probs.keys()):
                    p = hr_probs[day]
                    readmit_pct = (1 - p) * 100
                    narrative += f"  Day {day}: {readmit_pct:.1f}% cumulative readmission probability\n"

        narrative += (
            f"\nACTION ITEMS:\n"
            f"  1. Focus discharge planning on the first 14 days — this is when readmission risk rises fastest\n"
            f"  2. {'High-risk patients need immediate post-discharge follow-up (first 7 days)' if hr_probs else 'Implement post-discharge follow-up protocol'}\n"
            f"  3. Monitor the risk factors above — each is a lever for intervention\n"
        )

        quality = 'good' if c_idx > 0.65 else 'moderate' if c_idx > 0.55 else 'limited'
        narrative += f"\nMODEL QUALITY: c-index {c_idx:.3f} ({quality} — {'suitable for clinical decision support' if c_idx > 0.6 else 'useful for population-level trends, verify with clinical judgment for individual patients'})."

        return narrative

    def _build_fairness_narrative(self, result: Dict) -> str:
        overall = result.get('overall', 'unknown')
        n = result.get('n_predictions', 0)
        reports = result.get('reports', {})

        verdict_map = {
            'acceptable': 'No significant disparities detected — model appears equitable.',
            'minor_disparity': 'Minor disparities detected — monitoring recommended.',
            'material_disparity': 'Material disparities found — intervention required.',
            'severe_disparity': 'Severe disparities detected — immediate remediation needed.',
        }

        narrative = (
            f"Health Equity Audit: Evaluated {n:,} predictions across "
            f"{len(reports)} demographic dimensions. "
            f"Overall verdict: {overall.replace('_', ' ').title()}. "
            f"{verdict_map.get(overall, '')} "
        )

        for attr, rep in reports.items():
            dpd = rep.get('dpd', 0)
            auc_gap = rep.get('auc_gap', 0)
            narrative += (
                f"{attr.title()}: DPD={dpd:.3f}, AUC gap={auc_gap:.3f} "
                f"({rep.get('verdict', 'unknown').replace('_', ' ')}). "
            )

        return narrative

    def _build_causal_narrative(self, result: Dict, analysis_type: str) -> str:
        if 'ate' not in result:
            error = result.get('error', result.get('status', 'unknown'))
            return f"Causal analysis ({analysis_type}): Could not estimate — {error}"

        ate = result.get('ate', 0)
        ci = result.get('ate_ci95', (0, 0))
        n = result.get('n', 0)
        n_treated = result.get('n_treated', 0)
        n_control = result.get('n_control', 0)
        method = result.get('method', 'DoubleML')

        if analysis_type == 'telehealth':
            treatment_desc = "telehealth adoption"
            outcome_desc = "total annual cost"
            outcome_unit = "$"
        else:
            treatment_desc = "post-discharge follow-up within 14 days"
            outcome_desc = "30-day readmission probability"
            outcome_unit = ""

        significant = (ci[0] > 0 or ci[1] < 0)
        direction = 'increases' if ate > 0 else 'decreases'

        narrative = (
            f"CAUSAL IMPACT ANALYSIS — {treatment_desc.title()}\n\n"
            f"METHOD: {method} (controls for confounders via cross-fitted ML models)\n"
            f"SAMPLE: {n:,} observations ({n_treated:,} treated, {n_control:,} control)\n\n"
        )

        if outcome_unit == '$':
            narrative += (
                f"FINDING: {treatment_desc.title()} {direction} {outcome_desc} by "
                f"${abs(ate):,.0f} per member (95% CI: [${ci[0]:,.0f}, ${ci[1]:,.0f}])\n\n"
            )
        else:
            narrative += (
                f"FINDING: {treatment_desc.title()} {direction} {outcome_desc} by "
                f"{abs(ate):.1%} (95% CI: [{ci[0]:.1%}, {ci[1]:.1%}])\n\n"
            )

        if significant:
            narrative += (
                f"STATISTICAL SIGNIFICANCE: YES — the confidence interval does not include zero.\n"
                f"This means the effect is real, not due to chance.\n\n"
            )
        else:
            narrative += (
                f"STATISTICAL SIGNIFICANCE: NO — the confidence interval includes zero.\n"
                f"This means we cannot rule out that the observed difference is due to chance, "
                f"not the intervention itself.\n\n"
            )

        narrative += "WHAT THIS MEANS FOR THE BUSINESS:\n"
        if analysis_type == 'telehealth':
            if not significant:
                narrative += (
                    f"  Telehealth usage does NOT significantly change total costs. "
                    f"This is actually GOOD NEWS — it means telehealth provides equivalent care "
                    f"at similar cost, while improving access and convenience. "
                    f"Keep investing in telehealth for member satisfaction and access, "
                    f"not for cost reduction.\n"
                )
            elif ate < 0:
                narrative += (
                    f"  Telehealth saves ${abs(ate):,.0f} per member annually. "
                    f"Scaling to all {n_control:,} non-telehealth members could save "
                    f"${abs(ate) * n_control:,.0f}/year.\n"
                )
            else:
                narrative += (
                    f"  Telehealth increases costs by ${ate:,.0f} per member — likely because "
                    f"it increases access/utilization. Investigate whether this reflects "
                    f"better care (catching issues earlier) or overutilization.\n"
                )
        else:
            if significant and ate < 0:
                narrative += (
                    f"  Post-discharge follow-up reduces readmission by {abs(ate):.1%}. "
                    f"For every 100 discharged patients who get follow-up, "
                    f"we prevent {abs(ate)*100:.1f} readmissions.\n"
                )
            else:
                narrative += (
                    f"  The follow-up program's effect on readmission is not conclusive. "
                    f"Consider expanding the program or improving timing (earlier follow-up).\n"
                )

        interp = result.get('interpretation', '')
        if interp:
            narrative += f"\nMODEL INTERPRETATION: {interp}"

        return narrative

    def _build_forest_narrative(self, result: Dict) -> str:
        ate = result.get('ate', 0)
        ate_se = result.get('ate_se', 0)
        n = result.get('n', 0)
        top_decile = result.get('cate_top_decile', 0)
        bottom_decile = result.get('cate_bottom_decile', 0)
        het_ratio = result.get('heterogeneity_ratio', 0)

        narrative = (
            f"Causal Forest (Heterogeneous Treatment Effects): "
            f"Average treatment effect: {ate:.3f} ± {ate_se:.3f} (N={n:,}). "
            f"Top decile CATE: {top_decile:.3f}, Bottom decile: {bottom_decile:.3f}. "
            f"Heterogeneity ratio: {het_ratio:.2f}x — "
        )

        if het_ratio > 3:
            narrative += "Strong heterogeneity: treatment effects vary dramatically across patients. Personalized targeting recommended."
        elif het_ratio > 1.5:
            narrative += "Moderate heterogeneity: some patient subgroups benefit more than others."
        else:
            narrative += "Low heterogeneity: treatment effect is relatively uniform across patients."

        policy = result.get('policy', {})
        if policy:
            pv = policy.get('policy_value', 0)
            narrative += f" Policy evaluation (treat top 20%): value={pv:.3f}."

        return narrative

    def _build_dq_narrative(self, result: Dict) -> str:
        overall = result.get('overall_score', 0)
        tables = result.get('tables', {})
        n_tables = len(tables)

        narrative = (
            f"Data Quality Report: Overall score {overall:.1f}/100 across {n_tables} tables. "
        )

        if tables:
            sorted_tables = sorted(tables.items(), key=lambda x: x[1].get('table_score', 0))
            worst = sorted_tables[0]
            best = sorted_tables[-1]
            narrative += (
                f"Best: {best[0]} ({best[1].get('table_score', 0):.1f}). "
                f"Needs attention: {worst[0]} ({worst[1].get('table_score', 0):.1f}). "
            )

            issues = []
            for tbl_name, tbl_report in tables.items():
                for col_report in tbl_report.get('columns', {}).values():
                    if col_report.get('completeness', 100) < 80:
                        issues.append(f"{tbl_name}.{col_report.get('column', '?')} has {col_report.get('completeness', 0):.0f}% completeness")
            if issues:
                narrative += f"Key issues: {'; '.join(issues[:5])}."

        return narrative

    def _build_scan_narrative(self, report: Dict) -> str:
        engines = report.get('engines', {})
        ok = report['summary']['engines_ok']
        total = report['summary']['engines_run']
        latency = report['summary']['total_latency_ms']

        narrative = (
            f"Full Intelligence Scan: {ok}/{total} engines completed successfully "
            f"in {latency:,}ms. "
        )

        for name, result in engines.items():
            if result.get('status') != 'ok':
                continue
            eng_narrative = result.get('narrative', '')
            if eng_narrative:
                first_sentence = eng_narrative.split('. ')[0] + '.'
                narrative += f"\n• {name}: {first_sentence} "

        return narrative


    def _extract_primary_table(self, sql: str) -> Optional[str]:
        import re
        match = re.search(r'FROM\s+(\w+)', sql, re.IGNORECASE)
        return match.group(1) if match else None

    def _get_analytics_hint(self, engine: str) -> str:
        hints = {
            'survival': 'Survival curves and readmission probability analysis available — ask about "readmission survival analysis"',
            'fairness': 'Health equity audit available — ask about "fairness audit" or "bias analysis"',
            'causal': 'Causal impact analysis available — ask "what if we implemented follow-up care?"',
            'causal_forest': 'Personalized treatment effect analysis available — ask "who benefits most from this intervention?"',
            'dq_deep': 'Detailed data quality report available — ask about "data quality score"',
            'executive': 'Executive intelligence briefing available — ask for "CFO briefing" or "executive summary"',
            'preventive': 'Preventive care ROI analysis available — ask about "preventive care opportunities"',
            'chronic_risk': 'Chronic risk prediction available — ask about "pre-chronic risk scoring"',
        }
        return hints.get(engine, 'Advanced analytics available for this topic.')

    def get_available_engines(self) -> Dict[str, bool]:
        status = {}
        checkers = {
            'dq_engine': self._get_dq_engine,
            'survival': self._get_survival,
            'fairness': self._get_fairness,
            'causal_inference': self._get_causal,
            'causal_forest': self._get_causal_forest,
            'deep_analytics': self._get_deep_analytics,
            'chronic_risk': self._get_chronic_risk,
        }
        for name, checker in checkers.items():
            try:
                status[name] = checker() is not None
            except Exception:
                status[name] = False
        try:
            from business_insights import rising_risk_cohort
            status['business_insights'] = True
        except Exception:
            status['business_insights'] = False
        try:
            from kpi_discovery import get_discovery
            status['kpi_discovery'] = True
        except Exception:
            status['kpi_discovery'] = False
        return status
