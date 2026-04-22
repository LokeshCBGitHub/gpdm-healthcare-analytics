from __future__ import annotations

import re
import sqlite3
from typing import Dict, List, Optional, Tuple, Any

try:
    import business_insights as _bi
except Exception:
    _bi = None


_REGION_SYNONYMS = {
    'south': 'South', 'southern': 'South',
    'north': 'North', 'northern': 'North',
    'east': 'East', 'eastern': 'East',
    'west': 'West', 'western': 'West',
    'norcal': 'NorCal', 'socal': 'SoCal',
    'mid-atlantic': 'Mid-Atlantic', 'midwest': 'Midwest',
    'northwest': 'Northwest', 'southeast': 'Southeast',
    'southwest': 'Southwest',
}

_LOB_SYNONYMS = {
    'medicare advantage': 'MA', 'medicare': 'Medicare', 'ma plan': 'MA',
    'commercial': 'Commercial', 'employer': 'Commercial',
    'medicaid': 'Medicaid', 'medi-cal': 'Medicaid',
    'aca': 'Exchange', 'exchange': 'Exchange', 'individual': 'Exchange',
}

_CONDITION_TO_DX_PREFIX = {
    'diabetic': 'E11', 'diabetics': 'E11', 'diabetes': 'E11', 'dm': 'E11',
    'hypertension': 'I10', 'htn': 'I10', 'high blood pressure': 'I10',
    'copd': 'J44', 'asthma': 'J45', 'chf': 'I50', 'heart failure': 'I50',
    'ckd': 'N18', 'kidney disease': 'N18',
    'depression': 'F32', 'anxiety': 'F41',
    'cancer': 'C', 'oncology': 'C',
}

_HEDIS_ALIASES = {
    'cbp': 'CBP', 'blood pressure': 'CBP', 'hypertension control': 'CBP',
    'hbd': 'HBD', 'hba1c': 'HBD', 'a1c': 'HBD', 'diabetes control': 'HBD',
    'bcs': 'BCS', 'breast cancer': 'BCS', 'mammogram': 'BCS',
    'ccs': 'CCS', 'cervical': 'CCS', 'pap': 'CCS',
    'col': 'COL', 'colon': 'COL', 'colorectal': 'COL', 'colonoscopy': 'COL',
    'awv': 'AWV', 'wellness visit': 'AWV', 'annual wellness': 'AWV',
}

_SPECIALTY_SYNONYMS = {
    'cardiology': 'Cardiology', 'cardio': 'Cardiology', 'cardiologist': 'Cardiology',
    'primary care': 'Primary Care', 'pcp': 'Primary Care', 'family medicine': 'Primary Care',
    'internal medicine': 'Internal Medicine',
    'endocrinology': 'Endocrinology', 'endo': 'Endocrinology',
    'nephrology': 'Nephrology',
    'oncology': 'Oncology',
    'orthopedics': 'Orthopedics', 'ortho': 'Orthopedics',
    'behavioral health': 'Behavioral Health', 'psychiatry': 'Behavioral Health',
}


def _match_first(text: str, table: Dict[str, str]) -> Optional[str]:
    t = text.lower()
    for k in sorted(table.keys(), key=len, reverse=True):
        if re.search(rf"\b{re.escape(k)}\b", t):
            return table[k]
    words = set(re.findall(r'\b\w+\b', t))
    for k in sorted(table.keys(), key=len, reverse=True):
        if ' ' in k or len(k) < 4:
            continue
        for w in words:
            if abs(len(w) - len(k)) > 1:
                continue
            try:
                from dynamic_sql_engine import _edit_distance
                if 0 < _edit_distance(w, k) <= 1:
                    return table[k]
            except ImportError:
                break
    return None


def _extract_top_n(text: str, default: int = 100) -> int:
    m = re.search(r'\btop\s+(\d{1,4})\b', text, re.I)
    if m:
        return max(1, min(5000, int(m.group(1))))
    m = re.search(r'\b(\d{1,4})\s+(members|patients|providers|people)\b', text, re.I)
    if m:
        return max(1, min(5000, int(m.group(1))))
    return default


def _extract_horizon(text: str, default: int = 30) -> int:
    m = re.search(r'\b(30|60|90|180|365)\s*(?:day|days|d)\b', text, re.I)
    if m:
        return int(m.group(1))
    return default


def _detect_intent(text: str) -> str:
    t = text.lower()
    if re.search(r'\breadmit|readmission|post[- ]?discharge|tcm|transitional care\b', t):
        return 'readmit_watchlist'
    if re.search(r'\bhedis|stars?\b|quality gap|gap[s]? in care|preventive', t):
        return 'hedis_gap_list'
    if any(k in t for k in _HEDIS_ALIASES.keys()):
        return 'hedis_gap_list'
    if re.search(r'\bprovider|physician|doctor|network|specialist|cardio(logy|logist)|'
                  r'o/e|worst performing|best performing\b', t):
        return 'network_performance'
    if re.search(r'\brising[- ]?risk|cost trajectory|trending up|high[- ]?cost|'
                  r'pmpm|outreach|care management|high risk|risk stratif\b', t):
        return 'rising_risk_cohort'
    return 'unknown'


def _normalize_input(text: str) -> str:
    try:
        from dynamic_sql_engine import normalize_typos
        return normalize_typos(text)
    except ImportError:
        return text


def route(question: str, db_path: str) -> Dict[str, Any]:
    if not question or not question.strip():
        return {'intent': 'empty', 'headline': 'Please enter a question.',
                'columns': [], 'rows': [], 'row_count': 0}
    if _bi is None:
        return {'intent': 'error',
                'headline': 'business_insights module unavailable',
                'columns': [], 'rows': [], 'row_count': 0}

    text = _normalize_input(question.strip())
    intent = _detect_intent(text)
    top_n = _extract_top_n(text)
    region = _match_first(text, _REGION_SYNONYMS)
    lob = _match_first(text, _LOB_SYNONYMS)
    condition = _match_first(text, _CONDITION_TO_DX_PREFIX)

    try:
        if intent == 'readmit_watchlist':
            horizon = _extract_horizon(text, default=30)
            ans = _bi.readmit_watchlist(
                db_path, horizon_days=horizon,
                days_since_discharge_max=14, top_n=top_n, region=region
            )
        elif intent == 'hedis_gap_list':
            measure = _match_first(text, _HEDIS_ALIASES) or 'CBP'
            ans = _bi.hedis_gap_list(
                db_path, measure=measure, region=region, top_n=top_n
            )
        elif intent == 'network_performance':
            specialty = _match_first(text, _SPECIALTY_SYNONYMS)
            metric = 'cost_efficiency' if re.search(r'\bcost|expensive|efficien', text, re.I) \
                     else 'risk_adjusted_readmit'
            ans = _bi.network_performance(
                db_path, specialty=specialty, metric=metric, top_n=top_n
            )
        elif intent == 'rising_risk_cohort':
            ans = _bi.rising_risk_cohort(
                db_path, region=region, line_of_business=lob,
                chronic_condition=condition, top_n=top_n
            )
        else:
            return _fallback_to_intelligent_pipeline(text, db_path)

        payload = ans.to_dict()
        payload['narrative'] = _compose_narrative(payload)
        payload['intent_detected'] = intent
        payload['filters_detected'] = {
            'region': region, 'lob': lob, 'condition_dx_prefix': condition,
            'top_n': top_n,
        }
        payload['is_business_answer'] = True
        return payload

    except Exception as e:
        return {'intent': intent, 'error': str(e),
                 'headline': f"Business insight failed: {e}",
                 'columns': [], 'rows': [], 'row_count': 0,
                 'is_business_answer': False}


def _compose_narrative(payload: Dict[str, Any]) -> str:
    headline = payload.get('headline', '')
    action = payload.get('suggested_action', '')
    kpi = payload.get('kpi') or {}
    src = payload.get('data_source', 'unknown')

    kpi_lines = []
    for k, v in kpi.items():
        if isinstance(v, (int, float)) and 'usd' in k:
            kpi_lines.append(f"  • {k.replace('_',' ').title()}: ${v:,.0f}")
        elif isinstance(v, (int, float)):
            kpi_lines.append(f"  • {k.replace('_',' ').title()}: {v}")
    kpi_block = "\n".join(kpi_lines) if kpi_lines else "  (no aggregate KPIs)"

    return (f"{headline}\n\nKey figures:\n{kpi_block}\n\n"
            f"Recommended next step: {action}\n\nData source: {src}")


def _fallback_to_intelligent_pipeline(text: str, db_path: str) -> Dict[str, Any]:
    try:
        from intelligent_pipeline import IntelligentPipeline
        pipe = IntelligentPipeline(db_path=db_path)
        out = pipe.process(text)
        out['is_business_answer'] = False
        out['intent_detected'] = 'general_analytics'
        return out
    except Exception as e:
        return {'intent': 'unrouted',
                'headline': ("I couldn't map that to a business insight and the "
                              "general analytics pipeline is unavailable."),
                'error': str(e), 'columns': [], 'rows': [], 'row_count': 0,
                'is_business_answer': False,
                'suggestions': [
                    "Try: top 100 rising-risk members in South",
                    "Try: readmit watchlist for Medicare Advantage last 7 days",
                    "Try: CBP gaps in NorCal",
                    "Try: worst cardiology providers for readmit",
                ]}


def get_router_status() -> Dict[str, Any]:
    return {
        'status': 'ok',
        'business_insights_loaded': _bi is not None,
        'supported_intents': ['rising_risk_cohort', 'readmit_watchlist',
                               'hedis_gap_list', 'network_performance'],
        'fallback': 'intelligent_pipeline',
        'extractable_filters': ['region', 'line_of_business', 'chronic_condition',
                                 'hedis_measure', 'specialty', 'top_n', 'horizon_days'],
        'example_questions': [
            "top 100 rising-risk diabetics in South",
            "readmit watchlist Medicare 30 day",
            "CBP gaps in NorCal top 200",
            "worst cardiology providers for cost efficiency",
            "high risk CHF members",
        ],
    }
