from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

_log = logging.getLogger(__name__)


def smooth_rate(successes: float, trials: float,
                *, prior_rate: float, prior_strength: float = 30.0) -> float:
    prior_rate = max(min(prior_rate, 1.0), 0.0)
    alpha = prior_rate * prior_strength
    beta = (1 - prior_rate) * prior_strength
    return (successes + alpha) / max(trials + alpha + beta, 1e-9)


def smooth_mean(sum_x: float, n: float,
                *, prior_mean: float, prior_strength: float = 30.0) -> float:
    return (sum_x + prior_mean * prior_strength) / max(n + prior_strength, 1e-9)


def credible_interval(successes: float, trials: float,
                       *, prior_rate: float,
                       prior_strength: float = 30.0,
                       z: float = 1.96) -> Tuple[float, float]:
    p = smooth_rate(successes, trials,
                     prior_rate=prior_rate, prior_strength=prior_strength)
    n_eff = trials + prior_strength
    se = math.sqrt(max(p * (1 - p) / max(n_eff, 1.0), 1e-12))
    return max(p - z * se, 0.0), min(p + z * se, 1.0)


def reliability(trials: float, *, cutoff: float = 30.0) -> float:
    if trials <= 0:
        return 0.0
    return trials / (trials + cutoff)


_ICD_PREFIX_TO_GROUP: Dict[str, str] = {
    'E10':  'Diabetes Type 1',
    'E11':  'Diabetes Type 2',
    'E13':  'Diabetes (other)',
    'E66':  'Obesity',
    'E78':  'Hyperlipidemia',
    'I10':  'Hypertension',
    'I11':  'Hypertension',
    'I12':  'Hypertension',
    'I13':  'Hypertension',
    'I20':  'Ischemic heart disease',
    'I21':  'Acute MI',
    'I25':  'Chronic ischemic heart disease',
    'I48':  'Atrial fibrillation',
    'I50':  'Heart failure',
    'J44':  'COPD',
    'J45':  'Asthma',
    'J18':  'Pneumonia',
    'N18':  'Chronic kidney disease',
    'F32':  'Depression',
    'F33':  'Depression',
    'F41':  'Anxiety',
    'F17':  'Tobacco use disorder',
    'F10':  'Alcohol use disorder',
    'F11':  'Opioid use disorder',
    'C50':  'Breast cancer',
    'C34':  'Lung cancer',
    'C18':  'Colon cancer',
    'C61':  'Prostate cancer',
    'G30':  'Alzheimer disease',
    'G20':  'Parkinson disease',
    'G40':  'Epilepsy',
    'M17':  'Osteoarthritis knee',
    'M54':  'Back pain',
    'Z00':  'Preventive / wellness',
    'Z12':  'Screening',
    'Z13':  'Screening',
    'Z23':  'Immunization',
    'Z33':  'Pregnancy encounter',
    'O09':  'High-risk pregnancy',
}


def icd_to_group(icd: Optional[str]) -> str:
    if not icd:
        return 'Unspecified'
    code = icd.strip().upper().replace('.', '')
    for n in (3, 4):
        if len(code) >= n and code[:n] in _ICD_PREFIX_TO_GROUP:
            return _ICD_PREFIX_TO_GROUP[code[:n]]
    if code.startswith('I'):
        return 'Other cardiovascular'
    if code.startswith('J'):
        return 'Other respiratory'
    if code.startswith('E'):
        return 'Other endocrine'
    if code.startswith('F'):
        return 'Other behavioral health'
    if code.startswith('C') or code.startswith('D'):
        return 'Other oncology / hematology'
    if code.startswith('K'):
        return 'Other GI'
    if code.startswith('M'):
        return 'Other musculoskeletal'
    if code.startswith('N'):
        return 'Other GU / renal'
    if code.startswith('R'):
        return 'Signs & symptoms (ill-defined)'
    if code.startswith('S') or code.startswith('T'):
        return 'Injury / poisoning'
    if code.startswith('Z'):
        return 'Encounter for screening / preventive'
    return 'Other'


def group_counts(rows: List[Dict[str, str]],
                 dx_key: str = 'primary_dx') -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rows:
        g = icd_to_group(r.get(dx_key))
        out[g] = out.get(g, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def apply_hcc_map(icd: str, hcc_map: Optional[Dict[str, str]]) -> Optional[str]:
    if not hcc_map:
        return None
    code = (icd or '').strip().upper().replace('.', '')
    for n in (5, 4, 3):
        if code[:n] in hcc_map:
            return hcc_map[code[:n]]
    return None
