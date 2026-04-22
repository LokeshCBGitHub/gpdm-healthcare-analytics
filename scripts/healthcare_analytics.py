import re
import math
import sqlite3
import logging
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger('gpdm.analytics')


def _connect(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _percentile(data: list, p: float) -> float:
    if not data:
        return 0.0
    k = (len(data) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(data):
        return data[f]
    return data[f] + (data[c] - data[f]) * (k - f)


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ez = math.exp(x)
        return ez / (1.0 + ez)


def _zscore(value: float, mean: float, std: float) -> float:
    return (value - mean) / std if std > 0 else 0.0


@dataclass
class MemberRiskProfile:
    member_id: str
    name: str = ''
    risk_score: float = 0.0
    risk_tier: str = 'Low'
    clinical_score: float = 0.0
    utilization_score: float = 0.0
    cost_score: float = 0.0
    stability_score: float = 0.0
    top_factors: List[str] = field(default_factory=list)
    chronic_count: int = 0
    total_claims: int = 0
    total_cost: float = 0.0
    er_visits: int = 0
    inpatient_days: int = 0
    region: str = ''
    plan_type: str = ''


class MemberRiskStratifier:

    TIERS = [
        (85, 'Critical'),
        (70, 'High'),
        (55, 'Rising'),
        (35, 'Moderate'),
        (0, 'Low'),
    ]

    WEIGHTS = {
        'clinical': 0.30,
        'utilization': 0.25,
        'cost': 0.25,
        'stability': 0.20,
    }

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._profiles: Dict[str, MemberRiskProfile] = {}
        self._population_stats = {}

    def train(self):
        conn = _connect(self.db_path)

        members = {}
        for row in conn.execute("""
            SELECT MEMBER_ID, FIRST_NAME, LAST_NAME, DATE_OF_BIRTH, GENDER,
                   KP_REGION, PLAN_TYPE, RISK_SCORE, CHRONIC_CONDITIONS,
                   ENROLLMENT_DATE
            FROM members
        """):
            members[row['MEMBER_ID']] = dict(row)

        claim_stats = defaultdict(lambda: {'count': 0, 'total_cost': 0.0,
                                            'denied': 0, 'costs': []})
        for row in conn.execute("""
            SELECT MEMBER_ID, PAID_AMOUNT, CLAIM_STATUS, SERVICE_DATE
            FROM claims WHERE CLAIM_STATUS IN ('PAID', 'DENIED', 'ADJUSTED')
        """):
            mid = row['MEMBER_ID']
            cs = claim_stats[mid]
            cs['count'] += 1
            amt = float(row['PAID_AMOUNT'] or 0)
            cs['total_cost'] += amt
            cs['costs'].append(amt)
            if row['CLAIM_STATUS'] == 'DENIED':
                cs['denied'] += 1

        enc_stats = defaultdict(lambda: {'er': 0, 'inpatient': 0, 'ip_days': 0,
                                          'total': 0})
        for row in conn.execute("""
            SELECT MEMBER_ID, VISIT_TYPE, LENGTH_OF_STAY
            FROM encounters
        """):
            mid = row['MEMBER_ID']
            es = enc_stats[mid]
            es['total'] += 1
            if row['VISIT_TYPE'] == 'EMERGENCY':
                es['er'] += 1
            if row['VISIT_TYPE'] == 'INPATIENT':
                es['inpatient'] += 1
                es['ip_days'] += int(row['LENGTH_OF_STAY'] or 0)

        dx_stats = defaultdict(lambda: {'unique_dx': set(), 'chronic': 0})
        for row in conn.execute("""
            SELECT MEMBER_ID, ICD10_CODE, IS_CHRONIC FROM diagnoses
        """):
            mid = row['MEMBER_ID']
            dx_stats[mid]['unique_dx'].add(row['ICD10_CODE'])
            if row['IS_CHRONIC'] in ('TRUE', '1', 'Y', 'Yes'):
                dx_stats[mid]['chronic'] += 1

        rx_stats = defaultdict(lambda: {'unique_meds': set(), 'total_cost': 0.0})
        for row in conn.execute("""
            SELECT MEMBER_ID, MEDICATION_CLASS, COST FROM prescriptions
        """):
            mid = row['MEMBER_ID']
            rx_stats[mid]['unique_meds'].add(row['MEDICATION_CLASS'])
            rx_stats[mid]['total_cost'] += float(row['COST'] or 0)

        conn.close()

        all_costs = [cs['total_cost'] for cs in claim_stats.values() if cs['total_cost'] > 0]
        all_costs.sort()
        all_er = [es['er'] for es in enc_stats.values()]
        all_ip_days = [es['ip_days'] for es in enc_stats.values()]
        all_claims = [cs['count'] for cs in claim_stats.values()]

        self._population_stats = {
            'cost_median': _percentile(all_costs, 50),
            'cost_p75': _percentile(all_costs, 75),
            'cost_p90': _percentile(all_costs, 90),
            'cost_mean': np.mean(all_costs) if all_costs else 0,
            'cost_std': np.std(all_costs) if all_costs else 1,
            'er_mean': np.mean(all_er) if all_er else 0,
            'er_std': np.std(all_er) if all_er else 1,
            'ip_days_mean': np.mean(all_ip_days) if all_ip_days else 0,
            'ip_days_std': np.std(all_ip_days) if all_ip_days else 1,
            'claims_mean': np.mean(all_claims) if all_claims else 0,
            'claims_std': np.std(all_claims) if all_claims else 1,
            'total_members': len(members),
        }

        for mid, mdata in members.items():
            cs = claim_stats.get(mid, {'count': 0, 'total_cost': 0, 'denied': 0, 'costs': []})
            es = enc_stats.get(mid, {'er': 0, 'inpatient': 0, 'ip_days': 0, 'total': 0})
            dx = dx_stats.get(mid, {'unique_dx': set(), 'chronic': 0})
            rx = rx_stats.get(mid, {'unique_meds': set(), 'total_cost': 0})

            profile = self._score_member(mid, mdata, cs, es, dx, rx)
            self._profiles[mid] = profile

        self._profiles = dict(
            sorted(self._profiles.items(), key=lambda x: x[1].risk_score, reverse=True)
        )

        tier_counts = Counter(p.risk_tier for p in self._profiles.values())
        logger.info("MemberRiskStratifier: scored %d members — %s",
                     len(self._profiles),
                     ', '.join(f"{t}:{c}" for t, c in tier_counts.most_common()))

    def _score_member(self, mid, mdata, cs, es, dx, rx) -> MemberRiskProfile:
        ps = self._population_stats
        factors = []

        chronic_count = int(mdata.get('CHRONIC_CONDITIONS') or 0)
        unique_dx_count = len(dx['unique_dx'])
        polypharmacy = len(rx['unique_meds'])

        clinical = 0.0
        if chronic_count >= 5:
            clinical += 40
            factors.append(f"{chronic_count} chronic conditions")
        elif chronic_count >= 3:
            clinical += 25
        elif chronic_count >= 1:
            clinical += 10

        if unique_dx_count > 10:
            clinical += 30
            factors.append(f"{unique_dx_count} unique diagnoses")
        elif unique_dx_count > 5:
            clinical += 15

        if polypharmacy >= 5:
            clinical += 20
            factors.append(f"{polypharmacy} medication classes (polypharmacy)")
        elif polypharmacy >= 3:
            clinical += 10

        if dx['chronic'] > 5:
            clinical += 10

        clinical = min(clinical, 100)

        utilization = 0.0
        er_z = _zscore(es['er'], ps['er_mean'], ps['er_std'])
        ip_z = _zscore(es['ip_days'], ps['ip_days_mean'], ps['ip_days_std'])

        if er_z > 2.0:
            utilization += 40
            factors.append(f"{es['er']} ER visits (>2 SD above mean)")
        elif er_z > 1.0:
            utilization += 20
        elif es['er'] > 0:
            utilization += 5

        if ip_z > 2.0:
            utilization += 35
            factors.append(f"{es['ip_days']} inpatient days (high)")
        elif ip_z > 1.0:
            utilization += 15

        if es['total'] > 10:
            utilization += 15
        elif es['total'] > 5:
            utilization += 5

        if cs['count'] > 0:
            denial_rate = cs['denied'] / cs['count']
            if denial_rate > 0.2:
                utilization += 10
                factors.append(f"{denial_rate:.0%} claim denial rate")

        utilization = min(utilization, 100)

        cost_z = _zscore(cs['total_cost'], ps['cost_mean'], ps['cost_std'])
        cost = 0.0
        if cost_z > 2.5:
            cost = 90
            factors.append(f"${cs['total_cost']:,.0f} total cost (top 1%)")
        elif cost_z > 2.0:
            cost = 70
            factors.append(f"${cs['total_cost']:,.0f} total cost (top 5%)")
        elif cost_z > 1.0:
            cost = 45
        elif cost_z > 0:
            cost = 20
        else:
            cost = 5

        if rx['total_cost'] > 5000:
            cost += 10
            factors.append(f"${rx['total_cost']:,.0f} pharmacy spend")

        cost = min(cost, 100)

        stability = 0.0
        costs = cs.get('costs', [])
        if len(costs) >= 4:
            half = len(costs) // 2
            first_half_avg = np.mean(costs[:half])
            second_half_avg = np.mean(costs[half:])
            if first_half_avg > 0:
                growth = (second_half_avg - first_half_avg) / first_half_avg
                if growth > 0.5:
                    stability = 80
                    factors.append(f"Cost accelerating ({growth:.0%} growth)")
                elif growth > 0.2:
                    stability = 50
                elif growth > 0:
                    stability = 25
                else:
                    stability = 10
        else:
            stability = 30

        composite = (
            self.WEIGHTS['clinical'] * clinical +
            self.WEIGHTS['utilization'] * utilization +
            self.WEIGHTS['cost'] * cost +
            self.WEIGHTS['stability'] * stability
        )

        tier = 'Low'
        for threshold, tier_name in self.TIERS:
            if composite >= threshold:
                tier = tier_name
                break

        return MemberRiskProfile(
            member_id=mid,
            name=f"{mdata.get('FIRST_NAME', '')} {mdata.get('LAST_NAME', '')}",
            risk_score=round(composite, 2),
            risk_tier=tier,
            clinical_score=round(clinical, 2),
            utilization_score=round(utilization, 2),
            cost_score=round(cost, 2),
            stability_score=round(stability, 2),
            top_factors=factors[:5],
            chronic_count=chronic_count,
            total_claims=cs['count'],
            total_cost=round(cs['total_cost'], 2),
            er_visits=es['er'],
            inpatient_days=es['ip_days'],
            region=mdata.get('KP_REGION', ''),
            plan_type=mdata.get('PLAN_TYPE', ''),
        )

    def get_high_risk(self, top_n: int = 50) -> List[MemberRiskProfile]:
        return list(self._profiles.values())[:top_n]

    def get_tier_summary(self) -> Dict[str, Any]:
        tiers = defaultdict(lambda: {'count': 0, 'avg_score': 0, 'avg_cost': 0,
                                      'total_cost': 0, 'avg_er': 0, 'avg_chronic': 0})
        for p in self._profiles.values():
            t = tiers[p.risk_tier]
            t['count'] += 1
            t['avg_score'] += p.risk_score
            t['avg_cost'] += p.total_cost
            t['total_cost'] += p.total_cost
            t['avg_er'] += p.er_visits
            t['avg_chronic'] += p.chronic_count

        for tier, t in tiers.items():
            n = t['count'] or 1
            t['avg_score'] = round(t['avg_score'] / n, 2)
            t['avg_cost'] = round(t['avg_cost'] / n, 2)
            t['avg_er'] = round(t['avg_er'] / n, 2)
            t['avg_chronic'] = round(t['avg_chronic'] / n, 2)

        return dict(tiers)

    def get_profile(self, member_id: str) -> Optional[MemberRiskProfile]:
        return self._profiles.get(member_id)

    def get_region_risk(self) -> Dict[str, Dict]:
        regions = defaultdict(lambda: {'members': 0, 'high_risk': 0, 'avg_score': 0,
                                        'total_cost': 0})
        for p in self._profiles.values():
            r = regions[p.region]
            r['members'] += 1
            if p.risk_tier in ('Critical', 'High'):
                r['high_risk'] += 1
            r['avg_score'] += p.risk_score
            r['total_cost'] += p.total_cost

        for region, r in regions.items():
            n = r['members'] or 1
            r['avg_score'] = round(r['avg_score'] / n, 2)
            r['high_risk_pct'] = round(100 * r['high_risk'] / n, 1)

        return dict(regions)


class ReadmissionPredictor:

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._weights: Optional[np.ndarray] = None
        self._bias: float = 0.0
        self._feature_names: List[str] = []
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None
        self._metrics: Dict[str, float] = {}

    def train(self):
        conn = _connect(self.db_path)

        encounters = []
        for row in conn.execute("""
            SELECT e.ENCOUNTER_ID, e.MEMBER_ID, e.ADMIT_DATE, e.DISCHARGE_DATE,
                   e.VISIT_TYPE, e.LENGTH_OF_STAY, e.DISPOSITION,
                   m.DATE_OF_BIRTH, m.CHRONIC_CONDITIONS, m.RISK_SCORE, m.GENDER
            FROM encounters e
            JOIN members m ON e.MEMBER_ID = m.MEMBER_ID
            WHERE e.VISIT_TYPE IN ('INPATIENT', 'EMERGENCY')
            AND e.DISCHARGE_DATE IS NOT NULL AND e.DISCHARGE_DATE != ''
        """):
            encounters.append(dict(row))

        er_counts = defaultdict(int)
        for row in conn.execute("""
            SELECT MEMBER_ID, COUNT(*) as cnt FROM encounters
            WHERE VISIT_TYPE = 'EMERGENCY' GROUP BY MEMBER_ID
        """):
            er_counts[row['MEMBER_ID']] = row['cnt']

        dx_counts = defaultdict(int)
        for row in conn.execute("""
            SELECT MEMBER_ID, COUNT(DISTINCT ICD10_CODE) as cnt FROM diagnoses
            GROUP BY MEMBER_ID
        """):
            dx_counts[row['MEMBER_ID']] = row['cnt']

        rx_counts = defaultdict(int)
        for row in conn.execute("""
            SELECT MEMBER_ID, COUNT(DISTINCT MEDICATION_CLASS) as cnt FROM prescriptions
            GROUP BY MEMBER_ID
        """):
            rx_counts[row['MEMBER_ID']] = row['cnt']

        conn.close()

        enc_by_member = defaultdict(list)
        for e in encounters:
            enc_by_member[e['MEMBER_ID']].append(e)

        for mid in enc_by_member:
            enc_by_member[mid].sort(key=lambda x: x['DISCHARGE_DATE'] or '')

        self._feature_names = [
            'age', 'chronic_conditions', 'length_of_stay', 'is_emergency',
            'risk_score', 'er_visit_count', 'unique_diagnoses',
            'medication_classes', 'is_male'
        ]

        X, y = [], []
        for mid, member_encs in enc_by_member.items():
            for i, enc in enumerate(member_encs):
                readmitted = 0
                discharge = enc.get('DISCHARGE_DATE', '')
                if discharge and i + 1 < len(member_encs):
                    next_admit = member_encs[i + 1].get('ADMIT_DATE', '')
                    if next_admit and discharge:
                        try:
                            d1 = self._parse_date(discharge)
                            d2 = self._parse_date(next_admit)
                            if d2 and d1 and 0 < (d2 - d1) <= 30:
                                readmitted = 1
                        except (ValueError, TypeError):
                            pass

                age = self._compute_age(enc.get('DATE_OF_BIRTH', ''))
                features = [
                    age,
                    float(enc.get('CHRONIC_CONDITIONS') or 0),
                    float(enc.get('LENGTH_OF_STAY') or 0),
                    1.0 if enc.get('VISIT_TYPE') == 'EMERGENCY' else 0.0,
                    float(enc.get('RISK_SCORE') or 0),
                    float(er_counts.get(mid, 0)),
                    float(dx_counts.get(mid, 0)),
                    float(rx_counts.get(mid, 0)),
                    1.0 if enc.get('GENDER') == 'M' else 0.0,
                ]
                X.append(features)
                y.append(readmitted)

        if not X:
            logger.warning("ReadmissionPredictor: no training data found")
            return

        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        self._feature_means = np.mean(X, axis=0)
        self._feature_stds = np.std(X, axis=0)
        self._feature_stds[self._feature_stds == 0] = 1.0
        X_norm = (X - self._feature_means) / self._feature_stds

        n_samples, n_features = X_norm.shape
        self._weights = np.zeros(n_features)
        self._bias = 0.0
        lr = 0.01
        epochs = 200

        for epoch in range(epochs):
            z = X_norm @ self._weights + self._bias
            z = np.clip(z, -500, 500)
            predictions = 1.0 / (1.0 + np.exp(-z))

            error = predictions - y
            grad_w = (X_norm.T @ error) / n_samples
            grad_b = np.mean(error)

            self._weights -= lr * grad_w
            self._bias -= lr * grad_b

        z = X_norm @ self._weights + self._bias
        z = np.clip(z, -500, 500)
        final_pred = 1.0 / (1.0 + np.exp(-z))
        pred_labels = (final_pred >= 0.5).astype(int)

        tp = np.sum((pred_labels == 1) & (y == 1))
        fp = np.sum((pred_labels == 1) & (y == 0))
        fn = np.sum((pred_labels == 0) & (y == 1))
        tn = np.sum((pred_labels == 0) & (y == 0))

        accuracy = (tp + tn) / max(n_samples, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        readmit_rate = np.mean(y)

        self._metrics = {
            'accuracy': round(float(accuracy), 4),
            'precision': round(float(precision), 4),
            'recall': round(float(recall), 4),
            'f1': round(2 * precision * recall / max(precision + recall, 0.001), 4),
            'readmit_rate': round(float(readmit_rate), 4),
            'training_samples': n_samples,
            'positive_samples': int(np.sum(y)),
        }

        importance = np.abs(self._weights)
        self._feature_importance = {
            self._feature_names[i]: round(float(importance[i]), 4)
            for i in np.argsort(-importance)
        }

        logger.info("ReadmissionPredictor: trained on %d encounters, readmit_rate=%.2f%%, acc=%.2f%%",
                     n_samples, readmit_rate * 100, accuracy * 100)

    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        if self._weights is None:
            return {'error': 'Model not trained'}

        x = np.array([features.get(f, 0.0) for f in self._feature_names])
        x_norm = (x - self._feature_means) / self._feature_stds
        z = float(x_norm @ self._weights + self._bias)
        prob = _sigmoid(z)

        risk_level = 'High' if prob > 0.3 else 'Medium' if prob > 0.15 else 'Low'

        return {
            'readmission_probability': round(prob, 4),
            'risk_level': risk_level,
            'top_risk_factors': self._get_top_factors(x_norm),
        }

    def _get_top_factors(self, x_norm: np.ndarray) -> List[Dict]:
        contributions = x_norm * self._weights
        indices = np.argsort(-np.abs(contributions))
        factors = []
        for idx in indices[:5]:
            factors.append({
                'feature': self._feature_names[idx],
                'contribution': round(float(contributions[idx]), 4),
                'direction': 'increases risk' if contributions[idx] > 0 else 'decreases risk',
            })
        return factors

    def get_metrics(self) -> Dict[str, Any]:
        return {**self._metrics, 'feature_importance': self._feature_importance}

    @staticmethod
    def _parse_date(date_str: str) -> Optional[float]:
        if not date_str:
            return None
        try:
            parts = date_str.split('-')
            if len(parts) == 3:
                y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
                return y * 365.25 + m * 30.44 + d
        except (ValueError, TypeError):
            pass
        return None

    @staticmethod
    def _compute_age(dob: str) -> float:
        if not dob:
            return 50.0
        try:
            parts = dob.split('-')
            birth_year = int(parts[0])
            return 2026 - birth_year
        except (ValueError, IndexError):
            return 50.0


class CostAnomalyDetector:

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._anomalies: List[Dict] = []
        self._stats: Dict[str, Dict] = {}

    def train(self):
        conn = _connect(self.db_path)

        dimensions = [
            ('provider', 'RENDERING_NPI', 'providers'),
            ('region', 'KP_REGION', None),
            ('cpt_code', 'CPT_CODE', None),
            ('plan_type', 'PLAN_TYPE', None),
        ]

        all_anomalies = []

        for dim_name, dim_col, join_table in dimensions:
            query = f"""
                SELECT c.{dim_col} as group_key,
                       c.PAID_AMOUNT, c.CLAIM_ID, c.MEMBER_ID,
                       c.SERVICE_DATE, c.CPT_DESCRIPTION
                FROM claims c
                WHERE c.PAID_AMOUNT > 0 AND c.CLAIM_STATUS = 'PAID'
            """
            groups = defaultdict(list)
            claim_data = {}
            for row in conn.execute(query):
                gk = row['group_key']
                amt = float(row['PAID_AMOUNT'])
                groups[gk].append(amt)
                claim_data[(gk, len(groups[gk]) - 1)] = dict(row)

            dim_stats = {}
            for gk, costs in groups.items():
                if len(costs) < 10:
                    continue

                costs_sorted = sorted(costs)
                q1 = _percentile(costs_sorted, 25)
                q3 = _percentile(costs_sorted, 75)
                iqr = q3 - q1
                mean = np.mean(costs)
                std = np.std(costs)

                upper_fence = q3 + 2.0 * iqr
                lower_fence = max(0, q1 - 2.0 * iqr)

                dim_stats[gk] = {
                    'mean': round(mean, 2),
                    'median': round(_percentile(costs_sorted, 50), 2),
                    'std': round(std, 2),
                    'q1': round(q1, 2),
                    'q3': round(q3, 2),
                    'upper_fence': round(upper_fence, 2),
                    'count': len(costs),
                }

                for i, cost in enumerate(costs):
                    z = _zscore(cost, mean, std)
                    if cost > upper_fence and z > 2.0:
                        cd = claim_data.get((gk, i), {})
                        all_anomalies.append({
                            'dimension': dim_name,
                            'group': gk,
                            'claim_id': cd.get('CLAIM_ID', ''),
                            'member_id': cd.get('MEMBER_ID', ''),
                            'amount': cost,
                            'group_mean': round(mean, 2),
                            'group_median': round(_percentile(costs_sorted, 50), 2),
                            'z_score': round(z, 2),
                            'excess_pct': round(100 * (cost - mean) / mean, 1) if mean > 0 else 0,
                            'service_date': cd.get('SERVICE_DATE', ''),
                            'description': cd.get('CPT_DESCRIPTION', ''),
                        })

            self._stats[dim_name] = dim_stats

        conn.close()

        self._anomalies = sorted(all_anomalies, key=lambda x: x['z_score'], reverse=True)
        logger.info("CostAnomalyDetector: found %d anomalies across %d dimensions",
                     len(self._anomalies), len(dimensions))

    def get_top_anomalies(self, n: int = 50, dimension: str = None) -> List[Dict]:
        if dimension:
            filtered = [a for a in self._anomalies if a['dimension'] == dimension]
            return filtered[:n]
        return self._anomalies[:n]

    def get_anomaly_summary(self) -> Dict[str, Any]:
        summary = defaultdict(lambda: {'count': 0, 'total_excess': 0, 'avg_z': 0})
        for a in self._anomalies:
            dim = a['dimension']
            summary[dim]['count'] += 1
            summary[dim]['total_excess'] += a['amount'] - a['group_mean']
            summary[dim]['avg_z'] += a['z_score']

        for dim, s in summary.items():
            n = s['count'] or 1
            s['avg_z'] = round(s['avg_z'] / n, 2)
            s['total_excess'] = round(s['total_excess'], 2)

        return dict(summary)


@dataclass
class ProviderScore:
    npi: str
    name: str = ''
    specialty: str = ''
    region: str = ''
    composite_score: float = 0.0
    quality_score: float = 0.0
    efficiency_score: float = 0.0
    access_score: float = 0.0
    volume_score: float = 0.0
    total_encounters: int = 0
    total_claims: int = 0
    avg_cost: float = 0.0
    denial_rate: float = 0.0
    peer_rank: int = 0
    peer_total: int = 0
    grade: str = 'C'


class ProviderScorecard:

    GRADE_THRESHOLDS = [
        (90, 'A'), (75, 'B'), (55, 'C'), (35, 'D'), (0, 'F')
    ]

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._scores: Dict[str, ProviderScore] = {}

    def train(self):
        conn = _connect(self.db_path)

        providers = {}
        for row in conn.execute("""
            SELECT NPI, PROVIDER_FIRST_NAME, PROVIDER_LAST_NAME, SPECIALTY,
                   KP_REGION, PANEL_SIZE, ACCEPTS_NEW_PATIENTS, STATUS
            FROM providers WHERE STATUS = 'ACTIVE'
        """):
            providers[row['NPI']] = dict(row)

        prov_claims = defaultdict(lambda: {'count': 0, 'total_paid': 0, 'denied': 0})
        for row in conn.execute("""
            SELECT RENDERING_NPI, PAID_AMOUNT, CLAIM_STATUS FROM claims
        """):
            npi = row['RENDERING_NPI']
            pc = prov_claims[npi]
            pc['count'] += 1
            pc['total_paid'] += float(row['PAID_AMOUNT'] or 0)
            if row['CLAIM_STATUS'] == 'DENIED':
                pc['denied'] += 1

        prov_enc = defaultdict(lambda: {'count': 0, 'total_los': 0})
        for row in conn.execute("""
            SELECT RENDERING_NPI, LENGTH_OF_STAY FROM encounters
        """):
            npi = row['RENDERING_NPI']
            prov_enc[npi]['count'] += 1
            prov_enc[npi]['total_los'] += float(row['LENGTH_OF_STAY'] or 0)

        conn.close()

        specialty_groups = defaultdict(list)
        for npi, pdata in providers.items():
            pc = prov_claims.get(npi, {'count': 0, 'total_paid': 0, 'denied': 0})
            pe = prov_enc.get(npi, {'count': 0, 'total_los': 0})

            avg_cost = pc['total_paid'] / max(pe['count'], 1)
            denial_rate = pc['denied'] / max(pc['count'], 1)
            avg_los = pe['total_los'] / max(pe['count'], 1)
            panel_size = int(pdata.get('PANEL_SIZE') or 0)
            accepts_new = 1 if pdata.get('ACCEPTS_NEW_PATIENTS') == 'TRUE' else 0

            score = ProviderScore(
                npi=npi,
                name=f"{pdata.get('PROVIDER_FIRST_NAME', '')} {pdata.get('PROVIDER_LAST_NAME', '')}",
                specialty=pdata.get('SPECIALTY', ''),
                region=pdata.get('KP_REGION', ''),
                total_encounters=pe['count'],
                total_claims=pc['count'],
                avg_cost=round(avg_cost, 2),
                denial_rate=round(denial_rate, 4),
            )

            score.quality_score = round(max(0, 100 - denial_rate * 200), 2)

            score.efficiency_score = round(50, 2)

            score.access_score = round(
                min(100, (panel_size / 100) * 50 + accepts_new * 30), 2)

            score.volume_score = round(min(100, pe['count'] * 2), 2)

            specialty_groups[pdata.get('SPECIALTY', 'Unknown')].append(
                (score, avg_cost, avg_los))
            self._scores[npi] = score

        for specialty, group in specialty_groups.items():
            if len(group) < 2:
                for score, _, _ in group:
                    score.efficiency_score = 50
                    score.composite_score = self._compute_composite(score)
                continue

            costs = [g[1] for g in group]
            cost_mean = np.mean(costs)
            cost_std = np.std(costs)

            scored = []
            for score, avg_cost, avg_los in group:
                z = _zscore(avg_cost, cost_mean, cost_std) if cost_std > 0 else 0
                score.efficiency_score = round(max(0, min(100, 50 - z * 20)), 2)
                score.composite_score = self._compute_composite(score)
                scored.append(score)

            scored.sort(key=lambda s: s.composite_score, reverse=True)
            for i, s in enumerate(scored):
                s.peer_rank = i + 1
                s.peer_total = len(scored)
                for threshold, grade in self.GRADE_THRESHOLDS:
                    if s.composite_score >= threshold:
                        s.grade = grade
                        break

        logger.info("ProviderScorecard: scored %d providers across %d specialties",
                     len(self._scores), len(specialty_groups))

    def _compute_composite(self, score: ProviderScore) -> float:
        return round(
            0.35 * score.quality_score +
            0.30 * score.efficiency_score +
            0.20 * score.access_score +
            0.15 * score.volume_score,
            2
        )

    def get_top_providers(self, n: int = 20, specialty: str = None) -> List[ProviderScore]:
        scores = list(self._scores.values())
        if specialty:
            scores = [s for s in scores if s.specialty.lower() == specialty.lower()]
        scores.sort(key=lambda s: s.composite_score, reverse=True)
        return scores[:n]

    def get_bottom_providers(self, n: int = 20) -> List[ProviderScore]:
        scores = sorted(self._scores.values(), key=lambda s: s.composite_score)
        return scores[:n]

    def get_specialty_summary(self) -> Dict[str, Dict]:
        specs = defaultdict(lambda: {'count': 0, 'avg_composite': 0, 'avg_quality': 0,
                                      'avg_efficiency': 0, 'avg_denial_rate': 0})
        for s in self._scores.values():
            sp = specs[s.specialty]
            sp['count'] += 1
            sp['avg_composite'] += s.composite_score
            sp['avg_quality'] += s.quality_score
            sp['avg_efficiency'] += s.efficiency_score
            sp['avg_denial_rate'] += s.denial_rate

        for spec, sp in specs.items():
            n = sp['count'] or 1
            sp['avg_composite'] = round(sp['avg_composite'] / n, 2)
            sp['avg_quality'] = round(sp['avg_quality'] / n, 2)
            sp['avg_efficiency'] = round(sp['avg_efficiency'] / n, 2)
            sp['avg_denial_rate'] = round(sp['avg_denial_rate'] / n, 4)

        return dict(specs)


class PopulationSegmenter:

    def __init__(self, db_path: str, n_segments: int = 5):
        self.db_path = db_path
        self.n_segments = n_segments
        self._centroids: Optional[np.ndarray] = None
        self._segments: Dict[int, Dict] = {}
        self._member_segments: Dict[str, int] = {}
        self._feature_names = ['age', 'chronic_conditions', 'total_cost',
                                'encounter_count', 'er_visits', 'rx_count',
                                'risk_score', 'inpatient_days']

    def train(self):
        conn = _connect(self.db_path)

        member_features = {}
        for row in conn.execute("""
            SELECT m.MEMBER_ID,
                   CAST((julianday('now')-julianday(m.DATE_OF_BIRTH))/365.25 AS INT) as age,
                   CAST(m.CHRONIC_CONDITIONS AS INT) as chronic,
                   CAST(m.RISK_SCORE AS REAL) as risk_score
            FROM members m
        """):
            member_features[row['MEMBER_ID']] = {
                'age': float(row['age'] or 50),
                'chronic_conditions': float(row['chronic'] or 0),
                'risk_score': float(row['risk_score'] or 0),
                'total_cost': 0.0,
                'encounter_count': 0,
                'er_visits': 0,
                'rx_count': 0,
                'inpatient_days': 0,
            }

        for row in conn.execute("""
            SELECT MEMBER_ID, SUM(PAID_AMOUNT) as total FROM claims
            WHERE CLAIM_STATUS = 'PAID' GROUP BY MEMBER_ID
        """):
            mid = row['MEMBER_ID']
            if mid in member_features:
                member_features[mid]['total_cost'] = float(row['total'] or 0)

        for row in conn.execute("""
            SELECT MEMBER_ID, COUNT(*) as cnt,
                   SUM(CASE WHEN VISIT_TYPE='EMERGENCY' THEN 1 ELSE 0 END) as er,
                   SUM(CASE WHEN VISIT_TYPE='INPATIENT' THEN CAST(LENGTH_OF_STAY AS INT) ELSE 0 END) as ip_days
            FROM encounters GROUP BY MEMBER_ID
        """):
            mid = row['MEMBER_ID']
            if mid in member_features:
                member_features[mid]['encounter_count'] = float(row['cnt'] or 0)
                member_features[mid]['er_visits'] = float(row['er'] or 0)
                member_features[mid]['inpatient_days'] = float(row['ip_days'] or 0)

        for row in conn.execute("""
            SELECT MEMBER_ID, COUNT(DISTINCT MEDICATION_CLASS) as cnt FROM prescriptions
            GROUP BY MEMBER_ID
        """):
            mid = row['MEMBER_ID']
            if mid in member_features:
                member_features[mid]['rx_count'] = float(row['cnt'] or 0)

        conn.close()

        member_ids = list(member_features.keys())
        X = np.array([[member_features[mid][f] for f in self._feature_names]
                       for mid in member_ids])

        self._means = np.mean(X, axis=0)
        self._stds = np.std(X, axis=0)
        self._stds[self._stds == 0] = 1.0
        X_norm = (X - self._means) / self._stds

        self._centroids = self._kmeans(X_norm, self.n_segments, max_iter=50)

        assignments = self._assign(X_norm, self._centroids)

        for seg_id in range(self.n_segments):
            mask = assignments == seg_id
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue

            seg_features = X[indices]
            seg_members = [member_ids[i] for i in indices]

            profile = {
                self._feature_names[j]: round(float(np.mean(seg_features[:, j])), 2)
                for j in range(len(self._feature_names))
            }
            profile['member_count'] = len(seg_members)
            profile['pct_of_population'] = round(100 * len(seg_members) / len(member_ids), 1)

            profile['name'] = self._name_segment(profile)

            self._segments[seg_id] = profile
            for mid in seg_members:
                self._member_segments[mid] = seg_id

        logger.info("PopulationSegmenter: %d members → %d segments",
                     len(member_ids), len(self._segments))

    def _kmeans(self, X: np.ndarray, k: int, max_iter: int = 50) -> np.ndarray:
        n = X.shape[0]
        rng = np.random.RandomState(42)
        centroids = [X[rng.randint(n)]]
        for _ in range(1, k):
            dists = np.min([np.sum((X - c) ** 2, axis=1) for c in centroids], axis=0)
            probs = dists / dists.sum()
            centroids.append(X[rng.choice(n, p=probs)])
        centroids = np.array(centroids)

        for _ in range(max_iter):
            assignments = self._assign(X, centroids)
            new_centroids = np.array([
                X[assignments == i].mean(axis=0) if np.sum(assignments == i) > 0
                else centroids[i]
                for i in range(k)
            ])
            if np.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids

        return centroids

    def _assign(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        dists = np.array([np.sum((X - c) ** 2, axis=1) for c in centroids])
        return np.argmin(dists, axis=0)

    def _name_segment(self, profile: Dict) -> str:
        cost = profile.get('total_cost', 0)
        chronic = profile.get('chronic_conditions', 0)
        er = profile.get('er_visits', 0)
        age = profile.get('age', 0)
        enc = profile.get('encounter_count', 0)

        if cost > 8000 and chronic > 3:
            return 'Complex Chronic - High Cost'
        if er > 2 and cost > 5000:
            return 'Frequent ER - High Utilizer'
        if chronic > 3 and cost < 3000:
            return 'Managed Chronic - Stable'
        if age > 60 and chronic > 2:
            return 'Senior Complex Care'
        if enc < 2 and cost < 2000:
            return 'Healthy Low Utilizer'
        if cost > 5000:
            return 'Rising Risk - Cost Trending'
        if chronic <= 1 and enc < 4:
            return 'Generally Healthy'
        return 'Moderate Utilizer'

    def get_segments(self) -> Dict[int, Dict]:
        return self._segments

    def get_member_segment(self, member_id: str) -> Optional[Dict]:
        seg_id = self._member_segments.get(member_id)
        if seg_id is not None:
            return {'segment_id': seg_id, **self._segments.get(seg_id, {})}
        return None


class CareGapAnalyzer:

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._gaps: List[Dict] = []
        self._patterns: Dict[str, Dict] = {}

    def train(self):
        conn = _connect(self.db_path)

        chronic_members = {}
        for row in conn.execute("""
            SELECT d.MEMBER_ID, d.HCC_CATEGORY, COUNT(DISTINCT d.ENCOUNTER_ID) as visit_count,
                   m.CHRONIC_CONDITIONS
            FROM diagnoses d
            JOIN members m ON d.MEMBER_ID = m.MEMBER_ID
            WHERE d.IS_CHRONIC IN ('TRUE', '1', 'Y', 'Yes')
            GROUP BY d.MEMBER_ID, d.HCC_CATEGORY
        """):
            mid = row['MEMBER_ID']
            hcc = row['HCC_CATEGORY']
            if hcc:
                if hcc not in self._patterns:
                    self._patterns[hcc] = {'visit_counts': [], 'members': 0}
                self._patterns[hcc]['visit_counts'].append(row['visit_count'])
                self._patterns[hcc]['members'] += 1

                if mid not in chronic_members:
                    chronic_members[mid] = {'conditions': [], 'chronic_count': int(row['CHRONIC_CONDITIONS'] or 0)}
                chronic_members[mid]['conditions'].append({
                    'hcc': hcc,
                    'visit_count': row['visit_count'],
                })

        for hcc, data in self._patterns.items():
            counts = sorted(data['visit_counts'])
            data['median_visits'] = _percentile(counts, 50)
            data['p25_visits'] = _percentile(counts, 25)
            data['mean_visits'] = np.mean(counts)

        recent_visits = defaultdict(int)
        for row in conn.execute("""
            SELECT MEMBER_ID, COUNT(*) as cnt FROM encounters
            WHERE SERVICE_DATE >= date('now', '-12 months')
            GROUP BY MEMBER_ID
        """):
            recent_visits[row['MEMBER_ID']] = row['cnt']

        rx_adherence = defaultdict(lambda: {'authorized': 0, 'used': 0})
        for row in conn.execute("""
            SELECT MEMBER_ID, REFILLS_AUTHORIZED, REFILLS_USED
            FROM prescriptions
            WHERE STATUS = 'ACTIVE' OR STATUS = 'FILLED'
        """):
            mid = row['MEMBER_ID']
            rx_adherence[mid]['authorized'] += int(row['REFILLS_AUTHORIZED'] or 0)
            rx_adherence[mid]['used'] += int(row['REFILLS_USED'] or 0)

        missed_appts = defaultdict(int)
        for row in conn.execute("""
            SELECT MEMBER_ID, COUNT(*) as cnt FROM appointments
            WHERE STATUS IN ('CANCELLED', 'NO_SHOW', 'MISSED')
            GROUP BY MEMBER_ID
        """):
            missed_appts[row['MEMBER_ID']] = row['cnt']

        conn.close()

        for mid, mdata in chronic_members.items():
            gap_reasons = []
            worst_severity = 'Low'

            rx = rx_adherence.get(mid)
            rx_adh = None
            if rx and rx['authorized'] > 0:
                rx_adh = round(rx['used'] / rx['authorized'], 2)
                if rx_adh < 0.5:
                    gap_reasons.append('Low medication adherence (%.0f%%)' % (rx_adh * 100))
                    worst_severity = 'High'
                elif rx_adh < 0.75:
                    gap_reasons.append('Suboptimal medication adherence (%.0f%%)' % (rx_adh * 100))
                    if worst_severity == 'Low':
                        worst_severity = 'Medium'

            missed = missed_appts.get(mid, 0)
            if missed >= 3:
                gap_reasons.append('%d missed/cancelled appointments' % missed)
                worst_severity = 'High'
            elif missed >= 1:
                gap_reasons.append('%d missed/cancelled appointment(s)' % missed)
                if worst_severity == 'Low':
                    worst_severity = 'Medium'

            rv = recent_visits.get(mid, 0)
            if rv == 0 and mdata['chronic_count'] > 0:
                gap_reasons.append('No visits in past 12 months despite %d chronic conditions' % mdata['chronic_count'])
                worst_severity = 'High'

            for cond in mdata['conditions']:
                hcc = cond['hcc']
                pattern = self._patterns.get(hcc)
                if pattern and pattern['median_visits'] > 1 and cond['visit_count'] < pattern['p25_visits']:
                    gap_reasons.append('Below expected visits for %s (%d vs %.0f expected)' % (
                        hcc, cond['visit_count'], pattern['median_visits']))
                    if worst_severity != 'High':
                        worst_severity = 'Medium'

            if gap_reasons:
                primary_condition = mdata['conditions'][0]['hcc'] if mdata['conditions'] else 'Unknown'
                gap = {
                    'member_id': mid,
                    'condition': primary_condition,
                    'conditions': [c['hcc'] for c in mdata['conditions']],
                    'actual_visits': sum(c['visit_count'] for c in mdata['conditions']),
                    'expected_visits': round(sum(
                        self._patterns.get(c['hcc'], {}).get('median_visits', 1)
                        for c in mdata['conditions']
                    ), 1),
                    'gap_severity': worst_severity,
                    'gap_reasons': gap_reasons,
                    'recent_visits': rv,
                    'missed_appointments': missed,
                    'chronic_count': mdata['chronic_count'],
                }
                if rx_adh is not None:
                    gap['rx_adherence'] = rx_adh

                self._gaps.append(gap)

        severity_order = {'High': 0, 'Medium': 1, 'Low': 2}
        self._gaps.sort(key=lambda g: (severity_order.get(g['gap_severity'], 2),
                                        -g['chronic_count']))

        logger.info("CareGapAnalyzer: found %d care gaps across %d conditions, %d members",
                     len(self._gaps), len(self._patterns),
                     len(set(g['member_id'] for g in self._gaps)))

    def get_gaps(self, n: int = 50, severity: str = None) -> List[Dict]:
        if severity:
            return [g for g in self._gaps if g['gap_severity'] == severity][:n]
        return self._gaps[:n]

    def get_gap_summary(self) -> Dict[str, Any]:
        by_condition = defaultdict(lambda: {'count': 0, 'high_severity': 0})
        for g in self._gaps:
            bc = by_condition[g['condition']]
            bc['count'] += 1
            if g['gap_severity'] == 'High':
                bc['high_severity'] += 1

        return {
            'total_gaps': len(self._gaps),
            'unique_members': len(set(g['member_id'] for g in self._gaps)),
            'by_condition': dict(by_condition),
            'high_severity_count': sum(1 for g in self._gaps if g['gap_severity'] == 'High'),
        }

    def get_care_patterns(self) -> Dict[str, Dict]:
        return {hcc: {
            'median_visits': round(data['median_visits'], 1),
            'mean_visits': round(data['mean_visits'], 1),
            'members': data['members'],
        } for hcc, data in self._patterns.items()}


class HealthcareAnalyticsEngine:

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.risk = MemberRiskStratifier(db_path)
        self.readmission = ReadmissionPredictor(db_path)
        self.anomaly = CostAnomalyDetector(db_path)
        self.provider = ProviderScorecard(db_path)
        self.population = PopulationSegmenter(db_path)
        self.care_gaps = CareGapAnalyzer(db_path)
        self._trained = False

    def train_all(self):
        import time
        t0 = time.time()

        self.risk.train()
        self.readmission.train()
        self.anomaly.train()
        self.provider.train()
        self.population.train()
        self.care_gaps.train()

        self._trained = True
        elapsed = time.time() - t0
        logger.info("HealthcareAnalyticsEngine: all models trained in %.1fs", elapsed)

    def query(self, question: str) -> Dict[str, Any]:
        q = question.lower().strip()

        if any(kw in q for kw in ['readmission', 'readmit', '30 day', '30-day']):
            if 'metric' in q or 'performance' in q or 'accuracy' in q:
                return {'model': 'readmission_predictor', 'type': 'model_metrics',
                        'data': self.readmission.get_metrics()}
            return {'model': 'readmission_predictor', 'type': 'model_metrics',
                    'data': self.readmission.get_metrics()}

        if any(kw in q for kw in ['risk', 'high risk', 'rising risk', 'critical',
                                    'risk stratif', 'risk score', 'risk tier']):
            if 'region' in q:
                return {'model': 'risk_stratification', 'type': 'region_risk',
                        'data': self.risk.get_region_risk()}
            if 'summary' in q or 'tier' in q or 'breakdown' in q:
                return {'model': 'risk_stratification', 'type': 'tier_summary',
                        'data': self.risk.get_tier_summary()}
            n = self._extract_number(q, default=20)
            return {'model': 'risk_stratification', 'type': 'high_risk_members',
                    'data': [self._profile_to_dict(p) for p in self.risk.get_high_risk(n)]}

        if any(kw in q for kw in ['anomal', 'outlier', 'unusual cost', 'suspicious',
                                    'cost anomal', 'fraud', 'waste']):
            if 'summary' in q:
                return {'model': 'cost_anomaly', 'type': 'summary',
                        'data': self.anomaly.get_anomaly_summary()}
            n = self._extract_number(q, default=20)
            dim = None
            for d in ['provider', 'region', 'cpt', 'plan']:
                if d in q:
                    dim = d
                    if d == 'cpt':
                        dim = 'cpt_code'
                    elif d == 'plan':
                        dim = 'plan_type'
                    break
            return {'model': 'cost_anomaly', 'type': 'top_anomalies',
                    'data': self.anomaly.get_top_anomalies(n, dim)}

        if any(kw in q for kw in ['provider', 'scorecard', 'doctor', 'physician',
                                    'provider performance', 'provider quality']):
            if 'bottom' in q or 'worst' in q or 'low perform' in q:
                n = self._extract_number(q, default=20)
                return {'model': 'provider_scorecard', 'type': 'bottom_providers',
                        'data': [self._provider_to_dict(s) for s in self.provider.get_bottom_providers(n)]}
            if 'specialty' in q and 'summary' in q:
                return {'model': 'provider_scorecard', 'type': 'specialty_summary',
                        'data': self.provider.get_specialty_summary()}
            specialty = self._extract_specialty(q)
            n = self._extract_number(q, default=20)
            return {'model': 'provider_scorecard', 'type': 'top_providers',
                    'data': [self._provider_to_dict(s)
                             for s in self.provider.get_top_providers(n, specialty)]}

        if any(kw in q for kw in ['segment', 'cohort', 'population', 'cluster',
                                    'population health']):
            raw_segments = self.population.get_segments()
            segment_list = []
            for seg_id, profile in sorted(raw_segments.items()):
                segment_list.append({
                    'segment_id': seg_id,
                    'name': profile.get('name', f'Segment {seg_id}'),
                    'member_count': profile.get('member_count', 0),
                    'pct_of_population': profile.get('pct_of_population', 0),
                    'avg_age': profile.get('age', 0),
                    'avg_chronic_conditions': profile.get('chronic_conditions', 0),
                    'avg_total_cost': profile.get('total_cost', 0),
                    'avg_encounter_count': profile.get('encounter_count', 0),
                    'avg_er_visits': profile.get('er_visits', 0),
                    'avg_rx_count': profile.get('rx_count', 0),
                })
            return {'model': 'population_segments', 'type': 'segments',
                    'data': segment_list}

        if any(kw in q for kw in ['care gap', 'gap', 'missed', 'preventive',
                                    'follow up', 'adherence', 'non-compliance']):
            if 'summary' in q or 'overview' in q:
                return {'model': 'care_gaps', 'type': 'summary',
                        'data': self.care_gaps.get_gap_summary()}
            if 'pattern' in q:
                return {'model': 'care_gaps', 'type': 'care_patterns',
                        'data': self.care_gaps.get_care_patterns()}
            severity = None
            if 'high' in q:
                severity = 'High'
            n = self._extract_number(q, default=30)
            return {'model': 'care_gaps', 'type': 'gaps',
                    'data': self.care_gaps.get_gaps(n, severity)}

        return {
            'model': 'overview',
            'type': 'available_models',
            'data': {
                'risk_stratification': {
                    'description': 'Member risk scoring with multi-dimensional analysis',
                    'total_members': len(self.risk._profiles),
                    'tiers': self.risk.get_tier_summary(),
                },
                'readmission_predictor': {
                    'description': 'Logistic regression for 30-day readmission risk',
                    'metrics': self.readmission.get_metrics(),
                },
                'cost_anomaly_detector': {
                    'description': 'Statistical outlier detection across cost dimensions',
                    'total_anomalies': len(self.anomaly._anomalies),
                    'summary': self.anomaly.get_anomaly_summary(),
                },
                'provider_scorecard': {
                    'description': 'Composite quality/efficiency scoring by specialty',
                    'total_providers': len(self.provider._scores),
                },
                'population_segments': {
                    'description': 'K-means clustering into actionable cohorts',
                    'segments': {k: v['name'] for k, v in self.population.get_segments().items()},
                },
                'care_gaps': {
                    'description': 'Identifies members missing expected care',
                    'summary': self.care_gaps.get_gap_summary(),
                },
            }
        }

    def _extract_number(self, q: str, default: int = 20) -> int:
        m = re.search(r'\b(\d+)\b', q)
        return int(m.group(1)) if m else default

    def _extract_specialty(self, q: str) -> Optional[str]:
        specialties = ['cardiology', 'orthopedics', 'neurology', 'oncology',
                       'pediatrics', 'internal medicine', 'family medicine',
                       'emergency medicine', 'psychiatry', 'dermatology',
                       'radiology', 'surgery', 'urology', 'endocrinology']
        for s in specialties:
            if s in q.lower():
                return s.title()
        return None

    @staticmethod
    def _profile_to_dict(p: MemberRiskProfile) -> Dict:
        return {
            'member_id': p.member_id,
            'name': p.name,
            'risk_score': p.risk_score,
            'risk_tier': p.risk_tier,
            'clinical_score': p.clinical_score,
            'utilization_score': p.utilization_score,
            'cost_score': p.cost_score,
            'stability_score': p.stability_score,
            'top_factors': p.top_factors,
            'chronic_count': p.chronic_count,
            'total_claims': p.total_claims,
            'total_cost': p.total_cost,
            'er_visits': p.er_visits,
            'inpatient_days': p.inpatient_days,
            'region': p.region,
            'plan_type': p.plan_type,
        }

    @staticmethod
    def _provider_to_dict(s: ProviderScore) -> Dict:
        return {
            'npi': s.npi,
            'name': s.name,
            'specialty': s.specialty,
            'region': s.region,
            'composite_score': s.composite_score,
            'grade': s.grade,
            'quality_score': s.quality_score,
            'efficiency_score': s.efficiency_score,
            'access_score': s.access_score,
            'denial_rate': s.denial_rate,
            'total_encounters': s.total_encounters,
            'avg_cost': s.avg_cost,
            'peer_rank': s.peer_rank,
            'peer_total': s.peer_total,
        }
