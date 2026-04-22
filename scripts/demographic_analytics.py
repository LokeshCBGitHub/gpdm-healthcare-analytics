import sqlite3
import logging
import json
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter, defaultdict
from datetime import datetime
import math

logger = logging.getLogger(__name__)


class DemographicAnalytics:

    AGE_TIERS = {
        'Pediatric': (0, 18),
        'Young Adult': (18, 35),
        'Adult': (35, 55),
        'Middle Age': (55, 65),
        'Senior': (65, 150)
    }

    RISK_TIERS = {
        'Low': (0.0, 1.0),
        'Moderate': (1.0, 2.0),
        'High': (2.0, 3.0),
        'Very High': (3.0, 10.0)
    }

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self._pre_computed_metrics = {}
        self._initialize()

    def _initialize(self):
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            logger.info(f"Connected to database: {self.db_path}")

            self._precompute_metrics()
            logger.info("Pre-computed demographics metrics cached")
        except Exception as e:
            logger.error(f"Failed to initialize demographics engine: {e}")
            raise

    def _query(self, sql: str, params: tuple = ()) -> List[Dict]:
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql, params)
            return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Query failed: {e}\nSQL: {sql}")
            return []

    def _query_scalar(self, sql: str, params: tuple = ()) -> Optional[Any]:
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql, params)
            result = cursor.fetchone()
            return result[0] if result else None
        except Exception as e:
            logger.error(f"Scalar query failed: {e}")
            return None

    def _calculate_age_from_dob(self, dob: str) -> Optional[int]:
        try:
            if not dob:
                return None
            birth_date = datetime.strptime(dob, '%Y-%m-%d')
            today = datetime.now()
            age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
            return max(0, age)
        except:
            return None

    def _get_age_tier(self, age: Optional[int]) -> str:
        if age is None:
            return 'Unknown'
        for tier_name, (min_age, max_age) in self.AGE_TIERS.items():
            if min_age <= age < max_age:
                return tier_name
        return 'Unknown'

    def _get_risk_tier(self, risk_score: Optional[float]) -> str:
        if risk_score is None:
            return 'Unknown'
        try:
            score = float(risk_score)
            for tier_name, (min_risk, max_risk) in self.RISK_TIERS.items():
                if min_risk <= score < max_risk:
                    return tier_name
        except (ValueError, TypeError):
            pass
        return 'Unknown'

    def _safe_float(self, val: Any) -> Optional[float]:
        try:
            return float(val) if val else None
        except (ValueError, TypeError):
            return None

    def _parse_chronic_conditions(self, conditions_str: str) -> int:
        try:
            return int(conditions_str) if conditions_str else 0
        except (ValueError, TypeError):
            return 0

    def _precompute_metrics(self):
        logger.info("Pre-computing demographic metrics...")

        self._precompute_member_metrics()

        self._precompute_provider_metrics()

        self._precompute_claim_metrics()

    def _precompute_member_metrics(self):
        members = self._query(
            "SELECT MEMBER_ID, DATE_OF_BIRTH, GENDER, RACE, LANGUAGE, RISK_SCORE, CHRONIC_CONDITIONS, KP_REGION FROM members"
        )

        age_tiers = defaultdict(int)
        risk_tiers = defaultdict(int)
        gender_dist = defaultdict(int)
        race_dist = defaultdict(int)
        language_dist = defaultdict(int)
        region_dist = defaultdict(int)

        by_gender = defaultdict(lambda: {'costs': [], 'risks': [], 'burdens': [], 'encounters': []})
        by_race = defaultdict(lambda: {'costs': [], 'risks': [], 'burdens': [], 'encounters': []})
        by_language = defaultdict(lambda: {'costs': [], 'risks': [], 'burdens': [], 'encounters': []})
        by_region = defaultdict(lambda: {'costs': [], 'risks': [], 'burdens': [], 'encounters': []})

        for member in members:
            member_id = member['MEMBER_ID']
            age = self._calculate_age_from_dob(member['DATE_OF_BIRTH'])
            gender = member['GENDER'] or 'Unknown'
            race = member['RACE'] or 'Unknown'
            language = member['LANGUAGE'] or 'Unknown'
            risk_score = self._safe_float(member['RISK_SCORE'])
            chronic_burden = self._parse_chronic_conditions(member['CHRONIC_CONDITIONS'])
            region = member['KP_REGION'] or 'Unknown'

            age_tier = self._get_age_tier(age)
            age_tiers[age_tier] += 1

            risk_tier = self._get_risk_tier(risk_score)
            risk_tiers[risk_tier] += 1

            gender_dist[gender] += 1
            race_dist[race] += 1
            language_dist[language] += 1
            region_dist[region] += 1

            costs = self._query(
                "SELECT SUM(CAST(PAID_AMOUNT AS FLOAT)) as total FROM claims WHERE MEMBER_ID = ?",
                (member_id,)
            )
            cost = self._safe_float(costs[0]['total']) if costs and costs[0]['total'] else 0.0

            encounters = self._query_scalar(
                "SELECT COUNT(*) FROM encounters WHERE MEMBER_ID = ?",
                (member_id,)
            ) or 0

            if risk_score:
                by_gender[gender]['risks'].append(risk_score)
                by_race[race]['risks'].append(risk_score)
                by_language[language]['risks'].append(risk_score)
                by_region[region]['risks'].append(risk_score)

            by_gender[gender]['costs'].append(cost)
            by_gender[gender]['burdens'].append(chronic_burden)
            by_gender[gender]['encounters'].append(encounters)

            by_race[race]['costs'].append(cost)
            by_race[race]['burdens'].append(chronic_burden)
            by_race[race]['encounters'].append(encounters)

            by_language[language]['costs'].append(cost)
            by_language[language]['burdens'].append(chronic_burden)
            by_language[language]['encounters'].append(encounters)

            by_region[region]['costs'].append(cost)
            by_region[region]['burdens'].append(chronic_burden)
            by_region[region]['encounters'].append(encounters)

        self._pre_computed_metrics['age_tiers'] = dict(age_tiers)
        self._pre_computed_metrics['risk_tiers'] = dict(risk_tiers)
        self._pre_computed_metrics['gender_dist'] = dict(gender_dist)
        self._pre_computed_metrics['race_dist'] = dict(race_dist)
        self._pre_computed_metrics['language_dist'] = dict(language_dist)
        self._pre_computed_metrics['region_dist'] = dict(region_dist)
        self._pre_computed_metrics['by_gender'] = dict(by_gender)
        self._pre_computed_metrics['by_race'] = dict(by_race)
        self._pre_computed_metrics['by_language'] = dict(by_language)
        self._pre_computed_metrics['by_region'] = dict(by_region)

    def _precompute_provider_metrics(self):
        providers = self._query(
            "SELECT NPI, SPECIALTY, PANEL_SIZE, KP_REGION, PROVIDER_TYPE FROM providers"
        )

        specialty_dist = defaultdict(int)
        region_dist = defaultdict(int)
        provider_type_dist = defaultdict(int)
        panel_sizes = []
        by_specialty = defaultdict(list)
        by_region = defaultdict(list)

        for provider in providers:
            specialty = provider['SPECIALTY'] or 'Unknown'
            region = provider['KP_REGION'] or 'Unknown'
            ptype = provider['PROVIDER_TYPE'] or 'Unknown'
            panel_size = self._safe_float(provider['PANEL_SIZE']) or 0

            specialty_dist[specialty] += 1
            region_dist[region] += 1
            provider_type_dist[ptype] += 1
            panel_sizes.append(panel_size)

            by_specialty[specialty].append(panel_size)
            by_region[region].append(panel_size)

        self._pre_computed_metrics['provider_specialty_dist'] = dict(specialty_dist)
        self._pre_computed_metrics['provider_region_dist'] = dict(region_dist)
        self._pre_computed_metrics['provider_type_dist'] = dict(provider_type_dist)
        self._pre_computed_metrics['provider_panel_sizes'] = panel_sizes
        self._pre_computed_metrics['provider_by_specialty'] = dict(by_specialty)
        self._pre_computed_metrics['provider_by_region'] = dict(by_region)

    def _precompute_claim_metrics(self):
        claims = self._query(
            "SELECT MEMBER_ID, CAST(PAID_AMOUNT AS FLOAT) as paid, CLAIM_STATUS FROM claims WHERE MEMBER_ID IS NOT NULL"
        )

        denials_by_member = defaultdict(int)
        total_claims_by_member = defaultdict(int)

        for claim in claims:
            member_id = claim['MEMBER_ID']
            status = claim['CLAIM_STATUS'] or 'Unknown'
            total_claims_by_member[member_id] += 1
            if 'DENY' in str(status).upper() or status == 'DENIED':
                denials_by_member[member_id] += 1

        self._pre_computed_metrics['denials_by_member'] = dict(denials_by_member)
        self._pre_computed_metrics['total_claims_by_member'] = dict(total_claims_by_member)

    def _calculate_cohort_stats(self, values: List[float]) -> Dict[str, float]:
        if not values:
            return {'mean': 0, 'median': 0, 'p25': 0, 'p75': 0, 'std': 0, 'count': 0}

        sorted_vals = sorted(values)
        count = len(sorted_vals)
        mean = sum(sorted_vals) / count if count > 0 else 0

        mid = count // 2
        median = (sorted_vals[mid] + sorted_vals[mid - 1]) / 2 if count % 2 == 0 else sorted_vals[mid]

        p25_idx = max(0, int(count * 0.25))
        p75_idx = min(count - 1, int(count * 0.75))

        if count > 1:
            variance = sum((x - mean) ** 2 for x in sorted_vals) / (count - 1)
            std = math.sqrt(variance)
        else:
            std = 0

        return {
            'mean': round(mean, 2),
            'median': round(median, 2),
            'p25': round(sorted_vals[p25_idx], 2),
            'p75': round(sorted_vals[p75_idx], 2),
            'std': round(std, 2),
            'count': count
        }

    def _calculate_cramers_v(self, contingency_table: Dict[str, Dict[str, int]]) -> float:
        total = sum(sum(row.values()) for row in contingency_table.values())
        if total == 0:
            return 0.0

        rows = len(contingency_table)
        cols = len(next(iter(contingency_table.values()))) if contingency_table else 0

        if rows < 2 or cols < 2 or total < 2:
            return 0.0

        chi2 = 0.0
        row_totals = {k: sum(v.values()) for k, v in contingency_table.items()}
        col_totals = defaultdict(int)
        for row in contingency_table.values():
            for col, count in row.items():
                col_totals[col] += count

        for row_key, row_data in contingency_table.items():
            for col_key, observed in row_data.items():
                expected = (row_totals[row_key] * col_totals[col_key]) / total
                if expected > 0:
                    chi2 += ((observed - expected) ** 2) / expected

        min_dim = min(rows, cols) - 1
        if min_dim <= 0:
            return 0.0

        cramers_v = math.sqrt(chi2 / (total * min_dim))
        return round(cramers_v, 4)

    def _detect_statistical_disparity(self, group_data: Dict[str, List[float]]) -> Dict[str, Any]:
        if len(group_data) < 2:
            return {'status': 'insufficient_groups', 'disparities': []}

        groups = list(group_data.items())
        disparities = []

        for i, (group1_name, group1_values) in enumerate(groups):
            if not group1_values:
                continue

            mean1 = sum(group1_values) / len(group1_values)

            for group2_name, group2_values in groups[i+1:]:
                if not group2_values:
                    continue

                mean2 = sum(group2_values) / len(group2_values)

                n1, n2 = len(group1_values), len(group2_values)
                if n1 > 1 and n2 > 1:
                    var1 = sum((x - mean1) ** 2 for x in group1_values) / (n1 - 1)
                    var2 = sum((x - mean2) ** 2 for x in group2_values) / (n2 - 1)
                    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

                    if pooled_std > 0:
                        cohens_d = abs(mean1 - mean2) / pooled_std

                        if cohens_d > 0.2:
                            disparities.append({
                                'group1': group1_name,
                                'group2': group2_name,
                                'mean1': round(mean1, 2),
                                'mean2': round(mean2, 2),
                                'cohens_d': round(cohens_d, 4),
                                'significant': cohens_d > 0.5
                            })

        return {
            'status': 'detected',
            'disparity_count': len(disparities),
            'disparities': disparities
        }

    def analyze_member_demographics(self) -> Dict[str, Any]:
        result = {
            'timestamp': datetime.now().isoformat(),
            'entity_type': 'members',
            'age_stratification': {
                'tiers': self._pre_computed_metrics.get('age_tiers', {}),
                'total': sum(self._pre_computed_metrics.get('age_tiers', {}).values())
            },
            'risk_stratification': {
                'tiers': self._pre_computed_metrics.get('risk_tiers', {}),
                'total': sum(self._pre_computed_metrics.get('risk_tiers', {}).values())
            },
            'gender_distribution': {
                'distribution': self._pre_computed_metrics.get('gender_dist', {}),
                'total': sum(self._pre_computed_metrics.get('gender_dist', {}).values())
            },
            'race_distribution': {
                'distribution': self._pre_computed_metrics.get('race_dist', {}),
                'total': sum(self._pre_computed_metrics.get('race_dist', {}).values())
            },
            'language_distribution': {
                'distribution': self._pre_computed_metrics.get('language_dist', {}),
                'total': sum(self._pre_computed_metrics.get('language_dist', {}).values())
            },
            'region_distribution': {
                'distribution': self._pre_computed_metrics.get('region_dist', {}),
                'total': sum(self._pre_computed_metrics.get('region_dist', {}).values())
            }
        }

        equity_metrics = {}

        for demographic_group in ['gender', 'race', 'language', 'region']:
            group_key = f'by_{demographic_group}'
            group_data = self._pre_computed_metrics.get(group_key, {})

            group_equity = {}
            for group_name, metrics in group_data.items():
                cost_stats = self._calculate_cohort_stats(metrics['costs'])
                risk_stats = self._calculate_cohort_stats(metrics['risks'])
                burden_stats = self._calculate_cohort_stats(metrics['burdens'])
                encounter_stats = self._calculate_cohort_stats(metrics['encounters'])

                group_equity[group_name] = {
                    'avg_cost': cost_stats['mean'],
                    'cost_std': cost_stats['std'],
                    'avg_risk_score': risk_stats['mean'],
                    'avg_chronic_burden': burden_stats['mean'],
                    'avg_encounters': encounter_stats['mean'],
                    'member_count': cost_stats['count']
                }

            equity_metrics[demographic_group] = group_equity

        result['equity_metrics'] = equity_metrics

        disparities_summary = {}
        for demographic_group in ['gender', 'race', 'language', 'region']:
            group_key = f'by_{demographic_group}'
            group_data = self._pre_computed_metrics.get(group_key, {})

            cost_by_group = {k: v['costs'] for k, v in group_data.items()}
            cost_disparities = self._detect_statistical_disparity(cost_by_group)

            disparities_summary[demographic_group] = {
                'cost_disparities': cost_disparities,
                'disparity_flag': cost_disparities['disparity_count'] > 0
            }

        result['disparity_detection'] = disparities_summary

        return result

    def analyze_provider_demographics(self) -> Dict[str, Any]:
        result = {
            'timestamp': datetime.now().isoformat(),
            'entity_type': 'providers',
            'specialty_distribution': {
                'distribution': self._pre_computed_metrics.get('provider_specialty_dist', {}),
                'total': sum(self._pre_computed_metrics.get('provider_specialty_dist', {}).values())
            },
            'region_distribution': {
                'distribution': self._pre_computed_metrics.get('provider_region_dist', {}),
                'total': sum(self._pre_computed_metrics.get('provider_region_dist', {}).values())
            },
            'provider_type_distribution': {
                'distribution': self._pre_computed_metrics.get('provider_type_dist', {}),
                'total': sum(self._pre_computed_metrics.get('provider_type_dist', {}).values())
            }
        }

        panel_sizes = self._pre_computed_metrics.get('provider_panel_sizes', [])
        panel_stats = self._calculate_cohort_stats(panel_sizes)
        result['panel_size_analysis'] = {
            'statistics': panel_stats,
            'percentiles': {
                'p10': sorted(panel_sizes)[int(len(panel_sizes) * 0.1)] if panel_sizes else 0,
                'p50': panel_stats['median'],
                'p90': sorted(panel_sizes)[int(len(panel_sizes) * 0.9)] if panel_sizes else 0
            }
        }

        members_by_region = self._pre_computed_metrics.get('region_dist', {})
        providers_by_region = self._pre_computed_metrics.get('provider_region_dist', {})

        access_metrics = {}
        for region in set(list(members_by_region.keys()) + list(providers_by_region.keys())):
            member_count = members_by_region.get(region, 0)
            provider_count = providers_by_region.get(region, 1)
            ratio = member_count / provider_count if provider_count > 0 else 0

            access_metrics[region] = {
                'members': member_count,
                'providers': provider_count,
                'members_per_provider': round(ratio, 1)
            }

        result['regional_access_metrics'] = access_metrics

        panel_by_specialty = self._pre_computed_metrics.get('provider_by_specialty', {})
        specialty_panel_stats = {}
        for specialty, panels in panel_by_specialty.items():
            specialty_panel_stats[specialty] = self._calculate_cohort_stats(panels)

        result['panel_size_by_specialty'] = specialty_panel_stats

        return result

    def detect_disparities(self, metric: str, group_by: str) -> Dict[str, Any]:
        group_key = f'by_{group_by}'
        group_data = self._pre_computed_metrics.get(group_key, {})

        if not group_data:
            return {'error': f'Unknown grouping dimension: {group_by}', 'status': 'error'}

        metric_key_map = {
            'cost': 'costs',
            'risk': 'risks',
            'burden': 'burdens',
            'encounters': 'encounters'
        }

        if metric not in metric_key_map:
            return {'error': f'Unknown metric: {metric}', 'status': 'error'}

        metric_values_key = metric_key_map[metric]

        values_by_group = {
            group_name: metrics[metric_values_key]
            for group_name, metrics in group_data.items()
            if metrics[metric_values_key]
        }

        disparity_analysis = self._detect_statistical_disparity(values_by_group)

        group_summary = {}
        for group_name, values in values_by_group.items():
            stats = self._calculate_cohort_stats(values)
            group_summary[group_name] = stats

        return {
            'metric': metric,
            'group_by': group_by,
            'timestamp': datetime.now().isoformat(),
            'group_summary': group_summary,
            'disparity_analysis': disparity_analysis
        }

    def get_demographic_insight(self, question: str) -> Dict[str, Any]:
        question_lower = question.lower()

        result = {
            'question': question,
            'timestamp': datetime.now().isoformat()
        }

        if 'disparit' in question_lower or 'inequit' in question_lower:
            if 'cost' in question_lower:
                result.update(self.detect_disparities('cost', 'race'))
            elif 'risk' in question_lower:
                result.update(self.detect_disparities('risk', 'race'))
            else:
                result.update(self.detect_disparities('cost', 'race'))

        elif 'age' in question_lower:
            member_demo = self.analyze_member_demographics()
            result['analysis_type'] = 'age_stratification'
            result['data'] = member_demo['age_stratification']

        elif 'gender' in question_lower or 'sex' in question_lower:
            member_demo = self.analyze_member_demographics()
            result['analysis_type'] = 'gender_distribution'
            result['data'] = member_demo['gender_distribution']

        elif 'race' in question_lower or 'ethnicity' in question_lower:
            member_demo = self.analyze_member_demographics()
            result['analysis_type'] = 'race_distribution'
            result['data'] = member_demo['race_distribution']

        elif 'language' in question_lower:
            member_demo = self.analyze_member_demographics()
            result['analysis_type'] = 'language_distribution'
            result['data'] = member_demo['language_distribution']

        elif 'provider' in question_lower:
            provider_demo = self.analyze_provider_demographics()
            result['analysis_type'] = 'provider_demographics'
            result['data'] = provider_demo

        elif 'access' in question_lower:
            provider_demo = self.analyze_provider_demographics()
            result['analysis_type'] = 'provider_access'
            result['data'] = provider_demo.get('regional_access_metrics', {})

        elif 'burden' in question_lower or 'chronic' in question_lower:
            result.update(self.detect_disparities('burden', 'race'))
            result['analysis_type'] = 'chronic_burden'

        elif 'risk' in question_lower:
            member_demo = self.analyze_member_demographics()
            result['analysis_type'] = 'risk_stratification'
            result['data'] = member_demo['risk_stratification']

        elif 'region' in question_lower:
            member_demo = self.analyze_member_demographics()
            result['analysis_type'] = 'regional_distribution'
            result['data'] = member_demo['region_distribution']

        else:
            result['analysis_type'] = 'general_demographics'
            result['data'] = self.analyze_member_demographics()

        return result

    def get_health_equity_scorecard(self) -> Dict[str, Any]:
        member_demo = self.analyze_member_demographics()
        provider_demo = self.analyze_provider_demographics()

        scores = {}

        gender_equity = member_demo.get('equity_metrics', {}).get('gender', {})
        gender_cost_variance = 0
        if len(gender_equity) > 1:
            costs = [v['avg_cost'] for v in gender_equity.values()]
            if costs:
                mean_cost = sum(costs) / len(costs)
                variance = sum((c - mean_cost) ** 2 for c in costs) / len(costs)
                std = math.sqrt(variance)
                gender_cost_variance = (std / mean_cost * 100) if mean_cost > 0 else 0

        scores['gender_equity'] = max(0, 100 - gender_cost_variance)

        race_equity = member_demo.get('equity_metrics', {}).get('race', {})
        race_cost_variance = 0
        if len(race_equity) > 1:
            costs = [v['avg_cost'] for v in race_equity.values()]
            if costs:
                mean_cost = sum(costs) / len(costs)
                variance = sum((c - mean_cost) ** 2 for c in costs) / len(costs)
                std = math.sqrt(variance)
                race_cost_variance = (std / mean_cost * 100) if mean_cost > 0 else 0

        scores['race_equity'] = max(0, 100 - race_cost_variance)

        language_dist = member_demo['language_distribution']['distribution']
        language_equity = (100 - (language_dist.get('Unknown', 0) / sum(language_dist.values()) * 100)) if language_dist else 50
        scores['language_access_equity'] = round(language_equity, 1)

        regional_access = provider_demo.get('regional_access_metrics', {})
        member_providers = [v['members_per_provider'] for v in regional_access.values()]
        if member_providers and len(member_providers) > 1:
            mean_ratio = sum(member_providers) / len(member_providers)
            variance = sum((r - mean_ratio) ** 2 for r in member_providers) / len(member_providers)
            cv = (math.sqrt(variance) / mean_ratio * 100) if mean_ratio > 0 else 0
            scores['provider_access_equity'] = max(0, 100 - cv)
        else:
            scores['provider_access_equity'] = 100

        overall_score = sum(scores.values()) / len(scores) if scores else 0

        return {
            'timestamp': datetime.now().isoformat(),
            'entity': 'health_system',
            'scorecard': {
                'gender_equity': round(scores['gender_equity'], 1),
                'race_equity': round(scores['race_equity'], 1),
                'language_access_equity': round(scores['language_access_equity'], 1),
                'provider_access_equity': round(scores['provider_access_equity'], 1),
                'overall_equity_score': round(overall_score, 1)
            },
            'member_demographics': member_demo,
            'provider_demographics': provider_demo,
            'interpretation': {
                'overall': 'Excellent' if overall_score > 80 else 'Good' if overall_score > 60 else 'Needs Improvement',
                'key_gaps': [
                    dim for dim, score in scores.items() if score < 70
                ],
                'recommendations': self._get_equity_recommendations(scores)
            }
        }

    def _get_equity_recommendations(self, scores: Dict[str, float]) -> List[str]:
        recommendations = []

        if scores.get('gender_equity', 100) < 70:
            recommendations.append('Address gender-based cost disparities through care standardization programs')

        if scores.get('race_equity', 100) < 70:
            recommendations.append('Implement racial equity initiatives to reduce cost and outcome disparities')

        if scores.get('language_access_equity', 100) < 70:
            recommendations.append('Expand language access services and interpreter availability')

        if scores.get('provider_access_equity', 100) < 70:
            recommendations.append('Redistribute provider resources to underserved regions')

        if not recommendations:
            recommendations.append('Continue monitoring equity metrics and maintain current initiatives')

        return recommendations


    def analyze_geographic_demographics(self) -> Dict[str, Any]:
        try:
            conn = sqlite3.connect(self.db_path)
            result = {}

            rows = conn.execute("""
                SELECT m.KP_REGION,
                       COUNT(DISTINCT m.MEMBER_ID) as member_count,
                       ROUND(AVG(m.RISK_SCORE), 3) as avg_risk,
                       ROUND(AVG(m.CHRONIC_CONDITIONS), 2) as avg_chronic,
                       COUNT(DISTINCT CASE WHEN m.DISENROLLMENT_DATE IS NOT NULL THEN m.MEMBER_ID END) as disenrolled,
                       ROUND(100.0 * COUNT(DISTINCT CASE WHEN m.DISENROLLMENT_DATE IS NOT NULL THEN m.MEMBER_ID END) / COUNT(DISTINCT m.MEMBER_ID), 2) as disenroll_pct
                FROM MEMBERS m
                GROUP BY m.KP_REGION ORDER BY member_count DESC
            """).fetchall()
            result['regional_profile'] = [
                {'region': r[0], 'members': r[1], 'avg_risk': r[2], 'avg_chronic': r[3],
                 'disenrolled': r[4], 'disenroll_pct': r[5]} for r in rows
            ]

            rows = conn.execute("""
                SELECT STATE, COUNT(*) as count, ROUND(AVG(RISK_SCORE), 3) as avg_risk
                FROM MEMBERS GROUP BY STATE ORDER BY count DESC LIMIT 20
            """).fetchall()
            result['state_distribution'] = [{'state': r[0], 'count': r[1], 'avg_risk': r[2]} for r in rows]

            rows = conn.execute("""
                SELECT CITY, STATE, COUNT(*) as count, ROUND(AVG(RISK_SCORE), 3) as avg_risk,
                       ROUND(AVG(CHRONIC_CONDITIONS), 2) as avg_chronic
                FROM MEMBERS GROUP BY CITY, STATE ORDER BY count DESC LIMIT 20
            """).fetchall()
            result['city_distribution'] = [
                {'city': r[0], 'state': r[1], 'count': r[2], 'avg_risk': r[3], 'avg_chronic': r[4]} for r in rows
            ]

            rows = conn.execute("""
                SELECT m.KP_REGION,
                       ROUND(AVG(c.PAID_AMOUNT), 2) as avg_paid,
                       ROUND(AVG(c.BILLED_AMOUNT), 2) as avg_billed,
                       ROUND(AVG(c.COPAY), 2) as avg_copay,
                       COUNT(DISTINCT m.MEMBER_ID) as unique_members
                FROM CLAIMS c JOIN MEMBERS m ON c.MEMBER_ID = m.MEMBER_ID
                GROUP BY m.KP_REGION ORDER BY avg_paid DESC
            """).fetchall()
            result['regional_cost_profile'] = [
                {'region': r[0], 'avg_paid': r[1], 'avg_billed': r[2], 'avg_copay': r[3], 'unique_members': r[4]}
                for r in rows
            ]

            rows = conn.execute("""
                SELECT ZIP_CODE, COUNT(*) as count, ROUND(AVG(RISK_SCORE), 3) as avg_risk,
                       ROUND(AVG(CHRONIC_CONDITIONS), 2) as avg_chronic
                FROM MEMBERS GROUP BY ZIP_CODE ORDER BY count DESC LIMIT 15
            """).fetchall()
            result['zip_clusters'] = [
                {'zip': r[0], 'count': r[1], 'avg_risk': r[2], 'avg_chronic': r[3]} for r in rows
            ]

            rows = conn.execute("""
                SELECT m.KP_REGION,
                       COUNT(DISTINCT m.MEMBER_ID) as members,
                       COUNT(DISTINCT p.NPI) as providers,
                       ROUND(CAST(COUNT(DISTINCT m.MEMBER_ID) AS REAL) / NULLIF(COUNT(DISTINCT p.NPI), 0), 1) as members_per_provider
                FROM MEMBERS m LEFT JOIN PROVIDERS p ON m.KP_REGION = p.KP_REGION
                GROUP BY m.KP_REGION ORDER BY members_per_provider DESC
            """).fetchall()
            result['access_by_region'] = [
                {'region': r[0], 'members': r[1], 'providers': r[2], 'members_per_provider': r[3]} for r in rows
            ]

            conn.close()
            result['analysis_type'] = 'geographic_demographics'
            logger.info("Geographic demographics analysis complete: %d regions, %d states, %d cities",
                        len(result['regional_profile']), len(result['state_distribution']),
                        len(result['city_distribution']))
            return result
        except Exception as e:
            logger.error("Geographic demographics error: %s", e)
            return {'error': str(e), 'analysis_type': 'geographic_demographics'}

    def analyze_health_status_demographics(self) -> Dict[str, Any]:
        try:
            conn = sqlite3.connect(self.db_path)
            result = {}

            rows = conn.execute("""
                SELECT CHRONIC_CONDITIONS, COUNT(*) as count,
                       ROUND(AVG(RISK_SCORE), 3) as avg_risk,
                       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM MEMBERS), 2) as pct
                FROM MEMBERS GROUP BY CHRONIC_CONDITIONS ORDER BY CHRONIC_CONDITIONS
            """).fetchall()
            result['chronic_distribution'] = [
                {'conditions': r[0], 'count': r[1], 'avg_risk': r[2], 'pct': r[3]} for r in rows
            ]

            rows = conn.execute("""
                SELECT CASE
                    WHEN CHRONIC_CONDITIONS = 0 THEN 'No Chronic'
                    WHEN CHRONIC_CONDITIONS = 1 THEN '1 Condition'
                    WHEN CHRONIC_CONDITIONS = 2 THEN '2 Conditions'
                    WHEN CHRONIC_CONDITIONS BETWEEN 3 AND 4 THEN '3-4 Conditions'
                    ELSE '5+ Conditions'
                END as chronic_tier,
                COUNT(*) as count,
                ROUND(AVG(RISK_SCORE), 3) as avg_risk,
                MIN(RISK_SCORE) as min_risk,
                MAX(RISK_SCORE) as max_risk
                FROM MEMBERS GROUP BY chronic_tier ORDER BY avg_risk
            """).fetchall()
            result['chronic_risk_correlation'] = [
                {'tier': r[0], 'count': r[1], 'avg_risk': r[2], 'min_risk': r[3], 'max_risk': r[4]} for r in rows
            ]

            rows = conn.execute("""
                SELECT ICD10_DESCRIPTION, IS_CHRONIC, COUNT(*) as count,
                       COUNT(DISTINCT MEMBER_ID) as unique_members
                FROM DIAGNOSES GROUP BY ICD10_DESCRIPTION, IS_CHRONIC
                ORDER BY count DESC LIMIT 20
            """).fetchall()
            result['top_diagnoses'] = [
                {'diagnosis': r[0], 'is_chronic': r[1], 'count': r[2], 'unique_members': r[3]} for r in rows
            ]

            rows = conn.execute("""
                SELECT HCC_CATEGORY, COUNT(*) as count,
                       COUNT(DISTINCT MEMBER_ID) as unique_members,
                       ROUND(AVG(SEVERITY), 2) as avg_severity
                FROM DIAGNOSES WHERE HCC_CATEGORY IS NOT NULL
                GROUP BY HCC_CATEGORY ORDER BY count DESC LIMIT 15
            """).fetchall()
            result['hcc_distribution'] = [
                {'category': r[0], 'count': r[1], 'unique_members': r[2], 'avg_severity': r[3]} for r in rows
            ]

            rows = conn.execute("""
                SELECT m.RACE,
                       ROUND(AVG(m.CHRONIC_CONDITIONS), 2) as avg_chronic,
                       ROUND(AVG(m.RISK_SCORE), 3) as avg_risk,
                       COUNT(*) as count
                FROM MEMBERS m WHERE m.RACE IS NOT NULL
                GROUP BY m.RACE ORDER BY avg_chronic DESC
            """).fetchall()
            result['chronic_by_race'] = [
                {'race': r[0], 'avg_chronic': r[1], 'avg_risk': r[2], 'count': r[3]} for r in rows
            ]

            rows = conn.execute("""
                SELECT GENDER, ROUND(AVG(CHRONIC_CONDITIONS), 2) as avg_chronic,
                       ROUND(AVG(RISK_SCORE), 3) as avg_risk, COUNT(*) as count
                FROM MEMBERS GROUP BY GENDER ORDER BY avg_chronic DESC
            """).fetchall()
            result['chronic_by_gender'] = [
                {'gender': r[0], 'avg_chronic': r[1], 'avg_risk': r[2], 'count': r[3]} for r in rows
            ]

            rows = conn.execute("""
                SELECT SEVERITY, COUNT(*) as count,
                       COUNT(DISTINCT MEMBER_ID) as unique_members
                FROM DIAGNOSES GROUP BY SEVERITY ORDER BY SEVERITY
            """).fetchall()
            result['severity_distribution'] = [
                {'severity': r[0], 'count': r[1], 'unique_members': r[2]} for r in rows
            ]

            rows = conn.execute("""
                SELECT GENDER, RACE, KP_REGION, PLAN_TYPE,
                       COUNT(*) as count, ROUND(AVG(CHRONIC_CONDITIONS), 2) as avg_chronic
                FROM MEMBERS WHERE RISK_SCORE > 2.0
                GROUP BY GENDER, RACE, KP_REGION, PLAN_TYPE
                ORDER BY count DESC LIMIT 15
            """).fetchall()
            result['high_risk_profile'] = [
                {'gender': r[0], 'race': r[1], 'region': r[2], 'plan': r[3],
                 'count': r[4], 'avg_chronic': r[5]} for r in rows
            ]

            conn.close()
            result['analysis_type'] = 'health_status_demographics'
            logger.info("Health status demographics complete: %d chronic tiers, %d diagnoses, %d HCC categories",
                        len(result['chronic_distribution']), len(result['top_diagnoses']),
                        len(result['hcc_distribution']))
            return result
        except Exception as e:
            logger.error("Health status demographics error: %s", e)
            return {'error': str(e), 'analysis_type': 'health_status_demographics'}

    def analyze_socioeconomic_proxy(self) -> Dict[str, Any]:
        try:
            conn = sqlite3.connect(self.db_path)
            result = {}

            rows = conn.execute("""
                SELECT PLAN_TYPE, COUNT(*) as count,
                       ROUND(AVG(RISK_SCORE), 3) as avg_risk,
                       ROUND(AVG(CHRONIC_CONDITIONS), 2) as avg_chronic,
                       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM MEMBERS), 2) as pct
                FROM MEMBERS GROUP BY PLAN_TYPE ORDER BY count DESC
            """).fetchall()
            result['plan_type_profile'] = [
                {'plan': r[0], 'count': r[1], 'avg_risk': r[2], 'avg_chronic': r[3], 'pct': r[4]} for r in rows
            ]

            rows = conn.execute("""
                SELECT m.PLAN_TYPE,
                       ROUND(AVG(c.COPAY), 2) as avg_copay,
                       ROUND(AVG(c.DEDUCTIBLE), 2) as avg_deductible,
                       ROUND(AVG(c.MEMBER_RESPONSIBILITY), 2) as avg_member_resp,
                       ROUND(AVG(c.PAID_AMOUNT), 2) as avg_paid,
                       ROUND(100.0 * AVG(c.MEMBER_RESPONSIBILITY) / NULLIF(AVG(c.BILLED_AMOUNT), 0), 2) as burden_pct
                FROM CLAIMS c JOIN MEMBERS m ON c.MEMBER_ID = m.MEMBER_ID
                GROUP BY m.PLAN_TYPE ORDER BY burden_pct DESC
            """).fetchall()
            result['cost_burden_by_plan'] = [
                {'plan': r[0], 'avg_copay': r[1], 'avg_deductible': r[2], 'avg_member_resp': r[3],
                 'avg_paid': r[4], 'burden_pct': r[5]} for r in rows
            ]

            rows = conn.execute("""
                SELECT LANGUAGE, COUNT(*) as count,
                       ROUND(AVG(RISK_SCORE), 3) as avg_risk,
                       ROUND(AVG(CHRONIC_CONDITIONS), 2) as avg_chronic,
                       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM MEMBERS), 2) as pct
                FROM MEMBERS GROUP BY LANGUAGE ORDER BY count DESC
            """).fetchall()
            result['language_distribution'] = [
                {'language': r[0], 'count': r[1], 'avg_risk': r[2], 'avg_chronic': r[3], 'pct': r[4]} for r in rows
            ]

            rows = conn.execute("""
                SELECT PLAN_TYPE, RACE,
                       COUNT(*) as count,
                       ROUND(AVG(RISK_SCORE), 3) as avg_risk,
                       ROUND(AVG(CHRONIC_CONDITIONS), 2) as avg_chronic
                FROM MEMBERS WHERE RACE IS NOT NULL
                GROUP BY PLAN_TYPE, RACE
                HAVING count > 10
                ORDER BY avg_risk DESC LIMIT 20
            """).fetchall()
            result['plan_race_risk'] = [
                {'plan': r[0], 'race': r[1], 'count': r[2], 'avg_risk': r[3], 'avg_chronic': r[4]} for r in rows
            ]

            rows = conn.execute("""
                SELECT m.PLAN_TYPE,
                       ROUND(AVG(p.COST), 2) as avg_rx_cost,
                       ROUND(AVG(p.COPAY), 2) as avg_rx_copay,
                       ROUND(100.0 * AVG(p.COPAY) / NULLIF(AVG(p.COST), 0), 2) as rx_burden_pct,
                       COUNT(DISTINCT m.MEMBER_ID) as members_with_rx
                FROM PRESCRIPTIONS p JOIN MEMBERS m ON p.MEMBER_ID = m.MEMBER_ID
                GROUP BY m.PLAN_TYPE ORDER BY rx_burden_pct DESC
            """).fetchall()
            result['rx_cost_burden'] = [
                {'plan': r[0], 'avg_rx_cost': r[1], 'avg_rx_copay': r[2],
                 'rx_burden_pct': r[3], 'members_with_rx': r[4]} for r in rows
            ]

            result['ses_risk_indicators'] = {
                'high_burden_plans': [r['plan'] for r in result['cost_burden_by_plan'] if (r.get('burden_pct') or 0) > 20],
                'non_english_pct': sum(r['pct'] for r in result['language_distribution'] if r['language'] != 'English'),
                'analysis_note': 'These are proxy indicators derived from available data. '
                                 'Direct income/profession data not available in current dataset.',
            }

            conn.close()
            result['analysis_type'] = 'socioeconomic_proxy'
            logger.info("Socioeconomic proxy analysis complete")
            return result
        except Exception as e:
            logger.error("Socioeconomic proxy error: %s", e)
            return {'error': str(e), 'analysis_type': 'socioeconomic_proxy'}

    def analyze_utilization_demographics(self) -> Dict[str, Any]:
        try:
            conn = sqlite3.connect(self.db_path)
            result = {}

            rows = conn.execute("""
                SELECT m.GENDER, m.RACE,
                       COUNT(DISTINCT e.ENCOUNTER_ID) as encounters,
                       COUNT(DISTINCT m.MEMBER_ID) as members,
                       ROUND(CAST(COUNT(DISTINCT e.ENCOUNTER_ID) AS REAL) / NULLIF(COUNT(DISTINCT m.MEMBER_ID), 0), 2) as encounters_per_member
                FROM MEMBERS m LEFT JOIN ENCOUNTERS e ON m.MEMBER_ID = e.MEMBER_ID
                WHERE m.RACE IS NOT NULL
                GROUP BY m.GENDER, m.RACE
                ORDER BY encounters_per_member DESC
            """).fetchall()
            result['utilization_by_demo'] = [
                {'gender': r[0], 'race': r[1], 'encounters': r[2], 'members': r[3],
                 'encounters_per_member': r[4]} for r in rows
            ]

            rows = conn.execute("""
                SELECT m.RACE, e.VISIT_TYPE,
                       COUNT(*) as count,
                       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY m.RACE), 2) as pct_within_race
                FROM ENCOUNTERS e JOIN MEMBERS m ON e.MEMBER_ID = m.MEMBER_ID
                WHERE m.RACE IS NOT NULL
                GROUP BY m.RACE, e.VISIT_TYPE
                ORDER BY m.RACE, count DESC
            """).fetchall()
            result['visit_type_by_race'] = [
                {'race': r[0], 'visit_type': r[1], 'count': r[2], 'pct_within_race': r[3]} for r in rows
            ]

            rows = conn.execute("""
                SELECT m.RACE, p.MEDICATION_CLASS,
                       COUNT(*) as rx_count,
                       COUNT(DISTINCT m.MEMBER_ID) as unique_members,
                       ROUND(AVG(p.COST), 2) as avg_cost
                FROM PRESCRIPTIONS p JOIN MEMBERS m ON p.MEMBER_ID = m.MEMBER_ID
                WHERE m.RACE IS NOT NULL
                GROUP BY m.RACE, p.MEDICATION_CLASS
                HAVING rx_count > 5
                ORDER BY m.RACE, rx_count DESC
            """).fetchall()
            result['rx_by_race'] = [
                {'race': r[0], 'medication_class': r[1], 'rx_count': r[2],
                 'unique_members': r[3], 'avg_cost': r[4]} for r in rows
            ]

            rows = conn.execute("""
                SELECT m.RACE, r.SPECIALTY, r.URGENCY,
                       COUNT(*) as count
                FROM REFERRALS r JOIN MEMBERS m ON r.MEMBER_ID = m.MEMBER_ID
                WHERE m.RACE IS NOT NULL
                GROUP BY m.RACE, r.SPECIALTY, r.URGENCY
                ORDER BY m.RACE, count DESC
            """).fetchall()
            result['referral_patterns'] = [
                {'race': r[0], 'specialty': r[1], 'urgency': r[2], 'count': r[3]} for r in rows[:30]
            ]

            rows = conn.execute("""
                SELECT m.GENDER, m.RACE,
                       ROUND(AVG(e.LENGTH_OF_STAY), 2) as avg_los,
                       COUNT(*) as encounters
                FROM ENCOUNTERS e JOIN MEMBERS m ON e.MEMBER_ID = m.MEMBER_ID
                WHERE e.LENGTH_OF_STAY > 0 AND m.RACE IS NOT NULL
                GROUP BY m.GENDER, m.RACE
                HAVING encounters > 5
                ORDER BY avg_los DESC
            """).fetchall()
            result['los_by_demographics'] = [
                {'gender': r[0], 'race': r[1], 'avg_los': r[2], 'encounters': r[3]} for r in rows
            ]

            conn.close()
            result['analysis_type'] = 'utilization_demographics'
            logger.info("Utilization demographics complete: %d demo combinations, %d visit type patterns",
                        len(result['utilization_by_demo']), len(result['visit_type_by_race']))
            return result
        except Exception as e:
            logger.error("Utilization demographics error: %s", e)
            return {'error': str(e), 'analysis_type': 'utilization_demographics'}

    def get_comprehensive_demographic_report(self) -> Dict[str, Any]:
        report = {
            'report_type': 'comprehensive_demographics',
            'member_demographics': self.analyze_member_demographics(),
            'provider_demographics': self.analyze_provider_demographics(),
            'geographic_analysis': self.analyze_geographic_demographics(),
            'health_status': self.analyze_health_status_demographics(),
            'socioeconomic_proxy': self.analyze_socioeconomic_proxy(),
            'utilization_patterns': self.analyze_utilization_demographics(),
            'health_equity': self.get_health_equity_scorecard(),
        }

        summary_points = []
        mem = report.get('member_demographics', {})
        if mem.get('total_members'):
            summary_points.append(f"Total population: {mem['total_members']:,} members")
        geo = report.get('geographic_analysis', {})
        if geo.get('regional_profile'):
            top_region = geo['regional_profile'][0]
            summary_points.append(f"Largest region: {top_region['region']} ({top_region['members']:,} members)")
        health = report.get('health_status', {})
        if health.get('chronic_risk_correlation'):
            high_risk = [c for c in health['chronic_risk_correlation'] if c.get('avg_risk', 0) > 2.0]
            if high_risk:
                summary_points.append(f"High-risk chronic tiers: {len(high_risk)} groups with avg risk > 2.0")
        equity = report.get('health_equity', {})
        if equity.get('scorecard', {}).get('overall_score'):
            summary_points.append(f"Health equity score: {equity['scorecard']['overall_score']}/100")

        report['executive_summary'] = summary_points
        report['generated_at'] = datetime.now().isoformat()
        return report

    def close(self):
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


def main():
    import os

    db_path = '/sessions/great-gallant-allen/mnt/chatbot/mtp_demo/data/healthcare_production.db'

    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        return

    engine = DemographicAnalytics(db_path)

    print("\n=== MEMBER DEMOGRAPHICS ===")
    member_demo = engine.analyze_member_demographics()
    print(json.dumps({
        'age_stratification': member_demo['age_stratification'],
        'risk_stratification': member_demo['risk_stratification'],
        'gender_distribution': member_demo['gender_distribution']
    }, indent=2))

    print("\n=== PROVIDER DEMOGRAPHICS ===")
    provider_demo = engine.analyze_provider_demographics()
    print(json.dumps({
        'specialty_distribution': provider_demo['specialty_distribution'],
        'regional_access_metrics': provider_demo['regional_access_metrics']
    }, indent=2))

    print("\n=== DISPARITY DETECTION ===")
    disparities = engine.detect_disparities('cost', 'race')
    print(json.dumps(disparities, indent=2))

    print("\n=== HEALTH EQUITY SCORECARD ===")
    scorecard = engine.get_health_equity_scorecard()
    print(json.dumps({
        'scorecard': scorecard['scorecard'],
        'interpretation': scorecard['interpretation']
    }, indent=2))

    print("\n=== DEMOGRAPHIC INSIGHT ===")
    insight = engine.get_demographic_insight('Are there racial disparities in healthcare costs?')
    print(json.dumps(insight, indent=2, default=str))

    engine.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
