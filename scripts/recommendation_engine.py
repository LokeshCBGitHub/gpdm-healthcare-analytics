import math
import csv
import os
import json
import sqlite3
import time
import itertools
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Any


KPI_THRESHOLDS = {
    'high_value_claim': 5000,
    'denial_rate': 15,
    'processing_days': 30,
    'readmission_rate': 20,
    'er_outpatient_ratio': 0.5,
    'managed_care_pct': 70,
    'avg_los_days': 7,
}

CSV_BATCH_SIZE = 5000
MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.5
MIN_LIFT = 1.2


class KPIDefinition:

    def __init__(self, name: str, category: str, sql: str, description: str,
                 target: float = None, direction: str = 'higher_is_better',
                 unit: str = '', alert_threshold: float = None):
        self.name = name
        self.category = category
        self.sql = sql
        self.description = description
        self.target = target
        self.direction = direction
        self.unit = unit
        self.alert_threshold = alert_threshold


HEALTHCARE_KPIS = [
    KPIDefinition(
        name="Average Revenue Per Member (ARPM)",
        category="revenue",
        sql="SELECT AVG(PAID_AMOUNT) as arpm FROM claims",
        description="Average revenue generated per member across all claim types",
        direction="higher_is_better", unit="$",
    ),
    KPIDefinition(
        name="Total Revenue by Region",
        category="revenue",
        sql="SELECT KP_REGION, SUM(PAID_AMOUNT) as total_revenue FROM claims GROUP BY KP_REGION ORDER BY total_revenue DESC",
        description="Revenue breakdown by KP Healthcare region to identify high/low performing areas",
        direction="higher_is_better", unit="$",
    ),
    KPIDefinition(
        name="Revenue Per Encounter",
        category="revenue",
        sql="SELECT AVG(c.PAID_AMOUNT) as rev_per_encounter FROM claims c",
        description="Average revenue per healthcare encounter",
        direction="higher_is_better", unit="$",
    ),
    KPIDefinition(
        name="High-Value Claim Rate",
        category="revenue",
        sql=f"SELECT ROUND(100.0 * COUNT(CASE WHEN PAID_AMOUNT > {KPI_THRESHOLDS['high_value_claim']} THEN 1 END) / COUNT(*), 2) as high_value_pct FROM claims",
        description=f"Percentage of claims above ${KPI_THRESHOLDS['high_value_claim']} - indicates acuity mix",
        direction="monitor", unit="%",
    ),
    KPIDefinition(
        name="Claim Denial Rate",
        category="revenue",
        sql="SELECT ROUND(100.0 * COUNT(CASE WHEN CLAIM_STATUS = 'DENIED' THEN 1 END) / COUNT(*), 2) as denial_rate FROM claims",
        description="Percentage of denied claims - lower is better for revenue capture",
        direction="lower_is_better", unit="%", alert_threshold=KPI_THRESHOLDS['denial_rate'],
    ),
    KPIDefinition(
        name="Average Processing Time",
        category="revenue",
        sql="SELECT AVG(julianday(ADJUDICATED_DATE) - julianday(SUBMITTED_DATE)) as avg_process_days FROM claims WHERE ADJUDICATED_DATE IS NOT NULL AND SUBMITTED_DATE IS NOT NULL",
        description="Average days to process a claim - faster processing improves cash flow",
        direction="lower_is_better", unit="days", alert_threshold=KPI_THRESHOLDS['processing_days'],
    ),

    KPIDefinition(
        name="Readmission Rate",
        category="retention",
        sql=("SELECT ROUND(100.0 * COUNT(DISTINCT r.MEMBER_ID) / NULLIF(COUNT(DISTINCT e.MEMBER_ID), 0), 2) as readmit_rate "
             "FROM encounters e LEFT JOIN ("
             "  SELECT e1.MEMBER_ID FROM encounters e1 JOIN encounters e2 "
             "  ON e1.MEMBER_ID = e2.MEMBER_ID AND e1.ENCOUNTER_ID != e2.ENCOUNTER_ID "
             "  AND julianday(e2.ADMIT_DATE) - julianday(e1.DISCHARGE_DATE) BETWEEN 0 AND 30 "
             "  WHERE e1.DISCHARGE_DATE IS NOT NULL AND e2.ADMIT_DATE IS NOT NULL"
             ") r ON e.MEMBER_ID = r.MEMBER_ID"),
        description="Percentage of patients readmitted within 30 days - high rates indicate care quality issues",
        direction="lower_is_better", unit="%", alert_threshold=KPI_THRESHOLDS['readmission_rate'],
    ),
    KPIDefinition(
        name="Member Utilization Rate",
        category="retention",
        sql="SELECT ROUND(100.0 * COUNT(DISTINCT MEMBER_ID) / (SELECT COUNT(*) FROM members), 2) as utilization_rate FROM claims",
        description="Percentage of members actively using services - engagement indicator",
        direction="higher_is_better", unit="%",
    ),
    KPIDefinition(
        name="Average Encounters Per Member",
        category="retention",
        sql="SELECT AVG(enc_count) as avg_encounters FROM (SELECT MEMBER_ID, COUNT(*) as enc_count FROM encounters GROUP BY MEMBER_ID) t",
        description="How frequently members use services - higher means more engaged",
        direction="higher_is_better", unit="encounters",
    ),
    KPIDefinition(
        name="Emergency to Outpatient Ratio",
        category="retention",
        sql="SELECT ROUND(1.0 * COUNT(CASE WHEN VISIT_TYPE = 'EMERGENCY' THEN 1 END) / NULLIF(COUNT(CASE WHEN VISIT_TYPE = 'OUTPATIENT' THEN 1 END), 0), 3) as er_op_ratio FROM encounters",
        description="High ER-to-outpatient ratio suggests members lack preventive care access",
        direction="lower_is_better", unit="ratio", alert_threshold=KPI_THRESHOLDS['er_outpatient_ratio'],
    ),
    KPIDefinition(
        name="Preventive Care Adoption",
        category="retention",
        sql="SELECT ROUND(100.0 * COUNT(CASE WHEN VISIT_TYPE = 'PREVENTIVE' THEN 1 END) / COUNT(*), 2) as preventive_pct FROM encounters",
        description="Percentage of preventive visits - higher means better population health",
        direction="higher_is_better", unit="%",
    ),

    KPIDefinition(
        name="New Member Growth Rate",
        category="acquisition",
        sql="SELECT KP_REGION, COUNT(*) as member_count FROM members GROUP BY KP_REGION ORDER BY member_count DESC",
        description="Member count by region - identifies growth opportunities in underserved areas",
        direction="higher_is_better", unit="members",
    ),
    KPIDefinition(
        name="Service Coverage Gaps",
        category="acquisition",
        sql="SELECT SPECIALTY, COUNT(DISTINCT NPI) as provider_count FROM providers GROUP BY SPECIALTY ORDER BY provider_count ASC LIMIT 10",
        description="Specialties with fewest providers - expansion opportunities",
        direction="monitor", unit="providers",
    ),
    KPIDefinition(
        name="Network Adequacy",
        category="acquisition",
        sql="SELECT ROUND(100.0 * COUNT(CASE WHEN PLAN_TYPE IN ('HMO', 'EPO') THEN 1 END) / COUNT(*), 2) as managed_care_pct FROM claims",
        description="Managed care plan utilization - proxy for network retention",
        direction="higher_is_better", unit="%", alert_threshold=KPI_THRESHOLDS['managed_care_pct'],
    ),

    KPIDefinition(
        name="Cost Per Encounter",
        category="operations",
        sql="SELECT AVG(PAID_AMOUNT) as cost_per_encounter FROM claims",
        description="Average cost per healthcare encounter",
        direction="lower_is_better", unit="$",
    ),
    KPIDefinition(
        name="Provider Utilization Variance",
        category="operations",
        sql="SELECT RENDERING_NPI, COUNT(*) as encounters, AVG(PAID_AMOUNT) as avg_cost FROM claims GROUP BY RENDERING_NPI ORDER BY encounters DESC LIMIT 10",
        description="Provider workload distribution - identifies overloaded/underutilized providers",
        direction="monitor", unit="",
    ),
    KPIDefinition(
        name="Inpatient Average Length of Stay",
        category="operations",
        sql="SELECT AVG(CAST(LENGTH_OF_STAY AS REAL)) as avg_los FROM encounters WHERE VISIT_TYPE = 'INPATIENT' AND LENGTH_OF_STAY IS NOT NULL",
        description="Average inpatient stay duration - shorter is more efficient",
        direction="lower_is_better", unit="days", alert_threshold=KPI_THRESHOLDS['avg_los_days'],
    ),
]


class KPITracker:

    def __init__(self, db_path: str = None):
        self.kpis = {kpi.name: kpi for kpi in HEALTHCARE_KPIS}
        self.history = defaultdict(list)
        self.db_path = db_path

    def compute_kpi(self, kpi_name: str, db_conn=None) -> Optional[Dict]:
        if kpi_name not in self.kpis:
            return None
        kpi = self.kpis[kpi_name]

        if db_conn is None:
            return {'kpi': kpi_name, 'status': 'no_db_connection', 'sql': kpi.sql}

        try:
            cursor = db_conn.execute(kpi.sql)
            rows = cursor.fetchall()
            cols = [desc[0] for desc in cursor.description]

            value = None
            if rows and len(rows) == 1 and len(cols) == 1:
                value = rows[0][0]
            elif rows:
                value = [dict(zip(cols, row)) for row in rows]

            self.history[kpi_name].append((time.time(), value))

            trend = self._detect_trend(kpi_name)

            alert = None
            if kpi.alert_threshold and isinstance(value, (int, float)):
                if kpi.direction == 'lower_is_better' and value > kpi.alert_threshold:
                    alert = "ALERT: {} = {}{} exceeds threshold of {}{}".format(
                        kpi_name, value, kpi.unit, kpi.alert_threshold, kpi.unit)
                elif kpi.direction == 'higher_is_better' and value < kpi.alert_threshold:
                    alert = "ALERT: {} = {}{} below threshold of {}{}".format(
                        kpi_name, value, kpi.unit, kpi.alert_threshold, kpi.unit)

            return {
                'kpi': kpi_name,
                'category': kpi.category,
                'value': value,
                'unit': kpi.unit,
                'description': kpi.description,
                'direction': kpi.direction,
                'target': kpi.target,
                'trend': trend,
                'alert': alert,
            }
        except Exception as e:
            return {'kpi': kpi_name, 'error': str(e), 'sql': kpi.sql}

    def compute_all(self, db_conn=None, category: str = None) -> List[Dict]:
        results = []
        for name, kpi in self.kpis.items():
            if category and kpi.category != category:
                continue
            result = self.compute_kpi(name, db_conn)
            if result:
                results.append(result)
        return results

    def _detect_trend(self, kpi_name: str) -> str:
        history = self.history.get(kpi_name, [])
        numeric_vals = [(ts, v) for ts, v in history if isinstance(v, (int, float))]
        if len(numeric_vals) < 2:
            return "insufficient_data"

        times = [t for t, _ in numeric_vals]
        values = [v for _, v in numeric_vals]

        n = len(times)
        t_mean = sum(times) / n
        v_mean = sum(values) / n

        numerator = sum((times[i] - t_mean) * (values[i] - v_mean) for i in range(n))
        denominator = sum((times[i] - t_mean) ** 2 for i in range(n))

        if abs(denominator) < 1e-10:
            return "stable"

        slope = numerator / denominator
        rel_slope = slope / (abs(v_mean) + 1e-10)

        if rel_slope > 0.01:
            return "increasing"
        elif rel_slope < -0.01:
            return "decreasing"
        return "stable"

    def get_alerts(self, db_conn=None) -> List[Dict]:
        all_kpis = self.compute_all(db_conn)
        return [r for r in all_kpis if r.get('alert')]


class RecommendationEngine:

    def __init__(self, db_conn=None, data_dir: str = None):
        self.db_conn = db_conn
        self.data_dir = data_dir
        self.kpi_tracker = KPITracker()
        self.recommendations = []

    def set_db(self, db_conn):
        self.db_conn = db_conn

    def _query(self, sql: str) -> List[Dict]:
        if self.db_conn is None:
            return []
        try:
            cursor = self.db_conn.execute(sql)
            cols = [d[0] for d in cursor.description]
            return [dict(zip(cols, row)) for row in cursor.fetchall()]
        except Exception:
            return []

    def _query_scalar(self, sql: str):
        rows = self._query(sql)
        if rows:
            return list(rows[0].values())[0]
        return None

    def generate_all_recommendations(self) -> Dict:
        return {
            'revenue': self.revenue_recommendations(),
            'retention': self.retention_recommendations(),
            'acquisition': self.acquisition_recommendations(),
            'operations': self.operational_recommendations(),
            'kpi_summary': self.kpi_summary(),
            'top_actions': self.prioritized_actions(),
        }


    def revenue_recommendations(self) -> List[Dict]:
        recs = []

        denial_rate = self._query_scalar(
            "SELECT ROUND(100.0 * COUNT(CASE WHEN CLAIM_STATUS = 'DENIED' THEN 1 END) / COUNT(*), 2) FROM claims")
        if denial_rate and denial_rate > 10:
            denial_breakdown = self._query(
                "SELECT CPT_CODE, DENIAL_REASON, COUNT(*) as denial_count FROM claims "
                "WHERE CLAIM_STATUS = 'DENIED' GROUP BY CPT_CODE, DENIAL_REASON ORDER BY denial_count DESC LIMIT 5")
            recs.append({
                'title': 'Reduce Claim Denial Rate',
                'priority': 'HIGH',
                'metric': 'Denial Rate: {}%'.format(denial_rate),
                'insight': 'Denial rate exceeds 10% benchmark. Top denied procedures identified.',
                'actions': [
                    'Review coding accuracy for top denied procedure codes',
                    'Implement pre-authorization workflow for high-denial procedures',
                    'Train billing staff on common denial reasons',
                    'Set up automated eligibility verification before claim submission',
                ],
                'expected_impact': 'Reducing denials by 5% could recover ${}K annually'.format(
                    int(denial_rate * 50)),
                'data': denial_breakdown,
            })

        avg_paid = self._query_scalar("SELECT AVG(CAST(PAID_AMOUNT AS REAL)) FROM claims")
        high_value_pct = self._query_scalar(
            "SELECT ROUND(100.0 * COUNT(CASE WHEN PAID_AMOUNT > 5000 THEN 1 END) / COUNT(*), 2) FROM claims")
        if avg_paid and high_value_pct:
            region_revenue = self._query(
                "SELECT KP_REGION, AVG(PAID_AMOUNT) as avg_rev, COUNT(*) as volume "
                "FROM claims GROUP BY KP_REGION ORDER BY avg_rev DESC")
            recs.append({
                'title': 'Optimize Service Mix by Region',
                'priority': 'MEDIUM',
                'metric': 'Avg Revenue/Claim: ${:.2f} | High-Value: {}%'.format(avg_paid, high_value_pct),
                'insight': 'Revenue varies significantly by region. Equalizing toward top region could increase overall revenue.',
                'actions': [
                    'Expand high-margin specialty services in low-revenue regions',
                    'Analyze procedure mix differences between high and low revenue regions',
                    'Cross-train providers on higher-value procedures',
                ],
                'data': region_revenue,
            })

        avg_process = self._query_scalar(
            "SELECT AVG(julianday(ADJUDICATED_DATE) - julianday(SUBMITTED_DATE)) FROM claims "
            "WHERE ADJUDICATED_DATE IS NOT NULL AND SUBMITTED_DATE IS NOT NULL")
        if avg_process and avg_process > 14:
            recs.append({
                'title': 'Accelerate Claims Processing',
                'priority': 'HIGH',
                'metric': 'Avg Processing: {:.1f} days'.format(avg_process),
                'insight': 'Claims take >{:.0f} days to process. Faster processing improves cash flow.'.format(avg_process),
                'actions': [
                    'Implement auto-adjudication for clean claims (<$1000, common procedures)',
                    'Add real-time eligibility checks at point of service',
                    'Automate coding validation to reduce manual review queue',
                    'Set SLA targets: 7 days for clean claims, 14 days for complex',
                ],
                'expected_impact': 'Reducing to 7-day avg could improve cash flow by 30-40%',
            })

        external_rate = self._query_scalar(
            "SELECT ROUND(100.0 * COUNT(CASE WHEN BILLING_NPI != RENDERING_NPI THEN 1 END) / COUNT(*), 2) FROM claims "
            "WHERE BILLING_NPI IS NOT NULL AND RENDERING_NPI IS NOT NULL")
        if external_rate and external_rate > 15:
            recs.append({
                'title': 'Reduce External Referral Leakage',
                'priority': 'HIGH',
                'metric': 'External Referral Rate: {}%'.format(external_rate),
                'insight': 'High external referral rate means revenue may be flowing to outside providers.',
                'actions': [
                    'Identify top externally-referred specialties and recruit in-network alternatives',
                    'Implement smart referral routing to in-network providers',
                    'Add member education on in-network provider search tools',
                ],
            })

        return recs


    def retention_recommendations(self) -> List[Dict]:
        recs = []

        readmit_data = self._query(
            "SELECT e.KP_REGION, "
            "ROUND(100.0 * COUNT(DISTINCT CASE WHEN e2.ENCOUNTER_ID IS NOT NULL THEN e.MEMBER_ID END) "
            "/ NULLIF(COUNT(DISTINCT e.MEMBER_ID), 0), 2) as readmit_rate "
            "FROM encounters e LEFT JOIN encounters e2 "
            "ON e.MEMBER_ID = e2.MEMBER_ID AND e.ENCOUNTER_ID != e2.ENCOUNTER_ID "
            "AND julianday(e2.ADMIT_DATE) - julianday(e.DISCHARGE_DATE) BETWEEN 0 AND 30 "
            "GROUP BY e.KP_REGION ORDER BY readmit_rate DESC")
        if readmit_data:
            worst_region = readmit_data[0] if readmit_data else {}
            recs.append({
                'title': 'Reduce Readmission Rates',
                'priority': 'HIGH',
                'metric': 'Worst Region: {} at {}%'.format(
                    worst_region.get('KP_REGION', '?'), worst_region.get('readmit_rate', '?')),
                'insight': 'High readmissions indicate care gaps and drive member dissatisfaction.',
                'actions': [
                    'Implement 48-hour post-discharge follow-up calls',
                    'Create care transition coordinators for high-risk patients',
                    'Develop discharge planning checklists by condition',
                    'Set up remote patient monitoring for chronic conditions',
                ],
                'expected_impact': 'Each 1% reduction in readmissions saves ~$200K and improves retention',
                'data': readmit_data[:5],
            })

        er_data = self._query(
            "SELECT VISIT_TYPE, COUNT(*) as visit_count, "
            "ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM encounters), 2) as pct "
            "FROM encounters GROUP BY VISIT_TYPE ORDER BY visit_count DESC")
        er_row = next((r for r in er_data if r.get('VISIT_TYPE') == 'EMERGENCY'), None)
        if er_row and er_row.get('pct', 0) > 15:
            recs.append({
                'title': 'Reduce Unnecessary ER Visits',
                'priority': 'MEDIUM',
                'metric': 'ER Visit Rate: {}%'.format(er_row['pct']),
                'insight': 'High ER usage suggests members cannot access primary care easily.',
                'actions': [
                    'Expand same-day/next-day primary care appointment availability',
                    'Launch 24/7 nurse advice line for triage before ER',
                    'Open urgent care centers in high-ER-utilization ZIP codes',
                    'Implement telehealth as first line for non-acute complaints',
                ],
                'data': er_data,
            })

        low_utilizers = self._query_scalar(
            "SELECT COUNT(*) FROM members WHERE MEMBER_ID NOT IN (SELECT DISTINCT MEMBER_ID FROM claims)")
        total_members = self._query_scalar("SELECT COUNT(*) FROM members")
        if low_utilizers and total_members and total_members > 0:
            ghost_pct = 100.0 * low_utilizers / total_members
            if ghost_pct > 10:
                recs.append({
                    'title': 'Re-engage Inactive Members',
                    'priority': 'HIGH',
                    'metric': '{:.1f}% members ({:,}) have zero claims'.format(ghost_pct, low_utilizers),
                    'insight': 'Inactive members are highest churn risk. Precursor outreach can retain them.',
                    'actions': [
                        'Send personalized annual wellness visit reminders',
                        'Offer free health risk assessments to inactive members',
                        'Create "Welcome Back" program with incentivized first visit',
                        'Use predictive model to identify at-risk-to-leave members',
                    ],
                    'expected_impact': 'Re-engaging 20% of inactive members retains ~{:,} members'.format(
                        int(low_utilizers * 0.2)),
                })

        return recs


    def acquisition_recommendations(self) -> List[Dict]:
        recs = []

        region_data = self._query(
            "SELECT KP_REGION, COUNT(*) as member_count FROM members GROUP BY KP_REGION ORDER BY member_count ASC")
        if region_data and len(region_data) >= 2:
            smallest = region_data[0]
            largest = region_data[-1]
            recs.append({
                'title': 'Expand in Underserved Regions',
                'priority': 'HIGH',
                'metric': 'Smallest Region: {} ({:,} members) vs Largest: {} ({:,})'.format(
                    smallest.get('KP_REGION', '?'), smallest.get('member_count', 0),
                    largest.get('KP_REGION', '?'), largest.get('member_count', 0)),
                'insight': 'Significant membership disparity across regions indicates growth opportunity.',
                'actions': [
                    'Conduct market analysis in low-membership regions',
                    'Partner with local employers for group enrollment',
                    'Open satellite clinics in underserved areas',
                    'Launch targeted marketing campaigns in growth regions',
                ],
                'data': region_data,
            })

        specialty_data = self._query(
            "SELECT SPECIALTY, COUNT(DISTINCT NPI) as provider_count "
            "FROM providers GROUP BY SPECIALTY ORDER BY provider_count ASC LIMIT 5")
        if specialty_data:
            recs.append({
                'title': 'Fill Specialty Coverage Gaps',
                'priority': 'MEDIUM',
                'metric': 'Underserved Specialties: {}'.format(
                    ', '.join(r.get('SPECIALTY', '?') for r in specialty_data[:3])),
                'insight': 'Low provider counts in these specialties could be driving members to competitors.',
                'actions': [
                    'Recruit providers in underserved specialties',
                    'Build telehealth partnerships for rare specialties',
                    'Create referral agreements with specialty groups',
                ],
                'data': specialty_data,
            })

        avg_process = self._query_scalar(
            "SELECT AVG(julianday(ADJUDICATED_DATE) - julianday(SUBMITTED_DATE)) FROM claims "
            "WHERE ADJUDICATED_DATE IS NOT NULL AND SUBMITTED_DATE IS NOT NULL")
        denial_rate = self._query_scalar(
            "SELECT ROUND(100.0 * COUNT(CASE WHEN CLAIM_STATUS = 'DENIED' THEN 1 END) / COUNT(*), 2) FROM claims")
        if avg_process and denial_rate:
            recs.append({
                'title': 'Differentiate on Service Quality',
                'priority': 'MEDIUM',
                'metric': 'Processing: {:.0f} days | Denial Rate: {}%'.format(avg_process, denial_rate),
                'insight': 'Publish quality metrics (fast claims, low denials) as competitive advantage.',
                'actions': [
                    'Create member-facing quality dashboard',
                    'Publish NPS and satisfaction scores in marketing',
                    'Offer claims processing guarantees (7-day resolution)',
                    'Build employer reporting portal showing plan performance',
                ],
            })

        return recs


    def operational_recommendations(self) -> List[Dict]:
        recs = []

        provider_stats = self._query(
            "SELECT RENDERING_NPI, COUNT(*) as encounters, AVG(PAID_AMOUNT) as avg_cost "
            "FROM claims GROUP BY RENDERING_NPI HAVING COUNT(*) > 5 ORDER BY encounters DESC LIMIT 10")
        if provider_stats and len(provider_stats) >= 3:
            volumes = [r['encounters'] for r in provider_stats]
            mean_vol = sum(volumes) / len(volumes)
            variance = sum((v - mean_vol) ** 2 for v in volumes) / len(volumes)
            cv = math.sqrt(variance) / mean_vol if mean_vol > 0 else 0

            if cv > 0.5:
                recs.append({
                    'title': 'Balance Provider Workloads',
                    'priority': 'MEDIUM',
                    'metric': 'Workload CV: {:.2f} (target <0.5)'.format(cv),
                    'insight': 'High variation in provider encounter volumes indicates scheduling inefficiency.',
                    'actions': [
                        'Implement intelligent patient routing based on provider capacity',
                        'Create overflow protocols for high-volume providers',
                        'Analyze panel sizes and adjust patient assignments',
                    ],
                    'data': provider_stats[:5],
                })

        cost_data = self._query(
            "SELECT CPT_CODE, COUNT(*) as freq, AVG(PAID_AMOUNT) as avg_cost, "
            "MAX(PAID_AMOUNT) as max_cost "
            "FROM claims GROUP BY CPT_CODE HAVING COUNT(*) > 5 "
            "ORDER BY avg_cost DESC LIMIT 10")
        if cost_data:
            recs.append({
                'title': 'Address Cost Outliers by Procedure',
                'priority': 'MEDIUM',
                'metric': 'Top cost procedure: {} (avg ${:.0f})'.format(
                    cost_data[0].get('CPT_CODE', '?'),
                    cost_data[0].get('avg_cost', 0)),
                'insight': 'Review high-cost procedures for standardization and cost reduction.',
                'actions': [
                    'Develop clinical pathways for top 10 costliest procedures',
                    'Negotiate better rates with suppliers for common procedures',
                    'Implement utilization review for procedures >$5000',
                ],
                'data': cost_data[:5],
            })

        return recs


    def kpi_summary(self) -> Dict:
        all_kpis = self.kpi_tracker.compute_all(self.db_conn)
        alerts = [k for k in all_kpis if k.get('alert')]
        by_category = defaultdict(list)
        for k in all_kpis:
            by_category[k.get('category', 'other')].append(k)
        return {
            'total_kpis': len(all_kpis),
            'alerts': alerts,
            'by_category': dict(by_category),
        }


    def prioritized_actions(self) -> List[Dict]:
        all_recs = (
            self.revenue_recommendations() +
            self.retention_recommendations() +
            self.acquisition_recommendations() +
            self.operational_recommendations()
        )

        priority_score = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
        for r in all_recs:
            r['_score'] = priority_score.get(r.get('priority', 'LOW'), 1)

        all_recs.sort(key=lambda x: x['_score'], reverse=True)

        top_actions = []
        for r in all_recs[:5]:
            top_actions.append({
                'action': r['title'],
                'priority': r['priority'],
                'metric': r.get('metric', ''),
                'first_step': r.get('actions', ['No specific action'])[0],
            })

        return top_actions


class AssociationRuleMiner:

    def __init__(self, min_support: float = MIN_SUPPORT, min_confidence: float = MIN_CONFIDENCE,
                 min_lift: float = MIN_LIFT):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift

    def find_rules(self, transactions: List[set]) -> List[Dict]:
        n = len(transactions)
        if n == 0:
            return []

        item_counts = Counter()
        for t in transactions:
            for item in t:
                item_counts[item] += 1

        frequent_items = {item for item, count in item_counts.items()
                          if count / n >= self.min_support}

        pair_counts = Counter()
        for t in transactions:
            items = [i for i in t if i in frequent_items]
            for pair in itertools.combinations(items, 2):
                pair = tuple(sorted(pair))
                pair_counts[pair] += 1

        rules = []
        for (a, b), count in pair_counts.items():
            support = count / n
            if support < self.min_support:
                continue

            conf_ab = count / item_counts[a]
            lift_ab = conf_ab / (item_counts[b] / n)
            if conf_ab >= self.min_confidence and lift_ab >= self.min_lift:
                rules.append({
                    'antecedent': a,
                    'consequent': b,
                    'support': round(support, 4),
                    'confidence': round(conf_ab, 4),
                    'lift': round(lift_ab, 4),
                })

            conf_ba = count / item_counts[b]
            lift_ba = conf_ba / (item_counts[a] / n)
            if conf_ba >= self.min_confidence and lift_ba >= self.min_lift:
                rules.append({
                    'antecedent': b,
                    'consequent': a,
                    'support': round(support, 4),
                    'confidence': round(conf_ba, 4),
                    'lift': round(lift_ba, 4),
                })

        rules.sort(key=lambda r: r['lift'], reverse=True)
        return rules


def load_csv_to_sqlite(data_dir: str) -> sqlite3.Connection:
    conn = sqlite3.connect(':memory:')
    if not os.path.isdir(data_dir):
        return conn

    for fname in os.listdir(data_dir):
        if not fname.endswith('.csv'):
            continue
        table_name = fname.replace('.csv', '')
        fpath = os.path.join(data_dir, fname)
        try:
            with open(fpath, 'r') as f:
                reader = csv.reader(f)
                headers = next(reader)
                cols = ', '.join('"{}"'.format(h.strip()) for h in headers)
                placeholders = ', '.join(['?'] * len(headers))
                conn.execute("CREATE TABLE IF NOT EXISTS {} ({})".format(
                    table_name, ', '.join('"{}" TEXT'.format(h.strip()) for h in headers)))
                rows = list(reader)
                if rows:
                    conn.executemany("INSERT INTO {} VALUES ({})".format(
                        table_name, placeholders), rows)
            conn.commit()
        except Exception as e:
            print("  Warning loading {}: {}".format(fname, e))
    return conn


if __name__ == '__main__':
    print("=" * 70)
    print("HEALTHCARE RECOMMENDATION ENGINE")
    print("=" * 70)

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    print("\n[1] Loading data from {}...".format(data_dir))
    conn = load_csv_to_sqlite(data_dir)

    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    print("  Tables loaded: {}".format(tables))

    print("\n[2] Generating recommendations...")
    engine = RecommendationEngine(db_conn=conn)
    results = engine.generate_all_recommendations()

    for category in ['revenue', 'retention', 'acquisition', 'operations']:
        recs = results.get(category, [])
        print("\n--- {} ({} recommendations) ---".format(category.upper(), len(recs)))
        for r in recs:
            print("  [{}] {}".format(r.get('priority', '?'), r.get('title', '?')))
            print("    Metric: {}".format(r.get('metric', 'N/A')))
            print("    Insight: {}".format(r.get('insight', 'N/A')[:80]))
            if r.get('actions'):
                print("    Top Action: {}".format(r['actions'][0]))

    print("\n--- TOP 5 PRIORITIZED ACTIONS ---")
    for i, action in enumerate(results.get('top_actions', []), 1):
        print("  {}. [{}] {} -> {}".format(
            i, action['priority'], action['action'], action['first_step'][:60]))

    alerts = results.get('kpi_summary', {}).get('alerts', [])
    if alerts:
        print("\n--- KPI ALERTS ---")
        for a in alerts:
            print("  {} {}".format(a.get('alert', ''), a.get('kpi', '')))

    conn.close()
    print("\n" + "=" * 70)
