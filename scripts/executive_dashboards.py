import sqlite3
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
from statistics import mean, median, quantiles

logger = logging.getLogger('kp.executive_dashboards')
logger.setLevel(logging.DEBUG)


class ExecutiveDashboardEngine:

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._validate_db()

    def _validate_db(self):
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("SELECT 1")
            conn.close()
            logger.info(f"Connected to healthcare database: {self.db_path}")
        except Exception as e:
            logger.error(f"Database validation failed: {e}")
            raise

    def _query(self, sql: str, params: tuple = ()) -> List[Any]:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(sql, params)
            results = cursor.fetchall()
            conn.close()
            return results
        except Exception as e:
            logger.error(f"Query error: {e}\nSQL: {sql}")
            return []

    def _query_one(self, sql: str, params: tuple = ()) -> Any:
        results = self._query(sql, params)
        return results[0] if results else None

    def _query_scalar(self, sql: str, params: tuple = ()) -> Any:
        result = self._query_one(sql, params)
        return result[0] if result else None

    def get_member_experience(self) -> Dict[str, Any]:
        dashboard = {
            'title': 'Member Experience Dashboard',
            'subtitle': 'CGA Regional Overview, Satisfaction, Retention',
            'generated_at': datetime.now().isoformat(),
            'sections': {}
        }

        total_members = self._query_scalar(
            'SELECT COUNT(*) FROM members'
        ) or 0

        disenroll_data = self._query_one('''
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN DISENROLLMENT_DATE IS NOT NULL AND DISENROLLMENT_DATE != ''
                         THEN 1 ELSE 0 END) as disenrolled
            FROM members
        ''')

        if disenroll_data:
            total, disenrolled = disenroll_data[0], disenroll_data[1] or 0
            voluntary_termination_rate = (disenrolled / total * 100) if total > 0 else 0
        else:
            voluntary_termination_rate = 0

        dashboard['sections']['retention'] = {
            'title': 'Member Retention Metrics',
            'metrics': {
                'total_members': {
                    'value': total_members,
                    'label': 'Total Enrolled Members',
                    'format': 'number'
                },
                'voluntary_termination_rate': {
                    'value': round(voluntary_termination_rate, 2),
                    'label': 'Voluntary Termination Rate',
                    'format': 'percent',
                    'benchmark': '8-12%',
                    'status': 'green' if voluntary_termination_rate < 10 else 'amber' if voluntary_termination_rate < 12 else 'red'
                },
                'retention_rate': {
                    'value': round(100 - voluntary_termination_rate, 2),
                    'label': 'Member Retention Rate',
                    'format': 'percent',
                    'benchmark': '88-92%',
                    'status': 'green' if voluntary_termination_rate < 10 else 'amber' if voluntary_termination_rate < 12 else 'red'
                }
            }
        }

        regional_denials = self._query('''
            SELECT
                KP_REGION,
                COUNT(*) as total_claims,
                SUM(CASE WHEN CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) as denied_claims,
                COUNT(DISTINCT MEMBER_ID) as members
            FROM claims
            GROUP BY KP_REGION
            ORDER BY KP_REGION
        ''')

        dashboard['sections']['regional_comparison'] = {
            'title': 'Regional Performance Comparison',
            'data': [
                {
                    'region': r['KP_REGION'],
                    'members': r['members'],
                    'denial_rate': round(r['denied_claims'] / r['total_claims'] * 100, 2) if r['total_claims'] > 0 else 0,
                    'satisfaction_proxy': round(100 - (r['denied_claims'] / r['total_claims'] * 100), 2) if r['total_claims'] > 0 else 100,
                    'status': 'green' if (r['denied_claims'] / r['total_claims'] * 100 if r['total_claims'] > 0 else 0) < 10 else 'amber' if (r['denied_claims'] / r['total_claims'] * 100 if r['total_claims'] > 0 else 0) < 15 else 'red'
                }
                for r in regional_denials
            ]
        }

        denial_issues = self._query('''
            SELECT
                DENIAL_REASON,
                COUNT(*) as issue_count,
                SUM(CAST(BILLED_AMOUNT AS REAL)) as amount_impacted
            FROM claims
            WHERE DENIAL_REASON IS NOT NULL AND DENIAL_REASON != ''
            GROUP BY DENIAL_REASON
            ORDER BY COUNT(*) DESC
            LIMIT 10
        ''')

        dashboard['sections']['member_issues'] = {
            'title': 'Top Member Complaint Drivers (via Denials)',
            'insight': 'Denial reasons indicate member friction points requiring resolution',
            'data': [
                {
                    'issue': r['DENIAL_REASON'],
                    'frequency': r['issue_count'],
                    'revenue_impacted': round(r['amount_impacted'], 2),
                    'priority': 'critical' if r['issue_count'] > 50 else 'high' if r['issue_count'] > 20 else 'medium'
                }
                for r in denial_issues
            ]
        }

        enrollment_trend = self._query('''
            SELECT
                SUBSTR(ENROLLMENT_DATE, 1, 7) as month,
                COUNT(*) as new_members,
                COUNT(DISTINCT KP_REGION) as regions
            FROM members
            WHERE ENROLLMENT_DATE IS NOT NULL AND ENROLLMENT_DATE != ''
            GROUP BY SUBSTR(ENROLLMENT_DATE, 1, 7)
            ORDER BY month DESC
            LIMIT 12
        ''')

        dashboard['sections']['enrollment_trend'] = {
            'title': 'Monthly Enrollment Trend (Last 12 Months)',
            'data': [
                {
                    'period': r['month'],
                    'new_members': r['new_members'],
                    'regions_active': r['regions']
                }
                for r in enrollment_trend
            ]
        }

        dashboard['sections']['rag_status'] = {
            'red_count': 1 if voluntary_termination_rate > 12 else 0,
            'amber_count': 1 if 10 <= voluntary_termination_rate <= 12 else 0,
            'green_count': 1 if voluntary_termination_rate < 10 else 0,
            'overall': 'red' if voluntary_termination_rate > 12 else 'amber' if voluntary_termination_rate > 10 else 'green'
        }

        return dashboard

    def get_stars_performance(self) -> Dict[str, Any]:
        dashboard = {
            'title': 'Stars Measure Performance Dashboard',
            'subtitle': 'HEDIS, CAHPS, Clinical Quality, Administrative Performance',
            'generated_at': datetime.now().isoformat(),
            'sections': {}
        }

        total_members = self._query_scalar('SELECT COUNT(*) FROM members') or 1

        preventive_cpt_members = self._query_scalar('''
            SELECT COUNT(DISTINCT MEMBER_ID) FROM claims
            WHERE CPT_DESCRIPTION LIKE '%preventive%'
               OR CPT_DESCRIPTION LIKE '%screening%'
               OR CPT_DESCRIPTION LIKE '%wellness%'
               OR CPT_DESCRIPTION LIKE '%annual%'
               OR CPT_CODE IN ('99381','99382','99383','99384','99385','99386','99387',
                               '99391','99392','99393','99394','99395','99396','99397')
        ''') or 0

        outpatient_members = self._query_scalar('''
            SELECT COUNT(DISTINCT MEMBER_ID) FROM encounters WHERE VISIT_TYPE = 'OUTPATIENT'
        ''') or 0

        preventive_visits = preventive_cpt_members if preventive_cpt_members > 0 else outpatient_members
        preventive_care_pct = (preventive_visits / total_members * 100)

        outpatient_access_pct = (outpatient_members / total_members * 100)

        chronic_mgmt = self._query_one('''
            SELECT
                COUNT(DISTINCT d.MEMBER_ID) as members_with_chronic,
                COUNT(DISTINCT CASE WHEN e.ENCOUNTER_ID IS NOT NULL
                                    THEN d.MEMBER_ID END) as managed_members
            FROM diagnoses d
            LEFT JOIN encounters e ON d.MEMBER_ID = e.MEMBER_ID
                AND date(e.SERVICE_DATE) >= date(d.DIAGNOSIS_DATE)
            WHERE d.IS_CHRONIC = 'Y'
        ''')

        if chronic_mgmt:
            chronic_managed_pct = (chronic_mgmt[1] / max(chronic_mgmt[0], 1) * 100) if chronic_mgmt[0] else 0
        else:
            chronic_managed_pct = 0

        dashboard['sections']['hedis_measures'] = {
            'title': 'HEDIS Preventive Care Measures',
            'measures': [
                {
                    'measure': 'Preventive Care & Screenings (CPT-based)',
                    'numerator': int(preventive_cpt_members),
                    'denominator': total_members,
                    'rate': round(preventive_care_pct, 2),
                    'benchmark': 10.0,
                    'status': 'green' if preventive_care_pct >= 5 else 'amber' if preventive_care_pct >= 3 else 'red',
                    '5_star_cut': 10.0,
                    'note': 'Based on preventive CPT codes in claims data'
                },
                {
                    'measure': 'Outpatient Access Rate',
                    'numerator': int(outpatient_members),
                    'denominator': total_members,
                    'rate': round(outpatient_access_pct, 2),
                    'benchmark': 50.0,
                    'status': 'green' if outpatient_access_pct >= 40 else 'amber' if outpatient_access_pct >= 30 else 'red',
                    '5_star_cut': 50.0,
                    'note': 'Members with at least one outpatient visit'
                },
                {
                    'measure': 'Chronic Disease Management',
                    'numerator': int(chronic_mgmt[1] if chronic_mgmt else 0),
                    'denominator': int(chronic_mgmt[0] if chronic_mgmt else 1),
                    'rate': round(chronic_managed_pct, 2),
                    'benchmark': 55.0,
                    'status': 'green' if chronic_managed_pct >= 50 else 'amber' if chronic_managed_pct >= 40 else 'red',
                    '5_star_cut': 55.0
                }
            ]
        }

        disenroll_rate = self._query_one('''
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN DISENROLLMENT_DATE IS NOT NULL AND DISENROLLMENT_DATE != ''
                         THEN 1 ELSE 0 END) as disenrolled
            FROM members
        ''')

        if disenroll_rate:
            cahps_satisfaction = round(100 - (disenroll_rate[1] / max(disenroll_rate[0], 1) * 100), 2)
        else:
            cahps_satisfaction = 50.0

        dashboard['sections']['cahps_measures'] = {
            'title': 'CAHPS Member Satisfaction',
            'measures': [
                {
                    'measure': 'Overall Plan Rating',
                    'rate': cahps_satisfaction,
                    'benchmark': 85.0,
                    'status': 'green' if cahps_satisfaction >= 85 else 'amber' if cahps_satisfaction >= 75 else 'red',
                    '5_star_cut': 85.0
                },
                {
                    'measure': 'Getting Care & Service',
                    'rate': cahps_satisfaction,
                    'benchmark': 83.0,
                    'status': 'green' if cahps_satisfaction >= 83 else 'amber' if cahps_satisfaction >= 73 else 'red',
                    '5_star_cut': 83.0
                },
                {
                    'measure': 'Doctor Communication',
                    'rate': cahps_satisfaction,
                    'benchmark': 80.0,
                    'status': 'green' if cahps_satisfaction >= 80 else 'amber' if cahps_satisfaction >= 70 else 'red',
                    '5_star_cut': 80.0
                }
            ]
        }

        clinical_quality = self._query('''
            SELECT
                CASE WHEN d.ICD10_CODE LIKE 'E11%' THEN 'Diabetes Control'
                     WHEN d.ICD10_CODE LIKE 'I10%' THEN 'Hypertension Control'
                     WHEN d.ICD10_CODE LIKE 'J44%' THEN 'COPD Management'
                     WHEN d.ICD10_CODE LIKE 'I50%' THEN 'Heart Failure Care'
                     ELSE 'Other Chronic Care'
                END as condition,
                COUNT(DISTINCT d.MEMBER_ID) as patients,
                SUM(CASE WHEN d.SEVERITY IN ('SEVERE', 'CRITICAL') THEN 1 ELSE 0 END) as severe_count,
                COUNT(DISTINCT e.ENCOUNTER_ID) as total_encounters
            FROM diagnoses d
            LEFT JOIN encounters e ON d.MEMBER_ID = e.MEMBER_ID
            WHERE d.IS_CHRONIC = 'Y'
            GROUP BY condition
            ORDER BY COUNT(DISTINCT d.MEMBER_ID) DESC
            LIMIT 5
        ''')

        dashboard['sections']['clinical_quality'] = {
            'title': 'Clinical Quality Measures',
            'data': [
                {
                    'condition': r['condition'],
                    'patients': r['patients'],
                    'control_rate': round(100 - (r['severe_count'] / max(r['patients'], 1) * 100), 2),
                    'visit_frequency': round(r['total_encounters'] / max(r['patients'], 1), 1) if r['patients'] else 0
                }
                for r in clinical_quality
            ]
        }

        admin_metrics = self._query_one('''
            SELECT
                COUNT(*) as total_claims,
                SUM(CASE WHEN CLAIM_STATUS = 'PAID' THEN 1 ELSE 0 END) as paid_claims,
                SUM(CASE WHEN CLAIM_STATUS = 'PENDING' THEN 1 ELSE 0 END) as pending_claims,
                AVG(CASE WHEN ADJUDICATED_DATE != '' AND SUBMITTED_DATE != ''
                         THEN julianday(ADJUDICATED_DATE) - julianday(SUBMITTED_DATE)
                         ELSE NULL END) as avg_days_to_process
            FROM claims
        ''')

        if admin_metrics:
            claims_paid_pct = round(admin_metrics[1] / max(admin_metrics[0], 1) * 100, 2)
            claims_pending_pct = round(admin_metrics[2] / max(admin_metrics[0], 1) * 100, 2)
            days_to_process = round(admin_metrics[3], 1) if admin_metrics[3] else 0
        else:
            claims_paid_pct = 0
            claims_pending_pct = 0
            days_to_process = 0

        dashboard['sections']['admin_measures'] = {
            'title': 'Administrative Performance',
            'measures': [
                {
                    'measure': 'Clean Claims Rate',
                    'rate': claims_paid_pct,
                    'benchmark': 95.0,
                    'status': 'green' if claims_paid_pct >= 95 else 'amber' if claims_paid_pct >= 90 else 'red',
                    '5_star_cut': 95.0
                },
                {
                    'measure': 'Claims Processing Time (Days)',
                    'value': days_to_process,
                    'benchmark': 30.0,
                    'status': 'green' if days_to_process <= 30 else 'amber' if days_to_process <= 45 else 'red',
                    '5_star_cut': 30.0
                },
                {
                    'measure': 'Pending Claims %',
                    'rate': claims_pending_pct,
                    'benchmark': 5.0,
                    'status': 'green' if claims_pending_pct <= 5 else 'amber' if claims_pending_pct <= 10 else 'red',
                    '5_star_cut': 5.0
                }
            ]
        }

        hedis_score = min(outpatient_access_pct / 50 * 100, 100)
        overall_stars = (
            (hedis_score / 100 * 5) * 0.15 +
            (cahps_satisfaction / 100 * 5) * 0.25 +
            (chronic_managed_pct / 100 * 5) * 0.20 +
            (claims_paid_pct / 100 * 5) * 0.15 +
            3.5 * 0.25
        )

        dashboard['sections']['overall_rating'] = {
            'title': 'Overall Star Rating',
            'rating': round(overall_stars, 2),
            'max_rating': 5.0,
            'benchmark': 4.0,
            'status': 'green' if overall_stars >= 4.0 else 'amber' if overall_stars >= 3.5 else 'red',
            'composition': {
                'hedis_weight': 0.15,
                'cahps_weight': 0.25,
                'clinical_quality_weight': 0.25,
                'admin_weight': 0.15,
                'other_weight': 0.20
            }
        }

        return dashboard

    def get_risk_adjustment_coding(self) -> Dict[str, Any]:
        dashboard = {
            'title': 'Risk Adjustment & Coding Accuracy (RADA)',
            'subtitle': 'Medicare Advantage Risk Score Optimization',
            'generated_at': datetime.now().isoformat(),
            'sections': {}
        }

        risk_stats = self._query_one('''
            SELECT
                COUNT(*) as members,
                AVG(CAST(RISK_SCORE AS REAL)) as avg_risk,
                MIN(CAST(RISK_SCORE AS REAL)) as min_risk,
                MAX(CAST(RISK_SCORE AS REAL)) as max_risk
            FROM members
            WHERE RISK_SCORE IS NOT NULL AND RISK_SCORE != ''
        ''')

        if risk_stats:
            total_members = risk_stats[0]
            avg_risk = round(risk_stats[1], 3) if risk_stats[1] else 0
            min_risk = round(risk_stats[2], 3) if risk_stats[2] else 0
            max_risk = round(risk_stats[3], 3) if risk_stats[3] else 0
        else:
            total_members = 0
            avg_risk = 1.0
            min_risk = 0
            max_risk = 2.0

        dashboard['sections']['risk_score_summary'] = {
            'title': 'Risk Score Summary',
            'metrics': {
                'average_risk_score': {
                    'value': avg_risk,
                    'label': 'Average Risk Score',
                    'benchmark': 1.0,
                    'status': 'green' if avg_risk >= 1.0 else 'amber' if avg_risk >= 0.95 else 'red'
                },
                'members_assessed': {
                    'value': total_members,
                    'label': 'Members with Risk Scores',
                    'format': 'number'
                },
                'risk_range': {
                    'min': min_risk,
                    'max': max_risk,
                    'label': 'Risk Score Range'
                }
            }
        }

        risk_distribution = self._query('''
            SELECT
                CASE WHEN CAST(RISK_SCORE AS REAL) < 0.5 THEN '0.0-0.5 (Low)'
                     WHEN CAST(RISK_SCORE AS REAL) < 1.0 THEN '0.5-1.0 (Moderate)'
                     WHEN CAST(RISK_SCORE AS REAL) < 1.5 THEN '1.0-1.5 (High)'
                     WHEN CAST(RISK_SCORE AS REAL) < 2.0 THEN '1.5-2.0 (Very High)'
                     ELSE '2.0+ (Extreme)'
                END as risk_tier,
                COUNT(*) as members,
                ROUND(AVG(CAST(RISK_SCORE AS REAL)), 3) as avg_tier_risk,
                ROUND(AVG(CAST(CHRONIC_CONDITIONS AS INTEGER)), 1) as avg_conditions
            FROM members
            WHERE RISK_SCORE IS NOT NULL AND RISK_SCORE != ''
            GROUP BY risk_tier
            ORDER BY AVG(CAST(RISK_SCORE AS REAL))
        ''')

        dashboard['sections']['risk_distribution'] = {
            'title': 'Risk Score Distribution',
            'total_members': total_members,
            'data': [
                {
                    'tier': r['risk_tier'],
                    'members': r['members'],
                    'pct': round(r['members'] / max(total_members, 1) * 100, 2),
                    'avg_risk': r['avg_tier_risk'],
                    'avg_chronic_conditions': r['avg_conditions']
                }
                for r in risk_distribution
            ]
        }

        hcc_capture = self._query_one('''
            SELECT
                COUNT(DISTINCT d.MEMBER_ID) as members_with_hcc,
                COUNT(DISTINCT CASE WHEN d.HCC_CODE IS NOT NULL AND d.HCC_CODE != ''
                                    THEN d.HCC_CODE END) as unique_hccs,
                COUNT(CASE WHEN d.HCC_CODE IS NOT NULL AND d.HCC_CODE != '' THEN 1 END) as total_hccs
            FROM diagnoses d
            WHERE IS_CHRONIC = 'Y'
        ''')

        if hcc_capture:
            members_with_hcc = hcc_capture[0]
            unique_hccs = hcc_capture[1]
            total_hccs = hcc_capture[2]
            hcc_capture_rate = round(members_with_hcc / max(total_members, 1) * 100, 2)
            hccs_per_member = round(total_hccs / max(members_with_hcc, 1), 2) if members_with_hcc else 0
        else:
            members_with_hcc = 0
            unique_hccs = 0
            total_hccs = 0
            hcc_capture_rate = 0
            hccs_per_member = 0

        dashboard['sections']['hcc_capture'] = {
            'title': 'HCC Capture & Coding Accuracy',
            'metrics': {
                'hcc_capture_rate': {
                    'value': hcc_capture_rate,
                    'label': 'Members with HCC Codes',
                    'format': 'percent',
                    'benchmark': 95.0,
                    'status': 'green' if hcc_capture_rate >= 95 else 'amber' if hcc_capture_rate >= 85 else 'red'
                },
                'hccs_per_member': {
                    'value': hccs_per_member,
                    'label': 'Average HCCs per Member',
                    'benchmark': 2.5,
                    'status': 'green' if hccs_per_member >= 2.5 else 'amber' if hccs_per_member >= 2.0 else 'red'
                },
                'unique_hcc_types': {
                    'value': unique_hccs,
                    'label': 'Unique HCC Categories Captured',
                    'format': 'number'
                }
            }
        }

        top_hccs = self._query('''
            SELECT
                HCC_CATEGORY,
                COUNT(*) as frequency,
                COUNT(DISTINCT MEMBER_ID) as member_count
            FROM diagnoses
            WHERE HCC_CODE IS NOT NULL AND HCC_CODE != '' AND IS_CHRONIC = 'Y'
            GROUP BY HCC_CATEGORY
            ORDER BY COUNT(*) DESC
            LIMIT 15
        ''')

        dashboard['sections']['hcc_categories'] = {
            'title': 'Top HCC Categories',
            'data': [
                {
                    'category': r['HCC_CATEGORY'],
                    'frequency': r['frequency'],
                    'affected_members': r['member_count'],
                    'revenue_potential': r['member_count'] * 0.1
                }
                for r in top_hccs
            ]
        }

        regional_risk = self._query('''
            SELECT
                KP_REGION,
                COUNT(*) as members,
                ROUND(AVG(CAST(RISK_SCORE AS REAL)), 3) as avg_risk,
                ROUND(AVG(CAST(CHRONIC_CONDITIONS AS INTEGER)), 1) as avg_conditions
            FROM members
            WHERE RISK_SCORE IS NOT NULL AND RISK_SCORE != ''
            GROUP BY KP_REGION
            ORDER BY AVG(CAST(RISK_SCORE AS REAL)) DESC
        ''')

        dashboard['sections']['risk_by_region'] = {
            'title': 'Risk Score Performance by Region',
            'benchmark_avg': avg_risk,
            'data': [
                {
                    'region': r['KP_REGION'],
                    'members': r['members'],
                    'avg_risk': r['avg_risk'],
                    'vs_benchmark': round(r['avg_risk'] - avg_risk, 3),
                    'status': 'green' if r['avg_risk'] >= avg_risk else 'red'
                }
                for r in regional_risk
            ]
        }

        avg_pmpm_baseline = 100
        projected_pmpm = avg_pmpm_baseline * avg_risk
        total_projected_revenue = projected_pmpm * total_members / 12

        dashboard['sections']['revenue_impact'] = {
            'title': 'Risk-Adjusted Revenue Impact',
            'assumptions': {
                'baseline_pmpm': avg_pmpm_baseline,
                'members': total_members
            },
            'metrics': {
                'risk_adjusted_pmpm': {
                    'value': round(projected_pmpm, 2),
                    'label': 'Risk-Adjusted PMPM',
                    'calculation': f'${avg_pmpm_baseline} × {avg_risk} risk factor'
                },
                'monthly_revenue': {
                    'value': round(total_projected_revenue, 0),
                    'label': 'Projected Monthly Revenue',
                    'format': 'currency'
                },
                'annual_revenue': {
                    'value': round(total_projected_revenue * 12, 0),
                    'label': 'Projected Annual Revenue',
                    'format': 'currency'
                }
            }
        }

        return dashboard

    def get_financial_performance(self) -> Dict[str, Any]:
        dashboard = {
            'title': 'Financial Performance Dashboard',
            'subtitle': 'YTD PMPM Financials, Revenue/Expense Analysis',
            'generated_at': datetime.now().isoformat(),
            'sections': {}
        }

        total_members = self._query_scalar('SELECT COUNT(*) FROM members') or 1
        member_months = total_members * 12

        financials = self._query_one('''
            SELECT
                COUNT(*) as total_claims,
                SUM(CAST(BILLED_AMOUNT AS REAL)) as total_billed,
                SUM(CAST(PAID_AMOUNT AS REAL)) as total_paid,
                SUM(CAST(ALLOWED_AMOUNT AS REAL)) as total_allowed,
                SUM(CAST(COPAY AS REAL)) as total_copay,
                SUM(CASE WHEN CLAIM_STATUS = 'DENIED' THEN CAST(BILLED_AMOUNT AS REAL) ELSE 0 END) as denied_amount
            FROM claims
        ''')

        if financials:
            total_claims, total_billed, total_paid, total_allowed, total_copay, denied_amount = financials
            total_billed = total_billed or 0
            total_paid = total_paid or 0
            total_allowed = total_allowed or 0
            total_copay = total_copay or 0
            denied_amount = denied_amount or 0

            pmpm_revenue = total_paid / member_months
            pmpm_cost = total_billed / member_months
            pmpm_allowed = total_allowed / member_months
            mlr = (total_allowed / total_billed * 100) if total_billed > 0 else 0
            collection_rate = (total_paid / total_allowed * 100) if total_allowed > 0 else 0
            denial_savings = denied_amount
        else:
            pmpm_revenue = 0
            pmpm_cost = 0
            mlr = 0
            denial_savings = 0

        dashboard['sections']['ytd_summary'] = {
            'title': 'YTD Financial Summary',
            'metrics': {
                'pmpm_revenue': {
                    'value': round(pmpm_revenue, 2),
                    'label': 'PMPM Revenue (Paid)',
                    'format': 'currency',
                    'benchmark': '$350-$500 (commercial), $800-$1200 (MA)'
                },
                'pmpm_cost': {
                    'value': round(pmpm_cost, 2),
                    'label': 'PMPM Cost (Billed)',
                    'format': 'currency'
                },
                'medical_loss_ratio': {
                    'value': round(mlr, 2),
                    'label': 'Medical Loss Ratio (Allowed/Billed)',
                    'format': 'percent',
                    'benchmark': '65-75%',
                    'status': 'green' if 60 <= mlr <= 80 else 'amber' if 50 <= mlr <= 85 else 'red'
                },
                'collection_rate': {
                    'value': round(collection_rate, 2),
                    'label': 'Collection Rate (Paid/Allowed)',
                    'format': 'percent',
                    'benchmark': '50%+',
                    'status': 'green' if collection_rate >= 50 else 'amber' if collection_rate >= 40 else 'red'
                },
                'total_members': {
                    'value': total_members,
                    'label': 'Total Members',
                    'format': 'number'
                }
            }
        }

        by_claim_type = self._query('''
            SELECT
                CLAIM_TYPE,
                COUNT(*) as claims,
                SUM(CAST(BILLED_AMOUNT AS REAL)) as billed,
                SUM(CAST(PAID_AMOUNT AS REAL)) as paid,
                SUM(CAST(ALLOWED_AMOUNT AS REAL)) as allowed
            FROM claims
            GROUP BY CLAIM_TYPE
            ORDER BY SUM(CAST(PAID_AMOUNT AS REAL)) DESC
        ''')

        dashboard['sections']['expense_by_type'] = {
            'title': 'Expense Breakdown by Claim Type',
            'data': [
                {
                    'claim_type': r['CLAIM_TYPE'],
                    'claims': r['claims'],
                    'pct_of_claims': round(r['claims'] / max(total_claims, 1) * 100, 2),
                    'billed': round(r['billed'], 2),
                    'paid': round(r['paid'], 2),
                    'allowed': round(r['allowed'], 2),
                    'mlr': round(r['paid'] / max(r['billed'], 1) * 100, 2) if r['billed'] else 0
                }
                for r in by_claim_type
            ]
        }

        regional_pl = self._query('''
            SELECT
                KP_REGION,
                COUNT(*) as claims,
                SUM(CAST(BILLED_AMOUNT AS REAL)) as billed,
                SUM(CAST(PAID_AMOUNT AS REAL)) as paid,
                SUM(CAST(ALLOWED_AMOUNT AS REAL)) as allowed,
                COUNT(DISTINCT MEMBER_ID) as members
            FROM claims
            GROUP BY KP_REGION
            ORDER BY SUM(CAST(PAID_AMOUNT AS REAL)) DESC
        ''')

        dashboard['sections']['regional_pl'] = {
            'title': 'Financial Performance by Region',
            'data': [
                {
                    'region': r['KP_REGION'],
                    'members': r['members'],
                    'claims': r['claims'],
                    'billed': round(r['billed'], 2),
                    'paid': round(r['paid'], 2),
                    'pmpm': round(r['paid'] / max(r['members'] * 12, 1), 2) if r['members'] else 0,
                    'mlr': round(r['paid'] / max(r['billed'], 1) * 100, 2) if r['billed'] else 0
                }
                for r in regional_pl
            ]
        }

        monthly_trend = self._query('''
            SELECT
                SUBSTR(SERVICE_DATE, 1, 7) as month,
                COUNT(*) as claims,
                SUM(CAST(BILLED_AMOUNT AS REAL)) as billed,
                SUM(CAST(PAID_AMOUNT AS REAL)) as paid
            FROM claims
            WHERE SERVICE_DATE IS NOT NULL AND SERVICE_DATE != ''
            GROUP BY SUBSTR(SERVICE_DATE, 1, 7)
            ORDER BY month DESC
            LIMIT 12
        ''')

        dashboard['sections']['monthly_trend'] = {
            'title': 'Monthly PMPM Trend (Last 12 Months)',
            'data': [
                {
                    'month': r['month'],
                    'pmpm': round(r['paid'] / max(total_members, 1), 2) if r['paid'] else 0,
                    'mlr': round(r['paid'] / max(r['billed'], 1) * 100, 2) if r['billed'] else 0
                }
                for r in monthly_trend
            ]
        }

        dashboard['sections']['denial_savings'] = {
            'title': 'Revenue at Risk (Denials)',
            'metrics': {
                'total_denied': {
                    'value': round(denial_savings, 2),
                    'label': 'Total Denied Amount',
                    'format': 'currency',
                    'insight': 'Direct recovery opportunity via appeals and resubmission'
                },
                'potential_recovery_rate': {
                    'value': '60-70%',
                    'label': 'Typical Appeal Success Rate',
                    'format': 'percent'
                },
                'projected_recovery': {
                    'value': round(denial_savings * 0.65, 2),
                    'label': 'Projected Recovery (65% rate)',
                    'format': 'currency'
                }
            }
        }

        return dashboard

    def get_membership_market_share(self) -> Dict[str, Any]:
        dashboard = {
            'title': 'Membership & Market Share Dashboard',
            'subtitle': 'Growth Drivers, Regional Distribution, Demographics',
            'generated_at': datetime.now().isoformat(),
            'sections': {}
        }

        total_members = self._query_scalar('SELECT COUNT(*) FROM members') or 0

        plan_mix = self._query('''
            SELECT
                PLAN_TYPE,
                COUNT(*) as members,
                SUM(CASE WHEN DISENROLLMENT_DATE IS NOT NULL AND DISENROLLMENT_DATE != ''
                         THEN 1 ELSE 0 END) as disenrolled,
                ROUND(AVG(CAST(RISK_SCORE AS REAL)), 3) as avg_risk
            FROM members
            GROUP BY PLAN_TYPE
            ORDER BY COUNT(*) DESC
        ''')

        dashboard['sections']['plan_mix'] = {
            'title': 'Membership by Plan Type',
            'total_members': total_members,
            'data': [
                {
                    'plan': r['PLAN_TYPE'],
                    'members': r['members'],
                    'pct': round(r['members'] / max(total_members, 1) * 100, 2),
                    'disenrolled': r['disenrolled'] or 0,
                    'avg_risk_score': r['avg_risk'] or 1.0
                }
                for r in plan_mix
            ]
        }

        enrollment_growth = self._query('''
            SELECT
                SUBSTR(ENROLLMENT_DATE, 1, 7) as month,
                COUNT(*) as new_members,
                COUNT(DISTINCT KP_REGION) as active_regions
            FROM members
            WHERE ENROLLMENT_DATE IS NOT NULL AND ENROLLMENT_DATE != ''
            GROUP BY SUBSTR(ENROLLMENT_DATE, 1, 7)
            ORDER BY month DESC
            LIMIT 12
        ''')

        dashboard['sections']['enrollment_growth'] = {
            'title': 'Monthly Enrollment Growth (Last 12 Months)',
            'data': [
                {
                    'month': r['month'],
                    'new_members': r['new_members'],
                    'regions_active': r['active_regions']
                }
                for r in enrollment_growth
            ]
        }

        disenroll_stats = self._query_one('''
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN DISENROLLMENT_DATE IS NOT NULL AND DISENROLLMENT_DATE != ''
                         THEN 1 ELSE 0 END) as disenrolled,
                SUM(CASE WHEN DISENROLLMENT_DATE IS NOT NULL AND DISENROLLMENT_DATE != ''
                         THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as disenroll_rate
            FROM members
        ''')

        if disenroll_stats:
            disenroll_rate = round(disenroll_stats[2], 2) if disenroll_stats[2] else 0
        else:
            disenroll_rate = 0

        dashboard['sections']['disenrollment'] = {
            'title': 'Disenrollment Analysis',
            'metrics': {
                'disenrollment_rate': {
                    'value': disenroll_rate,
                    'label': 'Voluntary Disenrollment Rate',
                    'format': 'percent',
                    'benchmark': '8-12%',
                    'status': 'green' if disenroll_rate < 10 else 'amber' if disenroll_rate < 12 else 'red'
                },
                'retention_rate': {
                    'value': round(100 - disenroll_rate, 2),
                    'label': 'Member Retention Rate',
                    'format': 'percent'
                }
            }
        }

        regional_share = self._query('''
            SELECT
                KP_REGION,
                COUNT(*) as members,
                SUM(CASE WHEN DISENROLLMENT_DATE IS NOT NULL AND DISENROLLMENT_DATE != ''
                         THEN 1 ELSE 0 END) as disenrolled,
                COUNT(DISTINCT PLAN_TYPE) as plan_types
            FROM members
            GROUP BY KP_REGION
            ORDER BY COUNT(*) DESC
        ''')

        dashboard['sections']['market_share_by_region'] = {
            'title': 'Market Share by Region',
            'data': [
                {
                    'region': r['KP_REGION'],
                    'members': r['members'],
                    'pct_of_total': round(r['members'] / max(total_members, 1) * 100, 2),
                    'retention': round((r['members'] - (r['disenrolled'] or 0)) / r['members'] * 100, 2),
                    'plan_diversity': r['plan_types']
                }
                for r in regional_share
            ]
        }

        recent_enrollment = self._query('''
            SELECT
                KP_REGION,
                PLAN_TYPE,
                COUNT(*) as recent_enrollments
            FROM members
            WHERE ENROLLMENT_DATE IS NOT NULL
              AND ENROLLMENT_DATE != ''
              AND SUBSTR(ENROLLMENT_DATE, 1, 4) = '2024'
            GROUP BY KP_REGION, PLAN_TYPE
            ORDER BY COUNT(*) DESC
            LIMIT 10
        ''')

        dashboard['sections']['growth_drivers'] = {
            'title': 'Growth Drivers Framework (2024 Enrollments)',
            'data': [
                {
                    'region': r['KP_REGION'],
                    'plan': r['PLAN_TYPE'],
                    'enrollments': r['recent_enrollments']
                }
                for r in recent_enrollment
            ]
        }

        demographics = self._query('''
            SELECT
                GENDER,
                COUNT(*) as members,
                ROUND(AVG(2024 - CAST(SUBSTR(DATE_OF_BIRTH, 1, 4) AS INTEGER)), 1) as avg_age,
                ROUND(AVG(CAST(RISK_SCORE AS REAL)), 3) as avg_risk
            FROM members
            WHERE GENDER IS NOT NULL AND GENDER != ''
            GROUP BY GENDER
        ''')

        dashboard['sections']['demographics'] = {
            'title': 'Member Demographics',
            'data': [
                {
                    'gender': r['GENDER'],
                    'members': r['members'],
                    'pct': round(r['members'] / max(total_members, 1) * 100, 2),
                    'avg_age': r['avg_age'] or 0,
                    'avg_risk': r['avg_risk'] or 1.0
                }
                for r in demographics
            ]
        }

        return dashboard

    def get_service_utilization(self) -> Dict[str, Any]:
        dashboard = {
            'title': 'Service Utilization Metrics Dashboard',
            'subtitle': 'Encounters/1000, Unit Costs, ED Rates, Pharmacy PMPM',
            'generated_at': datetime.now().isoformat(),
            'sections': {}
        }

        total_members = self._query_scalar('SELECT COUNT(*) FROM members') or 1000

        util_by_type = self._query('''
            SELECT
                VISIT_TYPE,
                COUNT(*) as encounters,
                COUNT(DISTINCT MEMBER_ID) as unique_members,
                SUM(CAST(LENGTH_OF_STAY AS REAL)) as total_los,
                SUM(CAST(COALESCE(
                    (SELECT SUM(CAST(PAID_AMOUNT AS REAL))
                     FROM claims WHERE claims.ENCOUNTER_ID = encounters.ENCOUNTER_ID), 0
                ) AS REAL)) as encounter_cost
            FROM encounters
            GROUP BY VISIT_TYPE
            ORDER BY COUNT(*) DESC
        ''')

        dashboard['sections']['util_per_1000'] = {
            'title': 'Utilization Rates per 1,000 Members',
            'data': [
                {
                    'visit_type': r['VISIT_TYPE'],
                    'encounters': r['encounters'],
                    'per_1000': round(r['encounters'] / total_members * 1000, 1),
                    'unique_members': r['unique_members'],
                    'penetration_rate': round(r['unique_members'] / total_members * 100, 2),
                    'avg_cost': round(r['encounter_cost'] / max(r['encounters'], 1), 2) if r['encounter_cost'] else 0
                }
                for r in util_by_type
            ]
        }

        ed_visits = self._query_one('''
            SELECT
                COUNT(*) as ed_encounters,
                COUNT(DISTINCT MEMBER_ID) as ed_members,
                SUM(CAST(LENGTH_OF_STAY AS REAL)) as total_los
            FROM encounters
            WHERE VISIT_TYPE LIKE '%Emergency%' OR VISIT_TYPE LIKE '%ED%' OR VISIT_TYPE LIKE '%ER%'
        ''')

        if ed_visits:
            ed_rate_per_1000 = round(ed_visits[0] / total_members * 1000, 1)
            ed_penetration = round(ed_visits[1] / total_members * 100, 2)
        else:
            ed_rate_per_1000 = 0
            ed_penetration = 0

        dashboard['sections']['ed_metrics'] = {
            'title': 'Emergency Department Utilization',
            'metrics': {
                'ed_visits_per_1000': {
                    'value': ed_rate_per_1000,
                    'label': 'ED Visits per 1,000 Members',
                    'benchmark': 250.0,
                    'status': 'green' if ed_rate_per_1000 <= 250 else 'amber' if ed_rate_per_1000 <= 350 else 'red'
                },
                'ed_penetration': {
                    'value': ed_penetration,
                    'label': 'Members with ED Visit %',
                    'format': 'percent'
                }
            }
        }

        inpatient = self._query_one('''
            SELECT
                COUNT(*) as inpatient_encounters,
                COUNT(DISTINCT MEMBER_ID) as inpatient_members,
                SUM(CAST(LENGTH_OF_STAY AS REAL)) as total_bed_days
            FROM encounters
            WHERE VISIT_TYPE LIKE '%Inpatient%' OR VISIT_TYPE LIKE '%Hospital%' OR LENGTH_OF_STAY > 0
        ''')

        if inpatient:
            bed_days_per_1000 = round((inpatient[2] or 0) / total_members * 1000, 1)
            inpatient_penetration = round(inpatient[1] / total_members * 100, 2)
        else:
            bed_days_per_1000 = 0
            inpatient_penetration = 0

        dashboard['sections']['inpatient_metrics'] = {
            'title': 'Inpatient Utilization',
            'metrics': {
                'bed_days_per_1000': {
                    'value': bed_days_per_1000,
                    'label': 'Inpatient Bed Days per 1,000',
                    'benchmark': 400.0,
                    'status': 'green' if bed_days_per_1000 <= 400 else 'amber' if bed_days_per_1000 <= 600 else 'red'
                },
                'inpatient_penetration': {
                    'value': inpatient_penetration,
                    'label': 'Members with Inpatient Stay %',
                    'format': 'percent'
                }
            }
        }

        referral_analysis = self._query_one('''
            SELECT
                COUNT(*) as total_referrals,
                COUNT(DISTINCT MEMBER_ID) as referred_members,
                SUM(CASE WHEN STATUS = 'COMPLETED' THEN 1 ELSE 0 END) as completed
            FROM referrals
        ''')

        if referral_analysis:
            ref_per_1000 = round(referral_analysis[0] / total_members * 1000, 1)
            ref_penetration = round(referral_analysis[1] / total_members * 100, 2)
            ref_completion = round(referral_analysis[2] / max(referral_analysis[0], 1) * 100, 2) if referral_analysis[0] else 0
        else:
            ref_per_1000 = 0
            ref_penetration = 0
            ref_completion = 0

        dashboard['sections']['referral_metrics'] = {
            'title': 'Referral Management',
            'metrics': {
                'referrals_per_1000': {
                    'value': ref_per_1000,
                    'label': 'Referrals per 1,000 Members',
                    'benchmark': 150.0,
                    'format': 'number'
                },
                'referral_penetration': {
                    'value': ref_penetration,
                    'label': 'Members Referred %',
                    'format': 'percent'
                },
                'completion_rate': {
                    'value': ref_completion,
                    'label': 'Referral Completion Rate',
                    'format': 'percent',
                    'benchmark': 85.0,
                    'status': 'green' if ref_completion >= 85 else 'amber' if ref_completion >= 75 else 'red'
                }
            }
        }

        pharmacy_pmpm_data = self._query_one('''
            SELECT
                COUNT(*) as rx_count,
                COUNT(DISTINCT MEMBER_ID) as members_on_rx,
                SUM(CAST(COST AS REAL)) as total_cost,
                AVG(CAST(COST AS REAL)) as avg_cost
            FROM prescriptions
        ''')

        if pharmacy_pmpm_data:
            total_rx_cost = pharmacy_pmpm_data[2] or 0
            pharmacy_pmpm = round(total_rx_cost / (total_members * 12), 2)
        else:
            pharmacy_pmpm = 0

        dashboard['sections']['pharmacy_metrics'] = {
            'title': 'Pharmacy Utilization & Cost',
            'metrics': {
                'pharmacy_pmpm': {
                    'value': pharmacy_pmpm,
                    'label': 'Pharmacy PMPM',
                    'format': 'currency',
                    'benchmark': '$80-$150'
                },
                'rx_penetration': {
                    'value': round(pharmacy_pmpm_data[1] / total_members * 100, 2) if pharmacy_pmpm_data else 0,
                    'label': 'Members on Rx %',
                    'format': 'percent'
                }
            }
        }

        regional_util = self._query('''
            SELECT
                e.KP_REGION,
                COUNT(*) as encounters,
                COUNT(DISTINCT e.MEMBER_ID) as members,
                SUM(CAST(e.LENGTH_OF_STAY AS REAL)) as total_los
            FROM encounters e
            GROUP BY e.KP_REGION
            ORDER BY COUNT(*) DESC
        ''')

        dashboard['sections']['regional_utilization'] = {
            'title': 'Utilization by Region',
            'data': [
                {
                    'region': r['KP_REGION'],
                    'encounters': r['encounters'],
                    'per_1000': round(r['encounters'] / total_members * 1000, 1),
                    'unique_members': r['members'],
                    'bed_days': round(r['total_los'], 1) if r['total_los'] else 0
                }
                for r in regional_util
            ]
        }

        return dashboard

    def get_executive_summary(self) -> Dict[str, Any]:
        dashboard = {
            'title': 'Kaiser Permanente Executive Summary',
            'subtitle': 'Performance Scorecard & Strategic Priorities',
            'generated_at': datetime.now().isoformat(),
            'sections': {}
        }

        total_members = self._query_scalar('SELECT COUNT(*) FROM members') or 1

        member_data = self._query_one('''
            SELECT
                COUNT(*) as claims,
                SUM(CAST(BILLED_AMOUNT AS REAL)) as billed,
                SUM(CAST(PAID_AMOUNT AS REAL)) as paid,
                SUM(CAST(ALLOWED_AMOUNT AS REAL)) as allowed,
                SUM(CASE WHEN CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) as denied
            FROM claims
        ''')

        risk_data = self._query_one('''
            SELECT
                AVG(CAST(RISK_SCORE AS REAL)) as avg_risk,
                SUM(CASE WHEN DISENROLLMENT_DATE IS NOT NULL AND DISENROLLMENT_DATE != ''
                         THEN 1 ELSE 0 END) as disenrolled
            FROM members
        ''')

        if member_data:
            billed = member_data[1] or 0
            paid = member_data[2] or 0
            allowed = member_data[3] or 0
            denial_count = member_data[4] or 0
            mlr = (allowed / max(billed, 1) * 100)
            collection = (paid / max(allowed, 1) * 100)
            denial_rate = (denial_count / max(member_data[0], 1) * 100)
        else:
            mlr = 0
            collection = 0
            denial_rate = 0

        if risk_data:
            avg_risk = risk_data[0] or 1.0
            disenrolled = risk_data[1] or 0
        else:
            avg_risk = 1.0
            disenrolled = 0

        disenrollment_rate = (disenrolled / total_members * 100) if total_members else 0

        scorecard_items = [
            {
                'metric': 'Member Retention Rate',
                'value': round(100 - disenrollment_rate, 2),
                'unit': '%',
                'target': 92.0,
                'benchmark': '88-92%',
                'status': 'green' if disenrollment_rate < 10 else 'amber' if disenrollment_rate < 12 else 'red'
            },
            {
                'metric': 'Medical Loss Ratio',
                'value': round(mlr, 2),
                'unit': '%',
                'target': 70.0,
                'benchmark': '65-75% (allowed/billed)',
                'status': 'green' if 60 <= mlr <= 80 else 'amber' if 50 <= mlr <= 85 else 'red'
            },
            {
                'metric': 'Claims Collection Rate',
                'value': round(collection, 2),
                'unit': '%',
                'target': 55.0,
                'benchmark': '50%+ (paid/allowed)',
                'status': 'green' if collection >= 50 else 'amber' if collection >= 40 else 'red'
            },
            {
                'metric': 'Denial Rate',
                'value': round(denial_rate, 2),
                'unit': '%',
                'target': 7.0,
                'benchmark': '5-10%',
                'status': 'green' if denial_rate < 10 else 'amber' if denial_rate < 12 else 'red'
            },
            {
                'metric': 'Average Risk Score',
                'value': round(avg_risk, 3),
                'unit': 'score',
                'target': 2.0,
                'benchmark': '1.5-3.0 (HCC-adjusted MA)',
                'status': 'green' if 1.5 <= avg_risk <= 3.0 else 'amber' if 1.2 <= avg_risk <= 3.5 else 'red'
            }
        ]

        rag_counts = {
            'green': sum(1 for x in scorecard_items if x['status'] == 'green'),
            'amber': sum(1 for x in scorecard_items if x['status'] == 'amber'),
            'red': sum(1 for x in scorecard_items if x['status'] == 'red')
        }

        overall_status = 'green' if rag_counts['red'] == 0 and rag_counts['amber'] <= 1 else 'amber' if rag_counts['red'] == 0 else 'red'

        dashboard['sections']['performance_scorecard'] = {
            'title': 'KP Performance Scorecard',
            'overall_status': overall_status,
            'rag_summary': rag_counts,
            'kpis': scorecard_items
        }

        priorities = []
        if denial_rate > 10:
            priorities.append({
                'priority': 1,
                'area': 'Denial Management',
                'current_state': f'{denial_rate:.1f}% denial rate (target: 7%)',
                'action': 'Implement targeted denial prevention program, enhance pre-auth automation',
                'impact': 'Recovery of $100K-$500K annually'
            })
        if disenrollment_rate > 12:
            priorities.append({
                'priority': 1,
                'area': 'Member Retention',
                'current_state': f'{disenrollment_rate:.1f}% turnover (target: <10%)',
                'action': 'Launch member experience improvement initiative',
                'impact': 'Retain $1M+ in annual revenue per 1000 members'
            })
        if mlr > 80:
            priorities.append({
                'priority': 2,
                'area': 'Medical Cost Management',
                'current_state': f'{mlr:.1f}% MLR (target: <75%)',
                'action': 'Expand high-cost member disease management programs',
                'impact': '1-2% MLR improvement = $500K-$1M annually'
            })
        if avg_risk < 1.5:
            priorities.append({
                'priority': 2,
                'area': 'Risk Score Optimization',
                'current_state': f'{avg_risk:.3f} avg risk (target: 2.0+)',
                'action': 'HCC capture rate improvement via provider education',
                'impact': '0.05 risk increase = $50-$150 PMPM additional revenue'
            })
        if collection < 50:
            priorities.append({
                'priority': 2,
                'area': 'Claims Collections',
                'current_state': f'{collection:.1f}% collection (target: 50%+)',
                'action': 'Accelerate AR follow-up, implement payment term optimization',
                'impact': '1% improvement = $100K-$300K cash recovery'
            })

        dashboard['sections']['strategic_priorities'] = {
            'title': 'Strategic Priorities & Action Items',
            'priorities': priorities if priorities else [
                {
                    'priority': 'routine',
                    'area': 'Performance Maintenance',
                    'current_state': 'All key metrics within target ranges',
                    'action': 'Continue current operational execution with continuous optimization',
                    'impact': 'Sustain market leadership and margin health'
                }
            ]
        }

        dashboard['sections']['highlights'] = {
            'title': 'Performance Highlights',
            'positive_areas': [
                x for x in scorecard_items if x['status'] == 'green'
            ]
        }

        dashboard['sections']['concerns'] = {
            'title': 'Areas Requiring Attention',
            'concerns': [
                x for x in scorecard_items if x['status'] in ['red', 'amber']
            ]
        }

        return dashboard

    def get_full_dashboard(self, region: Optional[str] = None) -> Dict[str, Any]:
        full_dashboard = {
            'title': 'Kaiser Permanente Executive Dashboard Suite',
            'generated_at': datetime.now().isoformat(),
            'region_filter': region,
            'dashboards': {
                'member_experience': self.get_member_experience(),
                'stars_performance': self.get_stars_performance(),
                'risk_adjustment_coding': self.get_risk_adjustment_coding(),
                'financial_performance': self.get_financial_performance(),
                'membership_market_share': self.get_membership_market_share(),
                'service_utilization': self.get_service_utilization(),
                'executive_summary': self.get_executive_summary()
            }
        }

        return full_dashboard

    def to_json(self, obj: Dict[str, Any]) -> str:
        return json.dumps(obj, indent=2, default=str)


_engine = None

def get_engine(db_path: str) -> ExecutiveDashboardEngine:
    global _engine
    if _engine is None or _engine.db_path != db_path:
        _engine = ExecutiveDashboardEngine(db_path)
    return _engine


def generate_dashboard(dashboard_type: str, db_path: str, region: Optional[str] = None) -> Dict[str, Any]:
    engine = get_engine(db_path)

    dashboards = {
        'member_experience': engine.get_member_experience,
        'stars_performance': engine.get_stars_performance,
        'risk_adjustment_coding': engine.get_risk_adjustment_coding,
        'financial_performance': engine.get_financial_performance,
        'membership_market_share': engine.get_membership_market_share,
        'service_utilization': engine.get_service_utilization,
        'executive_summary': engine.get_executive_summary,
        'full': lambda: engine.get_full_dashboard(region=region)
    }

    func = dashboards.get(dashboard_type.lower())
    if not func:
        raise ValueError(f"Unknown dashboard type: {dashboard_type}")

    return func()


if __name__ == '__main__':
    import sys

    db_path = sys.argv[1] if len(sys.argv) > 1 else 'data/healthcare_demo.db'
    dashboard_type = sys.argv[2] if len(sys.argv) > 2 else 'full'

    engine = ExecutiveDashboardEngine(db_path)

    if dashboard_type == 'full':
        dashboard = engine.get_full_dashboard()
    elif dashboard_type == 'member_experience':
        dashboard = engine.get_member_experience()
    elif dashboard_type == 'stars':
        dashboard = engine.get_stars_performance()
    elif dashboard_type == 'rada':
        dashboard = engine.get_risk_adjustment_coding()
    elif dashboard_type == 'financial':
        dashboard = engine.get_financial_performance()
    elif dashboard_type == 'membership':
        dashboard = engine.get_membership_market_share()
    elif dashboard_type == 'utilization':
        dashboard = engine.get_service_utilization()
    elif dashboard_type == 'summary':
        dashboard = engine.get_executive_summary()
    else:
        raise ValueError(f"Unknown dashboard type: {dashboard_type}")

    print(engine.to_json(dashboard))
