import os
import re
import json
import time
import logging
import sqlite3
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import statistics

logger = logging.getLogger('gpdm.deep_analytics')


class DeepHealthcareAnalytics:

    ANNUAL_WORKING_DAYS = 250
    COST_THRESHOLDS = {
        'high_utilizer': 5,
        'high_cost_member': 75000,
        'outlier_percentile': 99
    }

    RISK_TIERS = {
        'Low': (0, 33),
        'Moderate': (33, 50),
        'High': (50, 75),
        'Very High': (75, 100)
    }

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._schema_cache = {}
        self._data_quality_cache = None
        self._last_quality_check = 0

        logger.info("Deep Healthcare Analytics Engine initialized at %s", db_path)

        self._load_schema()

    def _load_schema(self):
        try:
            conn = self._safe_connect()
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [r[0] for r in cursor.fetchall()]

            for table in tables:
                cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
                self._schema_cache[table.upper()] = {
                    c[1].upper(): c[2] for c in cols
                }
            conn.close()
            logger.debug("Schema loaded: %d tables", len(self._schema_cache))
        except Exception as e:
            logger.warning("Schema load failed: %s", e)

    def _safe_connect(self) -> sqlite3.Connection:
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            return conn
        except sqlite3.OperationalError as e:
            logger.error("Database connection failed: %s", e)
            raise

    def _safe_query(self, sql: str, params: Tuple = (), timeout: int = 10) -> List[Dict]:
        conn = None
        try:
            conn = self._safe_connect()
            conn.set_progress_handler(lambda: None, 1000)
            cursor = conn.cursor()

            start = time.time()
            cursor.execute(sql, params)
            results = cursor.fetchall()
            elapsed = time.time() - start

            if elapsed > timeout:
                logger.warning("Query exceeded timeout: %.2fs > %ds", elapsed, timeout)

            logger.debug("Query executed in %.2fs: %s", elapsed, sql[:100])
            return [dict(row) for row in results]

        except sqlite3.OperationalError as e:
            logger.error("Query execution failed: %s\nSQL: %s", e, sql)
            return []
        finally:
            if conn:
                conn.close()

    def run_quality_gate(self) -> Dict:
        logger.info("Running data quality gate")

        completeness = {}
        outliers = {}
        consistency = {}
        freshness = {}
        issues = []

        tables = ['members', 'claims', 'diagnoses', 'encounters', 'prescriptions']

        for table in tables:
            sql = f"SELECT COUNT(*) as cnt FROM {table}"
            result = self._safe_query(sql)
            total = result[0]['cnt'] if result else 0

            if total == 0:
                completeness[table] = 0.0
                issues.append(f"Table {table} is empty")
            else:
                sql_nulls = f"""
                SELECT
                    ROUND(100.0 * COUNT(CASE WHEN MEMBER_ID IS NULL THEN 1 END) / COUNT(*), 1) as null_pct
                FROM {table}
                """
                null_result = self._safe_query(sql_nulls)
                null_pct = null_result[0]['null_pct'] if null_result else 0
                completeness[table] = 100.0 - null_pct

                if null_pct > 10:
                    issues.append(f"Table {table} has {null_pct}% NULL member IDs")

        financial_cols = {
            'claims': ['BILLED_AMOUNT', 'ALLOWED_AMOUNT', 'PAID_AMOUNT'],
            'prescriptions': ['COST'],
        }

        for table, cols in financial_cols.items():
            for col in cols:
                sql = f"""
                SELECT
                    ROUND(AVG(CAST({col} AS REAL)), 2) as avg_val,
                    ROUND(MIN(CAST({col} AS REAL)), 2) as min_val,
                    ROUND(MAX(CAST({col} AS REAL)), 2) as max_val
                FROM {table} WHERE {col} IS NOT NULL
                """
                result = self._safe_query(sql)
                if result:
                    avg_val = result[0]['avg_val'] or 0
                    max_val = result[0]['max_val'] or 0

                    if max_val > 0 and avg_val > 0 and max_val > 10 * avg_val:
                        outlier_count = max_val / avg_val
                        outliers[f"{table}.{col}"] = int(outlier_count)
                        issues.append(
                            f"Outlier in {table}.{col}: max ${max_val:,.0f} "
                            f"({outlier_count:.0f}x mean ${avg_val:,.0f})"
                        )

        sql_consistency = """
        SELECT
            COUNT(*) as violations
        FROM claims
        WHERE CAST(PAID_AMOUNT AS REAL) > CAST(ALLOWED_AMOUNT AS REAL)
            OR CAST(ALLOWED_AMOUNT AS REAL) > CAST(BILLED_AMOUNT AS REAL)
        """
        result = self._safe_query(sql_consistency)
        violations = result[0]['violations'] if result else 0
        consistency['payment_order'] = violations == 0
        if violations > 0:
            issues.append(f"Payment order violations: {violations} claims with PAID > ALLOWED or ALLOWED > BILLED")

        sql_dates = """
        SELECT COUNT(*) as violations
        FROM encounters
        WHERE SERVICE_DATE IS NOT NULL AND LENGTH_OF_STAY < 0
        """
        result = self._safe_query(sql_dates)
        violations = result[0]['violations'] if result else 0
        consistency['length_of_stay'] = violations == 0
        if violations > 0:
            issues.append(f"Date logic violations: {violations} encounters with negative length of stay")

        sql_freshness = """
        SELECT
            MAX(SERVICE_DATE) as latest_service,
            JULIANDAY('now') - JULIANDAY(MAX(SERVICE_DATE)) as days_old
        FROM claims
        """
        result = self._safe_query(sql_freshness)
        if result and result[0]['days_old']:
            days_old = result[0]['days_old']
            freshness['claims'] = days_old
            if days_old > 90:
                issues.append(f"Data freshness concern: latest claims data is {days_old:.0f} days old")

        completeness_avg = statistics.mean(completeness.values()) if completeness else 0
        consistency_avg = statistics.mean([1.0 if v else 0.0 for v in consistency.values()]) if consistency else 0.5

        overall_confidence = (completeness_avg * 0.6 + consistency_avg * 100 * 0.3 - len(issues) * 5) * 0.9 + 10
        overall_confidence = max(0, min(100, overall_confidence))

        narrative = f"Data quality gate completed. {len(completeness)} tables evaluated with "
        narrative += f"average completeness {completeness_avg:.0f}%. "
        if issues:
            narrative += f"Identified {len(issues)} data quality concerns requiring attention. "
        else:
            narrative += "No critical quality issues detected. "
        narrative += f"Overall confidence score: {overall_confidence:.0f}/100."

        recommendations = []
        if len(issues) > 0:
            recommendations.append({
                'action': 'Address data quality issues',
                'impact': f"Reduce analytics confidence by ~{100 - overall_confidence:.0f}%",
                'priority': 'HIGH'
            })

        return {
            'data': {
                'completeness': completeness,
                'outliers': outliers,
                'consistency': consistency,
                'freshness': freshness,
            },
            'narrative': narrative,
            'recommendations': recommendations,
            'confidence': overall_confidence,
            'issues': issues,
        }

    def analyze_demographics(self) -> Dict:
        logger.info("Analyzing demographics")

        data = {
            'age_cohorts': {},
            'gender_analysis': {},
            'race_ethnicity': {},
            'regional': {},
            'plan_type': {},
        }

        age_cohorts = [
            ("Under 18", "WHERE CAST((JULIANDAY('now') - JULIANDAY(DATE_OF_BIRTH))/365.25 AS INT) < 18"),
            ("18-34", "WHERE CAST((JULIANDAY('now') - JULIANDAY(DATE_OF_BIRTH))/365.25 AS INT) BETWEEN 18 AND 34"),
            ("35-49", "WHERE CAST((JULIANDAY('now') - JULIANDAY(DATE_OF_BIRTH))/365.25 AS INT) BETWEEN 35 AND 49"),
            ("50-64", "WHERE CAST((JULIANDAY('now') - JULIANDAY(DATE_OF_BIRTH))/365.25 AS INT) BETWEEN 50 AND 64"),
            ("65+", "WHERE CAST((JULIANDAY('now') - JULIANDAY(DATE_OF_BIRTH))/365.25 AS INT) >= 65"),
        ]

        for cohort_name, cohort_where in age_cohorts:
            sql = f"""
            SELECT
                COUNT(DISTINCT m.MEMBER_ID) as member_count,
                ROUND(AVG(CAST(COALESCE(c.PAID_AMOUNT, 0) AS REAL)), 2) as avg_cost_per_member,
                ROUND(SUM(CAST(COALESCE(c.PAID_AMOUNT, 0) AS REAL)), 2) as total_cost,
                COUNT(DISTINCT c.CLAIM_ID) as claim_count,
                ROUND(AVG(m.RISK_SCORE), 1) as avg_risk_score
            FROM members m
            LEFT JOIN claims c ON m.MEMBER_ID = c.MEMBER_ID
            {cohort_where}
            """
            result = self._safe_query(sql)
            if result:
                r = result[0]
                data['age_cohorts'][cohort_name] = {
                    'member_count': r['member_count'] or 0,
                    'avg_cost_per_member': r['avg_cost_per_member'] or 0,
                    'total_cost': r['total_cost'] or 0,
                    'claim_count': r['claim_count'] or 0,
                    'avg_risk_score': r['avg_risk_score'] or 0,
                }

        sql_gender = """
        SELECT
            GENDER,
            COUNT(DISTINCT m.MEMBER_ID) as member_count,
            ROUND(AVG(CAST(COALESCE(c.PAID_AMOUNT, 0) AS REAL)), 2) as avg_cost,
            ROUND(AVG(m.RISK_SCORE), 1) as avg_risk,
            COUNT(DISTINCT CASE WHEN VISIT_TYPE = 'ER' THEN e.MEMBER_ID END) as er_visits
        FROM members m
        LEFT JOIN claims c ON m.MEMBER_ID = c.MEMBER_ID
        LEFT JOIN encounters e ON m.MEMBER_ID = e.MEMBER_ID
        WHERE GENDER IS NOT NULL
        GROUP BY GENDER
        """
        result = self._safe_query(sql_gender)
        for r in result:
            gender = r['GENDER'] or 'Unknown'
            data['gender_analysis'][gender] = {
                'member_count': r['member_count'] or 0,
                'avg_cost': r['avg_cost'] or 0,
                'avg_risk': r['avg_risk'] or 0,
                'er_visits': r['er_visits'] or 0,
            }

        sql_race = """
        SELECT
            RACE,
            COUNT(DISTINCT m.MEMBER_ID) as member_count,
            ROUND(AVG(CAST(COALESCE(c.PAID_AMOUNT, 0) AS REAL)), 2) as avg_cost,
            ROUND(100.0 * SUM(CASE WHEN c.CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) /
                  COUNT(DISTINCT c.CLAIM_ID), 1) as denial_rate_pct
        FROM members m
        LEFT JOIN claims c ON m.MEMBER_ID = c.MEMBER_ID
        WHERE RACE IS NOT NULL
        GROUP BY RACE
        """
        result = self._safe_query(sql_race)
        for r in result:
            race = r['RACE'] or 'Unknown'
            data['race_ethnicity'][race] = {
                'member_count': r['member_count'] or 0,
                'avg_cost': r['avg_cost'] or 0,
                'denial_rate_pct': r['denial_rate_pct'] or 0,
            }

        sql_region = """
        SELECT
            m.KP_REGION,
            COUNT(DISTINCT m.MEMBER_ID) as member_count,
            ROUND(AVG(CAST(COALESCE(c.PAID_AMOUNT, 0) AS REAL)), 2) as pmpm,
            ROUND(SUM(CAST(COALESCE(c.PAID_AMOUNT, 0) AS REAL)), 2) as total_cost,
            ROUND(AVG(CAST(m.RISK_SCORE AS REAL)), 1) as avg_risk_score
        FROM members m
        LEFT JOIN claims c ON m.MEMBER_ID = c.MEMBER_ID
        WHERE m.KP_REGION IS NOT NULL
        GROUP BY m.KP_REGION
        ORDER BY total_cost DESC
        """
        result = self._safe_query(sql_region)
        for r in result:
            region = r['KP_REGION'] or 'Unknown'
            data['regional'][region] = {
                'member_count': r['member_count'] or 0,
                'pmpm': r['pmpm'] or 0,
                'total_cost': r['total_cost'] or 0,
                'avg_risk_score': r['avg_risk_score'] or 0,
            }

        sql_plan = """
        SELECT
            m.PLAN_TYPE,
            COUNT(DISTINCT m.MEMBER_ID) as member_count,
            ROUND(AVG(CAST(COALESCE(c.PAID_AMOUNT, 0) AS REAL)), 2) as pmpm,
            ROUND(AVG(CAST(COALESCE(c.BILLED_AMOUNT, 0) AS REAL)), 2) as avg_billed,
            ROUND(100.0 * SUM(CASE WHEN c.CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END) /
                  NULLIF(COUNT(DISTINCT c.CLAIM_ID), 0), 1) as denial_rate_pct
        FROM members m
        LEFT JOIN claims c ON m.MEMBER_ID = c.MEMBER_ID
        WHERE m.PLAN_TYPE IS NOT NULL
        GROUP BY m.PLAN_TYPE
        """
        result = self._safe_query(sql_plan)
        for r in result:
            plan_type = r['PLAN_TYPE'] or 'Unknown'
            data['plan_type'][plan_type] = {
                'member_count': r['member_count'] or 0,
                'pmpm': r['pmpm'] or 0,
                'avg_billed': r['avg_billed'] or 0,
                'denial_rate_pct': r['denial_rate_pct'] or 0,
            }

        total_members = sum(ac.get('member_count', 0) for ac in data['age_cohorts'].values())
        narrative = f"Demographic analysis across {total_members:,} members. "

        if data['age_cohorts']:
            highest_cost_cohort = max(
                data['age_cohorts'].items(),
                key=lambda x: x[1].get('avg_cost_per_member', 0)
            )
            narrative += f"Highest-cost age group: {highest_cost_cohort[0]} "
            narrative += f"at ${highest_cost_cohort[1]['avg_cost_per_member']:,.0f} PMPM. "

        if data['regional']:
            highest_cost_region = max(
                data['regional'].items(),
                key=lambda x: x[1].get('total_cost', 0)
            )
            narrative += f"Highest-cost region: {highest_cost_region[0]} "
            narrative += f"(${highest_cost_region[1]['total_cost']:,.0f} total). "

        recommendations = []

        denial_rates = [r.get('denial_rate_pct', 0) for r in data['race_ethnicity'].values()]
        if denial_rates:
            max_denial = max(denial_rates)
            min_denial = min(denial_rates)
            if max_denial > min_denial * 1.5:
                recommendations.append({
                    'action': 'Address health equity gaps in claim denial rates',
                    'impact': f"Potential ${(max_denial - min_denial) * 100000 * 0.05:,.0f} in recoverable denials",
                    'priority': 'HIGH'
                })

        return {
            'data': data,
            'narrative': narrative,
            'recommendations': recommendations,
            'confidence': 85,
        }

    def analyze_cost_drivers(self) -> Dict:
        logger.info("Analyzing cost drivers")

        data = {
            'top_conditions': [],
            'top_cost_members': [],
            'cost_by_risk_tier': {},
            'quarterly_trend': {},
            'revenue_leakage': {},
            'pmpm_by_segment': {},
        }

        sql_conditions = """
        SELECT
            d.ICD10_CODE,
            d.ICD10_DESCRIPTION,
            d.HCC_CATEGORY,
            COUNT(DISTINCT d.MEMBER_ID) as affected_members,
            ROUND(SUM(CAST(COALESCE(c.PAID_AMOUNT, 0) AS REAL)), 2) as total_cost,
            ROUND(AVG(CAST(COALESCE(c.PAID_AMOUNT, 0) AS REAL)), 2) as avg_claim_cost
        FROM diagnoses d
        LEFT JOIN claims c ON d.MEMBER_ID = c.MEMBER_ID
        WHERE d.ICD10_CODE IS NOT NULL
        GROUP BY d.ICD10_CODE, d.ICD10_DESCRIPTION, d.HCC_CATEGORY
        ORDER BY total_cost DESC
        LIMIT 10
        """
        result = self._safe_query(sql_conditions)
        for r in result:
            data['top_conditions'].append({
                'icd10_code': r['ICD10_CODE'],
                'description': r['ICD10_DESCRIPTION'],
                'hcc_category': r['HCC_CATEGORY'],
                'affected_members': r['affected_members'] or 0,
                'total_cost': r['total_cost'] or 0,
                'avg_claim_cost': r['avg_claim_cost'] or 0,
            })

        sql_members = """
        SELECT
            m.MEMBER_ID,
            ROUND(AVG(m.RISK_SCORE), 1) as risk_score,
            m.PLAN_TYPE,
            m.KP_REGION,
            COUNT(DISTINCT d.DIAGNOSIS_DATE) as chronic_condition_count,
            COUNT(DISTINCT c.CLAIM_ID) as claim_count,
            ROUND(SUM(CAST(COALESCE(c.PAID_AMOUNT, 0) AS REAL)), 2) as total_cost
        FROM members m
        LEFT JOIN claims c ON m.MEMBER_ID = c.MEMBER_ID
        LEFT JOIN diagnoses d ON m.MEMBER_ID = d.MEMBER_ID
        GROUP BY m.MEMBER_ID
        ORDER BY total_cost DESC
        LIMIT 10
        """
        result = self._safe_query(sql_members)
        for r in result:
            data['top_cost_members'].append({
                'member_id': f"MBR_{r['MEMBER_ID'][-6:] if r['MEMBER_ID'] else 'UNKNOWN'[:6]}",
                'risk_score': r['risk_score'] or 0,
                'plan_type': r['PLAN_TYPE'],
                'region': r['KP_REGION'],
                'chronic_conditions': r['chronic_condition_count'] or 0,
                'claim_count': r['claim_count'] or 0,
                'total_cost': r['total_cost'] or 0,
            })

        sql_risk = """
        SELECT
            CASE
                WHEN RISK_SCORE >= 75 THEN 'Very High'
                WHEN RISK_SCORE >= 50 THEN 'High'
                WHEN RISK_SCORE >= 33 THEN 'Moderate'
                ELSE 'Low'
            END as risk_tier,
            COUNT(DISTINCT m.MEMBER_ID) as member_count,
            ROUND(AVG(CAST(COALESCE(c.PAID_AMOUNT, 0) AS REAL)), 2) as pmpm,
            ROUND(SUM(CAST(COALESCE(c.PAID_AMOUNT, 0) AS REAL)), 2) as total_cost
        FROM members m
        LEFT JOIN claims c ON m.MEMBER_ID = c.MEMBER_ID
        GROUP BY risk_tier
        """
        result = self._safe_query(sql_risk)
        for r in result:
            risk_tier = r['risk_tier'] or 'Unknown'
            data['cost_by_risk_tier'][risk_tier] = {
                'member_count': r['member_count'] or 0,
                'pmpm': r['pmpm'] or 0,
                'total_cost': r['total_cost'] or 0,
            }

        sql_quarterly = """
        SELECT
            SUBSTR(SERVICE_DATE, 1, 7) as year_month,
            ROUND(SUM(CAST(COALESCE(PAID_AMOUNT, 0) AS REAL)), 2) as monthly_spend,
            COUNT(DISTINCT MEMBER_ID) as active_members
        FROM claims
        WHERE SERVICE_DATE IS NOT NULL
        GROUP BY SUBSTR(SERVICE_DATE, 1, 7)
        ORDER BY year_month DESC
        LIMIT 12
        """
        result = self._safe_query(sql_quarterly)
        for r in result:
            month = r['year_month'] or 'Unknown'
            data['quarterly_trend'][month] = {
                'monthly_spend': r['monthly_spend'] or 0,
                'active_members': r['active_members'] or 0,
            }

        sql_denials = """
        SELECT
            CLAIM_STATUS,
            DENIAL_REASON,
            COUNT(*) as denial_count,
            ROUND(SUM(CAST(COALESCE(BILLED_AMOUNT, 0) AS REAL)), 2) as denied_amount,
            ROUND(AVG(CAST(COALESCE(BILLED_AMOUNT, 0) AS REAL)), 2) as avg_denied_claim
        FROM claims
        WHERE CLAIM_STATUS = 'DENIED'
        GROUP BY CLAIM_STATUS, DENIAL_REASON
        ORDER BY denied_amount DESC
        """
        result = self._safe_query(sql_denials)
        total_denied = 0
        for r in result:
            reason = r['DENIAL_REASON'] or 'Unknown'
            denied_amt = r['denied_amount'] or 0
            total_denied += denied_amt
            data['revenue_leakage'][reason] = {
                'denial_count': r['denial_count'] or 0,
                'denied_amount': denied_amt,
                'avg_denied_claim': r['avg_denied_claim'] or 0,
            }

        sql_pmpm = """
        SELECT
            m.PLAN_TYPE,
            CASE
                WHEN CAST((JULIANDAY('now') - JULIANDAY(m.DATE_OF_BIRTH))/365.25 AS INT) < 35 THEN 'Under 35'
                WHEN CAST((JULIANDAY('now') - JULIANDAY(m.DATE_OF_BIRTH))/365.25 AS INT) < 50 THEN '35-49'
                WHEN CAST((JULIANDAY('now') - JULIANDAY(m.DATE_OF_BIRTH))/365.25 AS INT) < 65 THEN '50-64'
                ELSE '65+'
            END as age_group,
            COUNT(DISTINCT m.MEMBER_ID) as member_count,
            ROUND(AVG(CAST(COALESCE(c.PAID_AMOUNT, 0) AS REAL)), 2) as pmpm
        FROM members m
        LEFT JOIN claims c ON m.MEMBER_ID = c.MEMBER_ID
        WHERE m.PLAN_TYPE IS NOT NULL
        GROUP BY m.PLAN_TYPE, age_group
        """
        result = self._safe_query(sql_pmpm)
        for r in result:
            key = f"{r['PLAN_TYPE']}_{r['age_group']}"
            data['pmpm_by_segment'][key] = {
                'member_count': r['member_count'] or 0,
                'pmpm': r['pmpm'] or 0,
            }

        total_cost = sum(tc.get('total_cost', 0) for tc in data['cost_by_risk_tier'].values())
        narrative = f"Cost driver analysis across ${total_cost:,.0f} in claims. "

        if data['top_conditions']:
            top_cond = data['top_conditions'][0]
            narrative += f"Top condition: {top_cond['description']} accounts for ${top_cond['total_cost']:,.0f} "
            narrative += f"({top_cond['affected_members']:,} members). "

        narrative += f"Revenue leakage: ${total_denied:,.0f} in denied claims. "

        very_high = data['cost_by_risk_tier'].get('Very High', {})
        if very_high.get('member_count', 0) > 0:
            very_high_pct = 100 * very_high.get('total_cost', 0) / total_cost if total_cost > 0 else 0
            narrative += f"Very High risk members ({very_high.get('member_count', 0)}) drive {very_high_pct:.0f}% of costs."

        recommendations = []

        if total_denied > 0:
            recovery_potential = total_denied * 0.20
            recommendations.append({
                'action': 'Implement denial appeal program',
                'impact': f"Recover approximately ${recovery_potential:,.0f} in denied claims",
                'priority': 'HIGH'
            })

        if data['top_conditions']:
            savings_potential = data['top_conditions'][0]['total_cost'] * 0.10
            recommendations.append({
                'action': f"Launch care management for {data['top_conditions'][0]['description']}",
                'impact': f"Potential annual savings of ${savings_potential:,.0f}",
                'priority': 'HIGH'
            })

        return {
            'data': data,
            'narrative': narrative,
            'recommendations': recommendations,
            'confidence': 80,
        }

    def analyze_preventive_opportunities(self) -> Dict:
        logger.info("Analyzing preventive opportunities")

        data = {
            'at_risk_members': [],
            'cost_of_inaction': {},
            'intervention_roi': [],
            'care_gaps': {},
            'high_utilizers': [],
        }

        pre_chronic_patterns = [
            ('E66%', 'Obesity/Metabolic Syndrome'),
            ('E78%', 'Hyperlipidemia → CVD Risk'),
            ('F41%', 'Anxiety → Chronic Mental Health'),
            ('F32%', 'Depression → Chronic Mental Health'),
            ('F43%', 'Stress/PTSD → Chronic Mental Health'),
            ('R73%', 'Pre-diabetes'),
            ('R03%', 'Elevated BP → Hypertension'),
            ('J06%', 'Upper Respiratory → COPD Risk'),
            ('N17%', 'Acute Kidney → CKD Risk'),
            ('R80%', 'Proteinuria → CKD Risk'),
        ]
        like_clauses = " OR ".join([f"d.ICD10_CODE LIKE '{p[0]}'" for p in pre_chronic_patterns])

        sql_at_risk = f"""
        SELECT
            d.MEMBER_ID,
            COUNT(d.DIAGNOSIS_DATE) as pre_chronic_count,
            GROUP_CONCAT(d.ICD10_CODE, ', ') as icd10_codes,
            ROUND(AVG(CAST(m.RISK_SCORE AS REAL)), 2) as risk_score,
            CAST(m.CHRONIC_CONDITIONS AS INT) as chronic_count,
            ROUND(SUM(CAST(COALESCE(c.PAID_AMOUNT, 0) AS REAL)), 2) as current_cost
        FROM diagnoses d
        JOIN members m ON d.MEMBER_ID = m.MEMBER_ID
        LEFT JOIN claims c ON d.MEMBER_ID = c.MEMBER_ID
        WHERE ({like_clauses})
          AND d.IS_CHRONIC = 'N'
        GROUP BY d.MEMBER_ID
        ORDER BY risk_score DESC, current_cost DESC
        LIMIT 200
        """
        result = self._safe_query(sql_at_risk)

        sql_risk_members = """
        SELECT COUNT(*) as cnt,
               ROUND(AVG(CAST(RISK_SCORE AS REAL)), 2) as avg_risk,
               ROUND(SUM(CAST(RISK_SCORE AS REAL)), 2) as total_risk
        FROM members
        WHERE CAST(CHRONIC_CONDITIONS AS INT) = 0
          AND CAST(RISK_SCORE AS REAL) >= 0.5
        """
        risk_result = self._safe_query(sql_risk_members)
        non_chronic_at_risk = risk_result[0]['cnt'] if risk_result else 0

        at_risk_count = len(result) if result else non_chronic_at_risk
        at_risk_cost = 0
        for r in result:
            cost_val = r['current_cost'] or 0
            at_risk_cost += cost_val
            data['at_risk_members'].append({
                'member_id': f"MBR_{str(r['MEMBER_ID'] or '')[-6:]}",
                'pre_chronic_count': r['pre_chronic_count'] or 0,
                'risk_score': r['risk_score'] or 0,
                'chronic_conditions': r['chronic_count'] or 0,
                'icd10_codes': r['icd10_codes'] or '',
                'current_annual_cost': cost_val,
            })
        if non_chronic_at_risk > at_risk_count:
            at_risk_count = non_chronic_at_risk

        data['at_risk_count'] = at_risk_count

        conversion_rate = 0.15
        cost_multiplier = 1.5

        data['cost_of_inaction'] = {
            'at_risk_member_count': at_risk_count,
            'current_annual_cost': at_risk_cost,
            'expected_conversions_yr1': int(at_risk_count * conversion_rate),
            'projected_cost_increase': at_risk_cost * conversion_rate * (cost_multiplier - 1),
            'cost_per_conversion': (at_risk_cost * conversion_rate * (cost_multiplier - 1)) / (at_risk_count * conversion_rate) if (at_risk_count * conversion_rate) > 0 else 0,
        }

        interventions = [
            {
                'name': 'Diabetes Prevention Program',
                'cost_per_member': 500,
                'target_members': int(at_risk_count * 0.3),
                'effectiveness': 0.40,
                'annual_savings_per_prevented': 8000,
            },
            {
                'name': 'Hypertension Management',
                'cost_per_member': 300,
                'target_members': int(at_risk_count * 0.25),
                'effectiveness': 0.35,
                'annual_savings_per_prevented': 6000,
            },
            {
                'name': 'Mental Health Screening & Support',
                'cost_per_member': 400,
                'target_members': int(at_risk_count * 0.20),
                'effectiveness': 0.30,
                'annual_savings_per_prevented': 5000,
            },
        ]

        for intervention in interventions:
            program_cost = intervention['cost_per_member'] * intervention['target_members']
            prevented = intervention['target_members'] * conversion_rate * intervention['effectiveness']
            annual_savings = prevented * intervention['annual_savings_per_prevented']
            roi = (annual_savings - program_cost) / program_cost if program_cost > 0 else 0

            data['intervention_roi'].append({
                'intervention': intervention['name'],
                'program_cost': program_cost,
                'target_members': intervention['target_members'],
                'prevented_conversions': prevented,
                'annual_savings': annual_savings,
                'roi': roi,
                'payback_months': (program_cost / (annual_savings / 12)) if annual_savings > 0 else 0,
            })

        sql_gaps = """
        SELECT COUNT(*) as members_without_visit
        FROM members m
        WHERE m.MEMBER_ID NOT IN (
            SELECT DISTINCT e.MEMBER_ID FROM encounters e
            WHERE UPPER(e.VISIT_TYPE) LIKE '%OUTPATIENT%'
              AND e.SERVICE_DATE >= date('now', '-12 months')
        )
        """
        result = self._safe_query(sql_gaps)
        care_gap_count = result[0]['members_without_visit'] if result else 0
        data['care_gaps'] = {'no_pcp_visit_12m': {'member_count': care_gap_count}}
        data['care_gap_count'] = care_gap_count

        sql_high_util = """
        SELECT
            e.MEMBER_ID,
            COUNT(*) as er_visit_count,
            ROUND(CAST(m.RISK_SCORE AS REAL), 2) as risk_score,
            ROUND(SUM(CAST(COALESCE(c.PAID_AMOUNT, 0) AS REAL)), 2) as annual_cost
        FROM encounters e
        JOIN members m ON e.MEMBER_ID = m.MEMBER_ID
        LEFT JOIN claims c ON e.MEMBER_ID = c.MEMBER_ID
        WHERE UPPER(e.VISIT_TYPE) LIKE '%EMERGENCY%'
        GROUP BY e.MEMBER_ID
        HAVING COUNT(*) >= 3
        ORDER BY annual_cost DESC
        LIMIT 50
        """
        result = self._safe_query(sql_high_util)
        for r in result:
            data['high_utilizers'].append({
                'member_id': f"MBR_{str(r['MEMBER_ID'] or '')[-6:]}",
                'er_visits': r['er_visit_count'] or 0,
                'risk_score': r['risk_score'] or 0,
                'annual_cost': r['annual_cost'] or 0,
            })
        data['high_utilizer_count'] = len(data['high_utilizers'])

        narrative = f"Preventive opportunity analysis identified {at_risk_count:,} members at risk of chronic conversion. "

        if data['cost_of_inaction']['expected_conversions_yr1'] > 0:
            narrative += f"Without intervention, expect {data['cost_of_inaction']['expected_conversions_yr1']:,} conversions "
            narrative += f"costing approximately ${data['cost_of_inaction']['projected_cost_increase']:,.0f} annually. "

        if data['intervention_roi']:
            best_roi = max(data['intervention_roi'], key=lambda x: x['roi'])
            narrative += f"Best ROI opportunity: {best_roi['intervention']} "
            narrative += f"({best_roi['roi']:.1%} return on ${best_roi['program_cost']:,.0f} investment). "

        narrative += f"Identified {len(data['high_utilizers']):,} high utilizers (5+ ER visits annually)."

        recommendations = []

        if data['intervention_roi']:
            best = max(data['intervention_roi'], key=lambda x: x['annual_savings'])
            recommendations.append({
                'action': f"Implement {best['intervention']}",
                'impact': f"Annual savings of ${best['annual_savings']:,.0f} with {best['roi']:.0%} ROI",
                'priority': 'HIGH'
            })

        if len(data['high_utilizers']) > 10:
            recommendations.append({
                'action': 'Launch care management for high-utilizer program',
                'impact': f"Target {len(data['high_utilizers'])} high utilizers to reduce ER visits by 30%",
                'priority': 'HIGH'
            })

        return {
            'data': data,
            'narrative': narrative,
            'recommendations': recommendations,
            'confidence': 75,
        }

    def get_financial_initiatives(self) -> Dict:
        logger.info("Generating financial initiatives")

        data = {
            'denial_reduction': {},
            'network_leakage': {},
            'outlier_management': {},
            'preventive_roi_summary': {},
            'revenue_cycle': {},
        }

        sql_denials = """
        SELECT
            DENIAL_REASON,
            COUNT(*) as denial_count,
            ROUND(SUM(CAST(COALESCE(BILLED_AMOUNT, 0) AS REAL)), 2) as total_denied,
            ROUND(AVG(CAST(COALESCE(BILLED_AMOUNT, 0) AS REAL)), 2) as avg_claim_value
        FROM claims
        WHERE CLAIM_STATUS = 'DENIED'
        GROUP BY DENIAL_REASON
        ORDER BY total_denied DESC
        LIMIT 5
        """
        result = self._safe_query(sql_denials)
        total_denial_value = 0
        for r in result:
            reason = r['DENIAL_REASON'] or 'Unknown'
            denied_amt = r['total_denied'] or 0
            total_denial_value += denied_amt

            recovery_rate = 0.30
            data['denial_reduction'][reason] = {
                'denial_count': r['denial_count'] or 0,
                'total_denied': denied_amt,
                'recovery_potential': denied_amt * recovery_rate,
                'recovery_rate': recovery_rate,
            }

        sql_network = """
        SELECT
            COUNT(*) as oon_claim_count,
            ROUND(SUM(CAST(COALESCE(BILLED_AMOUNT, 0) AS REAL)), 2) as oon_billed,
            ROUND(SUM(CAST(COALESCE(ALLOWED_AMOUNT, 0) AS REAL)), 2) as oon_allowed,
            ROUND(SUM(CAST(COALESCE(PAID_AMOUNT, 0) AS REAL)), 2) as oon_paid
        FROM claims
        WHERE CLAIM_TYPE = 'OUT_OF_NETWORK'
        """
        result = self._safe_query(sql_network)
        if result and result[0]['oon_claim_count']:
            r = result[0]
            oon_count = r['oon_claim_count'] or 0
            oon_paid = r['oon_paid'] or 0

            sql_ion = """
            SELECT
                ROUND(AVG(CAST(COALESCE(PAID_AMOUNT, 0) AS REAL)), 2) as ion_avg
            FROM claims
            WHERE CLAIM_TYPE = 'IN_NETWORK'
            """
            ion_result = self._safe_query(sql_ion)
            ion_avg = ion_result[0]['ion_avg'] if ion_result else 0

            data['network_leakage'] = {
                'oon_claim_count': oon_count,
                'oon_total_paid': oon_paid,
                'oon_avg_claim': oon_paid / oon_count if oon_count > 0 else 0,
                'ion_avg_claim': ion_avg,
                'premium_over_in_network_pct': ((oon_paid / oon_count) - ion_avg) / ion_avg * 100 if ion_avg > 0 else 0,
                'annual_leakage': (oon_paid / oon_count - ion_avg) * oon_count if ion_avg > 0 else 0,
            }

        sql_outliers = """
        SELECT
            COUNT(*) as outlier_claim_count,
            ROUND(SUM(CAST(COALESCE(PAID_AMOUNT, 0) AS REAL)), 2) as outlier_total,
            ROUND(AVG(CAST(COALESCE(PAID_AMOUNT, 0) AS REAL)), 2) as outlier_avg
        FROM claims
        WHERE CAST(COALESCE(PAID_AMOUNT, 0) AS REAL) > (
            SELECT
                ROUND(AVG(CAST(COALESCE(PAID_AMOUNT, 0) AS REAL)) +
                      3 * SQRT(AVG(CAST(COALESCE(PAID_AMOUNT, 0) AS REAL) *
                                    CAST(COALESCE(PAID_AMOUNT, 0) AS REAL))), 2)
            FROM claims
        )
        """
        result = self._safe_query(sql_outliers)
        if result:
            r = result[0]
            data['outlier_management'] = {
                'outlier_claim_count': r['outlier_claim_count'] or 0,
                'outlier_total_paid': r['outlier_total'] or 0,
                'outlier_avg_claim': r['outlier_avg'] or 0,
                'annual_savings_from_10pct_reduction': (r['outlier_total'] or 0) * 0.10,
            }

        preventive_analysis = self.analyze_preventive_opportunities()
        if preventive_analysis['data']['intervention_roi']:
            best_intervention = max(
                preventive_analysis['data']['intervention_roi'],
                key=lambda x: x['annual_savings']
            )
            data['preventive_roi_summary'] = {
                'top_intervention': best_intervention['intervention'],
                'program_cost': best_intervention['program_cost'],
                'annual_savings': best_intervention['annual_savings'],
                'roi': best_intervention['roi'],
                'payback_months': best_intervention['payback_months'],
            }

        sql_aging = """
        SELECT
            COUNT(*) as total_claims,
            ROUND(SUM(CAST(COALESCE(PAID_AMOUNT, 0) AS REAL)), 2) as total_paid,
            ROUND(AVG(JULIANDAY('now') - JULIANDAY(SERVICE_DATE)), 1) as avg_days_to_payment
        FROM claims
        WHERE CLAIM_STATUS = 'PAID'
        """
        result = self._safe_query(sql_aging)
        if result:
            r = result[0]
            avg_days = r['avg_days_to_payment'] or 0
            total_paid = r['total_paid'] or 0

            target_days = 30
            improvement_days = avg_days - target_days if avg_days > target_days else 0
            daily_cash_flow = total_paid / 365 if avg_days > 0 else 0
            cash_improvement = daily_cash_flow * improvement_days

            data['revenue_cycle'] = {
                'total_claims': r['total_claims'] or 0,
                'total_paid': total_paid,
                'avg_days_to_payment': avg_days,
                'target_days': target_days,
                'improvement_potential_days': improvement_days,
                'cash_flow_improvement_potential': cash_improvement,
            }

        narrative = "Financial optimization analysis identified multiple revenue leakage opportunities. "

        if total_denial_value > 0:
            recovery = total_denial_value * 0.30
            narrative += f"Denial recovery potential: ${recovery:,.0f} from ${total_denial_value:,.0f} in denied claims. "

        if data['network_leakage'].get('annual_leakage', 0) > 0:
            narrative += f"Network leakage: ${data['network_leakage']['annual_leakage']:,.0f} annual overspend "
            narrative += f"({data['network_leakage']['premium_over_in_network_pct']:.0f}% premium over in-network). "

        if data['outlier_management'].get('outlier_claim_count', 0) > 0:
            savings = data['outlier_management'].get('annual_savings_from_10pct_reduction', 0)
            narrative += f"High-cost outliers ({data['outlier_management']['outlier_claim_count']:,} claims): "
            narrative += f"10% reduction saves ~${savings:,.0f} annually."

        recommendations = []

        recommendations.append({
            'action': 'Launch denial appeal program targeting top reasons',
            'impact': f"Recover ${total_denial_value * 0.30:,.0f} annually",
            'priority': 'HIGH'
        })

        if data['network_leakage'].get('annual_leakage', 0) > 100000:
            recommendations.append({
                'action': 'Optimize network utilization and referral management',
                'impact': f"Reduce network leakage by ${data['network_leakage']['annual_leakage'] * 0.5:,.0f}",
                'priority': 'HIGH'
            })

        if data['revenue_cycle'].get('improvement_potential_days', 0) > 5:
            recommendations.append({
                'action': 'Implement revenue cycle process improvements',
                'impact': f"Improve cash flow by ${data['revenue_cycle']['cash_flow_improvement_potential']:,.0f}",
                'priority': 'MEDIUM'
            })

        return {
            'data': data,
            'narrative': narrative,
            'recommendations': recommendations,
            'confidence': 80,
        }

    def forecast_with_demographics(self, metric: str = 'pmpm', periods: int = 6) -> Dict:
        logger.info("Forecasting %s for %d periods by demographic", metric, periods)

        data = {
            'metric': metric,
            'periods': periods,
            'age_cohorts': {},
            'regions': {},
            'plan_types': {},
        }

        if metric == 'pmpm':
            sql = """
            SELECT
                CASE
                    WHEN CAST((JULIANDAY('now') - JULIANDAY(DATE_OF_BIRTH))/365.25 AS INT) < 35 THEN 'Under 35'
                    WHEN CAST((JULIANDAY('now') - JULIANDAY(DATE_OF_BIRTH))/365.25 AS INT) < 50 THEN '35-49'
                    WHEN CAST((JULIANDAY('now') - JULIANDAY(DATE_OF_BIRTH))/365.25 AS INT) < 65 THEN '50-64'
                    ELSE '65+'
                END as age_cohort,
                SUBSTR(SERVICE_DATE, 1, 7) as year_month,
                COUNT(DISTINCT m.MEMBER_ID) as member_count,
                ROUND(AVG(CAST(COALESCE(c.PAID_AMOUNT, 0) AS REAL)), 2) as pmpm
            FROM members m
            LEFT JOIN claims c ON m.MEMBER_ID = c.MEMBER_ID
            WHERE SERVICE_DATE IS NOT NULL
            GROUP BY age_cohort, year_month
            ORDER BY year_month DESC
            """
            result = self._safe_query(sql)

            pmpm_by_cohort = defaultdict(list)
            for r in result:
                cohort = r['age_cohort']
                pmpm_by_cohort[cohort].append(r['pmpm'] or 0)

            for cohort, values in pmpm_by_cohort.items():
                if values:
                    current_pmpm = statistics.mean(values[-3:]) if len(values) >= 3 else values[-1]
                    forecast = []
                    for p in range(1, periods + 1):
                        projected = current_pmpm * (1.05 ** p)
                        forecast.append({
                            'period': p,
                            'projected_pmpm': round(projected, 2),
                            'confidence_interval_lower': round(projected * 0.95, 2),
                            'confidence_interval_upper': round(projected * 1.05, 2),
                        })
                    data['age_cohorts'][cohort] = {
                        'current_pmpm': round(current_pmpm, 2),
                        'forecast': forecast,
                    }

        narrative = f"Demographic-aware forecast for {metric.upper()} over {periods} periods. "
        narrative += "Assumes 5% annual growth trend. "

        if data['age_cohorts']:
            highest_cohort = max(
                data['age_cohorts'].items(),
                key=lambda x: x[1]['current_pmpm']
            )
            narrative += f"Highest-cost cohort: {highest_cohort[0]} "
            narrative += f"(current PMPM ${highest_cohort[1]['current_pmpm']:,.0f}). "

        return {
            'data': data,
            'narrative': narrative,
            'recommendations': [],
            'confidence': 65,
        }

    def get_executive_summary(self, role: str = 'cfo') -> Dict:
        logger.info("Generating executive summary for role: %s", role)

        qg = self.run_quality_gate()
        demographics = self.analyze_demographics()
        costs = self.analyze_cost_drivers()
        preventive = self.analyze_preventive_opportunities()
        financial = self.get_financial_initiatives()

        denial_data = costs.get('data', {}).get('revenue_leakage', {})
        total_denied = sum(v.get('denied_amount', 0) for v in denial_data.values() if isinstance(v, dict))
        total_denial_count = sum(v.get('denial_count', 0) for v in denial_data.values() if isinstance(v, dict))
        top_denial_reason = max(denial_data.items(), key=lambda x: x[1].get('denied_amount', 0) if isinstance(x[1], dict) else 0)[0] if denial_data else 'Unknown'
        top_denial_amount = denial_data.get(top_denial_reason, {}).get('denied_amount', 0)
        recovery_potential = total_denied * 0.25

        prev_roi_data = financial.get('data', {}).get('preventive_roi_summary', {})
        prev_savings = prev_roi_data.get('annual_savings', 0)
        prev_cost = prev_roi_data.get('program_cost', 0)
        at_risk_members = len(preventive.get('data', {}).get('at_risk_members', []))
        at_risk_total = preventive.get('data', {}).get('at_risk_count', at_risk_members)
        cost_of_inaction = preventive.get('data', {}).get('cost_of_inaction', {}).get('projected_cost_increase', 0)

        total_members = sum(ac.get('member_count', 0) for ac in demographics.get('data', {}).get('age_cohorts', {}).values())

        fin_overview = costs.get('data', {})
        quarterly = fin_overview.get('quarterly_trend', {})
        quarterly_vals = list(quarterly.values()) if quarterly else []

        summary = {
            'role': role,
            'timestamp': datetime.now().isoformat(),
            'data': {'kpis': {}},
            'narrative': '',
            'recommendations': [],
            'confidence': qg.get('confidence', 70),
        }

        if role == 'cfo':
            summary['data']['kpis'] = {
                'total_denied_claims': total_denial_count,
                'total_denied_amount': total_denied,
                'recovery_potential': recovery_potential,
                'top_denial_reason': top_denial_reason,
                'top_denial_amount': top_denial_amount,
                'preventive_savings': prev_savings,
                'preventive_cost': prev_cost,
                'at_risk_count': at_risk_total,
                'cost_of_inaction': cost_of_inaction,
                'total_members': total_members,
                'data_confidence': qg.get('confidence', 0),
            }

            summary['recommendations'] = [
                {'rank': 1, 'action': f'Launch denial recovery program targeting "{top_denial_reason}" denials',
                 'impact': f'Recover ${recovery_potential:,.0f} annually from ${total_denied:,.0f} in denials',
                 'timeline': '30-60 days', 'priority': 'CRITICAL'},
                {'rank': 2, 'action': f'Implement preventive care for {at_risk_total:,} at-risk members',
                 'impact': f'Avoid ${cost_of_inaction:,.0f} in chronic disease costs',
                 'timeline': '90 days', 'priority': 'HIGH'},
                {'rank': 3, 'action': 'Deploy high-cost outlier management protocol',
                 'impact': 'Reduce top 1% claim costs by 15-20%',
                 'timeline': '60 days', 'priority': 'HIGH'},
            ]

            regional_data = demographics.get('data', {}).get('regional_performance', {})
            worst_region = None
            worst_avg_cost = 0
            best_region = None
            best_avg_cost = float('inf')
            for region, rdata in regional_data.items():
                mc = rdata.get('member_count', 0)
                if mc < 100:
                    continue
                avg_cost = rdata.get('avg_cost', 0)
                if isinstance(avg_cost, (int, float)):
                    if avg_cost > worst_avg_cost:
                        worst_avg_cost = avg_cost
                        worst_region = region
                    if avg_cost < best_avg_cost:
                        best_avg_cost = avg_cost
                        best_region = region

            trend_direction = "stable"
            qv_nums = []
            for qv in quarterly_vals:
                if isinstance(qv, (int, float)):
                    qv_nums.append(qv)
                elif isinstance(qv, dict):
                    qv_nums.append(qv.get('total_cost', qv.get('total', qv.get('amount', 0))))
            if len(qv_nums) >= 2:
                last_q = qv_nums[-1]
                prev_q = qv_nums[-2]
                if prev_q > 0:
                    change_pct = (last_q - prev_q) / prev_q * 100
                    if change_pct > 5:
                        trend_direction = f"accelerating ({change_pct:.1f}% QoQ increase)"
                    elif change_pct < -5:
                        trend_direction = f"improving ({abs(change_pct):.1f}% QoQ decrease)"
                    else:
                        trend_direction = f"stable ({change_pct:+.1f}% QoQ)"

            pmpm_data = fin_overview.get('pmpm_by_segment', {})
            highest_pmpm_segment = 'N/A'
            highest_pmpm_val = 0
            for seg, val in pmpm_data.items():
                if isinstance(val, (int, float)) and val > highest_pmpm_val:
                    highest_pmpm_val = val
                    highest_pmpm_segment = seg

            denial_rate_pct = round(total_denial_count / max(1, total_denial_count + 36020) * 100, 1)

            summary['narrative'] = (
                f"CFO INTELLIGENCE BRIEFING — {total_members:,} Members\n\n"
                f"PRIORITY #1 — Revenue Recovery (${recovery_potential:,.0f} opportunity):\n"
                f"  {total_denial_count:,} claims denied (${total_denied:,.0f} total, {denial_rate_pct}% denial rate). "
                f"Root cause: \"{top_denial_reason}\" accounts for ${top_denial_amount:,.0f} — "
                f"this is {'likely a coding/billing issue that can be systematically addressed' if top_denial_reason in ('Unknown','Other','N/A') else 'a targeted area for clinical documentation improvement'}. "
                f"Recommended: Launch denial recovery program within 30 days. Conservative recovery: ${recovery_potential:,.0f}/year.\n\n"
                f"PRIORITY #2 — Preventive Care Investment (${cost_of_inaction:,.0f} at stake):\n"
                f"  {at_risk_total:,} members showing pre-chronic indicators (obesity, pre-diabetes, anxiety/depression). "
                f"If untreated, projected additional cost: ${cost_of_inaction:,.0f}/year. "
                f"Recommended: ${prev_cost:,.0f} preventive investment yields ${prev_savings:,.0f} savings (ROI: {prev_savings/max(prev_cost,1):.1f}x). "
                f"Start with the {at_risk_total:,} at-risk members — they're the ones who will cost you the most in 12-24 months.\n\n"
                f"PRIORITY #3 — Cost Trend Management ({trend_direction}):\n"
                f"  Quarterly costs are {trend_direction}. "
                + (f"Worst-performing region: {worst_region} (avg cost ${worst_avg_cost:,.0f}) vs best: {best_region} (${best_avg_cost:,.0f}). " if worst_region and best_region and isinstance(worst_avg_cost, (int, float)) and isinstance(best_avg_cost, (int, float)) else "")
                + (f"Focus on {worst_region} — closing the gap to the average would save significant annual spend. " if worst_region else "")
                + (f"Highest PMPM segment: {highest_pmpm_segment} at ${highest_pmpm_val:,.0f}/member/month." if highest_pmpm_val else "")
                + f"\n\nDATA CONFIDENCE: {qg.get('confidence', 0):.0f}/100 "
                f"({'High — insights are reliable' if qg.get('confidence',0) >= 80 else 'Moderate — some data quality gaps exist' if qg.get('confidence',0) >= 60 else 'Low — verify key figures before acting'})."
            )

        elif role == 'vp_operations':
            high_util_count = len(preventive.get('data', {}).get('high_utilizers', []))
            care_gap_data = preventive.get('data', {}).get('care_gaps', {})
            care_gap_count = preventive.get('data', {}).get('care_gap_count', 0)

            summary['data']['kpis'] = {
                'total_members': total_members,
                'denial_count': total_denial_count,
                'denial_rate': round(total_denial_count * 100 / max(1, total_denial_count + 36020), 1),
                'high_utilizers': high_util_count,
                'care_gap_members': care_gap_count,
                'data_confidence': qg.get('confidence', 0),
            }

            summary['recommendations'] = [
                {'rank': 1, 'action': f'Implement high-utilizer care management for {high_util_count} members',
                 'impact': 'Reduce ER visits by 30%, saving ~$8K per redirected visit',
                 'timeline': '30 days', 'priority': 'HIGH'},
                {'rank': 2, 'action': f'Close care gaps — {care_gap_count} members need PCP engagement',
                 'impact': 'Early detection prevents costly emergency interventions',
                 'timeline': '45 days', 'priority': 'HIGH'},
                {'rank': 3, 'action': f'Improve data quality from {qg.get("confidence", 0):.0f}% to 95%',
                 'impact': 'Better data = better decisions across all operations',
                 'timeline': '60 days', 'priority': 'MEDIUM'},
            ]

            summary['narrative'] = (
                f"Operations Briefing: {total_members:,} enrolled members. "
                f"{total_denial_count:,} denied claims require process improvement. "
                f"{high_util_count} high-utilizer members identified for care management. "
                f"{care_gap_count} members have care gaps needing PCP engagement. "
                f"Data confidence: {qg.get('confidence', 0):.0f}/100."
            )

        elif role == 'clinical_director':
            race_data = demographics.get('data', {}).get('race_ethnicity', {})
            denial_rates = [d.get('denial_rate_pct', 0) for d in race_data.values() if isinstance(d, dict)]
            equity_gap = (max(denial_rates, default=0) - min(denial_rates, default=0))
            top_conditions = costs.get('data', {}).get('top_conditions', [])
            top_condition_name = top_conditions[0].get('description', 'Unknown') if top_conditions else 'Unknown'
            high_util_count = len(preventive.get('data', {}).get('high_utilizers', []))

            summary['data']['kpis'] = {
                'at_risk_members': at_risk_total,
                'health_equity_gap_pct': round(equity_gap, 1),
                'top_chronic_condition': top_condition_name,
                'high_utilizers': high_util_count,
                'chronic_prevalence': sum(c.get('affected_members', 0) for c in top_conditions[:5]),
            }

            summary['recommendations'] = [
                {'rank': 1, 'action': f'Address health equity gap: {equity_gap:.1f}% disparity in denial rates across demographics',
                 'impact': 'Ensure equitable care delivery across all populations',
                 'timeline': '30 days', 'priority': 'CRITICAL'},
                {'rank': 2, 'action': f'Launch chronic disease management for "{top_condition_name}"',
                 'impact': f'Top cost-driving condition affecting population health',
                 'timeline': '45 days', 'priority': 'HIGH'},
                {'rank': 3, 'action': f'Preventive intervention for {at_risk_total:,} at-risk members',
                 'impact': 'Prevent chronic conversion, reduce long-term cost burden',
                 'timeline': '60 days', 'priority': 'HIGH'},
            ]

            summary['narrative'] = (
                f"Clinical Director Briefing: {at_risk_total:,} members at risk of chronic conversion. "
                f"Health equity gap: {equity_gap:.1f}% disparity in denial rates across demographics. "
                f"Top chronic condition driver: \"{top_condition_name}\". "
                f"{high_util_count} high utilizers need intensive care management. "
                f"Preventive intervention recommended to avoid ${cost_of_inaction:,.0f} in projected costs."
            )

        else:
            expected_conversions = preventive.get('data', {}).get('cost_of_inaction', {}).get('expected_conversions_yr1', 0)
            roi_val = prev_roi_data.get('roi', 0)

            summary['data']['kpis'] = {
                'population_size': total_members,
                'at_risk_count': at_risk_total,
                'cost_of_inaction': cost_of_inaction,
                'expected_conversions': expected_conversions,
                'intervention_roi_pct': round(roi_val * 100, 1) if isinstance(roi_val, float) else 0,
                'preventive_savings': prev_savings,
            }

            summary['recommendations'] = [
                {'rank': 1, 'action': f'Execute diabetes prevention program for {int(at_risk_total * 0.3)} highest-risk members',
                 'impact': f'Prevent chronic conversion, save ~$8K per prevented case',
                 'timeline': '30 days', 'priority': 'CRITICAL'},
                {'rank': 2, 'action': f'Hypertension management — prevent {expected_conversions} chronic conversions',
                 'impact': f'Avoid ${cost_of_inaction:,.0f} annual cost increase',
                 'timeline': '45 days', 'priority': 'HIGH'},
                {'rank': 3, 'action': 'Deploy population health monitoring dashboard',
                 'impact': 'Track chronic prevalence trends quarterly for early intervention',
                 'timeline': 'Ongoing', 'priority': 'MEDIUM'},
            ]

            summary['narrative'] = (
                f"Population Health Briefing: {total_members:,} members, {at_risk_total:,} at risk of chronic conversion. "
                f"If untreated, projected {expected_conversions} conversions will cost an additional ${cost_of_inaction:,.0f}/year. "
                f"Recommended preventive program: ${prev_cost:,.0f} investment for ${prev_savings:,.0f} annual savings. "
                f"Data confidence: {qg.get('confidence', 0):.0f}/100."
            )

        return summary

    def get_full_intelligence_report(self) -> Dict:
        logger.info("Generating full intelligence report")

        start_time = time.time()

        report = {
            'timestamp': datetime.now().isoformat(),
            'quality_gate': self.run_quality_gate(),
            'demographics': self.analyze_demographics(),
            'cost_drivers': self.analyze_cost_drivers(),
            'preventive_opportunities': self.analyze_preventive_opportunities(),
            'financial_initiatives': self.get_financial_initiatives(),
            'forecast_pmpm': self.forecast_with_demographics('pmpm', periods=6),
            'executive_summaries': {
                'cfo': self.get_executive_summary('cfo'),
                'vp_operations': self.get_executive_summary('vp_operations'),
                'clinical_director': self.get_executive_summary('clinical_director'),
                'population_health': self.get_executive_summary('population_health'),
            }
        }

        elapsed = time.time() - start_time
        report['generation_time_seconds'] = elapsed
        report['overall_confidence'] = statistics.mean([
            report['quality_gate']['confidence'],
            report['demographics']['confidence'],
            report['cost_drivers']['confidence'],
            report['preventive_opportunities']['confidence'],
            report['financial_initiatives']['confidence'],
        ])

        logger.info("Full intelligence report generated in %.2fs", elapsed)

        return report


def create_deep_analytics(db_path: str) -> DeepHealthcareAnalytics:
    return DeepHealthcareAnalytics(db_path)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    db_path = '/sessions/great-gallant-allen/mnt/chatbot/mtp_demo/chatbot_healthcare/scripts/healthcare_production.db'

    print("Initializing Deep Healthcare Analytics Engine...")
    analytics = DeepHealthcareAnalytics(db_path)

    print("\nRunning quality gate...")
    qg_result = analytics.run_quality_gate()
    print(f"Data quality confidence: {qg_result['confidence']:.0f}%")

    print("\nAnalyzing demographics...")
    demo_result = analytics.analyze_demographics()
    print(f"Demographics analysis: {demo_result['narrative']}")

    print("\nAnalyzing cost drivers...")
    cost_result = analytics.analyze_cost_drivers()
    print(f"Cost analysis: {cost_result['narrative']}")

    print("\nGenerating full intelligence report...")
    full_report = analytics.get_full_intelligence_report()
    print(f"\nReport generated in {full_report['generation_time_seconds']:.2f}s")
    print(f"Overall confidence: {full_report['overall_confidence']:.0f}%")
