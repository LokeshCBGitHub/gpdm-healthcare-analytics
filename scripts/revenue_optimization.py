import sqlite3
import logging
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta
from decimal import Decimal
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RevenueOptimizationEngine:

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self._connect()
        logger.info(f"RevenueOptimizationEngine initialized with database: {db_path}")

    def _connect(self):
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            logger.debug("Database connection established")
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {e}")
            raise

    def _execute_query(self, query: str, params: Tuple = ()) -> List[sqlite3.Row]:
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Query execution error: {e}")
            return []

    def _get_scalar(self, query: str, params: Tuple = ()) -> Any:
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            result = cursor.fetchone()
            return result[0] if result else None
        except sqlite3.Error as e:
            logger.error(f"Scalar query error: {e}")
            return None

    def get_churn_risk_analysis(self) -> Dict[str, Any]:
        logger.info("Starting churn risk analysis")

        analysis = {
            "analysis_date": datetime.now().isoformat(),
            "total_members_analyzed": 0,
            "risk_segments": {},
            "top_risk_members": [],
            "interventions_by_segment": {},
            "summary": {}
        }

        try:
            total_members = self._get_scalar(
                "SELECT COUNT(*) FROM members WHERE (DISENROLLMENT_DATE IS NULL OR DISENROLLMENT_DATE = '')"
            ) or 0
            if total_members == 0:
                total_members = self._get_scalar("SELECT COUNT(*) FROM members") or 1
            analysis["total_members_analyzed"] = total_members

            risk_query = """
            SELECT
                m.MEMBER_ID,
                m.KP_REGION,
                m.PLAN_TYPE,
                CAST(m.RISK_SCORE AS REAL) as RISK_SCORE,
                CAST((julianday('now') - julianday(m.DATE_OF_BIRTH)) / 365.25 AS INTEGER) as AGE,
                COALESCE(d.denied_claims, 0) as denied_claims,
                COALESCE(e.encounter_count, 0) as encounter_count,
                CASE WHEN m.CHRONIC_CONDITIONS IS NOT NULL AND m.CHRONIC_CONDITIONS != '' THEN 1 ELSE 0 END as has_chronic,
                CAST(m.ENROLLMENT_DATE as TEXT) as enrollment_date,
                CASE
                    WHEN d.denied_claims > 5 THEN 30
                    WHEN d.denied_claims > 2 THEN 20
                    WHEN d.denied_claims > 0 THEN 10
                    ELSE 0
                END +
                CASE
                    WHEN e.encounter_count = 0 THEN 30
                    WHEN e.encounter_count < 2 THEN 20
                    ELSE 0
                END +
                CASE
                    WHEN CAST(m.RISK_SCORE AS REAL) > 4.0 THEN 20
                    WHEN CAST(m.RISK_SCORE AS REAL) > 3.0 THEN 10
                    ELSE 0
                END +
                CASE
                    WHEN m.DISENROLLMENT_DATE IS NOT NULL THEN 20
                    ELSE 0
                END as churn_risk_score
            FROM members m
            LEFT JOIN (
                SELECT MEMBER_ID, COUNT(*) as denied_claims
                FROM claims
                WHERE CLAIM_STATUS = 'Denied'
                GROUP BY MEMBER_ID
            ) d ON m.MEMBER_ID = d.MEMBER_ID
            LEFT JOIN (
                SELECT MEMBER_ID, COUNT(*) as encounter_count
                FROM encounters
                WHERE SERVICE_DATE >= date('now', '-12 months')
                GROUP BY MEMBER_ID
            ) e ON m.MEMBER_ID = e.MEMBER_ID
            WHERE (m.DISENROLLMENT_DATE IS NULL OR m.DISENROLLMENT_DATE = '')
            ORDER BY churn_risk_score DESC
            """

            members_risk = self._execute_query(risk_query)

            risk_tiers = {"Critical": [], "High": [], "Medium": [], "Low": []}

            for row in members_risk:
                member_data = {
                    "member_id": row["MEMBER_ID"],
                    "region": row["KP_REGION"],
                    "plan_type": row["PLAN_TYPE"],
                    "age": row["AGE"],
                    "churn_risk_score": row["churn_risk_score"],
                    "denied_claims": row["denied_claims"],
                    "encounters_12m": row["encounter_count"],
                    "has_chronic": row["has_chronic"],
                    "risk_factors": []
                }

                if row["denied_claims"] > 0:
                    member_data["risk_factors"].append(
                        f"Denied claims: {row['denied_claims']}"
                    )
                if row["encounter_count"] == 0:
                    member_data["risk_factors"].append("No recent encounters")
                elif row["encounter_count"] < 2:
                    member_data["risk_factors"].append("Low engagement (<2 encounters/year)")
                if row["RISK_SCORE"] and float(row["RISK_SCORE"]) > 4.0:
                    member_data["risk_factors"].append("High medical risk")

                score = row["churn_risk_score"]
                if score > 80:
                    risk_tiers["Critical"].append(member_data)
                elif score >= 60:
                    risk_tiers["High"].append(member_data)
                elif score >= 40:
                    risk_tiers["Medium"].append(member_data)
                else:
                    risk_tiers["Low"].append(member_data)

            for tier_name in ["Critical", "High", "Medium", "Low"]:
                tier_members = risk_tiers[tier_name]
                analysis["risk_segments"][tier_name] = {
                    "count": len(tier_members),
                    "percentage": round(len(tier_members) / max(total_members, 1) * 100, 2),
                    "top_members": tier_members[:10]
                }

                if len(analysis["top_risk_members"]) < 20:
                    analysis["top_risk_members"].extend(
                        tier_members[:max(5, 20 - len(analysis["top_risk_members"]))]
                    )

            analysis["interventions_by_segment"] = {
                "Critical": [
                    "Priority outreach (within 48 hours)",
                    "Executive care coordination",
                    "Root cause analysis for grievances/denials",
                    "Personalized service recovery plan"
                ],
                "High": [
                    "Dedicated account manager outreach",
                    "Claims assistance and appeals support",
                    "Preventive care engagement program",
                    "Member satisfaction survey"
                ],
                "Medium": [
                    "Routine care coordination contact",
                    "Wellness program enrollment",
                    "Online portal training",
                    "Quarterly check-in"
                ],
                "Low": [
                    "Standard engagement communications",
                    "Annual wellness reminders",
                    "Plan benefit updates"
                ]
            }

            analysis["summary"] = {
                "at_risk_members": sum(
                    analysis["risk_segments"][tier]["count"]
                    for tier in ["Critical", "High"]
                ),
                "at_risk_percentage": round(
                    sum(
                        analysis["risk_segments"][tier]["count"]
                        for tier in ["Critical", "High"]
                    ) / max(total_members, 1) * 100, 2
                ),
                "critical_interventions_needed": analysis["risk_segments"]["Critical"]["count"],
                "estimated_revenue_at_risk": self._estimate_revenue_at_risk(
                    analysis["risk_segments"]["Critical"]["count"]
                )
            }

            logger.info(f"Churn risk analysis complete: {analysis['summary']}")
            return analysis

        except Exception as e:
            logger.error(f"Error in churn risk analysis: {e}")
            raise

    def get_pmpm_optimization(self) -> Dict[str, Any]:
        logger.info("Starting PMPM optimization analysis")

        analysis = {
            "analysis_date": datetime.now().isoformat(),
            "pmpm_by_dimensions": {},
            "high_cost_members": [],
            "cost_split": {},
            "cost_reduction_opportunities": [],
            "summary": {}
        }

        try:
            total_active = self._get_scalar(
                "SELECT COUNT(*) FROM members WHERE DISENROLLMENT_DATE IS NULL"
            )

            member_months = total_active * 12

            region_pmpm = self._execute_query("""
            SELECT
                m.KP_REGION as dimension,
                SUM(CAST(c.PAID_AMOUNT AS REAL)) as total_medical_cost,
                COALESCE(SUM(CAST(rx.PAID_AMOUNT AS REAL)), 0) as total_pharmacy_cost,
                COUNT(DISTINCT m.MEMBER_ID) as member_count,
                ROUND(
                    (SUM(CAST(c.PAID_AMOUNT AS REAL)) + COALESCE(SUM(CAST(rx.PAID_AMOUNT AS REAL)), 0)) /
                    (COUNT(DISTINCT m.MEMBER_ID) * 12),
                    2
                ) as pmpm
            FROM members m
            LEFT JOIN claims c ON m.MEMBER_ID = c.MEMBER_ID
            LEFT JOIN prescriptions rx ON m.MEMBER_ID = rx.MEMBER_ID
            WHERE (m.DISENROLLMENT_DATE IS NULL OR m.DISENROLLMENT_DATE = '')
            GROUP BY m.KP_REGION
            ORDER BY pmpm DESC
            """)

            analysis["pmpm_by_dimensions"]["by_region"] = [
                {
                    "region": row["dimension"],
                    "pmpm": row["pmpm"],
                    "medical_cost": row["total_medical_cost"],
                    "pharmacy_cost": row["total_pharmacy_cost"],
                    "members": row["member_count"]
                }
                for row in region_pmpm
            ]

            plan_pmpm = self._execute_query("""
            SELECT
                m.PLAN_TYPE as dimension,
                SUM(c.PAID_AMOUNT) as total_medical_cost,
                COALESCE(SUM(CAST(rx.PAID_AMOUNT AS REAL)), 0) as total_pharmacy_cost,
                COUNT(DISTINCT m.MEMBER_ID) as member_count,
                ROUND(
                    (SUM(CAST(c.PAID_AMOUNT AS REAL)) + COALESCE(SUM(CAST(rx.PAID_AMOUNT AS REAL)), 0)) /
                    (COUNT(DISTINCT m.MEMBER_ID) * 12),
                    2
                ) as pmpm
            FROM members m
            LEFT JOIN claims c ON m.MEMBER_ID = c.MEMBER_ID
            LEFT JOIN prescriptions rx ON m.MEMBER_ID = rx.MEMBER_ID
            WHERE (m.DISENROLLMENT_DATE IS NULL OR m.DISENROLLMENT_DATE = '')
            GROUP BY m.PLAN_TYPE
            ORDER BY pmpm DESC
            """)

            analysis["pmpm_by_dimensions"]["by_plan_type"] = [
                {
                    "plan_type": row["dimension"],
                    "pmpm": row["pmpm"],
                    "medical_cost": row["total_medical_cost"],
                    "pharmacy_cost": row["total_pharmacy_cost"],
                    "members": row["member_count"]
                }
                for row in plan_pmpm
            ]

            risk_pmpm = self._execute_query("""
            SELECT
                CASE
                    WHEN CAST(m.RISK_SCORE AS REAL) >= 3.0 THEN 'High Risk'
                    WHEN CAST(m.RISK_SCORE AS REAL) >= 2.0 THEN 'Moderate Risk'
                    ELSE 'Low Risk'
                END as dimension,
                SUM(CAST(c.PAID_AMOUNT AS REAL)) as total_medical_cost,
                0 as total_pharmacy_cost,
                COUNT(DISTINCT m.MEMBER_ID) as member_count,
                ROUND(
                    SUM(CAST(c.PAID_AMOUNT AS REAL)) /
                    NULLIF(COUNT(DISTINCT m.MEMBER_ID) * 12, 0),
                    2
                ) as pmpm
            FROM members m
            LEFT JOIN claims c ON m.MEMBER_ID = c.MEMBER_ID
            WHERE (m.DISENROLLMENT_DATE IS NULL OR m.DISENROLLMENT_DATE = '')
            GROUP BY CASE
                WHEN CAST(m.RISK_SCORE AS REAL) >= 3.0 THEN 'High Risk'
                WHEN CAST(m.RISK_SCORE AS REAL) >= 2.0 THEN 'Moderate Risk'
                ELSE 'Low Risk'
            END
            ORDER BY pmpm DESC
            """)

            analysis["pmpm_by_dimensions"]["by_risk_tier"] = [
                {
                    "risk_tier": row["dimension"],
                    "pmpm": row["pmpm"],
                    "medical_cost": row["total_medical_cost"],
                    "pharmacy_cost": row["total_pharmacy_cost"],
                    "members": row["member_count"]
                }
                for row in risk_pmpm
            ]

            high_cost = self._execute_query("""
            SELECT
                m.MEMBER_ID,
                m.KP_REGION,
                m.PLAN_TYPE,
                CAST((julianday('now') - julianday(m.DATE_OF_BIRTH)) / 365.25 AS INTEGER) as AGE,
                CAST(m.RISK_SCORE AS REAL) as RISK_SCORE,
                m.CHRONIC_CONDITIONS,
                SUM(CAST(c.PAID_AMOUNT AS REAL)) as medical_cost,
                0 as pharmacy_cost,
                ROUND(SUM(CAST(c.PAID_AMOUNT AS REAL)) / 12, 2) as monthly_cost
            FROM members m
            LEFT JOIN claims c ON m.MEMBER_ID = c.MEMBER_ID
            WHERE (m.DISENROLLMENT_DATE IS NULL OR m.DISENROLLMENT_DATE = '')
            GROUP BY m.MEMBER_ID
            ORDER BY monthly_cost DESC
            LIMIT 100
            """)

            analysis["high_cost_members"] = [
                {
                    "member_id": row["MEMBER_ID"],
                    "region": row["KP_REGION"],
                    "plan_type": row["PLAN_TYPE"],
                    "age": row["AGE"],
                    "risk_score": row["RISK_SCORE"],
                    "chronic_conditions": row["CHRONIC_CONDITIONS"],
                    "monthly_pmpm": row["monthly_cost"],
                    "medical_cost": row["medical_cost"],
                    "pharmacy_cost": row["pharmacy_cost"]
                }
                for row in high_cost
            ]

            total_costs = self._execute_query("""
            SELECT
                (SELECT SUM(CAST(PAID_AMOUNT AS REAL)) FROM claims) as total_medical,
                (SELECT COALESCE(SUM(CAST(PAID_AMOUNT AS REAL)), 0) FROM prescriptions) as total_pharmacy
            """)

            if total_costs and total_costs[0]["total_medical"]:
                medical_total = total_costs[0]["total_medical"]
                pharmacy_total = total_costs[0]["total_pharmacy"]
                total = medical_total + pharmacy_total
                analysis["cost_split"] = {
                    "medical": {
                        "amount": medical_total,
                        "percentage": round(medical_total / total * 100, 2)
                    },
                    "pharmacy": {
                        "amount": pharmacy_total,
                        "percentage": round(pharmacy_total / total * 100, 2)
                    }
                }

            ed_visits = self._execute_query("""
            SELECT COUNT(*) as preventable_ed_visits
            FROM encounters
            WHERE VISIT_TYPE IN ('EMERGENCY', 'Emergency', 'ED', 'ER')
            AND SERVICE_DATE >= date('now', '-12 months')
            """)

            if ed_visits and ed_visits[0]["preventable_ed_visits"]:
                ed_count = ed_visits[0]["preventable_ed_visits"]
                analysis["cost_reduction_opportunities"].append({
                    "opportunity": "Preventable ED visits",
                    "current_volume": ed_count,
                    "reduction_potential": round(ed_count * 0.20, 0),
                    "cost_per_visit": 1250,
                    "annual_savings": round(ed_count * 0.20 * 1250, 0),
                    "description": "Divert 20% of low-acuity ED visits to urgent care/primary care"
                })

            readmissions = self._get_scalar("""
            SELECT COUNT(DISTINCT MEMBER_ID) as high_readmission_members
            FROM claims
            WHERE SERVICE_DATE >= date('now', '-12 months')
            AND CLAIM_STATUS = 'Approved'
            GROUP BY MEMBER_ID
            HAVING COUNT(*) > 20
            """) or 0

            if readmissions > 0:
                analysis["cost_reduction_opportunities"].append({
                    "opportunity": "Readmission prevention program",
                    "current_volume": readmissions,
                    "reduction_potential": round(readmissions * 0.15, 0),
                    "cost_per_intervention": 2000,
                    "annual_savings": round(readmissions * 0.15 * 8000, 0),
                    "description": "Intensive case management for high-risk readmission members"
                })

            analysis["cost_reduction_opportunities"].append({
                "opportunity": "Generic drug adoption",
                "current_volume": "N/A",
                "reduction_potential": "5-10%",
                "cost_per_intervention": 500,
                "annual_savings": round(
                    (analysis["cost_split"].get("pharmacy", {}).get("amount", 0) * 0.075),
                    0
                ),
                "description": "Increase generic drug utilization to 90%+"
            })

            avg_pmpm = self._get_scalar("""
            SELECT ROUND(
                SUM(CAST(c.PAID_AMOUNT AS REAL)) /
                NULLIF(COUNT(DISTINCT m.MEMBER_ID) * 12, 0),
                2
            )
            FROM members m
            LEFT JOIN claims c ON m.MEMBER_ID = c.MEMBER_ID
            WHERE (m.DISENROLLMENT_DATE IS NULL OR m.DISENROLLMENT_DATE = '')
            """) or 0

            total_opportunity = sum(
                opp.get("annual_savings", 0)
                for opp in analysis["cost_reduction_opportunities"]
            )

            analysis["summary"] = {
                "current_average_pmpm": avg_pmpm,
                "target_pmpm": round(avg_pmpm * 0.92, 2),
                "improvement_potential_percentage": 8.0,
                "total_cost_reduction_opportunity": total_opportunity,
                "high_cost_members_count": len(analysis["high_cost_members"]),
                "top_opportunity": max(
                    analysis["cost_reduction_opportunities"],
                    key=lambda x: x.get("annual_savings", 0)
                ) if analysis["cost_reduction_opportunities"] else None
            }

            logger.info(f"PMPM analysis complete: Average PMPM ${avg_pmpm}")
            return analysis

        except Exception as e:
            logger.error(f"Error in PMPM optimization: {e}")
            raise

    def get_denial_recovery(self) -> Dict[str, Any]:
        logger.info("Starting denial recovery analysis")

        analysis = {
            "analysis_date": datetime.now().isoformat(),
            "denial_overview": {},
            "denials_by_reason": [],
            "denials_by_provider": [],
            "denials_by_region": [],
            "recovery_opportunities": [],
            "summary": {}
        }

        try:
            denial_stats = self._execute_query("""
            SELECT
                COUNT(*) as total_claims,
                SUM(CASE WHEN CLAIM_STATUS = 'Denied' THEN 1 ELSE 0 END) as denied_count,
                SUM(CASE WHEN CLAIM_STATUS = 'Denied' THEN BILLED_AMOUNT ELSE 0 END) as denied_amount
            FROM claims
            """)

            if denial_stats and denial_stats[0]["total_claims"]:
                row = denial_stats[0]
                denied_count = row["denied_count"] or 0
                denied_amount = row["denied_amount"] or 0
                total_claims = row["total_claims"]

                analysis["denial_overview"] = {
                    "total_claims": total_claims,
                    "denied_claims": denied_count,
                    "denial_rate_percentage": round(denied_count / total_claims * 100, 2),
                    "revenue_at_risk": denied_amount,
                    "recovery_potential_percentage": 60
                }

            denials_by_reason = self._execute_query("""
            SELECT
                COALESCE(CPT_DESCRIPTION, 'Unknown') as denial_reason,
                COUNT(*) as denial_count,
                SUM(BILLED_AMOUNT) as billed_amount,
                ROUND(COUNT(*) * 100.0 /
                    (SELECT COUNT(*) FROM claims WHERE CLAIM_STATUS = 'Denied'), 2
                ) as percentage_of_denials
            FROM claims
            WHERE CLAIM_STATUS = 'Denied'
            GROUP BY CPT_DESCRIPTION
            ORDER BY denial_count DESC
            LIMIT 15
            """)

            analysis["denials_by_reason"] = [
                {
                    "reason": row["denial_reason"],
                    "count": row["denial_count"],
                    "billed_amount": row["billed_amount"],
                    "percentage": row["percentage_of_denials"],
                    "recovery_potential": round(row["billed_amount"] * 0.60, 2)
                }
                for row in denials_by_reason
            ]

            denials_by_provider = self._execute_query("""
            SELECT
                RENDERING_NPI as PROVIDER_ID,
                COUNT(*) as denial_count,
                SUM(BILLED_AMOUNT) as billed_amount,
                COUNT(*) * 100.0 /
                    (SELECT COUNT(*) FROM claims WHERE CLAIM_STATUS = 'Denied') as percentage
            FROM claims
            WHERE CLAIM_STATUS = 'Denied'
            GROUP BY RENDERING_NPI
            ORDER BY denial_count DESC
            LIMIT 10
            """)

            analysis["denials_by_provider"] = [
                {
                    "provider_id": row["PROVIDER_ID"],
                    "denial_count": row["denial_count"],
                    "billed_amount": row["billed_amount"],
                    "percentage_of_denials": round(row["percentage"], 2),
                    "recovery_opportunity": round(row["billed_amount"] * 0.60, 2)
                }
                for row in denials_by_provider
            ]

            denials_by_region = self._execute_query("""
            SELECT
                m.KP_REGION,
                COUNT(c.CLAIM_ID) as denial_count,
                SUM(c.BILLED_AMOUNT) as billed_amount,
                COUNT(c.CLAIM_ID) * 100.0 /
                    (SELECT COUNT(*) FROM claims WHERE CLAIM_STATUS = 'Denied') as percentage
            FROM claims c
            LEFT JOIN members m ON c.MEMBER_ID = m.MEMBER_ID
            WHERE c.CLAIM_STATUS = 'Denied'
            GROUP BY m.KP_REGION
            ORDER BY denial_count DESC
            """)

            analysis["denials_by_region"] = [
                {
                    "region": row["KP_REGION"],
                    "denial_count": row["denial_count"],
                    "billed_amount": row["billed_amount"],
                    "percentage_of_denials": round(row["percentage"], 2),
                    "recovery_opportunity": round(row["billed_amount"] * 0.60, 2)
                }
                for row in denials_by_region
            ]

            total_recovery_potential = sum(
                opp["recovery_potential"]
                for opp in analysis["denials_by_reason"]
            )

            analysis["recovery_opportunities"] = [
                {
                    "category": "Resubmission of top denial categories",
                    "potential_recovery": round(total_recovery_potential, 2),
                    "success_rate": 0.60,
                    "expected_recovery": round(total_recovery_potential * 0.60, 2),
                    "effort": "Medium",
                    "timeline_days": 30
                },
                {
                    "category": "Provider education on coding standards",
                    "potential_recovery": round(total_recovery_potential * 0.3, 2),
                    "success_rate": 0.75,
                    "expected_recovery": round(total_recovery_potential * 0.3 * 0.75, 2),
                    "effort": "High",
                    "timeline_days": 60
                },
                {
                    "category": "Member appeals assistance program",
                    "potential_recovery": round(total_recovery_potential * 0.2, 2),
                    "success_rate": 0.40,
                    "expected_recovery": round(total_recovery_potential * 0.2 * 0.40, 2),
                    "effort": "Low",
                    "timeline_days": 45
                }
            ]

            analysis["summary"] = {
                "total_denied_amount": analysis["denial_overview"].get("revenue_at_risk", 0),
                "recovery_potential": round(total_recovery_potential * 0.60, 2),
                "top_denial_reason": analysis["denials_by_reason"][0] if analysis["denials_by_reason"] else None,
                "highest_risk_provider": analysis["denials_by_provider"][0] if analysis["denials_by_provider"] else None,
                "recommended_action": "Implement denial management program focusing on top 3 denial reasons"
            }

            logger.info(f"Denial recovery analysis complete: {analysis['summary']}")
            return analysis

        except Exception as e:
            logger.error(f"Error in denial recovery analysis: {e}")
            raise

    def get_hcc_gap_closure(self) -> Dict[str, Any]:
        logger.info("Starting HCC gap closure analysis")

        analysis = {
            "analysis_date": datetime.now().isoformat(),
            "gap_analysis": {},
            "priority_members": [],
            "coding_opportunities": [],
            "summary": {}
        }

        try:
            gap_members = self._execute_query("""
            SELECT
                m.MEMBER_ID,
                m.KP_REGION,
                m.PLAN_TYPE,
                CAST((julianday('now') - julianday(m.DATE_OF_BIRTH)) / 365.25 AS INTEGER) as AGE,
                CAST(m.RISK_SCORE AS REAL) as RISK_SCORE,
                m.CHRONIC_CONDITIONS,
                COUNT(DISTINCT c.CLAIM_ID) as claim_count,
                SUM(CAST(c.PAID_AMOUNT AS REAL)) as total_medical_cost,
                COUNT(DISTINCT c.ICD10_CODE) as unique_diagnoses,
                COUNT(DISTINCT e.PRIMARY_DIAGNOSIS) as unique_encounter_diagnoses
            FROM members m
            LEFT JOIN claims c ON m.MEMBER_ID = c.MEMBER_ID
            LEFT JOIN encounters e ON m.MEMBER_ID = e.MEMBER_ID
            WHERE (m.DISENROLLMENT_DATE IS NULL OR m.DISENROLLMENT_DATE = '')
            GROUP BY m.MEMBER_ID
            HAVING COUNT(DISTINCT c.CLAIM_ID) > 10
                AND COUNT(DISTINCT c.ICD10_CODE) < 3
            ORDER BY total_medical_cost DESC
            LIMIT 100
            """)

            for row in gap_members:
                potential_risk_increase = min(row["claim_count"] * 0.15, 25)
                revenue_per_risk_point = 50

                member = {
                    "member_id": row["MEMBER_ID"],
                    "region": row["KP_REGION"],
                    "plan_type": row["PLAN_TYPE"],
                    "age": row["AGE"],
                    "current_risk_score": row["RISK_SCORE"],
                    "potential_risk_score": round(
                        row["RISK_SCORE"] + potential_risk_increase, 2
                    ),
                    "risk_gap": round(potential_risk_increase, 2),
                    "medical_cost": row["total_medical_cost"],
                    "claims_volume": row["claim_count"],
                    "documented_diagnoses": row["unique_diagnoses"],
                    "encounter_diagnoses": row["unique_encounter_diagnoses"],
                    "estimated_annual_revenue_uplift": round(
                        potential_risk_increase * 12 * revenue_per_risk_point, 2
                    ),
                    "priority_level": "High" if potential_risk_increase > 15 else "Medium"
                }
                analysis["priority_members"].append(member)

            analysis["priority_members"].sort(
                key=lambda x: x["estimated_annual_revenue_uplift"],
                reverse=True
            )

            coding_opps = self._execute_query("""
            SELECT
                ICD10_CODE as DIAGNOSIS_CODE,
                COUNT(*) as frequency,
                COUNT(DISTINCT MEMBER_ID) as member_count
            FROM claims
            WHERE ICD10_CODE IS NOT NULL
                AND ICD10_CODE != ''
            GROUP BY ICD10_CODE
            ORDER BY frequency DESC
            LIMIT 20
            """)

            analysis["coding_opportunities"] = [
                {
                    "diagnosis_code": row["DIAGNOSIS_CODE"],
                    "frequency": row["frequency"],
                    "affected_members": row["member_count"],
                    "typical_risk_points": 5,
                    "description": f"Ensure complete HCC coding for {row['DIAGNOSIS_CODE']}"
                }
                for row in coding_opps
            ]

            total_gap_revenue = sum(
                member["estimated_annual_revenue_uplift"]
                for member in analysis["priority_members"]
            )
            avg_risk_gap = (
                sum(member["risk_gap"] for member in analysis["priority_members"]) /
                len(analysis["priority_members"])
                if analysis["priority_members"] else 0
            )

            analysis["summary"] = {
                "members_with_gaps": len(analysis["priority_members"]),
                "total_annual_revenue_opportunity": round(total_gap_revenue, 2),
                "average_risk_gap_per_member": round(avg_risk_gap, 2),
                "top_priority_member": analysis["priority_members"][0] if analysis["priority_members"] else None,
                "recommended_actions": [
                    "Implement chart review process for high-gap members",
                    "Conduct provider education on complete HCC documentation",
                    "Establish quarterly coding gap analysis"
                ]
            }

            logger.info(f"HCC gap analysis complete: ${total_gap_revenue} opportunity identified")
            return analysis

        except Exception as e:
            logger.error(f"Error in HCC gap analysis: {e}")
            raise

    def get_retention_targeting(self) -> Dict[str, Any]:
        logger.info("Starting retention targeting analysis")

        analysis = {
            "analysis_date": datetime.now().isoformat(),
            "renewal_members": [],
            "engagement_distribution": {},
            "intervention_strategies": {},
            "summary": {}
        }

        try:
            renewal_date = datetime.now() + timedelta(days=60)
            renewal_members = self._execute_query("""
            SELECT
                m.MEMBER_ID,
                m.KP_REGION,
                m.PLAN_TYPE,
                CAST((julianday('now') - julianday(m.DATE_OF_BIRTH)) / 365.25 AS INTEGER) as AGE,
                CAST(m.RISK_SCORE AS REAL) as RISK_SCORE,
                m.ENROLLMENT_DATE,
                COUNT(DISTINCT e.ENCOUNTER_ID) as encounters_12m,
                COALESCE(SUM(CAST(c.PAID_AMOUNT AS REAL)), 0) as medical_cost,
                0 as unresolved_grievances,
                COUNT(DISTINCT CASE WHEN c.CLAIM_STATUS = 'Denied' THEN c.CLAIM_ID END) as denied_claims
            FROM members m
            LEFT JOIN encounters e ON m.MEMBER_ID = e.MEMBER_ID
                AND e.SERVICE_DATE >= date('now', '-12 months')
            LEFT JOIN claims c ON m.MEMBER_ID = c.MEMBER_ID
            WHERE (m.DISENROLLMENT_DATE IS NULL OR m.DISENROLLMENT_DATE = '')
            GROUP BY m.MEMBER_ID
            """)

            for row in renewal_members:
                engagement_score = 100
                engagement_score -= max(0, (12 - row["encounters_12m"]) * 5)
                engagement_score -= row["unresolved_grievances"] * 20
                engagement_score -= row["denied_claims"] * 10
                engagement_score = max(0, min(100, engagement_score))

                retention_risk = 100 - engagement_score

                member = {
                    "member_id": row["MEMBER_ID"],
                    "region": row["KP_REGION"],
                    "plan_type": row["PLAN_TYPE"],
                    "age": row["AGE"],
                    "risk_score": row["RISK_SCORE"],
                    "engagement_score": round(engagement_score, 1),
                    "retention_risk": round(retention_risk, 1),
                    "encounters_12m": row["encounters_12m"],
                    "medical_cost": row["medical_cost"],
                    "grievances": row["unresolved_grievances"],
                    "denied_claims": row["denied_claims"],
                    "lifetime_value": round(row["medical_cost"] / 12 * 12, 2),
                    "recommended_interventions": []
                }

                if retention_risk > 70:
                    member["recommended_interventions"].append({
                        "action": "Executive outreach",
                        "cost": 500,
                        "expected_roi": 0.85,
                        "description": "Senior account manager personal contact"
                    })
                elif retention_risk > 50:
                    member["recommended_interventions"].append({
                        "action": "Personalized renewal offer",
                        "cost": 200,
                        "expected_roi": 0.65,
                        "description": "Customized plan options and benefits review"
                    })
                elif retention_risk > 30:
                    member["recommended_interventions"].append({
                        "action": "Member engagement program",
                        "cost": 100,
                        "expected_roi": 0.50,
                        "description": "Wellness program enrollment and incentives"
                    })

                if row["unresolved_grievances"] > 0:
                    member["recommended_interventions"].append({
                        "action": "Grievance resolution follow-up",
                        "cost": 300,
                        "expected_roi": 0.90,
                        "description": "Dedicated resolution support"
                    })

                if row["denied_claims"] > 0:
                    member["recommended_interventions"].append({
                        "action": "Claims assistance",
                        "cost": 150,
                        "expected_roi": 0.75,
                        "description": "Appeals support and claim status assistance"
                    })

                analysis["renewal_members"].append(member)

            engagement_ranges = [
                (80, 100, "Highly Engaged"),
                (60, 79, "Engaged"),
                (40, 59, "At Risk"),
                (0, 39, "High Risk")
            ]

            for low, high, label in engagement_ranges:
                count = sum(
                    1 for m in analysis["renewal_members"]
                    if low <= m["engagement_score"] < high + 1
                )
                analysis["engagement_distribution"][label] = {
                    "count": count,
                    "percentage": round(
                        count / len(analysis["renewal_members"]) * 100, 2
                    ) if analysis["renewal_members"] else 0
                }

            strategies = {
                "High Risk": {
                    "target_members": sum(
                        1 for m in analysis["renewal_members"]
                        if m["engagement_score"] < 40
                    ),
                    "avg_intervention_cost": 950,
                    "expected_retention_lift": 0.50,
                    "priority": "Critical"
                },
                "At Risk": {
                    "target_members": sum(
                        1 for m in analysis["renewal_members"]
                        if 40 <= m["engagement_score"] < 60
                    ),
                    "avg_intervention_cost": 600,
                    "expected_retention_lift": 0.65,
                    "priority": "High"
                },
                "Engaged": {
                    "target_members": sum(
                        1 for m in analysis["renewal_members"]
                        if 60 <= m["engagement_score"] < 80
                    ),
                    "avg_intervention_cost": 200,
                    "expected_retention_lift": 0.25,
                    "priority": "Medium"
                }
            }

            for segment, data in strategies.items():
                total_cost = data["target_members"] * data["avg_intervention_cost"]
                retained_members = round(
                    data["target_members"] * data["expected_retention_lift"]
                )
                avg_member_value = (
                    sum(m["lifetime_value"] for m in analysis["renewal_members"]) /
                    len(analysis["renewal_members"])
                    if analysis["renewal_members"] else 0
                )
                roi = (
                    (retained_members * avg_member_value - total_cost) / total_cost
                    if total_cost > 0 else 0
                )

                analysis["intervention_strategies"][segment] = {
                    "target_members": data["target_members"],
                    "avg_intervention_cost": data["avg_intervention_cost"],
                    "total_campaign_cost": total_cost,
                    "expected_retention_lift": data["expected_retention_lift"],
                    "expected_retained_members": retained_members,
                    "roi": round(roi, 2),
                    "priority": data["priority"]
                }

            total_members = len(analysis["renewal_members"])
            high_risk_count = sum(
                1 for m in analysis["renewal_members"]
                if m["engagement_score"] < 60
            )

            analysis["summary"] = {
                "total_renewal_members": total_members,
                "high_risk_members": high_risk_count,
                "high_risk_percentage": round(
                    high_risk_count / max(total_members, 1) * 100, 2
                ) if total_members > 0 else 0,
                "total_member_lifetime_value": round(
                    sum(m["lifetime_value"] for m in analysis["renewal_members"]),
                    2
                ),
                "recommended_campaign_focus": "High Risk segment",
                "estimated_campaign_roi": analysis["intervention_strategies"].get(
                    "High Risk", {}
                ).get("roi", 0)
            }

            logger.info(f"Retention targeting complete: {high_risk_count} high-risk members identified")
            return analysis

        except Exception as e:
            logger.error(f"Error in retention targeting: {e}")
            raise

    def get_revenue_dashboard(self) -> Dict[str, Any]:
        logger.info("Starting revenue dashboard generation")

        dashboard = {
            "dashboard_date": datetime.now().isoformat(),
            "revenue_opportunities": {},
            "total_opportunity": 0,
            "priority_actions": [],
            "implementation_roadmap": [],
            "key_metrics": {},
            "executive_summary": ""
        }

        try:
            churn_analysis = self.get_churn_risk_analysis()
            pmpm_analysis = self.get_pmpm_optimization()
            denial_analysis = self.get_denial_recovery()
            hcc_analysis = self.get_hcc_gap_closure()
            retention_analysis = self.get_retention_targeting()

            denial_recovery_value = denial_analysis.get("summary", {}).get(
                "recovery_potential", 0
            )
            hcc_uplift_value = hcc_analysis.get("summary", {}).get(
                "total_annual_revenue_opportunity", 0
            )
            pmpm_savings_value = pmpm_analysis.get("summary", {}).get(
                "total_cost_reduction_opportunity", 0
            )
            retention_value = (
                retention_analysis.get("summary", {}).get(
                    "total_member_lifetime_value", 0
                ) * 0.10
            )

            dashboard["revenue_opportunities"] = {
                "denial_recovery": {
                    "opportunity": "Claims Denial Recovery",
                    "value": round(denial_recovery_value, 2),
                    "description": "Resubmit and appeal denied claims",
                    "timeline_days": 30,
                    "effort_level": "Medium",
                    "members_affected": denial_analysis.get("denial_overview", {}).get(
                        "denied_claims", 0
                    ),
                    "confidence_level": 0.60
                },
                "hcc_gaps": {
                    "opportunity": "HCC Risk Adjustment Gap Closure",
                    "value": round(hcc_uplift_value, 2),
                    "description": "Complete documentation and coding of HCC conditions",
                    "timeline_days": 60,
                    "effort_level": "High",
                    "members_affected": hcc_analysis.get("summary", {}).get(
                        "members_with_gaps", 0
                    ),
                    "confidence_level": 0.75
                },
                "pmpm_reduction": {
                    "opportunity": "PMPM Cost Optimization",
                    "value": round(pmpm_savings_value, 2),
                    "description": "Preventable ED visits, readmission prevention, pharmacy optimization",
                    "timeline_days": 90,
                    "effort_level": "High",
                    "members_affected": (
                        pmpm_analysis.get("summary", {}).get(
                            "high_cost_members_count", 0
                        )
                    ),
                    "confidence_level": 0.50
                },
                "retention": {
                    "opportunity": "Member Retention & Renewal",
                    "value": round(retention_value, 2),
                    "description": "Prevent disenrollment of at-risk members",
                    "timeline_days": 45,
                    "effort_level": "Medium",
                    "members_affected": retention_analysis.get("summary", {}).get(
                        "high_risk_members", 0
                    ),
                    "confidence_level": 0.65
                }
            }

            dashboard["total_opportunity"] = round(
                sum(
                    opp["value"]
                    for opp in dashboard["revenue_opportunities"].values()
                ),
                2
            )

            priority_actions = [
                {
                    "rank": 1,
                    "action": "Denial Management Program",
                    "value": denial_recovery_value,
                    "effort": "Medium",
                    "timeline_weeks": 4,
                    "roi": round(denial_recovery_value / 50000, 2) if denial_recovery_value > 0 else 0,
                    "owner": "Revenue Cycle",
                    "status": "Ready to Start"
                },
                {
                    "rank": 2,
                    "action": "HCC Documentation Improvement",
                    "value": hcc_uplift_value,
                    "effort": "High",
                    "timeline_weeks": 8,
                    "roi": round(hcc_uplift_value / 75000, 2) if hcc_uplift_value > 0 else 0,
                    "owner": "Clinical Documentation",
                    "status": "Planning"
                },
                {
                    "rank": 3,
                    "action": "High-Risk Member Retention Campaign",
                    "value": retention_value,
                    "effort": "Medium",
                    "timeline_weeks": 6,
                    "roi": round(
                        retention_analysis.get("intervention_strategies", {}).get(
                            "High Risk", {}
                        ).get("roi", 0),
                        2
                    ),
                    "owner": "Member Services",
                    "status": "Ready to Start"
                },
                {
                    "rank": 4,
                    "action": "PMPM Reduction Initiatives",
                    "value": pmpm_savings_value,
                    "effort": "High",
                    "timeline_weeks": 12,
                    "roi": round(pmpm_savings_value / 100000, 2) if pmpm_savings_value > 0 else 0,
                    "owner": "Medical Management",
                    "status": "Planning"
                }
            ]

            priority_actions.sort(key=lambda x: x["value"], reverse=True)
            dashboard["priority_actions"] = priority_actions

            dashboard["implementation_roadmap"] = [
                {
                    "phase": 1,
                    "name": "Quick Wins (Weeks 1-4)",
                    "initiatives": [
                        "Launch denial management program",
                        "Identify high-cost members for intervention"
                    ],
                    "expected_value": round(denial_recovery_value * 0.5, 2),
                    "resources_required": 5
                },
                {
                    "phase": 2,
                    "name": "Member Engagement (Weeks 5-8)",
                    "initiatives": [
                        "Execute retention campaign for high-risk members",
                        "Launch grievance resolution follow-up"
                    ],
                    "expected_value": round(retention_value * 0.7, 2),
                    "resources_required": 8
                },
                {
                    "phase": 3,
                    "name": "Structural Improvements (Weeks 9-16)",
                    "initiatives": [
                        "Implement HCC documentation review process",
                        "Deploy PMPM reduction initiatives",
                        "Provider education programs"
                    ],
                    "expected_value": round(
                        (hcc_uplift_value + pmpm_savings_value) * 0.6, 2
                    ),
                    "resources_required": 12
                },
                {
                    "phase": 4,
                    "name": "Optimization & Scaling (Weeks 17-26)",
                    "initiatives": [
                        "Refine and scale successful programs",
                        "Monitor KPIs and adjust strategies",
                        "Plan next generation improvements"
                    ],
                    "expected_value": round(dashboard["total_opportunity"] * 0.15, 2),
                    "resources_required": 6
                }
            ]

            at_risk_members = churn_analysis.get("summary", {}).get(
                "at_risk_members", 0
            )
            total_members = churn_analysis.get("total_members_analyzed", 1)

            dashboard["key_metrics"] = {
                "total_active_members": total_members,
                "churn_risk_percentage": round(
                    at_risk_members / max(total_members, 1) * 100, 2
                ) if total_members > 0 else 0,
                "current_pmpm": pmpm_analysis.get("summary", {}).get(
                    "current_average_pmpm", 0
                ),
                "denial_rate": denial_analysis.get("denial_overview", {}).get(
                    "denial_rate_percentage", 0
                ),
                "high_cost_members": pmpm_analysis.get("summary", {}).get(
                    "high_cost_members_count", 0
                ),
                "hcc_gap_members": hcc_analysis.get("summary", {}).get(
                    "members_with_gaps", 0
                )
            }

            dashboard["executive_summary"] = (
                f"Total revenue opportunity identified: ${dashboard['total_opportunity']:,.0f} "
                f"across {len([o for o in dashboard['revenue_opportunities'].values() if o['value'] > 0])} "
                f"initiatives. Highest ROI: Denial recovery ({priority_actions[0]['roi']:.2f}x). "
                f"Critical action required for {at_risk_members:,} at-risk members "
                f"({dashboard['key_metrics']['churn_risk_percentage']:.1f}% of membership). "
                f"Recommended phased implementation over 26 weeks with expected run-rate "
                f"improvement of ${dashboard['total_opportunity'] / 52:,.0f}/week."
            )

            logger.info(f"Revenue dashboard complete: ${dashboard['total_opportunity']:,.0f} opportunity")
            return dashboard

        except Exception as e:
            logger.error(f"Error generating revenue dashboard: {e}")
            raise

    def _estimate_revenue_at_risk(self, at_risk_member_count: int) -> float:
        avg_pmpm = self._get_scalar("""
        SELECT ROUND(
            SUM(CAST(c.PAID_AMOUNT AS REAL)) /
            NULLIF(COUNT(DISTINCT m.MEMBER_ID) * 12, 0),
            2
        )
        FROM members m
        LEFT JOIN claims c ON m.MEMBER_ID = c.MEMBER_ID
        WHERE (m.DISENROLLMENT_DATE IS NULL OR m.DISENROLLMENT_DATE = '')
        """) or 0

        return round(at_risk_member_count * avg_pmpm * 12, 2)

    def close(self):
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    """Example usage of the RevenueOptimizationEngine."""


    pass
