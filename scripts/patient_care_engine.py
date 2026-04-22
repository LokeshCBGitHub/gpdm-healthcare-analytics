import sqlite3
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger('gpdm.patient_care')


class PatientCareEngine:

    def __init__(self, db_path: str):
        self.db_path = db_path

    def _query(self, sql: str, params: tuple = ()) -> List[Any]:
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute(sql, params)
            results = c.fetchall()
            conn.close()
            return results
        except Exception as e:
            logger.error(f"Patient care query error: {e}")
            return []

    def _query_one(self, sql: str, params: tuple = ()) -> Any:
        results = self._query(sql, params)
        return results[0] if results else None

    def get_patient_360(self, member_id: str) -> Dict[str, Any]:
        result = {'member_id': member_id, 'found': False}

        demo = self._query_one('''
            SELECT MRN, FIRST_NAME, LAST_NAME, DATE_OF_BIRTH, GENDER, RACE,
                   LANGUAGE, KP_REGION, FACILITY, PLAN_TYPE, ENROLLMENT_DATE,
                   DISENROLLMENT_DATE, PCP_NPI, RISK_SCORE, CHRONIC_CONDITIONS
            FROM members WHERE MEMBER_ID = ?
        ''', (member_id,))

        if not demo:
            return result

        result['found'] = True
        result['demographics'] = {
            'mrn': demo[0],
            'name': f"{demo[1]} {demo[2]}",
            'dob': demo[3],
            'gender': demo[4],
            'race': demo[5],
            'language': demo[6],
            'region': demo[7],
            'facility': demo[8],
            'plan_type': demo[9],
            'enrolled_since': demo[10],
            'active': demo[11] == '' or demo[11] is None,
            'pcp_npi': demo[12],
            'risk_score': float(demo[13]) if demo[13] else 0,
            'chronic_conditions': int(demo[14]) if demo[14] else 0,
        }

        risk = float(demo[13]) if demo[13] else 0
        result['risk_assessment'] = {
            'score': risk,
            'tier': 'Very High' if risk >= 1.5 else 'High' if risk >= 1.0 else 'Moderate' if risk >= 0.5 else 'Low',
            'management_level': self._get_management_level(risk),
            'chronic_conditions': int(demo[14]) if demo[14] else 0,
            'cost_impact': self._estimate_cost_impact(risk, int(demo[14]) if demo[14] else 0)
        }

        diagnoses = self._query('''
            SELECT ICD10_CODE, ICD10_DESCRIPTION, DIAGNOSIS_TYPE, DIAGNOSIS_DATE,
                   IS_CHRONIC, SEVERITY, HCC_CODE, HCC_CATEGORY
            FROM diagnoses WHERE MEMBER_ID = ?
            ORDER BY DIAGNOSIS_DATE DESC
        ''', (member_id,))

        result['diagnoses'] = [
            {
                'code': d[0],
                'description': d[1],
                'type': d[2],
                'date': d[3],
                'is_chronic': d[4] == 'Y',
                'severity': d[5],
                'hcc_code': d[6],
                'hcc_category': d[7],
                'follow_up_needed': d[5] in ('SEVERE', 'CRITICAL') or d[4] == 'Y'
            }
            for d in diagnoses
        ]

        encounters = self._query('''
            SELECT ENCOUNTER_ID, SERVICE_DATE, VISIT_TYPE, DEPARTMENT, FACILITY,
                   PRIMARY_DIAGNOSIS, DIAGNOSIS_DESCRIPTION, CHIEF_COMPLAINT,
                   DISPOSITION, LENGTH_OF_STAY
            FROM encounters WHERE MEMBER_ID = ?
            ORDER BY SERVICE_DATE DESC LIMIT 20
        ''', (member_id,))

        result['encounters'] = [
            {
                'id': e[0],
                'date': e[1],
                'type': e[2],
                'department': e[3],
                'facility': e[4],
                'diagnosis': f"{e[5]} - {e[6]}",
                'complaint': e[7],
                'disposition': e[8],
                'los': int(e[9]) if e[9] else 0
            }
            for e in encounters
        ]

        claims_summary = self._query_one('''
            SELECT COUNT(*),
                   SUM(CAST(BILLED_AMOUNT AS REAL)),
                   SUM(CAST(PAID_AMOUNT AS REAL)),
                   SUM(CASE WHEN CLAIM_STATUS = 'DENIED' THEN 1 ELSE 0 END),
                   SUM(CASE WHEN CLAIM_STATUS = 'DENIED' THEN CAST(BILLED_AMOUNT AS REAL) ELSE 0 END)
            FROM claims WHERE MEMBER_ID = ?
        ''', (member_id,))

        if claims_summary:
            result['claims_summary'] = {
                'total_claims': claims_summary[0],
                'total_billed': round(claims_summary[1] or 0, 2),
                'total_paid': round(claims_summary[2] or 0, 2),
                'denied_claims': claims_summary[3],
                'denied_amount': round(claims_summary[4] or 0, 2),
                'denial_rate': round(claims_summary[3] / claims_summary[0] * 100, 1) if claims_summary[0] else 0
            }

        meds = self._query('''
            SELECT MEDICATION_NAME, MEDICATION_CLASS, QUANTITY, DAYS_SUPPLY,
                   REFILLS_AUTHORIZED, REFILLS_USED, PRESCRIPTION_DATE, STATUS, COST, COPAY
            FROM prescriptions WHERE MEMBER_ID = ?
            ORDER BY PRESCRIPTION_DATE DESC
        ''', (member_id,))

        result['medications'] = [
            {
                'name': m[0],
                'class': m[1],
                'quantity': m[2],
                'days_supply': m[3],
                'refills_remaining': int(m[4] or 0) - int(m[5] or 0),
                'date': m[6],
                'status': m[7],
                'cost': float(m[8]) if m[8] else 0,
                'copay': float(m[9]) if m[9] else 0
            }
            for m in meds
        ]

        active_meds = [m for m in result['medications'] if m['status'] in ('ACTIVE', 'FILLED')]
        unique_classes = set(m['class'] for m in active_meds)
        result['polypharmacy_risk'] = {
            'active_medications': len(active_meds),
            'unique_classes': len(unique_classes),
            'risk_level': 'High' if len(active_meds) >= 5 else 'Moderate' if len(active_meds) >= 3 else 'Low',
            'note': 'Polypharmacy (5+ medications) increases adverse drug event risk by 80%' if len(active_meds) >= 5 else ''
        }

        referrals = self._query('''
            SELECT REFERRAL_DATE, REFERRAL_REASON, URGENCY, REFERRAL_TYPE,
                   STATUS, SPECIALTY, APPOINTMENT_DATE
            FROM referrals WHERE MEMBER_ID = ?
            ORDER BY REFERRAL_DATE DESC
        ''', (member_id,))

        result['referrals'] = [
            {
                'date': r[0],
                'reason': r[1],
                'urgency': r[2],
                'type': r[3],
                'status': r[4],
                'specialty': r[5],
                'appointment_date': r[6],
                'overdue': r[4] == 'PENDING' and r[0] < '2025-06-01'
            }
            for r in referrals
        ]

        result['care_alerts'] = self._generate_care_alerts(result)

        return result

    def get_high_risk_patients(self, limit: int = 50) -> Dict[str, Any]:
        patients = self._query(f'''
            SELECT m.MEMBER_ID, m.FIRST_NAME || ' ' || m.LAST_NAME as name,
                   CAST(m.RISK_SCORE AS REAL) as risk,
                   CAST(m.CHRONIC_CONDITIONS AS INTEGER) as chronic,
                   m.KP_REGION, m.PLAN_TYPE,
                   (SELECT COUNT(*) FROM claims c WHERE c.MEMBER_ID = m.MEMBER_ID) as claims,
                   (SELECT SUM(CAST(c2.PAID_AMOUNT AS REAL)) FROM claims c2 WHERE c2.MEMBER_ID = m.MEMBER_ID) as total_cost,
                   (SELECT COUNT(*) FROM encounters e WHERE e.MEMBER_ID = m.MEMBER_ID AND e.VISIT_TYPE = 'EMERGENCY') as er_visits,
                   (SELECT COUNT(*) FROM prescriptions rx WHERE rx.MEMBER_ID = m.MEMBER_ID AND rx.STATUS IN ('ACTIVE','FILLED')) as active_meds
            FROM members m
            WHERE CAST(m.RISK_SCORE AS REAL) >= 1.5
            ORDER BY CAST(m.RISK_SCORE AS REAL) DESC
            LIMIT {limit}
        ''')

        return {
            'title': 'High-Risk Patient Registry',
            'total_very_high_risk': self._query_one("SELECT COUNT(*) FROM members WHERE CAST(RISK_SCORE AS REAL) >= 1.5")[0],
            'patients': [
                {
                    'member_id': p[0],
                    'name': p[1],
                    'risk_score': round(p[2], 2),
                    'chronic_conditions': p[3],
                    'region': p[4],
                    'plan': p[5],
                    'total_claims': p[6],
                    'total_cost': round(p[7] or 0, 2),
                    'er_visits': p[8],
                    'active_medications': p[9],
                    'intervention_priority': 'CRITICAL' if p[2] >= 2.5 else 'HIGH' if p[2] >= 2.0 else 'ELEVATED',
                    'recommended_actions': self._get_patient_interventions(p[2], p[3], p[8], p[9])
                }
                for p in patients
            ]
        }

    def get_care_gaps(self) -> Dict[str, Any]:

        no_recent_visit = self._query('''
            SELECT m.MEMBER_ID, m.FIRST_NAME || ' ' || m.LAST_NAME,
                   CAST(m.RISK_SCORE AS REAL), CAST(m.CHRONIC_CONDITIONS AS INTEGER),
                   MAX(e.SERVICE_DATE) as last_visit
            FROM members m
            LEFT JOIN encounters e ON m.MEMBER_ID = e.MEMBER_ID
            WHERE CAST(m.RISK_SCORE AS REAL) >= 1.0
            GROUP BY m.MEMBER_ID
            HAVING last_visit < '2025-06-01' OR last_visit IS NULL
            ORDER BY CAST(m.RISK_SCORE AS REAL) DESC
            LIMIT 30
        ''')

        pending_referrals = self._query('''
            SELECT r.MEMBER_ID, m.FIRST_NAME || ' ' || m.LAST_NAME,
                   r.REFERRAL_REASON, r.SPECIALTY, r.URGENCY, r.REFERRAL_DATE
            FROM referrals r
            JOIN members m ON r.MEMBER_ID = m.MEMBER_ID
            WHERE r.STATUS = 'PENDING'
            ORDER BY CASE r.URGENCY WHEN 'EMERGENT' THEN 1 WHEN 'URGENT' THEN 2 ELSE 3 END,
                     r.REFERRAL_DATE
            LIMIT 30
        ''')

        medication_gaps = self._query('''
            SELECT rx.MEMBER_ID, m.FIRST_NAME || ' ' || m.LAST_NAME,
                   rx.MEDICATION_NAME, rx.MEDICATION_CLASS,
                   CAST(rx.REFILLS_AUTHORIZED AS INTEGER) - CAST(rx.REFILLS_USED AS INTEGER) as remaining
            FROM prescriptions rx
            JOIN members m ON rx.MEMBER_ID = m.MEMBER_ID
            WHERE rx.STATUS = 'ACTIVE'
            AND CAST(rx.REFILLS_AUTHORIZED AS INTEGER) - CAST(rx.REFILLS_USED AS INTEGER) <= 0
            AND rx.MEDICATION_CLASS IN ('Blood Pressure', 'Diabetes', 'Cholesterol', 'Heart', 'Thyroid')
            LIMIT 30
        ''')

        return {
            'title': 'Population Care Gap Analysis',
            'high_risk_no_visit': {
                'title': 'High-Risk Members Without Recent Visit',
                'count': len(no_recent_visit),
                'action': 'Outreach and scheduling needed — these members need care management contact',
                'patients': [
                    {
                        'member_id': p[0],
                        'name': p[1],
                        'risk_score': round(p[2], 2),
                        'chronic_conditions': p[3],
                        'last_visit': p[4] or 'Never'
                    }
                    for p in no_recent_visit
                ]
            },
            'pending_referrals': {
                'title': 'Incomplete Referrals',
                'count': len(pending_referrals),
                'action': 'Care coordination follow-up — ensure specialist appointments are scheduled',
                'referrals': [
                    {
                        'member_id': r[0],
                        'name': r[1],
                        'reason': r[2],
                        'specialty': r[3],
                        'urgency': r[4],
                        'date': r[5]
                    }
                    for r in pending_referrals
                ]
            },
            'medication_gaps': {
                'title': 'Chronic Medication Adherence Gaps',
                'count': len(medication_gaps),
                'action': 'Pharmacy outreach — refill authorization and adherence counseling needed',
                'patients': [
                    {
                        'member_id': m[0],
                        'name': m[1],
                        'medication': m[2],
                        'class': m[3],
                        'refills_remaining': m[4]
                    }
                    for m in medication_gaps
                ]
            }
        }

    def get_chronic_disease_dashboard(self) -> Dict[str, Any]:
        disease_stats = self._query('''
            SELECT
                CASE
                    WHEN d.ICD10_CODE LIKE 'E11%' THEN 'Type 2 Diabetes'
                    WHEN d.ICD10_CODE LIKE 'I10%' OR d.ICD10_CODE LIKE 'I11%' THEN 'Hypertension'
                    WHEN d.ICD10_CODE LIKE 'J44%' THEN 'COPD'
                    WHEN d.ICD10_CODE LIKE 'J45%' THEN 'Asthma'
                    WHEN d.ICD10_CODE LIKE 'F32%' OR d.ICD10_CODE LIKE 'F33%' THEN 'Depression'
                    WHEN d.ICD10_CODE LIKE 'E78%' THEN 'Hyperlipidemia'
                    WHEN d.ICD10_CODE LIKE 'I50%' THEN 'Heart Failure'
                    WHEN d.ICD10_CODE LIKE 'N18%' THEN 'Chronic Kidney Disease'
                    WHEN d.ICD10_CODE LIKE 'C%' THEN 'Cancer'
                    ELSE NULL
                END as disease,
                COUNT(DISTINCT d.MEMBER_ID) as members,
                SUM(CASE WHEN d.SEVERITY IN ('SEVERE','CRITICAL') THEN 1 ELSE 0 END) as severe,
                COUNT(d.DIAGNOSIS_ID) as total_diagnoses
            FROM diagnoses d
            WHERE d.IS_CHRONIC = 'Y'
            GROUP BY disease
            HAVING disease IS NOT NULL
            ORDER BY members DESC
        ''')

        total_members = self._query_one('SELECT COUNT(*) FROM members')[0]

        return {
            'title': 'Chronic Disease Management Dashboard',
            'total_population': total_members,
            'diseases': [
                {
                    'disease': d[0],
                    'affected_members': d[1],
                    'prevalence_pct': round(d[1] / total_members * 100, 1),
                    'severe_cases': d[2],
                    'severe_pct': round(d[2] / d[3] * 100, 1) if d[3] else 0,
                    'management_priority': 'HIGH' if d[1] > total_members * 0.05 else 'MODERATE',
                    'recommended_programs': self._get_disease_programs(d[0])
                }
                for d in disease_stats
            ]
        }


    def _get_management_level(self, risk_score: float) -> str:
        if risk_score >= 2.0:
            return 'Intensive Care Management — dedicated care team, weekly touchpoints, palliative care assessment'
        elif risk_score >= 1.5:
            return 'Active Care Management — assigned coordinator, bi-weekly check-ins, medication reconciliation'
        elif risk_score >= 1.0:
            return 'Enhanced Monitoring — monthly outreach, disease management enrollment, care gap closure'
        elif risk_score >= 0.5:
            return 'Proactive Engagement — quarterly wellness checks, preventive screenings, lifestyle coaching'
        else:
            return 'Standard Care — annual wellness visit, age-appropriate screenings, immunizations'

    def _estimate_cost_impact(self, risk: float, chronic: int) -> str:
        base_pmpm = 400
        estimated_annual = base_pmpm * 12 * risk * (1 + chronic * 0.3)
        return f"Estimated annual cost: ${estimated_annual:,.0f} (based on risk score {risk:.2f} and {chronic} chronic conditions)"

    def _generate_care_alerts(self, patient_data: Dict) -> List[Dict[str, str]]:
        alerts = []
        risk = patient_data.get('risk_assessment', {}).get('score', 0)
        chronic = patient_data.get('risk_assessment', {}).get('chronic_conditions', 0)

        if risk >= 2.0:
            alerts.append({
                'severity': 'critical',
                'alert': f'Very high risk score ({risk:.2f}) — requires intensive care management',
                'action': 'Assign dedicated care coordinator, schedule comprehensive care plan review'
            })

        if chronic >= 4:
            alerts.append({
                'severity': 'high',
                'alert': f'{chronic} chronic conditions — high polypharmacy and care fragmentation risk',
                'action': 'Medication reconciliation and care coordination review needed'
            })

        severe_dx = [d for d in patient_data.get('diagnoses', []) if d.get('severity') in ('SEVERE', 'CRITICAL')]
        if severe_dx:
            alerts.append({
                'severity': 'high',
                'alert': f'{len(severe_dx)} severe/critical diagnosis(es) — active management required',
                'action': f'Review treatment plans for: {", ".join(d["description"][:50] for d in severe_dx[:3])}'
            })

        er_visits = [e for e in patient_data.get('encounters', []) if e.get('type') == 'EMERGENCY']
        if len(er_visits) >= 3:
            alerts.append({
                'severity': 'warning',
                'alert': f'{len(er_visits)} ER visits — potential for care redirection',
                'action': 'Assess for primary care access barriers, consider care management enrollment'
            })

        overdue = [r for r in patient_data.get('referrals', []) if r.get('overdue')]
        if overdue:
            alerts.append({
                'severity': 'warning',
                'alert': f'{len(overdue)} overdue referral(s) — care coordination gap',
                'action': f'Follow up on pending referrals: {", ".join(r["specialty"] for r in overdue[:3])}'
            })

        poly = patient_data.get('polypharmacy_risk', {})
        if poly.get('risk_level') == 'High':
            alerts.append({
                'severity': 'warning',
                'alert': f'{poly.get("active_medications", 0)} active medications — polypharmacy risk',
                'action': 'Schedule medication therapy management (MTM) review'
            })

        claims = patient_data.get('claims_summary', {})
        if claims.get('denial_rate', 0) > 15:
            alerts.append({
                'severity': 'info',
                'alert': f'{claims["denial_rate"]:.0f}% claim denial rate — above average',
                'action': 'Review denied claims for authorization and documentation gaps'
            })

        if not alerts:
            alerts.append({
                'severity': 'info',
                'alert': 'No critical care alerts at this time',
                'action': 'Continue standard care management'
            })

        return alerts

    def _get_patient_interventions(self, risk: float, chronic: int, er_visits: int, active_meds: int) -> List[str]:
        actions = []
        if risk >= 2.5:
            actions.append('Immediate: Assign intensive care management team')
        if risk >= 2.0:
            actions.append('Schedule comprehensive care plan review within 7 days')
        if chronic >= 4:
            actions.append('Medication therapy management (MTM) review — polypharmacy risk')
        if er_visits >= 3:
            actions.append('ER utilization review — assess for primary care alternatives')
        if active_meds >= 5:
            actions.append('Drug interaction screening and reconciliation')
        if not actions:
            actions.append('Standard care management protocols — monitor and engage proactively')
        return actions

    def _get_disease_programs(self, disease: str) -> List[str]:
        programs = {
            'Type 2 Diabetes': ['Diabetes Prevention Program (DPP)', 'A1C monitoring protocol', 'Endocrinology referral for uncontrolled', 'Nutrition counseling'],
            'Hypertension': ['Home BP monitoring program', 'Medication adherence support', 'Lifestyle modification counseling', 'Cardiology referral for resistant HTN'],
            'COPD': ['Pulmonary rehabilitation', 'Smoking cessation program', 'Action plan education', 'Pulmonology follow-up'],
            'Asthma': ['Asthma action plan', 'Inhaler technique education', 'Trigger avoidance counseling', 'Allergy testing'],
            'Depression': ['Behavioral health integration', 'PHQ-9 screening protocol', 'Medication management', 'Therapy referral'],
            'Hyperlipidemia': ['Statin therapy protocol', 'Lipid monitoring schedule', 'Diet and exercise counseling', 'Cardiovascular risk assessment'],
            'Heart Failure': ['Heart failure clinic enrollment', 'Daily weight monitoring', 'Fluid restriction education', 'Cardiology co-management'],
            'Chronic Kidney Disease': ['Nephrology co-management', 'GFR monitoring protocol', 'Diet counseling', 'Medication adjustment review'],
            'Cancer': ['Oncology care coordination', 'Treatment adherence monitoring', 'Palliative care assessment', 'Survivorship planning'],
        }
        return programs.get(disease, ['Standard chronic disease management protocol'])


_patient_care_engine = None

def get_patient_care_engine(db_path: str) -> PatientCareEngine:
    global _patient_care_engine
    if _patient_care_engine is None or _patient_care_engine.db_path != db_path:
        _patient_care_engine = PatientCareEngine(db_path)
    return _patient_care_engine
