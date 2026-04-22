import sqlite3
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger('gpdm.chronic_risk')


PRE_CHRONIC_INDICATORS = {
    'R73.01': {'predicts': 'Type 2 Diabetes', 'weight': 35, 'icd_chronic': 'E11'},
    'R73.02': {'predicts': 'Type 2 Diabetes', 'weight': 30, 'icd_chronic': 'E11'},
    'R73.03': {'predicts': 'Type 2 Diabetes', 'weight': 25, 'icd_chronic': 'E11'},
    'R73.09': {'predicts': 'Type 2 Diabetes', 'weight': 20, 'icd_chronic': 'E11'},
    'E11.65': {'predicts': 'Uncontrolled Diabetes (progression)', 'weight': 40, 'icd_chronic': 'E11'},
    'E66.01': {'predicts': 'Metabolic Syndrome', 'weight': 25, 'icd_chronic': 'E11'},
    'E66.09': {'predicts': 'Metabolic Syndrome', 'weight': 20, 'icd_chronic': 'E11'},
    'E66.1':  {'predicts': 'Metabolic Syndrome', 'weight': 15, 'icd_chronic': 'E11'},
    'R03.0':  {'predicts': 'Hypertension', 'weight': 30, 'icd_chronic': 'I10'},
    'R03.1':  {'predicts': 'Hypertension', 'weight': 15, 'icd_chronic': 'I10'},
    'E78.0':  {'predicts': 'Cardiovascular Disease', 'weight': 20, 'icd_chronic': 'I25'},
    'E78.1':  {'predicts': 'Cardiovascular Disease', 'weight': 18, 'icd_chronic': 'I25'},
    'E78.2':  {'predicts': 'Cardiovascular Disease', 'weight': 22, 'icd_chronic': 'I25'},
    'E78.5':  {'predicts': 'Cardiovascular Disease', 'weight': 25, 'icd_chronic': 'I25'},
    'F17.210': {'predicts': 'COPD / Lung Disease', 'weight': 20, 'icd_chronic': 'J44'},
    'F17.211': {'predicts': 'COPD / Lung Disease', 'weight': 15, 'icd_chronic': 'J44'},
    'Z87.891': {'predicts': 'COPD / Lung Disease', 'weight': 10, 'icd_chronic': 'J44'},
    'R05.9':  {'predicts': 'Chronic Respiratory Disease', 'weight': 8, 'icd_chronic': 'J44'},
    'R06.00': {'predicts': 'Chronic Respiratory Disease', 'weight': 12, 'icd_chronic': 'J44'},
    'J06.9':  {'predicts': 'Chronic Respiratory Disease', 'weight': 5, 'icd_chronic': 'J44'},
    'N17.9':  {'predicts': 'Chronic Kidney Disease', 'weight': 30, 'icd_chronic': 'N18'},
    'R80.9':  {'predicts': 'Chronic Kidney Disease', 'weight': 20, 'icd_chronic': 'N18'},
    'N28.9':  {'predicts': 'Chronic Kidney Disease', 'weight': 15, 'icd_chronic': 'N18'},
    'R45.89': {'predicts': 'Chronic Depression', 'weight': 15, 'icd_chronic': 'F32'},
    'F43.10': {'predicts': 'Chronic Depression', 'weight': 12, 'icd_chronic': 'F32'},
    'F41.1':  {'predicts': 'Chronic Depression', 'weight': 18, 'icd_chronic': 'F32'},
    'I25.10': {'predicts': 'Heart Failure', 'weight': 30, 'icd_chronic': 'I50'},
    'I48.91': {'predicts': 'Heart Failure', 'weight': 25, 'icd_chronic': 'I50'},
    'R00.0':  {'predicts': 'Cardiac Disease', 'weight': 10, 'icd_chronic': 'I50'},
    'R00.1':  {'predicts': 'Cardiac Disease', 'weight': 12, 'icd_chronic': 'I50'},
}

PRE_CHRONIC_MEDICATIONS = {
    'Blood Pressure': {'predicts': 'Hypertension', 'weight': 15},
    'Diabetes':       {'predicts': 'Type 2 Diabetes', 'weight': 20},
    'Cholesterol':    {'predicts': 'Cardiovascular Disease', 'weight': 12},
    'Thyroid':        {'predicts': 'Thyroid Disease', 'weight': 10},
    'Mental Health':  {'predicts': 'Chronic Mental Health', 'weight': 10},
    'Asthma':         {'predicts': 'Chronic Respiratory Disease', 'weight': 15},
}

PREVENTIVE_INTERVENTIONS = {
    'Type 2 Diabetes': {
        'programs': [
            'Diabetes Prevention Program (DPP) — 16-week lifestyle change',
            'A1C/Glucose monitoring every 3 months',
            'Nutrition counseling and meal planning',
            'Exercise prescription (150 min/week moderate activity)',
            'Weight management program if BMI > 30'
        ],
        'annual_intervention_cost': 1200,
        'annual_chronic_cost': 16750,
        'prevention_efficacy': 0.58,
        'net_savings_per_prevented': 15550,
    },
    'Hypertension': {
        'programs': [
            'DASH diet counseling and monitoring',
            'Home blood pressure monitoring program',
            'Sodium reduction education',
            'Stress management and exercise prescription',
            'Weight management if overweight'
        ],
        'annual_intervention_cost': 800,
        'annual_chronic_cost': 8500,
        'prevention_efficacy': 0.45,
        'net_savings_per_prevented': 7700,
    },
    'Cardiovascular Disease': {
        'programs': [
            'Lipid panel monitoring every 6 months',
            'Statin therapy evaluation',
            'Cardiac risk assessment (Framingham/ASCVD)',
            'Mediterranean diet counseling',
            'Smoking cessation if applicable'
        ],
        'annual_intervention_cost': 1500,
        'annual_chronic_cost': 28200,
        'prevention_efficacy': 0.35,
        'net_savings_per_prevented': 26700,
    },
    'COPD / Lung Disease': {
        'programs': [
            'Smoking cessation program (if smoker)',
            'Pulmonary function testing (spirometry)',
            'Respiratory health education',
            'Occupational exposure assessment',
            'Annual flu and pneumonia vaccination'
        ],
        'annual_intervention_cost': 900,
        'annual_chronic_cost': 14500,
        'prevention_efficacy': 0.40,
        'net_savings_per_prevented': 13600,
    },
    'Chronic Kidney Disease': {
        'programs': [
            'GFR/Creatinine monitoring every 3 months',
            'Blood pressure control optimization',
            'Diabetes management if diabetic',
            'Nephrotoxic medication review',
            'Dietary protein counseling'
        ],
        'annual_intervention_cost': 1100,
        'annual_chronic_cost': 22000,
        'prevention_efficacy': 0.30,
        'net_savings_per_prevented': 20900,
    },
    'Heart Failure': {
        'programs': [
            'Echocardiogram and cardiac workup',
            'Aggressive HTN and lipid management',
            'BNP monitoring protocol',
            'Fluid and sodium restriction education',
            'Cardiology referral for structural assessment'
        ],
        'annual_intervention_cost': 2000,
        'annual_chronic_cost': 35000,
        'prevention_efficacy': 0.25,
        'net_savings_per_prevented': 33000,
    },
    'Chronic Depression': {
        'programs': [
            'PHQ-9 screening protocol (quarterly)',
            'Behavioral health integration',
            'Cognitive behavioral therapy referral',
            'Social determinant of health assessment',
            'Peer support program enrollment'
        ],
        'annual_intervention_cost': 1500,
        'annual_chronic_cost': 10800,
        'prevention_efficacy': 0.40,
        'net_savings_per_prevented': 9300,
    },
    'Chronic Respiratory Disease': {
        'programs': [
            'Pulmonary function baseline testing',
            'Allergen/trigger identification',
            'Inhaler education and action plan',
            'Environmental assessment',
            'Annual respiratory health check'
        ],
        'annual_intervention_cost': 700,
        'annual_chronic_cost': 9200,
        'prevention_efficacy': 0.45,
        'net_savings_per_prevented': 8500,
    },
    'Metabolic Syndrome': {
        'programs': [
            'Comprehensive metabolic panel monitoring',
            'Weight management program',
            'Exercise prescription',
            'Nutrition counseling',
            'Annual cardiovascular risk assessment'
        ],
        'annual_intervention_cost': 1000,
        'annual_chronic_cost': 12000,
        'prevention_efficacy': 0.50,
        'net_savings_per_prevented': 11000,
    },
}


class ChronicRiskPredictor:

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
            logger.error(f"Chronic risk query error: {e}")
            return []

    def _query_one(self, sql: str, params: tuple = ()) -> Any:
        results = self._query(sql, params)
        return results[0] if results else None

    def score_member(self, member_id: str) -> Dict[str, Any]:
        result = {
            'member_id': member_id,
            'risk_factors': [],
            'predicted_conditions': {},
            'composite_score': 0,
            'risk_tier': 'LOW',
            'interventions': [],
            'cost_projection': {}
        }

        member = self._query_one('''
            SELECT MEMBER_ID, FIRST_NAME, LAST_NAME, DATE_OF_BIRTH, GENDER,
                   PLAN_TYPE, KP_REGION, RISK_SCORE, CHRONIC_CONDITIONS,
                   ENROLLMENT_DATE, PCP_NPI
            FROM members WHERE MEMBER_ID = ?
        ''', (member_id,))

        if not member:
            return result

        result['name'] = f"{member[1]} {member[2]}"
        result['dob'] = member[3]
        result['gender'] = member[4]
        result['plan_type'] = member[5]
        result['region'] = member[6]
        current_risk = float(member[7]) if member[7] else 0
        current_chronic = int(member[8]) if member[8] else 0

        score = 0.0
        condition_scores = {}

        age = self._calc_age(member[3])
        result['age'] = age
        if age >= 65:
            score += 15
            result['risk_factors'].append({
                'factor': 'Advanced Age',
                'detail': f'Age {age} — significantly elevated chronic disease risk',
                'points': 15
            })
        elif age >= 50:
            score += 10
            result['risk_factors'].append({
                'factor': 'Age Risk',
                'detail': f'Age {age} — entering higher risk bracket for chronic conditions',
                'points': 10
            })
        elif age >= 40:
            score += 5
            result['risk_factors'].append({
                'factor': 'Age Factor',
                'detail': f'Age {age} — screening age for metabolic conditions',
                'points': 5
            })

        diagnoses = self._query('''
            SELECT ICD10_CODE, ICD10_DESCRIPTION, DIAGNOSIS_DATE, SEVERITY, IS_CHRONIC
            FROM diagnoses WHERE MEMBER_ID = ?
        ''', (member_id,))

        existing_chronic_codes = set()
        for d in diagnoses:
            if d[4] == 'Y':
                existing_chronic_codes.add(d[0][:3])

        for d in diagnoses:
            code = d[0]
            if code in PRE_CHRONIC_INDICATORS:
                indicator = PRE_CHRONIC_INDICATORS[code]
                predicted = indicator['predicts']
                chronic_prefix = indicator['icd_chronic']

                if chronic_prefix not in existing_chronic_codes:
                    weight = indicator['weight']
                    score += weight
                    condition_scores[predicted] = condition_scores.get(predicted, 0) + weight
                    result['risk_factors'].append({
                        'factor': f'Pre-Chronic Indicator: {code}',
                        'detail': f'{d[1]} — predicts progression to {predicted}',
                        'points': weight,
                        'diagnosis_date': d[2]
                    })

        er_count = self._query_one('''
            SELECT COUNT(*) FROM encounters
            WHERE MEMBER_ID = ? AND VISIT_TYPE = 'EMERGENCY'
        ''', (member_id,))
        er_visits = er_count[0] if er_count else 0

        if er_visits >= 3:
            pts = min(er_visits * 5, 20)
            score += pts
            result['risk_factors'].append({
                'factor': 'High ER Utilization',
                'detail': f'{er_visits} ER visits — indicates unmanaged conditions or access barriers',
                'points': pts
            })
        elif er_visits >= 2:
            score += 5
            result['risk_factors'].append({
                'factor': 'Elevated ER Utilization',
                'detail': f'{er_visits} ER visits — monitor for unmanaged conditions',
                'points': 5
            })

        claim_stats = self._query_one('''
            SELECT COUNT(*), SUM(CAST(BILLED_AMOUNT AS REAL)),
                   SUM(CAST(PAID_AMOUNT AS REAL))
            FROM claims WHERE MEMBER_ID = ?
        ''', (member_id,))

        total_claims = claim_stats[0] if claim_stats else 0
        total_billed = claim_stats[1] if claim_stats and claim_stats[1] else 0
        total_paid = claim_stats[2] if claim_stats and claim_stats[2] else 0

        if total_claims >= 8 and current_chronic == 0:
            score += 12
            result['risk_factors'].append({
                'factor': 'High Utilization Without Chronic Dx',
                'detail': f'{total_claims} claims but 0 chronic conditions — possible undiagnosed chronic disease',
                'points': 12
            })

        meds = self._query('''
            SELECT MEDICATION_CLASS, COUNT(*) as cnt
            FROM prescriptions
            WHERE MEMBER_ID = ? AND STATUS IN ('ACTIVE', 'FILLED')
            GROUP BY MEDICATION_CLASS
        ''', (member_id,))

        for med_class, cnt in meds:
            if med_class in PRE_CHRONIC_MEDICATIONS:
                predicted = PRE_CHRONIC_MEDICATIONS[med_class]['predicts']
                chronic_prefix = {'Hypertension': 'I10', 'Type 2 Diabetes': 'E11',
                                  'Cardiovascular Disease': 'I25', 'Thyroid Disease': 'E03',
                                  'Chronic Mental Health': 'F32', 'Chronic Respiratory Disease': 'J44'
                                  }.get(predicted, '')

                if chronic_prefix and chronic_prefix not in existing_chronic_codes:
                    weight = PRE_CHRONIC_MEDICATIONS[med_class]['weight']
                    score += weight
                    condition_scores[predicted] = condition_scores.get(predicted, 0) + weight
                    result['risk_factors'].append({
                        'factor': f'Medication Signal: {med_class}',
                        'detail': f'On {med_class} medication(s) without formal chronic diagnosis — '
                                  f'suggests emerging {predicted}',
                        'points': weight
                    })

        has_pcp = member[10] and member[10] != ''
        if not has_pcp:
            score += 8
            result['risk_factors'].append({
                'factor': 'No Assigned PCP',
                'detail': 'No primary care physician — preventive care and early detection at risk',
                'points': 8
            })

        last_visit = self._query_one('''
            SELECT MAX(SERVICE_DATE) FROM encounters WHERE MEMBER_ID = ?
        ''', (member_id,))

        if last_visit and last_visit[0]:
            try:
                last_dt = datetime.strptime(last_visit[0], '%Y-%m-%d')
                days_since = (datetime.now() - last_dt).days
                if days_since > 365:
                    pts = min(int(days_since / 180) * 3, 15)
                    score += pts
                    result['risk_factors'].append({
                        'factor': 'Care Gap — No Recent Visit',
                        'detail': f'Last visit {days_since} days ago — risk of undetected disease progression',
                        'points': pts
                    })
            except (ValueError, TypeError):
                pass

        acute_conditions = len([d for d in diagnoses if d[4] == 'N'])
        if acute_conditions >= 4 and current_chronic <= 1:
            score += 10
            result['risk_factors'].append({
                'factor': 'Multi-Morbidity Loading',
                'detail': f'{acute_conditions} acute diagnoses with only {current_chronic} chronic — '
                          f'accumulating toward chronic threshold',
                'points': 10
            })

        if member[5] == 'Medicare Advantage':
            score += 8
            result['risk_factors'].append({
                'factor': 'Medicare Advantage Population',
                'detail': 'MA members have 2-3x chronic disease rate vs commercial — heightened surveillance',
                'points': 8
            })
        elif member[5] == 'Medicaid':
            score += 6
            result['risk_factors'].append({
                'factor': 'Medicaid Population',
                'detail': 'Medicaid population faces SDOH barriers that increase chronic disease risk',
                'points': 6
            })

        composite = min(round(score, 1), 100)
        result['composite_score'] = composite

        if composite >= 70:
            result['risk_tier'] = 'CRITICAL'
        elif composite >= 50:
            result['risk_tier'] = 'HIGH'
        elif composite >= 30:
            result['risk_tier'] = 'MODERATE'
        else:
            result['risk_tier'] = 'LOW'

        for condition, cond_score in sorted(condition_scores.items(), key=lambda x: -x[1]):
            intervention = PREVENTIVE_INTERVENTIONS.get(condition, {})
            result['predicted_conditions'][condition] = {
                'risk_points': cond_score,
                'likelihood': 'High' if cond_score >= 30 else 'Moderate' if cond_score >= 15 else 'Possible',
                'programs': intervention.get('programs', []),
                'annual_intervention_cost': intervention.get('annual_intervention_cost', 0),
                'annual_chronic_cost': intervention.get('annual_chronic_cost', 0),
                'prevention_efficacy': f"{intervention.get('prevention_efficacy', 0) * 100:.0f}%",
                'net_savings_if_prevented': intervention.get('net_savings_per_prevented', 0)
            }

        for condition in list(result['predicted_conditions'].keys())[:3]:
            programs = PREVENTIVE_INTERVENTIONS.get(condition, {}).get('programs', [])
            for prog in programs[:2]:
                result['interventions'].append({
                    'condition': condition,
                    'program': prog,
                    'priority': 'IMMEDIATE' if result['risk_tier'] in ('CRITICAL', 'HIGH') else 'SCHEDULED'
                })

        total_intervention_cost = sum(
            PREVENTIVE_INTERVENTIONS.get(c, {}).get('annual_intervention_cost', 0)
            for c in result['predicted_conditions']
        )
        total_chronic_cost = sum(
            PREVENTIVE_INTERVENTIONS.get(c, {}).get('annual_chronic_cost', 0)
            for c in result['predicted_conditions']
        )
        avg_efficacy = 0
        if result['predicted_conditions']:
            avg_efficacy = sum(
                PREVENTIVE_INTERVENTIONS.get(c, {}).get('prevention_efficacy', 0)
                for c in result['predicted_conditions']
            ) / len(result['predicted_conditions'])

        result['cost_projection'] = {
            'annual_intervention_cost': total_intervention_cost,
            'annual_cost_if_chronic': total_chronic_cost,
            'expected_savings': round(total_chronic_cost * avg_efficacy - total_intervention_cost, 2),
            'roi_ratio': round((total_chronic_cost * avg_efficacy) / total_intervention_cost, 1) if total_intervention_cost > 0 else 0,
            'note': f'Investing ${total_intervention_cost:,} in prevention could avoid up to '
                    f'${total_chronic_cost:,}/year in chronic disease costs '
                    f'(avg {avg_efficacy * 100:.0f}% prevention efficacy)'
        }

        return result

    def get_at_risk_population(self, min_score: int = 30, limit: int = 100) -> Dict[str, Any]:
        candidates = self._query('''
            SELECT m.MEMBER_ID,
                   m.FIRST_NAME || ' ' || m.LAST_NAME as name,
                   m.DATE_OF_BIRTH,
                   m.GENDER,
                   m.PLAN_TYPE,
                   m.KP_REGION,
                   CAST(m.RISK_SCORE AS REAL) as current_risk,
                   CAST(m.CHRONIC_CONDITIONS AS INTEGER) as chronic_count,
                   (SELECT COUNT(*) FROM diagnoses d WHERE d.MEMBER_ID = m.MEMBER_ID) as dx_count,
                   (SELECT COUNT(*) FROM encounters e WHERE e.MEMBER_ID = m.MEMBER_ID
                    AND e.VISIT_TYPE = 'EMERGENCY') as er_visits,
                   (SELECT COUNT(DISTINCT rx.MEDICATION_CLASS) FROM prescriptions rx
                    WHERE rx.MEMBER_ID = m.MEMBER_ID AND rx.STATUS IN ('ACTIVE','FILLED')) as med_classes
            FROM members m
            WHERE CAST(m.RISK_SCORE AS REAL) < 1.5
            ORDER BY CAST(m.RISK_SCORE AS REAL) DESC, dx_count DESC
            LIMIT 500
        ''')

        scored_members = []
        total_intervention_cost = 0
        total_potential_savings = 0
        condition_counts = {}

        for c in candidates:
            member_score = self._quick_score(c)
            if member_score['composite_score'] >= min_score:
                scored_members.append(member_score)
                total_intervention_cost += member_score.get('total_intervention_cost', 0)
                total_potential_savings += member_score.get('potential_savings', 0)
                for cond in member_score.get('predicted_conditions', []):
                    condition_counts[cond] = condition_counts.get(cond, 0) + 1

        scored_members.sort(key=lambda x: -x['composite_score'])
        scored_members = scored_members[:limit]

        critical = len([m for m in scored_members if m['risk_tier'] == 'CRITICAL'])
        high = len([m for m in scored_members if m['risk_tier'] == 'HIGH'])
        moderate = len([m for m in scored_members if m['risk_tier'] == 'MODERATE'])

        return {
            'title': 'Pre-Chronic Risk Population Report',
            'description': 'Members not yet chronic who show early warning signs — intervention candidates',
            'summary': {
                'total_at_risk': len(scored_members),
                'critical_risk': critical,
                'high_risk': high,
                'moderate_risk': moderate,
                'total_annual_intervention_cost': round(total_intervention_cost, 2),
                'total_annual_potential_savings': round(total_potential_savings, 2),
                'roi_ratio': round(total_potential_savings / total_intervention_cost, 1) if total_intervention_cost > 0 else 0,
                'top_predicted_conditions': sorted(condition_counts.items(), key=lambda x: -x[1])[:8]
            },
            'patients': scored_members,
            'investment_case': self._build_investment_case(scored_members, condition_counts,
                                                           total_intervention_cost, total_potential_savings)
        }

    def _quick_score(self, member_row: tuple) -> Dict[str, Any]:
        mid, name, dob, gender, plan_type, region, current_risk, chronic, dx_count, er_visits, med_classes = member_row
        score = 0.0
        factors = []
        predicted_conditions = []

        age = self._calc_age(dob)
        if age >= 65:
            score += 15
            factors.append('Advanced age (65+)')
        elif age >= 50:
            score += 10
            factors.append('Age 50-64')
        elif age >= 40:
            score += 5

        pre_chronic = self._query('''
            SELECT d.ICD10_CODE, d.ICD10_DESCRIPTION
            FROM diagnoses d
            WHERE d.MEMBER_ID = ? AND d.ICD10_CODE IN ({})
        '''.format(','.join(f"'{c}'" for c in PRE_CHRONIC_INDICATORS.keys())), (mid,))

        existing_chronic = set()
        chronic_dx = self._query('''
            SELECT SUBSTR(ICD10_CODE, 1, 3) FROM diagnoses
            WHERE MEMBER_ID = ? AND IS_CHRONIC = 'Y'
        ''', (mid,))
        for cd in chronic_dx:
            existing_chronic.add(cd[0])

        for pc in pre_chronic:
            code = pc[0]
            if code in PRE_CHRONIC_INDICATORS:
                ind = PRE_CHRONIC_INDICATORS[code]
                if ind['icd_chronic'] not in existing_chronic:
                    score += ind['weight']
                    if ind['predicts'] not in predicted_conditions:
                        predicted_conditions.append(ind['predicts'])
                    factors.append(f'{code}: {ind["predicts"]}')

        if er_visits >= 3:
            score += min(er_visits * 5, 20)
            factors.append(f'{er_visits} ER visits')
        elif er_visits >= 2:
            score += 5

        if dx_count >= 4 and chronic == 0:
            score += 10
            factors.append(f'{dx_count} diagnoses, 0 chronic')

        if med_classes >= 3 and chronic <= 1:
            score += 8
            factors.append(f'{med_classes} med classes, {chronic} chronic dx')

        if plan_type == 'Medicare Advantage':
            score += 8
            factors.append('Medicare Advantage population')
        elif plan_type == 'Medicaid':
            score += 6
            factors.append('Medicaid — SDOH risk')

        if current_risk >= 0.8 and current_risk < 1.5 and len(factors) >= 3:
            score += 8
            factors.append('Rising risk trajectory')

        composite = min(round(score, 1), 100)

        if composite >= 70:
            tier = 'CRITICAL'
        elif composite >= 50:
            tier = 'HIGH'
        elif composite >= 30:
            tier = 'MODERATE'
        else:
            tier = 'LOW'

        total_intervention = sum(
            PREVENTIVE_INTERVENTIONS.get(c, {}).get('annual_intervention_cost', 0)
            for c in predicted_conditions
        )
        total_chronic_cost = sum(
            PREVENTIVE_INTERVENTIONS.get(c, {}).get('annual_chronic_cost', 0)
            for c in predicted_conditions
        )
        avg_eff = 0
        if predicted_conditions:
            avg_eff = sum(
                PREVENTIVE_INTERVENTIONS.get(c, {}).get('prevention_efficacy', 0)
                for c in predicted_conditions
            ) / len(predicted_conditions)

        return {
            'member_id': mid,
            'name': name,
            'age': age,
            'gender': gender,
            'plan_type': plan_type,
            'region': region,
            'current_risk_score': current_risk,
            'current_chronic_conditions': chronic,
            'composite_score': composite,
            'risk_tier': tier,
            'risk_factors': factors,
            'predicted_conditions': predicted_conditions,
            'total_intervention_cost': total_intervention,
            'potential_savings': round(total_chronic_cost * avg_eff - total_intervention, 2) if total_intervention > 0 else 0
        }

    def get_diabetes_risk_cohort(self) -> Dict[str, Any]:
        return self._get_condition_cohort(
            condition='Type 2 Diabetes',
            indicator_codes=['R73.01', 'R73.02', 'R73.03', 'R73.09', 'E66.01', 'E66.09', 'E66.1'],
            chronic_prefix='E11',
            med_class='Diabetes'
        )

    def get_cardiac_risk_cohort(self) -> Dict[str, Any]:
        return self._get_condition_cohort(
            condition='Cardiovascular Disease / Heart Failure',
            indicator_codes=['E78.0', 'E78.1', 'E78.2', 'E78.5', 'I25.10', 'I48.91', 'R03.0', 'R00.0', 'R00.1'],
            chronic_prefix='I50',
            med_class='Cholesterol'
        )

    def get_ckd_risk_cohort(self) -> Dict[str, Any]:
        return self._get_condition_cohort(
            condition='Chronic Kidney Disease',
            indicator_codes=['N17.9', 'R80.9', 'N28.9'],
            chronic_prefix='N18',
            med_class=None
        )

    def get_respiratory_risk_cohort(self) -> Dict[str, Any]:
        return self._get_condition_cohort(
            condition='COPD / Chronic Respiratory Disease',
            indicator_codes=['F17.210', 'F17.211', 'Z87.891', 'R05.9', 'R06.00', 'J06.9'],
            chronic_prefix='J44',
            med_class='Asthma'
        )

    def _get_condition_cohort(self, condition: str, indicator_codes: List[str],
                              chronic_prefix: str, med_class: Optional[str]) -> Dict[str, Any]:
        code_list = ','.join(f"'{c}'" for c in indicator_codes)

        at_risk = self._query(f'''
            SELECT DISTINCT m.MEMBER_ID,
                   m.FIRST_NAME || ' ' || m.LAST_NAME as name,
                   m.DATE_OF_BIRTH, m.GENDER, m.PLAN_TYPE, m.KP_REGION,
                   CAST(m.RISK_SCORE AS REAL),
                   d.ICD10_CODE, d.ICD10_DESCRIPTION, d.SEVERITY
            FROM members m
            JOIN diagnoses d ON m.MEMBER_ID = d.MEMBER_ID
            WHERE d.ICD10_CODE IN ({code_list})
            AND m.MEMBER_ID NOT IN (
                SELECT MEMBER_ID FROM diagnoses WHERE ICD10_CODE LIKE '{chronic_prefix}%' AND IS_CHRONIC = 'Y'
            )
            ORDER BY CAST(m.RISK_SCORE AS REAL) DESC
            LIMIT 100
        ''')

        intervention = PREVENTIVE_INTERVENTIONS.get(condition.split(' / ')[0],
                       PREVENTIVE_INTERVENTIONS.get(condition, {}))

        patients = []
        seen = set()
        for r in at_risk:
            if r[0] not in seen:
                seen.add(r[0])
                patients.append({
                    'member_id': r[0],
                    'name': r[1],
                    'age': self._calc_age(r[2]),
                    'gender': r[3],
                    'plan': r[4],
                    'region': r[5],
                    'current_risk': round(r[6], 2),
                    'indicator_code': r[7],
                    'indicator_desc': r[8],
                    'severity': r[9]
                })

        total_patients = len(patients)
        intervention_cost = total_patients * intervention.get('annual_intervention_cost', 0)
        chronic_cost = total_patients * intervention.get('annual_chronic_cost', 0)
        efficacy = intervention.get('prevention_efficacy', 0)
        prevented = round(total_patients * efficacy)
        savings = round(prevented * intervention.get('annual_chronic_cost', 0) - intervention_cost, 2)

        return {
            'title': f'{condition} — At-Risk Cohort',
            'condition': condition,
            'total_at_risk': total_patients,
            'intervention': {
                'programs': intervention.get('programs', []),
                'annual_cost_per_member': intervention.get('annual_intervention_cost', 0),
                'total_program_cost': intervention_cost,
                'chronic_cost_per_member': intervention.get('annual_chronic_cost', 0),
                'total_exposure_if_chronic': chronic_cost,
                'prevention_efficacy': f'{efficacy * 100:.0f}%',
                'expected_prevented': prevented,
                'expected_annual_savings': savings,
                'roi': round(savings / intervention_cost, 1) if intervention_cost > 0 else 0
            },
            'patients': patients
        }

    def get_preventive_care_summary(self) -> Dict[str, Any]:
        total_members = self._query_one('SELECT COUNT(*) FROM members')[0]
        current_chronic = self._query_one(
            'SELECT COUNT(*) FROM members WHERE CAST(CHRONIC_CONDITIONS AS INTEGER) > 0'
        )[0]
        very_high_risk = self._query_one(
            'SELECT COUNT(*) FROM members WHERE CAST(RISK_SCORE AS REAL) >= 1.5'
        )[0]

        chronic_member_cost = self._query_one('''
            SELECT SUM(CAST(c.PAID_AMOUNT AS REAL))
            FROM claims c
            JOIN members m ON c.MEMBER_ID = m.MEMBER_ID
            WHERE CAST(m.CHRONIC_CONDITIONS AS INTEGER) > 0
        ''')
        chronic_paid = chronic_member_cost[0] if chronic_member_cost and chronic_member_cost[0] else 0

        non_chronic_cost = self._query_one('''
            SELECT SUM(CAST(c.PAID_AMOUNT AS REAL))
            FROM claims c
            JOIN members m ON c.MEMBER_ID = m.MEMBER_ID
            WHERE CAST(m.CHRONIC_CONDITIONS AS INTEGER) = 0
        ''')
        non_chronic_paid = non_chronic_cost[0] if non_chronic_cost and non_chronic_cost[0] else 0

        chronic_pmpm = round(chronic_paid / max(current_chronic, 1) / 12, 2) if current_chronic else 0
        non_chronic_pmpm = round(non_chronic_paid / max(total_members - current_chronic, 1) / 12, 2)

        at_risk_data = self.get_at_risk_population(min_score=30, limit=200)

        condition_investment = {}
        for patient in at_risk_data.get('patients', []):
            for cond in patient.get('predicted_conditions', []):
                if cond not in condition_investment:
                    int_data = PREVENTIVE_INTERVENTIONS.get(cond, {})
                    condition_investment[cond] = {
                        'members_at_risk': 0,
                        'intervention_cost_per_member': int_data.get('annual_intervention_cost', 0),
                        'chronic_cost_per_member': int_data.get('annual_chronic_cost', 0),
                        'efficacy': int_data.get('prevention_efficacy', 0),
                    }
                condition_investment[cond]['members_at_risk'] += 1

        investment_table = []
        grand_total_invest = 0
        grand_total_savings = 0
        for cond, data in sorted(condition_investment.items(), key=lambda x: -x[1]['members_at_risk']):
            n = data['members_at_risk']
            invest = n * data['intervention_cost_per_member']
            chronic_exposure = n * data['chronic_cost_per_member']
            prevented = round(n * data['efficacy'])
            savings = round(prevented * data['chronic_cost_per_member'] - invest, 2)
            grand_total_invest += invest
            grand_total_savings += savings

            investment_table.append({
                'condition': cond,
                'members_at_risk': n,
                'annual_program_cost': invest,
                'annual_chronic_exposure': chronic_exposure,
                'expected_prevented': prevented,
                'expected_annual_savings': savings,
                'roi': round(savings / invest, 1) if invest > 0 else 0
            })

        return {
            'title': 'Preventive Care Investment Summary — CFO View',
            'population_overview': {
                'total_members': total_members,
                'currently_chronic': current_chronic,
                'chronic_pct': round(current_chronic / total_members * 100, 1),
                'very_high_risk': very_high_risk,
                'at_risk_of_chronic_conversion': at_risk_data['summary']['total_at_risk'],
            },
            'cost_reality': {
                'chronic_member_total_paid': round(chronic_paid, 2),
                'non_chronic_member_total_paid': round(non_chronic_paid, 2),
                'chronic_pmpm': chronic_pmpm,
                'non_chronic_pmpm': non_chronic_pmpm,
                'cost_multiplier': round(chronic_pmpm / non_chronic_pmpm, 1) if non_chronic_pmpm > 0 else 0,
                'insight': f'Chronic members cost {round(chronic_pmpm / non_chronic_pmpm, 1) if non_chronic_pmpm > 0 else "N/A"}x more per month than non-chronic members'
            },
            'investment_by_condition': investment_table,
            'total_investment': {
                'total_annual_program_cost': round(grand_total_invest, 2),
                'total_annual_savings': round(grand_total_savings, 2),
                'net_roi': round(grand_total_savings / grand_total_invest, 1) if grand_total_invest > 0 else 0,
                'payback_message': f'Investing ${grand_total_invest:,.0f}/year in preventive programs '
                                   f'is projected to save ${grand_total_savings:,.0f}/year — '
                                   f'a {round(grand_total_savings / grand_total_invest, 1) if grand_total_invest > 0 else 0}:1 return'
            },
            'executive_recommendation': self._build_executive_recommendation(
                investment_table, grand_total_invest, grand_total_savings, at_risk_data
            )
        }


    def _calc_age(self, dob: str) -> int:
        try:
            birth = datetime.strptime(dob, '%Y-%m-%d')
            today = datetime.now()
            return today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
        except (ValueError, TypeError):
            return 0

    def _build_investment_case(self, patients: List[Dict], condition_counts: Dict,
                                total_invest: float, total_savings: float) -> Dict[str, Any]:
        return {
            'headline': f'{len(patients)} members identified for preventive intervention',
            'investment_required': f'${total_invest:,.0f}/year across all programs',
            'projected_savings': f'${total_savings:,.0f}/year in avoided chronic disease costs',
            'roi': f'{round(total_savings / total_invest, 1) if total_invest > 0 else 0}:1 return on investment',
            'top_conditions': [
                f'{cond}: {count} members at risk'
                for cond, count in sorted(condition_counts.items(), key=lambda x: -x[1])[:5]
            ],
            'recommendation': 'Prioritize Diabetes Prevention Program (DPP) and Cardiac Risk '
                            'Management — these have the highest evidence base and ROI. '
                            'Start with CRITICAL tier members for immediate impact.'
        }

    def _build_executive_recommendation(self, investment_table: List[Dict],
                                         total_invest: float, total_savings: float,
                                         at_risk_data: Dict) -> List[str]:
        recs = []

        sorted_by_roi = sorted(investment_table, key=lambda x: -x.get('roi', 0))
        if sorted_by_roi:
            top = sorted_by_roi[0]
            recs.append(
                f"PRIORITY 1: Launch {top['condition']} prevention program — "
                f"{top['members_at_risk']} members at risk, "
                f"projected {top['roi']}:1 ROI (${top['expected_annual_savings']:,.0f}/year savings)"
            )

        sorted_by_volume = sorted(investment_table, key=lambda x: -x.get('members_at_risk', 0))
        if len(sorted_by_volume) > 1 and sorted_by_volume[0] != sorted_by_roi[0]:
            top_vol = sorted_by_volume[0]
            recs.append(
                f"PRIORITY 2: Address {top_vol['condition']} — largest at-risk cohort "
                f"({top_vol['members_at_risk']} members), "
                f"${top_vol['annual_chronic_exposure']:,.0f} annual chronic cost exposure"
            )

        critical = at_risk_data['summary'].get('critical_risk', 0)
        if critical > 0:
            recs.append(
                f"PRIORITY 3: Immediate outreach to {critical} CRITICAL-risk members — "
                f"these are closest to chronic conversion and need intervention within 30 days"
            )

        recs.append(
            "PRIORITY 4: Close care gaps — ensure all at-risk members have PCP assignment, "
            "up-to-date screenings, and active care management enrollment"
        )

        recs.append(
            f"BOTTOM LINE: A ${total_invest:,.0f} annual investment in targeted prevention "
            f"programs is projected to return ${total_savings:,.0f} in avoided costs — "
            f"directly supporting loss-slump recovery through reduced medical expense"
        )

        return recs


_chronic_risk_predictor = None

def get_chronic_risk_predictor(db_path: str) -> ChronicRiskPredictor:
    global _chronic_risk_predictor
    if _chronic_risk_predictor is None or _chronic_risk_predictor.db_path != db_path:
        _chronic_risk_predictor = ChronicRiskPredictor(db_path)
    return _chronic_risk_predictor
