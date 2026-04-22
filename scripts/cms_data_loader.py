import json
import sqlite3
import logging
import os
import time
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger('gpdm.cms')


ICD10_REFERENCE = {
    'E11.9': 'Type 2 diabetes mellitus without complications',
    'E11.65': 'Type 2 diabetes mellitus with hyperglycemia',
    'E11.21': 'Type 2 diabetes mellitus with diabetic nephropathy',
    'E11.22': 'Type 2 diabetes mellitus with diabetic chronic kidney disease',
    'E11.40': 'Type 2 diabetes mellitus with diabetic neuropathy, unspecified',
    'E11.311': 'Type 2 diabetes with unspecified diabetic retinopathy with macular edema',
    'E10.9': 'Type 1 diabetes mellitus without complications',

    'I10': 'Essential (primary) hypertension',
    'I11.0': 'Hypertensive heart disease with heart failure',
    'I11.9': 'Hypertensive heart disease without heart failure',
    'I12.9': 'Hypertensive chronic kidney disease with stage 1-4 CKD',
    'I13.10': 'Hypertensive heart and CKD without heart failure, with stage 1-4 CKD',

    'I25.10': 'Atherosclerotic heart disease of native coronary artery without angina',
    'I50.9': 'Heart failure, unspecified',
    'I50.20': 'Unspecified systolic (congestive) heart failure',
    'I50.30': 'Unspecified diastolic (congestive) heart failure',
    'I48.91': 'Unspecified atrial fibrillation',
    'I21.9': 'Acute myocardial infarction, unspecified',

    'J44.1': 'Chronic obstructive pulmonary disease with acute exacerbation',
    'J44.0': 'COPD with acute lower respiratory infection',
    'J45.20': 'Mild intermittent asthma, uncomplicated',
    'J45.40': 'Moderate persistent asthma, uncomplicated',
    'J18.9': 'Pneumonia, unspecified organism',

    'F32.1': 'Major depressive disorder, single episode, moderate',
    'F33.0': 'Major depressive disorder, recurrent, mild',
    'F41.1': 'Generalized anxiety disorder',
    'F41.9': 'Anxiety disorder, unspecified',
    'F17.210': 'Nicotine dependence, cigarettes, uncomplicated',

    'C50.911': 'Malignant neoplasm of unspecified site of right female breast',
    'C34.90': 'Malignant neoplasm of unspecified part of bronchus or lung',
    'C61': 'Malignant neoplasm of prostate',
    'C18.9': 'Malignant neoplasm of colon, unspecified',

    'N18.3': 'Chronic kidney disease, stage 3 (moderate)',
    'N18.4': 'Chronic kidney disease, stage 4 (severe)',
    'N18.5': 'Chronic kidney disease, stage 5',
    'N18.6': 'End stage renal disease',

    'U07.1': 'COVID-19',
    'J12.82': 'Pneumonia due to coronavirus disease 2019',

    'E66.01': 'Morbid (severe) obesity due to excess calories',
    'E66.09': 'Other obesity due to excess calories',
    'Z68.35': 'Body mass index (BMI) 35.0-39.9, adult',
    'Z68.41': 'Body mass index (BMI) 40.0-44.9, adult',

    'M54.5': 'Low back pain',
    'G47.33': 'Obstructive sleep apnea',
    'K21.0': 'Gastro-esophageal reflux disease with esophagitis',
    'M17.11': 'Primary osteoarthritis, right knee',
    'E78.5': 'Hyperlipidemia, unspecified',
    'R73.03': 'Prediabetes',
}

HCC_CATEGORIES = {
    'HCC19': {'name': 'Diabetes without Complication', 'icd_prefixes': ['E11.9', 'E10.9'], 'raf': 0.105},
    'HCC18': {'name': 'Diabetes with Chronic Complications', 'icd_prefixes': ['E11.2', 'E11.3', 'E11.4', 'E11.5'], 'raf': 0.302},
    'HCC85': {'name': 'Congestive Heart Failure', 'icd_prefixes': ['I50'], 'raf': 0.323},
    'HCC86': {'name': 'Acute Myocardial Infarction', 'icd_prefixes': ['I21'], 'raf': 0.258},
    'HCC88': {'name': 'Angina Pectoris', 'icd_prefixes': ['I20'], 'raf': 0.141},
    'HCC96': {'name': 'Specified Heart Arrhythmias', 'icd_prefixes': ['I48'], 'raf': 0.279},
    'HCC111': {'name': 'COPD', 'icd_prefixes': ['J44'], 'raf': 0.335},
    'HCC59': {'name': 'Major Depressive Disorder', 'icd_prefixes': ['F32', 'F33'], 'raf': 0.309},
    'HCC8': {'name': 'Metastatic Cancer', 'icd_prefixes': ['C77', 'C78', 'C79'], 'raf': 2.484},
    'HCC9': {'name': 'Lung/Upper Digestive Tract Cancer', 'icd_prefixes': ['C34', 'C15', 'C16'], 'raf': 0.979},
    'HCC10': {'name': 'Breast/Prostate/Colorectal Cancer', 'icd_prefixes': ['C50', 'C61', 'C18'], 'raf': 0.150},
    'HCC136': {'name': 'CKD Stage 4', 'icd_prefixes': ['N18.4'], 'raf': 0.289},
    'HCC137': {'name': 'CKD Stage 5/ESRD', 'icd_prefixes': ['N18.5', 'N18.6'], 'raf': 0.289},
    'HCC22': {'name': 'Morbid Obesity', 'icd_prefixes': ['E66.01'], 'raf': 0.250},
}

CPT_CATEGORIES = {
    'evaluation_management': {
        'name': 'Evaluation & Management',
        'codes': {
            '99213': 'Office visit, established patient, low complexity',
            '99214': 'Office visit, established patient, moderate complexity',
            '99215': 'Office visit, established patient, high complexity',
            '99203': 'Office visit, new patient, low complexity',
            '99204': 'Office visit, new patient, moderate complexity',
            '99205': 'Office visit, new patient, high complexity',
            '99281': 'Emergency department visit, minor',
            '99283': 'Emergency department visit, moderate',
            '99285': 'Emergency department visit, high severity',
            '99291': 'Critical care, first 30-74 minutes',
        },
    },
    'preventive': {
        'name': 'Preventive Services',
        'codes': {
            '99381': 'Preventive visit, new patient, infant',
            '99391': 'Preventive visit, established, infant',
            '99395': 'Preventive visit, established, 18-39',
            '99396': 'Preventive visit, established, 40-64',
            '99397': 'Preventive visit, established, 65+',
            'G0438': 'Annual wellness visit, initial',
            'G0439': 'Annual wellness visit, subsequent',
        },
    },
    'lab_pathology': {
        'name': 'Lab & Pathology',
        'codes': {
            '80053': 'Comprehensive metabolic panel',
            '85025': 'Complete blood count (CBC) with differential',
            '83036': 'Hemoglobin A1c',
            '80061': 'Lipid panel',
            '81001': 'Urinalysis with microscopy',
            '87491': 'Chlamydia detection',
            '87591': 'Gonorrhea detection',
        },
    },
    'imaging': {
        'name': 'Imaging',
        'codes': {
            '77067': 'Screening mammography, bilateral',
            '71046': 'Chest X-ray, 2 views',
            '74177': 'CT abdomen and pelvis with contrast',
            '70553': 'MRI brain with and without contrast',
        },
    },
    'procedures': {
        'name': 'Procedures',
        'codes': {
            '45378': 'Colonoscopy, diagnostic',
            '45385': 'Colonoscopy with polyp removal',
            '93000': 'Electrocardiogram (ECG), 12-lead',
            '93306': 'Echocardiography, complete',
            '36415': 'Venipuncture (blood draw)',
        },
    },
}

QUALITY_MEASURES = {
    'CDC': {
        'name': 'Comprehensive Diabetes Care',
        'description': 'Members 18-75 with diabetes who had HbA1c testing, eye exam, kidney screening',
        'numerator': 'Members with HbA1c test in measurement year',
        'denominator': 'Members 18-75 with diabetes (ICD E11.x or E10.x)',
        'benchmark': 0.90,
        'icd_codes': ['E11', 'E10'],
        'cpt_codes': ['83036'],
    },
    'BCS': {
        'name': 'Breast Cancer Screening',
        'description': 'Women 50-74 who had mammogram in past 2 years',
        'numerator': 'Women with mammogram (CPT 77067) in measurement period',
        'denominator': 'Women 50-74 continuously enrolled',
        'benchmark': 0.80,
        'cpt_codes': ['77067'],
    },
    'CBP': {
        'name': 'Controlling High Blood Pressure',
        'description': 'Members 18-85 with hypertension whose BP was adequately controlled',
        'numerator': 'Members with most recent BP < 140/90',
        'denominator': 'Members 18-85 with hypertension diagnosis',
        'benchmark': 0.70,
        'icd_codes': ['I10', 'I11', 'I12', 'I13'],
    },
    'COL': {
        'name': 'Colorectal Cancer Screening',
        'description': 'Members 45-75 with appropriate colorectal cancer screening',
        'numerator': 'Members with colonoscopy in past 10 years or FIT in past year',
        'denominator': 'Members 45-75 continuously enrolled',
        'benchmark': 0.75,
        'cpt_codes': ['45378', '45385'],
    },
    'PCR': {
        'name': 'Plan All-Cause Readmissions',
        'description': '30-day readmission rate for members 18+ after inpatient discharge',
        'numerator': 'Readmissions within 30 days',
        'denominator': 'Eligible inpatient discharges',
        'benchmark': 0.10,
        'lower_is_better': True,
    },
    'AMM': {
        'name': 'Antidepressant Medication Management',
        'description': 'Members 18+ with depression who remained on antidepressant medication',
        'numerator': 'Members on antidepressant for 84+ days (acute), 180+ days (continuation)',
        'denominator': 'Members with new depression episode and antidepressant dispensing',
        'benchmark': 0.65,
        'icd_codes': ['F32', 'F33'],
    },
    'FUH': {
        'name': 'Follow-Up After Hospitalization for Mental Illness',
        'description': 'Members 6+ with follow-up within 7 and 30 days of MH discharge',
        'numerator': 'Members with outpatient/telehealth MH visit within 7/30 days',
        'denominator': 'Members discharged from MH inpatient stay',
        'benchmark': 0.60,
        'icd_codes': ['F20', 'F25', 'F30', 'F31', 'F32', 'F33'],
    },
    'EDU': {
        'name': 'ED Utilization',
        'description': 'Emergency department visits per 1,000 member months',
        'numerator': 'ED visits',
        'denominator': 'Total member months',
        'benchmark': 50.0,
        'lower_is_better': True,
        'cpt_codes': ['99281', '99282', '99283', '99284', '99285'],
    },
}

CHRONIC_CONDITIONS = {
    'diabetes': {
        'name': 'Diabetes Mellitus',
        'icd_prefixes': ['E10', 'E11', 'E13'],
        'prevalence_65plus': 0.27,
        'cost_multiplier': 2.3,
        'comorbidities': ['hypertension', 'ckd', 'heart_disease', 'obesity'],
    },
    'hypertension': {
        'name': 'Hypertension',
        'icd_prefixes': ['I10', 'I11', 'I12', 'I13'],
        'prevalence_65plus': 0.58,
        'cost_multiplier': 1.4,
        'comorbidities': ['heart_disease', 'ckd', 'diabetes', 'stroke'],
    },
    'heart_disease': {
        'name': 'Ischemic Heart Disease',
        'icd_prefixes': ['I20', 'I21', 'I22', 'I24', 'I25'],
        'prevalence_65plus': 0.29,
        'cost_multiplier': 2.8,
        'comorbidities': ['hypertension', 'diabetes', 'hyperlipidemia'],
    },
    'heart_failure': {
        'name': 'Heart Failure',
        'icd_prefixes': ['I50'],
        'prevalence_65plus': 0.14,
        'cost_multiplier': 3.5,
        'comorbidities': ['hypertension', 'heart_disease', 'ckd', 'afib'],
    },
    'ckd': {
        'name': 'Chronic Kidney Disease',
        'icd_prefixes': ['N18'],
        'prevalence_65plus': 0.24,
        'cost_multiplier': 2.1,
        'comorbidities': ['diabetes', 'hypertension', 'heart_failure'],
    },
    'copd': {
        'name': 'COPD',
        'icd_prefixes': ['J44'],
        'prevalence_65plus': 0.11,
        'cost_multiplier': 2.0,
        'comorbidities': ['heart_failure', 'lung_cancer', 'depression'],
    },
    'depression': {
        'name': 'Depression',
        'icd_prefixes': ['F32', 'F33'],
        'prevalence_65plus': 0.14,
        'cost_multiplier': 1.8,
        'comorbidities': ['anxiety', 'substance_use', 'chronic_pain'],
    },
    'alzheimers': {
        'name': "Alzheimer's Disease / Dementia",
        'icd_prefixes': ['F01', 'F02', 'F03', 'G30'],
        'prevalence_65plus': 0.11,
        'cost_multiplier': 4.2,
        'comorbidities': ['depression', 'falls', 'urinary_incontinence'],
    },
    'cancer_breast': {
        'name': 'Breast Cancer',
        'icd_prefixes': ['C50'],
        'prevalence_65plus': 0.04,
        'cost_multiplier': 3.0,
        'comorbidities': ['depression', 'osteoporosis'],
    },
    'cancer_lung': {
        'name': 'Lung Cancer',
        'icd_prefixes': ['C34'],
        'prevalence_65plus': 0.02,
        'cost_multiplier': 5.1,
        'comorbidities': ['copd', 'depression'],
    },
    'obesity': {
        'name': 'Obesity',
        'icd_prefixes': ['E66'],
        'prevalence_65plus': 0.19,
        'cost_multiplier': 1.5,
        'comorbidities': ['diabetes', 'hypertension', 'osteoarthritis', 'sleep_apnea'],
    },
}

def build_healthcare_vocabulary() -> List[str]:
    documents = []

    for code, desc in ICD10_REFERENCE.items():
        documents.append(f"ICD-10 code {code} diagnosis {desc}")
        documents.append(desc.lower())

    for hcc_id, info in HCC_CATEGORIES.items():
        documents.append(f"HCC category {hcc_id} {info['name']} risk adjustment factor {info['raf']}")
        documents.append(f"{info['name']} is a hierarchical condition category for risk scoring")

    for category, data in CPT_CATEGORIES.items():
        for code, desc in data['codes'].items():
            documents.append(f"CPT code {code} procedure {desc} category {data['name']}")

    for measure_id, info in QUALITY_MEASURES.items():
        documents.append(f"HEDIS measure {measure_id} {info['name']} {info['description']}")
        documents.append(f"{info['name']} quality measure numerator {info['numerator']} "
                        f"denominator {info['denominator']} benchmark {info.get('benchmark', 'N/A')}")

    for condition_id, info in CHRONIC_CONDITIONS.items():
        documents.append(f"{info['name']} chronic condition prevalence {info['prevalence_65plus']} "
                        f"cost multiplier {info['cost_multiplier']}")
        if info.get('comorbidities'):
            comorbidities = ' '.join(info['comorbidities'])
            documents.append(f"{info['name']} is commonly associated with {comorbidities}")

    domain_docs = [
        "claims are billing records for healthcare services rendered to patients",
        "claim types include professional institutional inpatient outpatient pharmacy",
        "claim status can be paid denied pending adjudicated submitted",
        "billed amount is what provider charges paid amount is what payer reimburses",

        "members are enrolled patients in the health plan",
        "member demographics include age gender region plan type enrollment date",
        "risk score measures predicted healthcare costs based on diagnoses",

        "providers are physicians nurses specialists facilities",
        "provider specialty includes primary care cardiology oncology pediatrics",
        "NPI is the national provider identifier unique to each provider",

        "encounters are patient visits to healthcare providers",
        "visit types include inpatient outpatient emergency telehealth office",
        "length of stay is days between admission and discharge",
        "readmission is returning to hospital within 30 days of discharge",

        "prescriptions are medication orders filled at pharmacy",
        "drug class therapeutic category generic brand formulary",
        "days supply quantity dispensed refills remaining",

        "HEDIS is Healthcare Effectiveness Data and Information Set quality measures",
        "STARS rating is CMS quality rating system for Medicare Advantage plans",
        "quality measures track preventive care chronic disease management outcomes",

        "total count of claims by type shows distribution across claim categories",
        "average paid amount by provider shows cost variation across physicians",
        "trend over time by month shows seasonal patterns and growth trajectories",
        "denial rate is percentage of claims rejected by payer",
        "readmission rate measures quality of inpatient discharge care",
        "cost per member per month PMPM is key managed care financial metric",
        "risk adjusted outcomes account for patient severity differences",
    ]
    documents.extend(domain_docs)

    return documents

def build_clinical_concept_graph() -> Dict[str, Any]:
    nodes = []
    edges = []

    for condition_id, info in CHRONIC_CONDITIONS.items():
        nodes.append({
            'id': f'condition:{condition_id}',
            'type': 'condition',
            'name': info['name'],
            'attributes': {
                'prevalence': info['prevalence_65plus'],
                'cost_multiplier': info['cost_multiplier'],
                'icd_prefixes': info['icd_prefixes'],
            },
        })

        for comorbidity in info.get('comorbidities', []):
            edges.append({
                'source': f'condition:{condition_id}',
                'target': f'condition:{comorbidity}',
                'type': 'comorbidity',
                'weight': 1.0,
            })

    for measure_id, info in QUALITY_MEASURES.items():
        nodes.append({
            'id': f'measure:{measure_id}',
            'type': 'quality_measure',
            'name': info['name'],
            'attributes': {
                'benchmark': info.get('benchmark'),
                'lower_is_better': info.get('lower_is_better', False),
            },
        })

        for icd in info.get('icd_codes', []):
            for condition_id, cinfo in CHRONIC_CONDITIONS.items():
                if any(icd.startswith(p) for p in cinfo['icd_prefixes']):
                    edges.append({
                        'source': f'measure:{measure_id}',
                        'target': f'condition:{condition_id}',
                        'type': 'measures',
                        'weight': 1.5,
                    })

    for cat_id, data in CPT_CATEGORIES.items():
        nodes.append({
            'id': f'cpt_category:{cat_id}',
            'type': 'procedure_category',
            'name': data['name'],
            'attributes': {
                'num_codes': len(data['codes']),
            },
        })

    return {'nodes': nodes, 'edges': edges}

class CMSKnowledgeBase:

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
        self._loaded = False

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.executescript('''
            CREATE TABLE IF NOT EXISTS icd10_codes (
                code TEXT PRIMARY KEY,
                description TEXT,
                category TEXT,
                hcc_mapping TEXT
            );
            CREATE TABLE IF NOT EXISTS cpt_codes (
                code TEXT PRIMARY KEY,
                description TEXT,
                category TEXT
            );
            CREATE TABLE IF NOT EXISTS hcc_categories (
                hcc_id TEXT PRIMARY KEY,
                name TEXT,
                raf_score REAL,
                icd_prefixes TEXT
            );
            CREATE TABLE IF NOT EXISTS quality_measures (
                measure_id TEXT PRIMARY KEY,
                name TEXT,
                description TEXT,
                benchmark REAL,
                definition TEXT
            );
            CREATE TABLE IF NOT EXISTS chronic_conditions (
                condition_id TEXT PRIMARY KEY,
                name TEXT,
                icd_prefixes TEXT,
                prevalence REAL,
                cost_multiplier REAL,
                comorbidities TEXT
            );
            CREATE TABLE IF NOT EXISTS clinical_vocabulary (
                term TEXT PRIMARY KEY,
                category TEXT,
                related_terms TEXT,
                embedding TEXT
            );
        ''')
        conn.commit()
        conn.close()

    def load_all(self) -> None:
        conn = sqlite3.connect(self.db_path)

        for code, desc in ICD10_REFERENCE.items():
            hcc = None
            for hcc_id, info in HCC_CATEGORIES.items():
                if any(code.startswith(p) for p in info['icd_prefixes']):
                    hcc = hcc_id
                    break
            conn.execute('INSERT OR REPLACE INTO icd10_codes VALUES (?, ?, ?, ?)',
                        (code, desc, code[0], hcc))

        for cat_id, data in CPT_CATEGORIES.items():
            for code, desc in data['codes'].items():
                conn.execute('INSERT OR REPLACE INTO cpt_codes VALUES (?, ?, ?)',
                            (code, desc, data['name']))

        for hcc_id, info in HCC_CATEGORIES.items():
            conn.execute('INSERT OR REPLACE INTO hcc_categories VALUES (?, ?, ?, ?)',
                        (hcc_id, info['name'], info['raf'],
                         json.dumps(info['icd_prefixes'])))

        for m_id, info in QUALITY_MEASURES.items():
            conn.execute('INSERT OR REPLACE INTO quality_measures VALUES (?, ?, ?, ?, ?)',
                        (m_id, info['name'], info['description'],
                         info.get('benchmark'), json.dumps(info)))

        for c_id, info in CHRONIC_CONDITIONS.items():
            conn.execute('INSERT OR REPLACE INTO chronic_conditions VALUES (?, ?, ?, ?, ?, ?)',
                        (c_id, info['name'], json.dumps(info['icd_prefixes']),
                         info['prevalence_65plus'], info['cost_multiplier'],
                         json.dumps(info.get('comorbidities', []))))

        conn.commit()
        conn.close()
        self._loaded = True
        logger.info("CMS KB: %d ICD-10, %d HCC, %d measures, %d conditions",
                     len(ICD10_REFERENCE), len(HCC_CATEGORIES),
                     len(QUALITY_MEASURES), len(CHRONIC_CONDITIONS))

    def lookup_icd10(self, code: str) -> Optional[Dict]:
        conn = sqlite3.connect(self.db_path)
        row = conn.execute('SELECT * FROM icd10_codes WHERE code = ?', (code,)).fetchone()
        conn.close()
        if row:
            return {'code': row[0], 'description': row[1], 'category': row[2], 'hcc': row[3]}
        return None

    def lookup_cpt(self, code: str) -> Optional[Dict]:
        conn = sqlite3.connect(self.db_path)
        row = conn.execute('SELECT * FROM cpt_codes WHERE code = ?', (code,)).fetchone()
        conn.close()
        if row:
            return {'code': row[0], 'description': row[1], 'category': row[2]}
        return None

    def get_quality_measure(self, measure_id: str) -> Optional[Dict]:
        conn = sqlite3.connect(self.db_path)
        row = conn.execute('SELECT * FROM quality_measures WHERE measure_id = ?',
                          (measure_id,)).fetchone()
        conn.close()
        if row:
            return {'id': row[0], 'name': row[1], 'description': row[2],
                    'benchmark': row[3], 'definition': json.loads(row[4])}
        return None

    def find_condition(self, term: str) -> Optional[Dict]:
        term_lower = term.lower()
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute('SELECT * FROM chronic_conditions').fetchall()
        conn.close()

        for row in rows:
            if term_lower in row[1].lower() or term_lower == row[0]:
                return {
                    'id': row[0], 'name': row[1],
                    'icd_prefixes': json.loads(row[2]),
                    'prevalence': row[3], 'cost_multiplier': row[4],
                    'comorbidities': json.loads(row[5]),
                }
        return None

    def get_pretraining_documents(self) -> List[str]:
        return build_healthcare_vocabulary()

    def get_concept_graph(self) -> Dict[str, Any]:
        return build_clinical_concept_graph()

    def get_statistics(self) -> Dict[str, int]:
        return {
            'icd10_codes': len(ICD10_REFERENCE),
            'hcc_categories': len(HCC_CATEGORIES),
            'cpt_codes': sum(len(d['codes']) for d in CPT_CATEGORIES.values()),
            'quality_measures': len(QUALITY_MEASURES),
            'chronic_conditions': len(CHRONIC_CONDITIONS),
            'vocabulary_documents': len(build_healthcare_vocabulary()),
        }

