import csv
import os
import random
import hashlib
from datetime import datetime, timedelta

random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
os.makedirs(RAW_DIR, exist_ok=True)


FIRST_NAMES = [
    'James', 'Mary', 'Robert', 'Patricia', 'John', 'Jennifer', 'Michael', 'Linda',
    'David', 'Elizabeth', 'William', 'Barbara', 'Richard', 'Susan', 'Joseph', 'Jessica',
    'Thomas', 'Sarah', 'Charles', 'Karen', 'Christopher', 'Lisa', 'Daniel', 'Nancy',
    'Matthew', 'Betty', 'Anthony', 'Margaret', 'Mark', 'Sandra', 'Donald', 'Ashley',
    'Steven', 'Kimberly', 'Paul', 'Emily', 'Andrew', 'Donna', 'Joshua', 'Michelle',
    'Kenneth', 'Dorothy', 'Kevin', 'Carol', 'Brian', 'Amanda', 'George', 'Melissa',
    'Timothy', 'Deborah', 'Ronald', 'Stephanie', 'Edward', 'Rebecca', 'Jason', 'Sharon',
    'Jeffrey', 'Laura', 'Ryan', 'Cynthia', 'Jacob', 'Kathleen', 'Gary', 'Amy',
    'Nicholas', 'Angela', 'Eric', 'Shirley', 'Jonathan', 'Anna', 'Stephen', 'Brenda',
    'Larry', 'Pamela', 'Justin', 'Emma', 'Scott', 'Nicole', 'Brandon', 'Helen',
    'Benjamin', 'Samantha', 'Samuel', 'Katherine', 'Raymond', 'Christine', 'Gregory', 'Debra',
    'Frank', 'Rachel', 'Alexander', 'Carolyn', 'Patrick', 'Janet', 'Jack', 'Catherine',
]

LAST_NAMES = [
    'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis',
    'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson',
    'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin', 'Lee', 'Perez', 'Thompson',
    'White', 'Harris', 'Sanchez', 'Clark', 'Ramirez', 'Lewis', 'Robinson', 'Walker',
    'Young', 'Allen', 'King', 'Wright', 'Scott', 'Torres', 'Nguyen', 'Hill', 'Flores',
    'Green', 'Adams', 'Nelson', 'Baker', 'Hall', 'Rivera', 'Campbell', 'Mitchell',
    'Carter', 'Roberts', 'Gomez', 'Phillips', 'Evans', 'Turner', 'Diaz', 'Parker',
    'Cruz', 'Edwards', 'Collins', 'Reyes', 'Stewart', 'Morris', 'Morales', 'Murphy',
]

REGIONS = ['NCAL', 'SCAL', 'NW', 'CO', 'MAS', 'HI', 'GA', 'MID']

FACILITIES = {
    'NCAL': ['Oakland Medical Center', 'San Francisco Medical Center', 'Santa Clara Medical Center',
             'Walnut Creek Medical Center', 'Sacramento Medical Center', 'Fremont Medical Center'],
    'SCAL': ['Los Angeles Medical Center', 'Fontana Medical Center', 'Irvine Medical Center',
             'San Diego Medical Center', 'Riverside Medical Center', 'Downey Medical Center'],
    'NW': ['Sunnyside Medical Center', 'Westside Medical Center', 'Interstate Medical Office'],
    'CO': ['Franklin Medical Center', 'Rock Creek Medical Office', 'Skyline Medical Center'],
    'MAS': ['Tysons Corner Medical Center', 'Capitol Hill Medical Center'],
    'HI': ['Moanalua Medical Center', 'Maui Lani Medical Office'],
    'GA': ['Crescent Center', 'Town Center Medical Office'],
    'MID': ['Mid-Atlantic Medical Center', 'Burke Medical Center'],
}

DEPARTMENTS = [
    'Internal Medicine', 'Cardiology', 'Orthopedics', 'Pediatrics', 'Neurology',
    'Oncology', 'Dermatology', 'Psychiatry', 'Emergency Medicine', 'Radiology',
    'Surgery', 'OB/GYN', 'Urology', 'Pulmonology', 'Endocrinology',
    'Gastroenterology', 'Nephrology', 'Rheumatology', 'Ophthalmology', 'ENT',
]

VISIT_TYPES = ['INPATIENT', 'OUTPATIENT', 'EMERGENCY', 'TELEHEALTH', 'URGENT_CARE', 'HOME_HEALTH']

CLAIM_STATUSES = ['PAID', 'DENIED', 'PENDING', 'ADJUSTED', 'APPEALED', 'VOIDED']

ICD10_CODES = [
    ('E11.9', 'Type 2 diabetes mellitus without complications'),
    ('I10', 'Essential hypertension'),
    ('J06.9', 'Acute upper respiratory infection'),
    ('M54.5', 'Low back pain'),
    ('K21.0', 'GERD with esophagitis'),
    ('F32.1', 'Major depressive disorder, single episode, moderate'),
    ('J45.20', 'Mild intermittent asthma, uncomplicated'),
    ('E78.5', 'Hyperlipidemia, unspecified'),
    ('G43.909', 'Migraine, unspecified'),
    ('N39.0', 'Urinary tract infection'),
    ('M79.3', 'Panniculitis, unspecified'),
    ('R10.9', 'Unspecified abdominal pain'),
    ('Z00.00', 'Encounter for general adult medical examination'),
    ('I25.10', 'Atherosclerotic heart disease'),
    ('E03.9', 'Hypothyroidism, unspecified'),
    ('J18.9', 'Pneumonia, unspecified organism'),
    ('K58.9', 'Irritable bowel syndrome without diarrhea'),
    ('G47.00', 'Insomnia, unspecified'),
    ('M17.11', 'Primary osteoarthritis, right knee'),
    ('R51', 'Headache'),
    ('L30.9', 'Dermatitis, unspecified'),
    ('F41.1', 'Generalized anxiety disorder'),
    ('I48.91', 'Unspecified atrial fibrillation'),
    ('E66.01', 'Morbid obesity due to excess calories'),
    ('N18.3', 'Chronic kidney disease, stage 3'),
    ('U07.1', 'COVID-19'),
    ('U07.2', 'COVID-19, virus not identified'),
    ('I50.9', 'Heart failure, unspecified'),
    ('J44.1', 'COPD with acute exacerbation'),
    ('Z23', 'Encounter for immunization'),
    ('R05.9', 'Cough, unspecified'),
    ('M25.50', 'Pain in unspecified joint'),
    ('E55.9', 'Vitamin D deficiency, unspecified'),
    ('Z12.31', 'Encounter for screening mammogram'),
    ('Z87.891', 'Personal history of nicotine dependence'),
]

CPT_CODES = [
    ('99213', 'Office visit, established patient, low complexity'),
    ('99214', 'Office visit, established patient, moderate complexity'),
    ('99215', 'Office visit, established patient, high complexity'),
    ('99203', 'Office visit, new patient, low complexity'),
    ('99281', 'Emergency department visit, minimal'),
    ('99283', 'Emergency department visit, moderate'),
    ('99285', 'Emergency department visit, high'),
    ('99386', 'Preventive visit, 40-64 years'),
    ('36415', 'Routine venipuncture'),
    ('85025', 'Complete blood count'),
    ('80053', 'Comprehensive metabolic panel'),
    ('71046', 'Chest x-ray, 2 views'),
    ('93000', 'Electrocardiogram, complete'),
    ('90837', 'Psychotherapy, 60 minutes'),
    ('97110', 'Therapeutic exercises'),
    ('99232', 'Subsequent hospital care'),
    ('99223', 'Initial hospital care, high complexity'),
    ('90471', 'Immunization administration'),
    ('11102', 'Tangential biopsy of skin'),
    ('43239', 'Upper GI endoscopy with biopsy'),
    ('99201', 'Office visit, new patient, straightforward'),
    ('99204', 'Office visit, new patient, moderate complexity'),
    ('99205', 'Office visit, new patient, high complexity'),
    ('99221', 'Initial hospital care, low complexity'),
    ('99222', 'Initial hospital care, moderate complexity'),
    ('99231', 'Subsequent hospital care, low complexity'),
    ('99233', 'Subsequent hospital care, high complexity'),
    ('99282', 'Emergency department visit, low'),
    ('99284', 'Emergency department visit, moderately high'),
    ('99291', 'Critical care, first 30-74 minutes'),
    ('99395', 'Preventive visit, 18-39 years'),
    ('99396', 'Preventive visit, 40-64 years'),
    ('99397', 'Preventive visit, 65+ years'),
    ('70553', 'MRI brain without/with contrast'),
    ('73721', 'MRI knee without contrast'),
    ('74177', 'CT abdomen/pelvis with contrast'),
    ('77067', 'Screening mammography, bilateral'),
    ('81001', 'Urinalysis, automated with microscopy'),
    ('82947', 'Glucose, quantitative'),
    ('83036', 'Hemoglobin A1c'),
    ('84443', 'Thyroid stimulating hormone'),
    ('87880', 'Strep A rapid test'),
    ('90651', 'HPV vaccine, 9-valent'),
    ('90658', 'Influenza virus vaccine'),
    ('90686', 'Flu vaccine, quadrivalent'),
    ('90715', 'Tdap vaccine'),
    ('96372', 'Therapeutic injection, SC/IM'),
    ('97140', 'Manual therapy techniques'),
    ('99211', 'Office visit, established, minimal'),
    ('99212', 'Office visit, established, straightforward'),
    ('99241', 'Office consultation, straightforward'),
    ('99243', 'Office consultation, moderate complexity'),
    ('10060', 'Incision and drainage of abscess'),
    ('20610', 'Arthrocentesis, major joint'),
    ('27447', 'Total knee arthroplasty'),
    ('29881', 'Arthroscopy, knee, meniscectomy'),
    ('45380', 'Colonoscopy with biopsy'),
    ('47562', 'Laparoscopic cholecystectomy'),
    ('49505', 'Inguinal hernia repair'),
    ('50590', 'Lithotripsy'),
    ('55700', 'Prostate biopsy'),
    ('58661', 'Laparoscopy, surgical; with removal of adnexal structures'),
    ('59400', 'Routine obstetric care, vaginal delivery'),
    ('62322', 'Lumbar epidural injection'),
    ('64483', 'Transforaminal epidural injection'),
    ('66984', 'Cataract surgery with IOL'),
    ('76830', 'Transvaginal ultrasound'),
    ('76856', 'Pelvic ultrasound, complete'),
    ('78452', 'Myocardial perfusion imaging'),
    ('80048', 'Basic metabolic panel'),
    ('85027', 'Complete blood count, automated'),
    ('86580', 'Tuberculin skin test'),
    ('88305', 'Surgical pathology, level IV'),
    ('90834', 'Psychotherapy, 45 minutes'),
    ('90847', 'Family psychotherapy with patient'),
    ('92014', 'Comprehensive eye exam, established'),
    ('93306', 'Echocardiography, complete'),
    ('95810', 'Polysomnography, sleep study'),
    ('96127', 'Brief emotional/behavioral assessment'),
    ('97530', 'Therapeutic activities'),
    ('99024', 'Postoperative follow-up visit'),
]

MEDICATIONS = [
    ('Metformin 500mg', 'Diabetes'), ('Lisinopril 10mg', 'Hypertension'),
    ('Atorvastatin 20mg', 'Cholesterol'), ('Amlodipine 5mg', 'Hypertension'),
    ('Omeprazole 20mg', 'GERD'), ('Sertraline 50mg', 'Depression'),
    ('Albuterol Inhaler', 'Asthma'), ('Levothyroxine 50mcg', 'Hypothyroid'),
    ('Gabapentin 300mg', 'Neuropathy'), ('Hydrochlorothiazide 25mg', 'Hypertension'),
    ('Metoprolol 50mg', 'Hypertension'), ('Losartan 50mg', 'Hypertension'),
    ('Escitalopram 10mg', 'Anxiety'), ('Prednisone 10mg', 'Inflammation'),
    ('Amoxicillin 500mg', 'Infection'), ('Azithromycin 250mg', 'Infection'),
    ('Ibuprofen 800mg', 'Pain'), ('Tramadol 50mg', 'Pain'),
    ('Pantoprazole 40mg', 'GERD'), ('Duloxetine 60mg', 'Depression'),
]

PLAN_TYPES = ['HMO', 'PPO', 'EPO', 'HDHP', 'Medicare Advantage', 'Medicaid']

STATES = ['CA', 'OR', 'WA', 'CO', 'VA', 'DC', 'MD', 'HI', 'GA']

GENDERS = ['M', 'F']

RACES = ['White', 'Black', 'Hispanic', 'Asian', 'Native Hawaiian', 'American Indian', 'Other', 'Unknown']

LANGUAGES = ['English', 'Spanish', 'Chinese', 'Vietnamese', 'Korean', 'Tagalog', 'Russian', 'Other']


def gen_mrn():
    return f"MRN{random.randint(1000000, 9999999)}"

def gen_member_id():
    return f"MBR{random.randint(100000000, 999999999)}"

def gen_npi():
    return f"{random.randint(1000000000, 1999999999)}"

def gen_encounter_id():
    return f"ENC{random.randint(10000000, 99999999)}"

def gen_claim_id():
    return f"CLM{random.randint(10000000, 99999999)}"

def gen_rx_id():
    return f"RX{random.randint(1000000, 9999999)}"

def gen_referral_id():
    return f"REF{random.randint(100000, 999999)}"

def gen_date(start_year=2023, end_year=2025):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = (end - start).days
    return (start + timedelta(days=random.randint(0, delta))).strftime('%Y-%m-%d')

def gen_dob():
    w = [1]*17 + [3]*8 + [5]*10 + [7]*15 + [6]*10 + [4]*10 + [3]*10 + [2]*9 + [1]*5
    age = random.choices(range(1, 95), weights=w, k=1)[0]
    base = datetime(2025, 6, 15)
    birth = base - timedelta(days=age * 365 + random.randint(0, 364))
    return birth.strftime('%Y-%m-%d')

def gen_phone():
    return f"({random.randint(200,999)}) {random.randint(200,999)}-{random.randint(1000,9999)}"

def gen_zip():
    return f"{random.randint(10000, 99999)}"

def gen_address():
    nums = random.randint(100, 9999)
    streets = ['Main St', 'Oak Ave', 'Elm Dr', 'Pine Rd', 'Maple Ln', 'Cedar Blvd',
               'Washington Ave', 'Park Dr', 'Lake Rd', 'Hill St', 'Valley Way', 'River Rd']
    return f"{nums} {random.choice(streets)}"

def write_csv(filepath, headers, rows):
    with open(filepath, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    print(f"  Written: {filepath} ({len(rows)} rows)")


def generate_members(n=25000):
    print(f"\n  Generating {n} members...")
    headers = [
        'MRN', 'MEMBER_ID', 'FIRST_NAME', 'LAST_NAME', 'DATE_OF_BIRTH', 'GENDER',
        'RACE', 'LANGUAGE', 'ADDRESS', 'CITY', 'STATE', 'ZIP_CODE', 'PHONE',
        'EMAIL', 'KP_REGION', 'FACILITY', 'PLAN_TYPE', 'ENROLLMENT_DATE',
        'DISENROLLMENT_DATE', 'PCP_NPI', 'RISK_SCORE', 'CHRONIC_CONDITIONS',
    ]
    rows = []
    member_ids = []
    mrns = []
    pcp_npis = [gen_npi() for _ in range(500)]

    for i in range(n):
        mrn = gen_mrn()
        mid = gen_member_id()
        mrns.append(mrn)
        member_ids.append(mid)
        fname = random.choice(FIRST_NAMES)
        lname = random.choice(LAST_NAMES)
        dob = gen_dob()
        gender = random.choice(GENDERS)
        race = random.choice(RACES)
        lang = random.choices(LANGUAGES, weights=[60, 15, 8, 5, 3, 3, 2, 4])[0]
        region = random.choice(REGIONS)
        facility = random.choice(FACILITIES[region])
        plan = random.choice(PLAN_TYPES)
        enroll = gen_date(2018, 2024)
        disenroll = gen_date(2024, 2025) if random.random() < 0.08 else ''
        pcp = random.choice(pcp_npis)
        risk = round(random.uniform(0.2, 4.5), 2)
        chronic = random.randint(0, 6)
        email = f"{fname.lower()}.{lname.lower()}{random.randint(1,99)}@email.com"

        if random.random() < 0.03:
            phone = ''
        else:
            phone = gen_phone()
        if random.random() < 0.05:
            email = ''
        if random.random() < 0.02:
            race = ''

        rows.append([
            mrn, mid, fname, lname, dob, gender, race, lang,
            gen_address(), random.choice(['Oakland', 'LA', 'Portland', 'Denver', 'Tysons', 'Honolulu', 'Atlanta', 'Baltimore']),
            random.choice(STATES), gen_zip(), phone, email,
            region, facility, plan, enroll, disenroll, pcp, risk, chronic,
        ])

    write_csv(os.path.join(RAW_DIR, 'members.csv'), headers, rows)
    return mrns, member_ids


def generate_providers(n=3000):
    print(f"\n  Generating {n} providers...")
    headers = [
        'NPI', 'PROVIDER_FIRST_NAME', 'PROVIDER_LAST_NAME', 'SPECIALTY', 'DEPARTMENT',
        'KP_REGION', 'FACILITY', 'PROVIDER_TYPE', 'DEA_NUMBER', 'LICENSE_STATE',
        'HIRE_DATE', 'STATUS', 'PANEL_SIZE', 'ACCEPTS_NEW_PATIENTS',
    ]
    rows = []
    npis = []
    types = ['MD', 'DO', 'NP', 'PA', 'RN', 'LCSW', 'PhD']

    for i in range(n):
        npi = gen_npi()
        npis.append(npi)
        region = random.choice(REGIONS)
        dept = random.choice(DEPARTMENTS)
        ptype = random.choices(types, weights=[40, 10, 15, 10, 15, 5, 5])[0]
        rows.append([
            npi, random.choice(FIRST_NAMES), random.choice(LAST_NAMES),
            dept, dept, region, random.choice(FACILITIES[region]),
            ptype, f"DEA{random.randint(1000000, 9999999)}",
            random.choice(STATES), gen_date(2005, 2023),
            random.choices(['ACTIVE', 'INACTIVE', 'ON_LEAVE'], weights=[85, 10, 5])[0],
            random.randint(100, 2500), random.choices(['Y', 'N'], weights=[70, 30])[0],
        ])

    write_csv(os.path.join(RAW_DIR, 'providers.csv'), headers, rows)
    return npis


def generate_encounters(n=50000, member_ids=None, npis=None):
    print(f"\n  Generating {n} encounters...")
    headers = [
        'ENCOUNTER_ID', 'MEMBER_ID', 'MRN', 'SERVICE_DATE', 'ADMIT_DATE', 'DISCHARGE_DATE',
        'RENDERING_NPI', 'SUPERVISING_NPI', 'VISIT_TYPE', 'DEPARTMENT', 'FACILITY',
        'KP_REGION', 'PRIMARY_DIAGNOSIS', 'DIAGNOSIS_DESCRIPTION', 'CHIEF_COMPLAINT',
        'DISPOSITION', 'LENGTH_OF_STAY', 'ENCOUNTER_STATUS',
    ]
    rows = []
    encounter_ids = []
    dispositions = ['Discharged', 'Admitted', 'Transferred', 'AMA', 'Expired', 'Observation']
    complaints = [
        'Chest pain', 'Shortness of breath', 'Abdominal pain', 'Headache', 'Back pain',
        'Fever', 'Cough', 'Dizziness', 'Joint pain', 'Fatigue', 'Nausea',
        'Skin rash', 'Sore throat', 'Ear pain', 'Eye irritation', 'Anxiety',
        'Routine checkup', 'Follow-up visit', 'Medication refill', 'Lab review',
    ]

    for i in range(n):
        eid = gen_encounter_id()
        encounter_ids.append(eid)
        mid = random.choice(member_ids) if member_ids else gen_member_id()
        vtype = random.choices(VISIT_TYPES, weights=[15, 40, 10, 20, 10, 5])[0]
        sdate = gen_date(2023, 2025)
        diag = random.choice(ICD10_CODES)
        region = random.choice(REGIONS)
        npi = random.choice(npis) if npis else gen_npi()

        admit = sdate if vtype == 'INPATIENT' else ''
        los = random.randint(1, 14) if vtype == 'INPATIENT' else 0
        if admit:
            discharge = (datetime.strptime(sdate, '%Y-%m-%d') + timedelta(days=los)).strftime('%Y-%m-%d')
        else:
            discharge = ''

        sup_npi = random.choice(npis) if npis and random.random() < 0.3 else ''

        rows.append([
            eid, mid, gen_mrn(), sdate, admit, discharge,
            npi, sup_npi, vtype, random.choice(DEPARTMENTS),
            random.choice(FACILITIES[region]), region,
            diag[0], diag[1], random.choice(complaints),
            random.choices(dispositions, weights=[60, 15, 5, 2, 1, 17])[0],
            los, random.choices(['COMPLETE', 'IN_PROGRESS', 'CANCELLED'], weights=[85, 10, 5])[0],
        ])

    write_csv(os.path.join(RAW_DIR, 'encounters.csv'), headers, rows)
    return encounter_ids


def generate_claims(n=60000, member_ids=None, npis=None, encounter_ids=None):
    print(f"\n  Generating {n} claims...")
    headers = [
        'CLAIM_ID', 'MEMBER_ID', 'ENCOUNTER_ID', 'SERVICE_DATE', 'RENDERING_NPI',
        'BILLING_NPI', 'CPT_CODE', 'CPT_DESCRIPTION', 'ICD10_CODE', 'ICD10_DESCRIPTION',
        'BILLED_AMOUNT', 'ALLOWED_AMOUNT', 'PAID_AMOUNT', 'MEMBER_RESPONSIBILITY',
        'COPAY', 'COINSURANCE', 'DEDUCTIBLE', 'CLAIM_STATUS', 'DENIAL_REASON',
        'KP_REGION', 'FACILITY', 'PLAN_TYPE', 'CLAIM_TYPE', 'SUBMITTED_DATE', 'ADJUDICATED_DATE',
    ]
    rows = []
    denial_reasons = ['', '', '', '', '', 'Not medically necessary', 'Out of network',
                      'Pre-auth required', 'Duplicate claim', 'Timely filing', 'Coding error']
    claim_types = ['PROFESSIONAL', 'INSTITUTIONAL', 'PHARMACY', 'DME']

    for i in range(n):
        cpt = random.choice(CPT_CODES)
        diag = random.choice(ICD10_CODES)
        sdate = gen_date(2023, 2025)
        billed = round(random.uniform(50, 15000), 2)
        allowed = round(billed * random.uniform(0.4, 0.95), 2)
        status = random.choices(CLAIM_STATUSES, weights=[60, 10, 15, 8, 5, 2])[0]
        paid = round(allowed * random.uniform(0.7, 1.0), 2) if status == 'PAID' else 0.0
        copay = random.choice([0, 15, 20, 25, 30, 40, 50, 75])
        region = random.choice(REGIONS)

        sub_date = sdate
        adj_date = (datetime.strptime(sdate, '%Y-%m-%d') + timedelta(days=random.randint(7, 45))).strftime('%Y-%m-%d')

        denial = random.choice(denial_reasons) if status == 'DENIED' else ''

        rows.append([
            gen_claim_id(),
            random.choice(member_ids) if member_ids else gen_member_id(),
            random.choice(encounter_ids) if encounter_ids and random.random() < 0.8 else '',
            sdate,
            random.choice(npis) if npis else gen_npi(),
            random.choice(npis) if npis else gen_npi(),
            cpt[0], cpt[1], diag[0], diag[1],
            billed, allowed, paid,
            round(billed - paid - copay, 2) if paid > 0 else 0,
            copay, round(random.uniform(0, billed * 0.2), 2), round(random.uniform(0, 500), 2),
            status, denial, region, random.choice(FACILITIES[region]),
            random.choice(PLAN_TYPES),
            random.choices(claim_types, weights=[50, 30, 15, 5])[0],
            sub_date, adj_date,
        ])

    write_csv(os.path.join(RAW_DIR, 'claims.csv'), headers, rows)


def generate_diagnoses(n=20000, member_ids=None, encounter_ids=None):
    print(f"\n  Generating {n} diagnoses...")
    headers = [
        'DIAGNOSIS_ID', 'MEMBER_ID', 'ENCOUNTER_ID', 'ICD10_CODE', 'ICD10_DESCRIPTION',
        'DIAGNOSIS_TYPE', 'DIAGNOSIS_DATE', 'RESOLVED_DATE', 'IS_CHRONIC', 'SEVERITY',
        'DIAGNOSING_NPI', 'HCC_CODE', 'HCC_CATEGORY',
    ]
    rows = []
    dtypes = ['PRIMARY', 'SECONDARY', 'ADMITTING', 'DISCHARGE']
    severities = ['MILD', 'MODERATE', 'SEVERE', 'CRITICAL']
    hcc_cats = ['Diabetes', 'Heart Failure', 'COPD', 'CKD', 'Cancer', 'Mental Health', 'None']

    for i in range(n):
        diag = random.choice(ICD10_CODES)
        ddate = gen_date(2023, 2025)
        is_chronic = 'Y' if random.random() < 0.35 else 'N'
        resolved = '' if is_chronic == 'Y' else gen_date(2024, 2025) if random.random() < 0.6 else ''

        rows.append([
            f"DX{random.randint(1000000, 9999999)}",
            random.choice(member_ids) if member_ids else gen_member_id(),
            random.choice(encounter_ids) if encounter_ids and random.random() < 0.7 else '',
            diag[0], diag[1],
            random.choice(dtypes), ddate, resolved, is_chronic,
            random.choice(severities),
            gen_npi(),
            f"HCC{random.randint(1, 189)}" if random.random() < 0.4 else '',
            random.choice(hcc_cats),
        ])

    write_csv(os.path.join(RAW_DIR, 'diagnoses.csv'), headers, rows)


def generate_prescriptions(n=12000, member_ids=None, npis=None):
    print(f"\n  Generating {n} prescriptions...")
    headers = [
        'RX_ID', 'MEMBER_ID', 'PRESCRIBING_NPI', 'MEDICATION_NAME', 'MEDICATION_CLASS',
        'NDC_CODE', 'QUANTITY', 'DAYS_SUPPLY', 'REFILLS_AUTHORIZED', 'REFILLS_USED',
        'PRESCRIPTION_DATE', 'FILL_DATE', 'PHARMACY', 'COST', 'COPAY', 'STATUS',
    ]
    rows = []
    pharmacies = ['KP Pharmacy Oakland', 'KP Pharmacy LA', 'KP Mail Order', 'CVS', 'Walgreens', 'Rite Aid']
    statuses = ['FILLED', 'PENDING', 'CANCELLED', 'EXPIRED', 'TRANSFERRED']

    for i in range(n):
        med = random.choice(MEDICATIONS)
        pdate = gen_date(2023, 2025)
        fill = pdate if random.random() < 0.85 else ''
        refills_auth = random.randint(0, 11)
        refills_used = random.randint(0, refills_auth) if refills_auth > 0 else 0

        rows.append([
            gen_rx_id(),
            random.choice(member_ids) if member_ids else gen_member_id(),
            random.choice(npis) if npis else gen_npi(),
            med[0], med[1],
            f"NDC{random.randint(10000000000, 99999999999)}",
            random.choice([30, 60, 90, 14, 7, 120]),
            random.choice([7, 14, 30, 60, 90]),
            refills_auth, refills_used, pdate, fill,
            random.choice(pharmacies),
            round(random.uniform(5, 800), 2),
            random.choice([0, 5, 10, 15, 20, 25, 30, 50]),
            random.choices(statuses, weights=[70, 10, 5, 10, 5])[0],
        ])

    write_csv(os.path.join(RAW_DIR, 'prescriptions.csv'), headers, rows)


def generate_referrals(n=5000, member_ids=None, npis=None):
    print(f"\n  Generating {n} referrals...")
    headers = [
        'REFERRAL_ID', 'MEMBER_ID', 'REFERRING_NPI', 'REFERRED_TO_NPI', 'REFERRAL_DATE',
        'REFERRAL_REASON', 'URGENCY', 'REFERRAL_TYPE', 'STATUS', 'APPOINTMENT_DATE',
        'SPECIALTY', 'AUTHORIZATION_NUMBER', 'KP_REGION',
    ]
    rows = []
    reasons = [
        'Specialist evaluation', 'Second opinion', 'Surgery consultation', 'Diagnostic workup',
        'Pain management', 'Physical therapy', 'Mental health assessment', 'Cardiac evaluation',
        'Cancer screening', 'Chronic disease management', 'Post-op follow-up',
    ]
    urgencies = ['ROUTINE', 'URGENT', 'STAT', 'ELECTIVE']
    rtypes = ['INTERNAL', 'EXTERNAL', 'SELF_REFERRAL']
    statuses = ['APPROVED', 'PENDING', 'DENIED', 'COMPLETED', 'EXPIRED', 'CANCELLED']

    for i in range(n):
        rdate = gen_date(2023, 2025)
        appt = (datetime.strptime(rdate, '%Y-%m-%d') + timedelta(days=random.randint(3, 60))).strftime('%Y-%m-%d')

        rows.append([
            gen_referral_id(),
            random.choice(member_ids) if member_ids else gen_member_id(),
            random.choice(npis) if npis else gen_npi(),
            random.choice(npis) if npis else gen_npi(),
            rdate, random.choice(reasons),
            random.choices(urgencies, weights=[50, 25, 5, 20])[0],
            random.choices(rtypes, weights=[60, 30, 10])[0],
            random.choices(statuses, weights=[35, 20, 5, 30, 5, 5])[0],
            appt, random.choice(DEPARTMENTS),
            f"AUTH{random.randint(100000, 999999)}" if random.random() < 0.7 else '',
            random.choice(REGIONS),
        ])

    write_csv(os.path.join(RAW_DIR, 'referrals.csv'), headers, rows)


def generate_cpt_codes():
    print(f"\n  Generating {len(CPT_CODES)} CPT code reference records...")
    headers = ['CPT_CODE', 'DESCRIPTION', 'CATEGORY', 'RVU']
    rows = []

    CATEGORY_MAP = {
        '992': 'Evaluation & Management', '993': 'Evaluation & Management',
        '990': 'Evaluation & Management', '991': 'Evaluation & Management',
        '994': 'Evaluation & Management',
        '364': 'Laboratory', '850': 'Laboratory', '800': 'Laboratory',
        '810': 'Laboratory', '829': 'Laboratory', '830': 'Laboratory',
        '844': 'Laboratory', '878': 'Laboratory', '865': 'Laboratory',
        '883': 'Laboratory',
        '710': 'Radiology', '705': 'Radiology', '737': 'Radiology',
        '741': 'Radiology', '770': 'Radiology', '768': 'Radiology',
        '765': 'Radiology', '784': 'Radiology',
        '930': 'Cardiology', '931': 'Cardiology', '933': 'Cardiology',
        '958': 'Sleep Medicine',
        '908': 'Psychiatry', '903': 'Psychiatry', '908': 'Psychiatry',
        '961': 'Behavioral Health',
        '971': 'Physical Therapy', '975': 'Physical Therapy', '974': 'Physical Therapy',
        '906': 'Vaccines', '907': 'Vaccines',
        '963': 'Injection',
        '111': 'Surgery', '100': 'Surgery', '206': 'Surgery', '274': 'Surgery',
        '298': 'Surgery', '453': 'Surgery', '455': 'Surgery', '475': 'Surgery',
        '495': 'Surgery', '505': 'Surgery', '557': 'Surgery', '586': 'Surgery',
        '594': 'Surgery', '623': 'Surgery', '644': 'Surgery', '669': 'Surgery',
        '432': 'Surgery',
        '920': 'Ophthalmology',
    }

    for code, desc in CPT_CODES:
        prefix = code[:3]
        category = CATEGORY_MAP.get(prefix, 'Other')
        rvu = round(random.uniform(0.5, 25.0), 2)
        rows.append([code, desc, category, rvu])

    write_csv(os.path.join(RAW_DIR, 'cpt_codes.csv'), headers, rows)


def generate_appointments(n=10000, member_ids=None, npis=None):
    print(f"\n  Generating {n} appointments...")
    headers = [
        'APPOINTMENT_ID', 'MEMBER_ID', 'PROVIDER_NPI', 'APPOINTMENT_DATE', 'APPOINTMENT_TIME',
        'APPOINTMENT_TYPE', 'DEPARTMENT', 'FACILITY', 'KP_REGION', 'STATUS',
        'REASON', 'IS_PCP_VISIT', 'DURATION_MINUTES', 'CHECK_IN_TIME', 'CHECK_OUT_TIME',
    ]
    rows = []
    appt_types = ['OFFICE_VISIT', 'TELEHEALTH', 'PROCEDURE', 'LAB', 'IMAGING', 'VACCINE', 'WELLNESS_CHECK']
    statuses = ['SCHEDULED', 'COMPLETED', 'CANCELLED', 'NO_SHOW', 'RESCHEDULED', 'CHECKED_IN']
    reasons = [
        'Annual wellness exam', 'Follow-up visit', 'New patient visit', 'Medication review',
        'Lab work', 'Imaging study', 'Pre-operative evaluation', 'Post-operative check',
        'Chronic disease management', 'Acute illness', 'Vaccination', 'Mental health consult',
        'Physical therapy', 'Specialist referral', 'Screening procedure',
    ]
    times = ['08:00', '08:30', '09:00', '09:30', '10:00', '10:30', '11:00', '11:30',
             '13:00', '13:30', '14:00', '14:30', '15:00', '15:30', '16:00', '16:30']

    for i in range(n):
        region = random.choice(REGIONS)
        adate = gen_date(2024, 2026)
        atime = random.choice(times)
        status = random.choices(statuses, weights=[25, 40, 10, 8, 7, 10])[0]
        is_pcp = random.choices(['Y', 'N'], weights=[40, 60])[0]
        duration = random.choice([15, 20, 30, 45, 60])

        rows.append([
            f"APT{random.randint(1000000, 9999999)}",
            random.choice(member_ids) if member_ids else gen_member_id(),
            random.choice(npis) if npis else gen_npi(),
            adate, atime,
            random.choices(appt_types, weights=[35, 20, 10, 10, 8, 7, 10])[0],
            random.choice(DEPARTMENTS),
            random.choice(FACILITIES[region]), region,
            status, random.choice(reasons), is_pcp, duration,
            f"{atime}" if status in ('COMPLETED', 'CHECKED_IN') else '',
            '' if status != 'COMPLETED' else f"{int(atime.split(':')[0]) + 1}:{atime.split(':')[1]}",
        ])

    write_csv(os.path.join(RAW_DIR, 'appointments.csv'), headers, rows)


def main():
    print("=" * 60)
    print("  SYNTHETIC HEALTHCARE DATA GENERATOR — v2 (Large Scale)")
    print("=" * 60)

    mrns, member_ids = generate_members(25000)
    npis = generate_providers(3000)
    encounter_ids = generate_encounters(50000, member_ids, npis)
    generate_claims(60000, member_ids, npis, encounter_ids)
    generate_diagnoses(20000, member_ids, encounter_ids)
    generate_prescriptions(12000, member_ids, npis)
    generate_referrals(5000, member_ids, npis)
    generate_appointments(10000, member_ids, npis)
    generate_cpt_codes()

    total = 25000 + 3000 + 50000 + 60000 + 20000 + 12000 + 5000 + 10000 + len(CPT_CODES)
    print(f"\n{'=' * 60}")
    print(f"  DONE — {total:,} total records across 9 files")
    print(f"  Location: {RAW_DIR}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
