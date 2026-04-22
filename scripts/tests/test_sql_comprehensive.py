import sys, os, re, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dynamic_sql_engine import DynamicSQLEngine
from production import setup_logging, resolve_config, DatabasePool


TESTS = [
    {"q": "how many claims are there", "tables": ["claims"], "cols": ["COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "claims-basic"},
    {"q": "total number of claims", "tables": ["claims"], "cols": ["COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "claims-basic"},
    {"q": "show me all claims", "tables": ["claims"], "cols": ["CLAIM_ID"], "agg": None, "min_rows": 1, "cat": "claims-basic"},
    {"q": "claims count by status", "tables": ["claims"], "cols": ["CLAIM_STATUS", "COUNT"], "agg": "COUNT", "min_rows": 2, "cat": "claims-group"},
    {"q": "claim status distribution", "tables": ["claims"], "cols": ["CLAIM_STATUS", "COUNT"], "agg": "COUNT", "min_rows": 2, "cat": "claims-group"},
    {"q": "claims breakdown by region", "tables": ["claims"], "cols": ["KP_REGION", "COUNT"], "agg": "COUNT", "min_rows": 2, "cat": "claims-group"},
    {"q": "claims count by plan type", "tables": ["claims"], "cols": ["PLAN_TYPE", "COUNT"], "agg": "COUNT", "min_rows": 2, "cat": "claims-group"},
    {"q": "how many denied claims", "tables": ["claims"], "cols": ["COUNT"], "agg": "COUNT", "min_rows": 1, "filter": "DENIED", "cat": "claims-filter"},
    {"q": "show denied claims", "tables": ["claims"], "cols": ["CLAIM_STATUS"], "agg": None, "min_rows": 1, "filter": "DENIED", "cat": "claims-filter"},
    {"q": "claims with status pending", "tables": ["claims"], "cols": ["CLAIM_STATUS"], "agg": None, "min_rows": 1, "filter": "PENDING", "cat": "claims-filter"},
    {"q": "total billed amount", "tables": ["claims"], "cols": ["BILLED_AMOUNT"], "agg": "SUM", "min_rows": 1, "cat": "claims-agg"},
    {"q": "average paid amount", "tables": ["claims"], "cols": ["PAID_AMOUNT"], "agg": "AVG", "min_rows": 1, "cat": "claims-agg"},
    {"q": "total billed amount by region", "tables": ["claims"], "cols": ["KP_REGION", "BILLED_AMOUNT"], "agg": "SUM", "min_rows": 2, "cat": "claims-agg"},
    {"q": "average copay by plan type", "tables": ["claims"], "cols": ["PLAN_TYPE", "COPAY"], "agg": "AVG", "min_rows": 2, "cat": "claims-agg"},
    {"q": "highest billed amount", "tables": ["claims"], "cols": ["BILLED_AMOUNT"], "agg": None, "min_rows": 1, "cat": "claims-agg"},
    {"q": "total paid amount by facility", "tables": ["claims"], "cols": ["FACILITY", "PAID_AMOUNT"], "agg": "SUM", "min_rows": 2, "cat": "claims-agg"},
    {"q": "denial reasons breakdown", "tables": ["claims"], "cols": ["DENIAL_REASON", "COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "claims-group"},
    {"q": "claims by CPT code", "tables": ["claims"], "cols": ["CPT_CODE"], "agg": None, "min_rows": 1, "cat": "claims-group"},
    {"q": "top 10 CPT codes by claim count", "tables": ["claims"], "cols": ["CPT_CODE", "COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "claims-topn"},
    {"q": "top 5 facilities by billed amount", "tables": ["claims"], "cols": ["FACILITY", "BILLED_AMOUNT"], "agg": None, "min_rows": 1, "cat": "claims-topn"},
    {"q": "claims per month", "tables": ["claims"], "cols": ["SERVICE_DATE"], "agg": "COUNT", "min_rows": 1, "cat": "claims-time"},
    {"q": "average billed amount per claim type", "tables": ["claims"], "cols": ["CLAIM_TYPE", "BILLED_AMOUNT"], "agg": "AVG", "min_rows": 1, "cat": "claims-agg"},
    {"q": "sum of allowed amount by region", "tables": ["claims"], "cols": ["KP_REGION", "ALLOWED_AMOUNT"], "agg": "SUM", "min_rows": 2, "cat": "claims-agg"},
    {"q": "how many claims in NCAL region", "tables": ["claims"], "cols": ["COUNT"], "agg": "COUNT", "min_rows": 1, "filter": "NCAL", "cat": "claims-filter"},
    {"q": "denied claims by denial reason", "tables": ["claims"], "cols": ["DENIAL_REASON", "COUNT"], "agg": "COUNT", "min_rows": 1, "filter": "DENIED", "cat": "claims-filter"},

    {"q": "how many members do we have", "tables": ["members"], "cols": ["COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "members-basic"},
    {"q": "total number of patients", "tables": ["members"], "cols": ["COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "members-basic"},
    {"q": "member count by gender", "tables": ["members"], "cols": ["GENDER", "COUNT"], "agg": "COUNT", "min_rows": 2, "cat": "members-group"},
    {"q": "members by region", "tables": ["members"], "cols": ["KP_REGION", "COUNT"], "agg": "COUNT", "min_rows": 2, "cat": "members-group"},
    {"q": "member distribution by plan type", "tables": ["members"], "cols": ["PLAN_TYPE", "COUNT"], "agg": "COUNT", "min_rows": 2, "cat": "members-group"},
    {"q": "average risk score by region", "tables": ["members"], "cols": ["KP_REGION", "RISK_SCORE"], "agg": "AVG", "min_rows": 2, "cat": "members-agg"},
    {"q": "members with high risk score", "tables": ["members"], "cols": ["RISK_SCORE"], "agg": None, "min_rows": 1, "cat": "members-filter"},
    {"q": "member count by race", "tables": ["members"], "cols": ["RACE", "COUNT"], "agg": "COUNT", "min_rows": 2, "cat": "members-group"},
    {"q": "members by language", "tables": ["members"], "cols": ["LANGUAGE", "COUNT"], "agg": "COUNT", "min_rows": 2, "cat": "members-group"},
    {"q": "how many members enrolled in HMO", "tables": ["members"], "cols": ["COUNT"], "agg": "COUNT", "min_rows": 1, "filter": "HMO", "cat": "members-filter"},
    {"q": "average risk score", "tables": ["members"], "cols": ["RISK_SCORE"], "agg": "AVG", "min_rows": 1, "cat": "members-agg"},
    {"q": "members with chronic conditions", "tables": ["members"], "cols": ["CHRONIC_CONDITIONS"], "agg": None, "min_rows": 1, "cat": "members-filter"},
    {"q": "gender distribution of members", "tables": ["members"], "cols": ["GENDER", "COUNT"], "agg": "COUNT", "min_rows": 2, "cat": "members-group"},

    {"q": "how many encounters", "tables": ["encounters"], "cols": ["COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "encounters-basic"},
    {"q": "total visits", "tables": ["encounters"], "cols": ["COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "encounters-basic"},
    {"q": "encounters by visit type", "tables": ["encounters"], "cols": ["VISIT_TYPE", "COUNT"], "agg": "COUNT", "min_rows": 2, "cat": "encounters-group"},
    {"q": "visit type distribution", "tables": ["encounters"], "cols": ["VISIT_TYPE", "COUNT"], "agg": "COUNT", "min_rows": 2, "cat": "encounters-group"},
    {"q": "how many emergency visits", "tables": ["encounters"], "cols": ["COUNT"], "agg": "COUNT", "min_rows": 1, "filter": "EMERGENCY", "cat": "encounters-filter"},
    {"q": "how many inpatient encounters", "tables": ["encounters"], "cols": ["COUNT"], "agg": "COUNT", "min_rows": 1, "filter": "INPATIENT", "cat": "encounters-filter"},
    {"q": "encounters by department", "tables": ["encounters"], "cols": ["DEPARTMENT", "COUNT"], "agg": "COUNT", "min_rows": 2, "cat": "encounters-group"},
    {"q": "average length of stay", "tables": ["encounters"], "cols": ["LENGTH_OF_STAY"], "agg": "AVG", "min_rows": 1, "cat": "encounters-agg"},
    {"q": "average length of stay by visit type", "tables": ["encounters"], "cols": ["VISIT_TYPE", "LENGTH_OF_STAY"], "agg": "AVG", "min_rows": 2, "cat": "encounters-agg"},
    {"q": "encounters by facility", "tables": ["encounters"], "cols": ["FACILITY"], "agg": None, "min_rows": 1, "cat": "encounters-group"},
    {"q": "telehealth visits count", "tables": ["encounters"], "cols": ["COUNT"], "agg": "COUNT", "min_rows": 1, "filter": "TELEHEALTH", "cat": "encounters-filter"},
    {"q": "encounter status breakdown", "tables": ["encounters"], "cols": ["ENCOUNTER_STATUS", "COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "encounters-group"},
    {"q": "top departments by encounter volume", "tables": ["encounters"], "cols": ["DEPARTMENT", "COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "encounters-topn"},
    {"q": "disposition distribution", "tables": ["encounters"], "cols": ["DISPOSITION", "COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "encounters-group"},
    {"q": "encounters per region", "tables": ["encounters"], "cols": ["KP_REGION", "COUNT"], "agg": "COUNT", "min_rows": 2, "cat": "encounters-group"},

    {"q": "how many diagnoses", "tables": ["diagnoses"], "cols": ["COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "diagnoses-basic"},
    {"q": "top 10 diagnosis codes", "tables": ["diagnoses"], "cols": ["ICD10_CODE", "COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "diagnoses-topn"},
    {"q": "diagnoses by severity", "tables": ["diagnoses"], "cols": ["SEVERITY", "COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "diagnoses-group"},
    {"q": "chronic vs non-chronic diagnoses", "tables": ["diagnoses"], "cols": ["IS_CHRONIC", "COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "diagnoses-group"},
    {"q": "top HCC categories", "tables": ["diagnoses"], "cols": ["HCC_CATEGORY", "COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "diagnoses-topn"},
    {"q": "diagnosis type distribution", "tables": ["diagnoses"], "cols": ["DIAGNOSIS_TYPE", "COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "diagnoses-group"},
    {"q": "most common ICD10 codes", "tables": ["diagnoses"], "cols": ["ICD10_CODE", "COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "diagnoses-topn"},

    {"q": "how many prescriptions", "tables": ["prescriptions"], "cols": ["COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "rx-basic"},
    {"q": "total prescriptions", "tables": ["prescriptions"], "cols": ["COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "rx-basic"},
    {"q": "prescriptions by medication class", "tables": ["prescriptions"], "cols": ["MEDICATION_CLASS", "COUNT"], "agg": "COUNT", "min_rows": 2, "cat": "rx-group"},
    {"q": "top 10 medications by prescription count", "tables": ["prescriptions"], "cols": ["MEDICATION_NAME", "COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "rx-topn"},
    {"q": "average prescription cost", "tables": ["prescriptions"], "cols": ["COST"], "agg": "AVG", "min_rows": 1, "cat": "rx-agg"},
    {"q": "total medication cost by class", "tables": ["prescriptions"], "cols": ["MEDICATION_CLASS", "COST"], "agg": "SUM", "min_rows": 2, "cat": "rx-agg"},
    {"q": "prescriptions by pharmacy", "tables": ["prescriptions"], "cols": ["PHARMACY"], "agg": None, "min_rows": 1, "cat": "rx-group"},
    {"q": "average days supply", "tables": ["prescriptions"], "cols": ["DAYS_SUPPLY"], "agg": "AVG", "min_rows": 1, "cat": "rx-agg"},
    {"q": "prescription status breakdown", "tables": ["prescriptions"], "cols": ["STATUS", "COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "rx-group"},

    {"q": "how many providers", "tables": ["providers"], "cols": ["COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "providers-basic"},
    {"q": "providers by specialty", "tables": ["providers"], "cols": ["SPECIALTY", "COUNT"], "agg": "COUNT", "min_rows": 2, "cat": "providers-group"},
    {"q": "provider count by region", "tables": ["providers"], "cols": ["KP_REGION", "COUNT"], "agg": "COUNT", "min_rows": 2, "cat": "providers-group"},
    {"q": "providers by department", "tables": ["providers"], "cols": ["DEPARTMENT", "COUNT"], "agg": "COUNT", "min_rows": 2, "cat": "providers-group"},
    {"q": "average panel size by specialty", "tables": ["providers"], "cols": ["SPECIALTY", "PANEL_SIZE"], "agg": "AVG", "min_rows": 2, "cat": "providers-agg"},
    {"q": "providers accepting new patients", "tables": ["providers"], "cols": ["ACCEPTS_NEW_PATIENTS"], "agg": None, "min_rows": 1, "cat": "providers-filter"},
    {"q": "provider type distribution", "tables": ["providers"], "cols": ["PROVIDER_TYPE", "COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "providers-group"},

    {"q": "how many referrals", "tables": ["referrals"], "cols": ["COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "referrals-basic"},
    {"q": "referrals by urgency", "tables": ["referrals"], "cols": ["URGENCY", "COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "referrals-group"},
    {"q": "referral type distribution", "tables": ["referrals"], "cols": ["REFERRAL_TYPE", "COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "referrals-group"},
    {"q": "referrals by specialty", "tables": ["referrals"], "cols": ["SPECIALTY", "COUNT"], "agg": "COUNT", "min_rows": 2, "cat": "referrals-group"},
    {"q": "referral status breakdown", "tables": ["referrals"], "cols": ["STATUS", "COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "referrals-group"},
    {"q": "top referral reasons", "tables": ["referrals"], "cols": ["REFERRAL_REASON", "COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "referrals-topn"},

    {"q": "claims count by member", "tables": ["claims"], "cols": ["MEMBER_ID", "COUNT"], "agg": "COUNT", "min_rows": 5, "cat": "join-member-claims"},
    {"q": "which members have most claims", "tables": ["claims", "members"], "cols": ["MEMBER_ID", "COUNT"], "agg": "COUNT", "min_rows": 5, "cat": "join-member-claims"},
    {"q": "total billed amount per member", "tables": ["claims", "members"], "cols": ["MEMBER_ID", "BILLED_AMOUNT"], "agg": "SUM", "min_rows": 5, "cat": "join-member-claims"},
    {"q": "average paid amount by member", "tables": ["claims"], "cols": ["MEMBER_ID", "PAID_AMOUNT"], "agg": "AVG", "min_rows": 5, "cat": "join-member-claims"},
    {"q": "denied claims per member", "tables": ["claims"], "cols": ["MEMBER_ID", "COUNT"], "agg": "COUNT", "min_rows": 1, "filter": "DENIED", "cat": "join-member-claims"},
    {"q": "members with highest billed amount", "tables": ["claims", "members"], "cols": ["MEMBER_ID", "BILLED_AMOUNT"], "agg": "SUM", "min_rows": 1, "cat": "join-member-claims"},
    {"q": "claims by member gender", "tables": ["claims", "members"], "cols": ["GENDER", "COUNT"], "agg": "COUNT", "min_rows": 2, "cat": "join-member-claims"},
    {"q": "total paid amount by member plan type", "tables": ["claims"], "cols": ["PLAN_TYPE", "PAID_AMOUNT"], "agg": "SUM", "min_rows": 2, "cat": "join-member-claims"},

    {"q": "encounters per member", "tables": ["encounters"], "cols": ["MEMBER_ID", "COUNT"], "agg": "COUNT", "min_rows": 5, "cat": "join-member-encounters"},
    {"q": "which members have most emergency visits", "tables": ["encounters"], "cols": ["MEMBER_ID", "COUNT"], "agg": "COUNT", "min_rows": 1, "filter": "EMERGENCY", "cat": "join-member-encounters"},
    {"q": "average length of stay by member", "tables": ["encounters"], "cols": ["MEMBER_ID", "LENGTH_OF_STAY"], "agg": "AVG", "min_rows": 5, "cat": "join-member-encounters"},
    {"q": "emergency visits by region", "tables": ["encounters"], "cols": ["KP_REGION", "COUNT"], "agg": "COUNT", "min_rows": 2, "filter": "EMERGENCY", "cat": "join-member-encounters"},
    {"q": "inpatient encounters by facility", "tables": ["encounters"], "cols": ["FACILITY", "COUNT"], "agg": "COUNT", "min_rows": 1, "filter": "INPATIENT", "cat": "join-member-encounters"},

    {"q": "claims by provider specialty", "tables": ["claims", "providers"], "cols": ["SPECIALTY", "COUNT"], "agg": "COUNT", "min_rows": 2, "cat": "join-claims-providers"},
    {"q": "top 5 providers by total billed amount", "tables": ["claims", "providers"], "cols": ["NPI", "BILLED_AMOUNT"], "agg": None, "min_rows": 1, "cat": "join-claims-providers"},
    {"q": "average paid amount by provider specialty", "tables": ["claims", "providers"], "cols": ["SPECIALTY", "PAID_AMOUNT"], "agg": "AVG", "min_rows": 2, "cat": "join-claims-providers"},
    {"q": "which provider has most denied claims", "tables": ["claims", "providers"], "cols": ["NPI", "COUNT"], "agg": "COUNT", "min_rows": 1, "filter": "DENIED", "cat": "join-claims-providers"},
    {"q": "claims per provider", "tables": ["claims"], "cols": ["RENDERING_NPI", "COUNT"], "agg": "COUNT", "min_rows": 5, "cat": "join-claims-providers"},
    {"q": "total billed by specialty", "tables": ["claims", "providers"], "cols": ["SPECIALTY", "BILLED_AMOUNT"], "agg": "SUM", "min_rows": 2, "cat": "join-claims-providers"},

    {"q": "top diagnosis codes in emergency visits", "tables": ["diagnoses", "encounters"], "cols": ["ICD10_CODE", "COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "join-encounters-diag"},
    {"q": "diagnosis count by visit type", "tables": ["diagnoses", "encounters"], "cols": ["VISIT_TYPE", "COUNT"], "agg": "COUNT", "min_rows": 2, "cat": "join-encounters-diag"},
    {"q": "chronic diagnoses by department", "tables": ["diagnoses", "providers"], "cols": ["DEPARTMENT", "COUNT"], "agg": "COUNT", "min_rows": 0, "filter": "CHRONIC", "cat": "join-encounters-diag"},

    {"q": "prescriptions per member", "tables": ["prescriptions"], "cols": ["MEMBER_ID", "COUNT"], "agg": "COUNT", "min_rows": 5, "cat": "join-member-rx"},
    {"q": "which members have most prescriptions", "tables": ["prescriptions", "members"], "cols": ["MEMBER_ID", "COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "join-member-rx"},
    {"q": "average prescription cost by member", "tables": ["prescriptions"], "cols": ["MEMBER_ID", "COST"], "agg": "AVG", "min_rows": 5, "cat": "join-member-rx"},
    {"q": "total medication cost per member", "tables": ["prescriptions"], "cols": ["MEMBER_ID", "COST"], "agg": "SUM", "min_rows": 5, "cat": "join-member-rx"},

    {"q": "referrals per member", "tables": ["referrals"], "cols": ["MEMBER_ID", "COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "join-member-referrals"},
    {"q": "urgent referrals by region", "tables": ["referrals"], "cols": ["KP_REGION", "COUNT"], "agg": "COUNT", "min_rows": 1, "filter": "URGENT", "cat": "join-referrals-region"},

    {"q": "average billed amount by provider specialty for denied claims", "tables": ["claims", "providers"], "cols": ["SPECIALTY", "BILLED_AMOUNT"], "agg": "AVG", "min_rows": 1, "filter": "DENIED", "cat": "complex-3table"},
    {"q": "claims by member gender and region", "tables": ["claims", "members"], "cols": ["GENDER", "KP_REGION"], "agg": None, "min_rows": 1, "cat": "complex-3table"},
    {"q": "emergency visits per provider specialty", "tables": ["encounters", "providers"], "cols": ["SPECIALTY", "COUNT"], "agg": "COUNT", "min_rows": 1, "filter": "EMERGENCY", "cat": "complex-3table"},

    {"q": "which region has most claims", "tables": ["claims"], "cols": ["KP_REGION", "COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "which-what"},
    {"q": "which facility has highest billed amount", "tables": ["claims"], "cols": ["FACILITY", "BILLED_AMOUNT"], "agg": "SUM", "min_rows": 1, "cat": "which-what"},
    {"q": "which specialty has most encounters", "tables": ["encounters", "providers"], "cols": ["SPECIALTY", "COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "which-what"},
    {"q": "which plan type has most members", "tables": ["members"], "cols": ["PLAN_TYPE", "COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "which-what"},
    {"q": "which medication is most prescribed", "tables": ["prescriptions"], "cols": ["MEDICATION_NAME", "COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "which-what"},
    {"q": "which department has most encounters", "tables": ["encounters"], "cols": ["DEPARTMENT", "COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "which-what"},
    {"q": "what are the top denial reasons", "tables": ["claims"], "cols": ["DENIAL_REASON", "COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "which-what"},
    {"q": "what is the average billed amount", "tables": ["claims"], "cols": ["BILLED_AMOUNT"], "agg": "AVG", "min_rows": 1, "cat": "which-what"},
    {"q": "what is the total paid amount", "tables": ["claims"], "cols": ["PAID_AMOUNT"], "agg": "SUM", "min_rows": 1, "cat": "which-what"},

    {"q": "top 5 members by claims count", "tables": ["claims", "members"], "cols": ["MEMBER_ID", "COUNT"], "agg": "COUNT", "min_rows": 5, "cat": "topn"},
    {"q": "top 10 providers by billed amount", "tables": ["claims", "providers"], "cols": ["NPI", "BILLED_AMOUNT"], "agg": None, "min_rows": 5, "cat": "topn"},
    {"q": "top 3 regions by denied claims", "tables": ["claims"], "cols": ["KP_REGION", "COUNT"], "agg": "COUNT", "min_rows": 3, "filter": "DENIED", "cat": "topn"},
    {"q": "bottom 5 facilities by paid amount", "tables": ["claims"], "cols": ["FACILITY", "PAID_AMOUNT"], "agg": None, "min_rows": 5, "cat": "topn"},
    {"q": "top 5 diagnosis codes", "tables": ["diagnoses"], "cols": ["ICD10_CODE", "COUNT"], "agg": "COUNT", "min_rows": 5, "cat": "topn"},
    {"q": "top medications by cost", "tables": ["prescriptions"], "cols": ["MEDICATION_NAME", "COST"], "agg": None, "min_rows": 1, "cat": "topn"},

    {"q": "members with more than 3 claims", "tables": ["claims"], "cols": ["MEMBER_ID", "COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "having"},
    {"q": "providers with more than 10 claims", "tables": ["claims"], "cols": ["RENDERING_NPI", "COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "having"},
    {"q": "facilities with more than 100 encounters", "tables": ["encounters"], "cols": ["FACILITY", "COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "having"},
    {"q": "members with more than 2 prescriptions", "tables": ["prescriptions"], "cols": ["MEMBER_ID", "COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "having"},

    {"q": "show me the claim count for each region", "tables": ["claims"], "cols": ["KP_REGION", "COUNT"], "agg": "COUNT", "min_rows": 2, "cat": "natural"},
    {"q": "I want to see claims grouped by facility", "tables": ["claims"], "cols": ["FACILITY"], "agg": None, "min_rows": 1, "cat": "natural"},
    {"q": "can you tell me how many patients we have", "tables": ["members"], "cols": ["COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "natural"},
    {"q": "give me the total spend by region", "tables": ["claims"], "cols": ["KP_REGION", "BILLED_AMOUNT"], "agg": "SUM", "min_rows": 2, "cat": "natural"},
    {"q": "list all providers in cardiology", "tables": ["providers"], "cols": ["SPECIALTY"], "agg": None, "min_rows": 0, "cat": "natural"},
    {"q": "what does the denial rate look like by facility", "tables": ["claims"], "cols": ["FACILITY"], "agg": None, "min_rows": 1, "filter": "", "cat": "natural"},
    {"q": "break down encounters by type", "tables": ["encounters"], "cols": ["VISIT_TYPE"], "agg": None, "min_rows": 2, "cat": "natural"},
    {"q": "how is the cost distributed across regions", "tables": ["claims"], "cols": ["KP_REGION"], "agg": None, "min_rows": 2, "cat": "natural"},

    {"q": "give me claims count by memeber", "tables": ["claims"], "cols": ["MEMBER_ID", "COUNT"], "agg": "COUNT", "min_rows": 5, "cat": "typo"},
    {"q": "which memebers have higest number of claims", "tables": ["claims"], "cols": ["MEMBER_ID", "COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "typo"},
    {"q": "show claim staus distribution", "tables": ["claims"], "cols": ["CLAIM_STATUS", "COUNT"], "agg": "COUNT", "min_rows": 2, "cat": "typo"},
    {"q": "top 5 provders by billed amount", "tables": ["claims"], "cols": ["BILLED_AMOUNT"], "agg": None, "min_rows": 1, "cat": "typo"},
    {"q": "claims by reigon", "tables": ["claims"], "cols": ["KP_REGION"], "agg": None, "min_rows": 2, "cat": "typo"},
    {"q": "total biled amount by member", "tables": ["claims"], "cols": ["MEMBER_ID", "BILLED_AMOUNT"], "agg": "SUM", "min_rows": 5, "cat": "typo"},
    {"q": "encountr count by visit tpye", "tables": ["encounters"], "cols": ["VISIT_TYPE", "COUNT"], "agg": "COUNT", "min_rows": 2, "cat": "typo"},
    {"q": "perscriptions by medicaton class", "tables": ["prescriptions"], "cols": ["MEDICATION_CLASS"], "agg": None, "min_rows": 1, "cat": "typo"},
    {"q": "wich facilty has most claims", "tables": ["claims"], "cols": ["FACILITY", "COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "typo"},
    {"q": "averge paid amout by reigon", "tables": ["claims"], "cols": ["KP_REGION", "PAID_AMOUNT"], "agg": "AVG", "min_rows": 2, "cat": "typo"},

    {"q": "claims", "tables": ["claims"], "cols": ["CLAIM_ID"], "agg": None, "min_rows": 1, "cat": "edge"},
    {"q": "show everything in claims table", "tables": ["claims"], "cols": ["CLAIM_ID"], "agg": None, "min_rows": 1, "cat": "edge"},
    {"q": "count", "tables": [], "cols": [], "agg": None, "min_rows": 0, "cat": "edge"},
    {"q": "all data", "tables": [], "cols": [], "agg": None, "min_rows": 0, "cat": "edge"},
    {"q": "claims and encounters", "tables": ["claims", "encounters"], "cols": [], "agg": None, "min_rows": 0, "cat": "edge"},

    {"q": "claims in 2024", "tables": ["claims"], "cols": ["SERVICE_DATE"], "agg": None, "min_rows": 1, "filter": "2024", "cat": "date"},
    {"q": "encounters in 2025", "tables": ["encounters"], "cols": ["SERVICE_DATE"], "agg": None, "min_rows": 1, "filter": "2025", "cat": "date"},
    {"q": "claims trend by month", "tables": ["claims"], "cols": ["SERVICE_DATE"], "agg": "COUNT", "min_rows": 1, "cat": "date"},
    {"q": "monthly claim volume", "tables": ["claims"], "cols": ["SERVICE_DATE"], "agg": "COUNT", "min_rows": 1, "cat": "date"},

    {"q": "denial rate by region", "tables": ["claims"], "cols": ["KP_REGION"], "agg": None, "min_rows": 2, "cat": "analytical"},
    {"q": "average cost per encounter by visit type", "tables": ["claims", "encounters"], "cols": ["VISIT_TYPE"], "agg": "AVG", "min_rows": 2, "cat": "analytical"},
    {"q": "claims per provider per region", "tables": ["claims"], "cols": ["RENDERING_NPI", "KP_REGION"], "agg": None, "min_rows": 1, "cat": "analytical"},
    {"q": "member utilization by plan type", "tables": ["encounters", "members"], "cols": ["PLAN_TYPE"], "agg": None, "min_rows": 2, "cat": "analytical"},
    {"q": "high cost claims above 10000", "tables": ["claims"], "cols": ["BILLED_AMOUNT"], "agg": None, "min_rows": 1, "filter": "10000", "cat": "analytical"},
    {"q": "readmission rate by facility", "tables": ["encounters"], "cols": ["FACILITY"], "agg": None, "min_rows": 1, "cat": "analytical"},

    {"q": "count of foster care members for each region on medicaid", "tables": ["members"], "cols": ["KP_REGION", "COUNT"], "agg": "COUNT", "min_rows": 1, "filter": "Medicaid", "cat": "domain-age"},
    {"q": "how many pediatric members are enrolled in each region", "tables": ["members"], "cols": ["KP_REGION", "COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "domain-age"},
    {"q": "count of senior patients on medicare by region", "tables": ["members"], "cols": ["KP_REGION", "COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "domain-age"},
    {"q": "total claims for elderly members by region", "tables": ["claims", "members"], "cols": ["KP_REGION", "COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "domain-age"},

    {"q": "how many diabetic members by region", "tables": ["claims", "members"], "cols": ["KP_REGION", "COUNT"], "agg": "COUNT", "min_rows": 1, "filter": "E11", "cat": "domain-clinical"},
    {"q": "total claims for members with hypertension", "tables": ["claims"], "cols": ["COUNT"], "agg": "COUNT", "min_rows": 1, "filter": "I10", "cat": "domain-clinical"},
    {"q": "average billed amount for asthma claims by region", "tables": ["claims"], "cols": ["KP_REGION", "BILLED_AMOUNT"], "agg": "AVG", "min_rows": 1, "filter": "J45", "cat": "domain-clinical"},
    {"q": "how many members with depression by gender", "tables": ["claims", "members"], "cols": ["GENDER", "COUNT"], "agg": "COUNT", "min_rows": 1, "filter": "F32", "cat": "domain-clinical"},
    {"q": "obesity claims by facility", "tables": ["claims"], "cols": ["FACILITY", "COUNT"], "agg": "COUNT", "min_rows": 1, "filter": "E66", "cat": "domain-clinical"},

    {"q": "count of high risk members by plan type", "tables": ["members"], "cols": ["PLAN_TYPE", "COUNT"], "agg": "COUNT", "min_rows": 1, "filter": "RISK_SCORE", "cat": "domain-risk"},
    {"q": "comorbid patients by region on medicaid", "tables": ["members"], "cols": ["KP_REGION", "COUNT"], "agg": "COUNT", "min_rows": 1, "filter": "Medicaid", "cat": "domain-risk"},
    {"q": "average billed amount for medicare members by region", "tables": ["members", "claims"], "cols": ["KP_REGION", "BILLED_AMOUNT"], "agg": "AVG", "min_rows": 1, "filter": "Medicare", "cat": "domain-risk"},
    {"q": "denial rate for medicaid members by region", "tables": ["members", "claims"], "cols": ["KP_REGION", "DENIED"], "agg": None, "min_rows": 1, "filter": "Medicaid", "cat": "domain-risk"},

    {"q": "count of emergency visits for medicaid members by facility", "tables": ["encounters", "members"], "cols": ["FACILITY", "COUNT"], "agg": "COUNT", "min_rows": 1, "filter": "EMERGENCY", "cat": "domain-visit"},
    {"q": "ed visits by region", "tables": ["encounters"], "cols": ["KP_REGION", "COUNT"], "agg": "COUNT", "min_rows": 1, "filter": "EMERGENCY", "cat": "domain-visit"},
    {"q": "telehealth visits by department", "tables": ["encounters"], "cols": ["DEPARTMENT", "COUNT"], "agg": "COUNT", "min_rows": 1, "filter": "TELEHEALTH", "cat": "domain-visit"},

    {"q": "members on antidepressants by region", "tables": ["prescriptions", "members"], "cols": ["KP_REGION", "COUNT"], "agg": "COUNT", "min_rows": 1, "filter": "Depression", "cat": "domain-medication"},
    {"q": "how many patients on pain medication by plan type", "tables": ["prescriptions", "members"], "cols": ["PLAN_TYPE", "COUNT"], "agg": "COUNT", "min_rows": 1, "filter": "Pain", "cat": "domain-medication"},

    {"q": "institutional claims by region", "tables": ["claims"], "cols": ["KP_REGION", "COUNT"], "agg": "COUNT", "min_rows": 1, "filter": "INSTITUTIONAL", "cat": "domain-complex"},
    {"q": "total claims for elderly diabetic patients on medicare by region", "tables": ["claims", "members"], "cols": ["KP_REGION", "COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "domain-complex"},
    {"q": "how many comorbid patients on medicaid by region", "tables": ["members"], "cols": ["KP_REGION", "COUNT"], "agg": "COUNT", "min_rows": 1, "filter": "Medicaid", "cat": "domain-complex"},

    {"q": "what is the most prescribed medication", "tables": ["prescriptions"], "cols": ["MEDICATION_NAME", "COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "production-natural"},
    {"q": "which facilities handle the most emergency cases", "tables": ["encounters"], "cols": ["FACILITY", "COUNT"], "agg": "COUNT", "min_rows": 1, "filter": "EMERGENCY", "cat": "production-natural"},
    {"q": "give me a count of members enrolled in each plan", "tables": ["members"], "cols": ["PLAN_TYPE", "COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "production-natural"},
    {"q": "how many claims were denied for each region", "tables": ["claims"], "cols": ["KP_REGION", "COUNT"], "agg": "COUNT", "min_rows": 1, "filter": "DENIED", "cat": "production-natural"},
    {"q": "show me all the inpatient encounters", "tables": ["encounters"], "cols": ["ENCOUNTER_ID"], "agg": None, "min_rows": 1, "filter": "INPATIENT", "cat": "production-natural"},
    {"q": "breakdown of claims by status", "tables": ["claims"], "cols": ["CLAIM_STATUS", "COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "production-natural"},
    {"q": "distribution of members across risk levels", "tables": ["members"], "cols": ["RISK_SCORE", "COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "production-natural"},

    {"q": "top specialties by encounter volume", "tables": ["encounters", "providers"], "cols": ["SPECIALTY", "COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "production-cross"},
    {"q": "average length of stay by facility", "tables": ["encounters"], "cols": ["FACILITY", "LENGTH_OF_STAY"], "agg": "AVG", "min_rows": 1, "cat": "production-cross"},
    {"q": "which provider has the most denied claims", "tables": ["claims", "providers"], "cols": ["NPI", "COUNT"], "agg": "COUNT", "min_rows": 1, "filter": "DENIED", "cat": "production-cross"},
    {"q": "prescriptions by medication class and plan type", "tables": ["prescriptions", "members"], "cols": ["MEDICATION_CLASS", "PLAN_TYPE", "COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "production-cross"},
    {"q": "which region has the highest average risk score", "tables": ["members"], "cols": ["KP_REGION", "RISK_SCORE"], "agg": "AVG", "min_rows": 1, "cat": "production-cross"},
    {"q": "claims where billed amount exceeds 10000", "tables": ["claims"], "cols": ["BILLED_AMOUNT"], "agg": None, "min_rows": 0, "filter": "BILLED_AMOUNT", "cat": "production-cross"},

    {"q": "high risk diabetic members by region", "tables": ["members"], "cols": ["KP_REGION", "COUNT"], "agg": "COUNT", "min_rows": 0, "filter": "RISK_SCORE", "cat": "production-domain"},
    {"q": "foster care members on medicaid by region", "tables": ["members"], "cols": ["KP_REGION", "COUNT"], "agg": "COUNT", "min_rows": 1, "filter": "Medicaid", "cat": "production-domain"},
    {"q": "members on antidepressants by region", "tables": ["members", "prescriptions"], "cols": ["KP_REGION", "COUNT"], "agg": "COUNT", "min_rows": 1, "filter": "Depression", "cat": "production-domain"},
    {"q": "average billed amount by specialty and region", "tables": ["claims", "providers"], "cols": ["SPECIALTY", "KP_REGION", "BILLED_AMOUNT"], "agg": "AVG", "min_rows": 1, "cat": "production-domain"},
    {"q": "denial rate by plan type", "tables": ["claims"], "cols": ["PLAN_TYPE", "DENIED"], "agg": None, "min_rows": 1, "cat": "production-domain"},

    {"q": "claims by region and gender", "tables": ["claims", "members"], "cols": ["KP_REGION", "GENDER", "COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "production-multidim"},
    {"q": "encounters by visit type and plan type", "tables": ["encounters", "members"], "cols": ["VISIT_TYPE", "PLAN_TYPE", "COUNT"], "agg": "COUNT", "min_rows": 1, "cat": "production-multidim"},
    {"q": "total billed amount per provider", "tables": ["claims", "providers"], "cols": ["NPI", "BILLED_AMOUNT"], "agg": "SUM", "min_rows": 1, "cat": "production-multidim"},
]


def run_tests():
    setup_logging()
    cfg = resolve_config({})
    pool = DatabasePool.get_instance()
    pool.initialize(cfg)
    engine = DynamicSQLEngine()

    results = []
    for i, test in enumerate(TESTS):
        q = test["q"]
        t0 = time.time()

        try:
            gen = engine.generate(q)
            sql = gen.get("sql", "")
            tables_used = gen.get("tables_used", [])
            agg_info = gen.get("agg_info", {})
        except Exception as e:
            results.append({
                "idx": i, "q": q, "cat": test["cat"],
                "status": "GEN_ERROR", "error": str(e),
                "sql": "", "rows": 0, "ms": 0,
            })
            continue

        exec_ok, rows, exec_err = False, [], ""
        if sql and not sql.startswith("--"):
            try:
                exec_ok, rows, exec_err = pool.execute_query(sql)
            except Exception as e:
                exec_err = str(e)

        elapsed = (time.time() - t0) * 1000
        sql_upper = sql.upper()

        errors = []

        for t in test.get("tables", []):
            if t.upper() not in sql_upper and t not in str(tables_used):
                errors.append(f"missing_table:{t}")

        for c in test.get("cols", []):
            if c.upper() not in sql_upper:
                errors.append(f"missing_col:{c}")

        expected_agg = test.get("agg")
        if expected_agg:
            agg_in_sql = (
                (expected_agg == "COUNT" and "COUNT(" in sql_upper) or
                (expected_agg == "SUM" and "SUM(" in sql_upper) or
                (expected_agg == "AVG" and "AVG(" in sql_upper) or
                (expected_agg == "MAX" and "MAX(" in sql_upper) or
                (expected_agg == "MIN" and "MIN(" in sql_upper)
            )
            if not agg_in_sql:
                errors.append(f"wrong_agg:expected={expected_agg},not_in_sql")

        min_rows = test.get("min_rows", 0)
        if min_rows > 0 and len(rows) < min_rows:
            if exec_err:
                errors.append(f"exec_error:{exec_err[:80]}")
            else:
                errors.append(f"too_few_rows:expected>={min_rows},got={len(rows)}")

        expected_filter = test.get("filter", "")
        if expected_filter and expected_filter.upper() not in sql_upper:
            errors.append(f"missing_filter:{expected_filter}")

        if sql and not exec_ok and not exec_err:
            errors.append("exec_failed_silently")

        status = "PASS" if not errors else "FAIL"

        results.append({
            "idx": i, "q": q, "cat": test["cat"],
            "status": status, "errors": errors,
            "sql": sql, "rows": len(rows), "ms": round(elapsed, 1),
            "agg": expected_agg if expected_agg else None, "tables": tables_used,
        })

    return results


def print_report(results):
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    gen_err = sum(1 for r in results if r["status"] == "GEN_ERROR")

    print("\n" + "=" * 80)
    print(f"COMPREHENSIVE SQL TEST REPORT — {total} tests")
    print("=" * 80)
    print(f"\n  PASSED: {passed}/{total} ({passed/total*100:.0f}%)")
    print(f"  FAILED: {failed}/{total}")
    print(f"  ERRORS: {gen_err}/{total}")

    cats = {}
    for r in results:
        cat = r["cat"]
        if cat not in cats:
            cats[cat] = {"pass": 0, "fail": 0, "total": 0}
        cats[cat]["total"] += 1
        if r["status"] == "PASS":
            cats[cat]["pass"] += 1
        else:
            cats[cat]["fail"] += 1

    print(f"\n{'Category':<30} {'Pass':>5} {'Fail':>5} {'Total':>6} {'Rate':>6}")
    print("-" * 55)
    for cat in sorted(cats.keys()):
        c = cats[cat]
        rate = c["pass"] / c["total"] * 100
        marker = " ✗" if c["fail"] > 0 else ""
        print(f"  {cat:<28} {c['pass']:>5} {c['fail']:>5} {c['total']:>6} {rate:>5.0f}%{marker}")

    failures = [r for r in results if r["status"] != "PASS"]
    if failures:
        print(f"\n{'=' * 80}")
        print(f"FAILURES ({len(failures)})")
        print("=" * 80)
        for r in failures:
            print(f"\n  [{r['cat']}] Q: {r['q']}")
            print(f"    SQL: {r['sql'][:120]}")
            errs = r.get("errors", [r.get("error", "unknown")])
            print(f"    Errors: {', '.join(str(e) for e in errs)}")
            print(f"    Rows: {r['rows']} | Agg: {r.get('agg')} | Tables: {r.get('tables')}")

    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results written to test_results.json")


if __name__ == "__main__":
    results = run_tests()
    print_report(results)
