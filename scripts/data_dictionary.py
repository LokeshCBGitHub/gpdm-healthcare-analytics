import sqlite3
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger('gpdm.data_dictionary')


COLUMN_DEFINITIONS: Dict[str, Dict[str, Any]] = {

    'claims.CLAIM_ID': {
        'description': 'Unique identifier assigned to each insurance claim submitted for reimbursement.',
        'business_purpose': 'Primary key for tracking every financial transaction between providers and payers. Essential for audit trails, duplicate detection, and appeals tracking.',
        'financial_impact': 'Each claim represents a revenue opportunity. Unresolved claims = unrealized revenue. Duplicate claims trigger compliance penalties.',
        'clinical_relevance': 'Links a specific clinical service to its financial outcome — bridges care delivery to reimbursement.',
        'stakeholder_relevance': {
            'CFO': 'Revenue cycle tracking — every dollar flows through a claim ID',
            'Operations': 'Workflow tracking from submission to adjudication',
            'Clinical': 'Connects patient encounters to billing outcomes',
            'Compliance': 'Audit trail for regulatory reporting',
            'Developer': 'Primary key, indexed, used in all claim-related JOINs'
        },
        'related_columns': ['claims.ENCOUNTER_ID', 'claims.MEMBER_ID', 'claims.RENDERING_NPI'],
        'kpis_affected': ['Total Claims Volume', 'Claims Processing Rate', 'Duplicate Claim Rate'],
        'data_quality_notes': 'Must be unique. Format: CLM + 8 digits. Nulls indicate data ingestion failures.'
    },
    'claims.MEMBER_ID': {
        'description': 'Unique identifier for the health plan member who received the service.',
        'business_purpose': 'Links claims to member enrollment, enabling per-member cost analysis (PMPM), utilization tracking, and risk stratification.',
        'financial_impact': 'High-cost members (top 5%) drive 50%+ of total spend. PMPM trending depends on accurate member-claim linkage.',
        'clinical_relevance': 'Tracks all services a patient receives across encounters — critical for care coordination and chronic disease management.',
        'stakeholder_relevance': {
            'CFO': 'PMPM calculations, risk-adjusted revenue forecasting',
            'Operations': 'Member utilization patterns, high-cost member identification',
            'Clinical': 'Longitudinal patient care history',
            'Compliance': 'Member eligibility verification',
            'Developer': 'Foreign key to members table, high-cardinality index'
        },
        'related_columns': ['members.MEMBER_ID', 'encounters.MEMBER_ID', 'diagnoses.MEMBER_ID', 'prescriptions.MEMBER_ID'],
        'kpis_affected': ['PMPM Cost', 'Member Utilization Rate', 'High-Cost Member %', 'Risk Score Distribution'],
        'data_quality_notes': 'Must match a valid member in the members table. Orphaned claims indicate enrollment gaps.'
    },
    'claims.ENCOUNTER_ID': {
        'description': 'Links the claim to the specific clinical encounter (visit) where the service was rendered.',
        'business_purpose': 'Enables encounter-level cost analysis and ensures proper grouping of charges per visit.',
        'financial_impact': 'Multiple claims per encounter is normal (bundled services), but excessive unbundling triggers fraud alerts.',
        'clinical_relevance': 'Groups all billed services for a single patient visit — essential for understanding total cost of care per episode.',
        'stakeholder_relevance': {
            'CFO': 'Episode-of-care costing, DRG-level revenue analysis',
            'Operations': 'Claims-to-encounter reconciliation',
            'Clinical': 'Correlates clinical decisions with their financial outcomes',
            'Developer': 'Foreign key to encounters table'
        },
        'related_columns': ['encounters.ENCOUNTER_ID', 'encounters.VISIT_TYPE', 'encounters.PRIMARY_DIAGNOSIS'],
        'kpis_affected': ['Cost Per Encounter', 'Claims Per Encounter Ratio', 'Episode Cost'],
        'data_quality_notes': 'Should match encounters table. Missing encounter IDs suggest standalone pharmacy or DME claims.'
    },
    'claims.SERVICE_DATE': {
        'description': 'The date when the healthcare service was actually provided to the patient.',
        'business_purpose': 'Anchors claims to the calendar for trend analysis, seasonal patterns, and timely filing compliance.',
        'financial_impact': 'Timely filing deadlines (typically 90-365 days) are measured from service date. Late submissions = denied revenue.',
        'clinical_relevance': 'Establishes the care timeline — when treatments occurred relative to diagnoses and outcomes.',
        'stakeholder_relevance': {
            'CFO': 'Revenue recognition timing, accrual accounting, trend forecasting',
            'Operations': 'Timely filing compliance, seasonal capacity planning',
            'Clinical': 'Treatment timeline and care sequence analysis',
            'Developer': 'Date column — indexed for range queries and trend aggregations'
        },
        'related_columns': ['claims.SUBMITTED_DATE', 'claims.ADJUDICATED_DATE', 'encounters.SERVICE_DATE'],
        'kpis_affected': ['Claims Lag (Service to Submit)', 'Monthly Revenue Trend', 'Seasonal Volume Patterns'],
        'data_quality_notes': 'Must be <= SUBMITTED_DATE. Future dates indicate data errors.'
    },
    'claims.RENDERING_NPI': {
        'description': 'National Provider Identifier of the clinician who directly provided the service.',
        'business_purpose': 'Identifies the individual provider responsible for the care — essential for provider profiling, productivity, and compensation.',
        'financial_impact': 'Provider-level cost variation can be 2-5x for identical procedures. High-cost outlier providers directly erode margins.',
        'clinical_relevance': 'Tracks which clinician delivered care — critical for quality scoring, outcomes analysis, and credential verification.',
        'stakeholder_relevance': {
            'CFO': 'Provider cost efficiency, compensation alignment with productivity',
            'Operations': 'Provider workload balancing, scheduling optimization',
            'Clinical': 'Quality scorecards, peer benchmarking, outcomes by provider',
            'Compliance': 'NPI validation, credentialing verification',
            'Developer': 'Foreign key to providers.NPI, 10-digit format'
        },
        'related_columns': ['providers.NPI', 'providers.SPECIALTY', 'claims.BILLING_NPI'],
        'kpis_affected': ['Provider Productivity', 'Cost Per Provider', 'Provider Denial Rate'],
        'data_quality_notes': 'Must be valid 10-digit NPI. Billing NPI may differ (group practice billing).'
    },
    'claims.BILLING_NPI': {
        'description': 'NPI of the entity (individual or organization) submitting the claim for payment.',
        'business_purpose': 'Identifies who is billing — may be a group practice, hospital, or individual. Critical for accounts receivable and contract management.',
        'financial_impact': 'Contract rates are often tied to billing NPI. Wrong billing NPI = wrong payment rate = revenue leakage.',
        'clinical_relevance': 'Less clinically relevant — primarily an administrative/financial identifier.',
        'stakeholder_relevance': {
            'CFO': 'Contract rate application, AR management by billing entity',
            'Operations': 'Billing entity reconciliation, group practice management',
            'Developer': 'May differ from RENDERING_NPI — common source of JOIN confusion'
        },
        'related_columns': ['claims.RENDERING_NPI', 'providers.NPI'],
        'kpis_affected': ['Revenue By Billing Entity', 'Contract Compliance Rate'],
        'data_quality_notes': 'When different from RENDERING_NPI, indicates group/facility billing arrangement.'
    },
    'claims.CPT_CODE': {
        'description': 'Current Procedural Terminology code identifying the specific medical procedure or service performed.',
        'business_purpose': 'The universal language of healthcare billing — determines what was done and at what rate it should be reimbursed.',
        'financial_impact': 'CPT codes directly map to reimbursement rates via fee schedules. Upcoding/downcoding creates compliance risk or revenue loss.',
        'clinical_relevance': 'Precisely identifies the clinical service — from office visits (99213) to complex surgeries. Essential for utilization review.',
        'stakeholder_relevance': {
            'CFO': 'Service mix analysis, fee schedule negotiations, revenue optimization',
            'Operations': 'Procedure volume tracking, OR/clinic utilization',
            'Clinical': 'Treatment pattern analysis, care pathway adherence',
            'Compliance': 'Coding accuracy audits, upcoding/downcoding detection',
            'Developer': 'Lookup key to cpt_codes reference table for descriptions and RVUs'
        },
        'related_columns': ['claims.CPT_DESCRIPTION', 'cpt_codes.CPT_CODE', 'cpt_codes.RVU', 'cpt_codes.CATEGORY'],
        'kpis_affected': ['Service Mix Index', 'Average RVU Per Visit', 'Procedure Volume Trends'],
        'data_quality_notes': '5-digit code. Must exist in cpt_codes reference table. Invalid codes cause claim rejections.'
    },
    'claims.CPT_DESCRIPTION': {
        'description': 'Human-readable description of the CPT code — what the medical procedure or service was.',
        'business_purpose': 'Provides plain-language context for billing codes, making reports accessible to non-coding stakeholders.',
        'financial_impact': 'Indirect — supports reporting clarity. Mismatched descriptions may indicate coding errors.',
        'clinical_relevance': 'Makes billing data clinically interpretable without requiring CPT code memorization.',
        'stakeholder_relevance': {
            'CFO': 'Report readability — understands what services drive revenue',
            'Clinical': 'Validates that coded services match clinical documentation',
            'Developer': 'Denormalized from cpt_codes table for query convenience'
        },
        'related_columns': ['claims.CPT_CODE', 'cpt_codes.DESCRIPTION'],
        'kpis_affected': ['Service Category Reporting'],
        'data_quality_notes': 'Should match cpt_codes.DESCRIPTION. Discrepancies may indicate stale data.'
    },
    'claims.ICD10_CODE': {
        'description': 'International Classification of Diseases, 10th Revision code representing the diagnosis or reason for the service.',
        'business_purpose': 'Drives medical necessity determination, HCC risk adjustment, and disease-specific analytics.',
        'financial_impact': 'ICD-10 codes determine HCC risk scores which directly affect Medicare Advantage capitation payments. Undercoding = lost revenue.',
        'clinical_relevance': 'The primary language of clinical diagnoses — tells us what condition the patient has and its severity.',
        'stakeholder_relevance': {
            'CFO': 'Risk adjustment revenue (RAF scores), disease burden quantification',
            'Operations': 'Population health stratification, care program targeting',
            'Clinical': 'Diagnosis tracking, comorbidity analysis, clinical decision support',
            'Compliance': 'Coding accuracy, HCC audit compliance',
            'Developer': 'Hierarchical code structure (letter + digits + decimal). Links to diagnoses table.'
        },
        'related_columns': ['claims.ICD10_DESCRIPTION', 'diagnoses.ICD10_CODE', 'diagnoses.HCC_CODE'],
        'kpis_affected': ['Risk Adjustment Factor', 'Disease Prevalence', 'HCC Capture Rate', 'Comorbidity Index'],
        'data_quality_notes': 'Format: Letter + 2 digits + optional decimal + digits. Must be valid ICD-10-CM code.'
    },
    'claims.ICD10_DESCRIPTION': {
        'description': 'Human-readable description of the ICD-10 diagnosis code.',
        'business_purpose': 'Makes diagnostic data accessible without requiring ICD-10 code memorization.',
        'financial_impact': 'Indirect — supports accurate reporting and clinical documentation improvement initiatives.',
        'clinical_relevance': 'Plain-language diagnosis name for clinical review and population health reporting.',
        'stakeholder_relevance': {
            'CFO': 'Readable disease-cost reports',
            'Clinical': 'Clinical documentation validation',
            'Developer': 'Denormalized field for query convenience'
        },
        'related_columns': ['claims.ICD10_CODE', 'diagnoses.ICD10_DESCRIPTION'],
        'kpis_affected': ['Disease Burden Reports'],
        'data_quality_notes': 'Should match standard ICD-10 terminology.'
    },
    'claims.BILLED_AMOUNT': {
        'description': 'The total amount the provider charged for the service — the "sticker price" before any payer adjustments.',
        'business_purpose': 'Represents gross revenue opportunity. The gap between billed and paid amounts reveals contract efficiency and payer mix impact.',
        'financial_impact': 'CRITICAL — Total billed is the starting point for all revenue analysis. Billed-to-paid ratio (collection rate) is a key financial health indicator. Industry benchmark: 30-45% collection rate for commercial, lower for government payers.',
        'clinical_relevance': 'Less clinically relevant — billing amounts are driven by chargemaster rates, not clinical outcomes.',
        'stakeholder_relevance': {
            'CFO': 'Gross revenue, chargemaster analysis, payer negotiation leverage',
            'Operations': 'Revenue cycle efficiency, charge capture completeness',
            'Clinical': 'Cost of treatment protocols (aggregate level)',
            'Developer': 'Stored as TEXT, must CAST to REAL for calculations. Can be NULL for voided claims.'
        },
        'related_columns': ['claims.ALLOWED_AMOUNT', 'claims.PAID_AMOUNT', 'claims.MEMBER_RESPONSIBILITY'],
        'kpis_affected': ['Gross Revenue', 'Collection Rate', 'Revenue Leakage', 'Billed-to-Paid Ratio', 'PMPM Billed'],
        'data_quality_notes': 'Must be > 0 for active claims. Amounts > $100K warrant review (potential data error or complex case).'
    },
    'claims.ALLOWED_AMOUNT': {
        'description': 'The maximum amount the insurance plan agrees to pay for the service — the contract-negotiated rate.',
        'business_purpose': 'Represents the effective revenue ceiling per service. The difference between billed and allowed is the contractual write-off.',
        'financial_impact': 'Allowed amount is the TRUE revenue target. Improving allowed amounts means better payer contracts. Contractual adjustments (billed - allowed) are the largest single revenue reduction.',
        'clinical_relevance': 'Indirect — may affect clinical decisions when cost-effectiveness is considered.',
        'stakeholder_relevance': {
            'CFO': 'Contract performance, payer mix optimization, fee schedule benchmarking',
            'Operations': 'Contract compliance monitoring, underpayment detection',
            'Developer': 'Stored as TEXT. Allowed should be <= Billed and >= Paid.'
        },
        'related_columns': ['claims.BILLED_AMOUNT', 'claims.PAID_AMOUNT', 'claims.PLAN_TYPE'],
        'kpis_affected': ['Net Revenue', 'Contractual Adjustment Rate', 'Payer Mix Impact'],
        'data_quality_notes': 'Should be <= BILLED_AMOUNT. If > BILLED, indicates contract override or data error.'
    },
    'claims.PAID_AMOUNT': {
        'description': 'The actual amount the insurance plan paid to the provider for the service.',
        'business_purpose': 'This is the REALIZED revenue — the money that actually flows in. The most important financial metric in healthcare.',
        'financial_impact': 'CRITICAL — Paid amount IS the revenue. Everything else (billed, allowed) is theoretical. Total paid minus total cost = operating margin.',
        'clinical_relevance': 'Indirect — reflects what the system actually invests in patient care.',
        'stakeholder_relevance': {
            'CFO': 'ACTUAL revenue, cash flow, operating margin, PMPM paid',
            'Operations': 'Payment velocity, underpayment recovery opportunities',
            'Clinical': 'Cost-effectiveness analysis (what did we spend on outcomes)',
            'Developer': 'Stored as TEXT. Only non-zero for PAID status claims.'
        },
        'related_columns': ['claims.BILLED_AMOUNT', 'claims.ALLOWED_AMOUNT', 'claims.CLAIM_STATUS'],
        'kpis_affected': ['Net Revenue', 'Operating Margin', 'PMPM Paid', 'Collection Rate', 'Days in AR'],
        'data_quality_notes': 'Zero for DENIED/PENDING/VOIDED claims. Should be <= ALLOWED_AMOUNT.'
    },
    'claims.MEMBER_RESPONSIBILITY': {
        'description': 'The portion of the bill the patient must pay out-of-pocket (copay + coinsurance + deductible).',
        'business_purpose': 'Tracks patient financial exposure and bad debt risk. High member responsibility correlates with payment delays and write-offs.',
        'financial_impact': 'Member responsibility collections are often the hardest dollars to collect. Bad debt from uncollected patient balances directly reduces margin.',
        'clinical_relevance': 'High out-of-pocket costs lead to care avoidance, medication non-adherence, and worse outcomes.',
        'stakeholder_relevance': {
            'CFO': 'Bad debt exposure, patient collections strategy, benefit design impact',
            'Operations': 'Patient billing, collections workflow, financial counseling needs',
            'Clinical': 'Social determinants — financial barriers to care adherence',
            'Developer': 'Sum of COPAY + COINSURANCE + DEDUCTIBLE'
        },
        'related_columns': ['claims.COPAY', 'claims.COINSURANCE', 'claims.DEDUCTIBLE'],
        'kpis_affected': ['Patient Bad Debt Rate', 'Member Out-of-Pocket Average', 'Collections Efficiency'],
        'data_quality_notes': 'Should equal COPAY + COINSURANCE + DEDUCTIBLE approximately.'
    },
    'claims.COPAY': {
        'description': 'Fixed dollar amount the member pays per visit or service, as defined by their benefit plan.',
        'business_purpose': 'Component of member cost-sharing. Varies by plan type and service category.',
        'financial_impact': 'Copays are typically collected at point of service. Uncollected copays = immediate revenue loss.',
        'clinical_relevance': 'Higher copays deter utilization — particularly for discretionary and preventive services.',
        'stakeholder_relevance': {
            'CFO': 'Point-of-service collection rates, benefit design analysis',
            'Clinical': 'Barrier to care access, particularly for low-income populations'
        },
        'related_columns': ['claims.MEMBER_RESPONSIBILITY', 'claims.PLAN_TYPE'],
        'kpis_affected': ['Copay Collection Rate', 'Member Cost Share'],
        'data_quality_notes': 'Typically $0-$75 for office visits, $100-$500 for ER. Zero for some Medicaid plans.'
    },
    'claims.COINSURANCE': {
        'description': 'Percentage-based amount the member pays after meeting their deductible.',
        'business_purpose': 'Represents the member\'s proportional cost-sharing obligation after the deductible is met.',
        'financial_impact': 'Coinsurance amounts can be substantial for expensive procedures. Major source of patient bad debt.',
        'clinical_relevance': 'High coinsurance deters use of expensive treatments and specialty care.',
        'stakeholder_relevance': {
            'CFO': 'Patient balance exposure, bad debt forecasting',
            'Operations': 'Payment plan administration',
            'Developer': 'Dollar amount (not percentage) — the actual calculated share'
        },
        'related_columns': ['claims.MEMBER_RESPONSIBILITY', 'claims.PLAN_TYPE'],
        'kpis_affected': ['Patient Balance Aging', 'Bad Debt Ratio'],
        'data_quality_notes': 'Usually 10-40% of allowed amount. Post-deductible.'
    },
    'claims.DEDUCTIBLE': {
        'description': 'Amount applied toward the member\'s annual deductible for this service.',
        'business_purpose': 'Tracks deductible accumulation per member per benefit year. Critical for benefit design analysis.',
        'financial_impact': 'Deductible period (typically Q1) shows highest member responsibility and lowest provider collection rates.',
        'clinical_relevance': 'High-deductible plans (HDHP) cause patients to delay care in early months of the year.',
        'stakeholder_relevance': {
            'CFO': 'Seasonal revenue patterns, HDHP impact on cash flow',
            'Operations': 'Q1 volume dips, deductible accumulation tracking',
            'Clinical': 'Deductible-driven care delays, especially for chronic disease management'
        },
        'related_columns': ['claims.MEMBER_RESPONSIBILITY', 'claims.PLAN_TYPE', 'claims.SERVICE_DATE'],
        'kpis_affected': ['Deductible Accumulation Rate', 'Q1 Volume Variance', 'HDHP Impact'],
        'data_quality_notes': 'Higher in Q1 (deductible reset). HDHP deductibles: $1,500-$7,000. HMO may be $0.'
    },
    'claims.CLAIM_STATUS': {
        'description': 'Current adjudication status of the claim: PAID, DENIED, PENDING, ADJUSTED, APPEALED, or VOIDED.',
        'business_purpose': 'THE most critical operational metric — shows where every dollar is in the revenue cycle pipeline.',
        'financial_impact': 'CRITICAL — Denial rate directly correlates with revenue loss. Industry benchmark: 5-10% initial denial rate. Anything above 10% signals systemic issues.',
        'clinical_relevance': 'Denials often indicate documentation gaps, medical necessity disputes, or authorization failures.',
        'stakeholder_relevance': {
            'CFO': 'Revenue realization rate, denial cost quantification, cash flow forecasting',
            'Operations': 'Revenue cycle efficiency, denial management workflow, staff productivity',
            'Clinical': 'Authorization compliance, documentation improvement needs',
            'Compliance': 'Denial pattern analysis, payer behavior monitoring',
            'Developer': 'ENUM-like field with 6 values. PAID = revenue realized. All others = revenue at risk.'
        },
        'related_columns': ['claims.DENIAL_REASON', 'claims.PAID_AMOUNT', 'claims.BILLED_AMOUNT'],
        'kpis_affected': ['Denial Rate', 'Clean Claim Rate', 'First-Pass Resolution Rate', 'Revenue Leakage', 'Days in AR'],
        'data_quality_notes': 'Valid values: PAID, DENIED, PENDING, ADJUSTED, APPEALED, VOIDED. No nulls allowed.'
    },
    'claims.DENIAL_REASON': {
        'description': 'Specific reason code or description for why a claim was denied.',
        'business_purpose': 'Root cause analysis for denials — identifies systemic issues that can be fixed to recover revenue.',
        'financial_impact': 'CRITICAL — Each denial reason maps to a corrective action. Fixing top 3 denial reasons typically recovers 60-70% of denied revenue.',
        'clinical_relevance': '"Not medically necessary" denials indicate clinical documentation gaps. "Pre-auth required" indicates care coordination failures.',
        'stakeholder_relevance': {
            'CFO': 'Revenue recovery prioritization — which denial reasons to attack first',
            'Operations': 'Process improvement targets, staff training needs',
            'Clinical': 'Documentation improvement, prior authorization compliance',
            'Compliance': 'Payer contract enforcement, appeal success rate by reason'
        },
        'related_columns': ['claims.CLAIM_STATUS', 'claims.BILLED_AMOUNT'],
        'kpis_affected': ['Denial Rate By Reason', 'Recoverable Revenue', 'Appeal Success Rate'],
        'data_quality_notes': 'Empty for non-denied claims. Common values: Coding error, Out of network, Pre-auth required, Duplicate, Timely filing, Not medically necessary.'
    },
    'claims.KP_REGION': {
        'description': 'The KP geographic region where the service was delivered.',
        'business_purpose': 'Enables regional performance benchmarking, resource allocation, and geographic trend analysis.',
        'financial_impact': 'Regional cost variation can be 20-40% due to local market dynamics, staffing costs, and payer mix differences.',
        'clinical_relevance': 'Regional health patterns (disease prevalence, social determinants) drive clinical program needs.',
        'stakeholder_relevance': {
            'CFO': 'Regional P&L, cost variation analysis, investment prioritization',
            'Operations': 'Regional staffing, facility planning, network adequacy',
            'Clinical': 'Population health variations by geography',
            'Developer': 'Categorical field, 8 distinct values. Used heavily in GROUP BY and filtering.'
        },
        'related_columns': ['members.KP_REGION', 'providers.KP_REGION', 'encounters.KP_REGION'],
        'kpis_affected': ['Regional Revenue', 'Regional Denial Rate', 'Regional Cost Per Member'],
        'data_quality_notes': 'Valid: NCAL, SCAL, NW, CO, MID, GA, HI, MAS. Must match a recognized KP region.'
    },
    'claims.FACILITY': {
        'description': 'The specific medical facility or clinic where the service was rendered.',
        'business_purpose': 'Facility-level cost center analysis, capacity utilization, and investment planning.',
        'financial_impact': 'Facility operating costs are the largest fixed expense. Underutilized facilities destroy margins.',
        'clinical_relevance': 'Facility capabilities determine what services can be delivered locally vs. referred out.',
        'stakeholder_relevance': {
            'CFO': 'Facility-level P&L, CapEx planning, lease vs. own decisions',
            'Operations': 'Capacity management, facility utilization rates',
            'Clinical': 'Service availability, patient access',
            'Developer': 'Text field, 26 distinct facilities. Can join to multiple tables.'
        },
        'related_columns': ['encounters.FACILITY', 'providers.FACILITY', 'appointments.FACILITY'],
        'kpis_affected': ['Facility Utilization', 'Revenue Per Facility', 'Patient Volume By Facility'],
        'data_quality_notes': 'Should be a recognized KP facility name. NULL indicates telehealth or external service.'
    },
    'claims.PLAN_TYPE': {
        'description': 'The type of insurance plan covering the member: HMO, PPO, EPO, HDHP, Medicare Advantage, or Medicaid.',
        'business_purpose': 'Plan type determines reimbursement rates, utilization management requirements, and member cost-sharing structures.',
        'financial_impact': 'CRITICAL — Plan type drives revenue per service. Medicare Advantage = capitated (fixed PMPM). Commercial = fee-for-service (per claim). Medicaid = lowest reimbursement. Plan mix directly determines margin.',
        'clinical_relevance': 'Plan type affects care access: HMO requires referrals, PPO allows direct specialist access, HDHP creates cost barriers.',
        'stakeholder_relevance': {
            'CFO': 'Payer mix optimization, contract negotiation strategy, revenue forecasting by plan',
            'Operations': 'Authorization requirements, network restrictions, benefit verification',
            'Clinical': 'Access barriers, referral requirements, formulary restrictions',
            'Developer': 'Categorical with 6 values. Key GROUP BY dimension for financial analysis.'
        },
        'related_columns': ['members.PLAN_TYPE', 'claims.BILLED_AMOUNT', 'claims.PAID_AMOUNT'],
        'kpis_affected': ['Payer Mix', 'Revenue By Plan Type', 'PMPM By Plan', 'Denial Rate By Plan'],
        'data_quality_notes': 'Valid: HMO, PPO, EPO, HDHP, Medicare Advantage, Medicaid.'
    },
    'claims.CLAIM_TYPE': {
        'description': 'Classification of the claim: INSTITUTIONAL (facility-based) or PROFESSIONAL (provider-based).',
        'business_purpose': 'Determines billing rules, reimbursement methodology, and regulatory requirements.',
        'financial_impact': 'Institutional claims (UB-04 forms) have different fee schedules and DRG-based payments vs. professional claims (CMS-1500). Mix affects revenue modeling.',
        'clinical_relevance': 'Institutional claims indicate facility-based care (hospital, SNF). Professional claims indicate outpatient/office-based care.',
        'stakeholder_relevance': {
            'CFO': 'Revenue methodology (DRG vs fee schedule), cost allocation',
            'Operations': 'Billing workflow routing, claim form requirements',
            'Developer': 'Binary categorical. Affects how reimbursement should be calculated.'
        },
        'related_columns': ['claims.FACILITY', 'claims.CPT_CODE', 'encounters.VISIT_TYPE'],
        'kpis_affected': ['Institutional vs Professional Mix', 'Average Payment By Claim Type'],
        'data_quality_notes': 'INSTITUTIONAL for facility claims, PROFESSIONAL for individual provider claims.'
    },
    'claims.SUBMITTED_DATE': {
        'description': 'The date the claim was submitted to the insurance payer for processing.',
        'business_purpose': 'Measures claim submission efficiency and timely filing compliance.',
        'financial_impact': 'Submission lag (service date to submitted date) indicates revenue cycle efficiency. Longer lag = higher denial risk from timely filing limits.',
        'clinical_relevance': 'Indirect — but delayed claim submission may indicate documentation completion delays.',
        'stakeholder_relevance': {
            'CFO': 'Days to bill, revenue cycle speed, cash flow timing',
            'Operations': 'Billing team productivity, submission backlog monitoring',
            'Developer': 'Date field. SUBMITTED_DATE >= SERVICE_DATE always.'
        },
        'related_columns': ['claims.SERVICE_DATE', 'claims.ADJUDICATED_DATE'],
        'kpis_affected': ['Days to Bill', 'Timely Filing Compliance', 'Submission Backlog'],
        'data_quality_notes': 'Must be >= SERVICE_DATE. Gap > 30 days is a billing efficiency concern.'
    },
    'claims.ADJUDICATED_DATE': {
        'description': 'The date the payer made a payment decision (approved, denied, adjusted) on the claim.',
        'business_purpose': 'Measures payer processing speed and total revenue cycle time.',
        'financial_impact': 'Adjudication speed determines cash flow timing. Slow payers increase days in AR and working capital needs.',
        'clinical_relevance': 'Indirect — but slow adjudication may delay authorization for ongoing treatments.',
        'stakeholder_relevance': {
            'CFO': 'Days in AR, cash flow forecasting, payer performance monitoring',
            'Operations': 'AR follow-up prioritization, payer escalation triggers',
            'Developer': 'Date field. ADJUDICATED_DATE >= SUBMITTED_DATE. May be NULL for PENDING claims.'
        },
        'related_columns': ['claims.SUBMITTED_DATE', 'claims.CLAIM_STATUS'],
        'kpis_affected': ['Days in AR', 'Payer Turnaround Time', 'Clean Claim Rate'],
        'data_quality_notes': 'NULL for PENDING claims. Gap from submit to adjudicate = payer processing time.'
    },

    'members.MRN': {
        'description': 'Medical Record Number — the internal patient identifier used within the health system.',
        'business_purpose': 'Links the member to their clinical record. Distinct from MEMBER_ID which is the insurance identifier.',
        'financial_impact': 'Indirect — but MRN-MEMBER_ID matching errors cause billing failures and duplicate records.',
        'clinical_relevance': 'PRIMARY clinical identifier — used in EHR systems, lab orders, and clinical documentation.',
        'stakeholder_relevance': {
            'CFO': 'Duplicate record costs (estimated $1,000+ per duplicate in rework)',
            'Clinical': 'Patient identification, clinical record access',
            'Developer': 'Unique within health system. Links to encounters.MRN.'
        },
        'related_columns': ['members.MEMBER_ID', 'encounters.MRN'],
        'kpis_affected': ['Duplicate Record Rate', 'Patient Identification Accuracy'],
        'data_quality_notes': 'Format: MRN + digits. Must be unique. One MRN per patient.'
    },
    'members.MEMBER_ID': {
        'description': 'Insurance plan member identifier — the payer-assigned ID for enrollment and claims tracking.',
        'business_purpose': 'THE primary key for all financial and enrollment analysis. Links claims, encounters, prescriptions, and referrals.',
        'financial_impact': 'Every financial metric (PMPM, risk scores, utilization) is anchored to MEMBER_ID. Accurate member identification = accurate financials.',
        'clinical_relevance': 'Links to all clinical encounters and diagnoses for longitudinal patient history.',
        'stakeholder_relevance': {
            'CFO': 'PMPM calculations, risk-adjusted revenue, member-level P&L',
            'Operations': 'Member eligibility, enrollment management',
            'Clinical': 'Patient identification across encounters',
            'Developer': 'Primary key. Foreign key reference in claims, encounters, diagnoses, prescriptions, appointments, referrals.'
        },
        'related_columns': ['claims.MEMBER_ID', 'encounters.MEMBER_ID', 'diagnoses.MEMBER_ID', 'prescriptions.MEMBER_ID'],
        'kpis_affected': ['Total Enrolled Members', 'PMPM', 'Member Retention Rate'],
        'data_quality_notes': 'Format: MBR + digits. Must be unique.'
    },
    'members.DATE_OF_BIRTH': {
        'description': 'Member\'s date of birth — used to calculate age for demographic and clinical stratification.',
        'business_purpose': 'Age is the single strongest predictor of healthcare utilization and cost. Essential for actuarial analysis and benefit design.',
        'financial_impact': 'Members 65+ cost 3-5x more than members under 40. Age distribution drives total medical expense and premium adequacy.',
        'clinical_relevance': 'Age-appropriate care guidelines (pediatric vs adult vs geriatric), screening schedules, and risk models all depend on accurate age.',
        'stakeholder_relevance': {
            'CFO': 'Actuarial aging, cost projection, Medicare Advantage risk adjustment',
            'Operations': 'Age-specific program design, staffing models',
            'Clinical': 'Age-appropriate screening (mammography, colonoscopy), pediatric care, geriatric care',
            'Developer': 'DATE type. Calculate age dynamically. PHI field.'
        },
        'related_columns': ['members.GENDER', 'members.RISK_SCORE', 'members.PLAN_TYPE'],
        'kpis_affected': ['Age Distribution', 'Cost By Age Band', 'Pediatric/Adult/Geriatric Mix'],
        'data_quality_notes': 'PHI. Must be valid date. Used to derive age bands for reporting.'
    },
    'members.GENDER': {
        'description': 'Member\'s recorded gender for demographic analysis and clinical care guidelines.',
        'business_purpose': 'Gender-specific utilization patterns (OB/GYN, prostate screening) affect benefit costs and program design.',
        'financial_impact': 'Gender-based actuarial analysis, maternity cost projections, gender-specific utilization rates.',
        'clinical_relevance': 'Gender-specific clinical guidelines (mammography, PSA, prenatal care), medication dosing considerations.',
        'stakeholder_relevance': {
            'CFO': 'Actuarial analysis, maternity cost reserve',
            'Clinical': 'Gender-specific care protocols and screening programs'
        },
        'related_columns': ['members.DATE_OF_BIRTH', 'members.PLAN_TYPE'],
        'kpis_affected': ['Gender Distribution', 'Maternity Utilization Rate'],
        'data_quality_notes': 'Values: M, F. Should not be NULL.'
    },
    'members.RACE': {
        'description': 'Member\'s self-reported race/ethnicity for health equity analysis.',
        'business_purpose': 'Required for CMS health equity reporting and disparity analysis. Essential for community health needs assessments.',
        'financial_impact': 'Health equity programs may qualify for CMS incentive payments. Disparities in access/outcomes represent addressable cost.',
        'clinical_relevance': 'CRITICAL — Race/ethnicity correlates with disease prevalence (diabetes in Hispanic populations, hypertension in Black populations), genetic risks, and social determinants.',
        'stakeholder_relevance': {
            'CFO': 'Health equity incentive programs, disparity-related avoidable costs',
            'Clinical': 'Culturally competent care, disparity reduction programs',
            'Compliance': 'CMS health equity reporting requirements'
        },
        'related_columns': ['members.LANGUAGE', 'members.KP_REGION'],
        'kpis_affected': ['Health Equity Index', 'Disparity Metrics', 'CMS Star Ratings (Equity)'],
        'data_quality_notes': 'PHI-adjacent. Values include White, Black, Hispanic, Asian, Native Hawaiian, etc.'
    },
    'members.LANGUAGE': {
        'description': 'Member\'s preferred language for communication.',
        'business_purpose': 'Required for care access compliance, interpreter service planning, and materials translation.',
        'financial_impact': 'Language barriers increase ER visits (miscommunication), reduce preventive care uptake, and increase readmissions.',
        'clinical_relevance': 'Language concordance between provider and patient improves outcomes, medication adherence, and patient satisfaction.',
        'stakeholder_relevance': {
            'CFO': 'Interpreter service costs, avoidable ER visits from language barriers',
            'Operations': 'Interpreter scheduling, multilingual staff recruitment',
            'Clinical': 'Patient safety, informed consent, medication counseling'
        },
        'related_columns': ['members.RACE', 'members.KP_REGION'],
        'kpis_affected': ['Language Access Compliance', 'Interpreter Utilization'],
        'data_quality_notes': 'Primary values: English, Spanish, Chinese, Korean, etc.'
    },
    'members.KP_REGION': {
        'description': 'The KP region where the member is enrolled and receives primary care.',
        'business_purpose': 'Drives regional enrollment reporting, capacity planning, and market share analysis.',
        'financial_impact': 'Regional enrollment size determines fixed cost allocation and economies of scale.',
        'clinical_relevance': 'Determines local care access, facility availability, and provider network.',
        'stakeholder_relevance': {
            'CFO': 'Regional enrollment revenue, market penetration, growth targets',
            'Operations': 'Regional capacity, network adequacy',
            'Clinical': 'Regional population health priorities'
        },
        'related_columns': ['claims.KP_REGION', 'providers.KP_REGION', 'encounters.KP_REGION'],
        'kpis_affected': ['Regional Enrollment', 'Market Share', 'Regional PMPM'],
        'data_quality_notes': '8 distinct regions: NCAL, SCAL, NW, CO, MID, GA, HI, MAS.'
    },
    'members.PLAN_TYPE': {
        'description': 'Insurance plan type the member is enrolled in.',
        'business_purpose': 'Plan type determines benefit structure, provider network rules, and revenue methodology (capitated vs. fee-for-service).',
        'financial_impact': 'Medicare Advantage members generate capitated PMPM revenue. Commercial members generate fee-for-service revenue. Mix drives financial model.',
        'clinical_relevance': 'Plan type affects care access patterns and referral requirements.',
        'stakeholder_relevance': {
            'CFO': 'Revenue mix, actuarial modeling, premium adequacy',
            'Operations': 'Benefit verification, network management',
            'Clinical': 'Formulary differences, authorization requirements'
        },
        'related_columns': ['claims.PLAN_TYPE', 'members.RISK_SCORE'],
        'kpis_affected': ['Plan Mix', 'Revenue By Plan', 'Member Retention By Plan'],
        'data_quality_notes': 'HMO, PPO, EPO, HDHP, Medicare Advantage, Medicaid.'
    },
    'members.ENROLLMENT_DATE': {
        'description': 'Date the member first enrolled in the health plan.',
        'business_purpose': 'Member tenure drives retention analysis, lifetime value calculations, and enrollment growth trends.',
        'financial_impact': 'New members cost more (onboarding, initial assessments). Long-tenure members are more profitable and have predictable utilization.',
        'clinical_relevance': 'New enrollees need health risk assessments (HRA), care gap identification, and PCP assignment.',
        'stakeholder_relevance': {
            'CFO': 'Member lifetime value, retention economics, growth metrics',
            'Operations': 'Onboarding workflows, new member outreach',
            'Clinical': 'New member health assessments, care gap closure'
        },
        'related_columns': ['members.DISENROLLMENT_DATE', 'members.PLAN_TYPE'],
        'kpis_affected': ['Enrollment Growth', 'Member Tenure', 'Retention Rate', 'New Member %'],
        'data_quality_notes': 'Must be valid date. Members without DISENROLLMENT_DATE are currently active.'
    },
    'members.DISENROLLMENT_DATE': {
        'description': 'Date the member left the health plan (empty if currently enrolled).',
        'business_purpose': 'Tracks member attrition — why members leave and when. Critical for retention strategies.',
        'financial_impact': 'Every disenrolled member = lost PMPM revenue. Acquisition cost of a new member is 5-10x the retention cost.',
        'clinical_relevance': 'Disenrolled members lose care continuity, increasing risk of adverse outcomes in gaps between coverage.',
        'stakeholder_relevance': {
            'CFO': 'Revenue attrition, retention ROI, churn forecasting',
            'Operations': 'Disenrollment reasons, retention program effectiveness',
            'Clinical': 'Care continuity gaps, transition of care protocols'
        },
        'related_columns': ['members.ENROLLMENT_DATE'],
        'kpis_affected': ['Disenrollment Rate', 'Churn Rate', 'Net Enrollment Change'],
        'data_quality_notes': 'Empty string means currently enrolled. Date present means disenrolled.'
    },
    'members.PCP_NPI': {
        'description': 'NPI of the member\'s assigned Primary Care Provider.',
        'business_purpose': 'PCP assignment is foundational for managed care. Empanelment determines care coordination responsibility and provider workload.',
        'financial_impact': 'Members with active PCP relationships have 15-20% lower total cost of care vs. unassigned/ER-dependent members.',
        'clinical_relevance': 'CRITICAL — PCP is the quarterback of patient care. Coordinates referrals, manages chronic conditions, provides preventive care.',
        'stakeholder_relevance': {
            'CFO': 'PCP-assigned members cost less — empanelment rate drives PMPM savings',
            'Operations': 'Panel size management, PCP assignment workflow',
            'Clinical': 'Care coordination, continuity of care, preventive care compliance',
            'Developer': 'Foreign key to providers.NPI. May be NULL for unassigned members.'
        },
        'related_columns': ['providers.NPI', 'providers.PANEL_SIZE', 'providers.SPECIALTY'],
        'kpis_affected': ['PCP Empanelment Rate', 'Panel Size', 'PCP Utilization Rate'],
        'data_quality_notes': 'Should reference a valid NPI. NULL indicates unassigned member.'
    },
    'members.RISK_SCORE': {
        'description': 'Hierarchical Condition Category (HCC) risk score — predicts expected healthcare spending relative to the average member.',
        'business_purpose': 'THE core metric for population health management and Medicare Advantage revenue. Score of 1.0 = average expected cost. 2.0 = 2x average.',
        'financial_impact': 'CRITICAL — For Medicare Advantage, risk scores directly determine capitation payments from CMS. Underdocumented risk scores = lost revenue (estimated $1,000-$3,000 per missed HCC per year).',
        'clinical_relevance': 'High risk scores indicate complex patients needing care management, disease management programs, and proactive outreach.',
        'stakeholder_relevance': {
            'CFO': 'Risk-adjusted revenue (RAF), PMPM adequacy, HCC capture ROI',
            'Operations': 'Care management program enrollment, risk stratification',
            'Clinical': 'Complex care identification, disease management, care gaps',
            'Compliance': 'Risk adjustment data validation (RADV) audits',
            'Developer': 'Numeric (stored as TEXT). Range 0.1 to 3.5+. Higher = sicker/costlier.'
        },
        'related_columns': ['members.CHRONIC_CONDITIONS', 'diagnoses.HCC_CODE', 'claims.ICD10_CODE'],
        'kpis_affected': ['Average Risk Score', 'HCC Capture Rate', 'Risk-Adjusted Revenue', 'Risk Stratification Distribution'],
        'data_quality_notes': 'Low <0.5, Moderate 0.5-1.0, High 1.0-1.5, Very High 1.5+.'
    },
    'members.CHRONIC_CONDITIONS': {
        'description': 'Count of active chronic conditions the member has (0-6+).',
        'business_purpose': 'Primary indicator of medical complexity. Members with 3+ chronic conditions account for 80%+ of healthcare spending.',
        'financial_impact': 'Each additional chronic condition increases PMPM cost by $200-$500/month. Multi-morbid patients are the key cost driver.',
        'clinical_relevance': 'CRITICAL — Multi-morbid patients need coordinated care management. Drug interactions, competing treatment priorities, and care fragmentation risks all increase with condition count.',
        'stakeholder_relevance': {
            'CFO': 'Cost stratification, disease management program ROI, predictive modeling',
            'Operations': 'Care management caseload planning',
            'Clinical': 'Care coordination complexity, polypharmacy risk, care gap prioritization',
            'Developer': 'Integer 0-6+. Correlates with RISK_SCORE.'
        },
        'related_columns': ['members.RISK_SCORE', 'diagnoses.IS_CHRONIC', 'prescriptions.MEDICATION_CLASS'],
        'kpis_affected': ['Chronic Condition Prevalence', 'Multi-Morbidity Rate', 'PMPM By Condition Count'],
        'data_quality_notes': 'Should correlate with actual diagnoses marked IS_CHRONIC=Y in diagnoses table.'
    },

    'encounters.ENCOUNTER_ID': {
        'description': 'Unique identifier for each patient visit/encounter.',
        'business_purpose': 'Core linkage between clinical care delivery and financial outcomes.',
        'financial_impact': 'Each encounter represents a revenue-generating event. Encounter volume x average revenue per encounter = facility revenue.',
        'clinical_relevance': 'A single unit of care delivery — everything from a 15-minute office visit to a multi-day inpatient stay.',
        'stakeholder_relevance': {
            'CFO': 'Volume-driven revenue model, encounter rate negotiation',
            'Operations': 'Scheduling, capacity utilization, throughput',
            'Clinical': 'Care episode tracking, documentation completeness',
            'Developer': 'Primary key. Links to claims.ENCOUNTER_ID.'
        },
        'related_columns': ['claims.ENCOUNTER_ID', 'encounters.MEMBER_ID', 'encounters.VISIT_TYPE'],
        'kpis_affected': ['Encounter Volume', 'Revenue Per Encounter', 'Utilization Rate'],
        'data_quality_notes': 'Format: ENC + digits. Must be unique.'
    },
    'encounters.VISIT_TYPE': {
        'description': 'Type of clinical visit: OUTPATIENT, INPATIENT, EMERGENCY, OBSERVATION, TELEHEALTH.',
        'business_purpose': 'Visit type determines reimbursement methodology, staffing requirements, and facility costs.',
        'financial_impact': 'Inpatient stays are the highest-cost encounters ($5K-$50K+). ER visits are expensive ($1K-$5K) and often avoidable. Telehealth is lowest-cost ($50-$200). Shifting visit mix toward lower-cost settings improves margins.',
        'clinical_relevance': 'Visit type indicates acuity level and care setting appropriateness.',
        'stakeholder_relevance': {
            'CFO': 'Revenue by care setting, ER avoidance savings, telehealth ROI',
            'Operations': 'Capacity by care setting, staffing models',
            'Clinical': 'Appropriate care setting, ER diversion, admission criteria'
        },
        'related_columns': ['encounters.LENGTH_OF_STAY', 'claims.BILLED_AMOUNT', 'claims.CLAIM_TYPE'],
        'kpis_affected': ['Visit Mix', 'ER Utilization Rate', 'Telehealth Adoption', 'Inpatient Days'],
        'data_quality_notes': 'OUTPATIENT, INPATIENT, EMERGENCY, OBSERVATION, TELEHEALTH.'
    },
    'encounters.PRIMARY_DIAGNOSIS': {
        'description': 'The primary ICD-10 diagnosis code for this encounter — the main reason for the visit.',
        'business_purpose': 'Drives DRG assignment for inpatient claims and case-mix index calculations.',
        'financial_impact': 'Primary diagnosis determines DRG which determines inpatient reimbursement. Wrong primary diagnosis = wrong DRG = wrong payment.',
        'clinical_relevance': 'THE reason the patient sought care. Drives treatment plan, care pathway, and follow-up requirements.',
        'stakeholder_relevance': {
            'CFO': 'Case mix index, DRG optimization, disease-specific cost analysis',
            'Clinical': 'Clinical pathway adherence, treatment protocols',
            'Developer': 'ICD-10 format. Links to diagnoses.ICD10_CODE.'
        },
        'related_columns': ['encounters.DIAGNOSIS_DESCRIPTION', 'diagnoses.ICD10_CODE', 'claims.ICD10_CODE'],
        'kpis_affected': ['Case Mix Index', 'Top Diagnoses', 'Disease-Specific Cost'],
        'data_quality_notes': 'Should be a valid ICD-10-CM code.'
    },
    'encounters.CHIEF_COMPLAINT': {
        'description': 'The patient\'s self-reported reason for the visit in plain language.',
        'business_purpose': 'Supports triage analysis and demand pattern identification.',
        'financial_impact': 'Indirect — but complaint-to-diagnosis alignment helps identify undertriage/overtriage patterns.',
        'clinical_relevance': 'The patient\'s voice — what brought them in. Often differs from the final coded diagnosis. Critical for quality of care analysis.',
        'stakeholder_relevance': {
            'CFO': 'Demand pattern analysis for service line planning',
            'Clinical': 'Triage accuracy, patient experience, symptom tracking',
            'Developer': 'Free-text field. Requires NLP for analysis.'
        },
        'related_columns': ['encounters.PRIMARY_DIAGNOSIS', 'encounters.VISIT_TYPE'],
        'kpis_affected': ['Top Chief Complaints', 'Complaint-to-Diagnosis Concordance'],
        'data_quality_notes': 'Free text. Common values: Chest pain, Fever, Follow-up, Back pain, etc.'
    },
    'encounters.DISPOSITION': {
        'description': 'The outcome/discharge status of the encounter: Discharged, Admitted, Observation, Transfer, etc.',
        'business_purpose': 'Tracks patient flow and care escalation patterns.',
        'financial_impact': 'ER-to-admission conversion rate affects inpatient revenue and bed utilization. Observation vs. admission has major billing implications.',
        'clinical_relevance': 'Clinical outcome of the encounter — did the patient go home, get admitted, or need transfer to higher-level care?',
        'stakeholder_relevance': {
            'CFO': 'ER conversion rate, observation-to-admission ratio (billing impact)',
            'Operations': 'Bed management, patient flow, discharge planning',
            'Clinical': 'Clinical decision-making patterns, escalation appropriateness'
        },
        'related_columns': ['encounters.VISIT_TYPE', 'encounters.LENGTH_OF_STAY'],
        'kpis_affected': ['ER Conversion Rate', 'Observation Rate', 'Discharge Disposition Mix'],
        'data_quality_notes': 'Values: Discharged, Admitted, Observation, Transfer, AMA, Expired.'
    },
    'encounters.LENGTH_OF_STAY': {
        'description': 'Number of days the patient spent in the facility (0 for outpatient/same-day visits).',
        'business_purpose': 'ALOS (Average Length of Stay) is a core operational metric. Longer stays = higher cost and lower bed availability.',
        'financial_impact': 'Each inpatient day costs $2,000-$5,000+. Reducing ALOS by 0.5 days across the system saves millions annually.',
        'clinical_relevance': 'LOS reflects recovery time, complication rates, and discharge readiness.',
        'stakeholder_relevance': {
            'CFO': 'Inpatient cost per day, ALOS benchmarking, bed day costs',
            'Operations': 'Bed management, discharge planning, throughput optimization',
            'Clinical': 'Recovery metrics, complication rates, readiness for discharge'
        },
        'related_columns': ['encounters.VISIT_TYPE', 'encounters.PRIMARY_DIAGNOSIS', 'encounters.DISPOSITION'],
        'kpis_affected': ['ALOS', 'Bed Days', 'Cost Per Inpatient Day', 'Excess Days'],
        'data_quality_notes': '0 for outpatient. Typically 1-14 for inpatient. >30 days flags long-stay review.'
    },

    'providers.NPI': {
        'description': 'National Provider Identifier — the 10-digit unique identifier assigned by CMS to every healthcare provider.',
        'business_purpose': 'Universal provider identification for credentialing, billing, and regulatory compliance.',
        'financial_impact': 'Invalid NPIs cause claim rejections. Provider credentialing status affects billing eligibility.',
        'clinical_relevance': 'Identifies the specific clinician for quality reporting and outcomes tracking.',
        'stakeholder_relevance': {
            'CFO': 'Provider cost profiling, compensation analysis',
            'Operations': 'Credentialing management, network adequacy',
            'Clinical': 'Quality scorecards, peer benchmarking',
            'Compliance': 'CMS NPI validation requirements',
            'Developer': 'Primary key. 10-digit number. Links to claims.RENDERING_NPI and claims.BILLING_NPI.'
        },
        'related_columns': ['claims.RENDERING_NPI', 'claims.BILLING_NPI', 'members.PCP_NPI'],
        'kpis_affected': ['Active Provider Count', 'Provider Productivity'],
        'data_quality_notes': '10-digit format. Must be valid with NPPES registry.'
    },
    'providers.SPECIALTY': {
        'description': 'Medical specialty the provider practices in.',
        'business_purpose': 'Drives specialty-specific analysis: referral patterns, cost per specialty, staffing needs by specialty.',
        'financial_impact': 'Specialty cost variation is massive: Oncology can be 10x more expensive per patient than Primary Care. Specialty mix drives total medical expense.',
        'clinical_relevance': 'Determines what conditions the provider treats and what procedures they can perform.',
        'stakeholder_relevance': {
            'CFO': 'Specialty-level P&L, referral cost analysis, FTE planning',
            'Operations': 'Specialist access, wait times, referral management',
            'Clinical': 'Specialty availability, subspecialty coverage gaps'
        },
        'related_columns': ['claims.CPT_CODE', 'referrals.SPECIALTY', 'encounters.DEPARTMENT'],
        'kpis_affected': ['Cost By Specialty', 'Referral Rate By Specialty', 'Provider FTE By Specialty'],
        'data_quality_notes': '20 specialties in dataset. Should match credentialing records.'
    },
    'providers.PANEL_SIZE': {
        'description': 'Number of patients currently assigned to this provider\'s care panel.',
        'business_purpose': 'Panel size management is critical for access and quality. Overloaded panels lead to long wait times and provider burnout.',
        'financial_impact': 'Optimal panel size maximizes revenue per provider while maintaining quality. Overpaneled providers generate more referrals (higher cost).',
        'clinical_relevance': 'Panel size directly affects appointment availability, visit duration, and preventive care completion rates.',
        'stakeholder_relevance': {
            'CFO': 'Provider productivity, revenue per FTE, optimal panel economics',
            'Operations': 'Access management, new patient capacity, workload balancing',
            'Clinical': 'Quality of care, burnout risk, preventive care compliance'
        },
        'related_columns': ['providers.ACCEPTS_NEW_PATIENTS', 'providers.STATUS'],
        'kpis_affected': ['Average Panel Size', 'Panel Capacity %', 'Access Wait Times'],
        'data_quality_notes': 'Range 101-2500. Optimal PCP panel: 1,200-1,800. Specialists: varies widely.'
    },
    'providers.STATUS': {
        'description': 'Provider employment status: ACTIVE, INACTIVE, or ON_LEAVE.',
        'business_purpose': 'Workforce management — tracks who is available to see patients.',
        'financial_impact': 'Inactive/on-leave providers reduce capacity and may require locum tenens at premium cost.',
        'clinical_relevance': 'Inactive providers cannot see patients — their panels need coverage.',
        'stakeholder_relevance': {
            'CFO': 'FTE costs, locum expense, workforce planning budget',
            'Operations': 'Staffing levels, coverage planning, recruitment needs',
            'Clinical': 'Panel reassignment, continuity of care'
        },
        'related_columns': ['providers.PANEL_SIZE', 'providers.ACCEPTS_NEW_PATIENTS'],
        'kpis_affected': ['Active Provider Count', 'Provider Turnover Rate', 'Coverage Ratio'],
        'data_quality_notes': 'ACTIVE (85.4%), INACTIVE (9.8%), ON_LEAVE (4.8%).'
    },

    'diagnoses.ICD10_CODE': {
        'description': 'ICD-10-CM diagnosis code from the clinical record.',
        'business_purpose': 'Drives HCC risk coding, disease prevalence analysis, and clinical program targeting.',
        'financial_impact': 'Accurate diagnosis coding drives risk adjustment revenue. Each HCC has a specific dollar value ($800-$8,000+/year in MA capitation).',
        'clinical_relevance': 'THE clinical record of what conditions a patient has. Foundation for treatment planning.',
        'stakeholder_relevance': {
            'CFO': 'HCC revenue capture, risk score optimization',
            'Clinical': 'Disease management, comorbidity tracking',
            'Compliance': 'Coding accuracy, RADV audit compliance'
        },
        'related_columns': ['diagnoses.HCC_CODE', 'diagnoses.HCC_CATEGORY', 'claims.ICD10_CODE'],
        'kpis_affected': ['HCC Capture Rate', 'Disease Prevalence', 'Risk Score Accuracy'],
        'data_quality_notes': 'ICD-10-CM format. Must map to valid HCC when applicable.'
    },
    'diagnoses.HCC_CODE': {
        'description': 'Hierarchical Condition Category code mapped from the ICD-10 diagnosis.',
        'business_purpose': 'HCC codes drive Medicare Advantage risk adjustment — the mechanism by which sicker members generate higher capitation payments.',
        'financial_impact': 'CRITICAL — Each HCC has a coefficient in the CMS-HCC model that translates to additional PMPM revenue. Missing HCC captures means leaving money on the table.',
        'clinical_relevance': 'HCC categories group clinically related diagnoses for risk stratification.',
        'stakeholder_relevance': {
            'CFO': 'Risk adjustment revenue — each HCC = $800-$8,000+ annual revenue per member',
            'Operations': 'HCC capture campaigns, provider education',
            'Clinical': 'Risk-stratified care planning',
            'Compliance': 'RADV (Risk Adjustment Data Validation) audit readiness'
        },
        'related_columns': ['diagnoses.ICD10_CODE', 'diagnoses.HCC_CATEGORY', 'members.RISK_SCORE'],
        'kpis_affected': ['HCC Capture Rate', 'Risk Adjustment Factor', 'Revenue Per HCC'],
        'data_quality_notes': 'Format: HCC + digits. Not all ICD-10 codes map to HCCs.'
    },
    'diagnoses.SEVERITY': {
        'description': 'Clinical severity rating of the diagnosis: MILD, MODERATE, SEVERE, CRITICAL.',
        'business_purpose': 'Severity stratification enables targeted intervention — severe cases need aggressive management, mild cases need monitoring.',
        'financial_impact': 'Severe cases cost 5-10x more than mild. Preventing escalation from mild to severe = massive cost avoidance.',
        'clinical_relevance': 'CRITICAL — Drives treatment intensity, care management enrollment, and urgency of follow-up.',
        'stakeholder_relevance': {
            'CFO': 'Cost stratification by severity, early intervention ROI',
            'Operations': 'Care management staffing by acuity level',
            'Clinical': 'Treatment intensity, care escalation protocols'
        },
        'related_columns': ['diagnoses.ICD10_CODE', 'diagnoses.IS_CHRONIC', 'members.RISK_SCORE'],
        'kpis_affected': ['Severity Distribution', 'Cost By Severity', 'Severity Escalation Rate'],
        'data_quality_notes': 'MILD, MODERATE, SEVERE, CRITICAL. Distribution should skew toward MILD/MODERATE.'
    },
    'diagnoses.IS_CHRONIC': {
        'description': 'Flag (Y/N) indicating whether this is a chronic (ongoing) condition.',
        'business_purpose': 'Chronic conditions drive 80%+ of healthcare spending. Identifying and managing chronic disease is the #1 cost management strategy.',
        'financial_impact': 'A single chronic condition adds $2,000-$6,000/year to PMPM cost. Multi-chronic patients (3+) cost $15,000-$50,000+/year.',
        'clinical_relevance': 'Chronic conditions need ongoing management — medication adherence, lifestyle modification, regular monitoring.',
        'stakeholder_relevance': {
            'CFO': 'Chronic disease cost burden, disease management ROI',
            'Clinical': 'Chronic care management programs, care gap closure',
            'Operations': 'Care management caseloads'
        },
        'related_columns': ['diagnoses.ICD10_CODE', 'members.CHRONIC_CONDITIONS', 'prescriptions.MEDICATION_CLASS'],
        'kpis_affected': ['Chronic Disease Prevalence', 'Chronic-to-Total Cost Ratio'],
        'data_quality_notes': 'Y or N. Should correlate with ICD-10 code clinical classification.'
    },

    'prescriptions.MEDICATION_NAME': {
        'description': 'Name and dosage of the prescribed medication.',
        'business_purpose': 'Tracks prescribing patterns, formulary compliance, and drug utilization.',
        'financial_impact': 'Pharmacy costs are 15-25% of total medical expense. Generic substitution, formulary compliance, and therapeutic alternatives drive savings.',
        'clinical_relevance': 'Medication management — right drug, right dose, right patient. Polypharmacy risk increases with each additional medication.',
        'stakeholder_relevance': {
            'CFO': 'Pharmacy spend, generic vs brand utilization, formulary compliance savings',
            'Clinical': 'Medication safety, drug interactions, adherence monitoring',
            'Operations': 'Pharmacy benefit management, prior authorization'
        },
        'related_columns': ['prescriptions.MEDICATION_CLASS', 'prescriptions.COST', 'prescriptions.NDC_CODE'],
        'kpis_affected': ['Top Medications By Cost', 'Generic Utilization Rate', 'Formulary Compliance'],
        'data_quality_notes': 'Includes drug name + dosage (e.g., "Atorvastatin 20mg").'
    },
    'prescriptions.MEDICATION_CLASS': {
        'description': 'Therapeutic class of the medication (e.g., Cholesterol, Blood Pressure, Diabetes).',
        'business_purpose': 'Enables drug class-level analysis for formulary management and disease-specific pharmacy cost tracking.',
        'financial_impact': 'Class-level analysis identifies opportunities for therapeutic interchange (same outcomes, lower cost alternative).',
        'clinical_relevance': 'Links medications to the conditions they treat — essential for adherence monitoring and care gap detection.',
        'stakeholder_relevance': {
            'CFO': 'Drug class spend trends, formulary optimization ROI',
            'Clinical': 'Therapeutic class utilization, treatment guideline compliance'
        },
        'related_columns': ['prescriptions.MEDICATION_NAME', 'diagnoses.ICD10_CODE'],
        'kpis_affected': ['Pharmacy Cost By Class', 'Therapeutic Interchange Rate'],
        'data_quality_notes': 'Categories: Cholesterol, Blood Pressure, Diabetes, Pain, Antibiotics, etc.'
    },
    'prescriptions.COST': {
        'description': 'Total cost of the prescription (plan + member cost).',
        'business_purpose': 'Pharmacy cost tracking per prescription, medication, and member.',
        'financial_impact': 'Drug costs are the fastest-growing healthcare expense category. High-cost specialty drugs can be $10,000+/month.',
        'clinical_relevance': 'Cost affects adherence — expensive medications have higher non-compliance rates.',
        'stakeholder_relevance': {
            'CFO': 'Total pharmacy spend, PMPM pharmacy cost, specialty drug impact',
            'Clinical': 'Cost-effective prescribing, patient financial burden'
        },
        'related_columns': ['prescriptions.COPAY', 'prescriptions.MEDICATION_CLASS'],
        'kpis_affected': ['Pharmacy PMPM', 'Cost Per Prescription', 'Specialty Drug Spend %'],
        'data_quality_notes': 'Numeric. Typical range $5-$500 for generics, $100-$10,000+ for specialty.'
    },
    'prescriptions.STATUS': {
        'description': 'Current status of the prescription: FILLED, ACTIVE, CANCELLED, EXPIRED.',
        'business_purpose': 'Tracks prescription fulfillment and identifies adherence gaps.',
        'financial_impact': 'Unfilled prescriptions indicate non-adherence. Non-adherent members have 2-3x higher hospitalization rates.',
        'clinical_relevance': 'CRITICAL — Medication non-adherence is the #1 modifiable factor in preventing hospitalizations and disease progression.',
        'stakeholder_relevance': {
            'CFO': 'Non-adherence costs ($100-$300B annually in US healthcare)',
            'Clinical': 'Adherence monitoring, patient outreach triggers',
            'Operations': 'Pharmacy fulfillment tracking'
        },
        'related_columns': ['prescriptions.REFILLS_AUTHORIZED', 'prescriptions.REFILLS_USED', 'prescriptions.FILL_DATE'],
        'kpis_affected': ['Prescription Fill Rate', 'Medication Adherence (PDC)', 'Unfilled Rx Rate'],
        'data_quality_notes': 'FILLED, ACTIVE, CANCELLED, EXPIRED.'
    },

    'appointments.STATUS': {
        'description': 'Appointment status: COMPLETED, CANCELLED, NO_SHOW, SCHEDULED, RESCHEDULED.',
        'business_purpose': 'Tracks scheduling efficiency, patient compliance, and capacity utilization.',
        'financial_impact': 'No-shows cost $150-$300 per occurrence in lost revenue and wasted capacity. A 15% no-show rate can cost millions annually.',
        'clinical_relevance': 'No-shows and cancellations indicate barriers to care access — transportation, cost, health literacy, or disengagement.',
        'stakeholder_relevance': {
            'CFO': 'Lost revenue from no-shows, scheduling efficiency',
            'Operations': 'Overbooking strategy, reminder system effectiveness',
            'Clinical': 'Patient engagement, barriers to access'
        },
        'related_columns': ['appointments.APPOINTMENT_TYPE', 'appointments.DEPARTMENT'],
        'kpis_affected': ['No-Show Rate', 'Cancellation Rate', 'Scheduling Utilization'],
        'data_quality_notes': 'COMPLETED, CANCELLED, NO_SHOW, SCHEDULED, RESCHEDULED.'
    },
    'appointments.APPOINTMENT_TYPE': {
        'description': 'Type of appointment: OFFICE_VISIT, TELEHEALTH, PROCEDURE, FOLLOW_UP, NEW_PATIENT, URGENT.',
        'business_purpose': 'Appointment mix analysis for capacity planning and resource allocation.',
        'financial_impact': 'New patient visits and procedures generate higher revenue than follow-ups. Telehealth has lower overhead.',
        'clinical_relevance': 'Visit type determines clinical preparation, duration, and staffing needs.',
        'stakeholder_relevance': {
            'CFO': 'Revenue per appointment type, telehealth adoption savings',
            'Operations': 'Scheduling templates, room/equipment allocation',
            'Clinical': 'Visit preparation, clinical workflow'
        },
        'related_columns': ['appointments.DURATION_MINUTES', 'appointments.DEPARTMENT'],
        'kpis_affected': ['Appointment Mix', 'New Patient Rate', 'Telehealth %'],
        'data_quality_notes': 'OFFICE_VISIT, TELEHEALTH, PROCEDURE, FOLLOW_UP, NEW_PATIENT, URGENT.'
    },

    'referrals.STATUS': {
        'description': 'Referral status: COMPLETED, PENDING, CANCELLED, EXPIRED.',
        'business_purpose': 'Tracks referral completion — are patients actually following through on specialist referrals?',
        'financial_impact': 'Completed referrals generate specialist revenue. Incomplete referrals indicate care leakage (patients going out of network).',
        'clinical_relevance': 'Referral completion is essential for care continuity. Patients who don\'t follow through on referrals may have unmanaged conditions.',
        'stakeholder_relevance': {
            'CFO': 'In-network referral revenue retention, leakage prevention',
            'Operations': 'Referral management workflow, specialist scheduling',
            'Clinical': 'Care coordination, closed-loop referral tracking'
        },
        'related_columns': ['referrals.REFERRAL_TYPE', 'referrals.URGENCY', 'referrals.SPECIALTY'],
        'kpis_affected': ['Referral Completion Rate', 'Care Leakage Rate', 'Referral-to-Appointment Lag'],
        'data_quality_notes': 'COMPLETED, PENDING, CANCELLED, EXPIRED.'
    },
    'referrals.URGENCY': {
        'description': 'Clinical urgency of the referral: ROUTINE, URGENT, EMERGENT.',
        'business_purpose': 'Prioritization for scheduling and access management.',
        'financial_impact': 'Urgent/emergent referrals that aren\'t seen promptly may result in ER visits (higher cost) or adverse outcomes (liability).',
        'clinical_relevance': 'CRITICAL — Urgent referrals need to be seen within days, not weeks. Delayed urgent referrals = patient safety risk.',
        'stakeholder_relevance': {
            'CFO': 'ER avoidance through timely urgent referrals',
            'Operations': 'Priority scheduling, access standards compliance',
            'Clinical': 'Patient safety, timely specialist consultation'
        },
        'related_columns': ['referrals.STATUS', 'referrals.APPOINTMENT_DATE'],
        'kpis_affected': ['Urgent Referral Wait Time', 'Referral Timeliness'],
        'data_quality_notes': 'ROUTINE, URGENT, EMERGENT.'
    },

    'cpt_codes.RVU': {
        'description': 'Relative Value Unit — CMS-assigned measure of resources required for a procedure, including physician work, practice expense, and malpractice.',
        'business_purpose': 'RVUs are the universal currency of physician productivity. Compensation models, fee schedules, and staffing are all built on RVUs.',
        'financial_impact': 'CRITICAL — Revenue = RVU × Conversion Factor ($33-$37 for Medicare). Higher RVU procedures generate more revenue per time unit.',
        'clinical_relevance': 'Reflects procedure complexity. Higher RVU = more complex clinical work.',
        'stakeholder_relevance': {
            'CFO': 'Provider productivity (wRVUs), fee schedule modeling, compensation design',
            'Operations': 'Workload measurement, staffing adequacy',
            'Clinical': 'Procedure complexity, appropriate care delivery',
            'Developer': 'Numeric. Multiply by conversion factor for dollar value.'
        },
        'related_columns': ['cpt_codes.CPT_CODE', 'cpt_codes.CATEGORY', 'claims.CPT_CODE'],
        'kpis_affected': ['Total wRVUs', 'wRVU Per Provider', 'Revenue Per wRVU'],
        'data_quality_notes': 'Range typically 0.5-50+. E&M visits: 1-5 RVU. Major surgery: 20-50+ RVU.'
    },
}


TABLE_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    'claims': {
        'description': 'Insurance claims submitted for reimbursement — the complete financial record of healthcare services delivered.',
        'business_purpose': 'Revenue cycle management, payer analysis, denial management, and financial forecasting.',
        'row_count': 60000,
        'primary_key': 'CLAIM_ID',
        'foreign_keys': {
            'MEMBER_ID': 'members.MEMBER_ID',
            'ENCOUNTER_ID': 'encounters.ENCOUNTER_ID',
            'RENDERING_NPI': 'providers.NPI',
            'CPT_CODE': 'cpt_codes.CPT_CODE'
        },
        'financial_relevance': 'CRITICAL — Contains ALL financial transaction data. Every dollar billed, allowed, paid, and denied lives here.',
        'key_metrics': ['Total Billed', 'Total Paid', 'Denial Rate', 'Collection Rate', 'PMPM', 'Revenue Leakage'],
        'stakeholder_focus': {
            'CFO': 'Revenue analysis, denial trending, payer mix, cash flow',
            'Operations': 'Claims processing efficiency, denial management workflow',
            'Clinical': 'Cost of care delivery, treatment cost-effectiveness'
        }
    },
    'members': {
        'description': 'Health plan enrollment roster — every person covered by the insurance plan with their demographics, risk profile, and PCP assignment.',
        'business_purpose': 'Population health management, actuarial analysis, member engagement, and risk stratification.',
        'row_count': 25000,
        'primary_key': 'MEMBER_ID',
        'financial_relevance': 'Foundation for PMPM calculations, risk-adjusted revenue, and enrollment-based metrics.',
        'key_metrics': ['Total Enrollment', 'Risk Score Distribution', 'Chronic Condition Prevalence', 'PCP Assignment Rate', 'Retention Rate'],
        'stakeholder_focus': {
            'CFO': 'Enrollment revenue, risk-adjusted capitation, member lifetime value',
            'Operations': 'Member services, enrollment management, PCP assignment',
            'Clinical': 'Population health, risk stratification, care management targeting'
        }
    },
    'encounters': {
        'description': 'Clinical visits and patient encounters — where care delivery happens, from office visits to hospital stays.',
        'business_purpose': 'Utilization management, capacity planning, clinical quality measurement.',
        'row_count': 50000,
        'primary_key': 'ENCOUNTER_ID',
        'foreign_keys': {
            'MEMBER_ID': 'members.MEMBER_ID',
            'RENDERING_NPI': 'providers.NPI'
        },
        'financial_relevance': 'Each encounter is a revenue-generating event. Volume × average revenue = total clinical revenue.',
        'key_metrics': ['Encounter Volume', 'ALOS', 'Visit Type Mix', 'ER Utilization', 'Readmission Rate'],
        'stakeholder_focus': {
            'CFO': 'Revenue per encounter, volume trends, capacity costs',
            'Operations': 'Scheduling, bed management, throughput',
            'Clinical': 'Care quality, clinical outcomes, documentation completeness'
        }
    },
    'providers': {
        'description': 'Healthcare provider workforce — physicians, nurses, PAs, and other clinicians in the care delivery network.',
        'business_purpose': 'Workforce management, provider productivity, network adequacy, and credentialing.',
        'row_count': 3000,
        'primary_key': 'NPI',
        'financial_relevance': 'Provider costs (salary + benefits) are the second-largest expense. Productivity (wRVUs) determines provider-level ROI.',
        'key_metrics': ['Active Provider Count', 'Avg Panel Size', 'Provider Productivity', 'Specialty Distribution'],
        'stakeholder_focus': {
            'CFO': 'Provider FTE costs, compensation ROI, productivity benchmarking',
            'Operations': 'Staffing, scheduling, credentialing, recruitment',
            'Clinical': 'Quality scorecards, peer benchmarking, clinical leadership'
        }
    },
    'diagnoses': {
        'description': 'Clinical diagnoses recorded for patients — ICD-10 codes, severity, chronicity, and HCC risk coding.',
        'business_purpose': 'Risk adjustment revenue optimization, disease prevalence tracking, and clinical program targeting.',
        'row_count': 20000,
        'primary_key': 'DIAGNOSIS_ID',
        'foreign_keys': {
            'MEMBER_ID': 'members.MEMBER_ID',
            'ENCOUNTER_ID': 'encounters.ENCOUNTER_ID',
            'DIAGNOSING_NPI': 'providers.NPI'
        },
        'financial_relevance': 'HCC codes from diagnoses directly drive Medicare Advantage risk adjustment revenue. Each missed HCC = $800-$8,000/year lost.',
        'key_metrics': ['HCC Capture Rate', 'Disease Prevalence', 'Severity Distribution', 'Chronic vs Acute Mix'],
        'stakeholder_focus': {
            'CFO': 'Risk adjustment revenue, HCC gap closure ROI',
            'Clinical': 'Disease management, comorbidity tracking, treatment planning'
        }
    },
    'prescriptions': {
        'description': 'Medication prescriptions — drugs, dosages, refills, costs, and fulfillment status.',
        'business_purpose': 'Pharmacy benefit management, medication adherence tracking, and drug cost control.',
        'row_count': 12000,
        'primary_key': 'RX_ID',
        'foreign_keys': {
            'MEMBER_ID': 'members.MEMBER_ID',
            'PRESCRIBING_NPI': 'providers.NPI'
        },
        'financial_relevance': 'Pharmacy costs are 15-25% of total medical expense and growing fastest. Specialty drugs are the biggest driver.',
        'key_metrics': ['Total Pharmacy Spend', 'PMPM Pharmacy Cost', 'Generic Utilization Rate', 'Adherence Rate'],
        'stakeholder_focus': {
            'CFO': 'Pharmacy PMPM, specialty drug spend, formulary savings',
            'Clinical': 'Medication safety, adherence, polypharmacy risk'
        }
    },
    'appointments': {
        'description': 'Patient scheduling and appointment records — bookings, completions, cancellations, and no-shows.',
        'business_purpose': 'Access management, scheduling optimization, and patient engagement tracking.',
        'row_count': 10000,
        'primary_key': 'APPOINTMENT_ID',
        'foreign_keys': {
            'MEMBER_ID': 'members.MEMBER_ID',
            'PROVIDER_NPI': 'providers.NPI'
        },
        'financial_relevance': 'No-shows cost $150-$300 each in lost revenue. Scheduling efficiency directly affects provider productivity.',
        'key_metrics': ['No-Show Rate', 'Scheduling Utilization', 'Wait Time', 'Appointment Mix'],
        'stakeholder_focus': {
            'CFO': 'Lost revenue from no-shows, provider utilization',
            'Operations': 'Scheduling optimization, access management',
            'Clinical': 'Patient access, engagement, follow-up compliance'
        }
    },
    'referrals': {
        'description': 'Specialist referrals — tracking care coordination from PCP to specialist and back.',
        'business_purpose': 'Care coordination, network management, and referral leakage prevention.',
        'row_count': 5000,
        'primary_key': 'REFERRAL_ID',
        'foreign_keys': {
            'MEMBER_ID': 'members.MEMBER_ID',
            'REFERRING_NPI': 'providers.NPI',
            'REFERRED_TO_NPI': 'providers.NPI'
        },
        'financial_relevance': 'In-network referral completion retains revenue. Out-of-network leakage costs 20-40% premium.',
        'key_metrics': ['Referral Completion Rate', 'Referral Leakage Rate', 'Wait Time', 'Urgency Mix'],
        'stakeholder_focus': {
            'CFO': 'Referral revenue retention, leakage costs',
            'Operations': 'Referral management, specialist scheduling',
            'Clinical': 'Care coordination, closed-loop referral tracking'
        }
    },
    'cpt_codes': {
        'description': 'CPT procedure code reference table — maps procedure codes to descriptions, categories, and RVUs.',
        'business_purpose': 'Reference data for billing, productivity measurement, and fee schedule management.',
        'row_count': 81,
        'primary_key': 'CPT_CODE',
        'financial_relevance': 'RVU values are the basis for Medicare fee schedule and most commercial contract rates.',
        'key_metrics': ['Service Mix', 'Average RVU', 'Category Distribution'],
        'stakeholder_focus': {
            'CFO': 'Fee schedule management, RVU-based compensation',
            'Developer': 'Lookup/reference table for claims.CPT_CODE'
        }
    }
}


KPI_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    'PMPM': {
        'name': 'Per Member Per Month',
        'formula': 'Total Paid Amount / (Total Member Months)',
        'description': 'The average monthly cost per enrolled member — THE core metric for health plan financial management.',
        'benchmark': '$350-$500 for commercial, $800-$1,200 for Medicare Advantage',
        'direction': 'lower_is_better',
        'category': 'Financial'
    },
    'MLR': {
        'name': 'Medical Loss Ratio',
        'formula': 'Total Medical Costs / Total Premium Revenue × 100',
        'description': 'Percentage of premium revenue spent on medical care. ACA requires minimum 80% (individual/small group) or 85% (large group).',
        'benchmark': '80-88% is healthy. Below 80% = premium rebate required. Above 90% = financial stress.',
        'direction': 'target_range',
        'target': '82-87%',
        'category': 'Financial'
    },
    'Denial Rate': {
        'name': 'Claims Denial Rate',
        'formula': 'Denied Claims / Total Claims × 100',
        'description': 'Percentage of submitted claims that are denied by payers.',
        'benchmark': '5-10% is industry average. Above 10% = systemic billing/coding issues.',
        'direction': 'lower_is_better',
        'category': 'Revenue Cycle'
    },
    'Collection Rate': {
        'name': 'Net Collection Rate',
        'formula': 'Total Paid / Total Allowed × 100',
        'description': 'Percentage of allowed charges that are actually collected.',
        'benchmark': '95-98% is excellent. Below 90% = significant underpayment/AR issues.',
        'direction': 'higher_is_better',
        'category': 'Revenue Cycle'
    },
    'ALOS': {
        'name': 'Average Length of Stay',
        'formula': 'Total Inpatient Days / Total Inpatient Discharges',
        'description': 'Average number of days patients spend in inpatient care.',
        'benchmark': '4.5-5.5 days for general acute care. Varies significantly by condition.',
        'direction': 'lower_is_better',
        'category': 'Operations'
    },
    'Readmission Rate': {
        'name': '30-Day Readmission Rate',
        'formula': 'Readmissions within 30 days / Total Discharges × 100',
        'description': 'Percentage of patients readmitted within 30 days of discharge.',
        'benchmark': '12-15% national average. CMS penalizes hospitals above expected rate.',
        'direction': 'lower_is_better',
        'category': 'Quality'
    },
    'No-Show Rate': {
        'name': 'Appointment No-Show Rate',
        'formula': 'No-Show Appointments / Total Scheduled × 100',
        'description': 'Percentage of scheduled appointments where the patient did not show up.',
        'benchmark': '5-10% is acceptable. Above 15% = significant access/engagement problem.',
        'direction': 'lower_is_better',
        'category': 'Access'
    },
    'ER Utilization': {
        'name': 'ER Visit Rate per 1000',
        'formula': 'ER Visits / (Total Members / 1000)',
        'description': 'Number of ER visits per 1,000 enrolled members.',
        'benchmark': '300-400 per 1,000 members/year. Higher = access/primary care gaps.',
        'direction': 'lower_is_better',
        'category': 'Utilization'
    },
    'HCC Capture Rate': {
        'name': 'HCC Capture Rate',
        'formula': 'Documented HCCs / Expected HCCs × 100',
        'description': 'Percentage of expected HCC diagnoses that are actually documented and coded.',
        'benchmark': '85-95% is strong. Each missed HCC = $800-$8,000/year in lost MA revenue.',
        'direction': 'higher_is_better',
        'category': 'Risk Adjustment'
    },
    'PCP Empanelment': {
        'name': 'PCP Empanelment Rate',
        'formula': 'Members with Assigned PCP / Total Members × 100',
        'description': 'Percentage of members with an assigned primary care provider.',
        'benchmark': '95%+ is target. Below 90% = care coordination gaps.',
        'direction': 'higher_is_better',
        'category': 'Access'
    },
}


class DataDictionary:

    def __init__(self, db_path: str = None):
        self.db_path = db_path
        self.column_defs = COLUMN_DEFINITIONS
        self.table_defs = TABLE_DEFINITIONS
        self.kpi_defs = KPI_DEFINITIONS

    def explain_column(self, table: str, column: str, role: str = 'general') -> Dict[str, Any]:
        key = f"{table.lower()}.{column.upper()}"
        definition = self.column_defs.get(key)

        if not definition:
            for k, v in self.column_defs.items():
                if column.upper() in k.upper():
                    definition = v
                    key = k
                    break

        if not definition:
            return {
                'column': column,
                'table': table,
                'found': False,
                'message': f"Column '{column}' in table '{table}' not found in data dictionary. "
                           f"Available tables: {', '.join(self.table_defs.keys())}"
            }

        result = {
            'column': column,
            'table': table,
            'found': True,
            'description': definition['description'],
            'business_purpose': definition['business_purpose'],
            'financial_impact': definition['financial_impact'],
            'clinical_relevance': definition['clinical_relevance'],
            'related_columns': definition.get('related_columns', []),
            'kpis_affected': definition.get('kpis_affected', []),
            'data_quality_notes': definition.get('data_quality_notes', ''),
        }

        stakeholder = definition.get('stakeholder_relevance', {})
        if role != 'general' and role in stakeholder:
            result['role_specific_insight'] = stakeholder[role]
        result['stakeholder_relevance'] = stakeholder

        return result

    def explain_table(self, table: str) -> Dict[str, Any]:
        table_lower = table.lower()
        definition = self.table_defs.get(table_lower)

        if not definition:
            return {'table': table, 'found': False, 'available_tables': list(self.table_defs.keys())}

        columns = {}
        for key, col_def in self.column_defs.items():
            if key.startswith(f"{table_lower}."):
                col_name = key.split('.')[1]
                columns[col_name] = {
                    'description': col_def['description'],
                    'financial_impact': col_def['financial_impact'][:100] + '...' if len(col_def.get('financial_impact', '')) > 100 else col_def.get('financial_impact', ''),
                    'kpis': col_def.get('kpis_affected', [])
                }

        return {
            'table': table,
            'found': True,
            'description': definition['description'],
            'business_purpose': definition['business_purpose'],
            'row_count': definition.get('row_count', 0),
            'primary_key': definition.get('primary_key', ''),
            'foreign_keys': definition.get('foreign_keys', {}),
            'financial_relevance': definition.get('financial_relevance', ''),
            'key_metrics': definition.get('key_metrics', []),
            'stakeholder_focus': definition.get('stakeholder_focus', {}),
            'columns': columns
        }

    def explain_kpi(self, kpi_name: str) -> Dict[str, Any]:
        kpi = self.kpi_defs.get(kpi_name)
        if not kpi:
            for k, v in self.kpi_defs.items():
                if kpi_name.lower() in k.lower() or kpi_name.lower() in v.get('name', '').lower():
                    kpi = v
                    kpi_name = k
                    break

        if not kpi:
            return {'kpi': kpi_name, 'found': False, 'available_kpis': list(self.kpi_defs.keys())}

        return {
            'kpi': kpi_name,
            'found': True,
            'name': kpi['name'],
            'formula': kpi['formula'],
            'description': kpi['description'],
            'benchmark': kpi['benchmark'],
            'direction': kpi['direction'],
            'category': kpi['category']
        }

    def get_cross_table_relationships(self, column: str) -> Dict[str, Any]:
        column_upper = column.upper()
        relationships = []

        for key, col_def in self.column_defs.items():
            if column_upper in key.upper():
                related = col_def.get('related_columns', [])
                relationships.append({
                    'source': key,
                    'related_to': related,
                    'financial_impact': col_def.get('financial_impact', ''),
                    'kpis': col_def.get('kpis_affected', [])
                })

        return {
            'column': column,
            'found': len(relationships) > 0,
            'relationships': relationships,
            'message': f"Column '{column}' appears in {len(relationships)} table(s) with cross-table relationships." if relationships else f"Column '{column}' not found."
        }

    def get_financial_impact_summary(self) -> Dict[str, Any]:
        critical = []
        high = []
        moderate = []

        for key, col_def in self.column_defs.items():
            impact = col_def.get('financial_impact', '')
            entry = {
                'column': key,
                'impact': impact[:150],
                'kpis': col_def.get('kpis_affected', [])
            }
            if 'CRITICAL' in impact:
                critical.append(entry)
            elif any(word in impact.lower() for word in ['directly', 'revenue', 'margin', 'cost']):
                high.append(entry)
            else:
                moderate.append(entry)

        return {
            'critical_financial_columns': critical,
            'high_impact_columns': high,
            'moderate_impact_columns': moderate,
            'total_columns_analyzed': len(self.column_defs)
        }

    def format_explanation_for_chat(self, explanation: Dict[str, Any], role: str = 'general') -> str:
        if not explanation.get('found'):
            return explanation.get('message', 'Information not found in data dictionary.')

        parts = []

        if 'description' in explanation:
            parts.append(explanation['description'])

        if 'business_purpose' in explanation:
            parts.append(f"\nBusiness Purpose: {explanation['business_purpose']}")

        if 'financial_impact' in explanation:
            parts.append(f"\nFinancial Impact: {explanation['financial_impact']}")

        if 'clinical_relevance' in explanation:
            parts.append(f"\nClinical Relevance: {explanation['clinical_relevance']}")

        if role != 'general' and 'role_specific_insight' in explanation:
            parts.append(f"\nFor {role}: {explanation['role_specific_insight']}")

        if explanation.get('related_columns'):
            parts.append(f"\nRelated Fields: {', '.join(explanation['related_columns'])}")

        if explanation.get('kpis_affected'):
            parts.append(f"\nKPIs Affected: {', '.join(explanation['kpis_affected'])}")

        if explanation.get('data_quality_notes'):
            parts.append(f"\nData Quality: {explanation['data_quality_notes']}")

        return '\n'.join(parts)

    def compute_live_kpis(self, db_path: str = None) -> Dict[str, Any]:
        path = db_path or self.db_path
        if not path:
            return {'error': 'No database path configured'}

        try:
            conn = sqlite3.connect(path)
            c = conn.cursor()
            kpis = {}

            c.execute('SELECT COUNT(*), SUM(CAST(BILLED_AMOUNT AS REAL)), SUM(CAST(PAID_AMOUNT AS REAL)), SUM(CAST(ALLOWED_AMOUNT AS REAL)) FROM claims')
            r = c.fetchone()
            total_claims, total_billed, total_paid, total_allowed = r
            kpis['total_claims'] = total_claims
            kpis['total_billed'] = round(total_billed, 2)
            kpis['total_paid'] = round(total_paid, 2)
            kpis['total_allowed'] = round(total_allowed, 2)
            kpis['revenue_leakage'] = round(total_billed - total_paid, 2)
            kpis['revenue_leakage_pct'] = round((total_billed - total_paid) / total_billed * 100, 1) if total_billed else 0
            kpis['collection_rate'] = round(total_paid / total_allowed * 100, 1) if total_allowed else 0

            c.execute("SELECT COUNT(*) FROM claims WHERE CLAIM_STATUS = 'DENIED'")
            denied = c.fetchone()[0]
            kpis['denial_rate'] = round(denied / total_claims * 100, 1) if total_claims else 0
            kpis['denied_claims'] = denied
            kpis['denied_amount'] = 0
            c.execute("SELECT SUM(CAST(BILLED_AMOUNT AS REAL)) FROM claims WHERE CLAIM_STATUS = 'DENIED'")
            r = c.fetchone()
            if r[0]:
                kpis['denied_amount'] = round(r[0], 2)

            c.execute('SELECT COUNT(*) FROM members')
            total_members = c.fetchone()[0]
            kpis['total_members'] = total_members
            member_months = total_members * 12
            kpis['pmpm_paid'] = round(total_paid / member_months, 2) if member_months else 0
            kpis['pmpm_billed'] = round(total_billed / member_months, 2) if member_months else 0

            c.execute('''SELECT
                CASE WHEN CAST(RISK_SCORE AS REAL) < 0.5 THEN 'Low'
                     WHEN CAST(RISK_SCORE AS REAL) < 1.0 THEN 'Moderate'
                     WHEN CAST(RISK_SCORE AS REAL) < 1.5 THEN 'High'
                     ELSE 'Very High' END as tier, COUNT(*)
                FROM members GROUP BY tier''')
            kpis['risk_distribution'] = dict(c.fetchall())

            c.execute('SELECT VISIT_TYPE, COUNT(*) FROM encounters GROUP BY VISIT_TYPE ORDER BY COUNT(*) DESC')
            kpis['visit_type_mix'] = dict(c.fetchall())

            c.execute("SELECT DENIAL_REASON, COUNT(*), SUM(CAST(BILLED_AMOUNT AS REAL)) FROM claims WHERE DENIAL_REASON != '' GROUP BY DENIAL_REASON ORDER BY COUNT(*) DESC")
            kpis['denial_reasons'] = [{'reason': r[0], 'count': r[1], 'amount': round(r[2], 2)} for r in c.fetchall()]

            c.execute("SELECT COUNT(*) FROM providers WHERE STATUS = 'ACTIVE'")
            kpis['active_providers'] = c.fetchone()[0]

            c.execute("SELECT COUNT(*) FROM appointments WHERE STATUS = 'NO_SHOW'")
            no_shows = c.fetchone()[0]
            c.execute('SELECT COUNT(*) FROM appointments')
            total_appts = c.fetchone()[0]
            kpis['no_show_rate'] = round(no_shows / total_appts * 100, 1) if total_appts else 0

            c.execute('SELECT SUM(CAST(COST AS REAL)), COUNT(*) FROM prescriptions')
            r = c.fetchone()
            kpis['total_pharmacy_spend'] = round(r[0], 2) if r[0] else 0
            kpis['total_prescriptions'] = r[1]
            kpis['pmpm_pharmacy'] = round(kpis['total_pharmacy_spend'] / member_months, 2) if member_months else 0

            c.execute("SELECT COUNT(*) FROM referrals WHERE STATUS = 'COMPLETED'")
            completed_refs = c.fetchone()[0]
            c.execute('SELECT COUNT(*) FROM referrals')
            total_refs = c.fetchone()[0]
            kpis['referral_completion_rate'] = round(completed_refs / total_refs * 100, 1) if total_refs else 0

            conn.close()
            return kpis

        except Exception as e:
            logger.error(f"Error computing live KPIs: {e}")
            return {'error': str(e)}


_data_dictionary = None

def get_data_dictionary(db_path: str = None) -> DataDictionary:
    global _data_dictionary
    if _data_dictionary is None:
        _data_dictionary = DataDictionary(db_path)
    elif db_path and _data_dictionary.db_path != db_path:
        _data_dictionary.db_path = db_path
    return _data_dictionary
