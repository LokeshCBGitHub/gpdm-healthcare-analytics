"""
Session Context Manager — Deep Conversational Memory & Intelligence

This module is the brain of the chatbot's conversational awareness:

1. SessionContextManager
   - Persistent per-user, per-session context (survives disconnections)
   - Stores: selected tables, active filters, catalog context, conversation turns
   - Cross-session memory: remembers user preferences across sessions

2. ConversationChainer
   - Detects if a new question relates to previous questions
   - Carries forward: tables, filters, GROUP BY from previous queries
   - Resolves pronouns: "those", "that", "same", "the above"
   - Example: Q1="total claims" → Q2="by region" → "total claims by region"

3. DuplicateDetector
   - Catches semantically similar repeated questions
   - Shows: "You asked this 3 min ago. Results: X rows. Run again?"
   - Configurable similarity threshold

4. TableHierarchyResolver
   - Multi-level drill-down: concept → type → subtype → specific table
   - Detects what level the user is asking about
   - Returns structured hierarchy for UI rendering

All data is SQLite-backed for persistence across server restarts.
"""

import os
import json
import sqlite3
import re
import time
import hashlib
import secrets
import threading
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from datetime import datetime, timedelta


# =============================================================================
# TABLE HIERARCHY — Multi-level drill-down structure
# =============================================================================
# This goes DEEP: concept → category → type → subtype
# When user says "claims", we show them ALL levels and let them pick.

TABLE_HIERARCHY = {
    'claims': {
        'label': 'Claims',
        'description': 'Healthcare claims and billing data',
        'icon': '&#128179;',  # receipt
        'categories': {
            'institutional': {
                'label': 'Institutional Claims',
                'description': 'Facility-based claims (hospitals, SNFs, etc.)',
                'icon': '&#127979;',  # hospital
                'types': {
                    'inpatient': {
                        'label': 'Inpatient Claims',
                        'description': 'Inpatient hospital stays (UB-04)',
                        'keywords': ['inpatient', 'hospital stay', 'admission', 'ub04', 'ub-04'],
                        'filter': "CLAIM_TYPE = 'INSTITUTIONAL' AND ENCOUNTER_ID IN (SELECT ENCOUNTER_ID FROM encounters WHERE VISIT_TYPE IN ('INPATIENT','OBSERVATION'))",
                        'simple_filter': "CLAIM_TYPE = 'INSTITUTIONAL'",
                        'columns': ['CLAIM_ID', 'MEMBER_ID', 'ENCOUNTER_ID', 'SERVICE_DATE',
                                    'BILLED_AMOUNT', 'PAID_AMOUNT', 'CLAIM_STATUS', 'ICD10_CODE',
                                    'FACILITY', 'KP_REGION'],
                    },
                    'outpatient_facility': {
                        'label': 'Outpatient Facility Claims',
                        'description': 'Outpatient facility claims (hospital outpatient dept)',
                        'keywords': ['outpatient facility', 'hospital outpatient', 'opd'],
                        'filter': "CLAIM_TYPE = 'INSTITUTIONAL'",
                        'columns': ['CLAIM_ID', 'MEMBER_ID', 'SERVICE_DATE', 'CPT_CODE',
                                    'BILLED_AMOUNT', 'PAID_AMOUNT', 'FACILITY'],
                    },
                    'snf': {
                        'label': 'Skilled Nursing Facility',
                        'description': 'SNF / long-term care claims',
                        'keywords': ['snf', 'skilled nursing', 'nursing facility', 'long term care', 'ltc'],
                        'filter': "FACILITY LIKE '%NURSING%' OR FACILITY LIKE '%SNF%'",
                        'columns': ['CLAIM_ID', 'MEMBER_ID', 'SERVICE_DATE', 'BILLED_AMOUNT',
                                    'PAID_AMOUNT', 'FACILITY'],
                    },
                },
            },
            'professional': {
                'label': 'Professional Claims',
                'description': 'Provider/physician claims (CMS-1500)',
                'icon': '&#129658;',  # stethoscope
                'types': {
                    'physician': {
                        'label': 'Physician Claims',
                        'description': 'Office visits and professional services',
                        'keywords': ['physician', 'doctor', 'office visit', 'professional', 'cms1500'],
                        'filter': "CLAIM_TYPE = 'PROFESSIONAL'",
                        'columns': ['CLAIM_ID', 'MEMBER_ID', 'RENDERING_NPI', 'SERVICE_DATE',
                                    'CPT_CODE', 'CPT_DESCRIPTION', 'BILLED_AMOUNT', 'PAID_AMOUNT',
                                    'CLAIM_STATUS'],
                    },
                    'specialist': {
                        'label': 'Specialist Claims',
                        'description': 'Claims from specialist providers',
                        'keywords': ['specialist', 'referral claim', 'specialty'],
                        'filter': "CLAIM_TYPE = 'PROFESSIONAL' AND RENDERING_NPI IN (SELECT NPI FROM providers WHERE SPECIALTY NOT IN ('GENERAL PRACTICE','FAMILY MEDICINE','INTERNAL MEDICINE','PEDIATRICS'))",
                        'simple_filter': "CLAIM_TYPE = 'PROFESSIONAL'",
                        'columns': ['CLAIM_ID', 'MEMBER_ID', 'RENDERING_NPI', 'CPT_CODE',
                                    'BILLED_AMOUNT', 'PAID_AMOUNT'],
                    },
                    'telehealth': {
                        'label': 'Telehealth Claims',
                        'description': 'Virtual/remote visit claims',
                        'keywords': ['telehealth', 'virtual', 'telemedicine', 'video visit', 'remote'],
                        'filter': "CPT_CODE IN ('99421','99422','99423','99441','99442','99443','99201','99202') OR CPT_DESCRIPTION LIKE '%TELEHEALTH%'",
                        'simple_filter': "CPT_DESCRIPTION LIKE '%TELEHEALTH%'",
                        'columns': ['CLAIM_ID', 'MEMBER_ID', 'SERVICE_DATE', 'CPT_CODE',
                                    'CPT_DESCRIPTION', 'PAID_AMOUNT'],
                    },
                },
            },
            'pharmacy': {
                'label': 'Pharmacy Claims',
                'description': 'Prescription drug and medication claims',
                'icon': '&#128138;',  # pill
                'types': {
                    'retail': {
                        'label': 'Retail Pharmacy',
                        'description': 'Retail pharmacy fills (CVS, Walgreens, etc.)',
                        'keywords': ['retail pharmacy', 'retail rx', 'drugstore'],
                        'filter': "CLAIM_TYPE = 'PHARMACY'",
                        'columns': ['CLAIM_ID', 'MEMBER_ID', 'SERVICE_DATE', 'BILLED_AMOUNT',
                                    'PAID_AMOUNT', 'CLAIM_STATUS'],
                    },
                    'specialty_rx': {
                        'label': 'Specialty Pharmacy',
                        'description': 'High-cost specialty medications',
                        'keywords': ['specialty pharmacy', 'specialty drug', 'specialty rx', 'biologics'],
                        'filter': "CLAIM_TYPE = 'PHARMACY' AND CAST(BILLED_AMOUNT AS REAL) > 1000",
                        'simple_filter': "CLAIM_TYPE = 'PHARMACY'",
                        'columns': ['CLAIM_ID', 'MEMBER_ID', 'BILLED_AMOUNT', 'PAID_AMOUNT'],
                    },
                    'mail_order': {
                        'label': 'Mail-Order Pharmacy',
                        'description': 'Mail-order prescription fills (90-day supply)',
                        'keywords': ['mail order', 'mail pharmacy', '90 day supply'],
                        'filter': "CLAIM_TYPE = 'PHARMACY'",
                        'columns': ['CLAIM_ID', 'MEMBER_ID', 'SERVICE_DATE', 'PAID_AMOUNT'],
                    },
                },
            },
            'dental': {
                'label': 'Dental Claims',
                'description': 'Dental and oral health claims',
                'icon': '&#129463;',  # tooth
                'types': {
                    'preventive': {
                        'label': 'Preventive Dental',
                        'description': 'Cleanings, exams, x-rays',
                        'keywords': ['dental preventive', 'cleaning', 'dental exam', 'dental xray'],
                        'filter': "CLAIM_TYPE = 'DENTAL'",
                        'columns': ['CLAIM_ID', 'MEMBER_ID', 'SERVICE_DATE', 'CPT_CODE',
                                    'BILLED_AMOUNT', 'PAID_AMOUNT'],
                    },
                    'restorative': {
                        'label': 'Restorative Dental',
                        'description': 'Fillings, crowns, bridges, implants',
                        'keywords': ['dental restorative', 'filling', 'crown', 'bridge', 'implant'],
                        'filter': "CLAIM_TYPE = 'DENTAL'",
                        'columns': ['CLAIM_ID', 'MEMBER_ID', 'SERVICE_DATE', 'CPT_CODE',
                                    'BILLED_AMOUNT', 'PAID_AMOUNT'],
                    },
                },
            },
            'behavioral': {
                'label': 'Behavioral Health Claims',
                'description': 'Mental health and substance use disorder claims',
                'icon': '&#129504;',  # brain
                'types': {
                    'mental_health': {
                        'label': 'Mental Health',
                        'description': 'Psychiatric, therapy, counseling claims',
                        'keywords': ['mental health', 'psychiatric', 'therapy', 'counseling', 'psychotherapy'],
                        'filter': "ICD10_CODE LIKE 'F%' OR CPT_DESCRIPTION LIKE '%PSYCH%' OR CPT_DESCRIPTION LIKE '%THERAP%'",
                        'simple_filter': "ICD10_CODE LIKE 'F%'",
                        'columns': ['CLAIM_ID', 'MEMBER_ID', 'SERVICE_DATE', 'ICD10_CODE',
                                    'ICD10_DESCRIPTION', 'CPT_CODE', 'PAID_AMOUNT'],
                    },
                    'substance_use': {
                        'label': 'Substance Use Disorder',
                        'description': 'SUD treatment, rehab, MAT claims',
                        'keywords': ['substance use', 'substance abuse', 'sud', 'rehab', 'addiction', 'mat'],
                        'filter': "ICD10_CODE LIKE 'F1%'",
                        'columns': ['CLAIM_ID', 'MEMBER_ID', 'SERVICE_DATE', 'ICD10_CODE',
                                    'ICD10_DESCRIPTION', 'PAID_AMOUNT'],
                    },
                },
            },
            'dme': {
                'label': 'DME Claims',
                'description': 'Durable medical equipment claims',
                'icon': '&#129468;',  # wheelchair
                'types': {
                    'dme_general': {
                        'label': 'DME Equipment',
                        'description': 'Wheelchairs, CPAP, prosthetics, etc.',
                        'keywords': ['dme', 'durable medical', 'equipment', 'wheelchair', 'cpap', 'prosthetic'],
                        'filter': "CLAIM_TYPE = 'DME'",
                        'columns': ['CLAIM_ID', 'MEMBER_ID', 'SERVICE_DATE', 'CPT_CODE',
                                    'CPT_DESCRIPTION', 'BILLED_AMOUNT', 'PAID_AMOUNT'],
                    },
                },
            },
            'all_claims': {
                'label': 'All Claims (Combined)',
                'description': 'All claim types combined — no filters applied',
                'icon': '&#128203;',  # clipboard
                'types': {
                    'all': {
                        'label': 'All Claims',
                        'description': 'Every claim regardless of type',
                        'keywords': ['all claims', 'total claims', 'combined', 'overall', 'general'],
                        'filter': '',
                        'columns': [],
                    },
                },
            },
        },
    },
    'members': {
        'label': 'Members',
        'description': 'Member demographics, enrollment, and risk data',
        'icon': '&#128101;',  # people
        'categories': {
            'demographics': {
                'label': 'Demographics',
                'description': 'Member personal and demographic information',
                'icon': '&#128100;',
                'types': {
                    'demographics': {
                        'label': 'Member Demographics',
                        'description': 'Age, gender, race, language, address, contact info',
                        'keywords': ['demographics', 'age', 'gender', 'race', 'personal info'],
                        'filter': '',
                        'columns': ['MEMBER_ID', 'FIRST_NAME', 'LAST_NAME', 'DATE_OF_BIRTH',
                                    'GENDER', 'RACE', 'LANGUAGE', 'ADDRESS', 'CITY', 'STATE',
                                    'ZIP', 'PHONE', 'EMAIL'],
                    },
                },
            },
            'enrollment': {
                'label': 'Enrollment',
                'description': 'Plan enrollment and eligibility',
                'icon': '&#128196;',
                'types': {
                    'enrollment': {
                        'label': 'Enrollment Records',
                        'description': 'Plan type, enrollment dates, coverage status',
                        'keywords': ['enrollment', 'eligibility', 'coverage', 'plan'],
                        'filter': '',
                        'columns': ['MEMBER_ID', 'PLAN_TYPE', 'ENROLLMENT_DATE', 'KP_REGION',
                                    'PCP_NPI'],
                    },
                },
            },
            'risk': {
                'label': 'Risk & Acuity',
                'description': 'Risk scores, HCC, chronic conditions',
                'icon': '&#9888;',
                'types': {
                    'risk_scores': {
                        'label': 'Risk Scores',
                        'description': 'HCC risk scores, RAF, acuity levels',
                        'keywords': ['risk', 'hcc', 'raf', 'acuity', 'risk score'],
                        'filter': '',
                        'columns': ['MEMBER_ID', 'RISK_SCORE', 'CHRONIC_CONDITIONS'],
                    },
                },
            },
        },
    },
    'encounters': {
        'label': 'Encounters',
        'description': 'Clinical encounters and visits',
        'icon': '&#127973;',
        'categories': {
            'inpatient': {
                'label': 'Inpatient Encounters',
                'description': 'Hospital admissions and stays',
                'icon': '&#128719;',
                'types': {
                    'inpatient_enc': {
                        'label': 'Inpatient Admissions',
                        'description': 'Hospital stays with admit/discharge dates',
                        'keywords': ['inpatient', 'admission', 'hospital stay'],
                        'filter': "VISIT_TYPE IN ('INPATIENT','OBSERVATION')",
                        'columns': ['ENCOUNTER_ID', 'MEMBER_ID', 'ADMIT_DATE', 'DISCHARGE_DATE',
                                    'VISIT_TYPE', 'LENGTH_OF_STAY', 'DISPOSITION', 'PRIMARY_DIAGNOSIS'],
                    },
                },
            },
            'outpatient': {
                'label': 'Outpatient Encounters',
                'description': 'Office visits and ambulatory care',
                'icon': '&#128203;',
                'types': {
                    'outpatient_enc': {
                        'label': 'Outpatient Visits',
                        'description': 'Office visits, ambulatory encounters',
                        'keywords': ['outpatient', 'office visit', 'ambulatory'],
                        'filter': "VISIT_TYPE = 'OUTPATIENT'",
                        'columns': ['ENCOUNTER_ID', 'MEMBER_ID', 'ADMIT_DATE', 'VISIT_TYPE',
                                    'CHIEF_COMPLAINT', 'PRIMARY_DIAGNOSIS'],
                    },
                },
            },
            'emergency': {
                'label': 'Emergency',
                'description': 'Emergency department visits',
                'icon': '&#128680;',
                'types': {
                    'ed_enc': {
                        'label': 'ED Visits',
                        'description': 'Emergency department encounters',
                        'keywords': ['emergency', 'er', 'ed', 'emergency room', 'emergency dept'],
                        'filter': "VISIT_TYPE = 'EMERGENCY'",
                        'columns': ['ENCOUNTER_ID', 'MEMBER_ID', 'ADMIT_DATE', 'DISCHARGE_DATE',
                                    'CHIEF_COMPLAINT', 'DISPOSITION'],
                    },
                },
            },
        },
    },
    'providers': {
        'label': 'Providers',
        'description': 'Provider directory and network',
        'icon': '&#129658;',
        'categories': {
            'directory': {
                'label': 'Provider Directory',
                'description': 'Provider demographics and specialties',
                'icon': '&#128209;',
                'types': {
                    'provider_info': {
                        'label': 'Provider Information',
                        'description': 'NPI, name, specialty, department, panel size',
                        'keywords': ['provider', 'doctor', 'physician', 'npi', 'specialty'],
                        'filter': '',
                        'columns': [],
                    },
                },
            },
        },
    },
    'diagnoses': {
        'label': 'Diagnoses',
        'description': 'Clinical diagnosis codes and conditions',
        'icon': '&#129657;',
        'categories': {
            'all_dx': {
                'label': 'All Diagnoses',
                'description': 'ICD-10, HCC codes, severity, chronic conditions',
                'icon': '&#128203;',
                'types': {
                    'diagnosis_info': {
                        'label': 'Diagnosis Records',
                        'description': 'ICD-10 codes, HCC categories, severity levels',
                        'keywords': ['diagnosis', 'icd', 'hcc', 'condition'],
                        'filter': '',
                        'columns': [],
                    },
                },
            },
        },
    },
    'prescriptions': {
        'label': 'Prescriptions',
        'description': 'Medication and prescription data',
        'icon': '&#128138;',
        'categories': {
            'all_rx': {
                'label': 'All Prescriptions',
                'description': 'Medications, refills, pharmacy, costs',
                'icon': '&#128138;',
                'types': {
                    'rx_info': {
                        'label': 'Prescription Records',
                        'description': 'Medication name, class, NDC, pharmacy, cost',
                        'keywords': ['prescription', 'medication', 'drug', 'rx'],
                        'filter': '',
                        'columns': [],
                    },
                },
            },
        },
    },
    'referrals': {
        'label': 'Referrals',
        'description': 'Care referrals and authorizations',
        'icon': '&#128228;',
        'categories': {
            'all_ref': {
                'label': 'All Referrals',
                'description': 'Referral reason, urgency, authorization status',
                'icon': '&#128228;',
                'types': {
                    'referral_info': {
                        'label': 'Referral Records',
                        'description': 'Referral reason, urgency, auth status, appointment date',
                        'keywords': ['referral', 'authorization', 'auth', 'referred'],
                        'filter': '',
                        'columns': [],
                    },
                },
            },
        },
    },
}


# =============================================================================
# SESSION CONTEXT MANAGER
# =============================================================================

class SessionContextManager:
    """
    Persistent conversational memory — survives disconnections and server restarts.

    Stores per-user, per-session:
    - Active table selections (from hierarchy drill-down)
    - Active filters (carried from previous queries)
    - Conversation turns (full Q&A history with SQL, results)
    - User preferences (default catalog, preferred tables, etc.)

    All data stored in SQLite for durability.
    """

    def __init__(self, db_path: Optional[str] = None):
        if not db_path:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(os.path.dirname(script_dir), 'data', 'sessions.db')
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        conn = self._get_conn()
        try:
            conn.executescript("""
                -- Conversation turns: every Q&A exchange
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    turn_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    turn_number INTEGER NOT NULL,
                    question TEXT NOT NULL,
                    normalized_question TEXT,
                    sql_generated TEXT,
                    tables_used TEXT,          -- JSON array
                    columns_used TEXT,         -- JSON array
                    filters_applied TEXT,      -- JSON array
                    group_by_cols TEXT,        -- JSON array
                    intent TEXT,
                    result_count INTEGER DEFAULT 0,
                    result_summary TEXT,       -- first few rows as JSON
                    is_followup INTEGER DEFAULT 0,
                    parent_turn_id TEXT,       -- which turn this follows up on
                    selected_hierarchy TEXT,   -- JSON: user's table hierarchy selection
                    created_at TEXT DEFAULT (datetime('now'))
                );

                -- Active session state: what the user currently has selected
                CREATE TABLE IF NOT EXISTS session_state (
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    state_key TEXT NOT NULL,
                    state_value TEXT NOT NULL,
                    updated_at TEXT DEFAULT (datetime('now')),
                    PRIMARY KEY (user_id, session_id, state_key)
                );

                -- Cross-session memory: persistent user preferences
                CREATE TABLE IF NOT EXISTS user_memory (
                    user_id TEXT NOT NULL,
                    memory_key TEXT NOT NULL,
                    memory_value TEXT NOT NULL,
                    source TEXT DEFAULT 'auto',  -- 'auto' or 'explicit'
                    confidence REAL DEFAULT 1.0,
                    access_count INTEGER DEFAULT 1,
                    updated_at TEXT DEFAULT (datetime('now')),
                    PRIMARY KEY (user_id, memory_key)
                );

                -- Indexes
                CREATE INDEX IF NOT EXISTS idx_turns_user_session
                    ON conversation_turns(user_id, session_id, turn_number);
                CREATE INDEX IF NOT EXISTS idx_turns_question
                    ON conversation_turns(normalized_question);
                CREATE INDEX IF NOT EXISTS idx_state_user
                    ON session_state(user_id, session_id);
                CREATE INDEX IF NOT EXISTS idx_memory_user
                    ON user_memory(user_id);
            """)
            conn.commit()
        finally:
            conn.close()

    # ─── CONVERSATION TURNS ───

    def add_turn(self, user_id: str, session_id: str, question: str,
                 sql: str = "", tables: List[str] = None,
                 columns: List[str] = None, filters: List[str] = None,
                 group_by: List[str] = None, intent: str = "",
                 result_count: int = 0, result_summary: Any = None,
                 is_followup: bool = False, parent_turn_id: str = "",
                 selected_hierarchy: Dict = None) -> str:
        """Record a conversation turn. Returns the turn_id."""
        turn_id = f"turn_{secrets.token_hex(8)}"
        normalized = self._normalize(question)

        # Get next turn number
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT MAX(turn_number) FROM conversation_turns WHERE user_id=? AND session_id=?",
                (user_id, session_id)
            ).fetchone()
            turn_number = (row[0] or 0) + 1

            # Summarize results (first 3 rows, truncated)
            summary = None
            if result_summary:
                if isinstance(result_summary, list):
                    summary = json.dumps(result_summary[:3])
                else:
                    summary = str(result_summary)[:500]

            conn.execute("""
                INSERT INTO conversation_turns
                (turn_id, user_id, session_id, turn_number, question, normalized_question,
                 sql_generated, tables_used, columns_used, filters_applied, group_by_cols,
                 intent, result_count, result_summary, is_followup, parent_turn_id,
                 selected_hierarchy)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                turn_id, user_id, session_id, turn_number, question, normalized,
                sql, json.dumps(tables or []), json.dumps(columns or []),
                json.dumps(filters or []), json.dumps(group_by or []),
                intent, result_count, summary,
                1 if is_followup else 0, parent_turn_id,
                json.dumps(selected_hierarchy) if selected_hierarchy else None,
            ))
            conn.commit()
            return turn_id
        finally:
            conn.close()

    def get_turns(self, user_id: str, session_id: str,
                  limit: int = 50) -> List[Dict]:
        """Get conversation history for a session."""
        conn = self._get_conn()
        try:
            rows = conn.execute("""
                SELECT * FROM conversation_turns
                WHERE user_id=? AND session_id=?
                ORDER BY turn_number ASC
                LIMIT ?
            """, (user_id, session_id, limit)).fetchall()
            return [self._row_to_dict(r) for r in rows]
        finally:
            conn.close()

    def get_last_turn(self, user_id: str, session_id: str) -> Optional[Dict]:
        """Get the most recent conversation turn."""
        conn = self._get_conn()
        try:
            row = conn.execute("""
                SELECT * FROM conversation_turns
                WHERE user_id=? AND session_id=?
                ORDER BY turn_number DESC LIMIT 1
            """, (user_id, session_id)).fetchone()
            return self._row_to_dict(row) if row else None
        finally:
            conn.close()

    def get_recent_turns(self, user_id: str, session_id: str,
                         count: int = 5) -> List[Dict]:
        """Get last N turns (most recent first)."""
        conn = self._get_conn()
        try:
            rows = conn.execute("""
                SELECT * FROM conversation_turns
                WHERE user_id=? AND session_id=?
                ORDER BY turn_number DESC LIMIT ?
            """, (user_id, session_id, count)).fetchall()
            return [self._row_to_dict(r) for r in reversed(rows)]
        finally:
            conn.close()

    # ─── SESSION STATE ───

    def set_state(self, user_id: str, session_id: str, key: str, value: Any):
        """Set a session state value (e.g., selected tables, active filters)."""
        conn = self._get_conn()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO session_state
                (user_id, session_id, state_key, state_value, updated_at)
                VALUES (?,?,?,?,datetime('now'))
            """, (user_id, session_id, key, json.dumps(value)))
            conn.commit()
        finally:
            conn.close()

    def get_state(self, user_id: str, session_id: str, key: str = None) -> Any:
        """Get session state. If key is None, returns all state as dict."""
        conn = self._get_conn()
        try:
            if key:
                row = conn.execute(
                    "SELECT state_value FROM session_state WHERE user_id=? AND session_id=? AND state_key=?",
                    (user_id, session_id, key)
                ).fetchone()
                return json.loads(row['state_value']) if row else None
            else:
                rows = conn.execute(
                    "SELECT state_key, state_value FROM session_state WHERE user_id=? AND session_id=?",
                    (user_id, session_id)
                ).fetchall()
                return {r['state_key']: json.loads(r['state_value']) for r in rows}
        finally:
            conn.close()

    # ─── CROSS-SESSION MEMORY ───

    def remember(self, user_id: str, key: str, value: Any,
                 source: str = 'auto', confidence: float = 1.0):
        """Store a persistent memory about the user."""
        conn = self._get_conn()
        try:
            existing = conn.execute(
                "SELECT access_count FROM user_memory WHERE user_id=? AND memory_key=?",
                (user_id, key)
            ).fetchone()
            count = (existing['access_count'] + 1) if existing else 1

            conn.execute("""
                INSERT OR REPLACE INTO user_memory
                (user_id, memory_key, memory_value, source, confidence, access_count, updated_at)
                VALUES (?,?,?,?,?,?,datetime('now'))
            """, (user_id, key, json.dumps(value), source, confidence, count))
            conn.commit()
        finally:
            conn.close()

    def recall(self, user_id: str, key: str = None) -> Any:
        """Recall a memory. If key is None, returns all memories."""
        conn = self._get_conn()
        try:
            if key:
                row = conn.execute(
                    "SELECT memory_value, confidence FROM user_memory WHERE user_id=? AND memory_key=?",
                    (user_id, key)
                ).fetchone()
                return json.loads(row['memory_value']) if row else None
            else:
                rows = conn.execute(
                    "SELECT memory_key, memory_value, confidence, access_count FROM user_memory WHERE user_id=? ORDER BY updated_at DESC",
                    (user_id,)
                ).fetchall()
                return {r['memory_key']: {
                    'value': json.loads(r['memory_value']),
                    'confidence': r['confidence'],
                    'access_count': r['access_count'],
                } for r in rows}
        finally:
            conn.close()

    def get_all_sessions(self, user_id: str) -> List[Dict]:
        """Get all sessions for a user with summary info."""
        conn = self._get_conn()
        try:
            rows = conn.execute("""
                SELECT session_id,
                       MIN(created_at) as started_at,
                       MAX(created_at) as last_activity,
                       COUNT(*) as turn_count,
                       GROUP_CONCAT(DISTINCT intent) as intents
                FROM conversation_turns
                WHERE user_id = ?
                GROUP BY session_id
                ORDER BY MAX(created_at) DESC
            """, (user_id,)).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # ─── HELPERS ───

    def _normalize(self, question: str) -> str:
        q = question.lower().strip()
        q = re.sub(r'\b\d+\b', '<N>', q)
        q = re.sub(r'"[^"]*"', '<V>', q)
        q = re.sub(r"'[^']*'", '<V>', q)
        q = re.sub(r'\s+', ' ', q)
        return q

    def _row_to_dict(self, row) -> Dict:
        d = dict(row)
        for k in ['tables_used', 'columns_used', 'filters_applied', 'group_by_cols']:
            if d.get(k):
                try:
                    d[k] = json.loads(d[k])
                except (json.JSONDecodeError, TypeError):
                    d[k] = []
        if d.get('selected_hierarchy'):
            try:
                d['selected_hierarchy'] = json.loads(d['selected_hierarchy'])
            except (json.JSONDecodeError, TypeError):
                d['selected_hierarchy'] = None
        return d


# =============================================================================
# CONVERSATION CHAINER — Contextual question linking
# =============================================================================

class ConversationChainer:
    """
    Detects when a new question relates to previous questions and
    carries forward context (tables, filters, GROUP BY).

    Handles:
    - "by region" → adds GROUP BY REGION to previous query
    - "now show denied" → adds WHERE CLAIM_STATUS='DENIED' to previous context
    - "same but for 2024" → modifies time filter
    - "those members" → uses member IDs from previous result
    - "also include encounters" → adds encounters table
    - "break it down by provider" → adds GROUP BY to previous
    - "what about pharmacy?" → switches claim type context
    """

    # Patterns that indicate a follow-up question
    FOLLOWUP_PATTERNS = [
        # Direct continuations
        r'\b(by|per|for each|broken down by|grouped by|group by)\s+\w+',
        r'\b(now|also|and|but|instead)\b',
        r'\b(same|those|that|these|the above|previous|last)\b',
        r'\b(what about|how about|and for|also for|also show|also include)\b',
        r'\b(more detail|drill down|break down|break it|split by)\b',
        r'\b(filter|only|just|exclude|remove|without)\b.*\b(from|in|the)\b',
        # Short queries (likely follow-ups)
        r'^(by\s+\w+)$',
        r'^(for\s+\w+)$',
        r'^(in\s+\w+)$',
        r'^(denied|paid|pending|approved)$',
        r'^(region|provider|specialty|plan|facility)$',
    ]

    # Words that signal a completely NEW question (not a follow-up)
    NEW_QUESTION_SIGNALS = [
        r'\bhow many\b',
        r'\bwhat is the\b',
        r'\bshow me all\b',
        r'\btotal\s+(number|count|amount)\b',
        r'\blist all\b',
        r'\bgive me\b',
        r'\bfind all\b',
        r'\bwhich\s+\w+\s+(have|has|are|is)\b',
        r'\bcompare\b',
    ]

    @classmethod
    def analyze(cls, question: str, recent_turns: List[Dict]) -> Dict:
        """
        Analyze if this question is a follow-up to recent conversation.

        Returns:
            {
                'is_followup': bool,
                'followup_type': str,  # 'add_groupby', 'add_filter', 'modify', 'drill_down', 'switch_context'
                'carry_forward': {
                    'tables': [...],
                    'filters': [...],
                    'group_by': [...],
                    'sql_base': str,  # the previous SQL to build upon
                },
                'confidence': float,
                'explanation': str,
            }
        """
        if not recent_turns:
            return {'is_followup': False, 'followup_type': None, 'carry_forward': {}, 'confidence': 0}

        q = question.lower().strip()
        last_turn = recent_turns[-1]

        # Check for follow-up patterns
        followup_score = 0
        followup_type = None

        # 1. Very short question → likely follow-up
        word_count = len(q.split())
        if word_count <= 3:
            followup_score += 0.4

        # 2. Check explicit follow-up patterns
        for pattern in cls.FOLLOWUP_PATTERNS:
            if re.search(pattern, q):
                followup_score += 0.3
                break

        # 3. Check if it references previous context
        if any(w in q for w in ['those', 'that', 'these', 'same', 'the above', 'previous', 'last']):
            followup_score += 0.4
            followup_type = 'reference'

        # 4. Check for "by X" pattern (adding GROUP BY)
        by_match = re.match(r'^(?:now\s+)?(?:break\s+(?:it\s+)?down\s+|group\s+)?by\s+(\w+(?:\s+\w+)?)\s*$', q)
        if by_match:
            followup_score = 0.95
            followup_type = 'add_groupby'

        # 5. Check for filter additions
        filter_match = re.match(r'^(?:only|just|filter|for)\s+(.+)$', q)
        if filter_match:
            followup_score = 0.9
            followup_type = 'add_filter'

        # 6. Check for "what about X" pattern
        what_about = re.match(r'^(?:what|how)\s+about\s+(.+)\??$', q)
        if what_about:
            followup_score = 0.85
            followup_type = 'switch_context'

        # 7. Check for drill-down
        if any(p in q for p in ['more detail', 'drill down', 'break down', 'expand', 'details']):
            followup_score = 0.9
            followup_type = 'drill_down'

        # 8. Counter-evidence: strong new-question signals
        for pattern in cls.NEW_QUESTION_SIGNALS:
            if re.search(pattern, q):
                # Only reduce if the question is long enough to be standalone
                if word_count >= 5:
                    followup_score *= 0.3
                break

        # 9. Check table overlap — if the question mentions the same tables
        if last_turn.get('tables_used'):
            last_tables = set(t.lower() for t in last_turn['tables_used'])
            for t in last_tables:
                if t.rstrip('s') in q or t in q:
                    followup_score += 0.1

        is_followup = followup_score >= 0.5

        # Build carry-forward context
        carry_forward = {}
        if is_followup:
            carry_forward = {
                'tables': last_turn.get('tables_used', []),
                'filters': last_turn.get('filters_applied', []),
                'group_by': last_turn.get('group_by_cols', []),
                'sql_base': last_turn.get('sql_generated', ''),
                'previous_question': last_turn.get('question', ''),
                'previous_intent': last_turn.get('intent', ''),
            }

            if not followup_type:
                followup_type = 'continuation'

        explanation = ''
        if is_followup:
            prev_q = last_turn.get('question', '')[:60]
            explanation = f'Continuing from: "{prev_q}"'
            if followup_type == 'add_groupby':
                explanation += f' — adding grouping'
            elif followup_type == 'add_filter':
                explanation += f' — adding filter'
            elif followup_type == 'switch_context':
                explanation += f' — switching focus'

        return {
            'is_followup': is_followup,
            'followup_type': followup_type,
            'carry_forward': carry_forward,
            'confidence': min(followup_score, 1.0),
            'explanation': explanation,
        }

    @classmethod
    def merge_context(cls, question: str, carry_forward: Dict,
                      followup_type: str) -> str:
        """
        Merge the follow-up question with carried-forward context
        to produce a complete standalone question.

        Example:
            previous: "total claims"
            current: "by region"
            merged: "total claims by region"
        """
        prev_question = carry_forward.get('previous_question', '')
        q = question.lower().strip()

        if followup_type == 'add_groupby':
            # "by region" → "total claims by region"
            by_match = re.match(r'^(?:now\s+)?(?:break\s+(?:it\s+)?down\s+|group\s+)?by\s+(.+)$', q)
            if by_match:
                group_term = by_match.group(1)
                # Remove any existing "by X" from previous question
                cleaned_prev = re.sub(r'\s+by\s+\w+(?:\s+\w+)?$', '', prev_question)
                return f"{cleaned_prev} by {group_term}"

        elif followup_type == 'add_filter':
            # "only denied" → "total claims where status is denied"
            filter_match = re.match(r'^(?:only|just|filter|for)\s+(.+)$', q)
            if filter_match:
                filter_term = filter_match.group(1)
                return f"{prev_question} {filter_term}"

        elif followup_type == 'switch_context':
            # "what about pharmacy?" → "total pharmacy claims"
            what_match = re.match(r'^(?:what|how)\s+about\s+(.+)\??$', q)
            if what_match:
                new_focus = what_match.group(1)
                # Try to replace the subject in the previous question
                return f"{new_focus} {prev_question}"

        elif followup_type == 'drill_down':
            return f"{prev_question} details"

        elif followup_type == 'reference':
            # Replace "those" / "that" with previous table reference
            tables = carry_forward.get('tables', [])
            if tables:
                main_table = tables[0]
                q = q.replace('those', main_table).replace('that', main_table)
                q = q.replace('the above', main_table).replace('previous', main_table)
            return q

        # Default: just combine
        if q.startswith(('and ', 'also ', 'but ')):
            return f"{prev_question} {q}"

        return question  # Can't merge, use as-is


# =============================================================================
# DUPLICATE DETECTOR
# =============================================================================

class DuplicateDetector:
    """
    Detects when a user asks the same or very similar question again.
    Uses normalized question comparison + word overlap scoring.
    """

    @classmethod
    def check(cls, question: str, recent_turns: List[Dict],
              threshold: float = 0.8) -> Optional[Dict]:
        """
        Check if question is a duplicate of a recent turn.

        Returns None if not a duplicate, or:
        {
            'is_duplicate': True,
            'original_turn': {...},
            'similarity': float,
            'time_ago': str,  # "3 minutes ago"
            'message': str,   # "You asked this 3 minutes ago..."
        }
        """
        if not recent_turns:
            return None

        q_normalized = cls._normalize(question)
        q_words = set(q_normalized.split())

        best_match = None
        best_score = 0

        for turn in recent_turns:
            turn_normalized = cls._normalize(turn.get('question', ''))
            turn_words = set(turn_normalized.split())

            # Exact match
            if q_normalized == turn_normalized:
                score = 1.0
            else:
                # Jaccard similarity on words
                if not q_words or not turn_words:
                    continue
                intersection = q_words & turn_words
                union = q_words | turn_words
                score = len(intersection) / len(union) if union else 0

            if score > best_score:
                best_score = score
                best_match = turn

        if best_score >= threshold and best_match:
            time_ago = cls._time_ago(best_match.get('created_at', ''))
            result_count = best_match.get('result_count', 0)

            return {
                'is_duplicate': True,
                'original_turn': best_match,
                'similarity': best_score,
                'time_ago': time_ago,
                'message': (
                    f"You asked this {time_ago}. "
                    f"It returned {result_count} result{'s' if result_count != 1 else ''}. "
                    f"Would you like to run it again?"
                ),
            }

        return None

    @staticmethod
    def _normalize(q: str) -> str:
        q = q.lower().strip()
        q = re.sub(r'[^\w\s]', '', q)
        q = re.sub(r'\s+', ' ', q)
        stop = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'me', 'my', 'i',
                'show', 'give', 'get', 'find', 'list', 'display', 'tell'}
        return ' '.join(w for w in q.split() if w not in stop)

    @staticmethod
    def _time_ago(timestamp: str) -> str:
        if not timestamp:
            return 'earlier'
        try:
            dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
            diff = datetime.now() - dt
            if diff.total_seconds() < 60:
                return 'just now'
            elif diff.total_seconds() < 3600:
                mins = int(diff.total_seconds() / 60)
                return f'{mins} minute{"s" if mins != 1 else ""} ago'
            elif diff.total_seconds() < 86400:
                hours = int(diff.total_seconds() / 3600)
                return f'{hours} hour{"s" if hours != 1 else ""} ago'
            else:
                days = int(diff.total_seconds() / 86400)
                return f'{days} day{"s" if days != 1 else ""} ago'
        except (ValueError, TypeError):
            return 'earlier'


# =============================================================================
# TABLE HIERARCHY RESOLVER — For the multi-select drill-down UI
# =============================================================================

class TableHierarchyResolver:
    """
    Resolves user queries against the TABLE_HIERARCHY to determine
    what level they're asking about, and returns structured data
    for the multi-select UI pane.
    """

    @classmethod
    def get_hierarchy_for_concept(cls, concept: str) -> Optional[Dict]:
        """Get the full hierarchy tree for a concept (e.g., 'claims')."""
        concept_lower = concept.lower().rstrip('s')
        for key, data in TABLE_HIERARCHY.items():
            if key == concept_lower or key.rstrip('s') == concept_lower or concept_lower in key:
                return {
                    'concept': key,
                    'label': data['label'],
                    'description': data['description'],
                    'icon': data.get('icon', ''),
                    'categories': {
                        cat_key: {
                            'label': cat['label'],
                            'description': cat['description'],
                            'icon': cat.get('icon', ''),
                            'types': {
                                type_key: {
                                    'label': t['label'],
                                    'description': t['description'],
                                    'keywords': t.get('keywords', []),
                                    'has_filter': bool(t.get('filter')),
                                }
                                for type_key, t in cat.get('types', {}).items()
                            }
                        }
                        for cat_key, cat in data.get('categories', {}).items()
                    },
                }
        return None

    @classmethod
    def get_full_hierarchy(cls) -> Dict:
        """Get the complete hierarchy tree for all concepts."""
        result = {}
        for key, data in TABLE_HIERARCHY.items():
            result[key] = {
                'label': data['label'],
                'description': data['description'],
                'icon': data.get('icon', ''),
                'category_count': len(data.get('categories', {})),
                'type_count': sum(
                    len(cat.get('types', {}))
                    for cat in data.get('categories', {}).values()
                ),
            }
        return result

    @classmethod
    def resolve_selection(cls, selections: List[str]) -> Dict:
        """
        Given a list of selected type keys (e.g., ['institutional.inpatient', 'pharmacy.retail']),
        resolve to actual SQL filters and columns.

        Returns:
            {
                'tables': ['claims'],
                'filters': ["CLAIM_TYPE = 'INSTITUTIONAL'", ...],
                'columns': [...],
                'description': 'Inpatient Claims + Retail Pharmacy Claims',
            }
        """
        all_filters = []
        all_columns = set()
        descriptions = []
        base_tables = set()

        for selection in selections:
            parts = selection.split('.')
            # Navigate hierarchy: concept.category.type
            concept_key = parts[0] if len(parts) > 0 else ''
            cat_key = parts[1] if len(parts) > 1 else ''
            type_key = parts[2] if len(parts) > 2 else ''

            concept_data = TABLE_HIERARCHY.get(concept_key, {})
            base_tables.add(concept_key)

            if cat_key:
                cat_data = concept_data.get('categories', {}).get(cat_key, {})
                if type_key:
                    type_data = cat_data.get('types', {}).get(type_key, {})
                    if type_data:
                        f = type_data.get('simple_filter') or type_data.get('filter', '')
                        if f:
                            all_filters.append(f)
                        all_columns.update(type_data.get('columns', []))
                        descriptions.append(type_data.get('label', type_key))
                else:
                    # Selected entire category — combine all type filters with OR
                    cat_filters = []
                    for t_key, t_data in cat_data.get('types', {}).items():
                        f = t_data.get('simple_filter') or t_data.get('filter', '')
                        if f:
                            cat_filters.append(f)
                        all_columns.update(t_data.get('columns', []))
                    if cat_filters:
                        if len(cat_filters) == 1:
                            all_filters.append(cat_filters[0])
                        else:
                            all_filters.append('(' + ' OR '.join(cat_filters) + ')')
                    descriptions.append(cat_data.get('label', cat_key))

        # Deduplicate filters
        unique_filters = list(dict.fromkeys(all_filters))

        return {
            'tables': list(base_tables),
            'filters': unique_filters,
            'columns': list(all_columns),
            'description': ' + '.join(descriptions) if descriptions else 'All data',
        }

    @classmethod
    def detect_from_question(cls, question: str) -> Optional[Dict]:
        """
        Detect which hierarchy level the user is asking about from their question.

        Returns the matched hierarchy path or None.
        """
        q = question.lower()

        for concept_key, concept_data in TABLE_HIERARCHY.items():
            for cat_key, cat_data in concept_data.get('categories', {}).items():
                for type_key, type_data in cat_data.get('types', {}).items():
                    for kw in type_data.get('keywords', []):
                        if kw in q:
                            return {
                                'concept': concept_key,
                                'category': cat_key,
                                'type': type_key,
                                'path': f"{concept_key}.{cat_key}.{type_key}",
                                'label': type_data['label'],
                                'filter': type_data.get('simple_filter') or type_data.get('filter', ''),
                                'columns': type_data.get('columns', []),
                            }

        return None
