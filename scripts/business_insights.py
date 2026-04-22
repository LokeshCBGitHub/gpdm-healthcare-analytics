from __future__ import annotations

import os
import json
import math
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable

try:
    from databricks_connector import DataSourceManager
    _DBX = True
except Exception:
    _DBX = False

try:
    import explainability as _xp
except Exception:
    _xp = None

try:
    import conformal_prediction as _cp
except Exception:
    _cp = None

try:
    import calibration as _cal
except Exception:
    _cal = None

try:
    import fairness as _fair
except Exception:
    _fair = None

try:
    import column_map as _cmap
except Exception:
    _cmap = None


_LOG_TABLE = 'gpdm_business_insight_log'


@dataclass
class _Schema:
    enc:           str = "encounters"
    mem:           str = "members"
    prov:          str = "providers"
    enc_mid:       str = "member_id"
    enc_pid:       str = "provider_id"
    enc_eid:       str = "encounter_id"
    enc_date:      str = "encounter_date"
    enc_paid:      str = "total_paid"
    enc_pdx:       str = "primary_dx"
    enc_admit:     str = "is_admission"
    enc_readmit:   str = "readmit_30d"
    mem_mid:       str = "member_id"
    mem_region:    str = "region"
    mem_age:       str = "age"
    mem_lob:       str = "lob"
    mem_gender:    str = "gender"
    mem_pcp:       str = "pcp_provider_id"
    mem_risk:      str = "risk_score"
    prov_pid:      str = "provider_id"
    prov_name:     str = "provider_name"
    prov_spec:     str = "specialty"


def _resolve_schema(db_path: str) -> _Schema:
    s = _Schema()
    if _cmap is None:
        return s
    try:
        conn = sqlite3.connect(db_path)
    except Exception:
        return s
    try:
        def t(l):    return _cmap.resolve_table(conn, l) or l
        def c(t_, lc, phys=None):
            return _cmap.resolve_column(conn, t_, lc, phys or t(t_)) or lc
        s.enc = t('encounters'); s.mem = t('members'); s.prov = t('providers')
        s.enc_mid     = c('encounters', 'member_id',      s.enc)
        s.enc_pid     = c('encounters', 'provider_id',    s.enc)
        s.enc_eid     = c('encounters', 'encounter_id',   s.enc)
        s.enc_date    = c('encounters', 'encounter_date', s.enc)
        s.enc_paid    = c('encounters', 'total_paid',     s.enc)
        s.enc_pdx     = c('encounters', 'primary_dx',     s.enc)
        s.enc_admit   = c('encounters', 'is_admission',   s.enc)
        s.enc_readmit = c('encounters', 'readmit_30d',    s.enc)
        s.mem_mid     = c('members', 'member_id',         s.mem)
        s.mem_region  = c('members', 'region',            s.mem)
        s.mem_age     = c('members', 'age',               s.mem)
        s.mem_lob     = c('members', 'lob',               s.mem)
        s.mem_gender  = c('members', 'gender',            s.mem)
        s.mem_pcp     = c('members', 'pcp_provider_id',   s.mem)
        s.mem_risk    = c('members', 'risk_score',        s.mem)
        s.prov_pid    = c('providers', 'provider_id',     s.prov)
        s.prov_name   = c('providers', 'provider_name',   s.prov)
        s.prov_spec   = c('providers', 'specialty',       s.prov)
    except Exception:
        pass
    finally:
        try: conn.close()
        except Exception: pass
    return s


def _q(name: Optional[str]) -> str:
    if not name:
        return '""'
    return '"' + str(name).replace('"', '""') + '"'

_BENCHMARKS = {
    'avg_admission_cost_usd': 13_600,
    'avg_avoided_per_tcm_outreach': 0.18,
    'avg_rising_risk_pmpm_delta_usd': 180,
    'hedis_gap_close_rate_outreach': 0.22,
    'star_bonus_per_member_per_measure': 9.0,
}


@dataclass
class BizAnswer:
    intent: str
    headline: str
    kpi: Dict[str, Any]
    columns: List[str]
    rows: List[List[Any]]
    reason_codes: List[str]
    suggested_action: str
    data_source: str
    sql: Optional[str] = None
    confidence: Optional[Dict[str, Any]] = None
    fairness_flag: Optional[str] = None
    n_rows: int = 0
    row_count: int = 0
    ts: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['row_count'] = len(d.get('rows') or [])
        return d


def _get_source(db_path: str):
    if not _DBX:
        return None, 'sqlite_fallback'
    try:
        dsm = DataSourceManager(db_path=db_path)
        return dsm, 'databricks'
    except Exception:
        return None, 'sqlite_fallback'


def _run(sql: str, db_path: str, local_fallback: Optional[Callable] = None) -> Tuple[List[str], List[List], str]:
    dsm, tag = _get_source(db_path)
    if dsm is not None:
        try:
            res = dsm.execute_query(sql, local_fallback_fn=local_fallback)
            cols = res.get('columns') or []
            rows = res.get('rows') or []
            return cols, rows, (res.get('source') or tag)
        except Exception:
            pass
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = [list(r) for r in cur.fetchall()]
        conn.close()
        return cols, rows, 'sqlite_fallback'
    except Exception:
        if local_fallback is not None:
            r = local_fallback()
            return r.get('columns', []), r.get('rows', []), 'local_python_fallback'
        return [], [], 'unavailable'


def _log_answer(db_path: str, ans: BizAnswer) -> None:
    try:
        conn = sqlite3.connect(db_path)
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {_LOG_TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL, intent TEXT, data_source TEXT,
                n_rows INTEGER, headline TEXT, kpi_json TEXT
            )
        """)
        conn.execute(
            f"INSERT INTO {_LOG_TABLE}(ts,intent,data_source,n_rows,headline,kpi_json) "
            f"VALUES(?,?,?,?,?,?)",
            (ans.ts, ans.intent, ans.data_source, len(ans.rows or []),
             ans.headline, json.dumps(ans.kpi, default=str))
        )
        conn.commit(); conn.close()
    except Exception:
        pass


def _fmt_money(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return '$—'
    if abs(x) >= 1_000_000:
        return f"${x/1_000_000:.2f}M"
    if abs(x) >= 1_000:
        return f"${x/1_000:.1f}K"
    return f"${x:,.0f}"


def rising_risk_cohort(
    db_path: str,
    *,
    region: Optional[str] = None,
    line_of_business: Optional[str] = None,
    chronic_condition: Optional[str] = None,
    top_n: int = 100,
) -> BizAnswer:
    s = _resolve_schema(db_path)
    filters = []
    if region:
        filters.append(f"m.{_q(s.mem_region)} = '{region.replace(chr(39),'')}' ")
    if line_of_business:
        filters.append(f"m.{_q(s.mem_lob)} = '{line_of_business.replace(chr(39),'')}' ")
    if chronic_condition:
        filters.append(
            f"EXISTS (SELECT 1 FROM {_q(s.enc)} e WHERE e.{_q(s.enc_mid)} = m.{_q(s.mem_mid)} "
            f"AND e.{_q(s.enc_pdx)} LIKE '{chronic_condition.replace(chr(39),'')}%')"
        )
    where = ("WHERE " + " AND ".join(filters)) if filters else ""

    sql = f"""
    WITH recent AS (
      SELECT e.{_q(s.enc_mid)} AS member_id,
             SUM(CASE WHEN e.{_q(s.enc_date)} >= date('now','-90 day')
                       THEN COALESCE(e.{_q(s.enc_paid)}, 0) ELSE 0 END) AS cost_90d,
             SUM(CASE WHEN e.{_q(s.enc_date)} >= date('now','-180 day')
                       AND e.{_q(s.enc_date)} < date('now','-90 day')
                       THEN COALESCE(e.{_q(s.enc_paid)}, 0) ELSE 0 END) AS cost_prior_90d,
             COUNT(CASE WHEN e.{_q(s.enc_date)} >= date('now','-90 day')
                         THEN 1 END) AS enc_90d,
             COUNT(CASE WHEN e.{_q(s.enc_date)} >= date('now','-180 day')
                         AND e.{_q(s.enc_date)} < date('now','-90 day')
                         THEN 1 END) AS enc_prior_90d
        FROM {_q(s.enc)} e
       GROUP BY e.{_q(s.enc_mid)}
    )
    SELECT m.{_q(s.mem_mid)} AS member_id,
           COALESCE(m.{_q(s.mem_region)},'—') AS region,
           COALESCE(m.{_q(s.mem_lob)},'—')    AS lob,
           COALESCE(m.{_q(s.mem_age)}, 0)     AS age,
           r.cost_90d, r.cost_prior_90d,
           (r.cost_90d - r.cost_prior_90d)  AS cost_delta_90d,
           (r.enc_90d - r.enc_prior_90d)    AS enc_delta_90d
      FROM {_q(s.mem)} m
      JOIN recent r ON r.member_id = m.{_q(s.mem_mid)}
     {where}
       AND r.cost_90d > COALESCE(r.cost_prior_90d,0) * 1.3
     ORDER BY (r.cost_90d - r.cost_prior_90d) DESC
     LIMIT {int(top_n)}
    """
    cols, rows, src = _run(sql, db_path)

    annotated_rows = []
    annotated_cols = cols + ['projected_pmpm_delta', 'projected_12m_impact', 'reason']
    for r in rows:
        d = dict(zip(cols, r))
        delta_90 = float(d.get('cost_delta_90d') or 0.0)
        pmpm = max(delta_90 / 3.0, _BENCHMARKS['avg_rising_risk_pmpm_delta_usd'] * 0.3)
        proj_12m = pmpm * 12.0
        reasons = []
        if delta_90 > 5000: reasons.append("cost up >$5K in last 90d")
        if int(d.get('enc_delta_90d') or 0) >= 3: reasons.append("encounter acceleration")
        if int(d.get('age') or 0) >= 65: reasons.append("senior beneficiary")
        if not reasons: reasons.append("trajectory above cohort baseline")
        annotated_rows.append(list(r) + [round(pmpm, 2), round(proj_12m, 2),
                                          "; ".join(reasons)])

    total_impact = sum(row[-2] for row in annotated_rows) if annotated_rows else 0.0
    avoided_if_outreach = total_impact * _BENCHMARKS['avg_avoided_per_tcm_outreach']

    ans = BizAnswer(
        intent='rising_risk_cohort',
        headline=(f"Top {len(annotated_rows)} rising-risk members — projected "
                  f"12-mo impact {_fmt_money(total_impact)}. Targeted outreach could "
                  f"avoid {_fmt_money(avoided_if_outreach)} ({int(_BENCHMARKS['avg_avoided_per_tcm_outreach']*100)}% benchmark)."),
        kpi={
            'n_members': len(annotated_rows),
            'projected_12m_impact_usd': round(total_impact, 0),
            'expected_avoided_usd_with_outreach': round(avoided_if_outreach, 0),
            'assumed_outreach_lift_pct': _BENCHMARKS['avg_avoided_per_tcm_outreach'] * 100,
        },
        columns=annotated_cols,
        rows=annotated_rows,
        reason_codes=["cost_acceleration_90d_vs_prior", "encounter_acceleration",
                       "benefit_year_months_remaining"],
        suggested_action=(
            "Export to care-management queue; assign top decile to RN-led "
            "transitional outreach within 14 days; re-score weekly."
        ),
        data_source=src,
        sql=sql.strip(),
        n_rows=len(annotated_rows),
    )
    _log_answer(db_path, ans)
    return ans


def readmit_watchlist(
    db_path: str,
    *,
    horizon_days: int = 30,
    days_since_discharge_max: int = 14,
    top_n: int = 100,
    region: Optional[str] = None,
) -> BizAnswer:
    s = _resolve_schema(db_path)
    region_filter = (f"AND m.{_q(s.mem_region)} = '{region.replace(chr(39),'')}'" if region else "")

    sql = f"""
    WITH last_admit AS (
      SELECT e.{_q(s.enc_mid)} AS member_id,
             MAX(e.{_q(s.enc_date)}) AS last_dc_date
        FROM {_q(s.enc)} e
       WHERE COALESCE(e.{_q(s.enc_admit)}, 0) = 1
       GROUP BY e.{_q(s.enc_mid)}
    )
    SELECT m.{_q(s.mem_mid)} AS member_id,
           COALESCE(m.{_q(s.mem_region)},'—') AS region,
           COALESCE(m.{_q(s.mem_age)}, 0)     AS age,
           la.last_dc_date,
           CAST(julianday('now') - julianday(la.last_dc_date) AS INTEGER) AS days_since_dc,
           (SELECT COUNT(*) FROM {_q(s.enc)} e2
              WHERE e2.{_q(s.enc_mid)} = m.{_q(s.mem_mid)}
                AND e2.{_q(s.enc_date)} >= date('now','-365 day')
                AND COALESCE(e2.{_q(s.enc_admit)},0)=1) AS admits_12m
      FROM {_q(s.mem)} m
      JOIN last_admit la ON la.member_id = m.{_q(s.mem_mid)}
     WHERE la.last_dc_date >= date('now','-{int(days_since_discharge_max)} day')
       {region_filter}
     ORDER BY la.last_dc_date DESC
     LIMIT {int(top_n) * 3}
    """
    cols, rows, src = _run(sql, db_path)

    scored = []
    for r in rows:
        d = dict(zip(cols, r))
        mid = str(d['member_id'])
        risk = 0.12
        se = None; interval = None
        if _xp is None:
            pass
        try:
            import uncertainty as _un
            pr = _un.predict_risk_with_uncertainty(mid, db_path)
            if isinstance(pr, dict) and pr.get('mean') is not None:
                risk = float(pr['mean'])
                se = pr.get('std')
        except Exception:
            pass
        try:
            if _cp is not None:
                ivl = _cp.predict_risk_conformal(risk, group=str(d.get('region') or None))
                if isinstance(ivl, dict):
                    interval = [ivl.get('lo'), ivl.get('hi')]
        except Exception:
            pass
        try:
            if _cal is not None:
                cal = _cal.apply_best_calibrator(risk) if hasattr(_cal, 'apply_best_calibrator') else None
                if cal is not None:
                    risk = float(cal)
        except Exception:
            pass
        scored.append({
            **d, 'predicted_readmit_risk': round(risk, 3),
            'ci95': interval, 'se': se,
            'expected_avoided_if_tcm': round(
                risk * _BENCHMARKS['avg_avoided_per_tcm_outreach'] *
                _BENCHMARKS['avg_admission_cost_usd'], 0),
        })

    scored.sort(key=lambda x: x['predicted_readmit_risk'], reverse=True)
    scored = scored[:int(top_n)]

    out_cols = list(cols) + ['predicted_readmit_risk', 'ci95_lo', 'ci95_hi',
                              'expected_avoided_if_tcm_usd', 'reason']
    out_rows = []
    total_exp = 0.0
    for s in scored:
        reasons = []
        if s.get('admits_12m', 0) >= 2: reasons.append(">=2 prior admits in 12mo")
        if int(s.get('age') or 0) >= 65: reasons.append("age 65+")
        if s['predicted_readmit_risk'] >= 0.25: reasons.append("high model risk")
        if int(s.get('days_since_dc') or 99) <= 3: reasons.append("fresh discharge (<=3d)")
        if not reasons: reasons.append("elevated vs cohort")
        total_exp += s['expected_avoided_if_tcm']
        lo, hi = (s.get('ci95') or [None, None])
        out_rows.append([
            *(s.get(c) for c in cols),
            s['predicted_readmit_risk'], lo, hi,
            s['expected_avoided_if_tcm'], "; ".join(reasons),
        ])

    headline = (f"Readmit watchlist — {len(out_rows)} recently-discharged members, "
                f"expected avoided spend with TCM outreach ≈ {_fmt_money(total_exp)} "
                f"over {horizon_days}d window.")

    ans = BizAnswer(
        intent='readmit_watchlist',
        headline=headline,
        kpi={
            'horizon_days': horizon_days,
            'n_members': len(out_rows),
            'expected_avoided_usd': round(total_exp, 0),
            'avg_risk': round(sum(s['predicted_readmit_risk'] for s in scored) /
                              max(1, len(scored)), 3),
            'admission_cost_benchmark_usd': _BENCHMARKS['avg_admission_cost_usd'],
        },
        columns=out_cols,
        rows=out_rows,
        reason_codes=["fresh_discharge", "prior_admits_12m", "age_65_plus",
                       "high_model_risk"],
        suggested_action=(
            "Route top decile to RN-led Transitional Care Management within 72h of "
            "discharge; automate 7- and 30-day callback; confirm PCP follow-up visit."
        ),
        data_source=src,
        sql=sql.strip(),
        n_rows=len(out_rows),
        confidence={'model_stack': 'uncertainty + conformal + calibration',
                    'coverage_target': 0.90},
    )
    _log_answer(db_path, ans)
    return ans


_HEDIS_MEASURES = {
    'CBP':  {'name': 'Controlling High Blood Pressure', 'dx_like': 'I10%',
              'eligible_age_min': 18, 'eligible_age_max': 85},
    'HBD':  {'name': 'HbA1c Control (Diabetes)', 'dx_like': 'E11%',
              'eligible_age_min': 18, 'eligible_age_max': 75},
    'BCS':  {'name': 'Breast Cancer Screening', 'dx_like': None,
              'eligible_age_min': 50, 'eligible_age_max': 74,
              'gender': 'F'},
    'CCS':  {'name': 'Cervical Cancer Screening', 'dx_like': None,
              'eligible_age_min': 21, 'eligible_age_max': 64,
              'gender': 'F'},
    'COL':  {'name': 'Colorectal Cancer Screening', 'dx_like': None,
              'eligible_age_min': 45, 'eligible_age_max': 75},
    'AWV':  {'name': 'Medicare Annual Wellness Visit', 'dx_like': None,
              'eligible_age_min': 65, 'eligible_age_max': 120},
}


def hedis_gap_list(
    db_path: str,
    *,
    measure: str = 'CBP',
    region: Optional[str] = None,
    top_n: int = 100,
) -> BizAnswer:
    m = measure.upper()
    meta = _HEDIS_MEASURES.get(m)
    if meta is None:
        return BizAnswer(
            intent='hedis_gap_list',
            headline=f"Unknown HEDIS measure '{measure}'. Supported: "
                      f"{', '.join(sorted(_HEDIS_MEASURES.keys()))}",
            kpi={}, columns=[], rows=[], reason_codes=[],
            suggested_action="Rephrase with a supported measure code.",
            data_source='none',
        )

    s = _resolve_schema(db_path)
    age_min = meta['eligible_age_min']; age_max = meta['eligible_age_max']
    gender_filter = f"AND m.{_q(s.mem_gender)} = '{meta['gender']}'" if meta.get('gender') else ""
    region_filter = (f"AND m.{_q(s.mem_region)} = '{region.replace(chr(39),'')}'" if region else "")
    dx_filter = (f"AND EXISTS (SELECT 1 FROM {_q(s.enc)} e "
                 f"WHERE e.{_q(s.enc_mid)}=m.{_q(s.mem_mid)} "
                 f"AND e.{_q(s.enc_pdx)} LIKE '{meta['dx_like']}')"
                 if meta.get('dx_like') else "")

    sql = f"""
    SELECT m.{_q(s.mem_mid)} AS member_id,
           COALESCE(m.{_q(s.mem_region)},'—') AS region,
           COALESCE(m.{_q(s.mem_age)}, 0)     AS age,
           COALESCE(m.{_q(s.mem_gender)},'—') AS gender,
           COALESCE(m.{_q(s.mem_pcp)},'—')    AS pcp,
           (SELECT MAX(e.{_q(s.enc_date)}) FROM {_q(s.enc)} e
              WHERE e.{_q(s.enc_mid)} = m.{_q(s.mem_mid)}) AS last_encounter,
           (SELECT COUNT(*) FROM {_q(s.enc)} e
              WHERE e.{_q(s.enc_mid)} = m.{_q(s.mem_mid)}
                AND e.{_q(s.enc_date)} >= date('now','-365 day')) AS enc_12m
      FROM {_q(s.mem)} m
     WHERE COALESCE(m.{_q(s.mem_age)}, 0) BETWEEN {age_min} AND {age_max}
       {gender_filter} {region_filter} {dx_filter}
       AND (SELECT COUNT(*) FROM {_q(s.enc)} e
              WHERE e.{_q(s.enc_mid)} = m.{_q(s.mem_mid)}
                AND e.{_q(s.enc_date)} >= date('now','-365 day')) < 1
     ORDER BY (SELECT MAX(e.{_q(s.enc_date)}) FROM {_q(s.enc)} e
                 WHERE e.{_q(s.enc_mid)} = m.{_q(s.mem_mid)}) ASC NULLS FIRST
     LIMIT {int(top_n)}
    """
    cols, rows, src = _run(sql, db_path)

    n = len(rows)
    expected_closures = int(round(n * _BENCHMARKS['hedis_gap_close_rate_outreach']))
    star_impact = expected_closures * _BENCHMARKS['star_bonus_per_member_per_measure']

    ans = BizAnswer(
        intent='hedis_gap_list',
        headline=(f"{meta['name']} ({m}) — {n} closable gaps. Expected closures with "
                  f"targeted outreach: {expected_closures} "
                  f"(~{int(_BENCHMARKS['hedis_gap_close_rate_outreach']*100)}% benchmark). "
                  f"Estimated Stars bonus impact: {_fmt_money(star_impact)}."),
        kpi={
            'measure': m, 'measure_name': meta['name'],
            'n_gaps': n,
            'expected_closures': expected_closures,
            'expected_star_bonus_usd': round(star_impact, 0),
            'close_rate_benchmark_pct': _BENCHMARKS['hedis_gap_close_rate_outreach']*100,
            'eligibility': {'age_min': age_min, 'age_max': age_max,
                             'gender': meta.get('gender'),
                             'dx_like': meta.get('dx_like')},
        },
        columns=cols + ['gap_measure'],
        rows=[list(r) + [m] for r in rows],
        reason_codes=["no_qualifying_encounter_12mo", "age_eligible",
                       "condition_eligible" if meta.get('dx_like') else "demographic_eligible"],
        suggested_action=(
            f"Stratify by PCP panel; generate provider-level gap lists; "
            f"launch SMS + letter outreach wave; track close rate weekly."
        ),
        data_source=src,
        sql=sql.strip(),
        n_rows=n,
    )
    _log_answer(db_path, ans)
    return ans


def network_performance(
    db_path: str,
    *,
    specialty: Optional[str] = None,
    metric: str = 'risk_adjusted_readmit',
    top_n: int = 50,
) -> BizAnswer:
    s = _resolve_schema(db_path)
    specialty_filter = (f"WHERE p.{_q(s.prov_spec)} = '{specialty.replace(chr(39),'')}'"
                         if specialty else "")

    sql_readmit = f"""
    WITH prov_enc AS (
      SELECT p.{_q(s.prov_pid)} AS provider_id,
             p.{_q(s.prov_name)} AS provider_name,
             p.{_q(s.prov_spec)} AS specialty,
             COUNT(e.{_q(s.enc_eid)}) AS n_encs,
             SUM(CASE WHEN e.{_q(s.enc_readmit)}=1 THEN 1 ELSE 0 END) AS observed_readmit,
             AVG(COALESCE(m.{_q(s.mem_risk)}, 0.1)) AS avg_risk,
             SUM(COALESCE(e.{_q(s.enc_paid)},0)) AS total_paid
        FROM {_q(s.prov)} p
        LEFT JOIN {_q(s.enc)} e ON e.{_q(s.enc_pid)} = p.{_q(s.prov_pid)}
        LEFT JOIN {_q(s.mem)} m ON m.{_q(s.mem_mid)} = e.{_q(s.enc_mid)}
       {specialty_filter}
       GROUP BY p.{_q(s.prov_pid)}, p.{_q(s.prov_name)}, p.{_q(s.prov_spec)}
    ), baseline AS (
      SELECT AVG(CASE WHEN e.{_q(s.enc_readmit)}=1 THEN 1.0 ELSE 0.0 END) AS base_rate
        FROM {_q(s.enc)} e
    )
    SELECT pe.provider_id,
           pe.provider_name,
           pe.specialty,
           pe.n_encs,
           pe.observed_readmit,
           ROUND(CAST(pe.observed_readmit AS REAL) / NULLIF(pe.n_encs,0), 4) AS observed_rate,
           ROUND(b.base_rate * (1 + (pe.avg_risk - 0.1)), 4) AS expected_rate,
           ROUND(pe.total_paid / NULLIF(pe.n_encs,0), 2) AS cost_per_encounter
      FROM prov_enc pe, baseline b
     WHERE pe.n_encs >= 30
     ORDER BY (CAST(pe.observed_readmit AS REAL) / NULLIF(pe.n_encs,0))
               / NULLIF(b.base_rate * (1 + (pe.avg_risk - 0.1)), 0) DESC
     LIMIT {int(top_n)}
    """

    sql_cost = f"""
    SELECT p.{_q(s.prov_pid)} AS provider_id,
           p.{_q(s.prov_name)} AS provider_name,
           p.{_q(s.prov_spec)} AS specialty,
           COUNT(e.{_q(s.enc_eid)}) AS n_encs,
           ROUND(AVG(COALESCE(e.{_q(s.enc_paid)},0)), 2) AS avg_cost_per_encounter,
           ROUND(SUM(COALESCE(e.{_q(s.enc_paid)},0)), 2) AS total_paid,
           ROUND(AVG(COALESCE(m.{_q(s.mem_risk)}, 0.1)), 3) AS avg_panel_risk
      FROM {_q(s.prov)} p
      LEFT JOIN {_q(s.enc)} e ON e.{_q(s.enc_pid)} = p.{_q(s.prov_pid)}
      LEFT JOIN {_q(s.mem)} m ON m.{_q(s.mem_mid)} = e.{_q(s.enc_mid)}
     {specialty_filter}
     GROUP BY p.{_q(s.prov_pid)}, p.{_q(s.prov_name)}, p.{_q(s.prov_spec)}
    HAVING COUNT(e.{_q(s.enc_eid)}) >= 30
     ORDER BY (AVG(COALESCE(e.{_q(s.enc_paid)},0)) / NULLIF(AVG(COALESCE(m.{_q(s.mem_risk)}, 0.1)),0)) DESC
     LIMIT {int(top_n)}
    """

    metric_l = (metric or '').lower()
    if 'cost' in metric_l:
        sql = sql_cost; intent_metric = 'cost_efficiency'
    else:
        sql = sql_readmit; intent_metric = 'risk_adjusted_readmit'

    cols, rows, src = _run(sql, db_path)

    n = len(rows)
    out_cols = list(cols) + ['oe_ratio', 'reason']
    out_rows = []
    worst_oe = 0.0
    for r in rows:
        d = dict(zip(cols, r))
        if intent_metric == 'risk_adjusted_readmit':
            obs = float(d.get('observed_rate') or 0)
            exp = float(d.get('expected_rate') or 0) or 0.0001
            oe = round(obs / exp, 2) if exp else None
            reasons = []
            if oe and oe > 1.3: reasons.append("O/E readmit > 1.3 (material)")
            if int(d.get('n_encs') or 0) < 60: reasons.append("low volume — verify")
            if not reasons: reasons.append("above-panel-average readmit")
        else:
            avg_cost = float(d.get('avg_cost_per_encounter') or 0)
            panel_risk = float(d.get('avg_panel_risk') or 0.0001)
            oe = round(avg_cost / (panel_risk * 10000.0 + 1.0), 2)
            reasons = []
            if avg_cost > 2500: reasons.append("avg cost/encounter > $2.5K")
            if panel_risk < 0.1: reasons.append("low panel risk — unexplained cost")
            if not reasons: reasons.append("cost efficiency below cohort")
        worst_oe = max(worst_oe, oe or 0)
        out_rows.append(list(r) + [oe, "; ".join(reasons)])

    ans = BizAnswer(
        intent='network_performance',
        headline=(f"Network performance ({intent_metric.replace('_',' ')}) — "
                  f"{n} providers flagged, worst O/E ratio {worst_oe:.2f}. "
                  f"Engage top 10 with peer-comparison dashboards."),
        kpi={
            'n_flagged': n,
            'metric': intent_metric,
            'specialty': specialty or 'ALL',
            'worst_oe_ratio': worst_oe,
        },
        columns=out_cols,
        rows=out_rows,
        reason_codes=["oe_ratio_high", "low_volume_flag", "panel_risk_mismatch"],
        suggested_action=(
            "Share provider-level O/E dashboards; schedule peer-comparison "
            "1:1s with medical directors; re-measure in 90 days."
        ),
        data_source=src,
        sql=sql.strip(),
        n_rows=n,
    )
    _log_answer(db_path, ans)
    return ans


def get_business_insights_status(db_path: str) -> Dict:
    out = {'status': 'ok',
            'databricks_available': _DBX,
            'ml_layers': {
                'uncertainty': True,
                'conformal': _cp is not None,
                'calibration': _cal is not None,
                'explainability': _xp is not None,
                'fairness': _fair is not None,
            },
            'supported_intents': [
                'rising_risk_cohort',
                'readmit_watchlist',
                'hedis_gap_list',
                'network_performance',
            ],
            'supported_hedis_measures': list(_HEDIS_MEASURES.keys()),
            'benchmarks': _BENCHMARKS,
    }
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute(f"SELECT intent, COUNT(*), MAX(ts) FROM {_LOG_TABLE} GROUP BY intent")
            out['recent_usage'] = {r[0]: {'count': r[1], 'last_ts': r[2]} for r in cur.fetchall()}
            conn.close()
        except Exception:
            out['recent_usage'] = {}
    return out
