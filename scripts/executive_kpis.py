from __future__ import annotations

import logging
import math
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

try:
    import cms_reference as _cms
    import live_cache as _lc
    import sparse_smoothing as _ss
    import kpi_cube as _cube
    _PERF_STACK = True
except ImportError:
    _PERF_STACK = False

_log = logging.getLogger(__name__)


_WATCHED_TABLES = (
    'MEMBERS', 'members',
    'ENCOUNTERS', 'encounters',
    'CLAIMS', 'claims',
    'gpdm_member_month_fact',
)


BENCHMARKS: Dict[str, Dict[str, Any]] = {
    'admits_per_1000': {
        'good': 55, 'warn': 70, 'bad': 90,
        'unit': 'per 1,000 MM (annualized)',
        'source': 'Commercial book-of-business median ~60; Medicare ~260',
        'direction': 'lower',
    },
    'er_per_1000': {
        'good': 150, 'warn': 200, 'bad': 260,
        'unit': 'per 1,000 MM (annualized)',
        'source': 'Commercial median ~180',
        'direction': 'lower',
    },
    'readmit_30d_rate': {
        'good': 0.10, 'warn': 0.15, 'bad': 0.20,
        'unit': '%',
        'source': 'CMS national avg ~15.5%',
        'direction': 'lower',
    },
    'pmpm_cost': {
        'good': 350, 'warn': 500, 'bad': 650,
        'unit': '$ PMPM',
        'source': 'Commercial ~$450–$550 PMPM',
        'direction': 'lower',
    },
    'denial_rate': {
        'good': 0.05, 'warn': 0.10, 'bad': 0.15,
        'unit': '%',
        'source': 'Best-in-class < 5%; industry avg ~10%',
        'direction': 'lower',
    },
    'ar_days': {
        'good': 30, 'warn': 45, 'bad': 60,
        'unit': 'days',
        'source': 'HFMA best-practice < 35 days',
        'direction': 'lower',
    },
    'preventive_rate': {
        'good': 0.55, 'warn': 0.40, 'bad': 0.25,
        'unit': '%',
        'source': 'HEDIS AWV/wellness ~50% target',
        'direction': 'higher',
    },
    'high_cost_claimant_share': {
        'good': 0.40, 'warn': 0.55, 'bad': 0.70,
        'unit': '% of spend in top 5% of members',
        'source': 'Payer norm ~50% of spend in top 5% members',
        'direction': 'lower',
    },
    'avg_los': {
        'good': 4.0, 'warn': 5.5, 'bad': 7.0,
        'unit': 'days',
        'source': 'Commercial acute inpatient ~4.5 days',
        'direction': 'lower',
    },
    'utilization_rate': {
        'good': 0.70, 'warn': 0.55, 'bad': 0.40,
        'unit': '%',
        'source': 'Engaged-member rate target ≥ 65%',
        'direction': 'higher',
    },
}

AVG_ADMIT_COST_USD = 13_600
AVG_ER_COST_USD = 1_400
AVG_READMIT_COST_USD = 15_500


def _tables(conn: sqlite3.Connection) -> List[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
    ).fetchall()
    return [r[0] for r in rows]


def _cols(conn: sqlite3.Connection, table: str) -> List[str]:
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return [r[1] for r in rows]
    except sqlite3.Error:
        return []


try:
    from . import column_map as _cmap
except Exception:
    try:
        import column_map as _cmap
    except Exception:
        _cmap = None


def _find_table(conn: sqlite3.Connection, *candidates: str) -> Optional[str]:
    existing = {t.lower(): t for t in _tables(conn)}
    for c in candidates:
        if c.lower() in existing:
            return existing[c.lower()]
    if _cmap is not None and candidates:
        resolved = _cmap.resolve_table(conn, candidates[0])
        if resolved and resolved.lower() in existing:
            return existing[resolved.lower()]
    return None


def _find_col(conn: sqlite3.Connection, table: str,
              *candidates: str) -> Optional[str]:
    if not table:
        return None
    cols = {c.lower(): c for c in _cols(conn, table)}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    if _cmap is not None and candidates:
        for logical_tbl, aliases in _cmap._DEFAULT_TABLE_ALIASES.items():
            if not (table.lower() in [a.lower() for a in aliases]
                    or table.lower() == logical_tbl.lower()):
                continue
            for logical_col in candidates:
                resolved = _cmap.resolve_column(conn, logical_tbl, logical_col, table)
                if resolved and resolved.lower() in cols:
                    return cols[resolved.lower()]
    return None


class _Schema:
    def __init__(self, conn: sqlite3.Connection):
        self.members = _find_table(conn, 'members', 'member')
        self.enc = _find_table(conn, 'encounters', 'encounter', 'visits')
        self.claims = _find_table(conn, 'claims', 'claim')
        self.providers = _find_table(conn, 'providers', 'provider')

        self.mem_id = _find_col(conn, self.members, 'member_id', 'MBR_ID', 'id')
        self.enc_mid = _find_col(conn, self.enc, 'member_id', 'mbr_id', 'MEMBER_ID')
        self.enc_date = _find_col(conn, self.enc,
                                   'admit_date', 'encounter_date', 'service_date',
                                   'visit_date', 'date')
        self.enc_disc = _find_col(conn, self.enc,
                                   'discharge_date', 'disch_date')
        self.enc_type = _find_col(conn, self.enc,
                                   'visit_type', 'encounter_type', 'type')
        self.enc_los = _find_col(conn, self.enc,
                                  'length_of_stay', 'los')
        self.enc_paid = _find_col(conn, self.enc,
                                   'total_paid', 'paid_amount', 'cost', 'allowed')

        self.cl_mid = _find_col(conn, self.claims, 'member_id', 'MEMBER_ID')
        self.cl_date = _find_col(conn, self.claims,
                                  'service_date', 'claim_date', 'submitted_date',
                                  'adjudicated_date')
        self.cl_sub = _find_col(conn, self.claims, 'submitted_date', 'sub_date')
        self.cl_adj = _find_col(conn, self.claims,
                                 'adjudicated_date', 'adj_date', 'paid_date')
        self.cl_status = _find_col(conn, self.claims, 'claim_status', 'status')
        self.cl_paid = _find_col(conn, self.claims, 'paid_amount', 'paid')
        self.cl_billed = _find_col(conn, self.claims,
                                    'billed_amount', 'billed', 'charge_amount')

    def has(self, *attrs: str) -> bool:
        return all(getattr(self, a, None) for a in attrs)


def _scalar(conn: sqlite3.Connection, sql: str,
            params: Tuple = ()) -> Optional[float]:
    try:
        row = conn.execute(sql, params).fetchone()
        if row is None:
            return None
        val = row[0]
        if val is None:
            return None
        return float(val)
    except (sqlite3.Error, ValueError, TypeError) as e:
        _log.debug("exec_kpi scalar failed: %s | sql=%s", e, sql[:120])
        return None


def _verdict(key: str, value: Optional[float]) -> str:
    if value is None:
        return 'unknown'
    b = BENCHMARKS.get(key)
    if not b:
        return 'unknown'
    g, w, bd = b['good'], b['warn'], b['bad']
    if b['direction'] == 'lower':
        if value <= g: return 'good'
        if value <= w: return 'warn'
        return 'bad'
    if value >= g: return 'good'
    if value >= w: return 'warn'
    return 'bad'


def _delta_pct(cur: Optional[float], prev: Optional[float]) -> Optional[float]:
    if cur is None or prev is None or prev == 0:
        return None
    return (cur - prev) / prev * 100.0


def _mk(key: str, label: str, value: Optional[float], unit: str,
        *, trend_pct: Optional[float] = None,
        dollar_impact: Optional[float] = None,
        so_what: str = '', narrative: str = '',
        drill_query: str = '') -> Dict[str, Any]:
    b = BENCHMARKS.get(key, {})
    return {
        'key': key,
        'label': label,
        'value': value,
        'unit': unit,
        'verdict': _verdict(key, value),
        'trend_pct': trend_pct,
        'benchmark': {
            'good': b.get('good'),
            'warn': b.get('warn'),
            'bad': b.get('bad'),
            'direction': b.get('direction'),
            'source': b.get('source'),
        },
        'dollar_impact_usd': dollar_impact,
        'so_what': so_what,
        'narrative': narrative,
        'drill_query': drill_query,
    }


def _kpi_admits_per_1000(conn: sqlite3.Connection,
                          s: _Schema) -> Optional[Dict[str, Any]]:
    if not s.has('enc', 'enc_mid'):
        return None
    type_pred = ''
    if s.enc_type:
        type_pred = f" AND UPPER({s.enc_type}) IN ('INPATIENT','IP','ADMIT') "
    elif s.enc_los:
        type_pred = f" AND {s.enc_los} > 0 "
    date_col = s.enc_date or ''

    def _count(window: str) -> Optional[float]:
        if date_col:
            return _scalar(conn,
                f"SELECT COUNT(*) FROM {s.enc} "
                f"WHERE {date_col} >= date('now', '{window}') {type_pred}")
        return _scalar(conn, f"SELECT COUNT(*) FROM {s.enc} WHERE 1=1 {type_pred}")

    cur = _count('-90 day')
    prev = _count('-180 day')
    if cur is None:
        return None
    prev_only = (prev - cur) if (prev is not None and cur is not None) else None

    members = _scalar(conn, f"SELECT COUNT(*) FROM {s.members}") if s.members else None
    if not members or members <= 0:
        return None
    ann_rate = (cur / (members * 3)) * 12 * 1000 if cur else 0
    prev_rate = (prev_only / (members * 3)) * 12 * 1000 if prev_only else None

    trend = _delta_pct(ann_rate, prev_rate)
    dlr = None
    b = BENCHMARKS['admits_per_1000']
    if ann_rate > b['good']:
        excess_admits = (ann_rate - b['good']) / 1000 * members
        dlr = excess_admits * AVG_ADMIT_COST_USD
    return _mk(
        'admits_per_1000', 'Admits per 1,000 members (ann.)',
        round(ann_rate, 1), 'per 1K',
        trend_pct=round(trend, 1) if trend is not None else None,
        dollar_impact=round(dlr) if dlr else None,
        so_what=('Every 1 admit/1K avoided ≈ '
                 f'${AVG_ADMIT_COST_USD:,} saved. Target a Transitions-of-Care '
                 'program on the top 5% risk tier to bend this down.'),
        narrative=(f"Last 90 days: {int(cur)} admits across ~{int(members)} members; "
                   f"annualized at {ann_rate:.1f} per 1,000. Benchmark good ≤ {b['good']}."),
        drill_query='show me top diagnoses driving admissions last 90 days',
    )


def _kpi_er_per_1000(conn: sqlite3.Connection,
                      s: _Schema) -> Optional[Dict[str, Any]]:
    if not (s.enc and s.enc_type):
        return None
    date_col = s.enc_date or ''
    pred = f"UPPER({s.enc_type}) IN ('EMERGENCY','ED','ER')"

    def _count(window: str) -> Optional[float]:
        if date_col:
            return _scalar(conn,
                f"SELECT COUNT(*) FROM {s.enc} "
                f"WHERE {pred} AND {date_col} >= date('now', '{window}')")
        return _scalar(conn, f"SELECT COUNT(*) FROM {s.enc} WHERE {pred}")

    cur = _count('-90 day')
    prev = _count('-180 day')
    members = _scalar(conn, f"SELECT COUNT(*) FROM {s.members}") if s.members else None
    if cur is None or not members:
        return None
    ann_rate = (cur / (members * 3)) * 12 * 1000
    prev_only = (prev - cur) if (prev is not None and cur is not None) else None
    prev_rate = (prev_only / (members * 3)) * 12 * 1000 if prev_only else None
    trend = _delta_pct(ann_rate, prev_rate)

    b = BENCHMARKS['er_per_1000']
    dlr = None
    if ann_rate > b['good']:
        excess = (ann_rate - b['good']) / 1000 * members
        dlr = excess * AVG_ER_COST_USD
    return _mk(
        'er_per_1000', 'ER visits per 1,000 (ann.)',
        round(ann_rate, 1), 'per 1K',
        trend_pct=round(trend, 1) if trend is not None else None,
        dollar_impact=round(dlr) if dlr else None,
        so_what=('Nurse triage + same-day PCP slots typically cut avoidable ER by '
                 '20–30%. Start with top ER-utilizer cohort.'),
        narrative=(f"{int(cur)} ER visits in last 90 days → {ann_rate:.0f}/1K annualized. "
                   f"Benchmark good ≤ {b['good']}/1K."),
        drill_query='show top 10 ER utilizers last 90 days',
    )


def _kpi_readmit_30d(conn: sqlite3.Connection,
                      s: _Schema) -> Optional[Dict[str, Any]]:
    if not s.has('enc', 'enc_mid', 'enc_disc') or not s.enc_date:
        return None
    sql = (
        f"SELECT "
        f"  SUM(CASE WHEN readmit_days BETWEEN 0 AND 30 THEN 1 ELSE 0 END)*1.0 / "
        f"  NULLIF(COUNT(*), 0) AS rate "
        f"FROM ("
        f"  SELECT e1.{s.enc_mid} as m, "
        f"  (SELECT MIN(julianday(e2.{s.enc_date}) - julianday(e1.{s.enc_disc})) "
        f"    FROM {s.enc} e2 "
        f"    WHERE e2.{s.enc_mid} = e1.{s.enc_mid} "
        f"      AND julianday(e2.{s.enc_date}) > julianday(e1.{s.enc_disc})) AS readmit_days "
        f"  FROM {s.enc} e1 "
        f"  WHERE e1.{s.enc_disc} IS NOT NULL "
        f"    AND e1.{s.enc_disc} >= date('now','-180 day')"
        f") t"
    )
    val = _scalar(conn, sql)
    if val is None:
        return None
    members = _scalar(conn, f"SELECT COUNT(DISTINCT {s.enc_mid}) FROM {s.enc}") or 0
    est_readmits = val * (_scalar(conn, f"SELECT COUNT(*) FROM {s.enc}") or 0)
    b = BENCHMARKS['readmit_30d_rate']
    dlr = None
    if val > b['good']:
        excess = est_readmits * ((val - b['good']) / max(val, 1e-9))
        dlr = excess * AVG_READMIT_COST_USD
    return _mk(
        'readmit_30d_rate', '30-day readmission rate',
        round(val * 100, 2), '%',
        dollar_impact=round(dlr) if dlr else None,
        so_what=('TCM (Transitional Care Management) calls within 48h post-discharge '
                 'typically reduce readmits 18%. Target CHF/COPD/pneumonia first.'),
        narrative=(f"Across recent discharges, 30-day readmit rate is {val*100:.2f}%. "
                   f"CMS national benchmark ~15.5%; good ≤ {b['good']*100:.0f}%."),
        drill_query='show 30-day readmission cohort',
    )


def _kpi_pmpm(conn: sqlite3.Connection,
              s: _Schema) -> Optional[Dict[str, Any]]:
    if not (s.claims and s.cl_paid):
        return None
    members = _scalar(conn, f"SELECT COUNT(*) FROM {s.members}") if s.members else None
    if not members:
        return None
    date_col = s.cl_date or s.cl_adj or s.cl_sub
    if not date_col:
        return None
    cur = _scalar(conn,
        f"SELECT SUM(CAST({s.cl_paid} AS FLOAT)) FROM {s.claims} "
        f"WHERE {date_col} >= date('now','-90 day')")
    prev = _scalar(conn,
        f"SELECT SUM(CAST({s.cl_paid} AS FLOAT)) FROM {s.claims} "
        f"WHERE {date_col} >= date('now','-180 day') "
        f"AND {date_col} < date('now','-90 day')")
    if cur is None:
        return None
    pmpm = cur / (members * 3)
    pmpm_prev = (prev / (members * 3)) if prev else None
    trend = _delta_pct(pmpm, pmpm_prev)
    b = BENCHMARKS['pmpm_cost']
    dlr = None
    if pmpm > b['good']:
        dlr = (pmpm - b['good']) * members * 12
    return _mk(
        'pmpm_cost', 'Medical cost PMPM',
        round(pmpm, 2), '$',
        trend_pct=round(trend, 1) if trend is not None else None,
        dollar_impact=round(dlr) if dlr else None,
        so_what=('PMPM is the single best composite cost signal. Decompose by '
                 'region, LOB, and place-of-service to find your lever.'),
        narrative=(f"Last 90-day paid spend ${cur:,.0f} / {int(members)} members × 3 = "
                   f"${pmpm:,.2f} PMPM. Benchmark ~${b['good']}–{b['warn']}."),
        drill_query='break down PMPM by region and LOB',
    )


def _kpi_denial_rate(conn: sqlite3.Connection,
                      s: _Schema) -> Optional[Dict[str, Any]]:
    if not (s.claims and s.cl_status):
        return None
    rate = _scalar(conn,
        f"SELECT SUM(CASE WHEN UPPER({s.cl_status}) IN ('DENIED','REJECTED') "
        f"THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0) FROM {s.claims}")
    if rate is None:
        return None
    denied_total = _scalar(conn,
        f"SELECT SUM(CAST({s.cl_paid} AS FLOAT)) FROM {s.claims} "
        f"WHERE UPPER({s.cl_status}) IN ('DENIED','REJECTED')") if s.cl_paid else None
    b = BENCHMARKS['denial_rate']
    dlr = None
    if rate > b['good'] and denied_total:
        dlr = denied_total * 0.5
    return _mk(
        'denial_rate', 'Claim denial rate',
        round(rate * 100, 2), '%',
        dollar_impact=round(dlr) if dlr else None,
        so_what=('Top 3 denial reasons usually drive 60% of dollars — work edits, '
                 'not individual claims. Target first-pass yield ≥ 95%.'),
        narrative=(f"{rate*100:.2f}% of claims denied. Industry best-in-class ≤ "
                   f"{b['good']*100:.0f}%."),
        drill_query='what are top denial reasons by dollars',
    )


def _kpi_ar_days(conn: sqlite3.Connection,
                  s: _Schema) -> Optional[Dict[str, Any]]:
    if not (s.claims and s.cl_sub and s.cl_adj):
        return None
    val = _scalar(conn,
        f"SELECT AVG(julianday({s.cl_adj}) - julianday({s.cl_sub})) "
        f"FROM {s.claims} WHERE {s.cl_adj} IS NOT NULL AND {s.cl_sub} IS NOT NULL")
    if val is None:
        return None
    return _mk(
        'ar_days', 'AR days (submit → adjudicate)',
        round(val, 1), 'days',
        so_what=('Every day off AR ≈ 0.25% cash flow improvement. Auto-adjudicate '
                 'clean claims and hold human review for exceptions only.'),
        narrative=(f"Average {val:.1f} days from submit to adjudication. "
                   f"HFMA best-practice < 35 days."),
        drill_query='show aging claims > 45 days',
    )


def _kpi_preventive_rate(conn: sqlite3.Connection,
                          s: _Schema) -> Optional[Dict[str, Any]]:
    if not (s.enc and s.enc_type):
        return None
    rate = _scalar(conn,
        f"SELECT SUM(CASE WHEN UPPER({s.enc_type}) IN ('PREVENTIVE','WELLNESS','AWV') "
        f"THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0) FROM {s.enc} "
        f"WHERE {s.enc_date} >= date('now','-365 day')"
        if s.enc_date else
        f"SELECT SUM(CASE WHEN UPPER({s.enc_type}) IN ('PREVENTIVE','WELLNESS','AWV') "
        f"THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0) FROM {s.enc}")
    if rate is None:
        return None
    return _mk(
        'preventive_rate', 'Preventive visit share (trailing 12 mo)',
        round(rate * 100, 1), '%',
        so_what=('Closing wellness-visit gaps drives HEDIS close-rate ~22%. Push '
                 'outreach to members with 0 visits in last 18 months.'),
        narrative=(f"{rate*100:.1f}% of visits are preventive. HEDIS target ≥ 50%."),
        drill_query='list members due for annual wellness visit',
    )


def _kpi_high_cost_share(conn: sqlite3.Connection,
                          s: _Schema) -> Optional[Dict[str, Any]]:
    if not (s.claims and s.cl_paid and s.cl_mid):
        return None
    sql = (
        f"WITH per_mem AS ("
        f"  SELECT {s.cl_mid} as mid, "
        f"         SUM(CAST({s.cl_paid} AS FLOAT)) AS spend "
        f"  FROM {s.claims} GROUP BY {s.cl_mid}"
        f"), "
        f"ranked AS ("
        f"  SELECT mid, spend, "
        f"    NTILE(20) OVER (ORDER BY spend DESC) AS bucket "
        f"  FROM per_mem"
        f") "
        f"SELECT SUM(CASE WHEN bucket = 1 THEN spend ELSE 0 END) * 1.0 / "
        f"       NULLIF(SUM(spend), 0) FROM ranked"
    )
    val = _scalar(conn, sql)
    if val is None:
        return None
    return _mk(
        'high_cost_claimant_share', 'Top 5% members — share of spend',
        round(val * 100, 1), '%',
        so_what=('Concentrated spend = concentrated opportunity. Case-manage the '
                 'top 5% with multi-disciplinary care teams.'),
        narrative=(f"Top 5% of members drive {val*100:.1f}% of total paid. "
                   f"Typical payer norm ~50%."),
        drill_query='show rising risk members with ≥2 chronic conditions',
    )


def _kpi_avg_los(conn: sqlite3.Connection,
                 s: _Schema) -> Optional[Dict[str, Any]]:
    if not (s.enc and s.enc_los):
        return None
    val = _scalar(conn,
        f"SELECT AVG(CAST({s.enc_los} AS FLOAT)) FROM {s.enc} "
        f"WHERE {s.enc_los} IS NOT NULL AND CAST({s.enc_los} AS FLOAT) > 0")
    if val is None:
        return None
    return _mk(
        'avg_los', 'Average length of stay (inpatient)',
        round(val, 2), 'days',
        so_what=('LOS reduction of 0.5 day ≈ 11% bed-day savings. Focus on '
                 'discharge planning and SNF/home-health pathways.'),
        narrative=(f"Average inpatient LOS = {val:.2f} days. "
                   f"Commercial benchmark ~4.5 days."),
        drill_query='which DRGs have longest LOS',
    )


def _kpi_utilization(conn: sqlite3.Connection,
                     s: _Schema) -> Optional[Dict[str, Any]]:
    if not (s.claims and s.cl_mid and s.members):
        return None
    val = _scalar(conn,
        f"SELECT COUNT(DISTINCT {s.cl_mid}) * 1.0 / "
        f"(SELECT COUNT(*) FROM {s.members}) FROM {s.claims}")
    if val is None:
        return None
    return _mk(
        'utilization_rate', 'Active member rate',
        round(val * 100, 1), '%',
        so_what=('Low active rate may signal disengaged members or data lag. '
                 'Check LOB-level engagement and outreach calls.'),
        narrative=(f"{val*100:.1f}% of members had at least one claim in the data. "
                   f"Healthy engagement ≥ 65%."),
        drill_query='members with zero claims in last 12 months',
    )


def _cube_has_data(conn: sqlite3.Connection) -> bool:
    try:
        r = conn.execute(
            "SELECT COUNT(*) FROM gpdm_member_month_fact LIMIT 1"
        ).fetchone()
        return bool(r and r[0])
    except sqlite3.Error:
        return False


def _kpis_from_cube(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    kpis: List[Dict[str, Any]] = []

    ym_rows = conn.execute(
        "SELECT DISTINCT year_month FROM gpdm_member_month_fact "
        "ORDER BY year_month DESC LIMIT 6"
    ).fetchall()
    if len(ym_rows) < 1:
        return kpis
    recent_yms = [r[0] for r in ym_rows[:3]]
    prior_yms = [r[0] for r in ym_rows[3:6]]
    placeholders_r = ','.join('?' * len(recent_yms))
    placeholders_p = ','.join('?' * len(prior_yms)) if prior_yms else "''"

    def _window(yms, expr):
        if not yms:
            return None
        ph = ','.join('?' * len(yms))
        row = conn.execute(
            f"SELECT {expr} FROM gpdm_member_month_fact "
            f"WHERE year_month IN ({ph})", tuple(yms)
        ).fetchone()
        return row[0] if row else None

    members_active = conn.execute(
        "SELECT COUNT(DISTINCT member_id) FROM gpdm_member_month_fact "
        f"WHERE year_month IN ({placeholders_r})", tuple(recent_yms)
    ).fetchone()[0] or 0

    admits_r = _window(recent_yms, "SUM(admits)") or 0
    admits_p = _window(prior_yms, "SUM(admits)") or 0
    er_r = _window(recent_yms, "SUM(er_visits)") or 0
    er_p = _window(prior_yms, "SUM(er_visits)") or 0
    paid_r = _window(recent_yms, "SUM(paid_usd)") or 0.0
    paid_p = _window(prior_yms, "SUM(paid_usd)") or 0.0
    billed_r = _window(recent_yms, "SUM(billed_usd)") or 0.0
    denials_r = _window(recent_yms, "SUM(denials)") or 0
    prev_r = _window(recent_yms, "SUM(preventive)") or 0
    total_r = _window(recent_yms, "SUM(total_visits)") or 0
    los_total = conn.execute(
        "SELECT SUM(los_days) FROM gpdm_member_month_fact "
        f"WHERE year_month IN ({placeholders_r})", tuple(recent_yms)
    ).fetchone()[0] or 0.0

    def per_1k(x):
        if not members_active or not x:
            return 0.0
        return x / (members_active * 3) * 12 * 1000

    admits_1k = per_1k(admits_r)
    admits_1k_prev = per_1k(admits_p)
    trend_admits = (_delta_pct(admits_1k, admits_1k_prev)
                     if admits_1k_prev else None)
    b_adm = BENCHMARKS['admits_per_1000']
    dlr = None
    if admits_1k > b_adm['good']:
        excess = (admits_1k - b_adm['good']) / 1000 * members_active
        dlr = excess * AVG_ADMIT_COST_USD
    kpis.append(_mk(
        'admits_per_1000', 'Admits per 1,000 members (ann.)',
        round(admits_1k, 1), 'per 1K',
        trend_pct=round(trend_admits, 1) if trend_admits is not None else None,
        dollar_impact=round(dlr) if dlr else None,
        so_what=('Transitions-of-Care at the top 5% risk tier typically '
                 'bends this 10–15%.'),
        narrative=(f"Last 3 mo: {int(admits_r)} admits / {members_active:,} "
                   f"active members → {admits_1k:.1f}/1K annualized. "
                   f"Good ≤ {b_adm['good']}."),
        drill_query='show me top diagnoses driving admissions last 90 days',
    ))

    er_1k = per_1k(er_r)
    er_1k_prev = per_1k(er_p)
    trend_er = _delta_pct(er_1k, er_1k_prev) if er_1k_prev else None
    b_er = BENCHMARKS['er_per_1000']
    dlr = None
    if er_1k > b_er['good']:
        excess = (er_1k - b_er['good']) / 1000 * members_active
        dlr = excess * AVG_ER_COST_USD
    kpis.append(_mk(
        'er_per_1000', 'ER visits per 1,000 (ann.)',
        round(er_1k, 1), 'per 1K',
        trend_pct=round(trend_er, 1) if trend_er is not None else None,
        dollar_impact=round(dlr) if dlr else None,
        so_what='Nurse triage + same-day PCP slots cut avoidable ER 20–30%.',
        narrative=(f"{int(er_r)} ER visits last 3 mo → {er_1k:.0f}/1K. "
                   f"Good ≤ {b_er['good']}."),
        drill_query='show top 10 ER utilizers last 90 days',
    ))

    if members_active and paid_r:
        pmpm = paid_r / (members_active * 3)
        pmpm_p = (paid_p / (members_active * 3)) if paid_p else None
        trend_pmpm = _delta_pct(pmpm, pmpm_p) if pmpm_p else None
        b_pm = BENCHMARKS['pmpm_cost']
        dlr = None
        if pmpm > b_pm['good']:
            dlr = (pmpm - b_pm['good']) * members_active * 12
        kpis.append(_mk(
            'pmpm_cost', 'Medical cost PMPM',
            round(pmpm, 2), '$',
            trend_pct=round(trend_pmpm, 1) if trend_pmpm is not None else None,
            dollar_impact=round(dlr) if dlr else None,
            so_what='Decompose by region, LOB, and place-of-service to find '
                     'the concentration.',
            narrative=(f"${paid_r:,.0f} paid / {members_active:,} × 3 MM = "
                       f"${pmpm:,.2f} PMPM. Good ≤ ${b_pm['good']}."),
            drill_query='break down PMPM by region and LOB',
        ))

    if total_r and denials_r is not None and paid_r:
        rate = denials_r / max(total_r, 1)
        b_d = BENCHMARKS['denial_rate']
        rec_est = paid_r * (rate / max(rate, 1e-9)) * 0.1
        kpis.append(_mk(
            'denial_rate', 'Claim denial rate',
            round(rate * 100, 2), '%',
            dollar_impact=round(rec_est) if rate > b_d['good'] else None,
            so_what='Top 3 denial reasons drive 60% of $. Work edits, not '
                     'individual claims.',
            narrative=(f"{denials_r:,} denials / {total_r:,} encounters last "
                       f"3 mo = {rate*100:.2f}%. Best-in-class ≤ "
                       f"{b_d['good']*100:.0f}%."),
            drill_query='top denial reasons by dollars',
        ))

    if total_r:
        prev_rate = prev_r / total_r
        kpis.append(_mk(
            'preventive_rate', 'Preventive visit share (trailing 3 mo)',
            round(prev_rate * 100, 1), '%',
            so_what='Closing wellness-visit gaps drives HEDIS close-rate '
                     '~22%. Outreach members with 0 visits in 18 mo.',
            narrative=(f"{prev_r:,} preventive / {total_r:,} total. HEDIS "
                       f"target ≥ 50%."),
            drill_query='list members due for annual wellness visit',
        ))

    admits_total_for_los = (_window(recent_yms, "SUM(admits)") or 0)
    if admits_total_for_los and los_total:
        avg_los = los_total / admits_total_for_los
        kpis.append(_mk(
            'avg_los', 'Average length of stay (inpatient)',
            round(avg_los, 2), 'days',
            so_what='LOS -0.5d ≈ 11% bed-day savings. Focus on discharge '
                     'planning and SNF/home-health pathways.',
            narrative=(f"{los_total:,.0f} bed-days / {admits_total_for_los:,} "
                       f"admits = {avg_los:.2f} d. Commercial ~4.5 d."),
            drill_query='which DRGs have longest LOS',
        ))

    hcc_row = conn.execute(
        "SELECT member_id, SUM(paid_usd) sp "
        "FROM gpdm_member_month_fact "
        f"WHERE year_month IN ({placeholders_r}) "
        f"GROUP BY member_id HAVING sp > 0 "
        "ORDER BY sp DESC"
        , tuple(recent_yms)
    ).fetchall()
    if hcc_row:
        total = sum(r[1] for r in hcc_row)
        top5n = max(1, int(len(hcc_row) * 0.05))
        top5_spend = sum(r[1] for r in hcc_row[:top5n])
        share = top5_spend / total if total else 0.0
        kpis.append(_mk(
            'high_cost_claimant_share', 'Top 5% members — share of spend',
            round(share * 100, 1), '%',
            so_what='Concentrated spend = concentrated opportunity. Case-'
                     'manage the top 5% with multi-disciplinary teams.',
            narrative=(f"Top {top5n:,} of {len(hcc_row):,} members drive "
                       f"${top5_spend:,.0f} / ${total:,.0f} = "
                       f"{share*100:.1f}% of spend."),
            drill_query='show rising risk members with ≥2 chronic conditions',
        ))

    if members_active:
        total_members = conn.execute(
            "SELECT COUNT(DISTINCT member_id) FROM gpdm_member_month_fact"
        ).fetchone()[0] or members_active
        util = members_active / max(total_members, 1)
        kpis.append(_mk(
            'utilization_rate', 'Active member rate (last 3 mo)',
            round(util * 100, 1), '%',
            so_what='Low active rate may signal disengaged members or data '
                     'lag. Check LOB-level engagement.',
            narrative=(f"{members_active:,} active / {total_members:,} total "
                       f"members in cube. Healthy ≥ 65%."),
            drill_query='members with zero claims in last 12 months',
        ))

    return kpis


def compute_executive_kpis(db_path: str) -> Dict[str, Any]:
    if _PERF_STACK:
        return _lc.CACHE.get_or_compute(
            namespace='exec_kpis',
            params=(db_path,),
            compute=lambda: _compute_executive_kpis_impl(db_path),
            db_path=db_path,
            tables=list(_WATCHED_TABLES),
            ttl=900.0,
        )
    return _compute_executive_kpis_impl(db_path)


def _compute_executive_kpis_impl(db_path: str) -> Dict[str, Any]:
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error as e:
        return {'error': f'Could not open DB: {e}', 'kpis': [], 'categories': {}}

    try:
        schema = _Schema(conn)
        source = 'raw'
        kpis: List[Dict[str, Any]] = []
        if _PERF_STACK and _cube_has_data(conn):
            try:
                kpis = _kpis_from_cube(conn)
                source = 'cube'
            except Exception as e:
                _log.warning("cube KPI path failed: %s", e); kpis = []

        if not kpis:
            builders = [
                _kpi_admits_per_1000,
                _kpi_er_per_1000,
                _kpi_readmit_30d,
                _kpi_pmpm,
                _kpi_denial_rate,
                _kpi_ar_days,
                _kpi_preventive_rate,
                _kpi_high_cost_share,
                _kpi_avg_los,
                _kpi_utilization,
            ]
            for fn in builders:
                try:
                    item = fn(conn, schema)
                    if item:
                        kpis.append(item)
                except Exception as e:
                    _log.warning("KPI %s failed: %s", fn.__name__, e)

        if not any(k['key'] == 'readmit_30d_rate' for k in kpis):
            try:
                r = _kpi_readmit_30d(conn, schema)
                if r: kpis.append(r)
            except Exception as e:
                _log.warning("readmit KPI failed: %s", e)

        if not any(k['key'] == 'ar_days' for k in kpis):
            try:
                r = _kpi_ar_days(conn, schema)
                if r: kpis.append(r)
            except Exception as e:
                _log.warning("AR-days KPI failed: %s", e)

        category_map = {
            'Utilization': ['admits_per_1000', 'er_per_1000',
                             'readmit_30d_rate', 'avg_los'],
            'Financial':    ['pmpm_cost', 'denial_rate',
                             'ar_days', 'high_cost_claimant_share'],
            'Engagement':   ['preventive_rate', 'utilization_rate'],
        }
        by_cat: Dict[str, List[Dict[str, Any]]] = {c: [] for c in category_map}
        for k in kpis:
            for cat, keys in category_map.items():
                if k['key'] in keys:
                    by_cat[cat].append(k); break

        total_impact = sum(k['dollar_impact_usd'] or 0 for k in kpis)

        return {
            'generated_at': _now_iso(),
            'schema_detected': {
                'members': schema.members, 'encounters': schema.enc,
                'claims': schema.claims, 'providers': schema.providers,
            },
            'kpi_count': len(kpis),
            'total_annualized_opportunity_usd': round(total_impact),
            'categories': by_cat,
            'kpis': kpis,
            'hipaa': 'All computed on-premise; no PHI transmitted.',
        }
    finally:
        conn.close()


def compute_most_asked_kpis(tracker, db_path: Optional[str] = None,
                             limit: int = 8,
                             persona_key: Optional[str] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        'generated_at': _now_iso(),
        'most_asked': [],
        'trending': [],
        'persona_key': persona_key,
    }
    try:
        if persona_key and hasattr(tracker, 'get_most_asked_for_persona'):
            out['most_asked'] = tracker.get_most_asked_for_persona(
                persona_key, limit=limit) or []
        else:
            out['most_asked'] = tracker.get_most_asked(limit=limit) or []
    except Exception as e:
        out['most_asked_error'] = str(e)
    try:
        if persona_key and hasattr(tracker, 'get_trending_for_persona'):
            out['trending'] = tracker.get_trending_for_persona(
                persona_key, limit=limit) or []
        else:
            out['trending'] = tracker.get_trending(limit=limit) or []
    except Exception as e:
        out['trending_error'] = str(e)
    for bucket in ('most_asked', 'trending'):
        for row in out[bucket]:
            q = (row.get('question') or '').strip()
            row['kpi_label'] = _kpi_name_from_question(q)
    matched = sum(1 for r in out['most_asked'] if r.get('match'))
    out['persona_match_count'] = matched
    return out


def _kpi_name_from_question(q: str) -> str:
    q = (q or '').lower().strip()
    if not q:
        return 'KPI'
    for lead in ('show me ', 'show ', 'how many ', 'what is the ',
                 'what is ', 'tell me ', 'give me ', 'list '):
        if q.startswith(lead):
            q = q[len(lead):]; break
    q = q.rstrip(' ?.!')
    return q[:1].upper() + q[1:]


def _now_iso() -> str:
    from datetime import datetime
    return datetime.now().isoformat(timespec='seconds')
