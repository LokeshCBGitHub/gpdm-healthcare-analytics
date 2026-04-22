from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple


INTENT_RULES: List[Tuple[str, str]] = [
    ("forecast",    r"\b(forecast|predict|projection|next (quarter|month|year)|will .* be|expect(ed)?)\b"),
    ("compare",     r"\b(compare|vs\.?|versus|benchmark|against|ratio|rank(ed|ing)?|top \d+|bottom \d+)\b"),
    ("trend",       r"\b(trend|over time|by (month|week|year)|month-over-month|yoy|growth|change over)\b"),
    ("root_cause",  r"\b(why|driver|driving|explain|cause|reason|root cause|contribut(ed|ing|ion))\b"),
    ("cohort",      r"\b(cohort|segment|break down|split by|by region|by plan|by provider|by diagnosis)\b"),
    ("anomaly",     r"\b(spike|drop|outlier|unusual|anomal(y|ies)|surge|fell|jumped)\b"),
    ("opportunity", r"\b(opportunity|savings|reduce cost|revenue impact|dollar impact|uplift|ROI)\b"),
    ("compliance",  r"\b(HEDIS|star rating|CMS|compliance|regulatory|HIPAA|audit)\b"),
    ("detail",      r"\b(show me|list|who|which|detail|breakdown|drill)\b"),
]

INTENT_ORDER = [name for name, _ in INTENT_RULES]
_INTENT_COMPILED = [(n, re.compile(p, re.IGNORECASE)) for n, p in INTENT_RULES]


@dataclass
class IntentTag:
    primary: str
    secondary: List[str] = field(default_factory=list)
    question: str = ""
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def classify_intent(question: str) -> IntentTag:
    q = (question or "").strip()
    if not q:
        return IntentTag(primary="summary", question=q, confidence=0.1)
    hits: List[str] = []
    for name, pat in _INTENT_COMPILED:
        if pat.search(q):
            hits.append(name)
    if not hits:
        return IntentTag(primary="summary", question=q, confidence=0.2)
    conf = min(0.95, 0.45 + 0.15 * len(hits))
    return IntentTag(primary=hits[0], secondary=hits[1:], question=q, confidence=conf)


PERSONA_CATALOG = {
    "executive":    {"dollar_weight": 1.0, "clinical_weight": 0.2, "ops_weight": 0.3,
                     "label": "Executive / CFO", "lens": "financial and enterprise-level"},
    "clinical":     {"dollar_weight": 0.2, "clinical_weight": 1.0, "ops_weight": 0.4,
                     "label": "Clinical / Care Management", "lens": "member health outcomes"},
    "operations":   {"dollar_weight": 0.4, "clinical_weight": 0.3, "ops_weight": 1.0,
                     "label": "Operations / Network", "lens": "utilization and throughput"},
    "actuarial":    {"dollar_weight": 0.9, "clinical_weight": 0.3, "ops_weight": 0.6,
                     "label": "Actuarial / Risk", "lens": "trend, risk and PMPM"},
    "compliance":   {"dollar_weight": 0.2, "clinical_weight": 0.6, "ops_weight": 0.5,
                     "label": "Quality / Compliance", "lens": "HEDIS, Stars, audit exposure"},
    "analyst":      {"dollar_weight": 0.5, "clinical_weight": 0.5, "ops_weight": 0.5,
                     "label": "Analyst", "lens": "balanced, with data-level drill-downs"},
}

PERSONA_KEYWORDS = {
    "executive":   [r"\brevenue\b", r"\bmargin\b", r"\bebitda\b", r"\bbottom line\b",
                     r"\bboard\b", r"\bCFO\b", r"\bannual\b", r"\bP&L\b"],
    "clinical":    [r"\breadmit", r"\bA1c\b", r"\bdiabet", r"\bHEDIS\b", r"\bgap in care\b",
                     r"\bcare management\b", r"\btransition of care\b", r"\bdiagnos"],
    "operations":  [r"\butilization\b", r"\bauthoriz", r"\bdenial\b", r"\bclaim volume\b",
                     r"\bnetwork\b", r"\bprovider\b", r"\bturnaround\b", r"\bbacklog\b"],
    "actuarial":   [r"\bPMPM\b", r"\bloss ratio\b", r"\btrend\b", r"\brisk adjust",
                     r"\bHCC\b", r"\bMLR\b", r"\breserv"],
    "compliance":  [r"\bHEDIS\b", r"\bstar rating\b", r"\bCMS\b", r"\bHIPAA\b", r"\baudit\b"],
}
_PERSONA_COMPILED = {
    p: [re.compile(pat, re.IGNORECASE) for pat in pats]
    for p, pats in PERSONA_KEYWORDS.items()
}

ROLE_TO_PERSONA = {
    "admin": "executive", "executive": "executive", "cfo": "executive", "ceo": "executive",
    "finance": "executive",
    "care_manager": "clinical", "clinical": "clinical", "physician": "clinical",
    "nurse": "clinical", "medical_director": "clinical",
    "ops": "operations", "operations": "operations", "network": "operations",
    "actuary": "actuarial", "actuarial": "actuarial",
    "quality": "compliance", "compliance": "compliance",
    "analyst": "analyst", "data_analyst": "analyst", "user": "analyst",
}


@dataclass
class Persona:
    key: str
    label: str
    lens: str
    dollar_weight: float
    clinical_weight: float
    ops_weight: float
    evidence: Dict[str, int] = field(default_factory=dict)
    source: str = "default"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def detect_persona(role: Optional[str], recent_questions: Optional[Iterable[str]]) -> Persona:
    role_key = None
    if role:
        r = str(role).strip().lower().replace(" ", "_")
        role_key = ROLE_TO_PERSONA.get(r)

    counts: Dict[str, int] = {p: 0 for p in PERSONA_CATALOG}
    total = 0
    for q in (recent_questions or []):
        if not q:
            continue
        total += 1
        text = str(q)
        for pk, pats in _PERSONA_COMPILED.items():
            for pat in pats:
                if pat.search(text):
                    counts[pk] += 1
                    break

    inferred_key = None
    if total and max(counts.values()) > 0:
        inferred_key = max(counts, key=counts.get)

    if role_key and inferred_key and role_key != inferred_key:
        key, source = role_key, "blended"
    elif role_key:
        key, source = role_key, "role"
    elif inferred_key:
        key, source = inferred_key, "inferred"
    else:
        key, source = "analyst", "default"

    meta = PERSONA_CATALOG[key]
    return Persona(
        key=key,
        label=meta["label"],
        lens=meta["lens"],
        dollar_weight=meta["dollar_weight"],
        clinical_weight=meta["clinical_weight"],
        ops_weight=meta["ops_weight"],
        evidence=counts,
        source=source,
    )


@dataclass
class FollowUp:
    prompt: str
    why: str
    kind: str = "question"
    priority: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_INTENT_FOLLOWUPS: Dict[str, List[FollowUp]] = {
    "summary": [
        FollowUp("What changed most versus last quarter?",
                 "Surfaces the biggest movers so the summary isn't flat.", "trend", 0.7),
        FollowUp("Which KPIs are missing benchmark targets right now?",
                 "Lets the user triage by verdict color.", "benchmark", 0.7),
    ],
    "forecast": [
        FollowUp("What are the main drivers of this forecast?",
                 "Decomposes the forecast into contributing signals.", "root_cause", 0.9),
        FollowUp("How does the 12-month projection compare to the 3-month one?",
                 "Exposes uncertainty scaling — critical for planning.", "forecast", 0.8),
        FollowUp("What does this forecast mean in annual dollars?",
                 "Ties the point estimate to budget impact.", "forecast", 0.7),
    ],
    "compare": [
        FollowUp("Show the trend that led to this gap over the last 12 months.",
                 "A point-in-time compare hides the trajectory.", "trend", 0.8),
        FollowUp("Which segments are driving the difference?",
                 "Helps move from diagnosis to action.", "cohort", 0.8),
    ],
    "trend": [
        FollowUp("Forecast this metric for the next 6 months.",
                 "Trend questions almost always precede forecast ones.", "forecast", 0.9),
        FollowUp("Is this trend above or below industry benchmark?",
                 "Contextualizes the direction.", "benchmark", 0.7),
        FollowUp("Where is the inflection point?",
                 "Finds when the trend shifted.", "anomaly", 0.6),
    ],
    "root_cause": [
        FollowUp("Quantify the dollar impact of the top driver.",
                 "Moves from 'why' to 'so what'.", "opportunity", 0.9),
        FollowUp("Show the top 10 members or providers contributing.",
                 "Enables targeted intervention.", "detail", 0.8),
        FollowUp("Forecast the metric if we intervene on that driver.",
                 "Counterfactual planning.", "forecast", 0.7),
    ],
    "cohort": [
        FollowUp("Rank cohorts by dollar impact.",
                 "Cohort lists are easier to act on when ranked by $.", "opportunity", 0.8),
        FollowUp("Which cohort has the worst trend over 6 months?",
                 "Lets the user prioritize by direction, not level.", "trend", 0.7),
    ],
    "anomaly": [
        FollowUp("Did claims volume spike in the same period?",
                 "Anomalies in cost usually have a volume companion.", "trend", 0.8),
        FollowUp("Which providers or diagnoses drove this spike?",
                 "Move from detection to owner.", "root_cause", 0.9),
    ],
    "opportunity": [
        FollowUp("What is the HIPAA-safe intervention playbook for this?",
                 "Tie dollars to an actual action.", "question", 0.8),
        FollowUp("Rank members by expected savings, top 100.",
                 "Enables outreach prioritization.", "detail", 0.8),
    ],
    "compliance": [
        FollowUp("Which members are below the measure threshold?",
                 "Gap-closing is the next step after a compliance view.", "detail", 0.9),
        FollowUp("What was the measure trend over the last 4 quarters?",
                 "Context for regulator-facing conversations.", "trend", 0.7),
    ],
    "detail": [
        FollowUp("Summarize these rows by region and plan.",
                 "A long list usually triggers a summary request.", "cohort", 0.7),
        FollowUp("Export this list to Excel.",
                 "Common next action on detail views.", "question", 0.6),
    ],
}

_PERSONA_BOOST: Dict[str, Dict[str, float]] = {
    "executive":  {"opportunity": 0.35, "forecast": 0.2, "benchmark": 0.2},
    "clinical":   {"cohort": 0.25, "root_cause": 0.25, "detail": 0.15},
    "operations": {"trend": 0.2, "anomaly": 0.2, "detail": 0.15},
    "actuarial":  {"forecast": 0.3, "trend": 0.25, "benchmark": 0.15},
    "compliance": {"benchmark": 0.3, "detail": 0.25},
    "analyst":    {"detail": 0.2, "cohort": 0.15},
}


def anticipate(intent: IntentTag, persona: Persona,
               question: str = "",
               last_answer_meta: Optional[Dict[str, Any]] = None,
               limit: int = 4,
               data_dir: Optional[str] = None) -> List[FollowUp]:
    base = list(_INTENT_FOLLOWUPS.get(intent.primary, _INTENT_FOLLOWUPS["summary"]))
    for sec in (intent.secondary or []):
        for fu in _INTENT_FOLLOWUPS.get(sec, []):
            if not any(b.prompt == fu.prompt for b in base):
                base.append(FollowUp(**{**asdict(fu), "priority": fu.priority * 0.8}))

    boost = _PERSONA_BOOST.get(persona.key, {})
    for fu in base:
        fu.priority = round(min(1.0, fu.priority + boost.get(fu.kind, 0.0)), 3)

    meta = last_answer_meta or {}
    if meta.get("ci_growth_factor", 0) > 3.0:
        base.append(FollowUp(
            "Why did the uncertainty band grow so much at the long horizon?",
            "The CI widened >3x; usually signals regime change or sparse history.",
            "root_cause", 0.85,
        ))
    if meta.get("dollar_impact_usd", 0) > 1_000_000:
        base.append(FollowUp(
            "Break this dollar opportunity down by intervention type.",
            "Opportunity is >$1M; worth decomposing.", "opportunity", 0.9,
        ))

    if data_dir and question:
        try:
            import grounded_followups as _gfu
            graph = _gfu.get_graph(data_dir)
            if graph is not None:
                for g in graph.suggest(question, k=limit):
                    why = (f"Observed follow-up: {g.support} user(s) asked "
                           f"this next after similar queries "
                           f"(score {g.score:.2f}).")
                    base.append(FollowUp(
                        prompt=g.prompt,
                        why=why,
                        kind="grounded",
                        priority=round(min(1.0, 0.82 + 0.15 * g.score), 3),
                    ))
        except Exception:
            pass

    base.sort(key=lambda f: -f.priority)

    seen, out = set(), []
    for fu in base:
        k = fu.prompt.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(fu)
        if len(out) >= limit:
            break
    return out


def enrich(question: str, role: Optional[str], recent_questions: Optional[Iterable[str]],
           last_answer_meta: Optional[Dict[str, Any]] = None,
           data_dir: Optional[str] = None) -> Dict[str, Any]:
    intent = classify_intent(question)
    persona = detect_persona(role, recent_questions)
    follow_ups = anticipate(intent, persona, question, last_answer_meta,
                            data_dir=data_dir)
    return {
        "intent": intent.to_dict(),
        "persona": persona.to_dict(),
        "follow_ups": [f.to_dict() for f in follow_ups],
        "lens_hint": f"Framed through a {persona.lens} lens.",
    }


def resolve_persona_key(role: Optional[str]) -> str:
    if not role:
        return "analyst"
    r = str(role).strip().lower().replace(" ", "_")
    return ROLE_TO_PERSONA.get(r, "analyst")


__all__ = [
    "IntentTag", "Persona", "FollowUp",
    "classify_intent", "detect_persona", "anticipate", "enrich",
    "resolve_persona_key", "ROLE_TO_PERSONA",
    "PERSONA_CATALOG", "INTENT_ORDER", "_PERSONA_COMPILED",
]
