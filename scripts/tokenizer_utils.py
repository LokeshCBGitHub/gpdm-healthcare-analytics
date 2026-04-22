import re
from typing import Set, List, Dict, Tuple, Optional, NamedTuple


_WORD_RE = re.compile(r'[a-z][a-z0-9_]*', re.IGNORECASE)

_WORD_3PLUS_RE = re.compile(r'\b[a-z][a-z0-9_]{2,}\b', re.IGNORECASE)

STOP_WORDS = frozenset({
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
    'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
    'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each',
    'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    'just', 'because', 'but', 'and', 'or', 'if', 'while', 'what', 'which',
    'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'it', 'its',
    'my', 'me', 'we', 'our', 'you', 'your', 'he', 'she', 'they', 'them',
    'his', 'her', 'their', 'about', 'up',
    'show', 'give', 'tell', 'list', 'get', 'find', 'display', 'many',
    'much', 'total', 'count', 'number', 'average', 'sum', 'per',
})


class SchemaMatch(NamedTuple):
    name: str
    entity_type: str
    source_table: str
    match_pos: int


def tokenize_query(text: str, min_length: int = 0) -> Set[str]:
    if not text:
        return set()
    tokens = set(_WORD_RE.findall(text.lower()))
    if min_length > 0:
        tokens = {t for t in tokens if len(t) >= min_length}
    return tokens


def tokenize_query_list(text: str, min_length: int = 0) -> List[str]:
    if not text:
        return []
    tokens = [t.lower() for t in _WORD_RE.findall(text)]
    if min_length > 0:
        tokens = [t for t in tokens if len(t) >= min_length]
    return tokens


def tokenize_no_stopwords(text: str, min_length: int = 0,
                          extra_stops: Optional[Set[str]] = None) -> Set[str]:
    tokens = tokenize_query(text, min_length)
    stops = STOP_WORDS | (extra_stops or set())
    return tokens - stops


def extract_schema_entities(
    text: str,
    schema_tables: Optional[List[str]] = None,
    schema_columns: Optional[Dict[str, List[str]]] = None,
) -> List[SchemaMatch]:
    if not text:
        return []

    q_lower = text.lower()
    matches = []

    if schema_tables:
        sorted_tables = sorted(schema_tables, key=len, reverse=True)
        for tbl in sorted_tables:
            tl = tbl.lower()
            pos = q_lower.find(tl)
            if pos >= 0:
                matches.append(SchemaMatch(
                    name=tbl,
                    entity_type='table',
                    source_table=tbl,
                    match_pos=pos,
                ))

    if schema_columns:
        all_cols = []
        for tbl, cols in schema_columns.items():
            for col in cols:
                all_cols.append((col, tbl))
        all_cols.sort(key=lambda x: len(x[0]), reverse=True)

        for col_name, tbl_name in all_cols:
            cl = col_name.lower()
            variants = [cl, cl.replace('_', ' ')]
            for variant in variants:
                pos = q_lower.find(variant)
                if pos >= 0:
                    matches.append(SchemaMatch(
                        name=col_name,
                        entity_type='column',
                        source_table=tbl_name,
                        match_pos=pos,
                    ))
                    break

    matches.sort(key=lambda m: m.match_pos)
    return matches


def tokenize_with_schema(
    text: str,
    schema_tables: Optional[List[str]] = None,
    schema_columns: Optional[Dict[str, List[str]]] = None,
    min_length: int = 0,
) -> Tuple[Set[str], List[SchemaMatch]]:
    tokens = tokenize_query(text, min_length)
    entities = extract_schema_entities(text, schema_tables, schema_columns)

    for ent in entities:
        tokens.add(ent.name.lower())

    return tokens, entities


def get_mentioned_tables(
    text: str,
    schema_tables: List[str],
) -> List[Tuple[str, int]]:
    if not text or not schema_tables:
        return []

    q_lower = text.lower()
    found = []
    for tbl in schema_tables:
        if tbl.lower() in q_lower:
            found.append((tbl, len(tbl)))

    found.sort(key=lambda x: x[1], reverse=True)
    return found


def get_mentioned_columns(
    text: str,
    schema_columns: Dict[str, List[str]],
    target_table: Optional[str] = None,
) -> List[SchemaMatch]:
    if target_table:
        schema_columns = {target_table: schema_columns.get(target_table, [])}

    entities = extract_schema_entities(text, schema_tables=None, schema_columns=schema_columns)
    return [e for e in entities if e.entity_type == 'column']
