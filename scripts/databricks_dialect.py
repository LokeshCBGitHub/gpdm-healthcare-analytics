import re
import logging
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger('gpdm.dialect')


class DatabricksDialect:

    _QUOTED_STRING_PATTERN = re.compile(r"'(?:''|[^'])*'")

    def __init__(self, catalog: str = 'healthcare', schema: str = 'production'):
        self.catalog = catalog
        self.schema = schema
        logger.debug(f"DatabricksDialect initialized: {catalog}.{schema}")

    def translate(self, sqlite_sql: str) -> str:
        if not sqlite_sql or not isinstance(sqlite_sql, str):
            logger.warning(f"Invalid SQL input: {type(sqlite_sql)}")
            return sqlite_sql

        logger.debug(f"Translating SQLite SQL (length={len(sqlite_sql)})")
        sql = sqlite_sql.strip()

        sql = self._translate_date_functions(sql)
        sql = self._translate_string_functions(sql)
        sql = self._translate_table_references(sql)
        sql = self._translate_type_casts(sql)
        sql = self._translate_limit_offset(sql)
        sql = self._translate_pragma(sql)
        sql = self._translate_misc(sql)

        logger.debug(f"Translation complete")
        return sql

    def translate_schema_discovery(self, query_type: str, table: str = None, column: str = None, limit: int = 100) -> str:
        query_type = query_type.lower().strip()

        if query_type == 'list_tables':
            return f"SHOW TABLES IN {self.catalog}.{self.schema}"

        elif query_type == 'describe_table':
            if not table:
                raise ValueError("describe_table requires 'table' parameter")
            return f"DESCRIBE TABLE {self.catalog}.{self.schema}.{table}"

        elif query_type == 'sample_values':
            if not table or not column:
                raise ValueError("sample_values requires 'table' and 'column' parameters")
            return f"SELECT DISTINCT {column} FROM {self.catalog}.{self.schema}.{table} LIMIT {limit}"

        elif query_type == 'count_distinct':
            if not table or not column:
                raise ValueError("count_distinct requires 'table' and 'column' parameters")
            return f"SELECT COUNT(DISTINCT {column}) FROM {self.catalog}.{self.schema}.{table}"

        elif query_type == 'row_count':
            if not table:
                raise ValueError("row_count requires 'table' parameter")
            return f"SELECT COUNT(*) FROM {self.catalog}.{self.schema}.{table}"

        else:
            raise ValueError(f"Unknown query_type: {query_type}")


    def _protect_quoted_strings(self, sql: str) -> Tuple[str, Dict[str, str]]:
        replacements = {}
        counter = 0

        def replace_quoted(match):
            nonlocal counter
            placeholder = f"__QUOTED_{counter}__"
            replacements[placeholder] = match.group(0)
            counter += 1
            return placeholder

        modified_sql = self._QUOTED_STRING_PATTERN.sub(replace_quoted, sql)
        return modified_sql, replacements

    def _restore_quoted_strings(self, sql: str, replacements: Dict[str, str]) -> str:
        for placeholder, original in replacements.items():
            sql = sql.replace(placeholder, original)
        return sql

    def _translate_date_functions(self, sql: str) -> str:

        sql = re.sub(
            r"julianday\(\s*'now'\s*\)\s*-\s*julianday\(\s*([^)]+)\s*\)",
            r"DATEDIFF(CURRENT_DATE(), \1)",
            sql,
            flags=re.IGNORECASE
        )
        sql = re.sub(
            r"julianday\(\s*([^)]+)\s*\)\s*-\s*julianday\(\s*([^)]+)\s*\)",
            r"DATEDIFF(\1, \2)",
            sql,
            flags=re.IGNORECASE
        )

        sql = re.sub(
            r"date\(\s*'now'\s*,\s*'-(\d+)\s+days?'\s*\)",
            r"DATE_SUB(CURRENT_DATE(), \1)",
            sql,
            flags=re.IGNORECASE
        )
        sql = re.sub(
            r"date\(\s*'now'\s*,\s*'\+(\d+)\s+days?'\s*\)",
            r"DATE_ADD(CURRENT_DATE(), \1)",
            sql,
            flags=re.IGNORECASE
        )
        sql = re.sub(
            r"date\(\s*'now'\s*,\s*'-(\d+)\s+months?'\s*\)",
            r"ADD_MONTHS(CURRENT_DATE(), -\1)",
            sql,
            flags=re.IGNORECASE
        )
        sql = re.sub(
            r"date\(\s*'now'\s*,\s*'\+(\d+)\s+months?'\s*\)",
            r"ADD_MONTHS(CURRENT_DATE(), \1)",
            sql,
            flags=re.IGNORECASE
        )
        sql = re.sub(
            r"date\(\s*'now'\s*,\s*'start of month'\s*\)",
            r"TRUNC(CURRENT_DATE(), 'MM')",
            sql,
            flags=re.IGNORECASE
        )
        sql = re.sub(
            r"date\(\s*'now'\s*,\s*'start of year'\s*\)",
            r"TRUNC(CURRENT_DATE(), 'YEAR')",
            sql,
            flags=re.IGNORECASE
        )

        sql = re.sub(
            r"date\(\s*'now'\s*\)",
            r"CURRENT_DATE()",
            sql,
            flags=re.IGNORECASE
        )

        sql = re.sub(
            r"strftime\(\s*'%Y-%m-%d'\s*,\s*([^)]+)\)",
            r"DATE_FORMAT(\1, 'yyyy-MM-dd')",
            sql,
            flags=re.IGNORECASE
        )
        sql = re.sub(
            r"strftime\(\s*'%Y-%m'\s*,\s*([^)]+)\)",
            r"DATE_FORMAT(\1, 'yyyy-MM')",
            sql,
            flags=re.IGNORECASE
        )
        sql = re.sub(
            r"strftime\(\s*'%Y'\s*,\s*([^)]+)\)",
            r"YEAR(\1)",
            sql,
            flags=re.IGNORECASE
        )
        sql = re.sub(
            r"strftime\(\s*'%m'\s*,\s*([^)]+)\)",
            r"MONTH(\1)",
            sql,
            flags=re.IGNORECASE
        )
        sql = re.sub(
            r"strftime\(\s*'%d'\s*,\s*([^)]+)\)",
            r"DAY(\1)",
            sql,
            flags=re.IGNORECASE
        )

        return sql

    def _translate_string_functions(self, sql: str) -> str:
        sql, quoted = self._protect_quoted_strings(sql)


        def replace_concat(match):
            concat_expr = match.group(0)
            operands = [op.strip() for op in concat_expr.split('||')]
            return f"CONCAT({', '.join(operands)})"


        pattern = r"([a-zA-Z_]\w*(?:\([^)]*\)|'[^']*'|\d+(?:\.\d+)?|\[[^\]]*\])?(?:\s*\|\|\s*[a-zA-Z_]\w*(?:\([^)]*\)|'[^']*'|\d+(?:\.\d+)?|\[[^\]]*\])?)+)"

        max_iterations = 10
        iterations = 0
        while '||' in sql and iterations < max_iterations:
            iterations += 1
            pattern = r"([a-zA-Z_]\w*(?:\([^)]*\))?|__QUOTED_\d+__|[a-zA-Z0-9_.]+(?:\([^)]*\))?)\s*\|\|\s*([a-zA-Z_]\w*(?:\([^)]*\))?|__QUOTED_\d+__|[a-zA-Z0-9_.]+(?:\([^)]*\))?)"

            def replace_concat_pair(m):
                left = m.group(1).strip()
                right = m.group(2).strip()
                return f"CONCAT({left}, {right})"

            new_sql = re.sub(pattern, replace_concat_pair, sql, count=1)
            if new_sql == sql:
                break
            sql = new_sql

        sql = self._restore_quoted_strings(sql, quoted)
        return sql

    def _translate_table_references(self, sql: str) -> str:
        sql, quoted = self._protect_quoted_strings(sql)

        keywords = ['FROM', 'JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'INNER JOIN', 'FULL JOIN', 'CROSS JOIN']

        for keyword in keywords:

            pattern = rf"({keyword})\s+([a-zA-Z_]\w*)(?!\.)\s+(?:([a-zA-Z_]\w+)\s+)?(?=WHERE|ON|JOIN|GROUP|ORDER|LIMIT|;|$|\))"

            def replace_table_ref(m):
                kw = m.group(1)
                table = m.group(2)
                alias = m.group(3) if m.group(3) else ""

                qualified = f"{self.catalog}.{self.schema}.{table}"

                if alias:
                    return f"{kw} {qualified} {alias}"
                else:
                    return f"{kw} {qualified}"

            sql = re.sub(pattern, replace_table_ref, sql, flags=re.IGNORECASE)

        sql = self._restore_quoted_strings(sql, quoted)
        return sql

    def _translate_type_casts(self, sql: str) -> str:
        sql, quoted = self._protect_quoted_strings(sql)

        sql = re.sub(
            r"CAST\s*\(\s*([^)]+)\s+AS\s+TEXT\s*\)",
            r"CAST(\1 AS STRING)",
            sql,
            flags=re.IGNORECASE
        )

        sql = re.sub(
            r"CAST\s*\(\s*([^)]+)\s+AS\s+REAL\s*\)",
            r"CAST(\1 AS DOUBLE)",
            sql,
            flags=re.IGNORECASE
        )

        sql = self._restore_quoted_strings(sql, quoted)
        return sql

    def _translate_limit_offset(self, sql: str) -> str:
        return sql

    def _translate_pragma(self, sql: str) -> str:
        sql, quoted = self._protect_quoted_strings(sql)

        pattern = r"PRAGMA\s+table_info\s*\(\s*([a-zA-Z_]\w*)\s*\)"

        def replace_pragma_table_info(m):
            table = m.group(1)
            return f"DESCRIBE TABLE {self.catalog}.{self.schema}.{table}"

        sql = re.sub(pattern, replace_pragma_table_info, sql, flags=re.IGNORECASE)

        pattern = r"SELECT\s+name\s+FROM\s+sqlite_master\s+WHERE\s+type\s*=\s*'table'"
        replacement = f"SHOW TABLES IN {self.catalog}.{self.schema}"
        sql = re.sub(pattern, replacement, sql, flags=re.IGNORECASE)

        sql = self._restore_quoted_strings(sql, quoted)
        return sql

    def _translate_misc(self, sql: str) -> str:
        sql, quoted = self._protect_quoted_strings(sql)

        pattern = r"GROUP_CONCAT\s*\(\s*([^,)]+)\s*,\s*([^)]+)\s*\)"

        def replace_group_concat_with_sep(m):
            col = m.group(1).strip()
            sep = m.group(2).strip()
            return f"CONCAT_WS({sep}, COLLECT_LIST({col}))"

        sql = re.sub(pattern, replace_group_concat_with_sep, sql, flags=re.IGNORECASE)

        pattern = r"GROUP_CONCAT\s*\(\s*([^)]+)\s*\)"

        def replace_group_concat_no_sep(m):
            col = m.group(1).strip()
            return f"CONCAT_WS(',', COLLECT_LIST({col}))"

        sql = re.sub(pattern, replace_group_concat_no_sep, sql, flags=re.IGNORECASE)

        pattern = r"IFNULL\s*\(\s*([^,)]+)\s*,\s*([^)]+)\s*\)"

        def replace_ifnull(m):
            a = m.group(1).strip()
            b = m.group(2).strip()
            return f"COALESCE({a}, {b})"

        sql = re.sub(pattern, replace_ifnull, sql, flags=re.IGNORECASE)

        sql = re.sub(r"RANDOM\s*\(\s*\)", "RAND()", sql, flags=re.IGNORECASE)

        sql = self._restore_quoted_strings(sql, quoted)
        return sql


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    dialect = DatabricksDialect(catalog='healthcare', schema='production')

    test_cases = [
        ("SELECT date('now') FROM patients",
         "SELECT CURRENT_DATE() FROM healthcare.production.patients"),

        ("SELECT * FROM patients WHERE created_at > date('now', '-30 days')",
         "SELECT * FROM healthcare.production.patients WHERE created_at > DATE_SUB(CURRENT_DATE(), 30)"),

        ("SELECT first_name || ' ' || last_name FROM patients",
         "SELECT CONCAT(first_name, ' ', last_name) FROM healthcare.production.patients"),

        ("SELECT CAST(age AS TEXT) FROM patients",
         "SELECT CAST(age AS STRING) FROM healthcare.production.patients"),

        ("SELECT IFNULL(address, 'Unknown') FROM patients",
         "SELECT COALESCE(address, 'Unknown') FROM healthcare.production.patients"),
    ]

    print("=" * 80)
    print("Databricks Dialect Translator - Test Cases")
    print("=" * 80)

    for i, (sqlite_sql, expected) in enumerate(test_cases, 1):
        result = dialect.translate(sqlite_sql)
        status = "PASS" if result == expected else "FAIL"
        print(f"\nTest {i}: {status}")
        print(f"Input:    {sqlite_sql}")
        print(f"Expected: {expected}")
        print(f"Got:      {result}")

    print("\n" + "=" * 80)
    print("Schema Discovery Examples")
    print("=" * 80)

    discovery_tests = [
        ("list_tables", {}),
        ("describe_table", {"table": "patients"}),
        ("sample_values", {"table": "patients", "column": "gender", "limit": 50}),
        ("count_distinct", {"table": "patients", "column": "gender"}),
        ("row_count", {"table": "patients"}),
    ]

    for query_type, kwargs in discovery_tests:
        result = dialect.translate_schema_discovery(query_type, **kwargs)
        print(f"\n{query_type}:")
        print(f"  {result}")
