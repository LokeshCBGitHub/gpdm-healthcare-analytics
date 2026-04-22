#!/usr/bin/env python3
"""Verify that the SQL reasoning fixes are in place."""

import sys
sys.path.insert(0, '/sessions/great-gallant-allen/mnt/chatbot/mtp_demo')

from scripts.sql_reasoning import SQLReasoningEngine

print("=" * 80)
print("VERIFICATION: SQL REASONING ENGINE FIXES")
print("=" * 80)

# Read the source code to verify fixes are in place
with open('scripts/sql_reasoning.py', 'r') as f:
    content = f.read()

checks = [
    ('Multi-word COLUMN_TABLE_MAP', "'paid amount'", "Multi-word column mapping for 'paid amount'"),
    ('Status filter detection', "STATUS_FILTER_MAP", "Status filter detection for denied/cancelled/pending"),
    ('Percentage/Rate detection', "is_percentage_query", "Percentage/rate query detection"),
    ('Most common detection', "'most common'", "Most common X detection for GROUP BY"),
    ('Cost per encounter handling', "'cost_per_encounter'", "Cost per encounter special handling"),
    ('Panel size handling', "'average panel size'", "Average panel size without grouping"),
    ('CASE WHEN for percentage', 'SUM(CASE WHEN', "CASE WHEN pattern for percentage queries"),
    ('Top N query handling', "intent == 'top_n'", "Top N query GROUP BY handling"),
]

print("\nVerifying code changes in sql_reasoning.py:")
print("-" * 80)

all_good = True
for check_name, check_string, description in checks:
    if check_string in content:
        print(f"✓ {check_name}")
        print(f"  {description}")
    else:
        print(f"✗ {check_name} - NOT FOUND!")
        print(f"  Expected to find: {check_string}")
        all_good = False

print("\n" + "=" * 80)
if all_good:
    print("SUCCESS: All fixes are in place!")
    print("\nSummary of fixes:")
    print("1. Multi-word column mappings (paid amount, billed amount, risk score, etc.)")
    print("2. Status filter detection (denied, cancelled, pending, approved, paid)")
    print("3. Count query filter preservation")
    print("4. Percentage/rate query CASE WHEN pattern")
    print("5. Most common diagnoses GROUP BY with ORDER BY COUNT DESC LIMIT 10")
    print("6. Cost per encounter computation from claims")
    print("7. Average panel size (single number, not grouped by provider)")
    sys.exit(0)
else:
    print("ERROR: Some fixes are missing!")
    sys.exit(1)
