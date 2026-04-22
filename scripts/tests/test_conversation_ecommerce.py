import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from conversation_intelligence import ConversationIntelligence

db = '../data/ecommerce_test.db'
ci = ConversationIntelligence(db)

CONVERSATIONS = [
    [
        "how many orders are there",
        "by status",
        "only completed",
        "by shipping method",
    ],
    [
        "total revenue by category",
        "show average instead",
        "by country",
    ],
    [
        "count of customers",
        "by loyalty tier",
        "what about products",
    ],
    [
        "orders greater than 500",
        "by month",
        "same for completed only",
    ],
]

for i, conv in enumerate(CONVERSATIONS):
    session = f"ecom_{i}"
    ci.reset_context(session)
    print(f"\n{'='*70}")
    print(f"CONVERSATION {i+1}")
    print(f"{'='*70}")

    for j, q in enumerate(conv):
        r = ci.process_turn(q, session_id=session)
        fu = "FOLLOW-UP" if r['is_followup'] else "NEW"
        print(f"\n  T{j+1}: \"{q}\"")
        print(f"      [{fu}:{r['followup_type']}] chain={r['chain_depth']}")
        if r['is_followup']:
            print(f"      Rewritten: \"{r['rewritten_question']}\"")
        print(f"      SQL: {r['sql'][:120]}")
        print(f"      Rows: {r['result_count']}")
        if r['modifications']:
            print(f"      Mods: {r['modifications']}")

    suggestions = ci.get_suggested_followups(session)
    if suggestions:
        print(f"\n  Suggestions: {suggestions[:3]}")

print(f"\n{'='*70}")
print("E-COMMERCE CONVERSATION TEST COMPLETE")
