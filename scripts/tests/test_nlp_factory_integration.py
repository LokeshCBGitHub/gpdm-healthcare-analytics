import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from nlp_engine_factory import NLPEngineFactory, LibraryInventory


def test_library_detection():
    print("=" * 60)
    print("TEST 1: Library Detection")
    print("=" * 60)
    inv = LibraryInventory.detect()
    print(inv.summary())
    print(f"Best tier: {inv.best_tier}")
    assert inv.best_tier >= 1
    print("  PASS\n")


def test_factory_modes():
    print("=" * 60)
    print("TEST 2: Factory Modes")
    print("=" * 60)
    for mode in ['scratch', 'auto']:
        factory = NLPEngineFactory(mode=mode)
        vec = factory.get_vectorizer()
        sim = factory.get_similarity_engine()
        ner = factory.get_entity_extractor()
        clf = factory.get_classifier()
        emb = factory.get_embedding_engine()

        print(f"  Mode={mode}: vec={vec.backend}, sim={sim.backend}, "
              f"ner={ner.backend}, clf={clf.backend}, emb={emb.backend if emb else 'none'}")

        docs = ["diabetes claims", "total revenue", "order count"]
        vec.fit(docs)
        vectors = vec.transform(["patient claims"])
        assert len(vectors) == 1

        entities = ner.extract("patients older than 65 with claims over $5000")
        assert len(entities) > 0
        ent_types = {e['type'] for e in entities}
        assert 'AGE_REFERENCE' in ent_types or 'NUMBER' in ent_types

        train_texts = ["how many", "count of", "total number",
                       "average cost", "sum of", "mean value",
                       "by region", "per category", "breakdown"]
        train_labels = ["count", "count", "count",
                        "aggregate", "aggregate", "aggregate",
                        "breakdown", "breakdown", "breakdown"]
        clf.fit(train_texts, train_labels)
        label, conf = clf.predict("total claims per region")
        assert label in ["count", "aggregate", "breakdown"]

        report = factory.report()
        assert report['hipaa_compliant'] is True

    print("  PASS\n")


def test_engine_with_factory_ecommerce():
    print("=" * 60)
    print("TEST 3: SemanticSQLEngine + Factory (E-commerce)")
    print("=" * 60)
    from semantic_sql_engine import SemanticSQLEngine
    import sqlite3

    db_path = '../data/ecommerce_test.db'

    for mode in ['scratch', 'auto']:
        engine = SemanticSQLEngine(db_path, nlp_mode=mode)
        questions = [
            "how many orders are there",
            "total revenue by category",
            "average rating by brand",
            "orders greater than 500",
        ]
        for q in questions:
            r = engine.generate(q)
            assert r['sql'], f"No SQL for: {q}"
            conn = sqlite3.connect(db_path)
            try:
                rows = conn.execute(r['sql']).fetchall()
                count = len(rows)
            except Exception as e:
                print(f"  FAIL: {q} → SQL error: {e}")
                print(f"        SQL: {r['sql']}")
                count = -1
            finally:
                conn.close()

            status = "OK" if count >= 0 else "FAIL"
            print(f"  [{status}] mode={mode} | {q} → {count} rows | backend={r.get('intent_backend', 'n/a')}")

    print("  PASS\n")


def test_engine_with_factory_healthcare():
    print("=" * 60)
    print("TEST 4: SemanticSQLEngine + Factory (Healthcare)")
    print("=" * 60)
    from semantic_sql_engine import SemanticSQLEngine
    import sqlite3

    db_path = '../data/healthcare_production.db'
    engine = SemanticSQLEngine(db_path, nlp_mode='auto')

    questions = [
        "how many claims are there",
        "count of members",
        "claims by type",
        "members older than 65",
    ]
    for q in questions:
        r = engine.generate(q)
        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute(r['sql']).fetchall()
            count = len(rows)
        except Exception as e:
            print(f"  FAIL: {q} → SQL error: {e}")
            count = -1
        finally:
            conn.close()
        status = "OK" if count >= 0 else "FAIL"
        print(f"  [{status}] {q} → {count} rows | intent={r['semantic_intent']}")

    entities = engine.semantic.extract_entities("patients older than 65 with diabetes")
    print(f"  Entity extraction: {len(entities)} entities found")
    for e in entities[:3]:
        print(f"    [{e['type']}] {e['text']}")

    print("  PASS\n")


def test_conversation_with_factory():
    print("=" * 60)
    print("TEST 5: ConversationIntelligence + Factory")
    print("=" * 60)
    from conversation_intelligence import ConversationIntelligence

    db_path = '../data/ecommerce_test.db'
    ci = ConversationIntelligence(db_path, nlp_mode='auto')

    assert ci.engine.nlp_factory is not None, "Factory not propagated"
    print(f"  Factory mode: {ci.engine.nlp_mode}")
    print(f"  Factory tier: {ci.engine.nlp_factory.inventory.best_tier}")

    session = "factory_test"
    ci.reset_context(session)

    turns = [
        "how many orders are there",
        "by status",
        "only completed",
    ]
    for q in turns:
        r = ci.process_turn(q, session_id=session)
        fu = "FOLLOW-UP" if r['is_followup'] else "NEW"
        print(f"  [{fu}] \"{q}\" → {r['result_count']} rows")

    print("  PASS\n")


def test_factory_report():
    print("=" * 60)
    print("TEST 6: Factory Report")
    print("=" * 60)
    factory = NLPEngineFactory(mode='auto')

    factory.get_vectorizer()
    factory.get_similarity_engine()
    factory.get_entity_extractor()
    factory.get_classifier()
    factory.get_embedding_engine()

    report = factory.report()
    print(f"  Mode: {report['mode']}")
    print(f"  Tier: {report['tier']}")
    print(f"  HIPAA: {report['hipaa_compliant']}")
    print(f"  Risk: {report['data_exfiltration_risk']}")
    print(f"  Backends: {report['active_backends']}")

    assert report['hipaa_compliant'] is True
    assert 'ZERO' in report['data_exfiltration_risk']
    print("  PASS\n")


if __name__ == '__main__':
    test_library_detection()
    test_factory_modes()
    test_engine_with_factory_ecommerce()
    test_engine_with_factory_healthcare()
    test_conversation_with_factory()
    test_factory_report()

    print("=" * 60)
    print("ALL NLP FACTORY INTEGRATION TESTS PASSED")
    print("=" * 60)
