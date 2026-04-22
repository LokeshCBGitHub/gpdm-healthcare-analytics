from __future__ import annotations
import logging
import math
import os
import sqlite3
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_BASE = (os.environ.get('GPDM_BASE_DIR', '') or
         os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DB = os.path.join(_BASE, 'data', 'healthcare_production.db')


class EmbeddingEnrichedFeatures:

    def __init__(self, n_components: int = 16):
        self.n_components = n_components
        self._embedder = None
        self._projections: Dict[str, np.ndarray] = {}
        self._text_cache: Dict[str, np.ndarray] = {}

    def _get_embedder(self):
        if self._embedder is None:
            try:
                from neural_engine import ClinicalEmbedder
                self._embedder = ClinicalEmbedder()
                logger.info("EmbeddingEnrichedFeatures: embedder ready (%s)",
                            self._embedder.backend)
            except Exception as e:
                logger.warning("Embedder unavailable: %s", e)
        return self._embedder

    def _random_projection_matrix(self, d_in: int, d_out: int,
                                   seed: int = 42) -> np.ndarray:
        rng = np.random.RandomState(seed)
        R = rng.randn(d_in, d_out) / math.sqrt(d_out)
        return R.astype(np.float32)

    def embed_column(self, values: pd.Series, col_name: str) -> np.ndarray:
        embedder = self._get_embedder()
        if embedder is None:
            return np.zeros((len(values), self.n_components), dtype=np.float32)

        uniq = values.fillna('UNK').astype(str).unique()
        cache_key = f"{col_name}_{hash(tuple(sorted(uniq)))}"

        if cache_key not in self._text_cache:
            embeddings = embedder.encode(list(uniq))

            if col_name not in self._projections:
                self._projections[col_name] = self._random_projection_matrix(
                    embeddings.shape[1], self.n_components, seed=hash(col_name) & 0x7FFFFFFF)

            projected = embeddings @ self._projections[col_name]
            self._text_cache[cache_key] = {str(v): projected[i]
                                            for i, v in enumerate(uniq)}

        lookup = self._text_cache[cache_key]
        result = np.zeros((len(values), self.n_components), dtype=np.float32)
        for i, v in enumerate(values.fillna('UNK').astype(str)):
            if v in lookup:
                result[i] = lookup[v]
        return result

    def enrich_claim_features(self, df: pd.DataFrame,
                               base_X: np.ndarray,
                               base_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        extra_features = []
        extra_names = []

        if 'diagnosis_code' in df.columns:
            diag_emb = self.embed_column(df['diagnosis_code'], 'diagnosis')
            extra_features.append(diag_emb)
            extra_names.extend([f'diag_emb_{i}' for i in range(self.n_components)])

        if 'denial_reason' in df.columns:
            denial_emb = self.embed_column(df['denial_reason'].fillna('NONE'), 'denial_reason')
            extra_features.append(denial_emb)
            extra_names.extend([f'denial_emb_{i}' for i in range(self.n_components)])

        if not extra_features:
            return base_X, base_names

        enriched_X = np.hstack([base_X] + extra_features).astype(np.float64)
        enriched_names = base_names + extra_names
        logger.info("Features enriched: %d → %d (added %d embedding dims)",
                    base_X.shape[1], enriched_X.shape[1],
                    enriched_X.shape[1] - base_X.shape[1])
        return enriched_X, enriched_names

    def enrich_member_features(self, conn: sqlite3.Connection,
                                member_ids: np.ndarray,
                                base_X: np.ndarray,
                                base_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        embedder = self._get_embedder()
        if embedder is None:
            return base_X, base_names

        try:
            diag_df = pd.read_sql("""
                SELECT member_id,
                       diagnosis_code,
                       COUNT(*) as cnt
                FROM claims_4m
                WHERE diagnosis_code IS NOT NULL
                GROUP BY member_id, diagnosis_code
                ORDER BY member_id, cnt DESC
            """, conn)
        except Exception:
            return base_X, base_names

        top_diag = diag_df.groupby('member_id').first().reset_index()
        diag_map = dict(zip(top_diag['member_id'].values,
                            top_diag['diagnosis_code'].fillna('UNK').values))

        member_diags = pd.Series([diag_map.get(m, 'UNK') for m in member_ids])
        diag_emb = self.embed_column(member_diags, 'member_top_diag')

        enriched_X = np.hstack([base_X, diag_emb]).astype(np.float64)
        enriched_names = base_names + [f'top_diag_emb_{i}' for i in range(self.n_components)]
        logger.info("Member features enriched: %d → %d dims",
                    base_X.shape[1], enriched_X.shape[1])
        return enriched_X, enriched_names


class CausalAwareRiskScorer:

    def __init__(self):
        self.member_scores: Dict[str, Dict[str, Any]] = {}
        self.population_ate = None
        self.cate_distribution = None

    def compute_causal_member_scores(
        self,
        conn: sqlite3.Connection,
        db_path: str,
        gbm_risk: Optional[Dict[str, Dict]] = None,
        max_causal_sample: int = 15000,
    ) -> Dict[str, Dict[str, Any]]:
        t0 = time.time()

        logger.info("Loading GBM risk scores...")
        try:
            risk_rows = conn.execute(
                "SELECT member_id, churn_risk, readmission_risk, risk_tier "
                "FROM _gpdm_member_risk").fetchall()
            risk_by_member = {
                r[0]: {'churn': float(r[1]), 'readmit': float(r[2]), 'tier': r[3]}
                for r in risk_rows
            }
        except Exception:
            risk_by_member = {}

        if not risk_by_member:
            logger.warning("No GBM risk scores found — using defaults")

        logger.info("Running causal inference for population ATE...")
        ate_followup = 0.0
        cate_quantiles = None
        try:
            from causal_inference import estimate_followup_effect
            result = estimate_followup_effect(db_path, max_rows=max_causal_sample)
            if 'error' not in result:
                ate_followup = result['ate']
                cate_quantiles = result.get('cate_quantiles')
                self.population_ate = ate_followup
                self.cate_distribution = cate_quantiles
                logger.info("Population ATE (followup→readmit): %+.4f", ate_followup)
        except Exception as e:
            logger.warning("Causal ATE failed: %s", e)

        logger.info("Computing covariate-stratified CATE approximations...")
        try:
            members = pd.read_sql("""
                SELECT MEMBER_ID, AGE, CHRONIC_CONDITIONS, RISK_SCORE
                FROM members
            """, conn)
        except Exception:
            members = pd.DataFrame()

        member_cate: Dict[str, float] = {}
        if len(members) > 0 and cate_quantiles:
            p10 = cate_quantiles.get('p10', ate_followup)
            p50 = cate_quantiles.get('p50', ate_followup)
            p90 = cate_quantiles.get('p90', ate_followup)

            for _, row in members.iterrows():
                mid = row['MEMBER_ID']
                age = float(row.get('AGE') or 50)
                chronic = float(row.get('CHRONIC_CONDITIONS') or 0)
                risk = float(row.get('RISK_SCORE') or 1.0)

                acuity = np.clip(
                    (chronic / 5.0) * 0.4 +
                    (min(age, 90) / 90.0) * 0.3 +
                    (min(risk, 5) / 5.0) * 0.3,
                    0.0, 1.0
                )

                if acuity >= 0.5:
                    t = (acuity - 0.5) / 0.5
                    cate = p10 * t + p50 * (1 - t)
                else:
                    t = acuity / 0.5
                    cate = p50 * t + p90 * (1 - t)

                member_cate[mid] = float(cate)

        logger.info("Fusing GBM risk + causal CATE into unified scores...")
        for mid in set(list(risk_by_member.keys()) + list(member_cate.keys())):
            gbm = risk_by_member.get(mid, {'churn': 0.5, 'readmit': 0.5, 'tier': 'MEDIUM'})
            cate = member_cate.get(mid, ate_followup)

            combined_risk = max(gbm['churn'], gbm['readmit'])

            intervention_benefit = max(0.0, -cate)

            priority = (combined_risk ** 0.6) * (intervention_benefit ** 0.4 + 0.01)

            if cate < -0.05:
                recommended_action = 'SCHEDULE_FOLLOWUP'
                action_label = 'Schedule 14-day follow-up (strong expected benefit)'
            elif cate < 0:
                recommended_action = 'CARE_COORDINATION'
                action_label = 'Assign care coordinator (moderate expected benefit)'
            else:
                recommended_action = 'MONITOR'
                action_label = 'Remote monitoring only (follow-up shows no measurable benefit)'

            self.member_scores[mid] = {
                'churn_risk': gbm['churn'],
                'readmission_risk': gbm['readmit'],
                'gbm_tier': gbm['tier'],
                'followup_cate': cate,
                'intervention_benefit': intervention_benefit,
                'priority_score': float(priority),
                'recommended_action': recommended_action,
                'action_label': action_label,
            }

        elapsed = time.time() - t0
        logger.info("Causal-aware scoring complete: %d members in %.1fs",
                    len(self.member_scores), elapsed)
        return self.member_scores

    def get_top_priority_members(self, n: int = 100) -> List[Dict[str, Any]]:
        ranked = sorted(self.member_scores.items(),
                       key=lambda x: x[1]['priority_score'], reverse=True)
        return [{'member_id': mid, **scores} for mid, scores in ranked[:n]]

    def get_intervention_summary(self) -> Dict[str, Any]:
        if not self.member_scores:
            return {}
        scores = list(self.member_scores.values())
        actions = defaultdict(int)
        for s in scores:
            actions[s['recommended_action']] += 1

        priorities = [s['priority_score'] for s in scores]
        return {
            'total_members': len(scores),
            'population_ate': self.population_ate,
            'cate_distribution': self.cate_distribution,
            'intervention_counts': dict(actions),
            'priority_stats': {
                'mean': float(np.mean(priorities)),
                'p25': float(np.percentile(priorities, 25)),
                'p50': float(np.percentile(priorities, 50)),
                'p75': float(np.percentile(priorities, 75)),
                'p90': float(np.percentile(priorities, 90)),
            },
            'high_priority_count': sum(1 for p in priorities
                                       if p > np.percentile(priorities, 90)),
        }


class InterventionOptimizer:

    def __init__(self, capacity: Dict[str, int] = None):
        self.capacity = capacity or {
            'SCHEDULE_FOLLOWUP': 5000,
            'CARE_COORDINATION': 1000,
            'MONITOR': 999999,
        }
        self.assignments: List[Dict] = []

    def optimize(self, scorer: CausalAwareRiskScorer) -> List[Dict]:
        if not scorer.member_scores:
            return []

        ranked = sorted(scorer.member_scores.items(),
                       key=lambda x: x[1]['priority_score'], reverse=True)

        remaining = dict(self.capacity)
        self.assignments = []

        for mid, scores in ranked:
            action = scores['recommended_action']

            if remaining.get(action, 0) > 0:
                self.assignments.append({
                    'member_id': mid,
                    'action': action,
                    'action_label': scores['action_label'],
                    'priority_score': scores['priority_score'],
                    'churn_risk': scores['churn_risk'],
                    'readmission_risk': scores['readmission_risk'],
                    'followup_cate': scores['followup_cate'],
                    'expected_benefit': scores['intervention_benefit'],
                })
                remaining[action] -= 1
            elif action == 'SCHEDULE_FOLLOWUP' and remaining.get('CARE_COORDINATION', 0) > 0:
                self.assignments.append({
                    'member_id': mid,
                    'action': 'CARE_COORDINATION',
                    'action_label': 'Care coordination (followup slots full)',
                    'priority_score': scores['priority_score'],
                    'churn_risk': scores['churn_risk'],
                    'readmission_risk': scores['readmission_risk'],
                    'followup_cate': scores['followup_cate'],
                    'expected_benefit': scores['intervention_benefit'] * 0.7,
                })
                remaining['CARE_COORDINATION'] -= 1

        total_benefit = sum(a['expected_benefit'] for a in self.assignments)
        logger.info("Optimized %d assignments, total expected benefit: %.4f",
                    len(self.assignments), total_benefit)
        return self.assignments

    def get_summary(self) -> Dict[str, Any]:
        if not self.assignments:
            return {}
        by_action = defaultdict(list)
        for a in self.assignments:
            by_action[a['action']].append(a)

        return {
            'total_assigned': len(self.assignments),
            'total_expected_benefit': sum(a['expected_benefit'] for a in self.assignments),
            'by_intervention': {
                action: {
                    'count': len(members),
                    'avg_priority': np.mean([m['priority_score'] for m in members]),
                    'avg_benefit': np.mean([m['expected_benefit'] for m in members]),
                    'avg_churn_risk': np.mean([m['churn_risk'] for m in members]),
                    'avg_readmit_risk': np.mean([m['readmission_risk'] for m in members]),
                }
                for action, members in by_action.items()
            },
            'capacity': self.capacity,
        }


class EnsembleScorer:

    def __init__(self):
        self.weights = None
        self.bias = 0.0
        self.calibration_map = None

    def fit(self, signals: np.ndarray, outcomes: np.ndarray,
            n_iter: int = 100, lr: float = 0.01):
        n, k = signals.shape
        self.weights = np.zeros(k)
        self.bias = 0.0

        for iteration in range(n_iter):
            logits = signals @ self.weights + self.bias
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))

            err = probs - outcomes
            grad_w = (signals.T @ err) / n
            grad_b = np.mean(err)

            grad_w += 0.01 * self.weights

            self.weights -= lr * grad_w
            self.bias -= lr * grad_b

            if (iteration + 1) % 25 == 0:
                loss = -np.mean(
                    outcomes * np.log(probs + 1e-10) +
                    (1 - outcomes) * np.log(1 - probs + 1e-10))
                logger.debug("Ensemble iter %d: loss=%.4f", iteration + 1, loss)

        final_scores = signals @ self.weights + self.bias
        final_probs = 1.0 / (1.0 + np.exp(-np.clip(final_scores, -30, 30)))

        n_bins = 20
        bins = np.linspace(0, 1, n_bins + 1)
        self.calibration_map = np.zeros(n_bins)
        for b in range(n_bins):
            mask = (final_probs >= bins[b]) & (final_probs < bins[b + 1])
            if mask.sum() > 0:
                self.calibration_map[b] = outcomes[mask].mean()
            else:
                self.calibration_map[b] = (bins[b] + bins[b + 1]) / 2

        logger.info("Ensemble weights: %s, bias: %.4f",
                    {f's{i}': round(w, 4) for i, w in enumerate(self.weights)},
                    self.bias)

    def predict(self, signals: np.ndarray) -> np.ndarray:
        if self.weights is None:
            return np.full(signals.shape[0], 0.5)
        raw = signals @ self.weights + self.bias
        probs = 1.0 / (1.0 + np.exp(-np.clip(raw, -30, 30)))
        if self.calibration_map is not None:
            n_bins = len(self.calibration_map)
            bin_idx = np.clip((probs * n_bins).astype(int), 0, n_bins - 1)
            probs = self.calibration_map[bin_idx]
        return probs

    def feature_importance(self) -> Dict[str, float]:
        if self.weights is None:
            return {}
        names = ['gbm_churn', 'gbm_readmit', 'causal_cate',
                 'anomaly', 'embed_similarity', 'provider_score']
        abs_w = np.abs(self.weights)
        total = abs_w.sum() + 1e-10
        return {names[i] if i < len(names) else f'signal_{i}':
                round(float(abs_w[i] / total), 4)
                for i in range(len(self.weights))}


class HybridIntelligence:

    def __init__(self, db_path: str = _DB):
        self.db_path = db_path
        self.enricher = EmbeddingEnrichedFeatures(n_components=16)
        self.scorer = CausalAwareRiskScorer()
        self.optimizer = InterventionOptimizer()
        self.ensemble = EnsembleScorer()
        self.status: Dict[str, Any] = {}

    def run_full_pipeline(self, max_causal_sample: int = 15000) -> Dict[str, Any]:
        t_total = time.time()
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA cache_size=-100000')

        results = {}

        print('\n[Hybrid 1/4] Causal-aware member risk scoring...')
        t0 = time.time()
        self.scorer.compute_causal_member_scores(
            conn, self.db_path, max_causal_sample=max_causal_sample)
        results['scoring'] = self.scorer.get_intervention_summary()
        results['scoring']['time_seconds'] = round(time.time() - t0, 1)
        print(f'  Scored {results["scoring"]["total_members"]} members in '
              f'{results["scoring"]["time_seconds"]}s')
        if results['scoring'].get('intervention_counts'):
            for action, count in results['scoring']['intervention_counts'].items():
                print(f'    {action}: {count:,} members')

        print('\n[Hybrid 2/4] Optimizing intervention assignments...')
        t0 = time.time()
        self.optimizer.optimize(self.scorer)
        opt_summary = self.optimizer.get_summary()
        results['optimization'] = opt_summary
        results['optimization']['time_seconds'] = round(time.time() - t0, 1)
        print(f'  Assigned {opt_summary.get("total_assigned", 0):,} interventions')
        for action, stats in opt_summary.get('by_intervention', {}).items():
            print(f'    {action}: {stats["count"]:,} (avg benefit: {stats["avg_benefit"]:.4f})')

        print('\n[Hybrid 3/4] Training ensemble scorer...')
        t0 = time.time()
        ensemble_result = self._train_ensemble(conn)
        results['ensemble'] = ensemble_result
        results['ensemble']['time_seconds'] = round(time.time() - t0, 1)
        print(f'  Ensemble trained: {ensemble_result}')

        print('\n[Hybrid 4/4] Storing unified scores in DB...')
        t0 = time.time()
        stored = self._store_unified_scores(conn)
        results['storage'] = {'rows_stored': stored, 'time_seconds': round(time.time() - t0, 1)}
        print(f'  Stored {stored:,} unified member records')

        conn.close()

        total_time = time.time() - t_total
        results['total_time_seconds'] = round(total_time, 1)
        self.status = results

        print(f'\n{"="*60}')
        print(f'Hybrid Intelligence pipeline complete in {total_time:.1f}s')
        print(f'  Members scored: {results["scoring"]["total_members"]:,}')
        print(f'  Interventions assigned: {opt_summary.get("total_assigned", 0):,}')
        print(f'  Population ATE: {results["scoring"].get("population_ate", "N/A")}')
        print(f'{"="*60}')

        return results

    def _train_ensemble(self, conn: sqlite3.Connection) -> Dict[str, Any]:
        if not self.scorer.member_scores:
            return {'status': 'no_scores'}

        member_ids = list(self.scorer.member_scores.keys())
        signals = []
        for mid in member_ids:
            s = self.scorer.member_scores[mid]
            signals.append([
                s['churn_risk'],
                s['readmission_risk'],
                s['followup_cate'],
                0.0,
                0.0,
                0.0,
            ])
        signals = np.array(signals, dtype=np.float64)

        outcomes = np.array([
            1.0 if self.scorer.member_scores[mid]['readmission_risk'] > 0.5
            else 0.0 for mid in member_ids
        ], dtype=np.float64)

        n = len(member_ids)
        train_n = min(n, 50000)
        idx = np.random.RandomState(42).choice(n, train_n, replace=False)
        self.ensemble.fit(signals[idx], outcomes[idx], n_iter=100, lr=0.01)

        importance = self.ensemble.feature_importance()
        return {
            'status': 'trained',
            'n_train': train_n,
            'weights': importance,
            'positive_rate': float(outcomes.mean()),
        }

    def _store_unified_scores(self, conn: sqlite3.Connection) -> int:
        conn.execute('DROP TABLE IF EXISTS _gpdm_hybrid_scores')
        conn.execute('''CREATE TABLE _gpdm_hybrid_scores (
            member_id TEXT PRIMARY KEY,
            churn_risk REAL,
            readmission_risk REAL,
            followup_cate REAL,
            intervention_benefit REAL,
            priority_score REAL,
            recommended_action TEXT,
            gbm_tier TEXT,
            ensemble_score REAL
        )''')

        member_ids = list(self.scorer.member_scores.keys())
        if member_ids and self.ensemble.weights is not None:
            signals = np.array([
                [s['churn_risk'], s['readmission_risk'], s['followup_cate'],
                 0.0, 0.0, 0.0]
                for s in [self.scorer.member_scores[m] for m in member_ids]
            ], dtype=np.float64)
            ens_scores = self.ensemble.predict(signals)
        else:
            ens_scores = np.full(len(member_ids), 0.5)

        rows = []
        for i, mid in enumerate(member_ids):
            s = self.scorer.member_scores[mid]
            rows.append((
                mid,
                s['churn_risk'],
                s['readmission_risk'],
                s['followup_cate'],
                s['intervention_benefit'],
                s['priority_score'],
                s['recommended_action'],
                s['gbm_tier'],
                float(ens_scores[i]),
            ))

        batch_size = 5000
        for i in range(0, len(rows), batch_size):
            conn.executemany(
                'INSERT INTO _gpdm_hybrid_scores VALUES (?,?,?,?,?,?,?,?,?)',
                rows[i:i + batch_size])
        conn.commit()

        conn.execute('CREATE INDEX IF NOT EXISTS idx_hybrid_priority '
                     'ON _gpdm_hybrid_scores(priority_score DESC)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_hybrid_action '
                     'ON _gpdm_hybrid_scores(recommended_action)')
        conn.commit()

        return len(rows)

    def get_top_actionable(self, n: int = 20) -> List[Dict]:
        return self.scorer.get_top_priority_members(n)

    def get_status(self) -> Dict[str, Any]:
        return self.status


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='GPDM Hybrid Intelligence')
    ap.add_argument('--db', default=_DB)
    ap.add_argument('--max-causal', type=int, default=15000)
    ap.add_argument('--status', action='store_true')
    args = ap.parse_args()

    if args.status:
        conn = sqlite3.connect(args.db)
        try:
            rows = conn.execute(
                "SELECT COUNT(*), AVG(priority_score) FROM _gpdm_hybrid_scores"
            ).fetchone()
            print(f"Hybrid scores: {rows[0]} members, avg priority: {rows[1]:.4f}")
            by_action = conn.execute(
                "SELECT recommended_action, COUNT(*) "
                "FROM _gpdm_hybrid_scores GROUP BY recommended_action"
            ).fetchall()
            for action, cnt in by_action:
                print(f"  {action}: {cnt:,}")
        except Exception as e:
            print(f"No hybrid scores found: {e}")
        conn.close()
    else:
        hybrid = HybridIntelligence(db_path=args.db)
        result = hybrid.run_full_pipeline(max_causal_sample=args.max_causal)
        import json
        print(json.dumps(result, indent=2, default=str))
