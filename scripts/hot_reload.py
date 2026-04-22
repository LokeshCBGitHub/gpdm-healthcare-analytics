from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger('gpdm.hot_reload')


@dataclass
class SchemaChange:
    table: str
    action: str
    columns_added: List[str] = field(default_factory=list)
    rows_added: int = 0
    source_file: str = ''
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()

    @property
    def affects_kpi_tables(self) -> bool:
        kpi_tables = {'members', 'encounters', 'claims', 'providers',
                      'diagnoses', 'prescriptions', 'appointments', 'referrals'}
        return self.table.lower() in kpi_tables


@dataclass
class ReloadReport:
    table: str
    steps_completed: List[str] = field(default_factory=list)
    steps_failed: List[str] = field(default_factory=list)
    duration_ms: int = 0
    success: bool = True


class HotReloadController:

    def __init__(self, pipeline=None):
        self._pipeline = pipeline
        self._reload_count = 0
        self._history: List[ReloadReport] = []

    def on_schema_change(self, change: SchemaChange) -> ReloadReport:
        t0 = time.time()
        self._reload_count += 1
        report = ReloadReport(table=change.table)

        logger.info("HOT RELOAD: table='%s', action='%s', rows=%d",
                    change.table, change.action, change.rows_added)

        self._step(report, 'schema_relearn', lambda: self._relearn_schema(change))

        self._step(report, 'perf_indexes', lambda: self._update_indexes(change))

        if change.affects_kpi_tables:
            self._step(report, 'kpi_cube_refresh', lambda: self._refresh_kpi_cube(change))

        self._step(report, 'cache_invalidation', lambda: self._invalidate_caches(change))

        self._step(report, 'answer_cache_clear', lambda: self._clear_answer_cache(change))

        if change.affects_kpi_tables:
            self._step(report, 'executive_kpis', lambda: self._refresh_executive_kpis(change))

        self._step(report, 'retrieval_index', lambda: self._update_retrieval_index(change))

        report.duration_ms = round((time.time() - t0) * 1000)
        report.success = len(report.steps_failed) == 0
        self._history.append(report)

        logger.info("HOT RELOAD complete: %d/%d steps OK in %dms (table='%s')",
                    len(report.steps_completed),
                    len(report.steps_completed) + len(report.steps_failed),
                    report.duration_ms, change.table)
        return report

    def on_data_update(self, table: str, rows_added: int = 0) -> ReloadReport:
        change = SchemaChange(table=table, action='append', rows_added=rows_added)
        return self.on_schema_change(change)

    def _step(self, report: ReloadReport, name: str, fn):
        try:
            fn()
            report.steps_completed.append(name)
        except Exception as e:
            report.steps_failed.append(f"{name}: {e}")
            logger.warning("Hot reload step '%s' failed: %s", name, e)


    def _relearn_schema(self, change: SchemaChange):
        if not self._pipeline:
            return
        learner = getattr(self._pipeline, 'learner', None)
        if learner and hasattr(learner, 'profile_table'):
            learner.profile_table(change.table)
            logger.info("Schema re-profiled: %s", change.table)
        elif learner and hasattr(learner, 'learn'):
            import sqlite3
            conn = sqlite3.connect(self._pipeline.db_path)
            try:
                learner.learn(conn)
            finally:
                conn.close()
            logger.info("Full schema re-learn triggered by %s change", change.table)

    def _update_indexes(self, change: SchemaChange):
        if not self._pipeline:
            return
        try:
            from perf_indexes import optimize_db
            optimize_db(self._pipeline.db_path, analyze=True, vacuum=False)
            logger.info("Performance indexes updated for %s", change.table)
        except ImportError:
            pass

    def _refresh_kpi_cube(self, change: SchemaChange):
        if not self._pipeline:
            return
        try:
            from kpi_cube import refresh_cube
            stats = refresh_cube(self._pipeline.db_path, full=False)
            self._pipeline._kpi_cube_stats = stats
            logger.info("KPI cube refreshed: %d rows", stats.get('rows_in_cube', 0))
        except ImportError:
            pass

    def _invalidate_caches(self, change: SchemaChange):
        if not self._pipeline:
            return
        live_cache = getattr(self._pipeline, 'live_cache', None)
        if live_cache:
            live_cache.invalidate_version(self._pipeline.db_path, [change.table])
            logger.info("LiveCache invalidated for %s", change.table)

        semantic_cache = getattr(self._pipeline, 'semantic_cache', None)
        if semantic_cache and hasattr(semantic_cache, 'clear'):
            semantic_cache.clear()
            logger.info("Semantic cache cleared")

    def _clear_answer_cache(self, change: SchemaChange):
        if not self._pipeline:
            return
        answer_cache = getattr(self._pipeline, 'answer_cache', None)
        if answer_cache and hasattr(answer_cache, 'clear'):
            answer_cache.clear()
            logger.info("Answer cache cleared due to schema change")

    def _refresh_executive_kpis(self, change: SchemaChange):
        if not self._pipeline:
            return
        try:
            from executive_kpis import compute_executive_kpis
            self._pipeline._executive_kpis = compute_executive_kpis(self._pipeline.db_path)
            logger.info("Executive KPIs refreshed")
        except ImportError:
            pass

    def _update_retrieval_index(self, change: SchemaChange):
        if not self._pipeline:
            return
        retrieval_index = getattr(self._pipeline, 'retrieval_index', None)
        if retrieval_index and hasattr(retrieval_index, 'rebuild'):
            retrieval_index.rebuild()
            logger.info("Retrieval index rebuilt")

    def history(self) -> List[Dict[str, Any]]:
        return [
            {
                'table': r.table,
                'steps_ok': len(r.steps_completed),
                'steps_failed': len(r.steps_failed),
                'duration_ms': r.duration_ms,
                'success': r.success,
            }
            for r in self._history[-20:]
        ]
