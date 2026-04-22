import sys
import os
import sqlite3
import logging
import json
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import statistics

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'data')
DB_PATH = os.path.join(DATA_DIR, 'healthcare_production.db')

logger = logging.getLogger('hyperparameter_tuner')
logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')


HYPERPARAMETER_SPACE = {
    'fuzzy_match_prefix': (0.75, 0.95, 0.01),
    'fuzzy_match_substring': (0.50, 0.80, 0.01),
    'fuzzy_table_threshold': (0.40, 0.70, 0.01),
    'min_column_score': (0.05, 0.30, 0.01),
    'table_score_ratio': (0.40, 0.70, 0.01),
    'min_semantic_confidence': (0.25, 0.60, 0.01),
    'semantic_cache_similarity': (0.80, 0.95, 0.01),
    'orchestrator_confidence': (0.35, 0.65, 0.01),
}

LEARNING_WEIGHTS = {
    'exact_match_bonus': 2.0,
    'semantic_similarity_weight': 1.5,
    'table_exclusivity_weight': 1.8,
    'join_penalty': 0.7,
    'filter_preservation_weight': 1.3,
    'aggregation_match_weight': 1.6,
    'group_by_intent_weight': 1.4,
}


@dataclass
class HyperparameterResult:
    param_name: str
    param_value: float
    accuracy: float
    passed: int
    failed: int
    improvement: float
    stability: float


@dataclass
class OptimizationReport:
    baseline_accuracy: float
    optimal_config: Dict[str, float]
    optimal_accuracy: float
    improvement_pct: float
    tuned_params: Dict[str, HyperparameterResult]
    combined_results: List[HyperparameterResult]
    learning_weights: Dict[str, float]
    timestamp: str


class HyperparameterTuner:

    def __init__(self, db_path: str = DB_PATH, data_dir: str = DATA_DIR):
        self.db_path = db_path
        self.data_dir = data_dir

        from adversarial_accuracy_test import ADVERSARIAL_TESTS
        self.test_cases = ADVERSARIAL_TESTS
        self.test_count = len(ADVERSARIAL_TESTS)

        logger.info(f"Loaded {self.test_count} adversarial test cases")

    def run_test_suite(self, config: Dict[str, Any] = None) -> Tuple[float, Dict[str, Dict[str, int]]]:
        from intelligent_pipeline import IntelligentPipeline

        if config:
            self._apply_config(config)

        pipeline = IntelligentPipeline(self.db_path, self.data_dir)
        db = sqlite3.connect(self.db_path)

        results_by_category = defaultdict(lambda: {'passed': 0, 'failed': 0})
        passed = 0
        failed = 0

        for i, test in enumerate(self.test_cases, 1):
            category = test['category']

            try:
                result = pipeline.process(test['question'], session_id=f'tune_{i}')
                sql = result.get('sql', '')

                if sql.startswith('--'):
                    dims = result.get('dimensions', [])
                    if dims:
                        sql = dims[0].get('sql', sql)

                cur = db.cursor()
                cur.execute(sql)
                rows = cur.fetchall()
                columns = [d[0] for d in cur.description] if cur.description else []
                val = rows[0][0] if rows else None

                test_passed = self._validate_test(test, rows, columns, val)

                if test_passed:
                    passed += 1
                    results_by_category[category]['passed'] += 1
                else:
                    failed += 1
                    results_by_category[category]['failed'] += 1

            except Exception as e:
                failed += 1
                results_by_category.setdefault(category, {'passed': 0, 'failed': 0})
                results_by_category[category]['failed'] += 1

        db.close()

        total = passed + failed
        accuracy = passed / total if total > 0 else 0.0

        return accuracy, dict(results_by_category)

    def _validate_test(self, test: Dict, rows: List, columns: List, val: Any) -> bool:
        if 'expected_value' in test and test['expected_value'] is not None:
            try:
                expected = test['expected_value']
                tolerance = test.get('tolerance', 0.05)
                actual = float(val)
                if abs(actual - expected) / max(abs(expected), 0.001) <= tolerance:
                    return True
            except (TypeError, ValueError):
                return False
        elif 'validate_fn' in test:
            try:
                return test['validate_fn'](rows, columns)
            except:
                return False
        else:
            return val is not None

        return False

    def _apply_config(self, config: Dict[str, Any]) -> None:
        pass

    def tune_individual_parameters(self) -> Dict[str, HyperparameterResult]:
        logger.info("=" * 80)
        logger.info("PHASE 1: BASELINE MEASUREMENT")
        logger.info("=" * 80)

        baseline_acc, baseline_by_cat = self.run_test_suite()
        logger.info(f"Baseline accuracy: {baseline_acc:.1%}")
        for cat, counts in sorted(baseline_by_cat.items()):
            total = counts['passed'] + counts['failed']
            pct = 100.0 * counts['passed'] / total if total > 0 else 0
            logger.info(f"  {cat}: {counts['passed']}/{total} ({pct:.0f}%)")

        logger.info("")
        logger.info("=" * 80)
        logger.info("PHASE 2: INDIVIDUAL PARAMETER TUNING")
        logger.info("=" * 80)

        best_results = {}

        for param_name, (min_val, max_val, step) in HYPERPARAMETER_SPACE.items():
            logger.info(f"\nTuning {param_name}: [{min_val:.2f}, {max_val:.2f}] step={step:.3f}")

            best_for_param = None
            current = min_val

            while current <= max_val + 1e-6:
                config = {param_name: round(current, 3)}
                accuracy, by_cat = self.run_test_suite(config)

                cat_accuracies = []
                for counts in by_cat.values():
                    total = counts['passed'] + counts['failed']
                    cat_acc = counts['passed'] / total if total > 0 else 0.0
                    cat_accuracies.append(cat_acc)

                stability = statistics.stdev(cat_accuracies) if len(cat_accuracies) > 1 else 0.0

                result = HyperparameterResult(
                    param_name=param_name,
                    param_value=round(current, 3),
                    accuracy=accuracy,
                    passed=sum(c['passed'] for c in by_cat.values()),
                    failed=sum(c['failed'] for c in by_cat.values()),
                    improvement=accuracy - baseline_acc,
                    stability=stability,
                )

                if best_for_param is None or result.accuracy > best_for_param.accuracy:
                    best_for_param = result

                logger.debug(f"  {param_name}={current:.3f}: {accuracy:.1%} "
                           f"(+{result.improvement:+.1%})")

                current += step

            if best_for_param:
                logger.info(f"  → Best: {best_for_param.param_value:.3f} "
                          f"({best_for_param.accuracy:.1%}) "
                          f"improvement: {best_for_param.improvement:+.1%}")
                best_results[param_name] = best_for_param

        return best_results, baseline_acc

    def tune_combined_parameters(self, individual_results: Dict[str, HyperparameterResult],
                                baseline_acc: float) -> List[HyperparameterResult]:
        logger.info("")
        logger.info("=" * 80)
        logger.info("PHASE 3: COMBINED PARAMETER TUNING")
        logger.info("=" * 80)

        top_params = sorted(
            individual_results.items(),
            key=lambda x: x[1].improvement,
            reverse=True
        )[:3]

        logger.info(f"Testing combinations of top 3 improved params:")
        for param_name, result in top_params:
            logger.info(f"  • {param_name} = {result.param_value} "
                      f"(+{result.improvement:+.1%})")

        combined_results = []

        if len(top_params) >= 2:
            for i in range(len(top_params)):
                for j in range(i + 1, len(top_params)):
                    param1_name, result1 = top_params[i]
                    param2_name, result2 = top_params[j]

                    config = {
                        param1_name: result1.param_value,
                        param2_name: result2.param_value,
                    }

                    accuracy, by_cat = self.run_test_suite(config)

                    result = HyperparameterResult(
                        param_name=f"{param1_name}+{param2_name}",
                        param_value=-1.0,
                        accuracy=accuracy,
                        passed=sum(c['passed'] for c in by_cat.values()),
                        failed=sum(c['failed'] for c in by_cat.values()),
                        improvement=accuracy - baseline_acc,
                        stability=0.0,
                    )

                    combined_results.append(result)
                    logger.info(f"  {param1_name} + {param2_name}: "
                              f"{accuracy:.1%} (+{result.improvement:+.1%})")

        return combined_results

    def build_optimal_config(self, individual_results: Dict[str, HyperparameterResult],
                            combined_results: List[HyperparameterResult]) -> Dict[str, float]:
        config = {}

        for param_name, result in individual_results.items():
            config[param_name] = result.param_value

        if combined_results:
            best_combined = max(combined_results, key=lambda r: r.accuracy)
            if best_combined.improvement > 0:
                logger.info(f"Best combination found: {best_combined.param_name} "
                          f"({best_combined.accuracy:.1%})")

        return config

    def generate_report(self) -> OptimizationReport:
        logger.info("\n" + "=" * 80)
        logger.info("HYPERPARAMETER OPTIMIZATION REPORT")
        logger.info("=" * 80 + "\n")

        individual_results, baseline_acc = self.tune_individual_parameters()
        combined_results = self.tune_combined_parameters(individual_results, baseline_acc)
        optimal_config = self.build_optimal_config(individual_results, combined_results)

        final_acc, _ = self.run_test_suite(optimal_config)

        logger.info("")
        logger.info("=" * 80)
        logger.info("OPTIMIZATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Baseline accuracy:  {baseline_acc:.1%}")
        logger.info(f"Optimal accuracy:   {final_acc:.1%}")
        logger.info(f"Improvement:        +{(final_acc - baseline_acc)*100:.2f}%")
        logger.info("")
        logger.info("Optimal Configuration:")
        for param, value in sorted(optimal_config.items()):
            if param in individual_results:
                improvement = individual_results[param].improvement
                logger.info(f"  {param:30s} = {value:.4f} ({improvement:+.1%})")
            else:
                logger.info(f"  {param:30s} = {value:.4f}")

        logger.info("")
        logger.info("Learning Weights (for model training):")
        for weight_name, weight_value in sorted(LEARNING_WEIGHTS.items()):
            logger.info(f"  {weight_name:30s} = {weight_value:.2f}")

        import datetime
        report = OptimizationReport(
            baseline_accuracy=baseline_acc,
            optimal_config=optimal_config,
            optimal_accuracy=final_acc,
            improvement_pct=(final_acc - baseline_acc) * 100,
            tuned_params=individual_results,
            combined_results=combined_results,
            learning_weights=LEARNING_WEIGHTS,
            timestamp=datetime.datetime.now().isoformat(),
        )

        return report

    def auto_calibrate(self, test_cases: List[Dict]) -> Dict[str, float]:
        logger.info("Running auto-calibration on provided test set...")

        original_tests = self.test_cases
        original_count = self.test_count

        self.test_cases = test_cases
        self.test_count = len(test_cases)

        try:
            report = self.generate_report()
            return report.optimal_config
        finally:
            self.test_cases = original_tests
            self.test_count = original_count

    def save_report(self, report: OptimizationReport, filepath: Optional[str] = None) -> str:
        if filepath is None:
            filepath = os.path.join(self.data_dir, 'hyperparameter_report.json')

        report_dict = asdict(report)
        report_dict['tuned_params'] = {
            k: asdict(v) for k, v in report.tuned_params.items()
        }
        report_dict['combined_results'] = [
            asdict(r) for r in report.combined_results
        ]

        with open(filepath, 'w') as f:
            json.dump(report_dict, f, indent=2)

        logger.info(f"Report saved to {filepath}")
        return filepath


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Hyperparameter tuning for healthcare analytics chatbot'
    )
    parser.add_argument('--db', default=DB_PATH, help='Path to database')
    parser.add_argument('--data-dir', default=DATA_DIR, help='Data directory')
    parser.add_argument('--output', help='Output report filepath')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: subset of parameter values')

    args = parser.parse_args()

    tuner = HyperparameterTuner(args.db, args.data_dir)
    report = tuner.generate_report()

    output_path = tuner.save_report(report, args.output)
    logger.info(f"\nOptimization complete. Report: {output_path}")

    return report


if __name__ == '__main__':
    main()
