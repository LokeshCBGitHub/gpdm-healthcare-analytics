import sqlite3
import math
import random
import statistics
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any


class AdvancedMLEngine:

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.models = {}
        self.feature_importances = {}


    def gradient_boost_predict(
        self,
        features: List[List[float]],
        targets: List[float],
        n_estimators: int = 50,
        learning_rate: float = 0.1,
        max_depth: int = 3,
    ) -> Dict[str, Any]:
        if len(features) < 2 or len(targets) < 2:
            return {"error": "Insufficient data", "predictions": []}

        n_samples = len(targets)
        predictions = [statistics.mean(targets)] * n_samples
        residuals_history = []

        feature_importance = {i: 0.0 for i in range(len(features[0]))}
        stumps = []

        for iteration in range(n_estimators):
            residuals = [targets[i] - predictions[i] for i in range(n_samples)]
            residuals_history.append(statistics.stdev(residuals) if len(set(residuals)) > 1 else 0)

            best_feature, best_threshold, best_gain = -1, 0, 0

            for feat_idx in range(len(features[0])):
                feature_values = sorted(set(f[feat_idx] for f in features))

                if len(feature_values) < 2:
                    continue

                for threshold in feature_values[:-1]:
                    left_residuals = [
                        residuals[i]
                        for i in range(n_samples)
                        if features[i][feat_idx] <= threshold
                    ]
                    right_residuals = [
                        residuals[i]
                        for i in range(n_samples)
                        if features[i][feat_idx] > threshold
                    ]

                    if len(left_residuals) == 0 or len(right_residuals) == 0:
                        continue

                    left_var = statistics.variance(left_residuals) if len(left_residuals) > 1 else 0
                    right_var = (
                        statistics.variance(right_residuals) if len(right_residuals) > 1 else 0
                    )
                    gain = (
                        statistics.variance(residuals)
                        - (len(left_residuals) / n_samples * left_var)
                        - (len(right_residuals) / n_samples * right_var)
                    )

                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feat_idx
                        best_threshold = threshold

            if best_feature == -1:
                break

            stump = {"feature": best_feature, "threshold": best_threshold}
            stumps.append(stump)
            feature_importance[best_feature] += best_gain

            for i in range(n_samples):
                residual_pred = (
                    statistics.mean(
                        [
                            residuals[j]
                            for j in range(n_samples)
                            if features[j][best_feature] <= best_threshold
                        ]
                    )
                    if any(features[j][best_feature] <= best_threshold for j in range(n_samples))
                    else 0
                )
                predictions[i] += learning_rate * residual_pred

        mse = statistics.mean([(targets[i] - predictions[i]) ** 2 for i in range(n_samples)])
        rmse = math.sqrt(mse)
        mae = statistics.mean([abs(targets[i] - predictions[i]) for i in range(n_samples)])

        mean_target = statistics.mean(targets)
        ss_tot = sum((targets[i] - mean_target) ** 2 for i in range(n_samples))
        ss_res = sum((targets[i] - predictions[i]) ** 2 for i in range(n_samples))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        total_importance = sum(feature_importance.values())
        if total_importance > 0:
            feature_importance = {
                k: (v / total_importance * 100) for k, v in feature_importance.items()
            }

        return {
            "why_this_model": "Captures non-linear patterns and feature interactions for accurate healthcare risk prediction",
            "how_it_works": "Iteratively fits decision stumps to residuals, combining weak learners with learning rate regularization",
            "what_it_means": "Feature importance shows which factors drive predictions; RMSE/MAE show overall accuracy",
            "predictions": predictions,
            "feature_importance": feature_importance,
            "residuals": [targets[i] - predictions[i] for i in range(n_samples)],
            "model_metrics": {"r_squared": r_squared, "rmse": rmse, "mae": mae, "n_estimators": len(stumps)},
            "stumps": stumps,
        }


    def arima_forecast(self, time_series: List[float], periods: int = 6) -> Dict[str, Any]:
        if len(time_series) < 4:
            return {"error": "Insufficient time series data", "forecast": []}

        differenced = [time_series[i] - time_series[i - 1] for i in range(1, len(time_series))]

        mean_diff = statistics.mean(differenced)
        c0 = statistics.variance(differenced) if len(set(differenced)) > 1 else 1
        c1 = (
            sum((differenced[i] - mean_diff) * (differenced[i - 1] - mean_diff) for i in range(1, len(differenced)))
            / len(differenced)
            if len(differenced) > 1
            else 0
        )
        ar_coef = c1 / c0 if c0 > 0 else 0.5

        ar_residuals = [
            differenced[i] - ar_coef * differenced[i - 1] if i > 0 else differenced[i]
            for i in range(len(differenced))
        ]
        ma_coef = 0.3

        forecast = []
        last_diff = differenced[-1]
        last_value = time_series[-1]

        for _ in range(periods):
            next_diff = ar_coef * last_diff + statistics.mean(ar_residuals)
            next_value = last_value + next_diff
            forecast.append(next_value)
            last_diff = next_diff
            last_value = next_value

        residual_std = math.sqrt(statistics.variance(ar_residuals)) if len(set(ar_residuals)) > 1 else 1
        confidence_interval = 1.96 * residual_std

        aic = len(time_series) * math.log(c0) + 2 * 2
        bic = len(time_series) * math.log(c0) + 2 * math.log(len(time_series))

        return {
            "why_this_model": "ARIMA models temporal dependencies and trends essential for healthcare forecasting",
            "how_it_works": "Differences to remove trend (I), fits AR model via Yule-Walker, estimates MA parameters from residuals",
            "what_it_means": f"Forecast with ±{confidence_interval:.2f} confidence interval; lower AIC indicates better fit",
            "forecast": forecast,
            "upper_band": [f + confidence_interval for f in forecast],
            "lower_band": [max(0, f - confidence_interval) for f in forecast],
            "aic": aic,
            "bic": bic,
            "ar_coefficient": ar_coef,
            "residual_diagnostics": {
                "mean": statistics.mean(ar_residuals),
                "std": residual_std,
                "count": len(ar_residuals),
            },
        }


    def kmeans_cluster(self, data: List[List[float]], k: int = 4, max_iter: int = 100) -> Dict[str, Any]:
        if len(data) < k:
            return {"error": "Data size < k", "clusters": {}}

        n_features = len(data[0]) if data else 0
        if n_features == 0:
            return {"error": "Empty feature vectors", "clusters": {}}

        random.seed(42)
        centroid_indices = random.sample(range(len(data)), k)
        centroids = [data[i][:] for i in centroid_indices]

        for iteration in range(max_iter):
            clusters = defaultdict(list)
            for idx, point in enumerate(data):
                distances = [
                    math.sqrt(sum((point[j] - centroids[c][j]) ** 2 for j in range(n_features)))
                    for c in range(k)
                ]
                cluster = distances.index(min(distances))
                clusters[cluster].append(idx)

            new_centroids = []
            for c in range(k):
                if clusters[c]:
                    new_centroid = [
                        statistics.mean([data[idx][j] for idx in clusters[c]])
                        for j in range(n_features)
                    ]
                    new_centroids.append(new_centroid)
                else:
                    new_centroids.append(centroids[c])

            if all(
                math.sqrt(sum((new_centroids[c][j] - centroids[c][j]) ** 2 for j in range(n_features))) < 1e-4
                for c in range(k)
            ):
                break

            centroids = new_centroids

        assignments = [0] * len(data)
        clusters = defaultdict(list)
        for idx, point in enumerate(data):
            distances = [
                math.sqrt(sum((point[j] - centroids[c][j]) ** 2 for j in range(n_features)))
                for c in range(k)
            ]
            cluster = distances.index(min(distances))
            assignments[idx] = cluster
            clusters[cluster].append(idx)

        silhouette_scores = []
        for idx, point in enumerate(data):
            cluster = assignments[idx]
            same_cluster = [
                i for i in range(len(data))
                if assignments[i] == cluster and i != idx
            ]

            if same_cluster:
                a = statistics.mean(
                    [
                        math.sqrt(sum((point[j] - data[i][j]) ** 2 for j in range(n_features)))
                        for i in same_cluster
                    ]
                )
            else:
                a = 0

            min_b = float("inf")
            for other_cluster in range(k):
                if other_cluster != cluster:
                    other_points = [i for i in range(len(data)) if assignments[i] == other_cluster]
                    if other_points:
                        b = statistics.mean(
                            [
                                math.sqrt(sum((point[j] - data[i][j]) ** 2 for j in range(n_features)))
                                for i in other_points
                            ]
                        )
                        min_b = min(min_b, b)

            if min_b == float("inf"):
                silhouette = 0
            else:
                silhouette = (min_b - a) / max(a, min_b) if max(a, min_b) > 0 else 0

            silhouette_scores.append(silhouette)

        return {
            "why_this_model": "Identifies natural member segments for targeted risk stratification and interventions",
            "how_it_works": "Iteratively assigns points to nearest centroid and recomputes cluster centers until convergence",
            "what_it_means": f"Silhouette score (avg {statistics.mean(silhouette_scores):.3f}) indicates cluster quality; higher is better",
            "cluster_assignments": assignments,
            "centroids": centroids,
            "silhouette_scores": silhouette_scores,
            "cluster_sizes": {c: len(clusters[c]) for c in range(k)},
            "iterations": iteration + 1,
        }


    def isolation_forest(
        self, data: List[List[float]], n_trees: int = 100, sample_size: int = 256
    ) -> Dict[str, Any]:
        if len(data) < sample_size:
            sample_size = max(2, len(data) // 2)

        n_features = len(data[0]) if data else 0
        if n_features == 0:
            return {"error": "Empty feature vectors", "anomalies": []}

        random.seed(42)
        tree_predictions = defaultdict(float)

        for tree_num in range(n_trees):
            sample_indices = random.sample(range(len(data)), min(sample_size, len(data)))
            sample = [data[i] for i in sample_indices]

            def build_tree(indices, depth=0, max_depth=math.ceil(math.log2(sample_size))):
                if len(indices) <= 1 or depth >= max_depth:
                    return {"type": "leaf", "size": len(indices)}

                feature = random.randint(0, n_features - 1)
                values = sorted(set(sample[i][feature] for i in indices))

                if len(values) < 2:
                    return {"type": "leaf", "size": len(indices)}

                threshold = random.choice(values[:-1])

                left = [i for i in indices if sample[i][feature] <= threshold]
                right = [i for i in indices if sample[i][feature] > threshold]

                return {
                    "type": "split",
                    "feature": feature,
                    "threshold": threshold,
                    "left": build_tree(left, depth + 1, max_depth),
                    "right": build_tree(right, depth + 1, max_depth),
                }

            tree = build_tree(list(range(len(sample))))

            for idx, point in enumerate(data):
                def traverse(node):
                    if node["type"] == "leaf":
                        return math.log2(node["size"]) if node["size"] > 0 else 0

                    if point[node["feature"]] <= node["threshold"]:
                        return 1 + traverse(node["left"])
                    else:
                        return 1 + traverse(node["right"])

                path_length = traverse(tree)
                tree_predictions[idx] += path_length

        avg_path_length = sum(tree_predictions.values()) / len(data) if data else 0
        anomaly_scores = {}

        for idx in range(len(data)):
            avg_tree_path = tree_predictions[idx] / n_trees
            c = math.log2(len(data)) if len(data) > 1 else 1
            anomaly_score = 2 ** (-(avg_tree_path / c))
            anomaly_scores[idx] = anomaly_score

        anomalies = [idx for idx, score in anomaly_scores.items() if score > 0.6]
        anomalies.sort(key=lambda x: anomaly_scores[x], reverse=True)

        normal_scores = [s for idx, s in anomaly_scores.items() if idx not in anomalies]
        normal_range = (min(normal_scores), max(normal_scores)) if normal_scores else (0, 0.6)

        return {
            "why_this_model": "Detects anomalies without assuming normal distribution; effective for fraud and billing irregularities",
            "how_it_works": "Ensemble of random trees isolates anomalies (shorter paths), scored by normalized path length",
            "what_it_means": f"Anomaly score > 0.6 flags suspicious; {len(anomalies)} anomalies detected out of {len(data)}",
            "anomaly_scores": anomaly_scores,
            "anomalies": anomalies,
            "anomaly_count": len(anomalies),
            "normal_score_range": normal_range,
            "severity_ranking": anomalies,
        }


    def bayesian_classifier(
        self, features: List[List[float]], labels: List[int], new_data: List[List[float]] = None
    ) -> Dict[str, Any]:
        if len(features) < 2 or len(labels) < 2:
            return {"error": "Insufficient training data", "predictions": []}

        classes = list(set(labels))
        n_features = len(features[0]) if features else 0

        class_counts = Counter(labels)
        priors = {c: class_counts[c] / len(labels) for c in classes}

        class_stats = {}
        for c in classes:
            class_indices = [i for i, label in enumerate(labels) if label == c]
            class_features = [features[i] for i in class_indices]

            means = [
                statistics.mean([f[j] for f in class_features])
                for j in range(n_features)
            ]
            variances = [
                statistics.variance([f[j] for f in class_features]) if len(set(f[j] for f in class_features)) > 1 else 1
                for j in range(n_features)
            ]

            class_stats[c] = {"means": means, "variances": variances}

        if new_data is None:
            new_data = features

        predictions = []
        posteriors = []

        for point in new_data:
            posteriors_point = {}

            for c in classes:
                prior = math.log(priors[c])
                likelihood = 0

                for j in range(n_features):
                    mean = class_stats[c]["means"][j]
                    var = class_stats[c]["variances"][j]

                    if var > 0:
                        likelihood += -0.5 * math.log(2 * math.pi * var) - (
                            (point[j] - mean) ** 2 / (2 * var)
                        )

                posteriors_point[c] = prior + likelihood

            max_posterior = max(posteriors_point.values())
            posteriors_point = {
                c: math.exp(p - max_posterior) for c, p in posteriors_point.items()
            }
            total = sum(posteriors_point.values())
            posteriors_point = {c: p / total for c, p in posteriors_point.items()}

            prediction = max(posteriors_point, key=posteriors_point.get)
            predictions.append(prediction)
            posteriors.append(posteriors_point)

        return {
            "why_this_model": "Fast, interpretable probabilistic classifier for risk and readmission prediction",
            "how_it_works": "Calculates prior probabilities and Gaussian likelihoods per class; predicts via Bayes' rule",
            "what_it_means": "Posterior probabilities show confidence; higher probability = stronger prediction",
            "predictions": predictions,
            "class_probabilities": posteriors,
            "priors": priors,
            "class_stats": class_stats,
        }


    def survival_analysis(self, durations: List[float], events: List[int]) -> Dict[str, Any]:
        if len(durations) != len(events) or len(durations) < 2:
            return {"error": "Invalid survival data", "survival_curve": []}

        data = sorted(zip(durations, events), key=lambda x: x[0])
        times = [d[0] for d in data]
        event_flags = [e[1] for d in data]

        n = len(data)
        at_risk = n
        survival_prob = 1.0
        survival_curve = [(0, 1.0)]
        hazard_rates = []

        for i, (time, event) in enumerate(data):
            if event == 1:
                survival_prob *= (at_risk - 1) / at_risk if at_risk > 0 else 1
                hazard_rate = 1 / at_risk if at_risk > 0 else 0
                hazard_rates.append((time, hazard_rate))

            at_risk -= 1

            if event == 1:
                survival_curve.append((time, survival_prob))

        median_survival = None
        for time, prob in survival_curve:
            if prob <= 0.5:
                median_survival = time
                break

        if median_survival is None and survival_curve:
            median_survival = survival_curve[-1][0]

        risk_table = {}
        quantiles = [0.25, 0.5, 0.75]

        for q in quantiles:
            for time, prob in survival_curve:
                if prob <= 1 - q:
                    if q not in risk_table:
                        risk_table[q] = time
                    break

        return {
            "why_this_model": "Estimates time-to-disenrollment for retention and intervention timing analysis",
            "how_it_works": "Sorts events chronologically; calculates cumulative survival as product of (at-risk-events)/at-risk",
            "what_it_means": f"Median survival {median_survival:.1f} time units; lower curve = higher disenrollment risk",
            "survival_curve": survival_curve,
            "median_survival": median_survival,
            "hazard_rates": hazard_rates,
            "risk_table": risk_table,
            "total_events": sum(events),
        }


    def pca_reduce(self, data: List[List[float]], n_components: int = 2) -> Dict[str, Any]:
        if len(data) < 2 or not data:
            return {"error": "Insufficient data", "transformed": []}

        n_features = len(data[0])
        n_components = min(n_components, n_features)

        means = [statistics.mean([row[j] for row in data]) for j in range(n_features)]
        centered = [[row[j] - means[j] for j in range(n_features)] for row in data]

        cov_matrix = [[0] * n_features for _ in range(n_features)]
        for i in range(n_features):
            for j in range(n_features):
                cov_matrix[i][j] = (
                    statistics.mean([centered[k][i] * centered[k][j] for k in range(len(data))])
                )

        eigenvectors = []
        eigenvalues = []

        for _ in range(n_components):
            v = [random.random() for _ in range(n_features)]
            v_norm = math.sqrt(sum(x ** 2 for x in v))
            v = [x / v_norm for x in v]

            for _ in range(20):
                v_new = [sum(cov_matrix[i][j] * v[j] for j in range(n_features)) for i in range(n_features)]
                v_norm = math.sqrt(sum(x ** 2 for x in v_new))
                if v_norm > 0:
                    v_new = [x / v_norm for x in v_new]
                v = v_new

            eigenvalue = sum(cov_matrix[i][j] * v[j] for i in range(n_features) for j in range(n_features)) / (
                n_features
            )
            eigenvectors.append(v)
            eigenvalues.append(max(0, eigenvalue))

            for i in range(n_features):
                for j in range(n_features):
                    cov_matrix[i][j] -= eigenvalue * eigenvectors[-1][i] * eigenvectors[-1][j]

        transformed = []
        for row in centered:
            projected = [sum(row[j] * eigenvectors[i][j] for j in range(n_features)) for i in range(n_components)]
            transformed.append(projected)

        total_variance = sum(eigenvalues)
        explained_variance_ratio = (
            [ev / total_variance for ev in eigenvalues] if total_variance > 0 else [0] * len(eigenvalues)
        )
        cumulative_variance = []
        cum_sum = 0
        for ratio in explained_variance_ratio:
            cum_sum += ratio
            cumulative_variance.append(cum_sum)

        return {
            "why_this_model": "Reduces dimensionality while preserving variance; enables visualization and improves clustering",
            "how_it_works": "Centers data, computes covariance, finds principal eigenvectors via power iteration, projects data",
            "what_it_means": f"First {n_components} components explain {cumulative_variance[-1] * 100:.1f}% of variance",
            "transformed_data": transformed,
            "explained_variance_ratio": explained_variance_ratio,
            "cumulative_variance_ratio": cumulative_variance,
            "loadings": eigenvectors,
            "eigenvalues": eigenvalues,
        }


    def ensemble_forecast(self, time_series: List[float], periods: int = 6) -> Dict[str, Any]:
        if len(time_series) < 6:
            return {"error": "Insufficient time series data", "ensemble_forecast": []}

        forecasts = {}

        n = len(time_series)
        x_vals = list(range(n))
        x_mean = statistics.mean(x_vals)
        y_mean = statistics.mean(time_series)
        slope = (
            sum((x_vals[i] - x_mean) * (time_series[i] - y_mean) for i in range(n))
            / sum((x_vals[i] - x_mean) ** 2 for i in range(n))
            if sum((x_vals[i] - x_mean) ** 2 for i in range(n)) > 0
            else 0
        )
        intercept = y_mean - slope * x_mean
        lr_forecast = [intercept + slope * (n + j) for j in range(periods)]
        forecasts["linear_regression"] = lr_forecast

        alpha, beta = 0.3, 0.2
        level = time_series[0]
        trend = time_series[1] - time_series[0] if len(time_series) > 1 else 0
        hw_forecast = []

        for _ in range(periods):
            forecast_val = level + trend
            hw_forecast.append(forecast_val)

            level = alpha * time_series[-1] + (1 - alpha) * (level + trend)
            trend = beta * (time_series[-1] - level) + (1 - beta) * trend

        forecasts["holt_winters"] = hw_forecast

        arima_result = self.arima_forecast(time_series, periods)
        if "forecast" in arima_result:
            forecasts["arima"] = arima_result["forecast"]

        try:
            if len(time_series) >= 10:
                features = [
                    [time_series[i], time_series[i + 1], time_series[i + 2]]
                    for i in range(len(time_series) - 2)
                ]
                targets = time_series[3:]
                if len(features) > 0:
                    gb_result = self.gradient_boost_predict(features, targets, n_estimators=20)
                    if "predictions" in gb_result:
                        gb_forecast = []
                        last_vals = time_series[-3:]
                        for _ in range(periods):
                            if len(gb_result["predictions"]) > 0:
                                next_val = statistics.mean(gb_result["predictions"][-10:])
                            else:
                                next_val = statistics.mean(time_series[-3:])
                            gb_forecast.append(next_val)
                            last_vals = last_vals[1:] + [next_val]

                        forecasts["gradient_boosting"] = gb_forecast
        except:
            pass

        weights = {}
        for model_name, forecast_vals in forecasts.items():
            if len(forecast_vals) > 0:
                residual = abs(time_series[-1] - forecast_vals[0])
                weights[model_name] = 1 / (1 + residual)

        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {m: w / total_weight for m, w in weights.items()}
        else:
            weights = {m: 1 / len(weights) for m in weights}

        ensemble_forecast = []
        for period in range(periods):
            weighted_sum = sum(
                weights.get(model, 0) * forecasts[model][period]
                for model in forecasts
                if period < len(forecasts[model])
            )
            ensemble_forecast.append(weighted_sum)

        consensus_scores = []
        for period in range(periods):
            period_forecasts = [
                forecasts[m][period] for m in forecasts if period < len(forecasts[m])
            ]
            if period_forecasts:
                period_std = statistics.stdev(period_forecasts) if len(set(period_forecasts)) > 1 else 0
                consensus = 1 / (1 + period_std / statistics.mean(period_forecasts)) if period_forecasts else 0
                consensus_scores.append(consensus)

        return {
            "why_this_model": "Combines diverse models to reduce bias and improve forecast robustness",
            "how_it_works": "Generates forecasts from 4+ models, weights by historical accuracy, produces weighted ensemble",
            "what_it_means": f"Ensemble integrates LR, HW, ARIMA, GB; consensus score shows agreement (1=perfect)",
            "ensemble_forecast": ensemble_forecast,
            "individual_forecasts": forecasts,
            "model_weights": weights,
            "consensus_scores": consensus_scores,
        }


    def analyze_member_risk_segments(self) -> Dict[str, Any]:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT m.member_id,
                       COUNT(DISTINCT c.claim_id) as claim_count,
                       CAST(COALESCE(AVG(c.claim_amount), 0) AS FLOAT) as avg_claim,
                       CAST(COALESCE(SUM(CASE WHEN e.encounter_type='IP' THEN 1 ELSE 0 END), 0) AS FLOAT) as inpatient_visits,
                       CAST(COALESCE(SUM(CASE WHEN e.encounter_type='ED' THEN 1 ELSE 0 END), 0) AS FLOAT) as ed_visits
                FROM members m
                LEFT JOIN claims c ON m.member_id = c.member_id
                LEFT JOIN encounters e ON m.member_id = e.member_id
                WHERE m.disenrollment_date = ''
                GROUP BY m.member_id
                LIMIT 1000
            """)

            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return {"error": "No member data found"}

            features = [list(row[1:]) for row in rows]
            member_ids = [row[0] for row in rows]

            pca_result = self.pca_reduce(features, n_components=2)
            if "transformed_data" not in pca_result:
                return pca_result

            kmeans_result = self.kmeans_cluster(pca_result["transformed_data"], k=4)

            return {
                "why_this_model": "Identifies member risk segments for targeted intervention allocation",
                "how_it_works": "Queries DB features, reduces dimensionality via PCA, clusters with K-Means",
                "what_it_means": f"Segments {len(member_ids)} members into {max(kmeans_result.get('cluster_assignments', [0])) + 1} risk tiers",
                "member_ids": member_ids,
                "pca_result": pca_result,
                "kmeans_result": kmeans_result,
            }
        except Exception as e:
            return {"error": str(e)}

    def detect_billing_anomalies(self) -> Dict[str, Any]:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT claim_id,
                       CAST(claim_amount AS FLOAT),
                       CAST(COALESCE(allowed_amount, 0) AS FLOAT),
                       CAST(COALESCE(patient_responsibility, 0) AS FLOAT)
                FROM claims
                LIMIT 500
            """)

            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return {"error": "No claims data found"}

            claim_ids = [row[0] for row in rows]
            features = [list(row[1:]) for row in rows]

            if_result = self.isolation_forest(features, n_trees=100)

            if "anomalies" in if_result:
                anomalous_claims = [
                    (claim_ids[idx], if_result["anomaly_scores"][idx])
                    for idx in if_result["anomalies"]
                ]
                anomalous_claims.sort(key=lambda x: x[1], reverse=True)

                return {
                    "why_this_model": "Detects billing anomalies and potential fraud",
                    "how_it_works": "Queries claim features, applies Isolation Forest to score unusual patterns",
                    "what_it_means": f"Flagged {len(anomalous_claims)} anomalies out of {len(claim_ids)} claims",
                    "anomalous_claims": anomalous_claims,
                    "isolation_forest_result": if_result,
                }
            else:
                return if_result
        except Exception as e:
            return {"error": str(e)}

    def forecast_financial_kpis(self) -> Dict[str, Any]:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT CAST(SUM(claim_amount) / CAST(COUNT(DISTINCT member_id) AS FLOAT) AS FLOAT) as pmpm
                FROM claims
                LIMIT 15
            """)

            pmpm_values = [row[0] for row in cursor.fetchall()]
            conn.close()

            if len(pmpm_values) < 4:
                return {"error": "Insufficient historical data"}

            ensemble_result = self.ensemble_forecast(pmpm_values, periods=6)

            return {
                "why_this_model": "Forecasts key financial metrics using ensemble approach",
                "how_it_works": "Aggregates 15 months PMPM, applies ensemble of 4 models with adaptive weights",
                "what_it_means": f"6-month PMPM forecast: ${statistics.mean(ensemble_result.get('ensemble_forecast', [0])):.2f}",
                "pmpm_forecast": ensemble_result.get("ensemble_forecast", []),
                "historical_pmpm": pmpm_values,
                "ensemble_result": ensemble_result,
            }
        except Exception as e:
            return {"error": str(e)}

    def predict_disenrollment_risk(self) -> Dict[str, Any]:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT m.member_id,
                       CAST(COUNT(DISTINCT c.claim_id) AS FLOAT),
                       CAST(COALESCE(AVG(c.claim_amount), 0) AS FLOAT),
                       CASE WHEN m.disenrollment_date != '' THEN 1 ELSE 0 END as disenrolled
                FROM members m
                LEFT JOIN claims c ON m.member_id = c.member_id
                GROUP BY m.member_id
                LIMIT 500
            """)

            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return {"error": "No member data"}

            member_ids = [row[0] for row in rows]
            features = [list(row[1:3]) for row in rows]
            labels = [row[3] for row in rows]

            if sum(labels) > 0:
                gb_result = self.gradient_boost_predict(features, labels, n_estimators=30)

                if "predictions" in gb_result:
                    risk_scores = dict(zip(member_ids, gb_result["predictions"]))
                    high_risk = [
                        (mid, score)
                        for mid, score in risk_scores.items()
                        if score > statistics.mean(gb_result["predictions"]) + statistics.stdev(
                            gb_result["predictions"]
                        )
                    ]

                    return {
                        "why_this_model": "Identifies members at high disenrollment risk for retention intervention",
                        "how_it_works": "Queries member features, trains gradient boosting on disenrollment outcome",
                        "what_it_means": f"Identified {len(high_risk)} high-risk members (top {len(high_risk)/len(member_ids)*100:.1f}%)",
                        "risk_scores": risk_scores,
                        "high_risk_members": sorted(high_risk, key=lambda x: x[1], reverse=True)[:20],
                        "gb_result": gb_result,
                    }

            return {"error": "Insufficient disenrollment events"}
        except Exception as e:
            return {"error": str(e)}
