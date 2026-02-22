import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from scipy import stats

MASTER_JSON = Path("experiments") / "master_results.json"
OUTPUT_DIR = Path("experiments") / "stat_tests"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_master():
    if not MASTER_JSON.exists():
        raise FileNotFoundError(f"Master results not found. Run tools/build_master_results.py first.")
    with open(MASTER_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def bootstrap_ci(data: List[float], n_bootstrap: int = 1000, confidence: float = 0.95) -> tuple:
    if not data or len(data) == 0:
        return None, None
    
    n = len(data)
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return float(lower), float(upper)


def paired_bootstrap_pvalue(
    model1_data: List[float],
    model2_data: List[float],
    n_bootstrap: int = 1000
) -> float:

    if len(model1_data) != len(model2_data) or len(model1_data) == 0:
        return 1.0
    
    differences = np.array(model1_data) - np.array(model2_data)
    observed_diff = np.mean(differences)
    
    n = len(differences)
    bootstrap_diffs = []
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        bootstrap_diff = np.mean(differences[indices])
        bootstrap_diffs.append(bootstrap_diff)
    
    p_value_one_tailed = np.mean(np.array(bootstrap_diffs) <= 0)
    p_value_two_tailed = np.mean(np.abs(np.array(bootstrap_diffs)) >= abs(observed_diff))
    
    return float(p_value_two_tailed)


def comprehensive_statistical_test(model1_data: List[float], model2_data: List[float], metric_name: str = "HR@10") -> Dict[str, Any]:
    if len(model1_data) != len(model2_data) or len(model1_data) == 0:
        return {"error": "Mismatched data lengths"}
    
    differences = np.array(model1_data) - np.array(model2_data)
    mean_diff = np.mean(differences)
    
    model1_mean = float(np.mean(model1_data))
    model2_mean = float(np.mean(model2_data))
    
    relative_improvement_pct = None
    if model2_mean != 0:
        relative_improvement_pct = float((mean_diff / model2_mean) * 100.0)
    
    t_stat, p_value_ttest = stats.ttest_rel(model1_data, model2_data)
    
    try:
        w_stat, p_value_wilcoxon = stats.wilcoxon(model1_data, model2_data)
    except ValueError:
        w_stat, p_value_wilcoxon = None, 1.0
    
    std_diff = np.std(differences, ddof=1)
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0
    
    ci_lower, ci_upper = bootstrap_ci(differences.tolist())
    
    p_value_bootstrap = paired_bootstrap_pvalue(model1_data, model2_data)
    
    return {
        "metric": metric_name,
        "n_pairs": len(model1_data),
        "model1_mean": model1_mean,
        "model1_std": float(np.std(model1_data)),
        "model2_mean": model2_mean,
        "model2_std": float(np.std(model2_data)),
        "mean_difference": float(mean_diff),
        "mean_difference_ci_95_lower": ci_lower,
        "mean_difference_ci_95_upper": ci_upper,
        "relative_improvement_pct": relative_improvement_pct,
        "t_statistic": float(t_stat),
        "p_value_ttest": float(p_value_ttest),
        "p_value_bootstrap": p_value_bootstrap,
        "w_statistic": float(w_stat) if w_stat is not None else None,
        "p_value_wilcoxon": float(p_value_wilcoxon),
        "cohens_d": float(cohens_d),
        "significant_ttest": p_value_ttest < 0.05,
        "significant_bootstrap": p_value_bootstrap < 0.05,
        "significant_wilcoxon": p_value_wilcoxon < 0.05,
        "effect_size_interpretation": "large" if abs(cohens_d) > 0.8 else "medium" if abs(cohens_d) > 0.5 else "small"
    }


def run_comprehensive_tests():
    master = load_master()
    
    runs_by_model = {}
    
    for run in master["runs"]:
        config = run.get("config", {})
        baseline = config.get("baseline")
        use_reranker = config.get("use_reranker", False)
        
        if baseline:
            model_key = baseline
        elif use_reranker:
            model_key = "ours_with_reranker"
        else:
            model_key = "candidates_only"
        
        if model_key not in runs_by_model:
            runs_by_model[model_key] = []
        runs_by_model[model_key].append(run)
    
    baseline_models = ["random", "popularity", "embedding_cosine"]
    baseline_hr10_means = {}
    
    for baseline in baseline_models:
        if baseline in runs_by_model:
            all_hr = []
            for run in runs_by_model[baseline]:
                hr_per_user = run.get("metrics", {}).get("hr@10", {}).get("per_user", [])
                all_hr.extend(hr_per_user)
            if all_hr:
                baseline_hr10_means[baseline] = np.mean(all_hr)
    
    if not baseline_hr10_means:
        print("No baseline models found")
        return
    
    best_baseline = max(baseline_hr10_means, key=baseline_hr10_means.get)
    print(f"Best baseline: {best_baseline} (HR@10 = {baseline_hr10_means[best_baseline]:.4f})")
    
    if "ours_with_reranker" not in runs_by_model:
        print("Warning: 'ours_with_reranker' not found")
        return
    
    ours_runs = runs_by_model["ours_with_reranker"]
    baseline_runs = runs_by_model[best_baseline]
    
    ours_hr10_all = []
    ours_ndcg10_all = []
    baseline_hr10_all = []
    baseline_ndcg10_all = []
    
    min_runs = min(len(ours_runs), len(baseline_runs))
    
    for i in range(min_runs):
        ours_hr = ours_runs[i].get("metrics", {}).get("hr@10", {}).get("per_user", [])
        ours_ndcg = ours_runs[i].get("metrics", {}).get("ndcg@10", {}).get("per_user", [])
        baseline_hr = baseline_runs[i].get("metrics", {}).get("hr@10", {}).get("per_user", [])
        baseline_ndcg = baseline_runs[i].get("metrics", {}).get("ndcg@10", {}).get("per_user", [])
        
        min_len = min(len(ours_hr), len(baseline_hr), len(ours_ndcg), len(baseline_ndcg))
        ours_hr10_all.extend(ours_hr[:min_len])
        ours_ndcg10_all.extend(ours_ndcg[:min_len])
        baseline_hr10_all.extend(baseline_hr[:min_len])
        baseline_ndcg10_all.extend(baseline_ndcg[:min_len])
    
    print("\n" + "="*60)
    print("Comprehensive Statistical Tests: Ours vs Best Baseline")
    print("="*60)
    
    hr10_test = comprehensive_statistical_test(ours_hr10_all, baseline_hr10_all, "HR@10")
    ndcg10_test = comprehensive_statistical_test(ours_ndcg10_all, baseline_ndcg10_all, "nDCG@10")
    
    if "error" not in hr10_test:
        print(f"\nHR@10:")
        print(f"  Ours: {hr10_test['model1_mean']:.4f} ± {hr10_test['model1_std']:.4f}")
        print(f"  {best_baseline}: {hr10_test['model2_mean']:.4f} ± {hr10_test['model2_std']:.4f}")
        print(f"  Mean difference (Δ): {hr10_test['mean_difference']:.4f}")
        if hr10_test.get('relative_improvement_pct') is not None:
            print(f"  Relative improvement: {hr10_test['relative_improvement_pct']:.2f}%")
        print(f"  95% CI for difference: [{hr10_test['mean_difference_ci_95_lower']:.4f}, {hr10_test['mean_difference_ci_95_upper']:.4f}]")
        print(f"  Paired t-test: t={hr10_test['t_statistic']:.4f}, p={hr10_test['p_value_ttest']:.4f} {'***' if hr10_test['significant_ttest'] else ''}")
        print(f"  Paired bootstrap: p={hr10_test['p_value_bootstrap']:.4f} {'***' if hr10_test['significant_bootstrap'] else ''}")
        w_stat_str = f"{hr10_test['w_statistic']:.4f}" if hr10_test['w_statistic'] is not None else "N/A"
        print(f"  Wilcoxon: W={w_stat_str}, p={hr10_test['p_value_wilcoxon']:.4f} {'***' if hr10_test['significant_wilcoxon'] else ''}")
        print(f"  Cohen's d: {hr10_test['cohens_d']:.4f} ({hr10_test['effect_size_interpretation']})")
        print(f"  n_pairs: {hr10_test['n_pairs']}")
    
    if "error" not in ndcg10_test:
        print(f"\nnDCG@10:")
        print(f"  Ours: {ndcg10_test['model1_mean']:.4f} ± {ndcg10_test['model1_std']:.4f}")
        print(f"  {best_baseline}: {ndcg10_test['model2_mean']:.4f} ± {ndcg10_test['model2_std']:.4f}")
        print(f"  Mean difference (ΔnDCG): {ndcg10_test['mean_difference']:.4f}")
        if ndcg10_test.get('relative_improvement_pct') is not None:
            print(f"  Relative improvement: {ndcg10_test['relative_improvement_pct']:.2f}%")
        print(f"  95% CI for difference: [{ndcg10_test['mean_difference_ci_95_lower']:.4f}, {ndcg10_test['mean_difference_ci_95_upper']:.4f}]")
        print(f"  Paired t-test: t={ndcg10_test['t_statistic']:.4f}, p={ndcg10_test['p_value_ttest']:.4f} {'***' if ndcg10_test['significant_ttest'] else ''}")
        print(f"  Paired bootstrap: p={ndcg10_test['p_value_bootstrap']:.4f} {'***' if ndcg10_test['significant_bootstrap'] else ''}")
        w_stat_str = f"{ndcg10_test['w_statistic']:.4f}" if ndcg10_test['w_statistic'] is not None else "N/A"
        print(f"  Wilcoxon: W={w_stat_str}, p={ndcg10_test['p_value_wilcoxon']:.4f} {'***' if ndcg10_test['significant_wilcoxon'] else ''}")
        print(f"  Cohen's d: {ndcg10_test['cohens_d']:.4f} ({ndcg10_test['effect_size_interpretation']})")
        print(f"  n_pairs: {ndcg10_test['n_pairs']}")
    
    def convert_for_json(obj):
        if isinstance(obj, dict):
            return {str(k): convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_for_json(item) for item in obj]
        elif isinstance(obj, bool):
            return int(obj)
        elif isinstance(obj, (np.integer, np.int_, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return int(bool(obj))
        elif obj is None:
            return None
        elif isinstance(obj, (int, float, str)):
            return obj
        else:
            try:
                return float(obj)
            except (ValueError, TypeError):
                try:
                    return str(obj)
                except:
                    return None
    
    results = {
        "best_baseline": best_baseline,
        "hr@10": convert_for_json(hr10_test),
        "ndcg@10": convert_for_json(ndcg10_test)
    }
    
    output_path = OUTPUT_DIR / "comprehensive_stat_test_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved comprehensive results to: {output_path}")


def main():
    print("Running comprehensive statistical significance tests...")
    try:
        run_comprehensive_tests()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
