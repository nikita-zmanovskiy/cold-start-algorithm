import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
from scipy import stats

AGGREGATED_JSON = Path("experiments") / "aggregated_results.json"
RUNS_LOG_PATH = Path("experiments") / "runs.jsonl"
OUTPUT_DIR = Path("experiments") / "stat_tests"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_runs() -> List[Dict[str, Any]]:
    if not RUNS_LOG_PATH.exists():
        return []
    
    runs = []
    with open(RUNS_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    runs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return runs


def get_model_key(run: Dict[str, Any]) -> str:
    config = run.get("config", {})
    baseline = config.get("baseline")
    use_reranker = config.get("use_reranker", False)
    
    if baseline:
        return baseline
    elif use_reranker:
        return "ours_with_reranker"
    else:
        return "candidates_only"


def extract_per_user_metrics(runs: List[Dict[str, Any]], model_key: str, metric: str = "hr@10") -> Dict[str, List[float]]:
    seed_metrics = {}
    
    for run in runs:
        if get_model_key(run) != model_key:
            continue
        
        seed = run.get("config", {}).get("seed")
        if seed is None:
            continue
        
        run_id = run.get("run_id")
        if not run_id:
            continue
        
        csv_path = Path("experiments") / f"{run_id}_per_user.csv"
        if not csv_path.exists():
            metrics = run.get("metrics", {})
            metric_data = metrics.get(metric, {})
            if isinstance(metric_data, dict):
                mean_val = metric_data.get("mean")
                if mean_val is not None:
                    seed_metrics[seed] = [mean_val] * run.get("config", {}).get("n_users", 500)
            continue
        
        import csv as csv_module
        per_user_vals = []
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv_module.DictReader(f)
                for row in reader:
                    val = row.get(metric, 0.0)
                    try:
                        per_user_vals.append(float(val))
                    except (ValueError, TypeError):
                        continue
            if per_user_vals:
                seed_metrics[seed] = per_user_vals
        except Exception as e:
            print(f"Warning: Could not load {csv_path}: {e}")
            continue
    
    return seed_metrics


def paired_test(model1_metrics: Dict[str, List[float]], 
                model2_metrics: Dict[str, List[float]],
                metric_name: str = "HR@10") -> Dict[str, Any]:
    common_seeds = set(model1_metrics.keys()) & set(model2_metrics.keys())
    if not common_seeds:
        return {"error": "No common seeds found"}
    
    all_model1 = []
    all_model2 = []
    
    for seed in common_seeds:
        m1_vals = model1_metrics[seed]
        m2_vals = model2_metrics[seed]
        
        min_len = min(len(m1_vals), len(m2_vals))
        all_model1.extend(m1_vals[:min_len])
        all_model2.extend(m2_vals[:min_len])
    
    if len(all_model1) != len(all_model2) or len(all_model1) == 0:
        return {"error": "Mismatched data lengths"}
    
    t_stat, p_value_ttest = stats.ttest_rel(all_model1, all_model2)
    
    try:
        w_stat, p_value_wilcoxon = stats.wilcoxon(all_model1, all_model2)
    except ValueError:
        w_stat, p_value_wilcoxon = None, 1.0
    
    differences = np.array(all_model1) - np.array(all_model2)
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0
    
    return {
        "metric": metric_name,
        "n_pairs": len(all_model1),
        "model1_mean": float(np.mean(all_model1)),
        "model1_std": float(np.std(all_model1)),
        "model2_mean": float(np.mean(all_model2)),
        "model2_std": float(np.std(all_model2)),
        "mean_difference": float(mean_diff),
        "t_statistic": float(t_stat),
        "p_value_ttest": float(p_value_ttest),
        "w_statistic": float(w_stat) if w_stat is not None else None,
        "p_value_wilcoxon": float(p_value_wilcoxon),
        "cohens_d": float(cohens_d),
        "significant_ttest": p_value_ttest < 0.05,
        "significant_wilcoxon": p_value_wilcoxon < 0.05
    }


def run_all_tests():
    runs = load_runs()
    
    model_keys = set(get_model_key(run) for run in runs)
    print(f"Found models: {sorted(model_keys)}")
    
    model_hr10 = {}
    model_ndcg10 = {}
    
    for model_key in model_keys:
        model_hr10[model_key] = extract_per_user_metrics(runs, model_key, "hr@10")
        model_ndcg10[model_key] = extract_per_user_metrics(runs, model_key, "ndcg@10")
    
    baseline_models = ["random", "popularity", "embedding_cosine"]
    baseline_means = {}
    
    for baseline in baseline_models:
        if baseline in model_hr10:
            all_vals = []
            for seed_vals in model_hr10[baseline].values():
                all_vals.extend(seed_vals)
            if all_vals:
                baseline_means[baseline] = np.mean(all_vals)
    
    if not baseline_means:
        print("No baseline models found")
        return
    
    best_baseline = max(baseline_means, key=baseline_means.get)
    print(f"\nBest baseline: {best_baseline} (HR@10 = {baseline_means[best_baseline]:.4f})")
    
    if "ours_with_reranker" not in model_hr10:
        print("Warning: 'ours_with_reranker' not found in runs")
        return
    
    print("\n" + "="*60)
    print("Statistical Tests: Ours vs Best Baseline")
    print("="*60)
    
    hr10_test = paired_test(
        model_hr10["ours_with_reranker"],
        model_hr10[best_baseline],
        "HR@10"
    )
    
    if "error" not in hr10_test:
        print(f"\nHR@10:")
        print(f"  Ours: {hr10_test['model1_mean']:.4f} ± {hr10_test['model1_std']:.4f}")
        print(f"  {best_baseline}: {hr10_test['model2_mean']:.4f} ± {hr10_test['model2_std']:.4f}")
        print(f"  Mean difference: {hr10_test['mean_difference']:.4f}")
        print(f"  Paired t-test: p = {hr10_test['p_value_ttest']:.4f} {'***' if hr10_test['significant_ttest'] else ''}")
        print(f"  Wilcoxon: p = {hr10_test['p_value_wilcoxon']:.4f} {'***' if hr10_test['significant_wilcoxon'] else ''}")
        print(f"  Cohen's d: {hr10_test['cohens_d']:.4f}")
    
    ndcg10_test = paired_test(
        model_ndcg10["ours_with_reranker"],
        model_ndcg10[best_baseline],
        "nDCG@10"
    )
    
    if "error" not in ndcg10_test:
        print(f"\nnDCG@10:")
        print(f"  Ours: {ndcg10_test['model1_mean']:.4f} ± {ndcg10_test['model1_std']:.4f}")
        print(f"  {best_baseline}: {ndcg10_test['model2_mean']:.4f} ± {ndcg10_test['model2_std']:.4f}")
        print(f"  Mean difference: {ndcg10_test['mean_difference']:.4f}")
        print(f"  Paired t-test: p = {ndcg10_test['p_value_ttest']:.4f} {'***' if ndcg10_test['significant_ttest'] else ''}")
        print(f"  Wilcoxon: p = {ndcg10_test['p_value_wilcoxon']:.4f} {'***' if ndcg10_test['significant_wilcoxon'] else ''}")
        print(f"  Cohen's d: {ndcg10_test['cohens_d']:.4f}")
    
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
    
    output_path = OUTPUT_DIR / "stat_test_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results to: {output_path}")


def main():
    print("Running statistical significance tests...")
    try:
        run_all_tests()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
