"""
Full research pipeline: run all experiments, then all post-processing (tables, plots, tests).

Usage:
  python -m tools.full_pipeline                      # Full run (paper-ready: N_TEST_USERS, all seeds, both datasets)
  python -m tools.full_pipeline --fast               # Quick run (30 users, 5 seeds, Serendipity only)
  python -m tools.full_pipeline --clean              # Clean all experiment artifacts first, then run pipeline
  python -m tools.full_pipeline --clean --fast --rebuild-gt  # Clean + quick run + rebuild GT

To only clean (no run):  python -m tools.clean_all_experiments
  --dry-run to preview,  --all to also remove ground_truth and split files.

Pipeline includes:
  - Baselines: random, popularity, content_bm25 (strong content-based), two_tower (ANN retrieval), embedding_cosine, itemknn, ease, mf
  - Sanity checks (oracle upper bound, random in candidate pool)
  - Ablations: retrieval (ANN / BM25 / hybrid), pool size (100/300/1000/5000), with/without reranker, filter already-seen, accuracy vs diversity (MMR/xQuAD)
  - Debias coefficient sweep (fast: 1 seed, full: 3 seeds)
  - Post-processing: tables, plots (including Pareto-front for serendipity/novelty), statistical tests, error analysis. See experiments/ABLATIONS.md for A-level checklist.
"""
import argparse
import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.evaluation_config import (
    N_TEST_USERS,
    EVAL_SEEDS,
    N_FAST_USERS,
    FAST_SEEDS,
    FAST_POOL_SIZES,
    ABLATION_POOL_SIZES,
)


def run_step(name: str, cmd: list, allow_fail: bool = False) -> bool:
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)
    print("Command:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=project_root)
    if result.returncode != 0:
        if allow_fail:
            print(f"\n[WARN] Step '{name}' failed (exit {result.returncode}). Continuing.")
            return True
        print(f"\n[ERROR] Step '{name}' failed with exit code {result.returncode}. Stopping.")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run full pipeline (experiments + tables + plots + tests). Use --fast for a quick end-to-end check."
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Quick run: few users, one seed, Serendipity only. Use to verify pipeline without full dataset.",
    )
    parser.add_argument(
        "--skip-experiments",
        action="store_true",
        help="Skip experiment steps (1–5); only run post-processing (build_master, aggregate, tables, plots, etc.).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Run clean_all_experiments first (remove runs, results, master, tables, plots, etc.), then run pipeline.",
    )
    parser.add_argument(
        "--rebuild-gt",
        action="store_true",
        help="Force rebuild ground_truth + split train/val/test via src.create_splits (recommended if GT may be stale/leaky).",
    )
    args = parser.parse_args()

    if args.clean:
        print("Running cleanup (tools.clean_all_experiments)...")
        clean_result = subprocess.run(
            [sys.executable, "-m", "tools.clean_all_experiments"],
            cwd=project_root,
        )
        if clean_result.returncode != 0:
            print("[ERROR] Cleanup failed. Aborting pipeline.")
            return
        print("Cleanup done. Starting pipeline.\n")

    gt_path = project_root / "experiments" / "ground_truth.json"
    split_train_out = project_root / "experiments" / "training_interactions.csv"
    if not args.skip_experiments and (args.rebuild_gt or not gt_path.exists()):
        if args.rebuild_gt and gt_path.exists():
            print("Rebuilding ground_truth.json (--rebuild-gt)...")
        else:
            print("experiments/ground_truth.json not found. Creating standardized splits + GT (no leakage)...")
        
           # (optional but recommended) run preprocess to produce data/processed/interactions_taobao.csv
        preprocess_res = subprocess.run([sys.executable, "-m", "src.preprocess"], cwd=project_root)
        if preprocess_res.returncode != 0:
            print("[WARN] preprocess failed; Taobao splits may be missing. Continuing...")

        # --- 1) Serendipity: build GT + splits (required) ---
        ensure_gt_ser = subprocess.run(
            [
                sys.executable, "-m", "src.create_splits",
                "--csv", "data/serendipity-sac2018/training.csv",
                "--out-train", str(split_train_out),
                "--out-gt", str(gt_path),
            ],
            cwd=project_root,
        )

        if ensure_gt_ser.returncode != 0 or not gt_path.exists():
            print("[ERROR] Could not create ground_truth.json via create_splits (serendipity).")
            print("Please create/restore it manually and re-run.")
            return

        print("Serendipity ground_truth.json created via create_splits.\n")

        # --- 2) Taobao: build GT + splits (only if interactions exist) ---
        taobao_csv = project_root / "data" / "processed" / "interactions_taobao.csv"
        gt_path_taobao = project_root / "experiments" / "ground_truth_taobao.json"
        split_train_out_taobao = project_root / "experiments" / "training_interactions_taobao.csv"
        split_val_out_taobao = project_root / "experiments" / "val_interactions_taobao.csv"
        split_test_out_taobao = project_root / "experiments" / "test_interactions_taobao.csv"
        split_meta_out_taobao = project_root / "experiments" / "split_metadata_taobao.json"

        if taobao_csv.exists():
            ensure_gt_tao = subprocess.run(
                [
                    sys.executable, "-m", "src.create_splits",
                    "--csv", str(taobao_csv),
                    "--random",  # safer: Taobao often has no usable timestamp
                    "--out-train", str(split_train_out_taobao),
                    "--out-val", str(split_val_out_taobao),
                    "--out-test", str(split_test_out_taobao),
                    "--out-gt", str(gt_path_taobao),
                    "--out-meta", str(split_meta_out_taobao),
                ],
                cwd=project_root,
            )

            if ensure_gt_tao.returncode != 0 or not gt_path_taobao.exists():
                print("[ERROR] Could not create Taobao GT/splits via create_splits.")
                print("Taobao experiments will likely fail until this is fixed.")
                return

            print("Taobao ground_truth_taobao.json created via create_splits.\n")
        else:
            print("[WARN] data/processed/interactions_taobao.csv not found -> skipping Taobao GT/splits.\n")
    elif not args.skip_experiments and gt_path.exists():

        try:
            import json
            with open(gt_path, "r", encoding="utf-8") as f:
                gt_raw = json.load(f)
            if isinstance(gt_raw, dict) and gt_raw.get("_split") != "test":
                print("[WARN] ground_truth.json does not look like a test-split GT (missing _split='test').")
                print("       If you see HR@10≈1.0 everywhere, rebuild GT: python -m tools.full_pipeline --rebuild-gt --fast")
        except Exception:
            pass

    fast = args.fast
    n_users = str(N_FAST_USERS) if fast else str(N_TEST_USERS)
    seeds = [str(s) for s in (FAST_SEEDS if fast else EVAL_SEEDS)]
    seeds_arg = ["--seeds"] + seeds
    pool_sizes_list = FAST_POOL_SIZES if fast else ABLATION_POOL_SIZES  
    pool_list = [str(p) for p in pool_sizes_list]

    print("=" * 60)
    print("Cold-Start Recommendation: Full Pipeline" + (" [FAST MODE]" if fast else ""))
    print("=" * 60)
    print(f"Project root: {project_root}")
    if fast:
        print(f"Fast mode: n_users={n_users}, seeds={seeds}, pool_sizes={pool_list}")

    experiment_steps = [
        (
            "1) Run main experiments (baselines + ablation) on Serendipity", #is done
            ["python", "-m", "src.run_all_experiments", "--n-users", n_users] + seeds_arg + ["--dataset", "serendipity"],
        ),
        (
            "1b) Run sanity check baselines (oracle upper bound, random in candidate pool) on Serendipity", #next step - Running ABLATION study (n_users=500, seeds=[42, 7, 123, 2024, 2025, 0, 13, 999, 1234, 777])
            ["python", "-m", "src.run_all_experiments", "--sanity-only", "--n-users", n_users] + seeds_arg + ["--dataset", "serendipity"],
        ),
        (
            "2) Run main experiments on Taobao",
            ["python", "-m", "src.run_all_experiments", "--n-users", n_users] + seeds_arg + ["--dataset", "taobao"],
        ),
        (
            "2b) Run sanity check baselines on Taobao",
            ["python", "-m", "src.run_all_experiments", "--sanity-only", "--n-users", n_users] + seeds_arg + ["--dataset", "taobao"],
        ),
        (
            "3) Retrieval + pool-size ablation (ANN / BM25 / hybrid)",
            ["python", "-m", "src.run_retrieval_ablation", "--n-users", n_users, "--pool-sizes"] + pool_list + seeds_arg + ["--dataset", "serendipity"],
        ),
        (
            "4) Pool size ablation for reranker",
            ["python", "-m", "src.run_ablation_pool_sizes", "--n-users", n_users, "--pool-sizes"] + pool_list + seeds_arg + ["--dataset", "serendipity"],
        ),
        (
            "5) Reranker variants (zeroshot / light / finetuned if present)",
            ["python", "-m", "src.run_reranker_variants", "--n-users", n_users] + seeds_arg + ["--dataset", "serendipity"],
        ),
        (
            "5b) Profile ablation (centroid / summary / last_k)",
            ["python", "-m", "src.run_ablation_profile", "--n-users", n_users] + seeds_arg + ["--dataset", "serendipity"],
        ),
        (
            "5c) Anti-bias ablation (none / popularity / exposure / MMR / xQuAD)",
            ["python", "-m", "src.run_ablation_debias", "--n-users", n_users] + seeds_arg + ["--dataset", "serendipity"],
        ),
        (
            "5e) Filter already-seen ablation (with / without excluding train items from candidates)",
            ["python", "-m", "src.run_ablation_filter_seen", "--n-users", n_users] + seeds_arg + ["--dataset", "serendipity"],
        ),
    ]
    
    if fast:

        sweep_seeds = [seeds[0]] if seeds else ["42"]
        experiment_steps.append(
            (
                "5d) Debias coefficient sweep (fast mode: 1 seed only)",
                ["python", "-m", "src.run_debias_sweep", "--n-users", n_users, "--seeds"] + sweep_seeds + ["--dataset", "serendipity"],
            )
        )
        experiment_steps.append(
            (
                "5f) Two-head reranker sweep (relevance vs novelty, fast mode: 1 seed)",
                ["python", "-m", "tools.run_pareto_sweep", "--run", "--n-users", n_users, "--seeds"] + sweep_seeds + ["--dataset", "serendipity"],
            )
        )
    else:
        sweep_seeds = seeds[:3] if len(seeds) >= 3 else seeds
        experiment_steps.append(
            (
                "5d) Debias coefficient sweep (full mode: 3 seeds)",
                ["python", "-m", "src.run_debias_sweep", "--n-users", n_users, "--seeds"] + sweep_seeds + ["--dataset", "serendipity"],
            )
        )
        experiment_steps.append(
            (
                "5f) Two-head reranker sweep (relevance vs novelty, alpha in [0, 0.25, 0.5, 0.75, 1.0])",
                ["python", "-m", "tools.run_pareto_sweep", "--run", "--n-users", n_users, "--seeds"] + sweep_seeds + ["--dataset", "serendipity"],
            )
        )

    robustness_n_users = "50" if fast else "200"
    robustness_seeds = [seeds[0]] if seeds else ["42"]
    post_steps = [
        ("6) Build master results (per-user metrics, exposure)", ["python", "-m", "tools.build_master_results"]),
        ("7) Aggregate runs (per-model metrics)", ["python", "-m", "tools.aggregate_runs"]),
        ("7b) Counterfactual / bias evaluation (IPS/SNIPS + popularity bias)", ["python", "-m", "tools.ips_counterfactual_eval"]),
        ("7c) Popularity bias analysis (head amplification vs popularity baseline)", ["python", "-m", "tools.popularity_bias_analysis"]),
        ("7d) Segmentation analysis (by bucket + scenario, where we win + why)", ["python", "-m", "tools.segmentation_analysis"]),
        ("7e) Segmentation plots (HR/nDCG by bucket and scenario)", ["python", "-m", "tools.plot_segmentation"]),
        ("8) Generate paper tables (Markdown + LaTeX)", ["python", "-m", "tools.generate_paper_tables"]),
        ("9) Basic plots (HR, nDCG, recall, unique top-1)", ["python", "-m", "tools.plotting"]),
        ("10) Advanced plots (distributions, exposure, top-1 concentration)", ["python", "-m", "tools.advanced_plotting"]),
        ("10b) Quality vs time (pool_size)", ["python", "-m", "tools.plot_quality_vs_time"]),
        ("10c) Trade-off plots (quality vs bias)", ["python", "-m", "tools.tradeoff_plots"]),
        ("10d) Serendipity/novelty Pareto-front plots (accuracy vs diversity/novelty)", ["python", "-m", "tools.plot_serendipity_tradeoff"]),
        ("10e) Multi-objective policy guide (Pareto trajectories + λ recommendations)", ["python", "-m", "tools.plot_multiobjective_policy"]),
        ("10f) Score calibration analysis (comparability, sorting preservation, calibration)", ["python", "-m", "tools.score_calibration_analysis"]),
        ("10g) Few-shot learning curve (quality vs n_train_interactions 0-20)", ["python", "-m", "tools.few_shot_learning_curve"]),
        ("10h) Hardest cold-start analysis (new_users + new_items simultaneously)", ["python", "-m", "tools.hardest_cold_start_analysis"]),
        ("11) Resource report (latency, throughput, quality vs compute)", ["python", "-m", "tools.resource_report"]),
        ("12) Hypothesis analysis (coverage, bias, calibration)", ["python", "-m", "tools.hypothesis_analysis"]),
        ("13) Reranker score analysis", ["python", "-m", "tools.analyze_scores"]),
        ("14) Error analysis (taxonomy + examples)", ["python", "-m", "tools.error_analysis"]),
        ("15) Statistical tests (paired t-test, Wilcoxon)", ["python", "-m", "tools.stat_tests"]),
        ("16) Enhanced statistical tests (CI, effect sizes)", ["python", "-m", "tools.enhanced_stat_tests"]),
        (
            "17) Robustness (noise + temporal shift + sensitivity)",
            ["python", "-m", "tools.robustness_all"]
            + (["--fast"] if fast else [])
            + ["--n-users", robustness_n_users, "--seeds"]
            + robustness_seeds
            + ["--dataset", "serendipity"],
        ),
    ]

    if not args.skip_experiments:
        for step in experiment_steps:
            if step is None:
                continue
            name, cmd = step
            if not run_step(name, cmd):
                return
    else:
        print("\n[--skip-experiments] Skipping steps 1–5c. Ensure experiments/runs.jsonl and results/*.json exist.")

    for i, (name, cmd) in enumerate(post_steps):
        allow_fail = (i == len(post_steps) - 1)  
        if not run_step(name, cmd, allow_fail=allow_fail):
            return

    print("\n" + "=" * 60)
    print("Pipeline completed successfully.")
    print("=" * 60)
    print("Outputs:")
    print("  - experiments/runs.jsonl")
    print("  - results/*.json")
    print("  - experiments/master_results.json")
    print("  - experiments/aggregated_results.{json,csv}")
    print("  - experiments/tables/")
    print("  - experiments/plots/ (PDF, SVG, PNG)")
    print("  - experiments/resources/resource_report.{json,md}")
    print("  - experiments/hypothesis_analysis/")
    print("  - experiments/score_analysis/")
    print("  - experiments/error_analysis/")
    print("  - experiments/stat_tests/")
    print("  - experiments/counterfactual_evaluation/ (IPS/SNIPS, popularity bias)")
    print("  - experiments/segmentation/ (by bucket/scenario, where we win, plots)")
    print("  - experiments/robustness/ (noise, temporal shift, sensitivity)")
    print("  - experiments/multiobjective/ (Pareto trajectories, policy guide)")
    print("  - experiments/score_calibration/ (score comparability, sorting preservation, calibration)")
    print("  - experiments/few_shot_learning/ (learning curve: quality vs n_train_interactions)")
    print("  - experiments/hardest_cold_start/ (new_users + new_items simultaneously)")


if __name__ == "__main__":
    main()
