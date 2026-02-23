import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

from .run_experiment import run_with_logging
from .evaluation_config import get_eval_paths


RUNS_LOG = Path("experiments") / "runs.jsonl"
RESULTS_DIR = Path("results")
EXPERIMENTS_DIR = Path("experiments")


def load_run_record(run_id: str, runs_log: Path = RUNS_LOG) -> Dict[str, Any]:
    if not runs_log.exists():
        return {}
    last = {}
    with open(runs_log, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("run_id") == run_id:
                last = rec
    return last


def read_per_user_csv(run_id: str) -> Dict[str, Dict[str, float]]:
    p = EXPERIMENTS_DIR / f"{run_id}_per_user.csv"
    if not p.exists():
        return {}
    out = {}
    with open(p, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            uid = str(row.get("user") or row.get("user_id") or "")
            if not uid:
                continue
            def fnum(x):
                try:
                    return float(x)
                except Exception:
                    return 0.0
            out[uid] = {
                "hr": fnum(row.get("hr@10") or row.get("hr@10.0") or row.get("hr")),
                "ndcg": fnum(row.get("ndcg@10") or row.get("ndcg")),
                "mrr": fnum(row.get("mrr@10") or row.get("mrr")),
                "map": fnum(row.get("map@10") or row.get("map")),
            }
    return out


def paired_bootstrap_pvalue(
    a: Dict[str, Dict[str, float]],
    b: Dict[str, Dict[str, float]],
    key: str = "ndcg",
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[str, float]:
    users = sorted(set(a.keys()) & set(b.keys()))
    if not users:
        return {"mean_diff": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "p_one_sided": 1.0}

    diffs = np.array([a[u][key] - b[u][key] for u in users], dtype=float)
    mean_diff = float(diffs.mean())

    rng = np.random.default_rng(seed)
    boots = []
    n = len(diffs)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(float(diffs[idx].mean()))
    boots = np.array(boots, dtype=float)

    ci_lo = float(np.percentile(boots, 2.5))
    ci_hi = float(np.percentile(boots, 97.5))

    # one-sided: P(diff <= 0) under bootstrap
    p_one_sided = float(np.mean(boots <= 0.0))
    return {"mean_diff": mean_diff, "ci_lo": ci_lo, "ci_hi": ci_hi, "p_one_sided": p_one_sided}


def run_fewshot_curve(
    dataset: str,
    n_users: int,
    seeds: List[int],
    caps: List[int],
    baseline: str = None,
    retrieval_mode: str = "hybrid",
):
    out_rows = []

    for seed in seeds:
        # main method
        for cap in caps:
            run_id = f"fewshot_{dataset}_cap{cap}_seed{seed}_n{n_users}"
            cfg = {
                "baseline": None,
                "use_reranker": True,
                "retrieval_mode": retrieval_mode,
                "fewshot_train_cap": int(cap),
                "eval_min_train_interactions": int(cap),  # users must have >= cap history for this point
                "dataset": dataset,
            }
            run_with_logging(run_id=run_id, n_users=n_users, seed=seed, config=cfg, dataset=dataset)

            rec = load_run_record(run_id)
            m = (rec.get("metrics") or {})
            out_rows.append({
                "run_id": run_id,
                "dataset": dataset,
                "seed": seed,
                "cap": cap,
                "method": "ours",
                "hr_mean": (m.get("hr@10") or {}).get("mean"),
                "hr_ci_lo": (m.get("hr@10") or {}).get("ci_95_lower"),
                "hr_ci_hi": (m.get("hr@10") or {}).get("ci_95_upper"),
                "ndcg_mean": (m.get("ndcg@10") or {}).get("mean"),
                "ndcg_ci_lo": (m.get("ndcg@10") or {}).get("ci_95_lower"),
                "ndcg_ci_hi": (m.get("ndcg@10") or {}).get("ci_95_upper"),
                "mrr_mean": (m.get("mrr@10") or {}).get("mean"),
                "map_mean": (m.get("map@10") or {}).get("mean"),
                "n_users_logged": (rec.get("diagnostics") or {}).get("n_users"),
            })

            # optional baseline comparison + paired significance
            if baseline:
                run_id_b = f"fewshot_{dataset}_cap{cap}_seed{seed}_n{n_users}_base_{baseline}"
                cfg_b = {
                    "baseline": baseline,
                    "use_reranker": False,
                    "retrieval_mode": retrieval_mode,
                    "fewshot_train_cap": int(cap),
                    "eval_min_train_interactions": int(cap),
                    "dataset": dataset,
                }
                run_with_logging(run_id=run_id_b, n_users=n_users, seed=seed, config=cfg_b, dataset=dataset)

                A = read_per_user_csv(run_id)     # ours
                B = read_per_user_csv(run_id_b)   # baseline

                sig_ndcg = paired_bootstrap_pvalue(A, B, key="ndcg", seed=seed)
                sig_hr = paired_bootstrap_pvalue(A, B, key="hr", seed=seed)

                out_rows.append({
                    "run_id": run_id_b,
                    "dataset": dataset,
                    "seed": seed,
                    "cap": cap,
                    "method": f"baseline:{baseline}",
                    "hr_mean": load_run_record(run_id_b).get("metrics", {}).get("hr@10", {}).get("mean"),
                    "ndcg_mean": load_run_record(run_id_b).get("metrics", {}).get("ndcg@10", {}).get("mean"),
                    "paired_ndcg_mean_diff": sig_ndcg["mean_diff"],
                    "paired_ndcg_ci_lo": sig_ndcg["ci_lo"],
                    "paired_ndcg_ci_hi": sig_ndcg["ci_hi"],
                    "paired_ndcg_p_one_sided": sig_ndcg["p_one_sided"],
                    "paired_hr_mean_diff": sig_hr["mean_diff"],
                    "paired_hr_ci_lo": sig_hr["ci_lo"],
                    "paired_hr_ci_hi": sig_hr["ci_hi"],
                    "paired_hr_p_one_sided": sig_hr["p_one_sided"],
                })

    out_path = EXPERIMENTS_DIR / f"paper_{dataset}_fewshot_curve.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in out_rows for k in r.keys()})
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(out_rows)

    print("Wrote:", out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="movielens", choices=["serendipity", "taobao", "movielens"])
    p.add_argument("--n-users", type=int, default=50)
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--caps", nargs="+", type=int, default=list(range(0, 21)))
    p.add_argument("--baseline", type=str, default=None, help="Optional baseline name for paired tests, e.g. embedding_cosine / popularity / mf / ease / itemknn")
    p.add_argument("--retrieval-mode", type=str, default="hybrid", choices=["ann", "bm25", "hybrid"])
    args = p.parse_args()

    run_fewshot_curve(
        dataset=args.dataset,
        n_users=args.n_users,
        seeds=args.seeds,
        caps=args.caps,
        baseline=args.baseline,
        retrieval_mode=args.retrieval_mode,
    )


if __name__ == "__main__":
    main()