# src/experiment_manager.py
import time, json, os
from pathlib import Path
from .utils import logger, save_json
from .run_experiment import small_demo_run  # reuse demo or refactor pipeline into callable functions
from .metrics import coverage, intra_list_diversity, compute_item_popularity
import datetime

RESULTS_DIR = Path("results")
EXPERIMENTS_DIR = Path("experiments")
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

def save_metadata(metadata: dict, filename: str):
    p = EXPERIMENTS_DIR / filename
    save_json(p, metadata)
    logger.info("Saved metadata to %s", p)

def run_multiple_seeds(config: dict, seeds=[42, 7, 123], n_users=20):
    summary = []
    for seed in seeds:
        t0 = time.time()
        logger.info("Running seed %s", seed)
        # small_demo_run currently runs full pipeline and returns results dict (modify small_demo_run to return results)
        results = small_demo_run(n_users=n_users)  # should be refactored to return results
        elapsed = time.time() - t0
        # results: dict user->list of dicts with 'item_id'
        all_rec_lists = [[r['item_id'] for r in results[str(u)]] for u in range(n_users)]
        # compute coverage etc (need item embeddings loaded)
        # placeholders:
        cov = coverage(all_rec_lists, config.get("n_items", 49157))
        summary.append({
            "seed": seed,
            "elapsed": elapsed,
            "coverage": cov,
            "n_users": n_users,
            "date": datetime.datetime.utcnow().isoformat()
        })
        fname = f"run_seed_{seed}_{int(time.time())}.json"
        save_json(RESULTS_DIR / fname, {"results": results, "meta": {"seed": seed, "elapsed": elapsed}})
    save_metadata({"config": config, "summary": summary}, "experiment_summary.json")
    return summary
