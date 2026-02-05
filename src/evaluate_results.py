# src/evaluate_results.py
import json
import csv
from pathlib import Path
import numpy as np
import logging
from typing import Dict, Any, List

logger = logging.getLogger("evaluate_results")
if not logger.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
    logger.setLevel(logging.INFO)

from src.evaluate import hr_at_k, ndcg_at_k

RESULTS_DIR = Path("results")
GT_PATH = Path("experiments") / "ground_truth.json"
OUT_DIR = Path("experiments")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_json_safe(p: Path):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        logger.warning("JSON decode error for %s", p)
        return None

def write_json(p: Path, data: Any):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def normalize_recs(raw):
    """
    Normalize different possible formats into: list of dicts [{item_id:.., score:.., ...}, ...]
    Acceptable inputs:
      - None -> []
      - dict -> (maybe it's already user->list map) -- handled outside
      - list of dicts -> return as is
      - list of strings (ids) -> convert to [{'item_id': id}, ...]
      - single string -> [{'item_id': string}]
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        if not raw:
            return []
        # list of dicts?
        if isinstance(raw[0], dict):
            return raw
        # list of ids?
        if isinstance(raw[0], (str, int)):
            return [{"item_id": str(x)} for x in raw]
        # fallback: convert to strings
        return [{"item_id": str(x)} for x in raw]
    if isinstance(raw, dict):
        # unexpected single dict -> wrap
        return [raw]
    # single primitive (str/int)
    return [{"item_id": str(raw)}]

def extract_rec_ids(rec_list: List[Dict[str,Any]]):
    return [str(r.get("item_id")) for r in (rec_list or [])]

def evaluate_single(results: Dict[str, Any], gt: Dict[str, List[str]], k=10):
    rows = []
    hr_vals = []
    ndcg_vals = []
    for uid, raw_recs in results.items():
        recs = normalize_recs(raw_recs)
        pos = gt.get(str(uid), [])
        try:
            if pos and isinstance(pos, list) and isinstance(pos[0], (str, int)):
                pos = [{"item_id": str(x)} for x in pos]
            # hr_at_k / ndcg_at_k expect 'recommended' as list of dicts
            hr = hr_at_k(pos, recs, k)
            ndcg = ndcg_at_k(pos, recs, k)
        except Exception as e:
            logger.exception("Metric calc failed for user %s: %s", uid, e)
            hr = 0.0
            ndcg = 0.0
        rec_ids = extract_rec_ids(recs)
        rows.append({"user": uid, f"hr@{k}": hr, f"ndcg@{k}": ndcg, "rec_ids": " ".join(rec_ids[:k])})
        hr_vals.append(hr); ndcg_vals.append(ndcg)
    summary = {
        "n_users": len(rows),
        "hr_mean": float(np.mean(hr_vals)) if hr_vals else None,
        "hr_std": float(np.std(hr_vals)) if hr_vals else None,
        "ndcg_mean": float(np.mean(ndcg_vals)) if ndcg_vals else None,
        "ndcg_std": float(np.std(ndcg_vals)) if ndcg_vals else None
    }
    return rows, summary

def auto_create_synthetic_gt(results_file: Path):
    logger.info("Creating synthetic ground-truth from top-1 of %s", results_file)
    r = load_json_safe(results_file)
    if not r:
        raise SystemExit("Cannot create synthetic GT: results file missing or invalid.")
    gt = {}
    for uid, recs in r.items():
        recs_norm = normalize_recs(recs)
        if recs_norm:
            gt[str(uid)] = [str(recs_norm[0].get("item_id"))]
        else:
            gt[str(uid)] = []
    wrapper = {"_synthetic": True, "data": gt}
    write_json(GT_PATH, wrapper)
    return wrapper

def load_gt_struct():
    raw = load_json_safe(GT_PATH)
    if raw is None:
        demo = RESULTS_DIR / "demo_rankings.json"
        if demo.exists():
            wrapper = auto_create_synthetic_gt(demo)
            return wrapper
        else:
            raise SystemExit("Ground-truth missing and demo_rankings.json not found to create synthetic GT.")
    if isinstance(raw, dict) and "_synthetic" in raw and "data" in raw:
        return raw
    if isinstance(raw, dict):
        return {"_synthetic": False, "data": raw}
    raise SystemExit("Unexpected ground-truth format in {}".format(GT_PATH))

def gather_result_files():
    files = sorted([p for p in RESULTS_DIR.glob("*.json") if p.is_file()])
    # prefer demo_rankings.json first
    files_sorted = sorted(files, key=lambda p: (p.name!="demo_rankings.json", p.name))
    return files_sorted

def main(k=10):
    gt_wrapper = load_gt_struct()
    gt = gt_wrapper["data"]
    synthetic = gt_wrapper.get("_synthetic", False)
    if synthetic:
        logger.warning("Using SYNTHETIC ground-truth generated from top-1 (not for paper).")

    result_files = gather_result_files()
    if not result_files:
        raise SystemExit("No result JSON files found in results/")

    agg_rows = []
    aggregated_summary = {}
    for rf in result_files:
        logger.info("Evaluating %s", rf)
        results = load_json_safe(rf)
        if not results or not isinstance(results, dict):
            logger.warning("Skipping %s (not a dict of user->recs)", rf)
            continue
        rows, summary = evaluate_single(results, gt, k=k)
        csv_path = OUT_DIR / f"eval_{rf.stem}_per_user.csv"
        if rows:
            keys = list(rows[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)
        json_path = OUT_DIR / f"eval_{rf.stem}_summary.json"
        write_json(json_path, {"file": str(rf), "summary": summary, "synthetic_gt": synthetic})
        logger.info("Saved %s and %s", csv_path, json_path)
        aggregated_summary[rf.name] = summary
        agg_rows.append({
            "result_file": rf.name,
            "n_users": summary["n_users"],
            "hr_mean": summary["hr_mean"],
            "hr_std": summary["hr_std"],
            "ndcg_mean": summary["ndcg_mean"],
            "ndcg_std": summary["ndcg_std"]
        })

    agg_csv = OUT_DIR / "eval_aggregated_results.csv"
    if agg_rows:
        keys = list(agg_rows[0].keys())
        with open(agg_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(agg_rows)
    write_json(OUT_DIR / "eval_aggregated_summary.json", aggregated_summary)
    logger.info("Wrote aggregated CSV: %s", agg_csv)
    logger.info("Done.")

if __name__ == "__main__":
    main(k=10)
