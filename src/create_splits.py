
import csv
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

from .evaluation_config import (
    SPLIT_RATIOS,
    SPLIT_METHOD,
    COLD_START_SCENARIOS,
    NEW_USER_MAX_TRAIN,
    INTERACTION_BUCKETS,
    GT_PATH,
    SPLIT_METADATA_PATH,
    TEST_CSV_PATH,
    VAL_CSV_PATH,
)
from .utils import logger

TRAINING_CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "serendipity-sac2018" / "training.csv"


def _bucket_label(n: int) -> str:
    for (lo, hi) in INTERACTION_BUCKETS:
        if hi is None:
            if n >= lo:
                return f"{lo}+"
            continue
        if lo <= n <= hi:
            return f"{lo}-{hi}"
    return "0" 


def _parse_ts(val: Optional[str]):

    if val is None or val == "":
        return None
    try:
        return float(val)
    except ValueError:
        return None


def load_interactions(
    path: Path,
    user_col: str = "user_id",
    item_col: str = "item_id",
    time_col: Optional[str] = "timestamp",
) -> List[Dict[str, Any]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            uid = r.get(user_col) or r.get("user") or r.get("userId")
            iid = r.get(item_col) or r.get("item") or r.get("movieId") or r.get("itemId")
            if uid is None or iid is None:
                vals = list(r.values())
                if len(vals) >= 2:
                    uid, iid = str(vals[0]), str(vals[1])
                else:
                    continue
            ts = r.get(time_col) or r.get("timestamp") or r.get("ts")
            rows.append({"user_id": str(uid), "item_id": str(iid), "timestamp": ts})
    return rows


def split_by_time(
    rows: List[Dict[str, Any]],
    ratios: Tuple[float, float, float] = SPLIT_RATIOS,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    train_r, val_r, test_r = ratios
    if abs(train_r + val_r + test_r - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1.0")

    with_ts = [(r, _parse_ts(r.get("timestamp"))) for r in rows]
    with_ts.sort(key=lambda x: (x[1] is None, x[1] or 0))
    n = len(with_ts)
    n_train = int(n * train_r)
    n_val = int(n * val_r)
    n_test = n - n_train - n_val

    train = [r for r, _ in with_ts[:n_train]]
    val = [r for r, _ in with_ts[n_train : n_train + n_val]]
    test = [r for r, _ in with_ts[n_train + n_val :]]
    return train, val, test


def split_random(
    rows: List[Dict[str, Any]],
    ratios: Tuple[float, float, float] = SPLIT_RATIOS,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    import random
    rng = random.Random(seed)
    shuffled = rows.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    n_test = n - n_train - n_val
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


def build_splits(
    interactions_path: Path,
    out_train: Path = TRAINING_CSV_PATH,
    out_val: Path = VAL_CSV_PATH,
    out_test: Path = TEST_CSV_PATH,
    out_gt: Path = GT_PATH,
    out_meta: Path = SPLIT_METADATA_PATH,
    by_time: Optional[bool] = None,
    ratios: Optional[Tuple[float, float, float]] = None,
    seed: int = 42,
    user_col: str = "user_id",
    item_col: str = "item_id",
) -> Dict[str, Any]:
    rows = load_interactions(interactions_path, user_col=user_col, item_col=item_col)
    if not rows:
        raise SystemExit(f"No rows loaded from {interactions_path}")

    if by_time is None:
        by_time = (SPLIT_METHOD == "time_based")
    
    logger.info("Split method: %s (by_time=%s)", SPLIT_METHOD, by_time)
    
    split_ratios = ratios if ratios is not None else SPLIT_RATIOS
    if by_time:
        train, val, test = split_by_time(rows, ratios=split_ratios)
    else:
        train, val, test = split_random(rows, ratios=split_ratios, seed=seed)

    train_items = set(r["item_id"] for r in train)

    train_count_by_user = defaultdict(int)
    for r in train:
        train_count_by_user[r["user_id"]] += 1

    gt = defaultdict(list)
    test_users = set()
    for r in test:
        uid, iid = r["user_id"], r["item_id"]
        gt[uid].append(iid)
        test_users.add(uid)
    gt = {uid: list(dict.fromkeys(items)) for uid, items in gt.items()}


    user_meta = {}
    for uid in test_users:
        n_train = train_count_by_user.get(uid, 0)
        user_meta[uid] = {
            "n_train_interactions": n_train,
            "bucket": _bucket_label(n_train),
            "is_new_user": n_train <= NEW_USER_MAX_TRAIN,
        }

    users_new_items = set()
    for r in test:
        uid, iid = r["user_id"], r["item_id"]
        if iid not in train_items:
            users_new_items.add(uid)

    scenario_to_users = {
        "new_users": [uid for uid in test_users if user_meta[uid]["is_new_user"]],
        "new_items": list(users_new_items),
        "both": list(set(u for u in test_users if user_meta[u]["is_new_user"] or u in users_new_items)),
    }

    bucket_to_users = defaultdict(list)
    for uid in test_users:
        bucket_to_users[user_meta[uid]["bucket"]].append(uid)

    split_metadata = {
        "split_ratios": list(split_ratios),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "n_test_users": len(test_users),
        "user_meta": user_meta,
        "scenario_to_users": scenario_to_users,
        "bucket_to_users": dict(bucket_to_users),
        "train_items": list(train_items),
        "new_user_max_train": NEW_USER_MAX_TRAIN,
    }


    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_val.parent.mkdir(parents=True, exist_ok=True)
    out_gt.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)

    for path, data in [
        (out_train, train),
        (out_val, val),
        (out_test, test),
    ]:
        if not data:
            continue
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["user_id", "item_id", "timestamp"])
            w.writeheader()
            for r in data:
                w.writerow({"user_id": r["user_id"], "item_id": r["item_id"], "timestamp": r.get("timestamp") or ""})

    with open(out_gt, "w", encoding="utf-8") as f:
        json.dump({"data": gt, "_split": "test"}, f, indent=2, ensure_ascii=False)

    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(split_metadata, f, indent=2, ensure_ascii=False)

    logger.info(
        "Splits written: train=%s (%d), val=%s (%d), test=%s (%d), gt=%s, meta=%s",
        out_train, len(train), out_val, len(val), out_test, len(test), out_gt, out_meta,
    )
    return {
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "n_test_users": len(test_users),
        "scenario_sizes": {s: len(scenario_to_users[s]) for s in COLD_START_SCENARIOS},
        "bucket_sizes": {b: len(bucket_to_users[b]) for b in sorted(bucket_to_users.keys(), key=lambda x: (0 if x == "0" else int(x.split("-")[0]) if "-" in x else int(x.replace("+", "") or 21)))},
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Create standardized train/val/test splits for cold-start evaluation.")
    p.add_argument("--csv", required=True, help="Path to interactions CSV (user_id, item_id, optional timestamp)")
    p.add_argument("--by-time", action="store_true", default=True, help="Split by timestamp (default: True)")
    p.add_argument("--random", action="store_true", help="Split randomly instead of by time")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ratios", nargs=3, type=float, default=None, help="Split ratios: train val test (must sum to 1.0)")
    p.add_argument("--out-train", default=None, help="Output path for training.csv")
    p.add_argument("--out-val", default=None, help="Output path for val_interactions.csv")
    p.add_argument("--out-test", default=None, help="Output path for test_interactions.csv")
    p.add_argument("--out-meta", default=None, help="Output path for split_metadata.json")
    p.add_argument("--out-gt", default=None, help="Output path for ground_truth.json")
    args = p.parse_args()
    out_train = Path(args.out_train) if args.out_train else TRAINING_CSV_PATH
    out_val = Path(args.out_val) if args.out_val else VAL_CSV_PATH
    out_test = Path(args.out_test) if args.out_test else TEST_CSV_PATH
    out_meta = Path(args.out_meta) if args.out_meta else SPLIT_METADATA_PATH
    out_gt = Path(args.out_gt) if args.out_gt else GT_PATH
    summary = build_splits(
        Path(args.csv),
        out_train=out_train,
        out_val=out_val,
        out_test=out_test,
        out_gt=out_gt,
        out_meta=out_meta,
        by_time=args.by_time and not args.random,
        seed=args.seed,
        ratios=tuple(args.ratios) if args.ratios else None,
    )
    print("Summary:", json.dumps(summary, indent=2))
