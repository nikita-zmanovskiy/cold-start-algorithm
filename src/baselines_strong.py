
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

from .utils import logger
from .config import PROCESSED_DIR, EMBEDDINGS_NPY, EMBEDDING_MAP

BASELINES_DIR = PROCESSED_DIR / "baselines"
EASE_W_PATH = BASELINES_DIR / "ease_W.npy"
EASE_META_PATH = BASELINES_DIR / "ease_meta.json"
MF_P_PATH = BASELINES_DIR / "mf_P.npy"
MF_Q_PATH = BASELINES_DIR / "mf_Q.npy"
MF_META_PATH = BASELINES_DIR / "mf_meta.json"

MAX_EASE_ITEMS = 5000


def _load_training_matrix(
    training_path: Path,
    item_ids_in_catalog: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str], List[str], Dict[str, int], Dict[str, int]]:

    import csv
    from collections import defaultdict
    user_items = defaultdict(list)
    with open(training_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        keys = [k for k in reader.fieldnames or [] if k]
        u_col = next((k for k in keys if "user" in k.lower() and "id" in k.lower()), keys[0] if keys else None)
        i_col = next((k for k in keys if "item" in k.lower() or "movie" in k.lower()), keys[1] if len(keys) > 1 else None)
        for row in reader:
            if not row or not u_col or not i_col:
                continue
            uid, iid = str(row.get(u_col, "")), str(row.get(i_col, ""))
            if not uid or not iid:
                continue
            if item_ids_in_catalog is not None and iid not in item_ids_in_catalog:
                continue
            user_items[uid].append(iid)
    user_ids = sorted(user_items.keys())
    all_items = set()
    for v in user_items.values():
        all_items.update(v)
    item_ids = sorted(all_items)
    u2idx = {u: i for i, u in enumerate(user_ids)}
    i2idx = {i: j for j, i in enumerate(item_ids)}
    n_u, n_i = len(user_ids), len(item_ids)
    X = np.zeros((n_u, n_i), dtype=np.float32)
    for uid, iids in user_items.items():
        ui = u2idx[uid]
        for iid in iids:
            X[ui, i2idx[iid]] = 1.0
    return X, user_ids, item_ids, u2idx, i2idx





def itemknn_recommend(
    user_train_items: List[str],
    embeddings: np.ndarray,
    id2idx: Dict[str, int],
    item_ids_candidate: List[str],
    k: int = 10,
    n_neighbors: int = 50,
) -> List[Dict[str, Any]]:

    if not user_train_items or not item_ids_candidate:
        return []

    train_idxs = []
    for iid in user_train_items:
        idx = id2idx.get(str(iid))
        if idx is not None and idx < len(embeddings):
            train_idxs.append(idx)
    if not train_idxs:
        return []
    train_vecs = embeddings[np.array(train_idxs)]

    norms = np.linalg.norm(train_vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    train_vecs = train_vecs / norms
    user_vec = np.mean(train_vecs, axis=0)
    user_vec = user_vec / (np.linalg.norm(user_vec) + 1e-9)
    train_set = set(str(i) for i in user_train_items)
    scores = []
    for iid in item_ids_candidate:
        if str(iid) in train_set:
            continue
        idx = id2idx.get(str(iid))
        if idx is None or idx >= len(embeddings):
            continue
        v = embeddings[idx]
        v = v / (np.linalg.norm(v) + 1e-9)
        sc = float(np.dot(user_vec, v))
        scores.append({"item_id": str(iid), "score": sc, "method": "itemknn"})
    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores[:k]





def ease_fit(X: np.ndarray, reg: float = 100.0) -> np.ndarray:
    if X.shape[0] == 0 or X.shape[1] == 0:
        logger.warning("EASE: empty training matrix (%s x %s). Skipping fit.", X.shape[0], X.shape[1])
        return None
    n_items = X.shape[1]

    G = X.T.reshape(n_items, -1) @ X  
    G = G.astype(np.float64)
    G[np.diag_indices_from(G)] += float(reg)
    B = G.copy()
    W = np.linalg.solve(G, B)
    np.fill_diagonal(W, 0.0)
    return W.astype(np.float32)


def ease_recommend(
    user_train_items: List[str],
    W: np.ndarray,
    i2idx: Dict[str, int],
    idx2i: Dict[int, str],
    item_ids_candidate: List[str],
    k: int = 10,
) -> List[Dict[str, Any]]:
    if not user_train_items or W.size == 0:
        return []
    n_items = W.shape[0]
    x_u = np.zeros(n_items, dtype=np.float32)
    for iid in user_train_items:
        j = i2idx.get(str(iid))
        if j is not None:
            x_u[j] = 1.0
    pred = x_u @ W
    train_set = set(str(i) for i in user_train_items)
    cand_set = set(str(i) for i in item_ids_candidate)
    scores = []
    for j in range(n_items):
        iid = idx2i.get(j)
        if iid is None or iid not in cand_set or iid in train_set:
            continue
        scores.append({"item_id": str(iid), "score": float(pred[j]), "method": "ease"})
    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores[:k]



def mf_als_fit(
    X: np.ndarray,
    n_factors: int = 64,
    reg: float = 0.01,
    n_iters: int = 15,
    init_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:

    n_users, n_items = X.shape
    rng = np.random.default_rng(init_seed)
    P = rng.standard_normal((n_users, n_factors)).astype(np.float32) * 0.01
    Q = rng.standard_normal((n_items, n_factors)).astype(np.float32) * 0.01
    for it in range(n_iters):
        logger.info("MF ALS iteration %d/%d (users then items)...", it + 1, n_iters)

        for u in range(n_users):
            Xu = X[u]
            mask = Xu > 0
            if not np.any(mask):
                continue
            Q_u = Q[mask]
            A = Q_u.T @ Q_u + reg * np.eye(n_factors, dtype=np.float32)
            b = Q_u.T @ Xu[mask]
            P[u] = np.linalg.solve(A, b)
        logger.info("MF ALS iteration %d/%d: P done, solving for Q...", it + 1, n_iters)

        for i in range(n_items):
            Xi = X[:, i]
            mask = Xi > 0
            if not np.any(mask):
                continue
            P_i = P[mask]
            A = P_i.T @ P_i + reg * np.eye(n_factors, dtype=np.float32)
            b = P_i.T @ Xi[mask]
            Q[i] = np.linalg.solve(A, b)
    return P, Q


def mf_recommend(
    user_train_items: List[str],
    P: np.ndarray,
    Q: np.ndarray,
    u2idx: Dict[str, int],
    i2idx: Dict[str, int],
    idx2i: Dict[int, str],
    item_ids_candidate: List[str],
    k: int = 10,
) -> List[Dict[str, Any]]:

    if not user_train_items or Q.size == 0:
        return []
    train_idxs = [i2idx[iid] for iid in user_train_items if i2idx.get(iid) is not None]
    if not train_idxs:
        return []
    user_vec = np.mean(Q[np.array(train_idxs)], axis=0)
    user_vec = user_vec.astype(np.float32)
    pred = user_vec @ Q.T
    cand_set = set(str(i) for i in item_ids_candidate)
    train_set = set(str(i) for i in user_train_items)
    scores = []
    for j in range(len(Q)):
        iid = idx2i.get(j)
        if iid is None or iid not in cand_set or iid in train_set:
            continue
        scores.append({"item_id": str(iid), "score": float(pred[j]), "method": "mf"})
    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores[:k]




def get_strong_baseline_model(
    baseline_type: str,
    training_path: Path,
    item_ids_catalog: List[str],
    embeddings: Optional[np.ndarray] = None,
    id2idx: Optional[Dict[str, int]] = None,
    ease_reg: float = 100.0,
    mf_factors: int = 64,
    mf_iters: int = 15,
    init_seed: int = 42,
    cache_dir: Optional[Path] = None,
) -> Dict[str, Any]:

    catalog_set = set(str(i) for i in item_ids_catalog)
    cache_dir = cache_dir or BASELINES_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    if baseline_type == "itemknn":
        if embeddings is None or id2idx is None:
            raise ValueError("ItemKNN requires embeddings and id2idx")
        return {"type": "itemknn", "embeddings": embeddings, "id2idx": id2idx}

    if not training_path.exists():
        raise FileNotFoundError(f"Training data required for {baseline_type}: {training_path}")

    X, user_ids, item_ids, u2idx, i2idx = _load_training_matrix(
        training_path, item_ids_in_catalog=item_ids_catalog
    )
    n_u, n_i = X.shape

    if baseline_type == "ease" and n_i > MAX_EASE_ITEMS:
        logger.warning(
            "EASE baseline: %d items in catalog, capping to top-%d by frequency to avoid memory issues.",
            n_i,
            MAX_EASE_ITEMS,
        )

        item_freq = np.asarray(X.sum(axis=0)).ravel()
        keep_idx = np.argsort(item_freq)[-MAX_EASE_ITEMS:]  
        keep_idx = np.sort(keep_idx)
        X = X[:, keep_idx]
        item_ids_reduced = [item_ids[j] for j in keep_idx]
        i2idx = {iid: idx for idx, iid in enumerate(item_ids_reduced)}
        idx2i = {idx: iid for iid, idx in i2idx.items()}
        n_u, n_i = X.shape
        logger.info(
            "EASE baseline: reduced to %d items for training (from %d).",
            n_i,
            len(item_ids),
        )
    else:
        idx2i = {j: i for i, j in i2idx.items()}

    logger.info("Training matrix: %d users x %d items for baseline %s", n_u, n_i, baseline_type)

    if baseline_type == "ease":
        meta_path = cache_dir / "ease_meta.json"
        w_path = cache_dir / "ease_W.npy"
        if w_path.exists() and meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("n_items") == n_i:
                    W = np.load(w_path)
                    logger.info("Loaded cached EASE W from %s", w_path)
                    return {"type": "ease", "W": W, "i2idx": i2idx, "idx2i": idx2i}
            except Exception:
                pass
        W = ease_fit(X, reg=ease_reg)
        np.save(w_path, W)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"n_users": n_u, "n_items": n_i}, f)
        return {"type": "ease", "W": W, "i2idx": i2idx, "idx2i": idx2i}

    if baseline_type == "mf":
        meta_path = cache_dir / "mf_meta.json"
        p_path = cache_dir / "mf_P.npy"
        q_path = cache_dir / "mf_Q.npy"
        if p_path.exists() and q_path.exists() and meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if (
                    meta.get("n_users") == n_u
                    and meta.get("n_items") == n_i
                    and int(meta.get("init_seed", 42)) == int(init_seed)
                ):
                    P = np.load(p_path)
                    Q = np.load(q_path)
                    logger.info("Loaded cached MF from %s", cache_dir)
                    return {"type": "mf", "P": P, "Q": Q, "u2idx": u2idx, "i2idx": i2idx, "idx2i": idx2i}
            except Exception:
                pass
        P, Q = mf_als_fit(X, n_factors=mf_factors, n_iters=mf_iters, init_seed=init_seed)
        np.save(p_path, P)
        np.save(q_path, Q)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"n_users": n_u, "n_items": n_i, "init_seed": int(init_seed)}, f)
        return {"type": "mf", "P": P, "Q": Q, "u2idx": u2idx, "i2idx": i2idx, "idx2i": idx2i}

    raise ValueError(f"Unknown strong baseline: {baseline_type}. Use itemknn, ease, mf.")


def strong_baseline_recommend(
    baseline_type: str,
    model: Dict[str, Any],
    user_id: str,
    user_train_items: List[str],
    item_ids_candidate: List[str],
    k: int = 10,
) -> List[Dict[str, Any]]:
    cand = list(item_ids_candidate)
    if baseline_type == "itemknn":
        return itemknn_recommend(
            user_train_items,
            model["embeddings"],
            model["id2idx"],
            cand,
            k=k,
        )
    if baseline_type == "ease":
        return ease_recommend(
            user_train_items,
            model["W"],
            model["i2idx"],
            model["idx2i"],
            cand,
            k=k,
        )
    if baseline_type == "mf":
        return mf_recommend(
            user_train_items,
            model["P"],
            model["Q"],
            model["u2idx"],
            model["i2idx"],
            model["idx2i"],
            cand,
            k=k,
        )
    raise ValueError(f"Unknown strong baseline: {baseline_type}")
