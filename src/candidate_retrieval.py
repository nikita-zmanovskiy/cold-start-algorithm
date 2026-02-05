# src/candidate_retrieval.py
from .vector_index import search_index
from .embeddings import load_embeddings
from .config import CANDIDATE_POOL, EMBED_MODEL
from .utils import logger
import numpy as np
from sentence_transformers import SentenceTransformer

# load once
_query_encoder = None
def _get_query_encoder():
    global _query_encoder
    if _query_encoder is None:
        logger.info("Loading SentenceTransformer model for queries: %s", EMBED_MODEL)
        _query_encoder = SentenceTransformer(EMBED_MODEL)
    return _query_encoder

def get_candidates_for_user(user_profile: dict, items_list: list, faiss_index=None, id2idx: dict=None, embeddings: np.ndarray=None, graph=None, pool_size=CANDIDATE_POOL):
    model = _get_query_encoder()
    qtext = user_profile.get("text_profile") or user_profile.get("goal") or ""
    if len(qtext.strip()) == 0:
        qtext = "user_vark:" + user_profile.get("vark", "visual")
    qvec = model.encode([qtext], convert_to_numpy=True)
    D, I = search_index(faiss_index, qvec, top_k=pool_size*2)  # get more, then trim
    candidates = []
    idx2id = {v: k for k, v in id2idx.items()}
    for idx in I[0]:
        item_id = idx2id.get(str(idx)) or idx2id.get(int(idx)) or None
        if item_id:
            candidates.append(str(item_id))
        if len(candidates) >= pool_size:
            break
    # graph expansion unchanged...
    if graph is not None:
        tokens = set(qtext.lower().split())
        for node, data in graph.nodes(data=True):
            if data.get("type") == "entity":
                name = data.get("name", "").lower()
                if any(t in name for t in tokens):
                    for nbr in graph.neighbors(node):
                        if nbr.startswith("item:"):
                            iid = nbr.split("item:")[1]
                            if iid not in candidates:
                                candidates.append(iid)
                                if len(candidates) >= pool_size:
                                    break
            if len(candidates) >= pool_size:
                break
    # popularity fallback
    if len(candidates) < pool_size:
        for it in items_list:
            iid = str(it["item_id"])
            if iid not in candidates:
                candidates.append(iid)
            if len(candidates) >= pool_size:
                break
    logger.info("Candidate pool size for user: %d", len(candidates))
    return candidates[:pool_size]
