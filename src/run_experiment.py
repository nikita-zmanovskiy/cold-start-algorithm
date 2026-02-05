# run_experiment.py
import random
import time
from pathlib import Path
from .utils import logger, set_seed, save_json
from .preprocess import run_all
from .llm_enrich import LLMEnricher, load_items_from_csv
from .embeddings import build_embeddings, load_embeddings
from .vector_index import build_faiss_index, load_faiss_index
from .graph_builder import build_graph_from_items, add_similarity_edges
from .candidate_retrieval import get_candidates_for_user

from .vark_simulator import build_user_profile_from_minimal
import numpy as np
from .config import HF_MODEL_NAME, HF_DEVICE
from .sanity_checks import sanity_check_rankings
from .config import PROCESSED_DIR, EMBED_MODEL, EMBEDDINGS_NPY, EMBEDDING_MAP, FAISS_INDEX_PATH, RESULTS_DIR

# from .rerank_crossencoder import CrossEncoderReranker
from .rerank_llm import CrossReranker

def small_demo_run(n_users=20):
    set_seed(42)
    # 1. preprocess (best-effort)
    run_all()

    # 2. load items csv produced earlier
    items_csv = PROCESSED_DIR / "items_serendipity.csv"
    if not items_csv.exists():
        logger.error("Items csv not found at %s. Run preprocess properly.", items_csv)
        return
    items = load_items_from_csv(items_csv)
 

    # 3. enrich
    enr = LLMEnricher(backend="heuristic")
    enriched = enr.enrich_items_list(items)  # list of dicts

    # 4. embeddings (load or compute)
    emb, id2idx = build_embeddings(enriched)

    # 5. build faiss index
    index = build_faiss_index(emb)

    # 6. graph
    G = build_graph_from_items(enriched)
    G = add_similarity_edges(G, enriched, emb, id2idx, top_k=5)
    items_meta_dict = { str(it["item_id"]): it for it in enriched }
    # 7. build reranker (CrossEncoder)
    # prepare items_meta_dict for reranker: id -> metadata
    # items_meta_dict = { str(it["item_id"]): it for it in enriched }
    # reranker = CrossEncoderReranker()  # uses default model and CPU
    reranker = CrossReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        items_meta=enriched,
        device="cpu"
    )
    # 8. simulate users and run pipeline
    results = {}
    user_times = []
    for uid in range(n_users):
        
        info = {
            "user_id": uid,
            "goal": f"Looking for interesting movies about topic {uid%5}",
            "time_of_day":"morning",
            "session_len": 5
        }
        profile = build_user_profile_from_minimal(info)

        # candidate retrieval (FAISS + graph)
        candidates = get_candidates_for_user(profile, enriched, index, id2idx, emb, graph=G, pool_size=200)
        logger.info("User %s: candidates after retrieval = %d", uid, len(candidates))

        # ensure we have some candidates; fallback to first items if empty
        if not candidates:
            logger.warning("No candidates for user %s from retrieval; using fallback items.", uid)
            candidates = [str(it["item_id"]) for it in enriched[:50]]

        # rerank with cross-encoder (fast on CPU)
        t1 = time.time()
        reranked = reranker.rerank(profile, candidates, topk=10)
        dt = time.time() - t1
        user_times.append(dt)
        logger.info("User %s reranked in %.2fs", uid, dt)
        results[str(uid)] = reranked

        # logging top result (if present)
        if reranked:
            logger.info("User %s: top-1 = %s", uid, reranked[0])
        else:
            logger.info("User %s: reranker returned no items", uid)
        pop_rec = popularity_baseline(enriched, k=10)
        save_json(RESULTS_DIR / f"popularity_demo_{uid}.json", pop_rec)  # или собрать в одном файле

        # 9. sanity check and save
        sanity = sanity_check_rankings(results)
        save_json(RESULTS_DIR / "demo_rankings.json", results)
        meta = {
        "dataset": "Serendipity-2018",
        "n_items": len(enriched),
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "retrieval_top_k": 400,
        "pool_size": 200,
        "final_k": 10,
        "n_users": n_users,
        "timestamp": time.time(),
        "unique_recommended_items": len(set([it for lst in [ [r['item_id'] for r in v] for v in results.values()] for it in lst])),
    }

    
    user_times = np.array(user_times)
    logger.info(
        "Rerank time per user: mean=%.3fs std=%.3fs",
        user_times.mean(),
        user_times.std()
    )

    save_json(RESULTS_DIR / "last_run_metadata.json", meta)
    save_json(RESULTS_DIR / "demo_sanity.json", sanity)
    logger.info("Demo finished. Results saved to %s", RESULTS_DIR)
    return results

def popularity_baseline(items_list, k=10):
    # if you have popularity ranking or ratings_count
    sorted_by_pop = sorted(items_list, key=lambda x: x.get('pop',0), reverse=True)
    return [str(it['item_id']) for it in sorted_by_pop[:k]]


if __name__ == "__main__":
    small_demo_run()
