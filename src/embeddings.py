
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from .config import EMBED_MODEL, EMBEDDINGS_NPY, EMBEDDING_MAP
from .utils import logger, save_json
import json



_EMBEDDER = None

def get_embedder(model_name: str = EMBED_MODEL):
    global _EMBEDDER
    if _EMBEDDER is None:
        logger.info("Loading SentenceTransformer model '%s' (this happens once)...", model_name)
        _EMBEDDER = SentenceTransformer(model_name)
    return _EMBEDDER

def build_embeddings(items: list, model_name=EMBED_MODEL, out_npy=EMBEDDINGS_NPY, map_path=EMBEDDING_MAP):

    out_npy.parent.mkdir(parents=True, exist_ok=True)


    if out_npy.exists() and map_path.exists():
        logger.info("Embeddings file exists, loading from disk: %s", out_npy)
        emb = np.load(out_npy)
        with open(map_path, "r", encoding="utf-8") as f:
            id2idx = json.load(f)
        return emb, id2idx


    model = get_embedder(model_name)
    texts = []
    ids = []
    for it in items:
        text = it.get("description") or it.get("title") or it.get("text") or ""
        texts.append(text)
        ids.append(str(it.get("item_id")))
    logger.info("Generating embeddings for %d items with %s", len(texts), model_name)
    emb = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    np.save(out_npy, emb)
    id2idx = {iid: int(idx) for idx, iid in enumerate(ids)}
    save_json(map_path, id2idx)
    logger.info("Saved embeddings to %s and map to %s", out_npy, map_path)
    return emb, id2idx

def load_embeddings(emb_path=EMBEDDINGS_NPY, map_path=EMBEDDING_MAP):
    emb = np.load(emb_path)
    with open(map_path, "r", encoding="utf-8") as f:
        id2idx = json.load(f)
    return emb, id2idx
