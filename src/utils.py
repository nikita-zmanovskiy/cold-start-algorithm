# utils.py
import json
import logging
from pathlib import Path
import random
from sentence_transformers import SentenceTransformer
import numpy as np

from .config import EMBED_MODEL
 

logger = logging.getLogger("coldstart")
if not logger.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
    logger.setLevel(logging.INFO)

def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


# Global embedder singleton (loads once)
_EMBEDDER = None
_EMBEDDER_NAME = None

def get_embedder(model_name: str = None):
    """
    Return a shared SentenceTransformer instance. Uses EMBED_MODEL from config by default.
    Ensures the same model_name is used across the project to prevent double-loading.
    """
    global _EMBEDDER, _EMBEDDER_NAME
    model_name = model_name or EMBED_MODEL
    # normalize name to canonical form
    model_name_norm = model_name if isinstance(model_name, str) else str(model_name)
    if _EMBEDDER is not None and _EMBEDDER_NAME == model_name_norm:
        return _EMBEDDER
    if _EMBEDDER is not None and _EMBEDDER_NAME != model_name_norm:
        # if a different model was requested earlier, warn and reuse the first one
        logger.info("get_embedder: different model requested (%s) but embedder already loaded (%s). Reusing existing.", model_name_norm, _EMBEDDER_NAME)
        return _EMBEDDER
    logger.info("Loading SentenceTransformer model '%s' (this happens once)...", model_name_norm)
    _EMBEDDER = SentenceTransformer(model_name_norm)
    _EMBEDDER_NAME = model_name_norm
    return _EMBEDDER