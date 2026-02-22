
import faiss
import numpy as np
from pathlib import Path
from .config import FAISS_INDEX_PATH
from .utils import logger

def build_faiss_index(embeddings: np.ndarray, index_path: Path = FAISS_INDEX_PATH):
    xb = embeddings.astype('float32')

    norms = np.linalg.norm(xb, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    xb = xb / norms
    d = xb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(xb)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    logger.info("FAISS index built with %d vectors and saved to %s", xb.shape[0], index_path)
    return index

def load_faiss_index(index_path: Path = FAISS_INDEX_PATH):
    index = faiss.read_index(str(index_path))
    return index

def search_index(index, query_vec: np.ndarray, top_k=50):
    q = query_vec.astype('float32')

    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
    D, I = index.search(q, top_k)
    return D, I
