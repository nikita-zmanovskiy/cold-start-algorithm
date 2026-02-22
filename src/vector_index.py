
import faiss
import numpy as np
from pathlib import Path
import os
import tempfile
import shutil
from .config import FAISS_INDEX_PATH
from .utils import logger

def build_faiss_index(embeddings: np.ndarray, index_path: Path = FAISS_INDEX_PATH, force_rebuild: bool = False):
    """
    Build FAISS index from embeddings.
    
    Args:
        embeddings: numpy array of embeddings
        index_path: path to save the index
        force_rebuild: if True, rebuild even if index exists
    """
    xb = embeddings.astype('float32')

    norms = np.linalg.norm(xb, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    xb = xb / norms
    d = xb.shape[1]
    
    # Check if index already exists and is valid
    if not force_rebuild and index_path.exists():
        try:
            existing_index = faiss.read_index(str(index_path))
            if existing_index.ntotal == xb.shape[0] and existing_index.d == d:
                logger.info("FAISS index already exists and matches dimensions (%d vectors, %d dims). Using existing index.", 
                           existing_index.ntotal, existing_index.d)
                return existing_index
            else:
                logger.info("Existing FAISS index dimensions don't match (%d vectors, %d dims vs %d vectors, %d dims). Rebuilding.",
                           existing_index.ntotal, existing_index.d, xb.shape[0], d)
        except Exception as e:
            logger.warning("Could not read existing FAISS index %s: %s. Will rebuild.", index_path, e)
    
    index = faiss.IndexFlatIP(d)
    index.add(xb)
    
    # Ensure directory exists
    index_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use atomic write: write to temp file first, then rename
    index_path_str = str(index_path.resolve())
    temp_path = None
    try:
        # Create temp file in same directory for atomic rename
        temp_fd, temp_path = tempfile.mkstemp(
            suffix='.faiss',
            dir=str(index_path.parent),
            prefix='.tmp_'
        )
        os.close(temp_fd)
        temp_path_str = str(temp_path)
        
        # Write to temp file
        faiss.write_index(index, temp_path_str)
        
        # Atomically replace old file
        # On Windows, remove old file first, then move temp file
        if index_path.exists():
            try:
                index_path.unlink()
            except Exception as unlink_err:
                logger.warning("Could not remove old index file %s: %s. Will try to overwrite.", index_path, unlink_err)
        
        # Use shutil.move for cross-platform atomic operation
        shutil.move(temp_path_str, index_path_str)
        logger.info("FAISS index built with %d vectors and saved to %s", xb.shape[0], index_path)
        
    except Exception as e:
        # Clean up temp file if it exists
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
        
        logger.error("Failed to write FAISS index to %s: %s", index_path_str, e)
        # Fallback: try direct write
        try:
            if index_path.exists():
                index_path.unlink()
            faiss.write_index(index, index_path_str)
            logger.info("FAISS index built with %d vectors and saved to %s (direct write)", xb.shape[0], index_path)
        except Exception as e2:
            logger.error("Failed to write FAISS index with direct write: %s", e2)
            raise RuntimeError(f"Could not write FAISS index to {index_path}: {e}, {e2}")
    
    return index

def load_faiss_index(index_path: Path = FAISS_INDEX_PATH):
    index = faiss.read_index(str(index_path))
    return index

def search_index(index, query_vec: np.ndarray, top_k=50):
    q = query_vec.astype('float32')

    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
    D, I = index.search(q, top_k)
    return D, I
