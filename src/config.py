# config.py
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
INDEX_DIR = DATA_DIR / "index"
RESULTS_DIR = ROOT / "results"

# ensure folders exist
for p in (PROCESSED_DIR, EMBEDDINGS_DIR, INDEX_DIR, RESULTS_DIR):
    p.mkdir(parents=True, exist_ok=True)

# embeddings model (sentence-transformers)
# EMBED_MODEL = "all-MiniLM-L6-v2"  # CPU-friendly

# FAISS config
FAISS_INDEX_PATH = INDEX_DIR / "items.faiss"
EMBEDDINGS_NPY = EMBEDDINGS_DIR / "item_embeddings.npy"
EMBEDDING_MAP = EMBEDDINGS_DIR / "id2idx.json"

# LLM backend: "hf" to use local HuggingFace model (if available), otherwise "heuristic"
# candidate retrieval
CANDIDATE_POOL = 500
RERANK_TOPK = 20

# seeds
RANDOM_SEED = 42


# LLM backend
LLM_BACKEND = "hf"  # "hf" or "heuristic"

HF_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# generation params (CPU-safe defaults)
HF_MAX_NEW_TOKENS = 256
HF_TEMPERATURE = 0.3
HF_TOP_P = 0.9
HF_DEVICE = "cpu"  # или "cuda"


EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # единая каноническая строка

