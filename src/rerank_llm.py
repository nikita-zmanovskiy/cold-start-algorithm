# src/rerank_llm.py
from sentence_transformers import CrossEncoder
import numpy as np
from pathlib import Path
from .utils import logger

class CrossReranker:
    """
    Batched CrossEncoder reranker using sentence-transformers CrossEncoder.
    items_meta: list of dicts with item_id, title, description, format_tags
    """
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", items_meta=None, device="cpu"):
        self.model_name = model_name
        self.device = device
        logger.info("Loading CrossEncoder: %s", model_name)
        # device can be "cpu" or "cuda"
        self.model = CrossEncoder(model_name, device=device)
        # map by id for fast lookup
        self.items = {}
        for it in (items_meta or []):
            self.items[str(it.get("item_id"))] = it

    def _make_text_for_item(self, meta):
        title = meta.get("title","")
        desc = meta.get("description","") or meta.get("text","") or ""
        tags = meta.get("format_tags") or meta.get("genres") or ""
        if isinstance(tags, (list,tuple)):
            tags = ", ".join(tags)
        s = f"Title: {title}. Description: {desc}. Tags: {tags}"
        return s.strip()

    def rerank(self, user_profile: dict, candidate_ids: list, topk=10, batch_size=128):
        qtext = user_profile.get("text_profile") or user_profile.get("goal") or ""
        # prepare pairs
        pairs = []
        ids = []
        for cid in candidate_ids:
            meta = self.items.get(str(cid), {})
            text = self._make_text_for_item(meta)
            pairs.append([qtext, text])
            ids.append(str(cid))
        # predict scores in batch
        if len(pairs) == 0:
            return []
        scores = self.model.predict(pairs, show_progress_bar=False, batch_size=batch_size)
        scored = list(zip(ids, scores))
        scored.sort(key=lambda x: x[1], reverse=True)  # descending
        out = []
        for cid, s in scored[:topk]:
            out.append({"item_id": cid, "score": float(s), "reason": "CrossEncoder rerank"})
        return out
