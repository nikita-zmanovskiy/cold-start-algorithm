from sentence_transformers import CrossEncoder
from .utils import logger

class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu"):
        logger.info("Loading CrossEncoder: %s", model_name)
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, user_profile, candidate_items, items_meta, topk=10):
        user_text = user_profile.get("text_profile") or user_profile.get("goal")
        pairs = []
        valid_ids = []

        for cid in candidate_items:
            meta = items_meta.get(str(cid), {})
            text = meta.get("title","") + ". " + meta.get("description","")
            pairs.append((user_text, text))
            valid_ids.append(cid)

        scores = self.model.predict(pairs, batch_size=32)
        ranked = sorted(zip(valid_ids, scores), key=lambda x: x[1], reverse=True)

        return [
            {"item_id": cid, "score": float(score), "reason": "CrossEncoder rerank"}
            for cid, score in ranked[:topk]
        ]
