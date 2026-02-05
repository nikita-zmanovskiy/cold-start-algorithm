# sanity_checks.py
from collections import Counter
from .utils import logger

def sanity_check_rankings(rankings_dict):
    """
    rankings_dict: user_id -> list of rec dicts (item_id,...)
    Prints distribution of lengths, unique recommended items, most common items
    """
    lens = [len(v) for v in rankings_dict.values()]
    dist = Counter(lens)
    all_items = []
    for v in rankings_dict.values():
        all_items.extend([str(x["item_id"]) for x in v])
    unique = set(all_items)
    common = Counter(all_items).most_common(10)
    logger.info("TopK length distribution: %s", dist)
    logger.info("Unique recommended items: %d", len(unique))
    logger.info("Most common items (top-10): %s", common)
    return {"length_dist": dist, "unique_items": len(unique), "common": common}
