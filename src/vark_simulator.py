
from typing import Dict, List, Any, Optional
from .utils import logger


VARK_TO_DESCRIPTION = {
    "visual": "User prefers visual learning; interested in diagrams, images, and visual content.",
    "auditory": "User prefers auditory learning; interested in podcasts, lectures, and spoken content.",
    "reading": "User prefers reading and text; interested in articles, books, and written material.",
    "kinesthetic": "User prefers learning by doing; interested in hands-on and interactive content.",
}

VARK_QUESTIONS = [

    "I prefer to learn with diagrams or images.",
    "I remember best when I hear information.",
    "I prefer reading instructions.",
    "I learn best by doing / practicing.",

] * 4  

def score_vark_from_answers(answers):

    from collections import Counter
    c = Counter(answers)
    if not c:
        return "visual"
    dom = c.most_common(1)[0][0]
    return dom

def simulate_vark_quiz_responses(seed=None):
    import random

    if isinstance(seed, str):
        seed_int = hash(seed) % (2**31)
        random.seed(seed_int)
    else:
        random.seed(seed)
    choices = ['visual','auditory','reading','kinesthetic']
    answers = [random.choice(choices) for _ in range(16)]
    dom = score_vark_from_answers(answers)
    return {"answers": answers, "dominant": dom}

def estimate_cognitive_state(time_of_day:str=None, session_length_min:int=5, device:str="desktop"):

    state = {"status":"neutral", "max_complexity":4}
    tod = (time_of_day or "").lower()
    

    if session_length_min is None:
        session_length_min = 5 
    
    if session_length_min > 30 and tod in ["evening","night","23:00","22:00"]:
        state["status"] = "tired"
        state["max_complexity"] = 2
    elif session_length_min < 10:
        state["status"] = "fresh"
        state["max_complexity"] = 5
    else:
        state["status"] = "neutral"
        state["max_complexity"] = 4
    return state

def build_text_profile_from_viewed_items(
    viewed_item_ids: List[str],
    items_meta: Any,
    max_items: int = 5,
    max_chars: int = 600,
) -> str:

    if not viewed_item_ids or not items_meta:
        return ""
    
    by_id = {}
    if isinstance(items_meta, list):
        for it in items_meta:
            by_id[str(it.get("item_id"))] = it
    else:
        by_id = {str(k): v for k, v in (items_meta or {}).items()}
    parts = []
    total_len = 0
    for iid in viewed_item_ids[:max_items]:
        if total_len >= max_chars:
            break
        it = by_id.get(str(iid))
        if not it:
            continue
        title = str(it.get("title", "")).strip()
        desc = (it.get("description") or it.get("text", "") or "")
        if isinstance(desc, list):
            desc = " ".join(str(x) for x in desc)
        desc = str(desc).strip()[:150]
        tags = it.get("format_tags") or it.get("genres", "")
        if isinstance(tags, (list, tuple)):
            tags = ", ".join(str(x) for x in tags)
        tags = str(tags).strip()[:80]
        s = title
        if desc:
            s += ". " + desc
        if tags:
            s += " [" + tags + "]"
        if s:
            parts.append(s)
            total_len += len(s) + 2
    if not parts:
        return ""
    return "Recently viewed interests: " + " | ".join(parts)


def build_user_profile_from_minimal(info: Dict):

    user_id = info.get("user_id", "")
    v = simulate_vark_quiz_responses(seed=user_id)
    state = estimate_cognitive_state(
        time_of_day=info.get("time_of_day"),
        session_length_min=info.get("session_len", 5),
        device=info.get("device", "desktop"),
    )
    goal = info.get("goal", "")
    vark = v["dominant"]

    text_profile = info.get("text_profile", goal)
    if (text_profile and text_profile.strip()):
        pass  
    else:

        viewed_ids = info.get("viewed_item_ids") or []
        items_meta = info.get("items_meta")
        if viewed_ids and items_meta:
            text_profile = build_text_profile_from_viewed_items(
                viewed_ids, items_meta, max_items=5, max_chars=600
            )
  
        if not text_profile or not text_profile.strip():
            desc = VARK_TO_DESCRIPTION.get(vark, "User exploring diverse recommendations.")
            text_profile = f"User profile (cold-start): {desc}"

    profile = {
        "user_id": user_id,
        "goal": goal or "learn about topic X",
        "vark": vark,
        "vark_answers": v["answers"],
        "max_complexity": state["max_complexity"],
        "time_of_day": info.get("time_of_day"),
        "text_profile": text_profile,
    }
    return profile
