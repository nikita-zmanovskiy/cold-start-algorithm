# vark_simulator.py
from typing import Dict
from .utils import logger

VARK_QUESTIONS = [
    # 4 options per question but for simplicity we'll store just labels to compute a mock response
    # For production, you should store full VARK questionnaire and scoring.
    "I prefer to learn with diagrams or images.",
    "I remember best when I hear information.",
    "I prefer reading instructions.",
    "I learn best by doing / practicing.",
    # duplicate to reach more questions
] * 4  # ~16 questions

def score_vark_from_answers(answers):
    """
    answers: list of choices among ['visual','auditory','reading','kinesthetic']
    returns dominant style
    """
    from collections import Counter
    c = Counter(answers)
    if not c:
        return "visual"
    dom = c.most_common(1)[0][0]
    return dom

def simulate_vark_quiz_responses(seed=None):
    import random
    random.seed(seed)
    choices = ['visual','auditory','reading','kinesthetic']
    answers = [random.choice(choices) for _ in range(16)]
    dom = score_vark_from_answers(answers)
    return {"answers": answers, "dominant": dom}

def estimate_cognitive_state(time_of_day:str=None, session_length_min:int=5, device:str="desktop"):
    """
    Very simple heuristic:
    - if session_length > 30 and time_of_day in evening -> 'tired' => lower max_complexity
    """
    state = {"status":"neutral", "max_complexity":4}
    tod = (time_of_day or "").lower()
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

def build_user_profile_from_minimal(info: Dict):
    """
    info: may contain 'goal','age','gender','time_of_day','session_len'
    Returns user_profile used by retrieval/rerank
    """
    v = simulate_vark_quiz_responses(seed=info.get("user_id", None))
    state = estimate_cognitive_state(time_of_day=info.get("time_of_day"), session_length_min=info.get("session_len",5), device=info.get("device","desktop"))
    profile = {
        "user_id": info.get("user_id"),
        "goal": info.get("goal","learn about topic X"),
        "vark": v["dominant"],
        "vark_answers": v["answers"],
        "max_complexity": state["max_complexity"],
        "time_of_day": info.get("time_of_day"),
        "text_profile": info.get("text_profile", info.get("goal",""))
    }
    return profile
