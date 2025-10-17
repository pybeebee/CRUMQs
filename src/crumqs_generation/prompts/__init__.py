from src.crumqs_generation.prompts.generation import *

DOC_PROMPTS = ["d1", "d2", "d3", "d4", "d5", "d6"]
CLAIM_PROMPTS = ["c1", "c2", "c3", "c4", "c5", "c6", "c7"]

PROMPT_REGISTRY = {
    "d1": d1,
    "d2": d2,
    "d3": d3,
    "d4": d4,
    "d5": d5,
    "d6": d6,
    "c1": c1,
    "c2": c2,
    "c3": c3,
    "c4": c4,
    "c5": c5,
    "c6": c6,
    "c7": c7,
}

CRITERION_NAMES = [
    "Context Necessity",
    "Context Sufficiency",
    "Answer Correctness",
    "Answer Uniqueness",
]