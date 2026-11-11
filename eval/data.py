"""Evaluation questions and their fixed dev/test split."""

from __future__ import annotations

import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name: str) -> list[dict]:
    with open(HERE / f"{name}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# Generated items found faulty on inspection, excluded with the reason (the split itself is unchanged).
EXCLUDED = {
    "gen-020": "not a question: the generator explained why it skipped the section",
    "gen-141": "question cut off mid-sentence (generator hit its token limit)",
}


def splits(seed: int = 13) -> dict[str, list[dict]]:
    """generated questions split 50/50 into dev (for decisions) and test (for reporting only)."""
    gen = load("generated")
    rng = random.Random(seed)
    order = list(range(len(gen)))
    rng.shuffle(order)
    half = len(gen) // 2
    hand = load("handwritten")
    return {
        "dev": [gen[i] for i in sorted(order[:half]) if gen[i]["id"] not in EXCLUDED],
        "test": [gen[i] for i in sorted(order[half:]) if gen[i]["id"] not in EXCLUDED],
        "handwritten": [r for r in hand if r["type"] == "in_scope"],
        "out_of_scope": [r for r in hand if r["type"] == "out_of_scope"],
    }
