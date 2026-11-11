"""Find the sections relevant to an English question.

Modes (compared in the evaluation):
- bm25_en:  keyword search with the English question as typed (expected to fail on Finnish text)
- bm25_fi:  keyword search after a small model rewrites the question as Finnish legal search terms
- dense:    multilingual embeddings, English question against Finnish sections
- bm25_fi_ngram: as bm25_fi, but over character 5-grams instead of stems (Finnish inflection)
- hybrid:   reciprocal rank fusion of bm25_fi and dense
- hybrid_ngram: reciprocal rank fusion of bm25_fi_ngram and dense
"""

from __future__ import annotations

import json
import pickle
from functools import lru_cache

import numpy as np

from .config import EMBED_MODEL, UTILITY_MODEL
from .index import INDEX, ngrams, tokens
from .llm import chat, embed, text_of
from .parse import Section, load

MODES = ("bm25_en", "bm25_fi", "bm25_fi_ngram", "dense", "hybrid", "hybrid_ngram")

TRANSLATE_PROMPT = """Rewrite this question about Finnish law as a Finnish keyword search query for \
Finlex. Use the Finnish legal terms the statute itself would use (for example "irtisanomisaika", \
"koeaika", "vuosiloma", "oleskelulupa"), plus close synonyms. Output only the search words on \
one line, no explanation.

Question: {q}"""


@lru_cache
def _store():
    sections = load()
    ids = json.loads((INDEX / "ids.json").read_text())
    by_id = {s.id: s for s in sections}
    with open(INDEX / "bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
    with open(INDEX / "bm25_ngram.pkl", "rb") as f:
        bm25_ng = pickle.load(f)
    return [by_id[i] for i in ids], {"stem": bm25, "ngram": bm25_ng}, np.load(INDEX / "dense.npy")


def sections_by_id() -> dict[str, Section]:
    return {s.id: s for s in _store()[0]}


def to_finnish_query(question: str) -> str:
    r = chat(UTILITY_MODEL, [{"role": "user", "content": TRANSLATE_PROMPT.format(q=question)}], max_tokens=3000, effort="low")
    text = text_of(r)
    if not text:
        raise RuntimeError("query translation returned no text")
    return " ".join(text.split())


def _rank_bm25(query: str, k: int, kind: str = "stem") -> list[tuple[int, float]]:
    _, bm25s, _ = _store()
    scores = bm25s[kind].get_scores(tokens(query) if kind == "stem" else ngrams(query))
    order = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in order if scores[i] > 0]


def _rank_dense(query: str, k: int) -> list[tuple[int, float]]:
    _, _, vecs = _store()
    q = np.array(embed([query], EMBED_MODEL)[0], dtype=np.float32)
    q /= np.linalg.norm(q)
    sims = vecs @ q
    order = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in order]


def rrf(rankings: list[list[tuple[int, float]]], k: int, c: int = 60) -> list[tuple[int, float]]:
    """Reciprocal rank fusion: score = sum of 1 / (c + rank) over the rankings."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, (i, _) in enumerate(ranking):
            scores[i] = scores.get(i, 0.0) + 1.0 / (c + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])[:k]


def search(question: str, mode: str = "hybrid", k: int = 8) -> list[tuple[Section, float]]:
    sections, _, _ = _store()
    if mode == "bm25_en":
        ranked = _rank_bm25(question, k)
    elif mode == "bm25_fi":
        ranked = _rank_bm25(to_finnish_query(question), k)
    elif mode == "bm25_fi_ngram":
        ranked = _rank_bm25(to_finnish_query(question), k, "ngram")
    elif mode == "dense":
        ranked = _rank_dense(question, k)
    elif mode == "hybrid":
        ranked = rrf([_rank_bm25(to_finnish_query(question), 50), _rank_dense(question, 50)], k)
    elif mode == "hybrid_ngram":
        ranked = rrf([_rank_bm25(to_finnish_query(question), 50, "ngram"), _rank_dense(question, 50)], k)
    else:
        raise ValueError(f"unknown mode {mode}")
    return [(sections[i], s) for i, s in ranked]
