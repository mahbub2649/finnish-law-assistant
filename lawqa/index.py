"""Search indexes over the sections: Finnish keyword (BM25) and multilingual embeddings.

    python -m lawqa.index      -> data/index/
"""

from __future__ import annotations

import json
import pickle
import re

import numpy as np
import snowballstemmer
from rank_bm25 import BM25Okapi

from .config import DATA, EMBED_MODEL
from .llm import embed
from .parse import Section, load

INDEX = DATA / "index"
_stem = snowballstemmer.stemmer("finnish")

# Very common Finnish function words; legal text repeats them constantly.
STOP = set("ja tai on ei se ne että jos kun sekä eikä joka jonka jota jotka mitä mikä tämän tässä tätä sen siitä siihen niin kuin myös vain olla ole ovat oli ollut sekä taikka mukaan mukaisesti tarkoitettu tarkoitetun tarkoitettua".split())


def tokens(text: str) -> list[str]:
    words = re.findall(r"[a-zåäö0-9]+", text.lower())
    return [w for w in _stem.stemWords(words) if w not in STOP and len(w) > 1]


def ngrams(text: str, n: int = 5) -> list[str]:
    """Character n-grams of each word, with word boundaries marked.

    Finnish inflection changes word stems (consonant gradation: irtisanomisaika ->
    irtisanomisajat), which defeats stemmers; most n-grams survive the change.
    """
    out = []
    for w in re.findall(r"[a-zåäö0-9]+", text.lower()):
        if w in STOP or len(w) < 2:
            continue
        w = f"_{w}_"
        out += [w[i:i + n] for i in range(max(1, len(w) - n + 1))]
    return out


def passage(s: Section) -> str:
    """What the embedding model sees: where the section sits, its heading, then its text."""
    return f"{s.law_title_fi} ({s.law_title_en}), {s.chapter}, {s.num} {s.heading}\n{s.text}"[:6000]


def build() -> None:
    sections = load()
    INDEX.mkdir(parents=True, exist_ok=True)
    bm25 = BM25Okapi([tokens(f"{s.heading} {s.heading} {s.text}") for s in sections])
    with open(INDEX / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    bm25_ng = BM25Okapi([ngrams(f"{s.heading} {s.heading} {s.text}") for s in sections])
    with open(INDEX / "bm25_ngram.pkl", "wb") as f:
        pickle.dump(bm25_ng, f)
    vecs = np.array(embed([passage(s) for s in sections], EMBED_MODEL), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    np.save(INDEX / "dense.npy", vecs)
    (INDEX / "ids.json").write_text(json.dumps([s.id for s in sections]))
    (INDEX / "meta.json").write_text(json.dumps({"embed_model": EMBED_MODEL, "sections": len(sections), "dim": int(vecs.shape[1])}))
    print(f"indexed {len(sections)} sections; embeddings {vecs.shape} with {EMBED_MODEL}")


if __name__ == "__main__":
    from .llm import USAGE
    build()
    print(f"embedding cost ${USAGE.cost_usd:.4f}, {USAGE.prompt_tokens} tokens")
