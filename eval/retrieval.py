"""Retrieval evaluation: does the gold section appear in the top k?

    python -m eval.retrieval --split dev
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from lawqa import retrieve
from lawqa.llm import USAGE

from .data import HERE, splits


def rank_of(question: dict, mode: str, k: int = 50) -> int | None:
    hits = [s.id for s, _ in retrieve.search(question["question"], mode, k)]
    ranks = [hits.index(g) + 1 for g in question["gold"] if g in hits]
    return min(ranks) if ranks else None


def score(ranks: list[int | None]) -> dict:
    n = len(ranks)
    return {
        "n": n,
        **{f"recall@{k}": round(sum(1 for r in ranks if r and r <= k) / n, 3) for k in (1, 5, 10)},
        "mrr": round(sum(1 / r for r in ranks if r) / n, 3),
    }


def run(split: str, modes=retrieve.MODES) -> dict:
    qs = splits()[split]
    out = {}
    for mode in modes:
        with ThreadPoolExecutor(6) as pool:
            ranks = list(pool.map(lambda q: rank_of(q, mode), qs))
        out[mode] = {**score(ranks), "misses": [q["id"] for q, r in zip(qs, ranks) if not r or r > 10]}
        print(f"{split:12s} {mode:14s} " + "  ".join(f"{k} {v}" for k, v in out[mode].items() if k != "misses"), flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--modes", default=",".join(retrieve.MODES))
    a = ap.parse_args()
    res = run(a.split, a.modes.split(","))
    (HERE.parent / "reports").mkdir(exist_ok=True)
    (HERE.parent / "reports" / f"retrieval_{a.split}.json").write_text(json.dumps(res, indent=1))
    print(f"cost so far ${USAGE.cost_usd:.4f} ({USAGE.calls} calls, {USAGE.cached} cached)")


if __name__ == "__main__":
    main()
