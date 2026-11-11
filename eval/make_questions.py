"""Generate evaluation questions from sections, with a stronger model than the one under test.

    python -m eval.make_questions --n 150

For each sampled section the model writes one question a non-lawyer might ask in English,
answerable from that section alone, without Finnish words or section numbers (otherwise
keyword search would be handed the answer). The section is the gold label. The model may
answer SKIP for sections that hold no real rule (definitions, entry into force, cross-references).

Caveat, reported in the README: another section can also answer a generated question, so
recall against a single gold section is a lower bound.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from lawqa.llm import USAGE, chat, text_of
from lawqa.parse import load

GEN_MODEL = "anthropic/claude-sonnet-5.5"
OUT = Path(__file__).resolve().parent / "generated.jsonl"

PROMPT = """You are writing test questions for an English-language assistant that answers \
questions about Finnish law for people living and working in Finland.

Below is one section of a Finnish statute (in Finnish). Write ONE question that:
- a non-lawyer would realistically ask in English, in their own words, about their situation
- is fully answered by this section
- contains no Finnish words, no law names and no section numbers
- is not a yes/no question about a definition

If the section contains no rule a person would ask about (for example it only defines terms,
sets an entry-into-force date, refers to other laws, or concerns only authorities' internal
procedures), output exactly: SKIP

Output only the question or SKIP.

Law: {law} ({law_en})
{chapter} {num} {heading}
{text}"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    by_law = defaultdict(list)
    for s in load():
        if len(s.text) >= 250 and s.heading:
            by_law[s.law].append(s)
    # Stratified: every law gets questions roughly in proportion to its size, at least 6.
    total = sum(len(v) for v in by_law.values())
    picks = []
    for law, secs in by_law.items():
        k = max(6, round(a.n * len(secs) / total))
        picks += rng.sample(secs, min(k, len(secs)))
    rng.shuffle(picks)

    rows = []
    for i, s in enumerate(picks):
        r = chat(GEN_MODEL, [{"role": "user", "content": PROMPT.format(
            law=s.law_title_fi, law_en=s.law_title_en, chapter=s.chapter, num=s.num, heading=s.heading, text=s.text[:5000])}],
            max_tokens=4000, temperature=0.3, effort="low")  # thinking counts against max_tokens
        q = text_of(r).strip().strip('"')
        if not q or "SKIP" in q.upper().split() or "\n" in q or not q.endswith("?") or re.search(r"§|\blaki\b", q):
            continue  # skipped, explained, multi-line or not a complete question
        rows.append({"id": f"gen-{len(rows) + 1:03d}", "question": q, "gold": [s.id], "type": "in_scope",
                     "source": "generated", "law": s.law})
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(picks)} sections, {len(rows)} questions, ${USAGE.cost_usd:.3f}", flush=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} questions from {len(picks)} sections; cost ${USAGE.cost_usd:.3f}")


if __name__ == "__main__":
    main()
