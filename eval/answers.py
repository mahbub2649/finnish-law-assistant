"""End-to-end answer evaluation for both pipelines.

    python -m eval.answers --pipelines rag,agent --splits test,handwritten,out_of_scope

For each answer:
- status (answered, not_covered, unsupported, error) and how many claims each check removed
- gold cited: a verified claim cites a gold section (in-scope questions)
- correctness: a separate judge model compares the shown answer with the gold section's text
- refusal: out-of-scope questions must end as not_covered or unsupported
- cost and latency
The judge sees only what a user would see (summary plus verified claims), not the model's raw output.
"""

from __future__ import annotations

import argparse
import json
import statistics
from concurrent.futures import ThreadPoolExecutor

from lawqa import answer, retrieve
from lawqa.llm import USAGE, chat, text_of

from .data import HERE, splits

JUDGE_MODEL = "anthropic/claude-sonnet-5.5"
JUDGE_PROMPT = """You grade an English answer to a question about Finnish law. You get the question, \
the Finnish statute section that answers it (the reference), and the answer a user was shown.

Grades:
- correct: answers the question, and agrees with the reference on everything that matters
- partial: nothing wrong, but leaves out the key rule or number the reference gives
- incorrect: states something the reference contradicts, or gets the key rule or number wrong
- no_answer: the assistant declined or said the law does not cover it

Information from other sections is fine unless it contradicts the reference.

Question: {question}

Reference ({label}):
{reference}

Answer shown to the user:
{shown}"""
JUDGE_SCHEMA = answer._schema("grade", {"type": "object", "additionalProperties": False, "required": ["grade", "reason"],
                                        "properties": {"grade": {"type": "string", "enum": ["correct", "partial", "incorrect", "no_answer"]},
                                                       "reason": {"type": "string"}}})


def judge(q: dict, a: answer.Answer) -> dict:
    sections = retrieve.sections_by_id()
    ref = sections[q["gold"][0]]
    shown = answer.render(a)
    r = chat(JUDGE_MODEL, [{"role": "user", "content": JUDGE_PROMPT.format(
        question=q["question"], label=ref.label, reference=f"{ref.heading}\n{ref.text[:6000]}", shown=shown)}],
        max_tokens=4000, effort="low", response_format=JUDGE_SCHEMA)
    return answer._parse(text_of(r), need="grade")


def run_one(q: dict, pipeline: str) -> dict:
    a = answer.ask(q["question"], pipeline)
    row = {"id": q["id"], "split": q.get("split"), "pipeline": pipeline, "status": a.status, "seconds": a.seconds,
           "cost_usd": a.cost_usd, "steps": a.steps, "claims": len(a.claims),
           "removed_quote": sum(c.reason in ("quote not found in section", "section not provided", "quote too short") for c in a.claims),
           "removed_support": sum(c.reason == "quote does not support claim" for c in a.claims),
           "shown_claims": len(a.valid_claims), "summary_kept": bool(a.summary),
           "gold_retrieved": bool(set(q["gold"]) & set(a.retrieved)),
           "gold_cited": bool(set(q["gold"]) & {c.section_id for c in a.valid_claims})}
    if q["type"] == "in_scope":
        g = judge(q, a)
        row.update(grade=g["grade"], grade_reason=g["reason"])
    else:
        row["refused"] = a.status in ("not_covered", "unsupported")
    row["shown"] = answer.render(a)
    return row


def summarise(rows: list[dict]) -> dict:
    ins = [r for r in rows if "grade" in r]
    oos = [r for r in rows if "refused" in r]
    out = {"n": len(rows)}
    if ins:
        n = len(ins)
        for g in ("correct", "partial", "incorrect", "no_answer"):
            out[g] = round(sum(r["grade"] == g for r in ins) / n, 3)
        out["gold_cited"] = round(sum(r["gold_cited"] for r in ins) / n, 3)
        out["gold_retrieved"] = round(sum(r["gold_retrieved"] for r in ins) / n, 3)
    if oos:
        out["refused_out_of_scope"] = round(sum(r["refused"] for r in oos) / len(oos), 3)
    claims = sum(r["claims"] for r in rows) or 1
    out["claims_removed_quote_check"] = round(sum(r["removed_quote"] for r in rows) / claims, 3)
    out["claims_removed_support_check"] = round(sum(r["removed_support"] for r in rows) / claims, 3)
    secs = sorted(r["seconds"] for r in rows)
    out["latency_p50_s"] = round(statistics.median(secs), 1)
    out["latency_p95_s"] = round(secs[int(0.95 * (len(secs) - 1))], 1)
    out["cost_per_question_usd"] = round(sum(r["cost_usd"] for r in rows) / len(rows), 4)
    out["errors"] = sum(r["status"] == "error" for r in rows)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipelines", default="rag,agent")
    ap.add_argument("--splits", default="test,handwritten,out_of_scope")
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. the checker model used")
    a = ap.parse_args()
    suffix = f"_{a.tag}" if a.tag else ""
    data = splits()
    questions = [{**q, "split": s} for s in a.splits.split(",") for q in data[s]]  # faulty items already excluded
    results = {}
    for pipeline in a.pipelines.split(","):
        with ThreadPoolExecutor(4) as pool:
            rows = list(pool.map(lambda q: run_one(q, pipeline), questions))
        (HERE.parent / "reports" / f"answers_{pipeline}{suffix}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
        results[pipeline] = {s: summarise([r for r in rows if r["split"] == s]) for s in a.splits.split(",")}
        results[pipeline]["all_in_scope"] = summarise([r for r in rows if "grade" in r])
        for s, m in results[pipeline].items():
            print(f"{pipeline:6s} {s:13s} " + "  ".join(f"{k} {v}" for k, v in m.items()), flush=True)
        print(f"   spent so far ${USAGE.cost_usd:.3f}", flush=True)
    (HERE.parent / "reports" / f"answers{suffix}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
