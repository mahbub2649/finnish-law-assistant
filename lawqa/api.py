"""HTTP API and the chat page.

    uvicorn lawqa.api:app --port 8200
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import answer, retrieve
from .config import ANSWER_MODEL, RAW

WEB = Path(__file__).resolve().parent.parent / "web"
app = FastAPI(title="Finnish law assistant")


class Ask(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    pipeline: str = Field(default="rag", pattern="^(rag|agent)$")


@app.get("/api/health")
def health():
    manifest = json.loads((RAW / "manifest.json").read_text()) if (RAW / "manifest.json").exists() else []
    return {"ok": True, "model": ANSWER_MODEL, "sections": len(retrieve.sections_by_id()),
            "laws": [{k: m[k] for k in ("name_en", "year", "number", "version", "up_to_date", "fetched")} for m in manifest]}


@app.post("/api/ask")
def ask(body: Ask):
    a = answer.ask(body.question.strip(), body.pipeline)
    sections = retrieve.sections_by_id()
    return {
        "status": a.status,
        "message": answer.render(a) if a.status != "answered" else None,
        "summary": a.summary,
        "claims": [{"text": c.text, "quote_fi": c.quote_fi, "section_id": c.section_id, "label": sections[c.section_id].label,
                    "heading": sections[c.section_id].heading, "url": sections[c.section_id].url} for c in a.valid_claims],
        "removed": [{"text": c.text, "reason": c.reason} for c in a.claims if not c.valid],
        "pipeline": a.pipeline, "model": a.model, "seconds": a.seconds, "cost_usd": a.cost_usd, "steps": a.steps,
    }


@app.get("/api/sections/{law_year}/{law_number}/{eid}")
def section(law_year: int, law_number: int, eid: str):
    s = retrieve.sections_by_id().get(f"{law_year}/{law_number}#{eid}")
    if not s:
        raise HTTPException(404, "No such section")
    return {"id": s.id, "label": s.label, "heading": s.heading, "text": s.text, "url": s.url}


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(WEB / "index.html")
