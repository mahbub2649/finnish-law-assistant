"""OpenRouter client for chat and embeddings, with a disk cache and cost accounting.

Every call is cached by a hash of its full request, so re-running an evaluation costs
nothing and gives identical results. Cost comes from OpenRouter's usage accounting.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field

from .config import CACHE, OPENROUTER_URL, openrouter_key
from .net import client


@dataclass
class Usage:
    calls: int = 0
    cached: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, u: dict, seconds: float) -> None:
        with self.lock:
            self.calls += 1
            self.prompt_tokens += u.get("prompt_tokens", 0)
            self.completion_tokens += u.get("completion_tokens", 0)
            self.cost_usd += float(u.get("cost") or 0)
            self.seconds += seconds


USAGE = Usage()
_http = None


def _client():
    global _http
    if _http is None:
        _http = client(timeout=120)
    return _http


def _cached(kind: str, payload: dict, call):
    key = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    path = CACHE / kind / key[:2] / f"{key}.json"
    if path.exists():
        with USAGE.lock:
            USAGE.cached += 1
        return json.loads(path.read_text())
    result = call()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False))
    return result


def _post(path: str, payload: dict) -> dict:
    for attempt in range(5):
        t0 = time.time()
        r = _client().post(f"{OPENROUTER_URL}{path}", json=payload, headers={"Authorization": f"Bearer {openrouter_key()}"})
        if r.status_code in (429, 500, 502, 503, 504) and attempt < 4:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise RuntimeError(f"OpenRouter error: {data['error']}")
        data["_seconds"] = round(time.time() - t0, 3)
        return data
    raise RuntimeError("OpenRouter: too many retries")


class EmptyResponse(RuntimeError):
    """The model stopped before writing any visible text (usually: thinking used up max_tokens)."""


def chat(model: str, messages: list[dict], tools: list[dict] | None = None, max_tokens: int = 4000,
         temperature: float = 0.0, response_format: dict | None = None, cache: bool = True,
         effort: str | None = None) -> dict:
    """Returns the raw OpenAI-style response (choices[0].message, usage, _seconds).

    Current models think before answering and the thinking counts against max_tokens, so limits
    must leave room for it. A response cut off before any text or tool call raises EmptyResponse
    instead of passing an empty string on.
    """
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
               "usage": {"include": True}}
    if effort:
        payload["reasoning"] = {"effort": effort}
    if tools:
        payload["tools"] = tools
    if response_format:
        payload["response_format"] = response_format

    def call():
        data = _post("/chat/completions", payload)
        USAGE.add(data.get("usage", {}), data["_seconds"])
        choice = data["choices"][0]
        if choice.get("finish_reason") == "length" and not (choice["message"].get("content") or choice["message"].get("tool_calls")):
            raise EmptyResponse(f"{model} used all {max_tokens} tokens without answering")
        return data

    return _cached("chat", payload, call) if cache else call()


def embed(texts: list[str], model: str, batch: int = 64) -> list[list[float]]:
    out = []
    for i in range(0, len(texts), batch):
        payload = {"model": model, "input": texts[i:i + batch]}

        def call():
            data = _post("/embeddings", payload)
            USAGE.add(data.get("usage", {}), data["_seconds"])
            return {"data": [d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"])]}

        out += _cached("embed", payload, call)["data"]
    return out


def text_of(response: dict) -> str:
    return (response["choices"][0]["message"].get("content") or "").strip()
