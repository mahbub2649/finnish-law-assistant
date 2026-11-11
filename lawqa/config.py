"""Settings and the list of laws in the corpus."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
CACHE = DATA / "cache"

# Laws that matter most to English speakers living and working in Finland.
# (year, number, English name). Text comes from Finlex in Finnish; Finlex has no English API.
LAWS = [
    (2001, 55, "Employment Contracts Act"),
    (2019, 872, "Working Hours Act"),
    (2005, 162, "Annual Holidays Act"),
    (2014, 1325, "Non-Discrimination Act"),
    (1986, 609, "Act on Equality between Women and Men"),
    (2004, 759, "Act on the Protection of Privacy in Working Life"),
    (2002, 738, "Occupational Safety and Health Act"),
    (1995, 481, "Act on Residential Leases"),
    (1978, 38, "Consumer Protection Act"),
    (2004, 301, "Aliens Act"),
]

FINLEX_API = "https://opendata.finlex.fi/finlex/avoindata/v1"
USER_AGENT = "finnish-law-assistant (portfolio project; github.com/mahbub2649)"

OPENROUTER_URL = "https://openrouter.ai/api/v1"
ANSWER_MODEL = os.environ.get("LAWQA_ANSWER_MODEL", "anthropic/claude-haiku-5.5")
UTILITY_MODEL = os.environ.get("LAWQA_UTILITY_MODEL", "anthropic/claude-haiku-5.5")  # query translation
EMBED_MODEL = os.environ.get("LAWQA_EMBED_MODEL", "openai/text-embedding-3-small")
# Does each quote support its claim? Haiku: same error rate as Sonnet in the evaluation, 4x cheaper (README).
CHECK_MODEL = os.environ.get("LAWQA_CHECK_MODEL", "anthropic/claude-haiku-5.5")


def openrouter_key() -> str:
    key = os.environ.get("OPENROUTER_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        env = ROOT / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith(("OPENROUTER_KEY=", "OPENROUTER_API_KEY=")):
                    key = line.split("=", 1)[1].strip().strip('"')
    if not key:
        raise RuntimeError("Set OPENROUTER_KEY in the environment or in .env")
    return key
