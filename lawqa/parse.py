"""Turn each law's Akoma Ntoso XML into citable sections.

    python -m lawqa.parse          -> data/sections.jsonl

One record per section (§) in force today. Rules:
- A section can appear in more than one wording when an amendment is already published but
  not yet in force. The newest wording whose amending statute is in force is kept, and the
  record notes the upcoming change and its date.
- Repealed sections ("on kumottu") are dropped.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import asdict, dataclass, field

from lxml import etree

from .config import DATA, RAW
from .fetch import amendments

NS = {"a": "http://docs.oasis-open.org/legaldocml/ns/akn/3.0", "f": "http://data.finlex.fi/schema/finlex"}
F_ORIGINAL = "{http://data.finlex.fi/schema/finlex}originalVersion"
# "37 § on kumottu L:lla 19.12.2024/944." (with or without the subsection prefix)
REPEALED = re.compile(r"^(?:\(1\) )?(?:\d+ ?[a-z]? ?§ )?on kumottu\b")


@dataclass
class Section:
    id: str                 # "2001/55#chp_7__sec_3"
    law: str                # "55/2001"
    law_title_fi: str
    law_title_en: str
    chapter: str            # "7 luku Työsopimuksen päättämisperusteet"
    num: str                # "3 §"
    heading: str
    text: str               # full section text, subsections separated by blank lines
    url: str
    amended_by: str | None = None
    upcoming_change: dict | None = field(default=None)

    @property
    def label(self) -> str:
        chap = self.chapter.split(" ", 2)[:2]
        return f"{self.law_title_en} ({self.law}), {' '.join(chap) + ', ' if self.chapter else ''}{self.num}"


def _text(el) -> str:
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def _section_text(sec) -> str:
    parts = []
    for i, sub in enumerate(sec.findall("a:subsection", NS), 1):
        parts.append(f"({i}) {_text(sub)}")
    if not parts:  # sections without subsection markup
        body = [_text(c) for c in sec if etree.QName(c).localname not in ("num", "heading")]
        parts = [" ".join(b for b in body if b)]
    return "\n\n".join(p for p in parts if p)


def _version_key(sec) -> tuple[int, int] | None:
    v = sec.get(F_ORIGINAL)
    m = re.fullmatch(r"@(\d{4})(\d+)", v or "")
    return (int(m[1]), int(m[2])) if m else None


def parse_law(xml: str, year: int, number: int, name_en: str, today: dt.date) -> list[Section]:
    root = etree.fromstring(xml.encode("utf-8"))
    title = _text(root.find(".//a:docTitle", NS)) if root.find(".//a:docTitle", NS) is not None else ""
    in_force = {(a["year"], a["number"]): a["in_force"] for a in amendments(xml)}

    groups: dict[str, list] = {}
    for sec in root.iter(f"{{{NS['a']}}}section"):
        base = re.sub(r"v\d{8}", "", sec.get("eId", ""))
        groups.setdefault(base, []).append(sec)

    out = []
    for base, versions in groups.items():
        dated = []
        for s in versions:
            k = _version_key(s)
            date = in_force.get(k) if k else None
            dated.append((date, k or (0, 0), s))
        live = [d for d in dated if d[0] is None or d[0] <= today]
        future = [d for d in dated if d[0] is not None and d[0] > today]
        chosen = max(live, key=lambda d: (d[0] or dt.date.min, d[1])) if live else min(future, key=lambda d: d[0])
        sec = chosen[2]

        num = _text(sec.find("a:num", NS)) if sec.find("a:num", NS) is not None else ""
        heading = _text(sec.find("a:heading", NS)) if sec.find("a:heading", NS) is not None else ""
        text = _section_text(sec)
        if not text or REPEALED.match(text):
            continue
        chap_el = next((a for a in sec.iterancestors() if etree.QName(a).localname == "chapter"), None)
        chapter = ""
        if chap_el is not None:
            cn = chap_el.find("a:num", NS)
            ch = chap_el.find("a:heading", NS)
            chapter = " ".join(x for x in (_text(cn) if cn is not None else "", _text(ch) if ch is not None else "") if x)
        upcoming = None
        newer_future = [d for d in future if d[1] > chosen[1]]
        if newer_future and live:
            d, k, _ = min(newer_future, key=lambda d: d[0])
            upcoming = {"amendment": f"{k[1]}/{k[0]}", "in_force_from": d.isoformat()}
        k = chosen[1]
        out.append(Section(
            id=f"{year}/{number}#{base}", law=f"{number}/{year}", law_title_fi=title, law_title_en=name_en,
            chapter=chapter, num=num, heading=heading, text=text,
            url=f"https://www.finlex.fi/fi/lainsaadanto/{year}/{number}",
            amended_by=f"{k[1]}/{k[0]}" if k != (0, 0) else None, upcoming_change=upcoming,
        ))
    return out


def build(today: dt.date | None = None) -> list[Section]:
    today = today or dt.date.today()
    manifest = json.loads((RAW / "manifest.json").read_text())
    sections = []
    for law in manifest:
        xml = (RAW / law["file"]).read_text(encoding="utf-8")
        secs = parse_law(xml, law["year"], law["number"], law["name_en"], today)
        print(f"{law['number']}/{law['year']} {law['name_en']}: {len(secs)} sections")
        sections += secs
    with open(DATA / "sections.jsonl", "w", encoding="utf-8") as f:
        for s in sections:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
    return sections


def load() -> list[Section]:
    with open(DATA / "sections.jsonl", encoding="utf-8") as f:
        return [Section(**json.loads(line)) for line in f]


if __name__ == "__main__":
    secs = build()
    lens = sorted(len(s.text) for s in secs)
    print(f"total {len(secs)} sections; text length median {lens[len(lens)//2]} chars, 95th pct {lens[int(.95*len(lens))]}, max {lens[-1]}")
    print("upcoming changes:", [(s.id, s.upcoming_change) for s in secs if s.upcoming_change][:5])
