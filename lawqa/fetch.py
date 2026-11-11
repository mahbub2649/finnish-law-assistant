"""Download the current Finnish text of each law from the Finlex open data API.

    python -m lawqa.fetch

How the current version is found (and why):
Finlex publishes a consolidated text per amendment, named fin@<year><number, 4 digits> after
the amending statute (fin@20260412 = text including amendment 412/2026). The API's version
listing is incomplete: for 8 of the 10 laws here it stops years before the newest version,
which would silently serve outdated law. So instead the downloader reads the law's own list
of amending statutes, takes the newest one already in force, and requests that version,
stepping back one amendment at a time if a version is missing. manifest.json records, per
law, which amendment the text includes and the newest amendment in force, so staleness is visible.

Files are cached in data/raw/ and not committed: the repository holds code, not the corpus.
"""

from __future__ import annotations

import datetime as dt
import gzip
import io
import json
import re
import time
import zipfile

from .config import FINLEX_API, LAWS, RAW
from .net import client


def _body(resp) -> str:
    """Document endpoints answer with a zip holding main.xml; list endpoints with (possibly gzipped) XML."""
    raw = resp.content
    if raw[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            name = next((n for n in z.namelist() if n.endswith(".xml")), z.namelist()[0])
            raw = z.read(name)
    while raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw.decode("utf-8")


def amendments(xml: str) -> list[dict]:
    """Amending statutes listed in a consolidated text, with their entry-into-force dates."""
    out = {}
    for block in re.findall(r"<finlex:amendedBy>.*?</finlex:amendedBy>", xml, re.S):
        for ref in re.findall(r"<finlex:statuteReference>.*?</finlex:statuteReference>", block, re.S):
            m = re.search(r'href="/akn/fi/act/statute/(\d{4})/(\d+)"', ref)
            d = re.search(r'dateEntryIntoForce date="(\d{4}-\d{2}-\d{2})"', ref)
            if m:
                out[(int(m[1]), int(m[2]))] = dt.date.fromisoformat(d[1]) if d else None
    return [{"year": y, "number": n, "in_force": d} for (y, n), d in out.items()]


def in_force_order(items: list[dict], today: dt.date) -> list[dict]:
    """Amendments already in force, newest first (by entry date, then by statute number)."""
    live = [a for a in items if a["in_force"] is None or a["in_force"] <= today]
    return sorted(live, key=lambda a: (a["in_force"] or dt.date.min, a["year"], a["number"]), reverse=True)


def _any_version(http, year: int, number: int) -> str:
    r = http.get(f"{FINLEX_API}/akn/fi/act/statute-consolidated/{year}/{number}")
    r.raise_for_status()
    return _body(r)


def fetch_current(http, year: int, number: int, today: dt.date) -> tuple[str, str, dict | None]:
    """Return (version id, xml, newest amendment in force)."""
    listing = _any_version(http, year, number)
    order = in_force_order(amendments(listing), today)
    newest = order[0] if order else None
    for a in order[:15]:
        version = f"{a['year']}{a['number']:04d}"  # the number is zero-padded to four digits
        r = http.get(f"{FINLEX_API}/akn/fi/act/statute-consolidated/{year}/{number}/fin@{version}/main.akn")
        if r.status_code == 200:
            return version, _body(r), newest
        time.sleep(0.3)
    # No amendment-named version found: fall back to the newest version the listing names.
    versions = sorted(set(re.findall(rf"/statute-consolidated/{year}/{number}/fin@(\d+)", listing)), key=lambda v: (int(v[:4]), int(v[4:])))
    if not versions:
        raise RuntimeError(f"No Finnish consolidated text found for {number}/{year}")
    r = http.get(f"{FINLEX_API}/akn/fi/act/statute-consolidated/{year}/{number}/fin@{versions[-1]}/main.akn")
    r.raise_for_status()
    return versions[-1], _body(r), newest


def fetch_all(today: dt.date | None = None) -> list[dict]:
    today = today or dt.date.today()
    RAW.mkdir(parents=True, exist_ok=True)
    for old in RAW.glob("*.xml"):
        old.unlink()
    manifest = []
    with client() as http:
        for year, number, name in LAWS:
            version, xml, newest = fetch_current(http, year, number, today)
            path = RAW / f"{year}_{number}.xml"
            path.write_text(xml, encoding="utf-8")
            vy, vn = int(version[:4]), int(version[4:])
            current = newest is None or (newest["year"], newest["number"]) == (vy, vn)
            manifest.append({
                "year": year, "number": number, "name_en": name, "file": path.name,
                "version": f"{vn}/{vy}",
                "newest_amendment_in_force": f"{newest['number']}/{newest['year']} ({newest['in_force']})" if newest else None,
                "up_to_date": current, "fetched": today.isoformat(),
            })
            flag = "up to date" if current else f"BEHIND: newest in force is {manifest[-1]['newest_amendment_in_force']}"
            print(f"{number}/{year} {name}: includes amendment {vn}/{vy}, {len(xml) // 1024} KB, {flag}")
            time.sleep(1)  # be polite to a public service
    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


if __name__ == "__main__":
    fetch_all()
