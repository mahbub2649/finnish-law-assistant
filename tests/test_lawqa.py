import datetime as dt
from pathlib import Path
from types import SimpleNamespace

import pytest

from lawqa import answer, fetch, index, parse, retrieve

FIXTURE = (Path(__file__).parent / "fixtures" / "mini_act.xml").read_text(encoding="utf-8")
TODAY = dt.date(2026, 11, 11)


def test_amendments_and_in_force_order():
    am = fetch.amendments(FIXTURE)
    assert {(a["year"], a["number"]) for a in am} == {(2020, 10), (2030, 5)}
    order = fetch.in_force_order(am, TODAY)
    assert [(a["year"], a["number"]) for a in order] == [(2020, 10)]   # 5/2030 is not in force yet


def test_parser_keeps_the_version_in_force_and_flags_the_upcoming_one():
    secs = parse.parse_law(FIXTURE, 2001, 99, "Test Act", TODAY)
    assert [s.id for s in secs] == ["2001/99#chp_1__sec_1"]          # repealed 2 § is dropped
    s = secs[0]
    assert "kuusi kuukautta" in s.text and "neljä" not in s.text
    assert s.text.startswith("(1) ") and "\n\n(2) " in s.text
    assert s.upcoming_change == {"amendment": "5/2030", "in_force_from": "2030-01-01"}
    assert s.chapter == "1 luku Yleiset säännökset" and s.law_title_fi == "Testilaki"
    later = parse.parse_law(FIXTURE, 2001, 99, "Test Act", dt.date(2031, 1, 1))[0]
    assert "neljä kuukautta" in later.text and later.upcoming_change is None


def test_ngrams_survive_finnish_consonant_gradation():
    a, b = set(index.ngrams("irtisanomisaika")), set(index.ngrams("irtisanomisajat"))
    assert len(a & b) / len(a) > 0.6
    assert index.tokens("irtisanomisaika") != index.tokens("irtisanomisajat")   # why stems alone fail


def test_reciprocal_rank_fusion_rewards_agreement():
    fused = retrieve.rrf([[(1, 9.0), (2, 8.0), (3, 7.0)], [(3, 0.9), (1, 0.8), (4, 0.7)]], k=3)
    assert [i for i, _ in fused][:2] == [1, 3]


def _section(text="Koeaika voi olla enintään kuusi kuukautta."):
    return SimpleNamespace(id="2001/55#chp_1__sec_4", text=text, heading="Koeaika", label="ECA 1:4", url="u")


def test_verify_checks_provenance_in_code():
    secs = {"2001/55#chp_1__sec_4": _section()}
    claims = [answer.Claim("ok", "2001/55#chp_1__sec_4", "koeaika voi olla  enintään kuusi"),
              answer.Claim("made up", "2001/55#chp_1__sec_4", "koeaika voi olla enintään vuosi"),
              answer.Claim("not given", "2001/55#chp_2__sec_1", "koeaika voi olla enintään kuusi"),
              answer.Claim("short", "2001/55#chp_1__sec_4", "koeaika")]
    answer.verify(claims, {"2001/55#chp_1__sec_4"}, secs)
    assert [c.reason for c in claims] == ["verified", "quote not found in section", "section not provided", "quote too short"]


def test_parse_takes_the_last_complete_json_object():
    raw = '{"claims": [true, false]}\nWait, corrected:\n{"claims": [true, true], "summary": true} trailing }'
    assert answer._parse(raw) == {"claims": [True, True], "summary": True}
    with pytest.raises(ValueError):
        answer._parse("no json here")


def _fake_chat(replies):
    def chat(model, messages, **kw):
        return {"choices": [{"message": {"content": replies.pop(0)}}], "usage": {}}
    return chat


@pytest.fixture
def fake_corpus(monkeypatch):
    sec = _section()
    sec.law, sec.num, sec.chapter = "55/2001", "4 §", "1 luku"
    monkeypatch.setattr(retrieve, "search", lambda q, mode="hybrid", k=8: [(sec, 1.0)])
    monkeypatch.setattr(retrieve, "sections_by_id", lambda: {sec.id: sec})
    return sec


def test_answer_with_a_fabricated_quote_is_refused(monkeypatch, fake_corpus):
    monkeypatch.setattr(answer, "chat", _fake_chat([
        '{"status": "answered", "claims": [{"text": "It can be a year.", "section_id": "2001/55#chp_1__sec_4", '
        '"quote_fi": "koeaika voi olla enintään vuoden"}], "summary": "A year."}']))
    a = answer.answer_rag("How long can a trial period be?")
    assert a.status == "unsupported" and not a.valid_claims
    assert "won't guess" in answer.render(a)


def test_answer_survives_both_checks(monkeypatch, fake_corpus):
    monkeypatch.setattr(answer, "chat", _fake_chat([
        '{"status": "answered", "claims": [{"text": "At most six months.", "section_id": "2001/55#chp_1__sec_4", '
        '"quote_fi": "enintään kuusi kuukautta"}], "summary": "Up to six months."}',
        '{"claims": [true], "summary": true}']))
    a = answer.answer_rag("How long can a trial period be?")
    assert a.status == "answered" and len(a.valid_claims) == 1 and a.summary == "Up to six months."


def test_support_check_removes_a_real_quote_attached_to_a_wrong_claim(monkeypatch, fake_corpus):
    monkeypatch.setattr(answer, "chat", _fake_chat([
        '{"status": "answered", "claims": [{"text": "At most one year.", "section_id": "2001/55#chp_1__sec_4", '
        '"quote_fi": "enintään kuusi kuukautta"}], "summary": "Up to a year."}',
        '{"claims": [false], "summary": false}']))
    a = answer.answer_rag("How long can a trial period be?")
    assert a.status == "unsupported" and a.claims[0].reason == "quote does not support claim"


def test_api_validates_input_and_returns_claims(monkeypatch, fake_corpus):
    from fastapi.testclient import TestClient
    from lawqa import api
    monkeypatch.setattr(answer, "chat", _fake_chat([
        '{"status": "answered", "claims": [{"text": "At most six months.", "section_id": "2001/55#chp_1__sec_4", '
        '"quote_fi": "enintään kuusi kuukautta"}], "summary": "Up to six months."}',
        '{"claims": [true], "summary": true}']))
    c = TestClient(api.app)
    r = c.post("/api/ask", json={"question": "How long can a trial period be?"})
    assert r.status_code == 200 and r.json()["claims"][0]["label"] == "ECA 1:4"
    assert c.post("/api/ask", json={"question": "x"}).status_code == 422
    assert c.post("/api/ask", json={"question": "valid question", "pipeline": "magic"}).status_code == 422
