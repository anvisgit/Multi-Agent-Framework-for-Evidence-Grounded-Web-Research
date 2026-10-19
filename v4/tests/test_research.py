import research


def test_validate_query_accepts_specific_question():
    ok, reason = research.validate_query("NVIDIA Blackwell data center deployment in 2024")
    assert ok is True
    assert reason == ""


def test_validate_query_rejects_vague_query():
    ok, reason = research.validate_query("analyze company")
    assert ok is False
    assert "Too short" in reason or "Too vague" in reason


def test_parse_plan_extracts_numbered_phrases():
    assert research.parse_plan("1. launch date\n2) mission goals\nIgnore this") == [
        "launch date", "mission goals"
    ]


def test_parse_followups_limits_and_cleans_search_lines():
    critique = "MISSING:\n- launch date\nSEARCH: Apollo 11 launch date\nSEARCH: Apollo 11 crew roles"
    assert research.parse_followups(critique, limit=1) == ["Apollo 11 launch date"]


def test_chunk_text_uses_overlap_and_omits_tiny_tail():
    text = "a" * 500
    chunks = research.chunk_text(text, size=200, overlap=50)
    assert len(chunks) == 3
    assert chunks[0][-50:] == chunks[1][:50]
    assert chunks[1][-50:] == chunks[2][:50]


def test_chunk_text_returns_empty_for_short_text():
    assert research.chunk_text("too short", size=200, overlap=50) == []
