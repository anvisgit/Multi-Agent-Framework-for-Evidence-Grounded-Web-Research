from claim_verification import split_sentences, verify_report


def test_split_sentences_keeps_sentence_punctuation():
    assert split_sentences("First fact. Second fact!") == ["First fact.", "Second fact!"]


def test_verifier_removes_unsupported_and_flags_uncertain():
    context = "Title: NASA\nURL: https://example.org\nMars rover launched in 2020."

    def embed(texts):
        return [[1.0, 0.0] for _ in texts]

    def judge(prompt):
        claim = prompt.split("CLAIM:", 1)[1].split("EVIDENCE:", 1)[0]
        if "Mars rover launched in 2020" in claim:
            return "SUPPORTED"
        if "Venus rover" in claim:
            return "UNSUPPORTED"
        return "UNCERTAIN"

    result = verify_report(
        "Key Findings\n\nMars rover launched in 2020. Venus rover landed in 2022.",
        context,
        embed,
        judge,
    )
    assert "Key Findings" in result["report"]
    assert "Mars rover launched in 2020." in result["report"]
    assert "Venus rover" not in result["report"]
    assert result["removed"] == 1
