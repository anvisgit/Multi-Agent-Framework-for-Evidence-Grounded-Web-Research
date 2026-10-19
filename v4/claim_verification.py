"""Sentence-level evidence checks for reports produced by the research pipeline."""

from __future__ import annotations

import math
import re
from typing import Callable


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences while keeping punctuation attached."""
    sentences = []
    for paragraph in re.split(r"\n\s*\n", text.strip()):
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        if not paragraph:
            continue
        sentences.extend(
            part.strip() for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[])", paragraph)
            if part.strip()
        )
    return sentences


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _overlap_score(claim: str, passage: str) -> float:
    words = set(re.findall(r"[a-z0-9]+", claim.lower()))
    words -= {"the", "and", "for", "from", "with", "that", "this", "was", "were", "are"}
    passage_words = set(re.findall(r"[a-z0-9]+", passage.lower()))
    return len(words & passage_words) / max(len(words), 1)


def _passages(context: str) -> list[dict[str, str]]:
    blocks = re.split(r"\n\s*---\s*\n", context or "")
    result = []
    for block in blocks:
        url_match = re.search(r"^URL:\s*(\S+)", block, re.M)
        title_match = re.search(r"^Title:\s*(.+)$", block, re.M)
        body = re.sub(r"^(?:Title|URL):.*(?:\n|$)", "", block, flags=re.M).strip()
        if body:
            result.append({"text": body, "url": url_match.group(1) if url_match else "", "title": title_match.group(1).strip() if title_match else "Source"})
    return result


def verify_report(
    report: str,
    context: str,
    embed: Callable[[list[str]], list[list[float]]],
    judge: Callable[[str], str],
    similarity_threshold: float = 0.12,
) -> dict:
    """Check report sentences against nearest evidence, then ask the model to judge support.

    Sentences judged unsupported are removed. Uncertain sentences remain visibly flagged.
    If embeddings fail, token overlap selects passages; every kept claim still receives
    the model support check.
    """
    passages = _passages(context)
    sentences = split_sentences(report)
    if not sentences:
        return {"report": report, "checks": [], "removed": 0, "uncertain": 0}
    if not passages:
        return {
            "report": "\n\n".join(f"[UNVERIFIED] {sentence}" for sentence in sentences),
            "checks": [{"claim": sentence, "status": "uncertain", "url": ""} for sentence in sentences],
            "removed": 0,
            "uncertain": len(sentences),
        }

    ranked: dict[str, list[tuple[float, dict[str, str]]]] = {}
    try:
        vectors = embed(sentences + [item["text"] for item in passages])
        claim_vectors, passage_vectors = vectors[:len(sentences)], vectors[len(sentences):]
        for claim, vector in zip(sentences, claim_vectors):
            ranked[claim] = sorted(
                ((_cosine(vector, pvec), passage) for pvec, passage in zip(passage_vectors, passages)),
                key=lambda item: item[0], reverse=True,
            )[:3]
    except Exception:
        for claim in sentences:
            ranked[claim] = sorted(
                ((_overlap_score(claim, item["text"]), item) for item in passages),
                key=lambda item: item[0], reverse=True,
            )[:3]

    checks, kept = [], []
    for claim in sentences:
        if re.fullmatch(r"(?:#{1,6}\s*)?[A-Z][^.!?]{0,80}", claim) and len(claim.split()) <= 8:
            kept.append(claim)
            continue
        candidates = ranked[claim]
        if not candidates or candidates[0][0] < similarity_threshold:
            checks.append({"claim": claim, "status": "unsupported", "url": candidates[0][1]["url"] if candidates else ""})
            continue
        evidence = "\n\n".join(
            f"Source: {item['title']}\nURL: {item['url']}\nPassage: {item['text']}"
            for _, item in candidates
        )
        verdict = judge(
            "Decide whether the evidence directly supports the claim. Do not use outside knowledge. "
            "Answer with exactly SUPPORTED, UNSUPPORTED, or UNCERTAIN.\n\n"
            f"CLAIM: {claim}\n\nEVIDENCE:\n{evidence}"
        ).strip().upper()
        status = "supported" if verdict.startswith("SUPPORTED") and not verdict.startswith("UNSUPPORTED") else (
            "unsupported" if verdict.startswith("UNSUPPORTED") else "uncertain"
        )
        checks.append({"claim": claim, "status": status, "url": candidates[0][1]["url"]})
        if status == "supported":
            kept.append(claim)
        elif status == "uncertain":
            kept.append(f"[UNVERIFIED] {claim}")

    return {
        "report": "\n\n".join(kept),
        "checks": checks,
        "removed": sum(check["status"] == "unsupported" for check in checks),
        "uncertain": sum(check["status"] == "uncertain" for check in checks),
    }
