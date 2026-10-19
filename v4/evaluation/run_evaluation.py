"""Run the fixed three-way benchmark. Requires Ollama and the v4 dependencies."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from claim_verification import split_sentences, verify_report
from research import ask, embed, fetch_docs, run_research


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "questions.json"


def _context(docs: list[dict]) -> str:
    return "\n\n---\n\n".join(
        f"Title: {d['title']}\nURL: {d['url']}\n{d['text']}" for d in docs
    )


def _docs_from_context(context: str) -> list[dict]:
    docs = []
    for block in re.split(r"\n\s*---\s*\n", context):
        title = re.search(r"^Title:\s*(.+)$", block, re.M)
        url = re.search(r"^URL:\s*(\S+)", block, re.M)
        body = re.sub(r"^(?:Title|URL):.*(?:\n|$)", "", block, flags=re.M).strip()
        if url and body:
            docs.append({"title": title.group(1) if title else url.group(1), "url": url.group(1), "text": body})
    return docs


def _judge_coverage(report: str, fact: str) -> bool:
    verdict = ask(
        "Does the report explicitly state this required fact? Use only the report. "
        "Reply exactly YES or NO.\n\n"
        f"REQUIRED FACT: {fact}\n\nREPORT:\n{report}"
    ).strip().upper()
    return verdict.startswith("YES")


def _citation_checks(report: str, docs: list[dict]) -> list[dict]:
    by_url = {doc["url"].rstrip("/"): doc for doc in docs}
    checks = []
    for sentence in split_sentences(report):
        urls = re.findall(r"\[Source:\s*(https?://[^\]\s]+)\]", sentence)
        if not urls:
            continue
        for url in urls:
            source = by_url.get(url.rstrip("/"))
            supported = False
            if source:
                verdict = ask(
                    "Does the passage from this exact URL directly support the claim? "
                    "Answer exactly SUPPORTED or UNSUPPORTED.\n\n"
                    f"CLAIM: {sentence}\n\nURL: {source['url']}\n"
                    f"PASSAGE: {source['text']}"
                ).strip().upper()
                supported = verdict.startswith("SUPPORTED") and not verdict.startswith("UNSUPPORTED")
            checks.append({"claim": sentence, "url": url, "supported": supported, "url_found": source is not None})
    return checks


def _score(question: dict, variant: str, report: str, evidence: str, citation_docs: list[dict]) -> dict:
    verification = verify_report(report, evidence, embed, ask)
    citations = _citation_checks(report, citation_docs)
    claims = verification["checks"]
    assessed = [c for c in claims if c["status"] != "uncertain"]
    unsupported = sum(c["status"] == "unsupported" for c in claims)
    coverage = [_judge_coverage(report, fact) for fact in question["key_facts"]]
    return {
        "question_id": question["id"],
        "variant": variant,
        "report": report,
        "claims": claims,
        "citation_checks": citations,
        "coverage_checks": dict(zip(question["key_facts"], coverage)),
        "metrics": {
            "citation_accuracy": (sum(c["supported"] for c in citations) / len(citations)) if citations else 0.0,
            "citation_count": len(citations),
            "claim_count": len(claims),
            "unsupported_claim_count": unsupported,
            "unsupported_claim_rate": (unsupported / len(assessed)) if assessed else None,
            "key_facts_present": sum(coverage),
            "key_fact_count": len(coverage),
            "coverage": sum(coverage) / len(coverage) if coverage else None,
        },
    }


def run(limit: int | None = None, output: Path | None = None) -> dict:
    questions = json.loads(DATASET.read_text(encoding="utf-8"))
    if limit is not None:
        questions = questions[:limit]
    results = []
    for index, question in enumerate(questions, start=1):
        print(f"[{index}/{len(questions)}] {question['id']} {question['question']}", flush=True)
        docs = fetch_docs(question["question"], n=4)
        if not docs:
            print("  no web evidence retrieved; recording failed retrieval", flush=True)
            continue
        evidence = _context(docs)
        no_web = ask(
            "Answer the research question as a concise, factual report with no web access. "
            "Do not invent citations.\n\nQUESTION: " + question["question"]
        )
        scraped = ask(
            "Write a concise research report answering the question using only the supplied scraped text. "
            "Put [Source: exact-url] after each factual sentence. State gaps.\n\n"
            f"QUESTION: {question['question']}\n\nSCRAPED TEXT:\n{evidence}"
        )
        full = run_research(question["question"])
        # The evaluator's fetched pages are held out from the no-web model prompt. They
        # provide a common, explicit reference corpus for baseline faithfulness scoring.
        for label, report, docs_for_citations in (
            ("single_prompt_no_web", no_web, docs),
            ("single_prompt_scraped_text", scraped, docs),
            ("full_pipeline", full["report"], _docs_from_context(full.get("verification_context", ""))),
        ):
            result = _score(question, label, report, evidence, docs_for_citations)
            results.append(result)
            print(
                f"  {label}: citation={result['metrics']['citation_accuracy']:.2f}, "
                f"unsupported={result['metrics']['unsupported_claim_count']}/"
                f"{result['metrics']['claim_count']}, coverage={result['metrics']['coverage']:.2f}",
                flush=True,
            )
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": "Configured local Ollama model",
        "dataset": str(DATASET.name),
        "question_count": len(questions),
        "results": results,
    }
    destination = output or ROOT / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved raw results to {destination}")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="run the first N benchmark questions")
    parser.add_argument("--output", type=Path, help="write JSON results to this path")
    args = parser.parse_args()
    run(args.limit, args.output)
