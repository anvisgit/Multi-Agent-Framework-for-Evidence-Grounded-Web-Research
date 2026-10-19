# second research round driven by the Critic's follow-up searches
# persistent memory (memory.json) so follow-up questions build on earlier reports
import json
import os
from datetime import datetime

import ollama
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS

MODEL = "qwen2.5:1.5b"
MEMORY_FILE = "memory.json"
MAX_SOURCE_CHARS = 6000
MAX_PREVIOUS_CHARS = 2000


def ask(prompt):
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"num_ctx": 8192},
    )
    return response["message"]["content"].strip()



def search(query, n=3):
    try:
        return list(DDGS().text(query, max_results=n))
    except Exception as e:
        print("Search failed:", e)
        return []


def scrape(url):
    try:
        html = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"}).text
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return " ".join(soup.get_text().split())[:1200]
    except Exception:
        return ""


def get_sources(query, n=3, seen=None):
    if seen is None:
        seen = set()
    pieces = []
    for r in search(query, n):
        url = r["href"]
        if url in seen:
            continue
        seen.add(url)
        text = scrape(url) or r["body"]
        pieces.append(f"Title: {r['title']}\nURL: {url}\n{text}")
    return "\n\n---\n\n".join(pieces)



def planner(query):
    return ask(
        f"Give 3 short web-search phrases about: {query}\n"
        "Output only a numbered list, one phrase per line. No other text."
    )


def researcher(query, sources):
    return ask(
        f"List the key facts about: {query}\n"
        "Use ONLY the sources below. Write bullets. After each fact, add the "
        "source title and URL. If it is not in the sources, leave it out.\n\n"
        f"SOURCES:\n{sources[:MAX_SOURCE_CHARS]}"
    )


def critic(facts):
    return ask(
        "You are a strict reviewer. Review these findings.\n"
        "Reply in EXACTLY this format:\n\n"
        "MISSING:\n- <specific missing fact>\n- <specific missing fact>\n\n"
        "WEAK CLAIMS:\n- <weak claim and why>\n\n"
        "SEARCH: <short web-search phrase that would find a missing fact>\n"
        "SEARCH: <another short web-search phrase>\n\n"
        "Be specific. Do not add new facts.\n\n"
        f"FINDINGS:\n{facts}"
    )


def writer(query, facts, critique, previous_report=""):
    previous = ""
    if previous_report:
        previous = (
            "PREVIOUS REPORT (the user is following up on this; build on it, "
            f"do not repeat it):\n{previous_report[:MAX_PREVIOUS_CHARS]}\n\n"
        )
    return ask(
        f"{previous}"
        f"Write a professional report on: {query}\n"
        "Use these headings: Executive Summary, Key Findings, Limitations, References.\n"
        "Rules: use ONLY the facts below, never invent anything, "
        "list every source as 'Title - URL' under References, "
        "and state honestly the gaps listed in GAPS.\n\n"
        f"FACTS:\n{facts}\n\nGAPS:\n{critique}"
    )



def parse_plan(text):
    phrases = []
    for line in text.splitlines():
        line = line.strip()
        if line and line[0].isdigit():
            phrases.append(line.lstrip("0123456789.) ").strip())
    return phrases


def parse_followups(critique, limit=2):
    searches = []
    for line in critique.splitlines():
        clean = line.strip().lstrip("-*0123456789. ").strip()
        if clean.upper().startswith("SEARCH:"):
            phrase = clean[7:].strip().strip("<>\"'")
            if phrase and "short web-search phrase" not in phrase.lower():
                searches.append(phrase)
    return searches[:limit]
def _load():
    if not os.path.exists(MEMORY_FILE):
        return []
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def save_memory(query, report):
    data = _load()
    data.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "query": query,
        "report": report,
    })
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_memory():
    return _load()


def get_last_report():
    """Return the most recent report, or '' if there is none."""
    data = _load()
    return data[-1]["report"] if data else ""


def clear_memory():
    """Delete all saved reports."""
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)


# ---------------------------------------------------------------- PIPELINE (B1)

def run_research(query, previous_report="", progress=print):
    """
    Full pipeline:
    plan -> search -> extract -> critique -> ROUND 2 (follow-up searches) -> report.
    'progress' is any function that accepts a string; the web app passes its own.
    Returns a dictionary with every stage's output.
    """
    seen = set()

    progress("1/5  Planning...")
    plan = planner(query)

    progress("2/5  Searching the web...")
    sources = get_sources(query, n=3, seen=seen)
    for phrase in parse_plan(plan)[:3]:
        progress(f"      searching: {phrase}")
        extra = get_sources(phrase, n=1, seen=seen)
        if extra:
            sources += "\n\n---\n\n" + extra

    progress("3/5  Extracting facts...")
    facts = researcher(query, sources)

    progress("4/5  Critiquing, then running round 2...")
    critique = critic(facts)
    followups = parse_followups(critique)

    round2_facts = ""
    if followups:
        round2_sources = ""
        for phrase in followups:
            progress(f"      follow-up search: {phrase}")
            found = get_sources(phrase, n=2, seen=seen)
            if found:
                round2_sources += found + "\n\n---\n\n"
        if round2_sources.strip():
            round2_facts = researcher(query, round2_sources)
            facts = facts + "\n\nADDITIONAL FACTS (round 2):\n" + round2_facts
    else:
        progress("      critic gave no usable follow-up searches; skipping round 2")

    progress("5/5  Writing report...")
    report = writer(query, facts, critique, previous_report)

    progress("Done.")
    return {
        "plan": plan,
        "facts": facts,
        "critique": critique,
        "followups": followups,
        "round2_facts": round2_facts,
        "report": report,
    }


#  TERMINAL MODE

def main():
    query = input("Research question: ").strip()
    if len(query.split()) < 4:
        print("Too vague. Try: NVIDIA Blackwell GPU data center 2024")
        return

    previous = get_last_report()
    if previous:
        print("(Previous report found in memory; this run will build on it.)")

    result = run_research(query, previous)

    print("\n" + "=" * 60)
    print(result["report"])

    save_memory(query, result["report"])
    with open("report.txt", "w", encoding="utf-8") as f:
        f.write(result["report"])
    print("\nSaved to report.txt and memory.json")


if __name__ == "__main__":
    main()
