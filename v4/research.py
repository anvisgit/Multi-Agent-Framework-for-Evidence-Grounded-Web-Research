

import json
import logging
import os
import re
import sqlite3
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import chromadb
import ollama
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS

from claim_verification import verify_report
from config import (
    CACHE_FILE, CHUNK_OVERLAP, CHUNK_SIZE, EMBED_MODEL, MAX_PREVIOUS_CHARS,
    MAX_SCRAPE_WORKERS, MEMORY_FILE, MODEL, PAGE_CHARS, REPORTS_DIR,
    SCRAPE_CACHE_TTL_SECONDS, TOP_K, TOP_K_ROUND2,
)

logger = logging.getLogger(__name__)


class ResearchError(Exception):
    """A problem the user can understand and fix (shown nicely in the UI)."""



_IGNORED_WORDS = {
    "analyze", "analyse", "research", "study", "explain", "tell", "about", "me",
    "the", "a", "an", "of", "on", "in", "for", "and", "to", "is", "are", "what",
    "how", "why", "who", "give", "show", "write", "report", "overview", "summary",
    "please", "can", "you", "i", "want", "info", "information", "details", "some",
}


def validate_query(query):
    """Return (True, '') if the query is specific enough, else (False, reason)."""
    query = (query or "").strip()
    if not query:
        return False, "Please enter a research question."
    if len(query) > 300:
        return False, "Too long. Keep the question under 300 characters."
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9.\-+']*", query)
    if len(words) < 4:
        return False, "Too short. Use at least 4 words, e.g. 'NVIDIA Blackwell GPU data center 2024'."
    content = [w for w in words if w.lower() not in _IGNORED_WORDS]
    if len(content) < 3:
        return False, ("Too vague. Name the specific company, product or technology "
                       "and, ideally, a time period.")
    return True, ""


def clean_output(text):
    """Remove 'Thought:', 'Final Answer:', <think> blocks and chatty openers."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I)
    kept = []
    for line in text.strip().splitlines():
        stripped = line.strip()
        if re.match(r"^(thought|action|observation)\s*:", stripped, re.I):
            continue
        kept.append(re.sub(r"^\s*final answer\s*:\s*", "", line, flags=re.I))
    text = "\n".join(kept).strip()
    text = re.sub(r"^(sure|certainly|of course|okay)\b[^\n]*\n+", "", text, count=1, flags=re.I)
    text = re.sub(r"^here\s+(is|are|'s)\b[^\n]*:\s*\n+", "", text, count=1, flags=re.I)
    return text.strip()


def _model_names():
    """Names of every model installed in Ollama (handles old and new library versions)."""
    try:
        response = ollama.list()
    except Exception:
        raise ResearchError(
            "Cannot reach Ollama. Open the Ollama app (llama icon in the system tray) and try again."
        )
    models = getattr(response, "models", None)
    if models is None:
        models = response["models"]
    names = []
    for m in models:
        if isinstance(m, dict):
            name = m.get("model") or m.get("name") or ""
        else:
            name = getattr(m, "model", None) or getattr(m, "name", "") or ""
        if name:
            names.append(name if ":" in name else name + ":latest")
    return names


def check_ollama():
    """Verify Ollama and the model are available. Returns True if RAG embeddings are available."""
    names = _model_names()
    wanted = MODEL if ":" in MODEL else MODEL + ":latest"
    if wanted not in names:
        raise ResearchError(f"Model '{MODEL}' is not installed. Run:  ollama pull {MODEL}")
    return any(n.split(":")[0] == EMBED_MODEL for n in names)



def ask(prompt, retries=2):
    last_error = None
    for attempt in range(retries + 1):
        try:
            response = ollama.chat(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"num_ctx": 8192},
            )
            return clean_output(response["message"]["content"])
        except Exception as e:
            last_error = e
            time.sleep(1.5)
    raise ResearchError(f"The model failed to answer after {retries + 1} attempts: {last_error}")



def search(query, n=3):
    for attempt in range(2):
        try:
            return list(DDGS().text(query, max_results=n))
        except Exception as e:
            logger.warning("Search failed for %r: %s", query, e)
            time.sleep(2)
    return []


def _cached_page(url):
    try:
        with sqlite3.connect(CACHE_FILE, timeout=10) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS pages (url TEXT PRIMARY KEY, fetched_at REAL NOT NULL, text TEXT NOT NULL)"
            )
            row = connection.execute("SELECT fetched_at, text FROM pages WHERE url = ?", (url,)).fetchone()
        if row and time.time() - row[0] < SCRAPE_CACHE_TTL_SECONDS:
            return row[1]
    except sqlite3.Error as error:
        logger.debug("Page cache read failed for %s: %s", url, error)
    return None


def _store_page(url, text):
    try:
        with sqlite3.connect(CACHE_FILE, timeout=10) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS pages (url TEXT PRIMARY KEY, fetched_at REAL NOT NULL, text TEXT NOT NULL)"
            )
            connection.execute(
                "INSERT OR REPLACE INTO pages (url, fetched_at, text) VALUES (?, ?, ?)",
                (url, time.time(), text),
            )
    except sqlite3.Error as error:
        logger.debug("Page cache write failed for %s: %s", url, error)


def scrape(url):
    cached = _cached_page(url)
    if cached is not None:
        return cached
    try:
        html = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"}).text
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()
        paragraphs = " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))
        text = paragraphs if len(paragraphs) > 300 else soup.get_text(" ", strip=True)
        text = " ".join(text.split())[:PAGE_CHARS]
        if text:
            _store_page(url, text)
        return text
    except Exception as error:
        logger.debug("Page scrape failed for %s: %s", url, error)
        return ""


def fetch_docs(query, n=3, seen=None):
    if seen is None:
        seen = set()
    docs = []
    results = []
    for r in search(query, n):
        url = r.get("href", "")
        if not url or url in seen:
            continue
        seen.add(url)
        results.append((r, url))
    if not results:
        return docs
    with ThreadPoolExecutor(max_workers=min(MAX_SCRAPE_WORKERS, len(results))) as pool:
        scraped_pages = list(pool.map(lambda pair: scrape(pair[1]), results))
    for (r, url), text in zip(results, scraped_pages):
        if len(text) < 200:                 # scrape failed or page is thin: use the search snippet
            text = r.get("body", "") or text
        if text:
            docs.append({"title": r.get("title", "Untitled"), "url": url, "text": text})
    return docs



_client = None


def _chroma():
    global _client
    if _client is None:
        _client = chromadb.EphemeralClient()
    return _client


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Cut text into overlapping pieces so no idea is split across a boundary."""
    chunks, start = [], 0
    step = max(size - overlap, 1)
    while start < len(text):
        piece = text[start:start + size].strip()
        if len(piece) > max(50, overlap):
            chunks.append(piece)
        start += step
    return chunks


def embed(texts):
    
    vectors = []
    for i in range(0, len(texts), 16):                       # small batches are safer
        response = ollama.embed(model=EMBED_MODEL, input=texts[i:i + 16])
        batch = getattr(response, "embeddings", None)
        if batch is None:
            batch = response["embeddings"]
        vectors.extend(batch)
    return vectors


def retrieve(question, docs, k=TOP_K, use_rag=True):
   
    chosen = None

    if use_rag:
        ids, texts, metas = [], [], []
        for d in docs:
            for chunk in chunk_text(d["text"]):
                ids.append(uuid.uuid4().hex)
                texts.append(chunk)
                metas.append({"title": d["title"], "url": d["url"]})
        if texts:
            name = "run_" + uuid.uuid4().hex[:10]
            try:
                client = _chroma()
                collection = client.create_collection(name=name, metadata={"hnsw:space": "cosine"})
                collection.add(ids=ids, documents=texts, embeddings=embed(texts), metadatas=metas)
                result = collection.query(
                    query_embeddings=embed([question]),
                    n_results=min(k, len(texts)),
                )
                chosen = list(zip(result["documents"][0], result["metadatas"][0]))
                client.delete_collection(name)
            except Exception as e:
                logger.warning("RAG unavailable; falling back to plain excerpts: %s", e)
                chosen = None

    if chosen is None:
        chosen = [(d["text"][:1200], {"title": d["title"], "url": d["url"]}) for d in docs[:k]]

    blocks, used, seen_urls = [], [], set()
    for text, meta in chosen:
        blocks.append(f"Title: {meta['title']}\nURL: {meta['url']}\n{text}")
        if meta["url"] not in seen_urls:
            seen_urls.add(meta["url"])
            used.append({"title": meta["title"], "url": meta["url"]})
    return "\n\n---\n\n".join(blocks), used



def planner(query):
    return ask(
        f"Give 3 short web-search phrases about: {query}\n"
        "Each phrase: 3 to 8 words, keyword style, covering a different angle "
        "(core facts, competition or market, risks or outlook).\n"
        "Output only a numbered list, one phrase per line. No other text."
    )


def researcher(query, context):
    return ask(
        f"Extract the key facts about: {query}\n"
        "Rules:\n"
        "- Use ONLY the sources below. Never use your own memory.\n"
        "- Write one bullet per fact. Copy numbers, names and dates exactly.\n"
        "- After each fact add: | Source: <title> | URL: <url>\n"
        "- If a fact is not in the sources, leave it out.\n"
        "- Finish with a short section 'NOT FOUND IN SOURCES' listing what the sources did not cover.\n\n"
        f"SOURCES:\n{context}"
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
        f"Write a professional research report on: {query}\n"
        "Use exactly these headings: Executive Summary, Key Findings, Limitations.\n"
        "Rules:\n"
        "- Use ONLY the facts below. Never invent a number, name or date.\n"
        "- Plain, formal language. No hype words.\n"
        "- Under Limitations, state honestly the gaps listed in GAPS.\n"
        "- After every factual sentence, add [Source: <URL>] using only a URL shown in FACTS.\n"
        "- Do NOT write a References section; it is added automatically.\n\n"
        f"FACTS:\n{facts}\n\nGAPS:\n{critique}"
    )


# ------------------------------------------------------------------ PARSING

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


def format_references(sources):
    if not sources:
        return ""
    lines = [f"[{i}] {s['title']} - {s['url']}" for i, s in enumerate(sources, 1)]
    return "\n\nReferences (retrieved pages)\n" + "\n".join(lines)


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
    data = _load()
    return data[-1]["report"] if data else ""


def clear_memory():
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)


# ------------------------------------------------------------------ PIPELINE

def run_research(query, previous_report="", progress=print):
    
    started = time.time()

    ok, reason = validate_query(query)
    if not ok:
        raise ResearchError(reason)

    rag_available = check_ollama()
    if not rag_available:
        progress(f"Note: '{EMBED_MODEL}' not installed, using plain excerpts instead of RAG. "
                 f"(Fix: ollama pull {EMBED_MODEL})")

    seen = set()

    progress("1/5  Planning...")
    plan = planner(query)

    progress("2/5  Searching the web...")
    docs = fetch_docs(query, n=3, seen=seen)
    for phrase in parse_plan(plan)[:3]:
        progress(f"      searching: {phrase}")
        docs += fetch_docs(phrase, n=1, seen=seen)
    if not docs:
        raise ResearchError("No web pages could be retrieved. Check your internet connection and try again.")
    progress(f"      collected {len(docs)} pages")

    progress("3/5  Retrieving the most relevant passages and extracting facts...")
    context, used = retrieve(query, docs, k=TOP_K, use_rag=rag_available)
    facts = researcher(query, context)

    progress("4/5  Critiquing, then running round 2...")
    critique = critic(facts)
    followups = parse_followups(critique)

    round2_facts = ""
    verification_context = context
    if followups:
        round2_docs = []
        for phrase in followups:
            progress(f"      follow-up search: {phrase}")
            round2_docs += fetch_docs(phrase, n=2, seen=seen)
        if round2_docs:
            question2 = f"{query}. {'; '.join(followups)}"
            context2, used2 = retrieve(question2, round2_docs, k=TOP_K_ROUND2, use_rag=rag_available)
            verification_context += "\n\n---\n\n" + context2
            round2_facts = researcher(query, context2)
            facts = facts + "\n\nADDITIONAL FACTS (round 2):\n" + round2_facts
            known = {s["url"] for s in used}
            used += [s for s in used2 if s["url"] not in known]
    else:
        progress("      critic gave no usable follow-up searches; skipping round 2")

    progress("5/5  Writing report...")
    body = writer(query, facts, critique, previous_report)
    progress("      verifying report sentences against retrieved evidence...")
    verification = verify_report(body, verification_context, embed, ask)
    report = verification["report"] + format_references(used)

    progress("Done.")
    return {
        "plan": plan,
        "context": context,
        "verification_context": verification_context,
        "facts": facts,
        "critique": critique,
        "followups": followups,
        "round2_facts": round2_facts,
        "sources": used,
        "report": report,
        "verification": verification,
        "mode": "RAG (ChromaDB vector search)" if rag_available else "Plain excerpts (no embeddings)",
        "seconds": round(time.time() - started),
    }


def save_report_file(query, report):
    """Write the report to reports/<timestamp>.txt and return the path."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(REPORTS_DIR, f"report_{stamp}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"QUERY: {query}\nGENERATED: {datetime.now():%Y-%m-%d %H:%M}\n\n{report}\n")
    return path




def main():
    query = input("Research question: ").strip()
    ok, reason = validate_query(query)
    if not ok:
        print(reason)
        return

    previous = get_last_report()
    if previous:
        print("(Previous report found in memory; this run will build on it.)")

    try:
        result = run_research(query, previous)
    except ResearchError as e:
        print("\nERROR:", e)
        return

    print("\n" + "=" * 60)
    print(result["report"])

    save_memory(query, result["report"])
    path = save_report_file(query, result["report"])
    print(f"\nMode: {result['mode']}  |  Time: {result['seconds']}s")
    print(f"Saved to {path} and memory.json")


if __name__ == "__main__":
    main()
