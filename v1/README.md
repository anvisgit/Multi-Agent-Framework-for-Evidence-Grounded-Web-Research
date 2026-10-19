# Agentic Research Analyst — v1

Version 1 is the first command-line prototype of the evidence-grounded research workflow. It turns a question into a search plan, gathers web pages, extracts facts, critiques gaps, and writes a report.

## Research workflow

1. A Planner creates search phrases for different angles of the question.
2. DuckDuckGo results are scraped and passed to the Researcher.
3. The Researcher extracts facts and attributes them to source URLs.
4. The Critic reviews gaps and weak claims.
5. The Writer creates a report saved as `report.txt`.

## What v1 includes

- Four role prompts: Planner, Researcher, Critic, and Writer.
- Web search and basic BeautifulSoup page extraction.
- Source-aware fact extraction and a structured report.
- A local command-line entry point in `main.py`.

## What v1 does not include

There is no second search round, report history, Streamlit UI, vector retrieval, or automatic claim verification. The Critic's feedback is passed to the Writer but does not trigger more research.

## Requirements and usage

Install Ollama and the model configured in `llm.py`, install the Python packages used by `web.py` and `llm.py`, then run:

```bash
python main.py
```

The project is a prototype. A listed URL is a retrieval trace, not proof that every sentence in the report is supported.
