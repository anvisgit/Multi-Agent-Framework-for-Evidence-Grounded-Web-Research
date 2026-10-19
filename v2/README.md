# Agentic Research Analyst — v2

Version 2 extends the v1 command-line prototype with a Streamlit interface, a follow-up search round, and report memory.

## What v2 adds over v1

- A Critic-driven second round of targeted web searches.
- Persistent `memory.json` report history and follow-up context.
- A Streamlit interface for entering queries, reviewing stages, and downloading a report.
- Cleaner page extraction and unique-URL tracking across search rounds.

## Workflow

The system plans searches, retrieves web text, extracts source-attributed facts, critiques the findings, searches again for missing information, and writes a report. Earlier reports can be supplied as context for a follow-up question.

## Requirements and usage

Install Ollama and the configured model, then install the dependencies used by `research.py` and `app.py`. From this directory:

```bash
python -m streamlit run app.py
```

## Limitations carried forward

Like v1, v2 does not verify each final sentence against source passages. It has no semantic vector retrieval or fixed evaluation benchmark. Search results and page extraction vary with network conditions, and saved report memory is a simple JSON history rather than a searchable index.
