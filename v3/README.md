# Agentic Research Analyst — v3

Version 3 adds semantic retrieval to the v2 multi-stage research assistant. It is a local-model prototype that searches the live web, retrieves relevant passages, asks role-specific prompts to extract and critique evidence, and writes a report with retrieved-page references.

## Research question

Can a multi-stage research workflow with retrieval and a targeted critique round produce reports that are more evidence-grounded than a single prompt?

This is a research question, not a result. Version 3 has no controlled benchmark, so it does not establish that the pipeline reduces hallucinations.

## What v3 adds over v2

- Chunking, local embeddings, and in-memory ChromaDB retrieval to select passages for fact extraction.
- A compatibility check for the local chat and embedding models, with a plain-excerpt fallback when embeddings are unavailable.
- Query validation, cleaner report output, timestamped report exports, and improved status details in the UI.
- An explicit project description and methodology for the research workflow.

It retains v2's Streamlit interface, Critic-driven follow-up searches, and JSON report memory.

## Pipeline

```text
Question → Planner → Web search and extraction → Chunking and retrieval
         → Researcher → Critic → targeted follow-up search → Writer → References
```

## Requirements and usage

Install Ollama, then retrieve the configured models and Python packages:

```bash
ollama pull qwen2.5:1.5b
ollama pull nomic-embed-text
pip install -r requirements.txt
```

Run the terminal workflow or Streamlit UI:

```bash
python research.py
python -m streamlit run app.py
```

## Limitations

- No fixed evaluation set or measured comparison against single-prompt baselines.
- No sentence-level support check; a retrieved URL can appear in References even when it does not support every report claim.
- Source quality is not scored, page extraction may return thin or noisy text, and search results vary over time.
- The second round is limited to the searches returned by one Critic call.
- Local model quality and latency depend on the model and hardware.

Version 4 introduces an evaluation runner and sentence-level claim verification to address the first two gaps. See [v4](../v4/README.md).
