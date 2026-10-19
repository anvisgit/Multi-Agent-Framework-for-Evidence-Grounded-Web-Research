# Agentic Research Analyst — v4

V4 extends the local-model, web research workflow with sentence-level evidence checks and a repeatable three-way evaluation harness. It is an experimental prototype: the evaluation has not yet been run on the machine used to prepare this version because Ollama and the project Python dependencies were unavailable.

## Research question

Does a multi-stage research pipeline with retrieval and claim verification reduce unsupported claims while retaining coverage of important facts, compared with a single prompt with no web access and a single prompt with scraped text?

The benchmark is designed to measure this question. No improvement claim is made until its runs complete.

## What v4 adds over v3

- **Claim-level verification:** each report sentence is matched to the closest retrieved passages using local embedding similarity, then judged by the configured model as supported, unsupported, or uncertain. Unsupported sentences are removed; uncertain sentences remain with an `[UNVERIFIED]` marker. The UI exposes each verdict and its nearest source URL.
- **Three-way benchmark runner:** 15 fixed research questions are run through (1) a single prompt without web access, (2) a single prompt with scraped text, and (3) the full pipeline.
- **Explicit metrics:** URL citation accuracy, unsupported-claim rate, and key-fact coverage are recorded per question and can be summarized as a Markdown table.
- **Practical retrieval improvements:** settings live in `config.py`, pages are scraped concurrently and cached in a small SQLite database for 24 hours, and operational failures use Python logging.
- **Unit coverage:** pytest cases cover query validation, search parsers, chunk boundaries, and claim verification. Install `requirements-dev.txt` to run them.
- **Version history:** README files explain what v1, v2, and v3 added and which gaps remained.

The benchmark scorer uses the same configured local model as a judge. Treat its scores as model-judged measurements, not independent ground truth.

## System flow

```text
Question → Plan → Search and scrape → Chunk, embed, retrieve
         → Extract facts → Critique → Follow-up search → Write report
         → Match each sentence to evidence → Keep, flag, or remove → References
```

## Installation

Install Ollama, download both models, and install the Python dependencies from this directory:

```bash
ollama pull qwen2.5:1.5b
ollama pull nomic-embed-text
pip install -r requirements.txt
```

Run the app or command-line workflow:

```bash
python research.py
python -m streamlit run app.py
# Unit tests
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## Evaluation

The dataset is in [`evaluation/questions.json`](evaluation/questions.json). It contains 15 fixed questions, expected key facts, and starting reference URLs. The harness collects web evidence, runs the three system variants, and writes raw per-question outputs and scores to a timestamped JSON file.

```bash
python evaluation/run_evaluation.py
# Quick smoke run on the first question
python evaluation/run_evaluation.py --limit 1 --output evaluation/smoke.json
python evaluation/summarize_results.py evaluation/smoke.json
```

Metrics use a 0–1 scale in JSON and percentages in the summary:

| Metric | Definition | Direction |
|---|---|---|
| Citation accuracy | Supported `[Source: URL]` citations divided by citation marks; uncited reports score 0 | Higher is better |
| Unsupported-claim rate | Claims judged unsupported divided by claims judged supported or unsupported; uncertain claims are excluded | Lower is better |
| Key-fact coverage | Required benchmark facts judged present in the report divided by all required facts | Higher is better |

The no-web baseline does not receive the scraped evidence in its prompt. The evaluator uses a separate copy of those pages after generation to score the baseline's claims. The full pipeline is evaluated against its own retrieved passages for faithfulness and its own cited URLs for citation accuracy.

### Results

No scores are reported yet. The preparation environment did not have Ollama, the configured models, or the project dependencies installed, so all 15 × 3 runs could not be executed. Run the harness on a machine with the prerequisites above, then replace the table with the output of `summarize_results.py`.

| System version | Questions completed | Citation accuracy ↑ | Unsupported-claim rate ↓ | Key-fact coverage ↑ |
|---|---:|---:|---:|---:|
| Single prompt, no web | 0 / 15 | — | — | — |
| Single prompt, scraped text | 0 / 15 | — | — | — |
| Full pipeline | 0 / 15 | — | — | — |

## Short findings and known failure modes

- **Observed result:** none yet; do not infer that v4 reduces hallucinations from the implementation alone.
- **Environment failure:** local model inference could not be attempted because Ollama and its models were unavailable in the preparation environment.
- **Expected retrieval failure:** inaccessible, script-rendered, or paywalled pages may yield empty or thin text, reducing both support checks and coverage.
- **Judge limitation:** the generator and evaluator use the same model family, so their errors may be correlated. Human review or a separately calibrated judge is needed for stronger claims.
- **Retrieval threshold:** an embedding mismatch can send a relevant passage to a weaker overlap fallback or mark a claim unsupported. The similarity threshold should be calibrated on reviewed examples before interpreting removal rates.
- **Temporal limits:** the web benchmark retrieves live pages and search results, so exact evidence can change between runs. Save each JSON result artifact with the model and date when publishing results.

## Project layout

```text
v4/
├── app.py
├── claim_verification.py
├── config.py
├── research.py
├── requirements.txt
├── evaluation/
│   ├── questions.json
│   ├── run_evaluation.py
│   └── summarize_results.py
└── tests/
    ├── test_claim_verification.py
    └── test_research.py
```

## Privacy and limitations

Chat and embedding inference run through local Ollama. Web queries are sent to the configured search provider, and pages are fetched from their hosts. V4 does not certify source truth, resolve conflicting sources automatically, or guarantee that a supported claim is accurate beyond the retrieved passage. The benchmark's live retrieval and model-as-judge scoring need to be interpreted with those constraints in mind.
