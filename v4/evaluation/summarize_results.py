"""Print aggregate metrics from a run_evaluation.py JSON artifact."""

import argparse
import json
from pathlib import Path


VARIANTS = ("single_prompt_no_web", "single_prompt_scraped_text", "full_pipeline")
METRICS = ("citation_accuracy", "unsupported_claim_rate", "coverage")


def summarize(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for variant in VARIANTS:
        subset = [r for r in payload["results"] if r["variant"] == variant]
        values = {}
        for metric in METRICS:
            scores = [r["metrics"][metric] for r in subset if r["metrics"].get(metric) is not None]
            values[metric] = sum(scores) / len(scores) if scores else None
        rows.append((variant, len(subset), values))
    lines = ["| System version | Questions | Citation accuracy ↑ | Unsupported-claim rate ↓ | Key-fact coverage ↑ |", "|---|---:|---:|---:|---:|"]
    for variant, count, values in rows:
        formatted = ["—" if values[m] is None else f"{100 * values[m]:.1f}%" for m in METRICS]
        lines.append(f"| {variant.replace('_', ' ')} | {count} | " + " | ".join(formatted) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    print(summarize(parser.parse_args().results))
