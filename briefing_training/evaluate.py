"""Evaluate structured output and source-fact coverage on synthetic cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .prompts import MODEL_NAME, parse_summary_response
from .smoke_test_model import (
    DEFAULT_CASES_PATH,
    generate_summary,
    load_cases,
    load_model,
)


def _expected_facts(case: Mapping[str, Any]) -> tuple[tuple[str, ...], ...]:
    facts = case.get("expected_facts", [])
    if not isinstance(facts, list):
        raise ValueError("expected_facts must be an array")

    normalized: list[tuple[str, ...]] = []
    for fact in facts:
        if not isinstance(fact, list) or not all(
            isinstance(alternative, str) and alternative.strip()
            for alternative in fact
        ):
            raise ValueError(
                "each expected fact must be an array of alternative phrases"
            )
        if not fact:
            raise ValueError("each expected fact must contain an alternative")
        normalized.append(tuple(alternative.strip() for alternative in fact))
    return tuple(normalized)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    args = parser.parse_args()

    cases = load_cases(args.cases)
    tokenizer, model = load_model()
    structured_count = 0
    fact_hits = 0
    fact_total = 0
    latencies: list[float] = []
    results: list[dict[str, Any]] = []

    for index, case in enumerate(cases):
        group = case.get("input")
        if not isinstance(group, Mapping):
            raise ValueError(f"case {index} does not contain an input object")

        raw_response, elapsed_seconds = generate_summary(
            tokenizer=tokenizer,
            model=model,
            group=group,
            seed=42 + index,
        )
        latencies.append(elapsed_seconds)
        expected_facts = _expected_facts(case)
        fact_total += len(expected_facts)

        try:
            summary_lines = parse_summary_response(raw_response)
            structured_count += 1
            combined = " ".join(summary_lines)
            hits = sum(
                any(alternative in combined for alternative in alternatives)
                for alternatives in expected_facts
            )
            fact_hits += hits
            error = None
        except ValueError as exc:
            summary_lines = ()
            hits = 0
            error = str(exc)

        results.append(
            {
                "case_id": case.get("case_id", index),
                "structured_output": error is None,
                "fact_hits": hits,
                "fact_total": len(expected_facts),
                "latency_seconds": round(elapsed_seconds, 2),
                "summary_lines": list(summary_lines),
                "error": error,
                "raw_response": raw_response,
            }
        )

    report = {
        "model": MODEL_NAME,
        "case_count": len(cases),
        "structured_output_rate": round(structured_count / len(cases), 4),
        "fact_coverage": round(fact_hits / fact_total, 4)
        if fact_total
        else None,
        "average_latency_seconds": round(sum(latencies) / len(latencies), 2),
        "results": results,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
