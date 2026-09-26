"""Evaluate structured-output and keyword coverage on synthetic cases."""

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


def _expected_keywords(case: Mapping[str, Any]) -> tuple[str, ...]:
    keywords = case.get("expected_keywords", [])
    if not isinstance(keywords, list) or not all(
        isinstance(keyword, str) and keyword.strip() for keyword in keywords
    ):
        raise ValueError("expected_keywords must be an array of non-empty strings")
    return tuple(keyword.strip() for keyword in keywords)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    args = parser.parse_args()

    cases = load_cases(args.cases)
    tokenizer, model = load_model()
    structured_count = 0
    keyword_hits = 0
    keyword_total = 0
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
        expected_keywords = _expected_keywords(case)
        keyword_total += len(expected_keywords)

        try:
            summary_lines = parse_summary_response(raw_response)
            structured_count += 1
            combined = " ".join(summary_lines)
            hits = sum(keyword in combined for keyword in expected_keywords)
            keyword_hits += hits
            error = None
        except ValueError as exc:
            summary_lines = ()
            hits = 0
            error = str(exc)

        results.append(
            {
                "case_id": case.get("case_id", index),
                "structured_output": error is None,
                "keyword_hits": hits,
                "keyword_total": len(expected_keywords),
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
        "keyword_coverage": round(keyword_hits / keyword_total, 4)
        if keyword_total
        else None,
        "average_latency_seconds": round(sum(latencies) / len(latencies), 2),
        "results": results,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

