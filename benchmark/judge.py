"""Semantic evaluation of saved benchmark runs.

This module uses a stronger language model as a judge. It reads a completed
candidate benchmark, reconstructs each test's instructions and rubric, asks
the judge to evaluate the candidate response, and saves a structured report.

Deterministic checks are retained as diagnostic evidence. They are not treated
as unquestionable truth because phrase-based checks can miss semantically
correct answers that use unexpected wording.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmark.client import FLMClient
from benchmark.models import BenchmarkTest
from benchmark.registry import discover_tests


# The judge's required scoring dimensions always total ten points.
DIMENSION_MAXIMUMS = {
    "factual_accuracy": 4,
    "instruction_following": 2,
    "completeness": 2,
    "usefulness": 2,
}


def print_metric(label: str, value: str) -> None:
    """Print one consistently aligned terminal metric."""

    print(f"{label:<28}{value}")


def load_json_file(path: Path) -> dict[str, Any]:
    """Load and validate a JSON object from disk."""

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Result file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Result file is not valid JSON: {path}") from exc

    if not isinstance(data, dict):
        raise RuntimeError("The result file must contain one top-level object.")

    if not isinstance(data.get("results"), list):
        raise RuntimeError("The result file does not contain a results list.")

    return data


def build_test_registry() -> dict[str, BenchmarkTest]:
    """Return all discovered tests indexed by their stable test identifier."""

    return {test.test_id: test for test in discover_tests()}


def format_static_checks(result: dict[str, Any]) -> str:
    """Convert prior deterministic checks into readable advisory evidence."""

    checks = result.get("checks", [])

    if not checks:
        return "No deterministic checks were recorded."

    lines: list[str] = []

    for check in checks:
        status = "PASS" if check.get("passed") else "FAIL"
        name = str(check.get("name", "Unnamed check"))
        detail = str(check.get("detail", "")).strip()

        lines.append(f"- [{status}] {name}")

        if detail:
            lines.append(f"  Explanation: {detail}")

    return "\n".join(lines)


def build_judge_system_prompt() -> str:
    """Create the fixed system instruction used for every judging request."""

    return """
You are the quality judge for a local-language-model benchmark.

Evaluate meaning, not keyword overlap. Do not penalize a candidate merely
because it uses different wording from an expected answer.

Use only:
1. the original system instruction,
2. the original user request,
3. the candidate response,
4. the supplied rubric and deterministic-check notes.

The deterministic checks are advisory. They can be wrong when a correct answer
uses unexpected wording. Independently inspect the actual response.

Scoring:
- factual_accuracy: 0 to 4
- instruction_following: 0 to 2
- completeness: 0 to 2
- usefulness: 0 to 2
- total score must equal the sum and therefore range from 0 to 10

Verdicts:
- PASS: 8 to 10, with no critical error
- PARTIAL: 5 to 7, or otherwise useful but materially incomplete
- FAIL: 0 to 4, or any critical factual/safety error

A critical error is a materially false, unsafe, fabricated, or task-defeating
claim. Minor omissions and stylistic weaknesses are not critical errors.

For every issue, quote exact evidence from the candidate response whenever
possible. Do not invent quotations.

Return only one valid JSON object. Do not use Markdown or code fences.

Required schema:
{
  "verdict": "PASS | PARTIAL | FAIL",
  "score": 0,
  "confidence": 0.0,
  "critical_error": false,
  "dimensions": {
    "factual_accuracy": 0,
    "instruction_following": 0,
    "completeness": 0,
    "usefulness": 0
  },
  "summary": "brief overall assessment",
  "strengths": ["specific strength"],
  "issues": [
    {
      "severity": "minor | major | critical",
      "description": "what is wrong or missing",
      "evidence": "exact candidate wording, or an empty string"
    }
  ]
}
""".strip()


def build_judge_user_prompt(
    test: BenchmarkTest,
    candidate_response: str,
    static_checks: str,
) -> str:
    """Assemble the complete evidence packet for one candidate response."""

    rubric_lines = []

    for check in test.checks:
        # Calling the checker is not necessary here. Its function name is less
        # useful than the human-readable CheckResult it produced during the run.
        rubric_lines.append(
            "- Evaluate whether the response satisfies the benchmark's "
            "documented expectations and avoids the documented errors."
        )

    if not rubric_lines:
        rubric_lines.append(
            "- Judge factual accuracy, instruction-following, completeness, "
            "and practical usefulness."
        )

    rubric_text = "\n".join(rubric_lines)

    return f"""
TEST IDENTIFIER:
{test.test_id}

TEST NAME:
{test.name}

CATEGORY:
{test.category}

ORIGINAL SYSTEM INSTRUCTION:
{test.system_prompt}

ORIGINAL USER REQUEST:
{test.user_prompt}

CANDIDATE RESPONSE:
{candidate_response}

DETERMINISTIC CHECK RESULTS AND EXPECTATIONS:
{static_checks}

GENERAL RUBRIC:
{rubric_text}

Judge the candidate response according to the required JSON schema.
""".strip()


def extract_json_object(raw_text: str) -> dict[str, Any]:
    """Extract one JSON object from a model response.

    The preferred response is pure JSON. The fallback extraction tolerates a
    judge that accidentally surrounds the object with prose or code fences.
    """

    cleaned = raw_text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned = "\n".join(lines).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        opening = cleaned.find("{")
        closing = cleaned.rfind("}")

        if opening == -1 or closing == -1 or closing <= opening:
            raise RuntimeError("Judge response did not contain a JSON object.")

        try:
            parsed = json.loads(cleaned[opening : closing + 1])
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Judge response contained malformed JSON."
            ) from exc

    if not isinstance(parsed, dict):
        raise RuntimeError("Judge response JSON was not an object.")

    return parsed


def coerce_integer(value: Any, minimum: int, maximum: int) -> int:
    """Convert a value to an integer and clamp it to the allowed range."""

    try:
        number = int(value)
    except (TypeError, ValueError):
        number = minimum

    return max(minimum, min(maximum, number))


def coerce_float(value: Any, minimum: float, maximum: float) -> float:
    """Convert a value to a float and clamp it to the allowed range."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        number = minimum

    return max(minimum, min(maximum, number))


def normalize_text_list(value: Any) -> list[str]:
    """Normalize an arbitrary value into a clean list of non-empty strings."""

    if not isinstance(value, list):
        return []

    return [
        str(item).strip()
        for item in value
        if str(item).strip()
    ]


def normalize_issues(value: Any) -> list[dict[str, str]]:
    """Validate and normalize the judge's issue records."""

    if not isinstance(value, list):
        return []

    issues: list[dict[str, str]] = []

    for item in value:
        if not isinstance(item, dict):
            continue

        severity = str(item.get("severity", "minor")).lower().strip()

        if severity not in {"minor", "major", "critical"}:
            severity = "minor"

        description = str(item.get("description", "")).strip()
        evidence = str(item.get("evidence", "")).strip()

        if description:
            issues.append(
                {
                    "severity": severity,
                    "description": description,
                    "evidence": evidence,
                }
            )

    return issues


def normalize_judgment(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate judge output and calculate a consistent score and verdict."""

    raw_dimensions = raw.get("dimensions", {})

    if not isinstance(raw_dimensions, dict):
        raw_dimensions = {}

    dimensions = {
        name: coerce_integer(
            raw_dimensions.get(name, 0),
            minimum=0,
            maximum=maximum,
        )
        for name, maximum in DIMENSION_MAXIMUMS.items()
    }

    # Derive the authoritative score from the dimensions instead of trusting
    # a possibly inconsistent total supplied by the judge.
    score = sum(dimensions.values())
    critical_error = bool(raw.get("critical_error", False))

    if critical_error or score <= 4:
        verdict = "FAIL"
    elif score <= 7:
        verdict = "PARTIAL"
    else:
        verdict = "PASS"

    return {
        "verdict": verdict,
        "score": score,
        "maximum_score": 10,
        "confidence": coerce_float(
            raw.get("confidence", 0.0),
            minimum=0.0,
            maximum=1.0,
        ),
        "critical_error": critical_error,
        "dimensions": dimensions,
        "summary": str(raw.get("summary", "")).strip(),
        "strengths": normalize_text_list(raw.get("strengths", [])),
        "issues": normalize_issues(raw.get("issues", [])),
    }


def request_judgment(
    client: FLMClient,
    judge_model: str,
    test: BenchmarkTest,
    candidate_response: str,
    static_checks: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Ask the judge model to evaluate one response.

    One retry is allowed if the first response is not valid structured JSON.
    Metrics from the successful request are returned alongside the judgment.
    """

    system_prompt = build_judge_system_prompt()
    user_prompt = build_judge_user_prompt(
        test=test,
        candidate_response=candidate_response,
        static_checks=static_checks,
    )

    last_error: Exception | None = None

    for attempt in range(1, 3):
        retry_suffix = ""

        if attempt == 2:
            retry_suffix = (
                "\n\nYour previous response could not be parsed. Return only "
                "the required valid JSON object with no surrounding text."
            )

        inference = client.chat(
            model=judge_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt + retry_suffix,
            max_tokens=900,
            temperature=0.0,
            timeout=240,
        )

        try:
            parsed = extract_json_object(inference.content)
            judgment = normalize_judgment(parsed)

            judge_metrics = {
                "prompt_tokens": inference.prompt_tokens,
                "completion_tokens": inference.completion_tokens,
                "total_tokens": inference.total_tokens,
                "ttft": inference.ttft,
                "prefill_speed_tps": inference.prefill_speed_tps,
                "decode_speed_tps": inference.decode_speed_tps,
                "decode_seconds": inference.decode_seconds,
                "request_seconds": inference.request_seconds,
                "attempts": attempt,
                "raw_judge_response": inference.content,
            }

            return judgment, judge_metrics
        except RuntimeError as exc:
            last_error = exc

    raise RuntimeError(
        f"Judge failed to produce valid JSON after two attempts: {last_error}"
    )


def print_judgment(
    test_number: int,
    total_tests: int,
    test: BenchmarkTest,
    judgment: dict[str, Any],
    judge_metrics: dict[str, Any],
) -> None:
    """Print one complete human-readable judging result."""

    print()
    print("=" * 88)
    print(
        f"[{test_number:02d}/{total_tests:02d}] "
        f"{test.test_id} — {test.name}"
    )
    print("=" * 88)

    print_metric("Category:", test.category)
    print_metric("Verdict:", judgment["verdict"])
    print_metric(
        "Judge score:",
        f"{judgment['score']}/{judgment['maximum_score']}",
    )
    print_metric(
        "Judge confidence:",
        f"{judgment['confidence'] * 100:.1f}%",
    )
    print_metric(
        "Critical error:",
        "YES" if judgment["critical_error"] else "No",
    )

    print()
    print("DIMENSIONS:")

    for name, maximum in DIMENSION_MAXIMUMS.items():
        label = name.replace("_", " ").title()
        print_metric(
            f"  {label}:",
            f"{judgment['dimensions'][name]}/{maximum}",
        )

    print()
    print("JUDGE SUMMARY:")
    print(judgment["summary"] or "No summary supplied.")

    if judgment["strengths"]:
        print()
        print("STRENGTHS:")

        for strength in judgment["strengths"]:
            print(f"  + {strength}")

    if judgment["issues"]:
        print()
        print("ISSUES:")

        for issue in judgment["issues"]:
            print(
                f"  - [{issue['severity'].upper()}] "
                f"{issue['description']}"
            )

            if issue["evidence"]:
                print(f'    Evidence: "{issue["evidence"]}"')

    print()
    print("JUDGE PERFORMANCE:")
    print_metric(
        "Judge prompt tokens:",
        str(judge_metrics["prompt_tokens"]),
    )
    print_metric(
        "Judge completion tokens:",
        str(judge_metrics["completion_tokens"]),
    )
    print_metric(
        "Judge TTFT:",
        f"{judge_metrics['ttft']:.3f} sec",
    )
    print_metric(
        "Judge decode:",
        f"{judge_metrics['decode_speed_tps']:.2f} tok/s",
    )
    print_metric(
        "Judge request time:",
        f"{judge_metrics['request_seconds']:.3f} sec",
    )
    print_metric(
        "JSON attempts:",
        str(judge_metrics["attempts"]),
    )


def print_test_breakdown(judged_results: list[dict[str, Any]]) -> None:
    """Print a recursive summary table containing every judged test."""

    print()
    print("=" * 112)
    print("SEMANTIC JUDGE — TEST BREAKDOWN")
    print("=" * 112)
    print(
        f"{'Test ID':<24}"
        f"{'Category':<27}"
        f"{'Verdict':<10}"
        f"{'Score':>8}"
        f"{'Confidence':>13}"
        f"{'Critical':>11}"
        f"{'Judge tok/s':>15}"
    )
    print("-" * 112)

    for result in judged_results:
        judgment = result.get("semantic_judgment")
        metrics = result.get("judge_metrics")

        if not judgment or not metrics:
            print(
                f"{result.get('test_id', 'unknown'):<24}"
                f"{result.get('category', 'unknown'):<27}"
                f"{'ERROR':<10}"
                f"{'-':>8}"
                f"{'-':>13}"
                f"{'-':>11}"
                f"{'-':>15}"
            )
            continue

        print(
            f"{result['test_id']:<24}"
            f"{result['category']:<27}"
            f"{judgment['verdict']:<10}"
            f"{judgment['score']:>5}/10"
            f"{judgment['confidence'] * 100:>12.1f}%"
            f"{('YES' if judgment['critical_error'] else 'No'):>11}"
            f"{metrics['decode_speed_tps']:>11.2f} t/s"
        )

    print("=" * 112)


def print_category_summary(judged_results: list[dict[str, Any]]) -> None:
    """Aggregate semantic scores automatically by test category."""

    categories: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "tests": 0,
            "pass": 0,
            "partial": 0,
            "fail": 0,
            "score": 0,
            "possible": 0,
        }
    )

    for result in judged_results:
        judgment = result.get("semantic_judgment")

        if not judgment:
            continue

        category = result.get("category", "Uncategorized")
        summary = categories[category]

        summary["tests"] += 1
        summary["score"] += judgment["score"]
        summary["possible"] += judgment["maximum_score"]
        summary[judgment["verdict"].lower()] += 1

    print()
    print("=" * 92)
    print("SEMANTIC JUDGE — CATEGORY SUMMARY")
    print("=" * 92)
    print(
        f"{'Category':<32}"
        f"{'Tests':>8}"
        f"{'Pass':>8}"
        f"{'Partial':>10}"
        f"{'Fail':>8}"
        f"{'Score':>13}"
        f"{'Percent':>11}"
    )
    print("-" * 92)

    for category in sorted(categories):
        summary = categories[category]
        percentage = (
            100 * summary["score"] / summary["possible"]
            if summary["possible"]
            else 0.0
        )

        print(
            f"{category:<32}"
            f"{summary['tests']:>8}"
            f"{summary['pass']:>8}"
            f"{summary['partial']:>10}"
            f"{summary['fail']:>8}"
            f"{summary['score']:>6}/{summary['possible']:<6}"
            f"{percentage:>9.1f}%"
        )

    print("=" * 92)


def run() -> int:
    """Run semantic judging for every response in one saved benchmark file."""

    parser = argparse.ArgumentParser(
        description="Judge a saved AI Gauntlet result semantically."
    )
    parser.add_argument(
        "result_file",
        help="Path to the candidate benchmark JSON file.",
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-oss:20b",
        help="Model currently served by FLM and used as the judge.",
    )
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8000",
        help="OpenAI-compatible FLM endpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark/judgments",
        help="Directory for judged JSON reports.",
    )
    args = parser.parse_args()

    result_path = Path(args.result_file)
    source_run = load_json_file(result_path)
    test_registry = build_test_registry()
    client = FLMClient(args.endpoint)

    candidate_model = str(source_run.get("model", "unknown"))
    source_results = source_run["results"]
    started_at = datetime.now()

    print("=" * 88)
    print("JOHN'S LOCAL AI GAUNTLET — SEMANTIC JUDGE")
    print("=" * 88)
    print_metric("Candidate model:", candidate_model)
    print_metric("Judge model:", args.judge_model)
    print_metric("Source file:", str(result_path))
    print_metric("Tests:", str(len(source_results)))
    print_metric(
        "Self-judged:",
        "YES" if candidate_model == args.judge_model else "No",
    )
    print_metric(
        "Started:",
        started_at.isoformat(timespec="seconds"),
    )
    print("=" * 88)

    judged_results: list[dict[str, Any]] = []

    for index, source_result in enumerate(source_results, start=1):
        test_id = str(source_result.get("test_id", ""))
        test = test_registry.get(test_id)
        copied_result = dict(source_result)

        if test is None:
            copied_result["judge_error"] = (
                f"No current benchmark definition found for {test_id}."
            )
            judged_results.append(copied_result)
            print(f"\nERROR: Unknown test identifier: {test_id}")
            continue

        candidate_response = str(source_result.get("response", "")).strip()

        if not candidate_response:
            copied_result["judge_error"] = "Candidate response was empty."
            judged_results.append(copied_result)
            print(f"\nERROR: Empty response for {test_id}")
            continue

        static_checks = format_static_checks(source_result)

        try:
            judgment, judge_metrics = request_judgment(
                client=client,
                judge_model=args.judge_model,
                test=test,
                candidate_response=candidate_response,
                static_checks=static_checks,
            )

            copied_result["semantic_judgment"] = judgment
            copied_result["judge_metrics"] = judge_metrics

            print_judgment(
                test_number=index,
                total_tests=len(source_results),
                test=test,
                judgment=judgment,
                judge_metrics=judge_metrics,
            )
        except Exception as exc:
            copied_result["judge_error"] = str(exc)
            print(f"\nERROR judging {test_id}: {exc}")

        judged_results.append(copied_result)

    completed_at = datetime.now()
    elapsed_seconds = (completed_at - started_at).total_seconds()

    print_test_breakdown(judged_results)
    print_category_summary(judged_results)

    successful_judgments = [
        result
        for result in judged_results
        if "semantic_judgment" in result
    ]

    successful_metrics = [
        result["judge_metrics"]
        for result in successful_judgments
    ]

    score = sum(
        result["semantic_judgment"]["score"]
        for result in successful_judgments
    )
    possible = sum(
        result["semantic_judgment"]["maximum_score"]
        for result in successful_judgments
    )

    print()
    print("=" * 72)
    print("SEMANTIC JUDGE — FINAL SUMMARY")
    print("=" * 72)
    print_metric("Candidate model:", candidate_model)
    print_metric("Judge model:", args.judge_model)
    print_metric("Judged tests:", str(len(successful_judgments)))
    print_metric("Semantic score:", f"{score}/{possible}")

    if possible:
        print_metric("Semantic percent:", f"{100 * score / possible:.1f}%")

    print_metric("Judge elapsed:", f"{elapsed_seconds:.2f} sec")

    if successful_metrics:
        print_metric(
            "Average judge TTFT:",
            f"{statistics.mean(m['ttft'] for m in successful_metrics):.3f} sec",
        )
        print_metric(
            "Average judge decode:",
            f"{statistics.mean(m['decode_speed_tps'] for m in successful_metrics):.2f} tok/s",
        )
        print_metric(
            "Judge prompt tokens:",
            str(sum(m["prompt_tokens"] for m in successful_metrics)),
        )
        print_metric(
            "Judge completion tokens:",
            str(sum(m["completion_tokens"] for m in successful_metrics)),
        )

    judged_report = {
        "judge_format_version": "0.1.0",
        "source_result_file": str(result_path),
        "candidate_model": candidate_model,
        "judge_model": args.judge_model,
        "self_judged": candidate_model == args.judge_model,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "semantic_score": score,
        "semantic_possible": possible,
        "source_run": source_run,
        "judged_results": judged_results,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = started_at.strftime("%Y%m%d-%H%M%S")
    safe_candidate = (
        candidate_model.replace(":", "-")
        .replace("/", "-")
        .replace(" ", "_")
    )
    safe_judge = (
        args.judge_model.replace(":", "-")
        .replace("/", "-")
        .replace(" ", "_")
    )

    output_path = (
        output_dir
        / f"{timestamp}_{safe_candidate}_judged-by_{safe_judge}.json"
    )

    output_path.write_text(
        json.dumps(judged_report, indent=2),
        encoding="utf-8",
    )

    print_metric("Saved judgment:", str(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
