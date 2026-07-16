from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmark.client import FLMClient
from benchmark.registry import discover_tests


def print_metric(label: str, value: str) -> None:
    print(f"{label:<24}{value}")


def run() -> int:
    parser = argparse.ArgumentParser(
        description="Run John's Local AI Gauntlet."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--results-dir",
        default="benchmark/results",
    )
    args = parser.parse_args()

    tests = discover_tests()

    if not tests:
        print("No benchmark tests were discovered.")
        return 1

    client = FLMClient(args.endpoint)
    started_at = datetime.now()

    print("=" * 64)
    print("JOHN'S LOCAL AI GAUNTLET")
    print("=" * 64)
    print_metric("Model:", args.model)
    print_metric("Endpoint:", args.endpoint)
    print_metric("Tests:", str(len(tests)))
    print_metric("Started:", started_at.isoformat(timespec="seconds"))
    print("=" * 64)

    results: list[dict[str, Any]] = []
    total_score = 0
    total_possible = 0

    for index, test in enumerate(tests, start=1):
        print()
        print("=" * 64)
        print(f"[{index:02d}/{len(tests):02d}] {test.test_id} — {test.name}")
        print("=" * 64)
        print_metric("Category:", test.category)
        print_metric("Max output tokens:", str(test.max_tokens))
        print()
        print("PROMPT:")
        print(test.user_prompt)
        print()
        print("Running...")

        try:
            inference = client.chat(
                model=args.model,
                system_prompt=test.system_prompt,
                user_prompt=test.user_prompt,
                max_tokens=test.max_tokens,
                temperature=test.temperature,
            )
        except Exception as exc:
            print()
            print(f"ERROR: {exc}")
            results.append(
                {
                    "test_id": test.test_id,
                    "name": test.name,
                    "category": test.category,
                    "error": str(exc),
                    "score": 0,
                    "possible": len(test.checks),
                }
            )
            total_possible += len(test.checks)
            continue

        check_results = [check(inference.content) for check in test.checks]
        score = sum(1 for check in check_results if check.passed)
        possible = len(check_results)

        total_score += score
        total_possible += possible

        print()
        print("ANSWER:")
        print(inference.content)
        print()
        print("METRICS:")
        print_metric("Prompt tokens:", str(inference.prompt_tokens))
        print_metric("Completion tokens:", str(inference.completion_tokens))
        print_metric("Total tokens:", str(inference.total_tokens))
        print_metric("TTFT:", f"{inference.ttft:.3f} sec")
        print_metric(
            "Prefill speed:",
            f"{inference.prefill_speed_tps:.2f} tok/s",
        )
        print_metric(
            "Decode speed:",
            f"{inference.decode_speed_tps:.2f} tok/s",
        )
        print_metric("Decode time:", f"{inference.decode_seconds:.3f} sec")
        print_metric("Request time:", f"{inference.request_seconds:.3f} sec")

        print()
        print("CHECKS:")
        for check in check_results:
            marker = "PASS" if check.passed else "FAIL"
            print(f"[{marker}] {check.name}")
            if check.detail:
                print(f"       {check.detail}")

        if possible == 0:
            status = "UNSCORED"
        elif score == possible:
            status = "PASS"
        elif score == 0:
            status = "FAIL"
        else:
            status = "PARTIAL"

        print()
        print_metric("Result:", status)
        print_metric("Test score:", f"{score}/{possible}")
        print_metric("Running total:", f"{total_score}/{total_possible}")

        results.append(
            {
                "test_id": test.test_id,
                "name": test.name,
                "category": test.category,
                "status": status,
                "score": score,
                "possible": possible,
                "response": inference.content,
                "checks": [
                    {
                        "name": check.name,
                        "passed": check.passed,
                        "detail": check.detail,
                    }
                    for check in check_results
                ],
                "metrics": {
                    "prompt_tokens": inference.prompt_tokens,
                    "completion_tokens": inference.completion_tokens,
                    "total_tokens": inference.total_tokens,
                    "ttft": inference.ttft,
                    "prefill_speed_tps": inference.prefill_speed_tps,
                    "decode_speed_tps": inference.decode_speed_tps,
                    "decode_seconds": inference.decode_seconds,
                    "request_seconds": inference.request_seconds,
                },
                "raw_response": inference.raw_response,
            }
        )

    completed_at = datetime.now()
    elapsed_seconds = (completed_at - started_at).total_seconds()

    successful_metrics = [
        result["metrics"]
        for result in results
        if "metrics" in result
    ]

    print()
    print("=" * 118)
    print("TEST BREAKDOWN")
    print("=" * 118)
    print(
        f"{'Test ID':<24}"
        f"{'Category':<28}"
        f"{'Result':<10}"
        f"{'Score':<9}"
        f"{'TTFT':>9}"
        f"{'Decode':>13}"
        f"{'Prompt':>10}"
        f"{'Output':>10}"
    )
    print("-" * 118)

    for result in results:
        if "metrics" not in result:
            print(
                f"{result['test_id']:<24}"
                f"{result['category']:<28}"
                f"{'ERROR':<10}"
                f"{result.get('score', 0)}/{result.get('possible', 0):<7}"
                f"{'-':>9}"
                f"{'-':>13}"
                f"{'-':>10}"
                f"{'-':>10}"
            )
            continue

        metrics = result["metrics"]
        score_text = f"{result['score']}/{result['possible']}"

        print(
            f"{result['test_id']:<24}"
            f"{result['category']:<28}"
            f"{result['status']:<10}"
            f"{score_text:<9}"
            f"{metrics['ttft']:>8.3f}s"
            f"{metrics['decode_speed_tps']:>10.2f} t/s"
            f"{metrics['prompt_tokens']:>10}"
            f"{metrics['completion_tokens']:>10}"
        )

    print("=" * 118)

    category_summary: dict[str, dict[str, int]] = {}

    for result in results:
        category = result["category"]
        summary = category_summary.setdefault(
            category,
            {
                "tests": 0,
                "pass": 0,
                "partial": 0,
                "fail": 0,
                "error": 0,
                "score": 0,
                "possible": 0,
            },
        )

        summary["tests"] += 1
        summary["score"] += int(result.get("score", 0))
        summary["possible"] += int(result.get("possible", 0))

        status = result.get("status", "ERROR").lower()

        if status == "pass":
            summary["pass"] += 1
        elif status == "partial":
            summary["partial"] += 1
        elif status == "fail":
            summary["fail"] += 1
        else:
            summary["error"] += 1

    print()
    print("=" * 96)
    print("CATEGORY SUMMARY")
    print("=" * 96)
    print(
        f"{'Category':<32}"
        f"{'Tests':>7}"
        f"{'Pass':>7}"
        f"{'Partial':>9}"
        f"{'Fail':>7}"
        f"{'Error':>8}"
        f"{'Score':>12}"
    )
    print("-" * 96)

    for category in sorted(category_summary):
        summary = category_summary[category]
        score_text = f"{summary['score']}/{summary['possible']}"

        print(
            f"{category:<32}"
            f"{summary['tests']:>7}"
            f"{summary['pass']:>7}"
            f"{summary['partial']:>9}"
            f"{summary['fail']:>7}"
            f"{summary['error']:>8}"
            f"{score_text:>12}"
        )

    print("=" * 96)

    failed_checks = []

    for result in results:
        for check in result.get("checks", []):
            if not check["passed"]:
                failed_checks.append(
                    (
                        result["test_id"],
                        result["name"],
                        check["name"],
                        check.get("detail", ""),
                    )
                )

    if failed_checks:
        print()
        print("=" * 96)
        print("FAILED CHECKS")
        print("=" * 96)

        for test_id, name, check_name, detail in failed_checks:
            print(f"[{test_id}] {name}")
            print(f"  - {check_name}")
            if detail:
                print(f"    {detail}")

        print("=" * 96)

    print()
    print("=" * 64)
    print("FINAL SUMMARY")
    print("=" * 64)
    print_metric("Model:", args.model)
    print_metric("Tests:", str(len(results)))
    print_metric("Score:", f"{total_score}/{total_possible}")
    print_metric("Elapsed:", f"{elapsed_seconds:.2f} sec")

    if successful_metrics:
        ttfts = [metric["ttft"] for metric in successful_metrics]
        decode_speeds = [
            metric["decode_speed_tps"] for metric in successful_metrics
        ]

        prompt_token_total = sum(
            metric["prompt_tokens"] for metric in successful_metrics
        )
        completion_token_total = sum(
            metric["completion_tokens"] for metric in successful_metrics
        )

        print_metric("Average TTFT:", f"{statistics.mean(ttfts):.3f} sec")
        print_metric("Median TTFT:", f"{statistics.median(ttfts):.3f} sec")
        print_metric(
            "Average decode:",
            f"{statistics.mean(decode_speeds):.2f} tok/s",
        )
        print_metric(
            "Median decode:",
            f"{statistics.median(decode_speeds):.2f} tok/s",
        )
        print_metric("Prompt tokens:", str(prompt_token_total))
        print_metric("Completion tokens:", str(completion_token_total))
        print_metric(
            "All tokens:",
            str(prompt_token_total + completion_token_total),
        )

    output = {
        "gauntlet_version": "0.1.0",
        "model": args.model,
        "endpoint": args.endpoint,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "score": total_score,
        "possible": total_possible,
        "results": results,
    }

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    safe_model = (
        args.model.replace(":", "-")
        .replace("/", "-")
        .replace(" ", "_")
    )
    timestamp = started_at.strftime("%Y%m%d-%H%M%S")
    output_path = results_dir / f"{timestamp}_{safe_model}.json"

    output_path.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print_metric("Saved results:", str(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
