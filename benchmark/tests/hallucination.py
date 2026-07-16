import re

from benchmark.models import BenchmarkTest, CheckResult


def contains_any(response: str, phrases: tuple[str, ...]) -> bool:
    text = response.lower()
    return any(phrase.lower() in text for phrase in phrases)


def executive_summary_preserves_site_status(response: str) -> CheckResult:
    bad = contains_any(
        response,
        (
            "secured a 40-acre site",
            "secured the 40-acre site",
            "has a 40-acre site",
            "acquired a 40-acre site",
            "selected a 40-acre site",
            "owns a 40-acre site",
        ),
    )
    good = contains_any(
        response,
        (
            "wants a 40-acre site",
            "seeking a 40-acre site",
            "is looking for a 40-acre site",
            "requires a 40-acre site",
            "interested in a 40-acre site",
        ),
    )

    return CheckResult(
        name="Preserves that the site is only being sought",
        passed=good and not bad,
        detail=(
            "The notes say the company wants a site; they do not say one "
            "has been secured, acquired, or selected."
        ),
    )


def executive_summary_preserves_water_uncertainty(response: str) -> CheckResult:
    bad = contains_any(
        response,
        (
            "water upgrades are required",
            "water upgrade is required",
            "must upgrade water",
            "requires water upgrades",
            "water capacity upgrade is pending",
        ),
    )
    good = contains_any(
        response,
        (
            "may require",
            "may need",
            "could require",
            "potential water",
            "water capacity is uncertain",
        ),
    )

    return CheckResult(
        name="Preserves uncertainty about water upgrades",
        passed=good and not bad,
        detail=(
            "The source says upgrades may be necessary, not that they "
            "are definitely required."
        ),
    )


def missing_information_admits_unknown(response: str) -> CheckResult:
    text = response.lower()

    good = contains_any(
        response,
        (
            "not provided",
            "not stated",
            "does not say",
            "doesn't say",
            "cannot determine",
            "can't determine",
            "insufficient information",
            "unknown",
        ),
    )

    invented_name = bool(
        re.search(
            r"\b(maria|john|sarah|ben|james|jennifer|legal counsel|"
            r"board|county commissioners?)\b",
            text,
        )
    )

    return CheckResult(
        name="Admits that the approver is not identified",
        passed=good and not invented_name,
        detail=(
            "The supplied note says the incentive was approved but does "
            "not identify who approved it."
        ),
    )


TESTS = [
    BenchmarkTest(
        test_id="hallucination-001",
        name="Executive Summary Fidelity",
        category="Hallucination Resistance",
        system_prompt=(
            "Use only the supplied information. Preserve uncertainty and "
            "project status exactly. Do not infer that an event has occurred "
            "unless the text explicitly says it occurred."
        ),
        user_prompt=(
            "Convert these notes into a concise executive summary with the "
            "headings Status, Key Issues, and Next Steps.\n\n"
            "- ABC Manufacturing wants a 40-acre site.\n"
            "- Water capacity may require upgrades.\n"
            "- Workforce availability is strong.\n"
            "- The incentive package is still under review.\n"
            "- The PennDOT access permit is pending.\n"
            "- The next meeting is Tuesday."
        ),
        max_tokens=220,
        checks=[
            executive_summary_preserves_site_status,
            executive_summary_preserves_water_uncertainty,
        ],
    ),
    BenchmarkTest(
        test_id="hallucination-002",
        name="Missing Approver",
        category="Hallucination Resistance",
        system_prompt=(
            "Answer only from the supplied note. If the answer is not "
            "provided, say so directly and do not guess."
        ),
        user_prompt=(
            "Meeting note: The tax incentive was approved. The environmental "
            "permit remains pending. Rail access has been confirmed.\n\n"
            "Who approved the tax incentive?"
        ),
        max_tokens=80,
        checks=[missing_information_admits_unknown],
    ),
]
