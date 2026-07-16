from benchmark.models import BenchmarkTest, CheckResult


def contains_any(response: str, phrases: tuple[str, ...]) -> bool:
    text = response.lower()
    return any(phrase.lower() in text for phrase in phrases)


def includes_core_mitosis_concepts(response: str) -> CheckResult:
    chromosome_copying = contains_any(
        response,
        (
            "dna is copied",
            "dna replication",
            "chromosomes are copied",
            "replicates its dna",
        ),
    )
    division = contains_any(
        response,
        (
            "two daughter cells",
            "two identical cells",
            "two genetically identical",
            "cell divides into two",
        ),
    )

    return CheckResult(
        name="Includes DNA copying and two daughter cells",
        passed=chromosome_copying and division,
        detail=(
            "A useful ninth-grade explanation should connect prior DNA "
            "replication with production of two daughter cells."
        ),
    )


def does_not_claim_mitosis_makes_gametes(response: str) -> CheckResult:
    text = response.lower()

    bad = any(
        phrase in text
        for phrase in (
            "mitosis produces sperm",
            "mitosis produces eggs",
            "mitosis makes gametes",
            "mitosis creates sex cells",
            "four daughter cells",
        )
    )

    return CheckResult(
        name="Does not confuse mitosis with meiosis",
        passed=not bad,
        detail=(
            "Mitosis should not be described as producing gametes or four "
            "daughter cells."
        ),
    )


TESTS = [
    BenchmarkTest(
        test_id="teacher-001",
        name="Ninth-Grade Mitosis Explanation",
        category="Science Teaching",
        system_prompt=(
            "Explain biology accurately for a ninth-grade student. Use clear "
            "language, preserve scientific accuracy, and avoid unnecessary "
            "jargon."
        ),
        user_prompt=(
            "Explain mitosis in about 120 words. Include why organisms use it "
            "and what the final products are."
        ),
        max_tokens=180,
        checks=[
            includes_core_mitosis_concepts,
            does_not_claim_mitosis_makes_gametes,
        ],
    )
]
