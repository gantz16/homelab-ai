from benchmark.models import BenchmarkTest, CheckResult


def final_answer_is_nine(response: str) -> CheckResult:
    normalized = response.lower()

    good_phrases = (
        "9 sheep",
        "nine sheep",
        "9 remain",
        "nine remain",
        "answer is 9",
        "answer is nine",
    )

    passed = any(phrase in normalized for phrase in good_phrases)

    return CheckResult(
        name="Final answer is 9",
        passed=passed,
        detail="Expected the model to conclude that 9 sheep remain.",
    )


def does_not_answer_eight(response: str) -> CheckResult:
    normalized = response.lower()

    bad_phrases = (
        "8 sheep",
        "eight sheep",
        "8 remain",
        "eight remain",
        "answer is 8",
        "answer is eight",
        "17 - 9 = 8",
    )

    passed = not any(phrase in normalized for phrase in bad_phrases)

    return CheckResult(
        name="Does not answer 8",
        passed=passed,
        detail='The phrase "all but 9" means 9 remain; subtraction is incorrect.',
    )


TESTS = [
    BenchmarkTest(
        test_id="reasoning-001",
        name="All But Nine",
        category="Reasoning",
        system_prompt=(
            "Read the wording literally. Answer the exact question asked. "
            "Give the final answer and one brief explanation."
        ),
        user_prompt=(
            "A farmer has 17 sheep. All but 9 run away. "
            "How many sheep remain?"
        ),
        max_tokens=80,
        checks=[
            final_answer_is_nine,
            does_not_answer_eight,
        ],
    )
]
