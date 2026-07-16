from benchmark.models import BenchmarkTest, CheckResult


def contains_any(response: str, phrases: tuple[str, ...]) -> bool:
    text = response.lower()
    return any(phrase.lower() in text for phrase in phrases)


def final_answer_is_nine(response: str) -> CheckResult:
    passed = contains_any(
        response,
        (
            "9 sheep",
            "nine sheep",
            "9 remain",
            "nine remain",
            "answer is 9",
            "answer is nine",
        ),
    )

    return CheckResult(
        name="Final answer is 9",
        passed=passed,
        detail="The phrase 'all but 9' means that 9 sheep remain.",
    )


def does_not_answer_eight(response: str) -> CheckResult:
    bad = contains_any(
        response,
        (
            "8 sheep",
            "eight sheep",
            "8 remain",
            "eight remain",
            "answer is 8",
            "answer is eight",
            "17 - 9 = 8",
        ),
    )

    return CheckResult(
        name="Does not answer 8",
        passed=not bad,
        detail="'All but 9' is an exception statement, not subtraction.",
    )


def pills_answer_is_one_hour(response: str) -> CheckResult:
    good = contains_any(
        response,
        (
            "1 hour",
            "one hour",
            "60 minutes",
        ),
    )
    bad = contains_any(
        response,
        (
            "90 minutes",
            "1.5 hours",
            "one and a half hours",
        ),
    )

    return CheckResult(
        name="Final answer is one hour",
        passed=good and not bad,
        detail=(
            "The pills are taken at time 0, 30 minutes, and 60 minutes."
        ),
    )


def bat_ball_answer_is_five_cents(response: str) -> CheckResult:
    good = contains_any(
        response,
        (
            "5 cents",
            "five cents",
            "$0.05",
            "0.05",
        ),
    )
    bad = contains_any(
        response,
        (
            "10 cents",
            "ten cents",
            "$0.10",
            "0.10",
        ),
    )

    return CheckResult(
        name="Ball costs five cents",
        passed=good and not bad,
        detail=(
            "If the ball costs $0.05, the bat costs $1.05, totaling $1.10."
        ),
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
    ),
    BenchmarkTest(
        test_id="reasoning-002",
        name="Three Pills",
        category="Reasoning",
        system_prompt=(
            "Solve the timing problem carefully. State the answer and briefly "
            "list the times at which each pill is taken."
        ),
        user_prompt=(
            "A doctor gives you three pills and tells you to take one pill "
            "every half hour. How long will it take until all three pills "
            "have been taken?"
        ),
        max_tokens=100,
        checks=[pills_answer_is_one_hour],
    ),
    BenchmarkTest(
        test_id="reasoning-003",
        name="Bat and Ball",
        category="Reasoning",
        system_prompt=(
            "Solve the arithmetic carefully. Verify that the two prices add "
            "to the stated total before answering."
        ),
        user_prompt=(
            "A bat and a ball cost $1.10 in total. The bat costs exactly "
            "$1.00 more than the ball. How much does the ball cost?"
        ),
        max_tokens=100,
        checks=[bat_ball_answer_is_five_cents],
    ),
]
