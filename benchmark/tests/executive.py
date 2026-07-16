from benchmark.models import BenchmarkTest, CheckResult


def contains_any(response: str, phrases: tuple[str, ...]) -> bool:
    text = response.lower()
    return any(phrase.lower() in text for phrase in phrases)


def preserves_followup_owner(response: str) -> CheckResult:
    good = contains_any(
        response,
        (
            "i will follow up",
            "i'll follow up",
            "i will reach out",
            "i'll reach out",
            "i plan to follow up",
        ),
    )

    bad = contains_any(
        response,
        (
            "please follow up",
            "please reach out",
            "you should follow up",
            "we ask that you follow up",
        ),
    )

    return CheckResult(
        name="Preserves Sarah's ownership of follow-up",
        passed=good and not bad,
        detail=(
            "The source says Sarah will perform the follow-up; the rewrite "
            "must not assign that task to the recipients."
        ),
    )


def preserves_meeting_uncertainty(response: str) -> CheckResult:
    good = contains_any(
        response,
        (
            "do not expect",
            "don't expect",
            "do not anticipate",
            "likely will not",
            "should not affect",
            "should not impact",
        ),
    )

    bad = contains_any(
        response,
        (
            "will not affect",
            "will not impact",
            "will proceed as scheduled",
            "is unaffected",
        ),
    )

    return CheckResult(
        name="Preserves uncertainty about meeting impact",
        passed=good and not bad,
        detail=(
            "The draft expresses an expectation, not certainty, about the "
            "meeting."
        ),
    )


def identifies_two_real_risks(response: str) -> CheckResult:
    environmental = contains_any(
        response,
        ("environmental permit", "environmental approval"),
    )
    water = contains_any(
        response,
        ("water capacity", "utility capacity", "water infrastructure"),
    )

    return CheckResult(
        name="Identifies permit and water capacity risks",
        passed=environmental and water,
        detail=(
            "Both the pending environmental permit and uncertain water "
            "capacity are genuine project risks."
        ),
    )


def does_not_treat_confirmed_items_as_risks(response: str) -> CheckResult:
    text = response.lower()

    rail_as_risk = (
        "rail access" in text
        and any(
            phrase in text
            for phrase in (
                "risk 3",
                "third risk",
                "3. rail",
                "risk: rail",
                "rail access risk",
            )
        )
    )

    approved_incentive_as_risk = (
        "tax incentive" in text
        and any(
            phrase in text
            for phrase in (
                "tax incentive risk",
                "risk: tax incentive",
                "3. tax incentive",
            )
        )
    )

    return CheckResult(
        name="Does not classify confirmed items as unresolved risks",
        passed=not rail_as_risk and not approved_incentive_as_risk,
        detail=(
            "Confirmed rail access and an approved tax incentive are not "
            "unresolved risks in the supplied facts."
        ),
    )


def identifies_announcement_timing(response: str) -> CheckResult:
    good = contains_any(
        response,
        (
            "announcement timing",
            "premature announcement",
            "announce before",
            "announcement before",
            "public announcement",
            "ceo wants to announce",
            "next month's announcement",
        ),
    )

    return CheckResult(
        name="Identifies premature announcement timing",
        passed=good,
        detail=(
            "Announcing before permits and utility feasibility are resolved "
            "creates reputational and execution risk."
        ),
    )


TESTS = [
    BenchmarkTest(
        test_id="executive-001",
        name="Diplomatic Email Rewrite",
        category="Executive Work",
        system_prompt=(
            "Rewrite the email professionally and concisely. Preserve who "
            "owns each action, preserve uncertainty, and do not add facts."
        ),
        user_prompt=(
            "Rewrite this email:\n\n"
            "Hi everyone,\n\n"
            "We are still waiting on PennDOT and have not heard back from "
            "legal. I do not think this will affect the meeting next week, "
            "but we probably need answers before we can finalize the "
            "incentive package. I will reach out again tomorrow.\n\n"
            "Thanks,\nSarah"
        ),
        max_tokens=180,
        checks=[
            preserves_followup_owner,
            preserves_meeting_uncertainty,
        ],
    ),
    BenchmarkTest(
        test_id="executive-002",
        name="Project Risk Prioritization",
        category="Executive Work",
        system_prompt=(
            "Use only the supplied project facts. Rank the top three actual "
            "unresolved risks. Do not call confirmed or approved items risks."
        ),
        user_prompt=(
            "A manufacturer wants to invest 80 million dollars and create "
            "250 jobs.\n\n"
            "Known facts:\n"
            "- Environmental permit pending\n"
            "- Rail access confirmed\n"
            "- Water capacity uncertain\n"
            "- Tax incentive approved\n"
            "- The CEO wants to announce the project next month\n\n"
            "List the top three risks in priority order. For each, explain "
            "why it matters and give one next action."
        ),
        max_tokens=260,
        checks=[
            identifies_two_real_risks,
            does_not_treat_confirmed_items_as_risks,
            identifies_announcement_timing,
        ],
    ),
]
