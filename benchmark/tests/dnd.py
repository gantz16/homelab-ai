from benchmark.models import BenchmarkTest, CheckResult


def contains_any(response: str, phrases: tuple[str, ...]) -> bool:
    text = response.lower()
    return any(phrase.lower() in text for phrase in phrases)


def recommends_area_control(response: str) -> CheckResult:
    good = contains_any(
        response,
        (
            "slow",
            "hypnotic pattern",
        ),
    )

    return CheckResult(
        name="Recommends an area-control spell",
        passed=good,
        detail=(
            "Slow or Hypnotic Pattern are the strongest supplied options "
            "against multiple approaching enemies."
        ),
    )


def does_not_invent_ray_of_frost_aoe(response: str) -> CheckResult:
    text = response.lower()

    bad = any(
        phrase in text
        for phrase in (
            "ray of frost on all three",
            "ray of frost the three",
            "freeze the three",
            "hits all three",
            "target the three wolves with ray",
            "ray of frost affects all",
        )
    )

    return CheckResult(
        name="Does not make Ray of Frost affect multiple wolves",
        passed=not bad,
        detail="Ray of Frost targets one creature.",
    )


def understands_hypnotic_pattern_area(response: str) -> CheckResult:
    text = response.lower()

    bad = any(
        phrase in text
        for phrase in (
            "target one wolf with hypnotic pattern",
            "hypnotic pattern on the injured wolf",
            "use hypnotic pattern against one",
        )
    )

    return CheckResult(
        name="Does not describe Hypnotic Pattern as single-target",
        passed=not bad,
        detail="Hypnotic Pattern affects an area rather than one creature.",
    )


def does_not_spend_misty_step_as_action(response: str) -> CheckResult:
    text = response.lower()

    bad = any(
        phrase in text
        for phrase in (
            "use your action to misty step",
            "cast misty step as your action",
            "action: misty step",
        )
    )

    return CheckResult(
        name="Does not call Misty Step an action",
        passed=not bad,
        detail="Misty Step normally uses a bonus action.",
    )


TESTS = [
    BenchmarkTest(
        test_id="dnd-001",
        name="Wizard Combat Tactics",
        category="D&D",
        system_prompt=(
            "You are a careful Dungeons & Dragons 5e combat assistant. Use "
            "only the supplied abilities and battlefield facts. Do not invent "
            "effects, targets, conditions, or action costs. Give the best "
            "action first and one backup option."
        ),
        user_prompt=(
            "I am a level 14 illusion wizard. Three dire wolves are 25 feet "
            "ahead and moving toward our back line. One is injured. My fighter "
            "ally is engaged with two other wolves 40 feet to my right. I have "
            "an action, movement, bonus action, and reaction available. My "
            "available spells are Ray of Frost, Fire Bolt, Slow, Hypnotic "
            "Pattern, and Misty Step. Concentration is free. What should I do?"
        ),
        max_tokens=180,
        checks=[
            recommends_area_control,
            does_not_invent_ray_of_frost_aoe,
            understands_hypnotic_pattern_area,
            does_not_spend_misty_step_as_action,
        ],
    )
]
