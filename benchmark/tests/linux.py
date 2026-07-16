from benchmark.models import BenchmarkTest, CheckResult


def contains_any(response: str, phrases: tuple[str, ...]) -> bool:
    text = response.lower()
    return any(phrase.lower() in text for phrase in phrases)


def identifies_render_group_issue(response: str) -> CheckResult:
    render = "render" in response.lower()
    permission = contains_any(
        response,
        (
            "permission",
            "not in the render group",
            "group membership",
            "access to /dev/accel",
        ),
    )

    return CheckResult(
        name="Identifies missing render-group access",
        passed=render and permission,
        detail=(
            "/dev/accel/accel0 is owned by root:render with mode 660, while "
            "the user is not listed in render."
        ),
    )


def proposes_safe_group_fix(response: str) -> CheckResult:
    text = response.lower().replace("  ", " ")

    good = (
        "usermod" in text
        and "-ag" in text
        and "render" in text
        and "gantz16" in text
    )

    destructive = any(
        phrase in text
        for phrase in (
            "chmod 777",
            "chmod 666",
            "chown gantz16 /dev/accel",
            "run everything as root",
        )
    )

    return CheckResult(
        name="Uses a persistent group-based fix",
        passed=good and not destructive,
        detail=(
            "The safe fix is to add gantz16 to render and begin a fresh login "
            "session, not loosen device permissions globally."
        ),
    )


def mentions_fresh_login(response: str) -> CheckResult:
    good = contains_any(
        response,
        (
            "log out and back in",
            "new login session",
            "fresh ssh session",
            "reconnect",
            "newgrp render",
        ),
    )

    return CheckResult(
        name="Explains that new group membership needs a fresh session",
        passed=good,
        detail=(
            "Supplementary group changes normally require reconnecting or "
            "starting a new group/session."
        ),
    )


TESTS = [
    BenchmarkTest(
        test_id="linux-001",
        name="NPU Device Permission Diagnosis",
        category="Linux and Homelab",
        system_prompt=(
            "Diagnose the Linux permission problem from the supplied output. "
            "Recommend the smallest persistent fix. Do not suggest chmod 777, "
            "chmod 666, or running the service permanently as root."
        ),
        user_prompt=(
            "Command output:\n\n"
            "$ ls -l /dev/accel/accel0\n"
            "crw-rw---- 1 root render 261, 0 /dev/accel/accel0\n\n"
            "$ id\n"
            "uid=1000(gantz16) gid=1000(gantz16) "
            "groups=1000(gantz16),27(sudo),983(ollama)\n\n"
            "$ flm validate\n"
            "Error: Open /dev/accel/accel0 failed: Permission denied\n\n"
            "Explain the cause and give the exact repair steps."
        ),
        max_tokens=220,
        checks=[
            identifies_render_group_issue,
            proposes_safe_group_fix,
            mentions_fresh_login,
        ],
    )
]
