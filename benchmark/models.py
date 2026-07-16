from dataclasses import dataclass, field
from typing import Callable


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class BenchmarkTest:
    test_id: str
    name: str
    category: str
    system_prompt: str
    user_prompt: str
    max_tokens: int = 200
    temperature: float = 0.1
    weight: float = 1.0
    checks: list[Callable[[str], CheckResult]] = field(default_factory=list)
