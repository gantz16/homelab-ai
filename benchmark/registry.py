from __future__ import annotations

import importlib
import pkgutil

from benchmark.models import BenchmarkTest


def discover_tests() -> list[BenchmarkTest]:
    import benchmark.tests as tests_package

    discovered: list[BenchmarkTest] = []

    for module_info in pkgutil.iter_modules(tests_package.__path__):
        if module_info.name.startswith("_"):
            continue

        module = importlib.import_module(
            f"{tests_package.__name__}.{module_info.name}"
        )

        module_tests = getattr(module, "TESTS", [])
        discovered.extend(module_tests)

    return sorted(discovered, key=lambda test: test.test_id)
