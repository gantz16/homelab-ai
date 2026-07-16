from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class InferenceResult:
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    ttft: float
    prefill_speed_tps: float
    decode_speed_tps: float
    decode_seconds: float
    request_seconds: float
    raw_response: dict[str, Any]


class FLMClient:
    def __init__(self, endpoint: str = "http://127.0.0.1:8000") -> None:
        self.endpoint = endpoint.rstrip("/")

    def chat(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
        timeout: int = 180,
    ) -> InferenceResult:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "stream": False,
            "max_tokens": max_tokens,
        }

        request = urllib.request.Request(
            f"{self.endpoint}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        started = time.perf_counter()

        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"FLM returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach FLM at {self.endpoint}: {exc.reason}"
            ) from exc

        elapsed = time.perf_counter() - started
        data = json.loads(body)

        try:
            content = data["choices"][0]["message"]["content"]
            usage = data["usage"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected FLM response: {data}") from exc

        return InferenceResult(
            content=content,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            total_tokens=int(usage.get("total_tokens", 0)),
            ttft=float(usage.get("prefill_duration_ttft", 0.0)),
            prefill_speed_tps=float(usage.get("prefill_speed_tps", 0.0)),
            decode_speed_tps=float(usage.get("decoding_speed_tps", 0.0)),
            decode_seconds=float(usage.get("decoding_duration", 0.0)),
            request_seconds=elapsed,
            raw_response=data,
        )
