from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass(frozen=True)
class BackendTextResponse:
    text: str
    backend_name: str
    model_name: str


class TextGenerationBackend(Protocol):
    backend_name: str

    def describe(self) -> str:
        ...

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 650,
    ) -> BackendTextResponse:
        ...


def _parse_retry_after_seconds(err_text: str) -> Optional[float]:
    match = re.search(r"try again in\s*([0-9]+(?:\.[0-9]+)?)s", err_text, re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except Exception:
        return None


class GroqTextGenerationBackend:
    backend_name = "groq"

    def __init__(self, api_key: str, default_model: str) -> None:
        self.api_key = api_key.strip()
        self.default_model = default_model.strip() or "llama-3.1-8b-instant"

    @classmethod
    def from_env(cls) -> "GroqTextGenerationBackend":
        return cls(
            api_key=os.getenv("GROQ_API_KEY", ""),
            default_model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        )

    def describe(self) -> str:
        key_state = "set" if self.api_key else "NOT SET"
        return f"backend={self.backend_name} | model={self.default_model} | GROQ_API_KEY={key_state}"

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 650,
    ) -> BackendTextResponse:
        if not self.api_key:
            raise RuntimeError("Missing GROQ_API_KEY environment variable.")

        from groq import Groq

        resolved_model = (model or self.default_model).strip() or self.default_model
        client = Groq(api_key=self.api_key, max_retries=0)

        max_attempts = 2
        base_sleep = float(os.getenv("GROQ_RETRY_SLEEP", "1.0"))
        max_rate_limit_hits = int(os.getenv("GROQ_MAX_429", "12"))
        debug = os.getenv("GROQ_DEBUG", "0").strip() == "1"

        non429_attempt = 0
        rate_limit_hits = 0
        last_err: Optional[Exception] = None

        while non429_attempt < max_attempts:
            try:
                resp = client.chat.completions.create(
                    model=resolved_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                raw = resp.choices[0].message.content or ""
                return BackendTextResponse(
                    text=raw,
                    backend_name=self.backend_name,
                    model_name=resolved_model,
                )
            except Exception as exc:
                last_err = exc
                err_text = str(exc)

                if "429" in err_text or "rate limit" in err_text.lower():
                    rate_limit_hits += 1
                    if rate_limit_hits > max_rate_limit_hits:
                        raise RuntimeError(f"Groq call failed: {exc}") from exc

                    wait_s = _parse_retry_after_seconds(err_text)
                    if wait_s is None:
                        wait_s = base_sleep * 2.0
                    wait_s = float(wait_s) + 0.35

                    if debug:
                        print(
                            f"[GROQ_DEBUG] 429 hit #{rate_limit_hits}; sleeping {wait_s:.2f}s then retry"
                        )
                    time.sleep(wait_s)
                    continue

                non429_attempt += 1
                if debug:
                    print(f"[GROQ_DEBUG] non-429 attempt {non429_attempt}/{max_attempts} failed: {exc}")
                if non429_attempt < max_attempts:
                    time.sleep(min(base_sleep * (2 ** (non429_attempt - 1)), 6.0))
                    continue

                raise RuntimeError(f"Groq call failed after {max_attempts} attempts: {exc}") from exc

        raise RuntimeError(f"Groq call failed: {last_err}")


def load_text_generation_backend(name: Optional[str] = None) -> TextGenerationBackend:
    backend_name = (name or os.getenv("M2_LLM_BACKEND", "groq")).strip().lower()
    if backend_name == "groq":
        return GroqTextGenerationBackend.from_env()
    raise RuntimeError(f"Unsupported M2 LLM backend '{backend_name}'.")
