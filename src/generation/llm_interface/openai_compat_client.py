"""
OpenAI-compatible client: one class for vLLM, Nebius Token Factory, Groq, or
anything else that speaks /v1/chat/completions.

Unlike the vendor clients, it does not hide failures: errors are counted and
the last one is kept, so a benchmark can tell "backend failed" from "model said
nothing". Retries are off by default (retries would hide tail latency).
"""

import threading
import time
from typing import Generator, Optional

from openai import OpenAI

from src.config.backends import Backend
from .base_client import BaseLLMClient


class OpenAICompatClient(BaseLLMClient):
    def __init__(
        self,
        backend: Backend,
        max_tokens: int = 150,
        temperature: float = 0.8,
        timeout: float = 30.0,
        max_retries: int = 0,
        include_usage: bool = True,
    ):
        self.backend = backend
        self.model = backend.model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.include_usage = include_usage
        self.client = OpenAI(
            base_url=backend.base_url,
            api_key=backend.api_key or "EMPTY",
            timeout=timeout,
            max_retries=max_retries,
        )

        self.error_count = 0
        self.last_error: Optional[str] = None
        self.request_count = 0

        # Usage of the most recent call *on this thread* (clients are shared across
        # concurrent fights, so a plain attribute would race).
        self._tl = threading.local()
        self._pace_lock = threading.Lock()
        self._last_request_at = 0.0

    # ── usage of the last call on the calling thread ─────────────────────────
    @property
    def last_completion_tokens(self) -> Optional[int]:
        return getattr(self._tl, "completion_tokens", None)

    @property
    def last_prompt_tokens(self) -> Optional[int]:
        return getattr(self._tl, "prompt_tokens", None)

    @property
    def last_cached_tokens(self) -> Optional[int]:
        return getattr(self._tl, "cached_tokens", None)

    def _set_usage(self, usage):
        self._tl.completion_tokens = getattr(usage, "completion_tokens", None)
        self._tl.prompt_tokens = getattr(usage, "prompt_tokens", None)
        details = getattr(usage, "prompt_tokens_details", None)
        self._tl.cached_tokens = getattr(details, "cached_tokens", None) if details else None

    def _clear_usage(self):
        self._tl.completion_tokens = None
        self._tl.prompt_tokens = None
        self._tl.cached_tokens = None

    def _pace(self):
        """Client-side spacing for rate-limited tiers (min_interval_s)."""
        if self.backend.min_interval_s <= 0:
            return
        with self._pace_lock:
            wait = self._last_request_at + self.backend.min_interval_s - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last_request_at = time.monotonic()

    def _fail(self, e: Exception):
        self.error_count += 1
        self.last_error = f"{type(e).__name__}: {e}"

    def _params(self, kwargs) -> dict:
        params = dict(
            model=kwargs.get("model", self.model),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
        )
        if self.backend.extra_body:
            params["extra_body"] = self.backend.extra_body
        return params

    # ── BaseLLMClient ────────────────────────────────────────────────────────
    def generate(self, prompt: str, **kwargs) -> str:
        self._pace()
        self.request_count += 1
        self._clear_usage()
        try:
            resp = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                **self._params(kwargs),
            )
            if resp.usage:
                self._set_usage(resp.usage)
            if resp.choices:
                return (resp.choices[0].message.content or "").strip()
            return ""
        except Exception as e:
            self._fail(e)
            return ""

    def generate_streaming(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        self._pace()
        self.request_count += 1
        self._clear_usage()
        extra = {"stream_options": {"include_usage": True}} if self.include_usage else {}
        try:
            stream = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                **extra,
                **self._params(kwargs),
            )
            for chunk in stream:
                if getattr(chunk, "usage", None):
                    self._set_usage(chunk.usage)
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        yield delta.content
        except Exception as e:
            self._fail(e)

    def health_check(self) -> bool:
        try:
            self.client.chat.completions.create(
                model=self.model,
                max_tokens=5,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return True
        except Exception as e:
            self._fail(e)
            return False

    def get_provider_name(self) -> str:
        return self.backend.name
