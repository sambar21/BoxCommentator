"""
Routing: pick a backend per track, with fallback.

  Track B (knockdowns, urgent calls) -> the fastest backend
  Track A (analysis, quiet stretches) -> the smartest backend

Configured in config/backends.yaml under `routing:` (track_a, track_b, fallback).
A FallbackLLM tries the primary backend and, if it errors or returns nothing,
the next one in the chain.
"""

import threading
from typing import Callable, Dict, Generator, List, Optional, Tuple

from .llm_interface.base_client import BaseLLMClient


class FallbackLLM(BaseLLMClient):
    """Try each client in order until one produces output."""

    def __init__(self, clients: List[BaseLLMClient]):
        if not clients:
            raise ValueError("FallbackLLM needs at least one client")
        self.clients = clients
        self._tl = threading.local()
        self.fallbacks_used = 0

    def _served(self) -> BaseLLMClient:
        return getattr(self._tl, "served", None) or self.clients[0]

    def generate(self, prompt: str, **kwargs) -> str:
        for i, client in enumerate(self.clients):
            out = client.generate(prompt, **kwargs)
            if out:
                self._tl.served = client
                if i > 0:
                    self.fallbacks_used += 1
                return out
        self._tl.served = self.clients[-1]
        return ""

    def generate_streaming(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        for i, client in enumerate(self.clients):
            produced = False
            for chunk in client.generate_streaming(prompt, **kwargs):
                if chunk:
                    produced = True
                yield chunk
            if produced:
                self._tl.served = client
                if i > 0:
                    self.fallbacks_used += 1
                return
        self._tl.served = self.clients[-1]

    def health_check(self) -> bool:
        return any(c.health_check() for c in self.clients)

    def get_provider_name(self) -> str:
        return self._served().get_provider_name()

    # usage of whichever client served the last request on this thread
    @property
    def last_completion_tokens(self):
        return getattr(self._served(), "last_completion_tokens", None)

    @property
    def last_prompt_tokens(self):
        return getattr(self._served(), "last_prompt_tokens", None)

    @property
    def last_cached_tokens(self):
        return getattr(self._served(), "last_cached_tokens", None)


def _chain(primary: str, fallback: List[str], make: Callable[[str], BaseLLMClient]) -> BaseLLMClient:
    names = [primary] + [f for f in fallback if f != primary]
    clients = [make(n) for n in names]
    return clients[0] if len(clients) == 1 else FallbackLLM(clients)


def build_routed_clients(
    routing: Dict,
    make: Callable[[str], BaseLLMClient],
) -> Tuple[BaseLLMClient, BaseLLMClient]:
    """
    Returns (track_a_llm, track_b_llm) from a routing dict like
      {default: groq-70b, track_a: groq-70b, track_b: vllm-3b, fallback: [groq-70b]}
    `make(name)` builds a client for a backend name (so callers can wrap clients,
    e.g. with budget metering).
    """
    default = routing.get("default")
    track_a = routing.get("track_a") or default
    track_b = routing.get("track_b") or default
    if not track_a or not track_b:
        raise ValueError("routing needs `default` or both `track_a` and `track_b`")
    fallback = routing.get("fallback") or []
    if isinstance(fallback, str):
        fallback = [fallback]
    return _chain(track_a, fallback, make), _chain(track_b, fallback, make)
