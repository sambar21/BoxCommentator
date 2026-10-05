"""
Stream timing helpers. Wall-clock (perf_counter) on purpose: these measure real
LLM latency and must not follow the virtual fight clock.
"""

import time
from dataclasses import dataclass, asdict
from typing import Iterable, Optional


@dataclass
class GenerationRecord:
    track: str                   # "A" or "B"
    backend: str
    event_type: str
    event_context: dict
    punch_index: int             # how many punches had been processed
    text: str
    ttft_ms: Optional[float]     # request start -> first non-empty chunk
    total_ms: float              # request start -> stream end
    n_chunks: int                # ~tokens (one chunk per token on OpenAI-style streams)
    n_tokens: Optional[int] = None   # exact count when the backend reports usage
    prompt_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None   # prompt tokens served from the prefix cache, if reported
    prompt_chars: int = 0
    prompt: str = ""                 # only kept when the orchestrator has keep_prompts=True
    interrupted: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def collect_timed(chunks: Iterable[str]):
    """
    Drain a chunk iterator, timing it.

    Returns (text, ttft_ms, total_ms, n_chunks). ttft_ms is None if nothing
    non-empty arrived (backend error or empty reply).
    """
    t0 = time.perf_counter()
    ttft_ms = None
    n_chunks = 0
    parts = []
    for chunk in chunks:
        if not chunk:
            continue
        if ttft_ms is None:
            ttft_ms = (time.perf_counter() - t0) * 1000
        n_chunks += 1
        parts.append(chunk)
    total_ms = (time.perf_counter() - t0) * 1000
    return "".join(parts).strip(), ttft_ms, total_ms, n_chunks
