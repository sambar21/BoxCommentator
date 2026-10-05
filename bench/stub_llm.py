"""
Deterministic stand-in LLM for tests and offline harness checks.
Echoes whether the prompt is a knockdown so recall plumbing can be verified
without a network call.
"""

from typing import Generator
from src.generation.llm_interface.base_client import BaseLLMClient


class StubLLM(BaseLLMClient):
    def __init__(self, drop_knockdowns: bool = False, delay_s: float = 0.0):
        self.drop_knockdowns = drop_knockdowns
        self.delay_s = delay_s
        self.calls = 0

    def _reply(self, prompt: str) -> str:
        self.calls += 1
        if self.delay_s:
            import time
            time.sleep(self.delay_s)
        if "KNOCKDOWN" in prompt:
            return "" if self.drop_knockdowns else "KNOCKDOWN! He is down on the canvas!"
        return "Steady pressure from the lead fighter."

    def generate(self, prompt: str, **kwargs) -> str:
        return self._reply(prompt)

    def generate_streaming(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        for word in self._reply(prompt).split(" "):
            if word:
                yield word + " "

    def health_check(self) -> bool:
        return True

    def get_provider_name(self) -> str:
        return "stub"
