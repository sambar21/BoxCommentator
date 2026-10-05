"""
Loads config/backends.yaml: one entry per OpenAI-compatible backend.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = REPO_ROOT / "config" / "backends.yaml"
_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def load_env(path: Optional[Path] = None) -> bool:
    """Load the repo's .env (keys live there, never in tracked files). Existing env vars win."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    return load_dotenv(path or REPO_ROOT / ".env", override=False)


load_env()


def _expand(value):
    """Expand ${VAR} and ${VAR:-default} from the environment."""
    if not isinstance(value, str):
        return value
    return _VAR.sub(lambda m: os.environ.get(m.group(1)) or (m.group(2) or ""), value)


@dataclass(frozen=True)
class Backend:
    name: str
    base_url: str
    model: str
    api_key: str = ""
    price_in: float = 0.0           # USD / 1M input tokens
    price_out: float = 0.0          # USD / 1M output tokens
    gpu_hour_usd: float = 0.0       # self-hosted: GPU-hour-equivalent price
    min_interval_s: float = 0.0     # client-side pacing for rate-limited tiers
    # Passed through as the request's extra_body (e.g. chat_template_kwargs to turn thinking off)
    extra_body: dict = field(default_factory=dict, compare=False)

    @property
    def self_hosted(self) -> bool:
        return self.gpu_hour_usd > 0

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.model)


def load_config(path: Optional[Path] = None) -> dict:
    with open(path or DEFAULT_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_backends(path: Optional[Path] = None) -> Dict[str, Backend]:
    raw = load_config(path).get("backends", {})
    out = {}
    for name, cfg in raw.items():
        out[name] = Backend(
            name=name,
            base_url=_expand(cfg.get("base_url", "")),
            model=_expand(cfg.get("model", "")),
            api_key=_expand(cfg.get("api_key", "")),
            price_in=float(cfg.get("price_in", 0.0)),
            price_out=float(cfg.get("price_out", 0.0)),
            gpu_hour_usd=float(cfg.get("gpu_hour_usd", 0.0)),
            min_interval_s=float(cfg.get("min_interval_s", 0.0)),
            extra_body=dict(cfg.get("extra_body") or {}),
        )
    return out


def get_backend(name: str, path: Optional[Path] = None) -> Optional[Backend]:
    return load_backends(path).get(name)


def load_routing(path: Optional[Path] = None) -> dict:
    return load_config(path).get("routing", {}) or {}
