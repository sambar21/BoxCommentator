"""
LLM Interface Package - Swappable LLM clients
"""

from .base_client import BaseLLMClient
from .ollama_client import OllamaClient
from .llm_factory import LLMFactory

# Try to import API clients (they may not be installed)
try:
    from .claude_client import ClaudeClient
except ImportError:
    ClaudeClient = None

try:
    from .openai_client import OpenAIClient
except ImportError:
    OpenAIClient = None

try:
    from .groq_client import GroqClient
except ImportError:
    GroqClient = None

__all__ = [
    'BaseLLMClient',
    'OllamaClient',
    'ClaudeClient',
    'OpenAIClient',
    'GroqClient',
    'LLMFactory'
]