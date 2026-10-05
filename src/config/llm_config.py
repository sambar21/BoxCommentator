"""
LLM Configuration - Switch providers easily
"""

import os

class LLMConfig:
    """
    Central config for LLM provider selection.
    Change PROVIDER to switch between Ollama, Claude, OpenAI, or Groq.
    """
    
    # Options: "ollama", "claude", "openai", "groq", or any backend name from config/backends.yaml
    # (e.g. "vllm-3b"). COMMENTARY_BACKEND overrides it, so containers can pick a backend without code changes.
    PROVIDER = os.getenv("COMMENTARY_BACKEND", "groq")

    # Ollama config
    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_MODEL = "qwen2.5:7b"  # Options: llama3.2:3b, mistral:7b, qwen2.5:7b
    OLLAMA_TIMEOUT = 5.0  # seconds

    # Claude config
    CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
    CLAUDE_MAX_TOKENS = 150
    CLAUDE_TEMPERATURE = 0.8

    # OpenAI config
    OPENAI_MODEL = "gpt-4o-mini"
    OPENAI_MAX_TOKENS = 150
    OPENAI_TEMPERATURE = 0.8

    # Groq config
    GROQ_MODEL = "llama-3.3-70b-versatile"
    GROQ_MAX_TOKENS = 150
    GROQ_TEMPERATURE = 0.8

    # Performance settings
    TRACK_A_TIMEOUT = 0.3  # 300ms max for Track A
    TRACK_B_TIMEOUT = 0.2  # 200ms max for Track B
    
    # Enable streaming for Track A (only works with API providers)
    ENABLE_STREAMING = True  # Set False for Ollama
    
    # Fallback behavior if LLM fails
    USE_TEMPLATE_FALLBACK = True
    
    # Cache common responses (reduces API calls)
    ENABLE_RESPONSE_CACHE = True
    CACHE_SIZE = 100
    
    @classmethod
    def get_provider_name(cls) -> str:
        """Get current provider name"""
        return cls.PROVIDER
    
    @classmethod
    def is_local(cls) -> bool:
        """Is current provider running locally?"""
        return cls.PROVIDER == "ollama"
    
    @classmethod
    def requires_api_key(cls) -> bool:
        """Does current provider need an API key?"""
        return cls.PROVIDER in ["claude", "openai", "groq"]