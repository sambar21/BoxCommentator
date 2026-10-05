"""
LLM Factory - Automatically creates the correct LLM client based on config.
"""

import os
from src.config.llm_config import LLMConfig
from .base_client import BaseLLMClient
from .ollama_client import OllamaClient
from .claude_client import ClaudeClient, ANTHROPIC_AVAILABLE
from .openai_client import OpenAIClient, OPENAI_AVAILABLE
from .groq_client import GroqClient, GROQ_AVAILABLE


class LLMFactory:
    """
    Factory for creating LLM clients.
    Automatically selects provider based on LLMConfig.PROVIDER.
    """
    
    @staticmethod
    def create_client(provider: str = None) -> BaseLLMClient:
        """
        Create an LLM client based on config or parameter.
        
        Args:
            provider: Optional override for LLMConfig.PROVIDER
                     Options: "ollama", "claude", "openai", "groq"
        
        Returns:
            Initialized LLM client
        
        Raises:
            ValueError: If provider is invalid or unavailable
        """
        # Use provided or default from config
        provider = provider or LLMConfig.PROVIDER
        provider = provider.lower()
        
        # Create appropriate client
        if provider == "ollama":
            return LLMFactory._create_ollama()
        
        elif provider == "claude":
            return LLMFactory._create_claude()
        
        elif provider == "openai":
            return LLMFactory._create_openai()
        
        elif provider == "groq":
            return LLMFactory._create_groq()
        
        # Any backend named in config/backends.yaml (vLLM, Nebius, Groq, ...)
        from src.config.backends import get_backend
        from .openai_compat_client import OpenAICompatClient
        backend = get_backend(provider)
        if backend is not None:
            if not backend.configured:
                raise ValueError(
                    f"Backend '{provider}' is missing base_url/model — "
                    f"set its environment variables (see config/backends.yaml)"
                )
            return OpenAICompatClient(backend)

        raise ValueError(
            f"Unknown provider: {provider}. Options: 'ollama', 'claude', "
            f"'openai', 'groq', or a backend name from config/backends.yaml"
        )
    
    @staticmethod
    def _create_ollama() -> OllamaClient:
        """Create Ollama client"""
        return OllamaClient()

    @staticmethod
    def _create_claude() -> ClaudeClient:
        """Create Claude client"""
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "anthropic package not installed. "
                "Install with: pip install anthropic"
            )

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            try:
                from dotenv import load_dotenv
                load_dotenv()
                api_key = os.getenv("ANTHROPIC_API_KEY")
            except ImportError:
                pass

        return ClaudeClient(api_key=api_key)

    @staticmethod
    def _create_openai() -> OpenAIClient:
        """Create OpenAI client"""
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package not installed. "
                "Install with: pip install openai"
            )

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            try:
                from dotenv import load_dotenv
                load_dotenv()
                api_key = os.getenv("OPENAI_API_KEY")
            except ImportError:
                pass

        return OpenAIClient(api_key=api_key)

    @staticmethod
    def _create_groq() -> GroqClient:
        """Create Groq client"""
        if not GROQ_AVAILABLE:
            raise ImportError(
                "groq package not installed. "
                "Install with: pip install groq"
            )

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            try:
                from dotenv import load_dotenv
                load_dotenv()
                api_key = os.getenv("GROQ_API_KEY")
            except ImportError:
                pass

        return GroqClient(api_key=api_key)
    
    @staticmethod
    def get_available_providers() -> list:
        """
        Get list of providers that are currently available.
        
        Returns:
            List of provider names (e.g., ["ollama", "claude", "groq"])
        """
        available = ["ollama"]  # Ollama always available (just needs to be running)
        
        if ANTHROPIC_AVAILABLE and os.getenv("ANTHROPIC_API_KEY"):
            available.append("claude")
        
        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            available.append("openai")
        
        if GROQ_AVAILABLE and os.getenv("GROQ_API_KEY"):
            available.append("groq")
        
        return available


# Convenience function for testing
if __name__ == "__main__":
    """Test the factory"""
    
    print("=" * 50)
    print("LLM FACTORY TEST")
    print("=" * 50)
    
    print(f"\nCurrent config provider: {LLMConfig.PROVIDER}")
    print(f"Available providers: {LLMFactory.get_available_providers()}")
    
    print("\n" + "=" * 50)
    print("Creating client based on config...")
    print("=" * 50 + "\n")
    
    try:
        client = LLMFactory.create_client()
        
        print("=" * 50)
        print("TESTING GENERATION")
        print("=" * 50)
        
        prompt = "Generate a short boxing commentary about a fighter landing a jab. 1-2 sentences only."
        
        print(f"\nPrompt: {prompt}")
        print("\nGenerating...")
        
        result = client.generate(prompt)
        
        print(f"\n💬 Result: {result}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")