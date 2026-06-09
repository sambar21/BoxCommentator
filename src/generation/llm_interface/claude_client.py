"""
Claude Client - High-quality AI commentary via Anthropic API.
Requires API key: https://console.anthropic.com
"""

import os
from typing import Generator
from .base_client import BaseLLMClient
from src.config.llm_config import LLMConfig

try:
    from anthropic import Anthropic, AnthropicError
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("WARNING: anthropic package not installed. Run: pip install anthropic")


class ClaudeClient(BaseLLMClient):
    """
    Client for Claude API (Anthropic).
    Best quality commentary, requires API key.
    """
    
    def __init__(self, api_key: str = None):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package required. Install with: pip install anthropic")
        
        # Get API key from param or environment
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            print("WARNING: No ANTHROPIC_API_KEY found.")
            print("   Set it in .env file or pass to constructor")
            print("   Get key at: https://console.anthropic.com")
        
        # Initialize client
        self.client = Anthropic(api_key=self.api_key) if self.api_key else None
        
        # Config from LLMConfig
        self.model = LLMConfig.CLAUDE_MODEL
        self.max_tokens = LLMConfig.CLAUDE_MAX_TOKENS
        self.temperature = LLMConfig.CLAUDE_TEMPERATURE
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate commentary using Claude.
        
        Args:
            prompt: Full prompt text
            **kwargs: Optional overrides (temperature, max_tokens, etc.)
            
        Returns:
            Generated commentary
        """
        if not self.client:
            return ""
        
        try:
            response = self.client.messages.create(
                model=kwargs.get("model", self.model),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract text from response
            if response.content and len(response.content) > 0:
                return response.content[0].text.strip()
            
            return ""
            
        except AnthropicError as e:
            print(f"❌ Claude API error: {e}")
            return ""
        except Exception as e:
            print(f"❌ Claude unexpected error: {e}")
            return ""
    
    def generate_streaming(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """
        Generate with streaming (for Track A interrupts).
        
        Yields:
            Text chunks as they're generated
        """
        if not self.client:
            yield ""
            return
        
        try:
            with self.client.messages.stream(
                model=kwargs.get("model", self.model),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                messages=[
                    {"role": "user", "content": prompt}
                ]
            ) as stream:
                for text in stream.text_stream:
                    yield text
                    
        except AnthropicError as e:
            print(f"❌ Claude streaming error: {e}")
            yield ""
        except Exception as e:
            print(f"❌ Claude unexpected error: {e}")
            yield ""
    
    def health_check(self) -> bool:
        """
        Check if Claude API is accessible.
        
        Returns:
            True if API key valid and service reachable
        """
        if not self.client:
            return False
        
        try:
            # Simple test request
            response = self.client.messages.create(
                model=self.model,
                max_tokens=10,
                messages=[
                    {"role": "user", "content": "Hi"}
                ]
            )
            return True
            
        except Exception:
            return False


# Convenience function for testing
if __name__ == "__main__":
    """Test Claude client"""
    
    # Load API key from .env if available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    client = ClaudeClient()
    
    print("🔍 Health check...")
    if client.health_check():
        print("✅ Claude API is healthy!")
        
        print("\n🎤 Generating test commentary...")
        prompt = "Generate a short boxing commentary about a fighter landing a jab. 1-2 sentences only."
        result = client.generate(prompt)
        print(f"💬 Result: {result}")
        
        print("\n🌊 Testing streaming...")
        for chunk in client.generate_streaming(prompt):
            print(chunk, end="", flush=True)
        print()
    else:
        print("❌ Claude API not healthy")
        print("\nTo fix:")
        print("1. Get API key: https://console.anthropic.com")
        print("2. Set ANTHROPIC_API_KEY in .env file")
        print("3. Or pass api_key to ClaudeClient(api_key='...')")
