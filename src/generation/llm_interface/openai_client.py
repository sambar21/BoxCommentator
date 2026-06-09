"""
OpenAI Client - Fast and cheap AI commentary via OpenAI API.
Requires API key: https://platform.openai.com
"""

import os
from typing import Generator
from .base_client import BaseLLMClient
from src.config.llm_config import LLMConfig
from dotenv import load_dotenv
load_dotenv()

try:
    from openai import OpenAI, OpenAIError
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("WARNING: openai package not installed. Run: pip install openai")


class OpenAIClient(BaseLLMClient):
    """
    Client for OpenAI API (GPT-4o-mini).
    Fast and cheap, requires API key.
    """
    
    def __init__(self, api_key: str = None):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package required. Install with: pip install openai")
        
        # Get API key from param or environment
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            print("WARNING: No OPENAI_API_KEY found.")
            print("   Set it in .env file or pass to constructor")
            print("   Get key at: https://platform.openai.com")
        
        # Initialize client
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
        # Config from LLMConfig
        self.model = LLMConfig.OPENAI_MODEL
        self.max_tokens = LLMConfig.OPENAI_MAX_TOKENS
        self.temperature = LLMConfig.OPENAI_TEMPERATURE
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate commentary using OpenAI.
        
        Args:
            prompt: Full prompt text
            **kwargs: Optional overrides (temperature, max_tokens, etc.)
            
        Returns:
            Generated commentary
        """
        if not self.client:
            return ""
        
        try:
            response = self.client.chat.completions.create(
                model=kwargs.get("model", self.model),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract text from response
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content.strip()
            
            return ""
            
        except OpenAIError as e:
            print(f"❌ OpenAI API error: {e}")
            return ""
        except Exception as e:
            print(f"❌ OpenAI unexpected error: {e}")
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
            stream = self.client.chat.completions.create(
                model=kwargs.get("model", self.model),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                messages=[
                    {"role": "user", "content": prompt}
                ],
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        yield delta.content
                    
        except OpenAIError as e:
            print(f"❌ OpenAI streaming error: {e}")
            yield ""
        except Exception as e:
            print(f"❌ OpenAI unexpected error: {e}")
            yield ""
    
    def health_check(self) -> bool:
        """
        Check if OpenAI API is accessible.
        
        Returns:
            True if API key valid and service reachable
        """
        if not self.client:
            return False
        
        try:
            # Simple test request
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=5,
                messages=[
                    {"role": "user", "content": "Hi"}
                ]
            )
            return True
            
        except Exception:
            return False


# Convenience function for testing
if __name__ == "__main__":
    """Test OpenAI client"""
    
    client = OpenAIClient()
    
    print("🔍 Health check...")
    if client.health_check():
        print("✅ OpenAI API is healthy!")
        
        print("\n🎤 Generating test commentary...")
        prompt = "Generate a short boxing commentary about a fighter landing a jab. 1-2 sentences only."
        result = client.generate(prompt)
        print(f"💬 Result: {result}")
        
        print("\n🌊 Testing streaming...")
        for chunk in client.generate_streaming(prompt):
            print(chunk, end="", flush=True)
        print()
    else:
        print("❌ OpenAI API not healthy")
        print("\nTo fix:")
        print("1. Get API key: https://platform.openai.com")
        print("2. Set OPENAI_API_KEY in .env file")
        print("3. Or pass api_key to OpenAIClient(api_key='...')")