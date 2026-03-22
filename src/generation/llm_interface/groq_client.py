"""
Groq Client - FREE and BLAZING FAST AI inference.
Uses Groq's free tier with Llama models.
"""

import os
from typing import Generator
from .base_client import BaseLLMClient
from src.config.llm_config import LLMConfig
from dotenv import load_dotenv
load_dotenv()

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("WARNING: groq package not installed. Run: pip install groq")


class GroqClient(BaseLLMClient):
    """
    Client for Groq API - FREE and FAST!
    Uses Llama 3.3 70B model.
    """
    
    def __init__(self, api_key: str = None):
        if not GROQ_AVAILABLE:
            raise ImportError("groq package required. Install with: pip install groq")
        
        # Get API key from param or environment
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        
        if not self.api_key:
            print("⚠️  WARNING: No GROQ_API_KEY found!")
            print("   Set it in .env file or pass to constructor")
            print("   Get key at: https://console.groq.com")
        
        # Initialize client
        self.client = Groq(api_key=self.api_key) if self.api_key else None

        # Config from LLMConfig
        self.model = LLMConfig.GROQ_MODEL
        self.max_tokens = LLMConfig.GROQ_MAX_TOKENS
        self.temperature = LLMConfig.GROQ_TEMPERATURE
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate commentary using Groq.
        
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
            
        except Exception as e:
            print(f"❌ Groq API error: {e}")
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
                    
        except Exception as e:
            print(f"❌ Groq streaming error: {e}")
            yield ""
    
    def health_check(self) -> bool:
        """
        Check if Groq API is accessible.
        
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
    """Test Groq client"""
    
    client = GroqClient()
    
    print("🔍 Health check...")
    if client.health_check():
        print("✅ Groq API is healthy!")
        
        print("\n🎤 Generating test commentary...")
        prompt = "Generate a short boxing commentary about a fighter landing a jab. 1-2 sentences only."
        result = client.generate(prompt)
        print(f"💬 Result: {result}")
        
        print("\n🌊 Testing streaming...")
        for chunk in client.generate_streaming(prompt):
            print(chunk, end="", flush=True)
        print()
    else:
        print("❌ Groq API not healthy")
        print("\nTo fix:")
        print("1. Get API key: https://console.groq.com")
        print("2. Set GROQ_API_KEY in .env file")
        print("3. Or pass api_key to GroqClient(api_key='...')")