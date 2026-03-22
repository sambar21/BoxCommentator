"""
Ollama Client - Free local LLM inference.
Requires Ollama installed: https://ollama.com
"""

import requests
import json
from typing import Generator, Dict, Any
from .base_client import BaseLLMClient
from src.config.llm_config import LLMConfig

class OllamaClient(BaseLLMClient):
    """
    Client for local Ollama LLM inference.
    100% free, runs on your machine.
    """
    
    def __init__(self):
        self.base_url = LLMConfig.OLLAMA_BASE_URL
        self.model = LLMConfig.OLLAMA_MODEL
        self.timeout = LLMConfig.OLLAMA_TIMEOUT
        
        # Check if Ollama is running
        if not self.health_check():
            print(f"  WARNING: Ollama not running at {self.base_url}")
            print(f"   Start it with: ollama serve")
            print(f"   Then pull model: ollama pull {self.model}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate commentary using Ollama.
        
        Args:
            prompt: Full prompt text
            **kwargs: Optional override for model, temperature, etc.
            
        Returns:
            Generated commentary
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": kwargs.get("model", self.model),
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": kwargs.get("temperature", 0.8),
                        "num_predict": kwargs.get("max_tokens", 150),
                    }
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                print(f"❌ Ollama error: {response.status_code}")
                return ""
                
        except requests.exceptions.Timeout:
            print(f"⏱️  Ollama timeout ({self.timeout}s)")
            return ""
        except requests.exceptions.ConnectionError:
            print(f"🔌 Can't connect to Ollama at {self.base_url}")
            return ""
        except Exception as e:
            print(f"❌ Ollama error: {e}")
            return ""
    
    def generate_streaming(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """
        Generate with streaming (for Track A interrupts).
        
        Yields:
            Text chunks as they're generated
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": kwargs.get("model", self.model),
                    "prompt": prompt,
                    "stream": True,  # Enable streaming
                    "options": {
                        "temperature": kwargs.get("temperature", 0.8),
                        "num_predict": kwargs.get("max_tokens", 150),
                    }
                },
                timeout=self.timeout,
                stream=True  # Important for streaming
            )
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            if "response" in chunk:
                                yield chunk["response"]
                        except json.JSONDecodeError:
                            continue
            
        except Exception as e:
            print(f"❌ Ollama streaming error: {e}")
            yield ""
    
    def health_check(self) -> bool:
        """
        Check if Ollama is running and model is available.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Check if Ollama is running
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=2.0
            )
            
            if response.status_code != 200:
                return False
            
            # Check if our model is available
            data = response.json()
            models = data.get("models", [])
            model_names = [m.get("name", "") for m in models]
            
            # Model names in Ollama include tags (e.g., "qwen2.5:7b")
            return any(self.model in name for name in model_names)
            
        except Exception:
            return False
    
    def get_available_models(self) -> list:
        """
        Get list of models currently downloaded in Ollama.
        
        Returns:
            List of model names
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=2.0
            )
            
            if response.status_code == 200:
                data = response.json()
                return [m.get("name", "") for m in data.get("models", [])]
            
        except Exception:
            pass
        
        return []


# Convenience function for testing
if __name__ == "__main__":
    """Test Ollama client"""
    client = OllamaClient()
    
    print("🔍 Health check...")
    if client.health_check():
        print("✅ Ollama is healthy!")
        print(f"📦 Available models: {client.get_available_models()}")
        
        print("\n🎤 Generating test commentary...")
        prompt = "Generate a short boxing commentary about a fighter landing a jab. 1-2 sentences only."
        result = client.generate(prompt)
        print(f"💬 Result: {result}")
        
        print("\n🌊 Testing streaming...")
        for chunk in client.generate_streaming(prompt):
            print(chunk, end="", flush=True)
        print()
    else:
        print("❌ Ollama not healthy")
        print("\nTo fix:")
        print("1. Install Ollama: https://ollama.com")
        print("2. Start server: ollama serve")
        print(f"3. Pull model: ollama pull {client.model}")
