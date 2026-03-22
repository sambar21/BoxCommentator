"""
Base LLM Interface - All providers must implement this.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Generator

class BaseLLMClient(ABC):
    """
    Abstract base class for LLM clients.
    Forces all providers to implement the same interface.
    """
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate commentary synchronously.
        
        Args:
            prompt: The full prompt to send to LLM
            **kwargs: Provider-specific options
            
        Returns:
            Generated commentary text
        """
        pass
    
    @abstractmethod
    def generate_streaming(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """
        Generate commentary with streaming (for Track A interrupts).
        
        Args:
            prompt: The full prompt to send to LLM
            **kwargs: Provider-specific options
            
        Yields:
            Chunks of generated text as they arrive
        """
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """
        Check if LLM provider is available.
        
        Returns:
            True if healthy, False otherwise
        """
        pass
    
    def get_provider_name(self) -> str:
        """Get name of this provider"""
        return self.__class__.__name__
