"""
Abstract Base Class for Hosted LLM Providers
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, AsyncIterator


class BaseLLMClient(ABC):
    @abstractmethod
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        """Generates a complete response asynchronously."""
        pass

    @abstractmethod
    async def stream_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        """Streams token chunks asynchronously."""
        pass
