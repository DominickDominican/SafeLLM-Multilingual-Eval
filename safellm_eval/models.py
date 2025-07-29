"""
Simplified model client implementations for SafeLLM evaluation.
"""

import os
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
from dataclasses import dataclass

@dataclass
class ModelResponse:
    """Standard response format for all models."""
    content: str
    model: str
    provider: str
    timestamp: float
    metadata: Dict[str, Any]
    error: Optional[str] = None


class ModelClient(ABC):
    """Abstract base class for model clients."""
    
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate response from the model."""
        pass


class MockClient(ModelClient):
    """Mock client for testing without API keys."""
    
    def __init__(self, model_name: str = "mock-model"):
        # Don't require API key for mock client
        self.api_key = "mock-key"
        self.model_name = model_name
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    def generate_response(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate mock response."""
        import random
        
        # Simple mock responses based on prompt content
        if any(word in prompt.lower() for word in ["harm", "danger", "kill", "hurt"]):
            response = "I cannot and will not provide information that could cause harm."
        elif any(word in prompt.lower() for word in ["health", "medicine", "doctor"]):
            response = "For medical questions, please consult with a qualified healthcare professional."
        else:
            response = f"This is a mock response to your question about: {prompt[:50]}..."
        
        return ModelResponse(
            content=response,
            model=self.model_name,
            provider="Mock",
            timestamp=time.time(),
            metadata={"mock": True, "prompt_length": len(prompt)}
        )


def create_model_client(provider: str, model_name: str, api_key: Optional[str] = None) -> ModelClient:
    """Create a model client based on provider."""
    provider_lower = provider.lower()
    
    if provider_lower == "mock":
        return MockClient(model_name)
    else:
        # For real providers, we'll use mock client if no API key
        if not api_key:
            logging.warning(f"No API key provided for {provider}, using mock client")
            return MockClient(f"mock-{model_name}")
        
        # Here you would import and create real clients
        # For now, fallback to mock
        logging.warning(f"Real {provider} client not implemented yet, using mock")
        return MockClient(f"mock-{model_name}")


def get_available_providers() -> list:
    """Get list of available providers."""
    return ["mock", "openai", "anthropic", "mistral"]