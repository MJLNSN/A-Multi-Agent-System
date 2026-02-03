"""
Model Registry for the Multi-Agent Chat Threading System.
Defines available LLM models and their configurations.
Uses OpenRouter's full model identifiers for consistency.
"""

from typing import Dict, List, Optional


# Model registry with OpenRouter full identifiers as keys
MODEL_REGISTRY: Dict[str, Dict] = {
    "openai/gpt-4-turbo": {
        "provider": "openai",
        "display_name": "GPT-4 Turbo",
        "context_window": 128000,
        "supports_streaming": True,
        "cost_per_1k_tokens": {"input": 0.01, "output": 0.03}
    },
    "anthropic/claude-3.5-sonnet": {
        "provider": "anthropic",
        "display_name": "Claude 3.5 Sonnet",
        "context_window": 200000,
        "supports_streaming": True,
        "cost_per_1k_tokens": {"input": 0.003, "output": 0.015}
    },
    "openai/gpt-3.5-turbo": {
        "provider": "openai",
        "display_name": "GPT-3.5 Turbo",
        "context_window": 16385,
        "supports_streaming": True,
        "cost_per_1k_tokens": {"input": 0.0005, "output": 0.0015}
    },
}


def validate_model(model_identifier: str) -> bool:
    """
    Validate if a model identifier exists in the registry.
    
    Args:
        model_identifier: OpenRouter full model identifier (e.g., "openai/gpt-4-turbo")
    
    Returns:
        True if model is valid, False otherwise
    """
    return model_identifier in MODEL_REGISTRY


def list_available_models() -> List[str]:
    """
    List all available model identifiers.
    
    Returns:
        List of model identifier strings
    """
    return list(MODEL_REGISTRY.keys())


def get_model_config(model_identifier: str) -> Optional[Dict]:
    """
    Get configuration for a specific model.
    
    Args:
        model_identifier: OpenRouter full model identifier
    
    Returns:
        Model configuration dict or None if not found
    """
    return MODEL_REGISTRY.get(model_identifier)


def get_context_window(model_identifier: str) -> int:
    """
    Get the context window size for a model.
    
    Args:
        model_identifier: OpenRouter full model identifier
    
    Returns:
        Context window size in tokens, defaults to 8000 if model not found
    """
    config = get_model_config(model_identifier)
    if config:
        return config.get("context_window", 8000)
    return 8000

