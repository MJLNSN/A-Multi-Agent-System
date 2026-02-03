"""
Unit tests for Model Registry.
"""

import pytest

from src.models.registry import (
    MODEL_REGISTRY,
    validate_model,
    list_available_models,
    get_model_config,
    get_context_window
)


class TestModelRegistry:
    """Test cases for Model Registry."""
    
    def test_model_registry_not_empty(self):
        """Test that model registry has models."""
        assert len(MODEL_REGISTRY) >= 2
    
    def test_model_registry_has_required_models(self):
        """Test that registry has at least 2 LLMs as required."""
        models = list_available_models()
        
        # Should have at least GPT-4 and Claude
        has_openai = any("openai" in m for m in models)
        has_anthropic = any("anthropic" in m for m in models)
        
        assert has_openai, "Missing OpenAI model"
        assert has_anthropic, "Missing Anthropic model"
    
    def test_validate_model_valid(self):
        """Test validation of valid models."""
        assert validate_model("openai/gpt-4-turbo") is True
        assert validate_model("anthropic/claude-3.5-sonnet") is True
        assert validate_model("openai/gpt-3.5-turbo") is True
    
    def test_validate_model_invalid(self):
        """Test validation of invalid models."""
        assert validate_model("invalid/model") is False
        assert validate_model("") is False
        assert validate_model("gpt-4") is False  # Missing provider prefix
    
    def test_list_available_models(self):
        """Test listing available models."""
        models = list_available_models()
        
        assert isinstance(models, list)
        assert len(models) >= 2
        
        # All should be strings
        for model in models:
            assert isinstance(model, str)
            assert "/" in model  # Should have provider/model format
    
    def test_get_model_config_valid(self):
        """Test getting config for valid model."""
        config = get_model_config("openai/gpt-4-turbo")
        
        assert config is not None
        assert "provider" in config
        assert "context_window" in config
        assert "display_name" in config
        assert config["provider"] == "openai"
    
    def test_get_model_config_invalid(self):
        """Test getting config for invalid model."""
        config = get_model_config("invalid/model")
        assert config is None
    
    def test_get_context_window_valid(self):
        """Test getting context window for valid model."""
        window = get_context_window("openai/gpt-4-turbo")
        assert window == 128000
        
        window = get_context_window("anthropic/claude-3.5-sonnet")
        assert window == 200000
    
    def test_get_context_window_invalid(self):
        """Test getting context window for invalid model returns default."""
        window = get_context_window("invalid/model")
        assert window == 8000  # Default
    
    def test_model_config_has_required_fields(self):
        """Test that all model configs have required fields."""
        required_fields = ["provider", "display_name", "context_window", "supports_streaming"]
        
        for model_id, config in MODEL_REGISTRY.items():
            for field in required_fields:
                assert field in config, f"Model {model_id} missing field {field}"
    
    def test_model_identifiers_use_openrouter_format(self):
        """Test that model IDs follow OpenRouter format (provider/model)."""
        for model_id in MODEL_REGISTRY.keys():
            assert "/" in model_id, f"Model {model_id} doesn't follow provider/model format"
            parts = model_id.split("/")
            assert len(parts) == 2, f"Model {model_id} has invalid format"
            assert len(parts[0]) > 0, f"Model {model_id} has empty provider"
            assert len(parts[1]) > 0, f"Model {model_id} has empty model name"
    
    def test_model_providers_are_valid(self):
        """Test that model providers are recognized."""
        valid_providers = ["openai", "anthropic", "google", "meta", "mistral"]
        
        for model_id, config in MODEL_REGISTRY.items():
            provider = config.get("provider", "")
            assert provider in valid_providers, f"Unknown provider {provider} for model {model_id}"

