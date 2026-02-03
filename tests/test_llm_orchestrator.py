"""
Tests for LLM Orchestrator Service.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.services.llm_orchestrator import LLMOrchestrator


@pytest.fixture
def mock_adapter():
    """Create a mock OpenRouter adapter."""
    adapter = AsyncMock()
    adapter.chat_completion = AsyncMock(return_value={
        "content": "Test response",
        "model": "openai/gpt-4-turbo",
        "tokens": 100,
        "finish_reason": "stop",
        "usage": {"total_tokens": 100}
    })
    return adapter


@pytest.fixture
def orchestrator(mock_adapter):
    """Create an LLMOrchestrator instance with mock adapter."""
    return LLMOrchestrator(mock_adapter)


class TestLLMOrchestrator:
    """Test cases for LLMOrchestrator."""
    
    def test_determine_effective_model_with_requested_model(self, orchestrator):
        """Test that requested model takes priority."""
        result = orchestrator._determine_effective_model(
            thread_current_model="openai/gpt-4-turbo",
            requested_model="anthropic/claude-3.5-sonnet"
        )
        assert result == "anthropic/claude-3.5-sonnet"
    
    def test_determine_effective_model_with_thread_default(self, orchestrator):
        """Test fallback to thread model when no request model."""
        result = orchestrator._determine_effective_model(
            thread_current_model="openai/gpt-4-turbo",
            requested_model=None
        )
        assert result == "openai/gpt-4-turbo"
    
    def test_determine_effective_model_invalid_requested(self, orchestrator):
        """Test error when requested model is invalid."""
        with pytest.raises(ValueError, match="Invalid requested model"):
            orchestrator._determine_effective_model(
                thread_current_model="openai/gpt-4-turbo",
                requested_model="invalid/model"
            )
    
    def test_determine_effective_model_invalid_thread(self, orchestrator):
        """Test error when both models are invalid."""
        with pytest.raises(ValueError, match="Invalid thread model"):
            orchestrator._determine_effective_model(
                thread_current_model="invalid/model",
                requested_model=None
            )
    
    @pytest.mark.asyncio
    async def test_generate_response(self, orchestrator, mock_adapter):
        """Test response generation."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"}
        ]
        
        result = await orchestrator.generate_response(
            messages=messages,
            thread_current_model="openai/gpt-4-turbo",
            requested_model=None
        )
        
        assert result["content"] == "Test response"
        assert result["model"] == "openai/gpt-4-turbo"
        assert result["tokens"] == 100
        
        mock_adapter.chat_completion.assert_called_once()

