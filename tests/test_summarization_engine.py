"""
Unit tests for Summarization Engine.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from datetime import datetime

from src.services.summarization_engine import SummarizationEngine
from src.constants import TRIGGER_MESSAGE_COUNT


class TestSummarizationEngine:
    """Test cases for SummarizationEngine."""
    
    @pytest.fixture
    def mock_adapter(self):
        """Create a mock OpenRouter adapter."""
        adapter = AsyncMock()
        adapter.chat_completion = AsyncMock(return_value={
            "content": "This is a test summary of the conversation.",
            "model": "openai/gpt-4-turbo",
            "tokens": 50
        })
        return adapter
    
    @pytest.fixture
    def engine(self, mock_adapter):
        """Create a SummarizationEngine instance with mock adapter."""
        return SummarizationEngine(mock_adapter)
    
    @pytest.mark.asyncio
    async def test_should_trigger_at_threshold(self, engine):
        """Test summarization triggers at message threshold."""
        thread_id = str(uuid4())
        
        with patch('src.services.summarization_engine.async_session_maker') as mock_session:
            mock_ctx = AsyncMock()
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=None)
            
            # Mock message count at threshold (10)
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = 10
            mock_ctx.execute = AsyncMock(return_value=mock_result)
            
            should_trigger = await engine.should_trigger(thread_id)
            assert should_trigger is True
    
    @pytest.mark.asyncio
    async def test_should_not_trigger_below_threshold(self, engine):
        """Test summarization doesn't trigger below threshold."""
        thread_id = str(uuid4())
        
        with patch('src.services.summarization_engine.async_session_maker') as mock_session:
            mock_ctx = AsyncMock()
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=None)
            
            # Mock message count below threshold
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = 7
            mock_ctx.execute = AsyncMock(return_value=mock_result)
            
            should_trigger = await engine.should_trigger(thread_id)
            assert should_trigger is False
    
    @pytest.mark.asyncio
    async def test_should_trigger_at_multiple_of_threshold(self, engine):
        """Test summarization triggers at multiples of threshold."""
        thread_id = str(uuid4())
        
        with patch('src.services.summarization_engine.async_session_maker') as mock_session:
            mock_ctx = AsyncMock()
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=None)
            
            # Mock message count at 20 (2x threshold)
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = 20
            mock_ctx.execute = AsyncMock(return_value=mock_result)
            
            should_trigger = await engine.should_trigger(thread_id)
            assert should_trigger is True
    
    @pytest.mark.asyncio
    async def test_should_trigger_invalid_thread_id(self, engine):
        """Test should_trigger returns False for invalid thread ID."""
        should_trigger = await engine.should_trigger("invalid-uuid")
        assert should_trigger is False
    
    def test_format_messages_for_summary(self, engine):
        """Test message formatting for summary generation."""
        class MockMessage:
            def __init__(self, role, content):
                self.role = role
                self.content = content
        
        messages = [
            MockMessage("user", "Hello, how are you?"),
            MockMessage("assistant", "I'm doing well, thank you!"),
            MockMessage("user", "Great to hear!"),
        ]
        
        formatted = engine._format_messages_for_summary(messages)
        
        assert "User: Hello, how are you?" in formatted
        assert "Assistant: I'm doing well, thank you!" in formatted
        assert "User: Great to hear!" in formatted
    
    def test_format_messages_truncates_long_content(self, engine):
        """Test that long messages are truncated in summary formatting."""
        class MockMessage:
            def __init__(self, role, content):
                self.role = role
                self.content = content
        
        long_content = "A" * 1000  # Very long message
        messages = [MockMessage("user", long_content)]
        
        formatted = engine._format_messages_for_summary(messages)
        
        # Should be truncated with "..."
        assert len(formatted) < len(long_content)
        assert "..." in formatted
    
    @pytest.mark.asyncio
    async def test_generate_summary_invalid_thread(self, engine):
        """Test generate_summary returns None for invalid thread ID."""
        result = await engine.generate_summary("invalid-uuid")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_latest_summary_invalid_thread(self, engine):
        """Test get_latest_summary returns None for invalid thread ID."""
        result = await engine.get_latest_summary("invalid-uuid")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_thread_summaries_invalid_thread(self, engine):
        """Test get_thread_summaries returns empty list for invalid thread ID."""
        result = await engine.get_thread_summaries("invalid-uuid")
        assert result == []

