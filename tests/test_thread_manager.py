"""
Tests for Thread Manager Service.
"""

import pytest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from src.services.thread_manager import ThreadManager


@pytest.fixture
def thread_manager():
    """Create a ThreadManager instance for testing."""
    return ThreadManager()


class TestThreadManager:
    """Test cases for ThreadManager."""
    
    @pytest.mark.asyncio
    async def test_create_thread_with_valid_model(self, thread_manager):
        """Test creating a thread with a valid model."""
        with patch('src.services.thread_manager.async_session_maker') as mock_session:
            # Setup mock
            mock_ctx = AsyncMock()
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=None)
            
            mock_ctx.add = AsyncMock()
            mock_ctx.commit = AsyncMock()
            mock_ctx.refresh = AsyncMock()
            
            # Test
            result = await thread_manager.create_thread(
                system_prompt="You are a helpful assistant.",
                title="Test Thread",
                current_model="openai/gpt-4-turbo",
                user_id="test_user"
            )
            
            # Verify
            assert mock_ctx.add.called
            assert mock_ctx.commit.called
    
    @pytest.mark.asyncio
    async def test_create_thread_with_invalid_model(self, thread_manager):
        """Test that creating a thread with invalid model raises ValueError."""
        with pytest.raises(ValueError, match="Invalid model"):
            await thread_manager.create_thread(
                system_prompt="Test prompt",
                current_model="invalid/model"
            )
    
    def test_thread_to_dict(self, thread_manager):
        """Test thread object to dictionary conversion."""
        # Create a mock thread object
        class MockThread:
            thread_id = uuid4()
            user_id = "test_user"
            title = "Test"
            system_prompt = "Test prompt"
            current_model = "openai/gpt-4-turbo"
            message_count = 5
            status = "active"
            created_at = "2025-01-01T00:00:00Z"
            updated_at = "2025-01-01T00:00:00Z"
        
        result = thread_manager._thread_to_dict(MockThread())
        
        assert "thread_id" in result
        assert result["user_id"] == "test_user"
        assert result["message_count"] == 5

