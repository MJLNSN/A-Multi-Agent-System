"""
Unit tests for Message Handler Service.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from datetime import datetime

from src.services.message_handler import MessageHandler


class TestMessageHandler:
    """Test cases for MessageHandler."""
    
    @pytest.fixture
    def mock_llm_orchestrator(self):
        """Create a mock LLM orchestrator."""
        orchestrator = AsyncMock()
        orchestrator.generate_response = AsyncMock(return_value={
            "content": "This is a test response from the LLM.",
            "model": "openai/gpt-4-turbo",
            "tokens": 20,
            "usage": {"total_tokens": 100}
        })
        return orchestrator
    
    @pytest.fixture
    def mock_summarizer(self):
        """Create a mock summarization engine."""
        summarizer = AsyncMock()
        summarizer.should_trigger = AsyncMock(return_value=False)
        summarizer.generate_summary = AsyncMock(return_value=None)
        return summarizer
    
    @pytest.fixture
    def handler(self, mock_llm_orchestrator, mock_summarizer):
        """Create a MessageHandler instance with mocks."""
        return MessageHandler(mock_llm_orchestrator, mock_summarizer)
    
    @pytest.mark.asyncio
    async def test_get_thread_lock_creates_new_lock(self, handler):
        """Test that _get_thread_lock creates a new lock for new thread."""
        thread_id = str(uuid4())
        
        lock = await handler._get_thread_lock(thread_id)
        
        assert isinstance(lock, asyncio.Lock)
        assert thread_id in handler._locks
    
    @pytest.mark.asyncio
    async def test_get_thread_lock_reuses_existing_lock(self, handler):
        """Test that _get_thread_lock reuses lock for same thread."""
        thread_id = str(uuid4())
        
        lock1 = await handler._get_thread_lock(thread_id)
        lock2 = await handler._get_thread_lock(thread_id)
        
        assert lock1 is lock2
    
    @pytest.mark.asyncio
    async def test_get_thread_lock_different_threads(self, handler):
        """Test that different threads get different locks."""
        thread_id1 = str(uuid4())
        thread_id2 = str(uuid4())
        
        lock1 = await handler._get_thread_lock(thread_id1)
        lock2 = await handler._get_thread_lock(thread_id2)
        
        assert lock1 is not lock2
    
    @pytest.mark.asyncio
    async def test_process_user_message_invalid_thread_id(self, handler):
        """Test process_user_message raises error for invalid thread ID."""
        with pytest.raises(ValueError, match="Invalid thread ID"):
            await handler.process_user_message(
                thread_id="invalid-uuid",
                content="Hello!"
            )
    
    @pytest.mark.asyncio
    async def test_generate_summary_async_handles_error(self, handler, mock_summarizer):
        """Test that _generate_summary_async handles errors gracefully."""
        thread_id = str(uuid4())
        mock_summarizer.generate_summary = AsyncMock(
            side_effect=Exception("Summary generation failed")
        )
        
        # Should not raise - errors are logged but not propagated
        await handler._generate_summary_async(thread_id)
        
        # Verify summarizer was called
        mock_summarizer.generate_summary.assert_called_once_with(thread_id)
    
    @pytest.mark.asyncio
    async def test_assemble_context_invalid_thread(self, handler):
        """Test _assemble_context returns empty list for invalid thread ID."""
        context = await handler._assemble_context("invalid-uuid")
        assert context == []
    
    @pytest.mark.asyncio
    async def test_get_thread_messages_invalid_thread(self, handler):
        """Test get_thread_messages returns empty for invalid thread ID."""
        result = await handler.get_thread_messages("invalid-uuid")
        assert result == {"messages": [], "total": 0}


class TestMessageHandlerConcurrency:
    """Concurrency tests for MessageHandler."""
    
    @pytest.fixture
    def mock_llm_orchestrator(self):
        """Create a mock LLM orchestrator with delay."""
        orchestrator = AsyncMock()
        
        async def slow_response(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate API latency
            return {
                "content": "Response",
                "model": "openai/gpt-4-turbo",
                "tokens": 10,
                "usage": {}
            }
        
        orchestrator.generate_response = slow_response
        return orchestrator
    
    @pytest.fixture
    def mock_summarizer(self):
        """Create a mock summarization engine."""
        summarizer = AsyncMock()
        summarizer.should_trigger = AsyncMock(return_value=False)
        return summarizer
    
    @pytest.fixture
    def handler(self, mock_llm_orchestrator, mock_summarizer):
        """Create a MessageHandler instance."""
        return MessageHandler(mock_llm_orchestrator, mock_summarizer)
    
    @pytest.mark.asyncio
    async def test_concurrent_lock_acquisition(self, handler):
        """Test that concurrent requests to same thread are serialized."""
        thread_id = str(uuid4())
        execution_order = []
        
        async def acquire_and_record(label: str):
            lock = await handler._get_thread_lock(thread_id)
            async with lock:
                execution_order.append(f"{label}_start")
                await asyncio.sleep(0.05)
                execution_order.append(f"{label}_end")
        
        # Run concurrently
        await asyncio.gather(
            acquire_and_record("A"),
            acquire_and_record("B")
        )
        
        # One should complete before the other starts
        # Either A_start, A_end, B_start, B_end or B_start, B_end, A_start, A_end
        assert execution_order[0].endswith("_start")
        assert execution_order[1].endswith("_end")
        assert execution_order[2].endswith("_start")
        assert execution_order[3].endswith("_end")
    
    @pytest.mark.asyncio
    async def test_different_threads_not_blocked(self, handler):
        """Test that different threads can process concurrently."""
        thread_id1 = str(uuid4())
        thread_id2 = str(uuid4())
        
        start_times = {}
        end_times = {}
        
        async def process_thread(thread_id: str, label: str):
            lock = await handler._get_thread_lock(thread_id)
            async with lock:
                start_times[label] = asyncio.get_event_loop().time()
                await asyncio.sleep(0.1)
                end_times[label] = asyncio.get_event_loop().time()
        
        # Run concurrently
        await asyncio.gather(
            process_thread(thread_id1, "A"),
            process_thread(thread_id2, "B")
        )
        
        # Both should start at approximately the same time
        time_diff = abs(start_times["A"] - start_times["B"])
        assert time_diff < 0.05  # Less than 50ms difference

