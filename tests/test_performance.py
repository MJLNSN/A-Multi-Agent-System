"""
Performance and stress tests for the Multi-Agent Chat Threading System.
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, patch
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
import statistics


class TestPerformance:
    """Performance benchmark tests."""
    
    @pytest.mark.asyncio
    async def test_token_counter_performance(self):
        """Benchmark token counting performance."""
        from src.utils.token_counter import TokenCounter
        
        counter = TokenCounter()
        
        # Test with various text sizes
        texts = [
            "Hello world",  # Short
            "The quick brown fox jumps over the lazy dog. " * 10,  # Medium
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 100,  # Long
        ]
        
        for text in texts:
            start = time.perf_counter()
            iterations = 1000
            
            for _ in range(iterations):
                counter.count_tokens(text, "openai/gpt-4-turbo")
            
            elapsed = time.perf_counter() - start
            avg_time_ms = (elapsed / iterations) * 1000
            
            # Token counting should be fast (< 1ms per call on average)
            assert avg_time_ms < 1.0, f"Token counting too slow: {avg_time_ms:.3f}ms for {len(text)} chars"
    
    @pytest.mark.asyncio
    async def test_context_trimming_performance(self):
        """Benchmark context trimming performance."""
        from src.utils.token_counter import TokenCounter
        
        counter = TokenCounter()
        
        # Create a large context
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for i in range(100):
            messages.append({
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"This is message number {i}. " * 20
            })
        
        start = time.perf_counter()
        iterations = 100
        
        for _ in range(iterations):
            counter.trim_context_to_fit(
                messages.copy(),
                model="openai/gpt-4-turbo",
                max_tokens=4000
            )
        
        elapsed = time.perf_counter() - start
        avg_time_ms = (elapsed / iterations) * 1000
        
        # Context trimming should complete in reasonable time (< 10ms)
        assert avg_time_ms < 10.0, f"Context trimming too slow: {avg_time_ms:.3f}ms"
    
    @pytest.mark.asyncio
    async def test_model_registry_lookup_performance(self):
        """Benchmark model registry lookups."""
        from src.models.registry import validate_model, get_model_config, list_available_models
        
        models = list_available_models()
        
        start = time.perf_counter()
        iterations = 10000
        
        for _ in range(iterations):
            for model in models:
                validate_model(model)
                get_model_config(model)
        
        elapsed = time.perf_counter() - start
        total_ops = iterations * len(models) * 2
        ops_per_sec = total_ops / elapsed
        
        # Should handle at least 100k ops/sec
        assert ops_per_sec > 100000, f"Registry lookup too slow: {ops_per_sec:.0f} ops/sec"


class TestConcurrency:
    """Concurrency and thread-safety tests."""
    
    @pytest.mark.asyncio
    async def test_concurrent_thread_locks(self):
        """Test concurrent access to thread locks."""
        from src.services.message_handler import MessageHandler
        
        mock_llm = AsyncMock()
        mock_summarizer = AsyncMock()
        handler = MessageHandler(mock_llm, mock_summarizer)
        
        thread_ids = [str(uuid4()) for _ in range(10)]
        
        async def get_locks_repeatedly(tid: str, count: int):
            for _ in range(count):
                await handler._get_thread_lock(tid)
        
        # Run many concurrent lock acquisitions
        tasks = [
            get_locks_repeatedly(tid, 100)
            for tid in thread_ids
        ]
        
        await asyncio.gather(*tasks)
        
        # Each thread should have exactly one lock
        assert len(handler._locks) == len(thread_ids)
    
    @pytest.mark.asyncio
    async def test_lock_contention_handling(self):
        """Test that lock contention is handled correctly."""
        from src.services.message_handler import MessageHandler
        
        mock_llm = AsyncMock()
        mock_summarizer = AsyncMock()
        handler = MessageHandler(mock_llm, mock_summarizer)
        
        thread_id = str(uuid4())
        execution_log = []
        
        async def critical_section(task_id: int):
            lock = await handler._get_thread_lock(thread_id)
            async with lock:
                execution_log.append(f"start_{task_id}")
                await asyncio.sleep(0.01)  # Simulate work
                execution_log.append(f"end_{task_id}")
        
        # Launch many concurrent tasks for same thread
        tasks = [critical_section(i) for i in range(5)]
        await asyncio.gather(*tasks)
        
        # Verify no interleaving (each start should be followed by its end)
        for i in range(5):
            start_idx = execution_log.index(f"start_{i}")
            end_idx = execution_log.index(f"end_{i}")
            
            # No other task should have started between this task's start and end
            for j in range(5):
                if i != j:
                    other_start_idx = execution_log.index(f"start_{j}")
                    assert not (start_idx < other_start_idx < end_idx), \
                        f"Task {j} interleaved with task {i}"
    
    @pytest.mark.asyncio
    async def test_parallel_different_threads(self):
        """Test that different threads can be processed in parallel."""
        from src.services.message_handler import MessageHandler
        
        mock_llm = AsyncMock()
        mock_summarizer = AsyncMock()
        handler = MessageHandler(mock_llm, mock_summarizer)
        
        start_times = {}
        end_times = {}
        
        async def process(thread_id: str, task_id: str):
            lock = await handler._get_thread_lock(thread_id)
            async with lock:
                start_times[task_id] = time.perf_counter()
                await asyncio.sleep(0.1)
                end_times[task_id] = time.perf_counter()
        
        # Different threads should process in parallel
        thread_ids = [str(uuid4()) for _ in range(3)]
        tasks = [
            process(thread_ids[i], f"task_{i}")
            for i in range(3)
        ]
        
        start = time.perf_counter()
        await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start
        
        # If parallel, total time should be ~0.1s, not ~0.3s
        assert total_time < 0.2, f"Parallel execution too slow: {total_time:.2f}s"


class TestStress:
    """Stress tests for edge cases and limits."""
    
    def test_large_message_content(self):
        """Test handling of very large message content."""
        from src.utils.token_counter import TokenCounter
        
        counter = TokenCounter()
        
        # 100KB of text
        large_text = "X" * 100000
        
        start = time.perf_counter()
        tokens = counter.count_tokens(large_text, "openai/gpt-4-turbo")
        elapsed = time.perf_counter() - start
        
        assert tokens > 0
        # tiktoken can be slow for very large texts - allow up to 10s
        assert elapsed < 10.0, f"Large text processing too slow: {elapsed:.2f}s"
    
    def test_many_messages_in_context(self):
        """Test handling many messages in context."""
        from src.utils.token_counter import TokenCounter
        
        counter = TokenCounter()
        
        # 1000 messages
        messages = [
            {
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"Message {i}"
            }
            for i in range(1000)
        ]
        
        start = time.perf_counter()
        trimmed = counter.trim_context_to_fit(
            messages,
            model="openai/gpt-4-turbo",
            max_tokens=4000
        )
        elapsed = time.perf_counter() - start
        
        assert len(trimmed) < len(messages)
        assert elapsed < 1.0, f"Many messages handling too slow: {elapsed:.2f}s"
    
    @pytest.mark.asyncio
    async def test_rapid_lock_acquisition(self):
        """Test rapid sequential lock acquisition."""
        from src.services.message_handler import MessageHandler
        
        mock_llm = AsyncMock()
        mock_summarizer = AsyncMock()
        handler = MessageHandler(mock_llm, mock_summarizer)
        
        thread_id = str(uuid4())
        
        start = time.perf_counter()
        for _ in range(1000):
            lock = await handler._get_thread_lock(thread_id)
            async with lock:
                pass  # Just acquire and release
        elapsed = time.perf_counter() - start
        
        # 1000 lock cycles should complete quickly
        assert elapsed < 1.0, f"Lock acquisition too slow: {elapsed:.2f}s"


class TestBoundaryConditions:
    """Test boundary conditions and edge cases."""
    
    def test_empty_message_content(self):
        """Test handling of empty message content."""
        from src.utils.token_counter import TokenCounter
        
        counter = TokenCounter()
        
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": ""},
            {"role": "assistant", "content": ""},
        ]
        
        # Should not raise
        tokens = counter.count_messages_tokens(messages, "openai/gpt-4-turbo")
        assert tokens >= 0
    
    def test_unicode_content(self):
        """Test handling of unicode content."""
        from src.utils.token_counter import TokenCounter
        
        counter = TokenCounter()
        
        # Various unicode characters
        texts = [
            "Hello 世界 🌍",
            "Привет мир",
            "مرحبا بالعالم",
            "🎉🎊🎈🎁",
        ]
        
        for text in texts:
            tokens = counter.count_tokens(text, "openai/gpt-4-turbo")
            assert tokens > 0
    
    def test_special_characters(self):
        """Test handling of special characters."""
        from src.utils.token_counter import TokenCounter
        
        counter = TokenCounter()
        
        special_text = """
        Code: `print("Hello")`
        Math: x² + y² = z²
        Symbols: ™ © ® § ¶
        Arrows: → ← ↑ ↓
        """
        
        tokens = counter.count_tokens(special_text, "openai/gpt-4-turbo")
        assert tokens > 0
    
    def test_model_registry_unknown_model(self):
        """Test registry behavior with unknown model."""
        from src.models.registry import validate_model, get_model_config
        
        assert validate_model("unknown/model") is False
        assert get_model_config("unknown/model") is None
    
    def test_max_context_limit(self):
        """Test that context is properly limited."""
        from src.utils.token_counter import TokenCounter
        
        counter = TokenCounter()
        
        # Create messages that exceed the limit
        messages = [{"role": "system", "content": "System prompt."}]
        for i in range(50):
            messages.append({
                "role": "user",
                "content": "A" * 500  # ~125 tokens each
            })
        
        trimmed = counter.trim_context_to_fit(
            messages,
            model="openai/gpt-4-turbo",
            max_tokens=1000
        )
        
        # Should be significantly trimmed
        total_tokens = counter.count_messages_tokens(trimmed, "openai/gpt-4-turbo")
        assert total_tokens <= 1000

