"""
Unit tests for Token Counter utility.
"""

import pytest
from src.utils.token_counter import TokenCounter, token_counter
from src.constants import MESSAGE_TOKEN_OVERHEAD, CONVERSATION_TOKEN_OVERHEAD


class TestTokenCounter:
    """Test cases for TokenCounter."""
    
    @pytest.fixture
    def counter(self):
        """Create a TokenCounter instance."""
        return TokenCounter()
    
    def test_count_tokens_openai_model(self, counter):
        """Test token counting for OpenAI models using tiktoken."""
        text = "Hello, how are you today?"
        tokens = counter.count_tokens(text, "openai/gpt-4-turbo")
        
        # tiktoken should give a reasonable count
        assert tokens > 0
        assert tokens < len(text)  # Tokens should be fewer than characters
    
    def test_count_tokens_non_openai_model(self, counter):
        """Test token counting approximation for non-OpenAI models."""
        text = "Hello, how are you today?"
        tokens = counter.count_tokens(text, "anthropic/claude-3.5-sonnet")
        
        # Approximation: ~4 chars per token
        expected_approx = len(text) // 4
        assert tokens == expected_approx
    
    def test_count_tokens_empty_string(self, counter):
        """Test token counting for empty string."""
        tokens = counter.count_tokens("", "openai/gpt-4-turbo")
        assert tokens == 0
    
    def test_count_messages_tokens(self, counter):
        """Test counting tokens for a list of messages."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        tokens = counter.count_messages_tokens(messages, "openai/gpt-4-turbo")
        
        # Should include content tokens + overhead
        assert tokens > 0
        # Should include message overhead and conversation overhead
        expected_min = (len(messages) * MESSAGE_TOKEN_OVERHEAD) + CONVERSATION_TOKEN_OVERHEAD
        assert tokens >= expected_min
    
    def test_count_messages_tokens_empty_list(self, counter):
        """Test counting tokens for empty message list."""
        tokens = counter.count_messages_tokens([], "openai/gpt-4-turbo")
        assert tokens == CONVERSATION_TOKEN_OVERHEAD
    
    def test_trim_context_preserves_system_prompt(self, counter):
        """Test that trimming preserves system prompt."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Message 2"},
            {"role": "assistant", "content": "Response 2"},
        ]
        
        trimmed = counter.trim_context_to_fit(
            messages, 
            model="openai/gpt-4-turbo",
            max_tokens=100
        )
        
        # System prompt should always be preserved
        assert trimmed[0]["role"] == "system"
        assert trimmed[0]["content"] == "You are a helpful assistant."
    
    def test_trim_context_preserves_summary(self, counter):
        """Test that trimming preserves summary message."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "assistant", "content": "[Previous conversation summary]: Some summary"},
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
        ]
        
        trimmed = counter.trim_context_to_fit(
            messages,
            model="openai/gpt-4-turbo",
            max_tokens=200,
            preserve_summary=True
        )
        
        # Both system and summary should be preserved
        assert len(trimmed) >= 2
        assert trimmed[0]["role"] == "system"
        assert "[Previous conversation summary]" in trimmed[1]["content"]
    
    def test_trim_context_removes_old_messages(self, counter):
        """Test that trimming removes oldest messages when exceeding limit."""
        # Create many messages
        messages = [{"role": "system", "content": "System prompt."}]
        for i in range(20):
            messages.append({"role": "user", "content": f"User message {i} " * 20})
            messages.append({"role": "assistant", "content": f"Assistant response {i} " * 20})
        
        trimmed = counter.trim_context_to_fit(
            messages,
            model="openai/gpt-4-turbo",
            max_tokens=500
        )
        
        # Should have fewer messages than original
        assert len(trimmed) < len(messages)
        # System prompt should still be first
        assert trimmed[0]["role"] == "system"
    
    def test_trim_context_empty_messages(self, counter):
        """Test trimming empty message list."""
        trimmed = counter.trim_context_to_fit(
            [],
            model="openai/gpt-4-turbo",
            max_tokens=1000
        )
        assert trimmed == []
    
    def test_global_token_counter_instance(self):
        """Test that global token_counter instance works."""
        tokens = token_counter.count_tokens("Hello world", "openai/gpt-4-turbo")
        assert tokens > 0

