"""
Token counting utilities for context management.
Supports accurate counting for OpenAI models via tiktoken,
and approximation for other models.
"""

from typing import List, Dict, Optional
import tiktoken

from src.constants import (
    MESSAGE_TOKEN_OVERHEAD,
    CONVERSATION_TOKEN_OVERHEAD,
    CHARS_PER_TOKEN_ESTIMATE,
    CONTEXT_SUMMARY_PREFIX
)


class TokenCounter:
    """
    Token counter for managing context window limits.
    Uses tiktoken for OpenAI models and approximation for others.
    """
    
    def __init__(self):
        """Initialize token counter with encoders for supported models."""
        # Cache tiktoken encoders for OpenAI models
        self._encoders = {}
        self._model_encodings = {
            "openai/gpt-4-turbo": "cl100k_base",
            "openai/gpt-4": "cl100k_base",
            "openai/gpt-3.5-turbo": "cl100k_base",
        }
    
    def _get_encoder(self, model: str):
        """
        Get the tiktoken encoder for a model.
        
        Args:
            model: Model identifier
        
        Returns:
            Tiktoken encoder or None for unsupported models
        """
        if model not in self._encoders:
            encoding_name = self._model_encodings.get(model)
            if encoding_name:
                try:
                    self._encoders[model] = tiktoken.get_encoding(encoding_name)
                except Exception:
                    self._encoders[model] = None
            else:
                self._encoders[model] = None
        
        return self._encoders[model]
    
    def count_tokens(self, text: str, model: str = "openai/gpt-4-turbo") -> int:
        """
        Count tokens in a text string.
        
        Args:
            text: Text to count tokens for
            model: Model identifier for accurate counting
        
        Returns:
            Token count (exact for OpenAI, approximate for others)
        """
        encoder = self._get_encoder(model)
        
        if encoder:
            # Exact counting with tiktoken
            return len(encoder.encode(text))
        else:
            # Approximation for non-tiktoken models
            return len(text) // CHARS_PER_TOKEN_ESTIMATE
    
    def count_messages_tokens(
        self, 
        messages: List[Dict[str, str]], 
        model: str = "openai/gpt-4-turbo"
    ) -> int:
        """
        Count total tokens in a list of messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model identifier
        
        Returns:
            Total token count including message overhead
        """
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            total += self.count_tokens(content, model)
            # Add overhead per message (role, formatting)
            total += MESSAGE_TOKEN_OVERHEAD
        
        # Add base overhead for the conversation
        total += CONVERSATION_TOKEN_OVERHEAD
        
        return total
    
    def trim_context_to_fit(
        self,
        messages: List[Dict[str, str]],
        model: str = "openai/gpt-4-turbo",
        max_tokens: int = 8000,
        preserve_system: bool = True,
        preserve_summary: bool = True
    ) -> List[Dict[str, str]]:
        """
        Trim messages to fit within token limit.
        Preserves system prompt and summary while trimming oldest messages.
        
        Args:
            messages: List of messages to trim
            model: Model identifier for token counting
            max_tokens: Maximum allowed tokens
            preserve_system: Whether to always keep system message
            preserve_summary: Whether to always keep summary message
        
        Returns:
            Trimmed list of messages
        """
        if not messages:
            return messages
        
        # Separate preserved messages from history
        preserved = []
        history = []
        
        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            # Keep system prompt
            if preserve_system and role == "system":
                preserved.append(msg)
            # Keep summary (assistant message starting with summary prefix)
            elif preserve_summary and role == "assistant" and (
                content.startswith("[Summary]") or 
                content.startswith(CONTEXT_SUMMARY_PREFIX)
            ):
                preserved.append(msg)
            else:
                history.append(msg)
        
        # Calculate tokens for preserved messages
        preserved_tokens = self.count_messages_tokens(preserved, model)
        
        # If preserved alone exceeds limit, return just preserved
        if preserved_tokens >= max_tokens:
            return preserved
        
        # Add history from most recent, respecting token limit
        remaining_tokens = max_tokens - preserved_tokens
        trimmed_history = []
        
        # Process from newest to oldest
        for msg in reversed(history):
            msg_tokens = self.count_tokens(msg.get("content", ""), model) + MESSAGE_TOKEN_OVERHEAD
            if remaining_tokens - msg_tokens >= 0:
                trimmed_history.insert(0, msg)
                remaining_tokens -= msg_tokens
            else:
                break
        
        # Combine preserved + trimmed history
        return preserved + trimmed_history


# Global instance for convenience
token_counter = TokenCounter()

