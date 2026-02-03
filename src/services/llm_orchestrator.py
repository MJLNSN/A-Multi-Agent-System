"""
LLM Orchestrator Service for the Multi-Agent Chat Threading System.
Coordinates model selection and LLM API calls.
"""

from typing import List, Dict, Any, Optional

from src.adapters.openrouter import OpenRouterAdapter
from src.models.registry import validate_model, list_available_models
from src.utils.logging import get_logger

logger = get_logger("llm_orchestrator")


class LLMOrchestrator:
    """
    Orchestrator for coordinating LLM model calls.
    Handles model selection logic and request routing.
    """
    
    def __init__(self, openrouter_adapter: OpenRouterAdapter):
        """
        Initialize the LLM Orchestrator.
        
        Args:
            openrouter_adapter: OpenRouter API adapter instance
        """
        self.adapter = openrouter_adapter
    
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        thread_current_model: str,
        requested_model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate a response from an LLM.
        
        Model selection priority:
        1. requested_model (if provided and valid)
        2. thread_current_model (fallback)
        
        Args:
            messages: List of message dicts for context
            thread_current_model: Thread's default model
            requested_model: Optional model override for this request
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
        
        Returns:
            Dict containing 'content', 'model', 'tokens', and 'finish_reason'
        
        Raises:
            ValueError: If no valid model can be determined
        """
        # Determine effective model
        effective_model = self._determine_effective_model(
            thread_current_model, 
            requested_model
        )
        
        logger.info(
            "llm_request",
            effective_model=effective_model,
            requested_model=requested_model,
            thread_model=thread_current_model,
            message_count=len(messages)
        )
        
        # Call OpenRouter API
        response = await self.adapter.chat_completion(
            model=effective_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        logger.info(
            "llm_response",
            model=effective_model,
            tokens=response.get("tokens", 0)
        )
        
        return {
            "content": response["content"],
            "model": effective_model,
            "tokens": response.get("tokens", 0),
            "finish_reason": response.get("finish_reason", "stop"),
            "usage": response.get("usage", {})
        }
    
    def _determine_effective_model(
        self,
        thread_current_model: str,
        requested_model: Optional[str]
    ) -> str:
        """
        Determine which model to use for a request.
        
        Priority: requested_model > thread_current_model
        
        Args:
            thread_current_model: Thread's default model
            requested_model: Optional model override
        
        Returns:
            Model identifier to use
        
        Raises:
            ValueError: If no valid model can be determined
        """
        # Try requested model first
        if requested_model:
            if validate_model(requested_model):
                return requested_model
            else:
                available = list_available_models()
                raise ValueError(
                    f"Invalid requested model: {requested_model}. "
                    f"Available models: {available}"
                )
        
        # Fall back to thread model
        if validate_model(thread_current_model):
            return thread_current_model
        
        # Neither model is valid
        available = list_available_models()
        raise ValueError(
            f"Invalid thread model: {thread_current_model}. "
            f"Available models: {available}"
        )

