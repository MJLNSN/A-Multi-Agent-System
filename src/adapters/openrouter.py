"""
OpenRouter API adapter for the Multi-Agent Chat Threading System.
Provides a unified interface for calling LLM models through OpenRouter.
Includes retry logic and error handling.
"""

import httpx
from typing import Dict, List, Any, Optional
from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_exponential,
    retry_if_exception_type
)
import structlog

from src.config import settings
from src.constants import (
    OPENROUTER_REFERER,
    OPENROUTER_TITLE,
    ERROR_RATE_LIMIT,
    ERROR_AUTH_FAILED,
    ERROR_OPENROUTER_TIMEOUT
)

logger = structlog.get_logger()


class OpenRouterError(Exception):
    """Base exception for OpenRouter API errors."""
    pass


class OpenRouterRateLimitError(OpenRouterError):
    """Raised when rate limit is exceeded."""
    def __init__(self, message: str, retry_after: Optional[str] = None):
        super().__init__(message)
        self.retry_after = retry_after


class OpenRouterAuthError(OpenRouterError):
    """Raised when authentication fails."""
    pass


class OpenRouterAdapter:
    """
    Adapter for OpenRouter API calls.
    Handles authentication, request formatting, and error handling.
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the OpenRouter adapter.
        
        Args:
            api_key: OpenRouter API key. If not provided, uses settings.
        """
        self.api_key = api_key or settings.openrouter_api_key
        self.base_url = settings.openrouter_base_url
        self.timeout = settings.openrouter_timeout
        
        # Create async HTTP client with timeout
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": OPENROUTER_REFERER,
                "X-Title": OPENROUTER_TITLE
            }
        )
    
    async def chat_completion(
        self, 
        model: str, 
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call OpenRouter chat completion API with automatic retry on timeout.
        
        Args:
            model: OpenRouter model identifier (e.g., "openai/gpt-4-turbo")
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            **kwargs: Additional parameters to pass to the API
        
        Returns:
            Dict containing 'content', 'model', 'tokens', and 'finish_reason'
        
        Raises:
            OpenRouterError: On API errors
            OpenRouterRateLimitError: When rate limited
            OpenRouterAuthError: On authentication failure
        """
        return await self._chat_completion_with_retry(
            model, messages, temperature, max_tokens, **kwargs
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(httpx.TimeoutException),
        reraise=True
    )
    async def _chat_completion_with_retry(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs
    ) -> Dict[str, Any]:
        """Internal method with retry logic. Timeout exceptions propagate for retry."""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            **kwargs
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        try:
            logger.info(
                "openrouter_request",
                model=model,
                message_count=len(messages)
            )
            
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                json=payload
            )
            response.raise_for_status()
            
            data = response.json()
            
            result = {
                "content": data["choices"][0]["message"]["content"],
                "model": data.get("model", model),
                "tokens": data.get("usage", {}).get("total_tokens", 0),
                "finish_reason": data["choices"][0].get("finish_reason", "stop"),
                "usage": data.get("usage", {})
            }
            
            logger.info(
                "openrouter_response",
                model=model,
                tokens=result["tokens"],
                finish_reason=result["finish_reason"]
            )
            
            return result
        
        except httpx.TimeoutException:
            # Let timeout propagate for retry by tenacity decorator
            logger.warning("openrouter_timeout_retry", model=model)
            raise
        
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            
            if status_code == 429:
                retry_after = e.response.headers.get("retry-after", "unknown")
                logger.warning(
                    "openrouter_rate_limit",
                    model=model,
                    retry_after=retry_after
                )
                raise OpenRouterRateLimitError(
                    ERROR_RATE_LIMIT.format(retry_after=retry_after),
                    retry_after=retry_after
                )
            
            elif status_code == 401:
                logger.error("openrouter_auth_error")
                raise OpenRouterAuthError(ERROR_AUTH_FAILED)
            
            elif status_code == 400:
                try:
                    error_detail = e.response.json().get("error", {}).get("message", str(e))
                except Exception:
                    error_detail = str(e)
                logger.error("openrouter_bad_request", detail=error_detail)
                raise OpenRouterError(f"Bad request: {error_detail}")
            
            else:
                logger.error(
                    "openrouter_http_error",
                    status_code=status_code,
                    detail=str(e)
                )
                raise OpenRouterError(f"OpenRouter API error ({status_code}): {str(e)}")
        
        except httpx.HTTPError as e:
            logger.error("openrouter_network_error", detail=str(e))
            raise OpenRouterError(f"Network error: {str(e)}")
    
    async def close(self):
        """Close the HTTP client and cleanup resources."""
        await self.client.aclose()
        logger.info("openrouter_adapter_closed")

