"""
Unit tests for OpenRouter Adapter.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

from src.adapters.openrouter import (
    OpenRouterAdapter,
    OpenRouterError,
    OpenRouterRateLimitError,
    OpenRouterAuthError
)


class TestOpenRouterAdapter:
    """Test cases for OpenRouterAdapter."""
    
    @pytest.fixture
    def adapter(self):
        """Create an adapter instance with test API key."""
        with patch('src.adapters.openrouter.settings') as mock_settings:
            mock_settings.openrouter_api_key = "test-api-key"
            mock_settings.openrouter_base_url = "https://openrouter.ai/api/v1"
            mock_settings.openrouter_timeout = 60
            return OpenRouterAdapter(api_key="test-api-key")
    
    @pytest.mark.asyncio
    async def test_chat_completion_success(self, adapter):
        """Test successful chat completion."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {"content": "Hello! How can I help?"},
                "finish_reason": "stop"
            }],
            "model": "openai/gpt-4-turbo",
            "usage": {"total_tokens": 50}
        }
        mock_response.raise_for_status = MagicMock()
        
        with patch.object(adapter.client, 'post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            
            result = await adapter.chat_completion(
                model="openai/gpt-4-turbo",
                messages=[{"role": "user", "content": "Hello!"}]
            )
            
            assert result["content"] == "Hello! How can I help?"
            assert result["model"] == "openai/gpt-4-turbo"
            assert result["tokens"] == 50
            assert result["finish_reason"] == "stop"
    
    @pytest.mark.asyncio
    async def test_chat_completion_timeout(self, adapter):
        """Test timeout handling - retries exhausted raises TimeoutException."""
        with patch.object(adapter.client, 'post', new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.TimeoutException("Connection timed out")
            
            # After 3 retries, the original TimeoutException is raised
            with pytest.raises(httpx.TimeoutException):
                await adapter.chat_completion(
                    model="openai/gpt-4-turbo",
                    messages=[{"role": "user", "content": "Hello!"}]
                )
    
    @pytest.mark.asyncio
    async def test_chat_completion_rate_limit(self, adapter):
        """Test rate limit (429) handling."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"retry-after": "30"}
        
        http_error = httpx.HTTPStatusError(
            "Rate limited",
            request=MagicMock(),
            response=mock_response
        )
        
        with patch.object(adapter.client, 'post', new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = http_error
            
            with pytest.raises(OpenRouterRateLimitError) as exc_info:
                await adapter.chat_completion(
                    model="openai/gpt-4-turbo",
                    messages=[{"role": "user", "content": "Hello!"}]
                )
            
            assert exc_info.value.retry_after == "30"
    
    @pytest.mark.asyncio
    async def test_chat_completion_auth_error(self, adapter):
        """Test authentication (401) handling."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        
        http_error = httpx.HTTPStatusError(
            "Unauthorized",
            request=MagicMock(),
            response=mock_response
        )
        
        with patch.object(adapter.client, 'post', new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = http_error
            
            with pytest.raises(OpenRouterAuthError):
                await adapter.chat_completion(
                    model="openai/gpt-4-turbo",
                    messages=[{"role": "user", "content": "Hello!"}]
                )
    
    @pytest.mark.asyncio
    async def test_chat_completion_bad_request(self, adapter):
        """Test bad request (400) handling."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": {"message": "Invalid model specified"}
        }
        
        http_error = httpx.HTTPStatusError(
            "Bad request",
            request=MagicMock(),
            response=mock_response
        )
        
        with patch.object(adapter.client, 'post', new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = http_error
            
            with pytest.raises(OpenRouterError, match="Bad request"):
                await adapter.chat_completion(
                    model="invalid/model",
                    messages=[{"role": "user", "content": "Hello!"}]
                )
    
    @pytest.mark.asyncio
    async def test_chat_completion_network_error(self, adapter):
        """Test network error handling."""
        with patch.object(adapter.client, 'post', new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.HTTPError("Connection failed")
            
            with pytest.raises(OpenRouterError, match="Network error"):
                await adapter.chat_completion(
                    model="openai/gpt-4-turbo",
                    messages=[{"role": "user", "content": "Hello!"}]
                )
    
    @pytest.mark.asyncio
    async def test_close(self, adapter):
        """Test adapter cleanup."""
        with patch.object(adapter.client, 'aclose', new_callable=AsyncMock) as mock_close:
            await adapter.close()
            mock_close.assert_called_once()
    
    def test_headers_set_correctly(self, adapter):
        """Test that headers are configured correctly."""
        headers = adapter.client.headers
        
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test-api-key"
        assert headers["Content-Type"] == "application/json"
        assert "HTTP-Referer" in headers
        assert "X-Title" in headers


class TestOpenRouterAdapterRetry:
    """Test retry behavior."""
    
    @pytest.fixture
    def adapter(self):
        """Create an adapter instance."""
        with patch('src.adapters.openrouter.settings') as mock_settings:
            mock_settings.openrouter_api_key = "test-api-key"
            mock_settings.openrouter_base_url = "https://openrouter.ai/api/v1"
            mock_settings.openrouter_timeout = 60
            return OpenRouterAdapter(api_key="test-api-key")
    
    @pytest.mark.asyncio
    async def test_retry_on_timeout(self, adapter):
        """Test that timeouts trigger retry."""
        call_count = 0
        
        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.TimeoutException("Timeout")
            
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Success"}, "finish_reason": "stop"}],
                "usage": {"total_tokens": 10}
            }
            mock_response.raise_for_status = MagicMock()
            return mock_response
        
        with patch.object(adapter.client, 'post', side_effect=mock_post):
            result = await adapter.chat_completion(
                model="openai/gpt-4-turbo",
                messages=[{"role": "user", "content": "Hello!"}]
            )
            
            assert result["content"] == "Success"
            assert call_count == 3  # 2 timeouts + 1 success
    
    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, adapter):
        """Test that max retries results in exception."""
        async def always_timeout(*args, **kwargs):
            raise httpx.TimeoutException("Timeout")
        
        with patch.object(adapter.client, 'post', side_effect=always_timeout):
            with pytest.raises(httpx.TimeoutException):
                await adapter.chat_completion(
                    model="openai/gpt-4-turbo",
                    messages=[{"role": "user", "content": "Hello!"}]
                )

