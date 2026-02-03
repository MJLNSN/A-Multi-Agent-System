"""
Pytest configuration and shared fixtures.
"""

import pytest
import asyncio
import sys
from typing import Generator
from unittest.mock import AsyncMock, patch

# Add src to path
sys.path.insert(0, str(pytest.importorskip("pathlib").Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an instance of the default event loop for each test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def anyio_backend():
    """Use asyncio backend for anyio."""
    return 'asyncio'


@pytest.fixture(autouse=True)
def mock_settings():
    """Mock settings for all tests to avoid env dependency."""
    with patch('src.config.Settings') as mock:
        mock.return_value.openrouter_api_key = "test-key"
        mock.return_value.database_url = "sqlite+aiosqlite:///./test.db"
        mock.return_value.default_model = "openai/gpt-4-turbo"
        mock.return_value.summarization_model = "openai/gpt-4-turbo"
        mock.return_value.summarization_message_threshold = 10
        mock.return_value.max_context_messages = 20
        mock.return_value.max_context_tokens = 8000
        mock.return_value.log_level = "DEBUG"
        mock.return_value.log_format = "console"
        mock.return_value.env = "test"
        mock.return_value.port = 8000
        mock.return_value.db_pool_size = 5
        mock.return_value.db_max_overflow = 2
        mock.return_value.openrouter_base_url = "https://openrouter.ai/api/v1"
        mock.return_value.openrouter_timeout = 30
        yield mock

