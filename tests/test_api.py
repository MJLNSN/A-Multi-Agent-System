"""
Integration tests for API endpoints.
"""

import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from uuid import uuid4


# Mock the database before importing app
@pytest.fixture(autouse=True)
def mock_db():
    """Mock database for all tests."""
    with patch('src.database.init_db', new_callable=AsyncMock):
        with patch('src.database.close_db', new_callable=AsyncMock):
            yield


@pytest.fixture
def client(mock_db):
    """Create a test client."""
    from src.main import app
    
    # Mock all state objects
    app.state.openrouter = AsyncMock()
    app.state.openrouter.close = AsyncMock()
    
    app.state.thread_manager = AsyncMock()
    app.state.summarizer = AsyncMock()
    app.state.llm_orchestrator = AsyncMock()
    app.state.message_handler = AsyncMock()
    
    return TestClient(app)


class TestRootEndpoints:
    """Test root and health endpoints."""
    
    def test_root(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "docs" in data
    
    def test_health(self, client):
        """Test health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_models(self, client):
        """Test models endpoint lists available models."""
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert len(data["models"]) >= 2


class TestThreadEndpoints:
    """Test thread API endpoints."""
    
    def test_create_thread(self, client):
        """Test creating a thread."""
        thread_id = uuid4()
        client.app.state.thread_manager.create_thread = AsyncMock(return_value={
            "thread_id": thread_id,
            "title": "Test Thread",
            "system_prompt": "Test prompt",
            "current_model": "openai/gpt-4-turbo",
            "message_count": 0,
            "status": "active",
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z"
        })
        
        response = client.post("/api/threads", json={
            "title": "Test Thread",
            "system_prompt": "Test prompt",
            "current_model": "openai/gpt-4-turbo"
        })
        
        assert response.status_code == 201
        data = response.json()
        assert data["title"] == "Test Thread"
    
    def test_get_thread_not_found(self, client):
        """Test getting a non-existent thread."""
        client.app.state.thread_manager.get_thread = AsyncMock(return_value=None)
        
        response = client.get(f"/api/threads/{uuid4()}")
        assert response.status_code == 404
    
    def test_list_threads(self, client):
        """Test listing threads."""
        client.app.state.thread_manager.list_threads = AsyncMock(return_value={
            "threads": [],
            "total": 0,
            "page": 1,
            "limit": 20
        })
        
        response = client.get("/api/threads")
        assert response.status_code == 200
        data = response.json()
        assert "threads" in data
        assert "total" in data


class TestMessageEndpoints:
    """Test message API endpoints."""
    
    def test_send_message_thread_not_found(self, client):
        """Test sending message to non-existent thread."""
        client.app.state.thread_manager.get_thread = AsyncMock(return_value=None)
        
        response = client.post(f"/api/threads/{uuid4()}/messages", json={
            "content": "Hello!"
        })
        assert response.status_code == 404
    
    def test_send_message_success(self, client):
        """Test successfully sending a message."""
        thread_id = uuid4()
        message_id = uuid4()
        
        client.app.state.thread_manager.get_thread = AsyncMock(return_value={
            "thread_id": thread_id,
            "current_model": "openai/gpt-4-turbo"
        })
        
        client.app.state.message_handler.process_user_message = AsyncMock(return_value={
            "message_id": message_id,
            "thread_id": thread_id,
            "role": "assistant",
            "content": "Hello! How can I help?",
            "model": "openai/gpt-4-turbo",
            "tokens": 50
        })
        
        response = client.post(f"/api/threads/{thread_id}/messages", json={
            "content": "Hello!"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["role"] == "assistant"
        assert data["content"] == "Hello! How can I help?"

