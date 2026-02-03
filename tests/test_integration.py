"""
Integration tests for the Multi-Agent Chat Threading System.
Tests the complete flow from API to database.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from uuid import uuid4
from datetime import datetime


# Fixtures for mocking
@pytest.fixture(autouse=True)
def mock_database():
    """Mock database for all integration tests."""
    with patch('src.database.init_db', new_callable=AsyncMock):
        with patch('src.database.close_db', new_callable=AsyncMock):
            with patch('src.database.async_session_maker') as mock_session:
                mock_ctx = AsyncMock()
                mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
                mock_session.return_value.__aexit__ = AsyncMock(return_value=None)
                mock_ctx.add = AsyncMock()
                mock_ctx.commit = AsyncMock()
                mock_ctx.refresh = AsyncMock()
                mock_ctx.execute = AsyncMock()
                yield mock_ctx


@pytest.fixture
def client(mock_database):
    """Create a test client with mocked services."""
    from src.main import app
    
    # Mock OpenRouter adapter
    mock_adapter = AsyncMock()
    mock_adapter.close = AsyncMock()
    mock_adapter.chat_completion = AsyncMock(return_value={
        "content": "Hello! How can I help you?",
        "model": "openai/gpt-4-turbo",
        "tokens": 50,
        "finish_reason": "stop",
        "usage": {"total_tokens": 100}
    })
    
    app.state.openrouter = mock_adapter
    
    # Mock services
    app.state.thread_manager = AsyncMock()
    app.state.summarizer = AsyncMock()
    app.state.llm_orchestrator = AsyncMock()
    app.state.message_handler = AsyncMock()
    
    return TestClient(app)


class TestAPIIntegration:
    """Integration tests for API endpoints."""
    
    def test_complete_thread_flow(self, client):
        """Test complete flow: create thread -> send message -> get history."""
        thread_id = uuid4()
        message_id = uuid4()
        
        # Mock thread creation
        client.app.state.thread_manager.create_thread = AsyncMock(return_value={
            "thread_id": thread_id,
            "title": "Test Thread",
            "system_prompt": "You are helpful.",
            "current_model": "openai/gpt-4-turbo",
            "message_count": 0,
            "status": "active",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        })
        
        # 1. Create thread
        response = client.post("/api/threads", json={
            "title": "Test Thread",
            "system_prompt": "You are helpful.",
            "current_model": "openai/gpt-4-turbo"
        })
        assert response.status_code == 201
        created_thread = response.json()
        assert created_thread["title"] == "Test Thread"
        
        # Mock for get thread
        client.app.state.thread_manager.get_thread = AsyncMock(return_value={
            "thread_id": thread_id,
            "title": "Test Thread",
            "system_prompt": "You are helpful.",
            "current_model": "openai/gpt-4-turbo",
            "message_count": 0,
            "status": "active",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        })
        
        # 2. Send message
        client.app.state.message_handler.process_user_message = AsyncMock(return_value={
            "message_id": message_id,
            "thread_id": thread_id,
            "role": "assistant",
            "content": "Hello! How can I help you today?",
            "model": "openai/gpt-4-turbo",
            "tokens": 50
        })
        
        response = client.post(f"/api/threads/{thread_id}/messages", json={
            "content": "Hello!"
        })
        assert response.status_code == 200
        message = response.json()
        assert message["role"] == "assistant"
        assert message["content"] == "Hello! How can I help you today?"
        
        # 3. Get message history
        client.app.state.message_handler.get_thread_messages = AsyncMock(return_value={
            "messages": [
                {
                    "message_id": uuid4(),
                    "thread_id": thread_id,
                    "role": "user",
                    "content": "Hello!",
                    "model": None,
                    "tokens": None,
                    "created_at": datetime.utcnow()
                },
                {
                    "message_id": message_id,
                    "thread_id": thread_id,
                    "role": "assistant",
                    "content": "Hello! How can I help you today?",
                    "model": "openai/gpt-4-turbo",
                    "tokens": 50,
                    "created_at": datetime.utcnow()
                }
            ],
            "total": 2
        })
        
        response = client.get(f"/api/threads/{thread_id}/messages")
        assert response.status_code == 200
        history = response.json()
        assert history["total"] == 2
        assert len(history["messages"]) == 2
    
    def test_model_switching_flow(self, client):
        """Test model switching within a conversation."""
        thread_id = uuid4()
        
        # Setup mocks
        client.app.state.thread_manager.get_thread = AsyncMock(return_value={
            "thread_id": thread_id,
            "current_model": "openai/gpt-4-turbo",
            "message_count": 5,
            "status": "active",
            "title": "Test",
            "system_prompt": "Test",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        })
        
        # First message with default model
        client.app.state.message_handler.process_user_message = AsyncMock(return_value={
            "message_id": uuid4(),
            "thread_id": thread_id,
            "role": "assistant",
            "content": "Response from GPT-4",
            "model": "openai/gpt-4-turbo",
            "tokens": 30
        })
        
        response = client.post(f"/api/threads/{thread_id}/messages", json={
            "content": "First message"
        })
        assert response.status_code == 200
        assert response.json()["model"] == "openai/gpt-4-turbo"
        
        # Second message with Claude (override)
        client.app.state.message_handler.process_user_message = AsyncMock(return_value={
            "message_id": uuid4(),
            "thread_id": thread_id,
            "role": "assistant",
            "content": "Response from Claude",
            "model": "anthropic/claude-3.5-sonnet",
            "tokens": 40
        })
        
        response = client.post(f"/api/threads/{thread_id}/messages", json={
            "content": "Second message",
            "model": "anthropic/claude-3.5-sonnet"
        })
        assert response.status_code == 200
        assert response.json()["model"] == "anthropic/claude-3.5-sonnet"
    
    def test_summarization_flow(self, client):
        """Test summarization retrieval."""
        thread_id = uuid4()
        
        client.app.state.thread_manager.get_thread = AsyncMock(return_value={
            "thread_id": thread_id,
            "message_count": 15,
            "status": "active",
            "title": "Test",
            "system_prompt": "Test",
            "current_model": "openai/gpt-4-turbo",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        })
        
        client.app.state.summarizer.get_thread_summaries = AsyncMock(return_value=[
            {
                "summary_id": uuid4(),
                "thread_id": thread_id,
                "summary_text": "Discussion about product strategy...",
                "covered_message_count": 10,
                "trigger_reason": "message_count",
                "created_at": datetime.utcnow()
            }
        ])
        
        response = client.get(f"/api/threads/{thread_id}/summaries")
        assert response.status_code == 200
        summaries = response.json()
        assert len(summaries["summaries"]) == 1
        assert "product strategy" in summaries["summaries"][0]["summary_text"]
    
    def test_thread_update_model(self, client):
        """Test updating thread's default model."""
        thread_id = uuid4()
        
        client.app.state.thread_manager.update_thread = AsyncMock(return_value={
            "thread_id": thread_id,
            "title": "Updated Title",
            "current_model": "anthropic/claude-3.5-sonnet",
            "message_count": 5,
            "status": "active",
            "system_prompt": "Test",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        })
        
        response = client.patch(f"/api/threads/{thread_id}", json={
            "title": "Updated Title",
            "current_model": "anthropic/claude-3.5-sonnet"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Updated Title"
        assert data["current_model"] == "anthropic/claude-3.5-sonnet"


class TestErrorHandling:
    """Test error handling in API."""
    
    def test_thread_not_found(self, client):
        """Test 404 when thread not found."""
        client.app.state.thread_manager.get_thread = AsyncMock(return_value=None)
        
        response = client.get(f"/api/threads/{uuid4()}")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    def test_invalid_model_in_create(self, client):
        """Test 400 for invalid model in thread creation."""
        client.app.state.thread_manager.create_thread = AsyncMock(
            side_effect=ValueError("Invalid model: fake/model")
        )
        
        response = client.post("/api/threads", json={
            "title": "Test",
            "system_prompt": "Test",
            "current_model": "fake/model"
        })
        
        assert response.status_code == 400
        assert "Invalid model" in response.json()["detail"]
    
    def test_message_to_nonexistent_thread(self, client):
        """Test sending message to non-existent thread."""
        client.app.state.thread_manager.get_thread = AsyncMock(return_value=None)
        
        response = client.post(f"/api/threads/{uuid4()}/messages", json={
            "content": "Hello"
        })
        
        assert response.status_code == 404
    
    def test_update_nonexistent_thread(self, client):
        """Test updating non-existent thread."""
        client.app.state.thread_manager.update_thread = AsyncMock(return_value=None)
        
        response = client.patch(f"/api/threads/{uuid4()}", json={
            "title": "New Title"
        })
        
        assert response.status_code == 404

