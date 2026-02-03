"""
Message routes for the Multi-Agent Chat Threading System.
Provides endpoints for sending messages and retrieving history.
"""

from fastapi import APIRouter, HTTPException, Request, Query

from src.models.schemas import (
    MessageCreate, AssistantMessageResponse, MessageListResponse
)
from src.utils.logging import get_logger

logger = get_logger("messages_route")
router = APIRouter(tags=["Messages"])


@router.post(
    "/threads/{thread_id}/messages", 
    response_model=AssistantMessageResponse
)
async def send_message(thread_id: str, request: MessageCreate, req: Request):
    """
    Send a user message and receive an assistant response.
    
    The message processing flow:
    1. Save user message to database
    2. Check if summarization should be triggered
    3. Assemble context (system prompt + summary + recent messages)
    4. Call LLM via OpenRouter
    5. Save assistant response to database
    
    Model selection:
    - If 'model' is provided in request, use that model
    - Otherwise, use the thread's default model (current_model)
    
    This enables real-time model switching within a conversation.
    """
    try:
        # Verify thread exists first
        thread = await req.app.state.thread_manager.get_thread(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")
        
        # Process message
        response = await req.app.state.message_handler.process_user_message(
            thread_id=thread_id,
            content=request.content,
            requested_model=request.model
        )
        
        logger.info(
            "message_processed_via_api",
            thread_id=thread_id,
            model=response["model"]
        )
        
        return response
    
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(
            "message_processing_failed",
            thread_id=thread_id,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/threads/{thread_id}/messages", 
    response_model=MessageListResponse
)
async def get_messages(
    thread_id: str,
    req: Request,
    limit: int = Query(20, ge=1, le=100, description="Maximum messages to return"),
    offset: int = Query(0, ge=0, description="Pagination offset")
):
    """
    Retrieve message history for a thread.
    
    Messages are returned in chronological order (oldest first).
    Each message includes:
    - role: 'user' or 'assistant'
    - content: Message text
    - model: Model used (for assistant messages)
    - tokens: Token count (for assistant messages)
    - created_at: Timestamp
    """
    # Verify thread exists
    thread = await req.app.state.thread_manager.get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    result = await req.app.state.message_handler.get_thread_messages(
        thread_id=thread_id,
        limit=limit,
        offset=offset
    )
    
    return result

