"""
Thread routes for the Multi-Agent Chat Threading System.
Provides endpoints for thread CRUD operations.
"""

from fastapi import APIRouter, HTTPException, Request, Query
from typing import Optional

from src.models.schemas import (
    ThreadCreate, ThreadUpdate, ThreadResponse, ThreadListResponse
)
from src.utils.logging import get_logger

logger = get_logger("threads_route")
router = APIRouter(tags=["Threads"])


@router.post("/threads", response_model=ThreadResponse, status_code=201)
async def create_thread(request: ThreadCreate, req: Request):
    """
    Create a new conversation thread.
    
    Each thread has:
    - A system prompt that defines the assistant's behavior
    - A default model that can be overridden per message
    - Persistent message history
    """
    try:
        thread = await req.app.state.thread_manager.create_thread(
            system_prompt=request.system_prompt,
            title=request.title,
            current_model=request.current_model,
            user_id=request.user_id
        )
        
        logger.info("thread_created_via_api", thread_id=str(thread["thread_id"]))
        return thread
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("thread_creation_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create thread")


@router.get("/threads/{thread_id}", response_model=ThreadResponse)
async def get_thread(thread_id: str, req: Request):
    """
    Retrieve a specific thread by ID.
    
    Returns thread metadata including:
    - System prompt
    - Current default model
    - Message count
    - Creation and update timestamps
    """
    thread = await req.app.state.thread_manager.get_thread(thread_id)
    
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    return thread


@router.patch("/threads/{thread_id}", response_model=ThreadResponse)
async def update_thread(thread_id: str, request: ThreadUpdate, req: Request):
    """
    Update a thread's configuration.
    
    Can update:
    - title: Thread display name
    - current_model: Default model for new messages
    """
    try:
        thread = await req.app.state.thread_manager.update_thread(
            thread_id=thread_id,
            title=request.title,
            current_model=request.current_model
        )
        
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")
        
        logger.info("thread_updated_via_api", thread_id=thread_id)
        return thread
    
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("thread_update_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to update thread")


@router.get("/threads", response_model=ThreadListResponse)
async def list_threads(
    req: Request,
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    limit: int = Query(20, ge=1, le=100, description="Maximum threads to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    status: str = Query("active", description="Filter by status")
):
    """
    List all threads with pagination.
    
    Supports filtering by:
    - user_id: Specific user's threads
    - status: Thread status (active, archived, deleted)
    
    Results are ordered by most recently updated first.
    """
    result = await req.app.state.thread_manager.list_threads(
        user_id=user_id,
        limit=limit,
        offset=offset,
        status=status
    )
    
    return result

