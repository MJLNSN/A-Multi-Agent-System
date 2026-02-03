"""
Summary routes for the Multi-Agent Chat Threading System.
Provides endpoints for retrieving conversation summaries.
"""

from fastapi import APIRouter, HTTPException, Request, Query

from src.models.schemas import SummaryListResponse
from src.utils.logging import get_logger

logger = get_logger("summaries_route")
router = APIRouter(tags=["Summaries"])


@router.get(
    "/threads/{thread_id}/summaries", 
    response_model=SummaryListResponse
)
async def get_summaries(
    thread_id: str,
    req: Request,
    limit: int = Query(10, ge=1, le=50, description="Maximum summaries to return")
):
    """
    Retrieve summaries for a thread.
    
    Summaries are automatically generated when the message count
    reaches a threshold (default: every 10 messages).
    
    Each summary includes:
    - summary_text: Compressed summary of conversation segment
    - covered_message_count: Number of messages covered
    - trigger_reason: What triggered the summary (e.g., 'message_count')
    - created_at: When the summary was generated
    
    Summaries are returned in reverse chronological order (newest first).
    """
    # Verify thread exists
    thread = await req.app.state.thread_manager.get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    summaries = await req.app.state.summarizer.get_thread_summaries(
        thread_id=thread_id,
        limit=limit
    )
    
    return {
        "summaries": summaries,
        "total": len(summaries)
    }

