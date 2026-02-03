"""
Pydantic schemas for request/response validation.
Defines all API data models.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from uuid import UUID


# ============== Thread Schemas ==============

class ThreadCreate(BaseModel):
    """Request schema for creating a new thread."""
    title: Optional[str] = Field(None, max_length=500, description="Thread title")
    system_prompt: str = Field(..., description="System prompt for the thread")
    current_model: Optional[str] = Field(
        "openai/gpt-4-turbo", 
        description="Default model for the thread"
    )
    user_id: str = Field("default_user", description="User identifier")


class ThreadUpdate(BaseModel):
    """Request schema for updating a thread."""
    title: Optional[str] = Field(None, max_length=500, description="New thread title")
    current_model: Optional[str] = Field(None, description="New default model")


class ThreadResponse(BaseModel):
    """Response schema for a single thread."""
    thread_id: UUID
    title: Optional[str]
    system_prompt: str
    current_model: str
    message_count: int
    status: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class ThreadListResponse(BaseModel):
    """Response schema for listing threads."""
    threads: List[ThreadResponse]
    total: int
    page: int
    limit: int


# ============== Message Schemas ==============

class MessageCreate(BaseModel):
    """Request schema for sending a message."""
    content: str = Field(..., min_length=1, description="Message content")
    model: Optional[str] = Field(
        None, 
        description="Model to use for this message (overrides thread default)"
    )


class MessageResponse(BaseModel):
    """Response schema for a single message."""
    message_id: UUID
    thread_id: UUID
    role: str
    content: str
    model: Optional[str]
    tokens: Optional[int]
    created_at: datetime
    
    class Config:
        from_attributes = True


class AssistantMessageResponse(BaseModel):
    """Response schema for assistant's response to a message."""
    message_id: UUID
    thread_id: UUID
    role: str = "assistant"
    content: str
    model: str
    tokens: Optional[int]


class MessageListResponse(BaseModel):
    """Response schema for listing messages."""
    messages: List[MessageResponse]
    total: int


# ============== Summary Schemas ==============

class SummaryResponse(BaseModel):
    """Response schema for a single summary."""
    summary_id: UUID
    thread_id: UUID
    summary_text: str
    covered_message_count: int
    trigger_reason: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True


class SummaryListResponse(BaseModel):
    """Response schema for listing summaries."""
    summaries: List[SummaryResponse]
    total: int


# ============== Health/Info Schemas ==============

class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str
    database: str = "connected"
    version: str = "1.0.0"


class RootResponse(BaseModel):
    """Response schema for root endpoint."""
    message: str
    docs: str
    version: str

