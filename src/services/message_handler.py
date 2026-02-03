"""
Message Handler Service for the Multi-Agent Chat Threading System.
Coordinates message processing, context assembly, and LLM interaction.
Includes token usage tracking for cost monitoring.
"""

import asyncio
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from uuid import UUID, uuid4
from datetime import datetime
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import Message, Thread, Summary, async_session_maker
from src.services.llm_orchestrator import LLMOrchestrator
from src.services.summarization_engine import SummarizationEngine
from src.utils.token_counter import token_counter
from src.config import settings
from src.constants import CONTEXT_SUMMARY_PREFIX
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.services.usage_tracker import UsageTracker

logger = get_logger("message_handler")


class MessageHandler:
    """
    Handler for processing user messages and generating responses.
    Manages context assembly, summarization triggers, and thread-level concurrency.
    Tracks token usage for cost monitoring.
    """
    
    def __init__(
        self,
        llm_orchestrator: LLMOrchestrator,
        summarization_engine: SummarizationEngine,
        usage_tracker: Optional["UsageTracker"] = None
    ):
        """
        Initialize the Message Handler.
        
        Args:
            llm_orchestrator: LLM orchestrator for generating responses
            summarization_engine: Engine for generating summaries
            usage_tracker: Optional usage tracker for cost monitoring
        """
        self.llm = llm_orchestrator
        self.summarizer = summarization_engine
        self.usage_tracker = usage_tracker
        
        # Thread-level locks to prevent concurrent message processing
        self._locks: Dict[str, asyncio.Lock] = {}
        self._lock_lock = asyncio.Lock()
    
    async def _get_thread_lock(self, thread_id: str) -> asyncio.Lock:
        """
        Get or create a lock for a specific thread.
        Ensures sequential message processing within a thread.
        
        Args:
            thread_id: Thread identifier
        
        Returns:
            asyncio.Lock for the thread
        """
        async with self._lock_lock:
            if thread_id not in self._locks:
                self._locks[thread_id] = asyncio.Lock()
            return self._locks[thread_id]
    
    async def process_user_message(
        self,
        thread_id: str,
        content: str,
        requested_model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a user message and generate an assistant response.
        
        Flow:
        1. Acquire thread lock (concurrency control)
        2. Save user message
        3. Increment thread message count
        4. Check summarization trigger (async, non-blocking)
        5. Assemble context
        6. Generate LLM response
        7. Save assistant message
        8. Increment thread message count
        
        Args:
            thread_id: Thread UUID string
            content: User message content
            requested_model: Optional model override
        
        Returns:
            Dict with assistant message details
        
        Raises:
            ValueError: If thread not found
        """
        # Acquire thread-level lock
        lock = await self._get_thread_lock(thread_id)
        
        async with lock:
            logger.info(
                "processing_message",
                thread_id=thread_id,
                content_length=len(content)
            )
            
            try:
                uuid_id = UUID(thread_id)
            except ValueError:
                raise ValueError(f"Invalid thread ID: {thread_id}")
            
            async with async_session_maker() as session:
                # Get thread info
                result = await session.execute(
                    select(Thread).where(Thread.thread_id == uuid_id)
                )
                thread = result.scalar_one_or_none()
                
                if not thread:
                    raise ValueError(f"Thread not found: {thread_id}")
                
                # 1. Save user message
                user_msg_id = uuid4()
                user_message = Message(
                    message_id=user_msg_id,
                    thread_id=uuid_id,
                    role="user",
                    content=content,
                    model=None,
                    created_at=datetime.utcnow()
                )
                session.add(user_message)
                
                # 2. Increment message count
                thread.message_count += 1
                thread.updated_at = datetime.utcnow()
                
                await session.commit()
                
                logger.info(
                    "user_message_saved",
                    thread_id=thread_id,
                    message_id=str(user_msg_id)
                )
            
            # 3. Check and trigger summarization (async, non-blocking)
            if await self.summarizer.should_trigger(thread_id):
                asyncio.create_task(self._generate_summary_async(thread_id))
            
            # 4. Assemble context
            context = await self._assemble_context(thread_id)
            
            # 5. Trim context if needed
            context = token_counter.trim_context_to_fit(
                context,
                model=requested_model or thread.current_model,
                max_tokens=settings.max_context_tokens
            )
            
            # 6. Generate LLM response
            response = await self.llm.generate_response(
                messages=context,
                thread_current_model=thread.current_model,
                requested_model=requested_model
            )
            
            # 6.5 Track token usage
            if self.usage_tracker and response.get("usage"):
                usage_data = response.get("usage", {})
                await self.usage_tracker.track_usage(
                    model=response["model"],
                    input_tokens=usage_data.get("prompt_tokens", 0),
                    output_tokens=usage_data.get("completion_tokens", 0),
                    operation_type="message",
                    thread_id=thread_id,
                    user_id=thread.user_id,
                    extra_data={"message_type": "assistant_response"}
                )
            
            # 7. Save assistant message
            async with async_session_maker() as session:
                result = await session.execute(
                    select(Thread).where(Thread.thread_id == uuid_id)
                )
                thread = result.scalar_one_or_none()
                
                assistant_msg_id = uuid4()
                assistant_message = Message(
                    message_id=assistant_msg_id,
                    thread_id=uuid_id,
                    role="assistant",
                    content=response["content"],
                    model=response["model"],
                    tokens=response.get("tokens"),
                    extra_data={"usage": response.get("usage", {})},
                    created_at=datetime.utcnow()
                )
                session.add(assistant_message)
                
                # 8. Increment message count again
                thread.message_count += 1
                thread.updated_at = datetime.utcnow()
                
                await session.commit()
                
                logger.info(
                    "assistant_message_saved",
                    thread_id=thread_id,
                    message_id=str(assistant_msg_id),
                    model=response["model"],
                    tokens=response.get("tokens")
                )
            
            return {
                "message_id": assistant_msg_id,
                "thread_id": uuid_id,
                "role": "assistant",
                "content": response["content"],
                "model": response["model"],
                "tokens": response.get("tokens")
            }
    
    async def _generate_summary_async(self, thread_id: str):
        """
        Async wrapper for summary generation with error isolation.
        Runs in background without blocking message processing.
        
        Args:
            thread_id: Thread UUID string
        """
        try:
            await self.summarizer.generate_summary(thread_id)
        except Exception as e:
            # Log error but don't propagate - summary failure shouldn't block messages
            logger.error(
                "summary_generation_error",
                thread_id=thread_id,
                error=str(e)
            )
    
    async def _assemble_context(self, thread_id: str) -> List[Dict[str, str]]:
        """
        Assemble the full context for an LLM call.
        
        Context structure:
        1. System prompt
        2. Latest summary (if exists)
        3. Recent messages (after last summary)
        
        Args:
            thread_id: Thread UUID string
        
        Returns:
            List of message dicts for LLM context
        """
        try:
            uuid_id = UUID(thread_id)
        except ValueError:
            return []
        
        context = []
        
        async with async_session_maker() as session:
            # 1. Get thread system prompt
            result = await session.execute(
                select(Thread).where(Thread.thread_id == uuid_id)
            )
            thread = result.scalar_one_or_none()
            
            if not thread:
                return []
            
            # Add system prompt
            context.append({
                "role": "system",
                "content": thread.system_prompt
            })
            
            # 2. Get latest summary
            summary_result = await session.execute(
                select(Summary)
                .where(Summary.thread_id == uuid_id)
                .order_by(Summary.created_at.desc())
                .limit(1)
            )
            latest_summary = summary_result.scalar_one_or_none()
            
            last_summary_time = None
            if latest_summary:
                # Add summary as context
                context.append({
                    "role": "assistant",
                    "content": f"{CONTEXT_SUMMARY_PREFIX}{latest_summary.summary_text}"
                })
                last_summary_time = latest_summary.created_at
            
            # 3. Get recent messages (after last summary to avoid duplication)
            if last_summary_time:
                # Only get messages after the summary was created
                messages_result = await session.execute(
                    select(Message)
                    .where(Message.thread_id == uuid_id)
                    .where(Message.created_at > last_summary_time)
                    .order_by(Message.created_at.asc())
                    .limit(settings.max_context_messages)
                )
            else:
                # No summary, get most recent messages
                messages_result = await session.execute(
                    select(Message)
                    .where(Message.thread_id == uuid_id)
                    .order_by(Message.created_at.desc())
                    .limit(settings.max_context_messages)
                )
            
            messages = messages_result.scalars().all()
            
            # If we did a desc query, reverse to chronological order
            if not last_summary_time:
                messages = list(reversed(messages))
            
            # Add messages to context
            for msg in messages:
                context.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        return context
    
    async def get_thread_messages(
        self,
        thread_id: str,
        limit: int = 20,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Get message history for a thread.
        
        Args:
            thread_id: Thread UUID string
            limit: Maximum messages to return
            offset: Pagination offset
        
        Returns:
            Dict with messages list and total count
        """
        try:
            uuid_id = UUID(thread_id)
        except ValueError:
            return {"messages": [], "total": 0}
        
        async with async_session_maker() as session:
            # Get total count
            from sqlalchemy import func
            count_result = await session.execute(
                select(func.count(Message.message_id))
                .where(Message.thread_id == uuid_id)
            )
            total = count_result.scalar()
            
            # Get paginated messages
            result = await session.execute(
                select(Message)
                .where(Message.thread_id == uuid_id)
                .order_by(Message.created_at.asc())
                .limit(limit)
                .offset(offset)
            )
            messages = result.scalars().all()
            
            return {
                "messages": [
                    {
                        "message_id": msg.message_id,
                        "thread_id": msg.thread_id,
                        "role": msg.role,
                        "content": msg.content,
                        "model": msg.model,
                        "tokens": msg.tokens,
                        "created_at": msg.created_at
                    }
                    for msg in messages
                ],
                "total": total
            }

