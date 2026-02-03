"""
Summarization Engine for the Multi-Agent Chat Threading System.
Handles automatic conversation summarization to compress context.
"""

from typing import List, Dict, Any, Optional
from uuid import UUID, uuid4
from datetime import datetime
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import Message, Summary, Thread, async_session_maker
from src.adapters.openrouter import OpenRouterAdapter
from src.config import settings
from src.constants import (
    SUMMARIZATION_PROMPT_TEMPLATE,
    SUMMARY_TEMPERATURE,
    SUMMARY_MAX_TOKENS,
    TRIGGER_MESSAGE_COUNT
)
from src.utils.logging import get_logger

logger = get_logger("summarization_engine")


class SummarizationEngine:
    """
    Engine for generating conversation summaries.
    Automatically triggered based on message count threshold.
    """
    
    def __init__(self, openrouter_adapter: OpenRouterAdapter):
        """
        Initialize the Summarization Engine.
        
        Args:
            openrouter_adapter: OpenRouter API adapter for LLM calls
        """
        self.adapter = openrouter_adapter
        self.threshold = settings.summarization_message_threshold
        self.model = settings.summarization_model
    
    async def should_trigger(self, thread_id: str) -> bool:
        """
        Check if summarization should be triggered for a thread.
        
        Triggers when message_count is a multiple of threshold (default: 10).
        
        Args:
            thread_id: Thread UUID string
        
        Returns:
            True if summarization should be triggered
        """
        try:
            uuid_id = UUID(thread_id)
        except ValueError:
            return False
        
        async with async_session_maker() as session:
            result = await session.execute(
                select(Thread.message_count).where(Thread.thread_id == uuid_id)
            )
            count = result.scalar_one_or_none()
            
            if count and count > 0 and count % self.threshold == 0:
                logger.info(
                    "summarization_triggered",
                    thread_id=thread_id,
                    message_count=count
                )
                return True
            
            return False
    
    async def generate_summary(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """
        Generate a summary for the most recent conversation segment.
        
        Args:
            thread_id: Thread UUID string
        
        Returns:
            Summary dict or None if generation fails
        """
        try:
            uuid_id = UUID(thread_id)
        except ValueError:
            logger.error("invalid_thread_id", thread_id=thread_id)
            return None
        
        async with async_session_maker() as session:
            # Get the last N messages
            result = await session.execute(
                select(Message)
                .where(Message.thread_id == uuid_id)
                .order_by(Message.created_at.desc())
                .limit(self.threshold)
            )
            messages = result.scalars().all()
            
            if not messages:
                logger.warning("no_messages_to_summarize", thread_id=thread_id)
                return None
            
            # Reverse to chronological order
            messages = list(reversed(messages))
            
            # Format messages for summarization
            conversation_text = self._format_messages_for_summary(messages)
            
            # Generate summary via LLM
            prompt = SUMMARIZATION_PROMPT_TEMPLATE.format(conversation=conversation_text)
            
            try:
                response = await self.adapter.chat_completion(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=SUMMARY_TEMPERATURE,
                    max_tokens=SUMMARY_MAX_TOKENS
                )
                
                summary_text = response["content"].strip()
                
            except Exception as e:
                logger.error(
                    "summary_generation_failed",
                    thread_id=thread_id,
                    error=str(e)
                )
                return None
            
            # Store summary in database
            summary_id = uuid4()
            summary = Summary(
                summary_id=summary_id,
                thread_id=uuid_id,
                summary_text=summary_text,
                covered_message_count=len(messages),
                covered_message_ids=[m.message_id for m in messages],
                trigger_reason=TRIGGER_MESSAGE_COUNT,
                extra_data={
                    "model": self.model,
                    "tokens": response.get("tokens", 0)
                }
            )
            
            session.add(summary)
            await session.commit()
            
            logger.info(
                "summary_generated",
                thread_id=thread_id,
                summary_id=str(summary_id),
                message_count=len(messages),
                tokens=response.get("tokens", 0)
            )
            
            return {
                "summary_id": summary_id,
                "thread_id": uuid_id,
                "summary_text": summary_text,
                "covered_message_count": len(messages),
                "trigger_reason": "message_count",
                "created_at": datetime.utcnow()
            }
    
    async def get_latest_summary(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the most recent summary for a thread.
        
        Args:
            thread_id: Thread UUID string
        
        Returns:
            Summary dict or None if no summary exists
        """
        try:
            uuid_id = UUID(thread_id)
        except ValueError:
            return None
        
        async with async_session_maker() as session:
            result = await session.execute(
                select(Summary)
                .where(Summary.thread_id == uuid_id)
                .order_by(Summary.created_at.desc())
                .limit(1)
            )
            summary = result.scalar_one_or_none()
            
            if summary:
                return {
                    "summary_id": summary.summary_id,
                    "thread_id": summary.thread_id,
                    "summary_text": summary.summary_text,
                    "covered_message_count": summary.covered_message_count,
                    "trigger_reason": summary.trigger_reason,
                    "created_at": summary.created_at
                }
            
            return None
    
    async def get_thread_summaries(
        self,
        thread_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get all summaries for a thread.
        
        Args:
            thread_id: Thread UUID string
            limit: Maximum number of summaries to return
        
        Returns:
            List of summary dicts
        """
        try:
            uuid_id = UUID(thread_id)
        except ValueError:
            return []
        
        async with async_session_maker() as session:
            result = await session.execute(
                select(Summary)
                .where(Summary.thread_id == uuid_id)
                .order_by(Summary.created_at.desc())
                .limit(limit)
            )
            summaries = result.scalars().all()
            
            return [
                {
                    "summary_id": s.summary_id,
                    "thread_id": s.thread_id,
                    "summary_text": s.summary_text,
                    "covered_message_count": s.covered_message_count,
                    "trigger_reason": s.trigger_reason,
                    "created_at": s.created_at
                }
                for s in summaries
            ]
    
    def _format_messages_for_summary(self, messages: List[Message]) -> str:
        """
        Format messages into text for summarization.
        
        Args:
            messages: List of Message objects
        
        Returns:
            Formatted conversation text
        """
        lines = []
        for msg in messages:
            role = "User" if msg.role == "user" else "Assistant"
            # Truncate long messages for summary
            content = msg.content[:500] + "..." if len(msg.content) > 500 else msg.content
            lines.append(f"{role}: {content}")
        
        return "\n".join(lines)

