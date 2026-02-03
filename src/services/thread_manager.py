"""
Thread Manager Service for the Multi-Agent Chat Threading System.
Handles CRUD operations for conversation threads.
"""

from typing import Optional, List, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime
from sqlalchemy import select, update, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import Thread, async_session_maker
from src.models.registry import validate_model
from src.config import settings
from src.utils.logging import get_logger

logger = get_logger("thread_manager")


class ThreadManager:
    """
    Service for managing conversation threads.
    Provides methods for creating, retrieving, updating, and listing threads.
    """
    
    async def create_thread(
        self,
        system_prompt: str,
        title: Optional[str] = None,
        current_model: Optional[str] = None,
        user_id: str = "default_user"
    ) -> Dict[str, Any]:
        """
        Create a new conversation thread.
        
        Args:
            system_prompt: System prompt for the thread
            title: Optional thread title
            current_model: Default model for the thread
            user_id: User identifier
        
        Returns:
            Dict with thread details
        
        Raises:
            ValueError: If model is invalid
        """
        model = current_model or settings.default_model
        
        # Validate model
        if not validate_model(model):
            raise ValueError(f"Invalid model: {model}")
        
        thread_id = uuid4()
        
        async with async_session_maker() as session:
            thread = Thread(
                thread_id=thread_id,
                user_id=user_id,
                title=title,
                system_prompt=system_prompt,
                current_model=model,
                message_count=0,
                status="active"
            )
            
            session.add(thread)
            await session.commit()
            await session.refresh(thread)
            
            logger.info(
                "thread_created",
                thread_id=str(thread_id),
                user_id=user_id,
                model=model
            )
            
            return self._thread_to_dict(thread)
    
    async def get_thread(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a thread by ID.
        
        Args:
            thread_id: Thread UUID string
        
        Returns:
            Thread dict or None if not found
        """
        try:
            uuid_id = UUID(thread_id)
        except ValueError:
            return None
        
        async with async_session_maker() as session:
            result = await session.execute(
                select(Thread).where(Thread.thread_id == uuid_id)
            )
            thread = result.scalar_one_or_none()
            
            if thread:
                return self._thread_to_dict(thread)
            return None
    
    async def update_thread(
        self,
        thread_id: str,
        title: Optional[str] = None,
        current_model: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Update a thread's configuration.
        
        Args:
            thread_id: Thread UUID string
            title: New title (optional)
            current_model: New default model (optional)
        
        Returns:
            Updated thread dict or None if not found
        
        Raises:
            ValueError: If model is invalid
        """
        try:
            uuid_id = UUID(thread_id)
        except ValueError:
            return None
        
        # Validate model if provided
        if current_model and not validate_model(current_model):
            raise ValueError(f"Invalid model: {current_model}")
        
        async with async_session_maker() as session:
            # Build update values
            update_values = {"updated_at": datetime.utcnow()}
            if title is not None:
                update_values["title"] = title
            if current_model is not None:
                update_values["current_model"] = current_model
            
            # Execute update
            await session.execute(
                update(Thread)
                .where(Thread.thread_id == uuid_id)
                .values(**update_values)
            )
            await session.commit()
            
            # Fetch updated thread
            result = await session.execute(
                select(Thread).where(Thread.thread_id == uuid_id)
            )
            thread = result.scalar_one_or_none()
            
            if thread:
                logger.info(
                    "thread_updated",
                    thread_id=thread_id,
                    updates=list(update_values.keys())
                )
                return self._thread_to_dict(thread)
            return None
    
    async def list_threads(
        self,
        user_id: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
        status: str = "active"
    ) -> Dict[str, Any]:
        """
        List threads with pagination.
        
        Args:
            user_id: Filter by user (optional)
            limit: Maximum number of threads to return
            offset: Pagination offset
            status: Filter by status
        
        Returns:
            Dict with threads list, total count, and pagination info
        """
        async with async_session_maker() as session:
            # Build base query
            query = select(Thread).where(Thread.status == status)
            count_query = select(func.count(Thread.thread_id)).where(Thread.status == status)
            
            if user_id:
                query = query.where(Thread.user_id == user_id)
                count_query = count_query.where(Thread.user_id == user_id)
            
            # Get total count
            total_result = await session.execute(count_query)
            total = total_result.scalar()
            
            # Get paginated results
            query = query.order_by(Thread.updated_at.desc()).limit(limit).offset(offset)
            result = await session.execute(query)
            threads = result.scalars().all()
            
            return {
                "threads": [self._thread_to_dict(t) for t in threads],
                "total": total,
                "page": offset // limit + 1,
                "limit": limit
            }
    
    async def increment_message_count(self, thread_id: str) -> bool:
        """
        Increment the message count for a thread.
        
        Args:
            thread_id: Thread UUID string
        
        Returns:
            True if successful, False if thread not found
        """
        try:
            uuid_id = UUID(thread_id)
        except ValueError:
            return False
        
        async with async_session_maker() as session:
            await session.execute(
                update(Thread)
                .where(Thread.thread_id == uuid_id)
                .values(
                    message_count=Thread.message_count + 1,
                    updated_at=datetime.utcnow()
                )
            )
            await session.commit()
            return True
    
    def _thread_to_dict(self, thread: Thread) -> Dict[str, Any]:
        """Convert Thread ORM object to dictionary."""
        return {
            "thread_id": thread.thread_id,
            "user_id": thread.user_id,
            "title": thread.title,
            "system_prompt": thread.system_prompt,
            "current_model": thread.current_model,
            "message_count": thread.message_count,
            "status": thread.status,
            "created_at": thread.created_at,
            "updated_at": thread.updated_at
        }

