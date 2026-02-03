"""
Database module for the Multi-Agent Chat Threading System.
Handles PostgreSQL connection pool and async session management.
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, String, Text, Integer, DateTime, ForeignKey, CheckConstraint, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.sql import func
import uuid
from typing import AsyncGenerator

from src.config import settings

# Create async engine with connection pooling
engine = create_async_engine(
    settings.database_url,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    echo=settings.env == "development",
)

# Session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Base class for ORM models
Base = declarative_base()


class Thread(Base):
    """
    Thread model representing a conversation thread.
    Each thread has a system prompt and tracks message count.
    """
    __tablename__ = "threads"
    
    thread_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), nullable=False, index=True)
    title = Column(String(500), nullable=True)
    system_prompt = Column(Text, nullable=False)
    current_model = Column(String(100), nullable=False)
    message_count = Column(Integer, default=0)
    status = Column(
        String(20), 
        default="active",
        nullable=False
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        CheckConstraint(status.in_(["active", "archived", "deleted"]), name="check_status"),
        Index("idx_user_threads", "user_id", "updated_at"),
    )


class Message(Base):
    """
    Message model representing a single message in a thread.
    Can be from user or assistant, with optional model specification.
    """
    __tablename__ = "messages"
    
    message_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    thread_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("threads.thread_id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    model = Column(String(100), nullable=True)  # Null means use thread.current_model
    tokens = Column(Integer, nullable=True)
    extra_data = Column(JSONB, nullable=True)  # Additional metadata (usage stats, etc.)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        CheckConstraint(role.in_(["user", "assistant", "system"]), name="check_role"),
        Index("idx_thread_messages", "thread_id", "created_at"),
    )


class Summary(Base):
    """
    Summary model representing a compressed summary of conversation segments.
    Tracks which messages are covered by the summary.
    """
    __tablename__ = "summaries"
    
    summary_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    thread_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("threads.thread_id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    summary_text = Column(Text, nullable=False)
    covered_message_count = Column(Integer, nullable=False)
    covered_message_ids = Column(ARRAY(UUID(as_uuid=True)), nullable=True)
    trigger_reason = Column(String(50), nullable=True)
    extra_data = Column(JSONB, nullable=True)  # Additional metadata (model, tokens, etc.)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        CheckConstraint(
            trigger_reason.in_(["message_count", "token_threshold", "manual"]), 
            name="check_trigger_reason"
        ),
        Index("idx_thread_summaries", "thread_id", "created_at"),
    )


async def init_db():
    """
    Initialize the database by creating all tables.
    Should be called on application startup.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db():
    """
    Close the database engine and cleanup connections.
    Should be called on application shutdown.
    """
    await engine.dispose()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency that provides an async database session.
    Automatically handles commit/rollback on context exit.
    """
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

