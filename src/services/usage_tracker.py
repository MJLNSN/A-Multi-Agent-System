"""
Token Usage Tracker Service for the Multi-Agent Chat Threading System.

Tracks and aggregates token usage and costs across all LLM operations.
Provides enterprise-grade monitoring and cost analysis capabilities.
"""

from typing import Dict, List, Any, Optional
from uuid import UUID, uuid4
from datetime import datetime, timedelta
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import TokenUsage, Thread, async_session_maker
from src.models.registry import MODEL_REGISTRY, get_model_config
from src.utils.logging import get_logger

logger = get_logger("usage_tracker")


class UsageTracker:
    """
    Service for tracking and analyzing token usage and API costs.
    
    Features:
    - Record token usage for every LLM call
    - Calculate costs based on model pricing
    - Aggregate usage by thread, user, model, and time period
    - Provide cost analysis and reporting
    """
    
    def __init__(self):
        """Initialize the Usage Tracker."""
        self._cost_cache = {}  # Cache model costs
    
    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> Dict[str, float]:
        """
        Calculate cost for a given model and token counts.
        
        Args:
            model: Model identifier (e.g., "openai/gpt-4-turbo")
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        
        Returns:
            Dict with input_cost, output_cost, and total_cost in USD
        """
        config = get_model_config(model)
        
        if config and "cost_per_1k_tokens" in config:
            costs = config["cost_per_1k_tokens"]
            input_cost = (input_tokens / 1000) * costs.get("input", 0)
            output_cost = (output_tokens / 1000) * costs.get("output", 0)
        else:
            # Default pricing if model not in registry
            input_cost = (input_tokens / 1000) * 0.002
            output_cost = (output_tokens / 1000) * 0.006
        
        return {
            "input": round(input_cost, 6),
            "output": round(output_cost, 6),
            "total": round(input_cost + output_cost, 6)
        }
    
    async def track_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        operation_type: str,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
        collaboration_id: Optional[str] = None,
        extra_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Record a token usage event.
        
        Args:
            model: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            operation_type: Type of operation (message, summarization, collaboration)
            thread_id: Optional thread UUID string
            user_id: Optional user identifier
            collaboration_id: Optional collaboration session ID
            extra_data: Additional metadata
        
        Returns:
            Dict with usage record details including cost
        """
        total_tokens = input_tokens + output_tokens
        cost = self.calculate_cost(model, input_tokens, output_tokens)
        
        usage_id = uuid4()
        
        async with async_session_maker() as session:
            usage_record = TokenUsage(
                usage_id=usage_id,
                thread_id=UUID(thread_id) if thread_id else None,
                user_id=user_id,
                collaboration_id=collaboration_id,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cost_usd=cost,
                operation_type=operation_type,
                extra_data=extra_data or {},
                created_at=datetime.utcnow()
            )
            session.add(usage_record)
            await session.commit()
            
            logger.info(
                "usage_tracked",
                usage_id=str(usage_id),
                model=model,
                total_tokens=total_tokens,
                cost_usd=cost["total"],
                operation_type=operation_type
            )
        
        return {
            "usage_id": str(usage_id),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cost_usd": cost,
            "operation_type": operation_type,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def get_usage_summary(
        self,
        days: int = 30,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get overall usage summary for a time period.
        
        Args:
            days: Number of days to look back
            user_id: Optional filter by user
        
        Returns:
            Dict with total tokens, cost breakdown, and model statistics
        """
        start_date = datetime.utcnow() - timedelta(days=days)
        
        async with async_session_maker() as session:
            # Build base query
            query = select(
                func.sum(TokenUsage.input_tokens).label("total_input"),
                func.sum(TokenUsage.output_tokens).label("total_output"),
                func.sum(TokenUsage.total_tokens).label("total_tokens"),
                func.count(TokenUsage.usage_id).label("request_count")
            ).where(TokenUsage.created_at >= start_date)
            
            if user_id:
                query = query.where(TokenUsage.user_id == user_id)
            
            result = await session.execute(query)
            row = result.one()
            
            total_input = row.total_input or 0
            total_output = row.total_output or 0
            total_tokens = row.total_tokens or 0
            request_count = row.request_count or 0
            
            # Get cost breakdown by model
            model_query = select(
                TokenUsage.model,
                func.sum(TokenUsage.input_tokens).label("input_tokens"),
                func.sum(TokenUsage.output_tokens).label("output_tokens"),
                func.sum(TokenUsage.total_tokens).label("total_tokens"),
                func.count(TokenUsage.usage_id).label("request_count")
            ).where(
                TokenUsage.created_at >= start_date
            ).group_by(TokenUsage.model)
            
            if user_id:
                model_query = model_query.where(TokenUsage.user_id == user_id)
            
            model_result = await session.execute(model_query)
            model_rows = model_result.all()
            
            model_breakdown = []
            total_cost = 0.0
            
            for model_row in model_rows:
                cost = self.calculate_cost(
                    model_row.model,
                    model_row.input_tokens or 0,
                    model_row.output_tokens or 0
                )
                total_cost += cost["total"]
                
                model_breakdown.append({
                    "model": model_row.model,
                    "input_tokens": model_row.input_tokens or 0,
                    "output_tokens": model_row.output_tokens or 0,
                    "total_tokens": model_row.total_tokens or 0,
                    "request_count": model_row.request_count or 0,
                    "cost_usd": cost
                })
            
            # Get operation breakdown
            op_query = select(
                TokenUsage.operation_type,
                func.sum(TokenUsage.total_tokens).label("total_tokens"),
                func.count(TokenUsage.usage_id).label("request_count")
            ).where(
                TokenUsage.created_at >= start_date
            ).group_by(TokenUsage.operation_type)
            
            if user_id:
                op_query = op_query.where(TokenUsage.user_id == user_id)
            
            op_result = await session.execute(op_query)
            op_rows = op_result.all()
            
            operation_breakdown = [
                {
                    "operation": row.operation_type,
                    "total_tokens": row.total_tokens or 0,
                    "request_count": row.request_count or 0
                }
                for row in op_rows
            ]
        
        return {
            "period_days": days,
            "summary": {
                "total_input_tokens": total_input,
                "total_output_tokens": total_output,
                "total_tokens": total_tokens,
                "total_requests": request_count,
                "total_cost_usd": round(total_cost, 4)
            },
            "by_model": model_breakdown,
            "by_operation": operation_breakdown,
            "generated_at": datetime.utcnow().isoformat()
        }
    
    async def get_thread_usage(self, thread_id: str) -> Dict[str, Any]:
        """
        Get usage statistics for a specific thread.
        
        Args:
            thread_id: Thread UUID string
        
        Returns:
            Dict with thread-specific usage statistics
        """
        try:
            uuid_id = UUID(thread_id)
        except ValueError:
            return {"error": "Invalid thread ID", "usage": []}
        
        async with async_session_maker() as session:
            # Get thread info
            thread_result = await session.execute(
                select(Thread).where(Thread.thread_id == uuid_id)
            )
            thread = thread_result.scalar_one_or_none()
            
            # Get usage records
            query = select(TokenUsage).where(
                TokenUsage.thread_id == uuid_id
            ).order_by(TokenUsage.created_at.desc())
            
            result = await session.execute(query)
            records = result.scalars().all()
            
            # Aggregate stats
            total_input = sum(r.input_tokens for r in records)
            total_output = sum(r.output_tokens for r in records)
            total_cost = sum(
                (r.cost_usd or {}).get("total", 0) for r in records
            )
            
            # Model breakdown
            model_stats = {}
            for record in records:
                if record.model not in model_stats:
                    model_stats[record.model] = {
                        "requests": 0,
                        "total_tokens": 0,
                        "cost_usd": 0
                    }
                model_stats[record.model]["requests"] += 1
                model_stats[record.model]["total_tokens"] += record.total_tokens
                model_stats[record.model]["cost_usd"] += (record.cost_usd or {}).get("total", 0)
        
        return {
            "thread_id": thread_id,
            "thread_title": thread.title if thread else None,
            "summary": {
                "total_input_tokens": total_input,
                "total_output_tokens": total_output,
                "total_tokens": total_input + total_output,
                "total_requests": len(records),
                "total_cost_usd": round(total_cost, 4)
            },
            "by_model": [
                {"model": model, **stats}
                for model, stats in model_stats.items()
            ],
            "recent_usage": [
                {
                    "usage_id": str(r.usage_id),
                    "model": r.model,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "total_tokens": r.total_tokens,
                    "cost_usd": r.cost_usd,
                    "operation_type": r.operation_type,
                    "created_at": r.created_at.isoformat() if r.created_at else None
                }
                for r in records[:10]  # Last 10 records
            ]
        }
    
    async def get_daily_usage(
        self,
        days: int = 7,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get daily usage statistics for trend analysis.
        
        Args:
            days: Number of days to look back
            user_id: Optional filter by user
        
        Returns:
            List of daily usage statistics
        """
        start_date = datetime.utcnow() - timedelta(days=days)
        
        async with async_session_maker() as session:
            # Get all records for the period
            query = select(TokenUsage).where(
                TokenUsage.created_at >= start_date
            ).order_by(TokenUsage.created_at)
            
            if user_id:
                query = query.where(TokenUsage.user_id == user_id)
            
            result = await session.execute(query)
            records = result.scalars().all()
        
        # Aggregate by day
        daily_stats = {}
        
        for record in records:
            date_key = record.created_at.date().isoformat() if record.created_at else "unknown"
            
            if date_key not in daily_stats:
                daily_stats[date_key] = {
                    "date": date_key,
                    "total_tokens": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "requests": 0,
                    "cost_usd": 0.0,
                    "models_used": set()
                }
            
            daily_stats[date_key]["total_tokens"] += record.total_tokens
            daily_stats[date_key]["input_tokens"] += record.input_tokens
            daily_stats[date_key]["output_tokens"] += record.output_tokens
            daily_stats[date_key]["requests"] += 1
            daily_stats[date_key]["cost_usd"] += (record.cost_usd or {}).get("total", 0)
            daily_stats[date_key]["models_used"].add(record.model)
        
        # Convert to list and format
        result = []
        for date_key in sorted(daily_stats.keys()):
            stats = daily_stats[date_key]
            result.append({
                "date": stats["date"],
                "total_tokens": stats["total_tokens"],
                "input_tokens": stats["input_tokens"],
                "output_tokens": stats["output_tokens"],
                "requests": stats["requests"],
                "cost_usd": round(stats["cost_usd"], 4),
                "models_used": list(stats["models_used"])
            })
        
        return result
    
    async def get_model_comparison(self) -> List[Dict[str, Any]]:
        """
        Get usage comparison across all models.
        
        Returns:
            List of model statistics with cost efficiency metrics
        """
        async with async_session_maker() as session:
            query = select(
                TokenUsage.model,
                func.sum(TokenUsage.input_tokens).label("total_input"),
                func.sum(TokenUsage.output_tokens).label("total_output"),
                func.sum(TokenUsage.total_tokens).label("total_tokens"),
                func.count(TokenUsage.usage_id).label("total_requests"),
                func.avg(TokenUsage.total_tokens).label("avg_tokens_per_request")
            ).group_by(TokenUsage.model)
            
            result = await session.execute(query)
            rows = result.all()
        
        comparison = []
        for row in rows:
            total_input = row.total_input or 0
            total_output = row.total_output or 0
            cost = self.calculate_cost(row.model, total_input, total_output)
            
            # Get model config for display
            config = get_model_config(row.model)
            
            comparison.append({
                "model": row.model,
                "display_name": config.get("display_name", row.model) if config else row.model,
                "provider": config.get("provider", "unknown") if config else "unknown",
                "total_input_tokens": total_input,
                "total_output_tokens": total_output,
                "total_tokens": row.total_tokens or 0,
                "total_requests": row.total_requests or 0,
                "avg_tokens_per_request": round(row.avg_tokens_per_request or 0, 1),
                "total_cost_usd": cost["total"],
                "cost_per_request": round(cost["total"] / max(row.total_requests or 1, 1), 6)
            })
        
        # Sort by total cost descending
        comparison.sort(key=lambda x: x["total_cost_usd"], reverse=True)
        
        return comparison


# Global instance for convenience
usage_tracker = UsageTracker()

