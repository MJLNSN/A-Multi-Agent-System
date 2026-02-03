"""
Usage API routes for the Multi-Agent Chat Threading System.

Provides endpoints for querying token usage statistics and cost analysis.
Enterprise-grade monitoring and cost tracking capabilities.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from src.services.usage_tracker import usage_tracker
from src.models.registry import MODEL_REGISTRY
from src.utils.logging import get_logger

logger = get_logger("usage_routes")

router = APIRouter(prefix="/usage", tags=["Usage & Cost Tracking"])


@router.get("/summary", summary="Get usage summary")
async def get_usage_summary(
    days: int = Query(default=30, ge=1, le=365, description="Number of days to look back"),
    user_id: Optional[str] = Query(default=None, description="Filter by user ID")
):
    """
    Get overall token usage and cost summary.
    
    Provides aggregated statistics including:
    - Total tokens (input/output)
    - Total cost in USD
    - Breakdown by model
    - Breakdown by operation type
    
    Useful for:
    - Budget monitoring
    - Cost allocation
    - Usage trend analysis
    """
    try:
        summary = await usage_tracker.get_usage_summary(days=days, user_id=user_id)
        return summary
    except Exception as e:
        logger.error("usage_summary_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get usage summary: {str(e)}")


@router.get("/thread/{thread_id}", summary="Get thread usage")
async def get_thread_usage(thread_id: str):
    """
    Get detailed token usage for a specific thread.
    
    Returns:
    - Thread summary (total tokens, cost)
    - Breakdown by model used
    - Recent usage records
    
    Useful for:
    - Per-conversation cost tracking
    - Analyzing model switching patterns
    """
    try:
        usage = await usage_tracker.get_thread_usage(thread_id)
        if "error" in usage:
            raise HTTPException(status_code=400, detail=usage["error"])
        return usage
    except HTTPException:
        raise
    except Exception as e:
        logger.error("thread_usage_error", thread_id=thread_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get thread usage: {str(e)}")


@router.get("/daily", summary="Get daily usage")
async def get_daily_usage(
    days: int = Query(default=7, ge=1, le=90, description="Number of days to look back"),
    user_id: Optional[str] = Query(default=None, description="Filter by user ID")
):
    """
    Get daily token usage statistics for trend analysis.
    
    Returns time-series data showing:
    - Daily token counts
    - Daily cost
    - Models used per day
    
    Useful for:
    - Usage trend visualization
    - Capacity planning
    - Detecting anomalies
    """
    try:
        daily_stats = await usage_tracker.get_daily_usage(days=days, user_id=user_id)
        return {
            "period_days": days,
            "daily_usage": daily_stats
        }
    except Exception as e:
        logger.error("daily_usage_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get daily usage: {str(e)}")


@router.get("/models", summary="Get model comparison")
async def get_model_comparison():
    """
    Get usage comparison across all models.
    
    Returns for each model:
    - Total tokens used
    - Total cost
    - Average tokens per request
    - Cost efficiency metrics
    
    Useful for:
    - Model selection optimization
    - Cost efficiency analysis
    - Understanding model usage patterns
    """
    try:
        comparison = await usage_tracker.get_model_comparison()
        return {
            "model_comparison": comparison,
            "available_models": list(MODEL_REGISTRY.keys())
        }
    except Exception as e:
        logger.error("model_comparison_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get model comparison: {str(e)}")


@router.get("/pricing", summary="Get model pricing")
async def get_model_pricing():
    """
    Get pricing information for all available models.
    
    Returns cost per 1K tokens for:
    - Input tokens
    - Output tokens
    
    Useful for:
    - Cost estimation
    - Budget planning
    - Model selection based on cost
    """
    pricing = []
    
    for model_id, config in MODEL_REGISTRY.items():
        costs = config.get("cost_per_1k_tokens", {})
        pricing.append({
            "model": model_id,
            "display_name": config.get("display_name", model_id),
            "provider": config.get("provider", "unknown"),
            "cost_per_1k_input_tokens": costs.get("input", 0),
            "cost_per_1k_output_tokens": costs.get("output", 0),
            "context_window": config.get("context_window", 8000)
        })
    
    # Sort by output cost (usually the main cost driver)
    pricing.sort(key=lambda x: x["cost_per_1k_output_tokens"])
    
    return {
        "pricing": pricing,
        "currency": "USD",
        "note": "Prices are approximate and may vary. Check OpenRouter for current rates."
    }


@router.post("/estimate", summary="Estimate cost")
async def estimate_cost(
    model: str = Query(..., description="Model identifier"),
    input_tokens: int = Query(..., ge=0, description="Estimated input tokens"),
    output_tokens: int = Query(..., ge=0, description="Estimated output tokens")
):
    """
    Estimate the cost for a hypothetical request.
    
    Useful for:
    - Pre-request cost estimation
    - Budget planning
    - Comparing costs across models
    """
    if model not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {model}. Available: {list(MODEL_REGISTRY.keys())}"
        )
    
    cost = usage_tracker.calculate_cost(model, input_tokens, output_tokens)
    
    return {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "estimated_cost_usd": cost,
        "note": "Actual cost may vary based on OpenRouter rates"
    }

