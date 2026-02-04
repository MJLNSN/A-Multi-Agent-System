"""
API routes for Multi-Agent Collaboration feature.

Provides endpoints for orchestrating multiple AI agents working together
on complex queries using the Planner → Writer → Reviewer pattern.

Optimizations:
- Simple queries automatically skip Reviewer (saves tokens and time)
- Reviewer only sees draft summary/key sections (saves 30-50% tokens)
- Use force_full_pipeline=true to always run all 3 agents
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from src.services.agent_collaboration import AgentRole

router = APIRouter(tags=["collaboration"])


class CollaborateRequest(BaseModel):
    """Request model for agent collaboration."""
    query: str = Field(..., min_length=1, max_length=10000, description="The question or task for agents to collaborate on")
    context: Optional[str] = Field(None, max_length=5000, description="Optional additional context")
    include_process: bool = Field(True, description="Include intermediate steps in response")
    force_full_pipeline: bool = Field(False, description="Force using all 3 agents even for simple queries")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "What are the key strategies for launching a successful SaaS product in 2025?",
                    "context": "We are a B2B startup targeting small businesses.",
                    "include_process": True,
                    "force_full_pipeline": False
                }
            ]
        }
    }


class UpdateAgentModelRequest(BaseModel):
    """Request model for updating an agent's model."""
    role: str = Field(..., description="Agent role: planner, writer, or reviewer")
    model: str = Field(..., description="New model identifier (e.g., 'openai/gpt-4-turbo')")


class CollaborateResponse(BaseModel):
    """Response model for collaboration endpoint."""
    collaboration_id: str
    final_response: str
    metadata: Dict[str, Any]
    collaboration_process: Optional[Dict[str, Any]] = None


@router.post("/collaborate", response_model=CollaborateResponse)
async def collaborate(request: CollaborateRequest, app_request: Request):
    """
    Execute multi-agent collaboration on a query.
    
    **Optimized Flow:**
    - **Simple queries**: Planner → Writer (Reviewer skipped, saves tokens)
    - **Complex queries**: Planner → Writer → Reviewer (with draft summary)
    
    **Agents:**
    1. **Planner Agent** (GPT-4): Analyzes the question and creates a response strategy
    2. **Writer Agent** (Claude 3.5): Generates detailed content following the plan
    3. **Reviewer Agent** (GPT-4): Reviews and polishes (only for complex queries)
    
    **Optimizations:**
    - Complexity classifier automatically determines if Reviewer is needed
    - Reviewer only sees key sections/summary (saves 30-50% tokens)
    - Use `force_full_pipeline=true` to always run all 3 agents
    
    **Complexity Factors:**
    - Query length (>100 chars)
    - Multiple sub-questions (numbered lists)
    - Analysis keywords (analyze, compare, strategy, etc.)
    - Additional context provided
    
    **Examples:**
    
    Simple query (Reviewer skipped):
    ```json
    {"query": "What is machine learning?"}
    ```
    
    Complex query (full pipeline):
    ```json
    {
        "query": "1. Market analysis 2. Competitive advantage 3. Success metrics",
        "context": "We are launching a SaaS product",
        "include_process": true
    }
    ```
    
    Force full pipeline:
    ```json
    {"query": "Simple question", "force_full_pipeline": true}
    ```
    """
    try:
        result = await app_request.app.state.collaboration_service.collaborate(
            query=request.query,
            context=request.context,
            include_process=request.include_process,
            force_full_pipeline=request.force_full_pipeline
        )
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Collaboration failed: {str(e)}")


@router.get("/agents")
async def list_agents(app_request: Request):
    """
    List all available collaboration agents and their configurations.
    
    Returns information about each agent role:
    - **planner**: Creates response strategies (GPT-4)
    - **writer**: Generates detailed content (Claude 3.5)
    - **reviewer**: Reviews and polishes output (GPT-4) - *may be skipped for simple queries*
    
    Each agent can be configured to use different LLM models.
    """
    try:
        agents = app_request.app.state.collaboration_service.list_agents()
        return {
            "agents": agents,
            "collaboration_flow": {
                "simple_queries": [
                    "1. User query → Planner (strategy)",
                    "2. Query + Plan → Writer (content) → Final Response"
                ],
                "complex_queries": [
                    "1. User query → Planner (strategy)",
                    "2. Query + Plan → Writer (content)",
                    "3. Query + Plan + Draft Summary → Reviewer (polish)"
                ]
            },
            "optimizations": {
                "complexity_classifier": "Automatically determines if Reviewer is needed",
                "draft_summary": "Reviewer sees key sections only (saves 30-50% tokens)",
                "force_full_pipeline": "Set to true to always use all 3 agents"
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/agents/{role}")
async def update_agent_model(
    role: str,
    request: UpdateAgentModelRequest,
    app_request: Request
):
    """
    Update the LLM model used by a specific agent role.
    
    Allows dynamic reconfiguration of which model each agent uses.
    This enables experimentation with different model combinations.
    
    **Parameters:**
    - `role`: One of 'planner', 'writer', or 'reviewer'
    - `model`: Model identifier (e.g., 'openai/gpt-4-turbo', 'anthropic/claude-3.5-sonnet')
    
    **Example:**
    ```
    PATCH /api/agents/writer
    {"role": "writer", "model": "openai/gpt-4-turbo"}
    ```
    """
    try:
        # Validate role
        try:
            agent_role = AgentRole(role.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid role: {role}. Must be one of: planner, writer, reviewer"
            )
        
        # Validate model
        from src.models.registry import validate_model, list_available_models
        if not validate_model(request.model):
            available = list_available_models()
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model: {request.model}. Available: {available}"
            )
        
        # Update the agent's model
        app_request.app.state.collaboration_service.update_agent_model(
            agent_role,
            request.model
        )
        
        # Return updated configuration
        config = app_request.app.state.collaboration_service.get_agent_config(agent_role)
        return {
            "message": f"Agent '{role}' updated successfully",
            "agent": {
                "role": config.role.value,
                "model": config.model,
                "description": config.description
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

