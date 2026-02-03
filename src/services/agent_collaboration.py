"""
Agent Collaboration Service for Multi-Agent Chat Threading System.

Implements a multi-agent collaboration pattern where different AI agents
(Planner, Writer, Reviewer) work together to produce high-quality responses.

This demonstrates true multi-agent orchestration beyond simple model switching.
Includes token usage tracking for cost monitoring.
"""

import asyncio
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from uuid import uuid4
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from src.adapters.openrouter import OpenRouterAdapter
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.services.usage_tracker import UsageTracker

logger = get_logger("agent_collaboration")


class AgentRole(Enum):
    """Available agent roles in the collaboration system."""
    PLANNER = "planner"
    WRITER = "writer"
    REVIEWER = "reviewer"


@dataclass
class AgentConfig:
    """Configuration for an agent role."""
    role: AgentRole
    model: str
    system_prompt: str
    description: str


# Default agent configurations with specialized system prompts
DEFAULT_AGENTS: Dict[AgentRole, AgentConfig] = {
    AgentRole.PLANNER: AgentConfig(
        role=AgentRole.PLANNER,
        model="openai/gpt-4-turbo",
        system_prompt="""You are a strategic PLANNER agent. Your role is to:
1. Analyze the user's question or request carefully
2. Identify key points that need to be addressed
3. Create a clear, structured outline for the response
4. Consider different perspectives and approaches

Output a concise plan (3-5 bullet points) that will guide the Writer agent.
Focus on WHAT should be covered, not the actual content.
Be specific and actionable in your planning.""",
        description="Analyzes questions and creates response strategies"
    ),
    
    AgentRole.WRITER: AgentConfig(
        role=AgentRole.WRITER,
        model="anthropic/claude-3.5-sonnet",
        system_prompt="""You are a skilled WRITER agent. Your role is to:
1. Follow the plan provided by the Planner agent
2. Generate detailed, well-structured content
3. Ensure clarity and coherence in your writing
4. Cover all points in the plan thoroughly

You will receive the original question AND the Planner's outline.
Write comprehensive content that addresses each planned point.
Be informative, accurate, and engaging.""",
        description="Generates detailed content based on the plan"
    ),
    
    AgentRole.REVIEWER: AgentConfig(
        role=AgentRole.REVIEWER,
        model="openai/gpt-4-turbo",
        system_prompt="""You are a critical REVIEWER agent. Your role is to:
1. Review the Writer's content for accuracy and completeness
2. Check if all points from the plan are addressed
3. Identify any gaps, errors, or areas for improvement
4. Provide a refined, polished final response

You will see: Original question → Plan → Draft content
Output the FINAL polished response, incorporating any necessary improvements.
Keep the improvements subtle - only fix real issues, don't over-edit.""",
        description="Reviews and polishes the final output"
    )
}


class AgentCollaborationService:
    """
    Orchestrates multi-agent collaboration for complex queries.
    
    Flow:
    1. User query → Planner (creates outline)
    2. Query + Plan → Writer (generates content)
    3. Query + Plan + Draft → Reviewer (final polish)
    
    Each step uses a different LLM model optimized for that role.
    Tracks token usage for cost monitoring.
    """
    
    def __init__(
        self,
        openrouter_adapter: OpenRouterAdapter,
        usage_tracker: Optional["UsageTracker"] = None
    ):
        """
        Initialize the Agent Collaboration Service.
        
        Args:
            openrouter_adapter: OpenRouter API adapter for LLM calls
            usage_tracker: Optional usage tracker for cost monitoring
        """
        self.adapter = openrouter_adapter
        self.usage_tracker = usage_tracker
        self.agents = DEFAULT_AGENTS.copy()
    
    def get_agent_config(self, role: AgentRole) -> AgentConfig:
        """Get configuration for a specific agent role."""
        return self.agents[role]
    
    def update_agent_model(self, role: AgentRole, model: str):
        """
        Update the model used by a specific agent role.
        
        Args:
            role: Agent role to update
            model: New model identifier
        """
        config = self.agents[role]
        self.agents[role] = AgentConfig(
            role=config.role,
            model=model,
            system_prompt=config.system_prompt,
            description=config.description
        )
        logger.info("agent_model_updated", role=role.value, new_model=model)
    
    async def collaborate(
        self,
        query: str,
        context: Optional[str] = None,
        include_process: bool = True
    ) -> Dict[str, Any]:
        """
        Execute multi-agent collaboration on a query.
        
        Args:
            query: User's question or request
            context: Optional additional context
            include_process: Whether to include intermediate steps in response
        
        Returns:
            Dict containing:
            - final_response: The polished final answer
            - collaboration_process: Details of each agent's contribution
            - metadata: Timing, tokens, models used
        """
        collaboration_id = str(uuid4())[:8]
        start_time = datetime.utcnow()
        
        logger.info(
            "collaboration_started",
            collaboration_id=collaboration_id,
            query_length=len(query)
        )
        
        process_steps = []
        total_tokens = 0
        
        # Build context prefix if provided
        context_prefix = f"Context: {context}\n\n" if context else ""
        
        # ============================================================
        # Step 1: PLANNER - Analyze and create response strategy
        # ============================================================
        planner_config = self.agents[AgentRole.PLANNER]
        
        planner_prompt = f"""{context_prefix}User Question: {query}

Create a clear plan for answering this question. Output 3-5 bullet points."""
        
        logger.info("agent_step", step=1, role="planner", model=planner_config.model)
        
        planner_response = await self.adapter.chat_completion(
            model=planner_config.model,
            messages=[
                {"role": "system", "content": planner_config.system_prompt},
                {"role": "user", "content": planner_prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        plan = planner_response["content"]
        planner_tokens = planner_response.get("tokens", 0)
        total_tokens += planner_tokens
        
        # Track planner usage
        if self.usage_tracker:
            usage_data = planner_response.get("usage", {})
            await self.usage_tracker.track_usage(
                model=planner_config.model,
                input_tokens=usage_data.get("prompt_tokens", 0),
                output_tokens=usage_data.get("completion_tokens", 0),
                operation_type="collaboration",
                collaboration_id=collaboration_id,
                extra_data={"agent_role": "planner", "step": 1}
            )
        
        process_steps.append({
            "step": 1,
            "role": AgentRole.PLANNER.value,
            "model": planner_config.model,
            "input": planner_prompt[:200] + "..." if len(planner_prompt) > 200 else planner_prompt,
            "output": plan,
            "tokens": planner_tokens
        })
        
        # ============================================================
        # Step 2: WRITER - Generate detailed content based on plan
        # ============================================================
        writer_config = self.agents[AgentRole.WRITER]
        
        writer_prompt = f"""{context_prefix}User Question: {query}

=== PLANNER'S OUTLINE ===
{plan}

Based on this plan, write a comprehensive response addressing each point."""
        
        logger.info("agent_step", step=2, role="writer", model=writer_config.model)
        
        writer_response = await self.adapter.chat_completion(
            model=writer_config.model,
            messages=[
                {"role": "system", "content": writer_config.system_prompt},
                {"role": "user", "content": writer_prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        
        draft = writer_response["content"]
        writer_tokens = writer_response.get("tokens", 0)
        total_tokens += writer_tokens
        
        # Track writer usage
        if self.usage_tracker:
            usage_data = writer_response.get("usage", {})
            await self.usage_tracker.track_usage(
                model=writer_config.model,
                input_tokens=usage_data.get("prompt_tokens", 0),
                output_tokens=usage_data.get("completion_tokens", 0),
                operation_type="collaboration",
                collaboration_id=collaboration_id,
                extra_data={"agent_role": "writer", "step": 2}
            )
        
        process_steps.append({
            "step": 2,
            "role": AgentRole.WRITER.value,
            "model": writer_config.model,
            "input": f"Question + Plan ({len(plan)} chars)",
            "output": draft[:500] + "..." if len(draft) > 500 else draft,
            "tokens": writer_tokens
        })
        
        # ============================================================
        # Step 3: REVIEWER - Polish and finalize the response
        # ============================================================
        reviewer_config = self.agents[AgentRole.REVIEWER]
        
        reviewer_prompt = f"""{context_prefix}User Question: {query}

=== PLANNER'S OUTLINE ===
{plan}

=== WRITER'S DRAFT ===
{draft}

Review this draft and provide the final, polished response. Fix any issues but keep changes minimal if the draft is good."""
        
        logger.info("agent_step", step=3, role="reviewer", model=reviewer_config.model)
        
        reviewer_response = await self.adapter.chat_completion(
            model=reviewer_config.model,
            messages=[
                {"role": "system", "content": reviewer_config.system_prompt},
                {"role": "user", "content": reviewer_prompt}
            ],
            temperature=0.5,  # Lower temperature for more consistent review
            max_tokens=1500
        )
        
        final_response = reviewer_response["content"]
        reviewer_tokens = reviewer_response.get("tokens", 0)
        total_tokens += reviewer_tokens
        
        # Track reviewer usage
        if self.usage_tracker:
            usage_data = reviewer_response.get("usage", {})
            await self.usage_tracker.track_usage(
                model=reviewer_config.model,
                input_tokens=usage_data.get("prompt_tokens", 0),
                output_tokens=usage_data.get("completion_tokens", 0),
                operation_type="collaboration",
                collaboration_id=collaboration_id,
                extra_data={"agent_role": "reviewer", "step": 3}
            )
        
        process_steps.append({
            "step": 3,
            "role": AgentRole.REVIEWER.value,
            "model": reviewer_config.model,
            "input": f"Question + Plan + Draft ({len(draft)} chars)",
            "output": final_response[:500] + "..." if len(final_response) > 500 else final_response,
            "tokens": reviewer_tokens
        })
        
        # ============================================================
        # Build final result
        # ============================================================
        end_time = datetime.utcnow()
        duration_ms = (end_time - start_time).total_seconds() * 1000
        
        logger.info(
            "collaboration_completed",
            collaboration_id=collaboration_id,
            total_tokens=total_tokens,
            duration_ms=duration_ms
        )
        
        # Calculate estimated cost
        estimated_cost = None
        if self.usage_tracker:
            # Sum up costs from all steps (rough estimate based on total tokens)
            planner_cost = self.usage_tracker.calculate_cost(
                planner_config.model, planner_tokens // 2, planner_tokens // 2
            )
            writer_cost = self.usage_tracker.calculate_cost(
                writer_config.model, writer_tokens // 2, writer_tokens // 2
            )
            reviewer_cost = self.usage_tracker.calculate_cost(
                reviewer_config.model, reviewer_tokens // 2, reviewer_tokens // 2
            )
            estimated_cost = {
                "planner": planner_cost["total"],
                "writer": writer_cost["total"],
                "reviewer": reviewer_cost["total"],
                "total": round(planner_cost["total"] + writer_cost["total"] + reviewer_cost["total"], 6)
            }
        
        result = {
            "collaboration_id": collaboration_id,
            "final_response": final_response,
            "metadata": {
                "total_tokens": total_tokens,
                "duration_ms": round(duration_ms, 2),
                "estimated_cost_usd": estimated_cost,
                "agents_used": [
                    {"role": AgentRole.PLANNER.value, "model": planner_config.model, "tokens": planner_tokens},
                    {"role": AgentRole.WRITER.value, "model": writer_config.model, "tokens": writer_tokens},
                    {"role": AgentRole.REVIEWER.value, "model": reviewer_config.model, "tokens": reviewer_tokens}
                ],
                "timestamp": start_time.isoformat()
            }
        }
        
        if include_process:
            result["collaboration_process"] = {
                "plan": plan,
                "draft": draft,
                "steps": process_steps
            }
        
        return result
    
    async def quick_collaborate(
        self,
        query: str,
        roles: List[AgentRole] = None
    ) -> Dict[str, Any]:
        """
        Run a quick collaboration with optional subset of agents.
        
        Args:
            query: User's question
            roles: Optional list of roles to use (default: all three)
        
        Returns:
            Collaboration result
        """
        if roles is None:
            roles = [AgentRole.PLANNER, AgentRole.WRITER, AgentRole.REVIEWER]
        
        # For now, delegate to full collaboration
        # Could be extended to support partial workflows
        return await self.collaborate(query, include_process=True)
    
    def list_agents(self) -> List[Dict[str, str]]:
        """List all available agents and their configurations."""
        return [
            {
                "role": config.role.value,
                "model": config.model,
                "description": config.description
            }
            for config in self.agents.values()
        ]

