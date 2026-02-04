"""
Unit tests for Agent Collaboration Service.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from uuid import uuid4

from src.services.agent_collaboration import (
    AgentCollaborationService,
    AgentRole,
    AgentConfig,
    DEFAULT_AGENTS
)


class TestAgentCollaborationService:
    """Test cases for AgentCollaborationService."""
    
    @pytest.fixture
    def mock_adapter(self):
        """Create a mock OpenRouter adapter."""
        adapter = AsyncMock()
        adapter.chat_completion = AsyncMock()
        return adapter
    
    @pytest.fixture
    def service(self, mock_adapter):
        """Create a service instance with mock adapter."""
        return AgentCollaborationService(mock_adapter)
    
    def test_initialization(self, service):
        """Test service initializes with default agents."""
        assert len(service.agents) == 3
        assert AgentRole.PLANNER in service.agents
        assert AgentRole.WRITER in service.agents
        assert AgentRole.REVIEWER in service.agents
    
    def test_default_agent_models(self, service):
        """Test default agent model assignments."""
        # Planner uses GPT-4
        assert "gpt-4" in service.agents[AgentRole.PLANNER].model.lower()
        # Writer uses Claude
        assert "claude" in service.agents[AgentRole.WRITER].model.lower()
        # Reviewer uses GPT-4
        assert "gpt-4" in service.agents[AgentRole.REVIEWER].model.lower()
    
    def test_get_agent_config(self, service):
        """Test getting agent configuration."""
        config = service.get_agent_config(AgentRole.PLANNER)
        assert config.role == AgentRole.PLANNER
        assert config.model is not None
        assert config.system_prompt is not None
    
    def test_update_agent_model(self, service):
        """Test updating an agent's model."""
        new_model = "openai/gpt-3.5-turbo"
        service.update_agent_model(AgentRole.WRITER, new_model)
        
        config = service.get_agent_config(AgentRole.WRITER)
        assert config.model == new_model
    
    def test_list_agents(self, service):
        """Test listing all agents."""
        agents = service.list_agents()
        assert len(agents) == 3
        
        roles = [a["role"] for a in agents]
        assert "planner" in roles
        assert "writer" in roles
        assert "reviewer" in roles
    
    @pytest.mark.asyncio
    async def test_collaborate_calls_all_agents(self, service, mock_adapter):
        """Test that collaboration calls all three agents in order when forced."""
        # Setup mock responses
        mock_adapter.chat_completion.side_effect = [
            {"content": "Plan: 1. Point A, 2. Point B", "tokens": 50},
            {"content": "Draft content based on the plan...", "tokens": 200},
            {"content": "Final polished response", "tokens": 150}
        ]
        
        # Use force_full_pipeline=True to ensure all agents are called
        result = await service.collaborate(
            "Test question", 
            include_process=True,
            force_full_pipeline=True
        )
        
        # Should call adapter 3 times (Planner, Writer, Reviewer)
        assert mock_adapter.chat_completion.call_count == 3
        
        # Verify result structure
        assert "collaboration_id" in result
        assert "final_response" in result
        assert "metadata" in result
        assert "collaboration_process" in result
    
    @pytest.mark.asyncio
    async def test_collaborate_result_structure(self, service, mock_adapter):
        """Test collaboration result has correct structure when forced full pipeline."""
        mock_adapter.chat_completion.side_effect = [
            {"content": "Plan", "tokens": 50},
            {"content": "Draft", "tokens": 200},
            {"content": "Final", "tokens": 150}
        ]
        
        # Use force_full_pipeline=True to test full structure
        result = await service.collaborate("Test", force_full_pipeline=True)
        
        # Check metadata
        assert "total_tokens" in result["metadata"]
        assert result["metadata"]["total_tokens"] == 400
        assert "duration_ms" in result["metadata"]
        assert "agents_used" in result["metadata"]
        assert len(result["metadata"]["agents_used"]) == 3
        
        # Check process
        assert "plan" in result["collaboration_process"]
        assert "draft" in result["collaboration_process"]
        assert "steps" in result["collaboration_process"]
        
        # Check complexity info is present
        assert "complexity" in result["metadata"]
        assert "optimization" in result["metadata"]
    
    @pytest.mark.asyncio
    async def test_collaborate_without_process(self, service, mock_adapter):
        """Test collaboration without including process details."""
        mock_adapter.chat_completion.side_effect = [
            {"content": "Plan", "tokens": 50},
            {"content": "Draft", "tokens": 200},
            {"content": "Final", "tokens": 150}
        ]
        
        result = await service.collaborate("Test", include_process=False)
        
        assert "collaboration_process" not in result
        assert "final_response" in result
    
    @pytest.mark.asyncio
    async def test_collaborate_with_context(self, service, mock_adapter):
        """Test collaboration with additional context."""
        mock_adapter.chat_completion.side_effect = [
            {"content": "Plan", "tokens": 50},
            {"content": "Draft", "tokens": 200},
            {"content": "Final", "tokens": 150}
        ]
        
        context = "Additional context information"
        result = await service.collaborate("Test", context=context)
        
        # Check that context was passed to the adapter
        first_call = mock_adapter.chat_completion.call_args_list[0]
        messages = first_call.kwargs["messages"]
        user_message = [m for m in messages if m["role"] == "user"][0]
        assert "Context:" in user_message["content"]
    
    @pytest.mark.asyncio
    async def test_collaborate_step_order(self, service, mock_adapter):
        """Test that agents are called in correct order when forced full pipeline."""
        call_order = []
        
        async def track_calls(**kwargs):
            model = kwargs.get("model", "")
            call_order.append(model)
            return {"content": "Response", "tokens": 100}
        
        mock_adapter.chat_completion.side_effect = track_calls
        
        # Use force_full_pipeline=True to ensure all agents are called
        await service.collaborate("Test", force_full_pipeline=True)
        
        # Should be: Planner (GPT-4), Writer (Claude), Reviewer (GPT-4)
        assert len(call_order) == 3
        assert "gpt-4" in call_order[0].lower()  # Planner
        assert "claude" in call_order[1].lower()  # Writer
        assert "gpt-4" in call_order[2].lower()  # Reviewer


class TestComplexityClassifier:
    """Test complexity classification and optimization features."""
    
    @pytest.fixture
    def mock_adapter(self):
        """Create a mock OpenRouter adapter."""
        adapter = AsyncMock()
        adapter.chat_completion = AsyncMock()
        return adapter
    
    @pytest.fixture
    def service(self, mock_adapter):
        """Create a service instance with mock adapter."""
        return AgentCollaborationService(mock_adapter)
    
    def test_simple_query_classification(self, service):
        """Test that simple queries are classified correctly."""
        # Simple short query
        result = service.classify_complexity("What is AI?")
        assert result["is_complex"] == False
        assert result["score"] < 4
        assert result["recommendation"] == "skip_reviewer"
    
    def test_complex_query_classification(self, service):
        """Test that complex queries are classified correctly."""
        # Complex query with multiple sub-questions and analysis keywords
        query = """
        分析我们的市场策略：
        1. 市场进入策略
        2. 竞争优势分析
        3. 成功指标评估
        """
        result = service.classify_complexity(query, context="我们是创业公司")
        assert result["is_complex"] == True
        assert result["score"] >= 4
        assert result["recommendation"] == "full_pipeline"
        assert "multi_questions" in str(result["reasons"])
    
    def test_context_increases_complexity(self, service):
        """Test that adding context increases complexity score."""
        query = "How to build a product?"
        
        result_no_context = service.classify_complexity(query)
        result_with_context = service.classify_complexity(
            query, 
            context="We are a 15-person startup with $2M funding targeting enterprise customers"
        )
        
        assert result_with_context["score"] > result_no_context["score"]
    
    @pytest.mark.asyncio
    async def test_simple_query_skips_reviewer(self, service, mock_adapter):
        """Test that simple queries skip the reviewer agent."""
        mock_adapter.chat_completion.side_effect = [
            {"content": "Plan", "tokens": 50},
            {"content": "Draft", "tokens": 200}
        ]
        
        # Simple query should skip reviewer
        result = await service.collaborate("What is AI?", include_process=True)
        
        # Should only call 2 agents (Planner, Writer)
        assert mock_adapter.chat_completion.call_count == 2
        
        # Verify metadata shows reviewer was skipped
        assert result["metadata"]["complexity"]["reviewer_used"] == False
        assert result["metadata"]["optimization"]["reviewer_skipped"] == True
        
        # Final response should be the writer's draft
        assert result["final_response"] == "Draft"
    
    @pytest.mark.asyncio
    async def test_force_full_pipeline_overrides_optimization(self, service, mock_adapter):
        """Test that force_full_pipeline=True uses all agents regardless of complexity."""
        mock_adapter.chat_completion.side_effect = [
            {"content": "Plan", "tokens": 50},
            {"content": "Draft", "tokens": 200},
            {"content": "Final", "tokens": 150}
        ]
        
        # Simple query but forced full pipeline
        result = await service.collaborate(
            "What is AI?", 
            include_process=True,
            force_full_pipeline=True
        )
        
        # Should call all 3 agents
        assert mock_adapter.chat_completion.call_count == 3
        assert result["metadata"]["complexity"]["reviewer_used"] == True
        assert result["metadata"]["optimization"]["reviewer_skipped"] == False
    
    def test_extract_key_sections(self, service):
        """Test draft summary extraction."""
        draft = """
        ## 一、市场分析
        这是市场分析的详细内容，包含很多信息。
        
        ## 二、竞争优势
        1. 第一个优势点
        2. 第二个优势点
        3. 第三个优势点
        
        ## 三、执行建议
        这是执行建议的详细说明。
        """
        
        summary = service.extract_key_sections(draft, max_chars=500)
        
        # Summary should be shorter than original
        assert len(summary) <= len(draft)
        # Should contain section headers
        assert "市场分析" in summary or "竞争优势" in summary


class TestAgentRole:
    """Test AgentRole enum."""
    
    def test_role_values(self):
        """Test role enum values."""
        assert AgentRole.PLANNER.value == "planner"
        assert AgentRole.WRITER.value == "writer"
        assert AgentRole.REVIEWER.value == "reviewer"
    
    def test_role_from_string(self):
        """Test creating role from string."""
        assert AgentRole("planner") == AgentRole.PLANNER
        assert AgentRole("writer") == AgentRole.WRITER
        assert AgentRole("reviewer") == AgentRole.REVIEWER


class TestDefaultAgents:
    """Test default agent configurations."""
    
    def test_all_roles_configured(self):
        """Test all roles have default configurations."""
        assert AgentRole.PLANNER in DEFAULT_AGENTS
        assert AgentRole.WRITER in DEFAULT_AGENTS
        assert AgentRole.REVIEWER in DEFAULT_AGENTS
    
    def test_planner_config(self):
        """Test planner agent configuration."""
        config = DEFAULT_AGENTS[AgentRole.PLANNER]
        assert "PLANNER" in config.system_prompt
        assert "outline" in config.system_prompt.lower() or "plan" in config.system_prompt.lower()
    
    def test_writer_config(self):
        """Test writer agent configuration."""
        config = DEFAULT_AGENTS[AgentRole.WRITER]
        assert "WRITER" in config.system_prompt
        assert "content" in config.system_prompt.lower() or "write" in config.system_prompt.lower()
    
    def test_reviewer_config(self):
        """Test reviewer agent configuration."""
        config = DEFAULT_AGENTS[AgentRole.REVIEWER]
        assert "REVIEWER" in config.system_prompt
        assert "review" in config.system_prompt.lower() or "polish" in config.system_prompt.lower()


class TestComplexScenarios:
    """Test complex real-world scenarios with sophisticated queries."""
    
    @pytest.fixture
    def mock_adapter(self):
        """Create a mock OpenRouter adapter with realistic responses."""
        adapter = AsyncMock()
        adapter.chat_completion = AsyncMock()
        return adapter
    
    @pytest.fixture
    def service(self, mock_adapter):
        """Create a service instance with mock adapter."""
        return AgentCollaborationService(mock_adapter)
    
    @pytest.mark.asyncio
    async def test_technical_architecture_question(self, service, mock_adapter):
        """
        Test collaboration on a complex technical architecture question.
        This simulates a real-world scenario where a user asks about designing
        a distributed system with specific requirements.
        """
        query = """
        Design a scalable microservices architecture for an e-commerce platform 
        that handles 1 million daily active users. The system must support:
        - Real-time inventory management
        - Payment processing with multiple providers
        - Order tracking and notifications
        - Product recommendations using ML
        - High availability (99.99% uptime)
        
        Include database choices, caching strategies, and message queuing.
        """
        
        # Simulate realistic agent responses
        mock_adapter.chat_completion.side_effect = [
            {
                "content": """Plan:
1. Define core microservices (User, Product, Order, Payment, Notification, Recommendation)
2. Design data layer (PostgreSQL for transactions, MongoDB for catalog, Redis for caching)
3. Implement event-driven architecture with Kafka for inter-service communication
4. Add API Gateway with rate limiting and authentication
5. Set up monitoring, logging, and disaster recovery""",
                "tokens": 120
            },
            {
                "content": """Based on the architectural plan, here's a comprehensive design:

**Core Microservices:**
- User Service: Handles authentication, profiles, preferences
- Product Service: Catalog management, search, filtering
- Order Service: Order creation, state management, history
- Payment Service: Multi-provider integration (Stripe, PayPal), transaction handling
- Notification Service: Email, SMS, push notifications via queues
- Recommendation Service: ML-based product suggestions using collaborative filtering

**Data Layer:**
- PostgreSQL: User data, orders, transactions (ACID compliance)
- MongoDB: Product catalog (flexible schema for varied products)
- Redis: Session management, product cache, real-time inventory
- Elasticsearch: Product search and analytics

**Event-Driven Architecture:**
- Apache Kafka: Event streaming for order events, inventory updates
- Message patterns: Event sourcing for order history, CQRS for read/write separation

**Infrastructure:**
- API Gateway (Kong/AWS API Gateway): Rate limiting, authentication, routing
- Load Balancers: Distribute traffic across service instances
- Container Orchestration: Kubernetes for auto-scaling and self-healing
- CDN: CloudFlare for static assets and global distribution

**High Availability:**
- Multi-region deployment with active-active configuration
- Database replication (master-slave for PostgreSQL, replica sets for MongoDB)
- Circuit breakers and fallback mechanisms
- Health checks and automatic failover""",
                "tokens": 450
            },
            {
                "content": """Final Architecture Review:

The proposed microservices architecture effectively addresses all requirements:

**Scalability (✓):** 
- Horizontal scaling via Kubernetes ensures handling of 1M+ DAU
- Redis caching reduces database load by 70-80%
- CDN offloads static content delivery

**Real-time Inventory (✓):**
- Redis provides sub-millisecond inventory checks
- Kafka streams ensure eventual consistency across services
- Optimistic locking prevents overselling

**Payment Processing (✓):**
- Dedicated Payment Service with provider abstraction
- PCI DSS compliance through tokenization
- Retry mechanisms and idempotency keys for reliability

**Order Tracking (✓):**
- Event sourcing maintains complete order history
- Notification Service triggers updates via Kafka events
- WebSocket connections for real-time status updates

**ML Recommendations (✓):**
- Separate Recommendation Service allows independent scaling
- Batch processing for model training, real-time inference
- A/B testing framework for algorithm optimization

**High Availability (✓):**
- Multi-region deployment achieves 99.99% uptime
- Database replication and automated failover
- Circuit breakers prevent cascading failures
- Comprehensive monitoring with Prometheus and Grafana

**Additional Considerations:**
- Implement API versioning for backward compatibility
- Use distributed tracing (Jaeger) for debugging
- Establish SLAs and SLOs for each service
- Regular chaos engineering tests to validate resilience

This architecture provides a robust foundation for a production-grade e-commerce platform.""",
                "tokens": 380
            }
        ]
        
        result = await service.collaborate(query, include_process=True)
        
        # Verify the collaboration process
        assert mock_adapter.chat_completion.call_count == 3
        assert result["metadata"]["total_tokens"] == 950
        
        # Verify process structure
        assert "Plan:" in result["collaboration_process"]["plan"]
        assert "microservices" in result["collaboration_process"]["draft"].lower()
        assert "architecture" in result["final_response"].lower()
        assert "scalability" in result["final_response"].lower()
    
    @pytest.mark.asyncio
    async def test_multi_step_problem_solving(self, service, mock_adapter):
        """
        Test collaboration on a complex problem requiring multiple reasoning steps.
        This tests the agents' ability to break down and solve a challenging problem.
        """
        query = """
        A startup has $500K in funding and 6 months runway. They need to:
        1. Build an MVP for a SaaS product
        2. Acquire first 100 paying customers
        3. Achieve $50K MRR
        
        Current team: 2 developers, 1 designer. No sales or marketing experience.
        Create a detailed execution plan with timeline, budget allocation, and key milestones.
        """
        
        mock_adapter.chat_completion.side_effect = [
            {
                "content": """Plan:
1. Month 1-2: MVP Development (Lean approach, core features only)
2. Month 2-3: Beta Testing & Iteration (20-30 beta users)
3. Month 3-4: Go-to-Market Strategy (Content marketing, partnerships)
4. Month 4-6: Customer Acquisition & Growth (Sales outreach, paid ads)
5. Throughout: Budget management and metric tracking""",
                "tokens": 100
            },
            {
                "content": """Detailed Execution Plan:

**Phase 1: MVP Development (Month 1-2) - Budget: $100K**
- Focus: Build minimum viable product with 3-5 core features
- Team allocation: 2 devs (full-time), 1 designer (full-time)
- Tech stack: Use proven technologies to minimize risk
- Deliverable: Functional MVP with basic user flow
- Milestone: Working demo by end of Month 2

**Phase 2: Beta Testing (Month 2-3) - Budget: $30K**
- Recruit 20-30 beta users through personal network, Reddit, Product Hunt
- Collect feedback via surveys, user interviews (15-20 sessions)
- Iterate on critical bugs and UX issues
- Hire part-time customer success person ($5K/month)
- Milestone: 80% beta user satisfaction score

**Phase 3: Go-to-Market Preparation (Month 3-4) - Budget: $80K**
- Content marketing: Blog posts, case studies, video demos
- SEO optimization and landing page A/B testing
- Build email list through lead magnets
- Establish partnerships with complementary tools
- Hire growth marketer ($8K/month)
- Milestone: 1000 email subscribers, 5 content pieces

**Phase 4: Customer Acquisition (Month 4-6) - Budget: $200K**
- Paid advertising: Google Ads, LinkedIn Ads ($60K total)
- Outbound sales: Hire 1 SDR ($6K/month), cold outreach campaign
- Referral program: Incentivize existing users
- Webinars and demos (2 per week)
- PR push: TechCrunch, industry publications
- Milestone: 100 paying customers, $50K MRR

**Budget Breakdown:**
- Development & Operations: $150K (30%)
- Marketing & Sales: $250K (50%)
- Team expansion: $70K (14%)
- Reserve fund: $30K (6%)

**Key Metrics to Track:**
- Weekly active users (WAU)
- Conversion rate (trial to paid)
- Customer acquisition cost (CAC)
- Monthly recurring revenue (MRR)
- Churn rate

**Risk Mitigation:**
- If customer acquisition is slow, pivot to freemium model
- If budget runs tight, reduce paid ads, focus on organic growth
- If product-market fit unclear, conduct more user research""",
                "tokens": 520
            },
            {
                "content": """Final Execution Plan Review:

**Feasibility Assessment:**

✓ **Timeline Realistic:** 6-month plan is aggressive but achievable with focused execution
✓ **Budget Allocation Appropriate:** 50% on marketing/sales is correct for B2B SaaS
✓ **Team Scaling Sensible:** Gradual hiring (CS → Growth → Sales) minimizes burn rate
✓ **Milestones Measurable:** Clear KPIs for each phase enable course correction

**Critical Success Factors:**

1. **Product-Market Fit (Months 1-3):** 
   - Beta testing phase is crucial - don't rush to paid launch
   - Target: 80%+ satisfaction score before scaling acquisition
   - Red flag: If beta users aren't willing to pay, pivot immediately

2. **Efficient Customer Acquisition (Months 4-6):**
   - Initial CAC target: <$500 (to achieve profitability within 12 months)
   - Focus on channels with shortest sales cycle
   - Prioritize: Content marketing (low cost) → Partnerships → Paid ads

3. **Revenue Target ($50K MRR):**
   - Requires 100 customers at $500/month average
   - Or 200 customers at $250/month
   - Recommend tiered pricing: Starter ($199), Pro ($499), Enterprise ($999)

4. **Runway Management:**
   - Monthly burn rate: ~$83K
   - Break-even target: Month 8-9 (2-3 months post-funding)
   - Contingency: Prepare bridge round pitch by Month 5

**Recommended Adjustments:**

- Allocate $20K for user research before MVP development
- Set up analytics infrastructure (Mixpanel, Amplitude) from Day 1
- Establish weekly OKR reviews to maintain momentum
- Build relationships with potential investors throughout process

**Probability of Success:**
- Achieving 100 customers: 70% (with strong execution)
- Achieving $50K MRR: 60% (depends on pricing power)
- Extending runway: 85% (if growth metrics are strong)

This plan provides a structured path to traction while maintaining financial discipline.""",
                "tokens": 480
            }
        ]
        
        result = await service.collaborate(query, include_process=True)
        
        # Verify comprehensive response
        assert "Budget" in result["collaboration_process"]["draft"]
        assert "Month" in result["collaboration_process"]["draft"]
        assert "Feasibility" in result["final_response"] or "Success" in result["final_response"]
        assert result["metadata"]["total_tokens"] == 1100
    
    @pytest.mark.asyncio
    async def test_creative_content_generation(self, service, mock_adapter):
        """
        Test collaboration on creative content requiring both structure and creativity.
        This tests the synergy between planning, creative writing, and editorial review.
        """
        query = """
        Write a compelling product launch announcement for an AI-powered code review tool 
        called "CodeGuardian". Target audience: Engineering managers at Series A-C startups.
        
        Key features:
        - Catches bugs before they reach production (95% accuracy)
        - Integrates with GitHub, GitLab, Bitbucket
        - Learns your team's coding standards
        - Reduces code review time by 60%
        
        Tone: Professional but approachable, emphasize ROI and team productivity.
        """
        
        mock_adapter.chat_completion.side_effect = [
            {
                "content": """Plan:
1. Hook: Open with relatable pain point (slow code reviews, bugs in production)
2. Introduce solution: CodeGuardian as AI-powered code review assistant
3. Key benefits: Time savings, bug prevention, team productivity
4. Social proof: Metrics and potential testimonial placeholder
5. Call-to-action: Free trial or demo signup""",
                "tokens": 90
            },
            {
                "content": """**Introducing CodeGuardian: Your AI-Powered Code Review Partner**

Every engineering manager knows the struggle: Pull requests pile up, reviews take days, and bugs still slip through to production. Your team is talented, but they're stretched thin. Code reviews become bottlenecks, and quality suffers.

What if you could catch 95% of bugs before they ever reach production—while cutting code review time by 60%?

**Meet CodeGuardian: The AI Code Review Tool Built for Fast-Moving Teams**

CodeGuardian is an AI-powered code review assistant that integrates seamlessly with your existing workflow. Whether your team uses GitHub, GitLab, or Bitbucket, CodeGuardian plugs in instantly and starts working alongside your developers.

**Here's How CodeGuardian Transforms Your Development Process:**

✓ **Catch Bugs Before Production:** Our AI analyzes every pull request with 95% accuracy, flagging potential issues, security vulnerabilities, and performance bottlenecks before they become costly problems.

✓ **Learn Your Team's Standards:** CodeGuardian doesn't just apply generic rules—it learns your team's coding conventions, style preferences, and architectural patterns, providing context-aware feedback that aligns with your standards.

✓ **Reduce Review Time by 60%:** By handling routine checks automatically, CodeGuardian frees your senior engineers to focus on architecture, logic, and mentoring—not syntax and formatting.

✓ **Seamless Integration:** Set up in under 5 minutes. No infrastructure changes, no workflow disruption. CodeGuardian works where your team already works.

**The ROI Is Clear:**

For a team of 10 engineers spending 5 hours/week on code reviews, CodeGuardian saves 30 hours per week—that's 1,560 hours per year. At an average engineering cost of $100/hour, that's $156,000 in reclaimed productivity annually.

Plus, catching bugs before production means fewer emergency fixes, less downtime, and happier customers.

**Join Engineering Teams Already Shipping Faster with CodeGuardian**

"CodeGuardian has become an essential part of our development workflow. It's like having an extra senior engineer reviewing every PR." — [Engineering Manager, Series B SaaS Company]

**Ready to Ship Faster and Safer?**

Start your free 14-day trial today—no credit card required. See why engineering teams at fast-growing startups trust CodeGuardian to maintain code quality while moving at startup speed.

👉 **[Start Free Trial]** | **[Schedule Demo]** | **[View Pricing]**

---

*CodeGuardian: Smarter Code Reviews, Faster Shipping.*""",
                "tokens": 580
            },
            {
                "content": """**FINAL POLISHED VERSION:**

**Introducing CodeGuardian: Your AI-Powered Code Review Partner**

Every engineering manager knows the struggle: Pull requests pile up, reviews take days, and bugs still slip through to production. Your team is talented, but stretched thin. Code reviews become bottlenecks, and quality suffers.

What if you could catch 95% of bugs before they reach production—while cutting review time by 60%?

**Meet CodeGuardian: The AI Code Review Tool Built for Fast-Moving Teams**

CodeGuardian is an AI-powered code review assistant that integrates seamlessly with your existing workflow. Whether your team uses GitHub, GitLab, or Bitbucket, CodeGuardian plugs in instantly and starts working alongside your developers.

**Here's How CodeGuardian Transforms Your Development Process:**

✓ **Catch Bugs Before Production:** Our AI analyzes every pull request with 95% accuracy, flagging potential issues, security vulnerabilities, and performance bottlenecks before they become costly problems.

✓ **Learn Your Team's Standards:** CodeGuardian doesn't just apply generic rules—it learns your team's coding conventions, style preferences, and architectural patterns, providing context-aware feedback that aligns with your standards.

✓ **Reduce Review Time by 60%:** By handling routine checks automatically, CodeGuardian frees your senior engineers to focus on architecture, logic, and mentoring—not syntax and formatting.

✓ **Seamless Integration:** Set up in under 5 minutes. No infrastructure changes, no workflow disruption. CodeGuardian works where your team already works.

**The ROI Is Clear:**

For a team of 10 engineers spending 5 hours/week on code reviews, CodeGuardian saves 30 hours per week—that's 1,560 hours per year. At an average engineering cost of $100/hour, that's $156,000 in reclaimed productivity annually.

Plus, catching bugs before production means fewer emergency fixes, less downtime, and happier customers.

**Join Engineering Teams Already Shipping Faster**

"CodeGuardian has become an essential part of our development workflow. It's like having an extra senior engineer reviewing every PR." — Engineering Manager, Series B SaaS Company

**Ready to Ship Faster and Safer?**

Start your free 14-day trial today—no credit card required. See why engineering teams at fast-growing startups trust CodeGuardian to maintain code quality while moving at startup speed.

👉 **[Start Free Trial]** | **[Schedule Demo]** | **[View Pricing]**

---

*CodeGuardian: Smarter Code Reviews, Faster Shipping.*

---

**Editorial Notes:**
- Strong hook addresses target audience pain points immediately
- ROI calculation provides concrete value proposition for decision-makers
- Feature presentation balances technical capability with business benefits
- CTA is clear with low-friction entry point (free trial, no credit card)
- Tone successfully balances professionalism with approachability
- Testimonial placeholder adds social proof without being overly salesy
- Tagline is memorable and reinforces core value proposition""",
                "tokens": 620
            }
        ]
        
        result = await service.collaborate(query, include_process=True)
        
        # Verify creative content quality
        assert "CodeGuardian" in result["final_response"]
        assert "95%" in result["final_response"]
        assert "60%" in result["final_response"]
        assert "ROI" in result["final_response"] or "productivity" in result["final_response"].lower()
        
        # Verify all three agents contributed
        assert len(result["collaboration_process"]["steps"]) == 3
        assert result["metadata"]["total_tokens"] == 1290
    
    @pytest.mark.asyncio
    async def test_error_handling_in_collaboration(self, service, mock_adapter):
        """
        Test that collaboration handles errors gracefully when an agent fails.
        """
        # Simulate planner success, but writer fails
        mock_adapter.chat_completion.side_effect = [
            {"content": "Plan: Step 1, Step 2, Step 3", "tokens": 50},
            Exception("API rate limit exceeded"),
        ]
        
        with pytest.raises(Exception) as exc_info:
            await service.collaborate("Test query")
        
        assert "rate limit" in str(exc_info.value).lower()
        # Verify only planner was called before failure
        assert mock_adapter.chat_completion.call_count == 2
    
    @pytest.mark.asyncio
    async def test_long_context_handling(self, service, mock_adapter):
        """
        Test collaboration with very long context to ensure proper handling.
        """
        # Create a very long query with substantial context
        long_query = "Analyze the following system architecture:\n" + "- Service detail\n" * 100
        long_context = "Historical context:\n" + "- Previous decision\n" * 50
        
        mock_adapter.chat_completion.side_effect = [
            {"content": "Comprehensive plan with 10 steps", "tokens": 200},
            {"content": "Detailed analysis covering all aspects", "tokens": 800},
            {"content": "Final review with recommendations", "tokens": 400}
        ]
        
        result = await service.collaborate(long_query, context=long_context, include_process=True)
        
        # Verify the system handled long input
        assert result["metadata"]["total_tokens"] == 1400
        assert mock_adapter.chat_completion.call_count == 3
        
        # Check that context was passed to all agents
        for call in mock_adapter.chat_completion.call_args_list:
            messages = call.kwargs["messages"]
            # At least one message should contain context reference
            assert any("Context:" in msg.get("content", "") for msg in messages if msg["role"] == "user")

