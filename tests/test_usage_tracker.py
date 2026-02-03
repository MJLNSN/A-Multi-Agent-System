"""
Unit tests for Usage Tracker Service.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from uuid import uuid4

from src.services.usage_tracker import UsageTracker, usage_tracker
from src.models.registry import MODEL_REGISTRY


class TestUsageTrackerCostCalculation:
    """Test cost calculation functionality."""
    
    @pytest.fixture
    def tracker(self):
        """Create a tracker instance."""
        return UsageTracker()
    
    def test_calculate_cost_gpt4(self, tracker):
        """Test cost calculation for GPT-4 Turbo."""
        cost = tracker.calculate_cost(
            model="openai/gpt-4-turbo",
            input_tokens=1000,
            output_tokens=500
        )
        
        # GPT-4 Turbo: $0.01/1k input, $0.03/1k output
        assert cost["input"] == 0.01  # 1000 * 0.01 / 1000
        assert cost["output"] == 0.015  # 500 * 0.03 / 1000
        assert cost["total"] == 0.025
    
    def test_calculate_cost_claude(self, tracker):
        """Test cost calculation for Claude 3.5 Sonnet."""
        cost = tracker.calculate_cost(
            model="anthropic/claude-3.5-sonnet",
            input_tokens=2000,
            output_tokens=1000
        )
        
        # Claude 3.5: $0.003/1k input, $0.015/1k output
        assert cost["input"] == 0.006  # 2000 * 0.003 / 1000
        assert cost["output"] == 0.015  # 1000 * 0.015 / 1000
        assert cost["total"] == 0.021
    
    def test_calculate_cost_gpt35(self, tracker):
        """Test cost calculation for GPT-3.5 Turbo."""
        cost = tracker.calculate_cost(
            model="openai/gpt-3.5-turbo",
            input_tokens=5000,
            output_tokens=2000
        )
        
        # GPT-3.5: $0.0005/1k input, $0.0015/1k output
        assert cost["input"] == 0.0025  # 5000 * 0.0005 / 1000
        assert cost["output"] == 0.003   # 2000 * 0.0015 / 1000
        assert cost["total"] == 0.0055
    
    def test_calculate_cost_unknown_model(self, tracker):
        """Test cost calculation for unknown model uses default pricing."""
        cost = tracker.calculate_cost(
            model="unknown/model",
            input_tokens=1000,
            output_tokens=1000
        )
        
        # Default: $0.002/1k input, $0.006/1k output
        assert cost["input"] == 0.002
        assert cost["output"] == 0.006
        assert cost["total"] == 0.008
    
    def test_calculate_cost_zero_tokens(self, tracker):
        """Test cost calculation with zero tokens."""
        cost = tracker.calculate_cost(
            model="openai/gpt-4-turbo",
            input_tokens=0,
            output_tokens=0
        )
        
        assert cost["input"] == 0
        assert cost["output"] == 0
        assert cost["total"] == 0
    
    def test_calculate_cost_large_values(self, tracker):
        """Test cost calculation with large token counts."""
        cost = tracker.calculate_cost(
            model="openai/gpt-4-turbo",
            input_tokens=100000,  # 100k tokens
            output_tokens=50000   # 50k tokens
        )
        
        # GPT-4 Turbo: $0.01/1k input, $0.03/1k output
        assert cost["input"] == 1.0     # 100000 * 0.01 / 1000
        assert cost["output"] == 1.5    # 50000 * 0.03 / 1000
        assert cost["total"] == 2.5


class TestUsageTrackerTracking:
    """Test usage tracking functionality with mocked database."""
    
    @pytest.fixture
    def tracker(self):
        """Create a tracker instance."""
        return UsageTracker()
    
    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        session = AsyncMock()
        session.add = MagicMock()
        session.commit = AsyncMock()
        return session
    
    @pytest.mark.asyncio
    async def test_track_usage_basic(self, tracker, mock_session):
        """Test basic usage tracking."""
        with patch('src.services.usage_tracker.async_session_maker') as mock_maker:
            mock_maker.return_value.__aenter__.return_value = mock_session
            mock_maker.return_value.__aexit__.return_value = AsyncMock()
            
            result = await tracker.track_usage(
                model="openai/gpt-4-turbo",
                input_tokens=100,
                output_tokens=50,
                operation_type="message"
            )
            
            assert "usage_id" in result
            assert result["model"] == "openai/gpt-4-turbo"
            assert result["input_tokens"] == 100
            assert result["output_tokens"] == 50
            assert result["total_tokens"] == 150
            assert result["operation_type"] == "message"
            assert "cost_usd" in result
            assert result["cost_usd"]["total"] > 0
    
    @pytest.mark.asyncio
    async def test_track_usage_with_thread(self, tracker, mock_session):
        """Test usage tracking with thread ID."""
        thread_id = str(uuid4())
        
        with patch('src.services.usage_tracker.async_session_maker') as mock_maker:
            mock_maker.return_value.__aenter__.return_value = mock_session
            mock_maker.return_value.__aexit__.return_value = AsyncMock()
            
            result = await tracker.track_usage(
                model="anthropic/claude-3.5-sonnet",
                input_tokens=200,
                output_tokens=100,
                operation_type="message",
                thread_id=thread_id,
                user_id="user123"
            )
            
            # Verify session.add was called
            assert mock_session.add.called
            
            # Check the result
            assert result["model"] == "anthropic/claude-3.5-sonnet"
            assert result["total_tokens"] == 300
    
    @pytest.mark.asyncio
    async def test_track_usage_collaboration(self, tracker, mock_session):
        """Test usage tracking for collaboration."""
        collaboration_id = "collab-abc123"
        
        with patch('src.services.usage_tracker.async_session_maker') as mock_maker:
            mock_maker.return_value.__aenter__.return_value = mock_session
            mock_maker.return_value.__aexit__.return_value = AsyncMock()
            
            result = await tracker.track_usage(
                model="openai/gpt-4-turbo",
                input_tokens=500,
                output_tokens=200,
                operation_type="collaboration",
                collaboration_id=collaboration_id,
                extra_data={"agent_role": "planner", "step": 1}
            )
            
            assert result["operation_type"] == "collaboration"
            assert result["total_tokens"] == 700
    
    @pytest.mark.asyncio
    async def test_track_usage_summarization(self, tracker, mock_session):
        """Test usage tracking for summarization."""
        with patch('src.services.usage_tracker.async_session_maker') as mock_maker:
            mock_maker.return_value.__aenter__.return_value = mock_session
            mock_maker.return_value.__aexit__.return_value = AsyncMock()
            
            result = await tracker.track_usage(
                model="openai/gpt-3.5-turbo",
                input_tokens=3000,
                output_tokens=500,
                operation_type="summarization"
            )
            
            assert result["operation_type"] == "summarization"
            assert result["total_tokens"] == 3500


class TestUsageTrackerAggregation:
    """Test usage aggregation and reporting functionality."""
    
    @pytest.fixture
    def tracker(self):
        """Create a tracker instance."""
        return UsageTracker()
    
    @pytest.fixture
    def mock_usage_records(self):
        """Create mock usage records for testing."""
        records = []
        models = ["openai/gpt-4-turbo", "anthropic/claude-3.5-sonnet"]
        operations = ["message", "collaboration", "summarization"]
        
        for i in range(10):
            record = MagicMock()
            record.usage_id = uuid4()
            record.model = models[i % len(models)]
            record.input_tokens = 100 * (i + 1)
            record.output_tokens = 50 * (i + 1)
            record.total_tokens = record.input_tokens + record.output_tokens
            record.operation_type = operations[i % len(operations)]
            record.cost_usd = {"input": 0.001, "output": 0.002, "total": 0.003}
            record.created_at = datetime.utcnow() - timedelta(days=i)
            records.append(record)
        
        return records
    
    @pytest.mark.asyncio
    async def test_get_usage_summary_empty(self, tracker):
        """Test getting usage summary with no data."""
        with patch('src.services.usage_tracker.async_session_maker') as mock_maker:
            mock_session = AsyncMock()
            mock_maker.return_value.__aenter__.return_value = mock_session
            mock_maker.return_value.__aexit__.return_value = AsyncMock()
            
            # Mock empty result
            mock_result = MagicMock()
            mock_result.one.return_value = MagicMock(
                total_input=0,
                total_output=0,
                total_tokens=0,
                request_count=0
            )
            mock_session.execute.return_value = mock_result
            
            summary = await tracker.get_usage_summary(days=30)
            
            assert summary["period_days"] == 30
            assert summary["summary"]["total_tokens"] == 0
            assert summary["summary"]["total_requests"] == 0
    
    @pytest.mark.asyncio
    async def test_get_thread_usage_invalid_id(self, tracker):
        """Test getting thread usage with invalid ID."""
        result = await tracker.get_thread_usage("invalid-uuid")
        
        assert "error" in result
        assert result["error"] == "Invalid thread ID"


class TestUsageTrackerModelComparison:
    """Test model comparison functionality."""
    
    @pytest.fixture
    def tracker(self):
        """Create a tracker instance."""
        return UsageTracker()
    
    def test_model_registry_has_costs(self):
        """Verify all models in registry have cost information."""
        for model_id, config in MODEL_REGISTRY.items():
            assert "cost_per_1k_tokens" in config, f"Model {model_id} missing cost info"
            assert "input" in config["cost_per_1k_tokens"]
            assert "output" in config["cost_per_1k_tokens"]
    
    def test_cost_comparison_gpt4_vs_gpt35(self, tracker):
        """Compare costs between GPT-4 and GPT-3.5."""
        gpt4_cost = tracker.calculate_cost(
            "openai/gpt-4-turbo",
            input_tokens=1000,
            output_tokens=500
        )
        
        gpt35_cost = tracker.calculate_cost(
            "openai/gpt-3.5-turbo",
            input_tokens=1000,
            output_tokens=500
        )
        
        # GPT-4 should be more expensive than GPT-3.5
        assert gpt4_cost["total"] > gpt35_cost["total"]
        
        # Calculate the cost ratio
        ratio = gpt4_cost["total"] / gpt35_cost["total"]
        assert ratio > 10  # GPT-4 is significantly more expensive
    
    def test_cost_comparison_gpt4_vs_claude(self, tracker):
        """Compare costs between GPT-4 and Claude 3.5."""
        gpt4_cost = tracker.calculate_cost(
            "openai/gpt-4-turbo",
            input_tokens=1000,
            output_tokens=500
        )
        
        claude_cost = tracker.calculate_cost(
            "anthropic/claude-3.5-sonnet",
            input_tokens=1000,
            output_tokens=500
        )
        
        # Both are premium models, verify costs are calculated correctly
        assert gpt4_cost["total"] > 0
        assert claude_cost["total"] > 0


class TestGlobalUsageTracker:
    """Test the global usage_tracker instance."""
    
    def test_global_instance_exists(self):
        """Test that global usage_tracker instance exists."""
        assert usage_tracker is not None
        assert isinstance(usage_tracker, UsageTracker)
    
    def test_global_instance_calculate_cost(self):
        """Test that global instance can calculate costs."""
        cost = usage_tracker.calculate_cost(
            "openai/gpt-4-turbo",
            input_tokens=100,
            output_tokens=50
        )
        
        assert "input" in cost
        assert "output" in cost
        assert "total" in cost


class TestUsageTrackerEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def tracker(self):
        """Create a tracker instance."""
        return UsageTracker()
    
    def test_negative_tokens_handled(self, tracker):
        """Test that negative token values are handled gracefully."""
        # In practice, negative tokens shouldn't occur, but the function should handle it
        cost = tracker.calculate_cost(
            "openai/gpt-4-turbo",
            input_tokens=-100,
            output_tokens=-50
        )
        
        # Should produce negative costs (edge case)
        assert cost["total"] < 0
    
    def test_very_small_token_count(self, tracker):
        """Test cost calculation with very small token counts."""
        cost = tracker.calculate_cost(
            "openai/gpt-4-turbo",
            input_tokens=1,
            output_tokens=1
        )
        
        # Should produce very small but non-zero cost
        assert cost["total"] > 0
        assert cost["total"] < 0.001
    
    def test_float_precision(self, tracker):
        """Test that cost calculations maintain appropriate precision."""
        cost = tracker.calculate_cost(
            "openai/gpt-4-turbo",
            input_tokens=123,
            output_tokens=456
        )
        
        # Verify costs are rounded appropriately
        assert isinstance(cost["input"], float)
        assert isinstance(cost["output"], float)
        assert isinstance(cost["total"], float)
        
        # Check precision (should be 6 decimal places)
        total_str = str(cost["total"])
        if '.' in total_str:
            decimals = len(total_str.split('.')[1])
            assert decimals <= 6


class TestUsageTrackerIntegration:
    """Integration tests for usage tracking with other components."""
    
    @pytest.fixture
    def tracker(self):
        """Create a tracker instance."""
        return UsageTracker()
    
    def test_collaboration_cost_estimation(self, tracker):
        """Test estimating cost for a typical collaboration workflow."""
        # Simulate a Planner → Writer → Reviewer workflow
        
        # Planner step (shorter response)
        planner_cost = tracker.calculate_cost(
            "openai/gpt-4-turbo",
            input_tokens=200,
            output_tokens=100
        )
        
        # Writer step (longer response)
        writer_cost = tracker.calculate_cost(
            "anthropic/claude-3.5-sonnet",
            input_tokens=500,
            output_tokens=800
        )
        
        # Reviewer step (medium response)
        reviewer_cost = tracker.calculate_cost(
            "openai/gpt-4-turbo",
            input_tokens=1300,
            output_tokens=400
        )
        
        total_cost = planner_cost["total"] + writer_cost["total"] + reviewer_cost["total"]
        
        # Verify total is reasonable for this workflow
        assert total_cost > 0
        assert total_cost < 1.0  # Should be less than $1 for a typical request
        
        # Writer with Claude should be cheaper than if using GPT-4
        writer_with_gpt4 = tracker.calculate_cost(
            "openai/gpt-4-turbo",
            input_tokens=500,
            output_tokens=800
        )
        assert writer_cost["total"] < writer_with_gpt4["total"]
    
    def test_message_thread_cost_estimation(self, tracker):
        """Test estimating cost for a conversation thread."""
        total_cost = 0
        
        # Simulate 10 messages in a thread
        for i in range(10):
            # Each message gets progressively longer as context grows
            input_tokens = 100 + (i * 50)  # Context grows
            output_tokens = 150  # Responses are roughly consistent
            
            cost = tracker.calculate_cost(
                "openai/gpt-4-turbo",
                input_tokens,
                output_tokens
            )
            total_cost += cost["total"]
        
        # Verify total cost is calculated correctly
        assert total_cost > 0
        # 10 messages shouldn't cost more than a few dollars
        assert total_cost < 5.0

