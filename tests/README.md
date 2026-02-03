# Test Documentation

## Test Pyramid

```
        /\
       /E2E\         17 tests (slow, real API)
      /------\
     / Integ  \      17 tests (medium)
    /----------\
   / Unit Tests \    68 tests (fast)
  /--------------\
```

## Quick Commands

```bash
# Use test runner (recommended)
./run_tests.sh quick        # Fast unit tests
./run_tests.sh all          # All pytest tests
./run_tests.sh e2e          # E2E with real API
./run_tests.sh coverage     # Generate coverage

# Or use pytest directly
pytest tests/ -v
pytest tests/test_token_counter.py -v
```

## Test Categories

### Unit Tests
| File | Focus | Tests |
|------|-------|-------|
| `test_model_registry.py` | Model validation | 12 |
| `test_token_counter.py` | Token counting | 10 |
| `test_openrouter_adapter.py` | API adapter | 10 |
| `test_thread_manager.py` | Thread CRUD | 3 |
| `test_llm_orchestrator.py` | Model selection | 3 |
| `test_summarization_engine.py` | Summarization | 9 |
| `test_message_handler.py` | Message processing | 7 |

### Integration Tests
| File | Focus | Tests |
|------|-------|-------|
| `test_api.py` | API endpoints | 9 |
| `test_integration.py` | Full flows | 8 |

### Performance Tests
| File | Focus | Tests |
|------|-------|-------|
| `test_performance.py` | Benchmarks, stress, concurrency | 14 |

### E2E Tests
- **Location**: `e2e_test.sh`
- **Tests**: 17 (real OpenRouter API calls)
- **Logs**: `logs/e2e_results_*.log`
- **Requires**: Running API (`./start.sh`)

## Coverage

**Target**: >80%  
**Current**: ~75%

```bash
# Generate HTML report
./run_tests.sh coverage
open htmlcov/index.html
```

## Writing Tests

### Unit Test Example
```python
@pytest.mark.asyncio
async def test_feature():
    result = await some_function()
    assert result == expected
```

### Markers
```python
@pytest.mark.slow          # Performance tests
@pytest.mark.integration   # Integration tests
```

## CI/CD Recommendations

```yaml
# Fast (PR checks)
./run_tests.sh quick

# Full (merge checks)
./run_tests.sh all

# E2E (pre-deploy)
./run_tests.sh e2e
```
