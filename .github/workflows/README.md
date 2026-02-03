# GitHub Actions CI

## CI Pipeline Overview

This project uses GitHub Actions for continuous integration to ensure code quality and test coverage.

### What Gets Tested

The CI pipeline runs **72 tests** that don't require external API keys:

| Test Suite | Tests | Description |
|------------|-------|-------------|
| `test_model_registry.py` | 12 | Model validation and configuration |
| `test_token_counter.py` | 10 | Token counting and context trimming |
| `test_openrouter_adapter.py` | 10 | OpenRouter adapter (mocked) |
| `test_thread_manager.py` | 3 | Thread CRUD operations |
| `test_llm_orchestrator.py` | 3 | Model selection logic |
| `test_summarization_engine.py` | 9 | Summary generation |
| `test_message_handler.py` | 7 | Message processing |
| `test_performance.py` | 14 | Performance benchmarks |
| **Total** | **68+** | **All use mocks, no API keys needed** |

### What's Skipped

These tests require real API keys and are **only run locally**:

- `test_integration.py` (8 tests) - Full API integration
- `test_api.py` (9 tests) - API endpoint tests
- `tests/e2e_test.sh` (17 tests) - End-to-end with real OpenRouter API

### CI Workflow

1. **Checkout code**
2. **Setup Python 3.11**
3. **Cache pip packages**
4. **Install dependencies**
5. **Run linting** (flake8)
6. **Run unit tests** (72 tests, ~17s)
7. **Display summary**

### Running Locally

To run the same tests as CI:

```bash
pytest tests/test_model_registry.py \
       tests/test_token_counter.py \
       tests/test_openrouter_adapter.py \
       tests/test_thread_manager.py \
       tests/test_llm_orchestrator.py \
       tests/test_summarization_engine.py \
       tests/test_message_handler.py \
       tests/test_performance.py \
       -v
```

### Badge

The CI status badge in README.md shows the current build status:

```markdown
[![CI](https://github.com/MJLNSN/A-Multi-Agent-System/actions/workflows/ci.yml/badge.svg)](https://github.com/MJLNSN/A-Multi-Agent-System/actions/workflows/ci.yml)
```

### Why Some Tests Are Skipped

Integration and E2E tests require:
- Valid OpenRouter API key
- Real API calls (costs money)
- External service availability

These tests are best run:
- Locally before commits
- In staging environment
- Not on every push to GitHub

