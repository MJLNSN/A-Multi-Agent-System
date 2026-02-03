# Multi-Agent Chat Threading System

A production-ready multi-agent chat service demonstrating threaded context management, LLM orchestration via OpenRouter, and intelligent auto-summarization.

## Features

- **Thread-based Conversations** - Persistent context with system prompts
- **Multi-LLM Support** - GPT-4, Claude 3.5, GPT-3.5 via OpenRouter
- **Real-time Model Switching** - Switch models mid-conversation
- **Auto-Summarization** - Compress context every 10 messages
- **Token Management** - Intelligent context trimming

## Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- [OpenRouter API Key](https://openrouter.ai/)

### Setup

```bash
# Clone and configure
git clone <repo-url> && cd multi-agent-chat
cp env.example .env
# Edit .env: add OPENROUTER_API_KEY

cd scripts/

# Start (first time)
./start.sh --install

# Start (subsequent)
./start.sh

# Stop
./stop.sh
```

**Access**: http://localhost:8001/docs

## API Overview

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/threads` | POST | Create thread |
| `/api/threads/{id}` | GET | Get thread |
| `/api/threads/{id}` | PATCH | Update model/title |
| `/api/threads` | GET | List threads |
| `/api/threads/{id}/messages` | POST | Send message |
| `/api/threads/{id}/messages` | GET | Get history |
| `/api/threads/{id}/summaries` | GET | Get summaries |

### Example

```bash
# Create thread
curl -X POST http://localhost:8001/api/threads \
  -H "Content-Type: application/json" \
  -d '{"title": "Demo", "system_prompt": "You are helpful.", "current_model": "openai/gpt-4-turbo"}'

# Send message (returns thread_id like "abc-123")
curl -X POST http://localhost:8001/api/threads/abc-123/messages \
  -H "Content-Type: application/json" \
  -d '{"content": "Hello!"}'

# Switch model mid-conversation
curl -X POST http://localhost:8001/api/threads/abc-123/messages \
  -H "Content-Type: application/json" \
  -d '{"content": "Continue with Claude", "model": "anthropic/claude-3.5-sonnet"}'
```

## Available Models

| Model | Provider | Context |
|-------|----------|---------|
| `openai/gpt-4-turbo` | OpenAI | 128K |
| `anthropic/claude-3.5-sonnet` | Anthropic | 200K |
| `openai/gpt-3.5-turbo` | OpenAI | 16K |

## Model Selection

```
Effective Model = message.model (if provided) || thread.current_model
```

## Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENROUTER_API_KEY` | Yes | - | OpenRouter API key |
| `PORT` | No | 8001 | API server port |
| `DATABASE_URL` | No | Auto | PostgreSQL connection |
| `DEFAULT_MODEL` | No | `openai/gpt-4-turbo` | Default LLM |
| `SUMMARIZATION_MESSAGE_THRESHOLD` | No | 10 | Messages before summary |

## Testing

```bash
# Quick unit tests
./run_tests.sh quick

# All tests
./run_tests.sh all

# E2E tests (requires API running)
./run_tests.sh e2e

# Coverage report
./run_tests.sh coverage
```

**Test Stats**: 102 tests (85 pytest + 17 E2E), ~75% coverage

## Project Structure

```
├── src/
│   ├── main.py              # FastAPI app
│   ├── config.py            # Environment config
│   ├── constants.py         # Prompts & constants
│   ├── database.py          # PostgreSQL models
│   ├── routes/              # API endpoints
│   ├── services/            # Business logic
│   ├── adapters/            # OpenRouter client
│   ├── models/              # Registry & schemas
│   └── utils/               # Token counter, logging
├── tests/                   # Test suite
├── scripts/                 # Shell scripts
├── start.sh                 # Start services
├── stop.sh                  # Stop services
├── run_tests.sh             # Test runner
└── design.md                # Architecture docs
```

## Architecture

```
┌─────────────────────────────────────────────┐
│              FastAPI Gateway                │
│  /threads  /messages  /summaries            │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│           Business Logic                    │
│  ThreadManager │ MessageHandler │ LLM Orch  │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│        OpenRouter Adapter                   │
│  (GPT-4 / Claude 3.5 / GPT-3.5)            │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│          PostgreSQL                         │
│  threads │ messages │ summaries             │
└─────────────────────────────────────────────┘
```

## Key Design Decisions

1. **Multi-Agent Pattern**: LLMs as pluggable agents sharing thread context
2. **Async Summarization**: Non-blocking, triggered every 10 messages
3. **Context Management**: System prompt + summary + recent messages (token-aware)
4. **Concurrency Control**: Thread-level async locks

## Docker Deployment

```bash
export OPENROUTER_API_KEY=your-key
docker-compose up -d
# Access: http://localhost:8000/docs
```

## License

MIT
