# Multi-Agent Chat Threading System

[![CI](https://github.com/MJLNSN/A-Multi-Agent-System/actions/workflows/ci.yml/badge.svg)](https://github.com/MJLNSN/A-Multi-Agent-System/actions/workflows/ci.yml)

A production-ready multi-agent chat service demonstrating threaded context management, LLM orchestration via OpenRouter, and intelligent auto-summarization.

## Features

- **Thread-based Conversations** - Persistent context with customizable system prompts
- **Multi-LLM Support** - GPT-4, Claude 3.5, GPT-3.5 via OpenRouter
- **Real-time Model Switching** - Switch models mid-conversation
- **Auto-Summarization** - Compress context every 10 messages
- **🚀 Multi-Agent Collaboration** - Planner → Writer → Reviewer pattern
- **💰 Token & Cost Tracking** - Enterprise-grade usage monitoring

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- [OpenRouter API Key](https://openrouter.ai/)

### Setup

```bash
# Clone repository
git clone https://github.com/MJLNSN/A-Multi-Agent-System.git
cd A-Multi-Agent-System

# Configure environment
cp env.example .env
# Edit .env and add your OPENROUTER_API_KEY

# Start services (first time with --install)
cd scripts
./start.sh --install

# Or without dependency installation
./start.sh
```

**Access**: http://localhost:8001/docs

### Stop Services

```bash
cd scripts
./stop.sh
```

## API Overview

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/threads` | POST | Create new thread |
| `/api/threads/{id}` | GET | Get thread details |
| `/api/threads/{id}` | PATCH | Update thread config |
| `/api/threads` | GET | List all threads |
| `/api/threads/{id}/messages` | POST | Send message |
| `/api/threads/{id}/messages` | GET | Get message history |
| `/api/threads/{id}/summaries` | GET | Get summaries |

### Multi-Agent Collaboration

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/collaborate` | POST | Execute multi-agent collaboration |
| `/api/agents` | GET | List collaboration agents |
| `/api/agents/{role}` | PATCH | Update agent's model |

### Usage Tracking

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/usage/summary` | GET | Get usage & cost summary |
| `/api/usage/thread/{id}` | GET | Get thread usage |
| `/api/usage/daily` | GET | Get daily trends |
| `/api/usage/models` | GET | Get model comparison |
| `/api/usage/pricing` | GET | Get model pricing |

## Multi-Agent Collaboration

True multi-agent orchestration where specialized AI agents work together:

```
Simple Query:  User → [Planner] → [Writer] → Final Response


Complex Query: User → [Planner] → [Writer] → [Reviewer] → Final Response
                       GPT-4      Claude       GPT-4
```

| Agent | Model | Role |
|-------|-------|------|
| **Planner** | GPT-4 | Analyzes question, creates response strategy |
| **Writer** | Claude 3.5 | Generates detailed content following plan |
| **Reviewer** | GPT-4 | Reviews and polishes (complex queries only) |


### Example

```bash
# Simple query (Reviewer skipped automatically)
curl -X POST http://localhost:8001/api/collaborate \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?"}'

# Complex query (full pipeline)
curl -X POST http://localhost:8001/api/collaborate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "1. Market strategy 2. Competitive advantage 3. Success metrics",
    "context": "We are a startup launching a SaaS product",
    "include_process": true
  }'

# Force full pipeline on simple query
curl -X POST http://localhost:8001/api/collaborate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is AI?",
    "force_full_pipeline": true
  }'
```

## Usage Examples

### Create Thread & Send Message

```bash
# Create thread
curl -X POST http://localhost:8001/api/threads \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Product Discussion",
    "system_prompt": "You are a helpful product consultant.",
    "current_model": "openai/gpt-4-turbo"
  }'

# Send message (replace {thread_id} with actual ID)
curl -X POST http://localhost:8001/api/threads/{thread_id}/messages \
  -H "Content-Type: application/json" \
  -d '{"content": "Hello! What can you help me with?"}'

# Switch model mid-conversation
curl -X POST http://localhost:8001/api/threads/{thread_id}/messages \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Continue with Claude",
    "model": "anthropic/claude-3.5-sonnet"
  }'
```

### Model Selection Rule

```
Effective Model = message.model (if provided) || thread.current_model
```

## Available Models

| Model | Provider | Context Window |
|-------|----------|----------------|
| `openai/gpt-4-turbo` | OpenAI | 128K |
| `anthropic/claude-3.5-sonnet` | Anthropic | 200K |
| `openai/gpt-3.5-turbo` | OpenAI | 16K |

## Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENROUTER_API_KEY` | Yes | - | OpenRouter API key |
| `PORT` | No | 8001 | API server port |
| `DATABASE_URL` | No | Auto | PostgreSQL connection |
| `DEFAULT_MODEL` | No | `openai/gpt-4-turbo` | Default LLM |
| `SUMMARIZATION_MESSAGE_THRESHOLD` | No | 10 | Messages before summary |

See `env.example` for all configuration options.

## Testing

```bash
cd scripts

# Quick unit tests
./run_tests.sh quick

# All tests
./run_tests.sh all

# E2E tests (requires running API)
./run_tests.sh e2e

# Coverage report
./run_tests.sh coverage
```

**Test Stats**: 100+ tests, ~80% coverage

## Project Structure

```
├── src/
│   ├── main.py              # FastAPI app entry
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
│   ├── start.sh             # Start services
│   ├── stop.sh              # Stop services
│   └── run_tests.sh         # Test runner
├── docs/                    # Documentation
├── docker-compose.yml       # Docker config
└── env.example              # Environment template
```

## Architecture

```
┌─────────────────────────────────────────────┐
│              FastAPI Gateway                │
│  /threads  /messages  /collaborate  /usage  │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│           Business Logic                    │
│  ThreadManager │ MessageHandler │ LLM Orch  │
│  AgentCollaboration │ UsageTracker          │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│        OpenRouter Adapter                   │
│  (GPT-4 / Claude 3.5 / GPT-3.5)            │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│          PostgreSQL                         │
│  threads │ messages │ summaries │ usage     │
└─────────────────────────────────────────────┘
```

## Key Design Decisions

1. **Multi-Agent Pattern**: LLMs as pluggable agents sharing thread context
2. **Async Summarization**: Non-blocking, triggered every 10 messages
3. **Context Management**: System prompt + summary + recent messages (token-aware)
4. **Concurrency Control**: Thread-level async locks

## Docker Deployment

```bash
# Set API key
export OPENROUTER_API_KEY=your-key

# Start all services
docker-compose up -d

# Access: http://localhost:8000/docs
```

## Documentation

- [API Docs](http://localhost:8001/docs) - Interactive Swagger UI

## License

MIT
