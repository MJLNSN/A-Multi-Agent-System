"""
Constants and default prompts for the Multi-Agent Chat Threading System.
Contains non-sensitive configuration that doesn't need to be in environment variables.
"""

# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant. You provide accurate, 
thoughtful, and well-structured responses. When you don't know something, 
you say so honestly."""

SUMMARIZATION_PROMPT_TEMPLATE = """Summarize the following conversation segment concisely.
Focus on:
- Key topics discussed
- Important decisions or conclusions made
- User's main intent and questions
- Any action items or follow-ups mentioned

Keep the summary under 150 words and maintain the essential context.

Conversation:
{conversation}

Summary:"""

# =============================================================================
# API CONFIGURATION
# =============================================================================

# API versioning
API_VERSION = "1.0.0"
API_PREFIX = "/api"

# Pagination defaults
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100

# =============================================================================
# THREAD CONFIGURATION
# =============================================================================

# Thread status options
THREAD_STATUS_ACTIVE = "active"
THREAD_STATUS_ARCHIVED = "archived"
THREAD_STATUS_DELETED = "deleted"
THREAD_STATUSES = [THREAD_STATUS_ACTIVE, THREAD_STATUS_ARCHIVED, THREAD_STATUS_DELETED]

# Message roles
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_SYSTEM = "system"
MESSAGE_ROLES = [ROLE_USER, ROLE_ASSISTANT, ROLE_SYSTEM]

# =============================================================================
# SUMMARIZATION CONFIGURATION
# =============================================================================

# Summary trigger reasons
TRIGGER_MESSAGE_COUNT = "message_count"
TRIGGER_TOKEN_THRESHOLD = "token_threshold"
TRIGGER_MANUAL = "manual"
SUMMARY_TRIGGERS = [TRIGGER_MESSAGE_COUNT, TRIGGER_TOKEN_THRESHOLD, TRIGGER_MANUAL]

# Summary generation settings
SUMMARY_MAX_WORDS = 150
SUMMARY_TEMPERATURE = 0.3  # Lower temperature for consistent summaries
SUMMARY_MAX_TOKENS = 200

# Context assembly
CONTEXT_SUMMARY_PREFIX = "[Previous conversation summary]: "

# =============================================================================
# TOKEN MANAGEMENT
# =============================================================================

# Per-message token overhead (for role, formatting, etc.)
MESSAGE_TOKEN_OVERHEAD = 4

# Base overhead for conversation
CONVERSATION_TOKEN_OVERHEAD = 3

# Approximate characters per token for non-tiktoken models
CHARS_PER_TOKEN_ESTIMATE = 4

# =============================================================================
# ERROR MESSAGES
# =============================================================================

ERROR_THREAD_NOT_FOUND = "Thread not found"
ERROR_INVALID_MODEL = "Invalid model: {model}. Available models: {available}"
ERROR_INVALID_THREAD_ID = "Invalid thread ID format"
ERROR_RATE_LIMIT = "Rate limit exceeded. Retry after: {retry_after}"
ERROR_AUTH_FAILED = "Authentication failed. Please check your API key."
ERROR_OPENROUTER_TIMEOUT = "OpenRouter API timeout for model {model}"

# =============================================================================
# HTTP HEADERS FOR OPENROUTER
# =============================================================================

OPENROUTER_REFERER = "https://multi-agent-chat.local"
OPENROUTER_TITLE = "Multi-Agent Chat Threading System"

