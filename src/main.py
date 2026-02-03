"""
Main application entry point for the Multi-Agent Chat Threading System.
Initializes FastAPI app with all routes and services.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os

from src.database import init_db, close_db
from src.adapters.openrouter import OpenRouterAdapter
from src.services.thread_manager import ThreadManager
from src.services.message_handler import MessageHandler
from src.services.llm_orchestrator import LLMOrchestrator
from src.services.summarization_engine import SummarizationEngine
from src.routes import threads_router, messages_router, summaries_router
from src.config import settings
from src.utils.logging import setup_logging, get_logger
from src.models.registry import list_available_models

# Initialize logging
setup_logging()
logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown tasks.
    """
    # Startup
    logger.info("application_starting", env=settings.env)
    
    # Initialize database
    await init_db()
    logger.info("database_initialized")
    
    # Initialize OpenRouter adapter
    app.state.openrouter = OpenRouterAdapter(
        api_key=settings.openrouter_api_key
    )
    logger.info("openrouter_adapter_initialized")
    
    # Initialize services
    app.state.thread_manager = ThreadManager()
    app.state.summarizer = SummarizationEngine(app.state.openrouter)
    app.state.llm_orchestrator = LLMOrchestrator(app.state.openrouter)
    app.state.message_handler = MessageHandler(
        app.state.llm_orchestrator,
        app.state.summarizer
    )
    logger.info("services_initialized")
    
    logger.info(
        "application_started",
        available_models=list_available_models()
    )
    
    yield
    
    # Shutdown
    logger.info("application_shutting_down")
    
    await app.state.openrouter.close()
    await close_db()
    
    logger.info("application_stopped")


# Create FastAPI application
app = FastAPI(
    title="Multi-Agent Chat Threading System",
    description="""
    A multi-agent chat service that handles chat threads with persistent context,
    supports multiple LLMs via OpenRouter API, and implements auto-summarization.
    
    ## Features
    - Thread-based conversation management
    - Multiple LLM support (GPT-4, Claude, GPT-3.5)
    - Real-time model switching within conversations
    - Automatic conversation summarization
    - Intelligent context management
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(threads_router, prefix="/api")
app.include_router(messages_router, prefix="/api")
app.include_router(summaries_router, prefix="/api")


@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "message": "Multi-Agent Chat Threading System",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint for monitoring.
    """
    return {
        "status": "healthy",
        "version": "1.0.0"
    }


@app.get("/models", tags=["Models"])
async def list_models():
    """
    List all available LLM models.
    """
    from src.models.registry import MODEL_REGISTRY
    
    models = []
    for model_id, config in MODEL_REGISTRY.items():
        models.append({
            "id": model_id,
            "display_name": config["display_name"],
            "provider": config["provider"],
            "context_window": config["context_window"]
        })
    
    return {"models": models}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.env == "development"
    )

