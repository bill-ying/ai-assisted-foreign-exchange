#!/usr/bin/env python3
"""
FastAPI REST API server for the AI-Assisted FX Rate Lookup.

Endpoints
---------
POST /chat
    Body:  {"message": "<natural language query>"}
    Returns: {"answer": "<response text>", "model": "<model-id>"}

GET /health
    Returns: {"status": "ok"}

Run with:
    uvicorn api:app --reload --host 0.0.0.0 --port 8000
"""

import logging
import sys

from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared assistant instance (initialised once at startup)
# ---------------------------------------------------------------------------

from ai import FxAssistant  # noqa: E402  (import after load_dotenv)

_assistant: Optional[FxAssistant] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create the assistant on startup and close it on shutdown."""
    global _assistant
    logger.info("Initialising FxAssistant …")
    _assistant = FxAssistant.create()
    logger.info("FxAssistant ready. Models: %s", FxAssistant.MODELS)
    yield
    if _assistant is not None:
        _assistant.close()
        logger.info("FxAssistant closed.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI-Assisted FX Rate API",
    description=(
        "Natural-language USD/CAD exchange rate lookup powered by "
        "Bank of Canada data via OpenRouter free-tier LLMs with automatic "
        "model fallback on HTTP 429."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        description="Natural-language question about USD/CAD exchange rates.",
        examples=["How much CAD was for USD$1 on 2024-01-15?"],
    )


class ChatResponse(BaseModel):
    answer: str = Field(description="The assistant's response.")
    model: str = Field(description="OpenRouter model ID that generated the answer.")


class HealthResponse(BaseModel):
    status: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["Utility"])
def health() -> HealthResponse:
    """Simple liveness check."""
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse, tags=["FX Assistant"])
def chat(request: ChatRequest) -> ChatResponse:
    """
    Send a natural-language question and receive an exchange-rate answer.

    The assistant maintains **no per-request session state** — each call is
    independent. Conversation history is cleared between requests.

    If the primary model is rate-limited (HTTP 429), the server automatically
    retries with the next model in the fallback list before responding.
    """
    if _assistant is None:
        raise HTTPException(status_code=503, detail="Assistant not initialised.")

    # Clear history so each HTTP request is stateless
    _assistant.clear_history()

    result = _assistant.chat(request.message)
    return ChatResponse(answer=result["answer"], model=result["model"])
