"""
FastAPI application entry point for TacticalGraph GraphRAG API.

Provides RESTful HTTP endpoints for health monitoring, natural language GraphRAG queries,
and direct tactical player scouting without LLM orchestration overhead.
"""

import io
import logging
import sys
from typing import Any, Dict, List

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from agent import query_graphrag
from config import settings
from database import close_neo4j_driver, get_neo4j_driver
from models import GraphRAGResponse, HealthResponse, QueryRequest, ReplacementCandidatesInput
from tools.dedicated_templates import execute_replacement_candidates_query

# Configure UTF-8 output encoding for Windows compatibility
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager handling startup and graceful driver shutdown.
    """
    yield
    close_neo4j_driver()


# Initialize FastAPI application with lifespan context manager
app = FastAPI(
    title="TacticalGraph GraphRAG API",
    version="1.0.0",
    description="GraphRAG LLM & Analytics REST API for Tactical Football Knowledge Graph",
    lifespan=lifespan,
)

# Enable CORS for local frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check() -> HealthResponse:
    """
    Health check endpoint returning API status, Neo4j connectivity, and active model.
    """
    neo4j_connected = False
    try:
        driver = get_neo4j_driver()
        driver.verify_connectivity()
        neo4j_connected = True
    except Exception as e:
        logger.warning("Neo4j connectivity check failed during /health: %s", e)

    return HealthResponse(
        status="ok",
        neo4j_connected=neo4j_connected,
        model=settings.LLM_MODEL,
    )



@app.post("/chat/{thread_id}", response_model=GraphRAGResponse, tags=["GraphRAG Agent"])
def execute_graphrag_query(request: QueryRequest, thread_id: str = "default") -> GraphRAGResponse:
    """
    Natural Language GraphRAG query endpoint.

    Translates user prompt to Cypher or routes to tactical tools, executes query against Neo4j,
    and returns a structured response payload containing natural language answer and Cypher used.
    """
    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty.",
        )

    try:
        logger.info("Received API /chat request (thread_id: '%s'): '%s'", thread_id, request.query)
        response = query_graphrag(request.query, thread_id=thread_id)
        return response
    except Exception as e:
        logger.error("Error executing /chat endpoint: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process GraphRAG query: {str(e)}",
        )


@app.post("/scout/replacements", response_model=List[Dict[str, Any]], tags=["Dedicated Scouting"])
def scout_replacement_candidates(payload: ReplacementCandidatesInput) -> List[Dict[str, Any]]:
    """
    Dedicated tactical scouting endpoint for replacement candidate analysis.

    Invokes get_replacement_candidates directly for fast scouting responses without full LLM orchestration overhead.
    """
    try:
        logger.info("Received API /scout/replacements request: %s", payload.model_dump())
        records = execute_replacement_candidates_query(payload.model_dump())
        return records
    except Exception as e:
        logger.error("Error executing /scout/replacements endpoint: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute replacement scouting query: {str(e)}",
        )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
