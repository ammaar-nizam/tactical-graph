"""
Pydantic data models for structured inputs, LLM outputs, REST API request/responses, and GraphRAG responses.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CypherGenerationOutput(BaseModel):
    """
    Structured output returned by the Text-to-Cypher LLM tool.
    """

    cypher_query: str = Field(
        ...,
        description="The generated read-only Cypher query matching the Neo4j schema.",
    )
    explanation: str = Field(
        ...,
        description="Brief natural language explanation of how the query satisfies the prompt.",
    )
    is_read_only: bool = Field(
        default=True,
        description="Boolean indicating whether the Cypher query contains only MATCH/RETURN operations.",
    )


class ReplacementCandidatesInput(BaseModel):
    """
    Structured input parameters for the get_replacement_candidates dedicated template tool.
    """

    target_club: str = Field(
        ...,
        description="Name of the target club (e.g. 'Manchester United').",
        examples=["Manchester United", "Barcelona", "Real Madrid"],
    )
    season: int = Field(
        ...,
        description="The starting year of the season to analyze (e.g. 2014 for 2014/15).",
        examples=[2012, 2013, 2014, 2017],
    )
    position: str = Field(
        ...,
        description="Player primary position (e.g. 'Attack', 'Midfield', 'Defender').",
        examples=["Attack", "Midfield", "Defender"],
    )
    benchmark_player: str = Field(
        ...,
        description="Name of the benchmark transfer target player to include for comparison (e.g. 'Falcao', 'Fellaini').",
        examples=["Falcao", "Fellaini", "Coutinho"],
    )
    min_minutes: int = Field(
        default=1000,
        description="Minimum match minutes played by candidates in the given season.",
    )


class GraphRAGResponse(BaseModel):
    """
    Structured response payload returned by the GraphRAG agent API.
    """

    answer: str = Field(
        ...,
        description="Natural language scout report or answer synthesized from graph query results.",
    )
    cypher_used: Optional[str] = Field(
        default=None,
        description="The exact Cypher query executed against Neo4j (if applicable).",
    )


class QueryRequest(BaseModel):
    """
    Pydantic request payload for /chat API endpoint.
    """

    query: str = Field(
        ...,
        description="Natural language question to ask the GraphRAG agent.",
        examples=[
            "Who would have been a better centre forward for Manchester United instead of Radamel Falcao in 2014/15 season?",
            "List top 5 transfers in history",
        ],
    )


class HealthResponse(BaseModel):
    """
    Pydantic response payload for /health API endpoint.
    """

    status: str = Field(..., description="API operational status ('ok').")
    neo4j_connected: bool = Field(..., description="Boolean indicating live Neo4j driver connectivity.")
    model: str = Field(..., description="Active LLM model identifier.")
