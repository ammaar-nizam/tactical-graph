"""
Configuration module for GraphRAG application using Pydantic Settings.

Reads database credentials, API keys, and LLM model specifications
from environment variables or root .env file.
"""

import logging
from pathlib import Path
from typing import Any, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# Locate root directory and .env file
_CURRENT_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _CURRENT_DIR.parent
_ENV_FILE_PATH = _ROOT_DIR / ".env"


class GraphRAGSettings(BaseSettings):
    """
    GraphRAG Application settings loaded from environment variables or .env file.
    """

    # Neo4j Database Connection Credentials
    NEO4J_URI: str = Field(
        default="neo4j+s://4446f270.databases.neo4j.io",
        description="Neo4j instance Bolt/SRAM connection URI",
    )
    NEO4J_USER: str = Field(
        default="neo4j",
        description="Neo4j database authentication username",
    )
    NEO4J_PASSWORD: str = Field(
        default="",
        description="Neo4j database authentication password",
    )
    NEO4J_DATABASE: str = Field(
        default="neo4j",
        description="Target Neo4j database name",
    )

    # Google AI Studio / Gemini API Settings
    GEMINI_API_KEY: Optional[str] = Field(
        default=None,
        description="Google AI Studio Gemini API Key",
    )
    GRAPH_LLM_MODEL: str = Field(
        default="gemini-3.6-flash",
        description="LLM model identifier for GraphRAG agent",
    )

    model_config = SettingsConfigDict(
        env_file=(_ENV_FILE_PATH, ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    def model_post_init(self, __context: Any) -> None:
        """
        Log non-sensitive configuration status post-initialization.
        """
        masked_key = (
            f"{self.GEMINI_API_KEY[:6]}...{self.GEMINI_API_KEY[-4:]}"
            if self.GEMINI_API_KEY and len(self.GEMINI_API_KEY) > 10
            else "<NOT_SET>"
        )
        logger.info("GraphRAG Settings initialized.")
        logger.info("  Neo4j URI: %s", self.NEO4J_URI)
        logger.info("  LLM Model: %s", self.GRAPH_LLM_MODEL)
        logger.info("  Gemini API Key: %s", masked_key)


# Global settings instance
settings = GraphRAGSettings()
