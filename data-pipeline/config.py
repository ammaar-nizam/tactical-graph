"""
Configuration module for TacticalGraph data pipeline.

Loads application settings and credentials from environment variables or .env files
using Pydantic BaseSettings for strict type validation and defaults management.
"""

from functools import lru_cache
import logging
import os
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# Compute project root directory to ensure .env is loaded regardless of working directory
_BASE_DIR = Path(__file__).resolve().parent.parent
_ENV_FILE_PATH = _BASE_DIR / ".env"


def mask_neo4j_uri(uri: str) -> str:
    """
    Mask sensitive host/id parts of Neo4j connection URI for safe startup logging.

    Example:
        'neo4j+s://4446f270.databases.neo4j.io' -> 'neo4j+s://4446****.databases.neo4j.io'
        'bolt://localhost:7687'                -> 'bolt://localhost:7687'
    """
    if not uri:
        return "<empty>"
    try:
        parsed = urlparse(uri)
        scheme = parsed.scheme or "bolt"
        hostname = parsed.hostname or "localhost"
        port = f":{parsed.port}" if parsed.port else ""

        if hostname in ("localhost", "127.0.0.1"):
            masked_host = hostname
        elif "." in hostname:
            parts = hostname.split(".", 1)
            prefix = parts[0]
            masked_prefix = prefix[:4] + "****" if len(prefix) > 4 else "****"
            masked_host = f"{masked_prefix}.{parts[1]}"
        else:
            masked_host = hostname[:4] + "****" if len(hostname) > 4 else "****"

        return f"{scheme}://{masked_host}{port}"
    except Exception:
        return "neo4j+s://****"


class Settings(BaseSettings):
    """
    Application configuration settings loaded from environment variables or .env file.

    Adheres to SOLID principles by segregating configuration concerns into a single,
    strongly typed settings model.
    """

    # Neo4j Database Connection Credentials & Endpoint
    NEO4J_URI: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j connection URI (e.g., bolt://localhost:7687 or neo4j+s://...)",
    )
    NEO4J_USER: str = Field(
        default="neo4j",
        description="Username for Neo4j authentication",
    )
    NEO4J_PASSWORD: str = Field(
        default="password",
        description="Password for Neo4j authentication",
    )
    NEO4J_DATABASE: str = Field(
        default="neo4j",
        description="Default Neo4j database instance name",
    )

    # Neo4j Driver Connection Pool Configuration
    NEO4J_MAX_CONNECTION_POOL_SIZE: int = Field(
        default=50,
        description="Maximum number of connections allowed in the driver connection pool",
    )
    NEO4J_CONNECTION_TIMEOUT: float = Field(
        default=30.0,
        description="Connection acquisition timeout in seconds",
    )

    # Kaggle Dataset Configuration
    KAGGLE_API_TOKEN: Optional[str] = Field(
        default=None,
        description="Optional Kaggle API token for kagglehub authentication",
    )
    KAGGLE_DATASET_HANDLE: str = Field(
        default="davidcariboo/player-scores",
        description="Primary Kaggle dataset handle in format 'owner/dataset-name'",
    )
    TRANSFER_DATASET_HANDLE: str = Field(
        default="mexwell/football-player-transfers",
        description="Supplementary transfer dataset handle in format 'owner/dataset-name'",
    )

    # Resilience & Retry Backoff Configuration
    MAX_RETRY_ATTEMPTS: int = Field(
        default=5,
        description="Maximum retry attempts for transient database operations",
    )
    RETRY_MIN_WAIT: float = Field(
        default=1.0,
        description="Minimum backoff wait time in seconds for retries",
    )
    RETRY_MAX_WAIT: float = Field(
        default=10.0,
        description="Maximum backoff wait time in seconds for retries",
    )

    # Gemini & LLM Configuration
    GEMINI_API_KEY: Optional[str] = Field(
        default=None,
        description="Google AI Studio Gemini API Key",
    )
    GRAPH_LLM_MODEL: str = Field(
        default="gemini-3.6-flash",
        description="LLM model identifier for GraphRAG agent",
    )

    model_config = SettingsConfigDict(
        env_file=(_ENV_FILE_PATH, ".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization hook to export KAGGLE_API_TOKEN to os.environ if provided
        and log masked active NEO4J_URI endpoint.
        """
        if self.KAGGLE_API_TOKEN:
            os.environ["KAGGLE_API_TOKEN"] = self.KAGGLE_API_TOKEN
            logger.info("Exported KAGGLE_API_TOKEN to os.environ for kagglehub authentication.")

        masked_endpoint = mask_neo4j_uri(self.NEO4J_URI)
        logger.info("Loaded configuration. Active NEO4J_URI: %s (database: '%s')", masked_endpoint, self.NEO4J_DATABASE)

    @field_validator("NEO4J_URI", mode="after")
    @classmethod
    def validate_neo4j_uri(cls, value: str) -> str:
        """Validate that the Neo4j URI starts with a recognized scheme."""
        valid_schemes = (
            "bolt://",
            "bolt+s://",
            "bolt+ssc://",
            "neo4j://",
            "neo4j+s://",
            "neo4j+ssc://",
        )
        if not any(value.startswith(scheme) for scheme in valid_schemes):
            logger.warning(
                "NEO4J_URI '%s' does not start with a standard scheme %s",
                value,
                valid_schemes,
            )
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Get cached instance of application settings.

    Returns:
        Settings: Validated application configuration instance.
    """
    return Settings()