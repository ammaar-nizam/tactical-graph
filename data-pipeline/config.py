"""
Configuration module for TacticalGraph data pipeline.

Loads application settings and credentials from environment variables or .env files
using Pydantic BaseSettings for strict type validation and defaults management.
"""

from functools import lru_cache
import logging
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


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
    KAGGLE_DATASET_HANDLE: str = Field(
        default="davidcariboo/player-scores",
        description="Kaggle dataset handle in format 'owner/dataset-name'",
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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

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