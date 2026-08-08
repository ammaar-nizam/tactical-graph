"""
Centralized Neo4j database driver pool management for GraphRAG application.

Provides a single application-wide shared Driver instance (singleton) with connection pooling
and automatic protocol handling for Neo4j database sessions.
"""

import logging
from typing import Optional

from neo4j import Driver, GraphDatabase

from config import settings

logger = logging.getLogger(__name__)

# Global singleton driver instance
_driver_instance: Optional[Driver] = None


def get_neo4j_driver() -> Driver:
    """
    Return the application-wide single shared Neo4j Driver instance (singleton).
    Initializes driver connection pool on first invocation.

    Returns:
        Shared Driver object.
    """
    global _driver_instance
    if _driver_instance is None:
        uri = settings.NEO4J_URI.replace("neo4j+s://", "neo4j+ssc://")
        auth = (settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        logger.info("Initializing single shared application-wide Neo4j driver pool to: %s", uri)
        _driver_instance = GraphDatabase.driver(uri, auth=auth)

    return _driver_instance


def close_neo4j_driver() -> None:
    """
    Close the shared application-wide Neo4j driver pool on application shutdown.
    """
    global _driver_instance
    if _driver_instance is not None:
        logger.info("Closing shared application-wide Neo4j driver pool.")
        try:
            _driver_instance.close()
        except Exception as e:
            logger.error("Error closing shared Neo4j driver pool: %s", e)
        finally:
            _driver_instance = None
