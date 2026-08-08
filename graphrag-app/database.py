"""
Centralized Neo4j database driver pool management for GraphRAG application.

Provides a single application-wide shared Driver instance (singleton) with connection pooling,
idle socket liveness checks, max connection lifetime recycling,
and resilient Cypher query execution retries.
"""

import logging
from typing import Any, Dict, List, Optional

from neo4j import Driver, GraphDatabase

from config import settings

logger = logging.getLogger(__name__)

# Global singleton driver instance
_driver_instance: Optional[Driver] = None


def get_neo4j_driver() -> Driver:
    """
    Return the application-wide single shared Neo4j Driver instance (singleton).
    Initializes driver connection pool with TCP keep-alive, liveness checks, and max connection lifetime parameters.

    Returns:
        Shared Driver object.
    """
    global _driver_instance
    if _driver_instance is None:
        uri = settings.NEO4J_URI.replace("neo4j+s://", "neo4j+ssc://")
        auth = (settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        logger.info("Initializing single shared application-wide Neo4j driver pool to: %s", uri)
        _driver_instance = GraphDatabase.driver(
            uri,
            auth=auth,
            max_connection_lifetime=1800,
            liveness_check_timeout=30,
            max_connection_pool_size=50,
        )

    return _driver_instance


def close_neo4j_driver() -> None:
    """
    Close the shared application-wide Neo4j driver pool on application shutdown or pool reset.
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


def execute_cypher_query(
    query: str,
    parameters: Optional[Dict[str, Any]] = None,
    database: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Execute a Cypher query using the shared driver pool with automatic retry on defunct / expired connections.

    Args:
        query: Cypher query string to execute.
        parameters: Optional dictionary of query parameters.
        database: Optional target database name. Defaults to settings.NEO4J_DATABASE.

    Returns:
        List of dicts containing query result records.
    """
    db_name = database or settings.NEO4J_DATABASE
    params = parameters or {}
    driver = get_neo4j_driver()

    try:
        with driver.session(database=db_name) as session:
            result = session.run(query, params)
            return [record.data() for record in result]
    except Exception as e:
        err_str = str(e)
        if "SessionExpired" in err_str or "defunct connection" in err_str or "ConnectionResetError" in err_str:
            logger.warning("Encountered defunct/expired connection pool socket: %s. Resetting pool and retrying...", e)
            close_neo4j_driver()
            fresh_driver = get_neo4j_driver()
            with fresh_driver.session(database=db_name) as session:
                result = session.run(query, params)
                return [record.data() for record in result]
        raise
