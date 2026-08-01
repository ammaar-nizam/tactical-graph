"""
Database abstraction and Neo4j connection pool management for TacticalGraph.

Implements connection pooling, session context management, resilient Cypher query execution
with exponential backoff retries (via tenacity), structured logging.
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
import logging
import time
from typing import Any, Dict, Generator, List, Optional

import neo4j
from neo4j import Driver, GraphDatabase, Result, Session
from neo4j.exceptions import (
    DriverError,
    Neo4jError,
    ServiceUnavailable,
    SessionExpired,
    TransientError,
)
from tenacity import (
    Retrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import Settings, get_settings

logger = logging.getLogger(__name__)

# Transient network and database exceptions eligible for exponential backoff retries
RETRYABLE_EXCEPTIONS = (
    ServiceUnavailable,
    SessionExpired,
    TransientError,
    DriverError,
)


class IDatabase(ABC):
    """
    Abstract interface for database access following the Dependency Inversion Principle (DIP).
    """

    @abstractmethod
    def connect(self) -> None:
        """Establish database connection driver pool."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close driver pool and release resources."""
        pass

    @abstractmethod
    def verify_connectivity(self) -> bool:
        """Verify that the database connection is alive and healthy."""
        pass

    @abstractmethod
    def session(
        self, db_name: Optional[str] = None, access_mode: str = "WRITE"
    ) -> Generator[Session, None, None]:
        """Context manager yielding a database session."""
        pass

    @abstractmethod
    def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        db_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query with exponential backoff retries and return record dicts."""
        pass


class Neo4jDatabase(IDatabase):
    """
    Neo4j Database Manager handling connection pooling, session lifecycle,
    and retryable Cypher query execution.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """
        Initialize Neo4jDatabase with application settings.

        Args:
            settings: Configuration settings instance. Defaults to cached global settings.
        """
        self._settings = settings or get_settings()
        self._driver: Optional[Driver] = None

    @property
    def driver(self) -> Driver:
        """
        Returns active Neo4j driver instance.

        Raises:
            RuntimeError: If database driver has not been connected.
        """
        if self._driver is None:
            raise RuntimeError(
                "Database driver is not initialized. Call connect() or use context manager."
            )
        return self._driver

    def connect(self) -> None:
        """
        Initialize Neo4j driver with connection pooling.
        """
        if self._driver is not None:
            logger.debug("Neo4j driver is already initialized.")
            return

        logger.info(
            "Initializing Neo4j connection pool to URI: %s", self._settings.NEO4J_URI
        )
        try:
            self._driver = GraphDatabase.driver(
                self._settings.NEO4J_URI,
                auth=(self._settings.NEO4J_USER, self._settings.NEO4J_PASSWORD),
                max_connection_pool_size=self._settings.NEO4J_MAX_CONNECTION_POOL_SIZE,
                connection_timeout=self._settings.NEO4J_CONNECTION_TIMEOUT,
            )
            self.verify_connectivity()
            logger.info("Successfully connected to Neo4j database.")
        except Exception as e:
            logger.error(
                "Failed to connect to Neo4j database at %s: %s",
                self._settings.NEO4J_URI,
                e,
            )
            self.close()
            raise

    def close(self) -> None:
        """
        Close the Neo4j driver and release connection pool resources.
        """
        if self._driver is not None:
            logger.info("Closing Neo4j connection driver pool.")
            try:
                self._driver.close()
            except Exception as e:
                logger.error("Error encountered while closing Neo4j driver: %s", e)
            finally:
                self._driver = None

    def verify_connectivity(self) -> bool:
        """
        Verify database connection health.

        Returns:
            bool: True if connection is healthy.

        Raises:
            RuntimeError: If driver is not initialized.
            ServiceUnavailable: If database is unreachable.
        """
        if self._driver is None:
            raise RuntimeError("Database driver is not initialized.")

        try:
            target_db = self._settings.NEO4J_DATABASE or "neo4j"
            self._driver.verify_connectivity(database=target_db)
            return True
        except Exception as e:
            logger.error("Neo4j connectivity check failed: %s", e)
            raise

    @contextmanager
    def session(
        self, db_name: Optional[str] = None, access_mode: str = "WRITE"
    ) -> Generator[Session, None, None]:
        """
        Context manager for managing Neo4j session lifecycle safely.

        Explicitly sets database (defaulting to "neo4j") to eliminate extra network
        roundtrips when connecting to Neo4j Aura instances.

        Args:
            db_name: Name of target Neo4j database. Defaults to configured database ("neo4j").
            access_mode: "WRITE" or "READ".

        Yields:
            Session: Neo4j session instance.
        """
        target_db = db_name or self._settings.NEO4J_DATABASE or "neo4j"
        mode = (
            neo4j.WRITE_ACCESS
            if access_mode.upper() == "WRITE"
            else neo4j.READ_ACCESS
        )

        if self._driver is None:
            self.connect()

        sess = self.driver.session(database=target_db, default_access_mode=mode)
        try:
            yield sess
        except Neo4jError as e:
            logger.error(
                "Neo4j session error [Code: %s]: %s",
                getattr(e, "code", "UNKNOWN"),
                e,
            )
            raise
        except Exception as e:
            logger.error("Unexpected error during Neo4j session: %s", e)
            raise
        finally:
            sess.close()

    def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        db_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query with automatic exponential backoff retry logic using tenacity.

        Args:
            query: Parameterized Cypher query string.
            parameters: Dictionary of query parameters.
            db_name: Target Neo4j database name.

        Returns:
            List[Dict[str, Any]]: Query result records formatted as dictionaries.
        """
        params = parameters or {}
        target_db = db_name or self._settings.NEO4J_DATABASE or "neo4j"

        def _raw_execution() -> List[Dict[str, Any]]:
            start_time = time.perf_counter()
            with self.session(db_name=target_db) as sess:
                result: Result = sess.run(query, params)
                records = [record.data() for record in result]
                elapsed = (time.perf_counter() - start_time) * 1000
                logger.debug(
                    "Executed Cypher query in %.2f ms, returned %d records.",
                    elapsed,
                    len(records),
                )
                return records

        retrier = Retrying(
            stop=stop_after_attempt(self._settings.MAX_RETRY_ATTEMPTS),
            wait=wait_exponential(
                min=self._settings.RETRY_MIN_WAIT, max=self._settings.RETRY_MAX_WAIT
            ),
            retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )

        try:
            return retrier(_raw_execution)
        except Exception as e:
            logger.error(
                "Cypher query execution failed after retries.\nQuery: %s\nParams: %s\nError: %s",
                query,
                params,
                e,
            )
            raise

    def execute_write(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        db_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Convenience wrapper for executing write Cypher queries with retries.

        Args:
            query: Parameterized Cypher query string.
            parameters: Dictionary of query parameters.
            db_name: Target Neo4j database name.

        Returns:
            List[Dict[str, Any]]: Query result records.
        """
        return self.execute_query(query=query, parameters=parameters, db_name=db_name)

    def execute_batch(
        self,
        query: str,
        batch_data: List[Dict[str, Any]],
        batch_param_name: str = "batch",
        db_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute Cypher query for batch processing (e.g. UNWIND $batch AS row...).

        Args:
            query: Cypher query expecting a batch list parameter.
            batch_data: List of dictionaries representing row items.
            batch_param_name: Name of batch parameter in Cypher query (default: 'batch').
            db_name: Target Neo4j database.

        Returns:
            List[Dict[str, Any]]: Execution result summary or returned records.
        """
        if not batch_data:
            logger.debug("Empty batch passed to execute_batch. Skipping.")
            return []

        logger.info("Executing batch Cypher write for %d records.", len(batch_data))
        return self.execute_query(
            query=query,
            parameters={batch_param_name: batch_data},
            db_name=db_name,
        )

    def __enter__(self) -> "Neo4jDatabase":
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()