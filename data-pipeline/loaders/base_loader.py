"""
Abstract base loader module for TacticalGraph data pipeline.

Provides reusable functionality for reading CSVs, data cleaning,
batch Cypher query execution with error handling and retry mechanisms,
and development mode filtering.
"""

from abc import ABC, abstractmethod
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd

from config import Settings, get_settings
from database import Neo4jDatabase
from utils import get_dev_subgraph_ids

logger = logging.getLogger(__name__)


class BaseLoader(ABC):
    """
    Abstract base class for domain-specific CSV ETL loaders.
    """

    def __init__(
        self,
        db: Optional[Neo4jDatabase] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """
        Initialize BaseLoader with database and settings instances.

        Args:
            db: Optional Neo4jDatabase instance. If None, a new instance is created.
            settings: Optional Settings instance. If None, default settings are loaded.
        """
        self.settings = settings or get_settings()
        self.db = db or Neo4jDatabase(settings=self.settings)

    def read_csv(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Read a CSV file into a pandas DataFrame with error handling.

        Args:
            file_path: Path to the CSV file.

        Returns:
            pd.DataFrame: Loaded DataFrame.

        Raises:
            FileNotFoundError: If the file does not exist.
            Exception: If reading the CSV fails.
        """
        path_obj = Path(file_path)
        if not path_obj.exists():
            error_msg = f"CSV file not found at path: {path_obj}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            logger.info("Loading CSV from %s", path_obj)
            df = pd.read_csv(path_obj)
            logger.info("Loaded %d rows from %s", len(df), path_obj)
            return df
        except Exception as e:
            logger.error("Failed to read CSV file %s: %s", path_obj, e)
            raise

    def sanitize_dataframe(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Convert DataFrame to a list of dicts, replacing NaN / NaT / inf values with None for Neo4j.

        Args:
            df: Pandas DataFrame.

        Returns:
            List[Dict[str, Any]]: Clean list of records suitable for Cypher parameters.
        """
        if df.empty:
            return []

        clean_df = df.astype(object).where(pd.notnull(df), None)
        return clean_df.to_dict(orient="records")

    def execute_batch(
        self,
        query: str,
        batch_data: List[Dict[str, Any]],
        batch_size: int = 5000,
        batch_param_name: str = "batch",
    ) -> int:
        """
        Execute a Cypher write query in chunks with retry handling.

        Args:
            query: Cypher query expecting a batch parameter (e.g. UNWIND $batch AS row...).
            batch_data: List of dictionary records.
            batch_size: Maximum records per batch execution (default: 5000).
            batch_param_name: Parameter key name in Cypher query (default: 'batch').

        Returns:
            int: Total number of records processed.
        """
        if not batch_data:
            logger.debug("Empty batch_data passed to execute_batch. Skipping execution.")
            return 0

        total_records = len(batch_data)
        logger.info("Executing batch load for %d records (batch_size=%d).", total_records, batch_size)

        processed = 0
        for i in range(0, total_records, batch_size):
            chunk = batch_data[i : i + batch_size]
            try:
                self.db.execute_batch(
                    query=query,
                    batch_data=chunk,
                    batch_param_name=batch_param_name,
                )
                processed += len(chunk)
                logger.info(
                    "Processed batch %d/%d (%d/%d records)",
                    (i // batch_size) + 1,
                    (total_records + batch_size - 1) // batch_size,
                    processed,
                    total_records,
                )
            except Exception as e:
                logger.error("Failed executing batch starting at row %d: %s", i, e)
                raise

        return processed

    def get_dev_filter_ids(self, data_dir: Union[str, Path]) -> Dict[str, Set[Any]]:
        """
        Retrieve valid competition, club, and player ID sets for DEV_MODE filtering.

        Args:
            data_dir: Path to directory containing raw CSVs.

        Returns:
            Dict[str, Set[Any]]: Active ID sets for dev subgraph.
        """
        logger.info("Fetching dev subgraph ID filter sets from %s", data_dir)
        return get_dev_subgraph_ids(data_dir=data_dir)

    @abstractmethod
    def load(self, data_dir: Union[str, Path], dev_mode: bool = False) -> None:
        """
        Abstract method to run data ingestion for concrete loaders.

        Args:
            data_dir: Directory containing domain CSV files.
            dev_mode: Whether to enable DEV_MODE subgraph filtering.
        """
        pass
