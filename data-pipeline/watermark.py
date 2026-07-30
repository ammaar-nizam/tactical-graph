"""
Watermark management module for TacticalGraph data pipeline.

Provides state tracking for incremental updates using a Neo4j Watermark node.
"""

import logging
from typing import Optional

from database import Neo4jDatabase

logger = logging.getLogger(__name__)


class WatermarkManager:
    """
    Manages the ETL process watermark stored in Neo4j to support delta ingestion.

    Interacts with a single node with label :Watermark and property id = 'global'.
    """

    WATERMARK_ID: str = "global"

    def __init__(self, db: Optional[Neo4jDatabase] = None) -> None:
        """
        Initialize WatermarkManager with a database instance.

        Args:
            db: Optional Neo4jDatabase instance. If omitted, defaults to a new instance.
        """
        self.db = db or Neo4jDatabase()

    def get_last_processed_date(self) -> Optional[str]:
        """
        Retrieve the last processed date from the global Watermark node in Neo4j.

        Returns:
            Optional[str]: The ISO date string of the last processed record (e.g., '2023-10-15'),
                           or None if no watermark node or date property exists.

        Raises:
            Exception: If database query execution fails.
        """
        query = """
        MATCH (w:Watermark {id: $id})
        RETURN w.last_processed_date AS last_processed_date
        """
        try:
            logger.debug("Fetching last_processed_date for watermark ID '%s'", self.WATERMARK_ID)
            records = self.db.execute_query(query, parameters={"id": self.WATERMARK_ID})

            if records and records[0].get("last_processed_date") is not None:
                last_date = str(records[0]["last_processed_date"])
                logger.info("Found active watermark date: %s", last_date)
                return last_date

            logger.info("No watermark found for ID '%s'. Returning None.", self.WATERMARK_ID)
            return None
        except Exception as e:
            logger.error("Failed to fetch last_processed_date from Neo4j: %s", e)
            raise

    def update_last_processed_date(self, new_date: str) -> None:
        """
        Update or create the global Watermark node in Neo4j with a new processed date.

        Args:
            new_date: ISO date string representing the updated watermark timestamp (e.g., '2023-10-15').

        Raises:
            ValueError: If new_date is empty or not a string.
            Exception: If database update operation fails.
        """
        if not new_date or not isinstance(new_date, str):
            error_msg = "new_date must be a non-empty string."
            logger.error(error_msg)
            raise ValueError(error_msg)

        query = """
        MERGE (w:Watermark {id: $id})
        SET w.last_processed_date = $new_date,
            w.updated_at = datetime()
        """
        try:
            logger.info("Updating watermark '%s' date to: %s", self.WATERMARK_ID, new_date)
            self.db.execute_write(query, parameters={"id": self.WATERMARK_ID, "new_date": new_date})
            logger.info("Successfully updated watermark to %s", new_date)
        except Exception as e:
            logger.error("Failed to update watermark date in Neo4j: %s", e)
            raise