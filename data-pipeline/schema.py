"""
Schema setup and constraint installation module for TacticalGraph Neo4j database.

Executes Cypher DDL queries to create uniqueness constraints, performance indexes,
and full-text indexes for graph node entities in Neo4j.
"""

import logging
from typing import List, Optional

from database import Neo4jDatabase

logger = logging.getLogger(__name__)


class SchemaInstaller:
    """
    Manages the installation of Neo4j schema constraints and indexes for TacticalGraph.
    """

    CONSTRAINTS: List[str] = [
        "CREATE CONSTRAINT country_id_unique IF NOT EXISTS FOR (c:Country) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT competition_id_unique IF NOT EXISTS FOR (c:Competition) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT national_team_id_unique IF NOT EXISTS FOR (n:NationalTeam) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT club_id_unique IF NOT EXISTS FOR (c:Club) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT player_id_unique IF NOT EXISTS FOR (p:Player) REQUIRE p.id IS UNIQUE",
        "CREATE CONSTRAINT player_valuation_id_unique IF NOT EXISTS FOR (pv:PlayerValuation) REQUIRE pv.id IS UNIQUE",
        "CREATE CONSTRAINT game_id_unique IF NOT EXISTS FOR (g:Game) REQUIRE g.id IS UNIQUE",
        "CREATE CONSTRAINT game_event_id_unique IF NOT EXISTS FOR (ge:GameEvent) REQUIRE ge.id IS UNIQUE",
    ]

    INDEXES: List[str] = [
        "CREATE INDEX player_name_index IF NOT EXISTS FOR (p:Player) ON (p.name)",
        "CREATE INDEX club_name_index IF NOT EXISTS FOR (c:Club) ON (c.name)",
        "CREATE INDEX game_date_index IF NOT EXISTS FOR (g:Game) ON (g.date)",
    ]

    FULLTEXT_INDEXES: List[str] = [
        "CREATE FULLTEXT INDEX player_name_fulltext IF NOT EXISTS FOR (p:Player) ON EACH [p.name]",
        "CREATE FULLTEXT INDEX club_name_fulltext IF NOT EXISTS FOR (c:Club) ON EACH [c.name]",
        "CREATE FULLTEXT INDEX country_name_fulltext IF NOT EXISTS FOR (c:Country) ON EACH [c.name]",
        "CREATE FULLTEXT INDEX national_team_name_fulltext IF NOT EXISTS FOR (n:NationalTeam) ON EACH [n.name]",
        "CREATE FULLTEXT INDEX competition_name_fulltext IF NOT EXISTS FOR (c:Competition) ON EACH [c.name]",
        "CREATE FULLTEXT INDEX game_event_description_fulltext IF NOT EXISTS FOR (ge:GameEvent) ON EACH [ge.description]",
    ]

    def __init__(self, db: Optional[Neo4jDatabase] = None) -> None:
        """
        Initialize SchemaInstaller.

        Args:
            db: Optional instance of Neo4jDatabase. If None, a new Neo4jDatabase instance is created.
        """
        self.db = db or Neo4jDatabase()

    def install_schema(self) -> None:
        """
        Execute schema constraint and index creation queries against Neo4j.

        Logs progress and errors using structured logging.

        Raises:
            Exception: Re-raises any database exception encountered during schema installation.
        """
        logger.info("Initiating Neo4j schema installation...")
        self.db.connect()

        installed_constraints = 0
        installed_indexes = 0
        installed_fulltext_indexes = 0

        for query in self.CONSTRAINTS:
            try:
                logger.info("Creating constraint: %s", query)
                self.db.execute_query(query)
                installed_constraints += 1
            except Exception as e:
                logger.error("Failed to create constraint with query [%s]: %s", query, e)
                raise

        for query in self.INDEXES:
            try:
                logger.info("Creating index: %s", query)
                self.db.execute_query(query)
                installed_indexes += 1
            except Exception as e:
                logger.error("Failed to create index with query [%s]: %s", query, e)
                raise

        for query in self.FULLTEXT_INDEXES:
            try:
                logger.info("Creating full-text index: %s", query)
                self.db.execute_query(query)
                installed_fulltext_indexes += 1
            except Exception as e:
                logger.error("Failed to create full-text index with query [%s]: %s", query, e)
                raise

        logger.info(
            "Schema installation completed successfully. Installed %d constraints, %d indexes, and %d full-text indexes.",
            installed_constraints,
            installed_indexes,
            installed_fulltext_indexes,
        )


def install_schema(db: Optional[Neo4jDatabase] = None) -> None:
    """
    Convenience function to instantiate SchemaInstaller and execute schema installation.

    Args:
        db: Optional Neo4jDatabase instance.
    """
    installer = SchemaInstaller(db=db)
    installer.install_schema()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    install_schema()
