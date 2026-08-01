"""
Matches loader module for TacticalGraph data pipeline.

Ingests games.csv and club_games.csv to create (:Game) nodes and the
(Game)-[:PART_OF_COMPETITION]->(Competition) and (Club)-[:PLAYED_IN]->(Game)
relationship edges with full per-club match performance properties.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

try:
    from loaders.base_loader import BaseLoader
except ImportError:
    from base_loader import BaseLoader

logger = logging.getLogger(__name__)


class MatchesLoader(BaseLoader):
    """
    Loader for match data from games.csv and club_games.csv.

    Creates (:Game) nodes linked to competitions and clubs via rich relationship edges.
    """

    CYPHER_MERGE_GAMES = """
    UNWIND $batch AS row
    MERGE (g:Game {id: row.id})
    SET g.season               = toInteger(row.season),
        g.round                = row.round,
        g.date                 = row.date,
        g.homeClubGoals        = toInteger(row.homeClubGoals),
        g.awayClubGoals        = toInteger(row.awayClubGoals),
        g.homeClubPosition     = toInteger(row.homeClubPosition),
        g.awayClubPosition     = toInteger(row.awayClubPosition),
        g.homeClubManagerName  = row.homeClubManagerName,
        g.awayClubManagerName  = row.awayClubManagerName,
        g.stadium              = row.stadium,
        g.attendance           = toInteger(row.attendance),
        g.referee              = row.referee,
        g.url                  = row.url,
        g.homeClubFormation    = row.homeClubFormation,
        g.awayClubFormation    = row.awayClubFormation,
        g.aggregate            = row.aggregate
    WITH g, row
    WHERE row.competitionId IS NOT NULL
    MERGE (comp:Competition {id: row.competitionId})
    MERGE (g)-[:PART_OF_COMPETITION]->(comp)
    """

    # club_games.csv provides one row per club per game (home + away separately)
    CYPHER_MERGE_CLUB_PLAYED_IN = """
    UNWIND $batch AS row
    MERGE (cl:Club {id: row.clubId})
    MERGE (g:Game {id: row.gameId})
    MERGE (cl)-[r:PLAYED_IN {clubId: row.clubId, gameId: row.gameId}]->(g)
    SET r.hosting              = row.hosting,
        r.isWin                = row.isWin,
        r.ownGoals             = toInteger(row.ownGoals),
        r.opponentGoals        = toInteger(row.opponentGoals),
        r.ownManagerName       = row.ownManagerName,
        r.opponentManagerName  = row.opponentManagerName,
        r.ownPosition          = toInteger(row.ownPosition),
        r.opponentPosition     = toInteger(row.opponentPosition)
    """

    def _prepare_games(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Map games DataFrame columns to camelCase Cypher parameter dicts.

        Args:
            df: Raw games DataFrame.

        Returns:
            List[Dict[str, Any]]: Sanitized records.
        """
        records: List[Dict[str, Any]] = []
        for row in df.to_dict(orient="records"):
            game_id = row.get("game_id") or row.get("id")
            if game_id is None or pd.isna(game_id):
                continue

            comp_id = row.get("competition_id")
            records.append(
                {
                    "id": str(game_id),
                    "season": row.get("season"),
                    "round": row.get("round"),
                    "date": str(row.get("date")) if row.get("date") and not pd.isna(row.get("date")) else None,
                    "homeClubGoals": row.get("home_club_goals"),
                    "awayClubGoals": row.get("away_club_goals"),
                    "homeClubPosition": row.get("home_club_position"),
                    "awayClubPosition": row.get("away_club_position"),
                    "homeClubManagerName": row.get("home_club_manager_name"),
                    "awayClubManagerName": row.get("away_club_manager_name"),
                    "stadium": row.get("stadium"),
                    "attendance": row.get("attendance"),
                    "referee": row.get("referee"),
                    "url": row.get("url"),
                    "homeClubFormation": row.get("home_club_formation"),
                    "awayClubFormation": row.get("away_club_formation"),
                    "aggregate": row.get("aggregate"),
                    "competitionId": str(comp_id) if comp_id and not pd.isna(comp_id) else None,
                }
            )
        return records

    def _prepare_club_games(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Map club_games DataFrame columns to camelCase Cypher parameter dicts.

        Args:
            df: Raw club_games DataFrame.

        Returns:
            List[Dict[str, Any]]: Sanitized records.
        """
        records: List[Dict[str, Any]] = []
        for row in df.to_dict(orient="records"):
            club_id = row.get("club_id")
            game_id = row.get("game_id")
            if club_id is None or pd.isna(club_id) or game_id is None or pd.isna(game_id):
                continue

            records.append(
                {
                    "clubId": str(club_id),
                    "gameId": str(game_id),
                    "hosting": row.get("hosting"),
                    "isWin": row.get("is_win"),
                    "ownGoals": row.get("own_goals"),
                    "opponentGoals": row.get("opponent_goals"),
                    "ownManagerName": row.get("own_manager_name"),
                    "opponentManagerName": row.get("opponent_manager_name"),
                    "ownPosition": row.get("own_position"),
                    "opponentPosition": row.get("opponent_position"),
                }
            )
        return records

    def load(
        self,
        data_dir: Union[str, Path],
        dev_mode: bool = False,
        watermark_date: Optional[str] = None,
    ) -> None:
        """
        Load games.csv and club_games.csv into Neo4j.

        Args:
            data_dir: Directory path containing raw CSV files.
            dev_mode: If True, filter by active competition_ids and club_ids.
            watermark_date: If set (incremental mode), only load games with date > watermark_date.
        """
        logger.info(
            "Starting Matches Data Ingestion (dev_mode=%s, watermark_date=%s)",
            dev_mode,
            watermark_date,
        )

        dev_filters = self.get_dev_filter_ids(data_dir) if dev_mode else None

        # ── 1. Ingest games.csv ──────────────────────────────────────────────
        games_path = os.path.join(data_dir, "games.csv")
        if not os.path.exists(games_path):
            logger.warning("games.csv not found at %s. Skipping.", games_path)
        else:
            games_df = self.read_csv(games_path)

            # Incremental watermark filter
            if watermark_date:
                date_col = "date"
                if date_col in games_df.columns:
                    games_df[date_col] = pd.to_datetime(games_df[date_col], errors="coerce")
                    wm_dt = pd.to_datetime(watermark_date, errors="coerce")
                    games_df = games_df[games_df[date_col] > wm_dt]
                    logger.info(
                        "Watermark filter applied: %d games after %s",
                        len(games_df),
                        watermark_date,
                    )

            # DEV_MODE competition filter
            if dev_mode and dev_filters and "competition_ids" in dev_filters:
                comp_col = "competition_id" if "competition_id" in games_df.columns else None
                if comp_col:
                    valid_comp_ids = {str(c).strip() for c in dev_filters["competition_ids"]}
                    games_df = games_df[
                        games_df[comp_col].astype(str).str.strip().isin(valid_comp_ids)
                    ]
                    logger.info(
                        "Filtered games.csv to %d rows for DEV_MODE (competition filter)",
                        len(games_df),
                    )

            game_records = self._prepare_games(games_df)
            total = self.execute_batch(self.CYPHER_MERGE_GAMES, game_records)
            logger.info("Merged %d Game nodes.", total)

        # ── 2. Ingest club_games.csv ─────────────────────────────────────────
        club_games_path = os.path.join(data_dir, "club_games.csv")
        if not os.path.exists(club_games_path):
            logger.warning("club_games.csv not found at %s. Skipping.", club_games_path)
        else:
            club_games_df = self.read_csv(club_games_path)

            # Sync with watermark-filtered game IDs to maintain referential integrity
            if watermark_date and not games_df.empty:
                game_id_col_cg = "game_id"
                if game_id_col_cg in club_games_df.columns:
                    games_id_col = "game_id" if "game_id" in games_df.columns else "id"
                    valid_game_ids = {str(gid) for gid in games_df[games_id_col].dropna().unique()}
                    club_games_df = club_games_df[
                        club_games_df[game_id_col_cg].astype(str).isin(valid_game_ids)
                    ]

            # DEV_MODE club filter
            if dev_mode and dev_filters and "club_ids" in dev_filters:
                valid_club_ids = {str(cid).strip() for cid in dev_filters["club_ids"]}
                club_id_col = "club_id" if "club_id" in club_games_df.columns else None
                if club_id_col:
                    club_games_df = club_games_df[
                        club_games_df[club_id_col].astype(str).str.strip().isin(valid_club_ids)
                    ]
                    logger.info(
                        "Filtered club_games.csv to %d rows for DEV_MODE (club filter)",
                        len(club_games_df),
                    )

            club_game_records = self._prepare_club_games(club_games_df)
            total_cg = self.execute_batch(self.CYPHER_MERGE_CLUB_PLAYED_IN, club_game_records)
            logger.info("Merged %d PLAYED_IN relationship edges.", total_cg)

        logger.info("Completed Matches Data Ingestion.")