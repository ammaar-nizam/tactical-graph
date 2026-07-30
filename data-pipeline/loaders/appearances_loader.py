"""
Appearances loader module for TacticalGraph data pipeline.

Merges appearances.csv and game_lineups.csv into a single rich
(Player)-[:APPEARED_IN]->(Game) edge payload capturing all match-level
player statistics and lineup metadata.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    from loaders.base_loader import BaseLoader
except ImportError:
    from base_loader import BaseLoader

logger = logging.getLogger(__name__)


class AppearancesLoader(BaseLoader):
    """
    Loader for player appearance data, merging appearances.csv and game_lineups.csv.

    Creates a single (Player)-[:APPEARED_IN {minutesPlayed, type, position,
    number, teamCaptain, goals, assists, yellowCards, redCards}]->(Game) edge
    per player-game pair, preventing graph bloat while retaining full statistics.
    """

    CYPHER_MERGE_APPEARED_IN = """
    UNWIND $batch AS row
    MERGE (p:Player {id: row.playerId})
    MERGE (g:Game {id: row.gameId})
    MERGE (p)-[a:APPEARED_IN {playerId: row.playerId, gameId: row.gameId}]->(g)
    SET a.minutesPlayed = toInteger(row.minutesPlayed),
        a.type          = row.type,
        a.position      = row.position,
        a.number        = toInteger(row.number),
        a.teamCaptain   = row.teamCaptain,
        a.goals         = toInteger(row.goals),
        a.assists       = toInteger(row.assists),
        a.yellowCards   = toInteger(row.yellowCards),
        a.redCards      = toInteger(row.redCards)
    """

    def _prepare_appearances(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Map appearances DataFrame columns to camelCase Cypher parameter dicts.

        Args:
            df: Merged appearances+lineups DataFrame.

        Returns:
            List[Dict[str, Any]]: Sanitized records.
        """
        records: List[Dict[str, Any]] = []
        for row in df.to_dict(orient="records"):
            player_id = row.get("player_id")
            game_id = row.get("game_id")
            if player_id is None or pd.isna(player_id) or game_id is None or pd.isna(game_id):
                continue

            records.append(
                {
                    "playerId": str(player_id),
                    "gameId": str(game_id),
                    "minutesPlayed": row.get("minutes_played"),
                    # lineup type: 'starting' or 'substitute'
                    "type": row.get("type") or row.get("lineup_type"),
                    "position": row.get("position"),
                    "number": row.get("jersey_number") or row.get("number"),
                    "teamCaptain": bool(row.get("team_captain")) if row.get("team_captain") is not None and not pd.isna(row.get("team_captain")) else None,
                    "goals": row.get("goals"),
                    "assists": row.get("assists"),
                    "yellowCards": row.get("yellow_cards"),
                    "redCards": row.get("red_cards"),
                }
            )
        return records

    def _merge_appearances_lineups(
        self, appearances_df: pd.DataFrame, lineups_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Left-merge game_lineups.csv into appearances.csv to enrich the payload
        with lineup metadata (type, number, teamCaptain) per player-game pair.

        Args:
            appearances_df: Raw appearances DataFrame.
            lineups_df: Raw game_lineups DataFrame.

        Returns:
            pd.DataFrame: Merged and deduplicated DataFrame.
        """
        if lineups_df.empty:
            logger.warning("game_lineups.csv is empty; using appearances.csv only.")
            return appearances_df

        # Normalise join key columns
        for df in (appearances_df, lineups_df):
            for col in ("player_id", "game_id"):
                if col in df.columns:
                    df[col] = df[col].astype(str).str.strip()

        # lineup columns to pull in (avoid collision with appearances columns)
        lineup_cols = ["player_id", "game_id", "type", "jersey_number", "team_captain"]
        available_lineup_cols = [c for c in lineup_cols if c in lineups_df.columns]
        lineups_subset = lineups_df[available_lineup_cols].drop_duplicates(
            subset=["player_id", "game_id"]
        )

        # Suffix appearances 'type' if it exists to avoid post-merge ambiguity
        suffix_app = "_app" if "type" in appearances_df.columns else ""
        merged = appearances_df.merge(
            lineups_subset,
            on=["player_id", "game_id"],
            how="left",
            suffixes=(suffix_app, ""),
        )

        # If appearances had its own 'type', prefer the lineup type (more authoritative)
        if suffix_app and f"type{suffix_app}" in merged.columns and "type" in merged.columns:
            merged["type"] = merged["type"].fillna(merged[f"type{suffix_app}"])
            merged.drop(columns=[f"type{suffix_app}"], inplace=True)

        logger.info(
            "Merged appearances and lineups: %d rows → %d rows after join",
            len(appearances_df),
            len(merged),
        )
        return merged

    def load(self, data_dir: str, dev_mode: bool = False) -> None:
        """
        Load and merge appearances.csv + game_lineups.csv into Neo4j APPEARED_IN edges.

        Args:
            data_dir: Directory path containing raw CSV files.
            dev_mode: If True, filter to active player_ids from utils.py.
        """
        logger.info("Starting Appearances Data Ingestion (dev_mode=%s)", dev_mode)

        dev_filters = self.get_dev_filter_ids(data_dir) if dev_mode else None

        # ── 1. Load appearances.csv ──────────────────────────────────────────
        appearances_path = os.path.join(data_dir, "appearances.csv")
        if not os.path.exists(appearances_path):
            logger.warning("appearances.csv not found at %s. Skipping.", appearances_path)
            return
        appearances_df = self.read_csv(appearances_path)

        # ── 2. Load game_lineups.csv (optional enrichment) ───────────────────
        lineups_path = os.path.join(data_dir, "game_lineups.csv")
        if os.path.exists(lineups_path):
            lineups_df = self.read_csv(lineups_path)
            merged_df = self._merge_appearances_lineups(appearances_df, lineups_df)
        else:
            logger.warning(
                "game_lineups.csv not found at %s. Proceeding with appearances only.",
                lineups_path,
            )
            merged_df = appearances_df

        # ── 3. DEV_MODE player filter ────────────────────────────────────────
        if dev_mode and dev_filters and "player_ids" in dev_filters:
            valid_player_ids = {str(pid).strip() for pid in dev_filters["player_ids"]}
            player_col = "player_id" if "player_id" in merged_df.columns else None
            if player_col:
                merged_df = merged_df[
                    merged_df[player_col].astype(str).str.strip().isin(valid_player_ids)
                ]
                logger.info(
                    "Filtered merged appearances to %d rows for DEV_MODE (player_ids filter)",
                    len(merged_df),
                )

        appearance_records = self._prepare_appearances(merged_df)
        total = self.execute_batch(self.CYPHER_MERGE_APPEARED_IN, appearance_records)
        logger.info(
            "Completed Appearances Data Ingestion. Processed %d APPEARED_IN edges.", total
        )