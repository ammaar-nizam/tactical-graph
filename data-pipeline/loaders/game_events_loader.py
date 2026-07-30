"""
Game Events loader module for TacticalGraph data pipeline.

Ingests game_events.csv and creates (:GameEvent) nodes alongside semantic
relationship edges that vary by event type (Goals, Substitutions, Cards).
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    from loaders.base_loader import BaseLoader
except ImportError:
    from base_loader import BaseLoader

logger = logging.getLogger(__name__)

# Event type constants matching Kaggle transfermarkt dataset values
EVENT_TYPE_GOALS = "Goals"
EVENT_TYPE_SUBSTITUTIONS = "Substitutions"
EVENT_TYPE_CARDS = "Cards"


class GameEventsLoader(BaseLoader):
    """
    Loader for game event data from game_events.csv.

    Creates (:GameEvent) nodes and semantic per-type relationship edges:
    - All events:       (GameEvent)-[:OCCURRED_IN]->(Game)
    - All events:       (Club)-[:INVOLVED_IN]->(GameEvent)
    - Goals:            (Player)-[:SCORED]->(GameEvent)
    - Goals w/ assist:  (Player)-[:ASSISTED]->(GameEvent)
    - Substitutions:    (Player)-[:SUBBED_OUT]->(GameEvent) via player_id
    - Substitutions:    (Player)-[:SUBBED_IN]->(GameEvent)  via player_in_id
    - Cards:            (Player)-[:RECEIVED_CARD]->(GameEvent)
    """

    # ── Node + structural edges (all event types) ───────────────────────────
    CYPHER_MERGE_GAME_EVENT_NODE = """
    UNWIND $batch AS row
    MERGE (ge:GameEvent {id: row.id})
    SET ge.date        = row.date,
        ge.minute      = toInteger(row.minute),
        ge.type        = row.type,
        ge.description = row.description
    WITH ge, row
    MERGE (g:Game {id: row.gameId})
    MERGE (ge)-[:OCCURRED_IN]->(g)
    WITH ge, row
    WHERE row.clubId IS NOT NULL
    MERGE (cl:Club {id: row.clubId})
    MERGE (cl)-[:INVOLVED_IN]->(ge)
    """

    # ── Goals: scorer ────────────────────────────────────────────────────────
    CYPHER_MERGE_SCORED = """
    UNWIND $batch AS row
    MATCH (ge:GameEvent {id: row.eventId})
    MERGE (p:Player {id: row.playerId})
    MERGE (p)-[:SCORED]->(ge)
    """

    # ── Goals: assister ──────────────────────────────────────────────────────
    CYPHER_MERGE_ASSISTED = """
    UNWIND $batch AS row
    MATCH (ge:GameEvent {id: row.eventId})
    MERGE (p:Player {id: row.assistPlayerId})
    MERGE (p)-[:ASSISTED]->(ge)
    """

    # ── Substitutions: player coming off ────────────────────────────────────
    CYPHER_MERGE_SUBBED_OUT = """
    UNWIND $batch AS row
    MATCH (ge:GameEvent {id: row.eventId})
    MERGE (p:Player {id: row.playerId})
    MERGE (p)-[:SUBBED_OUT]->(ge)
    """

    # ── Substitutions: player coming on ──────────────────────────────────────
    CYPHER_MERGE_SUBBED_IN = """
    UNWIND $batch AS row
    MATCH (ge:GameEvent {id: row.eventId})
    MERGE (p:Player {id: row.playerInId})
    MERGE (p)-[:SUBBED_IN]->(ge)
    """

    # ── Cards: recipient ──────────────────────────────────────────────────────
    CYPHER_MERGE_RECEIVED_CARD = """
    UNWIND $batch AS row
    MATCH (ge:GameEvent {id: row.eventId})
    MERGE (p:Player {id: row.playerId})
    MERGE (p)-[:RECEIVED_CARD]->(ge)
    """

    def _prepare_event_nodes(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Prepare base GameEvent node records (all event types).

        Args:
            df: Raw game_events DataFrame.

        Returns:
            List[Dict[str, Any]]: Base event node parameter dicts.
        """
        records: List[Dict[str, Any]] = []
        for row in df.to_dict(orient="records"):
            event_id = row.get("game_event_id") or row.get("id")
            game_id = row.get("game_id")
            if event_id is None or pd.isna(event_id) or game_id is None or pd.isna(game_id):
                continue

            club_id = row.get("club_id")
            records.append(
                {
                    "id": str(event_id),
                    "gameId": str(game_id),
                    "date": str(row.get("date")) if row.get("date") and not pd.isna(row.get("date")) else None,
                    "minute": row.get("minute"),
                    "type": row.get("type"),
                    "description": row.get("description"),
                    "clubId": str(club_id) if club_id is not None and not pd.isna(club_id) else None,
                }
            )
        return records

    def _split_by_type(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Partition the events DataFrame into goal, substitution, and card sub-frames.

        Args:
            df: Full game_events DataFrame.

        Returns:
            Tuple of (goals_df, subs_df, cards_df).
        """
        type_col = "type" if "type" in df.columns else None
        if not type_col:
            logger.warning("'type' column not found in game_events; semantic edges skipped.")
            empty = pd.DataFrame()
            return empty, empty, empty

        goals_df = df[df[type_col] == EVENT_TYPE_GOALS].copy()
        subs_df = df[df[type_col] == EVENT_TYPE_SUBSTITUTIONS].copy()
        cards_df = df[df[type_col] == EVENT_TYPE_CARDS].copy()
        logger.info(
            "Event type split — Goals: %d, Substitutions: %d, Cards: %d",
            len(goals_df),
            len(subs_df),
            len(cards_df),
        )
        return goals_df, subs_df, cards_df

    def _prepare_scored(self, goals_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Build SCORED relationship parameter dicts from goal events."""
        records: List[Dict[str, Any]] = []
        event_id_col = "game_event_id" if "game_event_id" in goals_df.columns else "id"
        player_col = "player_id" if "player_id" in goals_df.columns else None
        if not player_col:
            return records
        for row in goals_df.to_dict(orient="records"):
            event_id = row.get(event_id_col)
            player_id = row.get(player_col)
            if event_id and not pd.isna(event_id) and player_id and not pd.isna(player_id):
                records.append({"eventId": str(event_id), "playerId": str(player_id)})
        return records

    def _prepare_assisted(self, goals_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Build ASSISTED relationship parameter dicts from goal events with an assist player."""
        records: List[Dict[str, Any]] = []
        event_id_col = "game_event_id" if "game_event_id" in goals_df.columns else "id"
        assist_col = "player_assist_id" if "player_assist_id" in goals_df.columns else None
        if not assist_col:
            return records
        for row in goals_df.to_dict(orient="records"):
            event_id = row.get(event_id_col)
            assist_player_id = row.get(assist_col)
            if (
                event_id and not pd.isna(event_id)
                and assist_player_id and not pd.isna(assist_player_id)
            ):
                records.append(
                    {"eventId": str(event_id), "assistPlayerId": str(assist_player_id)}
                )
        return records

    def _prepare_subbed_out(self, subs_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Build SUBBED_OUT parameter dicts (player leaving the pitch)."""
        records: List[Dict[str, Any]] = []
        event_id_col = "game_event_id" if "game_event_id" in subs_df.columns else "id"
        player_col = "player_id" if "player_id" in subs_df.columns else None
        if not player_col:
            return records
        for row in subs_df.to_dict(orient="records"):
            event_id = row.get(event_id_col)
            player_id = row.get(player_col)
            if event_id and not pd.isna(event_id) and player_id and not pd.isna(player_id):
                records.append({"eventId": str(event_id), "playerId": str(player_id)})
        return records

    def _prepare_subbed_in(self, subs_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Build SUBBED_IN parameter dicts (player entering the pitch)."""
        records: List[Dict[str, Any]] = []
        event_id_col = "game_event_id" if "game_event_id" in subs_df.columns else "id"
        player_in_col = "player_in_id" if "player_in_id" in subs_df.columns else None
        if not player_in_col:
            return records
        for row in subs_df.to_dict(orient="records"):
            event_id = row.get(event_id_col)
            player_in_id = row.get(player_in_col)
            if event_id and not pd.isna(event_id) and player_in_id and not pd.isna(player_in_id):
                records.append({"eventId": str(event_id), "playerInId": str(player_in_id)})
        return records

    def _prepare_received_card(self, cards_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Build RECEIVED_CARD parameter dicts from card events."""
        records: List[Dict[str, Any]] = []
        event_id_col = "game_event_id" if "game_event_id" in cards_df.columns else "id"
        player_col = "player_id" if "player_id" in cards_df.columns else None
        if not player_col:
            return records
        for row in cards_df.to_dict(orient="records"):
            event_id = row.get(event_id_col)
            player_id = row.get(player_col)
            if event_id and not pd.isna(event_id) and player_id and not pd.isna(player_id):
                records.append({"eventId": str(event_id), "playerId": str(player_id)})
        return records

    def load(self, data_dir: str, dev_mode: bool = False) -> None:
        """
        Load game_events.csv into Neo4j as GameEvent nodes and semantic edges.

        Processing order (dependency-safe):
        1. Merge all GameEvent nodes + OCCURRED_IN + INVOLVED_IN.
        2. SCORED edges (Goals).
        3. ASSISTED edges (Goals with assist).
        4. SUBBED_OUT edges (Substitutions – player leaving).
        5. SUBBED_IN edges  (Substitutions – player entering).
        6. RECEIVED_CARD edges (Cards).

        Args:
            data_dir: Directory path containing raw CSV files.
            dev_mode: If True, filter events whose player_id is in active player_ids.
        """
        logger.info("Starting Game Events Data Ingestion (dev_mode=%s)", dev_mode)

        dev_filters = self.get_dev_filter_ids(data_dir) if dev_mode else None

        events_path = os.path.join(data_dir, "game_events.csv")
        if not os.path.exists(events_path):
            logger.warning("game_events.csv not found at %s. Skipping.", events_path)
            return

        events_df = self.read_csv(events_path)

        # DEV_MODE: filter events where the primary player is in our allowed set
        if dev_mode and dev_filters and "player_ids" in dev_filters:
            valid_player_ids = {str(pid).strip() for pid in dev_filters["player_ids"]}
            player_col = "player_id" if "player_id" in events_df.columns else None
            if player_col:
                events_df = events_df[
                    events_df[player_col].astype(str).str.strip().isin(valid_player_ids)
                ]
                logger.info(
                    "Filtered game_events.csv to %d rows for DEV_MODE", len(events_df)
                )

        # ── Step 1: merge base event nodes + structural edges ────────────────
        event_node_records = self._prepare_event_nodes(events_df)
        total_nodes = self.execute_batch(self.CYPHER_MERGE_GAME_EVENT_NODE, event_node_records)
        logger.info("Merged %d GameEvent nodes with OCCURRED_IN / INVOLVED_IN edges.", total_nodes)

        # ── Steps 2–6: semantic type-specific edges ──────────────────────────
        goals_df, subs_df, cards_df = self._split_by_type(events_df)

        scored_records = self._prepare_scored(goals_df)
        self.execute_batch(self.CYPHER_MERGE_SCORED, scored_records)
        logger.info("Merged %d SCORED edges.", len(scored_records))

        assisted_records = self._prepare_assisted(goals_df)
        self.execute_batch(self.CYPHER_MERGE_ASSISTED, assisted_records)
        logger.info("Merged %d ASSISTED edges.", len(assisted_records))

        subbed_out_records = self._prepare_subbed_out(subs_df)
        self.execute_batch(self.CYPHER_MERGE_SUBBED_OUT, subbed_out_records)
        logger.info("Merged %d SUBBED_OUT edges.", len(subbed_out_records))

        subbed_in_records = self._prepare_subbed_in(subs_df)
        self.execute_batch(self.CYPHER_MERGE_SUBBED_IN, subbed_in_records)
        logger.info("Merged %d SUBBED_IN edges.", len(subbed_in_records))

        received_card_records = self._prepare_received_card(cards_df)
        self.execute_batch(self.CYPHER_MERGE_RECEIVED_CARD, received_card_records)
        logger.info("Merged %d RECEIVED_CARD edges.", len(received_card_records))

        logger.info("Completed Game Events Data Ingestion.")