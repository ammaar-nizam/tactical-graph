"""
Utility functions for TacticalGraph data pipeline.

Provides dataset filtering and sub-graph extraction capabilities for development mode (DEV_MODE).
"""

import logging
import os
from typing import Any, Dict, Set

import pandas as pd

logger = logging.getLogger(__name__)


def get_dev_subgraph_ids(
    data_dir: str, target_competition_id: str = "GB1"
) -> Dict[str, Set[Any]]:
    """
    Extract active competition, club, and player IDs for development mode (DEV_MODE).

    Filters `clubs.csv` within `data_dir` for `target_competition_id` (e.g., Premier League 'GB1'),
    extracts the matching `club_id`s, and retrieves all corresponding `player_id`s from `players.csv`.

    Args:
        data_dir: Directory path containing the Kaggle CSV files (e.g. 'clubs.csv', 'players.csv').
        target_competition_id: Target competition code (default: 'GB1' for Premier League).

    Returns:
        Dict[str, Set[Any]]: Dictionary containing sets of active IDs:
            {
                "competition_ids": {target_competition_id},
                "club_ids": set of active club IDs,
                "player_ids": set of active player IDs
            }

    Raises:
        FileNotFoundError: If data_dir or required CSV files do not exist.
        KeyError: If required ID columns are missing from the CSV files.
        Exception: For general pandas or I/O processing errors.
    """
    if not os.path.exists(data_dir):
        error_msg = f"Data directory does not exist: {data_dir}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    clubs_path = os.path.join(data_dir, "clubs.csv")
    players_path = os.path.join(data_dir, "players.csv")

    if not os.path.exists(clubs_path):
        error_msg = f"Clubs file missing at: {clubs_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    if not os.path.exists(players_path):
        error_msg = f"Players file missing at: {players_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        logger.info("Reading clubs dataset from %s", clubs_path)
        clubs_df = pd.read_csv(clubs_path)

        # Handle column naming variations in Kaggle transfermarkt dataset
        comp_col = (
            "domestic_competition_id"
            if "domestic_competition_id" in clubs_df.columns
            else ("competition_id" if "competition_id" in clubs_df.columns else None)
        )
        if not comp_col:
            raise KeyError(
                "Neither 'domestic_competition_id' nor 'competition_id' found in clubs.csv"
            )

        club_id_col = (
            "club_id"
            if "club_id" in clubs_df.columns
            else ("id" if "id" in clubs_df.columns else None)
        )
        if not club_id_col:
            raise KeyError("Neither 'club_id' nor 'id' found in clubs.csv")

        # Filter clubs belonging to target competition
        target_comp_str = str(target_competition_id).strip()
        filtered_clubs_df = clubs_df[
            clubs_df[comp_col].astype(str).str.strip() == target_comp_str
        ]

        active_club_ids: Set[Any] = set(
            filtered_clubs_df[club_id_col].dropna().unique()
        )
        logger.info(
            "Extracted %d club IDs for competition '%s'",
            len(active_club_ids),
            target_competition_id,
        )

        logger.info("Reading players dataset from %s", players_path)
        players_df = pd.read_csv(players_path)

        player_id_col = (
            "player_id"
            if "player_id" in players_df.columns
            else ("id" if "id" in players_df.columns else None)
        )
        if not player_id_col:
            raise KeyError("Neither 'player_id' nor 'id' found in players.csv")

        player_club_col = (
            "current_club_id"
            if "current_club_id" in players_df.columns
            else ("club_id" if "club_id" in players_df.columns else None)
        )
        if not player_club_col:
            raise KeyError("Neither 'current_club_id' nor 'club_id' found in players.csv")

        # Normalize active club IDs for matching (both raw and string forms)
        club_ids_as_str = {str(cid).strip() for cid in active_club_ids}
        filtered_players_df = players_df[
            players_df[player_club_col].astype(str).str.strip().isin(club_ids_as_str)
        ]

        active_player_ids: Set[Any] = set(
            filtered_players_df[player_id_col].dropna().unique()
        )
        logger.info(
            "Extracted %d player IDs associated with the %d filtered clubs",
            len(active_player_ids),
            len(active_club_ids),
        )

        result = {
            "competition_ids": {target_competition_id},
            "club_ids": active_club_ids,
            "player_ids": active_player_ids,
        }
        return result

    except (FileNotFoundError, KeyError):
        raise
    except Exception as e:
        logger.error(
            "Unexpected error extracting dev subgraph IDs from %s: %s", data_dir, e
        )
        raise
