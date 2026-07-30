"""
Transfers loader module for TacticalGraph data pipeline.

Ingests transfers.csv and creates (Player)-[:TRANSFERRED_TO]->(Club) relationship
edges with full transfer metadata properties.
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


class TransfersLoader(BaseLoader):
    """
    Loader for player transfer data from transfers.csv.

    Creates (Player)-[:TRANSFERRED_TO {transferDate, transferSeason, transferFee,
    marketValueAtTransfer, fromClubId, fromClubName}]->(Club) relationships.
    """

    # The edge points to the destination club; fromClubId stored as property
    # for rapid filtering without multi-hop temporal traversals.
    CYPHER_MERGE_TRANSFERS = """
    UNWIND $batch AS row
    MERGE (p:Player {id: row.playerId})
    MERGE (toClub:Club {id: row.toClubId})
    MERGE (p)-[t:TRANSFERRED_TO {transferDate: row.transferDate, fromClubId: row.fromClubId}]->(toClub)
    SET t.transferSeason         = row.transferSeason,
        t.transferFee            = toFloat(row.transferFee),
        t.marketValueAtTransfer  = toFloat(row.marketValueAtTransfer),
        t.fromClubName           = row.fromClubName
    """

    def _prepare_transfers(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Map transfers DataFrame columns to camelCase Cypher parameter dicts.

        Args:
            df: Raw transfers DataFrame.

        Returns:
            List[Dict[str, Any]]: Sanitized records ready for batch execution.
        """
        records: List[Dict[str, Any]] = []
        for row in df.to_dict(orient="records"):
            player_id = row.get("player_id")
            to_club_id = row.get("to_club_id")

            # Both player and destination club are required to create the edge
            if player_id is None or pd.isna(player_id):
                continue
            if to_club_id is None or pd.isna(to_club_id):
                continue

            from_club_id = row.get("from_club_id")
            transfer_date = row.get("transfer_date") or row.get("date")
            transfer_fee = row.get("transfer_fee")
            market_value = row.get("market_value_in_eur") or row.get("market_value_at_transfer")
            from_club_name = row.get("from_club_name")
            transfer_season = row.get("season") or row.get("transfer_season")

            records.append(
                {
                    "playerId": str(player_id),
                    "toClubId": str(to_club_id),
                    "fromClubId": str(from_club_id) if from_club_id and not pd.isna(from_club_id) else None,
                    "fromClubName": from_club_name if from_club_name and not pd.isna(from_club_name) else None,
                    "transferDate": str(transfer_date) if transfer_date and not pd.isna(transfer_date) else None,
                    "transferSeason": str(transfer_season) if transfer_season and not pd.isna(transfer_season) else None,
                    "transferFee": float(transfer_fee) if transfer_fee is not None and not pd.isna(transfer_fee) else None,
                    "marketValueAtTransfer": float(market_value) if market_value is not None and not pd.isna(market_value) else None,
                }
            )
        return records

    def load(self, data_dir: str, dev_mode: bool = False) -> None:
        """
        Load transfers.csv into Neo4j as TRANSFERRED_TO relationship edges.

        Args:
            data_dir: Directory path containing raw CSV files.
            dev_mode: If True, filter transfers to active player_ids from utils.py.
        """
        logger.info("Starting Transfers Data Ingestion (dev_mode=%s)", dev_mode)

        dev_filters = self.get_dev_filter_ids(data_dir) if dev_mode else None

        transfers_path = os.path.join(data_dir, "transfers.csv")
        if not os.path.exists(transfers_path):
            logger.warning("transfers.csv not found at %s. Skipping.", transfers_path)
            return

        transfers_df = self.read_csv(transfers_path)

        if dev_mode and dev_filters and "player_ids" in dev_filters:
            valid_player_ids = {str(pid).strip() for pid in dev_filters["player_ids"]}
            player_col = "player_id" if "player_id" in transfers_df.columns else "id"
            transfers_df = transfers_df[
                transfers_df[player_col].astype(str).str.strip().isin(valid_player_ids)
            ]
            logger.info(
                "Filtered transfers.csv to %d rows for DEV_MODE (player_ids filter)",
                len(transfers_df),
            )

        transfer_records = self._prepare_transfers(transfers_df)
        total = self.execute_batch(self.CYPHER_MERGE_TRANSFERS, transfer_records)
        logger.info("Completed Transfers Data Ingestion. Processed %d transfer records.", total)