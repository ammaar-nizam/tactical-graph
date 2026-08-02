"""
Transfers loader module for TacticalGraph data pipeline.

Ingests transfers.csv and creates (Player)-[:TRANSFERRED_TO]->(Club) relationship
edges with full transfer metadata properties.
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

    CYPHER_MERGE_TRANSFERS_BY_NAME = """
    UNWIND $batch AS row
    MATCH (p:Player) WHERE toLower(p.name) = toLower(row.playerName)
    MATCH (toClub:Club) WHERE toLower(toClub.name) = toLower(row.toClubName)
    MERGE (p)-[t:TRANSFERRED_TO {transferSeason: row.transferSeason, fromClubName: row.fromClubName}]->(toClub)
    SET t.transferFee = CASE WHEN row.transferFee IS NOT NULL THEN toFloat(row.transferFee) ELSE t.transferFee END,
        t.transferPeriod = row.transferPeriod
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

    def _prepare_supplementary_transfers(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for row in df.to_dict(orient="records"):
            player_name = row.get("player_name")
            if pd.isna(player_name):
                continue
            player_name = str(player_name).strip()
            
            to_club_name = row.get("club_name")
            if pd.isna(to_club_name):
                continue
            to_club_name = str(to_club_name).strip()

            from_club_name = row.get("club_involved_name")
            from_club_name = str(from_club_name).strip() if not pd.isna(from_club_name) else None
            
            fee_cleaned = row.get("fee_cleaned")
            transfer_fee = float(fee_cleaned) * 1_000_000 if not pd.isna(fee_cleaned) else None
            
            season = row.get("season")
            transfer_season = str(season) if not pd.isna(season) else None
            
            transfer_period = row.get("transfer_period")
            transfer_period = str(transfer_period) if not pd.isna(transfer_period) else None

            records.append({
                "playerName": player_name,
                "toClubName": to_club_name,
                "fromClubName": from_club_name,
                "transferFee": transfer_fee,
                "transferSeason": transfer_season,
                "transferPeriod": transfer_period
            })
        return records

    def load(self, data_dir: Union[str, Path], dev_mode: bool = False) -> None:
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

    def load_supplementary_transfers(self, transfer_data_dir: Union[str, Path], primary_data_dir: Union[str, Path], dev_mode: bool = False) -> None:
        logger.info("Starting Supplementary Transfers Ingestion (dev_mode=%s)", dev_mode)
        
        transfer_dir = Path(transfer_data_dir)
        if not transfer_dir.exists() or not transfer_dir.is_dir():
            logger.warning("Supplementary transfers directory not found at %s. Skipping.", transfer_dir)
            return
            
        csv_files = list(transfer_dir.glob("*.csv"))
        if not csv_files:
            logger.warning("No CSV files found in %s.", transfer_dir)
            return
            
        dfs = []
        for csv_file in csv_files:
            try:
                dfs.append(self.read_csv(csv_file))
            except Exception as e:
                logger.error("Failed to read %s: %s", csv_file, e)
                
        if not dfs:
            logger.warning("No valid CSV files could be read in %s.", transfer_dir)
            return
            
        combined_df = pd.concat(dfs, ignore_index=True)
        total_rows = len(combined_df)
        
        if 'transfer_movement' in combined_df.columns:
            combined_df = combined_df[combined_df['transfer_movement'] == 'in']
        in_rows = len(combined_df)
        
        if dev_mode:
            dev_filters = self.get_dev_filter_ids(primary_data_dir)
            if dev_filters and "player_ids" in dev_filters:
                valid_player_ids = {str(pid).strip() for pid in dev_filters["player_ids"]}
                players_path = os.path.join(primary_data_dir, "players.csv")
                if os.path.exists(players_path):
                    try:
                        players_df = self.read_csv(players_path)
                        player_col = "player_id" if "player_id" in players_df.columns else "id"
                        players_df = players_df[players_df[player_col].astype(str).str.strip().isin(valid_player_ids)]
                        
                        if 'name' in players_df.columns:
                            valid_names = set(players_df['name'].str.lower().dropna())
                            if 'player_name' in combined_df.columns:
                                combined_df = combined_df[combined_df['player_name'].str.lower().isin(valid_names)]
                    except Exception as e:
                        logger.error("Failed to apply dev_mode filter for supplementary transfers: %s", e)
                        
        filtered_rows = len(combined_df)
        
        transfer_records = self._prepare_supplementary_transfers(combined_df)
        prepared_count = len(transfer_records)
        
        total_merged = self.execute_batch(self.CYPHER_MERGE_TRANSFERS_BY_NAME, transfer_records)
        
        logger.info(
            "Supplementary Transfers Summary: %d read, %d 'in', %d after dev_mode, %d prepared, %d merged.", 
            total_rows, in_rows, filtered_rows, prepared_count, total_merged
        )