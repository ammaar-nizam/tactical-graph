"""
Transfers loader module for TacticalGraph data pipeline.

Ingests transfers.csv (davidcariboo) and supplementary CSVs (mexwell) to create
(Player)-[:TRANSFERRED_TO]->(Club) relationship edges with full transfer metadata.
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
    Loader for player transfer data from transfers.csv (davidcariboo) and
    supplementary mexwell CSVs. Davidcariboo is the authoritative source;
    mexwell enriches existing edges with transferPeriod only.
    """

    # MATCH nodes (not MERGE) — nodes exist from entities_loader, no write-locks needed.
    # Canonical merge key (transferSeason, fromClubId) is always populated in DC and unique.
    # ON CREATE SET writes all properties; ON MATCH SET is an idempotent re-run guard.
    CYPHER_MERGE_TRANSFERS = """
    UNWIND $batch AS row
    MATCH (p:Player    {id: row.playerId})
    MATCH (toClub:Club {id: row.toClubId})
    MERGE (p)-[t:TRANSFERRED_TO {
        transferSeason: row.transferSeason,
        fromClubId:     row.fromClubId
    }]->(toClub)
    ON CREATE SET
        t.transferDate          = row.transferDate,
        t.transferFee           = row.transferFee,
        t.marketValueAtTransfer = row.marketValueAtTransfer,
        t.fromClubName          = row.fromClubName,
        t.transferPeriod        = null
    ON MATCH SET
        t.transferDate = CASE
            WHEN t.transferDate IS NULL THEN row.transferDate
            ELSE t.transferDate
        END
    """

    # Enrichment-only: MATCH existing edges, never create new ones.
    # fromClubName is excluded from WHERE — 24.8% mismatch rate across datasets.
    # Season is pre-normalised to YY/YY in Python before this query runs.
    # Only transferPeriod is SET — it is the sole net-new property mexwell provides.
    CYPHER_ENRICH_TRANSFER_PERIOD = """
    UNWIND $batch AS row
    MATCH (p:Player)    WHERE toLower(p.name)      = toLower(row.playerName)
    MATCH (toClub:Club) WHERE toLower(toClub.name) = toLower(row.toClubName)
    MATCH (p)-[t:TRANSFERRED_TO]->(toClub)
    WHERE t.transferSeason = row.transferSeason
    SET t.transferPeriod = row.transferPeriod
    """

    # IN TRANSACTIONS batching prevents OOM on large label sets (Club: 13,615 stubs).
    CYPHER_DELETE_STUB_CLUBS = """
    MATCH (c:Club) WHERE size(keys(c)) = 1
    CALL { WITH c DETACH DELETE c } IN TRANSACTIONS OF 5000 ROWS
    """

    CYPHER_DELETE_STUB_PLAYERS = """
    MATCH (p:Player) WHERE size(keys(p)) = 1
    CALL { WITH p DETACH DELETE p } IN TRANSACTIONS OF 5000 ROWS
    """

    CYPHER_DELETE_STUB_COMPETITIONS = """
    MATCH (c:Competition) WHERE size(keys(c)) = 1
    DETACH DELETE c
    """

    CYPHER_DELETE_STUB_COUNTRIES = """
    MATCH (c:Country) WHERE size(keys(c)) = 1
    DETACH DELETE c
    """

    @staticmethod
    def _normalise_season(season: str) -> Optional[str]:
        """Convert mexwell YYYY/YYYY season to davidcariboo YY/YY format."""
        if not season or not isinstance(season, str):
            return None
        parts = season.strip().split("/")
        if len(parts) == 2 and len(parts[0]) == 4 and parts[0].isdigit() and parts[1].isdigit():
            return f"{parts[0][2:]}/{parts[1][2:]}"
        return season.strip()

    def _prepare_transfers(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Map davidcariboo transfers DataFrame to camelCase Cypher parameter dicts."""
        records: List[Dict[str, Any]] = []

        for row in df.to_dict(orient="records"):
            player_id = row.get("player_id")
            to_club_id = row.get("to_club_id")

            if player_id is None or pd.isna(player_id):
                continue
            if to_club_id is None or pd.isna(to_club_id):
                continue

            from_club_id = row.get("from_club_id")
            transfer_date = row.get("transfer_date") or row.get("date")
            transfer_fee = row.get("transfer_fee")
            market_value = row.get("market_value_in_eur") or row.get("market_value_at_transfer")
            from_club_name = row.get("from_club_name")
            transfer_season = row.get("transfer_season") or row.get("season")

            records.append(
                {
                    "playerId": str(player_id),
                    "toClubId": str(to_club_id),
                    "fromClubId": str(from_club_id) if from_club_id is not None and not pd.isna(from_club_id) else None,
                    "fromClubName": from_club_name if from_club_name and not pd.isna(from_club_name) else None,
                    "transferDate": str(transfer_date) if transfer_date and not pd.isna(transfer_date) else None,
                    "transferSeason": str(transfer_season) if transfer_season and not pd.isna(transfer_season) else None,
                    "transferFee": float(transfer_fee) if transfer_fee is not None and not pd.isna(transfer_fee) else None,
                    "marketValueAtTransfer": float(market_value) if market_value is not None and not pd.isna(market_value) else None,
                }
            )

        return records

    def _prepare_supplementary_transfers(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Map mexwell DataFrame to enrichment-only Cypher parameter dicts.

        Normalises YYYY/YYYY seasons to YY/YY and deduplicates by
        (playerName, toClubName, transferSeason), keeping the highest-fee row
        to resolve fan-out where multiple mexwell rows match one DC edge.
        """
        if "transfer_movement" in df.columns:
            df = df[df["transfer_movement"] == "in"].copy()

        if df.empty:
            return []

        df["season_normalised"] = df["season"].apply(
            lambda s: self._normalise_season(str(s)) if not pd.isna(s) else None
        )
        df["fee_cleaned_num"] = pd.to_numeric(df["fee_cleaned"], errors="coerce")

        # Keep highest-fee row per group to avoid multiple SETs on the same relationship.
        df_deduped = (
            df.sort_values("fee_cleaned_num", ascending=False)
            .drop_duplicates(subset=["player_name", "club_name", "season_normalised"], keep="first")
        )

        records: List[Dict[str, Any]] = []

        for row in df_deduped.to_dict(orient="records"):
            player_name = row.get("player_name")
            if pd.isna(player_name):
                continue
            player_name = str(player_name).strip()

            to_club_name = row.get("club_name")
            if pd.isna(to_club_name):
                continue
            to_club_name = str(to_club_name).strip()

            season_normalised = row.get("season_normalised")
            if not season_normalised:
                continue

            transfer_period = row.get("transfer_period")
            transfer_period = str(transfer_period).strip() if not pd.isna(transfer_period) else None
            if not transfer_period:
                continue

            records.append(
                {
                    "playerName": player_name,
                    "toClubName": to_club_name,
                    "transferSeason": season_normalised,
                    "transferPeriod": transfer_period,
                }
            )

        return records

    def load(self, data_dir: Union[str, Path], dev_mode: bool = False) -> None:
        """Load transfers.csv into Neo4j as TRANSFERRED_TO relationship edges."""
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

    def load_supplementary_transfers(
        self,
        transfer_data_dir: Union[str, Path],
        primary_data_dir: Union[str, Path],
        dev_mode: bool = False,
    ) -> None:
        """Enrich existing TRANSFERRED_TO edges with transferPeriod from mexwell CSVs."""
        logger.info("Starting Supplementary Transfers Enrichment (dev_mode=%s)", dev_mode)

        transfer_dir = Path(transfer_data_dir)
        if not transfer_dir.exists() or not transfer_dir.is_dir():
            logger.warning(
                "Supplementary transfers directory not found at %s. Skipping.", transfer_dir
            )
            return

        csv_files = list(transfer_dir.glob("*.csv"))
        if not csv_files:
            logger.warning("No CSV files found in %s.", transfer_dir)
            return

        dfs: List[pd.DataFrame] = []
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

        if dev_mode:
            dev_filters = self.get_dev_filter_ids(primary_data_dir)
            if dev_filters and "player_ids" in dev_filters:
                valid_player_ids = {str(pid).strip() for pid in dev_filters["player_ids"]}
                players_path = os.path.join(primary_data_dir, "players.csv")
                if os.path.exists(players_path):
                    try:
                        players_df = self.read_csv(players_path)
                        player_col = "player_id" if "player_id" in players_df.columns else "id"
                        players_df = players_df[
                            players_df[player_col].astype(str).str.strip().isin(valid_player_ids)
                        ]
                        if "name" in players_df.columns:
                            valid_names = set(players_df["name"].str.lower().dropna())
                            if "player_name" in combined_df.columns:
                                combined_df = combined_df[
                                    combined_df["player_name"].str.lower().isin(valid_names)
                                ]
                    except Exception as e:
                        logger.error(
                            "Failed to apply dev_mode filter for supplementary transfers: %s", e
                        )

        filtered_rows = len(combined_df)
        transfer_records = self._prepare_supplementary_transfers(combined_df)
        prepared_count = len(transfer_records)

        total_enriched = self.execute_batch(self.CYPHER_ENRICH_TRANSFER_PERIOD, transfer_records)

        logger.info(
            "Supplementary Transfers Enrichment Summary: "
            "%d read, %d after dev_mode filter, %d prepared (deduped), %d enriched.",
            total_rows,
            filtered_rows,
            prepared_count,
            total_enriched,
        )

    def cleanup_stub_nodes(self) -> None:
        """Remove stub nodes (only 'id' property) from Club, Player, Competition, Country."""
        logger.info("Starting stub node cleanup across Club, Player, Competition, Country.")

        stub_queries = [
            ("Club", self.CYPHER_DELETE_STUB_CLUBS),
            ("Player", self.CYPHER_DELETE_STUB_PLAYERS),
            ("Competition", self.CYPHER_DELETE_STUB_COMPETITIONS),
            ("Country", self.CYPHER_DELETE_STUB_COUNTRIES),
        ]

        for label, query in stub_queries:
            try:
                self.db.execute_query(query)
                logger.info("Stub cleanup complete for label: %s", label)
            except Exception as e:
                logger.error("Stub cleanup failed for label %s: %s", label, e)
                raise