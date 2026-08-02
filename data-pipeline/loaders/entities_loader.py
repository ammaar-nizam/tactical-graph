"""
Entities loader module for TacticalGraph data pipeline.

Loads clubs.csv, players.csv, and player_valuations.csv into Neo4j.
Creates (:Club), (:Player), and (:PlayerValuation) nodes, as well as
[:COMPETES_IN], [:PLAYS_FOR], [:REPRESENTS], and [:HAS_VALUATION] relationships.
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


class EntitiesLoader(BaseLoader):
    """
    Loader for entity datasets (Clubs, Players, Player Valuations).
    """

    CYPHER_MERGE_CLUBS = """
    UNWIND $batch AS row
    MERGE (c:Club {id: row.id})
    SET c.code = row.code,
        c.name = row.name,
        c.totalMarketValue = toFloat(row.totalMarketValue),
        c.squadSize = toInteger(row.squadSize),
        c.averageAge = toFloat(row.averageAge),
        c.foreignersNumber = toInteger(row.foreignersNumber),
        c.foreignersPercentage = toFloat(row.foreignersPercentage),
        c.nationalTeamPlayers = toInteger(row.nationalTeamPlayers),
        c.stadiumName = row.stadiumName,
        c.stadiumSeats = toInteger(row.stadiumSeats),
        c.netTransferRecord = row.netTransferRecord,
        c.coachName = row.coachName,
        c.lastSeason = toInteger(row.lastSeason),
        c.fileName = row.fileName,
        c.url = row.url
    WITH c, row
    WHERE row.competitionId IS NOT NULL
    MERGE (comp:Competition {id: row.competitionId})
    MERGE (c)-[:COMPETES_IN]->(comp)
    """

    CYPHER_MERGE_PLAYERS = """
    UNWIND $batch AS row
    MERGE (p:Player {id: row.id})
    SET p.firstName = row.firstName,
        p.lastName = row.lastName,
        p.name = row.name,
        p.lastSeason = toInteger(row.lastSeason),
        p.code = row.code,
        p.countryOfBirth = row.countryOfBirth,
        p.cityOfBirth = row.cityOfBirth,
        p.countryOfCitizenship = row.countryOfCitizenship,
        p.dateOfBirth = row.dateOfBirth,
        p.subPosition = row.subPosition,
        p.position = row.position,
        p.foot = row.foot,
        p.heightInCm = toInteger(row.heightInCm),
        p.contractExpirationDate = row.contractExpirationDate,
        p.agentName = row.agentName,
        p.imageUrl = row.imageUrl,
        p.internationalCaps = toInteger(row.internationalCaps),
        p.internationalGoals = toInteger(row.internationalGoals),
        p.url = row.url
    WITH p, row
    WHERE row.currentClubId IS NOT NULL
    MERGE (cl:Club {id: row.currentClubId})
    MERGE (p)-[:PLAYS_FOR]->(cl)
    WITH p, row
    WHERE row.nationalTeamId IS NOT NULL
    MERGE (nt:NationalTeam {id: row.nationalTeamId})
    MERGE (p)-[:REPRESENTS]->(nt)
    """

    CYPHER_MERGE_VALUATIONS = """
    UNWIND $batch AS row
    MERGE (pv:PlayerValuation {id: row.id})
    SET pv.date             = row.date,
        pv.marketValueInEur = toFloat(row.marketValueInEur)
    WITH pv, row
    WHERE row.playerId IS NOT NULL
    MERGE (p:Player {id: row.playerId})
    MERGE (p)-[:HAS_VALUATION]->(pv)
    """

    def _prepare_clubs(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Map clubs DataFrame columns to camelCase parameter dicts."""
        records = []
        for row in df.to_dict(orient="records"):
            cid = row.get("club_id") or row.get("id")
            if cid is None or pd.isna(cid):
                continue
            comp_id = row.get("domestic_competition_id") or row.get("competition_id")
            records.append({
                "id": str(cid),
                "code": row.get("club_code") or row.get("code"),
                "name": row.get("name"),
                "totalMarketValue": row.get("total_market_value") or row.get("totalMarketValue"),
                "squadSize": row.get("squad_size") or row.get("squadSize"),
                "averageAge": row.get("average_age") or row.get("averageAge"),
                "foreignersNumber": row.get("foreigners_number") or row.get("foreignersNumber"),
                "foreignersPercentage": row.get("foreigners_percentage") or row.get("foreignersPercentage"),
                "nationalTeamPlayers": row.get("national_team_players") or row.get("nationalTeamPlayers"),
                "stadiumName": row.get("stadium_name") or row.get("stadiumName"),
                "stadiumSeats": row.get("stadium_seats") or row.get("stadiumSeats"),
                "netTransferRecord": row.get("net_transfer_record") or row.get("netTransferRecord"),
                "coachName": row.get("coach_name") or row.get("coachName"),
                "lastSeason": row.get("last_season") or row.get("lastSeason"),
                "fileName": row.get("filename") or row.get("file_name") or row.get("fileName"),
                "url": row.get("url"),
                "competitionId": str(comp_id) if comp_id and not pd.isna(comp_id) else None,
            })
        return records

    def _prepare_players(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Map players DataFrame columns to camelCase parameter dicts."""
        records = []
        for row in df.to_dict(orient="records"):
            pid = row.get("player_id") or row.get("id")
            if pid is None or pd.isna(pid):
                continue
            club_id = row.get("current_club_id") or row.get("club_id")
            nt_id = row.get("national_team_id")
            records.append({
                "id": str(pid),
                "firstName": row.get("first_name") or row.get("firstName"),
                "lastName": row.get("last_name") or row.get("lastName"),
                "name": row.get("name"),
                "lastSeason": row.get("last_season") or row.get("lastSeason"),
                "code": row.get("player_code") or row.get("code"),
                "countryOfBirth": row.get("country_of_birth") or row.get("countryOfBirth"),
                "cityOfBirth": row.get("city_of_birth") or row.get("cityOfBirth"),
                "countryOfCitizenship": row.get("country_of_citizenship") or row.get("countryOfCitizenship"),
                "dateOfBirth": row.get("date_of_birth") or row.get("dateOfBirth"),
                "subPosition": row.get("sub_position") or row.get("subPosition"),
                "position": row.get("position"),
                "foot": row.get("foot"),
                "heightInCm": row.get("height_in_cm") or row.get("heightInCm"),
                "contractExpirationDate": row.get("contract_expiration_date") or row.get("contractExpirationDate"),
                "agentName": row.get("agent_name") or row.get("agentName"),
                "imageUrl": row.get("image_url") or row.get("imageUrl"),
                "internationalCaps": row.get("international_caps") or row.get("internationalCaps"),
                "internationalGoals": row.get("international_goals") or row.get("internationalGoals"),
                "url": row.get("url"),
                "currentClubId": str(club_id) if club_id and not pd.isna(club_id) else None,
                "nationalTeamId": str(nt_id) if nt_id and not pd.isna(nt_id) else None,
            })
        return records

    def _prepare_valuations(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Map player_valuations DataFrame columns to camelCase parameter dicts."""
        records = []
        for row in df.to_dict(orient="records"):
            pid = row.get("player_id")
            val_date = row.get("date") or row.get("valuation_date")
            if pid is None or pd.isna(pid) or val_date is None or pd.isna(val_date):
                continue
            val_id = f"{str(pid).strip()}_{str(val_date).strip()}"
            records.append({
                "id": val_id,
                "playerId": str(pid),
                "date": str(val_date),
                "marketValueInEur": row.get("market_value_in_eur") or row.get("market_value") or row.get("marketValueInEur"),
            })
        return records


    def load(self, data_dir: Union[str, Path], dev_mode: bool = False) -> None:
        """
        Load entity domain CSV files into Neo4j.

        Args:
            data_dir: Path to directory containing raw CSV files.
            dev_mode: If True, filter DataFrames using allowed club_ids and player_ids from utils.py.
        """
        logger.info("Starting Entities Data Ingestion (dev_mode=%s)", dev_mode)

        dev_filters = self.get_dev_filter_ids(data_dir) if dev_mode else None

        # 1. Ingest Clubs
        clubs_path = os.path.join(data_dir, "clubs.csv")
        if os.path.exists(clubs_path):
            clubs_df = self.read_csv(clubs_path)
            if dev_mode and dev_filters and "club_ids" in dev_filters:
                club_col = "club_id" if "club_id" in clubs_df.columns else "id"
                valid_club_ids = {str(cid).strip() for cid in dev_filters["club_ids"]}
                clubs_df = clubs_df[
                    clubs_df[club_col].astype(str).str.strip().isin(valid_club_ids)
                ]
                logger.info("Filtered clubs.csv to %d rows for DEV_MODE", len(clubs_df))
            club_records = self._prepare_clubs(clubs_df)
            self.execute_batch(self.CYPHER_MERGE_CLUBS, club_records)
        else:
            logger.warning("clubs.csv not found at %s. Skipping.", clubs_path)

        # 2. Ingest Players
        players_path = os.path.join(data_dir, "players.csv")
        if os.path.exists(players_path):
            players_df = self.read_csv(players_path)
            if dev_mode and dev_filters and "player_ids" in dev_filters:
                player_col = "player_id" if "player_id" in players_df.columns else "id"
                valid_player_ids = {str(pid).strip() for pid in dev_filters["player_ids"]}
                players_df = players_df[
                    players_df[player_col].astype(str).str.strip().isin(valid_player_ids)
                ]
                logger.info("Filtered players.csv to %d rows for DEV_MODE", len(players_df))
            player_records = self._prepare_players(players_df)
            self.execute_batch(self.CYPHER_MERGE_PLAYERS, player_records)
        else:
            logger.warning("players.csv not found at %s. Skipping.", players_path)

        # 3. Ingest Player Valuations
        valuations_path = os.path.join(data_dir, "player_valuations.csv")
        if os.path.exists(valuations_path):
            valuations_df = self.read_csv(valuations_path)
            if dev_mode and dev_filters and "player_ids" in dev_filters:
                valid_player_ids = {str(pid).strip() for pid in dev_filters["player_ids"]}
                valuations_df = valuations_df[
                    valuations_df["player_id"].astype(str).str.strip().isin(valid_player_ids)
                ]
                logger.info("Filtered player_valuations.csv to %d rows for DEV_MODE", len(valuations_df))
            valuation_records = self._prepare_valuations(valuations_df)
            self.execute_batch(self.CYPHER_MERGE_VALUATIONS, valuation_records)
        else:
            logger.warning("player_valuations.csv not found at %s. Skipping.", valuations_path)

        logger.info("Completed Entities Data Ingestion.")