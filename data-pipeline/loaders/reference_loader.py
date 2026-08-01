"""
Reference loader module for TacticalGraph data pipeline.

Loads countries.csv, competitions.csv, and national_teams.csv into Neo4j.
Creates (:Country), (:Competition), and (:NationalTeam) nodes, as well as
[:LOCATED_IN] and [:REPRESENTS_COUNTRY] relationships using parameterized Cypher.
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


class ReferenceLoader(BaseLoader):
    """
    Loader for reference datasets (Countries, Competitions, National Teams).
    """

    CYPHER_MERGE_COUNTRIES = """
    UNWIND $batch AS row
    MERGE (c:Country {id: row.id})
    SET c.name = row.name,
        c.code = row.code,
        c.confederation = row.confederation,
        c.totalClubs = toInteger(row.totalClubs),
        c.totalPlayers = toInteger(row.totalPlayers),
        c.averageAge = toFloat(row.averageAge),
        c.url = row.url
    """

    CYPHER_MERGE_COMPETITIONS = """
    UNWIND $batch AS row
    MERGE (c:Competition {id: row.id})
    SET c.code = row.code,
        c.name = row.name,
        c.subType = row.subType,
        c.type = row.type,
        c.domesticLeagueCode = row.domesticLeagueCode,
        c.confederation = row.confederation,
        c.totalClubs = toInteger(row.totalClubs),
        c.url = row.url
    WITH c, row
    WHERE row.countryId IS NOT NULL
    MERGE (ct:Country {id: row.countryId})
    MERGE (c)-[:LOCATED_IN]->(ct)
    """

    CYPHER_MERGE_NATIONAL_TEAMS = """
    UNWIND $batch AS row
    MERGE (nt:NationalTeam {id: row.id})
    SET nt.name = row.name,
        nt.teamCode = row.teamCode,
        nt.confederation = row.confederation,
        nt.teamImageUrl = row.teamImageUrl,
        nt.squadSize = toInteger(row.squadSize),
        nt.averageAge = toFloat(row.averageAge),
        nt.foreignersNumber = toInteger(row.foreignersNumber),
        nt.foreignersPercentage = toFloat(row.foreignersPercentage),
        nt.totalMarketValue = toFloat(row.totalMarketValue),
        nt.coachName = row.coachName,
        nt.fifaRanking = toInteger(row.fifaRanking),
        nt.lastSeason = toInteger(row.lastSeason),
        nt.url = row.url
    WITH nt, row
    WHERE row.countryId IS NOT NULL
    MERGE (ct:Country {id: row.countryId})
    MERGE (nt)-[:REPRESENTS_COUNTRY]->(ct)
    """

    def _prepare_countries(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Map countries DataFrame columns to camelCase parameter dicts."""
        records = []
        for row in df.to_dict(orient="records"):
            cid = row.get("id") or row.get("country_id")
            if cid is None or pd.isna(cid):
                continue
            records.append({
                "id": str(cid),
                "name": row.get("name") or row.get("country_name"),
                "code": row.get("code"),
                "confederation": row.get("confederation"),
                "totalClubs": row.get("total_clubs") or row.get("totalClubs"),
                "totalPlayers": row.get("total_players") or row.get("totalPlayers"),
                "averageAge": row.get("average_age") or row.get("averageAge"),
                "url": row.get("url"),
            })
        return records

    def _prepare_competitions(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Map competitions DataFrame columns to camelCase parameter dicts."""
        records = []
        for row in df.to_dict(orient="records"):
            cid = row.get("competition_id") or row.get("id")
            if cid is None or pd.isna(cid):
                continue
            country_id = row.get("country_id") or row.get("country_name")
            records.append({
                "id": str(cid),
                "code": row.get("competition_code") or row.get("code"),
                "name": row.get("name"),
                "subType": row.get("sub_type") or row.get("subType"),
                "type": row.get("type"),
                "domesticLeagueCode": row.get("domestic_league_code") or row.get("domesticLeagueCode"),
                "confederation": row.get("confederation"),
                "totalClubs": row.get("total_clubs") or row.get("totalClubs"),
                "url": row.get("url"),
                "countryId": str(country_id) if country_id and not pd.isna(country_id) else None,
            })
        return records

    def _prepare_national_teams(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Map national teams DataFrame columns to camelCase parameter dicts."""
        records = []
        for row in df.to_dict(orient="records"):
            nt_id = row.get("national_team_id") or row.get("id")
            if nt_id is None or pd.isna(nt_id):
                continue
            country_id = row.get("country_id")
            records.append({
                "id": str(nt_id),
                "name": row.get("name"),
                "teamCode": row.get("team_code") or row.get("teamCode"),
                "confederation": row.get("confederation"),
                "teamImageUrl": row.get("team_image_url") or row.get("teamImageUrl"),
                "squadSize": row.get("squad_size") or row.get("squadSize"),
                "averageAge": row.get("average_age") or row.get("averageAge"),
                "foreignersNumber": row.get("foreigners_number") or row.get("foreignersNumber"),
                "foreignersPercentage": row.get("foreigners_percentage") or row.get("foreignersPercentage"),
                "totalMarketValue": row.get("total_market_value") or row.get("totalMarketValue"),
                "coachName": row.get("coach_name") or row.get("coachName"),
                "fifaRanking": row.get("fifa_ranking") or row.get("fifaRanking"),
                "lastSeason": row.get("last_season") or row.get("lastSeason"),
                "url": row.get("url"),
                "countryId": str(country_id) if country_id and not pd.isna(country_id) else None,
            })
        return records

    def load(self, data_dir: Union[str, Path], dev_mode: bool = False) -> None:
        """
        Load reference domain CSV files into Neo4j.

        Args:
            data_dir: Path to directory containing raw CSV files.
            dev_mode: If True, filter competitions by active DEV_MODE subgraph IDs.
        """
        logger.info("Starting Reference Data Ingestion (dev_mode=%s)", dev_mode)
        
        dev_filters = self.get_dev_filter_ids(data_dir) if dev_mode else None

        # 1. Ingest Countries
        countries_path = os.path.join(data_dir, "countries.csv")
        if os.path.exists(countries_path):
            countries_df = self.read_csv(countries_path)
            country_records = self._prepare_countries(countries_df)
            self.execute_batch(self.CYPHER_MERGE_COUNTRIES, country_records)
        else:
            logger.warning("countries.csv not found at %s. Skipping.", countries_path)

        # 2. Ingest Competitions
        competitions_path = os.path.join(data_dir, "competitions.csv")
        if os.path.exists(competitions_path):
            competitions_df = self.read_csv(competitions_path)
            if dev_mode and dev_filters and "competition_ids" in dev_filters:
                comp_col = "competition_id" if "competition_id" in competitions_df.columns else "id"
                valid_comp_ids = {str(cid).strip() for cid in dev_filters["competition_ids"]}
                competitions_df = competitions_df[
                    competitions_df[comp_col].astype(str).str.strip().isin(valid_comp_ids)
                ]
                logger.info("Filtered competitions.csv to %d rows for DEV_MODE", len(competitions_df))
            comp_records = self._prepare_competitions(competitions_df)
            self.execute_batch(self.CYPHER_MERGE_COMPETITIONS, comp_records)
        else:
            logger.warning("competitions.csv not found at %s. Skipping.", competitions_path)

        # 3. Ingest National Teams
        national_teams_path = os.path.join(data_dir, "national_teams.csv")
        if os.path.exists(national_teams_path):
            nt_df = self.read_csv(national_teams_path)
            nt_records = self._prepare_national_teams(nt_df)
            self.execute_batch(self.CYPHER_MERGE_NATIONAL_TEAMS, nt_records)
        else:
            logger.warning("national_teams.csv not found at %s. Skipping.", national_teams_path)

        logger.info("Completed Reference Data Ingestion.")