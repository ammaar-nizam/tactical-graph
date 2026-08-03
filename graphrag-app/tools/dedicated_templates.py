"""
Dedicated pre-written Cypher template tools for GraphRAG agent.

Implements get_replacement_candidates tool using LangChain @tool decorator
and ReplacementCandidatesInput Pydantic model with dynamic target club formation evaluation.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.tools import tool
from neo4j import GraphDatabase

# Adjust path to import config and models
_APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_APP_DIR))

from config import settings
from models import ReplacementCandidatesInput

logger = logging.getLogger(__name__)

# Dynamic parameterized Cypher template for scouting replacement candidates
DYNAMIC_REPLACEMENT_CANDIDATES_CYPHER = """
// Step 1: Dynamically determine target club's primary formation(s) in the given season from Game nodes
MATCH (targetClub:Club)-[piTarget:PLAYED_IN]->(gTarget:Game)
WHERE toLower(targetClub.name) CONTAINS toLower($target_club)
  AND gTarget.season = $season
WITH targetClub, 
     collect(DISTINCT CASE WHEN piTarget.hosting = 'Home' THEN gTarget.homeClubFormation ELSE gTarget.awayClubFormation END) AS clubFormations

// Step 2: Extract target club's squad roster for that season to exclude existing squad members except benchmark_player
MATCH (targetClub)-[:PLAYED_IN]->(gSeason:Game)<-[:APPEARED_IN]-(squadMember:Player)
WHERE gSeason.season = $season 
  AND squadMember.position = $position
WITH targetClub, clubFormations, collect(DISTINCT squadMember) AS seasonRoster

// Step 3: Match candidate players in Top 5 European Leagues for the given season
MATCH (candidate:Player)-[a:APPEARED_IN]->(g:Game)<-[pi:PLAYED_IN]-(c:Club)
MATCH (g)-[:PART_OF_COMPETITION]->(comp:Competition)
WHERE candidate.position = $position
  AND candidate.subPosition IS NOT NULL
  AND candidate.name IS NOT NULL
  AND (NOT candidate IN seasonRoster OR toLower(candidate.name) CONTAINS toLower($benchmark_player))
  AND g.season = $season
  AND comp.id IN ['GB1', 'ES1', 'IT1', 'BL1', 'L1', 'FR1']

WITH candidate, c, g, pi, a, clubFormations,
     CASE WHEN pi.hosting = 'Home' THEN g.homeClubFormation ELSE g.awayClubFormation END AS matchFormation,
     CASE WHEN toLower(candidate.name) CONTAINS toLower($benchmark_player) THEN true ELSE false END AS isBenchmark

WITH candidate, isBenchmark, clubFormations,
     count(g) AS totalMatches,
     sum(a.minutesPlayed) AS totalMinutes,
     sum(a.goals) AS goals,
     sum(a.assists) AS assists,
     sum(CASE WHEN matchFormation IN clubFormations THEN 1 ELSE 0 END) AS formationMatches,
     sum(CASE WHEN pi.isWin = 1 THEN 1 ELSE 0 END) AS winMatches,
     candidate.heightInCm AS heightCm,
     candidate.foot AS foot
WHERE totalMinutes >= $min_minutes

OPTIONAL MATCH (candidate)-[:HAS_VALUATION]->(pv:PlayerValuation) 
WHERE pv.date STARTS WITH toString($season + 1)

WITH candidate, isBenchmark, totalMatches, totalMinutes, goals, assists, formationMatches, winMatches, heightCm, foot,
     avg(pv.marketValueInEur) AS approxValuation

RETURN candidate.name AS candidate,
       isBenchmark AS isBenchmark,
       candidate.subPosition AS subPosition,
       heightCm,
       foot AS preferredFoot,
       totalMatches,
       totalMinutes,
       goals,
       assists,
       (goals + assists) AS contributions,
       round(CASE WHEN totalMatches > 0 THEN (100.0 * formationMatches / totalMatches) ELSE 0.0 END, 1) AS formationFitPct,
       round(CASE WHEN totalMatches > 0 THEN (100.0 * winMatches / totalMatches) ELSE 0.0 END, 1) AS winRatePct,
       approxValuation
ORDER BY isBenchmark DESC, contributions DESC, winRatePct DESC
LIMIT 12
"""


@tool("get_replacement_candidates", args_schema=ReplacementCandidatesInput)
def get_replacement_candidates(
    target_club: str,
    season: int,
    position: str,
    benchmark_player: str,
    min_minutes: int = 1000,
) -> List[Dict[str, Any]]:
    """
    Find suitable replacement player candidates in a given season compared to a benchmark transfer target.

    Evaluates candidates based on positional match, dynamic target club formation compatibility,
    win rate percentage, goal contributions, and market valuation in Top 5 European leagues.

    Args:
        target_club: Name of the target club (e.g. 'Manchester United').
        season: The season starting year (e.g. 2012 for 2012/13).
        position: Player position group (e.g. 'Midfield', 'Attack', 'Defender').
        benchmark_player: Name of benchmark player to include for comparison (e.g. 'Fellaini').
        min_minutes: Minimum minutes played in season (default 1000).

    Returns:
        List of dicts containing candidate player statistics, formation fit %, and valuation.
    """
    uri = settings.NEO4J_URI.replace("neo4j+s://", "neo4j+ssc://")
    auth = (settings.NEO4J_USER, settings.NEO4J_PASSWORD)

    parameters = {
        "target_club": target_club,
        "season": season,
        "position": position,
        "benchmark_player": benchmark_player,
        "min_minutes": min_minutes,
    }

    logger.info("Executing dedicated get_replacement_candidates with params: %s", parameters)

    driver = GraphDatabase.driver(uri, auth=auth)
    try:
        with driver.session(database=settings.NEO4J_DATABASE) as session:
            result = session.run(DYNAMIC_REPLACEMENT_CANDIDATES_CYPHER, parameters)
            records = [record.data() for record in result]
            logger.info("get_replacement_candidates returned %d record(s).", len(records))
            return records
    except Exception as e:
        logger.error("Failed to execute get_replacement_candidates query: %s", e)
        raise
    finally:
        driver.close()
