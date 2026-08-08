"""
Dedicated pre-written Cypher template tools for GraphRAG agent.

Implements get_replacement_candidates tool using LangChain @tool decorator
and ReplacementCandidatesInput Pydantic model with dynamic target club formation evaluation,
benchmark valuation lower/upper bounds, and exact subPosition sorting.
"""

import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from config import settings
from database import get_neo4j_driver
from models import ReplacementCandidatesInput
from utils import format_scouting_report

logger = logging.getLogger(__name__)

# Dynamic parameterized Cypher template
DYNAMIC_REPLACEMENT_CANDIDATES_CYPHER = """
// Step 1: Dynamically determine benchmark player's market valuation and subPosition in the season
OPTIONAL MATCH (bm:Player)
WHERE toLower(bm.name) CONTAINS toLower($benchmark_player)
OPTIONAL MATCH (bm)-[:HAS_VALUATION]->(bmVal:PlayerValuation)
WHERE bmVal.date STARTS WITH toString($season + 1) OR bmVal.date STARTS WITH toString($season)
WITH bm, avg(bmVal.marketValueInEur) AS bmMarketValuation

// Step 2: Dynamically determine target club's primary formation(s) in the given season from Game nodes
MATCH (targetClub:Club)-[piTarget:PLAYED_IN]->(gTarget:Game)
WHERE toLower(targetClub.name) CONTAINS toLower($target_club)
  AND gTarget.season = $season
WITH bm, bmMarketValuation, targetClub, 
     collect(DISTINCT CASE WHEN piTarget.hosting = 'Home' THEN gTarget.homeClubFormation ELSE gTarget.awayClubFormation END) AS clubFormations

// Step 3: Extract target club's squad roster for that season to exclude existing squad members except benchmark_player
MATCH (targetClub)-[:PLAYED_IN]->(gSeason:Game)<-[:APPEARED_IN]-(squadMember:Player)
WHERE gSeason.season = $season 
  AND squadMember.position = $position
WITH bm, bmMarketValuation, targetClub, clubFormations, collect(DISTINCT squadMember) AS seasonRoster

// Step 4: Match candidate players in Top 5 European Leagues for the given season
MATCH (candidate:Player)-[a:APPEARED_IN]->(g:Game)<-[pi:PLAYED_IN]-(c:Club)
MATCH (g)-[:PART_OF_COMPETITION]->(comp:Competition)
WHERE candidate.position = $position
  AND candidate.subPosition IS NOT NULL
  AND candidate.name IS NOT NULL
  AND (NOT candidate IN seasonRoster OR toLower(candidate.name) CONTAINS toLower($benchmark_player))
  AND g.season = $season
  AND comp.id IN ['GB1', 'ES1', 'IT1', 'BL1', 'L1', 'FR1']

WITH bm, bmMarketValuation, candidate, c, g, pi, a, clubFormations,
     CASE WHEN pi.hosting = 'Home' THEN g.homeClubFormation ELSE g.awayClubFormation END AS matchFormation,
     CASE WHEN toLower(candidate.name) CONTAINS toLower($benchmark_player) THEN true ELSE false END AS isBenchmark

WITH bm, bmMarketValuation, candidate, isBenchmark, clubFormations,
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
WHERE pv.date STARTS WITH toString($season + 1) OR pv.date STARTS WITH toString($season)

WITH candidate, isBenchmark, totalMatches, totalMinutes, goals, assists, formationMatches, winMatches, heightCm, foot,
     avg(pv.marketValueInEur) AS approxValuation,
     bmMarketValuation,
     CASE WHEN bm IS NOT NULL AND candidate.subPosition = bm.subPosition THEN 1 ELSE 0 END AS isSubPositionMatch

// Step 5: Filter candidates using benchmark valuation bounds (0.2x to 1.2x benchmark valuation)
WHERE isBenchmark 
   OR bmMarketValuation IS NULL 
   OR (approxValuation >= (bmMarketValuation * 0.2) AND approxValuation <= (bmMarketValuation * 1.2))

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
ORDER BY isBenchmark DESC, isSubPositionMatch DESC, winRatePct DESC, contributions DESC
LIMIT 5
"""


def execute_replacement_candidates_query(parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Execute dedicated replacement candidate Cypher query against Neo4j instance.

    Args:
        parameters: Parameter dict containing target_club, season, position, benchmark_player, min_minutes.

    Returns:
        List of dicts containing candidate player records.
    """
    logger.info("Executing dedicated get_replacement_candidates Cypher with params: %s", parameters)

    driver = get_neo4j_driver()
    with driver.session(database=settings.NEO4J_DATABASE) as session:
        result = session.run(DYNAMIC_REPLACEMENT_CANDIDATES_CYPHER, parameters)
        records = [record.data() for record in result]
        logger.info("get_replacement_candidates Cypher query returned %d record(s).", len(records))
        return records



def extract_scouting_parameters(
    question: str,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> ReplacementCandidatesInput:
    """
    Use Gemini structured output LLM call to extract scouting parameters from natural language prompt.

    Args:
        question: User natural language scouting question.
        model_name: Optional model override.
        api_key: Optional API key override.

    Returns:
        Structured ReplacementCandidatesInput instance.
    """
    model = model_name or settings.GRAPH_LLM_MODEL
    key = api_key or settings.GEMINI_API_KEY

    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=key,
        temperature=0.0,
        max_output_tokens=settings.MAX_OUTPUT_TOKENS,
        thinking_level="low",
    )

    extractor = llm.with_structured_output(ReplacementCandidatesInput)

    system_instruction = (
        "Extract target_club (str), season (int), position group ('Attack', 'Midfield', 'Defender', 'Goalkeeper'), "
        "benchmark_player name (str), and min_minutes (default 1000) from the user question.\n"
        "CRITICAL SEASON RULE: Always extract the PREVIOUS season starting year (target_season_start - 1) for scouting!\n"
        "- If the query asks for 2014/15 season (or 2014), season MUST be extracted as 2013.\n"
        "- If the query asks for 2013/14 season (or 2013), season MUST be extracted as 2012."
    )

    logger.info("Extracting scouting parameters via LLM call for question: '%s'", question)
    scout_params: ReplacementCandidatesInput = extractor.invoke([
        SystemMessage(content=system_instruction),
        HumanMessage(content=question),
    ])
    logger.info("Extracted scouting parameters: %s", scout_params.model_dump())
    return scout_params


@tool("get_replacement_candidates")
def get_replacement_candidates(question: str) -> str:
    """
    Find suitable replacement player candidates in a given season compared to a benchmark transfer target for a club.
    Use this tool when the user asks to scout, replace, find alternatives, or compare players for a team.

    Args:
        question: Natural language scouting query string.

    Returns:
        Formatted Scouting Report string.
    """
    scout_params = extract_scouting_parameters(question)
    params = scout_params.model_dump()

    # Season rule safeguard check
    season = params.get("season", 2013)
    if isinstance(season, int) and season >= 2000:
        for yr in range(2000, 2030):
            if str(yr) in question and season == yr:
                params["season"] = yr - 1
                break

    records = execute_replacement_candidates_query(params)
    return format_scouting_report(records, params)
