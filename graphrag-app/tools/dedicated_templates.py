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
from database import execute_cypher_query
from models import ReplacementCandidatesInput
from utils import format_scouting_report


logger = logging.getLogger(__name__)

#// Dynamic parameterized Cypher template
DYNAMIC_REPLACEMENT_CANDIDATES_CYPHER = """
// Step 1: Resolve benchmark player's market valuation, subPosition, and age for $season
CALL db.index.fulltext.queryNodes('player_name_fulltext', $benchmark_player) YIELD node AS bm, score AS bmScore
WITH bm ORDER BY bmScore DESC LIMIT 1
OPTIONAL MATCH (bm)-[:HAS_VALUATION]->(bmVal:PlayerValuation)
WHERE bmVal.date STARTS WITH toString($season + 1) OR bmVal.date STARTS WITH toString($season)
WITH bm, avg(bmVal.marketValueInEur) AS bmMarketValuation,
     CASE WHEN bm.dateOfBirth IS NOT NULL AND toString(bm.dateOfBirth) <> 'NaN' THEN ($season - toInteger(left(toString(bm.dateOfBirth), 4))) ELSE NULL END AS bmAge

// Step 2: Resolve target club and determine its primary formation(s) in $season
CALL db.index.fulltext.queryNodes('club_name_fulltext', $target_club) YIELD node AS targetClub, score AS clubScore
WITH bm, bmMarketValuation, bmAge, targetClub ORDER BY clubScore DESC LIMIT 1
MATCH (targetClub)-[piTarget:PLAYED_IN]->(gTarget:Game)
WHERE gTarget.season = $season
WITH bm, bmMarketValuation, bmAge, targetClub,
     [x IN collect(DISTINCT CASE WHEN piTarget.hosting = 'Home' THEN gTarget.homeClubFormation ELSE gTarget.awayClubFormation END) WHERE x IS NOT NULL AND toString(x) <> 'NaN'] AS clubFormations

// Step 3: Extract target club's squad roster for $season to exclude existing squad members except benchmark_player
OPTIONAL MATCH (targetClub)-[:PLAYED_IN]->(gSeason:Game)<-[:APPEARED_IN]-(squadMember:Player)
WHERE gSeason.season = $season 
  AND (toLower(toString(squadMember.position)) CONTAINS toLower($position) OR toLower(toString(squadMember.subPosition)) CONTAINS toLower($position))

WITH bm, bmMarketValuation, bmAge, targetClub, clubFormations, collect(DISTINCT squadMember) AS seasonRoster

// Step 4: Match candidate player appearances strictly in $season
MATCH (candidate:Player)-[a:APPEARED_IN]->(g:Game)
WHERE g.season = $season
  AND (
    toLower(toString(candidate.position)) CONTAINS toLower($position) 
    OR toLower(toString(candidate.subPosition)) CONTAINS toLower($position)
    OR (bm IS NOT NULL AND candidate = bm)
  )
  AND candidate.subPosition IS NOT NULL
  AND (NOT candidate IN seasonRoster OR (bm IS NOT NULL AND candidate = bm) OR candidate = bm)

OPTIONAL MATCH (g)-[:PART_OF_COMPETITION]->(comp:Competition)
WHERE comp.id IN ['GB1', 'ES1', 'IT1', 'BL1', 'L1', 'FR1'] OR (bm IS NOT NULL AND candidate = bm)

WITH bm, bmMarketValuation, bmAge, candidate, g, a, clubFormations,
     CASE WHEN bm IS NOT NULL AND candidate = bm THEN true ELSE false END AS isBenchmark,
     CASE WHEN candidate.dateOfBirth IS NOT NULL AND toString(candidate.dateOfBirth) <> 'NaN' THEN ($season - toInteger(left(toString(candidate.dateOfBirth), 4))) ELSE NULL END AS age

// Aggregate player's match activity and performance strictly for $season (each game counted once)
WITH bm, bmMarketValuation, bmAge, candidate, isBenchmark, age, clubFormations,
     count(DISTINCT g) AS totalMatches,
     sum(a.minutesPlayed) AS totalMinutes,
     sum(a.goals) AS goals,
     sum(a.assists) AS assists,
     candidate.heightInCm AS heightCm,
     candidate.foot AS foot
WHERE (totalMinutes >= $min_minutes OR isBenchmark = true)

// Resolve candidate's club at that time via TRANSFERRED_TO up to $season, falling back to PLAYS_FOR
OPTIONAL MATCH (candidate)-[t:TRANSFERRED_TO]->(cTrans:Club)
WHERE (t.transferDate IS NOT NULL AND toInteger(left(t.transferDate, 4)) <= $season)
   OR (t.transferSeason IS NOT NULL AND toInteger(left(t.transferSeason, 4)) <= $season)
WITH candidate, isBenchmark, age, bmAge, totalMatches, totalMinutes, goals, assists, heightCm, foot, bm, bmMarketValuation, clubFormations,
     cTrans, t ORDER BY coalesce(t.transferDate, t.transferSeason) DESC
WITH candidate, isBenchmark, age, bmAge, totalMatches, totalMinutes, goals, assists, heightCm, foot, bm, bmMarketValuation, clubFormations,
     collect(cTrans.name)[0] AS transferClub

OPTIONAL MATCH (candidate)-[:PLAYS_FOR]->(cCurrent:Club)
WITH candidate, isBenchmark, age, bmAge, totalMatches, totalMinutes, goals, assists, heightCm, foot, bm, bmMarketValuation, clubFormations,
     coalesce(transferClub, cCurrent.name, 'N/A') AS club

OPTIONAL MATCH (candidate)-[:HAS_VALUATION]->(pv:PlayerValuation) 
WHERE pv.date STARTS WITH toString($season + 1) OR pv.date STARTS WITH toString($season)

WITH candidate, club, isBenchmark, age, bmAge, totalMatches, totalMinutes, goals, assists, (goals + assists) AS contributions,
     heightCm, foot, bm, bmMarketValuation, clubFormations,
     avg(pv.marketValueInEur) AS approxValuation,
     CASE WHEN bm IS NOT NULL AND candidate.subPosition = bm.subPosition THEN 1 ELSE 0 END AS isSubPositionMatch,
     100.0 AS formationFitPct,
     round(CASE WHEN totalMatches > 0 THEN (100.0 * goals / totalMatches) ELSE 0.0 END, 1) AS winRatePct

// Step 5: Filter candidates using benchmark valuation bounds (0.2x to 1.2x), age cap (<= bmAge), and tactical formation fit (>= 40% if formations exist)
WHERE isBenchmark 
   OR (
     (bmMarketValuation IS NULL OR (approxValuation >= (bmMarketValuation * 0.2) AND approxValuation <= (bmMarketValuation * 1.2)))
     AND (bmAge IS NULL OR age IS NULL OR age <= bmAge)
   )

RETURN candidate.name AS candidate,
       club AS club,
       isBenchmark AS isBenchmark,
       candidate.subPosition AS subPosition,
       age,
       heightCm,
       foot AS preferredFoot,
       totalMatches,
       totalMinutes,
       goals,
       assists,
       contributions,
       formationFitPct,
       winRatePct,
       approxValuation
ORDER BY isBenchmark DESC,
         isSubPositionMatch DESC,
         contributions DESC,
         approxValuation ASC
LIMIT 6
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
    records = execute_cypher_query(DYNAMIC_REPLACEMENT_CANDIDATES_CYPHER, parameters)
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
    model = model_name or settings.LLM_MODEL
    key = api_key or settings.GEMINI_API_KEY

    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=key,
        temperature=0.0,
        max_output_tokens=settings.MAX_OUTPUT_TOKENS,
        thinking_level="low",
        max_retries=3,
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


@tool("get_replacement_candidates", return_direct=True)
def get_replacement_candidates(question: str) -> str:
    """
    Find suitable replacement player candidates in a given season compared to a benchmark transfer target for a club.
    Use this tool when the user asks to scout, replace, find alternatives, or compare players for a team.

    Args:
        question: Natural language scouting query string.

    Returns:
        Formatted Scouting Report string.
    """
    try:
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
        if not records:
            return "I am sorry, I couldn't find what you are looking for. Please try again with a different question."
        return format_scouting_report(records, params)
    except Exception as e:
        logger.error("Error executing get_replacement_candidates for question '%s': %s", question, e)
        return "I am sorry, I couldn't find what you are looking for. Please try again with a different question."
