"""
Few-shot query bank module for Text-to-Cypher prompt engineering.

Reads and reference data-pipeline/cyphers.json (consolidating non-analytical query categories)
and schema.md to provide curated Natural Language -> Cypher example pairs matching the graph schema.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

# File paths to data-pipeline assets
_APP_DIR = Path(__file__).resolve().parent
_DATA_PIPELINE_DIR = _APP_DIR.parent / "data-pipeline"
_CYPHERS_JSON_PATH = _DATA_PIPELINE_DIR / "cyphers.json"

# Fallback curated few-shot examples (strictly schema-compliant)
DEFAULT_FEW_SHOT_EXAMPLES: List[Dict[str, str]] = [
    {
        "question": "Find player profile for Kevin De Bruyne",
        "cypher": (
            "MATCH (p:Player) WHERE p.name IS NOT NULL AND toLower(p.name) CONTAINS 'de bruyne' "
            "OPTIONAL MATCH (p)-[:PLAYS_FOR]->(c:Club) "
            "OPTIONAL MATCH (p)-[:HAS_VALUATION]->(pv:PlayerValuation) "
            "WITH p, c, pv ORDER BY pv.date DESC "
            "WITH p, c, collect(pv)[0] AS latestVal "
            "RETURN p.name AS player, p.position AS position, p.dateOfBirth AS dob, "
            "p.countryOfCitizenship AS nationality, c.name AS club, latestVal.marketValueInEur AS currentValue"
        ),
    },
    {
        "question": "Which players currently play for Manchester City?",
        "cypher": (
            "MATCH (p:Player)-[:PLAYS_FOR]->(c:Club) "
            "WHERE c.name IS NOT NULL AND toLower(c.name) CONTAINS 'manchester city' AND p.name IS NOT NULL "
            "RETURN p.name AS player, p.position AS position, p.dateOfBirth AS dob ORDER BY p.position, p.name"
        ),
    },
    {
        "question": "List the top 10 most expensive transfers in history",
        "cypher": (
            "MATCH (p:Player)-[t:TRANSFERRED_TO]->(c:Club) "
            "WHERE t.transferFee IS NOT NULL AND t.transferFee > 0 AND p.name IS NOT NULL "
            "RETURN p.name AS player, c.name AS toClub, t.fromClubName AS fromClub, t.transferFee AS fee, t.transferDate AS date "
            "ORDER BY t.transferFee DESC LIMIT 10"
        ),
    },
    {
        "question": "What is the transfer history of Philippe Coutinho?",
        "cypher": (
            "MATCH (p:Player)-[t:TRANSFERRED_TO]->(c:Club) "
            "WHERE p.name IS NOT NULL AND toLower(p.name) CONTAINS 'coutinho' "
            "RETURN p.name AS player, t.fromClubName AS fromClub, c.name AS toClub, t.transferFee AS fee, t.transferDate AS date "
            "ORDER BY t.transferDate"
        ),
    },
    {
        "question": "Who are the top goal scorers in appearance statistics?",
        "cypher": (
            "MATCH (p:Player)-[a:APPEARED_IN]->(g:Game) "
            "WHERE a.goals IS NOT NULL AND a.goals > 0 AND p.name IS NOT NULL "
            "RETURN p.name AS player, sum(a.goals) AS totalGoals, count(a) AS matchesWithGoals "
            "ORDER BY totalGoals DESC LIMIT 10"
        ),
    },
    {
        "question": "List top goal assist providers",
        "cypher": (
            "MATCH (p:Player)-[a:APPEARED_IN]->(g:Game) "
            "WHERE a.assists IS NOT NULL AND a.assists > 0 AND p.name IS NOT NULL "
            "RETURN p.name AS player, sum(a.assists) AS totalAssists, count(a) AS matchesWithAssists "
            "ORDER BY totalAssists DESC LIMIT 10"
        ),
    },
    {
        "question": "What are the head-to-head match results between Barcelona and Real Madrid?",
        "cypher": (
            "MATCH (c1:Club)-[p1:PLAYED_IN]->(g:Game)<-[p2:PLAYED_IN]-(c2:Club) "
            "WHERE c1.name IS NOT NULL AND c2.name IS NOT NULL "
            "  AND toLower(c1.name) CONTAINS 'barcelona' AND toLower(c2.name) CONTAINS 'real madrid' "
            "RETURN g.date AS date, c1.name AS club1, p1.hosting AS club1Hosting, g.homeClubGoals AS homeGoals, g.awayClubGoals AS awayGoals, c2.name AS club2 "
            "ORDER BY g.date DESC LIMIT 10"
        ),
    },
    {
        "question": "Find players who captained Arsenal in matches",
        "cypher": (
            "MATCH (p:Player)-[a:APPEARED_IN]->(g:Game)<-[:PLAYED_IN]-(c:Club) "
            "WHERE a.teamCaptain = true AND c.name IS NOT NULL AND toLower(c.name) CONTAINS 'arsenal' "
            "RETURN DISTINCT p.name AS captain, count(a) AS matchesCaptained "
            "ORDER BY matchesCaptained DESC"
        ),
    },
    {
        "question": "List goal events for a specific match on 2024-03-10 between Liverpool and an opponent",
        "cypher": (
            "MATCH (home:Club)-[:PLAYED_IN {hosting: 'Home'}]->(g:Game)<-[:PLAYED_IN {hosting: 'Away'}]-(away:Club) "
            "WHERE g.date = '2024-03-10' AND home.name IS NOT NULL AND toLower(home.name) CONTAINS 'liverpool' "
            "WITH g, home, away "
            "MATCH (scorer:Player)-[:SCORED]->(ge:GameEvent)-[:OCCURRED_IN]->(g) "
            "OPTIONAL MATCH (assister:Player)-[:ASSISTED]->(ge) "
            "RETURN ge.minute AS minute, scorer.name AS scorer, assister.name AS assister, ge.description AS description "
            "ORDER BY ge.minute"
        ),
    },
]


def load_few_shot_bank() -> List[Dict[str, str]]:
    """
    Load natural language -> Cypher query examples from data-pipeline/cyphers.json,
    consolidating standard query categories while strictly excluding 'graphrag_analytical_queries'.

    Returns:
        List of dicts with keys 'question' and 'cypher'.
    """
    if not _CYPHERS_JSON_PATH.exists():
        logger.warning("cyphers.json not found at %s. Using default few-shot bank.", _CYPHERS_JSON_PATH)
        return DEFAULT_FEW_SHOT_EXAMPLES

    try:
        with open(_CYPHERS_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        queries_dict = data.get("queries", {})
        few_shots: List[Dict[str, str]] = []

        for category, q_list in queries_dict.items():
            # STRICTLY EXCLUDE graphrag_analytical_queries per prompt requirements
            if category == "graphrag_analytical_queries":
                continue

            for item in q_list:
                question = item.get("name") or item.get("description")
                cypher = item.get("cypher")
                if question and cypher:
                    few_shots.append({
                        "question": question,
                        "cypher": cypher,
                    })

        logger.info("Loaded %d few-shot query examples from %s", len(few_shots), _CYPHERS_JSON_PATH)
        return few_shots if few_shots else DEFAULT_FEW_SHOT_EXAMPLES

    except Exception as e:
        logger.error("Failed to parse cyphers.json: %s. Falling back to defaults.", e)
        return DEFAULT_FEW_SHOT_EXAMPLES


def get_few_shot_prompt() -> str:
    """
    Format few-shot examples into a concise text block for system prompt injection.

    Returns:
        Formatted string containing Question -> Cypher pairs.
    """
    examples = load_few_shot_bank()
    lines = ["Here are curated example pairs of User Questions and corresponding Cypher queries:"]

    # Select representative subset of up to 10 examples for prompt conciseness
    for idx, ex in enumerate(examples[:10], 1):
        lines.append(f"\nExample {idx}:")
        lines.append(f"Question: {ex['question']}")
        lines.append(f"Cypher:\n{ex['cypher']}")

    return "\n".join(lines)
