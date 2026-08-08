"""
Few-shot query bank module for Text-to-Cypher prompt engineering.

Provides a curated, token-efficient dictionary of Natural Language -> Cypher example pairs
matching the graph schema without reading external files.
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# Curated, token-efficient few-shot examples (strictly schema-compliant, max 6 items)
FEW_SHOT_EXAMPLES: List[Dict[str, str]] = [
    {
        "question": "Find player profile for Kevin De Bruyne",
        "cypher": (
            "MATCH (p:Player) WHERE p.name IS NOT NULL AND toLower(p.name) CONTAINS 'de bruyne' "
            "OPTIONAL MATCH (p)-[:PLAYS_FOR]->(c:Club) "
            "OPTIONAL MATCH (p)-[:HAS_VALUATION]->(pv:PlayerValuation) "
            "WITH p, c, pv ORDER BY pv.date DESC "
            "WITH p, c, collect(pv)[0] AS latestVal "
            "RETURN p.name AS player, p.position AS position, p.dateOfBirth AS dob, "
            "p.countryOfCitizenship AS nationality, c.name AS club, latestVal.marketValueInEur AS currentValue LIMIT 5"
        ),
    },
    {
        "question": "Which players currently play for Manchester City?",
        "cypher": (
            "MATCH (p:Player)-[:PLAYS_FOR]->(c:Club) "
            "WHERE c.name IS NOT NULL AND toLower(c.name) CONTAINS 'manchester city' AND p.name IS NOT NULL "
            "RETURN p.name AS player, p.position AS position, p.dateOfBirth AS dob ORDER BY p.position, p.name LIMIT 5"
        ),
    },
    {
        "question": "List the top 5 most expensive transfers in history",
        "cypher": (
            "MATCH (p:Player)-[t:TRANSFERRED_TO]->(c:Club) "
            "WHERE t.transferFee IS NOT NULL AND t.transferFee > 0 AND p.name IS NOT NULL "
            "RETURN p.name AS player, c.name AS toClub, t.fromClubName AS fromClub, t.transferFee AS fee, t.transferDate AS date "
            "ORDER BY t.transferFee DESC LIMIT 5"
        ),
    },
    {
        "question": "What is the transfer history of Philippe Coutinho?",
        "cypher": (
            "MATCH (p:Player)-[t:TRANSFERRED_TO]->(c:Club) "
            "WHERE p.name IS NOT NULL AND toLower(p.name) CONTAINS 'coutinho' "
            "RETURN p.name AS player, t.fromClubName AS fromClub, c.name AS toClub, t.transferFee AS fee, t.transferDate AS date "
            "ORDER BY t.transferDate LIMIT 5"
        ),
    },
    {
        "question": "Who are the top goal scorers in appearance statistics?",
        "cypher": (
            "MATCH (p:Player)-[a:APPEARED_IN]->(g:Game) "
            "WHERE a.goals IS NOT NULL AND a.goals > 0 AND p.name IS NOT NULL "
            "RETURN p.name AS player, sum(a.goals) AS totalGoals, count(a) AS matchesWithGoals "
            "ORDER BY totalGoals DESC LIMIT 5"
        ),
    },
]


def get_few_shot_prompt() -> str:
    """
    Format curated few-shot examples into a concise text block for prompt injection.

    Returns:
        Formatted string containing Question -> Cypher pairs.
    """
    lines = ["Here are curated example pairs of User Questions and corresponding Cypher queries:"]

    for idx, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        lines.append(f"\nExample {idx}:")
        lines.append(f"Question: {ex['question']}")
        lines.append(f"Cypher:\n{ex['cypher']}")

    return "\n".join(lines)
