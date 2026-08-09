"""
Few-shot query bank module for Text-to-Cypher prompt engineering.

Provides a curated, token-efficient dictionary of Natural Language -> Cypher example pairs
matching the graph schema.
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# Curated, token-efficient few-shot examples
FEW_SHOT_EXAMPLES: List[Dict[str, str]] = [
    {
        # Pattern: fulltext player lookup + OPTIONAL MATCH club + collect latest valuation
        "question": "Find player profile for Kevin De Bruyne including his current club and market value",
        "cypher": (
            "CALL db.index.fulltext.queryNodes('player_name_fulltext', 'de bruyne') YIELD node AS p, score "
            "WITH p ORDER BY score DESC LIMIT 1 "
            "OPTIONAL MATCH (p)-[:PLAYS_FOR]->(c:Club) "
            "OPTIONAL MATCH (p)-[:HAS_VALUATION]->(pv:PlayerValuation) "
            "WITH p, c, pv ORDER BY pv.date DESC "
            "WITH p, c, collect(pv)[0] AS latestVal "
            "RETURN p.name AS player, p.position AS position, p.dateOfBirth AS dob, "
            "p.countryOfCitizenship AS nationality, c.name AS club, latestVal.marketValueInEur AS currentValue"
        ),
    },
    {
        # Pattern: fulltext club lookup + player squad members
        "question": "Which players currently play for Manchester City?",
        "cypher": (
            "CALL db.index.fulltext.queryNodes('club_name_fulltext', 'manchester city') YIELD node AS c, score "
            "WITH c ORDER BY score DESC LIMIT 1 "
            "MATCH (p:Player)-[:PLAYS_FOR]->(c) "
            "WHERE p.name IS NOT NULL "
            "RETURN p.name AS player, p.position AS position, p.dateOfBirth AS dob "
            "ORDER BY p.position, p.name LIMIT 5"
        ),
    },
    {
        # Pattern: fulltext club lookup + OPTIONAL latest valuation per player
        "question": "Full squad roster for Chelsea with each player's latest market value",
        "cypher": (
            "CALL db.index.fulltext.queryNodes('club_name_fulltext', 'chelsea') YIELD node AS c, score "
            "WITH c ORDER BY score DESC LIMIT 1 "
            "MATCH (p:Player)-[:PLAYS_FOR]->(c) "
            "WHERE p.name IS NOT NULL "
            "OPTIONAL MATCH (p)-[:HAS_VALUATION]->(pv:PlayerValuation) "
            "WITH p, c, pv ORDER BY pv.date DESC "
            "WITH p, c, collect(pv)[0] AS latestVal "
            "RETURN p.name AS player, p.position AS position, latestVal.marketValueInEur AS marketValue "
            "ORDER BY latestVal.marketValueInEur DESC LIMIT 5"
        ),
    },
    {
        # Pattern: fulltext player lookup + transfer edges ordered by fee
        "question": "What is the full transfer history of Philippe Coutinho?",
        "cypher": (
            "CALL db.index.fulltext.queryNodes('player_name_fulltext', 'coutinho') YIELD node AS p, score "
            "WITH p ORDER BY score DESC LIMIT 1 "
            "MATCH (p)-[t:TRANSFERRED_TO]->(c:Club) "
            "RETURN p.name AS player, t.fromClubName AS fromClub, c.name AS toClub, "
            "t.transferFee AS fee, t.transferDate AS date ORDER BY t.transferDate LIMIT 5"
        ),
    },
    {
        # Pattern: fulltext player lookup + season filtering on integer g.season
        "question": "How many goals did Lionel Messi score in the 2014/15 season?",
        "cypher": (
            "CALL db.index.fulltext.queryNodes('player_name_fulltext', 'lionel messi') YIELD node AS p, score "
            "WITH p ORDER BY score DESC LIMIT 1 "
            "MATCH (p)-[a:APPEARED_IN]->(g:Game) "
            "WHERE g.season = 2014 AND a.goals IS NOT NULL "
            "RETURN p.name AS player, g.season AS season, sum(a.goals) AS totalGoals"
        ),
    },
    {
        # Pattern: fulltext player lookup + per-season appearance stats per competition
        "question": "Show Mohamed Salah's 2024/25 Premier League match-by-match stats",
        "cypher": (
            "CALL db.index.fulltext.queryNodes('player_name_fulltext', 'salah') YIELD node AS p, score "
            "WITH p ORDER BY score DESC LIMIT 1 "
            "MATCH (p)-[a:APPEARED_IN]->(g:Game)-[:PART_OF_COMPETITION]->(comp:Competition) "
            "WHERE comp.id = 'GB1' AND g.season = 2024 "
            "RETURN g.date AS date, a.minutesPlayed AS minutes, a.goals AS goals, "
            "a.assists AS assists, a.yellowCards AS yellows, a.position AS position ORDER BY g.date LIMIT 5"
        ),
    },
    {
        # Pattern: dual fulltext club lookup + co-occurrence match traversal (head-to-head)
        "question": "Head-to-head match record between Barcelona and Real Madrid",
        "cypher": (
            "CALL db.index.fulltext.queryNodes('club_name_fulltext', 'barcelona') YIELD node AS c1, score AS s1 "
            "WITH c1 ORDER BY s1 DESC LIMIT 1 "
            "CALL db.index.fulltext.queryNodes('club_name_fulltext', 'real madrid') YIELD node AS c2, score AS s2 "
            "WITH c1, c2 ORDER BY s2 DESC LIMIT 1 "
            "MATCH (c1)-[p1:PLAYED_IN]->(g:Game)<-[p2:PLAYED_IN]-(c2) "
            "RETURN g.date AS date, c1.name AS club1, p1.hosting AS club1Hosting, "
            "g.homeClubGoals AS homeGoals, g.awayClubGoals AS awayGoals, c2.name AS club2 "
            "ORDER BY g.date DESC LIMIT 5"
        ),
    },
    {
        # Pattern: fulltext club lookup + win/draw/loss aggregation using CASE on isWin (int)
        "question": "What is Liverpool's win, draw, and loss record across all matches?",
        "cypher": (
            "CALL db.index.fulltext.queryNodes('club_name_fulltext', 'liverpool') YIELD node AS club, score "
            "WITH club ORDER BY score DESC LIMIT 1 "
            "MATCH (club)-[pi:PLAYED_IN]->(g:Game) "
            "RETURN club.name AS club, "
            "count(CASE WHEN pi.isWin = 1 THEN 1 END) AS wins, "
            "count(CASE WHEN pi.ownGoals = pi.opponentGoals THEN 1 END) AS draws, "
            "count(CASE WHEN pi.isWin = 0 AND pi.ownGoals <> pi.opponentGoals THEN 1 END) AS losses, "
            "count(pi) AS totalMatches"
        ),
    },
    {
        # Pattern: fulltext player lookup + co-appearance game traversal (teammates)
        "question": "Find all current teammates of Jude Bellingham at his club",
        "cypher": (
            "CALL db.index.fulltext.queryNodes('player_name_fulltext', 'bellingham') YIELD node AS p, score "
            "WITH p ORDER BY score DESC LIMIT 1 "
            "MATCH (p)-[:PLAYS_FOR]->(c:Club)<-[:PLAYS_FOR]-(teammate:Player) "
            "WHERE p <> teammate AND teammate.name IS NOT NULL "
            "RETURN p.name AS player, c.name AS club, teammate.name AS teammate, teammate.position AS position "
            "ORDER BY teammate.position, teammate.name LIMIT 5"
        ),
    },
    {
        # Pattern: fulltext player + fulltext player co-appearance in same game
        "question": "Have Messi and Neymar ever played in the same match?",
        "cypher": (
            "CALL db.index.fulltext.queryNodes('player_name_fulltext', 'messi') YIELD node AS p1, score AS s1 "
            "WITH p1 ORDER BY s1 DESC LIMIT 1 "
            "CALL db.index.fulltext.queryNodes('player_name_fulltext', 'neymar') YIELD node AS p2, score AS s2 "
            "WITH p1, p2 ORDER BY s2 DESC LIMIT 1 "
            "MATCH (p1)-[:APPEARED_IN]->(g:Game)<-[:APPEARED_IN]-(p2) "
            "RETURN p1.name AS player1, p2.name AS player2, g.date AS matchDate, g.id AS gameId "
            "ORDER BY g.date DESC LIMIT 5"
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
