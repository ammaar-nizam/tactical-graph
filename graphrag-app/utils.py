"""
Utility functions for GraphRAG application, text extraction, and report formatting.
"""

from typing import Any, Dict, List


def extract_clean_text(content: Any) -> str:
    """
    Extract clean string text from LLM response content blocks, removing any dict/list wrappers.

    Args:
        content: Raw content from LangChain ChatModel response (string or list of content blocks).

    Returns:
        Cleaned string.
    """
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif isinstance(block, dict) and "text" in block:
                text_parts.append(str(block["text"]))
            elif isinstance(block, str):
                text_parts.append(block)
        return "\n".join(text_parts).strip()

    return str(content).strip()


def format_scouting_report(records: List[Dict[str, Any]], params: Dict[str, Any]) -> str:
    """
    Format scouting database records into a clean, professional executive scouting report string.

    Args:
        records: Database records returned by get_replacement_candidates query.
        params: Extracted parameters dict.

    Returns:
        Formatted text report string.
    """
    target_club = params.get("target_club", "Target Club")
    season = params.get("season", 2013)
    benchmark_player = params.get("benchmark_player", "Benchmark Player")

    if not records:
        return f"No replacement candidates found matching the scouting criteria for {target_club} in season {season}/{season + 1}."

    benchmarks = [r for r in records if r.get("isBenchmark")]
    candidates = [r for r in records if not r.get("isBenchmark")]

    report_lines = [
        f"Executive Scouting Report: Replacement Candidates for {benchmark_player} ({target_club}, Season {season}/{season + 1})",
        "",
    ]

    if benchmarks:
        bm = benchmarks[0]
        val = bm.get("approxValuation")
        val_str = f"EUR {val:,.0f}" if val is not None else "N/A"
        age_str = f"{bm.get('age')} yrs" if bm.get("age") is not None else "N/A"
        report_lines.extend([
            f"1. Benchmark Baseline ({bm.get('candidate')}):",
            f"   - Age in Season: {age_str} | Role / Sub-Position: {bm.get('subPosition', 'N/A')}",
            f"   - Formation Fit: {bm.get('formationFitPct', 0.0)}% | Team Win Rate: {bm.get('winRatePct', 0.0)}%",
            f"   - Market Valuation: {val_str} | Goal Contributions: {bm.get('goals', 0)} goals, {bm.get('assists', 0)} assists ({bm.get('contributions', 0)} total)",
            "",
        ])

    if candidates:
        report_lines.append("2. Positional Scouting Analysis (Top Replacement Candidates):")
        for i, c in enumerate(candidates, 1):
            val = c.get("approxValuation")
            val_str = f"EUR {val:,.0f}" if val is not None else "N/A"
            cand_age = f"{c.get('age')} yrs" if c.get("age") is not None else "N/A"
            report_lines.extend([
                f"   {i}) {c.get('candidate')} ({c.get('subPosition', 'N/A')}):",
                f"      - Match Activity: {c.get('totalMinutes', 0)} mins in {c.get('totalMatches', 0)} matches",
                f"      - Formation Compatibility: {c.get('formationFitPct', 0.0)}% | Team Win Rate: {c.get('winRatePct', 0.0)}%",
                f"      - Market Valuation: {val_str} | Goal Contributions: {c.get('goals', 0)} goals, {c.get('assists', 0)} assists ({c.get('contributions', 0)} total)",
                f"      - Profile & Age: Age: {cand_age} | Height: {c.get('heightCm', 'N/A')} cm | Preferred Foot: {c.get('preferredFoot', 'N/A')}",
            ])

        report_lines.append("")
        top_cand = candidates[0].get("candidate")
        report_lines.extend([
            "3. Final Recruitment Recommendation:",
            f"   Primary recruitment target is {top_cand}, offering the optimal sub-position match, tactical formation fit, and team win-rate compatibility for {target_club}.",
        ])
    else:
        report_lines.extend([
            "2. Positional Scouting Analysis:",
            "   No secondary candidates met all valuation and minute constraints.",
        ])

    return "\n".join(report_lines)


def format_custom_cypher_records(cypher_query: str, records: List[Dict[str, Any]]) -> str:
    """
    Format Cypher query string and retrieved database records into a clean, human-readable answer.

    Args:
        cypher_query: Executed Cypher query.
        records: Data records returned from Neo4j session.

    Returns:
        Formatted text response string.
    """
    if not records:
        return f"Cypher Query Executed:\n{cypher_query}\n\nResult: No records found matching the query criteria."

    formatted_items = []
    for i, record in enumerate(records, 1):
        props = ", ".join(f"{k}: {v}" for k, v in record.items() if v is not None)
        formatted_items.append(f"{i}) {props}")

    return f"Cypher Query Executed:\n{cypher_query}\n\nRetrieved Knowledge Graph Records ({len(records)} record(s)):\n" + "\n".join(formatted_items)
