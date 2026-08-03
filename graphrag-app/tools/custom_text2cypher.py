"""
Custom Text-to-Cypher LLM tool for translating user natural language questions into
validated, read-only Neo4j Cypher queries.
"""

import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from neo4j import GraphDatabase

# Adjust path to import config and models
_APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_APP_DIR))

from config import settings
from models import CypherGenerationOutput
from few_shot_bank import get_few_shot_prompt

logger = logging.getLogger(__name__)

# Path to schema reference
_SCHEMA_MD_PATH = _APP_DIR.parent / "data-pipeline" / "schema.md"

# Forbidden write/mutation Cypher keywords
MUTATION_KEYWORDS = {
    "CREATE", "MERGE", "DELETE", "DETACH", "SET", "REMOVE", "DROP", "ALTER", "GRANT", "REVOKE"
}


def load_schema_description() -> str:
    """
    Load graph schema definitions from data-pipeline/schema.md.

    Returns:
        String containing markdown schema text or default schema summary.
    """
    if _SCHEMA_MD_PATH.exists():
        try:
            return _SCHEMA_MD_PATH.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to read schema.md: %s", e)

    return (
        "Node Labels: :Player(id, name, position, subPosition, heightInCm, foot, dateOfBirth, countryOfCitizenship)\n"
        ":Club(id, name, code, totalMarketValue, squadSize)\n"
        ":Game(id, season, date, homeClubGoals, awayClubGoals, homeClubFormation, awayClubFormation)\n"
        ":GameEvent(id, type, minute, description)\n"
        ":PlayerValuation(id, date, marketValueInEur)\n"
        ":Competition(id, name, type, subType), :Country(id, name), :NationalTeam(id, name)\n"
        "Relationships: (Player)-[:PLAYS_FOR]->(Club), (Player)-[:HAS_VALUATION]->(PlayerValuation),\n"
        "(Player)-[:TRANSFERRED_TO {transferFee, transferDate, transferSeason, fromClubName}]->(Club),\n"
        "(Player)-[:APPEARED_IN {minutesPlayed, goals, assists, yellowCards, redCards, position, teamCaptain}]->(Game),\n"
        "(Club)-[:PLAYED_IN {hosting, isWin, ownGoals, opponentGoals}]->(Game),\n"
        "(Game)-[:PART_OF_COMPETITION]->(Competition), (Player)-[:SCORED]->(GameEvent)-[:OCCURRED_IN]->(Game)"
    )


class CustomText2CypherTool:
    """
    Custom Text-to-Cypher tool utilizing ChatGoogleGenerativeAI with structured output
    to translate natural language questions into validated read-only Cypher queries.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize CustomText2CypherTool.

        Args:
            model_name: Optional Gemini model identifier. Defaults to settings.GRAPH_LLM_MODEL.
            api_key: Optional Gemini API key. Defaults to settings.GEMINI_API_KEY.
        """
        self.model_name = model_name or settings.GRAPH_LLM_MODEL
        self.api_key = api_key or settings.GEMINI_API_KEY

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is missing. Set it in .env or pass to CustomText2CypherTool.")

        # Initialize LangChain Gemini LLM with structured output
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=0.0,
        )
        self.structured_llm = self.llm.with_structured_output(CypherGenerationOutput)

        # Build Neo4j connection URI (using SSL fallback if required on Windows)
        self.uri = settings.NEO4J_URI.replace("neo4j+s://", "neo4j+ssc://")
        self.auth = (settings.NEO4J_USER, settings.NEO4J_PASSWORD)

        # Load schema and few-shot prompt bank
        self.schema_text = load_schema_description()
        self.few_shot_prompt = get_few_shot_prompt()

    def _build_system_instructions(self) -> str:
        """
        Build concise, direct system instructions for Gemini 3.6 Flash.
        """
        return f"""You are a specialized Neo4j Cypher generation expert for a Football Knowledge Graph (TacticalGraph).
Your task is to translate user natural language questions into precise, production-ready, read-only Cypher queries.

### GRAPH SCHEMA DEFINITIONS:
{self.schema_text}

### RULES & CONVENTIONS:
1. Generate STRICTLY READ-ONLY queries using MATCH, OPTIONAL MATCH, WHERE, WITH, RETURN, ORDER BY, LIMIT.
2. NEVER use write/mutation clauses: CREATE, MERGE, DELETE, DETACH, SET, REMOVE, DROP.
3. Case-insensitive string matching: Always use toLower(n.prop) CONTAINS 'term' for names (e.g. toLower(p.name) CONTAINS 'fellaini').
4. Filter out stub/null names: Include `WHERE p.name IS NOT NULL` or `c.name IS NOT NULL` when querying players/clubs.
5. `isWin` on [:PLAYED_IN] is integer 1 or 0 (NOT boolean true/false).
6. Convert numeric/string properties safely: use `toLower(toString(p.subPosition))` for position checks to handle null/NaN float values safely.

### FEW-SHOT EXAMPLES:
{self.few_shot_prompt}

Return your response strictly adhering to the CypherGenerationOutput schema.
"""

    def validate_cypher(self, cypher: str) -> Tuple[bool, str]:
        """
        Validate that generated Cypher query is non-empty and strictly read-only.

        Args:
            cypher: The generated Cypher query string.

        Returns:
            Tuple of (is_valid: bool, error_message: str).
        """
        if not cypher or not cypher.strip():
            return False, "Generated Cypher query is empty."

        clean_cypher = cypher.strip().upper()

        # Check for mutation keywords as standalone tokens
        tokens = set(re.findall(r"\b[A-Z]+\b", clean_cypher))
        mutations_found = tokens.intersection(MUTATION_KEYWORDS)
        if mutations_found:
            return False, f"Cypher query contains non-read-only mutation keywords: {list(mutations_found)}"

        if not (clean_cypher.startswith("MATCH") or clean_cypher.startswith("WITH") or clean_cypher.startswith("OPTIONAL MATCH") or clean_cypher.startswith("//")):
            return False, "Query must begin with MATCH, OPTIONAL MATCH, or WITH."

        return True, ""

    def execute_cypher(self, cypher: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute validated Cypher query against Neo4j instance.

        Args:
            cypher: Read-only Cypher query to execute.
            parameters: Optional query parameters.

        Returns:
            List of record dicts.
        """
        driver = GraphDatabase.driver(self.uri, auth=self.auth)
        try:
            with driver.session(database=settings.NEO4J_DATABASE) as session:
                result = session.run(cypher, parameters or {})
                return [record.data() for record in result]
        finally:
            driver.close()

    def run(self, user_question: str) -> Tuple[CypherGenerationOutput, List[Dict[str, Any]]]:
        """
        Translate user prompt into Cypher, validate read-only status, execute query, and return structured output.

        Args:
            user_question: Natural language question asked by user.

        Returns:
            Tuple of (CypherGenerationOutput, raw_data_list).
        """
        system_instruction = self._build_system_instructions()

        messages = [
            SystemMessage(content=system_instruction),
            HumanMessage(content=user_question),
        ]

        logger.info("Translating question to Cypher: '%s'", user_question)
        output: CypherGenerationOutput = self.structured_llm.invoke(messages)

        # Validate Cypher query
        is_valid, err_msg = self.validate_cypher(output.cypher_query)
        if not is_valid:
            logger.error("Generated Cypher failed read-only validation: %s", err_msg)
            output.is_read_only = False
            raise ValueError(f"Invalid generated Cypher query: {err_msg}")

        # Execute query against Neo4j
        logger.info("Executing Cypher: %s", output.cypher_query)
        raw_data = self.execute_cypher(output.cypher_query)
        logger.info("Query returned %d record(s).", len(raw_data))

        return output, raw_data
