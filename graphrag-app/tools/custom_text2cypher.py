"""
Custom Text-to-Cypher LLM tool for translating user natural language questions into
validated, read-only Neo4j Cypher queries.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from config import settings
from database import execute_cypher_query
from few_shot_bank import get_few_shot_prompt
from models import CypherGenerationOutput
from utils import format_custom_cypher_records

logger = logging.getLogger(__name__)

# Forbidden write/mutation Cypher keywords
MUTATION_KEYWORDS = {
    "CREATE", "MERGE", "DELETE", "DETACH", "SET", "REMOVE", "DROP", "ALTER", "GRANT", "REVOKE"
}


def get_schema() -> str:
    """
    Extract dynamic graph schema (Node Labels, Relationship Types, and Properties)
    from Neo4j database using Cypher schema introspection procedures.

    Returns:
        Formatted string summarizing graph schema definitions.
    """
    # Query node labels and property keys via Cypher procedure
    node_records = execute_cypher_query(
        "CALL db.schema.nodeTypeProperties() YIELD nodeType, propertyName "
        "RETURN nodeType, collect(propertyName) AS properties"
    )

    # Query relationship types and property keys via Cypher procedure
    rel_records = execute_cypher_query(
        "CALL db.schema.relTypeProperties() YIELD relType, propertyName "
        "RETURN relType, collect(propertyName) AS properties"
    )

    schema_parts = ["### GRAPH SCHEMA DEFINITIONS (Extracted dynamically via Cypher):", ""]

    if node_records:
        schema_parts.append("Node Labels & Properties:")
        for row in node_records:
            node_label = row.get("nodeType", "")
            props = ", ".join(row.get("properties", []))
            schema_parts.append(f"- {node_label} ({props})")
        schema_parts.append("")

    if rel_records:
        schema_parts.append("Relationships & Edge Properties:")
        for row in rel_records:
            rel_type = row.get("relType", "")
            props = ", ".join(row.get("properties", []))
            if props:
                schema_parts.append(f"- {rel_type} {{{props}}}")
            else:
                schema_parts.append(f"- {rel_type}")

    return "\n".join(schema_parts)


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

        # Initialize LangChain Gemini LLM with structured output and max_output_tokens
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=0.0,
            max_output_tokens=settings.MAX_OUTPUT_TOKENS,
            thinking_level="low",
        )

        self.structured_llm = self.llm.with_structured_output(CypherGenerationOutput)

        # Dynamically extract graph schema via Cypher procedures & load few-shot prompt bank
        self.schema_text = get_schema()
        self.few_shot_prompt = get_few_shot_prompt()


    def _build_system_instructions(self) -> str:
        """
        Build concise, direct system instructions for Text-to-Cypher generation.
        """
        return f"""You are a specialized Neo4j Cypher generation expert for a Football Knowledge Graph (TacticalGraph).
Your task is to translate user natural language questions into precise, production-ready, read-only Cypher queries.

### GRAPH SCHEMA DEFINITIONS:
{self.schema_text}

### RULES & CONVENTIONS:
1. Generate strictly read-only queries using MATCH, OPTIONAL MATCH, WHERE, WITH, RETURN, ORDER BY, LIMIT.
2. Never use write/mutation clauses: CREATE, MERGE, DELETE, DETACH, SET, REMOVE, DROP.
3. Strict limit clause: Always append `LIMIT 5` (or lower) to every generated query unless explicitly requested otherwise.
4. Case-insensitive string matching: Always use toLower(n.prop) CONTAINS 'term' for names (e.g. toLower(p.name) CONTAINS 'fellaini').
5. Filter out stub/null names: Include `WHERE p.name IS NOT NULL` or `c.name IS NOT NULL` when querying players/clubs.
6. `isWin` on [:PLAYED_IN] is integer 1 or 0 (NOT boolean true/false).
7. Convert numeric/string properties safely: use `toLower(toString(p.subPosition))` for position checks.
8. Domain boundary & Out-of-scope rule: The domain is strictly football (soccer). If the user question is unrelated to football or non-queryable against the graph schema, do NOT invent queries for out-of-scope topics.


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
        return execute_cypher_query(cypher, parameters)



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


@tool("query_graph_with_custom_cypher")

def query_graph_with_custom_cypher(question: str) -> str:
    """
    Execute a custom natural language query on the TacticalGraph knowledge graph by generating read-only Cypher.
    Use this tool for general football queries, player statistics, match results, transfers, and general club info.

    Args:
        question: Natural language query string.

    Returns:
        Formatted string containing generated Cypher query and execution record results.
    """
    tool_inst = CustomText2CypherTool()
    out, data = tool_inst.run(question)
    return format_custom_cypher_records(out.cypher_query, data)


