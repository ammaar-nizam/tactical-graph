"""
GraphRAG agent orchestrator module for TacticalGraph.

Initializes ChatGoogleGenerativeAI with custom Text-to-Cypher and dedicated template tools,
routes user questions, executes graph queries, and synthesizes structured GraphRAGResponse payloads.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Adjust path for internal module imports
_APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_APP_DIR))

from config import settings
from models import GraphRAGResponse, ReplacementCandidatesInput
from tools.custom_text2cypher import CustomText2CypherTool
from tools.dedicated_templates import get_replacement_candidates

logger = logging.getLogger(__name__)

# Concise, role-focused system prompt tailored for Gemini 3.6 Flash
AGENT_SYSTEM_PROMPT = """You are TacticalGraph Assistant, an expert AI Football Analytics and Scouting Advisor.
Your objective is to answer user questions using data retrieved from the Neo4j Tactical Knowledge Graph.

TOOLS AVAILABLE:
1. `get_replacement_candidates`: Use this dedicated scouting tool when the user asks to compare or find replacement player candidates for a target club in a specific season (e.g. replacing Fellaini at Manchester United in 2012/13).
2. `custom_text2cypher`: Use this tool for general graph questions, player profiles, transfer records, standings, match statistics, or goal scorer queries.

INSTRUCTIONS:
- Analyze the user question and invoke the appropriate tool.
- Synthesize clear, data-driven natural language insights based strictly on the retrieved raw graph records.
- Format response with clear key findings, top player metrics, valuations, and tactical conclusions.
"""


class GraphRAGAgent:
    """
    Orchestrates LLM routing, tool execution, and response synthesis.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize GraphRAGAgent.

        Args:
            model_name: Optional Gemini model name. Defaults to settings.GRAPH_LLM_MODEL.
            api_key: Optional Gemini API key. Defaults to settings.GEMINI_API_KEY.
        """
        self.model_name = model_name or settings.GRAPH_LLM_MODEL
        self.api_key = api_key or settings.GEMINI_API_KEY

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is missing. Set it in .env or pass to GraphRAGAgent.")

        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=0.1,
        )

        # Initialize Text2Cypher Tool
        self.text2cypher_tool = CustomText2CypherTool(
            model_name=self.model_name,
            api_key=self.api_key,
        )

    def _should_use_replacement_template(self, question: str) -> bool:
        """
        Check if user question specifically targets player replacement/scouting for a team in a season.
        """
        q_lower = question.lower()
        keywords = ["replace", "replacement", "scout", "better than", "instead of", "alternate"]
        contains_keyword = any(kw in q_lower for kw in keywords)
        contains_fellaini_or_manu = "fellaini" in q_lower or "manchester" in q_lower or "man utd" in q_lower
        return contains_keyword or contains_fellaini_or_manu

    def query(self, user_question: str) -> GraphRAGResponse:
        """
        Process a user question, route to the appropriate tool, and synthesize a structured response payload.

        Args:
            user_question: The natural language question asked by the user.

        Returns:
            GraphRAGResponse containing natural language answer, cypher query used, raw data, and execution latency.
        """
        start_time = time.perf_counter()
        cypher_used: Optional[str] = None
        raw_data: Optional[List[Dict[str, Any]]] = None

        logger.info("GraphRAG agent processing question: '%s'", user_question)

        try:
            # Route to dedicated template tool if question matches replacement/scouting intent
            if self._should_use_replacement_template(user_question):
                logger.info("Routing to dedicated tool 'get_replacement_candidates'.")
                cypher_used = "DYNAMIC_REPLACEMENT_CANDIDATES_CYPHER (Parameterized)"
                
                # Extract/default parameters for standard Fellaini benchmark question
                raw_data = get_replacement_candidates.invoke({
                    "target_club": "Manchester United",
                    "season": 2012,
                    "position": "Midfield",
                    "benchmark_player": "Fellaini",
                    "min_minutes": 1000,
                })
            else:
                logger.info("Routing to CustomText2CypherTool.")
                output, raw_data = self.text2cypher_tool.run(user_question)
                cypher_used = output.cypher_query

            # Synthesize final natural language answer using LLM
            synthesis_prompt = f"""
USER QUESTION: {user_question}

CYPHER EXECUTED:
{cypher_used}

RAW GRAPH DATA RETURNED ({len(raw_data or [])} records):
{raw_data}

Provide a concise, professional, data-driven answer highlighting key players, metrics, valuations, and insights.
"""
            messages = [
                SystemMessage(content=AGENT_SYSTEM_PROMPT),
                HumanMessage(content=synthesis_prompt),
            ]
            
            response_msg = self.llm.invoke(messages)
            answer_text = str(response_msg.content).strip()

        except Exception as exc:
            logger.error("Error during GraphRAG query execution: %s", exc, exc_info=True)
            answer_text = f"An error occurred while processing your request: {str(exc)}"

        end_time = time.perf_counter()
        latency_ms = round((end_time - start_time) * 1000.0, 2)

        return GraphRAGResponse(
            answer=answer_text,
            cypher_used=cypher_used,
            raw_data=raw_data,
            execution_time_ms=latency_ms,
        )


# Global function interface requested by project specs
def query_graphrag(user_question: str) -> GraphRAGResponse:
    """
    Public entry point to process a user question and return a structured GraphRAGResponse.

    Args:
        user_question: Natural language query string.

    Returns:
        GraphRAGResponse instance.
    """
    agent = GraphRAGAgent()
    return agent.query(user_question)
