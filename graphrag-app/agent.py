"""
GraphRAG agent orchestrator module for TacticalGraph.

Initializes ChatGoogleGenerativeAI using native LangChain v1 create_agent
"""

import json
import logging
from typing import Optional

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from config import settings
from models import GraphRAGResponse
from tools.custom_text2cypher import query_graph_with_custom_cypher
from tools.dedicated_templates import get_replacement_candidates
from utils import extract_clean_text

logger = logging.getLogger(__name__)

# Default response returned when a question is out of domain scope
OUT_OF_SCOPE_RESPONSE = (
    "I am a specialized Football Knowledge Graph assistant for TacticalGraph. "
    "Your question appears to be out of scope. Please ask questions related to football players, "
    "clubs, match statistics, transfer history, or tactical scouting."
)

# Router system prompt enforcing strict domain scope & response formatting
AGENT_SYSTEM_PROMPT = """You are Chief Recruitment Analyst & Executive Football Scout for TacticalGraph, a specialized Football Knowledge Graph system.

STRICT DOMAIN SCOPE & BOUNDARIES:
- Your domain is EXCLUSIVELY football (soccer).
- Supported topics include: player statistics, squad rosters, player scouting/replacement candidate comparisons, match results, transfer history, game events, club performance, national teams, and football competitions.
- NON-FOOTBALL OR OUT-OF-SCOPE TOPICS ARE STRICTLY FORBIDDEN (e.g. general knowledge, non-football sports, coding, weather, politics, finance, entertainment, general conversation, math, or recipes).

TOOL ROUTING INSTRUCTIONS:
1. If the question is about scouting, replacing, finding alternatives, or comparing players for a football team:
   Select tool: 'get_replacement_candidates'
2. If the question is a general football query about player stats, matches, clubs, transfers, or competitions:
   Select tool: 'query_graph_with_custom_cypher'
3. If the user question is OUT OF SCOPE (not related to football):
   Do NOT call any tool. Respond directly stating politely that you are a specialized football assistant and the question is out of scope.

SINGLE TOOL EXECUTION RULE:
- Execute at most ONE tool call per user question.
- Do NOT retry calling tools or loop if a tool query returns 0 records or no matches found in the database. Synthesize your final natural language answer immediately based on the tool result returned.

RESPONSE FORMATTING INSTRUCTION:
When formatting responses from retrieved graph database records or tools, synthesize the raw data into a clear, direct, and concise natural language answer matching the user question without raw dictionary strings, debug blocks, or Cypher code.
"""

class GraphRAGAgent:
    """
    Orchestrates router tool selection and execution using native LangChain create_agent.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize GraphRAGAgent with native LangChain v1 create_agent.

        Args:
            model_name: Optional Gemini model identifier. Defaults to settings.LLM_MODEL.
            api_key: Optional Gemini API key. Defaults to settings.GEMINI_API_KEY.
        """
        self.model_name = model_name or settings.LLM_MODEL
        self.api_key = api_key or settings.GEMINI_API_KEY

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is missing. Set it in .env or pass to GraphRAGAgent.")

        # Initialize ChatGoogleGenerativeAI with thinking_level="low", max_output_tokens, and max 3 retries
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=0.0,
            max_output_tokens=settings.MAX_OUTPUT_TOKENS,
            thinking_level="low",
            max_retries=3,
        )
        
        # Define tools for the agent
        self.tools = [
            get_replacement_candidates,
            query_graph_with_custom_cypher,
        ]

        # Create agent
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=AGENT_SYSTEM_PROMPT,
        )

    def query(self, user_question: str) -> GraphRAGResponse:
        """
        Process a user question and return structured GraphRAGResponse.

        Args:
            user_question: Natural language question asked by the user.

        Returns:
            GraphRAGResponse containing natural language answer and cypher query used.
        """
        logger.info("GraphRAG agent processing question via LangChain agent: '%s'", user_question)

        cypher_used: Optional[str] = None
        answer_text: str = ""

        try:
            res = self.agent.invoke({"messages": [("user", user_question)]})
            messages = res.get("messages", [])

            if messages:
                last_msg = messages[-1]
                answer_text = extract_clean_text(getattr(last_msg, "content", ""))

            # Inspect message sequence to extract cypher_used if a tool was executed
            for msg in messages:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        tname = tool_call.get("name", "")
                        if tname == "get_replacement_candidates":
                            cypher_used = "DYNAMIC_REPLACEMENT_CANDIDATES_CYPHER"

                if hasattr(msg, "content") and isinstance(msg.content, str):
                    content = msg.content
                    if "Cypher Query Executed:\n" in content:
                        parts = content.split("Cypher Query Executed:\n")
                        if len(parts) > 1:
                            cypher_used = parts[1].split("\n\n")[0].strip()
                    elif content.startswith("{"):
                        try:
                            parsed = json.loads(content)
                            if "cypher_used" in parsed:
                                cypher_used = parsed.get("cypher_used")
                        except Exception:
                            pass

        except Exception as exc:
            logger.error("Error during GraphRAG query execution: %s", exc, exc_info=True)
            answer_text = f"An error occurred while processing your request: {str(exc)}"

        final_answer = answer_text or OUT_OF_SCOPE_RESPONSE
        logger.info("Answer: %s", final_answer)

        return GraphRAGResponse(
            answer=final_answer,
            cypher_used=cypher_used,
        )


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
