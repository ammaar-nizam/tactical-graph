"""
GraphRAG agent orchestrator module for TacticalGraph.

Initializes ChatGoogleGenerativeAI bound with tools, routes user questions to appropriate tools,
and returns structured GraphRAGResponse payloads.
"""

import logging
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
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

# Router system prompt enforcing strict domain scope
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
"""

class GraphRAGAgent:
    """
    Orchestrates router tool selection, invokes encapsulated tools, and returns GraphRAGResponse.
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

        # Initialize Base LLM with max_output_tokens
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=0.0,
            max_output_tokens=settings.MAX_OUTPUT_TOKENS,
            thinking_level="low",
        )


        # Bind tools to LLM for router selection
        self.llm_with_tools = self.llm.bind_tools([
            get_replacement_candidates,
            query_graph_with_custom_cypher,
        ])

    def query(self, user_question: str) -> GraphRAGResponse:
        """
        Process a user question, route to tool, execute tool, and return GraphRAGResponse.

        Args:
            user_question: Natural language question asked by the user.

        Returns:
            GraphRAGResponse containing natural language answer and cypher query used.
        """
        cypher_used: Optional[str] = None
        answer_text: str = ""

        logger.info("GraphRAG agent processing question via tool router: '%s'", user_question)

        try:
            # Step 1: Let Gemini select the tool or detect out-of-scope question
            tool_selection_msg: AIMessage = self.llm_with_tools.invoke([
                SystemMessage(content=AGENT_SYSTEM_PROMPT),
                HumanMessage(content=user_question),
            ])

            selected_tool_name = None
            if hasattr(tool_selection_msg, "tool_calls") and tool_selection_msg.tool_calls:
                selected_tool_name = tool_selection_msg.tool_calls[0]["name"]
                logger.info("LangChain LLM selected tool: '%s'", selected_tool_name)

            # Step 2: Execute selected tool path or handle out-of-scope questions
            if selected_tool_name == "get_replacement_candidates":
                logger.info("Executing dedicated 'get_replacement_candidates' path.")
                answer_text = get_replacement_candidates.invoke({"question": user_question})
                cypher_used = "DYNAMIC_REPLACEMENT_CANDIDATES_CYPHER"

            elif selected_tool_name == "query_graph_with_custom_cypher":
                logger.info("Executing CustomText2CypherTool path.")
                answer_text = query_graph_with_custom_cypher.invoke({"question": user_question})
                if answer_text.startswith("Cypher Query Executed:\n"):
                    lines = answer_text.split("\n\n")
                    cypher_used = lines[0].replace("Cypher Query Executed:\n", "").strip()
                else:
                    cypher_used = "DYNAMIC_CUSTOM_TEXT2CYPHER"

            else:
                # Out-of-scope query: LLM selected no tool
                logger.info("No tool selected for question '%s' - question determined to be out of scope.", user_question)
                raw_content = extract_clean_text(getattr(tool_selection_msg, "content", ""))
                if raw_content and not raw_content.startswith("{"):
                    answer_text = raw_content
                else:
                    answer_text = OUT_OF_SCOPE_RESPONSE
                cypher_used = None

        except Exception as exc:
            logger.error("Error during GraphRAG query execution: %s", exc, exc_info=True)
            answer_text = f"An error occurred while processing your request: {str(exc)}"

        return GraphRAGResponse(
            answer=answer_text,
            cypher_used=cypher_used,
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
