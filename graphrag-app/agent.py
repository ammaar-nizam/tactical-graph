"""
GraphRAG agent orchestrator module for TacticalGraph.

Initializes ChatGoogleGenerativeAI using native LangChain v1 create_agent
with short-term memory (InMemorySaver checkpointer) and context trimming middleware.
"""

import json
import logging
from typing import Any, Dict, Optional

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import before_model
from langchain_core.messages import RemoveMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

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

ERROR & NO MATCH HANDLING RULE:
- If a tool output or execution indicates an error or no matching records were found, return exactly this response:
  "I am sorry, I couldn't find what you are looking for. Please try again with a different question."

RESPONSE FORMATTING INSTRUCTION:
When formatting responses from retrieved graph database records or tools, synthesize the raw data into a clear, direct, and concise natural language answer matching the user question without raw dictionary strings, debug blocks, or Cypher code.
"""


@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> Dict[str, Any] | None:
    """
    Middleware function to trim message history, keeping the initial system message/context
    plus the most recent messages (3 user and 3 AI responses / recent turns) to fit the context window.
    """
    messages = state["messages"]

    if len(messages) <= 3:
        return None  # No changes needed

    first_msg = messages[0]
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    new_messages = [first_msg] + recent_messages

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages,
        ]
    }


# Shared default checkpointer for thread-level short-term memory persistence
_DEFAULT_CHECKPOINTER = InMemorySaver()


class GraphRAGAgent:
    """
    Orchestrates router tool selection and execution using native LangChain create_agent
    with short-term memory checkpointer and @before_model message trimming middleware.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        checkpointer: Optional[Any] = None,
    ) -> None:
        """
        Initialize GraphRAGAgent with native LangChain v1 create_agent and checkpointer.

        Args:
            model_name: Optional Gemini model identifier. Defaults to settings.LLM_MODEL.
            api_key: Optional Gemini API key. Defaults to settings.GEMINI_API_KEY.
            checkpointer: Optional Checkpointer instance. Defaults to global _DEFAULT_CHECKPOINTER.
        """
        self.model_name = model_name or settings.LLM_MODEL
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.checkpointer = checkpointer if checkpointer is not None else _DEFAULT_CHECKPOINTER

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is missing. Set it in .env or pass to GraphRAGAgent.")

        # Initialize ChatGoogleGenerativeAI
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

        # Create agent with short-term memory checkpointer & @before_model trimming middleware
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=AGENT_SYSTEM_PROMPT,
            checkpointer=self.checkpointer,
            middleware=[trim_messages],
        )

    def query(self, user_question: str, thread_id: str = "default") -> GraphRAGResponse:
        """
        Process a user question with short-term memory context and return structured GraphRAGResponse.

        Args:
            user_question: Natural language question asked by the user.
            thread_id: Thread/session identifier for maintaining short-term memory context.

        Returns:
            GraphRAGResponse containing natural language answer and cypher query used.
        """
        logger.info("GraphRAG agent processing question for thread_id '%s': '%s'", thread_id, user_question)

        cypher_used: Optional[str] = None
        answer_text: str = ""
        thread_config = {"configurable": {"thread_id": thread_id}}

        try:
            res = self.agent.invoke(
                {"messages": [("user", user_question)]},
                config=thread_config,
            )
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
            answer_text = "I am sorry, I couldn't find what you are looking for. Please try again with a different question."

        final_answer = answer_text or OUT_OF_SCOPE_RESPONSE
        logger.info("Answer: %s", final_answer)

        return GraphRAGResponse(
            answer=final_answer,
            cypher_used=cypher_used,
        )


def query_graphrag(user_question: str, thread_id: str = "default") -> GraphRAGResponse:
    """
    Public entry point to process a user question with short-term memory context.

    Args:
        user_question: Natural language query string.
        thread_id: Thread/session identifier for maintaining short-term context.

    Returns:
        GraphRAGResponse instance.
    """
    agent = GraphRAGAgent()
    return agent.query(user_question, thread_id=thread_id)

