"""
D&D Buddy Agent - LangGraph implementation with dual-model architecture.

Architecture:
- Cheap model (Nova Micro): Tool planning and execution loops
- Expensive model (configurable): Final creative response generation

Tools: search_campaign, roll_dice, get_file_content, get_conversation_history
"""
import os
import logging
from typing import Dict, Any, List, Optional, Callable
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.callbacks.base import BaseCallbackHandler
from langgraph.graph import StateGraph, END, MessagesState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import tools
from tools import search_campaign, roll_dice, get_file_content, get_conversation_history
from tools.get_history import get_history_messages, save_messages

# =============================================================================
# Configuration
# =============================================================================

BEDROCK_MODEL_CREATIVE = os.environ.get('BEDROCK_MODEL_ID', 'eu.amazon.nova-micro-v1:0')
BEDROCK_MODEL_PLANNING = os.environ.get('BEDROCK_MODEL_ID_TOOL', 'eu.amazon.nova-micro-v1:0')
MAX_ITERATIONS = int(os.environ.get('MAX_AGENT_ITERATIONS', '3'))

# Tool registry
TOOLS = [search_campaign, roll_dice, get_file_content, get_conversation_history]
TOOL_MAP = {tool.name: tool for tool in TOOLS}
CONTEXT_TOOLS = {'search_campaign', 'get_file_content'}
SESSION_TOOLS = {'get_conversation_history'}


# =============================================================================
# Streaming Handler
# =============================================================================

class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming LLM responses to websocket."""
    
    def __init__(self, stream_callback: Callable[[Any], None]):
        self.stream_callback = stream_callback
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Forward tokens to the stream callback."""
        if not token:
            return
        if isinstance(token, (dict, list)):
            self.stream_callback(token)
        else:
            self.stream_callback([{"type": "text", "text": token, "index": 0}])


# =============================================================================
# LLM Factory
# =============================================================================

def create_planning_llm() -> ChatBedrock:
    """Create the cheap model for tool planning (no streaming)."""
    return ChatBedrock(
        model_id=BEDROCK_MODEL_PLANNING,
        model_kwargs={
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1,
            "max_tokens": 400
        },
        streaming=False
    )


def create_creative_llm(stream_callback: Optional[Callable] = None) -> ChatBedrock:
    """Create the expensive model for final response (with streaming)."""
    callbacks = []
    if stream_callback:
        callbacks.append(StreamingCallbackHandler(stream_callback))
    
    return ChatBedrock(
        model_id=BEDROCK_MODEL_CREATIVE,
        model_kwargs={"temperature": 0.6, "max_tokens": 1500},
        streaming=True,
        callbacks=callbacks
    )


# =============================================================================
# Agent Graph Builder
# =============================================================================

class AgentGraphBuilder:
    """Builds the LangGraph agent with tool execution capability."""
    
    def __init__(self, user_id: str, campaign: str, session_id: str,
                 stream_callback: Optional[Callable] = None):
        self.user_id = user_id
        self.campaign = campaign
        self.session_id = session_id
        self.stream_callback = stream_callback
        self.tools_executed: List[str] = []
        
        self.planning_llm = create_planning_llm().bind_tools(TOOLS)
        self.creative_llm = create_creative_llm(stream_callback)
    
    def _agent_node(self, state: MessagesState) -> Dict[str, List]:
        """Agent reasoning node - uses cheap model for planning."""
        messages = state["messages"]
        
        iteration_count = sum(
            1 for msg in messages 
            if isinstance(msg, AIMessage) and getattr(msg, 'tool_calls', None)
        )
        
        logger.info(f"Agent planning - iteration {iteration_count + 1}/{MAX_ITERATIONS}")
        
        if iteration_count >= MAX_ITERATIONS:
            logger.warning("Max iterations reached, forcing final response")
            return {"messages": [AIMessage(content="[MAX_ITERATIONS_REACHED]")]}
        
        response = self.planning_llm.invoke(messages)
        
        if getattr(response, 'tool_calls', None):
            logger.info(f"Planning: {len(response.tool_calls)} tool call(s) requested")
            return {"messages": [response]}
        
        logger.info("Planning complete - no more tools needed")
        return {"messages": [AIMessage(content="[PLANNING_COMPLETE]")]}
    
    def _tool_node(self, state: MessagesState) -> Dict[str, List]:
        """Execute tools with automatic context injection."""
        messages = state["messages"]
        last_message = messages[-1]
        tool_calls = getattr(last_message, 'tool_calls', [])
        tool_messages = []
        
        logger.info(f"Executing {len(tool_calls)} tool(s)")
        
        for tool_call in tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args'].copy()
            
            if tool_name in CONTEXT_TOOLS:
                tool_args['user_id'] = self.user_id
                tool_args['campaign'] = self.campaign
            if tool_name in SESSION_TOOLS:
                tool_args['session_id'] = self.session_id
            
            tool_func = TOOL_MAP.get(tool_name)
            if tool_func:
                result = tool_func.invoke(tool_args)
                logger.info(f"  -> {tool_name}: {len(str(result))} chars")
            else:
                result = f"Unknown tool: {tool_name}"
                logger.error(f"  -> Unknown tool: {tool_name}")
            
            display_args = {k: v for k, v in tool_args.items() 
                          if k not in ['user_id', 'campaign', 'session_id']}
            if display_args:
                args_str = ', '.join(f"{k}={repr(v)}" for k, v in display_args.items())
                self.tools_executed.append(f"{tool_name}({args_str})")
            else:
                self.tools_executed.append(f"{tool_name}()")
            
            tool_messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_call['id'])
            )
        
        return {"messages": tool_messages}
    
    def _should_continue(self, state: MessagesState) -> str:
        """Route: tools if tool calls present, else end."""
        last_message = state["messages"][-1]
        if getattr(last_message, 'tool_calls', None):
            return "tools"
        return END
    
    def build(self) -> StateGraph:
        """Build and compile the agent graph."""
        workflow = StateGraph(MessagesState)
        
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", self._tool_node)
        
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", self._should_continue, {"tools": "tools", END: END})
        workflow.add_edge("tools", "agent")
        
        return workflow.compile()
    
    def generate_final_response(self, messages: List, tool_results: List) -> str:
        """Generate final creative response using expensive model."""
        logger.info("Generating final response with creative model (streaming)")
        
        final_messages = messages.copy()
        
        if tool_results:
            tool_context = "\n\n".join([msg.content for msg in tool_results])
            final_messages.append(HumanMessage(
                content=f"Based on the following information gathered:\n\n{tool_context}\n\nProvide a helpful response."
            ))
        
        response = self.creative_llm.invoke(final_messages)
        return response.content if hasattr(response, 'content') else str(response)
    
    def get_tools_summary(self) -> str:
        """Get formatted summary of tools used."""
        if not self.tools_executed:
            return ""
        return f"\n\n---\n_Tools used: {', '.join(self.tools_executed)}_"


# =============================================================================
# System Prompt Builder
# =============================================================================

def build_system_prompt(campaign: str, campaign_context: str, recent_sessions: str) -> str:
    """Build the system prompt with campaign context."""
    return f"""You are **D&D Buddy**, an expert D&D 5e campaign assistant for the campaign: **{campaign}**.

---

## CAMPAIGN CONTEXT

{campaign_context}

{recent_sessions}

---

## YOUR CAPABILITIES

- **Access**: You see the user's entire campaign database—NPCs, monsters, session logs, lore, organizations, homebrew, and more.
- **System awareness**: You understand the campaign's established tone, themes, magic system, current events, and style based on the context above.

---

## GENERATION PRINCIPLES

**Be campaign-specific:**  
- All responses, world integrations, and creative ideas must _fit this setting_ by leveraging campaign files and recent events.
- Use the exact tone, genre, and world logic you find in the campaign context.
- Never revert to "generic D&D" tropes. Always relate to the people, factions, magic, history, and style defined above.

**Prioritize facts and context:**  
- Synthesize from search results rather than making assumptions.
- If required context is missing from {campaign}, _say so clearly and suggest possible actions (e.g. ask GM, search another term)_.  
- For creative/world-integration queries ("how does X fit in?"), cross-reference NPCs, factions, lore, session events, and rules.

---

## RESPONSE FORMAT

- **Length**: Target 100–300 words (500 max if user requests detail).
- Use **bold** for names and places, `code` for D&D mechanics or dice, bullets for lists, ### for sections.
- For answers about characters, places, items: use this structure—**Name/Type:** one-sentence description, 1-2 key details, role/connections.
- Use bullets for lists (never overlong prose).

**DO NOT:**
- Add unnecessary filler or repetition.
- Summarize "the file says…" unless clarifying context.
- Use long narrative paragraphs when concise bullets/sections work.

---

## TOOLS AVAILABLE

1. **search_campaign**: Semantic search—use _first_ for all campaign info needs. Returns relevant snippets from ALL files (NPCs, monsters, sessions, lore, organizations, custom content).
2. **get_file_content**: Full file text—use only if user requests "everything", same file is referenced in multiple search results, or a section is missing.
3. **roll_dice**: D&D dice syntax (e.g., `1d20+5`, `2d6`).
4. **get_conversation_history**: Retrieve _earlier_ messages if referenced (user says "as we discussed"). Only specify number of messages needed.

---

## SEARCH STRATEGY

- Always begin with **search_campaign** (no categories; index is unified).
- Use focused queries: character name, monster, topic, place, event, or rules keyword.
- For creative/integration questions ("How does Warforged fit into this Axiom?"):
    - Search _multiple related terms_: race/species name, factions, magic history, relevant sessions.
    - Synthesize connections across all returned context.
- For multi-part or vague questions: run several targeted searches and aggregate.
- If nothing relevant returned: propose how the GM or players might establish this.

**Result handling:**
- Each search result includes filename and short snippet—use these as _authoritative context_.
- Prefer summarizing/synthesizing over direct quoting, but cite exact file when facts are precise: e.g., `According to [filename.md]: ...`.

---

## CONVERSATION CONTEXT

- By default, you only see the last exchange (user + your response).
- If user references an older conversation, or clarifying a prior answer is needed, call **get_conversation_history(message_count=N)** (N=2 per turn; more for longer memory).
- Never assume prior turns unless loaded.

---

## DICE ROLLS

- Interpret queries naturally ("roll perception" → `roll_dice("1d20+mod")`).
- Always present who/what is rolling. Call out crits: 20 = "🎉 Critical!" and 1 = "💀 Critical failure!"

---

## RULES OF GENERATION

- **NEVER invent info**—only synthesize/creatively build from search results/context above.
- **Combine** multiple results for each query into a coherent, non-redundant answer.
- If uncertain/absent: _clearly indicate the gap and suggest user/GM action_ rather than "filling in" with assumptions.
- _Always_ cite sources (filename, section name, or session number) for campaign specifics, e.g., "(source: sessions/session12.md)".

---
"""


# =============================================================================
# Context Loaders
# =============================================================================

def load_campaign_context(user_id: str, campaign: str) -> str:
    """Load campaign background context via search."""
    try:
        results = search_campaign.invoke({
            'query': 'world setting themes tone style history magic system background lore',
            'top_k': 3,
            'user_id': user_id,
            'campaign': campaign
        })
        
        if results and "No relevant information found" not in results:
            logger.info(f"Loaded campaign context: {len(results)} chars")
            return f"\n{results}\n"
    except Exception as e:
        logger.warning(f"Failed to load campaign context: {e}")
    
    return "\n_No campaign background information available yet._\n"


def load_recent_sessions(user_id: str, campaign: str) -> str:
    """Load recent session context via search."""
    try:
        results = search_campaign.invoke({
            'query': 'recent session last game latest adventure current quest',
            'top_k': 2,
            'user_id': user_id,
            'campaign': campaign
        })
        
        if results and "No relevant information found" not in results:
            logger.info(f"Loaded recent sessions: {len(results)} chars")
            return f"\n**RECENT SESSIONS:**\n{results}\n"
    except Exception as e:
        logger.warning(f"Failed to load recent sessions: {e}")
    
    return ""


# =============================================================================
# Main Entry Point
# =============================================================================

def main(input_data: Dict[str, Any], stream_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Main entry point for the D&D Buddy agent.
    
    Args:
        input_data: Dictionary containing userId, campaign, prompt, sessionId
        stream_callback: Optional callback function to stream response chunks
    
    Returns:
        Dictionary with response, userId, campaign, sessionId (or error)
    """
    logger.info("=" * 80)
    logger.info("Agent invocation started")
    logger.info(f"Streaming enabled: {stream_callback is not None}")
    
    # Extract and validate parameters
    user_id = input_data.get('userId')
    campaign = input_data.get('campaign')
    user_message = input_data.get('prompt')
    session_id = input_data.get('sessionId', f"{user_id}-{campaign}-default")
    
    logger.info(f"User: {user_id}, Campaign: {campaign}, Session: {session_id}")
    
    if not all([user_id, campaign, user_message]):
        logger.error("Missing required parameters")
        return {'error': 'Missing required parameters: userId, campaign, prompt'}
    
    # Security: validate session ownership
    if not session_id.startswith(f"{user_id}-"):
        logger.error(f"Session validation failed: '{session_id}' doesn't belong to '{user_id}'")
        return {'error': 'Invalid session: session does not belong to the authenticated user'}

    try:
        # Load conversation history
        history_messages = get_history_messages(session_id, message_count=2)
        logger.info(f"Loaded {len(history_messages)} messages from history")
        
        # Load campaign context
        campaign_context = load_campaign_context(user_id, campaign)
        recent_sessions = load_recent_sessions(user_id, campaign)
        
        # Build agent
        agent_builder = AgentGraphBuilder(user_id, campaign, session_id, stream_callback)
        agent = agent_builder.build()
        
        # Build messages
        system_message = SystemMessage(content=build_system_prompt(campaign, campaign_context, recent_sessions))
        current_user_message = HumanMessage(content=user_message)
        messages = [system_message] + history_messages + [current_user_message]
        
        logger.info(f"Invoking agent with {len(messages)} messages")
        
        # Phase 1: Tool planning and execution (cheap model, no streaming)
        result = agent.invoke({"messages": messages})
        
        # Collect tool results for context
        tool_results = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
        
        # Phase 2: Final response generation (expensive model, with streaming)
        final_messages = messages.copy()
        
        if tool_results:
            tool_context = "\n\n".join([f"**Tool Result:**\n{msg.content}" for msg in tool_results])
            final_messages.append(HumanMessage(
                content=f"I've gathered the following information:\n\n{tool_context}\n\nProvide a helpful response."
            ))
        
        # Generate final response with creative model
        response_text = agent_builder.generate_final_response(final_messages, tool_results)

        # Append tools summary
        tools_summary = agent_builder.get_tools_summary()
        if tools_summary:
            response_text += tools_summary
            if stream_callback:
                stream_callback([{"type": "text", "text": tools_summary, "index": 0}])
        
        # Save to history (without tools summary)
        save_messages(
            session_id=session_id,
            user_message=user_message,
            ai_message=response_text.replace(tools_summary, "").strip() if tools_summary else response_text
        )
        
        logger.info("Agent invocation completed successfully")
        logger.info("=" * 80)
        
        return {
            'response': response_text,
            'userId': user_id,
            'campaign': campaign,
            'sessionId': session_id
        }
        
    except Exception as e:
        logger.error(f"Agent invocation failed: {e}", exc_info=True)
        logger.info("=" * 80)
        return {'error': str(e)}
