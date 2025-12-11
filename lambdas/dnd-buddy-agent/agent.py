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
from tools import search_campaign, roll_dice, get_file_content, get_conversation_history, translate_runes
from tools.get_history import get_history_messages, save_messages

# =============================================================================
# Configuration
# =============================================================================

BEDROCK_MODEL_CREATIVE = os.environ.get('BEDROCK_MODEL_ID', 'eu.amazon.nova-micro-v1:0')
BEDROCK_MODEL_PLANNING = os.environ.get('BEDROCK_MODEL_ID_TOOL', 'eu.amazon.nova-micro-v1:0')
MAX_ITERATIONS = int(os.environ.get('MAX_AGENT_ITERATIONS', '3'))

# Tool registry
TOOLS = [search_campaign, roll_dice, get_file_content, get_conversation_history, translate_runes]
TOOL_MAP = {tool.name: tool for tool in TOOLS}
CONTEXT_TOOLS = {'search_campaign', 'get_file_content'}
SESSION_TOOLS = {'get_conversation_history'}
# Tools that should return results directly without creative model processing
PASSTHROUGH_TOOLS = {'translate_runes'}


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
# System Prompt Builders (Split for each model)
# =============================================================================

def build_planning_prompt(campaign: str, campaign_context: str, recent_sessions: str) -> str:
    """Build focused system prompt for the planning/tool model (Nova).
    
    This model ONLY decides which tools to call - it never generates user-facing responses.
    """
    return f"""You are **D&D Buddy**, a D&D 5e campaign assistant for **{campaign}**.
Your job: **decide which tools to call** to retrieve the information needed to answer the user's question.

---

## CAMPAIGN CONTEXT

{campaign_context}

{recent_sessions}

This is the established canon. If something is missing here or in search results, it does not exist yet.

---

## YOUR TOOLS

1. **search_campaign**
   Semantic search across NPCs, monsters, sessions, lore, organizations, homebrew.
   Returns relevant snippets with filenames.

2. **get_file_content**
   Retrieves full file text.
   Use only if:
   - User explicitly asks for "everything" from a file
   - Same file appears in 3+ search results
   - Key section is clearly missing from snippets

3. **roll_dice**
   D&D dice notation (e.g. `1d20+5`, `2d6`).
   Use when user asks for a roll or when in-world roll is clearly implied.

4. **get_conversation_history**
   Retrieve earlier messages when user references prior discussion.
   Specify only the number of messages needed (typically 2–6).

5. **translate_runes**
   Translate text to/from The Architects' ancient rune cipher.
   Use when user asks to encode a message in runes, decode rune inscriptions,
   or create riddles/puzzles using the ancient script.

---

## TOOL-SELECTION RULES

**Always start with search_campaign for campaign info.**

For most questions:
- Run 1–3 focused searches with different keywords (names, places, events, topics, mechanics).
- If results are sparse, try a second round with broader or related terms.
- Stop after 2 rounds unless user explicitly asks for more.

For creative/integration questions (e.g., "How do Warforged fit?"):
- Search: the concept, related factions, magic/tech, relevant sessions.
- Combine results into a coherent answer that respects existing tone and lore.

For multi-part questions:
- Run separate searches for each distinct aspect.
- Aggregate results; avoid contradicting established facts.

**When NOT to search:**
- Simple dice rolls → call `roll_dice` directly.
- User references prior conversation → call `get_conversation_history` first.
- User asks for full file content → call `get_file_content` (if filename is known).

**Stop calling tools when:**
- You have sufficient context to answer the question.
- Search returns empty/minimal results twice in a row.
- You've completed 3 tool calls (the other model will synthesize your results).

---

## HANDLING MISSING INFO

If search returns nothing or very little:
- Note what is undefined.
- Suggest 1–2 concrete options the GM/players could adopt.
- Do not invent canonical facts.

---

## OUTPUT

Your output is **only tool calls**.
Another model will generate the final response to the user based on your tool results."""


def build_creative_prompt(campaign: str, campaign_context: str, recent_sessions: str) -> str:
    """Build system prompt for the creative/response model.
    
    This model ONLY generates the final user-facing response using gathered context.
    It never calls tools.
    """
    return f"""You are **D&D Buddy**, an expert D&D 5e campaign assistant for **{campaign}**.
Your job: **generate a concise, campaign-accurate answer** based on the context and tool results provided.

---

## CAMPAIGN CONTEXT

{campaign_context}

{recent_sessions}

This is the established canon. Tool results supplement this.

---

## GENERATION PRINCIPLES

**1. Be campaign-specific**
- Fit this setting's tone, genre, world logic, and established NPCs/factions/history.
- Avoid generic D&D tropes unless explicitly supported by context.

**2. Ground in retrieved context**
- Base answers on the campaign context, recent sessions, and tool results.
- Synthesize from multiple snippets rather than quoting a single file.
- If context is missing, clearly state what is unknown and suggest next steps (e.g., "ask the GM," "define X").

**3. Handle uncertainty explicitly**
- Do not invent people, places, factions, timelines, or rules.
- You may propose **suggested ideas** (label them clearly), but distinguish them from canon.

---

## RESPONSE FORMAT

**Length**: 100–300 words by default (up to 500 only if user explicitly asks).

**Structure**:
- Use **bold** for names, places, factions, items.
- Use `code` for D&D mechanics, dice notation, stats.
- Use bullets for lists instead of long paragraphs.
- Use `###` section headers to organize when helpful.

**For characters/places/items**:
- **Name / Type:** one-sentence summary.
- 1–2 key campaign-specific details.
- Role, relationships, or hooks.

**Do not**:
- Add filler or explain your process.
- Say "the file says…" unless clarifying the source.
- Write long narrative walls when bullets/sections work.

---

## CITATIONS

- Cite tool results succinctly: `(source: sessions/session12.md)` or `(source: lore/world_history.md)`.
- Cite when presenting specific facts, not general synthesis.
- Prefer synthesizing over direct quotations.

---

## DICE ROLLS

If tool results include a dice roll:
- State who/what rolled and the result.
- Highlight crits:
  - `20` → "🎉 Critical!"
  - `1` → "💀 Critical failure!"

---

## RULES

- Never invent canonical facts; only build from:
  - Campaign context
  - Recent sessions
  - Tool outputs
- Combine multiple results into a single, non-redundant answer.
- Call out missing/conflicting info and propose how to resolve it.
- You may suggest flavorful hooks or scenes, but label them as **suggestions**, not established facts."""


# =============================================================================
# Agent Graph Builder
# =============================================================================

class AgentGraphBuilder:
    """Builds the LangGraph agent with tool execution capability."""
    
    def __init__(self, user_id: str, campaign: str, session_id: str,
                 stream_callback: Optional[Callable] = None,
                 campaign_context: str = "", recent_sessions: str = ""):
        self.user_id = user_id
        self.campaign = campaign
        self.session_id = session_id
        self.stream_callback = stream_callback
        self.tools_executed: List[str] = []
        
        # Store context for creative prompt
        self.campaign_context = campaign_context
        self.recent_sessions = recent_sessions
        
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
    
    def generate_final_response(self, user_message: str, history_messages: List, tool_results: List) -> str:
        """Generate final creative response using expensive model with its own prompt."""
        logger.info("Generating final response with creative model (streaming)")
        
        # Build messages with creative-specific system prompt
        creative_system = SystemMessage(content=build_creative_prompt(
            self.campaign, self.campaign_context, self.recent_sessions
        ))
        
        final_messages = [creative_system] + history_messages + [HumanMessage(content=user_message)]
        
        if tool_results:
            tool_context = "\n\n".join([msg.content for msg in tool_results])
            final_messages.append(HumanMessage(
                content=f"Here is the information gathered from the campaign:\n\n{tool_context}\n\nProvide a helpful response."
            ))
        
        response = self.creative_llm.invoke(final_messages)
        return response.content if hasattr(response, 'content') else str(response)
    
    def get_tools_summary(self) -> str:
        """Get formatted summary of tools used as a simple italic bullet list."""
        if not self.tools_executed:
            return ""
        bullets = "\n".join(f"- _{tool}_" for tool in self.tools_executed)
        return f"\n\n---\n{bullets}"


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
        
        # Build agent with context for later creative response
        agent_builder = AgentGraphBuilder(
            user_id, campaign, session_id, stream_callback,
            campaign_context=campaign_context,
            recent_sessions=recent_sessions
        )
        agent = agent_builder.build()
        
        # Build messages for planning model (tool-focused prompt with context)
        planning_system = SystemMessage(content=build_planning_prompt(campaign, campaign_context, recent_sessions))
        planning_messages = [planning_system] + history_messages + [HumanMessage(content=user_message)]
        
        logger.info(f"Invoking planning agent with {len(planning_messages)} messages")
        
        # Phase 1: Tool planning and execution (cheap model, no streaming)
        result = agent.invoke({"messages": planning_messages})
        
        # Collect tool results for context
        tool_results = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
        
        # Check if only passthrough tools were used (e.g., translate_runes)
        tools_used = set(agent_builder.tools_executed)
        passthrough_only = tools_used and all(
            any(pt in tool for pt in PASSTHROUGH_TOOLS) 
            for tool in tools_used
        )
        
        if passthrough_only and tool_results:
            # Return tool results directly without creative processing
            logger.info("Passthrough mode: returning tool results directly")
            response_text = tool_results[-1].content
            if stream_callback:
                stream_callback([{"type": "text", "text": response_text, "index": 0}])
        else:
            # Phase 2: Final response generation (expensive model, with streaming, own prompt)
            response_text = agent_builder.generate_final_response(user_message, history_messages, tool_results)

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
