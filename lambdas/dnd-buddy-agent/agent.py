"""
D&D Buddy Agent - AgentCore Runtime implementation with LangGraph.
Tools: search_campaign, roll_dice, get_file_content
"""
import os
import logging
import time
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
from tools.get_history import get_history_messages

# Environment variables
BEDROCK_MODEL_ID = os.environ.get('BEDROCK_MODEL_ID', 'eu.amazon.nova-micro-v1:0')
BEDROCK_MODEL_ID_TOOL = os.environ.get('BEDROCK_MODEL_ID_TOOL', 'eu.amazon.nova-micro-v1:0')
MAX_ITERATIONS = int(os.environ.get('MAX_AGENT_ITERATIONS', '3'))

class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming LLM responses."""
    
    def __init__(self, stream_callback: Callable[[Any], None]):
        self.stream_callback = stream_callback
        self.current_response = ""
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called when a new token is generated.
        
        Bedrock Converse API may send structured content blocks or plain strings.
        We need to handle both formats and pass them through to maintain consistency.
        """
        
        if token:
            # Check if token is already a structured format (from Bedrock Converse)
            # If it's a dict or list, pass it through as-is
            # Otherwise, wrap plain string in the expected format
            if isinstance(token, (dict, list)):
                self.stream_callback(token)
            else:
                # Plain string token - wrap in Bedrock format for consistency
                self.stream_callback([{"type": "text", "text": token, "index": 0}])
            
            # Track plain text for response building
            if isinstance(token, str):
                self.current_response += token
            elif isinstance(token, list):
                # Extract text from content blocks
                for block in token:
                    if isinstance(block, dict) and block.get('type') == 'text':
                        self.current_response += block.get('text', '')


def create_agent(user_id: str, campaign: str, session_id: str, stream_callback: Optional[Callable[[str], None]] = None):
    """
    Create a LangGraph agent with multi-step tool execution capability.
    
    Args:
        user_id: User ID for filtering data access
        campaign: Campaign name for filtering data access
        input_data: Full input data including sessionId
        stream_callback: Optional callback for streaming LLM responses
    """
    # Create streaming handler if callback provided
    streaming_handler = StreamingCallbackHandler(stream_callback) if stream_callback else None
    
    # Initialize Bedrock LLM with streaming support (only for final response)
    llm = ChatBedrock(
        model_id=BEDROCK_MODEL_ID,
        model_kwargs={"temperature": 0.6, "max_tokens": 1500},
        streaming=True,
        callbacks=[streaming_handler] if streaming_handler else []
    )

    # Tool planning LLM without streaming (we don't want to stream tool calls)
    llm_tool_planning = ChatBedrock(
        model_id=BEDROCK_MODEL_ID_TOOL,
        model_kwargs={
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1,
            "max_tokens": 400
        }
    )
    
    # Bind tools to LLM
    tools = [search_campaign, roll_dice, get_file_content, get_conversation_history]
    llm_with_tools = llm.bind_tools(tools)
    llm_tool_planning_tools = llm_tool_planning.bind_tools(tools)
    
    def agent_node(state: MessagesState):
        """Agent reasoning node with iteration limit."""
        messages = state["messages"]
        
        iteration_count = sum(
            1 for msg in messages 
            if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls
        )
        
        logger.info(f"Agent reasoning - iteration {iteration_count + 1}")
        
        # Emergency brake
        if iteration_count >= MAX_ITERATIONS:
            logger.warning(f"Max iterations ({MAX_ITERATIONS}) reached, forcing final response")
            final_prompt = messages + [
                HumanMessage(content="Provide your final answer based on the information gathered. Do not call any more tools.")
            ]
            response = llm_with_tools.invoke(final_prompt)
            return {"messages": [response]}
        
        # Use cheap model for all planning
        planning_response = llm_tool_planning_tools.invoke(messages)
        
        # Check if this response has NO tool calls (agent finished)
        has_tool_calls = hasattr(planning_response, 'tool_calls') and planning_response.tool_calls
        
        if not has_tool_calls:
            # Agent is done! Regenerate final response with main model for quality
            # IMPORTANT: Don't add planning_response to messages - it may contain thinking text
            logger.info("Agent finished planning, generating final response with main model (streaming enabled)")
            logger.info(f"Planning response content length: {len(str(planning_response.content)) if hasattr(planning_response, 'content') else 0}")
            
            # Generate clean final response with main model (streaming always enabled)
            final_response = llm_with_tools.invoke(messages)
            logger.info(f"Final response generated, content length: {len(str(final_response.content)) if hasattr(final_response, 'content') else 0}")
            return {"messages": [final_response]}
        
        # Agent wants to call tools - return the planning response with tool calls
        logger.info(f"Agent calling {len(planning_response.tool_calls)} tool(s)")
        return {"messages": [planning_response]}

    
    # Tool mapping
    tool_map = {
        'search_campaign': search_campaign,
        'roll_dice': roll_dice,
        'get_file_content': get_file_content,
        'get_conversation_history': get_conversation_history
    }
    
    # Tools that need user context
    context_tools = {'search_campaign', 'get_file_content'}
    
    # Tools that need session_id
    session_tools = {'get_conversation_history'}
    
    # Define tool execution node with user context injection
    def tool_node(state: MessagesState):
        """Execute tools with automatic user context injection."""
        messages = state["messages"]
        last_message = messages[-1]
        
        tool_calls = last_message.tool_calls if hasattr(last_message, 'tool_calls') else []
        tool_messages = []
        
        logger.info(f"Executing {len(tool_calls)} tool call(s)")
        
        for tool_call in tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args'].copy()
            
            logger.info(f"  - {tool_name}: {tool_args}")
            
            # Inject user_id and campaign for tools that need them
            if tool_name in context_tools:
                tool_args['user_id'] = user_id
                tool_args['campaign'] = campaign
            
            # Inject session_id for tools that need it
            if tool_name in session_tools:
                tool_args['session_id'] = session_id
            
            # Execute tool
            tool_func = tool_map.get(tool_name)
            if tool_func:
                result = tool_func.invoke(tool_args)
                logger.info(f"  ✓ {tool_name} returned {len(str(result))} chars")
            else:
                result = f"Unknown tool: {tool_name}"
                logger.error(f"  ✗ Unknown tool: {tool_name}")
            
            tool_messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_call['id'])
            )
        
        return {"messages": tool_messages}
    
    # Define routing logic
    def should_continue(state: MessagesState):
        """
        Determine if agent should continue to tools or end.
        
        Returns:
            - "tools" if agent wants to call tools
            - END if agent is done reasoning
        """
        last_message = state["messages"][-1]
        
        # If agent wants to use tools, route to tools node
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        
        # Otherwise, agent is done
        logger.info("Agent finished reasoning")
        return END
    
    # Build graph
    workflow = StateGraph(MessagesState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    
    # Add edges - THIS IS THE CRITICAL PART
    workflow.set_entry_point("agent")
    
    # After agent decides, either go to tools or end
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END}
    )
    
    # After tools execute, ALWAYS go back to agent for re-evaluation
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()


def main(input_data: Dict[str, Any], stream_callback=None) -> Dict[str, Any]:
    """
    Main entry point for AgentCore Runtime.
    This function is called by the AgentCore Runtime container.
    
    Args:
        input_data: Dictionary containing userId, campaign, prompt, sessionId
        stream_callback: Optional callback function to stream response chunks
    """
    logger.info("=" * 80)
    logger.info("Agent invocation started")
    logger.info(f"Input data: {input_data}")
    logger.info(f"Streaming enabled: {stream_callback is not None}")
    
    # Extract parameters from input
    user_id = input_data.get('userId')
    campaign = input_data.get('campaign')
    user_message = input_data.get('prompt')
    session_id = input_data.get('sessionId', f"{user_id}-{campaign}-default")
    
    logger.info(f"User: {user_id}, Campaign: {campaign}, Session: {session_id}")
    logger.info(f"Message: {user_message}")
    
    if not all([user_id, campaign, user_message]):
        logger.error("Missing required parameters")
        return {'error': 'Missing required parameters: userId, campaign, prompt'}
    
    # Validate that session_id starts with user_id to prevent session hijacking
    if not session_id.startswith(f"{user_id}-"):
        logger.error(f"Session validation failed: session_id '{session_id}' does not belong to user '{user_id}'")
        return {'error': 'Invalid session: session does not belong to the authenticated user'}
    
    try:
        # Load last exchange (2 messages) using get_history_messages
        # This loads full history once and caches it for subsequent tool calls
        history_messages = get_history_messages(session_id, message_count=2)
        logger.info(f"Loaded {len(history_messages)} messages from last exchange")
        
        # Load campaign context dynamically
        logger.info("Loading campaign context...")
        try:
            campaign_context_results = search_campaign.invoke({
                'query': 'world setting themes tone style history magic system background lore',
                'top_k': 3,
                'user_id': user_id,
                'campaign': campaign
            })
            
            # Format campaign context
            if campaign_context_results and "No relevant information found" not in campaign_context_results:
                campaign_context = f"\n{campaign_context_results}\n"
                logger.info(f"Loaded campaign context: {len(campaign_context)} chars")
            else:
                campaign_context = "\n_No campaign background information available yet._\n"
                logger.info("No campaign context found")
        except Exception as e:
            logger.warning(f"Failed to load campaign context: {e}")
            campaign_context = ""
        
        # Load recent session context
        try:
            recent_sessions_results = search_campaign.invoke({
                'query': 'recent session last game latest adventure current quest',
                'top_k': 2,
                'user_id': user_id,
                'campaign': campaign
            })
            
            if recent_sessions_results and "No relevant information found" not in recent_sessions_results:
                recent_sessions = f"\n**RECENT SESSIONS:**\n{recent_sessions_results}\n"
                logger.info(f"Loaded recent sessions: {len(recent_sessions)} chars")
            else:
                recent_sessions = ""
                logger.info("No recent sessions found")
        except Exception as e:
            logger.warning(f"Failed to load recent sessions: {e}")
            recent_sessions = ""
        
        # Create agent with user context and streaming callback
        agent = create_agent(user_id, campaign, session_id, stream_callback=stream_callback)
        
        # Build dynamic system message with campaign context
        system_content = f"""
You are **D&D Buddy**, an expert D&D 5e campaign assistant for the campaign: **{campaign}**.

---

## CAMPAIGN CONTEXT

{campaign_context}

{recent_sessions}

---

## YOUR CAPABILITIES

- **Access**: You see the user's entire campaign database—NPCs, monsters, session logs, lore, organizations, homebrew, and more.
- **System awareness**: You understand the campaign’s established tone, themes, magic system, current events, and style based on the context above.

---

## GENERATION PRINCIPLES

**Be campaign-specific:**  
- All responses, world integrations, and creative ideas must _fit this setting_ by leveraging campaign files and recent events.
- Use the exact tone, genre, and world logic you find in the campaign context.
- Never revert to "generic D&D" tropes. Always relate to the people, factions, magic, history, and style defined above.

**Prioritize facts and context:**  
- Synthesize from search results rather than making assumptions.
- If required context is missing from {campaign}, _say so clearly and suggest possible actions (e.g. ask GM, search another term)_.  
- For creative/world-integration queries (“how does X fit in?”), cross-reference NPCs, factions, lore, session events, and rules.

---

## RESPONSE FORMAT

- **Length**: Target 100–300 words (500 max if user requests detail).
- Use **bold** for names and places, `code` for D&D mechanics or dice, bullets for lists, ### for sections.
- For answers about characters, places, items: use this structure—**Name/Type:** one-sentence description, 1-2 key details, role/connections.
- Use bullets for lists (never overlong prose).

**DO NOT:**
- Add unnecessary filler or repetition.
- Summarize “the file says…” unless clarifying context.
- Use long narrative paragraphs when concise bullets/sections work.

---

## TOOLS AVAILABLE

1. **search_campaign**: Semantic search—use _first_ for all campaign info needs. Returns relevant snippets from ALL files (NPCs, monsters, sessions, lore, organizations, custom content).
2. **get_file_content**: Full file text—use only if user requests “everything”, same file is referenced in multiple search results, or a section is missing.
3. **roll_dice**: D&D dice syntax (e.g., `1d20+5`, `2d6`).
4. **get_conversation_history**: Retrieve _earlier_ messages if referenced (user says “as we discussed”). Only specify number of messages needed.

---

## SEARCH STRATEGY

- Always begin with **search_campaign** (no categories; index is unified).
- Use focused queries: character name, monster, topic, place, event, or rules keyword.
- For creative/integration questions (“How does Warforged fit into this Axiom?”):
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

- Interpret queries naturally (“roll perception” → `roll_dice("1d20+mod")`).
- Always present who/what is rolling. Call out crits: 20 = “🎉 Critical!” and 1 = “💀 Critical failure!”

---

## RULES OF GENERATION

- **NEVER invent info**—only synthesize/creatively build from search results/context above.
- **Combine** multiple results for each query into a coherent, non-redundant answer.
- If uncertain/absent: _clearly indicate the gap and suggest user/GM action_ rather than “filling in” with assumptions.
- _Always_ cite sources (filename, section name, or session number) for campaign specifics, e.g., “(source: sessions/session12.md)”.

---
"""
        
        system_message = SystemMessage(content=system_content)

        # Build messages: system message + history + new user message
        current_user_message = HumanMessage(content=user_message)
        messages = [system_message] + history_messages + [current_user_message]
        
        logger.info(f"Invoking agent with {len(messages)} total messages ({len(history_messages)} from history)")
        logger.info(f"Streaming enabled: {stream_callback is not None}")
        
        # Run agent (streaming happens via callbacks if enabled)
        result = agent.invoke({"messages": messages})
        
        # Track which tools were used with their parameters
        tools_used = []
        for message in result["messages"]:
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call['name']
                    tool_args = tool_call.get('args', {})
                    # Format args, excluding user_id and campaign (injected automatically)
                    display_args = {k: v for k, v in tool_args.items() if k not in ['user_id', 'campaign']}
                    if display_args:
                        args_str = ', '.join(f"{k}={repr(v)}" for k, v in display_args.items())
                        tools_used.append(f"{tool_name}({args_str})")
                    else:
                        tools_used.append(f"{tool_name}()")
        
        # Extract final response
        final_message = result["messages"][-1]
        response_text = final_message.content if hasattr(final_message, 'content') else str(final_message)
        
        # Append tool usage information if any tools were used
        if tools_used:
            tool_calls_str = ', '.join(tools_used)
            tool_info = f"\n\n---\n_Tools used: {tool_calls_str}_"
            response_text += tool_info
            # Stream the tool info if streaming is enabled
            # Use the same format as Bedrock streaming for consistency
            if stream_callback:
                stream_callback([{"type": "text", "text": tool_info, "index": 0}])
        
        # Save messages to chat history using the tool's save function
        from tools.get_history import save_messages
        save_messages(
            session_id=session_id,
            user_message=current_user_message.content,
            ai_message=final_message.content if hasattr(final_message, 'content') else str(final_message)
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
