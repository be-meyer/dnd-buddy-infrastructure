"""
D&D Buddy Agent - AgentCore Runtime implementation with LangGraph.
Tools: search_campaign, roll_dice, list_campaign_files, get_file_content
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
from tools import search_campaign, roll_dice, list_campaign_files, get_file_content, get_conversation_history
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
        model_kwargs={"temperature": 0.3, "max_tokens": 400}
    )
    
    # Bind tools to LLM
    tools = [search_campaign, roll_dice, list_campaign_files, get_file_content, get_conversation_history]
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
        'list_campaign_files': list_campaign_files,
        'get_file_content': get_file_content,
        'get_conversation_history': get_conversation_history
    }
    
    # Tools that need user context
    context_tools = {'search_campaign', 'list_campaign_files', 'get_file_content'}
    
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
        
        # Create agent with user context and streaming callback
        agent = create_agent(user_id, campaign, session_id, stream_callback=stream_callback)
        
        # System message with instructions
        system_message = SystemMessage(content="""You are D&D Buddy, a D&D 5e campaign assistant with access to the user's campaign database.

**RESPONSE LENGTH GUIDELINES:**
- Keep responses concise and focused
- Typical response: 100-300 words (unless user asks for details)
- Maximum: 500 words
- Break long information into sections with ### headers
- For entity summaries: name + quick description + 1-2 key details only
- For complex queries: organize into bullet points, not paragraphs

**Avoid:**
- Unnecessary elaboration or filler
- Repeating information already shared
- Long narrative descriptions (save for specific requests)
- Multiple paragraphs when bullets work better

## TOOLS (5 available)
1. **search_campaign**: Semantic search (RAG) - USE FIRST for questions about NPCs, locations, monsters, sessions, lore, species, players
2. **list_campaign_files**: Show available files by category
3. **get_file_content**: Retrieve complete file (after finding filename via search or list)
4. **roll_dice**: D&D dice notation (e.g., "1d20+5", "2d6")
5. **get_conversation_history**: Retrieve earlier messages from this conversation (use when user references past discussion)

## SEARCH STRATEGY
**Query optimization**: Use specific names/terms from user's question
**Categories**: npcs, monsters, sessions, lore, organizations, species, players (or None for broad search)
**Multi-step**: Complex questions = multiple searches
**Fallback**: No results → try broader query or list_campaign_files

**Search Result Count**:
- Specific entity question ("Who is X?"): top_k=3
- Broad question ("What monsters?"): top_k=5
- Multi-part question: top_k=5 per search

Examples:
- "Who is Baldric?" → search_campaign("Baldric", category="npcs")
- "Tell me about elves" → search_campaign("elves", category="species")
- "Who are the players?" → search_campaign("players party", category="players")
- "Last session recap?" → search_campaign("last session recent", category="sessions")
- "Dragon we fought + treasure?" → search("dragon fight", category="monsters") + search("dragon treasure", category="sessions")

## SEARCH → FILE WORKFLOW
- PREFER search_campaign over get_file_content (search is more targeted)
- Only use get_file_content when:
  1. User explicitly asks for "everything" or "complete file"
  2. Search results reference same file multiple times
  3. You need specific section search didn't return

## CONVERSATION CONTEXT
- You only see the last exchange by default (last 2 messages: user + your response)
- If user references earlier conversation ("as we discussed", "you mentioned", "earlier you said"), use **get_conversation_history(message_count=N)**
- Only specify message_count parameter: 10 = last 5 exchanges, 20 = last 10 exchanges
- DO NOT specify session_id - it's automatically provided

## DICE ROLLS
Interpret natural language:
- "roll perception" → roll_dice("1d20") or "1d20+modifier"
- "roll longsword damage" → roll_dice("1d8+modifier")

Present results with context:
- Keep tool output intact
- Add character context: "Your ranger rolled..."
- Note crits: Natural 20 = "🎉 Critical!", Natural 1 = "💀 Critical failure!"

## FORMATTING
**Bold** for names/places, `code` for mechanics/stats, bullets for lists, > for quotes, ### for sections

## RULES
- Never invent info - only use search results
- Cite sources: "According to session 12 notes..."
- Combine multiple results into coherent narrative
- If no results: acknowledge gap, suggest alternatives""")

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
