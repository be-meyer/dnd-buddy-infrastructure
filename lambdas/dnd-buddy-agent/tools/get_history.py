"""
Conversation history retrieval tool with caching.
"""
import logging
import os
from typing import Optional, List
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import DynamoDBChatMessageHistory

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Environment variables
CHAT_HISTORY_TABLE_NAME = os.environ.get('CHAT_HISTORY_TABLE_NAME', 'dnd-buddy-chat-history')

# TTL configuration: 7 days in seconds
TTL_SECONDS = 7 * 24 * 60 * 60  # 604,800 seconds

# Global cache for conversation history (per Lambda execution)
_history_cache = {}


def clean_history_messages(messages: List) -> List:
    """Strip unnecessary metadata from history messages."""
    cleaned = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            cleaned.append(HumanMessage(content=msg.content))
        elif isinstance(msg, AIMessage):
            # Skip tool calls - only keep final responses
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                continue
            cleaned.append(AIMessage(content=msg.content))
    return cleaned


def load_history(session_id: str) -> List:
    """
    Load conversation history from DynamoDB with caching.
    
    Args:
        session_id: Session ID
        
    Returns:
        List of cleaned messages
    """
    # Check cache first
    if session_id in _history_cache:
        logger.info(f"Using cached history for session {session_id}")
        return _history_cache[session_id]
    
    logger.info(f"Loading history from DynamoDB for session {session_id}")
    
    try:
        # Initialize chat history
        chat_history = DynamoDBChatMessageHistory(
            table_name=CHAT_HISTORY_TABLE_NAME,
            session_id=session_id,
            primary_key_name='SessionId'
        )
        
        # Get all messages
        all_messages = chat_history.messages
        all_messages = clean_history_messages(all_messages)
        
        # Cache the result
        _history_cache[session_id] = all_messages
        
        logger.info(f"Loaded and cached {len(all_messages)} messages")
        return all_messages
        
    except Exception as e:
        logger.error(f"Failed to load conversation history: {e}", exc_info=True)
        return []


def get_history_messages(session_id: str, message_count: int = 2) -> List:
    """
    Get raw message objects from history (for internal use by agent).
    
    Args:
        session_id: Session ID
        message_count: Number of messages to retrieve (default: 2 for last exchange)
        
    Returns:
        List of message objects
    """
    all_messages = load_history(session_id)
    
    if not all_messages:
        return []
    
    # Get the last N messages
    if len(all_messages) >= message_count:
        return all_messages[-message_count:]
    else:
        return all_messages


@tool
def get_conversation_history(session_id: str, message_count: int = 10) -> str:
    """
    Retrieve conversation history from cache (loaded once per Lambda execution).
    
    Use this tool when:
    - User references past conversation ("as we discussed", "you mentioned", "earlier you said")
    - You need more context beyond the last exchange
    
    Args:
        session_id: Session ID (automatically injected)
        message_count: Number of messages to retrieve (default: 10 = ~5 exchanges)
                      Each exchange = 2 messages (user + AI)
                      Examples: 2 = last exchange, 10 = last 5 exchanges, 20 = last 10 exchanges
        
    Returns:
        Formatted conversation history
    """
    logger.info(f"get_conversation_history: session_id={session_id}, message_count={message_count}")
    
    # Load history (uses cache if available)
    all_messages = load_history(session_id)
    
    if not all_messages:
        return "No conversation history found for this session."
    
    # Get the last N messages
    recent_messages = all_messages[-message_count:] if len(all_messages) > message_count else all_messages
    
    # Format messages for display
    formatted = []
    for i, msg in enumerate(recent_messages, 1):
        msg_type = "User" if msg.type == "human" else "Assistant"
        content = msg.content
        
        # Handle content as string or list
        if isinstance(content, list):
            # Extract text from content blocks
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get('type') == 'text':
                    text_parts.append(block.get('text', ''))
            content_str = ' '.join(text_parts)
        else:
            content_str = str(content)
        
        formatted.append(f"{i}. {msg_type}: {content_str}")
    
    result = f"📜 Last {len(recent_messages)} messages from conversation:\n\n"
    result += "\n\n".join(formatted)
    
    logger.info(f"Retrieved {len(recent_messages)} messages from cached history")
    return result


def save_messages(session_id: str, user_message: str, ai_message: str) -> None:
    """
    Save new messages to DynamoDB chat history.
    
    Args:
        session_id: Session ID
        user_message: User message content
        ai_message: AI response content
    """
    try:
        logger.info(f"Saving messages to chat history for session {session_id}")
        
        # Initialize chat history
        chat_history = DynamoDBChatMessageHistory(
            table_name=CHAT_HISTORY_TABLE_NAME,
            session_id=session_id,
            primary_key_name='SessionId',
            ttl=TTL_SECONDS,
            ttl_key_name='expireAt'
        )
        
        # Save the user message and AI response
        chat_history.add_messages([
            HumanMessage(content=user_message),
            AIMessage(content=ai_message)
        ])
        
        logger.info(f"Saved 2 messages to chat history for session {session_id}")
        
        # Invalidate cache so next invocation loads fresh data
        if session_id in _history_cache:
            del _history_cache[session_id]
            logger.info(f"Invalidated cache for session {session_id}")
            
    except Exception as e:
        logger.error(f"Failed to save chat history for session {session_id}: {e}", exc_info=True)
        # Don't raise - user still gets their response
