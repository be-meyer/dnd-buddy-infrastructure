"""
Lambda handler for session management.
Provides endpoints to list user sessions and retrieve session history.
"""
import json
import logging
import os
import boto3
from typing import Dict, Any, List
from datetime import datetime
from decimal import Decimal

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize DynamoDB client
dynamodb = boto3.resource('dynamodb')
CHAT_HISTORY_TABLE_NAME = os.environ.get('CHAT_HISTORY_TABLE_NAME', 'dnd-buddy-chat-history')


# Custom JSON encoder to handle Decimal types from DynamoDB
class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            # Convert Decimal to int if it's a whole number, otherwise to float
            if obj % 1 == 0:
                return int(obj)
            else:
                return float(obj)
        return super(DecimalEncoder, self).default(obj)


def lambda_handler(event, context):
    """
    Lambda handler for session management.
    
    Routes:
    - GET /sessions - List all sessions for the authenticated user
    - GET /sessions/{sessionId} - Get history for a specific session
    """
    logger.info(f"Session request: {json.dumps(event)}")
    
    cors_headers = {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type,Authorization',
        'Access-Control-Allow-Methods': 'GET,OPTIONS'
    }
    
    try:
        # Extract userId from Cognito authorizer claims
        user_id = event.get('requestContext', {}).get('authorizer', {}).get('claims', {}).get('cognito:username')
        
        if not user_id:
            return {
                'statusCode': 401,
                'headers': cors_headers,
                'body': json.dumps({'error': 'Unauthorized: No user ID found'})
            }
        
        # Determine the operation based on path parameters
        path_parameters = event.get('pathParameters') or {}
        session_id = path_parameters.get('sessionId')
        
        if session_id:
            # Get specific session history
            result = get_session_history(user_id, session_id)
        else:
            # List all sessions for user
            result = list_user_sessions(user_id)
        
        if 'error' in result:
            return {
                'statusCode': 400 if result.get('code') == 'validation_error' else 500,
                'headers': cors_headers,
                'body': json.dumps({'error': result['error']}, cls=DecimalEncoder)
            }
        
        return {
            'statusCode': 200,
            'headers': cors_headers,
            'body': json.dumps(result, cls=DecimalEncoder)
        }
        
    except Exception as e:
        logger.error(f"Lambda error: {str(e)}", exc_info=True)
        return {
            'statusCode': 500,
            'headers': cors_headers,
            'body': json.dumps({'error': str(e)}, cls=DecimalEncoder)
        }


def extract_preview(content: str, min_chars: int = 150, max_chars: int = 500) -> str:
    """
    Extract the first paragraph for preview, with min/max character limits.
    """
    if not content:
        return ''
    
    # Split on double newlines to get paragraphs
    paragraphs = content.split('\n\n')
    first_para = paragraphs[0].strip()
    
    # If first paragraph is too short and there are more, include more
    if len(first_para) < min_chars and len(paragraphs) > 1:
        preview = first_para
        for para in paragraphs[1:]:
            preview += '\n\n' + para.strip()
            if len(preview) >= min_chars:
                break
        first_para = preview
    
    # Truncate if too long
    if len(first_para) > max_chars:
        first_para = first_para[:max_chars].rsplit(' ', 1)[0] + '...'
    
    return first_para


def list_user_sessions(user_id: str) -> Dict[str, Any]:
    """
    List all sessions for a user, sorted by most recent first.
    
    Args:
        user_id: The Cognito username
        
    Returns:
        Dictionary with sessions list
    """
    try:
        table = dynamodb.Table(CHAT_HISTORY_TABLE_NAME)
        
        # Scan the table for sessions that start with user_id
        # Note: In production, consider using a GSI for better performance
        response = table.scan(
            FilterExpression='begins_with(SessionId, :user_prefix)',
            ExpressionAttributeValues={
                ':user_prefix': f"{user_id}-"
            }
        )
        
        sessions = []
        for item in response.get('Items', []):
            session_id = item.get('SessionId')
            history = item.get('History', [])
            
            # Extract timestamp from session_id (format: username-timestamp-random)
            try:
                parts = session_id.split('-')
                if len(parts) >= 2:
                    timestamp = int(parts[1])
                    last_updated = datetime.fromtimestamp(timestamp / 1000).isoformat()
                else:
                    last_updated = None
            except (ValueError, IndexError):
                last_updated = None
            
            # Get last agent message as preview (first paragraph, 150-500 chars)
            preview = None
            if history:
                # Find the last agent (ai) message for preview
                for msg in reversed(history):
                    if msg.get('type') == 'ai':
                        content = msg.get('data', {}).get('content', '')
                        
                        # Handle content as either string or list
                        if isinstance(content, list):
                            content_str = ' '.join(str(item) for item in content) if content else ''
                        else:
                            content_str = str(content)
                        
                        preview = extract_preview(content_str)
                        break
                
                # Fallback to last message if no agent message found
                if preview is None:
                    last_msg = history[-1]
                    content = last_msg.get('data', {}).get('content', '')
                    if isinstance(content, list):
                        content_str = ' '.join(str(item) for item in content) if content else ''
                    else:
                        content_str = str(content)
                    preview = extract_preview(content_str)
            
            sessions.append({
                'sessionId': session_id,
                'lastUpdated': last_updated,
                'messageCount': len(history),
                'preview': preview or 'No messages'
            })
        
        # Sort by timestamp (most recent first)
        sessions.sort(key=lambda x: x.get('lastUpdated') or '', reverse=True)
        
        logger.info(f"Found {len(sessions)} sessions for user {user_id}")
        
        return {
            'sessions': sessions,
            'count': len(sessions)
        }
        
    except Exception as e:
        logger.error(f"Failed to list sessions for user {user_id}: {e}", exc_info=True)
        return {'error': f"Failed to list sessions: {str(e)}"}


def get_session_history(user_id: str, session_id: str) -> Dict[str, Any]:
    """
    Get the full message history for a specific session.
    
    Args:
        user_id: The Cognito username
        session_id: The session ID to retrieve
        
    Returns:
        Dictionary with session history
    """
    try:
        # Validate that session belongs to user
        if not session_id.startswith(f"{user_id}-"):
            logger.warning(f"User {user_id} attempted to access session {session_id}")
            return {
                'error': 'Invalid session: session does not belong to the authenticated user',
                'code': 'validation_error'
            }
        
        table = dynamodb.Table(CHAT_HISTORY_TABLE_NAME)
        
        # Get the session item
        response = table.get_item(
            Key={'SessionId': session_id}
        )
        
        if 'Item' not in response:
            return {
                'error': 'Session not found',
                'code': 'validation_error'
            }
        
        item = response['Item']
        history = item.get('History', [])
        
        # Format messages for frontend consumption
        messages = []
        for msg in history:
            msg_type = msg.get('type')
            msg_data = msg.get('data', {})
            
            messages.append({
                'type': msg_type,
                'content': msg_data.get('content', ''),
                'timestamp': msg_data.get('timestamp'),
                'additionalKwargs': msg_data.get('additional_kwargs', {})
            })
        
        logger.info(f"Retrieved {len(messages)} messages for session {session_id}")
        
        return {
            'sessionId': session_id,
            'messages': messages,
            'messageCount': len(messages)
        }
        
    except Exception as e:
        logger.error(f"Failed to get session history for {session_id}: {e}", exc_info=True)
        return {'error': f"Failed to retrieve session history: {str(e)}"}
