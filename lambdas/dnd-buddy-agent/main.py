"""
Lambda handler for D&D Buddy agent.
Handles WebSocket API Gateway requests for real-time agent communication.
"""
import json
import logging
import os
import boto3
from agent import main as agent_main

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize API Gateway Management API client for WebSocket
WEBSOCKET_API_ENDPOINT = os.environ.get('WEBSOCKET_API_ENDPOINT')
if not WEBSOCKET_API_ENDPOINT:
    raise ValueError("WEBSOCKET_API_ENDPOINT environment variable is required")

apigw_management = boto3.client(
    'apigatewaymanagementapi',
    endpoint_url=WEBSOCKET_API_ENDPOINT
)


def send_websocket_message(connection_id, message_type, content):
    """Send a message to a WebSocket client."""
    try:
        message = json.dumps({
            'type': message_type,
            'content': content
        })
        
        apigw_management.post_to_connection(
            ConnectionId=connection_id,
            Data=message.encode('utf-8')
        )
        
        logger.info(f"WebSocket message sent successfully")
        return True
        
    except apigw_management.exceptions.GoneException:
        logger.warning(f"Connection {connection_id} is gone (GoneException)")
        return False
    except apigw_management.exceptions.ForbiddenException:
        logger.warning(f"Connection {connection_id} forbidden (ForbiddenException)")
        return False
    except Exception as e:
        logger.error(f"Failed to send WebSocket message: {type(e).__name__}: {e}", exc_info=True)
        return False


def lambda_handler(event, context):
    """
    Main Lambda handler for D&D Buddy agent.
    Handles WebSocket API Gateway requests for real-time agent communication.
    """
    logger.info(f"Agent request: {json.dumps(event)}")
    
    connection_id = event.get('requestContext', {}).get('connectionId')
    
    if not connection_id:
        logger.error("No connection ID in WebSocket event")
        return {'statusCode': 400}
    
    try:
        # Parse message body
        body = json.loads(event.get('body', '{}'))
        
        # Extract userId from authorizer context
        user_id = event.get('requestContext', {}).get('authorizer', {}).get('userId')
        
        # Extract parameters
        action = body.get('action')
        campaign = body.get('campaign')
        message = body.get('message')
        session_id = body.get('sessionId')
        
        logger.info(f"WebSocket - Action: {action}, User: {user_id}, Campaign: {campaign}")
        
        if action != 'chat':
            send_websocket_message(connection_id, 'error', f'Unknown action: {action}')
            return {'statusCode': 400}
        
        if not all([user_id, campaign, message, session_id]):
            send_websocket_message(connection_id, 'error', 'Missing required fields')
            return {'statusCode': 400}
        
        # Prepare input for agent
        input_data = {
            'userId': user_id,
            'campaign': campaign,
            'prompt': message,
            'sessionId': session_id
        }
        
        # Create streaming callback
        def stream_callback(chunk):
            """Send streaming chunks to WebSocket client.
            
            Chunk can be:
            - A list of content blocks: [{"type": "text", "text": "...", "index": 0}]
            - A plain string (for backward compatibility)
            """
            success = send_websocket_message(connection_id, 'chunk', chunk)
            if not success:
                logger.warning(f"Failed to send chunk to connection {connection_id}")
            return success
        
        # Invoke agent with streaming callback
        logger.info("Invoking agent...")
        result = agent_main(input_data, stream_callback=stream_callback)
        logger.info(f"Agent completed with result keys: {result.keys()}")
        
        # Check for error
        if 'error' in result:
            logger.error(f"Agent error: {result['error']}")
            send_websocket_message(connection_id, 'error', result['error'])
            return {'statusCode': 500}
        
        # Send completion signal
        logger.info("Sending completion signal")
        completion_sent = send_websocket_message(connection_id, 'complete', '')
        logger.info(f"Completion signal sent: {completion_sent}")
        
        return {'statusCode': 200}
        
    except Exception as e:
        logger.error(f"WebSocket handler error: {str(e)}", exc_info=True)
        send_websocket_message(connection_id, 'error', str(e))
        return {'statusCode': 500}



