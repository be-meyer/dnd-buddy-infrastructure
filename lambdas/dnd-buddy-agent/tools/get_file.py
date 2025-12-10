"""
Get full file content tool.
"""
import os
import logging
import boto3
from langchain_core.tools import tool

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Initialize client
s3_client = boto3.client('s3')

# Environment variables
CAMPAIGN_FILES_BUCKET = os.environ.get('CAMPAIGN_FILES_BUCKET')


@tool
def get_file_content(file_path: str, user_id: str = None, campaign: str = None) -> str:
    """
    Retrieve the complete, full content of a specific campaign file by its path.
    
    Use this tool when:
    - search_campaign returns partial information and you need the complete file
    - The user asks for "everything about [specific NPC/monster/etc]"
    - You need full details that might not be in search results
    - The user references a specific file by name
    
    First use list_campaign_files if you don't know the exact file path.
    
    Args:
        file_path: Full path to the file relative to campaign root
                   Examples: 'npcs/sildar-hallwinter.md', 'monsters/goblin.md',
                            'lore/phandalin/history.md', 'sessions/session-1.md'
        
    Returns:
        Complete file content with all details
    
    Note: user_id and campaign are automatically provided by the system.
    """
    
    logger.info(f"get_file_content: file_path={file_path}, user={user_id}, campaign={campaign}")
    
    if not user_id or not campaign or not CAMPAIGN_FILES_BUCKET:
        return "Error: User context or bucket not available"
    
    try:
        # Build S3 key
        s3_key = f"{user_id}/{campaign}/{file_path}"
        
        # Get file from S3
        response = s3_client.get_object(
            Bucket=CAMPAIGN_FILES_BUCKET,
            Key=s3_key
        )
        
        content = response['Body'].read().decode('utf-8')
        
        # Extract filename from path for display
        filename = file_path.split('/')[-1]
        
        result = f"📄 **{filename}** ({file_path}):\n\n"
        result += content
        
        return result
        
    except s3_client.exceptions.NoSuchKey:
        return f"File not found: {file_path}"
    except Exception as e:
        return f"Error retrieving file: {str(e)}"
