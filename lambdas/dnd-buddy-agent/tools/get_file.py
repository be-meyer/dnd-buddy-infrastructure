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
def get_file_content(category: str, filename: str, user_id: str = None, campaign: str = None) -> str:
    """
    Retrieve the complete, full content of a specific campaign file.
    
    Use this tool when:
    - search_campaign returns partial information and you need the complete file
    - The user asks for "everything about [specific NPC/monster/etc]"
    - You need full details that might not be in search results
    - The user references a specific file by name
    
    First use list_campaign_files if you don't know the exact filename.
    
    Args:
        category: File category - must be one of:
                  'npcs' (characters), 'lore' (world info),
                  'monsters' (creatures), 'sessions' (session notes)
        filename: Exact name of the file to retrieve (e.g., 'sildar-hallwinter.md')
        
    Returns:
        Complete file content with all details
    
    Note: user_id and campaign are automatically provided by the system.
    """
    
    logger.info(f"get_file_content: category={category}, filename={filename}, user={user_id}, campaign={campaign}")
    
    if not user_id or not campaign or not CAMPAIGN_FILES_BUCKET:
        return "Error: User context or bucket not available"
    
    # Validate category
    valid_categories = ['npcs', 'lore', 'monsters', 'sessions']
    if category not in valid_categories:
        return f"Invalid category '{category}'. Must be one of: {', '.join(valid_categories)}"
    
    try:
        # Build S3 key
        s3_key = f"{user_id}/{campaign}/{category}/{filename}"
        
        # Get file from S3
        response = s3_client.get_object(
            Bucket=CAMPAIGN_FILES_BUCKET,
            Key=s3_key
        )
        
        content = response['Body'].read().decode('utf-8')
        
        result = f"📄 **{filename}** (from {category}):\n\n"
        result += content
        
        return result
        
    except s3_client.exceptions.NoSuchKey:
        return f"File not found: {category}/{filename}"
    except Exception as e:
        return f"Error retrieving file: {str(e)}"
