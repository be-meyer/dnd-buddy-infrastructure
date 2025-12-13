"""
Get full D&D rules file content tool.
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
DND_RULES_BUCKET = os.environ.get('DND_RULES_BUCKET')


@tool
def get_dnd_file(file_path: str) -> str:
    """
    Retrieve the complete content of a D&D rules or compendium file.
    
    Use this tool when:
    - search_dnd_rules returns partial information and you need the full entry
    - The user asks for complete spell descriptions, full monster stat blocks, etc.
    - You need detailed class features or item descriptions
    
    Args:
        file_path: Path to the file as returned by search_dnd_rules
                   Examples: 'compendium/spells/fireball.md', 
                            'compendium/bestiary/beholder.md',
                            'rules/conditions.md'
        
    Returns:
        Complete file content with all details
    """
    
    logger.info(f"get_dnd_file: file_path={file_path}")
    
    if not DND_RULES_BUCKET:
        return "Error: D&D rules bucket not configured"
    
    try:
        # Get file from S3
        response = s3_client.get_object(
            Bucket=DND_RULES_BUCKET,
            Key=file_path
        )
        
        content = response['Body'].read().decode('utf-8')
        
        # Extract filename from path for display
        filename = file_path.split('/')[-1]
        
        result = f"📜 **{filename}** ({file_path}):\n\n"
        result += content
        
        return result
        
    except s3_client.exceptions.NoSuchKey:
        return f"File not found: {file_path}"
    except Exception as e:
        return f"Error retrieving file: {str(e)}"
