"""
List campaign files tool.
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
def list_campaign_files(category: str = None, user_id: str = None, campaign: str = None) -> str:
    """
    List all files available in the campaign to help discover what content exists.
    
    Use this tool when the user asks:
    - "What NPCs do I have?"
    - "Show me all my session notes"
    - "What files do I have?"
    - "List all monsters"
    - Or when you need to know what files exist before retrieving one
    
    This helps users discover their campaign content and helps you know what files
    are available to retrieve with get_file_content.
    
    Args:
        category: Optional category filter to show only specific type:
                  'npcs', 'lore', 'monsters', 'organizations', 'sessions', 'species', or 'players'
                  Leave empty to show all categories
        
    Returns:
        List of filenames organized by category
    
    Note: user_id and campaign are automatically provided by the system.
    """
    
    logger.info(f"list_campaign_files: category={category}, user={user_id}, campaign={campaign}")
    
    if not user_id or not campaign or not CAMPAIGN_FILES_BUCKET:
        return "Error: User context or bucket not available"
    
    try:
        # Build S3 prefix
        if category:
            categories = [category]
        else:
            categories = ['npcs', 'lore', 'monsters', 'sessions', 'organizations', 'species', 'players']
        
        # List files
        files_by_category = {}
        
        for cat in categories:
            cat_prefix = f"{user_id}/{campaign}/{cat}/"
            response = s3_client.list_objects_v2(
                Bucket=CAMPAIGN_FILES_BUCKET,
                Prefix=cat_prefix
            )
            
            files = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    key = obj['Key']
                    if key != cat_prefix:  # Skip directory marker
                        filename = key.split('/')[-1]
                        files.append(filename)
            
            if files:
                files_by_category[cat] = sorted(files)
        
        # Format output
        if not files_by_category:
            return f"No files found in campaign '{campaign}'" + (f" for category '{category}'" if category else "")
        
        result = f"📁 Campaign Files for '{campaign}':\n\n"
        for cat, files in files_by_category.items():
            result += f"**{cat.upper()}** ({len(files)} files):\n"
            for filename in files:
                result += f"  - {filename}\n"
            result += "\n"
        
        return result.strip()
        
    except Exception as e:
        return f"Error listing files: {str(e)}"
