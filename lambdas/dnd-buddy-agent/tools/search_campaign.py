"""
Campaign search tool using S3 Vectors semantic search.
"""
import json
import os
import logging
import boto3
from langchain_core.tools import tool

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Initialize clients
bedrock_runtime = boto3.client('bedrock-runtime')
s3vectors_client = boto3.client('s3vectors')

# Environment variables
VECTOR_BUCKET = os.environ.get('VECTOR_BUCKET_NAME')
VECTOR_INDEX = os.environ.get('VECTOR_INDEX_NAME', 'campaign-vectors-index')
EMBEDDING_MODEL_ID = os.environ.get('EMBEDDING_MODEL_ID', 'cohere.embed-english-v3')


@tool
def search_campaign(query: str, category: str = None, top_k: int = 5, user_id: str = None, campaign: str = None) -> str:
    """
    Search campaign information using semantic search across all campaign content.
    
    Use this tool when the user asks questions about their campaign content like:
    - "Tell me about [NPC name]"
    - "What happened in our last session?"
    - "What monsters have we encountered?"
    - "What do we know about [location/item/lore]?"
    
    This returns relevant chunks of text. If you need the complete file, use get_file_content instead.
    
    Args:
        query: The search query (what to look for)
        category: Optional category filter - use to narrow search to specific type:
                  'npcs' for characters, 
                  'lore' for world and general info, 
                  'monsters' for creatures, 
                  'sessions' for session notes, 
                  'organizations' for information about organizations,
                  'species' for races and species information,
                  'players' for player character information
        top_k: Number of results to return (default 5)
        
    Returns:
        Relevant campaign information as formatted text with source references
    
    Note: user_id and campaign are automatically provided by the system.
    """
    
    logger.info(f"search_campaign: query='{query}', category={category}, user={user_id}, campaign={campaign}")
    
    if not user_id or not campaign:
        return "Error: User context not available"
    
    # Generate embedding for query
    response = bedrock_runtime.invoke_model(
        modelId=EMBEDDING_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "texts": [query],
            "input_type": "search_query",
            "truncate": "END"
        })
    )
    
    result = json.loads(response['body'].read())
    query_embedding = result.get('embeddings', [[]])[0]
    
    # Build metadata filter using $and operator for multiple conditions
    filter_conditions = [
        {"userId": user_id},
        {"campaign": campaign}
    ]
    
    if category:
        filter_conditions.append({"category": category})
    
    metadata_filter = {"$and": filter_conditions}
    
    # Query vectors
    search_response = s3vectors_client.query_vectors(
        vectorBucketName=VECTOR_BUCKET,
        indexName=VECTOR_INDEX,
        queryVector={"float32": query_embedding},
        topK=top_k,
        returnMetadata=True,
        returnDistance=True,
        filter=metadata_filter
    )
    
    # Format results
    results = search_response.get('vectors', [])
    logger.info(f"Found {len(results)} results")
    
    # If no results and category was specified, try again without category filter
    if not results and category:
        logger.info(f"search_campaign: No results with category={category}, retrying without filter")
        metadata_filter_no_category = {
            "$and": [
                {"userId": user_id},
                {"campaign": campaign}
            ]
        }
        
        search_response = s3vectors_client.query_vectors(
            vectorBucketName=VECTOR_BUCKET,
            indexName=VECTOR_INDEX,
            queryVector={"float32": query_embedding},
            topK=top_k,
            returnMetadata=True,
            returnDistance=True,
            filter=metadata_filter_no_category
        )
        
        results = search_response.get('vectors', [])
    
    if not results:
        return f"No relevant information found in the campaign for query: '{query}' (category: {category or 'all'})"
    
    formatted_results = []
    for i, result in enumerate(results, 1):
        metadata = result.get('metadata', {})
        chunk_text = metadata.get('chunkText', '')
        result_category = metadata.get('category', 'unknown')
        filename = metadata.get('filename', 'unknown')
        
        formatted_results.append(
            f"Result {i} (from {result_category}/{filename}):\n{chunk_text}\n"
        )
    
    return "\n".join(formatted_results)
