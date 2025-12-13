"""
D&D rules and compendium search tool using S3 Vectors semantic search.
"""
import json
import os
import logging
import boto3
from langchain_core.tools import tool

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Initialize clients
bedrock_runtime = boto3.client('bedrock-runtime')
s3vectors_client = boto3.client('s3vectors')

# Environment variables
VECTOR_BUCKET = os.environ.get('VECTOR_BUCKET_NAME')
DND_VECTOR_INDEX = os.environ.get('DND_VECTOR_INDEX_NAME', 'dnd-vectors-index')
EMBEDDING_MODEL_ID = os.environ.get('EMBEDDING_MODEL_ID', 'cohere.embed-english-v3')


@tool
def search_dnd_rules(query: str, top_k: int = 5) -> str:
    """
    Search official D&D 5e rules, spells, monsters, items, classes, and compendium content.
    
    Use this tool when the user asks about:
    - D&D rules, mechanics, or how things work in 5e
    - Spell descriptions, effects, or components
    - Monster stats, abilities, or lore from official sources
    - Class features, abilities, or progression
    - Item properties, magic items, or equipment
    - Race/species traits and abilities
    - Feats, backgrounds, or other character options
    
    This searches the official D&D 5e content, NOT the user's campaign files.
    For campaign-specific content, use search_campaign instead.
    
    Args:
        query: The search query (spell name, monster, rule, etc.)
        top_k: Number of results to return (default 5)
        
    Returns:
        Relevant D&D rules/content as formatted text with source references
    """
    
    logger.info(f"search_dnd_rules: query='{query}'")
    
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
    
    # Query vectors across all categories
    search_response = s3vectors_client.query_vectors(
        vectorBucketName=VECTOR_BUCKET,
        indexName=DND_VECTOR_INDEX,
        queryVector={"float32": query_embedding},
        topK=top_k,
        returnMetadata=True,
        returnDistance=True,
    )
    
    # Format results
    results = search_response.get('vectors', [])
    logger.info(f"Found {len(results)} results")
    
    if not results:
        return f"No D&D rules found for query: '{query}'"
    
    formatted_results = []
    for i, result in enumerate(results, 1):
        metadata = result.get('metadata', {})
        chunk_text = metadata.get('chunkText', '')
        file_path = metadata.get('filePath', 'unknown')
        source = metadata.get('source', '')
        
        source_info = f" [{source}]" if source else ""
        formatted_results.append(
            f"Result {i} (from {file_path}{source_info}):\n{chunk_text}\n"
        )
    
    return "\n".join(formatted_results)
