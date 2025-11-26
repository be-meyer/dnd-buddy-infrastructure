"""
Lambda handler for indexing campaign files into S3 Vectors.
Simple implementation: read file -> chunk with overlap -> embed -> store in S3 Vectors.
"""
import json
import logging
import os
import re
import random
import math
from typing import Dict, Any, List
from datetime import datetime
import boto3

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
s3_client = boto3.client('s3')
bedrock_client = boto3.client('bedrock-runtime')
s3vectors_client = boto3.client('s3vectors')

# Get environment variables
CAMPAIGN_FILES_BUCKET = os.environ.get('CAMPAIGN_FILES_BUCKET')
VECTOR_BUCKET = os.environ.get('VECTOR_BUCKET_NAME')
VECTOR_INDEX = os.environ.get('VECTOR_INDEX_NAME', 'campaign-vectors-index')
BEDROCK_MODEL_ID = os.environ.get('BEDROCK_MODEL_ID', 'cohere.embed-english-v3')
# Add environment variables at top:
CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', '800'))
CHUNK_OVERLAP = int(os.environ.get('CHUNK_OVERLAP', '100'))

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """Split text into overlapping chunks."""
    if not text or not text.strip():
        return []
    
    if len(text) <= chunk_size:
        return [{'text': text, 'chunkIndex': 0, 'totalChunks': 1, 'startPosition': 0, 'endPosition': len(text)}]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        
        # Try to break at sentence boundary
        if end < len(text):
            search_start = max(start, end - int(chunk_size * 0.2))
            matches = list(re.finditer(r'[.!?][\s\n]', text[search_start:end]))
            if matches:
                end = search_start + matches[-1].end()
        
        chunks.append({
            'text': text[start:end],
            'chunkIndex': len(chunks),
            'totalChunks': 0,
            'startPosition': start,
            'endPosition': end
        })
        
        start = end - overlap if end < len(text) else end
        if start <= chunks[-1]['startPosition']:
            start = end
    
    # Update totalChunks
    for chunk in chunks:
        chunk['totalChunks'] = len(chunks)
    
    return chunks


def generate_embedding(text: str) -> List[float]:
    """Generate embedding using Bedrock Cohere."""
    response = bedrock_client.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "texts": [text],
            "input_type": "search_document",
            "truncate": "END"
        })
    )
    
    result = json.loads(response['body'].read())
    embeddings = result.get('embeddings', [])
    
    if embeddings and len(embeddings) > 0:
        return embeddings[0]
    
    return None


def delete_existing_vectors(user_id: str, campaign: str, file_path: str) -> int:
    """Delete all existing vectors for a given file using metadata filtering."""
    try:
        # Use QueryVectors with metadata filter to find all vectors for this file
        # Create a random unit vector (all zeros not allowed)
        
        dummy_vector = [random.random() for _ in range(1024)]
        norm = math.sqrt(sum(x * x for x in dummy_vector))
        dummy_vector = [x / norm for x in dummy_vector]
        
        deleted_count = 0
        
        # Query in batches to find all matching vectors
        while True:
            query_params = {
                'vectorBucketName': VECTOR_BUCKET,
                'indexName': VECTOR_INDEX,
                'queryVector': {'float32': dummy_vector},
                'topK': 30,  # Maximum allowed
                'returnMetadata': False,
                'returnDistance': False,
                'filter': {
                    '$and': [
                        {'userId': user_id},
                        {'campaign': campaign},
                        {'filePath': file_path}
                    ]
                }
            }
            
            response = s3vectors_client.query_vectors(**query_params)
            vectors = response.get('vectors', [])
            
            if not vectors:
                break
            
            # Extract keys and delete
            keys_to_delete = [v['key'] for v in vectors]
            
            if keys_to_delete:
                s3vectors_client.delete_vectors(
                    vectorBucketName=VECTOR_BUCKET,
                    indexName=VECTOR_INDEX,
                    keys=keys_to_delete
                )
                deleted_count += len(keys_to_delete)
                logger.info(f"Deleted {len(keys_to_delete)} vectors")
            
            # If we got fewer than 30 results, we're done
            if len(vectors) < 30:
                break
        
        logger.info(f"Total vectors deleted: {deleted_count}")
        return deleted_count
        
    except Exception as e:
        logger.warning(f"Error deleting existing vectors: {str(e)}")
        # Don't fail the whole operation if deletion fails
        return 0


def index_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Save and index campaign file: save to S3 -> chunk -> embed -> store in S3 Vectors."""
    logger.info(f"Save and index request: {json.dumps(event)}")
    
    # Parse body if it's a string (from API Gateway)
    if isinstance(event.get('body'), str):
        try:
            body = json.loads(event['body'])
        except json.JSONDecodeError:
            return {
                'statusCode': 400,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Content-Type,Authorization',
                    'Access-Control-Allow-Methods': 'POST,OPTIONS'
                },
                'body': json.dumps({'error': 'Invalid JSON in request body'})
            }
    else:
        body = event
    
    # Extract userId from Cognito authorizer claims
    user_id = event.get('requestContext', {}).get('authorizer', {}).get('claims', {}).get('cognito:username')
    
    # Extract other parameters from body
    campaign = body.get('campaign')
    file_path = body.get('filePath')  # Full path relative to campaign root
    file_content = body.get('content')
    
    logger.info(f"User ID from Cognito: {user_id}")
    
    cors_headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type,Authorization',
        'Access-Control-Allow-Methods': 'POST,OPTIONS'
    }
    
    if not all([user_id, campaign, file_path, file_content]):
        return {
            'statusCode': 400,
            'headers': cors_headers,
            'body': json.dumps({'error': 'Missing required parameters: userId, campaign, filePath, content'})
        }
    
    try:
        # 0. Delete existing vectors for this file
        deleted_count = delete_existing_vectors(user_id, campaign, file_path)
        logger.info(f"Deleted {deleted_count} existing vectors for {file_path}")
        original_text = file_content
        logger.info(f"Processing {len(original_text)} characters")
        
        # 1. Save file to campaign files bucket (overwrites existing)
        s3_key = f"{user_id}/{campaign}/{file_path}"
        logger.info(f"Saving file to s3://{CAMPAIGN_FILES_BUCKET}/{s3_key}")
        
        s3_client.put_object(
            Bucket=CAMPAIGN_FILES_BUCKET,
            Key=s3_key,
            Body=original_text.encode('utf-8'),
            ContentType='text/markdown',
            Metadata={
                'userId': user_id,
                'campaign': campaign,
                'filePath': file_path,
                'lastModified': datetime.utcnow().isoformat() + 'Z'
            }
        )
        logger.info(f"File saved successfully")
        
        # 2. Chunk text with overlap
        chunks = chunk_text(original_text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        logger.info(f"Created {len(chunks)} chunks")
        
        # 3. Generate embeddings and prepare vectors for batch insert
        indexed_at = datetime.utcnow().isoformat() + 'Z'
        vectors_to_insert = []
        
        for chunk in chunks:
            # Generate embedding
            embedding = generate_embedding(chunk['text'])
            if not embedding:
                logger.error(f"Failed to generate embedding for chunk {chunk['chunkIndex']}")
                continue
            
            # Prepare vector for batch insert
            # Replace slashes in file_path to avoid key format issues
            safe_file_path = file_path.replace('/', '|')
            vector_key = f"{user_id}#{campaign}#{safe_file_path}#{chunk['chunkIndex']}"
            
            vectors_to_insert.append({
                'key': vector_key,
                'data': {'float32': embedding},
                'metadata': {
                    'userId': user_id,
                    'campaign': campaign,
                    'filePath': file_path,
                    'chunkText': chunk['text'],
                    'chunkIndex': chunk['chunkIndex'],
                    'totalChunks': chunk['totalChunks'],
                    'startPosition': chunk['startPosition'],
                    'endPosition': chunk['endPosition'],
                    'indexedAt': indexed_at
                }
            })
        
        # 4. Store all vectors in S3 Vectors (batch up to 500 at a time)
        chunks_stored = 0
        batch_size = 500
        
        for i in range(0, len(vectors_to_insert), batch_size):
            batch = vectors_to_insert[i:i + batch_size]
            
            s3vectors_client.put_vectors(
                vectorBucketName=VECTOR_BUCKET,
                indexName=VECTOR_INDEX,
                vectors=batch
            )
            
            chunks_stored += len(batch)
            logger.info(f"Stored batch: {chunks_stored}/{len(vectors_to_insert)} vectors")
        
        logger.info(f"Successfully indexed {chunks_stored} chunks")
        
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type,Authorization',
                'Access-Control-Allow-Methods': 'POST,OPTIONS'
            },
            'body': json.dumps({
                'message': 'File indexed successfully',
                'chunksProcessed': chunks_stored,
                'chunksDeleted': deleted_count,
                'userId': user_id,
                'campaign': campaign,
                'filePath': file_path
            })
        }
        
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return {
            'statusCode': 500,
            'headers': cors_headers,
            'body': json.dumps({'error': str(e)})
        }
