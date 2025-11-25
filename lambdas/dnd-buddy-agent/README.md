# D&D Buddy Agent

Lambda function implementation using LangGraph for the D&D Buddy AI assistant.

## Overview

This agent provides campaign management assistance using:
- **LangGraph**: Multi-step reasoning workflow
- **LangChain**: Tool integration and LLM orchestration
- **Bedrock**: Nova Micro model for conversations
- **S3 Vectors**: Semantic search over campaign files

## Tools

1. **search_campaign**: Semantic search across all campaign content
2. **roll_dice**: Dice rolling (e.g., "2d6+3")
3. **list_campaign_files**: List files in a category
4. **get_file_content**: Retrieve full file contents

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally with test script
python test_agent.py
```

## Deployment

Deployed as a Lambda function via CDK in the API Stack:

```bash
cd ../
cdk deploy DndBuddyApiStack
```

## API Endpoint

POST `/agent`

Request body:
```json
{
  "campaign": "my-campaign",
  "message": "Tell me about the tavern",
  "sessionId": "optional-session-id"
}
```

Response:
```json
{
  "response": "The tavern is...",
  "userId": "user123",
  "campaign": "my-campaign",
  "sessionId": "session-id"
}
```

## Environment Variables

- `VECTOR_BUCKET_NAME`: S3 bucket for vector embeddings
- `VECTOR_INDEX_NAME`: Name of the vector index
- `CAMPAIGN_FILES_BUCKET`: S3 bucket for campaign files
- `BEDROCK_MODEL_ID`: Bedrock model ID (default: eu.amazon.nova-micro-v1:0)
- `AWS_DEFAULT_REGION`: AWS region
