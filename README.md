# D&D Buddy Infrastructure

AWS CDK infrastructure for D&D Buddy - a campaign management tool with AI assistant capabilities.

## Architecture Overview

D&D Buddy uses a serverless architecture with three main stacks:

### 1. AuthStack
- **Cognito User Pool**: User authentication and management
- **User Pool Client**: OAuth2 client for frontend authentication
- **Purpose**: Secure user authentication with JWT tokens

### 2. StorageStack
- **Campaign Files Bucket**: S3 storage for user campaign files
  - Structure: `{userId}/{campaign}/{category}/{filename}`
  - Categories: npcs, lore, monsters, sessions, organizations, species, players
- **Vector Store Bucket**: S3-based vector embeddings for semantic search
  - Enables RAG (Retrieval Augmented Generation) for campaign context
- **Purpose**: Persistent storage for campaign data and AI context

### 3. WebSocketStack
- **WebSocket API**: Real-time bidirectional communication
- **Connection Management**: DynamoDB table for active WebSocket connections
- **Chat History**: DynamoDB table for conversation persistence (7-day TTL)
- **Lambda Functions**:
  - **Connection Handler**: Manages WebSocket connect/disconnect with JWT authorization
  - **Agent Handler**: Processes chat messages and streams AI responses
  - **Sessions Handler**: Retrieves conversation history
  - **Indexing Handler**: Updates vector embeddings when files change
- **Purpose**: Real-time AI chat with streaming responses

## Agent Architecture

The D&D Buddy agent is built with **LangGraph + LangChain** and runs as a Lambda function.

### Agent Features
- **Multi-step reasoning**: Uses cheap model for planning, main model for final responses
- **Tool execution**: Can call multiple tools in sequence to answer complex queries
- **Streaming responses**: Real-time token streaming via WebSocket
- **Context management**: Only loads last exchange by default, retrieves more on demand
- **Conversation history**: Cached per Lambda execution to minimize DynamoDB calls

### Available Tools

1. **search_campaign**: Semantic search across campaign files using vector embeddings
   - Searches NPCs, monsters, sessions, lore, organizations, species, players
   - Returns relevant context for AI responses

2. **list_campaign_files**: Lists available files by category
   - Helps agent discover what content exists

3. **get_file_content**: Retrieves complete file content
   - Used when search results reference the same file multiple times

4. **roll_dice**: D&D dice notation parser (e.g., "2d6+3", "1d20")
   - Interprets natural language dice requests

5. **get_conversation_history**: Retrieves earlier messages from conversation
   - Cached to avoid repeated DynamoDB calls
   - Used when user references past discussion

### Agent Workflow

```
User Message → WebSocket → Agent Lambda
                              ↓
                    Load last exchange (cached)
                              ↓
                    LangGraph Agent Loop:
                    1. Planning (cheap model)
                    2. Tool execution (if needed)
                    3. Final response (main model, streaming)
                              ↓
                    Save to DynamoDB
                              ↓
                    Stream response → WebSocket → User
```

### Context Optimization

- **Initial context**: Only last 2 messages (last exchange) loaded by default
- **On-demand retrieval**: Agent calls `get_conversation_history` when user references past conversation
- **Single DynamoDB call**: Full history loaded once and cached per Lambda execution
- **Result**: Reduced latency and DynamoDB costs

## File Structure

```
cdk/
├── lib/
│   ├── dnd-buddy-stack.ts          # Main stack orchestration
│   └── stacks/
│       ├── auth-stack.ts           # Cognito authentication
│       ├── storage-stack.ts        # S3 buckets
│       └── websocket-stack.ts      # WebSocket API + Lambdas
├── lambdas/
│   ├── dnd-buddy-agent/            # AI agent
│   │   ├── agent.py                # LangGraph agent implementation
│   │   ├── main.py                 # Lambda handler
│   │   └── tools/                  # Agent tools
│   │       ├── search_campaign.py  # Vector search
│   │       ├── get_file.py         # File retrieval
│   │       ├── list_files.py       # File listing
│   │       ├── roll_dice.py        # Dice rolling
│   │       └── get_history.py      # Conversation history
│   ├── indexing/                   # Vector indexing
│   ├── sessions/                   # Session management
│   └── websocket-connection/       # WebSocket connection handler
└── bin/
    └── cdk.ts                      # CDK app entry point
```

## Deployment

### Prerequisites
- AWS CLI configured with credentials
- Node.js 18+ and npm
- Python 3.11+ (for Lambda functions)

### Deploy All Stacks
```bash
npm install
npm run build
npx cdk deploy --all
```

### Deploy Specific Stack
```bash
npx cdk deploy DndBuddyAuthStack
npx cdk deploy DndBuddyStorageStack
npx cdk deploy DndBuddyWebSocketStack
```

### Environment Variables

The agent Lambda uses these environment variables (automatically set by CDK):

- `BEDROCK_MODEL_ID`: Main model for final responses (default: eu.amazon.nova-micro-v1:0)
- `BEDROCK_MODEL_ID_TOOL`: Cheap model for planning (default: eu.amazon.nova-micro-v1:0)
- `MAX_AGENT_ITERATIONS`: Maximum tool execution loops (default: 3)
- `CHAT_HISTORY_TABLE_NAME`: DynamoDB table for conversation history
- `WEBSOCKET_API_ENDPOINT`: WebSocket API endpoint for streaming

## Development

### Local Testing
```bash
# Run tests
npm test

# Watch mode
npm run watch

# Synthesize CloudFormation
npx cdk synth
```

### Agent Development
```bash
cd lambdas/dnd-buddy-agent

# Install dependencies
pip install -r requirements.txt

# Test agent locally
python test_agent.py
```

## Cost Optimization

- **Serverless**: Pay only for actual usage (Lambda invocations, API calls)
- **S3**: Low-cost storage for campaign files and vectors
- **DynamoDB**: On-demand pricing with TTL for automatic cleanup
- **Bedrock**: Uses cost-effective models (nova-micro for planning)
- **Context caching**: Minimizes DynamoDB reads per conversation

## Security

- **Authentication**: Cognito JWT tokens required for all API calls
- **Authorization**: WebSocket authorizer validates tokens before connection
- **Session validation**: Agent validates session_id belongs to authenticated user
- **S3 access**: User-scoped paths prevent cross-user data access
- **CORS**: Configured for frontend domain only
