# D&D Buddy Infrastructure

AWS CDK infrastructure for D&D Buddy - a campaign management tool with AI assistant capabilities.

## Architecture Overview

D&D Buddy uses a serverless architecture built on AWS CDK with four main stacks:

### 1. AuthStack
Handles user authentication and authorization.

**Resources:**
- **Cognito User Pool**: User authentication with email/username sign-in
- **User Pool Client**: OAuth2 client for frontend authentication
- **Security**: Password policy enforcement, email verification, account recovery

**Outputs:** User Pool ID, Client ID, ARN

### 2. StorageStack
Manages all data persistence for the application.

**Resources:**
- **Campaign Files Bucket** (`dnd-buddy-files-{account}`): S3 storage for user campaign files
  - Structure: `{userId}/{campaign}/{category}/{filename}`
  - Categories: `npcs`, `lore`, `monsters`, `sessions`, `organizations`, `species`, `players`
  - Encryption: S3-managed (SSE-S3)
  
- **Vector Store** (`dnd-vec-{account}`): S3 Vectors for semantic search
  - Index: `campaign-vectors-index`
  - Dimensions: 1024 (Cohere embed-english-v3)
  - Distance metric: Cosine similarity
  - Enables RAG (Retrieval Augmented Generation) for campaign context
  
- **Chat History Table** (`dnd-buddy-chat-history`): DynamoDB for conversation persistence
  - Partition key: `SessionId`
  - TTL: 7 days (automatic cleanup)
  - Billing: On-demand
  
- **WebSocket Connection Table** (`dnd-buddy-websocket-connections`): DynamoDB for active connections
  - Partition key: `connectionId`
  - GSI: `UserIdIndex` for querying by user
  - TTL: Automatic cleanup of stale connections

**Outputs:** Bucket names, table names, vector index name

### 3. ApiStack
Provides REST API endpoints for file indexing and session management.

**Resources:**
- **REST API Gateway**: CORS-enabled API with Cognito authorization
- **Indexing Lambda** (`dnd-buddy-indexing`): Updates vector embeddings when files change
  - Runtime: Python 3.11
  - Timeout: 5 minutes
  - Memory: 512 MB
  - Endpoint: `POST /index`
  
- **Sessions Lambda** (`dnd-buddy-sessions`): Retrieves conversation history
  - Runtime: Python 3.11
  - Timeout: 30 seconds
  - Memory: 256 MB
  - Endpoints: `GET /sessions`, `GET /sessions/{sessionId}`

**Outputs:** API URL, Lambda ARNs

### 4. WebSocketStack
Enables real-time bidirectional communication for AI chat.

**Resources:**
- **WebSocket API Gateway**: Real-time communication with JWT authorization
  - Routes: `$connect`, `$disconnect`, `$default`
  - Stage: `prod` (auto-deploy enabled)
  
- **Connection Lambda** (`dnd-buddy-websocket-connection`): Handles WebSocket lifecycle
  - Runtime: Node.js 24.x
  - Handles: Authorization, connect, disconnect events
  - Validates JWT tokens from Cognito
  
- **Agent Lambda** (`dnd-buddy-agent`): AI assistant with streaming responses
  - Runtime: Python 3.14 (ARM64)
  - Timeout: 5 minutes
  - Memory: 1024 MB
  - Framework: LangGraph + LangChain
  - Models: Configurable (default: Claude Haiku 4.5 + Nova Micro)

**Outputs:** WebSocket URL, API ID, Lambda ARNs

## Agent Architecture

The D&D Buddy agent is built with **LangGraph + LangChain** and runs as a Lambda function with a dual-model architecture for cost optimization.

### Dual-Model Architecture

The agent uses two models for different purposes:

1. **Planning Model** (default: Nova Micro)
   - Purpose: Tool selection and execution planning
   - Temperature: 0.0 (deterministic)
   - Max tokens: 400
   - No streaming
   - Cost-optimized for iterative reasoning

2. **Creative Model** (default: Claude Haiku 4.5)
   - Purpose: Final response generation
   - Temperature: 0.6 (creative)
   - Max tokens: 1500
   - Streaming enabled
   - Used only once per request

**Why this matters:** The agent may call tools multiple times (up to 3 iterations) to gather information. Using a cheap model for planning and a better model for the final response significantly reduces costs while maintaining quality.

### Agent Workflow

```
User Message → WebSocket → Agent Lambda
                              ↓
                    Load conversation history (last 2 messages)
                    Load campaign context (background + recent sessions)
                              ↓
                    ┌─────────────────────────────────────┐
                    │  LangGraph Agent Loop (Planning)    │
                    │  Model: Nova Micro (cheap)          │
                    │  Max iterations: 3                  │
                    │                                     │
                    │  1. Analyze user request            │
                    │  2. Select tools to call            │
                    │  3. Execute tools with context      │
                    │  4. Repeat if more info needed      │
                    └─────────────────────────────────────┘
                              ↓
                    ┌─────────────────────────────────────┐
                    │  Final Response Generation          │
                    │  Model: Claude Haiku (creative)     │
                    │  Streaming: Yes                     │
                    │                                     │
                    │  Synthesize tool results            │
                    │  Generate helpful response          │
                    │  Stream tokens to WebSocket         │
                    └─────────────────────────────────────┘
                              ↓
                    Save to DynamoDB (with TTL)
                              ↓
                    Return response + tools summary
```

### Available Tools

The agent has access to 4 tools for campaign assistance:

1. **search_campaign** - Semantic search across all campaign files
   - Uses S3 Vectors with Cohere embeddings
   - Searches: NPCs, monsters, sessions, lore, organizations, species, players
   - Parameters: `query` (string), `top_k` (default: 5)
   - Auto-injected: `user_id`, `campaign`
   - Returns: Relevant snippets with filenames

2. **get_file_content** - Retrieves complete file content
   - Used when search results reference the same file multiple times
   - Parameters: `file_path` (relative to campaign root)
   - Auto-injected: `user_id`, `campaign`
   - Returns: Full markdown content

3. **roll_dice** - D&D dice notation parser
   - Supports standard notation: `1d20`, `2d6+3`, `4d8-2`
   - Interprets natural language: "roll perception", "attack roll"
   - Parameters: `dice_notation` (string)
   - Returns: Roll result with breakdown

4. **get_conversation_history** - Retrieves earlier messages
   - Used when user references past conversation
   - Parameters: `message_count` (default: 10)
   - Auto-injected: `session_id`
   - Cached per Lambda execution
   - Returns: List of previous messages

**Context Injection:** Tools that need user context (`search_campaign`, `get_file_content`) automatically receive `user_id` and `campaign`. Session-aware tools (`get_conversation_history`) receive `session_id`. This keeps tool definitions clean while ensuring security.

### Context Management

**Startup Context:**
- Last 2 messages from conversation history (last exchange)
- Campaign background context (via semantic search for "world setting themes")
- Recent sessions context (via semantic search for "recent session latest")

**On-Demand Context:**
- Agent calls `get_conversation_history` when user references earlier discussion
- Full history loaded once and cached per Lambda execution
- Reduces DynamoDB reads and latency

**System Prompt:**
- Built dynamically with campaign-specific context
- Includes campaign background, recent sessions, tool descriptions
- Emphasizes campaign-specific responses over generic D&D advice
- Enforces concise formatting (100-300 words, structured output)

### Security & Validation

- **Session ownership validation**: Session ID must start with `{userId}-`
- **JWT authorization**: WebSocket authorizer validates Cognito tokens
- **User-scoped paths**: All S3 operations scoped to `{userId}/{campaign}/`
- **No cross-user access**: Tools automatically inject authenticated user context

## Project Structure

```
cdk/
├── bin/
│   └── cdk.ts                      # CDK app entry point
│
├── lib/
│   ├── dnd-buddy-stack.ts          # Main stack orchestration
│   └── stacks/
│       ├── auth-stack.ts           # Cognito user pool & client
│       ├── storage-stack.ts        # S3 buckets, DynamoDB tables, S3 Vectors
│       ├── api-stack.ts            # REST API + indexing/sessions Lambdas
│       └── websocket-stack.ts      # WebSocket API + agent Lambda
│
├── lambdas/
│   ├── dnd-buddy-agent/            # AI agent (Python 3.14)
│   │   ├── agent.py                # LangGraph agent with dual-model architecture
│   │   ├── main.py                 # Lambda handler with WebSocket streaming
│   │   ├── requirements.txt        # Python dependencies
│   │   └── tools/
│   │       ├── __init__.py         # Tool exports
│   │       ├── search_campaign.py  # S3 Vectors semantic search
│   │       ├── get_file.py         # S3 file content retrieval
│   │       ├── roll_dice.py        # D&D dice notation parser
│   │       └── get_history.py      # DynamoDB conversation history
│   │
│   ├── indexing/                   # Vector indexing Lambda (Python 3.11)
│   │   ├── handler.py              # Processes files and creates embeddings
│   │   └── requirements.txt
│   │
│   ├── sessions/                   # Session management Lambda (Python 3.11)
│   │   ├── handler.py              # Retrieves chat history
│   │   └── requirements.txt
│   │
│   └── websocket-connection/       # WebSocket lifecycle Lambda (Node.js 24.x)
│       └── index.js                # Handles connect/disconnect + JWT auth
│
├── test/
│   └── cdk.test.ts                 # CDK stack tests
│
├── cdk.json                        # CDK configuration
├── package.json                    # Node.js dependencies
└── tsconfig.json                   # TypeScript configuration
```

## Deployment

### Prerequisites

- **AWS CLI**: Configured with credentials (`aws configure`)
- **Node.js**: 18+ with npm
- **Python**: 3.11+ (for Lambda bundling)
- **AWS Account**: With Bedrock model access enabled
  - Required models: Claude Haiku 4.5, Nova Micro, Cohere Embed English v3
  - Enable in AWS Console → Bedrock → Model access

### Quick Start

```bash
# Install dependencies
npm install

# Build TypeScript
npm run build

# Deploy all stacks (first time)
npx cdk bootstrap  # Only needed once per account/region
npx cdk deploy --all

# Deploy takes ~5-10 minutes
# Outputs will include WebSocket URL, API URL, and resource IDs
```

### Deploy Specific Stacks

```bash
# Deploy in order (respects dependencies)
npx cdk deploy DndBuddyAuthStack
npx cdk deploy DndBuddyStorageStack
npx cdk deploy DndBuddyApiStack
npx cdk deploy DndBuddyWebSocketStack

# Or deploy just what changed
npx cdk deploy DndBuddyWebSocketStack
```

### Configuration

**Model Selection** - Edit `cdk/lib/dnd-buddy-stack.ts`:

```typescript
const bedrockModelId = 'anthropic.claude-haiku-4-5-20251001-v1:0';  // Creative model
const bedrockModelIdTool = 'amazon.nova-micro-v1:0';                // Planning model
const embeddingModelId = 'cohere.embed-english-v3';                 // Embeddings
```

**Agent Parameters** - Set in WebSocketStack environment variables:

- `BEDROCK_MODEL_ID`: Creative model for final responses (prefixed with `eu.`)
- `BEDROCK_MODEL_ID_TOOL`: Planning model for tool execution (prefixed with `eu.`)
- `EMBEDDING_MODEL_ID`: Model for vector embeddings
- `MAX_AGENT_ITERATIONS`: Tool execution loop limit (default: 3)
- `MAX_HISTORY_TOKENS`: Conversation history token limit (default: 10000)

All environment variables are automatically set by CDK during deployment.

### Stack Outputs

After deployment, CDK outputs important values:

**AuthStack:**
- `UserPoolId`: For frontend Amplify configuration
- `UserPoolClientId`: For frontend Amplify configuration
- `UserPoolArn`: For IAM policies

**StorageStack:**
- `CampaignFilesBucketName`: S3 bucket for user files
- `VectorBucketName`: S3 Vectors bucket
- `VectorIndexName`: Index name for semantic search
- `ChatHistoryTableName`: DynamoDB table name
- `WebSocketConnectionTableName`: DynamoDB table name

**ApiStack:**
- `ApiUrl`: REST API endpoint
- `IndexingLambdaArn`: ARN for indexing function
- `SessionsLambdaArn`: ARN for sessions function

**WebSocketStack:**
- `WebSocketUrl`: WebSocket endpoint for frontend
- `WebSocketApiId`: API Gateway ID
- `AgentLambdaArn`: ARN for agent function
- `ConnectionLambdaArn`: ARN for connection handler

Save these outputs for frontend configuration.

## Development

### Local Testing

```bash
# Run CDK tests
npm test

# Watch mode for TypeScript changes
npm run watch

# Synthesize CloudFormation templates (no deployment)
npx cdk synth

# Show differences between deployed and local
npx cdk diff

# List all stacks
npx cdk list
```

### Agent Development

The agent can be tested locally with proper AWS credentials:

```bash
cd lambdas/dnd-buddy-agent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export VECTOR_BUCKET_NAME="dnd-vec-{your-account-id}"
export VECTOR_INDEX_NAME="campaign-vectors-index"
export CAMPAIGN_FILES_BUCKET="dnd-buddy-files-{your-account-id}"
export CHAT_HISTORY_TABLE_NAME="dnd-buddy-chat-history"
export BEDROCK_MODEL_ID="eu.anthropic.claude-haiku-4-5-20251001-v1:0"
export BEDROCK_MODEL_ID_TOOL="eu.amazon.nova-micro-v1:0"
export EMBEDDING_MODEL_ID="cohere.embed-english-v3"

# Test agent (requires deployed infrastructure)
python -c "from agent import main; print(main({'userId': 'test', 'campaign': 'test', 'prompt': 'Hello'}))"
```

**Note:** Local testing requires deployed AWS resources (S3 Vectors, DynamoDB, etc.) and proper IAM permissions.

### Lambda Hot Reloading

For rapid iteration on Lambda code without full CDK deployment:

```bash
# Update agent Lambda code only
cd lambdas/dnd-buddy-agent
zip -r function.zip .
aws lambda update-function-code \
  --function-name dnd-buddy-agent \
  --zip-file fileb://function.zip

# Update indexing Lambda
cd ../indexing
zip -r function.zip .
aws lambda update-function-code \
  --function-name dnd-buddy-indexing \
  --zip-file fileb://function.zip
```

### Debugging

**CloudWatch Logs:**
```bash
# Tail agent logs
aws logs tail /aws/lambda/dnd-buddy-agent --follow

# Tail WebSocket connection logs
aws logs tail /aws/lambda/dnd-buddy-websocket-connection --follow

# Filter for errors
aws logs tail /aws/lambda/dnd-buddy-agent --follow --filter-pattern "ERROR"
```

**Lambda Insights:**
- Enable in AWS Console → Lambda → Configuration → Monitoring
- Provides performance metrics and traces

**X-Ray Tracing:**
- Enable in Lambda configuration for distributed tracing
- Useful for debugging tool execution flow

## Cost Optimization

**Dual-Model Architecture:**
- Planning model (Nova Micro): ~$0.000035 per 1K input tokens
- Creative model (Claude Haiku): ~$0.00025 per 1K input tokens
- Agent may call planning model 3x, creative model 1x per request
- Estimated cost: $0.001-0.003 per conversation turn

**Serverless Pricing:**
- Lambda: Pay per invocation + compute time (ARM64 is 20% cheaper)
- API Gateway: $1 per million WebSocket messages
- DynamoDB: On-demand pricing (no idle costs)
- S3: $0.023 per GB/month for storage
- S3 Vectors: Query pricing based on vector dimensions

**Cost Reduction Strategies:**
- TTL on DynamoDB tables (7 days) for automatic cleanup
- Minimal conversation history loading (last 2 messages)
- ARM64 Lambda architecture (20% cost reduction)
- On-demand DynamoDB (no provisioned capacity waste)
- S3 lifecycle policies (optional, for old campaign files)

**Estimated Monthly Cost:**
- Light usage (100 conversations): ~$5-10
- Medium usage (1000 conversations): ~$30-50
- Heavy usage (10000 conversations): ~$200-300

## Security

**Authentication & Authorization:**
- Cognito JWT tokens required for all API calls
- WebSocket authorizer validates tokens before connection
- Session validation: `sessionId` must start with `{userId}-`
- No anonymous access to any resources

**Data Isolation:**
- S3 paths scoped to `{userId}/{campaign}/`
- DynamoDB queries filtered by authenticated user
- Vector search automatically scoped to user's campaign
- No cross-user data leakage possible

**Network Security:**
- S3 buckets: Block all public access
- API Gateway: CORS configured for specific origins
- Lambda: VPC deployment optional (not required for this architecture)
- Encryption: S3-managed encryption (SSE-S3) for all buckets

**IAM Least Privilege:**
- Each Lambda has minimal required permissions
- No wildcard resource access
- Separate roles for each function
- No hardcoded credentials (uses IAM roles)

**Compliance:**
- Data retention: 7-day TTL on chat history
- User data deletion: Manual S3/DynamoDB cleanup required
- Audit logging: CloudWatch Logs with 7-day retention
- Secrets management: Use AWS Secrets Manager for API keys (if needed)

## Monitoring & Observability

**CloudWatch Metrics:**
- Lambda invocations, duration, errors, throttles
- API Gateway request count, latency, 4xx/5xx errors
- DynamoDB read/write capacity, throttles
- S3 Vectors query latency

**CloudWatch Logs:**
- All Lambda functions: 7-day retention
- Structured logging with correlation IDs
- Agent logs include: tool calls, iterations, token counts

**Alarms (Recommended):**
```bash
# High error rate
aws cloudwatch put-metric-alarm \
  --alarm-name dnd-buddy-agent-errors \
  --metric-name Errors \
  --namespace AWS/Lambda \
  --statistic Sum \
  --period 300 \
  --threshold 10 \
  --comparison-operator GreaterThanThreshold

# High latency
aws cloudwatch put-metric-alarm \
  --alarm-name dnd-buddy-agent-latency \
  --metric-name Duration \
  --namespace AWS/Lambda \
  --statistic Average \
  --period 300 \
  --threshold 30000 \
  --comparison-operator GreaterThanThreshold
```

## Troubleshooting

**Agent not responding:**
- Check CloudWatch logs: `/aws/lambda/dnd-buddy-agent`
- Verify Bedrock model access enabled in AWS Console
- Check Lambda timeout (should be 5 minutes)
- Verify WebSocket connection is established

**Vector search returning no results:**
- Ensure files are indexed: `POST /index` with file metadata
- Check vector bucket and index exist in S3 Vectors
- Verify embedding model ID matches indexing and search
- Check CloudWatch logs for indexing errors

**WebSocket connection fails:**
- Verify JWT token is valid and not expired
- Check token is passed as query parameter: `?token=...`
- Review connection Lambda logs for authorization errors
- Ensure Cognito User Pool ID is correct

**High costs:**
- Review CloudWatch metrics for Lambda invocations
- Check if agent is hitting max iterations frequently
- Consider reducing `MAX_AGENT_ITERATIONS` to 2
- Monitor Bedrock token usage in CloudWatch

**DynamoDB throttling:**
- On-demand mode should auto-scale
- Check for hot partition keys (unlikely with sessionId)
- Review access patterns in CloudWatch Contributor Insights

## Cleanup

To delete all resources:

```bash
# Delete all stacks (in reverse order)
npx cdk destroy --all

# Confirm each stack deletion
# Note: S3 buckets with RETAIN policy must be manually deleted
```

**Manual cleanup required:**
- S3 buckets (if RETAIN policy is set)
- CloudWatch log groups (if retention is set to NEVER_EXPIRE)
- Cognito User Pool (if RETAIN policy is set)

## Contributing

When making changes:

1. Follow the project philosophy: **minimal and simple**
2. Update this README if architecture changes
3. Test locally before deploying
4. Use `cdk diff` to review changes
5. Deploy to dev environment first (if available)
