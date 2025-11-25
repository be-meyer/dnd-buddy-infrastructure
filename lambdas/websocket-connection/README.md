# WebSocket Connection Handler

This Lambda function handles WebSocket connection lifecycle and authentication for the D&D Buddy application.

## Functionality

This single Lambda function handles three different event types:

### 1. Authorization (Custom Authorizer)
- **Trigger**: WebSocket connection attempt with `?token=<jwt>` query parameter
- **Purpose**: Validates Cognito JWT tokens before allowing WebSocket connections
- **Process**:
  1. Extracts JWT token from query string
  2. Verifies token signature using Cognito JWKS
  3. Validates token expiration and claims
  4. Extracts `cognito:username` as userId
  5. Returns IAM policy (Allow/Deny) with userId in context

### 2. Connect ($connect route)
- **Trigger**: After successful authorization, when WebSocket connection is established
- **Purpose**: Stores connection metadata in DynamoDB
- **Process**:
  1. Extracts connectionId from API Gateway
  2. Extracts userId from authorizer context
  3. Creates DynamoDB record with:
     - `connectionId` (partition key)
     - `userId` (for GSI lookup)
     - `timestamp` (connection time)
     - `ttl` (2 hours expiration)

### 3. Disconnect ($disconnect route)
- **Trigger**: When WebSocket connection is closed (gracefully or unexpectedly)
- **Purpose**: Removes connection record from DynamoDB
- **Process**:
  1. Extracts connectionId from API Gateway
  2. Deletes record from DynamoDB

## Environment Variables

- `CONNECTION_TABLE_NAME`: DynamoDB table name for storing connections
- `COGNITO_USER_POOL_ID`: Cognito User Pool ID for JWT verification
- `COGNITO_REGION`: AWS region where Cognito User Pool is located

## DynamoDB Schema

### Connection Table
```
{
  connectionId: string (PK),
  userId: string (GSI),
  timestamp: number,
  ttl: number
}
```

## Dependencies

- `@aws-sdk/client-dynamodb`: DynamoDB client
- `@aws-sdk/lib-dynamodb`: DynamoDB document client
- `jsonwebtoken`: JWT verification
- `jwks-rsa`: JWKS client for Cognito public keys

## Installation

```bash
cd cdk/lambdas/websocket-connection
npm install
```

## Testing Locally

You can test the Lambda function locally by creating test events:

### Authorization Event
```json
{
  "type": "REQUEST",
  "methodArn": "arn:aws:execute-api:us-east-1:123456789012:abcdef123/prod/$connect",
  "queryStringParameters": {
    "token": "eyJraWQiOiJ..."
  }
}
```

### Connect Event
```json
{
  "requestContext": {
    "routeKey": "$connect",
    "connectionId": "abc123",
    "authorizer": {
      "userId": "user123"
    }
  }
}
```

### Disconnect Event
```json
{
  "requestContext": {
    "routeKey": "$disconnect",
    "connectionId": "abc123"
  }
}
```

## Security Considerations

- JWT tokens are validated using Cognito's public keys (JWKS)
- Tokens are verified for signature, expiration, and claims
- Connection records automatically expire after 2 hours (TTL)
- Each user's connections are isolated by userId
- Authorization happens before connection establishment
- **Invalid tokens prevent connection**: When authorization fails, the authorizer returns a "Deny" policy, causing API Gateway to reject the connection with 401 Unauthorized. The client never establishes a WebSocket connection.

## Error Handling

- **Authorization failures**: Returns "Deny" policy, API Gateway rejects connection with 401 Unauthorized
- **Invalid token signature**: Returns "Deny" policy, connection rejected before establishment
- **Missing token**: Returns "Deny" policy, connection rejected
- **Missing userId in token**: Returns "Deny" policy, connection rejected
- **DynamoDB failures**: Returns 500 error, logged to CloudWatch
- **Unknown routes**: Returns 400 Bad Request
- **Unknown routes**: Returns 400 Bad Request

## Monitoring

All events and errors are logged to CloudWatch Logs with the following information:
- Full event payload (for debugging)
- Authorization results
- Connection/disconnection events
- DynamoDB operation results
- Error details with stack traces
