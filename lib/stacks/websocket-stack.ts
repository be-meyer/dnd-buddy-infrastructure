import * as cdk from 'aws-cdk-lib/core';
import * as apigatewayv2 from 'aws-cdk-lib/aws-apigatewayv2';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import * as cognito from 'aws-cdk-lib/aws-cognito';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as logs from 'aws-cdk-lib/aws-logs';
import { WebSocketLambdaIntegration } from 'aws-cdk-lib/aws-apigatewayv2-integrations';
import { WebSocketLambdaAuthorizer } from 'aws-cdk-lib/aws-apigatewayv2-authorizers';
import { Construct } from 'constructs';
import * as path from 'path';

export interface WebSocketStackProps extends cdk.StackProps {
  userPool: cognito.IUserPool;
  connectionTable: dynamodb.ITable;
  campaignFilesBucket: any;
  vectorIndexName: string;
  chatHistoryTable: dynamodb.ITable;
  bedrockModelId: string;
  bedrockModelIdTool: string;
  embeddingModelId: string;
}

export class WebSocketStack extends cdk.Stack {
  public readonly webSocketApi: apigatewayv2.WebSocketApi;
  public readonly webSocketStage: apigatewayv2.WebSocketStage;
  public readonly agentLambda: lambda.Function;

  constructor(scope: Construct, id: string, props: WebSocketStackProps) {
    super(scope, id, props);

    // Log group for WebSocket connection Lambda
    const connectionLogGroup = new logs.LogGroup(this, 'WebSocketConnectionLogGroup', {
      logGroupName: '/aws/lambda/dnd-buddy-websocket-connection',
      retention: logs.RetentionDays.ONE_WEEK,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    // Lambda function for WebSocket connection management and authorization
    // This single Lambda handles: authorization, $connect, and $disconnect
    const connectionLambda = new lambda.Function(this, 'WebSocketConnectionLambda', {
      functionName: 'dnd-buddy-websocket-connection',
      runtime: lambda.Runtime.NODEJS_24_X,
      handler: 'index.handler',
      code: lambda.Code.fromAsset(path.join(__dirname, '../../lambdas/websocket-connection')),
      timeout: cdk.Duration.seconds(30),
      memorySize: 256,
      logGroup: connectionLogGroup,
      environment: {
        CONNECTION_TABLE_NAME: props.connectionTable.tableName,
        COGNITO_USER_POOL_ID: props.userPool.userPoolId,
        COGNITO_REGION: this.region,
      },
    });

    // Grant DynamoDB permissions
    props.connectionTable.grantReadWriteData(connectionLambda);

    // Log group for Agent Lambda
    const agentLogGroup = new logs.LogGroup(this, 'AgentLogGroup', {
      logGroupName: '/aws/lambda/dnd-buddy-agent',
      retention: logs.RetentionDays.ONE_WEEK,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    // Agent Lambda - Create before WebSocket API so we can reference it
    this.agentLambda = new lambda.Function(this, 'AgentLambda', {
      functionName: 'dnd-buddy-agent',
      architecture: lambda.Architecture.ARM_64,
      runtime: lambda.Runtime.PYTHON_3_14,
      handler: 'main.lambda_handler',
      code: lambda.Code.fromAsset(path.join(__dirname, '../../lambdas/dnd-buddy-agent'), {
        bundling: {
          platform: 'linux/arm64',
          image: lambda.Runtime.PYTHON_3_14.bundlingImage,
          command: [
            'bash', '-c',
            'pip install -r requirements.txt -t /asset-output && cp -au . /asset-output'
          ],
        },
      }),
      timeout: cdk.Duration.minutes(5),
      memorySize: 512,
      logGroup: agentLogGroup,
      environment: {
        VECTOR_BUCKET_NAME: `dnd-vec-${cdk.Stack.of(this).account}`,
        VECTOR_INDEX_NAME: props.vectorIndexName,
        CAMPAIGN_FILES_BUCKET: props.campaignFilesBucket.bucketName,
        CHAT_HISTORY_TABLE_NAME: props.chatHistoryTable.tableName,
        BEDROCK_MODEL_ID: `eu.${props.bedrockModelId}`,
        BEDROCK_MODEL_ID_TOOL: `eu.${props.bedrockModelIdTool}`,
        EMBEDDING_MODEL_ID: props.embeddingModelId,
        MAX_HISTORY_TOKENS: '10000',
        MAX_AGENT_ITERATIONS: '3'
      },
    });

    // Grant DynamoDB permissions for chat history
    props.chatHistoryTable.grantReadWriteData(this.agentLambda);

    // Grant S3 read permissions for campaign files
    props.campaignFilesBucket.grantRead(this.agentLambda);

    // Grant S3 Vectors query permissions
    this.agentLambda.addToRolePolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: ['s3vectors:QueryVectors', 's3vectors:GetVectors'],
        resources: [
          `arn:aws:s3vectors:${this.region}:${this.account}:bucket/dnd-vec-${cdk.Stack.of(this).account}`,
          `arn:aws:s3vectors:${this.region}:${this.account}:bucket/dnd-vec-${cdk.Stack.of(this).account}/index/*`,
        ],
      })
    );

    this.agentLambda.addToRolePolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: ['bedrock:InvokeModel', 'bedrock:Converse', 'bedrock:InvokeModelWithResponseStream'],
        resources: [
          `arn:aws:bedrock:${this.region}:${this.account}:inference-profile/eu.${props.bedrockModelId}`,
          `arn:aws:bedrock:*::foundation-model/${props.bedrockModelId}`,
          `arn:aws:bedrock:${this.region}:${this.account}:inference-profile/eu.${props.bedrockModelIdTool}`,
          `arn:aws:bedrock:*::foundation-model/${props.bedrockModelIdTool}`,
          `arn:aws:bedrock:${this.region}::foundation-model/${props.embeddingModelId}`,
        ],
      })
    );

    // Create WebSocket API with custom authorizer
    this.webSocketApi = new apigatewayv2.WebSocketApi(this, 'DndBuddyWebSocketApi', {
      apiName: 'dnd-buddy-websocket',
      description: 'WebSocket API for D&D Buddy agent streaming',
      connectRouteOptions: {
        integration: new WebSocketLambdaIntegration('ConnectIntegration', connectionLambda),
        authorizer: new WebSocketLambdaAuthorizer('WebSocketAuthorizer', connectionLambda, {
          identitySource: ['route.request.querystring.token'],
        }),
      },
      disconnectRouteOptions: {
        integration: new WebSocketLambdaIntegration('DisconnectIntegration', connectionLambda),
      },
      defaultRouteOptions: {
        integration: new WebSocketLambdaIntegration('DefaultIntegration', this.agentLambda),
      },
    });

    // Create production stage
    this.webSocketStage = new apigatewayv2.WebSocketStage(this, 'ProductionStage', {
      webSocketApi: this.webSocketApi,
      stageName: 'prod',
      autoDeploy: true,
    });

    // Grant agent Lambda permission to manage connections (send messages back)
    this.agentLambda.addToRolePolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: ['execute-api:ManageConnections'],
        resources: [
          `arn:aws:execute-api:${this.region}:${this.account}:${this.webSocketApi.apiId}/*`,
        ],
      })
    );

    // Add WebSocket API endpoint as environment variable to agent Lambda
    this.agentLambda.addEnvironment(
      'WEBSOCKET_API_ENDPOINT',
      `https://${this.webSocketApi.apiId}.execute-api.${this.region}.amazonaws.com/${this.webSocketStage.stageName}`
    );

    // Outputs
    new cdk.CfnOutput(this, 'WebSocketUrl', {
      value: this.webSocketStage.url,
      description: 'WebSocket API URL',
      exportName: 'DndBuddy-WebSocketUrl',
    });

    new cdk.CfnOutput(this, 'WebSocketApiId', {
      value: this.webSocketApi.apiId,
      description: 'WebSocket API ID',
      exportName: 'DndBuddy-WebSocketApiId',
    });

    new cdk.CfnOutput(this, 'ConnectionLambdaArn', {
      value: connectionLambda.functionArn,
      description: 'ARN of the WebSocket connection Lambda function',
    });

    new cdk.CfnOutput(this, 'AgentLambdaArn', {
      value: this.agentLambda.functionArn,
      description: 'ARN of the agent Lambda function',
    });
  }
}
