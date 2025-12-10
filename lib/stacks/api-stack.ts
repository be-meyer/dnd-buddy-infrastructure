import * as cdk from 'aws-cdk-lib/core';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import * as apigateway from 'aws-cdk-lib/aws-apigateway';
import * as cognito from 'aws-cdk-lib/aws-cognito';
import * as logs from 'aws-cdk-lib/aws-logs';
import { Construct } from 'constructs';
import * as path from 'path';

export interface ApiStackProps extends cdk.StackProps {
  userPool: cognito.IUserPool;
  campaignFilesBucket: s3.IBucket;
  vectorBucketName: string;
  vectorIndexName: string;
  chatHistoryTable: dynamodb.ITable;
}

export class ApiStack extends cdk.Stack {
  public readonly api: apigateway.RestApi;

  constructor(scope: Construct, id: string, props: ApiStackProps) {
    super(scope, id, props);

    // Log group for Indexing Lambda
    const indexingLogGroup = new logs.LogGroup(this, 'IndexingLogGroup', {
      logGroupName: '/aws/lambda/dnd-buddy-indexing',
      retention: logs.RetentionDays.ONE_WEEK,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    // Indexing Lambda function (boto3 is included in runtime, no layer needed)
    const indexingLambda = new lambda.Function(this, 'IndexingLambda', {
      functionName: 'dnd-buddy-indexing',
      runtime: lambda.Runtime.PYTHON_3_11,
      handler: 'handler.index_handler',
      code: lambda.Code.fromAsset(path.join(__dirname, '../../lambdas/indexing')),
      timeout: cdk.Duration.minutes(5),
      memorySize: 512,
      logGroup: indexingLogGroup,
      environment: {
        CAMPAIGN_FILES_BUCKET: props.campaignFilesBucket.bucketName,
        VECTOR_BUCKET_NAME: `dnd-vec-${cdk.Stack.of(this).account}`,
        VECTOR_INDEX_NAME: props.vectorIndexName,
        BEDROCK_MODEL_ID: 'cohere.embed-english-v3',
        VALID_CATEGORIES: 'npcs,lore,monsters,sessions,organizations,species,players',
      },
    });

    // Grant Lambda permissions
    props.campaignFilesBucket.grantReadWrite(indexingLambda);

    indexingLambda.addToRolePolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: ['s3vectors:PutVectors', 's3vectors:GetVectors', 's3vectors:QueryVectors', 's3vectors:DeleteVectors'],
        resources: [
          `arn:aws:s3vectors:${this.region}:${this.account}:bucket/dnd-vec-${cdk.Stack.of(this).account}`,
          `arn:aws:s3vectors:${this.region}:${this.account}:bucket/dnd-vec-${cdk.Stack.of(this).account}/index/*`,
        ],
      })
    );

    indexingLambda.addToRolePolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: ['bedrock:InvokeModel'],
        resources: [
          `arn:aws:bedrock:${this.region}::foundation-model/cohere.embed-english-v3`,
          `arn:aws:bedrock:${this.region}::foundation-model/cohere.embed-multilingual-v3`,
        ],
      })
    );

    // Log group for Sessions Lambda
    const sessionsLogGroup = new logs.LogGroup(this, 'SessionsLogGroup', {
      logGroupName: '/aws/lambda/dnd-buddy-sessions',
      retention: logs.RetentionDays.ONE_WEEK,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    // Sessions Lambda for managing chat history
    const sessionsLambda = new lambda.Function(this, 'SessionsLambda', {
      functionName: 'dnd-buddy-sessions',
      runtime: lambda.Runtime.PYTHON_3_11,
      handler: 'handler.lambda_handler',
      code: lambda.Code.fromAsset(path.join(__dirname, '../../lambdas/sessions')),
      timeout: cdk.Duration.seconds(30),
      memorySize: 256,
      logGroup: sessionsLogGroup,
      environment: {
        CHAT_HISTORY_TABLE_NAME: props.chatHistoryTable.tableName,
      },
    });

    // Grant DynamoDB read permissions for sessions
    props.chatHistoryTable.grantReadData(sessionsLambda);

    // API Gateway
    this.api = new apigateway.RestApi(this, 'DndBuddyApi', {
      restApiName: 'DnD Buddy API',
      description: 'API for D&D Buddy application',
      defaultCorsPreflightOptions: {
        allowOrigins: ['*'],
        allowMethods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
        allowHeaders: [
          'Content-Type',
          'X-Amz-Date',
          'Authorization',
          'X-Api-Key',
          'X-Amz-Security-Token',
        ],
        allowCredentials: false,
      },
    });

    // Cognito Authorizer
    const authorizer = new apigateway.CognitoUserPoolsAuthorizer(this, 'ApiAuthorizer', {
      cognitoUserPools: [props.userPool],
      authorizerName: 'dnd-buddy-authorizer',
      identitySource: 'method.request.header.Authorization',
    });

    // /index endpoint
    const indexResource = this.api.root.addResource('index');
    indexResource.addMethod('POST', new apigateway.LambdaIntegration(indexingLambda), {
      authorizer: authorizer,
      authorizationType: apigateway.AuthorizationType.COGNITO,
    });

    // /sessions endpoint - list all sessions for user
    const sessionsResource = this.api.root.addResource('sessions');
    sessionsResource.addMethod('GET', new apigateway.LambdaIntegration(sessionsLambda), {
      authorizer: authorizer,
      authorizationType: apigateway.AuthorizationType.COGNITO,
    });

    // /sessions/{sessionId} endpoint - get specific session history
    const sessionDetailResource = sessionsResource.addResource('{sessionId}');
    sessionDetailResource.addMethod('GET', new apigateway.LambdaIntegration(sessionsLambda), {
      authorizer: authorizer,
      authorizationType: apigateway.AuthorizationType.COGNITO,
    });

    // Outputs
    new cdk.CfnOutput(this, 'ApiUrl', {
      value: this.api.url,
      description: 'URL of the D&D Buddy API',
      exportName: 'DndBuddy-ApiUrl',
    });

    new cdk.CfnOutput(this, 'IndexingLambdaArn', {
      value: indexingLambda.functionArn,
      description: 'ARN of the indexing Lambda function',
    });

    new cdk.CfnOutput(this, 'SessionsLambdaArn', {
      value: sessionsLambda.functionArn,
      description: 'ARN of the sessions Lambda function',
    });
  }
}
