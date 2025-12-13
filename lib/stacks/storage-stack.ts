import * as cdk from 'aws-cdk-lib/core';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import { CfnVectorBucket, CfnIndex } from 'aws-cdk-lib/aws-s3vectors';
import { Construct } from 'constructs';

export interface StorageStackProps extends cdk.StackProps {}

export class StorageStack extends cdk.Stack {
  public readonly campaignFilesBucket: s3.Bucket;
  public readonly dndRulesBucket: s3.Bucket;
  public readonly vectorBucket: CfnVectorBucket;
  public readonly vectorIndex: CfnIndex;
  public readonly dndVectorIndex: CfnIndex;
  public readonly chatHistoryTable: dynamodb.Table;
  public readonly websocketConnectionTable: dynamodb.Table;

  constructor(scope: Construct, id: string, props?: StorageStackProps) {
    super(scope, id, props);

    // S3 bucket for campaign files
    this.campaignFilesBucket = new s3.Bucket(this, 'CampaignFilesBucket', {
      bucketName: `dnd-buddy-files-${cdk.Stack.of(this).account}`,
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      removalPolicy: cdk.RemovalPolicy.DESTROY
    });

    // S3 bucket for D&D rules and compendium files
    this.dndRulesBucket = new s3.Bucket(this, 'DnDRulesBucket', {
      bucketName: `dnd-buddy-rules-${cdk.Stack.of(this).account}`,
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      removalPolicy: cdk.RemovalPolicy.DESTROY
    });

    // DynamoDB table for chat history
    this.chatHistoryTable = new dynamodb.Table(this, 'ChatHistoryTable', {
      tableName: 'dnd-buddy-chat-history',
      partitionKey: { name: 'SessionId', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      timeToLiveAttribute: 'expireAt',
      removalPolicy: cdk.RemovalPolicy.DESTROY
    });

    // DynamoDB table for WebSocket connections
    this.websocketConnectionTable = new dynamodb.Table(this, 'WebSocketConnectionTable', {
      tableName: 'dnd-buddy-websocket-connections',
      partitionKey: { name: 'connectionId', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      timeToLiveAttribute: 'ttl',
      removalPolicy: cdk.RemovalPolicy.DESTROY
    });

    // Add GSI for querying connections by userId
    this.websocketConnectionTable.addGlobalSecondaryIndex({
      indexName: 'UserIdIndex',
      partitionKey: { name: 'userId', type: dynamodb.AttributeType.STRING },
      projectionType: dynamodb.ProjectionType.ALL,
    });

    // S3 Vectors bucket for vector embeddings
    const vectorBucketName = `dnd-vec-${cdk.Stack.of(this).account}`;
    
    this.vectorBucket = new CfnVectorBucket(this, 'VectorBucket', {
      vectorBucketName: vectorBucketName,
    });

    // S3 Vectors index for similarity search
    this.vectorIndex = new CfnIndex(this, 'VectorIndex', {
      indexName: 'campaign-vectors-index',
      vectorBucketName: vectorBucketName,
      dimension: 1024, // Cohere embed-english-v3 produces 1024-dimensional vectors
      distanceMetric: 'cosine',
      dataType: 'float32',
      metadataConfiguration: {
        nonFilterableMetadataKeys: ['chunkText'],
      },
    });

    this.vectorIndex.addDependency(this.vectorBucket);

    // S3 Vectors index for D&D rules and compendium
    this.dndVectorIndex = new CfnIndex(this, 'DnDVectorIndex', {
      indexName: 'dnd-vectors-index',
      vectorBucketName: vectorBucketName,
      dimension: 1024, // Cohere embed-english-v3 produces 1024-dimensional vectors
      distanceMetric: 'cosine',
      dataType: 'float32',
      metadataConfiguration: {
        nonFilterableMetadataKeys: ['chunkText'],
      },
    });

    this.dndVectorIndex.addDependency(this.vectorBucket);

    // Outputs
    new cdk.CfnOutput(this, 'CampaignFilesBucketName', {
      value: this.campaignFilesBucket.bucketName,
      description: 'S3 bucket for campaign files',
      exportName: 'DndBuddy-CampaignFilesBucketName',
    });

    new cdk.CfnOutput(this, 'DnDRulesBucketName', {
      value: this.dndRulesBucket.bucketName,
      description: 'S3 bucket for D&D rules and compendium',
      exportName: 'DndBuddy-DnDRulesBucketName',
    });

    new cdk.CfnOutput(this, 'VectorBucketName', {
      value: this.vectorBucket.ref,
      description: 'S3 Vectors bucket for embeddings',
      exportName: 'DndBuddy-VectorBucketName',
    });

    new cdk.CfnOutput(this, 'VectorIndexName', {
      value: this.vectorIndex.indexName || 'campaign-vectors-index',
      description: 'S3 Vectors index name',
      exportName: 'DndBuddy-VectorIndexName',
    });

    new cdk.CfnOutput(this, 'DnDVectorIndexName', {
      value: this.dndVectorIndex.indexName || 'dnd-vectors-index',
      description: 'S3 DnD Vectors index name',
      exportName: 'DndBuddy-DnDVectorIndexName',
    });

    new cdk.CfnOutput(this, 'ChatHistoryTableName', {
      value: this.chatHistoryTable.tableName,
      description: 'DynamoDB table for chat history',
      exportName: 'DndBuddy-ChatHistoryTableName',
    });

    new cdk.CfnOutput(this, 'WebSocketConnectionTableName', {
      value: this.websocketConnectionTable.tableName,
      description: 'DynamoDB table for WebSocket connections',
      exportName: 'DndBuddy-WebSocketConnectionTableName',
    });
  }
}
