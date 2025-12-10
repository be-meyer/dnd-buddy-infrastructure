import * as cdk from 'aws-cdk-lib/core';
import { Construct } from 'constructs';
import { AuthStack } from './stacks/auth-stack';
import { StorageStack } from './stacks/storage-stack';
import { ApiStack } from './stacks/api-stack';
import { WebSocketStack } from './stacks/websocket-stack';

export class DnDBuddyStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // Model configuration - easily change models here
    const bedrockModelId = 'anthropic.claude-haiku-4-5-20251001-v1:0';
    const bedrockModelIdTool = 'amazon.nova-2-lite-v1:0';
    const embeddingModelId = 'cohere.embed-english-v3';

    // Create Auth Stack
    const authStack = new AuthStack(scope, 'DndBuddyAuthStack', props);

    // Create Storage Stack
    const storageStack = new StorageStack(scope, 'DndBuddyStorageStack', props);

    // Create API Stack (depends on auth and storage)
    const apiStack = new ApiStack(scope, 'DndBuddyApiStack', {
      ...props,
      userPool: authStack.userPool,
      campaignFilesBucket: storageStack.campaignFilesBucket,
      vectorBucketName: storageStack.vectorBucket.ref,
      vectorIndexName: storageStack.vectorIndex.indexName || 'campaign-vectors-index',
      chatHistoryTable: storageStack.chatHistoryTable,
    });
    apiStack.addDependency(authStack);
    apiStack.addDependency(storageStack);

    // Create WebSocket Stack (depends on auth and storage, creates agent Lambda)
    const webSocketStack = new WebSocketStack(scope, 'DndBuddyWebSocketStack', {
      ...props,
      userPool: authStack.userPool,
      connectionTable: storageStack.websocketConnectionTable,
      campaignFilesBucket: storageStack.campaignFilesBucket,
      vectorIndexName: storageStack.vectorIndex.indexName || 'campaign-vectors-index',
      chatHistoryTable: storageStack.chatHistoryTable,
      bedrockModelId,
      bedrockModelIdTool,
      embeddingModelId,
    });
    webSocketStack.addDependency(authStack);
    webSocketStack.addDependency(storageStack);
  }
}
