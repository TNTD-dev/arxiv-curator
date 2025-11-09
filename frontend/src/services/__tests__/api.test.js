/**
 * API Service Tests
 * 
 * Basic tests for API service functionality
 */

import ragService from '../api';

/**
 * Test health check
 */
export async function testHealthCheck() {
  console.log('\n🧪 Testing Health Check...');
  
  try {
    const health = await ragService.checkHealth();
    console.log('✅ Health check passed:', health);
    return true;
  } catch (error) {
    console.error('❌ Health check failed:', error.message);
    return false;
  }
}

/**
 * Test stats endpoint
 */
export async function testGetStats() {
  console.log('\n🧪 Testing Get Stats...');
  
  try {
    const stats = await ragService.getStats();
    console.log('✅ Get stats passed:', stats);
    return true;
  } catch (error) {
    console.error('❌ Get stats failed:', error.message);
    return false;
  }
}

/**
 * Test synchronous query
 */
export async function testSyncQuery() {
  console.log('\n🧪 Testing Synchronous Query...');
  
  try {
    const result = await ragService.query(
      'What are transformers in machine learning?',
      {
        topK: 10,
        mode: 'default',
        useReranker: true,
      }
    );
    
    console.log('✅ Sync query passed');
    console.log('  Answer length:', result.answer?.length);
    console.log('  Contexts:', result.contexts?.length);
    console.log('  Metadata:', result.metadata);
    
    return true;
  } catch (error) {
    console.error('❌ Sync query failed:', error.message);
    return false;
  }
}

/**
 * Test streaming query
 */
export async function testStreamQuery() {
  console.log('\n🧪 Testing Streaming Query...');
  
  return new Promise((resolve) => {
    let chunkCount = 0;
    
    ragService.queryStream(
      'What is attention mechanism?',
      {
        topK: 10,
        mode: 'default',
        useReranker: true,
      },
      {
        onStatus: (status) => {
          console.log('  Status:', status);
        },
        
        onContexts: (contexts) => {
          console.log('  Contexts received:', contexts.length);
        },
        
        onAnswerChunk: (chunk) => {
          chunkCount++;
        },
        
        onDone: (metadata) => {
          console.log('✅ Stream query passed');
          console.log('  Total chunks:', chunkCount);
          console.log('  Metadata:', metadata);
          resolve(true);
        },
        
        onError: (error) => {
          console.error('❌ Stream query failed:', error);
          resolve(false);
        },
      }
    );
  });
}

/**
 * Test backend reachability
 */
export async function testBackendReachable() {
  console.log('\n🧪 Testing Backend Reachability...');
  
  try {
    const isReachable = await ragService.isBackendReachable();
    
    if (isReachable) {
      console.log('✅ Backend is reachable');
    } else {
      console.warn('⚠️ Backend is not reachable');
    }
    
    return isReachable;
  } catch (error) {
    console.error('❌ Reachability check failed:', error.message);
    return false;
  }
}

/**
 * Run all tests
 */
export async function runAllTests() {
  console.log('='.repeat(60));
  console.log('🧪 Running API Service Tests');
  console.log('='.repeat(60));
  
  const tests = [
    { name: 'Backend Reachable', fn: testBackendReachable },
    { name: 'Health Check', fn: testHealthCheck },
    { name: 'Get Stats', fn: testGetStats },
    { name: 'Sync Query', fn: testSyncQuery },
    { name: 'Stream Query', fn: testStreamQuery },
  ];
  
  const results = [];
  
  for (const test of tests) {
    try {
      const result = await test.fn();
      results.push({ name: test.name, passed: result });
    } catch (error) {
      console.error(`Error in ${test.name}:`, error);
      results.push({ name: test.name, passed: false });
    }
    
    // Wait a bit between tests
    await new Promise(resolve => setTimeout(resolve, 500));
  }
  
  console.log('\n' + '='.repeat(60));
  console.log('📊 Test Results');
  console.log('='.repeat(60));
  
  results.forEach(({ name, passed }) => {
    console.log(`${passed ? '✅' : '❌'} ${name}`);
  });
  
  const passed = results.filter(r => r.passed).length;
  const total = results.length;
  
  console.log('\n' + '='.repeat(60));
  console.log(`Summary: ${passed}/${total} tests passed`);
  console.log('='.repeat(60));
  
  return { results, passed, total };
}

export default {
  testHealthCheck,
  testGetStats,
  testSyncQuery,
  testStreamQuery,
  testBackendReachable,
  runAllTests,
};

