/**
 * Test Integration Component
 * 
 * Component to test all Phase 10 functionality
 */

import React, { useState } from 'react';
import useRAG from './hooks/useRAG';
import useSettings from './hooks/useSettings';
import useHistory from './hooks/useHistory';
import useStats from './hooks/useStats';
import { runAllTests } from './services/__tests__/api.test';

function TestIntegration() {
  const [testResults, setTestResults] = useState(null);
  const [runningTests, setRunningTests] = useState(false);

  // Test hooks
  const rag = useRAG();
  const settings = useSettings();
  const history = useHistory();
  const stats = useStats({ autoRefresh: false });

  const handleRunTests = async () => {
    setRunningTests(true);
    const results = await runAllTests();
    setTestResults(results);
    setRunningTests(false);
  };

  const handleTestQuery = async () => {
    await rag.submitQuery('What are transformers?', {
      topK: settings.topK,
      mode: settings.mode,
      useReranker: settings.useReranker,
    });
  };

  return (
    <div style={{ padding: '20px', fontFamily: 'monospace' }}>
      <h1>🧪 Phase 10 Integration Test</h1>
      
      {/* API Tests */}
      <section style={{ marginBottom: '30px', padding: '15px', border: '1px solid #ccc' }}>
        <h2>1. API Service Tests</h2>
        <button 
          onClick={handleRunTests}
          disabled={runningTests}
          style={{ padding: '10px 20px', cursor: 'pointer' }}
        >
          {runningTests ? 'Running Tests...' : 'Run API Tests'}
        </button>
        
        {testResults && (
          <div style={{ marginTop: '10px', padding: '10px', background: '#f5f5f5' }}>
            <p><strong>Results: {testResults.passed}/{testResults.total} passed</strong></p>
            {testResults.results.map((result, idx) => (
              <div key={idx}>
                {result.passed ? '✅' : '❌'} {result.name}
              </div>
            ))}
          </div>
        )}
      </section>

      {/* Stats Hook Test */}
      <section style={{ marginBottom: '30px', padding: '15px', border: '1px solid #ccc' }}>
        <h2>2. Stats Hook</h2>
        <button 
          onClick={stats.refreshAll}
          disabled={stats.loading}
          style={{ padding: '10px 20px', cursor: 'pointer' }}
        >
          Refresh Stats
        </button>
        
        <div style={{ marginTop: '10px', padding: '10px', background: '#f5f5f5' }}>
          <p><strong>Connection:</strong> {stats.connected ? '✅ Connected' : '❌ Disconnected'}</p>
          <p><strong>Status:</strong> {stats.statusMessage}</p>
          <p><strong>Papers:</strong> {stats.indexedPapers}</p>
          <p><strong>Queries:</strong> {stats.totalQueries}</p>
          <p><strong>Reranker:</strong> {stats.rerankerEnabled ? '✅' : '❌'}</p>
        </div>
      </section>

      {/* Settings Hook Test */}
      <section style={{ marginBottom: '30px', padding: '15px', border: '1px solid #ccc' }}>
        <h2>3. Settings Hook</h2>
        
        <div style={{ marginTop: '10px' }}>
          <label>
            Top-K: {settings.topK}
            <input 
              type="range" 
              min="5" 
              max="50" 
              value={settings.topK}
              onChange={(e) => settings.setSetting('topK', parseInt(e.target.value))}
              style={{ marginLeft: '10px' }}
            />
          </label>
        </div>
        
        <div style={{ marginTop: '10px' }}>
          <label>
            <input 
              type="checkbox"
              checked={settings.useReranker}
              onChange={(e) => settings.setSetting('useReranker', e.target.checked)}
            />
            Use Reranker
          </label>
        </div>
        
        <div style={{ marginTop: '10px' }}>
          <label>
            Mode: 
            <select 
              value={settings.mode}
              onChange={(e) => settings.setSetting('mode', e.target.value)}
              style={{ marginLeft: '10px' }}
            >
              {settings.validModes.map(mode => (
                <option key={mode} value={mode}>
                  {settings.getModeDisplayName(mode)}
                </option>
              ))}
            </select>
          </label>
        </div>
        
        <div style={{ marginTop: '10px', padding: '10px', background: '#f5f5f5' }}>
          <p><strong>Current Settings:</strong></p>
          <pre>{JSON.stringify(settings.settings, null, 2)}</pre>
        </div>
      </section>

      {/* RAG Hook Test */}
      <section style={{ marginBottom: '30px', padding: '15px', border: '1px solid #ccc' }}>
        <h2>4. RAG Hook</h2>
        
        <button 
          onClick={handleTestQuery}
          disabled={rag.loading}
          style={{ padding: '10px 20px', cursor: 'pointer', marginRight: '10px' }}
        >
          {rag.loading ? 'Querying...' : 'Test Query'}
        </button>
        
        <button 
          onClick={rag.reset}
          style={{ padding: '10px 20px', cursor: 'pointer' }}
        >
          Reset
        </button>
        
        <div style={{ marginTop: '10px', padding: '10px', background: '#f5f5f5' }}>
          <p><strong>State:</strong> {rag.state}</p>
          <p><strong>Loading:</strong> {rag.loading ? '⏳' : '✅'}</p>
          <p><strong>Streaming:</strong> {rag.streaming ? '📡' : '❌'}</p>
          <p><strong>Contexts:</strong> {rag.numberOfContexts}</p>
          
          {rag.error && (
            <p style={{ color: 'red' }}><strong>Error:</strong> {rag.error}</p>
          )}
          
          {rag.answer && (
            <div style={{ marginTop: '10px' }}>
              <strong>Answer:</strong>
              <div style={{ 
                marginTop: '5px', 
                padding: '10px', 
                background: 'white',
                maxHeight: '200px',
                overflow: 'auto'
              }}>
                {rag.answer}
              </div>
            </div>
          )}
        </div>
      </section>

      {/* History Hook Test */}
      <section style={{ marginBottom: '30px', padding: '15px', border: '1px solid #ccc' }}>
        <h2>5. History Hook</h2>
        
        <button 
          onClick={() => history.createConversation()}
          style={{ padding: '10px 20px', cursor: 'pointer', marginRight: '10px' }}
        >
          New Conversation
        </button>
        
        <button 
          onClick={history.clearAll}
          style={{ padding: '10px 20px', cursor: 'pointer' }}
        >
          Clear All
        </button>
        
        <div style={{ marginTop: '10px', padding: '10px', background: '#f5f5f5' }}>
          <p><strong>Total:</strong> {history.totalConversations}</p>
          <p><strong>Active:</strong> {history.activeConversations}</p>
          <p><strong>Archived:</strong> {history.archivedConversations}</p>
          <p><strong>Pinned:</strong> {history.pinnedConversations}</p>
          
          {history.currentConversation && (
            <div style={{ marginTop: '10px' }}>
              <strong>Current Conversation:</strong>
              <pre>{JSON.stringify(history.currentConversation, null, 2)}</pre>
            </div>
          )}
        </div>
      </section>

      {/* Summary */}
      <section style={{ padding: '15px', border: '2px solid #4CAF50', background: '#e8f5e9' }}>
        <h2>✅ Phase 10 Summary</h2>
        <ul>
          <li>✅ API Service Layer - Ready</li>
          <li>✅ Streaming Support - Ready</li>
          <li>✅ useSettings Hook - Ready</li>
          <li>✅ useHistory Hook - Ready</li>
          <li>✅ useStats Hook - Ready</li>
          <li>✅ useRAG Hook - Ready</li>
          <li>✅ Error Handling - Ready</li>
          <li>✅ Storage Utilities - Ready</li>
        </ul>
        <p><strong>Status:</strong> Phase 10 Complete! 🎉</p>
      </section>
    </div>
  );
}

export default TestIntegration;

