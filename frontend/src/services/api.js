/**
 * API Service
 * 
 * Main service for communicating with the FastAPI backend
 */

import axios from 'axios';
import { API_ENDPOINTS, API_CONFIG } from '../config/api';
import { createEventSource, StreamAccumulator } from './streamParser';

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_CONFIG.BASE_URL,
  timeout: API_CONFIG.TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
apiClient.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => {
    console.log(`API Response: ${response.config.url} - Status ${response.status}`);
    return response;
  },
  (error) => {
    console.error('API Response Error:', error);
    
    if (error.response) {
      // Server responded with error status
      const errorMessage = error.response.data?.detail || error.response.statusText;
      return Promise.reject(new Error(errorMessage));
    } else if (error.request) {
      // Request made but no response
      return Promise.reject(new Error('No response from server. Please check if the backend is running.'));
    } else {
      // Something else happened
      return Promise.reject(error);
    }
  }
);

/**
 * RAG Service Class
 */
class RAGService {
  constructor() {
    this.activeEventSource = null;
    this.abortController = null;
  }

  /**
   * Check backend health
   * @returns {Promise<object>} Health status
   */
  async checkHealth() {
    try {
      const response = await apiClient.get(API_ENDPOINTS.HEALTH);
      return response.data;
    } catch (error) {
      throw new Error(`Health check failed: ${error.message}`);
    }
  }

  /**
   * Get system statistics
   * @returns {Promise<object>} System stats
   */
  async getStats() {
    try {
      const response = await apiClient.get(API_ENDPOINTS.STATS);
      return response.data;
    } catch (error) {
      throw new Error(`Failed to get stats: ${error.message}`);
    }
  }

  /**
   * Submit query (non-streaming)
   * @param {string} question - User question
   * @param {object} options - Query options
   * @returns {Promise<object>} Query response
   */
  async query(question, options = {}) {
    const {
      topK = 20,
      mode = 'default',
      useReranker = true,
    } = options;

    try {
      this.abortController = new AbortController();

      const response = await apiClient.post(
        API_ENDPOINTS.QUERY,
        {
          question,
          top_k: topK,
          mode,
          use_reranker: useReranker,
          stream: false,
        },
        {
          signal: this.abortController.signal,
        }
      );

      return response.data;
    } catch (error) {
      if (axios.isCancel(error)) {
        throw new Error('Query cancelled');
      }
      throw new Error(`Query failed: ${error.message}`);
    } finally {
      this.abortController = null;
    }
  }

  /**
   * Submit query with streaming
   * @param {string} question - User question
   * @param {object} options - Query options
   * @param {object} callbacks - Streaming callbacks
   * @returns {Promise<object>} Stream controller
   */
  async queryStream(question, options = {}, callbacks = {}) {
    const {
      topK = 20,
      mode = 'default',
      useReranker = true,
    } = options;

    const {
      onStatus,
      onContexts,
      onAnswerChunk,
      onDone,
      onError,
      onComplete,
    } = callbacks;

    // Close any existing connection
    this.cancelQuery();

    const accumulator = new StreamAccumulator();

    // Build query URL with parameters
    const params = new URLSearchParams({
      question,
      top_k: topK,
      mode,
      use_reranker: useReranker,
      stream: true,
    });

    const streamUrl = `${API_ENDPOINTS.QUERY}?${params.toString()}`;

    return new Promise((resolve, reject) => {
      try {
        // Note: EventSource only supports GET requests
        // For POST with streaming, we need to use fetch with ReadableStream
        this.streamWithFetch(streamUrl, {
          question,
          top_k: topK,
          mode,
          use_reranker: useReranker,
          stream: true,
        }, {
          onStatus: (status) => {
            accumulator.setStatus(status);
            if (onStatus) onStatus(status);
          },
          onContexts: (contexts) => {
            accumulator.setContexts(contexts);
            if (onContexts) onContexts(contexts);
          },
          onAnswerChunk: (chunk, fullAnswer) => {
            accumulator.appendAnswer(chunk);
            if (onAnswerChunk) onAnswerChunk(chunk, fullAnswer);
          },
          onDone: (metadata, answer) => {
            accumulator.setMetadata(metadata);
            if (onDone) onDone(metadata, answer);
            if (onComplete) onComplete();
            resolve(accumulator.getResult());
          },
          onError: (error) => {
            if (onError) onError(error);
            if (onComplete) onComplete();
            reject(new Error(error));
          },
        });
      } catch (error) {
        if (onError) onError(error.message);
        if (onComplete) onComplete();
        reject(error);
      }
    });
  }

  /**
   * Stream with fetch API (supports POST)
   * @param {string} url - Stream URL
   * @param {object} body - Request body
   * @param {object} callbacks - Stream callbacks
   */
  async streamWithFetch(url, body, callbacks) {
    try {
      this.abortController = new AbortController();

      const response = await fetch(API_ENDPOINTS.QUERY, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
        signal: this.abortController.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      let currentEvent = null;
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        
        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        
        // Keep the last incomplete line in buffer
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.trim()) continue;

          if (line.startsWith('event:')) {
            currentEvent = line.substring(6).trim();
          } else if (line.startsWith('data:')) {
            const dataStr = line.substring(5).trim();
            
            try {
              const data = JSON.parse(dataStr);

              switch (currentEvent) {
                case 'status':
                  if (callbacks.onStatus) {
                    callbacks.onStatus(data.status);
                  }
                  break;

                case 'contexts':
                  if (callbacks.onContexts) {
                    callbacks.onContexts(data);
                  }
                  break;

                case 'answer':
                  if (data.chunk && callbacks.onAnswerChunk) {
                    callbacks.onAnswerChunk(data.chunk);
                  }
                  break;

                case 'done':
                  if (callbacks.onDone) {
                    callbacks.onDone(data);
                  }
                  return;

                case 'error':
                  if (callbacks.onError) {
                    callbacks.onError(data.error || 'Unknown error');
                  }
                  return;

                default:
                  console.warn('Unknown event type:', currentEvent);
              }
            } catch (e) {
              console.error('Failed to parse SSE data:', dataStr, e);
            }
          }
        }
      }
    } catch (error) {
      if (error.name === 'AbortError') {
        console.log('Stream aborted');
      } else {
        console.error('Stream error:', error);
        if (callbacks.onError) {
          callbacks.onError(error.message);
        }
      }
    } finally {
      this.abortController = null;
    }
  }

  /**
   * Cancel ongoing query
   */
  cancelQuery() {
    if (this.activeEventSource) {
      this.activeEventSource.close();
      this.activeEventSource = null;
    }

    if (this.abortController) {
      this.abortController.abort();
      this.abortController = null;
    }
  }

  /**
   * Check if backend is reachable
   * @returns {Promise<boolean>}
   */
  async isBackendReachable() {
    try {
      await this.checkHealth();
      return true;
    } catch (error) {
      return false;
    }
  }
}

// Export singleton instance
const ragService = new RAGService();
export default ragService;

// Export class for testing
export { RAGService };

