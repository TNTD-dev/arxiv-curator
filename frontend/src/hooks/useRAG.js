/**
 * useRAG Hook
 * 
 * Manage RAG query state with streaming support
 */

import { useState, useCallback, useRef } from 'react';
import ragService from '../services/api';
import { logError, getUserFriendlyMessage } from '../utils/errorHandler';

/**
 * Query states
 */
export const QueryState = {
  IDLE: 'idle',
  SEARCHING: 'searching',
  GENERATING: 'generating',
  DONE: 'done',
  ERROR: 'error',
  CANCELLED: 'cancelled',
};

/**
 * useRAG Hook
 * @param {object} options - Hook options
 * @returns {object} RAG state and functions
 */
export function useRAG(options = {}) {
  const {
    onQueryComplete,
    onError: onErrorCallback,
  } = options;

  // State
  const [state, setState] = useState(QueryState.IDLE);
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState(null);
  
  // Response data
  const [answer, setAnswer] = useState('');
  const [contexts, setContexts] = useState([]);
  const [metadata, setMetadata] = useState(null);
  
  // Streaming state
  const [streamStatus, setStreamStatus] = useState(null);

  // Refs
  const mountedRef = useRef(true);
  const abortRef = useRef(false);

  /**
   * Reset state
   */
  const reset = useCallback(() => {
    setState(QueryState.IDLE);
    setLoading(false);
    setStreaming(false);
    setError(null);
    setAnswer('');
    setContexts([]);
    setMetadata(null);
    setStreamStatus(null);
    abortRef.current = false;
  }, []);

  /**
   * Cancel ongoing query
   */
  const cancelQuery = useCallback(() => {
    abortRef.current = true;
    ragService.cancelQuery();
    
    if (mountedRef.current) {
      setState(QueryState.CANCELLED);
      setLoading(false);
      setStreaming(false);
    }
  }, []);

  /**
   * Submit query with streaming
   * @param {string} question - User question
   * @param {object} queryOptions - Query options
   */
  const submitQuery = useCallback(async (question, queryOptions = {}) => {
    if (!question?.trim()) {
      setError('Please enter a question');
      return;
    }

    // Reset state
    reset();
    
    setLoading(true);
    setStreaming(true);
    setState(QueryState.SEARCHING);
    abortRef.current = false;

    try {
      await ragService.queryStream(
        question,
        queryOptions,
        {
          onStatus: (status) => {
            if (!mountedRef.current || abortRef.current) return;
            
            setStreamStatus(status);
            
            if (status === 'searching') {
              setState(QueryState.SEARCHING);
            } else if (status === 'generating') {
              setState(QueryState.GENERATING);
            }
          },

          onContexts: (contextsData) => {
            if (!mountedRef.current || abortRef.current) return;
            
            setContexts(contextsData);
          },

          onAnswerChunk: (chunk) => {
            if (!mountedRef.current || abortRef.current) return;
            
            setAnswer(prev => prev + chunk);
          },

          onDone: (metadataData) => {
            if (!mountedRef.current || abortRef.current) return;
            
            setMetadata(metadataData);
            setState(QueryState.DONE);
            setLoading(false);
            setStreaming(false);

            if (onQueryComplete) {
              onQueryComplete({
                answer: answer,
                contexts: contexts,
                metadata: metadataData,
              });
            }
          },

          onError: (errorMsg) => {
            if (!mountedRef.current || abortRef.current) return;
            
            const errorMessage = typeof errorMsg === 'string' ? errorMsg : 'Query failed';
            setError(errorMessage);
            setState(QueryState.ERROR);
            setLoading(false);
            setStreaming(false);
            
            logError(new Error(errorMessage), 'RAG Query');
            
            if (onErrorCallback) {
              onErrorCallback(errorMessage);
            }
          },

          onComplete: () => {
            if (!mountedRef.current || abortRef.current) return;
            
            setStreaming(false);
          },
        }
      );
    } catch (err) {
      if (!mountedRef.current || abortRef.current) return;
      
      const errorMessage = getUserFriendlyMessage(err);
      setError(errorMessage);
      setState(QueryState.ERROR);
      setLoading(false);
      setStreaming(false);
      
      logError(err, 'RAG Query');
      
      if (onErrorCallback) {
        onErrorCallback(errorMessage);
      }
    }
  }, [reset, onQueryComplete, onErrorCallback, answer, contexts]);

  /**
   * Submit query without streaming (synchronous)
   * @param {string} question - User question
   * @param {object} queryOptions - Query options
   */
  const submitQuerySync = useCallback(async (question, queryOptions = {}) => {
    if (!question?.trim()) {
      setError('Please enter a question');
      return null;
    }

    // Reset state
    reset();
    
    setLoading(true);
    setState(QueryState.SEARCHING);
    abortRef.current = false;

    try {
      const result = await ragService.query(question, queryOptions);
      
      if (!mountedRef.current || abortRef.current) return null;
      
      setAnswer(result.answer);
      setContexts(result.contexts || []);
      setMetadata(result.metadata);
      setState(QueryState.DONE);
      setLoading(false);

      if (onQueryComplete) {
        onQueryComplete(result);
      }

      return result;
    } catch (err) {
      if (!mountedRef.current || abortRef.current) return null;
      
      const errorMessage = getUserFriendlyMessage(err);
      setError(errorMessage);
      setState(QueryState.ERROR);
      setLoading(false);
      
      logError(err, 'RAG Query Sync');
      
      if (onErrorCallback) {
        onErrorCallback(errorMessage);
      }

      return null;
    }
  }, [reset, onQueryComplete, onErrorCallback]);

  /**
   * Retry last query
   */
  const retry = useCallback(() => {
    // Would need to store last query params
    console.warn('Retry not implemented - store last query params to enable');
  }, []);

  /**
   * Get loading message based on state
   */
  const getLoadingMessage = useCallback(() => {
    switch (state) {
      case QueryState.SEARCHING:
        return 'Searching papers...';
      case QueryState.GENERATING:
        return 'Generating answer...';
      default:
        return 'Processing...';
    }
  }, [state]);

  /**
   * Check if query is in progress
   */
  const isQuerying = useCallback(() => {
    return loading || streaming;
  }, [loading, streaming]);

  /**
   * Get query result
   */
  const getResult = useCallback(() => {
    return {
      answer,
      contexts,
      metadata,
      state,
      error,
    };
  }, [answer, contexts, metadata, state, error]);

  /**
   * Cleanup on unmount
   */
  useCallback(() => {
    return () => {
      mountedRef.current = false;
      cancelQuery();
    };
  }, [cancelQuery]);

  return {
    // State
    state,
    loading,
    streaming,
    error,
    
    // Response data
    answer,
    contexts,
    metadata,
    streamStatus,

    // Actions
    submitQuery,
    submitQuerySync,
    cancelQuery,
    reset,
    retry,

    // Utilities
    getLoadingMessage,
    isQuerying,
    getResult,

    // Computed values
    hasAnswer: answer.length > 0,
    hasContexts: contexts.length > 0,
    isSuccess: state === QueryState.DONE,
    isError: state === QueryState.ERROR,
    isCancelled: state === QueryState.CANCELLED,
    numberOfContexts: contexts.length,
  };
}

export default useRAG;

