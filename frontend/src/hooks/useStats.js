/**
 * useStats Hook
 * 
 * Fetch and manage system statistics with auto-refresh
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import ragService from '../services/api';
import { logError } from '../utils/errorHandler';

/**
 * useStats Hook
 * @param {object} options - Hook options
 * @returns {object} Stats state and functions
 */
export function useStats(options = {}) {
  const {
    autoRefresh = false,
    refreshInterval = 30000, // 30 seconds
  } = options;

  // State
  const [stats, setStats] = useState(null);
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [connected, setConnected] = useState(false);

  // Refs for cleanup
  const intervalRef = useRef(null);
  const mountedRef = useRef(true);

  /**
   * Fetch health status
   */
  const fetchHealth = useCallback(async () => {
    try {
      const healthData = await ragService.checkHealth();
      
      if (mountedRef.current) {
        setHealth(healthData);
        setConnected(healthData.status === 'healthy');
        setError(null);
      }
      
      return healthData;
    } catch (err) {
      if (mountedRef.current) {
        setConnected(false);
        setError(err.message);
        logError(err, 'Health Check');
      }
      throw err;
    }
  }, []);

  /**
   * Fetch statistics
   */
  const fetchStats = useCallback(async () => {
    try {
      const statsData = await ragService.getStats();
      
      if (mountedRef.current) {
        setStats(statsData);
        setError(null);
      }
      
      return statsData;
    } catch (err) {
      if (mountedRef.current) {
        setError(err.message);
        logError(err, 'Fetch Stats');
      }
      throw err;
    }
  }, []);

  /**
   * Refresh all data
   */
  const refreshAll = useCallback(async () => {
    if (loading) return;

    setLoading(true);
    
    try {
      await Promise.all([
        fetchHealth(),
        fetchStats(),
      ]);
    } catch (err) {
      // Errors already handled in individual functions
    } finally {
      if (mountedRef.current) {
        setLoading(false);
      }
    }
  }, [loading, fetchHealth, fetchStats]);

  /**
   * Start auto-refresh polling
   */
  const startPolling = useCallback((interval = refreshInterval) => {
    stopPolling();

    console.log(`Starting stats polling every ${interval}ms`);
    
    // Initial fetch
    refreshAll();

    // Set up interval
    intervalRef.current = setInterval(() => {
      refreshAll();
    }, interval);
  }, [refreshInterval, refreshAll]);

  /**
   * Stop auto-refresh polling
   */
  const stopPolling = useCallback(() => {
    if (intervalRef.current) {
      console.log('Stopping stats polling');
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  /**
   * Check connection status
   */
  const checkConnection = useCallback(async () => {
    try {
      const isReachable = await ragService.isBackendReachable();
      if (mountedRef.current) {
        setConnected(isReachable);
      }
      return isReachable;
    } catch (err) {
      if (mountedRef.current) {
        setConnected(false);
      }
      return false;
    }
  }, []);

  // Initial load
  useEffect(() => {
    refreshAll();
  }, [refreshAll]);

  // Auto-refresh setup
  useEffect(() => {
    if (autoRefresh) {
      startPolling(refreshInterval);
    }

    return () => {
      stopPolling();
    };
  }, [autoRefresh, refreshInterval, startPolling, stopPolling]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      mountedRef.current = false;
      stopPolling();
    };
  }, [stopPolling]);

  /**
   * Get formatted stats for display
   */
  const getFormattedStats = useCallback(() => {
    if (!stats || !health) {
      return null;
    }

    return {
      backend: {
        status: health.status,
        connected: connected,
        pipelineInitialized: health.pipeline_initialized,
        apiConfigured: health.groq_api_configured,
      },
      papers: {
        total: health.num_indexed_papers || 0,
        vectorStoreSize: stats.vector_store_size || 0,
      },
      queries: {
        total: stats.total_queries || 0,
        rerankerEnabled: stats.reranker_enabled || false,
      },
    };
  }, [stats, health, connected]);

  /**
   * Check if backend is ready
   */
  const isBackendReady = useCallback(() => {
    return connected && 
           health?.status === 'healthy' && 
           health?.pipeline_initialized && 
           health?.groq_api_configured;
  }, [connected, health]);

  /**
   * Get status message
   */
  const getStatusMessage = useCallback(() => {
    if (!connected) {
      return 'Backend not connected. Please start the API server.';
    }

    if (health?.status !== 'healthy') {
      return 'Backend is degraded. Check server logs.';
    }

    if (!health?.pipeline_initialized) {
      return 'RAG pipeline not initialized. Check GROQ_API_KEY.';
    }

    if (!health?.groq_api_configured) {
      return 'GROQ API not configured. Set GROQ_API_KEY in environment.';
    }

    return 'Backend is ready';
  }, [connected, health]);

  return {
    // Raw data
    stats,
    health,

    // Status
    loading,
    error,
    connected,

    // Actions
    refreshAll,
    refreshStats: fetchStats,
    refreshHealth: fetchHealth,
    startPolling,
    stopPolling,
    checkConnection,

    // Utilities
    getFormattedStats,
    isBackendReady,
    getStatusMessage,

    // Computed values
    isReady: isBackendReady(),
    statusMessage: getStatusMessage(),
    totalQueries: stats?.total_queries || 0,
    indexedPapers: health?.num_indexed_papers || 0,
    rerankerEnabled: stats?.reranker_enabled || false,
  };
}

export default useStats;

