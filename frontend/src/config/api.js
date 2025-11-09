/**
 * API Configuration
 * 
 * Central configuration for API endpoints and base URL
 */

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export const API_ENDPOINTS = {
  HEALTH: `${API_BASE_URL}/api/health`,
  STATS: `${API_BASE_URL}/api/stats`,
  QUERY: `${API_BASE_URL}/api/query`,
};

export const API_CONFIG = {
  BASE_URL: API_BASE_URL,
  TIMEOUT: 30000, // 30 seconds
  STREAM_TIMEOUT: 300000, // 5 minutes for streaming
};

export default API_BASE_URL;

