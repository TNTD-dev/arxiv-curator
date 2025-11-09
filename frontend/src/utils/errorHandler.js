/**
 * Error Handler Utility
 * 
 * Centralized error handling for API calls
 */

/**
 * Error types
 */
export const ErrorTypes = {
  NETWORK: 'NETWORK_ERROR',
  SERVER: 'SERVER_ERROR',
  VALIDATION: 'VALIDATION_ERROR',
  TIMEOUT: 'TIMEOUT_ERROR',
  CANCELLED: 'CANCELLED_ERROR',
  UNKNOWN: 'UNKNOWN_ERROR',
};

/**
 * Parse error and extract relevant information
 * @param {Error} error - Error object
 * @returns {object} Parsed error
 */
export function parseError(error) {
  // Network errors
  if (error.message?.includes('Network Error') || error.message?.includes('No response from server')) {
    return {
      type: ErrorTypes.NETWORK,
      message: 'Cannot connect to the server. Please ensure the backend is running.',
      originalError: error,
    };
  }

  // Timeout errors
  if (error.code === 'ECONNABORTED' || error.message?.includes('timeout')) {
    return {
      type: ErrorTypes.TIMEOUT,
      message: 'Request timed out. The server is taking too long to respond.',
      originalError: error,
    };
  }

  // Cancelled errors
  if (error.message?.includes('cancelled') || error.message?.includes('aborted')) {
    return {
      type: ErrorTypes.CANCELLED,
      message: 'Request was cancelled.',
      originalError: error,
    };
  }

  // Server errors
  if (error.response) {
    return {
      type: ErrorTypes.SERVER,
      message: error.response.data?.detail || error.response.statusText || 'Server error occurred.',
      status: error.response.status,
      originalError: error,
    };
  }

  // Validation errors
  if (error.message?.includes('validation') || error.message?.includes('invalid')) {
    return {
      type: ErrorTypes.VALIDATION,
      message: error.message,
      originalError: error,
    };
  }

  // Unknown errors
  return {
    type: ErrorTypes.UNKNOWN,
    message: error.message || 'An unexpected error occurred.',
    originalError: error,
  };
}

/**
 * Get user-friendly error message
 * @param {Error} error - Error object
 * @returns {string} User-friendly message
 */
export function getUserFriendlyMessage(error) {
  const parsed = parseError(error);
  return parsed.message;
}

/**
 * Check if error is retryable
 * @param {Error} error - Error object
 * @returns {boolean} Whether error is retryable
 */
export function isRetryable(error) {
  const parsed = parseError(error);
  return [ErrorTypes.NETWORK, ErrorTypes.TIMEOUT].includes(parsed.type);
}

/**
 * Log error with context
 * @param {Error} error - Error object
 * @param {string} context - Error context
 */
export function logError(error, context = '') {
  const parsed = parseError(error);
  
  console.group(`🔴 Error${context ? ` [${context}]` : ''}`);
  console.error('Type:', parsed.type);
  console.error('Message:', parsed.message);
  if (parsed.status) {
    console.error('Status:', parsed.status);
  }
  console.error('Original Error:', parsed.originalError);
  console.groupEnd();
}

/**
 * Create error notification object
 * @param {Error} error - Error object
 * @param {string} context - Error context
 * @returns {object} Notification object
 */
export function createErrorNotification(error, context = '') {
  const parsed = parseError(error);
  
  return {
    type: 'error',
    title: context || 'Error',
    message: parsed.message,
    duration: parsed.type === ErrorTypes.NETWORK ? 0 : 5000, // Keep network errors visible
    action: isRetryable(error) ? 'Retry' : null,
  };
}

/**
 * Handle async errors with retry
 * @param {Function} fn - Async function to execute
 * @param {number} retries - Number of retries
 * @param {number} delay - Delay between retries (ms)
 * @returns {Promise} Result of function
 */
export async function withRetry(fn, retries = 3, delay = 1000) {
  let lastError;

  for (let i = 0; i < retries; i++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      
      if (!isRetryable(error) || i === retries - 1) {
        throw error;
      }

      console.log(`Retry attempt ${i + 1}/${retries} after ${delay}ms`);
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }

  throw lastError;
}

export default {
  ErrorTypes,
  parseError,
  getUserFriendlyMessage,
  isRetryable,
  logError,
  createErrorNotification,
  withRetry,
};

