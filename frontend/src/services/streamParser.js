/**
 * SSE (Server-Sent Events) Stream Parser
 * 
 * Utilities for parsing and handling streaming responses from the backend
 */

/**
 * Parse SSE data line
 * @param {string} line - SSE formatted line
 * @returns {object|null} - Parsed event object or null
 */
export function parseSSELine(line) {
  if (!line || line.trim() === '') {
    return null;
  }

  // Parse event type (e.g., "event: status")
  if (line.startsWith('event:')) {
    return {
      type: 'event',
      value: line.substring(6).trim()
    };
  }

  // Parse data (e.g., "data: {...}")
  if (line.startsWith('data:')) {
    const dataStr = line.substring(5).trim();
    try {
      return {
        type: 'data',
        value: JSON.parse(dataStr)
      };
    } catch (e) {
      console.error('Failed to parse SSE data:', dataStr, e);
      return {
        type: 'data',
        value: dataStr
      };
    }
  }

  return null;
}

/**
 * Create SSE event handler
 * @param {object} callbacks - Event callbacks
 * @returns {function} - Event handler function
 */
export function createSSEHandler(callbacks = {}) {
  const {
    onStatus,
    onContexts,
    onAnswerChunk,
    onDone,
    onError,
    onComplete
  } = callbacks;

  let currentEvent = null;
  let answer = '';

  return (event) => {
    const lines = event.data.split('\n');
    
    lines.forEach(line => {
      const parsed = parseSSELine(line);
      if (!parsed) return;

      if (parsed.type === 'event') {
        currentEvent = parsed.value;
      } else if (parsed.type === 'data' && currentEvent) {
        const data = parsed.value;

        switch (currentEvent) {
          case 'status':
            if (onStatus) {
              onStatus(data.status);
            }
            break;

          case 'contexts':
            if (onContexts) {
              onContexts(data);
            }
            break;

          case 'answer':
            if (data.chunk) {
              answer += data.chunk;
              if (onAnswerChunk) {
                onAnswerChunk(data.chunk, answer);
              }
            }
            break;

          case 'done':
            if (onDone) {
              onDone(data, answer);
            }
            if (onComplete) {
              onComplete();
            }
            break;

          case 'error':
            if (onError) {
              onError(data.error || 'Unknown error');
            }
            if (onComplete) {
              onComplete();
            }
            break;

          default:
            console.warn('Unknown SSE event type:', currentEvent);
        }
      }
    });
  };
}

/**
 * Create EventSource with error handling
 * @param {string} url - SSE endpoint URL
 * @param {object} callbacks - Event callbacks
 * @returns {EventSource} - EventSource instance
 */
export function createEventSource(url, callbacks = {}) {
  const eventSource = new EventSource(url);

  // Handle message events
  const handler = createSSEHandler(callbacks);
  eventSource.onmessage = handler;

  // Handle connection opened
  eventSource.onopen = () => {
    console.log('SSE connection opened');
    if (callbacks.onOpen) {
      callbacks.onOpen();
    }
  };

  // Handle errors
  eventSource.onerror = (error) => {
    console.error('SSE connection error:', error);
    if (callbacks.onError) {
      callbacks.onError('Connection error');
    }
    if (callbacks.onComplete) {
      callbacks.onComplete();
    }
  };

  return eventSource;
}

/**
 * Accumulate streaming chunks
 */
export class StreamAccumulator {
  constructor() {
    this.reset();
  }

  reset() {
    this.answer = '';
    this.contexts = [];
    this.metadata = null;
    this.status = 'idle';
  }

  setStatus(status) {
    this.status = status;
  }

  setContexts(contexts) {
    this.contexts = contexts;
  }

  appendAnswer(chunk) {
    this.answer += chunk;
    return this.answer;
  }

  setMetadata(metadata) {
    this.metadata = metadata;
  }

  getResult() {
    return {
      answer: this.answer,
      contexts: this.contexts,
      metadata: this.metadata,
      status: this.status
    };
  }
}

export default {
  parseSSELine,
  createSSEHandler,
  createEventSource,
  StreamAccumulator
};

