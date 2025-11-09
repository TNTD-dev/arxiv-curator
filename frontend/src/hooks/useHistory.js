/**
 * useHistory Hook
 * 
 * Manage conversation history with localStorage persistence
 */

import { useState, useEffect, useCallback } from 'react';
import { getItem, setItem, StorageKeys } from '../utils/storage';

/**
 * Generate unique ID
 * @returns {string} Unique ID
 */
function generateId() {
  return `${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Create new conversation
 * @param {string} title - Conversation title
 * @returns {object} New conversation
 */
function createNewConversation(title = 'New Conversation') {
  return {
    id: generateId(),
    title,
    messages: [],
    timestamp: Date.now(),
    pinned: false,
    archived: false,
    updatedAt: Date.now(),
  };
}

/**
 * Generate title from first message
 * @param {string} message - Message content
 * @returns {string} Generated title
 */
function generateTitle(message) {
  const maxLength = 50;
  const cleaned = message.trim().replace(/\s+/g, ' ');
  
  if (cleaned.length <= maxLength) {
    return cleaned;
  }
  
  return cleaned.substring(0, maxLength) + '...';
}

/**
 * useHistory Hook
 * @returns {object} History state and functions
 */
export function useHistory() {
  // Load conversations from localStorage
  const [conversations, setConversations] = useState(() => {
    const stored = getItem(StorageKeys.CONVERSATIONS, []);
    return Array.isArray(stored) ? stored : [];
  });

  // Current active conversation ID
  const [currentConversationId, setCurrentConversationId] = useState(() => {
    return getItem(StorageKeys.CURRENT_CONVERSATION, null);
  });

  // Save conversations to localStorage
  useEffect(() => {
    setItem(StorageKeys.CONVERSATIONS, conversations);
  }, [conversations]);

  // Save current conversation ID
  useEffect(() => {
    setItem(StorageKeys.CURRENT_CONVERSATION, currentConversationId);
  }, [currentConversationId]);

  /**
   * Get current conversation
   */
  const currentConversation = conversations.find(c => c.id === currentConversationId) || null;

  /**
   * Create new conversation
   * @param {string} title - Optional title
   * @returns {string} New conversation ID
   */
  const createConversation = useCallback((title) => {
    const newConv = createNewConversation(title);
    setConversations(prev => [newConv, ...prev]);
    setCurrentConversationId(newConv.id);
    return newConv.id;
  }, []);

  /**
   * Delete conversation
   * @param {string} id - Conversation ID
   */
  const deleteConversation = useCallback((id) => {
    setConversations(prev => prev.filter(c => c.id !== id));
    
    // If deleting current conversation, switch to most recent
    if (id === currentConversationId) {
      setConversations(prev => {
        const remaining = prev.filter(c => c.id !== id);
        if (remaining.length > 0) {
          setCurrentConversationId(remaining[0].id);
        } else {
          setCurrentConversationId(null);
        }
        return remaining;
      });
    }
  }, [currentConversationId]);

  /**
   * Add message to conversation
   * @param {string} conversationId - Conversation ID
   * @param {object} message - Message object
   */
  const addMessage = useCallback((conversationId, message) => {
    setConversations(prev => prev.map(conv => {
      if (conv.id === conversationId) {
        const messages = [...conv.messages, {
          ...message,
          id: generateId(),
          timestamp: Date.now(),
        }];

        // Auto-generate title from first user message if still default
        let title = conv.title;
        if (title === 'New Conversation' && messages.length === 1 && message.role === 'user') {
          title = generateTitle(message.content);
        }

        return {
          ...conv,
          messages,
          title,
          updatedAt: Date.now(),
        };
      }
      return conv;
    }));
  }, []);

  /**
   * Update message in conversation
   * @param {string} conversationId - Conversation ID
   * @param {string} messageId - Message ID
   * @param {object} updates - Message updates
   */
  const updateMessage = useCallback((conversationId, messageId, updates) => {
    setConversations(prev => prev.map(conv => {
      if (conv.id === conversationId) {
        return {
          ...conv,
          messages: conv.messages.map(msg =>
            msg.id === messageId ? { ...msg, ...updates } : msg
          ),
          updatedAt: Date.now(),
        };
      }
      return conv;
    }));
  }, []);

  /**
   * Clear all messages in conversation
   * @param {string} conversationId - Conversation ID
   */
  const clearMessages = useCallback((conversationId) => {
    setConversations(prev => prev.map(conv => {
      if (conv.id === conversationId) {
        return {
          ...conv,
          messages: [],
          updatedAt: Date.now(),
        };
      }
      return conv;
    }));
  }, []);

  /**
   * Pin/unpin conversation
   * @param {string} id - Conversation ID
   */
  const togglePin = useCallback((id) => {
    setConversations(prev => {
      const updated = prev.map(conv => {
        if (conv.id === id) {
          return { ...conv, pinned: !conv.pinned };
        }
        return conv;
      });

      // Sort: pinned first, then by updatedAt
      return updated.sort((a, b) => {
        if (a.pinned && !b.pinned) return -1;
        if (!a.pinned && b.pinned) return 1;
        return b.updatedAt - a.updatedAt;
      });
    });
  }, []);

  /**
   * Archive/unarchive conversation
   * @param {string} id - Conversation ID
   */
  const toggleArchive = useCallback((id) => {
    setConversations(prev => prev.map(conv => {
      if (conv.id === id) {
        return { ...conv, archived: !conv.archived };
      }
      return conv;
    }));
  }, []);

  /**
   * Rename conversation
   * @param {string} id - Conversation ID
   * @param {string} newTitle - New title
   */
  const renameConversation = useCallback((id, newTitle) => {
    setConversations(prev => prev.map(conv => {
      if (conv.id === id) {
        return { ...conv, title: newTitle };
      }
      return conv;
    }));
  }, []);

  /**
   * Set current conversation
   * @param {string} id - Conversation ID
   */
  const setCurrentConversation = useCallback((id) => {
    setCurrentConversationId(id);
  }, []);

  /**
   * Get filtered conversations
   * @param {object} filters - Filter options
   * @returns {array} Filtered conversations
   */
  const getFilteredConversations = useCallback((filters = {}) => {
    let filtered = [...conversations];

    if (filters.archived !== undefined) {
      filtered = filtered.filter(c => c.archived === filters.archived);
    }

    if (filters.pinned !== undefined) {
      filtered = filtered.filter(c => c.pinned === filters.pinned);
    }

    if (filters.search) {
      const searchLower = filters.search.toLowerCase();
      filtered = filtered.filter(c => 
        c.title.toLowerCase().includes(searchLower) ||
        c.messages.some(m => m.content.toLowerCase().includes(searchLower))
      );
    }

    return filtered;
  }, [conversations]);

  /**
   * Clear all conversations
   */
  const clearAll = useCallback(() => {
    setConversations([]);
    setCurrentConversationId(null);
  }, []);

  /**
   * Export conversations
   * @returns {string} JSON string of conversations
   */
  const exportConversations = useCallback(() => {
    return JSON.stringify(conversations, null, 2);
  }, [conversations]);

  /**
   * Import conversations
   * @param {string} jsonString - JSON string of conversations
   * @returns {boolean} Success status
   */
  const importConversations = useCallback((jsonString) => {
    try {
      const imported = JSON.parse(jsonString);
      if (Array.isArray(imported)) {
        setConversations(imported);
        return true;
      }
      return false;
    } catch (error) {
      console.error('Failed to import conversations:', error);
      return false;
    }
  }, []);

  return {
    // State
    conversations,
    currentConversation,
    currentConversationId,

    // Conversation actions
    createConversation,
    deleteConversation,
    setCurrentConversation,
    renameConversation,
    togglePin,
    toggleArchive,

    // Message actions
    addMessage,
    updateMessage,
    clearMessages,

    // Utilities
    getFilteredConversations,
    clearAll,
    exportConversations,
    importConversations,

    // Computed values
    totalConversations: conversations.length,
    activeConversations: conversations.filter(c => !c.archived).length,
    archivedConversations: conversations.filter(c => c.archived).length,
    pinnedConversations: conversations.filter(c => c.pinned).length,
  };
}

export default useHistory;

