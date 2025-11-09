/**
 * LocalStorage Utility
 * 
 * Wrapper for localStorage with JSON serialization and error handling
 */

const STORAGE_PREFIX = 'arxiv_curator_';

/**
 * Storage keys
 */
export const StorageKeys = {
  SETTINGS: 'settings',
  CONVERSATIONS: 'conversations',
  CURRENT_CONVERSATION: 'current_conversation',
  THEME: 'theme',
  USER_PREFERENCES: 'user_preferences',
};

/**
 * Check if localStorage is available
 * @returns {boolean}
 */
export function isStorageAvailable() {
  try {
    const test = '__storage_test__';
    localStorage.setItem(test, test);
    localStorage.removeItem(test);
    return true;
  } catch (e) {
    return false;
  }
}

/**
 * Get prefixed key
 * @param {string} key - Storage key
 * @returns {string} Prefixed key
 */
function getPrefixedKey(key) {
  return `${STORAGE_PREFIX}${key}`;
}

/**
 * Get item from localStorage
 * @param {string} key - Storage key
 * @param {*} defaultValue - Default value if not found
 * @returns {*} Stored value or default
 */
export function getItem(key, defaultValue = null) {
  if (!isStorageAvailable()) {
    console.warn('localStorage is not available');
    return defaultValue;
  }

  try {
    const prefixedKey = getPrefixedKey(key);
    const item = localStorage.getItem(prefixedKey);
    
    if (item === null) {
      return defaultValue;
    }

    return JSON.parse(item);
  } catch (error) {
    console.error(`Error getting item from localStorage: ${key}`, error);
    return defaultValue;
  }
}

/**
 * Set item in localStorage
 * @param {string} key - Storage key
 * @param {*} value - Value to store
 * @returns {boolean} Success status
 */
export function setItem(key, value) {
  if (!isStorageAvailable()) {
    console.warn('localStorage is not available');
    return false;
  }

  try {
    const prefixedKey = getPrefixedKey(key);
    const serialized = JSON.stringify(value);
    localStorage.setItem(prefixedKey, serialized);
    return true;
  } catch (error) {
    if (error.name === 'QuotaExceededError') {
      console.error('localStorage quota exceeded. Consider clearing old data.');
    } else {
      console.error(`Error setting item in localStorage: ${key}`, error);
    }
    return false;
  }
}

/**
 * Remove item from localStorage
 * @param {string} key - Storage key
 * @returns {boolean} Success status
 */
export function removeItem(key) {
  if (!isStorageAvailable()) {
    console.warn('localStorage is not available');
    return false;
  }

  try {
    const prefixedKey = getPrefixedKey(key);
    localStorage.removeItem(prefixedKey);
    return true;
  } catch (error) {
    console.error(`Error removing item from localStorage: ${key}`, error);
    return false;
  }
}

/**
 * Clear all items with prefix
 * @returns {boolean} Success status
 */
export function clear() {
  if (!isStorageAvailable()) {
    console.warn('localStorage is not available');
    return false;
  }

  try {
    const keys = Object.keys(localStorage);
    keys.forEach(key => {
      if (key.startsWith(STORAGE_PREFIX)) {
        localStorage.removeItem(key);
      }
    });
    return true;
  } catch (error) {
    console.error('Error clearing localStorage', error);
    return false;
  }
}

/**
 * Get all keys with prefix
 * @returns {string[]} Array of keys
 */
export function getAllKeys() {
  if (!isStorageAvailable()) {
    return [];
  }

  try {
    const keys = Object.keys(localStorage);
    return keys
      .filter(key => key.startsWith(STORAGE_PREFIX))
      .map(key => key.substring(STORAGE_PREFIX.length));
  } catch (error) {
    console.error('Error getting keys from localStorage', error);
    return [];
  }
}

/**
 * Get storage size in bytes
 * @returns {number} Size in bytes
 */
export function getStorageSize() {
  if (!isStorageAvailable()) {
    return 0;
  }

  try {
    let size = 0;
    const keys = Object.keys(localStorage);
    
    keys.forEach(key => {
      if (key.startsWith(STORAGE_PREFIX)) {
        const item = localStorage.getItem(key);
        if (item) {
          size += item.length + key.length;
        }
      }
    });

    return size;
  } catch (error) {
    console.error('Error calculating storage size', error);
    return 0;
  }
}

/**
 * Format storage size for display
 * @param {number} bytes - Size in bytes
 * @returns {string} Formatted size
 */
export function formatStorageSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

/**
 * Update item partially (merge with existing)
 * @param {string} key - Storage key
 * @param {object} updates - Updates to merge
 * @returns {boolean} Success status
 */
export function updateItem(key, updates) {
  const current = getItem(key, {});
  const updated = { ...current, ...updates };
  return setItem(key, updated);
}

/**
 * Get item with expiration
 * @param {string} key - Storage key
 * @param {*} defaultValue - Default value
 * @returns {*} Stored value or default
 */
export function getItemWithExpiry(key, defaultValue = null) {
  const item = getItem(key);
  
  if (!item) {
    return defaultValue;
  }

  const now = new Date().getTime();
  
  if (item.expiry && now > item.expiry) {
    removeItem(key);
    return defaultValue;
  }

  return item.value;
}

/**
 * Set item with expiration
 * @param {string} key - Storage key
 * @param {*} value - Value to store
 * @param {number} ttl - Time to live in milliseconds
 * @returns {boolean} Success status
 */
export function setItemWithExpiry(key, value, ttl) {
  const now = new Date().getTime();
  const item = {
    value: value,
    expiry: now + ttl,
  };
  return setItem(key, item);
}

export default {
  StorageKeys,
  isStorageAvailable,
  getItem,
  setItem,
  removeItem,
  clear,
  getAllKeys,
  getStorageSize,
  formatStorageSize,
  updateItem,
  getItemWithExpiry,
  setItemWithExpiry,
};

