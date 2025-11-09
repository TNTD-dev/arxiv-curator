/**
 * useSettings Hook
 * 
 * Manage query settings with localStorage persistence
 */

import { useState, useEffect, useCallback } from 'react';
import { getItem, setItem, StorageKeys } from '../utils/storage';

// Default settings
const DEFAULT_SETTINGS = {
  topK: 20,
  useReranker: true,
  mode: 'default',
  streamEnabled: true,
};

// Valid modes
const VALID_MODES = ['default', 'technical', 'beginner_friendly'];

/**
 * Validate settings
 * @param {object} settings - Settings to validate
 * @returns {object} Validated settings
 */
function validateSettings(settings) {
  const validated = { ...DEFAULT_SETTINGS };

  if (settings.topK !== undefined) {
    validated.topK = Math.min(Math.max(settings.topK, 5), 50);
  }

  if (settings.useReranker !== undefined) {
    validated.useReranker = Boolean(settings.useReranker);
  }

  if (settings.mode !== undefined && VALID_MODES.includes(settings.mode)) {
    validated.mode = settings.mode;
  }

  if (settings.streamEnabled !== undefined) {
    validated.streamEnabled = Boolean(settings.streamEnabled);
  }

  return validated;
}

/**
 * useSettings Hook
 * @returns {object} Settings state and functions
 */
export function useSettings() {
  // Initialize from localStorage or defaults
  const [settings, setSettingsState] = useState(() => {
    const stored = getItem(StorageKeys.SETTINGS);
    return stored ? validateSettings(stored) : DEFAULT_SETTINGS;
  });

  // Save to localStorage whenever settings change
  useEffect(() => {
    setItem(StorageKeys.SETTINGS, settings);
  }, [settings]);

  /**
   * Update settings
   * @param {object} updates - Settings updates
   */
  const updateSettings = useCallback((updates) => {
    setSettingsState(prevSettings => {
      const newSettings = { ...prevSettings, ...updates };
      return validateSettings(newSettings);
    });
  }, []);

  /**
   * Update individual setting
   * @param {string} key - Setting key
   * @param {*} value - Setting value
   */
  const setSetting = useCallback((key, value) => {
    updateSettings({ [key]: value });
  }, [updateSettings]);

  /**
   * Reset to default settings
   */
  const resetSettings = useCallback(() => {
    setSettingsState(DEFAULT_SETTINGS);
  }, []);

  /**
   * Get mode display name
   * @param {string} mode - Mode key
   * @returns {string} Display name
   */
  const getModeDisplayName = useCallback((mode) => {
    const names = {
      default: 'Default',
      technical: 'Technical',
      beginner_friendly: 'Beginner Friendly',
    };
    return names[mode] || mode;
  }, []);

  /**
   * Get mode description
   * @param {string} mode - Mode key
   * @returns {string} Description
   */
  const getModeDescription = useCallback((mode) => {
    const descriptions = {
      default: 'Balanced explanations for general audience',
      technical: 'Detailed technical information with jargon',
      beginner_friendly: 'Simple explanations avoiding complex terms',
    };
    return descriptions[mode] || '';
  }, []);

  return {
    // State
    settings,
    topK: settings.topK,
    useReranker: settings.useReranker,
    mode: settings.mode,
    streamEnabled: settings.streamEnabled,

    // Actions
    updateSettings,
    setSetting,
    resetSettings,

    // Utilities
    getModeDisplayName,
    getModeDescription,
    validModes: VALID_MODES,
    defaultSettings: DEFAULT_SETTINGS,
  };
}

export default useSettings;

