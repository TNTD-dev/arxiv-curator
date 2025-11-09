import React, { useState, useEffect, useRef } from 'react';
import AuthModal from './AuthModal';
import { 
  MessageSquare, 
  Moon, 
  Sun, 
  Send, 
  BookOpen, 
  Zap, 
  Search, 
  ExternalLink,
  FileText,
  ArrowRight,
  Sparkles,
  Brain,
  Database,
  ChevronDown,
  Star,
  Users,
  TrendingUp,
  Shield,
  Rocket,
  Globe,
  ChevronRight,
  Plus,
  Trash2,
  ThumbsUp,
  ThumbsDown,
  MessageCircle,
  X,
  Eye,
  Clock,
  Quote,
  PanelRight,
  PanelRightClose,
  PanelLeft,
  PanelLeftClose,
  HelpCircle,
  Lightbulb,
  User,
  Settings,
  LogOut,
  Edit3,
  Save,
  Copy,
  Check,
  Camera,
  Mail,
  MapPin,
  Calendar,
  Award,
  BarChart3,
  Menu,
  History,
  Filter,
  SortAsc,
  SortDesc,
  MoreVertical,
  Archive,
  Pin,
  Tag,
  AlertCircle,
  Sliders
} from 'lucide-react';
import './App.css';

// Phase 10 Hooks Integration
import { useRAG, useSettings as useRAGSettings, useStats } from './hooks';
import useHistory from './hooks/useHistory';

// ConversationItem Component
const ConversationItem = ({ conversation, isActive, onSelect, onDelete, onPin, onArchive }) => {
  const [showMenu, setShowMenu] = useState(false);
  
  const formatTime = (timestamp) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now - date;
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
    
    if (diffDays === 0) {
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } else if (diffDays === 1) {
      return 'Yesterday';
    } else if (diffDays < 7) {
      return `${diffDays} days ago`;
    } else {
      return date.toLocaleDateString();
    }
  };

  return (
    <div className={`conversation-item ${isActive ? 'active' : ''} ${conversation.pinned ? 'pinned' : ''} ${conversation.archived ? 'archived' : ''}`}>
      <div className="conversation-content" onClick={onSelect}>
        <div className="conversation-header">
          <h4 className="conversation-title">
            {conversation.pinned && <Pin size={12} className="pin-icon" />}
            {conversation.title}
          </h4>
          <button 
            className="conversation-menu"
            onClick={(e) => {
              e.stopPropagation();
              setShowMenu(!showMenu);
            }}
          >
            <MoreVertical size={14} />
          </button>
        </div>
        
        <p className="conversation-preview">
          {conversation.messages?.[conversation.messages.length - 1]?.content?.slice(0, 80)}...
        </p>
        
        <div className="conversation-footer">
          <span className="conversation-time">
            <Clock size={10} />
            {formatTime(conversation.timestamp)}
          </span>
          <span className="message-count">
            {conversation.messages?.length || 0} messages
          </span>
        </div>
      </div>

      {showMenu && (
        <div className="conversation-menu-dropdown">
          <button 
            className="menu-item"
            onClick={(e) => {
              e.stopPropagation();
              onPin();
              setShowMenu(false);
            }}
          >
            <Pin size={14} />
            {conversation.pinned ? 'Unpin' : 'Pin'}
          </button>
          <button 
            className="menu-item"
            onClick={(e) => {
              e.stopPropagation();
              onArchive();
              setShowMenu(false);
            }}
          >
            <Archive size={14} />
            {conversation.archived ? 'Unarchive' : 'Archive'}
          </button>
          <button 
            className="menu-item delete"
            onClick={(e) => {
              e.stopPropagation();
              onDelete();
              setShowMenu(false);
            }}
          >
            <Trash2 size={14} />
            Delete
          </button>
        </div>
      )}
    </div>
  );
};

// Settings Modal Component
const SettingsModal = ({ isOpen, onClose, darkMode }) => {
  const settings = useRAGSettings();
  
  if (!isOpen) return null;
  
  return (
    <div className="profile-modal">
      <div className="profile-form">
        <div className="profile-header">
          <h2><Sliders size={20} /> Query Settings</h2>
          <button onClick={onClose}>
            <X size={20} />
          </button>
        </div>
        
        <div className="profile-content">
          <div className="setting-group" style={{ marginBottom: '1.5rem' }}>
            <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '600' }}>
              Number of Contexts (Top-K): {settings.topK}
            </label>
            <input 
              type="range" 
              min="5" 
              max="50" 
              value={settings.topK}
              onChange={(e) => settings.setSetting('topK', parseInt(e.target.value))}
              style={{ width: '100%' }}
            />
            <span style={{ fontSize: '0.875rem', color: '#6b7280' }}>
              Higher values retrieve more papers (slower but more comprehensive)
            </span>
          </div>
          
          <div className="setting-group" style={{ marginBottom: '1.5rem' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
              <input 
                type="checkbox"
                checked={settings.useReranker}
                onChange={(e) => settings.setSetting('useReranker', e.target.checked)}
              />
              <span style={{ fontWeight: '600' }}>Enable Re-ranking (Cross-encoder)</span>
            </label>
            <span style={{ fontSize: '0.875rem', color: '#6b7280', marginLeft: '1.5rem' }}>
              Improves relevance by re-ranking results (slower but more accurate)
            </span>
          </div>
          
          <div className="setting-group" style={{ marginBottom: '1.5rem' }}>
            <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '600' }}>
              Response Mode
            </label>
            <select 
              value={settings.mode}
              onChange={(e) => settings.setSetting('mode', e.target.value)}
              style={{ width: '100%', padding: '0.5rem', borderRadius: '8px', border: '1px solid #d1d5db' }}
            >
              <option value="default">Default - Balanced explanations</option>
              <option value="technical">Technical - Detailed with jargon</option>
              <option value="beginner_friendly">Beginner Friendly - Simple explanations</option>
            </select>
            <span style={{ fontSize: '0.875rem', color: '#6b7280', marginTop: '0.25rem', display: 'block' }}>
              {settings.getModeDescription(settings.mode)}
            </span>
          </div>
          
          <div className="setting-group" style={{ marginBottom: '1.5rem' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
              <input 
                type="checkbox"
                checked={settings.streamEnabled}
                onChange={(e) => settings.setSetting('streamEnabled', e.target.checked)}
              />
              <span style={{ fontWeight: '600' }}>Enable Streaming Responses</span>
            </label>
            <span style={{ fontSize: '0.875rem', color: '#6b7280', marginLeft: '1.5rem' }}>
              Show responses in real-time as they're generated
            </span>
          </div>
        </div>
        
        <div className="profile-actions">
          <button 
            className="btn-secondary"
            onClick={settings.resetSettings}
          >
            Reset to Defaults
          </button>
          <button 
            className="btn-primary"
            onClick={onClose}
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
};

function App() {
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('darkMode');
    return saved ? JSON.parse(saved) : false;
  });
  
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [currentView, setCurrentView] = useState('landing'); // 'landing' or 'chat'
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(() => {
    const savedAuth = localStorage.getItem('isAuthenticated');
    return savedAuth === 'true';
  });
  const [isScrolled, setIsScrolled] = useState(false);
  const [conversations, setConversations] = useState(() => {
    const saved = localStorage.getItem('conversations');
    return saved ? JSON.parse(saved) : [];
  });
  const [currentConversationId, setCurrentConversationId] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [conversationFilter, setConversationFilter] = useState('all'); // 'all', 'today', 'week', 'month'
  const [sortBy, setSortBy] = useState('recent'); // 'recent', 'oldest', 'alphabetical'
  const [showCitationPanel, setShowCitationPanel] = useState(true);
  const [selectedCitation, setSelectedCitation] = useState(null);
  const [showFeedbackForm, setShowFeedbackForm] = useState(null);
  const [showOnboarding, setShowOnboarding] = useState(() => {
    const hasSeenOnboarding = localStorage.getItem('hasSeenOnboarding');
    return !hasSeenOnboarding;
  });
  const [userProfile, setUserProfile] = useState(() => {
    const saved = localStorage.getItem('userProfile');
    return saved ? JSON.parse(saved) : {
      name: 'Research Scholar',
      email: 'scholar@university.edu',
      institution: 'Research University',
      role: 'PhD Student',
      specialization: 'Machine Learning',
      joinDate: new Date().toISOString().split('T')[0],
      avatar: null,
      preferences: {
        language: 'en',
        citationStyle: 'APA',
        autoSave: true,
        notifications: true
      },
      stats: {
        totalQueries: 0,
        papersRead: 0,
        conversationsStarted: 0,
        feedbackGiven: 0
      }
    };
  });
  const [showProfileModal, setShowProfileModal] = useState(false);
  const [editingProfile, setEditingProfile] = useState(false);
  const [copiedMessageId, setCopiedMessageId] = useState(null);
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [showSettingsModal, setShowSettingsModal] = useState(false);
  const messagesEndRef = useRef(null);
  
  // Phase 10 Hooks Integration
  const ragSettings = useRAGSettings();
  const rag = useRAG();
  const ragHistory = useHistory();
  const stats = useStats({ autoRefresh: true, refreshInterval: 5000 });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', darkMode ? 'dark' : 'light');
    localStorage.setItem('darkMode', JSON.stringify(darkMode));
  }, [darkMode]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 50);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Handle RAG response when query completes
  useEffect(() => {
    if (rag.state === 'done' && rag.answer && currentConversationId) {
      const assistantMessage = {
        id: Date.now() + 1,
        type: 'assistant',
        content: rag.answer,
        contexts: rag.contexts,
        metadata: rag.metadata,
        timestamp: new Date().toISOString(),
        feedback: null
      };
      
      setMessages(prev => {
        const updatedMessages = [...prev, assistantMessage];
        updateConversationMessages(updatedMessages);
        return updatedMessages;
      });
      
      ragHistory.addMessage(currentConversationId, {
        role: 'assistant',
        content: rag.answer,
        contexts: rag.contexts,
        metadata: rag.metadata
      });
      
      // Update user stats
      updateUserStats('totalQueries');
      updateUserStats('papersRead');
      
      // Mark loading as complete
      setIsLoading(false);
    }
  }, [rag.state, rag.answer, rag.contexts, currentConversationId]);

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputValue.trim() || rag.loading) return;

    // Switch to chat view if on landing page
    if (currentView === 'landing') {
      setCurrentView('chat');
      setShowOnboarding(false);
      localStorage.setItem('hasSeenOnboarding', 'true');
    }

    // Create new conversation if none exists using RAG history hook
    let convId = currentConversationId;
    if (!convId) {
      convId = ragHistory.createConversation(inputValue.slice(0, 50));
      setCurrentConversationId(convId);
      
      // Also update local state for UI compatibility
      const newConversation = {
        id: convId,
        title: inputValue.slice(0, 30) + (inputValue.length > 30 ? '...' : ''),
        timestamp: new Date().toISOString(),
        messages: []
      };
      setConversations(prev => [newConversation, ...prev]);
    }

    const question = inputValue;
    
    // Add user message to display
    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: question,
      timestamp: new Date().toISOString()
    };
    setMessages(prev => [...prev, userMessage]);
    
    // Add to RAG history
    ragHistory.addMessage(convId, {
      role: 'user',
      content: question
    });

    setInputValue('');
    setIsLoading(true);

    try {
      // Submit query to RAG backend with settings
      await rag.submitQuery(question, {
        topK: ragSettings.topK,
        mode: ragSettings.mode,
        useReranker: ragSettings.useReranker
      });

      // Note: Response handling moved to useEffect that watches rag.state
      // This prevents accessing stale rag.answer from previous query
      
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        type: 'assistant',
        content: rag.error || 'Sorry, I encountered an error while processing your request. Please try again.',
        contexts: [],
        timestamp: new Date().toISOString(),
        feedback: null
      };
      setMessages(prev => {
        const updatedMessages = [...prev, errorMessage];
        updateConversationMessages(updatedMessages);
        return updatedMessages;
      });
    } finally {
      setIsLoading(false);
    }
  };

  const updateConversationMessages = (updatedMessages) => {
    if (!currentConversationId) return;
    
    setConversations(prev => {
      const updated = prev.map(conv => 
        conv.id === currentConversationId 
          ? { ...conv, messages: updatedMessages, timestamp: new Date().toISOString() }
          : conv
      );
      localStorage.setItem('conversations', JSON.stringify(updated));
      return updated;
    });
  };

  const startNewConversation = () => {
    rag.reset(); // Reset RAG state when starting new conversation
    setCurrentConversationId(null);
    setMessages([]);
    setSelectedCitation(null);
    updateUserStats('conversationsStarted');
  };

  const pinConversation = (conversationId) => {
    setConversations(prev => {
      const updated = prev.map(conv => 
        conv.id === conversationId 
          ? { ...conv, pinned: !conv.pinned }
          : conv
      );
      localStorage.setItem('conversations', JSON.stringify(updated));
      return updated;
    });
  };

  const archiveConversation = (conversationId) => {
    setConversations(prev => {
      const updated = prev.map(conv => 
        conv.id === conversationId 
          ? { ...conv, archived: !conv.archived }
          : conv
      );
      localStorage.setItem('conversations', JSON.stringify(updated));
      return updated;
    });
  };

  const clearSearch = () => {
    setSearchTerm('');
    setConversationFilter('all');
    setSortBy('recent');
  };

  const loadConversation = (conversationId) => {
    const conversation = conversations.find(c => c.id === conversationId);
    if (conversation) {
      rag.reset(); // Reset RAG state when switching conversations
      setCurrentConversationId(conversationId);
      setMessages(conversation.messages || []);
      setSelectedCitation(null);
    }
  };

  const deleteConversation = (conversationId) => {
    setConversations(prev => {
      const updated = prev.filter(c => c.id !== conversationId);
      localStorage.setItem('conversations', JSON.stringify(updated));
      return updated;
    });
    
    if (currentConversationId === conversationId) {
      startNewConversation();
    }
  };

  const handleFeedback = (messageId, feedback) => {
    setMessages(prev => prev.map(msg => 
      msg.id === messageId ? { ...msg, feedback } : msg
    ));
    updateConversationMessages(messages.map(msg => 
      msg.id === messageId ? { ...msg, feedback } : msg
    ));
    setShowFeedbackForm(null);
    
    // Update user stats
    updateUserStats('feedbackGiven');
  };

  const updateUserStats = (statType) => {
    setUserProfile(prev => {
      const updated = {
        ...prev,
        stats: {
          ...prev.stats,
          [statType]: prev.stats[statType] + 1
        }
      };
      localStorage.setItem('userProfile', JSON.stringify(updated));
      return updated;
    });
  };

  const updateUserProfile = (updatedProfile) => {
    setUserProfile(updatedProfile);
    localStorage.setItem('userProfile', JSON.stringify(updatedProfile));
  };

  const copyToClipboard = async (text, messageId) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedMessageId(messageId);
      setTimeout(() => setCopiedMessageId(null), 2000);
    } catch (err) {
      console.error('Failed to copy text: ', err);
    }
  };

  // Enhanced filtering and search
  const getFilteredConversations = () => {
    let filtered = conversations;

    // Apply text search
    if (searchTerm.trim()) {
      filtered = filtered.filter(conv => {
        const titleMatch = conv.title.toLowerCase().includes(searchTerm.toLowerCase());
        const messageMatch = conv.messages?.some(msg => 
          msg.content.toLowerCase().includes(searchTerm.toLowerCase())
        );
        return titleMatch || messageMatch;
      });
    }

    // Apply date filter
    const now = new Date();
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const week = new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000);
    const month = new Date(today.getTime() - 30 * 24 * 60 * 60 * 1000);

    switch (conversationFilter) {
      case 'today':
        filtered = filtered.filter(conv => new Date(conv.timestamp) >= today);
        break;
      case 'week':
        filtered = filtered.filter(conv => new Date(conv.timestamp) >= week);
        break;
      case 'month':
        filtered = filtered.filter(conv => new Date(conv.timestamp) >= month);
        break;
      default:
        break;
    }

    // Apply sorting
    switch (sortBy) {
      case 'oldest':
        filtered.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
        break;
      case 'alphabetical':
        filtered.sort((a, b) => a.title.localeCompare(b.title));
        break;
      default: // 'recent'
        filtered.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
        break;
    }

    return filtered;
  };

  const filteredConversations = getFilteredConversations();

  const exampleQuestions = [
    "What are the latest developments in transformer architectures?",
    "How do diffusion models work for image generation?",
    "What are the key innovations in few-shot learning?",
    "Explain the concept of attention mechanisms in neural networks",
    "What are the differences between BERT and GPT models?",
    "How does reinforcement learning work in language models?",
    "What are the applications of computer vision in healthcare?",
    "Explain quantum computing's potential for machine learning",
    "What are the latest breakthroughs in natural language processing?",
    "How do graph neural networks work?",
    "What are the challenges in federated learning?",
    "Explain the concept of transfer learning in deep learning",
    "What are the latest developments in autonomous driving?",
    "How do generative adversarial networks (GANs) work?",
    "What are the applications of AI in drug discovery?",
    "Explain the concept of explainable AI (XAI)",
    "What are the latest trends in computer vision?",
    "How does multi-modal learning work?",
    "What are the challenges in AI safety and alignment?",
    "Explain the concept of meta-learning",
    "What are the applications of AI in climate change research?",
    "How do neural architecture search methods work?",
    "What are the latest developments in robotics and AI?"
  ];

  const startChat = () => {
    if (isAuthenticated) {
      setCurrentView('chat');
    } else {
      setShowAuthModal(true);
    }
  };

  const goToLanding = () => {
    setCurrentView('landing');
  };

  const openAuthModal = () => {
    setShowAuthModal(true);
  };

  const closeAuthModal = () => {
    setShowAuthModal(false);
  };

  const handleGoogleSignIn = () => {
    // Simulate Google Sign-In (in real app, use Google OAuth)
    setIsAuthenticated(true);
    localStorage.setItem('isAuthenticated', 'true');
    
    // Update user profile with Google data (simulated)
    const googleUserData = {
      name: 'Duc Tran',
      email: 'ductran@uit.edu.vn',
      institution: 'University of Information Technology',
      role: 'Student',
      specialization: 'Computer Science',
      avatar: null, // In real app, get from Google profile
      joinDate: new Date().toISOString().split('T')[0],
      preferences: {
        language: 'en',
        citationStyle: 'APA',
        autoSave: true,
        notifications: true
      },
      stats: {
        totalQueries: 0,
        papersRead: 0,
        conversationsStarted: 0,
        feedbackGiven: 0
      }
    };
    
    updateUserProfile(googleUserData);
    setShowAuthModal(false);
    setCurrentView('chat');
  };

  const handleEmailSignIn = (email, password) => {
    // Simulate email sign-in
    setIsAuthenticated(true);
    localStorage.setItem('isAuthenticated', 'true');
    
    const emailUserData = {
      name: email.split('@')[0].replace('.', ' ').replace(/\b\w/g, l => l.toUpperCase()),
      email: email,
      institution: 'Research Institute',
      role: 'Researcher',
      specialization: 'Machine Learning',
      avatar: null,
      joinDate: new Date().toISOString().split('T')[0],
      preferences: {
        language: 'en',
        citationStyle: 'APA',
        autoSave: true,
        notifications: true
      },
      stats: {
        totalQueries: 0,
        papersRead: 0,
        conversationsStarted: 0,
        feedbackGiven: 0
      }
    };
    
    updateUserProfile(emailUserData);
    setShowAuthModal(false);
    setCurrentView('chat');
  };

  const handleSignOut = () => {
    setIsAuthenticated(false);
    localStorage.setItem('isAuthenticated', 'false');
    setCurrentView('landing');
    setMessages([]);
    setConversations([]);
    setCurrentConversationId(null);
  };

  if (currentView === 'landing') {
    return (
      <div className="app landing-page">
        {/* Navigation */}
        <nav className={`nav ${isScrolled ? 'nav-scrolled' : ''}`}>
          <div className="nav-container">
            <div className="logo">
              <div className="logo-icon">
                <Brain size={28} />
                <Sparkles className="sparkle" size={16} />
              </div>
              <span className="logo-text">arXiv Curator</span>
            </div>
            <div className="nav-actions">
              {isAuthenticated ? (
                <button className="nav-link" onClick={startChat}>Go to Chat</button>
              ) : (
                <button className="nav-link" onClick={openAuthModal}>Sign In</button>
              )}
              <button className="theme-toggle" onClick={toggleDarkMode}>
                {darkMode ? <Sun size={20} /> : <Moon size={20} />}
              </button>
            </div>
          </div>
        </nav>

        {/* Hero Section */}
        <section className="hero">
          <div className="hero-background">
            <div className="floating-particles">
              {[...Array(20)].map((_, i) => (
                <div key={i} className={`particle particle-${i % 4}`} />
              ))}
            </div>
          </div>
          
          <div className="hero-content">
            <div className="hero-badge">
              <Sparkles size={16} />
              <span>Powered by Advanced RAG Technology</span>
            </div>
            
            <h1 className="hero-title">
              Unlock Research
              <span className="gradient-text"> Intelligence</span>
            </h1>
            
            <p className="hero-subtitle">
              Experience the future of academic research with our AI-powered curator that 
              instantly synthesizes insights from thousands of arXiv papers
            </p>
            
            <div className="hero-cta">
              <form onSubmit={handleSubmit} className="hero-search">
                <div className="search-input-wrapper">
                  <Search className="search-icon" size={20} />
                  <input
                    type="text"
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    placeholder="Ask anything about research papers..."
                    className="hero-input"
                  />
                  <button type="submit" className="hero-search-btn" disabled={!inputValue.trim()}>
                    <ArrowRight size={20} />
                  </button>
                </div>
              </form>
              
              <div className="hero-suggestions">
                <span className="suggestions-label">Try asking:</span>
                <div className="suggestion-chips">
                  {[
                    "Latest in transformer models",
                    "Computer vision breakthroughs",
                    "Quantum computing progress"
                  ].map((suggestion, i) => (
                    <button 
                      key={i} 
                      className="suggestion-chip"
                      onClick={() => setInputValue(suggestion)}
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            <div className="hero-stats">
              <div className="stat">
                <div className="stat-number">2M+</div>
                <div className="stat-label">Papers Indexed</div>
              </div>
              <div className="stat">
                <div className="stat-number">50k+</div>
                <div className="stat-label">Researchers Trust Us</div>
              </div>
              <div className="stat">
                <div className="stat-number">99.9%</div>
                <div className="stat-label">Accuracy Rate</div>
              </div>
            </div>
          </div>
        </section>

        {/* Latest Papers Section */}
        <section className="latest-papers-section">
          <div className="section-container">
            <div className="section-header">
              <h2 className="section-title">Latest Papers</h2>
            </div>
            
            <div className="papers-grid">
              <div className="paper-card featured">
                <div className="paper-meta">
                  <span className="paper-category ml">Machine Learning</span>
                  <span className="paper-date">2 days ago</span>
                </div>
                <h3 className="paper-title">Attention Is All You Need: Transformer Architecture Advances</h3>
                <p className="paper-abstract">
                  Recent developments in transformer architectures show significant improvements in efficiency and performance across multiple domains, enabling faster inference and better scalability.
                </p>
                <div className="paper-stats">
                  <span><Eye size={14} /> 1.2k</span>
                  <span><Star size={14} /> 89</span>
                </div>
              </div>
              
              <div className="paper-card">
                <div className="paper-meta">
                  <span className="paper-category cv">Computer Vision</span>
                  <span className="paper-date">3 days ago</span>
                </div>
                <h3 className="paper-title">Self-Supervised Learning in Visual Representation</h3>
                <p className="paper-abstract">
                  Novel approaches to self-supervised learning demonstrate remarkable performance improvements in visual understanding tasks without labeled data.
                </p>
                <div className="paper-stats">
                  <span><Eye size={14} /> 956</span>
                  <span><Star size={14} /> 67</span>
                </div>
              </div>
              
              <div className="paper-card">
                <div className="paper-meta">
                  <span className="paper-category qc">Quantum Computing</span>
                  <span className="paper-date">5 days ago</span>
                </div>
                <h3 className="paper-title">Quantum Error Correction: Breakthrough Methods</h3>
                <p className="paper-abstract">
                  New quantum error correction techniques show promise for building more reliable quantum computing systems with reduced noise and improved fidelity.
                </p>
                <div className="paper-stats">
                  <span><Eye size={14} /> 743</span>
                  <span><Star size={14} /> 45</span>
                </div>
              </div>
            </div>
            
            <div className="papers-cta">
              <button className="view-all-papers-btn" onClick={startChat}>
                <FileText size={20} />
                Explore All Papers
                <ArrowRight size={16} />
              </button>
            </div>
          </div>
        </section>

        {/* Features Section */}
        <section className="features-section">
          <div className="section-container">
            <div className="section-header">
              <h2 className="section-title">Advanced Research Intelligence Platform</h2>
            </div>

            <div className="features-grid">
              <div className="feature-card feature-card-primary">
                <div className="feature-icon">
                  <Brain size={32} />
                </div>
                <h3>AI-Powered Synthesis</h3>
                <p>Advanced language models synthesize insights from multiple papers, delivering comprehensive answers with intelligent analysis and contextual understanding for complex research queries.</p>
                <div className="feature-highlight">
                  <Zap size={16} />
                  <span>Instant Analysis</span>
                </div>
              </div>

              <div className="feature-card">
                <div className="feature-icon">
                  <Database size={32} />
                </div>
                <h3>RAG Technology</h3>
                <p>Retrieval-Augmented Generation ensures every response is grounded in actual research papers, combining real-time data retrieval with sophisticated reasoning capabilities.</p>
                <div className="feature-highlight">
                  <Shield size={16} />
                  <span>Source Verified</span>
                </div>
              </div>

              <div className="feature-card">
                <div className="feature-icon">
                  <Search size={32} />
                </div>
                <h3>Smart Discovery</h3>
                <p>Intelligent semantic search capabilities help you find relevant papers across disciplines, using advanced algorithms to match your research interests with precision.</p>
                <div className="feature-highlight">
                  <TrendingUp size={16} />
                  <span>Always Current</span>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* How It Works */}
        <section className="how-it-works">
          <div className="section-container">
            <h2 className="section-title">How It Works</h2>
            <div className="steps">
              <div className="step">
                <div className="step-number">1</div>
                <div className="step-content">
                  <h3>Ask Your Question</h3>
                  <p>Type any research question in natural language</p>
                </div>
              </div>
              <ChevronRight className="step-arrow" size={24} />
              <div className="step">
                <div className="step-number">2</div>
                <div className="step-content">
                  <h3>AI Searches & Analyzes</h3>
                  <p>Our system retrieves and processes relevant papers</p>
                </div>
              </div>
              <ChevronRight className="step-arrow" size={24} />
              <div className="step">
                <div className="step-number">3</div>
                <div className="step-content">
                  <h3>Get Synthesized Answer</h3>
                  <p>Receive comprehensive insights with citations</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* CTA Section */}
        <section className="cta-section">
          <div className="cta-container">
            <div className="cta-content">
              <h2>Ready to Accelerate Your Research?</h2>
              <p>Join thousands of researchers who are already discovering insights faster</p>
              <button className="cta-button" onClick={startChat}>
                <Rocket size={20} />
                {isAuthenticated ? 'Start Exploring Now' : 'Sign In to Get Started'}
                <ArrowRight size={20} />
              </button>
            </div>
          </div>
        </section>

        {/* Footer */}
        <footer className="footer">
          <div className="footer-content">
            <div className="footer-section">
              <div className="logo">
                <Brain size={24} />
                <span>arXiv Curator</span>
              </div>
              <p>Revolutionizing academic research with AI</p>
            </div>
            <div className="footer-section">
              <h4>Features</h4>
              <ul>
                <li>AI Chat Interface</li>
                <li>Source Citations</li>
                <li>Dark Mode</li>
                <li>Mobile Responsive</li>
              </ul>
            </div>
            <div className="footer-section">
              <h4>Research Areas</h4>
              <ul>
                <li>Machine Learning</li>
                <li>Computer Vision</li>
                <li>Natural Language Processing</li>
                <li>Quantum Computing</li>
              </ul>
            </div>
          </div>
        </footer>

        {/* Authentication Modal */}
        {showAuthModal && (
          <AuthModal 
            onGoogleSignIn={handleGoogleSignIn}
            onEmailSignIn={handleEmailSignIn}
            onClose={closeAuthModal}
            darkMode={darkMode}
          />
        )}
      </div>
    );
  }


  // Chat View
  return (
    <div className="app chat-page">
      <header className="chat-header">
        <div className="header-left">
          <div className="logo" onClick={goToLanding} style={{ cursor: 'pointer' }}>
          <div className="logo-icon">
                <Brain size={28} />
                <Sparkles className="sparkle" size={16} />
              </div>
              <span className="logo-text">arXiv Curator</span>
          </div>
        </div>
        
        <div className="header-actions">
          {/* Backend Connection Status */}
          <div className="backend-status" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginRight: '1rem' }}>
            <div 
              className={`status-indicator ${stats.connected ? 'connected' : 'disconnected'}`}
              style={{
                width: '8px',
                height: '8px',
                borderRadius: '50%',
                backgroundColor: stats.connected ? '#10b981' : '#ef4444'
              }}
            />
            <span style={{ fontSize: '0.875rem', color: stats.connected ? '#10b981' : '#ef4444' }}>
              {stats.connected ? 'Connected' : 'Disconnected'}
            </span>
            {stats.connected && stats.indexedPapers > 0 && (
              <span style={{ fontSize: '0.875rem', color: '#6b7280' }}>
                ({stats.indexedPapers} papers)
              </span>
            )}
          </div>
          
          <button 
            className="panel-toggle citation-toggle"
            onClick={() => setShowCitationPanel(!showCitationPanel)}
            title={showCitationPanel ? "Hide citations" : "Show citations"}
          >
            {showCitationPanel ? <PanelRightClose size={20} /> : <PanelRight size={20} />}
            <span className="toggle-label">Citations</span>
          </button>
          
          <button 
            className="theme-toggle" 
            onClick={() => setShowSettingsModal(true)}
            title="Query Settings"
          >
            <Sliders size={20} />
          </button>
          
          <button className="theme-toggle" onClick={toggleDarkMode}>
            {darkMode ? <Sun size={20} /> : <Moon size={20} />}
          </button>
          
          {/* User Profile Menu */}
          <div className="user-menu-container">
            <button 
              className="user-menu-trigger"
              onClick={() => setShowUserMenu(!showUserMenu)}
            >
              <div className="user-avatar">
                {userProfile.avatar ? (
                  <img src={userProfile.avatar} alt="Profile" />
                ) : (
                  <User size={20} />
                )}
              </div>
              <span className="user-name">{userProfile.name}</span>
              <ChevronDown size={16} className={`chevron ${showUserMenu ? 'rotated' : ''}`} />
            </button>
            
            {showUserMenu && (
              <div className="user-menu-dropdown">
                <div className="user-menu-header">
                  <div className="user-avatar-large">
                    {userProfile.avatar ? (
                      <img src={userProfile.avatar} alt="Profile" />
                    ) : (
                      <User size={32} />
                    )}
                  </div>
                  <div className="user-info">
                    <h4>{userProfile.name}</h4>
                    <p>{userProfile.email}</p>
                    <span className="user-role">{userProfile.role} • {userProfile.institution}</span>
                  </div>
                </div>
                
                <div className="user-stats-mini">
                  <div className="stat-mini">
                    <MessageSquare size={14} />
                    <span>{userProfile.stats.totalQueries} queries</span>
                  </div>
                  <div className="stat-mini">
                    <FileText size={14} />
                    <span>{userProfile.stats.papersRead} papers</span>
                  </div>
                </div>
                
                <div className="user-menu-actions">
                  <button 
                    className="menu-action"
                    onClick={() => {
                      setShowProfileModal(true);
                      setShowUserMenu(false);
                    }}
                  >
                    <User size={16} />
                    View Profile
                  </button>
                  <button className="menu-action">
                    <Settings size={16} />
                    Settings
                  </button>
                  <button className="menu-action">
                    <Award size={16} />
                    Achievements
                  </button>
                  <button className="menu-action logout" onClick={handleSignOut}>
                    <LogOut size={16} />
                    Sign Out
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </header>

      <main className={`chat-layout ${!showCitationPanel ? 'no-citations' : ''}`}>
        {/* Left Panel - Conversations */}
        <aside className="conversations-panel">
            <div className="conversations-header">
              <div className="header-top">
                <button className="new-chat-btn full-width" onClick={startNewConversation}>
                  <Plus size={16} />
                  New Chat
                </button>
              </div>
              
              <div className="search-wrapper">
              <Search size={16} className="search-icon" />
              <input
                type="text"
                placeholder="Search conversations and messages..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="search-input"
              />
              {searchTerm && (
                <button className="clear-search" onClick={clearSearch}>
                  <X size={14} />
                </button>
              )}
            </div>

            <div className="filters-wrapper">
              <div className="filter-group">
                <label className="filter-label">Time</label>
                <select 
                  value={conversationFilter} 
                  onChange={(e) => setConversationFilter(e.target.value)}
                  className="filter-select"
                >
                  <option value="all">All time</option>
                  <option value="today">Today</option>
                  <option value="week">This week</option>
                  <option value="month">This month</option>
                </select>
              </div>
              
              <div className="filter-group">
                <label className="filter-label">Sort</label>
                <select 
                  value={sortBy} 
                  onChange={(e) => setSortBy(e.target.value)}
                  className="filter-select"
                >
                  <option value="recent">Most recent</option>
                  <option value="oldest">Oldest first</option>
                  <option value="alphabetical">A-Z</option>
                </select>
              </div>
            </div>
          </div>
          
          <div className="conversations-list">
            {filteredConversations.length === 0 ? (
              <div className="no-conversations">
                {searchTerm || conversationFilter !== 'all' ? (
                  <>
                    <Search size={48} opacity={0.3} />
                    <p>No matching conversations</p>
                    <span>Try adjusting your search or filters</span>
                    <button className="clear-filters-btn" onClick={clearSearch}>
                      Clear filters
                    </button>
                  </>
                ) : (
                  <>
                    <MessageSquare size={48} opacity={0.3} />
                    <p>No conversations yet</p>
                    <span>Start a new chat to begin</span>
                  </>
                )}
              </div>
            ) : (
              <>
                {/* Pinned conversations */}
                {filteredConversations.filter(conv => conv.pinned && !conv.archived).length > 0 && (
                  <div className="conversation-section">
                    <div className="section-header">
                      <Pin size={14} />
                      <span>Pinned</span>
                    </div>
                    {filteredConversations.filter(conv => conv.pinned && !conv.archived).map((conv) => (
                      <ConversationItem 
                        key={conv.id}
                        conversation={conv}
                        isActive={currentConversationId === conv.id}
                        onSelect={() => loadConversation(conv.id)}
                        onDelete={() => deleteConversation(conv.id)}
                        onPin={() => pinConversation(conv.id)}
                        onArchive={() => archiveConversation(conv.id)}
                      />
                    ))}
                  </div>
                )}

                {/* Recent conversations */}
                {filteredConversations.filter(conv => !conv.pinned && !conv.archived).length > 0 && (
                  <div className="conversation-section">
                    <div className="section-header">
                      <Clock size={14} />
                      <span>Recent</span>
                    </div>
                    {filteredConversations.filter(conv => !conv.pinned && !conv.archived).map((conv) => (
                      <ConversationItem 
                        key={conv.id}
                        conversation={conv}
                        isActive={currentConversationId === conv.id}
                        onSelect={() => loadConversation(conv.id)}
                        onDelete={() => deleteConversation(conv.id)}
                        onPin={() => pinConversation(conv.id)}
                        onArchive={() => archiveConversation(conv.id)}
                      />
                    ))}
                  </div>
                )}

                {/* Archived conversations */}
                {filteredConversations.filter(conv => conv.archived).length > 0 && (
                  <div className="conversation-section">
                    <div className="section-header">
                      <Archive size={14} />
                      <span>Archived</span>
                    </div>
                    {filteredConversations.filter(conv => conv.archived).map((conv) => (
                      <ConversationItem 
                        key={conv.id}
                        conversation={conv}
                        isActive={currentConversationId === conv.id}
                        onSelect={() => loadConversation(conv.id)}
                        onDelete={() => deleteConversation(conv.id)}
                        onPin={() => pinConversation(conv.id)}
                        onArchive={() => archiveConversation(conv.id)}
                      />
                    ))}
                  </div>
                )}
              </>
            )}
          </div>
        </aside>

        {/* Center Panel - Chat Area */}
        <div className="chat-area">
          {/* Onboarding Overlay */}
          {showOnboarding && messages.length === 0 && (
            <div className="onboarding-overlay">
              <div className="onboarding-content">
                <div className="onboarding-header">
                  <div className="onboarding-icon">
                    <Brain size={48} />
                    <Sparkles className="sparkle-large" size={24} />
                  </div>
                  <h2>Welcome to arXiv Curator! 🎉</h2>
                  <p>Your AI-powered research assistant for academic papers</p>
                  <button 
                    className="close-onboarding"
                    onClick={() => {
                      setShowOnboarding(false);
                      localStorage.setItem('hasSeenOnboarding', 'true');
                    }}
                  >
                    <X size={16} />
                  </button>
                </div>
                
                <div className="onboarding-tips">
                  <div className="tip">
                    <HelpCircle size={20} />
                    <span>Ask questions in natural language</span>
                  </div>
                  <div className="tip">
                    <Quote size={20} />
                    <span>Get answers with proper citations</span>
                  </div>
                  <div className="tip">
                    <Lightbulb size={20} />
                    <span>Rate responses to improve quality</span>
                  </div>
                </div>

                <div className="example-questions-onboarding">
                  <h3>Try these example questions:</h3>
                  <div className="example-grid">
                    {exampleQuestions.slice(0, 6).map((question, i) => (
                      <button 
                        key={i} 
                        className="example-chip"
                        onClick={() => {
                          setInputValue(question);
                          setShowOnboarding(false);
                          localStorage.setItem('hasSeenOnboarding', 'true');
                        }}
                      >
                        {question}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          <div className="chat-messages">
            {messages.length === 0 && !showOnboarding && (
              <div className="chat-welcome">
                <div className="welcome-icon">
                  <Brain size={64} />
                  <Sparkles className="sparkle-large" size={24} />
                </div>
                <h2>Ready to explore research?</h2>
                <p>Ask me anything about academic papers and I'll provide synthesized insights with proper citations.</p>
                <div className="quick-examples">
                  <h3>Quick examples:</h3>
                  <div className="example-buttons">
                    {exampleQuestions.slice(0, 3).map((question, i) => (
                      <button 
                        key={i} 
                        className="example-btn"
                        onClick={() => setInputValue(question)}
                      >
                        {question}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}
            
            {messages.map((message) => (
              <div key={message.id} className={`message ${message.type}`}>
                <div className="message-avatar">
                  {message.type === 'user' ? (
                    <Users size={20} />
                  ) : (
                    <Brain size={20} />
                  )}
                </div>
                <div className="message-content">
                  <div className="message-header">
                    <span className="message-sender">
                      {message.type === 'user' ? 'You' : 'AI Assistant'}
                    </span>
                    <span className="message-timestamp">
                      {new Date(message.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                  <div className="message-text">
                    {message.content}
                    {message.id === rag.currentMessageId && rag.streaming && (
                      <span className="streaming-cursor" style={{ animation: 'blink 1s infinite' }}>▊</span>
                    )}
                  </div>
                  
                  {/* Show real contexts from RAG */}
                  {message.contexts && message.contexts.length > 0 && (
                    <div className="message-sources">
                      <span className="sources-label">
                        <ExternalLink size={14} />
                        {message.contexts.length} context{message.contexts.length > 1 ? 's' : ''} retrieved
                      </span>
                      <div className="source-tags">
                        {message.contexts.map((ctx, index) => (
                          <button 
                            key={index}
                            className="source-tag"
                            onClick={() => setSelectedCitation({ ...ctx, messageId: message.id })}
                            style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}
                          >
                            <FileText size={12} />
                            {ctx.paper_id ? ctx.paper_id.replace(/_/g, '.') : `Context ${index + 1}`}
                            {ctx.section && <span style={{ fontSize: '0.75rem', opacity: 0.7 }}>({ctx.section})</span>}
                            {ctx.score && (
                              <span className="score-badge" style={{ 
                                marginLeft: '0.25rem',
                                padding: '0.125rem 0.375rem',
                                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                                color: '#3b82f6',
                                borderRadius: '999px',
                                fontSize: '0.75rem'
                              }}>
                                {(ctx.score * 100).toFixed(0)}%
                              </span>
                            )}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {/* Fallback for old sources format */}
                  {message.sources && message.sources.length > 0 && !message.contexts && (
                    <div className="message-sources">
                      <span className="sources-label">
                        <ExternalLink size={14} />
                        {message.sources.length} source{message.sources.length > 1 ? 's' : ''}
                      </span>
                      <div className="source-tags">
                        {message.sources.map((source, index) => (
                          <button 
                            key={index}
                            className="source-tag"
                            onClick={() => setSelectedCitation({ ...source, messageId: message.id })}
                          >
                            <FileText size={12} />
                            Source {index + 1}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}

                  <div className="message-actions">
                    <button 
                      className={`action-btn copy-btn ${copiedMessageId === message.id ? 'copied' : ''}`}
                      onClick={() => copyToClipboard(message.content, message.id)}
                      title="Copy message"
                    >
                      {copiedMessageId === message.id ? (
                        <Check size={14} />
                      ) : (
                        <Copy size={14} />
                      )}
                      {copiedMessageId === message.id ? 'Copied!' : 'Copy'}
                    </button>
                    
                    {message.type === 'assistant' && (
                      <>
                        <button 
                          className={`feedback-btn ${message.feedback === 'like' ? 'active' : ''}`}
                          onClick={() => handleFeedback(message.id, message.feedback === 'like' ? null : 'like')}
                          title="Like this response"
                        >
                          <ThumbsUp size={14} />
                        </button>
                        <button 
                          className={`feedback-btn ${message.feedback === 'dislike' ? 'active' : ''}`}
                          onClick={() => handleFeedback(message.id, message.feedback === 'dislike' ? null : 'dislike')}
                          title="Dislike this response"
                        >
                          <ThumbsDown size={14} />
                        </button>
                        <button 
                          className="feedback-btn"
                          onClick={() => setShowFeedbackForm(message.id)}
                          title="Provide detailed feedback"
                        >
                          <MessageCircle size={14} />
                          Feedback
                        </button>
                      </>
                    )}
                  </div>
                </div>
              </div>
            ))}
            
            {(isLoading || rag.loading) && (
              <div className="message assistant">
                <div className="message-avatar">
                  <Brain size={20} />
                </div>
                <div className="message-content">
                  <div className="loading-modern">
                    <div className="loading-brain">
                      <Brain className="brain-icon" size={24} />
                    </div>
                    <span>
                      {rag.loading ? rag.getLoadingMessage() : 'Analyzing research papers...'}
                      {rag.streaming && <span className="streaming-dots">...</span>}
                    </span>
                  </div>
                </div>
              </div>
            )}
            
            {/* RAG Error Display */}
            {rag.error && (
              <div className="message assistant" style={{ backgroundColor: '#fef2f2' }}>
                <div className="message-avatar">
                  <AlertCircle size={20} style={{ color: '#ef4444' }} />
                </div>
                <div className="message-content">
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#ef4444' }}>
                    <span>{rag.error}</span>
                    <button 
                      onClick={rag.reset}
                      style={{ 
                        padding: '0.25rem 0.5rem', 
                        fontSize: '0.875rem',
                        backgroundColor: '#fee2e2',
                        border: '1px solid #fecaca',
                        borderRadius: '4px',
                        cursor: 'pointer'
                      }}
                    >
                      Dismiss
                    </button>
                  </div>
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>

          <div className="chat-input-container">
            <form onSubmit={handleSubmit} className="chat-input-form">
              <div className="input-wrapper">
                <input
                  type="text"
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  placeholder="Ask a question about research papers..."
                  className="chat-input"
                  disabled={isLoading}
                />
                <button 
                  type="submit" 
                  className="send-button"
                  disabled={isLoading || !inputValue.trim()}
                >
                  {isLoading ? (
                    <div className="loading-spinner" />
                  ) : (
                    <Send size={18} />
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>

        {/* Right Panel - Citations */}
        {showCitationPanel && (
          <aside className="citation-panel">
            <div className="citation-header">
              <h3>Citations & Sources</h3>
              <button 
                className="close-citation"
                onClick={() => setShowCitationPanel(false)}
              >
                <X size={16} />
              </button>
            </div>
            
            {selectedCitation ? (
              <div className="citation-content">
                <div className="citation-info">
                  <div className="citation-badge">
                    <FileText size={16} />
                    <span>Paper Details</span>
                  </div>
                  <h4>{selectedCitation.paper_id ? selectedCitation.paper_id.replace(/_/g, '.') : 'Paper'}</h4>
                  <p className="citation-section">
                    <strong>Section:</strong> {selectedCitation.section || 'Unknown'}
                  </p>
                  {selectedCitation.score && (
                    <p className="citation-score">
                      <strong>Relevance Score:</strong> {(selectedCitation.score * 100).toFixed(1)}%
                    </p>
                  )}
                  
                  <div className="citation-actions">
                    <a 
                      href={`https://arxiv.org/pdf/${selectedCitation.paper_id.replace(/_/g, '.')}.pdf`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="citation-link"
                    >
                      <ExternalLink size={14} />
                      View PDF on arXiv
                    </a>
                  </div>
                </div>
                
                <div className="citation-preview">
                  <h5>
                    <Quote size={14} style={{ display: 'inline', marginRight: '0.5rem' }} />
                    Context Preview
                  </h5>
                  <div className="preview-text">
                    {selectedCitation.content || selectedCitation.text || 'No preview available'}
                  </div>
                </div>
              </div>
            ) : (
              <div className="no-citation">
                <Quote size={48} opacity={0.3} />
                <p>Select a source to view details</p>
                <span>Click on any source tag in the chat to see citation information</span>
              </div>
            )}
          </aside>
        )}

        {/* User Profile Modal */}
        {showProfileModal && (
          <div className="profile-modal">
            <div className="profile-form">
              <div className="profile-header">
                <h2>User Profile</h2>
                <button onClick={() => setShowProfileModal(false)}>
                  <X size={20} />
                </button>
              </div>
              
              <div className="profile-content">
                <div className="profile-avatar-section">
                  <div className="avatar-upload">
                    {userProfile.avatar ? (
                      <img src={userProfile.avatar} alt="Profile" className="profile-avatar-large" />
                    ) : (
                      <div className="profile-avatar-placeholder">
                        <User size={48} />
                      </div>
                    )}
                    <button className="change-avatar-btn">
                      <Camera size={16} />
                      Change Photo
                    </button>
                  </div>
                </div>
                
                <div className="profile-info-section">
                  {editingProfile ? (
                    <div className="profile-edit-form">
                      <div className="form-group">
                        <label>Full Name</label>
                        <input 
                          type="text" 
                          value={userProfile.name}
                          onChange={(e) => setUserProfile(prev => ({ ...prev, name: e.target.value }))}
                        />
                      </div>
                      <div className="form-group">
                        <label>Email</label>
                        <input 
                          type="email" 
                          value={userProfile.email}
                          onChange={(e) => setUserProfile(prev => ({ ...prev, email: e.target.value }))}
                        />
                      </div>
                      <div className="form-group">
                        <label>Institution</label>
                        <input 
                          type="text" 
                          value={userProfile.institution}
                          onChange={(e) => setUserProfile(prev => ({ ...prev, institution: e.target.value }))}
                        />
                      </div>
                      <div className="form-group">
                        <label>Role</label>
                        <select 
                          value={userProfile.role}
                          onChange={(e) => setUserProfile(prev => ({ ...prev, role: e.target.value }))}
                        >
                          <option value="PhD Student">PhD Student</option>
                          <option value="Master's Student">Master's Student</option>
                          <option value="Researcher">Researcher</option>
                          <option value="Professor">Professor</option>
                          <option value="Post-doc">Post-doc</option>
                          <option value="Industry Professional">Industry Professional</option>
                        </select>
                      </div>
                      <div className="form-group">
                        <label>Specialization</label>
                        <input 
                          type="text" 
                          value={userProfile.specialization}
                          onChange={(e) => setUserProfile(prev => ({ ...prev, specialization: e.target.value }))}
                        />
                      </div>
                    </div>
                  ) : (
                    <div className="profile-display">
                      <div className="profile-field">
                        <Mail size={16} />
                        <span>{userProfile.email}</span>
                      </div>
                      <div className="profile-field">
                        <MapPin size={16} />
                        <span>{userProfile.institution}</span>
                      </div>
                      <div className="profile-field">
                        <Award size={16} />
                        <span>{userProfile.role}</span>
                      </div>
                      <div className="profile-field">
                        <Brain size={16} />
                        <span>{userProfile.specialization}</span>
                      </div>
                      <div className="profile-field">
                        <Calendar size={16} />
                        <span>Joined {new Date(userProfile.joinDate).toLocaleDateString()}</span>
                      </div>
                    </div>
                  )}
                </div>
                
                <div className="profile-stats-section">
                  <h3>Research Activity</h3>
                  <div className="stats-grid">
                    <div className="stat-card">
                      <MessageSquare size={24} />
                      <div className="stat-info">
                        <span className="stat-number">{userProfile.stats.totalQueries}</span>
                        <span className="stat-label">Total Queries</span>
                      </div>
                    </div>
                    <div className="stat-card">
                      <FileText size={24} />
                      <div className="stat-info">
                        <span className="stat-number">{userProfile.stats.papersRead}</span>
                        <span className="stat-label">Papers Explored</span>
                      </div>
                    </div>
                    <div className="stat-card">
                      <Users size={24} />
                      <div className="stat-info">
                        <span className="stat-number">{userProfile.stats.conversationsStarted}</span>
                        <span className="stat-label">Conversations</span>
                      </div>
                    </div>
                    <div className="stat-card">
                      <ThumbsUp size={24} />
                      <div className="stat-info">
                        <span className="stat-number">{userProfile.stats.feedbackGiven}</span>
                        <span className="stat-label">Feedback Given</span>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="profile-preferences">
                  <h3>Preferences</h3>
                  <div className="preferences-grid">
                    <div className="preference-item">
                      <label>Citation Style</label>
                      <select 
                        value={userProfile.preferences.citationStyle}
                        onChange={(e) => setUserProfile(prev => ({ 
                          ...prev, 
                          preferences: { ...prev.preferences, citationStyle: e.target.value }
                        }))}
                      >
                        <option value="APA">APA</option>
                        <option value="MLA">MLA</option>
                        <option value="Chicago">Chicago</option>
                        <option value="IEEE">IEEE</option>
                      </select>
                    </div>
                    <div className="preference-item">
                      <label>
                        <input 
                          type="checkbox" 
                          checked={userProfile.preferences.autoSave}
                          onChange={(e) => setUserProfile(prev => ({ 
                            ...prev, 
                            preferences: { ...prev.preferences, autoSave: e.target.checked }
                          }))}
                        />
                        Auto-save conversations
                      </label>
                    </div>
                    <div className="preference-item">
                      <label>
                        <input 
                          type="checkbox" 
                          checked={userProfile.preferences.notifications}
                          onChange={(e) => setUserProfile(prev => ({ 
                            ...prev, 
                            preferences: { ...prev.preferences, notifications: e.target.checked }
                          }))}
                        />
                        Enable notifications
                      </label>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="profile-actions">
                {editingProfile ? (
                  <>
                    <button 
                      className="btn-secondary"
                      onClick={() => setEditingProfile(false)}
                    >
                      Cancel
                    </button>
                    <button 
                      className="btn-primary"
                      onClick={() => {
                        updateUserProfile(userProfile);
                        setEditingProfile(false);
                      }}
                    >
                      <Save size={16} />
                      Save Changes
                    </button>
                  </>
                ) : (
                  <button 
                    className="btn-primary"
                    onClick={() => setEditingProfile(true)}
                  >
                    <Edit3 size={16} />
                    Edit Profile
                  </button>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Feedback Form Modal */}
        {showFeedbackForm && (
          <div className="feedback-modal">
            <div className="feedback-form">
              <div className="feedback-header">
                <h3>Provide Feedback</h3>
                <button onClick={() => setShowFeedbackForm(null)}>
                  <X size={16} />
                </button>
              </div>
              <textarea 
                placeholder="What could be improved about this response?"
                className="feedback-textarea"
              />
              <div className="feedback-actions">
                <button 
                  className="btn-secondary"
                  onClick={() => setShowFeedbackForm(null)}
                >
                  Cancel
                </button>
                <button 
                  className="btn-primary"
                  onClick={() => {
                    handleFeedback(showFeedbackForm, 'feedback-submitted');
                  }}
                >
                  Submit Feedback
                </button>
              </div>
            </div>
          </div>
        )}
        
        {/* Settings Modal */}
        <SettingsModal 
          isOpen={showSettingsModal}
          onClose={() => setShowSettingsModal(false)}
          darkMode={darkMode}
        />
      </main>
    </div>
  );
}

export default App;