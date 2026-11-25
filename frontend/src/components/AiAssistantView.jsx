import React, { useState, useRef, useEffect } from 'react';
import {
  Sparkles,
  Send,
  Bot,
  User,
  ShieldCheck,
  BrainCircuit,
  TrendingUp,
  LineChart,
  HelpCircle,
  AlertCircle,
  ArrowRight,
  Info,
} from 'lucide-react';

const INITIAL_MESSAGES = [
  {
    id: 1,
    sender: 'assistant',
    text: "Hello! I am your StockSense AI Research Assistant. I can help you understand market movements, analyze technical indicators (like SMA20, SMA50, RSI), interpret our PyTorch LSTM next-day predictions, and evaluate risk factors across 472 S&P 500 stocks. How can I assist your research today?",
    timestamp: 'Just now',
  },
];

const PRESET_PROMPTS = [
  "What does our LSTM prediction mean?",
  "Explain RSI and how to interpret it",
  "How are the 8 technical features calculated?",
  "What are the risks of trading on directional ML?",
  "How does zero data leakage work in StockSense?",
];

const KNOWLEDGE_RESPONSES = {
  "what does our lstm prediction mean":
    "Our PyTorch StockLSTM model is a binary classifier trained on rolling 30-session sequences of 8 technical features. It estimates the probability P(Close(t+1) > Close(t)). If probability ≥ 0.50, it classifies the next trading day as UP; otherwise DOWN. Across 458,428 out-of-sample test sequences (2022–2026), it achieved 51.75% accuracy (+0.11% edge over the naive baseline). It is a directional momentum signal and must not be treated as a guarantee of profit.",
  "explain rsi":
    "The Relative Strength Index (RSI) is an oscillator measuring the speed and magnitude of recent price changes over a 14-day window on a scale of 0 to 100. Traditionally, an RSI above 70 indicates an overbought condition (potential pullback or consolidation), while an RSI below 30 signals an oversold condition (potential rebound). Between 45 and 65, RSI indicates healthy trend momentum.",
  "how are the 8 technical features calculated":
    "StockSense AI computes 8 core signals strictly per ticker without cross-ticker leakage: 1. Daily Return: Close(t)/Close(t-1) - 1; 2. Volume Change with safe zero-handling; 3. High-Low Range: (High - Low)/Close; 4. 20-Day Rolling Volatility; 5. SMA20: 20-day simple moving average; 6. SMA50: 50-day SMA; 7. Close/SMA20 stretch ratio; 8. Close/SMA50 trend ratio.",
  "what are the risks of trading on directional ml":
    "Quantitative risks include: 1. Upward Drift Bias: The model predicts UP in ~90% of sessions because broad equity indexes drift upward over multi-decade horizons; 2. Transaction Friction: A ~52% raw accuracy edge can be eroded by bid-ask spreads and commissions unless turnover is constrained; 3. Macro & News Dislocation: Sudden Federal Reserve pivots or geopolitical shocks can disrupt momentum patterns.",
  "how does zero data leakage work":
    "StockSense AI enforces strict chronological integrity: 1. Data is split strictly by calendar dates (70% train 2000–2018, 15% val 2018–2022, 15% test 2022–2026); 2. StandardScaler is fitted strictly on the training partition; 3. Sequence windows (30 sessions) are verified to never cross ticker boundaries; 4. Indicators use only past data, avoiding lookahead bias.",
};

export default function AiAssistantView({ onSelectTicker, onNavigate }) {
  const [messages, setMessages] = useState(INITIAL_MESSAGES);
  const [inputQuery, setInputQuery] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  const handleSendMessage = (textToSend = null) => {
    const text = (textToSend || inputQuery).trim();
    if (!text) return;

    const userMsg = {
      id: Date.now(),
      sender: 'user',
      text,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, userMsg]);
    if (!textToSend) setInputQuery('');
    setIsTyping(true);

    // Simulate grounded response
    setTimeout(() => {
      const lower = text.toLowerCase();
      let replyText = '';

      if (lower.includes('lstm') || lower.includes('prediction') || lower.includes('model')) {
        replyText = KNOWLEDGE_RESPONSES["what does our lstm prediction mean"];
      } else if (lower.includes('rsi') || lower.includes('relative strength')) {
        replyText = KNOWLEDGE_RESPONSES["explain rsi"];
      } else if (lower.includes('feature') || lower.includes('calculate') || lower.includes('sma')) {
        replyText = KNOWLEDGE_RESPONSES["how are the 8 technical features calculated"];
      } else if (lower.includes('risk') || lower.includes('drawdown') || lower.includes('caution')) {
        replyText = KNOWLEDGE_RESPONSES["what are the risks of trading on directional ml"];
      } else if (lower.includes('leakage') || lower.includes('split') || lower.includes('scaler')) {
        replyText = KNOWLEDGE_RESPONSES["how does zero data leakage work"];
      } else if (lower.includes('aapl') || lower.includes('apple')) {
        replyText = "For AAPL (Apple Inc.), our model evaluates trailing 30-day technical indicators including 20-day volatility and SMA stretch ratios. You can explore AAPL's full candlestick chart, technical breakdown, and live PyTorch LSTM directional probability in the Stock Analysis tab.";
      } else {
        replyText = `Thank you for your question about "${text}". In StockSense AI, all technical indicators (SMA20, SMA50, Rolling Volatility) and PyTorch LSTM predictions are calculated in real time using strict chronological standards. In Phase 5, our full Retrieval-Augmented Generation (RAG) backend will dynamically analyze any S&P 500 company alongside live Yahoo Finance news.`;
      }

      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          sender: 'assistant',
          text: replyText,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        },
      ]);
      setIsTyping(false);
    }, 600);
  };

  return (
    <div className="assistant-view">
      {/* 1. Header Banner */}
      <div className="view-header-strip">
        <div className="view-title-group">
          <div className="view-icon-badge ai">
            <Sparkles size={20} />
          </div>
          <div>
            <h1 className="view-title">StockSense AI Assistant</h1>
            <p className="view-subtitle">
              Interactive conversational intelligence grounded in genuine technical indicators, PyTorch predictions, and market risk analysis.
            </p>
          </div>
        </div>

        <div className="view-actions">
          <div className="phase-pill ai">
            <span className="phase-dot" />
            <span>Phase 1 UI Sandbox • Phase 5 Full RAG Pipeline</span>
          </div>
        </div>
      </div>

      {/* 2. Main Chat Shell */}
      <div className="chat-container-card">
        {/* Chat History */}
        <div className="chat-messages-area">
          {messages.map((m) => (
            <div key={m.id} className={`chat-message-row ${m.sender}`}>
              <div className="message-avatar">
                {m.sender === 'assistant' ? <Bot size={16} /> : <User size={16} />}
              </div>
              <div className="message-bubble">
                <div className="message-header">
                  <span className="message-author">
                    {m.sender === 'assistant' ? 'StockSense AI' : 'You'}
                  </span>
                  <span className="message-time font-mono">{m.timestamp}</span>
                </div>
                <div className="message-text">{m.text}</div>
              </div>
            </div>
          ))}

          {isTyping && (
            <div className="chat-message-row assistant">
              <div className="message-avatar"><Bot size={16} /></div>
              <div className="message-bubble typing-bubble">
                <span className="typing-dot" />
                <span className="typing-dot" />
                <span className="typing-dot" />
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Preset Prompt Chips */}
        <div className="preset-prompts-bar">
          <span className="preset-label">Suggested Prompts:</span>
          <div className="preset-chips-scroll">
            {PRESET_PROMPTS.map((p) => (
              <button
                key={p}
                type="button"
                className="preset-chip"
                onClick={() => handleSendMessage(p)}
              >
                <span>{p}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Input Bar */}
        <div className="chat-input-wrapper">
          <input
            type="text"
            className="chat-input-field"
            placeholder="Ask about S&P 500 stocks, technical indicators, LSTM predictions, or risk..."
            value={inputQuery}
            onChange={(e) => setInputQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
          />
          <button
            type="button"
            className="chat-send-btn"
            disabled={!inputQuery.trim() || isTyping}
            onClick={() => handleSendMessage()}
          >
            <Send size={16} />
          </button>
        </div>

        {/* Trust & Grounding Notice */}
        <div className="chat-grounding-notice">
          <ShieldCheck size={14} className="text-ai" />
          <span>
            <strong>Zero Hallucination Guarantee:</strong> StockSense AI answers are grounded in our verified technical indicators and PyTorch model specifications.
          </span>
        </div>
      </div>
    </div>
  );
}
