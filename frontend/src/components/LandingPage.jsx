import React from 'react';
import {
  TrendingUp,
  LineChart,
  BrainCircuit,
  Compass,
  Sparkles,
  Bookmark,
  ShieldCheck,
  ArrowRight,
  BarChart3,
  CheckCircle2,
  Lock,
  Layers,
  ChevronRight,
  Database,
  Cpu,
  Sliders,
  Activity,
  Zap,
} from 'lucide-react';

export default function LandingPage({ onNavigate, onSelectTicker }) {
  const handleLaunchTerminal = (ticker = 'AAPL') => {
    if (onSelectTicker) onSelectTicker(ticker);
    if (onNavigate) onNavigate('analysis');
  };

  return (
    <div className="landing-page">
      {/* 1. Landing Navigation Bar */}
      <header className="landing-nav">
        <div className="landing-nav-container">
          <div className="landing-brand" onClick={() => onNavigate('landing')}>
            <div className="landing-brand-logo">
              <TrendingUp size={20} />
            </div>
            <div className="landing-brand-text">
              <span className="brand-title">StockSense <span className="brand-ai">AI</span></span>
              <span className="brand-tag">Quantitative Intelligence</span>
            </div>
          </div>

          <nav className="landing-nav-links">
            <button type="button" onClick={() => onNavigate('market')}>Market</button>
            <button type="button" onClick={() => onNavigate('discovery')}>Discovery</button>
            <button type="button" onClick={() => onNavigate('analysis')}>Stock Analysis</button>
            <button type="button" onClick={() => onNavigate('assistant')}>AI Assistant</button>
            <button type="button" onClick={() => onNavigate('research')}>Model Research</button>
          </nav>

          <div className="landing-nav-actions">
            <button
              type="button"
              className="btn-secondary-sm"
              onClick={() => onNavigate('market')}
            >
              Explore Market
            </button>
            <button
              type="button"
              className="btn-primary-sm"
              onClick={() => handleLaunchTerminal('AAPL')}
            >
              Launch Terminal <ArrowRight size={14} />
            </button>
          </div>
        </div>
      </header>

      {/* 2. Hero Section */}
      <section className="landing-hero">
        <div className="hero-glow-emerald" />
        <div className="hero-glow-ai" />

        <div className="landing-hero-content">
          <div className="hero-badge">
            <span className="pulse-dot" />
            <span className="badge-text">S&P 500 Quantitative Machine Learning Terminal</span>
          </div>

          <h1 className="hero-headline">
            Stock Market Intelligence Driven by <span className="text-gradient-emerald">Machine Learning</span> &amp; <span className="text-gradient-ai">Real-Time Data</span>
          </h1>

          <p className="hero-subhead">
            Move beyond guesswork. Analyze market momentum, discover opportunities across 472 S&amp;P 500 stocks, and explore next-trading-day directional forecasts powered by our 2-layer PyTorch LSTM model trained across 2.7 million market sessions.
          </p>

          <div className="hero-cta-group">
            <button
              type="button"
              className="btn-hero-primary"
              onClick={() => handleLaunchTerminal('AAPL')}
            >
              <Zap size={18} />
              <span>Launch Stock Terminal</span>
            </button>
            <button
              type="button"
              className="btn-hero-secondary"
              onClick={() => onNavigate('market')}
            >
              <BarChart3 size={18} />
              <span>Explore S&amp;P 500 Market</span>
            </button>
            <button
              type="button"
              className="btn-hero-ghost"
              onClick={() => onNavigate('research')}
            >
              <BrainCircuit size={18} />
              <span>Model Research &amp; Backtests</span>
            </button>
          </div>

          {/* Trust & Proof Metric Strip */}
          <div className="hero-metrics-strip">
            <div className="metric-box">
              <span className="metric-value font-mono">472</span>
              <span className="metric-label">S&amp;P 500 Equities</span>
            </div>
            <div className="metric-divider" />
            <div className="metric-box">
              <span className="metric-value font-mono">2,703,531</span>
              <span className="metric-label">Daily Market Sessions</span>
            </div>
            <div className="metric-divider" />
            <div className="metric-box">
              <span className="metric-value font-mono">51.75%</span>
              <span className="metric-label">Test Set Directional Acc</span>
            </div>
            <div className="metric-divider" />
            <div className="metric-box">
              <span className="metric-value font-mono">8</span>
              <span className="metric-label">Zero-Leakage Signals</span>
            </div>
          </div>
        </div>

        {/* 3. Interactive Product Terminal Preview */}
        <div className="hero-preview-wrapper" onClick={() => handleLaunchTerminal('AAPL')}>
          <div className="terminal-window">
            <div className="terminal-header">
              <div className="window-dots">
                <span className="dot red" />
                <span className="dot yellow" />
                <span className="dot green" />
              </div>
              <div className="terminal-title">StockSense AI — Live Research Terminal [AAPL • Apple Inc.]</div>
              <div className="terminal-status">
                <span className="status-indicator" /> Live Inference Ready
              </div>
            </div>

            <div className="terminal-body-preview">
              <div className="preview-top-bar">
                <div className="ticker-info">
                  <span className="ticker-symbol">AAPL</span>
                  <span className="ticker-name">Apple Inc.</span>
                  <span className="ticker-sector">Information Technology</span>
                  <span className="ticker-price font-mono">$258.42</span>
                  <span className="ticker-change font-mono text-up">+1.34%</span>
                </div>
                <div className="preview-action-pill">
                  <span>Interactive Preview • Click to Open Full Terminal</span>
                  <ChevronRight size={14} />
                </div>
              </div>

              <div className="preview-grid">
                {/* Simulated Mini Chart */}
                <div className="preview-chart-pane">
                  <div className="chart-header-row">
                    <div className="chart-title-tag">
                      <Activity size={14} />
                      <span>OHLCV Candlestick &amp; Momentum Moving Averages</span>
                    </div>
                    <div className="chart-legends">
                      <span className="legend-item sma20"><span className="legend-dot sma20" /> SMA20</span>
                      <span className="legend-item sma50"><span className="legend-dot sma50" /> SMA50</span>
                    </div>
                  </div>
                  <div className="mock-chart-visual">
                    <svg viewBox="0 0 700 180" className="mock-svg-chart">
                      <defs>
                        <linearGradient id="chartGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#36D399" stopOpacity="0.25" />
                          <stop offset="100%" stopColor="#36D399" stopOpacity="0.0" />
                        </linearGradient>
                      </defs>
                      {/* Grid Lines */}
                      <line x1="0" y1="40" x2="700" y2="40" stroke="#1e293b" strokeDasharray="3 3" />
                      <line x1="0" y1="90" x2="700" y2="90" stroke="#1e293b" strokeDasharray="3 3" />
                      <line x1="0" y1="140" x2="700" y2="140" stroke="#1e293b" strokeDasharray="3 3" />
                      {/* Area & Price Line */}
                      <path d="M 20 130 Q 120 145 200 110 T 360 80 T 520 70 T 680 35 L 680 180 L 20 180 Z" fill="url(#chartGrad)" />
                      <path d="M 20 130 Q 120 145 200 110 T 360 80 T 520 70 T 680 35" fill="none" stroke="#36D399" strokeWidth="2.5" />
                      {/* SMA20 */}
                      <path d="M 20 140 Q 120 135 200 120 T 360 95 T 520 85 T 680 48" fill="none" stroke="#F59E0B" strokeWidth="1.8" strokeDasharray="4 2" />
                      {/* SMA50 */}
                      <path d="M 20 150 Q 120 148 200 135 T 360 115 T 520 102 T 680 65" fill="none" stroke="#38BDF8" strokeWidth="1.8" />
                    </svg>
                  </div>
                </div>

                {/* Simulated Prediction Card */}
                <div className="preview-prediction-pane">
                  <div className="pred-mini-card">
                    <div className="pred-mini-badge">
                      <BrainCircuit size={14} />
                      <span>PyTorch StockLSTM Forward Pass</span>
                    </div>
                    <div className="pred-mini-result">
                      <div className="pred-tag up">UP DIRECTION</div>
                      <div className="pred-prob font-mono">52.8% Probability</div>
                    </div>
                    <div className="pred-prob-bar">
                      <div className="pred-prob-fill" style={{ width: '52.8%' }} />
                    </div>
                    <div className="pred-meta-row font-mono">
                      <span>Threshold: &tau; = 0.50</span>
                      <span>Next Session Forecast</span>
                    </div>
                    <div className="pred-disclaimer">
                      Trained on 30 trailing sessions. Zero lookahead bias.
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 4. Core Capabilities Showcase */}
      <section className="landing-features">
        <div className="section-container">
          <div className="section-head text-center">
            <span className="section-tag-emerald">Five Core Pillars</span>
            <h2 className="section-heading">Designed for Systematic Equity Research</h2>
            <p className="section-subtext">
              StockSense AI bridges quantitative machine learning research and intuitive market exploration into a unified, lightning-fast terminal.
            </p>
          </div>

          <div className="features-grid">
            {/* Feature 1: Market Overview */}
            <div className="feature-card" onClick={() => onNavigate('market')}>
              <div className="feature-icon emerald">
                <BarChart3 size={24} />
              </div>
              <h3 className="feature-title">Market Overview</h3>
              <p className="feature-desc">
                Real-time monitoring of S&amp;P 500 and NASDAQ-100 broad market dynamics, top gainers, top losers, and market breadth across all 11 GICS sectors.
              </p>
              <div className="feature-action">
                <span>View Market Overview</span>
                <ChevronRight size={16} />
              </div>
            </div>

            {/* Feature 2: Stock Discovery */}
            <div className="feature-card" onClick={() => onNavigate('discovery')}>
              <div className="feature-icon emerald">
                <Compass size={24} />
              </div>
              <h3 className="feature-title">Stock Discovery &amp; Screener</h3>
              <p className="feature-desc">
                Screen the 472 S&amp;P 500 universe with precision filters for price momentum, volume surges, moving average stretches, and volatility.
              </p>
              <div className="feature-action">
                <span>Open Stock Screener</span>
                <ChevronRight size={16} />
              </div>
            </div>

            {/* Feature 3: Stock Analysis & LSTM */}
            <div className="feature-card highlighted" onClick={() => handleLaunchTerminal('NVDA')}>
              <div className="feature-icon ai">
                <LineChart size={24} />
              </div>
              <h3 className="feature-title">Stock Analysis &amp; PyTorch Predictions</h3>
              <p className="feature-desc">
                Interactive candlestick charting, SMA20/50 overlays, historical sessions, and live forward-pass inference from our 2-layer StockLSTM neural network.
              </p>
              <div className="feature-action text-ai">
                <span>Analyze Individual Equities</span>
                <ChevronRight size={16} />
              </div>
            </div>

            {/* Feature 4: AI Assistant */}
            <div className="feature-card" onClick={() => onNavigate('assistant')}>
              <div className="feature-icon ai">
                <Sparkles size={24} />
              </div>
              <h3 className="feature-title">Grounded AI Assistant</h3>
              <p className="feature-desc">
                Ask questions like &ldquo;Why is TSLA moving?&rdquo; or &ldquo;Compare AAPL and MSFT&rdquo;. Strictly grounded in genuine backend metrics with zero hallucinated prices.
              </p>
              <div className="feature-action text-ai">
                <span>Chat with StockSense AI</span>
                <ChevronRight size={16} />
              </div>
            </div>

            {/* Feature 5: Personal Watchlist */}
            <div className="feature-card" onClick={() => onNavigate('watchlist')}>
              <div className="feature-icon emerald">
                <Bookmark size={24} />
              </div>
              <h3 className="feature-title">Personal Watchlist</h3>
              <p className="feature-desc">
                Save your high-conviction equities to a persistent watchlist. Monitor percentage moves, track 52-week ranges, and trigger 1-click LSTM predictions.
              </p>
              <div className="feature-action">
                <span>Open Watchlist</span>
                <ChevronRight size={16} />
              </div>
            </div>

            {/* Feature 6: Quantitative Research */}
            <div className="feature-card" onClick={() => onNavigate('research')}>
              <div className="feature-icon neutral">
                <BrainCircuit size={24} />
              </div>
              <h3 className="feature-title">Backtesting &amp; Model Research</h3>
              <p className="feature-desc">
                Inspect walk-forward validation across 5 historical market regimes, run economic PnL backtests with transaction costs, and view out-of-sample confusion matrices.
              </p>
              <div className="feature-action">
                <span>Inspect Model Card &amp; PnL</span>
                <ChevronRight size={16} />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 5. How StockSense Works (Quantitative Pipeline) */}
      <section className="landing-methodology">
        <div className="section-container">
          <div className="section-head text-center">
            <span className="section-tag-neutral">Quantitative Rigor</span>
            <h2 className="section-heading">How StockSense AI Works</h2>
            <p className="section-subtext">
              Zero data leakage. Zero future lookahead bias. Every sequence is strictly computed within ticker boundaries.
            </p>
          </div>

          <div className="pipeline-stepper">
            <div className="pipeline-step">
              <div className="step-number font-mono">01</div>
              <div className="step-icon"><Database size={20} /></div>
              <h4 className="step-title">Lossless Data Ingestion</h4>
              <p className="step-desc">
                Standardized 2.7M daily market records across 472 tickers. Preserving zero-volume trading halts and historical extremes without synthetic smoothing.
              </p>
            </div>

            <div className="pipeline-step">
              <div className="step-number font-mono">02</div>
              <div className="step-icon"><Sliders size={20} /></div>
              <h4 className="step-title">8 Zero-Leakage Signals</h4>
              <p className="step-desc">
                Computes daily return, volume change, high-low range, 20-day rolling vol, SMA20, SMA50, and moving average stretch ratios strictly per ticker.
              </p>
            </div>

            <div className="pipeline-step">
              <div className="step-number font-mono">03</div>
              <div className="step-icon"><Cpu size={20} /></div>
              <h4 className="step-title">PyTorch 2-Layer LSTM</h4>
              <p className="step-desc">
                Input sequences of 30 trading sessions. StandardScaler fitted strictly on the 2000–2018 training split. Optimized with BCEWithLogitsLoss and Adam.
              </p>
            </div>

            <div className="pipeline-step">
              <div className="step-number font-mono">04</div>
              <div className="step-icon"><ShieldCheck size={20} /></div>
              <h4 className="step-title">Out-of-Sample Holdout</h4>
              <p className="step-desc">
                Evaluated on 458,428 unseen sequences (2022–2026). Yielding 51.75% directional accuracy (+0.11% edge over always-UP naive baseline).
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* 6. Final Call to Action */}
      <section className="landing-cta-banner">
        <div className="cta-banner-content">
          <div className="cta-brand-badge">
            <Sparkles size={16} />
            <span>StockSense AI Platform</span>
          </div>
          <h2 className="cta-title">Upgrade Your S&amp;P 500 Market Research</h2>
          <p className="cta-subtitle">
            Experience the combination of real market data, systematic momentum screening, and machine learning directional predictions.
          </p>
          <div className="cta-btn-group">
            <button
              type="button"
              className="btn-cta-primary"
              onClick={() => handleLaunchTerminal('AAPL')}
            >
              <span>Launch Live Terminal</span>
              <ArrowRight size={18} />
            </button>
            <button
              type="button"
              className="btn-cta-secondary"
              onClick={() => onNavigate('market')}
            >
              <span>Explore S&amp;P 500 Universe</span>
            </button>
          </div>
        </div>
      </section>

      {/* 7. Professional Footer */}
      <footer className="landing-footer">
        <div className="footer-container">
          <div className="footer-top-row">
            <div className="footer-brand-col">
              <div className="footer-brand">
                <TrendingUp size={18} className="text-emerald" />
                <span className="footer-brand-name">StockSense <span className="brand-ai">AI</span></span>
              </div>
              <p className="footer-tagline">
                Institutional-grade quantitative stock intelligence, real-time market overview, discovery screener, and out-of-sample PyTorch LSTM predictions.
              </p>
            </div>

            <div className="footer-nav-col">
              <span className="col-header">Platform Sections</span>
              <button type="button" onClick={() => onNavigate('market')}>Market Overview</button>
              <button type="button" onClick={() => onNavigate('discovery')}>Stock Discovery Screener</button>
              <button type="button" onClick={() => onNavigate('analysis')}>Stock Analysis Cockpit</button>
              <button type="button" onClick={() => onNavigate('assistant')}>AI Assistant</button>
              <button type="button" onClick={() => onNavigate('watchlist')}>Watchlist</button>
            </div>

            <div className="footer-nav-col">
              <span className="col-header">Quantitative ML</span>
              <button type="button" onClick={() => onNavigate('research')}>Model Performance Card</button>
              <button type="button" onClick={() => onNavigate('research')}>Strategy Backtesting &amp; PnL</button>
              <button type="button" onClick={() => onNavigate('research')}>Walk-Forward Validation</button>
              <button type="button" onClick={() => onNavigate('research')}>Zero-Leakage Architecture</button>
            </div>

            <div className="footer-specs-col">
              <span className="col-header">System Specifications</span>
              <div className="spec-row font-mono">
                <span className="spec-k">Universe:</span>
                <span className="spec-v">472 S&amp;P 500 Equities</span>
              </div>
              <div className="spec-row font-mono">
                <span className="spec-k">Model:</span>
                <span className="spec-v">StockLSTM (2-Layer, 128 Hidden)</span>
              </div>
              <div className="spec-row font-mono">
                <span className="spec-k">Test Set:</span>
                <span className="spec-v">458,428 Sequences (2022–2026)</span>
              </div>
              <div className="spec-row font-mono">
                <span className="spec-k">Backend:</span>
                <span className="spec-v">FastAPI • PyTorch • yfinance</span>
              </div>
            </div>
          </div>

          <div className="footer-disclaimer-row">
            <p className="disclaimer-text">
              <strong>Quantitative Research Disclaimer:</strong> StockSense AI is designed exclusively for quantitative research and educational evaluation. The machine learning directional outputs, probabilities, and research scores are algorithmic projections based on historical momentum and technical patterns. They do not constitute financial advice, investment recommendations, or guarantees of profit. All investments involve risk of capital loss.
            </p>
          </div>

          <div className="footer-bottom-row">
            <div className="footer-copy">
              &copy; {new Date().getFullYear()} StockSense AI • Built by Srinath Rao • All Rights Reserved.
            </div>
            <div className="footer-tech-stack">
              <span className="tech-badge">Python 3.11</span>
              <span className="tech-badge">PyTorch</span>
              <span className="tech-badge">FastAPI</span>
              <span className="tech-badge">React 19 + Vite</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
