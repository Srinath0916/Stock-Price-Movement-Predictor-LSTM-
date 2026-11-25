import React, { useState, useEffect } from 'react';
import {
  Search,
  LineChart,
  BrainCircuit,
  Compass,
  BarChart3,
  Sparkles,
  Bookmark,
  TrendingUp,
  X,
  Home,
  FlaskConical,
} from 'lucide-react';

export default function Header({
  activeView,
  setActiveView,
  selectedTicker,
  setSelectedTicker,
  tickerSamples = {},
  watchlistCount = 0,
  marketStream = null,
}) {
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [apiTickers, setApiTickers] = useState([]);

  useEffect(() => {
    fetch('/api/tickers')
      .then((res) => (res.ok ? res.json() : []))
      .then((data) => {
        if (Array.isArray(data) && data.length > 0) {
          setApiTickers(data);
        }
      })
      .catch((err) => console.warn('Header ticker fetch:', err));
  }, []);

  const safeTickers = tickerSamples || {};
  const tickerKeys = ['AAPL', 'NVDA', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA'];

  // If apiTickers are available, use full 472 S&P list, else fallback to sample tickers
  const tickerList = apiTickers.length > 0
    ? apiTickers.map((item) => ({
        ticker: item.ticker,
        name: item.name,
        sector: item.sector,
        price: item.latestPrice,
        change: item.changePct,
      }))
    : Object.keys(safeTickers).map((tk) => {
        const item = safeTickers[tk] || {};
        const hist = item.history || [];
        const latest = hist[hist.length - 1] || {};
        return {
          ticker: tk,
          name: item.name || tk,
          sector: item.sector || 'Equities',
          price: item.stats?.latestPrice || latest.close || 0,
          change: latest.dailyReturn !== undefined ? latest.dailyReturn : 0,
        };
      });

  const filteredTickers = tickerList.filter(
    (item) =>
      item.ticker.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.sector.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleSelect = (tk) => {
    setSelectedTicker(tk);
    if (activeView === 'landing') {
      setActiveView('analysis');
    }
    setSearchOpen(false);
    setSearchQuery('');
  };

  return (
    <header className="app-header">
      <div className="header-container">
        {/* Brand Area */}
        <div className="brand-area" onClick={() => setActiveView('landing')}>
          <div className="brand-symbol">
            <TrendingUp size={18} />
          </div>
          <div className="brand-text">
            <span className="brand-name">StockSense <span className="brand-ai-chip">AI</span></span>
            <span className="brand-sub">S&amp;P 500 Intelligence</span>
          </div>
        </div>

        {/* Center: Search & Quick Chips */}
        <div className="header-center">
          <div className="search-container">
            <div
              className="search-input-wrap"
              onClick={() => setSearchOpen(true)}
            >
              <Search size={14} className="search-icon" />
              <input
                type="text"
                className="search-input"
                placeholder={searchOpen ? 'Search 472 constituents...' : `${selectedTicker} - Search S&P 500`}
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onFocus={() => setSearchOpen(true)}
              />
              {searchQuery && (
                <button
                  type="button"
                  className="search-clear"
                  onClick={(e) => {
                    e.stopPropagation();
                    setSearchQuery('');
                  }}
                >
                  <X size={12} />
                </button>
              )}
            </div>

            {searchOpen && (
              <>
                <div
                  className="search-backdrop"
                  style={{ position: 'fixed', inset: 0, zIndex: 150 }}
                  onClick={() => setSearchOpen(false)}
                />
                <div className="search-dropdown">
                  <div className="dropdown-header">
                    Available S&amp;P 500 Constituents ({tickerList.length})
                  </div>
                  <div className="search-results-list">
                    {filteredTickers.length > 0 ? (
                      filteredTickers.slice(0, 50).map((item) => (
                        <button
                          key={item.ticker}
                          type="button"
                          className={`search-result-item ${selectedTicker === item.ticker ? 'selected' : ''}`}
                          onClick={() => handleSelect(item.ticker)}
                        >
                          <div className="result-main">
                            <span className="result-ticker font-mono">{item.ticker}</span>
                            <span className="result-name">{item.name}</span>
                          </div>
                          <div className="result-side">
                            <span className="result-price font-mono">${item.price ? item.price.toFixed(2) : '—'}</span>
                            <span className={`result-sector font-mono ${item.change >= 0 ? 'text-up' : 'text-down'}`}>
                              {item.change >= 0 ? '+' : ''}{item.change ? item.change.toFixed(2) : '0.00'}%
                            </span>
                          </div>
                        </button>
                      ))
                    ) : (
                      <div className="no-results">No constituents matching &ldquo;{searchQuery}&rdquo;</div>
                    )}
                  </div>
                </div>
              </>
            )}
          </div>

          {/* Phase 2: Live Market Status Pill & Indices Tape */}
          {marketStream && (
            <div className="header-live-tape">
              <div
                className="header-market-pill"
                title={`Status: ${marketStream.marketStatus || 'Market Closed'} (${marketStream.timestamp || ''})`}
              >
                <span className={`status-pulse ${marketStream.isOpen ? 'open' : marketStream.marketStatus?.includes('Pre') || marketStream.marketStatus?.includes('After') ? 'pre-post' : 'closed'}`} />
                <span>{marketStream.marketStatus || 'Market Closed'}</span>
              </div>

              {marketStream.indices && (
                <div className="header-indices-strip">
                  {marketStream.indices['^GSPC'] && (
                    <div
                      className="header-index-chip"
                      onClick={() => setActiveView('market')}
                      title="S&P 500 Index — Click to view Market Overview"
                    >
                      <span className="idx-sym">S&amp;P 500</span>
                      <span className="idx-price font-mono">{marketStream.indices['^GSPC'].price ? marketStream.indices['^GSPC'].price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : '—'}</span>
                      <span className={`idx-chg font-mono ${marketStream.indices['^GSPC'].changePct >= 0 ? 'text-up' : 'text-down'}`}>
                        {marketStream.indices['^GSPC'].changePct >= 0 ? '+' : ''}{marketStream.indices['^GSPC'].changePct ? marketStream.indices['^GSPC'].changePct.toFixed(2) : '0.00'}%
                      </span>
                    </div>
                  )}
                  {marketStream.indices['^IXIC'] && (
                    <div
                      className="header-index-chip"
                      onClick={() => setActiveView('market')}
                      title="NASDAQ Composite — Click to view Market Overview"
                    >
                      <span className="idx-sym">NASDAQ</span>
                      <span className="idx-price font-mono">{marketStream.indices['^IXIC'].price ? marketStream.indices['^IXIC'].price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : '—'}</span>
                      <span className={`idx-chg font-mono ${marketStream.indices['^IXIC'].changePct >= 0 ? 'text-up' : 'text-down'}`}>
                        {marketStream.indices['^IXIC'].changePct >= 0 ? '+' : ''}{marketStream.indices['^IXIC'].changePct ? marketStream.indices['^IXIC'].changePct.toFixed(2) : '0.00'}%
                      </span>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Quick Ticker Chips */}
          <div className="quick-chips">
            {tickerKeys.map((tk) => (
              <button
                key={tk}
                type="button"
                className={`ticker-chip ${selectedTicker === tk && activeView === 'analysis' ? 'active' : ''}`}
                onClick={() => {
                  setSelectedTicker(tk);
                  if (activeView === 'landing') setActiveView('analysis');
                }}
              >
                {tk}
              </button>
            ))}
          </div>
        </div>

        {/* Right: Redesigned View Navigation */}
        <div className="header-right">
          <nav className="view-nav">
            {/* Landing */}
            <button
              type="button"
              className={`nav-tab ${activeView === 'landing' ? 'active' : ''}`}
              onClick={() => setActiveView('landing')}
              title="Public Landing Page"
            >
              <span className="nav-icon"><Home size={14} /></span>
              <span>Home</span>
            </button>

            {/* A. Market Overview */}
            <button
              type="button"
              className={`nav-tab ${activeView === 'market' ? 'active' : ''}`}
              onClick={() => setActiveView('market')}
              title="Market Overview & Indices"
            >
              <span className="nav-icon"><BarChart3 size={14} /></span>
              <span>Market</span>
            </button>

            {/* B. Stock Discovery */}
            <button
              type="button"
              className={`nav-tab ${activeView === 'discovery' ? 'active' : ''}`}
              onClick={() => setActiveView('discovery')}
              title="Stock Discovery & Screener"
            >
              <span className="nav-icon"><Compass size={14} /></span>
              <span>Discovery</span>
            </button>

            {/* C. Stock Analysis */}
            <button
              type="button"
              className={`nav-tab ${activeView === 'analysis' ? 'active' : ''}`}
              onClick={() => setActiveView('analysis')}
              title="Individual Stock Analysis & LSTM Forecast"
            >
              <span className="nav-icon"><LineChart size={14} /></span>
              <span>Analysis</span>
            </button>

            {/* D. AI Assistant */}
            <button
              type="button"
              className={`nav-tab ${activeView === 'assistant' ? 'active' : ''}`}
              onClick={() => setActiveView('assistant')}
              title="AI Stock Research Assistant"
            >
              <span className="nav-icon text-ai"><Sparkles size={14} /></span>
              <span>AI Assistant</span>
            </button>

            {/* E. Watchlist */}
            <button
              type="button"
              className={`nav-tab ${activeView === 'watchlist' ? 'active' : ''}`}
              onClick={() => setActiveView('watchlist')}
              title="Personal Stock Watchlist"
            >
              <span className="nav-icon"><Bookmark size={14} /></span>
              <span>Watchlist</span>
              {watchlistCount > 0 && (
                <span className="nav-count-badge font-mono">{watchlistCount}</span>
              )}
            </button>

            {/* Research & Backtest */}
            <button
              type="button"
              className={`nav-tab ${activeView === 'research' ? 'active' : ''}`}
              onClick={() => setActiveView('research')}
              title="Quantitative ML Benchmarks & PnL Backtester"
            >
              <span className="nav-icon"><FlaskConical size={14} /></span>
              <span>Research</span>
            </button>
          </nav>
        </div>
      </div>
    </header>
  );
}
