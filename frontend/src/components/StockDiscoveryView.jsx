import React, { useState, useMemo } from 'react';
import {
  Compass,
  Search,
  SlidersHorizontal,
  ArrowUpDown,
  TrendingUp,
  TrendingDown,
  Layers,
  ChevronRight,
  Filter,
  Info,
  Check,
} from 'lucide-react';

export default function StockDiscoveryView({
  tickers = [],
  onSelectTicker,
  onNavigate,
}) {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedSector, setSelectedSector] = useState('ALL');
  const [directionFilter, setDirectionFilter] = useState('ALL'); // 'ALL' | 'UP' | 'DOWN'
  const [sortBy, setSortBy] = useState('return_desc'); // 'return_desc' | 'return_asc' | 'price_desc' | 'ticker'

  // Extract unique sectors
  const availableSectors = useMemo(() => {
    const set = new Set();
    tickers.forEach((t) => {
      if (t.sector) set.add(t.sector);
    });
    return Array.from(set).sort();
  }, [tickers]);

  // Filtered & Sorted Tickers
  const filteredTickers = useMemo(() => {
    if (!tickers || tickers.length === 0) return [];

    let list = tickers.filter((t) => {
      const q = searchQuery.toLowerCase().trim();
      const matchSearch =
        !q ||
        t.ticker.toLowerCase().includes(q) ||
        (t.name && t.name.toLowerCase().includes(q));

      const matchSector =
        selectedSector === 'ALL' || t.sector === selectedSector;

      const matchDir =
        directionFilter === 'ALL' ||
        (directionFilter === 'UP' && t.changePct > 0) ||
        (directionFilter === 'DOWN' && t.changePct < 0);

      return matchSearch && matchSector && matchDir;
    });

    // Sorting
    list.sort((a, b) => {
      if (sortBy === 'return_desc') return (b.changePct || 0) - (a.changePct || 0);
      if (sortBy === 'return_asc') return (a.changePct || 0) - (b.changePct || 0);
      if (sortBy === 'price_desc') return (b.latestPrice || 0) - (a.latestPrice || 0);
      if (sortBy === 'ticker') return a.ticker.localeCompare(b.ticker);
      return 0;
    });

    return list;
  }, [tickers, searchQuery, selectedSector, directionFilter, sortBy]);

  const handleStockClick = (tk) => {
    if (onSelectTicker) onSelectTicker(tk);
    if (onNavigate) onNavigate('analysis');
  };

  return (
    <div className="discovery-view">
      {/* 1. Header Banner */}
      <div className="view-header-strip">
        <div className="view-title-group">
          <div className="view-icon-badge emerald">
            <Compass size={20} />
          </div>
          <div>
            <h1 className="view-title">Stock Discovery &amp; Screener</h1>
            <p className="view-subtitle">
              Screen and filter 472 S&amp;P 500 constituents with latest available market sessions, sectors, and trading ranges.
            </p>
          </div>
        </div>

        <div className="view-actions">
          <div className="phase-pill">
            <span className="phase-dot" />
            <span>Phase 2: Live Market Data • 472 Constituents Spliced</span>
          </div>
        </div>
      </div>

      {/* 2. Screener Controls Bar */}
      <div className="screener-controls-card">
        <div className="controls-row">
          {/* Search Box */}
          <div className="screener-search-wrap">
            <Search size={15} className="search-icon" />
            <input
              type="text"
              className="screener-search-input"
              placeholder="Search by ticker symbol or company name..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
            {searchQuery && (
              <button
                type="button"
                className="clear-search-btn"
                onClick={() => setSearchQuery('')}
              >
                Clear
              </button>
            )}
          </div>

          {/* Direction Filter Pills */}
          <div className="filter-pill-group">
            <button
              type="button"
              className={`filter-btn ${directionFilter === 'ALL' ? 'active' : ''}`}
              onClick={() => setDirectionFilter('ALL')}
            >
              All Movements
            </button>
            <button
              type="button"
              className={`filter-btn ${directionFilter === 'UP' ? 'active text-up' : ''}`}
              onClick={() => setDirectionFilter('UP')}
            >
              <TrendingUp size={13} />
              <span>Gainers Only</span>
            </button>
            <button
              type="button"
              className={`filter-btn ${directionFilter === 'DOWN' ? 'active text-down' : ''}`}
              onClick={() => setDirectionFilter('DOWN')}
            >
              <TrendingDown size={13} />
              <span>Decliners Only</span>
            </button>
          </div>

          {/* Sort Selector */}
          <div className="sort-selector-wrap">
            <ArrowUpDown size={14} className="sort-icon" />
            <select
              className="sort-dropdown"
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
            >
              <option value="return_desc">Highest Daily Gain %</option>
              <option value="return_asc">Largest Daily Loss %</option>
              <option value="price_desc">Highest Price ($)</option>
              <option value="ticker">Ticker Symbol (A-Z)</option>
            </select>
          </div>
        </div>

        {/* Sector Filter Chips */}
        <div className="sector-filter-row">
          <span className="sector-filter-label">Sector:</span>
          <div className="sector-scroll-chips">
            <button
              type="button"
              className={`sector-chip ${selectedSector === 'ALL' ? 'active' : ''}`}
              onClick={() => setSelectedSector('ALL')}
            >
              All Sectors ({tickers.length})
            </button>
            {availableSectors.map((sec) => (
              <button
                key={sec}
                type="button"
                className={`sector-chip ${selectedSector === sec ? 'active' : ''}`}
                onClick={() => setSelectedSector(sec)}
              >
                {sec}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* 3. Results Count Bar */}
      <div className="results-status-bar">
        <span className="results-count font-mono">
          Showing <strong>{filteredTickers.length}</strong> of {tickers.length} S&amp;P 500 Equities
        </span>
        <span className="results-hint">
          Click any row to open full technical chart and PyTorch LSTM predictions
        </span>
      </div>

      {/* 4. Screener Table */}
      <div className="screener-table-card">
        <div className="table-wrapper">
          <table className="screener-table">
            <thead>
              <tr>
                <th className="th-ticker">Ticker &amp; Company</th>
                <th className="th-sector">Sector</th>
                <th className="th-price text-right">Latest Price</th>
                <th className="th-change text-right">Daily Change</th>
                <th className="th-range text-right">52-Week Range</th>
                <th className="th-action text-center">Action</th>
              </tr>
            </thead>
            <tbody>
              {filteredTickers.length > 0 ? (
                filteredTickers.map((stk) => {
                  const isUp = stk.changePct >= 0;
                  return (
                    <tr
                      key={stk.ticker}
                      className="screener-row"
                      onClick={() => handleStockClick(stk.ticker)}
                    >
                      <td className="td-ticker">
                        <div className="td-ticker-box">
                          <span className="td-symbol font-mono">{stk.ticker}</span>
                          <span className="td-name">{stk.name}</span>
                        </div>
                      </td>

                      <td className="td-sector">
                        <span className="td-sector-badge">{stk.sector}</span>
                      </td>

                      <td className="td-price text-right font-mono">
                        ${stk.latestPrice ? stk.latestPrice.toFixed(2) : '—'}
                      </td>

                      <td className="td-change text-right font-mono">
                        <span className={`change-badge ${isUp ? 'up' : 'down'}`}>
                          {isUp ? '+' : ''}{stk.changePct ? stk.changePct.toFixed(2) : '0.00'}%
                        </span>
                      </td>

                      <td className="td-range text-right font-mono">
                        <span className="range-text">
                          ${stk.low52w ? stk.low52w.toFixed(2) : '—'} &ndash; ${stk.high52w ? stk.high52w.toFixed(2) : '—'}
                        </span>
                      </td>

                      <td className="td-action text-center" onClick={(e) => e.stopPropagation()}>
                        <button
                          type="button"
                          className="btn-screener-action"
                          onClick={() => handleStockClick(stk.ticker)}
                        >
                          <span>Analyze</span>
                          <ChevronRight size={13} />
                        </button>
                      </td>
                    </tr>
                  );
                })
              ) : (
                <tr>
                  <td colSpan={6} className="no-screener-results">
                    No S&amp;P 500 constituents match your active search and filter criteria.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* 5. Phase 3 Notice */}
      <div className="market-roadmap-notice">
        <Info size={16} className="text-ai" />
        <span>
          <strong>Phase 3 Roadmap:</strong> Transparent multi-factor Research Recommendation Scoring (combining Momentum, Moving Average Stretch, Relative Volume, 20-Day Volatility, and PyTorch LSTM probabilities) will appear as interactive columns in this screener.
        </span>
      </div>
    </div>
  );
}
