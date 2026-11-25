import React from 'react';
import {
  Bookmark,
  Star,
  Trash2,
  TrendingUp,
  TrendingDown,
  ArrowRight,
  Plus,
  Compass,
  Zap,
} from 'lucide-react';

export default function WatchlistView({
  watchlist = [],
  tickers = [],
  onRemoveFromWatchlist,
  onAddToWatchlist,
  onSelectTicker,
  onNavigate,
}) {
  // Map watchlist symbols to rich ticker objects
  const tickerMap = new Map();
  tickers.forEach((t) => tickerMap.set(t.ticker, t));

  const watchlistItems = watchlist.map((symbol) => {
    return (
      tickerMap.get(symbol) || {
        ticker: symbol,
        name: `${symbol} Corporation`,
        sector: 'S&P 500 Equities',
        latestPrice: 0,
        changePct: 0,
        high52w: 0,
        low52w: 0,
      }
    );
  });

  const handleAnalyze = (symbol) => {
    if (onSelectTicker) onSelectTicker(symbol);
    if (onNavigate) onNavigate('analysis');
  };

  const sampleSuggestions = ['AAPL', 'NVDA', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA'].filter(
    (sym) => !watchlist.includes(sym)
  );

  return (
    <div className="watchlist-view">
      {/* 1. Header Banner */}
      <div className="view-header-strip">
        <div className="view-title-group">
          <div className="view-icon-badge emerald">
            <Bookmark size={20} />
          </div>
          <div>
            <h1 className="view-title">Personal Stock Watchlist</h1>
            <p className="view-subtitle">
              Track your favorite S&amp;P 500 equities, monitor percentage movements, and trigger 1-click LSTM predictions.
            </p>
          </div>
        </div>

        <div className="view-actions">
          <span className="badge badge-emerald font-mono">
            {watchlist.length} {watchlist.length === 1 ? 'Equity' : 'Equities'} Saved
          </span>
        </div>
      </div>

      {/* 2. Watchlist Grid or Empty State */}
      {watchlistItems.length > 0 ? (
        <div className="watchlist-grid">
          {watchlistItems.map((stk) => {
            const isUp = stk.changePct >= 0;
            return (
              <div key={stk.ticker} className="watchlist-card">
                <div className="wl-card-top">
                  <div className="wl-card-id">
                    <span className="wl-ticker font-mono">{stk.ticker}</span>
                    <span className="wl-name">{stk.name}</span>
                    <span className="wl-sector">{stk.sector}</span>
                  </div>

                  <button
                    type="button"
                    className="wl-remove-btn"
                    title={`Remove ${stk.ticker} from watchlist`}
                    onClick={() => onRemoveFromWatchlist(stk.ticker)}
                  >
                    <Trash2 size={15} />
                  </button>
                </div>

                <div className="wl-card-pricing">
                  <div className="wl-price-col">
                    <span className="wl-price-label">Latest Price</span>
                    <span className="wl-price-val font-mono">
                      ${stk.latestPrice ? stk.latestPrice.toFixed(2) : '—'}
                    </span>
                  </div>

                  <div className="wl-chg-col">
                    <span className="wl-price-label">24h Movement</span>
                    <span className={`change-badge ${isUp ? 'up' : 'down'} font-mono`}>
                      {isUp ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
                      <span>{isUp ? '+' : ''}{stk.changePct ? stk.changePct.toFixed(2) : '0.00'}%</span>
                    </span>
                  </div>
                </div>

                <div className="wl-card-range font-mono">
                  <span className="range-label">52W Range:</span>
                  <span className="range-val">
                    ${stk.low52w ? stk.low52w.toFixed(2) : '—'} &ndash; ${stk.high52w ? stk.high52w.toFixed(2) : '—'}
                  </span>
                </div>

                <div className="wl-card-actions">
                  <button
                    type="button"
                    className="btn-wl-analyze"
                    onClick={() => handleAnalyze(stk.ticker)}
                  >
                    <span>Analyze &amp; Predict</span>
                    <ArrowRight size={14} />
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="watchlist-empty-card">
          <div className="empty-icon-wrap">
            <Star size={32} />
          </div>
          <h3 className="empty-title">Your Watchlist is Empty</h3>
          <p className="empty-desc">
            Save stocks you are actively monitoring to track daily percentage movements and quickly trigger PyTorch LSTM directional predictions.
          </p>

          {sampleSuggestions.length > 0 && (
            <div className="empty-suggestions-box">
              <span className="sugg-label">Quickly Add Prominent Equities:</span>
              <div className="sugg-chips">
                {sampleSuggestions.map((sym) => (
                  <button
                    key={sym}
                    type="button"
                    className="sugg-chip"
                    onClick={() => onAddToWatchlist(sym)}
                  >
                    <Plus size={12} />
                    <span>{sym}</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="empty-action-row">
            <button
              type="button"
              className="btn-primary-sm"
              onClick={() => onNavigate('discovery')}
            >
              <Compass size={14} />
              <span>Browse Stock Discovery</span>
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
