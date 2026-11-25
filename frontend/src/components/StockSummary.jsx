import React from 'react';
import { TrendingUp, TrendingDown, Sparkles, Star, Bookmark } from 'lucide-react';

export default function StockSummary({
  ticker,
  stockData,
  predictionData,
  onPredictClick,
  predictionActive,
  isWatchlisted = false,
  onToggleWatchlist,
}) {
  if (!stockData) return null;

  const { name, sector, stats = {}, history = [] } = stockData;
  const latestSession = history[history.length - 1] || {};

  const currentPrice = stats.latestPrice || latestSession.close || 0;
  const dailyReturn = latestSession.dailyReturn !== undefined ? latestSession.dailyReturn : 0;
  const dollarChange = currentPrice * (dailyReturn / 100);

  const low52 = stats.low52w || currentPrice * 0.75;
  const high52 = stats.high52w || currentPrice * 1.25;
  const rangePercent = Math.max(0, Math.min(100, ((currentPrice - low52) / Math.max(0.01, high52 - low52)) * 100));

  return (
    <div className="stock-summary-bar">
      {/* Primary Ticker & Price Block */}
      <div className="summary-main">
        <div className="summary-ticker-row">
          <span className="summary-ticker font-mono">{ticker}</span>
          <span className="summary-name">{name}</span>
          <span className="summary-badge">{sector}</span>

          {onToggleWatchlist && (
            <button
              type="button"
              className={`watchlist-star-btn ${isWatchlisted ? 'active' : ''}`}
              title={isWatchlisted ? 'Remove from Watchlist' : 'Add to Watchlist'}
              onClick={onToggleWatchlist}
            >
              <Star size={16} fill={isWatchlisted ? '#F59E0B' : 'none'} color={isWatchlisted ? '#F59E0B' : '#64748B'} />
            </button>
          )}
        </div>

        <div className="summary-price-row">
          <span className="summary-price font-mono">${currentPrice.toFixed(2)}</span>
          <span className={`change-badge ${dailyReturn >= 0 ? 'badge-up' : 'badge-down'}`}>
            {dailyReturn >= 0 ? <TrendingUp size={14} /> : <TrendingDown size={14} />}
            <span>{dailyReturn >= 0 ? '+' : ''}{dailyReturn.toFixed(2)}%</span>
            <span className="font-mono">({dollarChange >= 0 ? '+' : ''}${dollarChange.toFixed(2)})</span>
          </span>
          <span className="summary-date-tag font-mono">
            Close as of {stats.endDate || latestSession.date || 'Feb 20, 2026'}
          </span>

          {latestSession.rsi14 !== undefined && latestSession.rsi14 !== null && (
            <span
              className="rsi-hud-badge font-mono"
              title="Wilder's 14-period Relative Strength Index"
            >
              RSI(14): {Number(latestSession.rsi14).toFixed(1)}
            </span>
          )}

          <span
            className="spliced-data-tag font-mono"
            title="Continuous historical Parquet data combined with Yahoo Finance recent daily sessions"
          >
            Live Spliced Data
          </span>
        </div>
      </div>

      {/* 52-Week Range Bar */}
      <div className="summary-middle">
        <div className="range-stat-block">
          <span className="stat-label">52-Week Price Range</span>
          <div className="range-slider-wrap">
            <div className="range-track">
              <div className="range-fill" style={{ width: `${rangePercent}%` }} />
              <div className="range-current-dot" style={{ left: `${rangePercent}%` }} />
            </div>
            <div className="range-labels font-mono">
              <span>${low52.toFixed(2)}</span>
              <span>${high52.toFixed(2)}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Primary Action: Predict Next Trading Day */}
      <div className="summary-action-block">
        <button
          type="button"
          className={`primary-predict-btn ${predictionActive ? 'active' : ''}`}
          onClick={onPredictClick}
        >
          <Sparkles size={16} />
          <span>{predictionActive ? 'Hide Prediction Card' : 'Predict Next Trading Day'}</span>
        </button>
      </div>
    </div>
  );
}
