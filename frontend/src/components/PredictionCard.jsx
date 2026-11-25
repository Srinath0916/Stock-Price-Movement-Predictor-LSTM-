import React, { useState, useEffect } from 'react';
import { Sparkles, TrendingUp, TrendingDown, AlertTriangle, RefreshCw, ChevronDown, ChevronUp, XCircle, Globe, Database } from 'lucide-react';

export default function PredictionCard({
  ticker,
  stockData,
  isOpen,
  onClose,
}) {
  const [loading, setLoading] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);
  const [apiError, setApiError] = useState(null);
  const [useLiveMarketData, setUseLiveMarketData] = useState(true);
  const [showTechDetails, setShowTechDetails] = useState(false);

  const executeInference = (isLive) => {
    setLoading(true);
    setApiError(null);
    setPredictionResult(null);

    // Actual HTTP call to FastAPI POST /api/predict
    fetch('/api/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        ticker: ticker,
        threshold: 0.50,
        use_live_market_data: isLive,
      }),
    })
      .then(async (res) => {
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.detail || `HTTP ${res.status}: ${res.statusText}`);
        }
        return data;
      })
      .then((data) => {
        setPredictionResult(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error('FastAPI Prediction Error:', err);
        // Never silently fall back to hardcoded predictions. Show real provider error.
        setApiError(err.message || 'Failed to connect to backend prediction service.');
        setLoading(false);
      });
  };

  useEffect(() => {
    if (isOpen) {
      executeInference(useLiveMarketData);
    } else {
      setLoading(false);
      setPredictionResult(null);
      setApiError(null);
      setShowTechDetails(false);
    }
  }, [isOpen, ticker, useLiveMarketData]);

  if (!isOpen) return null;

  const isUp = predictionResult?.prediction === 'UP';
  const prob = predictionResult?.probability !== undefined ? (predictionResult.probability * 100).toFixed(1) : null;
  const companyName = predictionResult?.companyName || stockData?.name || ticker;
  const lastClose = predictionResult?.baseClose || 0;
  const inputStartDate = predictionResult?.inputStartDate || '—';
  const inputEndDate = predictionResult?.inputEndDate || '—';
  const targetSession = predictionResult?.targetSession || 'Next Trading Session';
  const modelVersion = predictionResult?.modelVersion || 'StockLSTM_ExpC_Hidden128';
  const decisionThreshold = predictionResult?.decisionThreshold ?? 0.50;
  const isLive = predictionResult?.isLiveMarketData ?? useLiveMarketData;
  const dataSource = predictionResult?.dataSource || (useLiveMarketData ? 'Live Market Data (Yahoo Finance)' : 'Historical S&P 500 Dataset');
  const latestDate = predictionResult?.latestMarketDate || inputEndDate;

  return (
    <div className="prediction-result-panel">
      {/* Data Source Selector Pill Bar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px', flexWrap: 'wrap', gap: '10px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span className="stat-label">Market Data Mode:</span>
          <div className="toggle-pill-group">
            <button
              type="button"
              className={`toggle-pill ${useLiveMarketData ? 'active' : ''}`}
              onClick={() => setUseLiveMarketData(true)}
              disabled={loading}
              title="Fetch fresh real-market OHLCV via Yahoo Finance"
            >
              <Globe size={12} style={{ display: 'inline', marginRight: '4px' }} />
              Live Market Data
            </button>
            <button
              type="button"
              className={`toggle-pill ${!useLiveMarketData ? 'active' : ''}`}
              onClick={() => setUseLiveMarketData(false)}
              disabled={loading}
              title="Use benchmark historical S&P 500 dataset"
            >
              <Database size={12} style={{ display: 'inline', marginRight: '4px' }} />
              Historical Dataset
            </button>
          </div>
        </div>

        <button
          type="button"
          className="command-copy-button"
          onClick={() => executeInference(useLiveMarketData)}
          disabled={loading}
          style={{ padding: '4px 10px', fontSize: '12px' }}
        >
          <RefreshCw size={12} className={loading ? 'animate-spin' : ''} />
          <span>Re-run Inference</span>
        </button>
      </div>

      {loading ? (
        <div className="prediction-evaluating-state">
          <RefreshCw size={24} className="animate-spin text-accent" />
          <div className="evaluating-text-group">
            <span className="eval-headline">Running PyTorch LSTM Inference for {ticker}...</span>
            <span className="eval-subtext">
              {useLiveMarketData
                ? `Fetching recent market OHLCV for ${ticker} from Yahoo Finance and computing 8 features...`
                : `Extracting 30-session window for ${ticker} from historical S&P 500 dataset...`}
            </span>
          </div>
        </div>
      ) : apiError ? (
        <div className="prediction-unavailable-content">
          <div className="unavailable-header">
            <XCircle size={22} className="text-down" />
            <div>
              <h4 className="unavailable-title" style={{ color: 'var(--color-down)' }}>
                Market Data Provider Error for {ticker}
              </h4>
              <p className="unavailable-text" style={{ marginTop: '4px' }}>
                {apiError}
              </p>
              <p className="unavailable-text" style={{ marginTop: '8px', fontSize: '12px', color: 'var(--text-muted)' }}>
                Per project integrity rules, predictions are never fabricated or silently populated with stale data when live provider queries fail.
              </p>
            </div>
          </div>
          <div style={{ display: 'flex', gap: '10px', marginTop: '12px' }}>
            <button
              type="button"
              className="tech-toggle-btn"
              onClick={() => setUseLiveMarketData(false)}
            >
              Try Historical Dataset
            </button>
            <button type="button" className="close-prediction-btn" onClick={onClose}>
              Close Notice
            </button>
          </div>
        </div>
      ) : predictionResult ? (
        <div className="prediction-complete-content">
          {/* Active Status Badge */}
          <div
            className="offline-model-pill"
            style={{
              borderColor: isLive ? 'rgba(16, 185, 129, 0.4)' : 'rgba(245, 158, 11, 0.4)',
              background: isLive ? 'rgba(16, 185, 129, 0.08)' : 'rgba(245, 158, 11, 0.08)',
            }}
          >
            <span
              className="offline-indicator-dot"
              style={{
                backgroundColor: isLive ? '#10b981' : '#f59e0b',
                boxShadow: isLive ? '0 0 8px #10b981' : '0 0 8px #f59e0b',
              }}
            />
            <span className="offline-label-text" style={{ color: isLive ? '#10b981' : '#f59e0b' }}>
              {isLive
                ? `Live Market Data (${dataSource}) • Latest Market Session: ${latestDate}`
                : `Historical S&P 500 Dataset • Session as of ${latestDate}`}
            </span>
          </div>

          {/* Context Bar */}
          <div className="prediction-date-context-bar font-mono">
            <div className="date-context-item">
              <span className="context-label">Input Data Period:</span>
              <span className="context-value">{inputStartDate} &ndash; {inputEndDate} (30 sessions)</span>
            </div>
            <div className="date-context-sep">&bull;</div>
            <div className="date-context-item">
              <span className="context-label">Target Session:</span>
              <span className="context-value highlight">{targetSession}</span>
            </div>
            <div className="date-context-sep">&bull;</div>
            <div className="date-context-item">
              <span className="context-label">Base Close:</span>
              <span className="context-value">${Number(lastClose).toFixed(2)} ({latestDate})</span>
            </div>
          </div>

          {/* Direction & Probability Banner */}
          <div className={`prediction-outcome-banner ${isUp ? 'outcome-up' : 'outcome-down'}`}>
            <div className="outcome-left">
              <span className="outcome-eyebrow">Model Direction Forecast</span>
              <div className="outcome-direction">
                {isUp ? <TrendingUp size={32} /> : <TrendingDown size={32} />}
                <span className="direction-label">{isUp ? 'PREDICTED UP' : 'PREDICTED DOWN'}</span>
              </div>
              <span className="outcome-session font-mono">Target: {targetSession}</span>
            </div>

            <div className="outcome-middle">
              <span className="prob-eyebrow">Estimated UP Probability</span>
              <span className="prob-number font-mono">{prob}%</span>
              <span className="prob-caption">Model&rsquo;s estimated probability of an UP outcome</span>
              <span className="prob-subcaption font-mono">(Operating cutoff threshold: {decisionThreshold.toFixed(2)})</span>
            </div>

            <div className="outcome-right font-mono">
              <div className="spec-row">
                <span className="spec-k">Target Date:</span>
                <span className="spec-v">{targetSession}</span>
              </div>
              <div className="spec-row">
                <span className="spec-k">Input Window:</span>
                <span className="spec-v">30 trading sessions</span>
              </div>
              <div className="spec-row">
                <span className="spec-k">Base Price:</span>
                <span className="spec-v">${Number(lastClose).toFixed(2)}</span>
              </div>
            </div>
          </div>

          {/* Plain English Meaning */}
          <div className="prediction-plain-explanation">
            <p>
              <strong>What this means:</strong> The PyTorch LSTM neural network evaluated the 30 consecutive trading sessions from <strong>{inputStartDate} to {inputEndDate}</strong> for <strong>{companyName} ({ticker})</strong> using {dataSource}. Based on normalized price momentum, volume dynamics, volatility, and moving-average stretch ratios, the model computes an estimated <strong>{prob}% probability of an UP outcome</strong> (closing higher on {targetSession} relative to the baseline close of <strong>${Number(lastClose).toFixed(2)}</strong>).
            </p>
            <p className="prob-clarification-callout">
              <strong>Probability vs. Profit Clarification:</strong> The {prob}% figure is the model&rsquo;s estimated directional probability of an UP outcome, not a guaranteed probability of profit. Directional probability does not guarantee financial gain, nor does it measure price swing magnitude, trading spreads, or subsequent market reversals.
            </p>
          </div>

          {/* Caution Banner */}
          <div className="prediction-caution-banner">
            <AlertTriangle size={18} className="caution-icon" />
            <div className="caution-text-block">
              <strong>Caution &mdash; The model can be wrong:</strong> In out-of-sample holdout testing across 458,428 sequences, this model achieved <strong>51.75% directional accuracy</strong>, compared to the <strong>51.64% majority baseline</strong> (+0.11% difference). Because the broader stock market trends upward over multi-year horizons, the model produces a high number of false alarms (false positives). This prediction is experimental research and must not be used as financial advice.
            </div>
          </div>

          {/* Technical Details Toggle */}
          <div className="tech-toggle-bar">
            <button
              type="button"
              className="tech-toggle-btn"
              onClick={() => setShowTechDetails(!showTechDetails)}
            >
              <span>{showTechDetails ? 'Hide Model Specs & Features' : 'View Model Specs & Features'}</span>
              {showTechDetails ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
            </button>
            <button
              type="button"
              className="close-prediction-btn"
              onClick={onClose}
            >
              Close Result
            </button>
          </div>

          {showTechDetails && (
            <div className="tech-specs-box">
              <div className="tech-specs-grid">
                <div><span className="spec-label">Model Checkpoint:</span> <code className="spec-code">{predictionResult.modelCheckpoint}</code></div>
                <div><span className="spec-label">Model Version:</span> <span className="font-mono">{modelVersion}</span></div>
                <div><span className="spec-label">Decision Cutoff:</span> <span className="font-mono">{decisionThreshold.toFixed(2)}</span></div>
                <div><span className="spec-label">Raw Sigmoid Logit:</span> <span className="font-mono">{predictionResult.logit}</span></div>
                <div><span className="spec-label">Architecture:</span> <span className="font-mono">StockLSTM (2 layers, 128 hidden)</span></div>
                <div><span className="spec-label">Data Provider:</span> <span className="font-mono">{dataSource}</span></div>
                <div><span className="spec-label">Latest Market Date:</span> <span className="font-mono">{latestDate}</span></div>
                <div><span className="spec-label">Execution Mode:</span> <span className="font-mono">Real-time PyTorch forward pass via FastAPI</span></div>
              </div>

              {predictionResult.latestFeatures && (
                <div style={{ marginTop: '16px', paddingTop: '14px', borderTop: '1px solid var(--border-subtle)' }}>
                  <div className="stat-label" style={{ marginBottom: '8px' }}>Evaluated Feature Inputs (Latest Session):</div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '8px', fontSize: '12px' }}>
                    {Object.entries(predictionResult.latestFeatures).map(([k, v]) => (
                      <div key={k} style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 8px', background: 'var(--bg-surface)', borderRadius: '4px' }}>
                        <span className="text-muted font-mono">{k}:</span>
                        <span className="font-mono">{v !== null ? v : '—'}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      ) : null}
    </div>
  );
}
