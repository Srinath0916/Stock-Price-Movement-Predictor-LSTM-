import React, { useState, useEffect, useMemo } from 'react';
import { TrendingUp, RefreshCw } from 'lucide-react';

export default function BacktestView({ selectedTicker = 'AAPL', tickerSamples = {} }) {
  const [ticker, setTicker] = useState(selectedTicker);
  const [strategyMode, setStrategyMode] = useState('long_only');
  const [threshold, setThreshold] = useState(0.50);
  const [initialCapital, setInitialCapital] = useState(10000);
  const [feeBps, setFeeBps] = useState(5.0);

  const [loadingBacktest, setLoadingBacktest] = useState(false);
  const [backtestData, setBacktestData] = useState(null);
  const [backtestError, setBacktestError] = useState(null);

  const [walkForwardRegimes, setWalkForwardRegimes] = useState([]);
  const [expandingFolds, setExpandingFolds] = useState([]);
  const [wfMode, setWfMode] = useState('expanding'); // 'expanding' | 'regimes'
  const [loadingWF, setLoadingWF] = useState(false);
  const [hoverIdx, setHoverIdx] = useState(null);
  const [allTickers, setAllTickers] = useState([]);

  // Fetch full ticker list for backtest dropdown
  useEffect(() => {
    fetch('/api/tickers')
      .then((res) => (res.ok ? res.json() : []))
      .then((data) => {
        if (Array.isArray(data) && data.length > 0) {
          setAllTickers(data.map((d) => d.ticker));
        }
      })
      .catch((err) => console.warn('Backtest ticker fetch:', err));
  }, []);

  // Fetch Expanding Walk-Forward Folds
  useEffect(() => {
    fetch('/api/walk-forward/expanding')
      .then((res) => (res.ok ? res.json() : []))
      .then((data) => {
        if (Array.isArray(data)) setExpandingFolds(data);
      })
      .catch((err) => console.warn('Expanding walk forward fetch:', err));
  }, []);

  const defaultTickers = Object.keys(tickerSamples).length > 0
    ? Object.keys(tickerSamples)
    : ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'JPM', 'JNJ', 'CAT', 'XOM'];

  const availableTickers = useMemo(() => {
    if (!allTickers.length) return defaultTickers;
    const remaining = allTickers.filter((tk) => !defaultTickers.includes(tk));
    return [...defaultTickers, ...remaining];
  }, [allTickers, defaultTickers]);

  // Fetch Walk-Forward Regimes on mount
  useEffect(() => {
    setLoadingWF(true);
    fetch('/api/walk-forward')
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        setWalkForwardRegimes(data);
        setLoadingWF(false);
      })
      .catch((err) => {
        console.warn('Could not fetch /api/walk-forward:', err);
        setLoadingWF(false);
      });
  }, []);

  // Fetch or execute backtest
  const runBacktest = () => {
    setLoadingBacktest(true);
    setBacktestError(null);

    fetch('/api/backtest', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        ticker: ticker,
        start_date: '2022-03-25',
        end_date: '2026-02-19',
        strategy_mode: strategyMode,
        threshold: threshold,
        initial_capital: initialCapital,
        transaction_cost_bps: feeBps,
      }),
    })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
        return res.json();
      })
      .then((data) => {
        setBacktestData(data);
        setLoadingBacktest(false);
      })
      .catch((err) => {
        console.error('Backtest API error:', err);
        setBacktestError('Could not execute backtest. Ensure the FastAPI backend is running on port 8000.');
        setLoadingBacktest(false);
      });
  };

  useEffect(() => {
    runBacktest();
  }, [ticker, strategyMode]);

  const metrics = backtestData?.metrics;
  const equityCurve = backtestData?.equityCurve || [];

  // SVG Equity Curve Math
  const chartMath = useMemo(() => {
    if (!equityCurve || equityCurve.length === 0) return null;

    const width = 1100;
    const height = 360;
    const pad = { top: 25, right: 65, bottom: 40, left: 65 };
    const innerW = width - pad.left - pad.right;
    const innerH = height - pad.top - pad.bottom;

    const stratVals = equityCurve.map((d) => d.strategyEquity);
    const benchVals = equityCurve.map((d) => d.benchmarkEquity);
    const allVals = [...stratVals, ...benchVals];

    const minVal = Math.min(...allVals) * 0.98;
    const maxVal = Math.max(...allVals) * 1.02;

    const xScale = (i) => pad.left + (i / Math.max(1, equityCurve.length - 1)) * innerW;
    const yScale = (v) => pad.top + innerH - ((v - minVal) / Math.max(1, maxVal - minVal)) * innerH;

    const buildPath = (key) => {
      const pts = equityCurve.map((d, i) => `${xScale(i).toFixed(1)},${yScale(d[key]).toFixed(1)}`);
      return pts.length ? `M ${pts.join(' L ')}` : '';
    };

    const stratPath = buildPath('strategyEquity');
    const benchPath = buildPath('benchmarkEquity');

    // Horizontal Y reference ticks (5 levels)
    const yTicks = [0, 0.25, 0.5, 0.75, 1.0].map((pct) => minVal + pct * (maxVal - minVal));

    // X date labels
    const xTickIndices = [
      0,
      Math.floor(equityCurve.length * 0.25),
      Math.floor(equityCurve.length * 0.5),
      Math.floor(equityCurve.length * 0.75),
      equityCurve.length - 1,
    ];

    return {
      width,
      height,
      pad,
      innerW,
      innerH,
      minVal,
      maxVal,
      stratPath,
      benchPath,
      yTicks,
      xTickIndices,
      xScale,
      yScale,
    };
  }, [equityCurve]);

  return (
    <div className="research-view">
      {/* Header */}
      <div className="research-header">
        <div>
          <div className="view-badge">
            <TrendingUp size={14} />
            <span>Quantitative Strategy Simulation</span>
          </div>
          <h2 className="research-title">Economic Backtesting &amp; Walk-Forward Regimes</h2>
          <p className="research-subtitle">
            Simulate following the LSTM model&rsquo;s directional calls over the out-of-sample holdout test partition (2022 &ndash; 2026) with transaction cost modeling.
          </p>
        </div>

        <div className="research-header-actions">
          <button
            type="button"
            className="command-copy-button"
            onClick={runBacktest}
            disabled={loadingBacktest}
          >
            <RefreshCw size={14} className={loadingBacktest ? 'animate-spin' : ''} />
            <span>{loadingBacktest ? 'Simulating...' : 'Run Simulation'}</span>
          </button>
        </div>
      </div>

      {/* Interactive Backtest Controls */}
      <section className="research-section">
        <div className="section-title-row">
          <h3 className="section-heading">Strategy Parameters</h3>
          <span className="section-pill font-mono">Out-of-Sample Window: Mar 2022 &ndash; Feb 2026</span>
        </div>

        <div className="stock-summary-bar" style={{ padding: '16px 20px', gap: '16px', flexWrap: 'wrap' }}>
          {/* Ticker Selector */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <label className="stat-label">Asset Ticker</label>
            <select
              className="select-input"
              value={ticker}
              onChange={(e) => setTicker(e.target.value)}
              style={{ minWidth: '120px', padding: '6px 12px' }}
            >
              {availableTickers.map((tk) => (
                <option key={tk} value={tk}>{tk}</option>
              ))}
            </select>
          </div>

          {/* Strategy Mode Toggle */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <label className="stat-label">Strategy Mode</label>
            <div className="toggle-pill-group">
              <button
                type="button"
                className={`toggle-pill ${strategyMode === 'long_only' ? 'active' : ''}`}
                onClick={() => setStrategyMode('long_only')}
              >
                Long-Only
              </button>
              <button
                type="button"
                className={`toggle-pill ${strategyMode === 'long_short' ? 'active' : ''}`}
                onClick={() => setStrategyMode('long_short')}
              >
                Long-Short
              </button>
            </div>
          </div>

          {/* Threshold Slider */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', minWidth: '180px' }}>
            <label className="stat-label">
              Decision Cutoff: <span className="font-mono text-accent">{threshold.toFixed(2)}</span>
            </label>
            <input
              type="range"
              min="0.50"
              max="0.65"
              step="0.01"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              style={{ accentColor: 'var(--accent-primary)', cursor: 'pointer' }}
            />
          </div>

          {/* Initial Capital */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <label className="stat-label">Starting Capital ($)</label>
            <input
              type="number"
              className="select-input font-mono"
              value={initialCapital}
              onChange={(e) => setInitialCapital(Math.max(100, Number(e.target.value)))}
              style={{ width: '130px', padding: '6px 10px' }}
            />
          </div>

          {/* Transaction Cost */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <label className="stat-label">Slippage &amp; Fees (bps)</label>
            <input
              type="number"
              className="select-input font-mono"
              value={feeBps}
              onChange={(e) => setFeeBps(Math.max(0, Number(e.target.value)))}
              style={{ width: '100px', padding: '6px 10px' }}
            />
          </div>
        </div>
      </section>

      {/* Backtest Scorecard */}
      {metrics && (
        <section className="research-section">
          <div className="section-title-row">
            <h3 className="section-heading">Performance Scorecard: {ticker}</h3>
            <span className="section-pill font-mono">
              Initial: ${initialCapital.toLocaleString()} &rarr; Final: ${backtestData.finalStrategyEquity.toLocaleString()}
            </span>
          </div>

          <div className="metrics-grid">
            <div className="metric-cell">
              <span className="metric-label">Strategy Total Return</span>
              <span className={`metric-value font-mono ${metrics.strategyReturnPct >= 0 ? 'text-up' : 'text-down'}`}>
                {metrics.strategyReturnPct >= 0 ? '+' : ''}{metrics.strategyReturnPct.toFixed(2)}%
              </span>
              <span className={`metric-sublabel ${metrics.alphaPct >= 0 ? 'text-up' : 'text-down'}`}>
                {metrics.alphaPct >= 0 ? '+' : ''}{metrics.alphaPct.toFixed(2)}% Alpha vs Benchmark
              </span>
              <p className="metric-plain-desc">
                Final equity: ${backtestData.finalStrategyEquity.toLocaleString()}
              </p>
            </div>

            <div className="metric-cell">
              <span className="metric-label">Buy &amp; Hold Benchmark</span>
              <span className={`metric-value font-mono ${metrics.benchmarkReturnPct >= 0 ? 'text-up' : 'text-down'}`}>
                {metrics.benchmarkReturnPct >= 0 ? '+' : ''}{metrics.benchmarkReturnPct.toFixed(2)}%
              </span>
              <span className="metric-sublabel text-muted">Passive Asset Holding</span>
              <p className="metric-plain-desc">
                Final equity: ${backtestData.finalBenchmarkEquity.toLocaleString()}
              </p>
            </div>

            <div className="metric-cell">
              <span className="metric-label">Strategy Sharpe Ratio</span>
              <span className="metric-value font-mono">
                {metrics.sharpeRatio.toFixed(2)}
              </span>
              <span className="metric-sublabel text-muted">
                Benchmark: {metrics.benchmarkSharpeRatio.toFixed(2)} (Rf = 2.0%)
              </span>
              <p className="metric-plain-desc">
                Annualized risk-adjusted excess return per unit of volatility.
              </p>
            </div>

            <div className="metric-cell">
              <span className="metric-label">Max Drawdown</span>
              <span className="metric-value font-mono text-amber">
                -{metrics.maxDrawdownPct.toFixed(2)}%
              </span>
              <span className="metric-sublabel text-muted">
                Benchmark MDD: -{metrics.benchmarkMaxDrawdownPct.toFixed(2)}%
              </span>
              <p className="metric-plain-desc">
                Peak-to-trough capital decline during holding periods.
              </p>
            </div>

            <div className="metric-cell">
              <span className="metric-label">Directional Accuracy</span>
              <span className="metric-value font-mono">
                {metrics.directionalAccuracyPct.toFixed(2)}%
              </span>
              <span className="metric-sublabel text-muted">
                F1 Score: {metrics.f1Score}
              </span>
              <p className="metric-plain-desc">
                Win Rate: {metrics.winRatePct.toFixed(1)}% | Profit Factor: {metrics.profitFactor}
              </p>
            </div>

            <div className="metric-cell">
              <span className="metric-label">Confusion Matrix (Holdout)</span>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px', margin: '6px 0', fontSize: '11px' }} className="font-mono">
                <div style={{ background: 'var(--bg-surface)', padding: '3px 6px', borderRadius: '4px', borderLeft: '2px solid var(--color-up)' }}>
                  <span className="text-muted">TP: </span><span style={{ fontWeight: 600 }}>{metrics.tp ?? '—'}</span>
                </div>
                <div style={{ background: 'var(--bg-surface)', padding: '3px 6px', borderRadius: '4px', borderLeft: '2px solid var(--color-down)' }}>
                  <span className="text-muted">FP: </span><span style={{ fontWeight: 600 }}>{metrics.fp ?? '—'}</span>
                </div>
                <div style={{ background: 'var(--bg-surface)', padding: '3px 6px', borderRadius: '4px', borderLeft: '2px solid var(--color-down)' }}>
                  <span className="text-muted">FN: </span><span style={{ fontWeight: 600 }}>{metrics.fn ?? '—'}</span>
                </div>
                <div style={{ background: 'var(--bg-surface)', padding: '3px 6px', borderRadius: '4px', borderLeft: '2px solid var(--color-up)' }}>
                  <span className="text-muted">TN: </span><span style={{ fontWeight: 600 }}>{metrics.tn ?? '—'}</span>
                </div>
              </div>
              <span className="metric-sublabel text-muted">
                Prec: {metrics.precisionPct !== undefined ? metrics.precisionPct.toFixed(1) : '—'}% | Rec: {metrics.recallPct !== undefined ? metrics.recallPct.toFixed(1) : '—'}%
              </span>
              <p className="metric-plain-desc">
                True vs false signals over backtest period.
              </p>
            </div>
          </div>
        </section>
      )}

      {/* Interactive Equity Curve Chart */}
      {chartMath && (
        <section className="research-section">
          <div className="section-title-row">
            <h3 className="section-heading">Cumulative Equity Curve ($)</h3>
            <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', fontSize: '12px' }}>
                <span style={{ width: '12px', height: '3px', backgroundColor: '#10b981', display: 'inline-block' }} />
                <span>LSTM Strategy</span>
              </span>
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: '#94a3b8' }}>
                <span style={{ width: '12px', height: '3px', backgroundColor: '#64748b', display: 'inline-block' }} />
                <span>Buy &amp; Hold Benchmark</span>
              </span>
            </div>
          </div>

          <div className="chart-canvas-wrapper" style={{ padding: '16px 8px' }}>
            <svg
              viewBox={`0 0 ${chartMath.width} ${chartMath.height}`}
              className="stock-svg-chart"
              style={{ width: '100%', height: 'auto', maxHeight: '380px' }}
            >
              {/* Grid Lines */}
              {chartMath.yTicks.map((val, idx) => (
                <g key={idx}>
                  <line
                    x1={chartMath.pad.left}
                    y1={chartMath.yScale(val)}
                    x2={chartMath.width - chartMath.pad.right}
                    y2={chartMath.yScale(val)}
                    stroke="var(--border-subtle)"
                    strokeDasharray="4 4"
                    strokeWidth="1"
                  />
                  <text
                    x={chartMath.width - chartMath.pad.right + 8}
                    y={chartMath.yScale(val) + 4}
                    fill="var(--text-muted)"
                    fontSize="11"
                    fontFamily="var(--font-mono)"
                  >
                    ${val >= 1000 ? `${(val / 1000).toFixed(1)}k` : val.toFixed(0)}
                  </text>
                </g>
              ))}

              {/* X Axis Date Labels */}
              {chartMath.xTickIndices.map((i) => {
                const item = equityCurve[i];
                if (!item) return null;
                return (
                  <text
                    key={i}
                    x={chartMath.xScale(i)}
                    y={chartMath.height - 12}
                    fill="var(--text-muted)"
                    fontSize="11"
                    fontFamily="var(--font-mono)"
                    textAnchor="middle"
                  >
                    {item.date}
                  </text>
                );
              })}

              {/* Benchmark Path */}
              <path
                d={chartMath.benchPath}
                fill="none"
                stroke="#64748b"
                strokeWidth="2"
                strokeDasharray="3 3"
              />

              {/* Strategy Path */}
              <path
                d={chartMath.stratPath}
                fill="none"
                stroke="#10b981"
                strokeWidth="2.5"
              />
            </svg>
          </div>
        </section>
      )}

      {/* Walk-Forward Market Regimes Breakdown */}
      <section className="research-section">
        <div className="section-title-row" style={{ flexWrap: 'wrap', gap: '10px' }}>
          <div>
            <h3 className="section-heading">Walk-Forward Validation &amp; Regime Stability</h3>
            <span className="section-pill font-mono">
              {wfMode === 'expanding' ? 'Chronological Expanding Folds (Independent Models & Scalers)' : 'Chronological Regimes (2015 – 2026)'}
            </span>
          </div>

          <div className="toggle-pill-group">
            <button
              type="button"
              className={`toggle-pill ${wfMode === 'expanding' ? 'active' : ''}`}
              onClick={() => setWfMode('expanding')}
            >
              Expanding Folds (Separate Models)
            </button>
            <button
              type="button"
              className={`toggle-pill ${wfMode === 'regimes' ? 'active' : ''}`}
              onClick={() => setWfMode('regimes')}
            >
              5 Historical Regimes
            </button>
          </div>
        </div>

        {wfMode === 'expanding' ? (
          <>
            <p className="callout-text" style={{ marginBottom: '16px' }}>
              Genuine expanding-window walk-forward validation: each fold fits a <strong>new StandardScaler strictly on past training data</strong>, trains a <strong>fresh StockLSTM model from scratch</strong>, monitors early stopping on a separate validation window, and evaluates strictly out-of-sample on future untouched trading sessions.
            </p>

            <div className="table-scroll-wrapper">
              <table className="financial-table">
                <thead>
                  <tr>
                    <th className="text-left">Fold &amp; Regime Context</th>
                    <th className="text-left">Train / OOS Period</th>
                    <th className="text-right">OOS Sequences</th>
                    <th className="text-right">Accuracy</th>
                    <th className="text-right">Baseline</th>
                    <th className="text-right">Delta</th>
                    <th className="text-right">F1 Score</th>
                    <th className="text-center">Confusion Matrix (TN, FP, FN, TP)</th>
                    <th className="text-right">Fold Model Checkpoint</th>
                  </tr>
                </thead>
                <tbody>
                  {expandingFolds.map((fold, idx) => (
                    <tr key={idx} className="table-row">
                      <td className="text-left">
                        <strong>{fold.name}</strong>
                        <div className="font-mono text-muted" style={{ fontSize: '11px', marginTop: '2px' }}>
                          {fold.description}
                        </div>
                      </td>
                      <td className="text-left font-mono" style={{ fontSize: '11px' }}>
                        <div><span className="text-muted">Train:</span> {fold.train_period}</div>
                        <div><span className="text-muted">Test:</span> <strong>{fold.test_period}</strong></div>
                      </td>
                      <td className="text-right font-mono">{fold.test_sequences.toLocaleString()}</td>
                      <td className="text-right font-mono">{(fold.test_accuracy * 100).toFixed(2)}%</td>
                      <td className="text-right font-mono">{(fold.baseline_accuracy * 100).toFixed(2)}%</td>
                      <td className={`text-right font-mono ${fold.delta_accuracy >= 0 ? 'text-up' : 'text-down'}`}>
                        {fold.delta_accuracy >= 0 ? '+' : ''}{(fold.delta_accuracy * 100).toFixed(2)}%
                      </td>
                      <td className="text-right font-mono">{fold.test_f1.toFixed(4)}</td>
                      <td className="text-center font-mono" style={{ fontSize: '11px' }}>
                        [{fold.tn.toLocaleString()}, {fold.fp.toLocaleString()}] / [{fold.fn.toLocaleString()}, {fold.tp.toLocaleString()}]
                      </td>
                      <td className="text-right">
                        <code className="section-pill" style={{ fontSize: '10px' }}>{fold.checkpoint_path}</code>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        ) : (
          <>
            <p className="callout-text" style={{ marginBottom: '16px' }}>
              Multi-regime stress test: evaluates the production model across 5 chronological economic regimes covering over 1.27 million trading sequences.
            </p>

            <div className="table-scroll-wrapper">
              <table className="financial-table">
                <thead>
                  <tr>
                    <th className="text-left">Market Regime</th>
                    <th className="text-left">Period</th>
                    <th className="text-right">Sequences</th>
                    <th className="text-right">Accuracy</th>
                    <th className="text-right">Baseline</th>
                    <th className="text-right">Delta</th>
                    <th className="text-right">F1 Score</th>
                    <th className="text-right">Loss</th>
                    <th className="text-center">Partition</th>
                  </tr>
                </thead>
                <tbody>
                  {walkForwardRegimes.map((regime, idx) => (
                    <tr key={idx} className="table-row">
                      <td className="text-left">
                        <strong>{regime.name}</strong>
                        <div className="font-mono text-muted" style={{ fontSize: '11px', marginTop: '2px' }}>
                          {regime.description}
                        </div>
                      </td>
                      <td className="text-left font-mono">
                        {regime.start_date} &rarr; {regime.end_date}
                      </td>
                      <td className="text-right font-mono">{regime.total_sequences.toLocaleString()}</td>
                      <td className="text-right font-mono">{(regime.accuracy * 100).toFixed(2)}%</td>
                      <td className="text-right font-mono">{(regime.baseline_accuracy * 100).toFixed(2)}%</td>
                      <td className={`text-right font-mono ${regime.delta_accuracy >= 0 ? 'text-up' : 'text-down'}`}>
                        {regime.delta_accuracy >= 0 ? '+' : ''}{(regime.delta_accuracy * 100).toFixed(2)}%
                      </td>
                      <td className="text-right font-mono">{regime.f1.toFixed(4)}</td>
                      <td className="text-right font-mono">{regime.loss.toFixed(4)}</td>
                      <td className="text-center">
                        <span className={`section-pill ${regime.is_out_of_sample ? 'highlight' : ''}`} style={{ fontSize: '10px' }}>
                          {regime.is_out_of_sample ? 'Holdout Test' : 'In-Sample / Val'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </section>
    </div>
  );
}
