import React, { useState, useMemo } from 'react';

export default function StockChart({ ticker, stockData }) {
  const [selectedRange, setSelectedRange] = useState('1Y'); // '1M' | '3M' | '6M' | '1Y' | '3Y' | 'ALL'
  const [showSma20, setShowSma20] = useState(true);
  const [showSma50, setShowSma50] = useState(true);
  const [showVolume, setShowVolume] = useState(true);
  const [hoverIndex, setHoverIndex] = useState(null);

  const { history = [], allTimeHistory = [] } = stockData || {};

  // Filter dataset by range
  const displayData = useMemo(() => {
    if (selectedRange === 'ALL') {
      return allTimeHistory.length ? allTimeHistory : history;
    }
    const countMap = {
      '1M': 21,
      '3M': 63,
      '6M': 126,
      '1Y': 252,
      '3Y': 756,
    };
    const count = countMap[selectedRange] || 252;
    return history.slice(-count);
  }, [selectedRange, history, allTimeHistory]);

  const isAllTime = selectedRange === 'ALL';

  // SVG Chart Dimensions & Math
  const chartConfig = useMemo(() => {
    if (!displayData.length) return null;

    const width = 1100;
    const height = 440;
    const pad = { top: 25, right: 65, bottom: 40, left: 15 };
    const volumeHeight = showVolume ? 60 : 0;
    const priceHeight = height - pad.top - pad.bottom - volumeHeight - (showVolume ? 15 : 0);
    const innerW = width - pad.left - pad.right;

    // Price scaling
    const prices = displayData.map((d) => d.close);
    const smas20 = !isAllTime && showSma20 ? displayData.map((d) => d.sma20).filter((v) => v !== undefined && v !== null) : [];
    const smas50 = !isAllTime && showSma50 ? displayData.map((d) => d.sma50).filter((v) => v !== undefined && v !== null) : [];
    const allVals = [...prices, ...smas20, ...smas50];

    const minP = Math.min(...allVals) * 0.985;
    const maxP = Math.max(...allVals) * 1.015;

    // Volume scaling
    const volumes = displayData.map((d) => d.volume || 0);
    const maxVol = Math.max(...volumes, 1);

    const xScale = (idx) => pad.left + (idx / Math.max(1, displayData.length - 1)) * innerW;
    const yScale = (val) => pad.top + priceHeight - ((val - minP) / Math.max(0.001, maxP - minP)) * priceHeight;
    const yVolScale = (vol) => {
      const volBase = pad.top + priceHeight + 15;
      return volBase + volumeHeight - (vol / maxVol) * volumeHeight;
    };

    // Paths
    const buildPath = (accessor) => {
      const pts = displayData
        .map((d, i) => {
          const v = accessor(d);
          if (v === null || v === undefined) return null;
          return `${xScale(i).toFixed(1)},${yScale(v).toFixed(1)}`;
        })
        .filter(Boolean);
      return pts.length ? `M ${pts.join(' L ')}` : '';
    };

    const closeLine = buildPath((d) => d.close);
    const closeArea = closeLine
      ? `${closeLine} L ${xScale(displayData.length - 1).toFixed(1)},${(pad.top + priceHeight).toFixed(1)} L ${pad.left.toFixed(1)},${(pad.top + priceHeight).toFixed(1)} Z`
      : '';

    const sma20Line = !isAllTime && showSma20 ? buildPath((d) => d.sma20) : '';
    const sma50Line = !isAllTime && showSma50 ? buildPath((d) => d.sma50) : '';

    // Horizontal Price Reference Ticks (5 levels)
    const yTicks = [0, 0.25, 0.5, 0.75, 1].map((pct) => minP + pct * (maxP - minP));

    // X-axis date labels (5 distributed points)
    const xTickIndices = [
      0,
      Math.floor(displayData.length * 0.25),
      Math.floor(displayData.length * 0.5),
      Math.floor(displayData.length * 0.75),
      displayData.length - 1,
    ];

    return {
      width,
      height,
      pad,
      innerW,
      priceHeight,
      volumeHeight,
      minP,
      maxP,
      xScale,
      yScale,
      yVolScale,
      closeLine,
      closeArea,
      sma20Line,
      sma50Line,
      yTicks,
      xTickIndices,
    };
  }, [displayData, showSma20, showSma50, showVolume, isAllTime]);

  // Active record under mouse (defaults to latest session)
  const activeRecord = hoverIndex !== null && displayData[hoverIndex]
    ? displayData[hoverIndex]
    : displayData[displayData.length - 1] || {};

  return (
    <div className="stock-chart-card">
      {/* Top Controls Bar */}
      <div className="chart-controls-bar">
        {/* Timeframe Selector Pills */}
        <div className="timeframe-group">
          {['1M', '3M', '6M', '1Y', '3Y', 'ALL'].map((r) => (
            <button
              key={r}
              type="button"
              className={`tf-btn ${selectedRange === r ? 'active' : ''}`}
              onClick={() => {
                setSelectedRange(r);
                setHoverIndex(null);
              }}
            >
              {r}
            </button>
          ))}
        </div>

        {/* Technical Indicator Toggles */}
        <div className="chart-indicators-group">
          {!isAllTime && (
            <>
              <button
                type="button"
                className={`indicator-toggle ${showSma20 ? 'active' : ''}`}
                onClick={() => setShowSma20(!showSma20)}
              >
                <span className="indicator-legend-line sma20" />
                <span>SMA 20</span>
              </button>
              <button
                type="button"
                className={`indicator-toggle ${showSma50 ? 'active' : ''}`}
                onClick={() => setShowSma50(!showSma50)}
              >
                <span className="indicator-legend-line sma50" />
                <span>SMA 50</span>
              </button>
            </>
          )}
          <button
            type="button"
            className={`indicator-toggle ${showVolume ? 'active' : ''}`}
            onClick={() => setShowVolume(!showVolume)}
          >
            <span>Volume</span>
          </button>
        </div>
      </div>

      {/* Dynamic Hover HUD Bar */}
      <div className="chart-hud-row">
        <span className="hud-date font-mono">{activeRecord.date || '—'}</span>
        <div className="hud-item">
          <span className="hud-label">CLOSE:</span>
          <span className="hud-val font-mono">
            {activeRecord.close ? `$${activeRecord.close.toFixed(2)}` : '—'}
          </span>
        </div>
        {!isAllTime && (
          <>
            <div className="hud-item">
              <span className="hud-label" style={{ color: 'var(--color-sma20)' }}>SMA 20:</span>
              <span className="hud-val font-mono">
                {activeRecord.sma20 ? `$${activeRecord.sma20.toFixed(2)}` : '—'}
              </span>
            </div>
            <div className="hud-item">
              <span className="hud-label" style={{ color: 'var(--color-sma50)' }}>SMA 50:</span>
              <span className="hud-val font-mono">
                {activeRecord.sma50 ? `$${activeRecord.sma50.toFixed(2)}` : '—'}
              </span>
            </div>
          </>
        )}
        <div className="hud-item">
          <span className="hud-label">RETURN:</span>
          <span className={`hud-val font-mono ${activeRecord.dailyReturn >= 0 ? 'text-up' : 'text-down'}`}>
            {activeRecord.dailyReturn !== undefined && activeRecord.dailyReturn !== null
              ? `${activeRecord.dailyReturn >= 0 ? '+' : ''}${activeRecord.dailyReturn.toFixed(2)}%`
              : '—'}
          </span>
        </div>
        <div className="hud-item">
          <span className="hud-label">VOL:</span>
          <span className="hud-val font-mono">
            {activeRecord.volume ? activeRecord.volume.toLocaleString() : '—'}
          </span>
        </div>
        {activeRecord.rsi14 !== undefined && activeRecord.rsi14 !== null && (
          <div className="hud-item">
            <span className="hud-label" style={{ color: 'var(--accent-ai)' }}>RSI(14):</span>
            <span className="hud-val font-mono" style={{ color: 'var(--accent-ai)', fontWeight: 600 }}>
              {Number(activeRecord.rsi14).toFixed(1)}
            </span>
          </div>
        )}
        {!isAllTime && activeRecord.target !== undefined && activeRecord.target !== null && (
          <div className="hud-item">
            <span className="hud-label">TARGET (t+1):</span>
            <span className={`target-badge ${activeRecord.target === 1 ? 'target-up' : 'target-down'}`}>
              {activeRecord.target === 1 ? '▲ UP' : '▼ DOWN'}
            </span>
          </div>
        )}
      </div>

      {/* Main SVG Financial Chart */}
      {chartConfig && (
        <div className="svg-chart-container">
          <svg
            viewBox={`0 0 ${chartConfig.width} ${chartConfig.height}`}
            className="financial-svg"
            onMouseLeave={() => setHoverIndex(null)}
            onMouseMove={(e) => {
              const rect = e.currentTarget.getBoundingClientRect();
              const xPos = e.clientX - rect.left;
              const relativeX = (xPos / rect.width) * chartConfig.width - chartConfig.pad.left;
              const fraction = Math.max(0, Math.min(1, relativeX / chartConfig.innerW));
              const idx = Math.round(fraction * (displayData.length - 1));
              setHoverIndex(idx);
            }}
          >
            <defs>
              <linearGradient id="chartGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#10b981" stopOpacity="0.18" />
                <stop offset="100%" stopColor="#10b981" stopOpacity="0.0" />
              </linearGradient>
            </defs>

            {/* Background Watermark */}
            <text
              x={chartConfig.width / 2}
              y={chartConfig.height / 2}
              textAnchor="middle"
              className="watermark-text"
            >
              {ticker}
            </text>

            {/* Horizontal Gridlines & Price Ticks on Right */}
            {chartConfig.yTicks.map((val, i) => {
              const y = chartConfig.yScale(val);
              return (
                <g key={i}>
                  <line
                    x1={chartConfig.pad.left}
                    x2={chartConfig.width - chartConfig.pad.right}
                    y1={y}
                    y2={y}
                    className="chart-grid-line"
                  />
                  <text
                    x={chartConfig.width - chartConfig.pad.right + 8}
                    y={y + 4}
                    textAnchor="start"
                    className="axis-text"
                  >
                    ${val.toFixed(2)}
                  </text>
                </g>
              );
            })}

            {/* Volume Histogram Bars */}
            {showVolume &&
              displayData.map((d, i) => {
                const volY = chartConfig.yVolScale(d.volume || 0);
                const barH = chartConfig.pad.top + chartConfig.priceHeight + 15 + chartConfig.volumeHeight - volY;
                const barW = Math.max(1, (chartConfig.innerW / displayData.length) * 0.75);
                const isUp = d.dailyReturn !== undefined ? d.dailyReturn >= 0 : true;
                return (
                  <rect
                    key={i}
                    x={chartConfig.xScale(i) - barW / 2}
                    y={volY}
                    width={barW}
                    height={Math.max(1, barH)}
                    fill={isUp ? 'rgba(16, 185, 129, 0.35)' : 'rgba(239, 68, 68, 0.35)'}
                    className="volume-bar"
                  />
                );
              })}

            {/* Price Area Fill */}
            {chartConfig.closeArea && (
              <path d={chartConfig.closeArea} fill="url(#chartGradient)" />
            )}

            {/* Technical Indicator Lines */}
            {chartConfig.sma50Line && (
              <path
                d={chartConfig.sma50Line}
                fill="none"
                className="chart-line-sma50"
                strokeWidth="1.5"
              />
            )}
            {chartConfig.sma20Line && (
              <path
                d={chartConfig.sma20Line}
                fill="none"
                className="chart-line-sma20"
                strokeWidth="1.5"
              />
            )}

            {/* Main Price Line */}
            {chartConfig.closeLine && (
              <path
                d={chartConfig.closeLine}
                fill="none"
                className="chart-line"
                strokeWidth="2.2"
              />
            )}

            {/* Interactive Hairline Crosshair */}
            {hoverIndex !== null && displayData[hoverIndex] && (
              <g>
                <line
                  x1={chartConfig.xScale(hoverIndex)}
                  x2={chartConfig.xScale(hoverIndex)}
                  y1={chartConfig.pad.top}
                  y2={chartConfig.height - chartConfig.pad.bottom}
                  className="crosshair-line"
                />
                <circle
                  cx={chartConfig.xScale(hoverIndex)}
                  cy={chartConfig.yScale(displayData[hoverIndex].close)}
                  r="5"
                  className="crosshair-point"
                />
              </g>
            )}

            {/* X-Axis Date Labels */}
            {chartConfig.xTickIndices.map((idx, i) => {
              const rec = displayData[idx];
              if (!rec) return null;
              return (
                <text
                  key={i}
                  x={chartConfig.xScale(idx)}
                  y={chartConfig.height - 10}
                  textAnchor="middle"
                  className="axis-text"
                >
                  {rec.date}
                </text>
              );
            })}
          </svg>
        </div>
      )}

      {/* Chart Footer Metadata & Spliced Attribution */}
      <div className="chart-footer-meta font-mono">
        <div className="meta-left">
          <span className="spliced-data-tag">Parquet (2000–Feb 2026) + Yahoo Finance Live Spliced</span>
          <span>Zero duplicate dates</span>
        </div>
        <div className="meta-right">
          <span>{displayData.length} continuous sessions</span>
          <span>Wilder&apos;s RSI(14)</span>
        </div>
      </div>
    </div>
  );
}
