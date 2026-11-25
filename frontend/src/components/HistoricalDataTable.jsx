import React, { useState } from 'react';

export default function HistoricalDataTable({ ticker, stockData }) {
  const [viewMode, setViewMode] = useState('market'); // 'market' | 'features'
  const [rowsPerPage, setRowsPerPage] = useState(15);
  const [page, setPage] = useState(0);

  if (!stockData || !stockData.history) return null;

  // Most recent first for inspection
  const reversedHistory = [...stockData.history].reverse();
  const totalRows = reversedHistory.length;
  const totalPages = Math.max(1, Math.ceil(totalRows / rowsPerPage));
  const displayedRows = reversedHistory.slice(page * rowsPerPage, (page + 1) * rowsPerPage);

  const formatNumber = (num, decimals = 2) => {
    if (num === null || num === undefined || isNaN(num)) return '—';
    return Number(num).toLocaleString('en-US', {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    });
  };

  const formatVolume = (vol) => {
    if (!vol && vol !== 0) return '—';
    if (vol >= 1e9) return (vol / 1e9).toFixed(2) + 'B';
    if (vol >= 1e6) return (vol / 1e6).toFixed(2) + 'M';
    if (vol >= 1e3) return (vol / 1e3).toFixed(1) + 'K';
    return vol.toLocaleString();
  };

  return (
    <div className="historical-table-container">
      <div className="table-header-bar">
        <div className="table-title-group">
          <h3 className="table-title">Historical Trading Sessions &amp; Feature Matrix</h3>
          <span className="table-subtitle">
            Verified continuous records for {ticker} ({totalRows} total sessions, Parquet + Yahoo Finance live spliced)
          </span>
        </div>

        <div className="table-controls">
          <div className="toggle-pill-group">
            <button
              type="button"
              className={`toggle-pill ${viewMode === 'market' ? 'active' : ''}`}
              onClick={() => setViewMode('market')}
            >
              Market OHLCV
            </button>
            <button
              type="button"
              className={`toggle-pill ${viewMode === 'features' ? 'active' : ''}`}
              onClick={() => setViewMode('features')}
            >
              8 Engineered Model Features
            </button>
          </div>

          <div className="rows-selector">
            <label htmlFor="rows-select" className="rows-label">Rows:</label>
            <select
              id="rows-select"
              className="select-input"
              value={rowsPerPage}
              onChange={(e) => {
                setRowsPerPage(Number(e.target.value));
                setPage(0);
              }}
            >
              <option value={10}>10</option>
              <option value={15}>15</option>
              <option value={30}>30</option>
              <option value={50}>50</option>
            </select>
          </div>
        </div>
      </div>

      <div className="table-scroll-wrapper">
        <table className="financial-table">
          <thead>
            {viewMode === 'market' ? (
              <tr>
                <th className="text-left">Date</th>
                <th className="text-right">Close</th>
                <th className="text-right">Daily Return</th>
                <th className="text-right">Open</th>
                <th className="text-right">High</th>
                <th className="text-right">Low</th>
                <th className="text-right">Volume</th>
                <th className="text-center">Next Target</th>
              </tr>
            ) : (
              <tr>
                <th className="text-left">Date</th>
                <th className="text-right">Close</th>
                <th className="text-right">Daily Return</th>
                <th className="text-right">Vol Change</th>
                <th className="text-right">H-L Range</th>
                <th className="text-right">20D Vol</th>
                <th className="text-right">SMA20</th>
                <th className="text-right">SMA50</th>
                <th className="text-right">Close/SMA20</th>
                <th className="text-right">RSI (14D)</th>
                <th className="text-center">Target (t+1)</th>
              </tr>
            )}
          </thead>
          <tbody>
            {displayedRows.map((row, idx) => {
              const dateVal = row.date || row.Date;
              const closeVal = row.close ?? row.Close;
              const openVal = row.open ?? row.Open;
              const highVal = row.high ?? row.High;
              const lowVal = row.low ?? row.Low;
              const volumeVal = row.volume ?? row.Volume;
              const returnVal = row.dailyReturn ?? (row.Daily_Return !== undefined ? row.Daily_Return * 100 : null);
              const isPositive = returnVal >= 0;
              const targetVal = row.target ?? row.Target;

              const volChangeVal = row.volumeChange ?? (row.Volume_Change !== undefined ? row.Volume_Change * 100 : null);
              const hlRangeVal = row.highLowRange ?? (row.High_Low_Range !== undefined ? row.High_Low_Range * 100 : null);
              const rollVolVal = row.rollingVol20 ?? (row.Rolling_Volatility_20 !== undefined ? row.Rolling_Volatility_20 * 100 : null);
              const sma20Val = row.sma20 ?? row.SMA20;
              const sma50Val = row.sma50 ?? row.SMA50;
              const closeSma20Val = row.closeSma20 ?? row.Close_to_SMA20;

              return (
                <tr key={dateVal || idx} className="table-row">
                  <td className="text-left font-mono font-medium">{dateVal}</td>
                  <td className="text-right font-mono font-bold">${formatNumber(closeVal)}</td>

                  {viewMode === 'market' ? (
                    <>
                      <td className={`text-right font-mono ${isPositive ? 'text-up' : 'text-down'}`}>
                        {returnVal !== null && returnVal !== undefined ? (
                          `${isPositive ? '+' : ''}${Number(returnVal).toFixed(2)}%`
                        ) : '—'}
                      </td>
                      <td className="text-right font-mono text-muted">${formatNumber(openVal)}</td>
                      <td className="text-right font-mono text-muted">${formatNumber(highVal)}</td>
                      <td className="text-right font-mono text-muted">${formatNumber(lowVal)}</td>
                      <td className="text-right font-mono text-muted">{formatVolume(volumeVal)}</td>
                      <td className="text-center">
                        {targetVal !== null && targetVal !== undefined ? (
                          <span className={`target-badge ${targetVal === 1 ? 'target-up' : 'target-down'}`}>
                            {targetVal === 1 ? '▲ UP' : '▼ DOWN'}
                          </span>
                        ) : (
                          <span className="text-muted font-mono">—</span>
                        )}
                      </td>
                    </>
                  ) : (
                    <>
                      <td className={`text-right font-mono ${isPositive ? 'text-up' : 'text-down'}`}>
                        {returnVal !== null && returnVal !== undefined ? (
                          `${isPositive ? '+' : ''}${Number(returnVal).toFixed(2)}%`
                        ) : '—'}
                      </td>
                      <td className="text-right font-mono text-muted">
                        {volChangeVal !== null && volChangeVal !== undefined ? (
                          `${volChangeVal >= 0 ? '+' : ''}${Number(volChangeVal).toFixed(1)}%`
                        ) : '—'}
                      </td>
                      <td className="text-right font-mono text-muted">
                        {hlRangeVal !== null && hlRangeVal !== undefined ? (
                          `${Number(hlRangeVal).toFixed(2)}%`
                        ) : '—'}
                      </td>
                      <td className="text-right font-mono text-muted">
                        {rollVolVal !== null && rollVolVal !== undefined ? (
                          `${Number(rollVolVal).toFixed(2)}%`
                        ) : '—'}
                      </td>
                      <td className="text-right font-mono text-muted">${formatNumber(sma20Val)}</td>
                      <td className="text-right font-mono text-muted">${formatNumber(sma50Val)}</td>
                      <td className="text-right font-mono text-muted">
                        {closeSma20Val !== null && closeSma20Val !== undefined ? (
                          Number(closeSma20Val).toFixed(3)
                        ) : '—'}
                      </td>
                      <td className="text-right font-mono text-ai" style={{ fontWeight: 600 }}>
                        {row.rsi14 !== undefined && row.rsi14 !== null ? (
                          Number(row.rsi14).toFixed(1)
                        ) : '—'}
                      </td>
                      <td className="text-center">
                        {targetVal !== null && targetVal !== undefined ? (
                          <span className={`target-badge ${targetVal === 1 ? 'target-up' : 'target-down'}`}>
                            {targetVal === 1 ? '▲ 1 (UP)' : '▼ 0 (DOWN)'}
                          </span>
                        ) : (
                          <span className="text-muted font-mono">—</span>
                        )}
                      </td>
                    </>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="table-footer-pagination">
        <span className="pagination-info">
          Page {page + 1} of {totalPages} ({displayedRows.length} shown of {totalRows} records)
        </span>
        <div className="pagination-buttons">
          <button
            type="button"
            className="pagination-btn"
            disabled={page === 0}
            onClick={() => setPage((p) => Math.max(0, p - 1))}
          >
            ← Previous
          </button>
          <button
            type="button"
            className="pagination-btn"
            disabled={page >= totalPages - 1}
            onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
          >
            Next →
          </button>
        </div>
      </div>
    </div>
  );
}
