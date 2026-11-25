import React, { useState, useEffect } from 'react';
import projectData from './data/projectData.json';
import ErrorBoundary from './components/ErrorBoundary';
import Header from './components/Header';
import StockSummary from './components/StockSummary';
import PredictionCard from './components/PredictionCard';
import StockChart from './components/StockChart';
import HistoricalDataTable from './components/HistoricalDataTable';
import LandingPage from './components/LandingPage';
import MarketOverviewView from './components/MarketOverviewView';
import StockDiscoveryView from './components/StockDiscoveryView';
import WatchlistView from './components/WatchlistView';
import AiAssistantView from './components/AiAssistantView';
import ResearchView from './components/ResearchView';
import './App.css';

const staticTickerSamples = projectData?.tickerSamples || {};
const staticPredictions = projectData?.predictions || {};

export default function App() {
  const [activeView, setActiveView] = useState('landing'); // Default to new Landing Page
  const [selectedTicker, setSelectedTicker] = useState('AAPL');
  const [isPredictionOpen, setIsPredictionOpen] = useState(false);
  const [dynamicStocks, setDynamicStocks] = useState({});
  const [loadingDynamic, setLoadingDynamic] = useState(false);
  const [allTickers, setAllTickers] = useState([]);
  const [marketStream, setMarketStream] = useState(null);

  // Phase 2: Server-Sent Events (SSE) connection for live market ticks & index quotes
  useEffect(() => {
    let evtSource = null;
    try {
      evtSource = new EventSource('/api/stream/market');
      evtSource.addEventListener('market_update', (e) => {
        try {
          const payload = JSON.parse(e.data);
          setMarketStream(payload);
        } catch (err) {
          console.warn('Failed to parse SSE market update:', err);
        }
      });
      evtSource.onerror = (err) => {
        console.debug('SSE market stream reconnecting...', err);
      };
    } catch (e) {
      console.warn('Could not initialize EventSource for /api/stream/market:', e);
    }

    return () => {
      if (evtSource) {
        evtSource.close();
      }
    };
  }, []);

  // Watchlist stored in localStorage
  const [watchlist, setWatchlist] = useState(() => {
    try {
      const saved = localStorage.getItem('stocksense_watchlist');
      if (saved) return JSON.parse(saved);
    } catch (e) {
      console.warn('Could not read watchlist from localStorage:', e);
    }
    return ['AAPL', 'NVDA', 'MSFT', 'AMZN'];
  });

  // Persist watchlist changes
  useEffect(() => {
    try {
      localStorage.setItem('stocksense_watchlist', JSON.stringify(watchlist));
    } catch (e) {
      console.warn('Could not save watchlist to localStorage:', e);
    }
  }, [watchlist]);

  const addToWatchlist = (symbol) => {
    if (!watchlist.includes(symbol)) {
      setWatchlist((prev) => [...prev, symbol]);
    }
  };

  const removeFromWatchlist = (symbol) => {
    setWatchlist((prev) => prev.filter((s) => s !== symbol));
  };

  const toggleWatchlist = (symbol) => {
    if (watchlist.includes(symbol)) {
      removeFromWatchlist(symbol);
    } else {
      addToWatchlist(symbol);
    }
  };

  // Fetch full 472 S&P 500 universe for Market, Discovery, and Watchlist
  useEffect(() => {
    fetch('/api/tickers')
      .then((res) => (res.ok ? res.json() : []))
      .then((data) => {
        if (Array.isArray(data) && data.length > 0) {
          setAllTickers(data);
        }
      })
      .catch((err) => console.warn('Could not fetch all tickers:', err));
  }, []);

  const tickerSamples = staticTickerSamples;
  const predictionData = staticPredictions[selectedTicker] || null;

  // Resolve stockData (real backend data first, graceful fallback to sample while loading)
  const stockData = dynamicStocks[selectedTicker] || tickerSamples[selectedTicker] || null;

  // Dynamically fetch historical data from FastAPI backend for selected ticker
  useEffect(() => {
    if (dynamicStocks[selectedTicker]) return;

    if (!tickerSamples[selectedTicker]) {
      setLoadingDynamic(true);
    }

    fetch(`/api/historical/${encodeURIComponent(selectedTicker)}?days=756`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        setDynamicStocks((prev) => ({ ...prev, [selectedTicker]: data }));
        setLoadingDynamic(false);
      })
      .catch((err) => {
        console.warn(`Could not dynamically fetch stock data for ${selectedTicker}:`, err);
        setLoadingDynamic(false);
      });
  }, [selectedTicker, dynamicStocks, tickerSamples]);

  const handleTickerChange = (ticker) => {
    setSelectedTicker(ticker);
    setIsPredictionOpen(false);
  };

  // If user requests landing page, render LandingPage directly
  if (activeView === 'landing') {
    return (
      <ErrorBoundary>
        <LandingPage
          onNavigate={setActiveView}
          onSelectTicker={handleTickerChange}
        />
      </ErrorBoundary>
    );
  }

  return (
    <ErrorBoundary>
      <div className="app-shell">
        <Header
          activeView={activeView}
          setActiveView={setActiveView}
          selectedTicker={selectedTicker}
          setSelectedTicker={handleTickerChange}
          tickerSamples={tickerSamples}
          watchlistCount={watchlist.length}
          marketStream={marketStream}
        />

        <main className="main-viewport">
          {/* Section A: Market Overview */}
          {activeView === 'market' && (
            <MarketOverviewView
              tickers={allTickers}
              onSelectTicker={handleTickerChange}
              onNavigate={setActiveView}
              marketStream={marketStream}
            />
          )}

          {/* Section B: Stock Discovery & Screener */}
          {activeView === 'discovery' && (
            <StockDiscoveryView
              tickers={allTickers}
              onSelectTicker={handleTickerChange}
              onNavigate={setActiveView}
            />
          )}

          {/* Section C: Stock Analysis & Existing LSTM (Default / Primary Workspace) */}
          {(activeView === 'analysis' || activeView === 'dashboard') && (
            <div className="explorer-view">
              {loadingDynamic ? (
                <div className="no-data-placeholder">
                  <p>Loading market data and indicators for &ldquo;{selectedTicker}&rdquo; via FastAPI...</p>
                </div>
              ) : stockData ? (
                <>
                  {/* Selected Stock Overview & Primary "Predict Next Trading Day" Action */}
                  <StockSummary
                    ticker={selectedTicker}
                    stockData={stockData}
                    predictionData={predictionData}
                    onPredictClick={() => setIsPredictionOpen((prev) => !prev)}
                    predictionActive={isPredictionOpen}
                    isWatchlisted={watchlist.includes(selectedTicker)}
                    onToggleWatchlist={() => toggleWatchlist(selectedTicker)}
                  />

                  {/* Prediction Results Banner (Live PyTorch Inference via FastAPI /api/predict) */}
                  <PredictionCard
                    ticker={selectedTicker}
                    stockData={stockData}
                    predictionData={predictionData}
                    isOpen={isPredictionOpen}
                    onClose={() => setIsPredictionOpen(false)}
                  />

                  {/* Historical Interactive Price Chart with Timeframes & Volume */}
                  <StockChart ticker={selectedTicker} stockData={stockData} />

                  {/* Historical Trading Sessions Table */}
                  <HistoricalDataTable ticker={selectedTicker} stockData={stockData} />
                </>
              ) : (
                <div className="no-data-placeholder">
                  <p>No historical market data found for ticker &ldquo;{selectedTicker}&rdquo;.</p>
                </div>
              )}
            </div>
          )}

          {/* Section D: AI Assistant */}
          {activeView === 'assistant' && (
            <AiAssistantView
              onSelectTicker={handleTickerChange}
              onNavigate={setActiveView}
            />
          )}

          {/* Section E: Watchlist */}
          {activeView === 'watchlist' && (
            <WatchlistView
              watchlist={watchlist}
              tickers={allTickers}
              onRemoveFromWatchlist={removeFromWatchlist}
              onAddToWatchlist={addToWatchlist}
              onSelectTicker={handleTickerChange}
              onNavigate={setActiveView}
            />
          )}

          {/* Research & Backtest (Preserving ModelPerformanceView, BacktestView, HowItWorksView) */}
          {(activeView === 'research' || activeView === 'performance' || activeView === 'backtest' || activeView === 'how-it-works') && (
            <ResearchView
              projectData={projectData}
              selectedTicker={selectedTicker}
              tickerSamples={tickerSamples}
            />
          )}
        </main>

        <footer className="app-footer">
          <div className="footer-content">
            <div className="footer-left">
              <span className="footer-title">StockSense AI</span>
              <span className="footer-sep">•</span>
              <span className="footer-desc">
                PyTorch LSTM Directional Classifier • FastAPI Backend • Yahoo Finance Integration
              </span>
            </div>
            <div className="footer-right">
              <span className="footer-stat">472 S&amp;P 500 Equities</span>
              <span className="footer-sep">•</span>
              <span className="footer-stat">458,428 Test Sequences</span>
              <span className="footer-sep">•</span>
              <span className="footer-stat">51.75% Test Accuracy</span>
            </div>
          </div>
        </footer>
      </div>
    </ErrorBoundary>
  );
}
