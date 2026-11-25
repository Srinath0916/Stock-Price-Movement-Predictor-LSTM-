import React, { useState } from 'react';
import {
  BrainCircuit,
  TrendingUp,
  BookOpen,
  FlaskConical,
  ShieldCheck,
} from 'lucide-react';
import ModelPerformanceView from './ModelPerformanceView';
import BacktestView from './BacktestView';
import HowItWorksView from './HowItWorksView';

export default function ResearchView({
  projectData,
  selectedTicker,
  tickerSamples,
}) {
  const [subTab, setSubTab] = useState('performance'); // 'performance' | 'backtest' | 'methodology'

  return (
    <div className="research-view-wrapper">
      {/* 1. Header Banner */}
      <div className="view-header-strip">
        <div className="view-title-group">
          <div className="view-icon-badge neutral">
            <FlaskConical size={20} />
          </div>
          <div>
            <h1 className="view-title">Quantitative Research &amp; Model Validation</h1>
            <p className="view-subtitle">
              Inspect our PyTorch StockLSTM out-of-sample benchmarks, chronological walk-forward regimes, and simulated economic backtests.
            </p>
          </div>
        </div>

        <div className="view-actions">
          <span className="badge badge-neutral font-mono">
            458,428 Holdout Sequences Evaluated
          </span>
        </div>
      </div>

      {/* 2. Sub-Navigation Tabs */}
      <div className="research-subtabs-bar">
        <button
          type="button"
          className={`subtab-btn ${subTab === 'performance' ? 'active' : ''}`}
          onClick={() => setSubTab('performance')}
        >
          <BrainCircuit size={15} />
          <span>Model Performance &amp; Card</span>
        </button>

        <button
          type="button"
          className={`subtab-btn ${subTab === 'backtest' ? 'active' : ''}`}
          onClick={() => setSubTab('backtest')}
        >
          <TrendingUp size={15} />
          <span>Economic Backtesting &amp; PnL</span>
        </button>

        <button
          type="button"
          className={`subtab-btn ${subTab === 'methodology' ? 'active' : ''}`}
          onClick={() => setSubTab('methodology')}
        >
          <BookOpen size={15} />
          <span>System Methodology &amp; Zero Leakage</span>
        </button>
      </div>

      {/* 3. Sub-View Render */}
      <div className="research-subview-content">
        {subTab === 'performance' && (
          <ModelPerformanceView projectData={projectData} />
        )}

        {subTab === 'backtest' && (
          <BacktestView
            selectedTicker={selectedTicker}
            tickerSamples={tickerSamples}
          />
        )}

        {subTab === 'methodology' && (
          <HowItWorksView projectData={projectData} />
        )}
      </div>
    </div>
  );
}
