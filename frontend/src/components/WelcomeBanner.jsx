import React, { useState } from 'react';
import { Sparkles, TrendingUp, HelpCircle, AlertTriangle, ChevronUp, ChevronDown } from 'lucide-react';

export default function WelcomeBanner({ selectedTicker, onSelectTicker }) {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className="welcome-banner">
      <div className="welcome-header">
        <div className="welcome-title-group">
          <div className="welcome-badge">
            <Sparkles size={14} className="sparkle-icon" />
            <span>Interactive Research Tool</span>
          </div>
          <h2 className="welcome-heading">S&amp;P 500 Stock Movement Predictor</h2>
          <p className="welcome-subheading">
            Explore historical stock prices and predict whether a stock may move UP or DOWN in the next trading session using an LSTM model.
          </p>
        </div>

        <button
          type="button"
          className="collapse-banner-btn"
          onClick={() => setCollapsed(!collapsed)}
          title={collapsed ? 'Show quick guide' : 'Hide quick guide'}
        >
          <span>{collapsed ? 'How to Use' : 'Minimize Guide'}</span>
          {collapsed ? <ChevronDown size={14} /> : <ChevronUp size={14} />}
        </button>
      </div>

      {!collapsed && (
        <div className="welcome-steps-grid">
          {/* Step 1 */}
          <div className="step-card">
            <div className="step-num-badge">1</div>
            <div className="step-content">
              <h4 className="step-title">Choose a Stock</h4>
              <p className="step-desc">
                Select from popular S&amp;P 500 equities like{' '}
                <button
                  type="button"
                  className="inline-ticker-btn"
                  onClick={() => onSelectTicker('AAPL')}
                >
                  AAPL
                </button>
                ,{' '}
                <button
                  type="button"
                  className="inline-ticker-btn"
                  onClick={() => onSelectTicker('MSFT')}
                >
                  MSFT
                </button>
                , or{' '}
                <button
                  type="button"
                  className="inline-ticker-btn"
                  onClick={() => onSelectTicker('NVDA')}
                >
                  NVDA
                </button>
                .
              </p>
            </div>
          </div>

          {/* Step 2 */}
          <div className="step-card">
            <div className="step-num-badge">2</div>
            <div className="step-content">
              <h4 className="step-title">Explore History</h4>
              <p className="step-desc">
                View interactive closing prices, trading volume, and 20/50-day moving averages across custom date ranges.
              </p>
            </div>
          </div>

          {/* Step 3 */}
          <div className="step-card">
            <div className="step-num-badge">3</div>
            <div className="step-content">
              <h4 className="step-title">Request Prediction</h4>
              <p className="step-desc">
                Click <strong>&ldquo;Predict Next Trading Day&rdquo;</strong> to run the PyTorch LSTM model on the trailing 30 sessions.
              </p>
            </div>
          </div>

          {/* Step 4: Caution */}
          <div className="step-card caution-step">
            <div className="step-num-badge caution-badge">
              <AlertTriangle size={14} />
            </div>
            <div className="step-content">
              <h4 className="step-title">Experimental Disclaimer</h4>
              <p className="step-desc">
                Predictions are for machine-learning research (51.75% test accuracy) and do <strong>not</strong> guarantee profits or constitute financial advice.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
