import React, { useState } from 'react';
import { BookOpen, Database, Sliders, Cpu, Target, ChevronDown, ChevronUp, Layers, CheckCircle } from 'lucide-react';

export default function HowItWorksView({ projectData }) {
  const [expandedSection, setExpandedSection] = useState(null);

  const toggleSection = (id) => {
    setExpandedSection(expandedSection === id ? null : id);
  };

  const scalerFeatures = projectData?.scalerFeatures || [];

  return (
    <div className="how-it-works-view">
      <div className="how-it-works-header">
        <div className="view-badge">
          <BookOpen size={14} />
          <span>Educational Guide</span>
        </div>
        <h2 className="view-main-title">How the Stock Movement Predictor Works</h2>
        <p className="view-main-subtitle">
          A plain-English explanation of the data, the 8 engineered features, the 30-day LSTM model, and the next-day prediction target.
        </p>
      </div>

      <div className="how-steps-container">
        {/* Step 1: Historical Data */}
        <div className="how-card">
          <div className="how-card-header">
            <div className="how-icon-box">
              <Database size={22} className="text-accent" />
            </div>
            <div className="how-header-text">
              <span className="how-step-tag">Step 1</span>
              <h3 className="how-title">Historical S&amp;P 500 Constituent Data</h3>
              <p className="how-summary">
                The project analyzes 26 years of real market data (2000 &ndash; 2026) across 472 individual companies, totaling over 2.67 million trading days.
              </p>
            </div>
          </div>

          <div className="how-plain-box">
            <h4 className="plain-label">In Plain English:</h4>
            <p className="plain-text">
              Stock markets don&rsquo;t move in isolation. Rather than studying only a single stock, the model was trained on historical daily price records (Open, High, Low, Close) and trading volumes across 472 distinct S&amp;P 500 constituents to learn general market momentum patterns.
            </p>
          </div>

          <button
            type="button"
            className="how-expand-btn"
            onClick={() => toggleSection('data')}
          >
            <span>{expandedSection === 'data' ? 'Hide Technical Details' : 'Show Technical Data Details'}</span>
            {expandedSection === 'data' ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>

          {expandedSection === 'data' && (
            <div className="how-tech-panel">
              <div className="tech-bullet-list">
                <div className="tech-bullet-item">
                  <CheckCircle size={14} className="text-up" />
                  <span><strong>Total records:</strong> 2,678,863 cleaned daily rows after dropping 24,196 50-day warm-up NaNs.</span>
                </div>
                <div className="tech-bullet-item">
                  <CheckCircle size={14} className="text-up" />
                  <span><strong>Chronological split:</strong> 70% Train (2000–2018), 15% Validation (2018–2022), 15% Test (2022–2026).</span>
                </div>
                <div className="tech-bullet-item">
                  <CheckCircle size={14} className="text-up" />
                  <span><strong>No temporal leakage:</strong> All testing sequences occur strictly after training dates. Shuffling was forbidden.</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Step 2: The 8 Engineered Features */}
        <div className="how-card">
          <div className="how-card-header">
            <div className="how-icon-box">
              <Sliders size={22} className="text-accent" />
            </div>
            <div className="how-header-text">
              <span className="how-step-tag">Step 2</span>
              <h3 className="how-title">The Eight Engineered Market Features</h3>
              <p className="how-summary">
                Raw prices change over time, so the model instead looks at 8 normalized momentum, volume, and volatility indicators.
              </p>
            </div>
          </div>

          <div className="how-plain-box">
            <h4 className="plain-label">In Plain English:</h4>
            <p className="plain-text">
              A stock priced at $20 and a stock priced at $500 can behave similarly if their percentage returns and momentum match. By calculating percentages, volume changes, and moving average ratios, all stocks speak the same language to the neural network.
            </p>
          </div>

          <div className="features-preview-grid">
            <div className="feature-pill-card">
              <span className="feat-code font-mono">Daily_Return</span>
              <span className="feat-desc">Percentage price change from yesterday</span>
            </div>
            <div className="feature-pill-card">
              <span className="feat-code font-mono">Volume_Change</span>
              <span className="feat-desc">Surge or drop in trading volume</span>
            </div>
            <div className="feature-pill-card">
              <span className="feat-code font-mono">High_Low_Range</span>
              <span className="feat-desc">Intraday trading spread (volatility)</span>
            </div>
            <div className="feature-pill-card">
              <span className="feat-code font-mono">Rolling_Volatility_20</span>
              <span className="feat-desc">20-day standard deviation of returns</span>
            </div>
            <div className="feature-pill-card">
              <span className="feat-code font-mono">SMA20 &amp; SMA50</span>
              <span className="feat-desc">20-day and 50-day average price benchmarks</span>
            </div>
            <div className="feature-pill-card">
              <span className="feat-code font-mono">Close/SMA Ratios</span>
              <span className="feat-desc">How far current price is stretched from average</span>
            </div>
          </div>

          <button
            type="button"
            className="how-expand-btn"
            onClick={() => toggleSection('features')}
          >
            <span>{expandedSection === 'features' ? 'Hide Mathematical Formulas' : 'Show Mathematical Formulas'}</span>
            {expandedSection === 'features' ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>

          {expandedSection === 'features' && (
            <div className="how-tech-panel">
              <div className="table-scroll-wrapper">
                <table className="financial-table">
                  <thead>
                    <tr>
                      <th className="text-left">Feature Name</th>
                      <th className="text-right">Training Mean</th>
                      <th className="text-right">Training Std</th>
                      <th className="text-left">Formula / Definition</th>
                    </tr>
                  </thead>
                  <tbody>
                    {scalerFeatures.map((f, i) => (
                      <tr key={i} className="table-row">
                        <td className="text-left font-mono font-medium">{f.feature}</td>
                        <td className="text-right font-mono">{f.mean.toFixed(4)}</td>
                        <td className="text-right font-mono">{f.std.toFixed(4)}</td>
                        <td className="text-left font-sans text-secondary">{f.description}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>

        {/* Step 3: The LSTM Model & 30-Day Window */}
        <div className="how-card">
          <div className="how-card-header">
            <div className="how-icon-box">
              <Cpu size={22} className="text-accent" />
            </div>
            <div className="how-header-text">
              <span className="how-step-tag">Step 3</span>
              <h3 className="how-title">The 30-Session LSTM Neural Network</h3>
              <p className="how-summary">
                A Long Short-Term Memory (LSTM) network designed to remember sequential trends over a 30-trading-day sliding window.
              </p>
            </div>
          </div>

          <div className="how-plain-box">
            <h4 className="plain-label">In Plain English:</h4>
            <p className="plain-text">
              Standard neural networks look at each day in isolation. An LSTM has internal memory cells: it reads the last 30 trading sessions chronologically (about 6 calendar weeks), updating its hidden state each day to recognize patterns such as pullbacks, momentum exhaustion, or consolidation before making a prediction.
            </p>
          </div>

          <button
            type="button"
            className="how-expand-btn"
            onClick={() => toggleSection('model')}
          >
            <span>{expandedSection === 'model' ? 'Hide Architecture Details' : 'Show Model Architecture'}</span>
            {expandedSection === 'model' ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>

          {expandedSection === 'model' && (
            <div className="how-tech-panel">
              <div className="tech-bullet-list">
                <div className="tech-bullet-item">
                  <CheckCircle size={14} className="text-up" />
                  <span><strong>Input Dimensions:</strong> 30 time steps &times; 8 features = 240 normalized inputs per sequence.</span>
                </div>
                <div className="tech-bullet-item">
                  <CheckCircle size={14} className="text-up" />
                  <span><strong>Network Structure:</strong> 2 LSTM layers with 128 hidden units each, followed by a linear classification head.</span>
                </div>
                <div className="tech-bullet-item">
                  <CheckCircle size={14} className="text-up" />
                  <span><strong>Dropout:</strong> 0.3 dropout between layers to prevent overfitting on training noise.</span>
                </div>
                <div className="tech-bullet-item">
                  <CheckCircle size={14} className="text-up" />
                  <span><strong>Saved Weights:</strong> Stored in <code>models/best_lstm_model.pt</code> (selected from Experiment C).</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Step 4: The Next-Day UP/DOWN Target */}
        <div className="how-card">
          <div className="how-card-header">
            <div className="how-icon-box">
              <Target size={22} className="text-accent" />
            </div>
            <div className="how-header-text">
              <span className="how-step-tag">Step 4</span>
              <h3 className="how-title">The Next-Day UP / DOWN Target</h3>
              <p className="how-summary">
                Formulated as a binary classification problem: will tomorrow&rsquo;s close price be higher or lower than today&rsquo;s close?
              </p>
            </div>
          </div>

          <div className="how-plain-box">
            <h4 className="plain-label">In Plain English:</h4>
            <p className="plain-text">
              Predicting the exact dollar price of a stock is notoriously noisy. Instead, this model solves a simpler binary question: <strong>UP (1)</strong> if tomorrow&rsquo;s closing price is greater than today&rsquo;s, and <strong>DOWN (0)</strong> if it is equal or lower. The model outputs a probability between 0% and 100%.
            </p>
          </div>

          <button
            type="button"
            className="how-expand-btn"
            onClick={() => toggleSection('target')}
          >
            <span>{expandedSection === 'target' ? 'Hide Statistical Baseline' : 'Show Statistical Baseline'}</span>
            {expandedSection === 'target' ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>

          {expandedSection === 'target' && (
            <div className="how-tech-panel">
              <div className="tech-bullet-list">
                <div className="tech-bullet-item">
                  <CheckCircle size={14} className="text-up" />
                  <span><strong>Target formulation:</strong> <code>Target = 1 if Close(t+1) &gt; Close(t) else 0</code></span>
                </div>
                <div className="tech-bullet-item">
                  <CheckCircle size={14} className="text-up" />
                  <span><strong>Majority-Class Baseline:</strong> The test set has a 51.64% UP rate. An &ldquo;Always Predict UP&rdquo; strategy achieves 51.64% accuracy.</span>
                </div>
                <div className="tech-bullet-item">
                  <CheckCircle size={14} className="text-up" />
                  <span><strong>LSTM Result:</strong> The model achieves 51.75% accuracy (+0.11% difference), demonstrating high sensitivity to the secular equity drift.</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
