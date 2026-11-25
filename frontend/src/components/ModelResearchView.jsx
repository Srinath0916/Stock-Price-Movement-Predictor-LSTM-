import React, { useState } from 'react';

export default function ModelResearchView({ projectData }) {
  const [selectedExpTab, setSelectedExpTab] = useState('expC');
  const [copied, setCopied] = useState(false);

  const evalData = projectData?.finalTestResults || {};
  const metrics = evalData.metrics || {};
  const baseline = evalData.baseline_comparison || {};
  const experiments = projectData?.experiments || [];

  const evalCommand = '.venv/bin/python src/evaluation/evaluate_model.py';

  const copyCommand = () => {
    navigator.clipboard.writeText(evalCommand);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const expPlots = [
    { id: 'expC', name: 'Exp C: Hidden=128 (Best Model)', path: '/plots/training_curves_expC.png' },
    { id: 'expA', name: 'Exp A: Baseline (LB=30, Hidden=64)', path: '/plots/training_curves_expA.png' },
    { id: 'expB', name: 'Exp B: Lookback=60 (LB=60, Hidden=64)', path: '/plots/training_curves_expB.png' },
    { id: 'expD', name: 'Exp D: Low LR (LB=30, Hidden=64, LR=0.0005)', path: '/plots/training_curves_expD.png' },
  ];

  return (
    <div className="research-view">
      {/* Research Header */}
      <div className="research-header">
        <div>
          <h2 className="research-title">Model Research & Evaluation Results</h2>
          <p className="research-subtitle">
            Out-of-sample test results, 4 LSTM model experiments, confusion matrix, and training methodology across 458,428 holdout sequences.
          </p>
        </div>
        <div className="research-header-actions">
          <button
            type="button"
            className="command-copy-button"
            onClick={copyCommand}
            title="Click to copy evaluation CLI command"
          >
            <code className="cmd-code">{evalCommand}</code>
            <span className="copy-label">{copied ? '✓ Copied' : 'Copy Command'}</span>
          </button>
        </div>
      </div>

      {/* Out-of-sample Test Scorecard */}
      <section className="research-section">
        <div className="section-title-row">
          <h3 className="section-heading">Out-of-Sample Test Evaluation</h3>
          <span className="section-pill">Chronological Test Partition (15%)</span>
        </div>

        <div className="metrics-grid">
          <div className="metric-cell">
            <span className="metric-label">LSTM Test Accuracy</span>
            <span className="metric-value font-mono">
              {(metrics.accuracy * 100).toFixed(2)}%
            </span>
            <span className="metric-sublabel text-up">
              +{(baseline.delta_accuracy * 100).toFixed(2)}% vs Majority Baseline
            </span>
          </div>

          <div className="metric-cell">
            <span className="metric-label">Majority Class Baseline</span>
            <span className="metric-value font-mono">
              {(baseline.baseline_accuracy * 100).toFixed(2)}%
            </span>
            <span className="metric-sublabel text-muted">
              Always-Predict-UP strategy
            </span>
          </div>

          <div className="metric-cell">
            <span className="metric-label">Test Precision / Recall</span>
            <span className="metric-value font-mono">
              {(metrics.precision * 100).toFixed(1)}% / {(metrics.recall * 100).toFixed(1)}%
            </span>
            <span className="metric-sublabel text-muted">
              High positive recall (214k UP hits)
            </span>
          </div>

          <div className="metric-cell">
            <span className="metric-label">F1-Score / Test Loss</span>
            <span className="metric-value font-mono">
              {metrics.f1?.toFixed(4)} / {metrics.loss?.toFixed(4)}
            </span>
            <span className="metric-sublabel text-muted">
              Cross-entropy loss on 458k samples
            </span>
          </div>
        </div>

        {/* Honest Findings Commentary */}
        <div className="research-callout">
          <h4 className="callout-title">Honest Performance Interpretation</h4>
          <p className="callout-text">
            While the LSTM model achieves <strong>51.75% directional accuracy</strong> on 458,428 out-of-sample sequences—exceeding the 51.64% majority class baseline by <strong>0.11 percentage points</strong>—this difference is marginal. The model displays a high recall of 90.67% and produces 199,101 false positives, essentially reflecting the secular upward drift of the equity market rather than actionable timing alpha. In an operational trading environment, bid-ask spreads, transaction commissions, and slippage would consume this margin. Real-world predictive utility would require confidence calibration, selective trade filtering, and regime-aware features.
          </p>
        </div>
      </section>

      {/* Confusion Matrix & Error Analysis */}
      <section className="research-section">
        <div className="section-title-row">
          <h3 className="section-heading">Out-of-Sample Confusion Matrix</h3>
          <span className="section-pill">458,428 Sequences Evaluated</span>
        </div>

        <div className="confusion-matrix-wrapper">
          <div className="confusion-table-container">
            <table className="confusion-table">
              <thead>
                <tr>
                  <th className="diag-header">Actual \ Predicted</th>
                  <th>Predicted DOWN (0)</th>
                  <th>Predicted UP (1)</th>
                  <th>Row Total</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <th className="row-header">Actual DOWN (0)</th>
                  <td className="cm-cell cm-tn">
                    <span className="cm-val">{metrics.tn?.toLocaleString()}</span>
                    <span className="cm-desc">True Negatives (Correct Down)</span>
                  </td>
                  <td className="cm-cell cm-fp">
                    <span className="cm-val">{metrics.fp?.toLocaleString()}</span>
                    <span className="cm-desc">False Positives (Type I Error)</span>
                  </td>
                  <td className="cm-total">
                    {(metrics.tn + metrics.fp)?.toLocaleString()}
                  </td>
                </tr>
                <tr>
                  <th className="row-header">Actual UP (1)</th>
                  <td className="cm-cell cm-fn">
                    <span className="cm-val">{metrics.fn?.toLocaleString()}</span>
                    <span className="cm-desc">False Negatives (Type II Error)</span>
                  </td>
                  <td className="cm-cell cm-tp">
                    <span className="cm-val">{metrics.tp?.toLocaleString()}</span>
                    <span className="cm-desc">True Positives (Correct Up)</span>
                  </td>
                  <td className="cm-total">
                    {(metrics.fn + metrics.tp)?.toLocaleString()}
                  </td>
                </tr>
                <tr className="cm-col-total-row">
                  <th className="row-header">Column Total</th>
                  <td className="cm-total">{(metrics.tn + metrics.fn)?.toLocaleString()}</td>
                  <td className="cm-total">{(metrics.fp + metrics.tp)?.toLocaleString()}</td>
                  <td className="cm-grand-total">{metrics.total_samples?.toLocaleString()}</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="confusion-legend">
            <div className="legend-item">
              <span className="legend-dot dot-tp"></span>
              <div>
                <strong>True Positives (214,634):</strong> The model correctly anticipated upward next-day sessions.
              </div>
            </div>
            <div className="legend-item">
              <span className="legend-dot dot-fp"></span>
              <div>
                <strong>False Positives (199,101):</strong> Major failure mode—predicting UP when price dropped.
              </div>
            </div>
            <div className="legend-item">
              <span className="legend-dot dot-tn"></span>
              <div>
                <strong>True Negatives (22,597):</strong> Successfully identified downward sessions.
              </div>
            </div>
            <div className="legend-item">
              <span className="legend-dot dot-fn"></span>
              <div>
                <strong>False Negatives (22,096):</strong> Predicted DOWN when price actually rose.
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Model Experiments Table */}
      <section className="research-section">
        <div className="section-title-row">
          <h3 className="section-heading">Hyperparameter & Architecture Experiments</h3>
          <span className="section-pill">4 Model Variations Compared</span>
        </div>

        <div className="table-scroll-wrapper">
          <table className="financial-table">
            <thead>
              <tr>
                <th className="text-left">Experiment</th>
                <th className="text-right">Lookback</th>
                <th className="text-right">Hidden Units</th>
                <th className="text-right">Learning Rate</th>
                <th className="text-right">Best Epoch</th>
                <th className="text-right">Val Loss</th>
                <th className="text-right">Val Accuracy</th>
                <th className="text-right">Val F1</th>
                <th className="text-center">Status</th>
              </tr>
            </thead>
            <tbody>
              {experiments.map((exp, idx) => {
                const isSelected = exp.Experiment.includes('Experiment C');
                return (
                  <tr key={idx} className={`table-row ${isSelected ? 'selected-row' : ''}`}>
                    <td className="text-left font-medium">
                      {exp.Experiment}
                    </td>
                    <td className="text-right font-mono">{exp.Lookback} days</td>
                    <td className="text-right font-mono">{exp.Hidden}</td>
                    <td className="text-right font-mono">{exp.LR}</td>
                    <td className="text-right font-mono">{exp.Best_Epoch} / {exp.Total_Epochs}</td>
                    <td className="text-right font-mono">{exp.Val_Loss.toFixed(4)}</td>
                    <td className="text-right font-mono font-bold">{exp.Val_Acc.toFixed(2)}%</td>
                    <td className="text-right font-mono">{exp.Val_F1.toFixed(4)}</td>
                    <td className="text-center">
                      {isSelected ? (
                        <span className="status-badge-selected">★ Best Checkpoint</span>
                      ) : (
                        <span className="status-badge-archived">Archived</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      {/* Training Loss & Accuracy Curves */}
      <section className="research-section">
        <div className="section-title-row">
          <h3 className="section-heading">Training & Validation Loss Curves</h3>
          <div className="toggle-pill-group">
            {expPlots.map((plot) => (
              <button
                key={plot.id}
                type="button"
                className={`toggle-pill ${selectedExpTab === plot.id ? 'active' : ''}`}
                onClick={() => setSelectedExpTab(plot.id)}
              >
                {plot.id.toUpperCase()}
              </button>
            ))}
          </div>
        </div>

        <div className="plot-display-card">
          {expPlots.map((plot) => {
            if (plot.id !== selectedExpTab) return null;
            return (
              <div key={plot.id} className="plot-container">
                <div className="plot-meta-header">
                  <span className="plot-title font-medium">{plot.name}</span>
                  <span className="plot-source font-mono text-muted">{plot.path}</span>
                </div>
                <div className="plot-image-wrapper">
                  <img
                    src={plot.path}
                    alt={`${plot.name} loss and accuracy curve`}
                    className="research-plot-image"
                  />
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* Methodology & Reproducibility */}
      <section className="research-section">
        <div className="section-title-row">
          <h3 className="section-heading">Methodology & Rigorous Validation</h3>
          <span className="section-pill">Strict Prevention of Data Leakage</span>
        </div>

        <div className="methodology-grid">
          <div className="methodology-card">
            <h4 className="method-card-title">1. Chronological Partitioning</h4>
            <p className="method-card-body">
              All 472 tickers were split chronologically: <strong>70% Train</strong> (2,139,332 sequences), <strong>15% Validation</strong> (458,428 sequences), and <strong>15% Test</strong> (458,428 sequences). No random shuffling was permitted across time horizons, preserving realistic temporal sequence structure.
            </p>
          </div>

          <div className="methodology-card">
            <h4 className="method-card-title">2. Isolated Feature Scaling</h4>
            <p className="method-card-body">
              The <code>StandardScaler</code> was fitted strictly on the chronological training partition and saved to <code>models/scaler.joblib</code>. Validation and test sequences are transformed with frozen parameters, preventing lookahead leakage.
            </p>
          </div>

          <div className="methodology-card">
            <h4 className="method-card-title">3. Binary Next-Day Target</h4>
            <p className="method-card-body">
              Target is formulated as <code>1</code> if <code>Close(t+1) &gt; Close(t)</code>, and <code>0</code> otherwise. The test set exhibits a natural 51.64% positive base rate, serving as the benchmark for statistical significance.
            </p>
          </div>

          <div className="methodology-card">
            <h4 className="method-card-title">4. Model Artifacts Preserved</h4>
            <p className="method-card-body">
              PyTorch checkpoint <code>models/best_lstm_model.pt</code> and scaler <code>models/scaler.joblib</code> are fully reproducible. Run <code>python src/evaluation/evaluate_model.py</code> to reproduce identical metrics directly from disk.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
