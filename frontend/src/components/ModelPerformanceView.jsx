import React, { useState } from 'react';
import { BarChart3, TrendingUp, AlertTriangle, ShieldCheck, Copy, Check, Info } from 'lucide-react';

export default function ModelPerformanceView({ projectData }) {
  const [selectedExpTab, setSelectedExpTab] = useState('expC');
  const [copied, setCopied] = useState(false);
  const [showPlainGuide, setShowPlainGuide] = useState(true);

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
    { id: 'expC', name: 'Exp C: Hidden=128 (Selected Best Model)', path: '/plots/training_curves_expC.png' },
    { id: 'expA', name: 'Exp A: Baseline (LB=30, Hidden=64)', path: '/plots/training_curves_expA.png' },
    { id: 'expB', name: 'Exp B: Lookback=60 (LB=60, Hidden=64)', path: '/plots/training_curves_expB.png' },
    { id: 'expD', name: 'Exp D: Low LR (LB=30, Hidden=64, LR=0.0005)', path: '/plots/training_curves_expD.png' },
  ];

  return (
    <div className="research-view">
      {/* Header */}
      <div className="research-header">
        <div>
          <div className="view-badge">
            <BarChart3 size={14} />
            <span>Out-of-Sample Evaluation</span>
          </div>
          <h2 className="research-title">Model Performance &amp; Evaluation Results</h2>
          <p className="research-subtitle">
            Rigorous evaluation on 458,428 chronological holdout sequences (2022 &ndash; 2026) across 472 S&amp;P 500 companies.
          </p>
        </div>

        <div className="research-header-actions">
          <button
            type="button"
            className="command-copy-button"
            onClick={copyCommand}
            title="Click to copy CLI evaluation command"
          >
            <code className="cmd-code">{evalCommand}</code>
            <span className="copy-label">{copied ? '✓ Copied' : 'Copy Command'}</span>
          </button>
        </div>
      </div>

      {/* Primary Scorecard */}
      <section className="research-section">
        <div className="section-title-row">
          <h3 className="section-heading">Out-of-Sample Test Scorecard</h3>
          <span className="section-pill">Chronological Test Set (15% Holdout)</span>
        </div>

        <div className="metrics-grid">
          {/* Accuracy */}
          <div className="metric-cell">
            <span className="metric-label">LSTM Test Accuracy</span>
            <span className="metric-value font-mono">
              {(metrics.accuracy * 100).toFixed(2)}%
            </span>
            <span className="metric-sublabel text-up">
              +{(baseline.delta_accuracy * 100).toFixed(2)}% vs Majority Baseline
            </span>
            <p className="metric-plain-desc">
              Correctly predicted direction on 237,231 out of 458,428 holdout trading days.
            </p>
          </div>

          {/* Baseline */}
          <div className="metric-cell">
            <span className="metric-label">Majority Baseline</span>
            <span className="metric-value font-mono">
              {(baseline.baseline_accuracy * 100).toFixed(2)}%
            </span>
            <span className="metric-sublabel text-muted">
              Always-Predict-UP Benchmark
            </span>
            <p className="metric-plain-desc">
              What you get with zero machine learning simply guessing &ldquo;UP&rdquo; every day.
            </p>
          </div>

          {/* Precision */}
          <div className="metric-cell">
            <span className="metric-label">Test Precision</span>
            <span className="metric-value font-mono">
              {(metrics.precision * 100).toFixed(1)}%
            </span>
            <span className="metric-sublabel text-muted">
              Accuracy of UP Calls
            </span>
            <p className="metric-plain-desc">
              When the model called &ldquo;UP&rdquo;, the stock actually went up 51.9% of the time.
            </p>
          </div>

          {/* Recall */}
          <div className="metric-cell">
            <span className="metric-label">Test Recall</span>
            <span className="metric-value font-mono">
              {(metrics.recall * 100).toFixed(1)}%
            </span>
            <span className="metric-sublabel text-muted">
              True Positive Sensitivity
            </span>
            <p className="metric-plain-desc">
              Caught 90.7% of all upward days, but produced 199,101 false alarms doing so.
            </p>
          </div>
        </div>

        {/* Honest Commentary Callout */}
        <div className="research-callout">
          <div className="callout-header-row">
            <AlertTriangle size={18} className="text-amber" />
            <h4 className="callout-title">Honest Performance Interpretation</h4>
          </div>
          <p className="callout-text">
            While the LSTM model achieves <strong>51.75% directional accuracy</strong> on 458,428 out-of-sample sequences—exceeding the 51.64% majority baseline by <strong>0.11 percentage points</strong>—this difference does <strong>not</strong> establish a reliable or profitable trading advantage.
          </p>
          <p className="callout-text" style={{ marginTop: '8px' }}>
            In live financial trading, bid-ask spreads, broker commissions, and execution slippage would quickly eliminate this tiny margin. The model&rsquo;s high recall (90.7%) demonstrates that it has primarily learned the multi-year secular upward drift of the S&amp;P 500, rather than micro-timing alpha.
          </p>
        </div>
      </section>

      {/* Plain-English Confusion Matrix */}
      <section className="research-section">
        <div className="section-title-row">
          <h3 className="section-heading">Understanding the Errors: 2&times;2 Confusion Matrix</h3>
          <span className="section-pill">458,428 Total Out-of-Sample Sequences</span>
        </div>

        <div className="confusion-matrix-wrapper">
          <div className="confusion-table-container">
            <table className="confusion-table">
              <thead>
                <tr>
                  <th className="diag-header">Actual \ Predicted</th>
                  <th>Predicted DOWN (0)</th>
                  <th>Predicted UP (1)</th>
                  <th>Total Actual</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <th className="row-header">Actual DOWN (0)</th>
                  <td className="cm-cell cm-tn">
                    <span className="cm-val font-mono">{metrics.tn?.toLocaleString()}</span>
                    <span className="cm-desc">True Negatives (Correct Down)</span>
                  </td>
                  <td className="cm-cell cm-fp">
                    <span className="cm-val font-mono">{metrics.fp?.toLocaleString()}</span>
                    <span className="cm-desc">False Positives (False Alarms)</span>
                  </td>
                  <td className="cm-total font-mono">
                    {(metrics.tn + metrics.fp)?.toLocaleString()}
                  </td>
                </tr>
                <tr>
                  <th className="row-header">Actual UP (1)</th>
                  <td className="cm-cell cm-fn">
                    <span className="cm-val font-mono">{metrics.fn?.toLocaleString()}</span>
                    <span className="cm-desc">False Negatives (Missed Rallies)</span>
                  </td>
                  <td className="cm-cell cm-tp">
                    <span className="cm-val font-mono">{metrics.tp?.toLocaleString()}</span>
                    <span className="cm-desc">True Positives (Correct Up)</span>
                  </td>
                  <td className="cm-total font-mono">
                    {(metrics.fn + metrics.tp)?.toLocaleString()}
                  </td>
                </tr>
                <tr className="cm-col-total-row">
                  <th className="row-header">Total Predicted</th>
                  <td className="cm-total font-mono">{(metrics.tn + metrics.fn)?.toLocaleString()}</td>
                  <td className="cm-total font-mono">{(metrics.fp + metrics.tp)?.toLocaleString()}</td>
                  <td className="cm-grand-total font-mono">{metrics.total_samples?.toLocaleString()}</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="confusion-legend">
            <h4 className="legend-heading">Plain-English Translation:</h4>
            <div className="legend-item">
              <span className="legend-dot dot-tp" />
              <div>
                <strong>True Positives (214,634):</strong> The stock went UP, and the model correctly predicted UP.
              </div>
            </div>
            <div className="legend-item">
              <span className="legend-dot dot-fp" />
              <div>
                <strong>False Positives (199,101):</strong> The main weakness. The stock dropped, but the model mistakenly predicted UP.
              </div>
            </div>
            <div className="legend-item">
              <span className="legend-dot dot-tn" />
              <div>
                <strong>True Negatives (22,597):</strong> The stock dropped, and the model correctly predicted DOWN.
              </div>
            </div>
            <div className="legend-item">
              <span className="legend-dot dot-fn" />
              <div>
                <strong>False Negatives (22,096):</strong> The stock rose, but the model mistakenly predicted DOWN.
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Model Experiments Comparison */}
      <section className="research-section">
        <div className="section-title-row">
          <h3 className="section-heading">Architecture &amp; Hyperparameter Experiments</h3>
          <span className="section-pill">4 Model Checkpoints Evaluated</span>
        </div>

        <div className="table-scroll-wrapper">
          <table className="financial-table">
            <thead>
              <tr>
                <th className="text-left">Experiment</th>
                <th className="text-right">Window</th>
                <th className="text-right">Hidden Units</th>
                <th className="text-right">Learning Rate</th>
                <th className="text-right">Best Epoch</th>
                <th className="text-right">Val Loss</th>
                <th className="text-right">Val Accuracy</th>
                <th className="text-right">Val F1</th>
                <th className="text-center">Selection</th>
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
                        <span className="status-badge-selected">★ Best Model</span>
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

      {/* Training Loss & Accuracy Curves Viewer */}
      <section className="research-section">
        <div className="section-title-row">
          <h3 className="section-heading">Training &amp; Validation Loss Curves</h3>
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
                    alt={`${plot.name} training loss curve`}
                    className="research-plot-image"
                  />
                </div>
              </div>
            );
          })}
        </div>
      </section>
    </div>
  );
}
