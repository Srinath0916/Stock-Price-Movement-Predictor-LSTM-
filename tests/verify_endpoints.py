"""
API Endpoint Real Output Collector.
"""
import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def run():
    print("=== 1. API HEALTH CHECK ===")
    r = requests.get(f"{BASE_URL}/api/health")
    print(f"Status: {r.status_code}")
    print(json.dumps(r.json(), indent=2))

    print("\n=== 2. MODEL INFO CHECK ===")
    r = requests.get(f"{BASE_URL}/api/model-info")
    print(f"Status: {r.status_code}")
    info = r.json()
    print(f"Model: {info['model_name']} ({info['model_version']})")
    print(f"Input Dim: {info['input_dim']} | Sequence Length: {info['sequence_length']}")
    print(f"Features: {info['feature_names']}")
    print(f"Test Accuracy: {info['test_metrics']['directional_accuracy']*100:.2f}% (Baseline: {info['test_metrics']['baseline_accuracy']*100:.2f}%)")

    print("\n=== 3. THREE ACTUAL PREDICTION REQUESTS ===")
    
    # Request 1: AAPL (Live)
    p1 = requests.post(f"{BASE_URL}/api/predict", json={"ticker": "AAPL", "threshold": 0.50, "use_live_market_data": True}).json()
    print(f"1. AAPL [Live]: {p1['prediction']} (Prob: {p1['probability']*100:.2f}%) | Base: ${p1['baseClose']} | Date: {p1['latestMarketDate']} | Source: {p1['dataSource']}")

    # Request 2: MSFT (Live)
    p2 = requests.post(f"{BASE_URL}/api/predict", json={"ticker": "MSFT", "threshold": 0.50, "use_live_market_data": True}).json()
    print(f"2. MSFT [Live]: {p2['prediction']} (Prob: {p2['probability']*100:.2f}%) | Base: ${p2['baseClose']} | Date: {p2['latestMarketDate']} | Source: {p2['dataSource']}")

    # Request 3: NVDA (Historical Dataset Mode)
    p3 = requests.post(f"{BASE_URL}/api/predict", json={"ticker": "NVDA", "threshold": 0.50, "use_live_market_data": False}).json()
    print(f"3. NVDA [Historical]: {p3['prediction']} (Prob: {p3['probability']*100:.2f}%) | Base: ${p3['baseClose']} | Date: {p3['latestMarketDate']} | Source: {p3['dataSource']}")

    print("\n=== 4. ACTUAL BACKTEST REQUEST ===")
    bt_payload = {
        "ticker": "AAPL",
        "start_date": "2023-01-01",
        "end_date": "2026-02-19",
        "strategy_mode": "long_only",
        "threshold": 0.50,
        "initial_capital": 10000.0,
        "transaction_cost_bps": 5.0
    }
    r_bt = requests.post(f"{BASE_URL}/api/backtest", json=bt_payload)
    print(f"Status: {r_bt.status_code}")
    bt = r_bt.json()
    m = bt["metrics"]
    print(f"Backtest {bt['ticker']} ({bt['startDate']} to {bt['endDate']}, {bt['totalSessions']} sessions):")
    print(f"  Strategy Return:   {m['strategyReturnPct']:+.2f}% (${bt['finalStrategyEquity']:,.2f})")
    print(f"  Benchmark Return:  {m['benchmarkReturnPct']:+.2f}% (${bt['finalBenchmarkEquity']:,.2f})")
    print(f"  Alpha:             {m['alphaPct']:+.2f}%")
    print(f"  Sharpe:            {m['sharpeRatio']} (Bench: {m['benchmarkSharpeRatio']})")
    print(f"  Max Drawdown:      -{m['maxDrawdownPct']:.2f}% (Bench: -{m['benchmarkMaxDrawdownPct']:.2f}%)")
    print(f"  Directional Acc:   {m['directionalAccuracyPct']:.2f}% | F1: {m['f1Score']}")
    print(f"  Confusion Matrix:  TP={m['tp']}, FP={m['fp']}, FN={m['fn']}, TN={m['tn']}")

    print("\n=== 5. ACTUAL WALK-FORWARD REQUEST ===")
    r_wf = requests.get(f"{BASE_URL}/api/walk-forward")
    print(f"Status: {r_wf.status_code}")
    wf = r_wf.json()
    print(f"Retrieved {len(wf)} Regimes:")
    for regime in wf:
        print(f"  - {regime['name']} ({regime['start_date']} to {regime['end_date']}): Acc={regime['accuracy']*100:.2f}%, F1={regime['f1']:.4f}, Sequences={regime['total_sequences']:,}, OOS={regime['is_out_of_sample']}")

if __name__ == "__main__":
    run()

