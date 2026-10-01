"""
Exploratory Data Analysis (EDA) Script for S&P 500 Historical Data.
Analyzes the cleaned Parquet dataset, computes statistical distributions,
identifies anomalies/outliers, and exports visualizations to results/.
Does not modify the underlying data or create model pipelines.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script execution
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats

# Aesthetics setup
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 150


def run_eda(parquet_path: Path, results_dir: Path) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading cleaned dataset from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    # 1. Overall dataset size & structure
    total_rows = len(df)
    total_cols = len(df.columns)
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

    print("\n" + "=" * 80)
    print("1. OVERALL DATASET SIZE & STRUCTURE")
    print("=" * 80)
    print(f"Total Rows:    {total_rows:,}")
    print(f"Total Columns: {total_cols} ({list(df.columns)})")
    print(f"Memory Usage:  {memory_mb:.2f} MB in RAM")

    # 2. Number of unique tickers
    unique_tickers = df["Ticker"].unique()
    num_tickers = len(unique_tickers)
    print("\n" + "=" * 80)
    print("2. UNIQUE TICKERS")
    print("=" * 80)
    print(f"Total Unique Tickers: {num_tickers}")

    # 3. Date coverage
    min_date = df["Date"].min()
    max_date = df["Date"].max()
    unique_dates = df["Date"].nunique()
    print("\n" + "=" * 80)
    print("3. DATE COVERAGE")
    print("=" * 80)
    print(f"Date Range:       {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    print(f"Total Days Span:  {(max_date - min_date).days:,} calendar days")
    print(f"Trading Sessions: {unique_dates:,} unique market dates")

    # 4 & 5. Records per ticker & distribution
    ticker_counts = df["Ticker"].value_counts()
    print("\n" + "=" * 80)
    print("4 & 5. TRADING RECORDS PER TICKER & DISTRIBUTION")
    print("=" * 80)
    print(f"Max records per ticker:    {ticker_counts.max():,} (e.g. {ticker_counts.idxmax()})")
    print(f"Min records per ticker:    {ticker_counts.min():,} (e.g. {ticker_counts.idxmin()})")
    print(f"Mean records per ticker:   {ticker_counts.mean():.1f}")
    print(f"Median records per ticker: {ticker_counts.median():.1f}")
    print(f"Std dev of records:        {ticker_counts.std():.1f}")
    
    full_history_count = (ticker_counts == ticker_counts.max()).sum()
    lt_1000_count = (ticker_counts < 1000).sum()
    lt_2500_count = (ticker_counts < 2500).sum()
    print(f"Tickers with full history ({ticker_counts.max()} days): {full_history_count} ({full_history_count / num_tickers * 100:.1f}%)")
    print(f"Tickers with < 1,000 days history: {lt_1000_count} ({lt_1000_count / num_tickers * 100:.1f}%)")
    print(f"Tickers with < 2,500 days history: {lt_2500_count} ({lt_2500_count / num_tickers * 100:.1f}%)")

    # Chart 1: Distribution of records per ticker
    plt.figure(figsize=(9, 5))
    sns.histplot(ticker_counts, bins=40, kde=True, color="#2b5c8f", edgecolor="black")
    plt.title("Distribution of Historical Trading Days per Ticker", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Number of Trading Days", fontsize=11)
    plt.ylabel("Number of Tickers", fontsize=11)
    plt.axvline(ticker_counts.median(), color="red", linestyle="--", label=f"Median ({ticker_counts.median():.0f})")
    plt.legend()
    plt.tight_layout()
    chart1_path = results_dir / "eda_records_per_ticker.png"
    plt.savefig(chart1_path)
    plt.close()
    print(f"Saved: {chart1_path}")

    # 6. Closing price behavior for representative stocks
    rep_tickers = ["AAPL", "MSFT", "JPM", "JNJ", "CAT"]
    print("\n" + "=" * 80)
    print("6 & 11. CLOSING PRICE & VOLUME HISTORY FOR REPRESENTATIVE STOCKS")
    print("=" * 80)
    print(f"Representative stocks selected across sectors: {rep_tickers}")

    # Chart 2: Representative Stock Price History (Dual panel: Raw Price & Normalized Base 100)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ticker in rep_tickers:
        sub = df[df["Ticker"] == ticker].sort_values("Date")
        axes[0].plot(sub["Date"], sub["Close"], label=ticker, linewidth=1.4)
        # Normalized to 100 at start of ticker history
        normalized = sub["Close"] / sub["Close"].iloc[0] * 100
        axes[1].plot(sub["Date"], normalized, label=ticker, linewidth=1.4)

    axes[0].set_title("Historical Close Prices ($) - Representative Stocks", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Close Price ($)", fontsize=11)
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Normalized Performance (Base 100 at Start)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Normalized Index (Start = 100)", fontsize=11)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Date", fontsize=11)
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    chart2_path = results_dir / "eda_representative_prices.png"
    plt.savefig(chart2_path)
    plt.close()
    print(f"Saved: {chart2_path}")

    # 7. Daily price returns and distribution
    print("\n" + "=" * 80)
    print("7. DAILY RETURNS DISTRIBUTION & STATS")
    print("=" * 80)
    print("Calculating daily percentage returns per ticker...")
    df["Return"] = df.groupby("Ticker")["Close"].pct_change()
    returns = df["Return"].dropna()

    mean_ret = returns.mean()
    median_ret = returns.median()
    std_ret = returns.std()
    skew_ret = stats.skew(returns)
    kurt_ret = stats.kurtosis(returns)  # Fisher kurtosis: normal = 0, positive = fat tails

    print(f"Mean Daily Return:      {mean_ret * 100:.4f}%")
    print(f"Median Daily Return:    {median_ret * 100:.4f}%")
    print(f"Std Dev Daily Return:   {std_ret * 100:.4f}%")
    print(f"Skewness:               {skew_ret:.4f} (slight negative/positive skew)")
    print(f"Excess Kurtosis:        {kurt_ret:.2f} (Extreme leptokurtosis / fat tails >> 3)")

    quantiles = [0.001, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 0.999]
    q_vals = returns.quantile(quantiles)
    print("\nDaily Return Quantiles:")
    for q, val in q_vals.items():
        print(f"  {q*100:5.1f}% percentile: {val*100:+.2f}%")

    # Chart 3: Daily Returns Distribution (Linear & Log scale density)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # Clip returns visually to [-15%, +15%] for clean density inspection
    clipped_returns = returns[(returns >= -0.15) & (returns <= 0.15)] * 100
    sns.histplot(clipped_returns, bins=120, kde=True, ax=axes[0], color="#1f77b4", stat="density")
    axes[0].set_title("Daily Return Density (Central -15% to +15%)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Daily Return (%)")
    axes[0].set_ylabel("Density")

    # Log-density to show heavy fat tails
    counts, bin_edges = np.histogram(returns * 100, bins=200)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    valid_counts = counts > 0
    axes[1].semilogy(bin_centers[valid_counts], counts[valid_counts], "o-", markersize=2.5, color="#d62728", label="Empirical Returns")
    axes[1].set_title("Log-Count Return Distribution (Fat Tails)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Daily Return (%)")
    axes[1].set_ylabel("Log Frequency")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    chart3_path = results_dir / "eda_returns_distribution.png"
    plt.savefig(chart3_path)
    plt.close()
    print(f"Saved: {chart3_path}")

    # 8. Trading volume distribution
    print("\n" + "=" * 80)
    print("8. TRADING VOLUME DISTRIBUTION")
    print("=" * 80)
    vol_mean = df["Volume"].mean()
    vol_median = df["Volume"].median()
    vol_std = df["Volume"].std()
    print(f"Mean Daily Volume:   {vol_mean:,.0f} shares")
    print(f"Median Daily Volume: {vol_median:,.0f} shares")
    print(f"Std Dev Volume:      {vol_std:,.0f} shares")

    vol_quantiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    print("\nVolume Quantiles:")
    for q in vol_quantiles:
        print(f"  {q*100:5.1f}%: {df['Volume'].quantile(q):,.0f} shares")

    # Chart 4: Volume Distribution (Log10 scale)
    plt.figure(figsize=(9, 5))
    positive_vol = df[df["Volume"] > 0]["Volume"]
    log_vol = np.log10(positive_vol)
    sns.histplot(log_vol, bins=80, kde=True, color="#2ca02c", edgecolor="black")
    plt.title("Distribution of Daily Trading Volume (Log10 Scale)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Log10(Volume)", fontsize=11)
    plt.ylabel("Count", fontsize=11)
    plt.axvline(np.log10(vol_median), color="red", linestyle="--", label=f"Median: 10^{np.log10(vol_median):.2f} (~{vol_median:,.0f})")
    plt.legend()
    plt.tight_layout()
    chart4_path = results_dir / "eda_volume_distribution.png"
    plt.savefig(chart4_path)
    plt.close()
    print(f"Saved: {chart4_path}")

    # 9. Relationship between price movement and volume
    print("\n" + "=" * 80)
    print("9. RELATIONSHIP BETWEEN PRICE MOVEMENT & VOLUME")
    print("=" * 80)
    df["Abs_Return"] = df["Return"].abs()
    valid_sub = df.dropna(subset=["Abs_Return"]).copy()
    valid_sub = valid_sub[valid_sub["Volume"] > 0]
    valid_sub["Log_Volume"] = np.log10(valid_sub["Volume"])

    pearson_corr = valid_sub["Abs_Return"].corr(valid_sub["Log_Volume"], method="pearson")
    spearman_corr = valid_sub["Abs_Return"].corr(valid_sub["Log_Volume"], method="spearman")
    print(f"Pearson correlation (|Return| vs Log10 Volume):  {pearson_corr:+.4f}")
    print(f"Spearman rank correlation (|Return| vs Volume):  {spearman_corr:+.4f}")

    # Chart 5: Hexbin / Binned relationship between Abs Return and Volume
    # Subsample 100k points for clean, fast rendering
    sample_sub = valid_sub.sample(n=min(100_000, len(valid_sub)), random_state=42)
    plt.figure(figsize=(9, 5))
    plt.hexbin(sample_sub["Log_Volume"], sample_sub["Abs_Return"] * 100, gridsize=50, cmap="viridis", mincnt=1, yscale="linear")
    plt.colorbar(label="Sample Density")
    plt.title("Price Volatility (|Daily Return %|) vs Log10 Trading Volume", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Log10(Daily Volume)", fontsize=11)
    plt.ylabel("Absolute Daily Return (%)", fontsize=11)
    plt.ylim(0, 25)
    plt.tight_layout()
    chart5_path = results_dir / "eda_price_movement_vs_volume.png"
    plt.savefig(chart5_path)
    plt.close()
    print(f"Saved: {chart5_path}")

    # 10. Extreme values / Outliers
    print("\n" + "=" * 80)
    print("10. EXTREME VALUES & OUTLIER AUDIT (WITHOUT REMOVAL)")
    print("=" * 80)
    top_pos_returns = df.nlargest(10, "Return")[["Ticker", "Date", "Open", "Close", "Return", "Volume"]].copy()
    top_pos_returns["Date"] = top_pos_returns["Date"].dt.strftime("%Y-%m-%d")
    top_pos_returns["Return_%"] = (top_pos_returns["Return"] * 100).round(2)
    print("Top 10 Highest Single-Day Returns:")
    print(top_pos_returns[["Ticker", "Date", "Open", "Close", "Return_%", "Volume"]].to_string(index=False))

    top_neg_returns = df.nsmallest(10, "Return")[["Ticker", "Date", "Open", "Close", "Return", "Volume"]].copy()
    top_neg_returns["Date"] = top_neg_returns["Date"].dt.strftime("%Y-%m-%d")
    top_neg_returns["Return_%"] = (top_neg_returns["Return"] * 100).round(2)
    print("\nTop 10 Lowest Single-Day Returns (Drops):")
    print(top_neg_returns[["Ticker", "Date", "Open", "Close", "Return_%", "Volume"]].to_string(index=False))

    top_volume_days = df.nlargest(10, "Volume")[["Ticker", "Date", "Close", "Return", "Volume"]].copy()
    top_volume_days["Date"] = top_volume_days["Date"].dt.strftime("%Y-%m-%d")
    top_volume_days["Return_%"] = (top_volume_days["Return"] * 100).round(2)
    print("\nTop 10 Highest Daily Volume Occurrences:")
    print(top_volume_days[["Ticker", "Date", "Close", "Return_%", "Volume"]].to_string(index=False))

    print("\nEDA script completed successfully. All figures saved in results/.")

    return {
        "total_rows": total_rows,
        "num_tickers": num_tickers,
        "mean_ret": mean_ret,
        "std_ret": std_ret,
        "skew_ret": skew_ret,
        "kurt_ret": kurt_ret,
        "pearson_corr": pearson_corr,
        "spearman_corr": spearman_corr
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    cleaned_parquet = project_root / "data" / "processed" / "SP500_Cleaned_Data.parquet"
    results_path = project_root / "results"
    run_eda(cleaned_parquet, results_path)
