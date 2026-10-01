"""
Data Cleaning & Quality Investigation Script.
Performs in-depth analysis on zero-volume rows, anomaly rows, date sorting,
and column comparisons without modifying the raw data.
"""

from pathlib import Path
import pandas as pd
import numpy as np


def run_cleaning_investigation(csv_path: Path):
    print("Loading raw CSV...")
    df = pd.read_csv(csv_path)

    # 1 & 2. Date conversion and sorting
    print("Converting 'Date' to datetime and sorting by ['Ticker', 'Date']...")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values(by=["Ticker", "Date"]).reset_index(drop=True)

    # 3. Missing values & Duplicates
    print("\n--- Missing Values & Duplicates ---")
    null_counts = df.isnull().sum()
    print("Null counts per column:")
    print(null_counts)

    exact_dups = df.duplicated().sum()
    ticker_date_dups = df.duplicated(subset=["Ticker", "Date"]).sum()
    print(f"Exact row duplicates: {exact_dups}")
    print(f"Duplicate (Ticker, Date) pairs: {ticker_date_dups}")

    # Monotonicity check
    is_monotonic = df.groupby("Ticker")["Date"].apply(lambda s: s.is_monotonic_increasing).all()
    print(f"Are dates strictly sorted per ticker? {is_monotonic}")

    # 4. Zero-volume rows investigation
    print("\n--- Zero-Volume Rows Investigation ---")
    zero_vol = df[df["Volume"] == 0].copy()
    num_zero_vol = len(zero_vol)
    print(f"Total zero-volume rows: {num_zero_vol} ({num_zero_vol / len(df) * 100:.4f}% of data)")

    affected_tickers = zero_vol["Ticker"].value_counts()
    print(f"Number of distinct tickers affected: {len(affected_tickers)} / {df['Ticker'].nunique()}")
    print("Top 10 tickers with most zero-volume days:")
    print(affected_tickers.head(10))

    # Date distribution of zero-volume rows
    zero_vol["Year"] = zero_vol["Date"].dt.year
    year_dist = zero_vol["Year"].value_counts().sort_index()
    print("\nZero-volume rows by year:")
    print(year_dist)

    # Check top specific dates
    top_dates = zero_vol["Date"].value_counts().head(10)
    print("\nTop 10 dates with most zero-volume tickers:")
    print(top_dates)

    # OHLC validity of zero-volume rows
    ohlc_cols = ["Open", "High", "Low", "Close"]
    zero_vol_has_zeros = (zero_vol[ohlc_cols] == 0).any().any()
    zero_vol_has_negs = (zero_vol[ohlc_cols] < 0).any().any()
    zero_vol_has_nans = zero_vol[ohlc_cols].isnull().any().any()
    print(f"\nZero-volume rows have zero prices?     {zero_vol_has_zeros}")
    print(f"Zero-volume rows have negative prices? {zero_vol_has_negs}")
    print(f"Zero-volume rows have NaN prices?      {zero_vol_has_nans}")

    # Are prices flat on zero-volume days (Open == High == Low == Close)?
    flat_prices = (
        (zero_vol["Open"] == zero_vol["High"])
        & (zero_vol["High"] == zero_vol["Low"])
        & (zero_vol["Low"] == zero_vol["Close"])
    )
    print(f"Zero-volume rows where Open == High == Low == Close (flatline/halted): {flat_prices.sum()} / {num_zero_vol} ({flat_prices.mean()*100:.2f}%)")
    print(f"Zero-volume rows where prices fluctuated (High > Low): {(zero_vol['High'] > zero_vol['Low']).sum()} / {num_zero_vol}")

    # Are zero-volume rows isolated or consecutive?
    # For tickers with zero volume, calculate consecutive streaks
    df["is_zero_vol"] = df["Volume"] == 0
    # Group consecutive streaks per ticker
    df["streak_id"] = (df["is_zero_vol"] != df["is_zero_vol"].shift(1)).cumsum()
    zero_streaks = df[df["is_zero_vol"]].groupby(["Ticker", "streak_id"]).size()
    streak_counts = zero_streaks.value_counts().sort_index()
    print("\nConsecutive zero-volume streak lengths distribution:")
    print("Streak Length (days) -> Frequency:")
    for length, freq in streak_counts.head(10).items():
        print(f"  {length} day(s): {freq} occurrences")
    if len(streak_counts) > 10:
        print(f"  ... max streak length: {streak_counts.index.max()} days ({streak_counts.loc[streak_counts.index.max()]} occurrences)")

    df.drop(columns=["is_zero_vol", "streak_id"], inplace=True)

    # 5. HUBB 2021-05-05 investigation
    print("\n--- HUBB 2021-05-05 Investigation ---")
    hubb_df = df[df["Ticker"] == "HUBB"].copy()
    hubb_target_idx = hubb_df[hubb_df["Date"] == pd.Timestamp("2021-05-05")].index

    if len(hubb_target_idx) > 0:
        idx = hubb_target_idx[0]
        # Get surrounding 5 days before and after
        start_idx = max(0, idx - 5)
        end_idx = min(len(df), idx + 6)
        surrounding = df.iloc[start_idx:end_idx][["Ticker", "Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]].copy()
        surrounding["Date"] = surrounding["Date"].dt.strftime("%Y-%m-%d")
        surrounding["Low_gt_Open"] = surrounding["Low"] > surrounding["Open"]
        print("Surrounding rows for HUBB around 2021-05-05:")
        print(surrounding.to_string(index=False))

    # 6. Close vs Adj Close comparison
    print("\n--- Close vs Adj Close Comparison ---")
    diff_mask = df["Close"] != df["Adj Close"]
    diff_count = diff_mask.sum()
    print(f"Total rows where Close != Adj Close: {diff_count}")
    if diff_count > 0:
        max_abs_diff = (df["Close"] - df["Adj Close"]).abs().max()
        print(f"Max absolute difference: {max_abs_diff}")
    else:
        print("Confirmed: Close and Adj Close are 100% numerically identical across all 2,703,531 rows.")

    # 7. Summary metrics
    print("\n--- Summary Metrics ---")
    print(f"Rows:         {len(df):,}")
    print(f"Columns:      {df.shape[1]} ({list(df.columns)})")
    print(f"Date Range:   {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Unique Tickers: {df['Ticker'].nunique()}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    raw_csv = project_root / "data" / "raw" / "SP500_Historical_Data.csv"
    run_cleaning_investigation(raw_csv)
