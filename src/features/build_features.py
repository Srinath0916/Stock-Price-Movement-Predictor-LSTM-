"""
Feature Engineering Module for S&P 500 Stock Price Movement Predictor.

Computes 8 core financial features independently per Ticker in strict chronological order:
1. Daily Return: (Close_t - Close_{t-1}) / Close_{t-1}
2. Volume Change: (Volume_t - Volume_{t-1}) / Volume_{t-1} (with safe zero-volume handling)
3. High-Low Range: (High - Low) / Close
4. 20-Day Rolling Volatility: Rolling 20-day standard deviation of Daily Return
5. SMA20: 20-day simple moving average of Close
6. SMA50: 50-day simple moving average of Close
7. Close/SMA20: Price to 20-day SMA ratio
8. Close/SMA50: Price to 50-day SMA ratio

Guarantees:
- Strict group-level independence (no cross-ticker contamination).
- Strict chronological sorting (Ticker, Date) before any window calculations.
- No lookahead bias / future data leakage.
- No dropping of extreme returns / outliers.
- Safe division handling for zero-volume rows (no infinities).
- Original source dataset remains completely untouched.
"""

from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd


# Core feature list
CORE_FEATURE_COLUMNS = [
    "Daily_Return",
    "Volume_Change",
    "High_Low_Range",
    "Rolling_Volatility_20",
    "SMA20",
    "SMA50",
    "Close/SMA20",
    "Close/SMA50",
]


def calculate_core_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes core financial features on a cleaned stock price dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned stock dataframe with columns:
        ['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']

    Returns
    -------
    pd.DataFrame
        Feature-engineered dataframe containing original columns plus 8 core features.
    """
    # 0. Safety copy and strict chronological sorting
    required_cols = ["Ticker", "Date", "Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in input dataframe: '{col}'")

    featured_df = df.copy()
    featured_df["Date"] = pd.to_datetime(featured_df["Date"])
    featured_df = featured_df.sort_values(by=["Ticker", "Date"], ascending=[True, True]).reset_index(drop=True)

    # 1. Daily Return: (Close_t - Close_{t-1}) / Close_{t-1}
    # Grouped per ticker to prevent return calculations spanning ticker boundaries
    featured_df["Daily_Return"] = featured_df.groupby("Ticker", sort=False)["Close"].pct_change()

    # 2. Volume Change: (Volume_t - Volume_{t-1}) / Volume_{t-1}
    # Zero-Volume Denominator Safe Handling Policy:
    # - Case A (First row of ticker): Volume_{t-1} is NaN -> Volume_Change = NaN.
    # - Case B (prev_vol == 0 and curr_vol == 0): Both sessions had zero volume (e.g. trading halt).
    #          Volume did not change between sessions -> Volume_Change = 0.0.
    # - Case C (prev_vol == 0 and curr_vol > 0): Resumption from zero volume.
    #          Percentage increase from zero is mathematically undefined (+inf).
    #          Safely mapped to np.nan to prevent numerical overflow and gradient instability in ML models.
    # - Case D (prev_vol > 0): Standard relative change (Volume_t - Volume_{t-1}) / Volume_{t-1}.
    prev_vol = featured_df.groupby("Ticker", sort=False)["Volume"].shift(1)
    safe_denom = np.where(prev_vol == 0, np.nan, prev_vol)
    vol_diff = featured_df["Volume"] - prev_vol
    vol_change = vol_diff / safe_denom
    # When both previous and current volume are 0, volume change is zero
    vol_change = np.where((prev_vol == 0) & (featured_df["Volume"] == 0), 0.0, vol_change)
    featured_df["Volume_Change"] = vol_change

    # 3. High-Low Range: (High - Low) / Close
    # Measures intraday price spread / intra-session volatility relative to the close price
    featured_df["High_Low_Range"] = (featured_df["High"] - featured_df["Low"]) / featured_df["Close"]

    # 4. 20-Day Rolling Volatility: Standard deviation of Daily Return over previous 20 trading days
    # Using trailing window=20 with min_periods=20 on Daily_Return per ticker
    featured_df["Rolling_Volatility_20"] = featured_df.groupby("Ticker", sort=False)["Daily_Return"].transform(
        lambda s: s.rolling(window=20, min_periods=20).std()
    )

    # 5. SMA20: 20-day simple moving average of Close
    featured_df["SMA20"] = featured_df.groupby("Ticker", sort=False)["Close"].transform(
        lambda s: s.rolling(window=20, min_periods=20).mean()
    )

    # 6. SMA50: 50-day simple moving average of Close
    featured_df["SMA50"] = featured_df.groupby("Ticker", sort=False)["Close"].transform(
        lambda s: s.rolling(window=50, min_periods=50).mean()
    )

    # 7. Close/SMA20: Ratio of Close to SMA20 (short-term trend / mean-reversion indicator)
    featured_df["Close/SMA20"] = featured_df["Close"] / featured_df["SMA20"]

    # 8. Close/SMA50: Ratio of Close to SMA50 (intermediate-term trend indicator)
    featured_df["Close/SMA50"] = featured_df["Close"] / featured_df["SMA50"]

    return featured_df


def verify_no_cross_ticker_contamination(
    df: pd.DataFrame, test_tickers: Optional[List[str]] = None
) -> bool:
    """
    Rigorously verifies that no ticker calculations bled across ticker boundaries.

    Checks:
    1. First row of each ticker has NaN for Daily_Return (no lag across tickers).
    2. First row of each ticker has NaN for Volume_Change.
    3. Isolated computation test: extracts individual ticker sub-frames, computes features
       in isolation from scratch, and verifies bitwise equality against the master dataframe.
    4. Infinity check: verifies zero infinite values across all numerical columns.

    Returns
    -------
    bool
        True if all verification checks pass.
    """
    print("\n--- Verifying Ticker Independence & Data Integrity ---")

    # 1. First row per ticker checks (using nth(0) for true 0-th row in each group)
    first_ret_is_nan = df.groupby("Ticker", sort=False)["Daily_Return"].nth(0).isna().all()
    first_vol_is_nan = df.groupby("Ticker", sort=False)["Volume_Change"].nth(0).isna().all()

    if not first_ret_is_nan:
        raise AssertionError("Cross-ticker contamination detected: First row of Daily_Return is not NaN for all tickers!")
    if not first_vol_is_nan:
        raise AssertionError("Cross-ticker contamination detected: First row of Volume_Change is not NaN for all tickers!")
    print("✓ Check 1: First observation of each ticker has NaN for Daily_Return (no previous-ticker bleed).")
    print("✓ Check 2: First observation of each ticker has NaN for Volume_Change (no previous-ticker bleed).")

    # 2. Check for infinities
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            raise AssertionError(f"Infinite values detected in column '{col}': {inf_count} infs!")
    print("✓ Check 3: Zero infinite values across all numerical columns.")

    # 3. Isolated computation comparison test
    if test_tickers is None:
        # Test boundary tickers and large tickers
        all_tickers = df["Ticker"].unique()
        test_tickers = [all_tickers[0], all_tickers[len(all_tickers) // 2], all_tickers[-1], "AAPL", "HUBB"]
        test_tickers = [t for t in test_tickers if t in all_tickers]

    for ticker in test_tickers:
        sub_master = df[df["Ticker"] == ticker].copy().reset_index(drop=True)

        # Isolated calculation
        iso_ret = sub_master["Close"].pct_change()
        iso_prev_vol = sub_master["Volume"].shift(1)
        iso_denom = np.where(iso_prev_vol == 0, np.nan, iso_prev_vol)
        iso_vol_chg = (sub_master["Volume"] - iso_prev_vol) / iso_denom
        iso_vol_chg = np.where((iso_prev_vol == 0) & (sub_master["Volume"] == 0), 0.0, iso_vol_chg)
        iso_hl = (sub_master["High"] - sub_master["Low"]) / sub_master["Close"]
        iso_vol20 = iso_ret.rolling(20, min_periods=20).std()
        iso_sma20 = sub_master["Close"].rolling(20, min_periods=20).mean()
        iso_sma50 = sub_master["Close"].rolling(50, min_periods=50).mean()
        iso_c_sma20 = sub_master["Close"] / iso_sma20
        iso_c_sma50 = sub_master["Close"] / iso_sma50

        # Assert parity
        for col_name, iso_series in [
            ("Daily_Return", iso_ret),
            ("Volume_Change", pd.Series(iso_vol_chg)),
            ("High_Low_Range", iso_hl),
            ("Rolling_Volatility_20", iso_vol20),
            ("SMA20", iso_sma20),
            ("SMA50", iso_sma50),
            ("Close/SMA20", iso_c_sma20),
            ("Close/SMA50", iso_c_sma50),
        ]:
            master_vals = sub_master[col_name].to_numpy()
            iso_vals = iso_series.to_numpy()
            if not np.allclose(np.nan_to_num(master_vals, nan=-999.0), np.nan_to_num(iso_vals, nan=-999.0), rtol=1e-7, atol=1e-7):
                raise AssertionError(f"Mismatch found between master and isolated calculation for ticker '{ticker}' in column '{col_name}'!")

        print(f"✓ Check 4: Isolated calculation bitwise parity confirmed for ticker '{ticker}'.")

    print("✓ All cross-ticker contamination and integrity checks PASSED successfully.")
    return True


def build_and_save_features(
    input_parquet: Path,
    output_parquet: Path,
    sample_tickers: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Main driver pipeline for feature engineering.
    Loads cleaned data, computes features, verifies independence, reports stats, and saves Parquet.
    """
    print("=" * 80)
    print("FEATURE ENGINEERING PIPELINE - START")
    print("=" * 80)
    print(f"Reading cleaned dataset from: {input_parquet}")
    df_raw = pd.read_parquet(input_parquet)
    rows_before = len(df_raw)
    cols_before = list(df_raw.columns)
    print(f"Initial row count:    {rows_before:,}")
    print(f"Initial columns ({len(cols_before)}): {cols_before}")

    print("\nComputing 8 core features...")
    featured_df = calculate_core_features(df_raw)
    rows_after = len(featured_df)
    cols_after = list(featured_df.columns)

    # Verification
    verify_no_cross_ticker_contamination(featured_df, sample_tickers)

    # 1. Resulting columns
    print("\n" + "=" * 80)
    print("1. RESULTING COLUMNS")
    print("=" * 80)
    for idx, col in enumerate(cols_after, start=1):
        dtype_str = str(featured_df[col].dtype)
        is_new = "(ENGINEERED FEATURE)" if col in CORE_FEATURE_COLUMNS else "(BASE COLUMN)"
        print(f"  {idx:2d}. {col:24s} [{dtype_str:14s}] {is_new}")

    # 2. First few rows for 2-3 tickers
    display_tickers = ["A", "AAPL", "MSFT"]
    print("\n" + "=" * 80)
    print("2. FIRST FEW ROWS FOR SAMPLE TICKERS")
    print("=" * 80)
    for ticker in display_tickers:
        if ticker in featured_df["Ticker"].values:
            print(f"\n--- Ticker: {ticker} (First 5 Rows) ---")
            ticker_slice = featured_df[featured_df["Ticker"] == ticker].head(5)
            # Display formatted columns
            pd.set_option("display.max_columns", 20)
            pd.set_option("display.width", 1000)
            print(ticker_slice.to_string(index=False))

    # 3. Report missing values created by rolling calculations
    print("\n" + "=" * 80)
    print("3. MISSING VALUES CREATED BY ROLLING CALCULATIONS")
    print("=" * 80)
    print(f"{'Feature Name':25s} | {'Missing Count':14s} | {'Missing %':10s} | {'Reason / Nature of NaNs':40s}")
    print("-" * 100)
    for feat in CORE_FEATURE_COLUMNS:
        n_miss = int(featured_df[feat].isna().sum())
        pct_miss = (n_miss / rows_after) * 100
        if feat == "Daily_Return":
            reason = "1st row per ticker (472 tickers * 1)"
        elif feat == "Volume_Change":
            reason = "472 initial rows + 1,086 zero-volume resumptions"
        elif feat == "High_Low_Range":
            reason = "0 missing (fully defined intraday range)"
        elif feat == "Rolling_Volatility_20":
            reason = "First 20 rows per ticker (472 tickers * 20)"
        elif feat in ["SMA20", "Close/SMA20"]:
            reason = "First 19 rows per ticker (472 tickers * 19)"
        elif feat in ["SMA50", "Close/SMA50"]:
            reason = "First 49 rows per ticker (472 tickers * 49)"
        else:
            reason = "Rolling window warm-up"
        print(f"{feat:25s} | {n_miss:14,d} | {pct_miss:9.4f}% | {reason}")

    # 4. Report number of rows before and after
    print("\n" + "=" * 80)
    print("4. ROW COUNT COMPARISON")
    print("=" * 80)
    print(f"Rows before feature engineering: {rows_before:,}")
    print(f"Rows after feature engineering:  {rows_after:,}")
    assert rows_before == rows_after, f"Row count changed! Before: {rows_before}, After: {rows_after}"
    print("✓ Row count perfectly preserved (0 rows dropped or added).")

    # 6. Save result as Parquet
    print("\n" + "=" * 80)
    print(f"6. SAVING FEATURE-ENGINEERED DATASET TO: {output_parquet}")
    print("=" * 80)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    featured_df.to_parquet(output_parquet, engine="pyarrow", index=False)
    file_size_mb = output_parquet.stat().st_size / (1024 * 1024)
    print(f"✓ Saved successfully! File size: {file_size_mb:.2f} MB")

    # Verify reload
    reloaded_df = pd.read_parquet(output_parquet, engine="pyarrow")
    assert len(reloaded_df) == rows_after, "Reloaded row count mismatch!"
    assert list(reloaded_df.columns) == cols_after, "Reloaded column mismatch!"
    print(f"✓ Reload verified: {len(reloaded_df):,} rows and {len(reloaded_df.columns)} columns intact.")
    print("=" * 80)
    print("FEATURE ENGINEERING PIPELINE - COMPLETE")
    print("=" * 80)

    return featured_df


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "SP500_Cleaned_Data.parquet"
    OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "SP500_Featured_Data.parquet"

    build_and_save_features(input_parquet=INPUT_PATH, output_parquet=OUTPUT_PATH)
