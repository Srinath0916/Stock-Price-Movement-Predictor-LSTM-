"""
Target Creation Module for S&P 500 Stock Price Movement Predictor.

Computes a binary next-day price movement target independently per Ticker:
    Target_{t} = 1.0 if Close_{t+1} > Close_{t}
    Target_{t} = 0.0 if Close_{t+1} <= Close_{t}
    Target_{t} = NaN if Close_{t+1} is missing (final historical session of each ticker)

Guarantees:
- Strict group-level independence: shift(-1) is performed solely within ticker groups.
- Ticker boundaries are fully respected: the next row of ticker A never becomes the target for ticker B.
- Exactly one NaN target per ticker (at the latest available date).
- Target is purely an evaluation label and not used as an input feature.
- No scaling, sequencing, or train/validation splitting is performed here.
- Input files (SP500_Cleaned_Data.parquet, SP500_Featured_Data.parquet) remain untouched.
"""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def create_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the binary next-day direction target per ticker.

    Parameters
    ----------
    df : pd.DataFrame
        Featured dataset containing at least ['Ticker', 'Date', 'Close']
        along with all engineered features.

    Returns
    -------
    pd.DataFrame
        Dataset containing all original features plus the 'Target' column.
    """
    required_cols = ["Ticker", "Date", "Close"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Input dataframe missing required column: '{col}'")

    # 1. Sort strictly by Ticker and Date
    targeted_df = df.copy()
    targeted_df["Date"] = pd.to_datetime(targeted_df["Date"])
    targeted_df = targeted_df.sort_values(by=["Ticker", "Date"], ascending=[True, True]).reset_index(drop=True)

    # 2. Next-day close strictly per ticker (no boundary bleed)
    next_close = targeted_df.groupby("Ticker", sort=False)["Close"].shift(-1)

    # 3. Binary target formulation
    # Target = 1 if Close_{t+1} > Close_t
    # Target = 0 if Close_{t+1} <= Close_t
    # Target = NaN if Close_{t+1} is NaN (final row of ticker)
    targeted_df["Target"] = np.where(
        next_close.isna(),
        np.nan,
        np.where(next_close > targeted_df["Close"], 1.0, 0.0),
    )

    return targeted_df


def validate_target_integrity(df: pd.DataFrame, num_spot_checks: int = 5, random_seed: int = 42) -> Dict:
    """
    Performs comprehensive verification on the generated Target column:
    1. Confirms exactly one missing target per ticker (at the final row).
    2. Confirms Target values are strictly subset of {0.0, 1.0, NaN}.
    3. Confirms boundary rows have Target == NaN and interior rows have non-NaN Target.
    4. Randomly spot checks records against the subsequent row's close price.
    5. Audits missing values across all columns.
    """
    print("\n--- Validating Target Integrity & Boundary Isolation ---")

    # 1. Unique values check
    unique_vals = set(df["Target"].dropna().unique())
    if not unique_vals.issubset({0.0, 1.0}):
        raise AssertionError(f"Unexpected values in Target column: {unique_vals}")
    print(f"✓ Check 1: Target contains strictly valid values: {sorted(list(unique_vals))} and NaN.")

    # 2. Ticker boundary isolation check
    # The last row of each ticker must have Target == NaN
    ticker_change_indices = df[df["Ticker"] != df["Ticker"].shift(-1)].index
    num_tickers = df["Ticker"].nunique()

    if len(ticker_change_indices) != num_tickers:
        raise AssertionError(f"Boundary count ({len(ticker_change_indices)}) != Ticker count ({num_tickers})!")

    boundary_nans = df.loc[ticker_change_indices, "Target"].isna()
    if not boundary_nans.all():
        failed_count = (~boundary_nans).sum()
        raise AssertionError(f"Boundary failure: {failed_count} ticker final rows have non-NaN targets (cross-ticker bleed)!")
    print(f"✓ Check 2: All {num_tickers} ticker boundaries have Target == NaN (zero cross-ticker leakage).")

    # Interior rows check (every row except the last row of each ticker must have a valid target)
    interior_indices = df.index.difference(ticker_change_indices)
    interior_nans = df.loc[interior_indices, "Target"].isna().sum()
    if interior_nans != 0:
        raise AssertionError(f"Interior rows have unexpected {interior_nans} NaN targets!")
    print(f"✓ Check 3: All {len(interior_indices):,} interior rows have non-NaN targets (exactly 1 NaN per ticker).")

    # Groupby check: verify exactly 1 NaN per ticker
    nans_per_ticker = df.groupby("Ticker")["Target"].apply(lambda s: s.isna().sum())
    assert (nans_per_ticker == 1).all(), "Some tickers have != 1 missing target!"
    print("✓ Check 4: Exactly one missing target per ticker confirmed across all 472 tickers.")

    # 3. Random manual spot-checks
    np.random.seed(random_seed)
    sampled_indices = np.random.choice(interior_indices, size=num_spot_checks, replace=False)
    spot_check_results = []

    print(f"\n--- Random Manual Spot-Checks ({num_spot_checks} Samples) ---")
    for idx in sampled_indices:
        curr_row = df.iloc[idx]
        next_row = df.iloc[idx + 1]

        # Verify consecutive rows belong to the same ticker
        assert curr_row["Ticker"] == next_row["Ticker"], f"Spot check index {idx} crosses ticker boundary!"

        expected_target = 1.0 if next_row["Close"] > curr_row["Close"] else 0.0
        actual_target = curr_row["Target"]

        assert actual_target == expected_target, f"Target mismatch at index {idx}!"

        diff = next_row["Close"] - curr_row["Close"]
        direction_str = "UP (+)" if diff > 0 else ("DOWN (-)" if diff < 0 else "FLAT (=)")
        sample_info = {
            "index": int(idx),
            "ticker": curr_row["Ticker"],
            "date": curr_row["Date"].strftime("%Y-%m-%d"),
            "close": float(curr_row["Close"]),
            "next_date": next_row["Date"].strftime("%Y-%m-%d"),
            "next_close": float(next_row["Close"]),
            "diff": float(diff),
            "direction": direction_str,
            "target": float(actual_target),
        }
        spot_check_results.append(sample_info)
        print(
            f"  [{curr_row['Ticker']:5s} | {sample_info['date']}] Close: {sample_info['close']:8.2f} "
            f"-> Next ({sample_info['next_date']}): {sample_info['next_close']:8.2f} "
            f"| Δ: {diff:+8.2f} ({direction_str:8s}) => Target: {int(actual_target)} [MATCH]"
        )

    # 4. Class distribution metrics
    valid_targets = df["Target"].dropna()
    total_valid = len(valid_targets)
    count_up = int((valid_targets == 1.0).sum())
    count_down_flat = int((valid_targets == 0.0).sum())
    pct_up = (count_up / total_valid) * 100
    pct_down_flat = (count_down_flat / total_valid) * 100

    metrics = {
        "total_rows": len(df),
        "total_valid_targets": total_valid,
        "total_missing_targets": int(df["Target"].isna().sum()),
        "target_1_count": count_up,
        "target_0_count": count_down_flat,
        "target_1_pct": pct_up,
        "target_0_pct": pct_down_flat,
        "spot_checks": spot_check_results,
    }

    return metrics


def build_and_save_target(input_parquet: Path, output_parquet: Path) -> pd.DataFrame:
    """
    Main driver pipeline for target generation.
    Loads featured data, generates target, performs validations, and exports Parquet.
    """
    print("=" * 80)
    print("TARGET CREATION PIPELINE - START")
    print("=" * 80)
    print(f"Reading featured dataset from: {input_parquet}")
    df_raw = pd.read_parquet(input_parquet)
    rows_before = len(df_raw)
    cols_before = list(df_raw.columns)
    print(f"Initial row count:    {rows_before:,}")
    print(f"Initial columns ({len(cols_before)}): {cols_before}")

    print("\nGenerating binary next-day target (Target = 1 if Close_{t+1} > Close_t else 0)...")
    targeted_df = create_binary_target(df_raw)
    rows_after = len(targeted_df)
    cols_after = list(targeted_df.columns)

    # Validation
    metrics = validate_target_integrity(targeted_df, num_spot_checks=6)

    # Report class distribution
    print("\n" + "=" * 80)
    print("TARGET CLASS DISTRIBUTION & BALANCE")
    print("=" * 80)
    print(f"Target = 1 (UP / Close_{{t+1}} > Close_t):        {metrics['target_1_count']:12,d} ({metrics['target_1_pct']:.4f}%)")
    print(f"Target = 0 (DOWN or FLAT / Close_{{t+1}} <= Close_t): {metrics['target_0_count']:12,d} ({metrics['target_0_pct']:.4f}%)")
    print(f"Target = NaN (Final session per ticker):       {metrics['total_missing_targets']:12,d} ({metrics['total_missing_targets'] / rows_after * 100:.4f}%)")
    print(f"Total rows:                                    {rows_after:12,d}")

    # Missing value audit
    print("\n" + "=" * 80)
    print("MISSING VALUE AUDIT ACROSS ALL 16 COLUMNS")
    print("=" * 80)
    null_series = targeted_df.isnull().sum()
    for col, n_miss in null_series.items():
        pct_miss = (n_miss / rows_after) * 100
        print(f"  {col:24s}: {n_miss:10,d} missing ({pct_miss:7.4f}%)")

    # Confirm preservation of all rows and prior features
    assert rows_before == rows_after, f"Row count changed: {rows_before} vs {rows_after}"
    for col in cols_before:
        assert col in cols_after, f"Column '{col}' was dropped!"

    # Save to parquet
    print("\n" + "=" * 80)
    print(f"SAVING TARGETED DATASET TO: {output_parquet}")
    print("=" * 80)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    targeted_df.to_parquet(output_parquet, engine="pyarrow", index=False)
    file_size_mb = output_parquet.stat().st_size / (1024 * 1024)
    print(f"✓ Saved successfully! File size: {file_size_mb:.2f} MB")

    # Verify reload
    reloaded_df = pd.read_parquet(output_parquet, engine="pyarrow")
    assert len(reloaded_df) == rows_after, "Reloaded row count mismatch!"
    assert list(reloaded_df.columns) == cols_after, "Reloaded column mismatch!"
    print(f"✓ Reload verified: {len(reloaded_df):,} rows and {len(reloaded_df.columns)} columns intact.")
    print("=" * 80)
    print("TARGET CREATION PIPELINE - COMPLETE")
    print("=" * 80)

    return targeted_df


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "SP500_Featured_Data.parquet"
    OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "SP500_Targeted_Data.parquet"

    build_and_save_target(input_parquet=INPUT_PATH, output_parquet=OUTPUT_PATH)
