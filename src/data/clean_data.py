"""
Data Cleaning Pipeline for S&P 500 Historical Data.
Applies minimal, justified cleaning:
- Date string to datetime
- Sort chronologically by (Ticker, Date)
- Drop redundant identical 'Adj Close'
- Keep zero-volume rows intact
- Keep HUBB 2021-05-05 intact
- Save output as Parquet
"""

from pathlib import Path
import pandas as pd


def clean_sp500_data(raw_csv_path: Path, output_parquet_path: Path) -> dict:
    print(f"Loading raw dataset from {raw_csv_path}...")
    df_raw = pd.read_csv(raw_csv_path)

    # Initial stats
    rows_before = len(df_raw)
    cols_before = list(df_raw.columns)

    print("Executing minimal cleaning pipeline...")

    # 1. Convert Date to pandas datetime
    df = df_raw.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="raise")

    # 2. Sort by Ticker and Date in ascending order
    df = df.sort_values(by=["Ticker", "Date"], ascending=[True, True])

    # 3. Reset the index
    df = df.reset_index(drop=True)

    # 4. Verify duplicate Ticker-Date pairs
    dup_ticker_date = df.duplicated(subset=["Ticker", "Date"]).sum()
    assert dup_ticker_date == 0, f"Found {dup_ticker_date} duplicate (Ticker, Date) pairs!"

    # 5. Verify Close and Adj Close identity, then remove Adj Close
    diff_close_adj = (df["Close"] != df["Adj Close"]).sum()
    assert diff_close_adj == 0, "Adj Close and Close are not identical!"
    df = df.drop(columns=["Adj Close"])

    # 6. Verify zero-volume rows are preserved
    zero_vol_before = (df_raw["Volume"] == 0).sum()
    zero_vol_after = (df["Volume"] == 0).sum()
    assert zero_vol_before == zero_vol_after, f"Zero-volume mismatch: {zero_vol_before} vs {zero_vol_after}"

    # 7. Verify HUBB 2021-05-05 is unchanged
    hubb_row = df[(df["Ticker"] == "HUBB") & (df["Date"] == "2021-05-05")]
    assert len(hubb_row) == 1, "HUBB 2021-05-05 row not found!"
    assert hubb_row.iloc[0]["Low"] == 181.55, "HUBB 2021-05-05 Low price was modified!"
    assert hubb_row.iloc[0]["Open"] == 181.34, "HUBB 2021-05-05 Open price was modified!"

    # Final stats
    rows_after = len(df)
    cols_after = list(df.columns)
    missing_values_after = int(df.isnull().sum().sum())
    final_ticker_count = int(df["Ticker"].nunique())
    final_min_date = df["Date"].min().strftime("%Y-%m-%d")
    final_max_date = df["Date"].max().strftime("%Y-%m-%d")

    # Ensure output directory exists
    output_parquet_path.parent.mkdir(parents=True, exist_ok=True)

    # 10. Save to parquet
    print(f"Saving cleaned dataset to {output_parquet_path}...")
    df.to_parquet(output_parquet_path, engine="pyarrow", index=False)

    # Verify reload
    df_reloaded = pd.read_parquet(output_parquet_path, engine="pyarrow")
    assert len(df_reloaded) == rows_after, "Reloaded parquet row count mismatch!"

    print("\n" + "=" * 80)
    print("CLEANING PIPELINE COMPLETE - AUDIT REPORT")
    print("=" * 80)
    print(f"Rows before:                      {rows_before:,}")
    print(f"Rows after:                       {rows_after:,} (no rows deleted)")
    print(f"Columns before:                   {len(cols_before)} -> {cols_before}")
    print(f"Columns after:                    {len(cols_after)} -> {cols_after}")
    print(f"Missing values:                   {missing_values_after}")
    print(f"Duplicate (Ticker, Date) count:   {dup_ticker_date}")
    print(f"Zero-volume count:                {zero_vol_after:,} (preserved unchanged)")
    print(f"Final ticker count:               {final_ticker_count}")
    print(f"Final date range:                 {final_min_date} to {final_max_date}")
    print(f"Confirmation HUBB 2021-05-05:     Unchanged (Open={hubb_row.iloc[0]['Open']}, Low={hubb_row.iloc[0]['Low']})")
    print(f"Confirmation Adj Close removal:   Removed because it was 100% identical to Close across all {rows_after:,} rows")
    print(f"File saved:                       {output_parquet_path} ({output_parquet_path.stat().st_size / (1024 * 1024):.2f} MB)")
    print("=" * 80)

    return {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "cols_before": cols_before,
        "cols_after": cols_after,
        "missing_values": missing_values_after,
        "duplicate_count": dup_ticker_date,
        "zero_volume_count": zero_vol_after,
        "final_ticker_count": final_ticker_count,
        "final_date_range": f"{final_min_date} to {final_max_date}",
        "hubb_confirmed": True,
        "adj_close_removed": True,
        "output_path": str(output_parquet_path)
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    raw_csv = project_root / "data" / "raw" / "SP500_Historical_Data.csv"
    output_parquet = project_root / "data" / "processed" / "SP500_Cleaned_Data.parquet"
    clean_sp500_data(raw_csv, output_parquet)
