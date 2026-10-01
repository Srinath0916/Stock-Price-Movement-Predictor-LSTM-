"""
Dataset Inspection Script for S&P 500 Historical Data.
Inspects raw data without making any modifications, cleanings, or transformations.
"""

from pathlib import Path
import pandas as pd


def inspect_dataset(csv_path: Path) -> None:
    print(f"Loading dataset from: {csv_path} ...\n")
    df = pd.read_csv(csv_path)

    print("=" * 80)
    print("1. SHAPE (ROWS & COLUMNS)")
    print("=" * 80)
    print(f"Total Rows:    {df.shape[0]:,}")
    print(f"Total Columns: {df.shape[1]}")

    print("\n" + "=" * 80)
    print("2. COLUMN NAMES & DATA TYPES")
    print("=" * 80)
    col_info = pd.DataFrame({
        "Column": df.columns,
        "Dtype": df.dtypes.values,
        "Non-Null Count": df.notnull().sum().values,
        "Null Count": df.isnull().sum().values,
        "Null %": (df.isnull().sum().values / len(df) * 100).round(4)
    })
    print(col_info.to_string(index=False))

    print("\n" + "=" * 80)
    print("3. FIRST 5 ROWS")
    print("=" * 80)
    print(df.head())

    print("\n" + "=" * 80)
    print("4. LAST 5 ROWS")
    print("=" * 80)
    print(df.tail())

    print("\n" + "=" * 80)
    print("5. DUPLICATE CHECKS")
    print("=" * 80)
    exact_duplicates = df.duplicated().sum()
    print(f"Exact duplicate rows across all columns: {exact_duplicates:,}")
    if "Ticker" in df.columns and "Date" in df.columns:
        ticker_date_duplicates = df.duplicated(subset=["Ticker", "Date"]).sum()
        print(f"Duplicate (Ticker, Date) pairs:          {ticker_date_duplicates:,}")

    print("\n" + "=" * 80)
    print("6. TICKER / STOCK COVERAGE")
    print("=" * 80)
    if "Ticker" in df.columns:
        unique_tickers = df["Ticker"].dropna().unique()
        print(f"Unique Tickers Count: {len(unique_tickers)}")
        print(f"Sample Tickers (first 10 sorted): {sorted(unique_tickers)[:10]}")
        ticker_counts = df["Ticker"].value_counts()
        print(f"Max rows for single ticker:  {ticker_counts.max()} ({ticker_counts.idxmax()})")
        print(f"Min rows for single ticker:  {ticker_counts.min()} ({ticker_counts.idxmin()})")
        print(f"Average rows per ticker:     {ticker_counts.mean():.1f}")
        print(f"Median rows per ticker:      {ticker_counts.median():.1f}")

    print("\n" + "=" * 80)
    print("7. DATE RANGE & TEMPORAL COVERAGE")
    print("=" * 80)
    if "Date" in df.columns:
        dates_as_dt = pd.to_datetime(df["Date"], errors="coerce")
        invalid_dates = dates_as_dt.isnull().sum()
        print(f"Unparseable date entries: {invalid_dates}")
        print(f"Minimum Date:             {dates_as_dt.min().strftime('%Y-%m-%d')}")
        print(f"Maximum Date:             {dates_as_dt.max().strftime('%Y-%m-%d')}")
        print(f"Total Date Span:          {(dates_as_dt.max() - dates_as_dt.min()).days:,} days")

    print("\n" + "=" * 80)
    print("8. NUMERICAL SUMMARY STATISTICS")
    print("=" * 80)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    print(df[numeric_cols].describe().T.to_string())

    print("\n" + "=" * 80)
    print("9. ANOMALY & SUSPICIOUS VALUE AUDIT")
    print("=" * 80)
    price_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close"] if c in df.columns]

    # Check non-positive prices
    for col in price_cols:
        neg_count = (df[col] < 0).sum()
        zero_count = (df[col] == 0).sum()
        print(f"Negative values in '{col}': {neg_count:,}")
        print(f"Zero values in '{col}':     {zero_count:,}")

    # Check volume
    if "Volume" in df.columns:
        neg_vol = (df["Volume"] < 0).sum()
        zero_vol = (df["Volume"] == 0).sum()
        print(f"Negative Volume rows:     {neg_vol:,}")
        print(f"Zero Volume rows:         {zero_vol:,}")

    # Check logical price ordering constraints: High >= Low, High >= Open, High >= Close, Low <= Open, Low <= Close
    if all(col in df.columns for col in ["Open", "High", "Low", "Close"]):
        high_lt_low = (df["High"] < df["Low"]).sum()
        high_lt_open = (df["High"] < df["Open"]).sum()
        high_lt_close = (df["High"] < df["Close"]).sum()
        low_gt_open = (df["Low"] > df["Open"]).sum()
        low_gt_close = (df["Low"] > df["Close"]).sum()

        print(f"Rows where High < Low:    {high_lt_low:,}")
        print(f"Rows where High < Open:   {high_lt_open:,}")
        print(f"Rows where High < Close:  {high_lt_close:,}")
        print(f"Rows where Low > Open:    {low_gt_open:,}")
        print(f"Rows where Low > Close:   {low_gt_close:,}")

        if low_gt_open > 0:
            print("\nDetails of rows where Low > Open:")
            print(df[df["Low"] > df["Open"]][["Ticker", "Date", "Open", "High", "Low", "Close"]])

    if "Close" in df.columns and "Adj Close" in df.columns:
        diff_close = (df["Close"] != df["Adj Close"]).sum()
        print(f"\nRows where Close != Adj Close: {diff_close:,} (Columns are {'identical' if diff_close == 0 else 'different'})")

    print("\nInspection complete.")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    raw_csv = project_root / "data" / "raw" / "SP500_Historical_Data.csv"
    if not raw_csv.exists():
        raise FileNotFoundError(f"Dataset not found at expected path: {raw_csv}")
    inspect_dataset(raw_csv)
