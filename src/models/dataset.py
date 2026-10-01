"""
Dataset, Sequence Generation, and Preprocessing Pipeline for Stock Movement Prediction.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader


MODEL_FEATURES = [
    "Daily_Return",
    "Volume_Change",
    "High_Low_Range",
    "Rolling_Volatility_20",
    "SMA20",
    "SMA50",
    "Close/SMA20",
    "Close/SMA50",
]


class TimeSeriesDataset(Dataset):
    """
    Memory-efficient PyTorch Dataset for multi-ticker time-series sequences.
    Indexes directly into a shared contiguous 2D float32 array of shape (N_rows, num_features)
    to prevent memory bloat from duplicating 30-day sliding windows.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_end_indices: np.ndarray, lookback: int):
        self.X = torch.from_numpy(X) if isinstance(X, np.ndarray) else X
        self.y = torch.from_numpy(y) if isinstance(y, np.ndarray) else y
        self.indices = torch.from_numpy(seq_end_indices) if isinstance(seq_end_indices, np.ndarray) else seq_end_indices
        self.lookback = lookback

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        r = self.indices[idx]
        # Slice trailing lookback rows up to and including r
        return self.X[r - self.lookback + 1 : r + 1], self.y[r]


class DataPipeline:
    """
    Orchestrates:
    1. Loading targeted data.
    2. Missing value removal (warm-up NaNs & final target NaNs).
    3. Chronological splitting (70% train, 15% validation, 15% test by unique dates).
    4. StandardScaler fitted strictly on training data.
    5. Sequence generation per ticker without cross-ticker contamination.
    """

    def __init__(self, data_path: Path, lookback: int = 30, scaler_save_path: Optional[Path] = None):
        self.data_path = Path(data_path)
        self.lookback = lookback
        self.scaler_save_path = Path(scaler_save_path) if scaler_save_path else None
        self.scaler = StandardScaler()
        self.split_info: Dict = {}
        self.df_clean: pd.DataFrame = None
        self.X_scaled: np.ndarray = None
        self.y_all: np.ndarray = None
        self.train_indices: np.ndarray = None
        self.val_indices: np.ndarray = None
        self.test_indices: np.ndarray = None

    def prepare_data(self) -> Dict:
        print("\n" + "=" * 80)
        print("DATA PIPELINE & CHRONOLOGICAL SEQUENCE GENERATION")
        print("=" * 80)
        print(f"Loading data from: {self.data_path}")
        df = pd.read_parquet(self.data_path)
        total_rows_raw = len(df)
        print(f"Raw dataset rows: {total_rows_raw:,}")

        # 1. Handle missing values
        print("\n--- STAGE 3: Missing Value Handling ---")
        feature_nans = df[MODEL_FEATURES].isna().any(axis=1)
        target_nans = df["Target"].isna()
        dropped_mask = feature_nans | target_nans
        dropped_count = int(dropped_mask.sum())

        print(f"Total rows dropped:                 {dropped_count:,} ({dropped_count / total_rows_raw * 100:.2f}%)")
        print(f"  - Rows with feature warm-up NaNs:  {int(feature_nans.sum()):,} (primarily SMA50 warm-up of 49 rows/ticker)")
        print(f"  - Rows with Target NaNs:          {int(target_nans.sum()):,} (final session per ticker)")
        print(f"Rows retained for sequence creation: {total_rows_raw - dropped_count:,} ({(total_rows_raw - dropped_count) / total_rows_raw * 100:.2f}%)")

        self.df_clean = df[~dropped_mask].copy().sort_values(by=["Ticker", "Date"]).reset_index(drop=True)

        # 2. Chronological data split by unique trading dates
        print("\n--- STAGE 1: Chronological Data Split (70/15/15) ---")
        dates = self.df_clean["Date"].to_numpy()
        unique_dates = np.sort(np.unique(dates))
        num_dates = len(unique_dates)

        n_train_dates = int(num_dates * 0.70)
        n_val_dates = int(num_dates * 0.15)
        n_test_dates = num_dates - n_train_dates - n_val_dates

        train_end_date = unique_dates[n_train_dates - 1]
        val_end_date = unique_dates[n_train_dates + n_val_dates - 1]
        val_start_date = unique_dates[n_train_dates]
        test_start_date = unique_dates[n_train_dates + n_val_dates]
        test_end_date = unique_dates[-1]

        train_row_mask = dates <= train_end_date
        val_row_mask = (dates > train_end_date) & (dates <= val_end_date)
        test_row_mask = dates > val_end_date

        def get_split_stats(mask: np.ndarray, split_label: str) -> Dict:
            sub = self.df_clean[mask]
            targets = sub["Target"].to_numpy()
            n_up = int((targets == 1.0).sum())
            n_down = int((targets == 0.0).sum())
            total = len(targets)
            return {
                "name": split_label,
                "rows": total,
                "tickers": int(sub["Ticker"].nunique()),
                "target_1_count": n_up,
                "target_0_count": n_down,
                "target_1_pct": (n_up / total * 100) if total > 0 else 0.0,
                "target_0_pct": (n_down / total * 100) if total > 0 else 0.0,
            }

        train_stats = get_split_stats(train_row_mask, "TRAIN")
        val_stats = get_split_stats(val_row_mask, "VALIDATION")
        test_stats = get_split_stats(test_row_mask, "TEST")

        print(f"TRAIN:      {pd.to_datetime(unique_dates[0]).strftime('%Y-%m-%d')} to {pd.to_datetime(train_end_date).strftime('%Y-%m-%d')} | {n_train_dates} dates | {train_stats['rows']:,} rows | {train_stats['tickers']} tickers | UP: {train_stats['target_1_pct']:.2f}% | DOWN: {train_stats['target_0_pct']:.2f}%")
        print(f"VALIDATION: {pd.to_datetime(val_start_date).strftime('%Y-%m-%d')} to {pd.to_datetime(val_end_date).strftime('%Y-%m-%d')} | {n_val_dates} dates | {val_stats['rows']:,} rows | {val_stats['tickers']} tickers | UP: {val_stats['target_1_pct']:.2f}% | DOWN: {val_stats['target_0_pct']:.2f}%")
        print(f"TEST:       {pd.to_datetime(test_start_date).strftime('%Y-%m-%d')} to {pd.to_datetime(test_end_date).strftime('%Y-%m-%d')} | {n_test_dates} dates | {test_stats['rows']:,} rows | {test_stats['tickers']} tickers | UP: {test_stats['target_1_pct']:.2f}% | DOWN: {test_stats['target_0_pct']:.2f}%")

        # 3. Feature Scaling strictly on training data
        print("\n--- STAGE 4: Feature Scaling (StandardScaler) ---")
        print("Fitting StandardScaler strictly on training split...")
        self.scaler.fit(self.df_clean.loc[train_row_mask, MODEL_FEATURES])
        print(f"Fitted features ({len(MODEL_FEATURES)}): {MODEL_FEATURES}")
        print(f"Scaler means: {np.round(self.scaler.mean_, 4)}")
        print(f"Scaler stds:  {np.round(self.scaler.scale_, 4)}")

        if self.scaler_save_path:
            self.scaler_save_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.scaler, self.scaler_save_path)
            print(f"✓ Saved fitted scaler to: {self.scaler_save_path}")

        self.X_scaled = self.scaler.transform(self.df_clean[MODEL_FEATURES]).astype(np.float32)
        self.y_all = self.df_clean["Target"].to_numpy().astype(np.float32)[:, None]

        # 4. Sequence Generation per Ticker (Zero Cross-Ticker Leakage)
        print(f"\n--- STAGE 5: Sequence Generation (Lookback = {self.lookback} days) ---")
        tickers = self.df_clean["Ticker"].to_numpy()

        # Vectorized ticker boundary check:
        # Sequence ending at row r is valid iff row r and row (r - lookback + 1) belong to the same ticker
        is_valid_seq = tickers[self.lookback - 1 :] == tickers[: -self.lookback + 1]
        valid_indices = np.where(is_valid_seq)[0] + (self.lookback - 1)

        seq_dates = dates[valid_indices]
        self.train_indices = valid_indices[seq_dates <= train_end_date]
        self.val_indices = valid_indices[(seq_dates > train_end_date) & (seq_dates <= val_end_date)]
        self.test_indices = valid_indices[seq_dates > val_end_date]

        # Verify no cross-ticker sequences
        for name, idxs in [("Train", self.train_indices), ("Validation", self.val_indices), ("Test", self.test_indices)]:
            start_t = tickers[idxs - self.lookback + 1]
            end_t = tickers[idxs]
            if not np.all(start_t == end_t):
                raise AssertionError(f"Cross-ticker sequence detected in {name} split!")

        print(f"Total valid sequences:      {len(valid_indices):,}")
        print(f"  - Training sequences:     {len(self.train_indices):,}")
        print(f"  - Validation sequences:   {len(self.val_indices):,}")
        print(f"  - Test sequences:         {len(self.test_indices):,}")
        print(f"Sequence input shape:       (batch_size, {self.lookback}, {len(MODEL_FEATURES)})")
        print("✓ Verified: 100% of sequences start and end within the exact same ticker (0 boundary violations).")

        self.split_info = {
            "dates": {
                "train_start": pd.to_datetime(unique_dates[0]).strftime("%Y-%m-%d"),
                "train_end": pd.to_datetime(train_end_date).strftime("%Y-%m-%d"),
                "val_start": pd.to_datetime(val_start_date).strftime("%Y-%m-%d"),
                "val_end": pd.to_datetime(val_end_date).strftime("%Y-%m-%d"),
                "test_start": pd.to_datetime(test_start_date).strftime("%Y-%m-%d"),
                "test_end": pd.to_datetime(test_end_date).strftime("%Y-%m-%d"),
            },
            "train_stats": train_stats,
            "val_stats": val_stats,
            "test_stats": test_stats,
            "dropped_rows": dropped_count,
            "total_clean_rows": len(self.df_clean),
            "lookback": self.lookback,
            "num_train_seqs": len(self.train_indices),
            "num_val_seqs": len(self.val_indices),
            "num_test_seqs": len(self.test_indices),
        }
        return self.split_info

    def get_datasets(self) -> Tuple[TimeSeriesDataset, TimeSeriesDataset, TimeSeriesDataset]:
        if self.X_scaled is None:
            self.prepare_data()

        train_ds = TimeSeriesDataset(self.X_scaled, self.y_all, self.train_indices, self.lookback)
        val_ds = TimeSeriesDataset(self.X_scaled, self.y_all, self.val_indices, self.lookback)
        test_ds = TimeSeriesDataset(self.X_scaled, self.y_all, self.test_indices, self.lookback)
        return train_ds, val_ds, test_ds

    def get_dataloaders(self, batch_size: int = 2048) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_ds, val_ds, test_ds = self.get_datasets()

        # Training DataLoader shuffles sequences after the chronological split
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        # Validation and Test DataLoaders preserve chronological order
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        return train_loader, val_loader, test_loader
