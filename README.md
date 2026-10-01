# Stock Price Movement Predictor

A machine learning project to predict stock price movements using LSTM neural networks, technical indicators, and walk-forward validation.

## Project Structure

```text
stock-price-movement-predictor/
├── data/
│   ├── raw/             # Original Kaggle dataset
│   └── processed/       # Cleaned and preprocessed data
├── notebooks/           # Exploratory data analysis (EDA) and experiments
├── src/
│   ├── data/            # Data loading, validation, and cleaning scripts
│   ├── features/        # Feature engineering & technical indicators
│   ├── models/          # Model architecture (LSTM) & training routines
│   └── evaluation/      # Metrics, backtesting, and walk-forward validation
├── models/              # Saved trained model artifacts and checkpoints
├── results/             # Evaluation metrics, performance plots, and predictions
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## Setup & Environment

1. **Create & activate virtual environment:**
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python -c "import pandas, numpy, matplotlib, seaborn, sklearn, torch; print('All ML dependencies working'); print('PyTorch:', torch.__version__)"
   python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
   ```
