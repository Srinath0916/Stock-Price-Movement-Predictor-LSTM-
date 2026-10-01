"""
PyTorch LSTM Architecture for Next-Day Stock Movement Direction Prediction.
"""

import torch
import torch.nn as nn


class StockLSTM(nn.Module):
    """
    Standard 2-layer LSTM model for binary classification of next-day price movement.

    Architecture:
        Input features (batch_size, seq_len, input_dim)
        -> LSTM (num_layers=2, hidden_size, dropout, batch_first=True)
        -> Dropout (dropout_rate)
        -> Linear (hidden_size -> 1 logit)
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.bidirectional = bidirectional

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        effective_hidden = hidden_size * (2 if bidirectional else 1)

        # Output Head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(effective_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, seq_len, input_dim)

        Returns
        -------
        logits : torch.Tensor of shape (batch_size, 1)
        """
        # lstm_out: (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)

        # Extract representation from final sequence step
        last_step = lstm_out[:, -1, :]

        # Dropout and linear projection to single logit
        out = self.dropout(last_step)
        logits = self.fc(out)
        return logits
