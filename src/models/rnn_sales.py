import torch
import torch.nn as nn

class LSTMSalesModel(nn.Module):
    """
    LSTM Baseline Model chi dung lich su chuoi thoi gian (Sales-Only).
    """
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, pred_len=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, pred_len)
        )

    def forward(self, history, **kwargs):
        # history: (Batch, Seq_len, 1)
        lstm_out, (hn, cn) = self.lstm(history) # hn: (num_layers, Batch, hidden_dim)
        last_hidden = hn[-1] # (Batch, hidden_dim)
        out = self.head(last_hidden) # (Batch, pred_len)
        return out

class GRUSalesModel(nn.Module):
    """
    GRU Baseline Model chi dung lich su chuoi thoi gian (Sales-Only).
    """
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, pred_len=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, pred_len)
        )

    def forward(self, history, **kwargs):
        gru_out, hn = self.gru(history) # hn: (num_layers, Batch, hidden_dim)
        last_hidden = hn[-1]
        out = self.head(last_hidden)
        return out
