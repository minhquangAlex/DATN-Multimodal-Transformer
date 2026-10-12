import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # (1, max_len, d_model)

    def forward(self, x):
        # x: (Batch, Seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

class SalesTransformer(nn.Module):
    """
    Vanilla Time-Series Transformer (Sales-Only Baseline).
    Su dung co che Temporal Self-Attention tren chuoi lich su ban hang.
    """
    def __init__(
        self,
        input_dim=1,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        pred_len=2,
        dropout=0.1
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, pred_len)
        )

    def forward(self, history, **kwargs):
        # history: (Batch, Seq_len, 1)
        x = self.input_proj(history) # (Batch, Seq_len, d_model)
        x = self.pos_encoder(x)
        
        encoded = self.transformer_encoder(x) # (Batch, Seq_len, d_model)
        
        # Lay vector bieu dien tai moc thoi gian cuoi cung (hoac Mean Pooling)
        last_rep = encoded[:, -1, :] # (Batch, d_model)
        
        out = self.head(last_rep) # (Batch, pred_len)
        return out
