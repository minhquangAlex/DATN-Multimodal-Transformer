import numpy as np
import torch
import torch.nn as nn

class LastValueNaiveBaseline:
    """
    Last-value Naive Baseline:
    Du bao bang gia tri cua tuan cuoi cung trong chuoi lich su.
    """
    def __init__(self, pred_len=2):
        self.pred_len = pred_len

    def predict(self, history):
        """
        history: Tensor or ndarray shape (Batch, Seq_len, 1) hoac (Batch, Seq_len)
        return: ndarray shape (Batch, Pred_len)
        """
        if isinstance(history, torch.Tensor):
            history = history.detach().cpu().numpy()
        if history.ndim == 3:
            history = history.squeeze(-1)
            
        last_val = history[:, -1:] # (Batch, 1)
        # Lap lai gia tri tuan gan nhat cho toan bo pred_len tuan toi
        preds = np.repeat(last_val, self.pred_len, axis=1)
        return preds

class MovingAverageBaseline:
    """
    Moving Average Baseline:
    Du bao bang trung binh cong cua K tuan gan nhat.
    """
    def __init__(self, pred_len=2, window=4):
        self.pred_len = pred_len
        self.window = window

    def predict(self, history):
        if isinstance(history, torch.Tensor):
            history = history.detach().cpu().numpy()
        if history.ndim == 3:
            history = history.squeeze(-1)
            
        avg_val = np.mean(history[:, -self.window:], axis=1, keepdims=True) # (Batch, 1)
        preds = np.repeat(avg_val, self.pred_len, axis=1)
        return preds

class MLPBaseline(nn.Module):
    """
    MLP Baseline (Sales-Only)
    """
    def __init__(self, seq_len=12, pred_len=2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, pred_len)
        )

    def forward(self, history, **kwargs):
        # history: (Batch, Seq_len, 1) -> flatten thanh (Batch, Seq_len)
        if history.ndim == 3:
            history = history.squeeze(-1)
        return self.net(history)

class SimpleMultimodalBaseline(nn.Module):
    """
    Baseline đa phương thức tối giản nhưng thực sự dùng lịch sử + image + text.
    Dùng cho việc so sánh khi thiếu dữ liệu đa phương thức và đánh giá hiệu quả fusion.
    """
    def __init__(self, seq_len=12, pred_len=2, image_dim=768, text_dim=384, hidden_dim=64):
        super().__init__()
        self.seq_len = seq_len
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.net = nn.Sequential(
            nn.Linear(seq_len + image_dim + text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, pred_len)
        )

    def forward(self, history=None, image_emb=None, text_emb=None, **kwargs):
        if history is None:
            raise ValueError("history is required for SimpleMultimodalBaseline")

        if history.ndim == 3:
            history = history.squeeze(-1)

        batch_size = history.size(0)

        if image_emb is None:
            image_emb = torch.zeros(batch_size, self.image_dim, device=history.device, dtype=history.dtype)
        if text_emb is None:
            text_emb = torch.zeros(batch_size, self.text_dim, device=history.device, dtype=history.dtype)

        x = torch.cat([history, image_emb, text_emb], dim=-1)
        return self.net(x)


class ZeroBaseline(LastValueNaiveBaseline):
    def predict(self, history):
        return np.zeros((len(history), self.pred_len), dtype=np.float32)


class SeasonalNaiveBaseline(LastValueNaiveBaseline):
    """Repeat the last seasonal cycle. Requires at least season_length observations."""
    def __init__(self, pred_len=2, season_length=52):
        super().__init__(pred_len)
        if season_length < 1:
            raise ValueError('season_length must be positive')
        self.season_length = season_length

    def predict(self, history):
        values = history.detach().cpu().numpy() if isinstance(history, torch.Tensor) else np.asarray(history)
        if values.ndim == 3:
            values = values.squeeze(-1)
        if values.shape[1] < self.season_length:
            raise ValueError(f'Seasonal naive requires at least {self.season_length} weeks')
        offsets = np.arange(self.pred_len) % self.season_length
        return values[:, -self.season_length:][:, offsets]


class LinearRegressionBaseline(nn.Module):
    """A single affine layer, without hidden layers or activation."""
    def __init__(self, seq_len=12, pred_len=2):
        super().__init__()
        self.net = nn.Linear(seq_len, pred_len)

    def forward(self, history, **kwargs):
        if history.ndim == 3:
            history = history.squeeze(-1)
        return self.net(history)
