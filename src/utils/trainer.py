"""Train in raw/log1p space; select checkpoints using validation MAE in sales units."""
from pathlib import Path
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from src.utils.metrics import evaluate_all_metrics
from src.config import write_json


class BaseTrainer:
    def __init__(self, model, optimizer=None, criterion=None, device=None, checkpoint_dir='checkpoints/v2',
                 model_name='model', modality_config=None, target_transform='log1p', grad_clip=1.0):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = criterion or nn.MSELoss()
        self.modality_config = {'image': True, 'text': True, **(modality_config or {})}
        if target_transform not in {'none', 'log1p'}:
            raise ValueError('target_transform must be none or log1p')
        self.target_transform, self.grad_clip = target_transform, grad_clip
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.best_checkpoint_path = self.checkpoint_dir / f'best_{model_name}.pt'
        self.best_val_loss = float('inf')
        self.history = []

    def _prepare_batch(self, batch, modality_config=None):
        config = {**self.modality_config, **(modality_config or {})}
        history = batch['history'].to(self.device).clone()
        target = batch['target'].to(self.device)
        batch_size = history.shape[0]
        history_mask = torch.ones_like(history, dtype=torch.bool)
        keep = config.get('history_keep_weeks', history.shape[1])
        if not isinstance(keep, int) or not 1 <= keep <= history.shape[1]:
            raise ValueError('history_keep_weeks must be between 1 and L')
        if keep < history.shape[1]:
            history[:, :-keep] = 0
            history_mask[:, :-keep] = False
        image = batch.get('image_emb', torch.zeros(batch_size, getattr(self.model, 'image_dim', 768))).to(self.device).clone()
        text = batch.get('text_emb', torch.zeros(batch_size, getattr(self.model, 'text_dim', 384))).to(self.device).clone()
        masks = {}
        for name, vectors in [('image', image), ('text', text)]:
            available = batch.get(f'{name}_available', torch.ones(batch_size, dtype=torch.bool)).to(self.device).bool().clone()
            if not config.get(name, True):
                available[:] = False
            vectors[~available] = 0
            masks[f'{name}_available'] = available
        if self.target_transform == 'log1p':
            history = torch.log1p(history)
        inputs = {'history': history, 'image_emb': image, 'text_emb': text,
                  'history_mask': history_mask, **masks}
        # Calendar features are exposed to future models; current sales-only baselines ignore them.
        for name in ['history_calendar', 'future_calendar']:
            if name in batch:
                inputs[name] = batch[name].to(self.device)
        return inputs, target

    def _inverse(self, values):
        result = torch.expm1(values) if self.target_transform == 'log1p' else values
        if not torch.isfinite(result).all():
            raise ValueError('Nonfinite predictions after inverse transform')
        return result.clamp_min(0)

    def fit(self, train_loader, val_loader, epochs=30, patience=5):
        if epochs < 1 or patience < 1 or len(train_loader) == 0 or len(val_loader) == 0:
            raise ValueError('Training requires nonempty loaders, positive epochs and patience')
        self.best_val_loss = float('inf')
        self.history = []
        patience_counter = 0
        saved_this_run = False
        for epoch in range(1, epochs+1):
            self.model.train()
            total, count = 0.0, 0
            for batch in train_loader:
                inputs, raw_target = self._prepare_batch(batch)
                target = torch.log1p(raw_target) if self.target_transform == 'log1p' else raw_target
                self.optimizer.zero_grad(set_to_none=True)
                prediction = self.model(**inputs)
                if prediction.shape != target.shape:
                    raise ValueError('Prediction/target shape mismatch')
                loss = self.criterion(prediction, target)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip, error_if_nonfinite=True)
                self.optimizer.step()
                total += loss.item()*len(target)
                count += len(target)
            metrics = self.evaluate(val_loader)
            val_mae = metrics['MAE']
            row = {'epoch': epoch, 'train_loss': total/count, **{f'val_{k}': v for k, v in metrics.items()}}
            self.history.append(row)
            pd.DataFrame(self.history).to_csv(self.checkpoint_dir / f'{self.model_name}_history.csv', index=False)
            print(f'{self.model_name} epoch {epoch}: train loss={total/count:.5f}; val MAE={val_mae:.5f}', flush=True)
            if not math.isfinite(val_mae):
                raise ValueError('Nonfinite validation MAE')
            if val_mae < self.best_val_loss:
                self.best_val_loss, patience_counter = val_mae, 0
                torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(),
                            'epoch': epoch, 'val_mae': val_mae, 'target_transform': self.target_transform,
                            'modality_config': self.modality_config}, self.best_checkpoint_path)
                saved_this_run = True
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        if not saved_this_run:
            raise RuntimeError('No valid checkpoint was produced by this run')
        checkpoint = torch.load(self.best_checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        write_json(self.checkpoint_dir / f'{self.model_name}_training.json', {
            'best_epoch': checkpoint['epoch'], 'best_val_mae': checkpoint['val_mae'],
            'target_transform': self.target_transform, 'train_loss': type(self.criterion).__name__,
            'checkpoint_selection': 'validation MAE in original sales units',
            'prediction_policy': 'inverse transform then clamp negatives to zero',
            'modality_config': self.modality_config,
        })
        return self.history

    def predict(self, data_loader, modality_config=None):
        self.model.eval()
        predictions, targets, ids, origins = [], [], [], []
        with torch.inference_mode():
            for batch in data_loader:
                inputs, target = self._prepare_batch(batch, modality_config)
                prediction = self._inverse(self.model(**inputs))
                if prediction.shape != target.shape:
                    raise ValueError('Prediction/target shape mismatch')
                predictions.append(prediction.cpu().numpy())
                targets.append(target.cpu().numpy())
                ids.extend(batch.get('article_id', ['unknown']*len(target)))
                origins.extend(batch.get('forecast_origin', ['unknown']*len(target)))
        if not predictions:
            raise ValueError('Cannot evaluate an empty loader')
        return np.concatenate(targets), np.concatenate(predictions), ids, origins

    def evaluate(self, data_loader, modality_config=None):
        true, pred, _, _ = self.predict(data_loader, modality_config)
        return evaluate_all_metrics(true, pred)

    def prediction_frame(self, data_loader, modality_config=None):
        true, pred, ids, origins = self.predict(data_loader, modality_config)
        return prediction_frame(true, pred, ids, origins)


def prediction_frame(true, pred, ids, origins):
    horizon = true.shape[1]
    frame = pd.DataFrame({'article_id': np.repeat(ids, horizon), 'forecast_origin': np.repeat(origins, horizon),
                          'horizon': np.tile(np.arange(1, horizon+1), len(ids)),
                          'actual': true.reshape(-1), 'prediction': pred.reshape(-1)})
    starts = pd.to_datetime(frame.forecast_origin + '-1', format='%G-W%V-%u')
    target_dates = starts + pd.to_timedelta((frame.horizon-1)*7, unit='D')
    iso = target_dates.dt.isocalendar()
    frame['target_week'] = iso.year.astype(str) + '-W' + iso.week.astype(str).str.zfill(2)
    return frame
