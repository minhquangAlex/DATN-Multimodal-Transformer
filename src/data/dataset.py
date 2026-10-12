"""Lazy sliding windows over a single sales matrix; targets cannot cross splits."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from src.config import file_sha256
from src.data.splits import assign_time_split
from src.features.verify_embeddings import load_embedding_bundle


class HMDataset(Dataset):
    def __init__(self, sales, article_ids, weeks, origins, seq_len, pred_len,
                 image_embs=None, text_embs=None, availability=None, calendar=None):
        self.sales = sales
        self.article_ids = article_ids
        self.weeks = weeks
        self.origins = list(origins)
        self.seq_len, self.pred_len = seq_len, pred_len
        self.image_embs = image_embs or {}
        self.text_embs = text_embs or {}
        self.image_dim = next(iter(self.image_embs.values())).numel() if self.image_embs else 768
        self.text_dim = next(iter(self.text_embs.values())).numel() if self.text_embs else 384
        self.availability = availability
        self.calendar = calendar

    def __len__(self):
        return len(self.article_ids) * len(self.origins)

    def __getitem__(self, idx):
        origin_index, article_index = divmod(idx, len(self.article_ids))
        origin = self.origins[origin_index]
        key = self.article_ids[article_index]
        flags = self.availability.loc[key] if self.availability is not None else {}
        image = self.image_embs.get(key)
        text = self.text_embs.get(key)
        sample = {
            'article_id': key, 'forecast_origin': self.weeks[origin],
            'history': torch.from_numpy(self.sales[article_index, origin-self.seq_len:origin].copy()).unsqueeze(-1),
            'target': torch.from_numpy(self.sales[article_index, origin:origin+self.pred_len].copy()),
            'image_emb': image.float() if image is not None else torch.zeros(self.image_dim),
            'text_emb': text.float() if text is not None else torch.zeros(self.text_dim),
            'image_available': bool(flags.get('image_available', False)),
            'text_available': bool(flags.get('text_available', False)),
        }
        if self.calendar is not None:
            sample['history_calendar'] = torch.from_numpy(self.calendar[origin-self.seq_len:origin].copy())
            sample['future_calendar'] = torch.from_numpy(self.calendar[origin:origin+self.pred_len].copy())
        return sample


class HMDataModule:
    def __init__(self, dataset_index_path='processed_data/v2/dataset_index.csv',
                 image_emb_path='embeddings/v2/image/image_embeddings.pt',
                 text_emb_path='embeddings/v2/text/text_embeddings.pt',
                 selected_articles_path=None, metadata_path=None, seq_len=12, pred_len=2,
                 batch_size=256, num_workers=0, seed=42, use_embeddings=True, require_full=False):
        self.dataset_index_path = Path(dataset_index_path)
        self.selected_articles_path = Path(selected_articles_path) if selected_articles_path else self.dataset_index_path.parent / 'selected_articles.csv'
        self.metadata_path = Path(metadata_path) if metadata_path else Path(image_emb_path).parent.parent / 'metadata.json'
        self.image_emb_path, self.text_emb_path = image_emb_path, text_emb_path
        self.seq_len, self.pred_len = seq_len, pred_len
        self.batch_size, self.num_workers, self.seed = batch_size, num_workers, seed
        self.use_embeddings, self.require_full = use_embeddings, require_full
        if min(seq_len, pred_len, batch_size) < 1 or num_workers < 0:
            raise ValueError('Invalid window or loader settings')

    def setup(self):
        df = pd.read_csv(self.dataset_index_path, dtype={'article_id': str})
        required = {'article_id', 'year_week', 'demand', 'split', 'week_sin', 'week_cos', 'is_complete_week'}
        if not required.issubset(df.columns):
            raise ValueError('Rebuild stage 1 v2: index is missing split/calendar metadata')
        summary = json.loads((self.dataset_index_path.parent / 'stage1_summary.json').read_text(encoding='utf-8'))
        if summary.get('schema_version') != 2 or summary.get('selection_period') != 'train only':
            raise ValueError('Stage 1 must select products from train only')
        for path, field in [(self.dataset_index_path, 'dataset_index_sha256'), (self.selected_articles_path, 'selected_articles_sha256')]:
            if summary.get(field) != file_sha256(path):
                raise ValueError(f'Stage 1 hash mismatch: {path}')
        if df.article_id.isna().any() or df.duplicated(['article_id', 'year_week']).any():
            raise ValueError('Invalid or duplicate article/week keys')
        if not np.isfinite(df.demand).all() or (df.demand < 0).any():
            raise ValueError('Demand must be finite and nonnegative')
        if not df.is_complete_week.eq(True).all():
            raise ValueError('Incomplete weeks must be excluded')
        calendar = df[['year_week', 'split', 'week_sin', 'week_cos']].drop_duplicates().sort_values('year_week')
        if calendar.year_week.duplicated().any():
            raise ValueError('Inconsistent metadata for the same calendar week')
        expected = assign_time_split(calendar, summary['config']['train_end'], summary['config']['val_end'])
        if not calendar.split.equals(expected.split):
            raise ValueError('Index splits disagree with stage 1 cutoffs')
        weeks = calendar.year_week.tolist()
        dates = pd.to_datetime(calendar.year_week + '-1', format='%G-W%V-%u')
        if not dates.diff().dropna().eq(pd.Timedelta(days=7)).all():
            raise ValueError('Calendar weeks must be contiguous')
        selected = pd.read_csv(self.selected_articles_path, dtype={'article_id': str})
        pivot = df.pivot(index='article_id', columns='year_week', values='demand').reindex(columns=weeks)
        if pivot.isna().any().any() or set(pivot.index) != set(selected.article_id):
            raise ValueError('Index must contain a complete grid for the selected articles')
        self.image_embs, self.text_embs, availability = {}, {}, None
        self.embedding_report = None
        if self.use_embeddings:
            self.image_embs, self.text_embs, availability, self.embedding_report = load_embedding_bundle(
                self.selected_articles_path, self.image_emb_path, self.text_emb_path, self.metadata_path, self.require_full)
        sales = pivot.to_numpy(dtype=np.float32)
        labels = calendar.split.to_numpy()
        self.split_summary = {}
        for split in ['train', 'val', 'test']:
            origins = [t for t in range(self.seq_len, len(weeks)-self.pred_len+1)
                       if np.all(labels[t:t+self.pred_len] == split)]
            if not origins:
                raise ValueError(f'No valid {split} windows; check cutoffs, L and H')
            dataset = HMDataset(sales, pivot.index.tolist(), weeks, origins, self.seq_len, self.pred_len,
                                self.image_embs, self.text_embs, availability,
                                calendar[['week_sin', 'week_cos']].to_numpy(dtype=np.float32))
            setattr(self, f'{split}_dataset', dataset)
            self.split_summary[split] = {'samples': len(dataset), 'origins': len(origins),
                                         'first_target_week': weeks[origins[0]],
                                         'last_target_week': weeks[origins[-1]+self.pred_len-1]}
        print(self.split_summary)

    def _loader(self, split, shuffle=False):
        return DataLoader(getattr(self, f'{split}_dataset'), batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=self.num_workers, pin_memory=torch.cuda.is_available(),
                          generator=torch.Generator().manual_seed(self.seed),
                          persistent_workers=self.num_workers > 0)

    def train_dataloader(self):
        return self._loader('train', True)

    def val_dataloader(self):
        return self._loader('val')

    def test_dataloader(self):
        return self._loader('test')
