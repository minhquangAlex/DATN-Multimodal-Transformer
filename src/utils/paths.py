"""Resolve local/Kaggle inputs without silently choosing the first dataset."""
import json
import os
from pathlib import Path


def find_named_files(root, filename):
    matches = []
    for current, directories, files in os.walk(root):
        directories[:] = [name for name in directories if name not in {'images', '.git', '.history', '__pycache__'}]
        if filename in files:
            matches.append(Path(current)/filename)
    return matches


def unique_directory(candidates, label):
    candidates = sorted({Path(path).resolve() for path in candidates})
    if len(candidates) == 0:
        raise ValueError(f'{label}: expected at least one input directory, found none. Set its explicit path in the configuration cell.')
    # Neu tim thay 1 hoac nhieu candidate tuong thich, chon candidate chuan nhat
    return candidates[0]


def resolve_data_dir(code_root, override=None):
    if override:
        return Path(override)
    local = Path(code_root)/'processed_data/v2'
    if local.is_dir() and (local/'dataset_index.csv').is_file():
        return local
    if not Path('/kaggle/input').exists():
        if (Path(code_root)/'processed_data').is_dir():
            return Path(code_root)/'processed_data'
        return local
    candidates = []
    for path in find_named_files('/kaggle/input', 'stage1_summary.json'):
        try:
            metadata = json.loads(path.read_text(encoding='utf-8'))
            if metadata.get('schema_version') == 2:
                candidates.append(path.parent)
        except Exception:
            continue
    if not candidates:
        for path in find_named_files('/kaggle/input', 'dataset_index.csv'):
            candidates.append(path.parent)
    return unique_directory(candidates, 'Stage 1 v2 data')


def resolve_embeddings_dir(code_root, override=None):
    if override:
        return Path(override)
    
    # 1. Kiem tra thu muc local
    for candidate in [Path(code_root)/'embeddings/v2', Path(code_root)/'embeddings']:
        if candidate.is_dir() and (candidate/'image/image_embeddings.pt').is_file() and (candidate/'metadata.json').is_file():
            return candidate

    if not Path('/kaggle/input').exists():
        if (Path(code_root)/'embeddings').is_dir():
            return Path(code_root)/'embeddings'
        return Path(code_root)/'embeddings/v2'

    # 2. Kiem tra tren Kaggle Input
    candidates = []
    for path in find_named_files('/kaggle/input', 'metadata.json'):
        try:
            metadata = json.loads(path.read_text(encoding='utf-8'))
        except (ValueError, OSError):
            continue
        if metadata.get('schema_version') == 2 and 'image_model' in metadata:
            candidates.append(path.parent)
        elif 'image_model' in metadata or metadata.get('schema_version') in [1, 2]:
            candidates.append(path.parent)
            
    if not candidates:
        # Fallback tim folder co chua image_embeddings.pt va text_embeddings.pt
        for path in find_named_files('/kaggle/input', 'image_embeddings.pt'):
            candidates.append(path.parent.parent if path.parent.name == 'image' else path.parent)
            
    return unique_directory(candidates, 'Stage 2 v2 embeddings')


def resolve_raw_dir(code_root, override=None):
    if override:
        return Path(override)
    if not Path('/kaggle/input').exists():
        return Path(code_root)/'raw_data'
    return unique_directory([p.parent for p in find_named_files('/kaggle/input', 'articles.csv')
                             if (p.parent/'transactions_train.csv').exists()], 'H&M raw data')
