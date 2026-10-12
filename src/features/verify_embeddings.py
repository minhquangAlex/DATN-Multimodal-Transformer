"""Fail closed on stale IDs, collapsed vectors, missing provenance or bad shapes."""
import argparse
import json
from pathlib import Path
import pandas as pd
import torch
from src.config import file_sha256, write_json


def validate_embedding_dict(name, emb_dict, expected_dim=None, available_ids=None):
    if not isinstance(emb_dict, dict) or not emb_dict:
        raise ValueError(f'{name}: expected a nonempty dictionary')
    if not all(isinstance(k, str) and len(k) == 10 and k.isdigit() for k in emb_dict):
        raise ValueError(f'{name}: article IDs must be 10 digit strings')
    dim = expected_dim or next(iter(emb_dict.values())).numel()
    for key, value in emb_dict.items():
        if not isinstance(value, torch.Tensor) or value.shape != (dim,):
            raise ValueError(f'{name}: invalid tensor shape for {key}, expected ({dim},)')
        if not torch.isfinite(value).all():
            raise ValueError(f'{name}: NaN or Inf for {key}')
    keys = sorted(emb_dict) if available_ids is None else sorted(available_ids)
    if not keys:
        raise ValueError(f'{name}: no available observations')
    matrix = torch.stack([emb_dict[k].detach().cpu().float() for k in keys])
    if (matrix.norm(dim=1) == 0).any():
        raise ValueError(f'{name}: available observation has a zero embedding')
    distinct = int(torch.unique(matrix, dim=0).shape[0])
    if len(keys) > 1 and (distinct == 1 or matrix.var(dim=0, unbiased=False).mean().item() < 1e-12):
        raise ValueError(f'{name}: collapsed embeddings; all observations have the same vector')
    return {'count': len(emb_dict), 'dim': dim, 'available_count': len(keys), 'distinct_vectors': distinct}


def load_embedding_bundle(selected_articles_path, image_emb_path, text_emb_path, metadata_path, require_full=True):
    selected = pd.read_csv(selected_articles_path, dtype={'article_id': str})
    ids = set(selected.article_id)
    if len(ids) != len(selected) or not ids:
        raise ValueError('Selected article IDs must be nonempty and unique')
    metadata_path = Path(metadata_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(f'Missing v2 embedding metadata: {metadata_path}. Run notebooks/02_MultiModal_Vector.ipynb with the v2 selection first.')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if metadata.get('schema_version') != 2 or metadata.get('selected_articles_sha256') != file_sha256(selected_articles_path):
        raise ValueError('Embedding provenance does not match selected_articles.csv; re-extract embeddings')
    paths = {'image': Path(image_emb_path), 'text': Path(text_emb_path)}
    availability_path = metadata_path.parent / 'availability.csv'
    if metadata.get('availability_sha256') != file_sha256(availability_path):
        raise ValueError('Availability file hash mismatch')
    availability = pd.read_csv(availability_path, dtype={'article_id': str}).set_index('article_id')
    if not availability.index.is_unique or set(availability.index) != ids:
        raise ValueError('Availability IDs do not exactly match selection')
    bundles, report = {}, {}
    for name, path in paths.items():
        if metadata.get(f'{name}_sha256') != file_sha256(path):
            raise ValueError(f'{name}: embedding file hash mismatch')
        vectors = torch.load(path, map_location='cpu', weights_only=True)
        if not isinstance(vectors, dict) or set(vectors) != ids:
            raise ValueError(f'{name}: embedding IDs must match selected articles exactly')
        flags = availability[f'{name}_available']
        if flags.isna().any() or not flags.isin([True, False, 0, 1]).all():
            raise ValueError(f'{name}: invalid availability flags')
        available = set(flags[flags.eq(True)].index)
        if require_full and available != ids:
            raise ValueError(f'{name}: full-modality experiment requires every observation; {len(ids - available)} unavailable')
        for key in ids - available:
            if not isinstance(vectors[key], torch.Tensor) or torch.count_nonzero(vectors[key]):
                raise ValueError(f'{name}: unavailable observations must use explicit zero placeholders')
        report[name] = validate_embedding_dict(name, vectors, metadata[f'{name}_dim'], available)
        bundles[name] = vectors
    return bundles['image'], bundles['text'], availability, report


def verify_stage2_embeddings(selected_articles_path='processed_data/v2/selected_articles.csv',
                             image_emb_path='embeddings/v2/image/image_embeddings.pt',
                             text_emb_path='embeddings/v2/text/text_embeddings.pt',
                             metadata_path='embeddings/v2/metadata.json', require_full=True):
    report_path = Path(metadata_path).parent / 'validation_report.json'
    try:
        _, _, _, report = load_embedding_bundle(selected_articles_path, image_emb_path, text_emb_path, metadata_path, require_full)
    except Exception as exc:
        write_json(report_path, {'validation': 'failed', 'error': str(exc)})
        try:
            metadata = json.loads(Path(metadata_path).read_text(encoding='utf-8'))
        except (OSError, ValueError):
            metadata = None
        if isinstance(metadata, dict):
            metadata.update(validation='failed', validation_error=str(exc))
            write_json(metadata_path, metadata)
        raise
    write_json(report_path, {'validation': 'passed', 'require_full': require_full, 'modalities': report,
                             'selected_articles_sha256': file_sha256(selected_articles_path)})
    metadata = json.loads(Path(metadata_path).read_text(encoding='utf-8'))
    metadata['validation'] = 'passed'
    metadata.pop('validation_error', None)
    metadata['validation_require_full'] = require_full
    metadata['validation_report'] = report_path.name
    write_json(metadata_path, metadata)
    print(report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--selected-articles-path', default='processed_data/v2/selected_articles.csv')
    parser.add_argument('--image-emb-path', default='embeddings/v2/image/image_embeddings.pt')
    parser.add_argument('--text-emb-path', default='embeddings/v2/text/text_embeddings.pt')
    parser.add_argument('--metadata-path', default='embeddings/v2/metadata.json')
    parser.add_argument('--allow-natural-missing', action='store_true')
    args = vars(parser.parse_args())
    args['require_full'] = not args.pop('allow_natural_missing')
    verify_stage2_embeddings(**args)
