"""Shared defaults; cutoffs refer to the end of complete ISO weeks."""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json

@dataclass(frozen=True)
class DataConfig:
    train_end: str = "2020-04-05"
    val_end: str = "2020-06-28"
    min_active_weeks: int = 20
    min_total_transactions: int = 50
    target_sample_size: int = 8000
    seed: int = 42

def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
