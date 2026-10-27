"""Small serialization helpers shared by reproducible experiment commands."""
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def default(x):
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=default, allow_nan=False), encoding="utf-8")


def utc_now():
    return datetime.now(timezone.utc).isoformat()
