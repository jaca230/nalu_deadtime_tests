from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


def _get_active_channel_count(entry: dict) -> Optional[int]:
    capture = entry.get("capture_settings", {})
    channels = capture.get("active_channels")
    if channels is None:
        raw = capture.get("raw", {})
        channels = raw.get("active_channels")
    if channels is None:
        return None
    return len(channels)


def load_records(paths: Iterable[Path | str]) -> List[dict]:
    """Load a set of JSONL files into a list of dict records."""
    records: List[dict] = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    return records
