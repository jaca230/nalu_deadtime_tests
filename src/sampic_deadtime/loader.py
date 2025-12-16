from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List


def load_records(paths: Iterable[Path | str]) -> List[dict]:
    """Load JSONL files into a list of raw dict records."""
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
