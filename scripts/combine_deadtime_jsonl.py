from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUTS = [
    ROOT / "data" / "double_pulse_deadtime-11-20-25.jsonl",
    ROOT / "data" / "double_pulse_deadtime-11-22-25.jsonl",
    ROOT / "data" / "n_pulse_deadtime-12-14-25.jsonl",
    ROOT / "data" / "n_pulse_deadtime-12-15-25.jsonl",
]

DEFAULT_OUTPUT = ROOT / "data" / "combined_deadtime.jsonl"


def load_jsonl(path: Path) -> List[dict]:
    out = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def save_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec))
            fh.write("\n")


def combine(
    inputs: Iterable[Path],
    output: Path,
    *,
    restrict_12_14_to_three_pulses: bool = True,
) -> None:
    combined: List[dict] = []
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(path)
        records = load_jsonl(path)
        if restrict_12_14_to_three_pulses and path.name == "n_pulse_deadtime-12-14-25.jsonl":
            records = [
                r
                for r in records
                if (r.get("pulse_sequence") or {}).get("num_pulses") == 3
                or ((r.get("search") or {}).get("num_pulses") == 3)
            ]
        combined.extend(records)
    save_jsonl(output, combined)
    print(f"Wrote {len(combined)} records to {output}")


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Combine deadtime JSONL files.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=DEFAULT_INPUTS,
        help="Input JSONL files (order preserved).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--no-three-pulse-filter",
        action="store_true",
        help="Do not restrict n_pulse_deadtime-12-14-25.jsonl to 3-pulse entries.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    combine(
        inputs=args.inputs,
        output=args.output,
        restrict_12_14_to_three_pulses=not args.no_three_pulse_filter,
    )


if __name__ == "__main__":
    main()
