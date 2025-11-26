#!/usr/bin/env python3
"""Export minimum double-response separations for 10 Hz runs as JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from deadtime_analysis import DeadtimeAnalysis  # noqa: E402

DEFAULT_INPUTS: Sequence[Path] = [
    ROOT / "data" / "double_pulse_deadtime-11-19-25.jsonl",
    ROOT / "data" / "double_pulse_deadtime-11-20-25.jsonl",
    ROOT / "data" / "double_pulse_deadtime-11-21-25.jsonl",
    ROOT / "data" / "double_pulse_deadtime-11-22-25.jsonl",
]
DEFAULT_OUTPUT = ROOT / "data" / "estimated_deadtime_10hz.json"
DEFAULT_PULSE_RATE = 10.0
WINDOWS = [1, 2, 4, 8, 16, 32, 61]
CHANNELS = [1, 2, 4, 8, 16]


def _load_analysis(paths: Iterable[Path]) -> DeadtimeAnalysis:
    string_paths = [str(p) for p in paths]
    return DeadtimeAnalysis.from_jsonl(string_paths)


def _build_records(analysis: DeadtimeAnalysis, pulse_rate_hz: float) -> List[dict]:
    df = analysis.df
    min_double = analysis.min_double_table()
    converged = analysis.converged_table()
    records: List[dict] = []
    for channel_count in CHANNELS:
        for windows in WINDOWS:
            combo_mask = (
                (df["pulse_rate_hz"] == pulse_rate_hz)
                & (df["channel_count"] == channel_count)
                & (df["windows"] == windows)
            )
            combo_df = df[combo_mask]
            conv_row = converged[
                (converged["pulse_rate_hz"] == pulse_rate_hz)
                & (converged["channel_count"] == channel_count)
                & (converged["windows"] == windows)
            ]
            if not conv_row.empty:
                conv_ns_val = conv_row.iloc[0]["converged_deadtime_ns"]
                conv_low_val = conv_row.iloc[0]["converged_lower_bound_ns"]
                conv_high_val = conv_row.iloc[0]["converged_upper_bound_ns"]
                conv_ns = int(conv_ns_val) if pd.notna(conv_ns_val) else None
                conv_low = int(conv_low_val) if pd.notna(conv_low_val) else None
                conv_high = int(conv_high_val) if pd.notna(conv_high_val) else None
            else:
                conv_ns = conv_low = conv_high = None

            record = {
                "pulse_rate_hz": pulse_rate_hz,
                "channel_count": channel_count,
                "windows": windows,
                "tested_max_separation_ns": int(combo_df["separation_ns"].max()) if not combo_df.empty else None,
                "converged_deadtime_ns": conv_ns,
                "converged_deadtime_us": round(conv_ns / 1000.0, 3) if conv_ns is not None else None,
                "converged_lower_bound_ns": conv_low,
                "converged_upper_bound_ns": conv_high,
                "converged_lower_bound_us": round(conv_low / 1000.0, 3) if conv_low is not None else None,
                "converged_upper_bound_us": round(conv_high / 1000.0, 3) if conv_high is not None else None,
                "min_double_response_ns": None,
                "min_double_response_us": None,
                "min_double_lower_bound_ns": None,
                "min_double_lower_bound_us": None,
                "double_response_observed": False,
                "note": None,
            }
            if combo_df.empty:
                record["note"] = "No records found for this combination in the source files."
            else:
                min_row = min_double[
                    (min_double["pulse_rate_hz"] == pulse_rate_hz)
                    & (min_double["channel_count"] == channel_count)
                    & (min_double["windows"] == windows)
                ]
                if min_row.empty:
                    record["note"] = "Double-response threshold never crossed in provided runs."
                else:
                    min_sep_ns = int(min_row.iloc[0]["min_double_deadtime_ns"])
                    lower_ns = min_row.iloc[0]["min_double_lower_bound_ns"]
                    record.update(
                        {
                            "min_double_response_ns": min_sep_ns,
                            "min_double_response_us": round(min_sep_ns / 1000.0, 3),
                            "min_double_lower_bound_ns": int(lower_ns) if pd.notna(lower_ns) else None,
                            "min_double_lower_bound_us": round(lower_ns / 1000.0, 3)
                            if pd.notna(lower_ns)
                            else None,
                            "double_response_observed": True,
                            "note": None,
                        }
                    )
            records.append(record)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a JSON list of minimum double-response separations for 10 Hz runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=None,
        help="JSONL input files to scan (defaults to all bundled runs).",
    )
    parser.add_argument(
        "--pulse-rate",
        type=float,
        default=DEFAULT_PULSE_RATE,
        help="Pulse repetition rate to export.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination JSON file (list of dicts).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = args.inputs or DEFAULT_INPUTS
    missing = [p for p in inputs if not p.exists()]
    if missing:
        missing_str = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(f"Missing input file(s): {missing_str}")

    analysis = _load_analysis(inputs)
    records = _build_records(analysis, pulse_rate_hz=args.pulse_rate)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fh:
        json.dump(records, fh, indent=2)
        fh.write("\n")

    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
