from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

# Allow running as a script without installing the package
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

sys.path.append(str(SRC))
from deadtime_analysis import DeadtimeAnalysis  # noqa: E402


# Default to the precombined dataset (includes 3-pulse filter for 12-14-25).
DEFAULT_DATA_FILES = [
    ROOT / "data" / "combined_deadtime.jsonl",
]


def _to_int(val: Optional[float]) -> Optional[int]:
    if pd.isna(val):
        return None
    return int(val)


def _row_to_estimate(
    rate: float,
    channel_count: float,
    windows: float,
    num_pulses: Optional[float],
    group: pd.DataFrame,
    converged_row: Optional[pd.Series],
    min_double_row: Optional[pd.Series],
) -> dict:
    tested_max = float(group["separation_ns"].max())
    conv_deadtime = (
        float(converged_row["converged_deadtime_ns"]) if converged_row is not None else None
    )
    conv_lower = (
        float(converged_row["converged_lower_bound_ns"]) if converged_row is not None else None
    )
    conv_upper = (
        float(converged_row["converged_upper_bound_ns"]) if converged_row is not None else None
    )
    min_all = float(min_double_row["min_double_deadtime_ns"]) if min_double_row is not None else None
    min_all_lower = (
        float(min_double_row["min_double_lower_bound_ns"]) if min_double_row is not None else None
    )
    return {
        "pulse_rate_hz": float(rate),
        "channel_count": _to_int(channel_count),
        "windows": _to_int(windows),
        "num_pulses": _to_int(num_pulses),
        "tested_max_separation_ns": float(tested_max),
        "converged_deadtime_ns": conv_deadtime,
        "converged_deadtime_us": conv_deadtime / 1000.0 if conv_deadtime is not None else None,
        "converged_lower_bound_ns": conv_lower,
        "converged_upper_bound_ns": conv_upper,
        "converged_lower_bound_us": conv_lower / 1000.0 if conv_lower is not None else None,
        "converged_upper_bound_us": conv_upper / 1000.0 if conv_upper is not None else None,
        "min_all_pulses_response_ns": min_all,
        "min_all_pulses_response_us": min_all / 1000.0 if min_all is not None else None,
        "min_all_pulses_lower_bound_ns": min_all_lower,
        "min_all_pulses_lower_bound_us": (
            min_all_lower / 1000.0 if min_all_lower is not None else None
        ),
        "all_pulses_observed": min_double_row is not None,
        "note": None,
    }


def generate_estimates(analysis: DeadtimeAnalysis) -> List[dict]:
    df = analysis.df
    conv = analysis.converged_table()
    min_double = analysis.min_double_table()
    results: List[dict] = []
    for (rate, channel_count, windows, num_pulses), group_df in df.groupby(
        ["pulse_rate_hz", "channel_count", "windows", "num_pulses"]
    ):
        conv_row = conv[
            (conv["pulse_rate_hz"] == rate)
            & (conv["channel_count"] == channel_count)
            & (conv["windows"] == windows)
            & (conv["num_pulses"] == num_pulses)
        ]
        conv_row = conv_row.iloc[0] if not conv_row.empty else None
        md_row = min_double[
            (min_double["pulse_rate_hz"] == rate)
            & (min_double["channel_count"] == channel_count)
            & (min_double["windows"] == windows)
            & (min_double["num_pulses"] == num_pulses)
        ]
        md_row = md_row.iloc[0] if not md_row.empty else None
        results.append(
            _row_to_estimate(
                rate=rate,
                channel_count=channel_count,
                windows=windows,
                num_pulses=num_pulses,
                group=group_df,
                converged_row=conv_row,
                min_double_row=md_row,
            )
        )
    return sorted(
        results,
        key=lambda r: (
            r["pulse_rate_hz"],
            r["num_pulses"] if r["num_pulses"] is not None else 0,
            r["channel_count"],
            r["windows"],
        ),
    )


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate deadtime estimates JSON.")
    parser.add_argument(
        "--data-files",
        nargs="+",
        type=Path,
        default=DEFAULT_DATA_FILES,
        help="JSONL data files to include",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "estimated_deadtime_all.json",
        help="Path to write aggregated estimates JSON",
    )
    args = parser.parse_args(argv)

    analysis = DeadtimeAnalysis.from_jsonl([str(p) for p in args.data_files])
    estimates = generate_estimates(analysis)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(estimates, indent=2))
    print(f"Wrote {len(estimates)} estimates to {args.output}")


if __name__ == "__main__":
    main()
