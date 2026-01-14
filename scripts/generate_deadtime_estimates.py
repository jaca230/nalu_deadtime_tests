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
    *,
    channel_ratio_required: bool,
    full_channels_required: bool,
    packet_row: Optional[pd.Series],
    packet_ratio_target: float,
    packet_ratio_tolerance: float,
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
    packet_upper = (
        float(packet_row["packet_ratio_deadtime_ns"]) if packet_row is not None else None
    )
    packet_lower = (
        float(packet_row["packet_ratio_lower_bound_ns"]) if packet_row is not None else None
    )
    packet_threshold = (
        float(packet_row["packet_ratio_threshold"]) if packet_row is not None else None
    )
    return {
        "pulse_rate_hz": float(rate),
        "channel_count": _to_int(channel_count),
        "windows": _to_int(windows),
        "num_pulses": _to_int(num_pulses),
        "channel_ratio_required": channel_ratio_required,
        "full_channels_required": full_channels_required,
        "packet_ratio_target": packet_ratio_target,
        "packet_ratio_tolerance": packet_ratio_tolerance,
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
        "packet_ratio_threshold": packet_threshold,
        "packet_deadtime_upper_bound_ns": packet_upper,
        "packet_deadtime_upper_bound_us": packet_upper / 1000.0 if packet_upper is not None else None,
        "packet_deadtime_lower_bound_ns": packet_lower,
        "packet_deadtime_lower_bound_us": packet_lower / 1000.0 if packet_lower is not None else None,
        "all_pulses_observed": min_double_row is not None,
        "note": None,
    }


def generate_estimates(
    analysis: DeadtimeAnalysis,
    *,
    require_channel_ratio_pass: bool = False,
    require_full_channels: bool = False,
    packet_ratio_target: float = 2.0,
    packet_ratio_tolerance: float = 0.01,
) -> List[dict]:
    df = analysis.df
    conv = analysis.converged_table()
    min_double = analysis.min_double_table(
        require_channel_ratio_pass=require_channel_ratio_pass,
        require_full_channels=require_full_channels,
    )
    packet_bounds = analysis.packet_ratio_table(
        target_ratio=packet_ratio_target, tolerance=packet_ratio_tolerance
    )
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
        packet_row = packet_bounds[
            (packet_bounds["pulse_rate_hz"] == rate)
            & (packet_bounds["channel_count"] == channel_count)
            & (packet_bounds["windows"] == windows)
            & (packet_bounds["num_pulses"] == num_pulses)
        ]
        packet_row = packet_row.iloc[0] if not packet_row.empty else None
        results.append(
            _row_to_estimate(
                rate=rate,
                channel_count=channel_count,
                windows=windows,
                num_pulses=num_pulses,
                group=group_df,
                converged_row=conv_row,
                min_double_row=md_row,
                channel_ratio_required=require_channel_ratio_pass,
                full_channels_required=require_full_channels,
                packet_row=packet_row,
                packet_ratio_target=packet_ratio_target,
                packet_ratio_tolerance=packet_ratio_tolerance,
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
    parser.add_argument(
        "--require-channel-threshold",
        action="store_true",
        help="Require channel ratio to pass threshold when computing min-all-pulses response.",
    )
    parser.add_argument(
        "--require-full-channels",
        action="store_true",
        help="Require all channels to be observed when computing min-all-pulses response.",
    )
    parser.add_argument(
        "--packet-ratio-target",
        type=float,
        default=2.0,
        help="Target packet ratio (default=2.0 for double-pulse doubling).",
    )
    parser.add_argument(
        "--packet-ratio-tolerance",
        type=float,
        default=0.01,
        help="Tolerance subtracted from the target ratio when finding the upper bound.",
    )
    args = parser.parse_args(argv)

    analysis = DeadtimeAnalysis.from_jsonl([str(p) for p in args.data_files])
    estimates = generate_estimates(
        analysis,
        require_channel_ratio_pass=args.require_channel_threshold,
        require_full_channels=args.require_full_channels,
        packet_ratio_target=args.packet_ratio_target,
        packet_ratio_tolerance=args.packet_ratio_tolerance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(estimates, indent=2))
    print(f"Wrote {len(estimates)} estimates to {args.output}")


if __name__ == "__main__":
    main()
