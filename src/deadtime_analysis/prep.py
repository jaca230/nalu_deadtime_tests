from __future__ import annotations

from typing import List, Optional

import pandas as pd

from .constants import DEFAULT_DOUBLE_FACTOR, DEFAULT_SINGLE_FACTOR
from .loader import _get_active_channel_count


def _classify_rate(rate: float, single_threshold: float, double_threshold: float) -> str:
    if pd.isna(rate) or pd.isna(single_threshold) or pd.isna(double_threshold):
        return "mixed"
    if rate < single_threshold:
        return "single"
    if rate > double_threshold:
        return "double"
    return "mixed"


def build_dataframe(
    records: List[dict],
    single_factor: float = DEFAULT_SINGLE_FACTOR,
    double_factor: float = DEFAULT_DOUBLE_FACTOR,
) -> pd.DataFrame:
    """Flatten raw JSON records and derive classification fields."""
    rows = []
    for entry in records:
        capture = entry.get("capture_settings", {})
        raw_capture = capture.get("raw", {})
        search_meta = entry.get("search", {})
        double_pulse = entry.get("double_pulse", {})
        observed = entry.get("observed_rates", {})

        pulse_rate_hz = double_pulse.get("repetition_rate_hz")
        target_ratio = search_meta.get("target_ratio")
        single_threshold = (
            pulse_rate_hz * single_factor if pulse_rate_hz is not None else None
        )
        double_threshold = (
            pulse_rate_hz * double_factor if pulse_rate_hz is not None else None
        )
        target_line = (
            pulse_rate_hz * target_ratio
            if pulse_rate_hz is not None and target_ratio is not None
            else None
        )

        rows.append(
            {
                "timestamp": pd.to_datetime(entry.get("timestamp")),
                "run_number": entry.get("run_number"),
                "separation_ns": double_pulse.get("separation_ns"),
                "pulse_rate_hz": pulse_rate_hz,
                "target_ratio": target_ratio,
                "windows": capture.get("windows", raw_capture.get("windows")),
                "channel_count": _get_active_channel_count(entry),
                "observed_rate_hz": observed.get("events_per_second"),
                "expected_rate_hz": observed.get("expected_events_per_second"),
                "deadtime_fraction": observed.get("deadtime_fraction"),
                "search_iteration": search_meta.get("iteration"),
                "search_combo_index": search_meta.get("combo_index"),
                "search_low_ns": search_meta.get("low_ns"),
                "search_high_ns": search_meta.get("high_ns"),
                "single_threshold": single_threshold,
                "double_threshold": double_threshold,
                "target_line": target_line,
            }
        )

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["windows", "channel_count", "pulse_rate_hz"])
    df["tertiary_mode"] = df.apply(
        lambda row: _classify_rate(
            row.get("observed_rate_hz"),
            row.get("single_threshold"),
            row.get("double_threshold"),
        ),
        axis=1,
    )
    return df.sort_values(["pulse_rate_hz", "separation_ns"])
