from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd

from .loader import _get_active_channel_count


def _classify_pulse_region(
    observed_rate: Optional[float], pulse_rate_hz: Optional[float], num_pulses: Optional[int]
) -> Tuple[str, Optional[int], Optional[int], Optional[int]]:
    """
    Classify response using 25% bands around each integer pulse multiple.

    Returns:
        mode_label: human-readable label
        pulse_region: k if in pure k-pulse region else None
        mixed_lower: lower pulse count if in a mixed band else None
        mixed_upper: upper pulse count if in a mixed band else None
    """
    if observed_rate is None or pulse_rate_hz is None or num_pulses is None:
        return "mixed", None, None, None
    norm = observed_rate / pulse_rate_hz
    # Pure regions first
    for k in range(1, int(num_pulses) + 1):
        lower = max(0, k - 0.25)
        upper = k + 0.25
        if lower <= norm <= upper:
            label = "single" if k == 1 else ("all" if k == num_pulses else f"{k}-pulse")
            return label, k, None, None
    # Mixed regions between k and k+1
    for k in range(1, int(num_pulses)):
        lower = k + 0.25
        upper = k + 0.75
        if lower <= norm <= upper:
            return "mixed", None, k, k + 1
    # Above the top band: treat as all pulses
    label = "all" if num_pulses > 1 else "single"
    return label, num_pulses, None, None


def build_dataframe(
    records: List[dict],
    single_factor: float = 1.2,  # retained for API compatibility; unused in new logic
    double_factor: float = 1.8,  # retained for API compatibility; unused in new logic
) -> pd.DataFrame:
    """Flatten raw JSON records and derive classification fields."""
    rows = []
    for entry in records:
        capture = entry.get("capture_settings", {})
        raw_capture = capture.get("raw", {})
        search_meta = entry.get("search", {})
        pulse_meta = entry.get("pulse_sequence") or entry.get("double_pulse") or {}
        observed = entry.get("observed_rates", {})
        channel_rates = entry.get("channel_rates", {})
        custom_stats = entry.get("custom_stats", {})

        pulse_rate_hz = pulse_meta.get("repetition_rate_hz")
        num_pulses = (
            pulse_meta.get("num_pulses")
            or search_meta.get("num_pulses")
            or (2 if entry.get("double_pulse") else None)
        )
        expected_rate_hz = observed.get("expected_events_per_second")
        expected_pulses = (
            expected_rate_hz / pulse_rate_hz if pulse_rate_hz and expected_rate_hz else None
        )
        observed_rate_hz = observed.get("events_per_second")
        observed_ratio_vs_pulse = (
            observed_rate_hz / pulse_rate_hz if pulse_rate_hz and observed_rate_hz else None
        )
        observed_ratio_vs_expected = (
            observed_rate_hz / expected_rate_hz
            if expected_rate_hz and observed_rate_hz
            else None
        )
        target_ratio = search_meta.get("target_ratio")
        target_line = None
        if target_ratio is not None:
            if observed.get("ratio_vs_double_pulse") is not None:
                target_line = pulse_rate_hz * target_ratio if pulse_rate_hz else None
            elif expected_rate_hz is not None:
                target_line = expected_rate_hz * target_ratio
            else:
                target_line = pulse_rate_hz * target_ratio if pulse_rate_hz else None

        mode_label, pulse_region, mixed_lower, mixed_upper = _classify_pulse_region(
            observed_rate_hz, pulse_rate_hz, num_pulses
        )

        observed_channels_per_event = channel_rates.get("observed_channels_per_event")
        if observed_channels_per_event is None:
            observed_channels_per_event = custom_stats.get("channels_per_event")
        expected_channels_per_event = channel_rates.get("expected_channels_per_event")
        if expected_channels_per_event is None:
            expected_channels_per_event = _get_active_channel_count(entry)
        channel_ratio_vs_expected = channel_rates.get("ratio_vs_expected")
        if (
            channel_ratio_vs_expected is None
            and observed_channels_per_event is not None
            and expected_channels_per_event
        ):
            channel_ratio_vs_expected = observed_channels_per_event / expected_channels_per_event
        channel_ratio_threshold = channel_rates.get("threshold")
        if channel_ratio_threshold is None:
            channel_ratio_threshold = search_meta.get("channels_ratio_threshold")
        channel_ratio_pass = None
        if channel_ratio_vs_expected is not None and channel_ratio_threshold is not None:
            channel_ratio_pass = channel_ratio_vs_expected > channel_ratio_threshold
        channel_full_pass = None
        if observed_channels_per_event is not None and expected_channels_per_event is not None:
            channel_full_pass = observed_channels_per_event >= expected_channels_per_event

        rows.append(
            {
                "timestamp": pd.to_datetime(entry.get("timestamp")),
                "run_number": entry.get("run_number"),
                "separation_ns": pulse_meta.get("separation_ns"),
                "pulse_rate_hz": pulse_rate_hz,
                "num_pulses": num_pulses,
                "target_ratio": target_ratio,
                "windows": capture.get("windows", raw_capture.get("windows")),
                "channel_count": _get_active_channel_count(entry),
                "observed_rate_hz": observed_rate_hz,
                "expected_rate_hz": expected_rate_hz,
                "deadtime_fraction": observed.get("deadtime_fraction"),
                "search_iteration": search_meta.get("iteration"),
                "search_combo_index": search_meta.get("combo_index"),
                "search_low_ns": search_meta.get("low_ns"),
                "search_high_ns": search_meta.get("high_ns"),
                "target_line": target_line,
                "expected_pulses": expected_pulses,
                "observed_ratio_vs_pulse_rate": observed_ratio_vs_pulse,
                "observed_ratio_vs_expected": observed_ratio_vs_expected,
                "pulse_region": pulse_region,
                "mixed_lower_pulses": mixed_lower,
                "mixed_upper_pulses": mixed_upper,
                "tertiary_mode": mode_label,
                "observed_channels_per_event": observed_channels_per_event,
                "expected_channels_per_event": expected_channels_per_event,
                "channel_ratio_vs_expected": channel_ratio_vs_expected,
                "channel_ratio_threshold": channel_ratio_threshold,
                "channel_ratio_pass": channel_ratio_pass,
                "channel_full_pass": channel_full_pass,
            }
        )

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["windows", "channel_count", "pulse_rate_hz"])
    return df.sort_values(["pulse_rate_hz", "separation_ns"])
