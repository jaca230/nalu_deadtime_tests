from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd

from .constants import DEFAULT_PULSE_COUNT


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


def _latest_history(search: dict) -> dict:
    history = search.get("history") or []
    return history[-1] if history else {}


def build_dataframe(
    records: List[dict],
    *,
    default_num_pulses: int = DEFAULT_PULSE_COUNT,
) -> pd.DataFrame:
    """Flatten raw sampic JSON records and derive plotting-friendly fields."""
    rows = []
    for entry in records:
        params = entry.get("parameters", {})
        search = entry.get("search", {})
        lecroy = entry.get("lecroy", {})
        aggregate = entry.get("aggregate", {})
        readback = entry.get("readback", {})
        history_latest = _latest_history(search)

        pulse_rate_hz = (
            params.get("lecroy_frequency_hz")
            or lecroy.get("frequency_hz")
            or history_latest.get("frequency_hz")
            or search.get("frequency_hz")
        )
        # All current runs are double-pulse; keep a knob for future flexibility.
        num_pulses = default_num_pulses
        double_enabled = lecroy.get("double_pulse_enabled")
        if double_enabled is False:
            num_pulses = 1

        digitizer_rate_mhz = params.get("digitizer_rate_mhz") or readback.get(
            "sampling_frequency_mhz"
        )
        separation_ns = (
            params.get("current_delay_ns")
            or history_latest.get("applied_delay_ns")
            or search.get("current_delay_ns")
        )
        best_delay_ns = params.get("best_delay_ns") or search.get("best_delay_ns")
        search_min_ns = search.get("search_min_ns") or params.get("search_min_ns")
        search_max_ns = search.get("search_max_ns") or params.get("search_max_ns")
        ratio_threshold = search.get("ratio_threshold")

        observed_rate_hz = search.get("event_rate_hz") or aggregate.get("events_per_second")
        expected_rate_hz = (
            pulse_rate_hz * num_pulses if pulse_rate_hz is not None and num_pulses else None
        )
        observed_ratio_vs_pulse = (
            observed_rate_hz / pulse_rate_hz if observed_rate_hz and pulse_rate_hz else None
        )
        observed_ratio_vs_expected = (
            observed_rate_hz / expected_rate_hz if observed_rate_hz and expected_rate_hz else None
        )

        target_line = pulse_rate_hz * ratio_threshold if pulse_rate_hz and ratio_threshold else None

        mode_label, pulse_region, mixed_lower, mixed_upper = _classify_pulse_region(
            observed_rate_hz, pulse_rate_hz, num_pulses
        )

        rows.append(
            {
                "timestamp": pd.to_datetime(entry.get("timestamp")),
                "combo_key": entry.get("combo_key"),
                "board_index": params.get("board_index"),
                "digitizer_rate_mhz": digitizer_rate_mhz,
                "pulse_rate_hz": pulse_rate_hz,
                "num_pulses": num_pulses,
                "separation_ns": separation_ns,
                "best_delay_ns": best_delay_ns,
                "search_min_ns": search_min_ns,
                "search_max_ns": search_max_ns,
                "search_iteration": history_latest.get("iteration"),
                "double_detected": history_latest.get("double_detected"),
                "status": entry.get("status"),
                "observed_rate_hz": observed_rate_hz,
                "expected_rate_hz": expected_rate_hz,
                "observed_ratio_vs_pulse_rate": observed_ratio_vs_pulse,
                "observed_ratio_vs_expected": observed_ratio_vs_expected,
                "target_ratio": ratio_threshold,
                "target_line": target_line,
                "hit_ratio_vs_frequency": history_latest.get("hit_ratio_vs_frequency"),
                "ratio_vs_frequency": history_latest.get("ratio_vs_frequency"),
                "pulse_region": pulse_region,
                "mixed_lower_pulses": mixed_lower,
                "mixed_upper_pulses": mixed_upper,
                "tertiary_mode": mode_label,
            }
        )

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["digitizer_rate_mhz", "pulse_rate_hz", "separation_ns"])
    return df.sort_values(["pulse_rate_hz", "digitizer_rate_mhz", "separation_ns"])
