from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .constants import DEFAULT_DOUBLE_FACTOR, DEFAULT_SINGLE_FACTOR
from .loader import load_records
from .models import ClassificationThresholds
from .plotting import (
    apply_pulse_regions,
    dedup_legend,
    grouped_shades_for_rate,
    set_log2_with_decade_ticks,
)
from .prep import build_dataframe


class DeadtimeAnalysis:
    """High-level helper for loading, classifying, and plotting deadtime sweeps."""

    def __init__(
        self,
        df: pd.DataFrame,
        single_factor: float = DEFAULT_SINGLE_FACTOR,
        double_factor: float = DEFAULT_DOUBLE_FACTOR,
    ):
        self.single_factor = single_factor
        self.double_factor = double_factor
        self.df = df

    def _channel_label(self, channel_count: float) -> str:
        count = int(channel_count)
        return f"{count} channels"

    def _pulse_label(self, num_pulses: Optional[float]) -> str:
        if pd.isna(num_pulses):
            return "unknown pulses"
        count = int(num_pulses)
        return f"{count} pulse{'s' if count != 1 else ''}"

    def _tertiary_label(self, mode: str, num_pulses: Optional[int]) -> str:
        if num_pulses and num_pulses > 2 and mode == "double":
            return "all"
        return mode

    def _tertiary_plot_value(self, row: pd.Series, num_pulses: Optional[int]) -> str:
        """Map raw classification to plot categories."""
        if num_pulses:
            if row.get("pulse_region") == num_pulses:
                return "all pulses seen"
            if (
                row.get("mixed_lower_pulses") == num_pulses - 1
                and row.get("mixed_upper_pulses") == num_pulses
            ):
                return "sometimes all pulses seen"
            return "not all pulses seen"
        # Fallback if num_pulses missing
        mode = row.get("tertiary_mode")
        if mode == "double":
            return "all pulses seen"
        if mode == "single":
            return "not all pulses seen"
        return "sometimes all pulses seen"

    def _annotate_no_signal_channel_runs(
        self, 
        ax: plt.Axes, 
        show: bool = True,
        position: Tuple[float, float] = (0.01, 0.02),
        va: str = "bottom"
    ) -> None:
        if not show:
            return
        note = "32-channel runs had no injected signals on channels 16-31 (control)."
        ax.text(
            position[0],
            position[1],
            note,
            transform=ax.transAxes,
            fontsize=9,
            ha="left",
            va=va,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.65},
        )

    @classmethod
    def from_jsonl(
        cls,
        paths: Sequence[str],
        single_factor: float = DEFAULT_SINGLE_FACTOR,
        double_factor: float = DEFAULT_DOUBLE_FACTOR,
    ) -> "DeadtimeAnalysis":
        records = load_records(paths)
        df = build_dataframe(records, single_factor=single_factor, double_factor=double_factor)
        return cls(df, single_factor=single_factor, double_factor=double_factor)

    # ---- Data slices ----------------------------------------------------- #
    def subset(self, pulse_rate_hz: float, num_pulses: Optional[int] = None) -> pd.DataFrame:
        df = self.df[np.isclose(self.df["pulse_rate_hz"], pulse_rate_hz)]
        if num_pulses is not None:
            df = df[df["num_pulses"] == num_pulses]
        return df.copy()

    def _target_line(self, pulse_rate_hz: float, num_pulses: Optional[int]) -> Optional[float]:
        df = self.subset(pulse_rate_hz, num_pulses=num_pulses)
        target_vals = df["target_line"].dropna().unique()
        return target_vals[0] if len(target_vals) else None

    def converged_table(self) -> pd.DataFrame:
        conv = (
            self.df.sort_values("search_iteration")
            .groupby(
                ["pulse_rate_hz", "channel_count", "windows", "num_pulses"], as_index=False
            )
            .tail(1)
        )
        return conv.rename(
            columns={
                "separation_ns": "converged_deadtime_ns",
                "search_low_ns": "converged_lower_bound_ns",
                "search_high_ns": "converged_upper_bound_ns",
            }
        )

    def min_double_table(
        self, require_channel_ratio_pass: bool = False, require_full_channels: bool = False
    ) -> pd.DataFrame:
        rows = []
        channel_gate_column = None
        if require_full_channels:
            channel_gate_column = "channel_full_pass"
        elif require_channel_ratio_pass:
            channel_gate_column = "channel_ratio_pass"
        for (rate, channel_count, windows, num_pulses), group_df in self.df.groupby(
            ["pulse_rate_hz", "channel_count", "windows", "num_pulses"]
        ):
            if pd.isna(num_pulses):
                continue
            target_region = group_df[group_df["pulse_region"] == num_pulses]
            if channel_gate_column is not None:
                target_region = target_region[target_region[channel_gate_column] == True]
            target_region = target_region.sort_values("separation_ns")
            if target_region.empty:
                continue
            min_row = target_region.iloc[0]
            prev_region = group_df[group_df["pulse_region"] == (num_pulses - 1)]
            if channel_gate_column is not None:
                lower_candidates = group_df[group_df["separation_ns"] < min_row["separation_ns"]]
                lower_candidates = lower_candidates[
                    (lower_candidates["pulse_region"] == (num_pulses - 1))
                    | (lower_candidates[channel_gate_column] == False)
                    | (lower_candidates[channel_gate_column].isna())
                ]
                lower_prev = (
                    lower_candidates["separation_ns"].max() if not lower_candidates.empty else None
                )
            else:
                lower_prev = prev_region["separation_ns"].max() if not prev_region.empty else None
            search_lower = min_row.get("search_low_ns")
            lower_bound = lower_prev if lower_prev is not None else search_lower
            rows.append(
                {
                    "pulse_rate_hz": rate,
                    "channel_count": channel_count,
                    "windows": windows,
                    "num_pulses": num_pulses,
                    "min_double_deadtime_ns": min_row["separation_ns"],
                    "min_double_lower_bound_ns": lower_bound,
                }
            )
        return pd.DataFrame(rows)

    # ---- Rate plots ------------------------------------------------------ #
    def plot_rate_vs_separation_by_channels(
        self,
        pulse_rate_hz: float,
        num_pulses: Optional[int] = None,
        highlight_separations_ns: Optional[List[float]] = None,
        highlight_text: Optional[str] = None,
        show_stars: bool = True,
        show_vertical_bounds: bool = True,
        show_notes: bool = True,
        deadtime_range_ns: Optional[Tuple[float, float]] = None,
        deadtime_range_text: Optional[str] = None,
        deadtime_estimate_ns: Optional[float] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
    ) -> None:
        data = self.subset(pulse_rate_hz, num_pulses=num_pulses)
        if data.empty:
            raise ValueError(f"No data for pulse_rate_hz={pulse_rate_hz}")
        target_line = self._target_line(pulse_rate_hz, num_pulses)
        for channel_count, channel_df in data.groupby("channel_count"):
            fig, ax = plt.subplots(figsize=(10, 6))
            x_min = channel_df["separation_ns"].min() * 0.95
            x_max = channel_df["separation_ns"].max() * 1.05
            if deadtime_estimate_ns is not None:
                x_min = min(x_min, deadtime_estimate_ns * 0.95)
                x_max = max(x_max, deadtime_estimate_ns * 1.05)
            y_min_obs = channel_df["observed_rate_hz"].min() * 0.95
            y_max_obs = channel_df["observed_rate_hz"].max() * 1.05
            ax.set_xlim(x_min, x_max)
            # Use custom y limits if provided, otherwise use auto
            y_min_use = y_min if y_min is not None else y_min_obs
            y_max_use = y_max if y_max is not None else y_max_obs * 1.1
            ax.set_ylim(y_min_use, y_max_use)
            for (windows, pulse_count), window_df in channel_df.groupby(["windows", "num_pulses"]):
                sorted_df = window_df.sort_values("separation_ns")
                ax.plot(
                    sorted_df["separation_ns"],
                    sorted_df["observed_rate_hz"],
                    marker="o",
                    label=f"{int(windows)} windows",
                )
            # Vertical highlight region for deadtime range
            if show_vertical_bounds and deadtime_range_ns and len(deadtime_range_ns) == 2:
                lb, resp = deadtime_range_ns
                if lb is not None and resp is not None:
                    ax.axvspan(lb, resp, alpha=0.2, color='yellow', zorder=0)
                    # Add dashed vertical lines at boundaries
                    ax.axvline(lb, color='black', linestyle='--', linewidth=1.5, zorder=4)
                    ax.axvline(resp, color='black', linestyle='--', linewidth=1.5, zorder=4)
            if deadtime_estimate_ns is not None:
                ax.axvline(
                    deadtime_estimate_ns,
                    color="tab:red",
                    linestyle=":",
                    linewidth=2.0,
                    zorder=4,
                    label="Deadtime estimate",
                )
            
            if show_stars and highlight_separations_ns:
                for sep in highlight_separations_ns:
                    y_vals = channel_df[channel_df["separation_ns"] == sep]["observed_rate_hz"]
                    if y_vals.empty:
                        continue
                    ax.scatter(
                        [sep],
                        [y_vals.mean()],
                        marker="*",
                        s=150,
                        color="black",
                        zorder=5,
                        label=None,
                    )
            
            # Notes in top left
            notes_y_offset = 0.98
            if show_notes and deadtime_range_text:
                ax.text(
                    0.01,
                    notes_y_offset,
                    deadtime_range_text,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=9,
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.65},
                )
                notes_y_offset -= 0.05
            
            if show_notes:
                self._annotate_no_signal_channel_runs(
                    ax, 
                    show=True, 
                    position=(0.01, notes_y_offset), 
                    va="top"
                )
            
            if num_pulses:
                apply_pulse_regions(ax, pulse_rate_hz, int(num_pulses), target_line=target_line)
            # Re-apply custom y limits after apply_pulse_regions (which may have changed them)
            if y_min is not None or y_max is not None:
                current_ylim = ax.get_ylim()
                y_min_final = y_min if y_min is not None else current_ylim[0]
                y_max_final = y_max if y_max is not None else current_ylim[1]
                ax.set_ylim(y_min_final, y_max_final)
            set_log2_with_decade_ticks(ax, "x", unit="ns")
            if num_pulses is not None:
                pulse_note = f", {self._pulse_label(num_pulses)}"
            else:
                uniq = channel_df["num_pulses"].dropna().unique()
                pulse_note = (
                    f", {self._pulse_label(uniq[0])}" if len(uniq) == 1 else ", mixed pulse counts"
                )
            ax.set_title(
                f"Observed rate vs. separation ({self._channel_label(channel_count)}, pulser={pulse_rate_hz:.0f} Hz{pulse_note})"
            )
            ax.set_xlabel("Pulse separation (ns)")
            ax.set_ylabel("Observed rate (events/s)")
            ax.grid(True, linestyle="--", alpha=0.5)
            dedup_legend(ax, title="Capture windows")
            plt.show()

    def plot_rate_vs_separation_by_windows(
        self,
        pulse_rate_hz: float,
        num_pulses: Optional[int] = None,
        highlight_separations_ns: Optional[List[float]] = None,
        highlight_text: Optional[str] = None,
        show_stars: bool = True,
        show_vertical_bounds: bool = True,
        show_notes: bool = True,
        deadtime_range_ns: Optional[Tuple[float, float]] = None,
        deadtime_range_text: Optional[str] = None,
        deadtime_estimate_ns: Optional[float] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
    ) -> None:
        data = self.subset(pulse_rate_hz, num_pulses=num_pulses)
        if data.empty:
            raise ValueError(f"No data for pulse_rate_hz={pulse_rate_hz}")
        target_line = self._target_line(pulse_rate_hz, num_pulses)
        for windows, window_df in data.groupby("windows"):
            fig, ax = plt.subplots(figsize=(10, 6))
            x_min = window_df["separation_ns"].min() * 0.95
            x_max = window_df["separation_ns"].max() * 1.05
            if deadtime_estimate_ns is not None:
                x_min = min(x_min, deadtime_estimate_ns * 0.95)
                x_max = max(x_max, deadtime_estimate_ns * 1.05)
            y_min_obs = window_df["observed_rate_hz"].min() * 0.95
            y_max_obs = window_df["observed_rate_hz"].max() * 1.05
            ax.set_xlim(x_min, x_max)
            # Use custom y limits if provided, otherwise use auto
            y_min_use = y_min if y_min is not None else y_min_obs
            y_max_use = y_max if y_max is not None else y_max_obs * 1.1
            ax.set_ylim(y_min_use, y_max_use)
            for (channel_count, pulse_count), channel_df in window_df.groupby(
                ["channel_count", "num_pulses"]
            ):
                sorted_df = channel_df.sort_values("separation_ns")
                ax.plot(
                    sorted_df["separation_ns"],
                    sorted_df["observed_rate_hz"],
                    marker="o",
                    label=self._channel_label(channel_count),
                )
            # Vertical highlight region for deadtime range
            if show_vertical_bounds and deadtime_range_ns and len(deadtime_range_ns) == 2:
                lb, resp = deadtime_range_ns
                if lb is not None and resp is not None:
                    ax.axvspan(lb, resp, alpha=0.2, color='yellow', zorder=0)
                    # Add dashed vertical lines at boundaries
                    ax.axvline(lb, color='black', linestyle='--', linewidth=1.5, zorder=4)
                    ax.axvline(resp, color='black', linestyle='--', linewidth=1.5, zorder=4)
            if deadtime_estimate_ns is not None:
                ax.axvline(
                    deadtime_estimate_ns,
                    color="tab:red",
                    linestyle=":",
                    linewidth=2.0,
                    zorder=4,
                    label="Deadtime estimate",
                )
            
            if show_stars and highlight_separations_ns:
                for sep in highlight_separations_ns:
                    y_vals = window_df[window_df["separation_ns"] == sep]["observed_rate_hz"]
                    if y_vals.empty:
                        continue
                    ax.scatter(
                        [sep],
                        [y_vals.mean()],
                        marker="*",
                        s=150,
                        color="black",
                        zorder=5,
                        label=None,
                    )
            
            # Notes in top left
            notes_y_offset = 0.98
            if show_notes and deadtime_range_text:
                ax.text(
                    0.01,
                    notes_y_offset,
                    deadtime_range_text,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=9,
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.65},
                )
                notes_y_offset -= 0.05
            
            if show_notes:
                self._annotate_no_signal_channel_runs(
                    ax, 
                    show=True, 
                    position=(0.01, notes_y_offset), 
                    va="top"
                )
            
            if num_pulses:
                apply_pulse_regions(ax, pulse_rate_hz, int(num_pulses), target_line=target_line)
            # Re-apply custom y limits after apply_pulse_regions (which may have changed them)
            if y_min is not None or y_max is not None:
                current_ylim = ax.get_ylim()
                y_min_final = y_min if y_min is not None else current_ylim[0]
                y_max_final = y_max if y_max is not None else current_ylim[1]
                ax.set_ylim(y_min_final, y_max_final)
            set_log2_with_decade_ticks(ax, "x", unit="ns")
            if num_pulses is not None:
                pulse_note = f", {self._pulse_label(num_pulses)}"
            else:
                uniq = window_df["num_pulses"].dropna().unique()
                pulse_note = (
                    f", {self._pulse_label(uniq[0])}" if len(uniq) == 1 else ", mixed pulse counts"
                )
            ax.set_title(
                f"Observed rate vs. separation (windows={int(windows)}, pulser={pulse_rate_hz:.0f} Hz{pulse_note})"
            )
            ax.set_xlabel("Pulse separation (ns)")
            ax.set_ylabel("Observed rate (events/s)")
            ax.grid(True, linestyle="--", alpha=0.5)
            dedup_legend(ax, title="Active channels")
            plt.show()

    def plot_channels_per_event_vs_separation_by_channels(
        self,
        pulse_rate_hz: float,
        num_pulses: Optional[int] = None,
        show_expected: bool = True,
        show_threshold: bool = True,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
    ) -> None:
        data = self.subset(pulse_rate_hz, num_pulses=num_pulses)
        data = data.dropna(subset=["observed_channels_per_event"])
        if data.empty:
            raise ValueError(f"No channel-rate data for pulse_rate_hz={pulse_rate_hz}")
        for channel_count, channel_df in data.groupby("channel_count"):
            fig, ax = plt.subplots(figsize=(10, 6))
            x_min = channel_df["separation_ns"].min() * 0.95
            x_max = channel_df["separation_ns"].max() * 1.05
            y_min_obs = channel_df["observed_channels_per_event"].min() * 0.95
            y_max_obs = channel_df["observed_channels_per_event"].max() * 1.05
            ax.set_xlim(x_min, x_max)
            y_min_use = y_min if y_min is not None else y_min_obs
            y_max_use = y_max if y_max is not None else y_max_obs
            ax.set_ylim(y_min_use, y_max_use)
            for windows, window_df in channel_df.groupby("windows"):
                sorted_df = window_df.sort_values("separation_ns")
                ax.plot(
                    sorted_df["separation_ns"],
                    sorted_df["observed_channels_per_event"],
                    marker="o",
                    label=f"{int(windows)} windows",
                )
            expected_vals = channel_df["expected_channels_per_event"].dropna().unique()
            expected = expected_vals[0] if len(expected_vals) == 1 else None
            if show_expected and expected is not None:
                ax.axhline(
                    expected,
                    color="black",
                    linestyle="--",
                    linewidth=1.2,
                    label=f"Expected ({expected:.0f})",
                )
            threshold_vals = channel_df["channel_ratio_threshold"].dropna().unique()
            threshold = threshold_vals[0] if len(threshold_vals) == 1 else None
            if show_threshold and expected is not None and threshold is not None:
                ax.axhline(
                    expected * threshold,
                    color="tab:red",
                    linestyle=":",
                    linewidth=1.8,
                    label=f"Threshold ({threshold:.2f}×)",
                )
            set_log2_with_decade_ticks(ax, "x", unit="ns")
            if num_pulses is not None:
                pulse_note = f", {self._pulse_label(num_pulses)}"
            else:
                uniq = channel_df["num_pulses"].dropna().unique()
                pulse_note = (
                    f", {self._pulse_label(uniq[0])}" if len(uniq) == 1 else ", mixed pulse counts"
                )
            ax.set_title(
                f"Observed channels/event vs. separation ({self._channel_label(channel_count)}, pulser={pulse_rate_hz:.0f} Hz{pulse_note})"
            )
            ax.set_xlabel("Pulse separation (ns)")
            ax.set_ylabel("Observed channels/event")
            ax.grid(True, linestyle="--", alpha=0.5)
            dedup_legend(ax, title="Capture windows")
            plt.show()

    def plot_channels_per_event_vs_separation_by_windows(
        self,
        pulse_rate_hz: float,
        num_pulses: Optional[int] = None,
        show_expected: bool = True,
        show_threshold: bool = True,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
    ) -> None:
        data = self.subset(pulse_rate_hz, num_pulses=num_pulses)
        data = data.dropna(subset=["observed_channels_per_event"])
        if data.empty:
            raise ValueError(f"No channel-rate data for pulse_rate_hz={pulse_rate_hz}")
        for windows, window_df in data.groupby("windows"):
            fig, ax = plt.subplots(figsize=(10, 6))
            x_min = window_df["separation_ns"].min() * 0.95
            x_max = window_df["separation_ns"].max() * 1.05
            y_min_obs = window_df["observed_channels_per_event"].min() * 0.95
            y_max_obs = window_df["observed_channels_per_event"].max() * 1.05
            ax.set_xlim(x_min, x_max)
            y_min_use = y_min if y_min is not None else y_min_obs
            y_max_use = y_max if y_max is not None else y_max_obs
            ax.set_ylim(y_min_use, y_max_use)
            for channel_count, channel_df in window_df.groupby("channel_count"):
                sorted_df = channel_df.sort_values("separation_ns")
                ax.plot(
                    sorted_df["separation_ns"],
                    sorted_df["observed_channels_per_event"],
                    marker="o",
                    label=self._channel_label(channel_count),
                )
            expected_vals = window_df["expected_channels_per_event"].dropna().unique()
            expected = expected_vals[0] if len(expected_vals) == 1 else None
            if show_expected and expected is not None:
                ax.axhline(
                    expected,
                    color="black",
                    linestyle="--",
                    linewidth=1.2,
                    label=f"Expected ({expected:.0f})",
                )
            threshold_vals = window_df["channel_ratio_threshold"].dropna().unique()
            threshold = threshold_vals[0] if len(threshold_vals) == 1 else None
            if show_threshold and expected is not None and threshold is not None:
                ax.axhline(
                    expected * threshold,
                    color="tab:red",
                    linestyle=":",
                    linewidth=1.8,
                    label=f"Threshold ({threshold:.2f}×)",
                )
            set_log2_with_decade_ticks(ax, "x", unit="ns")
            if num_pulses is not None:
                pulse_note = f", {self._pulse_label(num_pulses)}"
            else:
                uniq = window_df["num_pulses"].dropna().unique()
                pulse_note = (
                    f", {self._pulse_label(uniq[0])}" if len(uniq) == 1 else ", mixed pulse counts"
                )
            ax.set_title(
                f"Observed channels/event vs. separation (windows={int(windows)}, pulser={pulse_rate_hz:.0f} Hz{pulse_note})"
            )
            ax.set_xlabel("Pulse separation (ns)")
            ax.set_ylabel("Observed channels/event")
            ax.grid(True, linestyle="--", alpha=0.5)
            dedup_legend(ax, title="Active channels")
            plt.show()

    # ---- Derived separation plots --------------------------------------- #
    def plot_converged_vs_windows(
        self,
        pulse_rates: Optional[Iterable[float]] = None,
        pulse_counts: Optional[Iterable[int]] = None,
    ) -> None:
        pulse_rates = list(pulse_rates) if pulse_rates else sorted(self.df["pulse_rate_hz"].unique())
        pulse_counts = list(pulse_counts) if pulse_counts else sorted(
            self.df["num_pulses"].dropna().unique()
        )
        conv = self.converged_table()
        for pulse_count in pulse_counts:
            subset_count = conv[conv["num_pulses"] == pulse_count]
            if subset_count.empty:
                continue
            fig, ax = plt.subplots(figsize=(12, 6))
            for idx, pulse_rate in enumerate(pulse_rates):
                subset = subset_count[subset_count["pulse_rate_hz"] == pulse_rate]
                if subset.empty:
                    continue
                channel_counts = sorted(subset["channel_count"].unique())
                palette = grouped_shades_for_rate(idx, channel_counts)
                for channel_count, channel_df in subset.groupby("channel_count"):
                    sorted_df = channel_df.sort_values("windows")
                    color = palette.get(channel_count, f"C{idx}")
                    lower = sorted_df["converged_lower_bound_ns"] / 1000.0
                    upper = sorted_df["converged_upper_bound_ns"] / 1000.0
                    y_vals = sorted_df["converged_deadtime_ns"] / 1000.0
                    yerr = np.vstack(
                        [
                            np.nan_to_num((y_vals - lower).to_numpy(dtype=float), nan=0.0),
                            np.nan_to_num((upper - y_vals).to_numpy(dtype=float), nan=0.0),
                        ]
                    )
                    ax.errorbar(
                        sorted_df["windows"],
                        y_vals,
                        yerr=yerr,
                        fmt="o-",
                        linewidth=1.8,
                        color=color,
                        label=f"{int(channel_count)} ch @ {pulse_rate:.0f}Hz",
                        capsize=4,
                    )
            ax.set_xlabel("Capture windows")
            ax.set_ylabel("Converged deadtime (µs)")
            ax.set_xscale("log", base=2)
            set_log2_with_decade_ticks(ax, "y", unit="µs")
            ax.set_title(
                f"Converged deadtime vs. capture windows (pulse count={int(pulse_count)}; axes log2, y decade ticks)"
            )
            ax.grid(True, linestyle="--", alpha=0.5)
            self._annotate_no_signal_channel_runs(ax)
            dedup_legend(ax, title="Channel / pulser")
            plt.show()

    def plot_converged_vs_channels(
        self,
        pulse_rates: Optional[Iterable[float]] = None,
        pulse_counts: Optional[Iterable[int]] = None,
    ) -> None:
        pulse_rates = list(pulse_rates) if pulse_rates else sorted(self.df["pulse_rate_hz"].unique())
        pulse_counts = list(pulse_counts) if pulse_counts else sorted(
            self.df["num_pulses"].dropna().unique()
        )
        conv = self.converged_table()
        for pulse_count in pulse_counts:
            subset_count = conv[conv["num_pulses"] == pulse_count]
            if subset_count.empty:
                continue
            fig, ax = plt.subplots(figsize=(12, 6))
            for idx, pulse_rate in enumerate(pulse_rates):
                subset = subset_count[subset_count["pulse_rate_hz"] == pulse_rate]
                if subset.empty:
                    continue
                window_counts = sorted(subset["windows"].unique())
                palette = grouped_shades_for_rate(idx, window_counts)
                for windows, window_df in subset.groupby("windows"):
                    sorted_df = window_df.sort_values("channel_count")
                    color = palette.get(windows, f"C{idx}")
                    lower = sorted_df["converged_lower_bound_ns"] / 1000.0
                    upper = sorted_df["converged_upper_bound_ns"] / 1000.0
                    y_vals = sorted_df["converged_deadtime_ns"] / 1000.0
                    yerr = np.vstack(
                        [
                            np.nan_to_num((y_vals - lower).to_numpy(dtype=float), nan=0.0),
                            np.nan_to_num((upper - y_vals).to_numpy(dtype=float), nan=0.0),
                        ]
                    )
                    ax.errorbar(
                        sorted_df["channel_count"],
                        y_vals,
                        yerr=yerr,
                        fmt="o-",
                        linewidth=1.8,
                        color=color,
                        label=f"{int(windows)} win @ {pulse_rate:.0f}Hz",
                        capsize=4,
                    )
            ax.set_xlabel("Active channels")
            ax.set_ylabel("Converged deadtime (µs)")
            ax.set_xscale("log", base=2)
            set_log2_with_decade_ticks(ax, "y", unit="µs")
            ax.set_title(
                f"Converged deadtime vs. active channels (pulse count={int(pulse_count)}; axes log2, y decade ticks)"
            )
            ax.grid(True, linestyle="--", alpha=0.5)
            self._annotate_no_signal_channel_runs(ax)
            dedup_legend(ax, title="Windows / pulser")
            plt.show()

    def plot_min_double_vs_windows(
        self,
        pulse_rates: Optional[Iterable[float]] = None,
        print_fits: bool = True,
        pulse_counts: Optional[Iterable[int]] = None,
    ) -> None:
        pulse_rates = list(pulse_rates) if pulse_rates else sorted(self.df["pulse_rate_hz"].unique())
        pulse_counts = list(pulse_counts) if pulse_counts else sorted(
            self.df["num_pulses"].dropna().unique()
        )
        min_double = self.min_double_table()
        for pulse_count in pulse_counts:
            subset_count = min_double[min_double["num_pulses"] == pulse_count]
            if subset_count.empty:
                continue
            fig, ax = plt.subplots(figsize=(12, 6))
            fit_lines: List[Tuple[str, float, float]] = []
            for idx, pulse_rate in enumerate(pulse_rates):
                subset = subset_count[subset_count["pulse_rate_hz"] == pulse_rate]
                if subset.empty:
                    continue
                channel_counts = sorted(subset["channel_count"].unique())
                palette = grouped_shades_for_rate(idx, channel_counts)
                for channel_count, channel_df in subset.groupby("channel_count"):
                    sorted_df = channel_df.sort_values("windows")
                    if sorted_df.empty:
                        continue
                    color = palette.get(channel_count, f"C{idx}")
                    label = f"{int(channel_count)} ch @ {pulse_rate:.0f}Hz"
                    x_vals = (sorted_df["windows"] * 32).to_numpy(dtype=float)
                    y_vals = (sorted_df["min_double_deadtime_ns"] / 1000.0).to_numpy(dtype=float)
                    lower = sorted_df["min_double_lower_bound_ns"] / 1000.0
                    lower_err = y_vals - lower.to_numpy(dtype=float)
                    lower_err = np.nan_to_num(lower_err, nan=0.0, posinf=0.0, neginf=0.0)
                    lower_err = np.clip(lower_err, a_min=0, a_max=None)
                    upper_err = np.zeros_like(lower_err)
                    yerr = np.vstack([lower_err, upper_err])
                    ax.errorbar(
                        x_vals,
                        y_vals,
                        yerr=yerr,
                        color=color,
                        linewidth=2,
                        marker="s",
                        label=label,
                        alpha=0.8,
                        capsize=4,
                    )
                    if sorted_df["windows"].nunique() > 1:
                        slope, intercept = np.polyfit(x_vals, y_vals, 1)
                        fit_lines.append((label, slope, intercept))
            ax.set_xlabel("Capture samples (windows × 32)")
            ax.set_ylabel("Minimum separation with multi-pulse response (µs)")
            ax.set_xscale("log", base=2)
            set_log2_with_decade_ticks(ax, "y", unit="µs")
            ax.set_title(
                f"Minimum multi-pulse separation vs. capture samples (pulse count={int(pulse_count)}; axes log2, y decade ticks)"
            )
            ax.grid(True, linestyle="--", alpha=0.5)
            self._annotate_no_signal_channel_runs(ax)
            dedup_legend(ax, title="Channel / pulser")
            plt.show()
            if print_fits:
                if fit_lines:
                    print(
                        f"Linear fits for minimum multi-pulse separation (windows on x-axis, pulse count={int(pulse_count)}):"
                    )
                    for label, slope, intercept in fit_lines:
                        print(f"  {label}: separation = {slope:.3f} * samples + {intercept:.2f}")
                else:
                    print(f"No linear fits computed (pulse count={int(pulse_count)}).")

    def plot_min_double_vs_channels(
        self,
        pulse_rates: Optional[Iterable[float]] = None,
        print_fits: bool = True,
        pulse_counts: Optional[Iterable[int]] = None,
    ) -> None:
        pulse_rates = list(pulse_rates) if pulse_rates else sorted(self.df["pulse_rate_hz"].unique())
        pulse_counts = list(pulse_counts) if pulse_counts else sorted(
            self.df["num_pulses"].dropna().unique()
        )
        min_double = self.min_double_table()
        for pulse_count in pulse_counts:
            subset_count = min_double[min_double["num_pulses"] == pulse_count]
            if subset_count.empty:
                continue
            fig, ax = plt.subplots(figsize=(12, 6))
            fit_lines: List[Tuple[str, float, float]] = []
            for idx, pulse_rate in enumerate(pulse_rates):
                subset = subset_count[subset_count["pulse_rate_hz"] == pulse_rate]
                if subset.empty:
                    continue
                window_counts = sorted(subset["windows"].unique())
                palette = grouped_shades_for_rate(idx, window_counts)
                for windows, window_df in subset.groupby("windows"):
                    sorted_df = window_df.sort_values("channel_count")
                    if sorted_df.empty:
                        continue
                    color = palette.get(windows, f"C{idx}")
                    label = f"{int(windows)} win @ {pulse_rate:.0f}Hz"
                    x_vals = sorted_df["channel_count"].to_numpy(dtype=float)
                    y_vals = (sorted_df["min_double_deadtime_ns"] / 1000.0).to_numpy(dtype=float)
                    lower = sorted_df["min_double_lower_bound_ns"] / 1000.0
                    lower_err = y_vals - lower.to_numpy(dtype=float)
                    lower_err = np.nan_to_num(lower_err, nan=0.0, posinf=0.0, neginf=0.0)
                    lower_err = np.clip(lower_err, a_min=0, a_max=None)
                    upper_err = np.zeros_like(lower_err)
                    yerr = np.vstack([lower_err, upper_err])
                    ax.errorbar(
                        x_vals,
                        y_vals,
                        yerr=yerr,
                        color=color,
                        linewidth=2,
                        marker="s",
                        label=label,
                        alpha=0.8,
                        capsize=4,
                    )
                    if sorted_df["channel_count"].nunique() > 1:
                        slope, intercept = np.polyfit(x_vals, y_vals, 1)
                        fit_lines.append((label, slope, intercept))
            ax.set_xlabel("Active channels")
            ax.set_ylabel("Minimum separation with multi-pulse response (µs)")
            ax.set_xscale("log", base=2)
            set_log2_with_decade_ticks(ax, "y", unit="µs")
            ax.set_title(
                f"Minimum multi-pulse separation vs. active channels (pulse count={int(pulse_count)}; axes log2, y decade ticks)"
            )
            ax.grid(True, linestyle="--", alpha=0.5)
            self._annotate_no_signal_channel_runs(ax)
            dedup_legend(ax, title="Windows / pulser")
            plt.show()
            if print_fits:
                if fit_lines:
                    print(
                        f"Linear fits for minimum multi-pulse separation (channels on x-axis, pulse count={int(pulse_count)}):"
                    )
                    for label, slope, intercept in fit_lines:
                        print(f"  {label}: separation = {slope:.3f} * channels + {intercept:.2f}")
            else:
                print(f"No linear fits computed (pulse count={int(pulse_count)}).")

    # ---- Pulse-count comparisons --------------------------------------- #
    def plot_min_double_vs_pulses_by_channels(self, pulse_rate_hz: float, windows: int) -> None:
        """Y: min separation (µs). X: number of pulses. Curves: channel counts. Hold pulser & windows fixed."""
        table = self.min_double_table()
        subset = table[(table["pulse_rate_hz"] == pulse_rate_hz) & (table["windows"] == windows)]
        if subset.empty:
            raise ValueError("No data for requested pulse_rate/windows")
        fig, ax = plt.subplots(figsize=(10, 6))
        channel_counts = sorted(subset["channel_count"].unique())
        palette = {ch: f"C{idx}" for idx, ch in enumerate(channel_counts)}
        for channel_count, channel_df in subset.groupby("channel_count"):
            sorted_df = channel_df.sort_values("num_pulses")
            ax.plot(
                sorted_df["num_pulses"],
                sorted_df["min_double_deadtime_ns"] / 1000.0,
                marker="o",
                linewidth=2,
                color=palette.get(channel_count, None),
                label=self._channel_label(channel_count),
            )
        ax.set_xlabel("Number of pulses")
        ax.set_ylabel("Minimum separation with multi-pulse response (µs)")
        set_log2_with_decade_ticks(ax, "y", unit="µs")
        ax.set_title(
            f"Min multi-pulse separation vs. pulse count (windows={int(windows)}, pulser={pulse_rate_hz:.0f} Hz)"
        )
        ax.grid(True, linestyle="--", alpha=0.5)
        dedup_legend(ax, title="Active channels")
        ax.set_xticks(sorted(subset["num_pulses"].unique()))
        plt.show()

    def plot_min_double_vs_pulses_by_pulser(self, channel_count: int, windows: int) -> None:
        """Y: min separation (µs). X: number of pulses. Curves: pulser rates. Hold channels & windows fixed."""
        table = self.min_double_table()
        subset = table[(table["channel_count"] == channel_count) & (table["windows"] == windows)]
        if subset.empty:
            raise ValueError("No data for requested channel_count/windows")
        fig, ax = plt.subplots(figsize=(10, 6))
        pulse_rates = sorted(subset["pulse_rate_hz"].unique())
        palette = {rate: f"C{idx}" for idx, rate in enumerate(pulse_rates)}
        for pulse_rate, rate_df in subset.groupby("pulse_rate_hz"):
            sorted_df = rate_df.sort_values("num_pulses")
            ax.plot(
                sorted_df["num_pulses"],
                sorted_df["min_double_deadtime_ns"] / 1000.0,
                marker="o",
                linewidth=2,
                color=palette.get(pulse_rate, None),
                label=f"{pulse_rate:.0f} Hz",
            )
        ax.set_xlabel("Number of pulses")
        ax.set_ylabel("Minimum separation with multi-pulse response (µs)")
        set_log2_with_decade_ticks(ax, "y", unit="µs")
        ax.set_title(
            f"Min multi-pulse separation vs. pulse count ({self._channel_label(channel_count)}, windows={int(windows)})"
        )
        ax.grid(True, linestyle="--", alpha=0.5)
        self._annotate_no_signal_channel_runs(ax)
        dedup_legend(ax, title="Pulser rate")
        ax.set_xticks(sorted(subset["num_pulses"].unique()))
        plt.show()

    def plot_min_double_vs_pulses_by_windows(self, pulse_rate_hz: float, channel_count: int) -> None:
        """Y: min separation (µs). X: number of pulses. Curves: capture windows. Hold pulser & channels fixed."""
        table = self.min_double_table()
        subset = table[
            (table["pulse_rate_hz"] == pulse_rate_hz) & (table["channel_count"] == channel_count)
        ]
        if subset.empty:
            raise ValueError("No data for requested pulse_rate/channel_count")
        fig, ax = plt.subplots(figsize=(10, 6))
        window_counts = sorted(subset["windows"].unique())
        palette = {w: f"C{idx}" for idx, w in enumerate(window_counts)}
        for windows, window_df in subset.groupby("windows"):
            sorted_df = window_df.sort_values("num_pulses")
            ax.plot(
                sorted_df["num_pulses"],
                sorted_df["min_double_deadtime_ns"] / 1000.0,
                marker="o",
                linewidth=2,
                color=palette.get(windows, None),
                label=f"{int(windows)} windows",
            )
        ax.set_xlabel("Number of pulses")
        ax.set_ylabel("Minimum separation with multi-pulse response (µs)")
        set_log2_with_decade_ticks(ax, "y", unit="µs")
        ax.set_title(
            f"Min multi-pulse separation vs. pulse count ({self._channel_label(channel_count)}, pulser={pulse_rate_hz:.0f} Hz)"
        )
        ax.grid(True, linestyle="--", alpha=0.5)
        self._annotate_no_signal_channel_runs(ax)
        dedup_legend(ax, title="Capture windows")
        ax.set_xticks(sorted(subset["num_pulses"].unique()))
        plt.show()

    # ---- Tertiary outcome plots ----------------------------------------- #
    def plot_tertiary_vs_separation_by_channels(
        self,
        pulse_rate_hz: float,
        num_pulses: Optional[int] = None,
    ) -> None:
        data = self.subset(pulse_rate_hz, num_pulses=num_pulses)
        if data.empty:
            raise ValueError(f"No data for pulse_rate_hz={pulse_rate_hz}")
        if num_pulses is not None:
            ordering = ["not all pulses seen", "sometimes all pulses seen", "all pulses seen"]
        else:
            ordering = ["not all pulses seen", "sometimes all pulses seen", "all pulses seen"]
        positions = {state: idx for idx, state in enumerate(ordering)}
        for channel_count, channel_df in data.groupby("channel_count"):
            fig, ax = plt.subplots(figsize=(10, 6))
            for (windows, pulse_count), window_df in channel_df.groupby(["windows", "num_pulses"]):
                sorted_df = window_df.sort_values("separation_ns")
                y_vals = sorted_df.apply(
                    lambda row: positions.get(self._tertiary_plot_value(row, num_pulses)),
                    axis=1,
                )
                ax.plot(
                    sorted_df["separation_ns"],
                    y_vals,
                    marker="o",
                    label=f"{int(windows)} windows ({self._pulse_label(pulse_count)})",
                )
            ax.set_xscale("log", base=2)
            if num_pulses is not None:
                pulse_note = f", {self._pulse_label(num_pulses)}"
            else:
                uniq = channel_df["num_pulses"].dropna().unique()
                pulse_note = (
                    f", {self._pulse_label(uniq[0])}" if len(uniq) == 1 else ", mixed pulse counts"
                )
            ax.set_title(
                f"Tertiary outcome vs. separation ({self._channel_label(channel_count)}, pulser={pulse_rate_hz:.0f} Hz{pulse_note})"
            )
            ax.set_xlabel("Pulse separation (ns)")
            ax.set_ylabel("Tertiary outcome")
            ax.set_yticks(range(len(ordering)))
            ax.set_yticklabels(ordering)
            ax.set_ylim(-0.2, len(ordering) - 0.8)
            ax.grid(True, linestyle="--", alpha=0.5)
            dedup_legend(ax, title="Capture windows")
            self._annotate_no_signal_channel_runs(ax)
            plt.show()

    def plot_tertiary_vs_separation_by_windows(
        self,
        pulse_rate_hz: float,
        num_pulses: Optional[int] = None,
    ) -> None:
        data = self.subset(pulse_rate_hz, num_pulses=num_pulses)
        if data.empty:
            raise ValueError(f"No data for pulse_rate_hz={pulse_rate_hz}")
        if num_pulses is not None:
            ordering = ["not all pulses seen", "sometimes all pulses seen", "all pulses seen"]
        else:
            ordering = ["not all pulses seen", "sometimes all pulses seen", "all pulses seen"]
        positions = {state: idx for idx, state in enumerate(ordering)}
        for windows, window_df in data.groupby("windows"):
            fig, ax = plt.subplots(figsize=(10, 6))
            for (channel_count, pulse_count), channel_df in window_df.groupby(
                ["channel_count", "num_pulses"]
            ):
                sorted_df = channel_df.sort_values("separation_ns")
                y_vals = sorted_df.apply(
                    lambda row: positions.get(self._tertiary_plot_value(row, num_pulses)),
                    axis=1,
                )
                ax.plot(
                    sorted_df["separation_ns"],
                    y_vals,
                    marker="o",
                    label=f"{self._channel_label(channel_count)} ({self._pulse_label(pulse_count)})",
                )
            ax.set_xscale("log", base=2)
            if num_pulses is not None:
                pulse_note = f", {self._pulse_label(num_pulses)}"
            else:
                uniq = window_df["num_pulses"].dropna().unique()
                pulse_note = (
                    f", {self._pulse_label(uniq[0])}" if len(uniq) == 1 else ", mixed pulse counts"
                )
            ax.set_title(
                f"Tertiary outcome vs. separation (windows={int(windows)}, pulser={pulse_rate_hz:.0f} Hz{pulse_note})"
            )
            ax.set_xlabel("Pulse separation (ns)")
            ax.set_ylabel("Tertiary outcome")
            ax.set_yticks(range(len(ordering)))
            ax.set_yticklabels(ordering)
            ax.set_ylim(-0.2, len(ordering) - 0.8)
            ax.grid(True, linestyle="--", alpha=0.5)
            dedup_legend(ax, title="Active channels")
            self._annotate_no_signal_channel_runs(ax)
            plt.show()
