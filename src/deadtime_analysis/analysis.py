from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .constants import DEFAULT_DOUBLE_FACTOR, DEFAULT_SINGLE_FACTOR
from .loader import load_records
from .models import ClassificationThresholds
from .plotting import (
    apply_rate_guides,
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
    def subset(self, pulse_rate_hz: float) -> pd.DataFrame:
        return self.df[self.df["pulse_rate_hz"] == pulse_rate_hz].copy()

    def _thresholds_for_rate(self, pulse_rate_hz: float) -> ClassificationThresholds:
        target_vals = (
            self.df.loc[self.df["pulse_rate_hz"] == pulse_rate_hz, "target_ratio"]
            .dropna()
            .unique()
        )
        target_ratio = target_vals[0] if len(target_vals) else None
        return ClassificationThresholds(
            pulse_rate_hz=pulse_rate_hz,
            single_factor=self.single_factor,
            double_factor=self.double_factor,
            target_ratio=target_ratio,
        )

    def converged_table(self) -> pd.DataFrame:
        return (
            self.df.sort_values("search_iteration")
            .groupby(["pulse_rate_hz", "channel_count", "windows"], as_index=False)
            .tail(1)
            .rename(columns={"separation_ns": "converged_deadtime_ns"})
        )

    def min_double_table(self) -> pd.DataFrame:
        double_df = self.df[self.df["tertiary_mode"] == "double"]
        return (
            double_df.sort_values("separation_ns")
            .groupby(["pulse_rate_hz", "channel_count", "windows"], as_index=False)
            .first()
            .rename(columns={"separation_ns": "min_double_deadtime_ns"})
        )

    # ---- Rate plots ------------------------------------------------------ #
    def plot_rate_vs_separation_by_channels(
        self,
        pulse_rate_hz: float,
    ) -> None:
        data = self.subset(pulse_rate_hz)
        if data.empty:
            raise ValueError(f"No data for pulse_rate_hz={pulse_rate_hz}")
        thresholds = self._thresholds_for_rate(pulse_rate_hz)
        for channel_count, channel_df in data.groupby("channel_count"):
            fig, ax = plt.subplots(figsize=(10, 6))
            x_min = channel_df["separation_ns"].min() * 0.95
            x_max = channel_df["separation_ns"].max() * 1.05
            y_min_obs = channel_df["observed_rate_hz"].min() * 0.95
            y_max_obs = channel_df["observed_rate_hz"].max() * 1.05
            ax.set_xlim(x_min, x_max)
            y_top = y_max_obs
            if thresholds.double_threshold:
                y_top = max(y_top, thresholds.double_threshold * 1.1)
            if thresholds.target_line:
                y_top = max(y_top, thresholds.target_line * 1.1)
            ax.set_ylim(y_min_obs, y_top)
            for windows, window_df in channel_df.groupby("windows"):
                sorted_df = window_df.sort_values("separation_ns")
                ax.plot(
                    sorted_df["separation_ns"],
                    sorted_df["observed_rate_hz"],
                    marker="o",
                    label=f"{int(windows)} windows",
                )
            apply_rate_guides(ax, thresholds)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log", base=2)
            ax.set_title(
                f"Observed rate vs. separation (channels={int(channel_count)}, pulser={pulse_rate_hz:.0f} Hz)"
            )
            ax.set_xlabel("Pulse separation (ns)")
            ax.set_ylabel("Observed rate (events/s)")
            ax.grid(True, linestyle="--", alpha=0.5)
            dedup_legend(ax, title="Capture windows")
            plt.show()

    def plot_rate_vs_separation_by_windows(
        self,
        pulse_rate_hz: float,
    ) -> None:
        data = self.subset(pulse_rate_hz)
        if data.empty:
            raise ValueError(f"No data for pulse_rate_hz={pulse_rate_hz}")
        thresholds = self._thresholds_for_rate(pulse_rate_hz)
        for windows, window_df in data.groupby("windows"):
            fig, ax = plt.subplots(figsize=(10, 6))
            x_min = window_df["separation_ns"].min() * 0.95
            x_max = window_df["separation_ns"].max() * 1.05
            y_min_obs = window_df["observed_rate_hz"].min() * 0.95
            y_max_obs = window_df["observed_rate_hz"].max() * 1.05
            ax.set_xlim(x_min, x_max)
            y_top = y_max_obs
            if thresholds.double_threshold:
                y_top = max(y_top, thresholds.double_threshold * 1.1)
            if thresholds.target_line:
                y_top = max(y_top, thresholds.target_line * 1.1)
            ax.set_ylim(y_min_obs, y_top)
            for channel_count, channel_df in window_df.groupby("channel_count"):
                sorted_df = channel_df.sort_values("separation_ns")
                ax.plot(
                    sorted_df["separation_ns"],
                    sorted_df["observed_rate_hz"],
                    marker="o",
                    label=f"{int(channel_count)} channels",
                )
            apply_rate_guides(ax, thresholds)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log", base=2)
            ax.set_title(
                f"Observed rate vs. separation (windows={int(windows)}, pulser={pulse_rate_hz:.0f} Hz)"
            )
            ax.set_xlabel("Pulse separation (ns)")
            ax.set_ylabel("Observed rate (events/s)")
            ax.grid(True, linestyle="--", alpha=0.5)
            dedup_legend(ax, title="Active channels")
            plt.show()

    # ---- Derived separation plots --------------------------------------- #
    def plot_converged_vs_windows(
        self,
        pulse_rates: Optional[Iterable[float]] = None,
    ) -> None:
        pulse_rates = list(pulse_rates) if pulse_rates else sorted(self.df["pulse_rate_hz"].unique())
        conv = self.converged_table()
        fig, ax = plt.subplots(figsize=(12, 6))
        for idx, pulse_rate in enumerate(pulse_rates):
            subset = conv[conv["pulse_rate_hz"] == pulse_rate]
            if subset.empty:
                continue
            channel_counts = sorted(subset["channel_count"].unique())
            palette = grouped_shades_for_rate(idx, channel_counts)
            for channel_count, channel_df in subset.groupby("channel_count"):
                sorted_df = channel_df.sort_values("windows")
                color = palette.get(channel_count, f"C{idx}")
                ax.plot(
                    sorted_df["windows"],
                    sorted_df["converged_deadtime_ns"] / 1000.0,
                    marker="o",
                    linewidth=1.8,
                    color=color,
                    label=f"{int(channel_count)} ch @ {pulse_rate:.0f}Hz",
                )
        ax.set_xlabel("Capture windows")
        ax.set_ylabel("Converged deadtime (µs)")
        ax.set_xscale("log", base=2)
        set_log2_with_decade_ticks(ax, "y", unit="µs")
        ax.set_title("Converged deadtime vs. capture windows (axes log2; y ticks in µs, log10 decades)")
        ax.grid(True, linestyle="--", alpha=0.5)
        dedup_legend(ax, title="Channel / pulser")
        plt.show()

    def plot_converged_vs_channels(
        self,
        pulse_rates: Optional[Iterable[float]] = None,
    ) -> None:
        pulse_rates = list(pulse_rates) if pulse_rates else sorted(self.df["pulse_rate_hz"].unique())
        conv = self.converged_table()
        fig, ax = plt.subplots(figsize=(12, 6))
        for idx, pulse_rate in enumerate(pulse_rates):
            subset = conv[conv["pulse_rate_hz"] == pulse_rate]
            if subset.empty:
                continue
            window_counts = sorted(subset["windows"].unique())
            palette = grouped_shades_for_rate(idx, window_counts)
            for windows, window_df in subset.groupby("windows"):
                sorted_df = window_df.sort_values("channel_count")
                color = palette.get(windows, f"C{idx}")
                ax.plot(
                    sorted_df["channel_count"],
                    sorted_df["converged_deadtime_ns"] / 1000.0,
                    marker="o",
                    linewidth=1.8,
                    color=color,
                    label=f"{int(windows)} win @ {pulse_rate:.0f}Hz",
                )
        ax.set_xlabel("Active channels")
        ax.set_ylabel("Converged deadtime (µs)")
        ax.set_xscale("log", base=2)
        set_log2_with_decade_ticks(ax, "y", unit="µs")
        ax.set_title("Converged deadtime vs. active channels (axes log2; y ticks in µs, log10 decades)")
        ax.grid(True, linestyle="--", alpha=0.5)
        dedup_legend(ax, title="Windows / pulser")
        plt.show()

    def plot_min_double_vs_windows(
        self,
        pulse_rates: Optional[Iterable[float]] = None,
        print_fits: bool = True,
    ) -> None:
        pulse_rates = list(pulse_rates) if pulse_rates else sorted(self.df["pulse_rate_hz"].unique())
        min_double = self.min_double_table()
        fig, ax = plt.subplots(figsize=(12, 6))
        fit_lines: List[Tuple[str, float, float]] = []
        for idx, pulse_rate in enumerate(pulse_rates):
            subset = min_double[min_double["pulse_rate_hz"] == pulse_rate]
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
                ax.plot(
                    x_vals,
                    y_vals,
                    color=color,
                    linewidth=2,
                    marker="s",
                    label=label,
                    alpha=0.8,
                )
                if sorted_df["windows"].nunique() > 1:
                    slope, intercept = np.polyfit(x_vals, y_vals, 1)
                    x_fit = np.array([x_vals.min(), x_vals.max()])
                    y_fit = slope * x_fit + intercept
                    ax.plot(
                        x_fit,
                        y_fit,
                        color=color,
                        linestyle="--",
                        linewidth=1.2,
                        alpha=0.8,
                    )
                    fit_lines.append((label, slope, intercept))
        ax.set_xlabel("Capture samples (windows × 32)")
        ax.set_ylabel("Minimum separation with double response (µs)")
        ax.set_xscale("log", base=2)
        set_log2_with_decade_ticks(ax, "y", unit="µs")
        ax.set_title("Minimum double-response separation vs. capture samples (axes log2; y ticks µs decades)")
        ax.grid(True, linestyle="--", alpha=0.5)
        dedup_legend(ax, title="Channel / pulser")
        plt.show()
        if print_fits:
            if fit_lines:
                print("Linear fits for minimum double-response separation (windows on x-axis):")
                for label, slope, intercept in fit_lines:
                    print(f"  {label}: separation = {slope:.3f} * samples + {intercept:.2f}")
            else:
                print("No linear fits computed (insufficient data).")

    def plot_min_double_vs_channels(
        self,
        pulse_rates: Optional[Iterable[float]] = None,
        print_fits: bool = True,
    ) -> None:
        pulse_rates = list(pulse_rates) if pulse_rates else sorted(self.df["pulse_rate_hz"].unique())
        min_double = self.min_double_table()
        fig, ax = plt.subplots(figsize=(12, 6))
        fit_lines: List[Tuple[str, float, float]] = []
        for idx, pulse_rate in enumerate(pulse_rates):
            subset = min_double[min_double["pulse_rate_hz"] == pulse_rate]
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
                ax.plot(
                    x_vals,
                    y_vals,
                    color=color,
                    linewidth=2,
                    marker="s",
                    label=label,
                    alpha=0.8,
                )
                if sorted_df["channel_count"].nunique() > 1:
                    slope, intercept = np.polyfit(x_vals, y_vals, 1)
                    x_fit = np.array([x_vals.min(), x_vals.max()])
                    y_fit = slope * x_fit + intercept
                    ax.plot(
                        x_fit,
                        y_fit,
                        color=color,
                        linestyle="--",
                        linewidth=1.2,
                        alpha=0.8,
                    )
                    fit_lines.append((label, slope, intercept))
        ax.set_xlabel("Active channels")
        ax.set_ylabel("Minimum separation with double response (µs)")
        ax.set_xscale("log", base=2)
        set_log2_with_decade_ticks(ax, "y", unit="µs")
        ax.set_title("Minimum double-response separation vs. active channels (axes log2; y ticks µs decades)")
        ax.grid(True, linestyle="--", alpha=0.5)
        dedup_legend(ax, title="Windows / pulser")
        plt.show()
        if print_fits:
            if fit_lines:
                print("Linear fits for minimum double-response separation (channels on x-axis):")
                for label, slope, intercept in fit_lines:
                    print(f"  {label}: separation = {slope:.3f} * channels + {intercept:.2f}")
            else:
                print("No linear fits computed (insufficient data).")

    # ---- Tertiary outcome plots ----------------------------------------- #
    def plot_tertiary_vs_separation_by_channels(
        self,
        pulse_rate_hz: float,
    ) -> None:
        data = self.subset(pulse_rate_hz)
        if data.empty:
            raise ValueError(f"No data for pulse_rate_hz={pulse_rate_hz}")
        ordering = ["single", "mixed", "double"]
        positions = {state: idx for idx, state in enumerate(ordering)}
        for channel_count, channel_df in data.groupby("channel_count"):
            fig, ax = plt.subplots(figsize=(10, 6))
            for windows, window_df in channel_df.groupby("windows"):
                sorted_df = window_df.sort_values("separation_ns")
                y_vals = sorted_df["tertiary_mode"].map(positions)
                ax.plot(
                    sorted_df["separation_ns"],
                    y_vals,
                    marker="o",
                    label=f"{int(windows)} windows",
                )
            ax.set_xscale("log", base=2)
            ax.set_title(
                f"Tertiary outcome vs. separation (channels={int(channel_count)}, pulser={pulse_rate_hz:.0f} Hz)"
            )
            ax.set_xlabel("Pulse separation (ns)")
            ax.set_ylabel("Tertiary outcome")
            ax.set_yticks(range(len(ordering)))
            ax.set_yticklabels(ordering)
            ax.set_ylim(-0.2, len(ordering) - 0.8)
            ax.grid(True, linestyle="--", alpha=0.5)
            dedup_legend(ax, title="Capture windows")
            plt.show()

    def plot_tertiary_vs_separation_by_windows(
        self,
        pulse_rate_hz: float,
    ) -> None:
        data = self.subset(pulse_rate_hz)
        if data.empty:
            raise ValueError(f"No data for pulse_rate_hz={pulse_rate_hz}")
        ordering = ["single", "mixed", "double"]
        positions = {state: idx for idx, state in enumerate(ordering)}
        for windows, window_df in data.groupby("windows"):
            fig, ax = plt.subplots(figsize=(10, 6))
            for channel_count, channel_df in window_df.groupby("channel_count"):
                sorted_df = channel_df.sort_values("separation_ns")
                y_vals = sorted_df["tertiary_mode"].map(positions)
                ax.plot(
                    sorted_df["separation_ns"],
                    y_vals,
                    marker="o",
                    label=f"{int(channel_count)} channels",
                )
            ax.set_xscale("log", base=2)
            ax.set_xscale("log", base=2)
            ax.set_title(
                f"Tertiary outcome vs. separation (windows={int(windows)}, pulser={pulse_rate_hz:.0f} Hz)"
            )
            ax.set_xlabel("Pulse separation (ns)")
            ax.set_ylabel("Tertiary outcome")
            ax.set_yticks(range(len(ordering)))
            ax.set_yticklabels(ordering)
            ax.set_ylim(-0.2, len(ordering) - 0.8)
            ax.grid(True, linestyle="--", alpha=0.5)
            dedup_legend(ax, title="Active channels")
            plt.show()
