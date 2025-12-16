from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .constants import DEFAULT_DOUBLE_FACTOR, DEFAULT_PULSE_COUNT, DEFAULT_SINGLE_FACTOR
from .loader import load_records
from .models import ClassificationThresholds
from .plotting import (
    apply_pulse_regions,
    dedup_legend,
    grouped_shades_for_rate,
    set_log2_with_decade_ticks,
)
from .prep import build_dataframe


class SampicDeadtimeAnalysis:
    """Load, classify, and plot Sampic deadtime sweeps."""

    def __init__(
        self,
        df: pd.DataFrame,
        single_factor: float = DEFAULT_SINGLE_FACTOR,
        double_factor: float = DEFAULT_DOUBLE_FACTOR,
        default_num_pulses: int = DEFAULT_PULSE_COUNT,
    ):
        self.single_factor = single_factor
        self.double_factor = double_factor
        self.default_num_pulses = default_num_pulses
        self.df = df

    def _digitizer_label(self, digitizer_rate_mhz: float) -> str:
        return f"{digitizer_rate_mhz:g} MHz"

    def _pulse_label(self, num_pulses: Optional[float]) -> str:
        if pd.isna(num_pulses):
            return "unknown pulses"
        count = int(num_pulses)
        return f"{count} pulse{'s' if count != 1 else ''}"

    def _target_line(self, pulse_rate_hz: float, num_pulses: Optional[int]) -> Optional[float]:
        df = self.subset(pulse_rate_hz, num_pulses=num_pulses)
        target_vals = df["target_line"].dropna().unique()
        return target_vals[0] if len(target_vals) else None

    @classmethod
    def from_jsonl(
        cls,
        paths: Sequence[str],
        *,
        single_factor: float = DEFAULT_SINGLE_FACTOR,
        double_factor: float = DEFAULT_DOUBLE_FACTOR,
        default_num_pulses: int = DEFAULT_PULSE_COUNT,
    ) -> "SampicDeadtimeAnalysis":
        records = load_records(paths)
        df = build_dataframe(records, default_num_pulses=default_num_pulses)
        return cls(
            df,
            single_factor=single_factor,
            double_factor=double_factor,
            default_num_pulses=default_num_pulses,
        )

    # ---- Data slices ----------------------------------------------------- #
    def subset(
        self,
        pulse_rate_hz: float,
        *,
        digitizer_rate_mhz: Optional[float] = None,
        num_pulses: Optional[int] = None,
    ) -> pd.DataFrame:
        df = self.df[np.isclose(self.df["pulse_rate_hz"], pulse_rate_hz)]
        if digitizer_rate_mhz is not None:
            df = df[np.isclose(df["digitizer_rate_mhz"], digitizer_rate_mhz)]
        if num_pulses is not None:
            df = df[df["num_pulses"] == num_pulses]
        return df.copy()

    def converged_table(self) -> pd.DataFrame:
        conv = (
            self.df.sort_values("search_iteration")
            .groupby(["pulse_rate_hz", "digitizer_rate_mhz", "board_index", "num_pulses"], as_index=False)
            .tail(1)
        )
        conv = conv.rename(
            columns={
                "separation_ns": "converged_deadtime_ns",
                "search_min_ns": "converged_lower_bound_ns",
                "search_max_ns": "converged_upper_bound_ns",
            }
        )
        conv["converged_deadtime_ns"] = conv["best_delay_ns"].fillna(conv["converged_deadtime_ns"])
        return conv

    def min_double_table(self) -> pd.DataFrame:
        rows = []
        for (pulse_rate, digitizer_rate, board_index, num_pulses), group_df in self.df.groupby(
            ["pulse_rate_hz", "digitizer_rate_mhz", "board_index", "num_pulses"]
        ):
            if pd.isna(num_pulses):
                continue
            target_region = group_df[group_df["pulse_region"] == num_pulses].sort_values(
                "separation_ns"
            )
            if target_region.empty:
                continue
            min_row = target_region.iloc[0]
            prev_region = group_df[group_df["pulse_region"] == (num_pulses - 1)]
            lower_prev = prev_region["separation_ns"].max() if not prev_region.empty else None
            search_lower = min_row.get("search_min_ns")
            lower_bound = lower_prev if lower_prev is not None else search_lower
            rows.append(
                {
                    "pulse_rate_hz": pulse_rate,
                    "digitizer_rate_mhz": digitizer_rate,
                    "board_index": board_index,
                    "num_pulses": num_pulses,
                    "min_double_deadtime_ns": min_row["separation_ns"],
                    "min_double_lower_bound_ns": lower_bound,
                }
            )
        return pd.DataFrame(rows)

    # ---- Rate plots ------------------------------------------------------ #
    def plot_rate_vs_separation_by_digitizer_rate(
        self,
        pulse_rate_hz: float,
        *,
        num_pulses: Optional[int] = None,
        highlight_separations_ns: Optional[List[float]] = None,
        highlight_text: Optional[str] = None,
    ) -> None:
        data = self.subset(pulse_rate_hz, num_pulses=num_pulses)
        if data.empty:
            raise ValueError(f"No data for pulse_rate_hz={pulse_rate_hz}")
        target_line = self._target_line(pulse_rate_hz, num_pulses)
        fig, ax = plt.subplots(figsize=(10, 6))
        palette = {rate: f"C{idx}" for idx, rate in enumerate(sorted(data["digitizer_rate_mhz"].unique()))}
        for digitizer_rate, rate_df in data.groupby("digitizer_rate_mhz"):
            sorted_df = rate_df.sort_values("separation_ns")
            ax.plot(
                sorted_df["separation_ns"],
                sorted_df["observed_rate_hz"],
                marker="o",
                label=self._digitizer_label(digitizer_rate),
                color=palette.get(digitizer_rate),
            )
        if highlight_separations_ns:
            for sep in highlight_separations_ns:
                y_vals = data[np.isclose(data["separation_ns"], sep)]["observed_rate_hz"]
                if y_vals.empty:
                    continue
                ax.scatter([sep], [y_vals.mean()], marker="*", s=150, color="black", zorder=5, label=None)
        if highlight_text:
            ax.text(
                0.02,
                0.95,
                highlight_text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.7},
            )
        if num_pulses:
            apply_pulse_regions(ax, pulse_rate_hz, int(num_pulses), target_line=target_line)
        set_log2_with_decade_ticks(ax, "x", unit="ns")
        if num_pulses is not None:
            pulse_note = f", {self._pulse_label(num_pulses)}"
        else:
            uniq = data["num_pulses"].dropna().unique()
            pulse_note = f", {self._pulse_label(uniq[0])}" if len(uniq) == 1 else ", mixed pulse counts"
        ax.set_title(
            f"Observed rate vs. separation (pulser={pulse_rate_hz:.0f} Hz{pulse_note})"
        )
        ax.set_xlabel("Pulse separation (ns)")
        ax.set_ylabel("Observed rate (events/s)")
        ax.grid(True, linestyle="--", alpha=0.5)
        dedup_legend(ax, title="Digitizer rate")
        plt.show()

    def plot_rate_vs_separation_by_pulse_rate(
        self,
        digitizer_rate_mhz: float,
        *,
        num_pulses: Optional[int] = None,
        highlight_separations_ns: Optional[List[float]] = None,
        highlight_text: Optional[str] = None,
    ) -> None:
        data = self.df[np.isclose(self.df["digitizer_rate_mhz"], digitizer_rate_mhz)]
        if num_pulses is not None:
            data = data[data["num_pulses"] == num_pulses]
        if data.empty:
            raise ValueError(f"No data for digitizer_rate_mhz={digitizer_rate_mhz}")
        fig, ax = plt.subplots(figsize=(10, 6))
        pulse_rates = sorted(data["pulse_rate_hz"].unique())
        palette = {rate: f"C{idx}" for idx, rate in enumerate(pulse_rates)}
        for pulse_rate, rate_df in data.groupby("pulse_rate_hz"):
            sorted_df = rate_df.sort_values("separation_ns")
            ax.plot(
                sorted_df["separation_ns"],
                sorted_df["observed_rate_hz"],
                marker="o",
                label=f"{pulse_rate:.0f} Hz",
                color=palette.get(pulse_rate),
            )
        if highlight_separations_ns:
            for sep in highlight_separations_ns:
                y_vals = data[np.isclose(data["separation_ns"], sep)]["observed_rate_hz"]
                if y_vals.empty:
                    continue
                ax.scatter([sep], [y_vals.mean()], marker="*", s=150, color="black", zorder=5, label=None)
        if highlight_text:
            ax.text(
                0.02,
                0.95,
                highlight_text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.7},
            )
        if num_pulses:
            target_line = self._target_line(pulse_rates[0], num_pulses)
            apply_pulse_regions(ax, pulse_rates[0], int(num_pulses), target_line=target_line)
        set_log2_with_decade_ticks(ax, "x", unit="ns")
        if num_pulses is not None:
            pulse_note = f", {self._pulse_label(num_pulses)}"
        else:
            uniq = data["num_pulses"].dropna().unique()
            pulse_note = f", {self._pulse_label(uniq[0])}" if len(uniq) == 1 else ", mixed pulse counts"
        ax.set_title(
            f"Observed rate vs. separation ({self._digitizer_label(digitizer_rate_mhz)}{pulse_note})"
        )
        ax.set_xlabel("Pulse separation (ns)")
        ax.set_ylabel("Observed rate (events/s)")
        ax.grid(True, linestyle="--", alpha=0.5)
        dedup_legend(ax, title="Pulser rate")
        plt.show()

    # ---- Derived separation plots --------------------------------------- #
    def plot_converged_vs_digitizer_rate(
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
                digitizer_rates = sorted(subset["digitizer_rate_mhz"].unique())
                palette = grouped_shades_for_rate(idx, digitizer_rates)
                sorted_df = subset.sort_values("digitizer_rate_mhz")
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
                    sorted_df["digitizer_rate_mhz"],
                    y_vals,
                    yerr=yerr,
                    fmt="o-",
                    linewidth=1.8,
                    color=f"C{idx}",
                    ecolor=[palette.get(r, f"C{idx}") for r in sorted_df["digitizer_rate_mhz"]],
                    label=f"{pulse_rate:.0f} Hz",
                    capsize=4,
                )
            ax.set_xlabel("Digitizer rate (MHz)")
            ax.set_ylabel("Converged deadtime (µs)")
            set_log2_with_decade_ticks(ax, "y", unit="µs")
            ax.set_title(
                f"Converged deadtime vs. digitizer rate (pulse count={int(pulse_count)}; y log2 with decade ticks)"
            )
            ax.grid(True, linestyle="--", alpha=0.5)
            dedup_legend(ax, title="Pulser rate")
            plt.show()

    def plot_min_double_vs_digitizer_rate(
        self,
        pulse_rates: Optional[Iterable[float]] = None,
        pulse_counts: Optional[Iterable[int]] = None,
        print_fits: bool = True,
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
            fit_lines: List[tuple[str, float, float]] = []
            for idx, pulse_rate in enumerate(pulse_rates):
                subset = subset_count[subset_count["pulse_rate_hz"] == pulse_rate]
                if subset.empty:
                    continue
                sorted_df = subset.sort_values("digitizer_rate_mhz")
                x_vals = sorted_df["digitizer_rate_mhz"].to_numpy(dtype=float)
                y_vals = (sorted_df["min_double_deadtime_ns"] / 1000.0).to_numpy(dtype=float)
                lower = sorted_df["min_double_lower_bound_ns"] / 1000.0
                lower_err = y_vals - lower.to_numpy(dtype=float)
                lower_err = np.nan_to_num(lower_err, nan=0.0, posinf=0.0, neginf=0.0)
                lower_err = np.clip(lower_err, a_min=0, a_max=None)
                upper_err = np.zeros_like(lower_err)
                yerr = np.vstack([lower_err, upper_err])
                color = f"C{idx}"
                ax.errorbar(
                    x_vals,
                    y_vals,
                    yerr=yerr,
                    color=color,
                    linewidth=2,
                    marker="s",
                    label=f"{pulse_rate:.0f} Hz",
                    alpha=0.85,
                    capsize=4,
                )
                if sorted_df["digitizer_rate_mhz"].nunique() > 1:
                    slope, intercept = np.polyfit(x_vals, y_vals, 1)
                    fit_lines.append((f"{pulse_rate:.0f} Hz", slope, intercept))
            ax.set_xlabel("Digitizer rate (MHz)")
            ax.set_ylabel("Minimum separation with multi-pulse response (µs)")
            set_log2_with_decade_ticks(ax, "y", unit="µs")
            ax.set_title(
                f"Minimum multi-pulse separation vs. digitizer rate (pulse count={int(pulse_count)}; y log2 with decade ticks)"
            )
            ax.grid(True, linestyle="--", alpha=0.5)
            dedup_legend(ax, title="Pulser rate")
            plt.show()
            if print_fits:
                if fit_lines:
                    print(
                        f"Linear fits for minimum multi-pulse separation (digitizer rate on x-axis, pulse count={int(pulse_count)}):"
                    )
                    for label, slope, intercept in fit_lines:
                        print(f"  {label}: separation = {slope:.3f} * MHz + {intercept:.2f}")
                else:
                    print(f"No linear fits computed (pulse count={int(pulse_count)}).")

    # ---- Tertiary outcome plots ----------------------------------------- #
    def plot_tertiary_vs_separation(
        self,
        pulse_rate_hz: float,
        *,
        num_pulses: Optional[int] = None,
    ) -> None:
        data = self.subset(pulse_rate_hz, num_pulses=num_pulses)
        if data.empty:
            raise ValueError(f"No data for pulse_rate_hz={pulse_rate_hz}")
        ordering = ["not all pulses seen", "sometimes all pulses seen", "all pulses seen"]
        positions = {state: idx for idx, state in enumerate(ordering)}
        fig, ax = plt.subplots(figsize=(10, 6))
        for digitizer_rate, rate_df in data.groupby("digitizer_rate_mhz"):
            sorted_df = rate_df.sort_values("separation_ns")
            y_vals = sorted_df.apply(
                lambda row: positions.get(row.get("tertiary_mode"), 1),
                axis=1,
            )
            ax.plot(
                sorted_df["separation_ns"],
                y_vals,
                marker="o",
                label=self._digitizer_label(digitizer_rate),
            )
        ax.set_xscale("log", base=2)
        if num_pulses is not None:
            pulse_note = f", {self._pulse_label(num_pulses)}"
        else:
            uniq = data["num_pulses"].dropna().unique()
            pulse_note = f", {self._pulse_label(uniq[0])}" if len(uniq) == 1 else ", mixed pulse counts"
        ax.set_title(
            f"Tertiary outcome vs. separation (pulser={pulse_rate_hz:.0f} Hz{pulse_note})"
        )
        ax.set_xlabel("Pulse separation (ns)")
        ax.set_ylabel("Tertiary outcome")
        ax.set_yticks(range(len(ordering)))
        ax.set_yticklabels(ordering)
        ax.set_ylim(-0.2, len(ordering) - 0.8)
        ax.grid(True, linestyle="--", alpha=0.5)
        dedup_legend(ax, title="Digitizer rate")
        plt.show()
