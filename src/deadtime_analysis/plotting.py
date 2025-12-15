from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors

from .models import ClassificationThresholds

Region = Tuple[str, float, float, str, Optional[str]]  # label, lower, upper, color, hatch


def apply_rate_guides(
    ax: plt.Axes, thresholds: ClassificationThresholds, *, high_label: Optional[str] = None
) -> None:
    """Overlay shaded single/mixed/upper bands and the binary-search target line."""
    single_thr = thresholds.single_threshold
    double_thr = thresholds.double_threshold
    if single_thr and double_thr:
        label_high = high_label
        if label_high is None:
            if thresholds.num_pulses and thresholds.num_pulses > 2:
                label_high = f"{int(thresholds.num_pulses)}-pulse region"
            else:
                label_high = "N-pulse region"
        regions: List[Region] = [
            ("Single region", 0, single_thr, "#d73027", None),
            ("Mixed region", single_thr, double_thr, "#fdae61", "//"),
        ]
        top = ax.get_ylim()[1]
        if top <= double_thr:
            top = double_thr * 1.1
        regions.append((label_high, double_thr, top, "#1a9850", None))
        for label, lower, upper, color, hatch in regions:
            ax.axhspan(lower, upper, color=color, alpha=0.08, label=label, hatch=hatch)

        if thresholds.target_line is not None:
            ax.axhline(
                thresholds.target_line,
                color="k",
                linestyle="--",
                linewidth=1.0,
                label=f"Binary-search target ({thresholds.target_line:.1f} Hz)",
            )


def pulse_region_spans(pulse_rate_hz: float, num_pulses: int) -> List[Region]:
    """
    Build pulse/mixed regions with 25% bands around each integer pulse multiple.

    Pure k-pulse region: [(k-0.25)f, (k+0.25)f]
    Mixed k-(k+1): [(k+0.25)f, (k+0.75)f]
    """
    regions: List[Region] = []
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    def _shade(base: str, factor: float) -> str:
        rgb = np.array(mcolors.to_rgb(base))
        rgb = rgb * factor + (1 - factor)
        return mcolors.to_hex(rgb)

    def _color(idx: int) -> str:
        if colors:
            return colors[idx % len(colors)]
        return f"C{idx}"

    # Pure pulse regions
    for k in range(1, num_pulses + 1):
        lower = max(0.0, (k - 0.25) * pulse_rate_hz)
        upper = (k + 0.25) * pulse_rate_hz
        base = _color(k - 1)
        regions.append((f"{k}-pulse region", lower, upper, base, None))
        if k < num_pulses:
            mix_lower = (k + 0.25) * pulse_rate_hz
            mix_upper = (k + 0.75) * pulse_rate_hz
            regions.append((f"Mixed {k}-{k+1}", mix_lower, mix_upper, _shade(base, 0.6), "//"))
    # Top open region to cover values beyond the last band
    top_lower = (num_pulses + 0.25) * pulse_rate_hz
    last_base = _color(num_pulses - 1)
    regions.append(
        (
            f"{num_pulses}-pulse+ region",
            top_lower,
            top_lower * 1.2,
            _shade(last_base, 0.85),
            "..",
        )
    )
    return regions


def apply_pulse_regions(
    ax: plt.Axes,
    pulse_rate_hz: float,
    num_pulses: int,
    target_line: Optional[float] = None,
    *,
    alpha: float = 0.08,
) -> None:
    """Overlay pulse/mixed regions for n-pulse sequences using 25% tolerance bands."""
    spans = pulse_region_spans(pulse_rate_hz, num_pulses)
    ylim_top = ax.get_ylim()[1]
    for label, lower, upper, color, hatch in spans:
        if upper > ylim_top:
            ylim_top = upper * 1.05
        ax.axhspan(lower, upper, color=color, alpha=alpha, label=label, hatch=hatch)
    if target_line is not None:
        ax.axhline(
            target_line,
            color="k",
            linestyle="--",
            linewidth=1.0,
            label=f"Binary-search target ({target_line:.1f} Hz)",
        )
    ax.set_ylim(ax.get_ylim()[0], ylim_top)


def dedup_legend(
    ax: plt.Axes,
    title: Optional[str] = None,
    *,
    outside: bool = True,
    loc: str = "center left",
    bbox_to_anchor: Tuple[float, float] = (1.02, 0.5),
) -> None:
    """Remove duplicate legend labels while preserving order and optionally place outside plot."""
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq_handles = []
    uniq_labels = []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        uniq_handles.append(h)
        uniq_labels.append(l)
    if uniq_handles:
        legend_title = title or _existing_title(ax)
        if outside:
            ax.legend(
                uniq_handles,
                uniq_labels,
                title=legend_title,
                loc=loc,
                bbox_to_anchor=bbox_to_anchor,
            )
        else:
            ax.legend(uniq_handles, uniq_labels, title=legend_title)


def set_log2_with_decade_ticks(ax: plt.Axes, axis: str, unit: Optional[str] = None) -> None:
    """
    Set log base-2 scaling while labeling ticks at decade values (base-10) for readability.
    Useful when we want linear-looking trends on log2 axes but human-friendly tick labels.
    """
    if axis not in {"x", "y"}:
        raise ValueError("axis must be 'x' or 'y'")
    getter = ax.get_xlim if axis == "x" else ax.get_ylim
    setter = ax.set_xlim if axis == "x" else ax.set_ylim
    scaler = ax.set_xscale if axis == "x" else ax.set_yscale
    scaler("log", base=2)
    lo, hi = getter()
    if lo <= 0:
        lo = max(lo, 1e-9)
        setter(lo, hi)
    exp_min = math.floor(math.log10(lo))
    exp_max = math.ceil(math.log10(hi))
    ticks = [10 ** k for k in range(exp_min, exp_max + 1)]
    if axis == "x":
        ax.set_xticks(ticks)
    else:
        ax.set_yticks(ticks)

    def _fmt(val, _pos):
        if val == 0:
            return "0"
        if unit:
            return f"{val:g} {unit}"
        return f"{val:g}"

    from matplotlib.ticker import FuncFormatter

    if axis == "x":
        ax.xaxis.set_major_formatter(FuncFormatter(_fmt))
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(_fmt))


def grouped_shades_for_rate(group_idx: int, series_labels: Sequence[object]) -> Dict[object, str]:
    """Return a light-to-dark palette for a pulser rate based on the default color cycle.

    Each pulser rate keeps its base hue (from the matplotlib color cycle), and the individual
    series under that rate get progressively darker shades of that hue. This makes curves
    distinguishable while still hinting that they belong to the same pulser-rate family.
    """

    labels = list(series_labels)
    if not labels:
        return {}

    base_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    base_color = base_cycle[group_idx % len(base_cycle)] if base_cycle else f"C{group_idx}"
    base_rgb = np.array(mcolors.to_rgb(base_color))

    # Lighten the base color toward white for early entries; keep last shade at the base hue.
    if len(labels) == 1:
        factors = [1.0]
    else:
        factors = np.linspace(0.55, 1.0, num=len(labels))

    def _shade(factor: float) -> str:
        rgb = base_rgb * factor + (1 - factor)
        return mcolors.to_hex(rgb)

    shades = [_shade(f) for f in factors]
    return {label: color for label, color in zip(sorted(labels), shades)}


def _existing_title(ax: plt.Axes) -> Optional[str]:
    existing = ax.get_legend()
    if existing and existing.get_title():
        return existing.get_title().get_text()
    return None
