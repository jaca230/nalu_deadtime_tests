from __future__ import annotations

from pathlib import Path

from sampic_deadtime import SampicDeadtimeAnalysis


def _default_data_files() -> list[str]:
    data_dir = Path(__file__).resolve().parents[1] / "data"
    return [str(path) for path in sorted(data_dir.glob("*.jsonl"))]


def main() -> None:
    files = _default_data_files()
    if not files:
        raise SystemExit("No JSONL files found under data/")

    analysis = SampicDeadtimeAnalysis.from_jsonl(files)

    for pulse_rate in (10, 100):
        analysis.plot_rate_vs_separation_by_digitizer_rate(pulse_rate_hz=pulse_rate)
        analysis.plot_tertiary_vs_separation(pulse_rate_hz=pulse_rate)

    analysis.plot_min_double_vs_digitizer_rate(pulse_rates=[10, 100])
    analysis.plot_converged_vs_digitizer_rate(pulse_rates=[10, 100])


if __name__ == "__main__":
    main()
