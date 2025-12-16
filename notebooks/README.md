You can mirror the existing NALU notebooks with the Sampic data using:

```python
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root / "src"))

from sampic_deadtime import SampicDeadtimeAnalysis

analysis = SampicDeadtimeAnalysis.from_jsonl([
    repo_root / "data" / "double_pulse_deadtime_scan.jsonl",
    repo_root / "data" / "double_pulse_deadtime_scan_12-2-2025.jsonl",
])

analysis.plot_rate_vs_separation_by_digitizer_rate(pulse_rate_hz=10)
analysis.plot_min_double_vs_digitizer_rate(pulse_rates=[10, 100])
```

Feel free to copy any of the plotting calls into a notebook cell to iterate interactively.
