from dataclasses import dataclass
from typing import Optional


@dataclass
class ClassificationThresholds:
    pulse_rate_hz: float
    single_factor: float
    double_factor: float
    target_ratio: Optional[float] = None

    @property
    def single_threshold(self) -> float:
        return self.single_factor * self.pulse_rate_hz

    @property
    def double_threshold(self) -> float:
        return self.double_factor * self.pulse_rate_hz

    @property
    def target_line(self) -> Optional[float]:
        if self.target_ratio is None:
            return None
        return self.target_ratio * self.pulse_rate_hz
