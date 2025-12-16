from dataclasses import dataclass
from typing import Optional


@dataclass
class ClassificationThresholds:
    pulse_rate_hz: float
    single_factor: float
    double_factor: float
    num_pulses: Optional[int] = None
    target_ratio: Optional[float] = None

    @property
    def single_threshold(self) -> float:
        return self.single_factor * self.pulse_rate_hz

    @property
    def double_threshold(self) -> float:
        if self.num_pulses and self.num_pulses > 2:
            return 0.9 * self.num_pulses * self.pulse_rate_hz
        return self.double_factor * self.pulse_rate_hz

    @property
    def target_line(self) -> Optional[float]:
        if self.target_ratio is None:
            return None
        if self.num_pulses and self.num_pulses > 2:
            return self.target_ratio * self.num_pulses * self.pulse_rate_hz
        return self.target_ratio * self.pulse_rate_hz
