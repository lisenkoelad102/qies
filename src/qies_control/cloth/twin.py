from dataclasses import dataclass, field

import numpy as np

from qies_control.atelier.settings import TwinConfig
from qies_control.atelier.types import FloatArray

STATE_DIM = 5
DOSE = 5.0


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))


@dataclass
class StepOutcome:
    state: FloatArray
    therapeutic_benefit: float
    interaction_penalty: float
    in_range: bool
    adverse_exposure: float
    healing: float


@dataclass
class CrohnsTwin:
    config: TwinConfig
    patient_phi: float
    clearance: float
    volume: float
    trough: float = 0.0
    damage: float = 0.5
    healing: float = 0.5
    last_interval: float = 6.0
    last_nutrition: float = 0.0

    def observation(self) -> FloatArray:
        span = self.config.therapeutic_high - self.config.therapeutic_low
        return np.array(
            [
                self.trough / self.config.therapeutic_high,
                self.damage,
                self.healing,
                (self.last_interval - self.config.therapeutic_low) / max(span, 1e-6),
                self.last_nutrition,
            ],
            dtype=np.float64,
        )

    def step(self, interval: float, nutrition: float) -> StepOutcome:
        rate = self.clearance / self.volume
        peak = self.trough + DOSE / self.volume
        trough_next = peak * float(np.exp(-rate * interval))
        in_range = self.config.therapeutic_low <= trough_next <= self.config.therapeutic_high
        drug_effect = _sigmoid(trough_next - self.config.therapeutic_low)
        burden_active = self.patient_phi * _sigmoid(trough_next - self.config.therapeutic_low)
        toxicity = self.patient_phi * max(0.0, trough_next - self.config.therapeutic_high)

        damage_next = self.damage + self.config.dt * (
            self.config.damage_gain * burden_active
            - self.config.healing_rate * drug_effect
            - self.config.nutrition_gain * nutrition
        )
        damage_next = float(np.clip(damage_next, 0.0, 1.0))
        healing_next = 1.0 - damage_next

        self.trough = trough_next
        self.damage = damage_next
        self.healing = healing_next
        self.last_interval = interval
        self.last_nutrition = nutrition

        benefit = float(in_range) + 0.5 * healing_next
        penalty = self.config.burden_sensitivity * (burden_active * float(in_range) + toxicity)
        return StepOutcome(
            state=self.observation(),
            therapeutic_benefit=benefit,
            interaction_penalty=penalty,
            in_range=in_range,
            adverse_exposure=toxicity + 0.25 * burden_active,
            healing=healing_next,
        )


@dataclass
class EnteralEnv:
    config: TwinConfig
    patient_phi: float
    rng: np.random.Generator
    twin: CrohnsTwin = field(init=False)
    _step_index: int = 0

    def __post_init__(self) -> None:
        self.twin = self._spawn()

    def _spawn(self) -> CrohnsTwin:
        clearance = self.config.clearance * float(self.rng.uniform(0.7, 1.3))
        volume = self.config.volume * float(self.rng.uniform(0.8, 1.2))
        twin = CrohnsTwin(self.config, self.patient_phi, clearance, volume)
        twin.damage = float(self.rng.uniform(0.4, 0.7))
        twin.healing = 1.0 - twin.damage
        return twin

    def reset(self) -> FloatArray:
        self._step_index = 0
        self.twin = self._spawn()
        return self.twin.observation()

    def step(self, action: FloatArray) -> tuple[FloatArray, StepOutcome, bool]:
        interval = float(np.clip(action[0], self.config.therapeutic_low, 24.0))
        nutrition = float(np.clip(action[1], 0.0, 1.0))
        outcome = self.twin.step(interval, nutrition)
        self._step_index += 1
        done = self._step_index >= self.config.horizon
        return outcome.state, outcome, done
