from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import optim
from torch.nn import functional as F

from qies_control.atelier.optim import descend
from qies_control.atelier.seeding import set_seed
from qies_control.atelier.settings import ControllerConfig
from qies_control.atelier.types import FloatArray
from qies_control.cloth.twin import STATE_DIM, EnteralEnv
from qies_control.fitting.controller import ActorCritic
from qies_control.fitting.reward import base_reward, shaped_reward
from qies_control.fitting.state import augment_state, burden_signal
from qies_control.notions.clinical import (
    adverse_interaction_exposure,
    mucosal_healing_proxy,
    time_in_therapeutic_range,
)

EnvFactory = Callable[[], EnteralEnv]
ActionFn = Callable[[FloatArray], FloatArray]


@dataclass(frozen=True)
class ControlFlags:
    use_energy: bool = True
    use_shaping: bool = True


def _discounted(rewards: list[float], gamma: float) -> FloatArray:
    out = np.zeros(len(rewards), dtype=np.float64)
    running = 0.0
    for index in reversed(range(len(rewards))):
        running = rewards[index] + gamma * running
        out[index] = running
    return out


def _potential(observation: FloatArray, patient_phi: float, flags: ControlFlags) -> float:
    return burden_signal(observation, patient_phi) if flags.use_energy else 0.0


class ControllerTrainer:
    def __init__(self, config: ControllerConfig) -> None:
        self.config = config

    def train(
        self, env_factory: EnvFactory, flags: ControlFlags, episodes: int, seed: int = 0
    ) -> tuple[ActorCritic, list[float]]:
        set_seed(seed)
        policy = ActorCritic(STATE_DIM + 1, self.config)
        optimizer = optim.Adam(policy.parameters(), lr=self.config.lr)
        history: list[float] = []
        for _ in range(episodes):
            env = env_factory()
            observation = env.reset()
            potential = _potential(observation, env.patient_phi, flags)
            augmented = augment_state(observation, potential)
            log_probs: list[torch.Tensor] = []
            values: list[torch.Tensor] = []
            entropies: list[torch.Tensor] = []
            rewards: list[float] = []
            done = False
            while not done:
                action, log_prob, value, entropy = policy.act(augmented)
                next_observation, outcome, done = env.step(action)
                reward = base_reward(
                    outcome.therapeutic_benefit,
                    outcome.interaction_penalty,
                    self.config.interaction_penalty,
                )
                next_potential = _potential(next_observation, env.patient_phi, flags)
                if flags.use_shaping:
                    reward = shaped_reward(
                        reward, potential, next_potential, self.config.shaping_c, self.config.gamma
                    )
                log_probs.append(log_prob)
                values.append(value)
                entropies.append(entropy)
                rewards.append(reward)
                potential = next_potential
                augmented = augment_state(next_observation, next_potential)
            returns = torch.tensor(_discounted(rewards, self.config.gamma), dtype=torch.float32)
            value_tensor = torch.stack(values)
            advantage = returns - value_tensor.detach()
            policy_loss = -(torch.stack(log_probs) * advantage).mean()
            value_loss = F.mse_loss(value_tensor, returns)
            entropy_term = torch.stack(entropies).mean()
            loss = (
                policy_loss
                + self.config.value_coef * value_loss
                - self.config.entropy_coef * entropy_term
            )
            descend(optimizer, loss)
            history.append(float(loss.item()))
        return policy, history


def rule_based_action(config: ControllerConfig) -> FloatArray:
    interval = 0.5 * (config.interval_low + config.interval_high)
    return np.array([interval, 0.5], dtype=np.float64)


def evaluate_policy(
    action_fn: ActionFn, env: EnteralEnv, flags: ControlFlags, episodes: int
) -> dict[str, float]:
    in_range: list[int] = []
    adverse: list[float] = []
    healing: list[float] = []
    for _ in range(episodes):
        observation = env.reset()
        potential = _potential(observation, env.patient_phi, flags)
        augmented = augment_state(observation, potential)
        done = False
        while not done:
            action = action_fn(augmented)
            next_observation, outcome, done = env.step(action)
            in_range.append(int(outcome.in_range))
            adverse.append(outcome.adverse_exposure)
            healing.append(outcome.healing)
            potential = _potential(next_observation, env.patient_phi, flags)
            augmented = augment_state(next_observation, potential)
    return {
        "TTR": time_in_therapeutic_range(np.array(in_range, dtype=np.int64)),
        "AIE": adverse_interaction_exposure(np.array(adverse, dtype=np.float64)),
        "MHP": mucosal_healing_proxy(np.array(healing, dtype=np.float64)),
    }
