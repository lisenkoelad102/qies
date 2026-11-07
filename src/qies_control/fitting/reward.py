def base_reward(
    therapeutic_benefit: float, interaction_penalty: float, penalty_weight: float
) -> float:
    return therapeutic_benefit - penalty_weight * interaction_penalty


def shaped_reward(
    reward: float, potential_now: float, potential_next: float, coefficient: float, gamma: float
) -> float:
    return reward + coefficient * (gamma * potential_next - potential_now)
