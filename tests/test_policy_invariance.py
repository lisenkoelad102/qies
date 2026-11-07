from qies_control.fitting.reward import base_reward, shaped_reward


def test_reward_algebra() -> None:
    assert base_reward(2.0, 1.0, 0.5) == 1.5
    assert abs(shaped_reward(1.0, 0.2, 0.3, 1.0, 0.9) - 1.07) < 1e-9
