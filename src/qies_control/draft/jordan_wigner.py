from qies_control.draft.pauli import PauliString, PauliSum


def annihilation(mode: int, n_modes: int) -> PauliSum:
    z_string = [3] * mode
    tail = [0] * (n_modes - mode - 1)
    x_key: PauliString = tuple(z_string + [1] + tail)
    y_key: PauliString = tuple(z_string + [2] + tail)
    return PauliSum(n_modes, {x_key: 0.5 + 0.0j, y_key: 0.5j})


def creation(mode: int, n_modes: int) -> PauliSum:
    return annihilation(mode, n_modes).dagger()


def number(mode: int, n_modes: int) -> PauliSum:
    identity_key: PauliString = (0,) * n_modes
    z_key = [0] * n_modes
    z_key[mode] = 3
    return PauliSum(n_modes, {identity_key: 0.5 + 0.0j, tuple(z_key): -0.5 + 0.0j})
