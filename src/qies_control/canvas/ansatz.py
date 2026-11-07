from qies_control.atelier.types import ComplexArray, FloatArray
from qies_control.canvas.statevector import apply_cnot, apply_ry, apply_rz, zero_state


def parameter_count(n_qubits: int, depth: int) -> int:
    return depth * n_qubits * 2


def hardware_efficient(params: FloatArray, n_qubits: int, depth: int) -> ComplexArray:
    grid = params.reshape(depth, n_qubits, 2)
    state = zero_state(n_qubits)
    for layer in range(depth):
        for qubit in range(n_qubits):
            state = apply_ry(state, float(grid[layer, qubit, 0]), qubit, n_qubits)
            state = apply_rz(state, float(grid[layer, qubit, 1]), qubit, n_qubits)
        for qubit in range(n_qubits - 1):
            state = apply_cnot(state, qubit, qubit + 1, n_qubits)
    return state
