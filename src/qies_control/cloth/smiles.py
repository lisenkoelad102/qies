from qies_control.atelier.types import Molecule

_TWO_LETTER = {"Cl", "Br"}
_ORGANIC = {"B", "C", "N", "O", "P", "S", "F", "I"}
_AROMATIC = "bcnops"
_BOND_ORDER = {"-": 1.0, "=": 2.0, "#": 3.0, ":": 1.5, "/": 1.0, "\\": 1.0}


def _element_from_bracket(inner: str) -> str:
    cursor = 0
    while cursor < len(inner) and not inner[cursor].isalpha():
        cursor += 1
    if cursor >= len(inner):
        return "C"
    head = inner[cursor].upper()
    if cursor + 1 < len(inner) and inner[cursor + 1].islower():
        candidate = head + inner[cursor + 1]
        if candidate in _TWO_LETTER:
            return candidate
    return head


def parse_smiles(text: str) -> Molecule:
    symbols: list[str] = []
    bonds: list[tuple[int, int, float]] = []
    ring_open: dict[str, tuple[int, float]] = {}
    branch_stack: list[int | None] = []
    previous: int | None = None
    pending = 1.0
    cursor = 0
    length = len(text)

    def attach(symbol: str) -> int:
        nonlocal previous, pending
        index = len(symbols)
        symbols.append(symbol)
        if previous is not None:
            bonds.append((previous, index, pending))
        previous = index
        pending = 1.0
        return index

    while cursor < length:
        char = text[cursor]
        if char == "(":
            branch_stack.append(previous)
            cursor += 1
        elif char == ")":
            previous = branch_stack.pop() if branch_stack else previous
            cursor += 1
        elif char in _BOND_ORDER:
            pending = _BOND_ORDER[char]
            cursor += 1
        elif char == "[":
            end = text.find("]", cursor)
            if end == -1:
                break
            symbol = _element_from_bracket(text[cursor + 1 : end])
            if symbol != "H":
                attach(symbol)
            cursor = end + 1
        elif char.isdigit():
            if char in ring_open:
                other, order = ring_open.pop(char)
                if previous is not None:
                    bonds.append((other, previous, max(order, pending)))
            elif previous is not None:
                ring_open[char] = (previous, pending)
            pending = 1.0
            cursor += 1
        else:
            pair = text[cursor : cursor + 2]
            if pair in _TWO_LETTER:
                attach(pair)
                cursor += 2
            elif char.upper() in _ORGANIC or char in _AROMATIC:
                attach(char.upper())
                cursor += 1
            else:
                cursor += 1

    if not symbols:
        return Molecule(("C",), ())
    return Molecule(tuple(symbols), tuple(bonds))


def heavy_atom_count(text: str) -> int:
    return parse_smiles(text).n_heavy
