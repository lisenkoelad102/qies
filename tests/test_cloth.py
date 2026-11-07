from qies_control.cloth.smiles import heavy_atom_count, parse_smiles


def test_smiles_heavy_atom_counts() -> None:
    assert heavy_atom_count("CCO") == 3
    assert heavy_atom_count("c1ccccc1") == 6
    assert heavy_atom_count("CC(=O)Cl") == 4
    molecule = parse_smiles("C1CC1")
    assert molecule.n_heavy == 3
    assert any(bond[0] == 0 and bond[1] == 2 for bond in molecule.bonds) or any(
        bond[1] == 0 and bond[0] == 2 for bond in molecule.bonds
    )
