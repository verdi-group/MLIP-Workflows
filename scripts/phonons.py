from __future__ import annotations

from pathlib import Path
import shutil
import numpy as np

from ase import Atoms
from ase.phonons import Phonons

EV_TO_THz = 241.79893
def compute_phonons(
    structure: Atoms,
    calculatoresults/pet-madr,
    supercell: tuple[int, int, int] = (2, 2, 2),
    delta: float = 0.01,
    outdir: Path | None = None,
) -> Phonons:
    """Run finite-displacement phonons using ASE. Will overwrite outdir. 

    input:
        structure: Primitive relaxed reference structure.
        calculator: ASE-compatible calculator (e.g. MLIP).
        supercell: Supercell size for force constants.
        delta: Displacement amplitude (Angstrom).
        outdir: Directory for phonon cache.

    output:
        ASE Phonons object with cache saved to outdir. This cache is accessible 
        by calling 
        >>>ph = Phonons(structure, calculator, supercell=supercell, 
                        delta=delta, name=str(outdir))
    )
    """

    if outdir is not None:
        if outdir.exists():
            shutil.rmtree(outdir)
        outdir.mkdir(parents=True)

    structure = structure.copy()
    structure.calc = calculator

    ph = Phonons(
        structure,
        calculator,
        supercell=supercell,
        delta=delta,
        name=str(outdir) if outdir else None,
    )

    ph.run()
    ph.read(acoustic=True)
    ph.clean()

    return ph

def compute_band_structure(
    ph: Phonons,
    structure: Atoms,
    k_path: str,
    special_points: dict[str, list[float]],
    npoints: int = 200,
):
    """
    Compute phonon band structure along a k-path.
    """
    path = structure.cell.bandpath(
        k_path,
        npoints=npoints,
        special_points=special_points,
    )

    bs = ph.get_band_structure(path)
    return bs, path


def compute_phonon_dos(
    ph: Phonons,
    kpts: tuple[int, int, int] = (12, 12, 12),
    npts: int = 4000,
    width_ev: float = 1e-3,
):
    """
    Compute phonon DOS on grid.
    """
    dos = ph.get_dos(kpts=kpts).sample_grid(
        npts=npts,
        width=width_ev,
    )

    energies = np.asarray(dos.get_energies())
    weights = np.asarray(dos.get_weights())

    return energies, weights


def print_dos_summary(energies_ev: np.ndarray, weights: np.ndarray, natoms: int):
    modes = np.trapezoid(weights, energies_ev)

    print("DOS summary")
    print("-----------")
    print(f"∫ g(E) dE = {modes:.3f}")
    print(f"Expected modes = {3 * natoms}")
    print(f"Min energy (eV) = {energies_ev.min():.6f}")
    print(f"Min freq (THz) = {energies_ev.min() * EV_TO_THz:.3f}")
