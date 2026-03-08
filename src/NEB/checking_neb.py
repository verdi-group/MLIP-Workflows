from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
from ase.geometry import find_mic
from ase.io import read, write

here = Path(__file__).resolve().parent
if str(here) not in sys.path:
    sys.path.insert(0, str(here))

from .neb_tools.neb_analysis import map_final_to_initial_by_species

"""
This script is used to a) produce and save a rearranged poscar_f.vasp file to 
best match the indices in the poscar_i.vasp file, but MAINLY its purpose is to compute 
basic metrics on the displacement patterns between poscar_i.vasp and poscar_f.vasp WITHOUT reordering
and poscar_i.vasp and modified_poscar_f.vasp WITH reordering.

does this by calculating the average interatomic displacement, the number of atoms with displacement greater than 
1 angstrom, and the maximum interatomig displacement. 

The algorithm used to reorder indices based on overlap is the element-wise hungarian 
algorithm implemeneted in run_neb_raw.py and called here. 

"""

def _metrics(a, b) -> tuple[float, int, float]:
    d = b.get_positions() - a.get_positions()
    d, _ = find_mic(d, cell=a.cell, pbc=a.pbc)
    r = np.linalg.norm(d, axis=1)
    return float(r.mean()), int((r > 1.0).sum()), float(r.max())


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    default_structures_dir = repo_root / "assets" / "structures" / "g_CsPbI3_I_vac_0"
    parser = argparse.ArgumentParser()
    parser.add_argument("--structures-dir", type=Path, default=default_structures_dir)
    parser.add_argument("--poscar-i", type=Path, default=None)
    parser.add_argument("--poscar-f", type=Path, default=None)
    args = parser.parse_args()

    structures_dir = args.structures_dir
    poscar_i = args.poscar_i or (structures_dir / "POSCAR_i")
    poscar_f = args.poscar_f or (structures_dir / "POSCAR_f")

    a = read(poscar_i)
    b = read(poscar_f)

    b_mapped = map_final_to_initial_by_species(a, b)
    modified_poscar_f = structures_dir / "modified_poscar_f.vasp"
    write(modified_poscar_f, b_mapped, format="vasp", direct=True)

    avg_f, n_gt1_f, max_f = _metrics(a, b)
    avg_m, n_gt1_m, max_m = _metrics(a, b_mapped)

    print(f"POSCAR_f: avg={avg_f:.6f}A, >1A={n_gt1_f}, max={max_f:.6f}A")
    print(f"modified: avg={avg_m:.6f}A, >1A={n_gt1_m}, max={max_m:.6f}A")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
