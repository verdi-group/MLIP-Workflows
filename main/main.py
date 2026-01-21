#!/usr/bin/env python3
from __future__ import annotations

#TODO: 
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from ase.io import read, write

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from scripts.get_calc import get_calc_object
from scripts.relax import relax
from scripts.phonons import (
    compute_band_structure,
    compute_phonon_dos,
    compute_phonons,
    print_dos_summary,
)
from scripts.plot import plot_dispersion_with_dos, plot_phonon_dos


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_path(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    value_path = Path(value)
    if value_path.is_absolute():
        return value_path
    return root / value_path


def _build_output_names(model_name: str, material: str) -> dict[str, str]:
    base = f"{model_name}_{material}"
    return {
        "relax_traj": f"{base}_relax.traj",
        "relaxed_poscar": f"{base}_relaxed.poscar",
        "phonons_dir": f"{base}_phonons",
        "phonon_force_constants": f"{base}_force_constants.npy",
        "phonon_dos_npz": f"{base}_phonon_dos.npz",
        "phonon_band_structure_json": f"{base}_phonon_band_structure.json",
        "phonon_dispersion_dos_png": f"{base}_phonon_dispersion_dos.png",
        "phonon_dos_png": f"{base}_phonon_dos.png",
        "phonon_bandpath": f"{material}_bandpath-object.json",
    }


def main() -> int:
    # bash arguments (model_name)
    parser = argparse.ArgumentParser(
        description="Compute phonons, DOS, and optional band structure for a model.",
    )
    parser.add_argument(
        "model_name",
        help="Model name from main/config.yml.",
    )
    args = parser.parse_args()

    # ════════════ LOADING CONFIG ════════════
    config = _load_yaml(repo_root / "main" / "config.yml")
    models = config.get("models", {})
    model_entry = models[args.model_name]
    if args.model_name not in models: # Check model exists
        model_list = ", ".join(sorted(models.keys()))
        raise SystemExit(f"Unknown model '{args.model_name}'. Options: {model_list}")
    
    material = model_entry["material"]
    structures = config.get("structures", {})
    
    if material not in structures: # Check material exists 
        raise SystemExit(f"Missing structure config for material '{material}'.")

    structure_entry = structures[material]
    phonon_cfg = structure_entry["phonons"]
    kpoint_dict = structure_config["kpoint_dictionary"]

    if bool(phonon_cfg.get("want_band_structure", False)):
        structure_path = _resolve_path(repo_root, phonon_cfg["band_structure"]["primitive_cell_path"])
        supercell = tuple(int(v) for v in phonon_cfg["supercell_matrix"])
    else: 
        structure_path = _resolve_path(repo_root, structure_entry.get["file_path"])
        supercell = (1,1,1)

    atoms = read(structure_path)
    calc = get_calc_object(args.model_name)

    results_root = repo_root / "results" / args.model_name / material
    raw_dir = results_root / "raw"
    plot_dir = results_root / "plot"
    raw_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    


    output_names = _build_output_names(args.model_name, material)

    if not structure_entry.get("is_file_relaxed", False):
        atoms.calc = calc
        atoms = relax(atoms, outdir=raw_dir, filename= output_names["relax_traj"])
        write(raw_dir / Path(output_names["relaxed_poscar"]), atoms, format="vasp")

    delta = float(phonon_cfg["delta"])
    ph = compute_phonons(
        atoms,
        calc,
        supercell=supercell,
        delta=delta,
        outdir=raw_dir / Path(output_names["phonons_dir"]),
    )

    np.save(raw_dir / output_names["phonon_force_constants"], ph.get_force_constant())

    dos_cfg = phonon_cfg["dos"]
    dos_energies_ev, dos_weights_per_ev = compute_phonon_dos(
        ph,
        kpts=tuple(int(v) for v in dos_cfg["kpts"]),
        npts=int(dos_cfg["npts"]),
        width_ev=float(dos_cfg["width_ev"]),
    )
    np.savez(
        raw_dir / Path(output_names["phonon_dos_npz"]),
        energies_ev=dos_energies_ev,
        weights_per_ev=dos_weights_per_ev,
    )
    print_dos_summary(dos_energies_ev, dos_weights_per_ev, len(atoms))

    want_band_structure = bool(phonon_cfg.get("want_band_structure", False))
    if want_band_structure:
        crystal_type = structure_entry["crystal_type"]
        special_points = kpoint_dict[crystal_type]["special_points"]
        band_cfg = phonon_cfg.get("band_structure", {})
        band_path_key = band_cfg.get("band_path")
        primitive_path = band_cfg.get("primitive_cell_path") or band_cfg.get(
            "primtive_cell_path"
        )
        band_path = kpoint_dict[crystal_type]["paths"][band_path_key]["path"]
        band_atoms = read(_resolve_path(repo_root, primitive_path)) if primitive_path else atoms

        bs, path = compute_band_structure(
            ph,
            band_atoms,
            band_path,
            special_points,
        )
        path.write(str(raw_dir/output_names["phonon_bandpath"]))

        bs.write(str(raw_dir/output_names["phonon_band_structure_json"]))

        plot_dispersion_with_dos(
            bs,
            dos_energies_ev,
            dos_weights_per_ev,
            material,
            outdir=str(plot_dir/output_names["phonon_dispersion_dos_png"]),
        )

    plot_phonon_dos(
        dos_energies_ev,
        dos_weights_per_ev,
        outdir=str(plot_dir/output_names["phonon_dos_png"]),
        title=f"{material} phonon DOS",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


