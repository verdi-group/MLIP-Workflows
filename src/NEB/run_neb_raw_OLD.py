#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import os

from ase.io import read, write
from ase.optimize import FIRE
from ase.geometry import find_mic
import torch

from ase.mep import SingleCalculatorNEB, NEB

#TODO: ensure that this script is runnable from command-line.
class LoopDetected(RuntimeError):
    pass


class LoopGuard:
    def __init__(
        self,
        opt: FIRE,
        *,
        window: int = 60,
        max_unique: int = 2,
        repeat: int = 30,
        rounding: int = 6,
        label: str = "",
    ):
        self.opt = opt
        self.window = int(window)
        self.max_unique = int(max_unique)
        self.repeat = int(repeat)
        self.rounding = int(rounding)
        self.label = str(label)
        self.history: list[tuple[float, float]] = []
        self.counts: dict[tuple[float, float], int] = {}
        self.last: tuple[float, float] | None = None
        self.last_count = 0

    def __call__(self) -> None:
        forces = self.opt.atoms.get_forces()
        fmax = float(np.sqrt((forces * forces).sum(axis=1)).max())
        energy = float(self.opt.atoms.get_potential_energy())

        key = (round(energy, self.rounding), round(fmax, self.rounding))

        if key == self.last:
            self.last_count += 1
        else:
            self.last = key
            self.last_count = 1

        if self.last_count >= self.repeat:
            label = f" ({self.label})" if self.label else ""
            raise LoopDetected(
                f"Loop guard{label}: same (energy,fmax) repeated {self.last_count} steps: {key}"
            )

        self.history.append(key)
        self.counts[key] = self.counts.get(key, 0) + 1
        if len(self.history) > self.window:
            old = self.history.pop(0)
            cnt = self.counts.get(old, 0) - 1
            if cnt <= 0:
                self.counts.pop(old, None)
            else:
                self.counts[old] = cnt

        if len(self.history) >= self.window and len(self.counts) <= self.max_unique:
            label = f" ({self.label})" if self.label else ""
            raise LoopDetected(
                f"Loop guard{label}: only {len(self.counts)} unique (energy,fmax) values over "
                f"{len(self.history)} steps"
            )


def attach_loop_guard(
    opt: FIRE,
    *,
    window: int = 60,
    max_unique: int = 2,
    repeat: int = 30,
    rounding: int = 6,
    label: str = "",
) -> None:
    guard = LoopGuard(
        opt,
        window=window,
        max_unique=max_unique,
        repeat=repeat,
        rounding=rounding,
        label=label,
    )
    opt.attach(guard, interval=1)



# IMPORTANT: this file lives in <repo>/src/NEB/run_neb_raw.py
# parents[2] => <repo>
repo_root = Path(__file__).resolve().parents[2]
src_root = repo_root / "src"
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from mlip_phonons.get_calc import get_calc_object
from mlip_phonons.relax import relax

structures_dir = repo_root / "assets" / "structures" / "NEB"
poscar_i = structures_dir / "POSCAR_i"
poscar_f = structures_dir / "POSCAR_f"
dft_neb_dat = structures_dir / "neb.dat"  # reference only

models_root = repo_root / "assets" / "models"

device = os.environ.get("MLIP_DEVICE", "cuda")
if device.startswith("cuda") and not torch.cuda.is_available():
    print("CUDA requested but not available; falling back to CPU.")
    device = "cpu"
dtype = os.environ.get("MLIP_DTYPE", "float32")

n_images_fallback = 9




# to quickly converge to an initial rough path
maxstep_mlip_guess = 0.05 # in angstroms, is the maximum atomic displacement per optimiser step
fmax_mlip_guess = 0.03
steps_mlip_guess = 3000
k_spring_mlip = 0.6


k_spring = 0.6

# to finely converge to the final path 
maxstep_mlip_d3 = 0.03 # to get the final path 
fmax_mlip_d3 = 0.03
steps_mlip_d3 = 1400

# to now shift the images up the path to get the maximum. 
maxstep_ci = 0.03
fmax_ci = 0.03
steps_ci = 1000

# below is not used anymmore. 
def hungarian_min(cost: np.ndarray) -> np.ndarray:
    a = np.asarray(cost, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"hungarian_min requires square matrix, got {a.shape}")
    n = int(a.shape[0])

    u = np.zeros(n + 1, dtype=float)
    v = np.zeros(n + 1, dtype=float)
    p = np.zeros(n + 1, dtype=int)
    way = np.zeros(n + 1, dtype=int)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf, dtype=float)
        used = np.zeros(n + 1, dtype=bool)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0

            for j in range(1, n + 1):
                if not used[j]:
                    cur = a[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assign = np.empty(n, dtype=int)
    for j in range(1, n + 1):
        i = p[j]
        if i != 0:
            assign[i - 1] = j - 1
    return assign


def map_final_to_initial_by_species(a, b):

    # purpose is to ensure that we are comparing the atoms correctly such that 
    # we have a minimum number of movements. This means finding the projection of one 
    # coord to another, building a cost matrix for all such projections, and then using hungarian 
    # min to map the indexes of the coordinates to each other such that there is maximal overlap
    
    a2 = a.copy()
    b2 = b.copy()

    cell = a2.cell
    pbc = a2.pbc
    sym = np.array(a2.get_chemical_symbols())
    pos_a = a2.get_positions()
    pos_b = b2.get_positions()
    new_pos_b = pos_b.copy()

    # minimisation is done per species
    
    for el in np.unique(sym):
        idx = np.where(sym == el)[0]
        xa = pos_a[idx]
        xb = pos_b[idx]

        c = np.zeros((len(idx), len(idx)), dtype=float)
        for i in range(len(idx)):
            d = xb - xa[i]
            d, _ = find_mic(d, cell=cell, pbc=pbc)
            c[i, :] = np.linalg.norm(d, axis=1)

        assign = hungarian_min(c)
        new_pos_b[idx] = xb[assign]

    b2.set_positions(new_pos_b)
    return b2


def choose_n_images(path: Path, fallback: int) -> int:
    try:
        data = np.loadtxt(path)
        n = int(np.atleast_2d(data).shape[0])
        return max(n, 3)
    except Exception:
        return int(fallback)


def build_images(a, b, n_images: int):
    if n_images < 3:
        raise ValueError("n_images must be >= 3")
    return [a] + [a.copy() for _ in range(n_images - 2)] + [b]


def energies_relative(images) -> np.ndarray:
    e = np.array([img.get_potential_energy() for img in images], dtype=float)
    return e - e[0]


def reaction_coordinate(images) -> np.ndarray:
    s = [0.0] 
    cell = images[0].cell
    pbc = images[0].pbc
    for a, b in zip(images[:-1], images[1:]):
        # images[:-1] is image n-1 and images[1:] is for images n. 
        d = b.get_positions() - a.get_positions()
        d, _ = find_mic(d, cell=cell, pbc=pbc)
        s.append(s[-1] + float(np.linalg.norm(d)))
    return np.array(s, dtype=float)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model_name")
    args = parser.parse_args()

    model_name = str(args.model_name)

    out_raw = (repo_root / "resultsNEB" / model_name / "raw").resolve()
    out_raw.mkdir(parents=True, exist_ok=True)


    a = read(poscar_i)
    b = read(poscar_f)

    a_mlip = a.copy()
    b_mlip = b.copy()

    if len(a) != len(b):
        raise ValueError("Different atom counts between POSCAR_i and POSCAR_f.")
    if a.get_chemical_symbols() != b.get_chemical_symbols():
        raise ValueError("Species/order differs between POSCAR_i and POSCAR_f.")
    if not np.allclose(a.cell.array, b.cell.array, atol=1e-8):
        raise ValueError("Cells differ between POSCAR_i and POSCAR_f.")

    calc_vdw = get_calc_object(
        model_name,
        models_root=models_root,
        device=device,
        dtype=dtype,
        include_vdw=True,
    )
    calc_mlip = get_calc_object(
        model_name,
        models_root=models_root,
        device=device,
        dtype=dtype,
        include_vdw=False,
    )

    #b = map_final_to_initial_by_species(a, b) # we do not need this anymore. 

    a.calc = calc_vdw
    b.calc = calc_vdw
    a_relaxed = None
    b_relaxed = None

    
    enable_cache = False
    if enable_cache: # allows the use of existing relaxed traj paths for initial and final points
        if os.path.exists(out_raw/'relaxed_traj_i.traj'): 
            a_relaxed = read(out_raw/'relaxed_traj_i.traj', index = -1) # index -1 to pull final
        if os.path.exists(out_raw/'relaxed_traj_f.traj'):
            b_relaxed = read(out_raw/'relaxed_traj_f.traj', index = -1)
    
    if not a_relaxed:
        a_relaxed = relax(a, outdir = out_raw, filename = 'relaxed_traj_i.traj')
        # write("POSCAR_i_relaxed.vasp", a_relaxed, format="vasp") # optional write 
    if not b_relaxed: 
        b_relaxed = relax(b, outdir = out_raw, filename = 'relaxed_traj_f.traj')
        # write("POSCAR_f_relaxed.vasp", b_relaxed, format="vasp") # optional write
    
    # relaxed with mlip + d3 calculator
    a = a_relaxed
    b = b_relaxed 


    #a = relax(a, outdir = out_raw, filename = 'relaxed_traj_i.traj')
    #b = relax(b, outdir = out_raw, filename = 'relaxed_traj_f.traj')

    print('Choosing number of images and building the images')
    n_images = choose_n_images(dft_neb_dat, n_images_fallback)
    
    

    # write the relaxed endpoints
    #write(out_raw / "POSCAR_i_used.vasp", images[0], format="vasp")
    #write(out_raw / "POSCAR_f_used_remapped.vasp", images[-1], format="vasp")

    #neb = SingleCalculatorNEB(images, k=k_spring, climb=False)
    
    images = build_images(a, b, n_images)
    
    
    # initial path with only MLIP  
    for img in images:
        img.calc = calc_mlip

    neb = NEB(images, k=k_spring_mlip, climb=False, allow_shared_calculator = True)

    print('Interpolating middle images')

    neb.interpolate(method="idpp", mic=True)

    print("assigning opts1 (MLIP)")
    opts1 = FIRE(
        neb,
        trajectory=str(out_raw / "neb_mlip.traj"),
        logfile=str(out_raw / "neb_mlip.log"),
        maxstep=maxstep_mlip_guess,
    )
    print("running opts1")
    attach_loop_guard(opts1, label="opts1")
    try:
        opts1.run(fmax=fmax_mlip_guess, steps=steps_mlip_guess)
    except LoopDetected as exc:
        print(str(exc))

  

    for img in images:
        img.calc = calc_vdw

    # reuse the same images; NEB object can be reused or rebuilt
    neb = NEB(images, k=k_spring, climb=False, allow_shared_calculator=True)

    print("assigning opts2 (Refining with D3)")
    opts2 = FIRE(
        neb,
        trajectory=str(out_raw / "neb_mlip_d3.traj"),
        logfile=str(out_raw / "neb_mlip_d3.log"),
        maxstep=maxstep_mlip_d3,
    )
    print("running opts2")
    attach_loop_guard(opts2, label="opts2")
    try:
        opts2.run(fmax=fmax_mlip_d3, steps=steps_mlip_d3)
    except LoopDetected as exc:
        print(str(exc))

    # ════════════ second optimisation *along the path* (get the barrier)
    # neb_ci = SingleCalculatorNEB(images, k=k_spring, climb=True)

    neb_ci = NEB(images, k=k_spring, climb=True, allow_shared_calculator = True)
    print("assigning opt_ci")
    opt_ci = FIRE(
        neb_ci,
        trajectory=str(out_raw / "neb_ci.traj"),
        logfile=str(out_raw / "neb_ci.log"),
        maxstep=maxstep_ci,
    )
    
    print("running opt_ci")
    attach_loop_guard(opt_ci, label="opt_ci")
    try:
        opt_ci.run(fmax=fmax_ci, steps=steps_ci)
    except LoopDetected as exc:
        print(str(exc))

    # ════════════ formatting data to be compared. 

    s_mlip = reaction_coordinate(images)
    # get the cumulative distance along image chain. 
    # (for each consecutive image pair, get the minimmum image displacement of every atom, 
    # and get the euclidean norm of the full displacement. Then sum these)

    e_mlip = energies_relative(images) 
    # get the potential energy for each image
    # and subtract the first images energy. 

    np.savez_compressed(
        out_raw / "neb_raw.npz",
        s_mlip=s_mlip,
        e_mlip=e_mlip,
        n_images=np.array([n_images], dtype=int),
        dft_neb_dat=str(dft_neb_dat),
        poscar_i=str(poscar_i),
        poscar_f=str(poscar_f),
    )

    barrier = float(np.max(e_mlip))
    delta_e = float(e_mlip[-1])
    (out_raw / "summary.txt").write_text(
        f"model={model_name}\n"
        f"n_images={n_images}\n"
        f"barrier_eV={barrier:.6f}\n"
        f"deltaE_eV={delta_e:.6f}\n"
        f"out_raw={out_raw}\n",
        encoding="utf-8",
    )

    print(f"SUCCESS wrote raw NEB to {out_raw}")
    return 0


if __name__ == "__main__":

    raise SystemExit(main())
