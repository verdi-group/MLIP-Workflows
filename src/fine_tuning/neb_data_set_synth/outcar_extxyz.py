from __future__ import annotations

import gzip
import re
from pathlib import Path
from typing import IO, Iterable


Vector3 = tuple[float, float, float]
Lattice = list[list[float]]
Frame = tuple[int, float, list[Vector3], list[Vector3]]

FORCE_HDR = "POSITION                                       TOTAL-FORCE (eV/Angst)"
TOTEN_RE = re.compile(r"free\s+energy\s+TOTEN\s+=\s+([-0-9.]+)\s+eV")
VRHFIN_RE = re.compile(r"VRHFIN\s*=\s*([A-Za-z]+)\s*:")
IONS_PER_TYPE_RE = re.compile(r"ions per type\s*=\s*(.*)")


def open_text(path: Path) -> IO[str]:
    return gzip.open(path, "rt", errors="replace") if path.suffix == ".gz" else open(path, "rt", errors="replace")


def find_outcar(image_dir: Path) -> Path:
    if (image_dir / "OUTCAR").exists():
        return image_dir / "OUTCAR"
    if (image_dir / "OUTCAR.gz").exists():
        return image_dir / "OUTCAR.gz"
    raise FileNotFoundError(f"Missing OUTCAR/OUTCAR.gz in {image_dir}")


def parse_header(outcar: Path) -> tuple[list[str], Lattice]:
    species: list[str] = []
    counts: list[int] | None = None
    lattice: Lattice | None = None

    with open_text(outcar) as fh:
        lines = iter(fh)
        for line in lines:
            match = VRHFIN_RE.search(line)
            if match:
                element = match.group(1)
                if element not in species:
                    species.append(element)

            match = IONS_PER_TYPE_RE.search(line)
            if match and counts is None:
                counts = [int(value) for value in match.group(1).split()]

            if lattice is None and "direct lattice vectors" in line:
                lattice = []
                for _ in range(3):
                    parts = next(lines).split()
                    lattice.append([float(parts[0]), float(parts[1]), float(parts[2])])

            if species and counts is not None and lattice is not None:
                break

    if not species or counts is None or lattice is None:
        raise ValueError(f"Could not parse species/counts/lattice from {outcar}")
    if len(species) != len(counts):
        raise ValueError(f"Species/count mismatch in {outcar}: {species} vs {counts}")

    symbols: list[str] = []
    for element, count in zip(species, counts):
        symbols.extend([element] * count)
    return symbols, lattice


def format_lattice(lattice: Lattice) -> str:
    return " ".join(f"{value:.16g}" for row in lattice for value in row)


def subtract_d3_correction(
    symbols: list[str],
    lattice: Lattice,
    positions: list[Vector3],
    forces: list[Vector3],
    energy_ev: float,
    *,
    method: str = "pbe",
    damping: str = "d3bj",
    params_tweaks: dict[str, float] | None = None,
    realspace_cutoff: dict[str, float] | None = None,
    cache_api: bool = True,
) -> tuple[float, list[Vector3], float]:
    from ase import Atoms
    from dftd3.ase import DFTD3

    d3_kwargs: dict[str, object] = {
        "method": method,
        "damping": damping,
        "cache_api": cache_api,
    }
    if params_tweaks is not None:
        d3_kwargs["params_tweaks"] = dict(params_tweaks)
    if realspace_cutoff is not None:
        d3_kwargs["realspace_cutoff"] = dict(realspace_cutoff)

    atoms = Atoms(symbols=symbols, positions=positions, cell=lattice, pbc=(True, True, True))
    atoms.calc = DFTD3(**d3_kwargs)

    d3_energy_ev = float(atoms.get_potential_energy())
    d3_forces = atoms.get_forces()
    corrected_forces = [
        (fx - float(dfx), fy - float(dfy), fz - float(dfz))
        for (fx, fy, fz), (dfx, dfy, dfz) in zip(forces, d3_forces)
    ]
    return energy_ev - d3_energy_ev, corrected_forces, d3_energy_ev


def write_frame(
    fh: IO[str],
    symbols: list[str],
    lattice: Lattice,
    positions: list[Vector3],
    forces: list[Vector3],
    energy_ev: float,
    image: str,
    ionic_step: int,
) -> None:
    fh.write(f"{len(symbols)}\n")
    fh.write(
        'Lattice="{lat}" Properties=species:S:1:pos:R:3:REF_forces:R:3 '
        'REF_energy={energy:.16g} pbc="T T T" neb_image="{image}" ionic_step={step}\n'.format(
            lat=format_lattice(lattice),
            energy=energy_ev,
            image=image,
            step=ionic_step,
        )
    )
    for symbol, (x, y, z), (fx, fy, fz) in zip(symbols, positions, forces):
        fh.write(f"{symbol} {x:.16g} {y:.16g} {z:.16g} {fx:.16g} {fy:.16g} {fz:.16g}\n")


def iter_force_tables(outcar: Path, n_atoms: int) -> Iterable[Frame]:
    with open_text(outcar) as fh:
        ionic_step = -1
        lines = iter(fh)
        for line in lines:
            if FORCE_HDR not in line:
                continue

            ionic_step += 1
            next(lines, "")
            positions: list[Vector3] = []
            forces: list[Vector3] = []
            for _ in range(n_atoms):
                parts = next(lines).split()
                x, y, z, fx, fy, fz = map(float, parts[:6])
                positions.append((x, y, z))
                forces.append((fx, fy, fz))

            energy_ev = None
            for next_line in lines:
                match = TOTEN_RE.search(next_line)
                if match:
                    energy_ev = float(match.group(1))
                    break
                if FORCE_HDR in next_line:
                    raise ValueError(f"Next force table encountered before TOTEN in {outcar} (step={ionic_step})")
            if energy_ev is None:
                raise ValueError(f"Missing TOTEN after force table in {outcar} (step={ionic_step})")

            yield ionic_step, energy_ev, positions, forces


def maybe_subtract_d3(
    remove_d3: bool,
    symbols: list[str],
    lattice: Lattice,
    positions: list[Vector3],
    forces: list[Vector3],
    energy_ev: float,
    *,
    d3_method: str,
    d3_damping: str,
    d3_params_tweaks: dict[str, float] | None,
    d3_cache_api: bool,
) -> tuple[float, list[Vector3]]:
    if not remove_d3:
        return energy_ev, forces
    corrected_energy, corrected_forces, _ = subtract_d3_correction(
        symbols,
        lattice,
        positions,
        forces,
        energy_ev,
        method=d3_method,
        damping=d3_damping,
        params_tweaks=d3_params_tweaks,
        cache_api=d3_cache_api,
    )
    return corrected_energy, corrected_forces
