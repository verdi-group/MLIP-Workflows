#!/usr/bin/env python3
"""
Convert VASP OUTCAR(/.gz) from NEB image folders (e.g. 00..06) into MACE-ready extxyz.

We write keys as REF_energy / REF_forces
Forces are parsed from the OUTCAR table:
  POSITION  TOTAL-FORCE (eV/Angst)
"""

from __future__ import annotations

import argparse
import gzip
import re
from pathlib import Path
from typing import IO, Iterable


FORCE_HDR = "POSITION                                       TOTAL-FORCE (eV/Angst)"
TOTEN_RE = re.compile(r"free\s+energy\s+TOTEN\s+=\s+([-0-9.]+)\s+eV")
VRHFIN_RE = re.compile(r"VRHFIN\s*=\s*([A-Za-z]+)\s*:")
IONS_PER_TYPE_RE = re.compile(r"ions per type\s*=\s*(.*)")


def open_text(p: Path) -> IO[str]:
    return gzip.open(p, "rt", errors="replace") if p.suffix == ".gz" else open(p, "rt", errors="replace")


def find_outcar(image_dir: Path) -> Path:
    if (image_dir / "OUTCAR").exists():
        return image_dir / "OUTCAR"
    if (image_dir / "OUTCAR.gz").exists():
        return image_dir / "OUTCAR.gz"
    raise FileNotFoundError(f"Missing OUTCAR/OUTCAR.gz in {image_dir}")


def parse_header(outcar: Path) -> tuple[list[str], list[list[float]]]:
    species: list[str] = []
    counts: list[int] | None = None
    lattice: list[list[float]] | None = None

    with open_text(outcar) as f:
        it = iter(f)
        for line in it:
            m = VRHFIN_RE.search(line)
            if m:
                el = m.group(1)
                if el not in species:
                    species.append(el)
            m = IONS_PER_TYPE_RE.search(line)
            if m and counts is None:
                counts = [int(x) for x in m.group(1).split()]
            if lattice is None and "direct lattice vectors" in line:
                lattice = []
                for _ in range(3):
                    parts = next(it).split()
                    lattice.append([float(parts[0]), float(parts[1]), float(parts[2])])

            if species and counts is not None and lattice is not None:
                break

    if not species or counts is None or lattice is None:
        raise ValueError(f"Could not parse species/counts/lattice from {outcar}")
    if len(species) != len(counts):
        raise ValueError(f"Species/count mismatch in {outcar}: {species} vs {counts}")

    symbols: list[str] = []
    for el, n in zip(species, counts):
        symbols.extend([el] * n)

    return symbols, lattice


def format_lattice(lattice: list[list[float]]) -> str:
    return " ".join(f"{x:.16g}" for row in lattice for x in row)


def subtract_d3_correction(
    symbols: list[str],
    lattice: list[list[float]],
    positions: list[tuple[float, float, float]],
    forces: list[tuple[float, float, float]],
    energy_ev: float,
    *,
    method: str = "pbe",
    damping: str = "d3bj",
    params_tweaks: dict[str, float] | None = None,
    realspace_cutoff: dict[str, float] | None = None,
    cache_api: bool = True,
) -> tuple[float, list[tuple[float, float, float]], float]:
    """Return labels with the D3 energy/forces subtracted off.

    The returned tuple is:
      (energy_without_d3_ev, forces_without_d3_ev_per_ang, d3_energy_ev)

    This function imports ASE/s-dftd3 lazily so the script still works in
    environments that only need counting or plain OUTCAR conversion.
    """
    
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
    f: IO[str],
    symbols: list[str],
    lattice: list[list[float]],
    positions: list[tuple[float, float, float]],
    forces: list[tuple[float, float, float]],
    energy_ev: float,
    image: str,
    ionic_step: int,
) -> None:
    n = len(symbols)
    f.write(f"{n}\n")
    f.write(
        'Lattice="{lat}" Properties=species:S:1:pos:R:3:REF_forces:R:3 '
        'REF_energy={E:.16g} pbc="T T T" neb_image="{img}" ionic_step={step}\n'.format(
            lat=format_lattice(lattice),
            E=energy_ev,
            img=image,
            step=ionic_step,
        )
    )
    for sym, (x, y, z), (fx, fy, fz) in zip(symbols, positions, forces):
        f.write(f"{sym} {x:.16g} {y:.16g} {z:.16g} {fx:.16g} {fy:.16g} {fz:.16g}\n")


def iter_force_tables(outcar: Path, n_atoms: int) -> Iterable[tuple[int, float, list[tuple[float, float, float]], list[tuple[float, float, float]]]]:
    """give (ionic_step, energy_ev, positions, forces) for each ionic step."""
    with open_text(outcar) as f:
        ionic_step = -1
        it = iter(f)
        for line in it:
            if FORCE_HDR not in line:
                continue
            ionic_step += 1
            next(it, "")  # dashed separator

            pos: list[tuple[float, float, float]] = []
            frc: list[tuple[float, float, float]] = []
            for _ in range(n_atoms):
                parts = next(it).split()
                x, y, z, fx, fy, fz = map(float, parts[:6])
                pos.append((x, y, z))
                frc.append((fx, fy, fz))

            energy = None
            for line2 in it:
                m = TOTEN_RE.search(line2)
                if m:
                    energy = float(m.group(1))
                    break
                if FORCE_HDR in line2:
                    raise ValueError(f"Next force table encountered before TOTEN in {outcar} (step={ionic_step})")
            if energy is None:
                raise ValueError(f"Missing TOTEN after force table in {outcar} (step={ionic_step})")

            yield ionic_step, energy, pos, frc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--neb-root", type=Path, default=Path("assets/training_data/CsPbI3/I_vac_0/output1"))
    ap.add_argument("--images", default="00,01,02,03,04,05,06")
    ap.add_argument("--out-dir", type=Path, default=Path("assets/training_data/CsPbI3/I_vac_0/processed_mace"))
    ap.add_argument("--prefix", default="ivac0_neb")
    ap.add_argument("--stride", type=int, default=1, help="Keep every Nth ionic step (per image).")
    ap.add_argument("--last-only", action="store_true", help="Only keep the final ionic step for each image.")
    ap.add_argument("--val-images", default="02")
    ap.add_argument("--test-images", default="03")
    ap.add_argument("--no-split", action="store_true")
    ap.add_argument("--count-only", action="store_true")
    ap.add_argument(
        "--remove-d3",
        action="store_true",
        help="Subtract a D3 correction from the parsed energies/forces before writing extxyz.",
    )
    ap.add_argument(
        "--d3-method",
        default="pbe",
        help="DFT-D3 method name used to parameterize the correction, e.g. pbe.",
    )
    ap.add_argument(
        "--d3-damping",
        default="d3bj",
        help="DFT-D3 damping scheme, e.g. d3bj or d3zero.",
    )
    ap.add_argument(
        "--d3-cache-api",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse the internal D3 API object across frames when possible.",
    )
    ap.add_argument(
        "--d3-param-tweak",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra D3 damping parameter tweak, repeatable, e.g. --d3-param-tweak s9=0.0",
    )
    args = ap.parse_args()

    images = [x.strip() for x in args.images.split(",") if x.strip()]
    val_images = {x.strip() for x in args.val_images.split(",") if x.strip()}
    test_images = {x.strip() for x in args.test_images.split(",") if x.strip()}
    if val_images & test_images:
        raise SystemExit("val-images and test-images overlap")
    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")

    if args.remove_d3:
        print("WARNING: d3 term correction will be subtracted.")

    d3_params_tweaks: dict[str, float] | None = None
    if args.d3_param_tweak:
        d3_params_tweaks = {}
        for item in args.d3_param_tweak:
            if "=" not in item:
                raise SystemExit(f"Invalid --d3-param-tweak '{item}', expected KEY=VALUE")
            key, value = item.split("=", 1)
            key = key.strip()
            if not key:
                raise SystemExit(f"Invalid --d3-param-tweak '{item}', empty key")
            try:
                d3_params_tweaks[key] = float(value)
            except ValueError as exc:
                raise SystemExit(f"Invalid float in --d3-param-tweak '{item}'") from exc

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_all = args.out_dir / f"{args.prefix}_all.extxyz"
    out_train = args.out_dir / f"{args.prefix}_train.extxyz"
    out_val = args.out_dir / f"{args.prefix}_val.extxyz"
    out_test = args.out_dir / f"{args.prefix}_test.extxyz"

    n_all = n_train = n_val = n_test = 0
    per_image = {}

    fout_all = None if args.count_only else open(out_all, "wt")
    fout_train = fout_val = fout_test = None
    if not args.count_only and not args.no_split:
        fout_train = open(out_train, "wt")
        fout_val = open(out_val, "wt")
        fout_test = open(out_test, "wt")

    try:
        for img in images:
            outcar = find_outcar(args.neb_root / img)
            symbols, lattice = parse_header(outcar)
            n_atoms = len(symbols)

            wrote_steps = set()
            last_buf = None  # (step, E, pos, frc)
            count_img = 0

            for step, E, pos, frc in iter_force_tables(outcar, n_atoms):
                if args.last_only:
                    last_buf = (step, E, pos, frc)
                    continue

                if args.stride == 1 or (step % args.stride == 0):
                    E_use = E
                    frc_use = frc
                    if args.remove_d3:
                        
                        E_use, frc_use, _ = subtract_d3_correction(
                            symbols,
                            lattice,
                            pos,
                            frc,
                            E,
                            method=args.d3_method,
                            damping=args.d3_damping,
                            params_tweaks=d3_params_tweaks,
                            cache_api=args.d3_cache_api,
                        )
                    count_img += 1
                    wrote_steps.add(step)
                    if not args.count_only:
                        write_frame(fout_all, symbols, lattice, pos, frc_use, E_use, img, step)
                        n_all += 1
                        if not args.no_split:
                            if img in test_images:
                                write_frame(fout_test, symbols, lattice, pos, frc_use, E_use, img, step)
                                n_test += 1
                            elif img in val_images:
                                write_frame(fout_val, symbols, lattice, pos, frc_use, E_use, img, step)
                                n_val += 1
                            else:
                                write_frame(fout_train, symbols, lattice, pos, frc_use, E_use, img, step)
                                n_train += 1
                last_buf = (step, E, pos, frc)

            if last_buf is not None:
                step, E, pos, frc = last_buf
                if args.last_only:
                    E_use = E
                    frc_use = frc
                    if args.remove_d3:
                        E_use, frc_use, _ = subtract_d3_correction(
                            symbols,
                            lattice,
                            pos,
                            frc,
                            E,
                            method=args.d3_method,
                            damping=args.d3_damping,
                            params_tweaks=d3_params_tweaks,
                            cache_api=args.d3_cache_api,
                        )
                    count_img = 1
                    if not args.count_only:
                        write_frame(fout_all, symbols, lattice, pos, frc_use, E_use, img, step)
                        n_all += 1
                        if not args.no_split:
                            if img in test_images:
                                write_frame(fout_test, symbols, lattice, pos, frc_use, E_use, img, step)
                                n_test += 1
                            elif img in val_images:
                                write_frame(fout_val, symbols, lattice, pos, frc_use, E_use, img, step)
                                n_val += 1
                            else:
                                write_frame(fout_train, symbols, lattice, pos, frc_use, E_use, img, step)
                                n_train += 1
                else:
                    # If striding, always include final step.
                    if args.stride > 1 and step not in wrote_steps:
                        E_use = E
                        frc_use = frc
                        if args.remove_d3:
                            E_use, frc_use, _ = subtract_d3_correction(
                                symbols,
                                lattice,
                                pos,
                                frc,
                                E,
                                method=args.d3_method,
                                damping=args.d3_damping,
                                params_tweaks=d3_params_tweaks,
                                cache_api=args.d3_cache_api,
                            )
                        count_img += 1
                        if not args.count_only:
                            write_frame(fout_all, symbols, lattice, pos, frc_use, E_use, img, step)
                            n_all += 1
                            if not args.no_split:
                                if img in test_images:
                                    write_frame(fout_test, symbols, lattice, pos, frc_use, E_use, img, step)
                                    n_test += 1
                                elif img in val_images:
                                    write_frame(fout_val, symbols, lattice, pos, frc_use, E_use, img, step)
                                    n_val += 1
                                else:
                                    write_frame(fout_train, symbols, lattice, pos, frc_use, E_use, img, step)
                                    n_train += 1

            per_image[img] = count_img

    finally:
        for fh in (fout_all, fout_train, fout_val, fout_test):
            if fh is not None:
                fh.close()

    print("Frames per image (after stride/last-only):")
    for img in images:
        print(f"  {img}: {per_image.get(img, 0)}")
    if args.count_only:
        print("TOTAL:", sum(per_image.values()))
        return 0

    print("Wrote:", out_all, f"({n_all})")
    if not args.no_split:
        print("Wrote:", out_train, f"({n_train})")
        print("Wrote:", out_val, f"({n_val})")
        print("Wrote:", out_test, f"({n_test})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
