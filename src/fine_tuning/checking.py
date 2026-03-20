#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

"""

small script to check that d3 term correction was successfully removed. 
turns out it was. got a result of around 39eV removed, which is acting as a near
constant offset . gives around 0.246eV/atom which is naively believable. 
"""

ref_energy = re.compile(r"REF_energy=([^\s]+)")


def read_extxyz(path: Path) -> list[tuple[float, list[tuple[float, float, float]]]]:
    frames = []
    with path.open() as f:
        while True:
            line = f.readline()
            if not line:
                break
            n = int(line.strip())
            header = f.readline()
            energy = float(ref_energy.search(header).group(1))
            forces = []
            for _ in range(n):
                parts = f.readline().split()
                forces.append((float(parts[4]), float(parts[5]), float(parts[6])))
            frames.append((energy, forces))
    return frames


def norm3(v: tuple[float, float, float]) -> float:
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reference", type=Path)
    ap.add_argument("corrected", type=Path)
    args = ap.parse_args()

    ref = read_extxyz(args.reference)
    corr = read_extxyz(args.corrected)

    energy_deltas = [ec - er for (er, _), (ec, _) in zip(ref, corr)]
    energy_props = [abs(de) / abs(er) for (er, _), de in zip(ref, energy_deltas)]

    force_deltas = []
    force_props = []
    for (_, fr), (_, fc) in zip(ref, corr):
        for vr, vc in zip(fr, fc):
            dv = (vc[0] - vr[0], vc[1] - vr[1], vc[2] - vr[2])
            dn = norm3(dv)
            force_deltas.append(dn)
            rn = norm3(vr)
            if rn > 0.0:
                force_props.append(dn / rn)

    print(f"Reference: {args.reference}")
    print(f"Corrected: {args.corrected}")
    print(f"Frames compared: {len(ref)}")
    print(f"Atomic force vectors compared: {len(force_deltas)}")
    print()
    print("Energy effect (corrected - reference):")
    print(f"  mean delta: {sum(energy_deltas) / len(energy_deltas):.6f} eV/frame")
    print(f"  min delta: {min(energy_deltas):.6f} eV/frame")
    print(f"  max delta: {max(energy_deltas):.6f} eV/frame")
    print()
    print("Force effect (corrected - reference):")
    print(f"  mean |delta F|: {sum(force_deltas) / len(force_deltas):.6f} eV/A")
    print(f"  max |delta F|: {max(force_deltas):.6f} eV/A")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
