from pathlib import Path
from ase import Atoms
from ase.optimize import BFGS, FIRE, LBFGS, MDMin

_relaxers: dict[str, type] = {
    "bfgs": BFGS,
    "fire": FIRE,
    "lbfgs": LBFGS,
    "mdmin": MDMin,
}

# I anticipated that several relaxers would have different formats for how 
# files are saved, however this does not appear to be the case. Thus, this function 
# is more or less redundant, since it essentially replaces: 
# opt = BFGS(atoms, trajectory = outdir); opt.run(fmax = fmax) with: 
# atoms = relax(atoms, fmax, outdir, filename, type)
# but i will keep this just in case later on more obscure relaxations need to be 
# accounted for.

# atoms = relax(atoms, fmax)

def relax(
    structure: Atoms,
    fmax: float = 0.01,
    outdir: Path | None = None,
    filename: str | None = None,
    *,
    type: str = "BFGS",
):
    """Relax an ASE Atoms structure with a selected optimizer.

    Args:
        structure (Atoms): ASE Atoms object to relax (modified in place).
        fmax (float): Maximum force criterion for convergence.
        outdir (Path | None): Directory to write a trajectory file into.
        filename (str | None): Trajectory filename to use when outdir is provided.
        type (str): Optimizer type (case-insensitive). Supported: bfgs, lbfgs, fire, mdmin.

    Returns:
        Atoms: Relaxed atoms object.
    """
    key = str(type).strip().lower()
    relaxer = _relaxers.get(key)
    if relaxer is None:
        supported = ", ".join(sorted(_relaxers.keys()))
        raise ValueError(f"Unknown relax type: {type!r}. Supported: {supported}")

    trajectory = None
    saved_outdir = None
    if outdir is not None and filename is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        trajectory = outdir / str(filename)
        saved_outdir = outdir

    if trajectory is not None:
        opt = relaxer(structure, trajectory=str(trajectory))
    else:
        opt = relaxer(structure)

    opt.run(fmax=fmax)

    message = f"Relaxation complete ({key})."
    if saved_outdir is not None:
        message += f' "{filename}" was saved to {saved_outdir}'
    print(message)

    return structure
