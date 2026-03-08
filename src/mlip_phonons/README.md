# `mlip_phonons`

Phonon workflow and MLIP calculator plumbing.

## Install

Recommended: create a conda env from `env/*.yml` (examples):

```bash
conda env create -f env/mace_env.yml
conda activate mace_env
python -m pip install -e '.[mace,vdw]'
```

Other model backends:

```bash
python -m pip install -e '.[mattersim,vdw]'
python -m pip install -e '.[matgl,vdw]'
python -m pip install -e '.[pet,vdw]'
python -m pip install -e '.[orb,vdw]'
```

Notes:
- Python support is `>=3.10,<3.13` (see `pyproject.toml`).
- The `vdw` extra installs `dftd3` for optional D3 corrections in ASE.

## CLI Usage

Phonons (defaults read from `config.yml` under `mlip_phonons.defaults`):

```bash
mlip-phonons
mlip-phonons <model_name>
mlip-phonons <model_name> --structure <structure_key>
mlip-phonons --config path/to/config.yml
```

Flags:
- `--config`: path to `config.yml` (default `./config.yml`).
- `--structure`: override structure key (default is `mlip_phonons.defaults.structure` or `models.<model>.material`).

NEB (MLIP + ASE NEB):

```bash
mlip-neb <model_name> --poscar-i <POSCAR_i> --poscar-f <POSCAR_f>
mlip-neb <model_name> --overwrite
```

Run `mlip-phonons --help` / `mlip-neb --help` for full flags.

## Key Modules

- `src/mlip_phonons/get_calc.py`
  - Maps a `model_name` string to an ASE calculator.
  - Expects model weights under `assets/models/`.
  - Supports optional D3 via `include_vdw=True` (requires `dftd3`).
- `src/mlip_phonons/main.py`
  - CLI entrypoint for the phonon workflow (`mlip-phonons`).
  - Reads `config.yml` and runs the selected model/structure workflow.
- `src/mlip_phonons/phonons.py`
  - Phonopy/ASE glue: supercells, displacements, force extraction, DOS/band outputs.
- `src/mlip_phonons/relax.py`
  - Structure relaxation helper used by NEB/phonon workflows.

## Inputs / Outputs

Inputs:
- `config.yml`: models, structures, phonon settings, paths.
- `assets/structures/...`: structures referenced by `config.yml`.
- `assets/models/...`: model checkpoint files (see `get_calc.py` keys).

Outputs (typical):
- `results/<model_name>/<structure_key>/raw/`
- `results/<model_name>/<structure_key>/plot/`
