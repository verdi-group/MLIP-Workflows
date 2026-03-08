# mlip-phonons

Compute phonons (Phonopy) using a variety of ML interatomic potentials (ASE calculators), DOS, optionally compute band structure with autofinding bandpath, for Defect supercells or prisitine unit cells, write results/plots, and emit Plumipy-compatible files for PL/vibronic calculations. The repo also contains scripts to compare ML vs DFT coupling modes. 

## Install

#TODO: WORK ON README
From the repo root:
```bash
python -m pip install -e .
```

Optional model backends (install what you need):
#TODO: TEST THESE

If these do not work, see env/ENVIRONMENTS.md

```bash
python -m pip install -e '.[mace]'
python -m pip install -e '.[mattersim]'
python -m pip install -e '.[matgl]'
python -m pip install -e '.[pet]'
python -m pip install -e '.[orb]'
```
#TODO: remove cuda condition.

Notes:
- Python requirement: `>=3.10` (see `pyproject.toml`).
- `src/mlip_phonons/main.py` currently requires CUDA (`torch.cuda.is_available()` must be true).

## Configure

Edit `config.yml`:
- `models`: model names (must match the keys supported by `src/mlip_phonons/get_calc.py`) and which structure/material they run on.
- `structures`: where structures live (`assets/structures/...`) plus phonon settings (supercell, displacement `delta`, DOS mesh, etc.).

see config.yml for how to structure your input data

Model weights/checkpoints are expected under `assets/models/` 
#TODO: possibly save the model files? either that or right a model file .md explaining the set up.

## Run Phonon Workflow

The main entry point is the console script (defaults read from `config.yml` under `mlip_phonons.defaults`):

```bash
mlip-phonons
mlip-phonons <model_name>
mlip-phonons --config config.yml
mlip-phonons <model_name> --structure <structure_key>
```

## Outputs

By default outputs are written under:

```text
results/<model_name>/<structure_key>/
  raw/
  plot/
```

Additionally, Plumipy-ready (a package for obtaining PL spectra) files are written under:

```text
results/<model_name>/<structure_key>/raw/Plumipy_Files/
```
#TODO: put these in separate projects. 
## DFT vs ML Coupling/Ranking 

`src/coupling_modes/phonon_coupling.py` compares a DFT `band.yaml` against one or more MLIP `band.yaml` files using:
- a GS→ES displacement vector (from `CONTCAR_GS`/`CONTCAR_ES`) for projection weights,
- mode matching via overlap + Hungarian assignment,
- weighted errors `E_freq`, `E_vec`, and combined `Score = E_freq + alpha * E_vec`,
- frequency-cluster window diagnostics.

Run:

```bash
python src/coupling_modes/phonon_coupling.py --alpha 0.5 --weight_kind S
```

Reports are printed and also saved to `resultsPhonCoupling/phonon_coupling_report_<i>.txt`.

## PL Plot Comparison 

`src/plumipy_run/exploratory_script.py` contains helpers to call `plumipy.calculate_spectrum` and generate comparison plots (partial Huang–Rhys, spectral function, PL, IPR). Requires plumipy package installed.




## 18/02 feedback 

Check with mace dispersion = True. see the performance between 

also check just calculating the d3 term with ase. 
Read up on the fine tuning strategies optimal for singularly NEB and Phonons. 

_focus on one or two families of models and try series of fine tuning strategies. 

have a look at connor's RDM for the data. 

in principle i should be able to leave the minimisation. 
 
check the output of the hungarian minimisation and that it is operating properly. 

next step is to clean up the repo and check that it is working for connors defect calculations. get it ready for actual industrial application. 

look at reproducing the structures with mlip. (shake n break/brake) 

We need to test this on a bunch of different defects and paths. 

fine tuning techniques for the top models. 

after this, molecular dynamics. _

















