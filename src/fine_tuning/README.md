# Fine-Tuning (MACE) From VASP NEB OUTCAR

This folder contains a minimal pipeline to fine-tune a MACE foundation model on VASP NEB
image outputs (OUTCAR/OUTCAR.gz).

We train against DFT labels:
- `REF_energy` (eV, per-configuration)
- `REF_forces` (eV/Ang, per-atom)

Fine-tuning technique:
- Transfer learning from a pretrained MACE model via `--foundation_model`.
- Single-head fine-tuning (`--multiheads_finetuning False`).
- Atomic reference energies taken from the foundation model (`--E0s foundation`).

Objective / cost function:
- `WeightedEnergyForcesLoss` with `energy_weight` and `forces_weight` (NEB is force-dominant).



## 1) Convert VASP OUTCAR -> extxyz

```bash
python src/fine_tuning/file_conversion_script.py \
  --neb-root assets/training_data/CsPbI3/I_vac_0/output1 \
  --out-dir assets/training_data/CsPbI3/I_vac_0/processed_mace \
  --prefix ivac0_neb_stride5 \
  --stride 5
```

This writes:

- `assets/training_data/CsPbI3/I_vac_0/processed_mace/<prefix>_train.extxyz`
- `assets/training_data/CsPbI3/I_vac_0/processed_mace/<prefix>_val.extxyz`
- `assets/training_data/CsPbI3/I_vac_0/processed_mace/<prefix>_test.extxyz`

## 2) Train With MACE CLI

Run the saved command script:

```bash
./src/fine_tuning/train_ivac0_neb.sh
```

To validate arguments without training:

```bash
./src/fine_tuning/train_ivac0_neb.sh --dry_run
```

## Notes On Key Names (ASE Compatibility)

The converter writes `REF_energy` and `REF_forces` into the `extxyz` files to avoid
ASE >= 3.23 warnings about using `energy` / `forces` as key names.
