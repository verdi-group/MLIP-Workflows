# `coupling_modes`

Tools for comparing DFT vs ML phonon coupling / mode matching.

## Key Files

- `src/coupling_modes/phonon_coupling.py`
  - Compares a DFT `band.yaml` against one or more MLIP `band.yaml` files.
  - Uses mode overlap + Hungarian assignment to match modes, then reports frequency/vector errors.
  - Outputs: prints a report and writes a text report under `resultsPhonCoupling/`.

- `src/coupling_modes/PHONONCOUPLING.md`
  - Notes and background on the coupling/matching approach.

- `src/coupling_modes/coup_tools/`
  - Helper utilities used by `phonon_coupling.py`.

## Usage

Typical:

```bash
python src/coupling_modes/phonon_coupling.py --alpha 0.5 --weight_kind S
```

Run `--help` for required input paths and options (DFT/ML `band.yaml`, GS/ES structures, etc.).

