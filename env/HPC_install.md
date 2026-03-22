
### HPC installation

On HPC systems, do the installation on a **compute node**, not on the login node. Specifically, this is a CUDA-dependent workflow, so use a **GPU node with CUDA support**.

#### 1. Request an interactive compute node

Example Slurm request for a CUDA-capable GPU node:

```bash
salloc \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=4 \
  --mem=16G \
  --job-name=environment_setup \
  --time=01:00:00 \
  --gres=gpu:1 \
  --partition=gpu_cuda \
  --qos=gpu \
  --account=ACCOUNT_STR \
  srun --export=PATH,TERM,HOME,LANG --pty /bin/bash -l
```

Notes:

* Replace `ACCOUNT_STR` with your Slurm account string.
* The exact partition, QoS, and GPU resource syntax may vary between clusters.
* An A100 is a tad overkill, was just the first example that came to mind. 

---

#### 2. Choose an environment manager

`miniforge` is recommended on HPC. 

##### Option A: Miniforge / Conda

Check available modules:

```bash
module avail anaconda miniconda miniforge
```

Load Miniforge (the below bash is something that you will use daily for miniforge usage):

```bash
module load miniforge/25.3.0-3
source "$ROOTMINIFORGE/etc/profile.d/conda.sh"
```

Create a working directory in scratch, then clone the repository:

```bash
mkdir -p /scratch/user/$USER/mlip_phonons
cd /scratch/user/$USER/mlip_phonons

git clone https://github.com/rohanplatts/phonons-mlip.git

```

---

#### 3. Create the environment from the .yml in env/

Create and activate an environment:

```bash
cd phonons-mlip
conda env create -f env/mace_env.yml
conda activate mace_env
```

---

#### 4. Install the package

From the repository root:

```bash
python -m pip install -e .
```
Above we are installing the package into the active environment. 

If you plan to run NEB with `include_vdw: true`, prefer using `conda env create -f env/mace_env.yml` (or other env YAMLs) so `dftd3-python` and `simple-dftd3` come from conda-forge. (Pip does not have them)

---

#### 6. Daily environment activation on HPC

If you are using Miniforge / Conda, a typical daily setup, with d3 acceleration (via parallelising over cores):

```bash
module purge
module load miniforge/25.3.0-3
source "$ROOTMINIFORGE/etc/profile.d/conda.sh"
conda activate mlip_phonons

export LD_PRELOAD="${CONDA_PREFIX}/lib/libgomp.so.1" 

export OMP_NUM_THREADS= <NUMBER OF CORES>
export MKL_NUM_THREADS= <NUMBER OF CORES>
export OPENBLAS_NUM_THREADS= <NUMBER OF CORES>
```

---

#### 5. CHECKS

Check that the package imports:

```bash
python -c "import mlip_phonons; print('mlip_phonons import successful')"
```

Check that PyTorch can see the GPU:

```bash
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
PY
```

---
