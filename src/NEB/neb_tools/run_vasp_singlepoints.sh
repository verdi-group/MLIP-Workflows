#!/usr/bin/env bash
#SBATCH -J vasp_sp
#SBATCH --ntasks=48
#SBATCH --ntasks-per-core=1
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --account=a_smp
#SBATCH --mem-per-cpu=1G
#SBATCH --time=12:00:00
#SBATCH --constraint=epyc3
#SBATCH --batch=epyc3

set -euo pipefail
ulimit -s unlimited

module purge
module load intel/2022a
export LD_LIBRARY_PATH=/scratch/project/cmp_scratch/hdf5-1.12.2-i22/hdf5/lib/:$LD_LIBRARY_PATH
export VASP=/scratch/project/cmp_scratch/vasp.6.4.3/vasp_std

cd /scratch/user/s4802880/mlip_phonons/vasp_pt/vasp_ci

for d in [0-9][0-9]/; do
  echo "=== Running $d ==="
  (
    cd "$d"
    srun --exclusive -n 48 "$VASP" > vasp.out 2> vasp.err
  )
done
