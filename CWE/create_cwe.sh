#!/bin/bash
#SBATCH --job-name=cwe
#SBATCH --output=logs/cwe_%j.out
#SBATCH --error=logs/cwe_%j.err
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=450G
#SBATCH --time=3:00:00
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5

OUTPUT_DIR=/capstor/scratch/cscs/dtamayomela/synthetic_data/CWE/fpdfs_en_general_16k
SCRIPT_DIR=/capstor/scratch/cscs/dtamayomela/synthetic_data/CWE
MEGATRON_PATH=/capstor/scratch/cscs/dtamayomela/megatron/pre-training/megatron_fixed

srun --environment="${SLURM_SUBMIT_DIR}/nemo.toml" bash -c "\
    export PYTHONPATH=${MEGATRON_PATH}
    python -u "${SLURM_SUBMIT_DIR}/create_cwe.py" \
        --source finepdfs-edu-preprocessed-16k \
        --n-docs 200000 \
        --output-dir '$OUTPUT_DIR' \
        --seed 123"