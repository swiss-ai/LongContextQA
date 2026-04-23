#!/bin/bash
#SBATCH --job-name=doc_ord
#SBATCH --output=logs/doc_ord_%j.out
#SBATCH --error=logs/doc_ord_%j.err
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=450G
#SBATCH --time=10:00:00
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5

MEGATRON_PATH=/capstor/scratch/cscs/dtamayomela/megatron/pre-training/megatron_fixed
OUTPUT_DIR=/capstor/scratch/cscs/dtamayomela/synthetic_data/DocOrder/fpdfs-en_DocOrder_corr

srun --environment="${SLURM_SUBMIT_DIR}/nemo.toml" bash -c "\
    export PYTHONPATH=${MEGATRON_PATH}
    python /capstor/scratch/cscs/dtamayomela/synthetic_data/DocOrder/create_DocOrder.py \\
        --source finepdfs-edu-preprocessed-general \\
        --n-docs 100000 \\
        --output-dir ${OUTPUT_DIR} \\
        --min-sections 2 \\
        --max-section 5 \\
        --seed 42"