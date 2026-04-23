#!/bin/bash
#SBATCH --job-name=art_need
#SBATCH --output=logs/art_need_idx_%j.out
#SBATCH --error=logs/art_need_idx_%j.err
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=450G
#SBATCH --time=6:00:00
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5

OUTPUT_DIR=/capstor/scratch/cscs/dtamayomela/synthetic_data/ArtificialNeedles/simplified_repo/output_files_newint
MEGATRON_PATH=/capstor/scratch/cscs/dtamayomela/megatron/pre-training/megatron_fixed

srun --environment="${SLURM_SUBMIT_DIR}/nemo.toml" bash -c "\
    export PYTHONPATH=${MEGATRON_PATH}
    python -u '${SLURM_SUBMIT_DIR}/generate_indexed.py' --seed 0 --num-train 244140 --out-dir ${OUTPUT_DIR}"