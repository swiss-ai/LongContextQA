#!/bin/bash
#SBATCH --job-name=sample_buckets
#SBATCH --output=logs/sample_buckets_%j.out
#SBATCH --error=logs/sample_buckets_%j.err
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=450G
#SBATCH --time=10:00:00
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5

set -euo pipefail
mkdir -p logs
 
SCRIPT_DIR="$SLURM_SUBMIT_DIR"
OUTPUT_DIR="${SLURM_SUBMIT_DIR}/multimodal_composition"
MEGATRON_PATH=/capstor/scratch/cscs/dtamayomela/megatron/pre-training/megatron_fixed

echo "Job $SLURM_JOB_ID started at $(date)"
echo "Output dir: $OUTPUT_DIR"

srun --environment="${SCRIPT_DIR}/nemo.toml" bash -c "\
    export PYTHONPATH=${MEGATRON_PATH}
    python -u ${SCRIPT_DIR}/multimodal_composition.py \
        --output-dir $OUTPUT_DIR \
        --seed 42"
 
echo "Job finished at $(date)"
 