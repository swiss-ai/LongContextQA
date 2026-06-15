#!/bin/bash
#SBATCH --job-name=70B_data
#SBATCH --output=logs/sample_buckets_%j.out
#SBATCH --error=logs/sample_buckets_%j.err
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=450G
#SBATCH --time=1:00:00
#SBATCH --account=infra01

SCRIPT_DIR="$SLURM_SUBMIT_DIR"
OUTPUT_DIR="${SLURM_SUBMIT_DIR}/multimodal_composition_70B"

TMP_REPO="/capstor/scratch/cscs/$USER/.tmp_pipeline_${SLURM_JOB_ID}"
trap 'rm -rf "$TMP_REPO"' EXIT

git clone --depth 1 \
    git@github.com:swiss-ai/Megatron-LM.git \
    "${TMP_REPO}/Megatron-LM"

MEGATRON_PATH="${TMP_REPO}/Megatron-LM"

echo "Job $SLURM_JOB_ID started at $(date)"
echo "Output dir: $OUTPUT_DIR"

srun --environment="${SCRIPT_DIR}/../nemo.toml" bash -c "\
    cd '${MEGATRON_PATH}' && \
    python setup.py build_ext --inplace && \
    export PYTHONPATH='${MEGATRON_PATH}' && \
    python -u '${SCRIPT_DIR}/multimodal_composition_70B.py' \
        --output-dir '${OUTPUT_DIR}' \
        --seed 42"

echo "Job finished at $(date)"