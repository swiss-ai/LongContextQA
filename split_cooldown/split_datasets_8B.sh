#!/bin/bash
#SBATCH --job-name=sample_buckets
#SBATCH --output=./logs/sample_buckets_%j.out
#SBATCH --error=./logs/sample_buckets_%j.err
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=450G
#SBATCH --time=10:00:00
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5

set -euo pipefail

mkdir -p "$SLURM_SUBMIT_DIR/logs"

TMP_REPO="/capstor/scratch/cscs/$USER/.tmp_pipeline_${SLURM_JOB_ID}"
trap 'rm -rf "$TMP_REPO"' EXIT

git clone --depth 1 \
    git@github.com:swiss-ai/Megatron-LM.git \
    "${TMP_REPO}/Megatron-LM"

MEGATRON_PATH="${TMP_REPO}/Megatron-LM"

SCRIPT_DIR="$SLURM_SUBMIT_DIR"

echo "Job $SLURM_JOB_ID started at $(date)"

# Build extensions once; both srun steps share the same MEGATRON_PATH
srun --environment="${SCRIPT_DIR}/../nemo.toml" bash -c "\
    cd '${MEGATRON_PATH}' && \
    python setup.py build_ext --inplace"

OUTPUT_DIR="${SLURM_SUBMIT_DIR}/cooldown_subsample"
echo "Output dir: $OUTPUT_DIR"

srun --environment="${SCRIPT_DIR}/../nemo.toml" bash -c "\
    export PYTHONPATH='${MEGATRON_PATH}' && \
    python -u '${SLURM_SUBMIT_DIR}/split_datasets.py' \
        --input-dir /iopsstor/scratch/cscs/ahuang/apertus_dataset/prepare-8b/text_stage_3 \
                    /capstor/store/cscs/swissai/infra01/audio-datasets/Apertus1p5_cooldown_tokenized \
                    /capstor/store/cscs/swissai/infra01/vision-datasets/Apertus1p5_cooldown_tokenized \
        --output-dir '${OUTPUT_DIR}' \
        --num-buckets 4 --seed 123 --workers 32"

OUTPUT_DIR="${SLURM_SUBMIT_DIR}/extra_vision_samples"
echo "Output dir: $OUTPUT_DIR"

srun --environment="${SCRIPT_DIR}/../nemo.toml" bash -c "\
    export PYTHONPATH='${MEGATRON_PATH}' && \
    python -u '${SLURM_SUBMIT_DIR}/split_datasets.py' \
        --input-dir /capstor/scratch/cscs/dtamayomela/added_long_context_vision \
        --output-dir '${OUTPUT_DIR}' \
        --num-buckets 4 --seed 123 --workers 32 --target-tokens 1_500_000_000"

echo "Job $SLURM_JOB_ID finished at $(date)"