#!/bin/bash
#SBATCH --job-name=cwe_multi
#SBATCH --output=logs/cwe_multi_%A_%a.out
#SBATCH --error=logs/cwe_multi_%A_%a.err
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=450G
#SBATCH --time=12:00:00
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --array=0-3

# Multi-turn CWE: combine each (hard_smaller, easy_larger) pair into  #
# a single output dataset.                                            #
#                                                                     #
# Pairs (one per array task):                                         #
#   0: 8k_16k/hard   + 16k_32k/easy   -> 32k_cwe                      #
#   1: 16k_32k/hard  + 32k_64k/easy   -> 64k_cwe                      #
#   2: 32k_64k/hard  + 64k_128k/easy  -> 128k_cwe                     #
#   3: 64k_128k/hard + 128k_256k/easy -> 256k_cwe                     #
#                                                                     #
# Submit with:                                                        #
#   sbatch submit_cwe_multi.sh                                        #
# (the job array fan-out is built in via #SBATCH --array=0-3).        #

set -euo pipefail
mkdir -p logs

ROOT=/capstor/scratch/cscs/dtamayomela/long_context/multimodal_composition_3

HARD_INPUTS=(
    "${ROOT}/mix_8k_16k_cwe/hard"
    "${ROOT}/mix_16k_32k_cwe/hard"
    "${ROOT}/mix_32k_64k_cwe/hard"
    "${ROOT}/mix_64k_128k_cwe/hard"
)
EASY_INPUTS=(
    "${ROOT}/mix_16k_32k_cwe/easy"
    "${ROOT}/mix_32k_64k_cwe/easy"
    "${ROOT}/mix_64k_128k_cwe/easy"
    "${ROOT}/mix_128k_256k_cwe/easy"
)
OUT_NAMES=(
    "32k_cwe"
    "64k_cwe"
    "128k_cwe"
    "256k_cwe"
)

# Per-task seeds so the four jobs don't draw correlated samples.
SEEDS=(
    101
    202
    303
    404
)

idx=${SLURM_ARRAY_TASK_ID:-0}

HARD_INPUT="${HARD_INPUTS[$idx]}"
EASY_INPUT="${EASY_INPUTS[$idx]}"
OUT_NAME="${OUT_NAMES[$idx]}"
SEED="${SEEDS[$idx]}"

OUTPUT_DIR="/capstor/scratch/cscs/dtamayomela/LongContextQA/CWE/data/${OUT_NAME}"

SCRIPT_DIR=/capstor/scratch/cscs/dtamayomela/LongContextQA/CWE
MEGATRON_PATH=/capstor/scratch/cscs/dtamayomela/megatron/pre-training/megatron_fixed

echo "[$(date)] === Task ${idx} : ${OUT_NAME} ==="
echo "  HARD input : ${HARD_INPUT}"
echo "  EASY input : ${EASY_INPUT}"
echo "  OUTPUT     : ${OUTPUT_DIR}"
echo "  SEED       : ${SEED}"
echo

srun --environment="${SLURM_SUBMIT_DIR}/../container/nemo.toml" bash -c "
    export PYTHONPATH=${MEGATRON_PATH}
    python -u '${SLURM_SUBMIT_DIR}/create_cwe.py' \
        --hard-input '${HARD_INPUT}' \
        --easy-input '${EASY_INPUT}' \
        --output-dir '${OUTPUT_DIR}' \
        --hard-questions 10 \
        --easy-questions 5 \
        --top-k 15 \
        --min-word-len 4 \
        --seed ${SEED}
"

echo
echo "[$(date)] === Task ${idx} done ==="