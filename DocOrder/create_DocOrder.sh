#!/bin/bash
#SBATCH --job-name=doc_ord
#SBATCH --output=logs/doc_ord_%x_%j.out
#SBATCH --error=logs/doc_ord_%x_%j.err
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=450G
#SBATCH --time=10:00:00
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5

# -----------------------------------------------------------------------------
# Run create_DocOrder.py for ONE pair of (hard, easy) source roots.
#
# Usage:
#     sbatch --job-name=doc_ord_<NAME> run_DocOrder.sh \
#            <NAME> <HARD_ROOT> <EASY_ROOT>
#
# Example:
#     sbatch --job-name=doc_ord_32k_cwe run_DocOrder.sh \
#            32k_cwe \
#            /capstor/.../mix_8k_16k_cwe/hard \
#            /capstor/.../mix_16k_32k_cwe/easy
# -----------------------------------------------------------------------------

set -euo pipefail

NAME="${1:?need NAME (e.g. 32k_cwe)}"
HARD_ROOT="${2:?need HARD_ROOT}"
EASY_ROOT="${3:?need EASY_ROOT}"

MEGATRON_PATH=/capstor/scratch/cscs/dtamayomela/megatron/pre-training/megatron_fixed
OUTPUT_BASE=/capstor/scratch/cscs/dtamayomela/LongContextQA/DocOrder/long_context_combined
OUTPUT_DIR="${OUTPUT_BASE}/${NAME}"

mkdir -p "${OUTPUT_DIR}" logs

echo "=========================================================="
echo "  pair name : ${NAME}"
echo "  hard root : ${HARD_ROOT}"
echo "  easy root : ${EASY_ROOT}"
echo "  output    : ${OUTPUT_DIR}"
echo "=========================================================="

srun --environment="${SLURM_SUBMIT_DIR}/../nemo.toml" bash -c "\
    export PYTHONPATH=${MEGATRON_PATH}
    python ${SLURM_SUBMIT_DIR}/create_DocOrder.py \
        --hard-root '${HARD_ROOT}' \
        --easy-root '${EASY_ROOT}' \
        --output-dir '${OUTPUT_DIR}' \
        --hard-sections 8 \
        --easy-sections 4 \
        --hard-tokens 1000000000 \
        --easy-tokens 1000000000 \
        --seed 42"