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

TMP_REPO="/capstor/scratch/cscs/$USER/.tmp_pipeline_${SLURM_JOB_ID}"
trap 'rm -rf "$TMP_REPO"' EXIT

git clone --depth 1 \
    git@github.com:swiss-ai/Megatron-LM.git \
    "${TMP_REPO}/Megatron-LM"

MEGATRON_PATH="${TMP_REPO}/Megatron-LM"

OUTPUT_BASE=/capstor/scratch/cscs/dtamayomela/LongContextQA/DocOrder/output_data
OUTPUT_DIR="${OUTPUT_BASE}/${NAME}"

mkdir -p "${OUTPUT_DIR}" logs

echo "=========================================================="
echo "  pair name : ${NAME}"
echo "  hard root : ${HARD_ROOT}"
echo "  easy root : ${EASY_ROOT}"
echo "  output    : ${OUTPUT_DIR}"
echo "=========================================================="

srun --environment="${SLURM_SUBMIT_DIR}/../nemo.toml" bash -c "\
    cd '${MEGATRON_PATH}' && \
    python setup.py build_ext --inplace && \
    export PYTHONPATH='${MEGATRON_PATH}' && \
    python -u '${SLURM_SUBMIT_DIR}/create_DocOrder.py' \
        --hard-root  '${HARD_ROOT}' \
        --easy-root  '${EASY_ROOT}' \
        --output-dir '${OUTPUT_DIR}' \
        --hard-sections 8 \
        --easy-sections 4 \
        --hard-tokens 1000000000 \
        --easy-tokens 1000000000 \
        --seed 42"