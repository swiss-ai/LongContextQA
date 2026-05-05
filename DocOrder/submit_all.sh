#!/bin/bash
# -----------------------------------------------------------------------------
# Submit four DocOrder jobs (one per pair). Run from the directory that
# contains create_DocOrder.py, create_DocOrder.sh and nemo.toml.
#
#   8k_16k_cwe/hard  +  16k_32k_cwe/easy   ->  32k_cwe
#  16k_32k_cwe/hard  +  32k_64k_cwe/easy   ->  64k_cwe
#  32k_64k_cwe/hard  +  64k_128k_cwe/easy  -> 128k_cwe
# 64k_128k_cwe/hard  + 128k_256k_cwe/easy  -> 256k_cwe
# -----------------------------------------------------------------------------

set -euo pipefail

ROOT=/capstor/scratch/cscs/dtamayomela/long_context/multimodal_composition_3

mkdir -p logs

submit() {
    local name="$1"
    local hard="$2"
    local easy="$3"
    echo "-> submitting ${name}"
    sbatch --job-name="doc_ord_${name}" create_DocOrder.sh \
        "${name}" \
        "${hard}" \
        "${easy}"
}

submit  32k_cwe  "${ROOT}/mix_8k_16k_cwe/hard"   "${ROOT}/mix_16k_32k_cwe/easy"
submit  64k_cwe  "${ROOT}/mix_16k_32k_cwe/hard"  "${ROOT}/mix_32k_64k_cwe/easy"
submit 128k_cwe  "${ROOT}/mix_32k_64k_cwe/hard"  "${ROOT}/mix_64k_128k_cwe/easy"
submit 256k_cwe  "${ROOT}/mix_64k_128k_cwe/hard" "${ROOT}/mix_128k_256k_cwe/easy"

echo "All four jobs submitted. Outputs go to:"
echo "  /capstor/scratch/cscs/dtamayomela/synthetic_data/DocOrder/long_context_combined/{32k,64k,128k,256k}_cwe"