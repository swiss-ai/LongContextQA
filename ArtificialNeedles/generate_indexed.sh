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

OUTPUT_DIR=/capstor/scratch/cscs/dtamayomela/LongContextQA/ArtificialNeedles/output_data/32k
MEGATRON_PATH=/capstor/scratch/cscs/dtamayomela/megatron/pre-training/megatron_fixed

srun --environment="${SLURM_SUBMIT_DIR}/../container/nemo.toml" bash -c "\
    export PYTHONPATH=${MEGATRON_PATH}
    python -u '${SLURM_SUBMIT_DIR}/generate_indexed.py' --seed 0 --num-train 55000 --out-dir ${OUTPUT_DIR} --min-token 4096 --max-token 32768"

OUTPUT_DIR=/capstor/scratch/cscs/dtamayomela/LongContextQA/ArtificialNeedles/output_data/64k
srun --environment="${SLURM_SUBMIT_DIR}/../container/nemo.toml" bash -c "\
    export PYTHONPATH=${MEGATRON_PATH}
    python -u '${SLURM_SUBMIT_DIR}/generate_indexed.py' --seed 0 --num-train 25000 --out-dir ${OUTPUT_DIR} --min-token 16384 --max-token 65536"

OUTPUT_DIR=/capstor/scratch/cscs/dtamayomela/LongContextQA/ArtificialNeedles/output_data/128k
srun --environment="${SLURM_SUBMIT_DIR}/../container/nemo.toml" bash -c "\
    export PYTHONPATH=${MEGATRON_PATH}
    python -u '${SLURM_SUBMIT_DIR}/generate_indexed.py' --seed 0 --num-train 12200 --out-dir ${OUTPUT_DIR} --min-token 32768 --max-token 131072"

OUTPUT_DIR=/capstor/scratch/cscs/dtamayomela/LongContextQA/ArtificialNeedles/output_data/256k
srun --environment="${SLURM_SUBMIT_DIR}/../container/nemo.toml" bash -c "\
    export PYTHONPATH=${MEGATRON_PATH}
    python -u '${SLURM_SUBMIT_DIR}/generate_indexed.py' --seed 0 --num-train 6100 --out-dir ${OUTPUT_DIR} --min-token 65536 --max-token 262144"