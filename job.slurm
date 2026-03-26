#!/bin/bash
#SBATCH -J aime_benchmark
#SBATCH -o slurm_logs/aime_%A_%a.out        # %A = job ID, %a = array task ID
#SBATCH -e slurm_logs/aime_%A_%a.err
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --cpus-per-task=5
#SBATCH --mem=100G
#SBATCH -t 24:00:00
#SBATCH --partition=gupta,desa,gpu        
#SBATCH --gres=gpu:1   
#SBATCH --constraint="a6000|6000ada|a100"
#SBATCH --array=0-5                        # seeds: 0 to 9
#SBATCH --requeue 

echo "Job $SLURM_JOB_ID, task $SLURM_ARRAY_TASK_ID running on $(hostname)"

SEED=$SLURM_ARRAY_TASK_ID
echo "Running seed: $SEED"

# (optional but useful for debugging)
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# run your benchmark
bash aime_benchmark.sh $SEED