#!/bin/bash
#SBATCH --job-name=humanoid_QRTD3
#SBATCH --output=/home/hice1/agupta886/scratch/slurm_outputs/humanoid_QRTD3.out
#SBATCH --error=/home/hice1/agupta886/scratch/slurm_errors/humanoid_QRTD3.err
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node="H100"

module load python cuda
source ~/scratch/python-envs/drl_env/bin/activate
cd ~/scratch/DRL_Group2_DDPG_TD3

for ((i=0;i<5;i+=1))
do 
	python main.py \
	--policy "QRTD3" \
	--env "Humanoid-v5" \
	--seed $i
done