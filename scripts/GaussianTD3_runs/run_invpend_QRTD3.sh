#!/bin/bash
#SBATCH --job-name=inverted_pendulum_GaussianTD3
#SBATCH --output=/home/hice1/usingh68/scratch/slurm_outputs/inverted_pendulum_GaussianTD3.out
#SBATCH --error=/home/hice1/usingh68/scratch/slurm_errors/inverted_pendulum_GaussianTD3.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node="V100"

module load python cuda
source ~/scratch/python_envs/DRL_env/bin/activate
cd ~/scratch/DRL_Group2_DDPG_TD3

for ((i=0;i<5;i+=1))
do 
	python main.py \
	--policy "GaussianTD3" \
	--env "InvertedPendulum-v5" \
	--seed $i \
	--start_timesteps 1000
done