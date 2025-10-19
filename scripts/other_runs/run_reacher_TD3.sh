#!/bin/bash
#SBATCH --job-name=reacher_TD3
#SBATCH --output=/home/hice1/agupta886/scratch/slurm_outputs/reacher_TD3.out
#SBATCH --error=/home/hice1/agupta886/scratch/slurm_errors/reacher_TD3.err
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
	--policy "TD3" \
	--env "Reacher-v5" \
	--seed $i \
	--start_timesteps 1000
done