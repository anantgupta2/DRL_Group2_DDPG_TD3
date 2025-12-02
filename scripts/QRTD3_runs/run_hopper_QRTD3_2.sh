#!/bin/bash
#SBATCH --job-name=hopper_QRTD3
#SBATCH --output=/home/hice1/usingh68/scratch/slurm_outputs/hopper_QRTD3_2.out
#SBATCH --error=/home/hice1/usingh68/scratch/slurm_errors/hopper_QRTD3_2.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node="V100"

module load python cuda
source ~/scratch/python_envs/DRL_env/bin/activate
cd ~/scratch/DRL_Group2_DDPG_TD3

for ((i=0;i<3;i+=1))
do 
	python main.py \
	--policy "QRTD3" \
	--env "Hopper-v5" \
	--seed $i \
	--K 2
done