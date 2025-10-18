#!/bin/bash
#SBATCH --job-name=TD3_experiments_all
#SBATCH --output=/home/hice1/agupta886/scratch/slurm_outputs/TD3_experiments_all.out
#SBATCH --error=/home/hice1/agupta886/scratch/slurm_errors/TD3_experiments_all.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node="H100"

module load python cuda
source ~/scratch/python-envs/drl_env/bin/activate
cd ~/scratch/DRL_Group2_DDPG_TD3

for ((i=0;i<10;i+=1))
do 
	python main.py \
	--policy "TD3" \
	--env "HalfCheetah-v5" \
	--seed $i

	python main.py \
	--policy "TD3" \
	--env "Hopper-v5" \
	--seed $i

	python main.py \
	--policy "TD3" \
	--env "Walker2d-v5" \
	--seed $i

	python main.py \
	--policy "TD3" \
	--env "Ant-v5" \
	--seed $i

	python main.py \
	--policy "TD3" \
	--env "Humanoid-v5" \
	--seed $i

	python main.py \
	--policy "TD3" \
	--env "InvertedPendulum-v5" \
	--seed $i \
	--start_timesteps 1000

	python main.py \
	--policy "TD3" \
	--env "InvertedDoublePendulum-v5" \
	--seed $i \
	--start_timesteps 1000

	python main.py \
	--policy "TD3" \
	--env "Reacher-v5" \
	--seed $i \
	--start_timesteps 1000
done
