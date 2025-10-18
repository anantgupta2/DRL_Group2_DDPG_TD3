#!/bin/bash
#SBATCH --job-name=queue_runs_relevant
#SBATCH --output=/home/hice1/agupta886/scratch/slurm_outputs/queue_runs_relevant.out
#SBATCH --error=/home/hice1/agupta886/scratch/slurm_errors/queue_runs_relevant.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

cd ~/scratch/DRL_Group2_DDPG_TD3/scripts/relevant_runs
job_id_ant_td3=$(sbatch run_ant_TD3.sh)
job_id_ant_ddpg=$(sbatch run_ant_DDPG.sh)
job_id_walker_td3=$(sbatch run_walker_TD3.sh)
job_id_walker_ddpg=$(sbatch run_walker_DDPG.sh)
job_id_invpend_td3=$(sbatch run_invpend_TD3.sh)
job_id_invpend_ddpg=$(sbatch run_invpend_DDPG.sh)

echo "Submitted jobs:"
echo "Ant TD3: $job_id_ant_td3"
echo "Ant DDPG: $job_id_ant_ddpg"
echo "Walker TD3: $job_id_walker_td3"
echo "Walker DDPG: $job_id_walker_ddpg"
echo "Inverse Pendulum TD3: $job_id_invpend_td3"
echo "Inverse Pendulum DDPG: $job_id_invpend_ddpg"