#!/bin/bash
#SBATCH --job-name=queue_runs_GaussianTD3
#SBATCH --output=/home/hice1/usingh68/scratch/slurm_outputs/queue_runs_GaussianTD3.out
#SBATCH --error=/home/hice1/usingh68/scratch/slurm_errors/queue_runs_GaussianTD3.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

cd ~/scratch/DRL_Group2_DDPG_TD3/scripts/GaussianTD3_runs
#cheetah, hopper, humanoid, invdoublepend, reacher
job_id_cheetah_td3=$(sbatch run_cheetah_QRTD3.sh)
job_id_hopper_td3=$(sbatch run_hopper_QRTD3.sh)
job_id_humanoid_td3=$(sbatch run_humanoid_QRTD3.sh)
job_id_invdoublepend_td3=$(sbatch run_invdoublepend_QRTD3.sh)
job_id_reacher_td3=$(sbatch run_reacher_QRTD3.sh)
job_id_ant_td3=$(sbatch run_ant_QRTD3.sh)
job_id_walker_td3=$(sbatch run_walker_QRTD3.sh)
job_id_invpend_td3=$(sbatch run_invpend_QRTD3.sh)

echo "Submitted jobs:"
echo "Ant TD3: $job_id_ant_td3"
echo "Walker TD3: $job_id_walker_td3"
echo "Inverse Pendulum TD3: $job_id_invpend_td3"
echo "Cheetah TD3: $job_id_cheetah_td3"
echo "Hopper TD3: $job_id_hopper_td3"
echo "Humanoid TD3: $job_id_humanoid_td3"
echo "Inverse Double Pendulum TD3: $job_id_invdoublepend_td3"
echo "Reacher TD3: $job_id_reacher_td3"