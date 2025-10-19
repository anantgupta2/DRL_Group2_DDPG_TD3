#!/bin/bash
#SBATCH --job-name=queue_runs_other
#SBATCH --output=/home/hice1/agupta886/scratch/slurm_outputs/queue_runs_other.out
#SBATCH --error=/home/hice1/agupta886/scratch/slurm_errors/queue_runs_other.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

cd ~/scratch/DRL_Group2_DDPG_TD3/scripts/other_runs
#cheetah, hopper, humanoid, invdoublepend, reacher
job_id_cheetah_td3=$(sbatch run_cheetah_TD3.sh)
job_id_cheetah_ddpg=$(sbatch run_cheetah_DDPG.sh)
job_id_hopper_td3=$(sbatch run_hopper_TD3.sh)
job_id_hopper_ddpg=$(sbatch run_hopper_DDPG.sh)
job_id_humanoid_td3=$(sbatch run_humanoid_TD3.sh)
job_id_humanoid_ddpg=$(sbatch run_humanoid_DDPG.sh)
job_id_invdoublepend_td3=$(sbatch run_invdoublepend_TD3.sh)
job_id_invdoublepend_ddpg=$(sbatch run_invdoublepend_DDPG.sh)
job_id_reacher_td3=$(sbatch run_reacher_TD3.sh)
job_id_reacher_ddpg=$(sbatch run_reacher_DDPG.sh)

echo "Submitted jobs:"
echo "Cheetah TD3: $job_id_cheetah_td3"
echo "Cheetah DDPG: $job_id_cheetah_ddpg"
echo "Hopper TD3: $job_id_hopper_td3"
echo "Hopper DDPG: $job_id_hopper_ddpg"
echo "Humanoid TD3: $job_id_humanoid_td3"
echo "Humanoid DDPG: $job_id_humanoid_ddpg"
echo "Inverse Double Pendulum TD3: $job_id_invdoublepend_td3"
echo "Inverse Double Pendulum DDPG: $job_id_invdoublepend_ddpg"
echo "Reacher TD3: $job_id_reacher_td3"
echo "Reacher DDPG: $job_id_reacher_ddpg"