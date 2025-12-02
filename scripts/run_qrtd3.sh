#!/bin/bash
#SBATCH --job-name=queue_runs_qrtd3
#SBATCH --output=/home/hice1/usingh68/scratch/slurm_outputs/queue_runs_qrtd3.out
#SBATCH --error=/home/hice1/usingh68/scratch/slurm_errors/queue_runs_qrd3.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

cd ~/scratch/DRL_Group2_DDPG_TD3/scripts/QRTD3_runs
#cheetah, hopper, humanoid, invdoublepend, reacher

job_id_hopper_td3=$(sbatch run_hopper_QRTD3_2.sh)
job_id_hopper_td3=$(sbatch run_hopper_QRTD3_5.sh)
job_id_hopper_td3=$(sbatch run_hopper_QRTD3_10.sh)
job_id_invdoublepend_td3=$(sbatch run_invdoublepend_QRTD3_2.sh)
job_id_invdoublepend_td3=$(sbatch run_invdoublepend_QRTD3_5.sh)
job_id_invdoublepend_td3=$(sbatch run_invdoublepend_QRTD3_10.sh)
job_id_walker_td3=$(sbatch run_walker_QRTD3_2.sh)
job_id_walker_td3=$(sbatch run_walker_QRTD3_5.sh)
job_id_walker_td3=$(sbatch run_walker_QRTD3_10.sh)

echo "Submitted jobs:"
echo "Walker TD3 2: $job_id_walker_td3_2"
echo "Walker TD3 5: $job_id_walker_td3_5"
echo "Walker TD3 10: $job_id_walker_td3_10"
echo "Inverse Pendulum TD3 2: $job_id_invpend_td3_2"
echo "Inverse Pendulum TD3 5: $job_id_invpend_td3_5"
echo "Inverse Pendulum TD3 10: $job_id_invpend_td3_10"
echo "Hopper TD3 2: $job_id_hopper_td3_2"
echo "Hopper TD3 5: $job_id_hopper_td3_5"
echo "Hopper TD3 10: $job_id_hopper_td3_10"
