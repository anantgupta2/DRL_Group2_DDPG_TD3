#!/bin/bash
#SBATCH --job-name=queue_TD3_all
#SBATCH --output=/home/hice1/agupta886/scratch/slurm_outputs/queue_TD3_all.out
#SBATCH --error=/home/hice1/agupta886/scratch/slurm_errors/queue_TD3_all.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

cd ~/scratch/DRL_Group2_DDPG_TD3/scripts/single_runs
job_id_ant=$(sbatch run_ant.sh)
job_id_hopper=$(sbatch run_hopper.sh)
job_id_cheetah=$(sbatch run_cheetah.sh)
job_id_walker=$(sbatch run_walker.sh)
job_id_humanoid=$(sbatch run_humanoid.sh)
job_id_reacher=$(sbatch run_reacher.sh)
job_id_invpend=$(sbatch run_invpend.sh)
job_id_invdoublepend=$(sbatch run_invdoublepend.sh)

echo "Submitted jobs:"
echo "Ant: $job_id_ant"
echo "Hopper: $job_id_hopper"
echo "Cheetah: $job_id_cheetah"
echo "Walker: $job_id_walker"
echo "Humanoid: $job_id_humanoid"
echo "Reacher: $job_id_reacher"
echo "Inverse Pendulum: $job_id_invpend"
echo "Inverse Double Pendulum: $job_id_invdoublepend"