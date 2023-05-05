#!/bin/bash
# Account name
#SBATCH --account=cse260
# Job name,  will be displayed on the showq command
#SBATCH --job-name=MMPY-CUDA
# Filename for standard output 
# At end of job, it is in directory from which sbatch was invoked
#SBATCH -o MMPY-CUDA.o%j
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -p CLUSTER
#  The requested wall clock job time limit in HH:MM:SS
#  Your job will end when it exceeds this time limit
#SBATCH -t 00:05:00 

##### CSE 260 - SPRING 2022 - You dont need this file. It is needed to run the code on a CLUSTER.

export OMP_NUM_THREADS=16
export KMP_AFFINITY="granularity=core,scatter"

module load cuda9


# Print out the environment
printenv


date

# Run
../mmpy -n 512 -x 32 -y 32 -r 10
echo ">>> Job Ends"

date


