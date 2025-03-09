#!/bin/bash -l

##################
# slurm settings #
##################

# where to put stdout / stderr
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --job-name=Bindcraft_score
#SBATCH --time=01:00:00

#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40

##################################
# make bash behave more robustly #
##################################
set -e
set -u
set -o pipefail


###################
# set environment #
###################

module load python/3.12-conda
module load cuda/12.6.1

conda activate BindCraft

###############
# run command #
###############

python bindcraft_score.py --fasta_file "fastain.fasta" --pdb_folder "./alphafold_output" --target_pdb "egfr.pdb" 

###############
# end message #
###############
echo [$(date +"%Y-%m-%d %H:%M:%S")] finished on $(hostname)
