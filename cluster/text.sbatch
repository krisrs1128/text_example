#!/bin/bash
#################
#set a job name
#SBATCH --job-name=text_example
#################
#a file for job output, you can check job progress
#SBATCH --account=rpp-bengioy
#SBATCH --output=text_example.out
#SBATCH --qos=high
#################
# a file for errors from the job
#SBATCH --error=text_example.err
#################
#time you think you need; default is one hour
#in minutes in this case, hh:mm:ss
#SBATCH --time=00:30:00
#################
#memory per node; default is 4000 MB per CPU
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sankaran.kris@gmail.com

singularity shell --nv --bind /scratch/sankarak/data:data,/home/sankarak/text_example:/home /scratch/sankarak/images/text.sif /home/cluster/main.sh