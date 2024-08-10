#!/bin/bash 

 

#SBATCH --nodes=1  

#SBATCH --mem=20G 

#SBATCH --job-name=SAmasc1

#SBATCH --time=32:00:00 

#SBATCH --output=masc1.txt 

#SBATCH --error=masc1.err

#SBATCH --cpus-per-task=20 

#SBATCH --ntasks=1 

#SBATCH --partition=highmem
#SBATCH --mail-user=ww1a23@soton.ac.uk  

python SAC1P1F.py --env1 
