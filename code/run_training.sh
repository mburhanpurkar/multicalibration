#!/bin/bash

#SBATCH -J test
#SBATCH -n 1                # Number of cores
#SBATCH -t 0-00:02          # Runtime in D-HH:MM
#SBATCH -p test             # Partition to submit to
#SBATCH --mem=400
#SBATCH --output=test
#SBATCH --mail-user=maya.burhanpurkar@gmail.com
#SBATCH --mail-type=ALL

module purge
module load Anaconda3/5.0.1-fasrc01
source activate tf

python multicalibration.py

source deactivate
