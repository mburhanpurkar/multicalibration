#!/bin/bash

#SBATCH -J data_gen
#SBATCH -n 1                # Number of cores
#SBATCH -t 0-00:05           # Runtime in D-HH:MM
#SBATCH -p shared           # Partition to submit to
#SBATCH --mem=5G
#SBATCH --account=dwork_lab
#SBATCH --output=data_gen
#SBATCH --mail-user=maya.burhanpurkar@gmail.com
#SBATCH --mail-type=ALL

module purge
module load Anaconda3/2019.10
source activate tf_mult


# Model parameter
# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  | 3 (2)| 92.16     | 91.25     | --c---     | -----     | 35 (---)
# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------

python generate_data.py

conda deactivate
