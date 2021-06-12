#!/bin/bash

#SBATCH -J even_uniform_tuning
#SBATCH -n 2                # Number of cores
#SBATCH -t 1-00:00           # Runtime in D-HH:MM
#SBATCH -p shared           # Partition to submit to
#SBATCH --mem=15G
#SBATCH --account=dwork_lab
#SBATCH --output=test_run
#SBATCH --mail-user=maya.burhanpurkar@gmail.com
#SBATCH --mail-type=ALL

# 8 hours, 10G, n=2

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


# python resume_resnet.py --version=1 --n=3 --id=0 --data=0 --epochs=76 --tuning_epoch=30
# python resume_resnet.py --version=1 --n=3 --id=${SLURM_ARRAY_TASK_ID} --data=0 --epochs=76 --tuning_epoch=30
# python sanity_check_resnet.py --version=1 --n=3 --id=0 --data=0 --epochs=75
# python resume_resnet.py --version=1 --n=3 --id=0 --data=0 --epochs=75 --tuning_epoch=0

# python train_resnet.py --version=1 --n=3 --id=0 --data=data_hybrids_fixed_even --epochs=200
python resume_resnet.py --version=1 --n=3 --id=0 --data=data_hybrids_uniform_even --epochs=100

conda deactivate
