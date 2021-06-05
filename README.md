# multicalibration

### `code/resnet`
- `generate_data.py` and `generate_data.sh`: data generation script and SLURM file for all CIFAR10 tests we have run so far
- `parse_file.py`: a script for parsing log files in case Callback logging fails
- `resnet.py`: has most of the base resnet code
- `train_resnet.py`: training and fine tuning fhat and w_new + lots of logging and plotting (remove plotting for reduced memory)
- `resume_resnet.py`: logic for resuming training of a model that failed midway (in either the training or fine tuning phases)
- `run_training.sh`: sample SLURM script
- `sanity_check_resnet.py`: deprecated code for integer probability resnet sanity check

### Linjun's Method
- `linear_model_r=1_multicalibration.py`: simple linear model with hand-implemented SGD for r = 1
- `deprecated_nn_multicalibration.py`: should be fixed
- `Debugging the Code.ipynb`: sanity checks on Linjun's method (locally)
- `Linear Model Experiments.ipynb`: sanity checks on Linjun's method (cluster)

### Notebooks
##### Deprecated MNIST and Logistic Regression Experiments
- `Logistic Regression Comparison.ipynb`: comparison of logistic regression distributions, accuracies
- `MNIST Calibration Test.ipynb`: original MNIST experiments with LeNet-5 (logistic regression dataset)
- `MNIST Calibration Test (Larger Models).ipynb`: original MNIST experiments with ResNet (logistic regression dataset)
- `Data Preparation for MNIST ResNet Test.ipynb`: hyperparameter tuning for the new experiment

##### Data Preparation
- `Hybrid Data Generation.ipynb `: testing the data generation script to show that it works
