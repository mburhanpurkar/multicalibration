import numpy as np
import matplotlib.pyplot as plt

# Load train and test data--outputs are the result of the second to last layer of the NN, labels are the 
# boolean labels and ground_truth are the probabilities from which the boolean labels were sampled
# model_name = "Zhun_FCN_0_data_hybrids_uniform_even" 
# model_name = "ResNet20v1_2_data_hybrids_uniform_even"
# data_dir = "/n/home10/mburhanpurkar/multicalibration/code/tensorflow/data_hybrids_uniform_even"
# network_dir = "/n/home10/mburhanpurkar/multicalibration/code/tensorflow/saved_models/" + model_name
# outputs = np.load(network_dir + "/hhat_train.npy")
# labels = np.load(data_dir + "/y_train.npy")[:, 0]
# ground_truth = np.load(data_dir + "/y_train_old.npy")[:, 0]
# probs = np.load(network_dir + '/pred_out_train.npy')[:, 0]

# test_outputs = np.load(network_dir + "/hhat_test.npy")
# test_labels = np.load(data_dir + "/y_test.npy")[:, 0]
# test_ground_truth = np.load(data_dir + "/y_test_old.npy")[:, 0]
# test_probs = np.load(network_dir + '/pred_out_test.npy')[:, 0]


outputs = np.load("colab_data/hhat_train_Zhun_half_relu.npy")
test_outputs = np.load("colab_data/hhat_test_Zhun_half_relu.npy")
probs = np.load("colab_data/pred_train_Zhun_half_relu.npy")[:, 1]
test_probs = np.load("colab_data/pred_test_Zhun_half_relu.npy")[:, 1]
labels = np.load("colab_data/y_train.npy")
test_labels = np.load("colab_data/y_test.npy")
ground_truth = np.load("colab_data/y_train_old.npy")
test_ground_truth = np.load("colab_data/y_test_old.npy")


N = len(outputs[0])
print("Running Linjun's method on", N, "neurons...")

B_max = 13
B_min = 1
train_mse = []
naive_train_mse = []
train_nn_mse = []
test_mse = []
naive_test_mse = []
test_nn_mse = []


for B in range(B_min, B_max):
    print(B)
    partitions = []
    partitioned_data = [[outputs]]
    for node in range(N):
        subset_partitions = []
        data_partitions = []
        for iB in range(max(int(B**(node-1)), 1)):
            for j, subset in enumerate(partitioned_data[iB]):
                args = np.argsort(subset[:, node])
                sorted_subsets = subset[args]

                new_subsets = [[] for k in range(B)]
                new_partitions = [[] for k in range(B)] 

                n_samples_tmp = len(args)

                for i in range(B):
                    new_subsets[i] = sorted_subsets[i * n_samples_tmp//B : (i + 1) * n_samples_tmp//B]
                    new_partitions[i] = new_subsets[i][-1, node]

                subset_partitions.append(new_partitions)
                data_partitions.append(new_subsets)

        partitions.append(subset_partitions)
        partitioned_data = data_partitions

    # Next, determine the label frequencies in each partition with the training data
    def search(target, array):
        for i, elt in enumerate(array):
            if target <= elt:
                return i
        return i

    label_counts = np.zeros(B**N)
    label_norms = np.zeros(B**N)

    for sample_idx, sample in enumerate(outputs):
        next_index = 0
        for i, x in enumerate(sample):
            prev_index = next_index
            index = search(x, partitions[i][next_index])
            next_index = prev_index * B + index
        label_counts[next_index] += labels[sample_idx]
        label_norms[next_index] += 1

    # (Mostly for fun) Get the linjun labels for the training data
    linjun_labels = []

    for sample_idx, sample in enumerate(outputs):
        next_index = 0
        for i, x in enumerate(sample):
            prev_index = next_index
            index = search(x, partitions[i][next_index])
            next_index = prev_index * B + index
        linjun_labels.append(label_counts[next_index] / label_norms[next_index])

    # Finally, apply to the test data (Linjun is the person who proposed this method)
    test_linjun_labels = []

    for sample_idx, sample in enumerate(test_outputs):
        next_index = 0
        for i, x in enumerate(sample):
            prev_index = next_index
            index = search(x, partitions[i][next_index])
            next_index = prev_index * B + index
        test_linjun_labels.append(label_counts[next_index] / label_norms[next_index])

    # Check how we did against some baselines--this is not good! We are hoping that the first number will be
    # lower than the next two!
    print("Train MSE:", np.average((linjun_labels - ground_truth)**2))
    print("Train Naive estimator MSE", np.average(([0.5]*len(linjun_labels) - ground_truth)**2))
    print("Train NN MSE", np.average((probs - ground_truth)**2))
    print()
    print("Test MSE:", np.average((test_linjun_labels - test_ground_truth)**2))
    print("Test Naive estimator MSE", np.average(([0.5]*len(test_linjun_labels) - test_ground_truth)**2))
    print("Test NN MSE", np.average((test_probs - test_ground_truth)**2))
    
    train_mse.append(np.average((linjun_labels - ground_truth)**2))
    naive_train_mse.append(np.average(([0.5]*len(linjun_labels) - ground_truth)**2))
    train_nn_mse.append(np.average((probs - ground_truth)**2))
    
    test_mse.append(np.average((test_linjun_labels - test_ground_truth)**2))
    naive_test_mse.append(np.average(([0.5]*len(test_linjun_labels) - test_ground_truth)**2))
    test_nn_mse.append(np.average((test_probs - test_ground_truth)**2))   
    
plt.plot(range(B_min, B_max), train_mse, 'o', label="Linjun (train)", alpha=0.5)
plt.plot(range(1, B_max), naive_train_mse, '-.', label="Naive (train)", alpha=0.5)
plt.plot(range(B_min, B_max), train_nn_mse, '--', label="NN (train)", alpha=0.5)
plt.plot(range(B_min, B_max), test_mse, 'o', label="Linjun (test)", alpha=0.5)
plt.plot(range(1, B_max), naive_test_mse, '-.', label="Naive (test)", alpha=0.5)
plt.plot(range(B_min, B_max), test_nn_mse, '--', label="NN (test)", alpha=0.5)
plt.xlabel("B")
plt.ylabel("MSE")
plt.legend()
plt.show()
