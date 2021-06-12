import os
import numpy as np
import pandas as pd
import tensorflow as tf



# Make data for 1/x^2 testing
os.mkdir('data_squared')

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train_old, y_test_old = y_train, y_test
y_train_old, y_test_old = (1. / (1 + y_train.astype(float)))**2,  (1. / (1 + y_test.astype(float)))**2

y_train, y_test = np.random.binomial(1, y_train_old), np.random.binomial(1, y_test_old)
y_train, y_test = tf.keras.utils.to_categorical(y_train, num_classes=2), tf.keras.utils.to_categorical(y_test, num_classes=2)

y_train_old_tmp = np.empty((len(y_train_old), 2))
y_test_old_tmp = np.empty((len(y_test_old), 2))
y_train_old_tmp[:, 1] = y_train_old[:, 0]
y_train_old_tmp[:, 0] = 1 - y_train_old[:, 0]
y_test_old_tmp[:, 1] = y_test_old[:, 0]
y_test_old_tmp[:, 0] = 1 - y_test_old[:, 0]
y_test_old = y_test_old_tmp
y_train_old = y_train_old_tmp

np.save('data_squared/x_train', x_train)
np.save('data_squared/x_test', x_test)
np.save('data_squared/y_train', y_train)
np.save('data_squared/y_test', y_test)
np.save('data_squared/y_train_old', y_train_old)
np.save('data_squared/y_test_old', y_test_old)



# Make data for integer probability sanity check
os.mkdir('data_sanity_check')

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train, y_test = y_train > 4, y_test > 4
y_train, y_test = tf.keras.utils.to_categorical(y_train, num_classes=2), tf.keras.utils.to_categorical(y_test, num_classes=2)
np.save('data_sanity_check/x_train', x_train)
np.save('data_sanity_check/x_test', x_test)
np.save('data_sanity_check/y_train', y_train)
np.save('data_sanity_check/y_test', y_test)



# Make data for normal testing
os.mkdir('data')

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train_old, y_test_old = y_train, y_test
y_train_old, y_test_old = y_train.astype(float) / 10.,  y_test.astype(float) / 10.

y_train, y_test = np.random.binomial(1, y_train_old), np.random.binomial(1, y_test_old)
y_train, y_test = tf.keras.utils.to_categorical(y_train, num_classes=2), tf.keras.utils.to_categorical(y_test, num_classes=2)

y_train_old_tmp = np.empty((len(y_train_old), 2))
y_test_old_tmp = np.empty((len(y_test_old), 2))
y_train_old_tmp[:, 1] = y_train_old[:, 0]
y_train_old_tmp[:, 0] = 1 - y_train_old[:, 0]
y_test_old_tmp[:, 1] = y_test_old[:, 0]
y_test_old_tmp[:, 0] = 1 - y_test_old[:, 0]
y_test_old = y_test_old_tmp
y_train_old = y_train_old_tmp

np.save('data/x_train', x_train)
np.save('data/x_test', x_test)
np.save('data/y_train', y_train)
np.save('data/y_test', y_test)
np.save('data/y_train_old', y_train_old)
np.save('data/y_test_old', y_test_old)



# Make data for hybrid test
# lambda_type = "fixed" will select hybrid lambdas of 0.1, 0.2, 0.3, ..., 0.9
# lambda_type = "uniform" will select hybrid lambdas uniformly at random between 0 and 1
# samples = "even" will make 90% of images hybrids and 10% pure
# samples = "uneven" will make 50% of images hybrids and 50% pure
# In all cases, pure images will never be used to generate hybrid images

lambda_type = "fixed"
samples = "even"
dirname = "data_hybrids_" + lambda_type + "_" + samples
os.mkdir(dirname)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
test = pd.DataFrame(data=np.ravel(y_test), columns=['y'])
train = pd.DataFrame(data=np.ravel(y_train), columns=['y'])

if samples == "even":
    n_pure = 500
    n_hybrid = 4500
else:
    n_pure = 2500
    n_hybrid = 2500

images_arr = np.empty((n_hybrid * 10, 32, 32, 3))
labels_arr = np.empty(n_hybrid * 10)
images_arr_pure = np.empty((n_pure * 10, 32, 32, 3))
labels_arr_pure = np.empty(n_pure * 10)
original_labels = [None] * n_hybrid * 10
original_labels_pure = [None] * n_pure * 10
n_disp = 4

for i in range(10):
    indices_1 = train[train['y'] == i].index.to_numpy()
    indices_2 = train[train['y'] == (i + 1) % 10].index.to_numpy()
    
    # Get the pure ones
    images_arr_pure[i * n_pure : (i + 1) * n_pure] = x_train[indices_1[:n_pure]]
    labels_arr_pure[i * n_pure : (i + 1) * n_pure] = np.ones(n_pure) * i / 10.
    original_labels_pure[i * n_pure : (i + 1) * n_pure] = [str(i) + "_1"] * n_pure

    # Remove the first n
    indices_1 = indices_1[n_pure:]
    indices_2 = indices_2[n_pure:]

    # Randomly sample n of the indices 
    indices_1 = np.random.choice(indices_1, size=n_hybrid, replace=True)
    indices_2 = np.random.choice(indices_2, size=n_hybrid, replace=True)

    # Generate n lambdas
    if lambda_type == "fixed":
        hybrids = np.random.choice(np.linspace(0, 1, 11)[1:-1], size=n_hybrid, replace=True)
    else:
        hybrids = np.random.uniform(size=n_hybrid)
    lambdas = [[k] for k in hybrids]
    lambdas = np.tile(lambdas, (32 * 32 * 3)).reshape(n_hybrid, 32, 32, 3)

    # Compute the probabilities 
    probs = (lambdas * i + (1 - lambdas) * (i + 1) % 10) / 10.

    # Create the hybrids 
    images = x_train[indices_1] * lambdas + x_train[indices_2] * (1 - lambdas)

    # Append images and labels to array
    images_arr[i * n_hybrid : (i + 1) * n_hybrid] = images
    labels_arr[i * n_hybrid : (i + 1) * n_hybrid] = [probs[k, 0, 0, 0] for k in range(len(probs))]
    for k in range(n_hybrid):
        original_labels[i * n_hybrid + k] = str(i) + "_" + str(hybrids[k])
        
# Do sampling and converting to one-hot
samples_arr = np.random.binomial(1, labels_arr)
samples_arr = tf.keras.utils.to_categorical(samples_arr, num_classes=2)
categorical_labels_arr = np.empty((len(labels_arr), 2))
categorical_labels_arr[:, 1] = labels_arr
categorical_labels_arr[:, 0] = 1 - labels_arr

samples_arr_pure = np.random.binomial(1, labels_arr_pure)
samples_arr_pure = tf.keras.utils.to_categorical(samples_arr_pure, num_classes=2)
categorical_labels_arr_pure = np.empty((len(labels_arr_pure), 2))
categorical_labels_arr_pure[:, 1] = labels_arr_pure
categorical_labels_arr_pure[:, 0] = 1 - labels_arr_pure

# Concatenate
images_arr = np.concatenate((images_arr, images_arr_pure))
categorical_labels_arr = np.concatenate((categorical_labels_arr, categorical_labels_arr_pure))
samples_arr = np.concatenate((samples_arr, samples_arr_pure))
original_labels = np.concatenate((original_labels, original_labels_pure))

# Save
np.save(dirname + "/x_train.npy", images_arr)
np.save(dirname + "/y_train_old.npy", categorical_labels_arr)
np.save(dirname + "/y_train.npy", samples_arr)
np.save(dirname + "/train_original_labels.npy", original_labels)


if samples == "even":
    n_pure = 100
    n_hybrid = 900
else:
    n_pure = 500
    n_hybrid = 500

images_arr = np.empty((n_hybrid * 10, 32, 32, 3))
labels_arr = np.empty(n_hybrid * 10)
images_arr_pure = np.empty((n_pure * 10, 32, 32, 3))
labels_arr_pure = np.empty(n_pure * 10)
original_labels = [None] * n_hybrid * 10
original_labels_pure = [None] * n_pure * 10
n_disp = 4

for i in range(10):
    indices_1 = test[test['y'] == i].index.to_numpy()
    indices_2 = test[test['y'] == (i + 1) % 10].index.to_numpy()
    
    # Get the pure ones
    images_arr_pure[i * n_pure : (i + 1) * n_pure] = x_test[indices_1[:n_pure]]
    labels_arr_pure[i * n_pure : (i + 1) * n_pure] = np.ones(n_pure) * i / 10.
    original_labels_pure[i * n_pure : (i + 1) * n_pure] = [str(i) + "_1"] * n_pure

    # Remove the first n
    indices_1 = indices_1[n_pure:]
    indices_2 = indices_2[n_pure:]

    # Randomly sample n of the indices 
    indices_1 = np.random.choice(indices_1, size=n_hybrid, replace=True)
    indices_2 = np.random.choice(indices_2, size=n_hybrid, replace=True)

    # Generate n lambdas
    if lambda_type == "fixed":
        hybrids = np.random.choice(np.linspace(0, 1, 11)[1:-1], size=n_hybrid, replace=True)
    else:
        hybrids = np.random.uniform(size=n_hybrid)
    lambdas = [[k] for k in hybrids]
    lambdas = np.tile(lambdas, (32 * 32 * 3)).reshape(n_hybrid, 32, 32, 3)

    # Compute the probabilities 
    probs = (lambdas * i + (1 - lambdas) * (i + 1) % 10) / 10.

    # Create the hybrids 
    images = x_test[indices_1] * lambdas + x_test[indices_2] * (1 - lambdas)

    # Append images and labels to array
    images_arr[i * n_hybrid : (i + 1) * n_hybrid] = images
    labels_arr[i * n_hybrid : (i + 1) * n_hybrid] = [probs[k, 0, 0, 0] for k in range(len(probs))]
    for k in range(n_hybrid):
        original_labels[i * n_hybrid + k] = str(i) + "_" + str(hybrids[k])
        
# Do sampling and converting to one-hot
samples_arr = np.random.binomial(1, labels_arr)
samples_arr = tf.keras.utils.to_categorical(samples_arr, num_classes=2)
categorical_labels_arr = np.empty((len(labels_arr), 2))
categorical_labels_arr[:, 1] = labels_arr
categorical_labels_arr[:, 0] = 1 - labels_arr

samples_arr_pure = np.random.binomial(1, labels_arr_pure)
samples_arr_pure = tf.keras.utils.to_categorical(samples_arr_pure, num_classes=2)
categorical_labels_arr_pure = np.empty((len(labels_arr_pure), 2))
categorical_labels_arr_pure[:, 1] = labels_arr_pure
categorical_labels_arr_pure[:, 0] = 1 - labels_arr_pure

# Concatenate
images_arr = np.concatenate((images_arr, images_arr_pure))
categorical_labels_arr = np.concatenate((categorical_labels_arr, categorical_labels_arr_pure))
samples_arr = np.concatenate((samples_arr, samples_arr_pure))
original_labels = np.concatenate((original_labels, original_labels_pure))

# Save
np.save(dirname + "/x_test.npy", images_arr)
np.save(dirname + "/y_test_old.npy", categorical_labels_arr)
np.save(dirname + "/y_test.npy", samples_arr)
np.save(dirname + "/test_original_labels.npy", original_labels)
