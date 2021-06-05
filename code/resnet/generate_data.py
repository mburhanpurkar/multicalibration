import os
import numpy as np
import tensorflow as tf

# NB: the p -> 1-p error has been fixed in the code below. This is NOT
# the original code that was used for the end of May experiments.
# Data generation script for the hybrid tests can be found in the data
# generation notebook.

# Make data for 1/x^2 testing
os.mkdir('data_squared')

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train_old, y_test_old = y_train, y_test
y_train_old, y_test_old = 1. / y_train.astype(float)**2,  1. / y_test.astype(float)**2

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

