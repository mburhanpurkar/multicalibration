import os
import numpy as np
import tensorflow as tf


os.mkdir('data')
os.mkdir('data_preprocessed')


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train_old, y_test_old = y_train, y_test
y_train_old, y_test_old = y_train.astype(float) / 10.,  y_test.astype(float) / 10.

y_train, y_test = np.random.binomial(1, y_train_old), np.random.binomial(1, y_test_old)

y_train, y_test = tf.keras.utils.to_categorical(y_train, num_classes=2), tf.keras.utils.to_categorical(y_test, num_classes=2)

y_train_old_tmp = np.empty((len(y_train_old), 2))
y_test_old_tmp = np.empty((len(y_test_old), 2))
y_train_old_tmp[:, 0] = y_train_old[:, 0]
y_train_old_tmp[:, 1] = 1 - y_train_old[:, 0]
y_test_old_tmp[:, 0] = y_test_old[:, 0]
y_test_old_tmp[:, 1] = 1 - y_test_old[:, 0]
y_test_old = y_test_old_tmp
y_train_old = y_train_old_tmp


np.save('data/x_train', x_train)
np.save('data/x_test', x_test)
np.save('data/y_train', y_train)
np.save('data/y_test', y_test)
np.save('data/y_train_old', y_train_old)
np.save('data/y_test_old', y_test_old)


x_test = tf.keras.applications.resnet_v2.preprocess_input(x_test)
x_train = tf.keras.applications.resnet_v2.preprocess_input(x_train)


np.save('data_preprocessed/x_train', x_train)
np.save('data_preprocessed/x_test', x_test)
np.save('data_preprocessed/y_train', y_train)
np.save('data_preprocessed/y_test', y_test)
np.save('data_preprocessed/y_train_old', y_train_old)
np.save('data_preprocessed/y_test_old', y_test_old)
