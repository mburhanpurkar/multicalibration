import numpy as np
import tensorflow as tf
from tensorflow import keras

print("hi")
# Parameters
r = 2
d = 5
C = 100
n_train, n_valid, n_test = 1000, 100, 100
eps = 0.01
batch_size = 32
steps_per_epoch = n_train / batch_size
n_epochs = 5
lr = 0.1  # TODO: optimize
B = round(n_train**(1./r + 1))

def get_data(r, d, C, n_train, n_valid, n_test):
    n = n_train + n_valid + n_test
    print("GOt n")
    # Define the ground truth
    w2_star = np.random.rand(1, r)
    W1_star = np.random.rand(r, d)
    M = w2_star @ W1_star
    M = np.resize(M, (n,) + np.shape(M))
    inputs = np.random.randn(d)
    y = M @ inputs + np.random.normal(scale=eps, size=(n, 1))
    x_train, x_valid, x_test = inputs[:n_train], inputs[n_train:n_train + n_valid], inputs[n_train + n_valid:]
    y_train, y_valid, y_test = y[:n_train], y[n_train:n_train + n_valid], y[n_train + n_valid:]

    # First half of training data is D1, second is D2
    D1 = x_train[:n_train // 2], y_train[:n_train // 2]
    D2 = x_train[n_train // 2:], y_train[n_train // 2:]

    return D1, D2, x_valid, y_valid, x_test, y_test


# """
# Let's make sure the code above works! 
#"""

D1, D2, x_valid, y_valid, x_test, y_test = get_data(r, d, C, n_train, n_valid, n_test)

# # Define our estimator
# def get_model():
#     inputs = keras.Input(shape=(None, 1))
#     dense = keras.layers.Dense(units=r)(inputs)
#     outputs = keras.layers.Dense(units=1)(dense)
#     model = keras.Model(inputs=inputs, outputs=outputs)
#     return model

# model = get_model()
# print(model.summary())

# # Optimize and measure MSE
# # TODO: make a callback to train until eps error with val data
# model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr),
#               loss=tf.keras.losses.MeanSquaredError())
# history = model.fit(D1[0], D1[1], batch_size=batch_size,
#                     validation_data=(x_valid, y_valid),
#                     epochs=n_epochs)

# # Compute the test MSE
# loss, MSE = model.evaluate((x_test, y_test))
# print("Test MSE:", MSE)


# """
# Let's make sure the code above works! 
#"""



# # Apply Linjun's method to pick sets--get second to last layer
# quantiles = np.linspace(0, 1, B + 1)
# set_ranges = np.empty((1, r, 2))
# set_ranges[0, :, 0] = -C
# set_ranges[0, :, 1] = C
# sets = [D1]
# extractor = keras.Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])
# quantiles_list = []

# for j in range(r):
#     new_ranges = np.tile(set_ranges, (B, 1, 1))
#     new_sets = []
#     quantiles_tmp = []

#     for i in range(len(sets)):
#         # Get jth neuron values in second last layer
#         features = extractor(sets[i][0])
#         feature = features[-2, j]
#         quantiles = np.quantile(feature, quantiles[1:-1])
#         quantiles = [-C] + quantiles + [C]
#         quantiles_tmp.append(quantiles)

#         for k in range(B):
#             # Update ranges and sets
#             new_ranges[i * B + k, j] = [quantiles[k], quantiles[k + 1]]
#             mask = feature > quantiles[k] and feature < quantiles[k + 1]
#             xs, ys = np.where(mask, sets[i][0]),  np.where(mask, sets[i][1])
#             new_sets.append((xs, ys))

#     quantlies_list.append(quantiles_tmp)
#     sets = new_sets
#     set_ranges = new_ranges

# # Get the number of elements in each set
# set_counts = [len(i[0]) for i in sets]


# """
# Let's make sure the code above works! 
#"""



# # Use set_ranges to make decision tree
# def get_set(sample):
#     features = extractor(sample)
#     indices = []
#     for j in range(r):
#         for i in range(B):
#             if features[j] > quantiles_list[j][i] and features[j] < quantiles_list[j][i + 1]:
#                 indices.append[i]
#                 break
#     ret = 0
#     for i in range(r):
#         ret += B**(r - i - 1) * indices[r]
#     return ret



# """
# Let's make sure the code above works! 
#"""



# # Use get_set to construct f hat using the D2 values
# def fhat(sample):
#     set_num = get_set(sample)
#     counts = np.sum(sets[set_num][1])
#     size = set_counts[set_num]
#     return counts / size

# # Finally, evaluate the predictor on the test data

    

