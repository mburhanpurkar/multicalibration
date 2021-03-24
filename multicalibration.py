import numpy as np
import tensorflow as tf
from tensorflow import keras

# Parameters
r = 2
d = 5
C = 100
n_train, n_valid, n_test = 1000, 100, 100
eps = 0.01
batch_size = 32
steps_per_epoch = n_train / batch_size
n_epochs = 5
B = round(n_train**(1./r + 1))

def get_data(r, d, C, n_train, n_valid, n_test, test=False):
    n = n_train + n_valid + n_test
    # Define the ground truth
    w2_star = np.random.rand(r)
    W1_star = np.random.rand(r, d)
    M = w2_star @ W1_star
    inputs = np.random.randn(d, n)
    if test:
        M = np.ones(d)
        assert(np.all(np.equal(M @ inputs, np.sum(inputs, axis=0))))
    y = M @ inputs + np.random.normal(scale=eps, size=(1, n))
    y = y[0]
    inputs = np.transpose(inputs)
    x_train, x_valid, x_test = inputs[:n_train], inputs[n_train:n_train + n_valid], inputs[n_train + n_valid:]
    y_train, y_valid, y_test = y[:n_train], y[n_train:n_train + n_valid], y[n_train + n_valid:]

    # First half of training data is D1, second is D2
    D1 = x_train[:n_train // 2], y_train[:n_train // 2]
    D2 = x_train[n_train // 2:], y_train[n_train // 2:]

    return D1, D2, x_valid, y_valid, x_test, y_test


"""
# Let's make sure the code above works! 
D1, D2, x_valid, y_valid, x_test, y_test = get_data(r, d, C, n_train, n_valid, n_test, test=True)
assert(len(x_valid) == len(y_valid) == n_valid)
assert(len(x_test) == len(y_test) == n_test)
assert(len(D1[0]) == len(D1[1]) == n_train // 2)
assert(len(D2[0]) == len(D2[1]) == n_train // 2)
assert(not(np.all(np.equal(D1[0], D2[0]))))
"""

# Define our estimator
def get_model(d, r, eps):
    inputs = keras.Input(shape=(d,))
    dense = keras.layers.Dense(units=r, 
                               use_bias=False, 
                               kernel_initializer=tf.keras.initializers.RandomNormal(stddev=eps))(inputs)
    outputs = keras.layers.Dense(units=1, 
                                 use_bias=False, 
                                 kernel_initializer=tf.keras.initializers.RandomNormal(stddev=eps))(dense)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def train_model(model, D1, x_valid, y_valid):
    # TODO: optimize params
    n_epochs = 3
    batch_size = 1
    lr = 0.1
    cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, 
                                          patience=10, verbose=0, baseline=None)

    model.compile(optimizer=keras.optimizers.Adam(lr=lr, clipnorm=1),
                  loss=tf.keras.losses.MSE)

    history = model.fit(D1[0], D1[1], batch_size=batch_size,
                        validation_data=(x_valid, y_valid),
                        epochs=n_epochs, callbacks=[cb])

    # Compute the test MSE
    out = model.evaluate(x=x_test, y=y_test)
    return out


"""
# Let's make sure the code above works! 
D1, D2, x_valid, y_valid, x_test, y_test = get_data(r, d, C, n_train, n_valid, n_test)
model = get_model(d, r, eps)
print(model.summary())
mse = train_model(model, D1, x_valid, y_valid)
"""
# Maybe make it the identity map. Then we expect it to partition in a 
# sensible way in x. Then make B = r = 2 so we can plot nicely
def get_sets(model, D1, C, B, r):
    model = keras.Model(inputs=model.input, outputs=model.get_layer('dense').output)
    xs = np.linspace(0, 1, B + 1)
    set_ranges = np.empty((1, r, 2))
    set_ranges[0, :, 0] = -C
    set_ranges[0, :, 1] = C
    sets = [D1]
    quantiles_list = []
    
    for j in range(r):
        new_ranges = np.tile(set_ranges, (B, 1, 1))
        new_sets = []
        quantiles_tmp = []

        for i in range(len(sets)):
            # Get jth neuron values in second last layer
            features = model.predict(sets[i][0])[j]
            quantiles = [-C] + list(np.quantile(features, xs[1:-1])) + [C]
            print(quantiles)
            quantiles_tmp.append(quantiles)

            for k in range(B):
                # Update ranges and sets
                print(i * B + k, j, k)
                new_ranges[i * B + k, j] = [quantiles[k], quantiles[k + 1]]
                mask = features > quantiles[k] and features < quantiles[k + 1]
                xs, ys = np.where(mask, sets[i][0]),  np.where(mask, sets[i][1])
                new_sets.append((xs, ys))

        quantlies_list.append(quantiles_tmp)
        sets = new_sets
        set_ranges = new_ranges

    # Get the number of elements in each set
    set_counts = [len(i[0]) for i in sets]

    return sets, set_ranges, set_counts, quantiles_list

# """
# Let's make sure the code above works! 
r = 10
d = 2
B = 2
D1, D2, x_valid, y_valid, x_test, y_test = get_data(r, d, C, n_train, n_valid, n_test)
model = get_model(d, r, eps)
mse = train_model(model, D1, x_valid, y_valid)
sets, set_ranges, set_counts, quantiles_list = get_sets(model, D1, C, B, r)
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

    

