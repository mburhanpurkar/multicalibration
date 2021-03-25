import numpy as np
import tensorflow as tf
from tensorflow import keras


# Parameters
r = 2
d = 5
C = 100
n_train, n_valid, n_test = 50, 100, 100
eps = 0.01
batch_size = 32
steps_per_epoch = n_train / batch_size
n_epochs = 5
B = round(n_train**(1./r + 1))


def get_data(r, d, C, n_train, n_valid, n_test, test=False):
    """Define the ground truth"""
    n = n_train + n_valid + n_test
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


def get_model(d, r, eps):
    """Define the linear model"""
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
    """Train the model (with Adam to avoid gradient problems)"""
    # TODO: optimize params!
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


def split_sets(model, D1, C, B, r):
    """Split D1 into sets by Linjun's method and get the corresponding intervals"""
    model = keras.Model(inputs=model.input, outputs=model.get_layer('dense').output)
    quantiles_B = np.linspace(0, 1, B + 1)
    sets = [D1]
    quantiles_list = []
    
    for j in range(r):
        new_sets = []
        quantiles_tmp = []
        #print("sets j =", j, " -> ", sets)
        for i in range(len(sets)):
            features = model.predict(sets[i][0])[:, j]
            quantiles = [-C] + list(np.quantile(features, quantiles_B[1:-1])) + [C]
            quantiles_tmp.append(quantiles)
            sets_tmp = [[] for i in range(B)]
            sets_tmp_y = [[] for i in range(B)]

            for sample_id in range(len(sets[i][0])):
                for k in range(B):
                    if features[sample_id] >= quantiles[k] and features[sample_id] <= quantiles[k + 1]:
                        sets_tmp[k].append(sets[i][0][sample_id])
                        sets_tmp_y[k].append(sets[i][1][sample_id])
                        break
                    assert(k != B - 1)

            for k in range(B):
                new_sets.append((np.array(sets_tmp[k]), np.array(sets_tmp_y[k])))

        quantiles_list.append(quantiles_tmp)
        sets = new_sets

    set_counts = [len(i[0]) for i in sets]
    return sets, set_counts, quantiles_list


def get_set(sample, r, model, quantiles_list):
    """Determine which set a sample is in"""
    model = keras.Model(inputs=model.input, outputs=model.get_layer('dense').output)
    features = model.predict(np.reshape(sample, (1, -1)))

    indices = []
    for j in range(r):
        #print(features[0][j], quantiles_list[j][0])
        for i in range(B):
            #print(features[0][j], quantiles_list[j][0][i], quantiles_list[j][0][i + 1])
            if features[0][j] >= quantiles_list[j][0][i] and features[0][j] <= quantiles_list[j][0][i + 1]:
                indices.append(i)
                break
        assert(len(indices) == j + 1)
    ret = 0
    for i in range(r):
        ret += B**(r - i - 1) * indices[i]
    return ret


def fhat(sample):
    """Construct fhat from D2 samples"""
    set_num = get_set(sample)
    counts = np.sum(sets[set_num][1])
    size = set_counts[set_num]
    return counts / size

# # Finally, evaluate the predictor on the test data

    


""" Run some basic checks...
# Test get_data()
D1, D2, x_valid, y_valid, x_test, y_test = get_data(r, d, C, n_train, n_valid, n_test, test=True)
assert(len(x_valid) == len(y_valid) == n_valid)
assert(len(x_test) == len(y_test) == n_test)
assert(len(D1[0]) == len(D1[1]) == n_train // 2)
assert(len(D2[0]) == len(D2[1]) == n_train // 2)
assert(not(np.all(np.equal(D1[0], D2[0]))))

# Test train_model() and get_model()
D1, D2, x_valid, y_valid, x_test, y_test = get_data(r, d, C, n_train, n_valid, n_test)
model = get_model(d, r, eps)
print(model.summary())
mse = train_model(model, D1, x_valid, y_valid)

# Test split_sets() and get_set()
r = 3
d = 3
B = 4
D1, D2, x_valid, y_valid, x_test, y_test = get_data(r, d, C, n_train, n_valid, n_test, test=True)
model = get_model(d, r, eps)
mse = train_model(model, D1, x_valid, y_valid)
sets, set_counts, quantiles_list = split_sets(model, D1, C, B, r)
model_top = keras.Model(inputs=model.input, outputs=model.get_layer('dense').output)

# Check that get_sets returns the set that the sample actually came from
for i in range(len(sets)):
    for j in range(len(sets[i][0])):
        x = sets[i][0][j]
        #print("x", np.shape(x), x)
        get = get_set(x, r, model, quantiles_list)
        if len(x) != 0 and get != i:
            #print("Bad:", get, i, model_top.predict(x.reshape((1, -1))), quantiles_list)
"""


