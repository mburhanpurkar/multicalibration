import numpy as np
import tensorflow as tf
from tensorflow import keras
np.set_printoptions(suppress=True)


def get_data(r, d, C, n_train, n_valid, n_test, eps, test=False):
    """Define the ground truth"""
    n = n_train + n_valid + n_test
    w2_star = np.random.rand(r)
    W1_star = np.random.rand(r, d)
    M = w2_star @ W1_star
    inputs = np.random.rand(d, n) + 1 # add 1 to avoid mean 0
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


def split_sets(model, D1, C, B, r):
    """Split D1 into sets by Linjun's method and get the corresponding intervals"""
    model = keras.Model(inputs=model.input, outputs=model.get_layer('dense').output)
    quantiles_B = np.linspace(0, 1, B + 1)
    sets = [D1]
    quantiles_list = []
    set_ranges = np.empty((1, r, 2))
    set_ranges[0, :, 0] = -C
    set_ranges[0, :, 1] = C
    
    for j in range(r):
        new_sets = []
        quantiles_tmp = []
        new_ranges = np.tile(set_ranges, (B, 1, 1))
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
                    assert(k != B - 1, 'features[sample_id] =', features[sample_id])

            for k in range(B):
                new_ranges[k * B**j + i, j] = [quantiles[k], quantiles[k + 1]]
                new_sets.append((np.array(sets_tmp[k]), np.array(sets_tmp_y[k])))

        quantiles_list.append(quantiles_tmp)
        sets = new_sets
        set_ranges = new_ranges

    set_counts = [len(i[0]) for i in sets]
    return sets, set_counts, set_ranges, quantiles_list


def get_set(sample, r, model, quantiles_list, set_ranges=None):
    """Determine which set a sample is in"""
    model = keras.Model(inputs=model.input, outputs=model.get_layer('dense').output)
    features = model.predict(np.reshape(sample, (1, -1)))

    for i in range(len(set_ranges)):
        flag = True
        for j in range(len(features[0])):
            if not(features[0][j] >= set_ranges[i][j][0] and features[0][j] <= set_ranges[i][j][1]):
                flag = False
                break
        if flag:
            return i
    print("The given element", sample, "could not be contained within set_ranges")
    raise ValueError


def get_sets(samples, r, model, quantiles_list, set_ranges=None):
    """Determine which set a sample is in"""
    model = keras.Model(inputs=model.input, outputs=model.get_layer('dense').output)
    features = model.predict(samples)
    ret = []
    for k, feature in enumerate(features):
        for i in range(len(set_ranges)):
            flag = True
            for j in range(len(set_ranges[i])):  ###### TODO I BELIEVE THIS IS A BUG FEATURE -> SET_RANGES[I]
                if not(feature[j] >= set_ranges[i][j][0] and feature[j] <= set_ranges[i][j][1]):
                    flag = False
                    break
            if flag:
                ret.append(i)
                break
            if i == len(set_ranges) - 1:
                raise ValueError("Couldn't place", feature, "within a set. feature corresponds to", samples[k], " input")
    return ret


def compute_D2(D2, r, model, quantiles_list, set_ranges=None):
    x, y = D2
    D2_counts = np.zeros(B**r)
    D2_weights = np.zeros(B**r)
    s = get_sets(D2[0], r, model, quantiles_list, set_ranges)

    for i in range(len(x)):
        D2_counts[s[i]] += 1
        D2_weights[s[i]] += y[i]
    return D2_counts, D2_weights


def fhat(sample, r, model, quantiles_list, D2_weights, D2_counts, set_ranges):
    """Construct fhat from D2 samples"""
    set_num = get_set(sample, r, model, quantiles_list, set_ranges)
    weights = D2_weights[set_num]
    counts = D2_counts[set_num]
    return weights / counts


def get_model(d, r, eps):
    """Define the linear model: consider adding regularization as needed"""
    inputs = keras.Input(shape=(d,))
    dense = keras.layers.Dense(units=r, 
                               use_bias=False, 
                               kernel_initializer=tf.keras.initializers.RandomNormal(stddev=eps))(inputs)
    outputs = keras.layers.Dense(units=1, 
                                 use_bias=False, 
                                 kernel_initializer=tf.keras.initializers.RandomNormal(stddev=eps))(dense)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


class CustomCallback(keras.callbacks.Callback):
    """Define a custom callback to compute sets using model"""
    def __init__(self, D1, D2, C, B, r, x_test, y_test):
        self.D1 = D1
        self.D2 = D2
        self.C = C
        self.B = B
        self.r = r
        self.x_test = x_test
        self.y_test = y_test
        self.n_test = len(x_test)
        self.nn_mses = []
        self.fhat_mses = []

    def on_epoch_end(self, epoch, logs=None):
        # Call split sets
        sets, set_counts, set_ranges, quantiles_list = split_sets(self.model, self.D1, self.C, self.B, self.r)
        
        # Call compute D2
        D2_counts, D2_weights = compute_D2(self.D2, self.r, self.model, quantiles_list, set_ranges)
        
        # Check that nothing is 0
        for i in range(len(D2_counts)):
            if D2_counts[i] == 0:
                print("D2_counts was 0 at i =", i, "with set_ranges[i]:")
                print(set_ranges[i])
                raise ValueError
        
        # Compute test accuracy
        fhat_test = np.empty(self.n_test)
        for i in range(self.n_test):
            fhat_test[i] = fhat(self.x_test[i], self.r, self.model, quantiles_list, D2_weights, D2_counts, set_ranges)
        mse_nn = model.evaluate(x=self.x_test, y=self.y_test)
        mse_fhat = np.square(np.subtract(fhat_test, self.y_test)).mean()
        print("Epoch", epoch, "fhat MSE:", mse_fhat)
        self.nn_mses.append(mse_nn)
        self.fhat_mses.append(mse_fhat)
        
        # Save to some outfile
        if epoch % 10 == 0:
            np.savetxt('nn_mse.txt', np.array(self.nn_mses))
            np.savetxt('fhat_mse.txt', np.array(self.fhat_mses))
        
        
def train_model(model, D1, D2, x_valid, y_valid, C, B, r, x_test, y_test):
    """Train the model (with Adam to avoid gradient problems)"""
    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        return lr * tf.math.exp(-0.1)    
    
    n_epochs = 100
    batch_size = 100
    lr = 0.1
    cb1 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, 
                                          patience=10, verbose=0, baseline=None)
    cb2 = tf.keras.callbacks.LearningRateScheduler(scheduler)
    cb3 = CustomCallback(D1, D2, C, B, r, x_test, y_test)

    model.compile(optimizer=keras.optimizers.Adam(lr=lr, clipnorm=1),
                  loss=tf.keras.losses.MSE)

    history = model.fit(D1[0], D1[1], batch_size=batch_size,
                        validation_data=(x_valid, y_valid),
                        epochs=n_epochs, callbacks=[cb1, cb2, cb3])

    # Compute the test MSE
    out = model.evaluate(x=x_test, y=y_test)
    return out


# Parameters
C = 100
n_train, n_valid, n_test = 32000, 100, 100
eps = 0.01
# B = round(n_train**(1./r + 1))
r = 3
d = 3
B = 4
D1, D2, x_valid, y_valid, x_test, y_test = get_data(r, d, C, n_train, n_valid, n_test, test=True)
model = get_model(d, r, eps)
mse = train_model(model, D1, x_valid, y_valid)
sets, set_counts, set_ranges, quantiles_list = split_sets(model, (x_test, y_test), C, B, r)
D2_counts, D2_weights = compute_D2(D2, r, model, quantiles_list, set_ranges)

for i in range(len(D2_counts)):
    if D2_counts[i] == 0:
        print("D2_counts was 0 at i =", i, "with set_ranges[i]:")
        print(set_ranges[i])
        raise ValueError

# Check the accuracy
fhat_test = np.empty(n_test)
for i in range(n_test):
    fhat_test[i] = fhat(x_test[i], r, model, quantiles_list, D2_weights, D2_counts)
mse_nn = model.evaluate(x=x_test, y=y_test)
mse_fhat = np.square(np.subtract(fhat_test, y_test)).mean()
print("NN MSE", mse_nn)
print("FH MSE", mse_fhat)
    
# Check for multicalibration: store MC quantities as set_id-val in dict
set_ids = get_sets(x_test, r, model, quantiles_list, set_ranges)
y_sums = dict()
fhat_sums = dict()
fhat_vals = dict()
for i in range(n_test):
    fx = fhat(x_test[i], r, model, quantiles_list, D2_weights, D2_counts)
    label = str(set_ids[i]) + "-" + str(fx)
    if label not in fhat_vals:
        fhat_vals[label] = 1
        y_sums[label] = y_test[i]
        fhat_sums[label] = fx
    else:
        fhat_vals[label] = 1 + fhat_vals[label]
        y_sums[label] = y_test[i] + y_sums[label]
        fhat_sums[label] = fx + fhat_sums[label]
mc = {key: abs(y_sums[key]-fhat_vals[key]) / fhat_vals[key] for key in fhat_vals}
print(mc)


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


