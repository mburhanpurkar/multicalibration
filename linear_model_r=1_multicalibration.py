import numpy as np
import sys
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=sys.maxsize)


def get_data(r, d, C, n_train, n_valid, n_test, eps, test=False, no_noise=False):
    """Define the ground truth"""
    n = n_train + n_valid + n_test
    w2_star = np.random.rand(r)
    W1_star = np.random.rand(r, d)
    M = w2_star @ W1_star
    inputs = np.random.rand(d, n) + 1 # add 1 to avoid everything being mean 0
    if test:
        M = np.ones(d)
        assert(np.all(np.equal(M @ inputs, np.sum(inputs, axis=0))))
    if no_noise:
        y = (M @ inputs).reshape(1, n)
    else:
        y = M @ inputs + np.random.normal(scale=eps, size=(1, n))
    y = y[0]
    inputs = np.transpose(inputs)
    x_train, x_valid, x_test = inputs[:n_train], inputs[n_train:n_train + n_valid], inputs[n_train + n_valid:]
    y_train, y_valid, y_test = y[:n_train], y[n_train:n_train + n_valid], y[n_train + n_valid:]

    # First half of training data is D1, second is D2
    D1 = x_train[:n_train // 2], y_train[:n_train // 2]
    D2 = x_train[n_train // 2:], y_train[n_train // 2:]

    return D1, D2, x_valid, y_valid, x_test, y_test, W1_star, w2_star


def split_sets(w2, W1, D1, C, B, r):
    """Split D1 into sets by Linjun's method and get the corresponding intervals"""
    model = W1
    quantiles_B = np.linspace(0, 1, B + 1)
    sets = [D1]
    quantiles_list = []
    set_ranges = np.empty((1, r, 2))
    set_ranges[0, :, 0] = -C
    set_ranges[0, :, 1] = C
    
    def predict(xs, j):
        n = len(xs)
        out = np.empty((n, len(xs[0])))
        for i in range(len(xs)):
            out[i] = W1 @ xs[i]
        return out[:, j]
    
    for j in range(r):
        new_sets = []
        quantiles_tmp = []
        new_ranges = np.tile(set_ranges, (B, 1, 1))
        for i in range(len(sets)):
            features = predict(sets[i][0], j)
            quantiles = [-C] + list(np.quantile(features, quantiles_B[1:-1])) + [C]
            quantiles_tmp.append(quantiles)
            sets_tmp = [[] for i in range(B)]
            sets_tmp_y = [[] for i in range(B)]
            # print(features)
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


def get_set(sample, r, w2, W1, quantiles_list, set_ranges=None):
    """Determine which set a sample is in"""
    features = W1 @ sample

    for i in range(len(set_ranges)):
        flag = True
        for j in range(len(features)):
            if not(features[j] >= set_ranges[i][j][0] and features[j] <= set_ranges[i][j][1]):
                flag = False
                break
        if flag:
            return i
    print("The given element", sample, "could not be contained within set_ranges")
    raise ValueError


def get_sets(samples, r, w2, W1, quantiles_list, set_ranges=None):
    """Determine which set a sample is in"""
    def predict(xs):
        n = len(xs)
        out = np.empty((n, len(xs[0])))
        for i in range(len(xs)):
            out[i] = W1 @ xs[i]
        return out
    
    features = predict(samples)
    ret = []
    for k, feature in enumerate(features):
        for i in range(len(set_ranges)):
            flag = True
            for j in range(len(set_ranges[i])):
                if not(feature[j] >= set_ranges[i][j][0] and feature[j] <= set_ranges[i][j][1]):
                    flag = False
                    break
            if flag:
                ret.append(i)
                break
            if i == len(set_ranges) - 1:
                raise ValueError("Couldn't place", feature, "within a set. feature corresponds to", samples[k], " input")
    return ret


def compute_D2(D2, r, w1, W2, quantiles_list, set_ranges=None):
    x, y = D2
    D2_counts = np.zeros(B**r)
    D2_weights = np.zeros(B**r)
    s = get_sets(D2[0], r, w1, W2, quantiles_list, set_ranges)

    for i in range(len(x)):
        D2_counts[s[i]] += 1
        D2_weights[s[i]] += y[i]
    return D2_counts, D2_weights


def fhat(sample, r, w2, W1, quantiles_list, D2_weights, D2_counts, set_ranges):
    """Construct fhat from D2 samples"""
    set_num = get_set(sample, r, w2, W1, quantiles_list, set_ranges)
    weights = D2_weights[set_num]
    counts = D2_counts[set_num]
    return weights / counts


def predict(xs, w2, W1, only_top=True):
    n = len(xs)
    if only_top:
        out = np.empty((n, len(xs[0])))
        for i in range(n):
            out[i] = W1 @ xs[i]
        return out
    out = np.empty(n)
    for i in range(n):
        out[i] = w2 @ W1 @ xs[i]
    return out


def get_model(B, d, r, eps, lr, reg, n_epochs, x_train, y_train, 
              x_valid, y_valid, n_train, n_valid, D2):
    """Define the linear model and optimize
       d: number of dimensions of the input
       r: number of dimensions in the "second to last layer"
       eps: error added to true function in data
       lr: learning rate for gradient descent
       reg: choice of lambda for L2 regularization
       n_epochs: number of epochs to train for
    """
    mses = []

    assert(r == 1)
    w2 = np.random.rand(r)
    W1 = np.random.rand(r, d)

    for epoch in range(n_epochs):
        mse = 0
        for i in range(n_valid):
            mse += (w2 @ W1 @ x_valid[i] - y_valid[i])**2
        mse /= n_valid
        # print("Epoch", epoch, "validation MSE", mse)
        
        if epoch > 20 and epoch % 10 == 0:
            lr /= 2

        grads = np.zeros(r + d)
        for i in range(n_train):
            xi = x_train[i]
            yi = y_train[i]
            tmp = 2 * (w2 @ W1 @ xi - yi)
            grads[:r] += tmp * W1 @ xi 
            grads[r:] += tmp * w2 * xi
        w2 = w2 - lr * (grads[:r]) - reg * np.abs(w2)
        W1 = W1 - lr * (grads[r:]) - reg * np.abs(W1)
        mse = 0
        for i in range(n_valid):
            mse += (w2 @ W1 @ x_valid[i] - y_valid[i])**2
        mse /= n_valid
        # print("Epoch", epoch, "validation MSE", mse)

        # At every epoch, run Linjun's method
        sets, set_counts, set_ranges, quantiles_list = split_sets(w2, W1, (x_test, y_test), C, B, r)
        #print(set_ranges)
        D2_counts, D2_weights = compute_D2(D2, r, w2, W1, quantiles_list, set_ranges)

        # Check that nothing is 0
        for i in range(len(D2_counts)):
            if D2_counts[i] == 0:
                print("D2_counts was 0 at i =", i, "with set_ranges[i]:")
                print(set_ranges[i])
                raise ValueError

        # Check the accuracy
        fhat_test = np.empty(n_test)
        for i in range(n_test):
            fhat_test[i] = fhat(x_test[i], r, w2, W1, quantiles_list, D2_weights, D2_counts, set_ranges)
        mse_nn = np.square(predict(x_test, w2, W1, only_top=False) - y_test).mean()
        mse_fhat = np.square(np.subtract(fhat_test, y_test)).mean()
        print("FHT MSE:", mse_fhat)
        mses.append(mse_fhat)

        # Print the function out
        #print(W1)
        #print(w2)
        
    return w2, W1, mses





####################################################################################################


"""Make plot for varying epsilon"""

# Parameters
C = 1000
n_train, n_valid, n_test = 32000, 1000, 1000
eps = 0.001
# B = round(n_train**(1./r + 1))
r = 1
d = 5
B = 8
lr = 0.000005 
reg = 0.01
n_epochs = 100
D1, D2_o, x_valid, y_valid_o, x_test, y_test_o, W1_star, w2_star = get_data(r, d, C, n_train, n_valid, n_test, eps, test=False, no_noise=True)
x_train = D1[0]
y_train_o = D1[1]
mse = []

for eps in [0.1, 0.01, 0.001]:
    print(eps)
    mse2 = []
    y_valid = y_valid_o + np.random.normal(scale=eps, size=(n_valid))
    y_test = y_test_o + np.random.normal(scale=eps, size=(n_test))
    D2 = D2_o[0], D2_o[1] + np.random.normal(scale=eps, size=(n_train // 2))
    y_train = y_train_o + np.random.normal(scale=eps, size=(n_train // 2))
    for i in range(5):
        print(i)
        w2, W1, mses = get_model(B, d, r, eps, lr, reg, n_epochs, x_train, y_train, x_valid, y_valid, n_train // 2, n_valid, D2)
        mse2.append(mses)
    mse.append(mse2)

print(mse)


# """Make plot for varying B"""

# # Parameters
# C = 1000
# n_train, n_valid, n_test = 32000, 1000, 1000
# eps = 0.1
# # B = round(n_train**(1./r + 1))
# r = 1
# d = 5
# B = 5
# lr = 0.000005
# reg = 0.01
# n_epochs = 100
# D1, D2, x_valid, y_valid, x_test, y_test, W1_star, w2_star = get_data(r, d, C, n_train, n_valid, n_test, eps, test=False, no_noise=False)
# x_train = D1[0]
# y_train = D1[1]
# mse = []
# for B in [4, 8, 16, 32, 64]:
#     mse2 = []
#     for i in range(5):
#         w2, W1, mses = get_model(B, d, r, eps, lr, reg, n_epochs, x_train, y_train, x_valid, y_valid, n_train // 2, n_valid, D2)
#         mse2.append(mses)
#     mse.append(mse2)

# print(mse)


 
# """Make plot for varying d"""

# # Parameters
# C = 100000
# n_train, n_valid, n_test = 32000, 1000, 1000
# eps = 0.01
# # B = round(n_train**(1./r + 1))
# r = 1
# B = 6
# lr = 0.000001
# reg = 0.01
# n_epochs = 30
# mse = []
# for d in [4]:
#     print("BEGIN d=" + str(d))
#     mse2 = []
#     for i in range(10):
#         D1, D2, x_valid, y_valid, x_test, y_test, W1_star, w2_star = get_data(r, d, C, n_train, n_valid, n_test, eps, test=False, no_noise=False)
#         print(W1_star)
#         print(w2_star)
#         x_train = D1[0]
#         y_train = D1[1]
#         print("BEGIN i=" + str(i))
#         w2, W1, mses = get_model(B, d, r, eps, lr, reg, n_epochs, x_train, y_train, x_valid, y_valid, n_train // 2, n_valid, D2)
#         mse2.append(mses)
#     mse.append(mse2)

# print(mse)
