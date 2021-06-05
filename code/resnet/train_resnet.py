import os
import sys
import time
import math
import glob
import resnet
import pickle
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=32)
parser.add_argument('--epochs', default=200)
parser.add_argument('--version', default=1)
parser.add_argument('--n', default=3)
parser.add_argument('--id', default=0)
parser.add_argument('--data', default=0)
args = vars(parser.parse_args())

batch_size = int(args['batch_size'])
epochs = int(args['epochs'])
version = int(args['version'])
n = int(args['n'])
identifier = int(args['id'])
data = int(args['data'])

num_classes = 2

if version == 1:
    depth = n * 6 + 2
elif version == 2:
    depth = n * 9 + 2

model_type = 'ResNet%dv%d_%d_data%d_flipped' % (depth, version, identifier, data)
save_dir = os.path.join(os.getcwd(), 'saved_models')
os.mkdir('saved_models/' + model_type)

old_stdout = sys.stdout
log_file = open(save_dir + "/" + model_type + "/log.txt", "w")
sys.stdout = log_file

if data == 0:
    x_train = np.load('data/x_train.npy')
    x_test = np.load('data/x_test.npy')
    y_train = np.load('data/y_train.npy')
    y_test = np.load('data/y_test.npy')
    y_train_old = np.load('data/y_train_old.npy')
    y_test_old = np.load('data/y_test_old.npy')
else:
    x_train = np.load('data_preprocessed/x_train.npy')
    x_test = np.load('data_preprocessed/x_test.npy')
    y_train = np.load('data_preprocessed/y_train.npy')
    y_test = np.load('data_preprocessed/y_test.npy')
    y_train_old = np.load('data_preprocessed/y_train_old.npy')
    y_test_old = np.load('data_preprocessed/y_test_old.npy')

y_test_old = 1 - y_test_old
y_train_old = 1 - y_train_old
input_shape = x_train.shape[1:]

if data == 0:
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

if version == 2:
    model = resnet.resnet_v2(input_shape=input_shape, depth=depth, num_classes=num_classes)
else:
    model = resnet.resnet_v1(input_shape=input_shape, depth=depth, num_classes=num_classes)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=resnet.lr_schedule(0)),
              metrics=['acc', 'mse'])
model.summary()

print("Model:", model_type)

model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

checkpoint = ModelCheckpoint(filepath=save_dir + "/" + model_type + "/" + model_name,
                             monitor='val_mse',
                             verbose=1,
                             save_best_only=False)

lr_scheduler = LearningRateScheduler(resnet.lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)


class PlottingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(logs)
        preds = self.model.predict(x_test)
        plt.clf()
        plt.hist(preds[:, 0])
        plt.title("Test Set Distribution, Epoch " + str(epoch) + ", MSE " + str(round(logs['mse'], 2)) + ", VAL MSE " + str(round(logs['val_mse'], 2)))
        plt.savefig(save_dir + "/" + model_type + "/" + "test_dist_" + str(epoch))       
        plt.clf()
plotting_callback = PlottingCallback()


class AdditionalValidationSets(Callback):
    def __init__(self, validation_sets, verbose=0, batch_size=None):
        """
        From https://stackoverflow.com/questions/47731935/using-multiple-validation-sets-with-keras
        """
        super(AdditionalValidationSets, self).__init__()
        self.validation_sets = validation_sets
        for validation_set in self.validation_sets:
            if len(validation_set) not in [3, 4]:
                raise ValueError()
        self.epoch = []
        self.history = {}
        self.verbose = verbose
        self.batch_size = batch_size

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)

        # record the same values as History() as well
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        # evaluate on the additional validation sets
        for validation_set in self.validation_sets:
            if len(validation_set) == 3:
                validation_data, validation_targets, validation_set_name = validation_set
                sample_weights = None
            elif len(validation_set) == 4:
                validation_data, validation_targets, sample_weights, validation_set_name = validation_set
            else:
                raise ValueError()

            results = self.model.evaluate(x=validation_data,
                                          y=validation_targets,
                                          verbose=self.verbose,
                                          sample_weight=sample_weights,
                                          batch_size=self.batch_size)

            for metric, result in zip(self.model.metrics_names,results):
                valuename = validation_set_name + '_' + metric
                self.history.setdefault(valuename, []).append(result)
                if self.verbose:
                    print(valuename + ": " + str(result))
        with open(save_dir + "/" + model_type  + "/history" + '.pkl', 'wb') as f:
            pickle.dump(self.history, f, pickle.HIGHEST_PROTOCOL)


validation_sets = AdditionalValidationSets([(x_test, y_test_old, 'p*')], verbose=1, batch_size=batch_size)


callbacks = [checkpoint, lr_reducer, lr_scheduler, plotting_callback, validation_sets]

print('Using real-time data augmentation.')
# this will do preprocessing and realtime data augmentation:
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=0,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False)

datagen.fit(x_train)

t1 = time.time()

steps_per_epoch =  math.ceil(len(x_train) / batch_size)
model.fit(x=datagen.flow(x_train, y_train, batch_size=batch_size),
          verbose=1,
          epochs=epochs,
          validation_data=(x_test, y_test),
          steps_per_epoch=steps_per_epoch,
          callbacks=callbacks)

t2 = time.time()
print("*******Train time" + str((t2 - t1)/60.))

with open(save_dir + "/" + model_type  + "/history" + '.pkl', 'wb') as f:
    pickle.dump(validation_sets.history, f, pickle.HIGHEST_PROTOCOL)


# score trained model
scores = model.evaluate(x_test,
                        y_test,
                        batch_size=batch_size,
                        verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


#####################################################################################################################################
#####################################################################################################################################


files = sorted(glob.glob(save_dir + "/" + model_type +  "/cifar10_%s_model.*.h5" % model_type))
#files = sorted(glob.glob("saved_models/cifar10_%s_model.*.h5" % model_type))
epochs = []
hists = []

t1 = time.time()

for file in files:
    # Save the epoch
    epoch = int(file[-5:-3])
    epochs.append(epoch)
    
    # Load the model
    if version == 1:
        model = resnet.resnet_v1(input_shape=input_shape, depth=depth)
    else:
        model = resnet.resnet_v2(input_shape=input_shape, depth=depth)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=resnet.lr_schedule(0)),
                  metrics=['acc', 'mse'])
    model.load_weights(file)
    
    print("BASELINE p*")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    res = model.evaluate(x_test, y_test_old)
    ax[0].hist(model.predict(x_test)[:, 0])
    ax[0].set_title("Epoch " + str(epoch) + " Test")
    ax[1].hist(model.predict(x_train)[:, 0])
    ax[1].set_title("Epoch " + str(epoch) + " Train")
    fig.suptitle("Baseline Model Evaluated on p*: " + str(round(res[-1], 2)) + " Test MSE")
    plt.savefig("saved_models/" + model_type + "/tune_dist_baseline")
    
    model.trainable = False
    predictions = Dense(8, activation='relu', kernel_initializer='he_normal')(model.layers[-3].output)
    predictions = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(predictions)
    new = Model(inputs=model.inputs, outputs=predictions)
    
    new.compile(optimizer=tf.keras.optimizers.SGD(momentum=0.9),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.MeanSquaredError()])
    
    def schedule_inner(epoch, lr):
        return lr * 0.96
    lr_scheduler_inner = LearningRateScheduler(schedule_inner, verbose=1)

    lr_reduce = ReduceLROnPlateau(
        monitor='val_mean_squared_error', factor=0.1, patience=1, verbose=1,
        mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

    class PlottingCallback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            plt.clf()
            preds = self.model.predict(x_test)
            plt.hist(preds[:, 0])
            plt.title("Fine Tuning Epoch: " + str(epoch + 1) + ", Test MSE: " + str(round(model.evaluate(x_test, y_test_old)[-1], 2)))
            plt.savefig("saved_models/" + model_type + "/tune_dist_" + str(epochs[-1]) + "_" + str(epoch))
            plt.clf()
        
    plotting_callback = PlottingCallback()
    history = AdditionalValidationSets([(x_test, y_test, 'y')], verbose=1, batch_size=batch_size)

    hist = new.fit(x=x_train, y=y_train_old, epochs=8, batch_size=batch_size, 
                  validation_data=(x_test, y_test_old), 
                  callbacks=[lr_scheduler_inner, history, plotting_callback])
    hists.append(history.history)


    with open(save_dir + "/" + model_type  + "/tuned_history" + '.pkl', 'wb') as f:
        pickle.dump(hists, f, pickle.HIGHEST_PROTOCOL)

t2 = time.time()

print("*******Train time" + str((t2 - t1)/60.))

sys.stdout = old_stdout
log_file.close()



